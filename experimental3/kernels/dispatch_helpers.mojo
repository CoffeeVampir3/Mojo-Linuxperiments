"""Reusable dispatch-shape helpers used by every prefill dispatcher.

Two abstractions, each captures one repeated pattern:

  - `tile_and_dispatch`: M-split job-build + pool dispatch. Replaces the
    12-line bookkeeping block that lives verbatim in every prefill dispatcher.
    Caller provides a `kernel` (comptime worker reference) and a `factory`
    closure that builds per-tile Args from (start, count).

  - `SeqDispatched`: trait + default `run` for the seq_len-driven choice
    between an inline decode path and a parallel prefill path. A phase
    becomes a struct that conforms; conformers must provide both paths and
    inherit the routing for free.

These compose: a `SeqDispatched` conformer typically uses `tile_and_dispatch`
inside its `prefill_dispatch` method.

Both helpers parameterize on the real `BurstThreadPool` trait and return
`PoolFence[P, origin]`, so they drop into existing dispatchers without API
change at the caller.
"""

from std.memory import UnsafePointer
from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY


def tile_and_dispatch[
    Args: Copyable & ImplicitlyCopyable & Defaultable,
    P: BurstThreadPool, origin: MutOrigin, //,
    kernel: def (Args) thin -> None,
    factory: def (Int, Int) capturing [_] -> Args,
](
    work_count: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """M-split tile build + dispatch.

    Tiles `work_count` rows across the pool's workers, builds one Args per
    tile via `factory(start, count)`, and dispatches to `kernel`. Returns a
    PoolFence the caller joins on.

    Returns a no-op fence when `work_count <= 0` so callers don't have to
    early-return.
    """
    var num_jobs = min(work_count, pool.get_capacity())
    if num_jobs <= 0:
        return PoolFence[P, origin].over(pool)

    var rows_per_job = (work_count + num_jobs - 1) // num_jobs

    var jobs = InlineArray[Args, MAX_POOL_CAPACITY](fill=Args())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= work_count:
            break
        var count = min(rows_per_job, work_count - start)
        jobs[actual] = factory(start, count)
        actual += 1

    pool.dispatch[Args, kernel](UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


trait SeqPathPrimitives:
    """A phase that has two implementations of its work, one for decode shape
    and one for prefill shape. Conformers must define both plus the predicate
    that selects between them.
    """
    comptime Args: Copyable & ImplicitlyCopyable & Defaultable

    @staticmethod
    def is_decode_shape(args: Self.Args) -> Bool: ...

    @staticmethod
    def decode_inline[
        P: BurstThreadPool, origin: MutOrigin, //,
    ](
        args: Self.Args,
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]: ...

    @staticmethod
    def prefill_dispatch[
        P: BurstThreadPool, origin: MutOrigin, //,
    ](
        args: Self.Args,
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]: ...


trait SeqDispatched(SeqPathPrimitives):
    """Routes between decode_inline and prefill_dispatch based on args shape.

    Conformers need only fill in the SeqPathPrimitives methods; `run` is the
    default routing impl and shouldn't be overridden.
    """
    @staticmethod
    def run[
        P: BurstThreadPool, origin: MutOrigin, //,
    ](
        args: Self.Args,
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        if Self.is_decode_shape(args):
            return Self.decode_inline(args, pool)
        return Self.prefill_dispatch(args, pool)
