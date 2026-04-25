"""Reusable dispatch-shape helpers shared by prefill dispatchers.

`tile_and_dispatch` replaces the 12-line M-split bookkeeping block that
otherwise lives verbatim in every prefill dispatcher. Caller provides a
`kernel` (comptime worker reference) and a `factory` closure that builds
per-tile Args from (start, count); the helper handles the InlineArray, the
loop, the actual counter, the pool dispatch, and the fence return.

Parameterizes on the real `BurstThreadPool` trait and returns
`PoolFence[P, origin]`, so it drops into existing dispatchers without
changing the caller's API.
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
