"""Side-by-side: q_prep written three ways.

Same problem (q_prep, the next phase to parallelize), three abstraction
levels. Read in order; the question is which one earns its keep.

Decode shape: pos_count == 1, all KV heads on main thread, no pool dispatch
(the work is so small that dispatch round-trip would dominate).

Prefill shape: pos_count > 1, position-split across pool workers per kv head.

The three versions all produce identical work. Only the dispatch *shape*
varies. None of them are wired into the model — this is a design probe.
"""

from std.memory import UnsafePointer
from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from experimental3.kernels.dispatch_helpers import (
    tile_and_dispatch, SeqDispatched, SeqPathPrimitives,
)


# ============================================================================
# Stub types — keep this file standalone. Production q_prep takes the real
# QPrepBatchArgs from minimax/kernels/dispatch_args.mojo; the *shape* of the
# dispatch is what we're comparing.
# ============================================================================


comptime NUM_KV_HEADS = 2  # tp=4 → KV_PER_RANK = 2
comptime HEAD_DIM = 128
comptime HPG = 6


@fieldwise_init
struct QPrepKvJob(Copyable, ImplicitlyCopyable, Defaultable):
    """One (kv head, position-range) tile of q_prep work."""
    var kv: Int
    var p_start: Int
    var p_count: Int

    def __init__(out self):
        self.kv = 0
        self.p_start = 0
        self.p_count = 0


def q_prep_kv_kernel(job: QPrepKvJob):
    """Stub: in production, loops `job.p_count` positions of head `job.kv`
    applying gamma + RoPE + FWHT + i8 quantize."""
    pass


@fieldwise_init
struct QPrepArgs(Copyable, ImplicitlyCopyable, Defaultable):
    """Dispatcher-level args. Single source of truth for both paths."""
    var start_pos: Int
    var pos_count: Int
    # production also carries qkv/q_norm/cos/sin/inv_rms/qi_out/biases/scales
    # pointers; omitted here for brevity.

    def __init__(out self):
        self.start_pos = 0
        self.pos_count = 0


# ============================================================================
# Style 1: raw — write the if/else and the M-split loop by hand.
# ============================================================================


def q_prep_dispatch_raw[
    P: BurstThreadPool, origin: MutOrigin, //,
](
    args: QPrepArgs,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    if args.pos_count == 1:
        # Decode: inline, all KV heads, single position.
        for kv in range(NUM_KV_HEADS):
            q_prep_kv_kernel(QPrepKvJob(kv, args.start_pos, 1))
        return PoolFence[P, origin].over(pool)

    # Prefill: position-split per kv head.
    var cap = pool.get_capacity()
    var jobs_per_kv = max(1, cap // NUM_KV_HEADS)
    var positions_per_job = (args.pos_count + jobs_per_kv - 1) // jobs_per_kv

    var jobs = InlineArray[QPrepKvJob, MAX_POOL_CAPACITY](fill=QPrepKvJob())
    var actual = 0
    for kv in range(NUM_KV_HEADS):
        for j in range(jobs_per_kv):
            var p_start = j * positions_per_job
            if p_start >= args.pos_count:
                break
            var p_count = min(positions_per_job, args.pos_count - p_start)
            jobs[actual] = QPrepKvJob(
                kv, args.start_pos + p_start, p_count)
            actual += 1
    pool.dispatch[QPrepKvJob, q_prep_kv_kernel](
        UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Style 2: tile_and_dispatch only — keep the if/else, hide the M-split.
#
# Note: q_prep tiles in TWO axes (kv, positions), so tile_and_dispatch
# doesn't apply directly — the flat row index has to encode both. The
# helper handles tiling on a single axis; we serialize (kv, pos-batch)
# pairs onto that axis.
# ============================================================================


def q_prep_dispatch_tile[
    P: BurstThreadPool, origin: MutOrigin, //,
](
    args: QPrepArgs,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    if args.pos_count == 1:
        for kv in range(NUM_KV_HEADS):
            q_prep_kv_kernel(QPrepKvJob(kv, args.start_pos, 1))
        return PoolFence[P, origin].over(pool)

    # Encode (kv, position-batch) onto a single work axis. Per kv: the pool
    # gets ⌈cap / NUM_KV_HEADS⌉ batches. Total work = NUM_KV_HEADS * batches.
    var cap = pool.get_capacity()
    var jobs_per_kv = max(1, cap // NUM_KV_HEADS)
    var positions_per_job = (args.pos_count + jobs_per_kv - 1) // jobs_per_kv
    var total_work = NUM_KV_HEADS * jobs_per_kv

    var start_pos = args.start_pos
    var pos_count = args.pos_count

    @parameter
    def factory(work_id: Int, _count: Int) -> QPrepKvJob:
        var kv = work_id // jobs_per_kv
        var batch = work_id - kv * jobs_per_kv
        var p_start = batch * positions_per_job
        var p_count = min(positions_per_job, pos_count - p_start)
        return QPrepKvJob(kv, start_pos + p_start, p_count)

    return tile_and_dispatch[
        kernel=q_prep_kv_kernel, factory=factory,
    ](total_work, pool)


# ============================================================================
# Style 3: SeqDispatched trait — phase struct, decode/prefill methods.
# ============================================================================


struct QPrepPhase(SeqDispatched):
    comptime Args = QPrepArgs

    @staticmethod
    def is_decode_shape(args: QPrepArgs) -> Bool:
        return args.pos_count == 1

    @staticmethod
    def decode_inline[
        P: BurstThreadPool, origin: MutOrigin, //,
    ](
        args: QPrepArgs,
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        for kv in range(NUM_KV_HEADS):
            q_prep_kv_kernel(QPrepKvJob(kv, args.start_pos, 1))
        return PoolFence[P, origin].over(pool)

    @staticmethod
    def prefill_dispatch[
        P: BurstThreadPool, origin: MutOrigin, //,
    ](
        args: QPrepArgs,
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        var cap = pool.get_capacity()
        var jobs_per_kv = max(1, cap // NUM_KV_HEADS)
        var positions_per_job = (args.pos_count + jobs_per_kv - 1) // jobs_per_kv
        var total_work = NUM_KV_HEADS * jobs_per_kv

        var start_pos = args.start_pos
        var pos_count = args.pos_count

        @parameter
        def factory(work_id: Int, _count: Int) -> QPrepKvJob:
            var kv = work_id // jobs_per_kv
            var batch = work_id - kv * jobs_per_kv
            var p_start = batch * positions_per_job
            var p_count = min(positions_per_job, pos_count - p_start)
            return QPrepKvJob(kv, start_pos + p_start, p_count)

        return tile_and_dispatch[
            kernel=q_prep_kv_kernel, factory=factory,
        ](total_work, pool)


# ============================================================================
# Reading guide
#
# Style 1 (raw):
#   - 27 lines including the prefill branch.
#   - Everything visible at the call site. Easy to read, easy to debug.
#   - The prefill bookkeeping (jobs array, actual counter, dispatch + fence)
#     is mechanical and identical to every other prefill dispatcher.
#
# Style 2 (tile_and_dispatch):
#   - 23 lines. Saves the array, the actual counter, the dispatch call,
#     and the fence return. What remains is the per-tile factory.
#   - Need to flatten (kv, position-batch) onto a single work axis since
#     tile_and_dispatch tiles on one axis. This is a small awkwardness but
#     not unreasonable — the same pattern shows up in chunked attention.
#   - The factory captures cleanly via @parameter; comptime-bound.
#
# Style 3 (SeqDispatched):
#   - 36 lines (counting the struct boundary). LONGER than raw because
#     each path is a static method with its own signature.
#   - Gains: a name for the pattern, compile-time enforcement that both
#     paths exist, a consistent shape across all phases that adopt it.
#   - Costs: every phase becomes a struct. The two paths can't share locals
#     (they're separate static methods). The factory closure inside
#     prefill_dispatch is identical to Style 2.
#
# My recommendation:
#   - Style 2 for new prefill dispatchers like q_prep, norm_prep, etc.
#     The if/else stays at the top, the M-split bookkeeping is gone.
#   - Don't adopt Style 3 unless you have 5+ seq-dispatched phases that
#     would benefit from the trait's compile-time guarantee. Today we have
#     ~3 candidates (q_prep, norm_prep, route_merge), which doesn't justify
#     the per-phase struct overhead.
#
# The brutally honest take: Style 2's win over Style 1 is "boilerplate
# reduction with no behavioral change." Style 3's win over Style 2 is
# "named pattern + compile-time enforcement, with no boilerplate reduction."
# The first is unambiguous. The second is taste.
# ============================================================================


def main():
    print("design_q_prep_three_ways: see source for the three styles")
