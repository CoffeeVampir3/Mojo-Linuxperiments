"""Design experiment: a comptime-engineered tile_and_dispatch helper that
collapses the M-split prefill dispatcher boilerplate into a single call.

Background — what's repeated today (verbatim, 5+ times):

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
    var jobs = InlineArray[Args, MAX_POOL_CAPACITY](fill=Args())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= seq_len: break
        var end = min(start + rows_per_job, seq_len)
        jobs[actual] = Args(... per-tile fields ...)
        actual += 1
    pool.dispatch[Args, worker](UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)

Only the per-tile Args constructor varies between call sites. Everything else
is mechanical.

Design — extract the mechanical part as `tile_and_dispatch`:

    return tile_and_dispatch[
        Args=WorkerConfig,
        kernel=int8_gemm_amx_worker[N, K],
        factory=fn,
    ](seq_len, pool)

The factory is a parametric closure that takes (start, count) and returns the
Args struct for that tile. Everything else — the InlineArray, the loop, the
fence return — is internal to the helper.

This file is a standalone, runnable design probe. It defines a fake mini-pool
and a fake worker so we can compile-check the abstraction without pulling in
the real BurstPool. The shape of the helper is what matters; the actual
production version would parameterize on the real pool trait.
"""

from std.memory import UnsafePointer
from std.collections import InlineArray


# ============================================================================
# Stand-in types for a real BurstPool so this file compiles standalone.
# ============================================================================


comptime MAX_POOL_CAPACITY = 32


@fieldwise_init
struct FakePool(Copyable):
    """Stand-in for IsolatedBurstPool / BurstPool. The real helper would
    take any P: BurstThreadPool — the shape is the same."""
    var capacity: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[
        Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) thin -> None,
        origin: MutOrigin,
    ](
        mut self,
        args_ptr: UnsafePointer[Args, origin],
        num_jobs: Int,
    ):
        for i in range(num_jobs):
            kernel((args_ptr + i)[])


@fieldwise_init
struct FakeFence[origin: MutOrigin](Copyable):
    """Stand-in for PoolFence."""
    @staticmethod
    def over(ref [Self.origin] pool: FakePool) -> Self:
        return Self()


# ============================================================================
# The helper.
# ============================================================================


def tile_and_dispatch[
    Args: Copyable & ImplicitlyCopyable & Defaultable,
    origin: MutOrigin, //,
    kernel: def (Args) thin -> None,
    factory: def (Int, Int) capturing [_] -> Args,
](
    work_count: Int,
    ref [origin] pool: FakePool,
) -> FakeFence[origin]:
    """Tile work_count rows across the pool's workers and dispatch one job
    per tile. The factory builds the Args for tile [start, start + count).

    Mirrors the body of int8_gemv's prefill branch verbatim — the only
    parts that vary across dispatchers are `Args`, `kernel`, and `factory`.
    """
    var num_jobs = min(work_count, pool.get_capacity())
    if num_jobs <= 0:
        return FakeFence[origin].over(pool)
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
    return FakeFence[origin].over(pool)


# ============================================================================
# Worked example: a fake int8_gemv prefill dispatcher.
#
# Compare the two functions below — `prefill_old` is the current shape;
# `prefill_new` uses tile_and_dispatch.
# ============================================================================


@fieldwise_init
struct WorkerConfig(Copyable, ImplicitlyCopyable, Defaultable):
    var act_off: Int
    var weight: Int
    var dst_off: Int
    var start: Int
    var count: Int

    def __init__(out self):
        self.act_off = 0
        self.weight = 0
        self.dst_off = 0
        self.start = 0
        self.count = 0


def fake_worker(cfg: WorkerConfig):
    pass


# --- Old shape: 12 lines of mechanical boilerplate per dispatcher --- #
def prefill_old[
    origin: MutOrigin, //, N: Int, K: Int,
](
    act_p: Int, wpacked_p: Int, dst_p: Int,
    seq_len: Int,
    ref [origin] pool: FakePool,
) -> FakeFence[origin]:
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
    var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](fill=WorkerConfig())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= seq_len:
            break
        var end = min(start + rows_per_job, seq_len)
        jobs[actual] = WorkerConfig(
            act_p + start * K, wpacked_p, dst_p + start * N,
            start, end - start)
        actual += 1
    pool.dispatch[WorkerConfig, fake_worker](
        UnsafePointer(to=jobs[0]), actual)
    return FakeFence[origin].over(pool)


# --- New shape: factory closure + one helper call --- #
def prefill_new[
    origin: MutOrigin, //, N: Int, K: Int,
](
    act_p: Int, wpacked_p: Int, dst_p: Int,
    seq_len: Int,
    ref [origin] pool: FakePool,
) -> FakeFence[origin]:
    @parameter
    def factory(start: Int, count: Int) -> WorkerConfig:
        return WorkerConfig(
            act_p + start * K, wpacked_p, dst_p + start * N,
            start, count)

    return tile_and_dispatch[
        kernel=fake_worker, factory=factory,
    ](seq_len, pool)


# ============================================================================
# Honest assessment
# ============================================================================
#
# Lines saved per dispatcher: ~9 → ~5. Not enormous, but the saved lines are
# mechanical (the loop, the InlineArray, the actual counter, the fence return)
# while the kept lines are the *meaningful* part (what goes into each tile's
# Args). The before/after isn't shorter by an order of magnitude, but it
# *reads* better — the per-tile field assignments are no longer buried under
# bookkeeping.
#
# Risks:
#   - The factory closure must capture cleanly. With Mojo's parametric closure
#     `def (Int, Int) capturing [_] -> Args`, this works and the captures are
#     comptime-bound. Test in a real dispatcher before claiming it composes.
#   - The InlineArray sizing on MAX_POOL_CAPACITY is hard-coded. If a future
#     pool ever exceeds it, the abstraction needs a comptime-parameterized
#     ceiling. Same constraint exists today; this doesn't make it worse.
#   - Error attribution: with the helper, a bug in the tiling logic shows up
#     in *every* dispatcher at once. That's a feature for fixing things, but a
#     liability if the helper has a subtle issue. Lean on the existing tests.
#
# Verdict: worth doing. But the M-split shape is well-understood, so this is
# an "easy refactor when convenient", not a critical change. The right time
# to land it is during the q_prep / norm_prep parallelization work — those
# new dispatchers can be the first consumers of the helper, and we backport
# the existing 5 dispatchers in a follow-up.


def main():
    var pool = FakePool(capacity=8)

    var fence_old = prefill_old[N=64, K=128](
        0, 0, 0, 256, pool)
    var fence_new = prefill_new[N=64, K=128](
        0, 0, 0, 256, pool)

    print("design_tile_dispatch: OK")
    _ = fence_old
    _ = fence_new
