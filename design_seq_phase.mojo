"""Design experiment: a trait-based abstraction that names the pattern
"this phase has a decode-inline path and a prefill-pool path, picked by
seq_len."

Background — what's repeated today:

    if seq_len == 1:
        # ... decode-shaped work, often inline on main thread or N-split ...
    else:
        # ... prefill-shaped work, M-split across pool ...

The branch itself is one line. What we lose by not having an abstraction is
the *name* — there's no compile-time concept that says "X is a seq-dispatched
phase." When adding a new phase like q_prep or norm_prep, there's no enforced
shape; each author re-derives the pattern.

Design — a subtrait template method (the `Syscalls(ArchPrimitives)` shape
from the Mojo docs):

    trait SeqPathPrimitives:
        # abstract: implementor provides both paths
        @staticmethod
        def decode_inline(args: Self.Args): ...
        @staticmethod
        def prefill_dispatch[P, origin](args, pool) -> Fence: ...

    trait SeqDispatched(SeqPathPrimitives):
        # default: pick based on pos_count
        @staticmethod
        def run[P, origin](args, pool) -> Fence:
            if Self.is_decode_shape(args):
                Self.decode_inline(args)
                return Fence.over(pool)
            return Self.prefill_dispatch(args, pool)

A phase is then a struct that conforms — it provides the two paths and
inherits the `run` dispatcher. Compile-time enforcement that both paths exist
is the only real value over the bare `if/else`.

This file probes whether that abstraction earns its keep on a concrete case
(q_prep, the next phase to parallelize).
"""

from std.memory import UnsafePointer
from std.collections import InlineArray


# ============================================================================
# Stand-in pool + fence (same as design_tile_dispatch.mojo).
# ============================================================================


comptime MAX_POOL_CAPACITY = 32


@fieldwise_init
struct FakePool(Copyable):
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
    @staticmethod
    def over(ref [Self.origin] pool: FakePool) -> Self:
        return Self()


# ============================================================================
# The abstraction.
# ============================================================================


trait SeqPathPrimitives:
    """A phase that has two implementations. Implementors must provide both."""
    comptime Args: Copyable & ImplicitlyCopyable & Defaultable

    @staticmethod
    def is_decode_shape(args: Self.Args) -> Bool: ...

    @staticmethod
    def decode_inline(args: Self.Args): ...

    @staticmethod
    def prefill_dispatch[origin: MutOrigin](
        args: Self.Args,
        ref [origin] pool: FakePool,
    ) -> FakeFence[origin]: ...


trait SeqDispatched(SeqPathPrimitives):
    """Routes between decode and prefill based on the args' shape.

    Conforming structs only need to fill in the SeqPathPrimitives methods;
    `run` is provided as a default and shouldn't be overridden.
    """
    @staticmethod
    def run[origin: MutOrigin](
        args: Self.Args,
        ref [origin] pool: FakePool,
    ) -> FakeFence[origin]:
        if Self.is_decode_shape(args):
            Self.decode_inline(args)
            return FakeFence[origin].over(pool)
        return Self.prefill_dispatch(args, pool)


# ============================================================================
# Worked example: q_prep, the next phase we want to parallelize.
#
# Args carries everything both paths need; the only thing that varies is
# pos_count, which drives the branch.
# ============================================================================


comptime NUM_KV_HEADS = 2
comptime HEAD_DIM = 128
comptime HPG = 6


@fieldwise_init
struct QPrepArgs(Copyable, ImplicitlyCopyable, Defaultable):
    """Single args type covering both decode and prefill q_prep paths.
    pos_count == 1 selects the decode path; > 1 selects prefill."""
    var qkv_ptr: Int
    var q_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var inv_rms_q_arr: Int
    var qi_out: Int
    var qi_biases_out: Int
    var q_scales_out: Int
    var start_pos: Int
    var pos_count: Int

    def __init__(out self):
        self.qkv_ptr = 0
        self.q_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.inv_rms_q_arr = 0
        self.qi_out = 0
        self.qi_biases_out = 0
        self.q_scales_out = 0
        self.start_pos = 0
        self.pos_count = 0


# The fake "kernel" that processes one (kv_head, position-range) tuple.
@fieldwise_init
struct QPrepKvJob(Copyable, ImplicitlyCopyable, Defaultable):
    var kv: Int
    var p_start: Int
    var p_count: Int

    def __init__(out self):
        self.kv = 0
        self.p_start = 0
        self.p_count = 0


def q_prep_kernel(job: QPrepKvJob):
    # In production: load Q, apply gamma + RoPE + FWHT + quantize for
    # `job.p_count` positions starting at `job.p_start` of head `job.kv`.
    pass


struct QPrepPhase(SeqDispatched):
    comptime Args = QPrepArgs

    @staticmethod
    def is_decode_shape(args: QPrepArgs) -> Bool:
        return args.pos_count == 1

    @staticmethod
    def decode_inline(args: QPrepArgs):
        # Decode: one position, all KV heads, on main thread.
        # Dispatch overhead would dominate, so we don't touch the pool.
        for kv in range(NUM_KV_HEADS):
            q_prep_kernel(QPrepKvJob(kv, args.start_pos, 1))

    @staticmethod
    def prefill_dispatch[origin: MutOrigin](
        args: QPrepArgs,
        ref [origin] pool: FakePool,
    ) -> FakeFence[origin]:
        # Prefill: position-split per kv head across pool workers.
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
                jobs[actual] = QPrepKvJob(kv, args.start_pos + p_start, p_count)
                actual += 1

        pool.dispatch[QPrepKvJob, q_prep_kernel](
            UnsafePointer(to=jobs[0]), actual)
        return FakeFence[origin].over(pool)


# ============================================================================
# Caller code: before vs. after.
# ============================================================================


# --- Before: caller writes the if/else explicitly --- #
def caller_old(args: QPrepArgs, mut pool: FakePool):
    if args.pos_count == 1:
        # decode: inline
        for kv in range(NUM_KV_HEADS):
            q_prep_kernel(QPrepKvJob(kv, args.start_pos, 1))
    else:
        # prefill: position-split, dispatch
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
                jobs[actual] = QPrepKvJob(kv, args.start_pos + p_start, p_count)
                actual += 1
        pool.dispatch[QPrepKvJob, q_prep_kernel](
            UnsafePointer(to=jobs[0]), actual)


# --- After: caller asks the phase to run; phase picks. --- #
def caller_new(args: QPrepArgs, mut pool: FakePool):
    _ = QPrepPhase.run(args, pool)


# ============================================================================
# Honest assessment
# ============================================================================
#
# What the trait buys you:
#   1. Compile-time guarantee that conformers provide BOTH paths. Forget
#      decode_inline and you get an unimplemented-trait-method error, not a
#      runtime regression where decode is silently dispatched.
#   2. A standard place to put each phase's logic. New phases (norm_prep,
#      moe_route_merge, moe_route_schedule) become structs that conform —
#      consistent shape across the project.
#   3. A name for the pattern. Reading `QPrepPhase(SeqDispatched)` tells you
#      everything about how it's structured.
#
# What it costs:
#   1. Every phase becomes a struct with three static methods instead of one
#      free function with an if/else. For phases where the two paths share
#      almost no code, the struct boundary feels artificial.
#   2. The Args type has to cover both paths. For q_prep that's natural —
#      the args genuinely are the same. For phases where decode and prefill
#      take different shapes (e.g., a kernel that takes a single value vs.
#      a batched buffer), you'd need a wider Args struct or two paths.
#   3. The compile-time check that both methods exist is not a huge win in
#      practice — the existing dispatchers already have both paths because
#      callers were going to break otherwise.
#
# Verdict: marginal. The abstraction is correct and the name is useful, but
# it doesn't dramatically reduce code. The bigger win in the current codebase
# is `tile_and_dispatch` (Pattern 1), which removes mechanical boilerplate.
# I'd land that first; consider this only if we add many more seq-dispatched
# phases (e.g., if a future model has 5+ phases needing this branch).
#
# A pragmatic middle ground: skip the trait, keep the pattern in mind, and
# write each phase as a free function with the seq-len branch internalized.
# The branch is one line; the trait is overhead.


def main():
    var pool = FakePool(capacity=8)

    # Decode shape
    var dec_args = QPrepArgs(
        qkv_ptr=0, q_norm_ptr=0, cos_ptr=0, sin_ptr=0,
        inv_rms_q_arr=0, qi_out=0, qi_biases_out=0, q_scales_out=0,
        start_pos=42, pos_count=1)

    # Prefill shape
    var pre_args = QPrepArgs(
        qkv_ptr=0, q_norm_ptr=0, cos_ptr=0, sin_ptr=0,
        inv_rms_q_arr=0, qi_out=0, qi_biases_out=0, q_scales_out=0,
        start_pos=0, pos_count=1024)

    _ = QPrepPhase.run(dec_args, pool)
    _ = QPrepPhase.run(pre_args, pool)
    print("design_seq_phase: OK")
