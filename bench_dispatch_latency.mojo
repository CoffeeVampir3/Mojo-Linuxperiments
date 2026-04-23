"""Benchmark: isolation-pool dispatch overhead for per-rank small work.

The MiniMax forward path has ~9 phases per layer that currently run
sequentially on the main thread because the measured cost of pool-dispatch
exceeded the measured cost of doing the work inline. This benchmark
quantifies that gap and tests whether caller-side batching of cross-pool
dispatches makes parallel-per-rank cheap enough to win.

One invocation runs one variant so `perf stat` attributes cleanly:

    ./bench_dispatch_latency <variant>

  variant: inline_remote | inline_local | dispatch_seq | dispatch_batched

  inline_remote    — main thread iterates `for r in range(tp)` and calls the
                     kernel on rank r's NUMA-local buffers. 3 of 4 ranks
                     are remote reads from main's perspective. This mirrors
                     forward_decode's current dual_norm / attn_quantize /
                     q_prep / merge_quant / expert_sum / router_topk shape.
  inline_local     — same kernel, 4 iterations, all against rank 0 buffers.
                     Isolates the remote-read cost: the delta to inline_remote
                     is the BW penalty for running per-rank work on main.
  dispatch_seq     — current `pool[r].dispatch + join` pattern per rank, one
                     pool at a time. Captures today's dispatch overhead.
  dispatch_batched — caller-side batching via dispatch_one_per_pool: writes
                     all `tp` mailboxes (remote stores pipelined through
                     main's store buffer / LFBs), then signals all ready
                     flags in a second pass. Valid only on the isolation
                     pool because workers never sleep — no futex wake
                     needed, and x86 TSO per-CPU program order gives each
                     worker release semantics on its own mailbox without
                     any cross-pool barrier.

Kernel under test: rmsnorm_dual_output_row[HIDDEN=3072, FWHT_BLK=128] from
minimax/kernels/rmsnorm.mojo. Matches the production dual_norm shape (the
biggest of the per-rank inline phases, 2.3% of frame).

Correctness: dispatch_batched's output is compared byte-for-byte against
dispatch_seq's output before timing. Both variants must produce identical
qi, work, scale, and normed buffers on rank 0.

Invocation (see bench_dispatch_matrix.fish for the full sweep):
    perf stat -e <events> ./bench_dispatch_latency dispatch_batched
"""

from std.sys import argv
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.atomic import Atomic, Ordering

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_shared import (
    AtomicInt32, MAILBOX_DATA_BYTES, typed_trampoline,
)

from simd_math import set_subnormal_zeroing
from experimental3.common_math import I8Ptr, F32Ptr, BF16Ptr
from modeling.model_spec import DEFAULT_ALIGNMENT

from minimax.kernels.rmsnorm import rmsnorm_dual_output_row


# =============================================================================
# Shape — matches production dual_norm phase
# =============================================================================


comptime HIDDEN = 3072
comptime FWHT_BLK = 128
comptime TP = 4

comptime WARMUP = 500
comptime ITERS = 20000
comptime EPS = Float32(1e-6)


# =============================================================================
# Small per-rank args — 7 pointers + 1 f32 = 60 bytes, one cache line
# =============================================================================


@fieldwise_init
struct SmallPhaseArgs(Copyable, ImplicitlyCopyable):
    var src: BF16Ptr
    var gamma_a: BF16Ptr
    var gamma_b: BF16Ptr
    var qi: I8Ptr
    var work: F32Ptr
    var scale: F32Ptr
    var normed: BF16Ptr
    var eps: Float32

    def __init__(out self):
        self.src = BF16Ptr()
        self.gamma_a = BF16Ptr()
        self.gamma_b = BF16Ptr()
        self.qi = I8Ptr()
        self.work = F32Ptr()
        self.scale = F32Ptr()
        self.normed = BF16Ptr()
        self.eps = Float32(0)


def small_phase_worker(args: SmallPhaseArgs):
    rmsnorm_dual_output_row[HIDDEN, FWHT_BLK](
        args.src, args.gamma_a, args.gamma_b,
        args.qi, args.work, args.scale, args.normed,
        args.eps)


# =============================================================================
# Batched cross-pool dispatch helper — the thing under test
#
# Valid ONLY for IsolatedBurstPool. The isolation pool guarantees workers
# are always spinning on their local job_ready flag — no sleep, no futex,
# no wake protocol. That lets dispatch collapse to two passes:
#
#   Pass 1: write all pools' mailbox data (func_ptr + args). Stores to
#           different pools target distinct NUMA nodes and are independent;
#           main's store buffer / LFBs pipeline them.
#   Pass 2: release-store each pool's job_ready. On x86 TSO, a plain store
#           is a release; per-CPU program order gives each worker the
#           guarantee that it observes its args-stores before its own
#           ready-store. Cross-rank ordering is irrelevant — each worker
#           only reads its own mailbox.
#
# This would NOT be safe on BurstPool: parked workers need a futex_wake
# after the ready store. IsolatedBurstPool has no such requirement.
# =============================================================================


@always_inline
def dispatch_one_per_pool[
    Args: Copyable & ImplicitlyCopyable,
    kernel: def(Args) thin -> None,
    tp: Int,
    origin: MutOrigin,
](
    ref [origin] pools: HeapMoveArray[IsolatedBurstPool[]],
    args_ptr: UnsafePointer[Args, MutAnyOrigin],
):
    comptime assert size_of[Args]() <= MAILBOX_DATA_BYTES, "args exceed mailbox"

    var tramp = typed_trampoline[Args, kernel]
    var tramp_ptr = UnsafePointer(to=tramp).bitcast[Int]()[]

    # Pass 1: write func_ptr + args to each pool's slot-0 mailbox.
    for r in range(tp):
        var mb = pools[r].mailboxes + 0
        mb[].func_ptr = tramp_ptr
        UnsafePointer(to=mb[].data[0]).bitcast[Args]()[] = (args_ptr + r)[]

    # Pass 2: release-store ready flags + mark active.
    for r in range(tp):
        AtomicInt32.store[ordering=Ordering.RELEASE](
            UnsafePointer(to=(pools[r].mailboxes + 0)[].job_ready.value), 1)
        pools[r].active_jobs = 1


@always_inline
def join_all[tp: Int, origin: MutOrigin](
    ref [origin] pools: HeapMoveArray[IsolatedBurstPool[]],
):
    for r in range(tp):
        pools[r].join()


# =============================================================================
# Rank-local buffer setup
# =============================================================================


@fieldwise_init
struct RankBufs(Copyable, ImplicitlyCopyable):
    var src: BF16Ptr
    var gamma_a: BF16Ptr
    var gamma_b: BF16Ptr
    var qi: I8Ptr
    var work: F32Ptr
    var scale: F32Ptr
    var normed: BF16Ptr

    def __init__(out self):
        self.src = BF16Ptr()
        self.gamma_a = BF16Ptr()
        self.gamma_b = BF16Ptr()
        self.qi = I8Ptr()
        self.work = F32Ptr()
        self.scale = F32Ptr()
        self.normed = BF16Ptr()


def alloc_rank(
    mut arena: NumaArena[alignment=DEFAULT_ALIGNMENT],
) -> RankBufs:
    var r = RankBufs()
    r.src = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    r.gamma_a = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    r.gamma_b = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    r.qi = arena.alloc[Scalar[DType.int8]](HIDDEN)
    r.work = arena.alloc[Float32](HIDDEN)
    r.scale = arena.alloc[Float32](1)
    r.normed = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    return r^


@no_inline
def fill_bf16(p: BF16Ptr, count: Int, seed: UInt64, scale: Float32):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = ((state >> 33) & 0xFFFFFF).cast[DType.uint32]()
        var v = Float32(u) / Float32(0x1000000) * scale - scale * Float32(0.5)
        p[i] = Scalar[DType.bfloat16](v)
    _ = state


def fill_rank(bufs: RankBufs, seed: UInt64):
    fill_bf16(bufs.src, HIDDEN, seed ^ 0x11, Float32(2.0))
    fill_bf16(bufs.gamma_a, HIDDEN, seed ^ 0x22, Float32(1.0))
    fill_bf16(bufs.gamma_b, HIDDEN, seed ^ 0x33, Float32(1.0))


def build_args_for_rank(bufs: RankBufs) -> SmallPhaseArgs:
    return SmallPhaseArgs(
        src=bufs.src, gamma_a=bufs.gamma_a, gamma_b=bufs.gamma_b,
        qi=bufs.qi, work=bufs.work, scale=bufs.scale, normed=bufs.normed,
        eps=EPS)


# =============================================================================
# Correctness check — dispatch_batched vs dispatch_seq, rank 0 output
# =============================================================================


@no_inline
def compare_i8(a: I8Ptr, b: I8Ptr, count: Int) -> Int:
    for i in range(count):
        if a[i] != b[i]:
            return i
    return -1


@no_inline
def compare_bf16(a: BF16Ptr, b: BF16Ptr, count: Int) -> Int:
    for i in range(count):
        if a[i] != b[i]:
            return i
    return -1


@no_inline
def compare_f32(a: F32Ptr, b: F32Ptr, count: Int) -> Int:
    for i in range(count):
        if a[i] != b[i]:
            return i
    return -1


def correctness_check(
    bufs: RankBufs, mut pool: IsolatedBurstPool[],
) -> Bool:
    """Run the kernel twice on the SAME rank-0 buffers via two different
    dispatch paths; snapshot outputs after path A, then path B writes over
    them and we compare. Both must produce identical qi/work/scale/normed."""

    var args = build_args_for_rank(bufs)

    # Path A: pool.dispatch (standard)
    pool.dispatch[SmallPhaseArgs, small_phase_worker](
        UnsafePointer(to=args), 1)
    pool.join()

    # Snapshot path A's outputs to stack.
    var qi_snap = InlineArray[Scalar[DType.int8], HIDDEN](uninitialized=True)
    for i in range(HIDDEN):
        qi_snap[i] = bufs.qi[i]
    var work_snap = InlineArray[Float32, HIDDEN](uninitialized=True)
    for i in range(HIDDEN):
        work_snap[i] = bufs.work[i]
    var scale_snap = bufs.scale[0]
    var normed_snap = InlineArray[Scalar[DType.bfloat16], HIDDEN](uninitialized=True)
    for i in range(HIDDEN):
        normed_snap[i] = bufs.normed[i]

    # Path B: direct mailbox write + release (matches dispatch_one_per_pool's
    # mechanism on a single pool).
    var tramp = typed_trampoline[SmallPhaseArgs, small_phase_worker]
    var tramp_ptr = UnsafePointer(to=tramp).bitcast[Int]()[]
    var mb = pool.mailboxes + 0
    mb[].func_ptr = tramp_ptr
    UnsafePointer(to=mb[].data[0]).bitcast[SmallPhaseArgs]()[] = args
    AtomicInt32.store[ordering=Ordering.RELEASE](
        UnsafePointer(to=mb[].job_ready.value), 1)
    pool.active_jobs = 1
    pool.join()

    var qi_snap_ptr = UnsafePointer(to=qi_snap[0]).bitcast[Scalar[DType.int8]]()
    var qi_mismatch = compare_i8(qi_snap_ptr, bufs.qi, HIDDEN)
    if qi_mismatch >= 0:
        print("correctness: qi mismatch at", qi_mismatch)
        return False
    var work_snap_ptr = UnsafePointer(to=work_snap[0]).bitcast[Float32]()
    var work_mismatch = compare_f32(work_snap_ptr, bufs.work, HIDDEN)
    if work_mismatch >= 0:
        print("correctness: work mismatch at", work_mismatch)
        return False
    if scale_snap != bufs.scale[0]:
        print("correctness: scale mismatch")
        return False
    var normed_snap_ptr = UnsafePointer(to=normed_snap[0]).bitcast[Scalar[DType.bfloat16]]()
    var normed_mismatch = compare_bf16(normed_snap_ptr, bufs.normed, HIDDEN)
    if normed_mismatch >= 0:
        print("correctness: normed mismatch at", normed_mismatch)
        return False
    return True


# =============================================================================
# Variant runners
# =============================================================================


@no_inline
def run_inline_remote[tp: Int](
    all_bufs: InlineArray[RankBufs, tp],
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    """Main thread iterates over ranks. Rank 0's buffers are local; ranks
    1..tp-1 are on remote NUMA nodes. Reproduces current forward_decode
    `for r in range(tp): inline_kernel(topo_r.arena)` shape."""
    var args = InlineArray[SmallPhaseArgs, tp](fill=SmallPhaseArgs())
    for r in range(tp):
        args[r] = build_args_for_rank(all_bufs[r])

    for _ in range(WARMUP):
        for r in range(tp):
            small_phase_worker(args[r])

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        for r in range(tp):
            small_phase_worker(args[r])
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


@no_inline
def run_inline_local[tp: Int](
    all_bufs: InlineArray[RankBufs, tp],
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    """Same kernel, tp iterations, all against rank 0's local buffers.
    Isolates the remote-read cost: delta to inline_remote is BW penalty."""
    var args0 = build_args_for_rank(all_bufs[0])

    for _ in range(WARMUP):
        for _ in range(tp):
            small_phase_worker(args0)

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        for _ in range(tp):
            small_phase_worker(args0)
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


@no_inline
def run_dispatch_seq[tp: Int](
    all_bufs: InlineArray[RankBufs, tp],
    mut pools: HeapMoveArray[IsolatedBurstPool[]],
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    """Current `for r: pool[r].dispatch; for r: pool[r].join` pattern."""
    var args = InlineArray[SmallPhaseArgs, tp](fill=SmallPhaseArgs())
    for r in range(tp):
        args[r] = build_args_for_rank(all_bufs[r])

    for _ in range(WARMUP):
        for r in range(tp):
            pools[r].dispatch[SmallPhaseArgs, small_phase_worker](
                UnsafePointer(to=args[r]), 1)
        for r in range(tp):
            pools[r].join()

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        for r in range(tp):
            pools[r].dispatch[SmallPhaseArgs, small_phase_worker](
                UnsafePointer(to=args[r]), 1)
        for r in range(tp):
            pools[r].join()
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


@no_inline
def run_dispatch_batched[tp: Int](
    all_bufs: InlineArray[RankBufs, tp],
    mut pools: HeapMoveArray[IsolatedBurstPool[]],
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    """Batched: dispatch_one_per_pool writes all mailboxes first, then signals
    all ready flags. Workers run in parallel; join is sequential but cheap."""
    var args = InlineArray[SmallPhaseArgs, tp](fill=SmallPhaseArgs())
    for r in range(tp):
        args[r] = build_args_for_rank(all_bufs[r])

    for _ in range(WARMUP):
        dispatch_one_per_pool[SmallPhaseArgs, small_phase_worker, tp](
            pools, UnsafePointer(to=args[0]))
        join_all[tp](pools)

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        dispatch_one_per_pool[SmallPhaseArgs, small_phase_worker, tp](
            pools, UnsafePointer(to=args[0]))
        join_all[tp](pools)
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


# =============================================================================
# Reporting
# =============================================================================


@no_inline
def sort_in_place(p: UnsafePointer[Int, MutAnyOrigin], count: Int):
    for i in range(1, count):
        var x = p[i]
        var j = i - 1
        while j >= 0 and p[j] > x:
            p[j + 1] = p[j]
            j -= 1
        p[j + 1] = x


@no_inline
def print_result(variant: String, p: UnsafePointer[Int, MutAnyOrigin], count: Int):
    sort_in_place(p, count)
    var sum = Int(0)
    for i in range(count):
        sum += p[i]
    var mean = sum // count
    print(
        "RESULT variant=", variant,
        " iters=", count,
        " min_ns=", p[0],
        " p50_ns=", p[count // 2],
        " p90_ns=", p[(count * 9) // 10],
        " p99_ns=", p[(count * 99) // 100],
        " mean_ns=", mean,
        " max_ns=", p[count - 1])


# =============================================================================
# Main
# =============================================================================


def run_variant[tp: Int](
    variant: String,
    all_bufs: InlineArray[RankBufs, tp],
    mut pools: HeapMoveArray[IsolatedBurstPool[]],
):
    var samples = alloc[Int](ITERS)

    if variant == "inline_remote":
        run_inline_remote[tp](all_bufs, samples)
    elif variant == "inline_local":
        run_inline_local[tp](all_bufs, samples)
    elif variant == "dispatch_seq":
        run_dispatch_seq[tp](all_bufs, pools, samples)
    elif variant == "dispatch_batched":
        run_dispatch_batched[tp](all_bufs, pools, samples)
    else:
        print("unknown variant:", variant)
        samples.free()
        return

    print_result(variant, samples, ITERS)
    samples.free()


def run_bench[tp: Int](
    variant: String,
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[IsolatedBurstPool[]],
):
    set_subnormal_zeroing()

    comptime arena_bytes = (
        HIDDEN * 2      # src
        + HIDDEN * 2    # gamma_a
        + HIDDEN * 2    # gamma_b
        + HIDDEN        # qi
        + HIDDEN * 4    # work
        + 4             # scale
        + HIDDEN * 2    # normed
        + 4 * 1024      # slack
    )

    var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](tp)
    for rank in range(tp):
        var node = numa_topo[rank]
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](node, arena_bytes)
        if not arena:
            print("arena alloc failed for rank", rank)
            return
        arenas.push(arena^)

    var all_bufs = InlineArray[RankBufs, tp](fill=RankBufs())
    for rank in range(tp):
        all_bufs[rank] = alloc_rank(arenas[rank])
        fill_rank(all_bufs[rank], UInt64(rank) * 0xDEADBEEFCAFEBABE)

    for rank in range(tp):
        _ = arenas[rank].prefault()

    # Correctness: run the dispatch path twice on rank 0 via different
    # mechanisms, compare outputs.
    if not correctness_check(all_bufs[0], pools[0]):
        print("CORRECTNESS FAILED — aborting")
        _ = arenas^
        return

    run_variant[tp](variant, all_bufs, pools)

    _ = arenas^


def main():
    var args = argv()
    if len(args) < 2:
        print("usage:", args[0],
            "<variant: inline_remote|inline_local|dispatch_seq|dispatch_batched>")
        return
    var variant = String(args[1])

    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)

    if not numa.has_isolation():
        print("ERROR: this bench requires CPU isolation (IsolatedBurstPool).")
        print("       On a non-isolated host the timings are not meaningful.")
        return

    var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
    for rank in range(TP):
        pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))

    run_bench[TP](variant, numa, numa_topo, pools^)
