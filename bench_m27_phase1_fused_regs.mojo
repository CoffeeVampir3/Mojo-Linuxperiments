"""Benchmark: register-resident vs stack-buffered w1/w3 -> SiLU -> FWHT -> quantize.

Tests optimization #2 from the MiniMax M2.7 priority ranking: whether keeping
the 128-element gate/up vectors in a SIMD register bank across the
SiLU -> FWHT -> quantize chain beats the current stack-buffer round-trip
implementation in minimax/kernels/gemm.mojo.

Dispatches through the same NumaArena + IsolatedBurstPool path the production
forward uses. Three weight-footprint regimes probe cache residency:
  HOT    = 1 expert (reused)   -> L3-hot after warmup, measures compute
  MIXED  = 4 experts rotated   -> L3-sized working set, mid-cache
  STREAM = 64 experts rotated  -> >> L3, DRAM-bound like prod phase1

Invocation: ./remote_perf.fish bench_m27_phase1_fused_regs.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool

from simd_math import sqrt, set_subnormal_zeroing
from experimental3.common_math import I8Ptr, F32Ptr
from experimental3.kernels.dot_prod import act_broadcast_vnni, dot_vnni_broadcasted
from experimental3.kernels.fwht import fwht_apply, fwht_width
from experimental3.kernels.rope_and_kv_cache_write import regs_absmax_quantize_i8
from kernels.vnni import (
    VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK,
    pack_and_colsum_vnni,
)
from kernels.kernel_ops import MAX_POOL_CAPACITY
from modeling.model_spec import DEFAULT_ALIGNMENT

from minimax.kernels.dispatch_args import FusedW1W3SiluArgs
from minimax.kernels.activations import silu_mul
from minimax.kernels.gemm import fused_w1_w3_silu_worker


# =============================================================================
# Problem shape — matches MiniMax M2.7 expert_phase1
# =============================================================================


comptime K_DIM = 3072          # HIDDEN
comptime INTERMEDIATE = 1536   # expert inner width
comptime FWHT_BLK = 128        # per-tile output size
comptime N_TILES = INTERMEDIATE // FWHT_BLK    # 12 tiles per expert

comptime TP = 4

comptime WARMUP = 200
comptime ITERS = 2000

# Number of synthetic experts in the per-rank weight bank per regime.
# Per-expert bytes: 2 * INTERMEDIATE * K_DIM + 4 * INTERMEDIATE * 4
#                 = 2 * 1536 * 3072 + 4 * 6KB ~= 9.47 MB
comptime BANK_HOT = 1
comptime BANK_MIXED = 4
comptime BANK_STREAM = 64
comptime BANK_MAX = BANK_STREAM


# =============================================================================
# Candidate kernel — register-resident GEMV -> SiLU -> FWHT -> quantize
# =============================================================================


@always_inline
def reg_fused_tile[K: Int, fwht_blk: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr, w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    qi_out: I8Ptr,
) -> Float32:
    """One fwht_blk=128-wide output tile: w1/w3 GEMV dequantized directly into
    a SIMD register bank, SiLU-mul'd in place, block-FWHT'd in place, then
    absmax-quantized and stored out as i8. No L1 round-trip for gate/up.

    Bit-exact with the existing baseline when given the same inputs and the
    same VNNI traversal order: all stages use identical math, and f32
    round-tripping through memory is value-preserving.

    Returns the absmax (= per-block scale); caller stores it to blk_scale.
    """
    debug_assert(K % VNNI_K_STEP == 0,
        "reg_fused_tile: K must be a multiple of VNNI_K_STEP (64)")
    debug_assert(fwht_blk % VNNI_N_STEP == 0,
        "reg_fused_tile: fwht_blk must be a multiple of VNNI_N_STEP (32)")

    # Use fwht_width[f32, fwht_blk]() as the single source of truth so the
    # register bank type matches fwht_apply's signature. We also need this
    # to equal the i32 SIMD width for the VNNI accumulators — true on
    # AVX-512 (both are 16), asserted here.
    comptime width = fwht_width[DType.float32, fwht_blk]()
    comptime assert width == simd_width_of[DType.int32](),
        "reg_fused_tile assumes fwht_width[f32, fwht_blk] == simd_width[i32]"
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_per_tile = VNNI_N_STEP // width
    comptime tiles_per_fwht = fwht_blk // VNNI_N_STEP
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes
    comptime total_regs = fwht_blk // width

    var gate_bank = InlineArray[SIMD[DType.float32, width], total_regs](
        uninitialized=True)
    var up_bank = InlineArray[SIMD[DType.float32, width], total_regs](
        uninitialized=True)

    var packed_off = 0

    comptime for tile in range(tiles_per_fwht):
        var w1_acc = InlineArray[SIMD[DType.int32, width], acc_per_tile](
            fill=SIMD[DType.int32, width](0))
        var w3_acc = InlineArray[SIMD[DType.int32, width], acc_per_tile](
            fill=SIMD[DType.int32, width](0))

        for ks in range(0, K, VNNI_K_STEP):
            for dc in range(dc_count):
                var k_pos = ks + dc * VNNI_BLK
                var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                var t0 = packed_off + dc * tile_dc_bytes
                var t1 = t0 + tile_ks_bytes

                comptime for p in range(passes):
                    var off = t0 + p * bytes_per_pass
                    w1_acc[p] = dot_vnni_broadcasted[width](
                        w1_acc[p], act_bytes, w1_packed + off)
                    w3_acc[p] = dot_vnni_broadcasted[width](
                        w3_acc[p], act_bytes, w3_packed + off)

                comptime for p in range(passes):
                    var off = t1 + p * bytes_per_pass
                    w1_acc[passes + p] = dot_vnni_broadcasted[width](
                        w1_acc[passes + p], act_bytes, w1_packed + off)
                    w3_acc[passes + p] = dot_vnni_broadcasted[width](
                        w3_acc[passes + p], act_bytes, w3_packed + off)

            packed_off += 2 * tile_ks_bytes

        comptime for a in range(acc_per_tile):
            var n_base = tile * VNNI_N_STEP + a * width
            var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
            var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
            comptime reg_idx = tile * acc_per_tile + a
            gate_bank[reg_idx] = w1_corr * act_sc * (w1_sc + n_base).load[width=width]()
            up_bank[reg_idx] = w3_corr * act_sc * (w3_sc + n_base).load[width=width]()

    comptime for i in range(total_regs):
        gate_bank[i] = silu_mul(gate_bank[i], up_bank[i])

    fwht_apply[DType.float32, fwht_blk](gate_bank)

    var quant = regs_absmax_quantize_i8(gate_bank, qi_out)
    return quant[0]


def candidate_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    """Pool worker wrapping reg_fused_tile. Same outer-loop shape as
    fused_w1_w3_silu_worker so the pool-dispatch contract is identical."""
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n
            var absmax = reg_fused_tile[K, fwht_blk](
                act_i8,
                args.w1_packed + n_off * K,
                args.w3_packed + n_off * K,
                dequant,
                args.w1_scale + n_off, args.w1_colsum + n_off,
                args.w3_scale + n_off, args.w3_colsum + n_off,
                qi_row + local_n)
            blk_row[local_n // fwht_blk] = absmax
            local_n += fwht_blk


# =============================================================================
# Rank-local buffer layout
# =============================================================================


@fieldwise_init
struct RankBuffers(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var act_scale: F32Ptr
    var w1_bank: I8Ptr
    var w3_bank: I8Ptr
    var w1_sc_bank: F32Ptr
    var w1_cs_bank: F32Ptr
    var w3_sc_bank: F32Ptr
    var w3_cs_bank: F32Ptr
    var qi_out_base: I8Ptr
    var qi_out_cand: I8Ptr
    var blk_sc_base: F32Ptr
    var blk_sc_cand: F32Ptr

    def __init__(out self):
        self.act = I8Ptr()
        self.act_scale = F32Ptr()
        self.w1_bank = I8Ptr()
        self.w3_bank = I8Ptr()
        self.w1_sc_bank = F32Ptr()
        self.w1_cs_bank = F32Ptr()
        self.w3_sc_bank = F32Ptr()
        self.w3_cs_bank = F32Ptr()
        self.qi_out_base = I8Ptr()
        self.qi_out_cand = I8Ptr()
        self.blk_sc_base = F32Ptr()
        self.blk_sc_cand = F32Ptr()


# =============================================================================
# Deterministic data fills
# =============================================================================


@no_inline
def fill_i8(p: I8Ptr, count: Int, seed: UInt64):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        p[i] = Scalar[DType.int8]((state >> 33).cast[DType.int8]())


@no_inline
def fill_f32_positive(p: F32Ptr, count: Int, seed: UInt64, scale: Float32):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = ((state >> 33) & 0xFFFFFF).cast[DType.uint32]()
        p[i] = Float32(u) / Float32(0x1000000) * scale + Float32(1e-4)


def alloc_rank(
    mut arena: NumaArena[alignment=DEFAULT_ALIGNMENT],
) -> RankBuffers:
    """Bump-allocate all rank-local buffers; return the pointer bundle."""
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var r = RankBuffers()
    r.act = arena.alloc[Scalar[DType.int8]](K_DIM)
    r.act_scale = arena.alloc[Float32](1)

    r.w1_bank = arena.alloc[Scalar[DType.int8]](BANK_MAX * weight_elems_per_expert)
    r.w3_bank = arena.alloc[Scalar[DType.int8]](BANK_MAX * weight_elems_per_expert)
    r.w1_sc_bank = arena.alloc[Float32](BANK_MAX * sc_count_per_expert)
    r.w1_cs_bank = arena.alloc[Float32](BANK_MAX * sc_count_per_expert)
    r.w3_sc_bank = arena.alloc[Float32](BANK_MAX * sc_count_per_expert)
    r.w3_cs_bank = arena.alloc[Float32](BANK_MAX * sc_count_per_expert)

    r.qi_out_base = arena.alloc[Scalar[DType.int8]](BANK_MAX * INTERMEDIATE)
    r.qi_out_cand = arena.alloc[Scalar[DType.int8]](BANK_MAX * INTERMEDIATE)
    r.blk_sc_base = arena.alloc[Float32](BANK_MAX * N_TILES)
    r.blk_sc_cand = arena.alloc[Float32](BANK_MAX * N_TILES)
    return r^


def fill_rank_data(
    bufs: RankBuffers,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    rank_seed: UInt64,
):
    """Fill + VNNI-pack the rank-local weight bank deterministically."""
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    fill_i8(bufs.act, K_DIM, rank_seed ^ 0xA1)
    bufs.act_scale[0] = Float32(0.05)

    for e in range(BANK_MAX):
        var w1 = bufs.w1_bank + e * weight_elems_per_expert
        var w3 = bufs.w3_bank + e * weight_elems_per_expert
        var w1_sc = bufs.w1_sc_bank + e * sc_count_per_expert
        var w1_cs = bufs.w1_cs_bank + e * sc_count_per_expert
        var w3_sc = bufs.w3_sc_bank + e * sc_count_per_expert
        var w3_cs = bufs.w3_cs_bank + e * sc_count_per_expert

        var seed_e = rank_seed ^ (UInt64(e) * 0x9E3779B97F4A7C15)
        fill_i8(w1, weight_elems_per_expert, seed_e)
        fill_i8(w3, weight_elems_per_expert, seed_e ^ 0x5555555555555555)
        fill_f32_positive(w1_sc, sc_count_per_expert, seed_e ^ 0xAA, Float32(0.01))
        fill_f32_positive(w3_sc, sc_count_per_expert, seed_e ^ 0xBB, Float32(0.01))

        # In-place VNNI pack + per-row colsum. See init_weights.mojo for the
        # same src==dst pattern.
        pack_and_colsum_vnni(
            w1.bitcast[UInt8](), w1.bitcast[UInt8](), scratch,
            INTERMEDIATE, K_DIM, K_DIM,
            w1_cs, True)
        pack_and_colsum_vnni(
            w3.bitcast[UInt8](), w3.bitcast[UInt8](), scratch,
            INTERMEDIATE, K_DIM, K_DIM,
            w3_cs, True)


# =============================================================================
# Job-array construction
# =============================================================================


@always_inline
def fill_jobs(
    mut jobs: InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
):
    """Build N_TILES worker jobs for one expert. Each worker owns one tile."""
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var w1 = bufs.w1_bank + expert_idx * weight_elems_per_expert
    var w3 = bufs.w3_bank + expert_idx * weight_elems_per_expert
    var w1_sc = bufs.w1_sc_bank + expert_idx * sc_count_per_expert
    var w1_cs = bufs.w1_cs_bank + expert_idx * sc_count_per_expert
    var w3_sc = bufs.w3_sc_bank + expert_idx * sc_count_per_expert
    var w3_cs = bufs.w3_cs_bank + expert_idx * sc_count_per_expert

    for t in range(N_TILES):
        var n_start = t * FWHT_BLK
        # Match production minimax_moe_phase1: pre-offset qi_out/blk_sc by the
        # tile's global N position. The worker writes to qi_out+local_n, so
        # without the pre-offset every tile races for qi_out[0..FWHT_BLK].
        jobs[t] = FusedW1W3SiluArgs(
            bufs.act, bufs.act_scale,
            w1, w1_sc, w1_cs,
            w3, w3_sc, w3_cs,
            qi_out + n_start, blk_sc + t,
            n_start, FWHT_BLK, 1)


# =============================================================================
# Correctness gate
# =============================================================================


@no_inline
def compare_i8(a: I8Ptr, b: I8Ptr, count: Int) -> Int:
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


def correctness_check[P: BurstThreadPool, //](
    bufs: RankBuffers, mut pool: P,
) -> Bool:
    """Run both workers on the SAME inputs; bitwise-compare outputs."""
    var jobs = InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs())

    fill_jobs(jobs, bufs, 0, bufs.qi_out_base, bufs.blk_sc_base)
    pool.dispatch[FusedW1W3SiluArgs,
        fused_w1_w3_silu_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
        UnsafePointer(to=jobs[0]), N_TILES)
    pool.join()

    fill_jobs(jobs, bufs, 0, bufs.qi_out_cand, bufs.blk_sc_cand)
    pool.dispatch[FusedW1W3SiluArgs,
        candidate_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
        UnsafePointer(to=jobs[0]), N_TILES)
    pool.join()

    var qi_mismatch = compare_i8(bufs.qi_out_base, bufs.qi_out_cand, INTERMEDIATE)
    if qi_mismatch >= 0:
        print("  correctness: qi_out mismatch at index", qi_mismatch,
            "baseline=", Int(bufs.qi_out_base[qi_mismatch]),
            "candidate=", Int(bufs.qi_out_cand[qi_mismatch]))
        return False

    var sc_mismatch = compare_f32(bufs.blk_sc_base, bufs.blk_sc_cand, N_TILES)
    if sc_mismatch >= 0:
        print("  correctness: blk_scale mismatch at index", sc_mismatch,
            "baseline=", bufs.blk_sc_base[sc_mismatch],
            "candidate=", bufs.blk_sc_cand[sc_mismatch])
        return False

    return True


# =============================================================================
# Timed runs — one per kernel variant
# =============================================================================


@no_inline
def time_baseline[P: BurstThreadPool, tp: Int, //](
    all_bufs: InlineArray[RankBuffers, tp],
    mut pools: HeapMoveArray[P],
    bank_experts: Int,
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    var all_jobs = InlineArray[
        InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY], tp
    ](fill=InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs()))

    for w in range(WARMUP):
        var e = w % bank_experts
        for r in range(tp):
            fill_jobs(all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base)
        for r in range(tp):
            pools[r].dispatch[FusedW1W3SiluArgs,
                fused_w1_w3_silu_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
                UnsafePointer(to=all_jobs[r][0]), N_TILES)
        for r in range(tp):
            pools[r].join()

    for i in range(ITERS):
        var e = i % bank_experts
        for r in range(tp):
            fill_jobs(all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base)
        var t0 = perf_counter_ns()
        for r in range(tp):
            pools[r].dispatch[FusedW1W3SiluArgs,
                fused_w1_w3_silu_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
                UnsafePointer(to=all_jobs[r][0]), N_TILES)
        for r in range(tp):
            pools[r].join()
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


@no_inline
def time_candidate[P: BurstThreadPool, tp: Int, //](
    all_bufs: InlineArray[RankBuffers, tp],
    mut pools: HeapMoveArray[P],
    bank_experts: Int,
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    var all_jobs = InlineArray[
        InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY], tp
    ](fill=InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs()))

    for w in range(WARMUP):
        var e = w % bank_experts
        for r in range(tp):
            fill_jobs(all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_cand, all_bufs[r].blk_sc_cand)
        for r in range(tp):
            pools[r].dispatch[FusedW1W3SiluArgs,
                candidate_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
                UnsafePointer(to=all_jobs[r][0]), N_TILES)
        for r in range(tp):
            pools[r].join()

    for i in range(ITERS):
        var e = i % bank_experts
        for r in range(tp):
            fill_jobs(all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_cand, all_bufs[r].blk_sc_cand)
        var t0 = perf_counter_ns()
        for r in range(tp):
            pools[r].dispatch[FusedW1W3SiluArgs,
                candidate_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
                UnsafePointer(to=all_jobs[r][0]), N_TILES)
        for r in range(tp):
            pools[r].join()
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


# =============================================================================
# Reporting
# =============================================================================


@no_inline
def sort_in_place(p: UnsafePointer[Int, MutAnyOrigin], count: Int):
    """Insertion sort; fine for count ~= 2000."""
    for i in range(1, count):
        var x = p[i]
        var j = i - 1
        while j >= 0 and p[j] > x:
            p[j + 1] = p[j]
            j -= 1
        p[j + 1] = x


@no_inline
def report(label: String, p: UnsafePointer[Int, MutAnyOrigin], count: Int):
    sort_in_place(p, count)
    var sum = Int(0)
    for i in range(count):
        sum += p[i]
    var mean = sum // count
    print("   ", label,
        " min=", p[0], "ns",
        " p50=", p[count // 2], "ns",
        " mean=", mean, "ns",
        " p90=", p[(count * 9) // 10], "ns",
        " p99=", p[(count * 99) // 100], "ns",
        " max=", p[count - 1], "ns")


# =============================================================================
# Regime runner
# =============================================================================


def run_regime[P: BurstThreadPool, tp: Int, //](
    label: String,
    bank_experts: Int,
    all_bufs: InlineArray[RankBuffers, tp],
    mut pools: HeapMoveArray[P],
):
    var footprint_mb = bank_experts * 2 * INTERMEDIATE * K_DIM // (1024 * 1024)
    print("  === ", label,
        " (bank_experts=", bank_experts,
        ", w1+w3 footprint per rank =", footprint_mb, "MB ) ===")

    var samples_base = alloc[Int](ITERS)
    var samples_cand = alloc[Int](ITERS)

    # Pass 1: baseline -> candidate
    time_baseline(all_bufs, pools, bank_experts, samples_base)
    time_candidate(all_bufs, pools, bank_experts, samples_cand)
    print("  -- pass 1: baseline -> candidate --")
    report("baseline ", samples_base, ITERS)
    report("candidate", samples_cand, ITERS)

    # Pass 2: candidate -> baseline (catches first-run/cache-warming bias)
    time_candidate(all_bufs, pools, bank_experts, samples_cand)
    time_baseline(all_bufs, pools, bank_experts, samples_base)
    print("  -- pass 2: candidate -> baseline --")
    report("candidate", samples_cand, ITERS)
    report("baseline ", samples_base, ITERS)

    samples_base.free()
    samples_cand.free()


# =============================================================================
# Main
# =============================================================================


def run_bench[P: BurstThreadPool, //, tp: Int](
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
):
    set_subnormal_zeroing()

    comptime weight_bytes_per_expert = (
        2 * INTERMEDIATE * K_DIM
        + 4 * INTERMEDIATE * 4
    )
    comptime arena_bytes = (
        BANK_MAX * weight_bytes_per_expert
        + 2 * BANK_MAX * INTERMEDIATE
        + 2 * BANK_MAX * N_TILES * 4
        + 4 * 1024 * 1024
    )
    # Pack scratch — compute_n_block(1536, 3072) = 64; scratch = 64 * K
    comptime scratch_bytes = 64 * K_DIM

    var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](tp)
    for rank in range(tp):
        var node = numa_topo[rank]
        print("rank", rank, "node", node,
            "allocating", arena_bytes // (1024 * 1024), "MB")
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](node, arena_bytes)
        if not arena:
            print("arena alloc failed")
            return
        arenas.push(arena^)

    var scratch = alloc[UInt8](scratch_bytes)

    var all_bufs = InlineArray[RankBuffers, tp](fill=RankBuffers())
    for rank in range(tp):
        all_bufs[rank] = alloc_rank(arenas[rank])
        fill_rank_data(all_bufs[rank], scratch,
            UInt64(rank) * 0xDEADBEEFCAFEBABE)

    scratch.free()

    for rank in range(tp):
        _ = arenas[rank].prefault()

    print("")
    print("running correctness check (rank 0)...")
    if not correctness_check(all_bufs[0], pools[0]):
        print("CORRECTNESS FAILED -- aborting timed section")
        _ = arenas^
        return
    print("correctness: PASS (bitwise qi_out, exact-eq blk_scale)")
    print("")

    print("config:",
        " TP=", tp,
        " INTERMEDIATE=", INTERMEDIATE,
        " K=", K_DIM,
        " FWHT_BLK=", FWHT_BLK,
        " N_TILES=", N_TILES,
        " WARMUP=", WARMUP,
        " ITERS=", ITERS)
    print("pool[0].capacity:", pools[0].get_capacity())
    print("")

    run_regime("HOT", BANK_HOT, all_bufs, pools)
    print("")
    run_regime("MIXED", BANK_MIXED, all_bufs, pools)
    print("")
    run_regime("STREAM", BANK_STREAM, all_bufs, pools)

    _ = arenas^


def main():
    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)

    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        run_bench[TP](numa, numa_topo, pools^)
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        run_bench[TP](numa, numa_topo, pools^)
