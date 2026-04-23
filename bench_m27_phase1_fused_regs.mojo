"""Benchmark: MiniMax M2.7 phase-1 expert GEMV kernel variants.

One invocation runs one (regime, variant) pair so `perf stat` counters
attribute cleanly:

    ./bench_m27_phase1_fused_regs <regime> <variant>

  regime:  hot | mixed | stream      (bank: 1 | 4 | 64 experts per rank)
  variant: baseline | candidate | prefetch_t0 | prefetch_nta
           | fanout_1 | fanout_2 | fanout_3 | fanout_4 | fanout_6
           | nblock32 | nblock64 | loadfirst | pipe2 | layout_zip
           | ksplit2 | ksplit4

Variants under test:
  baseline      — minimax/kernels/gemm.mojo fused_w1_w3_silu_worker, as-is.
  candidate     — register-resident gate/up through SiLU -> FWHT -> quantize
                  (bench-local; hypothesis refuted by the v1 run).
  prefetch_t0   — baseline inner loop + prefetcht0 ahead of w1/w3 stream.
  prefetch_nta  — baseline inner loop + prefetchnta ahead of w1/w3 stream.
                  Targets the LFB-saturation bottleneck seen in v1 perf stat
                  (fb_full cycles = 82B). NT hint marks lines for early L1
                  eviction so cache pressure on reused data (scales, colsums,
                  activation) drops.

STREAM setup uses pack-once-then-memcpy to keep pack time small; each expert
slot still lives on a distinct physical page so DRAM traffic is realistic.

Correctness: before timing, the selected variant is run against the baseline
worker on identical inputs; qi_out is compared byte-for-byte, blk_scale f32
by f32. Aborts if they diverge.

Invocation (see bench_m27_matrix.fish for the full sweep):
    perf stat -e <events> ./bench_m27_phase1_fused_regs hot baseline
"""

from std.sys import argv
from std.sys import llvm_intrinsic
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool

from simd_math import set_subnormal_zeroing
from experimental3.common_math import I8Ptr, F32Ptr
from experimental3.kernels.dot_prod import (
    act_broadcast_vnni, dot_vnni_broadcasted, vpdpbusd,
)
from experimental3.kernels.fwht import fwht_apply, fwht_width, fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
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


comptime K_DIM = 3072
comptime INTERMEDIATE = 1536
comptime FWHT_BLK = 128
comptime N_TILES = INTERMEDIATE // FWHT_BLK

comptime TP = 4

comptime WARMUP = 300
comptime ITERS = 10000

comptime BANK_HOT = 1
comptime BANK_MIXED = 4
comptime BANK_STREAM = 64
comptime BANK_MAX = BANK_STREAM
comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]
comptime KSPLIT_MAX = 4
comptime KSPLIT_PARTIAL_I32 = N_TILES * KSPLIT_MAX * 2 * FWHT_BLK

# Bytes of w1/w3 data ahead of the demand read to prefetch. 2 KB = 32 cache
# lines = one VNNI_K_STEP's worth of streaming. Starting point; easy to tune
# by editing this and rerunning the matrix.
comptime PREFETCH_AHEAD = 2048


# =============================================================================
# Prefetch intrinsic wrappers — pattern from amx_ablations2.mojo
# =============================================================================


comptime PF_NONE = 0
comptime PF_T0 = 1
comptime PF_NTA = 2


@always_inline
def do_prefetch[mode: Int](p: I8Ptr):
    comptime if mode == PF_T0:
        llvm_intrinsic["llvm.prefetch.p0", NoneType](
            p.bitcast[UInt8](), Int32(0), Int32(3), Int32(1))
    elif mode == PF_NTA:
        llvm_intrinsic["llvm.prefetch.p0", NoneType](
            p.bitcast[UInt8](), Int32(0), Int32(0), Int32(1))


# =============================================================================
# Candidate kernel — register-resident (reused from v1; hypothesis refuted
# but kept so the matrix has a reference point)
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
    debug_assert(K % VNNI_K_STEP == 0, "K must be a multiple of VNNI_K_STEP")
    debug_assert(fwht_blk % VNNI_N_STEP == 0, "fwht_blk must be a multiple of VNNI_N_STEP")

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
# Prefetch variants — forked baseline with prefetch injected in the GEMV
# inner loop. The non-prefetch parts (SiLU, FWHT, quantize) match baseline
# byte-for-byte to keep any delta attributable to the prefetch itself.
# =============================================================================


def gemv_row_pf[N: Int, K: Int, mode: Int, ahead: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr, w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    gate: F32Ptr, up: F32Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = _compute_n_block_local(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_off + dc * tile_dc_bytes
                    var t1 = t0 + tile_ks_bytes

                    # Prefetch hint for the next ks step on both matrices.
                    # Issued once per dc iter; the HW LFB coalesces redundant
                    # prefetches to the same line.
                    do_prefetch[mode](w1_packed + t0 + ahead)
                    do_prefetch[mode](w3_packed + t0 + ahead)

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

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


# Local copy of compute_n_block to keep traversal identical to baseline.
@always_inline
def _compute_n_block_local(n: Int, k: Int) -> Int:
    comptime L2_TARGET = 256 * 1024
    var max_n = L2_TARGET // k
    var n_block = (max_n // VNNI_N_STEP) * VNNI_N_STEP
    if n_block >= n:
        return n
    if n_block >= VNNI_N_STEP:
        return n_block
    return VNNI_N_STEP


def prefetch_worker[
    intermediate: Int, K: Int, fwht_blk: Int,
    mode: Int, ahead: Int,
](args: FusedW1W3SiluArgs):
    debug_assert(intermediate % fwht_blk == 0,
        "prefetch_worker: intermediate must be a multiple of fwht_blk")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()

            gemv_row_pf[fwht_blk, K, mode, ahead](
                act_i8,
                args.w1_packed + n_off * K,
                args.w3_packed + n_off * K,
                dequant,
                args.w1_scale + n_off, args.w1_colsum + n_off,
                args.w3_scale + n_off, args.w3_colsum + n_off,
                gate, up)

            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width

            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)
            local_n += fwht_blk


# =============================================================================
# Hypothesis variants: fixed n_block, explicit load scheduling, fused layout mock
# =============================================================================


def gemv_row_nblock[N: Int, K: Int, n_block_override: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr, w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    gate: F32Ptr, up: F32Ptr,
):
    comptime assert N % n_block_override == 0, "n_block must divide N"
    comptime assert n_block_override % VNNI_N_STEP == 0,
        "n_block must be a multiple of VNNI_N_STEP"
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var packed_off = 0
    for nb in range(0, N, n_block_override):
        for ns in range(0, n_block_override, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
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

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


def nblock_worker[
    intermediate: Int, K: Int, fwht_blk: Int, n_block_override: Int,
](args: FusedW1W3SiluArgs):
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n
            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()

            gemv_row_nblock[fwht_blk, K, n_block_override](
                act_i8,
                args.w1_packed + n_off * K,
                args.w3_packed + n_off * K,
                dequant,
                args.w1_scale + n_off, args.w1_colsum + n_off,
                args.w3_scale + n_off, args.w3_colsum + n_off,
                gate, up)

            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width

            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)
            local_n += fwht_blk


def gemv_row_loadfirst[N: Int, K: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr, w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    gate: F32Ptr, up: F32Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = _compute_n_block_local(N, K)
    var packed_off = 0
    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_off + dc * tile_dc_bytes
                    var t1 = t0 + tile_ks_bytes
                    comptime for p in range(passes):
                        var off = t0 + p * bytes_per_pass
                        var w1v = (w1_packed + off).load[width=width * 4]()
                        var w3v = (w3_packed + off).load[width=width * 4]()
                        w1_acc[p] = vpdpbusd[width](w1_acc[p], act_bytes, w1v)
                        w3_acc[p] = vpdpbusd[width](w3_acc[p], act_bytes, w3v)
                    comptime for p in range(passes):
                        var off = t1 + p * bytes_per_pass
                        var w1v = (w1_packed + off).load[width=width * 4]()
                        var w3v = (w3_packed + off).load[width=width * 4]()
                        w1_acc[passes + p] = vpdpbusd[width](
                            w1_acc[passes + p], act_bytes, w1v)
                        w3_acc[passes + p] = vpdpbusd[width](
                            w3_acc[passes + p], act_bytes, w3v)
                packed_off += 2 * tile_ks_bytes

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


def gemv_row_pipe2[N: Int, K: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr, w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    gate: F32Ptr, up: F32Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime assert VNNI_TILE_N == width,
        "pipe2 assumes one SIMD vector per VNNI half tile"
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime assert acc_count == 2, "pipe2 assumes two accumulators"
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = _compute_n_block_local(N, K)
    var packed_off = 0
    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                var dc = 0
                while dc + 1 < dc_count:
                    var k0 = ks + dc * VNNI_BLK
                    var k1 = ks + (dc + 1) * VNNI_BLK
                    var act0 = act_broadcast_vnni[width](act_row, k0)
                    var act1 = act_broadcast_vnni[width](act_row, k1)
                    var t00 = packed_off + dc * tile_dc_bytes
                    var t01 = packed_off + (dc + 1) * tile_dc_bytes
                    var t10 = t00 + tile_ks_bytes
                    var t11 = t01 + tile_ks_bytes

                    var w1_00 = (w1_packed + t00).load[width=width * 4]()
                    var w3_00 = (w3_packed + t00).load[width=width * 4]()
                    var w1_10 = (w1_packed + t10).load[width=width * 4]()
                    var w3_10 = (w3_packed + t10).load[width=width * 4]()
                    var w1_01 = (w1_packed + t01).load[width=width * 4]()
                    var w3_01 = (w3_packed + t01).load[width=width * 4]()
                    var w1_11 = (w1_packed + t11).load[width=width * 4]()
                    var w3_11 = (w3_packed + t11).load[width=width * 4]()

                    w1_acc[0] = vpdpbusd[width](w1_acc[0], act0, w1_00)
                    w3_acc[0] = vpdpbusd[width](w3_acc[0], act0, w3_00)
                    w1_acc[1] = vpdpbusd[width](w1_acc[1], act0, w1_10)
                    w3_acc[1] = vpdpbusd[width](w3_acc[1], act0, w3_10)
                    w1_acc[0] = vpdpbusd[width](w1_acc[0], act1, w1_01)
                    w3_acc[0] = vpdpbusd[width](w3_acc[0], act1, w3_01)
                    w1_acc[1] = vpdpbusd[width](w1_acc[1], act1, w1_11)
                    w3_acc[1] = vpdpbusd[width](w3_acc[1], act1, w3_11)
                    dc += 2
                packed_off += 2 * tile_ks_bytes

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


def gemv_row_layout_zip[N: Int, K: Int](
    act_row: I8Ptr,
    w13_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3_sc: F32Ptr, w3_cs: F32Ptr,
    gate: F32Ptr, up: F32Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = _compute_n_block_local(N, K)
    var packed_off = 0
    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_off + dc * tile_dc_bytes
                    var t1 = t0 + tile_ks_bytes
                    comptime for p in range(passes):
                        var off = t0 + p * bytes_per_pass
                        var w1v = (w13_packed + 2 * off).load[width=width * 4]()
                        var w3v = (w13_packed + 2 * off + bytes_per_pass).load[width=width * 4]()
                        w1_acc[p] = vpdpbusd[width](w1_acc[p], act_bytes, w1v)
                        w3_acc[p] = vpdpbusd[width](w3_acc[p], act_bytes, w3v)
                    comptime for p in range(passes):
                        var off = t1 + p * bytes_per_pass
                        var w1v = (w13_packed + 2 * off).load[width=width * 4]()
                        var w3v = (w13_packed + 2 * off + bytes_per_pass).load[width=width * 4]()
                        w1_acc[passes + p] = vpdpbusd[width](
                            w1_acc[passes + p], act_bytes, w1v)
                        w3_acc[passes + p] = vpdpbusd[width](
                            w3_acc[passes + p], act_bytes, w3v)
                packed_off += 2 * tile_ks_bytes

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


def load_sched_worker[
    intermediate: Int, K: Int, fwht_blk: Int, pipe2: Bool,
](args: FusedW1W3SiluArgs):
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row
        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n
            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            comptime if pipe2:
                gemv_row_pipe2[fwht_blk, K](
                    act_i8, args.w1_packed + n_off * K,
                    args.w3_packed + n_off * K, dequant,
                    args.w1_scale + n_off, args.w1_colsum + n_off,
                    args.w3_scale + n_off, args.w3_colsum + n_off,
                    gate, up)
            else:
                gemv_row_loadfirst[fwht_blk, K](
                    act_i8, args.w1_packed + n_off * K,
                    args.w3_packed + n_off * K, dequant,
                    args.w1_scale + n_off, args.w1_colsum + n_off,
                    args.w3_scale + n_off, args.w3_colsum + n_off,
                    gate, up)
            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width
            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)
            local_n += fwht_blk


def layout_zip_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row
        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n
            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row_layout_zip[fwht_blk, K](
                act_i8, args.w1_packed + 2 * n_off * K, dequant,
                args.w1_scale + n_off, args.w1_colsum + n_off,
                args.w3_scale + n_off, args.w3_colsum + n_off,
                gate, up)
            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width
            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)
            local_n += fwht_blk


# =============================================================================
# K-split variants — split the K dimension across workers, reduce int32
# partials, then run the same f32 epilogue as baseline.
# =============================================================================


@fieldwise_init
struct KSplitPartialArgs(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var w1_packed: I8Ptr
    var w3_packed: I8Ptr
    var gate_part: I32Ptr
    var up_part: I32Ptr
    var n_start: Int
    var k_start: Int
    var k_count: Int

    def __init__(out self):
        self.act_i8 = I8Ptr()
        self.w1_packed = I8Ptr()
        self.w3_packed = I8Ptr()
        self.gate_part = I32Ptr()
        self.up_part = I32Ptr()
        self.n_start = 0
        self.k_start = 0
        self.k_count = 0


@fieldwise_init
struct KSplitFinalArgs(Copyable, ImplicitlyCopyable):
    var act_scale: F32Ptr
    var w1_scale: F32Ptr
    var w1_colsum: F32Ptr
    var w3_scale: F32Ptr
    var w3_colsum: F32Ptr
    var partials: I32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var split_count: Int

    def __init__(out self):
        self.act_scale = F32Ptr()
        self.w1_scale = F32Ptr()
        self.w1_colsum = F32Ptr()
        self.w3_scale = F32Ptr()
        self.w3_colsum = F32Ptr()
        self.partials = I32Ptr()
        self.qi_out = I8Ptr()
        self.blk_scale = F32Ptr()
        self.n_start = 0
        self.split_count = 0


def gemv_row_ksplit_partial[N: Int, K: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr,
    w3_packed: I8Ptr,
    k_start: Int,
    k_count: Int,
    gate_part: I32Ptr,
    up_part: I32Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = _compute_n_block_local(N, K)
    var k_end = k_start + k_count
    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(k_start, k_end, VNNI_K_STEP):
                var packed_base = (nb + ns) * K + ks * VNNI_N_STEP
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_base + dc * tile_dc_bytes
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

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                (gate_part + n_base).store(w1_acc[a])
                (up_part + n_base).store(w3_acc[a])


def ksplit_partial_worker[fwht_blk: Int, K: Int](args: KSplitPartialArgs):
    gemv_row_ksplit_partial[fwht_blk, K](
        args.act_i8,
        args.w1_packed + args.n_start * K,
        args.w3_packed + args.n_start * K,
        args.k_start,
        args.k_count,
        args.gate_part,
        args.up_part)


def ksplit_final_worker[fwht_blk: Int](args: KSplitFinalArgs):
    comptime width = simd_width_of[DType.float32]()
    var dequant = args.act_scale[0] / 127.0

    var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
    var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()

    var k = 0
    while k + width <= fwht_blk:
        var gate_sum = (args.partials + k).load[width=width]()
        var up_sum = (args.partials + fwht_blk + k).load[width=width]()

        var split = 1
        while split < args.split_count:
            var split_off = split * 2 * fwht_blk
            gate_sum += (args.partials + split_off + k).load[width=width]()
            up_sum += (args.partials + split_off + fwht_blk + k).load[width=width]()
            split += 1

        var n_base = args.n_start + k
        var gate_f = (
            gate_sum.cast[DType.float32]()
            - Float32(128) * (args.w1_colsum + n_base).load[width=width]()
        ) * dequant * (args.w1_scale + n_base).load[width=width]()
        var up_f = (
            up_sum.cast[DType.float32]()
            - Float32(128) * (args.w3_colsum + n_base).load[width=width]()
        ) * dequant * (args.w3_scale + n_base).load[width=width]()
        (gate + k).store(silu_mul(gate_f, up_f))
        k += width

    fwht_block[fwht_blk](gate)
    args.blk_scale[0] = absmax_quantize_i8[fwht_blk](gate, args.qi_out)


# =============================================================================
# Rank-local buffer layout
# =============================================================================


@always_inline
def is_fanout_variant(variant: String) -> Bool:
    return (
        variant == "fanout_1" or variant == "fanout_2"
        or variant == "fanout_3" or variant == "fanout_4"
        or variant == "fanout_6")


@always_inline
def is_ksplit_variant(variant: String) -> Bool:
    return variant == "ksplit2" or variant == "ksplit4"


@always_inline
def ksplit_for_variant(variant: String) -> Int:
    if variant == "ksplit2":
        return 2
    elif variant == "ksplit4":
        return 4
    return 1


@always_inline
def fanout_jobs_for_variant(variant: String) -> Int:
    if variant == "fanout_1":
        return 1
    elif variant == "fanout_2":
        return 2
    elif variant == "fanout_3":
        return 3
    elif variant == "fanout_4":
        return 4
    elif variant == "fanout_6":
        return 6
    return N_TILES


@always_inline
def is_valid_variant(variant: String) -> Bool:
    return (
        variant == "baseline" or variant == "candidate"
        or variant == "prefetch_t0" or variant == "prefetch_nta"
        or is_fanout_variant(variant)
        or variant == "nblock32" or variant == "nblock64"
        or variant == "loadfirst" or variant == "pipe2"
        or variant == "layout_zip" or is_ksplit_variant(variant))


@fieldwise_init
struct RankBuffers(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var act_scale: F32Ptr
    var w1_bank: I8Ptr
    var w3_bank: I8Ptr
    var w13_bank: I8Ptr
    var ksplit_partials: I32Ptr
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
        self.w13_bank = I8Ptr()
        self.ksplit_partials = I32Ptr()
        self.w1_sc_bank = F32Ptr()
        self.w1_cs_bank = F32Ptr()
        self.w3_sc_bank = F32Ptr()
        self.w3_cs_bank = F32Ptr()
        self.qi_out_base = I8Ptr()
        self.qi_out_cand = I8Ptr()
        self.blk_sc_base = F32Ptr()
        self.blk_sc_cand = F32Ptr()


# =============================================================================
# Deterministic fills + rank setup
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
    bank_experts: Int,
    use_layout_zip: Bool,
    use_ksplit: Bool,
) -> RankBuffers:
    """Bump-alloc rank-local buffers sized for `bank_experts`.

    Full BANK_MAX would cost ~580 MB/rank; sizing to bank_experts keeps
    HOT/MIXED setup cheap (~10-40 MB) while STREAM still gets its full
    576 MB footprint.
    """
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var r = RankBuffers()
    r.act = arena.alloc[Scalar[DType.int8]](K_DIM)
    r.act_scale = arena.alloc[Float32](1)

    r.w1_bank = arena.alloc[Scalar[DType.int8]](bank_experts * weight_elems_per_expert)
    r.w3_bank = arena.alloc[Scalar[DType.int8]](bank_experts * weight_elems_per_expert)
    if use_layout_zip:
        r.w13_bank = arena.alloc[Scalar[DType.int8]](
            bank_experts * 2 * weight_elems_per_expert)
    if use_ksplit:
        r.ksplit_partials = arena.alloc[Int32](KSPLIT_PARTIAL_I32)
    r.w1_sc_bank = arena.alloc[Float32](bank_experts * sc_count_per_expert)
    r.w1_cs_bank = arena.alloc[Float32](bank_experts * sc_count_per_expert)
    r.w3_sc_bank = arena.alloc[Float32](bank_experts * sc_count_per_expert)
    r.w3_cs_bank = arena.alloc[Float32](bank_experts * sc_count_per_expert)

    r.qi_out_base = arena.alloc[Scalar[DType.int8]](bank_experts * INTERMEDIATE)
    r.qi_out_cand = arena.alloc[Scalar[DType.int8]](bank_experts * INTERMEDIATE)
    r.blk_sc_base = arena.alloc[Float32](bank_experts * N_TILES)
    r.blk_sc_cand = arena.alloc[Float32](bank_experts * N_TILES)
    return r^


@no_inline
def fill_w13_layout(bufs: RankBuffers, bank_experts: Int):
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime width = simd_width_of[DType.int32]()
    comptime chunk = width * VNNI_BLK

    for e in range(bank_experts):
        var w1 = bufs.w1_bank + e * weight_elems_per_expert
        var w3 = bufs.w3_bank + e * weight_elems_per_expert
        var w13 = bufs.w13_bank + e * 2 * weight_elems_per_expert
        for off in range(0, weight_elems_per_expert, chunk):
            memcpy(
                dest=(w13 + 2 * off).bitcast[UInt8](),
                src=(w1 + off).bitcast[UInt8](),
                count=chunk)
            memcpy(
                dest=(w13 + 2 * off + chunk).bitcast[UInt8](),
                src=(w3 + off).bitcast[UInt8](),
                count=chunk)


def fill_rank_data(
    bufs: RankBuffers,
    bank_experts: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    rank_seed: UInt64,
    use_layout_zip: Bool,
):
    """Pack expert 0 fully; replicate bytes + scales to experts 1..bank-1.

    Each slot still sits on its own physical page, so DRAM access pattern
    is realistic. Dropping the per-expert random fill + pack cost saves
    ~800 ms of setup for STREAM (64 experts) at the cost of identical
    weight bytes across slots — which is fine for a DRAM/cache microbench.
    """
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    fill_i8(bufs.act, K_DIM, rank_seed ^ 0xA1)
    bufs.act_scale[0] = Float32(0.05)

    var w1_0 = bufs.w1_bank
    var w3_0 = bufs.w3_bank
    var seed_e = rank_seed ^ 0x9E3779B97F4A7C15
    fill_i8(w1_0, weight_elems_per_expert, seed_e)
    fill_i8(w3_0, weight_elems_per_expert, seed_e ^ 0x5555555555555555)
    fill_f32_positive(bufs.w1_sc_bank, sc_count_per_expert, seed_e ^ 0xAA, Float32(0.01))
    fill_f32_positive(bufs.w3_sc_bank, sc_count_per_expert, seed_e ^ 0xBB, Float32(0.01))

    pack_and_colsum_vnni(
        w1_0.bitcast[UInt8](), w1_0.bitcast[UInt8](), scratch,
        INTERMEDIATE, K_DIM, K_DIM,
        bufs.w1_cs_bank, True)
    pack_and_colsum_vnni(
        w3_0.bitcast[UInt8](), w3_0.bitcast[UInt8](), scratch,
        INTERMEDIATE, K_DIM, K_DIM,
        bufs.w3_cs_bank, True)

    if bank_experts <= 1:
        if use_layout_zip:
            fill_w13_layout(bufs, bank_experts)
        return

    var weight_bytes = weight_elems_per_expert
    var sc_bytes = sc_count_per_expert * 4
    for e in range(1, bank_experts):
        memcpy(
            dest=(bufs.w1_bank + e * weight_elems_per_expert).bitcast[UInt8](),
            src=w1_0.bitcast[UInt8](),
            count=weight_bytes)
        memcpy(
            dest=(bufs.w3_bank + e * weight_elems_per_expert).bitcast[UInt8](),
            src=w3_0.bitcast[UInt8](),
            count=weight_bytes)
        memcpy(
            dest=(bufs.w1_sc_bank + e * sc_count_per_expert).bitcast[UInt8](),
            src=bufs.w1_sc_bank.bitcast[UInt8](),
            count=sc_bytes)
        memcpy(
            dest=(bufs.w1_cs_bank + e * sc_count_per_expert).bitcast[UInt8](),
            src=bufs.w1_cs_bank.bitcast[UInt8](),
            count=sc_bytes)
        memcpy(
            dest=(bufs.w3_sc_bank + e * sc_count_per_expert).bitcast[UInt8](),
            src=bufs.w3_sc_bank.bitcast[UInt8](),
            count=sc_bytes)
        memcpy(
            dest=(bufs.w3_cs_bank + e * sc_count_per_expert).bitcast[UInt8](),
            src=bufs.w3_cs_bank.bitcast[UInt8](),
            count=sc_bytes)

    if use_layout_zip:
        fill_w13_layout(bufs, bank_experts)


# =============================================================================
# Job construction (matches production minimax_moe_phase1 pre-offsetting)
# =============================================================================


@always_inline
def fill_jobs(
    mut jobs: InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
):
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
        jobs[t] = FusedW1W3SiluArgs(
            bufs.act, bufs.act_scale,
            w1, w1_sc, w1_cs,
            w3, w3_sc, w3_cs,
            qi_out + n_start, blk_sc + t,
            n_start, FWHT_BLK, 1)


@always_inline
def fill_jobs_grouped(
    mut jobs: InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
    jobs_per_expert: Int,
) -> Int:
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var w1 = bufs.w1_bank + expert_idx * weight_elems_per_expert
    var w3 = bufs.w3_bank + expert_idx * weight_elems_per_expert
    var w1_sc = bufs.w1_sc_bank + expert_idx * sc_count_per_expert
    var w1_cs = bufs.w1_cs_bank + expert_idx * sc_count_per_expert
    var w3_sc = bufs.w3_sc_bank + expert_idx * sc_count_per_expert
    var w3_cs = bufs.w3_cs_bank + expert_idx * sc_count_per_expert

    var groups = jobs_per_expert
    if groups < 1:
        groups = 1
    elif groups > N_TILES:
        groups = N_TILES
    var tiles_per_job = (N_TILES + groups - 1) // groups
    var count = 0
    for g in range(groups):
        var tile_start = g * tiles_per_job
        if tile_start >= N_TILES:
            break
        var tile_end = min(tile_start + tiles_per_job, N_TILES)
        var n_start = tile_start * FWHT_BLK
        var n_count = (tile_end - tile_start) * FWHT_BLK
        jobs[count] = FusedW1W3SiluArgs(
            bufs.act, bufs.act_scale,
            w1, w1_sc, w1_cs,
            w3, w3_sc, w3_cs,
            qi_out + n_start, blk_sc + tile_start,
            n_start, n_count, 1)
        count += 1
    return count


@always_inline
def fill_jobs_layout_zip(
    mut jobs: InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
) -> Int:
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var w13 = bufs.w13_bank + expert_idx * 2 * weight_elems_per_expert
    var w1_sc = bufs.w1_sc_bank + expert_idx * sc_count_per_expert
    var w1_cs = bufs.w1_cs_bank + expert_idx * sc_count_per_expert
    var w3_sc = bufs.w3_sc_bank + expert_idx * sc_count_per_expert
    var w3_cs = bufs.w3_cs_bank + expert_idx * sc_count_per_expert

    for t in range(N_TILES):
        var n_start = t * FWHT_BLK
        jobs[t] = FusedW1W3SiluArgs(
            bufs.act, bufs.act_scale,
            w13, w1_sc, w1_cs,
            I8Ptr(), w3_sc, w3_cs,
            qi_out + n_start, blk_sc + t,
            n_start, FWHT_BLK, 1)
    return N_TILES


@always_inline
def fill_jobs_for_variant(
    mut jobs: InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
    variant: String,
) -> Int:
    if variant == "layout_zip":
        return fill_jobs_layout_zip(jobs, bufs, expert_idx, qi_out, blk_sc)
    if is_fanout_variant(variant):
        return fill_jobs_grouped(
            jobs, bufs, expert_idx, qi_out, blk_sc,
            fanout_jobs_for_variant(variant))
    fill_jobs(jobs, bufs, expert_idx, qi_out, blk_sc)
    return N_TILES


@always_inline
def dispatch_variant[P: BurstThreadPool, origin: MutOrigin, //](
    mut pool: P,
    variant: String,
    jobs: UnsafePointer[FusedW1W3SiluArgs, origin],
    job_count: Int,
) -> Bool:
    if variant == "baseline" or is_fanout_variant(variant):
        pool.dispatch[FusedW1W3SiluArgs,
            fused_w1_w3_silu_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
            jobs, job_count)
    elif variant == "candidate":
        pool.dispatch[FusedW1W3SiluArgs,
            candidate_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
            jobs, job_count)
    elif variant == "prefetch_t0":
        pool.dispatch[FusedW1W3SiluArgs,
            prefetch_worker[INTERMEDIATE, K_DIM, FWHT_BLK,
                PF_T0, PREFETCH_AHEAD]](
            jobs, job_count)
    elif variant == "prefetch_nta":
        pool.dispatch[FusedW1W3SiluArgs,
            prefetch_worker[INTERMEDIATE, K_DIM, FWHT_BLK,
                PF_NTA, PREFETCH_AHEAD]](
            jobs, job_count)
    elif variant == "nblock32":
        pool.dispatch[FusedW1W3SiluArgs,
            nblock_worker[INTERMEDIATE, K_DIM, FWHT_BLK, 32]](
            jobs, job_count)
    elif variant == "nblock64":
        pool.dispatch[FusedW1W3SiluArgs,
            nblock_worker[INTERMEDIATE, K_DIM, FWHT_BLK, 64]](
            jobs, job_count)
    elif variant == "loadfirst":
        pool.dispatch[FusedW1W3SiluArgs,
            load_sched_worker[INTERMEDIATE, K_DIM, FWHT_BLK, False]](
            jobs, job_count)
    elif variant == "pipe2":
        pool.dispatch[FusedW1W3SiluArgs,
            load_sched_worker[INTERMEDIATE, K_DIM, FWHT_BLK, True]](
            jobs, job_count)
    elif variant == "layout_zip":
        pool.dispatch[FusedW1W3SiluArgs,
            layout_zip_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
            jobs, job_count)
    else:
        print("unknown variant:", variant)
        return False
    return True


@always_inline
def fill_ksplit_jobs(
    mut partial_jobs: InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY],
    mut final_jobs: InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY],
    bufs: RankBuffers,
    expert_idx: Int,
    qi_out: I8Ptr,
    blk_sc: F32Ptr,
    split_count: Int,
) -> Int:
    comptime weight_elems_per_expert = INTERMEDIATE * K_DIM
    comptime sc_count_per_expert = INTERMEDIATE

    var w1 = bufs.w1_bank + expert_idx * weight_elems_per_expert
    var w3 = bufs.w3_bank + expert_idx * weight_elems_per_expert
    var w1_sc = bufs.w1_sc_bank + expert_idx * sc_count_per_expert
    var w1_cs = bufs.w1_cs_bank + expert_idx * sc_count_per_expert
    var w3_sc = bufs.w3_sc_bank + expert_idx * sc_count_per_expert
    var w3_cs = bufs.w3_cs_bank + expert_idx * sc_count_per_expert
    var k_count = K_DIM // split_count

    var p = 0
    for t in range(N_TILES):
        var n_start = t * FWHT_BLK
        var tile_partials = bufs.ksplit_partials + t * split_count * 2 * FWHT_BLK
        for s in range(split_count):
            var part_base = tile_partials + s * 2 * FWHT_BLK
            partial_jobs[p] = KSplitPartialArgs(
                bufs.act,
                w1,
                w3,
                part_base,
                part_base + FWHT_BLK,
                n_start,
                s * k_count,
                k_count)
            p += 1

        final_jobs[t] = KSplitFinalArgs(
            bufs.act_scale,
            w1_sc,
            w1_cs,
            w3_sc,
            w3_cs,
            tile_partials,
            qi_out + n_start,
            blk_sc + t,
            n_start,
            split_count)

    return p


@always_inline
def dispatch_ksplit[P: BurstThreadPool, //](
    mut pool: P,
    mut partial_jobs: InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY],
    partial_count: Int,
    mut final_jobs: InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY],
):
    var offset = 0
    var capacity = pool.get_capacity()
    while offset < partial_count:
        var count = min(capacity, partial_count - offset)
        pool.dispatch[KSplitPartialArgs,
            ksplit_partial_worker[FWHT_BLK, K_DIM]](
            UnsafePointer(to=partial_jobs[offset]), count)
        pool.join()
        offset += count

    pool.dispatch[KSplitFinalArgs, ksplit_final_worker[FWHT_BLK]](
        UnsafePointer(to=final_jobs[0]), N_TILES)
    pool.join()


# =============================================================================
# Correctness gate — variant vs baseline on identical inputs
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


def correctness_check_variant[P: BurstThreadPool, //](
    bufs: RankBuffers, mut pool: P, variant: String,
) -> Bool:
    var jobs = InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs())

    # Baseline reference
    fill_jobs(jobs, bufs, 0, bufs.qi_out_base, bufs.blk_sc_base)
    pool.dispatch[FusedW1W3SiluArgs,
        fused_w1_w3_silu_worker[INTERMEDIATE, K_DIM, FWHT_BLK]](
        UnsafePointer(to=jobs[0]), N_TILES)
    pool.join()

    # Selected variant -> candidate buffers
    if is_ksplit_variant(variant):
        var partial_jobs = InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY](
            fill=KSplitPartialArgs())
        var final_jobs = InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY](
            fill=KSplitFinalArgs())
        var split_count = ksplit_for_variant(variant)
        var partial_count = fill_ksplit_jobs(
            partial_jobs, final_jobs, bufs, 0,
            bufs.qi_out_cand, bufs.blk_sc_cand, split_count)
        dispatch_ksplit(pool, partial_jobs, partial_count, final_jobs)
    else:
        var job_count = fill_jobs_for_variant(
            jobs, bufs, 0, bufs.qi_out_cand, bufs.blk_sc_cand, variant)
        if not dispatch_variant(pool, variant, UnsafePointer(to=jobs[0]), job_count):
            return False
        pool.join()

    var qi_mismatch = compare_i8(bufs.qi_out_base, bufs.qi_out_cand, INTERMEDIATE)
    if qi_mismatch >= 0:
        print("correctness: qi_out mismatch at", qi_mismatch,
            "baseline=", Int(bufs.qi_out_base[qi_mismatch]),
            "variant=", Int(bufs.qi_out_cand[qi_mismatch]))
        return False
    var sc_mismatch = compare_f32(bufs.blk_sc_base, bufs.blk_sc_cand, N_TILES)
    if sc_mismatch >= 0:
        print("correctness: blk_scale mismatch at", sc_mismatch)
        return False
    return True


# =============================================================================
# Timed run — one variant for the current regime
# =============================================================================


@always_inline
def dispatch_ksplit_tp[P: BurstThreadPool, tp: Int, //](
    mut pools: HeapMoveArray[P],
    mut partial_jobs: InlineArray[
        InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY], tp],
    partial_count: Int,
    mut final_jobs: InlineArray[
        InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY], tp],
):
    var offset = 0
    var chunk = pools[0].get_capacity()
    while offset < partial_count:
        var count = min(chunk, partial_count - offset)
        for r in range(tp):
            pools[r].dispatch[KSplitPartialArgs,
                ksplit_partial_worker[FWHT_BLK, K_DIM]](
                UnsafePointer(to=partial_jobs[r][offset]), count)
        for r in range(tp):
            pools[r].join()
        offset += count

    for r in range(tp):
        pools[r].dispatch[KSplitFinalArgs, ksplit_final_worker[FWHT_BLK]](
            UnsafePointer(to=final_jobs[r][0]), N_TILES)
    for r in range(tp):
        pools[r].join()


@no_inline
def time_run_ksplit[P: BurstThreadPool, tp: Int, //](
    all_bufs: InlineArray[RankBuffers, tp],
    mut pools: HeapMoveArray[P],
    bank_experts: Int,
    split_count: Int,
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    var partial_jobs = InlineArray[
        InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY], tp
    ](fill=InlineArray[KSplitPartialArgs, MAX_POOL_CAPACITY](
        fill=KSplitPartialArgs()))
    var final_jobs = InlineArray[
        InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY], tp
    ](fill=InlineArray[KSplitFinalArgs, MAX_POOL_CAPACITY](
        fill=KSplitFinalArgs()))
    var partial_count = N_TILES * split_count

    for w in range(WARMUP):
        var e = w % bank_experts
        for r in range(tp):
            _ = fill_ksplit_jobs(
                partial_jobs[r], final_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base,
                split_count)
        dispatch_ksplit_tp(pools, partial_jobs, partial_count, final_jobs)

    for i in range(ITERS):
        var e = i % bank_experts
        for r in range(tp):
            _ = fill_ksplit_jobs(
                partial_jobs[r], final_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base,
                split_count)
        var t0 = perf_counter_ns()
        dispatch_ksplit_tp(pools, partial_jobs, partial_count, final_jobs)
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


@no_inline
def time_run[P: BurstThreadPool, tp: Int, //](
    all_bufs: InlineArray[RankBuffers, tp],
    mut pools: HeapMoveArray[P],
    bank_experts: Int,
    variant: String,
    samples: UnsafePointer[Int, MutAnyOrigin],
):
    if is_ksplit_variant(variant):
        time_run_ksplit(
            all_bufs, pools, bank_experts,
            ksplit_for_variant(variant), samples)
        return

    var all_jobs = InlineArray[
        InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY], tp
    ](fill=InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs()))
    var job_counts = InlineArray[Int, tp](fill=0)

    for w in range(WARMUP):
        var e = w % bank_experts
        for r in range(tp):
            job_counts[r] = fill_jobs_for_variant(
                all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base,
                variant)
        for r in range(tp):
            _ = dispatch_variant(
                pools[r], variant,
                UnsafePointer(to=all_jobs[r][0]), job_counts[r])
        for r in range(tp):
            pools[r].join()

    for i in range(ITERS):
        var e = i % bank_experts
        for r in range(tp):
            job_counts[r] = fill_jobs_for_variant(
                all_jobs[r], all_bufs[r], e,
                all_bufs[r].qi_out_base, all_bufs[r].blk_sc_base,
                variant)
        var t0 = perf_counter_ns()
        for r in range(tp):
            _ = dispatch_variant(
                pools[r], variant,
                UnsafePointer(to=all_jobs[r][0]), job_counts[r])
        for r in range(tp):
            pools[r].join()
        var t1 = perf_counter_ns()
        samples[i] = Int(t1 - t0)


# =============================================================================
# Reporting — machine-parseable
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
def print_result(
    regime: String, variant: String, bank_experts: Int,
    p: UnsafePointer[Int, MutAnyOrigin], count: Int,
):
    sort_in_place(p, count)
    var sum = Int(0)
    for i in range(count):
        sum += p[i]
    var mean = sum // count
    print(
        "RESULT regime=", regime,
        " variant=", variant,
        " bank=", bank_experts,
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


def run_one[P: BurstThreadPool, //, tp: Int](
    regime: String, variant: String, bank_experts: Int,
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
):
    set_subnormal_zeroing()

    var weight_bytes_per_expert = (
        2 * INTERMEDIATE * K_DIM + 4 * INTERMEDIATE * 4)
    var arena_bytes = (
        bank_experts * weight_bytes_per_expert
        + 2 * bank_experts * INTERMEDIATE
        + 2 * bank_experts * N_TILES * 4
        + 4 * 1024 * 1024
    )
    var use_layout_zip = variant == "layout_zip"
    var use_ksplit = is_ksplit_variant(variant)
    if use_layout_zip:
        arena_bytes += bank_experts * 2 * INTERMEDIATE * K_DIM
    if use_ksplit:
        arena_bytes += KSPLIT_PARTIAL_I32 * 4
    comptime scratch_bytes = 64 * K_DIM

    var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](tp)
    for rank in range(tp):
        var node = numa_topo[rank]
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](node, arena_bytes)
        if not arena:
            print("arena alloc failed")
            return
        arenas.push(arena^)

    var scratch = alloc[UInt8](scratch_bytes)
    var all_bufs = InlineArray[RankBuffers, tp](fill=RankBuffers())
    for rank in range(tp):
        all_bufs[rank] = alloc_rank(
            arenas[rank], bank_experts, use_layout_zip, use_ksplit)
        fill_rank_data(all_bufs[rank], bank_experts, scratch,
            UInt64(rank) * 0xDEADBEEFCAFEBABE, use_layout_zip)
    scratch.free()

    for rank in range(tp):
        _ = arenas[rank].prefault()

    if not correctness_check_variant(all_bufs[0], pools[0], variant):
        print("CORRECTNESS FAILED -- aborting", regime, variant)
        _ = arenas^
        return

    var samples = alloc[Int](ITERS)
    time_run(all_bufs, pools, bank_experts, variant, samples)
    print_result(regime, variant, bank_experts, samples, ITERS)
    samples.free()

    _ = arenas^


def parse_bank(regime: String) -> Int:
    if regime == "hot":
        return BANK_HOT
    elif regime == "mixed":
        return BANK_MIXED
    elif regime == "stream":
        return BANK_STREAM
    return -1


def dispatch_pools[P: BurstThreadPool, //, tp: Int](
    regime: String, variant: String, bank_experts: Int,
    numa: NumaInfo, numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
):
    run_one[tp](
        regime, variant, bank_experts,
        numa, numa_topo, pools^)


def main():
    var args = argv()
    if len(args) < 3:
        print("usage:", args[0],
            "<regime:hot|mixed|stream> <variant:baseline|candidate|prefetch_t0|prefetch_nta|fanout_1|fanout_2|fanout_3|fanout_4|fanout_6|nblock32|nblock64|loadfirst|pipe2|layout_zip|ksplit2|ksplit4>")
        return

    var regime = String(args[1])
    var variant = String(args[2])
    var bank_experts = parse_bank(regime)
    if bank_experts < 0:
        print("unknown regime:", regime)
        return
    if not is_valid_variant(variant):
        print("unknown variant:", variant)
        return

    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)

    if numa.has_isolation():
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        dispatch_pools[TP](regime, variant, bank_experts, numa, numa_topo, pools^)
    else:
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        dispatch_pools[TP](regime, variant, bank_experts, numa, numa_topo, pools^)
