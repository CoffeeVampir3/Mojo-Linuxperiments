"""GEMM-shaped compute cores — args structs and pool-worker bodies.

Dispatchers for these workers live in dispatch_kernels.mojo. The workers here
take an Args struct from the pool and run one unit of compute (one row-range
of a seq_len × N × K problem, or one N-tile of a decode problem).

When seq_len == 1 the dispatch collapses to a single decode GEMV. When
seq_len > 1 (prompt prefill) the dispatch partitions rows across pool
workers, and each worker walks the per-row gemv.mojo kernels. That inner
per-row walk is the AMX-lift surface: replacing gemv_row / gemv_row_blocked
with tile-based AMX kernels here will give prefill throughput without
touching decode.

Also hosts the lm_head worker (decode N-parallel scan into bf16 logits).
Structurally M=1 always; workers N-parallelize the same dot across pool.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import vpdpbusd
from experimental3.kernels.gemv import (
    gemv_row, gemv_row_blocked, gemv_row_blocked_wa, lm_head_row_dot,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kernels.dispatch_args import (
    WorkerConfig, Int8GemvBlockedArgs, FusedGuGeluTanhArgs, LmHeadArgs,
)


# ============================================================================
# int8_gemv worker
# ============================================================================


def int8_gemv_worker[N: Int, K: Int](cfg: WorkerConfig):
    """M-split prefill worker: each worker handles a range of seq_len rows."""
    var act = cfg.act_ptr
    var wpacked = cfg.wpacked_ptr
    var colsum = cfg.colsum_ptr
    var wscale = cfg.weight_scale_ptr
    var dst = cfg.dst_ptr
    var act_scales = cfg.act_scale_ptr
    var start = cfg.start

    for m in range(cfg.count):
        var act_dequant = act_scales[start + m] / Float32(127)
        gemv_row[N, K, DType.bfloat16](
            act + m * K, wpacked, act_dequant, wscale, colsum, dst + m * N,
        )


def int8_gemv_decode_worker[N: Int, K: Int](cfg: WorkerConfig):
    """N-split decode worker: each worker handles a range of output elements.

    Pointers are pre-offset by the dispatcher: wpacked to n_start * K,
    colsum/weight_scale/dst to n_start. count holds the sub-N for this worker.
    """
    var act_dequant = cfg.act_scale_ptr[0] / Float32(127)
    gemv_row[N, K, DType.bfloat16](
        cfg.act_ptr, cfg.wpacked_ptr, act_dequant,
        cfg.weight_scale_ptr, cfg.colsum_ptr, cfg.dst_ptr, cfg.count,
    )


# ============================================================================
# int8_gemv_blocked workers (standard + _wa variant)
# ============================================================================


def int8_gemv_blocked_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """M-split prefill: full-N GEMV per row. Output bf16[N] * output_scale."""
    comptime width = simd_width_of[DType.float32]()

    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= N:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked_decode_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """N-split decode: each worker handles a sub-range of output elements.

    Pointers are pre-offset by the dispatcher. n_out holds the sub-N count.
    blk_colsum is pre-offset to n_start but the stride between blocks is
    still full N, passed via colsum_stride.
    """
    comptime width = simd_width_of[DType.float32]()

    var sub_n = args.n_out
    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp, sub_n, args.colsum_stride)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= sub_n:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked_wa_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    comptime width = simd_width_of[DType.float32]()
    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked_wa[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= N:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


# ============================================================================
# fused_gu_gelu_tanh workers (standard + _wa variant)
# ============================================================================


def fused_gu_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int,
    fwht: Bool = True](
    args: FusedGuGeluTanhArgs,
):
    """N-tiled gate+up GEMV -> GELU-tanh -> [FWHT] -> per-block i8.

    Processes row_count activation rows, each over N-range [n_start, n_start + n_count).
    Tiles the N-range in fwht_blk-sized chunks. Each tile does:
      gate GEMV[fwht_blk, K] + up GEMV[fwht_blk, K] -> gelu_tanh -> [FWHT] -> i8.
    When fwht=False, skips the FWHT rotation (channelwise quantization).
    Activation must be pre-quantized by caller.
    Fused weight is [2*intermediate, K]: gate = [0:intermediate], up = [intermediate:].
    """
    debug_assert(intermediate % fwht_blk == 0,
        "fused_gu_gelu_tanh: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "fused_gu_gelu_tanh: K must be a multiple of 64 (VNNI_K_STEP)")
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
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.wpacked + n_off * K,
                dequant,
                args.wscale + n_off,
                args.wcolsum + n_off,
                gate.bitcast[Scalar[DType.float32]]())

            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.wpacked + (intermediate + n_off) * K,
                dequant,
                args.wscale + intermediate + n_off,
                args.wcolsum + intermediate + n_off,
                up.bitcast[Scalar[DType.float32]]())

            var k = 0
            while k + width <= fwht_blk:
                var g = (gate + k).load[width=width]()
                var u = (up + k).load[width=width]()
                (gate + k).store(gelu_tanh_f32[width](g) * u)
                k += width

            comptime if fwht:
                fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)

            local_n += fwht_blk


# GEMV_TILE is shared with dispatch_kernels.mojo — the _wa worker uses the
# VNNI_N_STEP as its GEMV tile size while the outer dispatch iterates sub-blocks.
comptime GEMV_TILE = VNNI_N_STEP


def fused_gu_gelu_tanh_worker_wa[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    """Workaround: decouples GEMV tile (VNNI_N_STEP) from FWHT block (fwht_blk).

    Tiles the gate/up GEMV at GEMV_TILE for VNNI compatibility, then applies
    FWHT and quantize in fwht_blk-sized sub-blocks within each tile.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "intermediate must be a multiple of fwht_blk")
    debug_assert(GEMV_TILE % fwht_blk == 0,
        "GEMV_TILE must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0, "K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime sub_blocks_per_tile = GEMV_TILE // fwht_blk
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + n_off * K, dequant,
                args.wscale + n_off, args.wcolsum + n_off,
                gate.bitcast[Scalar[DType.float32]]())

            var up_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + (intermediate + n_off) * K, dequant,
                args.wscale + intermediate + n_off,
                args.wcolsum + intermediate + n_off,
                up.bitcast[Scalar[DType.float32]]())

            var k = 0
            while k + width <= GEMV_TILE:
                var g = (gate + k).load[width=width]()
                var u = (up + k).load[width=width]()
                (gate + k).store(gelu_tanh_f32[width](g) * u)
                k += width

            for sb in range(sub_blocks_per_tile):
                var sb_off = sb * fwht_blk
                fwht_block[fwht_blk](gate + sb_off)
                blk_row[(local_n + sb_off) // fwht_blk] = absmax_quantize_i8[fwht_blk](
                    gate + sb_off, qi_row + local_n + sb_off)

            local_n += GEMV_TILE


# ============================================================================
# lm_head worker — decode-only N-parallel scan
# ============================================================================


def lm_head_worker[K: Int, fwht_blk: Int](args: LmHeadArgs):
    comptime num_blocks = K // fwht_blk
    var act = args.act
    var weight = args.weight
    var act_blk_scales = args.act_blk_scales
    var w_blk_scales = args.w_blk_scales
    var w_blk_colsums = args.w_blk_colsums
    var dst = args.dst

    for n in range(args.n_start, args.n_start + args.n_count):
        var weight_row = weight + n * K
        var w_blk_scales_row = w_blk_scales + n * num_blocks
        var w_blk_colsums_row = w_blk_colsums + n * num_blocks
        var dot_f32 = lm_head_row_dot[K, fwht_blk](
            act, weight_row,
            act_blk_scales,
            w_blk_scales_row,
            w_blk_colsums_row)
        dst[n] = dot_f32.cast[DType.bfloat16]()

