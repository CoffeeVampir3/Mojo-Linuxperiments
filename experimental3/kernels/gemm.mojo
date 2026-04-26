"""GEMM-shaped compute cores — args structs and pool-worker bodies.

Dispatchers for these workers live in dispatch_kernels.mojo. The workers here
take an Args struct from the pool and run one unit of compute (one row-range
of a seq_len × N × K problem, or one N-tile of a decode problem).

When seq_len == 1 the dispatch collapses to a single decode GEMV. When
seq_len > 1 (prompt prefill) the dispatch partitions rows across pool
workers, and each worker walks the per-row gemv.mojo kernels. That inner
per-row walk is the AMX-lift surface: replacing gemv_row / blocked GEMV
with tile-based AMX kernels here will give prefill throughput without touching
decode.

Also hosts the lm_head worker (decode N-parallel scan into bf16 logits).
Structurally M=1 always; workers N-parallelize the same dot across pool.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP
from kernels.moe import fused_gateup_quant_row
from experimental3.kernels.dot_prod import vpdpbusd
from experimental3.kernels.gemv import (
    gemv_row, gemv_row_blocked_bf16_scaled, gemv_row_blocked_wa,
    lm_head_row_dot,
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
    """Blocked GEMV worker that writes final scaled bf16 directly."""
    comptime num_blocks = K // fwht_blk
    for m in range(args.row_count):
        gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
            args.act + m * K, args.wpacked, args.blk_scale + m * num_blocks,
            args.wscale, args.blk_colsum, args.dst + m * N,
            args.output_scale, args.n_out, args.colsum_stride)


def int8_gemv_blocked_decode_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """N-split decode worker that writes final scaled bf16 directly."""
    gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
        args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, args.dst,
        args.output_scale, args.n_out, args.colsum_stride)


def int8_gemv_blocked_wa_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = K // fwht_blk
    for m in range(args.row_count):
        var dst_buf = InlineArray[Float32, N](fill=Float32(0))
        var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
        gemv_row_blocked_wa[N, K, fwht_blk](
            args.act + m * K, args.wpacked, args.blk_scale + m * num_blocks,
            args.wscale, args.blk_colsum, dp)

        var scale = SIMD[DType.float32, width](args.output_scale)
        var k = 0
        while k + width <= N:
            (args.dst + m * N + k).store(
                ((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
            k += width


# ============================================================================
# fused_gu_gelu_tanh worker
# ============================================================================


def fused_gu_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int,
    fwht: Bool = True](
    args: FusedGuGeluTanhArgs,
):
    debug_assert(intermediate % fwht_blk == 0,
        "fused_gu_gelu_tanh: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "fused_gu_gelu_tanh: K must be a multiple of 64 (VNNI_K_STEP)")
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_dequant = args.act_scale[m] / Float32(127)
        fused_gateup_quant_row[fwht_blk, K, fwht, gelu_tanh_f32](
            args.act_i8 + m * K,
            args.wpacked, args.wscale, args.wcolsum,
            args.wpacked + intermediate * K,
            args.wscale + intermediate, args.wcolsum + intermediate,
            act_dequant,
            args.qi_out + m * intermediate,
            args.blk_scale + m * num_blk_per_row,
            args.n_start, args.n_count)


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
