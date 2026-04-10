"""Dense MLP dispatch kernels for Gemma 4 forward pass.

Phase 1: Fused gate_up GEMV + GELU-tanh + FWHT + per-block quantize.
Phase 2: Int8 GEMV with per-block activation scales for down projection.
Router:  Softmax + top-k dispatch.

All are BurstPool-dispatched — body threads never compute.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from kernels.kernel_ops import PoolFence
from experimental2.kernels.int8_gemv import gemv_row
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.moe import gemv_row_blocked
from experimental_gemma.router import softmax_topk_renorm, Gemma4TopKResult

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# Phase 1: Fused gate_up GEMV + GELU-tanh + FWHT + per-block quantize
# ============================================================================


@fieldwise_init
struct FusedGuGeluTanhArgs(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var wpacked: U8Ptr
    var wscale: F32Ptr
    var wcolsum: F32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var n_count: Int
    var row_count: Int


def fused_gu_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    """N-tiled gate+up GEMV -> GELU-tanh -> FWHT -> per-block i8.

    Processes row_count activation rows, each over N-range [n_start, n_start + n_count).
    Tiles the N-range in fwht_blk-sized chunks. Each tile does:
      gate GEMV[fwht_blk, K] + up GEMV[fwht_blk, K] -> gelu_tanh -> FWHT -> i8.
    Activation must be pre-quantized by caller.
    Fused weight is [2*intermediate, K]: gate = [0:intermediate], up = [intermediate:].
    """
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

            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)

            local_n += fwht_blk


def fused_gu_gelu_tanh[intermediate: Int, K: Int, fwht_blk: Int,
                       P: BurstThreadPool](
    act_i8: I8Ptr,
    act_scale: F32Ptr,
    wpacked: U8Ptr,
    wscale: F32Ptr,
    wcolsum: F32Ptr,
    qi_out: I8Ptr,
    blk_scale: F32Ptr,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch gate_up GEMV + GELU-tanh + FWHT + per-block quantize.

    Activation must be pre-quantized by caller (rmsnorm_gamma_fwht_quantize).
    For seq_len=1 (decode): parallelizes across the output dimension N.
    For seq_len>1 (prompt): parallelizes across sequence rows.
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime n_tiles = intermediate // fwht_blk
    comptime num_blk_per_row = intermediate // fwht_blk
    comptime MAX_POOL_CAPACITY = 128

    var zero_args = FusedGuGeluTanhArgs(
        I8Ptr(), F32Ptr(), U8Ptr(), F32Ptr(), F32Ptr(),
        I8Ptr(), F32Ptr(), 0, 0, 0)
    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](fill=zero_args)

    if seq_len == 1:
        var num_workers = min(n_tiles, pool.get_capacity())
        var tiles_per_worker = (n_tiles + num_workers - 1) // num_workers
        for i in range(num_workers):
            var tile_start = i * tiles_per_worker
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * fwht_blk
            var n_count = (tile_end - tile_start) * fwht_blk
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8, act_scale,
                wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_scale + tile_start,
                n_start, n_count, 1)
        pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)
    else:
        var num_workers = min(seq_len, pool.get_capacity())
        var rows_per_worker = (seq_len + num_workers - 1) // num_workers
        for i in range(num_workers):
            var row_start = i * rows_per_worker
            var row_count = min(rows_per_worker, seq_len - row_start)
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8 + row_start * K,
                act_scale + row_start,
                wpacked, wscale, wcolsum,
                qi_out + row_start * intermediate,
                blk_scale + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)

    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Phase 2: Int8 GEMV with per-block activation scales (down projection)
# ============================================================================


@fieldwise_init
struct Int8GemvBlockedArgs(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var wpacked: U8Ptr
    var blk_scale: F32Ptr
    var wscale: F32Ptr
    var blk_colsum: F32Ptr
    var dst: BF16Ptr


def int8_gemv_blocked_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """Down GEMV with per-block activation scales. Output bf16[N]."""
    comptime width = simd_width_of[DType.float32]()

    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var k = 0
    while k + width <= N:
        (args.dst + k).store((dp + k).load[width=width]().cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: I8Ptr,
    wpacked: U8Ptr,
    blk_scale: F32Ptr,
    wscale: F32Ptr,
    blk_colsum: F32Ptr,
    dst: BF16Ptr,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch int8 GEMV with per-block activation scales.

    act:        i8 [seq_len, K]
    wpacked:    VNNI [N, K]
    blk_scale:  f32 [seq_len, K/fwht_blk] per-block activation scales
    wscale:     f32 [N] weight scales
    blk_colsum: f32 [N, K/fwht_blk] per-block column sums
    dst:        bf16 [seq_len, N] output
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime num_blocks = K // fwht_blk
    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr())
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)
    for i in range(num_jobs):
        var start = i * rows_per_job
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K,
            wpacked,
            blk_scale + start * num_blocks,
            wscale,
            blk_colsum,
            dst + start * N)

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Router: softmax + top-k dispatch
# ============================================================================


@fieldwise_init
struct RouterTopkArgs(Copyable, ImplicitlyCopyable):
    var logits: BF16Ptr
    var per_expert_scale: BF16Ptr
    var result_ptr: Int


def router_topk_kernel[num_experts: Int, k: Int](args: RouterTopkArgs):
    """Softmax → top-k → renormalize → per-expert scale."""
    var result = softmax_topk_renorm[num_experts, k](
        args.logits, args.per_expert_scale)
    UnsafePointer[Gemma4TopKResult[k], MutAnyOrigin](unsafe_from_address=args.result_ptr)[] = result
