"""Dense MLP dispatch kernels for Gemma 4 forward pass.

Phase 1: Fused gate_up GEMV + GELU-tanh + FWHT + DC + per-block quantize.
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


# ============================================================================
# Phase 1: Fused gate_up GEMV + GELU-tanh + FWHT + DC + per-block quantize
# ============================================================================


@fieldwise_init
struct FusedGuGeluTanhArgs(Copyable, ImplicitlyCopyable):
    var act_ptr: Int
    var act_scale: Float32
    var wpacked_ptr: Int
    var wscale_ptr: Int
    var wcolsum_ptr: Int
    var qi_ptr: Int
    var blk_scale_ptr: Int


def fused_gu_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    """Fused: gate+up GEMV -> split -> GELU-tanh(gate)*up -> FWHT+DC -> per-block i8.

    Fused weight is [2*intermediate, K]. GEMV produces f32[2*intermediate] on stack.
    gate = [0:intermediate], up = [intermediate:2*intermediate].
    Output: i8[intermediate] + f32[intermediate/fwht_blk] block scales.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime gate_up_dim = 2 * intermediate
    comptime num_blocks = intermediate // fwht_blk
    comptime DC_SCALE = Float32(0.5)

    var act_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.act_ptr)

    # Phase 1: gate+up GEMV → f32 stack
    var gu_buf = InlineArray[Float32, gate_up_dim](fill=Float32(0))
    var work = UnsafePointer(to=gu_buf).bitcast[Float32]()
    gemv_row[gate_up_dim, K, DType.float32](
        act_i8,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=args.wpacked_ptr),
        args.act_scale / 127.0,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.wscale_ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.wcolsum_ptr),
        work.bitcast[Scalar[DType.float32]]())

    # Phase 2: gelu_tanh(gate) * up → overwrite first half
    var up_f32 = work + intermediate
    var k = 0
    while k + width <= intermediate:
        var g = (work + k).load[width=width]()
        var u = (up_f32 + k).load[width=width]()
        (work + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    # Phase 3: FWHT + DC correction + per-block quantize → write to scratch
    var qi_out = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var blk_sc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.blk_scale_ptr)
    for b in range(num_blocks):
        fwht_block[fwht_blk](work + b * fwht_blk)
        work[b * fwht_blk] *= DC_SCALE
        blk_sc[b] = absmax_quantize_i8[fwht_blk](work + b * fwht_blk, qi_out + b * fwht_blk)


def fused_gu_gelu_tanh[intermediate: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act_ptr: Int,
    act_scale_ptr: Int,
    wpacked_ptr: Int,
    wscale_ptr: Int,
    wcolsum_ptr: Int,
    qi_ptr: Int,
    blk_scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch fused gate_up GEMV + GELU-tanh + FWHT + DC + per-block quantize.

    act_ptr:        i8 [seq_len, K]
    act_scale_ptr:  f32 [seq_len] per-row activation scales
    wpacked_ptr:    VNNI [2*intermediate, K] fused gate+up weight
    wscale_ptr:     f32 [2*intermediate] weight scales
    wcolsum_ptr:    f32 [2*intermediate] weight column sums
    qi_ptr:         i8 [seq_len, intermediate] output
    blk_scale_ptr:  f32 [seq_len, intermediate/fwht_blk] per-block scales output
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime num_blocks = intermediate // fwht_blk
    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](
        fill=FusedGuGeluTanhArgs(0, Float32(0), 0, 0, 0, 0, 0))
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale_ptr)
    for i in range(num_jobs):
        var start = i * rows_per_job
        jobs[i] = FusedGuGeluTanhArgs(
            act_ptr + start * K,
            act_scales[start],
            wpacked_ptr, wscale_ptr, wcolsum_ptr,
            qi_ptr + start * intermediate,
            blk_scale_ptr + start * num_blocks * size_of[Float32]())

    pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Phase 2: Int8 GEMV with per-block activation scales (down projection)
# ============================================================================


@fieldwise_init
struct Int8GemvBlockedArgs(Copyable, ImplicitlyCopyable):
    var act_ptr: Int
    var wpacked_ptr: Int
    var blk_scale_ptr: Int
    var wscale_ptr: Int
    var blk_colsum_ptr: Int
    var dst_ptr: Int


def int8_gemv_blocked_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """Down GEMV with per-block activation scales. Output bf16[N]."""
    comptime width = simd_width_of[DType.float32]()

    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked[N, K, fwht_blk](
        UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.act_ptr),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=args.wpacked_ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.blk_scale_ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.wscale_ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.blk_colsum_ptr),
        dp)

    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dst_ptr)
    var k = 0
    while k + width <= N:
        (dst + k).store((dp + k).load[width=width]().cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act_ptr: Int,
    wpacked_ptr: Int,
    blk_scale_ptr: Int,
    wscale_ptr: Int,
    blk_colsum_ptr: Int,
    dst_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch int8 GEMV with per-block activation scales.

    act_ptr:        i8 [seq_len, K]
    wpacked_ptr:    VNNI [N, K]
    blk_scale_ptr:  f32 [seq_len, K/fwht_blk] per-block activation scales
    wscale_ptr:     f32 [N] weight scales
    blk_colsum_ptr: f32 [N, K/fwht_blk] per-block column sums
    dst_ptr:        bf16 [seq_len, N] output
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime num_blocks = K // fwht_blk
    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs(0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        jobs[i] = Int8GemvBlockedArgs(
            act_ptr + start * K,
            wpacked_ptr,
            blk_scale_ptr + start * num_blocks * size_of[Float32](),
            wscale_ptr,
            blk_colsum_ptr,
            dst_ptr + start * N * 2)

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Router: softmax + top-k dispatch
# ============================================================================


@fieldwise_init
struct RouterTopkArgs(Copyable, ImplicitlyCopyable):
    var logits_ptr: Int
    var per_expert_scale_ptr: Int
    var result_ptr: Int


def router_topk_kernel[num_experts: Int, k: Int](args: RouterTopkArgs):
    """Softmax → top-k → renormalize → per-expert scale."""
    var result = softmax_topk_renorm[num_experts, k](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.logits_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.per_expert_scale_ptr))
    UnsafePointer[Gemma4TopKResult[k], MutAnyOrigin](unsafe_from_address=args.result_ptr)[] = result
