"""LM head row-scan GEMV — per-block i8 dot product with VNNI."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from experimental3.kernels.int8_gemv import vpdpbusd
from experimental3.common_math import I8Ptr, F32Ptr, BF16Ptr


# ============================================================================
# Per-row dot product
# ============================================================================


@always_inline
def lm_head_row_dot[K: Int, fwht_blk: Int](
    act: I8Ptr,
    weight_row: I8Ptr,
    act_blk_scales: F32Ptr,
    w_blk_scales_row: F32Ptr,
    w_blk_colsums_row: F32Ptr,
) -> Float32:
    """Fully-dequant'd dot product for one output row."""
    comptime dp_width = simd_width_of[DType.int32]()
    comptime vnni_k_step = dp_width * 4
    debug_assert(K % fwht_blk == 0, "K must be a multiple of fwht_blk")
    debug_assert(
        fwht_blk % vnni_k_step == 0,
        "fwht_blk must be a multiple of the VNNI dot-product step",
    )
    comptime num_blocks = K // fwht_blk
    comptime steps_per_block = fwht_blk // vnni_k_step

    var act_u8 = act.bitcast[UInt8]()
    var u8_bias = SIMD[DType.uint8, vnni_k_step](0x80)
    var inv_i8_max = Float32(1.0) / Float32(127.0)
    var colsum_bias = Float32(128.0)
    var total = Float32(0)
    for b in range(num_blocks):
        var i32_acc = SIMD[DType.int32, dp_width](0)
        var k_base = b * fwht_blk
        comptime for s in range(steps_per_block):
            comptime step_k_off = s * vnni_k_step
            var k_off = k_base + step_k_off
            var act_step = (act_u8 + k_off).load[width=vnni_k_step]() ^ u8_bias
            var w_i8 = (weight_row + k_off).load[width=vnni_k_step]()
            i32_acc = vpdpbusd[dp_width](i32_acc, act_step, w_i8)
        var block_dot = i32_acc.reduce_add().cast[DType.float32]()
        var corrected = block_dot - colsum_bias * w_blk_colsums_row[b]
        var act_dequant = act_blk_scales[b] * inv_i8_max
        total += corrected * act_dequant * w_blk_scales_row[b]
    return total


# ============================================================================
# Worker
# ============================================================================


@fieldwise_init
struct LmHeadArgs(Copyable, ImplicitlyCopyable):
    var act: Int
    var weight: Int
    var act_blk_scales: Int
    var w_blk_scales: Int
    var w_blk_colsums: Int
    var dst: Int
    var n_start: Int
    var n_count: Int


def lm_head_worker[K: Int, fwht_blk: Int](args: LmHeadArgs):
    comptime num_blocks = K // fwht_blk
    var act = I8Ptr(unsafe_from_address=args.act)
    var weight = I8Ptr(unsafe_from_address=args.weight)
    var act_blk_scales = F32Ptr(unsafe_from_address=args.act_blk_scales)
    var w_blk_scales = F32Ptr(unsafe_from_address=args.w_blk_scales)
    var w_blk_colsums = F32Ptr(unsafe_from_address=args.w_blk_colsums)
    var dst = BF16Ptr(unsafe_from_address=args.dst)

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


# ============================================================================
# Dispatcher
# ============================================================================


def lm_head_gemv[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: Int,
    weight: Int,
    act_blk_scales: Int,
    w_blk_scales: Int,
    w_blk_colsums: Int,
    dst: Int,
    mut pool: P,
) -> PoolFence[P]:
    """LM head GEMV: i8 [N, K] x i8 [K] -> bf16 [N] with per-block dequant.

    Decode only (seq_len=1). Work is N-split across pool workers.
    """
    comptime MAX_POOL_CAPACITY = 128
    var num_workers = min(N, pool.get_capacity())
    var rows_per_worker = (N + num_workers - 1) // num_workers

    var zero_args = LmHeadArgs(0, 0, 0, 0, 0, 0, 0, 0)
    var jobs = InlineArray[LmHeadArgs, MAX_POOL_CAPACITY](fill=zero_args)
    var actual_jobs = 0
    for i in range(num_workers):
        var n_start = i * rows_per_worker
        if n_start >= N:
            break
        var n_count = min(rows_per_worker, N - n_start)
        jobs[i] = LmHeadArgs(
            act, weight, act_blk_scales, w_blk_scales, w_blk_colsums, dst,
            n_start, n_count)
        actual_jobs += 1

    pool.dispatch[LmHeadArgs, lm_head_worker[K, fwht_blk]](
        UnsafePointer(to=jobs[0]), actual_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
