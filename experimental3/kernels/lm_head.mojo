"""LM head row-scan GEMV — Gemma 4 tied unembedding.

Memory-bound at decode. The activation is small (HIDDEN bytes, in L1 after the
fused rmsnorm + FWHT + per-block quantize step). The cost per token is dominated
by streaming the weight matrix, so we want exactly one byte loaded per
(output, K position).

Layout:
  weight:        i8 [VOCAB, HIDDEN]              row-major
  w_blk_scales:  f32 [VOCAB, NUM_BLOCKS]         row-major (NUM_BLOCKS = HIDDEN/fwht_blk)
  w_blk_colsums: f32 [VOCAB, NUM_BLOCKS]         row-major
  act:           i8 [HIDDEN]                     post fused norm+FWHT+quantize
  act_blk_scales:f32 [NUM_BLOCKS]                per-K-block activation scales
  dst:           bf16 [VOCAB]                    output logits

No N-tile packing — the activation reuse benefit doesn't apply when the
activation is already cache-resident, and packing would force a second copy
of the table (defeats the bandwidth savings).

Per output row n:
  total = 0
  for each FWHT block b in 0..NUM_BLOCKS:
    block_dot = vpdpbusd-accumulate over the b-th K range, reduce to scalar
    corrected = block_dot - 128 * w_blk_colsums[n, b]
    dequant   = (act_blk_scales[b] / 127) * w_blk_scales[n, b]
    total    += corrected * dequant
  dst[n] = total.cast[bf16]()
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from experimental2.kernels.int8_gemv import vpdpbusd

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


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
    comptime num_blocks = K // fwht_blk
    comptime width = simd_width_of[DType.int32]()
    comptime k_per_step = width * 4
    comptime steps_per_block = fwht_blk // k_per_step

    var total = Float32(0)
    for b in range(num_blocks):
        var i32_acc = SIMD[DType.int32, width](0)
        var k_base = b * fwht_blk
        for s in range(steps_per_block):
            var k_off = k_base + s * k_per_step
            var act_u8 = (act + k_off).bitcast[UInt8]().load[width=k_per_step]() ^ SIMD[DType.uint8, k_per_step](0x80)
            var w_i8 = (weight_row + k_off).load[width=k_per_step]()
            i32_acc = vpdpbusd[width](i32_acc, act_u8, w_i8)
        var block_dot = i32_acc.reduce_add().cast[DType.float32]()
        var corrected = block_dot - Float32(128.0) * w_blk_colsums_row[b]
        total += corrected * (act_blk_scales[b] / Float32(127.0)) * w_blk_scales_row[b]
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
    """LM head GEMV: i8 [N, K] x i8 [K] -> bf16 [N], scheme 3 dequant.

    seq_len = 1 (decode). Work is N-split across pool workers.
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
