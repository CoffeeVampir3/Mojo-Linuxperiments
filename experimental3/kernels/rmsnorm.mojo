"""Centralized RMSNorm kernels for Gemma4.

All RMSNorm variants live here:
  - Plain bf16->bf16 norms: rmsnorm_no_scale, rmsnorm_per_head
  - FWHT+quantize fused norms: rmsnorm_fwht_quantize, rmsnorm_gamma_fwht_quantize,
    rmsnorm_dual_gamma_fwht_quantize, rmsnorm_gamma_fwht_per_block_quantize
  - Fused norm+residual: post_attn_norm_kernel, pre_reduce_kernel,
    post_reduce_kernel (single-row dispatched kernels for decode forward)
"""

from std.math import sqrt as std_sqrt
from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from simd_math import sqrt
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.fwht import fwht_block
from experimental3.moe import moe_combine


# ============================================================================
# Plain bf16->bf16 norms (no FWHT, no quantization)
# ============================================================================


@fieldwise_init
struct RMSNormNoScaleArgs(Copyable, ImplicitlyCopyable):
    var input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var start_row: Int
    var end_row: Int
    var eps: Float64


@fieldwise_init
struct RMSNormPerHeadArgs(Copyable, ImplicitlyCopyable):
    var input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var start_row: Int
    var end_row: Int
    var eps: Float64


def rmsnorm_no_scale_kernel[cols: Int](args: RMSNormNoScaleArgs):
    """RMSNorm without learnable scale. output = input / rms(input)."""
    var ip = args.input
    var op = args.output
    comptime width = simd_width_of[DType.float32]()

    for row in range(args.start_row, args.end_row):
        var row_in = ip + row * cols
        var row_out = op + row * cols

        var acc = SIMD[DType.float32, width](0)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            acc = x.fma(x, acc)
        var sum_sq = acc.reduce_add()
        var scale = Float32(1.0) / std_sqrt(sum_sq / Float32(cols) + Float32(args.eps))

        var sv = SIMD[DType.float32, width](scale)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            (row_out + j).store((x * sv).cast[DType.bfloat16]())


def rmsnorm_per_head_kernel[head_dim: Int, num_heads: Int](args: RMSNormPerHeadArgs):
    """RMSNorm with learnable scale, applied per-head.

    Input row is (num_heads * head_dim) wide. Each head_dim segment
    is normalized independently, then scaled by the shared weight
    vector of length head_dim.
    """
    var ip = args.input
    var wp = args.weight
    var op = args.output
    comptime width = simd_width_of[DType.float32]()
    comptime row_stride = num_heads * head_dim

    for row in range(args.start_row, args.end_row):
        var row_base_in = ip + row * row_stride
        var row_base_out = op + row * row_stride

        for h in range(num_heads):
            var head_in = row_base_in + h * head_dim
            var head_out = row_base_out + h * head_dim

            var acc = SIMD[DType.float32, width](0)
            for j in range(0, head_dim, width):
                var x = (head_in + j).load[width=width]().cast[DType.float32]()
                acc = x.fma(x, acc)
            var sum_sq = acc.reduce_add()
            var scale = Float32(1.0) / std_sqrt(sum_sq / Float32(head_dim) + Float32(args.eps))

            var sv = SIMD[DType.float32, width](scale)
            for j in range(0, head_dim, width):
                var x = (head_in + j).load[width=width]().cast[DType.float32]()
                var w = (wp + j).load[width=width]().cast[DType.float32]()
                (head_out + j).store((x * sv * w).cast[DType.bfloat16]())


def rmsnorm_no_scale[InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P]:
    """RMSNorm without learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_no_scale: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm_no_scale: cols must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = tptr[Scalar[DType.bfloat16]](input.ptr)
    var op = tptr[Scalar[DType.bfloat16]](output.ptr)
    var jobs = InlineArray[RMSNormNoScaleArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormNoScaleArgs(ip, op, start, end, Float64(eps))

    pool.dispatch[RMSNormNoScaleArgs, rmsnorm_no_scale_kernel[InT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_per_head[head_dim: Int, num_heads: Int,
    W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P] where W.DTYPE == DType.bfloat16:
    """Per-head RMSNorm with learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_per_head: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_per_head: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_per_head: input/output cols mismatch"
    comptime assert InT.COLS == head_dim * num_heads, "rmsnorm_per_head: cols != heads * dim"
    comptime assert W.ROWS * W.COLS == head_dim, "rmsnorm_per_head: weight size != head_dim"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "rmsnorm_per_head: head_dim must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = tptr[Scalar[DType.bfloat16]](input.ptr)
    var wp = tptr[Scalar[DType.bfloat16]](weight.ptr)
    var op = tptr[Scalar[DType.bfloat16]](output.ptr)
    var jobs = InlineArray[RMSNormPerHeadArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormPerHeadArgs(ip, wp, op, start, end, Float64(eps))

    pool.dispatch[RMSNormPerHeadArgs, rmsnorm_per_head_kernel[head_dim, num_heads]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# FWHT + quantize row kernels
# ============================================================================


def rmsnorm_gamma_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm * gamma -> FWHT -> dynamic i8."""
    comptime width = simd_width_of[DType.float32]()

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        var g = (gamma + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x * g)
        k += width

    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(cols // block):
        fwht_block[block](work + b * block)

    scale_out[] = absmax_quantize_i8[cols](work, row_qi)


def rmsnorm_dual_gamma_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma_a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma_b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi_a: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    row_qi_b: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work_a: UnsafePointer[Float32, MutAnyOrigin],
    work_b: UnsafePointer[Float32, MutAnyOrigin],
    scale_a: UnsafePointer[Float32, MutAnyOrigin],
    scale_b: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row, two gamma: bf16 -> RMSNorm * (ga, gb) -> FWHT -> dynamic i8.

    Iterate-once, write-twice: x is loaded once, both (x*ga) and (x*gb)
    share the RMS reduction. Two outputs, two scales, one pass over x.
    """
    comptime width = simd_width_of[DType.float32]()

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + k).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work_a + k).store(x * ga)
        (work_b + k).store(x * gb)
        k += width

    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work_a + k).store((work_a + k).load[width=width]() * vinv_rms)
        (work_b + k).store((work_b + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(cols // block):
        fwht_block[block](work_a + b * block)
        fwht_block[block](work_b + b * block)

    scale_a[] = absmax_quantize_i8[cols](work_a, row_qi_a)
    scale_b[] = absmax_quantize_i8[cols](work_b, row_qi_b)


def rmsnorm_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm -> FWHT -> dynamic i8. No gamma."""
    comptime width = simd_width_of[DType.float32]()

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x)
        k += width

    var rms = sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var inv_rms = Float32(1.0) / rms

    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(cols // block):
        fwht_block[block](work + b * block)

    scale_out[] = absmax_quantize_i8[cols](work, row_qi)


def rmsnorm_gamma_fwht_per_block_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    blk_scales_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm * gamma -> FWHT -> per-block dynamic i8."""
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        var g = (gamma + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x * g)
        k += width

    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(num_blocks):
        fwht_block[block](work + b * block)
        blk_scales_out[b] = absmax_quantize_i8[block](
            work + b * block, row_qi + b * block)


# ============================================================================
# Worker arg structs + workers
# ============================================================================


@fieldwise_init
struct RmsNormFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var start_row: Int
    var end_row: Int


def rmsnorm_fwht_quantize_worker[cols: Int, block: Int](args: RmsNormFwhtArgs):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_fwht_quantize_row[cols, block](
            inp + m * cols, qi + m * cols, work, scales + m, 1e-5,
        )


@fieldwise_init
struct RmsNormGammaFwhtPerBlockArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var gamma_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var blk_scale_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int


def rmsnorm_gamma_fwht_per_block_quantize_worker[cols: Int, block: Int](
    args: RmsNormGammaFwhtPerBlockArgs,
):
    comptime num_blocks = cols // block
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var gamma = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gamma_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var blk_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.blk_scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_gamma_fwht_per_block_quantize_row[cols, block](
            inp + m * cols, gamma, qi + m * cols, work,
            blk_scales + m * num_blocks, args.eps,
        )


@fieldwise_init
struct RmsNormGammaFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var gamma_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int


def rmsnorm_gamma_fwht_quantize_worker[cols: Int, block: Int](args: RmsNormGammaFwhtArgs):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var gamma = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gamma_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_gamma_fwht_quantize_row[cols, block](
            inp + m * cols, gamma, qi + m * cols, work, scales + m, args.eps,
        )


@fieldwise_init
struct RmsNormDualGammaFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var gamma_a_ptr: Int
    var gamma_b_ptr: Int
    var qi_a_ptr: Int
    var qi_b_ptr: Int
    var work_a_ptr: Int
    var work_b_ptr: Int
    var scale_a_ptr: Int
    var scale_b_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int


def rmsnorm_dual_gamma_fwht_quantize_worker[cols: Int, block: Int](
    args: RmsNormDualGammaFwhtArgs,
):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var gamma_a = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gamma_a_ptr)
    var gamma_b = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gamma_b_ptr)
    var qi_a = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_a_ptr)
    var qi_b = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_b_ptr)
    var work_a = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_a_ptr)
    var work_b = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_b_ptr)
    var scale_a = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_a_ptr)
    var scale_b = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_b_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_dual_gamma_fwht_quantize_row[cols, block](
            inp + m * cols, gamma_a, gamma_b,
            qi_a + m * cols, qi_b + m * cols,
            work_a, work_b,
            scale_a + m, scale_b + m,
            args.eps,
        )


# ============================================================================
# Dispatchers
# ============================================================================


def rmsnorm_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int,
    eps: Float32,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm * gamma + FWHT + dynamic-scale quantize.

    in_ptr:     bf16 [seq_len, cols]
    gamma_ptr:  bf16 [cols]
    qi_ptr:     i8   [seq_len, cols] output
    work_ptr:   f32  [cols] scratch per worker
    scale_ptr:  f32  [seq_len] output — per-row absmax scales
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormGammaFwhtArgs, MAX](
        fill=RmsNormGammaFwhtArgs(0, 0, 0, 0, 0, 0.0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormGammaFwhtArgs(
            in_ptr, gamma_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, eps, start, end)

    pool.dispatch[RmsNormGammaFwhtArgs, rmsnorm_gamma_fwht_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_gamma_fwht_per_block_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    blk_scale_ptr: Int,
    eps: Float32,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm * gamma + FWHT + per-block dynamic-scale quantize."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormGammaFwhtPerBlockArgs, MAX](
        fill=RmsNormGammaFwhtPerBlockArgs(0, 0, 0, 0, 0, 0.0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormGammaFwhtPerBlockArgs(
            in_ptr, gamma_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            blk_scale_ptr, eps, start, end)

    pool.dispatch[RmsNormGammaFwhtPerBlockArgs, rmsnorm_gamma_fwht_per_block_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_dual_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int,
    gamma_a_ptr: Int,
    gamma_b_ptr: Int,
    qi_a_ptr: Int,
    qi_b_ptr: Int,
    work_a_ptr: Int,
    work_b_ptr: Int,
    scale_a_ptr: Int,
    scale_b_ptr: Int,
    eps: Float32,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch dual-gamma RMSNorm + FWHT + dynamic-scale quantize.

    Single pass over x produces two quantized outputs (one per gamma),
    sharing the load and the RMS reduction.
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormDualGammaFwhtArgs, MAX](
        fill=RmsNormDualGammaFwhtArgs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormDualGammaFwhtArgs(
            in_ptr, gamma_a_ptr, gamma_b_ptr,
            qi_a_ptr, qi_b_ptr,
            work_a_ptr + i * cols * size_of[Float32](),
            work_b_ptr + i * cols * size_of[Float32](),
            scale_a_ptr, scale_b_ptr,
            eps, start, end)

    pool.dispatch[RmsNormDualGammaFwhtArgs, rmsnorm_dual_gamma_fwht_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm + FWHT + dynamic-scale quantize. No gamma."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormFwhtArgs, MAX](
        fill=RmsNormFwhtArgs(0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtArgs(
            in_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, start, end)

    pool.dispatch[RmsNormFwhtArgs, rmsnorm_fwht_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# Fused norm + residual kernels (single-row, dispatched for decode forward)
# ============================================================================


@fieldwise_init
struct PostAttnNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: Int
    var norm_w_ptr: Int
    var x_main_ptr: Int
    var eps: Float32

def post_attn_norm_kernel[hidden: Int](args: PostAttnNormArgs):
    """RMSNorm + residual add: x_main += rmsnorm(src, norm_w)."""
    comptime width = simd_width_of[DType.float32]()
    var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.src_ptr)
    var norm_w = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.norm_w_ptr)
    var x_main = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.x_main_ptr)
    var sum_sq = SIMD[DType.float32, width](0)
    for i in range(0, hidden, width):
        var v = (src + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + args.eps)
    for i in range(0, hidden, width):
        var v = (src + i).load[width=width]().cast[DType.float32]()
        var g = (norm_w + i).load[width=width]().cast[DType.float32]()
        var x = (x_main + i).load[width=width]().cast[DType.float32]()
        (x_main + i).store((x + v * inv_rms * g).cast[DType.bfloat16]())


@fieldwise_init
struct PreReduceArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: Int
    var local_count: Int
    var dst_ptr: Int
    var dense_ptr: Int
    var norm_w_ptr: Int
    var normed_ptr: Int
    var hidden: Int
    var eps: Float32

def pre_reduce_kernel(args: PreReduceArgs):
    """Accumulate local experts into dst + rmsnorm(dense, norm_w) into normed."""
    comptime width = simd_width_of[DType.float32]()
    var h = args.hidden
    var expert_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.expert_out_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dst_ptr)
    var dense = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dense_ptr)
    var norm_w = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.norm_w_ptr)
    var normed = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.normed_ptr)
    for i in range(0, h, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(args.local_count):
            acc += (expert_buf + e * h + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
    var sum_sq = SIMD[DType.float32, width](0)
    for i in range(0, h, width):
        var v = (dense + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(h) + args.eps)
    for i in range(0, h, width):
        var v = (dense + i).load[width=width]().cast[DType.float32]()
        var g = (norm_w + i).load[width=width]().cast[DType.float32]()
        (normed + i).store((v * inv_rms * g).cast[DType.bfloat16]())


@fieldwise_init
struct PostReduceArgs(Copyable, ImplicitlyCopyable):
    var moe_out_ptr: Int
    var moe_norm_w_ptr: Int
    var dense_normed_ptr: Int
    var combine_norm_w_ptr: Int
    var x_main_ptr: Int
    var layer_scalar: Float32
    var eps: Float32

def post_reduce_kernel[hidden: Int](args: PostReduceArgs):
    """Post-allreduce: norms + combine + residual + scalar."""
    moe_combine[hidden](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.moe_out_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.moe_norm_w_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dense_normed_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.combine_norm_w_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.x_main_ptr),
        args.layer_scalar, args.eps)
