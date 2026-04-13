"""RMSNorm kernels — composed from atomic stages in common_math."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from simd_math import sqrt
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.fwht import fwht_block
from experimental3.moe import moe_combine
from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr,
    rms_reduce_bf16 as rms_reduce,
    inv_rms_from_sum_sq,
    normalize_inplace,
)


# ============================================================================
# Fused load + reduce stages (rmsnorm-specific — combine bf16 load with
# RMS reduction and optional gamma application in a single pass)
# ============================================================================


@always_inline
def load_and_reduce[cols: Int, has_gamma: Bool](
    src: BF16Ptr, gamma: BF16Ptr, work: F32Ptr,
) -> Float32:
    """Load bf16 to f32 work buffer, accumulate sum(x^2).

    has_gamma=True: work[k] = x[k] * gamma[k], sum from raw x.
    has_gamma=False: work[k] = x[k]. gamma ptr ignored.
    """
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            (work + k).store(x * g)
        else:
            (work + k).store(x)
        k += width
    return vsum.reduce_add()


@always_inline
def load_and_reduce_dual[cols: Int](
    src: BF16Ptr,
    gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
) -> Float32:
    """Load bf16 to two f32 work buffers with two gammas, shared sum(x^2)."""
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + k).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work_a + k).store(x * ga)
        (work_b + k).store(x * gb)
        k += width
    return vsum.reduce_add()


@always_inline
def fwht_rotate[cols: Int, block: Int](work: F32Ptr):
    """Block-diagonal FWHT on f32 work buffer."""
    for b in range(cols // block):
        fwht_block[block](work + b * block)


@always_inline
def emit_quant[cols: Int, block: Int, per_block: Bool](
    work: F32Ptr, qi: I8Ptr, scales: F32Ptr,
):
    """Quantize f32 work buffer to i8.

    per_block=True: one absmax per FWHT block (cols/block scales).
    per_block=False: one absmax for the whole row (1 scale).
    """
    comptime if per_block:
        comptime num_blocks = cols // block
        for b in range(num_blocks):
            scales[b] = absmax_quantize_i8[block](
                work + b * block, qi + b * block)
    else:
        scales[] = absmax_quantize_i8[cols](work, qi)


# ============================================================================
# Composed row kernel 1: single-lane FWHT + quantize
#
#   [has_gamma=F, per_block=F]  smollm2 attn/MLP
#   [has_gamma=T, per_block=F]  gemma4 attn input, router
#   [has_gamma=T, per_block=T]  gemma4 lm head
# ============================================================================


@always_inline
def rmsnorm_fwht_quant_row[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scales: F32Ptr, eps: Float32,
):
    """Load -> RMSNorm [* gamma] -> FWHT -> quantize [per-row | per-block]."""
    var sum_sq = load_and_reduce[cols, has_gamma](src, gamma, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    normalize_inplace[cols](work, inv)
    fwht_rotate[cols, block](work)
    emit_quant[cols, block, per_block](work, qi, scales)


# ============================================================================
# Composed row kernel 2: dual-lane FWHT + quantize
#
# Two gammas share one RMS reduction. Used by FFN (pre_ffn_norm +
# pre_ffn_norm_2 produce dense and expert activations in one pass).
# ============================================================================


@always_inline
def rmsnorm_dual_gamma_fwht_quant_row[cols: Int, block: Int](
    src: BF16Ptr,
    gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    qi_a: I8Ptr, qi_b: I8Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
    scale_a: F32Ptr, scale_b: F32Ptr,
    eps: Float32,
):
    """Load -> RMSNorm * (gamma_a, gamma_b) -> 2x FWHT -> 2x per-row i8."""
    var sum_sq = load_and_reduce_dual[cols](src, gamma_a, gamma_b, work_a, work_b)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    normalize_inplace[cols](work_a, inv)
    normalize_inplace[cols](work_b, inv)
    fwht_rotate[cols, block](work_a)
    fwht_rotate[cols, block](work_b)
    emit_quant[cols, block, False](work_a, qi_a, scale_a)
    emit_quant[cols, block, False](work_b, qi_b, scale_b)


# ============================================================================
# Composed row kernel 3: bf16 output (no FWHT, no quantize)
#
#   [has_gamma=F, has_residual=F]  V-norm, router-norm
#   [has_gamma=T, has_residual=F]  standard rmsnorm with gamma
#   [has_gamma=T, has_residual=T]  post-attention residual add
# ============================================================================


@always_inline
def rmsnorm_bf16_row[cols: Int, has_gamma: Bool, has_residual: Bool](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    """RMSNorm to bf16 output. Reloads from bf16 (no work buffer).

    has_gamma=False: dst = src / rms(src). gamma ptr ignored.
    has_gamma=True, has_residual=False: dst = (src / rms(src)) * gamma.
    has_gamma=True, has_residual=True: dst += (src / rms(src)) * gamma.
    """
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce[cols](src)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k + width <= cols:
        var v = (src + k).load[width=width]().cast[DType.float32]()
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            var normed = v * vinv * g
            comptime if has_residual:
                var x = (dst + k).load[width=width]().cast[DType.float32]()
                (dst + k).store((x + normed).cast[DType.bfloat16]())
            else:
                (dst + k).store(normed.cast[DType.bfloat16]())
        else:
            (dst + k).store((v * vinv).cast[DType.bfloat16]())
        k += width


# ============================================================================
# FWHT + quantize: unified args, worker, dispatcher
# ============================================================================


@fieldwise_init
struct RmsNormFwhtQuantArgs(Copyable, ImplicitlyCopyable):
    """Unified args for all single-lane FWHT+quantize variants.

    gamma_ptr is 0 when has_gamma=False (never dereferenced).
    scale_ptr points to 1 scale per row (per-row) or cols/block per row (per-block).
    """
    var in_ptr: Int
    var gamma_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int


def rmsnorm_fwht_quant_worker[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    args: RmsNormFwhtQuantArgs,
):
    var inp = BF16Ptr(unsafe_from_address=args.in_ptr)
    var gamma = BF16Ptr(unsafe_from_address=args.gamma_ptr)
    var qi = I8Ptr(unsafe_from_address=args.qi_ptr)
    var work = F32Ptr(unsafe_from_address=args.work_ptr)
    var scales = F32Ptr(unsafe_from_address=args.scale_ptr)

    comptime scale_stride = (cols // block) if per_block else 1
    for m in range(args.start_row, args.end_row):
        rmsnorm_fwht_quant_row[cols, block, has_gamma, per_block](
            inp + m * cols, gamma, qi + m * cols,
            work, scales + m * scale_stride, args.eps)


def rmsnorm_fwht_quant[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Unified FWHT+quantize dispatcher. Use convenience wrappers below."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormFwhtQuantArgs, MAX](
        fill=RmsNormFwhtQuantArgs(0, 0, 0, 0, 0, 0.0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtQuantArgs(
            in_ptr, gamma_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, eps, start, end)

    pool.dispatch[RmsNormFwhtQuantArgs,
        rmsnorm_fwht_quant_worker[cols, block, has_gamma, per_block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ---- Convenience wrappers (preserve existing call-site names) ----


def rmsnorm_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm * gamma + FWHT + per-row i8."""
    return rmsnorm_fwht_quant[cols, block, True, False, P](
        in_ptr, gamma_ptr, qi_ptr, work_ptr, scale_ptr, eps, seq_len, pool)


def rmsnorm_gamma_fwht_per_block_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    blk_scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm * gamma + FWHT + per-block i8."""
    return rmsnorm_fwht_quant[cols, block, True, True, P](
        in_ptr, gamma_ptr, qi_ptr, work_ptr, blk_scale_ptr, eps, seq_len, pool)


def rmsnorm_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm + FWHT + per-row i8. No gamma."""
    return rmsnorm_fwht_quant[cols, block, False, False, P](
        in_ptr, 0, qi_ptr, work_ptr, scale_ptr, eps, seq_len, pool)


# ============================================================================
# Dual-lane FWHT + quantize: args, worker, dispatcher
# ============================================================================


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


def rmsnorm_dual_gamma_fwht_quant_worker[cols: Int, block: Int](
    args: RmsNormDualGammaFwhtArgs,
):
    var inp = BF16Ptr(unsafe_from_address=args.in_ptr)
    var gamma_a = BF16Ptr(unsafe_from_address=args.gamma_a_ptr)
    var gamma_b = BF16Ptr(unsafe_from_address=args.gamma_b_ptr)
    var qi_a = I8Ptr(unsafe_from_address=args.qi_a_ptr)
    var qi_b = I8Ptr(unsafe_from_address=args.qi_b_ptr)
    var work_a = F32Ptr(unsafe_from_address=args.work_a_ptr)
    var work_b = F32Ptr(unsafe_from_address=args.work_b_ptr)
    var scale_a = F32Ptr(unsafe_from_address=args.scale_a_ptr)
    var scale_b = F32Ptr(unsafe_from_address=args.scale_b_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_dual_gamma_fwht_quant_row[cols, block](
            inp + m * cols, gamma_a, gamma_b,
            qi_a + m * cols, qi_b + m * cols,
            work_a, work_b,
            scale_a + m, scale_b + m,
            args.eps,
        )


def rmsnorm_dual_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int,
    gamma_a_ptr: Int, gamma_b_ptr: Int,
    qi_a_ptr: Int, qi_b_ptr: Int,
    work_a_ptr: Int, work_b_ptr: Int,
    scale_a_ptr: Int, scale_b_ptr: Int,
    eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dual-gamma RMSNorm + FWHT + per-row i8. One pass, two outputs."""
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

    pool.dispatch[RmsNormDualGammaFwhtArgs,
        rmsnorm_dual_gamma_fwht_quant_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# bf16 norms: no FWHT, no quantize
#
# rmsnorm_no_scale: output = input / rms(input). No learnable weight.
# rmsnorm_per_head: per-head norm with shared weight vector.
# ============================================================================


@fieldwise_init
struct RMSNormNoScaleArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float32


def rmsnorm_no_scale_kernel[cols: Int](args: RMSNormNoScaleArgs):
    """RMSNorm without learnable scale, multi-row."""
    for row in range(args.start_row, args.end_row):
        rmsnorm_bf16_row[cols, False, False](
            args.input + row * cols,
            BF16Ptr(),
            args.output + row * cols,
            args.eps)


@fieldwise_init
struct RMSNormPerHeadArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float32


def rmsnorm_per_head_kernel[head_dim: Int, num_heads: Int](args: RMSNormPerHeadArgs):
    """Per-head RMSNorm with learnable scale. Each head_dim segment
    normalized independently, scaled by shared weight vector."""
    comptime row_stride = num_heads * head_dim
    for row in range(args.start_row, args.end_row):
        var row_in = args.input + row * row_stride
        var row_out = args.output + row * row_stride
        for h in range(num_heads):
            rmsnorm_bf16_row[head_dim, True, False](
                row_in + h * head_dim,
                args.weight,
                row_out + h * head_dim,
                args.eps)


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
        jobs[i] = RMSNormNoScaleArgs(ip, op, start, end, eps)

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
        jobs[i] = RMSNormPerHeadArgs(ip, wp, op, start, end, eps)

    pool.dispatch[RMSNormPerHeadArgs, rmsnorm_per_head_kernel[head_dim, num_heads]](
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
    rmsnorm_bf16_row[hidden, True, True](
        BF16Ptr(unsafe_from_address=args.src_ptr),
        BF16Ptr(unsafe_from_address=args.norm_w_ptr),
        BF16Ptr(unsafe_from_address=args.x_main_ptr),
        args.eps)


@fieldwise_init
struct PreReduceArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: Int
    var local_count: Int
    var dst_ptr: Int
    var dense_ptr: Int
    var norm_w_ptr: Int
    var normed_ptr: Int
    var eps: Float32

def pre_reduce_kernel[hidden: Int](args: PreReduceArgs):
    """Accumulate local experts into dst + rmsnorm(dense, norm_w) into normed."""
    comptime width = simd_width_of[DType.float32]()
    var expert_buf = BF16Ptr(unsafe_from_address=args.expert_out_ptr)
    var dst = BF16Ptr(unsafe_from_address=args.dst_ptr)
    for i in range(0, hidden, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(args.local_count):
            acc += (expert_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
    var dense = BF16Ptr(unsafe_from_address=args.dense_ptr)
    rmsnorm_bf16_row[hidden, True, False](
        dense,
        BF16Ptr(unsafe_from_address=args.norm_w_ptr),
        BF16Ptr(unsafe_from_address=args.normed_ptr),
        args.eps)


@fieldwise_init
struct ExpertSumArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: Int
    var local_count: Int
    var dst_ptr: Int

def expert_sum_kernel[hidden: Int](args: ExpertSumArgs):
    """Accumulate local expert outputs into dst."""
    comptime width = simd_width_of[DType.float32]()
    var expert_buf = BF16Ptr(unsafe_from_address=args.expert_out_ptr)
    var dst = BF16Ptr(unsafe_from_address=args.dst_ptr)
    for i in range(0, hidden, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(args.local_count):
            acc += (expert_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())


@fieldwise_init
struct DenseNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: Int
    var norm_w_ptr: Int
    var dst_ptr: Int
    var eps: Float32

def dense_norm_kernel[hidden: Int](args: DenseNormArgs):
    """RMSNorm dense output (no residual add)."""
    rmsnorm_bf16_row[hidden, True, False](
        BF16Ptr(unsafe_from_address=args.src_ptr),
        BF16Ptr(unsafe_from_address=args.norm_w_ptr),
        BF16Ptr(unsafe_from_address=args.dst_ptr),
        args.eps)


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
        BF16Ptr(unsafe_from_address=args.moe_out_ptr),
        BF16Ptr(unsafe_from_address=args.moe_norm_w_ptr),
        BF16Ptr(unsafe_from_address=args.dense_normed_ptr),
        BF16Ptr(unsafe_from_address=args.combine_norm_w_ptr),
        BF16Ptr(unsafe_from_address=args.x_main_ptr),
        args.layer_scalar, args.eps)
