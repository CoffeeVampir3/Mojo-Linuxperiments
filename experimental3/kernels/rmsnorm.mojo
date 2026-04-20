"""RMSNorm kernels — row-level math + workers. Dispatchers in dispatch_kernels.mojo."""

from std.sys.info import simd_width_of

from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.fwht import fwht_block
from experimental3.moe import moe_combine
from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr,
    rms_reduce_bf16 as rms_reduce,
    inv_rms_from_sum_sq,
    normalize_inplace,
    pick_port_unroll,
    tree_reduce_accs,
)
from std.collections import InlineArray
from experimental3.kernels.dispatch_args import (
    RmsNormFwhtQuantArgs, RmsNormDualGammaFwhtArgs,
    RMSNormNoScaleArgs, RMSNormPerHeadArgs,
    PostAttnNormArgs, ExpertSumArgs, DenseNormArgs, PostReduceArgs,
)


# ============================================================================
# Shared helpers
# ============================================================================


@always_inline
def accumulate_expert_outputs[hidden: Int](
    expert_buf: BF16Ptr,
    local_count: Int,
    dst: BF16Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    debug_assert(hidden % width == 0, "hidden must be f32-simd-aligned")
    for i in range(0, hidden, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(local_count):
            acc += (expert_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())


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
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        comptime if has_gamma:
            var g = (gamma + off).load[width=width]().cast[DType.float32]()
            (work + off).store(x * g)
        else:
            (work + off).store(x)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            comptime if has_gamma:
                var g = (gamma + off).load[width=width]().cast[DType.float32]()
                (work + off).store(x * g)
            else:
                (work + off).store(x)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            (work + k).store(x * g)
        else:
            (work + k).store(x)
        k += width
    return tree_reduce_accs(accs)


@always_inline
def load_and_reduce_dual[cols: Int](
    src: BF16Ptr,
    gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
) -> Float32:
    """Load bf16 to two f32 work buffers with two gammas, shared sum(x^2)."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work_a + off).store(x * ga)
        (work_b + off).store(x * gb)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
            var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work_a + off).store(x * ga)
            (work_b + off).store(x * gb)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + k).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work_a + k).store(x * ga)
        (work_b + k).store(x * gb)
        k += width
    return tree_reduce_accs(accs)


@always_inline
def fwht_rotate[cols: Int, block: Int](work: F32Ptr):
    """Block-diagonal FWHT on f32 work buffer."""
    debug_assert(cols % block == 0, "cols must be a multiple of block")
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
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    comptime if per_block:
        debug_assert(block % width == 0, "block must be a multiple of f32 SIMD width")
        debug_assert(cols % block == 0, "cols must be a multiple of block")
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
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var sum_sq = rms_reduce[cols](src)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
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
# FWHT + quantize: unified args, worker
# ============================================================================


def rmsnorm_fwht_quant_worker[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    args: RmsNormFwhtQuantArgs,
):
    var inp = args.in_ptr
    var gamma = args.gamma_ptr
    var qi = args.qi_ptr
    var work = args.work_ptr
    var scales = args.scale_ptr

    comptime scale_stride = (cols // block) if per_block else 1
    for m in range(args.start_row, args.end_row):
        rmsnorm_fwht_quant_row[cols, block, has_gamma, per_block](
            inp + m * cols, gamma, qi + m * cols,
            work, scales + m * scale_stride, args.eps)


# ============================================================================
# Dual-lane FWHT + quantize: worker
# ============================================================================


def rmsnorm_dual_gamma_fwht_quant_worker[cols: Int, block: Int](
    args: RmsNormDualGammaFwhtArgs,
):
    var inp = args.in_ptr
    var gamma_a = args.gamma_a_ptr
    var gamma_b = args.gamma_b_ptr
    var qi_a = args.qi_a_ptr
    var qi_b = args.qi_b_ptr
    var work_a = args.work_a_ptr
    var work_b = args.work_b_ptr
    var scale_a = args.scale_a_ptr
    var scale_b = args.scale_b_ptr

    for m in range(args.start_row, args.end_row):
        rmsnorm_dual_gamma_fwht_quant_row[cols, block](
            inp + m * cols, gamma_a, gamma_b,
            qi_a + m * cols, qi_b + m * cols,
            work_a, work_b,
            scale_a + m, scale_b + m,
            args.eps,
        )


# ============================================================================
# bf16 norms: no FWHT, no quantize
#
# rmsnorm_no_scale: output = input / rms(input). No learnable weight.
# rmsnorm_per_head: per-head norm with shared weight vector.
# ============================================================================


def rmsnorm_no_scale_kernel[cols: Int](args: RMSNormNoScaleArgs):
    """RMSNorm without learnable scale, multi-row."""
    for row in range(args.start_row, args.end_row):
        rmsnorm_bf16_row[cols, False, False](
            args.input + row * cols,
            BF16Ptr(),
            args.output + row * cols,
            args.eps)


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


# ============================================================================
# Fused norm + residual kernels (single-row)
# ============================================================================


def post_attn_norm_kernel[hidden: Int](args: PostAttnNormArgs):
    """RMSNorm + residual add: x_main += rmsnorm(src, norm_w)."""
    rmsnorm_bf16_row[hidden, True, True](
        args.src_ptr,
        args.norm_w_ptr,
        args.x_main_ptr,
        args.eps)


@fieldwise_init
struct PreReduceArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: BF16Ptr
    var local_count: Int
    var dst_ptr: BF16Ptr
    var dense_ptr: BF16Ptr
    var norm_w_ptr: BF16Ptr
    var normed_ptr: BF16Ptr
    var eps: Float32

    def __init__(out self):
        self.expert_out_ptr = BF16Ptr()
        self.local_count = 0
        self.dst_ptr = BF16Ptr()
        self.dense_ptr = BF16Ptr()
        self.norm_w_ptr = BF16Ptr()
        self.normed_ptr = BF16Ptr()
        self.eps = Float32(0)

def pre_reduce_kernel[hidden: Int](args: PreReduceArgs):
    """Accumulate local experts into dst + rmsnorm(dense, norm_w) into normed."""
    var expert_buf = args.expert_out_ptr
    var dst = args.dst_ptr
    accumulate_expert_outputs[hidden](expert_buf, args.local_count, dst)
    var dense = args.dense_ptr
    rmsnorm_bf16_row[hidden, True, False](
        dense,
        args.norm_w_ptr,
        args.normed_ptr,
        args.eps)


def expert_sum_kernel[hidden: Int](args: ExpertSumArgs):
    """Accumulate local expert outputs into dst."""
    var expert_buf = args.expert_out_ptr
    var dst = args.dst_ptr
    accumulate_expert_outputs[hidden](expert_buf, args.local_count, dst)


def dense_norm_kernel[hidden: Int](args: DenseNormArgs):
    """RMSNorm dense output (no residual add)."""
    rmsnorm_bf16_row[hidden, True, False](
        args.src_ptr,
        args.norm_w_ptr,
        args.dst_ptr,
        args.eps)


def post_reduce_kernel[hidden: Int](args: PostReduceArgs):
    """Post-allreduce: norms + combine + residual + scalar."""
    moe_combine[hidden](
        args.moe_out_ptr,
        args.moe_norm_w_ptr,
        args.dense_normed_ptr,
        args.combine_norm_w_ptr,
        args.x_main_ptr,
        args.layer_scalar, args.eps)
