"""RMSNorm kernel composition framework.

Three kernels cover every production RMSNorm variant via comptime Bool switches:

  1. rmsnorm_fwht_quant_row[cols, block, has_gamma, per_block]
     Single-lane: load → normalize → FWHT → quantize (row or block).
     has_gamma=False: no gamma multiplication.
     per_block=True: per-FWHT-block absmax scales instead of per-row.

  2. rmsnorm_dual_gamma_fwht_quant_row[cols, block]
     Dual-lane: two gamma vectors share one RMS reduction, produce two
     independent quantized outputs. Structurally distinct loop.

  3. rmsnorm_bf16_row[cols, has_gamma, has_residual]
     No FWHT, no quantization. Reloads from bf16 (no work buffer).
     has_gamma=False: output = x / rms(x).
     has_residual=True: dst += norm(src) * gamma (fused residual add).

Atomic stages are exposed for direct use where the composed kernels
don't fit (e.g. per-head norms with different loop structure).

Numerical conventions (standardized):
  - sqrt: simd_math.sqrt (never std.math.sqrt)
  - eps: Float32
  - inv_rms = 1.0 / sqrt(sum(x^2) / N + eps)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.fwht import fwht_block
from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr,
    rms_reduce_bf16, inv_rms_from_sum_sq, normalize_inplace,
)


# ============================================================================
# Atomic stages
# ============================================================================


@always_inline
def load_and_reduce[cols: Int, has_gamma: Bool](
    src: BF16Ptr, gamma: BF16Ptr, work: F32Ptr,
) -> Float32:
    """Load bf16 to f32 work buffer, accumulate sum(x^2).

    When has_gamma: work[k] = x[k] * gamma[k] (sum from raw x).
    Otherwise: work[k] = x[k].
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
# Composed kernel 1: single-lane FWHT + quantize
#
# Covers 3 former variants via 2 Bool switches:
#   [F, F] = rmsnorm_fwht_quantize_row         (smollm2)
#   [T, F] = rmsnorm_gamma_fwht_quantize_row   (gemma4 attn/router)
#   [T, T] = rmsnorm_gamma_fwht_per_block_...   (gemma4 lm head)
# ============================================================================


@always_inline
def rmsnorm_fwht_quant_row[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scales: F32Ptr, eps: Float32,
):
    """Load → RMSNorm [* gamma] → FWHT → quantize [per-row | per-block]."""
    var sum_sq = load_and_reduce[cols, has_gamma](src, gamma, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    normalize_inplace[cols](work, inv)
    fwht_rotate[cols, block](work)
    emit_quant[cols, block, per_block](work, qi, scales)


# ============================================================================
# Composed kernel 2: dual-lane FWHT + quantize
#
# Two gamma vectors share one RMS reduction, produce two independent
# quantized outputs. Used by FFN (pre_ffn_norm + pre_ffn_norm_2).
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
    """Load → RMSNorm * (gamma_a, gamma_b) → 2x FWHT → 2x per-row i8."""
    var sum_sq = load_and_reduce_dual[cols](src, gamma_a, gamma_b, work_a, work_b)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    normalize_inplace[cols](work_a, inv)
    normalize_inplace[cols](work_b, inv)
    fwht_rotate[cols, block](work_a)
    fwht_rotate[cols, block](work_b)
    emit_quant[cols, block, False](work_a, qi_a, scale_a)
    emit_quant[cols, block, False](work_b, qi_b, scale_b)


# ============================================================================
# Composed kernel 3: bf16 output (no FWHT, no quantize)
#
# Reloads from bf16 in pass 2 — no work buffer needed.
# Covers 3 former variants via 2 Bool switches:
#   [F, F] = rmsnorm_no_scale          (V-norm, router-norm)
#   [T, F] = rmsnorm with gamma        (kernel_ops.rmsnorm, per-head)
#   [T, T] = post_attn_norm_kernel     (norm + residual add)
# ============================================================================


@always_inline
def rmsnorm_bf16_row[cols: Int, has_gamma: Bool, has_residual: Bool](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    """RMSNorm → bf16 output.

    has_gamma=False: dst = src / rms(src). gamma ptr ignored.
    has_gamma=True, has_residual=False: dst = (src / rms(src)) * gamma.
    has_gamma=True, has_residual=True: dst += (src / rms(src)) * gamma.
    """
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce_bf16[cols](src)
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


def main():
    print("rmsnorm_compositions: 3 parameterized kernels")
    print()
    print("Atomic stages:")
    print("  rms_reduce            sum(x^2) reduction")
    print("  inv_rms_from_sum_sq   sum -> 1/sqrt(sum/N + eps)")
    print("  load_and_reduce       bf16 -> f32 + sum(x^2) [+ gamma]")
    print("  load_and_reduce_dual  bf16 -> 2x f32 + shared sum(x^2)")
    print("  normalize_inplace     work *= inv_rms")
    print("  fwht_rotate           block-diagonal FWHT")
    print("  emit_quant            i8 quantize (per-row or per-block)")
    print()
    print("Composed kernels:")
    print("  rmsnorm_fwht_quant_row[cols, block, has_gamma, per_block]")
    print("    [F,F] norm -> FWHT -> i8/row        (smollm2)")
    print("    [T,F] norm*g -> FWHT -> i8/row      (gemma4 attn/router)")
    print("    [T,T] norm*g -> FWHT -> i8/block    (gemma4 lm head)")
    print()
    print("  rmsnorm_dual_gamma_fwht_quant_row[cols, block]")
    print("    norm*(ga,gb) -> 2x FWHT -> 2x i8    (gemma4 FFN)")
    print()
    print("  rmsnorm_bf16_row[cols, has_gamma, has_residual]")
    print("    [F,F] norm -> bf16                   (V-norm)")
    print("    [T,F] norm*g -> bf16                 (standard rmsnorm)")
    print("    [T,T] dst += norm*g                  (post-attn residual)")
