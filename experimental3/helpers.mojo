"""Gemma 4 attention helpers — Q prep with per-head norm, partial RoPE variant.

Extends experimental2/helpers.mojo prep_q_row with:
  - Per-head RMS-divide before RoPE
  - Partial RoPE for full-attention layers (128 of 512 dims rotated)
  - Runtime q_norm gamma application
  - Scale = 1.0 (no inv_sqrt_hd — QK norms handle scaling)

The entire pipeline stays in registers: load bf16 → f32 regs → /rms →
RoPE → FWHT → absmax → quantize → i8 output. No work buffer needed.
"""

from std.memory import UnsafePointer
from std.collections import InlineArray

from simd_math import sqrt, quantize_i8
from experimental3.kernels.fwht import fwht_apply, fwht_width


# ============================================================================
# prep_q_row_normed_impl — shared Q prep with configurable RoPE span
# ============================================================================


@always_inline
def prep_q_row_normed_impl[head_dim: Int, rope_dims: Int](
    q_row: UnsafePointer[BFloat16, MutAnyOrigin],
    q_norm: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
    eps: Float32,
) -> Tuple[Float32, Float32]:
    """Per-head: /rms → *q_norm → RoPE prefix → FWHT → dynamic quantize.

    Returns (qi_bias, absmax).
    """
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    comptime fwht_regs = head_dim // fwht_w
    comptime half_regs = fwht_regs // 2
    comptime rope_half = rope_dims // 2
    comptime rope_regs = rope_half // fwht_w
    debug_assert(head_dim % 2 == 0, "head_dim must be even")
    debug_assert(rope_dims <= head_dim, "rope_dims must not exceed head_dim")
    debug_assert(rope_dims % 2 == 0, "rope_dims must be even")
    debug_assert(head_dim % fwht_w == 0, "head_dim must be a multiple of the FWHT SIMD width")
    debug_assert(
        (head_dim // 2) % fwht_w == 0,
        "head_dim / 2 must be a multiple of the FWHT SIMD width",
    )
    debug_assert(
        rope_half % fwht_w == 0,
        "rope_dims / 2 must be a multiple of the FWHT SIMD width",
    )

    # Load bf16 → f32 registers
    var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
        fill=SIMD[DType.float32, fwht_w](0))
    comptime for ri in range(fwht_regs):
        r[ri] = (q_row + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()

    # Per-head RMS-divide
    var vsum = SIMD[DType.float32, fwht_w](0)
    comptime for ri in range(fwht_regs):
        vsum = r[ri].fma(r[ri], vsum)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(head_dim) + eps)
    comptime for ri in range(fwht_regs):
        var gamma = (q_norm + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()
        r[ri] = (r[ri] * inv_rms) * gamma

    # Apply RoPE to the configured prefix of register pairs.
    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[half_regs + ri]
        var cv = (cos_row + ri * fwht_w).load[width=fwht_w]()
        var sv = (sin_row + ri * fwht_w).load[width=fwht_w]()
        r[ri] = x_lo * cv - x_hi * sv
        r[half_regs + ri] = x_hi * cv + x_lo * sv
    # Remaining register pairs are untouched when rope_dims < head_dim.

    # FWHT on full head_dim (all registers participate)
    fwht_apply[DType.float32, head_dim](r)

    # Dynamic absmax
    var vmax = SIMD[DType.float32, fwht_w](0)
    comptime for ri in range(fwht_regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    # Quantize + bias accumulation
    var vq_inv = SIMD[DType.float32, fwht_w](Float32(127.0) / absmax)
    var q_sum_acc = SIMD[DType.int32, fwht_w](0)
    comptime for ri in range(fwht_regs):
        var qi = quantize_i8[fwht_w](r[ri], vq_inv)
        (qi_out + ri * fwht_w).store(qi)
        q_sum_acc += qi.cast[DType.int32]()

    var qi_bias = Float32(q_sum_acc.reduce_add()) * 128.0
    return (qi_bias, absmax)


# ============================================================================
# prep_q_row_normed — full RoPE, with per-head RMS-divide
# ============================================================================


@always_inline
def prep_q_row_normed[head_dim: Int](
    q_row: UnsafePointer[BFloat16, MutAnyOrigin],
    q_norm: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
    eps: Float32,
) -> Tuple[Float32, Float32]:
    return prep_q_row_normed_impl[head_dim, head_dim](
        q_row, q_norm, cos_row, sin_row, qi_out, eps)


# ============================================================================
# prep_q_row_normed_partial — partial RoPE for full-attention layers
# ============================================================================


@always_inline
def prep_q_row_normed_partial[head_dim: Int, rope_dims: Int](
    q_row: UnsafePointer[BFloat16, MutAnyOrigin],
    q_norm: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
    eps: Float32,
) -> Tuple[Float32, Float32]:
    return prep_q_row_normed_impl[head_dim, rope_dims](
        q_row, q_norm, cos_row, sin_row, qi_out, eps)
