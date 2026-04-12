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
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import sqrt, quantize_i8
from experimental3.kernels.fwht import fwht_apply, fwht_width


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
    """Per-head: /rms → *q_norm → full RoPE → FWHT → dynamic quantize.

    Returns (qi_bias, q_scale).
    """
    comptime half = head_dim // 2
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    comptime fwht_regs = head_dim // fwht_w
    comptime half_regs = fwht_regs // 2

    # Load bf16 → f32 registers
    var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
        fill=SIMD[DType.float32, fwht_w](0))
    for ri in range(fwht_regs):
        r[ri] = (q_row + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()

    # Per-head RMS-divide (in registers)
    var vsum = SIMD[DType.float32, fwht_w](0)
    for ri in range(fwht_regs):
        vsum = r[ri].fma(r[ri], vsum)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(head_dim) + eps)
    for ri in range(fwht_regs):
        r[ri] *= inv_rms

    # Apply q_norm gamma
    for ri in range(fwht_regs):
        r[ri] *= (q_norm + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()

    # Full RoPE: rotate all half-dim pairs
    for ri in range(half_regs):
        var x_lo = r[ri]
        var x_hi = r[half_regs + ri]
        var cv = (cos_row + ri * fwht_w).load[width=fwht_w]()
        var sv = (sin_row + ri * fwht_w).load[width=fwht_w]()
        r[ri] = x_lo * cv - x_hi * sv
        r[half_regs + ri] = x_hi * cv + x_lo * sv

    # FWHT (in-register butterflies)
    fwht_apply[DType.float32, head_dim](r)

    # Dynamic absmax
    var vmax = SIMD[DType.float32, fwht_w](0)
    for ri in range(fwht_regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    # Quantize + bias accumulation
    var vq_inv = SIMD[DType.float32, fwht_w](Float32(127) / absmax)
    var q_sum_acc = SIMD[DType.int32, fwht_w](0)
    for ri in range(fwht_regs):
        var qi = quantize_i8[fwht_w](r[ri], vq_inv)
        (qi_out + ri * fwht_w).store(qi)
        q_sum_acc += qi.cast[DType.int32]()

    var qi_bias = Float32(q_sum_acc.reduce_add()) * 128.0
    return (qi_bias, absmax)


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
    """Per-head: /rms → *q_norm → partial RoPE → FWHT → dynamic quantize.

    Returns (qi_bias, q_scale).
    """
    comptime half = head_dim // 2
    comptime rope_half = rope_dims // 2
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    comptime fwht_regs = head_dim // fwht_w
    comptime half_regs = fwht_regs // 2
    comptime rope_regs = rope_half // fwht_w

    # Load bf16 → f32 registers
    var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
        fill=SIMD[DType.float32, fwht_w](0))
    for ri in range(fwht_regs):
        r[ri] = (q_row + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()

    # Per-head RMS-divide
    var vsum = SIMD[DType.float32, fwht_w](0)
    for ri in range(fwht_regs):
        vsum = r[ri].fma(r[ri], vsum)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(head_dim) + eps)
    for ri in range(fwht_regs):
        r[ri] *= inv_rms

    # Apply q_norm gamma
    for ri in range(fwht_regs):
        r[ri] *= (q_norm + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()

    # Partial RoPE: only first rope_regs pairs in each half
    for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[half_regs + ri]
        var cv = (cos_row + ri * fwht_w).load[width=fwht_w]()
        var sv = (sin_row + ri * fwht_w).load[width=fwht_w]()
        r[ri] = x_lo * cv - x_hi * sv
        r[half_regs + ri] = x_hi * cv + x_lo * sv
    # Registers [rope_regs:half_regs] and [half_regs+rope_regs:fwht_regs]
    # are untouched — those dims pass through unchanged.

    # FWHT on full head_dim (all registers participate)
    fwht_apply[DType.float32, head_dim](r)

    # Dynamic absmax
    var vmax = SIMD[DType.float32, fwht_w](0)
    for ri in range(fwht_regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    # Quantize + bias accumulation
    var vq_inv = SIMD[DType.float32, fwht_w](Float32(127) / absmax)
    var q_sum_acc = SIMD[DType.int32, fwht_w](0)
    for ri in range(fwht_regs):
        var qi = quantize_i8[fwht_w](r[ri], vq_inv)
        (qi_out + ri * fwht_w).store(qi)
        q_sum_acc += qi.cast[DType.int32]()

    var qi_bias = Float32(q_sum_acc.reduce_add()) * 128.0
    return (qi_bias, absmax)


# ============================================================================
# Validation
# ============================================================================


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def scalar_rms_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var sum_sq = Float64(0)
    for i in range(n):
        sum_sq += buf[i] * buf[i]
    return Float64(sqrt[DType.float64, 1](sum_sq / Float64(n) + Float64(1e-6)))


def scalar_fwht_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var stride = 1
    while stride < n:
        var i = 0
        while i < n:
            for j in range(stride):
                var a = buf[i + j]
                var b = buf[i + j + stride]
                buf[i + j] = a + b
                buf[i + j + stride] = a - b
            i += stride * 2
        stride *= 2
    var sc = 1.0 / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] *= sc


def cosine_sim_f64(a: UnsafePointer[Float64, MutAnyOrigin], b: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (Float64(sqrt[DType.float64, 1](na)) * Float64(sqrt[DType.float64, 1](nb)))


def validate_prep_q_normed[head_dim: Int]():
    comptime half = head_dim // 2
    var rng = UInt64(0xDEADCAFE98765432)

    var src = alloc[Scalar[DType.bfloat16]](head_dim)
    var q_norm = alloc[Scalar[DType.bfloat16]](head_dim)
    var cos_f32 = alloc[Float32](half)
    var sin_f32 = alloc[Float32](half)

    for i in range(head_dim):
        src[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        q_norm[i] = Scalar[DType.bfloat16](Float32(1.0))
    for i in range(half):
        var angle = xorshift64(rng) * 0.5
        cos_f32[i] = Float32(1.0 - angle * angle * 0.5)
        sin_f32[i] = Float32(angle)

    # f64 reference: load -> /rms -> full rope -> fwht
    var expected = alloc[Float64](head_dim)
    for i in range(head_dim):
        expected[i] = Float64(src[i])
    var rms = scalar_rms_f64(expected.bitcast[Float64](), head_dim)
    for i in range(head_dim):
        expected[i] /= rms
    var pre_fwht = alloc[Float64](head_dim)
    for j in range(half):
        var lo = expected[j]
        var hi = expected[half + j]
        expected[j] = lo * Float64(cos_f32[j]) - hi * Float64(sin_f32[j])
        expected[half + j] = hi * Float64(cos_f32[j]) + lo * Float64(sin_f32[j])
    for i in range(head_dim):
        pre_fwht[i] = expected[i]
    scalar_fwht_f64(expected.bitcast[Float64](), head_dim)

    # Kernel
    var qi = alloc[Scalar[DType.int8]](head_dim)
    var result = prep_q_row_normed[head_dim](
        src.bitcast[BFloat16](), q_norm, cos_f32, sin_f32, qi.bitcast[Int8](), Float32(1e-6))
    var qi_bias = result[0]
    var q_scale = result[1]

    # Compare FWHT output via dequant round-trip
    var recovered = alloc[Float64](head_dim)
    var dq = Float64(q_scale) / 127.0
    for i in range(head_dim):
        recovered[i] = Float64(Int64(qi[i])) * dq
    scalar_fwht_f64(recovered.bitcast[Float64](), head_dim)

    # Round-trip: quantize → dequant → inverse FWHT → compare vs pre-FWHT reference
    var cos_rt = cosine_sim_f64(pre_fwht.bitcast[Float64](), recovered.bitcast[Float64](), head_dim)

    # Element-wise error stats on the recovered (original domain) values
    var max_abs_err = Float64(0)
    var sum_sq_err = Float64(0)
    var sum_sq_expected = Float64(0)
    for i in range(head_dim):
        var err = (recovered[i] - pre_fwht[i]).__abs__()
        if err > max_abs_err:
            max_abs_err = err
        sum_sq_err += (recovered[i] - pre_fwht[i]) * (recovered[i] - pre_fwht[i])
        sum_sq_expected += pre_fwht[i] * pre_fwht[i]
    var rmse = Float64(sqrt[DType.float64, 1](sum_sq_err / Float64(head_dim)))
    var nrmse = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_expected))

    # Bias check
    var sum_check = Int(0)
    for i in range(head_dim):
        sum_check += Int(qi[i])
    var expected_bias = Float32(sum_check) * 128.0
    var bias_err = Float64((qi_bias - expected_bias).__abs__())

    print("  round-trip cosine:   " + String(cos_rt))
    print("  max abs error:       " + String(max_abs_err))
    print("  RMSE:                " + String(rmse))
    print("  NRMSE:               " + String(nrmse))
    print("  q_scale (absmax):    " + String(q_scale))
    print("  qi_bias:             " + String(qi_bias) + "  (expected " + String(expected_bias) + ", err=" + String(bias_err) + ")")

    src.free()
    q_norm.free()
    cos_f32.free()
    sin_f32.free()
    expected.free()
    pre_fwht.free()
    qi.free()
    recovered.free()


def validate_prep_q_partial[head_dim: Int, rope_dims: Int]():
    comptime half = head_dim // 2
    comptime rope_half = rope_dims // 2
    var rng = UInt64(0xABCD5678CAFE1234)

    var src = alloc[Scalar[DType.bfloat16]](head_dim)
    var q_norm = alloc[Scalar[DType.bfloat16]](head_dim)
    var cos_f32 = alloc[Float32](rope_half)
    var sin_f32 = alloc[Float32](rope_half)

    for i in range(head_dim):
        src[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        q_norm[i] = Scalar[DType.bfloat16](Float32(1.0))
    for i in range(rope_half):
        var angle = xorshift64(rng) * 0.5
        cos_f32[i] = Float32(1.0 - angle * angle * 0.5)
        sin_f32[i] = Float32(angle)

    var expected = alloc[Float64](head_dim)
    for i in range(head_dim):
        expected[i] = Float64(src[i])
    var rms = scalar_rms_f64(expected.bitcast[Float64](), head_dim)
    for i in range(head_dim):
        expected[i] /= rms
    var pre_fwht = alloc[Float64](head_dim)
    for j in range(rope_half):
        var lo = expected[j]
        var hi = expected[half + j]
        expected[j] = lo * Float64(cos_f32[j]) - hi * Float64(sin_f32[j])
        expected[half + j] = hi * Float64(cos_f32[j]) + lo * Float64(sin_f32[j])
    for i in range(head_dim):
        pre_fwht[i] = expected[i]
    scalar_fwht_f64(expected.bitcast[Float64](), head_dim)

    var qi = alloc[Scalar[DType.int8]](head_dim)
    var result = prep_q_row_normed_partial[head_dim, rope_dims](
        src.bitcast[BFloat16](), q_norm, cos_f32, sin_f32, qi.bitcast[Int8](), Float32(1e-6))
    var qi_bias = result[0]
    var q_scale = result[1]

    var recovered = alloc[Float64](head_dim)
    var dq = Float64(q_scale) / 127.0
    for i in range(head_dim):
        recovered[i] = Float64(Int64(qi[i])) * dq
    scalar_fwht_f64(recovered.bitcast[Float64](), head_dim)

    var cos_rt = cosine_sim_f64(pre_fwht.bitcast[Float64](), recovered.bitcast[Float64](), head_dim)

    var max_abs_err = Float64(0)
    var sum_sq_err = Float64(0)
    var sum_sq_expected = Float64(0)
    for i in range(head_dim):
        var err = (recovered[i] - pre_fwht[i]).__abs__()
        if err > max_abs_err:
            max_abs_err = err
        sum_sq_err += (recovered[i] - pre_fwht[i]) * (recovered[i] - pre_fwht[i])
        sum_sq_expected += pre_fwht[i] * pre_fwht[i]
    var rmse = Float64(sqrt[DType.float64, 1](sum_sq_err / Float64(head_dim)))
    var nrmse = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_expected))

    var sum_check = Int(0)
    for i in range(head_dim):
        sum_check += Int(qi[i])
    var expected_bias = Float32(sum_check) * 128.0
    var bias_err = Float64((qi_bias - expected_bias).__abs__())

    print("  round-trip cosine:   " + String(cos_rt))
    print("  max abs error:       " + String(max_abs_err))
    print("  RMSE:                " + String(rmse))
    print("  NRMSE:               " + String(nrmse))
    print("  q_scale (absmax):    " + String(q_scale))
    print("  qi_bias:             " + String(qi_bias) + "  (expected " + String(expected_bias) + ", err=" + String(bias_err) + ")")

    src.free()
    q_norm.free()
    cos_f32.free()
    sin_f32.free()
    expected.free()
    pre_fwht.free()
    qi.free()
    recovered.free()


def main():
    print("=== prep_q_row_normed validation ===")

    print("\nFull RoPE, head_dim=256 (sliding):")
    validate_prep_q_normed[256]()

    print("\nPartial RoPE, head_dim=512, rope_dims=128 (full):")
    validate_prep_q_partial[512, 128]()
