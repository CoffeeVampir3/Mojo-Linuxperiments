"""GELU-tanh + FWHT + dynamic-scale i8 quantization (Gemma 4 MLP domain exit).

Per row: bf16 gate, bf16 up -> f32 GELU_tanh(gate) * up -> block-diagonal FWHT
-> DC correction (element 0 × 0.5) -> per-block absmax -> quantize with dynamic scale.

Used as an inline building block by the expert kernel (moe.mojo) and the fused
dense gate_up kernel (int8_gemv.fused_gu_gelu_tanh). Not dispatched standalone.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32, sqrt, roundeven
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8


# ============================================================================
# GELU-tanh approximation (vectorized)
# ============================================================================


@always_inline
def tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Tanh via sigmoid: tanh(x) = 2 * sigmoid(2x) - 1."""
    var sig = 1.0 / (1.0 + exp_f32[width](-2.0 * x))
    return 2.0 * sig - 1.0


@always_inline
def gelu_tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """GELU with tanh approximation."""
    var inner = Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)
    return 0.5 * x * (1.0 + tanh_f32(inner))


# ============================================================================
# Row kernel — one row: gelu_tanh(gate) * up -> FWHT -> DC -> per-block i8
# ============================================================================


def gelu_tanh_fwht_quantize_row[cols: Int, block: Int](
    gate: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    up: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
):
    """One row: gelu_tanh(gate) * up -> FWHT -> per-block dynamic i8.

    scale_out must have space for cols // block f32 values (one per FWHT block).
    """
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block
    comptime DC_SCALE = Float32(0.5)

    var k = 0
    while k + width <= cols:
        var g = (gate + k).load[width=width]().cast[DType.float32]()
        var u = (up + k).load[width=width]().cast[DType.float32]()
        (work + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    for b in range(num_blocks):
        fwht_block[block](work + b * block)
        work[b * block] *= DC_SCALE
        scale_out[b] = absmax_quantize_i8[block](work + b * block, row_qi + b * block)


# ============================================================================
# Validation
# ============================================================================


def scalar_gelu_tanh_f64(x: Float64) -> Float64:
    var inner = Float64(0.7978845608028654) * (x + Float64(0.044715) * x * x * x)
    var neg2inner = Float64(-2.0) * inner
    var e = Float64(exp_f32[1](Float32(neg2inner)))
    var t = (Float64(1.0) - e) / (Float64(1.0) + e)
    return Float64(0.5) * x * (Float64(1.0) + t)


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
    var sc = Float64(1.0) / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] = buf[i] * sc


def cosine_sim_f64(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
) -> Float64:
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (Float64(sqrt[DType.float64, 1](na)) * Float64(sqrt[DType.float64, 1](nb)))


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * Float64(4.0) - Float64(2.0)


def error_stats_f64(
    expected: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    actual: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    n: Int,
    label: String,
):
    var cos = cosine_sim_f64(expected.bitcast[Float64](), actual.bitcast[Float64](), n)
    var max_abs = Float64(0)
    var sum_sq_err = Float64(0)
    var sum_sq_ref = Float64(0)
    for i in range(n):
        var err = (actual[i] - expected[i]).__abs__()
        if err > max_abs:
            max_abs = err
        sum_sq_err += (actual[i] - expected[i]) * (actual[i] - expected[i])
        sum_sq_ref += expected[i] * expected[i]
    var rmse = Float64(sqrt[DType.float64, 1](sum_sq_err / Float64(n)))
    var nrmse = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_ref)) if sum_sq_ref > 0 else Float64(0)
    print("  " + label + ":")
    print("    cosine:     " + String(cos))
    print("    max_abs:    " + String(max_abs))
    print("    RMSE:       " + String(rmse))
    print("    NRMSE:      " + String(nrmse))


def validate[cols: Int, block: Int]():
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block
    var rng = UInt64(0xDEADBEEFCAFE1234)

    var gate_bf16 = alloc[Scalar[DType.bfloat16]](cols)
    var up_bf16 = alloc[Scalar[DType.bfloat16]](cols)
    for i in range(cols):
        gate_bf16[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        up_bf16[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

    # f64 reference: gelu_tanh(gate) * up
    var expected_f64 = alloc[Scalar[DType.float64]](cols)
    for i in range(cols):
        expected_f64[i] = scalar_gelu_tanh_f64(Float64(gate_bf16[i])) * Float64(up_bf16[i])

    # gelu_tanh*up f32 vs f64
    var work = alloc[Float32](cols)
    var k = 0
    while k + width <= cols:
        var g = (gate_bf16 + k).load[width=width]().cast[DType.float32]()
        var u = (up_bf16 + k).load[width=width]().cast[DType.float32]()
        (work + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    var kernel_f64 = alloc[Scalar[DType.float64]](cols)
    for i in range(cols):
        kernel_f64[i] = Float64(work[i])
    error_stats_f64(expected_f64, kernel_f64, cols, "gelu_tanh*up (f32 kernel vs f64 ref)")

    # FWHT + DC correction + per-block quantize round-trip
    var qi_blk = alloc[Scalar[DType.int8]](cols)
    var blk_scales = alloc[Float32](num_blocks)
    var work2 = alloc[Float32](cols)
    gelu_tanh_fwht_quantize_row[cols, block](gate_bf16, up_bf16, qi_blk, work2, blk_scales)

    comptime DC_INV = Float64(1.0 / 0.5)
    var recovered_blk = alloc[Scalar[DType.float64]](cols)
    for b in range(num_blocks):
        var dq_b = Float64(blk_scales[b]) / 127.0
        recovered_blk[b * block] = Float64(Int64(qi_blk[b * block])) * dq_b * DC_INV
        for j in range(1, block):
            recovered_blk[b * block + j] = Float64(Int64(qi_blk[b * block + j])) * dq_b
    for b in range(num_blocks):
        scalar_fwht_f64(recovered_blk.bitcast[Float64]() + b * block, block)

    var min_scale = Float32(1e30)
    var max_scale = Float32(0)
    for b in range(num_blocks):
        if blk_scales[b] < min_scale:
            min_scale = blk_scales[b]
        if blk_scales[b] > max_scale:
            max_scale = blk_scales[b]
    print("  per-block scales: min=" + String(min_scale) + " max=" + String(max_scale)
        + " range=" + String(Float64(max_scale) / Float64(min_scale)) + "x"
        + " (" + String(num_blocks) + " blocks)")
    error_stats_f64(expected_f64, recovered_blk, cols, "per-block + DC correction round-trip")

    gate_bf16.free()
    up_bf16.free()
    expected_f64.free()
    work.free()
    kernel_f64.free()
    qi_blk.free()
    blk_scales.free()
    work2.free()
    recovered_blk.free()


def main():
    print("=== gelu_tanh_fwht_quantize validation ===")

    print("\ncols=704 (expert MOE_INTERMEDIATE), block=64:")
    validate[704, 64]()

    print("\ncols=2112 (dense INTERMEDIATE), block=64:")
    validate[2112, 64]()
