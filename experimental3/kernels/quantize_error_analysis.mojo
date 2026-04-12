"""Quantization error analysis for GELU-tanh vs SiLU post-nonlinearity.

Investigates:
1. Distribution shape after nonlinearity * up -> FWHT (absmax/rms ratio)
2. Int8 utilization (how much of the [-127,127] range is actually used)
3. Sigma-clipping: does clipping to N*sigma before quantize reduce NRMSE?
4. Block size effect on the distribution
5. Direct comparison: GELU-tanh vs SiLU at same dimensions
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32, sqrt, roundeven
from experimental3.kernels.fwht import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8


# ============================================================================
# Nonlinearities
# ============================================================================


@always_inline
def silu_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    var sig = 1.0 / (1.0 + exp_f32[width](-x))
    return x * sig


@always_inline
def gelu_tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    var inner = Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)
    var sig = 1.0 / (1.0 + exp_f32[width](-2.0 * inner))
    var t = 2.0 * sig - 1.0
    return 0.5 * x * (1.0 + t)


# ============================================================================
# PRNG
# ============================================================================


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


# ============================================================================
# Analysis for one configuration
# ============================================================================


def analyze[cols: Int, block: Int, use_gelu: Bool]():
    comptime width = simd_width_of[DType.float32]()
    var rng = UInt64(0xDEADBEEFCAFE1234)
    comptime label = "gelu_tanh" if use_gelu else "silu"

    var gate = alloc[Scalar[DType.bfloat16]](cols)
    var up = alloc[Scalar[DType.bfloat16]](cols)
    for i in range(cols):
        gate[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        up[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

    # Compute nonlinearity * up -> f32 work
    var work = alloc[Float32](cols)
    var k = 0
    while k + width <= cols:
        var g = (gate + k).load[width=width]().cast[DType.float32]()
        var u = (up + k).load[width=width]().cast[DType.float32]()
        comptime if use_gelu:
            (work + k).store(gelu_tanh_f32[width](g) * u)
        else:
            (work + k).store(silu_f32[width](g) * u)
        k += width

    # Pre-FWHT distribution stats
    var pre_absmax = Float32(0)
    var pre_sum_sq = Float64(0)
    for i in range(cols):
        var a = work[i].__abs__()
        if a > pre_absmax:
            pre_absmax = a
        pre_sum_sq += Float64(work[i]) * Float64(work[i])
    var pre_rms = Float64(sqrt[DType.float64, 1](pre_sum_sq / Float64(cols)))
    print("  pre-FWHT:  absmax=" + String(pre_absmax)
        + "  rms=" + String(pre_rms)
        + "  absmax/rms=" + String(Float64(pre_absmax) / pre_rms))

    # Save pre-FWHT for round-trip comparison
    var original = alloc[Float64](cols)
    for i in range(cols):
        original[i] = Float64(work[i])

    # Block-diagonal FWHT
    for b in range(cols // block):
        fwht_block[block](work + b * block)

    # Post-FWHT distribution stats
    var post_absmax = Float32(0)
    var post_sum_sq = Float64(0)
    var post_sum_abs = Float64(0)
    for i in range(cols):
        var a = work[i].__abs__()
        if a > post_absmax:
            post_absmax = a
        post_sum_sq += Float64(work[i]) * Float64(work[i])
        post_sum_abs += Float64(a)
    var post_rms = Float64(sqrt[DType.float64, 1](post_sum_sq / Float64(cols)))
    var post_mean_abs = post_sum_abs / Float64(cols)
    print("  post-FWHT: absmax=" + String(post_absmax)
        + "  rms=" + String(post_rms)
        + "  absmax/rms=" + String(Float64(post_absmax) / post_rms))

    # Quantize step size
    var step = Float64(post_absmax) * 2.0 / 255.0
    var theoretical_rmse = step / Float64(sqrt[DType.float64, 1](Float64(12.0)))
    print("  quant step=" + String(step) + "  theoretical_rmse=" + String(theoretical_rmse))

    # ---- Absmax quantize ----
    var qi = alloc[Scalar[DType.int8]](cols)
    var scale = absmax_quantize_i8[cols](work, qi)

    # Int8 utilization: what fraction of range is used?
    var qi_absmax = Int(0)
    var qi_sum_sq = Float64(0)
    var qi_count_near_zero = Int(0)
    for i in range(cols):
        var qv = Int(qi[i])
        if qv < 0:
            qv = -qv
        if qv > qi_absmax:
            qi_absmax = qv
        qi_sum_sq += Float64(qi[i]) * Float64(qi[i])
        if qv < 10:
            qi_count_near_zero += 1
    var qi_rms = Float64(sqrt[DType.float64, 1](qi_sum_sq / Float64(cols)))
    print("  i8 utilization: qi_absmax=" + String(qi_absmax)
        + "  qi_rms=" + String(qi_rms) + "/127"
        + "  near_zero(<10)=" + String(qi_count_near_zero) + "/" + String(cols)
        + " (" + String(Float64(qi_count_near_zero) / Float64(cols) * 100.0) + "%)")

    # Round-trip error with absmax
    var dq_scale = Float64(scale) / 127.0
    var sum_sq_err = Float64(0)
    var sum_sq_orig = Float64(0)
    var max_abs_err = Float64(0)
    for i in range(cols):
        var dequant = Float64(Int64(qi[i])) * dq_scale
        # Inverse FWHT to get back to original domain would be needed for true
        # comparison, but for error analysis the Hadamard-domain error equals
        # original-domain error (Parseval).
        var err = (Float64(work[i]) - dequant).__abs__()
        if err > max_abs_err:
            max_abs_err = err
        sum_sq_err += (Float64(work[i]) - dequant) * (Float64(work[i]) - dequant)
        sum_sq_orig += Float64(work[i]) * Float64(work[i])
    var nrmse_absmax = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_orig))
    print("  absmax quantize: NRMSE=" + String(nrmse_absmax)
        + "  max_err=" + String(max_abs_err))

    # ---- Sigma-clipping: try 3σ, 4σ, 5σ ----
    for sigma_mult in range(3, 6):
        var clip_scale = post_rms * Float64(sigma_mult)
        if clip_scale < Float64(post_absmax):
            var clip_inv = Float32(127.0 / Float32(clip_scale))
            comptime lo = SIMD[DType.float32, width](-128.0)
            comptime hi = SIMD[DType.float32, width](127.0)
            var clip_sum_sq_err = Float64(0)
            var clip_num_clipped = Int(0)
            k = 0
            while k + width <= cols:
                var v = (work + k).load[width=width]()
                var quantized = min(max(roundeven(v * clip_inv), lo), hi)
                var dequantized = quantized * (Float32(clip_scale) / 127.0)
                for j in range(width):
                    var err = Float64(v[j]) - Float64(dequantized[j])
                    clip_sum_sq_err += err * err
                    if quantized[j] == Float32(127.0) or quantized[j] == Float32(-128.0):
                        clip_num_clipped += 1
                k += width
            var clip_nrmse = Float64(sqrt[DType.float64, 1](clip_sum_sq_err / sum_sq_orig))
            print("  " + String(sigma_mult) + "σ clip: NRMSE=" + String(clip_nrmse)
                + "  clipped=" + String(clip_num_clipped) + "/" + String(cols)
                + " (" + String(Float64(clip_num_clipped) / Float64(cols) * 100.0) + "%)")

    gate.free()
    up.free()
    work.free()
    original.free()
    qi.free()


def main():
    print("=== quantization error analysis ===")

    print("\n--- GELU-tanh, cols=704, block=64 (expert) ---")
    analyze[704, 64, True]()

    print("\n--- SiLU, cols=704, block=64 (comparison) ---")
    analyze[704, 64, False]()

    print("\n--- GELU-tanh, cols=2112, block=64 (dense) ---")
    analyze[2112, 64, True]()

    print("\n--- SiLU, cols=2112, block=64 (comparison) ---")
    analyze[2112, 64, False]()

    print("\n--- GELU-tanh, cols=256, block=256 (baseline) ---")
    analyze[256, 256, True]()

    print("\n--- SiLU, cols=256, block=256 (comparison) ---")
    analyze[256, 256, False]()
