"""GELU-tanh activation helpers."""

from simd_math import exp_f32


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
