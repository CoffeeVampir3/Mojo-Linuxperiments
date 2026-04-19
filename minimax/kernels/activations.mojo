"""Activation primitives for MiniMax M2.7.

sigmoid: 1 / (1 + exp(-x))       — used by the sigmoid router
silu:    x * sigmoid(x)           — SwiGLU expert activation
silu_mul: silu(gate) * up         — fused SwiGLU combination
"""

from simd_math import exp_f32


@always_inline
def sigmoid_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return 1.0 / (1.0 + exp_f32[width](-x))


@always_inline
def silu_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return x * sigmoid_f32(x)


@always_inline
def silu_mul[width: Int](
    gate: SIMD[DType.float32, width],
    up: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    return silu_f32(gate) * up
