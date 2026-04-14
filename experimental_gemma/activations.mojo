"""GELU-tanh activation for Gemma4 dense and expert MLPs.

Gemma4 uses GELU with tanh approximation (not SiLU/SwiGLU):
  gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

The fused gelu_tanh_mul computes: dst = gelu_tanh(gate) * up
"""

from std.sys.info import simd_width_of

from modeling.model_spec import Encoding, Shaped, DynView
from simd_math import exp_f32


@always_inline
def tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """tanh via sigmoid: tanh(x) = 2 * sigmoid(2x) - 1."""
    var sig = 1.0 / (1.0 + exp_f32[width](-2.0 * x))
    return 2.0 * sig - 1.0


@always_inline
def gelu_tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    comptime SQRT_2_OVER_PI = Float32(0.7978845608028654)
    comptime COEFF = Float32(0.044715)
    var inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
    return 0.5 * x * (1.0 + tanh_f32(inner))


def gelu_tanh_mul[GT: Encoding & Shaped, UT: Encoding & Shaped, DstT: Encoding & Shaped](
    gate: DynView[GT], up: DynView[UT], dst: DynView[DstT],
):
    """dst = gelu_tanh(gate) * up. F32 compute, bf16 I/O."""
    comptime assert GT.DTYPE == DType.bfloat16, "gelu_tanh_mul: gate must be bf16"
    comptime assert UT.DTYPE == DType.bfloat16, "gelu_tanh_mul: up must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "gelu_tanh_mul: dst must be bf16"
    comptime assert GT.COLS == UT.COLS, "gelu_tanh_mul: gate/up cols mismatch"
    comptime assert GT.COLS == DstT.COLS, "gelu_tanh_mul: gate/dst cols mismatch"
    comptime assert GT.COLS % simd_width_of[DType.float32]() == 0, "gelu_tanh_mul: cols must be f32-simd-aligned"

    var seq_len = gate.seq_len
    if seq_len == 0:
        return

    var gp = gate.as_ptr[DType.bfloat16]()
    var up_ = up.as_ptr[DType.bfloat16]()
    var dp = dst.as_ptr[DType.bfloat16]()
    comptime cols = GT.COLS
    comptime width = simd_width_of[DType.float32]()

    for i in range(0, seq_len * cols, width):
        var g = (gp + i).load[width=width]().cast[DType.float32]()
        var u = (up_ + i).load[width=width]().cast[DType.float32]()
        (dp + i).store((gelu_tanh_f32(g) * u).cast[DType.bfloat16]())
