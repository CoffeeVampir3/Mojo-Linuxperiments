"""GELU-tanh activation + FWHT + per-block i8 quantization."""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32, sqrt, roundeven
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8


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
# Row kernel — one row: gelu_tanh(gate) * up -> FWHT -> per-block i8
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

    var k = 0
    while k + width <= cols:
        var g = (gate + k).load[width=width]().cast[DType.float32]()
        var u = (up + k).load[width=width]().cast[DType.float32]()
        (work + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    for b in range(num_blocks):
        fwht_block[block](work + b * block)
        scale_out[b] = absmax_quantize_i8[block](work + b * block, row_qi + b * block)