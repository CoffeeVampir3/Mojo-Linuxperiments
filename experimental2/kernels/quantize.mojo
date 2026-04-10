"""Shared i8 quantization primitives — dynamic absmax and fixed-scale."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import roundeven


@always_inline
def absmax_quantize_i8[cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> Float32:
    """Compute absmax of f32 buffer, quantize to i8 with dynamic scale.

    Returns the absmax value for the caller to store as needed.
    """
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % simd_width_of[DType.float32]() == 0, "cols must be a multiple of f32 SIMD width")

    var vmax = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        vmax = max(vmax, (src + k).load[width=width]().__abs__())
        k += width
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= cols:
        var v = (src + k).load[width=width]()
        (dst + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
        k += width

    return absmax


@always_inline
def fixed_quantize_i8[cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    quant_inv: Float32,
):
    """Quantize f32 buffer to i8 with pre-computed scale (127/S)."""
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % simd_width_of[DType.float32]() == 0, "cols must be a multiple of f32 SIMD width")
    var vinv = SIMD[DType.float32, width](quant_inv)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    var k = 0
    while k + width <= cols:
        var v = (src + k).load[width=width]()
        (dst + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
        k += width
