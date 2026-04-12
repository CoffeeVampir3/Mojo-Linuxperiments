"""Shared math primitives and pointer type aliases."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import sqrt


comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]


# ============================================================================
# RMS reduction
# ============================================================================


@always_inline
def inv_rms_from_sum_sq(sum_sq: Float32, n: Int, eps: Float32) -> Float32:
    """Convert sum(x^2) to inverse RMS scalar."""
    return Float32(1.0) / sqrt[DType.float32, 1](sum_sq / Float32(n) + eps)


@always_inline
def rms_reduce_bf16[cols: Int](src: BF16Ptr) -> Float32:
    """Sum-of-squares reduction over bf16 input. Returns sum(x^2)."""
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        k += width
    return vsum.reduce_add()


@always_inline
def rms_reduce_f32[cols: Int](src: F32Ptr) -> Float32:
    """Sum-of-squares reduction over f32 buffer. Returns sum(x^2)."""
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]()
        vsum = x.fma(x, vsum)
        k += width
    return vsum.reduce_add()


# ============================================================================
# Normalization
# ============================================================================


@always_inline
def normalize_inplace[cols: Int](work: F32Ptr, inv_rms: Float32):
    """Multiply f32 work buffer by scalar inv_rms."""
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)
    var k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv)
        k += width


@always_inline
def rms_normalize_inplace[cols: Int](work: F32Ptr, eps: Float32):
    """In-place RMS normalization: work /= rms(work).

    Combined reduce + normalize for code that already has data in an f32 buffer
    (e.g. per-head K/V norm after bf16 load, Q prep after load + gamma).
    """
    var sum_sq = rms_reduce_f32[cols](work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    normalize_inplace[cols](work, inv)
