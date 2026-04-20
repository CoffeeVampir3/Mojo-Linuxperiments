"""Shared math primitives and pointer type aliases."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

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
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    """Largest power-of-two N in {1,2,4,8} with N*width <= cols."""
    comptime n = cols // width
    return 8 if n >= 8 else 4 if n >= 4 else 2 if n >= 2 else 1


@always_inline
def tree_reduce_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> Scalar[T]:
    """Pairwise-add accumulator bank into lane-0, then horizontal reduce."""
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


@always_inline
def rms_reduce_bf16[cols: Int](src: BF16Ptr) -> Float32:
    """Sum-of-squares reduction over bf16 input. Returns sum(x^2).

    Uses N independent FMA accumulators (N chosen by pick_port_unroll to
    saturate the pipeline without outrunning cols). The first chunk primes
    each accumulator via x*x, so the register bank enters uninitialized.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var x = (src + i * width).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var x = (src + k + i * width).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
        k += step
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        k += width
    return tree_reduce_accs(accs)


@always_inline
def rms_reduce_f32[cols: Int](src: F32Ptr) -> Float32:
    """Sum-of-squares reduction over f32 buffer. Returns sum(x^2)."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var x = (src + i * width).load[width=width]()
        accs[i] = x * x
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var x = (src + k + i * width).load[width=width]()
            accs[i] = x.fma(x, accs[i])
        k += step
    while k + width <= cols:
        var x = (src + k).load[width=width]()
        accs[0] = x.fma(x, accs[0])
        k += width
    return tree_reduce_accs(accs)


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
