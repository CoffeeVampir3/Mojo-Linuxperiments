"""SIMD math primitives — no libc, all LLVM intrinsics or manual SIMD.

These are the atomic math operations used across the quantization pipeline
and inference kernels. Each is fully vectorized and branchless.
"""

from std.memory import UnsafePointer
from std.sys import llvm_intrinsic


# =============================================================================
# Type casting
# =============================================================================


@always_inline
def bf16_to_f32[width: Int](
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], offset: Int,
) -> SIMD[DType.float32, width]:
    """bf16 -> f32 via zero-extend + shift (vpmovzxwd + vpslld $16).

    The Mojo compiler lowers SIMD[bf16,N].cast[float32]() to scalar
    extract/shift/insert sequences. bf16 is f32 with the low 16 bits
    truncated, so reinterpreting as uint16 -> zero-extending to uint32
    -> shifting left 16 produces the identical IEEE 754 f32 bit pattern.
    """
    var raw = (ptr + offset).bitcast[Scalar[DType.uint16]]().load[width=width]()
    var wide = raw.cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=width]()


@always_inline
def bf16_load_as[target: DType, width: Int](
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], offset: Int,
) -> SIMD[target, width]:
    """Load bf16 and convert to any target dtype via f32 intermediate.

    Always goes through the bf16->f32 bit-shift path to avoid the
    compiler's scalarized cast lowering. The f32->target cast is a
    single SIMD instruction for all standard types.
    """
    var f32 = bf16_to_f32[width](ptr, offset)
    comptime if target == DType.float32:
        return rebind[SIMD[target, width]](f32)
    else:
        return f32.cast[target]()


# =============================================================================
# Rounding
# =============================================================================


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Round to nearest even — lowers to vroundps/vrndscaleps (f32) or
    vroundpd/vrndscalepd (f64)."""
    return llvm_intrinsic[
        "llvm.nearbyint",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


# =============================================================================
# Trigonometry
# =============================================================================


@fieldwise_init
struct SinCosResult[width: Int = 1]:
    var sin_val: SIMD[DType.float64, Self.width]
    var cos_val: SIMD[DType.float64, Self.width]


def sincos[width: Int = 1](angles: SIMD[DType.float64, width]) -> SinCosResult[width]:
    """sin/cos via Chebyshev minimax polynomials. SIMD-native, no libc.
    Degree-7 sin / degree-8 cos on [0, pi/2], Horner form."""
    comptime HALF_PI = Float64(1.57079632679489661923)
    comptime TWO_PI = Float64(6.28318530717958647692)
    comptime INV_TWO_PI = Float64(0.15915494309189533577)

    var x = angles - TWO_PI * (angles * INV_TWO_PI).cast[DType.int64]().cast[DType.float64]()
    var neg = x.to_bits() >> 63
    x = x + TWO_PI * neg.cast[DType.float64]()

    var quad = (x / HALF_PI).cast[DType.int64]()
    var under4 = ((quad - 4) >> 63) & 1
    quad = quad * under4 + SIMD[DType.int64, width](3) * (1 - under4)
    var r = x - quad.cast[DType.float64]() * HALF_PI

    var r2 = r * r
    var sin_r = r * (0.9999992413456921 + r2 * (
        -0.1666567961884791 + r2 * (
        0.008313225079910211 + r2 * (
        -0.0001852344833019604))))
    var cos_r = 0.9999999532476077 + r2 * (
        -0.4999990506281070 + r2 * (
        0.04166357893069784 + r2 * (
        -0.001385366693303192 + r2 * (
        0.00002315317415552132))))

    var swap = (quad & 1).cast[DType.float64]()
    var s_base = sin_r + swap * (cos_r - sin_r)
    var c_base = cos_r + swap * (sin_r - cos_r)
    var sin_sign = 1.0 - 2.0 * (quad >> 1).cast[DType.float64]()
    var cos_sign = 1.0 - 2.0 * ((quad & 1) ^ (quad >> 1)).cast[DType.float64]()

    return SinCosResult[width](s_base * sin_sign, c_base * cos_sign)


# =============================================================================
# Exponential
# =============================================================================


def exp_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Fast exp(x) for f32 via Cody-Waite range reduction + Chebyshev minimax
    polynomial. Fully branchless, SIMD-native."""
    comptime LN2_HI = Float32(0.693145751953125)
    comptime LN2_LO = Float32(1.4286068203094172e-06)
    comptime INV_LN2 = Float32(1.4426950408889634)
    comptime EXP_LO = Float32(-87.0)
    comptime EXP_HI = Float32(88.0)

    var lo_mask = ((x - EXP_LO).to_bits() >> 31) & 1
    var xc = x * (1 - lo_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_LO) * lo_mask.cast[DType.float32]()
    var hi_mask = ((EXP_HI - xc).to_bits() >> 31) & 1
    xc = xc * (1 - hi_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_HI) * hi_mask.cast[DType.float32]()

    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()

    var nf = n.cast[DType.float32]()
    var r = (xc - nf * LN2_HI) - nf * LN2_LO

    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999999995) + r * (
        Float32(0.5000000004) + r * (
        Float32(0.1666666456) + r * (
        Float32(0.04166685110) + r * (
        Float32(0.008333621758) + r * (
        Float32(0.001389404636)))))))

    var pow2n = SIMD[DType.float32, width](
        from_bits=(n + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n
