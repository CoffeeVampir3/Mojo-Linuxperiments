from std.memory import UnsafePointer
from std.sys import llvm_intrinsic



# =============================================================================
# Square root
# =============================================================================


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """SIMD sqrt — lowers to vsqrtps (f32) or vsqrtpd (f64)."""
    return llvm_intrinsic[
        "llvm.sqrt",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


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
# Int8 quantization
# =============================================================================


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width], inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    """Absmax quantize f32 → i8: round(v * inv_scale), clamp [-128, 127].

    inv_scale = 127.0 / absmax (precomputed by caller).
    """
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()


@always_inline
def quantize_i8_scalar(v: Float32, inv_scale: Float32) -> Scalar[DType.int8]:
    """Scalar absmax quantize f32 → i8."""
    var q = roundeven[DType.float32, 1](v * inv_scale)
    return min(max(q, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()


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


def exp_f32_fast[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Fast exp(x) for i8 quantization pipelines (~0.3% relative error).

    Uses degree-3 polynomial with Cody-Waite reduction. Sufficient
    for softmax outputs that will be quantized to 127 levels.
    Inputs should be <= 0 (softmax: x - max). For x < -87, returns 0.
    ~8 SIMD ops vs ~23 for exp_f32.
    """
    comptime LN2 = Float32(0.6931471805599453)
    comptime INV_LN2 = Float32(1.4426950408889634)

    # Clamp input to avoid i32 overflow in range reduction
    var xc = max(x, SIMD[DType.float32, width](-87.0))

    # n = round(x / ln2), r = x - n*ln2, exp(x) = 2^n * exp(r)
    var xn = xc * INV_LN2
    var n = roundeven(xn).cast[DType.int32]()
    var r = xc - n.cast[DType.float32]() * LN2

    # Degree-3 minimax on [-ln2/2, ln2/2]: max error ~3e-4
    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999) + r * (
        Float32(0.4985) + r * (
        Float32(0.1681))))

    # 2^n via IEEE exponent manipulation, clamp to avoid denormals
    var n_clamped = max(n, SIMD[DType.int32, width](-126))
    var pow2n = SIMD[DType.float32, width](
        from_bits=(n_clamped + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n


# =============================================================================
# Logarithm
# =============================================================================


def log_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """ln(x) for f32, x > 0. Bit-split + atanh series, branchless.

    Splits x = m * 2^e with m ∈ [1, 2), centers m to [sqrt(2)/2, sqrt(2)) by
    halving when m > sqrt(2), then uses log(m) = 2·atanh((m-1)/(m+1)) as a
    degree-11 odd polynomial in z = (m-1)/(m+1). See validate_log_f32.mojo.
    """
    comptime LN2 = Float32(0.6931471805599453)
    comptime SQRT2 = Float32(1.4142135623730951)

    var bits = x.to_bits()
    var e = (bits >> 23).cast[DType.int32]() - 127
    var m_bits = (bits & 0x007FFFFF) | 0x3F800000
    var m = SIMD[DType.float32, width](from_bits=m_bits)

    # Center: if m > sqrt(2), m *= 0.5 and e += 1. Sign-bit of (SQRT2 - m)
    # is 1 iff m > SQRT2.
    var big = (SIMD[DType.float32, width](SQRT2) - m).to_bits() >> 31
    var big_f = big.cast[DType.float32]()
    m = m * (1.0 - 0.5 * big_f)
    e = e + big.cast[DType.int32]()

    var z = (m - 1.0) / (m + 1.0)
    var z2 = z * z
    var t = SIMD[DType.float32, width](Float32(1.0 / 11.0))
    t = Float32(1.0 / 9.0) + z2 * t
    t = Float32(1.0 / 7.0) + z2 * t
    t = Float32(1.0 / 5.0) + z2 * t
    t = Float32(1.0 / 3.0) + z2 * t
    t = Float32(1.0) + z2 * t

    return 2.0 * z * t + e.cast[DType.float32]() * LN2
