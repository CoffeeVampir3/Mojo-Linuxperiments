"""Gamma split utilities for FWHT quantization."""

from std.sys.info import simd_width_of

from simd_math import sqrt

from experimental3.common_math import F32Ptr, BF16Ptr


@always_inline
def compute_sqrt_gamma[cols: Int](raw: BF16Ptr, dst: BF16Ptr):
    """Sign-preserving sqrt: dst[k] = sign(g[k]) * sqrt(|g[k]|).

    Output is bf16 for use as the activation-side gamma in rmsnorm kernels.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime floor = SIMD[DType.float32, width](1e-10)
    var k = 0
    while k + width <= cols:
        var g = (raw + k).load[width=width]().cast[DType.float32]()
        var ag = max(g.__abs__(), floor)
        var s = sqrt[DType.float32, width](ag)
        var neg = g.lt(0)
        (dst + k).store(neg.select(-s, s).cast[DType.bfloat16]())
        k += width


@always_inline
def compute_sqrt_abs_gamma[cols: Int](raw: BF16Ptr, dst: F32Ptr):
    """Unsigned sqrt: dst[k] = sqrt(|g[k]|).

    Output is f32 for use as the weight-side absorption factor.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime floor = SIMD[DType.float32, width](1e-10)
    var k = 0
    while k + width <= cols:
        var g = (raw + k).load[width=width]().cast[DType.float32]()
        var ag = max(g.__abs__(), floor)
        (dst + k).store(sqrt[DType.float32, width](ag))
        k += width


@always_inline
def compute_inv_sqrt_gamma[cols: Int](raw: BF16Ptr, dst: F32Ptr):
    """Inverse sqrt: dst[k] = 1 / sqrt(|g[k]|).

    Output is f32 for use as the embed lookup undo factor.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime floor = SIMD[DType.float32, width](1e-10)
    var k = 0
    while k + width <= cols:
        var g = (raw + k).load[width=width]().cast[DType.float32]()
        var ag = max(g.__abs__(), floor)
        (dst + k).store(SIMD[DType.float32, width](1.0) / sqrt[DType.float32, width](ag))
        k += width
