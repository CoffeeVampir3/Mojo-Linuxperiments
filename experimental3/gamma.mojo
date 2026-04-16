"""Gamma split utilities for FWHT quantization."""

from std.sys.info import simd_width_of

from simd_math import sqrt

from experimental3.common_math import F32Ptr, BF16Ptr


@always_inline
def load_sqrt_abs_gamma_chunk[width: Int](
    raw: BF16Ptr,
) -> Tuple[SIMD[DType.float32, width], SIMD[DType.float32, width]]:
    comptime abs_floor = SIMD[DType.float32, width](1e-10)
    var gamma = raw.load[width=width]().cast[DType.float32]()
    var sqrt_abs_gamma = sqrt[DType.float32, width](max(gamma.__abs__(), abs_floor))
    return (gamma, sqrt_abs_gamma)


@always_inline
def compute_sqrt_gamma[cols: Int](raw: BF16Ptr, dst: BF16Ptr):
    """Stabilized sign-preserving sqrt: dst[k] = sign(g[k]) * sqrt(max(|g[k]|, 1e-10)).

    Output is bf16 for use as the activation-side gamma in rmsnorm kernels.
    """
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var k = 0
    while k < cols:
        var gamma, sqrt_abs_gamma = load_sqrt_abs_gamma_chunk[width](raw + k)
        (dst + k).store(gamma.lt(0).select(-sqrt_abs_gamma, sqrt_abs_gamma).cast[DType.bfloat16]())
        k += width


@always_inline
def compute_sqrt_abs_gamma[cols: Int](raw: BF16Ptr, dst: F32Ptr):
    """Stabilized unsigned sqrt: dst[k] = sqrt(max(|g[k]|, 1e-10)).

    Output is f32 for use as the weight-side absorption factor.
    """
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var k = 0
    while k < cols:
        _, var sqrt_abs_gamma = load_sqrt_abs_gamma_chunk[width](raw + k)
        (dst + k).store(sqrt_abs_gamma)
        k += width


@always_inline
def compute_inv_sqrt_gamma[cols: Int](raw: BF16Ptr, dst: F32Ptr):
    """Stabilized inverse sqrt: dst[k] = 1 / sqrt(max(|g[k]|, 1e-10)).

    Output is f32 for use as the embed lookup undo factor.
    """
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % width == 0, "cols must be a multiple of f32 SIMD width")
    var one = SIMD[DType.float32, width](1.0)
    var k = 0
    while k < cols:
        _, var sqrt_abs_gamma = load_sqrt_abs_gamma_chunk[width](raw + k)
        (dst + k).store(one / sqrt_abs_gamma)
        k += width
