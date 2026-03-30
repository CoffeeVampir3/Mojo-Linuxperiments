"""Atomic quantization operations and composed recipes.

Ops are standalone functions parameterized over precision (P) and target
(T) types. QuantContext[P, T] carries typed buffer pointers and dimensions.
Recipes compose ops into complete strategies — the types flow from the
caller (typically the engine).

The ops handle math only — no I/O, no file formats, no allocation.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.math import sqrt

from simd_math import roundeven


# =============================================================================
# Quantization context — parameterized over working precision and output type
# =============================================================================


struct QuantContext[precision: DType, target: DType]:
    """Buffers and dimensions for one tensor being quantized.

    Owned by the engine. Ops read/write through these pointers.
    The work buffer is the primary surface — source data is loaded
    into it (e.g. bf16→P) before any ops run.
    """
    var work: UnsafePointer[Scalar[Self.precision], MutAnyOrigin]
    var qi: UnsafePointer[Scalar[Self.target], MutAnyOrigin]
    var scales: UnsafePointer[Scalar[Self.precision], MutAnyOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self,
                 work: UnsafePointer[Scalar[Self.precision], MutAnyOrigin],
                 qi: UnsafePointer[Scalar[Self.target], MutAnyOrigin],
                 scales: UnsafePointer[Scalar[Self.precision], MutAnyOrigin],
                 rows: Int, cols: Int):
        self.work = work
        self.qi = qi
        self.scales = scales
        self.rows = rows
        self.cols = cols


# =============================================================================
# Atomic op: scale_columns (gamma absorption)
# =============================================================================


def scale_columns[P: DType, T: DType](
    mut ctx: QuantContext[P, T],
    gamma: UnsafePointer[Scalar[P], MutAnyOrigin],
):
    """Column-wise multiply: ctx.work[r, k] *= gamma[k].

    Used for RMSNorm gamma absorption into weight matrices.
    gamma is a vector of length ctx.cols in precision P.
    """
    comptime width = simd_width_of[P]()
    for r in range(ctx.rows):
        var row = ctx.work + r * ctx.cols
        var k = 0
        while k + width <= ctx.cols:
            (row + k).store(
                (row + k).load[width=width]() * (gamma + k).load[width=width]()
            )
            k += width
        while k < ctx.cols:
            row[k] = row[k] * gamma[k]
            k += 1


# =============================================================================
# Atomic op: fwht (Fast Walsh-Hadamard Transform)
# =============================================================================


def fwht_inplace[P: DType](
    buf: UnsafePointer[Scalar[P], MutAnyOrigin], n: Int,
):
    """In-place normalized Walsh-Hadamard transform on n elements.

    n must be a power of 2. Performs log2(n) butterfly stages then
    scales by 1/sqrt(n) for orthonormality. Self-inverse: H(H(x)) = x.
    """
    var half = 1
    while half < n:
        var i = 0
        while i < n:
            for j in range(half):
                var a = buf[i + j]
                var b = buf[i + j + half]
                buf[i + j] = a + b
                buf[i + j + half] = a - b
            i += half * 2
        half *= 2
    var sc = Scalar[P](1.0 / sqrt(Float32(n)))
    for i in range(n):
        buf[i] = buf[i] * sc


def fwht_rows[P: DType, T: DType](
    mut ctx: QuantContext[P, T], block: Int,
):
    """Block-diagonal FWHT on each row of ctx.work[rows, cols].

    Partitions each row into cols/block independent blocks and applies
    fwht_inplace to each. block must be a power of 2 and divide cols.
    """
    for r in range(ctx.rows):
        var base = ctx.work + r * ctx.cols
        for b in range(ctx.cols // block):
            fwht_inplace[P](base + b * block, block)


# =============================================================================
# Atomic op: compute_scales (per-row absmax)
# =============================================================================


def compute_scales[P: DType, T: DType](mut ctx: QuantContext[P, T]):
    """Per-row absmax: ctx.scales[r] = max(|ctx.work[r, :]|) / 127."""
    comptime width = simd_width_of[P]()
    for r in range(ctx.rows):
        var row = ctx.work + r * ctx.cols
        var vmax = SIMD[P, width](0)
        var k = 0
        while k + width <= ctx.cols:
            vmax = max(vmax, (row + k).load[width=width]().__abs__())
            k += width
        var amax = vmax.reduce_max()
        while k < ctx.cols:
            var a = row[k]
            if a < 0: a = -a
            if a > amax: amax = a
            k += 1
        ctx.scales[r] = amax / Scalar[P](127.0)


# =============================================================================
# Atomic op: quantize (round-to-nearest with clamping)
# =============================================================================


def quantize[P: DType, T: DType](mut ctx: QuantContext[P, T]):
    """Per-row quantize: qi[r,k] = clamp(round(work[r,k] / scale[r])).

    Reads scales from ctx.scales (must be populated by compute_scales).
    Casts from precision P to target T.
    """
    comptime width = simd_width_of[P]()
    comptime lo = SIMD[P, width](-128.0)
    comptime hi = SIMD[P, width](127.0)

    for r in range(ctx.rows):
        var sc = ctx.scales[r]
        var inv_sc = Scalar[P](1.0) / sc if sc != Scalar[P](0) else Scalar[P](0)
        var inv = SIMD[P, width](inv_sc)
        var src_row = ctx.work + r * ctx.cols
        var dst_row = ctx.qi + r * ctx.cols
        var k = 0
        while k + width <= ctx.cols:
            var v = (src_row + k).load[width=width]()
            (dst_row + k).store(min(max(roundeven(v * inv), lo), hi).cast[T]())
            k += width
        while k < ctx.cols:
            var v = roundeven[P, 1](src_row[k] * inv_sc)
            dst_row[k] = min(max(v, Scalar[P](-128.0)), Scalar[P](127.0)).cast[T]()
            k += 1


# =============================================================================
# Composed recipes — generic, monomorphized at call site
# =============================================================================


def channelwise[P: DType, T: DType](mut ctx: QuantContext[P, T]):
    """Plain channelwise: absmax scales + round-to-nearest."""
    compute_scales(ctx)
    quantize(ctx)


def hadamard[P: DType, T: DType, block: Int](mut ctx: QuantContext[P, T]):
    """Hadamard rotation + channelwise. No gamma absorption."""
    fwht_rows(ctx, block)
    compute_scales(ctx)
    quantize(ctx)


def hadamard_gamma[P: DType, T: DType, block: Int](
    mut ctx: QuantContext[P, T],
    gamma: UnsafePointer[Scalar[P], MutAnyOrigin],
):
    """Gamma absorption + Hadamard rotation + channelwise."""
    scale_columns(ctx, gamma)
    fwht_rows(ctx, block)
    compute_scales(ctx)
    quantize(ctx)
