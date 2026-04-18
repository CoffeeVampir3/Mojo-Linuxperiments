"""Source-format converters for the offline quantizer.

Each vendor distribution (bf16, f32, FP8+block-scales, future AWQ/GPTQ)
is a struct conforming to `Converter`. The trait bundles the planner-side
descriptors (source dtype, per-element bytes, aux tensor suffix and size)
with the operation itself — dequant source bytes into a destination-dtype
buffer.

Today two parametric families cover everything shipping:

  * `Raw[dtype]`         — vendor-native scalar source with no companion
                           tensor. Bf16Converter, F32Converter, F16Converter
                           are all comptime aliases over it.
  * `Fp8E4M3Block[b]`    — FP8 E4M3 source with one F32 scale per b-by-b
                           tile in a companion "_scale_inv" tensor. Block
                           size is comptime so MiniMax's 128 and any
                           future 64 / 256 fall out as specializations.

The quantizer picks a `Converter` via the `SourceFormat` tag carried on
each `QuantizeTask`, dispatches once at panel entry, and the inner body
runs at native SIMD width for the chosen (source, destination) pair.
"""

from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of


# =============================================================================
# Trait
# =============================================================================


trait Converter:
    comptime SOURCE_DTYPE: DType
    comptime SOURCE_ELEMENT_BYTES: Int
    comptime AUX_SUFFIX: StaticString  # "" when no companion tensor

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int: ...

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[UInt8, MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ): ...


# =============================================================================
# Raw[dtype] — vendor-native scalar source, no aux. SIMD cast loop.
# =============================================================================


struct Raw[dtype: DType](Converter):
    comptime SOURCE_DTYPE = Self.dtype
    comptime SOURCE_ELEMENT_BYTES = size_of[Scalar[Self.dtype]]()
    comptime AUX_SUFFIX: StaticString = ""

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return 0

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[UInt8, MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        var n = rows * cols
        debug_assert(n % width == 0, "Raw.convert: n must be f32-simd-aligned")
        var typed = src.bitcast[Scalar[Self.dtype]]()
        var k = 0
        while k < n:
            (dst + k).store((typed + k).load[width=width]().cast[dst_dtype]())
            k += width


comptime Bf16Converter = Raw[DType.bfloat16]
comptime F32Converter  = Raw[DType.float32]
comptime F16Converter  = Raw[DType.float16]


# =============================================================================
# Fp8E4M3Block[block] — FP8 E4M3 + one F32 scale per block-by-block tile.
# =============================================================================


struct Fp8E4M3Block[block: Int](Converter):
    comptime SOURCE_DTYPE = DType.float8_e4m3fn
    comptime SOURCE_ELEMENT_BYTES = 1
    comptime AUX_SUFFIX: StaticString = "_scale_inv"
    comptime BLOCK = Self.block

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return (rows // Self.block) * (cols // Self.block) * 4

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[UInt8, MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        debug_assert(Self.block % width == 0,
            "Fp8E4M3Block: block must be f32-simd-aligned")
        debug_assert(rows % Self.block == 0 and cols % Self.block == 0,
            "Fp8E4M3Block: rows/cols must be multiples of block")

        var fp8 = src.bitcast[Scalar[DType.float8_e4m3fn]]()
        var scales = aux.bitcast[Scalar[DType.float32]]()
        var tiles_c = cols // Self.block
        var tiles_r = rows // Self.block

        for tr in range(tiles_r):
            for tc in range(tiles_c):
                var scale_vec = SIMD[DType.float32, width](
                    scales[tr * tiles_c + tc])
                var r0 = tr * Self.block
                var c0 = tc * Self.block
                for r_in in range(Self.block):
                    var row_off = (r0 + r_in) * cols + c0
                    var c = 0
                    while c < Self.block:
                        var v = (fp8 + row_off + c).load[width=width]().cast[DType.float32]()
                        (dst + row_off + c).store((v * scale_vec).cast[dst_dtype]())
                        c += width


comptime Fp8E4M3Block128Converter = Fp8E4M3Block[128]


# =============================================================================
# Tag-to-descriptor bridges
#
# The quantizer carries a `SourceFormat` Int tag on each task. Dispatch
# helpers below turn that tag into the converter-trait values the planner
# and panel loop need. They're one-liners per variant because that's all
# the trait exposes — but centralizing the switch here keeps the quantizer
# free of source-format branches.
#
# Adding a new `SourceFormat` variant means one new arm in each helper and
# one new `Converter` struct above. Nothing else moves.
# =============================================================================


from modeling.model_spec import SourceFormat


def source_element_bytes(source: Int) -> Int:
    if source == SourceFormat.BF16:
        return Bf16Converter.SOURCE_ELEMENT_BYTES
    if source == SourceFormat.F32:
        return F32Converter.SOURCE_ELEMENT_BYTES
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return Fp8E4M3Block128Converter.SOURCE_ELEMENT_BYTES
    return 0


def source_dtype(source: Int) -> DType:
    if source == SourceFormat.BF16:
        return Bf16Converter.SOURCE_DTYPE
    if source == SourceFormat.F32:
        return F32Converter.SOURCE_DTYPE
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return Fp8E4M3Block128Converter.SOURCE_DTYPE
    return DType.invalid


def source_aux_bytes(source: Int, rows: Int, cols: Int) -> Int:
    if source == SourceFormat.BF16:
        return Bf16Converter.aux_bytes_for(rows, cols)
    if source == SourceFormat.F32:
        return F32Converter.aux_bytes_for(rows, cols)
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return Fp8E4M3Block128Converter.aux_bytes_for(rows, cols)
    return 0


def source_aux_dtype(source: Int) -> DType:
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return DType.float32
    return DType.invalid


def source_aux_name(source: Int, weight_name: String) -> String:
    if source == SourceFormat.BF16:
        return ""
    if source == SourceFormat.F32:
        return ""
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return weight_name + String(Fp8E4M3Block128Converter.AUX_SUFFIX)
    return ""


def convert_to_f32(
    source: Int,
    src: UnsafePointer[UInt8, MutAnyOrigin],
    aux: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int, cols: Int,
):
    if source == SourceFormat.BF16:
        Bf16Converter.convert[DType.float32](src, aux, dst, rows, cols)
    elif source == SourceFormat.F32:
        F32Converter.convert[DType.float32](src, aux, dst, rows, cols)
    elif source == SourceFormat.FP8_E4M3_BLOCK128:
        Fp8E4M3Block128Converter.convert[DType.float32](src, aux, dst, rows, cols)


def source_aux_row_block(source: Int) -> Int:
    """Row-axis block size for the aux tensor, or 0 when no aux is used.
    Used to slice the in-memory aux buffer when the quantizer panels rows.
    """
    if source == SourceFormat.FP8_E4M3_BLOCK128:
        return 128
    return 0
