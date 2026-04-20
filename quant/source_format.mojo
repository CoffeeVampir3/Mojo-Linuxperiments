"""Source-format converters for the offline quantizer.

Each vendor distribution (bf16, f32, FP8+block-scales, future AWQ/GPTQ)
is a struct conforming to `Converter`. The trait bundles planner-side
descriptors with the conversion operation itself.

Two parametric families cover everything shipping:

  * `Raw[dtype]`         — vendor-native scalar source, no companion
                           tensor. Bf16Converter, F32Converter, F16Converter
                           are all comptime aliases over it.
  * `Fp8E4M3Block[b]`    — FP8 E4M3 source with one F32 scale per b-by-b
                           tile in a companion "_scale_inv" tensor.

The quantizer dispatches on the `SourceFormat` runtime tag exactly once
at the plan/execute boundary, calling into parametric functions where
all source-format knowledge flows through the Converter trait.
"""

from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


# =============================================================================
# Trait
# =============================================================================


trait Converter:
    comptime SOURCE_DTYPE: DType
    comptime AUX_DTYPE: DType
    comptime SOURCE_ELEMENT_BYTES: Int
    comptime AUX_SUFFIX: StaticString
    comptime AUX_ROW_BLOCK: Int

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int: ...

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ): ...


# =============================================================================
# Raw[dtype] — vendor-native scalar source, no aux. SIMD cast loop.
# =============================================================================


struct Raw[dtype: DType](Converter):
    comptime SOURCE_DTYPE = Self.dtype
    comptime AUX_DTYPE = DType.uint8
    comptime SOURCE_ELEMENT_BYTES = size_of[Scalar[Self.dtype]]()
    comptime AUX_SUFFIX: StaticString = ""
    comptime AUX_ROW_BLOCK = 0

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return 0

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        var n = rows * cols
        debug_assert(n % width == 0, "Raw.convert: n must be f32-simd-aligned")
        var k = 0
        while k < n:
            (dst + k).store((src + k).load[width=width]().cast[dst_dtype]())
            k += width


comptime Bf16Converter = Raw[DType.bfloat16]
comptime F32Converter  = Raw[DType.float32]
comptime F16Converter  = Raw[DType.float16]


# =============================================================================
# Fp8E4M3Block[block] — FP8 E4M3 + one F32 scale per block-by-block tile.
# =============================================================================


@always_inline
def e4m3fn_to_f32[width: Int](
    raw8: SIMD[DType.uint8, width],
) -> SIMD[DType.float32, width]:
    var raw = raw8.cast[DType.uint32]()
    var zero = SIMD[DType.uint32, width](0)
    var sign_bits = (raw & SIMD[DType.uint32, width](0x80)) << 24
    var exp = (raw >> 3) & SIMD[DType.uint32, width](0x0F)
    var mant = raw & SIMD[DType.uint32, width](0x07)

    var exp_is_15 = exp.eq(SIMD[DType.uint32, width](15))
    var mant_is_7 = mant.eq(SIMD[DType.uint32, width](7))
    var maxfinite_payload = exp_is_15 & mant_is_7
    var mant_finite = maxfinite_payload.select(
        SIMD[DType.uint32, width](6), mant)

    var normal_bits = (
        sign_bits
        | ((exp + SIMD[DType.uint32, width](120)) << 23)
        | (mant_finite << 20)
    )
    var normal = SIMD[DType.float32, width](from_bits=normal_bits)

    var sign_neg = (raw & SIMD[DType.uint32, width](0x80)).ne(zero)
    var sub_mag = mant.cast[DType.float32]() * SIMD[DType.float32, width](0.001953125)
    var subnormal = sign_neg.select(-sub_mag, sub_mag)
    return exp.eq(zero).select(subnormal, normal)


struct Fp8E4M3Block[block: Int](Converter):
    comptime SOURCE_DTYPE = DType.float8_e4m3fn
    comptime AUX_DTYPE = DType.float32
    comptime SOURCE_ELEMENT_BYTES = 1
    comptime AUX_SUFFIX: StaticString = "_scale_inv"
    comptime AUX_ROW_BLOCK = Self.block
    comptime BLOCK = Self.block

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return (rows // Self.block) * (cols // Self.block) * 4

    @staticmethod
    def decompress_to_bf16(
        src: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
        aux: PtrF32,
        dst: PtrBF16,
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        debug_assert(Self.block % width == 0,
            "Fp8E4M3Block: block must be f32-simd-aligned")
        debug_assert(rows % Self.block == 0 and cols % Self.block == 0,
            "Fp8E4M3Block: rows/cols must be multiples of block")

        var raw_bytes = src.bitcast[UInt8]()
        var tiles_c = cols // Self.block
        var tiles_r = rows // Self.block

        for tr in range(tiles_r):
            for tc in range(tiles_c):
                var scale_vec = SIMD[DType.float32, width](
                    aux[tr * tiles_c + tc])
                var r0 = tr * Self.block
                var c0 = tc * Self.block
                for r_in in range(Self.block):
                    var row_off = (r0 + r_in) * cols + c0
                    var c = 0
                    while c < Self.block:
                        var raw = (raw_bytes + row_off + c).load[width=width]()
                        var v = e4m3fn_to_f32[width](raw) * scale_vec
                        (dst + row_off + c).store(v.cast[DType.bfloat16]())
                        c += width

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
        aux: PtrF32,
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        var bf16_buf = List[Scalar[DType.bfloat16]](
            length=rows * cols, fill=Scalar[DType.bfloat16](0))
        var bf16 = PtrBF16(unsafe_from_address=Int(bf16_buf.unsafe_ptr()))
        Self.decompress_to_bf16(src, aux, bf16, rows, cols)
        var null_aux = PtrU8()
        Bf16Converter.convert[dst_dtype](bf16, null_aux,
            dst, rows, cols)
        _ = bf16_buf^


comptime Fp8E4M3Block128Converter = Fp8E4M3Block[128]


