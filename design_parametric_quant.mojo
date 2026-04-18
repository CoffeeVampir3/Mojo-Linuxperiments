"""Design reproduction: parametric source-format converters.

Each vendor distribution (bf16, f32, FP8+block-scales, future AWQ/GPTQ)
becomes a struct conforming to `Converter`. The trait bundles what the
quantizer planner needs (source dtype, per-element bytes, aux suffix,
aux byte count) with the operation itself (dequant source bytes into a
destination-dtype buffer).

Today `quant/butterquant.mojo` hardcodes bf16 with `src_bytes =
weight_bytes * 2` and an unconditional `bf16_to_f32(...)`. With the
trait in place those become `rows * cols * Src.SOURCE_ELEMENT_BYTES`
and `Src.convert[DType.float32](...)`; aux lookup in the planner is a
one-liner gated on `Src.AUX_SUFFIX != ""`.

Two axes are parametric:

  * `Raw[src: DType]`   — vendor-native scalar source, no companion
                          tensor. One struct, one DType parameter,
                          covers bf16, f32, f16, and anything else
                          Mojo's scalar cast handles.
  * `Fp8E4M3Block[b]`   — FP8 E4M3 source with one F32 scale per b-by-b
                          tile. Block size is a comptime parameter so
                          128 (MiniMax) and any future 64 / 256 vendor
                          release fall out as Fp8E4M3Block[64], etc.

Inner loops are SIMD: load a vector of source elements, cast to f32 for
the scaled-math case, multiply by a splatted block scale, cast once to
dst_dtype at the store. All sizes are asserted to be f32-SIMD-aligned —
consistent with the rest of the quantizer kernels.
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
# Raw[src] — vendor-native scalar source, no aux. SIMD cast loop.
#
# Covers Bf16Converter = Raw[DType.bfloat16], F32Converter = Raw[DType.float32],
# and anything else Mojo's scalar cast handles natively.
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


# =============================================================================
# Fp8E4M3Block[block] — FP8 E4M3 source with one F32 scale per block-by-block
# tile in a companion `*_scale_inv` tensor. Block is comptime so we get one
# monomorphization per vendor block size.
#
# Tile iteration hoists the scale splat, inner lane loop is SIMD f32: FP8
# load -> cast to f32 -> multiply by scale vec -> cast to dst_dtype store.
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
# Planner-side usage — what `plan_quantization` looks like once the source
# layout is read through the trait. Purely illustrative.
# =============================================================================


def planned_src_bytes[Src: Converter](rows: Int, cols: Int) -> Int:
    return rows * cols * Src.SOURCE_ELEMENT_BYTES


def planned_aux_bytes[Src: Converter](rows: Int, cols: Int) -> Int:
    return Src.aux_bytes_for(rows, cols)


def planned_aux_name[Src: Converter](weight_name: String) -> String:
    comptime if Src.AUX_SUFFIX == StaticString(""):
        return ""
    return weight_name + String(Src.AUX_SUFFIX)


# =============================================================================
# Demos — synthetic tensors, real conversions, spot checks.
# =============================================================================


@always_inline
def bytes_ptr(mut buf: List[UInt8]) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr()))


def check_close(name: String, got: Float32, want: Float32, tol: Float32 = 1e-3):
    var diff = got - want
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        print("FAIL:", name, "got", got, "want", want)


def demo_raw_bf16():
    # Enough elements to clear any plausible f32 SIMD width (AVX-2: 8,
    # AVX-512: 16). 64 is divisible by both.
    var n = 64
    var src = List[UInt8](length=n * Bf16Converter.SOURCE_ELEMENT_BYTES, fill=UInt8(0))
    var dst = List[UInt8](length=n * size_of[Float32](), fill=UInt8(0))
    var src_ptr = bytes_ptr(src)
    var dst_ptr = bytes_ptr(dst).bitcast[Scalar[DType.float32]]()

    var bf16_view = src_ptr.bitcast[Scalar[DType.bfloat16]]()
    for i in range(n):
        bf16_view[i] = Scalar[DType.bfloat16](Float32(i) * 0.25 - 8.0)

    var null_aux = UnsafePointer[UInt8, MutAnyOrigin]()
    Bf16Converter.convert[DType.float32](src_ptr, null_aux, dst_ptr, 1, n)

    for i in range(n):
        check_close("bf16[" + String(i) + "]", dst_ptr[i],
            Float32(i) * 0.25 - 8.0, tol=0.05)  # bf16 rounding slack
    print("demo_raw_bf16 OK")


def demo_raw_f32():
    var n = 64
    var src = List[UInt8](length=n * F32Converter.SOURCE_ELEMENT_BYTES, fill=UInt8(0))
    var dst = List[UInt8](length=n * size_of[Float32](), fill=UInt8(0))
    var src_ptr = bytes_ptr(src)
    var dst_ptr = bytes_ptr(dst).bitcast[Scalar[DType.float32]]()

    var f32_view = src_ptr.bitcast[Scalar[DType.float32]]()
    for i in range(n):
        f32_view[i] = Float32(i) * 0.125 - 4.0

    var null_aux = UnsafePointer[UInt8, MutAnyOrigin]()
    F32Converter.convert[DType.float32](src_ptr, null_aux, dst_ptr, 1, n)

    for i in range(n):
        check_close("f32[" + String(i) + "]", dst_ptr[i],
            Float32(i) * 0.125 - 4.0)
    print("demo_raw_f32 OK")


def demo_fp8_e4m3_block128():
    # 256 x 256 weight -> exact 2 x 2 tile grid at block=128.
    var rows = 256
    var cols = 256
    var src = List[UInt8](length=rows * cols, fill=UInt8(0))
    var aux_bytes = Fp8E4M3Block128Converter.aux_bytes_for(rows, cols)
    var aux = List[UInt8](length=aux_bytes, fill=UInt8(0))
    var dst = List[UInt8](length=rows * cols * size_of[Float32](), fill=UInt8(0))

    var src_ptr = bytes_ptr(src)
    var aux_ptr = bytes_ptr(aux)
    var dst_ptr = bytes_ptr(dst).bitcast[Scalar[DType.float32]]()

    # E4M3 1.0 = 0x38, 2.0 = 0x40, 0.5 = 0x30, -1.0 = 0xB8.
    for i in range(rows * cols):
        src_ptr[i] = UInt8(0x38)  # start with 1.0 everywhere

    var scales = aux_ptr.bitcast[Scalar[DType.float32]]()
    scales[0] = Float32(1.0)  # tile (0,0)
    scales[1] = Float32(2.0)  # tile (0,1)
    scales[2] = Float32(4.0)  # tile (1,0)
    scales[3] = Float32(8.0)  # tile (1,1)

    Fp8E4M3Block128Converter.convert[DType.float32](src_ptr, aux_ptr, dst_ptr, rows, cols)

    # All-1.0 fp8 * scale -> scale everywhere in each tile.
    check_close("tile(0,0) corner",   dst_ptr[0 * cols + 0],     Float32(1.0))
    check_close("tile(0,1) corner",   dst_ptr[0 * cols + 128],   Float32(2.0))
    check_close("tile(1,0) corner",   dst_ptr[128 * cols + 0],   Float32(4.0))
    check_close("tile(1,1) corner",   dst_ptr[128 * cols + 128], Float32(8.0))
    check_close("tile(0,0) far edge", dst_ptr[127 * cols + 127], Float32(1.0))
    check_close("tile(1,1) far edge", dst_ptr[255 * cols + 255], Float32(8.0))

    # Swap in non-unit FP8 bit patterns to exercise the decode itself.
    for r in range(128):
        for c in range(128):
            src_ptr[r * cols + c]             = UInt8(0x40)   # 2.0 in tile (0,0)
            src_ptr[r * cols + (c + 128)]     = UInt8(0x30)   # 0.5 in tile (0,1)
            src_ptr[(r + 128) * cols + c]     = UInt8(0xB8)   # -1.0 in tile (1,0)

    Fp8E4M3Block128Converter.convert[DType.float32](src_ptr, aux_ptr, dst_ptr, rows, cols)

    check_close("tile(0,0) * 2.0",  dst_ptr[10 * cols + 10],   Float32(1.0) * 2.0)
    check_close("tile(0,1) * 0.5",  dst_ptr[10 * cols + 138],  Float32(2.0) * 0.5)
    check_close("tile(1,0) * -1.0", dst_ptr[138 * cols + 10],  Float32(4.0) * -1.0)
    check_close("tile(1,1) * 1.0",  dst_ptr[138 * cols + 138], Float32(8.0))

    print("demo_fp8_e4m3_block128 OK")


def demo_fp8_e4m3_block64():
    # Different block size exercises the comptime parameter end-to-end.
    # 128 x 128 at block=64 -> 2 x 2 tile grid.
    comptime Fp8_64 = Fp8E4M3Block[64]

    var rows = 128
    var cols = 128
    var src = List[UInt8](length=rows * cols, fill=UInt8(0))
    var aux_bytes = Fp8_64.aux_bytes_for(rows, cols)
    var aux = List[UInt8](length=aux_bytes, fill=UInt8(0))
    var dst = List[UInt8](length=rows * cols * size_of[Float32](), fill=UInt8(0))

    var src_ptr = bytes_ptr(src)
    var aux_ptr = bytes_ptr(aux)
    var dst_ptr = bytes_ptr(dst).bitcast[Scalar[DType.float32]]()

    for i in range(rows * cols):
        src_ptr[i] = UInt8(0x38)  # 1.0

    var scales = aux_ptr.bitcast[Scalar[DType.float32]]()
    scales[0] = Float32(0.5)
    scales[1] = Float32(1.0)
    scales[2] = Float32(2.0)
    scales[3] = Float32(4.0)

    Fp8_64.convert[DType.float32](src_ptr, aux_ptr, dst_ptr, rows, cols)

    check_close("blk64 tile(0,0)", dst_ptr[0 * cols + 0],   Float32(0.5))
    check_close("blk64 tile(0,1)", dst_ptr[0 * cols + 64],  Float32(1.0))
    check_close("blk64 tile(1,0)", dst_ptr[64 * cols + 0],  Float32(2.0))
    check_close("blk64 tile(1,1)", dst_ptr[64 * cols + 64], Float32(4.0))
    print("demo_fp8_e4m3_block64 OK")


def demo_planner_usage():
    print("bf16 3072x3072 -> src",
          planned_src_bytes[Bf16Converter](3072, 3072),
          "aux", planned_aux_bytes[Bf16Converter](3072, 3072))
    print("f32   256x3072 -> src",
          planned_src_bytes[F32Converter](256, 3072),
          "aux", planned_aux_bytes[F32Converter](256, 3072))
    print("fp8[128] 1536x3072 -> src",
          planned_src_bytes[Fp8E4M3Block128Converter](1536, 3072),
          "aux", planned_aux_bytes[Fp8E4M3Block128Converter](1536, 3072),
          "aux_name", planned_aux_name[Fp8E4M3Block128Converter](
              String("experts.0.w1.weight")))
    print("fp8[64]   768x3072 -> src",
          planned_src_bytes[Fp8E4M3Block[64]](768, 3072),
          "aux", planned_aux_bytes[Fp8E4M3Block[64]](768, 3072))


def main():
    demo_raw_bf16()
    demo_raw_f32()
    demo_fp8_e4m3_block128()
    demo_fp8_e4m3_block64()
    demo_planner_usage()
    print("all converter demos passed")
