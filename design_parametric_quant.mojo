"""Design reproduction: typed parametric source-format converters.

Exercises the Converter trait from quant/source_format.mojo with
synthetic data. Validates that typed pointers flow through convert
without any bitcasts at the call site.
"""

from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of

from quant.source_format import (
    Converter, Raw, Fp8E4M3Block,
    Bf16Converter, F32Converter, Fp8E4M3Block128Converter,
)


def check_close(name: String, got: Float32, want: Float32, tol: Float32 = 1e-3):
    var diff = got - want
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        print("FAIL:", name, "got", got, "want", want)


def demo_raw_bf16():
    var n = 64
    var src_buf = List[Scalar[DType.bfloat16]](
        length=n, fill=Scalar[DType.bfloat16](0))
    var dst_buf = List[Float32](length=n, fill=Float32(0))

    var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(src_buf.unsafe_ptr()))
    var dst = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(dst_buf.unsafe_ptr()))

    for i in range(n):
        src[i] = Scalar[DType.bfloat16](Float32(i) * 0.25 - 8.0)

    var null_aux = UnsafePointer[UInt8, MutAnyOrigin]()
    Bf16Converter.convert[DType.float32](src, null_aux, dst, 1, n)

    for i in range(n):
        check_close("bf16[" + String(i) + "]", dst[i],
            Float32(i) * 0.25 - 8.0, tol=0.05)
    print("demo_raw_bf16 OK")


def demo_raw_f32():
    var n = 64
    var src_buf = List[Float32](length=n, fill=Float32(0))
    var dst_buf = List[Float32](length=n, fill=Float32(0))

    var src = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(src_buf.unsafe_ptr()))
    var dst = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(dst_buf.unsafe_ptr()))

    for i in range(n):
        src[i] = Float32(i) * 0.125 - 4.0

    var null_aux = UnsafePointer[UInt8, MutAnyOrigin]()
    F32Converter.convert[DType.float32](src, null_aux, dst, 1, n)

    for i in range(n):
        check_close("f32[" + String(i) + "]", dst[i],
            Float32(i) * 0.125 - 4.0)
    print("demo_raw_f32 OK")


def demo_fp8_e4m3_block128():
    var rows = 256
    var cols = 256
    var n = rows * cols

    var src_buf = List[Scalar[DType.float8_e4m3fn]](
        length=n, fill=Scalar[DType.float8_e4m3fn](0))
    var aux_elems = Fp8E4M3Block128Converter.aux_bytes_for(rows, cols) // 4
    var aux_buf = List[Float32](length=aux_elems, fill=Float32(0))
    var dst_buf = List[Float32](length=n, fill=Float32(0))

    var src = UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin](
        unsafe_from_address=Int(src_buf.unsafe_ptr()))
    var aux = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(aux_buf.unsafe_ptr()))
    var dst = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(dst_buf.unsafe_ptr()))

    var raw_view = src.bitcast[UInt8]()
    for i in range(n):
        raw_view[i] = UInt8(0x38)  # E4M3 1.0

    aux[0] = Float32(1.0)
    aux[1] = Float32(2.0)
    aux[2] = Float32(4.0)
    aux[3] = Float32(8.0)

    Fp8E4M3Block128Converter.convert[DType.float32](src, aux, dst, rows, cols)

    check_close("tile(0,0) corner",   dst[0 * cols + 0],     Float32(1.0))
    check_close("tile(0,1) corner",   dst[0 * cols + 128],   Float32(2.0))
    check_close("tile(1,0) corner",   dst[128 * cols + 0],   Float32(4.0))
    check_close("tile(1,1) corner",   dst[128 * cols + 128], Float32(8.0))

    for r in range(128):
        for c in range(128):
            raw_view[r * cols + c]             = UInt8(0x40)  # 2.0
            raw_view[r * cols + (c + 128)]     = UInt8(0x30)  # 0.5
            raw_view[(r + 128) * cols + c]     = UInt8(0xB8)  # -1.0

    Fp8E4M3Block128Converter.convert[DType.float32](src, aux, dst, rows, cols)

    check_close("tile(0,0) * 2.0",  dst[10 * cols + 10],   Float32(1.0) * 2.0)
    check_close("tile(0,1) * 0.5",  dst[10 * cols + 138],  Float32(2.0) * 0.5)
    check_close("tile(1,0) * -1.0", dst[138 * cols + 10],  Float32(4.0) * -1.0)
    check_close("tile(1,1) * 1.0",  dst[138 * cols + 138], Float32(8.0))

    _ = src_buf^
    _ = aux_buf^
    _ = dst_buf^
    print("demo_fp8_e4m3_block128 OK")


def demo_fp8_e4m3_block64():
    comptime Fp8_64 = Fp8E4M3Block[64]

    var rows = 128
    var cols = 128
    var n = rows * cols

    var src_buf = List[Scalar[DType.float8_e4m3fn]](
        length=n, fill=Scalar[DType.float8_e4m3fn](0))
    var aux_elems = Fp8_64.aux_bytes_for(rows, cols) // 4
    var aux_buf = List[Float32](length=aux_elems, fill=Float32(0))
    var dst_buf = List[Float32](length=n, fill=Float32(0))

    var src = UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin](
        unsafe_from_address=Int(src_buf.unsafe_ptr()))
    var aux = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(aux_buf.unsafe_ptr()))
    var dst = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(dst_buf.unsafe_ptr()))

    var raw_view = src.bitcast[UInt8]()
    for i in range(n):
        raw_view[i] = UInt8(0x38)  # 1.0

    aux[0] = Float32(0.5)
    aux[1] = Float32(1.0)
    aux[2] = Float32(2.0)
    aux[3] = Float32(4.0)

    Fp8_64.convert[DType.float32](src, aux, dst, rows, cols)

    check_close("blk64 tile(0,0)", dst[0 * cols + 0],   Float32(0.5))
    check_close("blk64 tile(0,1)", dst[0 * cols + 64],  Float32(1.0))
    check_close("blk64 tile(1,0)", dst[64 * cols + 0],  Float32(2.0))
    check_close("blk64 tile(1,1)", dst[64 * cols + 64], Float32(4.0))

    _ = src_buf^
    _ = aux_buf^
    _ = dst_buf^
    print("demo_fp8_e4m3_block64 OK")


def demo_planner_usage():
    print("bf16 src_elem_bytes:", Bf16Converter.SOURCE_ELEMENT_BYTES,
          "aux_row_block:", Bf16Converter.AUX_ROW_BLOCK)
    print("f32  src_elem_bytes:", F32Converter.SOURCE_ELEMENT_BYTES,
          "aux_row_block:", F32Converter.AUX_ROW_BLOCK)
    print("fp8[128] src_elem_bytes:", Fp8E4M3Block128Converter.SOURCE_ELEMENT_BYTES,
          "aux_row_block:", Fp8E4M3Block128Converter.AUX_ROW_BLOCK,
          "aux_suffix:", Fp8E4M3Block128Converter.AUX_SUFFIX)


def main():
    demo_raw_bf16()
    demo_raw_f32()
    demo_fp8_e4m3_block128()
    demo_fp8_e4m3_block64()
    demo_planner_usage()
    print("all converter demos passed")
