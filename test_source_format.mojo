"""Small source-format checks for BF16-staged FP8 E4M3 conversion."""

from std.memory import UnsafePointer
from std.sys.info import size_of

from modeling.model_spec import SourceFormat
from quant.source_format import (
    Fp8E4M3Block128Converter,
    convert_to_f32,
    source_aux_bytes,
    source_bf16_work_bytes,
)


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@always_inline
def u8_ptr(mut buf: List[UInt8]) -> PtrU8:
    return PtrU8(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def f32_ptr(mut buf: List[Float32]) -> PtrF32:
    return PtrF32(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def bf16_ptr(mut buf: List[Scalar[DType.bfloat16]]) -> PtrBF16:
    return PtrBF16(unsafe_from_address=Int(buf.unsafe_ptr()))


def check_close(name: String, got: Float32, want: Float32, tol: Float32) -> Int:
    var diff = got - want
    if diff < 0:
        diff = -diff
    if diff > tol:
        print("FAIL " + name + ": got=" + String(got) + " want=" + String(want))
        return 1
    return 0


def fill_fp8_pattern(ptr: PtrU8, rows: Int, cols: Int):
    for r in range(rows):
        for c in range(cols):
            var idx = r * cols + c
            if c < 128:
                ptr[idx] = UInt8(0x38)  # 1.0
            else:
                if (r & 1) == 0:
                    ptr[idx] = UInt8(0x40)  # 2.0
                else:
                    ptr[idx] = UInt8(0xB8)  # -1.0


def test_decompress_to_bf16() -> Int:
    comptime rows = 128
    comptime cols = 256
    var failures = 0

    var src_buf = List[UInt8](length=rows * cols, fill=UInt8(0))
    var aux_buf = List[UInt8](
        length=Fp8E4M3Block128Converter.aux_bytes_for(rows, cols),
        fill=UInt8(0),
    )
    var bf16_buf = List[Scalar[DType.bfloat16]](
        length=rows * cols, fill=Scalar[DType.bfloat16](0))

    var src = u8_ptr(src_buf)
    var aux = u8_ptr(aux_buf)
    var bf16 = bf16_ptr(bf16_buf)
    fill_fp8_pattern(src, rows, cols)

    var scales = aux.bitcast[Scalar[DType.float32]]()
    scales[0] = Float32(0.5)
    scales[1] = Float32(3.0)

    Fp8E4M3Block128Converter.decompress_to_bf16(src, aux, bf16, rows, cols)

    failures += check_close("bf16 tile0", Float32(bf16[0]), Float32(0.5), Float32(0.01))
    failures += check_close("bf16 tile1 even", Float32(bf16[130]), Float32(6.0), Float32(0.02))
    failures += check_close(
        "bf16 tile1 odd", Float32(bf16[1 * cols + 130]), Float32(-3.0), Float32(0.02))
    return failures


def test_convert_to_f32_ragged_tail() -> Int:
    comptime total_rows = 384
    comptime cols = 256
    comptime first_panel_rows = 256
    comptime tail_rows = total_rows - first_panel_rows
    var failures = 0

    var src_buf = List[UInt8](length=total_rows * cols, fill=UInt8(0))
    var aux_buf = List[UInt8](
        length=source_aux_bytes(SourceFormat.FP8_E4M3_BLOCK128, total_rows, cols),
        fill=UInt8(0),
    )
    var work_buf = List[Float32](length=first_panel_rows * cols, fill=Float32(0))
    var bf16_work_buf = List[Scalar[DType.bfloat16]](
        length=source_bf16_work_bytes(
            SourceFormat.FP8_E4M3_BLOCK128, first_panel_rows, cols) // 2,
        fill=Scalar[DType.bfloat16](0),
    )

    var src = u8_ptr(src_buf)
    var aux = u8_ptr(aux_buf)
    var work = f32_ptr(work_buf)
    var bf16_work = PtrU8(unsafe_from_address=Int(bf16_work_buf.unsafe_ptr()))
    fill_fp8_pattern(src, total_rows, cols)

    # Three row tiles x two column tiles. The tail panel should use row tile 2.
    var scales = aux.bitcast[Scalar[DType.float32]]()
    scales[0] = Float32(1.0)
    scales[1] = Float32(2.0)
    scales[2] = Float32(4.0)
    scales[3] = Float32(8.0)
    scales[4] = Float32(16.0)
    scales[5] = Float32(32.0)

    # First full-ish panel: two row tiles.
    convert_to_f32(
        SourceFormat.FP8_E4M3_BLOCK128,
        src,
        aux,
        bf16_work,
        work,
        first_panel_rows,
        cols,
    )
    failures += check_close("panel0 tile0", work[0], Float32(1.0), Float32(0.01))
    failures += check_close("panel0 tile1", work[130], Float32(4.0), Float32(0.02))
    failures += check_close(
        "panel1 rowtile1 coltile0",
        work[128 * cols],
        Float32(4.0),
        Float32(0.02),
    )
    failures += check_close(
        "panel1 rowtile1 coltile1",
        work[128 * cols + 130],
        Float32(16.0),
        Float32(0.03),
    )

    # Ragged final panel: one row tile. This mirrors execute_butterquant's
    # final smaller panel with the companion scale pointer advanced.
    var src_tail = src + first_panel_rows * cols
    var aux_tail = aux + (first_panel_rows // 128) * (cols // 128) * size_of[Float32]()
    convert_to_f32(
        SourceFormat.FP8_E4M3_BLOCK128,
        src_tail,
        aux_tail,
        bf16_work,
        work,
        tail_rows,
        cols,
    )
    failures += check_close("tail tile0", work[0], Float32(16.0), Float32(0.03))
    failures += check_close("tail tile1", work[130], Float32(64.0), Float32(0.10))
    failures += check_close(
        "tail odd negative",
        work[1 * cols + 130],
        Float32(-32.0),
        Float32(0.05),
    )
    return failures


def main():
    var failures = 0
    failures += test_decompress_to_bf16()
    failures += test_convert_to_f32_ragged_tail()
    if failures != 0:
        print("test_source_format FAILED failures=" + String(failures))
        return
    print("test_source_format OK")
