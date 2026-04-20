"""Prototype: can a Converter trait use typed pointers derived from its
associated comptime dtype, instead of forcing everything through UInt8*?"""

from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of
from std.collections import InlineArray

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]


trait Converter:
    comptime SOURCE_DTYPE: DType
    comptime AUX_DTYPE: DType

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        count: Int,
    ): ...


struct RawBF16(Converter):
    comptime SOURCE_DTYPE = DType.bfloat16
    comptime AUX_DTYPE = DType.uint8

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
        aux: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        count: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        var k = 0
        while k + width <= count:
            var v = (src + k).load[width=width]().cast[dst_dtype]()
            (dst + k).store(v)
            k += width


struct Fp8Block(Converter):
    comptime SOURCE_DTYPE = DType.float8_e4m3fn
    comptime AUX_DTYPE = DType.float32

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
        aux: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        count: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        var scale = aux[0]
        var k = 0
        while k + width <= count:
            var v = (src + k).load[width=width]().cast[DType.float32]() * scale
            (dst + k).store(v.cast[dst_dtype]())
            k += width


def run_converter[C: Converter](raw_src: PtrU8, raw_aux: PtrU8, count: Int):
    var src = raw_src.bitcast[Scalar[C.SOURCE_DTYPE]]()
    var aux = raw_aux.bitcast[Scalar[C.AUX_DTYPE]]()
    var dst_buf = InlineArray[Scalar[DType.bfloat16], 16](fill=Scalar[DType.bfloat16](0))
    var dst = UnsafePointer(to=dst_buf).bitcast[Scalar[DType.bfloat16]]()
    C.convert[DType.bfloat16](src, aux, dst, count)
    print("result[0]:", dst[0].cast[DType.float32]())


def main():
    comptime N = 16

    var bf16_buf = InlineArray[Scalar[DType.bfloat16], N](fill=Scalar[DType.bfloat16](0))
    for i in range(N):
        bf16_buf[i] = Scalar[DType.bfloat16](Float32(i + 1))
    var dummy_aux = InlineArray[UInt8, 4](fill=UInt8(0))

    print("--- RawBF16 ---")
    run_converter[RawBF16](
        UnsafePointer(to=bf16_buf).bitcast[UInt8](),
        UnsafePointer(to=dummy_aux).bitcast[UInt8](), N)

    var fp8_buf = InlineArray[Scalar[DType.float8_e4m3fn], N](fill=Scalar[DType.float8_e4m3fn](0))
    for i in range(N):
        fp8_buf[i] = Scalar[DType.float8_e4m3fn](Float32(i + 1))
    var scale_buf = InlineArray[Float32, 1](fill=Float32(2.0))

    print("--- Fp8Block ---")
    run_converter[Fp8Block](
        UnsafePointer(to=fp8_buf).bitcast[UInt8](),
        UnsafePointer(to=scale_buf).bitcast[UInt8](), N)
