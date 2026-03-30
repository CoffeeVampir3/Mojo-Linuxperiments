"""Test whether the bf16→f32 cast codegen has been fixed."""

from std.memory import UnsafePointer


@no_inline
def bf16_to_f32_native(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    return ptr.load[width=8]().cast[DType.float32]()


@no_inline
def bf16_to_f32_manual(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    var wide = ptr.bitcast[Scalar[DType.uint16]]().load[width=8]().cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=8]()


def main():
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=0x1000000)
    var a = bf16_to_f32_native(p)
    var b = bf16_to_f32_manual(p)
    # Force both to be used
    var result = a + b
    print(result[0])
