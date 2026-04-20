"""Prove whether bitcast, as_any_origin, or direct pass works for
UnsafePointer[Float32, local] -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]."""

from std.memory import UnsafePointer
from std.collections import InlineArray


def consumer(dst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin], n: Int):
    for i in range(n):
        dst[i] = Float32(i * 10)


def main():
    var buf = InlineArray[Float32, 4](fill=Float32(0))
    var ptr = UnsafePointer(to=buf).bitcast[Float32]()

    print("ptr type: UnsafePointer[Float32, local_origin]")
    print("consumer wants: UnsafePointer[Scalar[DType.float32], MutAnyOrigin]")
    print()

    # Method 1: direct pass (no conversion)
    consumer(ptr, 4)
    print("direct:", buf[0], buf[1], buf[2], buf[3])

    # Method 2: as_any_origin()
    consumer(ptr.as_any_origin(), 4)
    print("as_any_origin:", buf[0], buf[1], buf[2], buf[3])

    # Method 3: bitcast (current codebase pattern)
    consumer(ptr.bitcast[Scalar[DType.float32]](), 4)
    print("bitcast:", buf[0], buf[1], buf[2], buf[3])
