from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.benchmark import keep

comptime W = simd_width_of[DType.float32]()


@no_inline
def simd_min(a: SIMD[DType.float32, W], b: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    return min(a, b)


@no_inline
def simd_max(a: SIMD[DType.float32, W], b: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    return max(a, b)


def main():
    var a = alloc[Float32](W)
    var b = alloc[Float32](W)

    for i in range(W):
        a[i] = Float32(i) - 8.0
        b[i] = Float32(8) - Float32(i)

    var va = a.load[width=W]()
    var vb = b.load[width=W]()

    var lo = simd_min(va, vb)
    var hi = simd_max(va, vb)

    keep(lo)
    keep(hi)

    a.free()
    b.free()
