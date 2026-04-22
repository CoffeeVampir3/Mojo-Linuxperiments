from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of


comptime WIDTH = simd_width_of[DType.int32]()
comptime VNNI_BLK = 4
comptime BCAST_BYTES = WIDTH * VNNI_BLK


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


# === Variant A: original hand-unrolled join chain ===

@always_inline
def bcast_join_chain[width: Int](
    b4: SIMD[DType.uint8, 4],
) -> SIMD[DType.uint8, width * 4]:
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    comptime if width <= 4:
        return b16.slice[width * 4]()
    var b32 = b16.join(b16)
    comptime if width <= 8:
        return b32.slice[width * 4]()
    var b64 = b32.join(b32)
    return b64.slice[width * 4]()


# === Variant C: comptime for + insert into a fixed-width register ===

@always_inline
def bcast_insert[width: Int](
    b4: SIMD[DType.uint8, 4],
) -> SIMD[DType.uint8, width * 4]:
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset = lane * 4](b4)
    return out


# === Loaders that pull 4 bytes from memory then call each variant ===

@no_inline
def use_join_chain(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    weights: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
):
    var b4 = src.load[width=4]()
    var bcast = bcast_join_chain[WIDTH](b4)
    var w = weights.bitcast[Scalar[DType.int8]]().load[width=BCAST_BYTES]()
    var acc = vpdpbusd[WIDTH](SIMD[DType.int32, WIDTH](0), bcast, w)
    dst.store(acc)


@no_inline
def use_insert(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    weights: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
):
    var b4 = src.load[width=4]()
    var bcast = bcast_insert[WIDTH](b4)
    var w = weights.bitcast[Scalar[DType.int8]]().load[width=BCAST_BYTES]()
    var acc = vpdpbusd[WIDTH](SIMD[DType.int32, WIDTH](0), bcast, w)
    dst.store(acc)


@no_inline
def run_case() -> Int:
    var src = InlineArray[UInt8, 4](fill=UInt8(0))
    var weights = InlineArray[UInt8, BCAST_BYTES](fill=UInt8(0))
    var out_a = InlineArray[Scalar[DType.int32], WIDTH](
        fill=Scalar[DType.int32](0))
    var out_c = InlineArray[Scalar[DType.int32], WIDTH](
        fill=Scalar[DType.int32](0))

    var s = Int32(7)
    for i in range(4):
        s = (s * 17 + 29) % 101
        src[i] = UInt8((Int(s) & 0xFF))
    var w = Int32(11)
    for i in range(BCAST_BYTES):
        w = (w * 31 + 13) % 251
        weights[i] = UInt8(Int(w) & 0xFF)

    var src_p = UnsafePointer(to=src).bitcast[UInt8]()
    var weights_p = UnsafePointer(to=weights).bitcast[UInt8]()

    use_join_chain(src_p, weights_p,
        UnsafePointer(to=out_a).bitcast[Scalar[DType.int32]]())
    use_insert(src_p, weights_p,
        UnsafePointer(to=out_c).bitcast[Scalar[DType.int32]]())

    var checksum = Int(0)
    for i in range(WIDTH):
        checksum += Int(out_a[i])
        checksum += Int(out_c[i])
    return checksum


def main():
    keep(run_case())
