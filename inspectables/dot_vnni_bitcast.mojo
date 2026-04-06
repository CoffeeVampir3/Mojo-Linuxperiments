"""Inspect the dot_vnni dword broadcast + type reinterpret.

The question: does the InlineArray store-reload get elided, or does
it actually hit the stack? Compare the current approach (memory round-trip)
against a join-based alternative that stays in registers.
"""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from std.benchmark import keep


comptime W = simd_width_of[DType.int32]()
comptime W4 = W * 4


@always_inline
def vpdpbusd(
    acc: SIMD[DType.int32, W],
    a: SIMD[DType.uint8, W4],
    b: SIMD[DType.int8, W4],
) -> SIMD[DType.int32, W]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(W * 32),
        SIMD[DType.int32, W],
    ](acc, a, b)


# Current approach: broadcast u32, store to InlineArray, reload as u8
@no_inline
def dot_current(
    acc: SIMD[DType.int32, W],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, W]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width=W4]()
    var dword = (act_row + k_pos).bitcast[Scalar[DType.uint32]]()[0] ^ UInt32(0x80808080)
    var dwords = SIMD[DType.uint32, W](dword)
    var tmp = InlineArray[SIMD[DType.uint32, W], 1](fill=dwords)
    var a = UnsafePointer(to=tmp).bitcast[UInt8]().load[width=W4]()
    return vpdpbusd(acc, a, w)


# Alternative: load 4 bytes, join to target width — no memory round-trip
@no_inline
def dot_join(
    acc: SIMD[DType.int32, W],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, W]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width=W4]()
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[DType.uint8, 4](0x80)
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    var b32 = b16.join(b16)
    var b64 = b32.join(b32)
    # Slice to the exact width needed
    var a = b64.slice[W4]()
    return vpdpbusd(acc, a, w)


def main():
    var act = alloc[Scalar[DType.int8]](64)
    var weights = alloc[UInt8](64 * W)

    for i in range(64):
        act[i] = Scalar[DType.int8](i % 127)
    for i in range(64 * W):
        weights[i] = UInt8(i % 255)

    var acc = SIMD[DType.int32, W](0)

    var r1 = dot_current(acc, act, weights, 0)
    var r2 = dot_join(acc, act, weights, 0)

    keep(r1)
    keep(r2)

    act.free()
    weights.free()
