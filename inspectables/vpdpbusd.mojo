"""Smoke test for vpdpbusd intrinsic — check assembly is clean."""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.sys import llvm_intrinsic
from std.benchmark import keep

comptime W = simd_width_of[DType.int32]()


@always_inline
def vpdpbusd_string[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def vpdpbusd_static[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    comptime name = StaticString("llvm.x86.avx512.vpdpbusd.") + StaticString(String(width * 32))
    return llvm_intrinsic[
        name,
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def vpdpbusd_literal[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """Hardcoded intrinsic name per width — no string machinery at all."""
    comptime if width == 4:
        return llvm_intrinsic["llvm.x86.avx512.vpdpbusd.128", SIMD[DType.int32, width]](acc, a, b)
    elif width == 8:
        return llvm_intrinsic["llvm.x86.avx512.vpdpbusd.256", SIMD[DType.int32, width]](acc, a, b)
    elif width == 16:
        return llvm_intrinsic["llvm.x86.avx512.vpdpbusd.512", SIMD[DType.int32, width]](acc, a, b)
    else:
        constrained[False, "vpdpbusd: unsupported width"]()
        return acc


@no_inline
def test_string(
    acc: SIMD[DType.int32, W],
    a: SIMD[DType.int8, W * 4],
    b: SIMD[DType.int8, W * 4],
) -> SIMD[DType.int32, W]:
    return vpdpbusd_string[W](acc, a, b)


@no_inline
def test_static(
    acc: SIMD[DType.int32, W],
    a: SIMD[DType.int8, W * 4],
    b: SIMD[DType.int8, W * 4],
) -> SIMD[DType.int32, W]:
    return vpdpbusd_static[W](acc, a, b)


@no_inline
def test_literal(
    acc: SIMD[DType.int32, W],
    a: SIMD[DType.int8, W * 4],
    b: SIMD[DType.int8, W * 4],
) -> SIMD[DType.int32, W]:
    return vpdpbusd_literal[W](acc, a, b)


def main():
    var acc = SIMD[DType.int32, W](0)
    var a = SIMD[DType.int8, W * 4](1)
    var b = SIMD[DType.int8, W * 4](2)

    var r1 = test_string(acc, a, b)
    var r2 = test_static(acc, a, b)
    var r3 = test_literal(acc, a, b)

    keep(r1)
    keep(r2)
    keep(r3)
