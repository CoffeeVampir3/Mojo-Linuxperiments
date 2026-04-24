"""Dot-product primitives — VNNI intrinsic, broadcast helpers, scalar fallback.

Pure ISA layer: one call produces SIMD-lane-local i32 accumulators. Shared by
every int8 GEMV/GEMM kernel (gemv.mojo, gemm.mojo) and by attention scoring
(sliding_attention.mojo).

All VNNI paths compute `u8 × i8 → i32` via `vpdpbusd`. Activation bytes are
biased from i8 to u8 by the caller (typically XOR 0x80); the `-128 · colsum`
correction in the consuming kernel cancels the bias.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, CompilationTarget
from std.sys import llvm_intrinsic

from kernels.vnni import VNNI_TILE_N, VNNI_BLK


# ============================================================================
# VNNI intrinsic
# ============================================================================


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """x86 VNNI: u8 x i8 -> i32 dot product accumulate.
    Per dword lane i: acc[i] += sum_{j=0..3} u8(a[i].byte[j]) * i8(b[i].byte[j])
    """
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


# ============================================================================
# 4-byte -> width*4 broadcast (activation/query side of VNNI dots)
# ============================================================================


@always_inline
def bcast_4_vnni[width: Int](
    b4: SIMD[DType.uint8, 4],
) -> SIMD[DType.uint8, width * 4]:
    """Broadcast a 4-byte sequence into `width` dword lanes (the VNNI
    dword-broadcast shape). Lowers to a single vpbroadcastd, which
    `vpdpbusd` can fold via its {1to16} broadcast modifier — verified in
    inspectables/bcast_4_vnni.mojo.
    """
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset = lane * 4](b4)
    return out


@always_inline
def bcast_4u8_vnni[width: Int](
    p: UnsafePointer[UInt8, MutAnyOrigin],
) -> SIMD[DType.uint8, width * 4]:
    """Load 4 u8 bytes at `p` and broadcast to width*4 lanes."""
    return bcast_4_vnni[width](p.load[width=4]())


@always_inline
def act_broadcast_vnni[width: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.uint8, width * 4]:
    """Load 4 i8 activation bytes at k_pos, XOR 0x80 to u8-bias, replicate."""
    return bcast_4_vnni[width](
        (act_row + k_pos).bitcast[UInt8]().load[width=4]()
        ^ SIMD[DType.uint8, 4](0x80))


# ============================================================================
# VNNI dot — activation broadcast, weight VNNI-packed
# ============================================================================


@always_inline
def dot_vnni_broadcasted[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> SIMD[DType.int32, width]:
    var w = wpacked.load[width = width * 4, non_temporal=True]()
    return vpdpbusd[width](acc, act_bytes, w)


@always_inline
def dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    return dot_vnni_broadcasted[width](
        acc, act_broadcast_vnni[width](act_row, k_pos), wpacked,
    )


# ============================================================================
# Scalar fallback — widen u8/i8 to i32 and multiply explicitly
# ============================================================================


@always_inline
def dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """Non-VNNI fallback: width channels x 4 K values via widen-to-i32 multiply."""
    # i8 storage loaded as i32 dwords so we can mask out lanes with shifts.
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[width=width, non_temporal=True]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128) * ((wdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128) * ((wdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128) * ((wdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128) * (wdw >> 24)
    return result


# ============================================================================
# Runtime dispatcher
# ============================================================================


@always_inline
def dot[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return dot_vnni[width](acc, act_row, wpacked, k_pos)
    else:
        return dot_simd[width](acc, act_row, wpacked, k_pos)


# ============================================================================
# VNNI_TILE_N-wide tile dot — splits a full-tile accumulator across hardware
# SIMD width and reinserts the results. Used by sub-VNNI_K_STEP blocked GEMV.
# ============================================================================


def gemv_tile_width[T: DType, tile: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if tile <= hw:
        return tile
    else:
        return hw


@always_inline
def dot_tile_chunked[width: Int](
    acc: SIMD[DType.int32, VNNI_TILE_N],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, VNNI_TILE_N]:
    comptime assert VNNI_TILE_N % width == 0,
        "tile rewrite requires width to divide VNNI_TILE_N"
    comptime regs = VNNI_TILE_N // width
    var result = acc
    comptime for r in range(regs):
        comptime lane_off = r * width
        result = result.insert[offset=lane_off](
            dot[width](
                result.slice[width, offset=lane_off](),
                act_row,
                wpacked + lane_off * VNNI_BLK,
                k_pos,
            ))
    return result
