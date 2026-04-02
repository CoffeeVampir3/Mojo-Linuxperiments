"""Assembly inspection for pack_k_tile_vnni variants.

Compares: byte-by-byte, SIMD[uint8,4], and butterfly-transpose-based.
"""

from std.memory.unsafe_pointer import alloc
from std.collections import InlineArray
from std.utils import IndexList
from std.benchmark import keep


comptime K_STEP = 64
comptime VNNI_BLK = 4
comptime TILE_N = 16


# =============================================================================
# Inline butterfly transpose (can't import project modules from inspectables)
# =============================================================================

def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()

def bit_reverse[bits: Int, x: Int]() -> Int:
    comptime if bits == 0:
        return 0
    else:
        comptime lsb = x & 1
        comptime rest = x >> 1
        return (lsb << (bits - 1)) | bit_reverse[bits - 1, rest]()

def interleave_idx[N: Int, i: Int, stride: Int, high: Bool]() -> Int:
    comptime half = N // 2
    comptime src_offset = half if high else 0
    comptime pair = i // (2 * stride)
    comptime within = i % (2 * stride)
    comptime if within < stride:
        return src_offset + pair * stride + within
    else:
        return N + src_offset + pair * stride + (within - stride)

def interleave_mask[N: Int, stride: Int, high: Bool]() -> IndexList[N]:
    var result = IndexList[N]()
    comptime for i in range(N):
        result[i] = interleave_idx[N, i, stride, high]()
    return result

@always_inline
def simd_interleave[T: DType, N: Int, stride: Int, high: Bool](
    a: SIMD[T, N], b: SIMD[T, N],
) -> SIMD[T, N]:
    comptime idx = interleave_mask[N, stride, high]()
    return a.shuffle[mask=idx](b)

@always_inline
def butterfly_transpose[T: DType, N: Int](
    mut scratch: InlineArray[SIMD[T, N], N],
    dst: UnsafePointer[Scalar[T], MutAnyOrigin], dst_stride: Int,
):
    """Butterfly stages + bit-reverse + store. Scratch already loaded."""
    comptime num_stages = log2[N]()
    comptime for stage in range(num_stages):
        comptime stride = 1 << stage
        comptime groups = N // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime idx0 = g * 2 * stride + j
                comptime idx1 = idx0 + stride
                var lo = simd_interleave[T, N, stride, False](scratch[idx0], scratch[idx1])
                var hi = simd_interleave[T, N, stride, True](scratch[idx0], scratch[idx1])
                scratch[idx0] = lo
                scratch[idx1] = hi
    comptime for i in range(N):
        comptime j = bit_reverse[num_stages, i]()
        comptime if i < j:
            var tmp = scratch[i]
            scratch[i] = scratch[j]
            scratch[j] = tmp
    comptime for i in range(N):
        (dst + i * dst_stride).store(scratch[i])


# =============================================================================
# Variants
# =============================================================================

@no_inline
def pack_k_scalar(
    k_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    k_off: Int, n_off: Int, n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """4 individual byte loads/stores per column."""
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = k_base + (n_off + col) * head_dim + k_off + 4 * kg
            var d = kg * 64 + col * 4
            dst[d] = src[0]
            dst[d+1] = src[1]
            dst[d+2] = src[2]
            dst[d+3] = src[3]
        for col in range(n_cols, TILE_N):
            var d = kg * 64 + col * 4
            dst[d] = 0
            dst[d+1] = 0
            dst[d+2] = 0
            dst[d+3] = 0


@no_inline
def pack_k_simd4(
    k_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    k_off: Int, n_off: Int, n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """SIMD[uint8, 4] load/store."""
    comptime zero4 = SIMD[DType.uint8, 4](0)
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = k_base + (n_off + col) * head_dim + k_off + 4 * kg
            (dst + kg * 64 + col * 4).store(src.load[width=4]())
        for col in range(n_cols, TILE_N):
            (dst + kg * 64 + col * 4).store(zero4)


@no_inline
def pack_k_transpose(
    k_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    k_off: Int, n_off: Int, n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pre-load valid rows into scratch, zero-pad, butterfly transpose."""
    var scratch = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
        fill=SIMD[DType.uint32, TILE_N](0))
    for col in range(n_cols):
        scratch[col] = (k_base + (n_off + col) * head_dim + k_off).bitcast[Scalar[DType.uint32]]().load[width=TILE_N]()
    butterfly_transpose[DType.uint32, TILE_N](
        scratch, dst.bitcast[Scalar[DType.uint32]](), TILE_N)


def main():
    comptime HD = 128
    var k_data = alloc[UInt8](32 * HD)
    var dst1 = alloc[UInt8](K_STEP // VNNI_BLK * 64)
    var dst2 = alloc[UInt8](K_STEP // VNNI_BLK * 64)
    var dst3 = alloc[UInt8](K_STEP // VNNI_BLK * 64)

    for i in range(32 * HD):
        k_data[i] = UInt8(i % 256)

    pack_k_scalar(k_data, HD, 0, 0, 12, dst1)
    pack_k_simd4(k_data, HD, 0, 0, 12, dst2)
    pack_k_transpose(k_data, HD, 0, 0, 12, dst3)

    keep(dst1[0])
    keep(dst2[0])
    keep(dst3[0])

    k_data.free()
    dst1.free()
    dst2.free()
    dst3.free()
