"""SIMD matrix operations — generic transpose and layout transforms.

The butterfly transpose is generic over N and emits optimal SIMD shuffle
sequences via comptime-unrolled interleave stages. Works for any power-of-2 N.
"""

from std.collections import InlineArray
from std.memory import UnsafePointer
from std.utils import IndexList


# =============================================================================
# Compile-time helpers
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


# =============================================================================
# Butterfly transpose — generic over N, optimal SIMD shuffles
# =============================================================================


comptime Row = SIMD[DType.int8, _]
comptime Int8Ptr = UnsafePointer[Int8, origin = _]


@always_inline
def interleave[N: Int, stride: Int, high: Bool](a: Row[N], b: Row[N]) -> Row[N]:
    comptime idx = interleave_mask[N, stride, high]()
    return a.shuffle[mask=idx](b)


@always_inline
def transpose[N: Int](
    src: Int8Ptr, src_stride: Int,
    dst: Int8Ptr[MutAnyOrigin], dst_stride: Int,
    mut scratch: InlineArray[Row[N], N],
):
    """In-register NxN byte transpose via butterfly interleave network.

    Loads N rows of N bytes from src (strided), performs log2(N) stages
    of interleave shuffles, then stores N rows to dst (strided).
    The scratch buffer holds the intermediate SIMD registers.
    """
    comptime for i in range(N):
        scratch[i] = (src + i * src_stride).load[width=N]()

    comptime num_stages = log2[N]()
    comptime for stage in range(num_stages):
        comptime stride = 1 << stage
        comptime groups = N // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime idx0 = g * 2 * stride + j
                comptime idx1 = idx0 + stride
                var lo = interleave[N, stride, False](scratch[idx0], scratch[idx1])
                var hi = interleave[N, stride, True](scratch[idx0], scratch[idx1])
                scratch[idx0] = lo
                scratch[idx1] = hi

    # Bit-reverse to get sequential order. LLVM optimizes via register allocation.
    comptime for i in range(N):
        comptime j = bit_reverse[num_stages, i]()
        comptime if i < j:
            var tmp = scratch[i]
            scratch[i] = scratch[j]
            scratch[j] = tmp

    comptime for i in range(N):
        (dst + i * dst_stride).store(scratch[i])


@always_inline
def transpose_16x16(
    src: Int8Ptr, src_stride: Int,
    dst: Int8Ptr[MutAnyOrigin], dst_stride: Int,
    mut scratch: InlineArray[Row[16], 16],
):
    transpose[16](src, src_stride, dst, dst_stride, scratch)


@always_inline
def transpose_32x32(
    src: Int8Ptr, src_stride: Int,
    dst: Int8Ptr[MutAnyOrigin], dst_stride: Int,
    mut scratch: InlineArray[Row[32], 32],
):
    transpose[32](src, src_stride, dst, dst_stride, scratch)
