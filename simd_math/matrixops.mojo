"""SIMD matrix operations — generic transpose and layout transforms.

The butterfly transpose is generic over N and element type, emitting optimal
SIMD shuffle sequences via comptime-unrolled interleave stages. Works for
any power-of-2 N and any DType (int8 for byte transpose, int32 for dword).
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
# Generic butterfly transpose — parameterized on element DType
# =============================================================================


@always_inline
def simd_interleave[T: DType, N: Int, stride: Int, high: Bool](
    a: SIMD[T, N], b: SIMD[T, N],
) -> SIMD[T, N]:
    comptime idx = interleave_mask[N, stride, high]()
    return a.shuffle[mask=idx](b)


@always_inline
def transpose_generic[T: DType, N: Int](
    src: UnsafePointer[Scalar[T], _], src_stride: Int,
    dst: UnsafePointer[Scalar[T], MutAnyOrigin], dst_stride: Int,
    mut scratch: InlineArray[SIMD[T, N], N],
):
    """In-register NxN transpose via butterfly interleave network.

    Generic over element type: int8 for byte transpose, int32 for dword.
    Loads N rows of N elements from src (strided by elements), performs
    log2(N) stages of interleave shuffles, then stores N rows to dst.
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



