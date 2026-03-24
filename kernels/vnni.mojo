"""VNNI int8 weight packing — blocked transpose for vpdpbusd layout.

The vpdpbusd instruction expects weights arranged so that 4 consecutive
int8 values from one output channel occupy one 32-bit lane. This requires
transposing the weight matrix in 16x16 tiles, with two-level blocking
for cache locality.

Tile and stride parameters are fixed by the VNNI instruction layout:
    TILE_N = 16    (16 output channels per tile, matching xmm/ymm lane count)
    N_STEP = 32    (2 tiles of 16 for ymm processing)
    K_STEP = 64    (64 K elements per tile, matching cache line)
"""

from std.collections import InlineArray
from std.memory import UnsafePointer

from modeling.model_spec import PackingStrategy, PackFn
from simd_math.matrixops import transpose_16x16, Row


# =============================================================================
# VnniPacking — packing strategy for vpdpbusd kernels
# =============================================================================


def pack_vnni_runtime[n_block: Int, k_block: Int](
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Runtime N/K, comptime N_BLOCK/K_BLOCK. Conforms to PackFn."""
    comptime TILE_N = 16
    comptime TILE_K = 64
    comptime N_STEP = 32
    comptime K_STEP = 64

    var N = rows
    var K = cols
    var s = src.bitcast[Int8]()
    var d = dst.bitcast[Int8]()
    var transpose_buf = InlineArray[SIMD[DType.int8, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, n_block):
        var n_block_end = min(n_block_begin + n_block, N)
        var n_block_size = n_block_end - n_block_begin

        for k_block_begin in range(0, K, k_block):
            var k_block_end = min(k_block_begin + k_block, K)
            var k_block_size = k_block_end - k_block_begin

            for n_begin in range(0, n_block_size, N_STEP):
                for k_begin in range(0, k_block_size, K_STEP):
                    var tile_base = (
                        n_block_begin * K
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * N_STEP
                    )

                    for i in range(N_STEP):
                        var src_ptr = s + ((n_block_begin + n_begin + i) * K + k_block_begin + k_begin)
                        var dst_ptr = d + (tile_base + i * K_STEP)
                        dst_ptr.store(src_ptr.load[width=K_STEP]())

                    var tile0 = d + tile_base
                    var tile1 = d + (tile_base + TILE_N * K_STEP)
                    transpose_16x16(tile0, K_STEP, tile0, TILE_N, transpose_buf)
                    transpose_16x16(tile1, K_STEP, tile1, TILE_N, transpose_buf)


struct VnniPacking[n_block: Int, k_block: Int](PackingStrategy):
    comptime N_BLOCK = Self.n_block
    comptime K_BLOCK = Self.k_block
    comptime PACK_FN = pack_vnni_runtime[Self.n_block, Self.k_block]


# =============================================================================
# Comptime-dimension pack (for use outside the quantizer pipeline)
# =============================================================================


def pack_vnni_int8[
    K: Int, N: Int,
    N_BLOCK: Int, K_BLOCK: Int,
    TILE_K: Int = 64, TILE_N: Int = 16,
    N_STEP: Int = 32, K_STEP: Int = 64,
](
    src: UnsafePointer[Int8, ImmutAnyOrigin],
    dst: UnsafePointer[Int8, MutAnyOrigin],
):
    """Pack a [K, N] int8 weight matrix into blocked VNNI layout.
    Two-level blocking: outer N_BLOCK x K_BLOCK, inner N_STEP x K_STEP tiles.
    Each tile is copied then transposed in-place via 16x16 sub-tiles."""
    comptime assert N % N_STEP == 0, "N must be divisible by N_STEP"
    comptime assert K % K_STEP == 0, "K must be divisible by K_STEP"
    comptime assert N_STEP == TILE_N * 2, "N_STEP must be 2*TILE_N"
    comptime assert K_STEP == TILE_K, "K_STEP must equal TILE_K"

    var transpose_buf = InlineArray[SIMD[DType.int8, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, N_BLOCK):
        var n_block_end = min(n_block_begin + N_BLOCK, N)
        var n_block_size = n_block_end - n_block_begin

        for k_block_begin in range(0, K, K_BLOCK):
            var k_block_end = min(k_block_begin + K_BLOCK, K)
            var k_block_size = k_block_end - k_block_begin

            for n_begin in range(0, n_block_size, N_STEP):
                for k_begin in range(0, k_block_size, K_STEP):
                    var tile_base = (
                        n_block_begin * K
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * N_STEP
                    )

                    for i in range(N_STEP):
                        var src_ptr = src + ((n_block_begin + n_begin + i) * K + k_block_begin + k_begin)
                        var dst_ptr = dst + (tile_base + i * K_STEP)
                        dst_ptr.store(src_ptr.load[width=K_STEP]())

                    var tile0 = dst + tile_base
                    var tile1 = dst + (tile_base + TILE_N * K_STEP)
                    transpose_16x16(tile0, K_STEP, tile0, TILE_N, transpose_buf)
                    transpose_16x16(tile1, K_STEP, tile1, TILE_N, transpose_buf)
