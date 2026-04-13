"""VNNI int8 weight packing — 6D blocked layout for vpdpbusd.

The vpdpbusd instruction consumes weights as 16 dwords per register,
each dword holding 4 consecutive K values for one output channel.
The 6D layout is:

    [N/N_BLOCK, K/K_BLOCK, N_BLOCK/N_STEP, K_BLOCK/K_STEP,
     N_STEP/TILE_N, K_STEP/VNNI_BLK, TILE_N, VNNI_BLK]

Innermost [TILE_N=16, VNNI_BLK=4]: 16 output channels x 4 K values = 64 bytes.
This is achieved by copying N_STEP x K_STEP tiles and applying a 16x16
dword transpose on each TILE_N sub-tile, keeping 4-byte K groups intact.

Block sizes for cache locality:
    K_BLOCK = K    (full K — no redundant outer iterations)
    N_BLOCK = largest multiple of N_STEP that keeps tile <= L2_TARGET
"""

from std.collections import InlineArray
from std.memory import UnsafePointer, memcpy

from modeling.model_spec import PackingStrategy, PackFn
from simd_math.matrixops import transpose_generic


# =============================================================================
# VNNI packing constants
# =============================================================================

comptime L2_TARGET = 256 * 1024
comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4


# =============================================================================
# VnniPacked — packing strategy for vpdpbusd weight layout
# =============================================================================


@always_inline
def compute_n_block(n: Int, k: Int) -> Int:
    """Largest multiple of N_STEP that fits in L2_TARGET bytes.
    K_BLOCK is always K, so tile bytes = n_block * K."""
    var max_n = L2_TARGET // k
    var n_block = (max_n // VNNI_N_STEP) * VNNI_N_STEP
    if n_block >= n:
        return n
    if n_block >= VNNI_N_STEP:
        return n_block
    return VNNI_N_STEP


def pack_vnni(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Pack [rows, cols] int8 from src (row-major) into dst (6D VNNI).

    src and dst must not overlap. Reads row-major data from src,
    writes packed tiles with dword-transposed sub-tiles to dst.
    Conforms to PackFn.
    """
    var N = rows
    var K = cols
    debug_assert(K % VNNI_K_STEP == 0,
        "pack_vnni: K must be a multiple of VNNI_K_STEP (64)")
    debug_assert(N % VNNI_N_STEP == 0,
        "pack_vnni: N must be a multiple of VNNI_N_STEP (32)")
    var n_block = compute_n_block(N, K)
    var k_block = K
    var scratch = InlineArray[SIMD[DType.int32, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, n_block):
        var n_block_size = min(n_block, N - n_block_begin)

        for k_block_begin in range(0, K, k_block):
            var k_block_size = min(k_block, K - k_block_begin)

            for n_begin in range(0, n_block_size, VNNI_N_STEP):
                for k_begin in range(0, k_block_size, VNNI_K_STEP):
                    var tile_base = (
                        n_block_begin * K
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * VNNI_N_STEP
                    )

                    for i in range(VNNI_N_STEP):
                        var src_off = (n_block_begin + n_begin + i) * K + k_block_begin + k_begin
                        var dst_off = tile_base + i * VNNI_K_STEP
                        memcpy(
                            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=Int(dst) + dst_off),
                            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=Int(src) + src_off),
                            count=VNNI_K_STEP,
                        )

                    comptime dword_stride = VNNI_K_STEP // VNNI_BLK
                    var t0 = (dst + tile_base).bitcast[Int32]()
                    var t1 = (dst + tile_base + VNNI_TILE_N * VNNI_K_STEP).bitcast[Int32]()
                    transpose_generic[DType.int32, 16](t0, dword_stride, t0, VNNI_TILE_N, scratch)
                    transpose_generic[DType.int32, 16](t1, dword_stride, t1, VNNI_TILE_N, scratch)


struct VnniPacked(PackingStrategy):
    comptime PACK_FN = pack_vnni
