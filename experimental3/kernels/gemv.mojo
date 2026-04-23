"""Per-row GEMV kernels — one activation row x weight matrix -> N outputs.

These are the row-level math bodies. Dispatchers that fan out across
seq_len/N/expert (and so handle GEMM-shaped work as M-parallel GEMVs) live
in gemm.mojo. Pure dot primitives live in dot_prod.mojo.

Three blocked-dequant variants:
  gemv_row             per-row weight scale, per-row act scale
  gemv_row_blocked     per-row weight scale, per-K-block act scale
  gemv_row_blocked_wa  same as above, supports fwht_blk < VNNI_K_STEP

Plus:
  lm_head_row_dot      reduce-to-scalar per-K-block variant (decode lm head)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import (
    vpdpbusd, dot, dot_tile_chunked, gemv_tile_width,
)
from experimental3.common_math import I8Ptr, F32Ptr


# ============================================================================
# gemv_row — per-row dequant (classic)
# ============================================================================


def gemv_row[N: Int, K: Int, OutDType: DType](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
    subrange: Int = N,
):
    """One activation row x VNNI-packed weight -> subrange output elements.

    subrange defaults to N. For N-split decode, pre-offset pointers to
    n_start and pass the sub-range count. Must be a multiple of VNNI_N_STEP.
    """
    debug_assert(K % VNNI_K_STEP == 0,
        "gemv_row: K must be a multiple of VNNI_K_STEP (64)")
    debug_assert(subrange % VNNI_N_STEP == 0,
        "gemv_row: subrange must be a multiple of VNNI_N_STEP (32)")
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(subrange, K)
    var packed_off = 0

    for nb in range(0, subrange, n_block):
        var nb_size = min(n_block, subrange - nb)

        for ns in range(0, nb_size, VNNI_N_STEP):
            var acc_buf = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0)
            )
            var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, width]]()

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[p] = dot[width](
                            acc[p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[passes_per_subtile + p] = dot[width](
                            acc[passes_per_subtile + p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

            for a in range(acc_count):
                var n_base = nb + ns + a * width
                var corrected = acc[a].cast[DType.float32]() - Float32(128) * (wcs + n_base).load[width=width]()
                var result = corrected * act_sc * (wsc + n_base).load[width=width]()
                (dst + n_base).store(result.cast[OutDType]())


# ============================================================================
# gemv_row_blocked — per-K-block activation scales
# ============================================================================


def gemv_row_blocked[N: Int, K: Int, fwht_block_size: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    subrange: Int = N,
    colsum_stride: Int = N,
):
    """GEMV with per-K-block activation scales. Accumulates per block in i32,
    dequants to f32 per block, then applies per-row weight scale.

    subrange defaults to N. For N-split decode, pre-offset pointers and
    pass the sub-range count. colsum_stride is the row-stride of
    block_colsums (always full matrix N, since colsums are not split).
    subrange must be a multiple of VNNI_N_STEP (32).
    """
    debug_assert(K % fwht_block_size == 0,
        "gemv_row_blocked: K must be a multiple of fwht_block_size")
    debug_assert(fwht_block_size >= VNNI_K_STEP,
        "gemv_row_blocked: fwht_block_size must be >= VNNI_K_STEP (64)")
    debug_assert(subrange % VNNI_N_STEP == 0,
        "gemv_row_blocked: subrange must be a multiple of VNNI_N_STEP (32)")
    comptime num_blocks = K // fwht_block_size
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(subrange, K)
    var packed_off = 0

    for nb in range(0, subrange, n_block):
        var nb_size = min(n_block, subrange - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_acc = InlineArray[SIMD[DType.float32, width], acc_count](
                fill=SIMD[DType.float32, width](0))
            for blk in range(num_blocks):
                var i32_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                    fill=SIMD[DType.int32, width](0))
                for ks in range(0, fwht_block_size, VNNI_K_STEP):
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[p] = dot[width](
                                i32_acc[p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos,
                            )
                        packed_off += VNNI_TILE_N * VNNI_BLK
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[passes_per_subtile + p] = dot[width](
                                i32_acc[passes_per_subtile + p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos,
                            )
                        packed_off += VNNI_TILE_N * VNNI_BLK
                var blk_dequant = block_scales[blk] / 127.0
                for a in range(acc_count):
                    var n_base = nb + ns + a * width
                    var corrected = i32_acc[a].cast[DType.float32]() - 128.0 * (block_colsums + blk * colsum_stride + n_base).load[width=width]()
                    f32_acc[a] += corrected * blk_dequant
            for a in range(acc_count):
                var n_base = nb + ns + a * width
                (dst + n_base).store(f32_acc[a] * (wsc + n_base).load[width=width]())


# ============================================================================
# gemv_row_blocked_wa — supports fwht_blk <= VNNI_K_STEP
# ============================================================================


def gemv_row_blocked_wa[N: Int, K: Int, fwht_blk: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    subrange: Int = N,
    colsum_stride: Int = N,
):
    """Blocked GEMV supporting fwht_blk <= VNNI_K_STEP.

    Traverses packed data at VNNI_K_STEP granularity (tile0 all dc's, then
    tile1 all dc's) matching the 6D VNNI layout. Each logical VNNI tile is
    tracked as a full SIMD value and only sliced at the dot-product boundary
    to match the native hardware width. Dequantizes at fwht_blk boundaries
    within each K_STEP."""
    debug_assert(K % VNNI_K_STEP == 0, "K must be a multiple of VNNI_K_STEP")
    debug_assert(K % fwht_blk == 0, "K must be a multiple of fwht_blk")
    debug_assert(subrange % VNNI_N_STEP == 0, "subrange must be a multiple of VNNI_N_STEP")
    debug_assert(VNNI_K_STEP % fwht_blk == 0, "VNNI_K_STEP must be a multiple of fwht_blk")

    comptime width = gemv_tile_width[DType.int32, VNNI_TILE_N]()
    comptime dc_per_kstep = VNNI_K_STEP // VNNI_BLK
    comptime dc_per_fwht_blk = fwht_blk // VNNI_BLK
    comptime sub_blks_per_kstep = VNNI_K_STEP // fwht_blk
    comptime tiles_per_nstep = VNNI_N_STEP // VNNI_TILE_N

    var n_block = compute_n_block(subrange, K)
    var packed_off = 0

    for nb in range(0, subrange, n_block):
        var nb_size = min(n_block, subrange - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_tiles = InlineArray[SIMD[DType.float32, VNNI_TILE_N], tiles_per_nstep](
                fill=SIMD[DType.float32, VNNI_TILE_N](0))

            for ks in range(0, K, VNNI_K_STEP):
                var block_tiles = InlineArray[
                    SIMD[DType.int32, VNNI_TILE_N],
                    sub_blks_per_kstep * tiles_per_nstep
                ](fill=SIMD[DType.int32, VNNI_TILE_N](0))

                comptime for tile in range(tiles_per_nstep):
                    for dc in range(dc_per_kstep):
                        var sb = dc // dc_per_fwht_blk
                        var idx = sb * tiles_per_nstep + tile
                        var k_pos = ks + dc * VNNI_BLK
                        block_tiles[idx] = dot_tile_chunked[width](
                            block_tiles[idx], act_row, wpacked + packed_off, k_pos)
                        packed_off += VNNI_TILE_N * VNNI_BLK

                var n0 = nb + ns
                for sb in range(sub_blks_per_kstep):
                    var blk_idx = ks // fwht_blk + sb
                    var dq = block_scales[blk_idx] / 127.0
                    comptime for tile in range(tiles_per_nstep):
                        var idx = sb * tiles_per_nstep + tile
                        comptime tile_off = tile * VNNI_TILE_N
                        var cs = (block_colsums + blk_idx * colsum_stride + n0 + tile_off).load[width=VNNI_TILE_N]()
                        f32_tiles[tile] += (
                            block_tiles[idx].cast[DType.float32]() - 128.0 * cs
                        ) * dq

            var n0 = nb + ns
            comptime for tile in range(tiles_per_nstep):
                comptime tile_off = tile * VNNI_TILE_N
                (dst + n0 + tile_off).store(
                    f32_tiles[tile] * (wsc + n0 + tile_off).load[width=VNNI_TILE_N]())


# ============================================================================
# lm_head_row_dot — reduce-to-scalar per-K-block variant
# ============================================================================


@always_inline
def lm_head_row_dot[K: Int, fwht_blk: Int](
    act: I8Ptr,
    weight_row: I8Ptr,
    act_blk_scales: F32Ptr,
    w_blk_scales_row: F32Ptr,
    w_blk_colsums_row: F32Ptr,
) -> Float32:
    """Fully-dequant'd dot product for one output row."""
    comptime dp_width = simd_width_of[DType.int32]()
    comptime vnni_k_step = dp_width * 4
    debug_assert(K % fwht_blk == 0, "K must be a multiple of fwht_blk")
    debug_assert(
        fwht_blk % vnni_k_step == 0,
        "fwht_blk must be a multiple of the VNNI dot-product step",
    )
    comptime num_blocks = K // fwht_blk
    comptime steps_per_block = fwht_blk // vnni_k_step

    var act_u8 = act.bitcast[UInt8]()
    var u8_bias = SIMD[DType.uint8, vnni_k_step](0x80)
    var inv_i8_max = Float32(1.0) / Float32(127.0)
    var colsum_bias = Float32(128.0)
    var total = Float32(0)
    for b in range(num_blocks):
        var i32_acc = SIMD[DType.int32, dp_width](0)
        var k_base = b * fwht_blk
        comptime for s in range(steps_per_block):
            comptime step_k_off = s * vnni_k_step
            var k_off = k_base + step_k_off
            var act_step = (act_u8 + k_off).load[width=vnni_k_step]() ^ u8_bias
            var w_i8 = (weight_row + k_off).load[width=vnni_k_step]()
            i32_acc = vpdpbusd[dp_width](i32_acc, act_step, w_i8)
        var block_dot = i32_acc.reduce_add().cast[DType.float32]()
        var corrected = block_dot - colsum_bias * w_blk_colsums_row[b]
        var act_dequant = act_blk_scales[b] * inv_i8_max
        total += corrected * act_dequant * w_blk_scales_row[b]
    return total
