"""Dense down_proj workaround for TP-constrained FWHT block sizes.

When INTERMEDIATE doesn't divide cleanly by (tp * VNNI_K_STEP), the dense
down_proj needs a smaller FWHT block to align shard boundaries. This file
provides variants of fused_gu_gelu_tanh and int8_gemv_blocked that support
fwht_blk < VNNI_K_STEP (e.g., 16 or 32).

This is a design-space workaround, not production code. The standard kernels
in dense_ffn.mojo should be used whenever the block size permits.

Changes from the standard kernels:
  - fused_gu_gelu_tanh: tiles the GEMV at VNNI_N_STEP (32), then sub-blocks
    the FWHT + quantize at the smaller fwht_blk. Decouples GEMV tile from
    rotation block.
  - int8_gemv_blocked: accumulates across multiple small K-blocks within each
    VNNI_K_STEP, dequantizing per sub-block while keeping each logical VNNI
    tile as a full SIMD value instead of width-sized storage slices.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.int8_gemv import gemv_row, dot
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.kernels.dense_ffn import FusedGuGeluTanhArgs, Int8GemvBlockedArgs
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr


# ============================================================================
# Phase 1 workaround: GEMV tiles at VNNI_N_STEP, FWHT sub-blocks at fwht_blk
# ============================================================================


comptime GEMV_TILE = VNNI_N_STEP


def fused_gu_gelu_tanh_worker_wa[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    """Workaround: decouples GEMV tile (VNNI_N_STEP) from FWHT block (fwht_blk).

    Tiles the gate/up GEMV at GEMV_TILE for VNNI compatibility, then applies
    FWHT and quantize in fwht_blk-sized sub-blocks within each tile.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "intermediate must be a multiple of fwht_blk")
    debug_assert(GEMV_TILE % fwht_blk == 0,
        "GEMV_TILE must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0, "K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime sub_blocks_per_tile = GEMV_TILE // fwht_blk
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + n_off * K, dequant,
                args.wscale + n_off, args.wcolsum + n_off,
                gate.bitcast[Scalar[DType.float32]]())

            var up_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + (intermediate + n_off) * K, dequant,
                args.wscale + intermediate + n_off,
                args.wcolsum + intermediate + n_off,
                up.bitcast[Scalar[DType.float32]]())

            var k = 0
            while k + width <= GEMV_TILE:
                var g = (gate + k).load[width=width]()
                var u = (up + k).load[width=width]()
                (gate + k).store(gelu_tanh_f32[width](g) * u)
                k += width

            for sb in range(sub_blocks_per_tile):
                var sb_off = sb * fwht_blk
                fwht_block[fwht_blk](gate + sb_off)
                blk_row[(local_n + sb_off) // fwht_blk] = absmax_quantize_i8[fwht_blk](
                    gate + sb_off, qi_row + local_n + sb_off)

            local_n += GEMV_TILE


def fused_gu_gelu_tanh_wa[intermediate: Int, K: Int, fwht_blk: Int,
                          P: BurstThreadPool](
    act_i8: I8Ptr, act_scale: F32Ptr,
    wpacked: U8Ptr, wscale: F32Ptr, wcolsum: F32Ptr,
    qi_out: I8Ptr, blk_scale: F32Ptr,
    seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dispatch workaround fused gate_up + GELU-tanh + FWHT(sub-block) + quantize."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime n_tiles = intermediate // GEMV_TILE
    comptime MAX_POOL_CAPACITY = 128
    comptime num_blk_per_row = intermediate // fwht_blk

    var zero_args = FusedGuGeluTanhArgs(
        I8Ptr(), F32Ptr(), U8Ptr(), F32Ptr(), F32Ptr(),
        I8Ptr(), F32Ptr(), 0, 0, 0)
    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](fill=zero_args)

    if seq_len == 1:
        var num_workers = min(n_tiles, pool.get_capacity())
        var tiles_per_worker = (n_tiles + num_workers - 1) // num_workers
        for i in range(num_workers):
            var tile_start = i * tiles_per_worker
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * GEMV_TILE
            var n_count = (tile_end - tile_start) * GEMV_TILE
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8, act_scale, wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_scale + (n_start // fwht_blk),
                n_start, n_count, 1)
        pool.dispatch[FusedGuGeluTanhArgs,
            fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)
    else:
        var num_workers = min(seq_len, pool.get_capacity())
        var rows_per_worker = (seq_len + num_workers - 1) // num_workers
        for i in range(num_workers):
            var row_start = i * rows_per_worker
            var row_count = min(rows_per_worker, seq_len - row_start)
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8 + row_start * K, act_scale + row_start,
                wpacked, wscale, wcolsum,
                qi_out + row_start * intermediate,
                blk_scale + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs,
            fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)

    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Phase 2 workaround: blocked GEMV with sub-VNNI_K_STEP block sizes
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
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
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


def gemv_row_blocked_wa[N: Int, K: Int, fwht_blk: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    """Blocked GEMV supporting fwht_blk <= VNNI_K_STEP.

    Traverses packed data at VNNI_K_STEP granularity (tile0 all dc's, then
    tile1 all dc's) matching the 6D VNNI layout. Each logical VNNI tile is
    tracked as a full SIMD value and only sliced at the dot-product boundary
    to match the native hardware width. Dequantizes at fwht_blk boundaries
    within each K_STEP."""
    debug_assert(K % VNNI_K_STEP == 0, "K must be a multiple of VNNI_K_STEP")
    debug_assert(K % fwht_blk == 0, "K must be a multiple of fwht_blk")
    debug_assert(N % VNNI_N_STEP == 0, "N must be a multiple of VNNI_N_STEP")
    debug_assert(VNNI_K_STEP % fwht_blk == 0, "VNNI_K_STEP must be a multiple of fwht_blk")

    comptime width = gemv_tile_width[DType.int32, VNNI_TILE_N]()
    comptime dc_per_kstep = VNNI_K_STEP // VNNI_BLK
    comptime dc_per_fwht_blk = fwht_blk // VNNI_BLK
    comptime sub_blks_per_kstep = VNNI_K_STEP // fwht_blk
    comptime tiles_per_nstep = VNNI_N_STEP // VNNI_TILE_N

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
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
                        var cs = (block_colsums + blk_idx * N + n0 + tile_off).load[width=VNNI_TILE_N]()
                        f32_tiles[tile] += (
                            block_tiles[idx].cast[DType.float32]() - 128.0 * cs
                        ) * dq

            var n0 = nb + ns
            comptime for tile in range(tiles_per_nstep):
                comptime tile_off = tile * VNNI_TILE_N
                (dst + n0 + tile_off).store(
                    f32_tiles[tile] * (wsc + n0 + tile_off).load[width=VNNI_TILE_N]())


def int8_gemv_blocked_wa_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    comptime width = simd_width_of[DType.float32]()
    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked_wa[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= N:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked_wa[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: I8Ptr, wpacked: U8Ptr, blk_scale: F32Ptr,
    wscale: F32Ptr, blk_colsum: F32Ptr, dst: BF16Ptr,
    seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dispatch workaround blocked GEMV for sub-VNNI_K_STEP block sizes."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    comptime num_blocks = K // fwht_blk

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr(), Float32(0))
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)
    for i in range(num_jobs):
        var start = i
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K, wpacked,
            blk_scale + start * num_blocks,
            wscale, blk_colsum,
            dst + start * N, Float32(1.0))

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_wa_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))
