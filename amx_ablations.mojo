from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max, min

from experimental3.amx import (
    TILE_M, TILE_K, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    TileConfig, make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbssd, tdpbusd, tdpbsud, tile_dp,
)
from simd_math import exp_f32, roundeven, set_subnormal_zeroing
from experimental3.kernels.dot_prod import vpdpbusd
from notstdcollections import AlignedInlineArray

comptime HPG = 6
comptime HEAD_DIM = 128
comptime WIDTH = 16
comptime NUM_K_STEPS = HEAD_DIM // K_STEP
comptime V_CHANNEL_GROUPS = HEAD_DIM // WIDTH
comptime SIMD_W = simd_width_of[DType.float32]()
comptime K_PG_BYTES = HEAD_DIM // VNNI_BLK * WIDTH * VNNI_BLK
comptime SCORE_STRIDE = TILE_M * WIDTH
comptime V_ROW_BYTES = HEAD_DIM
comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
comptime V_PG_BYTES_VNNI = V_CHANNEL_GROUPS * V_CG_BYTES

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]


def alloc_zeroed[T: DType](count: Int) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
    var p = alloc[Scalar[T]](count)
    for i in range(count):
        (p + i)[] = Scalar[T](0)
    return UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=Int(p))


def fill_random_i8(ptr: I8Ptr, count: Int):
    var state = UInt64(0xDEADBEEF12345678)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = Scalar[DType.int8]((state >> 33).cast[DType.int8]())


def fill_random_u8(ptr: U8Ptr, count: Int):
    var state = UInt64(0xCAFEBABE87654321)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = UInt8((state >> 33).cast[DType.uint8]())


# =============================================================================
# Scoring configurations
# =============================================================================


def score_1_1_1(q_ptr: I8Ptr, k_base: U8Ptr, score_buf: I32Ptr, num_pgs: Int):
    for pg in range(num_pgs):
        var k_pg = k_base + pg * K_PG_BYTES
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_pg + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)


def score_2_2_4(q_ptr: I8Ptr, k_base: U8Ptr, score_buf: I32Ptr, num_pgs: Int):
    var pg = 0
    while pg + 4 <= num_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + (pg + 1) * SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + (pg + 2) * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + (pg + 3) * SCORE_STRIDE, WIDTH * 4)
        pg += 4

    while pg < num_pgs:
        var k_pg = k_base + pg * K_PG_BYTES
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_pg + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        pg += 1


def score_2_1_5(q_ptr: I8Ptr, k_base: U8Ptr, score_buf: I32Ptr, num_pgs: Int):
    var pg = 0
    while pg + 5 <= num_pgs:
        tilezero[3]()
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        # PG 0
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[3, 0, 2]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[3, 1, 2]()
        # PG 1
        tileload[2, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tileload[2, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        # PG 2
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[5, 0, 2]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[5, 1, 2]()
        # PG 3
        tileload[2, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tileload[2, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        # PG 4
        tileload[2, DType.uint8](k_base + (pg + 4) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[7, 0, 2]()
        tileload[2, DType.uint8](k_base + (pg + 4) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[7, 1, 2]()

        tilestore[3, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        tilestore[4, DType.int32](score_buf + (pg + 1) * SCORE_STRIDE, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + (pg + 2) * SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + (pg + 3) * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + (pg + 4) * SCORE_STRIDE, WIDTH * 4)
        pg += 5

    while pg < num_pgs:
        var k_pg = k_base + pg * K_PG_BYTES
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_pg + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        pg += 1


def score_1_1_6(q_ptr: I8Ptr, k_base: U8Ptr, score_buf: I32Ptr, num_pgs: Int):
    var pg = 0
    while pg + 6 <= num_pgs:
        tilezero[2]()
        tilezero[3]()
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()

        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            # PG 0-5 serial K loads
            tileload[1, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[2, 0, 1]()
            tileload[1, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[3, 0, 1]()
            tileload[1, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 1]()
            tileload[1, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[5, 0, 1]()
            tileload[1, DType.uint8](k_base + (pg + 4) * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[6, 0, 1]()
            tileload[1, DType.uint8](k_base + (pg + 5) * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[7, 0, 1]()

        tilestore[2, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        tilestore[3, DType.int32](score_buf + (pg + 1) * SCORE_STRIDE, WIDTH * 4)
        tilestore[4, DType.int32](score_buf + (pg + 2) * SCORE_STRIDE, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + (pg + 3) * SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + (pg + 4) * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + (pg + 5) * SCORE_STRIDE, WIDTH * 4)
        pg += 6

    while pg < num_pgs:
        var k_pg = k_base + pg * K_PG_BYTES
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_pg + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf + pg * SCORE_STRIDE, WIDTH * 4)
        pg += 1


# =============================================================================
# V-agg configurations
# =============================================================================


def vagg_simd(scores_f32: F32Ptr, v_base: I8Ptr, v_acc: F32Ptr, num_pgs: Int):
    for pg in range(num_pgs):
        var sc = scores_f32 + pg * HPG * WIDTH
        for qh in range(HPG):
            var acc_base = v_acc + qh * HEAD_DIM
            for pos in range(WIDTH):
                var w = SIMD[DType.float32, SIMD_W](sc[qh * WIDTH + pos])
                var v_row = v_base + (pg * WIDTH + pos) * V_ROW_BYTES
                var d = 0
                while d + SIMD_W <= HEAD_DIM:
                    var v = (v_row + d).load[width=SIMD_W]().cast[DType.float32]()
                    (acc_base + d).store((acc_base + d).load[width=SIMD_W]() + w * v)
                    d += SIMD_W


def vagg_simd_pos_inner(scores_f32: F32Ptr, v_base: I8Ptr, v_acc: F32Ptr, num_pgs: Int):
    for pg in range(num_pgs):
        var sc = scores_f32 + pg * HPG * WIDTH
        for pos in range(WIDTH):
            var v_row = v_base + (pg * WIDTH + pos) * V_ROW_BYTES
            for qh in range(HPG):
                var w = SIMD[DType.float32, SIMD_W](sc[qh * WIDTH + pos])
                var acc_base = v_acc + qh * HEAD_DIM
                var d = 0
                while d + SIMD_W <= HEAD_DIM:
                    var v = (v_row + d).load[width=SIMD_W]().cast[DType.float32]()
                    (acc_base + d).store((acc_base + d).load[width=SIMD_W]() + w * v)
                    d += SIMD_W


def vagg_amx_vnni(
    w_u8_base: U8Ptr, v_vnni_base: I8Ptr, v_acc: F32Ptr, w_dequants: F32Ptr,
    num_pgs: Int,
):
    var vc_arr = AlignedInlineArray[Int32, TILE_M * TILE_N * 2](uninitialized=True)
    var vc_ptr = vc_arr.unsafe_ptr()

    for pg in range(num_pgs):
        var w_ptr = w_u8_base + pg * HPG * K_STEP
        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        var wd_ptr = w_dequants + pg * HPG

        tileload[0, DType.uint8](w_ptr, K_STEP)

        comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
            comptime cg0 = cg_pair * 2
            comptime cg1 = cg0 + 1
            tilezero[4]()
            tilezero[5]()
            tileload[2, DType.int8](v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](vc_ptr, TILE_N * 4)
            tilestore[5, DType.int32](vc_ptr + TILE_M * TILE_N, TILE_N * 4)

            for qh in range(HPG):
                var wd = wd_ptr[qh]
                var i0 = (vc_ptr + qh * TILE_N).load[width=WIDTH]()
                var i1 = (vc_ptr + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]()
                var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                (a0).store((a0).load[width=WIDTH]() + i0.cast[DType.float32]() * wd)
                (a1).store((a1).load[width=WIDTH]() + i1.cast[DType.float32]() * wd)

    _ = vc_arr


def vagg_bf16_gemm(
    scores_f32: F32Ptr, v_i8_base: I8Ptr, v_acc: F32Ptr,
    channel_scales: F32Ptr, num_pgs: Int,
):
    """BF16 GEMM from row-major i8 V. Channel scale deferred to post-multiply.

    1:2:4 tiling. A = attn weights bf16. B = V cast i8→bf16, VNNI packed.
    4 C accumulators (f32). Channel scale applied once per CG on f32 output.
    """
    comptime BF16_K = 32
    comptime BF16_VNNI = 2
    comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]

    var a_arr = AlignedInlineArray[Scalar[DType.bfloat16], TILE_M * BF16_K](
        fill=Scalar[DType.bfloat16](0))
    var a_ptr = a_arr.unsafe_ptr()

    var b0_arr = AlignedInlineArray[UInt32, 16 * TILE_N](fill=UInt32(0))
    var b0_ptr = b0_arr.unsafe_ptr()

    var b1_arr = AlignedInlineArray[UInt32, 16 * TILE_N](fill=UInt32(0))
    var b1_ptr = b1_arr.unsafe_ptr()

    var c_arr = AlignedInlineArray[Float32, TILE_M * TILE_N * 4](uninitialized=True)
    var c_ptr = c_arr.unsafe_ptr()

    for pg in range(num_pgs):
        var sc = scores_f32 + pg * HPG * WIDTH

        for qh in range(HPG):
            for pos in range(WIDTH):
                a_ptr[qh * BF16_K + pos] = sc[qh * WIDTH + pos].cast[DType.bfloat16]()
        tileload[0, DType.bfloat16](a_ptr, BF16_K * 2)

        comptime for batch in range(2):
            comptime cg_base = batch * 4

            for cg_off in range(2):
                var cg = cg_base + cg_off
                var b_dst = b0_ptr if cg_off == 0 else b1_ptr
                for pair in range(WIDTH // 2):
                    var pos0 = pair * 2
                    var pos1 = pair * 2 + 1
                    var row0 = v_i8_base + (pg * WIDTH + pos0) * HEAD_DIM + cg * WIDTH
                    var row1 = v_i8_base + (pg * WIDTH + pos1) * HEAD_DIM + cg * WIDTH
                    var v0 = row0.load[width=WIDTH]().cast[DType.bfloat16]()
                    var v1 = row1.load[width=WIDTH]().cast[DType.bfloat16]()
                    var packed = (v1.cast[DType.uint16]().cast[DType.uint32]() << 16) | v0.cast[DType.uint16]().cast[DType.uint32]()
                    (b_dst + pair * TILE_N).store(packed)

            tileload[2, DType.uint32](b0_ptr, TILE_N * 4)
            tileload[3, DType.uint32](b1_ptr, TILE_N * 4)
            tilezero[4]()
            tilezero[5]()
            tile_dp[4, 0, 2, DType.bfloat16, DType.bfloat16]()
            tile_dp[5, 0, 3, DType.bfloat16, DType.bfloat16]()

            for cg_off in range(2):
                var cg = cg_base + 2 + cg_off
                var b_dst = b0_ptr if cg_off == 0 else b1_ptr
                for pair in range(WIDTH // 2):
                    var pos0 = pair * 2
                    var pos1 = pair * 2 + 1
                    var row0 = v_i8_base + (pg * WIDTH + pos0) * HEAD_DIM + cg * WIDTH
                    var row1 = v_i8_base + (pg * WIDTH + pos1) * HEAD_DIM + cg * WIDTH
                    var v0 = row0.load[width=WIDTH]().cast[DType.bfloat16]()
                    var v1 = row1.load[width=WIDTH]().cast[DType.bfloat16]()
                    var packed = (v1.cast[DType.uint16]().cast[DType.uint32]() << 16) | v0.cast[DType.uint16]().cast[DType.uint32]()
                    (b_dst + pair * TILE_N).store(packed)

            tileload[2, DType.uint32](b0_ptr, TILE_N * 4)
            tileload[3, DType.uint32](b1_ptr, TILE_N * 4)
            tilezero[6]()
            tilezero[7]()
            tile_dp[6, 0, 2, DType.bfloat16, DType.bfloat16]()
            tile_dp[7, 0, 3, DType.bfloat16, DType.bfloat16]()

            tilestore[4, DType.float32](c_ptr, TILE_N * 4)
            tilestore[5, DType.float32](c_ptr + TILE_M * TILE_N, TILE_N * 4)
            tilestore[6, DType.float32](c_ptr + 2 * TILE_M * TILE_N, TILE_N * 4)
            tilestore[7, DType.float32](c_ptr + 3 * TILE_M * TILE_N, TILE_N * 4)

            for ci in range(4):
                var cg = cg_base + ci
                var ch_sc = channel_scales + cg * WIDTH
                var src = c_ptr + ci * TILE_M * TILE_N
                for qh in range(HPG):
                    var acc = v_acc + qh * HEAD_DIM + cg * WIDTH
                    (acc).store((acc).load[width=WIDTH]() + (src + qh * TILE_N).load[width=WIDTH]() * ch_sc.load[width=WIDTH]())

    _ = a_arr
    _ = b0_arr
    _ = b1_arr
    _ = c_arr


# =============================================================================
# Fused score+softmax+vagg
# =============================================================================


def fused_224_amx_v2(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    """Optimized: wd computed once per (page,qh), 2-page V batching, all 8 tiles."""
    comptime W_PAGE_BYTES = TILE_M * K_STEP
    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg + 4 <= num_pgs:
        # Phase 1: QK scoring — 2:2:4 as before
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + 3 * SCORE_STRIDE, WIDTH * 4)

        # Phase 2: Compute W_u8 + wd ONCE per (page, qh).
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                var w_u8 = roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
                (w_pg + qh * K_STEP).store(w_u8)
                wd_arr[bp * HPG + qh] = w_max / Float32(255.0)

        # Phase 3: V-agg in 2-page pairs — 2:2:4 tile layout
        # W already materialized in w_base[0..3]. Just load and go.
        for pair in range(2):
            var bp0 = pair * 2
            var bp1 = pair * 2 + 1
            var v_pg0 = v_vnni_base + (pg + bp0) * V_PG_BYTES_VNNI
            var v_pg1 = v_vnni_base + (pg + bp1) * V_PG_BYTES_VNNI

            tileload[0, DType.uint8](w_base + bp0 * W_PAGE_BYTES, K_STEP)
            tileload[1, DType.uint8](w_base + bp1 * W_PAGE_BYTES, K_STEP)

            comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
                comptime cg0 = cg_pair * 2
                comptime cg1 = cg0 + 1
                tilezero[4]()
                tilezero[5]()
                tilezero[6]()
                tilezero[7]()

                tileload[2, DType.int8](v_pg0 + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[4, 0, 2]()
                tileload[2, DType.int8](v_pg1 + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[5, 1, 2]()

                tileload[3, DType.int8](v_pg0 + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[6, 0, 3]()
                tileload[3, DType.int8](v_pg1 + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[7, 1, 3]()

                # Spill to score_buf (dead after phase 2)
                tilestore[4, DType.int32](score_buf, TILE_N * 4)
                tilestore[5, DType.int32](score_buf + TILE_M * TILE_N, TILE_N * 4)
                tilestore[6, DType.int32](score_buf + 2 * TILE_M * TILE_N, TILE_N * 4)
                tilestore[7, DType.int32](score_buf + 3 * TILE_M * TILE_N, TILE_N * 4)

                for qh in range(HPG):
                    var wd0 = wd_arr[bp0 * HPG + qh]
                    var wd1 = wd_arr[bp1 * HPG + qh]
                    var acc_cg0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                    var acc_cg1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH

                    var r0_cg0 = (score_buf + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    var r1_cg0 = (score_buf + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    (acc_cg0).store((acc_cg0).load[width=WIDTH]() + r0_cg0 * wd0 + r1_cg0 * wd1)

                    var r0_cg1 = (score_buf + 2 * TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    var r1_cg1 = (score_buf + 3 * TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    (acc_cg1).store((acc_cg1).load[width=WIDTH]() + r0_cg1 * wd0 + r1_cg1 * wd1)

        pg += 4

    while pg < num_pgs:
        var w_ptr_tail = w_base
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf, WIDTH * 4)

        for qh in range(HPG):
            var scores_raw = (score_buf + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
            var m = scores_raw.reduce_max()
            var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
            var w_max = exp_scores.reduce_max()
            if w_max < Float32(1e-10):
                (w_ptr_tail + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                wd_arr[qh] = Float32(0)
                continue
            var w_scale = 255.0 / w_max
            (w_ptr_tail + qh * K_STEP).store(roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]())
            wd_arr[qh] = w_max / Float32(255.0)

        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        tileload[0, DType.uint8](w_ptr_tail, K_STEP)
        comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
            comptime cg0 = cg_pair * 2
            comptime cg1 = cg0 + 1
            tilezero[4]()
            tilezero[5]()
            tileload[2, DType.int8](v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](score_buf, TILE_N * 4)
            tilestore[5, DType.int32](score_buf + TILE_M * TILE_N, TILE_N * 4)
            for qh in range(HPG):
                var wd = wd_arr[qh]
                var i0 = (score_buf + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                var i1 = (score_buf + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                (a0).store((a0).load[width=WIDTH]() + i0 * wd)
                (a1).store((a1).load[width=WIDTH]() + i1 * wd)
        pg += 1

    _ = w_arr
    _ = wd_arr


def fused_224_amx_v2_positionwise(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
    v_scales: F32Ptr,
):
    """Same as v2 but with per-position V scale folding (original ButterQuant)."""
    comptime W_PAGE_BYTES = TILE_M * K_STEP
    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg + 4 <= num_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + 3 * SCORE_STRIDE, WIDTH * 4)

        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            var v_sc = v_scales + (pg + bp) * WIDTH
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                var w_max = w_eff.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                var w_u8 = roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
                (w_pg + qh * K_STEP).store(w_u8)
                wd_arr[bp * HPG + qh] = w_max / Float32(255.0)

        for pair in range(2):
            var bp0 = pair * 2
            var bp1 = pair * 2 + 1
            var v_pg0 = v_vnni_base + (pg + bp0) * V_PG_BYTES_VNNI
            var v_pg1 = v_vnni_base + (pg + bp1) * V_PG_BYTES_VNNI

            tileload[0, DType.uint8](w_base + bp0 * W_PAGE_BYTES, K_STEP)
            tileload[1, DType.uint8](w_base + bp1 * W_PAGE_BYTES, K_STEP)

            comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
                comptime cg0 = cg_pair * 2
                comptime cg1 = cg0 + 1
                tilezero[4]()
                tilezero[5]()
                tilezero[6]()
                tilezero[7]()

                tileload[2, DType.int8](v_pg0 + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[4, 0, 2]()
                tileload[2, DType.int8](v_pg1 + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[5, 1, 2]()
                tileload[3, DType.int8](v_pg0 + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[6, 0, 3]()
                tileload[3, DType.int8](v_pg1 + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[7, 1, 3]()

                tilestore[4, DType.int32](score_buf, TILE_N * 4)
                tilestore[5, DType.int32](score_buf + TILE_M * TILE_N, TILE_N * 4)
                tilestore[6, DType.int32](score_buf + 2 * TILE_M * TILE_N, TILE_N * 4)
                tilestore[7, DType.int32](score_buf + 3 * TILE_M * TILE_N, TILE_N * 4)

                for qh in range(HPG):
                    var wd0 = wd_arr[bp0 * HPG + qh]
                    var wd1 = wd_arr[bp1 * HPG + qh]
                    var acc_cg0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                    var acc_cg1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                    var r0_cg0 = (score_buf + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    var r1_cg0 = (score_buf + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    (acc_cg0).store((acc_cg0).load[width=WIDTH]() + r0_cg0 * wd0 + r1_cg0 * wd1)
                    var r0_cg1 = (score_buf + 2 * TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    var r1_cg1 = (score_buf + 3 * TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                    (acc_cg1).store((acc_cg1).load[width=WIDTH]() + r0_cg1 * wd0 + r1_cg1 * wd1)

        pg += 4

    while pg < num_pgs:
        var w_ptr_tail = w_base
        var v_sc = v_scales + pg * WIDTH
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        for qh in range(HPG):
            var scores_raw = (score_buf + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
            var m = scores_raw.reduce_max()
            var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
            var w_eff = exp_scores * v_sc.load[width=WIDTH]()
            var w_max = w_eff.reduce_max()
            if w_max < Float32(1e-10):
                (w_ptr_tail + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                wd_arr[qh] = Float32(0)
                continue
            var w_scale = 255.0 / w_max
            (w_ptr_tail + qh * K_STEP).store(roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[DType.uint8]())
            wd_arr[qh] = w_max / Float32(255.0)
        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        tileload[0, DType.uint8](w_ptr_tail, K_STEP)
        comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
            comptime cg0 = cg_pair * 2
            comptime cg1 = cg0 + 1
            tilezero[4]()
            tilezero[5]()
            tileload[2, DType.int8](v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](score_buf, TILE_N * 4)
            tilestore[5, DType.int32](score_buf + TILE_M * TILE_N, TILE_N * 4)
            for qh in range(HPG):
                var wd = wd_arr[qh]
                var i0 = (score_buf + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                var i1 = (score_buf + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]().cast[DType.float32]()
                var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                (a0).store((a0).load[width=WIDTH]() + i0 * wd)
                (a1).store((a1).load[width=WIDTH]() + i1 * wd)
        pg += 1

    _ = w_arr
    _ = wd_arr


def fused_224_amx(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    var w_arr = AlignedInlineArray[UInt8, TILE_M * K_STEP](fill=UInt8(0))
    var w_ptr = w_arr.unsafe_ptr()
    var vc_arr = AlignedInlineArray[Int32, TILE_M * TILE_N * 2](uninitialized=True)
    var vc_ptr = vc_arr.unsafe_ptr()

    var pg = 0
    while pg + 4 <= num_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + 3 * SCORE_STRIDE, WIDTH * 4)

        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
                var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_ptr + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    continue
                var w_scale = 255.0 / w_max
                var w_u8 = roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
                (w_ptr + qh * K_STEP).store(w_u8)

            var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
            tileload[0, DType.uint8](w_ptr, K_STEP)

            comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
                comptime cg0 = cg_pair * 2
                comptime cg1 = cg0 + 1
                tilezero[4]()
                tilezero[5]()
                tileload[2, DType.int8](v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tileload[3, DType.int8](v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
                tdpbusd[4, 0, 2]()
                tdpbusd[5, 0, 3]()
                tilestore[4, DType.int32](vc_ptr, TILE_N * 4)
                tilestore[5, DType.int32](vc_ptr + TILE_M * TILE_N, TILE_N * 4)

                for qh in range(HPG):
                    var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
                    var exp_sc = exp_f32[WIDTH](scores_raw * Float32(0.001))
                    var wm = exp_sc.reduce_max()
                    var wd = wm / Float32(255.0) if wm > Float32(1e-10) else Float32(0)
                    var i0 = (vc_ptr + qh * TILE_N).load[width=WIDTH]()
                    var i1 = (vc_ptr + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]()
                    var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                    var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                    (a0).store((a0).load[width=WIDTH]() + i0.cast[DType.float32]() * wd)
                    (a1).store((a1).load[width=WIDTH]() + i1.cast[DType.float32]() * wd)
        pg += 4

    while pg < num_pgs:
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf, WIDTH * 4)

        for qh in range(HPG):
            var scores_raw = (score_buf + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
            var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001))
            var w_max = exp_scores.reduce_max()
            if w_max < Float32(1e-10):
                (w_ptr + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                continue
            var w_scale = 255.0 / w_max
            var w_u8 = roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
            (w_ptr + qh * K_STEP).store(w_u8)

        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        tileload[0, DType.uint8](w_ptr, K_STEP)

        comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
            comptime cg0 = cg_pair * 2
            comptime cg1 = cg0 + 1
            tilezero[4]()
            tilezero[5]()
            tileload[2, DType.int8](v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](vc_ptr, TILE_N * 4)
            tilestore[5, DType.int32](vc_ptr + TILE_M * TILE_N, TILE_N * 4)

            for qh in range(HPG):
                var scores_raw = (score_buf + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
                var exp_sc = exp_f32[WIDTH](scores_raw * Float32(0.001))
                var wm = exp_sc.reduce_max()
                var wd = wm / Float32(255.0) if wm > Float32(1e-10) else Float32(0)
                var i0 = (vc_ptr + qh * TILE_N).load[width=WIDTH]()
                var i1 = (vc_ptr + TILE_M * TILE_N + qh * TILE_N).load[width=WIDTH]()
                var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                (a0).store((a0).load[width=WIDTH]() + i0.cast[DType.float32]() * wd)
                (a1).store((a1).load[width=WIDTH]() + i1.cast[DType.float32]() * wd)
        pg += 1

    _ = w_arr
    _ = vc_arr


def fused_224_simd(
    q_ptr: I8Ptr, k_base: U8Ptr, v_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    var pg = 0
    while pg + 4 <= num_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 1) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k_base + (pg + 2) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k_base + (pg + 3) * K_PG_BYTES + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        tilestore[5, DType.int32](score_buf + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_buf + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_buf + 3 * SCORE_STRIDE, WIDTH * 4)

        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            for pos in range(WIDTH):
                var v_row = v_base + ((pg + bp) * WIDTH + pos) * V_ROW_BYTES
                for qh in range(HPG):
                    var raw = Float32(s_base[qh * WIDTH + pos])
                    var w = exp_f32[1](SIMD[DType.float32, 1](raw * 0.001))[0]
                    var wv = SIMD[DType.float32, SIMD_W](w)
                    var acc = v_acc + qh * HEAD_DIM
                    var d = 0
                    while d + SIMD_W <= HEAD_DIM:
                        var v = (v_row + d).load[width=SIMD_W]().cast[DType.float32]()
                        (acc + d).store((acc + d).load[width=SIMD_W]() + wv * v)
                        d += SIMD_W
        pg += 4

    while pg < num_pgs:
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf, WIDTH * 4)
        for pos in range(WIDTH):
            var v_row = v_base + (pg * WIDTH + pos) * V_ROW_BYTES
            for qh in range(HPG):
                var raw = Float32(score_buf[qh * WIDTH + pos])
                var w = exp_f32[1](SIMD[DType.float32, 1](raw * 0.001))[0]
                var wv = SIMD[DType.float32, SIMD_W](w)
                var acc = v_acc + qh * HEAD_DIM
                var d = 0
                while d + SIMD_W <= HEAD_DIM:
                    var v = (v_row + d).load[width=SIMD_W]().cast[DType.float32]()
                    (acc + d).store((acc + d).load[width=SIMD_W]() + wv * v)
                    d += SIMD_W
        pg += 1


def fused_111_simd(
    q_ptr: I8Ptr, k_base: U8Ptr, v_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    for pg in range(num_pgs):
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, HEAD_DIM)
            tileload[2, DType.uint8](k_base + pg * K_PG_BYTES + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_buf, WIDTH * 4)

        for pos in range(WIDTH):
            var v_row = v_base + (pg * WIDTH + pos) * V_ROW_BYTES
            for qh in range(HPG):
                var raw = Float32(score_buf[qh * WIDTH + pos])
                var w = exp_f32[1](SIMD[DType.float32, 1](raw * 0.001))[0]
                var wv = SIMD[DType.float32, SIMD_W](w)
                var acc = v_acc + qh * HEAD_DIM
                var d = 0
                while d + SIMD_W <= HEAD_DIM:
                    var v = (v_row + d).load[width=SIMD_W]().cast[DType.float32]()
                    (acc + d).store((acc + d).load[width=SIMD_W]() + wv * v)
                    d += SIMD_W


# =============================================================================
# Harness
def fused_singularity(
    q_ptr: I8Ptr, k_rowmajor: U8Ptr, v_rowmajor: I8Ptr,
    v_acc: F32Ptr, num_positions: Int,
    k_scale_ptr: F32Ptr, q_bias_ptr: F32Ptr, q_factor_ptr: F32Ptr,
):
    """Single-pass per-position fused score+softmax+V-agg. No tiles. No phases.

    K: row-major u8 [pos, HEAD_DIM]. V: row-major i8 [pos, HEAD_DIM].
    Score via VNNI dot per head. Online softmax. Immediate SIMD V-agg.
    """
    comptime VNNI_STEP = SIMD_W * VNNI_BLK

    var running_max = InlineArray[Float32, HPG](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, HPG](fill=Float32(0))

    for pos in range(num_positions):
        var k_row = k_rowmajor + pos * HEAD_DIM
        var v_row = v_rowmajor + pos * HEAD_DIM
        var k_sc = k_scale_ptr[pos]

        var new_max_any = False
        for qh in range(HPG):
            var acc = SIMD[DType.int32, SIMD_W](0)
            var q_row = q_ptr + qh * HEAD_DIM
            var d = 0
            while d + VNNI_STEP <= HEAD_DIM:
                acc = vpdpbusd[SIMD_W](
                    acc,
                    (k_row + d).bitcast[UInt8]().load[width=SIMD_W * VNNI_BLK](),
                    (q_row + d).load[width=SIMD_W * VNNI_BLK]())
                d += VNNI_STEP
            var raw_score = acc.reduce_add()
            var score = (Float32(raw_score) - q_bias_ptr[qh]) * q_factor_ptr[qh] * k_sc

            var old_max = running_max[qh]
            var new_max = max(old_max, score)
            if new_max > old_max:
                var rescale = exp_f32[1](SIMD[DType.float32, 1](old_max - new_max))[0]
                var acc_base = v_acc + qh * HEAD_DIM
                var dd = 0
                while dd + SIMD_W <= HEAD_DIM:
                    (acc_base + dd).store((acc_base + dd).load[width=SIMD_W]() * rescale)
                    dd += SIMD_W
                running_sum[qh] *= rescale
                running_max[qh] = new_max

            var w = exp_f32[1](SIMD[DType.float32, 1](score - running_max[qh]))[0]
            running_sum[qh] += w
            var wv = SIMD[DType.float32, SIMD_W](w)
            var acc_base = v_acc + qh * HEAD_DIM
            var dd = 0
            while dd + SIMD_W <= HEAD_DIM:
                var v = (v_row + dd).load[width=SIMD_W]().cast[DType.float32]()
                (acc_base + dd).store((acc_base + dd).load[width=SIMD_W]() + wv * v)
                dd += SIMD_W

    for qh in range(HPG):
        var inv_sum = Float32(1.0) / (Float32(127) * running_sum[qh])
        var acc_base = v_acc + qh * HEAD_DIM
        var d = 0
        while d + SIMD_W <= HEAD_DIM:
            (acc_base + d).store((acc_base + d).load[width=SIMD_W]() * inv_sum)
            d += SIMD_W


# =============================================================================
# Harness
# =============================================================================


def run_score_bench(
    name: String, num_pgs: Int,
    func: def(I8Ptr, U8Ptr, I32Ptr, Int) thin -> None,
    q: I8Ptr, k: U8Ptr, sc: I32Ptr,
    warmup: Int = 100, iters: Int = 1000,
):
    for _ in range(warmup):
        func(q, k, sc, num_pgs)
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        func(q, k, sc, num_pgs)
    var elapsed = Int(perf_counter_ns()) - t0
    var ns = elapsed // iters
    print("  " + name + ": " + String(ns) + " ns  (" + String(ns // num_pgs) + " ns/PG)")


def run_vagg_simd_bench(
    name: String, num_pgs: Int,
    func: def(F32Ptr, I8Ptr, F32Ptr, Int) thin -> None,
    sf: F32Ptr, v: I8Ptr, va: F32Ptr,
    warmup: Int = 100, iters: Int = 1000,
):
    for _ in range(warmup):
        func(sf, v, va, num_pgs)
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        func(sf, v, va, num_pgs)
    var elapsed = Int(perf_counter_ns()) - t0
    var ns = elapsed // iters
    print("  " + name + ": " + String(ns) + " ns  (" + String(ns // num_pgs) + " ns/PG)")


def run_vagg_amx_bench(
    name: String, num_pgs: Int,
    wu: U8Ptr, vn: I8Ptr, va: F32Ptr, wd: F32Ptr,
    warmup: Int = 100, iters: Int = 1000,
):
    for _ in range(warmup):
        vagg_amx_vnni(wu, vn, va, wd, num_pgs)
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        vagg_amx_vnni(wu, vn, va, wd, num_pgs)
    var elapsed = Int(perf_counter_ns()) - t0
    var ns = elapsed // iters
    print("  " + name + ": " + String(ns) + " ns  (" + String(ns // num_pgs) + " ns/PG)")


def run_fused_bench(
    name: String, num_pgs: Int,
    func: def(I8Ptr, U8Ptr, I8Ptr, I32Ptr, F32Ptr, Int) thin -> None,
    q: I8Ptr, k: U8Ptr, v: I8Ptr, sc: I32Ptr, va: F32Ptr,
    warmup: Int = 100, iters: Int = 1000,
):
    for _ in range(warmup):
        func(q, k, v, sc, va, num_pgs)
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        func(q, k, v, sc, va, num_pgs)
    var elapsed = Int(perf_counter_ns()) - t0
    var ns = elapsed // iters
    print("  " + name + ": " + String(ns) + " ns  (" + String(ns // num_pgs) + " ns/PG)")


# =============================================================================
# Main
# =============================================================================


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    comptime MAX_PGS = 256
    comptime MAX_CTX = MAX_PGS * WIDTH

    var q = alloc_zeroed[DType.int8](HPG * HEAD_DIM)
    var k = alloc_zeroed[DType.uint8](MAX_PGS * K_PG_BYTES)
    var k_rm = alloc_zeroed[DType.uint8](MAX_CTX * HEAD_DIM)
    var v_rm = alloc_zeroed[DType.int8](MAX_CTX * HEAD_DIM)
    var v_vn = alloc_zeroed[DType.int8](MAX_PGS * V_PG_BYTES_VNNI)
    var sc = alloc_zeroed[DType.int32](MAX_PGS * TILE_M * WIDTH)
    var sf = alloc_zeroed[DType.float32](MAX_PGS * HPG * WIDTH)
    var va = alloc_zeroed[DType.float32](HPG * HEAD_DIM)
    var wu = alloc_zeroed[DType.uint8](MAX_PGS * HPG * K_STEP)
    var wd = alloc_zeroed[DType.float32](MAX_PGS * HPG)
    var ch_sc = alloc_zeroed[DType.float32](HEAD_DIM)

    fill_random_i8(q, HPG * HEAD_DIM)
    fill_random_u8(k.bitcast[UInt8](), MAX_PGS * K_PG_BYTES)
    fill_random_u8(k_rm.bitcast[UInt8](), MAX_CTX * HEAD_DIM)
    fill_random_i8(v_rm, MAX_CTX * HEAD_DIM)
    fill_random_i8(v_vn, MAX_PGS * V_PG_BYTES_VNNI)
    fill_random_u8(wu.bitcast[UInt8](), MAX_PGS * HPG * K_STEP)

    var k_scales = alloc_zeroed[DType.float32](MAX_CTX)
    var q_biases = alloc_zeroed[DType.float32](HPG)
    var q_factors = alloc_zeroed[DType.float32](HPG)
    for i in range(MAX_CTX):
        k_scales[i] = Float32(0.01)
    for h in range(HPG):
        q_biases[h] = Float32(100.0)
        q_factors[h] = Float32(0.00005)
    for i in range(MAX_PGS * HPG * WIDTH):
        sf[i] = Float32(0.01)
    for i in range(MAX_PGS * HPG):
        wd[i] = Float32(0.005)
    for i in range(HEAD_DIM):
        ch_sc[i] = Float32(0.1)

    var k_u8 = k.bitcast[UInt8]()
    var wu_u8 = wu.bitcast[UInt8]()
    var sc_i32 = sc.bitcast[Int32]()

    print("=== AMX Attention Kernel Ablations ===")
    print("HPG=" + String(HPG) + " HEAD_DIM=" + String(HEAD_DIM) + " WIDTH=" + String(WIDTH))
    print("")

    var pg_sizes = InlineArray[Int, 5](fill=0)
    pg_sizes[0] = 4
    pg_sizes[1] = 16
    pg_sizes[2] = 32
    pg_sizes[3] = 64
    pg_sizes[4] = 256

    print("--- SCORING ONLY ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        print(String(npg) + " PGs (" + String(npg * WIDTH) + " positions):")
        run_score_bench("1:1:1", npg, score_1_1_1, q, k_u8, sc_i32)
        run_score_bench("2:2:4", npg, score_2_2_4, q, k_u8, sc_i32)
        run_score_bench("2:1:5", npg, score_2_1_5, q, k_u8, sc_i32)
        run_score_bench("1:1:6", npg, score_1_1_6, q, k_u8, sc_i32)
        print("")

    print("--- V-AGG ONLY ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        print(String(npg) + " PGs:")
        run_vagg_simd_bench("simd_heads_inner", npg, vagg_simd, sf, v_rm, va)
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        run_vagg_simd_bench("simd_pos_inner  ", npg, vagg_simd_pos_inner, sf, v_rm, va)
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        run_vagg_amx_bench("amx_vnni        ", npg, wu_u8, v_vn, va, wd)
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        comptime BF16_WARMUP = 100
        comptime BF16_ITERS = 1000
        for _ in range(BF16_WARMUP):
            vagg_bf16_gemm(sf, v_rm, va, ch_sc, npg)
        var t0 = Int(perf_counter_ns())
        for _ in range(BF16_ITERS):
            vagg_bf16_gemm(sf, v_rm, va, ch_sc, npg)
        var elapsed = Int(perf_counter_ns()) - t0
        var ns = elapsed // BF16_ITERS
        print("  bf16_gemm_1:2:4   : " + String(ns) + " ns  (" + String(ns // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        print("")

    print("--- CORRECTNESS CHECK (16 PGs) ---")
    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_224_amx_v2(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_ch = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_ch += va[i].__abs__()
    print("  v2 channelwise v_acc abs sum: " + String(sum_ch))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_224_amx_v2_positionwise(q, k_u8, v_vn, sc_i32, va, 16, k_scales)
    var sum_pw = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_pw += va[i].__abs__()
    print("  v2 positionwise v_acc abs sum: " + String(sum_pw))
    print("")

    print("--- FUSED SCORE+SOFTMAX+VAGG ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        print(String(npg) + " PGs:")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        run_fused_bench("1:1:1+simd ", npg, fused_111_simd, q, k_u8, v_rm, sc_i32, va)
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        run_fused_bench("2:2:4+simd ", npg, fused_224_simd, q, k_u8, v_rm, sc_i32, va)
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        comptime WARMUP = 100
        comptime ITERS = 1000
        for _ in range(WARMUP):
            fused_224_amx(q, k_u8, v_vn, sc_i32, va, npg)
        var t0 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_224_amx(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed = Int(perf_counter_ns()) - t0
        var ns = elapsed // ITERS
        print("  2:2:4+amx  : " + String(ns) + " ns  (" + String(ns // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        for _ in range(WARMUP):
            fused_224_amx_v2(q, k_u8, v_vn, sc_i32, va, npg)
        var t0_v2 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_224_amx_v2(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v2 = Int(perf_counter_ns()) - t0_v2
        var ns_v2 = elapsed_v2 // ITERS
        print("  2:2:4+amx_v2: " + String(ns_v2) + " ns  (" + String(ns_v2 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        for _ in range(WARMUP):
            fused_224_amx_v2_positionwise(q, k_u8, v_vn, sc_i32, va, npg, k_scales)
        var t0_pw = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_224_amx_v2_positionwise(q, k_u8, v_vn, sc_i32, va, npg, k_scales)
        var elapsed_pw = Int(perf_counter_ns()) - t0_pw
        var ns_pw = elapsed_pw // ITERS
        print("  2:2:4+v2_pos: " + String(ns_pw) + " ns  (" + String(ns_pw // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        print("")

    var k_rm_u8 = k_rm.bitcast[UInt8]()
    print("--- FUSED SINGULARITY (per-position, no tiles for V-agg) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        var npos = npg * WIDTH
        print(String(npg) + " PGs (" + String(npos) + " positions):")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        comptime S_WARMUP = 100
        comptime S_ITERS = 1000
        for _ in range(S_WARMUP):
            fused_singularity(q, k_rm_u8, v_rm, va, npos, k_scales, q_biases, q_factors)
        var t0_s = Int(perf_counter_ns())
        for _ in range(S_ITERS):
            fused_singularity(q, k_rm_u8, v_rm, va, npos, k_scales, q_biases, q_factors)
        var elapsed_s = Int(perf_counter_ns()) - t0_s
        var ns_s = elapsed_s // S_ITERS
        print("  singularity: " + String(ns_s) + " ns  (" + String(ns_s // npg) + " ns/PG, " + String(ns_s // npos) + " ns/pos)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        print("")
