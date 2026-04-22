from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.sys import llvm_intrinsic
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbsud, tdpbusd,
)
from experimental3.kernels.dot_prod import vpdpbusd
from simd_math import exp_f32, exp_f32_fast, roundeven, set_subnormal_zeroing
from notstdcollections import AlignedInlineArray


@always_inline
def prefetch_t0(p: U8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(3), Int32(1))


@always_inline
def prefetch_t1(p: U8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(2), Int32(1))

comptime HPG = 6
comptime HEAD_DIM = 128
comptime WIDTH = 16
comptime NUM_K_STEPS = HEAD_DIM // K_STEP
comptime V_CHANNEL_GROUPS = HEAD_DIM // WIDTH
comptime SIMD_W = simd_width_of[DType.float32]()
comptime K_PG_BYTES = HEAD_DIM // VNNI_BLK * WIDTH * VNNI_BLK
comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
comptime V_PG_BYTES_VNNI = V_CHANNEL_GROUPS * V_CG_BYTES
comptime SCORE_STRIDE = TILE_M * WIDTH
comptime W_PAGE_BYTES = TILE_M * K_STEP
comptime V_ROW_BYTES = HEAD_DIM
comptime SQ_COUNT = WIDTH // VNNI_BLK
comptime SQ_BYTES = WIDTH * VNNI_BLK

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
# V-agg: FMA f32 — no W quantization, no tiles, head-outer
# =============================================================================


def vagg_fma_f32(scores_f32: F32Ptr, v_rm: I8Ptr, v_acc: F32Ptr, num_pgs: Int):
    for pg in range(num_pgs):
        var sc = scores_f32 + pg * HPG * WIDTH
        var v_pg = v_rm + pg * WIDTH * V_ROW_BYTES
        for qh in range(HPG):
            var acc = InlineArray[
                SIMD[DType.float32, WIDTH], V_CHANNEL_GROUPS](uninitialized=True)
            var acc_base = v_acc + qh * HEAD_DIM
            comptime for cg in range(V_CHANNEL_GROUPS):
                acc[cg] = (acc_base + cg * WIDTH).load[width=WIDTH]()
            for pos in range(WIDTH):
                var w = SIMD[DType.float32, WIDTH](sc[qh * WIDTH + pos])
                var v_row = v_pg + pos * V_ROW_BYTES
                comptime for cg in range(V_CHANNEL_GROUPS):
                    acc[cg] += w * (v_row + cg * WIDTH).load[
                        width=WIDTH]().cast[DType.float32]()
            comptime for cg in range(V_CHANNEL_GROUPS):
                (acc_base + cg * WIDTH).store(acc[cg])


# =============================================================================
# V-agg: SIMD VNNI — no tiles, no materialization
# =============================================================================


def vagg_vnni_simd(
    w_ptr: U8Ptr, v_vnni_base: I8Ptr, v_acc: F32Ptr,
    w_dequants: F32Ptr, num_pgs: Int,
):
    var bcast_arr = AlignedInlineArray[UInt32, WIDTH](uninitialized=True)
    var bcast_ptr = bcast_arr.unsafe_ptr()

    for pg in range(num_pgs):
        var w_pg = w_ptr + pg * HPG * K_STEP
        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        var wd_pg = w_dequants + pg * HPG

        for cg in range(V_CHANNEL_GROUPS):
            var v_cg = v_pg + cg * V_CG_BYTES

            for qh in range(HPG):
                var acc = SIMD[DType.int32, WIDTH](0)
                comptime for sq in range(SQ_COUNT):
                    var w4 = (w_pg + qh * K_STEP + sq * VNNI_BLK).bitcast[
                        UInt32]()[]
                    bcast_ptr.store(SIMD[DType.uint32, WIDTH](w4))
                    acc = vpdpbusd[WIDTH](
                        acc,
                        bcast_ptr.bitcast[UInt8]().load[
                            width=WIDTH * VNNI_BLK](),
                        (v_cg + sq * SQ_BYTES).load[
                            width=WIDTH * VNNI_BLK]())

                var wd = wd_pg[qh]
                var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                (dst).store((dst).load[width=WIDTH]()
                    + acc.cast[DType.float32]() * wd)

    _ = bcast_arr


# =============================================================================
# Fused: AMX scoring + common softmax/W-quant + SIMD VNNI V-agg
# =============================================================================


def fused_amx_score_vnni_vagg(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var bcast_arr = AlignedInlineArray[UInt32, WIDTH](uninitialized=True)
    var bcast_ptr = bcast_arr.unsafe_ptr()

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: Softmax + W quantization (identical to v2) ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: SIMD VNNI V-agg (no tiles, no materialization) ──
        for bp in range(4):
            var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
            var w_pg = w_base + bp * W_PAGE_BYTES

            for cg in range(V_CHANNEL_GROUPS):
                var v_cg = v_pg + cg * V_CG_BYTES

                for qh in range(HPG):
                    var acc = SIMD[DType.int32, WIDTH](0)
                    comptime for sq in range(SQ_COUNT):
                        var w4 = (w_pg + qh * K_STEP + sq * VNNI_BLK).bitcast[
                            UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w4))
                        acc = vpdpbusd[WIDTH](
                            acc,
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            (v_cg + sq * SQ_BYTES).load[
                                width=WIDTH * VNNI_BLK]())

                    var wd = wd_arr[bp * HPG + qh]
                    var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                    (dst).store((dst).load[width=WIDTH]()
                        + acc.cast[DType.float32]() * wd)

        pg += 4

    _ = w_arr
    _ = wd_arr
    _ = bcast_arr


# =============================================================================
# Fused: AMX scoring + exp → immediate FMA V-agg (no W quant at all)
# =============================================================================


def fused_amx_score_fma_vagg(
    q_ptr: I8Ptr, k_base: U8Ptr, v_rm: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: exp → immediate FMA V-agg (no W buffer, no quant) ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var v_pg = v_rm + (pg + bp) * WIDTH * V_ROW_BYTES

            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[
                    width=WIDTH]().cast[DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))

                var acc = InlineArray[
                    SIMD[DType.float32, WIDTH], V_CHANNEL_GROUPS](
                    uninitialized=True)
                var acc_base = v_acc + qh * HEAD_DIM
                comptime for cg in range(V_CHANNEL_GROUPS):
                    acc[cg] = (acc_base + cg * WIDTH).load[width=WIDTH]()

                for pos in range(WIDTH):
                    var w = SIMD[DType.float32, WIDTH](exp_scores[pos])
                    var v_row = v_pg + pos * V_ROW_BYTES
                    comptime for cg in range(V_CHANNEL_GROUPS):
                        acc[cg] += w * (v_row + cg * WIDTH).load[
                            width=WIDTH]().cast[DType.float32]()

                comptime for cg in range(V_CHANNEL_GROUPS):
                    (acc_base + cg * WIDTH).store(acc[cg])

        pg += 4


# =============================================================================
# Pipelined: AMX scoring batch N+1 || SIMD softmax+VNNI-vagg batch N
# =============================================================================


def fused_pipelined_vnni(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0 and num_pgs >= 8,
        "need >= 8 PGs, multiple of 4")

    var sc_a = AlignedInlineArray[Int32, 4 * SCORE_STRIDE](uninitialized=True)
    var sc_b = AlignedInlineArray[Int32, 4 * SCORE_STRIDE](uninitialized=True)
    var w_a = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_b = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var wd_a = InlineArray[Float32, 4 * HPG](fill=Float32(0))
    var wd_b = InlineArray[Float32, 4 * HPG](fill=Float32(0))
    var bcast_arr = AlignedInlineArray[UInt32, WIDTH](uninitialized=True)
    var bcast_ptr = bcast_arr.unsafe_ptr()

    var scp = InlineArray[I32Ptr, 2](fill=I32Ptr())
    scp[0] = sc_a.unsafe_ptr()
    scp[1] = sc_b.unsafe_ptr()
    var wp = InlineArray[U8Ptr, 2](fill=U8Ptr())
    wp[0] = w_a.unsafe_ptr()
    wp[1] = w_b.unsafe_ptr()
    var wdp = InlineArray[F32Ptr, 2](fill=F32Ptr())
    wdp[0] = F32Ptr(unsafe_from_address=Int(UnsafePointer(to=wd_a[0])))
    wdp[1] = F32Ptr(unsafe_from_address=Int(UnsafePointer(to=wd_b[0])))

    @always_inline
    def amx_score_batch(pg: Int, sc: I32Ptr) unified {read q_ptr, read k_base}:
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
        tilestore[4, DType.int32](sc, WIDTH * 4)
        tilestore[5, DType.int32](sc + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](sc + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](sc + 3 * SCORE_STRIDE, WIDTH * 4)

    @always_inline
    def simd_softmax_wquant(sc: I32Ptr, w: U8Ptr, wdq: F32Ptr):
        for bp in range(4):
            var s_base = sc + bp * SCORE_STRIDE
            var w_pg = w + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[
                    width=WIDTH]().cast[DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wdq[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wdq[bp * HPG + qh] = w_max / 255.0

    @always_inline
    def simd_vnni_vagg_batch(
        w: U8Ptr, wdq: F32Ptr, pg: Int,
    ) unified {read v_vnni_base, read bcast_ptr, read v_acc}:
        for bp in range(4):
            var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
            var w_pg = w + bp * W_PAGE_BYTES
            for cg in range(V_CHANNEL_GROUPS):
                var v_cg = v_pg + cg * V_CG_BYTES
                for qh in range(HPG):
                    var acc = SIMD[DType.int32, WIDTH](0)
                    comptime for sq in range(SQ_COUNT):
                        var w4 = (w_pg + qh * K_STEP + sq * VNNI_BLK).bitcast[
                            UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w4))
                        acc = vpdpbusd[WIDTH](
                            acc,
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            (v_cg + sq * SQ_BYTES).load[
                                width=WIDTH * VNNI_BLK]())
                    var wd = wdq[bp * HPG + qh]
                    var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                    (dst).store((dst).load[width=WIDTH]()
                        + acc.cast[DType.float32]() * wd)

    # ── Prologue: score first batch ──
    amx_score_batch(0, scp[0])

    # ── Pipelined main loop ──
    var cur = 0
    var pg = 4
    while pg < num_pgs:
        var nxt = 1 - cur
        # AMX scores next batch (AMX unit) — OOO can overlap with SIMD below
        amx_score_batch(pg, scp[nxt])
        # SIMD processes current batch (vector ports)
        simd_softmax_wquant(scp[cur], wp[cur], wdp[cur])
        simd_vnni_vagg_batch(wp[cur], wdp[cur], pg - 4)
        cur = nxt
        pg += 4

    # ── Epilogue: process last batch ──
    simd_softmax_wquant(scp[cur], wp[cur], wdp[cur])
    simd_vnni_vagg_batch(wp[cur], wdp[cur], pg - 4)

    _ = sc_a
    _ = sc_b
    _ = w_a
    _ = w_b
    _ = wd_a
    _ = wd_b
    _ = bcast_arr


# =============================================================================
# OPT A: V-agg SIMD VNNI — hoist V loads above qh, register-only broadcast
# =============================================================================
# Hypotheses:
#   H1: V data per (cg,sq) is reused HPG=6 times. Loading once into 4 ZMM
#       regs cuts V loads from 192 to 32 per PG.
#   H2: Use 4-byte load + repeated join to splat directly in registers
#       (LLVM folds to vpbroadcastd or the {1to16} EVEX modifier on
#       vpdpbusd), eliminating the scratch store/load roundtrip.


@always_inline
def broadcast_4u8(p: U8Ptr) -> SIMD[DType.uint8, WIDTH * VNNI_BLK]:
    var b4 = p.load[width=VNNI_BLK]()
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    var b32 = b16.join(b16)
    return b32.join(b32)


def vagg_vnni_simd_v2(
    w_ptr: U8Ptr, v_vnni_base: I8Ptr, v_acc: F32Ptr,
    w_dequants: F32Ptr, num_pgs: Int,
):
    for pg in range(num_pgs):
        var w_pg = w_ptr + pg * HPG * K_STEP
        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        var wd_pg = w_dequants + pg * HPG

        for cg in range(V_CHANNEL_GROUPS):
            var v_cg = v_pg + cg * V_CG_BYTES
            var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()

            for qh in range(HPG):
                var w_row = w_pg + qh * K_STEP
                var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                var acc = SIMD[DType.int32, WIDTH](0)
                acc = vpdpbusd[WIDTH](acc, w0, v0)
                acc = vpdpbusd[WIDTH](acc, w1, v1)
                acc = vpdpbusd[WIDTH](acc, w2_, v2)
                acc = vpdpbusd[WIDTH](acc, w3, v3)

                var wd = wd_pg[qh]
                var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                (dst).store((dst).load[width=WIDTH]()
                    + acc.cast[DType.float32]() * wd)


# =============================================================================
# OPT B: V-agg SIMD VNNI — 4 independent accumulators per (cg,qh)
# =============================================================================
# Hypothesis:
#   H1+H2 above PLUS break the vpdpbusd dependency chain. With 4-deep accs,
#   the 5-cycle vpdpbusd latency is hidden — only the final add + cast pays
#   it. Pairs nicely with HPG=6 in-flight (24 independent ops vs 6 chains).


def vagg_vnni_simd_v2_4acc(
    w_ptr: U8Ptr, v_vnni_base: I8Ptr, v_acc: F32Ptr,
    w_dequants: F32Ptr, num_pgs: Int,
):
    for pg in range(num_pgs):
        var w_pg = w_ptr + pg * HPG * K_STEP
        var v_pg = v_vnni_base + pg * V_PG_BYTES_VNNI
        var wd_pg = w_dequants + pg * HPG

        for cg in range(V_CHANNEL_GROUPS):
            var v_cg = v_pg + cg * V_CG_BYTES
            var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
            var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()

            for qh in range(HPG):
                var w_row = w_pg + qh * K_STEP
                var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                var a0 = vpdpbusd[WIDTH](
                    SIMD[DType.int32, WIDTH](0), w0, v0)
                var a1 = vpdpbusd[WIDTH](
                    SIMD[DType.int32, WIDTH](0), w1, v1)
                var a2 = vpdpbusd[WIDTH](
                    SIMD[DType.int32, WIDTH](0), w2_, v2)
                var a3 = vpdpbusd[WIDTH](
                    SIMD[DType.int32, WIDTH](0), w3, v3)

                var wd = wd_pg[qh]
                var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                (dst).store((dst).load[width=WIDTH]()
                    + ((a0 + a1) + (a2 + a3)).cast[DType.float32]() * wd)


# =============================================================================
# OPT C: Fused AMX SCORE + Optimized SIMD VNNI V-agg (uses OPT B body)
# =============================================================================


def fused_amx_score_vnni_vagg_v2(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: Softmax + W quantization ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: SIMD VNNI V-agg (hoist V, direct broadcast, 4 acc) ──
        for bp in range(4):
            var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
            var w_pg = w_base + bp * W_PAGE_BYTES

            for cg in range(V_CHANNEL_GROUPS):
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()

                for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    var wd = wd_arr[bp * HPG + qh]
                    var dst = v_acc + qh * HEAD_DIM + cg * WIDTH
                    (dst).store((dst).load[width=WIDTH]()
                        + ((a0 + a1) + (a2 + a3)).cast[DType.float32]() * wd)

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# OPT D: Fused — also hoist v_acc across the 4 batch PGs, restructure phase 3
# =============================================================================
# Hypothesis:
#   H3+H4: Within a 4-PG batch, the v_acc[qh][cg] location is loaded+stored
#          4 times (once per bp). Restructuring as `for cg: load 6 accs;
#          for bp: V-agg into accs; store 6 accs` cuts those 48 acc loads
#          and 48 acc stores down to 8+8 per batch — saves 80 mem ops per
#          (cg, batch) and breaks the load-store dependency chain that
#          serialises bp iterations through L1.


def fused_amx_score_vnni_vagg_v3(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: Softmax + W quantization ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: V-agg with cg outer, bp inner, acc hoisted ──
        for cg in range(V_CHANNEL_GROUPS):
            # Load HPG=6 acc registers for this cg, accumulate across all 4 bp.
            var acc_qh = InlineArray[
                SIMD[DType.float32, WIDTH], HPG](uninitialized=True)
            comptime for qh in range(HPG):
                acc_qh[qh] = (v_acc + qh * HEAD_DIM + cg * WIDTH).load[
                    width=WIDTH]()

            for bp in range(4):
                var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    acc_qh[qh] = acc_qh[qh] + (
                        (a0 + a1) + (a2 + a3)).cast[DType.float32]() * wd_arr[
                            bp * HPG + qh]

            comptime for qh in range(HPG):
                (v_acc + qh * HEAD_DIM + cg * WIDTH).store(acc_qh[qh])

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# OPT v4: Same acc-hoist as v3 but inner qh loop unrolled comptime
# =============================================================================
# v3 regressed because InlineArray[acc_qh] was indexed dynamically by qh
# inside the bp loop, which forced spilling. v4 unrolls qh comptime so each
# acc_qh[qh] resolves to a static index; LLVM should promote to registers.


def fused_amx_score_vnni_vagg_v4(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: Softmax + W quantization ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: V-agg with cg outer, comptime-unrolled qh per bp ──
        for cg in range(V_CHANNEL_GROUPS):
            var acc0 = (v_acc + 0 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc1 = (v_acc + 1 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc2 = (v_acc + 2 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc3 = (v_acc + 3 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc4 = (v_acc + 4 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc5 = (v_acc + 5 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()

            for bp in range(4):
                var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                comptime for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    var summed = ((a0 + a1) + (a2 + a3)).cast[
                        DType.float32]() * wd_arr[bp * HPG + qh]
                    comptime if qh == 0:
                        acc0 = acc0 + summed
                    elif qh == 1:
                        acc1 = acc1 + summed
                    elif qh == 2:
                        acc2 = acc2 + summed
                    elif qh == 3:
                        acc3 = acc3 + summed
                    elif qh == 4:
                        acc4 = acc4 + summed
                    else:
                        acc5 = acc5 + summed

            (v_acc + 0 * HEAD_DIM + cg * WIDTH).store(acc0)
            (v_acc + 1 * HEAD_DIM + cg * WIDTH).store(acc1)
            (v_acc + 2 * HEAD_DIM + cg * WIDTH).store(acc2)
            (v_acc + 3 * HEAD_DIM + cg * WIDTH).store(acc3)
            (v_acc + 4 * HEAD_DIM + cg * WIDTH).store(acc4)
            (v_acc + 5 * HEAD_DIM + cg * WIDTH).store(acc5)

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# OPT v5: v4 + prefetch next 4-PG K/V batch into L1 during phase 2/3
# =============================================================================
# Hypothesis:
#   At >=64 PGs the K/V working set spills L1 (each batch reads 16 KB,
#   accumulating 256 PG = 1 MB). Phase 2/3 take ~400 ns; the next batch's
#   K (8 KB / 128 lines) and V (8 KB / 128 lines) need to be in L1 before
#   the next iteration. Issue prefetches early in each iteration.


def fused_amx_score_vnni_vagg_v5(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # Prefetch next 4-PG batch's K and V data while we softmax/V-agg.
        var nxt = pg + 4
        if nxt < num_pgs:
            var k_nxt = k_base + nxt * K_PG_BYTES
            var v_nxt = v_vnni_base + nxt * V_PG_BYTES_VNNI
            # K: 4 PGs * 2 KB = 8 KB = 128 lines
            comptime for ln in range(0, 4 * K_PG_BYTES, 64):
                prefetch_t0(k_nxt + ln)
            comptime for ln in range(0, 4 * V_PG_BYTES_VNNI, 64):
                prefetch_t0((v_nxt + ln).bitcast[UInt8]())

        # ── Phase 2: Softmax + W quantization ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: V-agg (v4 layout: cg outer, comptime qh, hoisted acc) ──
        for cg in range(V_CHANNEL_GROUPS):
            var acc0 = (v_acc + 0 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc1 = (v_acc + 1 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc2 = (v_acc + 2 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc3 = (v_acc + 3 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc4 = (v_acc + 4 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc5 = (v_acc + 5 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()

            for bp in range(4):
                var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                comptime for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    var summed = ((a0 + a1) + (a2 + a3)).cast[
                        DType.float32]() * wd_arr[bp * HPG + qh]
                    comptime if qh == 0:
                        acc0 = acc0 + summed
                    elif qh == 1:
                        acc1 = acc1 + summed
                    elif qh == 2:
                        acc2 = acc2 + summed
                    elif qh == 3:
                        acc3 = acc3 + summed
                    elif qh == 4:
                        acc4 = acc4 + summed
                    else:
                        acc5 = acc5 + summed

            (v_acc + 0 * HEAD_DIM + cg * WIDTH).store(acc0)
            (v_acc + 1 * HEAD_DIM + cg * WIDTH).store(acc1)
            (v_acc + 2 * HEAD_DIM + cg * WIDTH).store(acc2)
            (v_acc + 3 * HEAD_DIM + cg * WIDTH).store(acc3)
            (v_acc + 4 * HEAD_DIM + cg * WIDTH).store(acc4)
            (v_acc + 5 * HEAD_DIM + cg * WIDTH).store(acc5)

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# OPT v6: v4 + exp_f32_fast (8 SIMD ops vs 23, ~0.3% rel err — below 1/255)
# =============================================================================
# Hypothesis:
#   Softmax/W-quant runs exp_f32 6x per PG. exp_f32 is ~23 SIMD ops, exp_f32_fast
#   is ~8. The W-quant target is uint8 (1/255 ~= 0.4%) so 0.3% exp error is
#   safely below quant noise. Should shave ~50-100 ns/PG off softmax phase.


def fused_amx_score_vnni_vagg_v6(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
        # ── Phase 1: Score 4 PGs (AMX 2:2:4) ──
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

        # ── Phase 2: Softmax + W quantization (using exp_f32_fast) ──
        for bp in range(4):
            var s_base = score_buf + bp * SCORE_STRIDE
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32_fast[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: V-agg (v4 layout: cg outer, comptime qh, hoisted acc) ──
        for cg in range(V_CHANNEL_GROUPS):
            var acc0 = (v_acc + 0 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc1 = (v_acc + 1 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc2 = (v_acc + 2 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc3 = (v_acc + 3 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc4 = (v_acc + 4 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()
            var acc5 = (v_acc + 5 * HEAD_DIM + cg * WIDTH).load[width=WIDTH]()

            for bp in range(4):
                var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                comptime for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    var summed = ((a0 + a1) + (a2 + a3)).cast[
                        DType.float32]() * wd_arr[bp * HPG + qh]
                    comptime if qh == 0:
                        acc0 = acc0 + summed
                    elif qh == 1:
                        acc1 = acc1 + summed
                    elif qh == 2:
                        acc2 = acc2 + summed
                    elif qh == 3:
                        acc3 = acc3 + summed
                    elif qh == 4:
                        acc4 = acc4 + summed
                    else:
                        acc5 = acc5 + summed

            (v_acc + 0 * HEAD_DIM + cg * WIDTH).store(acc0)
            (v_acc + 1 * HEAD_DIM + cg * WIDTH).store(acc1)
            (v_acc + 2 * HEAD_DIM + cg * WIDTH).store(acc2)
            (v_acc + 3 * HEAD_DIM + cg * WIDTH).store(acc3)
            (v_acc + 4 * HEAD_DIM + cg * WIDTH).store(acc4)
            (v_acc + 5 * HEAD_DIM + cg * WIDTH).store(acc5)

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# OPT v7: v6 with InlineArray-acc + comptime for qh (parametric in HPG)
# =============================================================================
# Same idea as v4/v6 but uses InlineArray[acc, HPG] indexed only at comptime
# qh values. If LLVM promotes the array to registers (as it does for the
# vagg_fma_f32 pattern above), this is identical-perf to v6 but works for
# any HPG without a hand-unrolled if/elif chain — needed for the minimax
# port which keeps HPG as a comptime parameter.


def fused_amx_score_vnni_vagg_v7(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
    debug_assert(num_pgs % 4 == 0, "num_pgs must be a multiple of 4")

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var pg = 0
    while pg < num_pgs:
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
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32_fast[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        for cg in range(V_CHANNEL_GROUPS):
            var acc = InlineArray[
                SIMD[DType.float32, WIDTH], HPG](uninitialized=True)
            comptime for qh in range(HPG):
                acc[qh] = (v_acc + qh * HEAD_DIM + cg * WIDTH).load[
                    width=WIDTH]()

            for bp in range(4):
                var v_pg = v_vnni_base + (pg + bp) * V_PG_BYTES_VNNI
                var v_cg = v_pg + cg * V_CG_BYTES
                var v0 = (v_cg + 0 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v1 = (v_cg + 1 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v2 = (v_cg + 2 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var v3 = (v_cg + 3 * SQ_BYTES).load[width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                comptime for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var w0 = broadcast_4u8(w_row + 0 * VNNI_BLK)
                    var w1 = broadcast_4u8(w_row + 1 * VNNI_BLK)
                    var w2_ = broadcast_4u8(w_row + 2 * VNNI_BLK)
                    var w3 = broadcast_4u8(w_row + 3 * VNNI_BLK)

                    var a0 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w0, v0)
                    var a1 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w1, v1)
                    var a2 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w2_, v2)
                    var a3 = vpdpbusd[WIDTH](
                        SIMD[DType.int32, WIDTH](0), w3, v3)

                    acc[qh] = acc[qh] + ((a0 + a1) + (a2 + a3)).cast[
                        DType.float32]() * wd_arr[bp * HPG + qh]

            comptime for qh in range(HPG):
                (v_acc + qh * HEAD_DIM + cg * WIDTH).store(acc[qh])

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# Baseline: v2 AMX V-agg (for comparison)
# =============================================================================


def fused_v2_baseline(
    q_ptr: I8Ptr, k_base: U8Ptr, v_vnni_base: I8Ptr,
    score_buf: I32Ptr, v_acc: F32Ptr, num_pgs: Int,
):
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
            for qh in range(HPG):
                var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var m = scores_raw.reduce_max()
                var exp_scores = exp_f32[WIDTH](
                    scores_raw * Float32(0.001) - m * Float32(0.001))
                var w_max = exp_scores.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

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
                tilestore[5, DType.int32](
                    score_buf + TILE_M * TILE_N, TILE_N * 4)
                tilestore[6, DType.int32](
                    score_buf + 2 * TILE_M * TILE_N, TILE_N * 4)
                tilestore[7, DType.int32](
                    score_buf + 3 * TILE_M * TILE_N, TILE_N * 4)

                for qh in range(HPG):
                    var wd0 = wd_arr[bp0 * HPG + qh]
                    var wd1 = wd_arr[bp1 * HPG + qh]
                    var a0 = v_acc + qh * HEAD_DIM + cg0 * WIDTH
                    var a1 = v_acc + qh * HEAD_DIM + cg1 * WIDTH
                    var r0 = (score_buf + qh * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]()
                    var r1 = (score_buf + TILE_M * TILE_N + qh * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]()
                    (a0).store((a0).load[width=WIDTH]() + r0 * wd0 + r1 * wd1)
                    var r2 = (score_buf + 2 * TILE_M * TILE_N + qh * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]()
                    var r3 = (score_buf + 3 * TILE_M * TILE_N + qh * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]()
                    (a1).store((a1).load[width=WIDTH]() + r2 * wd0 + r3 * wd1)

        pg += 4

    _ = w_arr
    _ = wd_arr


# =============================================================================
# Harness
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
    var v_vn = alloc_zeroed[DType.int8](MAX_PGS * V_PG_BYTES_VNNI)
    var v_rm = alloc_zeroed[DType.int8](MAX_CTX * HEAD_DIM)
    var sc = alloc_zeroed[DType.int32](MAX_PGS * TILE_M * WIDTH)
    var sf = alloc_zeroed[DType.float32](MAX_PGS * HPG * WIDTH)
    var va = alloc_zeroed[DType.float32](HPG * HEAD_DIM)
    var wu = alloc_zeroed[DType.uint8](MAX_PGS * HPG * K_STEP)
    var wd = alloc_zeroed[DType.float32](MAX_PGS * HPG)

    fill_random_i8(q, HPG * HEAD_DIM)
    fill_random_u8(k.bitcast[UInt8](), MAX_PGS * K_PG_BYTES)
    fill_random_i8(v_vn, MAX_PGS * V_PG_BYTES_VNNI)
    fill_random_i8(v_rm, MAX_CTX * HEAD_DIM)
    fill_random_u8(wu.bitcast[UInt8](), MAX_PGS * HPG * K_STEP)
    for i in range(MAX_PGS * HPG * WIDTH):
        sf[i] = Float32(0.01)
    for i in range(MAX_PGS * HPG):
        wd[i] = Float32(0.005)

    var k_u8 = k.bitcast[UInt8]()
    var wu_u8 = wu.bitcast[UInt8]()
    var sc_i32 = sc.bitcast[Int32]()

    print("=== SIMD VNNI V-Agg vs AMX V-Agg ===")
    print("HPG=" + String(HPG) + " HEAD_DIM=" + String(HEAD_DIM) +
        " WIDTH=" + String(WIDTH))
    print("")

    var pg_sizes = InlineArray[Int, 5](fill=0)
    pg_sizes[0] = 4
    pg_sizes[1] = 16
    pg_sizes[2] = 32
    pg_sizes[3] = 64
    pg_sizes[4] = 256

    comptime WARMUP = 200
    comptime ITERS = 2000

    print("--- V-AGG ONLY: AMX VNNI vs SIMD VNNI ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        print(String(npg) + " PGs:")

        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            vagg_vnni_simd(wu_u8, v_vn, va, wd, npg)
        var t_simd = Int(perf_counter_ns())
        for _ in range(ITERS):
            vagg_vnni_simd(wu_u8, v_vn, va, wd, npg)
        var e_simd = Int(perf_counter_ns()) - t_simd
        var ns_simd = e_simd // ITERS
        print("  simd_vnni:        " + String(ns_simd) + " ns  (" +
            String(ns_simd // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        for _ in range(WARMUP):
            vagg_vnni_simd_v2(wu_u8, v_vn, va, wd, npg)
        var t_v2 = Int(perf_counter_ns())
        for _ in range(ITERS):
            vagg_vnni_simd_v2(wu_u8, v_vn, va, wd, npg)
        var e_v2 = Int(perf_counter_ns()) - t_v2
        var ns_v2 = e_v2 // ITERS
        print("  simd_vnni_v2:     " + String(ns_v2) + " ns  (" +
            String(ns_v2 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)

        for _ in range(WARMUP):
            vagg_vnni_simd_v2_4acc(wu_u8, v_vn, va, wd, npg)
        var t_v24 = Int(perf_counter_ns())
        for _ in range(ITERS):
            vagg_vnni_simd_v2_4acc(wu_u8, v_vn, va, wd, npg)
        var e_v24 = Int(perf_counter_ns()) - t_v24
        var ns_v24 = e_v24 // ITERS
        print("  simd_vnni_v2_4acc:" + String(ns_v24) + " ns  (" +
            String(ns_v24 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        print("")

    print("")
    print("--- CORRECTNESS CHECK (16 PGs) ---")
    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_v2_baseline(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_v2 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_v2 += va[i].__abs__()
    print("  v2_amx abs sum: " + String(sum_v2))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv += va[i].__abs__()
    print("  amx_score+vnni_vagg abs sum: " + String(sum_sv))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v2(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv2 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv2 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v2 abs sum: " + String(sum_sv2))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v3(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv3 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv3 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v3 abs sum: " + String(sum_sv3))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v4(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv4 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv4 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v4 abs sum: " + String(sum_sv4))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v5(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv5 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv5 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v5 abs sum: " + String(sum_sv5))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v6(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv6 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv6 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v6 abs sum: " + String(sum_sv6))

    for i in range(HPG * HEAD_DIM):
        va[i] = Float32(0)
    fused_amx_score_vnni_vagg_v7(q, k_u8, v_vn, sc_i32, va, 16)
    var sum_sv7 = Float32(0)
    for i in range(HPG * HEAD_DIM):
        sum_sv7 += va[i].__abs__()
    print("  amx_score+vnni_vagg_v7 abs sum: " + String(sum_sv7))
    print("")

    print("--- FUSED: PIPELINED AMX SCORE || SIMD VNNI V-AGG ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        if npg < 8:
            print("  " + String(npg) + " PGs: skipped (need >= 8)")
            continue
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_pipelined_vnni(q, k_u8, v_vn, va, npg)
        var tp = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_pipelined_vnni(q, k_u8, v_vn, va, npg)
        var ep = Int(perf_counter_ns()) - tp
        var nsp = ep // ITERS
        print("  " + String(npg) + " PGs: " + String(nsp) + " ns  (" +
            String(nsp // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: V2 AMX BASELINE ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_v2_baseline(q, k_u8, v_vn, sc_i32, va, npg)
        var t0 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_v2_baseline(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed = Int(perf_counter_ns()) - t0
        var ns = elapsed // ITERS
        print("  " + String(npg) + " PGs: " + String(ns) + " ns  (" +
            String(ns // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg(q, k_u8, v_vn, sc_i32, va, npg)
        var t0 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed = Int(perf_counter_ns()) - t0
        var ns = elapsed // ITERS
        print("  " + String(npg) + " PGs: " + String(ns) + " ns  (" +
            String(ns // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v2) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v2(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v2(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v = Int(perf_counter_ns()) - t0v
        var ns_v = elapsed_v // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v) + " ns  (" +
            String(ns_v // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v3 acc-hoist) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v3(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v3 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v3(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v3 = Int(perf_counter_ns()) - t0v3
        var ns_v3 = elapsed_v3 // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v3) + " ns  (" +
            String(ns_v3 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v4 unrolled qh) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v4(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v4 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v4(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v4 = Int(perf_counter_ns()) - t0v4
        var ns_v4 = elapsed_v4 // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v4) + " ns  (" +
            String(ns_v4 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v5 +prefetch) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v5(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v5 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v5(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v5 = Int(perf_counter_ns()) - t0v5
        var ns_v5 = elapsed_v5 // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v5) + " ns  (" +
            String(ns_v5 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v6 fast exp) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v6(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v6 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v6(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v6 = Int(perf_counter_ns()) - t0v6
        var ns_v6 = elapsed_v6 // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v6) + " ns  (" +
            String(ns_v6 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + SIMD VNNI V-AGG (OPT v7 InlineArr-acc) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_vnni_vagg_v7(q, k_u8, v_vn, sc_i32, va, npg)
        var t0v7 = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_vnni_vagg_v7(q, k_u8, v_vn, sc_i32, va, npg)
        var elapsed_v7 = Int(perf_counter_ns()) - t0v7
        var ns_v7 = elapsed_v7 // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_v7) + " ns  (" +
            String(ns_v7 // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- V-AGG ONLY: FMA f32 (no W quant, row-major V) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            vagg_fma_f32(sf, v_rm, va, npg)
        var tf = Int(perf_counter_ns())
        for _ in range(ITERS):
            vagg_fma_f32(sf, v_rm, va, npg)
        var ef = Int(perf_counter_ns()) - tf
        var nsf = ef // ITERS
        print("  " + String(npg) + " PGs: " + String(nsf) + " ns  (" +
            String(nsf // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")

    print("--- FUSED: AMX SCORE + FMA V-AGG (no W quant, row-major V) ---")
    for idx in range(5):
        var npg = pg_sizes[idx]
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
        for _ in range(WARMUP):
            fused_amx_score_fma_vagg(q, k_u8, v_rm, sc_i32, va, npg)
        var t0f = Int(perf_counter_ns())
        for _ in range(ITERS):
            fused_amx_score_fma_vagg(q, k_u8, v_rm, sc_i32, va, npg)
        var elapsed_f = Int(perf_counter_ns()) - t0f
        var ns_f = elapsed_f // ITERS
        print("  " + String(npg) + " PGs: " + String(ns_f) + " ns  (" +
            String(ns_f // npg) + " ns/PG)")
        for i in range(HPG * HEAD_DIM):
            va[i] = Float32(0)
    print("")
