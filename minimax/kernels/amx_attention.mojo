from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.amx import (
    K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_decode_config, make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbsud,
)
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.dispatch_args import ChunkedAttnArgs
from experimental3.kernels.dot_prod import vpdpbusd, bcast_4u8_vnni
from notstdcollections import AlignedInlineArray
from simd_math import exp_f32, exp_f32_fast, roundeven


@fieldwise_init
struct AmxConfigArgs(Copyable, ImplicitlyCopyable):
    var dummy: Int

    def __init__(out self):
        self.dummy = 0


def amx_config_kernel[heads_per_group: Int](args: AmxConfigArgs):
    _ = init_intel_amx()
    var cfg = make_224_decode_config[heads_per_group]()
    ldtilecfg(UnsafePointer(to=cfg))


def amx_prefill_config_kernel(args: AmxConfigArgs):
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))


def amx_chunked_attn_kernel[
    head_dim: Int, local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int, max_attn_chunks: Int,
](args: ChunkedAttnArgs):
    var num_pgs = args.end_pg - args.start_pg
    debug_assert(num_pgs % 4 == 0,
        "chunk PG count must be a multiple of 4 — pad at dispatch")

    var cache = Gemma4KVCache[local_max_seq, head_dim, num_kv_heads, num_q_heads](
        Int(args.cache_base))
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)
    var qi_biases = args.qi_biases_base
    var q_scales = args.q_scales_base

    comptime HPG = heads_per_group
    comptime WIDTH = CACHE_WIDTH
    comptime V_CHANNEL_GROUPS = head_dim // WIDTH
    comptime NUM_K_STEPS = head_dim // K_STEP
    comptime Q_DENOM = Float32(127) * Float32(127)
    comptime HEAD_STRIDE = 2 + head_dim
    comptime K_PG_BYTES = head_dim // VNNI_BLK * WIDTH * VNNI_BLK
    comptime SQ_COUNT = WIDTH // VNNI_BLK
    comptime SQ_BYTES = WIDTH * VNNI_BLK
    comptime V_CG_BYTES = SQ_COUNT * SQ_BYTES
    comptime V_PG_BYTES = V_CHANNEL_GROUPS * V_CG_BYTES
    comptime SCORE_STRIDE = HPG * WIDTH
    comptime W_PAGE_BYTES = HPG * K_STEP
    comptime SCORE_F32_PG = HPG * WIDTH
    comptime SIMD_W = simd_width_of[DType.float32]()

    var q_factors = InlineArray[Float32, HPG](uninitialized=True)
    for qh in range(HPG):
        q_factors[qh] = q_scales[qh] / Q_DENOM

    var running_max = InlineArray[Float32, HPG](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, HPG](fill=Float32(0))

    var v_acc_arr = AlignedInlineArray[Float32, HPG * head_dim](fill=Float32(0))
    var v_acc = v_acc_arr.unsafe_ptr()

    var q_arr = AlignedInlineArray[Scalar[DType.int8], HPG * head_dim](
        fill=Scalar[DType.int8](0))
    var q_ptr = q_arr.unsafe_ptr()
    for qh in range(HPG):
        var src = args.q_i8_base + qh * head_dim
        var dst = q_ptr + qh * head_dim
        comptime for chunk in range(head_dim // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    var score_arr = AlignedInlineArray[Int32, 4 * SCORE_STRIDE](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    var w_arr = AlignedInlineArray[UInt8, 4 * W_PAGE_BYTES](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()

    var wd_arr = InlineArray[Float32, 4 * HPG](fill=Float32(0))

    var sf_arr = AlignedInlineArray[Float32, 4 * SCORE_F32_PG](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)

    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var pg = args.start_pg
    while pg < args.end_pg:
        # ── Phase 1: Score 4 PGs (2:2:4) ──
        var k0 = cache.k_pg_ptr(args.kv_head, pg)
        var k1 = cache.k_pg_ptr(args.kv_head, pg + 1)
        var k2 = cache.k_pg_ptr(args.kv_head, pg + 2)
        var k3 = cache.k_pg_ptr(args.kv_head, pg + 3)

        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, head_dim)
        tileload[1, DType.int8](q_ptr + K_STEP, head_dim)

        tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_ptr, WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + SCORE_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * SCORE_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * SCORE_STRIDE, WIDTH * 4)

        # ── Phase 2a: Dequant all 4 PGs, find batch max ──
        var batch_max = InlineArray[Float32, HPG](fill=Float32(-1e30))
        for bp in range(4):
            var s_base = score_ptr + bp * SCORE_STRIDE
            var k_sc = k_scales + (pg + bp) * WIDTH
            var sf_pg = sf_buf + bp * SCORE_F32_PG
            var group_start = (pg + bp) * WIDTH
            var valid = (valid_lanes + Int32(group_start)).lt(
                SIMD[DType.int32, WIDTH](args.context_len))
            for qh in range(HPG):
                var raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var scores = (raw - qi_biases[qh]) * q_factors[qh] * k_sc.load[
                    width=WIDTH]()
                scores = valid.select(scores, neg_inf)
                (sf_pg + qh * WIDTH).store(scores)
                var pg_max = scores.reduce_max()
                if pg_max > batch_max[qh]:
                    batch_max[qh] = pg_max

        # Single rescale for the entire 4-PG batch
        var diff_vec = SIMD[DType.float32, SIMD_W](0)
        var need_rescale = False
        for qh in range(HPG):
            var new_max = max(running_max[qh], batch_max[qh])
            diff_vec[qh] = running_max[qh] - new_max
            if new_max > running_max[qh]:
                need_rescale = True
            running_max[qh] = new_max

        if need_rescale:
            var rescales = exp_f32[SIMD_W](diff_vec)
            for qh in range(HPG):
                if running_sum[qh] > 0:
                    var rs = rescales[qh]
                    running_sum[qh] *= rs
                    var acc = v_acc + qh * head_dim
                    var d = 0
                    while d + WIDTH <= head_dim:
                        (acc + d).store((acc + d).load[width=WIDTH]() * rs)
                        d += WIDTH

        # ── Phase 2b: W quantization (all relative to final running_max) ──
        for bp in range(4):
            var sf_pg = sf_buf + bp * SCORE_F32_PG
            var v_sc = v_scales + (pg + bp) * WIDTH
            var group_start = (pg + bp) * WIDTH
            var valid = (valid_lanes + Int32(group_start)).lt(
                SIMD[DType.int32, WIDTH](args.context_len))
            var w_pg = w_base + bp * W_PAGE_BYTES
            for qh in range(HPG):
                var scores = (sf_pg + qh * WIDTH).load[width=WIDTH]()
                var exp_scores = valid.select(
                    exp_f32_fast[WIDTH](scores - running_max[qh]), zero_vec)
                running_sum[qh] += exp_scores.reduce_add()
                var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                var w_max = w_eff.reduce_max()
                if w_max < Float32(1e-10):
                    (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                    wd_arr[bp * HPG + qh] = Float32(0)
                    continue
                var w_scale = 255.0 / w_max
                (w_pg + qh * K_STEP).store(
                    roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[
                        DType.uint8]())
                wd_arr[bp * HPG + qh] = w_max / 255.0

        # ── Phase 3: SIMD VNNI V-agg ──
        for cg in range(V_CHANNEL_GROUPS):
            var acc = InlineArray[
                SIMD[DType.float32, WIDTH], HPG](uninitialized=True)
            comptime for qh in range(HPG):
                acc[qh] = (v_acc + qh * head_dim + cg * WIDTH).load[
                    width=WIDTH]()

            for bp in range(4):
                var v_cg = cache.v_pg_ptr(args.kv_head, pg + bp) + cg * V_CG_BYTES
                var v_loads = InlineArray[
                    SIMD[DType.int8, WIDTH * VNNI_BLK], SQ_COUNT](
                    uninitialized=True)
                comptime for sq in range(SQ_COUNT):
                    v_loads[sq] = (v_cg + sq * SQ_BYTES).load[
                        width=WIDTH * VNNI_BLK]()
                var w_pg = w_base + bp * W_PAGE_BYTES

                comptime for qh in range(HPG):
                    var w_row = w_pg + qh * K_STEP
                    var dot = SIMD[DType.int32, WIDTH](0)
                    comptime for sq in range(SQ_COUNT):
                        dot = vpdpbusd[WIDTH](
                            dot,
                            bcast_4u8_vnni[WIDTH](w_row + sq * VNNI_BLK),
                            v_loads[sq])
                    acc[qh] = acc[qh] + dot.cast[
                        DType.float32]() * wd_arr[bp * HPG + qh]

            comptime for qh in range(HPG):
                (v_acc + qh * head_dim + cg * WIDTH).store(acc[qh])

        pg += 4

    var out = args.partial_out
    for qh in range(HPG):
        var dst = out + qh * HEAD_STRIDE
        dst[] = running_max[qh]
        (dst + 1)[] = running_sum[qh]
        var src = v_acc + qh * head_dim
        var d = 0
        while d + WIDTH <= head_dim:
            (dst + 2 + d).store((src + d).load[width=WIDTH]())
            d += WIDTH

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
