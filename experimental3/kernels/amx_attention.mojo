from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.amx import (
    TILE_M, TILE_K, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    TileConfig, make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbsud, tdpbusd,
)
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.dispatch_args import ChunkedAttnArgs
from simd_math import exp_f32, roundeven


@fieldwise_init
struct AmxConfigArgs(Copyable, ImplicitlyCopyable):
    var dummy: Int

    def __init__(out self):
        self.dummy = 0


def amx_config_kernel(args: AmxConfigArgs):
    _ = init_intel_amx()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))


def amx_chunked_attn_kernel[
    head_dim: Int, local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int, max_attn_chunks: Int,
](args: ChunkedAttnArgs):
    """AMX-accelerated chunked attention — drop-in for cp_chunked_attn_kernel.

    Scoring:  tdpbsud — Q(i8) × K_cache(u8), direct cache tileloads.
    V-agg:   tdpbusd — W(u8, zero-padded to K_STEP) × V_cache(i8), direct.
    Zero-padding in W masks garbage B-tile rows from adjacent cache CGs.
    """
    var cache = Gemma4KVCache[local_max_seq, head_dim, num_kv_heads, num_q_heads](
        Int(args.cache_base))
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)
    var qi_biases = args.qi_biases_base
    var q_scales = args.q_scales_base

    comptime WIDTH = CACHE_WIDTH
    comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
    comptime V_CHANNEL_GROUPS = head_dim // WIDTH
    comptime NUM_K_STEPS = head_dim // K_STEP
    comptime Q_DENOM = Float32(127) * Float32(127)
    comptime HEAD_STRIDE = 2 + head_dim

    var q_factors = InlineArray[Float32, heads_per_group](uninitialized=True)
    for qh in range(heads_per_group):
        q_factors[qh] = q_scales[qh] / Q_DENOM

    var running_max = InlineArray[Float32, heads_per_group](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, heads_per_group](fill=Float32(0))

    comptime V_ACC_TOTAL = heads_per_group * head_dim
    var v_acc_arr = InlineArray[Float32, V_ACC_TOTAL](fill=Float32(0))
    var v_acc_base = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    # Q tile: [TILE_M, head_dim] i8 — zero-padded rows beyond heads_per_group
    var q_buf = InlineArray[Scalar[DType.int8], TILE_M * head_dim](
        fill=Scalar[DType.int8](0))
    var q_ptr = UnsafePointer(to=q_buf).bitcast[Scalar[DType.int8]]()
    for qh in range(heads_per_group):
        var src = args.q_i8_base + qh * head_dim
        var dst = q_ptr + qh * head_dim
        comptime for chunk in range(head_dim // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    # Score tile output: [TILE_M, WIDTH] i32
    var score_arr = InlineArray[Int32, TILE_M * WIDTH](uninitialized=True)
    var score_ptr = UnsafePointer(to=score_arr).bitcast[Int32]()

    # W tile: [TILE_M, K_STEP] u8 — cols WIDTH..K_STEP-1 always zero
    var w_buf = InlineArray[UInt8, TILE_M * K_STEP](fill=UInt8(0))
    var w_ptr = UnsafePointer(to=w_buf).bitcast[UInt8]()

    # V-agg C tile scratch: 2 tiles worth for CG pairs
    var vc_arr = InlineArray[Int32, TILE_M * TILE_N * 2](uninitialized=True)
    var vc_ptr = UnsafePointer(to=vc_arr).bitcast[Int32]()

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)

    comptime EXP_W = simd_width_of[DType.float32]()
    debug_assert(heads_per_group <= EXP_W,
        "heads_per_group must fit in one SIMD vector")
    var scores_per_head = InlineArray[
        SIMD[DType.float32, WIDTH], heads_per_group](uninitialized=True)
    var new_max_arr = InlineArray[Float32, heads_per_group](uninitialized=True)

    for pg in range(args.start_pg, args.end_pg):
        var k_pg = cache.k_pg_ptr(args.kv_head, pg)
        var v_pg = cache.v_pg_ptr(args.kv_head, pg)
        var k_sc_pg = k_scales + pg * WIDTH
        var v_sc_pg = v_scales + pg * WIDTH
        var group_start = pg * WIDTH

        var valid_lanes = SIMD[DType.int32, WIDTH]()
        comptime for lane in range(WIDTH):
            valid_lanes[lane] = Int32(lane)
        var valid = (valid_lanes + Int32(group_start)).lt(
            SIMD[DType.int32, WIDTH](args.context_len))

        # ── Scoring: Q_i8[HPG,128] × K_u8[128,16]^T → S_i32[HPG,16] ──
        tilezero[4]()
        comptime for ks in range(NUM_K_STEPS):
            tileload[0, DType.int8](q_ptr + ks * K_STEP, head_dim)
            tileload[2, DType.uint8](k_pg + ks * TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
        tilestore[4, DType.int32](score_ptr, WIDTH * 4)

        # Dequant to f32, find new max per head
        var diff_vec = SIMD[DType.float32, EXP_W](0)
        for qh in range(heads_per_group):
            var raw = (score_ptr + qh * WIDTH).load[width=WIDTH]().cast[
                DType.float32]()
            var scores = (raw - qi_biases[qh]) * q_factors[qh] * k_sc_pg.load[
                width=WIDTH]()
            scores = valid.select(scores, neg_inf)
            scores_per_head[qh] = scores
            var new_max = max(running_max[qh], scores.reduce_max())
            new_max_arr[qh] = new_max
            diff_vec[qh] = running_max[qh] - new_max

        var rescales = exp_f32[EXP_W](diff_vec)

        # ── Softmax rescale + weight quantize + AMX V-agg ──
        var w_dequants = InlineArray[Float32, heads_per_group](
            uninitialized=True)

        for qh in range(heads_per_group):
            var new_max = new_max_arr[qh]
            var v_acc = v_acc_base + qh * head_dim

            if running_sum[qh] > 0:
                var rescale = rescales[qh]
                var d = 0
                while d + WIDTH <= head_dim:
                    (v_acc + d).store(
                        (v_acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
                running_sum[qh] *= rescale

            running_max[qh] = new_max
            var exp_scores = valid.select(
                exp_f32[WIDTH](scores_per_head[qh] - new_max), zero_vec)
            running_sum[qh] += exp_scores.reduce_add()

            var w_eff = exp_scores * v_sc_pg.load[width=WIDTH]()
            var w_max = w_eff.reduce_max()
            if w_max < Float32(1e-10):
                (w_ptr + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                w_dequants[qh] = Float32(0)
                continue
            var w_scale = 255.0 / w_max
            var w_u8 = roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[
                DType.uint8]()
            w_dequants[qh] = w_max / 255.0
            (w_ptr + qh * K_STEP).store(w_u8)

        # Load W tile once — reused across all CG pairs
        tileload[0, DType.uint8](w_ptr, K_STEP)

        # V-agg: channel groups in pairs, direct cache B-tile loads
        comptime for cg_pair in range(V_CHANNEL_GROUPS // 2):
            comptime cg0 = cg_pair * 2
            comptime cg1 = cg0 + 1

            tilezero[4]()
            tilezero[5]()
            tileload[2, DType.int8](
                v_pg + cg0 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](
                v_pg + cg1 * V_CG_BYTES, WIDTH * VNNI_BLK)
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](vc_ptr, TILE_N * 4)
            tilestore[5, DType.int32](
                vc_ptr + TILE_M * TILE_N, TILE_N * 4)

            for qh in range(heads_per_group):
                var wd = w_dequants[qh]
                if wd == 0:
                    continue
                var i0 = (vc_ptr + qh * TILE_N).load[width=WIDTH]()
                var i1 = (vc_ptr + TILE_M * TILE_N + qh * TILE_N).load[
                    width=WIDTH]()
                var a0 = v_acc_base + qh * head_dim + cg0 * WIDTH
                var a1 = v_acc_base + qh * head_dim + cg1 * WIDTH
                (a0).store(
                    (a0).load[width=WIDTH]()
                    + i0.cast[DType.float32]() * wd)
                (a1).store(
                    (a1).load[width=WIDTH]()
                    + i1.cast[DType.float32]() * wd)

    # Write partial output: [max, sum, v_acc[head_dim]] per head
    var out = args.partial_out
    for qh in range(heads_per_group):
        var dst = out + qh * HEAD_STRIDE
        dst[] = running_max[qh]
        (dst + 1)[] = running_sum[qh]
        var src = v_acc_base + qh * head_dim
        var d = 0
        while d + WIDTH <= head_dim:
            (dst + 2 + d).store((src + d).load[width=WIDTH]())
            d += WIDTH

    _ = q_buf
    _ = score_arr
    _ = w_buf
    _ = vc_arr
    _ = v_acc_arr
