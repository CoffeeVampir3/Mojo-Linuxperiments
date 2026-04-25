from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from std.sys.info import simd_width_of

from simd_math import sqrt, exp_f32, exp_f32_fast, roundeven
from experimental3.amx import (
    K_STEP, VNNI_BLK, TILE_BYTES,
    tilezero, tileload, tilestore, tdpbsud,
)
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.dot_prod import vpdpbusd
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr, U8Ptr
from notstdcollections import AlignedInlineArray
from minimax.kernels.qk_prep import prep_q_head, write_k_head, write_v_direct
from minimax.kernels.dispatch_args import (
    AttnGroupArgs, MergeQuantArgs, KVWriteBatchArgs, QPrepBatchArgs,
    PrefillAttnArgs,
)


# ============================================================================
# K/V cache write — one job per KV head, loops all positions internally
# ============================================================================


def kv_write_batch_kernel[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    max_seq: Int, num_kv_heads: Int,
](args: KVWriteBatchArgs):
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads](
        Int(args.cache_base))

    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    for i in range(args.pos_count):
        var pos = args.start_pos + i
        var k_ptr = args.k_bf16_base + i * args.qkv_row_stride
        var v_ptr = args.v_bf16_base + i * args.qkv_row_stride
        var cos = args.cos_base + pos * args.rope_row_elems
        var sin = args.sin_base + pos * args.rope_row_elems

        write_k_head[head_dim, rope_dim, pair_stride](
            k_ptr, args.k_norm_ptr, cos, sin,
            args.inv_rms_k_arr[i], qi_buf,
            cache, pos, args.kv_head)

        write_v_direct[head_dim](
            v_ptr, qi_buf,
            cache, pos, args.kv_head)

    _ = qi_arr


# ============================================================================
# Q prep — one job per (KV head, rank), loops all positions internally
# ============================================================================


def q_prep_batch_kernel[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    heads_per_group: Int,
](args: QPrepBatchArgs):
    comptime inv_sqrt_hd = 1.0 / sqrt[DType.float32, 1](Float32(head_dim))

    for i in range(args.pos_count):
        var pos = args.start_pos + i
        var q_row = args.q_bf16_base + i * args.qkv_row_stride
        var cos = args.cos_base + pos * args.rope_row_elems
        var sin = args.sin_base + pos * args.rope_row_elems

        for qh in range(heads_per_group):
            var result = prep_q_head[head_dim, rope_dim, pair_stride](
                q_row + qh * head_dim,
                args.q_norm_ptr + qh * head_dim,
                cos, sin,
                args.inv_rms_q_arr[i],
                (args.qi_out + qh * args.qi_out_head_stride
                    + i * head_dim).bitcast[Int8]())
            (args.qi_biases_out + qh * args.qi_biases_head_stride)[i] = result[0]
            (args.q_scales_out + qh * args.q_scales_head_stride)[i] = (
                result[1] * inv_sqrt_hd)


# ============================================================================
# Phase C: local merge + quantize — replaces cross-rank cp_gather
# ============================================================================


def partial_head_stride[head_dim: Int]() -> Int:
    return 2 + head_dim


def partial_chunk_stride[head_dim: Int, heads_per_group: Int]() -> Int:
    return heads_per_group * partial_head_stride[head_dim]()


def merge_and_quantize_kernel[head_dim: Int, heads_per_group: Int, max_attn_chunks: Int](
    partial_base: F32Ptr,
    num_chunks: Int,
    qi_out: I8Ptr,
    head_scale_ptr: F32Ptr,
):
    """Merge position chunks + quantize output for O-projection.

    Reads partial (max, sum, v_acc) from each chunk, merges via online
    softmax, applies 1/(127*L) normalization, quantizes to i8.
    """
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()
    comptime CHUNK_STRIDE = partial_chunk_stride[head_dim, heads_per_group]()
    comptime width = simd_width_of[DType.float32]()

    var work_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var wp = UnsafePointer(to=work_arr).bitcast[Float32]()

    var m_arr = InlineArray[Float32, max_attn_chunks](fill=Float32(-1e30))
    var cs_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)
    var rescale_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)

    for qh in range(heads_per_group):
        # Gather per-chunk (max, sum) for this head; compute gmax.
        var gmax = Float32(-1e30)
        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var m = p[]
            m_arr[c] = m
            cs_arr[c] = (p + 1)[]
            gmax = max(gmax, m)

        # One batched SIMD exp covers all chunks; inactive lanes stay at -1e30.
        var gmax_vec = SIMD[DType.float32, width](gmax)
        var cg = 0
        while cg < max_attn_chunks:
            var mv = UnsafePointer(to=m_arr[cg]).load[width=width]()
            UnsafePointer(to=rescale_arr[cg]).store(exp_f32[width](mv - gmax_vec))
            cg += width

        var d = 0
        while d + width <= head_dim:
            (wp + d).store(SIMD[DType.float32, width](0))
            d += width

        var total_sum = Float32(0)
        for c in range(num_chunks):
            var cs = cs_arr[c]
            if cs <= 0:
                continue
            var rescale = rescale_arr[c]
            total_sum += cs * rescale
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            d = 0
            while d + width <= head_dim:
                (wp + d).store((p + 2 + d).load[width=width]().fma(
                    rescale, (wp + d).load[width=width]()))
                d += width

        var inv = Float32(1) / (Float32(127) * total_sum)
        d = 0
        while d + width <= head_dim:
            (wp + d).store((wp + d).load[width=width]() * inv)
            d += width

        head_scale_ptr[qh] = absmax_quantize_i8[head_dim](
            wp, qi_out + qh * head_dim)

    _ = work_arr


def merge_quant_worker[
    head_dim: Int,
    heads_per_group: Int,
    max_attn_chunks: Int,
    kv_heads: Int,
](args: MergeQuantArgs):
    if args.num_chunks <= 0:
        return

    comptime CHUNK_F32_STRIDE = partial_chunk_stride[
        head_dim, heads_per_group]()
    comptime KV_QI_STRIDE = heads_per_group * head_dim
    comptime KV_SCALE_STRIDE = heads_per_group

    for kv in range(kv_heads):
        merge_and_quantize_kernel[
            head_dim, heads_per_group, max_attn_chunks](
            args.partial_base + kv * args.num_chunks * CHUNK_F32_STRIDE,
            args.num_chunks,
            args.qi_out + kv * KV_QI_STRIDE,
            args.head_scale_ptr + kv * KV_SCALE_STRIDE,
        )


# ============================================================================
# Prefill attention — one (kv_head, q_tile) per worker, loops HPG heads
# ============================================================================


@always_inline
def prefetch_cacheline(p: U8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(3), Int32(1))


def prefill_attn_worker[
    head_dim: Int, max_seq: Int,
    num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int,
](args: PrefillAttnArgs):
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](
        Int(args.cache_base))
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)

    comptime Q_TILE = 16
    comptime WIDTH = CACHE_WIDTH
    comptime V_CHANNEL_GROUPS = head_dim // WIDTH
    comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
    comptime K_PG_BYTES = head_dim // VNNI_BLK * WIDTH * VNNI_BLK
    comptime SCORE_PG_STRIDE = Q_TILE * WIDTH
    comptime W_TILE_BYTES = Q_TILE * K_STEP
    comptime SQ_BYTES = WIDTH * VNNI_BLK
    comptime SIMD_W = simd_width_of[DType.float32]()

    var actual_q = min(args.q_count, Q_TILE)
    var num_pgs = (args.context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var last_q_pos = args.q_start + actual_q - 1
    var causal_padded = ((last_q_pos // WIDTH + 1) + 3) & ~3
    var effective_pgs = min(padded_pgs, causal_padded)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_f32 = SIMD[DType.float32, WIDTH](Float32(0))
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var score_arr = AlignedInlineArray[Int32, 4 * SCORE_PG_STRIDE](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()
    var w_arr = AlignedInlineArray[UInt8, W_TILE_BYTES](fill=UInt8(0))
    var w_buf = w_arr.unsafe_ptr()
    var wd_arr = InlineArray[Float32, 4 * Q_TILE](fill=Float32(0))
    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()
    var bcast_arr = AlignedInlineArray[UInt32, WIDTH](
        uninitialized=True)
    var bcast_ptr = bcast_arr.unsafe_ptr()

    comptime Q_DENOM = Float32(127) * Float32(127)

    for qh in range(heads_per_group):
        var q_head = args.q_i8 + qh * args.pos_count * head_dim
        var qb = args.qi_biases + qh * args.pos_count
        var qf_raw = args.q_factors + qh * args.pos_count
        var qf = InlineArray[Float32, Q_TILE](fill=Float32(0))
        for r in range(actual_q):
            qf[r] = qf_raw[r] / Q_DENOM

        var q_arr = AlignedInlineArray[Scalar[DType.int8], Q_TILE * head_dim](
            fill=Scalar[DType.int8](0))
        var q_ptr = q_arr.unsafe_ptr()
        for r in range(actual_q):
            var src = q_head + r * head_dim
            var dst = q_ptr + r * head_dim
            comptime for chunk in range(head_dim // K_STEP):
                (dst + chunk * K_STEP).store(
                    (src + chunk * K_STEP).load[width=K_STEP]())

        var running_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
        var running_sum = InlineArray[Float32, Q_TILE](fill=Float32(0))
        var v_acc_arr = AlignedInlineArray[Float32, Q_TILE * head_dim](
            fill=Float32(0))
        var v_acc = v_acc_arr.unsafe_ptr()

        var pg = 0
        while pg < effective_pgs:
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()
            tileload[0, DType.int8](q_ptr, head_dim)
            tileload[1, DType.int8](q_ptr + K_STEP, head_dim)

            var k0 = cache.k_pg_ptr(args.kv_head, pg)
            var k1 = cache.k_pg_ptr(args.kv_head, pg + 1)
            var k2 = cache.k_pg_ptr(args.kv_head, pg + 2)
            var k3 = cache.k_pg_ptr(args.kv_head, pg + 3)

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
            tilestore[5, DType.int32](score_ptr + SCORE_PG_STRIDE, WIDTH * 4)
            tilestore[6, DType.int32](score_ptr + 2 * SCORE_PG_STRIDE, WIDTH * 4)
            tilestore[7, DType.int32](score_ptr + 3 * SCORE_PG_STRIDE, WIDTH * 4)

            var nxt = pg + 4
            if nxt < effective_pgs:
                var k_nxt = cache.k_pg_ptr(args.kv_head, nxt)
                comptime for ln in range(0, K_PG_BYTES, 64):
                    prefetch_cacheline((k_nxt + ln).bitcast[UInt8]())

            var batch_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
            for r in range(actual_q):
                var q_pos = args.q_start + r
                for bp in range(4):
                    var group_start = (pg + bp) * WIDTH
                    var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                        SIMD[DType.int32, WIDTH](args.context_len))
                    var causal_valid = (valid_lanes + Int32(group_start)).le(
                        SIMD[DType.int32, WIDTH](q_pos))
                    var valid = ctx_valid & causal_valid
                    var raw = (score_ptr + bp * SCORE_PG_STRIDE + r * WIDTH).load[
                        width=WIDTH]().cast[DType.float32]()
                    var k_sc = k_scales + (pg + bp) * WIDTH
                    var scores = (raw - qb[r]) * qf[r] * k_sc.load[width=WIDTH]()
                    scores = valid.select(scores, neg_inf)
                    (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                    var pg_max = scores.reduce_max()
                    if pg_max > batch_max[r]:
                        batch_max[r] = pg_max

            for r in range(actual_q):
                var new_max = max(running_max[r], batch_max[r])
                if running_sum[r] > 0 and new_max > running_max[r]:
                    var rescale = Float32(exp_f32[1](running_max[r] - new_max))
                    running_sum[r] *= rescale
                    var acc = v_acc + r * head_dim
                    var d = 0
                    while d + WIDTH <= head_dim:
                        (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                        d += WIDTH
                running_max[r] = new_max

                var w_row = w_buf + r * K_STEP
                for bp in range(4):
                    var group_start = (pg + bp) * WIDTH
                    var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                        SIMD[DType.int32, WIDTH](args.context_len))
                    var causal_valid = (valid_lanes + Int32(group_start)).le(
                        SIMD[DType.int32, WIDTH](args.q_start + r))
                    var valid = ctx_valid & causal_valid
                    var v_sc = v_scales + (pg + bp) * WIDTH
                    var exp_scores = valid.select(exp_f32_fast[WIDTH](
                        (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[width=WIDTH]()
                        - running_max[r]), zero_f32)
                    running_sum[r] += exp_scores.reduce_add()
                    var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                    var w_max = w_eff.reduce_max()
                    if w_max < Float32(1e-10):
                        (w_row + bp * WIDTH).store[width=WIDTH](
                            SIMD[DType.uint8, WIDTH](0))
                        wd_arr[bp * Q_TILE + r] = Float32(0)
                        continue

                    var w_scale_val = 255.0 / w_max
                    var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                        0.0, 255.0).cast[DType.uint8]()
                    (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
                    wd_arr[bp * Q_TILE + r] = w_max / 255.0

            for cg in range(V_CHANNEL_GROUPS):
                for bp in range(4):
                    var v_pg = cache.v_pg_ptr(args.kv_head, pg + bp)
                    var v_cg = v_pg + cg * V_CG_BYTES
                    var v0 = (v_cg + 0 * SQ_BYTES).load[
                        width=WIDTH * VNNI_BLK]()
                    var v1 = (v_cg + 1 * SQ_BYTES).load[
                        width=WIDTH * VNNI_BLK]()
                    var v2 = (v_cg + 2 * SQ_BYTES).load[
                        width=WIDTH * VNNI_BLK]()
                    var v3 = (v_cg + 3 * SQ_BYTES).load[
                        width=WIDTH * VNNI_BLK]()

                    for r in range(actual_q):
                        var wd = wd_arr[bp * Q_TILE + r]
                        if wd <= Float32(0):
                            continue

                        var w_row = w_buf + r * K_STEP + bp * WIDTH
                        var w0 = (w_row + 0 * VNNI_BLK).bitcast[UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w0))
                        var a0 = vpdpbusd[WIDTH](
                            SIMD[DType.int32, WIDTH](0),
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            v0)

                        var w1 = (w_row + 1 * VNNI_BLK).bitcast[UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w1))
                        var a1 = vpdpbusd[WIDTH](
                            SIMD[DType.int32, WIDTH](0),
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            v1)

                        var w2 = (w_row + 2 * VNNI_BLK).bitcast[UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w2))
                        var a2 = vpdpbusd[WIDTH](
                            SIMD[DType.int32, WIDTH](0),
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            v2)

                        var w3 = (w_row + 3 * VNNI_BLK).bitcast[UInt32]()[]
                        bcast_ptr.store(SIMD[DType.uint32, WIDTH](w3))
                        var a3 = vpdpbusd[WIDTH](
                            SIMD[DType.int32, WIDTH](0),
                            bcast_ptr.bitcast[UInt8]().load[
                                width=WIDTH * VNNI_BLK](),
                            v3)

                        var dst = v_acc + r * head_dim + cg * WIDTH
                        dst.store(dst.load[width=WIDTH]() + ((a0 + a1)
                            + (a2 + a3)).cast[DType.float32]() * wd)

            pg += 4

        for r in range(actual_q):
            var row_out = args.qi_out + r * args.qi_out_row_stride
                + args.head_col_offset + qh * head_dim
            var inv = Float32(0)
            if running_sum[r] > Float32(1e-10):
                inv = Float32(1) / (Float32(127) * running_sum[r])
            var src = v_acc + r * head_dim
            var d = 0
            while d + SIMD_W <= head_dim:
                (src + d).store((src + d).load[width=SIMD_W]() * inv)
                d += SIMD_W
            var scale = absmax_quantize_i8[head_dim](src, row_out)
            (args.head_sc_out + r * args.head_sc_row_stride
                + args.head_col_offset // head_dim + qh)[] = scale

        _ = q_arr
        _ = v_acc_arr

    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = bcast_arr
