"""Fused AMX int8 GQA attention (FlashAttention-style).

One pass over KV cache: for each tile of 64 positions, fuse
score → dequant → online softmax → W quantize → V agg.

Score buffer (4KB) and W VNNI (1KB) live on stack, deep L1.
No separate passes over score data. Softmax work overlaps with
memory stalls from KV cache streaming.

Scoring:  S^T[64, gqa] = K_u8[64, hd] x Q_vnni[hd, gqa]
V agg:    O^T[hd, gqa] += V_u8[hd, 64] x W_vnni[64, gqa]
Both use tdpbusd: A=u8 (cache), B=i8 (Q/W VNNI-packed).
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32, exp_f32_fast, quantize_i8, quantize_i8_scalar
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.hadquant_attn import HadAttnCtx
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tdpbusd,
)


# ============================================================================
# Timing phases
# ============================================================================

comptime NUM_PHASES = 8
comptime PHASE_QPREP = 0
comptime PHASE_SCORE_GEMM = 1
comptime PHASE_DEQUANT_MAX = 2
comptime PHASE_SOFTMAX = 3
comptime PHASE_W_QUANT = 4
comptime PHASE_VAGG_GEMM = 5
comptime PHASE_VAGG_DEQUANT = 6
comptime PHASE_NORMALIZE = 7

comptime TIMING_BYTES = NUM_PHASES * size_of[Int64]()

def phase_name(p: Int) -> String:
    if p == 0: return "Q prep"
    if p == 1: return "Score GEMM"
    if p == 2: return "Dequant+Max"
    if p == 3: return "Softmax+WQ"
    if p == 4: return "W Quantize"
    if p == 5: return "V agg GEMM"
    if p == 6: return "V agg Dequant"
    if p == 7: return "Normalize"
    return "?"


# ============================================================================
# Scratch sizing (simplified: only running_o needed in scratch)
# ============================================================================

def attn_scratch_bytes_amx[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                           num_workers: Int = 1]() -> Int:
    comptime q_cols = num_heads * head_dim
    comptime per_worker = head_dim * TILE_N * size_of[Float32]()  # running_o [hd, TILE_N]
    return q_cols * size_of[Float32]() + num_workers * per_worker


# ============================================================================
# Dispatched Kernel (fused tile loop)
# ============================================================================

def had_attn_groups_amx[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, m: Int, group_start: Int, group_end: Int,
    worker_scratch: Int, timing_addr: Int,
):
    var ctx = UnsafePointer[HadAttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var timing = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=timing_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    comptime FUSED_TILE = K_STEP  # 64 positions per fused tile

    comptime DATA_HEAD_STRIDE = max_seq * head_dim
    comptime SCALE_HEAD_STRIDE = max_seq * size_of[Float32]()
    comptime QI_VNNI_TILES = head_dim // K_STEP

    var actual_pos = ctx[].pos + m
    var context = actual_pos + 1
    var cos_row = ctx[].cos + actual_pos * half
    var sin_row = ctx[].sin + actual_pos * half

    var running_o = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch)

    # Tile config
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    # Stack buffers (all L1-resident)
    var qi_vnni_arr = InlineArray[Scalar[DType.int8], QI_VNNI_TILES * TILE_BYTES](
        fill=Scalar[DType.int8](0))
    var qi_vnni = UnsafePointer(to=qi_vnni_arr).bitcast[Scalar[DType.int8]]()
    var score_arr = InlineArray[Int32, FUSED_TILE * TILE_N](fill=Int32(0))
    var score_i32 = UnsafePointer(to=score_arr).bitcast[Int32]()
    var score_f32 = UnsafePointer(to=score_arr).bitcast[Float32]()
    var w_vnni_arr = InlineArray[Scalar[DType.int8], TILE_BYTES](
        fill=Scalar[DType.int8](0))
    var w_vnni = UnsafePointer(to=w_vnni_arr).bitcast[Scalar[DType.int8]]()
    var vagg_arr = InlineArray[Int32, M_STEP * TILE_N](fill=Int32(0))
    var vagg_buf = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    var q_scales_arr = InlineArray[Float32, TILE_N](fill=Float32(0))
    var q_biases_arr = InlineArray[Float32, TILE_N](fill=Float32(0))
    var q_scales = UnsafePointer(to=q_scales_arr).bitcast[Float32]()
    var q_biases = UnsafePointer(to=q_biases_arr).bitcast[Float32]()

    comptime score_stride = TILE_N * size_of[Int32]()
    comptime vagg_stride = TILE_N * size_of[Int32]()

    for g in range(group_start, group_end):
        var t0 = perf_counter_ns()

        # =================================================================
        # 1. Q prep: RoPE + FWHT + quantize directly to VNNI layout
        # =================================================================
        for hi in range(gqa_factor):
            var h = g * gqa_factor + hi
            var q_head = InlineArray[Float32, head_dim](fill=Float32(0))
            var qp = UnsafePointer(to=q_head).bitcast[Float32]()
            var q_row = ctx[].q + m * q_cols + h * head_dim
            var k = 0
            while k + width <= head_dim:
                (qp + k).store((q_row + k).load[width=width]().cast[DType.float32]())
                k += width

            var j = 0
            while j + width <= half:
                var x_lo = (qp + j).load[width=width]()
                var x_hi = (qp + half + j).load[width=width]()
                var cv = (cos_row + j).load[width=width]()
                var sv = (sin_row + j).load[width=width]()
                (qp + j).store(x_lo * cv - x_hi * sv)
                (qp + half + j).store(x_hi * cv + x_lo * sv)
                j += width

            fwht_block[DType.float32, head_dim](qp)

            var vmax = SIMD[DType.float32, width](0)
            k = 0
            while k + width <= head_dim:
                vmax = max(vmax, (qp + k).load[width=width]().__abs__())
                k += width
            var q_absmax = vmax.reduce_max()
            var q_inv = Float32(127.0) / q_absmax if q_absmax > 0 else Float32(0)

            var q_sum = Int32(0)
            for d in range(0, head_dim, VNNI_BLK):
                var qi = quantize_i8(
                    (qp + d).load[width=VNNI_BLK](),
                    SIMD[DType.float32, VNNI_BLK](q_inv))
                var tile = d // K_STEP
                var kg = (d % K_STEP) // VNNI_BLK
                (qi_vnni + tile * TILE_BYTES + kg * 64 + hi * VNNI_BLK).store(qi)
                q_sum += qi.cast[DType.int32]().reduce_add()

            q_scales[hi] = (q_absmax / Float32(127.0)) * inv_sqrt_hd
            q_biases[hi] = Float32(128 * Int(q_sum))

        timing[PHASE_QPREP] += Int64(perf_counter_ns() - t0)

        # =================================================================
        # 2. Init running state + precompute fixed W quantize scale
        # =================================================================
        for i in range(head_dim * TILE_N):
            running_o[i] = Float32(0)
        var running_m_vec = SIMD[DType.float32, TILE_N](Float32(-1e30))
        var running_l_vec = SIMD[DType.float32, TILE_N](Float32(0))

        var k_base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=ctx[].k_data + g * DATA_HEAD_STRIDE)
        var k_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].k_scale + g * SCALE_HEAD_STRIDE)
        var v_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].v_scale + g * SCALE_HEAD_STRIDE)
        var v_head = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=ctx[].v_data + g * DATA_HEAD_STRIDE)

        # Fixed W quantize scale from global v_scale_max
        var v_sc_max = Float32(0)
        for t in range(context):
            if v_sc_base[t] > v_sc_max:
                v_sc_max = v_sc_base[t]
        var w_scale_vec = SIMD[DType.float32, TILE_N](v_sc_max / Float32(127.0))
        var w_inv_vec = SIMD[DType.float32, TILE_N](
            Float32(127.0) / v_sc_max if v_sc_max > 0 else Float32(0))

        var q_bias_vec = q_biases.load[width=TILE_N]()
        var q_scale_vec = q_scales.load[width=TILE_N]()

        # =================================================================
        # 3. Fused tile loop — one pass over KV cache
        # =================================================================
        for tile_start in range(0, context, FUSED_TILE):
            var tile_len = min(FUSED_TILE, context - tile_start)
            var tile_m_iters = (tile_len + M_STEP - 1) // M_STEP

            # --- Score GEMM ---
            t0 = perf_counter_ns()
            for m_idx in range(tile_m_iters):
                var m_blk = m_idx * M_STEP
                tilezero[4]()
                tilezero[6]()
                for k_blk in range(0, head_dim, K_STEP):
                    tileload[0](
                        k_base + (tile_start + m_blk) * head_dim + k_blk,
                        head_dim)
                    tileload[1](
                        k_base + (tile_start + m_blk + TILE_M) * head_dim + k_blk,
                        head_dim)
                    tileload[2](
                        qi_vnni + (k_blk // K_STEP) * TILE_BYTES,
                        TILE_N * VNNI_BLK)
                    tdpbusd[4, 0, 2]()
                    tdpbusd[6, 1, 2]()
                tilestore[4](score_i32 + m_blk * TILE_N, score_stride)
                tilestore[6](score_i32 + (m_blk + TILE_M) * TILE_N, score_stride)
            timing[PHASE_SCORE_GEMM] += Int64(perf_counter_ns() - t0)

            # --- Fused dequant + softmax + W quantize ---
            t0 = perf_counter_ns()

            # Dequant i32→f32 + find tile max (4KB, L1-hot from tilestore)
            var tile_max_vec = SIMD[DType.float32, TILE_N](Float32(-1e30))
            for t in range(tile_len):
                var raw = (score_i32 + t * TILE_N).load[width=TILE_N]().cast[DType.float32]()
                var ksc = SIMD[DType.float32, TILE_N](k_sc_base[tile_start + t])
                var dequant = (raw - q_bias_vec) * q_scale_vec * ksc
                (score_f32 + t * TILE_N).store(dequant)
                tile_max_vec = max(tile_max_vec, dequant)

            # Online softmax update + rescale running_o
            var m_new_vec = max(running_m_vec, tile_max_vec)
            var correction_vec = exp_f32(running_m_vec - m_new_vec)
            running_m_vec = m_new_vec
            running_l_vec = running_l_vec * correction_vec
            for d in range(head_dim):
                var row = (running_o + d * TILE_N).load[width=TILE_N]()
                (running_o + d * TILE_N).store(row * correction_vec)

            # Exp + V scale + quantize to W VNNI (fused, 4KB L1-hot)
            # Zero w_vnni for partial last tile
            var i8zero = SIMD[DType.int8, TILE_K](0)
            for zi in range(0, TILE_BYTES, TILE_K):
                (w_vnni + zi).store(i8zero)

            var l_contrib_vec = SIMD[DType.float32, TILE_N](0)
            var w_sum_vec = SIMD[DType.int32, TILE_N](0)

            var t = 0
            while t + VNNI_BLK <= tile_len:
                var e0 = exp_f32_fast((score_f32 + (t + 0) * TILE_N).load[width=TILE_N]() - m_new_vec)
                var e1 = exp_f32_fast((score_f32 + (t + 1) * TILE_N).load[width=TILE_N]() - m_new_vec)
                var e2 = exp_f32_fast((score_f32 + (t + 2) * TILE_N).load[width=TILE_N]() - m_new_vec)
                var e3 = exp_f32_fast((score_f32 + (t + 3) * TILE_N).load[width=TILE_N]() - m_new_vec)
                l_contrib_vec += e0 + e1 + e2 + e3

                var vs0 = SIMD[DType.float32, TILE_N](v_sc_base[tile_start + t + 0])
                var vs1 = SIMD[DType.float32, TILE_N](v_sc_base[tile_start + t + 1])
                var vs2 = SIMD[DType.float32, TILE_N](v_sc_base[tile_start + t + 2])
                var vs3 = SIMD[DType.float32, TILE_N](v_sc_base[tile_start + t + 3])

                var q0 = quantize_i8(e0 * vs0, w_inv_vec)
                var q1 = quantize_i8(e1 * vs1, w_inv_vec)
                var q2 = quantize_i8(e2 * vs2, w_inv_vec)
                var q3 = quantize_i8(e3 * vs3, w_inv_vec)
                w_sum_vec += q0.cast[DType.int32]() + q1.cast[DType.int32]() + q2.cast[DType.int32]() + q3.cast[DType.int32]()

                var d0 = q0.cast[DType.uint8]().cast[DType.uint32]()
                var d1 = q1.cast[DType.uint8]().cast[DType.uint32]()
                var d2 = q2.cast[DType.uint8]().cast[DType.uint32]()
                var d3 = q3.cast[DType.uint8]().cast[DType.uint32]()
                var dwords = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24)
                (w_vnni + (t // VNNI_BLK) * 64).bitcast[Scalar[DType.uint32]]().store(dwords)
                t += VNNI_BLK

            while t < tile_len:
                var e = exp_f32_fast((score_f32 + t * TILE_N).load[width=TILE_N]() - m_new_vec)
                l_contrib_vec += e
                var vs = SIMD[DType.float32, TILE_N](v_sc_base[tile_start + t])
                var qi = quantize_i8(e * vs, w_inv_vec)
                w_sum_vec += qi.cast[DType.int32]()
                var b = t % VNNI_BLK
                var kg = t // VNNI_BLK
                for hi in range(gqa_factor):
                    w_vnni[kg * 64 + hi * VNNI_BLK + b] = qi[hi]
                t += 1

            running_l_vec = running_l_vec + l_contrib_vec
            var w_bias_vec = SIMD[DType.float32, TILE_N](128.0) * w_sum_vec.cast[DType.float32]() * w_scale_vec
            timing[PHASE_SOFTMAX] += Int64(perf_counter_ns() - t0)

            # --- V agg GEMM + dequant ---
            t0 = perf_counter_ns()
            for m_blk in range(0, head_dim, M_STEP):
                tilezero[4]()
                tilezero[6]()
                tileload[0](v_head + m_blk * max_seq + tile_start, max_seq)
                tileload[1](v_head + (m_blk + TILE_M) * max_seq + tile_start, max_seq)
                tileload[2](w_vnni, TILE_N * VNNI_BLK)
                tdpbusd[4, 0, 2]()
                tdpbusd[6, 1, 2]()
                tilestore[4](vagg_buf, vagg_stride)
                tilestore[6](vagg_buf + TILE_M * TILE_N, vagg_stride)

                for r in range(M_STEP):
                    var raw = (vagg_buf + r * TILE_N).load[width=TILE_N]().cast[DType.float32]()
                    var dequant = raw * w_scale_vec - w_bias_vec
                    var existing = (running_o + (m_blk + r) * TILE_N).load[width=TILE_N]()
                    (running_o + (m_blk + r) * TILE_N).store(existing + dequant)
            timing[PHASE_VAGG_GEMM] += Int64(perf_counter_ns() - t0)

        # =================================================================
        # 4. Final normalization
        # =================================================================
        t0 = perf_counter_ns()
        var inv_l_vec = SIMD[DType.float32, TILE_N](1.0) / running_l_vec
        for d in range(head_dim):
            var row = (running_o + d * TILE_N).load[width=TILE_N]() * inv_l_vec
            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                ctx[].row_f32[h * head_dim + d] = row[hi]
        timing[PHASE_NORMALIZE] += Int64(perf_counter_ns() - t0)


# ============================================================================
# Public API
# ============================================================================

def int8_gqa_attention_amx[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
    QT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    q: DynView[QT],
    k_cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    v_cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    scratch: Int,
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused AMX int8 GQA attention. Drop-in replacement for int8_gqa_attention."""
    comptime assert QT.DTYPE == DType.bfloat16, "Q must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "scale output must be f32"

    comptime width = simd_width_of[DType.float32]()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS

    comptime PER_WORKER = head_dim * TILE_N * size_of[Float32]()
    comptime ROW_F32_OFF = 0
    comptime WORKERS_OFF = q_cols * size_of[Float32]()

    var row_f32 = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scratch + ROW_F32_OFF)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scale_out.ptr)

    var ctx = HadAttnCtx(
        q=UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr),
        qi=qi_ptr, sc=sc_ptr,
        cos=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        sin=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        k_data=k_cache.data_base, k_scale=k_cache.scale_base,
        v_data=v_cache.data_base, v_scale=v_cache.scale_base,
        row_f32=row_f32, pos=pos, seq_len=q.seq_len,
    )

    var ctx_ptr = UnsafePointer(to=ctx)
    var num_jobs = min(num_kv_heads, pool.capacity)
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

    comptime MAX_WORKERS = 16
    var timing_arr = InlineArray[Int64, MAX_WORKERS * NUM_PHASES](fill=Int64(0))
    var timing_base = UnsafePointer(to=timing_arr).bitcast[Int64]()

    for m in range(q.seq_len):
        for i in range(num_jobs):
            var start = i * groups_per_job
            var end = min(start + groups_per_job, num_kv_heads)
            var pack = pool.args_base + i
            pack[].arg0 = Int(ctx_ptr)
            pack[].arg1 = m
            pack[].arg2 = start
            pack[].arg3 = end
            pack[].arg4 = scratch + WORKERS_OFF + i * PER_WORKER
            pack[].arg5 = Int(timing_base + i * NUM_PHASES)

        pool.dispatch(
            had_attn_groups_amx[num_heads, num_kv_heads, head_dim, max_seq],
            pool.args_base, num_jobs,
        )
        pool.join()

        # Output quantization
        var out_row = qi_ptr + m * q_cols
        var rmax_v = SIMD[DType.float32, width](0)
        var d = 0
        while d + width <= q_cols:
            rmax_v = max(rmax_v, (row_f32 + d).load[width=width]().__abs__())
            d += width
        var row_absmax = rmax_v.reduce_max()
        while d < q_cols:
            var a = row_f32[d] if row_f32[d] >= 0 else -row_f32[d]
            if a > row_absmax:
                row_absmax = a
            d += 1

        sc_ptr[m] = row_absmax / Float32(127.0)
        var row_inv = Float32(127.0) / row_absmax if row_absmax > 0 else Float32(0)
        var vrow_inv = SIMD[DType.float32, width](row_inv)
        d = 0
        while d + width <= q_cols:
            (out_row + d).store(quantize_i8((row_f32 + d).load[width=width](), vrow_inv))
            d += width
        while d < q_cols:
            out_row[d] = quantize_i8_scalar(row_f32[d], row_inv)
            d += 1

    # Timing report
    var phase_max = InlineArray[Int64, NUM_PHASES](fill=Int64(0))
    var phase_max_ptr = UnsafePointer(to=phase_max).bitcast[Int64]()
    for i in range(num_jobs):
        var wt = timing_base + i * NUM_PHASES
        for p in range(NUM_PHASES):
            if wt[p] > phase_max_ptr[p]:
                phase_max_ptr[p] = wt[p]
    var total = Int64(0)
    for p in range(NUM_PHASES):
        total += phase_max_ptr[p]
    if total > 0:
        print("  Phases (max worker, us):")
        for p in range(NUM_PHASES):
            var us = Int(phase_max_ptr[p]) // 1000
            var pct = Int(phase_max_ptr[p] * 100 // total)
            print("    " + phase_name(p) + ": " + String(us) + " (" + String(pct) + "%)")

    return PoolFence.completed()
