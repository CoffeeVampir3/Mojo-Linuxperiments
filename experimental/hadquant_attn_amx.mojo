"""AMX int8 GQA attention with blocked online softmax.

Drop-in replacement for hadquant_attn using AMX tile GEMMs instead of
VNNI dot products. Same public API: int8_gqa_attention_amx[...](...).

Key structural differences from the VNNI kernel:
  - Scoring: tile GEMM Q_i8[gqa,hd] x K_u8[hd,block] via tdpbsud
  - V agg: batched tile GEMM W_i8[gqa,bc] x V_u8[bc,hd] via tdpbsud
  - Softmax runs for ALL heads before V agg (batch W matrix for tile op)
  - Pack functions gather KV cache data into VNNI layout per block

Uses the 2-2-4 tile config (set per worker dispatch, ~100 cycles).
For decode (gqa_factor=16=TILE_M), only the top A tile and C tiles
TMM4/TMM5 are used (TMM1/TMM6/TMM7 skipped).
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32, quantize_i8, quantize_i8_scalar
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.hadquant_attn import HadAttnCtx, ATTN_B_C
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tdpbsud,
)


# ============================================================================
# VNNI Packing for KV Cache
# ============================================================================

@always_inline
def pack_k_vnni(
    k_base: UnsafePointer[UInt8, MutAnyOrigin],
    head_dim: Int,
    k_off: Int,
    t_off: Int,
    n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pack K cache into VNNI for scoring B operand.

    K cache is [pos][dim] u8 per head. Packs K^T[k_off:k_off+64, t_off:t_off+n_cols]
    into one VNNI tile: dst[kg*64 + col*4 + b] = K[t_off + col, k_off + 4*kg + b].
    The 4 bytes per VNNI group are contiguous in K (dim-contiguous).
    """
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = k_base + (t_off + col) * head_dim + k_off + 4 * kg
            var d = kg * 64 + col * 4
            dst[d] = src[0]
            dst[d + 1] = src[1]
            dst[d + 2] = src[2]
            dst[d + 3] = src[3]
        for col in range(n_cols, TILE_N):
            var d = kg * 64 + col * 4
            dst[d] = 0
            dst[d + 1] = 0
            dst[d + 2] = 0
            dst[d + 3] = 0


@always_inline
def pack_v_vnni(
    v_base: UnsafePointer[UInt8, MutAnyOrigin],
    max_seq: Int,
    t_start: Int,
    k_off: Int,
    n_off: Int,
    n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pack V cache into VNNI for V-agg B operand.

    V cache is [dim][pos] u8 per head (transposed). Packs
    V[t_start+k_off:..+64, n_off:n_off+n_cols] into one VNNI tile:
    dst[kg*64 + col*4 + b] = V[t_start + k_off + 4*kg + b, n_off + col].
    The 4 bytes per VNNI group are contiguous (positions contiguous in transposed V).
    """
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = v_base + (n_off + col) * max_seq + t_start + k_off + 4 * kg
            var d = kg * 64 + col * 4
            dst[d] = src[0]
            dst[d + 1] = src[1]
            dst[d + 2] = src[2]
            dst[d + 3] = src[3]
        for col in range(n_cols, TILE_N):
            var d = kg * 64 + col * 4
            dst[d] = 0
            dst[d + 1] = 0
            dst[d + 2] = 0
            dst[d + 3] = 0


# ============================================================================
# Dispatched Kernel
# ============================================================================

def had_attn_groups_amx[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, m: Int, group_start: Int, group_end: Int,
    worker_scratch: Int, _a5: Int,
):
    var ctx = UnsafePointer[HadAttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    comptime B_c = ATTN_B_C

    comptime DATA_HEAD_STRIDE = max_seq * head_dim
    comptime SCALE_HEAD_STRIDE = max_seq * size_of[Float32]()

    # Per-worker scratch (same layout as VNNI kernel).
    comptime RUNNING_O_OFF = 0
    comptime BLOCK_SCORES_OFF = RUNNING_O_OFF + gqa_factor * head_dim * size_of[Float32]()
    comptime QI_QS_OFF = BLOCK_SCORES_OFF + gqa_factor * B_c * size_of[Float32]()

    var actual_pos = ctx[].pos + m
    var context = actual_pos + 1
    var cos_row = ctx[].cos + actual_pos * half
    var sin_row = ctx[].sin + actual_pos * half

    var running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + RUNNING_O_OFF)
    var block_scores = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + BLOCK_SCORES_OFF)
    var qi_qs = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=worker_scratch + QI_QS_OFF)
    # Alias block_scores as i32 for tilestore (same memory, same element size)
    var score_i32 = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=worker_scratch + BLOCK_SCORES_OFF)

    # Configure AMX tiles (per-thread, ~100 cycles, negligible)
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    # Stack buffers for AMX tile operations
    var vnni_arr = InlineArray[UInt8, 2 * TILE_BYTES](fill=UInt8(0))
    var vnni_buf = UnsafePointer(to=vnni_arr).bitcast[UInt8]()
    var w_matrix_arr = InlineArray[Scalar[DType.int8], gqa_factor * B_c](fill=Scalar[DType.int8](0))
    var w_matrix = UnsafePointer(to=w_matrix_arr).bitcast[Scalar[DType.int8]]()
    var vagg_arr = InlineArray[Int32, TILE_M * N_STEP](fill=Int32(0))
    var vagg_buf = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    var q_scales_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
    var q_biases_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
    var q_scales = UnsafePointer(to=q_scales_arr).bitcast[Float32]()
    var q_biases = UnsafePointer(to=q_biases_arr).bitcast[Float32]()

    for g in range(group_start, group_end):

        # === 1. Prep all Q heads in this GQA group (identical to VNNI) ===
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
            var q_vinv = SIMD[DType.float32, width](q_inv)

            var qi_qp = qi_qs + hi * head_dim
            var q_sum_acc = SIMD[DType.int32, width](0)
            k = 0
            while k + width <= head_dim:
                var qi = quantize_i8((qp + k).load[width=width](), q_vinv)
                (qi_qp + k).store(qi)
                q_sum_acc += qi.cast[DType.int32]()
                k += width

            q_scales[hi] = (q_absmax / Float32(127.0)) * inv_sqrt_hd
            q_biases[hi] = Float32(128 * Int(q_sum_acc.reduce_add()))

        # === 2. Blocked score + online softmax + AMX V agg ===
        var running_m_arr = InlineArray[Float32, gqa_factor](fill=Float32(-1e30))
        var running_l_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
        var running_m = UnsafePointer(to=running_m_arr).bitcast[Float32]()
        var running_l = UnsafePointer(to=running_l_arr).bitcast[Float32]()
        for i in range(gqa_factor * head_dim):
            running_o[i] = Float32(0)

        var k_base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=ctx[].k_data + g * DATA_HEAD_STRIDE)
        var k_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].k_scale + g * SCALE_HEAD_STRIDE)
        var v_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].v_scale + g * SCALE_HEAD_STRIDE)

        var num_blocks = (context + B_c - 1) // B_c

        for b in range(num_blocks):
            var t_start = b * B_c
            var block_len = min(B_c, context - t_start)

            # --- 2a. AMX Scoring ---
            # S_i32[gqa, block_len] = Q_i8[gqa, hd] x K_u8^T[hd, block_len]
            # A=Q (signed), B=K (unsigned, VNNI-packed) → tdpbsud
            var n_iters = (block_len + N_STEP - 1) // N_STEP
            for n_idx in range(n_iters):
                var n_blk = n_idx * N_STEP
                tilezero[4]()
                tilezero[5]()

                for k_blk in range(0, head_dim, K_STEP):
                    var n_lo = max(0, min(TILE_N, block_len - n_blk))
                    pack_k_vnni(k_base, head_dim, k_blk,
                                t_start + n_blk, n_lo, vnni_buf)
                    var n_hi = max(0, min(TILE_N, block_len - n_blk - TILE_N))
                    pack_k_vnni(k_base, head_dim, k_blk,
                                t_start + n_blk + TILE_N, n_hi, vnni_buf + TILE_BYTES)

                    tileload[0](qi_qs + k_blk, head_dim)
                    tileload[2](vnni_buf, TILE_N * VNNI_BLK)
                    tileload[3](vnni_buf + TILE_BYTES, TILE_N * VNNI_BLK)

                    tdpbsud[4, 0, 2]()
                    tdpbsud[5, 0, 3]()

                tilestore[4](score_i32 + n_blk, B_c * size_of[Int32]())
                tilestore[5](score_i32 + n_blk + TILE_N, B_c * size_of[Int32]())

            # --- 2b. Dequantize scores i32 → f32 ---
            for hi in range(gqa_factor):
                var vbias = SIMD[DType.float32, width](q_biases[hi])
                var vscale = SIMD[DType.float32, width](q_scales[hi])
                var base = hi * B_c
                var tl = 0
                while tl + width <= block_len:
                    var raw = (score_i32 + base + tl).load[width=width]().cast[DType.float32]()
                    var ksc = (k_sc_base + t_start + tl).load[width=width]()
                    (block_scores + base + tl).store((raw - vbias) * vscale * ksc)
                    tl += width
                while tl < block_len:
                    block_scores[base + tl] = (
                        Float32(score_i32[base + tl]) - q_biases[hi]
                    ) * q_scales[hi] * k_sc_base[t_start + tl]
                    tl += 1

            # --- 2c. Online softmax + quantize weights to w_matrix ---
            var w_scales_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
            var w_biases_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
            var w_scales = UnsafePointer(to=w_scales_arr).bitcast[Float32]()
            var w_biases = UnsafePointer(to=w_biases_arr).bitcast[Float32]()

            for hi in range(gqa_factor):
                var scores = block_scores + hi * B_c

                # Block max
                var smax_v = SIMD[DType.float32, width](Float32(-1e30))
                var tl = 0
                while tl + width <= block_len:
                    smax_v = max(smax_v, (scores + tl).load[width=width]())
                    tl += width
                var block_max = smax_v.reduce_max()
                while tl < block_len:
                    if scores[tl] > block_max:
                        block_max = scores[tl]
                    tl += 1

                # Rescale running accumulators
                var m_old = running_m[hi]
                var m_new = max(m_old, block_max)
                var correction = exp_f32[1](m_old - m_new)

                var op = running_o + hi * head_dim
                var vcorr = SIMD[DType.float32, width](correction)
                var d = 0
                while d + width <= head_dim:
                    (op + d).store((op + d).load[width=width]() * vcorr)
                    d += width

                running_l[hi] = running_l[hi] * correction
                running_m[hi] = m_new

                # Exp + absorb V scale
                var vm_new = SIMD[DType.float32, width](m_new)
                var l_acc_v = SIMD[DType.float32, width](0)
                tl = 0
                while tl + width <= block_len:
                    var ev = exp_f32((scores + tl).load[width=width]() - vm_new)
                    l_acc_v += ev
                    var vs = (v_sc_base + t_start + tl).load[width=width]()
                    (scores + tl).store(ev * vs)
                    tl += width
                var l_contrib = l_acc_v.reduce_add()
                while tl < block_len:
                    var e = exp_f32[1](scores[tl] - m_new)
                    l_contrib += e
                    scores[tl] = e * v_sc_base[t_start + tl]
                    tl += 1
                running_l[hi] += l_contrib

                # Zero padding for clean VNNI
                for tl in range(block_len, B_c):
                    scores[tl] = Float32(0)

                # Quantize attention weights into w_matrix row
                var wmax_v = SIMD[DType.float32, width](0)
                tl = 0
                while tl + width <= B_c:
                    wmax_v = max(wmax_v, (scores + tl).load[width=width]())
                    tl += width
                var w_absmax = wmax_v.reduce_max()

                var w_inv = Float32(127.0) / w_absmax if w_absmax > 0 else Float32(0)
                var vw_inv = SIMD[DType.float32, width](w_inv)
                var w_row = w_matrix + hi * B_c
                var w_sum_v = SIMD[DType.int32, width](0)
                tl = 0
                while tl + width <= B_c:
                    var qi = quantize_i8((scores + tl).load[width=width](), vw_inv)
                    (w_row + tl).store(qi)
                    w_sum_v += qi.cast[DType.int32]()
                    tl += width
                var w_i8_sum = Int(w_sum_v.reduce_add())

                var w_scale = w_absmax / Float32(127.0) if w_absmax > 0 else Float32(0)
                w_scales[hi] = w_scale
                w_biases[hi] = Float32(128 * w_i8_sum) * w_scale

            # --- 2d. AMX V aggregation ---
            # O_i32[gqa, hd] = W_i8[gqa, bc] x V_u8[bc, hd]
            # A=W (signed), B=V (unsigned, VNNI-packed) → tdpbsud
            var v_head_base = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=ctx[].v_data + g * DATA_HEAD_STRIDE)

            for n_blk in range(0, head_dim, N_STEP):
                tilezero[4]()
                tilezero[5]()

                for k_blk in range(0, B_c, K_STEP):
                    pack_v_vnni(v_head_base, max_seq, t_start, k_blk,
                                n_blk, TILE_N, vnni_buf)
                    pack_v_vnni(v_head_base, max_seq, t_start, k_blk,
                                n_blk + TILE_N, TILE_N, vnni_buf + TILE_BYTES)

                    tileload[0](w_matrix + k_blk, B_c)
                    tileload[2](vnni_buf, TILE_N * VNNI_BLK)
                    tileload[3](vnni_buf + TILE_BYTES, TILE_N * VNNI_BLK)

                    tdpbsud[4, 0, 2]()
                    tdpbsud[5, 0, 3]()

                # Store tile output and dequant+accumulate to running_o
                tilestore[4](vagg_buf, N_STEP * size_of[Int32]())
                tilestore[5](vagg_buf + TILE_N, N_STEP * size_of[Int32]())

                for hi in range(gqa_factor):
                    var op = running_o + hi * head_dim + n_blk
                    var vp = vagg_buf + hi * N_STEP
                    var ws = SIMD[DType.float32, width](w_scales[hi])
                    var wb = SIMD[DType.float32, width](w_biases[hi])
                    d = 0
                    while d + width <= N_STEP:
                        var raw = (vp + d).load[width=width]().cast[DType.float32]()
                        var existing = (op + d).load[width=width]()
                        (op + d).store(existing + raw * ws - wb)
                        d += width

        # === 3. Final normalization: O /= l, write to shared row_f32 ===
        for hi in range(gqa_factor):
            var h = g * gqa_factor + hi
            var inv_l = Float32(1.0) / running_l[hi]
            var vinv_l = SIMD[DType.float32, width](inv_l)
            var op = running_o + hi * head_dim
            var d = 0
            while d + width <= head_dim:
                (ctx[].row_f32 + h * head_dim + d).store(
                    (op + d).load[width=width]() * vinv_l)
                d += width


# ============================================================================
# Public API
# ============================================================================

def attn_scratch_bytes_amx[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                           num_workers: Int = 1]() -> Int:
    """Scratch size for int8_gqa_attention_amx (same layout as VNNI kernel)."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime per_worker = (
        gqa_factor * head_dim * size_of[Float32]()
        + gqa_factor * ATTN_B_C * size_of[Float32]()
        + gqa_factor * head_dim
    )
    return q_cols * size_of[Float32]() + num_workers * per_worker


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
    """AMX int8 GQA attention parallelized over KV groups.

    Drop-in replacement for int8_gqa_attention.
    Caller must provide scratch >= attn_scratch_bytes_amx[..., num_workers]().
    Caller must have called init_intel_amx() once per process.
    """
    comptime assert QT.DTYPE == DType.bfloat16, "Q must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "scale output must be f32"

    comptime width = simd_width_of[DType.float32]()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS

    comptime PER_WORKER = (
        gqa_factor * head_dim * size_of[Float32]()
        + gqa_factor * ATTN_B_C * size_of[Float32]()
        + gqa_factor * head_dim
    )
    comptime ROW_F32_OFF = 0
    comptime WORKERS_OFF = q_cols * size_of[Float32]()

    var row_f32 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch + ROW_F32_OFF)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_out.ptr)

    var ctx = HadAttnCtx(
        q=UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr),
        qi=qi_ptr,
        sc=sc_ptr,
        cos=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        sin=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        k_data=k_cache.data_base,
        k_scale=k_cache.scale_base,
        v_data=v_cache.data_base,
        v_scale=v_cache.scale_base,
        row_f32=row_f32,
        pos=pos,
        seq_len=q.seq_len,
    )

    var ctx_ptr = UnsafePointer(to=ctx)
    var num_jobs = min(num_kv_heads, pool.capacity)
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

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

        pool.dispatch(
            had_attn_groups_amx[num_heads, num_kv_heads, head_dim, max_seq],
            pool.args_base, num_jobs,
        )
        pool.join()

        # Output quantization (after all workers complete for this row)
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

    return PoolFence.completed()
