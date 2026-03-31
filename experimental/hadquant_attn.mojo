"""Int8 GQA attention with blocked online softmax and VNNI V aggregation.

Dispatch: KV groups are parallelized via BurstPool. Each job processes a
range of groups for one query row, using its own scratch region, writing
to disjoint slices of the shared row_f32 buffer. Output quantization
runs on the caller after join.

Scratch layout:
  [0]                              row_f32       q_cols * 4      (shared)
  [row_f32_size + 0*PER_WORKER]    worker 0 scratch
  [row_f32_size + 1*PER_WORKER]    worker 1 scratch
  ...
  Per-worker scratch contains: running_o, block_scores, qi_qs.
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
from experimental.hadquant_impl import fwht_block, i8_dot_row_major
from experimental.hadquant_kv_cache import HadQuantKVCache


comptime ATTN_B_C = 64


def attn_scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                       num_workers: Int = 1]() -> Int:
    """Scratch size required by int8_gqa_attention with num_workers parallel jobs."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime per_worker = (
        gqa_factor * head_dim * size_of[Float32]()      # running_o
        + gqa_factor * ATTN_B_C * size_of[Float32]()    # block_scores
        + gqa_factor * head_dim                          # qi_qs (i8)
    )
    return q_cols * size_of[Float32]() + num_workers * per_worker


# ---------------------------------------------------------------------------
# Context struct — passed by pointer through ArgPack to dispatched workers.
# Shared across all workers for a given call. Worker-local scratch is
# passed separately via arg4.
# ---------------------------------------------------------------------------

@fieldwise_init
struct HadAttnCtx:
    var q: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
    var sc: UnsafePointer[Float32, MutAnyOrigin]
    var cos: UnsafePointer[Float32, MutAnyOrigin]
    var sin: UnsafePointer[Float32, MutAnyOrigin]
    var k_data: Int
    var k_scale: Int
    var v_data: Int
    var v_scale: Int
    var row_f32: UnsafePointer[Float32, MutAnyOrigin]
    var pos: Int
    var seq_len: Int


# ---------------------------------------------------------------------------
# Dispatched kernel — processes KV groups [group_start, group_end) for row m.
# Each worker gets its own scratch region via worker_scratch.
# ---------------------------------------------------------------------------

def had_attn_groups[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
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

    # Per-worker scratch offsets.
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

    var q_scales_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
    var q_biases_arr = InlineArray[Float32, gqa_factor](fill=Float32(0))
    var q_scales = UnsafePointer(to=q_scales_arr).bitcast[Float32]()
    var q_biases = UnsafePointer(to=q_biases_arr).bitcast[Float32]()

    for g in range(group_start, group_end):

        # --- 1. Prep all Q heads in this GQA group ---
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

        # --- 2. Blocked score + online softmax + VNNI V agg ---
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

            for tl in range(block_len):
                var t = t_start + tl
                var k_entry = k_base + t * head_dim
                var k_sc = k_sc_base[t]
                for hi in range(gqa_factor):
                    var qi_qp = qi_qs + hi * head_dim
                    var raw = i8_dot_row_major[head_dim](k_entry, qi_qp)
                    block_scores[hi * B_c + tl] = (Float32(raw) - q_biases[hi]) * q_scales[hi] * k_sc

            for hi in range(gqa_factor):
                var scores = block_scores + hi * B_c

                var smax_v = SIMD[DType.float32, width](Float32(-1e30))
                var tl = 0
                while tl + width <= block_len:
                    smax_v = max(smax_v, (scores + tl).load[width=width]())
                    tl += width
                var block_max = smax_v.reduce_max()
                while tl < block_len:
                    if scores[tl] > block_max: block_max = scores[tl]
                    tl += 1

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

                for tl in range(block_len, B_c):
                    scores[tl] = Float32(0)

                var wmax_v = SIMD[DType.float32, width](0)
                tl = 0
                while tl + width <= B_c:
                    wmax_v = max(wmax_v, (scores + tl).load[width=width]())
                    tl += width
                var w_absmax = wmax_v.reduce_max()

                var w_inv = Float32(127.0) / w_absmax if w_absmax > 0 else Float32(0)
                var vw_inv = SIMD[DType.float32, width](w_inv)
                var w_block_arr = InlineArray[Scalar[DType.int8], B_c](fill=Scalar[DType.int8](0))
                var w_block = UnsafePointer(to=w_block_arr).bitcast[Scalar[DType.int8]]()
                var w_sum_v = SIMD[DType.int32, width](0)
                tl = 0
                while tl + width <= B_c:
                    var qi = quantize_i8((scores + tl).load[width=width](), vw_inv)
                    (w_block + tl).store(qi)
                    w_sum_v += qi.cast[DType.int32]()
                    tl += width
                var w_i8_sum = Int(w_sum_v.reduce_add())

                var w_scale = w_absmax / Float32(127.0) if w_absmax > 0 else Float32(0)
                var bias = Float32(128 * w_i8_sum) * w_scale
                var v_head_base = ctx[].v_data + g * DATA_HEAD_STRIDE
                for d in range(head_dim):
                    var v_dim = UnsafePointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=v_head_base + d * max_seq + t_start)
                    var raw = i8_dot_row_major[B_c](v_dim, w_block)
                    op[d] += Float32(raw) * w_scale - bias

        # --- Final normalization: O /= l, write to shared row_f32 ---
        for hi in range(gqa_factor):
            var h = g * gqa_factor + hi
            var inv_l = Float32(1.0) / running_l[hi]
            var vinv_l = SIMD[DType.float32, width](inv_l)
            var op = running_o + hi * head_dim
            d = 0
            while d + width <= head_dim:
                (ctx[].row_f32 + h * head_dim + d).store((op + d).load[width=width]() * vinv_l)
                d += width


# ---------------------------------------------------------------------------
# Public API — dispatches groups via BurstPool, quantizes output after join.
# ---------------------------------------------------------------------------

def int8_gqa_attention[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
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
    """Int8 GQA attention parallelized over KV groups via BurstPool.

    Caller must provide scratch >= attn_scratch_bytes[..., num_workers]().
    """
    comptime assert QT.DTYPE == DType.bfloat16, "int8_gqa_attention: Q must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "int8_gqa_attention: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gqa_attention: scale output must be f32"

    comptime width = simd_width_of[DType.float32]()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS

    # Per-worker scratch size (must match had_attn_groups layout).
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
        # Fill one ArgPack per job with its group range and scratch pointer.
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
            had_attn_groups[num_heads, num_kv_heads, head_dim, max_seq],
            pool.args_base, num_jobs,
        )
        pool.join()

        # Output quantization (after all workers complete for this row).
        var out_row = qi_ptr + m * q_cols
        var rmax_v = SIMD[DType.float32, width](0)
        var d = 0
        while d + width <= q_cols:
            rmax_v = max(rmax_v, (row_f32 + d).load[width=width]().__abs__())
            d += width
        var row_absmax = rmax_v.reduce_max()
        while d < q_cols:
            var a = row_f32[d] if row_f32[d] >= 0 else -row_f32[d]
            if a > row_absmax: row_absmax = a
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
