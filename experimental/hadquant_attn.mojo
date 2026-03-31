"""Int8 GQA attention with blocked online softmax and VNNI V aggregation.

Based on INT-FlashAttention (Chen et al., 2024): process KV in blocks,
quantize attention weights to i8, use vpdpbusd for P×V aggregation.

KV cache stores u8 (i8 XOR'd at write time) for native vpdpbusd. Q and
attention weights are i8 (signed operand). Bias correction is one scalar
per head: 128 * sum(signed_operand), computed once, not per cache entry.

Per query row m, per head h (kv group g = h / GQA_FACTOR):
  1. Load Q head bf16 -> f32, RoPE, FWHT, quantize -> qi_q (i8) + q_scale
  2. For each block of B_c KV entries:
     a. Score: dot(K_u8[t,g], qi_q_i8) via vpdpbusd
     b. Online softmax: update running max, rescale accumulator
     c. Absorb V scales, quantize weights -> w_i8
     d. Transpose V block -> [dim][pos], VNNI dot w_i8 x V_u8 per dim
  3. Final: normalize by 1/l, quantize output row to i8
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32, quantize_i8, quantize_i8_scalar
from experimental.hadquant_impl import fwht_block, i8_dot_row_major
from experimental.hadquant_kv_cache import HadQuantKVCache


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
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Int8 GQA attention with blocked online softmax and VNNI V aggregation."""
    comptime assert QT.DTYPE == DType.bfloat16, "int8_gqa_attention: Q must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "int8_gqa_attention: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gqa_attention: scale output must be f32"

    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    comptime B_c = 64  # KV block size

    var q_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_out.ptr)
    var cp = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr)
    var sn = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr)

    var t_q_prep = Int(0)
    var t_blocked = Int(0)
    var t_out_quant = Int(0)

    # Hoisted allocations.
    var row_f32 = alloc[Float32](q_cols)
    var qi_qs = alloc[Scalar[DType.int8]](gqa_factor * head_dim)
    var q_scales = alloc[Float32](gqa_factor)
    var q_biases = alloc[Float32](gqa_factor)
    # Blocked processing buffers (all fit in L1).
    var block_scores = alloc[Float32](gqa_factor * B_c)    # 16*64*4 = 4 KB
    var w_block = alloc[Scalar[DType.int8]](B_c)            # 64 B
    var running_o = alloc[Float32](gqa_factor * head_dim)   # 16*128*4 = 8 KB
    var running_m = alloc[Float32](gqa_factor)
    var running_l = alloc[Float32](gqa_factor)

    for m in range(q.seq_len):
        var actual_pos = pos + m
        var context = actual_pos + 1
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half
        var out_row = qi_ptr + m * q_cols

        for g in range(num_kv_heads):
            var t0 = Int(perf_counter_ns())

            # --- 1. Prep all Q heads in this GQA group ---
            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                var q_head = InlineArray[Float32, head_dim](fill=Float32(0))
                var qp = UnsafePointer(to=q_head).bitcast[Float32]()
                var q_row = q_ptr + m * q_cols + h * head_dim
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

            var t1 = Int(perf_counter_ns())
            t_q_prep += t1 - t0

            # --- 2. Blocked score + online softmax + VNNI V agg ---
            # Initialize running online softmax state.
            for hi in range(gqa_factor):
                running_m[hi] = Float32(-1e30)
                running_l[hi] = Float32(0)
            for i in range(gqa_factor * head_dim):
                running_o[i] = Float32(0)

            var k_base = k_cache.head_data_base(g)
            var k_sc_base = k_cache.head_scale_base(g)
            var v_sc_base = v_cache.head_scale_base(g)

            var num_full_blocks = context // B_c
            var remainder = context - num_full_blocks * B_c
            var num_blocks = num_full_blocks + (1 if remainder > 0 else 0)

            for b in range(num_blocks):
                var t_start = b * B_c
                var block_len = min(B_c, context - t_start)

                # --- 2a. Score this block ---
                for tl in range(block_len):
                    var t = t_start + tl
                    var k_entry = k_base + t * head_dim
                    var k_sc = k_sc_base[t]
                    for hi in range(gqa_factor):
                        var qi_qp = qi_qs + hi * head_dim
                        var raw = i8_dot_row_major[head_dim](k_entry, qi_qp)
                        block_scores[hi * B_c + tl] = (Float32(raw) - q_biases[hi]) * q_scales[hi] * k_sc

                # --- 2b. Per head: online softmax + quantize weights + VNNI V agg ---
                for hi in range(gqa_factor):
                    var scores = block_scores + hi * B_c

                    # Block max (SIMD).
                    var smax_v = SIMD[DType.float32, width](Float32(-1e30))
                    var tl = 0
                    while tl + width <= block_len:
                        smax_v = max(smax_v, (scores + tl).load[width=width]())
                        tl += width
                    var block_max = smax_v.reduce_max()
                    while tl < block_len:
                        if scores[tl] > block_max: block_max = scores[tl]
                        tl += 1

                    # Online softmax update.
                    var m_old = running_m[hi]
                    var m_new = max(m_old, block_max)
                    var correction = exp_f32[1](m_old - m_new)

                    # Rescale running output by exp(m_old - m_new).
                    var op = running_o + hi * head_dim
                    var vcorr = SIMD[DType.float32, width](correction)
                    var d = 0
                    while d + width <= head_dim:
                        (op + d).store((op + d).load[width=width]() * vcorr)
                        d += width

                    running_l[hi] = running_l[hi] * correction
                    running_m[hi] = m_new

                    # Exp + l accumulation + V scale absorption (fused).
                    # absorbed[tl] = exp(score - m_new) * v_scale[t], stored in-place.
                    # l tracks sum of exp(score - m_new) WITHOUT v_scale.
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

                    # Zero absorbed weights for padding.
                    for tl in range(block_len, B_c):
                        scores[tl] = Float32(0)

                    # Quantize absorbed weights -> i8 (SIMD).
                    # Weights are non-negative, so absmax = max.
                    var wmax_v = SIMD[DType.float32, width](0)
                    tl = 0
                    while tl + width <= B_c:
                        wmax_v = max(wmax_v, (scores + tl).load[width=width]())
                        tl += width
                    var w_absmax = wmax_v.reduce_max()

                    var w_inv = Float32(127.0) / w_absmax if w_absmax > 0 else Float32(0)
                    var vw_inv = SIMD[DType.float32, width](w_inv)
                    var w_sum_v = SIMD[DType.int32, width](0)
                    tl = 0
                    while tl + width <= B_c:
                        var qi = quantize_i8((scores + tl).load[width=width](), vw_inv)
                        (w_block + tl).store(qi)
                        w_sum_v += qi.cast[DType.int32]()
                        tl += width
                    var w_i8_sum = Int(w_sum_v.reduce_add())

                    # VNNI V aggregation: V stored as [head][dim][pos] (transposed).
                    # For each dim, positions are contiguous -> direct VNNI dot.
                    # raw = sum_t V_u8[t,d] * w_i8[t]
                    # true = (raw - 128 * w_i8_sum) * w_scale
                    var w_scale = w_absmax / Float32(127.0) if w_absmax > 0 else Float32(0)
                    var bias = Float32(128 * w_i8_sum) * w_scale
                    for d in range(head_dim):
                        var v_dim = v_cache.dim_data(d, g) + t_start
                        var raw = i8_dot_row_major[B_c](v_dim, w_block)
                        op[d] += Float32(raw) * w_scale - bias

            # --- Final normalization: O /= l ---
            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                var inv_l = Float32(1.0) / running_l[hi]
                var vinv_l = SIMD[DType.float32, width](inv_l)
                var op = running_o + hi * head_dim
                d = 0
                while d + width <= head_dim:
                    (row_f32 + h * head_dim + d).store((op + d).load[width=width]() * vinv_l)
                    d += width

            t_blocked += Int(perf_counter_ns()) - t1

        var t6 = Int(perf_counter_ns())
        # --- 3. Quantize full row -> i8 (SIMD along q_cols) ---
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

        t_out_quant += Int(perf_counter_ns()) - t6

    row_f32.free()
    qi_qs.free()
    q_scales.free()
    q_biases.free()
    block_scores.free()
    w_block.free()
    running_o.free()
    running_m.free()
    running_l.free()

    var total = t_q_prep + t_blocked + t_out_quant
    if total > 0:
        print("attn profile (" + String(q.seq_len) + " rows, " + String(num_heads)
              + " heads, ctx=" + String(pos + q.seq_len) + "):")
        print("  q_prep:    " + String(t_q_prep // 1000) + " us (" + String(t_q_prep * 100 // total) + "%)")
        print("  blocked:   " + String(t_blocked // 1000) + " us (" + String(t_blocked * 100 // total) + "%)")
        print("  out_quant: " + String(t_out_quant // 1000) + " us (" + String(t_out_quant * 100 // total) + "%)")
        print("  total:     " + String(total // 1000) + " us")

    return PoolFence.completed()
