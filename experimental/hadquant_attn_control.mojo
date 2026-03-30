"""Int8 GQA attention with fused RoPE on Q, pre-rotated u8 KV cache,
V scale absorption, and direct int8 output.

KV cache stores u8 (i8 XOR'd at write time) for native vpdpbusd. Q and
attention weights are i8 (signed operand). Bias correction is one scalar
per head: 128 * sum(signed_operand), computed once, not per cache entry.

Per query row m, per head h (kv group g = h / GQA_FACTOR):
  1. Load Q head bf16 → f32, RoPE, FWHT, quantize → qi_q (i8) + q_scale
  2. Score: dot(K_u8[t,g], qi_q_i8) via vpdpbusd, bias correct with sum(qi_q)
  3. Online softmax in f32 (batched SIMD)
  4. Absorb V per-entry scales into softmax weights
  5. Quantize absorbed weights to i8
  6. Aggregate: sum_t w_i8[t] * V_u8[t,g,d] via f32, bias correct with sum(w)
  7. Quantize output to i8 (stays in rotated domain for O projection)
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
    """Int8 GQA attention with fused RoPE on Q and pre-rotated int8 KV cache."""
    comptime assert QT.DTYPE == DType.bfloat16, "int8_gqa_attention: Q must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "int8_gqa_attention: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gqa_attention: scale output must be f32"

    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    var q_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_out.ptr)
    var cp = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr)
    var sn = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr)

    var max_ctx = pos + q.seq_len

    # [4] Hoist allocations out of the m loop.
    var row_f32 = alloc[Float32](q_cols)
    var qi_qs = alloc[Scalar[DType.int8]](gqa_factor * head_dim)
    var q_scales = alloc[Float32](gqa_factor)
    var q_biases = alloc[Float32](gqa_factor)
    var per_head_scores = alloc[Float32](gqa_factor * max_ctx)

    # [7] Wall-clock timer.
    var t_v_agg = Int(0)
    var t_wall = Int(perf_counter_ns())

    for m in range(q.seq_len):
        var actual_pos = pos + m
        var context = actual_pos + 1
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half
        var out_row = qi_ptr + m * q_cols

        for g in range(num_kv_heads):

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

                # [2] Fuse i8_sum into quantization.
                var qi_qp = qi_qs + hi * head_dim
                var q_sum_acc = SIMD[DType.int32, width](0)
                k = 0
                while k + width <= head_dim:
                    var qi = quantize_i8((qp + k).load[width=width](), q_vinv)
                    (qi_qp + k).store(qi)
                    q_sum_acc += qi.cast[DType.int32]()
                    k += width

                # [3] Precompute q_scale * inv_sqrt_hd.
                q_scales[hi] = (q_absmax / Float32(127.0)) * inv_sqrt_hd
                q_biases[hi] = Float32(128 * Int(q_sum_acc.reduce_add()))

            # --- 2. Score all heads in group ---
            for t in range(context):
                var k_entry = k_cache.head_data(t, g)
                var k_sc = k_cache.head_scale(t, g)
                for hi in range(gqa_factor):
                    var qi_qp = qi_qs + hi * head_dim
                    var raw = i8_dot_row_major[head_dim](k_entry, qi_qp)
                    (per_head_scores + hi * max_ctx)[t] = (Float32(raw) - q_biases[hi]) * q_scales[hi] * k_sc

            # --- 3. Softmax + [6] fused V scale absorption ---
            var v_scale_base = v_cache.head_scale_base(g)
            var w_sums = InlineArray[Float32, gqa_factor](fill=Float32(0))
            for hi in range(gqa_factor):
                var head_scores = per_head_scores + hi * max_ctx

                var smax_v = SIMD[DType.float32, width](Float32(-1e30))
                var t = 0
                while t + width <= context:
                    smax_v = max(smax_v, (head_scores + t).load[width=width]())
                    t += width
                var smax = smax_v.reduce_max()
                while t < context:
                    if head_scores[t] > smax: smax = head_scores[t]
                    t += 1

                var exp_sum_v = SIMD[DType.float32, width](0)
                var vsmax = SIMD[DType.float32, width](smax)
                t = 0
                while t + width <= context:
                    var ev = exp_f32((head_scores + t).load[width=width]() - vsmax)
                    (head_scores + t).store(ev)
                    exp_sum_v += ev
                    t += width
                var exp_sum = exp_sum_v.reduce_add()
                while t < context:
                    head_scores[t] = exp_f32[1](head_scores[t] - smax)
                    exp_sum += head_scores[t]
                    t += 1

                # Normalize + absorb V scale + accumulate w_sum
                var vinv_sum = SIMD[DType.float32, width](Float32(1.0) / exp_sum)
                var wsum_v = SIMD[DType.float32, width](0)
                t = 0
                while t + width <= context:
                    var normed = (head_scores + t).load[width=width]() * vinv_sum
                    var absorbed = normed * (v_scale_base + t).load[width=width]()
                    (head_scores + t).store(absorbed)
                    wsum_v += absorbed
                    t += width
                w_sums[hi] = wsum_v.reduce_add()
                var inv_sum = Float32(1.0) / exp_sum
                while t < context:
                    var absorbed = head_scores[t] * inv_sum * v_scale_base[t]
                    head_scores[t] = absorbed
                    w_sums[hi] += absorbed
                    t += 1

            # --- 4. [5] V aggregation — transposed layout, register accumulator ---
            var t_v_start = Int(perf_counter_ns())
            for d in range(head_dim):
                var v_dim = v_cache.dim_data(d, g)
                for hi in range(gqa_factor):
                    var h = g * gqa_factor + hi
                    var w_ptr = per_head_scores + hi * max_ctx
                    var acc_v = SIMD[DType.float32, width](0)
                    var t = 0
                    while t + width <= context:
                        var vv = (v_dim + t).load[width=width]().cast[DType.float32]()
                        acc_v += (w_ptr + t).load[width=width]() * vv
                        t += width
                    var acc = acc_v.reduce_add()
                    while t < context:
                        acc += w_ptr[t] * Float32(Int(v_dim[t]))
                        t += 1
                    row_f32[h * head_dim + d] = acc - Float32(128.0) * w_sums[hi]

            t_v_agg += Int(perf_counter_ns()) - t_v_start

        # --- 5. Quantize full row → i8 ---
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

    var t_total = Int(perf_counter_ns()) - t_wall

    row_f32.free()
    qi_qs.free()
    q_scales.free()
    q_biases.free()
    per_head_scores.free()

    if t_total > 0:
        print("attn wall (" + String(q.seq_len) + " rows, " + String(num_heads)
              + " heads, ctx=" + String(max_ctx) + "): "
              + String(t_total // 1000) + " us"
              + " v_agg=" + String(t_v_agg // 1000))

    return PoolFence.completed()
