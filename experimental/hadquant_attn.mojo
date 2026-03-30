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
from experimental.hadquant_impl import fwht_block, i8_dot_row_major, i8_sum
from experimental.hadquant_kv_cache import HadQuantKVCache


struct AttnProfile:
    """Accumulates per-section timing across calls. Reports averages."""
    var q_prep: Int
    var score: Int
    var softmax: Int
    var v_absorb: Int
    var w_quant: Int
    var v_agg: Int
    var out_quant: Int
    var calls: Int

    def __init__(out self):
        self.q_prep = 0
        self.score = 0
        self.softmax = 0
        self.v_absorb = 0
        self.w_quant = 0
        self.v_agg = 0
        self.out_quant = 0
        self.calls = 0

    def report(self):
        if self.calls == 0:
            return
        var total = self.q_prep + self.score + self.softmax + self.v_absorb + self.w_quant + self.v_agg + self.out_quant
        var n = self.calls
        var pct = def (v: Int) -> Int:
            return v * 100 // total if total > 0 else 0
        print("attn profile (avg over " + String(n) + " calls):")
        print("  q_prep:    " + String(self.q_prep // n // 1000) + " us (" + String(pct(self.q_prep)) + "%)")
        print("  score:     " + String(self.score // n // 1000) + " us (" + String(pct(self.score)) + "%)")
        print("  softmax:   " + String(self.softmax // n // 1000) + " us (" + String(pct(self.softmax)) + "%)")
        print("  v_absorb:  " + String(self.v_absorb // n // 1000) + " us (" + String(pct(self.v_absorb)) + "%)")
        print("  w_quant:   " + String(self.w_quant // n // 1000) + " us (" + String(pct(self.w_quant)) + "%)")
        print("  v_agg:     " + String(self.v_agg // n // 1000) + " us (" + String(pct(self.v_agg)) + "%)")
        print("  out_quant: " + String(self.out_quant // n // 1000) + " us (" + String(pct(self.out_quant)) + "%)")
        print("  total:     " + String(total // n // 1000) + " us")


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

    var t_q_prep = Int(0)
    var t_score = Int(0)
    var t_softmax = Int(0)
    var t_v_absorb = Int(0)
    var t_w_quant = Int(0)
    var t_v_agg = Int(0)
    var t_out_quant = Int(0)

    for m in range(q.seq_len):
        var actual_pos = pos + m
        var context = actual_pos + 1  # causal: attend to 0..actual_pos
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half
        var out_row = qi_ptr + m * q_cols

        # Full-row f32 buffer for aggregation before per-row quantization
        var row_f32 = alloc[Float32](q_cols)

        # Per-group buffers for gqa_factor Q heads
        var qi_qs = alloc[Scalar[DType.int8]](gqa_factor * head_dim)
        var q_scales = alloc[Float32](gqa_factor)
        var q_biases = alloc[Float32](gqa_factor)

        # Per-head scores and weights (reused across heads in group)
        var per_head_scores = alloc[Float32](gqa_factor * max_ctx)

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
                q_scales[hi] = q_absmax / Float32(127.0)
                var q_inv = Float32(127.0) / q_absmax if q_absmax > 0 else Float32(0)
                var q_vinv = SIMD[DType.float32, width](q_inv)

                var qi_qp = qi_qs + hi * head_dim
                k = 0
                while k + width <= head_dim:
                    (qi_qp + k).store(quantize_i8((qp + k).load[width=width](), q_vinv))
                    k += width

                q_biases[hi] = Float32(128 * i8_sum[head_dim](qi_qp))

            var t1 = Int(perf_counter_ns())
            t_q_prep += t1 - t0

            # --- 2. Score all heads in group: load K[t,g] once per entry ---
            for t in range(context):
                var k_entry = k_cache.head_data(t, g)  # loaded ONCE
                var k_sc = k_cache.head_scale(t, g)

                for hi in range(gqa_factor):
                    var qi_qp = qi_qs + hi * head_dim
                    var raw = i8_dot_row_major[head_dim](k_entry, qi_qp)
                    (per_head_scores + hi * max_ctx)[t] = (Float32(raw) - q_biases[hi]) * q_scales[hi] * inv_sqrt_hd * k_sc

            # --- 3. Softmax per head (SIMD over context) ---
            for hi in range(gqa_factor):
                var head_scores = per_head_scores + hi * max_ctx

                # Max
                var smax_v = SIMD[DType.float32, width](Float32(-1e30))
                var t = 0
                while t + width <= context:
                    smax_v = max(smax_v, (head_scores + t).load[width=width]())
                    t += width
                var smax = smax_v.reduce_max()
                while t < context:
                    if head_scores[t] > smax: smax = head_scores[t]
                    t += 1

                # Exp + sum
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

                # Normalize
                var vinv_sum = SIMD[DType.float32, width](Float32(1.0) / exp_sum)
                t = 0
                while t + width <= context:
                    (head_scores + t).store((head_scores + t).load[width=width]() * vinv_sum)
                    t += width
                var inv_sum = Float32(1.0) / exp_sum
                while t < context:
                    head_scores[t] = head_scores[t] * inv_sum
                    t += 1

            var t2 = Int(perf_counter_ns())
            t_score += t2 - t1

            # --- 4. Per-head: V scale absorption into softmax weights ---
            for hi in range(gqa_factor):
                var head_scores = per_head_scores + hi * max_ctx
                for t in range(context):
                    head_scores[t] = head_scores[t] * v_cache.head_scale(t, g)

            var t3 = Int(perf_counter_ns())
            t_softmax += t3 - t2
            t_v_absorb = 0
            t_w_quant = 0

            # --- 5. Aggregate V: f32 weights × u8 V, load V[t,g] once ---
            # Accumulate w[t] * V_u8[t,d] in raw u8 domain, then correct:
            #   true[d] = raw[d] - 128 * sum(w[t])
            # The -128 is one scalar per head, applied after the loop.
            var aggs = alloc[Float32](gqa_factor * head_dim)
            var w_sums = InlineArray[Float32, gqa_factor](fill=Float32(0))
            for i in range(gqa_factor * head_dim):
                aggs[i] = Float32(0)

            for t in range(context):
                var v_head = v_cache.head_data(t, g)
                for hi in range(gqa_factor):
                    var wt_val = (per_head_scores + hi * max_ctx)[t]
                    w_sums[hi] += wt_val
                    var wt = SIMD[DType.float32, width](wt_val)
                    var aggp = aggs + hi * head_dim
                    var d = 0
                    while d + width <= head_dim:
                        var vv = (v_head + d).load[width=width]().cast[DType.float32]()
                        (aggp + d).store((aggp + d).load[width=width]() + wt * vv)
                        d += width

            # Bias correction + write to f32 row buffer
            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                var aggp = aggs + hi * head_dim
                var bias = SIMD[DType.float32, width](Float32(128.0) * w_sums[hi])
                var d = 0
                while d + width <= head_dim:
                    (row_f32 + h * head_dim + d).store((aggp + d).load[width=width]() - bias)
                    d += width

            aggs.free()
            t_v_agg += Int(perf_counter_ns()) - t3

        var t6 = Int(perf_counter_ns())
        # --- 7. Quantize full row → i8 (SIMD along q_cols) ---
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

        var t7 = Int(perf_counter_ns())
        t_out_quant += t7 - t6

        row_f32.free()
        qi_qs.free()
        q_scales.free()
        q_biases.free()
        per_head_scores.free()

    var total = t_q_prep + t_score + t_softmax + t_v_absorb + t_w_quant + t_v_agg + t_out_quant
    if total > 0:
        print("attn profile (" + String(q.seq_len) + " rows, " + String(num_heads)
              + " heads, ctx=" + String(max_ctx) + "):")
        print("  q_prep:    " + String(t_q_prep // 1000) + " us (" + String(t_q_prep * 100 // total) + "%)")
        print("  score:     " + String(t_score // 1000) + " us (" + String(t_score * 100 // total) + "%)")
        print("  softmax:   " + String(t_softmax // 1000) + " us (" + String(t_softmax * 100 // total) + "%)")
        print("  v_absorb:  " + String(t_v_absorb // 1000) + " us (" + String(t_v_absorb * 100 // total) + "%)")
        print("  w_quant:   " + String(t_w_quant // 1000) + " us (" + String(t_w_quant * 100 // total) + "%)")
        print("  v_agg:     " + String(t_v_agg // 1000) + " us (" + String(t_v_agg * 100 // total) + "%)")
        print("  out_quant: " + String(t_out_quant // 1000) + " us (" + String(t_out_quant * 100 // total) + "%)")
        print("  total:     " + String(total // 1000) + " us")

    return PoolFence.completed()
