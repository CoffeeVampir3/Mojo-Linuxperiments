"""Correctness test for both AMX kernels against scalar f32 reference."""

from std.sys.info import simd_width_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental.hadquant_attn_amx import int8_gqa_attention_amx, attn_scratch_bytes_amx
from experimental.hadquant_attn_amx_prefill import int8_gqa_attention_amx_prefill, attn_scratch_bytes_amx_prefill
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32
from threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from numa.arena import NumaArena
from kernels.kernel_ops import init_rope_tables


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    comptime HD = 64
    comptime NH = 9
    comptime NKV = 3
    comptime GQA = NH // NKV
    comptime HIDDEN = NH * HD
    comptime HALF = HD // 2
    comptime MAX_SEQ = 1024
    comptime SL = 8
    comptime POS = 0

    print("=== Correctness: 9h/3kv, hd=64, prefill=4, pos=0 ===")

    var numa = NumaInfo()
    var local_node = get_current_cpu_and_node()[1]
    var arena = NumaArena(local_node, 8 * 1024 * 1024)

    comptime KVC = HadQuantKVCache[MAX_SEQ, HD, NKV]
    comptime DEC_SCRATCH = attn_scratch_bytes_amx[NH, NKV, HD, NKV]()
    comptime PRE_SCRATCH = attn_scratch_bytes_amx_prefill[NH, NKV, HD, SL, NKV]()

    var q_bf16 = arena.alloc[Scalar[DType.bfloat16]](SL * HIDDEN)
    var kv_mem = arena.alloc[UInt8](2 * KVC.TOTAL_BYTES)
    var k_cache = KVC(Int(kv_mem))
    var v_cache = KVC(Int(kv_mem) + KVC.TOTAL_BYTES)
    var qi_dec = arena.alloc[Scalar[DType.int8]](SL * HIDDEN)
    var sc_dec = arena.alloc[Float32](SL)
    var qi_pre = arena.alloc[Scalar[DType.int8]](SL * HIDDEN)
    var sc_pre = arena.alloc[Float32](SL)
    var cos_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    var dec_scratch = arena.alloc[UInt8](DEC_SCRATCH)
    var pre_scratch = arena.alloc[UInt8](PRE_SCRATCH)
    var expected = arena.alloc[Float32](SL * HIDDEN)
    var q_head = arena.alloc[Float32](HD)
    var head_buf = arena.alloc[Scalar[DType.int8]](HD)
    var burst = BurstPool[].for_numa_node(numa, 0)

    # Fill Q
    for m in range(SL):
        for k in range(HIDDEN):
            q_bf16[m * HIDDEN + k] = Scalar[DType.bfloat16](
                Float32(m * HIDDEN + k + 1) / Float32(SL * HIDDEN) - 0.5)

    # Fill KV
    for t in range(SL):
        for g in range(NKV):
            var k_absmax = Float32(0)
            for d in range(HD):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > k_absmax: k_absmax = a
            k_cache.write_head(t, g, head_buf, k_absmax / Float32(127.0))
            var v_absmax = Float32(0)
            for d in range(HD):
                var val = Float32((t * 11 + g * 17 + d * 5) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > v_absmax: v_absmax = a
            v_cache.write_head_transposed(t, g, head_buf, v_absmax / Float32(127.0))

    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)), Float64(100000.0))

    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]
    comptime QiSlot = Slot[I8, Replicated, 1, HIDDEN, 1]
    comptime ScSlot = Slot[F32, Replicated, 1, 1, 1]

    # --- F32 reference ---
    var inv_sqrt_hd = Float32(1.0) / sqrt[DType.float32, 1](Float32(HD))
    for m in range(SL):
        var actual_pos = POS + m
        var context = actual_pos + 1
        for h in range(NH):
            var g = h // GQA
            for d in range(HD):
                q_head[d] = Float32(q_bf16[m * HIDDEN + h * HD + d])
            for j in range(HALF):
                var x_lo = q_head[j]
                var x_hi = q_head[HALF + j]
                var cv = cos_tab[actual_pos * HALF + j]
                var sv = sin_tab[actual_pos * HALF + j]
                q_head[j] = x_lo * cv - x_hi * sv
                q_head[HALF + j] = x_hi * cv + x_lo * sv
            fwht_block[DType.float32, HD](q_head)

            var scores_mark = arena.mark()
            var scores = arena.alloc[Float32](context)
            for t in range(context):
                var dot = Float32(0)
                var k_sc = k_cache.head_scale(t, g)
                var k_data = k_cache.head_data(t, g)
                for d in range(HD):
                    dot += q_head[d] * Float32(Int(k_data[d]) - 128) * k_sc
                scores[t] = dot * inv_sqrt_hd
            var smax = scores[0]
            for t in range(1, context):
                if scores[t] > smax: smax = scores[t]
            var exp_sum = Float32(0)
            for t in range(context):
                scores[t] = exp_f32[1](scores[t] - smax)
                exp_sum += scores[t]
            var inv_sum = Float32(1.0) / exp_sum
            for t in range(context):
                scores[t] = scores[t] * inv_sum
            for d in range(HD):
                var acc = Float32(0)
                var v_dim = v_cache.dim_data(d, g)
                for t in range(context):
                    acc += scores[t] * Float32(Int(v_dim[t]) - 128) * v_cache.head_scale(t, g)
                expected[m * HIDDEN + h * HD + d] = acc
            arena.reset_to(scores_mark)

    # --- Run decode kernel ---
    int8_gqa_attention_amx[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), SL), k_cache, v_cache,
        DynView[QiSlot](Int(qi_dec), SL), DynView[ScSlot](Int(sc_dec), SL),
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(dec_scratch), POS, burst,
    ).join()

    # --- Run prefill kernel ---
    int8_gqa_attention_amx_prefill[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), SL), k_cache, v_cache,
        DynView[QiSlot](Int(qi_pre), SL), DynView[ScSlot](Int(sc_pre), SL),
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(pre_scratch), POS, burst,
    ).join()

    # --- Compare decode vs reference ---
    var dec_max_err = Float64(0)
    var dec_sum_err = Float64(0)
    var count = 0
    for m in range(SL):
        var out_scale = Float64(sc_dec[m])
        for d in range(HIDDEN):
            var got = Float64(qi_dec[m * HIDDEN + d]) * out_scale
            var exp_val = Float64(expected[m * HIDDEN + d])
            var err = got - exp_val
            if err < 0: err = -err
            if err > dec_max_err: dec_max_err = err
            dec_sum_err += err
            count += 1
    var dec_avg = dec_sum_err / Float64(count)
    print("Decode:  max_err=" + String(dec_max_err) + " avg_err=" + String(dec_avg))
    if dec_avg < 0.05:
        print("  PASS")
    else:
        print("  FAIL")

    # --- Compare prefill vs reference ---
    var pre_max_err = Float64(0)
    var pre_sum_err = Float64(0)
    count = 0
    for m in range(SL):
        var out_scale = Float64(sc_pre[m])
        for d in range(HIDDEN):
            var got = Float64(qi_pre[m * HIDDEN + d]) * out_scale
            var exp_val = Float64(expected[m * HIDDEN + d])
            var err = got - exp_val
            if err < 0: err = -err
            if err > pre_max_err: pre_max_err = err
            pre_sum_err += err
            count += 1
    var pre_avg = pre_sum_err / Float64(count)
    # Per-position error
    for m in range(SL):
        var pos_err = Float64(0)
        var pos_scale = Float64(sc_pre[m])
        for d in range(HIDDEN):
            var got = Float64(qi_pre[m * HIDDEN + d]) * pos_scale
            var exp_val = Float64(expected[m * HIDDEN + d])
            var e = got - exp_val
            if e < 0: e = -e
            pos_err += e
        print("  pos " + String(m) + " avg_err=" + String(pos_err / Float64(HIDDEN)))

    # Detailed look at high-error positions
    for m in range(min(SL, 32)):
        if True:
            var ps = Float64(sc_pre[m])
            var ds = Float64(sc_dec[m])
            print("  pos " + String(m) + " samples:")
            for d in range(min(4, HIDDEN)):
                var pv = Float64(qi_pre[m * HIDDEN + d]) * ps
                var dv = Float64(qi_dec[m * HIDDEN + d]) * ds
                var rv = Float64(expected[m * HIDDEN + d])
                print("    [" + String(d) + "] pre=" + String(pv) + " dec=" + String(dv) + " ref=" + String(rv))

    print("Prefill: max_err=" + String(pre_max_err) + " avg_err=" + String(pre_avg))
    if pre_avg < 0.05:
        print("  PASS")
    else:
        print("  FAIL")
        for d in range(min(8, HIDDEN)):
            var got_d = Float64(qi_pre[d]) * Float64(sc_pre[0])
            var got_p = Float64(qi_dec[d]) * Float64(sc_dec[0])
            var exp_val = Float64(expected[d])
            print("  [0," + String(d) + "] prefill=" + String(got_d)
                  + " decode=" + String(got_p) + " ref=" + String(exp_val))

    # --- Compare decode vs prefill directly ---
    var dp_max_err = Float64(0)
    var dp_sum_err = Float64(0)
    count = 0
    for m in range(SL):
        var ds = Float64(sc_dec[m])
        var ps = Float64(sc_pre[m])
        for d in range(HIDDEN):
            var dv = Float64(qi_dec[m * HIDDEN + d]) * ds
            var pv = Float64(qi_pre[m * HIDDEN + d]) * ps
            var err = dv - pv
            if err < 0: err = -err
            if err > dp_max_err: dp_max_err = err
            dp_sum_err += err
            count += 1
    print("Decode vs Prefill: max_err=" + String(dp_max_err)
          + " avg_err=" + String(dp_sum_err / Float64(count)))
