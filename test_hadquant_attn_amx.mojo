"""Test int8_gqa_attention_amx against scalar f32 reference.

Same structure as test_hadquant_attn.mojo but calls the AMX kernel.
Requires AMX-capable hardware (Sapphire Rapids+).
"""

from std.sys.info import simd_width_of
from std.math import sqrt as std_sqrt
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated, ColShard,
    Slot, Bound, DynView,
)
from experimental.hadquant_attn_amx import int8_gqa_attention_amx, attn_scratch_bytes_amx
from experimental.hadquant_attn import int8_gqa_attention, attn_scratch_bytes
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32, roundeven
from threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from numa.arena import NumaArena
from kernels.kernel_ops import PoolFence, init_rope_tables


def main():
    if not init_intel_amx():
        print("SKIP: AMX not available")
        return

    comptime HEAD_DIM = 64
    comptime NUM_HEADS = 9
    comptime NUM_KV_HEADS = 3
    comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS
    comptime HIDDEN = NUM_HEADS * HEAD_DIM
    comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
    comptime HALF = HEAD_DIM // 2
    comptime MAX_SEQ = 1024
    comptime BLOCK = HEAD_DIM
    comptime PREFILL_LEN = 4
    comptime POS = 0

    print("=== int8_gqa_attention_amx test ===")
    print("heads=" + String(NUM_HEADS) + " kv_heads=" + String(NUM_KV_HEADS)
          + " head_dim=" + String(HEAD_DIM) + " prefill=" + String(PREFILL_LEN))

    comptime KVCacheType = HadQuantKVCache[MAX_SEQ, HEAD_DIM, NUM_KV_HEADS]
    comptime SCRATCH_BYTES = attn_scratch_bytes_amx[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, NUM_KV_HEADS]()

    var numa = NumaInfo()
    var local_node = get_current_cpu_and_node()[1]
    var arena = NumaArena(local_node, 4 * 1024 * 1024)

    var q_bf16 = arena.alloc[Scalar[DType.bfloat16]](PREFILL_LEN * HIDDEN)
    var kv_mem = arena.alloc[UInt8](2 * KVCacheType.TOTAL_BYTES)
    var k_cache = KVCacheType(Int(kv_mem))
    var v_cache = KVCacheType(Int(kv_mem) + KVCacheType.TOTAL_BYTES)
    var qi_out = arena.alloc[Scalar[DType.int8]](PREFILL_LEN * HIDDEN)
    var scale_out = arena.alloc[Float32](PREFILL_LEN)
    var cos_table = arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_table = arena.alloc[Float32](MAX_SEQ * HALF)
    var attn_scratch = arena.alloc[UInt8](SCRATCH_BYTES)
    var expected = arena.alloc[Float32](PREFILL_LEN * HIDDEN)
    var q_head = arena.alloc[Float32](HEAD_DIM)
    var head_buf = arena.alloc[Scalar[DType.int8]](HEAD_DIM)
    var decode_q = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)

    # --- Fill Q with deterministic data ---
    for m in range(PREFILL_LEN):
        for k in range(HIDDEN):
            q_bf16[m * HIDDEN + k] = Scalar[DType.bfloat16](
                Float32(m * HIDDEN + k + 1) / Float32(PREFILL_LEN * HIDDEN) - 0.5
            )

    # --- Fill KV cache via write_head ---
    for t in range(PREFILL_LEN):
        for g in range(NUM_KV_HEADS):
            var k_absmax = Float32(0)
            for d in range(HEAD_DIM):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > k_absmax:
                    k_absmax = a
            k_cache.write_head(t, g, head_buf, k_absmax / Float32(127.0))

            var v_absmax = Float32(0)
            for d in range(HEAD_DIM):
                var val = Float32((t * 11 + g * 17 + d * 5) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > v_absmax:
                    v_absmax = a
            v_cache.write_head_transposed(t, g, head_buf, v_absmax / Float32(127.0))

    # --- Init RoPE tables ---
    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_table)), Bound[SinSlot](Int(sin_table)), Float64(100000.0))

    # --- Run AMX kernel ---
    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]
    comptime QiSlot = Slot[I8, Replicated, 1, HIDDEN, 1]
    comptime ScSlot = Slot[F32, Replicated, 1, 1, 1]

    var burst = BurstPool[].for_numa_node(numa, 0)

    int8_gqa_attention_amx[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](
        DynView[QSlot](Int(q_bf16), PREFILL_LEN),
        k_cache, v_cache,
        DynView[QiSlot](Int(qi_out), PREFILL_LEN),
        DynView[ScSlot](Int(scale_out), PREFILL_LEN),
        Bound[CosSlot](Int(cos_table)), Bound[SinSlot](Int(sin_table)),
        Int(attn_scratch), POS, burst,
    ).join()

    # --- F32 reference (identical to test_hadquant_attn) ---
    var inv_sqrt_hd = Float32(1.0) / sqrt[DType.float32, 1](Float32(HEAD_DIM))

    for m in range(PREFILL_LEN):
        var actual_pos = POS + m
        var context = actual_pos + 1

        for h in range(NUM_HEADS):
            var g = h // GQA_FACTOR

            for d in range(HEAD_DIM):
                q_head[d] = Float32(q_bf16[m * HIDDEN + h * HEAD_DIM + d])

            for j in range(HALF):
                var x_lo = q_head[j]
                var x_hi = q_head[HALF + j]
                var cv = cos_table[actual_pos * HALF + j]
                var sv = sin_table[actual_pos * HALF + j]
                q_head[j] = x_lo * cv - x_hi * sv
                q_head[HALF + j] = x_hi * cv + x_lo * sv

            fwht_block[DType.float32, BLOCK](q_head)

            var scores_mark = arena.mark()
            var scores = arena.alloc[Float32](context)
            for t in range(context):
                var dot = Float32(0)
                var k_sc = k_cache.head_scale(t, g)
                var k_data = k_cache.head_data(t, g)
                for d in range(HEAD_DIM):
                    dot += q_head[d] * Float32(Int(k_data[d]) - 128) * k_sc
                scores[t] = dot * inv_sqrt_hd

            var smax = scores[0]
            for t in range(1, context):
                if scores[t] > smax:
                    smax = scores[t]
            var exp_sum = Float32(0)
            for t in range(context):
                scores[t] = exp_f32[1](scores[t] - smax)
                exp_sum += scores[t]
            var inv_sum = Float32(1.0) / exp_sum
            for t in range(context):
                scores[t] = scores[t] * inv_sum

            for d in range(HEAD_DIM):
                var acc = Float32(0)
                var v_dim = v_cache.dim_data(d, g)
                for t in range(context):
                    var v_sc = v_cache.head_scale(t, g)
                    acc += scores[t] * Float32(Int(v_dim[t]) - 128) * v_sc
                expected[m * HIDDEN + h * HEAD_DIM + d] = acc

            arena.reset_to(scores_mark)

    # --- Compare ---
    var max_err = Float64(0)
    var sum_err = Float64(0)
    var count = 0

    for m in range(PREFILL_LEN):
        var out_scale = Float64(scale_out[m])
        for d in range(HIDDEN):
            var got = Float64(qi_out[m * HIDDEN + d]) * out_scale
            var exp_val = Float64(expected[m * HIDDEN + d])
            var err = got - exp_val
            if err < 0:
                err = -err
            if err > max_err:
                max_err = err
            sum_err += err
            count += 1

    var avg_err = sum_err / Float64(count)

    print("max_err=" + String(max_err) + " avg_err=" + String(avg_err))
    if avg_err < 0.05:
        print("PASS: AMX attention output within expected int8 quantization error")
    else:
        print("WARN: avg_err=" + String(avg_err) + " -- higher than expected")
        for d in range(min(8, HIDDEN)):
            var got = Float64(qi_out[d]) * Float64(scale_out[0])
            var exp_val = Float64(expected[d])
            print("  [0," + String(d) + "] got=" + String(got) + " exp=" + String(exp_val))

    # --- DeepSeek-like profile ---
    print("\n=== DeepSeek-V3-like AMX profile ===")
    print("128 heads, 8 kv_heads, head_dim=128, context=4096, decode")

    comptime DS_HEAD_DIM = 128
    comptime DS_NUM_HEADS = 128
    comptime DS_NUM_KV_HEADS = 8
    comptime DS_HIDDEN = DS_NUM_HEADS * DS_HEAD_DIM
    comptime DS_HALF = DS_HEAD_DIM // 2
    comptime DS_MAX_SEQ = 8192
    comptime DS_CTX = 4096
    comptime DsKVCache = HadQuantKVCache[DS_MAX_SEQ, DS_HEAD_DIM, DS_NUM_KV_HEADS]
    comptime DS_SCRATCH_BYTES = attn_scratch_bytes_amx[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM, DS_NUM_KV_HEADS]()

    var ds_arena = NumaArena(local_node, 32 * 1024 * 1024)

    var ds_q = ds_arena.alloc[Scalar[DType.bfloat16]](DS_HIDDEN)
    var ds_kv_mem = ds_arena.alloc[UInt8](2 * DsKVCache.TOTAL_BYTES)
    var ds_k = DsKVCache(Int(ds_kv_mem))
    var ds_v = DsKVCache(Int(ds_kv_mem) + DsKVCache.TOTAL_BYTES)
    var ds_qi_out = ds_arena.alloc[Scalar[DType.int8]](DS_HIDDEN)
    var ds_sc_out = ds_arena.alloc[Float32](1)
    var ds_cos = ds_arena.alloc[Float32](DS_MAX_SEQ * DS_HALF)
    var ds_sin = ds_arena.alloc[Float32](DS_MAX_SEQ * DS_HALF)
    var ds_scratch = ds_arena.alloc[UInt8](DS_SCRATCH_BYTES)
    var ds_head_buf = ds_arena.alloc[Scalar[DType.int8]](DS_HEAD_DIM)

    for k in range(DS_HIDDEN):
        ds_q[k] = Scalar[DType.bfloat16](Float32(k % 256 - 128) / 128.0)
    for t in range(DS_CTX):
        for g in range(DS_NUM_KV_HEADS):
            for d in range(DS_HEAD_DIM):
                ds_head_buf[d] = Scalar[DType.int8]((t * 7 + g * DS_HEAD_DIM + d * 3) % 251 - 125)
            ds_k.write_head(t, g, ds_head_buf, Float32(0.05))
            for d in range(DS_HEAD_DIM):
                ds_head_buf[d] = Scalar[DType.int8]((t * 11 + g * DS_HEAD_DIM + d * 5) % 251 - 125)
            ds_v.write_head_transposed(t, g, ds_head_buf, Float32(0.05))

    comptime DsCosSlot = Slot[F32, Replicated, DS_MAX_SEQ, DS_HALF, 1]
    comptime DsSinSlot = Slot[F32, Replicated, DS_MAX_SEQ, DS_HALF, 1]
    init_rope_tables(Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)), Float64(100000.0))

    comptime DsQSlot = Slot[BF16, Replicated, 1, DS_HIDDEN, 1]
    comptime DsQiSlot = Slot[I8, Replicated, 1, DS_HIDDEN, 1]
    comptime DsScSlot = Slot[F32, Replicated, 1, 1, 1]

    # --- VNNI baseline first (clean caches) ---
    print("\n=== VNNI baseline ===")
    comptime DS_VNNI_SCRATCH = attn_scratch_bytes[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM, DS_NUM_KV_HEADS]()
    var vnni_scratch = ds_arena.alloc[UInt8](DS_VNNI_SCRATCH)

    for run in range(3):
        int8_gqa_attention[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM](
            DynView[DsQSlot](Int(ds_q), 1), ds_k, ds_v,
            DynView[DsQiSlot](Int(ds_qi_out), 1),
            DynView[DsScSlot](Int(ds_sc_out), 1),
            Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)),
            Int(vnni_scratch), DS_CTX, burst,
        ).join()
    for run in range(10):
        var t0 = Int(perf_counter_ns())
        int8_gqa_attention[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM](
            DynView[DsQSlot](Int(ds_q), 1), ds_k, ds_v,
            DynView[DsQiSlot](Int(ds_qi_out), 1),
            DynView[DsScSlot](Int(ds_sc_out), 1),
            Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)),
            Int(vnni_scratch), DS_CTX, burst,
        ).join()
        var wall = Int(perf_counter_ns()) - t0
        keep(ds_qi_out[0])
        keep(ds_sc_out[0])
        print("  VNNI: " + String(wall // 1000) + " us")

    # --- AMX ---
    print("\n=== AMX (CHUNK=512) ===")
    for run in range(3):
        int8_gqa_attention_amx[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM](
            DynView[DsQSlot](Int(ds_q), 1), ds_k, ds_v,
            DynView[DsQiSlot](Int(ds_qi_out), 1),
            DynView[DsScSlot](Int(ds_sc_out), 1),
            Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)),
            Int(ds_scratch), DS_CTX, burst,
        ).join()
    for run in range(10):
        var t0 = Int(perf_counter_ns())
        int8_gqa_attention_amx[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM](
            DynView[DsQSlot](Int(ds_q), 1), ds_k, ds_v,
            DynView[DsQiSlot](Int(ds_qi_out), 1),
            DynView[DsScSlot](Int(ds_sc_out), 1),
            Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)),
            Int(ds_scratch), DS_CTX, burst,
        ).join()
        var wall = Int(perf_counter_ns()) - t0
        keep(ds_qi_out[0])
        keep(ds_sc_out[0])
        print("  AMX:  " + String(wall // 1000) + " us")
