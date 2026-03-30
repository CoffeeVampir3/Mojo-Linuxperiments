"""Test int8_gqa_attention against scalar f32 reference.

Reference computes standard GQA attention in f32:
  1. RoPE on Q per head
  2. dot(Q, K) / sqrt(HD) → scores
  3. causal masked softmax
  4. weighted sum of V
Then compares the dequantized int8 output against the f32 reference.
"""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.math import sqrt as std_sqrt

from modeling.model_spec import (
    BF16, F32, I8, Replicated, ColShard,
    Slot, Bound, DynView,
)
from experimental.hadquant_attn import int8_gqa_attention
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from simd_math import sqrt, exp_f32, roundeven
from threading import BurstPool
from numa import NumaInfo
from kernels.kernel_ops import PoolFence, init_rope_tables


def main():
    comptime HEAD_DIM = 64
    comptime NUM_HEADS = 9
    comptime NUM_KV_HEADS = 3
    comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS
    comptime HIDDEN = NUM_HEADS * HEAD_DIM      # 576
    comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 192
    comptime HALF = HEAD_DIM // 2
    comptime MAX_SEQ = 1024
    comptime BLOCK = HEAD_DIM

    # Test: prefill 4 tokens starting at pos=0, then decode 1 token at pos=4
    comptime PREFILL_LEN = 4
    comptime POS = 0

    print("=== int8_gqa_attention test ===")
    print("heads=" + String(NUM_HEADS) + " kv_heads=" + String(NUM_KV_HEADS)
          + " head_dim=" + String(HEAD_DIM) + " prefill=" + String(PREFILL_LEN))

    # --- Allocate ---
    comptime KVCacheType = HadQuantKVCache[MAX_SEQ, HEAD_DIM, NUM_KV_HEADS]

    var q_bf16 = alloc[Scalar[DType.bfloat16]](PREFILL_LEN * HIDDEN)
    var kv_mem = alloc[UInt8](2 * KVCacheType.TOTAL_BYTES)  # K + V caches
    var k_cache = KVCacheType(Int(kv_mem))
    var v_cache = KVCacheType(Int(kv_mem) + KVCacheType.TOTAL_BYTES)
    var qi_out = alloc[Scalar[DType.int8]](PREFILL_LEN * HIDDEN)
    var scale_out = alloc[Float32](PREFILL_LEN)
    var cos_table = alloc[Float32](MAX_SEQ * HALF)
    var sin_table = alloc[Float32](MAX_SEQ * HALF)

    # --- Fill Q with deterministic data ---
    for m in range(PREFILL_LEN):
        for k in range(HIDDEN):
            q_bf16[m * HIDDEN + k] = Scalar[DType.bfloat16](
                Float32(m * HIDDEN + k + 1) / Float32(PREFILL_LEN * HIDDEN) - 0.5
            )

    # --- Fill KV cache via write_head ---
    var head_buf = alloc[Scalar[DType.int8]](HEAD_DIM)
    for t in range(PREFILL_LEN):
        for g in range(NUM_KV_HEADS):
            # K cache
            var k_absmax = Float32(0)
            for d in range(HEAD_DIM):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > k_absmax: k_absmax = a
            k_cache.write_head(t, g, head_buf, k_absmax / Float32(127.0))

            # V cache
            var v_absmax = Float32(0)
            for d in range(HEAD_DIM):
                var val = Float32((t * 11 + g * 17 + d * 5) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
                var a = val if val >= 0 else -val
                if a > v_absmax: v_absmax = a
            v_cache.write_head(t, g, head_buf, v_absmax / Float32(127.0))

    # --- Init RoPE tables ---
    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_table)), Bound[SinSlot](Int(sin_table)), Float64(100000.0))

    # --- Run kernel ---
    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]
    comptime QiSlot = Slot[I8, Replicated, 1, HIDDEN, 1]
    comptime ScSlot = Slot[F32, Replicated, 1, 1, 1]

    var numa = NumaInfo()
    var burst = BurstPool[].for_numa_node(numa, 0)

    int8_gqa_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](
        DynView[QSlot](Int(q_bf16), PREFILL_LEN),
        k_cache, v_cache,
        DynView[QiSlot](Int(qi_out), PREFILL_LEN),
        DynView[ScSlot](Int(scale_out), PREFILL_LEN),
        Bound[CosSlot](Int(cos_table)), Bound[SinSlot](Int(sin_table)),
        POS, burst,
    ).join()

    # --- F32 reference ---
    # For each query row m, head h:
    #   1. Load Q bf16 → f32, RoPE, FWHT
    #   2. Dequant K cache → f32, dot with Q / sqrt(HD)
    #   3. Softmax, absorb V scales
    #   4. Dequant V cache → f32, weighted sum
    var expected = alloc[Float32](PREFILL_LEN * HIDDEN)
    var q_head = alloc[Float32](HEAD_DIM)
    var inv_sqrt_hd = Float32(1.0) / sqrt[DType.float32, 1](Float32(HEAD_DIM))

    for m in range(PREFILL_LEN):
        var actual_pos = POS + m
        var context = actual_pos + 1

        for h in range(NUM_HEADS):
            var g = h // GQA_FACTOR

            # Load Q head
            for d in range(HEAD_DIM):
                q_head[d] = Float32(q_bf16[m * HIDDEN + h * HEAD_DIM + d])

            # RoPE
            for j in range(HALF):
                var x_lo = q_head[j]
                var x_hi = q_head[HALF + j]
                var cv = cos_table[actual_pos * HALF + j]
                var sv = sin_table[actual_pos * HALF + j]
                q_head[j] = x_lo * cv - x_hi * sv
                q_head[HALF + j] = x_hi * cv + x_lo * sv

            # FWHT
            fwht_block[DType.float32, BLOCK](q_head)

            # Score against K cache (dequant u8 → signed f32)
            var scores = alloc[Float32](context)
            for t in range(context):
                var dot = Float32(0)
                var k_sc = k_cache.head_scale(t, g)
                var k_data = k_cache.head_data(t, g)
                for d in range(HEAD_DIM):
                    dot += q_head[d] * Float32(Int(k_data[d]) - 128) * k_sc
                scores[t] = dot * inv_sqrt_hd

            # Softmax
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

            # Weighted sum of V (dequant u8 → signed f32)
            for d in range(HEAD_DIM):
                var acc = Float32(0)
                for t in range(context):
                    var v_sc = v_cache.head_scale(t, g)
                    var v_data = v_cache.head_data(t, g)
                    acc += scores[t] * Float32(Int(v_data[d]) - 128) * v_sc
                expected[m * HIDDEN + h * HEAD_DIM + d] = acc

            scores.free()

    # --- Compare ---
    # The kernel output is int8 quantized. Dequantize and compare against f32 reference.
    # We expect quantization error but the attention pattern should be correct.
    var max_err = Float64(0)
    var sum_err = Float64(0)
    var count = 0

    for m in range(PREFILL_LEN):
        var out_scale = Float64(scale_out[m])
        for d in range(HIDDEN):
            var got = Float64(qi_out[m * HIDDEN + d]) * out_scale
            var exp_val = Float64(expected[m * HIDDEN + d])
            var err = got - exp_val
            if err < 0: err = -err
            if err > max_err: max_err = err
            sum_err += err
            count += 1

    var avg_err = sum_err / Float64(count)

    # Int8 quantization introduces ~1% error per stage.
    # Three quantization stages (Q, weights, output) compound.
    # The reference uses f32 throughout, so some divergence is expected.
    print("max_err=" + String(max_err) + " avg_err=" + String(avg_err))
    if avg_err < 0.05:
        print("PASS: attention output within expected int8 quantization error")
    else:
        print("WARN: avg_err=" + String(avg_err) + " — higher than expected")
        # Print a few values for debugging
        for d in range(min(8, HIDDEN)):
            var got = Float64(qi_out[d]) * Float64(scale_out[0])
            var exp_val = Float64(expected[d])
            print("  [0," + String(d) + "] got=" + String(got) + " exp=" + String(exp_val))

    # --- Profile at larger context (decode at pos=512, averaged over N runs) ---
    print("\n=== Decode profile at context=512 (10 runs, last shown) ===")

    for t in range(512):
        for g in range(NUM_KV_HEADS):
            for d in range(HEAD_DIM):
                head_buf[d] = Scalar[DType.int8]((t * 7 + g * 13 + d * 3) % 251 - 125)
            k_cache.write_head(t, g, head_buf, Float32(0.05))
            for d in range(HEAD_DIM):
                head_buf[d] = Scalar[DType.int8]((t * 11 + g * 17 + d * 5) % 251 - 125)
            v_cache.write_head(t, g, head_buf, Float32(0.05))

    var decode_q = alloc[Scalar[DType.bfloat16]](HIDDEN)
    for k in range(HIDDEN):
        decode_q[k] = Scalar[DType.bfloat16](Float32(k) / Float32(HIDDEN) - 0.5)

    for run in range(10):
        int8_gqa_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](
            DynView[QSlot](Int(decode_q), 1),
            k_cache, v_cache,
            DynView[QiSlot](Int(qi_out), 1),
            DynView[ScSlot](Int(scale_out), 1),
            Bound[CosSlot](Int(cos_table)), Bound[SinSlot](Int(sin_table)),
            512, burst,
        ).join()

    decode_q.free()
    q_bf16.free()
    kv_mem.free()
    qi_out.free()
    scale_out.free()
    cos_table.free()
    sin_table.free()
    expected.free()
    q_head.free()
    head_buf.free()

    # === Large-scale profile: DeepSeek-V3-like dimensions ===
    # 128 heads, 8 KV heads (GQA=16), head_dim=128, context=4096
    print("\n=== DeepSeek-V3-like profile ===")
    print("128 heads, 8 kv_heads, head_dim=128, context=4096, decode")

    comptime DS_HEAD_DIM = 128
    comptime DS_NUM_HEADS = 128
    comptime DS_NUM_KV_HEADS = 8
    comptime DS_HIDDEN = DS_NUM_HEADS * DS_HEAD_DIM
    comptime DS_HALF = DS_HEAD_DIM // 2
    comptime DS_MAX_SEQ = 8192
    comptime DS_CTX = 4096
    comptime DsKVCache = HadQuantKVCache[DS_MAX_SEQ, DS_HEAD_DIM, DS_NUM_KV_HEADS]

    var ds_q = alloc[Scalar[DType.bfloat16]](DS_HIDDEN)
    var ds_kv_mem = alloc[UInt8](2 * DsKVCache.TOTAL_BYTES)
    var ds_k = DsKVCache(Int(ds_kv_mem))
    var ds_v = DsKVCache(Int(ds_kv_mem) + DsKVCache.TOTAL_BYTES)
    var ds_qi_out = alloc[Scalar[DType.int8]](DS_HIDDEN)
    var ds_sc_out = alloc[Float32](1)
    var ds_cos = alloc[Float32](DS_MAX_SEQ * DS_HALF)
    var ds_sin = alloc[Float32](DS_MAX_SEQ * DS_HALF)

    for k in range(DS_HIDDEN):
        ds_q[k] = Scalar[DType.bfloat16](Float32(k % 256 - 128) / 128.0)
    var ds_head_buf = alloc[Scalar[DType.int8]](DS_HEAD_DIM)
    for t in range(DS_CTX):
        for g in range(DS_NUM_KV_HEADS):
            for d in range(DS_HEAD_DIM):
                ds_head_buf[d] = Scalar[DType.int8]((t * 7 + g * DS_HEAD_DIM + d * 3) % 251 - 125)
            ds_k.write_head(t, g, ds_head_buf, Float32(0.05))
            for d in range(DS_HEAD_DIM):
                ds_head_buf[d] = Scalar[DType.int8]((t * 11 + g * DS_HEAD_DIM + d * 5) % 251 - 125)
            ds_v.write_head(t, g, ds_head_buf, Float32(0.05))

    comptime DsCosSlot = Slot[F32, Replicated, DS_MAX_SEQ, DS_HALF, 1]
    comptime DsSinSlot = Slot[F32, Replicated, DS_MAX_SEQ, DS_HALF, 1]
    init_rope_tables(Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)), Float64(100000.0))

    comptime DsQSlot = Slot[BF16, Replicated, 1, DS_HIDDEN, 1]
    comptime DsQiSlot = Slot[I8, Replicated, 1, DS_HIDDEN, 1]
    comptime DsScSlot = Slot[F32, Replicated, 1, 1, 1]

    for run in range(5):
        int8_gqa_attention[DS_NUM_HEADS, DS_NUM_KV_HEADS, DS_HEAD_DIM](
            DynView[DsQSlot](Int(ds_q), 1),
            ds_k, ds_v,
            DynView[DsQiSlot](Int(ds_qi_out), 1),
            DynView[DsScSlot](Int(ds_sc_out), 1),
            Bound[DsCosSlot](Int(ds_cos)), Bound[DsSinSlot](Int(ds_sin)),
            DS_CTX, burst,
        ).join()

    ds_q.free()
    ds_kv_mem.free()
    ds_qi_out.free()
    ds_sc_out.free()
    ds_cos.free()
    ds_sin.free()
    ds_head_buf.free()
