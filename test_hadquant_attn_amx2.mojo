"""Correctness + performance test for experimental2 AMX prefill kernel.

Zero per-token scales. KV cache is flat u8. Fixed layer-level scales
for Q, K, V quantization.
"""

from std.sys.info import simd_width_of, size_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental2.attn_amx_prefill import prefill, scratch_bytes
from experimental.hadquant_impl import fwht_block
from experimental2.kv_cache import KVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32, quantize_i8
from threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from kernels.kernel_ops import init_rope_tables, parallel_for, PoolFence


comptime NUM_NODES = 4


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    var numa = NumaInfo()
    if numa.num_nodes < NUM_NODES:
        print("SKIP: need " + String(NUM_NODES) + " NUMA nodes, have " + String(numa.num_nodes))
        return
    var topo = numa.plan_topology(NUM_NODES)

    print("NUMA: " + String(NUM_NODES) + " nodes")
    for i in range(NUM_NODES):
        print("  node " + String(topo[i]) + ": " + String(numa.cpus_on_node(topo[i])) + " cpus")

    var pools = HeapMoveArray[BurstPool[]](NUM_NODES)
    for i in range(NUM_NODES):
        pools.push(BurstPool[].for_numa_node(numa, topo[i]))
    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], NUM_NODES](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for i in range(NUM_NODES):
        pool_ptrs[i] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[i])))

    # =====================================================================
    # Correctness test
    # =====================================================================
    comptime HD = 64
    comptime NH = 12
    comptime NKV = 4
    comptime GQA = NH // NKV
    comptime HIDDEN = NH * HD
    comptime HALF = HD // 2
    comptime MAX_SEQ = 1024
    comptime SL = 16
    comptime POS = 48

    # Fixed per-layer scales (simulating what model loading would provide)
    var q_layer_scale = Float32(0.15)
    var k_layer_scale = Float32(0.15)
    var v_layer_scale = Float32(0.15)
    var q_quant_inv = Float32(127.0) / q_layer_scale

    print("\n=== Correctness: " + String(NH) + "h/" + String(NKV)
          + "kv, hd=" + String(HD) + ", sl=" + String(SL)
          + ", pos=" + String(POS) + " ===")

    var corr_arena = NumaArena[](topo[0], 256 * 1024 * 1024)

    comptime KVC = KVCache[MAX_SEQ, HD, NKV]
    var corr_scratch_bytes = scratch_bytes[NH, NKV, HD, SL](pools[0].capacity)

    var q_bf16 = corr_arena.alloc[Scalar[DType.bfloat16]](SL * HIDDEN)
    var kv_mem = corr_arena.alloc[UInt8](2 * KVC.TOTAL_BYTES)
    var k_cache = KVC(Int(kv_mem))
    var v_cache = KVC(Int(kv_mem) + KVC.TOTAL_BYTES)
    var qi_out = corr_arena.alloc[Scalar[DType.int8]](SL * HIDDEN)
    var sc_out = corr_arena.alloc[Float32](SL)
    var cos_tab = corr_arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_tab = corr_arena.alloc[Float32](MAX_SEQ * HALF)
    var scratch = corr_arena.alloc[UInt8](corr_scratch_bytes)
    var expected = corr_arena.alloc[Float32](SL * HIDDEN)
    var q_head = corr_arena.alloc[Float32](HD)
    var head_buf = corr_arena.alloc[Scalar[DType.int8]](HD)

    # Fill Q
    for m in range(SL):
        for k in range(HIDDEN):
            q_bf16[m * HIDDEN + k] = Scalar[DType.bfloat16](
                Float32(m * HIDDEN + k + 1) / Float32(SL * HIDDEN) - 0.5)

    # Fill KV cache with fixed-scale quantized data (no per-token scales)
    for t in range(POS + SL):
        for g in range(NKV):
            for d in range(HD):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
            k_cache.write_k(t, g, head_buf)
            for d in range(HD):
                var val = Float32((t * 11 + g * 17 + d * 5) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
            v_cache.write_v(t, g, head_buf)

    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)), Float64(100000.0))

    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]

    # F32 reference: same pipeline as the kernel, in scalar
    var inv_sqrt_hd = Float32(1.0) / sqrt[DType.float32, 1](Float32(HD))
    for m in range(SL):
        var actual_pos = POS + m
        var context = actual_pos + 1
        for h in range(NH):
            var g = h // GQA
            # Q: bf16 → f32 → RoPE → FWHT → fixed-scale i8
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
            # Fixed-scale quantize Q
            var qi_ref = corr_arena.alloc[Scalar[DType.int8]](HD)
            var q_sum = Int(0)
            for d in range(HD):
                var qi = Int(max(min(Int(q_head[d] * q_quant_inv + 0.5), 127), -128))
                if q_head[d] * q_quant_inv < 0:
                    qi = Int(max(min(Int(q_head[d] * q_quant_inv - 0.5), 127), -128))
                qi_ref[d] = Scalar[DType.int8](qi)
                q_sum += qi
            var q_bias = Float32(128 * q_sum)
            var score_scale = q_layer_scale * k_layer_scale * inv_sqrt_hd / (Float32(127.0) * Float32(127.0))

            # Score against K cache
            var scores_mark = corr_arena.mark()
            var scores = corr_arena.alloc[Float32](context)
            for t in range(context):
                var k_data = k_cache.head_data(t, g)
                var dot = Int32(0)
                for d in range(HD):
                    dot += Int32(qi_ref[d]) * Int32(k_data[d])
                scores[t] = (Float32(dot) - q_bias) * score_scale

            # Softmax
            var smax = scores[0]
            for t in range(1, context):
                if scores[t] > smax: smax = scores[t]
            var exp_sum = Float32(0)
            for t in range(context):
                scores[t] = exp_f32[1](scores[t] - smax)
                exp_sum += scores[t]
            var vagg_scale = v_layer_scale / (Float32(255.0) * Float32(127.0))
            for d in range(HD):
                expected[m * HIDDEN + h * HD + d] = Float32(0)
            for t in range(context):
                var w_u8 = UInt8(max(min(Int(scores[t] * 255.0 + 0.5), 255), 0))
                var v_data = v_cache.head_data(t, g).bitcast[Scalar[DType.int8]]()
                for d in range(HD):
                    var v_i8 = Int32(v_data[d])
                    expected[m * HIDDEN + h * HD + d] += Float32(Int32(w_u8) * v_i8) * vagg_scale
            for d in range(HD):
                expected[m * HIDDEN + h * HD + d] /= exp_sum

            corr_arena.reset_to(scores_mark)

    # Run kernel
    prefill[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), SL),
        k_cache, v_cache,
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(scratch), POS,
        q_layer_scale, k_layer_scale, v_layer_scale,
        pool_ptrs[0][],
    ).join()

    # Compare
    var rf32 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scratch))
    var err_sum = Float64(0)
    var count = 0
    for m in range(SL):
        for d in range(HIDDEN):
            var got = Float64(rf32[m * HIDDEN + d])
            var rv = Float64(expected[m * HIDDEN + d])
            var e = got - rv
            if e < 0: e = -e
            err_sum += e
            count += 1
    var avg_err = err_sum / Float64(count)
    print("avg_err: " + String(avg_err))
    if avg_err < 0.1:
        print("PASS")
    else:
        print("FAIL")

    # =====================================================================
    # Performance sweep
    # =====================================================================
    comptime HD2 = 128
    comptime NH2 = 128
    comptime NKV2 = 8
    comptime HALF2 = HD2 // 2
    comptime MAX_SEQ2 = 8192
    comptime MAX_SL = 64

    comptime LOCAL_KV2 = NKV2 // NUM_NODES
    comptime LOCAL_NH2 = NH2 // NUM_NODES
    comptime LOCAL_HIDDEN2 = LOCAL_NH2 * HD2

    comptime LOCAL_KVC = KVCache[MAX_SEQ2, HD2, LOCAL_KV2]
    var local_scratch_bytes = scratch_bytes[LOCAL_NH2, LOCAL_KV2, HD2, MAX_SL](pools[0].capacity)

    print("\n=== Performance: 128h/8kv, hd=128, " + String(NUM_NODES) + " NUMA nodes ===")
    print("  Per node: " + String(LOCAL_NH2) + "h/" + String(LOCAL_KV2) + "kv, "
          + String(pools[0].capacity) + " cores")

    var perf_arenas = HeapMoveArray[NumaArena[]](NUM_NODES)
    for i in range(NUM_NODES):
        perf_arenas.push(NumaArena[](topo[i], 512 * 1024 * 1024))

    var q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var k_bases = InlineArray[Int, NUM_NODES](fill=0)
    var v_bases = InlineArray[Int, NUM_NODES](fill=0)
    var cos_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var sin_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var scratch_ptrs = InlineArray[Int, NUM_NODES](fill=0)

    comptime CS2 = Slot[F32, Replicated, MAX_SEQ2, HALF2, 1]
    comptime SS2 = Slot[F32, Replicated, MAX_SEQ2, HALF2, 1]

    var perf_q_scale = Float32(0.15)
    var perf_k_scale = Float32(0.15)
    var perf_v_scale = Float32(0.15)

    for node in range(NUM_NODES):
        var cos_node = perf_arenas[node].alloc[Float32](MAX_SEQ2 * HALF2)
        var sin_node = perf_arenas[node].alloc[Float32](MAX_SEQ2 * HALF2)
        init_rope_tables(Bound[CS2](Int(cos_node)), Bound[SS2](Int(sin_node)), Float64(100000.0))
        cos_ptrs[node] = Int(cos_node)
        sin_ptrs[node] = Int(sin_node)

        var q_node = perf_arenas[node].alloc[Scalar[DType.bfloat16]](MAX_SL * LOCAL_HIDDEN2)
        for i in range(MAX_SL * LOCAL_HIDDEN2):
            q_node[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)
        q_ptrs[node] = Int(q_node)

        var kv_mem = perf_arenas[node].alloc[UInt8](2 * LOCAL_KVC.TOTAL_BYTES)
        k_bases[node] = Int(kv_mem)
        v_bases[node] = Int(kv_mem) + LOCAL_KVC.TOTAL_BYTES
        var k_node = LOCAL_KVC(Int(kv_mem))
        var v_node = LOCAL_KVC(Int(kv_mem) + LOCAL_KVC.TOTAL_BYTES)
        var hb = perf_arenas[node].alloc[Scalar[DType.int8]](HD2)
        for t in range(MAX_SEQ2):
            for lg in range(LOCAL_KV2):
                var global_g = node * LOCAL_KV2 + lg
                for d in range(HD2):
                    hb[d] = Scalar[DType.int8]((t * 7 + global_g * HD2 + d * 3) % 251 - 125)
                k_node.write_k(t, lg, hb)
                for d in range(HD2):
                    hb[d] = Scalar[DType.int8]((t * 11 + global_g * HD2 + d * 5) % 251 - 125)
                v_node.write_v(t, lg, hb)

        scratch_ptrs[node] = Int(perf_arenas[node].alloc[UInt8](local_scratch_bytes))

    comptime QS2 = Slot[BF16, Replicated, 1, LOCAL_HIDDEN2, 1]

    print("\n  ctx  |  sl |  total us | us/pos")
    print("  -----|-----|-----------|-------")

    for ctx_idx in range(3):
        var ctx_pos = 128
        if ctx_idx == 1: ctx_pos = 2048
        if ctx_idx == 2: ctx_pos = 4096

        for sl_idx in range(3):
            var sl = 1
            if sl_idx == 1: sl = 16
            if sl_idx == 2: sl = 64

            for warmup in range(2):
                @parameter
                def wu[node: Int]() -> PoolFence:
                    return prefill[LOCAL_NH2, LOCAL_KV2, HD2](
                        DynView[QS2](q_ptrs[node], sl),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos,
                        perf_q_scale, perf_k_scale, perf_v_scale,
                        pool_ptrs[node][],
                    )
                parallel_for[NUM_NODES, wu]()

            var best = Int(1 << 60)
            for trial in range(5):
                var t0 = Int(perf_counter_ns())
                @parameter
                def run[node: Int]() -> PoolFence:
                    return prefill[LOCAL_NH2, LOCAL_KV2, HD2](
                        DynView[QS2](q_ptrs[node], sl),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos,
                        perf_q_scale, perf_k_scale, perf_v_scale,
                        pool_ptrs[node][],
                    )
                parallel_for[NUM_NODES, run]()
                var wall = Int(perf_counter_ns()) - t0
                keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
                if wall < best: best = wall

            var total_us = best // 1000
            var per_pos = total_us // sl if sl > 0 else 0
            print("  " + String(ctx_pos) + " | " + String(sl)
                  + " | " + String(total_us)
                  + " | " + String(per_pos))
