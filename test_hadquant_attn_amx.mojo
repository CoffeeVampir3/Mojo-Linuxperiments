"""Correctness + performance test for AMX bf16 V-agg attention kernel.

Correctness: single-node, validates kernel logic against f32 reference.
Performance: per-rank dispatch across NUM_NODES SNC clusters, sweep
over context lengths and sequence lengths.
"""

from std.sys.info import simd_width_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental.hadquant_attn_amx_prefill import int8_gqa_attention_amx_prefill, attn_scratch_bytes_amx_prefill, attn_per_worker_bytes
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32, quantize_i8
from threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from kernels.kernel_ops import init_rope_tables, parallel_for, PoolFence


comptime NUM_NODES = 4  # 4 SNC clusters on target SPR


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    # --- Discover NUMA topology ---
    var numa = NumaInfo()
    if numa.num_nodes < NUM_NODES:
        print("SKIP: need " + String(NUM_NODES) + " NUMA nodes, have " + String(numa.num_nodes))
        return
    var topo = numa.plan_topology(NUM_NODES)

    print("NUMA: " + String(NUM_NODES) + " nodes")
    for i in range(NUM_NODES):
        print("  node " + String(topo[i]) + ": " + String(numa.cpus_on_node(topo[i])) + " cpus")

    # --- Per-node BurstPools ---
    var pools = HeapMoveArray[BurstPool[]](NUM_NODES)
    for i in range(NUM_NODES):
        pools.push(BurstPool[].for_numa_node(numa, topo[i]))

    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], NUM_NODES](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for i in range(NUM_NODES):
        pool_ptrs[i] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[i])))

    # =====================================================================
    # Correctness test — single node, validates kernel logic
    # =====================================================================
    comptime HD = 64
    comptime NH = 12
    comptime NKV = 4
    comptime GQA = NH // NKV
    comptime HIDDEN = NH * HD
    comptime HALF = HD // 2
    comptime MAX_SEQ = 1024
    comptime SL = 16
    comptime POS = 0

    print("\n=== Correctness: " + String(NH) + "h/" + String(NKV)
          + "kv, hd=" + String(HD) + ", sl=" + String(SL) + " ===")

    var corr_arena = NumaArena[](topo[0], 256 * 1024 * 1024)

    comptime KVC = HadQuantKVCache[MAX_SEQ, HD, NKV]
    var corr_scratch_bytes = attn_scratch_bytes_amx_prefill[NH, NKV, HD, SL](pools[0].capacity)

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

    for m in range(SL):
        for k in range(HIDDEN):
            q_bf16[m * HIDDEN + k] = Scalar[DType.bfloat16](
                Float32(m * HIDDEN + k + 1) / Float32(SL * HIDDEN) - 0.5)
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

    # F32 reference
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
            var scores_mark = corr_arena.mark()
            var scores = corr_arena.alloc[Float32](context)
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
            corr_arena.reset_to(scores_mark)

    # Run kernel, join, output quantize
    int8_gqa_attention_amx_prefill[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), SL), k_cache, v_cache,
        DynView[QiSlot](Int(qi_out), SL), DynView[ScSlot](Int(sc_out), SL),
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(scratch), POS, pool_ptrs[0][],
    ).join()
    comptime width = simd_width_of[DType.float32]()
    var rf32 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scratch))
    for m in range(SL):
        var src = rf32 + m * HIDDEN
        var out = qi_out + m * HIDDEN
        var rmax = SIMD[DType.float32, width](0)
        var d = 0
        while d + width <= HIDDEN:
            rmax = max(rmax, (src + d).load[width=width]().__abs__())
            d += width
        var absmax = rmax.reduce_max()
        sc_out[m] = absmax / Float32(127.0)
        var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
        var vinv = SIMD[DType.float32, width](inv)
        d = 0
        while d + width <= HIDDEN:
            (out + d).store(quantize_i8((src + d).load[width=width](), vinv))
            d += width

    # Compare
    var err_sum = Float64(0)
    var count = 0
    for m in range(SL):
        for d in range(HIDDEN):
            var got = Float64(qi_out[m * HIDDEN + d]) * Float64(sc_out[m])
            var rv = Float64(expected[m * HIDDEN + d])
            var e = got - rv
            if e < 0: e = -e
            err_sum += e
            count += 1
    print("avg_err: " + String(err_sum / Float64(count)))
    if err_sum / Float64(count) < 0.05:
        print("PASS")
    else:
        print("FAIL")

    # =====================================================================
    # Performance sweep: bf16 V-agg at various context and seq lengths
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

    comptime LOCAL_KVC = HadQuantKVCache[MAX_SEQ2, HD2, LOCAL_KV2]
    var local_scratch_bytes = attn_scratch_bytes_amx_prefill[LOCAL_NH2, LOCAL_KV2, HD2, MAX_SL](pools[0].capacity)

    print("\n=== Performance: bf16 V-agg, 128h/8kv, hd=128, "
          + String(NUM_NODES) + " NUMA nodes ===")
    print("  Per node: " + String(LOCAL_NH2) + "h/" + String(LOCAL_KV2) + "kv, "
          + String(pools[0].capacity) + " cores")

    # Per-node arenas and state
    var perf_arenas = HeapMoveArray[NumaArena[]](NUM_NODES)
    for i in range(NUM_NODES):
        perf_arenas.push(NumaArena[](topo[i], 512 * 1024 * 1024))

    var q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var k_bases = InlineArray[Int, NUM_NODES](fill=0)
    var v_bases = InlineArray[Int, NUM_NODES](fill=0)
    var qi_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var sc_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var cos_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var sin_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var scratch_ptrs = InlineArray[Int, NUM_NODES](fill=0)

    comptime CS2 = Slot[F32, Replicated, MAX_SEQ2, HALF2, 1]
    comptime SS2 = Slot[F32, Replicated, MAX_SEQ2, HALF2, 1]

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
                k_node.write_head(t, lg, hb, Float32(0.05))
                for d in range(HD2):
                    hb[d] = Scalar[DType.int8]((t * 11 + global_g * HD2 + d * 5) % 251 - 125)
                v_node.write_head_transposed(t, lg, hb, Float32(0.05))

        qi_ptrs[node] = Int(perf_arenas[node].alloc[Scalar[DType.int8]](MAX_SL * LOCAL_HIDDEN2))
        sc_ptrs[node] = Int(perf_arenas[node].alloc[Float32](MAX_SL))
        scratch_ptrs[node] = Int(perf_arenas[node].alloc[UInt8](local_scratch_bytes))

    comptime QS2 = Slot[BF16, Replicated, 1, LOCAL_HIDDEN2, 1]
    comptime QI2 = Slot[I8, Replicated, 1, LOCAL_HIDDEN2, 1]
    comptime SC2 = Slot[F32, Replicated, 1, 1, 1]

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

            # Warmup
            for warmup in range(2):
                @parameter
                def wu[node: Int]() -> PoolFence:
                    return int8_gqa_attention_amx_prefill[LOCAL_NH2, LOCAL_KV2, HD2](
                        DynView[QS2](q_ptrs[node], sl),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        DynView[QI2](qi_ptrs[node], sl), DynView[SC2](sc_ptrs[node], sl),
                        Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos, pool_ptrs[node][],
                    )
                parallel_for[NUM_NODES, wu]()

            # Measure
            var best = Int(1 << 60)
            for trial in range(5):
                var t0 = Int(perf_counter_ns())
                @parameter
                def run[node: Int]() -> PoolFence:
                    return int8_gqa_attention_amx_prefill[LOCAL_NH2, LOCAL_KV2, HD2](
                        DynView[QS2](q_ptrs[node], sl),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        DynView[QI2](qi_ptrs[node], sl), DynView[SC2](sc_ptrs[node], sl),
                        Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos, pool_ptrs[node][],
                    )
                parallel_for[NUM_NODES, run]()
                var wall = Int(perf_counter_ns()) - t0
                keep(UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_ptrs[0])[0])
                if wall < best: best = wall

            var total_us = best // 1000
            var per_pos = total_us // sl if sl > 0 else 0
            print("  " + String(ctx_pos) + " | " + String(sl)
                  + " | " + String(total_us)
                  + " | " + String(per_pos))
