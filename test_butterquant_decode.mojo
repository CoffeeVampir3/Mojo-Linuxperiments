"""ButterQuant AMX decode attention — correctness + performance.

Zero per-token scales. KV cache is flat u8. Fixed layer-level scales.
Specialized for SL=1 (single token decode).
"""

from std.sys.info import simd_width_of, size_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental2.attn_amx_decode import decode, decode_merge, scratch_bytes, collect_profiles
from experimental.hadquant_impl import fwht_block
from experimental2.kv_cache import KVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32, quantize_i8
from threading import make_node_pools
from numa import NumaInfo
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
    var pools = make_node_pools(numa, stack_size=2 * 1024 * 1024)
    for i in range(NUM_NODES):
        print("  node " + String(topo[i]) + ": " + String(pools[topo[i]].capacity) + " workers")


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
    comptime POS = 48

    var q_layer_scale = Float32(0.15)
    var k_layer_scale = Float32(0.15)
    var v_layer_scale = Float32(0.15)
    var q_quant_inv = Float32(127.0) / q_layer_scale

    print("\n=== Correctness: " + String(NH) + "h/" + String(NKV)
          + "kv, hd=" + String(HD) + ", pos=" + String(POS) + " ===")

    var corr_arena = NumaArena[](topo[0], 256 * 1024 * 1024)

    comptime KVC = KVCache[MAX_SEQ, HD, NKV]
    var corr_scratch_bytes = scratch_bytes[NH, NKV, HD](NKV)

    var q_bf16 = corr_arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    var kv_mem = corr_arena.alloc[UInt8](KVC.TOTAL_BYTES)
    var kv_cache = KVC(Int(kv_mem))
    var cos_tab = corr_arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_tab = corr_arena.alloc[Float32](MAX_SEQ * HALF)
    var scratch = corr_arena.alloc[UInt8](corr_scratch_bytes)
    var expected = corr_arena.alloc[Float32](HIDDEN)
    var q_head = corr_arena.alloc[Float32](HD)
    var head_buf = corr_arena.alloc[Scalar[DType.int8]](HD)

    # Fill Q (single row)
    for k in range(HIDDEN):
        q_bf16[k] = Scalar[DType.bfloat16](Float32(k + 1) / Float32(HIDDEN) - 0.5)

    # Fill KV cache with fixed-scale quantized data
    for t in range(POS + 1):
        for g in range(NKV):
            for d in range(HD):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
            kv_cache.write_k(t, g, head_buf)
            for d in range(HD):
                var val = Float32((t * 11 + g * 17 + d * 5) % 251 - 125) / 125.0
                head_buf[d] = Scalar[DType.int8](Int8(max(min(Int(val * 127.0), 127), -128)))
            kv_cache.write_v(t, g, head_buf)

    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)), Float64(100000.0))

    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]

    # F32 reference: same pipeline as the kernel, in scalar
    var inv_sqrt_hd = Float32(1.0) / sqrt[DType.float32, 1](Float32(HD))
    var context = POS + 1
    for h in range(NH):
        var g = h // GQA
        # Q: bf16 → f32 → RoPE → FWHT → fixed-scale i8
        for d in range(HD):
            q_head[d] = Float32(q_bf16[h * HD + d])
        for j in range(HALF):
            var x_lo = q_head[j]
            var x_hi = q_head[HALF + j]
            var cv = cos_tab[POS * HALF + j]
            var sv = sin_tab[POS * HALF + j]
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

        # Score against K (recompute u8 from test data formula)
        var scores_mark = corr_arena.mark()
        var scores = corr_arena.alloc[Float32](context)
        for t in range(context):
            var dot = Int32(0)
            for d in range(HD):
                var val = Float32((t * 7 + g * 13 + d * 3) % 251 - 125) / 125.0
                var ki8 = Int8(max(min(Int(val * 127.0), 127), -128))
                var ku8 = UInt8(Int(ki8) + 128)  # XOR 0x80
                dot += Int32(qi_ref[d]) * Int32(ku8)
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
            expected[h * HD + d] = Float32(0)
        for t in range(context):
            var w_u8 = UInt8(max(min(Int(scores[t] * 255.0 + 0.5), 255), 0))
            var v_data = kv_cache.v_head(t, g)
            for d in range(HD):
                var v_i8 = Int32(v_data[d])
                expected[h * HD + d] += Float32(Int32(w_u8) * v_i8) * vagg_scale
        for d in range(HD):
            expected[h * HD + d] /= exp_sum

        corr_arena.reset_to(scores_mark)

    # Run kernel
    decode[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), 1),
        kv_cache, kv_cache,
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(scratch), POS,
        q_layer_scale, k_layer_scale, v_layer_scale,
        1,
        pools[topo[0]],
    ).join()
    var corr_vagg_scale = v_layer_scale / (Float32(255.0) * Float32(127.0))
    decode_merge[NH, NKV, HD](Int(scratch), 1, corr_vagg_scale)

    # Compare
    var rf32 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scratch))
    var err_sum = Float64(0)
    var count = 0
    for d in range(HIDDEN):
        var got = Float64(rf32[d])
        var rv = Float64(expected[d])
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
    comptime MAX_SEQ2 = 33000

    comptime LOCAL_KV2 = NKV2 // NUM_NODES
    comptime LOCAL_NH2 = NH2 // NUM_NODES
    comptime LOCAL_HIDDEN2 = LOCAL_NH2 * HD2

    comptime LOCAL_KVC = KVCache[MAX_SEQ2, HD2, LOCAL_KV2]
    # Size scratch for max workers: up to 16 per group × 2 groups = 32
    comptime MAX_WPG = 16
    var local_scratch_bytes = scratch_bytes[LOCAL_NH2, LOCAL_KV2, HD2](LOCAL_KV2 * MAX_WPG)

    print("\n=== Performance: 128h/8kv, hd=128, " + String(NUM_NODES) + " NUMA nodes ===")
    print("  Per node: " + String(LOCAL_NH2) + "h/" + String(LOCAL_KV2) + "kv, "
          + String(pools[topo[0]].capacity) + " cores")

    var perf_arenas = HeapMoveArray[NumaArena[]](NUM_NODES)
    for i in range(NUM_NODES):
        perf_arenas.push(NumaArena[](topo[i], 1024 * 1024 * 1024))

    var q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var kv_bases = InlineArray[Int, NUM_NODES](fill=0)
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

        var q_node = perf_arenas[node].alloc[Scalar[DType.bfloat16]](LOCAL_HIDDEN2)
        for i in range(LOCAL_HIDDEN2):
            q_node[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)
        q_ptrs[node] = Int(q_node)

        var kv_mem = perf_arenas[node].alloc[UInt8](LOCAL_KVC.TOTAL_BYTES)
        kv_bases[node] = Int(kv_mem)
        var kv_node = LOCAL_KVC(Int(kv_mem))
        var hb = perf_arenas[node].alloc[Scalar[DType.int8]](HD2)
        for t in range(MAX_SEQ2):
            for lg in range(LOCAL_KV2):
                var global_g = node * LOCAL_KV2 + lg
                for d in range(HD2):
                    hb[d] = Scalar[DType.int8]((t * 7 + global_g * HD2 + d * 3) % 251 - 125)
                kv_node.write_k(t, lg, hb)
                for d in range(HD2):
                    hb[d] = Scalar[DType.int8]((t * 11 + global_g * HD2 + d * 5) % 251 - 125)
                kv_node.write_v(t, lg, hb)

        scratch_ptrs[node] = Int(perf_arenas[node].alloc[UInt8](local_scratch_bytes))

    comptime QS2 = Slot[BF16, Replicated, 1, LOCAL_HIDDEN2, 1]

    var perf_vagg_scale = perf_v_scale / (Float32(255.0) * Float32(127.0))

    # -----------------------------------------------------------------
    # Worker scaling sweep at ctx=32000
    # -----------------------------------------------------------------
    print("\n--- Worker scaling at ctx=32000 ---")
    var ctx_pos_scale = 32000

    for wpg_idx in range(5):
        var wpg = 1
        if wpg_idx == 1: wpg = 2
        if wpg_idx == 2: wpg = 4
        if wpg_idx == 3: wpg = 8
        if wpg_idx == 4: wpg = 16

        for _ in range(2):
            @parameter
            def wu_s[node: Int]() -> PoolFence:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos_scale,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    wpg,
                    pools[topo[node]],
                )
            parallel_for[NUM_NODES, wu_s]()
            decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], wpg, perf_vagg_scale)

        var best = Int(1 << 60)
        for trial in range(5):
            var t0 = Int(perf_counter_ns())
            @parameter
            def run_s[node: Int]() -> PoolFence:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos_scale,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    wpg,
                    pools[topo[node]],
                )
            parallel_for[NUM_NODES, run_s]()
            for node in range(NUM_NODES):
                decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], wpg, perf_vagg_scale)
            var wall = Int(perf_counter_ns()) - t0
            keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
            if wall < best: best = wall

        var total_us = best // 1000
        var total_workers = LOCAL_KV2 * wpg
        print("  wpg=" + String(wpg) + " (" + String(total_workers) + "w/node)  wall=" + String(total_us) + " us")
        var agg = collect_profiles[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], wpg)
        agg.print_summary()

    # -----------------------------------------------------------------
    # Context sweep at max workers (wpg=16)
    # -----------------------------------------------------------------
    print("\n--- Context sweep (wpg=16) ---")
    for ctx_idx in range(7):
        var ctx_pos = 128
        if ctx_idx == 1: ctx_pos = 512
        if ctx_idx == 2: ctx_pos = 2048
        if ctx_idx == 3: ctx_pos = 4096
        if ctx_idx == 4: ctx_pos = 8000
        if ctx_idx == 5: ctx_pos = 16000
        if ctx_idx == 6: ctx_pos = 32000

        for _ in range(2):
            @parameter
            def wu[node: Int]() -> PoolFence:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    MAX_WPG,
                    pools[topo[node]],
                )
            parallel_for[NUM_NODES, wu]()
            decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], MAX_WPG, perf_vagg_scale)

        var best = Int(1 << 60)
        for trial in range(5):
            var t0 = Int(perf_counter_ns())
            @parameter
            def run[node: Int]() -> PoolFence:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    MAX_WPG,
                    pools[topo[node]],
                )
            parallel_for[NUM_NODES, run]()
            for node in range(NUM_NODES):
                decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], MAX_WPG, perf_vagg_scale)
            var wall = Int(perf_counter_ns()) - t0
            keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
            if wall < best: best = wall

        var total_us = best // 1000
        print("  ctx=" + String(ctx_pos) + "  wall=" + String(total_us) + " us")
