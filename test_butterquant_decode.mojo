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
from experimental2.attn_amx_decode import decode, decode_merge, scratch_bytes, collect_profiles, per_worker_merge_bytes, DecodeWorkerScratch
from experimental2.helpers import AttnCtx
from experimental.hadquant_impl import fwht_block
from experimental2.kv_cache import KVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt, exp_f32, quantize_i8, roundeven
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from numa import NumaInfo, NumaTopology
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from kernels.kernel_ops import init_rope_tables, parallel_for, timed_parallel_for, ParallelTiming, PoolFence


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
    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](NUM_NODES)
        for i in range(NUM_NODES):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_test(numa, topo, pools)
        _ = pools
    else:
        print("mode: cold")
        var pools = HeapMoveArray[BurstPool[]](NUM_NODES)
        for i in range(NUM_NODES):
            pools.push(BurstPool[].for_topology(numa, topo[i], stack_size=2 * 1024 * 1024))
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_test(numa, topo, pools)
        _ = pools
    _ = topo
    _ = numa


def run_test[P: BurstThreadPool](numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P]):


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
    var ref_exp_sum_h0 = Float32(0)
    var ref_raw_i32_d0 = Int32(0)
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
        var ref_raw_d0_acc = Int32(0)
        for t in range(context):
            var w_u8 = UInt8(max(min(Int(scores[t] * 255.0 + 0.5), 255), 0))
            var v_data = kv_cache.v_head(t, g)
            if h == 0:
                ref_raw_d0_acc += Int32(w_u8) * Int32(v_data[0])
            for d in range(HD):
                var v_i8 = Int32(v_data[d])
                expected[h * HD + d] += Float32(Int32(w_u8) * v_i8) * vagg_scale
        if h == 0:
            ref_exp_sum_h0 = exp_sum
            ref_raw_i32_d0 = ref_raw_d0_acc
        for d in range(HD):
            expected[h * HD + d] /= exp_sum

        corr_arena.reset_to(scores_mark)

    # Quantize expected f32 → expected i8 (same as fused kernel epilogue)
    var qi_inv = Float32(127) / v_layer_scale
    var expected_i8 = corr_arena.alloc[Scalar[DType.int8]](HIDDEN)
    for d in range(HIDDEN):
        var qf = roundeven[DType.float32, 1](expected[d] * qi_inv)
        var q = Int(min(max(qf, Float32(-128)), Float32(127)))
        expected_i8[d] = Scalar[DType.int8](q)

    # Run kernel
    decode[NH, NKV, HD](
        DynView[QSlot](Int(q_bf16), 1),
        kv_cache, kv_cache,
        Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
        Int(scratch), POS,
        q_layer_scale, k_layer_scale, v_layer_scale,
        1,
        pools[0],
    ).join()

    # === DIAGNOSTICS: compare kernel intermediates vs reference ===
    var diag_woff = HIDDEN * size_of[Scalar[DType.int8]]() + size_of[AttnCtx]()
    var diag_ws0 = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
        unsafe_from_address=Int(scratch) + diag_woff)
    var kern_running_o0 = diag_ws0[].running_o[0]
    var kern_running_l0 = diag_ws0[].running_l[0]
    var kern_running_m0 = diag_ws0[].running_m[0]
    print("\n--- DIAGNOSTICS (head 0, element 0) ---")
    print("kernel: running_o[0]=" + String(kern_running_o0)
        + "  running_l[0]=" + String(kern_running_l0)
        + "  running_m[0]=" + String(kern_running_m0))
    print("ref:    raw_i32_d0=" + String(ref_raw_i32_d0)
        + "  exp_sum_h0=" + String(ref_exp_sum_h0))
    print("ratio:  running_o/raw=" + String(kern_running_o0 / Float32(ref_raw_i32_d0))
        + "  l/exp_sum=" + String(kern_running_l0 / ref_exp_sum_h0))
    var corr_vagg_scale = v_layer_scale / (Float32(255.0) * Float32(127.0))
    var diag_merge_f32 = kern_running_o0 * corr_vagg_scale / kern_running_l0
    print("merge f32=" + String(diag_merge_f32) + "  expected f32=" + String(expected[0]))
    print("merge f32 ratio=" + String(diag_merge_f32 / expected[0]))
    print("--- END DIAGNOSTICS ---\n")

    decode_merge[NH, NKV, HD](Int(scratch), 1, corr_vagg_scale)

    # Compare i8 kernel output vs i8-quantized reference
    var ri8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=Int(scratch))
    var err_sum = Float64(0)
    var max_err = 0
    var count = 0
    for d in range(min(5, HIDDEN)):
        var got = Int(ri8[d])
        var exp_val = Int(expected_i8[d])
        print("[" + String(d) + "] got_i8=" + String(got) + " exp_i8=" + String(exp_val)
            + " exp_f32=" + String(expected[d]))
    for d in range(HIDDEN):
        var got = Int(ri8[d])
        var exp_val = Int(expected_i8[d])
        var e = got - exp_val
        if e < 0: e = -e
        err_sum += Float64(e)
        if e > max_err: max_err = e
        count += 1
    var avg_err = err_sum / Float64(count)
    print("avg_i8_err: " + String(avg_err) + " max_i8_err: " + String(max_err))
    if avg_err < 2.0 and max_err <= 3:
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
          + String(pools[0].get_capacity()) + " cores")

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

    # Max workers per group across all nodes (limited by smallest pool)
    var max_wpg = pools[0].get_capacity() // LOCAL_KV2
    for i in range(1, NUM_NODES):
        var nw = pools[i].get_capacity() // LOCAL_KV2
        if nw < max_wpg:
            max_wpg = nw

    # -----------------------------------------------------------------
    # Worker scaling sweep at ctx=32000
    # -----------------------------------------------------------------
    print("\n--- Worker scaling at ctx=32000 ---")
    var ctx_pos_scale = 32000

    var worker_counts = InlineArray[Int, 6](fill=0)
    worker_counts[0] = 2
    worker_counts[1] = 4
    worker_counts[2] = 8
    worker_counts[3] = 16
    worker_counts[4] = max_wpg * LOCAL_KV2
    # Deduplicate last entry if it matches
    var num_counts = 5
    if worker_counts[4] == worker_counts[3]:
        num_counts = 4

    for wi in range(num_counts):
        var total_workers = worker_counts[wi]
        var wpg = total_workers // LOCAL_KV2

        for _ in range(2):
            @parameter
            def wu_s[node: Int]() -> PoolFence[P]:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos_scale,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    wpg,
                    pools[node],
                )
            parallel_for[P, NUM_NODES, wu_s]()
            decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], wpg, perf_vagg_scale)

        var best = Int(1 << 60)
        var best_disp = 0
        var best_join = 0
        var best_join_oh = 0
        var best_merge = 0
        var best_kernel_max = 0
        var best_done_spread = 0
        var best_per_node = InlineArray[Int, NUM_NODES](fill=0)
        for trial in range(5):
            @parameter
            def run_s[node: Int]() -> PoolFence[P]:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos_scale,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    wpg,
                    pools[node],
                )
            var timing = timed_parallel_for[P, NUM_NODES, run_s]()
            var join_end = Int(perf_counter_ns())
            # True join overhead: time from last worker finishing to join returning
            var max_done_ts = 0
            var min_done_ts = Int(1 << 60)
            var global_kernel_max = 0
            var per_node_max = InlineArray[Int, NUM_NODES](fill=0)
            for node in range(NUM_NODES):
                var node_agg = collect_profiles[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], wpg)
                if node_agg.max_done_timestamp > max_done_ts:
                    max_done_ts = node_agg.max_done_timestamp
                if node_agg.max_done_timestamp < min_done_ts:
                    min_done_ts = node_agg.max_done_timestamp
                if node_agg.max_total > global_kernel_max:
                    global_kernel_max = node_agg.max_total
                per_node_max[node] = node_agg.max_total
            var join_oh = join_end - max_done_ts
            var done_spread = max_done_ts - min_done_ts
            var tm0 = Int(perf_counter_ns())
            for node in range(NUM_NODES):
                decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], wpg, perf_vagg_scale)
            var merge_ns = Int(perf_counter_ns()) - tm0
            var wall = timing.total_ns + merge_ns
            keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
            if wall < best:
                best = wall
                best_disp = timing.dispatch_ns
                best_join = timing.join_ns
                best_join_oh = join_oh
                best_merge = merge_ns
                best_kernel_max = global_kernel_max
                best_done_spread = done_spread
                for node in range(NUM_NODES):
                    best_per_node[node] = per_node_max[node]

        print("  " + String(total_workers) + "w/node"
            + "  wall=" + String(best // 1000) + " us"
            + "  disp=" + String(best_disp // 1000) + " us"
            + "  kernel=" + String(best_kernel_max // 1000) + " us"
            + "  join_oh=" + String(best_join_oh) + " ns"
            + "  merge=" + String(best_merge // 1000) + " us"
            + "  done_spread=" + String(best_done_spread // 1000) + " us")
        print("    per-node kernel(max): n0=" + String(best_per_node[0] // 1000)
            + " n1=" + String(best_per_node[1] // 1000)
            + " n2=" + String(best_per_node[2] // 1000)
            + " n3=" + String(best_per_node[3] // 1000) + " us")
        var unaccounted = best_join - best_kernel_max
        print("    unaccounted: " + String(unaccounted // 1000)
            + " us (join - global_kernel_max)")
        var agg = collect_profiles[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], wpg)
        agg.print_summary()

    # -----------------------------------------------------------------
    # Context sweep at max workers (wpg=16)
    # -----------------------------------------------------------------
    var sweep_workers = max_wpg * LOCAL_KV2
    print("\n--- Context sweep (" + String(sweep_workers) + "w/node) ---")
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
            def wu[node: Int]() -> PoolFence[P]:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    max_wpg,
                    pools[node],
                )
            parallel_for[P, NUM_NODES, wu]()
            decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[0], max_wpg, perf_vagg_scale)

        var best = Int(1 << 60)
        var best_join_oh = 0
        var best_kernel_max = 0
        var best_unaccounted = 0
        for trial in range(5):
            @parameter
            def run[node: Int]() -> PoolFence[P]:
                return decode[LOCAL_NH2, LOCAL_KV2, HD2](
                    DynView[QS2](q_ptrs[node], 1),
                    LOCAL_KVC(kv_bases[node]), LOCAL_KVC(kv_bases[node]),
                    Bound[CS2](cos_ptrs[node]), Bound[SS2](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    perf_q_scale, perf_k_scale, perf_v_scale,
                    max_wpg,
                    pools[node],
                )
            var timing = timed_parallel_for[P, NUM_NODES, run]()
            var join_end = Int(perf_counter_ns())
            var max_done_ts = 0
            var global_km = 0
            for node in range(NUM_NODES):
                var node_agg = collect_profiles[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], max_wpg)
                if node_agg.max_done_timestamp > max_done_ts:
                    max_done_ts = node_agg.max_done_timestamp
                if node_agg.max_total > global_km:
                    global_km = node_agg.max_total
            var join_oh = join_end - max_done_ts
            var tm0 = Int(perf_counter_ns())
            for node in range(NUM_NODES):
                decode_merge[LOCAL_NH2, LOCAL_KV2, HD2](scratch_ptrs[node], max_wpg, perf_vagg_scale)
            var merge_ns = Int(perf_counter_ns()) - tm0
            var wall = timing.total_ns + merge_ns
            keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
            if wall < best:
                best = wall
                best_join_oh = join_oh
                best_kernel_max = global_km
                best_unaccounted = timing.join_ns - global_km

        print("  ctx=" + String(ctx_pos) + "  wall=" + String(best // 1000) + " us"
            + "  kernel=" + String(best_kernel_max // 1000) + " us"
            + "  join_oh=" + String(best_join_oh) + " ns"
            + "  unaccounted=" + String(best_unaccounted // 1000) + " us")

    _ = corr_arena
    _ = perf_arenas
