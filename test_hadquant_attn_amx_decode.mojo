"""Decode attention: wpg sweep with new BurstPool (mailbox + hot mode)."""

from std.sys.info import simd_width_of, size_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental2.attn_amx_decode_control import (
    decode as decode_cold,
    decode_hot,
    scratch_bytes as control_scratch_bytes,
)
from experimental2.attn_amx_decode import (
    decode as decode_chunked,
    scratch_bytes as chunked_scratch_bytes,
)
from experimental2.kv_cache import KVCache
from experimental.amx import init_intel_amx
from simd_math import sqrt
from threading import BurstPool
from numa import NumaInfo
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from kernels.kernel_ops import init_rope_tables, parallel_for, PoolFence


comptime NUM_NODES = 4
comptime HEADROOM = 2


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    var numa = NumaInfo()
    if numa.num_nodes < NUM_NODES:
        print("SKIP: need " + String(NUM_NODES) + " NUMA nodes")
        return
    var topo = numa.plan_topology(NUM_NODES)

    print("NUMA: " + String(NUM_NODES) + " nodes")
    for i in range(NUM_NODES):
        print("  node " + String(topo[i]) + ": " + String(numa.cpus_on_node(topo[i])) + " cpus")

    # Hot pools with headroom
    var pools = HeapMoveArray[BurstPool[]](NUM_NODES)
    for i in range(NUM_NODES):
        pools.push(BurstPool[].for_numa_node(numa, topo[i], HEADROOM))
    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], NUM_NODES](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for i in range(NUM_NODES):
        pool_ptrs[i] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[i])))

    var cap = pools[0].capacity
    print("Per-node: " + String(cap) + " workers (headroom=" + String(HEADROOM) + ")")

    comptime HD = 128
    comptime NH = 128
    comptime NKV = 8
    comptime HALF = HD // 2
    comptime MAX_SEQ = 16384

    comptime LOCAL_KV = NKV // NUM_NODES
    comptime LOCAL_NH = NH // NUM_NODES
    comptime LOCAL_HIDDEN = LOCAL_NH * HD

    comptime LOCAL_KVC = KVCache[MAX_SEQ, HD, LOCAL_KV]

    # Scratch: sized for max wpg (full pool capacity)
    var ctrl_sz = control_scratch_bytes[LOCAL_NH, LOCAL_KV, HD]()
    var chunk_sz = chunked_scratch_bytes[LOCAL_NH, LOCAL_KV, HD](cap)
    var max_scratch = max(ctrl_sz, chunk_sz)

    print("Config: " + String(NH) + "h/" + String(NKV) + "kv, hd=" + String(HD))
    print("Per node: " + String(LOCAL_NH) + "h/" + String(LOCAL_KV) + "kv")

    var perf_arenas = HeapMoveArray[NumaArena[]](NUM_NODES)
    for i in range(NUM_NODES):
        perf_arenas.push(NumaArena[](topo[i], 1024 * 1024 * 1024))

    var q_scale = Float32(0.15)
    var k_scale = Float32(0.15)
    var v_scale = Float32(0.15)

    var q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var k_bases = InlineArray[Int, NUM_NODES](fill=0)
    var v_bases = InlineArray[Int, NUM_NODES](fill=0)
    var cos_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var sin_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var scratch_ptrs = InlineArray[Int, NUM_NODES](fill=0)

    comptime CS = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SS = Slot[F32, Replicated, MAX_SEQ, HALF, 1]

    for node in range(NUM_NODES):
        var cos_n = perf_arenas[node].alloc[Float32](MAX_SEQ * HALF)
        var sin_n = perf_arenas[node].alloc[Float32](MAX_SEQ * HALF)
        init_rope_tables(Bound[CS](Int(cos_n)), Bound[SS](Int(sin_n)), Float64(100000.0))
        cos_ptrs[node] = Int(cos_n)
        sin_ptrs[node] = Int(sin_n)

        var q_n = perf_arenas[node].alloc[Scalar[DType.bfloat16]](LOCAL_HIDDEN)
        for i in range(LOCAL_HIDDEN):
            q_n[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)
        q_ptrs[node] = Int(q_n)

        var kv_mem = perf_arenas[node].alloc[UInt8](2 * LOCAL_KVC.TOTAL_BYTES)
        k_bases[node] = Int(kv_mem)
        v_bases[node] = Int(kv_mem) + LOCAL_KVC.TOTAL_BYTES
        var k_n = LOCAL_KVC(Int(kv_mem))
        var v_n = LOCAL_KVC(Int(kv_mem) + LOCAL_KVC.TOTAL_BYTES)
        var hb = perf_arenas[node].alloc[Scalar[DType.int8]](HD)
        for t in range(MAX_SEQ):
            for lg in range(LOCAL_KV):
                for d in range(HD):
                    hb[d] = Scalar[DType.int8]((t * 7 + (node * LOCAL_KV + lg) * HD + d * 3) % 251 - 125)
                k_n.write_k(t, lg, hb)
                for d in range(HD):
                    hb[d] = Scalar[DType.int8]((t * 11 + (node * LOCAL_KV + lg) * HD + d * 5) % 251 - 125)
                v_n.write_v(t, lg, hb)

        scratch_ptrs[node] = Int(perf_arenas[node].alloc[UInt8](max_scratch))

    comptime QS = Slot[BF16, Replicated, 1, LOCAL_HIDDEN, 1]

    # =====================================================================
    # Control (wpg=1, no context split) with hot mode
    # =====================================================================
    print("\n=== Control (wpg=1, hot mode) ===")
    print("  context | us")
    print("  --------|----")

    var ctx_values = InlineArray[Int, 4](fill=0)
    ctx_values[0] = 512
    ctx_values[1] = 2048
    ctx_values[2] = 4096
    ctx_values[3] = 16384

    for ci in range(4):
        var ctx_pos = ctx_values[ci]

        # Warmup
        for node in range(NUM_NODES):
            pool_ptrs[node][].begin_forward()
        for warmup in range(3):
            @parameter
            def wu_ctrl[node: Int]() -> PoolFence:
                decode_hot[LOCAL_NH, LOCAL_KV, HD](
                    DynView[QS](q_ptrs[node], 1),
                    LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                    Bound[CS](cos_ptrs[node]), Bound[SS](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    q_scale, k_scale, v_scale,
                    pool_ptrs[node][],
                )
                return PoolFence.completed()
            parallel_for[NUM_NODES, wu_ctrl]()

        var best = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            @parameter
            def run_ctrl[node: Int]() -> PoolFence:
                decode_hot[LOCAL_NH, LOCAL_KV, HD](
                    DynView[QS](q_ptrs[node], 1),
                    LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                    Bound[CS](cos_ptrs[node]), Bound[SS](sin_ptrs[node]),
                    scratch_ptrs[node], ctx_pos,
                    q_scale, k_scale, v_scale,
                    pool_ptrs[node][],
                )
                return PoolFence.completed()
            parallel_for[NUM_NODES, run_ctrl]()
            var e = Int(perf_counter_ns()) - t0
            keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
            if e < best: best = e

        for node in range(NUM_NODES):
            pool_ptrs[node][].end_forward()
        print("  " + String(ctx_pos) + " | " + String(best // 1000))

    # =====================================================================
    # Chunked: sweep wpg with hot mode
    # =====================================================================
    print("\n=== Chunked (hot mode) wpg sweep ===")
    print("  context | wpg=2  | wpg=4  | wpg=8  | wpg=15")
    print("  --------|--------|--------|--------|-------")

    var wpg_values = InlineArray[Int, 4](fill=0)
    wpg_values[0] = 2
    wpg_values[1] = 4
    wpg_values[2] = 8
    wpg_values[3] = 15

    for ci in range(4):
        var ctx_pos = ctx_values[ci]
        var results = InlineArray[Int, 4](fill=0)

        for wi in range(4):
            var wpg = wpg_values[wi]

            for node in range(NUM_NODES):
                pool_ptrs[node][].begin_forward()

            # Warmup
            for warmup in range(3):
                @parameter
                def wu_chunk[node: Int]() -> PoolFence:
                    return decode_chunked[LOCAL_NH, LOCAL_KV, HD](
                        DynView[QS](q_ptrs[node], 1),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        Bound[CS](cos_ptrs[node]), Bound[SS](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos,
                        q_scale, k_scale, v_scale,
                        pool_ptrs[node][], wpg,
                    )
                parallel_for[NUM_NODES, wu_chunk]()

            var best = Int(1 << 60)
            for trial in range(10):
                var t0 = Int(perf_counter_ns())
                @parameter
                def run_chunk[node: Int]() -> PoolFence:
                    return decode_chunked[LOCAL_NH, LOCAL_KV, HD](
                        DynView[QS](q_ptrs[node], 1),
                        LOCAL_KVC(k_bases[node]), LOCAL_KVC(v_bases[node]),
                        Bound[CS](cos_ptrs[node]), Bound[SS](sin_ptrs[node]),
                        scratch_ptrs[node], ctx_pos,
                        q_scale, k_scale, v_scale,
                        pool_ptrs[node][], wpg,
                    )
                parallel_for[NUM_NODES, run_chunk]()
                var e = Int(perf_counter_ns()) - t0
                keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=scratch_ptrs[0])[0])
                if e < best: best = e

            for node in range(NUM_NODES):
                pool_ptrs[node][].end_forward()
            results[wi] = best // 1000

        print("  " + String(ctx_pos)
              + " | " + String(results[0])
              + " | " + String(results[1])
              + " | " + String(results[2])
              + " | " + String(results[3]))
