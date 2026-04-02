"""ButterQuant AMX decode attention — profiled with hot-mode BurstPool.

Runs control (wpg=1) and best chunked configs with per-phase profiling
to identify optimization targets now that dispatch overhead is eliminated.
"""

from std.sys.info import simd_width_of, size_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental2.attn_amx_decode_control import (
    decode_hot,
    scratch_bytes as control_scratch_bytes,
    collect_profiles as control_collect_profiles,
)
from experimental2.attn_amx_decode import (
    decode as decode_chunked,
    scratch_bytes as chunked_scratch_bytes,
    collect_profiles as chunk_collect_profiles,
    read_caller_profile as chunk_read_caller,
)
from experimental2.kv_cache import KVCache
from experimental2.kernel_profile import ProfileAggregator
from experimental.amx import init_intel_amx
from simd_math import sqrt
from threading import BurstPool
from numa import NumaInfo
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from kernels.kernel_ops import init_rope_tables


comptime HEADROOM = 2


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    var numa = NumaInfo()
    var node = numa.plan_topology(1)[0]
    var ncpus = numa.cpus_on_node(node)
    print("Node " + String(node) + ": " + String(ncpus) + " cpus")

    var pool = BurstPool[].for_numa_node(numa, node, HEADROOM)
    if not pool:
        print("pool creation failed")
        return
    var cap = pool.capacity
    print("Pool: " + String(cap) + " workers (headroom=" + String(HEADROOM) + ")")

    comptime HD = 128
    comptime NH = 32
    comptime NKV = 2
    comptime HALF = HD // 2
    comptime MAX_SEQ = 16384
    comptime HIDDEN = NH * HD

    comptime KVC = KVCache[MAX_SEQ, HD, NKV]
    var ctrl_sz = control_scratch_bytes[NH, NKV, HD]()
    var chunk_sz = chunked_scratch_bytes[NH, NKV, HD](cap)
    var max_scratch = max(ctrl_sz, chunk_sz)

    print("Config: " + String(NH) + "h/" + String(NKV) + "kv, hd=" + String(HD))

    var arena = NumaArena[](node, 1024 * 1024 * 1024)

    var q_scale = Float32(0.15)
    var k_scale = Float32(0.15)
    var v_scale = Float32(0.15)

    var q_bf16 = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    for i in range(HIDDEN):
        q_bf16[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)

    var kv_mem = arena.alloc[UInt8](2 * KVC.TOTAL_BYTES)
    var k_cache = KVC(Int(kv_mem))
    var v_cache = KVC(Int(kv_mem) + KVC.TOTAL_BYTES)
    var hb = arena.alloc[Scalar[DType.int8]](HD)
    for t in range(MAX_SEQ):
        for g in range(NKV):
            for d in range(HD):
                hb[d] = Scalar[DType.int8]((t * 7 + g * HD + d * 3) % 251 - 125)
            k_cache.write_k(t, g, hb)
            for d in range(HD):
                hb[d] = Scalar[DType.int8]((t * 11 + g * HD + d * 5) % 251 - 125)
            v_cache.write_v(t, g, hb)

    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    var cos_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    init_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)), Float64(100000.0))

    var scratch = arena.alloc[UInt8](max_scratch)

    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]

    var ctx_values = InlineArray[Int, 4](fill=0)
    ctx_values[0] = 512
    ctx_values[1] = 2048
    ctx_values[2] = 4096
    ctx_values[3] = 16384

    # =====================================================================
    # Timing sweep: control vs chunked wpg=4,8
    # =====================================================================
    print("\n=== Timing (hot mode, single node) ===")
    print("  context | control | wpg=4 | wpg=8 | wpg=15")
    print("  --------|---------|-------|-------|-------")

    for ci in range(4):
        var ctx_pos = ctx_values[ci]

        # Control
        pool.begin_forward()
        var best_ctrl = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            decode_hot[NH, NKV, HD](
                DynView[QSlot](Int(q_bf16), 1),
                k_cache, v_cache,
                Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                Int(scratch), ctx_pos,
                q_scale, k_scale, v_scale, pool,
            )
            var e = Int(perf_counter_ns()) - t0
            keep(scratch[0])
            if e < best_ctrl: best_ctrl = e
        pool.end_forward()

        # wpg=4
        pool.begin_forward()
        var best_4 = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            decode_chunked[NH, NKV, HD](
                DynView[QSlot](Int(q_bf16), 1),
                k_cache, v_cache,
                Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                Int(scratch), ctx_pos,
                q_scale, k_scale, v_scale, pool, 4,
            ).join()
            var e = Int(perf_counter_ns()) - t0
            keep(scratch[0])
            if e < best_4: best_4 = e
        pool.end_forward()

        # wpg=8
        pool.begin_forward()
        var best_8 = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            decode_chunked[NH, NKV, HD](
                DynView[QSlot](Int(q_bf16), 1),
                k_cache, v_cache,
                Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                Int(scratch), ctx_pos,
                q_scale, k_scale, v_scale, pool, 8,
            ).join()
            var e = Int(perf_counter_ns()) - t0
            keep(scratch[0])
            if e < best_8: best_8 = e
        pool.end_forward()

        # wpg=15 (full pool utilization: 15 * 2 KV heads = 30 jobs)
        pool.begin_forward()
        var best_15 = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            decode_chunked[NH, NKV, HD](
                DynView[QSlot](Int(q_bf16), 1),
                k_cache, v_cache,
                Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                Int(scratch), ctx_pos,
                q_scale, k_scale, v_scale, pool, 15,
            ).join()
            var e = Int(perf_counter_ns()) - t0
            keep(scratch[0])
            if e < best_15: best_15 = e
        pool.end_forward()

        print("  " + String(ctx_pos)
              + " | " + String(best_ctrl // 1000)
              + " | " + String(best_4 // 1000)
              + " | " + String(best_8 // 1000)
              + " | " + String(best_15 // 1000))

    # =====================================================================
    # Detailed profiles at 4k and 16k
    # =====================================================================
    for ci in range(2):
        var ctx_pos = 4096
        if ci == 1: ctx_pos = 16384

        print("\n=== Profile at ctx=" + String(ctx_pos) + " ===")

        # Control profile
        pool.begin_forward()
        decode_hot[NH, NKV, HD](
            DynView[QSlot](Int(q_bf16), 1),
            k_cache, v_cache,
            Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
            Int(scratch), ctx_pos,
            q_scale, k_scale, v_scale, pool,
        )
        pool.end_forward()

        print("control (wpg=1):")
        var ctrl_agg = ProfileAggregator()
        control_collect_profiles[NH, NKV, HD](Int(scratch), ctrl_agg)
        ctrl_agg.print_summary()

        # Chunked wpg=8 profile
        pool.begin_forward()
        decode_chunked[NH, NKV, HD](
            DynView[QSlot](Int(q_bf16), 1),
            k_cache, v_cache,
            Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
            Int(scratch), ctx_pos,
            q_scale, k_scale, v_scale, pool, 8,
        ).join()
        pool.end_forward()

        print("chunked wpg=8 caller:")
        chunk_read_caller[NH, NKV, HD](Int(scratch), cap, 8).print_summary()
        print("chunked wpg=8 workers:")
        var chunk_agg8 = ProfileAggregator()
        chunk_collect_profiles[NH, NKV, HD](Int(scratch), cap, chunk_agg8, 8)
        chunk_agg8.print_summary()

        # Chunked wpg=15 profile
        pool.begin_forward()
        decode_chunked[NH, NKV, HD](
            DynView[QSlot](Int(q_bf16), 1),
            k_cache, v_cache,
            Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
            Int(scratch), ctx_pos,
            q_scale, k_scale, v_scale, pool, 15,
        ).join()
        pool.end_forward()

        print("chunked wpg=15 caller:")
        chunk_read_caller[NH, NKV, HD](Int(scratch), cap, 15).print_summary()
        print("chunked wpg=15 workers:")
        var chunk_agg15 = ProfileAggregator()
        chunk_collect_profiles[NH, NKV, HD](Int(scratch), cap, chunk_agg15, 15)
        chunk_agg15.print_summary()
