"""Direct comparison: prefill vs decode at various configs."""

from std.sys.info import simd_width_of
from std.benchmark import keep
from std.time import perf_counter_ns

from modeling.model_spec import (
    BF16, F32, I8, Replicated,
    Slot, Bound, DynView,
)
from experimental.hadquant_attn_amx import int8_gqa_attention_amx, attn_scratch_bytes_amx
from experimental.hadquant_attn_amx_prefill import int8_gqa_attention_amx_prefill, attn_scratch_bytes_amx_prefill
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.amx import init_intel_amx
from threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from numa.arena import NumaArena
from kernels.kernel_ops import init_rope_tables


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    comptime HD = 128
    comptime NH = 128
    comptime NKV = 8
    comptime HIDDEN = NH * HD
    comptime HALF = HD // 2
    comptime MAX_SEQ = 16384
    comptime MAX_SL = 256

    var numa = NumaInfo()
    var local_node = get_current_cpu_and_node()[1]
    var arena = NumaArena(local_node, 512 * 1024 * 1024)

    comptime KVC = HadQuantKVCache[MAX_SEQ, HD, NKV]
    comptime DECODE_SCRATCH = attn_scratch_bytes_amx[NH, NKV, HD, NKV]()
    comptime PREFILL_SCRATCH = attn_scratch_bytes_amx_prefill[NH, NKV, HD, MAX_SL, NKV]()

    var q = arena.alloc[Scalar[DType.bfloat16]](MAX_SL * HIDDEN)
    var kv_mem = arena.alloc[UInt8](2 * KVC.TOTAL_BYTES)
    var k_cache = KVC(Int(kv_mem))
    var v_cache = KVC(Int(kv_mem) + KVC.TOTAL_BYTES)
    var qi_out = arena.alloc[Scalar[DType.int8]](MAX_SL * HIDDEN)
    var sc_out = arena.alloc[Float32](MAX_SL)
    var cos_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    var sin_tab = arena.alloc[Float32](MAX_SEQ * HALF)
    var decode_scratch = arena.alloc[UInt8](DECODE_SCRATCH)
    var prefill_scratch = arena.alloc[UInt8](PREFILL_SCRATCH)
    var head_buf = arena.alloc[Scalar[DType.int8]](HD)
    var burst = BurstPool[].for_numa_node(numa, 0)

    for i in range(MAX_SL * HIDDEN):
        q[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)
    for t in range(MAX_SEQ):
        for g in range(NKV):
            for d in range(HD):
                head_buf[d] = Scalar[DType.int8]((t * 7 + g * HD + d * 3) % 251 - 125)
            k_cache.write_head(t, g, head_buf, Float32(0.05))
            for d in range(HD):
                head_buf[d] = Scalar[DType.int8]((t * 11 + g * HD + d * 5) % 251 - 125)
            v_cache.write_head_transposed(t, g, head_buf, Float32(0.05))

    comptime CosSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_SEQ, HALF, 1]
    init_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)), Float64(100000.0))

    comptime QSlot = Slot[BF16, Replicated, 1, HIDDEN, 1]
    comptime QiSlot = Slot[I8, Replicated, 1, HIDDEN, 1]
    comptime ScSlot = Slot[F32, Replicated, 1, 1, 1]

    print("=== Prefill (KV-outer) vs Decode ===")
    print("128h/8kv, hd=128")
    print("ctx      sl    prefill(us/pos)  decode(us/pos)  ratio")

    for ctx_idx in range(4):
        var ctx_pos = 1024
        if ctx_idx == 1: ctx_pos = 2048
        elif ctx_idx == 2: ctx_pos = 4096
        elif ctx_idx == 3: ctx_pos = 8192

        for sl_idx in range(4):
            var sl = 16
            if sl_idx == 1: sl = 64
            elif sl_idx == 2: sl = 128
            elif sl_idx == 3: sl = 256

            # Prefill (suppress phase output for clean table)
            for r in range(2):
                int8_gqa_attention_amx_prefill[NH, NKV, HD](
                    DynView[QSlot](Int(q), sl), k_cache, v_cache,
                    DynView[QiSlot](Int(qi_out), sl), DynView[ScSlot](Int(sc_out), sl),
                    Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                    Int(prefill_scratch), ctx_pos, burst,
                ).join()
            var best_p = Int(1 << 60)
            for r in range(3):
                var t0 = Int(perf_counter_ns())
                int8_gqa_attention_amx_prefill[NH, NKV, HD](
                    DynView[QSlot](Int(q), sl), k_cache, v_cache,
                    DynView[QiSlot](Int(qi_out), sl), DynView[ScSlot](Int(sc_out), sl),
                    Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                    Int(prefill_scratch), ctx_pos, burst,
                ).join()
                var wall = Int(perf_counter_ns()) - t0
                keep(qi_out[0])
                if wall < best_p: best_p = wall

            # Decode
            for r in range(2):
                int8_gqa_attention_amx[NH, NKV, HD](
                    DynView[QSlot](Int(q), sl), k_cache, v_cache,
                    DynView[QiSlot](Int(qi_out), sl), DynView[ScSlot](Int(sc_out), sl),
                    Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                    Int(decode_scratch), ctx_pos, burst,
                ).join()
            var best_d = Int(1 << 60)
            for r in range(3):
                var t0 = Int(perf_counter_ns())
                int8_gqa_attention_amx[NH, NKV, HD](
                    DynView[QSlot](Int(q), sl), k_cache, v_cache,
                    DynView[QiSlot](Int(qi_out), sl), DynView[ScSlot](Int(sc_out), sl),
                    Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)),
                    Int(decode_scratch), ctx_pos, burst,
                ).join()
                var wall = Int(perf_counter_ns()) - t0
                keep(qi_out[0])
                if wall < best_d: best_d = wall

            var p_per = best_p // 1000 // sl
            var d_per = best_d // 1000 // sl
            var ratio = p_per * 10 // (d_per + 1) if d_per > 0 else 0
            print(String(ctx_pos) + "    " + String(sl) + "     "
                  + String(p_per) + "              " + String(d_per)
                  + "             " + String(ratio // 10) + "." + String(ratio % 10) + "x")
        print("")
