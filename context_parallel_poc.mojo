"""Context-parallel attention POC — distributed KV cache + softmax merge.

Validates that splitting a KV cache block-contiguously across simulated
NUMA ranks and merging partial softmax states produces the same output as
a single-pass reference over the full cache.

All memory is allocated from NumaArena (mmap-backed, invisible to the Mojo
runtime GC). The kernel is called directly on the main thread — threading
is already proven in the model; this POC validates the distributed cache
addressing and softmax merge correctness.

On 1-node systems: tests tp=1 (bit-exact) and tp=2,4 (simulated ranks,
same NUMA node but separate arenas).
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.sliding_attention import single_pass_attention
from experimental3.kernels.full_chunked_attention import (
    ChunkedAttnArgs, chunked_attn_kernel,
    partial_head_stride, partial_chunk_stride,
)
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.fwht import fwht_block
from experimental3.common_math import F32Ptr, I8Ptr
from simd_math import exp_f32


comptime HEAD_DIM = 512
comptime NUM_KV_HEADS = 1
comptime NUM_Q_HEADS = 8
comptime HPG = 8
comptime WIDTH = CACHE_WIDTH
comptime TEST_CONTEXT = 256
comptime CACHE = Gemma4KVCache[TEST_CONTEXT, HEAD_DIM, NUM_KV_HEADS, NUM_Q_HEADS]
comptime PARTIAL_H_STRIDE = partial_head_stride[HEAD_DIM]()
comptime PARTIAL_C_STRIDE = partial_chunk_stride[HEAD_DIM, HPG]()

# Per-rank arena: cache + 1 chunk of partial state + alignment headroom
comptime RANK_ARENA_BYTES = CACHE.TOTAL_BYTES + PARTIAL_C_STRIDE * 4 + 4096
# Shared arena: reference cache + Q vectors + ref output + gather + test output
comptime SHARED_ARENA_BYTES = (
    CACHE.TOTAL_BYTES
    + HPG * HEAD_DIM          # q_i8
    + HPG * 4                 # qi_biases
    + HPG * 4                 # q_scales
    + HPG * HEAD_DIM * 4      # ref_out
    + HPG * HEAD_DIM * 4      # test_out
    + 4 * PARTIAL_C_STRIDE * 4  # gather buffer (max 4 ranks)
    + 4096                    # alignment headroom
)


# ============================================================================
# Deterministic data generation
# ============================================================================


@always_inline
def xorshift(state: UInt64) -> UInt64:
    var s = state
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


def fill_random_f32(dst: F32Ptr, count: Int, seed: UInt64) -> UInt64:
    var state = seed
    for i in range(count):
        state = xorshift(state)
        dst[i] = Float32(Int32(state & 0x7FFFFFFF)) / Float32(0x3FFFFFFF) - 1.0
    return state


def seed_for_position(global_pos: Int) -> UInt64:
    return xorshift(UInt64(global_pos + 1) * UInt64(0x9E3779B97F4A7C15))


def write_kv_position(cache: CACHE, local_pos: Int, global_pos: Int):
    var state = seed_for_position(global_pos)
    var work_arr = InlineArray[Float32, HEAD_DIM](uninitialized=True)
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var qi_arr = InlineArray[Scalar[DType.int8], HEAD_DIM](uninitialized=True)
    var qi = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    state = fill_random_f32(F32Ptr(work), HEAD_DIM, state)
    fwht_block[HEAD_DIM](work)
    var k_absmax = absmax_quantize_i8[HEAD_DIM](work, qi)
    cache.write_k(local_pos, 0, qi)
    cache.write_k_scale(local_pos, 0, k_absmax)

    state = fill_random_f32(F32Ptr(work), HEAD_DIM, state)
    fwht_block[HEAD_DIM](work)
    var v_absmax = absmax_quantize_i8[HEAD_DIM](work, qi)
    cache.write_v(local_pos, 0, qi)
    cache.write_v_scale(local_pos, 0, v_absmax)


def generate_q_vectors(q_i8: I8Ptr, qi_biases: F32Ptr, q_scales: F32Ptr):
    var work_arr = InlineArray[Float32, HEAD_DIM](uninitialized=True)
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var seed = UInt64(0xDEADBEEF_CAFEBABE)
    for qh in range(HPG):
        seed = fill_random_f32(F32Ptr(work), HEAD_DIM, seed)
        fwht_block[HEAD_DIM](work)
        var dst = q_i8 + qh * HEAD_DIM
        q_scales[qh] = absmax_quantize_i8[HEAD_DIM](work, dst)
        var qsum = Int32(0)
        for d in range(HEAD_DIM):
            qsum += Int32(dst[d])
        qi_biases[qh] = Float32(qsum) * 128.0


# ============================================================================
# Merge partials to f32 (for comparison, no i8 quantization)
# ============================================================================


def merge_partials_f32(
    partials: InlineArray[F32Ptr, 4],
    num_ranks: Int,
    output: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()

    for qh in range(HPG):
        var global_max = Float32(-1e30)
        for r in range(num_ranks):
            var p = partials[r] + qh * PARTIAL_H_STRIDE
            global_max = max(global_max, p[])

        var total_sum = Float32(0)
        var merged = InlineArray[Float32, HEAD_DIM](fill=Float32(0))
        var mp = UnsafePointer(to=merged).bitcast[Float32]()

        for r in range(num_ranks):
            var p = partials[r] + qh * PARTIAL_H_STRIDE
            var chunk_sum = (p + 1)[]
            if chunk_sum <= 0:
                continue
            var rescale = Float32(exp_f32[1](p[] - global_max))
            total_sum += chunk_sum * rescale
            var v_acc = p + 2
            var d = 0
            while d + width <= HEAD_DIM:
                var acc = (mp + d).load[width=width]()
                (mp + d).store((v_acc + d).load[width=width]().fma(rescale, acc))
                d += width

        var inv_sum = Float32(1.0) / (Float32(127) * total_sum)
        var out = output + qh * HEAD_DIM
        var d = 0
        while d + width <= HEAD_DIM:
            (out + d).store((mp + d).load[width=width]() * inv_sum)
            d += width


# ============================================================================
# Comparison
# ============================================================================


def compare(expected: F32Ptr, actual: F32Ptr, label: String):
    var max_err = Float32(0)
    var sum_err = Float32(0)
    var count = 0
    for i in range(HPG * HEAD_DIM):
        var e = expected[i]
        var a = actual[i]
        if e != e or a != a:
            continue
        var err = (e - a).__abs__()
        if err > max_err:
            max_err = err
        sum_err += err
        count += 1
    var mean_err = sum_err / Float32(count) if count > 0 else Float32(0)
    print("  " + label + ": max_err=" + String(max_err)
          + "  mean_err=" + String(mean_err)
          + "  (" + String(count) + "/" + String(HPG * HEAD_DIM) + " valid)")


# ============================================================================
# Main
# ============================================================================


def main():
    var numa = NumaInfo()
    var node0 = 0
    if numa.num_nodes > 0:
        node0 = numa.nodes[0].id
    print("NUMA nodes:", numa.num_nodes, " using node", node0, "for shared data")

    # --- Shared arena (node 0) for Q, reference cache, outputs ---
    var shared = NumaArena[alignment=64](node0, SHARED_ARENA_BYTES)

    var q_i8_mem = shared.alloc[Scalar[DType.int8]](HPG * HEAD_DIM)
    var q_i8 = I8Ptr(q_i8_mem)
    var qi_biases_mem = shared.alloc[Float32](HPG)
    var qi_biases = F32Ptr(qi_biases_mem)
    var q_scales_mem = shared.alloc[Float32](HPG)
    var q_scales = F32Ptr(q_scales_mem)
    generate_q_vectors(q_i8, qi_biases, q_scales)

    var ref_cache_mem = shared.alloc[UInt8](CACHE.TOTAL_BYTES)
    var ref_cache = CACHE(Int(ref_cache_mem))
    for p in range(TEST_CONTEXT):
        write_kv_position(ref_cache, p, p)

    var ref_out = F32Ptr(shared.alloc[Float32](HPG * HEAD_DIM))
    for qh in range(HPG):
        single_pass_attention[HEAD_DIM, TEST_CONTEXT, NUM_KV_HEADS, NUM_Q_HEADS](
            q_i8 + qh * HEAD_DIM,
            qi_biases[qh], q_scales[qh],
            ref_cache, 0, TEST_CONTEXT,
            ref_out + qh * HEAD_DIM)
    print("Reference:", TEST_CONTEXT, "positions,", HPG, "heads")
    print("  sample:", ref_out[0], ref_out[1], ref_out[2])

    var test_out = F32Ptr(shared.alloc[Float32](HPG * HEAD_DIM))

    # --- Test tp=1: single rank, all positions, should be bit-exact ---
    print("\n--- tp=1 ---")
    test_distributed(shared, 1, node0, q_i8, qi_biases, q_scales, ref_out, test_out)

    # --- Test tp=2: 2 simulated ranks ---
    print("\n--- tp=2 ---")
    test_distributed(shared, 2, node0, q_i8, qi_biases, q_scales, ref_out, test_out)

    # --- Test tp=4: 4 simulated ranks ---
    print("\n--- tp=4 ---")
    test_distributed(shared, 4, node0, q_i8, qi_biases, q_scales, ref_out, test_out)

    _ = shared


def test_distributed(
    mut shared: NumaArena[alignment=64],
    tp: Int, node: Int,
    q_i8: I8Ptr, qi_biases: F32Ptr, q_scales: F32Ptr,
    ref_out: F32Ptr, test_out: F32Ptr,
):
    var local_ctx = TEST_CONTEXT // tp

    # Per-rank arenas (all on same node for single-node systems, still
    # validates the addressing and merge logic).
    var rank_arenas = HeapMoveArray[NumaArena[alignment=64]](tp)
    var cache_bases = InlineArray[Int, 4](fill=0)
    var partial_bases = InlineArray[Int, 4](fill=0)

    for rank in range(tp):
        rank_arenas.push(NumaArena[alignment=64](node, RANK_ARENA_BYTES))
        var cache_mem = rank_arenas[rank].alloc[UInt8](CACHE.TOTAL_BYTES)
        cache_bases[rank] = Int(cache_mem)
        var partial_mem = rank_arenas[rank].alloc[UInt8](PARTIAL_C_STRIDE * 4)
        partial_bases[rank] = Int(partial_mem)

    # Fill per-rank caches (block-contiguous)
    for rank in range(tp):
        var cache = CACHE(cache_bases[rank])
        for local_pos in range(local_ctx):
            var global_pos = rank * local_ctx + local_pos
            write_kv_position(cache, local_pos, global_pos)

    # Score: call kernel directly per rank (1 chunk per rank = all local pgs)
    var num_pg = (local_ctx + WIDTH - 1) // WIDTH
    var t0 = Int(perf_counter_ns())
    for rank in range(tp):
        var args = ChunkedAttnArgs(
            q_i8_base=Int(q_i8),
            qi_biases_base=Int(qi_biases),
            q_scales_base=Int(q_scales),
            cache_base=cache_bases[rank],
            kv_head=0,
            start_pg=0,
            end_pg=num_pg,
            partial_out=partial_bases[rank],
            context_len=local_ctx)
        chunked_attn_kernel[HEAD_DIM, TEST_CONTEXT, NUM_KV_HEADS, NUM_Q_HEADS, HPG](args)
    var t1 = Int(perf_counter_ns())
    print("  scoring:", (t1 - t0) / 1000, "us (" + String(tp) + " rank(s) x "
          + String(local_ctx) + " pos)")

    # Diagnostics
    for rank in range(tp):
        var p = F32Ptr(unsafe_from_address=partial_bases[rank])
        print("  rank", rank, "partial: max=" + String(p[0])
              + " sum=" + String(p[1]))

    # Merge
    var partial_ptrs = InlineArray[F32Ptr, 4](
        fill=F32Ptr(unsafe_from_address=0))
    for rank in range(tp):
        partial_ptrs[rank] = F32Ptr(unsafe_from_address=partial_bases[rank])
    merge_partials_f32(partial_ptrs, tp, test_out)

    compare(ref_out, test_out, "tp=" + String(tp))

    _ = rank_arenas
