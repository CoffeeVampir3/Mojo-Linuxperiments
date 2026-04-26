from std.collections import InlineArray
from std.memory import UnsafePointer, alloc

from numa import NumaInfo
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from simd_math import exp_f32
from experimental3.amx import VNNI_BLK
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.quantize import absmax_quantize_i8
from kernels.kernel_ops import MAX_POOL_CAPACITY
from modeling.minimax_m27_moe_butterquant_tp import MiniMaxM27Config, MiniMaxShapes
from minimax.kernels.amx_attention import AmxConfigArgs, amx_config_kernel
from minimax.kernels.attention import merge_quant_worker
from minimax.kernels.dispatch_args import MergeQuantArgs
from minimax.kernels.dispatch_kernels import (
    attn_chunk_count,
    chunked_score_dispatch_multi,
)


comptime C = MiniMaxM27Config
comptime TP = 4
comptime S = MiniMaxShapes[TP]
comptime HEAD_DIM = C.HEAD_DIM
comptime HPG = C.HPG
comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
comptime HEADS_PER_RANK = KV_PER_RANK * HPG
comptime TEST_MAX_SEQ = 8192
comptime TEST_POOL_CAP = 30
comptime Q_DENOM = Float32(127.0 * 127.0)
comptime INV_I8 = Float32(1.0 / 127.0)
comptime INV_SQRT_HEAD = Float32(0.08838834764831845)
comptime PARTIAL_F32S = KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + HEAD_DIM)
comptime CACHE_BYTES = Gemma4KVCache[
    TEST_MAX_SEQ, HEAD_DIM, KV_PER_RANK
].TOTAL_BYTES


def abs_f32(x: Float32) -> Float32:
    if x < Float32(0):
        return -x
    return x


def max_f32(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


def abs_i32(x: Int) -> Int:
    if x < 0:
        return -x
    return x


def prng_i8(index: Int, salt: Int) -> Scalar[DType.int8]:
    # Deterministic, bounded, non-symmetric enough to exercise q_bias correction.
    var x = (index * 1103515245 + salt * 12345 + 97) % 251
    return Scalar[DType.int8](x - 125)


def prng_scale(index: Int, salt: Int, base: Float32, span: Float32) -> Float32:
    var x = (index * 1664525 + salt * 1013904223 + 17) % 997
    return base + span * Float32(x) / Float32(996)


def packed_k_u8(
    cache: Gemma4KVCache[TEST_MAX_SEQ, HEAD_DIM, KV_PER_RANK],
    kv: Int,
    pos: Int,
    dim: Int,
) -> Int:
    var pos_group = pos // CACHE_WIDTH
    var slot = pos - pos_group * CACHE_WIDTH
    var kdg = dim // VNNI_BLK
    var lane = dim - kdg * VNNI_BLK
    var p = cache.k_pg_ptr(kv, pos_group)
    return Int((p + kdg * CACHE_WIDTH * VNNI_BLK + slot * VNNI_BLK + lane)[])


def packed_v_i8(
    cache: Gemma4KVCache[TEST_MAX_SEQ, HEAD_DIM, KV_PER_RANK],
    kv: Int,
    pos: Int,
    dim: Int,
) -> Int:
    comptime V_CG_BYTES = (CACHE_WIDTH // VNNI_BLK) * CACHE_WIDTH * VNNI_BLK
    var pos_group = pos // CACHE_WIDTH
    var in_group = pos - pos_group * CACHE_WIDTH
    var sub_quad = in_group // VNNI_BLK
    var vnni_slot = in_group - sub_quad * VNNI_BLK
    var cg = dim // CACHE_WIDTH
    var lane = dim - cg * CACHE_WIDTH
    var p = cache.v_pg_ptr(kv, pos_group)
    return Int(
        (p + cg * V_CG_BYTES + sub_quad * CACHE_WIDTH * VNNI_BLK
            + lane * VNNI_BLK + vnni_slot)[]
    )


def fill_inputs(
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    q_biases: UnsafePointer[Float32, MutAnyOrigin],
    q_scales: UnsafePointer[Float32, MutAnyOrigin],
    k_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    v_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache_base: UnsafePointer[UInt8, MutAnyOrigin],
):
    var cache = Gemma4KVCache[TEST_MAX_SEQ, HEAD_DIM, KV_PER_RANK](
        Int(cache_base))

    for kv in range(KV_PER_RANK):
        for qh in range(HPG):
            var head = kv * HPG + qh
            var qsum = 0
            for d in range(HEAD_DIM):
                var qv = prng_i8(head * HEAD_DIM + d, 11)
                q_i8[head * HEAD_DIM + d] = qv
                qsum += Int(qv)
            q_biases[head] = Float32(qsum) * Float32(128)
            q_scales[head] = (
                prng_scale(head, 23, Float32(0.65), Float32(0.55))
                * INV_SQRT_HEAD
            )

    for kv in range(KV_PER_RANK):
        for pos in range(TEST_MAX_SEQ):
            var row = (kv * TEST_MAX_SEQ + pos) * HEAD_DIM
            for d in range(HEAD_DIM):
                k_i8[row + d] = prng_i8(row + d, 101)
                v_i8[row + d] = prng_i8(row + d, 211)
            cache.write_k(pos, kv, k_i8 + row)
            cache.write_v(pos, kv, v_i8 + row)
            cache.write_k_scale(
                pos, kv, prng_scale(pos + kv * 10007, 307, Float32(0.55), Float32(0.90)))
            cache.write_v_scale(
                pos, kv, prng_scale(pos + kv * 10007, 409, Float32(0.55), Float32(0.90)))


def set_decode_amx[P: BurstThreadPool](mut pools: HeapMoveArray[P]):
    var args = InlineArray[AmxConfigArgs, MAX_POOL_CAPACITY](fill=AmxConfigArgs())
    var cap = pools[0].get_capacity()
    pools[0].dispatch[AmxConfigArgs, amx_config_kernel[HPG]](
        UnsafePointer(to=args[0]), cap)
    pools[0].join()


@fieldwise_init
struct CaseMetrics(Copyable, ImplicitlyCopyable):
    var f32_abs_sum: Float64
    var f32_sq_sum: Float64
    var f32_ref_abs_sum: Float64
    var f32_max_abs: Float32
    var f32_max_rel: Float32
    var i8_abs_sum: Int
    var i8_max_abs: Int
    var exact_i8: Int
    var elems: Int
    var worst_head: Int
    var worst_dim: Int

    def __init__(out self):
        self.f32_abs_sum = Float64(0)
        self.f32_sq_sum = Float64(0)
        self.f32_ref_abs_sum = Float64(0)
        self.f32_max_abs = Float32(0)
        self.f32_max_rel = Float32(0)
        self.i8_abs_sum = 0
        self.i8_max_abs = 0
        self.exact_i8 = 0
        self.elems = 0
        self.worst_head = 0
        self.worst_dim = 0


def reference_and_compare(
    context_len: Int,
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    q_biases: UnsafePointer[Float32, MutAnyOrigin],
    q_scales: UnsafePointer[Float32, MutAnyOrigin],
    cache_base: UnsafePointer[UInt8, MutAnyOrigin],
    kernel_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    kernel_scales: UnsafePointer[Float32, MutAnyOrigin],
    ref_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ref_scales: UnsafePointer[Float32, MutAnyOrigin],
) -> CaseMetrics:
    var cache = Gemma4KVCache[TEST_MAX_SEQ, HEAD_DIM, KV_PER_RANK](
        Int(cache_base))
    var metrics = CaseMetrics()
    var work_arr = InlineArray[Float32, HEAD_DIM](fill=Float32(0))
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()

    for kv in range(KV_PER_RANK):
        var k_scales = cache.k_scale_ptr(kv)
        var v_scales = cache.v_scale_ptr(kv)
        for qh in range(HPG):
            var head = kv * HPG + qh
            var max_score = Float32(-1e30)

            for pos in range(context_len):
                var raw = 0
                for d in range(HEAD_DIM):
                    raw += (
                        Int(q_i8[head * HEAD_DIM + d])
                        * packed_k_u8(cache, kv, pos, d)
                    )
                var score = (
                    (Float32(raw) - q_biases[head])
                    * q_scales[head] * k_scales[pos] / Q_DENOM
                )
                if score > max_score:
                    max_score = score

            for d in range(HEAD_DIM):
                work[d] = Float32(0)
            var total_sum = Float32(0)

            for pos in range(context_len):
                var raw = 0
                for d in range(HEAD_DIM):
                    raw += (
                        Int(q_i8[head * HEAD_DIM + d])
                        * packed_k_u8(cache, kv, pos, d)
                    )
                var score = (
                    (Float32(raw) - q_biases[head])
                    * q_scales[head] * k_scales[pos] / Q_DENOM
                )
                var w = Float32(exp_f32[1](score - max_score))
                total_sum += w
                var vw = w * v_scales[pos]
                for d in range(HEAD_DIM):
                    work[d] += vw * Float32(packed_v_i8(cache, kv, pos, d))

            var inv = INV_I8 / total_sum
            for d in range(HEAD_DIM):
                work[d] *= inv

            ref_scales[head] = absmax_quantize_i8[HEAD_DIM](
                work, ref_qi + head * HEAD_DIM)

            for d in range(HEAD_DIM):
                var idx = head * HEAD_DIM + d
                var kdq = (
                    Float32(Int(kernel_qi[idx])) * kernel_scales[head]
                    * INV_I8
                )
                var rdq = work[d]
                var fd = abs_f32(kdq - rdq)
                var denom = max_f32(abs_f32(rdq), Float32(1e-8))
                var rel = fd / denom
                metrics.f32_abs_sum += Float64(fd)
                metrics.f32_sq_sum += Float64(fd) * Float64(fd)
                metrics.f32_ref_abs_sum += Float64(abs_f32(rdq))
                metrics.elems += 1
                if fd > metrics.f32_max_abs:
                    metrics.f32_max_abs = fd
                    metrics.worst_head = head
                    metrics.worst_dim = d
                if rel > metrics.f32_max_rel:
                    metrics.f32_max_rel = rel

                var qi_diff = abs_i32(Int(kernel_qi[idx]) - Int(ref_qi[idx]))
                metrics.i8_abs_sum += qi_diff
                if qi_diff > metrics.i8_max_abs:
                    metrics.i8_max_abs = qi_diff
                if qi_diff == 0:
                    metrics.exact_i8 += 1

    _ = work_arr
    return metrics


def run_case[P: BurstThreadPool](
    mut pools: HeapMoveArray[P],
    context_len: Int,
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    q_biases: UnsafePointer[Float32, MutAnyOrigin],
    q_scales: UnsafePointer[Float32, MutAnyOrigin],
    cache_base: UnsafePointer[UInt8, MutAnyOrigin],
    partial: UnsafePointer[Float32, MutAnyOrigin],
    kernel_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    kernel_scales: UnsafePointer[Float32, MutAnyOrigin],
    ref_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ref_scales: UnsafePointer[Float32, MutAnyOrigin],
):
    var pool_cap = pools[0].get_capacity()
    if pool_cap > TEST_POOL_CAP:
        pool_cap = TEST_POOL_CAP

    chunked_score_dispatch_multi[
        HEAD_DIM, HPG, TEST_MAX_SEQ, KV_PER_RANK, C.MAX_ATTN_CHUNKS,
    ](
        Int(q_i8),
        Int(q_biases),
        Int(q_scales),
        Int(cache_base),
        0,
        KV_PER_RANK,
        context_len,
        pool_cap,
        Int(partial),
        pools[0],
    ).join()

    var merge_args = MergeQuantArgs(
        partial,
        attn_chunk_count(context_len, pool_cap, KV_PER_RANK, C.MAX_ATTN_CHUNKS),
        kernel_qi,
        kernel_scales,
    )
    merge_quant_worker[HEAD_DIM, HPG, C.MAX_ATTN_CHUNKS, KV_PER_RANK](
        merge_args)

    var metrics = reference_and_compare(
        context_len,
        q_i8,
        q_biases,
        q_scales,
        cache_base,
        kernel_qi,
        kernel_scales,
        ref_qi,
        ref_scales,
    )

    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var padded_pg = (num_pg + 3) & ~3
    var chunks = attn_chunk_count(
        context_len, pool_cap, KV_PER_RANK, C.MAX_ATTN_CHUNKS)
    var logical_jobs = chunks * KV_PER_RANK
    var workers = logical_jobs
    var mean_abs = metrics.f32_abs_sum / Float64(metrics.elems)
    var rms_abs = (metrics.f32_sq_sum / Float64(metrics.elems)) ** Float64(0.5)
    var rel_l1 = metrics.f32_abs_sum / max(metrics.f32_ref_abs_sum, Float64(1e-12))
    var mean_i8 = Float64(metrics.i8_abs_sum) / Float64(metrics.elems)
    var exact_pct = Float64(metrics.exact_i8) * Float64(100) / Float64(metrics.elems)

    print(
        context_len,
        padded_pg,
        chunks,
        logical_jobs,
        workers,
        "|",
        mean_abs,
        rms_abs,
        metrics.f32_max_abs,
        rel_l1,
        metrics.f32_max_rel,
        "|",
        mean_i8,
        metrics.i8_max_abs,
        exact_pct,
        "|",
        metrics.worst_head,
        metrics.worst_dim,
    )


def run_experiment[P: BurstThreadPool](mut pools: HeapMoveArray[P]):
    var q_i8 = alloc[Scalar[DType.int8]](HEADS_PER_RANK * HEAD_DIM, alignment=64)
    var q_biases = alloc[Float32](HEADS_PER_RANK, alignment=64)
    var q_scales = alloc[Float32](HEADS_PER_RANK, alignment=64)
    var k_i8 = alloc[Scalar[DType.int8]](
        KV_PER_RANK * TEST_MAX_SEQ * HEAD_DIM, alignment=64)
    var v_i8 = alloc[Scalar[DType.int8]](
        KV_PER_RANK * TEST_MAX_SEQ * HEAD_DIM, alignment=64)
    var cache_mem = alloc[UInt8](CACHE_BYTES, alignment=64)
    var partial = alloc[Float32](PARTIAL_F32S, alignment=64)
    var kernel_qi = alloc[Scalar[DType.int8]](
        HEADS_PER_RANK * HEAD_DIM, alignment=64)
    var ref_qi = alloc[Scalar[DType.int8]](
        HEADS_PER_RANK * HEAD_DIM, alignment=64)
    var kernel_scales = alloc[Float32](HEADS_PER_RANK, alignment=64)
    var ref_scales = alloc[Float32](HEADS_PER_RANK, alignment=64)

    print("m27 decode attention vs exact packed-cache quantized reference")
    print(
        "head_dim=", HEAD_DIM,
        " hpg=", HPG,
        " kv_per_rank=", KV_PER_RANK,
        " max_attn_chunks=", C.MAX_ATTN_CHUNKS,
        " cache_width=", CACHE_WIDTH,
        " test_pool_cap=", TEST_POOL_CAP,
    )
    print("initializing deterministic quantized Q/K/V cache")
    fill_inputs(q_i8, q_biases, q_scales, k_i8, v_i8, cache_mem)
    set_decode_amx(pools)

    print("")
    print(
        "ctx padded_pg chunks logical_jobs workers |",
        "f32_mean_abs f32_rms_abs f32_max_abs rel_l1 max_rel |",
        "i8_mean_abs i8_max_abs i8_exact_pct | worst_head worst_dim",
    )

    run_case(
        pools, 1, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 16, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 17, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 512, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 1024, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 2048, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 4096, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 7679, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 7680, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 7681, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)
    run_case(
        pools, 8192, q_i8, q_biases, q_scales, cache_mem,
        partial, kernel_qi, kernel_scales, ref_qi, ref_scales)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)

    if numa.has_isolation():
        var pools = HeapMoveArray[IsolatedBurstPool[]](1)
        pools.push(IsolatedBurstPool[].for_topology(numa, topo[0]))
        run_experiment(pools)
    else:
        var pools = HeapMoveArray[BurstPool[]](1)
        pools.push(BurstPool[].for_topology(numa, topo[0]))
        run_experiment(pools)
