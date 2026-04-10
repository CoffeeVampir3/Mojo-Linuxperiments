"""Microbenchmarks for experimental3 sliding decode attention.

Measures three layers of work:
  1. `single_pass_attention` only
  2. `sliding_attn_group_kernel` direct call for one KV group
  3. Full BurstPool dispatch across all KV groups

This mirrors the earlier experimental_gemma attention benchmarking, but keeps
the direct worker timings so dispatch and parallel speedup are visible.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns
from std.sys.info import size_of
from std.collections import InlineArray

from numa import NumaInfo
from threading import BurstPool

from experimental3.kv_cache import Gemma4KVCache
from experimental3.helpers import prep_q_row_normed
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.kernels.sliding_attention import (
    single_pass_attention,
    SlidingAttnGroupArgs,
    sliding_attn_group_kernel,
)


comptime BF16 = Scalar[DType.bfloat16]
comptime BF16Ptr = UnsafePointer[BF16, MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]


@fieldwise_init
struct DirectTiming:
    var wall_ns: Int


@fieldwise_init
struct DispatchTiming:
    var dispatch_ns: Int
    var wall_ns: Int
    var done_from_start_ns: Int
    var join_overhead_ns: Int


def checksum_f32(ptr: UnsafePointer[Float32, MutAnyOrigin], count: Int) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += ptr[i]
    return sum


def checksum_i8(ptr: I8Ptr, count: Int) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += Float32(ptr[i])
    return sum


def fill_bf16(ptr: BF16Ptr, count: Int, bias: Int):
    for i in range(count):
        ptr[i] = BF16(Float32(((i * 13 + bias) % 41) - 20) * Float32(0.03125))


def fill_ones_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BF16(Float32(1.0))


def fill_rope_tables(cos: UnsafePointer[Float32, MutAnyOrigin], sin: UnsafePointer[Float32, MutAnyOrigin], half_dim: Int):
    for i in range(half_dim):
        var angle = Float32(i + 1) * Float32(0.0025)
        cos[i] = Float32(1.0) - angle * angle * Float32(0.5)
        sin[i] = angle


def zero_cache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int](
    ptr: UnsafePointer[UInt8, MutAnyOrigin],
):
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]
    for i in range(Cache.TOTAL_BYTES):
        ptr[i] = UInt8(0)


def prefill_cache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int](
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    k_norm: BF16Ptr,
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    v_scale: Float32,
    eps: Float32,
):
    var k_row = alloc[BF16](head_dim)
    var v_row = alloc[BF16](head_dim)
    var work = alloc[Float32](head_dim)
    var qi = alloc[Scalar[DType.int8]](head_dim)
    var quant_inv = Float32(127.0) / v_scale

    for pos in range(max_seq):
        for kvh in range(num_kv_heads):
            for d in range(head_dim):
                k_row[d] = BF16(
                    Float32((((pos * 17) + (kvh * 5) + (d * 3)) % 47) - 23)
                    * Float32(0.03125)
                )
                v_row[d] = BF16(
                    Float32((((pos * 11) + (kvh * 7) + (d * 9)) % 53) - 26)
                    * Float32(0.03125)
                )

            write_k_head_normed[head_dim](
                k_row,
                k_norm,
                cos,
                sin,
                work,
                qi,
                cache,
                pos,
                kvh,
                eps,
            )
            write_v_head_normed[head_dim](
                v_row,
                quant_inv,
                work,
                qi,
                cache,
                pos,
                kvh,
                eps,
            )

    k_row.free()
    v_row.free()
    work.free()
    qi.free()


def timed_single_pass[
    head_dim: Int,
    max_seq: Int,
    num_kv_heads: Int,
    num_q_heads: Int,
](
    q_i8: I8Ptr,
    qi_bias: Float32,
    q_scale: Float32,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    kv_head: Int,
    context_len: Int,
    output: UnsafePointer[Float32, MutAnyOrigin],
) -> DirectTiming:
    var t0 = Int(perf_counter_ns())
    single_pass_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
        q_i8,
        qi_bias,
        q_scale,
        cache,
        kv_head,
        context_len,
        output,
    )
    var t1 = Int(perf_counter_ns())
    return DirectTiming(wall_ns=t1 - t0)


def timed_group_direct[
    head_dim: Int,
    heads_per_group: Int,
    max_seq: Int,
    num_kv_heads: Int,
    num_q_heads: Int,
](
    args: SlidingAttnGroupArgs,
) -> DirectTiming:
    var t0 = Int(perf_counter_ns())
    sliding_attn_group_kernel[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](args)
    var t1 = Int(perf_counter_ns())
    return DirectTiming(wall_ns=t1 - t0)


def timed_dispatch[
    head_dim: Int,
    heads_per_group: Int,
    max_seq: Int,
    num_kv_heads: Int,
    num_q_heads: Int,
](
    jobs_ptr: UnsafePointer[SlidingAttnGroupArgs, MutAnyOrigin],
    mut pool: BurstPool[],
) -> DispatchTiming:
    var t0 = Int(perf_counter_ns())
    pool.dispatch[
        SlidingAttnGroupArgs,
        sliding_attn_group_kernel[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads],
    ](jobs_ptr, num_kv_heads)
    var t1 = Int(perf_counter_ns())
    pool.join()
    var t2 = Int(perf_counter_ns())
    var done_ts = pool.last_worker_timestamp()
    return DispatchTiming(
        dispatch_ns=t1 - t0,
        wall_ns=t2 - t0,
        done_from_start_ns=done_ts - t0,
        join_overhead_ns=t2 - done_ts,
    )


def print_direct(label: String, iters: Int, timing: DirectTiming, checksum: Float32):
    var scale = Float64(iters) * 1000.0
    print(label
        + ": wall=" + String(Float64(timing.wall_ns) / scale) + " us"
        + "  checksum=" + String(checksum))


def print_dispatch(
    label: String,
    iters: Int,
    timing: DispatchTiming,
    checksum: Float32,
    serial_estimate_us: Float64,
):
    var scale = Float64(iters) * 1000.0
    var wall_us = Float64(timing.wall_ns) / scale
    var dispatch_us = Float64(timing.dispatch_ns) / scale
    var done_us = Float64(timing.done_from_start_ns) / scale
    var join_ns = Float64(timing.join_overhead_ns) / Float64(iters)
    var speedup = serial_estimate_us / wall_us if wall_us > 0.0 else Float64(0)
    print(label
        + ": wall=" + String(wall_us) + " us"
        + "  dispatch=" + String(dispatch_us) + " us"
        + "  last_worker=" + String(done_us) + " us"
        + "  join_oh=" + String(join_ns) + " ns"
        + "  checksum=" + String(checksum)
        + "  est_serial_speedup=" + String(speedup) + "x")


def bench_context[
    context_len: Int,
    head_dim: Int,
    heads_per_group: Int,
    max_seq: Int,
    num_kv_heads: Int,
    num_q_heads: Int,
](
    label: String,
    single_pass_iters: Int,
    group_iters: Int,
    dispatch_iters: Int,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    q_all: BF16Ptr,
    k_all: BF16Ptr,
    v_all: BF16Ptr,
    q_norm: BF16Ptr,
    k_norm: BF16Ptr,
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    v_scale: Float32,
    eps: Float32,
    mut pool: BurstPool[],
):
    var q_i8 = alloc[Scalar[DType.int8]](head_dim)
    var single_out = alloc[Float32](head_dim)
    var group_qi_out = alloc[Scalar[DType.int8]](heads_per_group * head_dim)
    var group_scales = alloc[Float32](heads_per_group)
    var dispatch_qi_out = alloc[Scalar[DType.int8]](num_q_heads * head_dim)
    var dispatch_scales = alloc[Float32](num_q_heads)
    var q0 = q_all
    var prep = prep_q_row_normed[head_dim](
        q0.bitcast[BFloat16](),
        q_norm,
        cos,
        sin,
        q_i8.bitcast[Int8](),
        eps,
    )
    var qi_bias = prep[0]
    var q_scale = prep[1]

    var base_addr = cache.k_base
    var current_pos = context_len - 1
    var group_args = SlidingAttnGroupArgs(
        Int(q_all),
        Int(k_all),
        Int(v_all),
        Int(q_norm),
        Int(k_norm),
        Int(cos),
        Int(sin),
        base_addr,
        0,
        current_pos,
        context_len,
        v_scale,
        Int(group_qi_out),
        Int(group_scales),
        eps,
    )

    var jobs = InlineArray[SlidingAttnGroupArgs, num_kv_heads](
        fill=SlidingAttnGroupArgs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Float32(0), 0, 0, Float32(0)))
    comptime q_group_elems = heads_per_group * head_dim
    comptime bf16_bytes = size_of[BF16]()
    comptime f32_bytes = size_of[Float32]()
    for kvh in range(num_kv_heads):
        jobs[kvh] = SlidingAttnGroupArgs(
            Int(q_all) + kvh * q_group_elems * bf16_bytes,
            Int(k_all) + kvh * head_dim * bf16_bytes,
            Int(v_all) + kvh * head_dim * bf16_bytes,
            Int(q_norm),
            Int(k_norm),
            Int(cos),
            Int(sin),
            base_addr,
            kvh,
            current_pos,
            context_len,
            v_scale,
            Int(dispatch_qi_out) + kvh * q_group_elems,
            Int(dispatch_scales) + kvh * heads_per_group * f32_bytes,
            eps,
        )

    for _ in range(5):
        _ = timed_single_pass[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8,
            qi_bias,
            q_scale,
            cache,
            0,
            context_len,
            single_out,
        )
    var single_total = DirectTiming(0)
    for _ in range(single_pass_iters):
        var t = timed_single_pass[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8,
            qi_bias,
            q_scale,
            cache,
            0,
            context_len,
            single_out,
        )
        single_total.wall_ns += t.wall_ns
    print_direct(
        label + " single_pass",
        single_pass_iters,
        single_total,
        checksum_f32(single_out, head_dim),
    )

    for _ in range(3):
        _ = timed_group_direct[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](group_args)
    var group_total = DirectTiming(0)
    for _ in range(group_iters):
        var t = timed_group_direct[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](group_args)
        group_total.wall_ns += t.wall_ns
    var group_avg_us = Float64(group_total.wall_ns) / (Float64(group_iters) * 1000.0)
    print_direct(
        label + " group_direct",
        group_iters,
        group_total,
        checksum_i8(group_qi_out, heads_per_group * head_dim)
            + checksum_f32(group_scales, heads_per_group),
    )

    for _ in range(3):
        _ = timed_dispatch[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](
            UnsafePointer(to=jobs[0]),
            pool,
        )
    var dispatch_total = DispatchTiming(0, 0, 0, 0)
    for _ in range(dispatch_iters):
        var t = timed_dispatch[head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](
            UnsafePointer(to=jobs[0]),
            pool,
        )
        dispatch_total.dispatch_ns += t.dispatch_ns
        dispatch_total.wall_ns += t.wall_ns
        dispatch_total.done_from_start_ns += t.done_from_start_ns
        dispatch_total.join_overhead_ns += t.join_overhead_ns
    print_dispatch(
        label + " dispatch",
        dispatch_iters,
        dispatch_total,
        checksum_i8(dispatch_qi_out, num_q_heads * head_dim)
            + checksum_f32(dispatch_scales, num_q_heads),
        group_avg_us * Float64(num_kv_heads),
    )

    q_i8.free()
    single_out.free()
    group_qi_out.free()
    group_scales.free()
    dispatch_qi_out.free()
    dispatch_scales.free()


def main():
    comptime head_dim = 256
    comptime max_seq = 1024
    comptime num_kv_heads = 8
    comptime num_q_heads = 16
    comptime heads_per_group = num_q_heads // num_kv_heads
    comptime half_dim = head_dim // 2
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]
    var eps = Float32(1e-6)
    var v_scale = Float32(5.0)

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    zero_cache[max_seq, head_dim, num_kv_heads, num_q_heads](cache_buf)
    var cache = Cache(Int(cache_buf))

    var cos = alloc[Float32](half_dim)
    var sin = alloc[Float32](half_dim)
    fill_rope_tables(cos, sin, half_dim)

    print("=== experimental3 sliding attention microbench ===")
    print("shape: q_heads=16 kv_heads=8 heads_per_group=2 head_dim=256 max_seq=1024")

    var q_all = alloc[BF16](num_q_heads * head_dim)
    var k_all = alloc[BF16](num_kv_heads * head_dim)
    var v_all = alloc[BF16](num_kv_heads * head_dim)
    var q_norm = alloc[BF16](head_dim)
    var k_norm = alloc[BF16](head_dim)
    fill_bf16(q_all, num_q_heads * head_dim, 3)
    fill_bf16(k_all, num_kv_heads * head_dim, 11)
    fill_bf16(v_all, num_kv_heads * head_dim, 19)
    fill_ones_bf16(q_norm, head_dim)
    fill_ones_bf16(k_norm, head_dim)

    prefill_cache[max_seq, head_dim, num_kv_heads, num_q_heads](cache, k_norm, cos, sin, v_scale, eps)

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])
    print("pool workers=" + String(pool.get_capacity())
        + " dispatch_jobs=" + String(num_kv_heads))

    bench_context[128, head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](
        "ctx=128",
        160,
        80,
        80,
        cache,
        q_all,
        k_all,
        v_all,
        q_norm,
        k_norm,
        cos,
        sin,
        v_scale,
        eps,
        pool,
    )
    bench_context[512, head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](
        "ctx=512",
        120,
        60,
        60,
        cache,
        q_all,
        k_all,
        v_all,
        q_norm,
        k_norm,
        cos,
        sin,
        v_scale,
        eps,
        pool,
    )
    bench_context[1024, head_dim, heads_per_group, max_seq, num_kv_heads, num_q_heads](
        "ctx=1024",
        80,
        40,
        40,
        cache,
        q_all,
        k_all,
        v_all,
        q_norm,
        k_norm,
        cos,
        sin,
        v_scale,
        eps,
        pool,
    )

    q_all.free()
    k_all.free()
    v_all.free()
    q_norm.free()
    k_norm.free()
    cos.free()
    sin.free()
    cache_buf.free()
    _ = pool^
