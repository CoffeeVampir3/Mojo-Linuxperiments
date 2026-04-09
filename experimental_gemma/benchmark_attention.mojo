"""Attention-focused microbenchmarks for the experimental Gemma kernels.

Benchmarks report:
  - dispatch time
  - time to last worker completion
  - join overhead
  - total wall time

Cases cover the Gemma4 sliding and full attention geometries for decode and
short block/prefill-like runs.
"""

from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from numa import NumaInfo
from threading import BurstPool

from modeling.model_spec import BF16, Slot, Replicated, DynView, CacheView
from experimental_gemma.attention import local_attention, global_attention


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@fieldwise_init
struct DispatchTiming:
    var dispatch_ns: Int
    var wall_ns: Int
    var done_from_start_ns: Int
    var join_overhead_ns: Int


def checksum_bf16(ptr: BF16Ptr, count: Int) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += Float32(ptr[i])
    return sum


def print_attention_timing(label: String, iters: Int, timing: DispatchTiming, checksum: Float32):
    var scale = Float64(iters) * 1000.0
    print(label
        + ": wall=" + String(Float64(timing.wall_ns) / scale) + " us"
        + "  dispatch=" + String(Float64(timing.dispatch_ns) / scale) + " us"
        + "  last_worker=" + String(Float64(timing.done_from_start_ns) / scale) + " us"
        + "  join_oh=" + String(Float64(timing.join_overhead_ns) / Float64(iters)) + " ns"
        + "  checksum=" + String(checksum))


def fill_query[seq_len: Int, cols: Int](ptr: BF16Ptr):
    for i in range(seq_len * cols):
        ptr[i] = Scalar[DType.bfloat16](
            Float32(((i * 7) + 3) % 37 - 18) * Float32(0.0625)
        )


def fill_cache[max_pos: Int, cols: Int](ptr: BF16Ptr, bias: Int):
    for i in range(max_pos * cols):
        ptr[i] = Scalar[DType.bfloat16](
            Float32((i + bias) % 41 - 20) * Float32(0.03125)
        )


def timed_local_attention[
    seq_len: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    window_size: Int,
    max_pos: Int,
](
    q: DynView[Slot[BF16, Replicated, seq_len, num_heads * head_dim, 1]],
    k_cache: CacheView[Slot[BF16, Replicated, max_pos, num_kv_heads * head_dim, 1]],
    v_cache: CacheView[Slot[BF16, Replicated, max_pos, num_kv_heads * head_dim, 1]],
    output: DynView[Slot[BF16, Replicated, seq_len, num_heads * head_dim, 1]],
    pos: Int,
    mut pool: BurstPool[],
) -> DispatchTiming:
    var t0 = Int(perf_counter_ns())
    var pool_ptr = local_attention[num_heads, num_kv_heads, head_dim, window_size](
        q, k_cache, v_cache, output, pos, pool).take()
    var t1 = Int(perf_counter_ns())
    if pool_ptr:
        pool_ptr[].join()
    var t2 = Int(perf_counter_ns())
    var done_ts = pool.last_worker_timestamp()
    return DispatchTiming(
        dispatch_ns=t1 - t0,
        wall_ns=t2 - t0,
        done_from_start_ns=done_ts - t0,
        join_overhead_ns=t2 - done_ts,
    )


def timed_global_attention[
    seq_len: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    max_pos: Int,
](
    q: DynView[Slot[BF16, Replicated, seq_len, num_heads * head_dim, 1]],
    k_cache: CacheView[Slot[BF16, Replicated, max_pos, num_kv_heads * head_dim, 1]],
    v_cache: CacheView[Slot[BF16, Replicated, max_pos, num_kv_heads * head_dim, 1]],
    output: DynView[Slot[BF16, Replicated, seq_len, num_heads * head_dim, 1]],
    pos: Int,
    mut pool: BurstPool[],
) -> DispatchTiming:
    var t0 = Int(perf_counter_ns())
    var pool_ptr = global_attention[num_heads, num_kv_heads, head_dim](
        q, k_cache, v_cache, output, pos, pool).take()
    var t1 = Int(perf_counter_ns())
    if pool_ptr:
        pool_ptr[].join()
    var t2 = Int(perf_counter_ns())
    var done_ts = pool.last_worker_timestamp()
    return DispatchTiming(
        dispatch_ns=t1 - t0,
        wall_ns=t2 - t0,
        done_from_start_ns=done_ts - t0,
        join_overhead_ns=t2 - done_ts,
    )


def bench_local_attention[
    seq_len: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    window_size: Int,
    max_pos: Int,
](
    label: String,
    iters: Int,
    pos: Int,
    mut pool: BurstPool[],
):
    comptime q_cols = num_heads * head_dim
    comptime kv_cols = num_kv_heads * head_dim
    comptime QSlot = Slot[BF16, Replicated, seq_len, q_cols, 1]
    comptime KCSlot = Slot[BF16, Replicated, max_pos, kv_cols, 1]
    comptime VCSlot = Slot[BF16, Replicated, max_pos, kv_cols, 1]
    comptime OutSlot = Slot[BF16, Replicated, seq_len, q_cols, 1]

    var q_buf = alloc[Scalar[DType.bfloat16]](seq_len * q_cols)
    var k_buf = alloc[Scalar[DType.bfloat16]](max_pos * kv_cols)
    var v_buf = alloc[Scalar[DType.bfloat16]](max_pos * kv_cols)
    var out_buf = alloc[Scalar[DType.bfloat16]](seq_len * q_cols)

    fill_query[seq_len, q_cols](BF16Ptr(unsafe_from_address=Int(q_buf)))
    fill_cache[max_pos, kv_cols](BF16Ptr(unsafe_from_address=Int(k_buf)), 5)
    fill_cache[max_pos, kv_cols](BF16Ptr(unsafe_from_address=Int(v_buf)), 19)

    var q_view = DynView[QSlot](Int(q_buf), seq_len)
    var out_view = DynView[OutSlot](Int(out_buf), seq_len)
    var kc_view = CacheView[KCSlot](Int(k_buf))
    var vc_view = CacheView[VCSlot](Int(v_buf))

    for _ in range(3):
        _ = timed_local_attention[seq_len, num_heads, num_kv_heads, head_dim, window_size, max_pos](
            q_view, kc_view, vc_view, out_view, pos, pool)

    var total = DispatchTiming(0, 0, 0, 0)
    for _ in range(iters):
        var t = timed_local_attention[seq_len, num_heads, num_kv_heads, head_dim, window_size, max_pos](
            q_view, kc_view, vc_view, out_view, pos, pool)
        total.dispatch_ns += t.dispatch_ns
        total.wall_ns += t.wall_ns
        total.done_from_start_ns += t.done_from_start_ns
        total.join_overhead_ns += t.join_overhead_ns

    print_attention_timing(
        label,
        iters,
        total,
        checksum_bf16(BF16Ptr(unsafe_from_address=Int(out_buf)), seq_len * q_cols),
    )

    q_buf.free()
    k_buf.free()
    v_buf.free()
    out_buf.free()


def bench_global_attention[
    seq_len: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    max_pos: Int,
](
    label: String,
    iters: Int,
    pos: Int,
    mut pool: BurstPool[],
):
    comptime q_cols = num_heads * head_dim
    comptime kv_cols = num_kv_heads * head_dim
    comptime QSlot = Slot[BF16, Replicated, seq_len, q_cols, 1]
    comptime KCSlot = Slot[BF16, Replicated, max_pos, kv_cols, 1]
    comptime VCSlot = Slot[BF16, Replicated, max_pos, kv_cols, 1]
    comptime OutSlot = Slot[BF16, Replicated, seq_len, q_cols, 1]

    var q_buf = alloc[Scalar[DType.bfloat16]](seq_len * q_cols)
    var k_buf = alloc[Scalar[DType.bfloat16]](max_pos * kv_cols)
    var v_buf = alloc[Scalar[DType.bfloat16]](max_pos * kv_cols)
    var out_buf = alloc[Scalar[DType.bfloat16]](seq_len * q_cols)

    fill_query[seq_len, q_cols](BF16Ptr(unsafe_from_address=Int(q_buf)))
    fill_cache[max_pos, kv_cols](BF16Ptr(unsafe_from_address=Int(k_buf)), 7)
    fill_cache[max_pos, kv_cols](BF16Ptr(unsafe_from_address=Int(v_buf)), 23)

    var q_view = DynView[QSlot](Int(q_buf), seq_len)
    var out_view = DynView[OutSlot](Int(out_buf), seq_len)
    var kc_view = CacheView[KCSlot](Int(k_buf))
    var vc_view = CacheView[VCSlot](Int(v_buf))

    for _ in range(2):
        _ = timed_global_attention[seq_len, num_heads, num_kv_heads, head_dim, max_pos](
            q_view, kc_view, vc_view, out_view, pos, pool)

    var total = DispatchTiming(0, 0, 0, 0)
    for _ in range(iters):
        var t = timed_global_attention[seq_len, num_heads, num_kv_heads, head_dim, max_pos](
            q_view, kc_view, vc_view, out_view, pos, pool)
        total.dispatch_ns += t.dispatch_ns
        total.wall_ns += t.wall_ns
        total.done_from_start_ns += t.done_from_start_ns
        total.join_overhead_ns += t.join_overhead_ns

    print_attention_timing(
        label,
        iters,
        total,
        checksum_bf16(BF16Ptr(unsafe_from_address=Int(out_buf)), seq_len * q_cols),
    )

    q_buf.free()
    k_buf.free()
    v_buf.free()
    out_buf.free()


def main():
    print("=== experimental_gemma attention microbench ===")
    print("sliding: heads=16 kv_heads=8 dim=256 window=1024")
    print("global: heads=16 kv_heads=2 dim=512")

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])
    print("pool workers=" + String(pool.get_capacity()))

    print("-- decode --")
    print("jobs: sliding=" + String(min(16, pool.get_capacity()))
        + "  global=" + String(min(16, pool.get_capacity())))
    bench_local_attention[1, 16, 8, 256, 1024, 1024](
        "local_attention seq=1 pos=1023",
        80,
        1023,
        pool,
    )
    bench_global_attention[1, 16, 2, 512, 4096](
        "global_attention seq=1 pos=4095",
        20,
        4095,
        pool,
    )

    print("-- block/prefill-like --")
    print("jobs: sliding=" + String(min(64 * 16, pool.get_capacity()))
        + "  global=" + String(min(8 * 16, pool.get_capacity())))
    bench_local_attention[64, 16, 8, 256, 1024, 1087](
        "local_attention seq=64 pos=1023",
        8,
        1023,
        pool,
    )
    bench_global_attention[8, 16, 2, 512, 4103](
        "global_attention seq=8 pos=4095",
        4,
        4095,
        pool,
    )

    _ = pool^
