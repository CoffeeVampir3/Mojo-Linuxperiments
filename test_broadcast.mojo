"""Streaming cursor broadcast experiment.

Producer-consumer broadcast over a NUMA-aware dependency tree.
Each producer writes data and advances a write cursor (atomic).
Consumers poll the cursor and copy available bytes incrementally,
then advance their own cursor for downstream consumers.

No barriers, no phases, no tiling. Data flows through the tree
as a continuous stream — downstream nodes begin copying as soon
as their upstream producer has written enough bytes. Cross-socket
latency is hidden behind intra-socket copy overlap.

For topology {0,1}=12, {2,3}=12, cross=21:
  Tree: rank 0 → rank 1 (intra-socket, direct)
        rank 0 → rank 2 (cross-socket, streamed)
        rank 2 → rank 3 (intra-socket, trails rank 2's cursor)

  Rank 3 starts copying from rank 2 while rank 2 is still
  pulling from rank 0 cross-socket — the cross-socket latency
  is hidden behind rank 3's intra-socket copy.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.os.atomic import Atomic, Consistency
from std.collections import InlineArray
from numa import NumaArena, NumaInfo, get_current_cpu_and_node
from notstdcollections import HeapMoveArray
from threading import BurstPool
import linux.sys as linux

comptime PoolPtr = UnsafePointer[BurstPool[], MutAnyOrigin]
comptime AtomicInt64 = Atomic[DType.int64]
# Cursors are Int64 values spaced 64 bytes apart (cache line isolation).
# Accessed atomically via raw pointers. Layout: cursor[rank] at base + rank * 64.
comptime CURSOR_STRIDE = 64

comptime TP = 4
comptime COLS = 4096 * 4
comptime SEQ_LEN = 256
comptime TOTAL_BYTES = SEQ_LEN * COLS * 2
comptime TOTAL_ELEMENTS = SEQ_LEN * COLS

# Copy granularity: how many bytes a consumer copies per poll iteration.
# Should be large enough to amortize atomic load cost, small enough for
# responsive streaming. 4KB = 1 page = good balance.
comptime CHUNK = 4096 * 32


# =============================================================================
# Cursor helpers — raw Int64 accessed atomically, spaced 64B apart
# =============================================================================


@always_inline
def cursor_ptr(base: Int, rank: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=base + rank * CURSOR_STRIDE)


@always_inline
def cursor_store(base: Int, rank: Int, val: Int):
    AtomicInt64.store[ordering=Consistency.RELEASE](cursor_ptr(base, rank), Int64(val))


@always_inline
def cursor_load(base: Int, rank: Int) -> Int:
    return Int(AtomicInt64.load[ordering=Consistency.ACQUIRE](cursor_ptr(base, rank)))


# =============================================================================
# Streaming broadcast kernel
# =============================================================================


def stream_copy_kernel(
    src_buf: Int, dst_buf: Int, total_bytes: Int,
    cursors_base: Int, producer_rank: Int, my_rank: Int,
):
    """Stream-copy from src to dst, polling producer's cursor for availability.

    Copies data in CHUNK-byte increments as the producer makes it available.
    Advances own cursor after each copy so downstream consumers can follow.

    Args: src_buf, dst_buf, total_bytes, cursors_base, producer_rank, my_rank.
    Cursors are Int64 values at cursors_base + rank * CURSOR_STRIDE.
    """
    var src = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_buf)
    var dst = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_buf)
    var sys = linux.linux_sys()
    var copied = 0

    while copied < total_bytes:
        # Poll: how far has the producer written?
        var available = cursor_load(cursors_base, producer_rank)
        if available <= copied:
            sys.arch_cpu_relax()
            continue

        # Copy everything available, aligned to CHUNK (except final tail).
        var end = min(available, total_bytes)
        var copy_end = end
        if copy_end < total_bytes:
            copy_end = (copy_end // CHUNK) * CHUNK
        if copy_end <= copied:
            if end == total_bytes:
                copy_end = end
            else:
                sys.arch_cpu_relax()
                continue

        memcpy(dest=dst + copied, src=src + copied, count=copy_end - copied)
        copied = copy_end

        # Publish progress for downstream consumers.
        cursor_store(cursors_base, my_rank, copied)


# =============================================================================
# Parallel pull — all ranks memcpy from source simultaneously
# =============================================================================


def memcpy_kernel(
    dst: Int, src: Int, count: Int,
    n3: Int, n4: Int, n5: Int,
):
    memcpy(
        dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst),
        src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src),
        count=count,
    )


def parallel_pull_broadcast(
    bases: InlineArray[Int, TP],
    total_bytes: Int,
    pool_ptrs: InlineArray[PoolPtr, TP],
):
    """Every destination rank pulls the full buffer from rank 0 simultaneously."""
    for r in range(1, TP):
        pool_ptrs[r][].args_base[].arg0 = bases[r]
        pool_ptrs[r][].args_base[].arg1 = bases[0]
        pool_ptrs[r][].args_base[].arg2 = total_bytes
        pool_ptrs[r][].dispatch(memcpy_kernel, pool_ptrs[r][].args_base, 1)
    for r in range(1, TP):
        pool_ptrs[r][].join()


# =============================================================================
# Naive baseline (single-threaded, sequential)
# =============================================================================


def naive_broadcast(bases: InlineArray[Int, TP], total_bytes: Int):
    for r in range(1, TP):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[r]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=total_bytes,
        )


# =============================================================================
# Streaming cursor broadcast
# =============================================================================


def streaming_broadcast(
    bases: InlineArray[Int, TP],
    total_bytes: Int,
    pool_ptrs: InlineArray[PoolPtr, TP],
):
    """Broadcast via streaming cursors over the NUMA dependency tree.

    Tree for 4-NUMA {0,1}=close, {2,3}=close:
      rank 0 is source (cursor starts at total_bytes)
      rank 1 reads from rank 0 (intra-socket)
      rank 2 reads from rank 0 (cross-socket)
      rank 3 reads from rank 2 (intra-socket, streams behind rank 2)
    """
    # Allocate cursor array: TP * CURSOR_STRIDE bytes, cache-line aligned.
    # Each cursor is an Int64 at offset rank * CURSOR_STRIDE.
    # Stack-allocated, alive until after all joins.
    var cursor_mem = InlineArray[UInt8, TP * CURSOR_STRIDE](fill=0)
    var cursors_base = Int(UnsafePointer(to=cursor_mem))

    # Rank 0 has all data ready.
    cursor_store(cursors_base, 0, total_bytes)
    # Ranks 1-3 start at 0 (already zeroed).

    # rank 1: reads from rank 0 (intra-socket)
    var p1 = pool_ptrs[1][].args_base
    p1[].arg0 = bases[0]
    p1[].arg1 = bases[1]
    p1[].arg2 = total_bytes
    p1[].arg3 = cursors_base
    p1[].arg4 = 0  # producer_rank
    p1[].arg5 = 1  # my_rank

    # rank 2: reads from rank 0 (cross-socket)
    var p2 = pool_ptrs[2][].args_base
    p2[].arg0 = bases[0]
    p2[].arg1 = bases[2]
    p2[].arg2 = total_bytes
    p2[].arg3 = cursors_base
    p2[].arg4 = 0  # producer_rank
    p2[].arg5 = 2  # my_rank

    # rank 3: reads from rank 2 (intra-socket, streams behind rank 2)
    var p3 = pool_ptrs[3][].args_base
    p3[].arg0 = bases[2]
    p3[].arg1 = bases[3]
    p3[].arg2 = total_bytes
    p3[].arg3 = cursors_base
    p3[].arg4 = 2  # producer_rank
    p3[].arg5 = 3  # my_rank

    # Dispatch all three simultaneously. Rank 0 needs no worker.
    pool_ptrs[1][].dispatch(stream_copy_kernel, pool_ptrs[1][].args_base, 1)
    pool_ptrs[2][].dispatch(stream_copy_kernel, pool_ptrs[2][].args_base, 1)
    pool_ptrs[3][].dispatch(stream_copy_kernel, pool_ptrs[3][].args_base, 1)

    pool_ptrs[1][].join()
    pool_ptrs[2][].join()
    pool_ptrs[3][].join()


# =============================================================================
# Helpers
# =============================================================================


def fill_pattern(ptr: Int, elements: Int, seed: Int):
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(elements):
        p[i] = UInt16((seed + i * 7 + 13) & 0xFFFF)


def clear_buffer(ptr: Int, total_bytes: Int):
    var p = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_bytes):
        p[i] = 0


def buffers_match(a: Int, b: Int, elements: Int) -> Bool:
    var pa = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=a)
    var pb = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=b)
    for i in range(elements):
        if pa[i] != pb[i]:
            return False
    return True


def sort_times(mut times: List[Int]):
    for i in range(1, len(times)):
        var c = times[i]
        var j = i
        while j > 0 and times[j - 1] > c:
            times[j] = times[j - 1]
            j -= 1
        times[j] = c


# =============================================================================
# Main
# =============================================================================


def main():
    print("=== Streaming Broadcast (TP=" + String(TP) + ", "
          + String(TOTAL_BYTES) + " B, chunk=" + String(CHUNK) + " B) ===")
    print()

    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)
    print("NUMA:", numa.num_nodes, "nodes")
    print("Rank placement:", end=" ")
    for r in range(TP):
        print(topo[r], end=" ")
    print()
    if numa.num_nodes > 1:
        print("Distances:")
        for i in range(numa.num_nodes):
            for j in range(numa.num_nodes):
                print(numa.distance(i, j), end=" ")
            print()
    print()

    # Arenas
    var arenas = HeapMoveArray[NumaArena[alignment=64]](TP)
    var bases = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        var arena = NumaArena[alignment=64](topo[rank], TOTAL_BYTES)
        if not arena:
            print("FATAL: arena alloc failed rank", rank)
            return
        bases[rank] = Int(arena.base)
        arenas.push(arena^)
    for rank in range(TP):
        _ = arenas[rank].prefault()

    # Validate placement
    print("Memory:")
    for rank in range(TP):
        var ok = arenas[rank].verify_placement()
        print("  rank", rank, "-> node", topo[rank], ":", "OK" if ok else "MISS")
    print()

    # Per-node pools (1 worker each)
    var pools = HeapMoveArray[BurstPool[]](TP)
    for rank in range(TP):
        var pool = BurstPool[](1, numa.get_node_mask[128](topo[rank]), topo[rank])
        if not pool:
            print("FATAL: pool failed rank", rank)
            return
        pools.push(pool^)
    var pool_ptrs = InlineArray[PoolPtr, TP](fill=PoolPtr())
    for rank in range(TP):
        pool_ptrs[rank] = PoolPtr(unsafe_from_address=Int(UnsafePointer(to=pools[rank])))

    # Worker placement check
    print("Workers:")
    for rank in range(TP):
        var pack = pool_ptrs[rank][].args_base
        pack[].arg0 = rank
        pack[].arg1 = topo[rank]
        pool_ptrs[rank][].dispatch(worker_check_kernel, pool_ptrs[rank][].args_base, 1)
        pool_ptrs[rank][].join()
    print()

    # =================================================================
    # Correctness
    # =================================================================
    print("--- Correctness ---")

    # Naive
    fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
    for r in range(1, TP):
        clear_buffer(bases[r], TOTAL_BYTES)
    naive_broadcast(bases, TOTAL_BYTES)
    var naive_ok = True
    for r in range(1, TP):
        if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
            print("FAIL naive rank", r)
            naive_ok = False
    print("naive:", "PASS" if naive_ok else "FAIL")

    # Streaming
    fill_pattern(bases[0], TOTAL_ELEMENTS, 77)
    for r in range(1, TP):
        clear_buffer(bases[r], TOTAL_BYTES)
    streaming_broadcast(bases, TOTAL_BYTES, pool_ptrs)
    var stream_ok = True
    for r in range(1, TP):
        if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
            print("FAIL streaming rank", r)
            stream_ok = False
    print("streaming:", "PASS" if stream_ok else "FAIL")

    print()

    # =================================================================
    # Bandwidth probe: measure raw link bandwidth and contention
    # =================================================================
    print("--- Bandwidth probe: " + String(TOTAL_BYTES) + " B ---")
    comptime PROBE_ITERS = 20

    # Single-pair: one memcpy at a time, measures raw link bandwidth
    # 0→1 intra-socket
    for _ in range(3):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[1]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=TOTAL_BYTES,
        )
    var t_01 = List[Int]()
    for _ in range(PROBE_ITERS):
        var t0 = perf_counter_ns()
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[1]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=TOTAL_BYTES,
        )
        t_01.append(Int(perf_counter_ns() - t0))
    sort_times(t_01)
    var med_01 = t_01[PROBE_ITERS // 2]
    print("  0->1 (intra, d=12):", med_01 // 1000, "us,",
          Float64(TOTAL_BYTES) / Float64(med_01), "GB/s")

    # 0→2 inter-socket
    for _ in range(3):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[2]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=TOTAL_BYTES,
        )
    var t_02 = List[Int]()
    for _ in range(PROBE_ITERS):
        var t0 = perf_counter_ns()
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[2]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=TOTAL_BYTES,
        )
        t_02.append(Int(perf_counter_ns() - t0))
    sort_times(t_02)
    var med_02 = t_02[PROBE_ITERS // 2]
    print("  0->2 (inter, d=21):", med_02 // 1000, "us,",
          Float64(TOTAL_BYTES) / Float64(med_02), "GB/s")

    # 2→3 intra-socket (different socket pair)
    for _ in range(3):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[3]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[2]),
            count=TOTAL_BYTES,
        )
    var t_23 = List[Int]()
    for _ in range(PROBE_ITERS):
        var t0 = perf_counter_ns()
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[3]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[2]),
            count=TOTAL_BYTES,
        )
        t_23.append(Int(perf_counter_ns() - t0))
    sort_times(t_23)
    var med_23 = t_23[PROBE_ITERS // 2]
    print("  2->3 (intra, d=12):", med_23 // 1000, "us,",
          Float64(TOTAL_BYTES) / Float64(med_23), "GB/s")

    # Concurrent: 0→1 AND 0→2 simultaneously (contention on rank 0's controller)
    for _ in range(3):
        pool_ptrs[1][].args_base[].arg0 = bases[1]
        pool_ptrs[1][].args_base[].arg1 = bases[0]
        pool_ptrs[1][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[2][].args_base[].arg0 = bases[2]
        pool_ptrs[2][].args_base[].arg1 = bases[0]
        pool_ptrs[2][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[1][].dispatch(memcpy_kernel, pool_ptrs[1][].args_base, 1)
        pool_ptrs[2][].dispatch(memcpy_kernel, pool_ptrs[2][].args_base, 1)
        pool_ptrs[1][].join()
        pool_ptrs[2][].join()

    var t_concurrent_same = List[Int]()
    for _ in range(PROBE_ITERS):
        pool_ptrs[1][].args_base[].arg0 = bases[1]
        pool_ptrs[1][].args_base[].arg1 = bases[0]
        pool_ptrs[1][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[2][].args_base[].arg0 = bases[2]
        pool_ptrs[2][].args_base[].arg1 = bases[0]
        pool_ptrs[2][].args_base[].arg2 = TOTAL_BYTES
        var t0 = perf_counter_ns()
        pool_ptrs[1][].dispatch(memcpy_kernel, pool_ptrs[1][].args_base, 1)
        pool_ptrs[2][].dispatch(memcpy_kernel, pool_ptrs[2][].args_base, 1)
        pool_ptrs[1][].join()
        pool_ptrs[2][].join()
        t_concurrent_same.append(Int(perf_counter_ns() - t0))
    sort_times(t_concurrent_same)
    var med_cs = t_concurrent_same[PROBE_ITERS // 2]
    print("  0->{1,2} concurrent (same src):", med_cs // 1000, "us,",
          Float64(TOTAL_BYTES * 2) / Float64(med_cs), "GB/s aggregate")

    # Concurrent: 0→1 AND 2→3 simultaneously (no contention — different sources)
    for _ in range(3):
        pool_ptrs[1][].args_base[].arg0 = bases[1]
        pool_ptrs[1][].args_base[].arg1 = bases[0]
        pool_ptrs[1][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[3][].args_base[].arg0 = bases[3]
        pool_ptrs[3][].args_base[].arg1 = bases[2]
        pool_ptrs[3][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[1][].dispatch(memcpy_kernel, pool_ptrs[1][].args_base, 1)
        pool_ptrs[3][].dispatch(memcpy_kernel, pool_ptrs[3][].args_base, 1)
        pool_ptrs[1][].join()
        pool_ptrs[3][].join()

    var t_concurrent_diff = List[Int]()
    for _ in range(PROBE_ITERS):
        pool_ptrs[1][].args_base[].arg0 = bases[1]
        pool_ptrs[1][].args_base[].arg1 = bases[0]
        pool_ptrs[1][].args_base[].arg2 = TOTAL_BYTES
        pool_ptrs[3][].args_base[].arg0 = bases[3]
        pool_ptrs[3][].args_base[].arg1 = bases[2]
        pool_ptrs[3][].args_base[].arg2 = TOTAL_BYTES
        var t0 = perf_counter_ns()
        pool_ptrs[1][].dispatch(memcpy_kernel, pool_ptrs[1][].args_base, 1)
        pool_ptrs[3][].dispatch(memcpy_kernel, pool_ptrs[3][].args_base, 1)
        pool_ptrs[1][].join()
        pool_ptrs[3][].join()
        t_concurrent_diff.append(Int(perf_counter_ns() - t0))
    sort_times(t_concurrent_diff)
    var med_cd = t_concurrent_diff[PROBE_ITERS // 2]
    print("  0->1 + 2->3 concurrent (diff src):", med_cd // 1000, "us,",
          Float64(TOTAL_BYTES * 2) / Float64(med_cd), "GB/s aggregate")

    # Concurrent: all 3 broadcast copies from rank 0 simultaneously
    for _ in range(3):
        for r in range(1, TP):
            pool_ptrs[r][].args_base[].arg0 = bases[r]
            pool_ptrs[r][].args_base[].arg1 = bases[0]
            pool_ptrs[r][].args_base[].arg2 = TOTAL_BYTES
            pool_ptrs[r][].dispatch(memcpy_kernel, pool_ptrs[r][].args_base, 1)
        for r in range(1, TP):
            pool_ptrs[r][].join()

    var t_all3 = List[Int]()
    for _ in range(PROBE_ITERS):
        for r in range(1, TP):
            pool_ptrs[r][].args_base[].arg0 = bases[r]
            pool_ptrs[r][].args_base[].arg1 = bases[0]
            pool_ptrs[r][].args_base[].arg2 = TOTAL_BYTES
            pool_ptrs[r][].dispatch(memcpy_kernel, pool_ptrs[r][].args_base, 1)
        var t0 = perf_counter_ns()
        for r in range(1, TP):
            pool_ptrs[r][].join()
        t_all3.append(Int(perf_counter_ns() - t0))
    sort_times(t_all3)
    var med_all3 = t_all3[PROBE_ITERS // 2]
    print("  0->{1,2,3} concurrent (all pull):", med_all3 // 1000, "us,",
          Float64(TOTAL_BYTES * 3) / Float64(med_all3), "GB/s aggregate")

    print()

    # Warmup
    for _ in range(5):
        fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
        naive_broadcast(bases, TOTAL_BYTES)
    for _ in range(5):
        fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
        for r in range(1, TP):
            clear_buffer(bases[r], TOTAL_BYTES)
        streaming_broadcast(bases, TOTAL_BYTES, pool_ptrs)

    # Naive timing
    var naive_times = List[Int]()
    for _ in range(50):
        var t0 = perf_counter_ns()
        naive_broadcast(bases, TOTAL_BYTES)
        var t1 = perf_counter_ns()
        naive_times.append(Int(t1 - t0))

    sort_times(naive_times)
    var naive_med = naive_times[25]
    var naive_us = naive_med // 1000
    var naive_bw = Float64(TOTAL_BYTES * (TP - 1)) / Float64(naive_med)
    print("  naive         :", naive_us, "us  (", TOTAL_BYTES, "B/rank,", naive_bw, "GB/s )")

    # Parallel pull timing
    var pull_times = List[Int]()
    for _ in range(50):
        var t0 = perf_counter_ns()
        parallel_pull_broadcast(bases, TOTAL_BYTES, pool_ptrs)
        var t1 = perf_counter_ns()
        pull_times.append(Int(t1 - t0))

    sort_times(pull_times)
    var pull_med = pull_times[25]
    var pull_us = pull_med // 1000
    var pull_bw = Float64(TOTAL_BYTES * (TP - 1)) / Float64(pull_med)
    print("  parallel_pull :", pull_us, "us  (", TOTAL_BYTES, "B/rank,", pull_bw, "GB/s )")

    # Streaming timing
    var stream_times = List[Int]()
    for _ in range(50):
        var t0 = perf_counter_ns()
        streaming_broadcast(bases, TOTAL_BYTES, pool_ptrs)
        var t1 = perf_counter_ns()
        stream_times.append(Int(t1 - t0))

    sort_times(stream_times)
    var stream_med = stream_times[25]
    var stream_us = stream_med // 1000
    var stream_bw = Float64(TOTAL_BYTES * (TP - 1)) / Float64(stream_med)
    print("  streaming     :", stream_us, "us  (", TOTAL_BYTES, "B/rank,", stream_bw, "GB/s )")

    print()
    print("=== Done ===")

    _ = arenas
    _ = pools


def worker_check_kernel(
    rank: Int, expected_node: Int, n2: Int,
    n3: Int, n4: Int, n5: Int,
):
    var sys = linux.linux_sys()
    var result = sys.sys_getcpu()
    print("  rank", rank, ": cpu", result[0], "node", result[1],
          "expected", expected_node,
          "OK" if result[1] == expected_node else "MISMATCH")
