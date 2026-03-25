"""Ring allreduce and broadcast with persistent per-NUMA workers.

Bandwidth-optimal ring allreduce:
  1. Reduce-scatter: tp-1 steps. Each step, every rank accumulates one
     chunk from its predecessor. All tp links active every step.
     After completion, rank r owns fully-reduced chunk (r+1)%tp.
  2. Allgather: tp-1 steps. Each step, every rank forwards one chunk
     to its successor. All tp links active every step.
     After completion, all ranks have the full result.

Threading: one persistent worker per NUMA node, dispatched once for the
entire protocol. Workers synchronize via generation-based spin barriers
between steps. Zero dispatch overhead between steps.

Broadcast: pipelined ring from rank 0. Data split into chunks that
flow through the ring, one hop per step. Steady-state has all links
active simultaneously.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.os.atomic import Atomic, Consistency
from std.collections import InlineArray
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from simd_math import bf16_load_as
import linux.sys as linux

from modeling.model_spec import (
    Encoding, Shaped, BF16, Slot, Replicated,
)
from numa import get_current_cpu_and_node

comptime PoolPtr = UnsafePointer[BurstPool[], MutAnyOrigin]
comptime AtomicInt32 = Atomic[DType.int32]

comptime TP = 4
comptime COLS_MEDIUM = 4096
comptime COLS_LARGE = 4096*8
comptime MAX_SEQ = 1024


# =============================================================================
# Barrier
# =============================================================================


@align(64)
struct BarrierState:
    """Generation-based spin barrier. Fresh instance per dispatch."""
    var gen: AtomicInt32
    var count: AtomicInt32

    def __init__(out self):
        self.gen = AtomicInt32(0)
        self.count = AtomicInt32(0)


@always_inline
def spin_barrier(
    bar: UnsafePointer[BarrierState, MutAnyOrigin],
    mut my_gen: Int,
    tp: Int,
):
    my_gen += 1
    var gen_ptr = UnsafePointer(to=bar[].gen.value)
    var count_ptr = UnsafePointer(to=bar[].count.value)
    var old = AtomicInt32.fetch_add[ordering=Consistency.ACQUIRE_RELEASE](count_ptr, 1)
    if Int(old) + 1 == tp:
        AtomicInt32.store[ordering=Consistency.MONOTONIC](count_ptr, 0)
        AtomicInt32.store[ordering=Consistency.RELEASE](gen_ptr, Int32(my_gen))
    else:
        var sys = linux.linux_sys()
        while Int(AtomicInt32.load[ordering=Consistency.ACQUIRE](gen_ptr)) < my_gen:
            sys.arch_cpu_relax()


# =============================================================================
# Accumulate helper
# =============================================================================


@always_inline
def accumulate_range(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.float32]()
    var i = 0
    while i + width <= count:
        var s = bf16_load_as[DType.float32, width](src, i)
        var d = bf16_load_as[DType.float32, width](dst, i)
        (dst + i).store((s + d).cast[DType.bfloat16]())
        i += width
    while i < count:
        dst[i] = Scalar[DType.bfloat16](Float32(dst[i]) + Float32(src[i]))
        i += 1


# =============================================================================
# Ring Allreduce kernel
# =============================================================================


def ring_allreduce_kernel(
    rank: Int, ptrs_addr: Int, total_elements: Int,
    barrier_addr: Int, tp: Int, n5: Int,
):
    """Full reduce-scatter + allgather protocol. Runs on each worker."""
    var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=ptrs_addr)
    var bar = UnsafePointer[BarrierState, MutAnyOrigin](unsafe_from_address=barrier_addr)
    var chunk = total_elements // tp
    var rem = total_elements - chunk * tp
    var my_gen = 0
    var prev = (rank - 1 + tp) % tp
    var next = (rank + 1) % tp
    var my_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[rank])
    var prev_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[prev])
    var my_addr = ptrs[rank]
    var next_addr = ptrs[next]

    # Reduce-scatter: each step, accumulate one chunk from predecessor.
    for step in range(tp - 1):
        var c = (rank - step - 1 + tp) % tp
        var start = c * chunk
        var count = chunk + (rem if c == tp - 1 else 0)
        accumulate_range(my_buf + start, prev_buf + start, count)
        spin_barrier(bar, my_gen, tp)

    # Allgather: each step, copy one chunk to successor.
    for step in range(tp - 1):
        var c = (rank + 1 - step + tp) % tp
        var start = c * chunk
        var count = chunk + (rem if c == tp - 1 else 0)
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=next_addr + start * 2),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=my_addr + start * 2),
            count=count * 2,
        )
        spin_barrier(bar, my_gen, tp)


# =============================================================================
# Ring Broadcast kernel
# =============================================================================


def ring_broadcast_kernel(
    rank: Int, ptrs_addr: Int, total_bytes: Int,
    barrier_addr: Int, tp: Int, nchunks: Int,
):
    """Pipelined broadcast from rank 0. Each step, rank r forwards
    chunk (step - rank) to rank r+1, if valid."""
    var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=ptrs_addr)
    var bar = UnsafePointer[BarrierState, MutAnyOrigin](unsafe_from_address=barrier_addr)
    var chunk_bytes = total_bytes // nchunks
    var byte_rem = total_bytes - chunk_bytes * nchunks
    var my_gen = 0
    var next = rank + 1
    var total_steps = (tp - 1) + (nchunks - 1)

    for s in range(total_steps):
        var c = s - rank
        if c >= 0 and c < nchunks and next < tp:
            var offset = c * chunk_bytes
            var size = chunk_bytes + (byte_rem if c == nchunks - 1 else 0)
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[next] + offset),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[rank] + offset),
                count=size,
            )
        spin_barrier(bar, my_gen, tp)


# =============================================================================
# Diagnostic kernel — reports actual CPU/node of worker thread
# =============================================================================


def worker_placement_kernel(
    rank: Int, expected_node: Int, n2: Int,
    n3: Int, n4: Int, n5: Int,
):
    """Reports which CPU and NUMA node this worker is actually on."""
    var sys = linux.linux_sys()
    var result = sys.sys_getcpu()
    var cpu = result[0]
    var node = result[1]
    var ok = node == expected_node
    print("  rank", rank, ": cpu", cpu, "node", node,
          "expected", expected_node,
          "OK" if ok else "MISMATCH")


# =============================================================================
# Dispatch wrappers
# =============================================================================


def ring_allreduce(
    ptrs: InlineArray[Int, TP],
    total_elements: Int,
    pool_ptrs: InlineArray[PoolPtr, TP],
):
    if total_elements <= 0 or TP <= 1:
        return
    var bar = BarrierState()
    for r in range(TP):
        var pack = pool_ptrs[r][].args_base
        pack[].arg0 = r
        pack[].arg1 = Int(UnsafePointer(to=ptrs))
        pack[].arg2 = total_elements
        pack[].arg3 = Int(UnsafePointer(to=bar))
        pack[].arg4 = TP
        pool_ptrs[r][].dispatch(ring_allreduce_kernel, pool_ptrs[r][].args_base, 1)
    for r in range(TP):
        pool_ptrs[r][].join()


def ring_broadcast(
    src_ptr: Int,
    ptrs: InlineArray[Int, TP],
    total_bytes: Int,
    pool_ptrs: InlineArray[PoolPtr, TP],
):
    if total_bytes <= 0 or TP <= 1:
        return
    if src_ptr != ptrs[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )
    var bar = BarrierState()
    for r in range(TP):
        var pack = pool_ptrs[r][].args_base
        pack[].arg0 = r
        pack[].arg1 = Int(UnsafePointer(to=ptrs))
        pack[].arg2 = total_bytes
        pack[].arg3 = Int(UnsafePointer(to=bar))
        pack[].arg4 = TP
        pack[].arg5 = TP  # nchunks = tp for max pipeline depth
        pool_ptrs[r][].dispatch(ring_broadcast_kernel, pool_ptrs[r][].args_base, 1)
    for r in range(TP):
        pool_ptrs[r][].join()


# =============================================================================
# Helpers
# =============================================================================


def fill_value(ptr: Int, elements: Int, value: Float32):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    var v = Scalar[DType.bfloat16](value)
    for i in range(elements):
        p[i] = v


def fill_pattern(ptr: Int, elements: Int, seed: Int):
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(elements):
        p[i] = UInt16((seed + i * 7 + 13) & 0xFFFF)


def read_f32(ptr: Int, idx: Int) -> Float32:
    return Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)[idx])


def buffers_match(a: Int, b: Int, elements: Int) -> Bool:
    var pa = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=a)
    var pb = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=b)
    for i in range(elements):
        if pa[i] != pb[i]:
            return False
    return True


# =============================================================================
# Benchmark harness
# =============================================================================


def bench_allreduce(
    name: String,
    cols: Int,
    seq_len: Int,
    bases: InlineArray[Int, TP],
    vals: InlineArray[Float32, TP],
    pool_ptrs: InlineArray[PoolPtr, TP],
    warmup: Int = 5,
    iters: Int = 50,
):
    var total = seq_len * cols
    var total_bytes = total * 2

    for _ in range(warmup):
        for r in range(TP):
            fill_value(bases[r], total, vals[r])
        ring_allreduce(bases, total, pool_ptrs)

    var times = List[Int]()
    for _ in range(iters):
        for r in range(TP):
            fill_value(bases[r], total, vals[r])
        var t0 = perf_counter_ns()
        ring_allreduce(bases, total, pool_ptrs)
        var t1 = perf_counter_ns()
        times.append(Int(t1 - t0))

    sort_times(times)
    var med = times[len(times) // 2]
    var med_us = med // 1000
    var data = total_bytes * TP
    var bw = Float64(data) / Float64(med)
    print("  ", name, ":", med_us, "us  (", total_bytes, "B/rank,", bw, "GB/s )")


def bench_broadcast(
    name: String,
    cols: Int,
    seq_len: Int,
    bases: InlineArray[Int, TP],
    pool_ptrs: InlineArray[PoolPtr, TP],
    warmup: Int = 5,
    iters: Int = 50,
):
    var total = seq_len * cols
    var total_bytes = total * 2

    for _ in range(warmup):
        fill_pattern(bases[0], total, 42)
        ring_broadcast(bases[0], bases, total_bytes, pool_ptrs)

    var times = List[Int]()
    for _ in range(iters):
        fill_pattern(bases[0], total, 42)
        var t0 = perf_counter_ns()
        ring_broadcast(bases[0], bases, total_bytes, pool_ptrs)
        var t1 = perf_counter_ns()
        times.append(Int(t1 - t0))

    sort_times(times)
    var med = times[len(times) // 2]
    var med_us = med // 1000
    var data = total_bytes * (TP - 1)
    var bw = Float64(data) / Float64(med)
    print("  ", name, ":", med_us, "us  (", total_bytes, "B/rank,", bw, "GB/s )")


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
    print("=== Ring Allreduce/Broadcast (TP=" + String(TP) + ") ===")
    print()

    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)
    print("NUMA:", numa.num_nodes, "nodes")
    print("Ring:", end=" ")
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

    # Arenas — sized for largest test (4096 cols × 1024 seq × 2B)
    var arena_size = MAX_SEQ * COLS_LARGE * 2
    var arenas = HeapMoveArray[NumaArena[alignment=64]](TP)
    var bases = InlineArray[Int, TP](fill=0)

    for rank in range(TP):
        var arena = NumaArena[alignment=64](topo[rank], arena_size)
        if not arena:
            print("FATAL: arena alloc failed rank", rank)
            return
        bases[rank] = Int(arena.base)
        arenas.push(arena^)

    for rank in range(TP):
        _ = arenas[rank].prefault()
        var ok = arenas[rank].verify_placement()
        print("rank", rank, "-> node", topo[rank], ":", "OK" if ok else "MISS")
    print()

    # Per-node pools (1 worker each, pinned)
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

    # =================================================================
    # Validate: worker thread placement
    # =================================================================
    print("--- Worker placement ---")
    for rank in range(TP):
        var pack = pool_ptrs[rank][].args_base
        pack[].arg0 = rank
        pack[].arg1 = topo[rank]
        pool_ptrs[rank][].dispatch(worker_placement_kernel, pool_ptrs[rank][].args_base, 1)
    for rank in range(TP):
        pool_ptrs[rank][].join()
    print()

    # =================================================================
    # Validate: memory placement (probe multiple offsets per arena)
    # =================================================================
    print("--- Memory placement ---")
    var sys = linux.linux_sys()
    var page_size = 4096
    for rank in range(TP):
        var expected_node = topo[rank]
        var pages_to_check = min(8, arena_size // page_size)
        var ok_count = 0
        var fail_count = 0
        for p in range(pages_to_check):
            var addr = bases[rank] + p * (arena_size // pages_to_check)
            var actual_node = sys.sys_move_pages_query(addr)
            if actual_node == expected_node:
                ok_count += 1
            else:
                fail_count += 1
                if fail_count <= 2:
                    print("  rank", rank, "page", p, "at offset",
                          p * (arena_size // pages_to_check),
                          "-> node", actual_node, "expected", expected_node)
        print("  rank", rank, ":", ok_count, "/", pages_to_check,
              "pages on node", expected_node,
              "PASS" if fail_count == 0 else "FAIL")
    print()

    var vals = InlineArray[Float32, TP](fill=Float32(0))
    vals[0] = Float32(1.0)
    vals[1] = Float32(2.0)
    vals[2] = Float32(3.0)
    vals[3] = Float32(4.0)

    # =================================================================
    # Correctness
    # =================================================================
    print("--- Correctness ---")

    # Allreduce
    var total_med = 64 * COLS_MEDIUM
    for r in range(TP):
        fill_value(bases[r], total_med, vals[r])
    ring_allreduce(bases, total_med, pool_ptrs)

    var expected = Float32(0)
    for r in range(TP):
        expected += Float32(Scalar[DType.bfloat16](vals[r]))
    var expected_bf16 = Float32(Scalar[DType.bfloat16](expected))

    var ar_ok = True
    for r in range(TP):
        var spots = InlineArray[Int, 3](fill=0)
        spots[0] = 0
        spots[1] = total_med // 2
        spots[2] = total_med - 1
        for ci in range(3):
            var v = read_f32(bases[r], spots[ci])
            if v != expected_bf16:
                print("FAIL allreduce rank", r, "idx", spots[ci], "got", v, "expected", expected_bf16)
                ar_ok = False
    if ar_ok:
        for r in range(1, TP):
            if not buffers_match(bases[0], bases[r], total_med):
                print("FAIL allreduce rank", r, "differs from rank 0")
                ar_ok = False
    print("allreduce:", "PASS" if ar_ok else "FAIL",
          "( sum =", expected_bf16, ",", total_med, "elements )")

    # Broadcast
    fill_pattern(bases[0], total_med, 42)
    ring_broadcast(bases[0], bases, total_med * 2, pool_ptrs)
    var bc_ok = True
    for r in range(1, TP):
        if not buffers_match(bases[0], bases[r], total_med):
            print("FAIL broadcast rank", r, "differs from rank 0")
            bc_ok = False
    print("broadcast:", "PASS" if bc_ok else "FAIL")
    print()

    # =================================================================
    # Benchmark: Allreduce
    # =================================================================
    # Same sizes as original test_rings.mojo for direct comparison

    print("--- Allreduce: hidden=576, seq=1 (decode) ---")
    bench_allreduce("ring", COLS_MEDIUM, 1, bases, vals, pool_ptrs)

    print()
    print("--- Allreduce: hidden=576, seq=64 (prefill) ---")
    bench_allreduce("ring", COLS_MEDIUM, 64, bases, vals, pool_ptrs)

    print()
    print("--- Allreduce: hidden=4096, seq=256 ---")
    bench_allreduce("ring", COLS_LARGE, 256, bases, vals, pool_ptrs)

    print()

    # =================================================================
    # Benchmark: Broadcast
    # =================================================================

    print("--- Broadcast: hidden=576, seq=64 ---")
    bench_broadcast("ring", COLS_MEDIUM, 64, bases, pool_ptrs)

    print()
    print("--- Broadcast: hidden=4096, seq=256 ---")
    bench_broadcast("ring", COLS_LARGE, 256, bases, pool_ptrs)

    print()
    print("=== Done ===")

    _ = arenas
    _ = pools
