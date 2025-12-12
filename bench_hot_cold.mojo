"""Benchmark work queue dispatch latency with spin-then-sleep model."""

from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from numa import NumaInfo, CpuMask
import threading.linux as linux
from time import perf_counter_ns

fn empty_kernel():
    pass

fn bench[POOL_SIZE: Int, ITERATIONS: Int]():
    """Benchmark spin-then-sleep dispatch latency."""
    var pool = BurstPool(POOL_SIZE)
    if not pool:
        print("Pool creation failed")
        return

    var packs = HeapMoveArray[ArgPack](pool.capacity)
    for _ in range(pool.capacity):
        packs.push(ArgPack())

    # Warmup
    pool.dispatch(empty_kernel, packs.ptr)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        pool.dispatch(empty_kernel, packs.ptr)
    var end = perf_counter_ns()

    var total_ns = Int(end - start)
    var ns_per_iteration = total_ns // ITERATIONS

    print("  ", POOL_SIZE, "threads | Per-batch:", ns_per_iteration, "ns")

fn bench_pinned[ITERATIONS: Int]():
    """Benchmark with main thread pinned, workers on remaining NUMA node CPUs."""
    var numa = NumaInfo()
    if numa.num_nodes == 0:
        print("  NUMA info not available")
        return

    # Pin main thread to CPU 0
    var main_cpu = 0
    var main_mask = CpuMask[128]()
    main_mask.set(main_cpu)
    _ = linux.sys_sched_setaffinity(0, 128, Int(main_mask.ptr()))

    # Create pool on node 0, excluding main thread's CPU
    var pool = BurstPool.for_numa_node_excluding(numa, 0, main_cpu)
    if not pool:
        print("  Pool creation failed")
        return

    var pool_size = pool.capacity

    var packs = HeapMoveArray[ArgPack](pool.capacity)
    for _ in range(pool.capacity):
        packs.push(ArgPack())

    # Benchmark
    pool.dispatch(empty_kernel, packs.ptr)  # warmup
    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        pool.dispatch(empty_kernel, packs.ptr)
    var end = perf_counter_ns()
    var ns_per_iter = Int(end - start) // ITERATIONS

    print("  pinned:", pool_size, "threads | Per-batch:", ns_per_iter, "ns")

fn main():
    print("=== Work Queue Dispatch Benchmark ===")
    print("Workers use spin-then-sleep: spin briefly, then futex_wait")
    print("(1000 iterations each)\n")

    print("Pool size 1:")
    bench[1, 1000]()

    print("\nPool size 4:")
    bench[4, 1000]()

    print("\nPool size 8:")
    bench[8, 1000]()

    print("\nPool size 16:")
    bench[16, 1000]()

    print("\nPool size 32:")
    bench[32, 1000]()

    print("\nPool size 64:")
    bench[64, 1000]()

    print("\nPool size 128 (oversubscribed):")
    bench[128, 1000]()

    print("\n--- Pinned test (main on CPU 0, workers on rest of NUMA node 0) ---")
    bench_pinned[1000]()
