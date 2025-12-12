from threading.linux_threading import ThreadPool
from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from time import perf_counter_ns

fn empty_work():
    pass

fn empty_kernel():
    pass

fn bench_clone[POOL_SIZE: Int, ITERATIONS: Int]():
    var pool = ThreadPool(POOL_SIZE)
    if not pool:
        print("Pool creation failed")
        return

    # Warmup
    for _ in range(POOL_SIZE):
        _ = pool.launch(empty_work)
    pool.wait_all()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        for _ in range(POOL_SIZE):
            _ = pool.launch(empty_work)
        pool.wait_all()
    var end = perf_counter_ns()

    var total_ns = Int(end - start)
    var ns_per_iteration = total_ns // ITERATIONS

    print("  clone3:  ", POOL_SIZE, "threads | Per-batch:", ns_per_iteration, "ns")

fn bench_burst[POOL_SIZE: Int, ITERATIONS: Int]():
    var pool = BurstPool(POOL_SIZE)
    if not pool:
        print("BurstPool creation failed")
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

    print("  burst:   ", POOL_SIZE, "threads | Per-batch:", ns_per_iteration, "ns")

fn main():
    print("=== Dispatch Overhead Benchmark ===")
    print("(1000 iterations each)\n")

    print("Pool size 1:")
    bench_clone[1, 1000]()
    bench_burst[1, 1000]()

    print("\nPool size 4:")
    bench_clone[4, 1000]()
    bench_burst[4, 1000]()

    print("\nPool size 8:")
    bench_clone[8, 1000]()
    bench_burst[8, 1000]()

    print("\nPool size 16:")
    bench_clone[16, 1000]()
    bench_burst[16, 1000]()

    print("\nPool size 32:")
    bench_clone[32, 1000]()
    bench_burst[32, 1000]()

    print("\nPool size 64:")
    bench_clone[64, 1000]()
    bench_burst[64, 1000]()
