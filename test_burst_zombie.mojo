"""Stress test for BurstPool destructor — provokes the TOCTOU shutdown race.

Creates and destroys pools repeatedly with dispatch/join cycles in between.
If the old futex_wait race in __del__ triggers, the process hangs forever.
If this test completes, the destructor is working correctly.
"""

from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns


def noop_kernel(a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int):
    pass

def busy_kernel(a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int):
    var x = a0
    for _ in range(200):
        x = x ^ (x << 13)
        x = x ^ (x >> 7)
        x = x ^ (x << 17)
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=a1)[] = x


def main():
    comptime POOL_CYCLES = 5000
    comptime DISPATCHES = 200
    comptime CAPACITY = 15

    var packs = HeapMoveArray[ArgPack](CAPACITY)
    for _ in range(CAPACITY):
        packs.push(ArgPack())

    var sink = HeapMoveArray[Int](CAPACITY)
    for _ in range(CAPACITY):
        sink.push(0)

    var t_start = Int(perf_counter_ns())

    for cycle in range(POOL_CYCLES):
        var tc0 = Int(perf_counter_ns())

        var pool = BurstPool[](CAPACITY)
        if not pool:
            print("cycle", cycle, ": pool creation failed")
            continue

        # Cold mode dispatch/join
        for d in range(DISPATCHES // 2):
            var jobs = CAPACITY
            if d % 3 == 1:
                jobs = 1
            elif d % 3 == 2:
                jobs = CAPACITY // 2
            for j in range(jobs):
                packs.ptr[j].arg0 = cycle * 1000 + d
                packs.ptr[j].arg1 = Int(sink.ptr + j)
            pool.dispatch(busy_kernel, packs.ptr, jobs)
            pool.join()

        # Hot mode dispatch/join
        pool.begin_forward()
        for d in range(DISPATCHES // 2):
            var jobs = CAPACITY
            if d % 4 == 0:
                jobs = CAPACITY // 3
            for j in range(jobs):
                packs.ptr[j].arg0 = cycle * 1000 + d
                packs.ptr[j].arg1 = Int(sink.ptr + j)
            pool.dispatch(busy_kernel, packs.ptr, jobs)
            pool.join()
        pool.end_forward()

        # Rapid noop dispatches
        for _ in range(20):
            pool.dispatch(noop_kernel, packs.ptr, CAPACITY)
            pool.join()

        # Destructor runs here — the thing we're testing
        pool^.__del__()

        var tc1 = Int(perf_counter_ns())
        var cycle_ms = (tc1 - tc0) // 1_000_000
        if cycle % 10 == 0:
            print("cycle", cycle, ": ok (", cycle_ms, "ms)")

    var elapsed_ms = (Int(perf_counter_ns()) - t_start) // 1_000_000
    print()
    print("PASS:", POOL_CYCLES, "pool lifecycles,", POOL_CYCLES * DISPATCHES, "total dispatches,", elapsed_ms, "ms")
