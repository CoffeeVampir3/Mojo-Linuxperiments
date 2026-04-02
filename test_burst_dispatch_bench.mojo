"""Dispatch/join overhead benchmark for BurstPool.

Measures the pure threading overhead by dispatching near-zero-work kernels.
Sweeps: worker counts, job counts, back-to-back dispatches (simulating
multi-layer forward pass), and hot vs cold workers.

The goal is to isolate dispatch+wake+claim+join latency from actual compute.
"""

from threading.burst_threading import BurstPool, ArgPack
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from numa import NumaInfo
from notstdcollections import HeapMoveArray


# Near-zero-work kernel: just write a sentinel to prove it ran.
def noop_kernel(out_addr: Int, job_id: Int, unused0: Int,
                unused1: Int, unused2: Int, unused3: Int):
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = job_id + 1


# Light work kernel: ~1us of compute (a few hundred mix iterations).
def light_kernel(out_addr: Int, job_id: Int, iters: Int,
                 unused1: Int, unused2: Int, unused3: Int):
    var x = job_id
    for i in range(iters):
        x = x ^ (x >> 17)
        x = x * 0xBF58476D1CE4E5B9
        x = x ^ (x >> 31)
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = x


def main():
    var numa = NumaInfo()
    var node = numa.plan_topology(1)[0]
    var ncpus = numa.cpus_on_node(node)
    print("Node " + String(node) + ": " + String(ncpus) + " cpus")

    var pool = BurstPool[].for_numa_node(numa, node)
    var cap = pool.capacity
    print("Pool capacity: " + String(cap))

    var output = HeapMoveArray[Int](cap)
    for i in range(cap):
        output.push(0)

    # =====================================================================
    # Test 1: Single dispatch/join — cold workers (measure futex wake cost)
    # Vary job count: 1, 2, 4, 8, 16, cap/2, cap
    # =====================================================================
    print("\n=== Test 1: Single cold dispatch/join (noop kernel) ===")
    print("  jobs | dispatch us | join us | total us")
    print("  -----|------------|---------|--------")

    var job_counts = InlineArray[Int, 7](fill=0)
    job_counts[0] = 1
    job_counts[1] = 2
    job_counts[2] = 4
    job_counts[3] = 8
    job_counts[4] = 16
    job_counts[5] = cap // 2
    job_counts[6] = cap

    for ji in range(7):
        var jobs = job_counts[ji]
        if jobs > cap:
            continue

        # Warmup once to page in everything
        for j in range(jobs):
            var pack = pool.args_base + j
            pack[].arg0 = Int(output.ptr + j)
            pack[].arg1 = j
        pool.dispatch(noop_kernel, pool.args_base, jobs)
        pool.join()

        # Measure 20 trials with cold gap (workers sleep between trials)
        var best_dispatch = Int(1 << 60)
        var best_join = Int(1 << 60)
        var best_total = Int(1 << 60)
        for trial in range(20):
            # Let workers go cold
            var cold_start = Int(perf_counter_ns())
            while Int(perf_counter_ns()) - cold_start < 200_000:  # 200us sleep
                pass

            for j in range(jobs):
                var pack = pool.args_base + j
                pack[].arg0 = Int(output.ptr + j)
                pack[].arg1 = j

            var t0 = Int(perf_counter_ns())
            pool.dispatch(noop_kernel, pool.args_base, jobs)
            var t1 = Int(perf_counter_ns())
            pool.join()
            var t2 = Int(perf_counter_ns())

            keep(output[0])
            var d = t1 - t0
            var jn = t2 - t1
            if d + jn < best_total:
                best_dispatch = d
                best_join = jn
                best_total = d + jn

        print("  " + String(jobs)
              + " | " + String(best_dispatch // 1000)
              + " | " + String(best_join // 1000)
              + " | " + String(best_total // 1000))

    # =====================================================================
    # Test 2: Back-to-back dispatches — measures hot-path overhead
    # (Workers stay hot between dispatches within a burst)
    # =====================================================================
    print("\n=== Test 2: Back-to-back dispatches (40 layers, simulated forward) ===")
    print("  jobs | avg dispatch+join us | total 40-layer us")
    print("  -----|---------------------|------------------")

    comptime LAYERS = 40

    for ji in range(7):
        var jobs = job_counts[ji]
        if jobs > cap:
            continue

        for j in range(jobs):
            var pack = pool.args_base + j
            pack[].arg0 = Int(output.ptr + j)
            pack[].arg1 = j

        # Warmup
        for warmup in range(2):
            for layer in range(LAYERS):
                pool.dispatch(noop_kernel, pool.args_base, jobs)
                pool.join()

        var best_total = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                pool.dispatch(noop_kernel, pool.args_base, jobs)
                pool.join()
            var elapsed = Int(perf_counter_ns()) - t0
            keep(output[0])
            if elapsed < best_total:
                best_total = elapsed

        var avg_per_layer = best_total // LAYERS // 1000
        print("  " + String(jobs)
              + " | " + String(avg_per_layer)
              + " | " + String(best_total // 1000))

    # =====================================================================
    # Test 3: Back-to-back with light work (~1us per worker)
    # Shows dispatch overhead relative to actual compute
    # =====================================================================
    print("\n=== Test 3: Back-to-back with ~1us work per worker ===")
    print("  jobs | avg dispatch+join us | compute fraction")
    print("  -----|---------------------|------------------")

    comptime WORK_ITERS = 200  # ~1us of compute

    for ji in range(7):
        var jobs = job_counts[ji]
        if jobs > cap:
            continue

        for j in range(jobs):
            var pack = pool.args_base + j
            pack[].arg0 = Int(output.ptr + j)
            pack[].arg1 = j
            pack[].arg2 = WORK_ITERS

        # Warmup
        for warmup in range(2):
            for layer in range(LAYERS):
                pool.dispatch(light_kernel, pool.args_base, jobs)
                pool.join()

        var best_total = Int(1 << 60)
        for trial in range(10):
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                pool.dispatch(light_kernel, pool.args_base, jobs)
                pool.join()
            var elapsed = Int(perf_counter_ns()) - t0
            keep(output[0])
            if elapsed < best_total:
                best_total = elapsed

        var avg_per_layer = best_total // LAYERS // 1000

        # Estimate compute fraction: ~1us work, rest is overhead
        var overhead_pct = 0
        if avg_per_layer > 1:
            overhead_pct = 100 - (100 // avg_per_layer)

        print("  " + String(jobs)
              + " | " + String(avg_per_layer)
              + " | ~" + String(100 - overhead_pct) + "% compute")
