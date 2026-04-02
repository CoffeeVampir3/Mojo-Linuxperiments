"""Dispatch/join overhead benchmark: BurstPool cold/hot vs BurstPool.

Compares three configurations across noop and heavy (~200us) work:
  1. BurstPool cold (default spin_limit=1000)
  2. BurstPool hot (begin_forward/end_forward + headroom)
  3. BurstPool (per-worker mailbox + begin_forward/end_forward + headroom)
"""

from threading.burst_threading import BurstPool, ArgPack
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from numa import NumaInfo
from notstdcollections import HeapMoveArray


def noop_kernel(out_addr: Int, job_id: Int, unused0: Int,
                unused1: Int, unused2: Int, unused3: Int):
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = job_id + 1


def heavy_kernel(out_addr: Int, job_id: Int, iters: Int,
                 unused1: Int, unused2: Int, unused3: Int):
    var x = job_id
    for i in range(iters):
        x = x ^ (x >> 17)
        x = x * 0xBF58476D1CE4E5B9
        x = x ^ (x >> 31)
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = x


comptime LAYERS = 40
comptime HEADROOM = 2
comptime TRIALS = 10
comptime HEAVY_ITERS = 50000


def main():
    var numa = NumaInfo()
    var node = numa.plan_topology(1)[0]
    var ncpus = numa.cpus_on_node(node)
    print("Node " + String(node) + ": " + String(ncpus) + " cpus")

    var cold_pool = BurstPool[].for_numa_node(numa, node)
    var burst_hot = BurstPool[].for_numa_node_hot(numa, node, HEADROOM)
    var hp = BurstPool[].for_numa_node(numa, node, HEADROOM)

    if not cold_pool:
        print("cold_pool creation failed")
        return
    if not burst_hot:
        print("burst_hot creation failed")
        return
    if not hp:
        print("BurstPool creation failed")
        return

    print("BurstPool cold: " + String(cold_pool.capacity) + " workers")
    print("BurstPool hot:  " + String(burst_hot.capacity) + " workers")
    print("BurstPool:        " + String(hp.capacity) + " workers")

    var max_cap = max(cold_pool.capacity, max(burst_hot.capacity, hp.capacity))
    var output = HeapMoveArray[Int](max_cap)
    for i in range(max_cap):
        output.push(0)

    var bp_packs = HeapMoveArray[ArgPack](max_cap)
    var hp_packs = HeapMoveArray[ArgPack](max_cap)
    for i in range(max_cap):
        bp_packs.push(ArgPack())
        hp_packs.push(ArgPack())

    var job_counts = InlineArray[Int, 4](fill=0)
    job_counts[0] = 2
    job_counts[1] = 4
    job_counts[2] = 8
    job_counts[3] = 16

    # =====================================================================
    # Noop kernel — pure dispatch overhead
    # =====================================================================
    print("\n=== 40-layer forward, noop kernel (us/layer) ===")
    print("  jobs | cold | burst-hot | hotpool")
    print("  -----|------|-----------|--------")

    for ji in range(4):
        var jobs = job_counts[ji]
        for j in range(jobs):
            (bp_packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (bp_packs.ptr + j)[].arg1 = j
            (hp_packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (hp_packs.ptr + j)[].arg1 = j

        # Cold
        var best_cold = Int(1 << 60)
        for trial in range(TRIALS):
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                cold_pool.dispatch(noop_kernel, bp_packs.ptr, jobs)
                cold_pool.join()
            var e = Int(perf_counter_ns()) - t0
            keep(output[0])
            if e < best_cold: best_cold = e

        # Burst hot
        var best_bh = Int(1 << 60)
        for trial in range(TRIALS):
            burst_hot.begin_forward()
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                burst_hot.dispatch(noop_kernel, bp_packs.ptr, jobs)
                burst_hot.join()
            var e = Int(perf_counter_ns()) - t0
            burst_hot.end_forward()
            keep(output[0])
            if e < best_bh: best_bh = e

        # BurstPool
        var best_hp = Int(1 << 60)
        for trial in range(TRIALS):
            hp.begin_forward()
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                hp.dispatch(noop_kernel, hp_packs.ptr, jobs)
                hp.join()
            var e = Int(perf_counter_ns()) - t0
            hp.end_forward()
            keep(output[0])
            if e < best_hp: best_hp = e

        print("  " + String(jobs)
              + " | " + String(best_cold // LAYERS // 1000)
              + " | " + String(best_bh // LAYERS // 1000)
              + " | " + String(best_hp // LAYERS // 1000))

    # =====================================================================
    # Heavy kernel (~200us) — decode-like workload
    # =====================================================================
    print("\n=== 40-layer forward, ~200us work (us/layer) ===")
    print("  jobs | cold | burst-hot | hotpool")
    print("  -----|------|-----------|--------")

    for ji in range(4):
        var jobs = job_counts[ji]
        for j in range(jobs):
            (bp_packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (bp_packs.ptr + j)[].arg1 = j
            (bp_packs.ptr + j)[].arg2 = HEAVY_ITERS
            (hp_packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (hp_packs.ptr + j)[].arg1 = j
            (hp_packs.ptr + j)[].arg2 = HEAVY_ITERS

        # Cold
        var best_cold = Int(1 << 60)
        for trial in range(TRIALS):
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                cold_pool.dispatch(heavy_kernel, bp_packs.ptr, jobs)
                cold_pool.join()
            var e = Int(perf_counter_ns()) - t0
            keep(output[0])
            if e < best_cold: best_cold = e

        # Burst hot
        var best_bh = Int(1 << 60)
        for trial in range(TRIALS):
            burst_hot.begin_forward()
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                burst_hot.dispatch(heavy_kernel, bp_packs.ptr, jobs)
                burst_hot.join()
            var e = Int(perf_counter_ns()) - t0
            burst_hot.end_forward()
            keep(output[0])
            if e < best_bh: best_bh = e

        # BurstPool
        var best_hp = Int(1 << 60)
        for trial in range(TRIALS):
            hp.begin_forward()
            var t0 = Int(perf_counter_ns())
            for layer in range(LAYERS):
                hp.dispatch(heavy_kernel, hp_packs.ptr, jobs)
                hp.join()
            var e = Int(perf_counter_ns()) - t0
            hp.end_forward()
            keep(output[0])
            if e < best_hp: best_hp = e

        print("  " + String(jobs)
              + " | " + String(best_cold // LAYERS // 1000)
              + " | " + String(best_bh // LAYERS // 1000)
              + " | " + String(best_hp // LAYERS // 1000))
