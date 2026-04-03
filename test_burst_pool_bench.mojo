"""BurstPool dispatch/join overhead benchmark.

Automatically detects CPU isolation and runs the appropriate model:
  Isolated cores: BurstPool[IsolatedModel] — zero-syscall dispatch.
  No isolation:   BurstPool[ColdModel] — futex-based sleep/wake.

Reports min / median / p99 in nanoseconds per layer.
"""

from threading.burst_threading import BurstPool, ColdModel, IsolatedModel, ThreadingModel, ArgPack, make_node_pools
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
    for _ in range(iters):
        x = x ^ (x >> 17)
        x = x * 0xBF58476D1CE4E5B9
        x = x ^ (x >> 31)
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = x


comptime LAYERS = 40
comptime WARMUP = 5
comptime TRIALS = 50
comptime HEAVY_ITERS = 50000


def sort_ints(p: UnsafePointer[Int, MutAnyOrigin], n: Int):
    for i in range(1, n):
        var key = (p + i)[]
        var j = i - 1
        while j >= 0 and (p + j)[] > key:
            (p + j + 1)[] = (p + j)[]
            j -= 1
        (p + j + 1)[] = key


def rjust(val: Int, width: Int) -> String:
    var s = String(val)
    while len(s) < width:
        s = " " + s
    return s


def fmt_stats(p: UnsafePointer[Int, MutAnyOrigin]) -> String:
    return (rjust(p[], 7) + " /"
          + rjust((p + TRIALS // 2)[], 7) + " /"
          + rjust((p + TRIALS * 99 // 100)[], 7))


def bench_pool[Model: ThreadingModel, F: TrivialRegisterPassable](
    mut pool: BurstPool[Model],
    kernel: F,
    label: String,
    packs: UnsafePointer[ArgPack, MutAnyOrigin],
    output: UnsafePointer[Int, MutAnyOrigin],
    dispatch_t: UnsafePointer[Int, MutAnyOrigin],
    join_t: UnsafePointer[Int, MutAnyOrigin],
    total_t: UnsafePointer[Int, MutAnyOrigin],
    job_counts: InlineArray[Int, 8],
    n_counts: Int,
):
    """Run benchmark for one pool across all job counts."""
    var cap = pool.capacity

    for ji in range(n_counts):
        var jobs = job_counts[ji]
        if jobs > cap:
            continue

        for _ in range(WARMUP):
            for _ in range(LAYERS):
                pool.dispatch(kernel, packs, jobs)
                pool.join()

        for trial in range(TRIALS):
            var disp_accum = 0
            var join_accum = 0
            var t_start = Int(perf_counter_ns())
            for _ in range(LAYERS):
                var td0 = Int(perf_counter_ns())
                pool.dispatch(kernel, packs, jobs)
                var td1 = Int(perf_counter_ns())
                pool.join()
                var td2 = Int(perf_counter_ns())
                disp_accum += td1 - td0
                join_accum += td2 - td1
            var t_total = Int(perf_counter_ns()) - t_start
            keep(output[])
            (dispatch_t + trial)[] = disp_accum // LAYERS
            (join_t + trial)[] = join_accum // LAYERS
            (total_t + trial)[] = t_total // LAYERS

        sort_ints(dispatch_t, TRIALS)
        sort_ints(join_t, TRIALS)
        sort_ints(total_t, TRIALS)
        print(rjust(jobs, 4) + " " + label
            + "  disp " + fmt_stats(dispatch_t)
            + "  join " + fmt_stats(join_t)
            + "  total " + fmt_stats(total_t))


def main():
    var numa = NumaInfo()
    var node = numa.plan_topology(1)[0]
    var ncpus = numa.cpus_on_node(node)
    var isolated = numa.has_isolation()
    print("Node " + String(node) + ": " + String(ncpus) + " cpus"
        + (", isolation active" if isolated else ", no isolation"))

    # Calibrate heavy kernel
    var cal_out = 0
    var cal_t0 = Int(perf_counter_ns())
    heavy_kernel(Int(UnsafePointer(to=cal_out)), 42, HEAVY_ITERS, 0, 0, 0)
    var cal_dur = Int(perf_counter_ns()) - cal_t0
    print("Heavy kernel: ~" + String(cal_dur // 1000) + " us ("
        + String(HEAVY_ITERS) + " iters)")
    print(String(TRIALS) + " trials, " + String(WARMUP) + " warmup, "
        + String(LAYERS) + " layers")

    if isolated:
        var pool = BurstPool[IsolatedModel].for_topology(numa, node)
        if not pool:
            print("pool creation failed")
            return
        print("IsolatedModel: " + String(pool.capacity) + " workers")

        var cap = pool.capacity
        var job_counts = InlineArray[Int, 8](fill=0)
        var n_counts = 0
        if cap >= 2:
            job_counts[n_counts] = 2
            n_counts += 1
        if cap >= 4:
            job_counts[n_counts] = 4
            n_counts += 1
        if cap >= 8:
            job_counts[n_counts] = 8
            n_counts += 1
        if cap >= 16:
            job_counts[n_counts] = 16
            n_counts += 1
        if cap > 16:
            job_counts[n_counts] = cap
            n_counts += 1

        var output = HeapMoveArray[Int](cap)
        for _ in range(cap):
            output.push(0)
        var packs = HeapMoveArray[ArgPack](cap)
        for _ in range(cap):
            packs.push(ArgPack())
        var dispatch_t = HeapMoveArray[Int](TRIALS)
        var join_t = HeapMoveArray[Int](TRIALS)
        var total_t = HeapMoveArray[Int](TRIALS)
        for _ in range(TRIALS):
            dispatch_t.push(0)
            join_t.push(0)
            total_t.push(0)

        print("\n=== noop kernel, ns/layer (min / med / p99) ===")
        for j in range(cap):
            (packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (packs.ptr + j)[].arg1 = j
        bench_pool(pool, noop_kernel, "iso  ", packs.ptr, output.ptr,
            dispatch_t.ptr, join_t.ptr, total_t.ptr,
            job_counts, n_counts)

        print("\n=== heavy kernel ~" + String(cal_dur // 1000)
            + "us, ns/layer (min / med / p99) ===")
        for j in range(cap):
            (packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (packs.ptr + j)[].arg1 = j
            (packs.ptr + j)[].arg2 = HEAVY_ITERS
        bench_pool(pool, heavy_kernel, "iso  ", packs.ptr, output.ptr,
            dispatch_t.ptr, join_t.ptr, total_t.ptr,
            job_counts, n_counts)

    else:
        var pool = BurstPool[ColdModel].for_topology(numa, node)
        if not pool:
            print("pool creation failed")
            return
        print("ColdModel: " + String(pool.capacity) + " workers")

        var cap = pool.capacity
        var job_counts = InlineArray[Int, 8](fill=0)
        var n_counts = 0
        if cap >= 2:
            job_counts[n_counts] = 2
            n_counts += 1
        if cap >= 4:
            job_counts[n_counts] = 4
            n_counts += 1
        if cap >= 8:
            job_counts[n_counts] = 8
            n_counts += 1
        if cap >= 16:
            job_counts[n_counts] = 16
            n_counts += 1
        if cap > 16:
            job_counts[n_counts] = cap
            n_counts += 1

        var output = HeapMoveArray[Int](cap)
        for _ in range(cap):
            output.push(0)
        var packs = HeapMoveArray[ArgPack](cap)
        for _ in range(cap):
            packs.push(ArgPack())
        var dispatch_t = HeapMoveArray[Int](TRIALS)
        var join_t = HeapMoveArray[Int](TRIALS)
        var total_t = HeapMoveArray[Int](TRIALS)
        for _ in range(TRIALS):
            dispatch_t.push(0)
            join_t.push(0)
            total_t.push(0)

        print("\n=== noop kernel, ns/layer (min / med / p99) ===")
        for j in range(cap):
            (packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (packs.ptr + j)[].arg1 = j
        bench_pool(pool, noop_kernel, "cold ", packs.ptr, output.ptr,
            dispatch_t.ptr, join_t.ptr, total_t.ptr,
            job_counts, n_counts)

        print("\n=== heavy kernel ~" + String(cal_dur // 1000)
            + "us, ns/layer (min / med / p99) ===")
        for j in range(cap):
            (packs.ptr + j)[].arg0 = Int(output.ptr + j)
            (packs.ptr + j)[].arg1 = j
            (packs.ptr + j)[].arg2 = HEAVY_ITERS
        bench_pool(pool, heavy_kernel, "cold ", packs.ptr, output.ptr,
            dispatch_t.ptr, join_t.ptr, total_t.ptr,
            job_counts, n_counts)
