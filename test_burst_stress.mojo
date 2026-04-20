"""BurstPool stress test — correctness + timing across all NUMA nodes."""

from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from numa import NumaInfo
from notstdcollections import HeapMoveArray
from std.memory import UnsafePointer
from std.collections import InlineArray
from std.time import perf_counter_ns


def mix64(x: Int) -> Int:
    var z = x + (-7046029254386353131)
    z = (z ^ (z >> 30)) * (-4658895280553007687)
    z = (z ^ (z >> 27)) * (-7723592293110705685)
    return z ^ (z >> 31)


def calc_result(iter: Int, job_idx: Int) -> Int:
    var x = mix64(iter ^ job_idx)
    var spins = Int(x & 0xFF)
    for _ in range(spins):
        x = mix64(x)
    return x


def calc_scratch_sum(iter: Int, job_idx: Int) -> Int:
    return (iter + job_idx) * 128 + 8128


@fieldwise_init
struct StressArgs(Copyable, ImplicitlyCopyable):
    var out_addr: UnsafePointer[Int, MutAnyOrigin]
    var iter: Int
    var job_idx: Int


def stress_kernel(args: StressArgs):
    var scratch = InlineArray[Int, 128](uninitialized=True)
    for i in range(128):
        scratch[i] = args.iter + args.job_idx + i

    var x = calc_result(args.iter, args.job_idx)

    var scratch_sum = 0
    for i in range(128):
        scratch_sum += scratch[i]

    args.out_addr[] = x + scratch_sum


comptime ITERATIONS = 5000


def jobs_for_iter(cap: Int, iter_i: Int) -> Int:
    var jobs = cap
    if iter_i % 5 == 1:
        jobs = cap // 2
    elif iter_i % 5 == 2:
        jobs = 1
    elif iter_i % 5 == 3:
        jobs = (cap * 3) // 4
    return max(1, jobs)


def run_stress[P: BurstThreadPool](mut pools: HeapMoveArray[P], num_nodes: Int):
    var node_args = HeapMoveArray[HeapMoveArray[StressArgs]](num_nodes)
    var node_outputs = HeapMoveArray[HeapMoveArray[Int]](num_nodes)
    for i in range(num_nodes):
        var cap = pools[i].get_capacity()
        var args = HeapMoveArray[StressArgs](cap)
        var outs = HeapMoveArray[Int](cap)
        for j in range(cap):
            args.push(StressArgs(UnsafePointer[Int, MutAnyOrigin](), 0, 0))
            outs.push(0)
        for j in range(cap):
            (args.ptr + j)[].out_addr = UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(outs.ptr + j))
        node_args.push(args^)
        node_outputs.push(outs^)

    var total_dispatch_ns = 0
    var total_join_ns = 0
    var max_dispatch_ns = 0
    var max_join_ns = 0
    var total_dispatches = 0

    var bench_start_ns = Int(perf_counter_ns())
    for iter_i in range(ITERATIONS):
        for n in range(num_nodes):
            var jobs = jobs_for_iter(pools[n].get_capacity(), iter_i)
            for j in range(jobs):
                (node_args[n].ptr + j)[].iter = iter_i
                (node_args[n].ptr + j)[].job_idx = j

        var t0 = Int(perf_counter_ns())
        for n in range(num_nodes):
            var jobs = jobs_for_iter(pools[n].get_capacity(), iter_i)
            pools[n].dispatch[StressArgs, stress_kernel](node_args[n].ptr, jobs)
        var t1 = Int(perf_counter_ns())
        for n in range(num_nodes):
            pools[n].join()
        var t2 = Int(perf_counter_ns())

        var dispatch_ns = t1 - t0
        var join_ns = t2 - t1
        total_dispatch_ns += dispatch_ns
        total_join_ns += join_ns
        total_dispatches += 1
        if dispatch_ns > max_dispatch_ns:
            max_dispatch_ns = dispatch_ns
        if join_ns > max_join_ns:
            max_join_ns = join_ns

        for n in range(num_nodes):
            var jobs = jobs_for_iter(pools[n].get_capacity(), iter_i)
            for j in range(jobs):
                var got = (node_outputs[n].ptr + j)[]
                var exp = calc_result(iter_i, j) + calc_scratch_sum(iter_i, j)
                if got != exp:
                    print("Mismatch at iter", iter_i, "node", n, "job", j,
                          "got", got, "expected", exp)
                    return

        if iter_i % 1000 == 0 and iter_i != 0:
            print("ok through iter", iter_i)

    var bench_end_ns = Int(perf_counter_ns())
    var total_ns = bench_end_ns - bench_start_ns
    var total_workers = 0
    for n in range(num_nodes):
        total_workers += pools[n].get_capacity()

    print("Stress test passed.")
    print("  iterations:", ITERATIONS, " nodes:", num_nodes,
          " total workers:", total_workers)
    print("  avg dispatch:", total_dispatch_ns // total_dispatches, "ns")
    print("  avg join:    ", total_join_ns // total_dispatches, "ns")
    print("  max dispatch:", max_dispatch_ns, "ns")
    print("  max join:    ", max_join_ns, "ns")
    print("  total:       ", total_ns // 1_000_000, "ms")


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var num_nodes = numa.num_nodes

    print(String(num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pools = HeapMoveArray[IsolatedBurstPool[]](num_nodes)
        for i in range(num_nodes):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        run_stress(pools, num_nodes)
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](num_nodes)
        for i in range(num_nodes):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        run_stress(pools, num_nodes)
