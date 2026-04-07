"""Multi-node dispatch/join benchmark.

Dispatches to all NUMA nodes, then joins all.
Workers write completion timestamps. Join overhead is measured as
the gap between the last worker finishing and the main thread returning
from join.

Uses IsolatedBurstPool when CPU isolation is configured, BurstPool otherwise.
"""

from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from numa import NumaInfo
from notstdcollections import HeapMoveArray
import linux.sys as linux


@fieldwise_init
struct DelayArgs(Copyable, ImplicitlyCopyable):
    var out_addr: UnsafePointer[Int, MutAnyOrigin]
    var job_id: Int
    var pauses: Int


def delay_kernel(args: DelayArgs):
    var sys = linux.linux_sys()
    for _ in range(args.pauses):
        sys.arch_cpu_relax()
    args.out_addr[] = args.job_id + 1


comptime LAYERS = 40
comptime WARMUP = 5
comptime TRIALS = 50


def run_bench[P: BurstThreadPool](mut pools: HeapMoveArray[P], num_nodes: Int):
    var node_args = HeapMoveArray[HeapMoveArray[DelayArgs]](num_nodes)
    var node_outputs = HeapMoveArray[HeapMoveArray[Int]](num_nodes)
    for i in range(num_nodes):
        var nc = pools[i].get_capacity()
        var na = HeapMoveArray[DelayArgs](nc)
        var no = HeapMoveArray[Int](nc)
        for j in range(nc):
            na.push(DelayArgs(UnsafePointer[Int, MutAnyOrigin](), j, 0))
            no.push(0)
        for j in range(nc):
            (na.ptr + j)[].out_addr = UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(no.ptr + j))
        node_args.push(na^)
        node_outputs.push(no^)

    print("\nwork       dispatch    join      join_overhead")

    for wi in range(7):
        var pauses = 0
        var label = "noop"
        if wi == 1: pauses = 1000;   label = "~10us"
        if wi == 2: pauses = 5000;   label = "~50us"
        if wi == 3: pauses = 20000;  label = "~200us"
        if wi == 4: pauses = 50000;  label = "~500us"
        if wi == 5: pauses = 100000; label = "~1ms"
        if wi == 6: pauses = 200000; label = "~2ms"

        for i in range(num_nodes):
            for j in range(pools[i].get_capacity()):
                (node_args[i].ptr + j)[].pauses = pauses

        for _ in range(WARMUP):
            for _ in range(LAYERS):
                for i in range(num_nodes):
                    pools[i].dispatch[DelayArgs, delay_kernel](node_args[i].ptr, pools[i].get_capacity())
                for i in range(num_nodes):
                    pools[i].join()

        var best_d = Int(1 << 60)
        var best_j = Int(1 << 60)
        var best_overhead = Int(1 << 60)
        for _ in range(TRIALS):
            var da = 0
            var ja = 0
            var oa = 0
            for _ in range(LAYERS):
                var t0 = Int(perf_counter_ns())
                for i in range(num_nodes):
                    pools[i].dispatch[DelayArgs, delay_kernel](node_args[i].ptr, pools[i].get_capacity())
                var t1 = Int(perf_counter_ns())
                for i in range(num_nodes):
                    pools[i].join()
                var t2 = Int(perf_counter_ns())
                var max_ts = 0
                for i in range(num_nodes):
                    var ts = pools[i].last_worker_timestamp()
                    if ts > max_ts:
                        max_ts = ts
                da += t1 - t0
                ja += t2 - t1
                oa += t2 - max_ts
            keep(node_outputs[0].ptr[])
            var d = da // LAYERS
            var j = ja // LAYERS
            var o = oa // LAYERS
            if d + j < best_d + best_j:
                best_d = d
                best_j = j
                best_overhead = o

        print("  " + label + "\t   "
            + String(best_d // 1000) + " us\t   "
            + String(best_j // 1000) + " us\t   "
            + String(best_overhead) + " ns")


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
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_bench(pools, num_nodes)
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](num_nodes)
        for i in range(num_nodes):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_bench(pools, num_nodes)
