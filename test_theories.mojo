"""Debug: Verify worker CPU affinity"""

from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from numa import NumaInfo, CpuMask, get_current_cpu_and_node
import threading.linux as linux

fn report_cpu(worker: Int64):
    var cpu_node = get_current_cpu_and_node()
    print("Worker", worker, "on CPU", cpu_node[0])

fn main():
    var numa = NumaInfo()
    var cpus = numa.cpus_on_node(0)
    var node0_cpus = numa.get_node_cpus(0)
    print("CPUs on node 0:", cpus)

    var main_mask = CpuMask[128]()
    main_mask.set(0)
    _ = linux.sys_sched_setaffinity(0, 128, Int(main_mask.ptr()))
    print("Main pinned to CPU 0")

    var mask = CpuMask[128]()
    for i in range(20):
        mask.set(node0_cpus[i + 1])

    var pool = BurstPool(20, mask^)
    if not pool:
        print("Pool creation failed")
        return

    print("Pool created, asking each worker to report its CPU:")
    var packs = HeapMoveArray[ArgPack](pool.capacity)
    for _ in range(pool.capacity):
        packs.push(ArgPack())
    pool.dispatch(report_cpu, packs.ptr)
