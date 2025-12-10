"""Benchmark BurstPool with NUMA-aware node-local pools.

Tests the pattern where each NUMA node has its own BurstPool,
with the main thread participating in work for residual handling.
"""

from threading.burst_threading import BurstPool
from numa import NumaInfo, get_current_cpu_and_node
from time import perf_counter_ns

fn empty_work():
    pass

fn bench_single_node_pool(numa: NumaInfo, node: Int, exclude_cpu: Int, iterations: Int):
    """Benchmark a single node's pool excluding the orchestrator CPU."""
    var pool = BurstPool.for_numa_node_excluding(numa, node, exclude_cpu)
    if not pool:
        print("  Node", node, "pool creation failed")
        return

    var pool_size = pool.capacity
    print("  Node", node, ":", pool_size, "workers (excluding CPU", exclude_cpu, ")")

    # Warmup
    pool.sync(empty_work)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(iterations):
        pool.sync(empty_work)
    var end = perf_counter_ns()

    var total_ns = Int(end - start)
    var ns_per_iteration = total_ns // iterations
    print("    Per-batch:", ns_per_iteration, "ns")

fn bench_full_node_pool(numa: NumaInfo, node: Int, iterations: Int):
    """Benchmark a full node's pool (no exclusion)."""
    var pool = BurstPool.for_numa_node(numa, node)
    if not pool:
        print("  Node", node, "pool creation failed")
        return

    var pool_size = pool.capacity
    print("  Node", node, ":", pool_size, "workers (full node)")

    # Warmup
    pool.sync(empty_work)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(iterations):
        pool.sync(empty_work)
    var end = perf_counter_ns()

    var total_ns = Int(end - start)
    var ns_per_iteration = total_ns // iterations
    print("    Per-batch:", ns_per_iteration, "ns")

fn main():
    var numa = NumaInfo()
    var current = get_current_cpu_and_node()
    var current_cpu = current[0]
    var current_node = current[1]

    print("=== NUMA-Aware BurstPool Benchmark ===")
    print("System:", numa.num_nodes, "NUMA nodes")
    print("Main thread on CPU", current_cpu, ", Node", current_node)
    print("(1000 iterations each)\n")

    # Print topology
    print("Topology:")
    for i in range(numa.num_nodes):
        print("  Node", i, ":", numa.cpus_on_node(i), "CPUs")
    print("")

    alias ITERATIONS = 1000

    # Test 1: Full node pools (one per node)
    print("--- Full Node Pools ---")
    for i in range(numa.num_nodes):
        bench_full_node_pool(numa, i, ITERATIONS)
    print("")

    # Test 2: Local node excluding orchestrator
    print("--- Local Node (excluding main thread CPU) ---")
    bench_single_node_pool(numa, current_node, current_cpu, ITERATIONS)
    print("")

    # Test 3: Compare local vs remote node access
    if numa.num_nodes > 1:
        print("--- Local vs Remote Node Comparison ---")
        print("Local (Node", current_node, "):")
        bench_full_node_pool(numa, current_node, ITERATIONS)

        var remote_node = (current_node + 1) % numa.num_nodes
        print("Remote (Node", remote_node, "):")
        bench_full_node_pool(numa, remote_node, ITERATIONS)
