from threading.linux_threading import ThreadPool, set_thread_affinity, get_cpu_and_node_rseq, MutexGate
from threading.numa_arena import NumaArena
from numa import NumaInfo, CpuMask, get_current_cpu_and_node
from memory import UnsafePointer, MutUnsafePointer
from os import Atomic
from notstdcollections import HeapMoveArray

alias IntPtr = MutUnsafePointer[Int, MutOrigin.external]
alias AtomicPtr = MutUnsafePointer[Atomic[DType.int64], MutOrigin.external]

fn report(name: String, passed: Bool, detail: String = ""):
    var status = "PASS" if passed else "FAIL"
    if detail:
        print("[" + status + "]", name, "-", detail)
    else:
        print("[" + status + "]", name)

fn compute_square(idx: Int, dst: IntPtr):
    dst[idx] += idx * idx

fn write_value(idx: Int, val: Int, dst: IntPtr):
    dst[idx] = val

fn test_thread_constructor():
    var output = InlineArray[Int, 4](0)
    var dst = IntPtr(unsafe_from_address=Int(UnsafePointer(to=output)))

    var t0 = ThreadPool[1*4096](1)
    var t1 = ThreadPool[2*4096](1)
    var t2 = ThreadPool[3*4096](1)
    var t3 = ThreadPool[](1)

    if not (t0 and t1 and t2 and t3):
        report("thread_constructor", False, "pool creation failed")
        return

    _ = t0.launch(write_value, 0, 100, dst)
    _ = t1.launch(write_value, 1, 200, dst)
    _ = t2.launch(write_value, 2, 300, dst)
    _ = t3.launch(write_value, 3, 400, dst)

    t0.wait_all()
    t1.wait_all()
    t2.wait_all()
    t3.wait_all()

    var errors = 0
    for i in range(4):
        if output[i] != (i + 1) * 100:
            errors += 1

    report("thread_constructor", errors == 0, "4 pools, varying stack sizes")

fn test_thread_pool():
    alias N = 32
    var output = InlineArray[Int, N](0)
    var dst = IntPtr(unsafe_from_address=Int(UnsafePointer(to=output)))

    var pool = ThreadPool[2*1024](N)
    if not pool:
        report("thread_pool", False, "pool creation failed")
        return

    for i in pool:
        _ = pool.launch(compute_square, i, dst)
    pool.wait_all()

    for i in pool:
        _ = pool.launch(compute_square, i, dst)
    pool.wait_all()

    var errors = 0
    for i in range(N):
        if output[i] != i * i * 2:
            errors += 1

    report("thread_pool", errors == 0, String(N) + " threads, 2 batches")

fn pool_reuse_worker(ctr: AtomicPtr):
    _ = ctr[].fetch_add(1)

fn test_pool_reuse():
    var pool = ThreadPool[](2)
    if not pool:
        report("pool_reuse", False, "pool creation failed")
        return

    var counter = Atomic[DType.int64](0)
    var ctr = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=counter)))

    _ = pool.launch(pool_reuse_worker, ctr)
    _ = pool.launch(pool_reuse_worker, ctr)
    pool.wait_all()

    _ = pool.launch(pool_reuse_worker, ctr)
    _ = pool.launch(pool_reuse_worker, ctr)
    pool.wait_all()

    report("pool_reuse", Int(counter.load()) == 4, "2 threads, 2 batches")

fn pinning_worker(expected_node: Int, err: AtomicPtr):
    var loc = get_current_cpu_and_node()
    if loc[1] != expected_node:
        _ = err[].fetch_add(1)

fn test_numa_pinning():
    var numa = NumaInfo()

    var orch_mask = CpuMask[128]()
    orch_mask.set(0)
    _ = set_thread_affinity(orch_mask)
    var (orch_cpu, orch_node) = get_current_cpu_and_node()

    var err_ct = Atomic[DType.int64](0)
    var err = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=err_ct)))

    var pools = HeapMoveArray[ThreadPool[]](numa.num_nodes)
    var total_threads = 0
    for node in range(numa.num_nodes):
        pools.push(ThreadPool[].for_numa_node_excluding(numa, node, orch_cpu))
        total_threads += pools[node][].capacity

    for node in range(numa.num_nodes):
        for i in pools[node][]:
            _ = pools[node][].launch(pinning_worker, node, err)

    pinning_worker(orch_node, err)

    for pool in pools:
        pool[].wait_all()

    var errors = Int(err_ct.load())
    report("numa_pinning", errors == 0, String(total_threads + 1) + " threads across " + String(numa.num_nodes) + " nodes")

fn rseq_worker(err: AtomicPtr):
    var rseq_loc = get_cpu_and_node_rseq()
    var syscall_loc = get_current_cpu_and_node()
    if rseq_loc[0] != syscall_loc[0] or rseq_loc[1] != syscall_loc[1]:
        _ = err[].fetch_add(1)

fn test_rseq_api():
    var numa = NumaInfo()
    var err_ct = Atomic[DType.int64](0)
    var err = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=err_ct)))

    var pools = HeapMoveArray[ThreadPool[]](numa.num_nodes)
    var total = 0
    for node in range(numa.num_nodes):
        pools.push(ThreadPool[].for_numa_node(numa, node))
        total += pools[node][].capacity

    for node in range(numa.num_nodes):
        for i in pools[node][]:
            _ = pools[node][].launch(rseq_worker, err)

    for pool in pools:
        pool[].wait_all()

    report("rseq_api", Int(err_ct.load()) == 0, String(total) + " threads validated")

fn arena_writer(slot_idx: Int, buffer: IntPtr):
    var loc = get_current_cpu_and_node()
    buffer[slot_idx] = slot_idx * 100 + loc[1]

fn test_numa_arena_buffer():
    var numa = NumaInfo()
    alias BUFFER_SIZE = 32

    var arena = NumaArena[](node=0, size=4096)
    if not arena:
        report("numa_arena_buffer", False, "arena allocation failed")
        return

    var buffer = arena.alloc[Int](BUFFER_SIZE)
    if not buffer:
        report("numa_arena_buffer", False, "buffer allocation failed")
        return

    for i in range(BUFFER_SIZE):
        buffer[i] = 0

    var pools = HeapMoveArray[ThreadPool[]](numa.num_nodes)
    for node in range(numa.num_nodes):
        pools.push(ThreadPool[].for_numa_node(numa, node))

    var threads_per_node = BUFFER_SIZE // numa.num_nodes
    var remainder = BUFFER_SIZE % numa.num_nodes
    var thread_idx = 0

    for node in range(numa.num_nodes):
        var count = threads_per_node + (1 if node < remainder else 0)
        for _ in range(count):
            if thread_idx < BUFFER_SIZE:
                _ = pools[node][].launch(arena_writer, thread_idx, buffer)
                thread_idx += 1

    for pool in pools:
        pool[].wait_all()

    var errors = 0
    for i in range(BUFFER_SIZE):
        var value = buffer[i]
        var expected_base = i * 100
        if value < expected_base or value >= expected_base + numa.num_nodes:
            errors += 1

    var placement_ok = arena.verify_placement()
    report("numa_arena_buffer", errors == 0 and placement_ok,
           String(BUFFER_SIZE) + " slots, placement=" + ("OK" if placement_ok else "FAIL"))

fn arena_writer_local(slot_idx: Int, expected_node: Int, buffer: IntPtr, err: AtomicPtr):
    import threading.linux as linux
    var loc = get_current_cpu_and_node()
    var thread_node = loc[1]
    var mem_node = linux.sys_move_pages_query(Int(buffer))
    buffer[slot_idx] = slot_idx * 1000 + expected_node * 10 + thread_node
    if thread_node != expected_node or mem_node != expected_node:
        _ = err[].fetch_add(1)

fn test_numa_arena_scatter():
    var numa = NumaInfo()
    var num_nodes = numa.num_nodes
    var cpus_per_node = numa.cpus_per_node()
    var total_cpus = num_nodes * cpus_per_node

    var orch_mask = CpuMask[128]()
    orch_mask.set(0)
    _ = set_thread_affinity(orch_mask)
    var (orch_cpu, orch_node) = get_current_cpu_and_node()

    var arenas = HeapMoveArray[NumaArena[]](num_nodes)
    for node in range(num_nodes):
        arenas.push(NumaArena[](node=node, size=4096))
        if not arenas[node][]:
            report("numa_arena_scatter", False, "arena allocation failed on node " + String(node))
            return

    var buffers = InlineArray[UnsafePointer[Int, MutAnyOrigin], 8](UnsafePointer[Int, MutAnyOrigin]())
    for node in range(num_nodes):
        buffers[node] = arenas[node][].alloc[Int](cpus_per_node)
        if not buffers[node]:
            report("numa_arena_scatter", False, "buffer allocation failed on node " + String(node))
            return

    var pools = HeapMoveArray[ThreadPool[]](num_nodes)
    for node in range(num_nodes):
        pools.push(ThreadPool[].for_numa_node_excluding(numa, node, orch_cpu))
        if not pools[node][]:
            report("numa_arena_scatter", False, "pool creation failed for node " + String(node))
            return

    var err_ct = Atomic[DType.int64](0)
    var err = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=err_ct)))
    var write_errors = 0

    for iteration in range(2):
        for node in range(num_nodes):
            for i in range(cpus_per_node):
                buffers[node][i] = -1

        for node in range(num_nodes):
            var buffer = IntPtr(unsafe_from_address=Int(buffers[node]))
            var start_slot = 1 if node == orch_node else 0
            for i in pools[node][]:
                var slot = start_slot + i
                _ = pools[node][].launch(arena_writer_local, slot, node, buffer, err)

        var orch_buffer = IntPtr(unsafe_from_address=Int(buffers[orch_node]))
        arena_writer_local(0, orch_node, orch_buffer, err)

        for pool in pools:
            pool[].wait_all()

        for node in range(num_nodes):
            for slot in range(cpus_per_node):
                var value = buffers[node][slot]
                if value == -1:
                    write_errors += 1
                    continue
                var decoded_slot = value // 1000
                var decoded_expected = (value % 1000) // 10
                var decoded_actual = value % 10
                if decoded_slot != slot or decoded_expected != node or decoded_actual != node:
                    write_errors += 1

    var placement_errors = 0
    for node in range(num_nodes):
        if not arenas[node][].verify_placement():
            placement_errors += 1

    var pinning_errors = Int(err_ct.load())
    var passed = write_errors == 0 and placement_errors == 0 and pinning_errors == 0
    report("numa_arena_scatter", passed,
           String(total_cpus) + " CPUs, " + String(num_nodes) + " arenas, 2 iterations reuse, " +
           "write err=" + String(write_errors) + " placement err=" + String(placement_errors) +
           " pinning err=" + String(pinning_errors))

fn wait_any_worker(slot_idx: Int, sleep_ms: Int, ctr: AtomicPtr):
    from time import sleep
    sleep(sleep_ms / 1000.0)
    _ = ctr[].fetch_add(1)

fn run_wait_any_test(mut pool: ThreadPool, name: String) -> Bool:
    alias N = 4
    var completion_counter = Atomic[DType.int64](0)
    var ctr = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=completion_counter)))

    # Launch threads with inverse sleep times - slot 3 sleeps least, slot 0 sleeps most
    for i in range(N):
        var sleep_ms = (N - i) * 5
        var ok = pool.launch(wait_any_worker, i, sleep_ms, ctr)
        if not ok:
            report(name, False, "failed to launch slot " + String(i))
            return False

    var completion_order = List[Int](capacity=N)
    for _ in range(N):
        var idx = pool.wait_any()
        if idx >= 0:
            completion_order.append(idx)
        else:
            break

    var all_completed = len(completion_order) == N
    var counter_correct = Int(completion_counter.load()) == N
    var expected: List[Int] = [3, 2, 1, 0]
    var correct_order = completion_order == expected
    var passed = all_completed and counter_correct and correct_order
    var order_str = String("")
    for i in range(len(completion_order)):
        if i > 0:
            order_str += ","
        order_str += String(completion_order[i])

    report(name, passed, "order=[" + order_str + "] expected=[3,2,1,0]")
    return passed

fn test_wait_any():
    var pool = ThreadPool(4)
    if not pool:
        report("wait_any", False, "pool creation failed")
        return
    _ = run_wait_any_test(pool, "wait_any")

fn test_wait_any_numa():
    var numa = NumaInfo()
    var pool = ThreadPool[].for_numa_node(numa, 0)
    if not pool:
        report("wait_any_numa", False, "pool creation failed")
        return
    _ = run_wait_any_test(pool, "wait_any_numa")

fn quick_worker(slot_idx: Int, ctr: AtomicPtr):
    _ = ctr[].fetch_add(1)

fn test_wait_any_reuse():
    alias POOL_SIZE = 4
    alias ITERATIONS = 10
    var pool = ThreadPool(POOL_SIZE)
    if not pool:
        report("wait_any_reuse", False, "pool creation failed")
        return

    var counter = Atomic[DType.int64](0)
    var ctr = AtomicPtr(unsafe_from_address=Int(UnsafePointer(to=counter)))
    var total_launched = 0

    # Launch all slots, wait for one, then try to relaunch on freed slots
    # Repeat many times - if slots leak, we'll run out of capacity
    for iteration in range(ITERATIONS):
        var launched = 0
        for i in range(POOL_SIZE):
            if pool.launch(quick_worker, i, ctr):
                launched += 1
        total_launched += launched

        var idx = pool.wait_any()
        if idx < 0:
            report("wait_any_reuse", False, "wait_any returned -1 on iteration " + String(iteration))
            return

        # Wait for the rest
        pool.wait_all()

    var final_count = Int(counter.load())
    var passed = final_count == total_launched
    report("wait_any_reuse", passed,
           String(ITERATIONS) + " iterations, launched=" + String(total_launched) +
           " completed=" + String(final_count))

fn ring_reduce_worker(
    node_id: Int,
    num_nodes: Int,
    my_value: IntPtr,
    next_value: IntPtr,
    wait_gate: MutexGate,
    release_gate: MutexGate,
):
    my_value[] = node_id + 1

    if node_id < num_nodes - 1:
        wait_gate.wait()
        my_value[] += next_value[]

    release_gate.release()

fn test_ring_reduce():
    var numa = NumaInfo()
    var num_nodes = numa.num_nodes

    if num_nodes < 2:
        report("ring_reduce", True, "skipped (need >= 2 NUMA nodes)")
        return

    # Allocate gates and values - one per node
    # Gate[i] is released by node i to signal node i+1
    var arena = NumaArena[](node=0, size=4096)
    if not arena:
        report("ring_reduce", False, "arena allocation failed")
        return

    var gates = arena.alloc[Int32](num_nodes)
    var values = arena.alloc[Int](num_nodes)
    if not gates or not values:
        report("ring_reduce", False, "buffer allocation failed")
        return

    var pools = HeapMoveArray[ThreadPool[]](num_nodes)
    for node in range(num_nodes):
        var mask = numa.get_node_mask[128](node)
        pools.push(ThreadPool[](1, mask^, node))
        if not pools[node][]:
            report("ring_reduce", False, "pool creation failed for node " + String(node))
            return

    # Launch workers (reversed chain: N-1 -> N-2 -> ... -> 0)
    # Node i:
    #   - writes to values[i]
    #   - waits on gates[i] (for node i+1 to finish)
    #   - reads from values[i+1]
    #   - releases gates[i-1] (signals node i-1)
    for node in range(num_nodes):
        var my_value = IntPtr(unsafe_from_address=Int(values + node))
        var next_value = IntPtr(unsafe_from_address=Int(values + node + 1)) if node < num_nodes - 1 else IntPtr()
        var wait_gate = MutexGate.from_addr(Int(gates + node))
        var release_gate = MutexGate.from_addr(Int(gates + node - 1) if node > 0 else Int(gates))
        _ = pools[node][].launch(
            ring_reduce_worker,
            node,
            num_nodes,
            my_value,
            next_value,
            wait_gate,
            release_gate,
        )

    for pool in pools:
        pool[].wait_all()

    # After ring reduce: node 0 should have sum of all (1 + 2 + ... + N) = N*(N+1)/2
    var expected_final = num_nodes * (num_nodes + 1) // 2
    var final_value = values[0]
    var passed = final_value == expected_final

    report("ring_reduce", passed,
           String(num_nodes) + " nodes, final=" + String(final_value) +
           " expected=" + String(expected_final))

fn all_reduce_read(partner_value: IntPtr, scratch: IntPtr):
    scratch[] = partner_value[]

fn all_reduce_write(my_value: IntPtr, scratch: IntPtr):
    my_value[] += scratch[]

fn test_all_reduce():
    var numa = NumaInfo()
    var num_nodes = numa.num_nodes

    # Numa topology must be power of 2 aligned. AFAIK this is basically always true in modern server hardware.
    if num_nodes < 2 or (num_nodes & (num_nodes - 1)) != 0:
        report("all_reduce", True, "skipped (need power-of-2 NUMA nodes, have " + String(num_nodes) + ")")
        return

    var num_rounds = 0
    var tmp = num_nodes
    while tmp > 1:
        tmp >>= 1
        num_rounds += 1

    var arena = NumaArena[](node=0, size=4096)
    if not arena:
        report("all_reduce", False, "arena allocation failed")
        return

    var values = arena.alloc[Int](num_nodes)
    var scratch = arena.alloc[Int](num_nodes)
    if not values or not scratch:
        report("all_reduce", False, "buffer allocation failed")
        return

    for i in range(num_nodes):
        values[i] = i + 1

    var pools = HeapMoveArray[ThreadPool[]](num_nodes)
    for node in range(num_nodes):
        var mask = numa.get_node_mask[128](node)
        pools.push(ThreadPool[](1, mask^, node))
        if not pools[node][]:
            report("all_reduce", False, "pool creation failed for node " + String(node))
            return

    # Butterfly log2(N) rounds
    # Round r: each node i exchanges with node i XOR (1 << r)
    for round in range(num_rounds):
        var dist = 1 << round

        # Phase 1: read
        for node in range(num_nodes):
            var partner = node ^ dist
            var partner_value = IntPtr(unsafe_from_address=Int(values + partner))
            var my_scratch = IntPtr(unsafe_from_address=Int(scratch + node))
            _ = pools[node][].launch(all_reduce_read, partner_value, my_scratch)

        for pool in pools:
            pool[].wait_all()

        # Phase 2: add
        for node in range(num_nodes):
            var my_value = IntPtr(unsafe_from_address=Int(values + node))
            var my_scratch = IntPtr(unsafe_from_address=Int(scratch + node))
            _ = pools[node][].launch(all_reduce_write, my_value, my_scratch)

        for pool in pools:
            pool[].wait_all()

    # After all rounds, every node should have the sum 1+2+...+N
    var expected = num_nodes * (num_nodes + 1) // 2
    var errors = 0
    for i in range(num_nodes):
        if values[i] != expected:
            errors += 1

    report("all_reduce", errors == 0,
           String(num_nodes) + " nodes, " + String(num_rounds) + " rounds, all values=" + String(expected))

fn main():
    var numa = NumaInfo()
    print("NUMA:", numa.num_nodes, "nodes,", numa.cpus_per_node(), "CPUs/node")
    print()
    test_thread_constructor()
    test_thread_pool()
    test_pool_reuse()
    test_wait_any()
    test_wait_any_numa()
    test_wait_any_reuse()
    test_numa_pinning()
    test_rseq_api()
    test_numa_arena_buffer()
    test_numa_arena_scatter()
    test_ring_reduce()
    test_all_reduce()
