from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from memory import MutUnsafePointer
from collections import InlineArray
from time import perf_counter_ns


fn mix64(x: Int) -> Int:
    # SplitMix64
    var z = x + (-7046029254386353131)
    z = (z ^ (z >> 30)) * (-4658895280553007687)
    z = (z ^ (z >> 27)) * (-7723592293110705685)
    return z ^ (z >> 31)


fn calc_result(iter: Int, job_idx: Int) -> Int:
    var x = mix64(iter ^ job_idx)
    # Variable work to allow scheduler preemption/deschedule.
    var spins = Int(x & 0xFF)
    for _ in range(spins):
        x = mix64(x)
    return x


fn calc_scratch_sum(iter: Int, job_idx: Int) -> Int:
    # Sum_{i=0..127} (iter + job_idx + i)
    return (iter + job_idx) * 128 + 8128


fn stress_kernel(out_ptr: MutUnsafePointer[Int, MutOrigin.external], iter: Int, job_idx: Int):
    # Heavy-ish stack usage to stress small worker stacks.
    var scratch = InlineArray[Int, 128](uninitialized=True)  # 1KB
    for i in range(128):
        scratch[i] = iter + job_idx + i

    var x = calc_result(iter, job_idx)

    var scratch_sum = 0
    for i in range(128):
        scratch_sum += scratch[i]

    out_ptr[] = x + scratch_sum


fn main():
    comptime CAPACITY = 15
    comptime ITERATIONS = 5000
    comptime STACK_BYTES = 4096  # small, page-aligned stack to stress guard/reset behavior

    print("mew")

    var pool = BurstPool[STACK_BYTES](CAPACITY)
    if not pool:
        print("BurstPool creation failed")
        return

    print("mew")

    var output = HeapMoveArray[Int](CAPACITY)
    for _ in range(CAPACITY):
        output.push(0)

    var packs = HeapMoveArray[ArgPack](CAPACITY)
    for _ in range(CAPACITY):
        packs.push(ArgPack())

    var max_dispatch_ns = 0
    var max_join_ns = 0

    print("mew")

    var bench_start_ns = Int(perf_counter_ns())
    for iter_i in range(ITERATIONS):
        # Vary job count to hit partial bursts.
        var jobs = CAPACITY
        if iter_i % 5 == 1:
            jobs = CAPACITY // 2
        elif iter_i % 5 == 2:
            jobs = 1
        elif iter_i % 5 == 3:
            jobs = (CAPACITY * 3) // 4

        for j in range(jobs):
            var pack_ptr = packs.ptr + j
            pack_ptr[].arg0 = Int(output.ptr + j)
            pack_ptr[].arg1 = iter_i
            pack_ptr[].arg2 = j

        var t0 = Int(perf_counter_ns())
        pool.dispatch(stress_kernel, packs.ptr, jobs)
        var t1 = Int(perf_counter_ns())
        pool.join()
        var t2 = Int(perf_counter_ns())

        var dispatch_ns = t1 - t0
        var join_ns = t2 - t1
        if dispatch_ns > max_dispatch_ns:
            max_dispatch_ns = dispatch_ns
        if join_ns > max_join_ns:
            max_join_ns = join_ns

        for j in range(jobs):
            var got = (output.ptr + j)[]
            var exp = calc_result(iter_i, j) + calc_scratch_sum(iter_i, j)
            if got != exp:
                print("Mismatch at iter", iter_i, "job", j, "got", got, "expected", exp)
                return

        if iter_i % 1000 == 0 and iter_i != 0:
            print("ok through iter", iter_i)

    var bench_end_ns = Int(perf_counter_ns())
    var total_ns = bench_end_ns - bench_start_ns
    var total_s = total_ns // 1_000_000_000
    var rem_ms = (total_ns % 1_000_000_000) // 1_000_000

    print("Stress test passed.")
    print("max dispatch ns:", max_dispatch_ns)
    print("max join ns:", max_join_ns)
    print("total benchmark ns:", total_ns)
    print("total benchmark:", total_s, "s", rem_ms, "ms")
