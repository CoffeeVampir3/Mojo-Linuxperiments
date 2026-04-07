"""BurstPool stress test — correctness + timing comparison with jthread."""

from threading.burst_threading import BurstPool
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


def main():
    comptime CAPACITY = 15
    comptime ITERATIONS = 5000

    var pool = BurstPool[](CAPACITY)
    if not pool:
        print("BurstPool creation failed")
        return

    var output = HeapMoveArray[Int](CAPACITY)
    for _ in range(CAPACITY):
        output.push(0)

    var total_dispatch_ns = 0
    var total_join_ns = 0
    var max_dispatch_ns = 0
    var max_join_ns = 0
    var total_dispatches = 0

    var bench_start_ns = Int(perf_counter_ns())
    for iter_i in range(ITERATIONS):
        var jobs = CAPACITY
        if iter_i % 5 == 1:
            jobs = CAPACITY // 2
        elif iter_i % 5 == 2:
            jobs = 1
        elif iter_i % 5 == 3:
            jobs = (CAPACITY * 3) // 4

        var job_args = InlineArray[StressArgs, CAPACITY](uninitialized=True)
        for j in range(jobs):
            job_args[j] = StressArgs(
                UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(output.ptr + j)),
                iter_i, j)

        var t0 = Int(perf_counter_ns())
        pool.dispatch[StressArgs, stress_kernel](UnsafePointer(to=job_args[0]), jobs)
        var t1 = Int(perf_counter_ns())
        pool.join()
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

    print("Stress test passed.")
    print("  iterations:", ITERATIONS, " workers:", CAPACITY)
    print("  avg dispatch:", total_dispatch_ns // total_dispatches, "ns")
    print("  avg join:    ", total_join_ns // total_dispatches, "ns")
    print("  max dispatch:", max_dispatch_ns, "ns")
    print("  max join:    ", max_join_ns, "ns")
    print("  total:       ", total_ns // 1_000_000, "ms")
