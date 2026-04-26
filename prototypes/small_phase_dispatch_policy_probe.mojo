from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns

from threading.threading_traits import BurstThreadPool


comptime MAX_MOCK_JOBS = 8


@fieldwise_init
struct SmallPhaseTiming(Copyable, ImplicitlyCopyable):
    var used_pool_dispatch: Bool
    var dispatch_ns: Int
    var kernel_ns: Int
    var join_ns: Int

    @staticmethod
    def inline(total_ns: Int) -> Self:
        return Self(False, 0, total_ns, 0)

    @staticmethod
    def dispatched(
        dispatch_start_ns: Int,
        dispatch_end_ns: Int,
        worker_done_ns: Int,
        join_end_ns: Int,
    ) -> Self:
        var kernel_end_ns = worker_done_ns
        if kernel_end_ns < dispatch_end_ns:
            kernel_end_ns = dispatch_end_ns

        var kernel_ns = kernel_end_ns - dispatch_end_ns
        if kernel_ns < 0:
            kernel_ns = 0

        var join_base_ns = kernel_end_ns
        if dispatch_end_ns > join_base_ns:
            join_base_ns = dispatch_end_ns

        var join_ns = join_end_ns - join_base_ns
        if join_ns < 0:
            join_ns = 0

        return Self(
            True,
            dispatch_end_ns - dispatch_start_ns,
            kernel_ns,
            join_ns,
        )

    def total(self) -> Int:
        return self.dispatch_ns + self.kernel_ns + self.join_ns


def run_small_phase_jobs[
    P: BurstThreadPool,
    Args: Copyable & ImplicitlyCopyable,
    args_origin: MutOrigin,
    pool_origin: MutOrigin,
    //,
    use_pool_dispatch: Bool,
    kernel: def(Args) thin -> None,
](
    args: UnsafePointer[Args, args_origin],
    num_jobs: Int,
    ref [pool_origin] pool: P,
) -> SmallPhaseTiming:
    """Owns the inline-vs-pool decision for already-prepared small jobs.

    A real integration can put this helper beside the kernel dispatch wrappers.
    The caller does not receive a fence and the profiler does not need to know
    which execution path was used.
    """
    var t0 = Int(perf_counter_ns())
    if num_jobs <= 0:
        return SmallPhaseTiming.inline(Int(perf_counter_ns()) - t0)

    comptime if use_pool_dispatch:
        pool.dispatch[Args, kernel, args_origin](args, num_jobs)
        var dispatch_end = Int(perf_counter_ns())
        pool.join()
        var join_end = Int(perf_counter_ns())
        return SmallPhaseTiming.dispatched(
            t0, dispatch_end, pool.last_worker_timestamp(), join_end)
    else:
        for i in range(num_jobs):
            kernel(args[i])
        return SmallPhaseTiming.inline(Int(perf_counter_ns()) - t0)


@fieldwise_init
struct MockNormArgs(Copyable, ImplicitlyCopyable):
    var out_addr: Int
    var value: Int


def mock_norm_worker(args: MockNormArgs):
    var out = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=args.out_addr)
    out[] += args.value


def prepare_mock_norm_jobs[origin: MutOrigin](
    jobs: UnsafePointer[MockNormArgs, origin],
    pool_capacity: Int,
    out_addr: Int,
    first_value: Int,
) -> Int:
    var num_jobs = min(4, pool_capacity)
    for i in range(num_jobs):
        jobs[i] = MockNormArgs(out_addr, first_value + i)
    return num_jobs


trait MockSmallPhaseDispatchPool(BurstThreadPool):
    pass


comptime MockNormUsePoolDispatch[P: BurstThreadPool]: Bool = conforms_to(
    P, MockSmallPhaseDispatchPool,
)


comptime MockTinyNormUsePoolDispatch[P: BurstThreadPool]: Bool = False


def mock_norm_dispatch[
    P: BurstThreadPool,
    pool_origin: MutOrigin,
](
    ref [pool_origin] pool: P,
    out_addr: Int,
) -> SmallPhaseTiming:
    """Mock kernel wrapper: prepare jobs and own the execution decision."""
    var jobs = InlineArray[MockNormArgs, MAX_MOCK_JOBS](uninitialized=True)
    var num_jobs = prepare_mock_norm_jobs(
        UnsafePointer(to=jobs[0]), pool.get_capacity(), out_addr, 1)

    return run_small_phase_jobs[
        MockNormUsePoolDispatch[P], mock_norm_worker,
    ](
        UnsafePointer(to=jobs[0]), num_jobs, pool)


def mock_tiny_norm_dispatch[
    P: BurstThreadPool,
    pool_origin: MutOrigin,
](
    ref [pool_origin] pool: P,
    out_addr: Int,
) -> SmallPhaseTiming:
    """A second kernel can choose inline even for a dispatch-capable pool."""
    var jobs = InlineArray[MockNormArgs, MAX_MOCK_JOBS](uninitialized=True)
    var num_jobs = prepare_mock_norm_jobs(
        UnsafePointer(to=jobs[0]), pool.get_capacity(), out_addr, 10)

    return run_small_phase_jobs[
        MockTinyNormUsePoolDispatch[P], mock_norm_worker,
    ](
        UnsafePointer(to=jobs[0]), num_jobs, pool)


struct MockPool(BurstThreadPool):
    var capacity: Int
    var dispatch_calls: Int
    var join_calls: Int
    var worker_done_ns: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.dispatch_calls = 0
        self.join_calls = 0
        self.worker_done_ns = 0

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[
        Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None,
        origin: MutOrigin,
    ](mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        self.dispatch_calls += 1
        for i in range(num_jobs):
            kernel(args[i])
        self.worker_done_ns = Int(perf_counter_ns())

    def join(mut self):
        self.join_calls += 1

    def last_worker_timestamp(self) -> Int:
        return self.worker_done_ns

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


struct MockIsolatedPool(MockSmallPhaseDispatchPool):
    var capacity: Int
    var dispatch_calls: Int
    var join_calls: Int
    var worker_done_ns: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.dispatch_calls = 0
        self.join_calls = 0
        self.worker_done_ns = 0

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[
        Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None,
        origin: MutOrigin,
    ](mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        self.dispatch_calls += 1
        for i in range(num_jobs):
            kernel(args[i])
        self.worker_done_ns = Int(perf_counter_ns())

    def join(mut self):
        self.join_calls += 1

    def last_worker_timestamp(self) -> Int:
        return self.worker_done_ns

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


def main():
    var inline_pool = MockPool(4)
    var inline_sum = 0
    var inline_timing = mock_norm_dispatch(
        inline_pool, Int(UnsafePointer(to=inline_sum)))

    var dispatch_pool = MockIsolatedPool(4)
    var dispatch_sum = 0
    var dispatch_timing = mock_norm_dispatch(
        dispatch_pool, Int(UnsafePointer(to=dispatch_sum)))

    var tiny_inline_pool = MockIsolatedPool(4)
    var tiny_inline_sum = 0
    var tiny_inline_timing = mock_tiny_norm_dispatch(
        tiny_inline_pool, Int(UnsafePointer(to=tiny_inline_sum)))

    print("inline sum:", inline_sum)
    print("inline used_pool_dispatch:", inline_timing.used_pool_dispatch)
    print("inline dispatch_calls:", inline_pool.dispatch_calls)
    print("inline join_calls:", inline_pool.join_calls)
    print("inline total_ns:", inline_timing.total())

    print("dispatch sum:", dispatch_sum)
    print("dispatch used_pool_dispatch:", dispatch_timing.used_pool_dispatch)
    print("dispatch dispatch_calls:", dispatch_pool.dispatch_calls)
    print("dispatch join_calls:", dispatch_pool.join_calls)
    print("dispatch total_ns:", dispatch_timing.total())

    print("tiny inline sum:", tiny_inline_sum)
    print("tiny inline used_pool_dispatch:", tiny_inline_timing.used_pool_dispatch)
    print("tiny inline dispatch_calls:", tiny_inline_pool.dispatch_calls)
    print("tiny inline join_calls:", tiny_inline_pool.join_calls)
    print("tiny inline total_ns:", tiny_inline_timing.total())
