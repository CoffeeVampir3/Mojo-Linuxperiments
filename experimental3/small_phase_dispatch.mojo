from std.memory import UnsafePointer
from std.time import perf_counter_ns

from threading.threading_traits import (
    BurstThreadPool, CheapSmallPhaseDispatchPool,
)
from notstdcollections import HeapMoveArray

from experimental3.profiler import PhaseTiming, phase_timing_from_points


comptime UseSmallPhasePoolDispatch[P: BurstThreadPool]: Bool = conforms_to(
    P, CheapSmallPhaseDispatchPool,
)


def run_tp_single_job_phase[
    P: BurstThreadPool,
    Args: Copyable & ImplicitlyCopyable,
    args_origin: MutOrigin,
    //,
    tp: Int,
    kernel: def(Args) thin -> None,
](
    args: UnsafePointer[Args, args_origin],
    mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    """Run one prepared small-phase job per TP rank.

    Regular pools run inline to avoid expensive wake/fence overhead. Pools that
    opt in to CheapSmallPhaseDispatchPool dispatch all ranks first, then join,
    preserving cross-rank overlap without leaking PoolFence or policy decisions
    to the caller.
    """
    var t0 = Int(perf_counter_ns())

    comptime if UseSmallPhasePoolDispatch[P]:
        for r in range(tp):
            pools[r].dispatch[Args, kernel, args_origin](args + r, 1)
        var dispatch_end = Int(perf_counter_ns())

        var max_done_ns = 0
        for r in range(tp):
            pools[r].join()
            var done_ns = pools[r].last_worker_timestamp()
            if done_ns > max_done_ns:
                max_done_ns = done_ns

        var join_end = Int(perf_counter_ns())
        return phase_timing_from_points(
            t0, dispatch_end, max_done_ns, dispatch_end, join_end, tp > 0)
    else:
        for r in range(tp):
            kernel(args[r])
        return PhaseTiming.opaque(Int(perf_counter_ns()) - t0)
