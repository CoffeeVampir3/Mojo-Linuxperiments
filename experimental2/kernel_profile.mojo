"""Per-worker kernel profiler — nanosecond tap points for each phase.

Each worker gets its own KernelProfile. Caller collects after join
and feeds to ProfileAggregator for min/max/avg breakdown.
"""

from std.time import perf_counter_ns


@fieldwise_init
struct KernelProfile:
    """Cumulative nanoseconds per kernel phase. Zero-initialized."""
    var amx_config: Int
    var q_prep: Int
    var running_init: Int
    var k_pack: Int
    var v_pack: Int
    var score_gemm: Int
    var softmax: Int
    var vagg: Int
    var merge_wait: Int
    var merge_work: Int
    var total: Int
    var overhead: Int
    var done_timestamp: Int

    def __init__(out self):
        self.amx_config = 0
        self.q_prep = 0
        self.running_init = 0
        self.k_pack = 0
        self.v_pack = 0
        self.score_gemm = 0
        self.softmax = 0
        self.vagg = 0
        self.merge_wait = 0
        self.merge_work = 0
        self.total = 0
        self.overhead = 0
        self.done_timestamp = 0


@always_inline
def tap() -> Int:
    """Read the high-resolution timer."""
    return Int(perf_counter_ns())


struct ProfileAggregator(Movable, Copyable, ImplicitlyCopyable):
    """Collects profiles from N workers, prints per-phase breakdown."""
    var count: Int
    var sum_amx_config: Int
    var sum_q_prep: Int
    var sum_running_init: Int
    var sum_k_pack: Int
    var sum_v_pack: Int
    var sum_score_gemm: Int
    var sum_softmax: Int
    var sum_vagg: Int
    var sum_merge_wait: Int
    var sum_merge_work: Int
    var sum_total: Int
    var max_total: Int
    var max_merge_wait: Int
    var sum_overhead: Int
    var max_overhead: Int
    var max_done_timestamp: Int

    def __init__(out self):
        self.count = 0
        self.sum_amx_config = 0
        self.sum_q_prep = 0
        self.sum_running_init = 0
        self.sum_k_pack = 0
        self.sum_v_pack = 0
        self.sum_score_gemm = 0
        self.sum_softmax = 0
        self.sum_vagg = 0
        self.sum_merge_wait = 0
        self.sum_merge_work = 0
        self.sum_total = 0
        self.max_total = 0
        self.max_merge_wait = 0
        self.sum_overhead = 0
        self.max_overhead = 0
        self.max_done_timestamp = 0

    def add(mut self, p: KernelProfile):
        self.count += 1
        self.sum_amx_config += p.amx_config
        self.sum_q_prep += p.q_prep
        self.sum_running_init += p.running_init
        self.sum_k_pack += p.k_pack
        self.sum_v_pack += p.v_pack
        self.sum_score_gemm += p.score_gemm
        self.sum_softmax += p.softmax
        self.sum_vagg += p.vagg
        self.sum_merge_wait += p.merge_wait
        self.sum_merge_work += p.merge_work
        self.sum_total += p.total
        self.sum_overhead += p.overhead
        if p.total > self.max_total:
            self.max_total = p.total
        if p.merge_wait > self.max_merge_wait:
            self.max_merge_wait = p.merge_wait
        if p.overhead > self.max_overhead:
            self.max_overhead = p.overhead
        if p.done_timestamp > self.max_done_timestamp:
            self.max_done_timestamp = p.done_timestamp

    def print_summary(self):
        if self.count == 0:
            print("  (no profiles)")
            return
        var n = self.count
        var wall = self.max_total
        print("  workers: " + String(n) + "  wall(max): " + String(wall // 1000) + " us")
        _print_phase("amx_config", self.sum_amx_config, n, wall)
        _print_phase("q_prep", self.sum_q_prep, n, wall)
        _print_phase("running_init", self.sum_running_init, n, wall)
        _print_phase("k_pack", self.sum_k_pack, n, wall)
        _print_phase("v_pack", self.sum_v_pack, n, wall)
        _print_phase("score_gemm", self.sum_score_gemm, n, wall)
        _print_phase("softmax", self.sum_softmax, n, wall)
        _print_phase("vagg", self.sum_vagg, n, wall)
        _print_phase("merge_wait", self.sum_merge_wait, n, wall)
        print("    merge_wait(max): " + String(self.max_merge_wait // 1000) + " us")
        _print_phase("merge_work", self.sum_merge_work, n, wall)
        _print_phase("overhead", self.sum_overhead, n, wall)
        print("    overhead(max): " + String(self.max_overhead // 1000) + " us")


def _print_phase(name: String, total_ns: Int, count: Int, wall_ns: Int):
    var avg_us = total_ns // count // 1000
    var pct = 0
    if wall_ns > 0:
        pct = (total_ns // count) * 100 // wall_ns
    print("    " + name + ": " + String(avg_us) + " us avg (" + String(pct) + "%)")


