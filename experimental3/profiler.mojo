"""Forward-pass profiling — per-phase timing capture and reporting."""

from std.time import perf_counter_ns
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence


comptime NUM_FORWARD_PHASES = 27


struct PhaseTiming(Copyable, ImplicitlyCopyable):
    var dispatch_ns: Int
    var kernel_ns: Int
    var join_ns: Int

    def __init__(out self):
        self.dispatch_ns = 0
        self.kernel_ns = 0
        self.join_ns = 0

    def __init__(out self, dispatch_ns: Int, kernel_ns: Int, join_ns: Int):
        self.dispatch_ns = dispatch_ns
        self.kernel_ns = kernel_ns
        self.join_ns = join_ns

    @always_inline
    def total(self) -> Int:
        return self.dispatch_ns + self.kernel_ns + self.join_ns

    def add(mut self, other: Self):
        self.dispatch_ns += other.dispatch_ns
        self.kernel_ns += other.kernel_ns
        self.join_ns += other.join_ns

    @staticmethod
    def opaque(total_ns: Int) -> Self:
        return Self(0, total_ns, 0)


@always_inline
def phase_timing_from_points(
    dispatch_start_ns: Int,
    dispatch_end_ns: Int,
    worker_done_ns: Int,
    join_start_ns: Int,
    join_end_ns: Int,
    active: Bool,
) -> PhaseTiming:
    var dispatch_ns = dispatch_end_ns - dispatch_start_ns
    if dispatch_ns < 0:
        dispatch_ns = 0
    if not active:
        return PhaseTiming(dispatch_ns, 0, 0)

    var kernel_end_ns = worker_done_ns
    if kernel_end_ns < dispatch_end_ns:
        kernel_end_ns = dispatch_end_ns
    var kernel_ns = kernel_end_ns - dispatch_end_ns
    if kernel_ns < 0:
        kernel_ns = 0

    var join_base_ns = join_start_ns
    if kernel_end_ns > join_base_ns:
        join_base_ns = kernel_end_ns
    var join_ns = join_end_ns - join_base_ns
    if join_ns < 0:
        join_ns = 0
    return PhaseTiming(dispatch_ns, kernel_ns, join_ns)


def finish_single_pool_fence[P: BurstThreadPool](
    dispatch_start_ns: Int,
    dispatch_end_ns: Int,
    var fence: PoolFence[P],
) -> PhaseTiming:
    var pool_ptr = fence^.take()
    if not pool_ptr:
        return phase_timing_from_points(
            dispatch_start_ns, dispatch_end_ns,
            dispatch_end_ns, dispatch_end_ns, dispatch_end_ns, False)
    pool_ptr[].join()
    var join_end_ns = Int(perf_counter_ns())
    return phase_timing_from_points(
        dispatch_start_ns, dispatch_end_ns,
        pool_ptr[].last_worker_timestamp(),
        dispatch_end_ns, join_end_ns, True)


struct ForwardSample(Copyable, ImplicitlyCopyable):
    var pos: Int
    var wall_ns: Int
    var embed: PhaseTiming
    var broadcast: PhaseTiming
    var local_attn_quantize: PhaseTiming
    var local_attn_proj: PhaseTiming
    var local_attention: PhaseTiming
    var local_o_proj: PhaseTiming
    var local_attn_reduce: PhaseTiming
    var global_attn_quantize: PhaseTiming
    var global_attn_proj: PhaseTiming
    var global_attention: PhaseTiming
    var global_o_proj: PhaseTiming
    var global_attn_reduce: PhaseTiming
    var post_attn_norm: PhaseTiming
    var router_quantize: PhaseTiming
    var router_proj: PhaseTiming
    var router_topk: PhaseTiming
    var ffn_quantize: PhaseTiming
    var expert_phase1: PhaseTiming
    var dense_phase1: PhaseTiming
    var expert_phase2: PhaseTiming
    var dense_phase2: PhaseTiming
    var pre_reduce: PhaseTiming
    var mlp_reduce: PhaseTiming
    var post_reduce: PhaseTiming
    var final_norm: PhaseTiming
    var lm_head: PhaseTiming
    var softcap: PhaseTiming

    def __init__(out self, pos: Int):
        self.pos = pos
        self.wall_ns = 0
        self.embed = PhaseTiming()
        self.broadcast = PhaseTiming()
        self.local_attn_quantize = PhaseTiming()
        self.local_attn_proj = PhaseTiming()
        self.local_attention = PhaseTiming()
        self.local_o_proj = PhaseTiming()
        self.local_attn_reduce = PhaseTiming()
        self.global_attn_quantize = PhaseTiming()
        self.global_attn_proj = PhaseTiming()
        self.global_attention = PhaseTiming()
        self.global_o_proj = PhaseTiming()
        self.global_attn_reduce = PhaseTiming()
        self.post_attn_norm = PhaseTiming()
        self.router_quantize = PhaseTiming()
        self.router_proj = PhaseTiming()
        self.router_topk = PhaseTiming()
        self.ffn_quantize = PhaseTiming()
        self.expert_phase1 = PhaseTiming()
        self.dense_phase1 = PhaseTiming()
        self.expert_phase2 = PhaseTiming()
        self.dense_phase2 = PhaseTiming()
        self.pre_reduce = PhaseTiming()
        self.mlp_reduce = PhaseTiming()
        self.post_reduce = PhaseTiming()
        self.final_norm = PhaseTiming()
        self.lm_head = PhaseTiming()
        self.softcap = PhaseTiming()

    def phase(self, idx: Int) -> PhaseTiming:
        if idx == 0:
            return self.embed
        if idx == 1:
            return self.broadcast
        if idx == 2:
            return self.local_attn_quantize
        if idx == 3:
            return self.local_attn_proj
        if idx == 4:
            return self.local_attention
        if idx == 5:
            return self.local_o_proj
        if idx == 6:
            return self.local_attn_reduce
        if idx == 7:
            return self.global_attn_quantize
        if idx == 8:
            return self.global_attn_proj
        if idx == 9:
            return self.global_attention
        if idx == 10:
            return self.global_o_proj
        if idx == 11:
            return self.global_attn_reduce
        if idx == 12:
            return self.post_attn_norm
        if idx == 13:
            return self.router_quantize
        if idx == 14:
            return self.router_proj
        if idx == 15:
            return self.router_topk
        if idx == 16:
            return self.ffn_quantize
        if idx == 17:
            return self.expert_phase1
        if idx == 18:
            return self.dense_phase1
        if idx == 19:
            return self.expert_phase2
        if idx == 20:
            return self.dense_phase2
        if idx == 21:
            return self.pre_reduce
        if idx == 22:
            return self.mlp_reduce
        if idx == 23:
            return self.post_reduce
        if idx == 24:
            return self.final_norm
        if idx == 25:
            return self.lm_head
        if idx == 26:
            return self.softcap
        return PhaseTiming()

    def phase_sum_ns(self) -> Int:
        var total = 0
        for idx in range(NUM_FORWARD_PHASES):
            total += self.phase(idx).total()
        return total

    @staticmethod
    def phase_name(idx: Int) -> String:
        if idx == 0:
            return "embed"
        if idx == 1:
            return "broadcast"
        if idx == 2:
            return "local_attn_quant"
        if idx == 3:
            return "local_attn_proj"
        if idx == 4:
            return "local_attention"
        if idx == 5:
            return "local_o_proj"
        if idx == 6:
            return "local_attn_reduce"
        if idx == 7:
            return "global_attn_quant"
        if idx == 8:
            return "global_attn_proj"
        if idx == 9:
            return "global_attention"
        if idx == 10:
            return "global_o_proj"
        if idx == 11:
            return "global_attn_reduce"
        if idx == 12:
            return "post_attn_norm"
        if idx == 13:
            return "router_quantize"
        if idx == 14:
            return "router_proj"
        if idx == 15:
            return "router_topk"
        if idx == 16:
            return "ffn_quantize"
        if idx == 17:
            return "expert_phase1"
        if idx == 18:
            return "dense_phase1"
        if idx == 19:
            return "expert_phase2"
        if idx == 20:
            return "dense_phase2"
        if idx == 21:
            return "pre_reduce"
        if idx == 22:
            return "mlp_reduce"
        if idx == 23:
            return "post_reduce"
        if idx == 24:
            return "final_norm"
        if idx == 25:
            return "lm_head"
        if idx == 26:
            return "softcap"
        return "unknown"


struct NsStats(Copyable, ImplicitlyCopyable):
    var mean_ns: Int
    var stddev_ns: Int
    var p50_ns: Int
    var p90_ns: Int
    var p99_ns: Int
    var max_ns: Int

    def __init__(out self):
        self.mean_ns = 0
        self.stddev_ns = 0
        self.p50_ns = 0
        self.p90_ns = 0
        self.p99_ns = 0
        self.max_ns = 0


struct PhaseStats(Copyable, ImplicitlyCopyable):
    var total: NsStats
    var dispatch: NsStats
    var kernel: NsStats
    var join: NsStats

    def __init__(out self):
        self.total = NsStats()
        self.dispatch = NsStats()
        self.kernel = NsStats()
        self.join = NsStats()


@fieldwise_init
struct PhaseReportRow(Copyable, ImplicitlyCopyable):
    var phase_idx: Int
    var stats: PhaseStats


@always_inline
def percentile_index(count: Int, pct: Int) -> Int:
    if count <= 1:
        return 0
    var idx = (count * pct + 99) // 100 - 1
    if idx < 0:
        idx = 0
    if idx >= count:
        idx = count - 1
    return idx


def sort_ints(mut vals: List[Int]):
    for i in range(1, len(vals)):
        var x = vals[i]
        var j = i
        while j > 0 and vals[j - 1] > x:
            vals[j] = vals[j - 1]
            j -= 1
        vals[j] = x


def runtime_sqrt(x: Float64) -> Float64:
    if x <= Float64(0):
        return Float64(0)
    var g = x
    for _ in range(30):
        g = (g + x / g) * Float64(0.5)
    return g


def ns_stats(values: List[Int]) -> NsStats:
    var out = NsStats()
    var n = len(values)
    if n == 0:
        return out^

    var sorted = List[Int](capacity=n)
    var sum = Float64(0)
    var sum_sq = Float64(0)
    var max_ns = 0
    for i in range(n):
        var v = values[i]
        sorted.append(v)
        sum += Float64(v)
        sum_sq += Float64(v) * Float64(v)
        if v > max_ns:
            max_ns = v
    sort_ints(sorted)

    var mean = sum / Float64(n)
    var variance = sum_sq / Float64(n) - mean * mean
    if variance < Float64(0):
        variance = Float64(0)

    out.mean_ns = Int(mean)
    out.stddev_ns = Int(runtime_sqrt(variance))
    out.p50_ns = sorted[percentile_index(n, 50)]
    out.p90_ns = sorted[percentile_index(n, 90)]
    out.p99_ns = sorted[percentile_index(n, 99)]
    out.max_ns = max_ns
    return out^


def phase_stats(dispatch_vals: List[Int], kernel_vals: List[Int], join_vals: List[Int]) -> PhaseStats:
    var totals = List[Int](capacity=len(dispatch_vals))
    for i in range(len(dispatch_vals)):
        totals.append(dispatch_vals[i] + kernel_vals[i] + join_vals[i])
    var out = PhaseStats()
    out.total = ns_stats(totals)
    out.dispatch = ns_stats(dispatch_vals)
    out.kernel = ns_stats(kernel_vals)
    out.join = ns_stats(join_vals)
    return out^


def repeat_spaces(count: Int) -> String:
    var out = ""
    for _ in range(count):
        out += " "
    return out


def pad_left(text: String, width: Int) -> String:
    var n = text.byte_length()
    if n >= width:
        return text
    return repeat_spaces(width - n) + text


def pad_right(text: String, width: Int) -> String:
    var n = text.byte_length()
    if n >= width:
        return text
    return text + repeat_spaces(width - n)


def format_ms3(ns: Int) -> String:
    var x = ns
    var sign = ""
    if x < 0:
        sign = "-"
        x = -x
    var whole = x // 1_000_000
    var frac = (x % 1_000_000) // 1_000
    var frac_s = String(frac)
    if frac < 10:
        frac_s = "00" + frac_s
    elif frac < 100:
        frac_s = "0" + frac_s
    return sign + String(whole) + "." + frac_s


def format_pct1(part: Int, total: Int) -> String:
    if total <= 0:
        return "0.0%"
    var tenths = (part * 1000 + total // 2) // total
    return String(tenths // 10) + "." + String(tenths % 10) + "%"


def sort_phase_rows(mut rows: List[PhaseReportRow]):
    for i in range(1, len(rows)):
        var x = rows[i]
        var j = i
        while j > 0 and rows[j - 1].stats.total.mean_ns < x.stats.total.mean_ns:
            rows[j] = rows[j - 1]
            j -= 1
        rows[j] = x


struct ForwardLogger(Movable):
    var samples: List[ForwardSample]

    def __init__(out self):
        self.samples = List[ForwardSample]()

    def clear(mut self):
        self.samples = List[ForwardSample]()

    def record(mut self, sample: ForwardSample):
        self.samples.append(sample)

    def report(self, label: String):
        var n = len(self.samples)
        if n == 0:
            print(label + ": no profiled forwards")
            return

        var wall_vals = List[Int](capacity=n)
        var phase_sum_vals = List[Int](capacity=n)
        var min_pos = self.samples[0].pos
        var max_pos = self.samples[0].pos
        for i in range(n):
            var s = self.samples[i]
            wall_vals.append(s.wall_ns)
            phase_sum_vals.append(s.phase_sum_ns())
            if s.pos < min_pos:
                min_pos = s.pos
            if s.pos > max_pos:
                max_pos = s.pos
        var wall = ns_stats(wall_vals)
        var phase_sum = ns_stats(phase_sum_vals)

        print(label + " forward profile")
        print("  samples:   " + String(n))
        print("  positions: " + String(min_pos) + ".." + String(max_pos))
        print("  wall / token")
        print("    avg    " + pad_left(format_ms3(wall.mean_ns), 9)
            + " ms    stddev " + pad_left(format_ms3(wall.stddev_ns), 9) + " ms")
        print("    p50    " + pad_left(format_ms3(wall.p50_ns), 9)
            + " ms    p90    " + pad_left(format_ms3(wall.p90_ns), 9) + " ms")
        print("    p99    " + pad_left(format_ms3(wall.p99_ns), 9)
            + " ms    max    " + pad_left(format_ms3(wall.max_ns), 9) + " ms")
        print("  phase-sum / token")
        print("    avg    " + pad_left(format_ms3(phase_sum.mean_ns), 9)
            + " ms    overlap-counted " + pad_left(format_pct1(phase_sum.mean_ns, wall.mean_ns), 8) + " of wall")
        var rows = List[PhaseReportRow](capacity=NUM_FORWARD_PHASES)
        for phase_idx in range(NUM_FORWARD_PHASES):
            var dispatch_vals = List[Int](capacity=n)
            var kernel_vals = List[Int](capacity=n)
            var join_vals = List[Int](capacity=n)
            for i in range(n):
                var t = self.samples[i].phase(phase_idx)
                dispatch_vals.append(t.dispatch_ns)
                kernel_vals.append(t.kernel_ns)
                join_vals.append(t.join_ns)
            rows.append(PhaseReportRow(phase_idx, phase_stats(dispatch_vals, kernel_vals, join_vals)))
        sort_phase_rows(rows)

        var omitted = ""
        var top_summary = ""
        var shown = 0
        for i in range(len(rows)):
            var row = rows[i]
            var ps = row.stats
            if ps.total.max_ns < 50_000 and ps.dispatch.p99_ns < 50_000 and ps.join.p99_ns < 50_000:
                if omitted.byte_length() > 0:
                    omitted += ", "
                omitted += ForwardSample.phase_name(row.phase_idx)
                continue
            if shown < 5:
                if top_summary.byte_length() > 0:
                    top_summary += "  "
                top_summary += ForwardSample.phase_name(row.phase_idx) + " "
                top_summary += format_pct1(ps.total.mean_ns, wall.mean_ns)
                shown += 1

        if top_summary.byte_length() > 0:
            print("  hot path:  " + top_summary)
        print("  note: phase timings are local elapsed times; overlapped phases can sum above 100% of wall.")

        print("  phases")
        print("    "
            + pad_right("phase", 18)
            + pad_left("vs wall", 8) + "  "
            + pad_left("avg", 9) + "  "
            + pad_left("p99", 9) + "  "
            + pad_left("max", 9)
            + " | dispatch avg/p99 | kernel avg/p99 | join avg/p99  [ms]")
        print("    " + repeat_spaces(18) + "--------  ---------  ---------  --------- | ---------------- | -------------- | ------------")

        for i in range(len(rows)):
            var row = rows[i]
            var ps = row.stats
            if ps.total.max_ns < 50_000 and ps.dispatch.p99_ns < 50_000 and ps.join.p99_ns < 50_000:
                continue
            print("    "
                + pad_right(ForwardSample.phase_name(row.phase_idx), 18)
                + pad_left(format_pct1(ps.total.mean_ns, wall.mean_ns), 8) + "  "
                + pad_left(format_ms3(ps.total.mean_ns), 9) + "  "
                + pad_left(format_ms3(ps.total.p99_ns), 9) + "  "
                + pad_left(format_ms3(ps.total.max_ns), 9)
                + " | " + pad_left(format_ms3(ps.dispatch.mean_ns), 8)
                + "/" + pad_left(format_ms3(ps.dispatch.p99_ns), 8)
                + " | " + pad_left(format_ms3(ps.kernel.mean_ns), 8)
                + "/" + pad_left(format_ms3(ps.kernel.p99_ns), 8)
                + " | " + pad_left(format_ms3(ps.join.mean_ns), 8)
                + "/" + pad_left(format_ms3(ps.join.p99_ns), 8))
        if omitted.byte_length() > 0:
            print("    omitted tiny phases: " + omitted)
