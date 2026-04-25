from std.time import perf_counter_ns
from std.collections import InlineArray
from std.memory import UnsafePointer
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray

from kernels.kernel_ops import PoolFence


comptime MAX_PHASES = 64


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


def finish_single_pool_fence[
    P: BurstThreadPool, origin: MutOrigin, //,
](
    dispatch_start_ns: Int,
    dispatch_end_ns: Int,
    var fence: PoolFence[P, origin],
) -> PhaseTiming:
    var worker_done_ns = fence^.finish()
    var join_end_ns = Int(perf_counter_ns())
    return phase_timing_from_points(
        dispatch_start_ns, dispatch_end_ns,
        worker_done_ns,
        dispatch_end_ns, join_end_ns, True)


def timed_tp_dispatch_recursive[
    Pool: BurstThreadPool,
    Topo: Copyable & ImplicitlyCopyable, //,
    rank: Int, tp: Int,
    body: def[r: Int, origin: MutOrigin](Topo, ref [origin] Pool) capturing -> PoolFence[Pool, origin],
](
    topos: InlineArray[Topo, tp],
    mut pools: HeapMoveArray[Pool],
    mut max_done_ns: Int,
    mut dispatch_end_ns: Int,
):
    comptime if rank < tp:
        var fence = body[rank, origin_of(pools)](topos[rank], pools[rank])
        timed_tp_dispatch_recursive[rank + 1, tp, body](
            topos, pools, max_done_ns, dispatch_end_ns)
        var done_ns = fence^.finish()
        if done_ns > max_done_ns:
            max_done_ns = done_ns
    else:
        dispatch_end_ns = Int(perf_counter_ns())


def timed_tp_parallel[
    Pool: BurstThreadPool,
    Topo: Copyable & ImplicitlyCopyable, //,
    tp: Int,
    body: def[r: Int, origin: MutOrigin](Topo, ref [origin] Pool) capturing -> PoolFence[Pool, origin],
](
    topos: InlineArray[Topo, tp],
    mut pools: HeapMoveArray[Pool],
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    var max_done_ns: Int = 0
    var dispatch_end_ns = t0
    timed_tp_dispatch_recursive[0, tp, body](
        topos, pools, max_done_ns, dispatch_end_ns)
    var t_end = Int(perf_counter_ns())
    return phase_timing_from_points(
        t0, dispatch_end_ns, max_done_ns,
        dispatch_end_ns, t_end, max_done_ns > 0)


struct ForwardSample(Copyable, ImplicitlyCopyable):
    var pos: Int
    var token_count: Int
    var produces_next_token: Bool
    var wall_ns: Int
    var phases: InlineArray[PhaseTiming, MAX_PHASES]

    def __init__(
        out self,
        pos: Int,
        token_count: Int = 1,
        produces_next_token: Bool = True,
    ):
        self.pos = pos
        self.token_count = token_count if token_count > 0 else 1
        self.produces_next_token = produces_next_token
        self.wall_ns = 0
        self.phases = InlineArray[PhaseTiming, MAX_PHASES](fill=PhaseTiming())

    def add(mut self, phase: Int, timing: PhaseTiming):
        debug_assert(phase >= 0 and phase < MAX_PHASES, "invalid profile phase")
        self.phases[phase].add(timing)

    def phase_sum_ns(self, count: Int) -> Int:
        var total = 0
        for i in range(count):
            total += self.phases[i].total()
        return total


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
    var active_total: NsStats
    var active_dispatch: NsStats
    var active_kernel: NsStats
    var active_join: NsStats
    var active_count: Int
    var sum_ns: Int

    def __init__(out self):
        self.total = NsStats()
        self.dispatch = NsStats()
        self.kernel = NsStats()
        self.join = NsStats()
        self.active_total = NsStats()
        self.active_dispatch = NsStats()
        self.active_kernel = NsStats()
        self.active_join = NsStats()
        self.active_count = 0
        self.sum_ns = 0


@fieldwise_init
struct PhaseReportRow(Copyable, ImplicitlyCopyable):
    var phase_idx: Int
    var stats: PhaseStats


struct MoeHotExpertRow(Copyable, ImplicitlyCopyable):
    var layer: Int
    var expert: Int
    var rank: Int
    var count: Int
    var events: Int

    def __init__(out self):
        self.layer = -1
        self.expert = -1
        self.rank = -1
        self.count = 0
        self.events = 1

    def __init__(out self, layer: Int, expert: Int, rank: Int, count: Int, events: Int):
        self.layer = layer
        self.expert = expert
        self.rank = rank
        self.count = count
        self.events = events if events > 0 else 1


struct MoeLayerRankRow(Copyable, ImplicitlyCopyable):
    var layer: Int
    var rank: Int
    var count: Int
    var events: Int

    def __init__(out self):
        self.layer = -1
        self.rank = -1
        self.count = 0
        self.events = 1

    def __init__(out self, layer: Int, rank: Int, count: Int, events: Int):
        self.layer = layer
        self.rank = rank
        self.count = count
        self.events = events if events > 0 else 1


struct MoeLayerHotnessRow(Copyable, ImplicitlyCopyable):
    var layer: Int
    var expert: Int
    var top1_count: Int
    var top8_count: Int
    var events: Int
    var avg_max_load_milli: Int
    var avg_same_rank_pairs_milli: Int

    def __init__(out self):
        self.layer = -1
        self.expert = -1
        self.top1_count = 0
        self.top8_count = 0
        self.events = 1
        self.avg_max_load_milli = 0
        self.avg_same_rank_pairs_milli = 0

    def __init__(
        out self,
        layer: Int,
        expert: Int,
        top1_count: Int,
        top8_count: Int,
        events: Int,
        avg_max_load_milli: Int,
        avg_same_rank_pairs_milli: Int,
    ):
        self.layer = layer
        self.expert = expert
        self.top1_count = top1_count
        self.top8_count = top8_count
        self.events = events if events > 0 else 1
        self.avg_max_load_milli = avg_max_load_milli
        self.avg_same_rank_pairs_milli = avg_same_rank_pairs_milli


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


def histogram_select(counts: List[Int], base: Int, target_idx: Int) -> Int:
    var seen = 0
    for offset in range(len(counts)):
        seen += counts[offset]
        if seen > target_idx:
            return base + offset
    return base + len(counts) - 1


def ns_stats(values: List[Int]) -> NsStats:
    var out = NsStats()
    var n = len(values)
    if n == 0:
        return out^

    var sum = Float64(0)
    var sum_sq = Float64(0)
    var min_ns = values[0]
    var max_ns = values[0]
    for i in range(n):
        var v = values[i]
        sum += Float64(v)
        sum_sq += Float64(v) * Float64(v)
        if v < min_ns:
            min_ns = v
        if v > max_ns:
            max_ns = v
    var p50_idx = percentile_index(n, 50)
    var p90_idx = percentile_index(n, 90)
    var p99_idx = percentile_index(n, 99)

    var mean = sum / Float64(n)
    var variance = sum_sq / Float64(n) - mean * mean
    if variance < Float64(0):
        variance = Float64(0)

    out.mean_ns = Int(mean)
    out.stddev_ns = Int(runtime_sqrt(variance))
    out.max_ns = max_ns

    var value_range = max_ns - min_ns
    if n > 4096 and value_range >= 0 and value_range <= 262_144:
        var counts = List[Int](length=value_range + 1, fill=0)
        for i in range(n):
            var idx = values[i] - min_ns
            counts[idx] = counts[idx] + 1
        out.p50_ns = histogram_select(counts, min_ns, p50_idx)
        out.p90_ns = histogram_select(counts, min_ns, p90_idx)
        out.p99_ns = histogram_select(counts, min_ns, p99_idx)
        return out^

    var sorted = List[Int](capacity=n)
    for i in range(n):
        sorted.append(values[i])
    sort_ints(sorted)

    out.p50_ns = sorted[p50_idx]
    out.p90_ns = sorted[p90_idx]
    out.p99_ns = sorted[p99_idx]
    return out^


def phase_stats(dispatch_vals: List[Int], kernel_vals: List[Int], join_vals: List[Int]) -> PhaseStats:
    var totals = List[Int](capacity=len(dispatch_vals))
    var active_totals = List[Int]()
    var active_dispatch = List[Int]()
    var active_kernel = List[Int]()
    var active_join = List[Int]()
    var sum_ns = 0
    for i in range(len(dispatch_vals)):
        var total = dispatch_vals[i] + kernel_vals[i] + join_vals[i]
        totals.append(total)
        sum_ns += total
        if total > 0:
            active_totals.append(total)
            active_dispatch.append(dispatch_vals[i])
            active_kernel.append(kernel_vals[i])
            active_join.append(join_vals[i])
    var out = PhaseStats()
    out.total = ns_stats(totals)
    out.dispatch = ns_stats(dispatch_vals)
    out.kernel = ns_stats(kernel_vals)
    out.join = ns_stats(join_vals)
    out.active_total = ns_stats(active_totals)
    out.active_dispatch = ns_stats(active_dispatch)
    out.active_kernel = ns_stats(active_kernel)
    out.active_join = ns_stats(active_join)
    out.active_count = len(active_totals)
    out.sum_ns = sum_ns
    return out^


@always_inline
def safe_div_int(numer: Int, denom: Int) -> Int:
    if denom <= 0:
        return 0
    return numer // denom


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


def format_milli3(value: Int) -> String:
    var x = value
    var sign = ""
    if x < 0:
        sign = "-"
        x = -x
    var whole = x // 1000
    var frac = x % 1000
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


def moe_ratio_better(count: Int, events: Int, other_count: Int, other_events: Int) -> Bool:
    if other_count <= 0:
        return count > 0
    var e = events if events > 0 else 1
    var oe = other_events if other_events > 0 else 1
    return count * oe > other_count * e


def random_max_rank_load_mean_milli(num_experts: Int, tp: Int, top_k: Int) -> Int:
    # Exact equal-shard hypergeometric baseline for MiniMax/Gemma top-8 over
    # four ranks. Other configurations still get ideal and same-rank baselines.
    if num_experts == 256 and tp == 4 and top_k == 8:
        return 3512
    return 0


def random_max_rank_load_p90(num_experts: Int, tp: Int, top_k: Int) -> Int:
    if num_experts == 256 and tp == 4 and top_k == 8:
        return 5
    return 0


def random_max_rank_load_p99(num_experts: Int, tp: Int, top_k: Int) -> Int:
    if num_experts == 256 and tp == 4 and top_k == 8:
        return 6
    return 0


def random_same_rank_pairs_milli(num_experts: Int, tp: Int, top_k: Int) -> Int:
    if num_experts <= 1 or tp <= 0 or top_k <= 1:
        return 0
    if num_experts % tp != 0:
        return 0
    var experts_per_rank = num_experts // tp
    var pairs = (top_k * (top_k - 1)) // 2
    return (pairs * (experts_per_rank - 1) * 1000 + (num_experts - 1) // 2) // (num_experts - 1)


def ideal_same_rank_pairs_milli(tp: Int, top_k: Int) -> Int:
    if tp <= 0 or top_k <= 1:
        return 0
    var base = top_k // tp
    var rem = top_k - base * tp
    var pairs = 0
    for r in range(tp):
        var load = base
        if r < rem:
            load += 1
        pairs += (load * (load - 1)) // 2
    return pairs * 1000


def sort_phase_rows(mut rows: List[PhaseReportRow]):
    for i in range(1, len(rows)):
        var x = rows[i]
        var j = i
        while j > 0 and rows[j - 1].stats.sum_ns < x.stats.sum_ns:
            rows[j] = rows[j - 1]
            j -= 1
        rows[j] = x


struct ForwardLogger(Movable):
    var samples: List[ForwardSample]
    var names: List[String]
    var moe_layers: Int
    var moe_experts: Int
    var moe_tp: Int
    var moe_top_k: Int
    var moe_layer_events: List[Int]
    var moe_rank_counts: List[Int]
    var moe_expert_counts: List[Int]
    var moe_max_rank_load_milli: List[Int]
    var moe_active_rank_count_milli: List[Int]
    var moe_same_rank_pairs_milli: List[Int]
    var moe_layer_max_rank_load: List[Int]
    var moe_layer_same_rank_pairs: List[Int]

    def __init__(out self):
        self.samples = List[ForwardSample]()
        self.names = List[String]()
        self.moe_layers = 0
        self.moe_experts = 0
        self.moe_tp = 0
        self.moe_top_k = 0
        self.moe_layer_events = List[Int]()
        self.moe_rank_counts = List[Int]()
        self.moe_expert_counts = List[Int]()
        self.moe_max_rank_load_milli = List[Int]()
        self.moe_active_rank_count_milli = List[Int]()
        self.moe_same_rank_pairs_milli = List[Int]()
        self.moe_layer_max_rank_load = List[Int]()
        self.moe_layer_same_rank_pairs = List[Int]()

    def clear(mut self):
        self.samples = List[ForwardSample]()
        self.clear_moe()

    def clear_moe(mut self):
        self.moe_layer_events = List[Int](capacity=self.moe_layers)
        self.moe_rank_counts = List[Int](capacity=self.moe_layers * self.moe_tp)
        self.moe_expert_counts = List[Int](capacity=self.moe_layers * self.moe_experts)
        self.moe_max_rank_load_milli = List[Int]()
        self.moe_active_rank_count_milli = List[Int]()
        self.moe_same_rank_pairs_milli = List[Int]()
        self.moe_layer_max_rank_load = List[Int](capacity=self.moe_layers)
        self.moe_layer_same_rank_pairs = List[Int](capacity=self.moe_layers)

        for _ in range(self.moe_layers):
            self.moe_layer_events.append(0)
            self.moe_layer_max_rank_load.append(0)
            self.moe_layer_same_rank_pairs.append(0)
        for _ in range(self.moe_layers * self.moe_tp):
            self.moe_rank_counts.append(0)
        for _ in range(self.moe_layers * self.moe_experts):
            self.moe_expert_counts.append(0)

    def configure_moe(mut self, layers: Int, experts: Int, tp: Int, top_k: Int):
        if layers <= 0 or experts <= 0 or tp <= 0 or top_k <= 0:
            return
        if (
            self.moe_layers == layers and self.moe_experts == experts
            and self.moe_tp == tp and self.moe_top_k == top_k
            and len(self.moe_layer_events) == layers
        ):
            return
        self.moe_layers = layers
        self.moe_experts = experts
        self.moe_tp = tp
        self.moe_top_k = top_k
        self.clear_moe()

    def phase(mut self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        var idx = len(self.names)
        debug_assert(idx < MAX_PHASES, "too many profile phases")
        self.names.append(name)
        return idx

    def record(mut self, sample: ForwardSample):
        self.samples.append(sample)

    def record_moe_route[top_k: Int](
        mut self,
        layer_idx: Int,
        num_layers: Int,
        num_experts: Int,
        tp: Int,
        indices: InlineArray[Int, top_k],
        weights: InlineArray[Float32, top_k],
    ):
        if layer_idx < 0 or layer_idx >= num_layers:
            return
        if num_experts <= 0 or tp <= 0 or top_k <= 0:
            return
        if num_experts % tp != 0:
            return

        self.configure_moe(num_layers, num_experts, tp, top_k)
        if len(self.moe_layer_events) != num_layers:
            return

        var experts_per_rank = num_experts // tp
        var layer_rank_base = layer_idx * tp
        var layer_expert_base = layer_idx * num_experts

        var max_count = 0
        var min_count = top_k
        var active_count = 0
        var same_rank_pairs = 0
        for r in range(tp):
            var rank_start = r * experts_per_rank
            var rank_end = rank_start + experts_per_rank
            var count = 0
            for s in range(top_k):
                var eid = indices[s]
                if eid >= rank_start and eid < rank_end:
                    count += 1
            self.moe_rank_counts[layer_rank_base + r] = self.moe_rank_counts[layer_rank_base + r] + count
            if count > max_count:
                max_count = count
            if count < min_count:
                min_count = count
            if count > 0:
                active_count += 1
            same_rank_pairs += (count * (count - 1)) // 2

        for s in range(top_k):
            var eid = indices[s]
            if eid >= 0 and eid < num_experts:
                var idx = layer_expert_base + eid
                self.moe_expert_counts[idx] = self.moe_expert_counts[idx] + 1
            _ = weights[s]

        self.moe_layer_events[layer_idx] = self.moe_layer_events[layer_idx] + 1
        self.moe_layer_max_rank_load[layer_idx] = self.moe_layer_max_rank_load[layer_idx] + max_count
        self.moe_layer_same_rank_pairs[layer_idx] = self.moe_layer_same_rank_pairs[layer_idx] + same_rank_pairs
        self.moe_max_rank_load_milli.append(max_count * 1000)
        self.moe_active_rank_count_milli.append(active_count * 1000)
        self.moe_same_rank_pairs_milli.append(same_rank_pairs * 1000)

    def report_moe(self):
        var events = len(self.moe_max_rank_load_milli)
        if events == 0 or self.moe_layers <= 0 or self.moe_experts <= 0 or self.moe_tp <= 0:
            return

        var total_slots = events * self.moe_top_k
        var max_load = ns_stats(self.moe_max_rank_load_milli)
        var active_ranks = ns_stats(self.moe_active_rank_count_milli)
        var same_rank_pairs = ns_stats(self.moe_same_rank_pairs_milli)
        var ideal_rank_load = (self.moe_top_k * 1000 + self.moe_tp // 2) // self.moe_tp
        var ideal_pairs = ideal_same_rank_pairs_milli(self.moe_tp, self.moe_top_k)
        var random_pairs = random_same_rank_pairs_milli(self.moe_experts, self.moe_tp, self.moe_top_k)
        var random_max_mean = random_max_rank_load_mean_milli(self.moe_experts, self.moe_tp, self.moe_top_k)
        var random_max_p90 = random_max_rank_load_p90(self.moe_experts, self.moe_tp, self.moe_top_k)
        var random_max_p99 = random_max_rank_load_p99(self.moe_experts, self.moe_tp, self.moe_top_k)

        print("  moe routing")
        print("    layer-events: " + String(events)
            + "  slots: " + String(total_slots)
            + "  experts/layer: " + String(self.moe_experts)
            + "  top-k: " + String(self.moe_top_k)
            + "  tp: " + String(self.moe_tp))
        print("    max rank load/event: ideal " + format_milli3(ideal_rank_load)
            + "  avg " + format_milli3(max_load.mean_ns)
            + "  p90 " + format_milli3(max_load.p90_ns)
            + "  p99 " + format_milli3(max_load.p99_ns)
            + "  max " + format_milli3(max_load.max_ns)
            + "  active-ranks avg " + format_milli3(active_ranks.mean_ns))
        if random_max_mean > 0:
            print("    random occupancy baseline: max-load avg "
                + format_milli3(random_max_mean)
                + "  p90 " + String(random_max_p90)
                + "  p99 " + String(random_max_p99))
        print("    same-rank expert pairs/event: ideal " + format_milli3(ideal_pairs)
            + "  random " + format_milli3(random_pairs)
            + "  avg " + format_milli3(same_rank_pairs.mean_ns)
            + "  p90 " + format_milli3(same_rank_pairs.p90_ns)
            + "  p99 " + format_milli3(same_rank_pairs.p99_ns)
            + "  max " + format_milli3(same_rank_pairs.max_ns))

        var rank_line = "    rank slot share:"
        for r in range(self.moe_tp):
            var rank_total = 0
            for layer in range(self.moe_layers):
                rank_total += self.moe_rank_counts[layer * self.moe_tp + r]
            rank_line += " r" + String(r) + " " + format_pct1(rank_total, total_slots)
        print(rank_line)

        var layer_rows = InlineArray[MoeLayerRankRow, 5](fill=MoeLayerRankRow())
        for layer in range(self.moe_layers):
            var layer_events = self.moe_layer_events[layer]
            if layer_events <= 0:
                continue
            var best_rank = 0
            var best_count = 0
            for r in range(self.moe_tp):
                var c = self.moe_rank_counts[layer * self.moe_tp + r]
                if c > best_count:
                    best_count = c
                    best_rank = r
            var insert = -1
            for slot in range(5):
                if moe_ratio_better(best_count, layer_events, layer_rows[slot].count, layer_rows[slot].events):
                    insert = slot
                    break
            if insert >= 0:
                for slot in range(4, insert, -1):
                    layer_rows[slot] = layer_rows[slot - 1]
                layer_rows[insert] = MoeLayerRankRow(layer, best_rank, best_count, layer_events)

        print("    most imbalanced layers")
        for slot in range(5):
            var row = layer_rows[slot]
            if row.layer < 0:
                continue
            var avg_milli = (row.count * 1000 + row.events // 2) // row.events
            print("      L" + pad_left(String(row.layer), 2)
                + " rank " + String(row.rank)
                + " avg-load " + format_milli3(avg_milli)
                + " share " + format_pct1(row.count, row.events * self.moe_top_k))

        var hot_layer_rows = InlineArray[MoeLayerHotnessRow, 5](
            fill=MoeLayerHotnessRow())
        for layer in range(self.moe_layers):
            var layer_events = self.moe_layer_events[layer]
            if layer_events <= 0:
                continue
            var top_counts = InlineArray[Int, 8](fill=0)
            var top_ids = InlineArray[Int, 8](fill=-1)
            var base = layer * self.moe_experts
            for expert in range(self.moe_experts):
                var count = self.moe_expert_counts[base + expert]
                if count > top_counts[7]:
                    var insert = 7
                    while insert > 0 and count > top_counts[insert - 1]:
                        top_counts[insert] = top_counts[insert - 1]
                        top_ids[insert] = top_ids[insert - 1]
                        insert -= 1
                    top_counts[insert] = count
                    top_ids[insert] = expert
            var top8_count = 0
            for i in range(8):
                top8_count += top_counts[i]
            var avg_max_load_milli = (
                self.moe_layer_max_rank_load[layer] * 1000 + layer_events // 2
            ) // layer_events
            var avg_same_pairs_milli = (
                self.moe_layer_same_rank_pairs[layer] * 1000 + layer_events // 2
            ) // layer_events

            var insert = -1
            for slot in range(5):
                if moe_ratio_better(
                    top_counts[0], layer_events,
                    hot_layer_rows[slot].top1_count,
                    hot_layer_rows[slot].events,
                ):
                    insert = slot
                    break
            if insert >= 0:
                for slot in range(4, insert, -1):
                    hot_layer_rows[slot] = hot_layer_rows[slot - 1]
                hot_layer_rows[insert] = MoeLayerHotnessRow(
                    layer, top_ids[0], top_counts[0], top8_count,
                    layer_events, avg_max_load_milli,
                    avg_same_pairs_milli)

        print("    most concentrated layers")
        for slot in range(5):
            var row = hot_layer_rows[slot]
            if row.layer < 0:
                continue
            print("      L" + pad_left(String(row.layer), 2)
                + " top E" + pad_left(String(row.expert), 3)
                + " hit " + format_pct1(row.top1_count, row.events)
                + "  top8-cover " + format_pct1(row.top8_count, row.events * self.moe_top_k)
                + "  max-load " + format_milli3(row.avg_max_load_milli)
                + "  same-rank-pairs " + format_milli3(row.avg_same_rank_pairs_milli))

        var hot_rows = InlineArray[MoeHotExpertRow, 8](fill=MoeHotExpertRow())
        var experts_per_rank = self.moe_experts // self.moe_tp
        for layer in range(self.moe_layers):
            var layer_events = self.moe_layer_events[layer]
            if layer_events <= 0:
                continue
            var base = layer * self.moe_experts
            for expert in range(self.moe_experts):
                var count = self.moe_expert_counts[base + expert]
                if count <= 0:
                    continue
                var rank = expert // experts_per_rank
                var insert = -1
                for slot in range(8):
                    if moe_ratio_better(count, layer_events, hot_rows[slot].count, hot_rows[slot].events):
                        insert = slot
                        break
                if insert >= 0:
                    for slot in range(7, insert, -1):
                        hot_rows[slot] = hot_rows[slot - 1]
                    hot_rows[insert] = MoeHotExpertRow(layer, expert, rank, count, layer_events)

        print("    hottest experts")
        for slot in range(8):
            var row = hot_rows[slot]
            if row.layer < 0:
                continue
            print("      L" + pad_left(String(row.layer), 2)
                + ":E" + pad_left(String(row.expert), 3)
                + " rank " + String(row.rank)
                + " hits " + String(row.count) + "/" + String(row.events)
                + " (" + format_pct1(row.count, row.events) + " of layer routes)")

    def report(self, label: String):
        var n = len(self.samples)
        var np = len(self.names)
        if n == 0:
            print(label + ": no profiled forwards")
            return

        var wall_vals = List[Int](capacity=n)
        var phase_sum_vals = List[Int](capacity=n)
        var min_pos = self.samples[0].pos
        var max_pos = self.samples[0].pos
        var total_wall_ns = 0
        var total_phase_sum_ns = 0
        var total_tokens = 0
        var next_token_forwards = 0
        for i in range(n):
            var s = self.samples[i]
            var tokens = s.token_count
            if tokens <= 0:
                tokens = 1
            var end_pos = s.pos + tokens - 1
            wall_vals.append(s.wall_ns)
            var phase_sum_ns = s.phase_sum_ns(np)
            phase_sum_vals.append(phase_sum_ns)
            total_wall_ns += s.wall_ns
            total_phase_sum_ns += phase_sum_ns
            total_tokens += tokens
            if s.produces_next_token:
                next_token_forwards += 1
            if s.pos < min_pos:
                min_pos = s.pos
            if end_pos > max_pos:
                max_pos = end_pos
        var wall = ns_stats(wall_vals)
        var phase_sum = ns_stats(phase_sum_vals)
        var wall_per_token = safe_div_int(total_wall_ns, total_tokens)
        var phase_sum_per_token = safe_div_int(total_phase_sum_ns, total_tokens)

        print(label + " forward profile")
        print("  forwards:  " + String(n)
            + "  input-tokens: " + String(total_tokens)
            + "  next-token-forwards: " + String(next_token_forwards))
        print("  positions: " + String(min_pos) + ".." + String(max_pos))
        print("  wall / forward")
        print("    avg    " + pad_left(format_ms3(wall.mean_ns), 9)
            + " ms    stddev " + pad_left(format_ms3(wall.stddev_ns), 9) + " ms")
        print("    p50    " + pad_left(format_ms3(wall.p50_ns), 9)
            + " ms    p90    " + pad_left(format_ms3(wall.p90_ns), 9) + " ms")
        print("    p99    " + pad_left(format_ms3(wall.p99_ns), 9)
            + " ms    max    " + pad_left(format_ms3(wall.max_ns), 9) + " ms")
        print("  wall / input token")
        print("    aggregate " + pad_left(format_ms3(wall_per_token), 9)
            + " ms/token")
        print("  phase-sum")
        print("    avg/forward " + pad_left(format_ms3(phase_sum.mean_ns), 9)
            + " ms    aggregate " + pad_left(format_ms3(phase_sum_per_token), 9)
            + " ms/input-token"
            + "    overlap-counted "
            + pad_left(format_pct1(total_phase_sum_ns, total_wall_ns), 8)
            + " of wall")
        var rows = List[PhaseReportRow](capacity=np)
        for phase_idx in range(np):
            var dispatch_vals = List[Int](capacity=n)
            var kernel_vals = List[Int](capacity=n)
            var join_vals = List[Int](capacity=n)
            for i in range(n):
                var t = self.samples[i].phases[phase_idx]
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
            if ps.active_count == 0:
                continue
            if ps.active_total.max_ns < 50_000 and ps.active_dispatch.p99_ns < 50_000 and ps.active_join.p99_ns < 50_000:
                if omitted.byte_length() > 0:
                    omitted += ", "
                omitted += self.names[row.phase_idx]
                continue
            if shown < 5:
                if top_summary.byte_length() > 0:
                    top_summary += "  "
                top_summary += self.names[row.phase_idx] + " "
                top_summary += format_pct1(ps.sum_ns, total_wall_ns)
                shown += 1

        if top_summary.byte_length() > 0:
            print("  hot path:  " + top_summary)
        print("  note: share is total phase time / total wall time. avg/forward includes skipped forwards; active columns only include forwards where the phase ran.")

        print("  phases")
        print("    "
            + pad_right("phase", 20)
            + pad_left("share", 8) + "  "
            + pad_left("avg/fwd", 9) + "  "
            + pad_left("avg/tok", 9) + "  "
            + pad_left("act avg", 9) + "  "
            + pad_left("act p99", 9) + "  "
            + pad_left("act max", 9) + "  "
            + pad_left("runs", 9)
            + " | dispatch a/p99 | kernel a/p99 | join a/p99  [ms]")
        print("    " + repeat_spaces(20) + "--------  ---------  ---------  ---------  ---------  ---------  --------- | -------------- | ------------ | ----------")

        for i in range(len(rows)):
            var row = rows[i]
            var ps = row.stats
            if ps.active_count == 0:
                continue
            if ps.active_total.max_ns < 50_000 and ps.active_dispatch.p99_ns < 50_000 and ps.active_join.p99_ns < 50_000:
                continue
            print("    "
                + pad_right(self.names[row.phase_idx], 20)
                + pad_left(format_pct1(ps.sum_ns, total_wall_ns), 8) + "  "
                + pad_left(format_ms3(ps.total.mean_ns), 9) + "  "
                + pad_left(format_ms3(safe_div_int(ps.sum_ns, total_tokens)), 9) + "  "
                + pad_left(format_ms3(ps.active_total.mean_ns), 9) + "  "
                + pad_left(format_ms3(ps.active_total.p99_ns), 9) + "  "
                + pad_left(format_ms3(ps.active_total.max_ns), 9) + "  "
                + pad_left(String(ps.active_count) + "/" + String(n), 9)
                + " | " + pad_left(format_ms3(ps.active_dispatch.mean_ns), 6)
                + "/" + pad_left(format_ms3(ps.active_dispatch.p99_ns), 6)
                + " | " + pad_left(format_ms3(ps.active_kernel.mean_ns), 6)
                + "/" + pad_left(format_ms3(ps.active_kernel.p99_ns), 6)
                + " | " + pad_left(format_ms3(ps.active_join.mean_ns), 6)
                + "/" + pad_left(format_ms3(ps.active_join.p99_ns), 6))
        if omitted.byte_length() > 0:
            print("    omitted tiny phases: " + omitted)
        self.report_moe()
