from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import simd_width_of

from numa import NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from experimental3.common_math import BF16Ptr, F32Ptr
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs
from minimax.kernels.activations import sigmoid_f32
from minimax.kernels.gemm import f32_gemv_row
from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config, MiniMaxM27ButterQuant,
)


comptime C = MiniMaxM27Config
comptime TP = 4
comptime MODEL_DIR = "quantized_models"
comptime DUMP_DIR = "m27_dump_prefill"
comptime PROMPT_TOKENS = 42
comptime BF_CANDIDATES = 32


@always_inline
def dot_bf16_weight_row[K: Int](act: BF16Ptr, weight_row: F32Ptr) -> Float32:
    """Simulate AMX-bf16 router inputs: bf16 activation, bf16-rounded weights,
    f32 accumulation.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, K]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var a = (act + off).load[width=width]().cast[DType.float32]()
        var w = (weight_row + off).load[width=width, non_temporal=True]()
        var wb = w.cast[DType.bfloat16]().cast[DType.float32]()
        accs[i] = a * wb
    var k = step
    while k + step <= K:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var a = (act + off).load[width=width]().cast[DType.float32]()
            var w = (weight_row + off).load[width=width, non_temporal=True]()
            var wb = w.cast[DType.bfloat16]().cast[DType.float32]()
            accs[i] = a.fma(wb, accs[i])
        k += step
    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width, non_temporal=True]()
        var wb = w.cast[DType.bfloat16]().cast[DType.float32]()
        accs[0] = a.fma(wb, accs[0])
        k += width
    return tree_reduce_accs(accs)


struct TopScores[k: Int]:
    var eids: InlineArray[Int, Self.k]
    var scores: InlineArray[Float32, Self.k]
    var raws: InlineArray[Float32, Self.k]

    def __init__(out self):
        self.eids = InlineArray[Int, Self.k](fill=-1)
        self.scores = InlineArray[Float32, Self.k](fill=Float32(-1e30))
        self.raws = InlineArray[Float32, Self.k](fill=Float32(0))

    def offer(mut self, eid: Int, score: Float32, raw: Float32):
        if score > self.scores[Self.k - 1]:
            var slot = Self.k - 1
            while slot > 0 and score > self.scores[slot - 1]:
                self.eids[slot] = self.eids[slot - 1]
                self.scores[slot] = self.scores[slot - 1]
                self.raws[slot] = self.raws[slot - 1]
                slot -= 1
            self.eids[slot] = eid
            self.scores[slot] = score
            self.raws[slot] = raw

    def contains(self, eid: Int, limit: Int) -> Bool:
        var n = limit
        if n > Self.k:
            n = Self.k
        for i in range(n):
            if self.eids[i] == eid:
                return True
        return False

    def print_eids(self, limit: Int):
        var n = limit
        if n > Self.k:
            n = Self.k
        for i in range(n):
            print(" ", self.eids[i], end="")
        print()


def same_order[a: Int, b: Int](lhs: TopScores[a], rhs: TopScores[b], n: Int) -> Bool:
    for i in range(n):
        if lhs.eids[i] != rhs.eids[i]:
            return False
    return True


def same_set[a: Int, b: Int](lhs: TopScores[a], rhs: TopScores[b], n: Int) -> Bool:
    for i in range(n):
        if not rhs.contains(lhs.eids[i], n):
            return False
    return True


def recall_count[a: Int, b: Int](lhs: TopScores[a], rhs: TopScores[b], n: Int) -> Int:
    var hits = 0
    for i in range(C.TOP_K):
        if rhs.contains(lhs.eids[i], n):
            hits += 1
    return hits


struct RouterAnalysisStats:
    var events: Int
    var candidates: Int
    var logit_min: Float32
    var logit_max: Float32
    var score_min: Float32
    var score_max: Float32
    var raw_min: Float32
    var raw_max: Float32
    var logit_sum: Float64
    var raw_sum: Float64
    var score_sum: Float64
    var abs_logit_err_sum: Float64
    var abs_score_err_sum: Float64
    var max_abs_logit_err: Float32
    var max_abs_score_err: Float32
    var margin_sum: Float64
    var margin_min: Float32
    var margin_lt_1e6: Int
    var margin_lt_1e5: Int
    var margin_lt_1e4: Int
    var margin_lt_1e3: Int
    var margin_lt_1e2: Int
    var low_raw: Int
    var high_raw: Int
    var low_logit: Int
    var high_logit: Int
    var exact_order: Int
    var exact_set: Int
    var raw_score_same_order: Int
    var raw_score_same_set: Int
    var recall8: Int
    var recall12: Int
    var recall16: Int
    var recall24: Int
    var recall32: Int
    var all8_in8: Int
    var all8_in12: Int
    var all8_in16: Int
    var all8_in24: Int
    var all8_in32: Int
    var mismatch_printed: Int

    def __init__(out self):
        self.events = 0
        self.candidates = 0
        self.logit_min = Float32(1e30)
        self.logit_max = Float32(-1e30)
        self.score_min = Float32(1e30)
        self.score_max = Float32(-1e30)
        self.raw_min = Float32(1e30)
        self.raw_max = Float32(-1e30)
        self.logit_sum = Float64(0)
        self.raw_sum = Float64(0)
        self.score_sum = Float64(0)
        self.abs_logit_err_sum = Float64(0)
        self.abs_score_err_sum = Float64(0)
        self.max_abs_logit_err = Float32(0)
        self.max_abs_score_err = Float32(0)
        self.margin_sum = Float64(0)
        self.margin_min = Float32(1e30)
        self.margin_lt_1e6 = 0
        self.margin_lt_1e5 = 0
        self.margin_lt_1e4 = 0
        self.margin_lt_1e3 = 0
        self.margin_lt_1e2 = 0
        self.low_raw = 0
        self.high_raw = 0
        self.low_logit = 0
        self.high_logit = 0
        self.exact_order = 0
        self.exact_set = 0
        self.raw_score_same_order = 0
        self.raw_score_same_set = 0
        self.recall8 = 0
        self.recall12 = 0
        self.recall16 = 0
        self.recall24 = 0
        self.recall32 = 0
        self.all8_in8 = 0
        self.all8_in12 = 0
        self.all8_in16 = 0
        self.all8_in24 = 0
        self.all8_in32 = 0
        self.mismatch_printed = 0

    def record_candidate(mut self, logit: Float32, raw: Float32, score: Float32,
                         bf_logit: Float32, bf_score: Float32):
        self.candidates += 1
        if logit < self.logit_min:
            self.logit_min = logit
        if logit > self.logit_max:
            self.logit_max = logit
        if raw < self.raw_min:
            self.raw_min = raw
        if raw > self.raw_max:
            self.raw_max = raw
        if score < self.score_min:
            self.score_min = score
        if score > self.score_max:
            self.score_max = score
        self.logit_sum += Float64(logit)
        self.raw_sum += Float64(raw)
        self.score_sum += Float64(score)
        var le = abs(logit - bf_logit)
        var se = abs(score - bf_score)
        self.abs_logit_err_sum += Float64(le)
        self.abs_score_err_sum += Float64(se)
        if le > self.max_abs_logit_err:
            self.max_abs_logit_err = le
        if se > self.max_abs_score_err:
            self.max_abs_score_err = se
        if raw < Float32(0.0001):
            self.low_raw += 1
        if raw > Float32(0.9999):
            self.high_raw += 1
        if logit < Float32(-10):
            self.low_logit += 1
        if logit > Float32(10):
            self.high_logit += 1

    def record_event(mut self, layer: Int, pos: Int,
                     fp: TopScores[9], bf: TopScores[BF_CANDIDATES],
                     raw_top: TopScores[C.TOP_K]):
        self.events += 1
        var margin = fp.scores[C.TOP_K - 1] - fp.scores[C.TOP_K]
        self.margin_sum += Float64(margin)
        if margin < self.margin_min:
            self.margin_min = margin
        if margin < Float32(0.000001):
            self.margin_lt_1e6 += 1
        if margin < Float32(0.00001):
            self.margin_lt_1e5 += 1
        if margin < Float32(0.0001):
            self.margin_lt_1e4 += 1
        if margin < Float32(0.001):
            self.margin_lt_1e3 += 1
        if margin < Float32(0.01):
            self.margin_lt_1e2 += 1

        if same_order(fp, bf, C.TOP_K):
            self.exact_order += 1
        if same_set(fp, bf, C.TOP_K):
            self.exact_set += 1
        if same_order(fp, raw_top, C.TOP_K):
            self.raw_score_same_order += 1
        if same_set(fp, raw_top, C.TOP_K):
            self.raw_score_same_set += 1

        var r8 = recall_count(fp, bf, 8)
        var r12 = recall_count(fp, bf, 12)
        var r16 = recall_count(fp, bf, 16)
        var r24 = recall_count(fp, bf, 24)
        var r32 = recall_count(fp, bf, 32)
        self.recall8 += r8
        self.recall12 += r12
        self.recall16 += r16
        self.recall24 += r24
        self.recall32 += r32
        if r8 == C.TOP_K:
            self.all8_in8 += 1
        if r12 == C.TOP_K:
            self.all8_in12 += 1
        if r16 == C.TOP_K:
            self.all8_in16 += 1
        if r24 == C.TOP_K:
            self.all8_in24 += 1
        if r32 == C.TOP_K:
            self.all8_in32 += 1

        if r8 != C.TOP_K and self.mismatch_printed < 12:
            print("mismatch layer", layer, "pos", pos, "margin", margin)
            print("  fp top8:", end="")
            fp.print_eids(C.TOP_K)
            print("  bf top8:", end="")
            bf.print_eids(C.TOP_K)
            print("  bf top16:", end="")
            bf.print_eids(16)
            self.mismatch_printed += 1


def pct(part: Int, total: Int) -> Float64:
    if total == 0:
        return Float64(0)
    return Float64(100.0) * Float64(part) / Float64(total)


def print_summary(stats: RouterAnalysisStats):
    print()
    print("=== M27 router fp32 vs bf16-weight simulation ===")
    print("events:", stats.events, "candidate scores:", stats.candidates)
    print("logit range:", stats.logit_min, stats.logit_max,
          "mean:", stats.logit_sum / Float64(stats.candidates))
    print("raw sigmoid range:", stats.raw_min, stats.raw_max,
          "mean:", stats.raw_sum / Float64(stats.candidates))
    print("score range:", stats.score_min, stats.score_max,
          "mean:", stats.score_sum / Float64(stats.candidates))
    print("raw saturation <1e-4:", stats.low_raw,
          ">", "0.9999:", stats.high_raw,
          "logit <-10:", stats.low_logit,
          "logit >10:", stats.high_logit)
    print("abs logit error mean:",
          stats.abs_logit_err_sum / Float64(stats.candidates),
          "max:", stats.max_abs_logit_err)
    print("abs score error mean:",
          stats.abs_score_err_sum / Float64(stats.candidates),
          "max:", stats.max_abs_score_err)
    print("fp32 top8/top9 margin min:", stats.margin_min,
          "mean:", stats.margin_sum / Float64(stats.events))
    print("margin counts <1e-6/<1e-5/<1e-4/<1e-3/<1e-2:",
          stats.margin_lt_1e6, stats.margin_lt_1e5,
          stats.margin_lt_1e4, stats.margin_lt_1e3,
          stats.margin_lt_1e2)
    print("bf16 exact top8 order:",
          stats.exact_order, "/", stats.events, pct(stats.exact_order, stats.events), "%")
    print("bf16 exact top8 set:",
          stats.exact_set, "/", stats.events, pct(stats.exact_set, stats.events), "%")
    var slot_total = stats.events * C.TOP_K
    print("fp32 top8 slot recall in bf top8/top12/top16/top24/top32:",
          stats.recall8, stats.recall12, stats.recall16,
          stats.recall24, stats.recall32, "of", slot_total)
    print("all fp32 top8 contained in bf top8/top12/top16/top24/top32:",
          stats.all8_in8, stats.all8_in12, stats.all8_in16,
          stats.all8_in24, stats.all8_in32, "of", stats.events)
    print("score top8 equals raw-only top8 order/set:",
          stats.raw_score_same_order, stats.raw_score_same_set,
          "of", stats.events)


def analyze[
    P: BurstThreadPool, //,
](
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
) raises:
    var model_opt = MiniMaxM27ButterQuant[TP, P].load(
        Path(MODEL_DIR), numa, numa_topo, pools^)
    if not model_opt:
        print("failed to load model")
        return
    var model = model_opt.take()
    var stats = RouterAnalysisStats()

    for layer_idx in range(C.NUM_LAYERS):
        print("layer", layer_idx)
        for pos in range(PROMPT_TOKENS):
            var path = Path(DUMP_DIR) / (
                "p" + String(pos) + "_l" + String(layer_idx)
                + "_r0_07_normed_bf16.bin")
            if not path.exists():
                raise Error("missing router activation dump: " + String(path))
            var bytes = path.read_bytes()
            if len(bytes) != C.HIDDEN * 2:
                raise Error("bad dump size: " + String(path))
            var act = BF16Ptr(unsafe_from_address=Int(bytes.unsafe_ptr()))

            var fp_top = TopScores[9]()
            var bf_top = TopScores[BF_CANDIDATES]()
            var raw_top = TopScores[C.TOP_K]()

            for rank in range(TP):
                var topo = model.topos[rank]
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var w = layer.body.router_proj.bound(lb).as_ptr[DType.float32]()
                var b = layer.body.router_bias.bound(lb).as_ptr[DType.float32]()
                comptime experts_per_rank = C.NUM_EXPERTS // TP
                var eid_base = rank * experts_per_rank

                for local_eid in range(experts_per_rank):
                    var eid = eid_base + local_eid
                    var row = w + local_eid * C.HIDDEN
                    var logit = f32_gemv_row[C.HIDDEN](act, row)
                    var raw = sigmoid_f32[1](SIMD[DType.float32, 1](logit))[0]
                    var score = raw + b[local_eid]

                    var bf_logit = dot_bf16_weight_row[C.HIDDEN](act, row)
                    var bf_raw = sigmoid_f32[1](SIMD[DType.float32, 1](bf_logit))[0]
                    var bf_score = bf_raw + b[local_eid]

                    stats.record_candidate(logit, raw, score, bf_logit, bf_score)
                    fp_top.offer(eid, score, raw)
                    bf_top.offer(eid, bf_score, bf_raw)
                    raw_top.offer(eid, raw, raw)

            stats.record_event(layer_idx, pos, fp_top, bf_top, raw_top)

    print_summary(stats)
    _ = model


def main() raises:
    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)
    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        analyze(numa, numa_topo, pools^)
    else:
        print("mode: burst")
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        analyze(numa, numa_topo, pools^)
