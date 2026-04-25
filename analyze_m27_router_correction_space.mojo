from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import simd_width_of
from std.collections import InlineArray

from numa import NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from experimental3.common_math import BF16Ptr, F32Ptr
from minimax.kernels.activations import sigmoid_f32
from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config, MiniMaxM27ButterQuant,
)


comptime C = MiniMaxM27Config
comptime TP = 4
comptime MODEL_DIR = "quantized_models"
comptime DUMP_DIR = "m27_dump_prefill"
comptime PROMPT_TOKENS = 42
comptime BF_CANDIDATES = 32
comptime CENTER_BLOCK = 64
comptime CENTER_BLOCKS = C.HIDDEN // CENTER_BLOCK


struct TopScores[k: Int]:
    var eids: InlineArray[Int, Self.k]
    var scores: InlineArray[Float32, Self.k]

    def __init__(out self):
        self.eids = InlineArray[Int, Self.k](fill=-1)
        self.scores = InlineArray[Float32, Self.k](fill=Float32(-1e30))

    def offer(mut self, eid: Int, score: Float32):
        if score > self.scores[Self.k - 1]:
            var slot = Self.k - 1
            while slot > 0 and score > self.scores[slot - 1]:
                self.eids[slot] = self.eids[slot - 1]
                self.scores[slot] = self.scores[slot - 1]
                slot -= 1
            self.eids[slot] = eid
            self.scores[slot] = score

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


struct VariantStats:
    var events: Int
    var candidates: Int
    var exact_order: Int
    var exact_set: Int
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
    var abs_logit_err_sum: Float64
    var abs_score_err_sum: Float64
    var max_abs_logit_err: Float32
    var max_abs_score_err: Float32
    var mismatch_printed: Int

    def __init__(out self):
        self.events = 0
        self.candidates = 0
        self.exact_order = 0
        self.exact_set = 0
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
        self.abs_logit_err_sum = Float64(0)
        self.abs_score_err_sum = Float64(0)
        self.max_abs_logit_err = Float32(0)
        self.max_abs_score_err = Float32(0)
        self.mismatch_printed = 0

    def record_candidate(mut self, fp_logit: Float32, alt_logit: Float32,
                         fp_score: Float32, alt_score: Float32):
        self.candidates += 1
        var le = abs(fp_logit - alt_logit)
        var se = abs(fp_score - alt_score)
        self.abs_logit_err_sum += Float64(le)
        self.abs_score_err_sum += Float64(se)
        if le > self.max_abs_logit_err:
            self.max_abs_logit_err = le
        if se > self.max_abs_score_err:
            self.max_abs_score_err = se

    def record_event(mut self, label: String, layer: Int, pos: Int,
                     fp: TopScores[9], alt: TopScores[BF_CANDIDATES]):
        self.events += 1
        if same_order(fp, alt, C.TOP_K):
            self.exact_order += 1
        if same_set(fp, alt, C.TOP_K):
            self.exact_set += 1
        var r8 = recall_count(fp, alt, 8)
        var r12 = recall_count(fp, alt, 12)
        var r16 = recall_count(fp, alt, 16)
        var r24 = recall_count(fp, alt, 24)
        var r32 = recall_count(fp, alt, 32)
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
        if r8 != C.TOP_K and self.mismatch_printed < 4:
            print("mismatch", label, "layer", layer, "pos", pos)
            print("  fp top8:", end="")
            fp.print_eids(C.TOP_K)
            print("  alt top8:", end="")
            alt.print_eids(C.TOP_K)
            print("  alt top16:", end="")
            alt.print_eids(16)
            self.mismatch_printed += 1


@fieldwise_init
struct RouterLogits(Copyable, ImplicitlyCopyable):
    var fp: Float32
    var naive: Float32
    var gauge_no_pivot: Float32
    var gauge_mean: Float32
    var gauge_bf16_pivot: Float32
    var gauge_layer_pivot: Float32
    var gauge_row_mean: Float32
    var gauge_block_mean: Float32
    var gauge_block_mid: Float32
    var row_mean: Float32
    var block_mean: Float32
    var block_mid: Float32


@always_inline
def compute_router_logits[K: Int, block: Int](
    act: BF16Ptr,
    weight_row: F32Ptr,
    gauge: F32Ptr,
    gauge_pivot: Float32,
    gauge_pivot_bf16: Float32,
    gauge_layer_pivot: Float32,
    gauge_row_center: Float32,
    gauge_mean_centers: F32Ptr,
    gauge_mid_centers: F32Ptr,
    row_center: Float32,
    mean_centers: F32Ptr,
    mid_centers: F32Ptr,
    block_sums: F32Ptr,
    sum_x: Float32,
) -> RouterLogits:
    """One SIMD pass over the router row, evaluating all correction-space
    variants. This keeps the ablation full-set while avoiding separate dot
    products for each variant.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime blocks = K // block
    var fp_acc = SIMD[DType.float32, width](0)
    var naive_acc = SIMD[DType.float32, width](0)
    var gauge_acc = SIMD[DType.float32, width](0)
    var gauge_row_acc = SIMD[DType.float32, width](0)
    var gauge_block_mean_acc = SIMD[DType.float32, width](0)
    var gauge_block_mid_acc = SIMD[DType.float32, width](0)
    var row_acc = SIMD[DType.float32, width](0)
    var block_mean_acc = SIMD[DType.float32, width](0)
    var block_mid_acc = SIMD[DType.float32, width](0)
    var gauge_row_center_v = SIMD[DType.float32, width](gauge_row_center)
    var row_center_v = SIMD[DType.float32, width](row_center)
    var gauge_block_mean_correction = Float32(0)
    var gauge_block_mid_correction = Float32(0)
    var block_mean_correction = Float32(0)
    var block_mid_correction = Float32(0)
    for blk in range(blocks):
        var gauge_mean_center = gauge_mean_centers[blk]
        var gauge_mid_center = gauge_mid_centers[blk]
        var mean_center = mean_centers[blk]
        var mid_center = mid_centers[blk]
        var gauge_mean_center_v = SIMD[DType.float32, width](gauge_mean_center)
        var gauge_mid_center_v = SIMD[DType.float32, width](gauge_mid_center)
        var mean_center_v = SIMD[DType.float32, width](mean_center)
        var mid_center_v = SIMD[DType.float32, width](mid_center)
        gauge_block_mean_correction += gauge_mean_center * block_sums[blk]
        gauge_block_mid_correction += gauge_mid_center * block_sums[blk]
        block_mean_correction += mean_center * block_sums[blk]
        block_mid_correction += mid_center * block_sums[blk]
        var base = blk * block
        for k in range(0, block, width):
            var off = base + k
            var a = (act + off).load[width=width]().cast[DType.float32]()
            var w = (weight_row + off).load[width=width, non_temporal=True]()
            var g = (gauge + off).load[width=width]()
            fp_acc = a.fma(w, fp_acc)
            naive_acc = a.fma(
                w.cast[DType.bfloat16]().cast[DType.float32](), naive_acc)
            gauge_acc = a.fma(
                (w - g).cast[DType.bfloat16]().cast[DType.float32](),
                gauge_acc)
            var resid = w - g
            gauge_row_acc = a.fma(
                (resid - gauge_row_center_v).cast[DType.bfloat16]().cast[
                    DType.float32](),
                gauge_row_acc)
            gauge_block_mean_acc = a.fma(
                (resid - gauge_mean_center_v).cast[DType.bfloat16]().cast[
                    DType.float32](),
                gauge_block_mean_acc)
            gauge_block_mid_acc = a.fma(
                (resid - gauge_mid_center_v).cast[DType.bfloat16]().cast[
                    DType.float32](),
                gauge_block_mid_acc)
            row_acc = a.fma(
                (w - row_center_v).cast[DType.bfloat16]().cast[DType.float32](),
                row_acc)
            block_mean_acc = a.fma(
                (w - mean_center_v).cast[DType.bfloat16]().cast[DType.float32](),
                block_mean_acc)
            block_mid_acc = a.fma(
                (w - mid_center_v).cast[DType.bfloat16]().cast[DType.float32](),
                block_mid_acc)
    var gauge_delta = gauge_acc.reduce_add()
    return RouterLogits(
        fp_acc.reduce_add(),
        naive_acc.reduce_add(),
        gauge_delta,
        gauge_delta + gauge_pivot,
        gauge_delta + gauge_pivot_bf16,
        gauge_delta + gauge_layer_pivot,
        gauge_row_acc.reduce_add() + gauge_pivot + gauge_row_center * sum_x,
        gauge_block_mean_acc.reduce_add() + gauge_pivot
            + gauge_block_mean_correction,
        gauge_block_mid_acc.reduce_add() + gauge_pivot
            + gauge_block_mid_correction,
        row_acc.reduce_add() + row_center * sum_x,
        block_mean_acc.reduce_add() + block_mean_correction,
        block_mid_acc.reduce_add() + block_mid_correction,
    )


def build_layer_centers[
    P: BurstThreadPool, //,
](
    model: MiniMaxM27ButterQuant[TP, P],
    layer_idx: Int,
    gauge: F32Ptr,
    gauge_row_centers: F32Ptr,
    gauge_block_mean_centers: F32Ptr,
    gauge_block_mid_centers: F32Ptr,
    row_centers: F32Ptr,
    block_mean_centers: F32Ptr,
    block_mid_centers: F32Ptr,
):
    comptime experts_per_rank = C.NUM_EXPERTS // TP
    for k in range(C.HIDDEN):
        gauge[k] = Float32(0)
    for rank in range(TP):
        var topo = model.topos[rank]
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var layer = topo.layers.proto
        var w = layer.body.router_proj.bound(lb).as_ptr[DType.float32]()
        var eid_base = rank * experts_per_rank
        for local_eid in range(experts_per_rank):
            var eid = eid_base + local_eid
            var row = w + local_eid * C.HIDDEN
            var sum = Float32(0)
            for k in range(C.HIDDEN):
                var v = row[k]
                sum += v
                gauge[k] += v
            row_centers[eid] = sum / Float32(C.HIDDEN)
            for blk in range(CENTER_BLOCKS):
                var base = blk * CENTER_BLOCK
                var bsum = Float32(0)
                var bmin = Float32(1e30)
                var bmax = Float32(-1e30)
                for k in range(CENTER_BLOCK):
                    var v = row[base + k]
                    bsum += v
                    if v < bmin:
                        bmin = v
                    if v > bmax:
                        bmax = v
                block_mean_centers[eid * CENTER_BLOCKS + blk] = (
                    bsum / Float32(CENTER_BLOCK))
                block_mid_centers[eid * CENTER_BLOCKS + blk] = (
                    (bmin + bmax) * Float32(0.5))
    for k in range(C.HIDDEN):
        gauge[k] = gauge[k] / Float32(C.NUM_EXPERTS)

    for rank in range(TP):
        var topo = model.topos[rank]
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var layer = topo.layers.proto
        var w = layer.body.router_proj.bound(lb).as_ptr[DType.float32]()
        var eid_base = rank * experts_per_rank
        for local_eid in range(experts_per_rank):
            var eid = eid_base + local_eid
            var row = w + local_eid * C.HIDDEN
            var sum = Float32(0)
            for k in range(C.HIDDEN):
                sum += row[k] - gauge[k]
            gauge_row_centers[eid] = sum / Float32(C.HIDDEN)
            for blk in range(CENTER_BLOCKS):
                var base = blk * CENTER_BLOCK
                var bsum = Float32(0)
                var bmin = Float32(1e30)
                var bmax = Float32(-1e30)
                for k in range(CENTER_BLOCK):
                    var v = row[base + k] - gauge[base + k]
                    bsum += v
                    if v < bmin:
                        bmin = v
                    if v > bmax:
                        bmax = v
                gauge_block_mean_centers[eid * CENTER_BLOCKS + blk] = (
                    bsum / Float32(CENTER_BLOCK))
                gauge_block_mid_centers[eid * CENTER_BLOCKS + blk] = (
                    (bmin + bmax) * Float32(0.5))


def fill_activation_sums(
    act: BF16Ptr,
    block_sums: F32Ptr,
) -> Float32:
    var sum = Float32(0)
    for blk in range(CENTER_BLOCKS):
        var bsum = Float32(0)
        var base = blk * CENTER_BLOCK
        for k in range(CENTER_BLOCK):
            bsum += Float32(act[base + k])
        block_sums[blk] = bsum
        sum += bsum
    return sum


@always_inline
def dot_activation_f32[K: Int](act: BF16Ptr, weight: F32Ptr) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0)
    for k in range(0, K, width):
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight + k).load[width=width]()
        acc = a.fma(w, acc)
    return acc.reduce_add()


@always_inline
def dot_activation_bf16_weight[K: Int](act: BF16Ptr, weight: F32Ptr) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0)
    for k in range(0, K, width):
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight + k).load[width=width]().cast[DType.bfloat16]().cast[
            DType.float32]()
        acc = a.fma(w, acc)
    return acc.reduce_add()


def pct(part: Int, total: Int) -> Float64:
    if total == 0:
        return Float64(0)
    return Float64(100.0) * Float64(part) / Float64(total)


def print_variant(label: String, stats: VariantStats):
    print()
    print("==", label, "==")
    print("candidate scores:", stats.candidates)
    print("abs logit error mean:",
          stats.abs_logit_err_sum / Float64(stats.candidates),
          "max:", stats.max_abs_logit_err)
    print("abs score error mean:",
          stats.abs_score_err_sum / Float64(stats.candidates),
          "max:", stats.max_abs_score_err)
    print("exact top8 order:",
          stats.exact_order, "/", stats.events, pct(stats.exact_order, stats.events), "%")
    print("exact top8 set:",
          stats.exact_set, "/", stats.events, pct(stats.exact_set, stats.events), "%")
    var slot_total = stats.events * C.TOP_K
    print("slot recall top8/top12/top16/top24/top32:",
          stats.recall8, stats.recall12, stats.recall16,
          stats.recall24, stats.recall32, "of", slot_total)
    print("all top8 contained in top8/top12/top16/top24/top32:",
          stats.all8_in8, stats.all8_in12, stats.all8_in16,
          stats.all8_in24, stats.all8_in32, "of", stats.events)


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

    var naive = VariantStats()
    var gauge_no_pivot = VariantStats()
    var gauge_mean = VariantStats()
    var gauge_bf16_pivot = VariantStats()
    var gauge_layer_pivot = VariantStats()
    var gauge_row_mean = VariantStats()
    var gauge_block_mean = VariantStats()
    var gauge_block_mid = VariantStats()
    var row_mean = VariantStats()
    var block_mean = VariantStats()
    var block_mid = VariantStats()

    var gauge_arr = InlineArray[Float32, C.HIDDEN](fill=Float32(0))
    var gauge_row_center_arr = InlineArray[
        Float32, C.NUM_EXPERTS](fill=Float32(0))
    var gauge_block_mean_arr = InlineArray[
        Float32, C.NUM_EXPERTS * CENTER_BLOCKS](fill=Float32(0))
    var gauge_block_mid_arr = InlineArray[
        Float32, C.NUM_EXPERTS * CENTER_BLOCKS](fill=Float32(0))
    var row_center_arr = InlineArray[Float32, C.NUM_EXPERTS](fill=Float32(0))
    var block_mean_arr = InlineArray[
        Float32, C.NUM_EXPERTS * CENTER_BLOCKS](fill=Float32(0))
    var block_mid_arr = InlineArray[
        Float32, C.NUM_EXPERTS * CENTER_BLOCKS](fill=Float32(0))
    var block_sum_arr = InlineArray[Float32, CENTER_BLOCKS](fill=Float32(0))

    var gauge = UnsafePointer(to=gauge_arr).bitcast[Float32]()
    var gauge_row_centers = UnsafePointer(to=gauge_row_center_arr).bitcast[
        Float32]()
    var gauge_block_mean_centers = UnsafePointer(to=gauge_block_mean_arr).bitcast[
        Float32]()
    var gauge_block_mid_centers = UnsafePointer(to=gauge_block_mid_arr).bitcast[
        Float32]()
    var row_centers = UnsafePointer(to=row_center_arr).bitcast[Float32]()
    var block_mean_centers = UnsafePointer(to=block_mean_arr).bitcast[Float32]()
    var block_mid_centers = UnsafePointer(to=block_mid_arr).bitcast[Float32]()
    var block_sums = UnsafePointer(to=block_sum_arr).bitcast[Float32]()

    for layer_idx in range(C.NUM_LAYERS):
        print("layer", layer_idx)
        build_layer_centers(
            model, layer_idx, gauge,
            gauge_row_centers, gauge_block_mean_centers,
            gauge_block_mid_centers, row_centers,
                            block_mean_centers, block_mid_centers)
        var gauge_pivot_sum = Float64(0)
        for pos in range(PROMPT_TOKENS):
            var pivot_path = Path(DUMP_DIR) / (
                "p" + String(pos) + "_l" + String(layer_idx)
                + "_r0_07_normed_bf16.bin")
            if not pivot_path.exists():
                raise Error("missing router activation dump: " + String(pivot_path))
            var pivot_bytes = pivot_path.read_bytes()
            if len(pivot_bytes) != C.HIDDEN * 2:
                raise Error("bad dump size: " + String(pivot_path))
            var pivot_act = BF16Ptr(
                unsafe_from_address=Int(pivot_bytes.unsafe_ptr()))
            gauge_pivot_sum += Float64(
                dot_activation_f32[C.HIDDEN](pivot_act, gauge))
        var gauge_layer_pivot_value = Float32(
            gauge_pivot_sum / Float64(PROMPT_TOKENS))
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
            var sum_x = fill_activation_sums(act, block_sums)
            var gauge_pivot = dot_activation_f32[C.HIDDEN](act, gauge)
            var gauge_pivot_bf16 = dot_activation_bf16_weight[
                C.HIDDEN](act, gauge)

            var fp_top = TopScores[9]()
            var naive_top = TopScores[BF_CANDIDATES]()
            var gauge_no_pivot_top = TopScores[BF_CANDIDATES]()
            var gauge_top = TopScores[BF_CANDIDATES]()
            var gauge_bf16_pivot_top = TopScores[BF_CANDIDATES]()
            var gauge_layer_pivot_top = TopScores[BF_CANDIDATES]()
            var gauge_row_top = TopScores[BF_CANDIDATES]()
            var gauge_block_mean_top = TopScores[BF_CANDIDATES]()
            var gauge_block_mid_top = TopScores[BF_CANDIDATES]()
            var row_top = TopScores[BF_CANDIDATES]()
            var block_mean_top = TopScores[BF_CANDIDATES]()
            var block_mid_top = TopScores[BF_CANDIDATES]()

            comptime experts_per_rank = C.NUM_EXPERTS // TP
            for rank in range(TP):
                var topo = model.topos[rank]
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var w = layer.body.router_proj.bound(lb).as_ptr[DType.float32]()
                var b = layer.body.router_bias.bound(lb).as_ptr[DType.float32]()
                var eid_base = rank * experts_per_rank

                for local_eid in range(experts_per_rank):
                    var eid = eid_base + local_eid
                    var row = w + local_eid * C.HIDDEN
                    var bias = b[local_eid]
                    var gauge_mean_centers = (
                        gauge_block_mean_centers + eid * CENTER_BLOCKS)
                    var gauge_mid_centers = (
                        gauge_block_mid_centers + eid * CENTER_BLOCKS)
                    var mean_centers = block_mean_centers + eid * CENTER_BLOCKS
                    var mid_centers = block_mid_centers + eid * CENTER_BLOCKS
                    var logits = compute_router_logits[
                        C.HIDDEN, CENTER_BLOCK](
                        act, row, gauge, gauge_pivot,
                        gauge_pivot_bf16, gauge_layer_pivot_value,
                        gauge_row_centers[eid],
                        gauge_mean_centers, gauge_mid_centers,
                        row_centers[eid],
                        mean_centers, mid_centers, block_sums, sum_x)

                    var fp_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.fp))[0]
                    var fp_score = fp_raw + bias

                    var naive_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.naive))[0]
                    var naive_score = naive_raw + bias

                    var gauge_no_pivot_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_no_pivot))[0]
                    var gauge_no_pivot_score = gauge_no_pivot_raw + bias

                    var gauge_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_mean))[0]
                    var gauge_score = gauge_raw + bias

                    var gauge_bf16_pivot_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_bf16_pivot))[0]
                    var gauge_bf16_pivot_score = gauge_bf16_pivot_raw + bias

                    var gauge_layer_pivot_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_layer_pivot))[0]
                    var gauge_layer_pivot_score = gauge_layer_pivot_raw + bias

                    var gauge_row_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_row_mean))[0]
                    var gauge_row_score = gauge_row_raw + bias

                    var gauge_block_mean_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_block_mean))[0]
                    var gauge_block_mean_score = gauge_block_mean_raw + bias

                    var gauge_block_mid_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.gauge_block_mid))[0]
                    var gauge_block_mid_score = gauge_block_mid_raw + bias

                    var row_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.row_mean))[0]
                    var row_score = row_raw + bias

                    var block_mean_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.block_mean))[0]
                    var block_mean_score = block_mean_raw + bias

                    var block_mid_raw = sigmoid_f32[1](
                        SIMD[DType.float32, 1](logits.block_mid))[0]
                    var block_mid_score = block_mid_raw + bias

                    fp_top.offer(eid, fp_score)
                    naive_top.offer(eid, naive_score)
                    gauge_no_pivot_top.offer(eid, gauge_no_pivot_score)
                    gauge_top.offer(eid, gauge_score)
                    gauge_bf16_pivot_top.offer(eid, gauge_bf16_pivot_score)
                    gauge_layer_pivot_top.offer(eid, gauge_layer_pivot_score)
                    gauge_row_top.offer(eid, gauge_row_score)
                    gauge_block_mean_top.offer(eid, gauge_block_mean_score)
                    gauge_block_mid_top.offer(eid, gauge_block_mid_score)
                    row_top.offer(eid, row_score)
                    block_mean_top.offer(eid, block_mean_score)
                    block_mid_top.offer(eid, block_mid_score)

                    naive.record_candidate(
                        logits.fp, logits.naive, fp_score, naive_score)
                    gauge_no_pivot.record_candidate(
                        logits.fp, logits.gauge_no_pivot,
                        fp_score, gauge_no_pivot_score)
                    gauge_mean.record_candidate(
                        logits.fp, logits.gauge_mean, fp_score, gauge_score)
                    gauge_bf16_pivot.record_candidate(
                        logits.fp, logits.gauge_bf16_pivot,
                        fp_score, gauge_bf16_pivot_score)
                    gauge_layer_pivot.record_candidate(
                        logits.fp, logits.gauge_layer_pivot,
                        fp_score, gauge_layer_pivot_score)
                    gauge_row_mean.record_candidate(
                        logits.fp, logits.gauge_row_mean,
                        fp_score, gauge_row_score)
                    gauge_block_mean.record_candidate(
                        logits.fp, logits.gauge_block_mean,
                        fp_score, gauge_block_mean_score)
                    gauge_block_mid.record_candidate(
                        logits.fp, logits.gauge_block_mid,
                        fp_score, gauge_block_mid_score)
                    row_mean.record_candidate(
                        logits.fp, logits.row_mean, fp_score, row_score)
                    block_mean.record_candidate(
                        logits.fp, logits.block_mean, fp_score, block_mean_score)
                    block_mid.record_candidate(
                        logits.fp, logits.block_mid, fp_score, block_mid_score)

            naive.record_event("naive", layer_idx, pos, fp_top, naive_top)
            gauge_no_pivot.record_event(
                "gauge_no_pivot", layer_idx, pos, fp_top, gauge_no_pivot_top)
            gauge_mean.record_event(
                "gauge_mean", layer_idx, pos, fp_top, gauge_top)
            gauge_bf16_pivot.record_event(
                "gauge_bf16_pivot", layer_idx, pos,
                fp_top, gauge_bf16_pivot_top)
            gauge_layer_pivot.record_event(
                "gauge_layer_pivot", layer_idx, pos,
                fp_top, gauge_layer_pivot_top)
            gauge_row_mean.record_event(
                "gauge_row_mean", layer_idx, pos, fp_top, gauge_row_top)
            gauge_block_mean.record_event(
                "gauge_block64_mean", layer_idx, pos,
                fp_top, gauge_block_mean_top)
            gauge_block_mid.record_event(
                "gauge_block64_mid", layer_idx, pos,
                fp_top, gauge_block_mid_top)
            row_mean.record_event("row_mean", layer_idx, pos, fp_top, row_top)
            block_mean.record_event(
                "block64_mean", layer_idx, pos, fp_top, block_mean_top)
            block_mid.record_event(
                "block64_mid", layer_idx, pos, fp_top, block_mid_top)

    print()
    print("=== M27 router correction-space ablation ===")
    print("events:", naive.events,
          "candidate scores per variant:", naive.candidates)
    print_variant("naive bf16(W)", naive)
    print_variant(
        "gauge no pivot: sigmoid(bf16(W - mean) dot x) + bias",
        gauge_no_pivot)
    print_variant(
        "gauge mean: bf16(W - mean_expert_weight) + dot(x, mean)",
        gauge_mean)
    print_variant(
        "gauge with bf16 pivot: bf16(W - mean) + bf16(mean)",
        gauge_bf16_pivot)
    print_variant(
        "gauge with per-layer constant pivot",
        gauge_layer_pivot)
    print_variant(
        "gauge + row mean residual",
        gauge_row_mean)
    print_variant(
        "gauge + block64 mean residual",
        gauge_block_mean)
    print_variant(
        "gauge + block64 midrange residual",
        gauge_block_mid)
    print_variant("row mean: bf16(W - mean_e) + mean_e*sum(x)", row_mean)
    print_variant(
        "block64 mean: bf16(W_b - mean_eb) + mean_eb*sum(x_b)",
        block_mean)
    print_variant(
        "block64 midrange: bf16(W_b - mid_eb) + mid_eb*sum(x_b)",
        block_mid)
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
