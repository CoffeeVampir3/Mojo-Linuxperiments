"""Router top-k argmax: verification + microbenchmark of three strategies.

Compares three implementations of sigmoid + top-k + renormalize (the MiniMax
M2.7 router). All three match the signature of
`minimax.kernels.router.sigmoid_topk_renorm`:

  1. scalar_topk_renorm   baseline — k-iteration scalar argmax with -inf
                          masking. Mirrors the existing kernel.
  2. parallel_topk_renorm SIMD tournament butterfly with parallel (value,
                          index) register banks. Exact f32 compare semantics.
  3. packed_topk_renorm   SIMD tournament on i32 lanes packed as
                          (canonicalized f32 bit pattern with low log2(N) bits
                          cleared) | index. One register bank, shorter
                          critical path, drops ~log2(N) bits of mantissa.

Correctness test: runs all three over a seeded batch of (logits, bias) inputs
generated to span both positive and negative score regimes. Counts index
disagreements vs. the scalar baseline and reports max absolute weight
difference after renorm.

Benchmark: warm-then-measure loop timed with `perf_counter_ns`. Reports
median nanoseconds per call and speedup over scalar.

Invoke: pixi run mojo router_simd_verification.mojo
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import abs

from simd_math.matrixops import (
    log2, butterfly_shuffle, reduce_top_k, fill_lane_iota,
)
from minimax.kernels.activations import sigmoid_f32
from minimax.kernels.router import TopKResult


# =============================================================================
# Problem shape
# =============================================================================

comptime NUM_EXPERTS = 256
comptime TOP_K = 8
comptime NEG_INF = Float32(-1e30)


# =============================================================================
# Minimal RNG — self-contained so this file can run standalone
# =============================================================================


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed
        for _ in range(8):
            _ = self.next()

    @always_inline
    def next(mut self) -> UInt64:
        var x = self.s
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.s = x
        return x

    @always_inline
    def uniform(mut self) -> Float32:
        return (Float32(self.next() & 0xFFFFFF) + Float32(1.0)) / Float32(0x1000001)


# =============================================================================
# Strategy 1: scalar baseline
#
# Mirrors minimax/kernels/router.mojo::sigmoid_topk_renorm. Kept here as a
# standalone reference so this file is self-testing without depending on the
# production kernel's internals.
# =============================================================================


def scalar_topk_renorm[num_experts: Int, k: Int](
    logits_ptr: UnsafePointer[Float32, MutAnyOrigin],
    correction_bias_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> TopKResult[k]:
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = num_experts // width
    comptime assert num_experts % width == 0

    var weights = InlineArray[Float32, num_experts](uninitialized=True)
    var scores = InlineArray[Float32, num_experts](uninitialized=True)
    var wp = UnsafePointer(to=weights[0])
    var sp = UnsafePointer(to=scores[0])

    for c in range(chunks):
        var logits = (logits_ptr + c * width).load[width=width]()
        var w = sigmoid_f32(logits)
        (wp + c * width).store(w)
        var bias = (correction_bias_ptr + c * width).load[width=width]()
        (sp + c * width).store(w + bias)

    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )
    for sel in range(k):
        var best_idx = 0
        var best_val = scores[0]
        for i in range(1, num_experts):
            if scores[i] > best_val:
                best_val = scores[i]
                best_idx = i
        result.indices[sel] = best_idx
        result.weights[sel] = weights[best_idx]
        scores[best_idx] = NEG_INF

    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv = Float32(1.0) / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv
    return result^


# =============================================================================
# Strategy 2: SIMD tournament with parallel (value, index) banks
#
# Phase A (across-register): log2(regs) butterfly stages. Each stage pairs
# registers stride-apart, computes a bool mask = values[a].gt(values[b]),
# uses it to select the winning value AND the winning index from the two
# registers. After the phase, reg[0] holds lane-wise per-register winners.
#
# Phase B (in-lane): log2(width) butterfly stages inside reg[0]. Uses the
# same shuffle pattern as fwht_apply — shuffle the register with a
# partner-index mask, compare, select on both value and index.
#
# After both phases: lane 0 of the value reg = global max, lane 0 of the
# index reg = its argmax. Mask out the winner lane on the persistent
# value reg (setting it to NEG_INF), then repeat k times.
# =============================================================================


def parallel_topk_renorm[num_experts: Int, k: Int](
    logits_ptr: UnsafePointer[Float32, MutAnyOrigin],
    correction_bias_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> TopKResult[k]:
    comptime width = simd_width_of[DType.float32]()
    comptime regs = num_experts // width
    comptime assert num_experts % width == 0

    var weights_buf = InlineArray[Float32, num_experts](uninitialized=True)
    var wp = UnsafePointer(to=weights_buf[0])

    var score_regs = InlineArray[SIMD[DType.float32, width], regs](
        fill=SIMD[DType.float32, width](0))
    var index_regs = InlineArray[SIMD[DType.int32, width], regs](
        fill=SIMD[DType.int32, width](0))
    fill_lane_iota[width, regs](index_regs)

    comptime for r in range(regs):
        var logits = (logits_ptr + r * width).load[width=width]()
        var w = sigmoid_f32(logits)
        (wp + r * width).store(w)
        var bias = (correction_bias_ptr + r * width).load[width=width]()
        score_regs[r] = w + bias

    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )

    var topk_values = InlineArray[Float32, k](fill=Float32(0))
    reduce_top_k[DType.float32, width, regs, k](
        score_regs, index_regs, NEG_INF, result.indices, topk_values)

    # Gather raw sigmoid weights at selected indices, renormalize to sum=1.
    for sel in range(k):
        result.weights[sel] = weights_buf[result.indices[sel]]
    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv = Float32(1.0) / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv
    return result^


# =============================================================================
# Strategy 3: SIMD tournament on packed u32 lanes
#
# Canonicalization (so f32 ordering matches UNSIGNED u32 ordering):
#   MSB set      (negative f32):  XOR with 0xFFFFFFFF  (flip all bits)
#   MSB clear    (positive f32):  XOR with 0x80000000  (set MSB)
# After this transform, the f32 ordering is preserved under unsigned-int
# compare across the whole real line. Concrete check:
#   +2.0 0x40000000 → 0xC0000000     -0.5 0xBF000000 → 0x40FFFFFF
#   +1.0 0x3F800000 → 0xBF800000     -1.0 0xBF800000 → 0x407FFFFF
#   +0.5 0x3F000000 → 0xBF000000     -2.0 0xC0000000 → 0x3FFFFFFF
# Positives land in [0x80..., 0xFF...], negatives in [0x00..., 0x7F...],
# and within each half the ordering matches the original f32.
#
# Pack: clear the low log2(num_experts) bits of canon, OR in the index.
# Costs log2(num_experts) bits of mantissa precision on the score.
# Tournament uses unsigned-integer max (vpmaxud on x86) — one op per pair,
# no parallel index register, no parallel select chain.
#
# Winner extract: low log2(num_experts) bits = argmax index.
# Mask: set that lane to 0 (smallest unsigned sentinel — safely below any
# reachable canon since canon always has at least one nonzero bit for
# sigmoid+bias scores).
# =============================================================================


def packed_topk_renorm[num_experts: Int, k: Int](
    logits_ptr: UnsafePointer[Float32, MutAnyOrigin],
    correction_bias_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> TopKResult[k]:
    comptime width = simd_width_of[DType.uint32]()
    comptime regs = num_experts // width
    comptime stages_across = log2[regs]()
    comptime stages_within = log2[width]()
    comptime index_bits = log2[num_experts]()
    comptime index_mask_scalar = UInt32((1 << index_bits) - 1)
    # clear_mask = ~INDEX_MASK in u32 = 0xFFFF_FF00 for index_bits=8
    comptime clear_mask_scalar = UInt32(0xFFFFFFFF) ^ index_mask_scalar
    comptime msb_mask_scalar = UInt32(0x80000000)
    comptime low31_mask_scalar = UInt32(0x7FFFFFFF)
    comptime assert num_experts % width == 0

    var weights_buf = InlineArray[Float32, num_experts](uninitialized=True)
    var wp = UnsafePointer(to=weights_buf[0])

    var lane_iota = SIMD[DType.uint32, width]()
    comptime for lane in range(width):
        lane_iota[lane] = UInt32(lane)

    var clear_vec = SIMD[DType.uint32, width](clear_mask_scalar)
    var msb_vec = SIMD[DType.uint32, width](msb_mask_scalar)
    var low31_vec = SIMD[DType.uint32, width](low31_mask_scalar)
    var zero_sentinel = SIMD[DType.uint32, width](0)

    # Build packed persistent bank.
    var packed_regs = InlineArray[SIMD[DType.uint32, width], regs](
        fill=SIMD[DType.uint32, width](0)
    )
    comptime for r in range(regs):
        var logits = (logits_ptr + r * width).load[width=width]()
        var w = sigmoid_f32(logits)
        (wp + r * width).store(w)
        var bias = (correction_bias_ptr + r * width).load[width=width]()
        var scores = w + bias
        var bits = scores.to_bits().cast[DType.uint32]()
        # sign_expand: 0xFFFF_FFFF if negative, 0 if positive (arithmetic right
        # shift via signed cast).
        var sign_expand = (bits.cast[DType.int32]() >> 31).cast[DType.uint32]()
        # xor_mask = 0x80000000 | (sign_expand & 0x7FFFFFFF)
        #          = negative → 0xFFFFFFFF, positive → 0x80000000
        var xor_mask = msb_vec | (sign_expand & low31_vec)
        var canon = bits ^ xor_mask
        var idx_vec = lane_iota + SIMD[DType.uint32, width](UInt32(r * width))
        packed_regs[r] = (canon & clear_vec) | idx_vec

    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )

    for sel in range(k):
        # Tournament workspace.
        var work = InlineArray[SIMD[DType.uint32, width], regs](
            fill=SIMD[DType.uint32, width](0)
        )
        comptime for r in range(regs):
            work[r] = packed_regs[r]

        # Phase A: across-register unsigned max.
        comptime for stage in range(stages_across):
            comptime stride = 1 << stage
            comptime groups = regs // (2 * stride)
            comptime for g in range(groups):
                comptime for j in range(stride):
                    comptime a = g * 2 * stride + j
                    comptime b = a + stride
                    work[a] = max(work[a], work[b])

        # Phase B: in-lane butterfly + unsigned max.
        comptime for stage in range(stages_within):
            comptime stride = 1 << stage
            comptime shuf_mask = butterfly_shuffle[width, stride]()
            var partner = work[0].shuffle[mask=shuf_mask](work[0])
            work[0] = max(work[0], partner)

        var winner_packed = UInt32(work[0][0])
        var winner_idx = Int(winner_packed & index_mask_scalar)
        result.indices[sel] = winner_idx
        result.weights[sel] = weights_buf[winner_idx]

        # Mask winner lane in persistent bank.
        var wr = winner_idx // width
        var wl = winner_idx - wr * width
        var lane_eq = lane_iota.eq(SIMD[DType.uint32, width](UInt32(wl)))
        packed_regs[wr] = lane_eq.select(zero_sentinel, packed_regs[wr])

    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv = Float32(1.0) / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv
    return result^


# =============================================================================
# Correctness test
# =============================================================================


def run_correctness_test():
    comptime TRIALS = 200
    var rng = Rng(seed=UInt64(42))

    var par_index_mismatches = 0
    var pak_index_mismatches = 0
    var par_weight_max_diff = Float32(0)
    var pak_weight_max_diff = Float32(0)
    var trials_with_par_any_mismatch = 0
    var trials_with_pak_any_mismatch = 0

    var logits_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var bias_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var lp = UnsafePointer(to=logits_buf[0])
    var bp = UnsafePointer(to=bias_buf[0])

    for trial in range(TRIALS):
        # logits in [-3, 3] → sigmoid in roughly (0.05, 0.95).
        # bias in [-0.5, 0.5] → scores span both signs, exercises canonicalization.
        for i in range(NUM_EXPERTS):
            lp[i] = (rng.uniform() - Float32(0.5)) * Float32(6.0)
            bp[i] = (rng.uniform() - Float32(0.5)) * Float32(1.0)

        var r_s = scalar_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        var r_p = parallel_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        var r_k = packed_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)

        var par_mis = 0
        var pak_mis = 0
        for sel in range(TOP_K):
            if r_s.indices[sel] != r_p.indices[sel]:
                par_index_mismatches += 1
                par_mis += 1
            if r_s.indices[sel] != r_k.indices[sel]:
                pak_index_mismatches += 1
                pak_mis += 1
            var dp = abs(r_s.weights[sel] - r_p.weights[sel])
            var dk = abs(r_s.weights[sel] - r_k.weights[sel])
            if dp > par_weight_max_diff:
                par_weight_max_diff = dp
            if dk > pak_weight_max_diff:
                pak_weight_max_diff = dk
        if par_mis > 0:
            trials_with_par_any_mismatch += 1
        if pak_mis > 0:
            trials_with_pak_any_mismatch += 1

    print("")
    print("=== CORRECTNESS ===")
    print("  trials:", TRIALS, " num_experts:", NUM_EXPERTS, " k:", TOP_K)
    print("  parallel vs scalar:")
    print("    index mismatches (per-slot):", par_index_mismatches, "/", TRIALS * TOP_K)
    print("    trials with any mismatch:   ", trials_with_par_any_mismatch, "/", TRIALS)
    print("    max weight |diff|:          ", par_weight_max_diff)
    print("  packed   vs scalar:")
    print("    index mismatches (per-slot):", pak_index_mismatches, "/", TRIALS * TOP_K)
    print("    trials with any mismatch:   ", trials_with_pak_any_mismatch, "/", TRIALS)
    print("    max weight |diff|:          ", pak_weight_max_diff)


# =============================================================================
# Benchmark
# =============================================================================


def time_scalar(
    lp: UnsafePointer[Float32, MutAnyOrigin],
    bp: UnsafePointer[Float32, MutAnyOrigin],
    iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = scalar_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        keep(r.indices[0])
        keep(r.weights[0])
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_parallel(
    lp: UnsafePointer[Float32, MutAnyOrigin],
    bp: UnsafePointer[Float32, MutAnyOrigin],
    iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = parallel_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        keep(r.indices[0])
        keep(r.weights[0])
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_packed(
    lp: UnsafePointer[Float32, MutAnyOrigin],
    bp: UnsafePointer[Float32, MutAnyOrigin],
    iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = packed_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        keep(r.indices[0])
        keep(r.weights[0])
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def run_benchmark():
    comptime WARMUP = 2000
    comptime TRIALS = 50000

    var rng = Rng(seed=UInt64(123))
    var logits_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var bias_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var lp = UnsafePointer(to=logits_buf[0])
    var bp = UnsafePointer(to=bias_buf[0])
    for i in range(NUM_EXPERTS):
        lp[i] = (rng.uniform() - Float32(0.5)) * Float32(6.0)
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(1.0)

    # Warmups for each (separately — avoid cache pollution leaking between).
    _ = time_scalar(lp, bp, WARMUP)
    _ = time_parallel(lp, bp, WARMUP)
    _ = time_packed(lp, bp, WARMUP)

    var ns_scalar = time_scalar(lp, bp, TRIALS)
    var ns_parallel = time_parallel(lp, bp, TRIALS)
    var ns_packed = time_packed(lp, bp, TRIALS)

    var per_scalar = ns_scalar // TRIALS
    var per_parallel = ns_parallel // TRIALS
    var per_packed = ns_packed // TRIALS

    print("")
    print("=== BENCHMARK ===")
    print("  iters:", TRIALS, " (warmup:", WARMUP, ")")
    print("  scalar   ns/call:", per_scalar)
    print("  parallel ns/call:", per_parallel)
    print("  packed   ns/call:", per_packed)
    if per_parallel > 0:
        print("  parallel speedup: ", Float32(per_scalar) / Float32(per_parallel), "x")
    if per_packed > 0:
        print("  packed   speedup: ", Float32(per_scalar) / Float32(per_packed), "x")


# =============================================================================
# Adversarial fuzzer — deliberately construct inputs likely to expose the
# packed strategy's precision loss and index-tie-break behavior.
#
# Six strategies, each run for many trials:
#   1. uniform           control: wide-range random logits + bias
#   2. tight_cluster     all scores in a narrow band (forces small gaps)
#   3. ulp_separated     top-k candidates separated by near-1-ulp in the
#                        canonicalized regime (~2^-15 at score scale 1)
#   4. exact_ties        multiple experts given identical scores — scalar
#                        tie-breaks to smallest index, packed to largest
#   5. saturated_sigmoid logits at ±large; sigmoid ≈ 0 or ≈ 1, tiny bias
#                        variations decide ranking
#   6. sign_boundary     scores straddling 0, stressing the canonicalization
#                        flip at the positive/negative boundary
#
# For each trial we record:
#   - whether parallel's top-k set differs from scalar's
#   - whether packed's top-k set differs from scalar's
#   - the minimum gap between consecutive top-k+1 scores (tells us how close
#     to a tie the boundary was — small gap means precision matters)
# =============================================================================


def _fill_uniform(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                  bp: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(NUM_EXPERTS):
        lp[i] = (rng.uniform() - Float32(0.5)) * Float32(6.0)
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(1.0)


def _fill_tight_cluster(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                        bp: UnsafePointer[Float32, MutAnyOrigin], eps: Float32):
    # logits large-negative → sigmoid ≈ 0; bias carries the score in a tiny band.
    for i in range(NUM_EXPERTS):
        lp[i] = Float32(-20.0)
        bp[i] = Float32(0.5) + (rng.uniform() - Float32(0.5)) * eps


def _fill_ulp_separated(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                        bp: UnsafePointer[Float32, MutAnyOrigin]):
    # Sigmoid ≈ 0, score = bias. Pack scores with gaps ≈ 2^-15 so that with 8
    # bits of mantissa dropped the top several may rank differently.
    for i in range(NUM_EXPERTS):
        lp[i] = Float32(-20.0)
    # Base noise, then inject an adversarial block of very close scores.
    for i in range(NUM_EXPERTS):
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(0.1)
    # Override 20 experts with nearly-equal scores separated by ~3e-5.
    var base = Float32(0.6)
    var step = Float32(3e-5)
    for i in range(20):
        var idx = Int(rng.next() % UInt64(NUM_EXPERTS))
        bp[idx] = base + Float32(i) * step


def _fill_exact_ties(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                     bp: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(NUM_EXPERTS):
        lp[i] = Float32(-20.0)
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(0.1)
    # Inject several exact ties at high scores.
    var base = Float32(0.9)
    for group in range(6):
        var tie_val = base - Float32(group) * Float32(0.001)
        for _ in range(3):
            var idx = Int(rng.next() % UInt64(NUM_EXPERTS))
            bp[idx] = tie_val


def _fill_saturated_sigmoid(mut rng: Rng,
                            lp: UnsafePointer[Float32, MutAnyOrigin],
                            bp: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(NUM_EXPERTS):
        # Large-magnitude logits → sigmoid saturates near 0 or 1.
        lp[i] = (rng.uniform() - Float32(0.5)) * Float32(40.0)
        # Tiny bias decides ranking among saturated experts.
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(1e-4)


def _fill_sign_boundary(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                        bp: UnsafePointer[Float32, MutAnyOrigin]):
    # Force scores to straddle 0: sigmoid ≈ 0.5 (logit near 0), bias ∈ [-0.5, 0.5]
    # → score ∈ [0, 1] or negative depending on bias sign. Many near-zero crossings.
    for i in range(NUM_EXPERTS):
        lp[i] = (rng.uniform() - Float32(0.5)) * Float32(0.5)
        bp[i] = (rng.uniform() - Float32(0.5)) * Float32(1.2)


def _compute_scores(lp: UnsafePointer[Float32, MutAnyOrigin],
                    bp: UnsafePointer[Float32, MutAnyOrigin],
                    mut scores: InlineArray[Float32, NUM_EXPERTS]):
    # Reference-compute the scalar scores to measure top-(k+1) min gap.
    comptime width = simd_width_of[DType.float32]()
    var sp = UnsafePointer(to=scores[0])
    var c = 0
    while c + width <= NUM_EXPERTS:
        var logits = (lp + c).load[width=width]()
        var w = sigmoid_f32(logits)
        var bias = (bp + c).load[width=width]()
        (sp + c).store(w + bias)
        c += width


def _top_k_plus_one_min_gap(scores_arr: InlineArray[Float32, NUM_EXPERTS]) -> Float32:
    # Returns min gap between consecutive ranked scores within the top (k+1).
    # Small gap = ambiguity likely.
    var buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    for i in range(NUM_EXPERTS):
        buf[i] = scores_arr[i]
    # Selection sort just the top K+1.
    var limit = TOP_K + 1
    for sel in range(limit):
        var best = sel
        for i in range(sel + 1, NUM_EXPERTS):
            if buf[i] > buf[best]:
                best = i
        var tmp = buf[sel]
        buf[sel] = buf[best]
        buf[best] = tmp
    var gap = Float32(1e30)
    for sel in range(limit - 1):
        var g = buf[sel] - buf[sel + 1]
        if g < gap:
            gap = g
    return gap


def _indices_as_set_match[k: Int](
    a: TopKResult[k], b: TopKResult[k],
) -> Bool:
    # Set-equivalence: do they select the same k experts (ignoring order)?
    for i in range(k):
        var found = False
        for j in range(k):
            if a.indices[i] == b.indices[j]:
                found = True
                break
        if not found:
            return False
    return True


def _run_strategy[fill: def (mut Rng, UnsafePointer[Float32, MutAnyOrigin],
                             UnsafePointer[Float32, MutAnyOrigin]) thin](
    name: String, trials: Int, seed: UInt64,
):
    var rng = Rng(seed=seed)
    var logits_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var bias_buf = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)
    var lp = UnsafePointer(to=logits_buf[0])
    var bp = UnsafePointer(to=bias_buf[0])
    var scores = InlineArray[Float32, NUM_EXPERTS](uninitialized=True)

    var par_set_mismatches = 0
    var pak_set_mismatches = 0
    var par_order_mismatches = 0
    var pak_order_mismatches = 0
    var gap_sum = Float32(0)
    var gap_min = Float32(1e30)

    for _ in range(trials):
        fill(rng, lp, bp)
        _compute_scores(lp, bp, scores)
        var gap = _top_k_plus_one_min_gap(scores)
        gap_sum += gap
        if gap < gap_min:
            gap_min = gap

        var r_s = scalar_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        var r_p = parallel_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)
        var r_k = packed_topk_renorm[NUM_EXPERTS, TOP_K](lp, bp)

        if not _indices_as_set_match[TOP_K](r_s, r_p):
            par_set_mismatches += 1
        if not _indices_as_set_match[TOP_K](r_s, r_k):
            pak_set_mismatches += 1
        for sel in range(TOP_K):
            if r_s.indices[sel] != r_p.indices[sel]:
                par_order_mismatches += 1
                break
        for sel in range(TOP_K):
            if r_s.indices[sel] != r_k.indices[sel]:
                pak_order_mismatches += 1
                break

    print("  [" + name + "]")
    print("    trials:", trials)
    print("    mean top-(k+1) min gap:", gap_sum / Float32(trials),
          " min observed:", gap_min)
    print("    parallel set-mismatches:   ", par_set_mismatches, "/", trials)
    print("    parallel order-mismatches: ", par_order_mismatches, "/", trials)
    print("    packed   set-mismatches:   ", pak_set_mismatches, "/", trials)
    print("    packed   order-mismatches: ", pak_order_mismatches, "/", trials)


# Strategy wrapper funcs (needed as separate named fns for the function
# parameter binding — Mojo doesn't accept closures with captured comptime
# context for free-function parameters of this shape).
def _s_uniform(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
               bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_uniform(rng, lp, bp)


def _s_tight(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
             bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_tight_cluster(rng, lp, bp, Float32(1e-3))


def _s_tight_tiny(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                  bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_tight_cluster(rng, lp, bp, Float32(1e-5))


def _s_ulp(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
           bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_ulp_separated(rng, lp, bp)


def _s_ties(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
            bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_exact_ties(rng, lp, bp)


def _s_saturated(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
                 bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_saturated_sigmoid(rng, lp, bp)


def _s_signb(mut rng: Rng, lp: UnsafePointer[Float32, MutAnyOrigin],
             bp: UnsafePointer[Float32, MutAnyOrigin]):
    _fill_sign_boundary(rng, lp, bp)


def run_adversarial_fuzz():
    print("")
    print("=== ADVERSARIAL FUZZ ===")
    print("  set-mismatch   = different unordered top-k experts")
    print("  order-mismatch = same set but at least one slot differs")
    print("")
    _run_strategy[_s_uniform](     String("uniform"),            2000, UInt64(1))
    _run_strategy[_s_tight](       String("tight_cluster_1e-3"), 2000, UInt64(2))
    _run_strategy[_s_tight_tiny](  String("tight_cluster_1e-5"), 2000, UInt64(3))
    _run_strategy[_s_ulp](         String("ulp_separated"),      2000, UInt64(4))
    _run_strategy[_s_ties](        String("exact_ties"),         2000, UInt64(5))
    _run_strategy[_s_saturated](   String("saturated_sigmoid"),  2000, UInt64(6))
    _run_strategy[_s_signb](       String("sign_boundary"),      2000, UInt64(7))


def main():
    run_correctness_test()
    run_adversarial_fuzz()
    run_benchmark()
