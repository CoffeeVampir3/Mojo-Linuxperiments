"""Chi-squared goodness-of-fit + scalar↔SIMD bit-exactness for lm_head_flash.

Each test runs two samplers in parallel on fresh RNGs with the same seed:
  - scalar path: gumbel_from_u32 + scalar argmax
  - SIMD path  : mirrors lm_head_flash_worker batching/reduce/horizontal-pick

Reports χ² for both (agreement with softmax) and the per-bin count delta
(agreement with each other). Bit-exact equivalence confirms the worker's
SIMD batching and horizontal reduce logic is correct.

Run: pixi run mojo -I . validate_gumbel_max.mojo
"""

from std.math import exp as ref_exp
from std.sys.info import simd_width_of
from std.collections import List
from std.random.philox import Random

from experimental3.kernels.gemm import (
    gumbel_from_u32, softcap_scalar, softcap_simd, gumbel_simd,
)


comptime V = 32
comptime N_TRIALS = 200000
comptime DF = V - 1

# Chi-squared critical values for df = V - 1 = 31.
comptime CRIT_99 = 52.19
comptime CRIT_999 = 58.30


def softcap_f64(x: Float64, cap: Float64) -> Float64:
    # Matches the sigmoid-based tanh used by softcap_simd: tanh(u) = 2 * sigmoid(2u) - 1.
    var u = Float64(2.0) * x / cap
    var e = ref_exp(-u)
    return (Float64(2.0) / (Float64(1.0) + e) - Float64(1.0)) * cap


def softmax_f64(logits: List[Float32], cap: Float32) -> List[Float64]:
    var capped = List[Float64](capacity=V)
    for i in range(V):
        capped.append(softcap_f64(Float64(logits[i]), Float64(cap)))

    var max_l = capped[0]
    for i in range(1, V):
        if capped[i] > max_l:
            max_l = capped[i]

    var exps = List[Float64](capacity=V)
    var z_sum = Float64(0)
    for i in range(V):
        var e = ref_exp(capped[i] - max_l)
        exps.append(e)
        z_sum += e

    var probs = List[Float64](capacity=V)
    for i in range(V):
        probs.append(exps[i] / z_sum)
    return probs^


def sample_histogram_scalar(
    logits: List[Float32], cap: Float32, seed: UInt64
) -> List[Int]:
    var counts = List[Int](capacity=V)
    for _ in range(V):
        counts.append(0)

    var rng = Random[rounds=10](
        seed=seed, subsequence=UInt64(0), offset=UInt64(0))
    var raw = rng.step()
    var tick = 0

    for _ in range(N_TRIALS):
        var best_score = Float32(-1.0e30)
        var best_idx = 0
        for i in range(V):
            if tick == 4:
                raw = rng.step()
                tick = 0
            var capped = softcap_scalar(logits[i], cap)
            var g = gumbel_from_u32(raw[tick])
            tick += 1
            var score = capped + g
            if score > best_score:
                best_score = score
                best_idx = i
        counts[best_idx] += 1

    return counts^


def sample_histogram_simd(
    logits: List[Float32], cap: Float32, seed: UInt64
) -> List[Int]:
    """Mirrors lm_head_flash_worker: W-batched SIMD epilogue + horizontal reduce."""
    comptime W = simd_width_of[DType.float32]()
    comptime PHILOX_STEPS = W // 4

    var counts = List[Int](capacity=V)
    for _ in range(V):
        counts.append(0)

    var rng = Random[rounds=10](
        seed=seed, subsequence=UInt64(0), offset=UInt64(0))

    for _ in range(N_TRIALS):
        var running_best = SIMD[DType.float32, W](Float32(-1.0e30))
        var running_idx = SIMD[DType.int32, W](-1)

        var batch = 0
        while batch < V:
            var batch_logits = SIMD[DType.float32, W](Float32(-1.0e30))
            var batch_indices = SIMD[DType.int32, W](-1)

            comptime for lane in range(W):
                var i = batch + lane
                if i < V:
                    batch_logits[lane] = logits[i]
                    batch_indices[lane] = Int32(i)

            var capped = softcap_simd[W](batch_logits, cap)

            var raw_W = SIMD[DType.uint32, W](0)
            comptime for i in range(PHILOX_STEPS):
                raw_W = raw_W.insert[offset=i * 4](rng.step())

            var g = gumbel_simd[W](raw_W)
            var scores = capped + g

            var valid = batch_indices.ge(SIMD[DType.int32, W](0))
            var take = scores.gt(running_best) & valid
            running_best = take.select(scores, running_best)
            running_idx = take.select(batch_indices, running_idx)

            batch += W

        var best_val = running_best.reduce_max()
        var is_best = running_best.eq(SIMD[DType.float32, W](best_val))
        var cand_idx = is_best.select(
            running_idx, SIMD[DType.int32, W](Int32.MAX))
        var best_idx = Int(cand_idx.reduce_min())
        counts[best_idx] += 1

    return counts^


def chi_squared(counts: List[Int], probs: List[Float64]) -> Float64:
    var chi2 = Float64(0)
    for i in range(V):
        var expected = Float64(N_TRIALS) * probs[i]
        if expected < Float64(1.0):
            continue
        var diff = Float64(counts[i]) - expected
        chi2 += diff * diff / expected
    return chi2


def run_test(
    label: String, logits: List[Float32], cap: Float32, seed: UInt64
) -> Tuple[Float64, Float64, Int]:
    var probs = softmax_f64(logits, cap)
    var counts_s = sample_histogram_scalar(logits, cap, seed)
    var counts_v = sample_histogram_simd(logits, cap, seed)

    var chi2_s = chi_squared(counts_s, probs)
    var chi2_v = chi_squared(counts_v, probs)

    var total_abs_diff = 0
    for i in range(V):
        var d = counts_s[i] - counts_v[i]
        if d < 0:
            d = -d
        total_abs_diff += d

    print(
        "  ", label, " seed=", seed,
        " χ²scalar=", chi2_s,
        " χ²simd=", chi2_v,
        " |Δcount|=", total_abs_diff)
    return (chi2_s, chi2_v, total_abs_diff)


def make_uniform() -> List[Float32]:
    var l = List[Float32](capacity=V)
    for _ in range(V):
        l.append(Float32(0))
    return l^


def make_peaked(idx: Int, peak: Float32) -> List[Float32]:
    var l = List[Float32](capacity=V)
    for _ in range(V):
        l.append(Float32(0))
    l[idx] = peak
    return l^


def make_bimodal(i1: Int, v1: Float32, i2: Int, v2: Float32) -> List[Float32]:
    var l = List[Float32](capacity=V)
    for _ in range(V):
        l.append(Float32(0))
    l[i1] = v1
    l[i2] = v2
    return l^


def make_gradient(slope: Float32) -> List[Float32]:
    var l = List[Float32](capacity=V)
    for i in range(V):
        l.append(Float32(i) * slope)
    return l^


def main():
    comptime W = simd_width_of[DType.float32]()
    print("V=", V, " N_TRIALS=", N_TRIALS, " df=", DF, " SIMD W=", W)
    print("χ²(df=31) critical: 99%=", CRIT_99, " 99.9%=", CRIT_999)
    print()

    var max_chi2_s = Float64(0)
    var max_chi2_v = Float64(0)
    var max_abs_diff = 0
    var num_runs = 0

    print("---- uniform (flat), no softcap ----")
    var uniform_l = make_uniform()
    for s in range(5):
        var r = run_test("uniform       ", uniform_l, Float32(1.0e5), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- peaked @ 10, peak=3.0 ----")
    var peaked_l = make_peaked(10, Float32(3.0))
    for s in range(5):
        var r = run_test("peaked@10     ", peaked_l, Float32(1.0e5), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- bimodal @ 5 (v=2.0) and 20 (v=1.5) ----")
    var bimodal_l = make_bimodal(5, Float32(2.0), 20, Float32(1.5))
    for s in range(5):
        var r = run_test("bimodal       ", bimodal_l, Float32(1.0e5), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- gradient slope=0.2 ----")
    var gradient_l = make_gradient(Float32(0.2))
    for s in range(5):
        var r = run_test("gradient      ", gradient_l, Float32(1.0e5), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- peaked @ 15, peak=4.0 ----")
    var strong_l = make_peaked(15, Float32(4.0))
    for s in range(5):
        var r = run_test("strong@15     ", strong_l, Float32(1.0e5), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- softcapped gradient (slope=0.3, cap=2.0) ----")
    var raw_grad = make_gradient(Float32(0.3))
    for s in range(5):
        var r = run_test("softcap_grad  ", raw_grad, Float32(2.0), UInt64(s))
        if r[0] > max_chi2_s: max_chi2_s = r[0]
        if r[1] > max_chi2_v: max_chi2_v = r[1]
        if r[2] > max_abs_diff: max_abs_diff = r[2]
        num_runs += 1

    print()
    print("---- summary ----")
    print("  runs=", num_runs)
    print("  max χ² scalar=", max_chi2_s)
    print("  max χ² simd  =", max_chi2_v)
    print("  max |Δcount| between paths=", max_abs_diff,
          "  (0 = bit-identical histograms)")
    # With 30 runs under H0, expect ~0.3 exceedances of the 99% critical value,
    # so a single one is normal. Hard-fail only at 99.9%.
    if max_abs_diff != 0:
        print("  FAIL: SIMD histograms differ from scalar.")
    elif max_chi2_v >= CRIT_999:
        print("  FAIL: SIMD max χ² exceeds 99.9% envelope.")
    elif max_chi2_v >= CRIT_99:
        print("  PASS: SIMD bit-exact to scalar; one χ² in (99%, 99.9%) — expected under H0.")
    else:
        print("  PASS: SIMD bit-exact to scalar; all χ² within 99% envelope.")
