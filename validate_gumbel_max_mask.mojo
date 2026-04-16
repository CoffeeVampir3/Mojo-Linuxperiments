"""Scalar↔SIMD bit-exact equivalence at V values not divisible by SIMD width.

Exercises lm_head_flash_worker's last-batch padding mask. Motivated by GPT-2's
50,257-entry vocabulary (50257 % 8 = 1), which leaves 7 padded lanes in the
final SIMD batch on AVX2 hardware.

Per-trial fresh Philox: both paths create Random(seed, subsequence=trial) from
scratch, so they see the same u32 stream for the first V entries even though
SIMD consumes extra u32s for padded lanes (which then go to /dev/null).

Any disagreement implies the mask is broken.

Run: pixi run mojo -I . validate_gumbel_max_mask.mojo
"""

from std.sys.info import simd_width_of
from std.collections import List
from std.random.philox import Random

from experimental3.kernels.gemm import (
    softcap_simd, gumbel_simd, softcap_scalar, gumbel_from_u32,
)


def scalar_sample[V: Int](
    logits: List[Float32], cap: Float32,
    rng_seed: UInt64, subseq: UInt64,
) -> Int32:
    var rng = Random[rounds=10](
        seed=rng_seed, subsequence=subseq, offset=UInt64(0))
    var raw = rng.step()
    var tick = 0
    var best_score = Float32(-1.0e30)
    var best_idx = Int32(-1)
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
            best_idx = Int32(i)
    return best_idx


def simd_sample[V: Int, W: Int](
    logits: List[Float32], cap: Float32,
    rng_seed: UInt64, subseq: UInt64,
) -> Int32:
    comptime PHILOX_STEPS = W // 4
    var rng = Random[rounds=10](
        seed=rng_seed, subsequence=subseq, offset=UInt64(0))
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
    return cand_idx.reduce_min()


def make_uniform[V: Int]() -> List[Float32]:
    var l = List[Float32](capacity=V)
    for _ in range(V):
        l.append(Float32(0))
    return l^


def make_peaked[V: Int](idx: Int, peak: Float32) -> List[Float32]:
    var l = List[Float32](capacity=V)
    for _ in range(V):
        l.append(Float32(0))
    l[idx] = peak
    return l^


def run_test[V: Int, W: Int](
    label: String, logits: List[Float32], cap: Float32, n_trials: Int
) -> Int:
    var disagreements = 0
    var first_bad_trial = -1
    var first_bad_s = Int32(-1)
    var first_bad_v = Int32(-1)
    for trial in range(n_trials):
        var s = scalar_sample[V](
            logits, cap, UInt64(42), UInt64(trial))
        var v = simd_sample[V, W](
            logits, cap, UInt64(42), UInt64(trial))
        if s != v:
            disagreements += 1
            if first_bad_trial < 0:
                first_bad_trial = trial
                first_bad_s = s
                first_bad_v = v
    if disagreements == 0:
        print(
            "  ", label,
            "  V=", V, " V%W=", V % W,
            "  trials=", n_trials,
            "  disagreements=0")
    else:
        print(
            "  ", label,
            "  V=", V, " V%W=", V % W,
            "  trials=", n_trials,
            "  DISAGREEMENTS=", disagreements,
            "  first_bad trial=", first_bad_trial,
            " scalar=", first_bad_s,
            " simd=", first_bad_v)
    return disagreements


def test_V[V: Int](n_trials: Int) -> Int:
    comptime W = simd_width_of[DType.float32]()
    var cap = Float32(1.0e5)
    var total = 0
    total += run_test[V, W](
        "uniform       ", make_uniform[V](), cap, n_trials)
    total += run_test[V, W](
        "peak@0        ", make_peaked[V](0, Float32(3.0)), cap, n_trials)
    total += run_test[V, W](
        "peak@mid      ", make_peaked[V](V // 2, Float32(3.0)), cap, n_trials)
    total += run_test[V, W](
        "peak@last     ", make_peaked[V](V - 1, Float32(3.0)), cap, n_trials)
    total += run_test[V, W](
        "strong@last   ", make_peaked[V](V - 1, Float32(10.0)), cap, n_trials)
    return total


def main():
    comptime W = simd_width_of[DType.float32]()
    print("SIMD W=", W)
    print()

    var overall = 0

    print("--- V=33 (V%W=1: last batch has 1 valid + 7 padded) ---")
    overall += test_V[33](2000)
    print()

    print("--- V=39 (V%W=7: last batch has 7 valid + 1 padded) ---")
    overall += test_V[39](2000)
    print()

    print("--- V=47 (V%W=7: last batch has 7 valid + 1 padded, wider) ---")
    overall += test_V[47](2000)
    print()

    print("--- V=50257 (GPT-2 vocab, V%W=1) ---")
    overall += test_V[50257](200)
    print()

    print("---- summary ----")
    print("  total disagreements across all V:", overall)
    if overall == 0:
        print("  PASS: mask is bit-exact against scalar on non-W-aligned V.")
    else:
        print("  FAIL: mask produces divergent results.")
