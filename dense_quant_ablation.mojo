"""Ablation: FWHT vs channelwise i8 quantization under varying conditions.

Reproduces the ~1.7-2.2x NRMSE advantage of FWHT seen on real Gemma4 dense
FFN activations, then sweeps over strategies to find viable TP approaches.

Calibrated against dense_ch_vs_fwht.mojo real-weight measurements:
  - Intermediate dim: 2112 (33 * 64)
  - Real peak/rms: 10-40 (heavy outliers from gelu_tanh*up)
  - Real NRMSE ratio (ch/fwht): 1.6-2.2x

Run: pixi run mojo build -I . dense_quant_ablation.mojo && ./dense_quant_ablation
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.math import log, abs
from std.collections import InlineArray

from simd_math import sqrt as simd_sqrt, roundeven, sincos, exp_f32
from experimental3.kernels.fwht import fwht_block


comptime INTERMEDIATE = 2112
comptime HIDDEN = 2816
comptime NUM_TRIALS = 256
comptime WEIGHT_ROWS = 64


# ── PRNG ─────────────────────────────────────────────────────────────

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

    @always_inline
    def normal(mut self) -> Float32:
        var u1 = self.uniform()
        var u2 = self.uniform()
        var r = simd_sqrt(Float32(-2.0) * log(u1))
        var sc = sincos[1](SIMD[DType.float64, 1](Float64(u2) * Float64(6.2831853)))
        return r * Float32(sc.cos_val)


# ── Activation generators ────────────────────────────────────────────

def fill_gaussian(mut rng: Rng, buf: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    for i in range(n):
        buf[i] = rng.normal()

def fill_outlier_gaussian(mut rng: Rng, buf: UnsafePointer[Float32, MutAnyOrigin],
    n: Int, outlier_frac: Float32, outlier_scale: Float32):
    """Gaussian with outlier_frac of channels scaled up by outlier_scale.

    Produces peak/rms ratios similar to real gelu_tanh*up activations.
    """
    for i in range(n):
        var v = rng.normal()
        if rng.uniform() < outlier_frac:
            v *= outlier_scale
        buf[i] = v


# ── Quantization + FWHT ─────────────────────────────────────────────

def absmax_quantize_row(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cols: Int,
) -> Float32:
    var amax = Float32(0)
    for k in range(cols):
        var v = abs(src[k])
        if v > amax:
            amax = v
    if amax < Float32(1e-10):
        amax = Float32(1e-10)
    var inv = Float32(127.0) / amax
    for k in range(cols):
        dst[k] = roundeven(src[k] * inv).cast[DType.int8]()
    return amax / Float32(127.0)


def absmax_quantize_block[blk: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    for b in range(n // blk):
        var off = b * blk
        var amax = Float32(0)
        for k in range(blk):
            var v = abs(src[off + k])
            if v > amax:
                amax = v
        if amax < Float32(1e-10):
            amax = Float32(1e-10)
        scales[b] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax
        for k in range(blk):
            dst[off + k] = roundeven(src[off + k] * inv).cast[DType.int8]()


def fwht_rotate[blk: Int](buf: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    for b in range(n // blk):
        fwht_block[blk](buf + b * blk)


def dot_i8_blocked[blk: Int](
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: UnsafePointer[Float32, MutAnyOrigin],
    w_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    w_sc: Float32,
    cols: Int,
) -> Float32:
    """One row dot: w_sc * sum_blk(act_sc[b] * sum_k(w[k]*a[k]))."""
    var acc = Float32(0)
    for b in range(cols // blk):
        var off = b * blk
        var blk_acc = Float32(0)
        for k in range(blk):
            blk_acc += Float32(Int(w_i8[off + k])) * Float32(Int(act_i8[off + k]))
        acc += blk_acc * act_sc[b]
    return acc * w_sc


# ── Error measurement ────────────────────────────────────────────────

@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable):
    var nrmse_sum: Float64
    var cos_sum: Float64
    var count: Int

    @staticmethod
    def zero() -> Stats:
        return Stats(Float64(0), Float64(0), 0)

    def avg_nrmse(self) -> Float64:
        if self.count == 0:
            return Float64(0)
        return self.nrmse_sum / Float64(self.count)

    def avg_cos(self) -> Float64:
        if self.count == 0:
            return Float64(0)
        return self.cos_sum / Float64(self.count)


def measure_nrmse(
    ref_buf: UnsafePointer[Float32, MutAnyOrigin],
    test: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float64:
    var sum_sq_err = Float64(0)
    var sum_sq_ref = Float64(0)
    for i in range(n):
        var e = Float64(test[i]) - Float64(ref_buf[i])
        sum_sq_err += e * e
        sum_sq_ref += Float64(ref_buf[i]) * Float64(ref_buf[i])
    if sum_sq_ref < Float64(1e-30):
        return Float64(0)
    return Float64(simd_sqrt(Float32(sum_sq_err / sum_sq_ref)))


# ── Core comparison: FWHT with independent quant block ───────────────

def run_trial_fq[fwht_blk: Int, quant_blk: Int](
    act: UnsafePointer[Float32, MutAnyOrigin],
    w_rows: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: UnsafePointer[Float32, MutAnyOrigin],
    w_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    y_ref: UnsafePointer[Float32, MutAnyOrigin],
    y_test: UnsafePointer[Float32, MutAnyOrigin],
    num_rows: Int,
) -> Float64:
    """FWHT at fwht_blk, per-block quantize at quant_blk, dot at quant_blk.

    quant_blk must be a multiple of fwht_blk. FWHT spreads energy within
    fwht_blk, then a coarser quant_blk scale covers multiple FWHT blocks.
    """
    # f32 reference
    for n in range(num_rows):
        var acc = Float32(0)
        for k in range(INTERMEDIATE):
            acc += w_rows[n * INTERMEDIATE + k] * act[k]
        y_ref[n] = acc

    # FWHT-rotate activation, then quantize at quant_blk granularity
    memcpy(dest=work, src=act, count=INTERMEDIATE)
    fwht_rotate[fwht_blk](work, INTERMEDIATE)
    absmax_quantize_block[quant_blk](work, act_i8, act_sc, INTERMEDIATE)

    # FWHT-rotate each weight row (same fwht_blk), per-row quantize, dot
    for n in range(num_rows):
        memcpy(dest=work, src=w_rows + n * INTERMEDIATE, count=INTERMEDIATE)
        fwht_rotate[fwht_blk](work, INTERMEDIATE)
        var w_sc = absmax_quantize_row(work, w_i8, INTERMEDIATE)
        y_test[n] = dot_i8_blocked[quant_blk](act_i8, act_sc, w_i8, w_sc, INTERMEDIATE)

    return measure_nrmse(y_ref, y_test, num_rows)


def run_fq_sweep[fwht_blk: Int, quant_blk: Int](
    label: String,
    outlier_frac: Float32,
    outlier_scale: Float32,
    mut rng: Rng,
    act: UnsafePointer[Float32, MutAnyOrigin],
    w_rows: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: UnsafePointer[Float32, MutAnyOrigin],
    w_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    y_ref: UnsafePointer[Float32, MutAnyOrigin],
    y_test: UnsafePointer[Float32, MutAnyOrigin],
):
    var nrmse_sum = Float64(0)
    for trial in range(NUM_TRIALS):
        fill_outlier_gaussian(rng, act, INTERMEDIATE, outlier_frac, outlier_scale)
        for i in range(WEIGHT_ROWS * INTERMEDIATE):
            w_rows[i] = rng.normal() / simd_sqrt(Float32(INTERMEDIATE))
        nrmse_sum += run_trial_fq[fwht_blk, quant_blk](
            act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test, WEIGHT_ROWS)
    var avg = nrmse_sum / Float64(NUM_TRIALS)
    print("    " + label + ": NRMSE=" + String(avg))


# ── Core comparison: one trial of FWHT vs channelwise ────────────────

def run_trial[act_blk: Int, w_fwht_blk: Int](
    act: UnsafePointer[Float32, MutAnyOrigin],
    w_rows: UnsafePointer[Float32, MutAnyOrigin],
    # scratch
    work: UnsafePointer[Float32, MutAnyOrigin],
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: UnsafePointer[Float32, MutAnyOrigin],
    w_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    y_ref: UnsafePointer[Float32, MutAnyOrigin],
    y_test: UnsafePointer[Float32, MutAnyOrigin],
    num_rows: Int,
) -> InlineArray[Float64, 2]:
    """Returns [nrmse_fwht, nrmse_ch]."""
    # f32 reference: y_ref[n] = sum_k w[n,k] * act[k]
    for n in range(num_rows):
        var acc = Float32(0)
        for k in range(INTERMEDIATE):
            acc += w_rows[n * INTERMEDIATE + k] * act[k]
        y_ref[n] = acc

    # --- FWHT path ---
    # Rotate + quantize activation
    memcpy(dest=work, src=act, count=INTERMEDIATE)
    fwht_rotate[act_blk](work, INTERMEDIATE)
    absmax_quantize_block[act_blk](work, act_i8, act_sc, INTERMEDIATE)

    # Rotate + quantize each weight row, compute dot
    for n in range(num_rows):
        memcpy(dest=work, src=w_rows + n * INTERMEDIATE, count=INTERMEDIATE)
        fwht_rotate[w_fwht_blk](work, INTERMEDIATE)
        var w_sc = absmax_quantize_row(work, w_i8, INTERMEDIATE)
        y_test[n] = dot_i8_blocked[act_blk](act_i8, act_sc, w_i8, w_sc, INTERMEDIATE)
    var nrmse_fwht = measure_nrmse(y_ref, y_test, num_rows)

    # --- Channelwise path ---
    absmax_quantize_block[act_blk](act, act_i8, act_sc, INTERMEDIATE)
    for n in range(num_rows):
        var w_sc = absmax_quantize_row(w_rows + n * INTERMEDIATE, w_i8, INTERMEDIATE)
        y_test[n] = dot_i8_blocked[act_blk](act_i8, act_sc, w_i8, w_sc, INTERMEDIATE)
    var nrmse_ch = measure_nrmse(y_ref, y_test, num_rows)

    var result = InlineArray[Float64, 2](fill=Float64(0))
    result[0] = nrmse_fwht
    result[1] = nrmse_ch
    return result


# ── Sweep runner ─────────────────────────────────────────────────────

def run_sweep[act_blk: Int, w_fwht_blk: Int](
    label: String,
    outlier_frac: Float32,
    outlier_scale: Float32,
    mut rng: Rng,
    # scratch
    act: UnsafePointer[Float32, MutAnyOrigin],
    w_rows: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: UnsafePointer[Float32, MutAnyOrigin],
    w_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    y_ref: UnsafePointer[Float32, MutAnyOrigin],
    y_test: UnsafePointer[Float32, MutAnyOrigin],
):
    var fwht_sum = Float64(0)
    var ch_sum = Float64(0)
    var peak_rms_sum = Float64(0)

    for trial in range(NUM_TRIALS):
        fill_outlier_gaussian(rng, act, INTERMEDIATE, outlier_frac, outlier_scale)

        # Measure peak/rms
        var amax = Float32(0)
        var ss = Float64(0)
        for i in range(INTERMEDIATE):
            var v = abs(act[i])
            if v > amax:
                amax = v
            ss += Float64(act[i]) * Float64(act[i])
        var rms_val = Float64(simd_sqrt(Float32(ss / Float64(INTERMEDIATE))))
        if rms_val > Float64(1e-10):
            peak_rms_sum += Float64(amax) / rms_val

        # Generate weight rows
        for i in range(WEIGHT_ROWS * INTERMEDIATE):
            w_rows[i] = rng.normal() / simd_sqrt(Float32(INTERMEDIATE))

        var result = run_trial[act_blk, w_fwht_blk](
            act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test, WEIGHT_ROWS)
        fwht_sum += result[0]
        ch_sum += result[1]

    var avg_fwht = fwht_sum / Float64(NUM_TRIALS)
    var avg_ch = ch_sum / Float64(NUM_TRIALS)
    var ratio = avg_ch / avg_fwht if avg_fwht > Float64(1e-10) else Float64(0)
    var avg_peak_rms = peak_rms_sum / Float64(NUM_TRIALS)

    print("  " + label)
    print("    act_blk=" + String(act_blk) + " w_fwht_blk=" + String(w_fwht_blk)
        + " peak/rms=" + String(avg_peak_rms))
    print("    FWHT NRMSE=" + String(avg_fwht)
        + " CH NRMSE=" + String(avg_ch)
        + " ratio=" + String(ratio))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=== Dense FFN Quantization Ablation ===")
    print("INTERMEDIATE=" + String(INTERMEDIATE) + " trials=" + String(NUM_TRIALS)
        + " weight_rows=" + String(WEIGHT_ROWS))
    print()

    var rng = Rng(0xDEADBEEF42)

    # Allocate scratch
    var act = alloc[Float32](INTERMEDIATE)
    var w_rows = alloc[Float32](WEIGHT_ROWS * INTERMEDIATE)
    var work = alloc[Float32](INTERMEDIATE)
    var act_i8 = alloc[Scalar[DType.int8]](INTERMEDIATE)
    var act_sc = alloc[Float32](INTERMEDIATE // 16 + 1)  # max blocks
    var w_i8 = alloc[Scalar[DType.int8]](INTERMEDIATE)
    var y_ref = alloc[Float32](WEIGHT_ROWS)
    var y_test = alloc[Float32](WEIGHT_ROWS)

    # -----------------------------------------------------------------
    # 1. Calibration: match real Gemma4 distribution
    # -----------------------------------------------------------------
    print("--- 1. Calibration vs real data ---")
    print("  (real baseline: peak/rms=10-40, NRMSE ratio=1.6-2.2)")
    print()

    # Mild outliers (early layers, peak/rms ~12)
    run_sweep[64, 64]("mild outliers (frac=0.02, scale=8)",
        Float32(0.02), Float32(8.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # Moderate outliers (mid layers, peak/rms ~20)
    run_sweep[64, 64]("moderate outliers (frac=0.01, scale=15)",
        Float32(0.01), Float32(15.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # Heavy outliers (deep layers, peak/rms ~35)
    run_sweep[64, 64]("heavy outliers (frac=0.005, scale=30)",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()

    # -----------------------------------------------------------------
    # 2. Strategy: decouple GEMV tile from FWHT block
    #    GEMV tiles at 32 or 64 (VNNI-compatible), FWHT at smaller blocks
    # -----------------------------------------------------------------
    print("--- 2. Decoupled GEMV tile vs FWHT block ---")
    print("  (activation quantized per act_blk, weight FWHT'd at w_fwht_blk)")
    print()

    # Moderate outliers for all strategy tests
    comptime frac = Float32(0.01)
    comptime scale = Float32(15.0)

    # Baseline: both at 64
    run_sweep[64, 64]("baseline: act_blk=64, w_fwht=64",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # FWHT at 32, quantize blocks at 32
    run_sweep[32, 32]("act_blk=32, w_fwht=32",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # FWHT at 16, quantize blocks at 16
    run_sweep[16, 16]("act_blk=16, w_fwht=16",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # Decoupled: FWHT weight at 64, activation quantize at 16
    run_sweep[16, 64]("DECOUPLED: act_blk=16, w_fwht=64",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # Decoupled: FWHT weight at 32, activation quantize at 16
    run_sweep[16, 32]("DECOUPLED: act_blk=16, w_fwht=32",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()

    # -----------------------------------------------------------------
    # 3. TP-relevant: K=528 (2112/4) — what actually divides cleanly?
    # -----------------------------------------------------------------
    print("--- 3. Block sizes that divide 528 (=INTERMEDIATE/4) ---")
    print("  528 = 16 * 33 — divisible by 16 but not 32 or 64")
    print()

    # These represent what's possible per-rank at tp=4
    # Block=16 is the largest power-of-2 that divides 528
    run_sweep[16, 16]("tp=4 viable: act_blk=16, w_fwht=16",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # What if we pad to 576 (=64*9) and use block=64?
    # (simulated here at full 2112 — padding zeros don't affect FWHT result
    #  because 2112 already divides by 64)
    run_sweep[64, 64]("tp=4 padded to 576: act_blk=64, w_fwht=64",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # Decoupled: weight FWHT at 16 (divides 528), act quantize at 16
    # This is the "keep FWHT but use small blocks" approach
    run_sweep[16, 16]("tp=4 small FWHT: act_blk=16, w_fwht=16",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()

    # -----------------------------------------------------------------
    # 4. Sensitivity: how much does FWHT block size matter?
    # -----------------------------------------------------------------
    print("--- 4. FWHT block size sweep (matched act+weight) ---")
    print()

    run_sweep[64, 64]("blk=64", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_sweep[32, 32]("blk=32", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_sweep[16, 16]("blk=16", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_sweep[8, 8]("blk=8", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_sweep[4, 4]("blk=4", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_sweep[2, 2]("blk=2", frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # -----------------------------------------------------------------
    # 5. FWHT=16 fixed, sweep quant block size
    # -----------------------------------------------------------------
    print("--- 5. FWHT=16 fixed, increasing quant block size ---")
    print("  (can we recover quality lost from smaller FWHT by using")
    print("   finer or coarser quant granularity?)")
    print()

    print("  moderate outliers (frac=0.01, scale=15):")
    run_fq_sweep[16, 16]("fwht=16 quant=16",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 32]("fwht=16 quant=32",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 48]("fwht=16 quant=48",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 64]("fwht=16 quant=64",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 96]("fwht=16 quant=96",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 192]("fwht=16 quant=192",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 528]("fwht=16 quant=528 (per-row at tp=4)",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()
    print("  reference points:")
    run_fq_sweep[64, 64]("fwht=64 quant=64 (current tp=1 baseline)",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    # channelwise reference at same quant blocks
    print()
    print("  channelwise (no FWHT) reference at same quant sizes:")
    run_fq_sweep[2, 16]("no-fwht quant=16 (fwht=2 ≈ identity)",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[2, 64]("no-fwht quant=64 (fwht=2 ≈ identity)",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[2, 528]("no-fwht quant=528 (fwht=2 ≈ identity)",
        frac, scale, rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()

    # -----------------------------------------------------------------
    # 6. Heavy outliers version of the same sweep
    # -----------------------------------------------------------------
    print("--- 6. Same sweep, heavy outliers (frac=0.005, scale=30) ---")
    print()

    run_fq_sweep[16, 16]("fwht=16 quant=16",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 32]("fwht=16 quant=32",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 48]("fwht=16 quant=48",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 64]("fwht=16 quant=64",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 192]("fwht=16 quant=192",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    run_fq_sweep[16, 528]("fwht=16 quant=528 (per-row at tp=4)",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)
    print()
    print("  reference:")
    run_fq_sweep[64, 64]("fwht=64 quant=64 (current tp=1 baseline)",
        Float32(0.005), Float32(30.0), rng,
        act, w_rows, work, act_i8, act_sc, w_i8, y_ref, y_test)

    print()
    print("=== done ===")

    act.free()
    w_rows.free()
    work.free()
    act_i8.free()
    act_sc.free()
    w_i8.free()
    y_ref.free()
    y_test.free()
