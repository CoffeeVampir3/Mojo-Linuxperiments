"""Int8 channelwise quantization — distribution analysis.

Per-row absmax int8 quantization of weights and activations, followed by
integer-accumulated matmul with epilogue rescale, compared against f32.

Data generation:
  - Weights: N(0, 0.02), [N x K]
  - Activations: N(0, 1) with ~5% of channels scaled 30x, [M x K]

Measurements:
  - Per-row scale statistics
  - SQNR of quantized weights vs originals
  - SQNR of quantized activations vs originals, split by channel type
  - Quantized value utilization (max |qi|, zero fraction)
  - End-to-end matmul SQNR and error distribution

Matmul arithmetic:
    out[m,n] = (sum_k int(a_qi[m,k]) * int(w_qi[n,k])) * a_scale[m] * w_scale[n]

Usage: mojo design_analysis/int8ch.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import InlineArray


# ============================================================================
# Dimensions
# ============================================================================

comptime M = 64        # activation rows (tokens)
comptime K = 2048      # contraction dimension
comptime N = 2048      # output dimension (weight rows)


# ============================================================================
# PRNG — xorshift64 with CLT-based normal approximation
# ============================================================================


struct Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else UInt64(1)

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def uniform(mut self) -> Float32:
        """Uniform on [0, 1)."""
        return Float32(self.next() & 0xFFFFFF) / 16777216.0

    def normal(mut self) -> Float32:
        """Approximate N(0,1) via central limit theorem (sum of 12 uniforms - 6)."""
        var s = Float32(0)
        for _ in range(12):
            s += self.uniform()
        return s - 6.0


# ============================================================================
# Math helpers
# ============================================================================


@always_inline
def absf(x: Float32) -> Float32:
    return x if x >= 0 else -x


def log2f(x: Float32) -> Float32:
    """Fast log2 via IEEE 754 exponent + linear mantissa interpolation."""
    if x <= 0:
        return Float32(-200.0)
    var v = x
    var bits = UnsafePointer(to=v).bitcast[UInt32]()[]
    return Float32(Int((bits >> 23) & 0xFF) - 127) + Float32(
        bits & 0x7FFFFF
    ) / 8388608.0


def sqnr_db(signal: Float32, noise: Float32) -> Float32:
    """Signal-to-quantization-noise ratio in dB."""
    if noise <= 0:
        return Float32(999.0)
    if signal <= 0:
        return Float32(-999.0)
    # 10 * log10(s/n) = 10 / log2(10) * log2(s/n) ≈ 3.0103 * log2(s/n)
    return 3.01029995664 * log2f(signal / noise)


def round_nearest(x: Float32) -> Float32:
    """Round to nearest integer (ties away from zero)."""
    if x >= 0:
        return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))


def d2(x: Float32) -> String:
    """Format float with 2 decimal places."""
    var neg = x < 0
    var v = absf(x)
    var cents = Int(v * 100.0 + 0.5)
    var w = cents // 100
    var f = cents % 100
    var s = String(w) + "."
    if f < 10:
        s += "0"
    s += String(f)
    if neg:
        return "-" + s
    return s


# ============================================================================
# Histogram helpers
# ============================================================================


def bin_label(b: Int) -> String:
    if b == 0:
        return "  <1%  "
    if b == 1:
        return " 1-2%  "
    if b == 2:
        return " 2-5%  "
    if b == 3:
        return " 5-10% "
    if b == 4:
        return "10-20% "
    if b == 5:
        return "20-50% "
    if b == 6:
        return "50-100%"
    return " >100% "


def bar_str(count: Int, total: Int) -> String:
    if total == 0:
        return ""
    var n = Int(Float32(count) / Float32(total) * 50.0 + 0.5)
    var s = String("")
    for _ in range(n):
        s += "#"
    return s


# ============================================================================
# Channelwise int8 quantization
# ============================================================================


def channelwise_quantize(
    src: UnsafePointer[Float32, MutAnyOrigin],
    qi: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    """Per-row symmetric int8 quantization.

    scale[r] = max(|src[r, :]|) / 127
    qi[r, k] = clamp(round(src[r, k] / scale[r]), -128, 127)
    """
    for r in range(rows):
        var base = r * cols

        # Pass 1: per-row absmax
        var amax = Float32(0)
        for k in range(cols):
            var a = absf(src[base + k])
            if a > amax:
                amax = a

        var scale = amax / 127.0
        scales[r] = scale
        var inv = 1.0 / scale if scale > 0 else Float32(0)

        # Pass 2: quantize
        for k in range(cols):
            var v = round_nearest(src[base + k] * inv)
            if v > 127.0:
                v = 127.0
            elif v < -128.0:
                v = -128.0
            qi[base + k] = v


# ============================================================================
# Main
# ============================================================================


def main():
    var t_start = perf_counter_ns()

    print("=" * 70)
    print("  Int8 Channelwise Quantization — Distribution Analysis")
    print("=" * 70)
    print("  M=" + String(M) + "  K=" + String(K) + "  N=" + String(N))
    print("")

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    var weights = alloc[Float32](N * K)
    var act = alloc[Float32](M * K)
    var w_qi = alloc[Float32](N * K)  # quantized weights (integer values as float)
    var a_qi = alloc[Float32](M * K)  # quantized activations
    var w_scales = alloc[Float32](N)
    var a_scales = alloc[Float32](M)
    var out_ref = alloc[Float32](M * N)  # f32 reference output
    var out_q = alloc[Float32](M * N)  # int8-rescaled output
    var is_outlier = alloc[UInt8](K)  # per-column outlier flag

    var rng = Rng(seed=42)

    # ------------------------------------------------------------------
    # Generate weights: N(0, 0.02)
    # ------------------------------------------------------------------

    for i in range(N * K):
        weights[i] = rng.normal() * 0.02

    # ------------------------------------------------------------------
    # Generate activations: N(0, 1) with ~5% outlier channels at 30x
    # ------------------------------------------------------------------

    var outlier_mag = Float32(30.0)
    var target_outliers = K * 5 // 100

    for k in range(K):
        is_outlier[k] = UInt8(0)

    # Place outlier channels randomly
    var placed = 0
    while placed < target_outliers:
        var idx = Int(rng.next() % UInt64(K))
        if is_outlier[idx] == UInt8(0):
            is_outlier[idx] = UInt8(1)
            placed += 1

    var num_outliers = 0
    for k in range(K):
        if is_outlier[k] != UInt8(0):
            num_outliers += 1

    for m in range(M):
        for k in range(K):
            var v = rng.normal()
            if is_outlier[k] != UInt8(0):
                v *= outlier_mag
            act[m * K + k] = v

    print("  Weights:     N(0, 0.02)")
    print(
        "  Activations: N(0, 1) + "
        + String(num_outliers)
        + "/"
        + String(K)
        + " outlier channels at "
        + d2(outlier_mag)
        + "x"
    )
    print("")

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    var tq0 = perf_counter_ns()
    channelwise_quantize(weights, w_qi, w_scales, N, K)
    var tq1 = perf_counter_ns()
    channelwise_quantize(act, a_qi, a_scales, M, K)
    var tq2 = perf_counter_ns()

    # ------------------------------------------------------------------
    # Reference f32 matmul: out[m,n] = sum_k act[m,k] * weights[n,k]
    # (weights stored as [N, K] — one row per output neuron)
    # ------------------------------------------------------------------

    var tm0 = perf_counter_ns()
    for m in range(M):
        for n in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += act[m * K + k] * weights[n * K + k]
            out_ref[m * N + n] = acc
    var tm1 = perf_counter_ns()

    # ------------------------------------------------------------------
    # Int8 matmul: integer accumulation + epilogue rescale
    #   acc_i32 = sum_k qi_a[m,k] * qi_w[n,k]      (integer multiply-accumulate)
    #   out[m,n] = float(acc_i32) * a_scale[m] * w_scale[n]   (epilogue)
    # ------------------------------------------------------------------

    for m in range(M):
        for n in range(N):
            var acc = 0
            for k in range(K):
                acc += Int(a_qi[m * K + k]) * Int(w_qi[n * K + k])
            out_q[m * N + n] = Float32(acc) * a_scales[m] * w_scales[n]
    var tm2 = perf_counter_ns()

    # ==================================================================
    # ANALYSIS
    # ==================================================================

    # --- weight quantization quality -----------------------------------
    print("--- Weight Quantization ---")

    var ws_min = w_scales[0]
    var ws_max = w_scales[0]
    for i in range(N):
        if w_scales[i] < ws_min:
            ws_min = w_scales[i]
        if w_scales[i] > ws_max:
            ws_max = w_scales[i]
    print(
        "  Scale range: ["
        + d2(ws_min * 10000)
        + ", "
        + d2(ws_max * 10000)
        + "] x 1e-4"
    )
    print("  Scale max/min: " + d2(ws_max / ws_min))

    var w_sig = Float32(0)
    var w_noise = Float32(0)
    var w_worst = Float32(999.0)
    var w_best = Float32(-999.0)
    for r in range(N):
        var rs = Float32(0)
        var rn = Float32(0)
        for k in range(K):
            var orig = weights[r * K + k]
            var deq = w_qi[r * K + k] * w_scales[r]
            var err = orig - deq
            rs += orig * orig
            rn += err * err
        w_sig += rs
        w_noise += rn
        var db = sqnr_db(rs, rn)
        if db < w_worst:
            w_worst = db
        if db > w_best:
            w_best = db

    print("  SQNR overall: " + d2(sqnr_db(w_sig, w_noise)) + " dB")
    print(
        "  SQNR per-row: " + d2(w_worst) + " (worst) .. " + d2(w_best) + " (best) dB"
    )
    print("")

    # --- activation quantization quality --------------------------------
    print("--- Activation Quantization ---")

    var as_min = a_scales[0]
    var as_max = a_scales[0]
    for i in range(M):
        if a_scales[i] < as_min:
            as_min = a_scales[i]
        if a_scales[i] > as_max:
            as_max = a_scales[i]
    print("  Scale range: [" + d2(as_min) + ", " + d2(as_max) + "]")
    print("  Scale max/min: " + d2(as_max / as_min))

    var a_sig = Float32(0)
    var a_noise = Float32(0)
    for m in range(M):
        for k in range(K):
            var orig = act[m * K + k]
            var deq = a_qi[m * K + k] * a_scales[m]
            var err = orig - deq
            a_sig += orig * orig
            a_noise += err * err

    print("  SQNR overall: " + d2(sqnr_db(a_sig, a_noise)) + " dB")
    print("")

    # --- per-channel breakdown ------------------------------------------
    print("--- Per-Channel Breakdown ---")

    var norm_sig = Float32(0)
    var norm_noise = Float32(0)
    var out_sig = Float32(0)
    var out_noise = Float32(0)
    var norm_max_qi = Float32(0)
    var out_max_qi = Float32(0)
    var norm_zeros = 0
    var norm_total = 0

    for m in range(M):
        for k in range(K):
            var orig = act[m * K + k]
            var deq = a_qi[m * K + k] * a_scales[m]
            var err = orig - deq
            var aq = absf(a_qi[m * K + k])
            if is_outlier[k] != UInt8(0):
                out_sig += orig * orig
                out_noise += err * err
                if aq > out_max_qi:
                    out_max_qi = aq
            else:
                norm_sig += orig * orig
                norm_noise += err * err
                if aq > norm_max_qi:
                    norm_max_qi = aq
                norm_total += 1
                if aq < 0.5:
                    norm_zeros += 1

    var eff_bits = log2f(
        2.0 * norm_max_qi + 1.0
    ) if norm_max_qi >= 1 else Float32(0)
    var zero_pct = Float32(norm_zeros) / Float32(max(norm_total, 1)) * 100.0

    print("  1x channels (" + String(K - num_outliers) + "):")
    print("    SQNR            = " + d2(sqnr_db(norm_sig, norm_noise)) + " dB")
    print("    Max |qi|         = " + d2(norm_max_qi) + " / 127")
    print("    Effective bits   = " + d2(eff_bits) + " / 8")
    print("    qi=0 fraction    = " + d2(zero_pct) + "%")
    print("")
    print("  " + d2(outlier_mag) + "x channels (" + String(num_outliers) + "):")
    print("    SQNR            = " + d2(sqnr_db(out_sig, out_noise)) + " dB")
    print("    Max |qi|         = " + d2(out_max_qi) + " / 127")
    print("")

    # --- matmul comparison ----------------------------------------------
    print("--- End-to-End Matmul ---")

    var m_sig = Float32(0)
    var m_noise = Float32(0)
    var max_ae = Float32(0)
    var sum_ae = Float32(0)

    for i in range(M * N):
        var expected = out_ref[i]
        var err = expected - out_q[i]
        m_sig += expected * expected
        m_noise += err * err
        var ae = absf(err)
        if ae > max_ae:
            max_ae = ae
        sum_ae += ae

    var rms_out = sqrt(m_sig / Float32(M * N))
    var rms_err = sqrt(m_noise / Float32(M * N))
    var mean_ae = sum_ae / Float32(M * N)

    print("  Reference RMS output  = " + d2(rms_out))
    print("  SQNR                  = " + d2(sqnr_db(m_sig, m_noise)) + " dB")
    print("  RMS error             = " + d2(rms_err))
    print("  Max absolute error    = " + d2(max_ae))
    print("  Mean absolute error   = " + d2(mean_ae))
    print("  RMS error / RMS out   = " + d2(rms_err / rms_out * 100) + "%")
    print("")

    # --- error histogram ------------------------------------------------
    print("--- Relative Error Distribution ---")

    var bins = InlineArray[Int, 8](fill=0)
    var counted = 0

    for i in range(M * N):
        var ra = absf(out_ref[i])
        if ra < 0.001:
            continue
        var re = absf(out_ref[i] - out_q[i]) / ra
        counted += 1
        if re < 0.01:
            bins[0] += 1
        elif re < 0.02:
            bins[1] += 1
        elif re < 0.05:
            bins[2] += 1
        elif re < 0.10:
            bins[3] += 1
        elif re < 0.20:
            bins[4] += 1
        elif re < 0.50:
            bins[5] += 1
        elif re < 1.00:
            bins[6] += 1
        else:
            bins[7] += 1

    for b in range(8):
        var p = Float32(bins[b]) / Float32(max(counted, 1)) * 100.0
        print(
            "  "
            + bin_label(b)
            + "  "
            + d2(p)
            + "%  "
            + bar_str(bins[b], counted)
        )
    print("")

    # --- timing ---------------------------------------------------------
    print("--- Timing ---")
    print(
        "  Quantize weights: " + String(Int(tq1 - tq0) // 1000) + " us"
    )
    print(
        "  Quantize acts:    " + String(Int(tq2 - tq1) // 1000) + " us"
    )
    print(
        "  Ref f32 matmul:   " + String(Int(tm1 - tm0) // 1000) + " us"
    )
    print(
        "  Int8 matmul:      " + String(Int(tm2 - tm1) // 1000) + " us"
    )
    print("")

    # --- summary --------------------------------------------------------
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(
        "  Weight SQNR:              "
        + d2(sqnr_db(w_sig, w_noise))
        + " dB"
    )
    print(
        "  Activation SQNR:          "
        + d2(sqnr_db(a_sig, a_noise))
        + " dB"
    )
    print(
        "    1x channels:            "
        + d2(sqnr_db(norm_sig, norm_noise))
        + " dB"
    )
    print(
        "    " + d2(outlier_mag) + "x channels:           "
        + d2(sqnr_db(out_sig, out_noise))
        + " dB"
    )
    print(
        "  Matmul SQNR:              "
        + d2(sqnr_db(m_sig, m_noise))
        + " dB"
    )
    print("")
    print(
        "  1x channel max |qi|:      "
        + d2(norm_max_qi)
        + " / 127"
    )
    print(
        "  1x channel eff. bits:     "
        + d2(eff_bits)
        + " / 8"
    )
    print(
        "  1x channel qi=0 frac:     "
        + d2(zero_pct)
        + "%"
    )

    # cleanup
    weights.free()
    act.free()
    w_qi.free()
    a_qi.free()
    w_scales.free()
    a_scales.free()
    out_ref.free()
    out_q.free()
    is_outlier.free()
