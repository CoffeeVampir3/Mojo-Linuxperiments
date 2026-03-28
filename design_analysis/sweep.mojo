"""Distribution sensitivity: channelwise vs Hadamard int8 across 4 cases.

Same weights, same int8 pipeline. Only the activation outlier severity varies.

Case A: Uniform     — all channels N(0, 1)
Case B: Mild        — 5% of channels at 5x
Case C: Severe      — 5% of channels at 100x
Case D: Extreme     — 0.5% of channels at 1000x

Outlier channels are evenly spaced across K to distribute across
Hadamard blocks. Weights are N(0, 0.02) and shared across all cases.

Usage: mojo design_analysis/sweep.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import InlineArray


comptime M = 64
comptime K = 2048
comptime N = 2048
comptime BLOCK_SIZE = 512
comptime NC = 4


# ============================================================================
# PRNG
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
        return Float32(self.next() & 0xFFFFFF) / 16777216.0

    def normal(mut self) -> Float32:
        var s = Float32(0)
        for _ in range(12):
            s += self.uniform()
        return s - 6.0


# ============================================================================
# Helpers
# ============================================================================


@always_inline
def absf(x: Float32) -> Float32:
    return x if x >= 0 else -x


def log2f(x: Float32) -> Float32:
    if x <= 0:
        return Float32(-200.0)
    var v = x
    var bits = UnsafePointer(to=v).bitcast[UInt32]()[]
    return Float32(Int((bits >> 23) & 0xFF) - 127) + Float32(
        bits & 0x7FFFFF
    ) / 8388608.0


def sqnr_db(signal: Float32, noise: Float32) -> Float32:
    if noise <= 0:
        return Float32(999.0)
    if signal <= 0:
        return Float32(-999.0)
    return 3.01029995664 * log2f(signal / noise)


def round_nearest(x: Float32) -> Float32:
    if x >= 0:
        return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))


def d2(x: Float32) -> String:
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


def col(x: Float32) -> String:
    """Format value into a 11-char table column."""
    var s = d2(x)
    while len(s) < 11:
        s += " "
    return s


def col_s(x: Float32) -> String:
    """Signed column: +X.XX or -X.XX, 11 chars."""
    var s = String("")
    if x >= 0:
        s = "+" + d2(x)
    else:
        s = d2(x)
    while len(s) < 11:
        s += " "
    return s


def case_desc(c: Int) -> String:
    if c == 0:
        return "Uniform"
    if c == 1:
        return "5% @ 5x"
    if c == 2:
        return "5% @ 100x"
    return "0.5% @ 1000x"


# ============================================================================
# FWHT
# ============================================================================


def fwht_inplace(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var half = 1
    while half < n:
        var i = 0
        while i < n:
            for j in range(half):
                var a = buf[i + j]
                var b = buf[i + j + half]
                buf[i + j] = a + b
                buf[i + j + half] = a - b
            i += half * 2
        half *= 2
    var scale = 1.0 / sqrt(Float32(n))
    for i in range(n):
        buf[i] = buf[i] * scale


def fwht_rows(
    mat: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    block_size: Int,
):
    var num_blocks = cols // block_size
    for r in range(rows):
        var row_base = mat + r * cols
        for b in range(num_blocks):
            fwht_inplace(row_base + b * block_size, block_size)


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
    for r in range(rows):
        var base = r * cols
        var amax = Float32(0)
        for k in range(cols):
            var a = absf(src[base + k])
            if a > amax:
                amax = a
        var scale = amax / 127.0
        scales[r] = scale
        var inv = 1.0 / scale if scale > 0 else Float32(0)
        for k in range(cols):
            var v = round_nearest(src[base + k] * inv)
            if v > 127.0:
                v = 127.0
            elif v < -128.0:
                v = -128.0
            qi[base + k] = v


def int8_matmul(
    a_qi: UnsafePointer[Float32, MutAnyOrigin],
    w_qi: UnsafePointer[Float32, MutAnyOrigin],
    a_scales: UnsafePointer[Float32, MutAnyOrigin],
    w_scales: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    m_dim: Int,
    n_dim: Int,
    k_dim: Int,
):
    for m in range(m_dim):
        for n in range(n_dim):
            var acc = 0
            for k in range(k_dim):
                acc += Int(a_qi[m * k_dim + k]) * Int(w_qi[n * k_dim + k])
            dst[m * n_dim + n] = Float32(acc) * a_scales[m] * w_scales[n]


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("  Int8 Quantization: Distribution Sensitivity")
    print("=" * 70)
    print(
        "  M="
        + String(M)
        + "  K="
        + String(K)
        + "  N="
        + String(N)
        + "  block="
        + String(BLOCK_SIZE)
    )
    print("  Weights: N(0, 0.02)")
    print("")

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    var weights = alloc[Float32](N * K)
    var act_base = alloc[Float32](M * K)
    var act = alloc[Float32](M * K)
    var w_rot = alloc[Float32](N * K)
    var a_rot = alloc[Float32](M * K)
    var w_qi = alloc[Float32](N * K)
    var a_qi = alloc[Float32](M * K)
    var w_scales = alloc[Float32](N)
    var a_scales = alloc[Float32](M)
    var out_ref = alloc[Float32](M * N)
    var out_test = alloc[Float32](M * N)
    var is_outlier = alloc[UInt8](K)

    # ------------------------------------------------------------------
    # Generate shared data
    # ------------------------------------------------------------------

    var rng = Rng(seed=42)
    for i in range(N * K):
        weights[i] = rng.normal() * 0.02
    for i in range(M * K):
        act_base[i] = rng.normal()

    # ------------------------------------------------------------------
    # Case parameters
    # ------------------------------------------------------------------

    var case_n = InlineArray[Int, NC](fill=0)
    case_n[0] = 0
    case_n[1] = K * 5 // 100  # 102
    case_n[2] = K * 5 // 100  # 102
    case_n[3] = K * 5 // 1000  # 10

    var case_mag = InlineArray[Float32, NC](fill=1.0)
    case_mag[0] = 1.0
    case_mag[1] = 5.0
    case_mag[2] = 100.0
    case_mag[3] = 1000.0

    print("  Case A: Uniform       0 outliers")
    print(
        "  Case B: Mild          "
        + String(case_n[1])
        + " channels (5%) at 5x"
    )
    print(
        "  Case C: Severe        "
        + String(case_n[2])
        + " channels (5%) at 100x"
    )
    print(
        "  Case D: Extreme       "
        + String(case_n[3])
        + " channels (0.5%) at 1000x"
    )
    print("")

    # ------------------------------------------------------------------
    # Result storage
    # ------------------------------------------------------------------

    var ch_act_db = InlineArray[Float32, NC](fill=0)
    var ch_mat_db = InlineArray[Float32, NC](fill=0)
    var ch_eff = InlineArray[Float32, NC](fill=0)
    var ch_zpct = InlineArray[Float32, NC](fill=0)
    var had_act_db = InlineArray[Float32, NC](fill=0)
    var had_mat_db = InlineArray[Float32, NC](fill=0)
    var had_eff = InlineArray[Float32, NC](fill=0)
    var had_zpct = InlineArray[Float32, NC](fill=0)
    var ratio_pre = InlineArray[Float32, NC](fill=0)
    var ratio_post = InlineArray[Float32, NC](fill=0)

    # ------------------------------------------------------------------
    # Run cases
    # ------------------------------------------------------------------

    print("  Running...")

    for c in range(NC):
        var tc0 = perf_counter_ns()
        var n_out = case_n[c]
        var mag_c = case_mag[c]

        # Mark outlier channels (evenly spaced across K)
        for k in range(K):
            is_outlier[k] = UInt8(0)
        if n_out > 0:
            var stride = K // n_out
            for i in range(n_out):
                is_outlier[i * stride] = UInt8(1)

        # Build activations: copy base, scale outlier channels
        for i in range(M * K):
            act[i] = act_base[i]
        if n_out > 0:
            for m in range(M):
                for k in range(K):
                    if is_outlier[k] != UInt8(0):
                        act[m * K + k] = act[m * K + k] * mag_c

        # Column RMS ratio (before rotation)
        var pre_min = Float32(1e30)
        var pre_max = Float32(0)
        for k in range(K):
            var ss = Float32(0)
            for m in range(M):
                var v = act[m * K + k]
                ss += v * v
            var rms = sqrt(ss / Float32(M))
            if rms > 0:
                if rms < pre_min:
                    pre_min = rms
                if rms > pre_max:
                    pre_max = rms
        ratio_pre[c] = pre_max / pre_min

        # F32 reference matmul
        for m in range(M):
            for n in range(N):
                var acc = Float32(0)
                for k in range(K):
                    acc += act[m * K + k] * weights[n * K + k]
                out_ref[m * N + n] = acc

        # === CHANNELWISE PATH ===
        channelwise_quantize(weights, w_qi, w_scales, N, K)
        channelwise_quantize(act, a_qi, a_scales, M, K)
        int8_matmul(a_qi, w_qi, a_scales, w_scales, out_test, M, N, K)

        var ca_s = Float32(0)
        var ca_n = Float32(0)
        var c_mqi = Float32(0)
        var c_zeros = 0
        var c_total = 0
        for m in range(M):
            for k in range(K):
                var orig = act[m * K + k]
                var deq = a_qi[m * K + k] * a_scales[m]
                var err = orig - deq
                ca_s += orig * orig
                ca_n += err * err
                if is_outlier[k] == UInt8(0):
                    var aq = absf(a_qi[m * K + k])
                    if aq > c_mqi:
                        c_mqi = aq
                    c_total += 1
                    if aq < 0.5:
                        c_zeros += 1
        ch_act_db[c] = sqnr_db(ca_s, ca_n)
        ch_eff[c] = log2f(
            2.0 * c_mqi + 1.0
        ) if c_mqi >= 1 else Float32(0)
        ch_zpct[c] = Float32(c_zeros) / Float32(
            max(c_total, 1)
        ) * 100.0

        var cm_s = Float32(0)
        var cm_n = Float32(0)
        for i in range(M * N):
            var expected = out_ref[i]
            var err = expected - out_test[i]
            cm_s += expected * expected
            cm_n += err * err
        ch_mat_db[c] = sqnr_db(cm_s, cm_n)

        # === HADAMARD PATH ===
        for i in range(N * K):
            w_rot[i] = weights[i]
        for i in range(M * K):
            a_rot[i] = act[i]
        fwht_rows(w_rot, N, K, BLOCK_SIZE)
        fwht_rows(a_rot, M, K, BLOCK_SIZE)

        # Column RMS ratio (after rotation)
        var post_min = Float32(1e30)
        var post_max = Float32(0)
        for k in range(K):
            var ss = Float32(0)
            for m in range(M):
                var v = a_rot[m * K + k]
                ss += v * v
            var rms = sqrt(ss / Float32(M))
            if rms > 0:
                if rms < post_min:
                    post_min = rms
                if rms > post_max:
                    post_max = rms
        ratio_post[c] = post_max / post_min

        channelwise_quantize(w_rot, w_qi, w_scales, N, K)
        channelwise_quantize(a_rot, a_qi, a_scales, M, K)
        int8_matmul(a_qi, w_qi, a_scales, w_scales, out_test, M, N, K)

        var ha_s = Float32(0)
        var ha_n = Float32(0)
        var h_mqi = Float32(0)
        var h_zeros = 0
        for m in range(M):
            for k in range(K):
                var orig = a_rot[m * K + k]
                var deq = a_qi[m * K + k] * a_scales[m]
                var err = orig - deq
                ha_s += orig * orig
                ha_n += err * err
                var aq = absf(a_qi[m * K + k])
                if aq > h_mqi:
                    h_mqi = aq
                if aq < 0.5:
                    h_zeros += 1
        had_act_db[c] = sqnr_db(ha_s, ha_n)
        had_eff[c] = log2f(
            2.0 * h_mqi + 1.0
        ) if h_mqi >= 1 else Float32(0)
        had_zpct[c] = Float32(h_zeros) / Float32(M * K) * 100.0

        var hm_s = Float32(0)
        var hm_n = Float32(0)
        for i in range(M * N):
            var expected = out_ref[i]
            var err = expected - out_test[i]
            hm_s += expected * expected
            hm_n += err * err
        had_mat_db[c] = sqnr_db(hm_s, hm_n)

        var ms = Int(perf_counter_ns() - tc0) // 1_000_000
        print("    Case " + case_desc(c) + ": " + String(ms) + " ms")

    # ==================================================================
    # RESULTS TABLE
    # ==================================================================

    print("")
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print("")
    print(
        "                          A          B          C          D"
    )
    print(
        "                          Uniform    5%@5x      5%@100x    .5%@1000x"
    )

    # Column RMS ratio
    print("")
    print("  Column RMS ratio")
    var s = "    Before rotation     "
    for c in range(NC):
        s += col(ratio_pre[c])
    print(s)
    s = "    After rotation      "
    for c in range(NC):
        s += col(ratio_post[c])
    print(s)

    # Activation SQNR
    print("")
    print("  Activation SQNR (dB)")
    s = "    Channelwise         "
    for c in range(NC):
        s += col(ch_act_db[c])
    print(s)
    s = "    Hadamard            "
    for c in range(NC):
        s += col(had_act_db[c])
    print(s)
    s = "    Delta               "
    for c in range(NC):
        s += col_s(had_act_db[c] - ch_act_db[c])
    print(s)

    # Matmul SQNR
    print("")
    print("  Matmul SQNR (dB)")
    s = "    Channelwise         "
    for c in range(NC):
        s += col(ch_mat_db[c])
    print(s)
    s = "    Hadamard            "
    for c in range(NC):
        s += col(had_mat_db[c])
    print(s)
    s = "    Delta               "
    for c in range(NC):
        s += col_s(had_mat_db[c] - ch_mat_db[c])
    print(s)

    # Effective bits
    print("")
    print("  Effective bits (/8)")
    s = "    Channelwise         "
    for c in range(NC):
        s += col(ch_eff[c])
    print(s)
    s = "    Hadamard            "
    for c in range(NC):
        s += col(had_eff[c])
    print(s)

    # qi=0 fraction
    print("")
    print("  qi=0 fraction (%)")
    s = "    Channelwise         "
    for c in range(NC):
        s += col(ch_zpct[c])
    print(s)
    s = "    Hadamard            "
    for c in range(NC):
        s += col(had_zpct[c])
    print(s)

    print("")

    # cleanup
    weights.free()
    act_base.free()
    act.free()
    w_rot.free()
    a_rot.free()
    w_qi.free()
    a_qi.free()
    w_scales.free()
    a_scales.free()
    out_ref.free()
    out_test.free()
    is_outlier.free()
