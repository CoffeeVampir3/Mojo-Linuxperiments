"""Hadamard-rotated int8 quantization — comparison with channelwise int8.

Same data and same int8 pipeline as int8ch.mojo. The only difference is a
block-diagonal Fast Walsh-Hadamard Transform applied to weights and
activations before quantization.

Single-sided rotation on K:
    W' = H_block(W)  per row, offline
    x~ = H_block(x)  per row, online
    y  = x~ * W'^T = x * H^T * H * W^T = x * W^T   (H cancels)

Both paths use identical int8 quantization and matmul:
    qi = clamp(round(value / scale), -128, 127)
    out[m,n] = (sum_k int(a_qi[m,k]) * int(w_qi[n,k])) * a_scale[m] * w_scale[n]

Usage: mojo design_analysis/hadamard.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import InlineArray


# ============================================================================
# Dimensions
# ============================================================================

comptime M = 64
comptime K = 2048
comptime N = 2048
comptime BLOCK_SIZE = 512


# ============================================================================
# PRNG (identical to int8ch.mojo — same seed produces same data)
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
# Math helpers
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
# Fast Walsh-Hadamard Transform
# ============================================================================


def fwht_inplace(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """In-place normalized FWHT. Computes (1/sqrt(n)) * H * x.

    H is the Walsh-Hadamard matrix with entries +/-1.
    The normalized transform is orthonormal and self-inverse:
        H_norm * H_norm = I
    so forward and inverse transforms are the same operation.

    n must be a power of 2. Cost: n * log2(n) additions.
    """
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
    """Block-diagonal FWHT applied to each row of [rows, cols].

    Each row is partitioned into cols/block_size blocks, and fwht_inplace
    is applied to each block independently. This implements:
        H_block = diag(H_n, H_n, ..., H_n)
    where n = block_size and there are cols/block_size blocks.
    """
    var num_blocks = cols // block_size
    for r in range(rows):
        var row_base = mat + r * cols
        for b in range(num_blocks):
            fwht_inplace(row_base + b * block_size, block_size)


# ============================================================================
# Channelwise int8 quantization (identical to int8ch.mojo)
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


# ============================================================================
# Int8 matmul
# ============================================================================


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
    """dst[m,n] = float(sum_k int(a[m,k]) * int(w[n,k])) * a_scale[m] * w_scale[n]"""
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
    var t_start = perf_counter_ns()

    print("=" * 70)
    print("  Hadamard + Int8 vs Channelwise Int8")
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
    print("")

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    var weights = alloc[Float32](N * K)
    var act = alloc[Float32](M * K)
    var w_rot = alloc[Float32](N * K)
    var a_rot = alloc[Float32](M * K)
    var w_qi = alloc[Float32](N * K)
    var a_qi = alloc[Float32](M * K)
    var w_scales = alloc[Float32](N)
    var a_scales = alloc[Float32](M)
    var out_ref = alloc[Float32](M * N)
    var out_ch = alloc[Float32](M * N)
    var out_had = alloc[Float32](M * N)
    var is_outlier = alloc[UInt8](K)

    var rng = Rng(seed=42)

    # ------------------------------------------------------------------
    # Generate data (same seed / same procedure as int8ch.mojo)
    # ------------------------------------------------------------------

    for i in range(N * K):
        weights[i] = rng.normal() * 0.02

    var outlier_mag = Float32(30.0)
    var target_outliers = K * 5 // 100
    for k in range(K):
        is_outlier[k] = UInt8(0)
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
        + " channels at "
        + d2(outlier_mag)
        + "x"
    )
    print("")

    # ------------------------------------------------------------------
    # F32 reference matmul (computed once, shared by both paths)
    # ------------------------------------------------------------------

    var tm0 = perf_counter_ns()
    for m in range(M):
        for n in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += act[m * K + k] * weights[n * K + k]
            out_ref[m * N + n] = acc
    var tm1 = perf_counter_ns()

    # ==================================================================
    # PATH A: CHANNELWISE INT8 (no rotation)
    # ==================================================================

    var tch0 = perf_counter_ns()
    channelwise_quantize(weights, w_qi, w_scales, N, K)
    channelwise_quantize(act, a_qi, a_scales, M, K)
    int8_matmul(a_qi, w_qi, a_scales, w_scales, out_ch, M, N, K)
    var tch1 = perf_counter_ns()

    # Collect channelwise metrics
    var ch_w_sig = Float32(0)
    var ch_w_noise = Float32(0)
    for r in range(N):
        for k in range(K):
            var orig = weights[r * K + k]
            var deq = w_qi[r * K + k] * w_scales[r]
            var err = orig - deq
            ch_w_sig += orig * orig
            ch_w_noise += err * err

    var ch_a_sig = Float32(0)
    var ch_a_noise = Float32(0)
    var ch_1x_sig = Float32(0)
    var ch_1x_noise = Float32(0)
    var ch_1x_max_qi = Float32(0)
    var ch_1x_zeros = 0
    var ch_1x_total = 0
    var ch_as_min = a_scales[0]
    var ch_as_max = a_scales[0]
    for i in range(M):
        if a_scales[i] < ch_as_min:
            ch_as_min = a_scales[i]
        if a_scales[i] > ch_as_max:
            ch_as_max = a_scales[i]
    for m in range(M):
        for k in range(K):
            var orig = act[m * K + k]
            var deq = a_qi[m * K + k] * a_scales[m]
            var err = orig - deq
            ch_a_sig += orig * orig
            ch_a_noise += err * err
            if is_outlier[k] == UInt8(0):
                ch_1x_sig += orig * orig
                ch_1x_noise += err * err
                var aq = absf(a_qi[m * K + k])
                if aq > ch_1x_max_qi:
                    ch_1x_max_qi = aq
                ch_1x_total += 1
                if aq < 0.5:
                    ch_1x_zeros += 1

    var ch_1x_eff = log2f(
        2.0 * ch_1x_max_qi + 1.0
    ) if ch_1x_max_qi >= 1 else Float32(0)
    var ch_1x_zpct = Float32(ch_1x_zeros) / Float32(
        max(ch_1x_total, 1)
    ) * 100.0

    var ch_m_sig = Float32(0)
    var ch_m_noise = Float32(0)
    for i in range(M * N):
        var expected = out_ref[i]
        var err = expected - out_ch[i]
        ch_m_sig += expected * expected
        ch_m_noise += err * err

    # ==================================================================
    # HADAMARD ROTATION
    # ==================================================================

    for i in range(N * K):
        w_rot[i] = weights[i]
    for i in range(M * K):
        a_rot[i] = act[i]

    var th0 = perf_counter_ns()
    fwht_rows(w_rot, N, K, BLOCK_SIZE)
    fwht_rows(a_rot, M, K, BLOCK_SIZE)
    var th1 = perf_counter_ns()

    # ==================================================================
    # DISTRIBUTION ANALYSIS
    # ==================================================================

    print("--- Activation Column RMS: Before Rotation ---")
    var pre_min = Float32(1e30)
    var pre_max = Float32(0)
    for k in range(K):
        var ss = Float32(0)
        for m in range(M):
            var v = act[m * K + k]
            ss += v * v
        var rms = sqrt(ss / Float32(M))
        if rms < pre_min:
            pre_min = rms
        if rms > pre_max:
            pre_max = rms
    print(
        "  min="
        + d2(pre_min)
        + "  max="
        + d2(pre_max)
        + "  ratio="
        + d2(pre_max / pre_min)
    )
    print("")

    print(
        "--- Activation Column RMS: After Rotation (block="
        + String(BLOCK_SIZE)
        + ") ---"
    )
    var post_min = Float32(1e30)
    var post_max = Float32(0)
    for k in range(K):
        var ss = Float32(0)
        for m in range(M):
            var v = a_rot[m * K + k]
            ss += v * v
        var rms = sqrt(ss / Float32(M))
        if rms < post_min:
            post_min = rms
        if rms > post_max:
            post_max = rms
    print(
        "  min="
        + d2(post_min)
        + "  max="
        + d2(post_max)
        + "  ratio="
        + d2(post_max / post_min)
    )

    # Norm preservation
    var norm_orig = Float32(0)
    var norm_rot = Float32(0)
    for i in range(M * K):
        norm_orig += act[i] * act[i]
        norm_rot += a_rot[i] * a_rot[i]
    print(
        "  Norm: ||rot||/||orig|| = "
        + d2(sqrt(norm_rot) / sqrt(norm_orig))
    )
    print("")

    # ==================================================================
    # PATH B: HADAMARD + INT8
    # ==================================================================

    var thad0 = perf_counter_ns()
    channelwise_quantize(w_rot, w_qi, w_scales, N, K)
    channelwise_quantize(a_rot, a_qi, a_scales, M, K)
    int8_matmul(a_qi, w_qi, a_scales, w_scales, out_had, M, N, K)
    var thad1 = perf_counter_ns()

    # Hadamard weight quantization SQNR
    var had_w_sig = Float32(0)
    var had_w_noise = Float32(0)
    for r in range(N):
        for k in range(K):
            var orig = w_rot[r * K + k]
            var deq = w_qi[r * K + k] * w_scales[r]
            var err = orig - deq
            had_w_sig += orig * orig
            had_w_noise += err * err

    # Hadamard activation quantization
    var had_a_sig = Float32(0)
    var had_a_noise = Float32(0)
    var had_max_qi = Float32(0)
    var had_zeros = 0
    var had_total = M * K
    var had_as_min = a_scales[0]
    var had_as_max = a_scales[0]
    for i in range(M):
        if a_scales[i] < had_as_min:
            had_as_min = a_scales[i]
        if a_scales[i] > had_as_max:
            had_as_max = a_scales[i]
    for m in range(M):
        for k in range(K):
            var orig = a_rot[m * K + k]
            var deq = a_qi[m * K + k] * a_scales[m]
            var err = orig - deq
            had_a_sig += orig * orig
            had_a_noise += err * err
            var aq = absf(a_qi[m * K + k])
            if aq > had_max_qi:
                had_max_qi = aq
            if aq < 0.5:
                had_zeros += 1

    var had_eff = log2f(
        2.0 * had_max_qi + 1.0
    ) if had_max_qi >= 1 else Float32(0)
    var had_zpct = Float32(had_zeros) / Float32(max(had_total, 1)) * 100.0

    # Hadamard matmul error
    var had_m_sig = Float32(0)
    var had_m_noise = Float32(0)
    var had_max_ae = Float32(0)
    var had_sum_ae = Float32(0)
    for i in range(M * N):
        var expected = out_ref[i]
        var err = expected - out_had[i]
        had_m_sig += expected * expected
        had_m_noise += err * err
        var ae = absf(err)
        if ae > had_max_ae:
            had_max_ae = ae
        had_sum_ae += ae

    # ==================================================================
    # HADAMARD DETAILED RESULTS
    # ==================================================================

    print("--- Hadamard: Weight Quantization ---")
    var had_ws_min = w_scales[0]
    var had_ws_max = w_scales[0]
    for i in range(N):
        if w_scales[i] < had_ws_min:
            had_ws_min = w_scales[i]
        if w_scales[i] > had_ws_max:
            had_ws_max = w_scales[i]
    print(
        "  Scale range: ["
        + d2(had_ws_min * 10000)
        + ", "
        + d2(had_ws_max * 10000)
        + "] x 1e-4"
    )
    print("  Scale max/min: " + d2(had_ws_max / had_ws_min))
    print("  SQNR: " + d2(sqnr_db(had_w_sig, had_w_noise)) + " dB")
    print("")

    print("--- Hadamard: Activation Quantization ---")
    print(
        "  Scale range: ["
        + d2(had_as_min)
        + ", "
        + d2(had_as_max)
        + "]"
    )
    print("  Scale max/min: " + d2(had_as_max / had_as_min))
    print(
        "  SQNR:          "
        + d2(sqnr_db(had_a_sig, had_a_noise))
        + " dB"
    )
    print("  Max |qi|:      " + d2(had_max_qi) + " / 127")
    print("  Eff. bits:     " + d2(had_eff) + " / 8")
    print("  qi=0 fraction: " + d2(had_zpct) + "%")
    print("")

    print("--- Hadamard: End-to-End Matmul ---")
    var had_rms_out = sqrt(had_m_sig / Float32(M * N))
    var had_rms_err = sqrt(had_m_noise / Float32(M * N))
    var had_mean_ae = had_sum_ae / Float32(M * N)
    print("  Reference RMS  = " + d2(had_rms_out))
    print(
        "  SQNR           = "
        + d2(sqnr_db(had_m_sig, had_m_noise))
        + " dB"
    )
    print("  RMS error      = " + d2(had_rms_err))
    print("  Max abs error  = " + d2(had_max_ae))
    print("  Mean abs error = " + d2(had_mean_ae))
    print(
        "  RMS err / out  = "
        + d2(had_rms_err / had_rms_out * 100)
        + "%"
    )
    print("")

    # Error histogram
    print("--- Hadamard: Relative Error Distribution ---")
    var bins = InlineArray[Int, 8](fill=0)
    var counted = 0
    for i in range(M * N):
        var ra = absf(out_ref[i])
        if ra < 0.001:
            continue
        var re = absf(out_ref[i] - out_had[i]) / ra
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

    # ==================================================================
    # TIMING
    # ==================================================================

    print("--- Timing ---")
    print(
        "  F32 reference:    "
        + String(Int(tm1 - tm0) // 1000)
        + " us"
    )
    print(
        "  FWHT rotation:    "
        + String(Int(th1 - th0) // 1000)
        + " us"
    )
    print(
        "  Channelwise path: "
        + String(Int(tch1 - tch0) // 1000)
        + " us"
    )
    print(
        "  Hadamard path:    "
        + String(Int(thad1 - thad0) // 1000)
        + " us"
    )
    print("")

    # ==================================================================
    # SIDE-BY-SIDE COMPARISON
    # ==================================================================

    print("=" * 70)
    print("  Comparison")
    print("=" * 70)
    print("")
    print("                            Channelwise    Hadamard+Int8")
    print(
        "  Weight SQNR:              "
        + d2(sqnr_db(ch_w_sig, ch_w_noise))
        + " dB"
        + "       "
        + d2(sqnr_db(had_w_sig, had_w_noise))
        + " dB"
    )
    print(
        "  Activation SQNR:          "
        + d2(sqnr_db(ch_a_sig, ch_a_noise))
        + " dB"
        + "       "
        + d2(sqnr_db(had_a_sig, had_a_noise))
        + " dB"
    )
    print(
        "    1x ch. only:            "
        + d2(sqnr_db(ch_1x_sig, ch_1x_noise))
        + " dB"
        + "       (all mixed)"
    )
    print(
        "  Matmul SQNR:              "
        + d2(sqnr_db(ch_m_sig, ch_m_noise))
        + " dB"
        + "       "
        + d2(sqnr_db(had_m_sig, had_m_noise))
        + " dB"
    )
    print("")
    print(
        "  Act. scale range:         ["
        + d2(ch_as_min)
        + ", "
        + d2(ch_as_max)
        + "]"
        + "   ["
        + d2(had_as_min)
        + ", "
        + d2(had_as_max)
        + "]"
    )
    print(
        "  Act. scale ratio:         "
        + d2(ch_as_max / ch_as_min)
        + "            "
        + d2(had_as_max / had_as_min)
    )
    print(
        "  Act. max |qi|:            "
        + d2(ch_1x_max_qi)
        + " / 127"
        + "      "
        + d2(had_max_qi)
        + " / 127"
    )
    print(
        "  Act. eff. bits:           "
        + d2(ch_1x_eff)
        + " / 8"
        + "        "
        + d2(had_eff)
        + " / 8"
    )
    print(
        "  Act. qi=0 frac:           "
        + d2(ch_1x_zpct)
        + "%"
        + "          "
        + d2(had_zpct)
        + "%"
    )
    print("")
    print(
        "  Col RMS ratio:            "
        + d2(pre_max / pre_min)
        + "            "
        + d2(post_max / post_min)
    )

    # cleanup
    weights.free()
    act.free()
    w_rot.free()
    a_rot.free()
    w_qi.free()
    a_qi.free()
    w_scales.free()
    a_scales.free()
    out_ref.free()
    out_ch.free()
    out_had.free()
    is_outlier.free()
