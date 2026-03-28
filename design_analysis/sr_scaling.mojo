"""Stochastic rounding: scaling with inner product dimension.

Sweeps the contraction dimension K to observe whether stochastic rounding
improves relative to deterministic as inner products grow longer.

Theory: stochastic rounding makes quantization error unbiased per element.
For inner products of length K, unbiased errors cancel as O(1/sqrt(K)).
Deterministic rounding has systematic bias that doesn't cancel as cleanly.
The effect should become visible at larger K.

Fixed: M=32 (batch), N=512 (output). Activations have 5% outliers at 30x.
Weights quantized deterministically in both paths (offline quantization).
Only activation quantization varies (deterministic vs stochastic).

Usage: mojo design_analysis/sr_scaling.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import InlineArray

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime M_DIM = 32
comptime N_DIM = 512
comptime K_MAX = 4096
comptime NUM_K = 8
comptime TRIALS = 20


struct Rng:
    var state: UInt64
    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else UInt64(1)
    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state
    def normal(mut self) -> Float32:
        var s = Float32(0)
        for _ in range(12):
            s += Float32(self.next() & 0xFFFFFF) / 16777216.0
        return s - 6.0


struct SrRng:
    var state: UInt64
    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else UInt64(1)
    def next_f32(mut self) -> Float32:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return Float32(self.state & 0xFFFFFF) / 16777216.0


@always_inline
def absf(x: Float32) -> Float32:
    return x if x >= 0 else -x

def log2f(x: Float32) -> Float32:
    if x <= 0: return Float32(-200.0)
    var v = x
    var bits = UnsafePointer(to=v).bitcast[UInt32]()[]
    return Float32(Int((bits >> 23) & 0xFF) - 127) + Float32(bits & 0x7FFFFF) / 8388608.0

def sqnr_db(sig: Float32, noise: Float32) -> Float32:
    if noise <= 0: return Float32(999.0)
    if sig <= 0: return Float32(-999.0)
    return 3.01029995664 * log2f(sig / noise)

def round_nearest(x: Float32) -> Float32:
    if x >= 0: return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))

def d2(x: Float32) -> String:
    var neg = x < 0
    var v = absf(x)
    var cents = Int(v * 100.0 + 0.5)
    var w = cents // 100
    var f = cents % 100
    var s = String(w) + "."
    if f < 10: s += "0"
    s += String(f)
    if neg: return "-" + s
    return s

def col(x: Float32) -> String:
    var s = d2(x)
    while len(s) < 9: s += " "
    return s

def col_s(x: Float32) -> String:
    var s = String("")
    if x >= 0: s = "+" + d2(x)
    else: s = d2(x)
    while len(s) < 9: s += " "
    return s


def fwht_inplace(buf: Ptr, n: Int):
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
    var sc = 1.0 / sqrt(Float32(n))
    for i in range(n): buf[i] = buf[i] * sc

def fwht_rows(mat: Ptr, rows: Int, cols: Int, block: Int):
    for r in range(rows):
        var base = mat + r * cols
        for b in range(cols // block):
            fwht_inplace(base + b * block, block)


def quantize_det(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int):
    """Per-row deterministic (round-to-nearest) int8 quantize."""
    for r in range(rows):
        var base = r * cols
        var amax = Float32(0)
        for k in range(cols):
            var a = absf(src[base + k])
            if a > amax: amax = a
        var sc = amax / 127.0
        scales[r] = sc
        var inv = 1.0 / sc if sc > 0 else Float32(0)
        for k in range(cols):
            var v = round_nearest(src[base + k] * inv)
            if v > 127.0: v = 127.0
            elif v < -128.0: v = -128.0
            qi[base + k] = v


def quantize_sr(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int,
                mut sr: SrRng):
    """Per-row stochastic rounding int8 quantize."""
    for r in range(rows):
        var base = r * cols
        var amax = Float32(0)
        for k in range(cols):
            var a = absf(src[base + k])
            if a > amax: amax = a
        var sc = amax / 127.0
        scales[r] = sc
        var inv = 1.0 / sc if sc > 0 else Float32(0)
        for k in range(cols):
            var v = src[base + k] * inv
            var fl = Float32(0)
            if v >= 0: fl = Float32(Int(v))
            else:
                fl = -Float32(Int(-v))
                if fl > v: fl = fl - 1.0
            var qi_val = fl
            if sr.next_f32() < (v - fl): qi_val = fl + 1.0
            if qi_val > 127.0: qi_val = 127.0
            elif qi_val < -128.0: qi_val = -128.0
            qi[base + k] = qi_val


def matmul_sqnr(
    a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr,
    out_ref: Ptr, out_test: Ptr,
    m: Int, n: Int, k: Int,
) -> Float32:
    """Int8 matmul + compare to reference. Returns SQNR in dB."""
    for i in range(m):
        for j in range(n):
            var acc = 0
            for kk in range(k):
                acc += Int(a_qi[i * k + kk]) * Int(w_qi[j * k + kk])
            out_test[i * n + j] = Float32(acc) * a_sc[i] * w_sc[j]
    var sig = Float32(0)
    var noise = Float32(0)
    for i in range(m * n):
        var e = out_ref[i]
        var err = e - out_test[i]
        sig += e * e
        noise += err * err
    return sqnr_db(sig, noise)


def main():
    print("=" * 70)
    print("  Stochastic Rounding: Scaling with Contraction Dimension K")
    print("=" * 70)
    print("  M=" + String(M_DIM) + "  N=" + String(N_DIM)
          + "  trials=" + String(TRIALS))
    print("  Activations: N(0,1) + 5% outlier channels at 30x")
    print("  Weights: deterministic quantization in both paths")
    print("  Only activation quantization varies (det vs stochastic)")
    print("")

    # Allocate max-size buffers
    var weights = alloc[Float32](N_DIM * K_MAX)
    var act = alloc[Float32](M_DIM * K_MAX)
    var w_rot = alloc[Float32](N_DIM * K_MAX)
    var a_rot = alloc[Float32](M_DIM * K_MAX)
    var w_qi = alloc[Float32](N_DIM * K_MAX)
    var a_qi = alloc[Float32](M_DIM * K_MAX)
    var w_sc = alloc[Float32](N_DIM)
    var a_sc = alloc[Float32](M_DIM)
    var out_ref = alloc[Float32](M_DIM * N_DIM)
    var out_test = alloc[Float32](M_DIM * N_DIM)
    var is_outlier = alloc[UInt8](K_MAX)

    # K values to sweep
    var k_vals = InlineArray[Int, NUM_K](fill=0)
    k_vals[0] = 32
    k_vals[1] = 64
    k_vals[2] = 128
    k_vals[3] = 256
    k_vals[4] = 512
    k_vals[5] = 1024
    k_vals[6] = 2048
    k_vals[7] = 4096

    # Result arrays
    var det_db = InlineArray[Float32, NUM_K](fill=0)
    var sr_mean_db = InlineArray[Float32, NUM_K](fill=0)
    var sr_min_db = InlineArray[Float32, NUM_K](fill=0)
    var sr_max_db = InlineArray[Float32, NUM_K](fill=0)

    var rng = Rng(seed=42)

    print("  Running...")

    for ci in range(NUM_K):
        var t0 = perf_counter_ns()
        var k = k_vals[ci]
        var block = k if k <= 64 else 64

        # Generate data for this K
        for i in range(N_DIM * k):
            weights[i] = rng.normal() * 0.02
        for i in range(M_DIM * k):
            act[i] = rng.normal()

        # Outlier mask: 5% of K channels, evenly spaced
        var n_out = k * 5 // 100
        for i in range(k): is_outlier[i] = UInt8(0)
        if n_out > 0:
            var stride = k // n_out
            for i in range(n_out):
                is_outlier[i * stride] = UInt8(1)
        for m in range(M_DIM):
            for j in range(k):
                if is_outlier[j] != UInt8(0):
                    act[m * k + j] = act[m * k + j] * Float32(30.0)

        # F32 reference matmul
        for m in range(M_DIM):
            for n in range(N_DIM):
                var acc = Float32(0)
                for kk in range(k):
                    acc += act[m * k + kk] * weights[n * k + kk]
                out_ref[m * N_DIM + n] = acc

        # FWHT both (single-sided rotation)
        for i in range(N_DIM * k): w_rot[i] = weights[i]
        for i in range(M_DIM * k): a_rot[i] = act[i]
        fwht_rows(w_rot, N_DIM, k, block)
        fwht_rows(a_rot, M_DIM, k, block)

        # Quantize weights (deterministic, shared by both paths)
        quantize_det(w_rot, w_qi, w_sc, N_DIM, k)

        # Deterministic activation quantization
        quantize_det(a_rot, a_qi, a_sc, M_DIM, k)
        det_db[ci] = matmul_sqnr(a_qi, w_qi, a_sc, w_sc, out_ref, out_test, M_DIM, N_DIM, k)

        # Stochastic activation quantization — multiple trials
        var sr_sum = Float32(0)
        var sr_lo = Float32(999.0)
        var sr_hi = Float32(-999.0)
        for trial in range(TRIALS):
            var sr = SrRng(seed=UInt64(10000 + ci * 1000 + trial))
            quantize_sr(a_rot, a_qi, a_sc, M_DIM, k, sr)
            var db = matmul_sqnr(a_qi, w_qi, a_sc, w_sc, out_ref, out_test, M_DIM, N_DIM, k)
            sr_sum += db
            if db < sr_lo: sr_lo = db
            if db > sr_hi: sr_hi = db
        sr_mean_db[ci] = sr_sum / Float32(TRIALS)
        sr_min_db[ci] = sr_lo
        sr_max_db[ci] = sr_hi

        var ms = Int(perf_counter_ns() - t0) // 1_000_000
        print("    K=" + String(k) + ": " + String(ms) + " ms")

    # ==================================================================
    # Results
    # ==================================================================

    print("")
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print("")
    print("  K      block  Determ.   Stoch.    Delta     Stoch. range")

    for ci in range(NUM_K):
        var k = k_vals[ci]
        var block = k if k <= 64 else 64
        var delta = sr_mean_db[ci] - det_db[ci]
        var pad = "    " if k < 100 else ("   " if k < 1000 else "  ")
        print(
            "  " + String(k) + pad
            + String(block)
            + ("     " if block < 100 else "    ")
            + col(det_db[ci])
            + col(sr_mean_db[ci])
            + col_s(delta)
            + d2(sr_min_db[ci]) + ".." + d2(sr_max_db[ci]) + " dB"
        )

    print("")

    # Cleanup
    weights.free()
    act.free()
    w_rot.free()
    a_rot.free()
    w_qi.free()
    a_qi.free()
    w_sc.free()
    a_sc.free()
    out_ref.free()
    out_test.free()
    is_outlier.free()
