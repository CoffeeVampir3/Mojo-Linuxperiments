"""FWHT block size sweep — effect on int8 quantization quality.

Same data as hadamard.mojo (5% outlier channels at 30x). Sweeps FWHT block
size from 2 to K=2048 and reports matmul SQNR for each.

Usage: mojo design_analysis/block_sweep.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime M = 64
comptime K = 2048
comptime N = 2048


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

def channelwise_quantize(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        var base = r * cols
        var amax = Float32(0)
        for k in range(cols):
            var a = absf(src[base + k])
            if a > amax: amax = a
        var scale = amax / 127.0
        scales[r] = scale
        var inv = 1.0 / scale if scale > 0 else Float32(0)
        for k in range(cols):
            var v = round_nearest(src[base + k] * inv)
            if v > 127.0: v = 127.0
            elif v < -128.0: v = -128.0
            qi[base + k] = v


def main():
    print("=" * 70)
    print("  FWHT Block Size Sweep")
    print("=" * 70)
    print("  M=" + String(M) + "  K=" + String(K) + "  N=" + String(N))
    print("  Activations: N(0,1) + 5% channels at 30x")
    print("")

    var rng = Rng(seed=42)

    var weights = alloc[Float32](N * K)
    var act_base = alloc[Float32](M * K)
    var w_rot = alloc[Float32](N * K)
    var a_rot = alloc[Float32](M * K)
    var w_qi = alloc[Float32](N * K)
    var a_qi = alloc[Float32](M * K)
    var w_sc = alloc[Float32](N)
    var a_sc = alloc[Float32](M)
    var out_ref = alloc[Float32](M * N)
    var out_test = alloc[Float32](M * N)
    var is_outlier = alloc[UInt8](K)

    # Generate weights
    for i in range(N * K):
        weights[i] = rng.normal() * 0.02

    # Generate activations with 5% outliers at 30x
    for i in range(M * K):
        act_base[i] = rng.normal()
    var num_out = K * 5 // 100
    for k in range(K): is_outlier[k] = UInt8(0)
    var placed = 0
    while placed < num_out:
        var idx = Int(rng.next() % UInt64(K))
        if is_outlier[idx] == UInt8(0):
            is_outlier[idx] = UInt8(1)
            placed += 1
    for m in range(M):
        for k in range(K):
            if is_outlier[k] != UInt8(0):
                act_base[m * K + k] = act_base[m * K + k] * Float32(30.0)

    # F32 reference matmul
    for m in range(M):
        for n in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += act_base[m * K + k] * weights[n * K + k]
            out_ref[m * N + n] = acc

    # Channelwise baseline (no rotation)
    channelwise_quantize(weights, w_qi, w_sc, N, K)
    channelwise_quantize(act_base, a_qi, a_sc, M, K)
    for m in range(M):
        for n in range(N):
            var acc = 0
            for k in range(K):
                acc += Int(a_qi[m * K + k]) * Int(w_qi[n * K + k])
            out_test[m * N + n] = Float32(acc) * a_sc[m] * w_sc[n]
    var ch_sig = Float32(0)
    var ch_noise = Float32(0)
    for i in range(M * N):
        var e = out_ref[i]
        var err = e - out_test[i]
        ch_sig += e * e
        ch_noise += err * err

    print(
        "  block=none (channelwise):  SQNR = "
        + d2(sqnr_db(ch_sig, ch_noise)) + " dB"
    )

    # Sweep block sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    var block = 2
    while block <= K:
        # Copy and rotate
        for i in range(N * K): w_rot[i] = weights[i]
        for i in range(M * K): a_rot[i] = act_base[i]
        fwht_rows(w_rot, N, K, block)
        fwht_rows(a_rot, M, K, block)

        channelwise_quantize(w_rot, w_qi, w_sc, N, K)
        channelwise_quantize(a_rot, a_qi, a_sc, M, K)

        for m in range(M):
            for n in range(N):
                var acc = 0
                for k in range(K):
                    acc += Int(a_qi[m * K + k]) * Int(w_qi[n * K + k])
                out_test[m * N + n] = Float32(acc) * a_sc[m] * w_sc[n]

        var sig = Float32(0)
        var noise = Float32(0)
        for i in range(M * N):
            var e = out_ref[i]
            var err = e - out_test[i]
            sig += e * e
            noise += err * err

        var db = sqnr_db(sig, noise)
        var pct = sqrt(noise / sig) * 100.0
        var pad = "    " if block < 10 else ("   " if block < 100 else ("  " if block < 1000 else " "))
        print(
            "  block=" + String(block) + pad
            + "(" + String(block) + "x" + String(block) + " Hadamard, "
            + String(K // block) + " blocks):  SQNR = "
            + d2(db) + " dB  RMS err = " + d2(pct) + "%"
        )
        block *= 2

    print("")

    weights.free()
    act_base.free()
    w_rot.free()
    a_rot.free()
    w_qi.free()
    a_qi.free()
    w_sc.free()
    a_sc.free()
    out_ref.free()
    out_test.free()
    is_outlier.free()
