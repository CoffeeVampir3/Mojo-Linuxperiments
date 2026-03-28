"""Stochastic rounding: depth accumulation across layers.

Chains L layers of Hadamard int8 (simplified: just the 7 projection matmuls +
residual adds, no attention) and measures how deterministic vs stochastic
rounding errors accumulate with depth.

The single-matmul test showed deterministic wins at 8 bits. The question is
whether biases compound across layers in a way that stochastic rounding avoids.

Usage: mojo design_analysis/sr_depth.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import InlineArray

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime HIDDEN = 256
comptime INTERMEDIATE = 512
comptime BLOCK = 64
comptime MAX_LAYERS = 30
comptime NUM_DEPTH = 6    # depths to test: 1, 2, 5, 10, 20, 30
comptime SR_TRIALS = 10


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

def expf(x: Float32) -> Float32:
    var xc = x
    if xc < Float32(-87.0): xc = Float32(-87.0)
    if xc > Float32(88.0): xc = Float32(88.0)
    var nf = xc * Float32(1.4426950409)
    if nf >= 0: nf = nf + Float32(0.5)
    else: nf = nf - Float32(0.5)
    var n = Int(nf)
    var r = xc - Float32(n) * Float32(0.6931458) - Float32(n) * Float32(1.4286068e-06)
    var p = Float32(1.0) + r * (Float32(1.0) + r * (Float32(0.5) + r * (
        Float32(0.16666667) + r * (Float32(0.041666668) + r * Float32(0.008333334)))))
    var exp_bits = UInt32(n + 127) << 23
    var pow2n = UnsafePointer(to=exp_bits).bitcast[Float32]()[]
    return p * pow2n


# ============================================================================
# FWHT
# ============================================================================

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


# ============================================================================
# Quantization
# ============================================================================

def quantize_det(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int):
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


# ============================================================================
# Fused operations
# ============================================================================

def rms_fwht_q(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
               cols: Int, block: Int):
    """Fused rms_norm + FWHT + deterministic quantize. Single row (M=1)."""
    for k in range(cols): scratch[k] = inp[k]
    var num_blocks = cols // block
    for b in range(num_blocks):
        fwht_inplace(scratch + b * block, block)
    var ss = Float32(0)
    var amax = Float32(0)
    for k in range(cols):
        var v = scratch[k]
        ss += v * v
        var a = absf(v)
        if a > amax: amax = a
    var rms = sqrt(ss / Float32(cols) + 1e-5)
    var scale = amax / (rms * 127.0) if amax > 0 else Float32(0)
    scales[0] = scale
    var inv = 127.0 / amax if amax > 0 else Float32(0)
    for k in range(cols):
        var v = round_nearest(scratch[k] * inv)
        if v > 127.0: v = 127.0
        elif v < -128.0: v = -128.0
        qi[k] = v

def rms_fwht_q_sr(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                   cols: Int, block: Int, mut sr: SrRng):
    """Fused rms_norm + FWHT + stochastic quantize. Single row."""
    for k in range(cols): scratch[k] = inp[k]
    var num_blocks = cols // block
    for b in range(num_blocks):
        fwht_inplace(scratch + b * block, block)
    var ss = Float32(0)
    var amax = Float32(0)
    for k in range(cols):
        var v = scratch[k]
        ss += v * v
        var a = absf(v)
        if a > amax: amax = a
    var rms = sqrt(ss / Float32(cols) + 1e-5)
    var scale = amax / (rms * 127.0) if amax > 0 else Float32(0)
    scales[0] = scale
    var inv = 127.0 / amax if amax > 0 else Float32(0)
    for k in range(cols):
        var v = scratch[k] * inv
        var fl = Float32(0)
        if v >= 0: fl = Float32(Int(v))
        else:
            fl = -Float32(Int(-v))
            if fl > v: fl = fl - 1.0
        var qi_val = fl
        if sr.next_f32() < (v - fl): qi_val = fl + 1.0
        if qi_val > 127.0: qi_val = 127.0
        elif qi_val < -128.0: qi_val = -128.0
        qi[k] = qi_val

def silu_fwht_q(gate: Ptr, up: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                cols: Int, block: Int):
    for k in range(cols):
        var g = gate[k]
        scratch[k] = g * (1.0 / (1.0 + expf(-g))) * up[k]
    var num_blocks = cols // block
    for b in range(num_blocks):
        fwht_inplace(scratch + b * block, block)
    quantize_det(scratch, qi, scales, 1, cols)

def silu_fwht_q_sr(gate: Ptr, up: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                    cols: Int, block: Int, mut sr: SrRng):
    for k in range(cols):
        var g = gate[k]
        scratch[k] = g * (1.0 / (1.0 + expf(-g))) * up[k]
    var num_blocks = cols // block
    for b in range(num_blocks):
        fwht_inplace(scratch + b * block, block)
    quantize_sr(scratch, qi, scales, 1, cols, sr)


def int8_gemv(a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr, dst: Ptr,
              n: Int, k: Int):
    """Int8 GEMV: single activation row × [N, K] weights → [N] output."""
    for j in range(n):
        var acc = 0
        for kk in range(k):
            acc += Int(a_qi[kk]) * Int(w_qi[j * k + kk])
        dst[j] = Float32(acc) * a_sc[0] * w_sc[j]


# ============================================================================
# F32 reference layer ops
# ============================================================================

def f32_rmsnorm(inp: Ptr, gamma: Ptr, dst: Ptr, cols: Int):
    var ss = Float32(0)
    for k in range(cols): ss += inp[k] * inp[k]
    var inv_rms = 1.0 / sqrt(ss / Float32(cols) + 1e-5)
    for k in range(cols): dst[k] = inp[k] * inv_rms * gamma[k]

def f32_gemv(inp: Ptr, weight: Ptr, dst: Ptr, n: Int, k: Int):
    for j in range(n):
        var acc = Float32(0)
        for kk in range(k): acc += inp[kk] * weight[j * k + kk]
        dst[j] = acc

def f32_silu_mul(gate: Ptr, up: Ptr, dst: Ptr, count: Int):
    for i in range(count):
        var g = gate[i]
        dst[i] = g * (1.0 / (1.0 + expf(-g))) * up[i]

def f32_elem_add(a: Ptr, b: Ptr, dst: Ptr, count: Int):
    for i in range(count): dst[i] = a[i] + b[i]


# ============================================================================
# One layer (simplified: projections + MLP, no attention)
# ============================================================================

def ref_layer(
    x: Ptr,
    gamma_in: Ptr, gamma_post: Ptr,
    w_q: Ptr, w_k: Ptr, w_v: Ptr, w_o: Ptr,
    w_gate: Ptr, w_up: Ptr, w_down: Ptr,
    scratch: Ptr, x_res: Ptr, gate: Ptr, up: Ptr,
):
    """F32 reference layer: norm→QKV→O+res, norm→gate/up→silu→down+res."""
    f32_rmsnorm(x, gamma_in, scratch, HIDDEN)
    # Q→attn→O simplified as: O * (Q * x_normed) — single path through
    f32_gemv(scratch, w_q, x_res, HIDDEN, HIDDEN)
    f32_gemv(x_res, w_o, scratch, HIDDEN, HIDDEN)
    f32_elem_add(x, scratch, x, HIDDEN)
    f32_rmsnorm(x, gamma_post, scratch, HIDDEN)
    f32_gemv(scratch, w_gate, gate, INTERMEDIATE, HIDDEN)
    f32_gemv(scratch, w_up, up, INTERMEDIATE, HIDDEN)
    f32_silu_mul(gate, up, gate, INTERMEDIATE)
    f32_gemv(gate, w_down, x_res, HIDDEN, INTERMEDIATE)
    f32_elem_add(x, x_res, x, HIDDEN)

def had_layer_det(
    x: Ptr,
    qi_q: Ptr, qi_k: Ptr, qi_v: Ptr, qi_o: Ptr,
    qi_gate: Ptr, qi_up: Ptr, qi_down: Ptr,
    s_q: Ptr, s_k: Ptr, s_v: Ptr, s_o: Ptr,
    s_gate: Ptr, s_up: Ptr, s_down: Ptr,
    scratch: Ptr, qi_act: Ptr, sc_act: Ptr,
    x_res: Ptr, gate: Ptr, up: Ptr,
):
    """Hadamard int8 layer, deterministic rounding."""
    # Attention: norm → Q → (treat Q as attn output) → fwht_quantize → O → residual
    rms_fwht_q(x, scratch, qi_act, sc_act, HIDDEN, BLOCK)
    int8_gemv(qi_act, qi_q, sc_act, s_q, x_res, HIDDEN, HIDDEN)
    # x_res = Q output (original domain). FWHT + quantize before O:
    for k in range(HIDDEN): scratch[k] = x_res[k]
    fwht_rows(scratch, 1, HIDDEN, BLOCK)
    quantize_det(scratch, qi_act, sc_act, 1, HIDDEN)
    int8_gemv(qi_act, qi_o, sc_act, s_o, x_res, HIDDEN, HIDDEN)
    f32_elem_add(x, x_res, x, HIDDEN)
    # MLP: norm → gate/up → silu → down → residual
    rms_fwht_q(x, scratch, qi_act, sc_act, HIDDEN, BLOCK)
    int8_gemv(qi_act, qi_gate, sc_act, s_gate, gate, INTERMEDIATE, HIDDEN)
    int8_gemv(qi_act, qi_up, sc_act, s_up, up, INTERMEDIATE, HIDDEN)
    silu_fwht_q(gate, up, scratch, qi_act, sc_act, INTERMEDIATE, BLOCK)
    int8_gemv(qi_act, qi_down, sc_act, s_down, x_res, HIDDEN, INTERMEDIATE)
    f32_elem_add(x, x_res, x, HIDDEN)

def had_layer_sr(
    x: Ptr,
    qi_q: Ptr, qi_k: Ptr, qi_v: Ptr, qi_o: Ptr,
    qi_gate: Ptr, qi_up: Ptr, qi_down: Ptr,
    s_q: Ptr, s_k: Ptr, s_v: Ptr, s_o: Ptr,
    s_gate: Ptr, s_up: Ptr, s_down: Ptr,
    scratch: Ptr, qi_act: Ptr, sc_act: Ptr,
    x_res: Ptr, gate: Ptr, up: Ptr,
    mut sr: SrRng,
):
    """Hadamard int8 layer, stochastic rounding."""
    rms_fwht_q_sr(x, scratch, qi_act, sc_act, HIDDEN, BLOCK, sr)
    int8_gemv(qi_act, qi_q, sc_act, s_q, x_res, HIDDEN, HIDDEN)
    for k in range(HIDDEN): scratch[k] = x_res[k]
    fwht_rows(scratch, 1, HIDDEN, BLOCK)
    quantize_sr(scratch, qi_act, sc_act, 1, HIDDEN, sr)
    int8_gemv(qi_act, qi_o, sc_act, s_o, x_res, HIDDEN, HIDDEN)
    f32_elem_add(x, x_res, x, HIDDEN)
    rms_fwht_q_sr(x, scratch, qi_act, sc_act, HIDDEN, BLOCK, sr)
    int8_gemv(qi_act, qi_gate, sc_act, s_gate, gate, INTERMEDIATE, HIDDEN)
    int8_gemv(qi_act, qi_up, sc_act, s_up, up, INTERMEDIATE, HIDDEN)
    silu_fwht_q_sr(gate, up, scratch, qi_act, sc_act, INTERMEDIATE, BLOCK, sr)
    int8_gemv(qi_act, qi_down, sc_act, s_down, x_res, HIDDEN, INTERMEDIATE)
    f32_elem_add(x, x_res, x, HIDDEN)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Stochastic Rounding: Depth Accumulation")
    print("=" * 70)
    print("  hidden=" + String(HIDDEN) + "  intermediate=" + String(INTERMEDIATE)
          + "  block=" + String(BLOCK))
    print("  " + String(MAX_LAYERS) + " layers, " + String(SR_TRIALS)
          + " SR trials per depth")
    print("  Simplified layer: norm→Q→O+res, norm→gate/up→silu→down+res")
    print("")

    var rng = Rng(seed=42)

    # Allocate per-layer weights (shared across all layers for simplicity —
    # same weights every layer, like a very deep residual net with tied weights)
    var w_q    = alloc[Float32](HIDDEN * HIDDEN)
    var w_k    = alloc[Float32](HIDDEN * HIDDEN)  # unused but keeps interface
    var w_v    = alloc[Float32](HIDDEN * HIDDEN)
    var w_o    = alloc[Float32](HIDDEN * HIDDEN)
    var w_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var qi_q    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_k    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_v    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_o    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var s_q    = alloc[Float32](HIDDEN)
    var s_k    = alloc[Float32](HIDDEN)
    var s_v    = alloc[Float32](HIDDEN)
    var s_o    = alloc[Float32](HIDDEN)
    var s_gate = alloc[Float32](INTERMEDIATE)
    var s_up   = alloc[Float32](INTERMEDIATE)
    var s_down = alloc[Float32](HIDDEN)
    var gamma_in   = alloc[Float32](HIDDEN)
    var gamma_post = alloc[Float32](HIDDEN)

    # Activation buffers
    var x_init  = alloc[Float32](HIDDEN)
    var x_ref   = alloc[Float32](HIDDEN)
    var x_det   = alloc[Float32](HIDDEN)
    var x_sr    = alloc[Float32](HIDDEN)
    var scratch = alloc[Float32](INTERMEDIATE)
    var qi_act  = alloc[Float32](INTERMEDIATE)
    var sc_act  = alloc[Float32](1)
    var x_res   = alloc[Float32](HIDDEN)
    var gate_buf = alloc[Float32](INTERMEDIATE)
    var up_buf  = alloc[Float32](INTERMEDIATE)

    # Generate weights
    for i in range(HIDDEN * HIDDEN):       w_q[i] = rng.normal() * 0.05
    for i in range(HIDDEN * HIDDEN):       w_o[i] = rng.normal() * 0.05
    for i in range(INTERMEDIATE * HIDDEN): w_gate[i] = rng.normal() * 0.03
    for i in range(INTERMEDIATE * HIDDEN): w_up[i] = rng.normal() * 0.03
    for i in range(HIDDEN * INTERMEDIATE): w_down[i] = rng.normal() * 0.03
    for i in range(HIDDEN):
        gamma_in[i] = 0.8 + rng.normal() * 0.1
        gamma_post[i] = 0.8 + rng.normal() * 0.1
    for i in range(HIDDEN):
        x_init[i] = rng.normal()

    # Depths to test
    var depths = InlineArray[Int, NUM_DEPTH](fill=0)
    depths[0] = 1
    depths[1] = 2
    depths[2] = 5
    depths[3] = 10
    depths[4] = 20
    depths[5] = 30

    # Run f32 reference FIRST (before weight modification) and save snapshots
    var ref_snaps = alloc[Float32](NUM_DEPTH * HIDDEN)
    for i in range(HIDDEN): x_ref[i] = x_init[i]
    for layer in range(MAX_LAYERS):
        ref_layer(x_ref, gamma_in, gamma_post, w_q, w_k, w_v, w_o,
                  w_gate, w_up, w_down, scratch, x_res, gate_buf, up_buf)
        var l = layer + 1
        for di in range(NUM_DEPTH):
            if depths[di] == l:
                for i in range(HIDDEN):
                    ref_snaps[di * HIDDEN + i] = x_ref[i]

    # NOW prepare weights (modifies w_q, w_gate, w_up, w_o, w_down in place)
    for n in range(HIDDEN):
        for k in range(HIDDEN):
            w_q[n * HIDDEN + k] = w_q[n * HIDDEN + k] * gamma_in[k]
    for n in range(INTERMEDIATE):
        for k in range(HIDDEN):
            w_gate[n * HIDDEN + k] = w_gate[n * HIDDEN + k] * gamma_post[k]
            w_up[n * HIDDEN + k] = w_up[n * HIDDEN + k] * gamma_post[k]

    fwht_rows(w_q, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_o, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_gate, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_up, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_down, HIDDEN, INTERMEDIATE, BLOCK)

    quantize_det(w_q, qi_q, s_q, HIDDEN, HIDDEN)
    quantize_det(w_o, qi_o, s_o, HIDDEN, HIDDEN)
    quantize_det(w_gate, qi_gate, s_gate, INTERMEDIATE, HIDDEN)
    quantize_det(w_up, qi_up, s_up, INTERMEDIATE, HIDDEN)
    quantize_det(w_down, qi_down, s_down, HIDDEN, INTERMEDIATE)

    var det_db = InlineArray[Float32, NUM_DEPTH](fill=0)
    var sr_mean = InlineArray[Float32, NUM_DEPTH](fill=0)
    var sr_min = InlineArray[Float32, NUM_DEPTH](fill=0)
    var sr_max = InlineArray[Float32, NUM_DEPTH](fill=0)

    print("  Running...")

    for di in range(NUM_DEPTH):
        var t0 = perf_counter_ns()
        var num_layers = depths[di]

        # Deterministic: run num_layers layers
        for i in range(HIDDEN): x_det[i] = x_init[i]
        for layer in range(num_layers):
            had_layer_det(x_det, qi_q, qi_k, qi_v, qi_o,
                          qi_gate, qi_up, qi_down,
                          s_q, s_k, s_v, s_o, s_gate, s_up, s_down,
                          scratch, qi_act, sc_act, x_res, gate_buf, up_buf)

        var snap = ref_snaps + di * HIDDEN
        var sig = Float32(0)
        var noise = Float32(0)
        for i in range(HIDDEN):
            sig += snap[i] * snap[i]
            var err = snap[i] - x_det[i]
            noise += err * err
        det_db[di] = sqnr_db(sig, noise)

        # Stochastic: multiple trials
        var s_sum = Float32(0)
        var s_lo = Float32(999.0)
        var s_hi = Float32(-999.0)
        for trial in range(SR_TRIALS):
            var sr = SrRng(seed=UInt64(5000 + di * 100 + trial))
            for i in range(HIDDEN): x_sr[i] = x_init[i]
            for layer in range(num_layers):
                had_layer_sr(x_sr, qi_q, qi_k, qi_v, qi_o,
                             qi_gate, qi_up, qi_down,
                             s_q, s_k, s_v, s_o, s_gate, s_up, s_down,
                             scratch, qi_act, sc_act, x_res, gate_buf, up_buf, sr)

            var s2 = Float32(0)
            var n2 = Float32(0)
            for i in range(HIDDEN):
                s2 += snap[i] * snap[i]
                var err = snap[i] - x_sr[i]
                n2 += err * err
            var db = sqnr_db(s2, n2)
            s_sum += db
            if db < s_lo: s_lo = db
            if db > s_hi: s_hi = db

        sr_mean[di] = s_sum / Float32(SR_TRIALS)
        sr_min[di] = s_lo
        sr_max[di] = s_hi

        var ms = Int(perf_counter_ns() - t0) // 1_000_000
        print("    L=" + String(num_layers) + ": " + String(ms) + " ms")

    # ==================================================================
    # Results
    # ==================================================================

    print("")
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print("")
    print("  Layers  Determ.   Stoch.    Delta     Stoch. range")

    for di in range(NUM_DEPTH):
        var num_layers = depths[di]
        var delta = sr_mean[di] - det_db[di]
        var pad = "     " if num_layers < 10 else "    "
        print(
            "  " + String(num_layers) + pad
            + col(det_db[di])
            + col(sr_mean[di])
            + col_s(delta)
            + d2(sr_min[di]) + ".." + d2(sr_max[di]) + " dB"
        )

    print("")

    # Cleanup
    w_q.free(); w_k.free(); w_v.free(); w_o.free()
    w_gate.free(); w_up.free(); w_down.free()
    qi_q.free(); qi_k.free(); qi_v.free(); qi_o.free()
    qi_gate.free(); qi_up.free(); qi_down.free()
    s_q.free(); s_k.free(); s_v.free(); s_o.free()
    s_gate.free(); s_up.free(); s_down.free()
    gamma_in.free(); gamma_post.free()
    x_init.free(); x_ref.free(); x_det.free(); x_sr.free()
    scratch.free(); qi_act.free(); sc_act.free()
    x_res.free(); gate_buf.free(); up_buf.free()
    ref_snaps.free()
