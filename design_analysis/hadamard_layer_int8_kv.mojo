"""Int8 KV cache with RoPE — numerical validation.

Compares three attention paths against an f32 reference:
  1. bf16 KV cache: store bf16 after RoPE, FWHT+quantize at attention time
  2. int8 KV cache: FWHT+quantize K after RoPE at write time, skip re-rotation
  3. f32 reference: no quantization, no rotation

The key question: does pre-rotating position-encoded keys preserve quality?
Parseval guarantees <FWHT(RoPE(Q)), FWHT(RoPE(K))> = <RoPE(Q), RoPE(K)>,
so the scoring is exact in infinite precision. The error comes only from
int8 quantization of the rotated+position-encoded vectors.

Usage: mojo design_analysis/hadamard_layer_int8_kv.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]

comptime HIDDEN = 576
comptime NUM_HEADS = 9
comptime NUM_KV_HEADS = 3
comptime HEAD_DIM = HIDDEN // NUM_HEADS          # 64
comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM     # 192
comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS  # 3
comptime CONTEXT = 256                           # KV cache depth
comptime BLOCK = HEAD_DIM


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
    def normal(mut self) -> Float32:
        var s = Float32(0)
        for _ in range(12):
            s += Float32(self.next() & 0xFFFFFF) / 16777216.0
        return s - 6.0


# ============================================================================
# Helpers
# ============================================================================

@always_inline
def absf(x: Float32) -> Float32:
    return x if x >= 0 else -x

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

def log2f(x: Float32) -> Float32:
    if x <= 0: return Float32(-200.0)
    var v = x
    var bits = UnsafePointer(to=v).bitcast[UInt32]()[]
    return Float32(Int((bits >> 23) & 0xFF) - 127) + Float32(bits & 0x7FFFFF) / 8388608.0

def sqnr_db(sig: Float32, noise: Float32) -> Float32:
    if noise <= 0: return Float32(999.0)
    if sig <= 0: return Float32(-999.0)
    return 3.01029995664 * log2f(sig / noise)

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

def round_nearest(x: Float32) -> Float32:
    if x >= 0: return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))


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


# ============================================================================
# Quantization
# ============================================================================

def quantize_vec(src: Ptr, qi: Ptr, scale: Ptr, count: Int):
    var amax = Float32(0)
    for i in range(count):
        var a = absf(src[i])
        if a > amax: amax = a
    var sc = amax / 127.0
    scale[0] = sc
    var inv = 1.0 / sc if sc > 0 else Float32(0)
    for i in range(count):
        var v = round_nearest(src[i] * inv)
        if v > 127.0: v = 127.0
        elif v < -128.0: v = -128.0
        qi[i] = v


# ============================================================================
# RoPE — simplified scalar, per-head, applied to a single position
# ============================================================================

def apply_rope(vec: Ptr, head_dim: Int, num_heads: Int, pos: Int, theta: Float64):
    """Apply RoPE in-place to vec[num_heads * head_dim]."""
    var half = head_dim // 2
    for h in range(num_heads):
        var base = h * head_dim
        for i in range(half):
            var freq = 1.0 / Float64(theta) ** (Float64(2 * i) / Float64(head_dim))
            var angle = Float64(pos) * freq
            var cos_v = Float32(cos(angle))
            var sin_v = Float32(sin(angle))
            var x0 = vec[base + i]
            var x1 = vec[base + half + i]
            vec[base + i] = x0 * cos_v - x1 * sin_v
            vec[base + half + i] = x0 * sin_v + x1 * cos_v

def cos(x: Float64) -> Float64:
    # Reduce to [0, 2pi]
    var TWO_PI = Float64(6.283185307179586)
    var v = x - TWO_PI * Float64(Int(x / TWO_PI))
    if v < 0: v = v + TWO_PI
    # Taylor series (sufficient for our range)
    var HALF_PI = Float64(1.5707963267948966)
    var PI = Float64(3.141592653589793)
    # Map to [0, pi/2]
    var sign = Float64(1.0)
    if v > PI:
        v = v - PI
        sign = -1.0
    if v > HALF_PI:
        v = PI - v
        sign = -sign
    var v2 = v * v
    return sign * (1.0 - v2 / 2.0 + v2 * v2 / 24.0 - v2 * v2 * v2 / 720.0
                   + v2 * v2 * v2 * v2 / 40320.0)

def sin(x: Float64) -> Float64:
    return cos(x - Float64(1.5707963267948966))


# ============================================================================
# F32 reference attention with RoPE
# ============================================================================

def f32_attention_with_rope(
    q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr,
    context_len: Int, query_pos: Int, theta: Float64,
):
    """F32 GQA attention. Q has RoPE at query_pos. K cache has RoPE at each position."""
    var scores = alloc[Float32](context_len)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        for t in range(context_len):
            var acc = Float32(0)
            for d in range(HEAD_DIM):
                acc += q[h * HEAD_DIM + d] * k_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            scores[t] = acc * inv_sqrt

        var max_s = scores[0]
        for t in range(1, context_len):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(context_len):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(context_len):
            scores[t] = scores[t] / sum_e

        for d in range(HEAD_DIM):
            var acc = Float32(0)
            for t in range(context_len):
                acc += scores[t] * v_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            dst[h * HEAD_DIM + d] = acc

    scores.free()


# ============================================================================
# Path A: bf16-style KV — rotate on the fly at attention time
# ============================================================================

def int8_attention_bf16_kv(
    q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr, context_len: Int,
):
    """Int8 attention, KV cache stores f32 (bf16-equivalent).
    Rotates K and V per-head at attention time."""
    var head_buf = alloc[Float32](HEAD_DIM)
    var head_qi = alloc[Float32](HEAD_DIM)
    var head_sc = alloc[Float32](1)
    var k_qi = alloc[Float32](context_len * KV_HIDDEN)
    var k_sc = alloc[Float32](context_len * NUM_KV_HEADS)
    var v_qi = alloc[Float32](context_len * KV_HIDDEN)
    var v_sc = alloc[Float32](context_len * NUM_KV_HEADS)
    var scores = alloc[Float32](context_len)
    var w_qi = alloc[Float32](context_len)
    var w_sc = alloc[Float32](1)

    # Rotate + quantize K cache on the fly
    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM): head_buf[d] = k_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec(head_buf, k_qi + off, k_sc + t * NUM_KV_HEADS + g, HEAD_DIM)

    # Rotate + quantize V cache on the fly
    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM): head_buf[d] = v_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec(head_buf, v_qi + off, v_sc + t * NUM_KV_HEADS + g, HEAD_DIM)

    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR

        # Rotate + quantize Q
        for d in range(HEAD_DIM): head_buf[d] = q[h * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, head_qi, head_sc, HEAD_DIM)

        # Int8 scoring
        for t in range(context_len):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc[t * NUM_KV_HEADS + g] * inv_sqrt

        # Softmax
        var max_s = scores[0]
        for t in range(1, context_len):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(context_len):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(context_len):
            scores[t] = scores[t] / sum_e

        # V scale absorption + quantize weights
        for t in range(context_len):
            scores[t] = scores[t] * v_sc[t * NUM_KV_HEADS + g]
        quantize_vec(scores, w_qi, w_sc, context_len)

        # Int8 aggregation — output in rotated domain
        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(context_len):
                acc += Int(w_qi[t]) * Int(v_qi[t * KV_HIDDEN + g * HEAD_DIM + d])
            dst[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    head_buf.free(); head_qi.free(); head_sc.free()
    k_qi.free(); k_sc.free(); v_qi.free(); v_sc.free()
    scores.free(); w_qi.free(); w_sc.free()


# ============================================================================
# Path B: int8 KV — pre-rotate at write time, skip re-rotation at attention
# ============================================================================

def write_kv_int8(
    k_bf16: Ptr, v_bf16: Ptr,
    k_qi_cache: Ptr, k_sc_cache: Ptr,
    v_qi_cache: Ptr, v_sc_cache: Ptr,
    pos: Int,
):
    """Write one position to int8 KV cache. K already has RoPE applied.
    FWHT per head + quantize to int8."""
    var head_buf = alloc[Float32](HEAD_DIM)
    for g in range(NUM_KV_HEADS):
        var off = pos * KV_HIDDEN + g * HEAD_DIM
        for d in range(HEAD_DIM): head_buf[d] = k_bf16[g * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, k_qi_cache + off, k_sc_cache + pos * NUM_KV_HEADS + g, HEAD_DIM)

        for d in range(HEAD_DIM): head_buf[d] = v_bf16[g * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, v_qi_cache + off, v_sc_cache + pos * NUM_KV_HEADS + g, HEAD_DIM)
    head_buf.free()


def int8_attention_int8_kv(
    q: Ptr,
    k_qi_cache: Ptr, k_sc_cache: Ptr,
    v_qi_cache: Ptr, v_sc_cache: Ptr,
    dst: Ptr, context_len: Int,
):
    """Int8 attention with pre-rotated int8 KV cache.
    Only Q needs FWHT at attention time."""
    var head_buf = alloc[Float32](HEAD_DIM)
    var head_qi = alloc[Float32](HEAD_DIM)
    var head_sc = alloc[Float32](1)
    var scores = alloc[Float32](context_len)
    var w_qi = alloc[Float32](context_len)
    var w_sc = alloc[Float32](1)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR

        # FWHT + quantize Q (only rotation needed at attention time)
        for d in range(HEAD_DIM): head_buf[d] = q[h * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, head_qi, head_sc, HEAD_DIM)

        # Int8 scoring — K cache already rotated
        for t in range(context_len):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi_cache[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc_cache[t * NUM_KV_HEADS + g] * inv_sqrt

        # Softmax
        var max_s = scores[0]
        for t in range(1, context_len):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(context_len):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(context_len):
            scores[t] = scores[t] / sum_e

        # V scale absorption + quantize weights
        for t in range(context_len):
            scores[t] = scores[t] * v_sc_cache[t * NUM_KV_HEADS + g]
        quantize_vec(scores, w_qi, w_sc, context_len)

        # Int8 aggregation — V cache already rotated, output in rotated domain
        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(context_len):
                acc += Int(w_qi[t]) * Int(v_qi_cache[t * KV_HIDDEN + g * HEAD_DIM + d])
            dst[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    head_buf.free(); head_qi.free(); head_sc.free()
    scores.free(); w_qi.free(); w_sc.free()


# ============================================================================
# Main — compare all three paths
# ============================================================================

def main():
    print("=" * 70)
    print("  Int8 KV Cache with RoPE — Numerical Validation")
    print("=" * 70)
    print("  HIDDEN=" + String(HIDDEN) + " HEADS=" + String(NUM_HEADS)
          + " KV_HEADS=" + String(NUM_KV_HEADS) + " HEAD_DIM=" + String(HEAD_DIM))
    print("  CONTEXT=" + String(CONTEXT) + " BLOCK=" + String(BLOCK))
    print("")

    var rng = Rng(seed=42)
    comptime THETA = 10000.0

    # Allocate
    var q_raw = alloc[Float32](HIDDEN)
    var q_roped = alloc[Float32](HIDDEN)
    var k_raw = alloc[Float32](KV_HIDDEN)
    var v_raw = alloc[Float32](KV_HIDDEN)

    # bf16-path KV cache (stores f32 with RoPE)
    var kc_f32 = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_f32 = alloc[Float32](CONTEXT * KV_HIDDEN)

    # int8-path KV cache (pre-rotated, quantized)
    var kc_qi = alloc[Float32](CONTEXT * KV_HIDDEN)
    var kc_sc = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    var vc_qi = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_sc = alloc[Float32](CONTEXT * NUM_KV_HEADS)

    # Outputs
    var out_ref = alloc[Float32](HIDDEN)
    var out_bf16kv = alloc[Float32](HIDDEN)
    var out_i8kv = alloc[Float32](HIDDEN)

    # Fill KV cache with structured attention data.
    # A few "important" positions have K vectors aligned with Q,
    # creating a realistic attention pattern where some positions dominate.

    # Generate query first so we can plant aligned keys
    var query_pos = CONTEXT - 1
    for i in range(HIDDEN):
        q_raw[i] = rng.normal() * 0.5
        q_roped[i] = q_raw[i]

    for pos in range(CONTEXT):
        for i in range(KV_HIDDEN):
            k_raw[i] = rng.normal() * 0.3
            v_raw[i] = rng.normal() * 0.5

        # Make ~5% of positions strongly aligned with Q (per kv_head)
        if pos % 20 == 0 or pos == CONTEXT - 2:
            for g in range(NUM_KV_HEADS):
                # Copy Q head pattern into this K head (scaled)
                var qh = g * GQA_FACTOR  # first query head in this group
                for d in range(HEAD_DIM):
                    k_raw[g * HEAD_DIM + d] = q_raw[qh * HEAD_DIM + d] * 2.0
                # V at important positions has larger magnitude (clear signal)
                for d in range(HEAD_DIM):
                    v_raw[g * HEAD_DIM + d] = rng.normal() * 2.0

        # Apply RoPE to K at this position
        apply_rope(k_raw, HEAD_DIM, NUM_KV_HEADS, pos, THETA)

        # bf16 path: store K(roped), V as-is
        for i in range(KV_HIDDEN):
            kc_f32[pos * KV_HIDDEN + i] = k_raw[i]
            vc_f32[pos * KV_HIDDEN + i] = v_raw[i]

        # int8 path: FWHT + quantize K(roped) and V per head
        write_kv_int8(k_raw, v_raw, kc_qi, kc_sc, vc_qi, vc_sc, pos)

    # Apply RoPE to Q
    apply_rope(q_roped, HEAD_DIM, NUM_HEADS, query_pos, THETA)

    # ====================================================================
    # Path 1: F32 reference (RoPE applied, no rotation, no quantization)
    # ====================================================================
    f32_attention_with_rope(q_roped, kc_f32, vc_f32, out_ref, CONTEXT, query_pos, THETA)

    # ====================================================================
    # Path 2: bf16 KV cache — rotate K,V on the fly at attention time
    # ====================================================================
    int8_attention_bf16_kv(q_roped, kc_f32, vc_f32, out_bf16kv, CONTEXT)

    # ====================================================================
    # Path 3: int8 KV cache — pre-rotated at write time
    # ====================================================================
    int8_attention_int8_kv(q_roped, kc_qi, kc_sc, vc_qi, vc_sc, out_i8kv, CONTEXT)

    # ====================================================================
    # Un-rotate int8 outputs to original domain for comparison.
    # Int8 attention produces output in per-head rotated domain (V is
    # rotated, so weighted sum is rotated). F32 reference is in original
    # domain. FWHT is self-inverse, so apply again to un-rotate.
    # ====================================================================
    for h in range(NUM_HEADS):
        fwht_inplace(out_bf16kv + h * HEAD_DIM, HEAD_DIM)
        fwht_inplace(out_i8kv + h * HEAD_DIM, HEAD_DIM)

    # ====================================================================
    # Compare (all outputs now in original domain)
    # ====================================================================

    var sig = Float32(0)
    var noise_bf16 = Float32(0)
    var noise_i8 = Float32(0)
    var noise_diff = Float32(0)
    for i in range(HIDDEN):
        var ref_val = out_ref[i]
        sig += ref_val * ref_val
        var e_bf16 = ref_val - out_bf16kv[i]
        var e_i8 = ref_val - out_i8kv[i]
        var e_diff = out_bf16kv[i] - out_i8kv[i]
        noise_bf16 += e_bf16 * e_bf16
        noise_i8 += e_i8 * e_i8
        noise_diff += e_diff * e_diff

    print("  Attention output SQNR vs f32 reference:")
    print("    bf16 KV (rotate at attn):   " + d2(sqnr_db(sig, noise_bf16)) + " dB")
    print("    int8 KV (pre-rotate):       " + d2(sqnr_db(sig, noise_i8)) + " dB")
    print("    bf16 vs int8 KV difference: " + d2(sqnr_db(sig, noise_diff)) + " dB")
    print("")

    var rms_bf16 = sqrt(noise_bf16 / Float32(HIDDEN))
    var rms_i8 = sqrt(noise_i8 / Float32(HIDDEN))
    var rms_sig = sqrt(sig / Float32(HIDDEN))
    print("  RMS error:")
    print("    bf16 KV: " + d2(rms_bf16) + "  (" + d2(rms_bf16 / rms_sig * 100.0) + "%)")
    print("    int8 KV: " + d2(rms_i8) + "  (" + d2(rms_i8 / rms_sig * 100.0) + "%)")
    print("")

    # Per-head analysis
    print("  Per-head SQNR (dB):")
    print("  head   bf16_KV   int8_KV   delta")
    for h in range(NUM_HEADS):
        var h_sig = Float32(0)
        var h_n_bf16 = Float32(0)
        var h_n_i8 = Float32(0)
        for d in range(HEAD_DIM):
            var idx = h * HEAD_DIM + d
            var rv = out_ref[idx]
            h_sig += rv * rv
            var eb = rv - out_bf16kv[idx]
            var ei = rv - out_i8kv[idx]
            h_n_bf16 += eb * eb
            h_n_i8 += ei * ei
        var db_bf16 = sqnr_db(h_sig, h_n_bf16)
        var db_i8 = sqnr_db(h_sig, h_n_i8)
        var delta = db_i8 - db_bf16
        print("  " + String(h)
              + "      " + d2(db_bf16)
              + "     " + d2(db_i8)
              + "     " + d2(delta))

    print("")

    # Cleanup
    q_raw.free(); q_roped.free(); k_raw.free(); v_raw.free()
    kc_f32.free(); vc_f32.free()
    kc_qi.free(); kc_sc.free(); vc_qi.free(); vc_sc.free()
    out_ref.free(); out_bf16kv.free(); out_i8kv.free()
