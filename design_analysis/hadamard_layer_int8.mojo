"""Hadamard int8 layer — full int8 path, deterministic vs stochastic rounding.

Runs the complete single-layer flow twice from the same input:
  1. Deterministic rounding (round-to-nearest)
  2. Stochastic rounding (probabilistic, unbiased)

Both paths use identical Hadamard rotation, int8 quantization, and int8
matmuls. The only difference is the rounding rule at quantization points.

Stochastic rounding: for a value v between integer levels floor(v) and
ceil(v), round up with probability (v - floor(v)), down otherwise.
This makes E[qi] = v (unbiased). For inner products, the error concentrates
as O(1/sqrt(d)) rather than worst-case O(1).

Usage: mojo design_analysis/hadamard_layer_int8.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]

comptime HIDDEN = 64
comptime NUM_HEADS = 4
comptime NUM_KV_HEADS = 2
comptime HEAD_DIM = HIDDEN // NUM_HEADS          # 16
comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM     # 32
comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS  # 2
comptime INTERMEDIATE = 128
comptime SEQ = 1
comptime BLOCK = HEAD_DIM                         # 16 — matches per-head rotation
comptime CONTEXT = 4                              # KV cache depth for test
comptime POS = 3                                  # current token position


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

def round_nearest(x: Float32) -> Float32:
    if x >= 0: return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))

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
# Scalar f32 reference ops
# ============================================================================

def f32_matmul(inp: Ptr, weight: Ptr, dst: Ptr, m: Int, n: Int, k: Int):
    for i in range(m):
        for j in range(n):
            var acc = Float32(0)
            for kk in range(k):
                acc += inp[i * k + kk] * weight[j * k + kk]
            dst[i * n + j] = acc

def f32_rmsnorm(inp: Ptr, gamma: Ptr, dst: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        var base = r * cols
        var ss = Float32(0)
        for k in range(cols): ss += inp[base + k] * inp[base + k]
        var inv_rms = 1.0 / sqrt(ss / Float32(cols) + 1e-5)
        for k in range(cols):
            dst[base + k] = inp[base + k] * inv_rms * gamma[k]

def f32_silu_mul(gate: Ptr, up: Ptr, dst: Ptr, count: Int):
    for i in range(count):
        var g = gate[i]
        dst[i] = g * (1.0 / (1.0 + expf(-g))) * up[i]

def f32_elem_add(a: Ptr, b: Ptr, dst: Ptr, count: Int):
    for i in range(count): dst[i] = a[i] + b[i]

def f32_gqa_attention(
    q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr, context_len: Int,
):
    """F32 GQA attention. q:[HIDDEN], cache:[T, KV_HIDDEN], dst:[HIDDEN]."""
    var scores = alloc[Float32](context_len)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))
    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        # Score
        for t in range(context_len):
            var acc = Float32(0)
            for d in range(HEAD_DIM):
                acc += q[h * HEAD_DIM + d] * k_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            scores[t] = acc * inv_sqrt
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
        # Weighted sum
        for d in range(HEAD_DIM):
            var acc = Float32(0)
            for t in range(context_len):
                acc += scores[t] * v_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            dst[h * HEAD_DIM + d] = acc
    scores.free()


# ============================================================================
# Quantization primitives
# ============================================================================

def quantize_vec(src: Ptr, qi: Ptr, scale: Ptr, count: Int):
    """Quantize a single vector to int8 with one scalar scale."""
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

def channelwise_quantize(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        quantize_vec(src + r * cols, qi + r * cols, scales + r, cols)


# ============================================================================
# Stochastic rounding variants
# ============================================================================

struct SrRng:
    """Fast PRNG for stochastic rounding. One u32 of entropy per element."""
    var state: UInt64
    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else UInt64(1)
    def next_f32(mut self) -> Float32:
        """Uniform [0, 1) via xorshift64."""
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return Float32(self.state & 0xFFFFFF) / 16777216.0


def quantize_vec_sr(src: Ptr, qi: Ptr, scale: Ptr, count: Int, mut sr: SrRng):
    """Stochastic rounding quantization.

    For value v between floor(v) and ceil(v):
      P(qi = ceil(v))  = v - floor(v)
      P(qi = floor(v)) = ceil(v) - v

    This makes E[qi] = v (unbiased). The per-element MSE is slightly
    higher than deterministic rounding, but the bias elimination gives
    O(1/sqrt(d)) error concentration for inner products.
    """
    var amax = Float32(0)
    for i in range(count):
        var a = absf(src[i])
        if a > amax: amax = a
    var sc = amax / 127.0
    scale[0] = sc
    var inv = 1.0 / sc if sc > 0 else Float32(0)
    for i in range(count):
        var v = src[i] * inv
        # Decompose into integer and fractional parts
        var fl = Float32(0)
        if v >= 0:
            fl = Float32(Int(v))          # floor for positive
        else:
            fl = -Float32(Int(-v))        # ceil toward zero for negative
            if fl > v:
                fl = fl - 1.0             # actual floor
        var frac = v - fl
        # Stochastic round: go up with probability = frac
        var qi_val = fl
        if sr.next_f32() < frac:
            qi_val = fl + 1.0
        if qi_val > 127.0: qi_val = 127.0
        elif qi_val < -128.0: qi_val = -128.0
        qi[i] = qi_val

def channelwise_quantize_sr(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int,
                             mut sr: SrRng):
    for r in range(rows):
        quantize_vec_sr(src + r * cols, qi + r * cols, scales + r, cols, sr)

def absorb_gamma(weight: Ptr, gamma: Ptr, rows: Int, cols: Int):
    for n in range(rows):
        for k in range(cols):
            weight[n * cols + k] = weight[n * cols + k] * gamma[k]


# ============================================================================
# Fused online operations
# ============================================================================

def rms_fwht_quantize(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                       rows: Int, cols: Int, block: Int):
    """Fused: RMS norm (no gamma) + FWHT + int8 quantize."""
    for r in range(rows):
        for k in range(cols): scratch[r * cols + k] = inp[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    for r in range(rows):
        var base = r * cols
        var ss = Float32(0)
        var amax = Float32(0)
        for k in range(cols):
            var v = scratch[base + k]
            ss += v * v
            var a = absf(v)
            if a > amax: amax = a
        var rms = sqrt(ss / Float32(cols) + 1e-5)
        var scale = amax / (rms * 127.0) if amax > 0 else Float32(0)
        scales[r] = scale
        var inv = 127.0 / amax if amax > 0 else Float32(0)
        for k in range(cols):
            var v = round_nearest(scratch[base + k] * inv)
            if v > 127.0: v = 127.0
            elif v < -128.0: v = -128.0
            qi[base + k] = v

def silu_fwht_quantize(gate: Ptr, up: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                        rows: Int, cols: Int, block: Int):
    """Fused: silu(gate)*up + FWHT + int8 quantize."""
    for r in range(rows):
        for k in range(cols):
            var g = gate[r * cols + k]
            scratch[r * cols + k] = g * (1.0 / (1.0 + expf(-g))) * up[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    channelwise_quantize(scratch, qi, scales, rows, cols)

def rms_fwht_quantize_sr(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                          rows: Int, cols: Int, block: Int, mut sr: SrRng):
    """Fused: RMS norm + FWHT + stochastic int8 quantize."""
    for r in range(rows):
        for k in range(cols): scratch[r * cols + k] = inp[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    for r in range(rows):
        var base = r * cols
        var ss = Float32(0)
        var amax = Float32(0)
        for k in range(cols):
            var v = scratch[base + k]
            ss += v * v
            var a = absf(v)
            if a > amax: amax = a
        var rms = sqrt(ss / Float32(cols) + 1e-5)
        var scale = amax / (rms * 127.0) if amax > 0 else Float32(0)
        scales[r] = scale
        var inv = 127.0 / amax if amax > 0 else Float32(0)
        for k in range(cols):
            var v = scratch[base + k] * inv
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

def silu_fwht_quantize_sr(gate: Ptr, up: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                            rows: Int, cols: Int, block: Int, mut sr: SrRng):
    """Fused: silu(gate)*up + FWHT + stochastic int8 quantize."""
    for r in range(rows):
        for k in range(cols):
            var g = gate[r * cols + k]
            scratch[r * cols + k] = g * (1.0 / (1.0 + expf(-g))) * up[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    channelwise_quantize_sr(scratch, qi, scales, rows, cols, sr)

def int8_gemm(a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr, dst: Ptr,
              m: Int, n: Int, k: Int):
    """Int8 matmul: i32 acc + epilogue rescale."""
    for i in range(m):
        for j in range(n):
            var acc = 0
            for kk in range(k):
                acc += Int(a_qi[i * k + kk]) * Int(w_qi[j * k + kk])
            dst[i * n + j] = Float32(acc) * a_sc[i] * w_sc[j]


# ============================================================================
# Int8 GQA attention
# ============================================================================

def int8_gqa_attention(
    q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr, context_len: Int,
):
    """Int8 GQA attention with per-head FWHT rotation.

    1. FWHT + quantize K cache entries per kv_head over head_dim
    2. FWHT + quantize V cache entries per kv_head over head_dim
    3. Per query head:
       a. FWHT + quantize Q over head_dim
       b. Int8 dot products for scores (Parseval: exact)
       c. Softmax (f32)
       d. Absorb V per-entry scales into attention weights
       e. Quantize absorbed weights
       f. Int8 weighted sum of V
    4. Output is in per-head rotated domain

    q:[HIDDEN], cache:[T, KV_HIDDEN], dst:[HIDDEN] (rotated).
    """
    # Working memory
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

    # Step 1: FWHT + quantize all K cache entries
    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                head_buf[d] = k_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec(head_buf, k_qi + off, k_sc + t * NUM_KV_HEADS + g, HEAD_DIM)

    # Step 2: FWHT + quantize all V cache entries
    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                head_buf[d] = v_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec(head_buf, v_qi + off, v_sc + t * NUM_KV_HEADS + g, HEAD_DIM)

    # Step 3: Per query head
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR

        # 3a: FWHT + quantize Q for this head
        for d in range(HEAD_DIM):
            head_buf[d] = q[h * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, head_qi, head_sc, HEAD_DIM)

        # 3b: Int8 dot products for scores
        for t in range(context_len):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc[t * NUM_KV_HEADS + g] * inv_sqrt

        # 3c: Softmax
        var max_s = scores[0]
        for t in range(1, context_len):
            if scores[t] > max_s:
                max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(context_len):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(context_len):
            scores[t] = scores[t] / sum_e

        # 3d: Absorb V per-entry scales into attention weights
        for t in range(context_len):
            scores[t] = scores[t] * v_sc[t * NUM_KV_HEADS + g]

        # 3e: Quantize absorbed weights
        quantize_vec(scores, w_qi, w_sc, context_len)

        # 3f: Int8 weighted sum of V (output in rotated domain)
        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(context_len):
                acc += Int(w_qi[t]) * Int(v_qi[t * KV_HIDDEN + g * HEAD_DIM + d])
            dst[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    head_buf.free()
    head_qi.free()
    head_sc.free()
    k_qi.free()
    k_sc.free()
    v_qi.free()
    v_sc.free()
    scores.free()
    w_qi.free()
    w_sc.free()


def int8_gqa_attention_sr(
    q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr, context_len: Int,
    mut sr: SrRng,
):
    """Int8 GQA attention with stochastic rounding at all quantization points."""
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

    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM): head_buf[d] = k_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec_sr(head_buf, k_qi + off, k_sc + t * NUM_KV_HEADS + g, HEAD_DIM, sr)

    for t in range(context_len):
        for g in range(NUM_KV_HEADS):
            var off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM): head_buf[d] = v_cache[off + d]
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec_sr(head_buf, v_qi + off, v_sc + t * NUM_KV_HEADS + g, HEAD_DIM, sr)

    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))
    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        for d in range(HEAD_DIM): head_buf[d] = q[h * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec_sr(head_buf, head_qi, head_sc, HEAD_DIM, sr)

        for t in range(context_len):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc[t * NUM_KV_HEADS + g] * inv_sqrt

        var max_s = scores[0]
        for t in range(1, context_len):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(context_len):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(context_len):
            scores[t] = scores[t] / sum_e

        for t in range(context_len):
            scores[t] = scores[t] * v_sc[t * NUM_KV_HEADS + g]
        quantize_vec_sr(scores, w_qi, w_sc, context_len, sr)

        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(context_len):
                acc += Int(w_qi[t]) * Int(v_qi[t * KV_HIDDEN + g * HEAD_DIM + d])
            dst[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    head_buf.free(); head_qi.free(); head_sc.free()
    k_qi.free(); k_sc.free(); v_qi.free(); v_sc.free()
    scores.free(); w_qi.free(); w_sc.free()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Hadamard Int8 Layer — Full Int8 Path (incl. Attention)")
    print("=" * 70)
    print("  hidden=" + String(HIDDEN) + "  heads=" + String(NUM_HEADS)
          + "  kv_heads=" + String(NUM_KV_HEADS) + "  head_dim=" + String(HEAD_DIM))
    print("  intermediate=" + String(INTERMEDIATE) + "  block=" + String(BLOCK)
          + "  context=" + String(CONTEXT))
    print("")

    var rng = Rng(seed=7)

    # ------------------------------------------------------------------
    # Allocate weights
    # ------------------------------------------------------------------
    var w_q    = alloc[Float32](HIDDEN * HIDDEN)
    var w_k    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_v    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_o    = alloc[Float32](HIDDEN * HIDDEN)
    var w_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var qi_q    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_k    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_v    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_o    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var s_q    = alloc[Float32](HIDDEN)
    var s_k    = alloc[Float32](KV_HIDDEN)
    var s_v    = alloc[Float32](KV_HIDDEN)
    var s_o    = alloc[Float32](HIDDEN)
    var s_gate = alloc[Float32](INTERMEDIATE)
    var s_up   = alloc[Float32](INTERMEDIATE)
    var s_down = alloc[Float32](HIDDEN)
    var gamma_in   = alloc[Float32](HIDDEN)
    var gamma_post = alloc[Float32](HIDDEN)

    # Activation buffers
    var x_init  = alloc[Float32](HIDDEN)
    var x_ref   = alloc[Float32](HIDDEN)
    var x_had   = alloc[Float32](HIDDEN)
    var x_norm  = alloc[Float32](HIDDEN)
    var q_buf   = alloc[Float32](HIDDEN)
    var k_buf   = alloc[Float32](KV_HIDDEN)
    var v_buf   = alloc[Float32](KV_HIDDEN)
    var attn_buf = alloc[Float32](HIDDEN)
    var x_res   = alloc[Float32](HIDDEN)
    var gate_buf = alloc[Float32](INTERMEDIATE)
    var up_buf  = alloc[Float32](INTERMEDIATE)
    var scratch = alloc[Float32](INTERMEDIATE)
    var qi_act  = alloc[Float32](INTERMEDIATE)
    var sc_act  = alloc[Float32](SEQ)

    # KV caches (separate for ref/had since K,V projections differ slightly)
    var kc_ref = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_ref = alloc[Float32](CONTEXT * KV_HIDDEN)
    var kc_had = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_had = alloc[Float32](CONTEXT * KV_HIDDEN)

    # ------------------------------------------------------------------
    # Generate random data
    # ------------------------------------------------------------------
    for i in range(HIDDEN * HIDDEN):       w_q[i] = rng.normal() * 0.1
    for i in range(KV_HIDDEN * HIDDEN):    w_k[i] = rng.normal() * 0.1
    for i in range(KV_HIDDEN * HIDDEN):    w_v[i] = rng.normal() * 0.1
    for i in range(HIDDEN * HIDDEN):       w_o[i] = rng.normal() * 0.1
    for i in range(INTERMEDIATE * HIDDEN): w_gate[i] = rng.normal() * 0.05
    for i in range(INTERMEDIATE * HIDDEN): w_up[i] = rng.normal() * 0.05
    for i in range(HIDDEN * INTERMEDIATE): w_down[i] = rng.normal() * 0.05
    for i in range(HIDDEN):
        gamma_in[i] = 0.8 + rng.normal() * 0.1
        gamma_post[i] = 0.8 + rng.normal() * 0.1
    for i in range(HIDDEN):
        x_init[i] = rng.normal()
        x_ref[i] = x_init[i]
        x_had[i] = x_init[i]

    # Pre-fill KV cache positions 0..POS-1 with random data (shared)
    for i in range(POS * KV_HIDDEN):
        var kval = rng.normal()
        var vval = rng.normal()
        kc_ref[i] = kval
        kc_had[i] = kval
        vc_ref[i] = vval
        vc_had[i] = vval

    # ==================================================================
    # F32 REFERENCE LAYER
    # ==================================================================
    print("--- F32 Reference Layer ---")

    # Pre-attention norm → QKV
    f32_rmsnorm(x_ref, gamma_in, x_norm, SEQ, HIDDEN)
    f32_matmul(x_norm, w_q, q_buf, SEQ, HIDDEN, HIDDEN)
    f32_matmul(x_norm, w_k, k_buf, SEQ, KV_HIDDEN, HIDDEN)
    f32_matmul(x_norm, w_v, v_buf, SEQ, KV_HIDDEN, HIDDEN)
    print("  rmsnorm → Q K V")

    # Write K,V to cache at POS
    for i in range(KV_HIDDEN):
        kc_ref[POS * KV_HIDDEN + i] = k_buf[i]
        vc_ref[POS * KV_HIDDEN + i] = v_buf[i]

    # Attention over full context
    f32_gqa_attention(q_buf, kc_ref, vc_ref, attn_buf, CONTEXT)
    print("  attention (f32, T=" + String(CONTEXT) + ")")

    # O projection + residual
    f32_matmul(attn_buf, w_o, x_res, SEQ, HIDDEN, HIDDEN)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    print("  O + residual")

    # Pre-MLP norm → GATE UP → SiLU → DOWN + residual
    f32_rmsnorm(x_ref, gamma_post, x_norm, SEQ, HIDDEN)
    f32_matmul(x_norm, w_gate, gate_buf, SEQ, INTERMEDIATE, HIDDEN)
    f32_matmul(x_norm, w_up, up_buf, SEQ, INTERMEDIATE, HIDDEN)
    f32_silu_mul(gate_buf, up_buf, gate_buf, INTERMEDIATE)
    f32_matmul(gate_buf, w_down, x_res, SEQ, HIDDEN, INTERMEDIATE)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    print("  rmsnorm → GATE UP → silu → DOWN + residual")
    print("")

    # ==================================================================
    # OFFLINE: WEIGHT PREPARATION
    # ==================================================================
    print("--- Offline: Weight Preparation ---")

    absorb_gamma(w_q, gamma_in, HIDDEN, HIDDEN)
    absorb_gamma(w_k, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_v, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_gate, gamma_post, INTERMEDIATE, HIDDEN)
    absorb_gamma(w_up, gamma_post, INTERMEDIATE, HIDDEN)
    print("  Absorb gamma: Q K V GATE UP")

    fwht_rows(w_q, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_k, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_v, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_o, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_gate, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_up, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_down, HIDDEN, INTERMEDIATE, BLOCK)

    channelwise_quantize(w_q, qi_q, s_q, HIDDEN, HIDDEN)
    channelwise_quantize(w_k, qi_k, s_k, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_v, qi_v, s_v, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_o, qi_o, s_o, HIDDEN, HIDDEN)
    channelwise_quantize(w_gate, qi_gate, s_gate, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_up, qi_up, s_up, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_down, qi_down, s_down, HIDDEN, INTERMEDIATE)
    print("  FWHT + int8 quantize: Q K V O GATE UP DOWN")
    print("")

    # ==================================================================
    # HADAMARD INT8 — DETERMINISTIC ROUNDING
    # ==================================================================
    print("--- Hadamard Int8: Deterministic Rounding ---")

    rms_fwht_quantize(x_had, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK)
    int8_gemm(qi_act, qi_q, sc_act, s_q, q_buf, SEQ, HIDDEN, HIDDEN)
    int8_gemm(qi_act, qi_k, sc_act, s_k, k_buf, SEQ, KV_HIDDEN, HIDDEN)
    int8_gemm(qi_act, qi_v, sc_act, s_v, v_buf, SEQ, KV_HIDDEN, HIDDEN)
    for i in range(KV_HIDDEN):
        kc_had[POS * KV_HIDDEN + i] = k_buf[i]
        vc_had[POS * KV_HIDDEN + i] = v_buf[i]
    int8_gqa_attention(q_buf, kc_had, vc_had, attn_buf, CONTEXT)
    quantize_vec(attn_buf, qi_act, sc_act, HIDDEN)
    int8_gemm(qi_act, qi_o, sc_act, s_o, x_res, SEQ, HIDDEN, HIDDEN)
    f32_elem_add(x_had, x_res, x_had, HIDDEN)
    rms_fwht_quantize(x_had, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK)
    int8_gemm(qi_act, qi_gate, sc_act, s_gate, gate_buf, SEQ, INTERMEDIATE, HIDDEN)
    int8_gemm(qi_act, qi_up, sc_act, s_up, up_buf, SEQ, INTERMEDIATE, HIDDEN)
    silu_fwht_quantize(gate_buf, up_buf, scratch, qi_act, sc_act, SEQ, INTERMEDIATE, BLOCK)
    int8_gemm(qi_act, qi_down, sc_act, s_down, x_res, SEQ, HIDDEN, INTERMEDIATE)
    f32_elem_add(x_had, x_res, x_had, HIDDEN)
    print("  Full layer complete")

    var det_sig = Float32(0)
    var det_noise = Float32(0)
    var det_max = Float32(0)
    for i in range(HIDDEN):
        det_sig += x_ref[i] * x_ref[i]
        var err = x_ref[i] - x_had[i]
        det_noise += err * err
        var ae = absf(err)
        if ae > det_max: det_max = ae
    print("  SQNR = " + d2(sqnr_db(det_sig, det_noise)) + " dB"
          + "  max|err| = " + d2(det_max)
          + "  rms err/out = " + d2(sqrt(det_noise / det_sig) * 100) + "%")
    print("")

    # ==================================================================
    # HADAMARD INT8 — STOCHASTIC ROUNDING (run N trials, show statistics)
    # ==================================================================

    comptime NUM_TRIALS = 50

    print("--- Hadamard Int8: Stochastic Rounding (" + String(NUM_TRIALS) + " trials) ---")

    var x_sr = alloc[Float32](HIDDEN)
    var sr_sqnr_sum = Float32(0)
    var sr_sqnr_min = Float32(999.0)
    var sr_sqnr_max = Float32(-999.0)
    var sr_max_err_sum = Float32(0)
    var sr_rms_pct_sum = Float32(0)

    for trial in range(NUM_TRIALS):
        var sr = SrRng(seed=UInt64(1000 + trial))

        # Reset to same initial state
        for i in range(HIDDEN): x_sr[i] = x_init[i]

        # Re-fill had cache positions 0..POS-1 (same data as ref)
        for i in range(POS * KV_HIDDEN):
            kc_had[i] = kc_ref[i]
            vc_had[i] = vc_ref[i]

        # -- Attention block (stochastic rounding at all activation quant points) --
        rms_fwht_quantize_sr(x_sr, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK, sr)
        int8_gemm(qi_act, qi_q, sc_act, s_q, q_buf, SEQ, HIDDEN, HIDDEN)
        int8_gemm(qi_act, qi_k, sc_act, s_k, k_buf, SEQ, KV_HIDDEN, HIDDEN)
        int8_gemm(qi_act, qi_v, sc_act, s_v, v_buf, SEQ, KV_HIDDEN, HIDDEN)
        for i in range(KV_HIDDEN):
            kc_had[POS * KV_HIDDEN + i] = k_buf[i]
            vc_had[POS * KV_HIDDEN + i] = v_buf[i]
        int8_gqa_attention_sr(q_buf, kc_had, vc_had, attn_buf, CONTEXT, sr)
        quantize_vec_sr(attn_buf, qi_act, sc_act, HIDDEN, sr)
        int8_gemm(qi_act, qi_o, sc_act, s_o, x_res, SEQ, HIDDEN, HIDDEN)
        f32_elem_add(x_sr, x_res, x_sr, HIDDEN)

        # -- MLP block --
        rms_fwht_quantize_sr(x_sr, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK, sr)
        int8_gemm(qi_act, qi_gate, sc_act, s_gate, gate_buf, SEQ, INTERMEDIATE, HIDDEN)
        int8_gemm(qi_act, qi_up, sc_act, s_up, up_buf, SEQ, INTERMEDIATE, HIDDEN)
        silu_fwht_quantize_sr(gate_buf, up_buf, scratch, qi_act, sc_act, SEQ, INTERMEDIATE, BLOCK, sr)
        int8_gemm(qi_act, qi_down, sc_act, s_down, x_res, SEQ, HIDDEN, INTERMEDIATE)
        f32_elem_add(x_sr, x_res, x_sr, HIDDEN)

        # Measure
        var sig = Float32(0)
        var noise = Float32(0)
        var mx = Float32(0)
        for i in range(HIDDEN):
            sig += x_ref[i] * x_ref[i]
            var err = x_ref[i] - x_sr[i]
            noise += err * err
            var ae = absf(err)
            if ae > mx: mx = ae
        var db = sqnr_db(sig, noise)
        sr_sqnr_sum += db
        if db < sr_sqnr_min: sr_sqnr_min = db
        if db > sr_sqnr_max: sr_sqnr_max = db
        sr_max_err_sum += mx
        sr_rms_pct_sum += sqrt(noise / sig) * 100.0

    var sr_sqnr_mean = sr_sqnr_sum / Float32(NUM_TRIALS)
    var sr_max_mean = sr_max_err_sum / Float32(NUM_TRIALS)
    var sr_pct_mean = sr_rms_pct_sum / Float32(NUM_TRIALS)

    print("  SQNR:  mean=" + d2(sr_sqnr_mean)
          + "  min=" + d2(sr_sqnr_min)
          + "  max=" + d2(sr_sqnr_max) + " dB")
    print("  Max|err| mean  = " + d2(sr_max_mean))
    print("  RMS err/out    = " + d2(sr_pct_mean) + "% (mean over trials)")
    print("")

    x_sr.free()

    # ==================================================================
    # COMPARISON
    # ==================================================================
    print("=" * 70)
    print("  Deterministic vs Stochastic Rounding")
    print("=" * 70)
    print("")
    print("                          Deterministic    Stochastic (mean)")
    print("  SQNR (dB):              "
          + d2(sqnr_db(det_sig, det_noise))
          + "             " + d2(sr_sqnr_mean))
    print("  Max |error|:            "
          + d2(det_max)
          + "              " + d2(sr_max_mean))
    print("  RMS err / out:          "
          + d2(sqrt(det_noise / det_sig) * 100)
          + "%             " + d2(sr_pct_mean) + "%")
    print("")
    print("  Stochastic SQNR range over " + String(NUM_TRIALS) + " trials: "
          + d2(sr_sqnr_min) + " .. " + d2(sr_sqnr_max) + " dB")

    # cleanup
    w_q.free(); w_k.free(); w_v.free(); w_o.free()
    w_gate.free(); w_up.free(); w_down.free()
    qi_q.free(); qi_k.free(); qi_v.free(); qi_o.free()
    qi_gate.free(); qi_up.free(); qi_down.free()
    s_q.free(); s_k.free(); s_v.free(); s_o.free()
    s_gate.free(); s_up.free(); s_down.free()
    gamma_in.free(); gamma_post.free()
    x_init.free(); x_ref.free(); x_had.free(); x_norm.free()
    q_buf.free(); k_buf.free(); v_buf.free()
    attn_buf.free(); x_res.free()
    gate_buf.free(); up_buf.free(); scratch.free()
    qi_act.free(); sc_act.free()
    kc_ref.free(); vc_ref.free()
    kc_had.free(); vc_had.free()
