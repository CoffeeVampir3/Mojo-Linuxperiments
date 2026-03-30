"""Optimal fused layer flow — numerical validation.

Proves that the fused flow (12 steps) produces identical results to the
unfused flow (15 steps) and both match the f32 reference.

Fusions validated:
  1. K gemm → RoPE → FWHT → quantize → cache (no bf16 K intermediate)
  2. V gemm → FWHT → quantize → cache (no bf16 V intermediate)
  3. Q RoPE fused inside attention (no separate RoPE pass)
  4. GATE+UP as one wider gemm (one activation read)

Usage: mojo design_analysis/optimal_flow.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]

comptime HIDDEN = 576
comptime NUM_HEADS = 9
comptime NUM_KV_HEADS = 3
comptime HEAD_DIM = HIDDEN // NUM_HEADS          # 64
comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM     # 192
comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS  # 3
comptime INTERMEDIATE = 1536
comptime CONTEXT = 32
comptime BLOCK = HEAD_DIM
comptime POS = CONTEXT - 1


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
# Core ops
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

def channelwise_quantize(src: Ptr, qi: Ptr, scales: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        quantize_vec(src + r * cols, qi + r * cols, scales + r, cols)

def absorb_gamma(weight: Ptr, gamma: Ptr, rows: Int, cols: Int):
    for n in range(rows):
        for k in range(cols):
            weight[n * cols + k] = weight[n * cols + k] * gamma[k]

def apply_rope_head(vec: Ptr, head_dim: Int, pos: Int):
    """Apply RoPE to a single head's vector."""
    var half = head_dim // 2
    for i in range(half):
        var freq = 1.0 / Float32(10000.0 ** (Float64(2 * i) / Float64(head_dim)))
        var angle = Float32(pos) * freq
        # Scalar sin/cos via Taylor
        var a = Float64(angle)
        var a2 = a * a
        var cos_v = Float32(1.0 - a2/2.0 + a2*a2/24.0 - a2*a2*a2/720.0)
        var sin_v = Float32(a - a2*a/6.0 + a2*a2*a/120.0 - a2*a2*a2*a/5040.0)
        var x0 = vec[i]
        var x1 = vec[half + i]
        vec[i] = x0 * cos_v - x1 * sin_v
        vec[half + i] = x0 * sin_v + x1 * cos_v

def apply_rope(vec: Ptr, head_dim: Int, num_heads: Int, pos: Int):
    for h in range(num_heads):
        apply_rope_head(vec + h * head_dim, head_dim, pos)

def rms_fwht_quantize(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                       rows: Int, cols: Int, block: Int):
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
    for r in range(rows):
        for k in range(cols):
            var g = gate[r * cols + k]
            scratch[r * cols + k] = g * (1.0 / (1.0 + expf(-g))) * up[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    channelwise_quantize(scratch, qi, scales, rows, cols)

def int8_gemm(a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr, dst: Ptr,
              m: Int, n: Int, k: Int):
    for i in range(m):
        for j in range(n):
            var acc = 0
            for kk in range(k):
                acc += Int(a_qi[i * k + kk]) * Int(w_qi[j * k + kk])
            dst[i * n + j] = Float32(acc) * a_sc[i] * w_sc[j]

def f32_matmul(inp: Ptr, weight: Ptr, dst: Ptr, m: Int, n: Int, k: Int):
    for i in range(m):
        for j in range(n):
            var acc = Float32(0)
            for kk in range(k):
                acc += inp[i * k + kk] * weight[j * k + kk]
            dst[i * n + j] = acc

def f32_elem_add(a: Ptr, b: Ptr, dst: Ptr, count: Int):
    for i in range(count): dst[i] = a[i] + b[i]


# ============================================================================
# F32 reference attention (with RoPE)
# ============================================================================

def f32_attention(q: Ptr, k_cache: Ptr, v_cache: Ptr, dst: Ptr, ctx: Int):
    var scores = alloc[Float32](ctx)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))
    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        for t in range(ctx):
            var acc = Float32(0)
            for d in range(HEAD_DIM):
                acc += q[h * HEAD_DIM + d] * k_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            scores[t] = acc * inv_sqrt
        var max_s = scores[0]
        for t in range(1, ctx):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(ctx):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(ctx): scores[t] = scores[t] / sum_e
        for d in range(HEAD_DIM):
            var acc = Float32(0)
            for t in range(ctx):
                acc += scores[t] * v_cache[t * KV_HIDDEN + g * HEAD_DIM + d]
            dst[h * HEAD_DIM + d] = acc
    scores.free()


# ============================================================================
# UNFUSED int8 attention (15-step, our current implementation)
# ============================================================================

def int8_attention_unfused(
    q_bf16: Ptr,       # bf16 Q with RoPE already applied
    k_qi_cache: Ptr, k_sc_cache: Ptr,
    v_qi_cache: Ptr, v_sc_cache: Ptr,
    qi_out: Ptr, sc_out: Ptr,
    ctx: Int,
):
    """Steps 5-7 unfused: separate RoPE on Q, then attention with int8 output."""
    var head_buf = alloc[Float32](HEAD_DIM)
    var head_qi = alloc[Float32](HEAD_DIM)
    var head_sc = alloc[Float32](1)
    var scores = alloc[Float32](ctx)
    var w_qi = alloc[Float32](ctx)
    var w_sc = alloc[Float32](1)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))
    var out_buf = alloc[Float32](HIDDEN)

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        # FWHT + quantize Q (RoPE already applied externally)
        for d in range(HEAD_DIM): head_buf[d] = q_bf16[h * HEAD_DIM + d]
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, head_qi, head_sc, HEAD_DIM)
        # Int8 scoring
        for t in range(ctx):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi_cache[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc_cache[t * NUM_KV_HEADS + g] * inv_sqrt
        # Softmax
        var max_s = scores[0]
        for t in range(1, ctx):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(ctx):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(ctx): scores[t] = scores[t] / sum_e
        # V scale absorption + quantize weights
        for t in range(ctx):
            scores[t] = scores[t] * v_sc_cache[t * NUM_KV_HEADS + g]
        quantize_vec(scores, w_qi, w_sc, ctx)
        # Int8 V aggregation → f32 rotated output
        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(ctx):
                acc += Int(w_qi[t]) * Int(v_qi_cache[t * KV_HIDDEN + g * HEAD_DIM + d])
            out_buf[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    # Quantize full output to int8 (from f32, in rotated domain)
    channelwise_quantize(out_buf, qi_out, sc_out, 1, HIDDEN)

    head_buf.free(); head_qi.free(); head_sc.free()
    scores.free(); w_qi.free(); w_sc.free(); out_buf.free()


# ============================================================================
# FUSED int8 attention (RoPE on Q fused inside, int8 output directly)
# ============================================================================

def int8_attention_fused(
    q_bf16: Ptr,       # bf16 Q WITHOUT RoPE (applied inside)
    k_qi_cache: Ptr, k_sc_cache: Ptr,
    v_qi_cache: Ptr, v_sc_cache: Ptr,
    qi_out: Ptr, sc_out: Ptr,
    ctx: Int, query_pos: Int,
):
    """Fused step 5+7: RoPE on Q + attention + int8 output, all in one kernel."""
    var head_buf = alloc[Float32](HEAD_DIM)
    var head_qi = alloc[Float32](HEAD_DIM)
    var head_sc = alloc[Float32](1)
    var scores = alloc[Float32](ctx)
    var w_qi = alloc[Float32](ctx)
    var w_sc = alloc[Float32](1)
    var inv_sqrt = 1.0 / sqrt(Float32(HEAD_DIM))
    var out_buf = alloc[Float32](HIDDEN)

    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        # Load Q head, apply RoPE, FWHT, quantize — all fused
        for d in range(HEAD_DIM): head_buf[d] = q_bf16[h * HEAD_DIM + d]
        apply_rope_head(head_buf, HEAD_DIM, query_pos)
        fwht_inplace(head_buf, HEAD_DIM)
        quantize_vec(head_buf, head_qi, head_sc, HEAD_DIM)
        # Int8 scoring (same as unfused from here)
        for t in range(ctx):
            var acc = 0
            var k_off = t * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                acc += Int(head_qi[d]) * Int(k_qi_cache[k_off + d])
            scores[t] = Float32(acc) * head_sc[0] * k_sc_cache[t * NUM_KV_HEADS + g] * inv_sqrt
        var max_s = scores[0]
        for t in range(1, ctx):
            if scores[t] > max_s: max_s = scores[t]
        var sum_e = Float32(0)
        for t in range(ctx):
            scores[t] = expf(scores[t] - max_s)
            sum_e += scores[t]
        for t in range(ctx): scores[t] = scores[t] / sum_e
        for t in range(ctx):
            scores[t] = scores[t] * v_sc_cache[t * NUM_KV_HEADS + g]
        quantize_vec(scores, w_qi, w_sc, ctx)
        for d in range(HEAD_DIM):
            var acc = 0
            for t in range(ctx):
                acc += Int(w_qi[t]) * Int(v_qi_cache[t * KV_HIDDEN + g * HEAD_DIM + d])
            out_buf[h * HEAD_DIM + d] = Float32(acc) * w_sc[0]

    # Quantize directly from f32 (same as unfused)
    channelwise_quantize(out_buf, qi_out, sc_out, 1, HIDDEN)

    head_buf.free(); head_qi.free(); head_sc.free()
    scores.free(); w_qi.free(); w_sc.free(); out_buf.free()


# ============================================================================
# Fused K gemm-to-cache: gemm epilogue → RoPE → FWHT → quantize → cache
# ============================================================================

def int8_gemm_k_to_cache(
    a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr,
    k_qi_cache: Ptr, k_sc_cache: Ptr,
    pos: Int, k_dim: Int,
):
    """Fused: int8_gemm for K → RoPE per head → FWHT per head → quantize → cache.
    K never materializes as bf16."""
    var head_buf = alloc[Float32](HEAD_DIM)
    for g in range(NUM_KV_HEADS):
        # Compute gemm output for this head's rows (HEAD_DIM output elements)
        for d in range(HEAD_DIM):
            var row = g * HEAD_DIM + d
            var acc = 0
            for kk in range(k_dim):
                acc += Int(a_qi[kk]) * Int(w_qi[row * k_dim + kk])
            head_buf[d] = Float32(acc) * a_sc[0] * w_sc[row]
        # RoPE + FWHT + quantize → directly to cache
        apply_rope_head(head_buf, HEAD_DIM, pos)
        fwht_inplace(head_buf, HEAD_DIM)
        var off = pos * KV_HIDDEN + g * HEAD_DIM
        quantize_vec(head_buf, k_qi_cache + off, k_sc_cache + pos * NUM_KV_HEADS + g, HEAD_DIM)
    head_buf.free()


# ============================================================================
# Fused V gemm-to-cache: gemm epilogue → FWHT → quantize → cache
# ============================================================================

def int8_gemm_v_to_cache(
    a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr,
    v_qi_cache: Ptr, v_sc_cache: Ptr,
    pos: Int, k_dim: Int,
):
    """Fused: int8_gemm for V → FWHT per head → quantize → cache.
    V never materializes as bf16."""
    var head_buf = alloc[Float32](HEAD_DIM)
    for g in range(NUM_KV_HEADS):
        for d in range(HEAD_DIM):
            var row = g * HEAD_DIM + d
            var acc = 0
            for kk in range(k_dim):
                acc += Int(a_qi[kk]) * Int(w_qi[row * k_dim + kk])
            head_buf[d] = Float32(acc) * a_sc[0] * w_sc[row]
        fwht_inplace(head_buf, HEAD_DIM)
        var off = pos * KV_HIDDEN + g * HEAD_DIM
        quantize_vec(head_buf, v_qi_cache + off, v_sc_cache + pos * NUM_KV_HEADS + g, HEAD_DIM)
    head_buf.free()


# ============================================================================
# Fused GATE+UP gemm (one activation read, two outputs)
# ============================================================================

def int8_gemm_gate_up(
    a_qi: Ptr, gate_qi: Ptr, up_qi: Ptr,
    a_sc: Ptr, gate_sc: Ptr, up_sc: Ptr,
    gate_out: Ptr, up_out: Ptr,
    m: Int, n: Int, k: Int,
):
    """Fused GATE+UP: reads activation once, produces both outputs."""
    for i in range(m):
        for j in range(n):
            var acc_g = 0
            var acc_u = 0
            for kk in range(k):
                var a = Int(a_qi[i * k + kk])
                acc_g += a * Int(gate_qi[j * k + kk])
                acc_u += a * Int(up_qi[j * k + kk])
            gate_out[i * n + j] = Float32(acc_g) * a_sc[i] * gate_sc[j]
            up_out[i * n + j] = Float32(acc_u) * a_sc[i] * up_sc[j]


# ============================================================================
# Full layer: unfused (current 15-step flow)
# ============================================================================

def layer_unfused(
    x: Ptr, scratch: Ptr,
    qi_q: Ptr, qi_k: Ptr, qi_v: Ptr, qi_o: Ptr,
    qi_gate: Ptr, qi_up: Ptr, qi_down: Ptr,
    s_q: Ptr, s_k: Ptr, s_v: Ptr, s_o: Ptr,
    s_gate: Ptr, s_up: Ptr, s_down: Ptr,
    kc_qi: Ptr, kc_sc: Ptr, vc_qi: Ptr, vc_sc: Ptr,
    qi_act: Ptr, sc_act: Ptr,
    q_buf: Ptr, k_buf: Ptr, v_buf: Ptr,
    attn_buf: Ptr, gate_buf: Ptr, up_buf: Ptr, x_res: Ptr,
):
    """15-step unfused flow."""
    # 1. rms_fwht_quantize
    rms_fwht_quantize(x, scratch, qi_act, sc_act, 1, HIDDEN, BLOCK)
    # 2-4. Q, K, V gemms
    int8_gemm(qi_act, qi_q, sc_act, s_q, q_buf, 1, HIDDEN, HIDDEN)
    int8_gemm(qi_act, qi_k, sc_act, s_k, k_buf, 1, KV_HIDDEN, HIDDEN)
    int8_gemm(qi_act, qi_v, sc_act, s_v, v_buf, 1, KV_HIDDEN, HIDDEN)
    # 5. RoPE on Q and K
    apply_rope(q_buf, HEAD_DIM, NUM_HEADS, POS)
    apply_rope(k_buf, HEAD_DIM, NUM_KV_HEADS, POS)
    # 6a. K cache write: FWHT + quantize
    for g in range(NUM_KV_HEADS):
        var off = POS * KV_HIDDEN + g * HEAD_DIM
        var head = alloc[Float32](HEAD_DIM)
        for d in range(HEAD_DIM): head[d] = k_buf[g * HEAD_DIM + d]
        fwht_inplace(head, HEAD_DIM)
        quantize_vec(head, kc_qi + off, kc_sc + POS * NUM_KV_HEADS + g, HEAD_DIM)
        head.free()
    # 6b. V cache write: FWHT + quantize
    for g in range(NUM_KV_HEADS):
        var off = POS * KV_HIDDEN + g * HEAD_DIM
        var head = alloc[Float32](HEAD_DIM)
        for d in range(HEAD_DIM): head[d] = v_buf[g * HEAD_DIM + d]
        fwht_inplace(head, HEAD_DIM)
        quantize_vec(head, vc_qi + off, vc_sc + POS * NUM_KV_HEADS + g, HEAD_DIM)
        head.free()
    # 7. Attention (Q already has RoPE) → int8 output
    int8_attention_unfused(q_buf, kc_qi, kc_sc, vc_qi, vc_sc, qi_act, sc_act, CONTEXT)
    # 8. O gemm
    int8_gemm(qi_act, qi_o, sc_act, s_o, x_res, 1, HIDDEN, HIDDEN)
    # 9. Residual
    f32_elem_add(x, x_res, x, HIDDEN)
    # 10. rms_fwht_quantize
    rms_fwht_quantize(x, scratch, qi_act, sc_act, 1, HIDDEN, BLOCK)
    # 11-12. GATE, UP gemms (separate)
    int8_gemm(qi_act, qi_gate, sc_act, s_gate, gate_buf, 1, INTERMEDIATE, HIDDEN)
    int8_gemm(qi_act, qi_up, sc_act, s_up, up_buf, 1, INTERMEDIATE, HIDDEN)
    # 13. silu_fwht_quantize
    silu_fwht_quantize(gate_buf, up_buf, scratch, qi_act, sc_act, 1, INTERMEDIATE, BLOCK)
    # 14. DOWN gemm
    int8_gemm(qi_act, qi_down, sc_act, s_down, x_res, 1, HIDDEN, INTERMEDIATE)
    # 15. Residual
    f32_elem_add(x, x_res, x, HIDDEN)


# ============================================================================
# Full layer: fused (optimal 12-step flow)
# ============================================================================

def layer_fused(
    x: Ptr, scratch: Ptr,
    qi_q: Ptr, qi_k: Ptr, qi_v: Ptr, qi_o: Ptr,
    qi_gate: Ptr, qi_up: Ptr, qi_down: Ptr,
    s_q: Ptr, s_k: Ptr, s_v: Ptr, s_o: Ptr,
    s_gate: Ptr, s_up: Ptr, s_down: Ptr,
    kc_qi: Ptr, kc_sc: Ptr, vc_qi: Ptr, vc_sc: Ptr,
    qi_act: Ptr, sc_act: Ptr,
    q_buf: Ptr, gate_buf: Ptr, up_buf: Ptr, x_res: Ptr,
):
    """12-step fused flow. No k_buf, v_buf, no separate RoPE on Q."""
    # 1. rms_fwht_quantize
    rms_fwht_quantize(x, scratch, qi_act, sc_act, 1, HIDDEN, BLOCK)
    # 2. Q gemm (only Q materializes as bf16)
    int8_gemm(qi_act, qi_q, sc_act, s_q, q_buf, 1, HIDDEN, HIDDEN)
    # 3. K gemm → RoPE → FWHT → cache (no bf16 K)
    int8_gemm_k_to_cache(qi_act, qi_k, sc_act, s_k, kc_qi, kc_sc, POS, HIDDEN)
    # 4. V gemm → FWHT → cache (no bf16 V)
    int8_gemm_v_to_cache(qi_act, qi_v, sc_act, s_v, vc_qi, vc_sc, POS, HIDDEN)
    # 5. Attention with fused RoPE on Q → int8 output
    int8_attention_fused(q_buf, kc_qi, kc_sc, vc_qi, vc_sc, qi_act, sc_act, CONTEXT, POS)
    # 6. O gemm
    int8_gemm(qi_act, qi_o, sc_act, s_o, x_res, 1, HIDDEN, HIDDEN)
    # 7. Residual
    f32_elem_add(x, x_res, x, HIDDEN)
    # 8. rms_fwht_quantize
    rms_fwht_quantize(x, scratch, qi_act, sc_act, 1, HIDDEN, BLOCK)
    # 9. Fused GATE+UP gemm (one activation read)
    int8_gemm_gate_up(qi_act, qi_gate, qi_up, sc_act, s_gate, s_up,
                       gate_buf, up_buf, 1, INTERMEDIATE, HIDDEN)
    # 10. silu_fwht_quantize
    silu_fwht_quantize(gate_buf, up_buf, scratch, qi_act, sc_act, 1, INTERMEDIATE, BLOCK)
    # 11. DOWN gemm
    int8_gemm(qi_act, qi_down, sc_act, s_down, x_res, 1, HIDDEN, INTERMEDIATE)
    # 12. Residual
    f32_elem_add(x, x_res, x, HIDDEN)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Optimal Fused Flow — Numerical Validation")
    print("=" * 70)
    print("  HIDDEN=" + String(HIDDEN) + " HEADS=" + String(NUM_HEADS)
          + " KV_HEADS=" + String(NUM_KV_HEADS) + " HEAD_DIM=" + String(HEAD_DIM))
    print("  INTERMEDIATE=" + String(INTERMEDIATE)
          + " CONTEXT=" + String(CONTEXT) + " BLOCK=" + String(BLOCK))
    print("")

    var rng = Rng(seed=42)

    # Allocate weights
    var w_q = alloc[Float32](HIDDEN * HIDDEN)
    var w_k = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_v = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_o = alloc[Float32](HIDDEN * HIDDEN)
    var w_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_up = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var gamma_in = alloc[Float32](HIDDEN)
    var gamma_post = alloc[Float32](HIDDEN)

    # Generate weights
    for i in range(HIDDEN * HIDDEN): w_q[i] = rng.normal() * 0.02
    for i in range(KV_HIDDEN * HIDDEN): w_k[i] = rng.normal() * 0.02
    for i in range(KV_HIDDEN * HIDDEN): w_v[i] = rng.normal() * 0.02
    for i in range(HIDDEN * HIDDEN): w_o[i] = rng.normal() * 0.02
    for i in range(INTERMEDIATE * HIDDEN): w_gate[i] = rng.normal() * 0.02
    for i in range(INTERMEDIATE * HIDDEN): w_up[i] = rng.normal() * 0.02
    for i in range(HIDDEN * INTERMEDIATE): w_down[i] = rng.normal() * 0.02
    for i in range(HIDDEN):
        gamma_in[i] = 0.8 + rng.normal() * 0.1
        gamma_post[i] = 0.8 + rng.normal() * 0.1

    # Input and KV cache
    var x_init = alloc[Float32](HIDDEN)
    for i in range(HIDDEN): x_init[i] = rng.normal()

    # Build shared KV cache (positions 0..POS-1, shared between all paths)
    var kc_qi = alloc[Float32](CONTEXT * KV_HIDDEN)
    var kc_sc = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    var vc_qi = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_sc = alloc[Float32](CONTEXT * NUM_KV_HEADS)

    # Pre-fill cache with random rotated int8 KV entries
    var head_buf = alloc[Float32](HEAD_DIM)
    for pos in range(POS):
        for g in range(NUM_KV_HEADS):
            for d in range(HEAD_DIM): head_buf[d] = rng.normal() * 0.5
            apply_rope_head(head_buf, HEAD_DIM, pos)
            fwht_inplace(head_buf, HEAD_DIM)
            var off = pos * KV_HIDDEN + g * HEAD_DIM
            quantize_vec(head_buf, kc_qi + off, kc_sc + pos * NUM_KV_HEADS + g, HEAD_DIM)
            for d in range(HEAD_DIM): head_buf[d] = rng.normal() * 0.5
            fwht_inplace(head_buf, HEAD_DIM)
            quantize_vec(head_buf, vc_qi + off, vc_sc + pos * NUM_KV_HEADS + g, HEAD_DIM)
    head_buf.free()

    # Offline: absorb gamma + FWHT + quantize weights
    absorb_gamma(w_q, gamma_in, HIDDEN, HIDDEN)
    absorb_gamma(w_k, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_v, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_gate, gamma_post, INTERMEDIATE, HIDDEN)
    absorb_gamma(w_up, gamma_post, INTERMEDIATE, HIDDEN)
    fwht_rows(w_q, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_k, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_v, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_o, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_gate, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_up, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_down, HIDDEN, INTERMEDIATE, BLOCK)
    var qi_q = alloc[Float32](HIDDEN * HIDDEN)
    var qi_k = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_v = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_o = alloc[Float32](HIDDEN * HIDDEN)
    var qi_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_up = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_down = alloc[Float32](HIDDEN * INTERMEDIATE)
    var s_q = alloc[Float32](HIDDEN)
    var s_k = alloc[Float32](KV_HIDDEN)
    var s_v = alloc[Float32](KV_HIDDEN)
    var s_o = alloc[Float32](HIDDEN)
    var s_gate = alloc[Float32](INTERMEDIATE)
    var s_up = alloc[Float32](INTERMEDIATE)
    var s_down = alloc[Float32](HIDDEN)
    channelwise_quantize(w_q, qi_q, s_q, HIDDEN, HIDDEN)
    channelwise_quantize(w_k, qi_k, s_k, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_v, qi_v, s_v, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_o, qi_o, s_o, HIDDEN, HIDDEN)
    channelwise_quantize(w_gate, qi_gate, s_gate, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_up, qi_up, s_up, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_down, qi_down, s_down, HIDDEN, INTERMEDIATE)

    # Scratch buffers
    var scratch = alloc[Float32](INTERMEDIATE)
    var qi_act = alloc[Float32](INTERMEDIATE)
    var sc_act = alloc[Float32](1)
    var q_buf = alloc[Float32](HIDDEN)
    var k_buf = alloc[Float32](KV_HIDDEN)
    var v_buf = alloc[Float32](KV_HIDDEN)
    var attn_buf = alloc[Float32](HIDDEN)
    var gate_buf = alloc[Float32](INTERMEDIATE)
    var up_buf = alloc[Float32](INTERMEDIATE)
    var x_res = alloc[Float32](HIDDEN)

    # ==================================================================
    # F32 reference layer
    # ==================================================================
    var x_ref = alloc[Float32](HIDDEN)
    for i in range(HIDDEN): x_ref[i] = x_init[i]

    # Build f32 reference KV cache at POS
    var kc_ref = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_ref = alloc[Float32](CONTEXT * KV_HIDDEN)
    # Dequantize the shared cache for f32 reference
    for pos in range(POS):
        for g in range(NUM_KV_HEADS):
            var off = pos * KV_HIDDEN + g * HEAD_DIM
            for d in range(HEAD_DIM):
                kc_ref[off + d] = kc_qi[off + d] * kc_sc[pos * NUM_KV_HEADS + g]
                vc_ref[off + d] = vc_qi[off + d] * vc_sc[pos * NUM_KV_HEADS + g]
            # Un-rotate for f32 reference
            fwht_inplace(kc_ref + off, HEAD_DIM)
            fwht_inplace(vc_ref + off, HEAD_DIM)

    # F32 reference layer
    var f32_norm = alloc[Float32](HIDDEN)
    var ss = Float32(0)
    for k in range(HIDDEN): ss += x_ref[k] * x_ref[k]
    var inv_rms = 1.0 / sqrt(ss / Float32(HIDDEN) + 1e-5)
    for k in range(HIDDEN): f32_norm[k] = x_ref[k] * inv_rms * gamma_in[k]
    f32_matmul(f32_norm, w_q, q_buf, 1, HIDDEN, HIDDEN)
    f32_matmul(f32_norm, w_k, k_buf, 1, KV_HIDDEN, HIDDEN)
    f32_matmul(f32_norm, w_v, v_buf, 1, KV_HIDDEN, HIDDEN)
    apply_rope(q_buf, HEAD_DIM, NUM_HEADS, POS)
    apply_rope(k_buf, HEAD_DIM, NUM_KV_HEADS, POS)
    for i in range(KV_HIDDEN):
        kc_ref[POS * KV_HIDDEN + i] = k_buf[i]
        vc_ref[POS * KV_HIDDEN + i] = v_buf[i]
    f32_attention(q_buf, kc_ref, vc_ref, attn_buf, CONTEXT)
    f32_matmul(attn_buf, w_o, x_res, 1, HIDDEN, HIDDEN)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    ss = Float32(0)
    for k in range(HIDDEN): ss += x_ref[k] * x_ref[k]
    inv_rms = 1.0 / sqrt(ss / Float32(HIDDEN) + 1e-5)
    for k in range(HIDDEN): f32_norm[k] = x_ref[k] * inv_rms * gamma_post[k]
    f32_matmul(f32_norm, w_gate, gate_buf, 1, INTERMEDIATE, HIDDEN)
    f32_matmul(f32_norm, w_up, up_buf, 1, INTERMEDIATE, HIDDEN)
    for i in range(INTERMEDIATE):
        var g = gate_buf[i]
        gate_buf[i] = g * (1.0 / (1.0 + expf(-g))) * up_buf[i]
    f32_matmul(gate_buf, w_down, x_res, 1, HIDDEN, INTERMEDIATE)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    f32_norm.free()

    # ==================================================================
    # Unfused int8 layer (15 steps)
    # ==================================================================
    var x_unfused = alloc[Float32](HIDDEN)
    for i in range(HIDDEN): x_unfused[i] = x_init[i]
    # Copy KV cache (each path writes POS entry)
    var kc_qi_uf = alloc[Float32](CONTEXT * KV_HIDDEN)
    var kc_sc_uf = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    var vc_qi_uf = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_sc_uf = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    for i in range(CONTEXT * KV_HIDDEN): kc_qi_uf[i] = kc_qi[i]
    for i in range(CONTEXT * NUM_KV_HEADS): kc_sc_uf[i] = kc_sc[i]
    for i in range(CONTEXT * KV_HIDDEN): vc_qi_uf[i] = vc_qi[i]
    for i in range(CONTEXT * NUM_KV_HEADS): vc_sc_uf[i] = vc_sc[i]

    layer_unfused(x_unfused, scratch,
                  qi_q, qi_k, qi_v, qi_o, qi_gate, qi_up, qi_down,
                  s_q, s_k, s_v, s_o, s_gate, s_up, s_down,
                  kc_qi_uf, kc_sc_uf, vc_qi_uf, vc_sc_uf,
                  qi_act, sc_act, q_buf, k_buf, v_buf, attn_buf,
                  gate_buf, up_buf, x_res)

    # ==================================================================
    # Fused int8 layer (12 steps)
    # ==================================================================
    var x_fused = alloc[Float32](HIDDEN)
    for i in range(HIDDEN): x_fused[i] = x_init[i]
    var kc_qi_f = alloc[Float32](CONTEXT * KV_HIDDEN)
    var kc_sc_f = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    var vc_qi_f = alloc[Float32](CONTEXT * KV_HIDDEN)
    var vc_sc_f = alloc[Float32](CONTEXT * NUM_KV_HEADS)
    for i in range(CONTEXT * KV_HIDDEN): kc_qi_f[i] = kc_qi[i]
    for i in range(CONTEXT * NUM_KV_HEADS): kc_sc_f[i] = kc_sc[i]
    for i in range(CONTEXT * KV_HIDDEN): vc_qi_f[i] = vc_qi[i]
    for i in range(CONTEXT * NUM_KV_HEADS): vc_sc_f[i] = vc_sc[i]

    layer_fused(x_fused, scratch,
                qi_q, qi_k, qi_v, qi_o, qi_gate, qi_up, qi_down,
                s_q, s_k, s_v, s_o, s_gate, s_up, s_down,
                kc_qi_f, kc_sc_f, vc_qi_f, vc_sc_f,
                qi_act, sc_act, q_buf, gate_buf, up_buf, x_res)

    # ==================================================================
    # Compare
    # ==================================================================

    var sig = Float32(0)
    var noise_uf = Float32(0)
    var noise_f = Float32(0)
    var noise_diff = Float32(0)
    for i in range(HIDDEN):
        var rv = x_ref[i]
        sig += rv * rv
        var e_uf = rv - x_unfused[i]
        var e_f = rv - x_fused[i]
        var e_d = x_unfused[i] - x_fused[i]
        noise_uf += e_uf * e_uf
        noise_f += e_f * e_f
        noise_diff += e_d * e_d

    print("  Layer output SQNR vs f32 reference:")
    print("    unfused (15 steps):   " + d2(sqnr_db(sig, noise_uf)) + " dB")
    print("    fused   (12 steps):   " + d2(sqnr_db(sig, noise_f)) + " dB")
    print("    unfused vs fused:     " + d2(sqnr_db(sig, noise_diff)) + " dB")
    print("")

    var rms_uf = sqrt(noise_uf / Float32(HIDDEN))
    var rms_f = sqrt(noise_f / Float32(HIDDEN))
    var rms_sig = sqrt(sig / Float32(HIDDEN))
    print("  RMS error:")
    print("    unfused: " + d2(rms_uf) + "  (" + d2(rms_uf / rms_sig * 100.0) + "%)")
    print("    fused:   " + d2(rms_f) + "  (" + d2(rms_f / rms_sig * 100.0) + "%)")
    print("")

    # Check KV cache entries at POS match between unfused and fused
    var kv_diff = Float32(0)
    for i in range(KV_HIDDEN):
        var dk = kc_qi_uf[POS * KV_HIDDEN + i] - kc_qi_f[POS * KV_HIDDEN + i]
        var dv = vc_qi_uf[POS * KV_HIDDEN + i] - vc_qi_f[POS * KV_HIDDEN + i]
        kv_diff += dk * dk + dv * dv
    print("  KV cache at POS (unfused vs fused): " +
          ("IDENTICAL" if kv_diff == 0 else "DIFFER (diff=" + d2(kv_diff) + ")"))

    # Cleanup
    w_q.free(); w_k.free(); w_v.free(); w_o.free()
    w_gate.free(); w_up.free(); w_down.free()
    gamma_in.free(); gamma_post.free()
    x_init.free(); x_ref.free(); x_unfused.free(); x_fused.free()
    qi_q.free(); qi_k.free(); qi_v.free(); qi_o.free()
    qi_gate.free(); qi_up.free(); qi_down.free()
    s_q.free(); s_k.free(); s_v.free(); s_o.free()
    s_gate.free(); s_up.free(); s_down.free()
    kc_qi.free(); kc_sc.free(); vc_qi.free(); vc_sc.free()
    kc_qi_uf.free(); kc_sc_uf.free(); vc_qi_uf.free(); vc_sc_uf.free()
    kc_qi_f.free(); kc_sc_f.free(); vc_qi_f.free(); vc_sc_f.free()
    kc_ref.free(); vc_ref.free()
    scratch.free(); qi_act.free(); sc_act.free()
    q_buf.free(); k_buf.free(); v_buf.free()
    attn_buf.free(); gate_buf.free(); up_buf.free(); x_res.free()
