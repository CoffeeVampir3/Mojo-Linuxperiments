"""Hadamard int8 layer — operational flow stub.

Complete operation flow for one transformer layer using Hadamard-rotated
int8 quantization. SmolLM2 architecture at small scale for validation.

Offline:
  1. Absorb RMSNorm gamma into subsequent weights: W' = W * diag(gamma)
  2. FWHT each weight row (K dimension)
  3. Per-row int8 quantize

Online fused kernels (4 per layer):
  rms_fwht_quantize  — norm(no gamma) + FWHT + i8    [2 of 4 quant points]
  fwht_quantize      — FWHT + i8                      [1 of 4 quant points]
  silu_fwht_quantize — silu_mul + FWHT + i8           [1 of 4 quant points]

Online matmuls (7 per layer):
  int8_gemm — i32 accumulation + epilogue rescale, identical to channelwise

Unchanged bf16 ops:
  rope, kv_cache_write, attention, elem_add, allreduce

Usage: mojo design_analysis/hadamard_layer.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.math import sqrt
from std.time import perf_counter_ns
comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]


def expf(x: Float32) -> Float32:
    """Scalar exp via Cody-Waite range reduction + polynomial."""
    var xc = x
    if xc < Float32(-87.0): xc = Float32(-87.0)
    if xc > Float32(88.0): xc = Float32(88.0)
    var nf = xc * Float32(1.4426950409)
    if nf >= 0:
        nf = nf + Float32(0.5)
    else:
        nf = nf - Float32(0.5)
    var n = Int(nf)
    var r = xc - Float32(n) * Float32(0.6931458) - Float32(n) * Float32(1.4286068e-06)
    var p = Float32(1.0) + r * (Float32(1.0) + r * (Float32(0.5) + r * (
        Float32(0.16666667) + r * (Float32(0.041666668) + r * Float32(0.008333334)))))
    var exp_bits = UInt32(n + 127) << 23
    var pow2n = UnsafePointer(to=exp_bits).bitcast[Float32]()[]
    return p * pow2n

comptime HIDDEN = 64
comptime NUM_HEADS = 4
comptime NUM_KV_HEADS = 2
comptime HEAD_DIM = HIDDEN // NUM_HEADS          # 16
comptime KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM     # 32
comptime GQA_FACTOR = NUM_HEADS // NUM_KV_HEADS  # 2
comptime INTERMEDIATE = 128
comptime SEQ = 1
comptime BLOCK = 64


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
# Scalar reference operations (f32)
# ============================================================================

def f32_matmul(inp: Ptr, weight: Ptr, dst: Ptr, m: Int, n: Int, k: Int):
    """dst[m,n] = sum_k inp[m,k] * weight[n,k]."""
    for i in range(m):
        for j in range(n):
            var acc = Float32(0)
            for kk in range(k):
                acc += inp[i * k + kk] * weight[j * k + kk]
            dst[i * n + j] = acc

def f32_rmsnorm(inp: Ptr, gamma: Ptr, dst: Ptr, rows: Int, cols: Int):
    """dst = (inp / rms(inp)) * gamma, per row."""
    for r in range(rows):
        var base = r * cols
        var ss = Float32(0)
        for k in range(cols):
            ss += inp[base + k] * inp[base + k]
        var inv_rms = 1.0 / sqrt(ss / Float32(cols) + 1e-5)
        for k in range(cols):
            dst[base + k] = inp[base + k] * inv_rms * gamma[k]

def f32_silu_mul(gate: Ptr, up: Ptr, dst: Ptr, count: Int):
    for i in range(count):
        var g = gate[i]
        var sig = 1.0 / (1.0 + expf(-g))
        dst[i] = g * sig * up[i]

def f32_elem_add(a: Ptr, b: Ptr, dst: Ptr, count: Int):
    for i in range(count): dst[i] = a[i] + b[i]

def gqa_expand_v(v: Ptr, dst: Ptr):
    """Attention placeholder for seq=1: output = GQA-expanded V."""
    for h in range(NUM_HEADS):
        var g = h // GQA_FACTOR
        for d in range(HEAD_DIM):
            dst[h * HEAD_DIM + d] = v[g * HEAD_DIM + d]


# ============================================================================
# Weight preparation (offline)
# ============================================================================

def absorb_gamma(weight: Ptr, gamma: Ptr, rows: Int, cols: Int):
    """W'[n,k] = W[n,k] * gamma[k] — fold norm gain into weight columns."""
    for n in range(rows):
        for k in range(cols):
            weight[n * cols + k] = weight[n * cols + k] * gamma[k]

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


# ============================================================================
# Fused online operations
# ============================================================================

def rms_fwht_quantize(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                       rows: Int, cols: Int, block: Int):
    """Fused: RMS norm (no gamma) + FWHT + per-row int8 quantize.

    gamma is absent — absorbed into weights offline.
    rms is computed from FWHT output (Parseval: ||Hx|| = ||x||).
    rms cancels in qi values; encoded in scale instead:
        scale[r] = max(|FWHT(x[r])|) / (rms(x[r]) * 127)
        qi ≈ FWHT(x[r]) * 127 / max(|FWHT(x[r])|)

    Fusion: 1 read bf16, butterfly in registers, dual reduction
    (sum-of-squares + absmax), quantize, 1 write int8.
    """
    for r in range(rows):
        for k in range(cols):
            scratch[r * cols + k] = inp[r * cols + k]
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

def fwht_quantize(inp: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                   rows: Int, cols: Int, block: Int):
    """Fused: FWHT + per-row int8 quantize (no preceding norm).

    For quantization points without a preceding RMSNorm (before O_PROJ).
    Fusion: 1 read bf16, butterfly in registers, absmax + quantize, 1 write int8.
    """
    for r in range(rows):
        for k in range(cols):
            scratch[r * cols + k] = inp[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    channelwise_quantize(scratch, qi, scales, rows, cols)

def silu_fwht_quantize(gate: Ptr, up: Ptr, scratch: Ptr, qi: Ptr, scales: Ptr,
                        rows: Int, cols: Int, block: Int):
    """Fused: SiLU(gate)*up + FWHT + per-row int8 quantize.

    Fusion: read gate+up, compute silu_mul, butterfly in registers,
    absmax + quantize, 1 write int8. The nonlinearity and rotation
    share a single pass over the intermediate activation.
    """
    for r in range(rows):
        for k in range(cols):
            var g = gate[r * cols + k]
            var sig = 1.0 / (1.0 + expf(-g))
            scratch[r * cols + k] = g * sig * up[r * cols + k]
    fwht_rows(scratch, rows, cols, block)
    channelwise_quantize(scratch, qi, scales, rows, cols)

def int8_gemm(a_qi: Ptr, w_qi: Ptr, a_sc: Ptr, w_sc: Ptr, dst: Ptr,
              m: Int, n: Int, k: Int):
    """Int8 matmul: i32 accumulation + epilogue rescale.

    dst[m,n] = float(sum_k int(a[m,k]) * int(w[n,k])) * a_scale[m] * w_scale[n]
    Maps directly to vpdpbusd (AVX-512 VNNI) or tdpbssd (AMX).
    Identical kernel to channelwise int8 — rotation is transparent.
    """
    for i in range(m):
        for j in range(n):
            var acc = 0
            for kk in range(k):
                acc += Int(a_qi[i * k + kk]) * Int(w_qi[j * k + kk])
            dst[i * n + j] = Float32(acc) * a_sc[i] * w_sc[j]


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Hadamard Int8 Layer — Operational Flow")
    print("=" * 70)
    print("  hidden=" + String(HIDDEN) + "  heads=" + String(NUM_HEADS)
          + "  kv_heads=" + String(NUM_KV_HEADS) + "  head_dim=" + String(HEAD_DIM))
    print("  intermediate=" + String(INTERMEDIATE) + "  seq=" + String(SEQ)
          + "  block=" + String(BLOCK))
    print("")

    var rng = Rng(seed=7)

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    # Weights [N, K] — will be modified in-place for hadamard path
    var w_q    = alloc[Float32](HIDDEN * HIDDEN)
    var w_k    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_v    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var w_o    = alloc[Float32](HIDDEN * HIDDEN)
    var w_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var w_down = alloc[Float32](HIDDEN * INTERMEDIATE)

    # Weight quantized values (after preparation)
    var qi_q    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_k    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_v    = alloc[Float32](KV_HIDDEN * HIDDEN)
    var qi_o    = alloc[Float32](HIDDEN * HIDDEN)
    var qi_gate = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_up   = alloc[Float32](INTERMEDIATE * HIDDEN)
    var qi_down = alloc[Float32](HIDDEN * INTERMEDIATE)

    # Weight scales
    var s_q    = alloc[Float32](HIDDEN)
    var s_k    = alloc[Float32](KV_HIDDEN)
    var s_v    = alloc[Float32](KV_HIDDEN)
    var s_o    = alloc[Float32](HIDDEN)
    var s_gate = alloc[Float32](INTERMEDIATE)
    var s_up   = alloc[Float32](INTERMEDIATE)
    var s_down = alloc[Float32](HIDDEN)

    # Norm gains
    var gamma_in   = alloc[Float32](HIDDEN)
    var gamma_post = alloc[Float32](HIDDEN)

    # Activation buffers
    var x_init = alloc[Float32](HIDDEN)
    var x_ref  = alloc[Float32](HIDDEN)         # reference path
    var x_had  = alloc[Float32](HIDDEN)         # hadamard path
    var x_norm = alloc[Float32](HIDDEN)         # scratch: normalized
    var q_buf  = alloc[Float32](HIDDEN)
    var k_buf  = alloc[Float32](KV_HIDDEN)
    var v_buf  = alloc[Float32](KV_HIDDEN)
    var attn   = alloc[Float32](HIDDEN)
    var x_res  = alloc[Float32](HIDDEN)
    var gate   = alloc[Float32](INTERMEDIATE)
    var up     = alloc[Float32](INTERMEDIATE)
    var scratch = alloc[Float32](INTERMEDIATE)   # FWHT scratch
    var qi_act  = alloc[Float32](INTERMEDIATE)   # quantized activations
    var sc_act  = alloc[Float32](SEQ)            # activation scale

    # ------------------------------------------------------------------
    # Generate random data
    # ------------------------------------------------------------------

    for i in range(HIDDEN * HIDDEN):   w_q[i] = rng.normal() * 0.1
    for i in range(KV_HIDDEN * HIDDEN): w_k[i] = rng.normal() * 0.1
    for i in range(KV_HIDDEN * HIDDEN): w_v[i] = rng.normal() * 0.1
    for i in range(HIDDEN * HIDDEN):   w_o[i] = rng.normal() * 0.1
    for i in range(INTERMEDIATE * HIDDEN): w_gate[i] = rng.normal() * 0.05
    for i in range(INTERMEDIATE * HIDDEN): w_up[i] = rng.normal() * 0.05
    for i in range(HIDDEN * INTERMEDIATE): w_down[i] = rng.normal() * 0.05

    for i in range(HIDDEN):
        gamma_in[i] = 0.8 + rng.normal() * 0.1    # ~1.0
        gamma_post[i] = 0.8 + rng.normal() * 0.1

    for i in range(HIDDEN):
        x_init[i] = rng.normal()
        x_ref[i] = x_init[i]
        x_had[i] = x_init[i]

    # ==================================================================
    # F32 REFERENCE LAYER
    # ==================================================================

    print("--- F32 Reference Layer ---")

    # Pre-attention norm
    f32_rmsnorm(x_ref, gamma_in, x_norm, SEQ, HIDDEN)

    # QKV projections
    f32_matmul(x_norm, w_q, q_buf, SEQ, HIDDEN, HIDDEN)
    f32_matmul(x_norm, w_k, k_buf, SEQ, KV_HIDDEN, HIDDEN)
    f32_matmul(x_norm, w_v, v_buf, SEQ, KV_HIDDEN, HIDDEN)
    print("  rmsnorm → Q K V projections")

    # Attention (seq=1: output = GQA-expanded V)
    gqa_expand_v(v_buf, attn)
    print("  attention (seq=1 placeholder)")

    # O projection + residual
    f32_matmul(attn, w_o, x_res, SEQ, HIDDEN, HIDDEN)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    print("  O projection + residual")

    # Pre-MLP norm
    f32_rmsnorm(x_ref, gamma_post, x_norm, SEQ, HIDDEN)

    # Gate/Up + SiLU
    f32_matmul(x_norm, w_gate, gate, SEQ, INTERMEDIATE, HIDDEN)
    f32_matmul(x_norm, w_up, up, SEQ, INTERMEDIATE, HIDDEN)
    f32_silu_mul(gate, up, gate, INTERMEDIATE)
    print("  rmsnorm → GATE UP → silu_mul")

    # Down + residual
    f32_matmul(gate, w_down, x_res, SEQ, HIDDEN, INTERMEDIATE)
    f32_elem_add(x_ref, x_res, x_ref, HIDDEN)
    print("  DOWN projection + residual")
    print("")

    # ==================================================================
    # OFFLINE: WEIGHT PREPARATION
    # ==================================================================

    print("--- Offline: Weight Preparation ---")

    # Absorb gamma
    absorb_gamma(w_q, gamma_in, HIDDEN, HIDDEN)
    absorb_gamma(w_k, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_v, gamma_in, KV_HIDDEN, HIDDEN)
    absorb_gamma(w_gate, gamma_post, INTERMEDIATE, HIDDEN)
    absorb_gamma(w_up, gamma_post, INTERMEDIATE, HIDDEN)
    print("  Absorb input_norm.gamma  → Q K V")
    print("  Absorb post_attn_norm.gamma → GATE UP")
    print("  O DOWN: no gamma to absorb")

    # FWHT each weight's K dimension
    fwht_rows(w_q, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_k, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_v, KV_HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_o, HIDDEN, HIDDEN, BLOCK)
    fwht_rows(w_gate, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_up, INTERMEDIATE, HIDDEN, BLOCK)
    fwht_rows(w_down, HIDDEN, INTERMEDIATE, BLOCK)

    # Per-row int8 quantize
    channelwise_quantize(w_q, qi_q, s_q, HIDDEN, HIDDEN)
    channelwise_quantize(w_k, qi_k, s_k, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_v, qi_v, s_v, KV_HIDDEN, HIDDEN)
    channelwise_quantize(w_o, qi_o, s_o, HIDDEN, HIDDEN)
    channelwise_quantize(w_gate, qi_gate, s_gate, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_up, qi_up, s_up, INTERMEDIATE, HIDDEN)
    channelwise_quantize(w_down, qi_down, s_down, HIDDEN, INTERMEDIATE)
    print("  FWHT + int8 quantize     → Q K V O GATE UP DOWN")
    print("")

    # ==================================================================
    # HADAMARD INT8 LAYER
    # ==================================================================

    print("--- Hadamard Int8 Layer ---")

    # -- ATTENTION BLOCK --

    rms_fwht_quantize(x_had, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK)
    print("  rms_fwht_quantize(x[1," + String(HIDDEN) + "]) → x_i8, scale")

    int8_gemm(qi_act, qi_q, sc_act, s_q, q_buf, SEQ, HIDDEN, HIDDEN)
    print("  int8_gemm(x_i8, Q'[" + String(HIDDEN) + "," + String(HIDDEN) + "]) → q")

    int8_gemm(qi_act, qi_k, sc_act, s_k, k_buf, SEQ, KV_HIDDEN, HIDDEN)
    print("  int8_gemm(x_i8, K'[" + String(KV_HIDDEN) + "," + String(HIDDEN) + "]) → k")

    int8_gemm(qi_act, qi_v, sc_act, s_v, v_buf, SEQ, KV_HIDDEN, HIDDEN)
    print("  int8_gemm(x_i8, V'[" + String(KV_HIDDEN) + "," + String(HIDDEN) + "]) → v")

    # Attention placeholder (seq=1)
    gqa_expand_v(v_buf, attn)
    print("  attention(q, k, v) → attn  [seq=1: gqa_expand(v)]")

    fwht_quantize(attn, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK)
    print("  fwht_quantize(attn[1," + String(HIDDEN) + "]) → attn_i8, scale")

    int8_gemm(qi_act, qi_o, sc_act, s_o, x_res, SEQ, HIDDEN, HIDDEN)
    print("  int8_gemm(attn_i8, O[" + String(HIDDEN) + "," + String(HIDDEN) + "]) → o_out")

    f32_elem_add(x_had, x_res, x_had, HIDDEN)
    print("  elem_add(x, o_out) → x  [residual]")

    # -- MLP BLOCK --

    rms_fwht_quantize(x_had, scratch, qi_act, sc_act, SEQ, HIDDEN, BLOCK)
    print("  rms_fwht_quantize(x[1," + String(HIDDEN) + "]) → x_i8, scale")

    int8_gemm(qi_act, qi_gate, sc_act, s_gate, gate, SEQ, INTERMEDIATE, HIDDEN)
    print("  int8_gemm(x_i8, GATE'[" + String(INTERMEDIATE) + "," + String(HIDDEN) + "]) → gate")

    int8_gemm(qi_act, qi_up, sc_act, s_up, up, SEQ, INTERMEDIATE, HIDDEN)
    print("  int8_gemm(x_i8, UP'[" + String(INTERMEDIATE) + "," + String(HIDDEN) + "]) → up")

    silu_fwht_quantize(gate, up, scratch, qi_act, sc_act, SEQ, INTERMEDIATE, BLOCK)
    print("  silu_fwht_quantize(gate, up) → silu_i8, scale")

    int8_gemm(qi_act, qi_down, sc_act, s_down, x_res, SEQ, HIDDEN, INTERMEDIATE)
    print("  int8_gemm(silu_i8, DOWN[" + String(HIDDEN) + "," + String(INTERMEDIATE) + "]) → down_out")

    f32_elem_add(x_had, x_res, x_had, HIDDEN)
    print("  elem_add(x, down_out) → x  [residual]")
    print("")

    # ==================================================================
    # COMPARISON
    # ==================================================================

    print("--- Comparison ---")
    var sig = Float32(0)
    var noise = Float32(0)
    var max_ae = Float32(0)
    for i in range(HIDDEN):
        sig += x_ref[i] * x_ref[i]
        var err = x_ref[i] - x_had[i]
        noise += err * err
        var ae = absf(err)
        if ae > max_ae: max_ae = ae

    print("  ||x_ref||     = " + d2(sqrt(sig)))
    print("  ||x_ref-x_had|| = " + d2(sqrt(noise)))
    print("  SQNR          = " + d2(sqnr_db(sig, noise)) + " dB")
    print("  Max |error|   = " + d2(max_ae))
    print("")

    # ==================================================================
    # OPERATION SUMMARY
    # ==================================================================

    print("=" * 70)
    print("  Per-Layer Operation Summary")
    print("=" * 70)
    print("")
    print("  Fused quantization kernels:        4")
    print("    rms_fwht_quantize:               2  (before QKV, before GATE/UP)")
    print("    fwht_quantize:                   1  (before O)")
    print("    silu_fwht_quantize:              1  (before DOWN)")
    print("")
    print("  Int8 GEMMs (i32 acc + rescale):    7")
    print("    Q, K, V:                         3  (shared quantized input)")
    print("    O:                               1")
    print("    GATE, UP:                        2  (shared quantized input)")
    print("    DOWN:                            1")
    print("")
    print("  Unchanged bf16 ops:                5+")
    print("    rope(q), rope(k):                2")
    print("    kv_cache_write(k), write(v):     2")
    print("    attention:                        1")
    print("    elem_add (residual):             2")
    print("    allreduce (TP):                  2")

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
    attn.free(); x_res.free()
    gate.free(); up.free(); scratch.free()
    qi_act.free(); sc_act.free()
