"""Gamma absorption via splitting — decomposed error analysis on real model weights.

Tests whether gamma can be split (absorbed) into both sides of an i8 dot product
without catastrophic error. Decomposes the error into activation-only, weight-only,
and combined contributions at each alpha level.

  alpha=0:   act = RMSNorm(x),                  wt = W * gamma       (full absorption)
  alpha=0.5: act = RMSNorm(x) * sqrt_gamma,     wt = W * sqrt(|g|)   (split absorption)
  alpha=1:   act = RMSNorm(x) * gamma,           wt = W               (current approach)

Tests per-row quantization (layer projection scenario) and per-block quantization
(lm_head scenario). Uses real gamma vectors from the bf16 checkpoint.

Run: pixi run mojo -I . gamma_absorb_validation.mojo
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.math import log, exp, abs
from std.pathlib import Path
from std.collections import InlineArray

from simd_math import sqrt as simd_sqrt, roundeven, sincos
from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from notstdcollections import HeapMoveArray
from modeling.loader import discover_shards, find_tensor

from experimental3.kernels.fwht import fwht_block
from experimental3.common_math import F32Ptr, I8Ptr, BF16Ptr


comptime K = 2816
comptime BLK = 256
comptime NUM_BLK = K // BLK
comptime NUM_TRIALS = 512
comptime ROWS_PER_TRIAL = 32
comptime BF16_MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime NUM_ALPHAS = 5


# ── PRNG ─────────────────────────────────────────────────────────────

struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed
        for _ in range(8):
            _ = self.next()

    @always_inline
    def next(mut self) -> UInt64:
        var x = self.s
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.s = x
        return x

    @always_inline
    def uniform(mut self) -> Float32:
        return (Float32(self.next() & 0xFFFFFF) + Float32(1.0)) / Float32(0x1000001)

    @always_inline
    def normal(mut self) -> Float32:
        var u1 = self.uniform()
        var u2 = self.uniform()
        var r = simd_sqrt(Float32(-2.0) * log(u1))
        var sc = sincos[1](SIMD[DType.float64, 1](Float64(u2) * Float64(6.2831853)))
        return r * Float32(sc.cos_val)


# ── Tensor loading ───────────────────────────────────────────────────

def load_bf16_tensor(
    name: String,
    ref headers: HeapMoveArray[SafetensorsHeader],
    ref shards: List[Path],
) -> Optional[BF16Ptr]:
    var found = find_tensor(name, headers)
    if not found:
        print("  missing: " + name)
        return None
    var shard_idx = found.value()[0]
    var meta = found.value()[1].copy()
    var byte_off = headers[shard_idx].data_offset + meta.start
    var nbytes = meta.end - meta.start
    try:
        with open(shards[shard_idx], "r") as f:
            _ = f.seek(UInt64(byte_off), 0)
            var data = f.read_bytes(size=nbytes)
            if len(data) != nbytes:
                return None
            var buf = alloc[Scalar[DType.bfloat16]](nbytes // 2)
            memcpy(dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf)),
                   src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(data.unsafe_ptr())),
                   count=nbytes)
            return BF16Ptr(unsafe_from_address=Int(buf))
    except:
        return None


# ── Primitives ───────────────────────────────────────────────────────

def fill_normal(mut rng: Rng, p: F32Ptr, n: Int, stddev: Float32):
    for i in range(n):
        p[i] = rng.normal() * stddev


def rms_normalize(p: F32Ptr, n: Int):
    var ss = Float32(0)
    for i in range(n):
        ss += p[i] * p[i]
    var inv = Float32(1.0) / simd_sqrt(ss / Float32(n) + Float32(1e-6))
    for i in range(n):
        p[i] *= inv


def fwht_rotate(buf: F32Ptr, n: Int):
    for b in range(n // BLK):
        fwht_block[BLK](buf + b * BLK)


# ── Quantization ─────────────────────────────────────────────────────

def quantize_per_row(src: F32Ptr, qi: I8Ptr, n: Int) -> Float32:
    """Per-row absmax i8 quantize. Returns scale = amax/127."""
    var amax = Float32(0)
    for k in range(n):
        var v = abs(src[k])
        if v > amax:
            amax = v
    if amax < Float32(1e-10):
        amax = Float32(1e-10)
    var inv = Float32(127.0) / amax
    for k in range(n):
        qi[k] = roundeven(src[k] * inv).cast[DType.int8]()
    return amax / Float32(127.0)


def quantize_per_block(src: F32Ptr, qi: I8Ptr, scales: F32Ptr, n: Int):
    """Per-block absmax i8 quantize."""
    for b in range(n // BLK):
        var off = b * BLK
        var amax = Float32(0)
        for k in range(BLK):
            var v = abs(src[off + k])
            if v > amax:
                amax = v
        if amax < Float32(1e-10):
            amax = Float32(1e-10)
        scales[b] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax
        for k in range(BLK):
            qi[off + k] = roundeven(src[off + k] * inv).cast[DType.int8]()


# ── Dot products (per-row scale) ─────────────────────────────────────

def dot_f32_f32(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var acc = Float32(0)
    for k in range(n):
        acc += a[k] * b[k]
    return acc


def dot_qi_f32_row(qi: I8Ptr, scale: Float32, f: F32Ptr, n: Int) -> Float32:
    """Dequantized i8 (per-row scale) dot f32."""
    var acc = Float32(0)
    for k in range(n):
        acc += Float32(Int(qi[k])) * f[k]
    return acc * scale


def dot_f32_qi_row(f: F32Ptr, qi: I8Ptr, scale: Float32, n: Int) -> Float32:
    """F32 dot dequantized i8 (per-row scale)."""
    var acc = Float32(0)
    for k in range(n):
        acc += f[k] * Float32(Int(qi[k]))
    return acc * scale


def dot_qi_qi_row(a_qi: I8Ptr, a_sc: Float32,
                  b_qi: I8Ptr, b_sc: Float32, n: Int) -> Float32:
    """Dequantized i8 dot dequantized i8, both per-row scale."""
    var acc = Float32(0)
    for k in range(n):
        acc += Float32(Int(a_qi[k])) * Float32(Int(b_qi[k]))
    return acc * a_sc * b_sc


# ── Dot products (per-block scale) ───────────────────────────────────

def dot_qi_f32_blk(qi: I8Ptr, scales: F32Ptr, f: F32Ptr, n: Int) -> Float32:
    var total = Float32(0)
    for b in range(n // BLK):
        var off = b * BLK
        var acc = Float32(0)
        for k in range(BLK):
            acc += Float32(Int(qi[off + k])) * f[off + k]
        total += acc * scales[b]
    return total


def dot_f32_qi_blk(f: F32Ptr, qi: I8Ptr, scales: F32Ptr, n: Int) -> Float32:
    var total = Float32(0)
    for b in range(n // BLK):
        var off = b * BLK
        var acc = Float32(0)
        for k in range(BLK):
            acc += f[off + k] * Float32(Int(qi[off + k]))
        total += acc * scales[b]
    return total


def dot_qi_qi_blk(a_qi: I8Ptr, a_sc: F32Ptr,
                  b_qi: I8Ptr, b_sc: F32Ptr, n: Int) -> Float32:
    var total = Float32(0)
    for b in range(n // BLK):
        var off = b * BLK
        var acc = Float32(0)
        for k in range(BLK):
            acc += Float32(Int(a_qi[off + k])) * Float32(Int(b_qi[off + k]))
        total += acc * a_sc[b] * b_sc[b]
    return total


# ── Stats ────────────────────────────────────────────────────────────

@fieldwise_init
struct ErrStats(Copyable, ImplicitlyCopyable):
    var sum: Float64
    var peak: Float64
    var count: Int

    @staticmethod
    def zero() -> ErrStats:
        return ErrStats(Float64(0), Float64(0), 0)

    def record(mut self, err: Float32):
        var e = Float64(err)
        self.sum += e
        if e > self.peak:
            self.peak = e
        self.count += 1

    def mean_pct(self) -> Float32:
        if self.count == 0:
            return Float32(0)
        return Float32(self.sum / Float64(self.count) * Float64(100))

    def peak_pct(self) -> Float32:
        return Float32(self.peak * Float64(100))


# ── Gamma prep ───────────────────────────────────────────────────────

def prepare_gamma_powers(bf16_gamma: BF16Ptr, gamma_f32: F32Ptr,
                         gamma_powers: F32Ptr, n: Int, num_alphas: Int):
    """Precompute |gamma|^alpha for alpha in [0, 0.25, 0.5, 0.75, 1.0].

    gamma_powers layout: [num_alphas][n] where powers[ai * n + k] = |gamma[k]|^(alpha[ai]).
    Also stores sign-preserving gamma_f32.
    """
    for k in range(n):
        var v = Float32(bf16_gamma[k])
        gamma_f32[k] = v
        var av = v
        if av < Float32(0):
            av = -av
        if av < Float32(1e-10):
            av = Float32(1e-10)
        # Powers via nested sqrt: g1 = |g|, g^0.5, g^0.25
        var g1 = av
        var g05 = simd_sqrt(av)
        var g025 = simd_sqrt(g05)
        # alpha=0: |g|^0 = 1
        gamma_powers[0 * n + k] = Float32(1.0)
        # alpha=0.25: |g|^0.25
        gamma_powers[1 * n + k] = g025
        # alpha=0.5: |g|^0.5
        gamma_powers[2 * n + k] = g05
        # alpha=0.75: |g|^0.75 = g^0.5 * g^0.25
        gamma_powers[3 * n + k] = g05 * g025
        # alpha=1.0: |g|^1
        gamma_powers[4 * n + k] = g1


def apply_act_split(x: F32Ptr, gamma_f32: F32Ptr, gamma_power: F32Ptr,
                    dst: F32Ptr, n: Int):
    """dst = x * sign(gamma) * |gamma|^alpha. gamma_power is precomputed |g|^alpha."""
    for k in range(n):
        var sign = Float32(1.0)
        if gamma_f32[k] < Float32(0):
            sign = Float32(-1.0)
        dst[k] = x[k] * sign * gamma_power[k]


def apply_wt_split(w: F32Ptr, inv_gamma_power: F32Ptr, full_gamma_power: F32Ptr,
                   dst: F32Ptr, n: Int):
    """dst = w * |gamma|^(1-alpha). We compute this as w * |g|^1 / |g|^alpha."""
    for k in range(n):
        dst[k] = w[k] * full_gamma_power[k] / inv_gamma_power[k]


# ── Per-norm trial ───────────────────────────────────────────────────

@fieldwise_init
struct AlphaResult(Copyable, ImplicitlyCopyable):
    # Per-row quantization
    var row_act_only: Float32
    var row_wt_only: Float32
    var row_combined: Float32
    # Per-block quantization
    var blk_act_only: Float32
    var blk_wt_only: Float32
    var blk_combined: Float32

    @staticmethod
    def zero() -> AlphaResult:
        return AlphaResult(Float32(0), Float32(0), Float32(0),
                          Float32(0), Float32(0), Float32(0))


def run_alpha_trial(
    gamma_f32: F32Ptr,
    act_power: F32Ptr,    # |gamma|^alpha
    wt_power_num: F32Ptr, # |gamma|^1 (full magnitude)
    wt_power_den: F32Ptr, # |gamma|^alpha (to divide out)
    mut rng: Rng,
    # Scratch
    xnp: F32Ptr, act_f32: F32Ptr, wt_f32: F32Ptr, wrp: F32Ptr,
    act_qi: I8Ptr, wt_qi: I8Ptr,
    act_blk_sc: F32Ptr, wt_blk_sc: F32Ptr,
) -> AlphaResult:
    var row_act = ErrStats.zero()
    var row_wt = ErrStats.zero()
    var row_both = ErrStats.zero()
    var blk_act = ErrStats.zero()
    var blk_wt = ErrStats.zero()
    var blk_both = ErrStats.zero()

    for trial in range(NUM_TRIALS):
        fill_normal(rng, xnp, K, Float32(1.0))
        rms_normalize(xnp, K)

        # Activation side: x * sign(g) * |g|^alpha, then FWHT
        apply_act_split(xnp, gamma_f32, act_power, act_f32, K)
        fwht_rotate(act_f32, K)

        # Per-row quantize activation
        var act_row_sc = quantize_per_row(act_f32, act_qi, K)
        # Per-block quantize activation (into same qi buffer — do block first, save scales)
        quantize_per_block(act_f32, act_qi, act_blk_sc, K)

        for row in range(ROWS_PER_TRIAL):
            fill_normal(rng, wrp, K, Float32(1.0) / simd_sqrt(Float32(K)))

            # Exact reference
            var exact = Float32(0)
            for k in range(K):
                exact += xnp[k] * gamma_f32[k] * wrp[k]
            var exact_abs = abs(exact)
            if exact_abs < Float32(1e-10):
                continue

            # Weight side: w * |g|^(1-alpha), then FWHT
            apply_wt_split(wrp, wt_power_den, wt_power_num, wt_f32, K)
            fwht_rotate(wt_f32, K)

            # Per-row quantize weight
            var wt_row_sc = quantize_per_row(wt_f32, wt_qi, K)

            # === Per-row decomposed errors ===
            # Re-quantize act per-row (we overwrote qi with per-block above)
            act_row_sc = quantize_per_row(act_f32, act_qi, K)

            var r_act = dot_qi_f32_row(act_qi, act_row_sc, wt_f32, K)
            var r_wt = dot_f32_qi_row(act_f32, wt_qi, wt_row_sc, K)
            var r_both = dot_qi_qi_row(act_qi, act_row_sc, wt_qi, wt_row_sc, K)

            row_act.record(abs(r_act - exact) / exact_abs)
            row_wt.record(abs(r_wt - exact) / exact_abs)
            row_both.record(abs(r_both - exact) / exact_abs)

            # === Per-block decomposed errors ===
            quantize_per_block(act_f32, act_qi, act_blk_sc, K)
            quantize_per_block(wt_f32, wt_qi, wt_blk_sc, K)

            var b_act = dot_qi_f32_blk(act_qi, act_blk_sc, wt_f32, K)
            var b_wt = dot_f32_qi_blk(act_f32, wt_qi, wt_blk_sc, K)
            var b_both = dot_qi_qi_blk(act_qi, act_blk_sc, wt_qi, wt_blk_sc, K)

            blk_act.record(abs(b_act - exact) / exact_abs)
            blk_wt.record(abs(b_wt - exact) / exact_abs)
            blk_both.record(abs(b_both - exact) / exact_abs)

    return AlphaResult(
        row_act_only=row_act.mean_pct(), row_wt_only=row_wt.mean_pct(),
        row_combined=row_both.mean_pct(),
        blk_act_only=blk_act.mean_pct(), blk_wt_only=blk_wt.mean_pct(),
        blk_combined=blk_both.mean_pct(),
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=== Gamma Absorption via Splitting — Decomposed Error Analysis ===")
    print("K=" + String(K) + "  FWHT_BLK=" + String(BLK)
        + "  trials=" + String(NUM_TRIALS) + "  rows/trial=" + String(ROWS_PER_TRIAL))
    print()

    var shards = discover_shards(Path(BF16_MODEL_DIR))
    if len(shards) == 0:
        print("no shards found in " + BF16_MODEL_DIR)
        return
    print("found " + String(len(shards)) + " shard(s)")

    var headers = HeapMoveArray[SafetensorsHeader](len(shards))
    for i in range(len(shards)):
        var h = parse_safetensors_header(shards[i])
        if not h:
            print("failed to parse header")
            return
        headers.push(h.take())

    # Norm sources to test
    var norm_names = List[String]()
    var norm_labels = List[String]()
    var sample_layers = InlineArray[Int, 3](fill=0)
    sample_layers[0] = 0
    sample_layers[1] = 15
    sample_layers[2] = 29
    for li_idx in range(3):
        var li = sample_layers[li_idx]
        var prefix = "model.language_model.layers." + String(li) + "."
        norm_names.append(prefix + "input_layernorm.weight")
        norm_labels.append("L" + String(li) + " input_norm")
        norm_names.append(prefix + "pre_feedforward_layernorm.weight")
        norm_labels.append("L" + String(li) + " pre_ffn")
    norm_names.append("model.language_model.norm.weight")
    norm_labels.append("final_norm")

    # Allocate scratch
    var gamma_f32 = alloc[Float32](K)
    var gamma_powers = alloc[Float32](K * NUM_ALPHAS)
    var x_norm = alloc[Float32](K)
    var act_f32 = alloc[Float32](K)
    var wt_f32 = alloc[Float32](K)
    var w_raw = alloc[Float32](K)
    var act_qi = alloc[Scalar[DType.int8]](K)
    var wt_qi = alloc[Scalar[DType.int8]](K)
    var act_blk_sc = alloc[Float32](NUM_BLK)
    var wt_blk_sc = alloc[Float32](NUM_BLK)

    var gfp = F32Ptr(unsafe_from_address=Int(gamma_f32))
    var gpp = F32Ptr(unsafe_from_address=Int(gamma_powers))
    var xnp = F32Ptr(unsafe_from_address=Int(x_norm))
    var afp = F32Ptr(unsafe_from_address=Int(act_f32))
    var wfp = F32Ptr(unsafe_from_address=Int(wt_f32))
    var wrp = F32Ptr(unsafe_from_address=Int(w_raw))
    var aqp = I8Ptr(unsafe_from_address=Int(act_qi))
    var wqp = I8Ptr(unsafe_from_address=Int(wt_qi))
    var asp = F32Ptr(unsafe_from_address=Int(act_blk_sc))
    var wsp = F32Ptr(unsafe_from_address=Int(wt_blk_sc))

    var alpha_labels = InlineArray[Float32, NUM_ALPHAS](fill=Float32(0))
    alpha_labels[0] = Float32(0.0)
    alpha_labels[1] = Float32(0.25)
    alpha_labels[2] = Float32(0.5)
    alpha_labels[3] = Float32(0.75)
    alpha_labels[4] = Float32(1.0)

    for ni in range(len(norm_names)):
        var raw = load_bf16_tensor(norm_names[ni], headers, shards)
        if not raw:
            continue
        var bf16p = raw.value()
        prepare_gamma_powers(bf16p, gfp, gpp, K, NUM_ALPHAS)

        # Gamma statistics
        var g_abs_sum = Float32(0)
        var g_abs_max = Float32(0)
        var g_abs_min = Float32(1e30)
        for k in range(K):
            var av = abs(gfp[k])
            g_abs_sum += av
            if av > g_abs_max:
                g_abs_max = av
            if av < g_abs_min:
                g_abs_min = av
        var g_mean = g_abs_sum / Float32(K)
        var g_ratio = g_abs_max / g_mean

        print("━━━ " + norm_labels[ni] + " ━━━")
        print("  mean|g|=" + String(g_mean) + "  max|g|=" + String(g_abs_max)
            + "  max/mean=" + String(g_ratio))
        print()
        print("  Per-row quantization (layer projection scenario):")
        print("  alpha | act_only%  wt_only%  combined% ")
        print("  ------|------------------------------")

        for ai in range(NUM_ALPHAS):
            var act_pow = F32Ptr(unsafe_from_address=Int(gamma_powers) + ai * K * 4)
            var full_pow = F32Ptr(unsafe_from_address=Int(gamma_powers) + 4 * K * 4)  # alpha=1.0
            var rng = Rng(seed=42)
            var result = run_alpha_trial(
                gfp, act_pow, full_pow, act_pow,
                rng, xnp, afp, wfp, wrp, aqp, wqp, asp, wsp)
            print("  " + String(alpha_labels[ai])
                + "   | " + String(result.row_act_only)
                + "  " + String(result.row_wt_only)
                + "  " + String(result.row_combined))

        print()
        print("  Per-block quantization (lm_head scenario):")
        print("  alpha | act_only%  wt_only%  combined% ")
        print("  ------|------------------------------")

        for ai in range(NUM_ALPHAS):
            var act_pow = F32Ptr(unsafe_from_address=Int(gamma_powers) + ai * K * 4)
            var full_pow = F32Ptr(unsafe_from_address=Int(gamma_powers) + 4 * K * 4)
            var rng = Rng(seed=42)
            var result = run_alpha_trial(
                gfp, act_pow, full_pow, act_pow,
                rng, xnp, afp, wfp, wrp, aqp, wqp, asp, wsp)
            print("  " + String(alpha_labels[ai])
                + "   | " + String(result.blk_act_only)
                + "  " + String(result.blk_wt_only)
                + "  " + String(result.blk_combined))

        print()
        bf16p.free()

    gamma_f32.free()
    gamma_powers.free()
    x_norm.free()
    act_f32.free()
    wt_f32.free()
    w_raw.free()
    act_qi.free()
    wt_qi.free()
    act_blk_sc.free()
    wt_blk_sc.free()
