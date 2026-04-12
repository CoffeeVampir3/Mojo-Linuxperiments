"""Gamma-split benefit analysis using real model weights.

Loads actual norm weight vectors from the bf16 checkpoint and measures
dot-product error with vs without sqrt gamma splitting under FWHT + per-block
i8 quantization. Tests every distinct norm that feeds an FWHT-quantized
projection, plus the final norm (lm_head), to determine whether the lm_head
gamma regime is uniquely problematic or if other norms would also benefit.

Run: pixi run mojo -I . quantization_analysis.mojo
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
                print("  short read: " + name)
                return None
            var buf = alloc[Scalar[DType.bfloat16]](nbytes // 2)
            memcpy(dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf)),
                   src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(data.unsafe_ptr())),
                   count=nbytes)
            return BF16Ptr(unsafe_from_address=Int(buf))
    except:
        print("  read failed: " + name)
        return None


# ── Helpers ──────────────────────────────────────────────────────────

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


def quantize_block(src: F32Ptr, qi: I8Ptr, scales: F32Ptr, n: Int):
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


def dot_i8xi8_blk(a_qi: I8Ptr, a_sc: F32Ptr,
                   b_qi: I8Ptr, b_sc: F32Ptr, n: Int) -> Float32:
    var total = Float32(0)
    for b in range(n // BLK):
        var off = b * BLK
        var acc = Float32(0)
        for k in range(BLK):
            acc += Float32(Int(a_qi[off + k])) * Float32(Int(b_qi[off + k]))
        total += acc * a_sc[b] * b_sc[b]
    return total


def block_energy_ratio(buf: F32Ptr, n: Int) -> Float32:
    var emin = Float32(1e30)
    var emax = Float32(0)
    for b in range(n // BLK):
        var e = Float32(0)
        for k in range(BLK):
            var v = buf[b * BLK + k]
            e += v * v
        if e < emin:
            emin = e
        if e > emax:
            emax = e
    if emin < Float32(1e-20):
        return Float32(1e10)
    return emax / emin


# ── Stats ────────────────────────────────────────────────────────────

@fieldwise_init
struct ErrStats(Copyable, ImplicitlyCopyable):
    var sum: Float64
    var sum_sq: Float64
    var peak: Float64
    var count: Int

    @staticmethod
    def zero() -> ErrStats:
        return ErrStats(Float64(0), Float64(0), Float64(0), 0)

    def record(mut self, err: Float32):
        var e = Float64(err)
        self.sum += e
        self.sum_sq += e * e
        if e > self.peak:
            self.peak = e
        self.count += 1

    def mean(self) -> Float64:
        if self.count == 0:
            return Float64(0)
        return self.sum / Float64(self.count)


# ── Per-gamma trial ──────────────────────────────────────────────────

def run_gamma_trial(
    gamma_f32: F32Ptr, sqrt_gamma: F32Ptr,
    # Scratch
    mut rng: Rng,
    xnp: F32Ptr, aap: F32Ptr, abp: F32Ptr,
    wap: F32Ptr, wbp: F32Ptr, wrp: F32Ptr,
    aaqp: I8Ptr, abqp: I8Ptr, waqp: I8Ptr, wbqp: I8Ptr,
    aasp: F32Ptr, absp: F32Ptr, wasp: F32Ptr, wbsp: F32Ptr,
) -> InlineArray[Float32, 4]:
    """Returns [err_a_mean%, err_b_mean%, energy_a_mean, energy_b_mean]."""
    var err_a = ErrStats.zero()
    var err_b = ErrStats.zero()
    var energy_a_sum = Float64(0)
    var energy_b_sum = Float64(0)

    for trial in range(NUM_TRIALS):
        fill_normal(rng, xnp, K, Float32(1.0))
        rms_normalize(xnp, K)

        # Case A: full gamma on activation
        for k in range(K):
            aap[k] = xnp[k] * gamma_f32[k]
        fwht_rotate(aap, K)
        energy_a_sum += Float64(block_energy_ratio(aap, K))
        quantize_block(aap, aaqp, aasp, K)

        # Case B: sqrt-split
        for k in range(K):
            abp[k] = xnp[k] * sqrt_gamma[k]
        fwht_rotate(abp, K)
        energy_b_sum += Float64(block_energy_ratio(abp, K))
        quantize_block(abp, abqp, absp, K)

        for row in range(ROWS_PER_TRIAL):
            fill_normal(rng, wrp, K, Float32(1.0) / simd_sqrt(Float32(K)))

            var exact = Float32(0)
            for k in range(K):
                exact += xnp[k] * gamma_f32[k] * wrp[k]
            var exact_abs = abs(exact)
            if exact_abs < Float32(1e-10):
                continue

            # Case A weight: plain
            for k in range(K):
                wap[k] = wrp[k]
            fwht_rotate(wap, K)
            quantize_block(wap, waqp, wasp, K)

            # Case B weight: absorb sqrt(|gamma|)
            for k in range(K):
                wbp[k] = wrp[k] * abs(sqrt_gamma[k])
            fwht_rotate(wbp, K)
            quantize_block(wbp, wbqp, wbsp, K)

            var approx_a = dot_i8xi8_blk(aaqp, aasp, waqp, wasp, K)
            var approx_b = dot_i8xi8_blk(abqp, absp, wbqp, wbsp, K)

            err_a.record(abs(approx_a - exact) / exact_abs)
            err_b.record(abs(approx_b - exact) / exact_abs)

    var result = InlineArray[Float32, 4](fill=Float32(0))
    result[0] = Float32(err_a.mean() * Float64(100))
    result[1] = Float32(err_b.mean() * Float64(100))
    result[2] = Float32(energy_a_sum / Float64(NUM_TRIALS))
    result[3] = Float32(energy_b_sum / Float64(NUM_TRIALS))
    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=== Gamma-Split Analysis on Real Model Weights ===")
    print("K=" + String(K) + "  FWHT_BLK=" + String(BLK)
        + "  trials=" + String(NUM_TRIALS) + "  rows/trial=" + String(ROWS_PER_TRIAL))
    print()

    # Parse safetensors headers
    var shards = discover_shards(Path(BF16_MODEL_DIR))
    if len(shards) == 0:
        print("no shards found in " + BF16_MODEL_DIR)
        return
    print("found " + String(len(shards)) + " shard(s) in " + BF16_MODEL_DIR)

    var headers = HeapMoveArray[SafetensorsHeader](len(shards))
    for i in range(len(shards)):
        var h = parse_safetensors_header(shards[i])
        if not h:
            print("failed to parse header for shard " + String(i))
            return
        headers.push(h.take())

    # Norms to test: (tensor_name, label, which_projection)
    # We test a few representative layers — layer 0 (early), layer 15 (mid), layer 37 (late)
    comptime MAX_NORMS = 16
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
        norm_labels.append("L" + String(li) + " input_norm (attn qkv)")
        norm_names.append(prefix + "pre_feedforward_layernorm.weight")
        norm_labels.append("L" + String(li) + " pre_ffn_norm (dense)")
        norm_names.append(prefix + "pre_feedforward_layernorm_2.weight")
        norm_labels.append("L" + String(li) + " pre_ffn_norm_2 (expert)")
    norm_names.append("model.language_model.norm.weight")
    norm_labels.append("final_norm (lm_head)")

    # Allocate scratch once
    var x_norm = alloc[Float32](K)
    var act_a = alloc[Float32](K)
    var act_b = alloc[Float32](K)
    var wt_a = alloc[Float32](K)
    var wt_b = alloc[Float32](K)
    var w_raw = alloc[Float32](K)
    var gamma_f32 = alloc[Float32](K)
    var sqrt_gamma_buf = alloc[Float32](K)
    var act_a_qi = alloc[Scalar[DType.int8]](K)
    var act_b_qi = alloc[Scalar[DType.int8]](K)
    var wt_a_qi = alloc[Scalar[DType.int8]](K)
    var wt_b_qi = alloc[Scalar[DType.int8]](K)
    var act_a_sc = alloc[Float32](NUM_BLK)
    var act_b_sc = alloc[Float32](NUM_BLK)
    var wt_a_sc = alloc[Float32](NUM_BLK)
    var wt_b_sc = alloc[Float32](NUM_BLK)

    var xnp = F32Ptr(unsafe_from_address=Int(x_norm))
    var aap = F32Ptr(unsafe_from_address=Int(act_a))
    var abp = F32Ptr(unsafe_from_address=Int(act_b))
    var wap = F32Ptr(unsafe_from_address=Int(wt_a))
    var wbp = F32Ptr(unsafe_from_address=Int(wt_b))
    var wrp = F32Ptr(unsafe_from_address=Int(w_raw))
    var gfp = F32Ptr(unsafe_from_address=Int(gamma_f32))
    var sgp = F32Ptr(unsafe_from_address=Int(sqrt_gamma_buf))
    var aaqp = I8Ptr(unsafe_from_address=Int(act_a_qi))
    var abqp = I8Ptr(unsafe_from_address=Int(act_b_qi))
    var waqp = I8Ptr(unsafe_from_address=Int(wt_a_qi))
    var wbqp = I8Ptr(unsafe_from_address=Int(wt_b_qi))
    var aasp = F32Ptr(unsafe_from_address=Int(act_a_sc))
    var absp = F32Ptr(unsafe_from_address=Int(act_b_sc))
    var wasp = F32Ptr(unsafe_from_address=Int(wt_a_sc))
    var wbsp = F32Ptr(unsafe_from_address=Int(wt_b_sc))

    print()
    print("norm                             | mean|g|  max|g|  | energy_A  energy_B | err_A%    err_B%   | A/B    | benefit")
    print("---------------------------------|------------------|--------------------|--------------------|--------|--------")

    for ni in range(len(norm_names)):
        var raw = load_bf16_tensor(norm_names[ni], headers, shards)
        if not raw:
            continue
        var bf16p = BF16Ptr(unsafe_from_address=Int(raw.value()))

        # Convert to f32 gamma + precompute sqrt split
        var g_abs_sum = Float32(0)
        var g_abs_max = Float32(0)
        for k in range(K):
            var v = Float32(bf16p[k])
            gfp[k] = v

            var sign = Float32(1.0)
            var av = v
            if av < Float32(0):
                sign = Float32(-1.0)
                av = -av
            if av < Float32(1e-10):
                av = Float32(1e-10)
            sgp[k] = sign * simd_sqrt(av)
            g_abs_sum += av
            if av > g_abs_max:
                g_abs_max = av

        var g_mean = g_abs_sum / Float32(K)

        # Run trial with deterministic seed (same x/W for all norms)
        var rng = Rng(seed=42)
        var result = run_gamma_trial(gfp, sgp, rng,
            xnp, aap, abp, wap, wbp, wrp,
            aaqp, abqp, waqp, wbqp, aasp, absp, wasp, wbsp)

        var err_a = result[0]
        var err_b = result[1]
        var energy_a = result[2]
        var energy_b = result[3]
        var ratio = Float32(1.0)
        if err_b > Float32(0):
            ratio = err_a / err_b
        var benefit = (Float32(1.0) - Float32(1.0) / ratio) * Float32(100)
        if ratio < Float32(1.0):
            benefit = -(Float32(1.0) - ratio) * Float32(100)

        print(norm_labels[ni]
            + " | " + String(g_mean) + "  " + String(g_abs_max)
            + " | " + String(energy_a) + "  " + String(energy_b)
            + " | " + String(err_a) + "  " + String(err_b)
            + " | " + String(ratio) + "x"
            + " | " + String(benefit) + "%")

        raw.value().free()

    print()
    print("A/B > 1 means sqrt-split helps. benefit = (1 - 1/ratio) * 100.")

    x_norm.free()
    act_a.free()
    act_b.free()
    wt_a.free()
    wt_b.free()
    w_raw.free()
    gamma_f32.free()
    sqrt_gamma_buf.free()
    act_a_qi.free()
    act_b_qi.free()
    wt_a_qi.free()
    wt_b_qi.free()
    act_a_sc.free()
    act_b_sc.free()
    wt_a_sc.free()
    wt_b_sc.free()
