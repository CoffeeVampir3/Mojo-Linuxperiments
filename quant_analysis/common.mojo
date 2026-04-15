"""Shared helpers for quantization analysis mocks."""

from std.math import log, abs
from std.random.philox import Random

from simd_math import sqrt as simd_sqrt, roundeven, sincos
from experimental3.kernels.fwht import fwht_block
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr


comptime ANALYSIS_EPS = Float32(1e-6)


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
        var sc = sincos[1](SIMD[DType.float64, 1](Float64(u2) * Float64(6.283185307179586)))
        return r * Float32(sc.cos_val)


def fill_normal_bf16(mut rng: Rng, dst: BF16Ptr, count: Int, stddev: Float32):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](rng.normal() * stddev)


@always_inline
def activation_like_sample(mut rng: Rng, stddev: Float32) -> Float32:
    """Skewed heavy-tail synthetic activation sample.

    Base mass is approximately Gaussian, but every sample gets a mild
    multiplicative heavy-tail factor and a small fraction of coordinates
    receive larger positive outliers. A smaller negative-outlier path is
    retained so the distribution is not one-sided, but the total shape stays
    positively skewed.
    """
    var v = rng.normal() * stddev
    v *= Float32(1.0) + Float32(0.30) * abs(rng.normal())

    if rng.uniform() < Float32(0.03):
        v += stddev * (Float32(4.0) + Float32(3.0) * (-log(rng.uniform())))

    if rng.uniform() < Float32(0.01):
        v -= stddev * (Float32(2.0) + Float32(2.0) * (-log(rng.uniform())))

    return v


def fill_activation_bf16(mut rng: Rng, dst: BF16Ptr, count: Int, stddev: Float32):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](activation_like_sample(rng, stddev))


@always_inline
def activation_like_sample_stress(mut rng: Rng, stddev: Float32) -> Float32:
    """More exaggerated skewed heavy-tail activation sample.

    This profile is intentionally harsher than `activation_like_sample`:
    stronger multiplicative heavy-tail, more frequent outliers, and larger
    positive spikes so accumulation effects are easier to observe.
    """
    var v = rng.normal() * stddev
    v *= Float32(1.0) + Float32(0.85) * abs(rng.normal())

    if rng.uniform() < Float32(0.08):
        v += stddev * (Float32(7.0) + Float32(5.0) * (-log(rng.uniform())))

    if rng.uniform() < Float32(0.03):
        v -= stddev * (Float32(3.0) + Float32(3.0) * (-log(rng.uniform())))

    return v


def fill_activation_bf16_stress(mut rng: Rng, dst: BF16Ptr, count: Int, stddev: Float32):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](activation_like_sample_stress(rng, stddev))


def fill_gamma_bf16(mut rng: Rng, dst: BF16Ptr, count: Int,
    center: Float32, stddev: Float32,
):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](center + rng.normal() * stddev)


def copy_bf16_to_f32(src: BF16Ptr, dst: F32Ptr, count: Int):
    for i in range(count):
        dst[i] = Float32(src[i])


def copy_f32(src: F32Ptr, dst: F32Ptr, count: Int):
    for i in range(count):
        dst[i] = src[i]


def rmsnorm_gamma_from_bf16(
    src: BF16Ptr, gamma: BF16Ptr, dst: F32Ptr, count: Int,
    eps: Float32 = ANALYSIS_EPS,
):
    var sum_sq = Float32(0)
    for i in range(count):
        var v = Float32(src[i])
        sum_sq += v * v
    var inv = Float32(1.0) / simd_sqrt(sum_sq / Float32(count) + eps)
    for i in range(count):
        dst[i] = Float32(src[i]) * inv * Float32(gamma[i])


def rmsnorm_gamma_from_f32(
    src: F32Ptr, gamma: BF16Ptr, dst: F32Ptr, count: Int,
    eps: Float32 = ANALYSIS_EPS,
):
    var sum_sq = Float32(0)
    for i in range(count):
        var v = src[i]
        sum_sq += v * v
    var inv = Float32(1.0) / simd_sqrt(sum_sq / Float32(count) + eps)
    for i in range(count):
        dst[i] = src[i] * inv * Float32(gamma[i])


def rmsnorm_from_bf16(
    src: BF16Ptr, dst: F32Ptr, count: Int,
    eps: Float32 = ANALYSIS_EPS,
):
    var sum_sq = Float32(0)
    for i in range(count):
        var v = Float32(src[i])
        sum_sq += v * v
    var inv = Float32(1.0) / simd_sqrt(sum_sq / Float32(count) + eps)
    for i in range(count):
        dst[i] = Float32(src[i]) * inv


def rmsnorm_f32_inplace(dst: F32Ptr, count: Int, eps: Float32 = ANALYSIS_EPS):
    var sum_sq = Float32(0)
    for i in range(count):
        sum_sq += dst[i] * dst[i]
    var inv = Float32(1.0) / simd_sqrt(sum_sq / Float32(count) + eps)
    for i in range(count):
        dst[i] *= inv


def fwht_rotate[block: Int](buf: F32Ptr, count: Int):
    for b in range(count // block):
        fwht_block[block](buf + b * block)


@always_inline
def kron_rank[block: Int]() -> Int:
    var rank = Int(0)
    var x = block
    while x > 1:
        rank += 1
        x >>= 1
    return rank


@always_inline
def kron_parity_sign(mask: UInt64, coord: Int) -> Float32:
    var bits = mask & UInt64(coord)
    var parity = Bool(False)
    while bits != 0:
        parity = not parity
        bits &= bits - UInt64(1)
    return Float32(-1.0) if parity else Float32(1.0)


def kron_sign_mask[block: Int](
    seed: UInt64, subsequence: UInt64, block_idx: Int,
) -> UInt64:
    # This samples the Kronecker/Walsh sign family, not an arbitrary
    # Rademacher diagonal. For Sylvester Hadamard blocks this means H * D_kron
    # is exactly a row permutation of H.
    var rank = kron_rank[block]()
    var rng = Random[rounds=10](
        seed=seed,
        subsequence=subsequence,
        offset=UInt64(block_idx * rank),
    )
    var mask = UInt64(0)
    var raw = rng.step()
    for bit in range(rank):
        if bit > 0 and bit % 4 == 0:
            raw = rng.step()
        var lane = bit % 4
        if (raw[lane] & UInt32(1)) != UInt32(0):
            mask |= UInt64(1) << UInt64(bit)
    return mask


def kron_sign_inplace[block: Int](
    buf: F32Ptr, count: Int, seed: UInt64, subsequence: UInt64,
):
    for blk in range(count // block):
        var off = blk * block
        var mask = kron_sign_mask[block](seed, subsequence, blk)
        for i in range(block):
            buf[off + i] *= kron_parity_sign(mask, i)


def kron_fwht_rotate[block: Int](
    buf: F32Ptr, count: Int, seed: UInt64, subsequence: UInt64,
):
    kron_sign_inplace[block](buf, count, seed, subsequence)
    fwht_rotate[block](buf, count)


def philox_sign_inplace[block: Int](
    buf: F32Ptr, count: Int, seed: UInt64, subsequence: UInt64,
):
    var steps_per_block = (block + 127) // 128
    for blk in range(count // block):
        var off = blk * block
        var rng = Random[rounds=10](
            seed=seed,
            subsequence=subsequence,
            offset=UInt64(blk * steps_per_block),
        )
        var raw = rng.step()
        for i in range(block):
            if i > 0 and i % 128 == 0:
                raw = rng.step()
            var local = i % 128
            var lane = local // 32
            var bit = local % 32
            if ((raw[lane] >> UInt32(bit)) & UInt32(1)) != UInt32(0):
                buf[off + i] = -buf[off + i]


def fill_philox_signs[block: Int](
    signs: F32Ptr, count: Int, seed: UInt64, subsequence: UInt64,
):
    var steps_per_block = (block + 127) // 128
    for blk in range(count // block):
        var off = blk * block
        var rng = Random[rounds=10](
            seed=seed,
            subsequence=subsequence,
            offset=UInt64(blk * steps_per_block),
        )
        var raw = rng.step()
        for i in range(block):
            if i > 0 and i % 128 == 0:
                raw = rng.step()
            var local = i % 128
            var lane = local // 32
            var bit = local % 32
            signs[off + i] = Float32(-1.0) if ((raw[lane] >> UInt32(bit)) & UInt32(1)) != UInt32(0) else Float32(1.0)


def apply_signs_inplace(
    buf: F32Ptr, signs: F32Ptr, count: Int,
):
    for i in range(count):
        buf[i] *= signs[i]


def philox_fwht_rotate[block: Int](
    buf: F32Ptr, count: Int, seed: UInt64, subsequence: UInt64,
):
    philox_sign_inplace[block](buf, count, seed, subsequence)
    fwht_rotate[block](buf, count)


def quantize_absmax_row(src: F32Ptr, dst: I8Ptr, count: Int) -> Float32:
    var amax = Float32(0)
    for i in range(count):
        var v = abs(src[i])
        if v > amax:
            amax = v
    if amax < Float32(1e-10):
        amax = Float32(1e-10)

    var inv = Float32(127.0) / amax
    for i in range(count):
        var q = roundeven(src[i] * inv)
        if q < Float32(-128.0):
            q = Float32(-128.0)
        if q > Float32(127.0):
            q = Float32(127.0)
        dst[i] = q.cast[DType.int8]()
    return amax


def quantize_absmax_block[block: Int](
    src: F32Ptr, dst: I8Ptr, absmax_out: F32Ptr, count: Int,
):
    for b in range(count // block):
        var off = b * block
        absmax_out[b] = quantize_absmax_row(src + off, dst + off, block)


def dequant_row(src: I8Ptr, absmax: Float32, dst: F32Ptr, count: Int):
    var scale = absmax / Float32(127.0)
    for i in range(count):
        dst[i] = Float32(Int(src[i])) * scale


def quantize_rotated_weight_rows[block: Int](
    weight_bf16: BF16Ptr,
    q_weight: I8Ptr,
    weight_scale: F32Ptr,
    rows: Int,
    cols: Int,
    row_work: F32Ptr,
):
    for n in range(rows):
        var row_off = n * cols
        copy_bf16_to_f32(weight_bf16 + row_off, row_work, cols)
        fwht_rotate[block](row_work, cols)
        weight_scale[n] = quantize_absmax_row(row_work, q_weight + row_off, cols) / Float32(127.0)


def quantize_kron_rotated_weight_rows[block: Int](
    weight_bf16: BF16Ptr,
    q_weight: I8Ptr,
    weight_scale: F32Ptr,
    rows: Int,
    cols: Int,
    row_work: F32Ptr,
    seed: UInt64,
    subsequence: UInt64,
):
    for n in range(rows):
        var row_off = n * cols
        copy_bf16_to_f32(weight_bf16 + row_off, row_work, cols)
        kron_fwht_rotate[block](row_work, cols, seed, subsequence)
        weight_scale[n] = quantize_absmax_row(row_work, q_weight + row_off, cols) / Float32(127.0)


def quantize_philox_rotated_weight_rows[block: Int](
    weight_bf16: BF16Ptr,
    q_weight: I8Ptr,
    weight_scale: F32Ptr,
    rows: Int,
    cols: Int,
    row_work: F32Ptr,
    seed: UInt64,
    subsequence: UInt64,
):
    for n in range(rows):
        var row_off = n * cols
        copy_bf16_to_f32(weight_bf16 + row_off, row_work, cols)
        philox_fwht_rotate[block](row_work, cols, seed, subsequence)
        weight_scale[n] = quantize_absmax_row(row_work, q_weight + row_off, cols) / Float32(127.0)


def compute_colsum_rows(
    weight_i8: I8Ptr, rows: Int, cols: Int, colsum: F32Ptr,
):
    for n in range(rows):
        var acc = Int(0)
        var row_off = n * cols
        for k in range(cols):
            acc += Int(weight_i8[row_off + k])
        colsum[n] = Float32(acc)


def compute_colsum_blocks_transposed[block: Int](
    weight_i8: I8Ptr, rows: Int, cols: Int, colsum: F32Ptr,
):
    var num_blocks = cols // block
    for blk in range(num_blocks):
        var blk_off = blk * block
        for n in range(rows):
            var acc = Int(0)
            var row_off = n * cols + blk_off
            for k in range(block):
                acc += Int(weight_i8[row_off + k])
            colsum[blk * rows + n] = Float32(acc)


def gemv_f32(
    weight_bf16: BF16Ptr, x_f32: F32Ptr,
    dst: F32Ptr, rows: Int, cols: Int,
):
    for n in range(rows):
        var acc = Float32(0)
        var row_off = n * cols
        for k in range(cols):
            acc += Float32(weight_bf16[row_off + k]) * x_f32[k]
        dst[n] = acc


def gemv_rowwise_corrected(
    act_i8: I8Ptr,
    act_absmax: Float32,
    weight_i8: I8Ptr,
    weight_scale: F32Ptr,
    colsum: F32Ptr,
    dst: F32Ptr,
    rows: Int,
    cols: Int,
):
    var act_dequant = act_absmax / Float32(127.0)
    for n in range(rows):
        var acc = Int(0)
        var row_off = n * cols
        for k in range(cols):
            acc += (Int(act_i8[k]) + 128) * Int(weight_i8[row_off + k])
        var corrected = Float32(acc) - Float32(128.0) * colsum[n]
        dst[n] = corrected * act_dequant * weight_scale[n]


def gemv_blocked_corrected[block: Int](
    act_i8: I8Ptr,
    act_absmax: F32Ptr,
    weight_i8: I8Ptr,
    weight_scale: F32Ptr,
    colsum: F32Ptr,
    dst: F32Ptr,
    rows: Int,
    cols: Int,
):
    var num_blocks = cols // block
    for n in range(rows):
        var total = Float32(0)
        var row_off = n * cols
        for blk in range(num_blocks):
            var blk_off = blk * block
            var acc = Int(0)
            for k in range(block):
                acc += (Int(act_i8[blk_off + k]) + 128) * Int(weight_i8[row_off + blk_off + k])
            var corrected = Float32(acc) - Float32(128.0) * colsum[blk * rows + n]
            total += corrected * (act_absmax[blk] / Float32(127.0))
        dst[n] = total * weight_scale[n]


@fieldwise_init
struct ErrorReport(Copyable, ImplicitlyCopyable):
    var rmse: Float64
    var cosine: Float64


def compute_error(ref_buf: F32Ptr, test_buf: F32Ptr, count: Int) -> ErrorReport:
    var dot_rt = Float64(0)
    var norm_r = Float64(0)
    var norm_t = Float64(0)
    var sum_sq_err = Float64(0)
    for i in range(count):
        var r = Float64(ref_buf[i])
        var t = Float64(test_buf[i])
        dot_rt += r * t
        norm_r += r * r
        norm_t += t * t
        sum_sq_err += (t - r) * (t - r)
    var denom = Float64(simd_sqrt(Float32(norm_r))) * Float64(simd_sqrt(Float32(norm_t)))
    var cosine = dot_rt / denom if denom > Float64(0) else Float64(0)
    var rmse = Float64(simd_sqrt(Float32(sum_sq_err / Float64(count))))
    return ErrorReport(rmse, cosine)


@fieldwise_init
struct ErrorAggregate(Copyable, ImplicitlyCopyable):
    var count: Int
    var mean_rmse: Float64
    var mean_cosine: Float64
    var worst_rmse: Float64
    var worst_cosine: Float64


def init_aggregate() -> ErrorAggregate:
    return ErrorAggregate(0, Float64(0), Float64(0), Float64(0), Float64(1))


def add_report(mut agg: ErrorAggregate, rep: ErrorReport):
    agg.count += 1
    agg.mean_rmse += rep.rmse
    agg.mean_cosine += rep.cosine
    if rep.rmse > agg.worst_rmse:
        agg.worst_rmse = rep.rmse
    if rep.cosine < agg.worst_cosine:
        agg.worst_cosine = rep.cosine


def print_summary(label: String, agg: ErrorAggregate):
    if agg.count == 0:
        print(label + ": no trials")
        return
    print(label
        + ": mean_rmse=" + String(agg.mean_rmse / Float64(agg.count))
        + " mean_cosine=" + String(agg.mean_cosine / Float64(agg.count))
        + " worst_rmse=" + String(agg.worst_rmse)
        + " worst_cosine=" + String(agg.worst_cosine))
