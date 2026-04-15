"""Block autocovariance of transformed-domain quantization error.

This measures the autocovariance structure of quantization error *inside*
rotated blocks before those blocks are consumed by the matched linear map.

Two sites are analyzed for the dense FFN path:

1. entry blocks
   `rmsnorm*gamma -> rotation(256) -> rowwise absmax i8`

2. post-activation blocks
   `gelu_tanh(gate) * up -> rotation(64) -> per-block absmax i8`

For each site, we report lagged autocovariance and normalized autocorrelation
of the transformed-domain error for:

- fwht
- philox_fwht
- antithetic_pair_mean
"""

from std.collections import InlineArray
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from modeling.gemma4_common import Gemma4BaseConfig
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_activation_bf16, fill_normal_bf16, fill_gamma_bf16,
    fill_philox_signs, apply_signs_inplace,
    rmsnorm_gamma_from_bf16, copy_f32, fwht_rotate, philox_fwht_rotate,
    quantize_absmax_row, quantize_absmax_block, dequant_row, gemv_f32,
)


comptime C = Gemma4BaseConfig
comptime ENTRY_BLK = 256
comptime POST_BLK = 64
comptime NUM_POST_BLK = C.INTERMEDIATE // POST_BLK
comptime NUM_TRIALS = 96
comptime MAX_LAG = 8
comptime MAX_BLOCK = ENTRY_BLK if ENTRY_BLK > POST_BLK else POST_BLK
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime PHILOX_ENTRY_SEED = UInt64(0x626C6B5F61757431)
comptime PHILOX_POST_SEED = UInt64(0x626C6B5F61757432)


@fieldwise_init
struct AutoCovSummary(Copyable, ImplicitlyCopyable):
    var blocks: Int
    var var_sum: InlineArray[Float64, MAX_LAG + 1]
    var cov_sum: InlineArray[Float64, MAX_LAG + 1]


def init_autocov_summary() -> AutoCovSummary:
    return AutoCovSummary(
        0,
        InlineArray[Float64, MAX_LAG + 1](fill=Float64(0)),
        InlineArray[Float64, MAX_LAG + 1](fill=Float64(0)),
    )


def reflect_neg_signs[block: Int](src: F32Ptr, dst: F32Ptr, count: Int):
    for blk in range(count // block):
        var off = blk * block
        for i in range(block):
            dst[off + i] = -src[off + (block - 1 - i)]


def accumulate_block_autocov(
    mut summary: AutoCovSummary,
    err: F32Ptr,
    block: Int,
):
    var mean = Float64(0)
    for i in range(block):
        mean += Float64(err[i])
    mean /= Float64(block)

    var centered = InlineArray[Float64, MAX_BLOCK](fill=Float64(0))
    for i in range(block):
        centered[i] = Float64(err[i]) - mean

    var var0 = Float64(0)
    for i in range(block):
        var0 += centered[i] * centered[i]
    var0 /= Float64(block)

    summary.blocks += 1
    for lag in range(MAX_LAG + 1):
        var n = block - lag
        var cov = Float64(0)
        for i in range(n):
            cov += centered[i] * centered[i + lag]
        cov /= Float64(n)
        summary.var_sum[lag] += var0
        summary.cov_sum[lag] += cov


def print_autocov_summary(label: String, summary: AutoCovSummary):
    print(label)
    print("lag mean_var mean_cov mean_corr")
    if summary.blocks == 0:
        return
    var inv = Float64(1.0 / Float64(summary.blocks))
    for lag in range(MAX_LAG + 1):
        var mean_var = summary.var_sum[lag] * inv
        var mean_cov = summary.cov_sum[lag] * inv
        var mean_corr = mean_cov / mean_var if mean_var > Float64(0) else Float64(0)
        print(String(lag)
            + " " + String(mean_var)
            + " " + String(mean_cov)
            + " " + String(mean_corr))


def analyze_entry_scheme[use_philox: Bool](
    x_norm: F32Ptr,
    work: F32Ptr,
    q_buf: I8Ptr,
    dq_buf: F32Ptr,
    err_buf: F32Ptr,
    entry_signs: F32Ptr,
    mut summary: AutoCovSummary,
):
    copy_f32(x_norm, work, C.HIDDEN)
    if use_philox:
        apply_signs_inplace(work, entry_signs, C.HIDDEN)
    fwht_rotate[ENTRY_BLK](work, C.HIDDEN)
    var absmax = quantize_absmax_row(work, q_buf, C.HIDDEN)
    dequant_row(q_buf, absmax, dq_buf, C.HIDDEN)

    for blk in range(C.HIDDEN // ENTRY_BLK):
        var off = blk * ENTRY_BLK
        for i in range(ENTRY_BLK):
            err_buf[i] = dq_buf[off + i] - work[off + i]
        accumulate_block_autocov(summary, err_buf, ENTRY_BLK)


def analyze_post_scheme[use_signs: Bool](
    inter_ref: F32Ptr,
    work: F32Ptr,
    q_buf: I8Ptr,
    absmax_buf: F32Ptr,
    dq_buf: F32Ptr,
    err_buf: F32Ptr,
    post_signs: F32Ptr,
    mut summary: AutoCovSummary,
):
    copy_f32(inter_ref, work, C.INTERMEDIATE)
    if use_signs:
        apply_signs_inplace(work, post_signs, C.INTERMEDIATE)
    fwht_rotate[POST_BLK](work, C.INTERMEDIATE)
    quantize_absmax_block[POST_BLK](work, q_buf, absmax_buf, C.INTERMEDIATE)

    for blk in range(NUM_POST_BLK):
        var off = blk * POST_BLK
        dequant_row(q_buf + off, absmax_buf[blk], dq_buf, POST_BLK)
        for i in range(POST_BLK):
            err_buf[i] = dq_buf[i] - work[off + i]
        accumulate_block_autocov(summary, err_buf, POST_BLK)


def main():
    print("=== quant_analysis/dense_ffn_block_autocovariance ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " trials=" + String(NUM_TRIALS)
        + " max_lag=" + String(MAX_LAG))

    var x_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gamma_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gate_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))
    var up_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))

    var x_norm = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var work_hidden = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var work_inter = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var gate_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var up_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var inter_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var q_hidden = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.HIDDEN)))
    var q_inter = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.INTERMEDIATE)))
    var dq_hidden = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var dq_block = F32Ptr(unsafe_from_address=Int(alloc[Float32](POST_BLK)))
    var err_buf = F32Ptr(unsafe_from_address=Int(alloc[Float32](ENTRY_BLK)))
    var inter_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_POST_BLK)))

    var entry_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var post_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var entry_fwht = init_autocov_summary()
    var entry_philox = init_autocov_summary()
    var entry_antithetic = init_autocov_summary()
    var post_fwht = init_autocov_summary()
    var post_philox = init_autocov_summary()
    var post_antithetic = init_autocov_summary()

    var rng = Rng(seed=0x626C6B5F636F765F)
    comptime width = simd_width_of[DType.float32]()

    for trial in range(NUM_TRIALS):
        fill_activation_bf16(rng, x_bf16, C.HIDDEN, X_STDDEV)
        fill_gamma_bf16(rng, gamma_bf16, C.HIDDEN, GAMMA_CENTER, GAMMA_STDDEV)
        fill_normal_bf16(rng, gate_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)
        fill_normal_bf16(rng, up_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)

        rmsnorm_gamma_from_bf16(x_bf16, gamma_bf16, x_norm, C.HIDDEN)
        gemv_f32(gate_w_bf16, x_norm, gate_ref, C.INTERMEDIATE, C.HIDDEN)
        gemv_f32(up_w_bf16, x_norm, up_ref, C.INTERMEDIATE, C.HIDDEN)

        var k = 0
        while k + width <= C.INTERMEDIATE:
            var g = (gate_ref + k).load[width=width]()
            var u = (up_ref + k).load[width=width]()
            (inter_ref + k).store(gelu_tanh_f32[width](g) * u)
            k += width

        fill_philox_signs[ENTRY_BLK](entry_signs_a, C.HIDDEN, PHILOX_ENTRY_SEED, UInt64(trial))
        fill_philox_signs[POST_BLK](post_signs_a, C.INTERMEDIATE, PHILOX_POST_SEED, UInt64(trial))
        reflect_neg_signs[ENTRY_BLK](entry_signs_a, entry_signs_b, C.HIDDEN)
        reflect_neg_signs[POST_BLK](post_signs_a, post_signs_b, C.INTERMEDIATE)

        analyze_entry_scheme[False](
            x_norm, work_hidden, q_hidden, dq_hidden, err_buf, entry_signs_a, entry_fwht)
        analyze_entry_scheme[True](
            x_norm, work_hidden, q_hidden, dq_hidden, err_buf, entry_signs_a, entry_philox)

        analyze_entry_scheme[True](
            x_norm, work_hidden, q_hidden, dq_hidden, err_buf, entry_signs_a, entry_antithetic)
        analyze_entry_scheme[True](
            x_norm, work_hidden, q_hidden, dq_hidden, err_buf, entry_signs_b, entry_antithetic)

        analyze_post_scheme[False](
            inter_ref, work_inter, q_inter, inter_absmax, dq_block, err_buf, post_signs_a, post_fwht)
        analyze_post_scheme[True](
            inter_ref, work_inter, q_inter, inter_absmax, dq_block, err_buf, post_signs_a, post_philox)
        analyze_post_scheme[True](
            inter_ref, work_inter, q_inter, inter_absmax, dq_block, err_buf, post_signs_a, post_antithetic)
        analyze_post_scheme[True](
            inter_ref, work_inter, q_inter, inter_absmax, dq_block, err_buf, post_signs_b, post_antithetic)

    print_autocov_summary("entry/fwht", entry_fwht)
    print_autocov_summary("entry/philox_fwht", entry_philox)
    print_autocov_summary("entry/antithetic_pair_mean", entry_antithetic)
    print_autocov_summary("post/fwht", post_fwht)
    print_autocov_summary("post/philox_fwht", post_philox)
    print_autocov_summary("post/antithetic_pair_mean", post_antithetic)

    x_bf16.free()
    gamma_bf16.free()
    gate_w_bf16.free()
    up_w_bf16.free()
    x_norm.free()
    work_hidden.free()
    work_inter.free()
    gate_ref.free()
    up_ref.free()
    inter_ref.free()
    q_hidden.free()
    q_inter.free()
    dq_hidden.free()
    dq_block.free()
    err_buf.free()
    inter_absmax.free()
    entry_signs_a.free()
    entry_signs_b.free()
    post_signs_a.free()
    post_signs_b.free()
