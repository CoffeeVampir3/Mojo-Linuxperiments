"""Direct paired-cancellation analysis for the dense FFN quantization path.

This harness asks a narrower question than rollout drift:

Given one bf16 reference layer output `y_ref` and two quantized variants
`y_A`, `y_B` built from paired randomized pre-signing patterns, are the
quantization errors `e_A = y_A - y_ref` and `e_B = y_B - y_ref`
anticorrelated enough to cancel when averaged?

Two B constructions are tested:

1. `neg`
   `B = -A`
   This is the literal "opposite sign vector" construction.

2. `reflect_neg`
   `B[i] = -A[block - 1 - i]` inside each transform block.
   This is a nontrivial reflected antithetic pair.
"""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from simd_math import sqrt as simd_sqrt
from modeling.gemma4_common import Gemma4BaseConfig
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_activation_bf16, fill_normal_bf16, fill_gamma_bf16,
    fill_philox_signs, apply_signs_inplace,
    rmsnorm_gamma_from_bf16, copy_f32, fwht_rotate,
    quantize_absmax_row, quantize_absmax_block,
    compute_colsum_rows, compute_colsum_blocks_transposed, gemv_f32,
    gemv_rowwise_corrected, gemv_blocked_corrected,
    compute_error,
)


comptime C = Gemma4BaseConfig
comptime ENTRY_BLK = 256
comptime POST_BLK = 64
comptime NUM_POST_BLK = C.INTERMEDIATE // POST_BLK
comptime NUM_TRIALS = 64
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime PHILOX_ENTRY_SEED = UInt64(0x706169725F656E31)
comptime PHILOX_POST_SEED = UInt64(0x706169725F706F31)


def negate_signs(src: F32Ptr, dst: F32Ptr, count: Int):
    for i in range(count):
        dst[i] = -src[i]


def reflect_neg_signs[block: Int](src: F32Ptr, dst: F32Ptr, count: Int):
    for blk in range(count // block):
        var off = blk * block
        for i in range(block):
            dst[off + i] = -src[off + (block - 1 - i)]


def quantize_weight_rows_with_signs[block: Int](
    weight_bf16: BF16Ptr,
    q_weight: I8Ptr,
    weight_scale: F32Ptr,
    rows: Int,
    cols: Int,
    row_work: F32Ptr,
    signs: F32Ptr,
):
    for n in range(rows):
        var row_off = n * cols
        for k in range(cols):
            row_work[k] = Float32(weight_bf16[row_off + k])
        apply_signs_inplace(row_work, signs, cols)
        fwht_rotate[block](row_work, cols)
        weight_scale[n] = quantize_absmax_row(row_work, q_weight + row_off, cols) / Float32(127.0)


def run_dense_ffn_quant_with_signs(
    x_bf16: BF16Ptr,
    gamma_bf16: BF16Ptr,
    gate_w_bf16: BF16Ptr,
    up_w_bf16: BF16Ptr,
    down_w_bf16: BF16Ptr,
    entry_signs: F32Ptr,
    post_signs: F32Ptr,
    x_norm: F32Ptr,
    x_work: F32Ptr,
    gate_q: F32Ptr,
    up_q: F32Ptr,
    inter_q: F32Ptr,
    y_q: F32Ptr,
    act_i8: I8Ptr,
    inter_i8: I8Ptr,
    gate_w_i8: I8Ptr,
    up_w_i8: I8Ptr,
    down_w_i8: I8Ptr,
    gate_w_scale: F32Ptr,
    up_w_scale: F32Ptr,
    down_w_scale: F32Ptr,
    gate_colsum: F32Ptr,
    up_colsum: F32Ptr,
    down_colsum: F32Ptr,
    inter_absmax: F32Ptr,
    row_work_hidden: F32Ptr,
    row_work_inter: F32Ptr,
):
    rmsnorm_gamma_from_bf16(x_bf16, gamma_bf16, x_norm, C.HIDDEN)
    copy_f32(x_norm, x_work, C.HIDDEN)
    apply_signs_inplace(x_work, entry_signs, C.HIDDEN)
    fwht_rotate[ENTRY_BLK](x_work, C.HIDDEN)
    var act_absmax = quantize_absmax_row(x_work, act_i8, C.HIDDEN)

    quantize_weight_rows_with_signs[ENTRY_BLK](
        gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN,
        row_work_hidden, entry_signs)
    quantize_weight_rows_with_signs[ENTRY_BLK](
        up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN,
        row_work_hidden, entry_signs)
    compute_colsum_rows(gate_w_i8, C.INTERMEDIATE, C.HIDDEN, gate_colsum)
    compute_colsum_rows(up_w_i8, C.INTERMEDIATE, C.HIDDEN, up_colsum)

    gemv_rowwise_corrected(
        act_i8, act_absmax,
        gate_w_i8, gate_w_scale, gate_colsum,
        gate_q, C.INTERMEDIATE, C.HIDDEN)
    gemv_rowwise_corrected(
        act_i8, act_absmax,
        up_w_i8, up_w_scale, up_colsum,
        up_q, C.INTERMEDIATE, C.HIDDEN)

    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= C.INTERMEDIATE:
        var g = (gate_q + k).load[width=width]()
        var u = (up_q + k).load[width=width]()
        (inter_q + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    copy_f32(inter_q, row_work_inter, C.INTERMEDIATE)
    apply_signs_inplace(row_work_inter, post_signs, C.INTERMEDIATE)
    fwht_rotate[POST_BLK](row_work_inter, C.INTERMEDIATE)
    quantize_absmax_block[POST_BLK](row_work_inter, inter_i8, inter_absmax, C.INTERMEDIATE)

    quantize_weight_rows_with_signs[POST_BLK](
        down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE,
        row_work_inter, post_signs)
    compute_colsum_blocks_transposed[POST_BLK](
        down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)
    gemv_blocked_corrected[POST_BLK](
        inter_i8, inter_absmax,
        down_w_i8, down_w_scale, down_colsum,
        y_q, C.HIDDEN, C.INTERMEDIATE)


@fieldwise_init
struct PairSummary(Copyable, ImplicitlyCopyable):
    var count: Int
    var mean_rmse_a: Float64
    var mean_rmse_b: Float64
    var mean_rmse_avg: Float64
    var mean_err_cos: Float64


def init_pair_summary() -> PairSummary:
    return PairSummary(0, Float64(0), Float64(0), Float64(0), Float64(0))


def error_cosine(ref_buf: F32Ptr, a: F32Ptr, b: F32Ptr, count: Int) -> Float64:
    var dot = Float64(0)
    var norm_a = Float64(0)
    var norm_b = Float64(0)
    for i in range(count):
        var ea = Float64(a[i] - ref_buf[i])
        var eb = Float64(b[i] - ref_buf[i])
        dot += ea * eb
        norm_a += ea * ea
        norm_b += eb * eb
    var denom = Float64(0)
    if norm_a > Float64(0) and norm_b > Float64(0):
        denom = Float64(simd_sqrt(Float32(norm_a))) * Float64(simd_sqrt(Float32(norm_b)))
    return dot / denom if denom > Float64(0) else Float64(0)


def add_pair_result(
    mut summary: PairSummary,
    ref_buf: F32Ptr,
    a: F32Ptr,
    b: F32Ptr,
    avg: F32Ptr,
    count: Int,
):
    summary.count += 1
    summary.mean_rmse_a += compute_error(ref_buf, a, count).rmse
    summary.mean_rmse_b += compute_error(ref_buf, b, count).rmse
    summary.mean_rmse_avg += compute_error(ref_buf, avg, count).rmse
    summary.mean_err_cos += error_cosine(ref_buf, a, b, count)


def print_pair_summary(label: String, s: PairSummary):
    if s.count == 0:
        print(label + ": no trials")
        return
    var inv = Float64(1.0) / Float64(s.count)
    print(label
        + ": mean_rmse_a=" + String(s.mean_rmse_a * inv)
        + " mean_rmse_b=" + String(s.mean_rmse_b * inv)
        + " mean_rmse_avg=" + String(s.mean_rmse_avg * inv)
        + " mean_err_cos=" + String(s.mean_err_cos * inv))


def main():
    print("=== quant_analysis/dense_ffn_pair_cancellation ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " trials=" + String(NUM_TRIALS))

    var x_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gamma_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gate_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))
    var up_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))
    var down_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN * C.INTERMEDIATE)))

    var x_norm = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_work = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var gate_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var up_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var inter_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var y_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var gate_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var up_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var inter_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var y_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var y_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var y_avg = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var act_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.HIDDEN)))
    var inter_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.INTERMEDIATE)))
    var gate_w_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.INTERMEDIATE * C.HIDDEN)))
    var up_w_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.INTERMEDIATE * C.HIDDEN)))
    var down_w_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.HIDDEN * C.INTERMEDIATE)))
    var gate_w_scale = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var up_w_scale = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var down_w_scale = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var gate_colsum = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var up_colsum = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var down_colsum = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN * NUM_POST_BLK)))
    var inter_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_POST_BLK)))
    var row_work_hidden = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var row_work_inter = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var entry_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_b_neg = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_b_reflect = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var post_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_b_neg = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_b_reflect = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var rng = Rng(seed=0x706169725F63616E)
    var neg_summary = init_pair_summary()
    var reflect_summary = init_pair_summary()
    comptime width = simd_width_of[DType.float32]()

    for trial in range(NUM_TRIALS):
        fill_activation_bf16(rng, x_bf16, C.HIDDEN, X_STDDEV)
        fill_gamma_bf16(rng, gamma_bf16, C.HIDDEN, GAMMA_CENTER, GAMMA_STDDEV)
        fill_normal_bf16(rng, gate_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)
        fill_normal_bf16(rng, up_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)
        fill_normal_bf16(rng, down_w_bf16, C.HIDDEN * C.INTERMEDIATE, W_STDDEV)

        rmsnorm_gamma_from_bf16(x_bf16, gamma_bf16, x_norm, C.HIDDEN)
        gemv_f32(gate_w_bf16, x_norm, gate_ref, C.INTERMEDIATE, C.HIDDEN)
        gemv_f32(up_w_bf16, x_norm, up_ref, C.INTERMEDIATE, C.HIDDEN)

        var k = 0
        while k + width <= C.INTERMEDIATE:
            var g = (gate_ref + k).load[width=width]()
            var u = (up_ref + k).load[width=width]()
            (inter_ref + k).store(gelu_tanh_f32[width](g) * u)
            k += width
        gemv_f32(down_w_bf16, inter_ref, y_ref, C.HIDDEN, C.INTERMEDIATE)

        fill_philox_signs[ENTRY_BLK](entry_a, C.HIDDEN, PHILOX_ENTRY_SEED, UInt64(trial))
        fill_philox_signs[POST_BLK](post_a, C.INTERMEDIATE, PHILOX_POST_SEED, UInt64(trial))
        negate_signs(entry_a, entry_b_neg, C.HIDDEN)
        negate_signs(post_a, post_b_neg, C.INTERMEDIATE)
        reflect_neg_signs[ENTRY_BLK](entry_a, entry_b_reflect, C.HIDDEN)
        reflect_neg_signs[POST_BLK](post_a, post_b_reflect, C.INTERMEDIATE)

        run_dense_ffn_quant_with_signs(
            x_bf16, gamma_bf16, gate_w_bf16, up_w_bf16, down_w_bf16,
            entry_a, post_a,
            x_norm, x_work, gate_q, up_q, inter_q, y_a,
            act_i8, inter_i8, gate_w_i8, up_w_i8, down_w_i8,
            gate_w_scale, up_w_scale, down_w_scale,
            gate_colsum, up_colsum, down_colsum, inter_absmax,
            row_work_hidden, row_work_inter)

        run_dense_ffn_quant_with_signs(
            x_bf16, gamma_bf16, gate_w_bf16, up_w_bf16, down_w_bf16,
            entry_b_neg, post_b_neg,
            x_norm, x_work, gate_q, up_q, inter_q, y_b,
            act_i8, inter_i8, gate_w_i8, up_w_i8, down_w_i8,
            gate_w_scale, up_w_scale, down_w_scale,
            gate_colsum, up_colsum, down_colsum, inter_absmax,
            row_work_hidden, row_work_inter)
        for i in range(C.HIDDEN):
            y_avg[i] = Float32(0.5) * (y_a[i] + y_b[i])
        add_pair_result(neg_summary, y_ref, y_a, y_b, y_avg, C.HIDDEN)

        run_dense_ffn_quant_with_signs(
            x_bf16, gamma_bf16, gate_w_bf16, up_w_bf16, down_w_bf16,
            entry_b_reflect, post_b_reflect,
            x_norm, x_work, gate_q, up_q, inter_q, y_b,
            act_i8, inter_i8, gate_w_i8, up_w_i8, down_w_i8,
            gate_w_scale, up_w_scale, down_w_scale,
            gate_colsum, up_colsum, down_colsum, inter_absmax,
            row_work_hidden, row_work_inter)
        for i in range(C.HIDDEN):
            y_avg[i] = Float32(0.5) * (y_a[i] + y_b[i])
        add_pair_result(reflect_summary, y_ref, y_a, y_b, y_avg, C.HIDDEN)

    print_pair_summary("pair/neg", neg_summary)
    print_pair_summary("pair/reflect_neg", reflect_summary)

    x_bf16.free()
    gamma_bf16.free()
    gate_w_bf16.free()
    up_w_bf16.free()
    down_w_bf16.free()
    x_norm.free()
    x_work.free()
    gate_ref.free()
    up_ref.free()
    inter_ref.free()
    y_ref.free()
    gate_q.free()
    up_q.free()
    inter_q.free()
    y_a.free()
    y_b.free()
    y_avg.free()
    act_i8.free()
    inter_i8.free()
    gate_w_i8.free()
    up_w_i8.free()
    down_w_i8.free()
    gate_w_scale.free()
    up_w_scale.free()
    down_w_scale.free()
    gate_colsum.free()
    up_colsum.free()
    down_colsum.free()
    inter_absmax.free()
    row_work_hidden.free()
    row_work_inter.free()
    entry_a.free()
    entry_b_neg.free()
    entry_b_reflect.free()
    post_a.free()
    post_b_neg.free()
    post_b_reflect.free()
