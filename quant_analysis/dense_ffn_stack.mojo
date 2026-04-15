"""Layer-stack drift analysis for the dense FFN quantization path.

This script measures two different error notions over repeated layers:

1. teacher-forced local error
   The quantized layer receives the bf16-reference hidden state for that layer.
   This isolates single-layer distortion and does not accumulate past errors.

2. closed-loop drift
   The quantized stack feeds its own previous hidden state into the next layer.
   This measures true error accumulation across layers.

Both paths are compared against the same f32 reference rollout.
"""

from std.collections import InlineArray
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from simd_math import sqrt as simd_sqrt
from modeling.gemma4_common import Gemma4BaseConfig
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    activation_like_sample_stress, fill_gamma_bf16, fill_normal_bf16,
    fill_philox_signs, apply_signs_inplace,
    rmsnorm_gamma_from_f32, copy_f32, fwht_rotate, philox_fwht_rotate,
    quantize_absmax_row, quantize_absmax_block,
    quantize_rotated_weight_rows, compute_colsum_rows,
    quantize_philox_rotated_weight_rows,
    compute_colsum_blocks_transposed, gemv_f32,
    gemv_rowwise_corrected, gemv_blocked_corrected,
    compute_error,
)


comptime C = Gemma4BaseConfig
comptime ENTRY_BLK = 256
comptime POST_BLK = 64
comptime NUM_POST_BLK = C.INTERMEDIATE // POST_BLK
comptime DEFAULT_NUM_LAYERS = 16
comptime DEFAULT_NUM_TRIALS = 4
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime LAYER_SCALE = Float32(0.5)
comptime PHILOX_ENTRY_SEED = UInt64(0x737461636B5F7031)
comptime PHILOX_POST_SEED = UInt64(0x737461636B5F7032)
comptime PHILOX_ENTRY_SUBSEQ_BASE = UInt64(0x31)
comptime PHILOX_POST_SUBSEQ_BASE = UInt64(0x32)


def hidden_rms(x: F32Ptr) -> Float32:
    var sum_sq = Float32(0)
    for i in range(C.HIDDEN):
        sum_sq += x[i] * x[i]
    return simd_sqrt(sum_sq / Float32(C.HIDDEN))


def fill_activation_f32(mut rng: Rng, dst: F32Ptr, count: Int, stddev: Float32):
    for i in range(count):
        dst[i] = activation_like_sample_stress(rng, stddev)


def add_residual(dst: F32Ptr, src: F32Ptr, scale: Float32):
    for i in range(C.HIDDEN):
        dst[i] += src[i] * scale


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


def should_report(layer: Int, num_layers: Int, report_first: Int, report_every: Int) -> Bool:
    if layer < report_first or layer == num_layers - 1:
        return True
    if report_every > 0 and (layer + 1) % report_every == 0:
        return True
    return False


def consecutive_error_cos(
    ref_prev: F32Ptr, quant_prev: F32Ptr,
    ref_cur: F32Ptr, quant_cur: F32Ptr,
    count: Int,
) -> Float64:
    var dot = Float64(0)
    var prev_norm_sq = Float64(0)
    var cur_norm_sq = Float64(0)
    for i in range(count):
        var prev_eta = Float64(quant_prev[i] - ref_prev[i])
        var cur_eta = Float64(quant_cur[i] - ref_cur[i])
        dot += prev_eta * cur_eta
        prev_norm_sq += prev_eta * prev_eta
        cur_norm_sq += cur_eta * cur_eta
    if prev_norm_sq <= Float64(0) or cur_norm_sq <= Float64(0):
        return Float64(0)
    return dot / (Float64(simd_sqrt(Float32(prev_norm_sq))) * Float64(simd_sqrt(Float32(cur_norm_sq))))


def run_dense_block_ref(
    x_in: F32Ptr,
    gamma: BF16Ptr,
    gate_w: BF16Ptr,
    up_w: BF16Ptr,
    down_w: BF16Ptr,
    x_norm: F32Ptr,
    gate: F32Ptr,
    up: F32Ptr,
    inter: F32Ptr,
    branch: F32Ptr,
    x_out: F32Ptr,
):
    rmsnorm_gamma_from_f32(x_in, gamma, x_norm, C.HIDDEN)
    gemv_f32(gate_w, x_norm, gate, C.INTERMEDIATE, C.HIDDEN)
    gemv_f32(up_w, x_norm, up, C.INTERMEDIATE, C.HIDDEN)

    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= C.INTERMEDIATE:
        var g = (gate + k).load[width=width]()
        var u = (up + k).load[width=width]()
        (inter + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    gemv_f32(down_w, inter, branch, C.HIDDEN, C.INTERMEDIATE)
    copy_f32(x_in, x_out, C.HIDDEN)
    add_residual(x_out, branch, LAYER_SCALE)


def run_dense_block_quant(
    x_in: F32Ptr,
    gamma: BF16Ptr,
    gate_w_i8: I8Ptr,
    up_w_i8: I8Ptr,
    down_w_i8: I8Ptr,
    gate_w_scale: F32Ptr,
    up_w_scale: F32Ptr,
    down_w_scale: F32Ptr,
    gate_colsum: F32Ptr,
    up_colsum: F32Ptr,
    down_colsum: F32Ptr,
    x_norm: F32Ptr,
    x_work: F32Ptr,
    act_i8: I8Ptr,
    gate_q: F32Ptr,
    up_q: F32Ptr,
    inter: F32Ptr,
    inter_work: F32Ptr,
    inter_i8: I8Ptr,
    inter_absmax: F32Ptr,
    branch: F32Ptr,
    x_out: F32Ptr,
    use_philox: Bool,
    entry_seed: UInt64,
    entry_subsequence: UInt64,
    post_seed: UInt64,
    post_subsequence: UInt64,
):
    rmsnorm_gamma_from_f32(x_in, gamma, x_norm, C.HIDDEN)
    copy_f32(x_norm, x_work, C.HIDDEN)
    if use_philox:
        philox_fwht_rotate[ENTRY_BLK](
            x_work, C.HIDDEN, entry_seed, entry_subsequence)
    else:
        fwht_rotate[ENTRY_BLK](x_work, C.HIDDEN)
    var act_absmax = quantize_absmax_row(x_work, act_i8, C.HIDDEN)

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
        (inter + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    copy_f32(inter, inter_work, C.INTERMEDIATE)
    if use_philox:
        philox_fwht_rotate[POST_BLK](
            inter_work, C.INTERMEDIATE, post_seed, post_subsequence)
    else:
        fwht_rotate[POST_BLK](inter_work, C.INTERMEDIATE)
    quantize_absmax_block[POST_BLK](
        inter_work, inter_i8, inter_absmax, C.INTERMEDIATE)
    gemv_blocked_corrected[POST_BLK](
        inter_i8, inter_absmax,
        down_w_i8, down_w_scale, down_colsum,
        branch, C.HIDDEN, C.INTERMEDIATE)

    copy_f32(x_in, x_out, C.HIDDEN)
    add_residual(x_out, branch, LAYER_SCALE)


def run_dense_block_quant_with_signs(
    x_in: F32Ptr,
    gamma: BF16Ptr,
    gate_w_i8: I8Ptr,
    up_w_i8: I8Ptr,
    down_w_i8: I8Ptr,
    gate_w_scale: F32Ptr,
    up_w_scale: F32Ptr,
    down_w_scale: F32Ptr,
    gate_colsum: F32Ptr,
    up_colsum: F32Ptr,
    down_colsum: F32Ptr,
    x_norm: F32Ptr,
    x_work: F32Ptr,
    act_i8: I8Ptr,
    gate_q: F32Ptr,
    up_q: F32Ptr,
    inter: F32Ptr,
    inter_work: F32Ptr,
    inter_i8: I8Ptr,
    inter_absmax: F32Ptr,
    branch: F32Ptr,
    x_out: F32Ptr,
    entry_signs: F32Ptr,
    post_signs: F32Ptr,
):
    rmsnorm_gamma_from_f32(x_in, gamma, x_norm, C.HIDDEN)
    copy_f32(x_norm, x_work, C.HIDDEN)
    apply_signs_inplace(x_work, entry_signs, C.HIDDEN)
    fwht_rotate[ENTRY_BLK](x_work, C.HIDDEN)
    var act_absmax = quantize_absmax_row(x_work, act_i8, C.HIDDEN)

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
        (inter + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    copy_f32(inter, inter_work, C.INTERMEDIATE)
    apply_signs_inplace(inter_work, post_signs, C.INTERMEDIATE)
    fwht_rotate[POST_BLK](inter_work, C.INTERMEDIATE)
    quantize_absmax_block[POST_BLK](
        inter_work, inter_i8, inter_absmax, C.INTERMEDIATE)
    gemv_blocked_corrected[POST_BLK](
        inter_i8, inter_absmax,
        down_w_i8, down_w_scale, down_colsum,
        branch, C.HIDDEN, C.INTERMEDIATE)

    copy_f32(x_in, x_out, C.HIDDEN)
    add_residual(x_out, branch, LAYER_SCALE)


def run_dense_ffn_stack_analysis[num_layers: Int, num_trials: Int](
    label: String,
    report_first: Int = 8,
    report_every: Int = 25,
):
    print("=== quant_analysis/" + label + " ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " layers=" + String(num_layers)
        + " trials=" + String(num_trials)
        + " layer_scale=" + String(LAYER_SCALE)
        + " dist=stress"
        + " transform_compare=fwht_vs_philox_vs_antithetic")

    var x_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_ref_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_plain = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_plain_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_plain_local_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_philox = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_philox_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_philox_local_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_antithetic = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_antithetic_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_antithetic_local_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var gamma_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gate_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))
    var up_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.INTERMEDIATE * C.HIDDEN)))
    var down_w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN * C.INTERMEDIATE)))

    var ref_x_norm = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var ref_gate = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var ref_up = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var ref_inter = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var ref_branch = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var q_x_norm = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var q_x_work = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var q_gate = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var q_up = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var q_inter = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var q_inter_work = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var q_branch = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

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

    var plain_local_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_local_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_drift_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_drift_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_drift_rms_ratio_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_eta_prev_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_drift_rmse_worst = InlineArray[Float64, num_layers](fill=Float64(0))
    var plain_drift_cos_worst = InlineArray[Float64, num_layers](fill=Float64(1))

    var philox_local_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_local_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_drift_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_drift_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_drift_rms_ratio_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_eta_prev_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_drift_rmse_worst = InlineArray[Float64, num_layers](fill=Float64(0))
    var philox_drift_cos_worst = InlineArray[Float64, num_layers](fill=Float64(1))

    var antithetic_local_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_local_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_drift_rmse_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_drift_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_drift_rms_ratio_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_eta_prev_cos_sum = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_drift_rmse_worst = InlineArray[Float64, num_layers](fill=Float64(0))
    var antithetic_drift_cos_worst = InlineArray[Float64, num_layers](fill=Float64(1))

    var entry_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var post_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var rng = Rng(seed=0x64656E73655F7374)

    for _ in range(num_trials):
        fill_activation_f32(rng, x_ref, C.HIDDEN, X_STDDEV)
        copy_f32(x_ref, x_plain, C.HIDDEN)
        copy_f32(x_ref, x_philox, C.HIDDEN)
        copy_f32(x_ref, x_antithetic, C.HIDDEN)

        for layer in range(num_layers):
            var entry_subsequence = PHILOX_ENTRY_SUBSEQ_BASE + UInt64(layer)
            var post_subsequence = PHILOX_POST_SUBSEQ_BASE + UInt64(layer)
            var pair_entry_subsequence = PHILOX_ENTRY_SUBSEQ_BASE + UInt64(layer // 2)
            var pair_post_subsequence = PHILOX_POST_SUBSEQ_BASE + UInt64(layer // 2)

            fill_gamma_bf16(rng, gamma_bf16, C.HIDDEN, GAMMA_CENTER, GAMMA_STDDEV)
            fill_normal_bf16(rng, gate_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)
            fill_normal_bf16(rng, up_w_bf16, C.INTERMEDIATE * C.HIDDEN, W_STDDEV)
            fill_normal_bf16(rng, down_w_bf16, C.HIDDEN * C.INTERMEDIATE, W_STDDEV)

            run_dense_block_ref(
                x_ref, gamma_bf16,
                gate_w_bf16, up_w_bf16, down_w_bf16,
                ref_x_norm, ref_gate, ref_up, ref_inter, ref_branch,
                x_ref_next)

            quantize_rotated_weight_rows[ENTRY_BLK](
                gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN, row_work_hidden)
            quantize_rotated_weight_rows[ENTRY_BLK](
                up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN, row_work_hidden)
            quantize_rotated_weight_rows[POST_BLK](
                down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE, row_work_inter)
            compute_colsum_rows(gate_w_i8, C.INTERMEDIATE, C.HIDDEN, gate_colsum)
            compute_colsum_rows(up_w_i8, C.INTERMEDIATE, C.HIDDEN, up_colsum)
            compute_colsum_blocks_transposed[POST_BLK](
                down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)

            run_dense_block_quant(
                x_ref, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_plain_local_next,
                False, UInt64(0), UInt64(0), UInt64(0), UInt64(0))

            run_dense_block_quant(
                x_plain, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_plain_next,
                False, UInt64(0), UInt64(0), UInt64(0), UInt64(0))

            var plain_local_err = compute_error(x_ref_next, x_plain_local_next, C.HIDDEN)
            var plain_drift_err = compute_error(x_ref_next, x_plain_next, C.HIDDEN)

            plain_local_rmse_sum[layer] += plain_local_err.rmse
            plain_local_cos_sum[layer] += plain_local_err.cosine
            plain_drift_rmse_sum[layer] += plain_drift_err.rmse
            plain_drift_cos_sum[layer] += plain_drift_err.cosine
            plain_drift_rms_ratio_sum[layer] += Float64(hidden_rms(x_plain_next) / hidden_rms(x_ref_next))
            if layer > 0:
                plain_eta_prev_cos_sum[layer] += consecutive_error_cos(
                    x_ref, x_plain,
                    x_ref_next, x_plain_next,
                    C.HIDDEN)
            if plain_drift_err.rmse > plain_drift_rmse_worst[layer]:
                plain_drift_rmse_worst[layer] = plain_drift_err.rmse
            if plain_drift_err.cosine < plain_drift_cos_worst[layer]:
                plain_drift_cos_worst[layer] = plain_drift_err.cosine

            quantize_philox_rotated_weight_rows[ENTRY_BLK](
                gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN,
                row_work_hidden, PHILOX_ENTRY_SEED, entry_subsequence)
            quantize_philox_rotated_weight_rows[ENTRY_BLK](
                up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN,
                row_work_hidden, PHILOX_ENTRY_SEED, entry_subsequence)
            quantize_philox_rotated_weight_rows[POST_BLK](
                down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE,
                row_work_inter, PHILOX_POST_SEED, post_subsequence)
            compute_colsum_rows(gate_w_i8, C.INTERMEDIATE, C.HIDDEN, gate_colsum)
            compute_colsum_rows(up_w_i8, C.INTERMEDIATE, C.HIDDEN, up_colsum)
            compute_colsum_blocks_transposed[POST_BLK](
                down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)

            run_dense_block_quant(
                x_ref, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_philox_local_next,
                True,
                PHILOX_ENTRY_SEED, entry_subsequence,
                PHILOX_POST_SEED, post_subsequence)

            run_dense_block_quant(
                x_philox, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_philox_next,
                True,
                PHILOX_ENTRY_SEED, entry_subsequence,
                PHILOX_POST_SEED, post_subsequence)

            var philox_local_err = compute_error(x_ref_next, x_philox_local_next, C.HIDDEN)
            var philox_drift_err = compute_error(x_ref_next, x_philox_next, C.HIDDEN)

            philox_local_rmse_sum[layer] += philox_local_err.rmse
            philox_local_cos_sum[layer] += philox_local_err.cosine
            philox_drift_rmse_sum[layer] += philox_drift_err.rmse
            philox_drift_cos_sum[layer] += philox_drift_err.cosine
            philox_drift_rms_ratio_sum[layer] += Float64(hidden_rms(x_philox_next) / hidden_rms(x_ref_next))
            if layer > 0:
                philox_eta_prev_cos_sum[layer] += consecutive_error_cos(
                    x_ref, x_philox,
                    x_ref_next, x_philox_next,
                    C.HIDDEN)
            if philox_drift_err.rmse > philox_drift_rmse_worst[layer]:
                philox_drift_rmse_worst[layer] = philox_drift_err.rmse
            if philox_drift_err.cosine < philox_drift_cos_worst[layer]:
                philox_drift_cos_worst[layer] = philox_drift_err.cosine

            fill_philox_signs[ENTRY_BLK](
                entry_signs_a, C.HIDDEN, PHILOX_ENTRY_SEED, pair_entry_subsequence)
            fill_philox_signs[POST_BLK](
                post_signs_a, C.INTERMEDIATE, PHILOX_POST_SEED, pair_post_subsequence)
            reflect_neg_signs[ENTRY_BLK](entry_signs_a, entry_signs_b, C.HIDDEN)
            reflect_neg_signs[POST_BLK](post_signs_a, post_signs_b, C.INTERMEDIATE)

            if layer % 2 == 0:
                quantize_weight_rows_with_signs[ENTRY_BLK](
                    gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN,
                    row_work_hidden, entry_signs_a)
                quantize_weight_rows_with_signs[ENTRY_BLK](
                    up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN,
                    row_work_hidden, entry_signs_a)
                quantize_weight_rows_with_signs[POST_BLK](
                    down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE,
                    row_work_inter, post_signs_a)
                compute_colsum_rows(gate_w_i8, C.INTERMEDIATE, C.HIDDEN, gate_colsum)
                compute_colsum_rows(up_w_i8, C.INTERMEDIATE, C.HIDDEN, up_colsum)
                compute_colsum_blocks_transposed[POST_BLK](
                    down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)

                run_dense_block_quant_with_signs(
                    x_ref, gamma_bf16,
                    gate_w_i8, up_w_i8, down_w_i8,
                    gate_w_scale, up_w_scale, down_w_scale,
                    gate_colsum, up_colsum, down_colsum,
                    q_x_norm, q_x_work, act_i8,
                    q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                    q_branch, x_antithetic_local_next,
                    entry_signs_a, post_signs_a)
                run_dense_block_quant_with_signs(
                    x_antithetic, gamma_bf16,
                    gate_w_i8, up_w_i8, down_w_i8,
                    gate_w_scale, up_w_scale, down_w_scale,
                    gate_colsum, up_colsum, down_colsum,
                    q_x_norm, q_x_work, act_i8,
                    q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                    q_branch, x_antithetic_next,
                    entry_signs_a, post_signs_a)
            else:
                quantize_weight_rows_with_signs[ENTRY_BLK](
                    gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN,
                    row_work_hidden, entry_signs_b)
                quantize_weight_rows_with_signs[ENTRY_BLK](
                    up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN,
                    row_work_hidden, entry_signs_b)
                quantize_weight_rows_with_signs[POST_BLK](
                    down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE,
                    row_work_inter, post_signs_b)
                compute_colsum_rows(gate_w_i8, C.INTERMEDIATE, C.HIDDEN, gate_colsum)
                compute_colsum_rows(up_w_i8, C.INTERMEDIATE, C.HIDDEN, up_colsum)
                compute_colsum_blocks_transposed[POST_BLK](
                    down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)

                run_dense_block_quant_with_signs(
                    x_ref, gamma_bf16,
                    gate_w_i8, up_w_i8, down_w_i8,
                    gate_w_scale, up_w_scale, down_w_scale,
                    gate_colsum, up_colsum, down_colsum,
                    q_x_norm, q_x_work, act_i8,
                    q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                    q_branch, x_antithetic_local_next,
                    entry_signs_b, post_signs_b)
                run_dense_block_quant_with_signs(
                    x_antithetic, gamma_bf16,
                    gate_w_i8, up_w_i8, down_w_i8,
                    gate_w_scale, up_w_scale, down_w_scale,
                    gate_colsum, up_colsum, down_colsum,
                    q_x_norm, q_x_work, act_i8,
                    q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                    q_branch, x_antithetic_next,
                    entry_signs_b, post_signs_b)

            var antithetic_local_err = compute_error(x_ref_next, x_antithetic_local_next, C.HIDDEN)
            var antithetic_drift_err = compute_error(x_ref_next, x_antithetic_next, C.HIDDEN)

            antithetic_local_rmse_sum[layer] += antithetic_local_err.rmse
            antithetic_local_cos_sum[layer] += antithetic_local_err.cosine
            antithetic_drift_rmse_sum[layer] += antithetic_drift_err.rmse
            antithetic_drift_cos_sum[layer] += antithetic_drift_err.cosine
            antithetic_drift_rms_ratio_sum[layer] += Float64(hidden_rms(x_antithetic_next) / hidden_rms(x_ref_next))
            if layer > 0:
                antithetic_eta_prev_cos_sum[layer] += consecutive_error_cos(
                    x_ref, x_antithetic,
                    x_ref_next, x_antithetic_next,
                    C.HIDDEN)
            if antithetic_drift_err.rmse > antithetic_drift_rmse_worst[layer]:
                antithetic_drift_rmse_worst[layer] = antithetic_drift_err.rmse
            if antithetic_drift_err.cosine < antithetic_drift_cos_worst[layer]:
                antithetic_drift_cos_worst[layer] = antithetic_drift_err.cosine

            copy_f32(x_ref_next, x_ref, C.HIDDEN)
            copy_f32(x_plain_next, x_plain, C.HIDDEN)
            copy_f32(x_philox_next, x_philox, C.HIDDEN)
            copy_f32(x_antithetic_next, x_antithetic, C.HIDDEN)

    print("fwht")
    print("layer local_rmse local_cos drift_rmse drift_cos drift_rms_ratio eta_prev_cos worst_drift_rmse worst_drift_cos")
    for layer in range(num_layers):
        if not should_report(layer, num_layers, report_first, report_every):
            continue
        var inv_trials = Float64(1.0 / Float64(num_trials))
        var plain_eta_prev_cos_str = "na"
        if layer > 0:
            plain_eta_prev_cos_str = String(plain_eta_prev_cos_sum[layer] * inv_trials)
        print(String(layer)
            + " " + String(plain_local_rmse_sum[layer] * inv_trials)
            + " " + String(plain_local_cos_sum[layer] * inv_trials)
            + " " + String(plain_drift_rmse_sum[layer] * inv_trials)
            + " " + String(plain_drift_cos_sum[layer] * inv_trials)
            + " " + String(plain_drift_rms_ratio_sum[layer] * inv_trials)
            + " " + plain_eta_prev_cos_str
            + " " + String(plain_drift_rmse_worst[layer])
            + " " + String(plain_drift_cos_worst[layer]))

    print("philox_fwht")
    print("layer local_rmse local_cos drift_rmse drift_cos drift_rms_ratio eta_prev_cos worst_drift_rmse worst_drift_cos")
    for layer in range(num_layers):
        if not should_report(layer, num_layers, report_first, report_every):
            continue
        var inv_trials = Float64(1.0 / Float64(num_trials))
        var philox_eta_prev_cos_str = "na"
        if layer > 0:
            philox_eta_prev_cos_str = String(philox_eta_prev_cos_sum[layer] * inv_trials)
        print(String(layer)
            + " " + String(philox_local_rmse_sum[layer] * inv_trials)
            + " " + String(philox_local_cos_sum[layer] * inv_trials)
            + " " + String(philox_drift_rmse_sum[layer] * inv_trials)
            + " " + String(philox_drift_cos_sum[layer] * inv_trials)
            + " " + String(philox_drift_rms_ratio_sum[layer] * inv_trials)
            + " " + philox_eta_prev_cos_str
            + " " + String(philox_drift_rmse_worst[layer])
            + " " + String(philox_drift_cos_worst[layer]))

    print("antithetic_pair_fwht")
    print("layer local_rmse local_cos drift_rmse drift_cos drift_rms_ratio eta_prev_cos worst_drift_rmse worst_drift_cos")
    for layer in range(num_layers):
        if not should_report(layer, num_layers, report_first, report_every):
            continue
        var inv_trials = Float64(1.0 / Float64(num_trials))
        var antithetic_eta_prev_cos_str = "na"
        if layer > 0:
            antithetic_eta_prev_cos_str = String(antithetic_eta_prev_cos_sum[layer] * inv_trials)
        print(String(layer)
            + " " + String(antithetic_local_rmse_sum[layer] * inv_trials)
            + " " + String(antithetic_local_cos_sum[layer] * inv_trials)
            + " " + String(antithetic_drift_rmse_sum[layer] * inv_trials)
            + " " + String(antithetic_drift_cos_sum[layer] * inv_trials)
            + " " + String(antithetic_drift_rms_ratio_sum[layer] * inv_trials)
            + " " + antithetic_eta_prev_cos_str
            + " " + String(antithetic_drift_rmse_worst[layer])
            + " " + String(antithetic_drift_cos_worst[layer]))

    x_ref.free()
    x_ref_next.free()
    x_plain.free()
    x_plain_next.free()
    x_plain_local_next.free()
    x_philox.free()
    x_philox_next.free()
    x_philox_local_next.free()
    x_antithetic.free()
    x_antithetic_next.free()
    x_antithetic_local_next.free()
    gamma_bf16.free()
    gate_w_bf16.free()
    up_w_bf16.free()
    down_w_bf16.free()
    ref_x_norm.free()
    ref_gate.free()
    ref_up.free()
    ref_inter.free()
    ref_branch.free()
    q_x_norm.free()
    q_x_work.free()
    q_gate.free()
    q_up.free()
    q_inter.free()
    q_inter_work.free()
    q_branch.free()
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
    entry_signs_a.free()
    entry_signs_b.free()
    post_signs_a.free()
    post_signs_b.free()


def main():
    run_dense_ffn_stack_analysis[DEFAULT_NUM_LAYERS, DEFAULT_NUM_TRIALS](
        "dense_ffn_stack")
