"""Closed-loop error covariance analysis for dense FFN stack techniques.

This focuses on 10-layer rollout behavior and measures pairwise covariance of
the error vectors produced by:

- fwht
- philox_fwht
- alternating reflected-antithetic A/B pairing

The reported covariance is over hidden-dimension coordinates of the closed-loop
error vectors `e = x_quant - x_ref` at each layer, averaged over trials.
"""

from std.collections import InlineArray
from std.memory.unsafe_pointer import alloc

from simd_math import sqrt as simd_sqrt
from modeling.gemma4_common import Gemma4BaseConfig
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_gamma_bf16, fill_normal_bf16,
    fill_philox_signs,
    copy_f32,
    quantize_rotated_weight_rows, compute_colsum_rows,
    quantize_philox_rotated_weight_rows,
    compute_colsum_blocks_transposed,
)
from quant_analysis.dense_ffn_stack import (
    fill_activation_f32,
    reflect_neg_signs,
    quantize_weight_rows_with_signs,
    run_dense_block_ref,
    run_dense_block_quant,
    run_dense_block_quant_with_signs,
)


comptime C = Gemma4BaseConfig
comptime ENTRY_BLK = 256
comptime POST_BLK = 64
comptime NUM_POST_BLK = C.INTERMEDIATE // POST_BLK
comptime NUM_LAYERS = 10
comptime NUM_TRIALS = 12
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime PHILOX_ENTRY_SEED = UInt64(0x737461636B5F7031)
comptime PHILOX_POST_SEED = UInt64(0x737461636B5F7032)
comptime PHILOX_ENTRY_SUBSEQ_BASE = UInt64(0x31)
comptime PHILOX_POST_SUBSEQ_BASE = UInt64(0x32)


@fieldwise_init
struct PairCovariance(Copyable, ImplicitlyCopyable):
    var var_a: Float64
    var var_b: Float64
    var cov_ab: Float64
    var corr_ab: Float64


def error_pair_covariance(
    ref_buf: F32Ptr, a: F32Ptr, b: F32Ptr, count: Int,
) -> PairCovariance:
    var sum_a = Float64(0)
    var sum_b = Float64(0)
    var sum_aa = Float64(0)
    var sum_bb = Float64(0)
    var sum_ab = Float64(0)
    var inv_n = Float64(1.0 / Float64(count))

    for i in range(count):
        var ea = Float64(a[i] - ref_buf[i])
        var eb = Float64(b[i] - ref_buf[i])
        sum_a += ea
        sum_b += eb
        sum_aa += ea * ea
        sum_bb += eb * eb
        sum_ab += ea * eb

    var mean_a = sum_a * inv_n
    var mean_b = sum_b * inv_n
    var var_a = sum_aa * inv_n - mean_a * mean_a
    var var_b = sum_bb * inv_n - mean_b * mean_b
    var cov_ab = sum_ab * inv_n - mean_a * mean_b
    var corr_ab = Float64(0)
    if var_a > Float64(0) and var_b > Float64(0):
        corr_ab = cov_ab / (Float64(simd_sqrt(Float32(var_a))) * Float64(simd_sqrt(Float32(var_b))))
    return PairCovariance(var_a, var_b, cov_ab, corr_ab)


def main():
    print("=== quant_analysis/dense_ffn_stack_covariance ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " layers=" + String(NUM_LAYERS)
        + " trials=" + String(NUM_TRIALS)
        + " metric=closed_loop_error_covariance")

    var x_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_ref_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_fwht = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_fwht_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_philox = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_philox_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_antithetic = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_antithetic_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

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

    var entry_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var post_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var fwht_var_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var philox_var_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var antithetic_var_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var cov_fwht_philox_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var corr_fwht_philox_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var cov_fwht_antithetic_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var corr_fwht_antithetic_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var cov_philox_antithetic_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var corr_philox_antithetic_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))

    var rng = Rng(seed=0x636F765F73746163)

    for _ in range(NUM_TRIALS):
        fill_activation_f32(rng, x_ref, C.HIDDEN, X_STDDEV)
        copy_f32(x_ref, x_fwht, C.HIDDEN)
        copy_f32(x_ref, x_philox, C.HIDDEN)
        copy_f32(x_ref, x_antithetic, C.HIDDEN)

        for layer in range(NUM_LAYERS):
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
                x_fwht, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_fwht_next,
                False, UInt64(0), UInt64(0), UInt64(0), UInt64(0))

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
                    x_antithetic, gamma_bf16,
                    gate_w_i8, up_w_i8, down_w_i8,
                    gate_w_scale, up_w_scale, down_w_scale,
                    gate_colsum, up_colsum, down_colsum,
                    q_x_norm, q_x_work, act_i8,
                    q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                    q_branch, x_antithetic_next,
                    entry_signs_b, post_signs_b)

            var fwht_philox = error_pair_covariance(x_ref_next, x_fwht_next, x_philox_next, C.HIDDEN)
            var fwht_antithetic = error_pair_covariance(x_ref_next, x_fwht_next, x_antithetic_next, C.HIDDEN)
            var philox_antithetic = error_pair_covariance(x_ref_next, x_philox_next, x_antithetic_next, C.HIDDEN)

            fwht_var_sum[layer] += fwht_philox.var_a
            philox_var_sum[layer] += fwht_philox.var_b
            antithetic_var_sum[layer] += fwht_antithetic.var_b
            cov_fwht_philox_sum[layer] += fwht_philox.cov_ab
            corr_fwht_philox_sum[layer] += fwht_philox.corr_ab
            cov_fwht_antithetic_sum[layer] += fwht_antithetic.cov_ab
            corr_fwht_antithetic_sum[layer] += fwht_antithetic.corr_ab
            cov_philox_antithetic_sum[layer] += philox_antithetic.cov_ab
            corr_philox_antithetic_sum[layer] += philox_antithetic.corr_ab

            copy_f32(x_ref_next, x_ref, C.HIDDEN)
            copy_f32(x_fwht_next, x_fwht, C.HIDDEN)
            copy_f32(x_philox_next, x_philox, C.HIDDEN)
            copy_f32(x_antithetic_next, x_antithetic, C.HIDDEN)

    print("layer var_fwht var_philox var_antithetic cov_fwht_philox corr_fwht_philox cov_fwht_antithetic corr_fwht_antithetic cov_philox_antithetic corr_philox_antithetic")
    for layer in range(NUM_LAYERS):
        var inv = Float64(1.0 / Float64(NUM_TRIALS))
        print(String(layer)
            + " " + String(fwht_var_sum[layer] * inv)
            + " " + String(philox_var_sum[layer] * inv)
            + " " + String(antithetic_var_sum[layer] * inv)
            + " " + String(cov_fwht_philox_sum[layer] * inv)
            + " " + String(corr_fwht_philox_sum[layer] * inv)
            + " " + String(cov_fwht_antithetic_sum[layer] * inv)
            + " " + String(corr_fwht_antithetic_sum[layer] * inv)
            + " " + String(cov_philox_antithetic_sum[layer] * inv)
            + " " + String(corr_philox_antithetic_sum[layer] * inv))

    x_ref.free()
    x_ref_next.free()
    x_fwht.free()
    x_fwht_next.free()
    x_philox.free()
    x_philox_next.free()
    x_antithetic.free()
    x_antithetic_next.free()
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
