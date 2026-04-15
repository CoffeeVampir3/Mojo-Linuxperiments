"""Terminal-direction alignment analysis for dense FFN stack rollout.

This harness tests two medium-depth diagnostics over a 24-layer rollout:

1. Terminal-direction alignment
   Let `v = eta_L / ||eta_L||` where `eta_l = x_quant^(l) - x_ref^(l)`.
   For each layer `l`, report
   `<v, eta_l> / ||eta_l||`.

2. Clean-input injection projection
   Let `xi_l` be the single-layer quantization injection at layer `l`, i.e.
   the difference between the quantized and reference layer outputs when both
   receive the clean reference hidden state as input.
   For each layer `l`, report the mean and variance of `<v, xi_l>` across
   trials.

The goal is to separate "all layers are locally aligned" from "all layers are
aligning to the same terminal drift direction."
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
comptime NUM_LAYERS = 24
comptime NUM_TRIALS = 12
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime PHILOX_ENTRY_SEED = UInt64(0x737461636B5F7031)
comptime PHILOX_POST_SEED = UInt64(0x737461636B5F7032)
comptime PHILOX_ENTRY_SUBSEQ_BASE = UInt64(0x31)
comptime PHILOX_POST_SUBSEQ_BASE = UInt64(0x32)


def store_error(ref_buf: F32Ptr, quant_buf: F32Ptr, dst: F32Ptr, count: Int):
    for i in range(count):
        dst[i] = quant_buf[i] - ref_buf[i]


def vector_norm(buf: F32Ptr, count: Int) -> Float64:
    var sum_sq = Float64(0)
    for i in range(count):
        var v = Float64(buf[i])
        sum_sq += v * v
    return Float64(simd_sqrt(Float32(sum_sq)))


def normalize_into(src: F32Ptr, dst: F32Ptr, count: Int) -> Float64:
    var norm = vector_norm(src, count)
    if norm <= Float64(0):
        for i in range(count):
            dst[i] = Float32(0)
        return Float64(0)
    var inv = Float32(1.0 / norm)
    for i in range(count):
        dst[i] = src[i] * inv
    return norm


def dot_with_unit(unit: F32Ptr, buf: F32Ptr, count: Int) -> Float64:
    var dot = Float64(0)
    for i in range(count):
        dot += Float64(unit[i]) * Float64(buf[i])
    return dot


def alignment_with_unit(unit: F32Ptr, buf: F32Ptr, count: Int) -> Float64:
    var norm = vector_norm(buf, count)
    if norm <= Float64(0):
        return Float64(0)
    return dot_with_unit(unit, buf, count) / norm


def main():
    print("=== quant_analysis/dense_ffn_stack_terminal_direction ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " layers=" + String(NUM_LAYERS)
        + " trials=" + String(NUM_TRIALS)
        + " metric=terminal_direction_alignment_and_injection_projection")

    var x_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_ref_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_fwht = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_fwht_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var x_fwht_local_next = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
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

    var entry_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var entry_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var post_signs_a = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))
    var post_signs_b = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.INTERMEDIATE)))

    var fwht_eta = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))
    var fwht_xi = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))
    var philox_eta = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))
    var philox_xi = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))
    var antithetic_eta = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))
    var antithetic_xi = F32Ptr(unsafe_from_address=Int(alloc[Float32](NUM_LAYERS * C.HIDDEN)))

    var v_fwht = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var v_philox = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var v_antithetic = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var fwht_align_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var philox_align_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var antithetic_align_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var fwht_proj_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var fwht_proj_sq_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var philox_proj_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var philox_proj_sq_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var antithetic_proj_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))
    var antithetic_proj_sq_sum = InlineArray[Float64, NUM_LAYERS](fill=Float64(0))

    var rng = Rng(seed=0x7465726D5F646972)

    for _ in range(NUM_TRIALS):
        fill_activation_f32(rng, x_ref, C.HIDDEN, X_STDDEV)
        copy_f32(x_ref, x_fwht, C.HIDDEN)
        copy_f32(x_ref, x_philox, C.HIDDEN)
        copy_f32(x_ref, x_antithetic, C.HIDDEN)

        for layer in range(NUM_LAYERS):
            var eta_off = layer * C.HIDDEN
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
                q_branch, x_fwht_local_next,
                False, UInt64(0), UInt64(0), UInt64(0), UInt64(0))
            run_dense_block_quant(
                x_fwht, gamma_bf16,
                gate_w_i8, up_w_i8, down_w_i8,
                gate_w_scale, up_w_scale, down_w_scale,
                gate_colsum, up_colsum, down_colsum,
                q_x_norm, q_x_work, act_i8,
                q_gate, q_up, q_inter, q_inter_work, inter_i8, inter_absmax,
                q_branch, x_fwht_next,
                False, UInt64(0), UInt64(0), UInt64(0), UInt64(0))
            store_error(x_ref_next, x_fwht_next, fwht_eta + eta_off, C.HIDDEN)
            store_error(x_ref_next, x_fwht_local_next, fwht_xi + eta_off, C.HIDDEN)

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
            store_error(x_ref_next, x_philox_next, philox_eta + eta_off, C.HIDDEN)
            store_error(x_ref_next, x_philox_local_next, philox_xi + eta_off, C.HIDDEN)

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
            store_error(x_ref_next, x_antithetic_next, antithetic_eta + eta_off, C.HIDDEN)
            store_error(x_ref_next, x_antithetic_local_next, antithetic_xi + eta_off, C.HIDDEN)

            copy_f32(x_ref_next, x_ref, C.HIDDEN)
            copy_f32(x_fwht_next, x_fwht, C.HIDDEN)
            copy_f32(x_philox_next, x_philox, C.HIDDEN)
            copy_f32(x_antithetic_next, x_antithetic, C.HIDDEN)

        _ = normalize_into(fwht_eta + (NUM_LAYERS - 1) * C.HIDDEN, v_fwht, C.HIDDEN)
        _ = normalize_into(philox_eta + (NUM_LAYERS - 1) * C.HIDDEN, v_philox, C.HIDDEN)
        _ = normalize_into(antithetic_eta + (NUM_LAYERS - 1) * C.HIDDEN, v_antithetic, C.HIDDEN)

        for layer in range(NUM_LAYERS):
            var eta_off = layer * C.HIDDEN

            var fwht_align = alignment_with_unit(v_fwht, fwht_eta + eta_off, C.HIDDEN)
            var fwht_proj = dot_with_unit(v_fwht, fwht_xi + eta_off, C.HIDDEN)
            fwht_align_sum[layer] += fwht_align
            fwht_proj_sum[layer] += fwht_proj
            fwht_proj_sq_sum[layer] += fwht_proj * fwht_proj

            var philox_align = alignment_with_unit(v_philox, philox_eta + eta_off, C.HIDDEN)
            var philox_proj = dot_with_unit(v_philox, philox_xi + eta_off, C.HIDDEN)
            philox_align_sum[layer] += philox_align
            philox_proj_sum[layer] += philox_proj
            philox_proj_sq_sum[layer] += philox_proj * philox_proj

            var antithetic_align = alignment_with_unit(v_antithetic, antithetic_eta + eta_off, C.HIDDEN)
            var antithetic_proj = dot_with_unit(v_antithetic, antithetic_xi + eta_off, C.HIDDEN)
            antithetic_align_sum[layer] += antithetic_align
            antithetic_proj_sum[layer] += antithetic_proj
            antithetic_proj_sq_sum[layer] += antithetic_proj * antithetic_proj

    print("fwht")
    print("layer eta_to_v mean_xi_proj var_xi_proj")
    for layer in range(NUM_LAYERS):
        var inv = Float64(1.0 / Float64(NUM_TRIALS))
        var mean_proj = fwht_proj_sum[layer] * inv
        var var_proj = fwht_proj_sq_sum[layer] * inv - mean_proj * mean_proj
        print(String(layer)
            + " " + String(fwht_align_sum[layer] * inv)
            + " " + String(mean_proj)
            + " " + String(var_proj))

    print("philox_fwht")
    print("layer eta_to_v mean_xi_proj var_xi_proj")
    for layer in range(NUM_LAYERS):
        var inv = Float64(1.0 / Float64(NUM_TRIALS))
        var mean_proj = philox_proj_sum[layer] * inv
        var var_proj = philox_proj_sq_sum[layer] * inv - mean_proj * mean_proj
        print(String(layer)
            + " " + String(philox_align_sum[layer] * inv)
            + " " + String(mean_proj)
            + " " + String(var_proj))

    print("antithetic_pair_fwht")
    print("layer eta_to_v mean_xi_proj var_xi_proj")
    for layer in range(NUM_LAYERS):
        var inv = Float64(1.0 / Float64(NUM_TRIALS))
        var mean_proj = antithetic_proj_sum[layer] * inv
        var var_proj = antithetic_proj_sq_sum[layer] * inv - mean_proj * mean_proj
        print(String(layer)
            + " " + String(antithetic_align_sum[layer] * inv)
            + " " + String(mean_proj)
            + " " + String(var_proj))

    x_ref.free()
    x_ref_next.free()
    x_fwht.free()
    x_fwht_next.free()
    x_fwht_local_next.free()
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
    fwht_eta.free()
    fwht_xi.free()
    philox_eta.free()
    philox_xi.free()
    antithetic_eta.free()
    antithetic_xi.free()
    v_fwht.free()
    v_philox.free()
    v_antithetic.free()
