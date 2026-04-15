"""Mock of the blocked post-nonlinearity FFN scheme.

Representative layer: dense FFN.

Baseline:
  x_norm = gamma .* x / rms(x)
  gate = W_gate * x_norm
  up   = W_up   * x_norm
  inter = gelu_tanh(gate) .* up
  y_ref = W_down * inter

Quantized path:
  1. Entry quantization on x_norm: FWHT(256) + rowwise absmax i8
  2. gate/up use rotated rowwise i8 weights + corrected int GEMV
  3. inter_quant = gelu_tanh(gate_q) .* up_q
  4. Post-activation re-entry: FWHT(64) + per-block absmax i8
  5. down uses rotated rowwise i8 weights + blocked corrected int GEMV

Reports RMSE and cosine similarity against the bf16 baseline.
"""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from modeling.gemma4_common import Gemma4BaseConfig
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_activation_bf16, fill_normal_bf16, fill_gamma_bf16,
    rmsnorm_gamma_from_bf16, copy_f32, fwht_rotate, philox_fwht_rotate,
    quantize_absmax_row, quantize_absmax_block,
    quantize_rotated_weight_rows, compute_colsum_rows,
    quantize_philox_rotated_weight_rows,
    compute_colsum_blocks_transposed, gemv_f32,
    gemv_rowwise_corrected, gemv_blocked_corrected,
    compute_error, init_aggregate, add_report, print_summary,
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
comptime PHILOX_ENTRY_SEED = UInt64(0x64656E73655F7031)
comptime PHILOX_POST_SEED = UInt64(0x64656E73655F7032)
comptime PHILOX_ENTRY_SUBSEQ = UInt64(0x11)
comptime PHILOX_POST_SUBSEQ = UInt64(0x12)


def main():
    print("=== quant_analysis/dense_ffn ===")
    print("hidden=" + String(C.HIDDEN)
        + " intermediate=" + String(C.INTERMEDIATE)
        + " entry_blk=" + String(ENTRY_BLK)
        + " post_blk=" + String(POST_BLK)
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
    var y_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

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

    var rng = Rng(seed=0x64656E73655F6666)
    var agg_plain = init_aggregate()
    var agg_philox = init_aggregate()
    comptime width = simd_width_of[DType.float32]()

    for _ in range(NUM_TRIALS):
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

        copy_f32(x_norm, x_work, C.HIDDEN)
        fwht_rotate[ENTRY_BLK](x_work, C.HIDDEN)
        var act_absmax = quantize_absmax_row(x_work, act_i8, C.HIDDEN)

        quantize_rotated_weight_rows[ENTRY_BLK](
            gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN, row_work_hidden)
        quantize_rotated_weight_rows[ENTRY_BLK](
            up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN, row_work_hidden)
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

        k = 0
        while k + width <= C.INTERMEDIATE:
            var g = (gate_q + k).load[width=width]()
            var u = (up_q + k).load[width=width]()
            (inter_q + k).store(gelu_tanh_f32[width](g) * u)
            k += width

        copy_f32(inter_q, row_work_inter, C.INTERMEDIATE)
        fwht_rotate[POST_BLK](row_work_inter, C.INTERMEDIATE)
        quantize_absmax_block[POST_BLK](row_work_inter, inter_i8, inter_absmax, C.INTERMEDIATE)

        quantize_rotated_weight_rows[POST_BLK](
            down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE, row_work_inter)
        compute_colsum_blocks_transposed[POST_BLK](
            down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)
        gemv_blocked_corrected[POST_BLK](
            inter_i8, inter_absmax,
            down_w_i8, down_w_scale, down_colsum,
            y_q, C.HIDDEN, C.INTERMEDIATE)
        add_report(agg_plain, compute_error(y_ref, y_q, C.HIDDEN))

        copy_f32(x_norm, x_work, C.HIDDEN)
        philox_fwht_rotate[ENTRY_BLK](x_work, C.HIDDEN, PHILOX_ENTRY_SEED, PHILOX_ENTRY_SUBSEQ)
        act_absmax = quantize_absmax_row(x_work, act_i8, C.HIDDEN)

        quantize_philox_rotated_weight_rows[ENTRY_BLK](
            gate_w_bf16, gate_w_i8, gate_w_scale, C.INTERMEDIATE, C.HIDDEN,
            row_work_hidden, PHILOX_ENTRY_SEED, PHILOX_ENTRY_SUBSEQ)
        quantize_philox_rotated_weight_rows[ENTRY_BLK](
            up_w_bf16, up_w_i8, up_w_scale, C.INTERMEDIATE, C.HIDDEN,
            row_work_hidden, PHILOX_ENTRY_SEED, PHILOX_ENTRY_SUBSEQ)
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

        k = 0
        while k + width <= C.INTERMEDIATE:
            var g = (gate_q + k).load[width=width]()
            var u = (up_q + k).load[width=width]()
            (inter_q + k).store(gelu_tanh_f32[width](g) * u)
            k += width

        copy_f32(inter_q, row_work_inter, C.INTERMEDIATE)
        philox_fwht_rotate[POST_BLK](
            row_work_inter, C.INTERMEDIATE, PHILOX_POST_SEED, PHILOX_POST_SUBSEQ)
        quantize_absmax_block[POST_BLK](row_work_inter, inter_i8, inter_absmax, C.INTERMEDIATE)

        quantize_philox_rotated_weight_rows[POST_BLK](
            down_w_bf16, down_w_i8, down_w_scale, C.HIDDEN, C.INTERMEDIATE,
            row_work_inter, PHILOX_POST_SEED, PHILOX_POST_SUBSEQ)
        compute_colsum_blocks_transposed[POST_BLK](
            down_w_i8, C.HIDDEN, C.INTERMEDIATE, down_colsum)
        gemv_blocked_corrected[POST_BLK](
            inter_i8, inter_absmax,
            down_w_i8, down_w_scale, down_colsum,
            y_q, C.HIDDEN, C.INTERMEDIATE)

        add_report(agg_philox, compute_error(y_ref, y_q, C.HIDDEN))
    print_summary("dense_ffn/fwht", agg_plain)
    print_summary("dense_ffn/philox_fwht", agg_philox)

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
    y_q.free()
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
