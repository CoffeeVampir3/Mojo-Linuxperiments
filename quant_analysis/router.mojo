"""Mock of the generic rotated rowwise projection scheme.

Representative layer: router projection.

Baseline:
  y_ref = W * (gamma .* x / rms(x))

Quantized path:
  1. z = H(gamma .* x / rms(x))
  2. q_x = absmax_i8(z), store activation absmax
  3. q_w = absmax_i8(H(W_row)), store weight scale = absmax/127
  4. y_q = corrected_int_dot(q_x, q_w)

Reports RMSE and cosine similarity against the bf16 baseline.
"""

from std.memory.unsafe_pointer import alloc

from modeling.gemma4_common import Gemma4BaseConfig
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_activation_bf16, fill_normal_bf16, fill_gamma_bf16,
    rmsnorm_gamma_from_bf16, copy_f32, fwht_rotate, philox_fwht_rotate,
    quantize_absmax_row, quantize_rotated_weight_rows,
    quantize_philox_rotated_weight_rows,
    compute_colsum_rows, gemv_f32, gemv_rowwise_corrected,
    compute_error, init_aggregate, add_report, print_summary,
)


comptime C = Gemma4BaseConfig
comptime ROT_BLK = 256
comptime OUT_ROWS = C.NUM_EXPERTS
comptime NUM_TRIALS = 96
comptime X_STDDEV = Float32(0.8)
comptime W_STDDEV = Float32(0.02)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)
comptime PHILOX_SEED = UInt64(0x726F757465725F70)
comptime PHILOX_SUBSEQ = UInt64(0x01)


def main():
    print("=== quant_analysis/router ===")
    print("hidden=" + String(C.HIDDEN)
        + " rows=" + String(OUT_ROWS)
        + " rot_blk=" + String(ROT_BLK)
        + " trials=" + String(NUM_TRIALS))

    var x_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var gamma_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](C.HIDDEN)))
    var w_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](OUT_ROWS * C.HIDDEN)))

    var x_norm = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var work = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))
    var y_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](OUT_ROWS)))
    var y_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](OUT_ROWS)))
    var q_x = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](C.HIDDEN)))
    var q_w = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](OUT_ROWS * C.HIDDEN)))
    var w_scale = F32Ptr(unsafe_from_address=Int(alloc[Float32](OUT_ROWS)))
    var w_colsum = F32Ptr(unsafe_from_address=Int(alloc[Float32](OUT_ROWS)))
    var row_work = F32Ptr(unsafe_from_address=Int(alloc[Float32](C.HIDDEN)))

    var rng = Rng(seed=0x726F757465725F31)
    var agg_plain = init_aggregate()
    var agg_philox = init_aggregate()

    for _ in range(NUM_TRIALS):
        fill_activation_bf16(rng, x_bf16, C.HIDDEN, X_STDDEV)
        fill_gamma_bf16(rng, gamma_bf16, C.HIDDEN, GAMMA_CENTER, GAMMA_STDDEV)
        fill_normal_bf16(rng, w_bf16, OUT_ROWS * C.HIDDEN, W_STDDEV)

        rmsnorm_gamma_from_bf16(x_bf16, gamma_bf16, x_norm, C.HIDDEN)
        gemv_f32(w_bf16, x_norm, y_ref, OUT_ROWS, C.HIDDEN)

        copy_f32(x_norm, work, C.HIDDEN)
        fwht_rotate[ROT_BLK](work, C.HIDDEN)
        var act_absmax = quantize_absmax_row(work, q_x, C.HIDDEN)

        quantize_rotated_weight_rows[ROT_BLK](
            w_bf16, q_w, w_scale, OUT_ROWS, C.HIDDEN, row_work)
        compute_colsum_rows(q_w, OUT_ROWS, C.HIDDEN, w_colsum)
        gemv_rowwise_corrected(
            q_x, act_absmax,
            q_w, w_scale, w_colsum,
            y_q, OUT_ROWS, C.HIDDEN)
        add_report(agg_plain, compute_error(y_ref, y_q, OUT_ROWS))

        copy_f32(x_norm, work, C.HIDDEN)
        philox_fwht_rotate[ROT_BLK](work, C.HIDDEN, PHILOX_SEED, PHILOX_SUBSEQ)
        act_absmax = quantize_absmax_row(work, q_x, C.HIDDEN)

        quantize_philox_rotated_weight_rows[ROT_BLK](
            w_bf16, q_w, w_scale, OUT_ROWS, C.HIDDEN, row_work, PHILOX_SEED, PHILOX_SUBSEQ)
        compute_colsum_rows(q_w, OUT_ROWS, C.HIDDEN, w_colsum)
        gemv_rowwise_corrected(
            q_x, act_absmax,
            q_w, w_scale, w_colsum,
            y_q, OUT_ROWS, C.HIDDEN)
        add_report(agg_philox, compute_error(y_ref, y_q, OUT_ROWS))

    print_summary("router/fwht", agg_plain)
    print_summary("router/philox_fwht", agg_philox)

    x_bf16.free()
    gamma_bf16.free()
    w_bf16.free()
    x_norm.free()
    work.free()
    y_ref.free()
    y_q.free()
    q_x.free()
    q_w.free()
    w_scale.free()
    w_colsum.free()
    row_work.free()
