"""Mock of the quantized sliding-attention subsystem before O projection.

Representative layer: sliding self attention.

Baseline:
  q' = rope(q_norm .* q / rms(q))
  k_t' = rope(k_norm .* k_t / rms(k_t))
  v_tilde = H(v_t / rms(v_t))
  out_ref = sum_t softmax(dot(q', k_t')) * v_tilde_t

Quantized path:
  1. q, k use dynamic signed absmax i8 after RoPE + FWHT
  2. scores use the corrected QK dot dequant formula
  3. exp(scores) * v_absmax is dynamically quantized to u8 per position group
  4. final attention output is absmax-quantized to i8 per head

Reports RMSE and cosine similarity against the bf16 baseline at the attention
handoff point, i.e. after the final output quantization/dequantization and
before O projection.
"""

from std.math import exp
from std.memory.unsafe_pointer import alloc
from std.collections import InlineArray

from simd_math import sincos, roundeven
from modeling.gemma4_common import Gemma4BaseConfig
from quant_analysis.common import (
    BF16Ptr, F32Ptr, I8Ptr, Rng,
    fill_activation_bf16, fill_gamma_bf16,
    copy_f32, rmsnorm_from_bf16, rmsnorm_f32_inplace, fwht_rotate,
    quantize_absmax_row, dequant_row,
    compute_error, init_aggregate, add_report, print_summary,
)


comptime C = Gemma4BaseConfig
comptime HEAD_DIM = C.HEAD_DIM_SLIDING
comptime HALF = HEAD_DIM // 2
comptime HEADS_PER_GROUP = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
comptime CONTEXT_LEN = 128
comptime POS_GROUP = 16
comptime NUM_TRIALS = 48
comptime X_STDDEV = Float32(0.8)
comptime GAMMA_CENTER = Float32(1.0)
comptime GAMMA_STDDEV = Float32(0.15)


def fill_rotary_tables(cos_tbl: F32Ptr, sin_tbl: F32Ptr):
    for pos in range(CONTEXT_LEN):
        var row_off = pos * HALF
        for i in range(HALF):
            var angle = SIMD[DType.float64, 1](Float64(pos + 1) * Float64(i + 1) * Float64(0.0015))
            var sc = sincos[1](angle)
            cos_tbl[row_off + i] = Float32(sc.cos_val)
            sin_tbl[row_off + i] = Float32(sc.sin_val)


def apply_rope(buf: F32Ptr, cos_row: F32Ptr, sin_row: F32Ptr):
    for i in range(HALF):
        var x_lo = buf[i]
        var x_hi = buf[HALF + i]
        var c = cos_row[i]
        var s = sin_row[i]
        buf[i] = x_lo * c - x_hi * s
        buf[HALF + i] = x_hi * c + x_lo * s


def dot_f32(a: F32Ptr, b: F32Ptr, count: Int) -> Float32:
    var acc = Float32(0)
    for i in range(count):
        acc += a[i] * b[i]
    return acc


def softmax(scores: F32Ptr, probs: F32Ptr, count: Int):
    var max_score = Float32(-1e30)
    for i in range(count):
        if scores[i] > max_score:
            max_score = scores[i]
    var sum_exp = Float32(0)
    for i in range(count):
        var v = Float32(exp(scores[i] - max_score))
        probs[i] = v
        sum_exp += v
    var inv = Float32(1.0) / sum_exp
    for i in range(count):
        probs[i] *= inv


def score_quantized(
    q_i8: I8Ptr,
    q_bias: Float32,
    q_absmax: Float32,
    k_i8: I8Ptr,
    k_absmax: Float32,
) -> Float32:
    var acc = Int(0)
    for i in range(HEAD_DIM):
        acc += (Int(k_i8[i]) + 128) * Int(q_i8[i])
    var corrected = Float32(acc) - q_bias
    return corrected * (q_absmax / (Float32(127.0) * Float32(127.0))) * k_absmax


def prep_q_head(
    q_src: BF16Ptr,
    q_norm: BF16Ptr,
    cos_row: F32Ptr,
    sin_row: F32Ptr,
    q_proc: F32Ptr,
    q_i8: I8Ptr,
) -> Tuple[Float32, Float32]:
    rmsnorm_from_bf16(q_src, q_proc, HEAD_DIM)
    for i in range(HEAD_DIM):
        q_proc[i] *= Float32(q_norm[i])
    apply_rope(q_proc, cos_row, sin_row)
    fwht_rotate[HEAD_DIM](q_proc, HEAD_DIM)
    var q_absmax = quantize_absmax_row(q_proc, q_i8, HEAD_DIM)
    var q_sum = Int(0)
    for i in range(HEAD_DIM):
        q_sum += Int(q_i8[i])
    return (Float32(q_sum) * Float32(128.0), q_absmax)


def prep_kv_position(
    k_src: BF16Ptr,
    v_src: BF16Ptr,
    k_norm: BF16Ptr,
    cos_row: F32Ptr,
    sin_row: F32Ptr,
    k_proc: F32Ptr,
    v_tilde: F32Ptr,
    k_i8: I8Ptr,
    v_i8: I8Ptr,
) -> Tuple[Float32, Float32]:
    rmsnorm_from_bf16(k_src, k_proc, HEAD_DIM)
    for i in range(HEAD_DIM):
        k_proc[i] *= Float32(k_norm[i])
    apply_rope(k_proc, cos_row, sin_row)

    rmsnorm_from_bf16(v_src, v_tilde, HEAD_DIM)
    fwht_rotate[HEAD_DIM](v_tilde, HEAD_DIM)

    fwht_rotate[HEAD_DIM](k_proc, HEAD_DIM)
    var k_absmax = quantize_absmax_row(k_proc, k_i8, HEAD_DIM)
    var v_absmax = quantize_absmax_row(v_tilde, v_i8, HEAD_DIM)
    return (k_absmax, v_absmax)


def quantized_attention_one_head(
    q_i8: I8Ptr,
    q_bias: Float32,
    q_absmax: Float32,
    k_i8_all: I8Ptr,
    v_i8_all: I8Ptr,
    k_absmax_all: F32Ptr,
    v_absmax_all: F32Ptr,
    out_tilde: F32Ptr,
):
    for d in range(HEAD_DIM):
        out_tilde[d] = Float32(0)

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)
    var scores = InlineArray[Float32, POS_GROUP](fill=Float32(0))
    var exp_scores = InlineArray[Float32, POS_GROUP](fill=Float32(0))
    var w_u8 = InlineArray[UInt8, POS_GROUP](fill=UInt8(0))

    for base in range(0, CONTEXT_LEN, POS_GROUP):
        var group_len = min(POS_GROUP, CONTEXT_LEN - base)

        var group_max = Float32(-1e30)
        for p in range(group_len):
            var pos = base + p
            var s = score_quantized(
                q_i8, q_bias, q_absmax,
                k_i8_all + pos * HEAD_DIM,
                k_absmax_all[pos])
            scores[p] = s
            if s > group_max:
                group_max = s

        var new_max = running_max if running_max > group_max else group_max
        if running_sum > Float32(0):
            var rescale = Float32(exp(running_max - new_max))
            for d in range(HEAD_DIM):
                out_tilde[d] *= rescale
            running_sum *= rescale
        running_max = new_max

        var w_max = Float32(0)
        for p in range(group_len):
            var pos = base + p
            var e = Float32(exp(scores[p] - new_max))
            exp_scores[p] = e
            running_sum += e
            var w_eff = e * v_absmax_all[pos]
            if w_eff > w_max:
                w_max = w_eff
        if w_max < Float32(1e-10):
            continue

        var w_scale = Float32(255.0) / w_max
        var w_dequant = w_max / Float32(255.0)
        for p in range(group_len):
            var q = roundeven(exp_scores[p] * v_absmax_all[base + p] * w_scale)
            if q < Float32(0):
                q = Float32(0)
            if q > Float32(255.0):
                q = Float32(255.0)
            w_u8[p] = q.cast[DType.uint8]()

        for d in range(HEAD_DIM):
            var acc = Int(0)
            for p in range(group_len):
                acc += Int(w_u8[p]) * Int(v_i8_all[(base + p) * HEAD_DIM + d])
            out_tilde[d] += Float32(acc) * w_dequant
    var final_scale = Float32(1.0) / (Float32(127.0) * running_sum)
    for d in range(HEAD_DIM):
        out_tilde[d] *= final_scale


def main():
    print("=== quant_analysis/sliding_attention ===")
    print("head_dim=" + String(HEAD_DIM)
        + " heads_per_group=" + String(HEADS_PER_GROUP)
        + " context_len=" + String(CONTEXT_LEN)
        + " pos_group=" + String(POS_GROUP)
        + " trials=" + String(NUM_TRIALS))

    var q_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](HEADS_PER_GROUP * HEAD_DIM)))
    var k_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](CONTEXT_LEN * HEAD_DIM)))
    var v_bf16 = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](CONTEXT_LEN * HEAD_DIM)))
    var q_norm = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](HEAD_DIM)))
    var k_norm = BF16Ptr(unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](HEAD_DIM)))
    var cos_tbl = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN * HALF)))
    var sin_tbl = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN * HALF)))

    var q_proc = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP * HEAD_DIM)))
    var k_proc = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN * HEAD_DIM)))
    var v_tilde = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN * HEAD_DIM)))
    var ref_scores = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN)))
    var ref_probs = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN)))
    var out_ref = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP * HEAD_DIM)))
    var out_q = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP * HEAD_DIM)))
    var out_tmp = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEAD_DIM)))

    var q_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](HEADS_PER_GROUP * HEAD_DIM)))
    var k_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](CONTEXT_LEN * HEAD_DIM)))
    var v_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](CONTEXT_LEN * HEAD_DIM)))
    var out_i8 = I8Ptr(unsafe_from_address=Int(alloc[Scalar[DType.int8]](HEADS_PER_GROUP * HEAD_DIM)))
    var q_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP)))
    var q_bias = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP)))
    var k_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN)))
    var v_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](CONTEXT_LEN)))
    var out_absmax = F32Ptr(unsafe_from_address=Int(alloc[Float32](HEADS_PER_GROUP)))

    fill_rotary_tables(cos_tbl, sin_tbl)

    var rng = Rng(seed=0x736C6964655F7170)
    var agg = init_aggregate()

    for _ in range(NUM_TRIALS):
        fill_activation_bf16(rng, q_bf16, HEADS_PER_GROUP * HEAD_DIM, X_STDDEV)
        fill_activation_bf16(rng, k_bf16, CONTEXT_LEN * HEAD_DIM, X_STDDEV)
        fill_activation_bf16(rng, v_bf16, CONTEXT_LEN * HEAD_DIM, X_STDDEV)
        fill_gamma_bf16(rng, q_norm, HEAD_DIM, GAMMA_CENTER, GAMMA_STDDEV)
        fill_gamma_bf16(rng, k_norm, HEAD_DIM, GAMMA_CENTER, GAMMA_STDDEV)

        for pos in range(CONTEXT_LEN):
            var cos_row = cos_tbl + pos * HALF
            var sin_row = sin_tbl + pos * HALF
            var res = prep_kv_position(
                k_bf16 + pos * HEAD_DIM,
                v_bf16 + pos * HEAD_DIM,
                k_norm,
                cos_row, sin_row,
                k_proc + pos * HEAD_DIM,
                v_tilde + pos * HEAD_DIM,
                k_i8 + pos * HEAD_DIM,
                v_i8 + pos * HEAD_DIM,
            )
            k_absmax[pos] = res[0]
            v_absmax[pos] = res[1]

        var cur_cos = cos_tbl + (CONTEXT_LEN - 1) * HALF
        var cur_sin = sin_tbl + (CONTEXT_LEN - 1) * HALF

        for qh in range(HEADS_PER_GROUP):
            var q_res = prep_q_head(
                q_bf16 + qh * HEAD_DIM,
                q_norm,
                cur_cos, cur_sin,
                q_proc + qh * HEAD_DIM,
                q_i8 + qh * HEAD_DIM,
            )
            q_bias[qh] = q_res[0]
            q_absmax[qh] = q_res[1]

            for pos in range(CONTEXT_LEN):
                ref_scores[pos] = dot_f32(
                    q_proc + qh * HEAD_DIM,
                    k_proc + pos * HEAD_DIM,
                    HEAD_DIM)
            softmax(ref_scores, ref_probs, CONTEXT_LEN)
            for d in range(HEAD_DIM):
                var acc = Float32(0)
                for pos in range(CONTEXT_LEN):
                    acc += ref_probs[pos] * v_tilde[pos * HEAD_DIM + d]
                out_ref[qh * HEAD_DIM + d] = acc

            quantized_attention_one_head(
                q_i8 + qh * HEAD_DIM,
                q_bias[qh],
                q_absmax[qh],
                k_i8, v_i8,
                k_absmax, v_absmax,
                out_tmp)
            out_absmax[qh] = quantize_absmax_row(
                out_tmp, out_i8 + qh * HEAD_DIM, HEAD_DIM)
            dequant_row(out_i8 + qh * HEAD_DIM, out_absmax[qh], out_q + qh * HEAD_DIM, HEAD_DIM)
        add_report(agg, compute_error(out_ref, out_q, HEADS_PER_GROUP * HEAD_DIM))
    print_summary("sliding_attention", agg)

    q_bf16.free()
    k_bf16.free()
    v_bf16.free()
    q_norm.free()
    k_norm.free()
    cos_tbl.free()
    sin_tbl.free()
    q_proc.free()
    k_proc.free()
    v_tilde.free()
    ref_scores.free()
    ref_probs.free()
    out_ref.free()
    out_q.free()
    out_tmp.free()
    q_i8.free()
    k_i8.free()
    v_i8.free()
    out_i8.free()
    q_absmax.free()
    q_bias.free()
    k_absmax.free()
    v_absmax.free()
    out_absmax.free()
