"""Statistical attention evaluation for Gemma 4's live ButterQuant kernels.

This harness drives the active `experimental3` sliding/full attention path on
model-shaped BF16 Q/K/V states, then compares the result against an exact f32
causal reference on the same normalized states.

For each context length it reports:
  - `pre_quant`: kernel output before the final per-head i8 epilogue
  - `post_quant`: kernel output after the same i8 epilogue used by O projection
  - `bf16_floor`: plain BF16 rounding error on the f32 reference output
  - `v_clip`: fixed-scale V-cache clipping statistics

The goal is to measure divergence in the attention implementation itself,
without conflating it with the later O projection.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.collections import InlineArray
from std.math import abs, exp

from modeling.model_spec import F32, Slot, Replicated, Bound
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental3.kv_cache import Gemma4KVCache
from experimental3.helpers import prep_q_row_normed, prep_q_row_normed_partial
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed,
    write_k_head_normed_partial,
    write_v_head_normed,
)
from experimental3.kernels.sliding_attention import single_pass_attention, score_group, WIDTH
from simd_math import sqrt, roundeven, exp_f32


struct EvalConfig:
    comptime NUM_HEADS = 16
    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime FULL_ROPE_DIMS = 128
    comptime SLIDING_WINDOW = 1024
    comptime MAX_SEQ_LEN = 4096
    comptime EPS = Float32(1e-6)
    comptime TRIALS = 16


comptime C = EvalConfig


@fieldwise_init
struct MetricAggregate(Copyable, ImplicitlyCopyable):
    var samples: Int
    var cosine_sum: Float64
    var cosine_min: Float64
    var mean_abs_sum: Float64
    var max_abs: Float64
    var nrmse_sum: Float64
    var nrmse_max: Float64

    def __init__(out self):
        self.samples = 0
        self.cosine_sum = Float64(0)
        self.cosine_min = Float64(1)
        self.mean_abs_sum = Float64(0)
        self.max_abs = Float64(0)
        self.nrmse_sum = Float64(0)
        self.nrmse_max = Float64(0)


@fieldwise_init
struct ClipAggregate(Copyable, ImplicitlyCopyable):
    var values_total: Int
    var values_clipped: Int
    var rows_total: Int
    var rows_clipped: Int
    var peak_ratio_sum: Float64
    var peak_ratio_max: Float64

    def __init__(out self):
        self.values_total = 0
        self.values_clipped = 0
        self.rows_total = 0
        self.rows_clipped = 0
        self.peak_ratio_sum = Float64(0)
        self.peak_ratio_max = Float64(0)


def rand_uniform01(mut state: UInt64) -> Float32:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float32(Int(state & 0xFFFFFF)) / Float32(0x1000000)


def rand_gaussianish(mut state: UInt64) -> Float32:
    var acc = Float32(0)
    for _ in range(12):
        acc += rand_uniform01(state)
    return acc - Float32(6.0)


def comptime_sqrt(x: Float64) -> Float64:
    if x <= Float64(0):
        return Float64(0)
    var g = x
    for _ in range(60):
        g = (g + x / g) * Float64(0.5)
    return g


def concentration_constant[n: Int, num_samples: Int = 10000]() -> Float64:
    var state = UInt64(0xDEADBEEF12345678)
    var total = Float64(0)
    var sqrt_n = comptime_sqrt(Float64(n))
    var rsqrt_n = Float64(1) / sqrt_n

    for _ in range(num_samples):
        var vec = InlineArray[Float64, n](fill=Float64(0))
        var norm_sq = Float64(0)
        for i in range(n):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var u = Float64(Int(state & 0xFFFFFFFF)) / Float64(0xFFFFFFFF)
            var z = Float64(2) * u - Float64(1)
            vec[i] = z
            norm_sq += z * z

        var inv_norm = Float64(1) / comptime_sqrt(norm_sq)
        for i in range(n):
            vec[i] *= inv_norm

        var stride = 1
        while stride < n:
            var i = 0
            while i < n:
                for j in range(stride):
                    var a = vec[i + j]
                    var b = vec[i + j + stride]
                    vec[i + j] = a + b
                    vec[i + j + stride] = a - b
                i += 2 * stride
            stride *= 2
        for i in range(n):
            vec[i] *= rsqrt_n

        var max_abs = Float64(0)
        for i in range(n):
            var a = vec[i].__abs__()
            if a > max_abs:
                max_abs = a
        total += sqrt_n * max_abs

    return total / Float64(num_samples)


def add_metrics[head_dim: Int](
    mut agg: MetricAggregate,
    got: UnsafePointer[Float32, MutAnyOrigin],
    ref_buf: UnsafePointer[Float32, MutAnyOrigin],
):
    var dot = Float64(0)
    var got_sq = Float64(0)
    var ref_sq = Float64(0)
    var err_sq = Float64(0)
    var abs_sum = Float64(0)
    var max_abs = Float64(0)

    for i in range(head_dim):
        var gv = Float64(got[i])
        var rv = Float64(ref_buf[i])
        var diff = gv - rv
        var ad = abs(diff)
        dot += gv * rv
        got_sq += gv * gv
        ref_sq += rv * rv
        err_sq += diff * diff
        abs_sum += ad
        if ad > max_abs:
            max_abs = ad

    var denom = Float64(sqrt[DType.float64, 1](got_sq * ref_sq))
    var cosine = dot / denom if denom > 1e-30 else Float64(1)
    var nrmse = Float64(sqrt[DType.float64, 1](err_sq / ref_sq)) if ref_sq > 1e-30 else Float64(0)
    var mean_abs = abs_sum / Float64(head_dim)

    agg.samples += 1
    agg.cosine_sum += cosine
    if cosine < agg.cosine_min:
        agg.cosine_min = cosine
    agg.mean_abs_sum += mean_abs
    if max_abs > agg.max_abs:
        agg.max_abs = max_abs
    agg.nrmse_sum += nrmse
    if nrmse > agg.nrmse_max:
        agg.nrmse_max = nrmse


def add_clip_row[head_dim: Int](
    mut agg: ClipAggregate,
    fwht_buf: UnsafePointer[Float32, MutAnyOrigin],
    v_scale: Float32,
):
    var row_clipped = False
    var row_peak = Float64(0)

    for i in range(head_dim):
        var av = abs(Float64(fwht_buf[i]))
        if av > row_peak:
            row_peak = av
        agg.values_total += 1
        if av > Float64(v_scale):
            agg.values_clipped += 1
            row_clipped = True

    agg.rows_total += 1
    if row_clipped:
        agg.rows_clipped += 1

    var ratio = row_peak / Float64(v_scale)
    agg.peak_ratio_sum += ratio
    if ratio > agg.peak_ratio_max:
        agg.peak_ratio_max = ratio


def print_metric(label: String, read agg: MetricAggregate):
    print(
        "    ", label,
        " cos_mean=", agg.cosine_sum / Float64(agg.samples),
        " cos_min=", agg.cosine_min,
        " nrmse_mean=", agg.nrmse_sum / Float64(agg.samples),
        " nrmse_max=", agg.nrmse_max,
        " mean_abs=", agg.mean_abs_sum / Float64(agg.samples),
        " max_abs=", agg.max_abs,
    )


def print_clip(read agg: ClipAggregate):
    print(
        "    v_clip elem_frac=", Float64(agg.values_clipped) / Float64(agg.values_total),
        " row_frac=", Float64(agg.rows_clipped) / Float64(agg.rows_total),
        " peak_ratio_mean=", agg.peak_ratio_sum / Float64(agg.rows_total),
        " peak_ratio_max=", agg.peak_ratio_max,
    )


def rmsnorm_gamma_copy[head_dim: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    var sum_sq = Float32(0)
    for i in range(head_dim):
        var v = Float32(src[i])
        dst[i] = v
        sum_sq += v * v
    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](sum_sq / Float32(head_dim) + eps)
    for i in range(head_dim):
        dst[i] = dst[i] * inv_rms * Float32(gamma[i])


def rmsnorm_no_scale_copy[head_dim: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    var sum_sq = Float32(0)
    for i in range(head_dim):
        var v = Float32(src[i])
        dst[i] = v
        sum_sq += v * v
    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](sum_sq / Float32(head_dim) + eps)
    for i in range(head_dim):
        dst[i] *= inv_rms


def apply_full_rope_f32[head_dim: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime half = head_dim // 2
    for j in range(half):
        var lo = buf[j]
        var hi = buf[half + j]
        var cv = cos_row[j]
        var sv = sin_row[j]
        buf[j] = lo * cv - hi * sv
        buf[half + j] = hi * cv + lo * sv


def apply_partial_rope_f32[head_dim: Int, rope_dims: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime half = head_dim // 2
    comptime rope_half = rope_dims // 2
    for j in range(rope_half):
        var lo = buf[j]
        var hi = buf[half + j]
        var cv = cos_row[j]
        var sv = sin_row[j]
        buf[j] = lo * cv - hi * sv
        buf[half + j] = hi * cv + lo * sv


def reference_attention[head_dim: Int](
    q_std: UnsafePointer[Float32, MutAnyOrigin],
    k_std: UnsafePointer[Float32, MutAnyOrigin],
    v_std: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
    scores: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var max_score = Float32(-1e30)
    for t in range(context_len):
        var dot = Float32(0)
        var k_row = k_std + t * head_dim
        for d in range(head_dim):
            dot += q_std[d] * k_row[d]
        scores[t] = dot
        if dot > max_score:
            max_score = dot

    var exp_sum = Float32(0)
    for t in range(context_len):
        scores[t] = exp(Float64(scores[t] - max_score)).cast[DType.float32]()
        exp_sum += scores[t]

    for d in range(head_dim):
        dst[d] = Float32(0)
    for t in range(context_len):
        var w = scores[t] / exp_sum
        var v_row = v_std + t * head_dim
        for d in range(head_dim):
            dst[d] += w * v_row[d]


@always_inline
def v_agg_group_dynamic_exact[head_dim: Int](
    exp_scores: SIMD[DType.float32, WIDTH],
    v_scales_pg: UnsafePointer[Float32, MutAnyOrigin],
    v_q_pg: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    v_acc: UnsafePointer[Float32, MutAnyOrigin],
):
    for p in range(WIDTH):
        var scaled_w = exp_scores[p] * v_scales_pg[p]
        if scaled_w < Float32(1e-10):
            continue
        var v_row = v_q_pg + p * head_dim
        for d in range(head_dim):
            v_acc[d] += scaled_w * Float32(Int(v_row[d]))


@always_inline
def v_agg_group_dynamic_folded[head_dim: Int](
    exp_scores: SIMD[DType.float32, WIDTH],
    v_scales_pg: UnsafePointer[Float32, MutAnyOrigin],
    v_q_pg: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    v_acc: UnsafePointer[Float32, MutAnyOrigin],
):
    var modified_w = exp_scores * v_scales_pg.load[width=WIDTH]()
    var w_max = modified_w.reduce_max()
    if w_max < Float32(1e-10):
        return

    var w_scale = Float32(255.0) / w_max
    var w_u8 = roundeven(modified_w * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
    var w_dequant = w_max / Float32(255.0)

    for p in range(WIDTH):
        var scaled_w = Float32(Int(w_u8[p])) * w_dequant
        if scaled_w < Float32(1e-10):
            continue
        var v_row = v_q_pg + p * head_dim
        for d in range(head_dim):
            v_acc[d] += scaled_w * Float32(Int(v_row[d]))


def single_pass_attention_dynamic_v_exact[head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    qi_bias: Float32,
    q_scale: Float32,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    kv_head: Int,
    context_len: Int,
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    v_q: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    var q_factor = q_scale / (Float32(127.0) * Float32(127.0))
    var k_scales = cache.k_scale_ptr(kv_head)
    var num_pos_groups = (context_len + WIDTH - 1) // WIDTH

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)

    var v_acc_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var v_acc = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    var scores_arr = InlineArray[Float32, WIDTH](fill=Float32(0))
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    for pg in range(num_pos_groups):
        var k_pg = cache.k_pg_ptr(kv_head, pg)
        score_group[head_dim](
            q_i8, k_pg, qi_bias, q_factor,
            k_scales + pg * WIDTH, scores)

        var group_start = pg * WIDTH
        for p in range(WIDTH):
            if group_start + p >= context_len:
                scores[p] = Float32(-1e30)

        var group_max = scores.load[width=WIDTH]().reduce_max()
        var new_max = max(running_max, group_max)

        if running_sum > 0:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            for d in range(0, head_dim, WIDTH):
                (v_acc + d).store((v_acc + d).load[width=WIDTH]() * rescale)
            running_sum *= rescale

        running_max = new_max
        var exp_scores = exp_f32[WIDTH](scores.load[width=WIDTH]() - new_max)

        for p in range(WIDTH):
            if group_start + p >= context_len:
                exp_scores[p] = Float32(0)

        running_sum += exp_scores.reduce_add()
        v_agg_group_dynamic_exact[head_dim](
            exp_scores,
            v_scales + group_start,
            v_q + group_start * head_dim,
            v_acc)

    var inv_sum = Float32(1.0) / (Float32(127.0) * running_sum)
    for d in range(0, head_dim, WIDTH):
        (output + d).store((v_acc + d).load[width=WIDTH]() * inv_sum)


def single_pass_attention_dynamic_v_folded[head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    qi_bias: Float32,
    q_scale: Float32,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    kv_head: Int,
    context_len: Int,
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    v_q: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    var q_factor = q_scale / (Float32(127.0) * Float32(127.0))
    var k_scales = cache.k_scale_ptr(kv_head)
    var num_pos_groups = (context_len + WIDTH - 1) // WIDTH

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)

    var v_acc_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var v_acc = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    var scores_arr = InlineArray[Float32, WIDTH](fill=Float32(0))
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    for pg in range(num_pos_groups):
        var k_pg = cache.k_pg_ptr(kv_head, pg)
        score_group[head_dim](
            q_i8, k_pg, qi_bias, q_factor,
            k_scales + pg * WIDTH, scores)

        var group_start = pg * WIDTH
        for p in range(WIDTH):
            if group_start + p >= context_len:
                scores[p] = Float32(-1e30)

        var group_max = scores.load[width=WIDTH]().reduce_max()
        var new_max = max(running_max, group_max)

        if running_sum > 0:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            for d in range(0, head_dim, WIDTH):
                (v_acc + d).store((v_acc + d).load[width=WIDTH]() * rescale)
            running_sum *= rescale

        running_max = new_max
        var exp_scores = exp_f32[WIDTH](scores.load[width=WIDTH]() - new_max)

        for p in range(WIDTH):
            if group_start + p >= context_len:
                exp_scores[p] = Float32(0)

        running_sum += exp_scores.reduce_add()
        v_agg_group_dynamic_folded[head_dim](
            exp_scores,
            v_scales + group_start,
            v_q + group_start * head_dim,
            v_acc)

    var inv_sum = Float32(1.0) / (Float32(127.0) * running_sum)
    for d in range(0, head_dim, WIDTH):
        (output + d).store((v_acc + d).load[width=WIDTH]() * inv_sum)


def copy_and_inverse_fwht[head_dim: Int](
    src_fwht: UnsafePointer[Float32, MutAnyOrigin],
    dst_std: UnsafePointer[Float32, MutAnyOrigin],
):
    for d in range(head_dim):
        dst_std[d] = src_fwht[d]
    fwht_block[head_dim](dst_std)


def quantize_dequant_inverse_fwht[head_dim: Int](
    src_fwht: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    dst_std: UnsafePointer[Float32, MutAnyOrigin],
):
    var scale = absmax_quantize_i8[head_dim](src_fwht, qi_buf)
    var dq = scale / Float32(127)
    for d in range(head_dim):
        dst_std[d] = Float32(Int(qi_buf[d])) * dq
    fwht_block[head_dim](dst_std)


def bf16_roundtrip_copy[head_dim: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    for d in range(head_dim):
        dst[d] = Float32(Scalar[DType.bfloat16](src[d]))


def fill_random_bf16(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
    mut state: UInt64,
):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](rand_gaussianish(state))


def fill_random_gamma(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
    mut state: UInt64,
):
    for i in range(count):
        dst[i] = Scalar[DType.bfloat16](Float32(0.5) + rand_uniform01(state))


def zero_bytes(dst: UnsafePointer[UInt8, MutAnyOrigin], count: Int):
    for i in range(count):
        dst[i] = UInt8(0)


def eval_sliding_case(
    context_len: Int,
    v_scale: Float32,
    cos_tab: UnsafePointer[Float32, MutAnyOrigin],
    sin_tab: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime HD = C.HEAD_DIM_SLIDING
    comptime HALF = HD // 2
    comptime NH = C.NUM_HEADS
    comptime NKV = C.NUM_KV_HEADS_SLIDING
    comptime HPG = NH // NKV
    comptime Cache = Gemma4KVCache[C.SLIDING_WINDOW, HD, NKV, NH]

    var v_quant_inv = Float32(127) / v_scale

    var cache_mem = alloc[UInt8](Cache.TOTAL_BYTES)
    var cache = Cache(Int(cache_mem))
    var q_raw = alloc[Scalar[DType.bfloat16]](NH * HD)
    var k_raw = alloc[Scalar[DType.bfloat16]](NKV * context_len * HD)
    var v_raw = alloc[Scalar[DType.bfloat16]](NKV * context_len * HD)
    var q_norm = alloc[Scalar[DType.bfloat16]](HD)
    var k_norm = alloc[Scalar[DType.bfloat16]](HD)
    var k_ref = alloc[Float32](NKV * context_len * HD)
    var v_ref = alloc[Float32](NKV * context_len * HD)
    var v_dyn_q = alloc[Scalar[DType.int8]](NKV * context_len * HD)
    var v_dyn_scales = alloc[Float32](NKV * context_len)
    var q_ref = alloc[Float32](HD)
    var scores = alloc[Float32](context_len)
    var work = alloc[Float32](HD)
    var kernel_std = alloc[Float32](HD)
    var kernel_post_std = alloc[Float32](HD)
    var dyn_exact_std = alloc[Float32](HD)
    var dyn_fold_std = alloc[Float32](HD)
    var ref_std = alloc[Float32](HD)
    var ref_bf16 = alloc[Float32](HD)
    var clip_tmp = alloc[Float32](HD)
    var q_i8 = alloc[Scalar[DType.int8]](HD)
    var qi_buf = alloc[Scalar[DType.int8]](HD)
    var metrics_pre = MetricAggregate()
    var metrics_post = MetricAggregate()
    var metrics_dyn_exact = MetricAggregate()
    var metrics_dyn_fold = MetricAggregate()
    var metrics_bf16 = MetricAggregate()
    var clip_stats = ClipAggregate()

    for trial in range(C.TRIALS):
        var rng = UInt64(0x1020304050607080 + trial * 131 + context_len * 17)
        fill_random_gamma(q_norm, HD, rng)
        fill_random_gamma(k_norm, HD, rng)
        fill_random_bf16(q_raw, NH * HD, rng)
        fill_random_bf16(k_raw, NKV * context_len * HD, rng)
        fill_random_bf16(v_raw, NKV * context_len * HD, rng)
        zero_bytes(cache_mem, Cache.TOTAL_BYTES)

        for g in range(NKV):
            for t in range(context_len):
                var raw_k = k_raw + (g * context_len + t) * HD
                var raw_v = v_raw + (g * context_len + t) * HD
                var cos_row = cos_tab + t * HALF
                var sin_row = sin_tab + t * HALF

                write_k_head_normed[HD](
                    raw_k, k_norm,
                    cos_row, sin_row,
                    work, qi_buf,
                    cache, t, g, C.EPS)
                write_v_head_normed[HD](
                    raw_v, work, qi_buf,
                    cache, t, g, v_quant_inv, C.EPS)

                var k_dst = k_ref + (g * context_len + t) * HD
                var v_dst = v_ref + (g * context_len + t) * HD
                rmsnorm_gamma_copy[HD](raw_k, k_norm, k_dst, C.EPS)
                apply_full_rope_f32[HD](k_dst, cos_row, sin_row)
                rmsnorm_no_scale_copy[HD](raw_v, v_dst, C.EPS)

                for d in range(HD):
                    clip_tmp[d] = v_dst[d]
                fwht_block[HD](clip_tmp)
                add_clip_row[HD](clip_stats, clip_tmp, v_scale)
                v_dyn_scales[g * context_len + t] = absmax_quantize_i8[HD](
                    clip_tmp,
                    v_dyn_q + (g * context_len + t) * HD)

        var pos = context_len - 1
        var q_cos = cos_tab + pos * HALF
        var q_sin = sin_tab + pos * HALF

        for h in range(NH):
            var g = h // HPG
            var raw_q = q_raw + h * HD

            rmsnorm_gamma_copy[HD](raw_q, q_norm, q_ref, C.EPS)
            apply_full_rope_f32[HD](q_ref, q_cos, q_sin)

            var q_result = prep_q_row_normed[HD](
                raw_q.bitcast[BFloat16](),
                q_norm,
                q_cos, q_sin,
                q_i8.bitcast[Int8](),
                C.EPS)

            single_pass_attention[HD, C.SLIDING_WINDOW, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len, v_scale,
                work)

            reference_attention[HD](
                q_ref,
                k_ref + g * context_len * HD,
                v_ref + g * context_len * HD,
                context_len,
                scores,
                ref_std)

            copy_and_inverse_fwht[HD](work, kernel_std)
            quantize_dequant_inverse_fwht[HD](work, qi_buf, kernel_post_std)
            single_pass_attention_dynamic_v_exact[HD, C.SLIDING_WINDOW, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len,
                v_dyn_scales + g * context_len,
                v_dyn_q + g * context_len * HD,
                work)
            copy_and_inverse_fwht[HD](work, dyn_exact_std)
            single_pass_attention_dynamic_v_folded[HD, C.SLIDING_WINDOW, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len,
                v_dyn_scales + g * context_len,
                v_dyn_q + g * context_len * HD,
                work)
            copy_and_inverse_fwht[HD](work, dyn_fold_std)
            bf16_roundtrip_copy[HD](ref_std, ref_bf16)

            add_metrics[HD](metrics_pre, kernel_std, ref_std)
            add_metrics[HD](metrics_post, kernel_post_std, ref_std)
            add_metrics[HD](metrics_dyn_exact, dyn_exact_std, ref_std)
            add_metrics[HD](metrics_dyn_fold, dyn_fold_std, ref_std)
            add_metrics[HD](metrics_bf16, ref_bf16, ref_std)

    print(
        "  ctx=", context_len,
        " scale=", v_scale,
        " samples=", metrics_pre.samples,
    )
    print_metric("pre_quant ", metrics_pre)
    print_metric("post_quant", metrics_post)
    print_metric("dyn_exact", metrics_dyn_exact)
    print_metric("dyn_fold ", metrics_dyn_fold)
    print_metric("bf16_floor", metrics_bf16)
    print_clip(clip_stats)

    cache_mem.free()
    q_raw.free()
    k_raw.free()
    v_raw.free()
    q_norm.free()
    k_norm.free()
    k_ref.free()
    v_ref.free()
    v_dyn_q.free()
    v_dyn_scales.free()
    q_ref.free()
    scores.free()
    work.free()
    kernel_std.free()
    kernel_post_std.free()
    dyn_exact_std.free()
    dyn_fold_std.free()
    ref_std.free()
    ref_bf16.free()
    clip_tmp.free()
    q_i8.free()
    qi_buf.free()


def eval_full_case(
    context_len: Int,
    v_scale: Float32,
    cos_tab: UnsafePointer[Float32, MutAnyOrigin],
    sin_tab: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime HD = C.HEAD_DIM_FULL
    comptime HALF = C.FULL_ROPE_DIMS // 2
    comptime NH = C.NUM_HEADS
    comptime NKV = C.NUM_KV_HEADS_FULL
    comptime HPG = NH // NKV
    comptime Cache = Gemma4KVCache[C.MAX_SEQ_LEN, HD, NKV, NH]

    var v_quant_inv = Float32(127) / v_scale

    var cache_mem = alloc[UInt8](Cache.TOTAL_BYTES)
    var cache = Cache(Int(cache_mem))
    var q_raw = alloc[Scalar[DType.bfloat16]](NH * HD)
    var kv_raw = alloc[Scalar[DType.bfloat16]](NKV * context_len * HD)
    var q_norm = alloc[Scalar[DType.bfloat16]](HD)
    var k_norm = alloc[Scalar[DType.bfloat16]](HD)
    var k_ref = alloc[Float32](NKV * context_len * HD)
    var v_ref = alloc[Float32](NKV * context_len * HD)
    var v_dyn_q = alloc[Scalar[DType.int8]](NKV * context_len * HD)
    var v_dyn_scales = alloc[Float32](NKV * context_len)
    var q_ref = alloc[Float32](HD)
    var scores = alloc[Float32](context_len)
    var work = alloc[Float32](HD)
    var kernel_std = alloc[Float32](HD)
    var kernel_post_std = alloc[Float32](HD)
    var dyn_exact_std = alloc[Float32](HD)
    var dyn_fold_std = alloc[Float32](HD)
    var ref_std = alloc[Float32](HD)
    var ref_bf16 = alloc[Float32](HD)
    var clip_tmp = alloc[Float32](HD)
    var q_i8 = alloc[Scalar[DType.int8]](HD)
    var qi_buf = alloc[Scalar[DType.int8]](HD)
    var metrics_pre = MetricAggregate()
    var metrics_post = MetricAggregate()
    var metrics_dyn_exact = MetricAggregate()
    var metrics_dyn_fold = MetricAggregate()
    var metrics_bf16 = MetricAggregate()
    var clip_stats = ClipAggregate()

    for trial in range(C.TRIALS):
        var rng = UInt64(0x8877665544332211 + trial * 257 + context_len * 29)
        fill_random_gamma(q_norm, HD, rng)
        fill_random_gamma(k_norm, HD, rng)
        fill_random_bf16(q_raw, NH * HD, rng)
        fill_random_bf16(kv_raw, NKV * context_len * HD, rng)
        zero_bytes(cache_mem, Cache.TOTAL_BYTES)

        for g in range(NKV):
            for t in range(context_len):
                var raw_kv = kv_raw + (g * context_len + t) * HD
                var cos_row = cos_tab + t * HALF
                var sin_row = sin_tab + t * HALF

                write_k_head_normed_partial[HD, C.FULL_ROPE_DIMS](
                    raw_kv, k_norm,
                    cos_row, sin_row,
                    work, qi_buf,
                    cache, t, g, C.EPS)
                write_v_head_normed[HD](
                    raw_kv, work, qi_buf,
                    cache, t, g, v_quant_inv, C.EPS)

                var k_dst = k_ref + (g * context_len + t) * HD
                var v_dst = v_ref + (g * context_len + t) * HD
                rmsnorm_gamma_copy[HD](raw_kv, k_norm, k_dst, C.EPS)
                apply_partial_rope_f32[HD, C.FULL_ROPE_DIMS](k_dst, cos_row, sin_row)
                rmsnorm_no_scale_copy[HD](raw_kv, v_dst, C.EPS)

                for d in range(HD):
                    clip_tmp[d] = v_dst[d]
                fwht_block[HD](clip_tmp)
                add_clip_row[HD](clip_stats, clip_tmp, v_scale)
                v_dyn_scales[g * context_len + t] = absmax_quantize_i8[HD](
                    clip_tmp,
                    v_dyn_q + (g * context_len + t) * HD)

        var pos = context_len - 1
        var q_cos = cos_tab + pos * HALF
        var q_sin = sin_tab + pos * HALF

        for h in range(NH):
            var g = h // HPG
            var raw_q = q_raw + h * HD

            rmsnorm_gamma_copy[HD](raw_q, q_norm, q_ref, C.EPS)
            apply_partial_rope_f32[HD, C.FULL_ROPE_DIMS](q_ref, q_cos, q_sin)

            var q_result = prep_q_row_normed_partial[HD, C.FULL_ROPE_DIMS](
                raw_q.bitcast[BFloat16](),
                q_norm,
                q_cos, q_sin,
                q_i8.bitcast[Int8](),
                C.EPS)

            single_pass_attention[HD, C.MAX_SEQ_LEN, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len, v_scale,
                work)

            reference_attention[HD](
                q_ref,
                k_ref + g * context_len * HD,
                v_ref + g * context_len * HD,
                context_len,
                scores,
                ref_std)

            copy_and_inverse_fwht[HD](work, kernel_std)
            quantize_dequant_inverse_fwht[HD](work, qi_buf, kernel_post_std)
            single_pass_attention_dynamic_v_exact[HD, C.MAX_SEQ_LEN, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len,
                v_dyn_scales + g * context_len,
                v_dyn_q + g * context_len * HD,
                work)
            copy_and_inverse_fwht[HD](work, dyn_exact_std)
            single_pass_attention_dynamic_v_folded[HD, C.MAX_SEQ_LEN, NKV, NH](
                q_i8, q_result[0], q_result[1],
                cache, g, context_len,
                v_dyn_scales + g * context_len,
                v_dyn_q + g * context_len * HD,
                work)
            copy_and_inverse_fwht[HD](work, dyn_fold_std)
            bf16_roundtrip_copy[HD](ref_std, ref_bf16)

            add_metrics[HD](metrics_pre, kernel_std, ref_std)
            add_metrics[HD](metrics_post, kernel_post_std, ref_std)
            add_metrics[HD](metrics_dyn_exact, dyn_exact_std, ref_std)
            add_metrics[HD](metrics_dyn_fold, dyn_fold_std, ref_std)
            add_metrics[HD](metrics_bf16, ref_bf16, ref_std)

    print(
        "  ctx=", context_len,
        " scale=", v_scale,
        " samples=", metrics_pre.samples,
    )
    print_metric("pre_quant ", metrics_pre)
    print_metric("post_quant", metrics_post)
    print_metric("dyn_exact", metrics_dyn_exact)
    print_metric("dyn_fold ", metrics_dyn_fold)
    print_metric("bf16_floor", metrics_bf16)
    print_clip(clip_stats)

    cache_mem.free()
    q_raw.free()
    kv_raw.free()
    q_norm.free()
    k_norm.free()
    k_ref.free()
    v_ref.free()
    v_dyn_q.free()
    v_dyn_scales.free()
    q_ref.free()
    scores.free()
    work.free()
    kernel_std.free()
    kernel_post_std.free()
    dyn_exact_std.free()
    dyn_fold_std.free()
    ref_std.free()
    ref_bf16.free()
    clip_tmp.free()
    q_i8.free()
    qi_buf.free()


def eval_sliding_suite():
    comptime HALF = C.HEAD_DIM_SLIDING // 2
    comptime CosSlot = Slot[F32, Replicated, C.SLIDING_WINDOW, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, C.SLIDING_WINDOW, HALF, 1]

    var cos_tab = alloc[Float32](C.SLIDING_WINDOW * HALF)
    var sin_tab = alloc[Float32](C.SLIDING_WINDOW * HALF)
    init_sliding_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)))
    var v_scale = Float32(concentration_constant[C.HEAD_DIM_SLIDING]())

    print("=== Sliding Attention Eval ===")
    var contexts = InlineArray[Int, 5](fill=0)
    contexts[0] = 1
    contexts[1] = 16
    contexts[2] = 64
    contexts[3] = 256
    contexts[4] = C.SLIDING_WINDOW
    for i in range(5):
        eval_sliding_case(contexts[i], v_scale, cos_tab, sin_tab)

    cos_tab.free()
    sin_tab.free()


def eval_full_suite():
    comptime HALF = C.FULL_ROPE_DIMS // 2
    comptime CosSlot = Slot[F32, Replicated, C.MAX_SEQ_LEN, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, C.MAX_SEQ_LEN, HALF, 1]

    var cos_tab = alloc[Float32](C.MAX_SEQ_LEN * HALF)
    var sin_tab = alloc[Float32](C.MAX_SEQ_LEN * HALF)
    init_full_rope_tables(Bound[CosSlot](Int(cos_tab)), Bound[SinSlot](Int(sin_tab)))
    var v_scale = Float32(concentration_constant[C.HEAD_DIM_FULL]())

    print()
    print("=== Full Attention Eval ===")
    var contexts = InlineArray[Int, 6](fill=0)
    contexts[0] = 1
    contexts[1] = 16
    contexts[2] = 64
    contexts[3] = 256
    contexts[4] = 1024
    contexts[5] = C.MAX_SEQ_LEN
    for i in range(6):
        eval_full_case(contexts[i], v_scale, cos_tab, sin_tab)

    cos_tab.free()
    sin_tab.free()


def main():
    print("Gemma 4 attention evaluator")
    print("trials/context=", C.TRIALS, " heads=", C.NUM_HEADS)
    print("measures kernel vs exact f32 reference on shared BF16 Q/K/V samples")
    print()

    eval_sliding_suite()
    eval_full_suite()
