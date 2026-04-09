"""Sliding window decode attention for Gemma 4.

Single-pass kernel: K scoring + online softmax + VNNI V-agg in one streaming
iteration over position groups. Each position group is WIDTH positions.

K scoring: vpdpbusd(acc, K_u8_packed, Q_i8_broadcast) — K is unsigned first arg.
V-agg:    vpdpbusd(v_acc, W_u8_broadcast, V_i8_packed) — attention weights quantized to u8.

One worker per KV group. Each worker:
1. Writes K/V for new position to cache
2. Preps Q heads (per-head norm + RoPE + FWHT + quantize)
3. Single pass: score → online softmax → V-agg per position group
4. FWHT on output for O projection quantization
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from experimental.amx import VNNI_BLK
from experimental2.kernels.int8_gemv import vpdpbusd
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed
from simd_math import exp_f32, sqrt, roundeven


comptime WIDTH = CACHE_WIDTH


# ============================================================================
# K scoring dot — K_u8 (packed) × Q_i8 (broadcast)
# ============================================================================


@always_inline
def dot_score[width: Int](
    acc: SIMD[DType.int32, width],
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed_u8: UnsafePointer[UInt8, MutAnyOrigin],
    q_offset: Int,
) -> SIMD[DType.int32, width]:
    """Score dot: vpdpbusd(acc, K_u8[width×4], Q_i8_broadcast[width×4])."""
    var k = k_packed_u8.load[width = width * 4]()
    var q4 = (q_i8 + q_offset).load[width=4]()
    var q8 = q4.join(q4)
    var q16 = q8.join(q8)
    var q32 = q16.join(q16)
    var q64 = q32.join(q32)
    return vpdpbusd[width](acc, k, q64.slice[width * 4]())


# ============================================================================
# Score one position group (WIDTH positions) against Q
# ============================================================================


@always_inline
def score_group[head_dim: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pg: UnsafePointer[UInt8, MutAnyOrigin],
    qi_bias: Float32,
    q_factor: Float32,
    k_scales: UnsafePointer[Float32, MutAnyOrigin],
    scores_out: UnsafePointer[Float32, MutAnyOrigin],
):
    """Score WIDTH positions, dequant to f32, write to scores_out."""
    comptime K_DIM_GROUPS = head_dim // VNNI_BLK

    var acc = SIMD[DType.int32, WIDTH](0)
    for kdg in range(K_DIM_GROUPS):
        acc = dot_score[WIDTH](
            acc, q_i8,
            k_pg + kdg * WIDTH * VNNI_BLK,
            kdg * VNNI_BLK)
    var corrected = acc.cast[DType.float32]() - qi_bias
    var k_sc = k_scales.load[width=WIDTH]()
    scores_out.store(corrected * q_factor * k_sc)


# ============================================================================
# V-agg one position group via VNNI — W_u8 (broadcast) × V_i8 (packed)
# ============================================================================


@always_inline
def v_agg_group[head_dim: Int](
    exp_scores: SIMD[DType.float32, WIDTH],
    v_pg: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    v_acc: UnsafePointer[Float32, MutAnyOrigin],
):
    """Accumulate V-agg for WIDTH positions using VNNI.

    Quantizes exp_scores to u8, broadcasts per sub_quad, then vpdpbusd
    with packed V data. Accumulates into f32 v_acc[head_dim].
    """
    comptime V_CHANNEL_GROUPS = head_dim // WIDTH
    comptime V_SUB_QUADS = WIDTH // VNNI_BLK
    comptime V_CG_BYTES = V_SUB_QUADS * WIDTH * VNNI_BLK

    # Quantize WIDTH attention weights to u8
    var w_max = exp_scores.reduce_max()
    if w_max < Float32(1e-10):
        return
    var w_scale = 255.0 / w_max
    var w_u8_wide = roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
    var w_dequant = w_max / 255.0

    # Split into sub_quads of VNNI_BLK and broadcast each
    for cg in range(V_CHANNEL_GROUPS):
        var cg_base = v_pg + cg * V_CG_BYTES
        var cg_acc = (v_acc + cg * WIDTH).load[width=WIDTH]()

        comptime for sq in range(V_SUB_QUADS):
            # Extract 4 u8 weights for this sub_quad
            var w4 = w_u8_wide.slice[VNNI_BLK, offset = sq * VNNI_BLK]()
            # Broadcast to WIDTH × VNNI_BLK bytes
            var w8 = w4.join(w4)
            var w16 = w8.join(w8)
            var w32 = w16.join(w16)
            var w64 = w32.join(w32)
            var w_broadcast = w64.slice[WIDTH * VNNI_BLK]()

            var v_data = (cg_base + sq * WIDTH * VNNI_BLK).load[width = WIDTH * VNNI_BLK]()
            var i32_result = vpdpbusd[WIDTH](SIMD[DType.int32, WIDTH](0), w_broadcast, v_data)
            cg_acc += i32_result.cast[DType.float32]() * w_dequant

        (v_acc + cg * WIDTH).store(cg_acc)


# ============================================================================
# Single-pass decode: score + online softmax + V-agg
# ============================================================================


def single_pass_attention[head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    qi_bias: Float32,
    q_scale: Float32,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    kv_head: Int,
    context_len: Int,
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Score Q against K cache, online softmax, V-agg. Single pass over positions.

    Output: f32[head_dim] (in Hadamard domain, before final FWHT).
    """
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]

    var q_factor = q_scale / (127.0 * 127.0)
    var k_scales = cache.k_scale_ptr(kv_head)
    var num_pos_groups = (context_len + WIDTH - 1) // WIDTH

    # Online softmax state
    var running_max = Float32(-1e30)
    var running_sum = Float32(0)

    # V accumulator: f32[head_dim], tracks unnormalized weighted sum
    var v_acc_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var v_acc = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    # Score buffer for one position group
    var scores_arr = InlineArray[Float32, WIDTH](fill=Float32(0))
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    for pg in range(num_pos_groups):
        var k_pg = cache.k_pg_ptr(kv_head, pg)
        var v_pg = cache.v_pg_ptr(kv_head, pg)

        # Score WIDTH positions
        score_group[head_dim](
            q_i8, k_pg, qi_bias, q_factor,
            k_scales + pg * WIDTH, scores)

        # Mask invalid positions in last group
        var group_start = pg * WIDTH
        for p in range(WIDTH):
            if group_start + p >= context_len:
                scores[p] = Float32(-1e30)

        # Online softmax: update max, rescale, compute exp
        var group_max = scores.load[width=WIDTH]().reduce_max()
        var new_max = max(running_max, group_max)

        if running_sum > 0:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            for d in range(0, head_dim, WIDTH):
                (v_acc + d).store((v_acc + d).load[width=WIDTH]() * rescale)
            running_sum *= rescale

        running_max = new_max
        var exp_scores = exp_f32[WIDTH](scores.load[width=WIDTH]() - new_max)

        # Mask exp_scores for invalid positions
        for p in range(WIDTH):
            if group_start + p >= context_len:
                exp_scores[p] = Float32(0)

        running_sum += exp_scores.reduce_add()

        # V-agg via VNNI
        v_agg_group[head_dim](exp_scores, v_pg, v_acc)

    # Finalize: divide by sum, apply V scale (baked into dequant elsewhere)
    var inv_sum = 1.0 / running_sum
    for d in range(0, head_dim, WIDTH):
        (output + d).store((v_acc + d).load[width=WIDTH]() * inv_sum)


# ============================================================================
# Per-KV-group worker
# ============================================================================


@fieldwise_init
struct SlidingAttnGroupArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_ptr: Int
    var k_bf16_ptr: Int
    var v_bf16_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var kv_head: Int
    var cache_pos: Int
    var context_len: Int
    var v_scale: Float32
    var qi_out_ptr: Int        # i8[heads_per_group × head_dim] output
    var head_scale_ptr: Int    # f32[heads_per_group] per-head absmax scales
    var eps: Float32


def sliding_attn_group_kernel[
    head_dim: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: SlidingAttnGroupArgs):
    """One KV group: write K/V, prep Q, single-pass attention, FWHT, quantize to i8.

    Output: i8[heads_per_group × head_dim] + f32[heads_per_group] per-head scales.
    Ready for int8_gemv_blocked O projection with block=head_dim.
    """
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](args.cache_base)
    var cos = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.cos_ptr)
    var sin = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.sin_ptr)
    var qi_out = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_out_ptr)
    var head_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.head_scale_ptr)

    # Stack buffers for K/V cache write
    var work_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    # 1. Write K to cache
    write_k_head_normed[head_dim](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_bf16_ptr),
        cos, sin, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    # 2. Write V to cache
    write_v_head_normed[head_dim](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.v_bf16_ptr),
        Float32(127.0) / args.v_scale,
        work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    # 3. Process each Q head: score → V-agg → FWHT → quantize to i8
    for qh in range(heads_per_group):
        var q_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.q_bf16_ptr + qh * head_dim * 2)

        # Prep Q: /rms → RoPE → FWHT → quantize
        var q_i8_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
        var q_i8 = UnsafePointer(to=q_i8_arr).bitcast[Scalar[DType.int8]]()
        var result = prep_q_row_normed[head_dim](
            q_bf16.bitcast[BFloat16](), cos, sin,
            q_i8.bitcast[Int8](), args.eps)
        var qi_bias = result[0]
        var q_scale = result[1]

        # Single-pass: score + softmax + V-agg → f32[head_dim] on stack
        single_pass_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8, qi_bias, q_scale,
            cache, args.kv_head, args.context_len,
            work)

        # FWHT → quantize → i8 output for O projection
        fwht_block[head_dim](work)
        head_scales[qh] = absmax_quantize_i8[head_dim](work, qi_out + qh * head_dim)


# ============================================================================
# Validation
# ============================================================================


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def cosine_sim_f64(a: UnsafePointer[Float64, MutAnyOrigin], b: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na < 1e-30 or nb < 1e-30:
        return Float64(0)
    return dot / (Float64(sqrt[DType.float64, 1](na)) * Float64(sqrt[DType.float64, 1](nb)))


def validate_single_pass():
    """Validate single-pass attention: scoring accuracy + V-agg output quality."""
    comptime head_dim = 256
    comptime max_seq = 1024
    comptime num_kv_heads = 2
    comptime num_q_heads = 4
    comptime half = head_dim // 2
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]
    comptime num_positions = 64
    comptime kv_head = 0
    var rng = UInt64(0xDEADCAFE12345678)

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    # RoPE tables
    var cos_f32 = alloc[Float32](half)
    var sin_f32 = alloc[Float32](half)
    for i in range(half):
        var angle = xorshift64(rng) * 0.1
        cos_f32[i] = Float32(1.0 - angle * angle * 0.5)
        sin_f32[i] = Float32(angle)

    # Write K and V positions to cache
    var work = alloc[Float32](head_dim)
    var qi_buf = alloc[Scalar[DType.int8]](head_dim)
    var v_scale = Float32(5.0)

    for pos in range(num_positions):
        var k_data = alloc[Scalar[DType.bfloat16]](head_dim)
        var v_data = alloc[Scalar[DType.bfloat16]](head_dim)
        for d in range(head_dim):
            k_data[d] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
            v_data[d] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        write_k_head_normed[head_dim](k_data, cos_f32, sin_f32, work, qi_buf,
            cache, pos, kv_head, Float32(1e-6))
        write_v_head_normed[head_dim](v_data, Float32(127.0) / v_scale, work, qi_buf,
            cache, pos, kv_head, Float32(1e-6))
        k_data.free()
        v_data.free()

    # Prep Q
    var q_bf16 = alloc[Scalar[DType.bfloat16]](head_dim)
    for d in range(head_dim):
        q_bf16[d] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

    var q_i8 = alloc[Scalar[DType.int8]](head_dim)
    var qr = prep_q_row_normed[head_dim](
        q_bf16.bitcast[BFloat16](), cos_f32, sin_f32,
        q_i8.bitcast[Int8](), Float32(1e-6))
    var qi_bias = qr[0]
    var q_scale = qr[1]

    # --- Two-pass reference: score all, softmax, then V-agg ---
    var q_factor = q_scale / (127.0 * 127.0)
    var k_scales = cache.k_scale_ptr(kv_head)
    var ref_scores = alloc[Float32](num_positions)

    var num_pg = (num_positions + WIDTH - 1) // WIDTH
    for pg in range(num_pg):
        var scores_buf = alloc[Float32](WIDTH)
        score_group[head_dim](
            q_i8, cache.k_pg_ptr(kv_head, pg),
            qi_bias, q_factor, k_scales + pg * WIDTH, scores_buf)
        for p in range(WIDTH):
            if pg * WIDTH + p < num_positions:
                ref_scores[pg * WIDTH + p] = scores_buf[p]
        scores_buf.free()

    # Softmax reference
    var max_s = Float32(-1e30)
    for i in range(num_positions):
        if ref_scores[i] > max_s:
            max_s = ref_scores[i]
    var sum_e = Float32(0)
    for i in range(num_positions):
        ref_scores[i] = Float32(exp_f32[1](ref_scores[i] - max_s))
        sum_e += ref_scores[i]
    for i in range(num_positions):
        ref_scores[i] /= sum_e

    # --- Single-pass kernel ---
    var kernel_out = alloc[Float32](head_dim)
    single_pass_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
        q_i8, qi_bias, q_scale,
        cache, kv_head, num_positions, kernel_out)

    # --- f32 reference V-agg (read V from cache, multiply by ref softmax weights) ---
    var ref_out = alloc[Float64](head_dim)
    for d in range(head_dim):
        ref_out[d] = Float64(0)

    for pos in range(num_positions):
        var w = Float64(ref_scores[pos])
        var pos_group = pos // WIDTH
        var sub_quad = (pos % WIDTH) // VNNI_BLK
        var vnni_slot = pos % VNNI_BLK
        var v_pg = cache.v_pg_ptr(kv_head, pos_group)
        for cg in range(Cache.V_CHANNEL_GROUPS):
            var cg_base = v_pg + cg * Cache.V_CG_BYTES + sub_quad * WIDTH * VNNI_BLK
            for ci in range(WIDTH):
                var v_val = Float64(Int(cg_base[ci * VNNI_BLK + vnni_slot]))
                ref_out[cg * WIDTH + ci] += w * v_val

    # Compare kernel output (pre-FWHT) vs f64 reference V-agg
    var kernel_f64 = alloc[Float64](head_dim)
    for i in range(head_dim):
        kernel_f64[i] = Float64(kernel_out[i])
    var cos_sim = cosine_sim_f64(ref_out.bitcast[Float64](), kernel_f64.bitcast[Float64](), head_dim)

    var max_err = Float64(0)
    var sum_sq_err = Float64(0)
    var sum_sq_ref = Float64(0)
    for i in range(head_dim):
        var err = (kernel_f64[i] - ref_out[i]).__abs__()
        if err > max_err:
            max_err = err
        sum_sq_err += (kernel_f64[i] - ref_out[i]) * (kernel_f64[i] - ref_out[i])
        sum_sq_ref += ref_out[i] * ref_out[i]
    var nrmse = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_ref)) if sum_sq_ref > 0 else Float64(0)

    print("  cosine (kernel vs f64 ref V-agg):  " + String(cos_sim))
    print("  NRMSE:                             " + String(nrmse))
    print("  max abs error:                     " + String(max_err))
    print("  q_scale=" + String(q_scale) + " qi_bias=" + String(qi_bias))

    # Verify FWHT preserves energy
    var pre_fwht_rms = Float64(0)
    for d in range(head_dim):
        pre_fwht_rms += Float64(kernel_out[d]) * Float64(kernel_out[d])
    pre_fwht_rms = Float64(sqrt[DType.float64, 1](pre_fwht_rms / Float64(head_dim)))

    fwht_block[head_dim](kernel_out)
    var post_fwht_rms = Float64(0)
    for d in range(head_dim):
        post_fwht_rms += Float64(kernel_out[d]) * Float64(kernel_out[d])
    post_fwht_rms = Float64(sqrt[DType.float64, 1](post_fwht_rms / Float64(head_dim)))
    print("  FWHT energy ratio:                 " + String(post_fwht_rms / pre_fwht_rms))

    cache_buf.free()
    cos_f32.free()
    sin_f32.free()
    work.free()
    qi_buf.free()
    q_bf16.free()
    q_i8.free()
    ref_scores.free()
    kernel_out.free()
    ref_out.free()
    kernel_f64.free()


def main():
    print("=== sliding_attention validation (single-pass, width=" + String(WIDTH) + ") ===")
    print("\nSingle-pass scoring + V-agg (head_dim=256, 64 positions):")
    validate_single_pass()
