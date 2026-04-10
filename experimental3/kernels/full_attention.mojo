"""Full (global) attention decode for Gemma 4 — chunked with flash-combine.

Context grows to 256K. Too large for single-worker or L2-resident. Chunked
across pool workers, each chunk does single-pass score + local softmax + V-agg.
Cross-chunk reduction uses the log-sum-exp correction trick (FlashAttention).

Per chunk worker: processes all Q heads sharing one KV head over a position range.
K data stays hot across Q heads within the same chunk.

head_dim=512, 2 KV heads, 16 Q heads (GQA 8:1), partial RoPE (128/512 dims).
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from experimental.amx import VNNI_BLK
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.sliding_attention import (
    dot_score, score_group, v_agg_group, single_pass_attention, WIDTH,
)
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed_partial, write_v_head_with_inv_rms,
)
from experimental3.helpers import prep_q_row_normed_partial
from simd_math import exp_f32, sqrt


# ============================================================================
# Chunk partial result — per Q head per chunk
# ============================================================================


@fieldwise_init
struct ChunkPartial[head_dim: Int](Copyable, ImplicitlyCopyable, Movable):
    """Partial attention result from one chunk: v_acc, max, sum_exp."""
    var v_ptr: Int       # f32[head_dim] unnormalized weighted V sum
    var chunk_max: Float32
    var chunk_sum: Float32


# ============================================================================
# Chunk kernel: score + local softmax + V-agg for a position range
# ============================================================================


def chunk_attention[head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    qi_bias: Float32,
    q_scale: Float32,
    cache: Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    kv_head: Int,
    pg_start: Int,
    pg_end: Int,
    context_len: Int,
    v_out: UnsafePointer[Float32, MutAnyOrigin],
    max_out: UnsafePointer[Float32, MutAnyOrigin],
    sum_out: UnsafePointer[Float32, MutAnyOrigin],
):
    """Score + local softmax + V-agg for position groups [pg_start, pg_end).

    Writes unnormalized v_acc to v_out, chunk_max to max_out, chunk_sum to sum_out.
    Uses online softmax within the chunk, same as the sliding kernel.
    """
    var q_factor = q_scale / (127.0 * 127.0)
    var k_scales = cache.k_scale_ptr(kv_head)

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)

    var v_acc_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var v_acc = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    var scores_arr = InlineArray[Float32, WIDTH](fill=Float32(0))
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    for pg in range(pg_start, pg_end):
        var k_pg = cache.k_pg_ptr(kv_head, pg)
        var v_pg = cache.v_pg_ptr(kv_head, pg)

        score_group[head_dim](
            q_i8, k_pg, qi_bias, q_factor,
            k_scales + pg * WIDTH, scores)

        # Mask invalid positions
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
        v_agg_group[head_dim](exp_scores, v_pg, v_acc)

    # Write partial results (unnormalized)
    for d in range(0, head_dim, WIDTH):
        (v_out + d).store((v_acc + d).load[width=WIDTH]())
    max_out[] = running_max
    sum_out[] = running_sum


# ============================================================================
# Cross-chunk reduction — flash-combine
# ============================================================================


def flash_combine[head_dim: Int](
    partials_v: UnsafePointer[Float32, MutAnyOrigin],
    partials_max: UnsafePointer[Float32, MutAnyOrigin],
    partials_sum: UnsafePointer[Float32, MutAnyOrigin],
    num_chunks: Int,
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Combine chunk partials via log-sum-exp correction.

    partials_v:   f32[num_chunks, head_dim] — unnormalized V accumulators
    partials_max: f32[num_chunks] — per-chunk max scores
    partials_sum: f32[num_chunks] — per-chunk exp sums
    output:       f32[head_dim] — final normalized attention output
    """
    # Global max
    var global_max = Float32(-1e30)
    for c in range(num_chunks):
        if partials_max[c] > global_max:
            global_max = partials_max[c]

    # Combine with correction
    var total_sum = Float32(0)
    for d in range(0, head_dim, WIDTH):
        (output + d).store(SIMD[DType.float32, WIDTH](0))

    for c in range(num_chunks):
        var correction = Float32(exp_f32[1](partials_max[c] - global_max))
        var corrected_sum = partials_sum[c] * correction
        total_sum += corrected_sum
        var v_base = partials_v + c * head_dim
        for d in range(0, head_dim, WIDTH):
            var existing = (output + d).load[width=WIDTH]()
            var partial = (v_base + d).load[width=WIDTH]()
            (output + d).store(existing + partial * correction)

    # Normalize
    var inv_sum = 1.0 / total_sum
    for d in range(0, head_dim, WIDTH):
        (output + d).store((output + d).load[width=WIDTH]() * inv_sum)


# ============================================================================
# Chunk worker args + kernel
# ============================================================================


@fieldwise_init
struct FullAttnChunkArgs(Copyable, ImplicitlyCopyable):
    var q_i8_base: Int         # i8[heads_per_group, head_dim] — all Q heads for this KV group
    var qi_bias_base: Int      # f32[heads_per_group]
    var q_scale_base: Int      # f32[heads_per_group]
    var cache_base: Int
    var kv_head: Int
    var pg_start: Int
    var pg_end: Int
    var context_len: Int
    var partials_base: Int     # output: f32[heads_per_group, head_dim + 2] per chunk
    var heads_per_group: Int


def full_attn_chunk_kernel[head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int](
    args: FullAttnChunkArgs,
):
    """Chunk worker: process all Q heads for one position range.

    K data stays hot across Q heads within the chunk.
    Writes partial (v_acc, max, sum) per Q head to partials buffer.
    """
    comptime PARTIAL_STRIDE = head_dim + 2  # f32[head_dim] + max + sum per Q head
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](args.cache_base)
    var partials = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.partials_base)

    for qh in range(args.heads_per_group):
        var q_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=args.q_i8_base + qh * head_dim)
        var qi_bias = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.qi_bias_base)[qh]
        var q_scale = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.q_scale_base)[qh]

        var p_base = partials + qh * PARTIAL_STRIDE
        chunk_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8, qi_bias, q_scale,
            cache, args.kv_head,
            args.pg_start, args.pg_end, args.context_len,
            p_base,              # v_out
            p_base + head_dim,   # max_out
            p_base + head_dim + 1)  # sum_out


# ============================================================================
# Full attention reduction — combine chunks per Q head
# ============================================================================


def full_attn_reduce[head_dim: Int](
    chunk_partials: UnsafePointer[Float32, MutAnyOrigin],
    num_chunks: Int,
    heads_per_group: Int,
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    head_scales: UnsafePointer[Float32, MutAnyOrigin],
):
    """Reduce all chunks for all Q heads → FWHT → i8 + per-head scales.

    chunk_partials: f32[num_chunks, heads_per_group, head_dim + 2]
    qi_out:         i8[heads_per_group, head_dim] output
    head_scales:    f32[heads_per_group] per-head absmax scales
    """
    comptime PARTIAL_STRIDE = head_dim + 2

    var qh_v = alloc[Float32](num_chunks * head_dim)
    var qh_max = alloc[Float32](num_chunks)
    var qh_sum = alloc[Float32](num_chunks)
    var combined = alloc[Float32](head_dim)

    for qh in range(heads_per_group):
        for c in range(num_chunks):
            var src = chunk_partials + c * heads_per_group * PARTIAL_STRIDE + qh * PARTIAL_STRIDE
            for d in range(0, head_dim, WIDTH):
                (qh_v + c * head_dim + d).store((src + d).load[width=WIDTH]())
            qh_max[c] = src[head_dim]
            qh_sum[c] = src[head_dim + 1]

        flash_combine[head_dim](qh_v, qh_max, qh_sum, num_chunks, combined)
        fwht_block[head_dim](combined)
        head_scales[qh] = absmax_quantize_i8[head_dim](combined, qi_out + qh * head_dim)

    qh_v.free()
    qh_max.free()
    qh_sum.free()
    combined.free()


# ============================================================================
# Fused full attention group kernel — KV write + Q prep + score + quantize
# ============================================================================


@fieldwise_init
struct FullAttnGroupArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_ptr: Int        # bf16 Q heads for this group
    var k_bf16_ptr: Int        # bf16 K head (also used for V — K=V shared)
    var q_norm_ptr: Int        # bf16[head_dim] per-head Q norm gamma
    var k_norm_ptr: Int        # bf16[head_dim] per-head K norm gamma
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var kv_head: Int
    var cache_pos: Int
    var context_len: Int
    var qi_out_ptr: Int        # i8[heads_per_group × head_dim] output
    var head_scale_ptr: Int    # f32[heads_per_group] per-head scales
    var v_inv_rms: Float32
    var eps: Float32


def full_attn_group_kernel[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: FullAttnGroupArgs):
    """Full attention: KV write + Q prep + single-pass score + FWHT + quantize.

    K=V shared: both K and V cache are written from the K projection output.
    K gets partial RoPE, V gets scale-free norm only.
    Uses single-pass online softmax (same as sliding) — for decode, context
    fits one pass. Chunked dispatch is for prefill/very long context.
    """
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](args.cache_base)
    var cos = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.cos_ptr)
    var sin = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.sin_ptr)
    var qi_out = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_out_ptr)
    var head_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.head_scale_ptr)

    var work_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    # K=V shared: both cache writes use the same K projection bf16 output
    var k_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_bf16_ptr)

    # Write K with partial RoPE
    write_k_head_normed_partial[head_dim, rope_dims](
        k_bf16,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_norm_ptr),
        cos, sin, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    # Write V from same K output (no RoPE, scale-free norm only)
    write_v_head_with_inv_rms[head_dim](
        k_bf16, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.v_inv_rms)

    # Process each Q head
    for qh in range(heads_per_group):
        var q_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.q_bf16_ptr + qh * head_dim * 2)

        var q_i8_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
        var q_i8 = UnsafePointer(to=q_i8_arr).bitcast[Scalar[DType.int8]]()
        var result = prep_q_row_normed_partial[head_dim, rope_dims](
            q_bf16.bitcast[BFloat16](),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.q_norm_ptr),
            cos, sin,
            q_i8.bitcast[Int8](), args.eps)
        var qi_bias = result[0]
        var q_scale = result[1]

        # Single-pass scoring + V-agg
        single_pass_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8, qi_bias, q_scale,
            cache, args.kv_head, args.context_len,
            work)

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


def validate_chunked_attention():
    """Validate chunked attention with flash-combine against single-pass reference."""
    comptime head_dim = 512
    comptime max_seq = 4096
    comptime num_kv_heads = 2
    comptime num_q_heads = 16
    comptime num_positions = 256
    comptime kv_head = 0
    comptime heads_per_group = num_q_heads // num_kv_heads
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]

    var rng = UInt64(0xCAFEDEAD12345678)

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    # Write K and V positions (using scalar writes for validation)
    for pos in range(num_positions):
        for h in range(num_kv_heads):
            var k_data = alloc[Scalar[DType.int8]](head_dim)
            var v_data = alloc[Scalar[DType.int8]](head_dim)
            for d in range(head_dim):
                k_data[d] = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))
                v_data[d] = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))
            # Write K as u8 (XOR 0x80) directly
            cache.write_k(pos, h, k_data)
            cache.write_v(pos, h, v_data)
            cache.write_k_scale(pos, h, Float32(5.0 + xorshift64(rng) * 2.0))
            k_data.free()
            v_data.free()

    # Create Q heads (pre-quantized i8 for simplicity)
    var q_i8 = alloc[Scalar[DType.int8]](heads_per_group * head_dim)
    var qi_biases = alloc[Float32](heads_per_group)
    var q_scales = alloc[Float32](heads_per_group)
    for qh in range(heads_per_group):
        var q_sum = Int(0)
        for d in range(head_dim):
            var val = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))
            q_i8[qh * head_dim + d] = val
            q_sum += Int(val)
        qi_biases[qh] = Float32(q_sum) * 128.0
        q_scales[qh] = Float32(8.0 + xorshift64(rng) * 2.0)

    # --- Single-pass reference (one chunk covering all positions) ---
    var ref_out = alloc[Float32](heads_per_group * head_dim)
    comptime REF_PARTIAL_STRIDE = head_dim + 2
    var ref_partial = alloc[Float32](heads_per_group * REF_PARTIAL_STRIDE)

    var num_pg = (num_positions + WIDTH - 1) // WIDTH
    for qh in range(heads_per_group):
        chunk_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8 + qh * head_dim, qi_biases[qh], q_scales[qh],
            cache, kv_head, 0, num_pg, num_positions,
            ref_partial + qh * REF_PARTIAL_STRIDE,
            ref_partial + qh * REF_PARTIAL_STRIDE + head_dim,
            ref_partial + qh * REF_PARTIAL_STRIDE + head_dim + 1)

    # Normalize reference
    for qh in range(heads_per_group):
        var p_base = ref_partial + qh * REF_PARTIAL_STRIDE
        var inv_sum = 1.0 / p_base[head_dim + 1]
        for d in range(0, head_dim, WIDTH):
            (ref_out + qh * head_dim + d).store(
                (p_base + d).load[width=WIDTH]() * inv_sum)

    # --- Chunked: split into 4 chunks ---
    comptime num_chunks = 4
    var pgs_per_chunk = (num_pg + num_chunks - 1) // num_chunks
    var chunk_partials = alloc[Float32](num_chunks * heads_per_group * REF_PARTIAL_STRIDE)

    for c in range(num_chunks):
        var pg_start = c * pgs_per_chunk
        var pg_end = min(pg_start + pgs_per_chunk, num_pg)
        var args = FullAttnChunkArgs(
            Int(q_i8), Int(qi_biases), Int(q_scales),
            Int(cache_buf), kv_head,
            pg_start, pg_end, num_positions,
            Int(chunk_partials) + c * heads_per_group * REF_PARTIAL_STRIDE * size_of[Float32](),
            heads_per_group)
        full_attn_chunk_kernel[head_dim, max_seq, num_kv_heads, num_q_heads](args)

    # Reduce to i8 + per-head scales
    var chunked_qi = alloc[Scalar[DType.int8]](heads_per_group * head_dim)
    var chunked_scales = alloc[Float32](heads_per_group)
    full_attn_reduce[head_dim](chunk_partials, num_chunks, heads_per_group,
        chunked_qi, chunked_scales)

    # Reference: also FWHT + quantize for apples-to-apples i8 comparison
    var ref_qi = alloc[Scalar[DType.int8]](heads_per_group * head_dim)
    var ref_scales = alloc[Float32](heads_per_group)
    for qh in range(heads_per_group):
        fwht_block[head_dim](ref_out + qh * head_dim)
        ref_scales[qh] = absmax_quantize_i8[head_dim](ref_out + qh * head_dim, ref_qi + qh * head_dim)

    # Compare dequantized i8 outputs
    var min_cos = Float64(1.0)
    var max_nrmse = Float64(0)
    for qh in range(heads_per_group):
        var ref_f64 = alloc[Float64](head_dim)
        var chunked_f64 = alloc[Float64](head_dim)
        var ref_dq = Float64(ref_scales[qh]) / 127.0
        var chunked_dq = Float64(chunked_scales[qh]) / 127.0
        for d in range(head_dim):
            ref_f64[d] = Float64(Int(ref_qi[qh * head_dim + d])) * ref_dq
            chunked_f64[d] = Float64(Int(chunked_qi[qh * head_dim + d])) * chunked_dq
        var cos = cosine_sim_f64(ref_f64.bitcast[Float64](), chunked_f64.bitcast[Float64](), head_dim)
        var sum_sq_err = Float64(0)
        var sum_sq_ref = Float64(0)
        for d in range(head_dim):
            sum_sq_err += (chunked_f64[d] - ref_f64[d]) * (chunked_f64[d] - ref_f64[d])
            sum_sq_ref += ref_f64[d] * ref_f64[d]
        var nrmse = Float64(sqrt[DType.float64, 1](sum_sq_err / sum_sq_ref)) if sum_sq_ref > 0 else Float64(0)
        if cos < min_cos:
            min_cos = cos
        if nrmse > max_nrmse:
            max_nrmse = nrmse
        ref_f64.free()
        chunked_f64.free()

    print("  heads compared:                    " + String(heads_per_group))
    print("  chunks:                            " + String(num_chunks))
    print("  min cosine (chunked vs ref i8):    " + String(min_cos))
    print("  max NRMSE (dequant comparison):    " + String(max_nrmse))

    cache_buf.free()
    q_i8.free()
    qi_biases.free()
    q_scales.free()
    ref_out.free()
    ref_partial.free()
    chunk_partials.free()
    chunked_qi.free()
    chunked_scales.free()
    ref_qi.free()
    ref_scales.free()


def main():
    print("=== full_attention validation (chunked flash-combine, width=" + String(WIDTH) + ") ===")
    print("\nChunked attention vs single-pass (head_dim=512, 256 positions, 4 chunks, 8 Q heads):")
    validate_chunked_attention()
