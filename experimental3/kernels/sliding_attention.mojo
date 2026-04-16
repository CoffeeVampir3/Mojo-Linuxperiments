"""Sliding window decode attention — single-pass online softmax with VNNI."""

from std.memory import UnsafePointer
from std.collections import InlineArray

from kernels.kernel_ops import PoolFence
from threading.threading_traits import BurstThreadPool
from experimental3.amx import VNNI_BLK
from experimental3.kernels.int8_gemv import vpdpbusd
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed
from simd_math import exp_f32, roundeven


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
    v_scales_pg: UnsafePointer[Float32, MutAnyOrigin],
    v_pg: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    v_acc: UnsafePointer[Float32, MutAnyOrigin],
):
    """Accumulate V-agg for WIDTH positions using VNNI.

    Folds per-position v_scales into exp_scores before u8 quantization,
    so w_dequant = w_max/255 absorbs both attention weight and V scale.
    """
    comptime V_CHANNEL_GROUPS = head_dim // WIDTH
    comptime V_SUB_QUADS = WIDTH // VNNI_BLK
    comptime V_CG_BYTES = V_SUB_QUADS * WIDTH * VNNI_BLK

    # Fold per-position V scale into the attention weight before u8 quantize.
    var v_sc = v_scales_pg.load[width=WIDTH]()
    var w_eff = exp_scores * v_sc

    var w_max = w_eff.reduce_max()
    if w_max < Float32(1e-10):
        return
    var w_scale = 255.0 / w_max
    var w_u8_wide = roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
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

    V is stored with per-token dynamic absmax scales. v_agg_group folds those
    scales into the attention-weight u8 quantization, so dequant is one mul
    at the end (1/(127*sum)) regardless of how many positions there are.
    Output: f32[head_dim] in FWHT domain.
    """
    var q_factor = q_scale / (127.0 * 127.0)
    var k_scales = cache.k_scale_ptr(kv_head)
    var v_scales = cache.v_scale_ptr(kv_head)
    var num_pos_groups = (context_len + WIDTH - 1) // WIDTH

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)

    var v_acc_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var v_acc = UnsafePointer(to=v_acc_arr).bitcast[Float32]()

    var scores_arr = InlineArray[Float32, WIDTH](fill=Float32(0))
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    for pg in range(num_pos_groups):
        var k_pg = cache.k_pg_ptr(kv_head, pg)
        var v_pg = cache.v_pg_ptr(kv_head, pg)

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
        v_agg_group[head_dim](exp_scores, v_scales + pg * WIDTH, v_pg, v_acc)

    var inv_sum = Float32(1.0) / (Float32(127) * running_sum)
    for d in range(0, head_dim, WIDTH):
        (output + d).store((v_acc + d).load[width=WIDTH]() * inv_sum)


# ============================================================================
# Per-KV-group worker
# ============================================================================


@fieldwise_init
struct AttnGroupArgs(Copyable, ImplicitlyCopyable):
    """Shared args for both sliding and full attention group kernels.

    For full attention (K=V shared), set v_bf16_ptr = k_bf16_ptr.
    """
    var q_bf16_ptr: Int
    var k_bf16_ptr: Int
    var v_bf16_ptr: Int
    var q_norm_ptr: Int
    var k_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var kv_head: Int
    var cache_pos: Int
    var context_len: Int
    var qi_out_ptr: Int
    var head_scale_ptr: Int
    var eps: Float32

    def __init__(out self):
        self.q_bf16_ptr = 0
        self.k_bf16_ptr = 0
        self.v_bf16_ptr = 0
        self.q_norm_ptr = 0
        self.k_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.cache_base = 0
        self.kv_head = 0
        self.cache_pos = 0
        self.context_len = 0
        self.qi_out_ptr = 0
        self.head_scale_ptr = 0
        self.eps = Float32(0)


def sliding_attn_group_kernel[
    head_dim: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: AttnGroupArgs):
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
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_norm_ptr),
        cos, sin, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    # 2. Write V to cache (per-token absmax, scale stored in cache)
    write_v_head_normed[head_dim](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.v_bf16_ptr),
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
            q_bf16.bitcast[BFloat16](),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.q_norm_ptr),
            cos, sin,
            q_i8.bitcast[Int8](), args.eps)
        var qi_bias = result[0]
        var q_scale = result[1]

        # Single-pass: score + softmax + V-agg → f32[head_dim] on stack
        single_pass_attention[head_dim, max_seq, num_kv_heads, num_q_heads](
            q_i8, qi_bias, q_scale,
            cache, args.kv_head, args.context_len,
            work)

        head_scales[qh] = absmax_quantize_i8[head_dim](work, qi_out + qh * head_dim)


# ============================================================================
# Dispatch wrapper
# ============================================================================


def sliding_attn_dispatch[
    head_dim: Int, heads_per_group: Int, window_size: Int,
    num_kv_heads: Int, num_q_heads: Int, P: BurstThreadPool,
](
    q_base: Int, k_base: Int, v_base: Int,
    q_norm_ptr: Int, k_norm_ptr: Int,
    cos_ptr: Int, sin_ptr: Int,
    cache_base: Int, cache_pos: Int, context_len: Int,
    qi_out_ptr: Int, head_scale_ptr: Int,
    eps: Float32, mut pool: P,
) -> PoolFence[P]:
    comptime NKV = num_kv_heads
    var jobs = InlineArray[AttnGroupArgs, 8](fill=AttnGroupArgs())
    for g in range(NKV):
        jobs[g] = AttnGroupArgs(
            q_base + g * heads_per_group * head_dim * 2,
            k_base + g * head_dim * 2,
            v_base + g * head_dim * 2,
            q_norm_ptr, k_norm_ptr,
            cos_ptr, sin_ptr,
            cache_base, g,
            cache_pos, context_len,
            qi_out_ptr + g * heads_per_group * head_dim,
            head_scale_ptr + g * heads_per_group * 4,
            eps)
    pool.dispatch[AttnGroupArgs,
        sliding_attn_group_kernel[head_dim, heads_per_group,
            window_size, num_kv_heads, num_q_heads]](
        UnsafePointer(to=jobs[0]), NKV)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))
