"""Full (global) attention decode for Gemma 4.

K=V is shared, so the kernel writes both K and V cache entries from the same
projection output. Q heads sharing one KV head are processed together.

head_dim=512, 2 KV heads, 16 Q heads (GQA 8:1), partial RoPE (128/512 dims).
"""

from std.memory import UnsafePointer
from std.collections import InlineArray

from experimental2.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.sliding_attention import (
    single_pass_attention,
)
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed_partial, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed_partial


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
    var eps: Float32


def full_attn_group_kernel[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: FullAttnGroupArgs):
    """Full attention: KV write + Q prep + single-pass score + FWHT + quantize.

    K=V shared: both K and V cache are written from the K projection output.
    K gets partial RoPE, V gets per-head /rms then per-token absmax quantization.
    Uses the same single-pass online softmax path as sliding attention.
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

    # Write V from same K output (no RoPE, per-head /rms, per-token absmax)
    write_v_head_normed[head_dim](
        k_bf16, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

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
