"""Full (global) attention decode — K=V shared, partial RoPE."""

from std.memory import UnsafePointer
from std.collections import InlineArray

from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.sliding_attention import (
    AttnGroupArgs,
    single_pass_attention,
)
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed_partial


# ============================================================================
# Fused full attention group kernel — KV write + Q prep + score + quantize
# ============================================================================


def full_attn_group_kernel[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: AttnGroupArgs):
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

    # Write K with partial RoPE
    write_k_head_normed[head_dim, rope_dims](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_bf16_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.k_norm_ptr),
        cos, sin, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    # Write V (for full attention, v_bf16_ptr == k_bf16_ptr since K=V shared)
    write_v_head_normed[head_dim](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.v_bf16_ptr),
        work, qi_buf,
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
