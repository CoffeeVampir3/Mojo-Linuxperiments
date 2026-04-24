from std.collections import InlineArray
from std.sys.info import simd_width_of

from simd_math import sqrt, exp_f32
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from minimax.kernels.qk_prep import prep_q_head, write_k_head, write_v_direct
from minimax.kernels.dispatch_args import AttnGroupArgs, MergeQuantArgs


# ============================================================================
# Phase A: K/V cache write — one job per local KV head
# ============================================================================


def kv_write_kernel[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    max_seq: Int, num_kv_heads: Int,
](args: AttnGroupArgs):
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads](
        Int(args.cache_base))

    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    write_k_head[head_dim, rope_dim, pair_stride](
        args.k_bf16_ptr, args.k_norm_ptr,
        args.cos_ptr, args.sin_ptr,
        args.inv_rms_k,
        qi_buf,
        cache, args.cache_pos, args.kv_head)

    write_v_direct[head_dim](
        args.v_bf16_ptr,
        qi_buf,
        cache, args.cache_pos, args.kv_head)

    _ = qi_arr


# ============================================================================
# Phase C: local merge + quantize — replaces cross-rank cp_gather
# ============================================================================


def partial_head_stride[head_dim: Int]() -> Int:
    return 2 + head_dim


def partial_chunk_stride[head_dim: Int, heads_per_group: Int]() -> Int:
    return heads_per_group * partial_head_stride[head_dim]()


def merge_and_quantize_kernel[head_dim: Int, heads_per_group: Int, max_attn_chunks: Int](
    partial_base: F32Ptr,
    num_chunks: Int,
    qi_out: I8Ptr,
    head_scale_ptr: F32Ptr,
):
    """Merge position chunks + quantize output for O-projection.

    Reads partial (max, sum, v_acc) from each chunk, merges via online
    softmax, applies 1/(127*L) normalization, quantizes to i8.
    """
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()
    comptime CHUNK_STRIDE = partial_chunk_stride[head_dim, heads_per_group]()
    comptime width = simd_width_of[DType.float32]()

    var work_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var wp = UnsafePointer(to=work_arr).bitcast[Float32]()

    var m_arr = InlineArray[Float32, max_attn_chunks](fill=Float32(-1e30))
    var cs_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)
    var rescale_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)

    for qh in range(heads_per_group):
        # Gather per-chunk (max, sum) for this head; compute gmax.
        var gmax = Float32(-1e30)
        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var m = p[]
            m_arr[c] = m
            cs_arr[c] = (p + 1)[]
            gmax = max(gmax, m)

        # One batched SIMD exp covers all chunks; inactive lanes stay at -1e30.
        var gmax_vec = SIMD[DType.float32, width](gmax)
        var cg = 0
        while cg < max_attn_chunks:
            var mv = UnsafePointer(to=m_arr[cg]).load[width=width]()
            UnsafePointer(to=rescale_arr[cg]).store(exp_f32[width](mv - gmax_vec))
            cg += width

        var d = 0
        while d + width <= head_dim:
            (wp + d).store(SIMD[DType.float32, width](0))
            d += width

        var total_sum = Float32(0)
        for c in range(num_chunks):
            var cs = cs_arr[c]
            if cs <= 0:
                continue
            var rescale = rescale_arr[c]
            total_sum += cs * rescale
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            d = 0
            while d + width <= head_dim:
                (wp + d).store((p + 2 + d).load[width=width]().fma(
                    rescale, (wp + d).load[width=width]()))
                d += width

        var inv = Float32(1) / (Float32(127) * total_sum)
        d = 0
        while d + width <= head_dim:
            (wp + d).store((wp + d).load[width=width]() * inv)
            d += width

        head_scale_ptr[qh] = absmax_quantize_i8[head_dim](
            wp, qi_out + qh * head_dim)

    _ = work_arr


def merge_quant_worker[
    head_dim: Int,
    heads_per_group: Int,
    max_attn_chunks: Int,
    kv_heads: Int,
](args: MergeQuantArgs):
    if args.num_chunks <= 0:
        return

    comptime CHUNK_F32_STRIDE = partial_chunk_stride[
        head_dim, heads_per_group]()
    comptime KV_QI_STRIDE = heads_per_group * head_dim
    comptime KV_SCALE_STRIDE = heads_per_group

    for kv in range(kv_heads):
        merge_and_quantize_kernel[
            head_dim, heads_per_group, max_attn_chunks](
            args.partial_base + kv * args.num_chunks * CHUNK_F32_STRIDE,
            args.num_chunks,
            args.qi_out + kv * KV_QI_STRIDE,
            args.head_scale_ptr + kv * KV_SCALE_STRIDE,
        )
