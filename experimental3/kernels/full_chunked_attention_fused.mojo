"""Context-parallel full attention — prep, scoring, and fused reduction.

When TP exceeds NUM_KV_HEADS_FULL, full attention uses context parallelism:
each rank owns a round-robin slice of KV cache positions and computes
partial softmax states for all Q heads against its local positions.

Position routing: rank r owns global positions where pos % tp == r.
Local position index: local_pos = pos // tp.
Local cache: Gemma4KVCache[local_max_seq, head_dim, num_kv_full, num_heads].

Pipeline per layer:
  1. cp_attn_prep_dispatch: Q prep (all ranks) + KV cache write (owning rank)
  2. chunked_attn_dispatch: Q heads × local KV positions → partial states
  3. cp_merge_and_quantize: cross-rank scalar gather → rescale → quantize
  4. int8_gemv_blocked: O-proj GEMV (existing, column-sharded)
  5. ring_allreduce: sum per-rank bf16 outputs (existing)

The reduction absorbs the cross-rank softmax correction into quantization:
  O(merged_v) = sum_r(O(v_r * alpha_r / L))    [O-proj linearity]
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.sliding_attention import score_group, v_agg_group
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed_partial
from experimental3.kernels.dispatch_args import (
    ChunkedAttnArgs, CpAttnPrepArgs, MergeChunksArgs, CpGatherArgs,
    MAX_CP_RANKS,
)
from simd_math import exp_f32


# ============================================================================
# Position routing
# ============================================================================


# ============================================================================
# Partial state layout
# ============================================================================
#
# Per Q head per chunk: [max: f32, sum: f32, v_acc: f32 × head_dim]
# Stride per head   = (2 + head_dim) f32s
# Stride per chunk  = heads_per_group × stride_per_head


@always_inline
def partial_head_stride[head_dim: Int]() -> Int:
    """F32 element stride between consecutive Q heads in a partial buffer."""
    return 2 + head_dim


@always_inline
def partial_chunk_stride[head_dim: Int, heads_per_group: Int]() -> Int:
    """F32 element stride between consecutive chunks in a partial buffer."""
    return heads_per_group * partial_head_stride[head_dim]()


@always_inline
def cp_owning_rank(pos: Int, tp: Int) -> Int:
    return pos % tp


@always_inline
def cp_local_pos(pos: Int, tp: Int) -> Int:
    return pos // tp


@always_inline
def cp_local_context_len(global_context_len: Int, rank: Int, tp: Int) -> Int:
    """How many positions this rank holds for a given global context length."""
    return (global_context_len + tp - 1 - rank) // tp


@always_inline
def cp_local_max_seq[max_seq: Int, tp: Int]() -> Int:
    return (max_seq + tp - 1) // tp


@always_inline
def cp_group_valid_mask[width: Int](
    group_start: Int, context_len: Int,
) -> SIMD[DType.bool, width]:
    var lanes = SIMD[DType.int32, width]()
    comptime for lane in range(width):
        lanes[lane] = Int32(group_start + lane)
    return lanes.lt(SIMD[DType.int32, width](context_len))


# ============================================================================
# CP prep kernel — Q prep (all ranks) + conditional KV cache write
# ============================================================================


def cp_attn_prep_kernel[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: CpAttnPrepArgs):
    """Q prep for all heads + conditional KV cache write.

    When write_kv is set, writes K and V to the cache at cache_pos (which
    should be the local position index, not global). When clear, only Q
    prep runs — used on non-owning CP ranks.
    """
    var cache = Gemma4KVCache[local_max_seq, head_dim, num_kv_heads, num_q_heads](
        Int(args.cache_base))
    var cos = args.cos_ptr
    var sin = args.sin_ptr

    if args.write_kv != 0:
        var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
        var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

        write_k_head_normed[head_dim, rope_dims](
            args.k_bf16_ptr,
            args.k_norm_ptr,
            cos, sin, qi_buf,
            cache, args.cache_pos, args.kv_head, args.eps)

        write_v_head_normed[head_dim](
            args.k_bf16_ptr,
            qi_buf,
            cache, args.cache_pos, args.kv_head, args.eps)

        _ = qi_arr

    var qi_biases = args.qi_biases_out
    var q_scales = args.q_scales_out

    for qh in range(heads_per_group):
        var q_bf16 = args.q_bf16_base + qh * head_dim
        var q_i8_dst = args.q_i8_out + qh * head_dim
        var result = prep_q_row_normed_partial[head_dim, rope_dims](
            q_bf16.bitcast[BFloat16](),
            args.q_norm_ptr,
            cos, sin,
            q_i8_dst.bitcast[Int8](), args.eps)
        qi_biases[qh] = result[0]
        q_scales[qh] = result[1]


# ============================================================================
# CP chunked attention worker
# ============================================================================


def cp_chunked_attn_kernel[
    head_dim: Int, local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int, max_attn_chunks: Int,
](args: ChunkedAttnArgs):
    """Score one KV-cache chunk against all Q heads — CP variant.

    Identical to chunked_attn_kernel but instantiated with CP cache params
    (local_max_seq instead of global max_seq, all KV heads).
    """
    var cache = Gemma4KVCache[local_max_seq, head_dim, num_kv_heads, num_q_heads](
        Int(args.cache_base))
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)

    var qi_biases = args.qi_biases_base
    var q_scales = args.q_scales_base

    comptime WIDTH = CACHE_WIDTH
    var running_max = InlineArray[Float32, heads_per_group](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, heads_per_group](fill=Float32(0))

    comptime V_ACC_TOTAL = heads_per_group * head_dim
    var v_acc_storage = InlineArray[Float32, V_ACC_TOTAL](fill=Float32(0))
    var v_acc_base = UnsafePointer(to=v_acc_storage).bitcast[Float32]()

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero = SIMD[DType.float32, WIDTH](0)

    comptime Q_DENOM = Float32(127) * Float32(127)
    var q_factors = InlineArray[Float32, heads_per_group](uninitialized=True)
    for qh in range(heads_per_group):
        q_factors[qh] = q_scales[qh] / Q_DENOM

    comptime EXP_W = simd_width_of[DType.float32]()
    debug_assert(heads_per_group <= EXP_W, "heads_per_group must fit in one SIMD vector")
    var scores_storage = InlineArray[
        SIMD[DType.float32, WIDTH], heads_per_group](uninitialized=True)
    var new_max_arr = InlineArray[Float32, heads_per_group](uninitialized=True)

    for pg in range(args.start_pg, args.end_pg):
        var k_pg = cache.k_pg_ptr(args.kv_head, pg)
        var v_pg = cache.v_pg_ptr(args.kv_head, pg)
        var k_sc_pg = k_scales + pg * WIDTH
        var v_sc_pg = v_scales + pg * WIDTH
        var group_start = pg * WIDTH
        var valid = cp_group_valid_mask[WIDTH](group_start, args.context_len)

        # Phase A: score all heads, compute new_max per head.
        var diff_vec = SIMD[DType.float32, EXP_W](0)
        for qh in range(heads_per_group):
            var q_i8 = args.q_i8_base + qh * head_dim
            var scores_vec = valid.select(
                score_group[head_dim](
                    q_i8, k_pg, qi_biases[qh], q_factors[qh], k_sc_pg),
                neg_inf,
            )
            scores_storage[qh] = scores_vec
            var new_max = max(running_max[qh], scores_vec.reduce_max())
            new_max_arr[qh] = new_max
            diff_vec[qh] = running_max[qh] - new_max

        # One batched SIMD exp replaces heads_per_group scalar exp calls.
        var rescales = exp_f32[EXP_W](diff_vec)

        # Phase B: rescale v_acc, accumulate scores into running state.
        for qh in range(heads_per_group):
            var new_max = new_max_arr[qh]
            var v_acc = v_acc_base + qh * head_dim

            if running_sum[qh] > 0:
                var rescale = rescales[qh]
                var d = 0
                while d + WIDTH <= head_dim:
                    (v_acc + d).store((v_acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
                running_sum[qh] *= rescale

            running_max[qh] = new_max
            var exp_scores = valid.select(
                exp_f32[WIDTH](scores_storage[qh] - new_max), zero)

            running_sum[qh] += exp_scores.reduce_add()
            v_agg_group[head_dim](exp_scores, v_sc_pg, v_pg, v_acc)

    var out = args.partial_out
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()

    for qh in range(heads_per_group):
        var dst = out + qh * HEAD_STRIDE
        dst[] = running_max[qh]
        (dst + 1)[] = running_sum[qh]
        var src = v_acc_base + qh * head_dim
        var d = 0
        while d + WIDTH <= head_dim:
            (dst + 2 + d).store((src + d).load[width=WIDTH]())
            d += WIDTH


# ============================================================================
# Merge + quantize: cross-rank reduction
# ============================================================================


def merge_local_chunks[head_dim: Int, heads_per_group: Int, max_attn_chunks: Int](
    partial_base: UnsafePointer[Float32, MutAnyOrigin],
    num_chunks: Int,
    out_m: UnsafePointer[Float32, MutAnyOrigin],
    out_l: UnsafePointer[Float32, MutAnyOrigin],
    out_v: UnsafePointer[Float32, MutAnyOrigin],
):
    """Merge this rank's partial chunks for one KV group into (m, l, v) per head."""
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()
    comptime CHUNK_STRIDE = partial_chunk_stride[head_dim, heads_per_group]()
    comptime width = simd_width_of[DType.float32]()

    var m_arr = InlineArray[Float32, max_attn_chunks](fill=Float32(-1e30))
    var cs_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)
    var rescale_arr = InlineArray[Float32, max_attn_chunks](uninitialized=True)

    for qh in range(heads_per_group):
        # Gather (max, sum) from each chunk; compute gmax.
        var gmax = Float32(-1e30)
        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var m = p[]
            m_arr[c] = m
            cs_arr[c] = (p + 1)[]
            gmax = max(gmax, m)

        # Batched SIMD exp across all chunks — unused slots are -1e30 → 0.
        var gmax_vec = SIMD[DType.float32, width](gmax)
        var cg = 0
        while cg < max_attn_chunks:
            var mv = UnsafePointer(to=m_arr[cg]).load[width=width]()
            UnsafePointer(to=rescale_arr[cg]).store(exp_f32[width](mv - gmax_vec))
            cg += width

        var vp = out_v + qh * head_dim
        var d = 0
        while d + width <= head_dim:
            (vp + d).store(SIMD[DType.float32, width](0))
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
                (vp + d).store((p + 2 + d).load[width=width]().fma(
                    rescale, (vp + d).load[width=width]()))
                d += width

        out_m[qh] = gmax
        out_l[qh] = total_sum


def cp_gather_and_quantize[head_dim: Int, num_heads: Int, tp: Int](
    rank: Int,
    all_m_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    all_l_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    all_v_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    head_scales: UnsafePointer[Float32, MutAnyOrigin],
    head_start: Int,
    head_count: Int,
):
    """Cross-rank V merge + quantize for column-sharded O-proj.

    For each head this rank owns, reads (m, l, v) from ALL ranks and
    computes the fully merged attention output before quantization.
    Remote reads: head_count × head_dim × tp × 4 bytes (~32KB at TP=4).

    The 1/127 pre-division matches merge_and_quantize's contract with the
    GEMV kernel: blk_scale carries the absmax, GEMV dequant divides by 127.
    """
    comptime width = simd_width_of[DType.float32]()

    var work = InlineArray[Float32, head_dim](fill=Float32(0))
    var wp = UnsafePointer(to=work).bitcast[Float32]()

    for local_h in range(head_count):
        var gh = head_start + local_h

        var global_M = Float32(-1e30)
        for r in range(tp):
            global_M = max(global_M, all_m_ptrs[r][gh])

        var global_L = Float32(0)
        for r in range(tp):
            global_L += all_l_ptrs[r][gh] * Float32(exp_f32[1](
                all_m_ptrs[r][gh] - global_M))

        var inv = Float32(1) / (Float32(127) * global_L)

        var d = 0
        while d + width <= head_dim:
            (wp + d).store(SIMD[DType.float32, width](0))
            d += width

        for r in range(tp):
            var alpha_r = Float32(exp_f32[1](all_m_ptrs[r][gh] - global_M))
            var src = all_v_ptrs[r] + gh * head_dim
            d = 0
            while d + width <= head_dim:
                (wp + d).store((src + d).load[width=width]().fma(
                    alpha_r, (wp + d).load[width=width]()))
                d += width

        d = 0
        while d + width <= head_dim:
            (wp + d).store((wp + d).load[width=width]() * inv)
            d += width

        head_scales[local_h] = absmax_quantize_i8[head_dim](
            wp, qi_out + local_h * head_dim)

    _ = work


def cp_merge_and_quantize[
    head_dim: Int,
    heads_per_group: Int,
    num_kv: Int,
    num_heads: Int,
    tp: Int,
    max_attn_chunks: Int,
](
    rank: Int,
    partial_base: UnsafePointer[Float32, MutAnyOrigin],
    num_chunks: Int,
    local_m: UnsafePointer[Float32, MutAnyOrigin],
    local_l: UnsafePointer[Float32, MutAnyOrigin],
    local_v: UnsafePointer[Float32, MutAnyOrigin],
    all_m_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    all_l_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    all_v_ptrs: InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp],
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    head_scales: UnsafePointer[Float32, MutAnyOrigin],
    head_start: Int,
    head_count: Int,
):
    """Full fused CP merge: local chunk merge → cross-rank V gather → quantize.

    Called once per rank after chunked_attn_dispatch join. Merges this rank's
    local chunks, then gathers V from all ranks to produce the fully merged
    attention output before quantization. ~32KB cross-rank reads at TP=4.
    """
    comptime CHUNK_STRIDE = partial_chunk_stride[head_dim, heads_per_group]()

    for kv in range(num_kv):
        merge_local_chunks[head_dim, heads_per_group, max_attn_chunks](
            partial_base + kv * num_chunks * CHUNK_STRIDE,
            num_chunks,
            local_m + kv * heads_per_group,
            local_l + kv * heads_per_group,
            local_v + kv * heads_per_group * head_dim)

    cp_gather_and_quantize[head_dim, num_heads, tp](
        rank, all_m_ptrs, all_l_ptrs, all_v_ptrs,
        qi_out, head_scales,
        head_start, head_count)


# ============================================================================
# NUMA-safe worker wrappers
# ============================================================================
#
# The merge and gather kernels above are called directly by the benchmark.
# For the model's forward path, dispatchers in dispatch_kernels.mojo run the
# same work on NUMA-local pool workers so all writes target local memory.

def merge_local_chunks_kernel[head_dim: Int, heads_per_group: Int, max_attn_chunks: Int](
    args: MergeChunksArgs,
):
    merge_local_chunks[head_dim, heads_per_group, max_attn_chunks](
        args.partial_base,
        args.num_chunks,
        args.out_m,
        args.out_l,
        args.out_v)


def cp_gather_kernel[head_dim: Int, num_heads: Int, tp: Int](
    args: CpGatherArgs,
):
    var all_m_ptrs = InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    var all_l_ptrs = InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    var all_v_ptrs = InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    for r in range(tp):
        all_m_ptrs[r] = args.all_m[r]
        all_l_ptrs[r] = args.all_l[r]
        all_v_ptrs[r] = args.all_v[r]
    cp_gather_and_quantize[head_dim, num_heads, tp](
        args.rank,
        all_m_ptrs, all_l_ptrs, all_v_ptrs,
        args.qi_out,
        args.head_scales,
        args.head_start, args.head_count)
