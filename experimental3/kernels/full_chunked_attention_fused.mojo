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

from kernels.kernel_ops import PoolFence
from threading.threading_traits import BurstThreadPool

from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.full_chunked_attention import (
    partial_head_stride, partial_chunk_stride,
    ChunkedAttnArgs,
)
from experimental3.kernels.sliding_attention import score_group, v_agg_group
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed_partial
from simd_math import exp_f32


# ============================================================================
# Position routing
# ============================================================================


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


# ============================================================================
# CP prep kernel — Q prep (all ranks) + conditional KV cache write
# ============================================================================


@fieldwise_init
struct CpAttnPrepArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_base: Int
    var k_bf16_ptr: Int
    var q_norm_ptr: Int
    var k_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var cache_pos: Int
    var kv_head: Int
    var eps: Float32
    var q_i8_out: Int
    var qi_biases_out: Int
    var q_scales_out: Int
    var write_kv: Int32

    def __init__(out self):
        self.q_bf16_base = 0
        self.k_bf16_ptr = 0
        self.q_norm_ptr = 0
        self.k_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.cache_base = 0
        self.cache_pos = 0
        self.kv_head = 0
        self.eps = Float32(0)
        self.q_i8_out = 0
        self.qi_biases_out = 0
        self.q_scales_out = 0
        self.write_kv = Int32(0)


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
        args.cache_base)
    var cos = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.cos_ptr)
    var sin = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.sin_ptr)

    if args.write_kv != 0:
        var work_arr = InlineArray[Float32, head_dim](uninitialized=True)
        var work = UnsafePointer(to=work_arr).bitcast[Float32]()
        var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
        var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

        write_k_head_normed[head_dim, rope_dims](
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=args.k_bf16_ptr),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=args.k_norm_ptr),
            cos, sin, work, qi_buf,
            cache, args.cache_pos, args.kv_head, args.eps)

        write_v_head_normed[head_dim](
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=args.k_bf16_ptr),
            work, qi_buf,
            cache, args.cache_pos, args.kv_head, args.eps)

        _ = work_arr
        _ = qi_arr

    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.qi_biases_out)
    var q_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.q_scales_out)

    for qh in range(heads_per_group):
        var q_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.q_bf16_base + qh * head_dim * 2)
        var q_i8_dst = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=args.q_i8_out + qh * head_dim)
        var result = prep_q_row_normed_partial[head_dim, rope_dims](
            q_bf16.bitcast[BFloat16](),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=args.q_norm_ptr),
            cos, sin,
            q_i8_dst.bitcast[Int8](), args.eps)
        qi_biases[qh] = result[0]
        q_scales[qh] = result[1]


def cp_attn_prep_dispatch[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    P: BurstThreadPool,
](
    q_bf16_base: Int, k_bf16_ptr: Int,
    q_norm_ptr: Int, k_norm_ptr: Int,
    cos_ptr: Int, sin_ptr: Int,
    cache_base: Int, local_pos: Int, kv_head: Int,
    eps: Float32, write_kv: Bool,
    q_i8_out: Int, qi_biases_out: Int, q_scales_out: Int,
    mut pool: P,
) -> PoolFence[P]:
    var args = CpAttnPrepArgs(
        q_bf16_base=q_bf16_base, k_bf16_ptr=k_bf16_ptr,
        q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
        cos_ptr=cos_ptr, sin_ptr=sin_ptr,
        cache_base=cache_base, cache_pos=local_pos, kv_head=kv_head,
        eps=eps,
        q_i8_out=q_i8_out, qi_biases_out=qi_biases_out, q_scales_out=q_scales_out,
        write_kv=Int32(1) if write_kv else Int32(0))
    pool.dispatch[CpAttnPrepArgs,
        cp_attn_prep_kernel[head_dim, rope_dims, heads_per_group,
            local_max_seq, num_kv_heads, num_q_heads]](
        UnsafePointer(to=args), 1)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# CP chunked attention dispatch
# ============================================================================
#
# Re-uses the existing chunked_attn_kernel from full_chunked_attention.mojo
# but with CP cache parameters (local_max_seq, all KV heads).


comptime MAX_CHUNKS = 32


def cp_chunked_attn_kernel[
    head_dim: Int, local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int,
](args: ChunkedAttnArgs):
    """Score one KV-cache chunk against all Q heads — CP variant.

    Identical to chunked_attn_kernel but instantiated with CP cache params
    (local_max_seq instead of global max_seq, all KV heads).
    """
    var cache = Gemma4KVCache[local_max_seq, head_dim, num_kv_heads, num_q_heads](
        args.cache_base)
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)

    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.qi_biases_base)
    var q_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.q_scales_base)

    comptime WIDTH = CACHE_WIDTH
    var running_max = InlineArray[Float32, heads_per_group](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, heads_per_group](fill=Float32(0))

    comptime V_ACC_TOTAL = heads_per_group * head_dim
    var v_acc_storage = InlineArray[Float32, V_ACC_TOTAL](fill=Float32(0))
    var v_acc_base = UnsafePointer(to=v_acc_storage).bitcast[Float32]()

    var scores_arr = InlineArray[Float32, WIDTH](uninitialized=True)
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    comptime Q_DENOM = Float32(127) * Float32(127)
    var q_factors = InlineArray[Float32, heads_per_group](uninitialized=True)
    for qh in range(heads_per_group):
        q_factors[qh] = q_scales[qh] / Q_DENOM

    for pg in range(args.start_pg, args.end_pg):
        var k_pg = cache.k_pg_ptr(args.kv_head, pg)
        var v_pg = cache.v_pg_ptr(args.kv_head, pg)
        var k_sc_pg = k_scales + pg * WIDTH
        var v_sc_pg = v_scales + pg * WIDTH
        var group_start = pg * WIDTH

        for qh in range(heads_per_group):
            var q_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
                unsafe_from_address=args.q_i8_base + qh * head_dim)

            score_group[head_dim](
                q_i8, k_pg, qi_biases[qh], q_factors[qh], k_sc_pg, scores)

            for lane in range(WIDTH):
                if group_start + lane >= args.context_len:
                    scores[lane] = Float32(-1e30)

            var scores_vec = scores.load[width=WIDTH]()
            var group_max = scores_vec.reduce_max()
            var new_max = max(running_max[qh], group_max)
            var v_acc = v_acc_base + qh * head_dim

            if running_sum[qh] > 0:
                var rescale = Float32(exp_f32[1](running_max[qh] - new_max))
                var d = 0
                while d + WIDTH <= head_dim:
                    (v_acc + d).store((v_acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
                running_sum[qh] *= rescale

            running_max[qh] = new_max
            var exp_scores = exp_f32[WIDTH](scores_vec - new_max)

            for lane in range(WIDTH):
                if group_start + lane >= args.context_len:
                    exp_scores[lane] = Float32(0)

            running_sum[qh] += exp_scores.reduce_add()
            v_agg_group[head_dim](exp_scores, v_sc_pg, v_pg, v_acc)

    var out = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.partial_out)
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


def cp_chunked_attn_dispatch[
    head_dim: Int, local_max_seq: Int,
    num_kv_heads: Int, num_q_heads: Int, heads_per_group: Int,
    P: BurstThreadPool,
](
    q_i8_base: Int, qi_biases_base: Int, q_scales_base: Int,
    cache_base: Int, kv_head: Int,
    local_context_len: Int, pool_capacity: Int,
    partial_out_base: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch chunked attention with CP cache parameters."""
    var num_pg = (local_context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var num_chunks = min(pool_capacity, MAX_CHUNKS)
    if num_chunks > num_pg:
        num_chunks = num_pg
    if num_chunks <= 0:
        return PoolFence[P].completed()
    var pgs_per_chunk = (num_pg + num_chunks - 1) // num_chunks
    comptime CHUNK_F32_STRIDE = heads_per_group * (2 + head_dim)
    var chunk_args = InlineArray[ChunkedAttnArgs, MAX_CHUNKS](fill=ChunkedAttnArgs())
    for c in range(num_chunks):
        var start = c * pgs_per_chunk
        var end = min((c + 1) * pgs_per_chunk, num_pg)
        chunk_args[c] = ChunkedAttnArgs(
            q_i8_base=q_i8_base,
            qi_biases_base=qi_biases_base,
            q_scales_base=q_scales_base,
            cache_base=cache_base,
            kv_head=kv_head,
            start_pg=start,
            end_pg=end,
            partial_out=partial_out_base + c * CHUNK_F32_STRIDE * 4,
            context_len=local_context_len)
    pool.dispatch[ChunkedAttnArgs,
        cp_chunked_attn_kernel[head_dim, local_max_seq, num_kv_heads, num_q_heads, heads_per_group]](
        UnsafePointer(to=chunk_args[0]), num_chunks)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Merge + quantize: cross-rank reduction
# ============================================================================


def merge_local_chunks[head_dim: Int, heads_per_group: Int](
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

    for qh in range(heads_per_group):
        var gmax = Float32(-1e30)
        for c in range(num_chunks):
            gmax = max(gmax, (partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE)[])

        var vp = out_v + qh * head_dim
        var d = 0
        while d + width <= head_dim:
            (vp + d).store(SIMD[DType.float32, width](0))
            d += width

        var total_sum = Float32(0)
        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var cs = (p + 1)[]
            if cs <= 0:
                continue
            var rescale = Float32(exp_f32[1](p[] - gmax))
            total_sum += cs * rescale
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
        merge_local_chunks[head_dim, heads_per_group](
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
# NUMA-safe dispatch wrappers
# ============================================================================
#
# The merge and gather kernels above are called directly by the benchmark.
# For the model's forward path, these dispatch wrappers run the same work
# on NUMA-local pool workers so all writes target local memory.

comptime MAX_CP_RANKS = 8


# --- merge_local_chunks dispatch: runs chunk merge on rank-local pool ---


@fieldwise_init
struct MergeChunksArgs(Copyable, ImplicitlyCopyable):
    var partial_base: Int
    var num_chunks: Int
    var out_m: Int
    var out_l: Int
    var out_v: Int

    def __init__(out self):
        self.partial_base = 0
        self.num_chunks = 0
        self.out_m = 0
        self.out_l = 0
        self.out_v = 0


def merge_local_chunks_kernel[head_dim: Int, heads_per_group: Int](
    args: MergeChunksArgs,
):
    merge_local_chunks[head_dim, heads_per_group](
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.partial_base),
        args.num_chunks,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.out_m),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.out_l),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.out_v))


def merge_local_chunks_dispatch[
    head_dim: Int, heads_per_group: Int, P: BurstThreadPool,
](
    partial_base: Int, num_chunks: Int,
    out_m: Int, out_l: Int, out_v: Int,
    mut pool: P,
) -> PoolFence[P]:
    if num_chunks <= 0:
        return PoolFence[P].completed()
    var args = MergeChunksArgs(
        partial_base=partial_base, num_chunks=num_chunks,
        out_m=out_m, out_l=out_l, out_v=out_v)
    pool.dispatch[MergeChunksArgs,
        merge_local_chunks_kernel[head_dim, heads_per_group]](
        UnsafePointer(to=args), 1)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# --- cp_gather_and_quantize dispatch: cross-rank V merge on rank-local pool ---


@fieldwise_init
struct CpGatherArgs(Copyable, ImplicitlyCopyable):
    var rank: Int
    var head_start: Int
    var head_count: Int
    var qi_out: Int
    var head_scales: Int
    var all_m: InlineArray[Int, MAX_CP_RANKS]
    var all_l: InlineArray[Int, MAX_CP_RANKS]
    var all_v: InlineArray[Int, MAX_CP_RANKS]

    def __init__(out self):
        self.rank = 0
        self.head_start = 0
        self.head_count = 0
        self.qi_out = 0
        self.head_scales = 0
        self.all_m = InlineArray[Int, MAX_CP_RANKS](fill=0)
        self.all_l = InlineArray[Int, MAX_CP_RANKS](fill=0)
        self.all_v = InlineArray[Int, MAX_CP_RANKS](fill=0)


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
        all_m_ptrs[r] = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.all_m[r])
        all_l_ptrs[r] = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.all_l[r])
        all_v_ptrs[r] = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.all_v[r])
    cp_gather_and_quantize[head_dim, num_heads, tp](
        args.rank,
        all_m_ptrs, all_l_ptrs, all_v_ptrs,
        UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=args.qi_out),
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=args.head_scales),
        args.head_start, args.head_count)


def cp_gather_dispatch[
    head_dim: Int, num_heads: Int, tp: Int, P: BurstThreadPool,
](
    rank: Int,
    all_m: InlineArray[Int, MAX_CP_RANKS],
    all_l: InlineArray[Int, MAX_CP_RANKS],
    all_v: InlineArray[Int, MAX_CP_RANKS],
    qi_out: Int, head_scales: Int,
    head_start: Int, head_count: Int,
    mut pool: P,
) -> PoolFence[P]:
    var args = CpGatherArgs(
        rank=rank, head_start=head_start, head_count=head_count,
        qi_out=qi_out, head_scales=head_scales,
        all_m=all_m, all_l=all_l, all_v=all_v)
    pool.dispatch[CpGatherArgs,
        cp_gather_kernel[head_dim, num_heads, tp]](
        UnsafePointer(to=args), 1)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))
