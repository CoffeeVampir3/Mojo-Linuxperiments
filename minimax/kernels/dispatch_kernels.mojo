from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from simd_math import sqrt
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kernels.dispatch_args import Int8GemvBlockedArgs, ChunkedAttnArgs
from experimental3.kernels.gemm import int8_gemv_blocked_worker, int8_gemv_blocked_decode_worker
from kernels.vnni import VNNI_N_STEP
from experimental3.kv_cache import CACHE_WIDTH
from experimental3.kernels.full_chunked_attention_fused import (
    cp_chunked_attn_kernel,
)

from minimax.kernels.qk_prep import prep_q_head
from minimax.kernels.dispatch_args import (
    FusedW1W3SiluArgs,
    AttnGroupArgs,
    RouterCandidate,
    RouterFusedArgs,
)
from minimax.kernels.gemm import fused_w1_w3_silu_worker
from minimax.kernels.attention import kv_write_kernel
from minimax.kernels.router import TopKResult, router_fused_worker


@always_inline
def pool_fence[P: BurstThreadPool](mut pool: P) -> PoolFence[P]:
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# Fused router — f32 GEMV + sigmoid + bias + local top-K (phase 1)
# and cross-worker merge + renorm (phase 2).
# ============================================================================


@always_inline
def router_num_workers[num_experts: Int, k: Int](pool_capacity: Int) -> Int:
    """Shared formula: phase-1 worker count = min(num_experts/k, pool cap).
    Constrained by k so every worker has ≥ k rows and produces a full
    local top-K."""
    comptime max_workers = num_experts // k
    return min(max_workers, pool_capacity)


def router_fused_dispatch[num_experts: Int, hidden: Int, k: Int, P: BurstThreadPool](
    act_bf16: BF16Ptr,
    weight_f32: F32Ptr,
    bias_f32: F32Ptr,
    candidates: U8Ptr,
    mut pool: P,
) -> PoolFence[P]:
    var num_workers = router_num_workers[num_experts, k](pool.get_capacity())
    var rows_per_worker = (num_experts + num_workers - 1) // num_workers

    var jobs = InlineArray[RouterFusedArgs, MAX_POOL_CAPACITY](
        fill=RouterFusedArgs())
    var actual = 0
    for i in range(num_workers):
        var start = i * rows_per_worker
        if start >= num_experts:
            break
        var count = min(rows_per_worker, num_experts - start)
        var slot = candidates + i * k * size_of[RouterCandidate]()
        jobs[actual] = RouterFusedArgs(
            act_bf16, weight_f32, bias_f32, slot, start, count)
        actual += 1

    pool.dispatch[RouterFusedArgs, router_fused_worker[hidden, k]](
        UnsafePointer(to=jobs[0]), actual)
    return pool_fence(pool)


# ============================================================================
# Attention phase A: K/V cache write — NKV_LOCAL parallel jobs
# ============================================================================


def kv_write_dispatch[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    max_seq: Int, num_kv_heads: Int, P: BurstThreadPool,
](
    q_bf16_base: Int, k_bf16_base: Int, v_bf16_base: Int,
    q_norm_ptr: Int, k_norm_ptr: Int,
    cos_ptr: Int, sin_ptr: Int,
    inv_rms_q: Float32, inv_rms_k: Float32,
    cache_base: Int, cache_pos: Int,
    mut pool: P,
) -> PoolFence[P]:
    comptime HPG = 1
    var jobs = InlineArray[AttnGroupArgs, 8](fill=AttnGroupArgs())
    comptime K_HEAD_BF16 = head_dim * 2
    comptime V_HEAD_BF16 = head_dim * 2
    for g in range(num_kv_heads):
        jobs[g] = AttnGroupArgs(
            q_bf16_base=BF16Ptr(unsafe_from_address=q_bf16_base),
            k_bf16_ptr=BF16Ptr(unsafe_from_address=k_bf16_base + g * K_HEAD_BF16),
            v_bf16_ptr=BF16Ptr(unsafe_from_address=v_bf16_base + g * V_HEAD_BF16),
            q_norm_ptr=BF16Ptr(unsafe_from_address=q_norm_ptr),
            k_norm_ptr=BF16Ptr(unsafe_from_address=k_norm_ptr + g * K_HEAD_BF16),
            cos_ptr=F32Ptr(unsafe_from_address=cos_ptr),
            sin_ptr=F32Ptr(unsafe_from_address=sin_ptr),
            inv_rms_q=inv_rms_q,
            inv_rms_k=inv_rms_k,
            cache_base=U8Ptr(unsafe_from_address=cache_base),
            cache_pos=cache_pos,
            kv_head=g,
            context_len=0,
            qi_out=I8Ptr(),
            head_scale_ptr=F32Ptr())
    pool.dispatch[AttnGroupArgs,
        kv_write_kernel[head_dim, rope_dim, pair_stride, max_seq, num_kv_heads]](
        UnsafePointer(to=jobs[0]), num_kv_heads)
    return pool_fence(pool)


# ============================================================================
# Attention phase B: Q prep + chunked scoring — reuses cp_chunked_attn_kernel
# ============================================================================


def q_prep_kernel[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    heads_per_group: Int,
](args: AttnGroupArgs):
    comptime inv_sqrt_hd = 1.0 / sqrt[DType.float32, 1](Float32(head_dim))
    var qi_biases = F32Ptr(unsafe_from_address=Int(args.head_scale_ptr))
    var q_scales = F32Ptr(unsafe_from_address=args.context_len)
    for qh in range(heads_per_group):
        var result = prep_q_head[head_dim, rope_dim, pair_stride](
            args.q_bf16_base + qh * head_dim,
            args.q_norm_ptr + qh * head_dim,
            args.cos_ptr, args.sin_ptr,
            args.inv_rms_q,
            (args.qi_out + qh * head_dim).bitcast[Int8]())
        qi_biases[qh] = result[0]
        q_scales[qh] = result[1] * inv_sqrt_hd


def chunked_score_dispatch[
    head_dim: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, max_attn_chunks: Int,
    P: BurstThreadPool,
](
    q_i8_base: Int, qi_biases_base: Int, q_scales_base: Int,
    cache_base: Int, kv_head: Int,
    context_len: Int, pool_capacity: Int,
    partial_out_base: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch chunked scoring for one KV group across pool workers."""
    comptime WIDTH = CACHE_WIDTH
    var num_pg = (context_len + WIDTH - 1) // WIDTH
    var num_chunks = min(pool_capacity, max_attn_chunks)
    if num_chunks > num_pg:
        num_chunks = num_pg
    if num_chunks <= 0:
        return PoolFence[P].completed()
    var pgs_per_chunk = (num_pg + num_chunks - 1) // num_chunks
    comptime CHUNK_F32_STRIDE = heads_per_group * (2 + head_dim)

    var q_i8 = I8Ptr(unsafe_from_address=q_i8_base)
    var qi_biases = F32Ptr(unsafe_from_address=qi_biases_base)
    var q_scales = F32Ptr(unsafe_from_address=q_scales_base)
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var partial_out = F32Ptr(unsafe_from_address=partial_out_base)

    var chunk_args = InlineArray[ChunkedAttnArgs, max_attn_chunks](fill=ChunkedAttnArgs())
    for c in range(num_chunks):
        var start = c * pgs_per_chunk
        var end = min((c + 1) * pgs_per_chunk, num_pg)
        chunk_args[c] = ChunkedAttnArgs(
            q_i8_base=q_i8,
            qi_biases_base=qi_biases,
            q_scales_base=q_scales,
            cache_base=cache,
            kv_head=kv_head,
            start_pg=start,
            end_pg=end,
            partial_out=partial_out + c * CHUNK_F32_STRIDE,
            context_len=context_len)
    pool.dispatch[ChunkedAttnArgs,
        cp_chunked_attn_kernel[head_dim, max_seq, num_kv_heads, 0, heads_per_group, max_attn_chunks]](
        UnsafePointer(to=chunk_args[0]), num_chunks)
    return pool_fence(pool)


# ============================================================================
# Attention phase C: local merge + quantize — no cross-rank gather
# ============================================================================


# ============================================================================
# MoE phase 1 — expert gate(w1) + up(w3) + SiLU + FWHT + quantize
# ============================================================================


def minimax_moe_phase1[
    intermediate: Int, hidden: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int, P: BurstThreadPool,
](
    act_i8: I8Ptr,
    act_scale: F32Ptr,
    routing: TopKResult[top_k],
    w1_base: Int, w1_stride: Int,
    w1_sc_base: Int, w1_sc_stride: Int,
    w1_cs_base: Int, w1_cs_stride: Int,
    w3_base: Int, w3_stride: Int,
    w3_sc_base: Int, w3_sc_stride: Int,
    w3_cs_base: Int, w3_cs_stride: Int,
    expert_qi: I8Ptr,
    expert_blk_scale: F32Ptr,
    rank: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Multi-expert phase 1: gate(w1) + up(w3) + SiLU + FWHT + per-block i8.

    Filters routing.indices to experts owned by this rank, then builds
    N-tile-sharded FusedW1W3SiluArgs jobs across all local experts.
    """
    comptime n_tiles = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp

    var pool_capacity = pool.get_capacity()
    var expert_base = rank * experts_per_rank

    var local_count = 0
    var local_slots = InlineArray[Int, top_k](fill=0)
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid >= expert_base and eid < expert_base + experts_per_rank:
            local_slots[local_count] = s
            local_count += 1

    if local_count == 0:
        return PoolFence[P].completed()

    var workers_per_expert = pool_capacity // local_count
    if workers_per_expert < 1:
        workers_per_expert = 1
    if workers_per_expert > n_tiles:
        workers_per_expert = n_tiles
    var tiles_per_worker = (n_tiles + workers_per_expert - 1) // workers_per_expert

    var jobs = InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs())

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var w1p = U8Ptr(unsafe_from_address=w1_base + local_idx * w1_stride)
        var w1s = F32Ptr(unsafe_from_address=w1_sc_base + local_idx * w1_sc_stride)
        var w1c = F32Ptr(unsafe_from_address=w1_cs_base + local_idx * w1_cs_stride)
        var w3p = U8Ptr(unsafe_from_address=w3_base + local_idx * w3_stride)
        var w3s = F32Ptr(unsafe_from_address=w3_sc_base + local_idx * w3_sc_stride)
        var w3c = F32Ptr(unsafe_from_address=w3_cs_base + local_idx * w3_cs_stride)
        var qi_out = expert_qi + li * intermediate
        var blk_out = expert_blk_scale + li * n_tiles

        for w in range(workers_per_expert):
            var tile_start = w * tiles_per_worker
            if tile_start >= n_tiles:
                break
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * fwht_blk
            var n_count = (tile_end - tile_start) * fwht_blk
            jobs[num_jobs] = FusedW1W3SiluArgs(
                act_i8, act_scale,
                w1p, w1s, w1c,
                w3p, w3s, w3c,
                qi_out + n_start,
                blk_out + tile_start,
                n_start, n_count, 1)
            num_jobs += 1

    pool.dispatch[FusedW1W3SiluArgs, fused_w1_w3_silu_worker[intermediate, hidden, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


# ============================================================================
# MoE phase 2 — expert down projection with routing weight
# ============================================================================


def minimax_moe_phase2[
    hidden: Int, intermediate: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int, P: BurstThreadPool,
](
    expert_qi: I8Ptr,
    expert_blk_scale: F32Ptr,
    routing: TopKResult[top_k],
    down_base: Int, down_stride: Int,
    down_sc_base: Int, down_sc_stride: Int,
    down_bcs_base: Int, down_bcs_stride: Int,
    expert_out_buf: BF16Ptr,
    rank: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Per-local-expert down GEMV with routing weight folded into output scale.

    N-tile sharded: each expert's hidden-dimension output is split across
    workers, matching the phase1 pattern.
    """
    comptime num_blocks = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp
    var pool_capacity = pool.get_capacity()
    var expert_base = rank * experts_per_rank

    var local_count = 0
    var local_slots = InlineArray[Int, top_k](fill=0)
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid >= expert_base and eid < expert_base + experts_per_rank:
            local_slots[local_count] = s
            local_count += 1

    if local_count == 0:
        return PoolFence[P].completed()

    var workers_per_expert = pool_capacity // local_count
    if workers_per_expert < 1:
        workers_per_expert = 1
    comptime max_n_workers = hidden // VNNI_N_STEP
    if workers_per_expert > max_n_workers:
        workers_per_expert = max_n_workers
    var n_per_worker = ((max_n_workers + workers_per_expert - 1) // workers_per_expert) * VNNI_N_STEP

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var act = expert_qi + li * intermediate
        var wpacked = U8Ptr(unsafe_from_address=down_base + local_idx * down_stride)
        var blk_scale = expert_blk_scale + li * num_blocks
        var wscale = F32Ptr(unsafe_from_address=down_sc_base + local_idx * down_sc_stride)
        var blk_colsum = F32Ptr(unsafe_from_address=down_bcs_base + local_idx * down_bcs_stride)
        var dst = expert_out_buf + li * hidden
        var weight = routing.weights[s]

        for w in range(workers_per_expert):
            var n_start = w * n_per_worker
            if n_start >= hidden:
                break
            var n_count = min(n_per_worker, hidden - n_start)
            jobs[num_jobs] = Int8GemvBlockedArgs(
                act, wpacked + n_start * intermediate,
                blk_scale, wscale + n_start,
                blk_colsum + n_start, dst + n_start,
                weight, n_count, hidden)
            num_jobs += 1

    pool.dispatch[Int8GemvBlockedArgs,
        int8_gemv_blocked_decode_worker[hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


