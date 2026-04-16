"""Pool-facing dispatch wrappers for every experimental3 kernel.

Centralizes all `BurstThreadPool` / `PoolFence` plumbing in one file so the
compute-path kernels (row math, workers, args structs) stay free of scheduler
noise. Each dispatcher:
  - Checks seq_len / work-item count (returns a completed fence for zero work).
  - Partitions work into jobs.
  - Fills an InlineArray of the compute file's Args struct.
  - Calls `pool.dispatch[ArgsT, worker_fn](...)`.
  - Returns a `PoolFence[P]` the caller joins on.

Compute files export Args structs + worker functions; this file orchestrates.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY

from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kv_cache import CACHE_WIDTH
from experimental_gemma.router import Gemma4TopKResult

from experimental3.kernels.gemm import (
    WorkerConfig, int8_gemv_worker,
    Int8GemvBlockedArgs, int8_gemv_blocked_worker, int8_gemv_blocked_wa_worker,
    FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker, fused_gu_gelu_tanh_worker_wa,
    GEMV_TILE,
    LmHeadArgs, lm_head_worker,
    LmHeadFlashArgs, lm_head_flash_worker, LmHeadCands, LmHeadCandPtr,
)
from experimental3.moe import (
    RouterTopkArgs, router_topk_kernel,
)
from experimental3.kernels.sliding_attention import (
    AttnGroupArgs, sliding_attn_group_kernel,
)
from experimental3.kernels.full_chunked_attention_fused import (
    CpAttnPrepArgs, cp_attn_prep_kernel,
    ChunkedAttnArgs, cp_chunked_attn_kernel, MAX_CHUNKS,
    MergeChunksArgs, merge_local_chunks_kernel,
    CpGatherArgs, cp_gather_kernel, MAX_CP_RANKS,
)
from experimental3.kernels.rmsnorm import (
    RmsNormFwhtQuantArgs, rmsnorm_fwht_quant_worker,
    RmsNormDualGammaFwhtArgs, rmsnorm_dual_gamma_fwht_quant_worker,
    RMSNormNoScaleArgs, rmsnorm_no_scale_kernel,
    RMSNormPerHeadArgs, rmsnorm_per_head_kernel,
    PostAttnNormArgs, post_attn_norm_kernel,
    ExpertSumArgs, expert_sum_kernel,
    DenseNormArgs, dense_norm_kernel,
    PostReduceArgs, post_reduce_kernel,
)


# ============================================================================
# Shared fence helper — every dispatch function returns one of these.
# ============================================================================


@always_inline
def pool_fence[P: BurstThreadPool](mut pool: P) -> PoolFence[P]:
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# int8_gemv — per-row weight + per-row act scale
# ============================================================================


def int8_gemv[N: Int, K: Int, P: BurstThreadPool](
    act_ptr: Int, wpacked_ptr: Int,
    colsum_ptr: Int, weight_scale_ptr: Int, dst_ptr: Int,
    seq_len: Int,
    act_scale_ptr: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch int8 GEMV: [seq_len, K] x [N, K]^T -> [seq_len, N] bf16.

    act_scale_ptr: f32[seq_len] per-row activation scales (absmax from quantize).
    Dequant per row: (raw - 128*colsum) * (act_scale[m]/127) * weight_scale[n].
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](
        fill=WorkerConfig(0, 0, 0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = WorkerConfig(
            act_ptr + start * K, wpacked_ptr, colsum_ptr,
            weight_scale_ptr, dst_ptr + start * N * 2,
            act_scale_ptr, start, end - start)

    pool.dispatch[WorkerConfig, int8_gemv_worker[N, K]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


# ============================================================================
# int8_gemv_blocked — per-K-block activation scales (down projection)
# ============================================================================


def int8_gemv_blocked[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: I8Ptr,
    wpacked: U8Ptr,
    blk_scale: F32Ptr,
    wscale: F32Ptr,
    blk_colsum: F32Ptr,
    dst: BF16Ptr,
    seq_len: Int,
    mut pool: P,
    output_scale: Float32 = Float32(1.0),
) -> PoolFence[P]:
    """Dispatch int8 GEMV with per-block activation scales."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime num_blocks = K // fwht_blk
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr(), Float32(0))
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)
    for i in range(num_jobs):
        var start = i * rows_per_job
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K,
            wpacked,
            blk_scale + start * num_blocks,
            wscale,
            blk_colsum,
            dst + start * N,
            output_scale)

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def int8_gemv_blocked_wa[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: I8Ptr, wpacked: U8Ptr, blk_scale: F32Ptr,
    wscale: F32Ptr, blk_colsum: F32Ptr, dst: BF16Ptr,
    seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dispatch workaround blocked GEMV for sub-VNNI_K_STEP block sizes."""
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    comptime num_blocks = K // fwht_blk

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr(), Float32(0))
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)
    for i in range(num_jobs):
        var start = i
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K, wpacked,
            blk_scale + start * num_blocks,
            wscale, blk_colsum,
            dst + start * N, Float32(1.0))

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_wa_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


# ============================================================================
# fused_gu_gelu_tanh — gate+up GEMV -> gelu -> [FWHT] -> per-block quantize
# ============================================================================


def fused_gu_gelu_tanh[intermediate: Int, K: Int, fwht_blk: Int,
                       P: BurstThreadPool, fwht: Bool = True](
    act_i8: I8Ptr,
    act_scale: F32Ptr,
    wpacked: U8Ptr,
    wscale: F32Ptr,
    wcolsum: F32Ptr,
    qi_out: I8Ptr,
    blk_scale: F32Ptr,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch gate_up GEMV + GELU-tanh + [FWHT] + per-block quantize.

    For seq_len=1 (decode): parallelizes across the output dimension N.
    For seq_len>1 (prompt): parallelizes across sequence rows.
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime n_tiles = intermediate // fwht_blk
    comptime num_blk_per_row = intermediate // fwht_blk

    var zero_args = FusedGuGeluTanhArgs(
        I8Ptr(), F32Ptr(), U8Ptr(), F32Ptr(), F32Ptr(),
        I8Ptr(), F32Ptr(), 0, 0, 0)
    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](fill=zero_args)

    if seq_len == 1:
        var num_workers = min(n_tiles, pool.get_capacity())
        var tiles_per_worker = (n_tiles + num_workers - 1) // num_workers
        for i in range(num_workers):
            var tile_start = i * tiles_per_worker
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * fwht_blk
            var n_count = (tile_end - tile_start) * fwht_blk
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8, act_scale,
                wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_scale + tile_start,
                n_start, n_count, 1)
        pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk, fwht]](
            UnsafePointer(to=jobs[0]), num_workers)
    else:
        var num_workers = min(seq_len, pool.get_capacity())
        var rows_per_worker = (seq_len + num_workers - 1) // num_workers
        for i in range(num_workers):
            var row_start = i * rows_per_worker
            var row_count = min(rows_per_worker, seq_len - row_start)
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8 + row_start * K,
                act_scale + row_start,
                wpacked, wscale, wcolsum,
                qi_out + row_start * intermediate,
                blk_scale + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk, fwht]](
            UnsafePointer(to=jobs[0]), num_workers)

    return pool_fence(pool)


def fused_gu_gelu_tanh_wa[intermediate: Int, K: Int, fwht_blk: Int,
                          P: BurstThreadPool](
    act_i8: I8Ptr, act_scale: F32Ptr,
    wpacked: U8Ptr, wscale: F32Ptr, wcolsum: F32Ptr,
    qi_out: I8Ptr, blk_scale: F32Ptr,
    seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dispatch workaround fused gate_up + GELU-tanh + FWHT(sub-block) + quantize."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime n_tiles = intermediate // GEMV_TILE
    comptime num_blk_per_row = intermediate // fwht_blk

    var zero_args = FusedGuGeluTanhArgs(
        I8Ptr(), F32Ptr(), U8Ptr(), F32Ptr(), F32Ptr(),
        I8Ptr(), F32Ptr(), 0, 0, 0)
    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](fill=zero_args)

    if seq_len == 1:
        var num_workers = min(n_tiles, pool.get_capacity())
        var tiles_per_worker = (n_tiles + num_workers - 1) // num_workers
        for i in range(num_workers):
            var tile_start = i * tiles_per_worker
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * GEMV_TILE
            var n_count = (tile_end - tile_start) * GEMV_TILE
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8, act_scale, wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_scale + (n_start // fwht_blk),
                n_start, n_count, 1)
        pool.dispatch[FusedGuGeluTanhArgs,
            fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)
    else:
        var num_workers = min(seq_len, pool.get_capacity())
        var rows_per_worker = (seq_len + num_workers - 1) // num_workers
        for i in range(num_workers):
            var row_start = i * rows_per_worker
            var row_count = min(rows_per_worker, seq_len - row_start)
            jobs[i] = FusedGuGeluTanhArgs(
                act_i8 + row_start * K, act_scale + row_start,
                wpacked, wscale, wcolsum,
                qi_out + row_start * intermediate,
                blk_scale + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs,
            fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)

    return pool_fence(pool)


# ============================================================================
# lm_head dispatchers — N-parallel decode (seq_len=1)
# ============================================================================


def lm_head_gemv[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: Int,
    weight: Int,
    act_blk_scales: Int,
    w_blk_scales: Int,
    w_blk_colsums: Int,
    dst: Int,
    mut pool: P,
) -> PoolFence[P]:
    """LM head GEMV: i8 [N, K] x i8 [K] -> bf16 [N] with per-block dequant.

    Decode only (seq_len=1). Work is N-split across pool workers.
    """
    var num_workers = min(N, pool.get_capacity())
    var rows_per_worker = (N + num_workers - 1) // num_workers

    var zero_args = LmHeadArgs(0, 0, 0, 0, 0, 0, 0, 0)
    var jobs = InlineArray[LmHeadArgs, MAX_POOL_CAPACITY](fill=zero_args)
    var actual_jobs = 0
    for i in range(num_workers):
        var n_start = i * rows_per_worker
        if n_start >= N:
            break
        var n_count = min(rows_per_worker, N - n_start)
        jobs[i] = LmHeadArgs(
            act, weight, act_blk_scales, w_blk_scales, w_blk_colsums, dst,
            n_start, n_count)
        actual_jobs += 1

    pool.dispatch[LmHeadArgs, lm_head_worker[K, fwht_blk]](
        UnsafePointer(to=jobs[0]), actual_jobs)
    return pool_fence(pool)


def lm_head_flash[N: Int, K: Int, fwht_blk: Int, P: BurstThreadPool](
    act: I8Ptr,
    weight: I8Ptr,
    act_blk_scales: F32Ptr,
    w_blk_scales: F32Ptr,
    w_blk_colsums: F32Ptr,
    candidates: LmHeadCandPtr,
    rng_key: UInt64,
    softcap_val: Float32,
    mut pool: P,
) -> PoolFence[P]:
    """Fused LM-head GEMV + Gumbel-Max sample.

    `candidates` is a single LmHeadCandidates[MAX_POOL_CAPACITY] struct; each
    worker writes its result into scores[worker_idx] and indices[worker_idx].
    After the returned fence joins, call lm_head_flash_reduce(candidates) for
    the winner.
    """
    var num_workers = min(N, pool.get_capacity())
    var rows_per_worker = (N + num_workers - 1) // num_workers

    var jobs = InlineArray[LmHeadFlashArgs, MAX_POOL_CAPACITY](uninitialized=True)
    var actual_jobs = 0
    for i in range(num_workers):
        var n_start = i * rows_per_worker
        if n_start >= N:
            break
        var n_count = min(rows_per_worker, N - n_start)
        jobs[i] = LmHeadFlashArgs(
            act, weight, act_blk_scales, w_blk_scales, w_blk_colsums,
            candidates, n_start, n_count, rng_key, UInt64(i), softcap_val)
        actual_jobs += 1

    # Sentinel only the tail that no worker will write, so the reducer's SIMD
    # sweep of the full buffer still sees valid -inf / -1 in unused lanes.
    for i in range(actual_jobs, MAX_POOL_CAPACITY):
        candidates[].scores[i] = Float32(-1.0e30)
        candidates[].indices[i] = Int32(-1)

    pool.dispatch[LmHeadFlashArgs, lm_head_flash_worker[K, fwht_blk]](
        UnsafePointer(to=jobs[0]), actual_jobs)
    return pool_fence(pool)


# ============================================================================
# MoE — phase 1, phase 2, router top-k
# ============================================================================


def gemma4_moe_phase1[
    intermediate: Int, hidden: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int, P: BurstThreadPool,
](
    act_i8: I8Ptr,
    act_scale: F32Ptr,
    routing: Gemma4TopKResult[top_k],
    gate_up_base: Int,
    gate_up_stride: Int,
    gate_up_sc_base: Int,
    gate_up_sc_stride: Int,
    gate_up_cs_base: Int,
    gate_up_cs_stride: Int,
    expert_qi: I8Ptr,
    expert_blk_scale: F32Ptr,
    rank: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Multi-expert phase 1: gate_up + gelu_tanh + FWHT + per-block i8 quantize.

    Filters routing.indices to experts owned by this rank (block sharding:
    expert e on rank e // EPR, where EPR = num_experts // tp), then builds
    N-tile-sharded FusedGuGeluTanhArgs jobs across all local experts in a
    single pool.dispatch.
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

    var zero_args = FusedGuGeluTanhArgs(
        I8Ptr(), F32Ptr(), U8Ptr(), F32Ptr(), F32Ptr(),
        I8Ptr(), F32Ptr(), 0, 0, 0)
    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](fill=zero_args)

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var wpacked = U8Ptr(unsafe_from_address=gate_up_base + local_idx * gate_up_stride)
        var wscale = F32Ptr(unsafe_from_address=gate_up_sc_base + local_idx * gate_up_sc_stride)
        var wcolsum = F32Ptr(unsafe_from_address=gate_up_cs_base + local_idx * gate_up_cs_stride)
        var qi_out = expert_qi + li * intermediate
        var blk_out = expert_blk_scale + li * n_tiles

        for w in range(workers_per_expert):
            var tile_start = w * tiles_per_worker
            if tile_start >= n_tiles:
                break
            var tile_end = tile_start + tiles_per_worker
            if tile_end > n_tiles:
                tile_end = n_tiles
            var n_start = tile_start * fwht_blk
            var n_count = (tile_end - tile_start) * fwht_blk
            jobs[num_jobs] = FusedGuGeluTanhArgs(
                act_i8, act_scale,
                wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_out + tile_start,
                n_start, n_count, 1)
            num_jobs += 1

    pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, hidden, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def gemma4_moe_phase2[
    hidden: Int, intermediate: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int, P: BurstThreadPool,
](
    expert_qi: I8Ptr,
    expert_blk_scale: F32Ptr,
    routing: Gemma4TopKResult[top_k],
    down_base: Int,
    down_stride: Int,
    down_sc_base: Int,
    down_sc_stride: Int,
    down_bcs_base: Int,
    down_bcs_stride: Int,
    expert_out_buf: BF16Ptr,
    rank: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Multi-expert phase 2: per-local-expert down GEMV with routing weight.

    One job per local expert: int8_gemv_blocked_worker reading the expert's
    expert_qi slot (written by phase 1), with output_scale = routing.weights[s]
    so the routing scale is folded into the bf16 cast for free.
    """
    comptime num_blocks = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp
    var expert_base = rank * experts_per_rank

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr(), Float32(0))
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)

    var local_count = 0
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid < expert_base or eid >= expert_base + experts_per_rank:
            continue
        var local_idx = eid - expert_base

        jobs[local_count] = Int8GemvBlockedArgs(
            expert_qi + local_count * intermediate,
            U8Ptr(unsafe_from_address=down_base + local_idx * down_stride),
            expert_blk_scale + local_count * num_blocks,
            F32Ptr(unsafe_from_address=down_sc_base + local_idx * down_sc_stride),
            F32Ptr(unsafe_from_address=down_bcs_base + local_idx * down_bcs_stride),
            expert_out_buf + local_count * hidden,
            routing.weights[s],
        )
        local_count += 1

    if local_count == 0:
        return PoolFence[P].completed()

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_worker[hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), local_count)
    return pool_fence(pool)


def router_topk_dispatch[num_experts: Int, k: Int, P: BurstThreadPool](
    logits: BF16Ptr, per_expert_scale: BF16Ptr, result_ptr: Int, mut pool: P,
) -> PoolFence[P]:
    var args = RouterTopkArgs(logits, per_expert_scale, result_ptr)
    pool.dispatch[RouterTopkArgs, router_topk_kernel[num_experts, k]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


# ============================================================================
# Sliding attention — per-KV-group worker
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
    return pool_fence(pool)


# ============================================================================
# Context-parallel full attention — prep, chunked score, merge, gather
# ============================================================================


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
    return pool_fence(pool)


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
    return pool_fence(pool)


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
    return pool_fence(pool)


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
    return pool_fence(pool)


# ============================================================================
# RMSNorm family — FWHT+quantize, dual-gamma, bf16-out, post-*norm composites
# ============================================================================


def rmsnorm_fwht_quant[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Unified FWHT+quantize dispatcher. Use convenience wrappers below."""
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormFwhtQuantArgs, MAX_POOL_CAPACITY](
        fill=RmsNormFwhtQuantArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtQuantArgs(
            in_ptr, gamma_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, eps, start, end)

    pool.dispatch[RmsNormFwhtQuantArgs,
        rmsnorm_fwht_quant_worker[cols, block, has_gamma, per_block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def rmsnorm_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm * gamma + FWHT + per-row i8."""
    return rmsnorm_fwht_quant[cols, block, True, False, P](
        in_ptr, gamma_ptr, qi_ptr, work_ptr, scale_ptr, eps, seq_len, pool)


def rmsnorm_gamma_fwht_per_block_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    blk_scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm * gamma + FWHT + per-block i8."""
    return rmsnorm_fwht_quant[cols, block, True, True, P](
        in_ptr, gamma_ptr, qi_ptr, work_ptr, blk_scale_ptr, eps, seq_len, pool)


def rmsnorm_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int, eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """RMSNorm + FWHT + per-row i8. No gamma."""
    return rmsnorm_fwht_quant[cols, block, False, False, P](
        in_ptr, 0, qi_ptr, work_ptr, scale_ptr, eps, seq_len, pool)


def rmsnorm_dual_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int,
    gamma_a_ptr: Int, gamma_b_ptr: Int,
    qi_a_ptr: Int, qi_b_ptr: Int,
    work_a_ptr: Int, work_b_ptr: Int,
    scale_a_ptr: Int, scale_b_ptr: Int,
    eps: Float32, seq_len: Int, mut pool: P,
) -> PoolFence[P]:
    """Dual-gamma RMSNorm + FWHT + per-row i8. One pass, two outputs."""
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormDualGammaFwhtArgs, MAX_POOL_CAPACITY](
        fill=RmsNormDualGammaFwhtArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormDualGammaFwhtArgs(
            in_ptr, gamma_a_ptr, gamma_b_ptr,
            qi_a_ptr, qi_b_ptr,
            work_a_ptr + i * cols * size_of[Float32](),
            work_b_ptr + i * cols * size_of[Float32](),
            scale_a_ptr, scale_b_ptr,
            eps, start, end)

    pool.dispatch[RmsNormDualGammaFwhtArgs,
        rmsnorm_dual_gamma_fwht_quant_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def rmsnorm_no_scale[InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P]:
    """RMSNorm without learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_no_scale: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm_no_scale: cols must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = input.as_ptr[DType.bfloat16]()
    var op = output.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[RMSNormNoScaleArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormNoScaleArgs(ip, op, start, end, eps)

    pool.dispatch[RMSNormNoScaleArgs, rmsnorm_no_scale_kernel[InT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def rmsnorm_per_head[head_dim: Int, num_heads: Int,
    W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P] where W.DTYPE == DType.bfloat16:
    """Per-head RMSNorm with learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_per_head: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_per_head: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_per_head: input/output cols mismatch"
    comptime assert InT.COLS == head_dim * num_heads, "rmsnorm_per_head: cols != heads * dim"
    comptime assert W.ROWS * W.COLS == head_dim, "rmsnorm_per_head: weight size != head_dim"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "rmsnorm_per_head: head_dim must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = input.as_ptr[DType.bfloat16]()
    var wp = weight.as_ptr[DType.bfloat16]()
    var op = output.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[RMSNormPerHeadArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormPerHeadArgs(ip, wp, op, start, end, eps)

    pool.dispatch[RMSNormPerHeadArgs, rmsnorm_per_head_kernel[head_dim, num_heads]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return pool_fence(pool)


def post_attn_norm_dispatch[hidden: Int, P: BurstThreadPool](
    src_ptr: Int, norm_w_ptr: Int, x_main_ptr: Int, eps: Float32, mut pool: P,
) -> PoolFence[P]:
    var args = PostAttnNormArgs(src_ptr, norm_w_ptr, x_main_ptr, eps)
    pool.dispatch[PostAttnNormArgs, post_attn_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def expert_sum_dispatch[hidden: Int, P: BurstThreadPool](
    expert_out_ptr: Int, local_count: Int, dst_ptr: Int, mut pool: P,
) -> PoolFence[P]:
    var args = ExpertSumArgs(expert_out_ptr, local_count, dst_ptr)
    pool.dispatch[ExpertSumArgs, expert_sum_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def dense_norm_dispatch[hidden: Int, P: BurstThreadPool](
    src_ptr: Int, norm_w_ptr: Int, dst_ptr: Int, eps: Float32, mut pool: P,
) -> PoolFence[P]:
    var args = DenseNormArgs(src_ptr, norm_w_ptr, dst_ptr, eps)
    pool.dispatch[DenseNormArgs, dense_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def post_reduce_dispatch[hidden: Int, P: BurstThreadPool](
    moe_out_ptr: Int, moe_norm_w_ptr: Int, dense_normed_ptr: Int,
    combine_norm_w_ptr: Int, x_main_ptr: Int,
    layer_scalar: Float32, eps: Float32, mut pool: P,
) -> PoolFence[P]:
    var args = PostReduceArgs(moe_out_ptr, moe_norm_w_ptr, dense_normed_ptr,
        combine_norm_w_ptr, x_main_ptr, layer_scalar, eps)
    pool.dispatch[PostReduceArgs, post_reduce_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)
