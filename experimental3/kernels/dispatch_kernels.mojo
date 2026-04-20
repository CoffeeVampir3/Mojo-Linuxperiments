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
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY

from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kv_cache import CACHE_WIDTH
from experimental_gemma.router import Gemma4TopKResult

from experimental3.kernels.dispatch_args import (
    WorkerConfig,
    Int8GemvBlockedArgs,
    FusedGuGeluTanhArgs,
    LmHeadArgs,
    RouterTopkArgs,
    AttnGroupArgs,
    CpAttnPrepArgs,
    ChunkedAttnArgs,
    MergeChunksArgs,
    CpGatherArgs,
    MAX_CP_RANKS,
    RmsNormFwhtQuantArgs,
    RmsNormDualGammaFwhtArgs,
    RMSNormNoScaleArgs,
    RMSNormPerHeadArgs,
    PostAttnNormArgs,
    ExpertSumArgs,
    DenseNormArgs,
    PostReduceArgs,
)
from experimental3.kernels.gemm import (
    int8_gemv_worker, int8_gemv_decode_worker,
    int8_gemv_blocked_worker, int8_gemv_blocked_decode_worker,
    int8_gemv_blocked_wa_worker,
    fused_gu_gelu_tanh_worker, fused_gu_gelu_tanh_worker_wa,
    GEMV_TILE,
    lm_head_worker,
)
from kernels.vnni import VNNI_N_STEP
from experimental3.moe import router_topk_kernel
from experimental3.kernels.sliding_attention import sliding_attn_group_kernel
from experimental3.kernels.full_chunked_attention_fused import (
    cp_attn_prep_kernel,
    cp_chunked_attn_kernel, MAX_CHUNKS,
    merge_local_chunks_kernel,
    cp_gather_kernel,
)
from experimental3.kernels.rmsnorm import (
    rmsnorm_fwht_quant_worker,
    rmsnorm_dual_gamma_fwht_quant_worker,
    rmsnorm_no_scale_kernel,
    rmsnorm_per_head_kernel,
    post_attn_norm_kernel,
    expert_sum_kernel,
    dense_norm_kernel,
    post_reduce_kernel,
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

    Decode (seq_len=1): N-split across workers. Each worker computes a
    sub-range of output elements from the same activation row.
    Prefill (seq_len>1): M-split across workers. Each worker computes
    full N outputs for a subset of rows.
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    var act = I8Ptr(unsafe_from_address=act_ptr)
    var wpacked = U8Ptr(unsafe_from_address=wpacked_ptr)
    var colsum = F32Ptr(unsafe_from_address=colsum_ptr)
    var weight_scale = F32Ptr(unsafe_from_address=weight_scale_ptr)
    var dst = BF16Ptr(unsafe_from_address=dst_ptr)
    var act_scale = F32Ptr(unsafe_from_address=act_scale_ptr)

    if seq_len == 1:
        var num_workers = min(N // VNNI_N_STEP, pool.get_capacity())
        var n_per_worker = ((N // VNNI_N_STEP + num_workers - 1) // num_workers) * VNNI_N_STEP
        var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](fill=WorkerConfig())
        var actual = 0
        for i in range(num_workers):
            var n_start = i * n_per_worker
            if n_start >= N:
                break
            var n_count = min(n_per_worker, N - n_start)
            jobs[actual] = WorkerConfig(
                act, wpacked + n_start * K, colsum + n_start,
                weight_scale + n_start, dst + n_start,
                act_scale, 0, n_count)
            actual += 1
        pool.dispatch[WorkerConfig, int8_gemv_decode_worker[N, K]](
            UnsafePointer(to=jobs[0]), actual)
        return pool_fence(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
    var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](fill=WorkerConfig())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = WorkerConfig(
            act + start * K, wpacked, colsum,
            weight_scale, dst + start * N,
            act_scale, start, end - start)
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
    """Dispatch int8 GEMV with per-block activation scales.

    Decode (seq_len=1): N-split across workers.
    Prefill (seq_len>1): M-split across workers.
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime num_blocks = K // fwht_blk

    if seq_len == 1:
        var num_workers = min(N // VNNI_N_STEP, pool.get_capacity())
        var n_per_worker = ((N // VNNI_N_STEP + num_workers - 1) // num_workers) * VNNI_N_STEP
        var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
            fill=Int8GemvBlockedArgs())
        var actual = 0
        for i in range(num_workers):
            var n_start = i * n_per_worker
            if n_start >= N:
                break
            var n_count = min(n_per_worker, N - n_start)
            jobs[actual] = Int8GemvBlockedArgs(
                act, wpacked + n_start * K, blk_scale,
                wscale + n_start, blk_colsum + n_start,
                dst + n_start, output_scale, n_count, N)
            actual += 1
        pool.dispatch[Int8GemvBlockedArgs,
            int8_gemv_blocked_decode_worker[N, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), actual)
        return pool_fence(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K, wpacked,
            blk_scale + start * num_blocks, wscale, blk_colsum,
            dst + start * N, output_scale, N, N)
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

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())
    for i in range(num_jobs):
        var start = i
        jobs[i] = Int8GemvBlockedArgs(
            act + start * K, wpacked,
            blk_scale + start * num_blocks,
            wscale, blk_colsum,
            dst + start * N, Float32(1.0), N, N)

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

    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](
        fill=FusedGuGeluTanhArgs())

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

    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](
        fill=FusedGuGeluTanhArgs())

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

    var act_ptr = I8Ptr(unsafe_from_address=act)
    var weight_ptr = I8Ptr(unsafe_from_address=weight)
    var act_blk_scales_ptr = F32Ptr(unsafe_from_address=act_blk_scales)
    var w_blk_scales_ptr = F32Ptr(unsafe_from_address=w_blk_scales)
    var w_blk_colsums_ptr = F32Ptr(unsafe_from_address=w_blk_colsums)
    var dst_ptr = BF16Ptr(unsafe_from_address=dst)

    var jobs = InlineArray[LmHeadArgs, MAX_POOL_CAPACITY](fill=LmHeadArgs())
    var actual_jobs = 0
    for i in range(num_workers):
        var n_start = i * rows_per_worker
        if n_start >= N:
            break
        var n_count = min(rows_per_worker, N - n_start)
        jobs[i] = LmHeadArgs(
            act_ptr, weight_ptr, act_blk_scales_ptr,
            w_blk_scales_ptr, w_blk_colsums_ptr, dst_ptr,
            n_start, n_count)
        actual_jobs += 1

    pool.dispatch[LmHeadArgs, lm_head_worker[K, fwht_blk]](
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

    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](
        fill=FusedGuGeluTanhArgs())

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

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())

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
            routing.weights[s], hidden, hidden,
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
    var args = RouterTopkArgs(
        logits, per_expert_scale, U8Ptr(unsafe_from_address=result_ptr))
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
    var q = BF16Ptr(unsafe_from_address=q_base)
    var k = BF16Ptr(unsafe_from_address=k_base)
    var v = BF16Ptr(unsafe_from_address=v_base)
    var q_norm = BF16Ptr(unsafe_from_address=q_norm_ptr)
    var k_norm = BF16Ptr(unsafe_from_address=k_norm_ptr)
    var cos = F32Ptr(unsafe_from_address=cos_ptr)
    var sin = F32Ptr(unsafe_from_address=sin_ptr)
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var qi_out = I8Ptr(unsafe_from_address=qi_out_ptr)
    var head_scale = F32Ptr(unsafe_from_address=head_scale_ptr)
    for g in range(NKV):
        jobs[g] = AttnGroupArgs(
            q + g * heads_per_group * head_dim,
            k + g * head_dim,
            v + g * head_dim,
            q_norm, k_norm,
            cos, sin,
            cache, g,
            cache_pos, context_len,
            qi_out + g * heads_per_group * head_dim,
            head_scale + g * heads_per_group,
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
        q_bf16_base=BF16Ptr(unsafe_from_address=q_bf16_base),
        k_bf16_ptr=BF16Ptr(unsafe_from_address=k_bf16_ptr),
        q_norm_ptr=BF16Ptr(unsafe_from_address=q_norm_ptr),
        k_norm_ptr=BF16Ptr(unsafe_from_address=k_norm_ptr),
        cos_ptr=F32Ptr(unsafe_from_address=cos_ptr),
        sin_ptr=F32Ptr(unsafe_from_address=sin_ptr),
        cache_base=U8Ptr(unsafe_from_address=cache_base),
        cache_pos=local_pos, kv_head=kv_head,
        eps=eps,
        q_i8_out=I8Ptr(unsafe_from_address=q_i8_out),
        qi_biases_out=F32Ptr(unsafe_from_address=qi_biases_out),
        q_scales_out=F32Ptr(unsafe_from_address=q_scales_out),
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
    var q_i8 = I8Ptr(unsafe_from_address=q_i8_base)
    var qi_biases = F32Ptr(unsafe_from_address=qi_biases_base)
    var q_scales = F32Ptr(unsafe_from_address=q_scales_base)
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var partial_out = F32Ptr(unsafe_from_address=partial_out_base)
    var chunk_args = InlineArray[ChunkedAttnArgs, MAX_CHUNKS](fill=ChunkedAttnArgs())
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
        partial_base=F32Ptr(unsafe_from_address=partial_base),
        num_chunks=num_chunks,
        out_m=F32Ptr(unsafe_from_address=out_m),
        out_l=F32Ptr(unsafe_from_address=out_l),
        out_v=F32Ptr(unsafe_from_address=out_v))
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
    var all_m_ptrs = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())
    var all_l_ptrs = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())
    var all_v_ptrs = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())
    for r in range(MAX_CP_RANKS):
        if all_m[r] != 0:
            all_m_ptrs[r] = F32Ptr(unsafe_from_address=all_m[r])
        if all_l[r] != 0:
            all_l_ptrs[r] = F32Ptr(unsafe_from_address=all_l[r])
        if all_v[r] != 0:
            all_v_ptrs[r] = F32Ptr(unsafe_from_address=all_v[r])
    var args = CpGatherArgs(
        rank=rank, head_start=head_start, head_count=head_count,
        qi_out=I8Ptr(unsafe_from_address=qi_out),
        head_scales=F32Ptr(unsafe_from_address=head_scales),
        all_m=all_m_ptrs, all_l=all_l_ptrs, all_v=all_v_ptrs)
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

    var inp = BF16Ptr(unsafe_from_address=in_ptr)
    var gamma = BF16Ptr()
    if gamma_ptr != 0:
        gamma = BF16Ptr(unsafe_from_address=gamma_ptr)
    var qi = I8Ptr(unsafe_from_address=qi_ptr)
    var work = F32Ptr(unsafe_from_address=work_ptr)
    var scales = F32Ptr(unsafe_from_address=scale_ptr)
    var jobs = InlineArray[RmsNormFwhtQuantArgs, MAX_POOL_CAPACITY](
        fill=RmsNormFwhtQuantArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtQuantArgs(
            inp, gamma, qi,
            work + i * cols,
            scales, eps, start, end)

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

    var inp = BF16Ptr(unsafe_from_address=in_ptr)
    var gamma_a = BF16Ptr(unsafe_from_address=gamma_a_ptr)
    var gamma_b = BF16Ptr(unsafe_from_address=gamma_b_ptr)
    var qi_a = I8Ptr(unsafe_from_address=qi_a_ptr)
    var qi_b = I8Ptr(unsafe_from_address=qi_b_ptr)
    var work_a = F32Ptr(unsafe_from_address=work_a_ptr)
    var work_b = F32Ptr(unsafe_from_address=work_b_ptr)
    var scale_a = F32Ptr(unsafe_from_address=scale_a_ptr)
    var scale_b = F32Ptr(unsafe_from_address=scale_b_ptr)
    var jobs = InlineArray[RmsNormDualGammaFwhtArgs, MAX_POOL_CAPACITY](
        fill=RmsNormDualGammaFwhtArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormDualGammaFwhtArgs(
            inp, gamma_a, gamma_b,
            qi_a, qi_b,
            work_a + i * cols,
            work_b + i * cols,
            scale_a, scale_b,
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
    var jobs = InlineArray[RMSNormNoScaleArgs, MAX_POOL_CAPACITY](
        fill=RMSNormNoScaleArgs())
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
    var jobs = InlineArray[RMSNormPerHeadArgs, MAX_POOL_CAPACITY](
        fill=RMSNormPerHeadArgs())
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
    var args = PostAttnNormArgs(
        BF16Ptr(unsafe_from_address=src_ptr),
        BF16Ptr(unsafe_from_address=norm_w_ptr),
        BF16Ptr(unsafe_from_address=x_main_ptr),
        eps)
    pool.dispatch[PostAttnNormArgs, post_attn_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def expert_sum_dispatch[hidden: Int, P: BurstThreadPool](
    expert_out_ptr: Int, local_count: Int, dst_ptr: Int, mut pool: P,
) -> PoolFence[P]:
    var args = ExpertSumArgs(
        BF16Ptr(unsafe_from_address=expert_out_ptr),
        local_count,
        BF16Ptr(unsafe_from_address=dst_ptr))
    pool.dispatch[ExpertSumArgs, expert_sum_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def dense_norm_dispatch[hidden: Int, P: BurstThreadPool](
    src_ptr: Int, norm_w_ptr: Int, dst_ptr: Int, eps: Float32, mut pool: P,
) -> PoolFence[P]:
    var args = DenseNormArgs(
        BF16Ptr(unsafe_from_address=src_ptr),
        BF16Ptr(unsafe_from_address=norm_w_ptr),
        BF16Ptr(unsafe_from_address=dst_ptr),
        eps)
    pool.dispatch[DenseNormArgs, dense_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)


def post_reduce_dispatch[hidden: Int, P: BurstThreadPool](
    moe_out_ptr: Int, moe_norm_w_ptr: Int, dense_normed_ptr: Int,
    combine_norm_w_ptr: Int, x_main_ptr: Int,
    layer_scalar: Float32, eps: Float32, mut pool: P,
) -> PoolFence[P]:
    var args = PostReduceArgs(
        BF16Ptr(unsafe_from_address=moe_out_ptr),
        BF16Ptr(unsafe_from_address=moe_norm_w_ptr),
        BF16Ptr(unsafe_from_address=dense_normed_ptr),
        BF16Ptr(unsafe_from_address=combine_norm_w_ptr),
        BF16Ptr(unsafe_from_address=x_main_ptr),
        layer_scalar, eps)
    pool.dispatch[PostReduceArgs, post_reduce_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return pool_fence(pool)
