from std.memory import UnsafePointer
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from modeling.model_spec import StaticTensor, DynamicTensor
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kernels.dispatch_args import Int8GemvBlockedArgs, ChunkedAttnArgs
from experimental3.kernels.gemm import int8_gemv_blocked_decode_worker
from kernels.vnni import VNNI_N_STEP
from experimental3.kv_cache import CACHE_WIDTH
from minimax.kernels.amx_attention import amx_chunked_attn_kernel

from minimax.kernels.dispatch_args import (
    FusedW1W3SiluArgs,
    SparseRoute,
    SparseMoePhase1Args,
    SparseMoePhase2Args,
    KVWriteBatchArgs,
    QPrepBatchArgs,
    PrefillAttnArgs,
    RouterCandidate,
    TopKResult,
    RouterFusedArgs,
    ChunkedScoreMultiArgs,
)
from minimax.kernels.gemm import (
    fused_w1_w3_silu_worker, sparse_moe_phase1_worker,
    sparse_moe_phase2_worker,
)
from minimax.kernels.attention import (
    kv_write_batch_kernel, q_prep_batch_kernel,
    prefill_attn_persistent_worker,
)
from minimax.kernels.router import router_fused_worker


# ============================================================================
# Fused router — centered bf16 GEMV + gauge pivot + local top-K (phase 1)
# ============================================================================


def router_fused_dispatch[
    ActT: DynamicTensor, WT: StaticTensor, GT: StaticTensor, BT: StaticTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    num_experts: Int, hidden: Int, k: Int,
](
    act_bf16: ActT,
    weight_bf16: WT,
    gauge_bf16: GT,
    bias_f32: BT,
    candidates: UnsafePointer[RouterCandidate, MutAnyOrigin],
    ref [origin] pool: P,
    expert_base: Int = 0,
) -> PoolFence[P, origin]:
    comptime assert ActT.DTYPE == DType.bfloat16, "router_fused: act must be bf16"
    comptime assert WT.DTYPE == DType.bfloat16, "router_fused: weight must be bf16"
    comptime assert GT.DTYPE == DType.bfloat16, "router_fused: gauge must be bf16"
    comptime assert BT.DTYPE == DType.float32, "router_fused: bias must be f32"

    var seq_len = act_bf16.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_p = act_bf16.as_ptr[DType.bfloat16]()
    var weight_p = weight_bf16.as_ptr[DType.bfloat16]()
    var gauge_p = gauge_bf16.as_ptr[DType.bfloat16]()
    var bias_p = bias_f32.as_ptr[DType.float32]()

    var jobs = InlineArray[RouterFusedArgs, MAX_POOL_CAPACITY](
        fill=RouterFusedArgs())
    var actual = 0

    var num_workers = min(seq_len, pool.get_capacity())
    var rows_per_worker = (seq_len + num_workers - 1) // num_workers
    for i in range(num_workers):
        var start_row = i * rows_per_worker
        if start_row >= seq_len:
            break
        var row_count = min(rows_per_worker, seq_len - start_row)
        jobs[actual] = RouterFusedArgs(
            act_p + start_row * hidden, weight_p, gauge_p, bias_p,
            candidates + start_row * k, expert_base,
            0, num_experts, row_count, k)
        actual += 1

    pool.dispatch[RouterFusedArgs, router_fused_worker[hidden, k]](
        UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Attention phase A: K/V cache write — NKV_LOCAL parallel jobs
# ============================================================================


def kv_write_dispatch[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    max_seq: Int, num_kv_heads: Int, qkv_local: Int,
    q_local: Int, rope_half: Int,
](
    qkv_ptr: BF16Ptr,
    k_norm_ptr: BF16Ptr,
    cos_base: F32Ptr,
    sin_base: F32Ptr,
    inv_rms_k_arr: F32Ptr,
    cache_base: Int,
    start_pos: Int,
    pos_count: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var jobs = InlineArray[KVWriteBatchArgs, 8](fill=KVWriteBatchArgs())

    for g in range(num_kv_heads):
        jobs[g] = KVWriteBatchArgs(
            k_bf16_base=qkv_ptr + g * head_dim + q_local,
            v_bf16_base=qkv_ptr + g * head_dim + q_local + num_kv_heads * head_dim,
            qkv_row_stride=qkv_local,
            k_norm_ptr=k_norm_ptr + g * head_dim,
            cos_base=cos_base,
            sin_base=sin_base,
            rope_row_elems=rope_half,
            inv_rms_k_arr=inv_rms_k_arr,
            cache_base=cache,
            start_pos=start_pos,
            pos_count=pos_count,
            kv_head=g)
    pool.dispatch[KVWriteBatchArgs,
        kv_write_batch_kernel[head_dim, rope_dim, pair_stride, max_seq, num_kv_heads]](
        UnsafePointer(to=jobs[0]), num_kv_heads)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Q prep dispatch — one job per (KV head, rank), batch all positions
# ============================================================================


def q_prep_batch_dispatch[
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    heads_per_group: Int, num_kv_heads: Int,
    qkv_local: Int, rope_half: Int,
](
    qkv_ptr: BF16Ptr,
    q_norm_ptr: BF16Ptr,
    cos_base: F32Ptr,
    sin_base: F32Ptr,
    inv_rms_q_arr: F32Ptr,
    qi_out: I8Ptr,
    qi_biases_out: F32Ptr,
    q_scales_out: F32Ptr,
    start_pos: Int,
    pos_count: Int,
):
    for kv in range(num_kv_heads):
        var args = QPrepBatchArgs(
            q_bf16_base=qkv_ptr + kv * heads_per_group * head_dim,
            qkv_row_stride=qkv_local,
            q_norm_ptr=q_norm_ptr + kv * heads_per_group * head_dim,
            cos_base=cos_base,
            sin_base=sin_base,
            rope_row_elems=rope_half,
            inv_rms_q_arr=inv_rms_q_arr,
            qi_out=qi_out + kv * heads_per_group * pos_count * head_dim,
            qi_out_head_stride=pos_count * head_dim,
            qi_biases_out=qi_biases_out + kv * heads_per_group * pos_count,
            qi_biases_head_stride=pos_count,
            q_scales_out=q_scales_out + kv * heads_per_group * pos_count,
            q_scales_head_stride=pos_count,
            start_pos=start_pos,
            pos_count=pos_count,
            kv_head=kv)
        q_prep_batch_kernel[head_dim, rope_dim, pair_stride, heads_per_group](args)


comptime ATTN_PG_PER_WORKER = 32


@always_inline
def attn_chunk_count(
    context_len: Int, pool_capacity: Int, kv_count: Int, max_attn_chunks: Int,
) -> Int:
    """Number of page chunks per KV head.

    The score phase dispatches one physical job for every (KV head, chunk).
    Cap per-head chunks so the total job count fits the worker pool.
    """
    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var padded_pg = (num_pg + 3) & ~3
    if padded_pg <= 0 or kv_count <= 0:
        return 0
    var raw_chunks = (padded_pg + ATTN_PG_PER_WORKER - 1) // ATTN_PG_PER_WORKER
    var chunk_cap = max_attn_chunks
    chunk_cap = min(chunk_cap, pool_capacity // kv_count)
    chunk_cap = min(chunk_cap, padded_pg // 4)
    var capped = min(raw_chunks, chunk_cap)
    if capped < 1:
        return 1
    return capped


def chunked_score_multi_worker[
    head_dim: Int, max_seq: Int, num_kv_heads: Int,
    heads_per_group: Int, max_attn_chunks: Int,
](args: ChunkedScoreMultiArgs):
    comptime CHUNK_F32_STRIDE = heads_per_group * (2 + head_dim)
    comptime Q_STRIDE = heads_per_group * head_dim
    comptime QSCALE_STRIDE = heads_per_group

    var total_jobs = args.num_chunks * args.kv_count
    var job = args.worker_id
    while job < total_jobs:
        var kv_off = job // args.num_chunks
        var c = job - kv_off * args.num_chunks
        var chunk_blocks = args.blocks_per_chunk
        if c < args.extra_blocks:
            chunk_blocks += 1
        var block_start = c * args.blocks_per_chunk + min(c, args.extra_blocks)
        var start = block_start * 4
        var end = (block_start + chunk_blocks) * 4
        var chunk_args = ChunkedAttnArgs(
            q_i8_base=args.q_i8_base + kv_off * Q_STRIDE,
            qi_biases_base=args.qi_biases_base + kv_off * QSCALE_STRIDE,
            q_scales_base=args.q_scales_base + kv_off * QSCALE_STRIDE,
            cache_base=args.cache_base,
            kv_head=args.kv_start + kv_off,
            start_pg=start,
            end_pg=end,
            partial_out=args.partial_out_base + (
                kv_off * args.num_chunks + c) * CHUNK_F32_STRIDE,
            context_len=args.context_len)
        amx_chunked_attn_kernel[
            head_dim, max_seq, num_kv_heads, 0, heads_per_group,
            max_attn_chunks](chunk_args)
        job += args.num_workers


def chunked_score_dispatch_multi[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, max_attn_chunks: Int,
](
    q_i8_base: Int, qi_biases_base: Int, q_scales_base: Int,
    cache_base: Int, kv_start: Int, kv_count: Int,
    context_len: Int, pool_capacity: Int,
    partial_out_base: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Pack all (kv, chunk) work items into one pool dispatch + fence."""
    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var padded_pg = (num_pg + 3) & ~3
    var num_chunks = attn_chunk_count(
        context_len, pool_capacity, kv_count, max_attn_chunks)
    if num_chunks <= 0 or kv_count <= 0:
        return PoolFence[P, origin].over(pool)
    var total_jobs = num_chunks * kv_count
    var num_workers = total_jobs
    if num_workers <= 0:
        return PoolFence[P, origin].over(pool)
    var pg_blocks = padded_pg // 4
    var blocks_per_chunk = pg_blocks // num_chunks
    var extra_blocks = pg_blocks - blocks_per_chunk * num_chunks

    var worker_args = InlineArray[
        ChunkedScoreMultiArgs, MAX_POOL_CAPACITY](
        fill=ChunkedScoreMultiArgs())
    for i in range(num_workers):
        worker_args[i] = ChunkedScoreMultiArgs(
            q_i8_base=I8Ptr(unsafe_from_address=q_i8_base),
            qi_biases_base=F32Ptr(unsafe_from_address=qi_biases_base),
            q_scales_base=F32Ptr(unsafe_from_address=q_scales_base),
            cache_base=U8Ptr(unsafe_from_address=cache_base),
            partial_out_base=F32Ptr(unsafe_from_address=partial_out_base),
            kv_start=kv_start,
            kv_count=kv_count,
            context_len=context_len,
            num_chunks=num_chunks,
            blocks_per_chunk=blocks_per_chunk,
            extra_blocks=extra_blocks,
            worker_id=i,
            num_workers=num_workers)
    pool.dispatch[
        ChunkedScoreMultiArgs,
        chunked_score_multi_worker[
            head_dim, max_seq, num_kv_heads, heads_per_group,
            max_attn_chunks]](
        UnsafePointer(to=worker_args[0]), num_workers)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Prefill attention dispatch — bounded persistent workers across tiles
# ============================================================================


def prefill_attn_dispatch[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    q_local: Int, heads_per_rank: Int,
](
    q_i8_base: I8Ptr,
    qi_biases_base: F32Ptr,
    q_scales_base: F32Ptr,
    cache_base: Int,
    start_pos: Int,
    seq_len: Int,
    qi_out: I8Ptr,
    head_sc_out: F32Ptr,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime Q_TILE = 16
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var num_q_tiles = (seq_len + Q_TILE - 1) // Q_TILE
    var total_work = num_kv_heads * num_q_tiles
    var num_workers = min(pool.get_capacity(), total_work)
    if num_workers <= 0:
        return PoolFence[P, origin].over(pool)

    var jobs = InlineArray[PrefillAttnArgs, MAX_POOL_CAPACITY](
        fill=PrefillAttnArgs())
    for worker in range(num_workers):
        jobs[worker] = PrefillAttnArgs(
            q_i8_base=q_i8_base,
            qi_biases_base=qi_biases_base,
            q_factors_base=q_scales_base,
            cache_base=cache,
            start_pos=start_pos,
            seq_len=seq_len,
            qi_out_base=qi_out,
            qi_out_row_stride=q_local,
            head_sc_out_base=head_sc_out,
            head_sc_row_stride=heads_per_rank,
            worker_id=worker,
            num_workers=num_workers)
    pool.dispatch[PrefillAttnArgs,
        prefill_attn_persistent_worker[
            head_dim, max_seq, num_kv_heads, num_q_heads,
            heads_per_group, q_local, heads_per_rank]](
        UnsafePointer(to=jobs[0]), num_workers)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# MoE phase 1 — expert gate(w1) + up(w3) + SiLU + FWHT + quantize
# ============================================================================


def minimax_moe_phase1[
    AT: DynamicTensor, AsT: DynamicTensor,
    W1T: StaticTensor, W1ScT: StaticTensor, W1CsT: StaticTensor,
    W3T: StaticTensor, W3ScT: StaticTensor, W3CsT: StaticTensor,
    QiT: DynamicTensor, BScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    intermediate: Int, hidden: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int,
](
    act_i8: AT,
    act_scale: AsT,
    routing: TopKResult[top_k],
    w1: W1T, w1_stride: Int,
    w1_sc: W1ScT, w1_sc_stride: Int,
    w1_cs: W1CsT, w1_cs_stride: Int,
    w3: W3T, w3_stride: Int,
    w3_sc: W3ScT, w3_sc_stride: Int,
    w3_cs: W3CsT, w3_cs_stride: Int,
    expert_qi: QiT,
    expert_blk_scale: BScT,
    rank: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Multi-expert phase 1: gate(w1) + up(w3) + SiLU + FWHT + per-block i8.

    Filters routing.indices to experts owned by this rank, then builds
    N-tile-sharded FusedW1W3SiluArgs jobs across all local experts.
    """
    comptime assert AT.DTYPE == DType.int8, "moe_phase1: act must be i8"
    comptime assert AsT.DTYPE == DType.float32, "moe_phase1: act_scale must be f32"
    comptime assert W1T.DTYPE == DType.int8, "moe_phase1: w1 must be i8"
    comptime assert W1ScT.DTYPE == DType.float32, "moe_phase1: w1_sc must be f32"
    comptime assert W1CsT.DTYPE == DType.float32, "moe_phase1: w1_cs must be f32"
    comptime assert W3T.DTYPE == DType.int8, "moe_phase1: w3 must be i8"
    comptime assert W3ScT.DTYPE == DType.float32, "moe_phase1: w3_sc must be f32"
    comptime assert W3CsT.DTYPE == DType.float32, "moe_phase1: w3_cs must be f32"
    comptime assert QiT.DTYPE == DType.int8, "moe_phase1: expert_qi must be i8"
    comptime assert BScT.DTYPE == DType.float32, "moe_phase1: expert_blk_scale must be f32"

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
        return PoolFence[P, origin].over(pool)

    var workers_per_expert = pool_capacity // local_count
    if workers_per_expert < 1:
        workers_per_expert = 1
    if workers_per_expert > n_tiles:
        workers_per_expert = n_tiles
    var tiles_per_worker = (n_tiles + workers_per_expert - 1) // workers_per_expert

    var act_p = act_i8.as_ptr[DType.int8]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()
    var w1_base = w1.addr()
    var w1_sc_base = w1_sc.addr()
    var w1_cs_base = w1_cs.addr()
    var w3_base = w3.addr()
    var w3_sc_base = w3_sc.addr()
    var w3_cs_base = w3_cs.addr()
    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()

    var jobs = InlineArray[FusedW1W3SiluArgs, MAX_POOL_CAPACITY](
        fill=FusedW1W3SiluArgs())

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var w1p = I8Ptr(unsafe_from_address=w1_base + local_idx * w1_stride)
        var w1s = F32Ptr(unsafe_from_address=w1_sc_base + local_idx * w1_sc_stride)
        var w1c = F32Ptr(unsafe_from_address=w1_cs_base + local_idx * w1_cs_stride)
        var w3p = I8Ptr(unsafe_from_address=w3_base + local_idx * w3_stride)
        var w3s = F32Ptr(unsafe_from_address=w3_sc_base + local_idx * w3_sc_stride)
        var w3c = F32Ptr(unsafe_from_address=w3_cs_base + local_idx * w3_cs_stride)
        var qi_out = expert_qi_p + li * intermediate
        var blk_out = expert_blk_scale_p + li * n_tiles

        for w in range(workers_per_expert):
            var tile_start = w * tiles_per_worker
            if tile_start >= n_tiles:
                break
            var tile_end = min(tile_start + tiles_per_worker, n_tiles)
            var n_start = tile_start * fwht_blk
            var n_count = (tile_end - tile_start) * fwht_blk
            jobs[num_jobs] = FusedW1W3SiluArgs(
                act_p, act_scale_p,
                w1p, w1s, w1c,
                w3p, w3s, w3c,
                qi_out + n_start,
                blk_out + tile_start,
                n_start, n_count, 1)
            num_jobs += 1

    pool.dispatch[FusedW1W3SiluArgs, fused_w1_w3_silu_worker[intermediate, hidden, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


def minimax_sparse_moe_phase1[
    AT: DynamicTensor, AsT: DynamicTensor,
    W1T: StaticTensor, W1ScT: StaticTensor,
    W3T: StaticTensor, W3ScT: StaticTensor,
    QiT: DynamicTensor, BScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    experts_per_rank: Int, intermediate: Int, hidden: Int, fwht_blk: Int,
](
    act_i8: AT,
    act_scale: AsT,
    offsets: UnsafePointer[Int32, MutAnyOrigin],
    routes: UnsafePointer[SparseRoute, MutAnyOrigin],
    w1: W1T, w1_stride_elems: Int,
    w1_sc: W1ScT, w1_sc_stride_elems: Int,
    w3: W3T, w3_stride_elems: Int,
    w3_sc: W3ScT, w3_sc_stride_elems: Int,
    expert_qi: QiT,
    expert_blk_scale: BScT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Persistent sparse prefill phase1 over rank-local expert buckets."""
    comptime assert AT.DTYPE == DType.int8, "sparse_moe_phase1: act must be i8"
    comptime assert AsT.DTYPE == DType.float32, "sparse_moe_phase1: act_scale must be f32"
    comptime assert W1T.DTYPE == DType.int8, "sparse_moe_phase1: w1 must be i8"
    comptime assert W1ScT.DTYPE == DType.float32, "sparse_moe_phase1: w1_sc must be f32"
    comptime assert W3T.DTYPE == DType.int8, "sparse_moe_phase1: w3 must be i8"
    comptime assert W3ScT.DTYPE == DType.float32, "sparse_moe_phase1: w3_sc must be f32"
    comptime assert QiT.DTYPE == DType.int8, "sparse_moe_phase1: expert_qi must be i8"
    comptime assert BScT.DTYPE == DType.float32, "sparse_moe_phase1: expert_blk_scale must be f32"
    debug_assert(w1_stride_elems == w3_stride_elems,
        "sparse_moe_phase1: w1/w3 expert strides must match")
    debug_assert(w1_sc_stride_elems == w3_sc_stride_elems,
        "sparse_moe_phase1: w1/w3 scale strides must match")

    var num_workers = min(experts_per_rank, pool.get_capacity())
    if num_workers <= 0:
        return PoolFence[P, origin].over(pool)

    var act_p = act_i8.as_ptr[DType.int8]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()
    var w1_p = I8Ptr(unsafe_from_address=w1.addr())
    var w1_sc_p = F32Ptr(unsafe_from_address=w1_sc.addr())
    var w3_p = I8Ptr(unsafe_from_address=w3.addr())
    var w3_sc_p = F32Ptr(unsafe_from_address=w3_sc.addr())
    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()

    var jobs = InlineArray[SparseMoePhase1Args, MAX_POOL_CAPACITY](
        fill=SparseMoePhase1Args())
    for i in range(num_workers):
        jobs[i] = SparseMoePhase1Args(
            act_p, act_scale_p,
            offsets, routes,
            w1_p, w1_sc_p,
            w3_p, w3_sc_p,
            expert_qi_p, expert_blk_scale_p,
            w1_stride_elems, w1_sc_stride_elems, i, num_workers)

    pool.dispatch[SparseMoePhase1Args,
        sparse_moe_phase1_worker[
            experts_per_rank, intermediate, hidden, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_workers)
    return PoolFence[P, origin].over(pool)


def minimax_sparse_moe_phase2[
    QiT: DynamicTensor, BScT: DynamicTensor,
    DnT: StaticTensor, DnScT: StaticTensor,
    AccT: DynamicTensor, OutT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    experts_per_rank: Int, hidden: Int, intermediate: Int, fwht_blk: Int,
](
    offsets: UnsafePointer[Int32, MutAnyOrigin],
    routes: UnsafePointer[SparseRoute, MutAnyOrigin],
    expert_qi: QiT,
    expert_blk_scale: BScT,
    down: DnT, down_stride_elems: Int,
    down_sc: DnScT, down_sc_stride_elems: Int,
    accum: AccT,
    dst: OutT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Bucketed sparse prefill phase2 over hidden stripes."""
    comptime assert QiT.DTYPE == DType.int8, "sparse_moe_phase2: expert_qi must be i8"
    comptime assert BScT.DTYPE == DType.float32, "sparse_moe_phase2: expert_blk_scale must be f32"
    comptime assert DnT.DTYPE == DType.int8, "sparse_moe_phase2: down must be i8"
    comptime assert DnScT.DTYPE == DType.float32, "sparse_moe_phase2: down_sc must be f32"
    comptime assert AccT.DTYPE == DType.float32, "sparse_moe_phase2: accum must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "sparse_moe_phase2: dst must be bf16"
    debug_assert(down_sc_stride_elems == hidden,
        "sparse_moe_phase2: down scale stride must be hidden")

    var seq_len = dst.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    comptime max_n_workers = hidden // VNNI_N_STEP
    var num_workers = min(pool.get_capacity(), max_n_workers)
    if num_workers <= 0:
        return PoolFence[P, origin].over(pool)
    var n_per_worker = (
        ((max_n_workers + num_workers - 1) // num_workers) * VNNI_N_STEP
    )

    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()
    var down_p = I8Ptr(unsafe_from_address=down.addr())
    var down_sc_p = F32Ptr(unsafe_from_address=down_sc.addr())
    var accum_p = accum.as_ptr[DType.float32]()
    var dst_p = dst.as_ptr[DType.bfloat16]()

    var jobs = InlineArray[SparseMoePhase2Args, MAX_POOL_CAPACITY](
        fill=SparseMoePhase2Args())
    var actual = 0
    for i in range(num_workers):
        var n_start = i * n_per_worker
        if n_start >= hidden:
            break
        var n_count = min(n_per_worker, hidden - n_start)
        jobs[actual] = SparseMoePhase2Args(
            offsets, routes,
            expert_qi_p, expert_blk_scale_p,
            down_p, down_sc_p, accum_p, dst_p,
            down_stride_elems, down_sc_stride_elems,
            seq_len, n_start, n_count)
        actual += 1

    pool.dispatch[SparseMoePhase2Args,
        sparse_moe_phase2_worker[
            experts_per_rank, hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# MoE phase 2 — expert down projection with routing weight
# ============================================================================


def minimax_moe_phase2[
    QiT: DynamicTensor, BScT: DynamicTensor,
    DnT: StaticTensor, DnScT: StaticTensor, DnCsT: StaticTensor,
    OutT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    hidden: Int, intermediate: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int,
](
    expert_qi: QiT,
    expert_blk_scale: BScT,
    routing: TopKResult[top_k],
    down: DnT, down_stride: Int,
    down_sc: DnScT, down_sc_stride: Int,
    down_bcs: DnCsT, down_bcs_stride: Int,
    expert_out_buf: OutT,
    rank: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Per-local-expert down GEMV with routing weight folded into output scale.

    N-tile sharded: each expert's hidden-dimension output is split across
    workers, matching the phase1 pattern.
    """
    comptime assert QiT.DTYPE == DType.int8, "moe_phase2: expert_qi must be i8"
    comptime assert BScT.DTYPE == DType.float32, "moe_phase2: expert_blk_scale must be f32"
    comptime assert DnT.DTYPE == DType.int8, "moe_phase2: down must be i8"
    comptime assert DnScT.DTYPE == DType.float32, "moe_phase2: down_sc must be f32"
    comptime assert DnCsT.DTYPE == DType.float32, "moe_phase2: down_bcs must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "moe_phase2: expert_out_buf must be bf16"

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
        return PoolFence[P, origin].over(pool)

    var workers_per_expert = pool_capacity // local_count
    if workers_per_expert < 1:
        workers_per_expert = 1
    comptime max_n_workers = hidden // VNNI_N_STEP
    if workers_per_expert > max_n_workers:
        workers_per_expert = max_n_workers
    var n_per_worker = ((max_n_workers + workers_per_expert - 1) // workers_per_expert) * VNNI_N_STEP

    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()
    var down_base = down.addr()
    var down_sc_base = down_sc.addr()
    var down_bcs_base = down_bcs.addr()
    var expert_out_p = expert_out_buf.as_ptr[DType.bfloat16]()

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var act = expert_qi_p + li * intermediate
        var wpacked = I8Ptr(unsafe_from_address=down_base + local_idx * down_stride)
        var blk_scale = expert_blk_scale_p + li * num_blocks
        var wscale = F32Ptr(unsafe_from_address=down_sc_base + local_idx * down_sc_stride)
        var blk_colsum = F32Ptr(unsafe_from_address=down_bcs_base + local_idx * down_bcs_stride)
        var dst = expert_out_p + li * hidden
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
                weight, n_count, hidden, 1)
            num_jobs += 1

    pool.dispatch[Int8GemvBlockedArgs,
        int8_gemv_blocked_decode_worker[hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)
