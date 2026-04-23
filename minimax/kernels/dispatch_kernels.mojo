from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from simd_math import sqrt
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from modeling.model_spec import StaticTensor, DynamicTensor
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.kernels.dispatch_args import Int8GemvBlockedArgs, ChunkedAttnArgs
from experimental3.kernels.gemm import int8_gemv_blocked_worker, int8_gemv_blocked_decode_worker
from kernels.vnni import VNNI_N_STEP
from experimental3.kv_cache import CACHE_WIDTH
from minimax.kernels.amx_attention import amx_chunked_attn_kernel

from minimax.kernels.qk_prep import prep_q_head
from minimax.kernels.dispatch_args import (
    FusedW1W3SiluArgs,
    AttnGroupArgs,
    RouterCandidate,
    TopKResult,
    RouterFusedArgs,
)
from minimax.kernels.gemm import fused_w1_w3_silu_worker
from minimax.kernels.attention import kv_write_kernel
from minimax.kernels.router import router_fused_worker


# ============================================================================
# Fused router — f32 GEMV + sigmoid + bias + local top-K (phase 1)
# ============================================================================


@always_inline
def router_num_workers[num_experts: Int, k: Int](pool_capacity: Int) -> Int:
    """Shared formula: phase-1 worker count = min(num_experts/k, pool cap).
    Constrained by k so every worker has ≥ k rows and produces a full
    local top-K."""
    comptime max_workers = num_experts // k
    return min(max_workers, pool_capacity)


def router_fused_dispatch[
    ActT: DynamicTensor, WT: StaticTensor, BT: StaticTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    num_experts: Int, hidden: Int, k: Int,
](
    act_bf16: ActT,
    weight_f32: WT,
    bias_f32: BT,
    candidates: UnsafePointer[RouterCandidate, MutAnyOrigin],
    ref [origin] pool: P,
    expert_base: Int = 0,
) -> PoolFence[P, origin]:
    comptime assert ActT.DTYPE == DType.bfloat16, "router_fused: act must be bf16"
    comptime assert WT.DTYPE == DType.float32, "router_fused: weight must be f32"
    comptime assert BT.DTYPE == DType.float32, "router_fused: bias must be f32"

    var num_workers = router_num_workers[num_experts, k](pool.get_capacity())
    if num_workers <= 0:
        return PoolFence[P, origin].over(pool)
    var rows_per_worker = (num_experts + num_workers - 1) // num_workers

    var act_p = act_bf16.as_ptr[DType.bfloat16]()
    var weight_p = weight_f32.as_ptr[DType.float32]()
    var bias_p = bias_f32.as_ptr[DType.float32]()

    var jobs = InlineArray[RouterFusedArgs, MAX_POOL_CAPACITY](
        fill=RouterFusedArgs())
    var actual = 0
    for i in range(num_workers):
        var start = i * rows_per_worker
        if start >= num_experts:
            break
        var count = min(rows_per_worker, num_experts - start)
        var slot = candidates + i * k
        jobs[actual] = RouterFusedArgs(
            act_p, weight_p, bias_p, slot, expert_base, start, count)
        actual += 1

    pool.dispatch[RouterFusedArgs, router_fused_worker[hidden, k]](
        UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Attention phase A: K/V cache write — NKV_LOCAL parallel jobs
# ============================================================================


def kv_write_dispatch[
    QT: DynamicTensor, KT: DynamicTensor, VT: DynamicTensor,
    QnT: StaticTensor, KnT: StaticTensor,
    CosT: StaticTensor, SinT: StaticTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, rope_dim: Int, pair_stride: Int,
    max_seq: Int, num_kv_heads: Int,
](
    q_bf16: QT, k_bf16: KT, v_bf16: VT,
    q_norm: QnT, k_norm: KnT,
    cos_table: CosT, sin_table: SinT,
    inv_rms_q: Float32, inv_rms_k: Float32,
    cache_base: Int, cache_pos: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert QT.DTYPE == DType.bfloat16, "kv_write: q must be bf16"
    comptime assert KT.DTYPE == DType.bfloat16, "kv_write: k must be bf16"
    comptime assert VT.DTYPE == DType.bfloat16, "kv_write: v must be bf16"
    comptime assert QnT.DTYPE == DType.bfloat16, "kv_write: q_norm must be bf16"
    comptime assert KnT.DTYPE == DType.bfloat16, "kv_write: k_norm must be bf16"
    comptime assert CosT.DTYPE == DType.float32, "kv_write: cos must be f32"
    comptime assert SinT.DTYPE == DType.float32, "kv_write: sin must be f32"

    comptime HPG = 1
    var jobs = InlineArray[AttnGroupArgs, 8](fill=AttnGroupArgs())
    comptime K_HEAD_BF16 = head_dim * 2
    comptime V_HEAD_BF16 = head_dim * 2

    var q_p = q_bf16.as_ptr[DType.bfloat16]()
    var k_p = k_bf16.as_ptr[DType.bfloat16]()
    var v_p = v_bf16.as_ptr[DType.bfloat16]()
    var q_norm_p = q_norm.as_ptr[DType.bfloat16]()
    var k_norm_p = k_norm.as_ptr[DType.bfloat16]()
    var cos_p = cos_table.as_ptr[DType.float32]()
    var sin_p = sin_table.as_ptr[DType.float32]()
    var cache = U8Ptr(unsafe_from_address=cache_base)

    for g in range(num_kv_heads):
        jobs[g] = AttnGroupArgs(
            q_bf16_base=q_p,
            k_bf16_ptr=k_p + g * head_dim,
            v_bf16_ptr=v_p + g * head_dim,
            q_norm_ptr=q_norm_p,
            k_norm_ptr=k_norm_p + g * head_dim,
            cos_ptr=cos_p,
            sin_ptr=sin_p,
            inv_rms_q=inv_rms_q,
            inv_rms_k=inv_rms_k,
            cache_base=cache,
            cache_pos=cache_pos,
            kv_head=g,
            context_len=0,
            qi_out=I8Ptr(),
            head_scale_ptr=F32Ptr())
    pool.dispatch[AttnGroupArgs,
        kv_write_kernel[head_dim, rope_dim, pair_stride, max_seq, num_kv_heads]](
        UnsafePointer(to=jobs[0]), num_kv_heads)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Q prep kernel — called on the main thread, reuses AttnGroupArgs plumbing
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


comptime ATTN_PG_PER_WORKER = 32


@always_inline
def attn_chunk_count(
    context_len: Int, pool_capacity: Int, max_attn_chunks: Int,
) -> Int:
    """Floor of pgs-per-worker. The score dispatch and merge phase must
    agree on num_chunks; this is the single source of truth."""
    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var padded_pg = (num_pg + 3) & ~3
    if padded_pg <= 0:
        return 0
    var raw_chunks = (padded_pg + ATTN_PG_PER_WORKER - 1) // ATTN_PG_PER_WORKER
    var capped = min(raw_chunks, min(pool_capacity, max_attn_chunks))
    if capped < 1:
        return 1
    return capped


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
        context_len, pool_capacity, max_attn_chunks)
    if num_chunks <= 0 or kv_count <= 0:
        return PoolFence[P, origin].over(pool)
    var pgs_per_chunk = ((padded_pg + num_chunks - 1) // num_chunks + 3) & ~3
    comptime CHUNK_F32_STRIDE = heads_per_group * (2 + head_dim)
    comptime Q_STRIDE = heads_per_group * head_dim
    comptime QSCALE_STRIDE = heads_per_group

    var q_i8 = I8Ptr(unsafe_from_address=q_i8_base)
    var qi_biases = F32Ptr(unsafe_from_address=qi_biases_base)
    var q_scales = F32Ptr(unsafe_from_address=q_scales_base)
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var partial_out = F32Ptr(unsafe_from_address=partial_out_base)

    var chunk_args = InlineArray[
        ChunkedAttnArgs, max_attn_chunks * num_kv_heads](
        fill=ChunkedAttnArgs())
    var idx = 0
    for kv_off in range(kv_count):
        var kv = kv_start + kv_off
        for c in range(num_chunks):
            var start = c * pgs_per_chunk
            var end = min((c + 1) * pgs_per_chunk, padded_pg)
            chunk_args[idx] = ChunkedAttnArgs(
                q_i8_base=q_i8 + kv_off * Q_STRIDE,
                qi_biases_base=qi_biases + kv_off * QSCALE_STRIDE,
                q_scales_base=q_scales + kv_off * QSCALE_STRIDE,
                cache_base=cache,
                kv_head=kv,
                start_pg=start,
                end_pg=end,
                partial_out=partial_out + (
                    kv_off * num_chunks + c) * CHUNK_F32_STRIDE,
                context_len=context_len)
            idx += 1
    pool.dispatch[ChunkedAttnArgs,
        amx_chunked_attn_kernel[
            head_dim, max_seq, num_kv_heads, 0, heads_per_group,
            max_attn_chunks]](
        UnsafePointer(to=chunk_args[0]), idx)
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
                weight, n_count, hidden)
            num_jobs += 1

    pool.dispatch[Int8GemvBlockedArgs,
        int8_gemv_blocked_decode_worker[hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)
