"""Gemma 4 int8 MoE — phase dispatch helpers, post-reduce combine.

Entry points used by the forward pass:

  gemma4_moe_phase1 — filter local experts, build N-tile-sharded
    fused_gu_gelu_tanh jobs across all of them, single pool.dispatch.
    Mirrors fused_gu_gelu_tanh's fence-return shape.

  gemma4_moe_phase2 — one int8_gemv_blocked job per local expert, single
    pool.dispatch. routing weight is passed as output_scale.

  moe_combine — post-allreduce: rmsnorm(moe_out) + dense_normed →
  rmsnorm(combined) → residual add → layer_scalar. Called from a dispatched
  kernel wrapper (body threads never compute).

Expert weights are block-sharded along the expert dimension: rank `r` owns
experts `[r*EPR, (r+1)*EPR)` where `EPR = num_experts // tp`. Local index
for expert `e` on its owning rank is `e - rank*EPR`. The flat
`[num_experts*M, K]` source tensor collapses to a contiguous row range
per rank, so the loader stores one contiguous slice per rank with no
strided reads. All memory is rank-local. Phase 1 inputs (act_i8/act_scale)
must be produced by the rank-local rmsnorm_dual_gamma_fwht_quantize.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from kernels.kernel_ops import PoolFence
from simd_math import sqrt
from experimental3.kernels.int8_gemv import dot
from experimental3.kernels.dense_ffn import (
    FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker,
    Int8GemvBlockedArgs, int8_gemv_blocked_worker,
)
from experimental_gemma.router import Gemma4TopKResult
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr


# ============================================================================
# Blocked GEMV — per-K-block activation scales for down projection
# ============================================================================


def gemv_row_blocked[N: Int, K: Int, fwht_block_size: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    """GEMV with per-K-block activation scales. Accumulates per block in i32,
    dequants to f32 per block, then applies per-row weight scale."""
    debug_assert(K % fwht_block_size == 0,
        "gemv_row_blocked: K must be a multiple of fwht_block_size")
    debug_assert(fwht_block_size >= VNNI_K_STEP,
        "gemv_row_blocked: fwht_block_size must be >= VNNI_K_STEP (64)")
    debug_assert(N % VNNI_N_STEP == 0,
        "gemv_row_blocked: N must be a multiple of VNNI_N_STEP (32)")
    comptime num_blocks = K // fwht_block_size
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_acc = InlineArray[SIMD[DType.float32, width], acc_count](
                fill=SIMD[DType.float32, width](0))
            for blk in range(num_blocks):
                var i32_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                    fill=SIMD[DType.int32, width](0))
                for ks in range(0, fwht_block_size, VNNI_K_STEP):
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[p] = dot[width](i32_acc[p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos)
                        packed_off += VNNI_TILE_N * VNNI_BLK
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[passes_per_subtile + p] = dot[width](
                                i32_acc[passes_per_subtile + p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos)
                        packed_off += VNNI_TILE_N * VNNI_BLK
                var blk_dequant = block_scales[blk] / 127.0
                for a in range(acc_count):
                    var n_base = nb + ns + a * width
                    var corrected = i32_acc[a].cast[DType.float32]() - 128.0 * (block_colsums + blk * N + n_base).load[width=width]()
                    f32_acc[a] += corrected * blk_dequant
            for a in range(acc_count):
                var n_base = nb + ns + a * width
                (dst + n_base).store(f32_acc[a] * (wsc + n_base).load[width=width]())


# ============================================================================
# Phase 1: rank-local multi-expert N-tile-sharded fused gate_up + gelu + fwht + quant
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

    Iterates routing.indices, filters to experts owned by this rank
    (block sharding: expert e on rank e // EPR, where EPR = num_experts // tp),
    and for each local expert builds N-tile-sharded `FusedGuGeluTanhArgs` jobs
    targeting that expert's slot in `expert_qi` / `expert_blk_scale`. One
    `pool.dispatch` with `local_count * tiles_per_expert` jobs total.
    """
    comptime n_tiles = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp
    comptime MAX_POOL_CAPACITY = 128

    var pool_capacity = pool.get_capacity()
    var expert_base = rank * experts_per_rank

    # Count local experts first to decide workers per expert.
    var local_count = 0
    var local_slots = InlineArray[Int, top_k](fill=0)
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid >= expert_base and eid < expert_base + experts_per_rank:
            local_slots[local_count] = s
            local_count += 1

    if local_count == 0:
        return PoolFence[P].completed()

    # Workers per expert: try to spread the pool across local experts, but cap
    # by n_tiles (no point making more workers than tiles).
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
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Phase 2: rank-local per-expert down GEMV with routing weight
# ============================================================================


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

    Expert ownership matches phase 1 (block sharding on the expert dim).
    One job per local expert: int8_gemv_blocked_worker reading the expert's
    `expert_qi` slot (written by phase 1), with `output_scale = routing.weights[s]`
    so the routing scale is folded into the bf16 cast for free.
    """
    comptime num_blocks = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp
    comptime MAX_POOL_CAPACITY = 128
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
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


# ============================================================================
# Post-allreduce combine
# ============================================================================


def moe_combine[hidden: Int](
    moe_out: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    moe_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dense_normed: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    combine_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    x_main: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    layer_scalar: Float32,
    eps: Float32,
):
    """Post-allreduce combine. All inline on 2816 elements.

    1. moe_normed = rmsnorm(moe_out, POST_FFN_NORM_2.γ)
    2. combined = moe_normed + dense_normed
    3. combined_normed = rmsnorm(combined, POST_FFN_NORM.γ)
    4. x_main = (x_main + combined_normed) * layer_scalar

    moe_out:       bf16[hidden] — allreduced MoE output (read, overwritten as scratch)
    dense_normed:  bf16[hidden] — pre-computed rmsnorm(dense_out, POST_FFN_NORM_1.γ)
    x_main:        bf16[hidden] — residual stream (read-modify-write)
    """
    comptime width = simd_width_of[DType.float32]()

    # Step 1: rmsnorm(moe_out, γ₂) → overwrite moe_out with moe_normed
    var sum_sq = SIMD[DType.float32, width](0)
    var i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (moe_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 2: combined = moe_normed + dense_normed (in-place into moe_out)
    i = 0
    while i + width <= hidden:
        var m = (moe_out + i).load[width=width]().cast[DType.float32]()
        var d = (dense_normed + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((m + d).cast[DType.bfloat16]())
        i += width

    # Step 3: rmsnorm(combined, γ₃)
    sum_sq = SIMD[DType.float32, width](0)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (combine_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 4: x_main = (x_main + combined_normed) * layer_scalar
    i = 0
    while i + width <= hidden:
        var x = (x_main + i).load[width=width]().cast[DType.float32]()
        var c = (moe_out + i).load[width=width]().cast[DType.float32]()
        (x_main + i).store(((x + c) * layer_scalar).cast[DType.bfloat16]())
        i += width
