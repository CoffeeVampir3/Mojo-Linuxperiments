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

from modeling.model_spec import (
    Encoding, Shaped, Aligned, HasPtr, Dynamic,
    StaticTensor, DynamicTensor,
    StaticView, DynamicView, Shape, BF16,
)
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
    int8_gemv_decode_worker,
    int8_gemv_blocked_worker, int8_gemv_blocked_decode_worker,
    int8_gemv_blocked_wa_worker,
    fused_gu_gelu_tanh_worker, fused_gu_gelu_tanh_worker_wa,
    GEMV_TILE,
    lm_head_worker,
)
from experimental3.kernels.gemm_amx import (
    int8_gemm_amx_worker, int8_gemm_blocked_amx_worker,
)
from experimental3.kernels.dispatch_helpers import tile_and_dispatch
from kernels.vnni import VNNI_N_STEP
from experimental3.moe import router_topk_kernel
from experimental3.kernels.sliding_attention import sliding_attn_group_kernel
from experimental3.kernels.full_chunked_attention_fused import (
    cp_attn_prep_kernel,
    cp_chunked_attn_kernel,
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




# ============================================================================
# int8_gemv — per-row weight + per-row act scale
# ============================================================================


def int8_gemv[
    AT: DynamicTensor, WT: StaticTensor, CsT: StaticTensor,
    WsT: StaticTensor, DstT: DynamicTensor, AsT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    N: Int, K: Int,
](
    act: AT, wpacked: WT, colsum: CsT, weight_scale: WsT,
    dst: DstT, act_scale: AsT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dispatch int8 GEMV: [seq_len, K] x [N, K]^T -> [seq_len, N] bf16.

    Decode (seq_len=1): N-split across workers. Each worker computes a
    sub-range of output elements from the same activation row.
    Prefill (seq_len>1): M-split across workers. Each worker computes
    full N outputs for a subset of rows.
    """
    comptime assert AT.DTYPE == DType.int8, "int8_gemv: act must be i8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemv: wpacked must be i8"
    comptime assert CsT.DTYPE == DType.float32, "int8_gemv: colsum must be f32"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemv: weight_scale must be f32"
    comptime assert DstT.DTYPE == DType.bfloat16, "int8_gemv: dst must be bf16"
    comptime assert AsT.DTYPE == DType.float32, "int8_gemv: act_scale must be f32"

    var seq_len = act.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_p = act.as_ptr[DType.int8]()
    var wpacked_p = wpacked.as_ptr[DType.int8]()
    var colsum_p = colsum.as_ptr[DType.float32]()
    var weight_scale_p = weight_scale.as_ptr[DType.float32]()
    var dst_p = dst.as_ptr[DType.bfloat16]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()

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
                act_p, wpacked_p + n_start * K, colsum_p + n_start,
                weight_scale_p + n_start, dst_p + n_start,
                act_scale_p, 0, n_count)
            actual += 1
        pool.dispatch[WorkerConfig, int8_gemv_decode_worker[N, K]](
            UnsafePointer(to=jobs[0]), actual)
        return PoolFence[P, origin].over(pool)

    @parameter
    def factory(start: Int, count: Int) -> WorkerConfig:
        return WorkerConfig(
            act_p + start * K, wpacked_p, colsum_p,
            weight_scale_p, dst_p + start * N,
            act_scale_p, start, count)
    return tile_and_dispatch[
        kernel=int8_gemm_amx_worker[N, K], factory=factory,
    ](seq_len, pool)


# ============================================================================
# int8_gemv_blocked — per-K-block activation scales (down projection)
# ============================================================================


def int8_gemv_blocked[
    AT: DynamicTensor, WT: StaticTensor, BScT: DynamicTensor,
    WsT: StaticTensor, BCsT: StaticTensor, DstT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    N: Int, K: Int, fwht_blk: Int,
](
    act: AT, wpacked: WT, blk_scale: BScT,
    wscale: WsT, blk_colsum: BCsT, dst: DstT,
    ref [origin] pool: P,
    output_scale: Float32 = Float32(1.0),
) -> PoolFence[P, origin]:
    """Dispatch int8 GEMV with per-block activation scales.

    Decode (seq_len=1): N-split across workers.
    Prefill (seq_len>1): M-split across workers.
    """
    comptime assert AT.DTYPE == DType.int8, "int8_gemv_blocked: act must be i8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemv_blocked: wpacked must be i8"
    comptime assert BScT.DTYPE == DType.float32, "int8_gemv_blocked: blk_scale must be f32"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemv_blocked: wscale must be f32"
    comptime assert BCsT.DTYPE == DType.float32, "int8_gemv_blocked: blk_colsum must be f32"
    comptime assert DstT.DTYPE == DType.bfloat16, "int8_gemv_blocked: dst must be bf16"

    var seq_len = act.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_p = act.as_ptr[DType.int8]()
    var wpacked_p = wpacked.as_ptr[DType.int8]()
    var blk_scale_p = blk_scale.as_ptr[DType.float32]()
    var wscale_p = wscale.as_ptr[DType.float32]()
    var blk_colsum_p = blk_colsum.as_ptr[DType.float32]()
    var dst_p = dst.as_ptr[DType.bfloat16]()

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
                act_p, wpacked_p + n_start * K, blk_scale_p,
                wscale_p + n_start, blk_colsum_p + n_start,
                dst_p + n_start, output_scale, n_count, N, 1)
            actual += 1
        pool.dispatch[Int8GemvBlockedArgs,
            int8_gemv_blocked_decode_worker[N, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), actual)
        return PoolFence[P, origin].over(pool)

    @parameter
    def factory(start: Int, count: Int) -> Int8GemvBlockedArgs:
        return Int8GemvBlockedArgs(
            act_p + start * K, wpacked_p,
            blk_scale_p + start * num_blocks, wscale_p, blk_colsum_p,
            dst_p + start * N, output_scale, N, N, count)
    return tile_and_dispatch[
        kernel=int8_gemm_blocked_amx_worker[N, K, fwht_blk], factory=factory,
    ](seq_len, pool)


def int8_gemv_blocked_wa[
    AT: DynamicTensor, WT: StaticTensor, BScT: DynamicTensor,
    WsT: StaticTensor, BCsT: StaticTensor, DstT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    N: Int, K: Int, fwht_blk: Int,
](
    act: AT, wpacked: WT, blk_scale: BScT,
    wscale: WsT, blk_colsum: BCsT, dst: DstT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dispatch workaround blocked GEMV for sub-VNNI_K_STEP block sizes."""
    comptime assert AT.DTYPE == DType.int8, "int8_gemv_blocked_wa: act must be i8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemv_blocked_wa: wpacked must be i8"
    comptime assert BScT.DTYPE == DType.float32, "int8_gemv_blocked_wa: blk_scale must be f32"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemv_blocked_wa: wscale must be f32"
    comptime assert BCsT.DTYPE == DType.float32, "int8_gemv_blocked_wa: blk_colsum must be f32"
    comptime assert DstT.DTYPE == DType.bfloat16, "int8_gemv_blocked_wa: dst must be bf16"

    var seq_len = act.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_p = act.as_ptr[DType.int8]()
    var wpacked_p = wpacked.as_ptr[DType.int8]()
    var blk_scale_p = blk_scale.as_ptr[DType.float32]()
    var wscale_p = wscale.as_ptr[DType.float32]()
    var blk_colsum_p = blk_colsum.as_ptr[DType.float32]()
    var dst_p = dst.as_ptr[DType.bfloat16]()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
    comptime num_blocks = K // fwht_blk

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= seq_len:
            break
        var end = min(start + rows_per_job, seq_len)
        jobs[actual] = Int8GemvBlockedArgs(
            act_p + start * K, wpacked_p,
            blk_scale_p + start * num_blocks,
            wscale_p, blk_colsum_p,
            dst_p + start * N, Float32(1.0), N, N, end - start)
        actual += 1

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_wa_worker[N, K, fwht_blk]](
        UnsafePointer(to=jobs[0]), actual)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# fused_gu_gelu_tanh — gate+up GEMV -> gelu -> [FWHT] -> per-block quantize
# ============================================================================


def fused_gu_gelu_tanh[
    AT: DynamicTensor, AsT: DynamicTensor, WT: StaticTensor,
    WsT: StaticTensor, WcT: StaticTensor,
    QiT: DynamicTensor, BScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    intermediate: Int, K: Int, fwht_blk: Int, fwht: Bool = True,
](
    act_i8: AT, act_scale: AsT,
    wpacked: WT, wscale: WsT, wcolsum: WcT,
    qi_out: QiT, blk_scale: BScT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dispatch gate_up GEMV + GELU-tanh + [FWHT] + per-block quantize.

    For seq_len=1 (decode): parallelizes across the output dimension N.
    For seq_len>1 (prompt): parallelizes across sequence rows.
    """
    comptime assert AT.DTYPE == DType.int8, "fused_gu_gelu_tanh: act_i8 must be i8"
    comptime assert AsT.DTYPE == DType.float32, "fused_gu_gelu_tanh: act_scale must be f32"
    comptime assert WT.DTYPE == DType.int8, "fused_gu_gelu_tanh: wpacked must be i8"
    comptime assert WsT.DTYPE == DType.float32, "fused_gu_gelu_tanh: wscale must be f32"
    comptime assert WcT.DTYPE == DType.float32, "fused_gu_gelu_tanh: wcolsum must be f32"
    comptime assert QiT.DTYPE == DType.int8, "fused_gu_gelu_tanh: qi_out must be i8"
    comptime assert BScT.DTYPE == DType.float32, "fused_gu_gelu_tanh: blk_scale must be f32"

    var seq_len = act_i8.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_i8_p = act_i8.as_ptr[DType.int8]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()
    var wpacked_p = wpacked.as_ptr[DType.int8]()
    var wscale_p = wscale.as_ptr[DType.float32]()
    var wcolsum_p = wcolsum.as_ptr[DType.float32]()
    var qi_out_p = qi_out.as_ptr[DType.int8]()
    var blk_scale_p = blk_scale.as_ptr[DType.float32]()

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
                act_i8_p, act_scale_p,
                wpacked_p, wscale_p, wcolsum_p,
                qi_out_p + n_start,
                blk_scale_p + tile_start,
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
                act_i8_p + row_start * K,
                act_scale_p + row_start,
                wpacked_p, wscale_p, wcolsum_p,
                qi_out_p + row_start * intermediate,
                blk_scale_p + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk, fwht]](
            UnsafePointer(to=jobs[0]), num_workers)

    return PoolFence[P, origin].over(pool)


def fused_gu_gelu_tanh_wa[
    AT: DynamicTensor, AsT: DynamicTensor, WT: StaticTensor,
    WsT: StaticTensor, WcT: StaticTensor,
    QiT: DynamicTensor, BScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    intermediate: Int, K: Int, fwht_blk: Int,
](
    act_i8: AT, act_scale: AsT,
    wpacked: WT, wscale: WsT, wcolsum: WcT,
    qi_out: QiT, blk_scale: BScT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dispatch workaround fused gate_up + GELU-tanh + FWHT(sub-block) + quantize."""
    comptime assert AT.DTYPE == DType.int8, "fused_gu_gelu_tanh_wa: act_i8 must be i8"
    comptime assert AsT.DTYPE == DType.float32, "fused_gu_gelu_tanh_wa: act_scale must be f32"
    comptime assert WT.DTYPE == DType.int8, "fused_gu_gelu_tanh_wa: wpacked must be i8"
    comptime assert WsT.DTYPE == DType.float32, "fused_gu_gelu_tanh_wa: wscale must be f32"
    comptime assert WcT.DTYPE == DType.float32, "fused_gu_gelu_tanh_wa: wcolsum must be f32"
    comptime assert QiT.DTYPE == DType.int8, "fused_gu_gelu_tanh_wa: qi_out must be i8"
    comptime assert BScT.DTYPE == DType.float32, "fused_gu_gelu_tanh_wa: blk_scale must be f32"

    var seq_len = act_i8.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var act_i8_p = act_i8.as_ptr[DType.int8]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()
    var wpacked_p = wpacked.as_ptr[DType.int8]()
    var wscale_p = wscale.as_ptr[DType.float32]()
    var wcolsum_p = wcolsum.as_ptr[DType.float32]()
    var qi_out_p = qi_out.as_ptr[DType.int8]()
    var blk_scale_p = blk_scale.as_ptr[DType.float32]()

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
                act_i8_p, act_scale_p, wpacked_p, wscale_p, wcolsum_p,
                qi_out_p + n_start,
                blk_scale_p + (n_start // fwht_blk),
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
                act_i8_p + row_start * K, act_scale_p + row_start,
                wpacked_p, wscale_p, wcolsum_p,
                qi_out_p + row_start * intermediate,
                blk_scale_p + row_start * num_blk_per_row,
                0, intermediate, row_count)
        pool.dispatch[FusedGuGeluTanhArgs,
            fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk]](
            UnsafePointer(to=jobs[0]), num_workers)

    return PoolFence[P, origin].over(pool)


# ============================================================================
# lm_head dispatchers — N-parallel decode (seq_len=1)
# ============================================================================


def lm_head_gemv[
    AT: DynamicTensor, WT: StaticTensor, ABsT: DynamicTensor,
    WBsT: StaticTensor, WBcT: StaticTensor, DstT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    N: Int, K: Int, fwht_blk: Int,
](
    act: AT, weight: WT, act_blk_scales: ABsT,
    w_blk_scales: WBsT, w_blk_colsums: WBcT, dst: DstT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """LM head GEMV: i8 [N, K] x i8 [K] -> bf16 [N] with per-block dequant.

    Decode only (seq_len=1). Work is N-split across pool workers.
    """
    comptime assert AT.DTYPE == DType.int8, "lm_head_gemv: act must be i8"
    comptime assert WT.DTYPE == DType.int8, "lm_head_gemv: weight must be i8"
    comptime assert ABsT.DTYPE == DType.float32, "lm_head_gemv: act_blk_scales must be f32"
    comptime assert WBsT.DTYPE == DType.float32, "lm_head_gemv: w_blk_scales must be f32"
    comptime assert WBcT.DTYPE == DType.float32, "lm_head_gemv: w_blk_colsums must be f32"
    comptime assert DstT.DTYPE == DType.bfloat16, "lm_head_gemv: dst must be bf16"

    var num_workers = min(N, pool.get_capacity())
    var rows_per_worker = (N + num_workers - 1) // num_workers

    var act_ptr = act.as_ptr[DType.int8]()
    var weight_ptr = weight.as_ptr[DType.int8]()
    var act_blk_scales_ptr = act_blk_scales.as_ptr[DType.float32]()
    var w_blk_scales_ptr = w_blk_scales.as_ptr[DType.float32]()
    var w_blk_colsums_ptr = w_blk_colsums.as_ptr[DType.float32]()
    var dst_ptr = dst.as_ptr[DType.bfloat16]()

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
    return PoolFence[P, origin].over(pool)


# ============================================================================
# MoE — phase 1, phase 2, router top-k
# ============================================================================


def gemma4_moe_phase1[
    AT: DynamicTensor, AsT: DynamicTensor,
    GuT: StaticTensor, GuScT: StaticTensor, GuCsT: StaticTensor,
    QiT: DynamicTensor, BScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    intermediate: Int, hidden: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int,
](
    act_i8: AT,
    act_scale: AsT,
    routing: Gemma4TopKResult[top_k],
    gate_up: GuT, gate_up_stride: Int,
    gate_up_sc: GuScT, gate_up_sc_stride: Int,
    gate_up_cs: GuCsT, gate_up_cs_stride: Int,
    expert_qi: QiT,
    expert_blk_scale: BScT,
    rank: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Multi-expert phase 1: gate_up + gelu_tanh + FWHT + per-block i8 quantize.

    Filters routing.indices to experts owned by this rank (block sharding:
    expert e on rank e // EPR, where EPR = num_experts // tp), then builds
    N-tile-sharded FusedGuGeluTanhArgs jobs across all local experts in a
    single pool.dispatch.
    """
    comptime assert AT.DTYPE == DType.int8, "moe_phase1: act_i8 must be i8"
    comptime assert AsT.DTYPE == DType.float32, "moe_phase1: act_scale must be f32"
    comptime assert GuT.DTYPE == DType.int8, "moe_phase1: gate_up must be i8"
    comptime assert GuScT.DTYPE == DType.float32, "moe_phase1: gate_up_sc must be f32"
    comptime assert GuCsT.DTYPE == DType.float32, "moe_phase1: gate_up_cs must be f32"
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

    var act_i8_p = act_i8.as_ptr[DType.int8]()
    var act_scale_p = act_scale.as_ptr[DType.float32]()
    var gate_up_base = gate_up.addr()
    var gate_up_sc_base = gate_up_sc.addr()
    var gate_up_cs_base = gate_up_cs.addr()
    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()

    var jobs = InlineArray[FusedGuGeluTanhArgs, MAX_POOL_CAPACITY](
        fill=FusedGuGeluTanhArgs())

    var num_jobs = 0
    for li in range(local_count):
        var s = local_slots[li]
        var eid = routing.indices[s]
        var local_idx = eid - expert_base

        var wpacked = I8Ptr(unsafe_from_address=gate_up_base + local_idx * gate_up_stride)
        var wscale = F32Ptr(unsafe_from_address=gate_up_sc_base + local_idx * gate_up_sc_stride)
        var wcolsum = F32Ptr(unsafe_from_address=gate_up_cs_base + local_idx * gate_up_cs_stride)
        var qi_out = expert_qi_p + li * intermediate
        var blk_out = expert_blk_scale_p + li * n_tiles

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
                act_i8_p, act_scale_p,
                wpacked, wscale, wcolsum,
                qi_out + n_start,
                blk_out + tile_start,
                n_start, n_count, 1)
            num_jobs += 1

    pool.dispatch[FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker[intermediate, hidden, fwht_blk]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


def gemma4_moe_phase2[
    QiT: DynamicTensor, BScT: DynamicTensor,
    DnT: StaticTensor, DnScT: StaticTensor, DnCsT: StaticTensor,
    OutT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    hidden: Int, intermediate: Int, fwht_blk: Int,
    top_k: Int, num_experts: Int, tp: Int,
](
    expert_qi: QiT,
    expert_blk_scale: BScT,
    routing: Gemma4TopKResult[top_k],
    down: DnT, down_stride: Int,
    down_sc: DnScT, down_sc_stride: Int,
    down_bcs: DnCsT, down_bcs_stride: Int,
    expert_out_buf: OutT,
    rank: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Multi-expert phase 2: per-local-expert down GEMV with routing weight.

    One job per local expert: int8_gemv_blocked_worker reading the expert's
    expert_qi slot (written by phase 1), with output_scale = routing.weights[s]
    so the routing scale is folded into the bf16 cast for free.
    """
    comptime assert QiT.DTYPE == DType.int8, "moe_phase2: expert_qi must be i8"
    comptime assert BScT.DTYPE == DType.float32, "moe_phase2: expert_blk_scale must be f32"
    comptime assert DnT.DTYPE == DType.int8, "moe_phase2: down must be i8"
    comptime assert DnScT.DTYPE == DType.float32, "moe_phase2: down_sc must be f32"
    comptime assert DnCsT.DTYPE == DType.float32, "moe_phase2: down_bcs must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "moe_phase2: expert_out_buf must be bf16"

    comptime num_blocks = intermediate // fwht_blk
    comptime experts_per_rank = num_experts // tp
    var expert_base = rank * experts_per_rank

    var expert_qi_p = expert_qi.as_ptr[DType.int8]()
    var expert_blk_scale_p = expert_blk_scale.as_ptr[DType.float32]()
    var down_base = down.addr()
    var down_sc_base = down_sc.addr()
    var down_bcs_base = down_bcs.addr()
    var expert_out_buf_p = expert_out_buf.as_ptr[DType.bfloat16]()

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())

    var local_count = 0
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid < expert_base or eid >= expert_base + experts_per_rank:
            continue
        var local_idx = eid - expert_base

        jobs[local_count] = Int8GemvBlockedArgs(
            expert_qi_p + local_count * intermediate,
            I8Ptr(unsafe_from_address=down_base + local_idx * down_stride),
            expert_blk_scale_p + local_count * num_blocks,
            F32Ptr(unsafe_from_address=down_sc_base + local_idx * down_sc_stride),
            F32Ptr(unsafe_from_address=down_bcs_base + local_idx * down_bcs_stride),
            expert_out_buf_p + local_count * hidden,
            routing.weights[s], hidden, hidden, 1,
        )
        local_count += 1

    if local_count == 0:
        return PoolFence[P, origin].over(pool)

    pool.dispatch[Int8GemvBlockedArgs, int8_gemv_blocked_worker[hidden, intermediate, fwht_blk]](
        UnsafePointer(to=jobs[0]), local_count)
    return PoolFence[P, origin].over(pool)


def router_topk_dispatch[
    LgT: DynamicTensor, PesT: StaticTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    num_experts: Int, k: Int,
](
    logits: LgT, per_expert_scale: PesT,
    result_ptr: UnsafePointer[Gemma4TopKResult[k], MutAnyOrigin],
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert LgT.DTYPE == DType.bfloat16, "router_topk: logits must be bf16"
    comptime assert PesT.DTYPE == DType.bfloat16, "router_topk: per_expert_scale must be bf16"
    var args = RouterTopkArgs[k](
        logits.as_ptr[DType.bfloat16](),
        per_expert_scale.as_ptr[DType.bfloat16](),
        result_ptr)
    pool.dispatch[RouterTopkArgs[k], router_topk_kernel[num_experts, k]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Sliding attention — per-KV-group worker
# ============================================================================


def sliding_attn_dispatch[
    QT: DynamicTensor, KT: DynamicTensor, VT: DynamicTensor,
    QnT: StaticTensor, KnT: StaticTensor,
    CosT: StaticTensor, SinT: StaticTensor,
    QiT: DynamicTensor, HsT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, heads_per_group: Int, window_size: Int,
    num_kv_heads: Int, num_q_heads: Int,
](
    q: QT, k: KT, v: VT,
    q_norm: QnT, k_norm: KnT,
    cos_table: CosT, sin_table: SinT,
    cache_base: Int, cache_pos: Int, context_len: Int,
    qi_out: QiT, head_scale: HsT,
    eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert QT.DTYPE == DType.bfloat16, "sliding_attn: q must be bf16"
    comptime assert KT.DTYPE == DType.bfloat16, "sliding_attn: k must be bf16"
    comptime assert VT.DTYPE == DType.bfloat16, "sliding_attn: v must be bf16"
    comptime assert QnT.DTYPE == DType.bfloat16, "sliding_attn: q_norm must be bf16"
    comptime assert KnT.DTYPE == DType.bfloat16, "sliding_attn: k_norm must be bf16"
    comptime assert CosT.DTYPE == DType.float32, "sliding_attn: cos must be f32"
    comptime assert SinT.DTYPE == DType.float32, "sliding_attn: sin must be f32"
    comptime assert QiT.DTYPE == DType.int8, "sliding_attn: qi_out must be i8"
    comptime assert HsT.DTYPE == DType.float32, "sliding_attn: head_scale must be f32"

    comptime NKV = num_kv_heads
    var jobs = InlineArray[AttnGroupArgs, 8](fill=AttnGroupArgs())
    var q_p = q.as_ptr[DType.bfloat16]()
    var k_p = k.as_ptr[DType.bfloat16]()
    var v_p = v.as_ptr[DType.bfloat16]()
    var q_norm_p = q_norm.as_ptr[DType.bfloat16]()
    var k_norm_p = k_norm.as_ptr[DType.bfloat16]()
    var cos_p = cos_table.as_ptr[DType.float32]()
    var sin_p = sin_table.as_ptr[DType.float32]()
    var cache = U8Ptr(unsafe_from_address=cache_base)
    var qi_out_p = qi_out.as_ptr[DType.int8]()
    var head_scale_p = head_scale.as_ptr[DType.float32]()
    for g in range(NKV):
        jobs[g] = AttnGroupArgs(
            q_p + g * heads_per_group * head_dim,
            k_p + g * head_dim,
            v_p + g * head_dim,
            q_norm_p, k_norm_p,
            cos_p, sin_p,
            cache, g,
            cache_pos, context_len,
            qi_out_p + g * heads_per_group * head_dim,
            head_scale_p + g * heads_per_group,
            eps)
    pool.dispatch[AttnGroupArgs,
        sliding_attn_group_kernel[head_dim, heads_per_group,
            window_size, num_kv_heads, num_q_heads]](
        UnsafePointer(to=jobs[0]), NKV)
    return PoolFence[P, origin].over(pool)


# ============================================================================
# Context-parallel full attention — prep, chunked score, merge, gather
# ============================================================================


def cp_attn_prep_dispatch[
    QT: DynamicTensor, KT: DynamicTensor,
    QnT: StaticTensor, KnT: StaticTensor,
    CosT: StaticTensor, SinT: StaticTensor,
    QiT: DynamicTensor, QbT: DynamicTensor, QsT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    local_max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](
    q_bf16: QT, k_bf16: KT,
    q_norm: QnT, k_norm: KnT,
    cos_table: CosT, sin_table: SinT,
    cache_base: Int, local_pos: Int, kv_head: Int,
    eps: Float32, write_kv: Bool,
    q_i8_out: QiT, qi_biases_out: QbT, q_scales_out: QsT,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert QT.DTYPE == DType.bfloat16, "cp_attn_prep: q_bf16 must be bf16"
    comptime assert KT.DTYPE == DType.bfloat16, "cp_attn_prep: k_bf16 must be bf16"
    comptime assert QnT.DTYPE == DType.bfloat16, "cp_attn_prep: q_norm must be bf16"
    comptime assert KnT.DTYPE == DType.bfloat16, "cp_attn_prep: k_norm must be bf16"
    comptime assert CosT.DTYPE == DType.float32, "cp_attn_prep: cos must be f32"
    comptime assert SinT.DTYPE == DType.float32, "cp_attn_prep: sin must be f32"
    comptime assert QiT.DTYPE == DType.int8, "cp_attn_prep: q_i8_out must be i8"
    comptime assert QbT.DTYPE == DType.float32, "cp_attn_prep: qi_biases_out must be f32"
    comptime assert QsT.DTYPE == DType.float32, "cp_attn_prep: q_scales_out must be f32"
    var args = CpAttnPrepArgs(
        q_bf16_base=q_bf16.as_ptr[DType.bfloat16](),
        k_bf16_ptr=k_bf16.as_ptr[DType.bfloat16](),
        q_norm_ptr=q_norm.as_ptr[DType.bfloat16](),
        k_norm_ptr=k_norm.as_ptr[DType.bfloat16](),
        cos_ptr=cos_table.as_ptr[DType.float32](),
        sin_ptr=sin_table.as_ptr[DType.float32](),
        cache_base=U8Ptr(unsafe_from_address=cache_base),
        cache_pos=local_pos, kv_head=kv_head,
        eps=eps,
        q_i8_out=q_i8_out.as_ptr[DType.int8](),
        qi_biases_out=qi_biases_out.as_ptr[DType.float32](),
        q_scales_out=q_scales_out.as_ptr[DType.float32](),
        write_kv=Int32(1) if write_kv else Int32(0))
    pool.dispatch[CpAttnPrepArgs,
        cp_attn_prep_kernel[head_dim, rope_dims, heads_per_group,
            local_max_seq, num_kv_heads, num_q_heads]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


def cp_chunked_attn_dispatch[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, local_max_seq: Int,
    num_kv_heads: Int, num_q_heads: Int, heads_per_group: Int,
    max_attn_chunks: Int,
](
    q_i8_base: Int, qi_biases_base: Int, q_scales_base: Int,
    cache_base: Int, kv_head: Int,
    local_context_len: Int, pool_capacity: Int,
    partial_out_base: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dispatch chunked attention with CP cache parameters."""
    var num_pg = (local_context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var num_chunks = min(pool_capacity, max_attn_chunks)
    if num_chunks > num_pg:
        num_chunks = num_pg
    if num_chunks <= 0:
        return PoolFence[P, origin].over(pool)
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
            context_len=local_context_len)
    pool.dispatch[ChunkedAttnArgs,
        cp_chunked_attn_kernel[head_dim, local_max_seq, num_kv_heads, num_q_heads, heads_per_group, max_attn_chunks]](
        UnsafePointer(to=chunk_args[0]), num_chunks)
    return PoolFence[P, origin].over(pool)


def merge_local_chunks_dispatch[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, heads_per_group: Int, max_attn_chunks: Int,
](
    partial_base: Int, num_chunks: Int,
    out_m: Int, out_l: Int, out_v: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    if num_chunks <= 0:
        return PoolFence[P, origin].over(pool)
    var args = MergeChunksArgs(
        partial_base=F32Ptr(unsafe_from_address=partial_base),
        num_chunks=num_chunks,
        out_m=F32Ptr(unsafe_from_address=out_m),
        out_l=F32Ptr(unsafe_from_address=out_l),
        out_v=F32Ptr(unsafe_from_address=out_v))
    pool.dispatch[MergeChunksArgs,
        merge_local_chunks_kernel[head_dim, heads_per_group, max_attn_chunks]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


def cp_gather_dispatch[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, num_heads: Int, tp: Int,
](
    rank: Int,
    all_m: InlineArray[Int, MAX_CP_RANKS],
    all_l: InlineArray[Int, MAX_CP_RANKS],
    all_v: InlineArray[Int, MAX_CP_RANKS],
    qi_out: Int, head_scales: Int,
    head_start: Int, head_count: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
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
    return PoolFence[P, origin].over(pool)


# ============================================================================
# RMSNorm family — FWHT+quantize, dual-gamma, bf16-out, post-*norm composites
# ============================================================================


def rmsnorm_fwht_quant[
    InT: DynamicTensor, GT: StaticTensor,
    QiT: DynamicTensor, WkT: DynamicTensor, ScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool,
](
    input: InT, gamma: GT, qi: QiT, work: WkT,
    scale: ScT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Unified FWHT+quantize dispatcher. Use convenience wrappers below."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_fwht_quant: input must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "rmsnorm_fwht_quant: qi must be i8"
    comptime assert WkT.DTYPE == DType.float32, "rmsnorm_fwht_quant: work must be f32"
    comptime assert ScT.DTYPE == DType.float32, "rmsnorm_fwht_quant: scale must be f32"

    var seq_len = input.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var inp = input.as_ptr[DType.bfloat16]()
    var gamma_p = BF16Ptr()
    comptime if has_gamma:
        comptime assert GT.DTYPE == DType.bfloat16, "rmsnorm_fwht_quant: gamma must be bf16"
        gamma_p = gamma.as_ptr[DType.bfloat16]()
    var qi_p = qi.as_ptr[DType.int8]()
    var work_p = work.as_ptr[DType.float32]()
    var scales = scale.as_ptr[DType.float32]()
    var jobs = InlineArray[RmsNormFwhtQuantArgs, MAX_POOL_CAPACITY](
        fill=RmsNormFwhtQuantArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtQuantArgs(
            inp, gamma_p, qi_p,
            work_p + i * cols,
            scales, eps, start, end)

    pool.dispatch[RmsNormFwhtQuantArgs,
        rmsnorm_fwht_quant_worker[cols, block, has_gamma, per_block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


def rmsnorm_gamma_fwht_quantize[
    InT: DynamicTensor, GT: StaticTensor,
    QiT: DynamicTensor, WkT: DynamicTensor, ScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    cols: Int, block: Int,
](
    input: InT, gamma: GT, qi: QiT, work: WkT,
    scale: ScT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """RMSNorm * gamma + FWHT + per-row i8."""
    return rmsnorm_fwht_quant[cols, block, True, False](
        input, gamma, qi, work, scale, eps, pool)


def rmsnorm_gamma_fwht_per_block_quantize[
    InT: DynamicTensor, GT: StaticTensor,
    QiT: DynamicTensor, WkT: DynamicTensor, BsT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    cols: Int, block: Int,
](
    input: InT, gamma: GT, qi: QiT, work: WkT,
    blk_scale: BsT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """RMSNorm * gamma + FWHT + per-block i8."""
    return rmsnorm_fwht_quant[cols, block, True, True](
        input, gamma, qi, work, blk_scale, eps, pool)


def rmsnorm_fwht_quantize[
    InT: DynamicTensor,
    QiT: DynamicTensor, WkT: DynamicTensor, ScT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    cols: Int, block: Int,
](
    input: InT, qi: QiT, work: WkT,
    scale: ScT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """RMSNorm + FWHT + per-row i8. No gamma.

    Passes a dummy gamma view — has_gamma=False means the worker never
    dereferences it."""
    var dummy_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=0)
    var dummy_gamma = StaticView[BF16, Shape[1, 1]](dummy_ptr)
    return rmsnorm_fwht_quant[cols, block, False, False](
        input, dummy_gamma, qi, work, scale, eps, pool)


def rmsnorm_dual_gamma_fwht_quantize[
    InT: DynamicTensor, GaT: StaticTensor, GbT: StaticTensor,
    QiAT: DynamicTensor, QiBT: DynamicTensor,
    WkAT: DynamicTensor, WkBT: DynamicTensor,
    ScAT: DynamicTensor, ScBT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //,
    cols: Int, block: Int,
](
    input: InT,
    gamma_a: GaT, gamma_b: GbT,
    qi_a: QiAT, qi_b: QiBT,
    work_a: WkAT, work_b: WkBT,
    scale_a: ScAT, scale_b: ScBT,
    eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Dual-gamma RMSNorm + FWHT + per-row i8. One pass, two outputs."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_dual_gamma: input must be bf16"
    comptime assert GaT.DTYPE == DType.bfloat16, "rmsnorm_dual_gamma: gamma_a must be bf16"
    comptime assert GbT.DTYPE == DType.bfloat16, "rmsnorm_dual_gamma: gamma_b must be bf16"
    comptime assert QiAT.DTYPE == DType.int8, "rmsnorm_dual_gamma: qi_a must be i8"
    comptime assert QiBT.DTYPE == DType.int8, "rmsnorm_dual_gamma: qi_b must be i8"
    comptime assert WkAT.DTYPE == DType.float32, "rmsnorm_dual_gamma: work_a must be f32"
    comptime assert WkBT.DTYPE == DType.float32, "rmsnorm_dual_gamma: work_b must be f32"
    comptime assert ScAT.DTYPE == DType.float32, "rmsnorm_dual_gamma: scale_a must be f32"
    comptime assert ScBT.DTYPE == DType.float32, "rmsnorm_dual_gamma: scale_b must be f32"

    var seq_len = input.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var inp = input.as_ptr[DType.bfloat16]()
    var gamma_a_p = gamma_a.as_ptr[DType.bfloat16]()
    var gamma_b_p = gamma_b.as_ptr[DType.bfloat16]()
    var qi_a_p = qi_a.as_ptr[DType.int8]()
    var qi_b_p = qi_b.as_ptr[DType.int8]()
    var work_a_p = work_a.as_ptr[DType.float32]()
    var work_b_p = work_b.as_ptr[DType.float32]()
    var scale_a_p = scale_a.as_ptr[DType.float32]()
    var scale_b_p = scale_b.as_ptr[DType.float32]()
    var jobs = InlineArray[RmsNormDualGammaFwhtArgs, MAX_POOL_CAPACITY](
        fill=RmsNormDualGammaFwhtArgs())
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormDualGammaFwhtArgs(
            inp, gamma_a_p, gamma_b_p,
            qi_a_p, qi_b_p,
            work_a_p + i * cols,
            work_b_p + i * cols,
            scale_a_p, scale_b_p,
            eps, start, end)

    pool.dispatch[RmsNormDualGammaFwhtArgs,
        rmsnorm_dual_gamma_fwht_quant_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


def rmsnorm_no_scale[
    P: BurstThreadPool, origin: MutOrigin, //,
    InT: DynamicTensor,
    OutT: DynamicTensor,
](
    input: InT, output: OutT,
    ref [origin] pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P, origin]:
    """RMSNorm without learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_no_scale: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm_no_scale: cols must be f32-simd-aligned"

    var seq_len = input.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

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
    return PoolFence[P, origin].over(pool)


def rmsnorm_per_head[
    P: BurstThreadPool, origin: MutOrigin, //,
    head_dim: Int, num_heads: Int,
    W: StaticTensor,
    InT: DynamicTensor,
    OutT: DynamicTensor,
](
    input: InT, weight: W, output: OutT,
    ref [origin] pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P, origin] where W.DTYPE == DType.bfloat16:
    """Per-head RMSNorm with learnable scale via BurstPool."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_per_head: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_per_head: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_per_head: input/output cols mismatch"
    comptime assert InT.COLS == head_dim * num_heads, "rmsnorm_per_head: cols != heads * dim"
    comptime assert W.ROWS * W.COLS == head_dim, "rmsnorm_per_head: weight size != head_dim"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "rmsnorm_per_head: head_dim must be f32-simd-aligned"

    var seq_len = input.seq_len()
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

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
    return PoolFence[P, origin].over(pool)


def post_attn_norm_dispatch[
    SrcT: DynamicTensor, NwT: StaticTensor, XmT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //, hidden: Int,
](
    src: SrcT, norm_w: NwT, x_main: XmT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert SrcT.DTYPE == DType.bfloat16, "post_attn_norm: src must be bf16"
    comptime assert NwT.DTYPE == DType.bfloat16, "post_attn_norm: norm_w must be bf16"
    comptime assert XmT.DTYPE == DType.bfloat16, "post_attn_norm: x_main must be bf16"
    var args = PostAttnNormArgs(
        src.as_ptr[DType.bfloat16](),
        norm_w.as_ptr[DType.bfloat16](),
        x_main.as_ptr[DType.bfloat16](),
        eps)
    pool.dispatch[PostAttnNormArgs, post_attn_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


def expert_sum_dispatch[
    ExT: DynamicTensor, DstT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //, hidden: Int, max_local: Int,
](
    expert_out: ExT, local_count: Int, dst: DstT, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert ExT.DTYPE == DType.bfloat16, "expert_sum: expert_out must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "expert_sum: dst must be bf16"
    var args = ExpertSumArgs(
        expert_out.as_ptr[DType.bfloat16](),
        local_count,
        dst.as_ptr[DType.bfloat16]())
    pool.dispatch[ExpertSumArgs, expert_sum_kernel[hidden, max_local]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


def dense_norm_dispatch[
    SrcT: DynamicTensor, NwT: StaticTensor, DstT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //, hidden: Int,
](
    src: SrcT, norm_w: NwT, dst: DstT, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert SrcT.DTYPE == DType.bfloat16, "dense_norm: src must be bf16"
    comptime assert NwT.DTYPE == DType.bfloat16, "dense_norm: norm_w must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "dense_norm: dst must be bf16"
    var args = DenseNormArgs(
        src.as_ptr[DType.bfloat16](),
        norm_w.as_ptr[DType.bfloat16](),
        dst.as_ptr[DType.bfloat16](),
        eps)
    pool.dispatch[DenseNormArgs, dense_norm_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)


def post_reduce_dispatch[
    MoeT: DynamicTensor, MnT: StaticTensor,
    DnT: DynamicTensor, CnT: StaticTensor, XmT: DynamicTensor,
    P: BurstThreadPool, origin: MutOrigin, //, hidden: Int,
](
    moe_out: MoeT, moe_norm_w: MnT, dense_normed: DnT,
    combine_norm_w: CnT, x_main: XmT,
    layer_scalar: Float32, eps: Float32, ref [origin] pool: P,
) -> PoolFence[P, origin]:
    comptime assert MoeT.DTYPE == DType.bfloat16, "post_reduce: moe_out must be bf16"
    comptime assert MnT.DTYPE == DType.bfloat16, "post_reduce: moe_norm_w must be bf16"
    comptime assert DnT.DTYPE == DType.bfloat16, "post_reduce: dense_normed must be bf16"
    comptime assert CnT.DTYPE == DType.bfloat16, "post_reduce: combine_norm_w must be bf16"
    comptime assert XmT.DTYPE == DType.bfloat16, "post_reduce: x_main must be bf16"
    var args = PostReduceArgs(
        moe_out.as_ptr[DType.bfloat16](),
        moe_norm_w.as_ptr[DType.bfloat16](),
        dense_normed.as_ptr[DType.bfloat16](),
        combine_norm_w.as_ptr[DType.bfloat16](),
        x_main.as_ptr[DType.bfloat16](),
        layer_scalar, eps)
    pool.dispatch[PostReduceArgs, post_reduce_kernel[hidden]](
        UnsafePointer(to=args), 1)
    return PoolFence[P, origin].over(pool)
