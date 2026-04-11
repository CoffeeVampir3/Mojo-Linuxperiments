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

Expert weights are distributed round-robin (expert e on rank e % tp, local
index e // tp). All memory is rank-local. Phase 1 inputs (act_i8/act_scale)
must be produced by the rank-local rmsnorm_dual_gamma_fwht_quantize.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from kernels.kernel_ops import PoolFence
from simd_math import exp_f32, sqrt, roundeven
from experimental2.kernels.int8_gemv import dot
from experimental3.kernels.dense_ffn import (
    FusedGuGeluTanhArgs, fused_gu_gelu_tanh_worker,
    Int8GemvBlockedArgs, int8_gemv_blocked_worker,
)
from experimental_gemma.router import Gemma4TopKResult

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


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
    dequants to f32 per block, applies weight scale at the end."""
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
    top_k: Int, tp: Int, P: BurstThreadPool,
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

    Iterates routing.indices, filters by `eid % tp == rank`, and for each local
    expert builds N-tile-sharded `FusedGuGeluTanhArgs` jobs targeting that
    expert's slot in `expert_qi` / `expert_blk_scale`. One pool.dispatch with
    `local_count * tiles_per_expert` jobs total. Same fence shape as
    fused_gu_gelu_tanh — caller joins via timed_parallel or via discard+drain.
    """
    comptime n_tiles = intermediate // fwht_blk
    comptime MAX_POOL_CAPACITY = 128

    var pool_capacity = pool.get_capacity()

    # Count local experts first to decide workers per expert.
    var local_count = 0
    var local_slots = InlineArray[Int, top_k](fill=0)
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid % tp == rank:
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
        var local_idx = eid // tp

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
    top_k: Int, tp: Int, P: BurstThreadPool,
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
    `expert_qi` slot (written by phase 1), with `output_scale = routing.weights[s]`
    so the routing scale is folded into the bf16 cast for free.
    """
    comptime num_blocks = intermediate // fwht_blk
    comptime MAX_POOL_CAPACITY = 128

    var zero_args = Int8GemvBlockedArgs(
        I8Ptr(), U8Ptr(), F32Ptr(), F32Ptr(), F32Ptr(), BF16Ptr(), Float32(0))
    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](fill=zero_args)

    var local_count = 0
    for s in range(top_k):
        var eid = routing.indices[s]
        if eid % tp != rank:
            continue
        var local_idx = eid // tp

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


# ============================================================================
# Validation
# ============================================================================


def scalar_fwht_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var s = 1
    while s < n:
        var i = 0
        while i < n:
            for j in range(s):
                var a = buf[i + j]
                var b = buf[i + j + s]
                buf[i + j] = a + b
                buf[i + j + s] = a - b
            i += s * 2
        s *= 2
    var sc = 1.0 / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] *= sc


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def validate_expert_pipeline():
    """Validate one expert's full i8 pipeline against scalar f64 reference."""
    comptime intermediate = 704
    comptime hidden = 2816
    comptime gate_up_dim = 1408
    comptime fwht_blk = 64
    comptime num_blocks = intermediate // fwht_blk

    var rng = UInt64(0xCAFEDEAD12345678)

    var act_i8 = alloc[Scalar[DType.int8]](hidden)
    var act_scale = Float32(15.0)
    for i in range(hidden):
        act_i8[i] = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))

    var gate_up_w = alloc[Float32](gate_up_dim * hidden)
    var down_w = alloc[Float32](hidden * intermediate)
    for i in range(gate_up_dim * hidden):
        gate_up_w[i] = Float32(xorshift64(rng) * 0.01)
    for i in range(hidden * intermediate):
        down_w[i] = Float32(xorshift64(rng) * 0.01)

    var routing_weight = Float32(0.15)
    var act_dequant = Float64(act_scale) / 127.0

    # f64 reference: gate_up → gelu_tanh → FWHT+quantize → dequant → down
    var gu_f64 = alloc[Float64](gate_up_dim)
    for n in range(gate_up_dim):
        var acc = Float64(0)
        for k in range(hidden):
            acc += Float64(Int(act_i8[k])) * act_dequant * Float64(gate_up_w[n * hidden + k])
        gu_f64[n] = acc

    var activated = alloc[Float64](intermediate)
    for i in range(intermediate):
        var g = gu_f64[i]
        var inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        var e = Float64(exp_f32[1](Float32(-2.0 * inner)))
        activated[i] = 0.5 * g * (1.0 + (1.0 - e) / (1.0 + e)) * gu_f64[intermediate + i]

    # FWHT + per-block quantize round-trip
    var fwht_buf = alloc[Float64](intermediate)
    for i in range(intermediate):
        fwht_buf[i] = activated[i]

    for b in range(num_blocks):
        scalar_fwht_f64(fwht_buf + b * fwht_blk, fwht_blk)

    var qi = alloc[Scalar[DType.int8]](intermediate)
    var bsc = alloc[Float64](num_blocks)
    for b in range(num_blocks):
        var bmax = Float64(0)
        for j in range(fwht_blk):
            var a = fwht_buf[b * fwht_blk + j].__abs__()
            if a > bmax:
                bmax = a
        if bmax < 1e-10:
            bmax = 1e-10
        bsc[b] = bmax
        var inv = 127.0 / bmax
        for j in range(fwht_blk):
            qi[b * fwht_blk + j] = Scalar[DType.int8](Int(Float64(roundeven[DType.float64, 1](fwht_buf[b * fwht_blk + j] * inv)).clamp(-128.0, 127.0)))

    var dequant = alloc[Float64](intermediate)
    for b in range(num_blocks):
        var dq = bsc[b] / 127.0
        for j in range(fwht_blk):
            dequant[b * fwht_blk + j] = Float64(Int(qi[b * fwht_blk + j])) * dq
    for b in range(num_blocks):
        scalar_fwht_f64(dequant + b * fwht_blk, fwht_blk)

    # Down matmul
    var output_quant = alloc[Float64](hidden)
    var output_exact = alloc[Float64](hidden)
    for n in range(hidden):
        var acc_q = Float64(0)
        var acc_e = Float64(0)
        for k in range(intermediate):
            acc_q += dequant[k] * Float64(down_w[n * intermediate + k])
            acc_e += activated[k] * Float64(down_w[n * intermediate + k])
        output_quant[n] = acc_q * Float64(routing_weight)
        output_exact[n] = acc_e * Float64(routing_weight)

    # Report
    var dot_val = Float64(0)
    var n_e = Float64(0)
    var n_q = Float64(0)
    var sq_err = Float64(0)
    for i in range(hidden):
        dot_val += output_exact[i] * output_quant[i]
        n_e += output_exact[i] * output_exact[i]
        n_q += output_quant[i] * output_quant[i]
        sq_err += (output_exact[i] - output_quant[i]) * (output_exact[i] - output_quant[i])

    print("  output RMS:    " + String(Float64(sqrt[DType.float64, 1](n_e / Float64(hidden)))))
    print("  cosine:        " + String(dot_val / (Float64(sqrt[DType.float64, 1](n_e)) * Float64(sqrt[DType.float64, 1](n_q)))))
    print("  NRMSE:         " + String(Float64(sqrt[DType.float64, 1](sq_err / n_e))))

    act_i8.free()
    gate_up_w.free()
    down_w.free()
    gu_f64.free()
    activated.free()
    fwht_buf.free()
    qi.free()
    bsc.free()
    dequant.free()
    output_quant.free()
    output_exact.free()


def main():
    print("=== Gemma4 MoE expert pipeline validation ===")
    print("\nSingle expert i8 pipeline (gate_up → gelu_tanh → FWHT → down):")
    validate_expert_pipeline()
