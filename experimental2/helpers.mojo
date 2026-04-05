"""Shared attention helpers — structs and tile ops used by prefill and decode.

AttnCtx: shared context (Q/K/V pointers, scales, position).
pack_v_tile_vnni: i8 V rows → VNNI [K/4,N,4] via SIMD interleave.
amx_gemm_2x2: dtype-parameterized 2×2 tile GEMM with auto instruction select.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from simd_math import roundeven, exp_f32_fast, quantize_i8
from experimental.hadquant_impl import fwht_apply, fwht_width
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    tilezero, tileload, tilestore, tile_dp,
)


# ============================================================================
# Context — shared across all workers
# ============================================================================

@fieldwise_init
struct AttnCtx:
    """Pointers to long-lived allocations shared across all workers.
    Lives in scratch buffer — no per-dispatch temporaries."""
    var q: UnsafePointer[BFloat16, MutAnyOrigin]
    var cos: UnsafePointer[Float32, MutAnyOrigin]
    var sin: UnsafePointer[Float32, MutAnyOrigin]
    var k_base: UnsafePointer[UInt8, MutAnyOrigin]
    var v_base: UnsafePointer[Int8, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]
    var qi_output: UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
    var qi_scale: Float32


# ============================================================================
# Q row prep — RoPE → FWHT → quantize (one row)
# ============================================================================

@always_inline
def prep_q_row[head_dim: Int](
    q_row: UnsafePointer[BFloat16, MutAnyOrigin],
    cos_row: UnsafePointer[Float32, MutAnyOrigin],
    sin_row: UnsafePointer[Float32, MutAnyOrigin],
    q_quant_inv: Float32,
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
) -> Float32:
    """RoPE → FWHT → quantize one Q row. Returns qi_bias (128 * sum)."""
    comptime half = head_dim // 2
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    comptime fwht_regs = head_dim // fwht_w
    comptime half_regs = fwht_regs // 2
    var vq_inv = SIMD[DType.float32, fwht_w](q_quant_inv)

    var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
        fill=SIMD[DType.float32, fwht_w](0))
    for ri in range(half_regs):
        var j = ri * fwht_w
        var x_lo = (q_row + j).load[width=fwht_w]().cast[DType.float32]()
        var x_hi = (q_row + half + j).load[width=fwht_w]().cast[DType.float32]()
        var cv = (cos_row + j).load[width=fwht_w]()
        var sv = (sin_row + j).load[width=fwht_w]()
        r[ri] = x_lo * cv - x_hi * sv
        r[half_regs + ri] = x_hi * cv + x_lo * sv

    fwht_apply[DType.float32, head_dim](r)

    var q_sum_acc = SIMD[DType.int32, fwht_w](0)
    for ri in range(fwht_regs):
        var qi = quantize_i8[fwht_w](r[ri], vq_inv)
        (qi_out + ri * fwht_w).store(qi)
        q_sum_acc += qi.cast[DType.int32]()
    return Float32(q_sum_acc.reduce_add()) * Float32(128)


# ============================================================================
# Softmax row — max pass, correction, exp→u8, zero-pad (one row)
# ============================================================================

@always_inline
def softmax_row[head_dim: Int](
    si_row: UnsafePointer[Int32, MutAnyOrigin],
    q_bias: Float32,
    score_scale: Float32,
    causal_limit: Int,
    padded_chunk: Int,
    running_m: UnsafePointer[Float32, MutAnyOrigin],
    running_l: UnsafePointer[Float32, MutAnyOrigin],
    accum: UnsafePointer[Float32, MutAnyOrigin],
    w_row: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Online softmax for one Q row: max, correct accum, exp→u8."""
    comptime width = simd_width_of[DType.float32]()

    # Max pass
    var vmax = SIMD[DType.float32, width](Float32(-1e30))
    var t = 0
    while t + width <= causal_limit:
        var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bias) * score_scale
        vmax = max(vmax, dq)
        t += width
    var scalar_max = Float32(-1e30)
    while t < causal_limit:
        scalar_max = max(scalar_max, (Float32(si_row[t]) - q_bias) * score_scale)
        t += 1
    var row_max = max(vmax.reduce_max(), scalar_max)

    # Correction
    var m_old = running_m[]
    var m_new = max(m_old, row_max)
    running_m[] = m_new
    if m_new > m_old:
        var correction = exp_f32_fast[1](m_old - m_new)
        running_l[] = running_l[] * correction
        var d = 0
        while d + width <= head_dim:
            (accum + d).store((accum + d).load[width=width]() * correction)
            d += width

    # Fused dequant + exp → u8
    var l_acc = SIMD[DType.float32, width](0)
    t = 0
    while t + width <= causal_limit:
        var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bias) * score_scale
        var e = exp_f32_fast(dq - m_new)
        (w_row + t).store(roundeven(e * Float32(255)).cast[DType.uint8]())
        l_acc += e
        t += width
    var l_contrib = l_acc.reduce_add()
    while t < causal_limit:
        var dq = (Float32(si_row[t]) - q_bias) * score_scale
        var e = exp_f32_fast[1](dq - m_new)
        w_row[t] = roundeven[DType.float32, 1](e * Float32(255)).cast[DType.uint8]()
        l_contrib += e
        t += 1
    # Zero-pad
    comptime u8w = simd_width_of[DType.uint8]()
    comptime u8zeros = SIMD[DType.uint8, u8w](0)
    while t + u8w <= padded_chunk:
        (w_row + t).store(u8zeros)
        t += u8w
    running_l[] += l_contrib


# ============================================================================
# V VNNI pack — interleave i8 rows to [K/4,N,4]
# ============================================================================

@always_inline
def pack_v_tile_vnni(
    v_base: UnsafePointer[Int8, MutAnyOrigin], head_dim: Int,
    pos_off: Int, dim_off: Int, n_pos: Int,
    dst: UnsafePointer[Int8, MutAnyOrigin],
):
    """Pack V[pos,dim] i8 -> VNNI [K/4,N,4] via SIMD interleave."""
    comptime VNNI_GROUP_BYTES = TILE_N * VNNI_BLK
    comptime zeros = SIMD[DType.int8, VNNI_GROUP_BYTES](0)
    comptime zero_row = SIMD[DType.int8, TILE_N](0)
    var full_groups = n_pos // VNNI_BLK
    for kg in range(full_groups):
        var p = pos_off + kg * VNNI_BLK
        var r0 = (v_base + (p + 0) * head_dim + dim_off).load[width=TILE_N]()
        var r1 = (v_base + (p + 1) * head_dim + dim_off).load[width=TILE_N]()
        var r2 = (v_base + (p + 2) * head_dim + dim_off).load[width=TILE_N]()
        var r3 = (v_base + (p + 3) * head_dim + dim_off).load[width=TILE_N]()
        (dst + kg * VNNI_GROUP_BYTES).store(r0.interleave(r2).interleave(r1.interleave(r3)))
    var remainder = n_pos - full_groups * VNNI_BLK
    if remainder > 0:
        var p = pos_off + full_groups * VNNI_BLK
        var r0 = zero_row
        var r1 = zero_row
        var r2 = zero_row
        var r3 = zero_row
        if remainder >= 1:
            r0 = (v_base + (p + 0) * head_dim + dim_off).load[width=TILE_N]()
        if remainder >= 2:
            r1 = (v_base + (p + 1) * head_dim + dim_off).load[width=TILE_N]()
        if remainder >= 3:
            r2 = (v_base + (p + 2) * head_dim + dim_off).load[width=TILE_N]()
        (dst + full_groups * VNNI_GROUP_BYTES).store(r0.interleave(r2).interleave(r1.interleave(r3)))
        for kg in range(full_groups + 1, K_STEP // VNNI_BLK):
            (dst + kg * VNNI_GROUP_BYTES).store(zeros)
    else:
        for kg in range(full_groups, K_STEP // VNNI_BLK):
            (dst + kg * VNNI_GROUP_BYTES).store(zeros)


# ============================================================================
# 2x2 AMX tile GEMM — shared by score and V-agg
# ============================================================================

@always_inline
def amx_gemm_2x2[a_dtype: DType, b_dtype: DType, dst_n: Int](
    a0: UnsafePointer[Scalar[a_dtype], MutAnyOrigin],
    a1: UnsafePointer[Scalar[a_dtype], MutAnyOrigin],
    a_stride: Int,
    b0: UnsafePointer[Scalar[b_dtype], MutAnyOrigin],
    b1: UnsafePointer[Scalar[b_dtype], MutAnyOrigin],
    b_k_step: Int,
    k_iters: Int,
    dst: UnsafePointer[Int32, MutAnyOrigin],
):
    """Zero accumulators, K-reduce with 2x2 tile MACs, store.

    Tile layout: A[0,1] x B[2,3] -> C[4,5,6,7].
    Instruction selected from A/B dtypes (e.g. int8 x uint8 -> tdpbsud).
    """
    comptime dst_stride = dst_n * size_of[Int32]()
    tilezero[4](); tilezero[5](); tilezero[6](); tilezero[7]()
    for ki in range(k_iters):
        tileload[0](a0 + ki * K_STEP, a_stride)
        tileload[1](a1 + ki * K_STEP, a_stride)
        tileload[2](b0 + ki * b_k_step, TILE_N * VNNI_BLK)
        tileload[3](b1 + ki * b_k_step, TILE_N * VNNI_BLK)
        tile_dp[4, 0, 2, a_dtype, b_dtype]()
        tile_dp[5, 0, 3, a_dtype, b_dtype]()
        tile_dp[6, 1, 2, a_dtype, b_dtype]()
        tile_dp[7, 1, 3, a_dtype, b_dtype]()
    tilestore[4](dst, dst_stride)
    tilestore[5](dst + TILE_N, dst_stride)
    tilestore[6](dst + TILE_M * dst_n, dst_stride)
    tilestore[7](dst + TILE_M * dst_n + TILE_N, dst_stride)


# ============================================================================
# 1x3 AMX tile GEMM — optimized for decode (M=16, wide N)
# ============================================================================

@always_inline
def amx_gemm_1x3[a_dtype: DType, b_dtype: DType, dst_n: Int](
    a0: UnsafePointer[Scalar[a_dtype], MutAnyOrigin],
    a_stride: Int,
    b0: UnsafePointer[Scalar[b_dtype], MutAnyOrigin],
    b1: UnsafePointer[Scalar[b_dtype], MutAnyOrigin],
    b2: UnsafePointer[Scalar[b_dtype], MutAnyOrigin],
    b_k_step: Int,
    k_iters: Int,
    dst: UnsafePointer[Int32, MutAnyOrigin],
):
    """Zero accumulators, K-reduce with 1x3 tile MACs, store.

    Tile layout: A[0] x B[1,2,3] -> C[4,5,6].
    One A tile reused across 3 B tiles — 50% more N per A load vs 2x2.
    """
    comptime dst_stride = dst_n * size_of[Int32]()
    tilezero[4](); tilezero[5](); tilezero[6]()
    for ki in range(k_iters):
        tileload[0](a0 + ki * K_STEP, a_stride)
        tileload[1](b0 + ki * b_k_step, TILE_N * VNNI_BLK)
        tileload[2](b1 + ki * b_k_step, TILE_N * VNNI_BLK)
        tileload[3](b2 + ki * b_k_step, TILE_N * VNNI_BLK)
        tile_dp[4, 0, 1, a_dtype, b_dtype]()
        tile_dp[5, 0, 2, a_dtype, b_dtype]()
        tile_dp[6, 0, 3, a_dtype, b_dtype]()
    tilestore[4](dst, dst_stride)
    tilestore[5](dst + TILE_N, dst_stride)
    tilestore[6](dst + 2 * TILE_N, dst_stride)
