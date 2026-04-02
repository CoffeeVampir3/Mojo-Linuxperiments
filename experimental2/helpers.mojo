"""Shared attention helpers — structs and tile ops used by prefill and decode.

AttnCtx: shared context (Q/K/V pointers, scales, position).
pack_v_tile_vnni: i8 V rows → VNNI [K/4,N,4] via SIMD interleave.
amx_gemm_2x2: dtype-parameterized 2×2 tile GEMM with auto instruction select.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    tilezero, tileload, tilestore, tile_dp,
)


# ============================================================================
# Context — shared across all workers
# ============================================================================

@fieldwise_init
struct AttnCtx:
    var q: UnsafePointer[BFloat16, MutAnyOrigin]
    var cos: UnsafePointer[Float32, MutAnyOrigin]
    var sin: UnsafePointer[Float32, MutAnyOrigin]
    var k_base: UnsafePointer[UInt8, MutAnyOrigin]
    var v_base: UnsafePointer[Int8, MutAnyOrigin]
    var row_f32: UnsafePointer[Float32, MutAnyOrigin]
    var q_quant_inv: Float32
    var score_scale: Float32
    var vagg_scale: Float32
    var pos: Int
    var seq_len: Int


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
    var full_groups = n_pos // VNNI_BLK
    for kg in range(full_groups):
        var p = pos_off + kg * VNNI_BLK
        var r0 = (v_base + (p + 0) * head_dim + dim_off).load[width=TILE_N]()
        var r1 = (v_base + (p + 1) * head_dim + dim_off).load[width=TILE_N]()
        var r2 = (v_base + (p + 2) * head_dim + dim_off).load[width=TILE_N]()
        var r3 = (v_base + (p + 3) * head_dim + dim_off).load[width=TILE_N]()
        (dst + kg * VNNI_GROUP_BYTES).store(r0.interleave(r2).interleave(r1.interleave(r3)))
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
