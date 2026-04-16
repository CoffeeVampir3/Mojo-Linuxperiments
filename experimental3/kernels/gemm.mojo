"""GEMM-shaped compute cores — args structs and pool-worker bodies.

Dispatchers for these workers live in dispatch_kernels.mojo. The workers here
take an Args struct from the pool and run one unit of compute (one row-range
of a seq_len × N × K problem, or one N-tile of a decode problem).

When seq_len == 1 the dispatch collapses to a single decode GEMV. When
seq_len > 1 (prompt prefill) the dispatch partitions rows across pool
workers, and each worker walks the per-row gemv.mojo kernels. That inner
per-row walk is the AMX-lift surface: replacing gemv_row / gemv_row_blocked
with tile-based AMX kernels here will give prefill throughput without
touching decode.

Also hosts the lm_head workers (lm_head_worker, lm_head_flash_worker) and the
post-fence `lm_head_flash_reduce`. Structurally M=1 always; they N-parallelize
the same dot across workers.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.random.philox import Random

from simd_math import log_f32
from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import vpdpbusd
from experimental3.kernels.gemv import (
    gemv_row, gemv_row_blocked, gemv_row_blocked_wa, lm_head_row_dot,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32, tanh_f32
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr


# ============================================================================
# int8_gemv worker
# ============================================================================


@fieldwise_init
struct WorkerConfig(Copyable, ImplicitlyCopyable):
    var act_ptr: Int
    var wpacked_ptr: Int
    var colsum_ptr: Int
    var weight_scale_ptr: Int
    var dst_ptr: Int
    var act_scale_ptr: Int
    var start_row: Int
    var row_count: Int


def int8_gemv_worker[N: Int, K: Int](cfg: WorkerConfig):
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=cfg.act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cfg.wpacked_ptr)
    var colsum = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.colsum_ptr)
    var wscale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.weight_scale_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=cfg.dst_ptr)
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.act_scale_ptr)
    var start = cfg.start_row

    for m in range(cfg.row_count):
        var act_dequant = act_scales[start + m] / Float32(127)
        gemv_row[N, K, DType.bfloat16](
            act + m * K, wpacked, act_dequant, wscale, colsum, dst + m * N,
        )


# ============================================================================
# int8_gemv_blocked workers (standard + _wa variant)
# ============================================================================


@fieldwise_init
struct Int8GemvBlockedArgs(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var wpacked: U8Ptr
    var blk_scale: F32Ptr
    var wscale: F32Ptr
    var blk_colsum: F32Ptr
    var dst: BF16Ptr
    var output_scale: Float32


def int8_gemv_blocked_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    """Down GEMV with per-block activation scales. Output bf16[N] * output_scale."""
    comptime width = simd_width_of[DType.float32]()

    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= N:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


def int8_gemv_blocked_wa_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    comptime width = simd_width_of[DType.float32]()
    var dst_buf = InlineArray[Float32, N](fill=Float32(0))
    var dp = UnsafePointer(to=dst_buf).bitcast[Float32]()
    gemv_row_blocked_wa[N, K, fwht_blk](args.act, args.wpacked, args.blk_scale,
        args.wscale, args.blk_colsum, dp)

    var scale = SIMD[DType.float32, width](args.output_scale)
    var k = 0
    while k + width <= N:
        (args.dst + k).store(((dp + k).load[width=width]() * scale).cast[DType.bfloat16]())
        k += width


# ============================================================================
# fused_gu_gelu_tanh workers (standard + _wa variant)
# ============================================================================


@fieldwise_init
struct FusedGuGeluTanhArgs(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var wpacked: U8Ptr
    var wscale: F32Ptr
    var wcolsum: F32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var n_count: Int
    var row_count: Int


def fused_gu_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int,
    fwht: Bool = True](
    args: FusedGuGeluTanhArgs,
):
    """N-tiled gate+up GEMV -> GELU-tanh -> [FWHT] -> per-block i8.

    Processes row_count activation rows, each over N-range [n_start, n_start + n_count).
    Tiles the N-range in fwht_blk-sized chunks. Each tile does:
      gate GEMV[fwht_blk, K] + up GEMV[fwht_blk, K] -> gelu_tanh -> [FWHT] -> i8.
    When fwht=False, skips the FWHT rotation (channelwise quantization).
    Activation must be pre-quantized by caller.
    Fused weight is [2*intermediate, K]: gate = [0:intermediate], up = [intermediate:].
    """
    debug_assert(intermediate % fwht_blk == 0,
        "fused_gu_gelu_tanh: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "fused_gu_gelu_tanh: K must be a multiple of 64 (VNNI_K_STEP)")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.wpacked + n_off * K,
                dequant,
                args.wscale + n_off,
                args.wcolsum + n_off,
                gate.bitcast[Scalar[DType.float32]]())

            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.wpacked + (intermediate + n_off) * K,
                dequant,
                args.wscale + intermediate + n_off,
                args.wcolsum + intermediate + n_off,
                up.bitcast[Scalar[DType.float32]]())

            var k = 0
            while k + width <= fwht_blk:
                var g = (gate + k).load[width=width]()
                var u = (up + k).load[width=width]()
                (gate + k).store(gelu_tanh_f32[width](g) * u)
                k += width

            comptime if fwht:
                fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)

            local_n += fwht_blk


# GEMV_TILE is shared with dispatch_kernels.mojo — the _wa worker uses the
# VNNI_N_STEP as its GEMV tile size while the outer dispatch iterates sub-blocks.
comptime GEMV_TILE = VNNI_N_STEP


def fused_gu_gelu_tanh_worker_wa[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    """Workaround: decouples GEMV tile (VNNI_N_STEP) from FWHT block (fwht_blk).

    Tiles the gate/up GEMV at GEMV_TILE for VNNI compatibility, then applies
    FWHT and quantize in fwht_blk-sized sub-blocks within each tile.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "intermediate must be a multiple of fwht_blk")
    debug_assert(GEMV_TILE % fwht_blk == 0,
        "GEMV_TILE must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0, "K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime sub_blocks_per_tile = GEMV_TILE // fwht_blk
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + n_off * K, dequant,
                args.wscale + n_off, args.wcolsum + n_off,
                gate.bitcast[Scalar[DType.float32]]())

            var up_buf = InlineArray[Float32, GEMV_TILE](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[GEMV_TILE, K, DType.float32](
                act_i8, args.wpacked + (intermediate + n_off) * K, dequant,
                args.wscale + intermediate + n_off,
                args.wcolsum + intermediate + n_off,
                up.bitcast[Scalar[DType.float32]]())

            var k = 0
            while k + width <= GEMV_TILE:
                var g = (gate + k).load[width=width]()
                var u = (up + k).load[width=width]()
                (gate + k).store(gelu_tanh_f32[width](g) * u)
                k += width

            for sb in range(sub_blocks_per_tile):
                var sb_off = sb * fwht_blk
                fwht_block[fwht_blk](gate + sb_off)
                blk_row[(local_n + sb_off) // fwht_blk] = absmax_quantize_i8[fwht_blk](
                    gate + sb_off, qi_row + local_n + sb_off)

            local_n += GEMV_TILE


# ============================================================================
# lm_head worker — decode-only N-parallel scan
# ============================================================================


@fieldwise_init
struct LmHeadArgs(Copyable, ImplicitlyCopyable):
    var act: Int
    var weight: Int
    var act_blk_scales: Int
    var w_blk_scales: Int
    var w_blk_colsums: Int
    var dst: Int
    var n_start: Int
    var n_count: Int


def lm_head_worker[K: Int, fwht_blk: Int](args: LmHeadArgs):
    comptime num_blocks = K // fwht_blk
    var act = I8Ptr(unsafe_from_address=args.act)
    var weight = I8Ptr(unsafe_from_address=args.weight)
    var act_blk_scales = F32Ptr(unsafe_from_address=args.act_blk_scales)
    var w_blk_scales = F32Ptr(unsafe_from_address=args.w_blk_scales)
    var w_blk_colsums = F32Ptr(unsafe_from_address=args.w_blk_colsums)
    var dst = BF16Ptr(unsafe_from_address=args.dst)

    for n in range(args.n_start, args.n_start + args.n_count):
        var weight_row = weight + n * K
        var w_blk_scales_row = w_blk_scales + n * num_blocks
        var w_blk_colsums_row = w_blk_colsums + n * num_blocks
        var dot_f32 = lm_head_row_dot[K, fwht_blk](
            act, weight_row,
            act_blk_scales,
            w_blk_scales_row,
            w_blk_colsums_row)
        dst[n] = dot_f32.cast[DType.bfloat16]()


# ============================================================================
# lm_head_flash — fused decode GEMV + softcap + Gumbel-Max sampling
# ============================================================================


# 23-bit uniform: u_int in [1, 2^23], u = u_int / (2^23 + 1) strictly in (0, 1).
comptime GUMBEL_BITS = 23
comptime GUMBEL_SHIFT = 32 - GUMBEL_BITS
comptime GUMBEL_DENOM_INT = (1 << GUMBEL_BITS) + 1

comptime MAX_POOL_CAPACITY = 128


@fieldwise_init
struct LmHeadCandidates[N: Int](Copyable, ImplicitlyCopyable, Movable):
    var scores: InlineArray[Float32, Self.N]
    var indices: InlineArray[Int32, Self.N]


comptime LmHeadCands = LmHeadCandidates[MAX_POOL_CAPACITY]
comptime LmHeadCandPtr = UnsafePointer[LmHeadCands, MutAnyOrigin]


@always_inline
def softcap_simd[W: Int](
    x: SIMD[DType.float32, W], cap: Float32
) -> SIMD[DType.float32, W]:
    var inv_cap = SIMD[DType.float32, W](1.0 / cap)
    var cap_v = SIMD[DType.float32, W](cap)
    return tanh_f32[W](x * inv_cap) * cap_v


@always_inline
def gumbel_simd[W: Int](
    raw: SIMD[DType.uint32, W]
) -> SIMD[DType.float32, W]:
    var shifted = raw >> GUMBEL_SHIFT
    var u_int = shifted + SIMD[DType.uint32, W](1)
    var u = u_int.cast[DType.float32]() / SIMD[DType.float32, W](Float32(GUMBEL_DENOM_INT))
    var neg_log_u = -log_f32[W](u)
    return -log_f32[W](neg_log_u)


@always_inline
def softcap_scalar(x: Float32, cap: Float32) -> Float32:
    return softcap_simd[1](SIMD[DType.float32, 1](x), cap)[0]


@always_inline
def gumbel_from_u32(raw_u32: Scalar[DType.uint32]) -> Float32:
    return gumbel_simd[1](SIMD[DType.uint32, 1](raw_u32))[0]


@fieldwise_init
struct LmHeadFlashArgs(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var weight: I8Ptr
    var act_blk_scales: F32Ptr
    var w_blk_scales: F32Ptr
    var w_blk_colsums: F32Ptr
    var candidates: LmHeadCandPtr
    var n_start: Int
    var n_count: Int
    var rng_key: UInt64
    var worker_idx: UInt64
    var softcap_val: Float32


def lm_head_flash_worker[K: Int, fwht_blk: Int](args: LmHeadFlashArgs):
    comptime num_blocks = K // fwht_blk
    comptime W = simd_width_of[DType.float32]()
    comptime PHILOX_STEPS = W // 4

    var rng = Random[rounds=10](
        seed=args.rng_key,
        subsequence=args.worker_idx,
        offset=UInt64(0),
    )

    var running_best = SIMD[DType.float32, W](Float32(-1.0e30))
    var running_idx = SIMD[DType.int32, W](-1)

    var n_start = args.n_start
    var n_stop = args.n_start + args.n_count

    var batch = n_start
    while batch < n_stop:
        var batch_logits = SIMD[DType.float32, W](Float32(-1.0e30))
        var batch_indices = SIMD[DType.int32, W](-1)

        comptime for lane in range(W):
            var n = batch + lane
            if n < n_stop:
                var logit = lm_head_row_dot[K, fwht_blk](
                    args.act,
                    args.weight + n * K,
                    args.act_blk_scales,
                    args.w_blk_scales + n * num_blocks,
                    args.w_blk_colsums + n * num_blocks,
                )
                batch_logits[lane] = logit
                batch_indices[lane] = Int32(n)

        var capped = softcap_simd[W](batch_logits, args.softcap_val)

        var raw_W = SIMD[DType.uint32, W](0)
        comptime for i in range(PHILOX_STEPS):
            raw_W = raw_W.insert[offset=i * 4](rng.step())

        var g = gumbel_simd[W](raw_W)
        var scores = capped + g

        var valid = batch_indices.ge(SIMD[DType.int32, W](0))
        var take = scores.gt(running_best) & valid
        running_best = take.select(scores, running_best)
        running_idx = take.select(batch_indices, running_idx)

        batch += W

    var best_val = running_best.reduce_max()
    var is_best = running_best.eq(SIMD[DType.float32, W](best_val))
    var cand_idx = is_best.select(running_idx, SIMD[DType.int32, W](Int32.MAX))
    var best_idx = cand_idx.reduce_min()

    var slot = Int(args.worker_idx)
    args.candidates[].scores[slot] = best_val
    args.candidates[].indices[slot] = best_idx


def lm_head_flash_reduce(candidates: LmHeadCandPtr) -> Int32:
    """Pick the global winner after the dispatcher fence joins."""
    comptime W = simd_width_of[DType.float32]()
    comptime CHUNKS = MAX_POOL_CAPACITY // W

    var scores_ptr = UnsafePointer(to=candidates[].scores[0])
    var indices_ptr = UnsafePointer(to=candidates[].indices[0])

    var running_best = SIMD[DType.float32, W](Float32(-1.0e30))
    var running_idx = SIMD[DType.int32, W](-1)

    comptime for c in range(CHUNKS):
        var s = (scores_ptr + c * W).load[width=W]()
        var i = (indices_ptr + c * W).load[width=W]()
        var take = s.gt(running_best)
        running_best = take.select(s, running_best)
        running_idx = take.select(i, running_idx)

    var best_val = running_best.reduce_max()
    var is_best = running_best.eq(SIMD[DType.float32, W](best_val))
    var cand = is_best.select(running_idx, SIMD[DType.int32, W](Int32.MAX))
    return cand.reduce_min()
