from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.sys import llvm_intrinsic
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbssd, tdpbf16ps,
)
from experimental3.kernels.gemv import gemv_row
from simd_math import set_subnormal_zeroing
from notstdcollections import AlignedInlineArray


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime SIMD_W = simd_width_of[DType.float32]()
comptime BF16_K_STEP = 32
comptime BF16_BLK = 2
comptime BF16_TILE_ELEMS = TILE_BYTES // 2
comptime BF16_TILE_ROW_ELEMS = TILE_N * BF16_BLK

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def alloc_zeroed[T: DType](count: Int) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
    var p = alloc[Scalar[T]](count, alignment=64)
    for i in range(count):
        (p + i)[] = Scalar[T](0)
    return UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=Int(p))


def fill_random_i8(ptr: I8Ptr, count: Int, seed: UInt64 = 0xDEADBEEF12345678):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = Scalar[DType.int8]((state >> 33).cast[DType.int8]())


def fill_random_f32(ptr: F32Ptr, count: Int, scale: Float32):
    var state = UInt64(0xFEEDFACE11111111)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var frac = Float32(Int(state >> 33)) / Float32(2147483648)
        ptr[i] = frac * scale + Float32(0.001)


def fill_random_bf16(ptr: BF16Ptr, count: Int, scale: Float32):
    var state = UInt64(0xA5A5BEEF22222222)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var frac = Float32(Int(state >> 33)) / Float32(2147483648)
        ptr[i] = (frac * scale + Float32(0.001)).cast[DType.bfloat16]()


# =============================================================================
# VNNI packing: [N, K] row-major i8 → VNNI layout for AMX B tiles.
#
# Output layout per (n_block=32, k_step=64):
#   2 subtiles × [16 K-groups × (16 N elements × 4 bytes)] = 2 × 1024 bytes.
#   Subtile base = n_block * K + k_step * N_STEP + sub * TILE_BYTES.
#   Tileload stride = K_STEP (= 64 bytes per K-group row).
# =============================================================================


def pack_vnni(src: I8Ptr, dst: I8Ptr, N: Int, K: Int):
    for nb in range(0, N, N_STEP):
        for kb in range(0, K, K_STEP):
            var tile_base = nb * K + kb * N_STEP
            for sub in range(2):
                var sub_base = tile_base + sub * TILE_BYTES
                for kg in range(TILE_M):
                    for n in range(TILE_N):
                        comptime for s in range(VNNI_BLK):
                            var src_row = nb + sub * TILE_N + n
                            var src_col = kb + kg * VNNI_BLK + s
                            dst[sub_base + kg * K_STEP + n * VNNI_BLK + s] = (
                                src[src_row * K + src_col])


def pack_bf16_amx(src: BF16Ptr, dst: BF16Ptr, N: Int, K: Int):
    for nb in range(0, N, N_STEP):
        for kb in range(0, K, BF16_K_STEP):
            var tile_base = nb * K + kb * N_STEP
            for sub in range(2):
                var sub_base = tile_base + sub * BF16_TILE_ELEMS
                for kg in range(TILE_M):
                    for n in range(TILE_N):
                        comptime for s in range(BF16_BLK):
                            var src_row = nb + sub * TILE_N + n
                            var src_col = kb + kg * BF16_BLK + s
                            dst[
                                sub_base + kg * BF16_TILE_ROW_ELEMS
                                + n * BF16_BLK + s
                            ] = src[src_row * K + src_col]


def compute_colsum(src: I8Ptr, colsum: F32Ptr, N: Int, K: Int):
    for n in range(N):
        var acc = Int32(0)
        for k in range(K):
            acc += Int32(src[n * K + k])
        colsum[n] = Float32(acc)


# =============================================================================
# AMX GEMM v1: i8 × i8 → i32 → f32 using tdpbssd (signed × signed).
#
# 2-2-4 tile config (same as prefill attention — no config switch needed):
#   TMM0, TMM1: A tiles (activation rows, 16×64 each)
#   TMM2, TMM3: B tiles (VNNI-packed weight, 16×64 each)
#   TMM4–TMM7:  C tiles (i32 accumulators, 16×16 each)
#
# Per M_STEP×N_STEP output block: tilezero C, sweep K, tilestore, dequant.
# Produces true signed dot product — no u8 bias trick, no colsum correction.
# =============================================================================


def amx_gemm[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
):
    comptime assert N % N_STEP == 0, "N must be a multiple of N_STEP"
    comptime assert K % K_STEP == 0, "K must be a multiple of K_STEP"

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, stride)
            tilestore[5, DType.int32](c_buf + 16, stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, stride)

            for m in range(m_count):
                var dq = act_scale[mb + m] / Float32(127)
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                    var f32_v = i32_v.cast[DType.float32]() * dq
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(f32_v * ws)

    _ = c_arr


def amx_gemm_bf16_out[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: BF16Ptr,
    M: Int,
):
    comptime assert N % N_STEP == 0, "N must be a multiple of N_STEP"
    comptime assert K % K_STEP == 0, "K must be a multiple of K_STEP"

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, stride)
            tilestore[5, DType.int32](c_buf + 16, stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, stride)

            for m in range(m_count):
                var dq = act_scale[mb + m] / Float32(127)
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                    var f32_v = i32_v.cast[DType.float32]() * dq
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(
                        (f32_v * ws).cast[DType.bfloat16]())

    _ = c_arr


def amx_bf16_gemm[N: Int, K: Int](
    act: BF16Ptr,
    wpacked: BF16Ptr,
    dst: F32Ptr,
    M: Int,
):
    comptime assert N % N_STEP == 0, "N must be a multiple of N_STEP"
    comptime assert K % BF16_K_STEP == 0, "K must be a multiple of 32"

    var c_arr = AlignedInlineArray[Float32, M_STEP * N_STEP](fill=Float32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, BF16_K_STEP):
                tileload[0, DType.bfloat16](act_mb + k, K * 2)
                tileload[1, DType.bfloat16](act_mb + 16 * K + k, K * 2)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.bfloat16](b_base, K_STEP)
                tileload[3, DType.bfloat16](
                    b_base + BF16_TILE_ELEMS, K_STEP)

                tdpbf16ps[4, 0, 2]()
                tdpbf16ps[5, 0, 3]()
                tdpbf16ps[6, 1, 2]()
                tdpbf16ps[7, 1, 3]()

            comptime stride = N_STEP * 4
            tilestore[4, DType.float32](c_buf, stride)
            tilestore[5, DType.float32](c_buf + 16, stride)
            tilestore[6, DType.float32](c_buf + 16 * N_STEP, stride)
            tilestore[7, DType.float32](c_buf + 16 * N_STEP + 16, stride)

            for m in range(m_count):
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    (dst + (mb + m) * N + nb + n).store(
                        (c_buf + row_off + n).load[width=SIMD_W]())

    _ = c_arr


# =============================================================================
# H1: Prefetch — prefetch next K step's B tiles during current compute.
#
# Hypothesis: B tiles stream from L3 (~40 cycle latency). The 4 tdpbssd ops
# per K step give ~8 cycles of compute. If we start B fetches early, we
# overlap fetch with compute and reduce stall cycles.
# =============================================================================


@always_inline
def prefetch_l2(p: I8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(2), Int32(1))


def amx_gemm_v2_prefetch[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
):
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                if k + K_STEP < K:
                    var nxt = wpacked + nb * K + (k + K_STEP) * N_STEP
                    for cl in range(0, TILE_BYTES * 2, 64):
                        prefetch_l2(nxt + cl)

                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, stride)
            tilestore[5, DType.int32](c_buf + 16, stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, stride)

            for m in range(m_count):
                var dq = act_scale[mb + m] / Float32(127)
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                    var f32_v = i32_v.cast[DType.float32]() * dq
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(f32_v * ws)

    _ = c_arr


# =============================================================================
# H2: K-outer loop — load A tiles once per K step, sweep N blocks inside.
#
# Hypothesis: A tiles get reloaded 48x per M block (once per N block).
# K-outer loads A once and reuses across all N blocks. Cost: tilezero +
# tilestore + SIMD f32 accumulate per (nb, k) since C tiles can't persist
# across N blocks. Tests whether A traffic reduction > SIMD accumulate cost.
# =============================================================================


def amx_gemm_v3_k_outer_scratch[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
    f32_acc: F32Ptr,
):
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for i in range(M_STEP * N):
            f32_acc[i] = Float32(0)

        for k in range(0, K, K_STEP):
            tileload[0, DType.int8](act_mb + k, K)
            tileload[1, DType.int8](act_mb + 16 * K + k, K)

            for nb in range(0, N, N_STEP):
                tilezero[4]()
                tilezero[5]()
                tilezero[6]()
                tilezero[7]()

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

                comptime stride = N_STEP * 4
                tilestore[4, DType.int32](c_buf, stride)
                tilestore[5, DType.int32](c_buf + 16, stride)
                tilestore[6, DType.int32](c_buf + 16 * N_STEP, stride)
                tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, stride)

                for m in range(M_STEP):
                    var src_off = m * N_STEP
                    var dst_off = m * N + nb
                    for n in range(0, N_STEP, SIMD_W):
                        var prev = (f32_acc + dst_off + n).load[width=SIMD_W]()
                        var cur = (c_buf + src_off + n).load[width=SIMD_W]()
                        (f32_acc + dst_off + n).store(
                            prev + cur.cast[DType.float32]())

        for m in range(m_count):
            var dq = act_scale[mb + m] / Float32(127)
            for n in range(0, N, SIMD_W):
                var acc_v = (f32_acc + m * N + n).load[width=SIMD_W]()
                var ws = (w_scale + n).load[width=SIMD_W]()
                (dst + (mb + m) * N + n).store(acc_v * dq * ws)

    _ = c_arr


def amx_gemm_v3_k_outer[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
):
    var f32_acc = alloc_zeroed[DType.float32](M_STEP * N)
    amx_gemm_v3_k_outer_scratch[N, K](
        act, wpacked, act_scale, w_scale, dst, M, f32_acc)
    f32_acc.free()


# =============================================================================
# H3: Non-temporal output stores — avoid write-allocate RFO on output buffer.
#
# Hypothesis: the dequant cost is dominated by RFO (read-for-ownership) misses
# on output stores, not the SIMD arithmetic. Each cold cache line forces a
# fetch from L2/L3 before we can write. Non-temporal stores bypass the cache
# and write directly to WC buffers, eliminating RFO traffic entirely.
# =============================================================================


def amx_gemm_v4_nontemporal[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
):
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime c_stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, c_stride)
            tilestore[5, DType.int32](c_buf + 16, c_stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, c_stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, c_stride)

            for m in range(m_count):
                var dq = act_scale[mb + m] / Float32(127)
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                    var f32_v = i32_v.cast[DType.float32]() * dq
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store[non_temporal=True](
                        f32_v * ws)

    _ = c_arr


# =============================================================================
# H4: True compute baseline — tilestore to L1-warm scratch every iteration.
#
# Measures pure tile pipeline cost: tilezero + tileload A/B + tdpbssd +
# tilestore to a fixed 4KB scratch that stays in L1. No dequant, no output
# writes. Keeps scratch live via final read to prevent dead store elim.
# =============================================================================


def amx_gemm_v5_compute_only[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    M: Int,
) -> Int32:
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)

                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)

                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime c_stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, c_stride)
            tilestore[5, DType.int32](c_buf + 16, c_stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, c_stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, c_stride)

    var sentinel = c_buf[0]
    _ = c_arr
    return sentinel


# =============================================================================
# VNNI GEMV baseline: sequential per-row GEMV using production VNNI path.
# Uses vpdpbusd (u8 act × i8 weight) with colsum correction.
# Produces identical f32 output to the AMX path (bias cancels out).
# =============================================================================


def vnni_gemv_loop[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    colsum: F32Ptr,
    dst: F32Ptr,
    M: Int,
):
    for m in range(M):
        var dq = act_scale[m] / Float32(127)
        gemv_row[N, K, DType.float32](
            act + m * K, wpacked, dq, w_scale, colsum, dst + m * N)


# =============================================================================
# Scalar reference for correctness validation.
# True i8×i8 signed dot product, dequantized to f32.
# =============================================================================


def reference_gemm_scalar(
    act: I8Ptr,
    weight_raw: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int, N: Int, K: Int,
):
    for m in range(M):
        var dq = act_scale[m] / Float32(127)
        for n in range(N):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(act[m * K + k]) * Int32(weight_raw[n * K + k])
            dst[m * N + n] = Float32(acc) * dq * w_scale[n]


def convert_f32_to_bf16(src: F32Ptr, dst: BF16Ptr, count: Int):
    for i in range(0, count, SIMD_W):
        (dst + i).store((src + i).load[width=SIMD_W]().cast[DType.bfloat16]())


def chain_f32_materialized_then_bf16_amx[MID: Int, K: Int, N2: Int](
    act: I8Ptr,
    w1_packed: I8Ptr,
    act_scale: F32Ptr,
    w1_scale: F32Ptr,
    tmp_f32: F32Ptr,
    tmp_bf16: BF16Ptr,
    w2_packed: BF16Ptr,
    dst_out: F32Ptr,
    M: Int,
):
    amx_gemm[MID, K](act, w1_packed, act_scale, w1_scale, tmp_f32, M)
    convert_f32_to_bf16(tmp_f32, tmp_bf16, M * MID)
    amx_bf16_gemm[N2, MID](tmp_bf16, w2_packed, dst_out, M)


def chain_direct_bf16_then_bf16_amx[MID: Int, K: Int, N2: Int](
    act: I8Ptr,
    w1_packed: I8Ptr,
    act_scale: F32Ptr,
    w1_scale: F32Ptr,
    tmp_bf16: BF16Ptr,
    w2_packed: BF16Ptr,
    dst_out: F32Ptr,
    M: Int,
):
    amx_gemm_bf16_out[MID, K](
        act, w1_packed, act_scale, w1_scale, tmp_bf16, M)
    amx_bf16_gemm[N2, MID](tmp_bf16, w2_packed, dst_out, M)


def bench_chain_f32_materialized[MID: Int, K: Int, N2: Int](
    act: I8Ptr,
    w1_packed: I8Ptr,
    act_scale: F32Ptr,
    w1_scale: F32Ptr,
    tmp_f32: F32Ptr,
    tmp_bf16: BF16Ptr,
    w2_packed: BF16Ptr,
    dst_out: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            chain_f32_materialized_then_bf16_amx[MID, K, N2](
                act, w1_packed, act_scale, w1_scale,
                tmp_f32, tmp_bf16, w2_packed, dst_out, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            chain_f32_materialized_then_bf16_amx[MID, K, N2](
                act, w1_packed, act_scale, w1_scale,
                tmp_f32, tmp_bf16, w2_packed, dst_out, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_chain_direct_bf16[MID: Int, K: Int, N2: Int](
    act: I8Ptr,
    w1_packed: I8Ptr,
    act_scale: F32Ptr,
    w1_scale: F32Ptr,
    tmp_bf16: BF16Ptr,
    w2_packed: BF16Ptr,
    dst_out: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            chain_direct_bf16_then_bf16_amx[MID, K, N2](
                act, w1_packed, act_scale, w1_scale,
                tmp_bf16, w2_packed, dst_out, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            chain_direct_bf16_then_bf16_amx[MID, K, N2](
                act, w1_packed, act_scale, w1_scale,
                tmp_bf16, w2_packed, dst_out, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_vnni_gemv[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    colsum: F32Ptr,
    dst: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            vnni_gemv_loop[N, K](
                act, wpacked, act_scale, w_scale, colsum, dst, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            vnni_gemv_loop[N, K](
                act, wpacked, act_scale, w_scale, colsum, dst, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_amx_v1[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm[N, K](act, wpacked, act_scale, w_scale, dst, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm[N, K](act, wpacked, act_scale, w_scale, dst, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_amx_v2_prefetch[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm_v2_prefetch[N, K](
                act, wpacked, act_scale, w_scale, dst, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm_v2_prefetch[N, K](
                act, wpacked, act_scale, w_scale, dst, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_amx_v3_k_outer[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
    f32_acc: F32Ptr,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm_v3_k_outer_scratch[N, K](
                act, wpacked, act_scale, w_scale, dst, M, f32_acc)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm_v3_k_outer_scratch[N, K](
                act, wpacked, act_scale, w_scale, dst, M, f32_acc)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_amx_v4_nontemporal[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: F32Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm_v4_nontemporal[N, K](
                act, wpacked, act_scale, w_scale, dst, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm_v4_nontemporal[N, K](
                act, wpacked, act_scale, w_scale, dst, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_amx_v5_compute[N: Int, K: Int](
    act: I8Ptr,
    wpacked: I8Ptr,
    M: Int,
    iters: Int,
    warmup: Int,
    samples: Int,
) -> Int:
    var best = Int(0)
    var sink = Int32(0)
    for sample in range(samples):
        for _ in range(warmup):
            sink = sink ^ amx_gemm_v5_compute_only[N, K](act, wpacked, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            sink = sink ^ amx_gemm_v5_compute_only[N, K](act, wpacked, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns

    if sink == Int32(-2147483648):
        print(String(sink))
    return best


# =============================================================================
# Benchmark harness
# =============================================================================


def run_ablation[N: Int, K: Int](label: String):
    print("=== " + label + " (N=" + String(N) + ", K=" + String(K) + ") ===")

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xCAFEBABE87654321)

    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)

    var colsum = alloc_zeroed[DType.float32](N)
    compute_colsum(weight_raw, colsum, N, K)

    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))

    comptime MAX_M = 8192
    comptime ACT_ROWS = MAX_M + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K)

    var act_scale = alloc_zeroed[DType.float32](ACT_ROWS)
    fill_random_f32(act_scale, ACT_ROWS, Float32(1.0))

    var out_ref = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_vnni = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_v1 = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_v2 = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_v3 = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_v4 = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var scratch_v3 = alloc_zeroed[DType.float32](M_STEP * N)

    # --- Correctness ---
    comptime TEST_M = 32
    reference_gemm_scalar(act, weight_raw, act_scale, w_scale,
        out_ref, TEST_M, N, K)
    vnni_gemv_loop[N, K](act, wpacked, act_scale, w_scale, colsum,
        out_vnni, TEST_M)
    amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_v1, TEST_M)
    amx_gemm_v2_prefetch[N, K](act, wpacked, act_scale, w_scale, out_v2, TEST_M)
    amx_gemm_v3_k_outer[N, K](act, wpacked, act_scale, w_scale, out_v3, TEST_M)
    amx_gemm_v4_nontemporal[N, K](act, wpacked, act_scale, w_scale, out_v4, TEST_M)

    var max_vnni = Float32(0)
    var max_v1 = Float32(0)
    var max_v2 = Float32(0)
    var max_v3 = Float32(0)
    var max_v4 = Float32(0)
    for i in range(TEST_M * N):
        var r = out_ref[i]
        var dv = (out_vnni[i] - r).__abs__()
        var d1 = (out_v1[i] - r).__abs__()
        var d2 = (out_v2[i] - r).__abs__()
        var d3 = (out_v3[i] - r).__abs__()
        var d4 = (out_v4[i] - r).__abs__()
        if dv > max_vnni:
            max_vnni = dv
        if d1 > max_v1:
            max_v1 = d1
        if d2 > max_v2:
            max_v2 = d2
        if d3 > max_v3:
            max_v3 = d3
        if d4 > max_v4:
            max_v4 = d4

    print("  correctness vs scalar (M=" + String(TEST_M) + "):")
    print("    vnni row loop:   max_el=" + String(max_vnni))
    print("    v1 baseline:     max_el=" + String(max_v1))
    print("    v2 prefetch:     max_el=" + String(max_v2))
    print("    v3 k-outer:      max_el=" + String(max_v3))
    print("    v4 nontemporal:  max_el=" + String(max_v4))
    print("")

    # --- Prefill scaling: compare the actual candidate designs side by side. ---
    var bench_ms = InlineArray[Int, 10](fill=0)
    bench_ms[0] = 16
    bench_ms[1] = 32
    bench_ms[2] = 64
    bench_ms[3] = 128
    bench_ms[4] = 256
    bench_ms[5] = 512
    bench_ms[6] = 1024
    bench_ms[7] = 2048
    bench_ms[8] = 4096
    bench_ms[9] = 8192

    print("  prefill token-batch scaling (best of samples, ns/token):")
    print("       M | it | smp |    v1 | v2 pf | v3 kO | v4 NT | v5 cmp | cmp/v1 | GOPS v1")
    print("  -------|----|-----|-------|-------|-------|-------|--------|--------|--------")

    for idx in range(10):
        var test_m = bench_ms[idx]
        if test_m > MAX_M:
            continue

        var iters = 200
        if test_m >= 128:
            iters = 100
        if test_m >= 512:
            iters = 30
        if test_m >= 2048:
            iters = 5
        if test_m >= 4096:
            iters = 3
        if test_m >= 8192:
            iters = 2

        var samples = 2
        if test_m >= 2048:
            samples = 1
        var warmup = max(iters // 10, 2)

        var ns_v1 = bench_amx_v1[N, K](
            act, wpacked, act_scale, w_scale, out_v1,
            test_m, iters, warmup, samples)
        var ns_v2 = bench_amx_v2_prefetch[N, K](
            act, wpacked, act_scale, w_scale, out_v2,
            test_m, iters, warmup, samples)

        # v3 is expected to be expensive; still include it so the table answers
        # whether A-tile reuse can ever pay for the partial-accumulator traffic.
        var ns_v3 = bench_amx_v3_k_outer[N, K](
            act, wpacked, act_scale, w_scale, out_v3,
            test_m, scratch_v3, iters, warmup, samples)
        var ns_v4 = bench_amx_v4_nontemporal[N, K](
            act, wpacked, act_scale, w_scale, out_v4,
            test_m, iters, warmup, samples)
        var ns_v5 = bench_amx_v5_compute[N, K](
            act, wpacked, test_m, iters, warmup, samples)

        var cmp_pct = Float32(0)
        if ns_v1 > 0:
            cmp_pct = Float32(ns_v5) / Float32(ns_v1) * Float32(100)

        var total_macs = test_m * N * K
        var gops = Float32(total_macs) / Float32(ns_v1)

        print("  " + String(test_m)
            + "  | " + String(iters)
            + " | " + String(samples)
            + " | " + String(ns_v1 // test_m)
            + " | " + String(ns_v2 // test_m)
            + " | " + String(ns_v3 // test_m)
            + " | " + String(ns_v4 // test_m)
            + " | " + String(ns_v5 // test_m)
            + " | " + String(cmp_pct) + "%"
            + " | " + String(gops))

    print("")
    print("  small-M row-loop crossover (total ns, not a prefill target):")
    print("       M | AMX v1 | VNNI rows | VNNI/AMX")
    print("  -------|--------|-----------|---------")

    var small_ms = InlineArray[Int, 6](fill=0)
    small_ms[0] = 1
    small_ms[1] = 2
    small_ms[2] = 4
    small_ms[3] = 8
    small_ms[4] = 16
    small_ms[5] = 32

    for idx in range(6):
        var test_m = small_ms[idx]
        var iters = 100
        if test_m >= 16:
            iters = 50
        var warmup = max(iters // 10, 2)
        var samples = 2

        var ns_amx = bench_amx_v1[N, K](
            act, wpacked, act_scale, w_scale, out_v1,
            test_m, iters, warmup, samples)
        var ns_vnni = bench_vnni_gemv[N, K](
            act, wpacked, act_scale, w_scale, colsum, out_vnni,
            test_m, iters, warmup, samples)
        var ratio = Float32(0)
        if ns_amx > 0:
            ratio = Float32(ns_vnni) / Float32(ns_amx)

        print("  " + String(test_m)
            + "  | " + String(ns_amx)
            + " | " + String(ns_vnni)
            + " | " + String(ratio))

    weight_raw.free()
    wpacked.free()
    colsum.free()
    w_scale.free()
    act.free()
    act_scale.free()
    out_ref.free()
    out_vnni.free()
    out_v1.free()
    out_v2.free()
    out_v3.free()
    out_v4.free()
    scratch_v3.free()


def run_bf16_chain_ablation[MID: Int, K: Int, N2: Int](label: String):
    print("=== AMX BF16 post-GEMM chain: " + label
        + " (MID=" + String(MID) + ", K=" + String(K)
        + ", N2=" + String(N2) + ") ===")

    var w1_raw = alloc_zeroed[DType.int8](MID * K)
    fill_random_i8(w1_raw, MID * K, seed=0x1122334455667788)

    var w1_packed = alloc_zeroed[DType.int8](MID * K)
    pack_vnni(w1_raw, w1_packed, MID, K)

    var w1_scale = alloc_zeroed[DType.float32](MID)
    fill_random_f32(w1_scale, MID, Float32(0.01))

    var w2_raw = alloc_zeroed[DType.bfloat16](N2 * MID)
    fill_random_bf16(w2_raw, N2 * MID, Float32(0.01))

    var w2_packed = alloc_zeroed[DType.bfloat16](N2 * MID)
    pack_bf16_amx(w2_raw, w2_packed, N2, MID)

    comptime MAX_M_CHAIN = 1024
    comptime ACT_ROWS = MAX_M_CHAIN + M_STEP

    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K, seed=0x99AABBCCDDEEFF00)

    var act_scale = alloc_zeroed[DType.float32](ACT_ROWS)
    fill_random_f32(act_scale, ACT_ROWS, Float32(1.0))

    var tmp_f32 = alloc_zeroed[DType.float32](ACT_ROWS * MID)
    var tmp_bf16_a = alloc_zeroed[DType.bfloat16](ACT_ROWS * MID)
    var tmp_bf16_b = alloc_zeroed[DType.bfloat16](ACT_ROWS * MID)
    var out_a = alloc_zeroed[DType.float32](ACT_ROWS * N2)
    var out_b = alloc_zeroed[DType.float32](ACT_ROWS * N2)

    comptime TEST_M = 32
    chain_f32_materialized_then_bf16_amx[MID, K, N2](
        act, w1_packed, act_scale, w1_scale,
        tmp_f32, tmp_bf16_a, w2_packed, out_a, TEST_M)
    chain_direct_bf16_then_bf16_amx[MID, K, N2](
        act, w1_packed, act_scale, w1_scale,
        tmp_bf16_b, w2_packed, out_b, TEST_M)

    var max_tmp_diff = Float32(0)
    for i in range(TEST_M * MID):
        var d = (Float32(tmp_bf16_a[i]) - Float32(tmp_bf16_b[i])).__abs__()
        if d > max_tmp_diff:
            max_tmp_diff = d

    var max_out_diff = Float32(0)
    for i in range(TEST_M * N2):
        var d = (out_a[i] - out_b[i]).__abs__()
        if d > max_out_diff:
            max_out_diff = d

    print("  correctness vs f32-materialized handoff (M="
        + String(TEST_M) + "):")
    print("    tmp bf16 max_el: " + String(max_tmp_diff))
    print("    out f32 max_el:  " + String(max_out_diff))
    print("")

    var bench_ms = InlineArray[Int, 4](fill=0)
    bench_ms[0] = 32
    bench_ms[1] = 128
    bench_ms[2] = 512
    bench_ms[3] = 1024

    print("  chain scaling (best of samples, ns/token):")
    print("       M | it | f32 handoff | direct bf16 | speedup")
    print("  -------|----|-------------|-------------|--------")

    for idx in range(4):
        var test_m = bench_ms[idx]
        var iters = 20
        if test_m >= 128:
            iters = 10
        if test_m >= 512:
            iters = 3
        if test_m >= 1024:
            iters = 2
        var warmup = max(iters // 5, 1)
        var samples = 2

        var ns_f32 = bench_chain_f32_materialized[MID, K, N2](
            act, w1_packed, act_scale, w1_scale,
            tmp_f32, tmp_bf16_a, w2_packed, out_a,
            test_m, iters, warmup, samples)
        var ns_bf16 = bench_chain_direct_bf16[MID, K, N2](
            act, w1_packed, act_scale, w1_scale,
            tmp_bf16_b, w2_packed, out_b,
            test_m, iters, warmup, samples)

        var speedup = Float32(0)
        if ns_bf16 > 0:
            speedup = Float32(ns_f32) / Float32(ns_bf16)

        print("  " + String(test_m)
            + "  | " + String(iters)
            + " | " + String(ns_f32 // test_m)
            + " | " + String(ns_bf16 // test_m)
            + " | " + String(speedup))

    w1_raw.free()
    w1_packed.free()
    w1_scale.free()
    w2_raw.free()
    w2_packed.free()
    act.free()
    act_scale.free()
    tmp_f32.free()
    tmp_bf16_a.free()
    tmp_bf16_b.free()
    out_a.free()
    out_b.free()


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    print("=== AMX Prefill GEMM Ablations ===")
    print("M_STEP=" + String(M_STEP) + " N_STEP=" + String(N_STEP)
        + " K_STEP=" + String(K_STEP))
    print("Tile config: 2-2-4 (all tiles 16x64), tdpbssd (i8 x i8 -> i32)")
    print("")

    run_ablation[1536, 3072]("MoE expert W1/W3")
    run_ablation[3072, 3072]("O-proj shape")
    run_bf16_chain_ablation[1536, 3072, 3072]("direct bf16 handoff")
