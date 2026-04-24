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
    tilezero, tileload, tilestore, tdpbssd,
)
from experimental3.kernels.gemv import gemv_row
from simd_math import set_subnormal_zeroing
from notstdcollections import AlignedInlineArray


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime SIMD_W = simd_width_of[DType.float32]()

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


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


def amx_gemm_v3_k_outer[N: Int, K: Int](
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

    var f32_acc = alloc_zeroed[DType.float32](M_STEP * N)

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

    f32_acc.free()
    _ = c_arr


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

    # --- Correctness ---
    comptime TEST_M = 32
    reference_gemm_scalar(act, weight_raw, act_scale, w_scale,
        out_ref, TEST_M, N, K)
    amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_v1, TEST_M)
    amx_gemm_v2_prefetch[N, K](act, wpacked, act_scale, w_scale, out_v2, TEST_M)
    amx_gemm_v3_k_outer[N, K](act, wpacked, act_scale, w_scale, out_v3, TEST_M)
    amx_gemm_v4_nontemporal[N, K](act, wpacked, act_scale, w_scale, out_v4, TEST_M)

    var max_v1 = Float32(0)
    var max_v2 = Float32(0)
    var max_v3 = Float32(0)
    var max_v4 = Float32(0)
    for i in range(TEST_M * N):
        var r = out_ref[i]
        var d1 = (out_v1[i] - r).__abs__()
        var d2 = (out_v2[i] - r).__abs__()
        var d3 = (out_v3[i] - r).__abs__()
        var d4 = (out_v4[i] - r).__abs__()
        if d1 > max_v1:
            max_v1 = d1
        if d2 > max_v2:
            max_v2 = d2
        if d3 > max_v3:
            max_v3 = d3
        if d4 > max_v4:
            max_v4 = d4

    print("  correctness vs scalar (M=" + String(TEST_M) + "):")
    print("    v1 baseline:     max_el=" + String(max_v1))
    print("    v2 prefetch:     max_el=" + String(max_v2))
    print("    v3 k-outer:      max_el=" + String(max_v3))
    print("    v4 nontemporal:  max_el=" + String(max_v4))
    print("")

    # --- Scaling benchmark: v1 vs v5 compute-only across real sizes ---
    var bench_ms = InlineArray[Int, 10](fill=0)
    bench_ms[0] = 32
    bench_ms[1] = 64
    bench_ms[2] = 128
    bench_ms[3] = 256
    bench_ms[4] = 512
    bench_ms[5] = 1024
    bench_ms[6] = 2048
    bench_ms[7] = 4096
    bench_ms[8] = 8192
    bench_ms[9] = 0

    print("       M |    v1 (ns) | ns/row |   v5 cmp (ns) | ns/row |  dequant% |    GOPS")
    print("  -------|------------|--------|---------------|--------|-----------|--------")

    for idx in range(9):
        var test_m = bench_ms[idx]
        if test_m > MAX_M:
            continue
        var iters = 500
        if test_m >= 256:
            iters = 100
        if test_m >= 1024:
            iters = 30
        if test_m >= 4096:
            iters = 10
        var warmup = max(iters // 5, 2)

        for _ in range(warmup):
            amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_v1, test_m)
        var t1 = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_v1, test_m)
        var ns_v1 = (Int(perf_counter_ns()) - t1) // iters

        for _ in range(warmup):
            _ = amx_gemm_v5_compute_only[N, K](act, wpacked, test_m)
        var t5 = Int(perf_counter_ns())
        for _ in range(iters):
            _ = amx_gemm_v5_compute_only[N, K](act, wpacked, test_m)
        var ns_v5 = (Int(perf_counter_ns()) - t5) // iters

        var dq_pct = Float32(0)
        if ns_v1 > 0 and ns_v5 > 0 and ns_v1 > ns_v5:
            dq_pct = Float32(ns_v1 - ns_v5) / Float32(ns_v1) * Float32(100)

        var total_macs = test_m * N * K
        var gops = Float32(total_macs) / Float32(ns_v1)

        print("  " + String(test_m)
            + "  |  " + String(ns_v1)
            + "  |  " + String(ns_v1 // test_m)
            + "  |  " + String(ns_v5)
            + "  |  " + String(ns_v5 // test_m)
            + "  |  " + String(dq_pct) + "%"
            + "  |  " + String(gops))

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
