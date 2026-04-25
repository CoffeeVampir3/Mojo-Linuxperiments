from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_i8_config, init_intel_amx, ldtilecfg,
)
from experimental3.kernels.gemv import gemv_row, gemv_row_blocked_bf16_scaled
from experimental3.kernels.gemm_amx import amx_gemm, amx_gemm_blocked
from simd_math import set_subnormal_zeroing


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime SIMD_W = simd_width_of[DType.float32]()

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# Test data utilities (mirrors prefill_kernel_proposals.mojo)
# ============================================================================


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


def compute_block_colsums(
    src: I8Ptr, colsum: F32Ptr, N: Int, K: Int, fwht_blk: Int,
):
    var num_blocks = K // fwht_blk
    for n in range(N):
        for blk in range(num_blocks):
            var acc = Int32(0)
            for kk in range(fwht_blk):
                acc += Int32(src[n * K + blk * fwht_blk + kk])
            colsum[blk * N + n] = Float32(acc)


def reference_per_row(
    act: I8Ptr, weight: I8Ptr, act_scale: F32Ptr, w_scale: F32Ptr,
    dst: F32Ptr, M: Int, N: Int, K: Int,
):
    for m in range(M):
        var dq = act_scale[m] / Float32(127)
        for n in range(N):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(act[m * K + k]) * Int32(weight[n * K + k])
            dst[m * N + n] = Float32(acc) * dq * w_scale[n]


def reference_blocked(
    act: I8Ptr, weight: I8Ptr, blk_scales: F32Ptr, w_scale: F32Ptr,
    dst: F32Ptr, M: Int, N: Int, K: Int, fwht_blk: Int,
    output_scale: Float32,
):
    var num_blocks = K // fwht_blk
    for m in range(M):
        for n in range(N):
            var f32_sum = Float32(0)
            for blk in range(num_blocks):
                var dq = blk_scales[m * num_blocks + blk] / Float32(127)
                var i32_acc = Int32(0)
                for kk in range(fwht_blk):
                    var k = blk * fwht_blk + kk
                    i32_acc += Int32(act[m * K + k]) * Int32(weight[n * K + k])
                f32_sum += Float32(i32_acc) * dq
            dst[m * N + n] = f32_sum * w_scale[n] * output_scale


# ============================================================================
# Bench helpers
# ============================================================================


def bench_iters(M: Int) -> Int:
    if M >= 1024:
        return 20
    if M >= 512:
        return 30
    if M >= 128:
        return 100
    if M >= 16:
        return 200
    return 500


def bench_samples(M: Int) -> Int:
    if M >= 1024:
        return 1
    return 2


def max_abs_f32_minus_bf16(
    f32_ref: F32Ptr, bf16_test: BF16Ptr, count: Int,
) -> Tuple[Float32, Float32]:
    """Returns (max_abs_diff, max_abs_ref)."""
    var max_diff = Float32(0)
    var max_ref = Float32(0)
    for i in range(count):
        var rv = f32_ref[i]
        var tv = Float32(bf16_test[i])
        var d = (rv - tv).__abs__()
        if d > max_diff:
            max_diff = d
        var ar = rv.__abs__()
        if ar > max_ref:
            max_ref = ar
    return (max_diff, max_ref)


def max_abs_bf16_diff(
    a: BF16Ptr, b: BF16Ptr, count: Int,
) -> Float32:
    var mx = Float32(0)
    for i in range(count):
        var d = (Float32(a[i]) - Float32(b[i])).__abs__()
        if d > mx:
            mx = d
    return mx


# ============================================================================
# Correctness panels
# ============================================================================


def correctness_per_row[N: Int, K: Int](label: String, M: Int = 64):
    print("--- correctness per_row: " + label
        + " (N=" + String(N) + ", K=" + String(K) + ", M=" + String(M) + ") ---")

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xCAFEBABE87654321)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var colsum = alloc_zeroed[DType.float32](N)
    compute_colsum(weight_raw, colsum, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))

    var ACT_ROWS = M + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K)
    var act_scale = alloc_zeroed[DType.float32](ACT_ROWS)
    fill_random_f32(act_scale, ACT_ROWS, Float32(1.0))

    var out_ref = alloc_zeroed[DType.float32](M * N)
    var out_amx = alloc_zeroed[DType.bfloat16](M * N)
    var out_vnni = alloc_zeroed[DType.bfloat16](M * N)

    reference_per_row(act, weight_raw, act_scale, w_scale, out_ref, M, N, K)
    amx_gemm[N, K, DType.bfloat16](act, wpacked, act_scale, w_scale, out_amx, M)
    for m in range(M):
        var dq = act_scale[m] / Float32(127)
        gemv_row[N, K, DType.bfloat16](
            act + m * K, wpacked, dq, w_scale, colsum, out_vnni + m * N)

    var amx_diff_ref = max_abs_f32_minus_bf16(out_ref, out_amx, M * N)
    var vnni_diff_ref = max_abs_f32_minus_bf16(out_ref, out_vnni, M * N)
    var amx_vs_vnni = max_abs_bf16_diff(out_amx, out_vnni, M * N)

    var amx_rel = amx_diff_ref[0] / max(amx_diff_ref[1], Float32(1e-30))
    var vnni_rel = vnni_diff_ref[0] / max(vnni_diff_ref[1], Float32(1e-30))
    var amx_vs_vnni_rel = amx_vs_vnni / max(amx_diff_ref[1], Float32(1e-30))

    print("  max|ref|=" + String(amx_diff_ref[1])
        + " amx_vs_ref_max=" + String(amx_diff_ref[0])
        + " (rel=" + String(amx_rel) + ")"
        + " vnni_vs_ref_max=" + String(vnni_diff_ref[0])
        + " (rel=" + String(vnni_rel) + ")"
        + " amx_vs_vnni_max=" + String(amx_vs_vnni)
        + " (rel=" + String(amx_vs_vnni_rel) + ")")

    var pass_amx_ref = amx_rel < Float32(0.02)
    var pass_vnni_ref = vnni_rel < Float32(0.02)
    var pass_match = amx_vs_vnni_rel < Float32(0.005)
    if pass_amx_ref and pass_vnni_ref and pass_match:
        print("  PASS")
    else:
        print("  FAIL  amx_ref=" + String(pass_amx_ref)
            + " vnni_ref=" + String(pass_vnni_ref)
            + " amx_vs_vnni=" + String(pass_match))

    weight_raw.free()
    wpacked.free()
    colsum.free()
    w_scale.free()
    act.free()
    act_scale.free()
    out_ref.free()
    out_amx.free()
    out_vnni.free()


def correctness_blocked[N: Int, K: Int, fwht_blk: Int](
    label: String, M: Int = 64,
    output_scale: Float32 = Float32(0.85),
):
    print("--- correctness blocked: " + label
        + " (N=" + String(N) + ", K=" + String(K)
        + ", blk=" + String(fwht_blk) + ", M=" + String(M)
        + ", output_scale=" + String(output_scale) + ") ---")

    comptime num_blocks = K // fwht_blk

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xB10C4ED123456789)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))
    var blk_colsum = alloc_zeroed[DType.float32](num_blocks * N)
    compute_block_colsums(weight_raw, blk_colsum, N, K, fwht_blk)

    var ACT_ROWS = M + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K, seed=0xB10CA0C712340000)
    var blk_scales = alloc_zeroed[DType.float32](ACT_ROWS * num_blocks)
    fill_random_f32(blk_scales, ACT_ROWS * num_blocks, Float32(1.0))

    var out_ref = alloc_zeroed[DType.float32](M * N)
    var out_amx = alloc_zeroed[DType.bfloat16](M * N)
    var out_vnni = alloc_zeroed[DType.bfloat16](M * N)

    reference_blocked(act, weight_raw, blk_scales, w_scale,
        out_ref, M, N, K, fwht_blk, output_scale)
    amx_gemm_blocked[N, K, fwht_blk, DType.bfloat16](
        act, wpacked, blk_scales, w_scale, out_amx, M, output_scale)
    for m in range(M):
        gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
            act + m * K, wpacked, blk_scales + m * num_blocks,
            w_scale, blk_colsum, out_vnni + m * N, output_scale)

    var amx_diff_ref = max_abs_f32_minus_bf16(out_ref, out_amx, M * N)
    var vnni_diff_ref = max_abs_f32_minus_bf16(out_ref, out_vnni, M * N)
    var amx_vs_vnni = max_abs_bf16_diff(out_amx, out_vnni, M * N)

    var amx_rel = amx_diff_ref[0] / max(amx_diff_ref[1], Float32(1e-30))
    var vnni_rel = vnni_diff_ref[0] / max(vnni_diff_ref[1], Float32(1e-30))
    var amx_vs_vnni_rel = amx_vs_vnni / max(amx_diff_ref[1], Float32(1e-30))

    print("  max|ref|=" + String(amx_diff_ref[1])
        + " amx_vs_ref_max=" + String(amx_diff_ref[0])
        + " (rel=" + String(amx_rel) + ")"
        + " vnni_vs_ref_max=" + String(vnni_diff_ref[0])
        + " (rel=" + String(vnni_rel) + ")"
        + " amx_vs_vnni_max=" + String(amx_vs_vnni)
        + " (rel=" + String(amx_vs_vnni_rel) + ")")

    var pass_amx_ref = amx_rel < Float32(0.02)
    var pass_vnni_ref = vnni_rel < Float32(0.02)
    var pass_match = amx_vs_vnni_rel < Float32(0.005)
    if pass_amx_ref and pass_vnni_ref and pass_match:
        print("  PASS")
    else:
        print("  FAIL  amx_ref=" + String(pass_amx_ref)
            + " vnni_ref=" + String(pass_vnni_ref)
            + " amx_vs_vnni=" + String(pass_match))

    weight_raw.free()
    wpacked.free()
    w_scale.free()
    blk_colsum.free()
    act.free()
    blk_scales.free()
    out_ref.free()
    out_amx.free()
    out_vnni.free()


# ============================================================================
# Perf panels
# ============================================================================


comptime M_SWEEP_LEN = 11


def make_m_sweep() -> InlineArray[Int, M_SWEEP_LEN]:
    var s = InlineArray[Int, M_SWEEP_LEN](fill=0)
    s[0] = 1
    s[1] = 2
    s[2] = 4
    s[3] = 8
    s[4] = 16
    s[5] = 32
    s[6] = 64
    s[7] = 128
    s[8] = 256
    s[9] = 512
    s[10] = 1024
    return s^


def time_amx_per_row[N: Int, K: Int](
    act: I8Ptr, wpacked: I8Ptr, act_scale: F32Ptr, w_scale: F32Ptr,
    dst: BF16Ptr, M: Int,
) -> Int:
    var iters = bench_iters(M)
    var samples = bench_samples(M)
    var warmup = max(iters // 10, 2)
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm[N, K, DType.bfloat16](
                act, wpacked, act_scale, w_scale, dst, M)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm[N, K, DType.bfloat16](
                act, wpacked, act_scale, w_scale, dst, M)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def time_vnni_per_row[N: Int, K: Int](
    act: I8Ptr, wpacked: I8Ptr, act_scale: F32Ptr, w_scale: F32Ptr,
    colsum: F32Ptr, dst: BF16Ptr, M: Int,
) -> Int:
    var iters = bench_iters(M)
    var samples = bench_samples(M)
    var warmup = max(iters // 10, 2)
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            for m in range(M):
                var dq = act_scale[m] / Float32(127)
                gemv_row[N, K, DType.bfloat16](
                    act + m * K, wpacked, dq, w_scale, colsum, dst + m * N)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            for m in range(M):
                var dq = act_scale[m] / Float32(127)
                gemv_row[N, K, DType.bfloat16](
                    act + m * K, wpacked, dq, w_scale, colsum, dst + m * N)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def time_amx_blocked[N: Int, K: Int, fwht_blk: Int](
    act: I8Ptr, wpacked: I8Ptr, blk_scales: F32Ptr, w_scale: F32Ptr,
    dst: BF16Ptr, M: Int, output_scale: Float32,
) -> Int:
    var iters = bench_iters(M)
    var samples = bench_samples(M)
    var warmup = max(iters // 10, 2)
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            amx_gemm_blocked[N, K, fwht_blk, DType.bfloat16](
                act, wpacked, blk_scales, w_scale, dst, M, output_scale)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            amx_gemm_blocked[N, K, fwht_blk, DType.bfloat16](
                act, wpacked, blk_scales, w_scale, dst, M, output_scale)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def time_vnni_blocked[N: Int, K: Int, fwht_blk: Int](
    act: I8Ptr, wpacked: I8Ptr, blk_scales: F32Ptr, w_scale: F32Ptr,
    blk_colsum: F32Ptr, dst: BF16Ptr, M: Int, output_scale: Float32,
) -> Int:
    comptime num_blocks = K // fwht_blk
    var iters = bench_iters(M)
    var samples = bench_samples(M)
    var warmup = max(iters // 10, 2)
    var best = Int(0)
    for sample in range(samples):
        for _ in range(warmup):
            for m in range(M):
                gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
                    act + m * K, wpacked, blk_scales + m * num_blocks,
                    w_scale, blk_colsum, dst + m * N, output_scale)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            for m in range(M):
                gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
                    act + m * K, wpacked, blk_scales + m * num_blocks,
                    w_scale, blk_colsum, dst + m * N, output_scale)
        var ns = (Int(perf_counter_ns()) - t) // iters
        if sample == 0 or ns < best:
            best = ns
    return best


def bench_per_row[N: Int, K: Int](label: String):
    print("--- bench per_row: " + label
        + " (N=" + String(N) + ", K=" + String(K) + ") ---")

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xCAFEBABE87654321)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var colsum = alloc_zeroed[DType.float32](N)
    compute_colsum(weight_raw, colsum, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))

    var ACT_ROWS = 1024 + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K)
    var act_scale = alloc_zeroed[DType.float32](ACT_ROWS)
    fill_random_f32(act_scale, ACT_ROWS, Float32(1.0))
    var out_amx = alloc_zeroed[DType.bfloat16](ACT_ROWS * N)
    var out_vnni = alloc_zeroed[DType.bfloat16](ACT_ROWS * N)

    print("       M |  AMX ns/M | VNNI ns/M |  speedup |  AMX GOPS")
    var speedup_at_64 = Float32(0)
    var speedup_at_1024 = Float32(0)
    var sweep = make_m_sweep()
    for idx in range(M_SWEEP_LEN):
        var test_m = sweep[idx]
        var amx_ns = time_amx_per_row[N, K](
            act, wpacked, act_scale, w_scale, out_amx, test_m)
        var vnni_ns = time_vnni_per_row[N, K](
            act, wpacked, act_scale, w_scale, colsum, out_vnni, test_m)
        var speedup = Float32(vnni_ns) / Float32(amx_ns)
        var gops = Float32(test_m * N * K * 2) / Float32(amx_ns)
        print("   " + String(test_m) + "    | " + String(amx_ns // test_m)
            + " | " + String(vnni_ns // test_m)
            + " | " + String(speedup) + " | " + String(gops))
        if test_m == 64:
            speedup_at_64 = speedup
        if test_m == 1024:
            speedup_at_1024 = speedup

    print("  summary: VNNI/AMX speedup at M=64 = " + String(speedup_at_64)
        + "  at M=1024 = " + String(speedup_at_1024))
    if speedup_at_64 >= Float32(2.0):
        print("    GATE_PASS_M64 (>=2x speedup at production per-worker M)")
    else:
        print("    GATE_FAIL_M64 (<2x speedup at M=64)")

    weight_raw.free()
    wpacked.free()
    colsum.free()
    w_scale.free()
    act.free()
    act_scale.free()
    out_amx.free()
    out_vnni.free()


def bench_blocked[N: Int, K: Int, fwht_blk: Int](
    label: String, output_scale: Float32 = Float32(0.85),
):
    print("--- bench blocked: " + label
        + " (N=" + String(N) + ", K=" + String(K)
        + ", blk=" + String(fwht_blk) + ") ---")

    comptime num_blocks = K // fwht_blk

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xB10C4ED123456789)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))
    var blk_colsum = alloc_zeroed[DType.float32](num_blocks * N)
    compute_block_colsums(weight_raw, blk_colsum, N, K, fwht_blk)

    var ACT_ROWS = 1024 + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K, seed=0xB10CA0C712340000)
    var blk_scales = alloc_zeroed[DType.float32](ACT_ROWS * num_blocks)
    fill_random_f32(blk_scales, ACT_ROWS * num_blocks, Float32(1.0))
    var out_amx = alloc_zeroed[DType.bfloat16](ACT_ROWS * N)
    var out_vnni = alloc_zeroed[DType.bfloat16](ACT_ROWS * N)

    print("       M |  AMX ns/M | VNNI ns/M |  speedup |  AMX GOPS")
    var speedup_at_64 = Float32(0)
    var speedup_at_1024 = Float32(0)
    var sweep = make_m_sweep()
    for idx in range(M_SWEEP_LEN):
        var test_m = sweep[idx]
        var amx_ns = time_amx_blocked[N, K, fwht_blk](
            act, wpacked, blk_scales, w_scale, out_amx, test_m, output_scale)
        var vnni_ns = time_vnni_blocked[N, K, fwht_blk](
            act, wpacked, blk_scales, w_scale, blk_colsum, out_vnni,
            test_m, output_scale)
        var speedup = Float32(vnni_ns) / Float32(amx_ns)
        var gops = Float32(test_m * N * K * 2) / Float32(amx_ns)
        print("   " + String(test_m) + "    | " + String(amx_ns // test_m)
            + " | " + String(vnni_ns // test_m)
            + " | " + String(speedup) + " | " + String(gops))
        if test_m == 64:
            speedup_at_64 = speedup
        if test_m == 1024:
            speedup_at_1024 = speedup

    print("  summary: VNNI/AMX speedup at M=64 = " + String(speedup_at_64)
        + "  at M=1024 = " + String(speedup_at_1024))
    if speedup_at_64 >= Float32(2.0):
        print("    GATE_PASS_M64 (>=2x speedup at production per-worker M)")
    else:
        print("    GATE_FAIL_M64 (<2x speedup at M=64)")

    weight_raw.free()
    wpacked.free()
    w_scale.free()
    blk_colsum.free()
    act.free()
    blk_scales.free()
    out_amx.free()
    out_vnni.free()


# ============================================================================
# Main
# ============================================================================


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    print("=== AMX vs VNNI: attn_proj / o_proj microbench ===")
    print("M_STEP=" + String(M_STEP) + " N_STEP=" + String(N_STEP)
        + " K_STEP=" + String(K_STEP))
    print("ratio < 1.0 means AMX is faster than VNNI; report ns/M = ns / row.")
    print("")

    print("[1/8] correctness panels")
    correctness_per_row[2048, 3072]("QKV TP=4 (N=2048, K=3072)")
    correctness_per_row[1024, 3072]("QKV TP=8 (N=1024, K=3072)")
    correctness_blocked[3072, 1536, 128]("O TP=4 (N=3072, K=1536, blk=128)")
    correctness_blocked[3072, 768, 128]("O TP=8 (N=3072, K=768, blk=128)")
    print("")

    print("[2/8] perf panels — QKV (per-row act scale)")
    bench_per_row[2048, 3072]("QKV TP=4")
    bench_per_row[1024, 3072]("QKV TP=8")
    print("")

    print("[3/8] perf panels — O (per-K-block act scale)")
    bench_blocked[3072, 1536, 128]("O TP=4")
    bench_blocked[3072, 768, 128]("O TP=8")
    print("")

    print("Done. If correctness panels all PASS and bench summaries show")
    print("AMX/VNNI ratio < 0.5 at M=64, gate to dispatcher edit is open.")
