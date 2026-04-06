"""Validate int8_gemv against scalar f32 reference.

Tests single dispatch at SmolLM2-scale dimensions with combined QKV packing.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from numa import NumaInfo, NumaTopology
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from threading.burst_threading import BurstPool
from kernels.vnni import pack_vnni

from experimental2.kernels.int8_gemv import int8_gemv, WorkerConfig, gemv_row


def f32_reference(
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    weight_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    weight_scale: UnsafePointer[Float32, MutAnyOrigin],
    act_dequant: Float32,
    dst: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int, n: Int, k: Int,
):
    """Scalar reference matching int8_gemm_row epilogue:
    dst[m,n] = (sum_k (act_i8+128) * w_i8 - 128*colsum[n]) * act_dequant * weight_scale[n]
    """
    for m in range(seq_len):
        for col in range(n):
            var acc = Int32(0)
            for ki in range(k):
                var a_u8 = Int32(act_i8[m * k + ki]) + 128
                var w = Int32(weight_i8[col * k + ki])
                acc += a_u8 * w
            var corrected = Float32(acc) - Float32(128) * colsum[col]
            dst[m * n + col] = corrected * act_dequant * weight_scale[col]


def fill_i8(
    p: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    count: Int, seed: Int,
):
    for i in range(count):
        p[i] = Scalar[DType.int8]((i * seed + 37) % 251 - 125)


def compute_colsum(
    weight_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
):
    for n in range(rows):
        var acc = Int(0)
        for ki in range(cols):
            acc += Int(weight_i8[n * cols + ki])
        colsum[n] = Float32(acc)


def compute_weight_scale(
    weight_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    weight_scale: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
):
    for n in range(rows):
        var absmax = Float32(0)
        for ki in range(cols):
            var v = Float32(weight_i8[n * cols + ki])
            if v < 0: v = -v
            if v > absmax: absmax = v
        weight_scale[n] = absmax / Float32(127)


def compare(
    name: String,
    got: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    expected: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) -> Bool:
    var err_sum = Float64(0)
    var max_err = Float64(0)
    for i in range(count):
        var g = Float64(Float32(got[i]))
        var e = Float64(expected[i])
        var d = g - e
        if d < 0: d = -d
        err_sum += d
        if d > max_err: max_err = d
    var avg_err = err_sum / Float64(count)
    print("  " + name + ": avg_err=" + String(avg_err) + " max_err=" + String(max_err))
    # bf16 has 10 mantissa bits → relative precision ~1/1024 ≈ 0.1%.
    # For values in the hundreds, absolute error of ~1.0 is expected.
    var rel_err = Float64(0)
    var rel_count = 0
    for i in range(count):
        var e = Float64(expected[i])
        if e != 0:
            var d = Float64(Float32(got[i])) - e
            if d < 0: d = -d
            rel_err += d / (e if e > 0 else -e)
            rel_count += 1
    var avg_rel = rel_err / Float64(max(rel_count, 1)) * 100
    print("  avg_rel=" + String(avg_rel) + "%")
    if avg_rel < 1.0:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)

    var arena = NumaArena[](topo[0], 256 * 1024 * 1024)
    var pool = BurstPool[].for_topology(numa, topo[0], stack_size=2 * 1024 * 1024)
    print("workers: " + String(pool.get_capacity()))

    var act_dequant = Float32(2.5) / Float32(127)

    # =====================================================================
    # Test 1: Single [1, 576] × [576, 576]^T (O-projection scale)
    # =====================================================================
    comptime K1 = 576
    comptime N1 = 576

    print("\n=== Test 1: int8_gemv [1," + String(K1) + "] x [" + String(N1) + "," + String(K1) + "]^T ===")

    var mark1 = arena.mark()

    var act1 = arena.alloc[Scalar[DType.int8]](K1)
    var w1_row = arena.alloc[Scalar[DType.int8]](N1 * K1)
    var w1_packed = arena.alloc[UInt8](N1 * K1)
    var cs1 = arena.alloc[Float32](N1)
    var ws1 = arena.alloc[Float32](N1)
    var dst1 = arena.alloc[Scalar[DType.bfloat16]](N1)
    var ref1 = arena.alloc[Float32](N1)

    fill_i8(act1, K1, 7)
    fill_i8(w1_row, N1 * K1, 11)
    compute_colsum(w1_row, cs1, N1, K1)
    compute_weight_scale(w1_row, ws1, N1, K1)

    # Pack and run reference BEFORE packing overwrites
    f32_reference(act1, w1_row, cs1, ws1, act_dequant, ref1, 1, N1, K1)

    # VNNI pack (overwrites w1_row region via scratch→dst)
    var scratch1 = arena.alloc[UInt8](N1 * K1)
    memcpy(dest=scratch1, src=w1_row.bitcast[UInt8](), count=N1 * K1)
    pack_vnni(scratch1, w1_packed, N1, K1)

    var cfgs1 = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
    int8_gemv[N1, K1](Int(act1), Int(w1_packed), Int(cs1), Int(ws1), Int(dst1),
        1, act_dequant, cfgs1, pool).join()

    _ = compare("single [576x576]", dst1, ref1, N1)
    arena.reset_to(mark1)

    # =====================================================================
    # Test 2: Combined QKV [1, 576] × [320, 576]^T (SmolLM2 TP=3)
    # =====================================================================
    comptime K2 = 576
    comptime Q_N = 192
    comptime KV_N = 64
    comptime QKV_N = Q_N + KV_N + KV_N

    print("\n=== Test 2: combined QKV [1," + String(K2) + "] x [" + String(QKV_N) + "," + String(K2) + "]^T ===")

    var mark2 = arena.mark()

    var act2 = arena.alloc[Scalar[DType.int8]](K2)
    var wqkv_row = arena.alloc[Scalar[DType.int8]](QKV_N * K2)
    var wqkv_packed = arena.alloc[UInt8](QKV_N * K2)
    var cs_qkv = arena.alloc[Float32](QKV_N)
    var ws_qkv = arena.alloc[Float32](QKV_N)
    var dst_qkv = arena.alloc[Scalar[DType.bfloat16]](QKV_N)
    var ref_qkv = arena.alloc[Float32](QKV_N)

    fill_i8(act2, K2, 3)
    fill_i8(wqkv_row, QKV_N * K2, 13)
    compute_colsum(wqkv_row, cs_qkv, QKV_N, K2)
    compute_weight_scale(wqkv_row, ws_qkv, QKV_N, K2)

    f32_reference(act2, wqkv_row, cs_qkv, ws_qkv, act_dequant, ref_qkv, 1, QKV_N, K2)

    var scratch2 = arena.alloc[UInt8](QKV_N * K2)
    memcpy(dest=scratch2, src=wqkv_row.bitcast[UInt8](), count=QKV_N * K2)
    pack_vnni(scratch2, wqkv_packed, QKV_N, K2)

    var cfgs2 = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
    int8_gemv[QKV_N, K2](Int(act2), Int(wqkv_packed), Int(cs_qkv), Int(ws_qkv), Int(dst_qkv),
        1, act_dequant, cfgs2, pool).join()

    var pass_qkv = compare("combined QKV [320x576]", dst_qkv, ref_qkv, QKV_N)

    # Verify Q/K/V slices individually
    print("  slices: Q=[0:" + String(Q_N) + "] K=[" + String(Q_N) + ":" + String(Q_N + KV_N)
        + "] V=[" + String(Q_N + KV_N) + ":" + String(QKV_N) + "]")
    _ = compare("  Q slice", dst_qkv, ref_qkv, Q_N)
    _ = compare("  K slice",
        (dst_qkv + Q_N).bitcast[Scalar[DType.bfloat16]](),
        (ref_qkv + Q_N).bitcast[Float32](),
        KV_N)
    _ = compare("  V slice",
        (dst_qkv + Q_N + KV_N).bitcast[Scalar[DType.bfloat16]](),
        (ref_qkv + Q_N + KV_N).bitcast[Float32](),
        KV_N)

    arena.reset_to(mark2)

    # =====================================================================
    # Test 3: Multi-row prefill [4, 576] × [320, 576]^T
    # =====================================================================
    comptime SL3 = 4

    print("\n=== Test 3: prefill [" + String(SL3) + "," + String(K2) + "] x [" + String(QKV_N) + "," + String(K2) + "]^T ===")

    var act3 = arena.alloc[Scalar[DType.int8]](SL3 * K2)
    var w3_row = arena.alloc[Scalar[DType.int8]](QKV_N * K2)
    var w3_packed = arena.alloc[UInt8](QKV_N * K2)
    var cs3 = arena.alloc[Float32](QKV_N)
    var ws3 = arena.alloc[Float32](QKV_N)
    var dst3 = arena.alloc[Scalar[DType.bfloat16]](SL3 * QKV_N)
    var ref3 = arena.alloc[Float32](SL3 * QKV_N)

    fill_i8(act3, SL3 * K2, 31)
    fill_i8(w3_row, QKV_N * K2, 41)
    compute_colsum(w3_row, cs3, QKV_N, K2)
    compute_weight_scale(w3_row, ws3, QKV_N, K2)

    f32_reference(act3, w3_row, cs3, ws3, act_dequant, ref3, SL3, QKV_N, K2)

    var scratch3 = arena.alloc[UInt8](QKV_N * K2)
    memcpy(dest=scratch3, src=w3_row.bitcast[UInt8](), count=QKV_N * K2)
    pack_vnni(scratch3, w3_packed, QKV_N, K2)

    print("  dispatching...")
    var cfgs3 = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
    int8_gemv[QKV_N, K2](Int(act3), Int(w3_packed), Int(cs3), Int(ws3), Int(dst3),
        SL3, act_dequant, cfgs3, pool).join()
    print("  joined")

    _ = compare("prefill [4x320x576]", dst3, ref3, SL3 * QKV_N)

    print("\ndone")
    _ = pool
    _ = arena
    _ = pool
    _ = arena
