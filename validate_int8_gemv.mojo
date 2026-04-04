"""Validate int8_gemv against scalar f32 reference.

Tests both single and batched dispatch modes at SmolLM2-scale dimensions.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from numa import NumaInfo, NumaTopology
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from threading.burst_threading import BurstPool
from kernels.vnni import pack_vnni

from experimental2.kernels.int_kernels.int8_gemv import (
    int8_gemv, int8_gemv_batched,
    int8_gemv_row,
    GemvCtx, BatchedGemvCtx, BatchedGemvEntry,
)


def f32_reference(
    act_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    weight_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    bias: UnsafePointer[Float32, MutAnyOrigin],
    scale: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int, n: Int, k: Int,
):
    """Scalar reference: dst[m,n] = (sum_k (act+128)*w - bias[n]) * scale[n]."""
    for m in range(seq_len):
        for col in range(n):
            var acc = Int32(0)
            for ki in range(k):
                var a_u8 = Int32(act_i8[m * k + ki]) + 128
                var w = Int32(weight_i8[col * k + ki])
                acc += a_u8 * w
            dst[m * n + col] = (Float32(acc) - bias[col]) * scale[col]


def fill_weights(
    w: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    rows: Int, cols: Int, seed: Int,
):
    for i in range(rows * cols):
        w[i] = Scalar[DType.int8]((i * seed + 37) % 251 - 125)


def fill_activation(
    a: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    seq_len: Int, cols: Int, seed: Int,
):
    for i in range(seq_len * cols):
        a[i] = Scalar[DType.int8]((i * seed + 13) % 251 - 125)


def compute_colsum_and_scales(
    weight_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    weight_scale: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
):
    for n in range(rows):
        var acc = Int(0)
        var absmax = Float32(0)
        for ki in range(cols):
            var v = weight_i8[n * cols + ki]
            acc += Int(v)
            var av = Float32(v)
            if av < 0: av = -av
            if av > absmax: absmax = av
        colsum[n] = Float32(acc)
        weight_scale[n] = absmax / Float32(127)


def precompute_bias_scale(
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    weight_scale: UnsafePointer[Float32, MutAnyOrigin],
    bias_out: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int, act_dequant: Float32,
):
    for i in range(n):
        bias_out[i] = Float32(128) * colsum[i]
        scale_out[i] = weight_scale[i] * act_dequant


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
    if avg_err < 0.5:
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

    # =====================================================================
    # Test 1: Single GEMV (SmolLM2 Q-projection scale: K=576, N=576)
    # =====================================================================
    comptime K = 576
    comptime N = 576
    comptime SL = 1
    var act_dequant = Float32(2.5) / Float32(127)  # simulated S_act/127

    print("\n=== Test 1: single int8_gemv [" + String(SL) + "," + String(K)
        + "] x [" + String(N) + "," + String(K) + "]^T ===")

    var mark = arena.mark()

    # Allocate
    var act_i8 = arena.alloc[Scalar[DType.int8]](SL * K)
    var w_i8_rowmajor = arena.alloc[Scalar[DType.int8]](N * K)
    var w_packed = arena.alloc[UInt8](N * K)
    var colsum = arena.alloc[Float32](N)
    var w_scale = arena.alloc[Float32](N)
    var bias = arena.alloc[Float32](N)
    var scale = arena.alloc[Float32](N)
    var dst_kernel = arena.alloc[Scalar[DType.bfloat16]](SL * N)
    var dst_ref = arena.alloc[Float32](SL * N)

    # Fill data
    fill_activation(act_i8, SL, K, 7)
    fill_weights(w_i8_rowmajor, N, K, 11)
    compute_colsum_and_scales(w_i8_rowmajor, colsum, w_scale, N, K)
    precompute_bias_scale(colsum, w_scale, bias, scale, N, act_dequant)

    # VNNI pack
    pack_vnni(w_i8_rowmajor.bitcast[UInt8](), w_packed, N, K)

    # Reference
    f32_reference(act_i8, w_i8_rowmajor, bias, scale, dst_ref, SL, N, K)

    # Kernel (single dispatch)
    var ctx = GemvCtx(Int(act_i8), Int(dst_kernel), N, K, SL)
    int8_gemv(ctx, Int(w_packed), Int(bias), Int(scale), pool).join()

    _ = compare("single gemv [576x576]", dst_kernel, dst_ref, SL * N)

    arena.reset_to(mark)

    # =====================================================================
    # Test 2: Batched Q+K+V (K=576, Q_N=192, KV_N=64)
    # =====================================================================
    comptime Q_N = 192
    comptime KV_N = 64
    comptime SL2 = 1

    print("\n=== Test 2: batched Q+K+V [" + String(SL2) + "," + String(K)
        + "] x Q[" + String(Q_N) + "] K[" + String(KV_N) + "] V[" + String(KV_N) + "] ===")

    var act2 = arena.alloc[Scalar[DType.int8]](SL2 * K)
    fill_activation(act2, SL2, K, 3)

    # Q weights
    var wq_row = arena.alloc[Scalar[DType.int8]](Q_N * K)
    var wq_packed = arena.alloc[UInt8](Q_N * K)
    var q_colsum = arena.alloc[Float32](Q_N)
    var q_wscale = arena.alloc[Float32](Q_N)
    var q_bias = arena.alloc[Float32](Q_N)
    var q_scale = arena.alloc[Float32](Q_N)
    var dq = arena.alloc[Scalar[DType.bfloat16]](SL2 * Q_N)
    var dq_ref = arena.alloc[Float32](SL2 * Q_N)
    fill_weights(wq_row, Q_N, K, 11)
    compute_colsum_and_scales(wq_row, q_colsum, q_wscale, Q_N, K)
    precompute_bias_scale(q_colsum, q_wscale, q_bias, q_scale, Q_N, act_dequant)
    pack_vnni(wq_row.bitcast[UInt8](), wq_packed, Q_N, K)

    # K weights
    var wk_row = arena.alloc[Scalar[DType.int8]](KV_N * K)
    var wk_packed = arena.alloc[UInt8](KV_N * K)
    var k_colsum = arena.alloc[Float32](KV_N)
    var k_wscale = arena.alloc[Float32](KV_N)
    var k_bias = arena.alloc[Float32](KV_N)
    var k_scale = arena.alloc[Float32](KV_N)
    var dk = arena.alloc[Scalar[DType.bfloat16]](SL2 * KV_N)
    var dk_ref = arena.alloc[Float32](SL2 * KV_N)
    fill_weights(wk_row, KV_N, K, 17)
    compute_colsum_and_scales(wk_row, k_colsum, k_wscale, KV_N, K)
    precompute_bias_scale(k_colsum, k_wscale, k_bias, k_scale, KV_N, act_dequant)
    pack_vnni(wk_row.bitcast[UInt8](), wk_packed, KV_N, K)

    # V weights
    var wv_row = arena.alloc[Scalar[DType.int8]](KV_N * K)
    var wv_packed = arena.alloc[UInt8](KV_N * K)
    var v_colsum = arena.alloc[Float32](KV_N)
    var v_wscale = arena.alloc[Float32](KV_N)
    var v_bias = arena.alloc[Float32](KV_N)
    var v_scale = arena.alloc[Float32](KV_N)
    var dv = arena.alloc[Scalar[DType.bfloat16]](SL2 * KV_N)
    var dv_ref = arena.alloc[Float32](SL2 * KV_N)
    fill_weights(wv_row, KV_N, K, 23)
    compute_colsum_and_scales(wv_row, v_colsum, v_wscale, KV_N, K)
    precompute_bias_scale(v_colsum, v_wscale, v_bias, v_scale, KV_N, act_dequant)
    pack_vnni(wv_row.bitcast[UInt8](), wv_packed, KV_N, K)

    # References
    f32_reference(act2, wq_row, q_bias, q_scale, dq_ref, SL2, Q_N, K)
    f32_reference(act2, wk_row, k_bias, k_scale, dk_ref, SL2, KV_N, K)
    f32_reference(act2, wv_row, v_bias, v_scale, dv_ref, SL2, KV_N, K)

    # Batched kernel
    var bctx = BatchedGemvCtx()
    bctx.act_ptr = Int(act2)
    bctx.k_total = K
    bctx.seq_len = SL2
    bctx.count = 3
    bctx.entries[0] = BatchedGemvEntry(Int(wq_packed), Int(q_bias), Int(q_scale), Int(dq), Q_N)
    bctx.entries[1] = BatchedGemvEntry(Int(wk_packed), Int(k_bias), Int(k_scale), Int(dk), KV_N)
    bctx.entries[2] = BatchedGemvEntry(Int(wv_packed), Int(v_bias), Int(v_scale), Int(dv), KV_N)

    int8_gemv_batched(bctx, pool).join()

    var pass_q = compare("Q [192]", dq, dq_ref, SL2 * Q_N)
    var pass_k = compare("K [64]", dk, dk_ref, SL2 * KV_N)
    var pass_v = compare("V [64]", dv, dv_ref, SL2 * KV_N)

    # =====================================================================
    # Test 3: Multi-row (SL=4, simulating small prefill)
    # =====================================================================
    comptime SL3 = 4

    print("\n=== Test 3: single int8_gemv [" + String(SL3) + "," + String(K)
        + "] x [" + String(N) + "," + String(K) + "]^T ===")

    arena.reset_to(mark)
    var act3 = arena.alloc[Scalar[DType.int8]](SL3 * K)
    var w3_row = arena.alloc[Scalar[DType.int8]](N * K)
    var w3_packed = arena.alloc[UInt8](N * K)
    var cs3 = arena.alloc[Float32](N)
    var ws3 = arena.alloc[Float32](N)
    var b3 = arena.alloc[Float32](N)
    var s3 = arena.alloc[Float32](N)
    var d3 = arena.alloc[Scalar[DType.bfloat16]](SL3 * N)
    var r3 = arena.alloc[Float32](SL3 * N)

    fill_activation(act3, SL3, K, 31)
    fill_weights(w3_row, N, K, 41)
    compute_colsum_and_scales(w3_row, cs3, ws3, N, K)
    precompute_bias_scale(cs3, ws3, b3, s3, N, act_dequant)
    pack_vnni(w3_row.bitcast[UInt8](), w3_packed, N, K)
    f32_reference(act3, w3_row, b3, s3, r3, SL3, N, K)

    var ctx3 = GemvCtx(Int(act3), Int(d3), N, K, SL3)
    int8_gemv(ctx3, Int(w3_packed), Int(b3), Int(s3), pool).join()

    _ = compare("multi-row [4x576x576]", d3, r3, SL3 * N)

    print("\ndone")
