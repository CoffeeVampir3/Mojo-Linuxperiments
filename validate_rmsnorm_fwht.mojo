"""Validate rmsnorm_fwht_quantize against scalar f32 reference."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from numa import NumaInfo, NumaTopology
from numa.arena import NumaArena
from threading.burst_threading import BurstPool
from simd_math import sqrt

from experimental2.kernels.rmsnorm_fwht_quantize import (
    rmsnorm_fwht_quantize, fwht_block,
)


def scalar_fwht(buf: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    """Scalar in-place FWHT for reference."""
    var h = 1
    while h < n:
        var i = 0
        while i < n:
            for j in range(h):
                var a = buf[i + j]
                var b = buf[i + j + h]
                buf[i + j] = a + b
                buf[i + j + h] = a - b
            i += h * 2
        h *= 2
    var sc = Float32(1.0) / sqrt[DType.float32, 1](Float32(n))
    for i in range(n):
        buf[i] = buf[i] * sc


def f32_reference(
    inp_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int, cols: Int, block: Int,
    s_act: Float32,
):
    """Scalar reference: RMSNorm -> block FWHT -> fixed-scale quantize."""
    var quant_inv = Float32(127) / s_act
    for m in range(seq_len):
        var row_in = inp_bf16 + m * cols
        var row_qi = dst_i8 + m * cols

        # Load bf16 -> f32
        var sum_sq = Float32(0)
        for k in range(cols):
            var v = Float32(row_in[k])
            work[k] = v
            sum_sq += v * v

        # RMSNorm (no gamma)
        var rms = sqrt[DType.float32, 1](sum_sq / Float32(cols) + 1e-5)
        var inv_rms = Float32(1.0) / rms
        for k in range(cols):
            work[k] = work[k] * inv_rms

        # Block-diagonal FWHT
        for b in range(cols // block):
            scalar_fwht(work + b * block, block)

        # Fixed-scale quantize
        for k in range(cols):
            var q = Int(work[k] * quant_inv + (Float32(0.5) if work[k] >= 0 else Float32(-0.5)))
            if q > 127: q = 127
            if q < -128: q = -128
            row_qi[k] = Scalar[DType.int8](q)


def compare_i8(
    name: String,
    got: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    expected: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    count: Int,
) -> Bool:
    var err_sum = Float64(0)
    var max_err = Int(0)
    var exact = 0
    for i in range(count):
        var g = Int(got[i])
        var e = Int(expected[i])
        var d = g - e
        if d < 0: d = -d
        err_sum += Float64(d)
        if d > max_err: max_err = d
        if d == 0: exact += 1
    var avg_err = err_sum / Float64(count)
    var pct_exact = Float64(exact) / Float64(count) * 100
    print("  " + name + ": avg_err=" + String(avg_err)
        + " max_err=" + String(max_err)
        + " exact=" + String(pct_exact) + "%")
    if avg_err < 1.0 and max_err <= 1:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var arena = NumaArena[](topo[0], 64 * 1024 * 1024)
    var pool = BurstPool[].for_topology(numa, topo[0], stack_size=2 * 1024 * 1024)
    print("workers: " + String(pool.get_capacity()))

    comptime COLS = 576
    comptime BLOCK = 64
    var s_act = Float32(2.5)

    # =====================================================================
    # Test 1: Single row (decode)
    # =====================================================================
    print("\n=== Test 1: single row [1, " + String(COLS) + "] block=" + String(BLOCK) + " ===")

    var mark = arena.mark()
    var inp = arena.alloc[Scalar[DType.bfloat16]](COLS)
    var qi_kernel = arena.alloc[Scalar[DType.int8]](COLS)
    var qi_ref = arena.alloc[Scalar[DType.int8]](COLS)
    var work_kernel = arena.alloc[Float32](COLS * pool.get_capacity())
    var work_ref = arena.alloc[Float32](COLS)

    for i in range(COLS):
        inp[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 64.0)

    f32_reference(inp, qi_ref, work_ref, 1, COLS, BLOCK, s_act)
    rmsnorm_fwht_quantize[COLS, BLOCK](
        Int(inp), Int(qi_kernel), Int(work_kernel), 1, s_act, pool,
    ).join()

    _ = compare_i8("decode [1x576]", qi_kernel, qi_ref, COLS)
    arena.reset_to(mark)

    # =====================================================================
    # Test 2: Multi-row (prefill)
    # =====================================================================
    comptime SL = 8
    print("\n=== Test 2: prefill [" + String(SL) + ", " + String(COLS) + "] block=" + String(BLOCK) + " ===")

    var inp2 = arena.alloc[Scalar[DType.bfloat16]](SL * COLS)
    var qi2_kernel = arena.alloc[Scalar[DType.int8]](SL * COLS)
    var qi2_ref = arena.alloc[Scalar[DType.int8]](SL * COLS)
    var work2_kernel = arena.alloc[Float32](COLS * pool.get_capacity())
    var work2_ref = arena.alloc[Float32](COLS)

    for i in range(SL * COLS):
        inp2[i] = Scalar[DType.bfloat16](Float32((i * 7 + 13) % 256 - 128) / 64.0)

    f32_reference(inp2, qi2_ref, work2_ref, SL, COLS, BLOCK, s_act)
    rmsnorm_fwht_quantize[COLS, BLOCK](
        Int(inp2), Int(qi2_kernel), Int(work2_kernel), SL, s_act, pool,
    ).join()

    _ = compare_i8("prefill [8x576]", qi2_kernel, qi2_ref, SL * COLS)

    # =====================================================================
    # Test 3: FWHT correctness (standalone block)
    # =====================================================================
    print("\n=== Test 3: FWHT block correctness ===")
    var fwht_in = arena.alloc[Float32](BLOCK)
    var fwht_ref = arena.alloc[Float32](BLOCK)
    for i in range(BLOCK):
        var v = Float32(i) - Float32(BLOCK // 2)
        fwht_in[i] = v
        fwht_ref[i] = v

    fwht_block[BLOCK](fwht_in)
    scalar_fwht(fwht_ref, BLOCK)

    var fwht_err = Float64(0)
    for i in range(BLOCK):
        var d = Float64(fwht_in[i]) - Float64(fwht_ref[i])
        if d < 0: d = -d
        fwht_err += d
    var avg = fwht_err / Float64(BLOCK)
    print("  fwht avg_err=" + String(avg))
    if avg < 1e-5:
        print("  PASS")
    else:
        print("  FAIL")

    print("\ndone")
    _ = pool
    _ = arena
