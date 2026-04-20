"""FMA reduction chain: verification + microbenchmark of accumulator unrolling.

The baseline pattern found in rms_reduce_bf16, rms_reduce_f32, load_and_reduce,
and f32_gemv_row uses a single accumulator with a loop-carried FMA dependency:

    var acc = SIMD[f32, width](0)
    while k + width <= cols:
        acc = x.fma(w, acc)   # RAW on acc every iter
        k += width

Modern x86 FMA units pipeline at ~2 ops/cycle with ~4-cycle latency, so one
serial chain uses ~12-25% of available FMA throughput. Unrolling into n_acc
independent accumulators decouples the dependency chain; n_acc=4-8 typically
saturates the unit.

Two shapes tested:
  * rms_reduce_bf16   — sum(x^2) over bf16[cols]
  * f32_gemv_row      — dot(bf16[K], f32[K])

Per shape, three variants: n_acc=1 (baseline), 4, 8. Correctness is checked by
max abs/rel diff vs. the 1-accumulator result (FP associativity differs, but
relative error should stay near ~1e-6 at cols=3072).

Invoke: pixi run mojo fma_reduce_verification.mojo
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import abs

from std.algorithm.functional import vectorize


# =============================================================================
# Problem shape
# =============================================================================

comptime COLS = 3072   # HIDDEN for MiniMax M2.7


# =============================================================================
# Minimal RNG
# =============================================================================


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed
        for _ in range(8):
            _ = self.next()

    @always_inline
    def next(mut self) -> UInt64:
        var x = self.s
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.s = x
        return x

    @always_inline
    def uniform(mut self) -> Float32:
        return (Float32(self.next() & 0xFFFFFF) + Float32(1.0)) / Float32(0x1000001)


# =============================================================================
# rms_reduce_bf16 — sum(x^2) over bf16 input
# =============================================================================


@always_inline
def rms_reduce_bf16_serial[cols: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        k += width
    return vsum.reduce_add()


@always_inline
def rms_reduce_bf16_unrolled[cols: Int, n_acc: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime step = n_acc * width
    var accs = InlineArray[SIMD[DType.float32, width], n_acc](
        fill=SIMD[DType.float32, width](0))

    var k = 0
    while k + step <= cols:
        comptime for i in range(n_acc):
            var x = (src + k + i * width).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
        k += step

    # Tail (width-stride leftover).
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        k += width

    # Tree-reduce across accumulators.
    comptime for stride in range(1, n_acc):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, n_acc, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


# -----------------------------------------------------------------------------
# vectorize-based variants (stdlib). `vectorize[fn, width, unroll_factor=U]`
# unrolls the closure U times. If LLVM's reassociator sees the single captured
# accumulator and splits the chain, we'd get the same speedup as hand-rolled
# N-accumulator code without the ceremony.
# -----------------------------------------------------------------------------


@always_inline
def rms_reduce_bf16_vectorize[cols: Int, unroll: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    debug_assert(cols % width == 0, "cols must divide SIMD width")
    var vsum = SIMD[DType.float32, width](0)

    @parameter
    def body[w: Int](idx: Int) unified {read src, mut vsum}:
        var xw = (src + idx).load[width=w]().cast[DType.float32]()
        var x = rebind[SIMD[DType.float32, width]](xw)
        vsum = x.fma(x, vsum)

    vectorize[width, size=cols, unroll_factor=unroll](body)
    return vsum.reduce_add()


# =============================================================================
# f32_gemv_row — dot(bf16[K], f32[K])
# =============================================================================


@always_inline
def f32_gemv_row_serial[K: Int](
    act: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    weight_row: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width]()
        acc = a.fma(w, acc)
        k += width
    return acc.reduce_add()


@always_inline
def f32_gemv_row_unrolled[K: Int, n_acc: Int](
    act: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    weight_row: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime step = n_acc * width
    var accs = InlineArray[SIMD[DType.float32, width], n_acc](
        fill=SIMD[DType.float32, width](0))

    var k = 0
    while k + step <= K:
        comptime for i in range(n_acc):
            var a = (act + k + i * width).load[width=width]().cast[DType.float32]()
            var w = (weight_row + k + i * width).load[width=width]()
            accs[i] = a.fma(w, accs[i])
        k += step

    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width]()
        accs[0] = a.fma(w, accs[0])
        k += width

    comptime for stride in range(1, n_acc):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, n_acc, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


# =============================================================================
# Correctness
# =============================================================================


def run_correctness():
    var rng = Rng(seed=UInt64(42))
    var x_buf = InlineArray[Scalar[DType.bfloat16], COLS](uninitialized=True)
    var w_buf = InlineArray[Float32, COLS](uninitialized=True)
    var xp = UnsafePointer(to=x_buf[0])
    var wp = UnsafePointer(to=w_buf[0])

    # Values in [-1, 1]: realistic post-norm activation magnitudes.
    for i in range(COLS):
        xp[i] = Scalar[DType.bfloat16](Float32(rng.uniform() * Float32(2.0) - Float32(1.0)))
        wp[i] = rng.uniform() * Float32(2.0) - Float32(1.0)

    print("")
    print("=== CORRECTNESS (cols =", COLS, ") ===")

    var r1 = rms_reduce_bf16_serial[COLS](xp)
    var r4 = rms_reduce_bf16_unrolled[COLS, 4](xp)
    var r8 = rms_reduce_bf16_unrolled[COLS, 8](xp)
    var rv1 = rms_reduce_bf16_vectorize[COLS, 1](xp)
    var rv4 = rms_reduce_bf16_vectorize[COLS, 4](xp)
    print("  rms_reduce_bf16")
    print("    serial          :", r1)
    print("    4-accum         :", r4, " rel:", abs(r1 - r4) / abs(r1))
    print("    8-accum         :", r8, " rel:", abs(r1 - r8) / abs(r1))
    print("    vectorize u=1   :", rv1, " rel:", abs(r1 - rv1) / abs(r1))
    print("    vectorize u=4   :", rv4, " rel:", abs(r1 - rv4) / abs(r1))

    var g1 = f32_gemv_row_serial[COLS](xp, wp)
    var g4 = f32_gemv_row_unrolled[COLS, 4](xp, wp)
    var g8 = f32_gemv_row_unrolled[COLS, 8](xp, wp)
    print("  f32_gemv_row")
    print("    serial      :", g1)
    print("    4-accum     :", g4, " |diff|:", abs(g1 - g4),
          " rel:", abs(g1 - g4) / (abs(g1) + Float32(1e-20)))
    print("    8-accum     :", g8, " |diff|:", abs(g1 - g8),
          " rel:", abs(g1 - g8) / (abs(g1) + Float32(1e-20)))


# =============================================================================
# Benchmark
# =============================================================================


def time_rms_serial(
    xp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = rms_reduce_bf16_serial[COLS](xp)
        keep(r)
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_rms_unrolled[n_acc: Int](
    xp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = rms_reduce_bf16_unrolled[COLS, n_acc](xp)
        keep(r)
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_rms_vectorize[unroll: Int](
    xp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = rms_reduce_bf16_vectorize[COLS, unroll](xp)
        keep(r)
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_gemv_serial(
    xp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    wp: UnsafePointer[Float32, MutAnyOrigin], iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = f32_gemv_row_serial[COLS](xp, wp)
        keep(r)
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def time_gemv_unrolled[n_acc: Int](
    xp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    wp: UnsafePointer[Float32, MutAnyOrigin], iters: Int,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = f32_gemv_row_unrolled[COLS, n_acc](xp, wp)
        keep(r)
    var t1 = perf_counter_ns()
    return Int(t1 - t0)


def run_benchmark():
    comptime WARMUP = 2000
    comptime TRIALS = 50000

    var rng = Rng(seed=UInt64(123))
    var x_buf = InlineArray[Scalar[DType.bfloat16], COLS](uninitialized=True)
    var w_buf = InlineArray[Float32, COLS](uninitialized=True)
    var xp = UnsafePointer(to=x_buf[0])
    var wp = UnsafePointer(to=w_buf[0])
    for i in range(COLS):
        xp[i] = Scalar[DType.bfloat16](Float32(rng.uniform() * Float32(2.0) - Float32(1.0)))
        wp[i] = rng.uniform() * Float32(2.0) - Float32(1.0)

    _ = time_rms_serial(xp, WARMUP)
    _ = time_rms_unrolled[4](xp, WARMUP)
    _ = time_rms_unrolled[8](xp, WARMUP)
    _ = time_rms_vectorize[1](xp, WARMUP)
    _ = time_rms_vectorize[4](xp, WARMUP)
    _ = time_gemv_serial(xp, wp, WARMUP)
    _ = time_gemv_unrolled[4](xp, wp, WARMUP)
    _ = time_gemv_unrolled[8](xp, wp, WARMUP)

    var r_s = time_rms_serial(xp, TRIALS)
    var r_4 = time_rms_unrolled[4](xp, TRIALS)
    var r_8 = time_rms_unrolled[8](xp, TRIALS)
    var r_v1 = time_rms_vectorize[1](xp, TRIALS)
    var r_v4 = time_rms_vectorize[4](xp, TRIALS)
    var g_s = time_gemv_serial(xp, wp, TRIALS)
    var g_4 = time_gemv_unrolled[4](xp, wp, TRIALS)
    var g_8 = time_gemv_unrolled[8](xp, wp, TRIALS)

    print("")
    print("=== BENCHMARK (cols =", COLS, ", iters =", TRIALS, ") ===")
    print("  rms_reduce_bf16")
    print("    serial         ns/call:", r_s // TRIALS)
    print("    4-accum        ns/call:", r_4 // TRIALS,
          " speedup:", Float32(r_s) / Float32(r_4), "x")
    print("    8-accum        ns/call:", r_8 // TRIALS,
          " speedup:", Float32(r_s) / Float32(r_8), "x")
    print("    vectorize u=1  ns/call:", r_v1 // TRIALS,
          " speedup:", Float32(r_s) / Float32(r_v1), "x")
    print("    vectorize u=4  ns/call:", r_v4 // TRIALS,
          " speedup:", Float32(r_s) / Float32(r_v4), "x")
    print("  f32_gemv_row")
    print("    serial  ns/call:", g_s // TRIALS)
    print("    4-accum ns/call:", g_4 // TRIALS,
          " speedup:", Float32(g_s) / Float32(g_4), "x")
    print("    8-accum ns/call:", g_8 // TRIALS,
          " speedup:", Float32(g_s) / Float32(g_8), "x")


def main():
    run_correctness()
    run_benchmark()
