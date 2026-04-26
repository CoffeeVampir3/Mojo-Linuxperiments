"""Real perf benchmark with per-iteration sampling + median/percentile stats.

Records each iteration's wall time into a samples array, sorts, and reports
median + percentiles. Also alternates old/new across trials to amortize
cache/frequency state.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.benchmark import keep

from simd_math import set_subnormal_zeroing
from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq, rms_reduce_bf16,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs

from kernels.rmsnorm import (
    Bf16Spec, I8Spec,
    rmsnorm_n_bf16_row, rmsnorm_n_i8_row, rmsnorm_i8_and_bf16_row,
)


comptime HIDDEN = 3072
comptime FWHT_BLK = 128
comptime INNER = 100
comptime TRIALS = 5000


# ============================================================================
# OLD KERNEL BODIES — copied from git pre-refactor
# ============================================================================

@always_inline
def old_load_and_reduce[cols: Int, has_gamma: Bool](
    src: BF16Ptr, gamma: BF16Ptr, work: F32Ptr,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        comptime if has_gamma:
            var g = (gamma + off).load[width=width]().cast[DType.float32]()
            (work + off).store(x * g)
        else:
            (work + off).store(x)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            comptime if has_gamma:
                var g = (gamma + off).load[width=width]().cast[DType.float32]()
                (work + off).store(x * g)
            else:
                (work + off).store(x)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            (work + k).store(x * g)
        else:
            (work + k).store(x)
        k += width
    return tree_reduce_accs(accs)


@always_inline
def old_normalize_inplace[cols: Int](work: F32Ptr, inv: Float32):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
        (work + k).store((work + k).load[width=width]() * vinv)
        k += width


@always_inline
def old_fwht_rotate[cols: Int, block: Int](work: F32Ptr):
    for b in range(cols // block):
        fwht_block[block](work + b * block)


@always_inline
def old_rmsnorm_fwht_quant_row[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scales: F32Ptr, eps: Float32,
):
    var sum_sq = old_load_and_reduce[cols, has_gamma](src, gamma, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    old_normalize_inplace[cols](work, inv)
    old_fwht_rotate[cols, block](work)
    comptime if per_block:
        for b in range(cols // block):
            scales[b] = absmax_quantize_i8[block](
                work + b * block, qi + b * block)
    else:
        scales[0] = absmax_quantize_i8[cols](work, qi)


@always_inline
def old_rmsnorm_dual_output_row[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed_bf16: BF16Ptr,
    eps: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width

    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work + off).store(x)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work + off).store(x)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work + k).store(x)
        k += width
    var sum_sq = tree_reduce_accs(accs)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)

    k = 0
    while k < cols:
        var x = (work + k).load[width=width]()
        var normed = x * vinv
        var sg = (split_gamma + k).load[width=width]().cast[DType.float32]()
        var fg = (full_gamma + k).load[width=width]().cast[DType.float32]()
        (work + k).store(normed * sg)
        (normed_bf16 + k).store((normed * fg).cast[DType.bfloat16]())
        k += width

    old_fwht_rotate[cols, block](work)
    scale[0] = absmax_quantize_i8[cols](work, qi)


@always_inline
def old_rmsnorm_bf16_row[cols: Int, has_gamma: Bool, has_residual: Bool](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce_bf16[cols](src)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
        var v = (src + k).load[width=width]().cast[DType.float32]()
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            var normed = v * vinv * g
            comptime if has_residual:
                var x = (dst + k).load[width=width]().cast[DType.float32]()
                (dst + k).store((x + normed).cast[DType.bfloat16]())
            else:
                (dst + k).store(normed.cast[DType.bfloat16]())
        else:
            (dst + k).store((v * vinv).cast[DType.bfloat16]())
        k += width


# ============================================================================
# NEW unified — wrappers
# ============================================================================

@always_inline
def new_single_i8[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scale: F32Ptr, eps: Float32,
):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[I8Ptr, 1] = [qi]
    var scales: InlineArray[F32Ptr, 1] = [scale]
    rmsnorm_n_i8_row[cols, block, 1,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
    ](src, work, eps, gammas, dsts, scales)


@always_inline
def old_load_and_reduce_dual[cols: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work_a + off).store(x * ga)
        (work_b + off).store(x * gb)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
            var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work_a + off).store(x * ga)
            (work_b + off).store(x * gb)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + k).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work_a + k).store(x * ga)
        (work_b + k).store(x * gb)
        k += width
    return tree_reduce_accs(accs)


@always_inline
def old_dual_i8[cols: Int, block: Int](
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr,
    qa: I8Ptr, qb: I8Ptr,
    wa: F32Ptr, wb: F32Ptr, sa: F32Ptr, sb: F32Ptr, eps: Float32,
):
    var sum_sq = old_load_and_reduce_dual[cols](src, ga, gb, wa, wb)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    old_normalize_inplace[cols](wa, inv)
    old_normalize_inplace[cols](wb, inv)
    old_fwht_rotate[cols, block](wa)
    old_fwht_rotate[cols, block](wb)
    sa[0] = absmax_quantize_i8[cols](wa, qa)
    sb[0] = absmax_quantize_i8[cols](wb, qb)


@always_inline
def new_dual_i8[cols: Int, block: Int](
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr,
    qa: I8Ptr, qb: I8Ptr,
    work: F32Ptr, sa: F32Ptr, sb: F32Ptr, eps: Float32,
):
    var gammas: InlineArray[BF16Ptr, 2] = [ga, gb]
    var dsts: InlineArray[I8Ptr, 2] = [qa, qb]
    var scales: InlineArray[F32Ptr, 2] = [sa, sb]
    rmsnorm_n_i8_row[cols, block, 2,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
        I8Spec(has_gamma=True, fwht=True, per_block=False),
    ](src, work, eps, gammas, dsts, scales)


@always_inline
def new_dual_output[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed_bf16: BF16Ptr,
    eps: Float32,
):
    rmsnorm_i8_and_bf16_row[cols, block](
        src, split_gamma, full_gamma, qi, scale, normed_bf16, work, eps)


@always_inline
def new_bf16[cols: Int](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[BF16Ptr, 1] = [dst]
    rmsnorm_n_bf16_row[cols, 1,
        Bf16Spec(has_gamma=True, has_residual=False),
    ](src, eps, gammas, dsts)


# ============================================================================
# Bench loops — INNER iters per sample, TRIALS samples
# ============================================================================

@no_inline
def trial_old_single_i8(
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        old_rmsnorm_fwht_quant_row[HIDDEN, FWHT_BLK, True, False](
            src, gamma, qi, work, scale, Float32(1e-6))
        keep(qi)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_new_single_i8(
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        new_single_i8[HIDDEN, FWHT_BLK](
            src, gamma, qi, work, scale, Float32(1e-6))
        keep(qi)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_old_dual_output(
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        old_rmsnorm_dual_output_row[HIDDEN, FWHT_BLK](
            src, sg, fg, qi, work, scale, normed, Float32(1e-6))
        keep(qi)
        keep(normed)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_new_dual_output(
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        new_dual_output[HIDDEN, FWHT_BLK](
            src, sg, fg, qi, work, scale, normed, Float32(1e-6))
        keep(qi)
        keep(normed)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_old_dual_i8(
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr,
    qa: I8Ptr, qb: I8Ptr,
    wa: F32Ptr, wb: F32Ptr, sa: F32Ptr, sb: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        old_dual_i8[HIDDEN, FWHT_BLK](
            src, ga, gb, qa, qb, wa, wb, sa, sb, Float32(1e-6))
        keep(qa)
        keep(qb)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_new_dual_i8(
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr,
    qa: I8Ptr, qb: I8Ptr,
    work: F32Ptr, sa: F32Ptr, sb: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        new_dual_i8[HIDDEN, FWHT_BLK](
            src, ga, gb, qa, qb, work, sa, sb, Float32(1e-6))
        keep(qa)
        keep(qb)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_old_bf16(src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        old_rmsnorm_bf16_row[HIDDEN, True, False](
            src, gamma, dst, Float32(1e-6))
        keep(dst)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_new_bf16(src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        new_bf16[HIDDEN](src, gamma, dst, Float32(1e-6))
        keep(dst)
    return Int(perf_counter_ns() - t0)


def insertion_sort(p: UnsafePointer[Int, MutAnyOrigin], n: Int):
    for i in range(1, n):
        var x = p[i]
        var j = i - 1
        while j >= 0 and p[j] > x:
            p[j + 1] = p[j]
            j -= 1
        p[j + 1] = x


def report(label: String,
    old_samples: UnsafePointer[Int, MutAnyOrigin],
    new_samples: UnsafePointer[Int, MutAnyOrigin],
    n_trials: Int, inner: Int):
    insertion_sort(old_samples, n_trials)
    insertion_sort(new_samples, n_trials)
    var p10 = n_trials // 10
    var p50 = n_trials // 2
    var p90 = n_trials - n_trials // 10 - 1

    var old_p50_per = Float64(old_samples[p50]) / Float64(inner)
    var new_p50_per = Float64(new_samples[p50]) / Float64(inner)
    var old_p10_per = Float64(old_samples[p10]) / Float64(inner)
    var new_p10_per = Float64(new_samples[p10]) / Float64(inner)
    var old_p90_per = Float64(old_samples[p90]) / Float64(inner)
    var new_p90_per = Float64(new_samples[p90]) / Float64(inner)
    var pct = (new_p50_per - old_p50_per) / old_p50_per * 100.0

    print(label)
    print("  old: p10=", old_p10_per, " p50=", old_p50_per,
          " p90=", old_p90_per, " ns/iter")
    print("  new: p10=", new_p10_per, " p50=", new_p50_per,
          " p90=", new_p90_per, " ns/iter")
    print("  delta(p50):", pct, "% (negative = unified faster)")
    print()


def fill_bf16(p: BF16Ptr, count: Int, seed: UInt64):
    var s = seed
    for i in range(count):
        s = s * 6364136223846793005 + 1442695040888963407
        p[i] = Scalar[DType.bfloat16](
            (Float32(Int((s >> 32) & 0xFFFF)) / 65536.0 - 0.5))


def main():
    set_subnormal_zeroing()

    var src = alloc[Scalar[DType.bfloat16]](HIDDEN, alignment=64)
    var ga  = alloc[Scalar[DType.bfloat16]](HIDDEN, alignment=64)
    var gb  = alloc[Scalar[DType.bfloat16]](HIDDEN, alignment=64)
    var qi  = alloc[Scalar[DType.int8]](HIDDEN, alignment=64)
    var wa  = alloc[Float32](HIDDEN, alignment=64)
    var sa  = alloc[Float32](HIDDEN // FWHT_BLK, alignment=64)
    var dst = alloc[Scalar[DType.bfloat16]](HIDDEN, alignment=64)

    fill_bf16(src, HIDDEN, UInt64(1))
    fill_bf16(ga, HIDDEN, UInt64(2))
    fill_bf16(gb, HIDDEN, UInt64(3))

    var old_samples = alloc[Int](TRIALS, alignment=64)
    var new_samples = alloc[Int](TRIALS, alignment=64)

    print("=" * 70)
    print("RMSNorm refactor benchmark — cols=", HIDDEN, " fwht_blk=", FWHT_BLK)
    print("inner=", INNER, " trials=", TRIALS,
          " (alternating old/new per trial)")
    print("=" * 70)
    print()

    # ---- S1: single i8 ----
    for w in range(100):
        _ = trial_old_single_i8(src, ga, qi, wa, sa)
        _ = trial_new_single_i8(src, ga, qi, wa, sa)
    for t in range(TRIALS):
        old_samples[t] = trial_old_single_i8(src, ga, qi, wa, sa)
        new_samples[t] = trial_new_single_i8(src, ga, qi, wa, sa)
    report("S1 single i8 + gamma + FWHT (attn_quantize)",
        old_samples, new_samples, TRIALS, INNER)

    # ---- S4: dual i8 (Gemma4 dense+expert) ----
    var qb_arr = alloc[Scalar[DType.int8]](HIDDEN, alignment=64)
    var wb_arr = alloc[Float32](HIDDEN, alignment=64)
    var sb_arr = alloc[Float32](HIDDEN // FWHT_BLK, alignment=64)
    for w in range(100):
        _ = trial_old_dual_i8(src, ga, gb, qi, qb_arr, wa, wb_arr, sa, sb_arr)
        _ = trial_new_dual_i8(src, ga, gb, qi, qb_arr, wa, sa, sb_arr)
    for t in range(TRIALS):
        old_samples[t] = trial_old_dual_i8(
            src, ga, gb, qi, qb_arr, wa, wb_arr, sa, sb_arr)
        new_samples[t] = trial_new_dual_i8(
            src, ga, gb, qi, qb_arr, wa, sa, sb_arr)
    report("S4 dual i8 + i8 (Gemma4 dense+expert)",
        old_samples, new_samples, TRIALS, INNER)
    qb_arr.free()
    wb_arr.free()
    sb_arr.free()

    # ---- S5: minimax dual_output ----
    for w in range(100):
        _ = trial_old_dual_output(src, ga, gb, qi, wa, sa, dst)
        _ = trial_new_dual_output(src, ga, gb, qi, wa, sa, dst)
    for t in range(TRIALS):
        old_samples[t] = trial_old_dual_output(src, ga, gb, qi, wa, sa, dst)
        new_samples[t] = trial_new_dual_output(src, ga, gb, qi, wa, sa, dst)
    report("S5 i8 + bf16 (minimax dual_norm)",
        old_samples, new_samples, TRIALS, INNER)

    # ---- S7: bf16 single ----
    for w in range(100):
        _ = trial_old_bf16(src, ga, dst)
        _ = trial_new_bf16(src, ga, dst)
    for t in range(TRIALS):
        old_samples[t] = trial_old_bf16(src, ga, dst)
        new_samples[t] = trial_new_bf16(src, ga, dst)
    report("S7 single bf16 + gamma (standard rmsnorm)",
        old_samples, new_samples, TRIALS, INNER)

    src.free()
    ga.free()
    gb.free()
    qi.free()
    wa.free()
    sa.free()
    dst.free()
    old_samples.free()
    new_samples.free()
