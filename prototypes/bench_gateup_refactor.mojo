"""Real perf benchmark for fused_w1_w3_silu refactor with proper sampling."""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.benchmark import keep

from simd_math import set_subnormal_zeroing
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8

from kernels.moe import fused_w1w3_gemv_row, fused_gateup_quant_row
from minimax.kernels.activations import silu_mul, silu_f32


comptime INTERMEDIATE = 1536
comptime K = 3072
comptime FWHT_BLK = 128
comptime INNER = 10
comptime TRIALS = 2000


@always_inline
def old_silu_row(
    act_i8: I8Ptr, act_dequant: Float32,
    w1_packed: I8Ptr, w1_scale: F32Ptr, w1_colsum: F32Ptr,
    w3_packed: I8Ptr, w3_scale: F32Ptr, w3_colsum: F32Ptr,
    qi_row: I8Ptr, blk_row: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var local_n = 0
    while local_n < INTERMEDIATE:
        var n_off = local_n
        var gate_buf = InlineArray[Float32, FWHT_BLK](fill=Float32(0))
        var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
        var up_buf = InlineArray[Float32, FWHT_BLK](fill=Float32(0))
        var up = UnsafePointer(to=up_buf).bitcast[Float32]()

        fused_w1w3_gemv_row[FWHT_BLK, K](
            act_i8,
            w1_packed + n_off * K,
            w3_packed + n_off * K,
            act_dequant,
            w1_scale + n_off, w1_colsum + n_off,
            w3_scale + n_off, w3_colsum + n_off,
            gate, up)

        var k = 0
        while k + width <= FWHT_BLK:
            (gate + k).store(silu_mul(
                (gate + k).load[width=width](),
                (up + k).load[width=width]()))
            k += width

        fwht_block[FWHT_BLK](gate)
        blk_row[local_n // FWHT_BLK] = absmax_quantize_i8[FWHT_BLK](
            gate, qi_row + local_n)

        local_n += FWHT_BLK


@no_inline
def trial_old(
    act: I8Ptr, act_sc: Float32,
    w1: I8Ptr, w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3: I8Ptr, w3_sc: F32Ptr, w3_cs: F32Ptr,
    qi: I8Ptr, blk: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        old_silu_row(act, act_sc,
            w1, w1_sc, w1_cs, w3, w3_sc, w3_cs,
            qi, blk)
        keep(qi)
    return Int(perf_counter_ns() - t0)


@no_inline
def trial_new(
    act: I8Ptr, act_sc: Float32,
    w1: I8Ptr, w1_sc: F32Ptr, w1_cs: F32Ptr,
    w3: I8Ptr, w3_sc: F32Ptr, w3_cs: F32Ptr,
    qi: I8Ptr, blk: F32Ptr,
) -> Int:
    var t0 = perf_counter_ns()
    for _ in range(INNER):
        fused_gateup_quant_row[FWHT_BLK, K, True, silu_f32](
            act,
            w1, w1_sc, w1_cs,
            w3, w3_sc, w3_cs,
            act_sc,
            qi, blk, 0, INTERMEDIATE)
        keep(qi)
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

    var old_p50 = Float64(old_samples[p50]) / Float64(inner) / 1000.0
    var new_p50 = Float64(new_samples[p50]) / Float64(inner) / 1000.0
    var old_p10 = Float64(old_samples[p10]) / Float64(inner) / 1000.0
    var new_p10 = Float64(new_samples[p10]) / Float64(inner) / 1000.0
    var old_p90 = Float64(old_samples[p90]) / Float64(inner) / 1000.0
    var new_p90 = Float64(new_samples[p90]) / Float64(inner) / 1000.0
    var pct = (new_p50 - old_p50) / old_p50 * 100.0

    print(label)
    print("  old: p10=", old_p10, " p50=", old_p50, " p90=", old_p90, " us/iter")
    print("  new: p10=", new_p10, " p50=", new_p50, " p90=", new_p90, " us/iter")
    print("  delta(p50):", pct, "% (negative = unified faster)")
    print()


def fill_i8(p: I8Ptr, count: Int, seed: UInt64):
    var s = seed
    for i in range(count):
        s = s * 6364136223846793005 + 1442695040888963407
        p[i] = Scalar[DType.int8](Int((s >> 32) & 0xFF) - 128)


def fill_f32(p: F32Ptr, count: Int, val: Float32):
    for i in range(count):
        p[i] = val


def main():
    set_subnormal_zeroing()

    var act = alloc[Scalar[DType.int8]](K, alignment=64)
    var w1 = alloc[Scalar[DType.int8]](INTERMEDIATE * K, alignment=64)
    var w3 = alloc[Scalar[DType.int8]](INTERMEDIATE * K, alignment=64)
    var w1_sc = alloc[Float32](INTERMEDIATE, alignment=64)
    var w3_sc = alloc[Float32](INTERMEDIATE, alignment=64)
    var w1_cs = alloc[Float32](INTERMEDIATE, alignment=64)
    var w3_cs = alloc[Float32](INTERMEDIATE, alignment=64)
    var qi = alloc[Scalar[DType.int8]](INTERMEDIATE, alignment=64)
    var blk = alloc[Float32](INTERMEDIATE // FWHT_BLK, alignment=64)

    fill_i8(act, K, UInt64(1))
    fill_i8(w1, INTERMEDIATE * K, UInt64(2))
    fill_i8(w3, INTERMEDIATE * K, UInt64(3))
    fill_f32(w1_sc, INTERMEDIATE, Float32(0.01))
    fill_f32(w3_sc, INTERMEDIATE, Float32(0.012))
    fill_f32(w1_cs, INTERMEDIATE, Float32(0.5))
    fill_f32(w3_cs, INTERMEDIATE, Float32(0.7))

    var old_samples = alloc[Int](TRIALS, alignment=64)
    var new_samples = alloc[Int](TRIALS, alignment=64)

    var act_sc = Float32(1.5) / Float32(127)

    print("=" * 70)
    print("fused_w1_w3_silu refactor benchmark")
    print("intermediate=", INTERMEDIATE, " K=", K, " fwht_blk=", FWHT_BLK)
    print("inner=", INNER, " trials=", TRIALS,
          " (alternating old/new per trial)")
    print("=" * 70)
    print()

    for w in range(50):
        _ = trial_old(act, act_sc, w1, w1_sc, w1_cs, w3, w3_sc, w3_cs, qi, blk)
        _ = trial_new(act, act_sc, w1, w1_sc, w1_cs, w3, w3_sc, w3_cs, qi, blk)

    for t in range(TRIALS):
        old_samples[t] = trial_old(
            act, act_sc, w1, w1_sc, w1_cs, w3, w3_sc, w3_cs, qi, blk)
        new_samples[t] = trial_new(
            act, act_sc, w1, w1_sc, w1_cs, w3, w3_sc, w3_cs, qi, blk)

    report("SiLU minimax (intermediate=1536, K=3072)",
        old_samples, new_samples, TRIALS, INNER)

    act.free()
    w1.free()
    w3.free()
    w1_sc.free()
    w3_sc.free()
    w1_cs.free()
    w3_cs.free()
    qi.free()
    blk.free()
    old_samples.free()
    new_samples.free()
