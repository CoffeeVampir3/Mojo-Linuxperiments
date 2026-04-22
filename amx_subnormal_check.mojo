from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from experimental3.amx import (
    TILE_M, TILE_K, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    TileConfig, make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbsud, tdpbusd,
)
from simd_math import exp_f32, roundeven

comptime HPG = 6
comptime HEAD_DIM = 128
comptime WIDTH = 16
comptime NUM_K_STEPS = HEAD_DIM // K_STEP
comptime V_CHANNEL_GROUPS = HEAD_DIM // WIDTH
comptime SIMD_W = simd_width_of[DType.float32]()
comptime K_PG_BYTES = HEAD_DIM // VNNI_BLK * WIDTH * VNNI_BLK
comptime SCORE_STRIDE = TILE_M * WIDTH
comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
comptime V_PG_BYTES_VNNI = V_CHANNEL_GROUPS * V_CG_BYTES

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]


def alloc_zeroed[T: DType](count: Int) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
    var p = alloc[Scalar[T]](count)
    for i in range(count):
        (p + i)[] = Scalar[T](0)
    return UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=Int(p))


def fill_random_i8(ptr: I8Ptr, count: Int):
    var state = UInt64(0xDEADBEEF12345678)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = Scalar[DType.int8]((state >> 33).cast[DType.int8]())


def fill_random_u8(ptr: U8Ptr, count: Int):
    var state = UInt64(0xCAFEBABE87654321)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = UInt8((state >> 33).cast[DType.uint8]())


def phase2_channelwise(score_buf: I32Ptr, w_base: U8Ptr, wd_arr: F32Ptr, pg: Int):
    comptime W_PAGE_BYTES = TILE_M * K_STEP
    for bp in range(4):
        var s_base = score_buf + bp * SCORE_STRIDE
        var w_pg = w_base + bp * W_PAGE_BYTES
        for qh in range(HPG):
            var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
            var m = scores_raw.reduce_max()
            var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
            var w_max = exp_scores.reduce_max()
            if w_max < Float32(1e-10):
                (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                wd_arr[bp * HPG + qh] = Float32(0)
                continue
            var w_scale = 255.0 / w_max
            var w_u8 = roundeven(exp_scores * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
            (w_pg + qh * K_STEP).store(w_u8)
            wd_arr[bp * HPG + qh] = w_max / Float32(255.0)


def phase2_positionwise(
    score_buf: I32Ptr, w_base: U8Ptr, wd_arr: F32Ptr,
    v_scales: F32Ptr, pg: Int,
):
    comptime W_PAGE_BYTES = TILE_M * K_STEP
    for bp in range(4):
        var s_base = score_buf + bp * SCORE_STRIDE
        var w_pg = w_base + bp * W_PAGE_BYTES
        var v_sc = v_scales + (pg + bp) * WIDTH
        for qh in range(HPG):
            var scores_raw = (s_base + qh * WIDTH).load[width=WIDTH]().cast[DType.float32]()
            var m = scores_raw.reduce_max()
            var exp_scores = exp_f32[WIDTH](scores_raw * Float32(0.001) - m * Float32(0.001))
            var w_eff = exp_scores * v_sc.load[width=WIDTH]()
            var w_max = w_eff.reduce_max()
            if w_max < Float32(1e-10):
                (w_pg + qh * K_STEP).store(SIMD[DType.uint8, WIDTH](0))
                wd_arr[bp * HPG + qh] = Float32(0)
                continue
            var w_scale = 255.0 / w_max
            var w_u8 = roundeven(w_eff * w_scale).clamp(0.0, 255.0).cast[DType.uint8]()
            (w_pg + qh * K_STEP).store(w_u8)
            wd_arr[bp * HPG + qh] = w_max / Float32(255.0)


def main():
    _ = init_intel_amx()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    comptime NUM_PGS = 256
    comptime W_PAGE_BYTES = TILE_M * K_STEP

    var score_buf = alloc_zeroed[DType.int32](4 * SCORE_STRIDE)
    var w_base = alloc_zeroed[DType.uint8](4 * W_PAGE_BYTES)
    var wd_arr = alloc_zeroed[DType.float32](4 * HPG)
    var v_scales = alloc_zeroed[DType.float32](NUM_PGS * WIDTH)

    for i in range(4 * SCORE_STRIDE):
        score_buf[i] = Int32(i * 7 - 500)
    for i in range(NUM_PGS * WIDTH):
        v_scales[i] = Float32(0.01)

    comptime ITERS = 10000

    print("--- Phase 2 only: channelwise vs positionwise ---")
    print("Score data: synthetic i32, v_scales=0.01, " + String(ITERS) + " iterations")
    print("")

    var t0 = Int(perf_counter_ns())
    for it in range(ITERS):
        for pg in range(0, NUM_PGS, 4):
            phase2_channelwise(score_buf, w_base.bitcast[UInt8](), wd_arr, pg)
    var ch_ns = (Int(perf_counter_ns()) - t0) // ITERS
    print("channelwise phase2: " + String(ch_ns) + " ns (" +
          String(ch_ns // (NUM_PGS // 4)) + " ns/batch)")

    t0 = Int(perf_counter_ns())
    for it in range(ITERS):
        for pg in range(0, NUM_PGS, 4):
            phase2_positionwise(score_buf, w_base.bitcast[UInt8](), wd_arr, v_scales, pg)
    var pw_ns = (Int(perf_counter_ns()) - t0) // ITERS
    print("positionwise phase2: " + String(pw_ns) + " ns (" +
          String(pw_ns // (NUM_PGS // 4)) + " ns/batch)")
    print("ratio: " + String(pw_ns) + "/" + String(ch_ns) + " = " +
          String(pw_ns // max(ch_ns, 1)) + "x")

    print("")
    print("Run with: perf stat -e fp_assist.any ./amx_subnormal_check")
    print("to see if subnormal assists explain the gap.")
