from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.sys import llvm_intrinsic
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_i8_config, init_intel_amx, ldtilecfg,
    tilezero, tileload, tilestore, tdpbsud, tdpbusd,
)
from experimental3.kernels.dot_prod import vpdpbusd, bcast_4u8_vnni
from simd_math import exp_f32, exp_f32_fast, roundeven, set_subnormal_zeroing
from notstdcollections import AlignedInlineArray


comptime Q_TILE = 16
comptime HEAD_DIM = 128
comptime WIDTH = 16
comptime K_PG_BYTES = HEAD_DIM // VNNI_BLK * WIDTH * VNNI_BLK
comptime V_CHANNEL_GROUPS = HEAD_DIM // WIDTH
comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
comptime V_PG_BYTES = V_CHANNEL_GROUPS * V_CG_BYTES
comptime SQ_COUNT = WIDTH // VNNI_BLK
comptime SQ_BYTES = WIDTH * VNNI_BLK
comptime SIMD_W = simd_width_of[DType.float32]()

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


def fill_random_f32(ptr: F32Ptr, count: Int, scale: Float32):
    var state = UInt64(0xFEEDFACE11111111)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var frac = Float32(Int(state >> 33)) / Float32(2147483648)
        ptr[i] = frac * scale + Float32(0.001)


# =============================================================================
# Reference: single-Q sequential attention (one head, one position at a time)
# Sweeps all PGs for one Q position, producing f32 v_acc output.
# =============================================================================


def reference_single_q_attn(
    q_ptr: I8Ptr,
    qi_bias: Float32,
    q_factor: Float32,
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    context_len: Int,
    v_out: F32Ptr,
):
    comptime Q_DENOM = Float32(127) * Float32(127)
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3

    var running_max = Float32(-1e30)
    var running_sum = Float32(0)
    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)

    for d in range(HEAD_DIM):
        v_out[d] = Float32(0)

    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var w_arr = AlignedInlineArray[UInt8, K_STEP](fill=UInt8(0))
    var w_base = w_arr.unsafe_ptr()

    for pg in range(padded_pgs):
        var group_start = pg * WIDTH
        var ctx_valid = (valid_lanes + Int32(group_start)).lt(
            SIMD[DType.int32, WIDTH](context_len))

        var k_pg = k_base + pg * K_PG_BYTES
        var raw = SIMD[DType.int32, WIDTH](0)
        for kstep in range(HEAD_DIM // K_STEP):
            var q_chunk = (q_ptr + kstep * K_STEP).load[width=K_STEP]()
            var k_chunk_base = k_pg + kstep * WIDTH * VNNI_BLK
            for lane in range(WIDTH):
                var dot = Int32(0)
                for b in range(K_STEP):
                    var ki = k_chunk_base[lane * VNNI_BLK + (b % VNNI_BLK) + (b // VNNI_BLK) * WIDTH * VNNI_BLK]
                    dot += Int32(q_chunk[b]) * Int32(ki.cast[DType.int8]())
                raw[lane] += dot

        var k_sc = k_scales + pg * WIDTH
        var scores = (raw.cast[DType.float32]() - qi_bias) * q_factor * k_sc.load[width=WIDTH]()
        scores = ctx_valid.select(scores, neg_inf)

        var pg_max = scores.reduce_max()
        var new_max = max(running_max, pg_max)
        if running_sum > 0 and new_max > running_max:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            running_sum *= rescale
            for d in range(HEAD_DIM):
                v_out[d] *= rescale
        running_max = new_max

        var exp_scores = ctx_valid.select(
            exp_f32_fast[WIDTH](scores - running_max), zero_vec)
        running_sum += exp_scores.reduce_add()

        var v_sc = v_scales + pg * WIDTH
        var w_eff = exp_scores * v_sc.load[width=WIDTH]()
        var w_max = w_eff.reduce_max()
        if w_max < Float32(1e-10):
            continue
        var w_scale_val = 255.0 / w_max
        var w_u8 = roundeven(w_eff * w_scale_val).clamp(0.0, 255.0).cast[DType.uint8]()
        var wd = w_max / 255.0
        (w_base).store[width=WIDTH](w_u8)

        var v_pg = v_base + pg * V_PG_BYTES
        for cg in range(V_CHANNEL_GROUPS):
            var v_cg = v_pg + cg * V_CG_BYTES
            var dot = SIMD[DType.int32, WIDTH](0)
            comptime for sq in range(SQ_COUNT):
                dot = vpdpbusd[WIDTH](
                    dot,
                    bcast_4u8_vnni[WIDTH](w_base + sq * VNNI_BLK),
                    (v_cg + sq * SQ_BYTES).load[width=WIDTH * VNNI_BLK]())
            var acc_ptr = v_out + cg * WIDTH
            acc_ptr.store(acc_ptr.load[width=WIDTH]() + dot.cast[DType.float32]() * wd)

    if running_sum > Float32(1e-10):
        var inv = Float32(1) / (Float32(127) * running_sum)
        for d in range(HEAD_DIM):
            v_out[d] *= inv

    _ = w_arr


# =============================================================================
# Prefill kernel: multi-Q AMX attention (AMX scoring + AMX V-agg)
# Processes up to Q_TILE=16 query positions per call, one head at a time.
# =============================================================================


def amx_prefill_attn(
    q_base: I8Ptr,
    qi_biases: F32Ptr,
    q_factors: F32Ptr,
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    q_start: Int,
    q_count: Int,
    context_len: Int,
    v_out: F32Ptr,
):
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var actual_q = min(q_count, Q_TILE)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var running_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, Q_TILE](fill=Float32(0))
    var v_acc_arr = AlignedInlineArray[Float32, Q_TILE * HEAD_DIM](fill=Float32(0))
    var v_acc = v_acc_arr.unsafe_ptr()

    var q_arr = AlignedInlineArray[Scalar[DType.int8], Q_TILE * HEAD_DIM](
        fill=Scalar[DType.int8](0))
    var q_ptr = q_arr.unsafe_ptr()
    for r in range(actual_q):
        var src = q_base + r * HEAD_DIM
        var dst = q_ptr + r * HEAD_DIM
        comptime for chunk in range(HEAD_DIM // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    comptime SCORE_STRIDE_PREFILL = Q_TILE * 4 * WIDTH
    var score_arr = AlignedInlineArray[Int32, SCORE_STRIDE_PREFILL](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    comptime W_TILE_BYTES = Q_TILE * K_STEP
    var w_arr = AlignedInlineArray[UInt8, W_TILE_BYTES](fill=UInt8(0))
    var w_buf = w_arr.unsafe_ptr()

    var wd_arr = InlineArray[Float32, Q_TILE](fill=Float32(0))

    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    comptime V_GATHER_BYTES = 4 * V_CG_BYTES
    var v_tile_even = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_tile_odd = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_even_ptr = v_tile_even.unsafe_ptr()
    var v_odd_ptr = v_tile_odd.unsafe_ptr()

    var result_arr = AlignedInlineArray[Int32, 2 * TILE_M * TILE_N](
        uninitialized=True)
    var result_ptr = result_arr.unsafe_ptr()

    var pg = 0
    while pg < padded_pgs:
        # ── Phase 1: AMX Score 4 PGs (2:2:4) ──
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        var k0 = k_base + pg * K_PG_BYTES
        var k1 = k_base + (pg + 1) * K_PG_BYTES
        var k2 = k_base + (pg + 2) * K_PG_BYTES
        var k3 = k_base + (pg + 3) * K_PG_BYTES

        tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        # Interleaved tilestore: row r has all 64 K-position scores contiguous
        tilestore[4, DType.int32](score_ptr, 4 * WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + WIDTH, 4 * WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * WIDTH, 4 * WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * WIDTH, 4 * WIDTH * 4)

        # ── Phase 2a: Dequant + causal/context mask + find batch max per row ──
        var batch_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
        for r in range(actual_q):
            var q_pos = q_start + r
            var row_base = score_ptr + r * 4 * WIDTH
            for bp in range(4):
                var group_start = (pg + bp) * WIDTH
                var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                    SIMD[DType.int32, WIDTH](context_len))
                var causal_valid = (valid_lanes + Int32(group_start)).le(
                    SIMD[DType.int32, WIDTH](q_pos))
                var valid = ctx_valid & causal_valid

                var raw = (row_base + bp * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var k_sc = k_scales + (pg + bp) * WIDTH
                var scores = (raw - qi_biases[r]) * q_factors[r] * k_sc.load[
                    width=WIDTH]()
                scores = valid.select(scores, neg_inf)
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                var pg_max = scores.reduce_max()
                if pg_max > batch_max[r]:
                    batch_max[r] = pg_max

        # ── Phase 2b: Rescale running state, exp, W packing ──
        for r in range(actual_q):
            var new_max = max(running_max[r], batch_max[r])
            if running_sum[r] > 0 and new_max > running_max[r]:
                var rescale = Float32(exp_f32[1](running_max[r] - new_max))
                running_sum[r] *= rescale
                var acc = v_acc + r * HEAD_DIM
                var d = 0
                while d + WIDTH <= HEAD_DIM:
                    (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
            running_max[r] = new_max

            var w_row = w_buf + r * K_STEP
            var w_max_all = Float32(-1e30)
            for bp in range(4):
                var v_sc = v_scales + (pg + bp) * WIDTH
                var scores = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var exp_scores = exp_f32_fast[WIDTH](scores - running_max[r])
                running_sum[r] += exp_scores.reduce_add()
                var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(w_eff)
                var local_max = w_eff.reduce_max()
                if local_max > w_max_all:
                    w_max_all = local_max

            if w_max_all < Float32(1e-10):
                for b in range(K_STEP):
                    w_row[b] = UInt8(0)
                wd_arr[r] = Float32(0)
                continue

            var w_scale_val = 255.0 / w_max_all
            for bp in range(4):
                var w_eff = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                    0.0, 255.0).cast[DType.uint8]()
                (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
            wd_arr[r] = w_max_all / 255.0

        # ── Phase 3: AMX V-Agg (W[16×64] × V[64×16] per channel-group pair) ──
        tileload[0, DType.uint8](w_buf, K_STEP)

        for cg_pair in range(V_CHANNEL_GROUPS // 2):
            var cg_even = cg_pair * 2
            var cg_odd = cg_even + 1

            for bp in range(4):
                var v_pg = v_base + (pg + bp) * V_PG_BYTES
                var src_even = v_pg + cg_even * V_CG_BYTES
                var src_odd = v_pg + cg_odd * V_CG_BYTES
                var dst_off = bp * V_CG_BYTES
                comptime for chunk in range(V_CG_BYTES // K_STEP):
                    (v_even_ptr + dst_off + chunk * K_STEP).store(
                        (src_even + chunk * K_STEP).load[width=K_STEP]())
                    (v_odd_ptr + dst_off + chunk * K_STEP).store(
                        (src_odd + chunk * K_STEP).load[width=K_STEP]())

            tileload[2, DType.int8](v_even_ptr, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_odd_ptr, WIDTH * VNNI_BLK)
            tilezero[4]()
            tilezero[5]()
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](result_ptr, TILE_N * 4)
            tilestore[5, DType.int32](result_ptr + TILE_M * TILE_N, TILE_N * 4)

            for r in range(actual_q):
                var wd = wd_arr[r]
                var r0 = (result_ptr + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var r1 = (result_ptr + TILE_M * TILE_N + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var a_even = v_acc + r * HEAD_DIM + cg_even * WIDTH
                var a_odd = v_acc + r * HEAD_DIM + cg_odd * WIDTH
                a_even.store(a_even.load[width=WIDTH]() + r0)
                a_odd.store(a_odd.load[width=WIDTH]() + r1)

        pg += 4

    for r in range(actual_q):
        var inv = Float32(0)
        if running_sum[r] > Float32(1e-10):
            inv = Float32(1) / (Float32(127) * running_sum[r])
        var dst = v_out + r * HEAD_DIM
        var src = v_acc + r * HEAD_DIM
        for d in range(HEAD_DIM):
            dst[d] = src[d] * inv

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# =============================================================================
# v2: Prefetch next 4-PG K/V batch during Phase 2/3
# =============================================================================


@always_inline
def prefetch_t0(p: U8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(3), Int32(1))


def amx_prefill_attn_v2_prefetch(
    q_base: I8Ptr,
    qi_biases: F32Ptr,
    q_factors: F32Ptr,
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    q_start: Int,
    q_count: Int,
    context_len: Int,
    v_out: F32Ptr,
):
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var actual_q = min(q_count, Q_TILE)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var running_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, Q_TILE](fill=Float32(0))
    var v_acc_arr = AlignedInlineArray[Float32, Q_TILE * HEAD_DIM](fill=Float32(0))
    var v_acc = v_acc_arr.unsafe_ptr()

    var q_arr = AlignedInlineArray[Scalar[DType.int8], Q_TILE * HEAD_DIM](
        fill=Scalar[DType.int8](0))
    var q_ptr = q_arr.unsafe_ptr()
    for r in range(actual_q):
        var src = q_base + r * HEAD_DIM
        var dst = q_ptr + r * HEAD_DIM
        comptime for chunk in range(HEAD_DIM // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    comptime SCORE_STRIDE_PREFILL = Q_TILE * 4 * WIDTH
    var score_arr = AlignedInlineArray[Int32, SCORE_STRIDE_PREFILL](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    comptime W_TILE_BYTES = Q_TILE * K_STEP
    var w_arr = AlignedInlineArray[UInt8, W_TILE_BYTES](fill=UInt8(0))
    var w_buf = w_arr.unsafe_ptr()

    var wd_arr = InlineArray[Float32, Q_TILE](fill=Float32(0))

    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    comptime V_GATHER_BYTES = 4 * V_CG_BYTES
    var v_tile_even = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_tile_odd = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_even_ptr = v_tile_even.unsafe_ptr()
    var v_odd_ptr = v_tile_odd.unsafe_ptr()

    var result_arr = AlignedInlineArray[Int32, 2 * TILE_M * TILE_N](
        uninitialized=True)
    var result_ptr = result_arr.unsafe_ptr()

    var pg = 0
    while pg < padded_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        var k0 = k_base + pg * K_PG_BYTES
        var k1 = k_base + (pg + 1) * K_PG_BYTES
        var k2 = k_base + (pg + 2) * K_PG_BYTES
        var k3 = k_base + (pg + 3) * K_PG_BYTES

        tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_ptr, 4 * WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + WIDTH, 4 * WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * WIDTH, 4 * WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * WIDTH, 4 * WIDTH * 4)

        # Prefetch next batch K + V into L1
        var nxt = pg + 4
        if nxt < padded_pgs:
            var k_nxt = k_base + nxt * K_PG_BYTES
            var v_nxt = v_base + nxt * V_PG_BYTES
            comptime for ln in range(0, 4 * K_PG_BYTES, 64):
                prefetch_t0((k_nxt + ln).bitcast[UInt8]())
            comptime for ln in range(0, 4 * V_PG_BYTES, 64):
                prefetch_t0((v_nxt + ln).bitcast[UInt8]())

        var batch_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
        for r in range(actual_q):
            var q_pos = q_start + r
            var row_base = score_ptr + r * 4 * WIDTH
            for bp in range(4):
                var group_start = (pg + bp) * WIDTH
                var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                    SIMD[DType.int32, WIDTH](context_len))
                var causal_valid = (valid_lanes + Int32(group_start)).le(
                    SIMD[DType.int32, WIDTH](q_pos))
                var valid = ctx_valid & causal_valid

                var raw = (row_base + bp * WIDTH).load[width=WIDTH]().cast[
                    DType.float32]()
                var k_sc = k_scales + (pg + bp) * WIDTH
                var scores = (raw - qi_biases[r]) * q_factors[r] * k_sc.load[
                    width=WIDTH]()
                scores = valid.select(scores, neg_inf)
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                var pg_max = scores.reduce_max()
                if pg_max > batch_max[r]:
                    batch_max[r] = pg_max

        for r in range(actual_q):
            var new_max = max(running_max[r], batch_max[r])
            if running_sum[r] > 0 and new_max > running_max[r]:
                var rescale = Float32(exp_f32[1](running_max[r] - new_max))
                running_sum[r] *= rescale
                var acc = v_acc + r * HEAD_DIM
                var d = 0
                while d + WIDTH <= HEAD_DIM:
                    (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
            running_max[r] = new_max

            var w_row = w_buf + r * K_STEP
            var w_max_all = Float32(-1e30)
            for bp in range(4):
                var v_sc = v_scales + (pg + bp) * WIDTH
                var scores = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var exp_scores = exp_f32_fast[WIDTH](scores - running_max[r])
                running_sum[r] += exp_scores.reduce_add()
                var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(w_eff)
                var local_max = w_eff.reduce_max()
                if local_max > w_max_all:
                    w_max_all = local_max

            if w_max_all < Float32(1e-10):
                for b in range(K_STEP):
                    w_row[b] = UInt8(0)
                wd_arr[r] = Float32(0)
                continue

            var w_scale_val = 255.0 / w_max_all
            for bp in range(4):
                var w_eff = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                    0.0, 255.0).cast[DType.uint8]()
                (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
            wd_arr[r] = w_max_all / 255.0

        tileload[0, DType.uint8](w_buf, K_STEP)

        for cg_pair in range(V_CHANNEL_GROUPS // 2):
            var cg_even = cg_pair * 2
            var cg_odd = cg_even + 1

            for bp in range(4):
                var v_pg = v_base + (pg + bp) * V_PG_BYTES
                var src_even = v_pg + cg_even * V_CG_BYTES
                var src_odd = v_pg + cg_odd * V_CG_BYTES
                var dst_off = bp * V_CG_BYTES
                comptime for chunk in range(V_CG_BYTES // K_STEP):
                    (v_even_ptr + dst_off + chunk * K_STEP).store(
                        (src_even + chunk * K_STEP).load[width=K_STEP]())
                    (v_odd_ptr + dst_off + chunk * K_STEP).store(
                        (src_odd + chunk * K_STEP).load[width=K_STEP]())

            tileload[2, DType.int8](v_even_ptr, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_odd_ptr, WIDTH * VNNI_BLK)
            tilezero[4]()
            tilezero[5]()
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](result_ptr, TILE_N * 4)
            tilestore[5, DType.int32](result_ptr + TILE_M * TILE_N, TILE_N * 4)

            for r in range(actual_q):
                var wd = wd_arr[r]
                var r0 = (result_ptr + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var r1 = (result_ptr + TILE_M * TILE_N + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var a_even = v_acc + r * HEAD_DIM + cg_even * WIDTH
                var a_odd = v_acc + r * HEAD_DIM + cg_odd * WIDTH
                a_even.store(a_even.load[width=WIDTH]() + r0)
                a_odd.store(a_odd.load[width=WIDTH]() + r1)

        pg += 4

    for r in range(actual_q):
        var inv = Float32(0)
        if running_sum[r] > Float32(1e-10):
            inv = Float32(1) / (Float32(127) * running_sum[r])
        var dst = v_out + r * HEAD_DIM
        var src = v_acc + r * HEAD_DIM
        for d in range(HEAD_DIM):
            dst[d] = src[d] * inv

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# =============================================================================
# v3: Non-interleaved score layout + causal PG skip
# Hypothesis: interleaved tilestore strides hurt Phase 2 access. Also skip
# PG batches entirely when the earliest K position > latest Q position.
# =============================================================================


def amx_prefill_attn_v3_skip(
    q_base: I8Ptr,
    qi_biases: F32Ptr,
    q_factors: F32Ptr,
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    q_start: Int,
    q_count: Int,
    context_len: Int,
    v_out: F32Ptr,
):
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var actual_q = min(q_count, Q_TILE)

    var last_q_pos = q_start + actual_q - 1
    var max_relevant_pg = last_q_pos // WIDTH
    var causal_padded = ((max_relevant_pg + 1) + 3) & ~3
    var effective_pgs = min(padded_pgs, causal_padded)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var running_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, Q_TILE](fill=Float32(0))
    var v_acc_arr = AlignedInlineArray[Float32, Q_TILE * HEAD_DIM](fill=Float32(0))
    var v_acc = v_acc_arr.unsafe_ptr()

    var q_arr = AlignedInlineArray[Scalar[DType.int8], Q_TILE * HEAD_DIM](
        fill=Scalar[DType.int8](0))
    var q_ptr = q_arr.unsafe_ptr()
    for r in range(actual_q):
        var src = q_base + r * HEAD_DIM
        var dst = q_ptr + r * HEAD_DIM
        comptime for chunk in range(HEAD_DIM // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    comptime SCORE_PG_STRIDE = Q_TILE * WIDTH
    var score_arr = AlignedInlineArray[Int32, 4 * SCORE_PG_STRIDE](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    comptime W_TILE_BYTES = Q_TILE * K_STEP
    var w_arr = AlignedInlineArray[UInt8, W_TILE_BYTES](fill=UInt8(0))
    var w_buf = w_arr.unsafe_ptr()

    var wd_arr = InlineArray[Float32, Q_TILE](fill=Float32(0))

    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    comptime V_GATHER_BYTES = 4 * V_CG_BYTES
    var v_tile_even = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_tile_odd = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_even_ptr = v_tile_even.unsafe_ptr()
    var v_odd_ptr = v_tile_odd.unsafe_ptr()

    var result_arr = AlignedInlineArray[Int32, 2 * TILE_M * TILE_N](
        uninitialized=True)
    var result_ptr = result_arr.unsafe_ptr()

    var pg = 0
    while pg < effective_pgs:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        var k0 = k_base + pg * K_PG_BYTES
        var k1 = k_base + (pg + 1) * K_PG_BYTES
        var k2 = k_base + (pg + 2) * K_PG_BYTES
        var k3 = k_base + (pg + 3) * K_PG_BYTES

        tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        # Non-interleaved: each tile stored contiguously
        tilestore[4, DType.int32](score_ptr, WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + SCORE_PG_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * SCORE_PG_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * SCORE_PG_STRIDE, WIDTH * 4)

        # Prefetch next batch
        var nxt = pg + 4
        if nxt < effective_pgs:
            var k_nxt = k_base + nxt * K_PG_BYTES
            var v_nxt = v_base + nxt * V_PG_BYTES
            comptime for ln in range(0, 4 * K_PG_BYTES, 64):
                prefetch_t0((k_nxt + ln).bitcast[UInt8]())
            comptime for ln in range(0, 4 * V_PG_BYTES, 64):
                prefetch_t0((v_nxt + ln).bitcast[UInt8]())

        var batch_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
        for r in range(actual_q):
            var q_pos = q_start + r
            for bp in range(4):
                var group_start = (pg + bp) * WIDTH
                var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                    SIMD[DType.int32, WIDTH](context_len))
                var causal_valid = (valid_lanes + Int32(group_start)).le(
                    SIMD[DType.int32, WIDTH](q_pos))
                var valid = ctx_valid & causal_valid

                var raw = (score_ptr + bp * SCORE_PG_STRIDE + r * WIDTH).load[
                    width=WIDTH]().cast[DType.float32]()
                var k_sc = k_scales + (pg + bp) * WIDTH
                var scores = (raw - qi_biases[r]) * q_factors[r] * k_sc.load[
                    width=WIDTH]()
                scores = valid.select(scores, neg_inf)
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                var pg_max = scores.reduce_max()
                if pg_max > batch_max[r]:
                    batch_max[r] = pg_max

        for r in range(actual_q):
            var new_max = max(running_max[r], batch_max[r])
            if running_sum[r] > 0 and new_max > running_max[r]:
                var rescale = Float32(exp_f32[1](running_max[r] - new_max))
                running_sum[r] *= rescale
                var acc = v_acc + r * HEAD_DIM
                var d = 0
                while d + WIDTH <= HEAD_DIM:
                    (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
            running_max[r] = new_max

            var w_row = w_buf + r * K_STEP
            var w_max_all = Float32(-1e30)
            for bp in range(4):
                var v_sc = v_scales + (pg + bp) * WIDTH
                var scores = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var exp_scores = exp_f32_fast[WIDTH](scores - running_max[r])
                running_sum[r] += exp_scores.reduce_add()
                var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(w_eff)
                var local_max = w_eff.reduce_max()
                if local_max > w_max_all:
                    w_max_all = local_max

            if w_max_all < Float32(1e-10):
                for b in range(K_STEP):
                    w_row[b] = UInt8(0)
                wd_arr[r] = Float32(0)
                continue

            var w_scale_val = 255.0 / w_max_all
            for bp in range(4):
                var w_eff = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                    0.0, 255.0).cast[DType.uint8]()
                (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
            wd_arr[r] = w_max_all / 255.0

        tileload[0, DType.uint8](w_buf, K_STEP)

        for cg_pair in range(V_CHANNEL_GROUPS // 2):
            var cg_even = cg_pair * 2
            var cg_odd = cg_even + 1

            for bp in range(4):
                var v_pg = v_base + (pg + bp) * V_PG_BYTES
                var src_even = v_pg + cg_even * V_CG_BYTES
                var src_odd = v_pg + cg_odd * V_CG_BYTES
                var dst_off = bp * V_CG_BYTES
                comptime for chunk in range(V_CG_BYTES // K_STEP):
                    (v_even_ptr + dst_off + chunk * K_STEP).store(
                        (src_even + chunk * K_STEP).load[width=K_STEP]())
                    (v_odd_ptr + dst_off + chunk * K_STEP).store(
                        (src_odd + chunk * K_STEP).load[width=K_STEP]())

            tileload[2, DType.int8](v_even_ptr, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_odd_ptr, WIDTH * VNNI_BLK)
            tilezero[4]()
            tilezero[5]()
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](result_ptr, TILE_N * 4)
            tilestore[5, DType.int32](result_ptr + TILE_M * TILE_N, TILE_N * 4)

            for r in range(actual_q):
                var wd = wd_arr[r]
                var r0 = (result_ptr + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var r1 = (result_ptr + TILE_M * TILE_N + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var a_even = v_acc + r * HEAD_DIM + cg_even * WIDTH
                var a_odd = v_acc + r * HEAD_DIM + cg_odd * WIDTH
                a_even.store(a_even.load[width=WIDTH]() + r0)
                a_odd.store(a_odd.load[width=WIDTH]() + r1)

        pg += 4

    for r in range(actual_q):
        var inv = Float32(0)
        if running_sum[r] > Float32(1e-10):
            inv = Float32(1) / (Float32(127) * running_sum[r])
        var dst = v_out + r * HEAD_DIM
        var src = v_acc + r * HEAD_DIM
        for d in range(HEAD_DIM):
            dst[d] = src[d] * inv

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# =============================================================================
# v4: Head-blocked — PG-outer, head-inner. K/V loaded once per PG batch,
# shared across all heads in the block. V tiles loaded once per CG pair,
# reused across heads.
#
# State: running_max/sum/v_acc for HEADS_PER_BLOCK heads × Q_TILE positions.
# W buffer: HEADS_PER_BLOCK × Q_TILE × K_STEP.
# =============================================================================


def amx_prefill_attn_v4_headblock[heads_per_group: Int](
    q_bases: InlineArray[I8Ptr, heads_per_group],
    qi_biases_all: InlineArray[F32Ptr, heads_per_group],
    q_factors_all: InlineArray[F32Ptr, heads_per_group],
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    q_start: Int,
    q_count: Int,
    context_len: Int,
    v_outs: InlineArray[F32Ptr, heads_per_group],
):
    comptime HPG = heads_per_group
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var actual_q = min(q_count, Q_TILE)

    var last_q_pos = q_start + actual_q - 1
    var max_relevant_pg = last_q_pos // WIDTH
    var causal_padded = ((max_relevant_pg + 1) + 3) & ~3
    var effective_pgs = min(padded_pgs, causal_padded)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var running_max = InlineArray[InlineArray[Float32, Q_TILE], HPG](
        fill=InlineArray[Float32, Q_TILE](fill=Float32(-1e30)))
    var running_sum = InlineArray[InlineArray[Float32, Q_TILE], HPG](
        fill=InlineArray[Float32, Q_TILE](fill=Float32(0)))

    comptime V_ACC_TOTAL = HPG * Q_TILE * HEAD_DIM
    var v_acc_flat = alloc_zeroed[DType.float32](V_ACC_TOTAL)
    var v_acc_ptrs = InlineArray[F32Ptr, HPG](fill=F32Ptr())
    for qh in range(HPG):
        v_acc_ptrs[qh] = v_acc_flat + qh * Q_TILE * HEAD_DIM

    comptime Q_BUF_TOTAL = HPG * Q_TILE * HEAD_DIM
    var q_flat = alloc_zeroed[DType.int8](Q_BUF_TOTAL)
    var q_ptrs = InlineArray[I8Ptr, HPG](fill=I8Ptr())
    for qh in range(HPG):
        q_ptrs[qh] = q_flat + qh * Q_TILE * HEAD_DIM
        for r in range(actual_q):
            var src = q_bases[qh] + r * HEAD_DIM
            var dst = q_ptrs[qh] + r * HEAD_DIM
            comptime for chunk in range(HEAD_DIM // K_STEP):
                (dst + chunk * K_STEP).store(
                    (src + chunk * K_STEP).load[width=K_STEP]())

    var score_arr = AlignedInlineArray[Int32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    comptime W_TILE_BYTES = Q_TILE * K_STEP
    comptime W_BUF_TOTAL = HPG * W_TILE_BYTES
    var w_flat = alloc_zeroed[DType.uint8](W_BUF_TOTAL)
    var w_ptrs = InlineArray[U8Ptr, HPG](fill=U8Ptr())
    for qh in range(HPG):
        w_ptrs[qh] = w_flat.bitcast[UInt8]() + qh * W_TILE_BYTES

    var wd_arrs = InlineArray[InlineArray[Float32, Q_TILE], HPG](
        fill=InlineArray[Float32, Q_TILE](fill=Float32(0)))

    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    comptime V_GATHER_BYTES = 4 * V_CG_BYTES
    var v_tile_even = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_tile_odd = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_even_ptr = v_tile_even.unsafe_ptr()
    var v_odd_ptr = v_tile_odd.unsafe_ptr()

    var result_arr = AlignedInlineArray[Int32, 2 * TILE_M * TILE_N](
        uninitialized=True)
    var result_ptr = result_arr.unsafe_ptr()

    var pg = 0
    while pg < effective_pgs:

        # ── Phase 1+2: Score + softmax + W_pack for each head ──
        # K data loaded from L1 for subsequent heads (hot from first head).
        for qh in range(HPG):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()
            tileload[0, DType.int8](q_ptrs[qh], HEAD_DIM)
            tileload[1, DType.int8](q_ptrs[qh] + K_STEP, HEAD_DIM)

            var k0 = k_base + pg * K_PG_BYTES
            var k1 = k_base + (pg + 1) * K_PG_BYTES
            var k2 = k_base + (pg + 2) * K_PG_BYTES
            var k3 = k_base + (pg + 3) * K_PG_BYTES

            tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
            tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
            tdpbsud[4, 0, 2]()
            tdpbsud[5, 0, 3]()
            tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[4, 1, 2]()
            tdpbsud[5, 1, 3]()

            tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
            tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
            tdpbsud[6, 0, 2]()
            tdpbsud[7, 0, 3]()
            tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
            tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
            tdpbsud[6, 1, 2]()
            tdpbsud[7, 1, 3]()

            tilestore[4, DType.int32](score_ptr, 4 * WIDTH * 4)
            tilestore[5, DType.int32](score_ptr + WIDTH, 4 * WIDTH * 4)
            tilestore[6, DType.int32](score_ptr + 2 * WIDTH, 4 * WIDTH * 4)
            tilestore[7, DType.int32](score_ptr + 3 * WIDTH, 4 * WIDTH * 4)

            var qi_biases = qi_biases_all[qh]
            var q_factors = q_factors_all[qh]
            var w_buf = w_ptrs[qh]
            var v_acc = v_acc_ptrs[qh]

            var batch_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
            for r in range(actual_q):
                var q_pos = q_start + r
                var row_base = score_ptr + r * 4 * WIDTH
                for bp in range(4):
                    var group_start = (pg + bp) * WIDTH
                    var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                        SIMD[DType.int32, WIDTH](context_len))
                    var causal_valid = (valid_lanes + Int32(group_start)).le(
                        SIMD[DType.int32, WIDTH](q_pos))
                    var valid = ctx_valid & causal_valid
                    var raw = (row_base + bp * WIDTH).load[width=WIDTH]().cast[
                        DType.float32]()
                    var k_sc = k_scales + (pg + bp) * WIDTH
                    var scores = (raw - qi_biases[r]) * q_factors[r] * k_sc.load[
                        width=WIDTH]()
                    scores = valid.select(scores, neg_inf)
                    (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                    var pg_max = scores.reduce_max()
                    if pg_max > batch_max[r]:
                        batch_max[r] = pg_max

            for r in range(actual_q):
                var new_max = max(running_max[qh][r], batch_max[r])
                if running_sum[qh][r] > 0 and new_max > running_max[qh][r]:
                    var rescale = Float32(exp_f32[1](running_max[qh][r] - new_max))
                    running_sum[qh][r] *= rescale
                    var acc = v_acc + r * HEAD_DIM
                    var d = 0
                    while d + WIDTH <= HEAD_DIM:
                        (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                        d += WIDTH
                running_max[qh][r] = new_max

                var w_row = w_buf + r * K_STEP
                var w_max_all = Float32(-1e30)
                for bp in range(4):
                    var v_sc = v_scales + (pg + bp) * WIDTH
                    var scores = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                        width=WIDTH]()
                    var exp_scores = exp_f32_fast[WIDTH](scores - running_max[qh][r])
                    running_sum[qh][r] += exp_scores.reduce_add()
                    var w_eff = exp_scores * v_sc.load[width=WIDTH]()
                    (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(w_eff)
                    var local_max = w_eff.reduce_max()
                    if local_max > w_max_all:
                        w_max_all = local_max

                if w_max_all < Float32(1e-10):
                    for b in range(K_STEP):
                        w_row[b] = UInt8(0)
                    wd_arrs[qh][r] = Float32(0)
                    continue

                var w_scale_val = 255.0 / w_max_all
                for bp in range(4):
                    var w_eff = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                        width=WIDTH]()
                    var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                        0.0, 255.0).cast[DType.uint8]()
                    (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
                wd_arrs[qh][r] = w_max_all / 255.0

        # ── Phase 3: V-agg — V loaded ONCE per CG pair, shared across heads ──
        for cg_pair in range(V_CHANNEL_GROUPS // 2):
            var cg_even = cg_pair * 2
            var cg_odd = cg_even + 1

            for bp in range(4):
                var v_pg = v_base + (pg + bp) * V_PG_BYTES
                var src_even = v_pg + cg_even * V_CG_BYTES
                var src_odd = v_pg + cg_odd * V_CG_BYTES
                var dst_off = bp * V_CG_BYTES
                comptime for chunk in range(V_CG_BYTES // K_STEP):
                    (v_even_ptr + dst_off + chunk * K_STEP).store(
                        (src_even + chunk * K_STEP).load[width=K_STEP]())
                    (v_odd_ptr + dst_off + chunk * K_STEP).store(
                        (src_odd + chunk * K_STEP).load[width=K_STEP]())

            tileload[2, DType.int8](v_even_ptr, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_odd_ptr, WIDTH * VNNI_BLK)

            for qh in range(HPG):
                tileload[0, DType.uint8](w_ptrs[qh], K_STEP)
                tilezero[4]()
                tilezero[5]()
                tdpbusd[4, 0, 2]()
                tdpbusd[5, 0, 3]()
                tilestore[4, DType.int32](result_ptr, TILE_N * 4)
                tilestore[5, DType.int32](
                    result_ptr + TILE_M * TILE_N, TILE_N * 4)

                var v_acc = v_acc_ptrs[qh]
                for r in range(actual_q):
                    var wd = wd_arrs[qh][r]
                    var r0 = (result_ptr + r * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]() * wd
                    var r1 = (result_ptr + TILE_M * TILE_N + r * TILE_N).load[
                        width=WIDTH]().cast[DType.float32]() * wd
                    var a_even = v_acc + r * HEAD_DIM + cg_even * WIDTH
                    var a_odd = v_acc + r * HEAD_DIM + cg_odd * WIDTH
                    a_even.store(a_even.load[width=WIDTH]() + r0)
                    a_odd.store(a_odd.load[width=WIDTH]() + r1)

        pg += 4

    for qh in range(HPG):
        var v_acc = v_acc_ptrs[qh]
        for r in range(actual_q):
            var inv = Float32(0)
            if running_sum[qh][r] > Float32(1e-10):
                inv = Float32(1) / (Float32(127) * running_sum[qh][r])
            var dst = v_outs[qh] + r * HEAD_DIM
            var src = v_acc + r * HEAD_DIM
            for d in range(HEAD_DIM):
                dst[d] = src[d] * inv

    v_acc_flat.free()
    q_flat.free()
    w_flat.free()
    _ = score_arr
    _ = sf_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# =============================================================================
# v5: Deferred horizontal reductions — accumulate elementwise across PGs,
# reduce once per row. Cuts 192 reduce_max/reduce_add → 48.
# Also restructures Phase 2a to PG-outer for k_scale locality.
# =============================================================================


def amx_prefill_attn_v5_deferred(
    q_base: I8Ptr,
    qi_biases: F32Ptr,
    q_factors: F32Ptr,
    k_base: U8Ptr,
    v_base: I8Ptr,
    k_scales: F32Ptr,
    v_scales: F32Ptr,
    q_start: Int,
    q_count: Int,
    context_len: Int,
    v_out: F32Ptr,
):
    var num_pgs = (context_len + WIDTH - 1) // WIDTH
    var padded_pgs = (num_pgs + 3) & ~3
    var actual_q = min(q_count, Q_TILE)

    var last_q_pos = q_start + actual_q - 1
    var max_relevant_pg = last_q_pos // WIDTH
    var causal_padded = ((max_relevant_pg + 1) + 3) & ~3
    var effective_pgs = min(padded_pgs, causal_padded)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
    var zero_vec = SIMD[DType.float32, WIDTH](0)
    var valid_lanes = SIMD[DType.int32, WIDTH]()
    comptime for lane in range(WIDTH):
        valid_lanes[lane] = Int32(lane)

    var running_max = InlineArray[Float32, Q_TILE](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, Q_TILE](fill=Float32(0))
    var v_acc_arr = AlignedInlineArray[Float32, Q_TILE * HEAD_DIM](fill=Float32(0))
    var v_acc = v_acc_arr.unsafe_ptr()

    var q_arr = AlignedInlineArray[Scalar[DType.int8], Q_TILE * HEAD_DIM](
        fill=Scalar[DType.int8](0))
    var q_ptr = q_arr.unsafe_ptr()
    for r in range(actual_q):
        var src = q_base + r * HEAD_DIM
        var dst = q_ptr + r * HEAD_DIM
        comptime for chunk in range(HEAD_DIM // K_STEP):
            (dst + chunk * K_STEP).store(
                (src + chunk * K_STEP).load[width=K_STEP]())

    comptime SCORE_STRIDE_PREFILL = Q_TILE * 4 * WIDTH
    var score_arr = AlignedInlineArray[Int32, SCORE_STRIDE_PREFILL](
        uninitialized=True)
    var score_ptr = score_arr.unsafe_ptr()

    comptime W_TILE_BYTES = Q_TILE * K_STEP
    var w_arr = AlignedInlineArray[UInt8, W_TILE_BYTES](fill=UInt8(0))
    var w_buf = w_arr.unsafe_ptr()

    var wd_arr = InlineArray[Float32, Q_TILE](fill=Float32(0))

    var sf_arr = AlignedInlineArray[Float32, Q_TILE * 4 * WIDTH](
        uninitialized=True)
    var sf_buf = sf_arr.unsafe_ptr()

    comptime V_GATHER_BYTES = 4 * V_CG_BYTES
    var v_tile_even = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_tile_odd = AlignedInlineArray[Scalar[DType.int8], V_GATHER_BYTES](
        uninitialized=True)
    var v_even_ptr = v_tile_even.unsafe_ptr()
    var v_odd_ptr = v_tile_odd.unsafe_ptr()

    var result_arr = AlignedInlineArray[Int32, 2 * TILE_M * TILE_N](
        uninitialized=True)
    var result_ptr = result_arr.unsafe_ptr()

    var pg = 0
    while pg < effective_pgs:
        # ── Phase 1: AMX Score 4 PGs (2:2:4) ──
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        tileload[0, DType.int8](q_ptr, HEAD_DIM)
        tileload[1, DType.int8](q_ptr + K_STEP, HEAD_DIM)

        var k0 = k_base + pg * K_PG_BYTES
        var k1 = k_base + (pg + 1) * K_PG_BYTES
        var k2 = k_base + (pg + 2) * K_PG_BYTES
        var k3 = k_base + (pg + 3) * K_PG_BYTES

        tileload[2, DType.uint8](k0, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1, WIDTH * VNNI_BLK)
        tdpbsud[4, 0, 2]()
        tdpbsud[5, 0, 3]()
        tileload[2, DType.uint8](k0 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k1 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[4, 1, 2]()
        tdpbsud[5, 1, 3]()

        tileload[2, DType.uint8](k2, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3, WIDTH * VNNI_BLK)
        tdpbsud[6, 0, 2]()
        tdpbsud[7, 0, 3]()
        tileload[2, DType.uint8](k2 + TILE_BYTES, WIDTH * VNNI_BLK)
        tileload[3, DType.uint8](k3 + TILE_BYTES, WIDTH * VNNI_BLK)
        tdpbsud[6, 1, 2]()
        tdpbsud[7, 1, 3]()

        tilestore[4, DType.int32](score_ptr, 4 * WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + WIDTH, 4 * WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * WIDTH, 4 * WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * WIDTH, 4 * WIDTH * 4)

        # Prefetch next batch
        var nxt = pg + 4
        if nxt < effective_pgs:
            var k_nxt = k_base + nxt * K_PG_BYTES
            var v_nxt = v_base + nxt * V_PG_BYTES
            comptime for ln in range(0, 4 * K_PG_BYTES, 64):
                prefetch_t0((k_nxt + ln).bitcast[UInt8]())
            comptime for ln in range(0, 4 * V_PG_BYTES, 64):
                prefetch_t0((v_nxt + ln).bitcast[UInt8]())

        # ── Phase 2a: PG-outer dequant, deferred batch_max ──
        # Accumulate elementwise max across PGs, reduce once per row.
        var max_vecs = InlineArray[SIMD[DType.float32, WIDTH], Q_TILE](
            fill=neg_inf)
        for bp in range(4):
            var group_start = (pg + bp) * WIDTH
            var ctx_valid = (valid_lanes + Int32(group_start)).lt(
                SIMD[DType.int32, WIDTH](context_len))
            var k_sc = (k_scales + (pg + bp) * WIDTH).load[width=WIDTH]()
            for r in range(actual_q):
                var q_pos = q_start + r
                var causal_valid = (valid_lanes + Int32(group_start)).le(
                    SIMD[DType.int32, WIDTH](q_pos))
                var valid = ctx_valid & causal_valid
                var raw = (score_ptr + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]().cast[DType.float32]()
                var scores = (raw - qi_biases[r]) * q_factors[r] * k_sc
                scores = valid.select(scores, neg_inf)
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(scores)
                max_vecs[r] = max(max_vecs[r], scores)

        # ── Phase 2b: Rescale + exp + W pack with deferred reductions ──
        for r in range(actual_q):
            var batch_max_r = max_vecs[r].reduce_max()
            var new_max = max(running_max[r], batch_max_r)
            if running_sum[r] > 0 and new_max > running_max[r]:
                var rescale = Float32(exp_f32[1](running_max[r] - new_max))
                running_sum[r] *= rescale
                var acc = v_acc + r * HEAD_DIM
                var d = 0
                while d + WIDTH <= HEAD_DIM:
                    (acc + d).store((acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
            running_max[r] = new_max

            var w_row = w_buf + r * K_STEP
            var exp_sum_vec = SIMD[DType.float32, WIDTH](0)
            var w_max_vec = SIMD[DType.float32, WIDTH](0)

            for bp in range(4):
                var v_sc = (v_scales + (pg + bp) * WIDTH).load[width=WIDTH]()
                var scores = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var exp_scores = exp_f32_fast[WIDTH](scores - running_max[r])
                exp_sum_vec += exp_scores
                var w_eff = exp_scores * v_sc
                (sf_buf + r * 4 * WIDTH + bp * WIDTH).store(w_eff)
                w_max_vec = max(w_max_vec, w_eff)

            running_sum[r] += exp_sum_vec.reduce_add()
            var w_max_all = w_max_vec.reduce_max()

            if w_max_all < Float32(1e-10):
                for b in range(K_STEP):
                    w_row[b] = UInt8(0)
                wd_arr[r] = Float32(0)
                continue

            var w_scale_val = 255.0 / w_max_all
            for bp in range(4):
                var w_eff = (sf_buf + r * 4 * WIDTH + bp * WIDTH).load[
                    width=WIDTH]()
                var w_u8 = roundeven(w_eff * w_scale_val).clamp(
                    0.0, 255.0).cast[DType.uint8]()
                (w_row + bp * WIDTH).store[width=WIDTH](w_u8)
            wd_arr[r] = w_max_all / 255.0

        # ── Phase 3: AMX V-Agg ──
        tileload[0, DType.uint8](w_buf, K_STEP)

        for cg_pair in range(V_CHANNEL_GROUPS // 2):
            var cg_even = cg_pair * 2
            var cg_odd = cg_even + 1

            for bp in range(4):
                var v_pg = v_base + (pg + bp) * V_PG_BYTES
                var src_even = v_pg + cg_even * V_CG_BYTES
                var src_odd = v_pg + cg_odd * V_CG_BYTES
                var dst_off = bp * V_CG_BYTES
                comptime for chunk in range(V_CG_BYTES // K_STEP):
                    (v_even_ptr + dst_off + chunk * K_STEP).store(
                        (src_even + chunk * K_STEP).load[width=K_STEP]())
                    (v_odd_ptr + dst_off + chunk * K_STEP).store(
                        (src_odd + chunk * K_STEP).load[width=K_STEP]())

            tileload[2, DType.int8](v_even_ptr, WIDTH * VNNI_BLK)
            tileload[3, DType.int8](v_odd_ptr, WIDTH * VNNI_BLK)
            tilezero[4]()
            tilezero[5]()
            tdpbusd[4, 0, 2]()
            tdpbusd[5, 0, 3]()
            tilestore[4, DType.int32](result_ptr, TILE_N * 4)
            tilestore[5, DType.int32](result_ptr + TILE_M * TILE_N, TILE_N * 4)

            for r in range(actual_q):
                var wd = wd_arr[r]
                var r0 = (result_ptr + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var r1 = (result_ptr + TILE_M * TILE_N + r * TILE_N).load[
                    width=WIDTH]().cast[DType.float32]() * wd
                var a_even = v_acc + r * HEAD_DIM + cg_even * WIDTH
                var a_odd = v_acc + r * HEAD_DIM + cg_odd * WIDTH
                a_even.store(a_even.load[width=WIDTH]() + r0)
                a_odd.store(a_odd.load[width=WIDTH]() + r1)

        pg += 4

    for r in range(actual_q):
        var inv = Float32(0)
        if running_sum[r] > Float32(1e-10):
            inv = Float32(1) / (Float32(127) * running_sum[r])
        var dst = v_out + r * HEAD_DIM
        var src = v_acc + r * HEAD_DIM
        for d in range(HEAD_DIM):
            dst[d] = src[d] * inv

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# =============================================================================
# Harness
# =============================================================================


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    comptime MAX_PGS = 512
    comptime MAX_CTX = MAX_PGS * WIDTH

    var k = alloc_zeroed[DType.uint8](MAX_PGS * K_PG_BYTES)
    var v = alloc_zeroed[DType.int8](MAX_PGS * V_PG_BYTES)
    var k_sc = alloc_zeroed[DType.float32](MAX_CTX)
    var v_sc = alloc_zeroed[DType.float32](MAX_CTX)

    fill_random_u8(k.bitcast[UInt8](), MAX_PGS * K_PG_BYTES)
    fill_random_i8(v, MAX_PGS * V_PG_BYTES)
    fill_random_f32(k_sc, MAX_CTX, Float32(0.01))
    fill_random_f32(v_sc, MAX_CTX, Float32(0.01))

    var k_u8 = k.bitcast[UInt8]()

    comptime TEST_SEQ = 32
    comptime TEST_PGS = (TEST_SEQ + WIDTH - 1) // WIDTH

    var q_all = alloc_zeroed[DType.int8](TEST_SEQ * HEAD_DIM)
    var qi_biases = alloc_zeroed[DType.float32](TEST_SEQ)
    var q_factors = alloc_zeroed[DType.float32](TEST_SEQ)

    for pos in range(TEST_SEQ):
        fill_random_i8(q_all + pos * HEAD_DIM, HEAD_DIM)
        qi_biases[pos] = Float32(128 * 64)
        q_factors[pos] = Float32(0.00001)

    print("=== AMX Prefill Attention Ablations ===")
    print("HEAD_DIM=" + String(HEAD_DIM) + " WIDTH=" + String(WIDTH) +
        " Q_TILE=" + String(Q_TILE))
    print("")

    # --- Correctness: compare sequential single-Q vs prefill multi-Q ---
    print("--- CORRECTNESS CHECK (seq_len=" + String(TEST_SEQ) + ") ---")

    var ref_out = alloc_zeroed[DType.float32](TEST_SEQ * HEAD_DIM)
    for pos in range(TEST_SEQ):
        var ctx = pos + 1
        reference_single_q_attn(
            q_all + pos * HEAD_DIM,
            qi_biases[pos],
            q_factors[pos],
            k_u8, v, k_sc, v_sc,
            ctx,
            ref_out + pos * HEAD_DIM)

    var prefill_out = alloc_zeroed[DType.float32](TEST_SEQ * HEAD_DIM)
    for tile_start in range(0, TEST_SEQ, Q_TILE):
        var tile_count = min(Q_TILE, TEST_SEQ - tile_start)
        var ctx = tile_start + tile_count
        amx_prefill_attn(
            q_all + tile_start * HEAD_DIM,
            qi_biases + tile_start,
            q_factors + tile_start,
            k_u8, v, k_sc, v_sc,
            tile_start, tile_count, ctx,
            prefill_out + tile_start * HEAD_DIM)

    var total_ref_abs = Float32(0)
    var total_diff_abs = Float32(0)
    var global_max_diff = Float32(0)

    print("  per-token error (abs_sum_ref | abs_sum_diff | rel%):")
    for pos in range(TEST_SEQ):
        var tok_ref_abs = Float32(0)
        var tok_diff_abs = Float32(0)
        var tok_max_diff = Float32(0)
        for d in range(HEAD_DIM):
            var idx = pos * HEAD_DIM + d
            var rv = ref_out[idx].__abs__()
            var dv = (ref_out[idx] - prefill_out[idx]).__abs__()
            tok_ref_abs += rv
            tok_diff_abs += dv
            if dv > tok_max_diff:
                tok_max_diff = dv
        total_ref_abs += tok_ref_abs
        total_diff_abs += tok_diff_abs
        if tok_max_diff > global_max_diff:
            global_max_diff = tok_max_diff
        var tok_rel = Float32(0)
        if tok_ref_abs > Float32(1e-10):
            tok_rel = tok_diff_abs / tok_ref_abs * Float32(100)
        print("    pos " + String(pos) + ": " +
            String(tok_ref_abs) + " | " +
            String(tok_diff_abs) + " | " +
            String(tok_rel) + "%  max_el=" + String(tok_max_diff))

    print("  aggregate:")
    print("    total ref abs:   " + String(total_ref_abs))
    print("    total diff abs:  " + String(total_diff_abs))
    print("    max element:     " + String(global_max_diff))
    var agg_rel = Float32(0)
    if total_ref_abs > Float32(1e-10):
        agg_rel = total_diff_abs / total_ref_abs * Float32(100)
    print("    relative error:  " + String(agg_rel) + "%")
    print("")

    # --- Performance ablations ---

    var seq_sizes = InlineArray[Int, 9](fill=0)
    seq_sizes[0] = 16
    seq_sizes[1] = 64
    seq_sizes[2] = 128
    seq_sizes[3] = 256
    seq_sizes[4] = 512
    seq_sizes[5] = 1024
    seq_sizes[6] = 2048
    seq_sizes[7] = 4096
    seq_sizes[8] = 8192

    print("--- ABLATION: v3 skip+prefetch vs v5 deferred reduce ---")
    for idx in range(9):
        var seq_len = seq_sizes[idx]
        if seq_len > MAX_CTX:
            continue

        var iters = 1000
        var warmup = 200
        if seq_len >= 1024:
            iters = 100
            warmup = 20
        if seq_len >= 4096:
            iters = 20
            warmup = 5

        var q_perf = alloc_zeroed[DType.int8](seq_len * HEAD_DIM)
        var qb_perf = alloc_zeroed[DType.float32](seq_len)
        var qf_perf = alloc_zeroed[DType.float32](seq_len)
        var out_perf = alloc_zeroed[DType.float32](seq_len * HEAD_DIM)

        for pos in range(seq_len):
            fill_random_i8(q_perf + pos * HEAD_DIM, HEAD_DIM)
            qb_perf[pos] = Float32(128 * 64)
            qf_perf[pos] = Float32(0.00001)

        # v3: skip + prefetch
        for _ in range(warmup):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn_v3_skip(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var t3 = Int(perf_counter_ns())
        for _ in range(iters):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn_v3_skip(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var ns3 = (Int(perf_counter_ns()) - t3) // iters

        print("  seq_len=" + String(seq_len) + " (iters=" + String(iters) + "):")
        print("    v3 skip+prefetch:     " + String(ns3) + " ns  (" +
            String(ns3 // seq_len) + " ns/tok)")

        # v5: deferred reductions
        for _ in range(warmup):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn_v5_deferred(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var t5 = Int(perf_counter_ns())
        for _ in range(iters):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn_v5_deferred(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var ns5 = (Int(perf_counter_ns()) - t5) // iters

        print("    v5 deferred reduce:   " + String(ns5) + " ns  (" +
            String(ns5 // seq_len) + " ns/tok)")
        var v5_vs_v3 = Float32(0)
        if ns5 > 0:
            v5_vs_v3 = Float32(ns3) / Float32(ns5)
        print("    v5 vs v3 speedup:     " + String(v5_vs_v3) + "x")
        print("")

        q_perf.free()
        qb_perf.free()
        qf_perf.free()
        out_perf.free()

    ref_out.free()
    prefill_out.free()
    q_all.free()
    qi_biases.free()
    q_factors.free()
    k.free()
    v.free()
    k_sc.free()
    v_sc.free()
