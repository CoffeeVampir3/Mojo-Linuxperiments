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
    tilezero, tileload, tilestore, tdpbssd, tdpbsud, tdpbusd,
)
from experimental3.kernels.dot_prod import vpdpbusd, bcast_4u8_vnni
from experimental3.kernels.gemv import gemv_row, gemv_row_blocked_bf16_scaled
from simd_math import exp_f32, exp_f32_fast, roundeven, set_subnormal_zeroing
from notstdcollections import AlignedInlineArray


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime SIMD_W = simd_width_of[DType.float32]()

comptime Q_TILE = 16
comptime HEAD_DIM = 128
comptime WIDTH = 16
comptime K_PG_BYTES = HEAD_DIM // VNNI_BLK * WIDTH * VNNI_BLK
comptime V_CHANNEL_GROUPS = HEAD_DIM // WIDTH
comptime V_CG_BYTES = (WIDTH // VNNI_BLK) * WIDTH * VNNI_BLK
comptime V_PG_BYTES = V_CHANNEL_GROUPS * V_CG_BYTES
comptime SQ_COUNT = WIDTH // VNNI_BLK
comptime SQ_BYTES = WIDTH * VNNI_BLK

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# Kernel 1: Per-row dequant AMX GEMM
# Covers: QKV projection, expert W1, expert W3
# ============================================================================


def amx_gemm[N: Int, K: Int, OutDType: DType = DType.float32](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
    M: Int,
):
    # tdpbssd (signed x signed) — no colsum correction needed.
    # Activation buffer must have M_STEP padding past M.
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
                    (dst + (mb + m) * N + nb + n).store(
                        (f32_v * ws).cast[OutDType]())

    _ = c_arr


# ============================================================================
# Kernel 2: Per-K-block dequant AMX GEMM
# Covers: O-projection, expert W2
# ============================================================================


def amx_gemm_blocked[N: Int, K: Int, fwht_blk: Int,
    OutDType: DType = DType.float32](
    act: I8Ptr,
    wpacked: I8Ptr,
    blk_scales: F32Ptr,
    w_scale: F32Ptr,
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
    M: Int,
    output_scale: Float32 = Float32(1.0),
):
    # Accumulates fwht_blk K elements in i32, dequants per block to f32.
    # tdpbssd eliminates colsum. blk_scales layout: [row * num_blocks + blk].
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0
    comptime assert K % fwht_blk == 0
    comptime assert fwht_blk % K_STEP == 0
    comptime num_blocks = K // fwht_blk
    comptime k_steps_per_block = fwht_blk // K_STEP

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()
    var f32_arr = AlignedInlineArray[Float32, M_STEP * N_STEP](fill=Float32(0))
    var f32_buf = f32_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            for i in range(M_STEP * N_STEP):
                f32_buf[i] = Float32(0)

            for blk in range(num_blocks):
                tilezero[4]()
                tilezero[5]()
                tilezero[6]()
                tilezero[7]()

                var k_base = blk * fwht_blk
                for ks in range(k_steps_per_block):
                    var k = k_base + ks * K_STEP
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
                    var dq = blk_scales[(mb + m) * num_blocks + blk] / Float32(127)
                    var row_off = m * N_STEP
                    for n in range(0, N_STEP, SIMD_W):
                        var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                        (f32_buf + row_off + n).store(
                            (f32_buf + row_off + n).load[width=SIMD_W]()
                            + i32_v.cast[DType.float32]() * dq)

            for m in range(m_count):
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var acc = (f32_buf + row_off + n).load[width=SIMD_W]()
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(
                        (acc * ws * output_scale).cast[OutDType]())

    _ = c_arr
    _ = f32_arr


# ============================================================================
# Kernel 3: Multi-Q AMX prefill attention
# Processes up to Q_TILE=16 query positions per call, one head at a time.
# AMX scoring (tdpbsud, 4 PGs per batch) + AMX V-agg (tdpbusd).
# Causal PG skip avoids processing fully-masked PG batches.
# ============================================================================


@always_inline
def prefetch_t0(p: U8Ptr):
    llvm_intrinsic["llvm.prefetch.p0", NoneType](
        p, Int32(0), Int32(3), Int32(1))


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

    var last_q_pos = q_start + actual_q - 1
    var max_relevant_pg = last_q_pos // WIDTH
    var causal_padded = ((max_relevant_pg + 1) + 3) & ~3
    var effective_pgs = min(padded_pgs, causal_padded)

    var neg_inf = SIMD[DType.float32, WIDTH](Float32(-1e30))
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
        # Phase 1: AMX Score 4 PGs
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

        tilestore[4, DType.int32](score_ptr, WIDTH * 4)
        tilestore[5, DType.int32](score_ptr + SCORE_PG_STRIDE, WIDTH * 4)
        tilestore[6, DType.int32](score_ptr + 2 * SCORE_PG_STRIDE, WIDTH * 4)
        tilestore[7, DType.int32](score_ptr + 3 * SCORE_PG_STRIDE, WIDTH * 4)

        var nxt = pg + 4
        if nxt < effective_pgs:
            var k_nxt = k_base + nxt * K_PG_BYTES
            var v_nxt = v_base + nxt * V_PG_BYTES
            comptime for ln in range(0, 4 * K_PG_BYTES, 64):
                prefetch_t0((k_nxt + ln).bitcast[UInt8]())
            comptime for ln in range(0, 4 * V_PG_BYTES, 64):
                prefetch_t0((v_nxt + ln).bitcast[UInt8]())

        # Phase 2a: Dequant + causal/context mask + batch max
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

        # Phase 2b: Rescale running state, exp, W packing
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

        # Phase 3: AMX V-Agg
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
        var out = v_out + r * HEAD_DIM
        var src = v_acc + r * HEAD_DIM
        for d in range(HEAD_DIM):
            out[d] = src[d] * inv

    _ = q_arr
    _ = score_arr
    _ = w_arr
    _ = sf_arr
    _ = v_acc_arr
    _ = v_tile_even
    _ = v_tile_odd
    _ = result_arr


# ============================================================================
# Test utilities
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


def fill_random_u8(ptr: U8Ptr, count: Int, seed: UInt64 = 0xCAFEBABE87654321):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = UInt8((state >> 33).cast[DType.uint8]())


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


# ============================================================================
# Benchmark helpers
# ============================================================================


def bench_iters(M: Int) -> Int:
    if M >= 4096:
        return 2
    if M >= 2048:
        return 5
    if M >= 512:
        return 30
    if M >= 128:
        return 100
    return 200


def bench_samples(M: Int) -> Int:
    if M >= 2048:
        return 1
    return 2


def max_abs_error(a: F32Ptr, b: F32Ptr, count: Int) -> Float32:
    var mx = Float32(0)
    for i in range(count):
        var d = (a[i] - b[i]).__abs__()
        if d > mx:
            mx = d
    return mx


# ============================================================================
# GEMM benchmarks
# ============================================================================


def run_per_row[N: Int, K: Int](label: String):
    print("=== Per-row GEMM: " + label
        + " (N=" + String(N) + ", K=" + String(K) + ") ===")

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xCAFEBABE87654321)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var colsum = alloc_zeroed[DType.float32](N)
    compute_colsum(weight_raw, colsum, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))

    comptime MAX_M = 1024
    comptime ACT_ROWS = MAX_M + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K)
    var act_scale = alloc_zeroed[DType.float32](ACT_ROWS)
    fill_random_f32(act_scale, ACT_ROWS, Float32(1.0))

    var out_ref = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_amx = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_vnni = alloc_zeroed[DType.float32](ACT_ROWS * N)

    comptime TEST_M = 32
    reference_per_row(act, weight_raw, act_scale, w_scale, out_ref, TEST_M, N, K)
    amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_amx, TEST_M)

    for m in range(TEST_M):
        var dq = act_scale[m] / Float32(127)
        gemv_row[N, K, DType.float32](
            act + m * K, wpacked, dq, w_scale, colsum, out_vnni + m * N)

    print("  correctness: amx=" + String(max_abs_error(out_amx, out_ref, TEST_M * N))
        + " vnni=" + String(max_abs_error(out_vnni, out_ref, TEST_M * N)))

    var bench_ms = InlineArray[Int, 5](fill=0)
    bench_ms[0] = 32
    bench_ms[1] = 128
    bench_ms[2] = 256
    bench_ms[3] = 512
    bench_ms[4] = 1024

    print("       M |    AMX |   VNNI | AMX/VNNI | GOPS")
    for idx in range(5):
        var test_m = bench_ms[idx]
        var iters = bench_iters(test_m)
        var samples = bench_samples(test_m)
        var warmup = max(iters // 10, 2)

        var best_amx = Int(0)
        for sample in range(samples):
            for _ in range(warmup):
                amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_amx, test_m)
            var t = Int(perf_counter_ns())
            for _ in range(iters):
                amx_gemm[N, K](act, wpacked, act_scale, w_scale, out_amx, test_m)
            var ns = (Int(perf_counter_ns()) - t) // iters
            if sample == 0 or ns < best_amx:
                best_amx = ns

        var best_vnni = Int(0)
        for sample in range(samples):
            for _ in range(warmup):
                for m in range(test_m):
                    var dq = act_scale[m] / Float32(127)
                    gemv_row[N, K, DType.float32](
                        act + m * K, wpacked, dq, w_scale, colsum,
                        out_vnni + m * N)
            var t = Int(perf_counter_ns())
            for _ in range(iters):
                for m in range(test_m):
                    var dq = act_scale[m] / Float32(127)
                    gemv_row[N, K, DType.float32](
                        act + m * K, wpacked, dq, w_scale, colsum,
                        out_vnni + m * N)
            var ns = (Int(perf_counter_ns()) - t) // iters
            if sample == 0 or ns < best_vnni:
                best_vnni = ns

        var ratio = Float32(best_amx) / Float32(best_vnni)
        var gops = Float32(test_m * N * K) / Float32(best_amx)
        print("  " + String(test_m) + "  | " + String(best_amx // test_m)
            + " | " + String(best_vnni // test_m)
            + " | " + String(ratio) + " | " + String(gops))

    print("")
    weight_raw.free()
    wpacked.free()
    colsum.free()
    w_scale.free()
    act.free()
    act_scale.free()
    out_ref.free()
    out_amx.free()
    out_vnni.free()


def run_blocked[N: Int, K: Int, fwht_blk: Int](label: String):
    print("=== Blocked GEMM: " + label
        + " (N=" + String(N) + ", K=" + String(K)
        + ", blk=" + String(fwht_blk) + ") ===")

    comptime num_blocks = K // fwht_blk
    var output_scale = Float32(0.85)

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xB10C4ED123456789)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))
    var blk_colsum = alloc_zeroed[DType.float32](num_blocks * N)
    compute_block_colsums(weight_raw, blk_colsum, N, K, fwht_blk)

    comptime MAX_M = 1024
    comptime ACT_ROWS = MAX_M + M_STEP
    var act = alloc_zeroed[DType.int8](ACT_ROWS * K)
    fill_random_i8(act, ACT_ROWS * K, seed=0xB10CA0C712340000)
    var blk_scales = alloc_zeroed[DType.float32](ACT_ROWS * num_blocks)
    fill_random_f32(blk_scales, ACT_ROWS * num_blocks, Float32(1.0))

    var out_ref = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_amx = alloc_zeroed[DType.float32](ACT_ROWS * N)
    var out_vnni = alloc_zeroed[DType.bfloat16](ACT_ROWS * N)

    comptime TEST_M = 32
    reference_blocked(act, weight_raw, blk_scales, w_scale,
        out_ref, TEST_M, N, K, fwht_blk, output_scale)
    amx_gemm_blocked[N, K, fwht_blk](act, wpacked, blk_scales, w_scale,
        out_amx, TEST_M, output_scale)

    for m in range(TEST_M):
        gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
            act + m * K, wpacked, blk_scales + m * num_blocks,
            w_scale, blk_colsum, out_vnni + m * N, output_scale)

    var max_vnni = Float32(0)
    for i in range(TEST_M * N):
        var d = (Float32(out_vnni[i]) - out_ref[i]).__abs__()
        if d > max_vnni:
            max_vnni = d
    print("  correctness: amx=" + String(max_abs_error(out_amx, out_ref, TEST_M * N))
        + " vnni=" + String(max_vnni))

    var bench_ms = InlineArray[Int, 5](fill=0)
    bench_ms[0] = 32
    bench_ms[1] = 128
    bench_ms[2] = 256
    bench_ms[3] = 512
    bench_ms[4] = 1024

    print("       M |    AMX |   VNNI | AMX/VNNI | GOPS")
    for idx in range(5):
        var test_m = bench_ms[idx]
        var iters = bench_iters(test_m)
        var samples = bench_samples(test_m)
        var warmup = max(iters // 10, 2)

        var best_amx = Int(0)
        for sample in range(samples):
            for _ in range(warmup):
                amx_gemm_blocked[N, K, fwht_blk](act, wpacked, blk_scales,
                    w_scale, out_amx, test_m, output_scale)
            var t = Int(perf_counter_ns())
            for _ in range(iters):
                amx_gemm_blocked[N, K, fwht_blk](act, wpacked, blk_scales,
                    w_scale, out_amx, test_m, output_scale)
            var ns = (Int(perf_counter_ns()) - t) // iters
            if sample == 0 or ns < best_amx:
                best_amx = ns

        var best_vnni = Int(0)
        for sample in range(samples):
            for _ in range(warmup):
                for m in range(test_m):
                    gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
                        act + m * K, wpacked, blk_scales + m * num_blocks,
                        w_scale, blk_colsum, out_vnni + m * N, output_scale)
            var t = Int(perf_counter_ns())
            for _ in range(iters):
                for m in range(test_m):
                    gemv_row_blocked_bf16_scaled[N, K, fwht_blk](
                        act + m * K, wpacked, blk_scales + m * num_blocks,
                        w_scale, blk_colsum, out_vnni + m * N, output_scale)
            var ns = (Int(perf_counter_ns()) - t) // iters
            if sample == 0 or ns < best_vnni:
                best_vnni = ns

        var ratio = Float32(best_amx) / Float32(best_vnni)
        var gops = Float32(test_m * N * K) / Float32(best_amx)
        print("  " + String(test_m) + "  | " + String(best_amx // test_m)
            + " | " + String(best_vnni // test_m)
            + " | " + String(ratio) + " | " + String(gops))

    print("")
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
# Attention benchmark
# ============================================================================


def run_attention():
    print("=== Prefill Attention (HEAD_DIM=" + String(HEAD_DIM)
        + ", Q_TILE=" + String(Q_TILE) + ") ===")

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
    var q_all = alloc_zeroed[DType.int8](TEST_SEQ * HEAD_DIM)
    var qi_biases = alloc_zeroed[DType.float32](TEST_SEQ)
    var q_factors = alloc_zeroed[DType.float32](TEST_SEQ)

    for pos in range(TEST_SEQ):
        fill_random_i8(q_all + pos * HEAD_DIM, HEAD_DIM)
        qi_biases[pos] = Float32(128 * 64)
        q_factors[pos] = Float32(0.00001)

    var ref_out = alloc_zeroed[DType.float32](TEST_SEQ * HEAD_DIM)
    for pos in range(TEST_SEQ):
        reference_single_q_attn(
            q_all + pos * HEAD_DIM, qi_biases[pos], q_factors[pos],
            k_u8, v, k_sc, v_sc, pos + 1,
            ref_out + pos * HEAD_DIM)

    var prefill_out = alloc_zeroed[DType.float32](TEST_SEQ * HEAD_DIM)
    for tile_start in range(0, TEST_SEQ, Q_TILE):
        var tile_count = min(Q_TILE, TEST_SEQ - tile_start)
        amx_prefill_attn(
            q_all + tile_start * HEAD_DIM,
            qi_biases + tile_start, q_factors + tile_start,
            k_u8, v, k_sc, v_sc,
            tile_start, tile_count, tile_start + tile_count,
            prefill_out + tile_start * HEAD_DIM)

    var total_ref = Float32(0)
    var total_diff = Float32(0)
    var global_max = Float32(0)
    for i in range(TEST_SEQ * HEAD_DIM):
        var rv = ref_out[i].__abs__()
        var dv = (ref_out[i] - prefill_out[i]).__abs__()
        total_ref += rv
        total_diff += dv
        if dv > global_max:
            global_max = dv
    var rel = Float32(0)
    if total_ref > Float32(1e-10):
        rel = total_diff / total_ref * Float32(100)
    print("  correctness (seq=" + String(TEST_SEQ) + "): rel="
        + String(rel) + "% max_el=" + String(global_max))

    var seq_sizes = InlineArray[Int, 6](fill=0)
    seq_sizes[0] = 64
    seq_sizes[1] = 128
    seq_sizes[2] = 256
    seq_sizes[3] = 512
    seq_sizes[4] = 1024
    seq_sizes[5] = 4096

    print("     seq |   total ns | ns/tok")
    for idx in range(6):
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

        for _ in range(warmup):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var t = Int(perf_counter_ns())
        for _ in range(iters):
            for tile_start in range(0, seq_len, Q_TILE):
                var tile_count = min(Q_TILE, seq_len - tile_start)
                amx_prefill_attn(
                    q_perf + tile_start * HEAD_DIM,
                    qb_perf + tile_start, qf_perf + tile_start,
                    k_u8, v, k_sc, v_sc,
                    tile_start, tile_count, tile_start + tile_count,
                    out_perf + tile_start * HEAD_DIM)
        var ns = (Int(perf_counter_ns()) - t) // iters

        print("  " + String(seq_len) + "  | " + String(ns)
            + " | " + String(ns // seq_len))

        q_perf.free()
        qb_perf.free()
        qf_perf.free()
        out_perf.free()

    print("")
    ref_out.free()
    prefill_out.free()
    q_all.free()
    qi_biases.free()
    q_factors.free()
    k.free()
    v.free()
    k_sc.free()
    v_sc.free()


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    print("=== Prefill Kernel Proposals ===")
    print("All kernels use 2-2-4 tile config (16x64 bytes)")
    print("")

    run_per_row[1536, 3072]("MoE expert W1/W3")
    run_per_row[2048, 3072]("QKV proj (TP=4)")
    run_blocked[3072, 1536, 128]("O-proj / expert W2 (TP=4)")
    run_attention()
