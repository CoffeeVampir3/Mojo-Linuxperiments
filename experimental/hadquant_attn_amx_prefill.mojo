"""AMX prefill attention — KV-outer, query-inner with Q-row parallelism.

Workers split Q rows across cores within each KV group. Each worker
redundantly packs K and converts V (no barriers needed), but processes
only its assigned Q rows. K/V cache data read once per KV block per worker.

For each KV block:
  1. Pack K and convert V (redundant per worker, amortized by Q reuse)
  2. For each query batch (Q_BATCH=32 rows → running_o fits L1):
     - Score GEMM (tdpbsud): Q_i8 × K_vnni → i32 scores
     - Softmax: dequant + online update
     - V agg GEMM (tdpbf16ps): W_bf16 × V_vnni → f32 output

running_o per batch = Q_BATCH × head_dim × 4 = 16KB → L1.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.time import perf_counter_ns
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32_fast, quantize_i8, quantize_i8_scalar
from simd_math.matrixops import transpose_generic
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.hadquant_attn import HadAttnCtx
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tdpbusd, tdpbsud, tdpbf16ps,
)


comptime PREFILL_BLOCK_N = 512
comptime BF16_K_STEP = 32
comptime Q_BATCH = 32  # running_o = 32*128*4 = 16KB → L1

# Profiling
comptime PF_NUM_PHASES = 7
comptime PF_QPREP = 0
comptime PF_K_PACK = 1
comptime PF_SCORE_GEMM = 2
comptime PF_SOFTMAX = 3
comptime PF_W_CONVERT = 4
comptime PF_V_CONVERT = 5
comptime PF_VAGG_GEMM = 6

def pf_phase_name(p: Int) -> String:
    if p == 0: return "Q prep"
    if p == 1: return "K pack"
    if p == 2: return "Score GEMM"
    if p == 3: return "Softmax"
    if p == 4: return "W convert"
    if p == 5: return "V convert+pack"
    if p == 6: return "V agg GEMM"
    return "?"


@always_inline
def pack_k_tile_vnni(
    k_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    k_off: Int, n_off: Int, n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = k_base + (n_off + col) * head_dim + k_off + 4 * kg
            var d = kg * 64 + col * 4
            dst[d] = src[0]; dst[d+1] = src[1]; dst[d+2] = src[2]; dst[d+3] = src[3]
        for col in range(n_cols, TILE_N):
            var d = kg * 64 + col * 4
            dst[d] = 0; dst[d+1] = 0; dst[d+2] = 0; dst[d+3] = 0


# ============================================================================
# Scratch sizing
# ============================================================================

def attn_per_worker_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                          max_prefill: Int = 512]() -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    comptime total_q_max = ((max_prefill * gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime padded_bn = PREFILL_BLOCK_N
    return (
        total_q_max * head_dim                                      # qi_buf [total_q, hd] i8
        + total_q_max * size_of[Float32]() * 2                      # qi_scales, qi_biases
        + total_q_max * head_dim * size_of[Float32]()               # running_o [total_q, hd] f32
        + total_q_max * size_of[Float32]() * 2                      # running_m, running_l
        + Q_BATCH * padded_bn * size_of[Float32]()                  # score_buf [batch, block_n] f32
        + Q_BATCH * padded_bn * size_of[Scalar[DType.bfloat16]]()   # w_bf16 [batch, block_n] bf16
        + (head_dim // K_STEP) * 2 * TILE_BYTES                     # k_vnni
        + 2 * padded_bn * head_dim * size_of[Scalar[DType.bfloat16]]()  # v_bf16 flat + packed
    )

def attn_scratch_bytes_amx_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                                   max_prefill: Int = 512](num_workers: Int) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime per_worker = attn_per_worker_bytes[num_heads, num_kv_heads, head_dim, max_prefill]()
    return max_prefill * q_cols * size_of[Float32]() + num_workers * per_worker


# ============================================================================
# Dispatched kernel — one group, subset of Q rows
# ============================================================================

def had_attn_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, group: Int, q_row_start: Int, q_row_end: Int,
    worker_scratch: Int, timing_addr: Int,
):
    var ctx = UnsafePointer[HadAttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var timing = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=timing_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    comptime BLOCK_N = PREFILL_BLOCK_N
    comptime BF16_TILE_BYTES = (BF16_K_STEP // 2) * TILE_N * 4

    comptime DATA_HEAD_STRIDE = max_seq * head_dim
    comptime SCALE_HEAD_STRIDE = max_seq * size_of[Float32]()

    var seq_len = ctx[].seq_len
    var pos = ctx[].pos
    var g = group
    var my_q_rows = q_row_end - q_row_start
    if my_q_rows <= 0:
        return
    var padded_q_rows = ((my_q_rows + M_STEP - 1) // M_STEP) * M_STEP

    # Tile config
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    # Scratch layout — sized for this worker's Q rows
    var off = 0
    var qi_buf = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += my_q_rows * head_dim
    var qi_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += my_q_rows * size_of[Float32]()
    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += my_q_rows * size_of[Float32]()
    var running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * head_dim * size_of[Float32]()
    var running_m = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * size_of[Float32]()
    var running_l = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * size_of[Float32]()
    var score_buf = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    var score_i32 = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += Q_BATCH * BLOCK_N * size_of[Float32]()
    var w_bf16_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += Q_BATCH * BLOCK_N * size_of[Scalar[DType.bfloat16]]()
    var k_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += (head_dim // K_STEP) * 2 * TILE_BYTES
    var v_bf16_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=worker_scratch + off)

    comptime score_stride_i32 = BLOCK_N * size_of[Int32]()

    var k_base = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=ctx[].k_data + g * DATA_HEAD_STRIDE)
    var k_sc_base = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=ctx[].k_scale + g * SCALE_HEAD_STRIDE)
    var v_base = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=ctx[].v_data + g * DATA_HEAD_STRIDE)
    var v_sc_base = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=ctx[].v_scale + g * SCALE_HEAD_STRIDE)

    # =================================================================
    # 1. Q prep — only this worker's rows
    # =================================================================
    var t0 = perf_counter_ns()
    for local_idx in range(my_q_rows):
        var abs_row = q_row_start + local_idx
        var m = abs_row // gqa_factor
        var hi = abs_row % gqa_factor
        var h = g * gqa_factor + hi
        var actual_pos = pos + m
        var cos_row = ctx[].cos + actual_pos * half
        var sin_row = ctx[].sin + actual_pos * half
        var q_head = InlineArray[Float32, head_dim](fill=Float32(0))
        var qp = UnsafePointer(to=q_head).bitcast[Float32]()
        var q_row = ctx[].q + m * q_cols + h * head_dim
        var k = 0
        while k + width <= head_dim:
            (qp + k).store((q_row + k).load[width=width]().cast[DType.float32]())
            k += width
        var j = 0
        while j + width <= half:
            var x_lo = (qp + j).load[width=width]()
            var x_hi = (qp + half + j).load[width=width]()
            var cv = (cos_row + j).load[width=width]()
            var sv = (sin_row + j).load[width=width]()
            (qp + j).store(x_lo * cv - x_hi * sv)
            (qp + half + j).store(x_hi * cv + x_lo * sv)
            j += width
        fwht_block[DType.float32, head_dim](qp)
        var vmax = SIMD[DType.float32, width](0)
        k = 0
        while k + width <= head_dim:
            vmax = max(vmax, (qp + k).load[width=width]().__abs__())
            k += width
        var q_absmax = vmax.reduce_max()
        var q_inv = Float32(127.0) / q_absmax if q_absmax > 0 else Float32(0)
        var q_vinv = SIMD[DType.float32, width](q_inv)
        var qi_row = qi_buf + local_idx * head_dim
        var q_sum_acc = SIMD[DType.int32, width](0)
        k = 0
        while k + width <= head_dim:
            var qi = quantize_i8((qp + k).load[width=width](), q_vinv)
            (qi_row + k).store(qi)
            q_sum_acc += qi.cast[DType.int32]()
            k += width
        qi_scales[local_idx] = (q_absmax / Float32(127.0)) * inv_sqrt_hd
        qi_biases[local_idx] = Float32(128 * Int(q_sum_acc.reduce_add()))
    timing[PF_QPREP] += Int64(perf_counter_ns() - t0)

    # =================================================================
    # 2. Init running state
    # =================================================================
    for i in range(padded_q_rows * head_dim):
        running_o[i] = Float32(0)
    for i in range(padded_q_rows):
        running_m[i] = Float32(-1e30)
        running_l[i] = Float32(0)

    var max_context = pos + seq_len

    # =================================================================
    # 3. KV-outer, query-inner loop
    # =================================================================
    for block_start in range(0, max_context, BLOCK_N):
        var block_len = min(BLOCK_N, max_context - block_start)
        var padded_bn = ((block_len + BF16_K_STEP - 1) // BF16_K_STEP) * BF16_K_STEP
        var bf16_k_iters = padded_bn // BF16_K_STEP

        # --- Pack K (redundant per worker in same group) ---
        t0 = perf_counter_ns()
        var n_tiles = (block_len + TILE_N - 1) // TILE_N
        var k_scratch_arr = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
            fill=SIMD[DType.uint32, TILE_N](0))
        for nt in range(0, n_tiles, 2):
            var n_off = block_start + nt * TILE_N
            for ki in range(head_dim // K_STEP):
                var k_off = ki * K_STEP
                var n0 = min(TILE_N, block_len - nt * TILE_N)
                if n0 == TILE_N:
                    transpose_generic[DType.uint32, TILE_N](
                        (k_base + n_off * head_dim + k_off).bitcast[Scalar[DType.uint32]](),
                        head_dim // 4,
                        (k_vnni + ki * 2 * TILE_BYTES).bitcast[Scalar[DType.uint32]](),
                        TILE_N, k_scratch_arr)
                else:
                    pack_k_tile_vnni(k_base, head_dim, k_off, n_off, n0,
                                     k_vnni + ki * 2 * TILE_BYTES)
                var n1 = max(0, min(TILE_N, block_len - (nt + 1) * TILE_N))
                if n1 == TILE_N:
                    transpose_generic[DType.uint32, TILE_N](
                        (k_base + (n_off + TILE_N) * head_dim + k_off).bitcast[Scalar[DType.uint32]](),
                        head_dim // 4,
                        (k_vnni + ki * 2 * TILE_BYTES + TILE_BYTES).bitcast[Scalar[DType.uint32]](),
                        TILE_N, k_scratch_arr)
                else:
                    pack_k_tile_vnni(k_base, head_dim, k_off, n_off + TILE_N, n1,
                                     k_vnni + ki * 2 * TILE_BYTES + TILE_BYTES)
        timing[PF_K_PACK] += Int64(perf_counter_ns() - t0)

        # --- Convert V + VNNI pack (redundant per worker in same group) ---
        t0 = perf_counter_ns()
        comptime V128 = SIMD[DType.float32, width](128.0)
        var hd_n_iters = head_dim // TILE_N
        for d in range(head_dim):
            var v_dim = v_base + d * max_seq + block_start
            var vb = v_bf16_buf + d * padded_bn
            var t = 0
            while t + width <= block_len:
                var raw = (v_dim + t).load[width=width]().cast[DType.float32]()
                var vsc = (v_sc_base + block_start + t).load[width=width]()
                (vb + t).store(((raw - V128) * vsc).cast[DType.bfloat16]())
                t += width
            while t < padded_bn:
                if t < block_len:
                    vb[t] = Scalar[DType.bfloat16](
                        (Float32(Int(v_dim[t])) - 128.0) * v_sc_base[block_start + t])
                else:
                    vb[t] = Scalar[DType.bfloat16](0)
                t += 1

        var v_packed = v_bf16_buf + head_dim * padded_bn
        var v_scratch_arr = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
            fill=SIMD[DType.uint32, TILE_N](0))
        for nt in range(hd_n_iters):
            for kt in range(bf16_k_iters):
                transpose_generic[DType.uint32, TILE_N](
                    (v_bf16_buf + nt * TILE_N * padded_bn + kt * BF16_K_STEP).bitcast[Scalar[DType.uint32]](),
                    padded_bn // 2,
                    (v_packed + (nt * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2)).bitcast[Scalar[DType.uint32]](),
                    TILE_N, v_scratch_arr)
        timing[PF_V_CONVERT] += Int64(perf_counter_ns() - t0)

        # --- Query-inner loop: process Q_BATCH rows at a time ---
        for qb_start in range(0, my_q_rows, Q_BATCH):
            var qb_len = min(Q_BATCH, my_q_rows - qb_start)
            var qb_m_iters = (qb_len + M_STEP - 1) // M_STEP

            # Score GEMM for this batch
            t0 = perf_counter_ns()
            for nt in range(0, n_tiles, 2):
                for mi in range(qb_m_iters):
                    var m_off = qb_start + mi * M_STEP
                    tilezero[4]()
                    tilezero[5]()
                    tilezero[6]()
                    tilezero[7]()
                    for ki in range(head_dim // K_STEP):
                        tileload[0](qi_buf + m_off * head_dim + ki * K_STEP, head_dim)
                        tileload[1](qi_buf + (m_off + TILE_M) * head_dim + ki * K_STEP, head_dim)
                        tileload[2](k_vnni + ki * 2 * TILE_BYTES, TILE_N * VNNI_BLK)
                        tileload[3](k_vnni + ki * 2 * TILE_BYTES + TILE_BYTES, TILE_N * VNNI_BLK)
                        tdpbsud[4, 0, 2]()
                        tdpbsud[5, 0, 3]()
                        tdpbsud[6, 1, 2]()
                        tdpbsud[7, 1, 3]()
                    var sb = score_i32 + (mi * M_STEP) * BLOCK_N + nt * TILE_N
                    tilestore[4](sb, score_stride_i32)
                    tilestore[5](sb + TILE_N, score_stride_i32)
                    tilestore[6](sb + TILE_M * BLOCK_N, score_stride_i32)
                    tilestore[7](sb + TILE_M * BLOCK_N + TILE_N, score_stride_i32)
            timing[PF_SCORE_GEMM] += Int64(perf_counter_ns() - t0)

            # Softmax for this batch
            t0 = perf_counter_ns()
            for qi_local in range(qb_len):
                var qi_row = qb_start + qi_local
                var abs_row = q_row_start + qi_row
                var m_pos = abs_row // gqa_factor
                var causal_limit = pos + m_pos + 1 - block_start
                var q_sc = qi_scales[qi_row]
                var q_bi = qi_biases[qi_row]
                var vq_sc = SIMD[DType.float32, width](q_sc)
                var vq_bi = SIMD[DType.float32, width](q_bi)

                var s_row = score_buf + qi_local * BLOCK_N
                var si_row = score_i32 + qi_local * BLOCK_N

                # Dequant + max
                var vmax = SIMD[DType.float32, width](Float32(-1e30))
                var t = 0
                while t + width <= min(causal_limit, block_len):
                    var raw = (si_row + t).load[width=width]().cast[DType.float32]()
                    var ksc = (k_sc_base + block_start + t).load[width=width]()
                    var dq = (raw - vq_bi) * vq_sc * ksc
                    (s_row + t).store(dq)
                    vmax = max(vmax, dq)
                    t += width
                while t < min(causal_limit, block_len):
                    var dq = (Float32(si_row[t]) - q_bi) * q_sc * k_sc_base[block_start + t]
                    s_row[t] = dq
                    if dq > vmax.reduce_max(): vmax = SIMD[DType.float32, width](dq)
                    t += 1
                var row_max = vmax.reduce_max()
                while t < block_len:
                    s_row[t] = Float32(-1e30)
                    t += 1

                # Online softmax update
                var m_old = running_m[qi_row]
                var m_new = max(m_old, row_max)
                var correction = exp_f32_fast[1](m_old - m_new)
                running_m[qi_row] = m_new
                running_l[qi_row] = running_l[qi_row] * correction

                var ro = running_o + qi_row * head_dim
                var vcorr = SIMD[DType.float32, width](correction)
                var d = 0
                while d + width <= head_dim:
                    (ro + d).store((ro + d).load[width=width]() * vcorr)
                    d += width

                # Exp
                var vm_new = SIMD[DType.float32, width](m_new)
                var l_acc = SIMD[DType.float32, width](0)
                t = 0
                while t + width <= block_len:
                    var e = exp_f32_fast((s_row + t).load[width=width]() - vm_new)
                    (s_row + t).store(e)
                    l_acc += e
                    t += width
                var l_contrib = l_acc.reduce_add()
                while t < block_len:
                    var e = exp_f32_fast[1](s_row[t] - m_new)
                    s_row[t] = e
                    l_contrib += e
                    t += 1
                running_l[qi_row] += l_contrib
            timing[PF_SOFTMAX] += Int64(perf_counter_ns() - t0)

            # W convert for this batch
            t0 = perf_counter_ns()
            for qi_local in range(qb_len):
                var s_row = score_buf + qi_local * BLOCK_N
                var w_row = w_bf16_buf + qi_local * padded_bn
                var t = 0
                while t + width <= block_len:
                    (w_row + t).store((s_row + t).load[width=width]().cast[DType.bfloat16]())
                    t += width
                while t < padded_bn:
                    if t < block_len:
                        w_row[t] = Scalar[DType.bfloat16](s_row[t])
                    else:
                        w_row[t] = Scalar[DType.bfloat16](0)
                    t += 1
            timing[PF_W_CONVERT] += Int64(perf_counter_ns() - t0)

            # V agg bf16 GEMM for this batch
            t0 = perf_counter_ns()
            var hd_ns = head_dim // N_STEP
            for ns in range(hd_ns):
                var d_off = ns * N_STEP
                for mi in range(qb_m_iters):
                    var m_off = qb_start + mi * M_STEP
                    var c_base = running_o + m_off * head_dim + d_off
                    var c_stride = head_dim * size_of[Float32]()
                    tileload[4](c_base, c_stride)
                    tileload[5](c_base + TILE_N, c_stride)
                    tileload[6](c_base + TILE_M * head_dim, c_stride)
                    tileload[7](c_base + TILE_M * head_dim + TILE_N, c_stride)
                    for kt in range(bf16_k_iters):
                        var a_ptr = w_bf16_buf + (mi * M_STEP) * padded_bn + kt * BF16_K_STEP
                        tileload[0](a_ptr, padded_bn * 2)
                        tileload[1](a_ptr + TILE_M * padded_bn, padded_bn * 2)
                        var nt_lo = d_off // TILE_N
                        tileload[2](v_packed + (nt_lo * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2), TILE_N * 4)
                        tileload[3](v_packed + ((nt_lo+1) * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2), TILE_N * 4)
                        tdpbf16ps[4, 0, 2]()
                        tdpbf16ps[5, 0, 3]()
                        tdpbf16ps[6, 1, 2]()
                        tdpbf16ps[7, 1, 3]()
                    tilestore[4](c_base, c_stride)
                    tilestore[5](c_base + TILE_N, c_stride)
                    tilestore[6](c_base + TILE_M * head_dim, c_stride)
                    tilestore[7](c_base + TILE_M * head_dim + TILE_N, c_stride)
            timing[PF_VAGG_GEMM] += Int64(perf_counter_ns() - t0)

    # =================================================================
    # 4. Final normalize — write to shared row_f32 at absolute positions
    # =================================================================
    for local_idx in range(my_q_rows):
        var abs_row = q_row_start + local_idx
        var m = abs_row // gqa_factor
        var hi = abs_row % gqa_factor
        var h = g * gqa_factor + hi
        var inv_l = Float32(1.0) / running_l[local_idx]
        var ro = running_o + local_idx * head_dim
        for d in range(head_dim):
            ctx[].row_f32[m * q_cols + h * head_dim + d] = ro[d] * inv_l


# ============================================================================
# Public API
# ============================================================================

def int8_gqa_attention_amx_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
    QT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    q: DynView[QT],
    k_cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    v_cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    scratch: Int,
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Async AMX prefill attention with Q-row parallelism.

    Dispatches workers_per_group * num_kv_heads jobs. Each worker handles
    a subset of Q rows for one KV group, redundantly packing K/V.
    Caller must join the returned PoolFence, then do output quantization
    of row_f32 (scratch base) which holds seq_len * q_cols f32 values.
    """
    comptime assert QT.DTYPE == DType.bfloat16
    comptime assert QiT.DTYPE == DType.int8
    comptime assert ScT.DTYPE == DType.float32

    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime max_prefill = 512
    comptime total_q_max = ((max_prefill * gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime padded_bn = PREFILL_BLOCK_N
    comptime BF16_TILE_BYTES = (BF16_K_STEP // 2) * TILE_N * 4

    comptime PER_WORKER = (
        total_q_max * head_dim
        + total_q_max * size_of[Float32]() * 2
        + total_q_max * head_dim * size_of[Float32]()
        + total_q_max * size_of[Float32]() * 2
        + Q_BATCH * padded_bn * size_of[Float32]()
        + Q_BATCH * padded_bn * size_of[Scalar[DType.bfloat16]]()
        + (head_dim // K_STEP) * 2 * TILE_BYTES
        + 2 * padded_bn * head_dim * size_of[Scalar[DType.bfloat16]]()
    )
    comptime WORKERS_OFF = max_prefill * q_cols * size_of[Float32]()

    var ctx = HadAttnCtx(
        q=UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr),
        qi=UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_out.ptr),
        sc=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_out.ptr),
        cos=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        sin=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        k_data=k_cache.data_base, k_scale=k_cache.scale_base,
        v_data=v_cache.data_base, v_scale=v_cache.scale_base,
        row_f32=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch),
        pos=pos, seq_len=q.seq_len,
    )
    var ctx_ptr = UnsafePointer(to=ctx)

    # Split Q rows across workers within each KV group
    var workers_per_group = max(1, pool.capacity // num_kv_heads)
    var total_q_rows = q.seq_len * gqa_factor
    var rows_per_worker = (total_q_rows + workers_per_group - 1) // workers_per_group
    var total_jobs = num_kv_heads * workers_per_group

    comptime MAX_TOTAL_WORKERS = 64
    var timing_arr = InlineArray[Int64, MAX_TOTAL_WORKERS * PF_NUM_PHASES](fill=Int64(0))
    var timing_base = UnsafePointer(to=timing_arr).bitcast[Int64]()

    for g in range(num_kv_heads):
        for w in range(workers_per_group):
            var job_idx = g * workers_per_group + w
            var q_start = w * rows_per_worker
            var q_end = min(q_start + rows_per_worker, total_q_rows)
            var pack = pool.args_base + job_idx
            pack[].arg0 = Int(ctx_ptr)
            pack[].arg1 = g
            pack[].arg2 = q_start
            pack[].arg3 = q_end
            pack[].arg4 = scratch + WORKERS_OFF + job_idx * PER_WORKER
            pack[].arg5 = Int(timing_base + job_idx * PF_NUM_PHASES)

    pool.dispatch(
        had_attn_prefill[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, total_jobs,
    )
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
