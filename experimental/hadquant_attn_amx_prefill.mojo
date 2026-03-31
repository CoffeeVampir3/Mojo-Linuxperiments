"""AMX prefill attention kernel.

Processes all query positions at once per KV group:
  Q[seq_len, hd] as A operand (many rows, both tiles full)
  K/V packed as B operand per context block (amortized over seq_len)
  bf16 V agg via tdpbf16ps (no i8 quantize pipeline)

Scoring uses tdpbusd (i8): S[seq_len, block_n] = Q_i8[seq_len, hd] x K_vnni[hd, block_n]
V agg uses tdpbf16ps: O[seq_len, hd] += W_bf16[seq_len, block_n] x V_bf16_vnni[block_n, hd]

Online softmax between scoring and V agg per context block.
Causal mask applied post-GEMM on the boundary block.
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
from experimental.hadquant_impl import fwht_block
from experimental.hadquant_kv_cache import HadQuantKVCache
from experimental.hadquant_attn import HadAttnCtx
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tdpbusd, tdpbf16ps,
)


comptime PREFILL_BLOCK_N = 512  # context positions per block
comptime BF16_K_STEP = 32       # bf16 VNNI groups of 2


# ============================================================================
# K-cache VNNI packing (pack once per N-tile, reuse across M iterations)
# ============================================================================

@always_inline
def pack_k_tile_vnni(
    k_base: UnsafePointer[UInt8, MutAnyOrigin],
    head_dim: Int,
    k_off: Int,
    n_off: Int,
    n_cols: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pack K[n_off:n_off+n_cols, k_off:k_off+64] into VNNI for B operand.

    K cache is [pos][dim] u8. 4 bytes per VNNI group are contiguous in K.
    """
    for kg in range(K_STEP // VNNI_BLK):
        for col in range(n_cols):
            var src = k_base + (n_off + col) * head_dim + k_off + 4 * kg
            var d = kg * 64 + col * 4
            dst[d] = src[0]
            dst[d + 1] = src[1]
            dst[d + 2] = src[2]
            dst[d + 3] = src[3]
        for col in range(n_cols, TILE_N):
            var d = kg * 64 + col * 4
            dst[d] = 0
            dst[d + 1] = 0
            dst[d + 2] = 0
            dst[d + 3] = 0


# ============================================================================
# Scratch sizing
# ============================================================================

def attn_scratch_bytes_amx_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                                   max_prefill: Int = 512,
                                   num_workers: Int = 1]() -> Int:
    comptime q_cols = num_heads * head_dim
    comptime per_worker = (
        max_prefill * head_dim * size_of[Float32]()                # output_f32 [seq, hd]
        + max_prefill * PREFILL_BLOCK_N * size_of[Float32]()       # score_buf [seq, block_n]
        + max_prefill * head_dim * size_of[Float32]()              # running_o [seq, hd]
        + max_prefill * size_of[Float32]() * 2                     # running_m, running_l [seq]
        + max_prefill * head_dim                                    # qi [seq, hd] i8
        + (head_dim // K_STEP) * 2 * TILE_BYTES                   # k_vnni packed [2 K-tiles × 2 N-tiles]
        + 2 * PREFILL_BLOCK_N * head_dim * size_of[Scalar[DType.bfloat16]]()  # v_bf16 flat + packed
        + max_prefill * PREFILL_BLOCK_N * size_of[Scalar[DType.bfloat16]]()  # w_bf16 scratch
    )
    return q_cols * size_of[Float32]() + num_workers * per_worker


# ============================================================================
# Dispatched kernel
# ============================================================================

def had_attn_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, unused0: Int, group_start: Int, group_end: Int,
    worker_scratch: Int, unused5: Int,
):
    var ctx = UnsafePointer[HadAttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    comptime BLOCK_N = PREFILL_BLOCK_N

    comptime DATA_HEAD_STRIDE = max_seq * head_dim
    comptime SCALE_HEAD_STRIDE = max_seq * size_of[Float32]()

    var seq_len = ctx[].seq_len
    var pos = ctx[].pos

    # Tile config
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    # Scratch pointers (per-worker, laid out sequentially)
    var scratch_off = 0
    var qi_buf = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * head_dim
    var qi_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * size_of[Float32]()
    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * size_of[Float32]()
    var score_buf = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * BLOCK_N * size_of[Float32]()
    var running_o = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * head_dim * size_of[Float32]()
    var running_m = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * size_of[Float32]()
    var running_l = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += seq_len * size_of[Float32]()
    var k_vnni = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += (head_dim // K_STEP) * 2 * TILE_BYTES
    var v_bf16_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)
    scratch_off += BLOCK_N * head_dim * size_of[Scalar[DType.bfloat16]]()
    var w_bf16_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=worker_scratch + scratch_off)

    var score_i32 = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(score_buf))

    comptime score_stride_i32 = BLOCK_N * size_of[Int32]()

    for g in range(group_start, group_end):
        var k_base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=ctx[].k_data + g * DATA_HEAD_STRIDE)
        var k_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].k_scale + g * SCALE_HEAD_STRIDE)
        var v_base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=ctx[].v_data + g * DATA_HEAD_STRIDE)
        var v_sc_base = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=ctx[].v_scale + g * SCALE_HEAD_STRIDE)

        # =================================================================
        # 1. Q prep: RoPE + FWHT + quantize for ALL query positions × heads
        #    qi_buf: [seq_len * gqa_factor, head_dim] row-major i8
        #    qi_scales/biases: [seq_len * gqa_factor] per-head
        # =================================================================
        for m in range(seq_len):
            var actual_pos = pos + m
            var cos_row = ctx[].cos + actual_pos * half
            var sin_row = ctx[].sin + actual_pos * half

            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                var row_idx = m * gqa_factor + hi
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

                var qi_row = qi_buf + row_idx * head_dim
                var q_sum_acc = SIMD[DType.int32, width](0)
                k = 0
                while k + width <= head_dim:
                    var qi = quantize_i8((qp + k).load[width=width](), q_vinv)
                    (qi_row + k).store(qi)
                    q_sum_acc += qi.cast[DType.int32]()
                    k += width

                qi_scales[row_idx] = (q_absmax / Float32(127.0)) * inv_sqrt_hd
                qi_biases[row_idx] = Float32(128 * Int(q_sum_acc.reduce_add()))

        # Total Q rows = seq_len * gqa_factor
        var total_q_rows = seq_len * gqa_factor

        # =================================================================
        # 2. Init running state
        # =================================================================
        for i in range(total_q_rows * head_dim):
            running_o[i] = Float32(0)
        for i in range(total_q_rows):
            running_m[i] = Float32(-1e30)
            running_l[i] = Float32(0)

        # Max context any position attends to
        var max_context = pos + seq_len

        # =================================================================
        # 3. Block loop over context (FlashAttention-style)
        # =================================================================
        for block_start in range(0, max_context, BLOCK_N):
            var block_len = min(BLOCK_N, max_context - block_start)
            var m_iters = (total_q_rows + M_STEP - 1) // M_STEP

            # --- 3a. Scoring GEMM ---
            # S[total_q_rows, block_len] = Q_i8[total_q_rows, hd] x K_vnni[hd, block_len]
            # Q is A (row-major), K is B (VNNI-packed per N-tile, reused across M)

            # Pack K for this block: N tiles cover block_len positions
            var n_tiles = (block_len + TILE_N - 1) // TILE_N
            for nt in range(0, n_tiles, 2):
                var n_off = block_start + nt * TILE_N
                for ki in range(head_dim // K_STEP):
                    var k_off = ki * K_STEP
                    var n0 = min(TILE_N, block_len - nt * TILE_N)
                    pack_k_tile_vnni(k_base, head_dim, k_off, n_off, n0,
                                     k_vnni + ki * 2 * TILE_BYTES)
                    var n1 = max(0, min(TILE_N, block_len - (nt + 1) * TILE_N))
                    pack_k_tile_vnni(k_base, head_dim, k_off, n_off + TILE_N, n1,
                                     k_vnni + ki * 2 * TILE_BYTES + TILE_BYTES)

                # Score with packed K (all M iterations share this B)
                for mi in range(m_iters):
                    var m_off = mi * M_STEP
                    tilezero[4]()
                    tilezero[5]()
                    tilezero[6]()
                    tilezero[7]()
                    for ki in range(head_dim // K_STEP):
                        tileload[0](qi_buf + m_off * head_dim + ki * K_STEP, head_dim)
                        tileload[1](qi_buf + (m_off + TILE_M) * head_dim + ki * K_STEP, head_dim)
                        tileload[2](k_vnni + ki * 2 * TILE_BYTES, TILE_N * VNNI_BLK)
                        tileload[3](k_vnni + ki * 2 * TILE_BYTES + TILE_BYTES, TILE_N * VNNI_BLK)
                        tdpbusd[4, 0, 2]()
                        tdpbusd[5, 0, 3]()
                        tdpbusd[6, 1, 2]()
                        tdpbusd[7, 1, 3]()

                    # Store to score_buf[m_off:, nt*TILE_N:]
                    var s_base = score_i32 + m_off * BLOCK_N + nt * TILE_N
                    tilestore[4](s_base, score_stride_i32)
                    tilestore[5](s_base + TILE_N, score_stride_i32)
                    tilestore[6](s_base + TILE_M * BLOCK_N, score_stride_i32)
                    tilestore[7](s_base + TILE_M * BLOCK_N + TILE_N, score_stride_i32)

            # --- 3b. Dequant + causal mask + online softmax (SIMD over block_len) ---
            for qi_row in range(total_q_rows):
                var m_pos = qi_row // gqa_factor
                var context_for_pos = pos + m_pos + 1
                var causal_limit = context_for_pos - block_start  # positions visible in this block
                var q_sc = qi_scales[qi_row]
                var q_bi = qi_biases[qi_row]
                var vq_sc = SIMD[DType.float32, width](q_sc)
                var vq_bi = SIMD[DType.float32, width](q_bi)

                var s_row = score_buf + qi_row * BLOCK_N
                var si_row = score_i32 + qi_row * BLOCK_N

                # Dequant + find max (SIMD over block positions)
                var vmax = SIMD[DType.float32, width](Float32(-1e30))
                var t = 0
                while t + width <= min(causal_limit, block_len):
                    var raw = (si_row + t).load[width=width]().cast[DType.float32]()
                    var ksc = (k_sc_base + block_start + t).load[width=width]()
                    var dequant = (raw - vq_bi) * vq_sc * ksc
                    (s_row + t).store(dequant)
                    vmax = max(vmax, dequant)
                    t += width
                # Scalar tail for causal boundary
                while t < min(causal_limit, block_len):
                    var raw = Float32(si_row[t])
                    var dequant = (raw - q_bi) * q_sc * k_sc_base[block_start + t]
                    s_row[t] = dequant
                    if dequant > vmax.reduce_max():
                        vmax = SIMD[DType.float32, width](dequant)
                    t += 1
                var row_max = vmax.reduce_max()
                # Fill masked positions with -inf
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

                # Exp (SIMD) + accumulate l
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

            # --- 3c. Convert attention weights to bf16 ---
            # W as A operand: [total_q_rows, padded_block_n] bf16, row-major
            # Pad block_len to BF16_K_STEP (32) for clean tiling
            var padded_bn = ((block_len + BF16_K_STEP - 1) // BF16_K_STEP) * BF16_K_STEP
            for qi_row in range(total_q_rows):
                var s_row = score_buf + qi_row * BLOCK_N
                var w_row = w_bf16_buf + qi_row * padded_bn
                var t = 0
                while t + width <= block_len:
                    (w_row + t).store((s_row + t).load[width=width]().cast[DType.bfloat16]())
                    t += width
                while t < padded_bn:
                    w_row[t] = Scalar[DType.bfloat16](0)
                    t += 1

            # --- 3d. Convert V cache u8→bf16 (SIMD) then VNNI pack ---
            var bf16_k_iters = padded_bn // BF16_K_STEP
            var hd_n_iters = (head_dim + TILE_N - 1) // TILE_N
            comptime BF16_TILE_BYTES = (BF16_K_STEP // 2) * TILE_N * 4

            # Step 1: Convert V[hd, block_len] u8 → bf16 into flat buffer
            # v_bf16_buf used as [hd, padded_bn] bf16 temporarily
            comptime V128 = SIMD[DType.float32, width](128.0)
            for d in range(head_dim):
                var v_dim = v_base + d * max_seq + block_start
                var vb = v_bf16_buf + d * padded_bn
                var t = 0
                while t + width <= block_len:
                    # Load u8, widen to f32, dequant, truncate to bf16
                    var raw_u8 = (v_dim + t).load[width=width]().cast[DType.float32]()
                    var vsc = (v_sc_base + block_start + t).load[width=width]()
                    var val = (raw_u8 - V128) * vsc
                    (vb + t).store(val.cast[DType.bfloat16]())
                    t += width
                while t < padded_bn:
                    if t < block_len:
                        vb[t] = Scalar[DType.bfloat16](
                            (Float32(Int(v_dim[t])) - 128.0) * v_sc_base[block_start + t])
                    else:
                        vb[t] = Scalar[DType.bfloat16](0)
                    t += 1

            # Step 2: VNNI pack V^T from [hd, padded_bn] → tiles
            # V^T_vnni[kg, 2*d + b] = V[d, 2*kg + b] = v_bf16_flat[d * padded_bn + 2*kg + b]
            # Repurpose v_bf16_buf tail for packed tiles (or use separate area)
            # Pack directly into v_bf16_buf offset after flat data
            var v_flat_size = head_dim * padded_bn
            var v_packed = v_bf16_buf + v_flat_size
            for nt in range(hd_n_iters):
                var d_off = nt * TILE_N
                var d_cols = min(TILE_N, head_dim - d_off)
                for kt in range(bf16_k_iters):
                    var k_off = kt * BF16_K_STEP
                    var tile_dst = v_packed + (nt * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2)
                    for kg in range(BF16_K_STEP // 2):
                        for dc in range(d_cols):
                            var flat_off = (d_off + dc) * padded_bn + k_off + 2 * kg
                            tile_dst[kg * TILE_N * 2 + dc * 2] = v_bf16_buf[flat_off]
                            tile_dst[kg * TILE_N * 2 + dc * 2 + 1] = v_bf16_buf[flat_off + 1]
                        for dc in range(d_cols, TILE_N):
                            tile_dst[kg * TILE_N * 2 + dc * 2] = Scalar[DType.bfloat16](0)
                            tile_dst[kg * TILE_N * 2 + dc * 2 + 1] = Scalar[DType.bfloat16](0)

            # --- 3e. V agg bf16 GEMM (tdpbf16ps) ---
            # O[total_q_rows, hd] += W_bf16[total_q_rows, padded_bn] × V^T_vnni[padded_bn, hd]
            # W as A (row-major bf16), V^T as B (VNNI-packed bf16)
            # C accumulates f32 directly — no dequant needed!
            comptime bf16_stride = BF16_K_STEP * 2  # bf16 A row stride in bytes for one K-tile
            # Iterate N in steps of N_STEP=32 (2 B tiles of TILE_N=16 each)
            var hd_ns_iters = (head_dim + N_STEP - 1) // N_STEP
            for ns in range(hd_ns_iters):
                var d_off = ns * N_STEP
                for mi in range(m_iters):
                    var m_off = mi * M_STEP
                    # Load existing running_o into C tiles (f32 accumulate)
                    var c_base = running_o + m_off * head_dim + d_off
                    var c_stride_bytes = head_dim * size_of[Float32]()
                    tileload[4](c_base, c_stride_bytes)
                    tileload[5](c_base + TILE_N, c_stride_bytes)
                    tileload[6](c_base + TILE_M * head_dim, c_stride_bytes)
                    tileload[7](c_base + TILE_M * head_dim + TILE_N, c_stride_bytes)

                    for kt in range(bf16_k_iters):
                        # A = W_bf16[m_off:, kt*BF16_K_STEP:]
                        var a_ptr = w_bf16_buf + m_off * padded_bn + kt * BF16_K_STEP
                        tileload[0](a_ptr, padded_bn * 2)
                        tileload[1](a_ptr + TILE_M * padded_bn, padded_bn * 2)
                        # B lo = V^T_vnni tile for dims [d_off:d_off+16]
                        var nt_lo = (d_off // TILE_N)
                        var b_lo = v_packed + (nt_lo * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2)
                        tileload[2](b_lo, TILE_N * 2 * 2)
                        # B hi = V^T_vnni tile for dims [d_off+16:d_off+32]
                        var nt_hi = nt_lo + 1
                        var b_hi = v_packed + (nt_hi * bf16_k_iters + kt) * (BF16_TILE_BYTES // 2)
                        tileload[3](b_hi, TILE_N * 2 * 2)
                        tdpbf16ps[4, 0, 2]()
                        tdpbf16ps[5, 0, 3]()
                        tdpbf16ps[6, 1, 2]()
                        tdpbf16ps[7, 1, 3]()

                    # Store back
                    tilestore[4](c_base, c_stride_bytes)
                    tilestore[5](c_base + TILE_N, c_stride_bytes)
                    tilestore[6](c_base + TILE_M * head_dim, c_stride_bytes)
                    tilestore[7](c_base + TILE_M * head_dim + TILE_N, c_stride_bytes)

        # =================================================================
        # 4. Final normalize + write output
        # =================================================================
        for m in range(seq_len):
            for hi in range(gqa_factor):
                var h = g * gqa_factor + hi
                var qi_row = m * gqa_factor + hi
                var inv_l = Float32(1.0) / running_l[qi_row]
                var ro = running_o + qi_row * head_dim
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
    """AMX prefill attention."""
    comptime assert QT.DTYPE == DType.bfloat16
    comptime assert QiT.DTYPE == DType.int8
    comptime assert ScT.DTYPE == DType.float32

    comptime width = simd_width_of[DType.float32]()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime max_prefill = 512

    comptime PER_WORKER = (
        max_prefill * head_dim * size_of[Float32]()
        + max_prefill * PREFILL_BLOCK_N * size_of[Float32]()
        + max_prefill * head_dim * size_of[Float32]()
        + max_prefill * size_of[Float32]() * 2
        + max_prefill * head_dim
        + (head_dim // K_STEP) * 2 * TILE_BYTES
        + PREFILL_BLOCK_N * head_dim * size_of[Scalar[DType.bfloat16]]()
        + max_prefill * PREFILL_BLOCK_N * size_of[Scalar[DType.bfloat16]]()
    )
    comptime ROW_F32_OFF = 0
    comptime WORKERS_OFF = q_cols * size_of[Float32]()

    var row_f32 = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scratch + ROW_F32_OFF)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scale_out.ptr)

    var ctx = HadAttnCtx(
        q=UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr),
        qi=qi_ptr, sc=sc_ptr,
        cos=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        sin=UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        k_data=k_cache.data_base, k_scale=k_cache.scale_base,
        v_data=v_cache.data_base, v_scale=v_cache.scale_base,
        row_f32=row_f32, pos=pos, seq_len=q.seq_len,
    )

    var ctx_ptr = UnsafePointer(to=ctx)
    var num_jobs = min(num_kv_heads, pool.capacity)
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

    # Single dispatch for all positions
    for i in range(num_jobs):
        var start = i * groups_per_job
        var end = min(start + groups_per_job, num_kv_heads)
        var pack = pool.args_base + i
        pack[].arg0 = Int(ctx_ptr)
        pack[].arg1 = 0
        pack[].arg2 = start
        pack[].arg3 = end
        pack[].arg4 = scratch + WORKERS_OFF + i * PER_WORKER

    pool.dispatch(
        had_attn_prefill[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, num_jobs,
    )
    pool.join()

    # Output quantization for all positions
    for m in range(q.seq_len):
        var src = row_f32 + m * q_cols
        var out_row = qi_ptr + m * q_cols
        var rmax_v = SIMD[DType.float32, width](0)
        var d = 0
        while d + width <= q_cols:
            rmax_v = max(rmax_v, (src + d).load[width=width]().__abs__())
            d += width
        var row_absmax = rmax_v.reduce_max()
        while d < q_cols:
            var a = src[d] if src[d] >= 0 else -src[d]
            if a > row_absmax:
                row_absmax = a
            d += 1

        sc_ptr[m] = row_absmax / Float32(127.0)
        var row_inv = Float32(127.0) / row_absmax if row_absmax > 0 else Float32(0)
        var vrow_inv = SIMD[DType.float32, width](row_inv)
        d = 0
        while d + width <= q_cols:
            (out_row + d).store(quantize_i8((src + d).load[width=width](), vrow_inv))
            d += width
        while d < q_cols:
            out_row[d] = quantize_i8_scalar(src[d], row_inv)
            d += 1

    return PoolFence.completed()
