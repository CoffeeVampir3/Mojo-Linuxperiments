"""AMX decode attention — context-split, true GQA, multiple workers per KV group.

Same compute as the control kernel but splits the context dimension across
W workers per KV group. Each worker processes a disjoint range of KV
positions, producing partial softmax state (running_m, running_l, running_o).
After all workers complete, a merge step combines partial results using the
online softmax combination rule.

Q prep is done once per worker (redundant but cheap — gqa_factor rows).
K/V packing is not redundant — each worker packs only its own context range.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32_fast, quantize_i8
from simd_math.matrixops import transpose_rows
from experimental.hadquant_impl import fwht_apply, fwht_width
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_2x2
from experimental2.kernel_profile import KernelProfile, ProfileAggregator, CallerProfile, tap
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tile_dp,
)


comptime BLOCK_N = 512


# ============================================================================
# Per-worker scratch — typed pointers produced by caller
# ============================================================================

@fieldwise_init
struct DecodeWorkerScratch:
    var group: Int
    var ctx_start: Int
    var ctx_end: Int
    var profile: UnsafePointer[KernelProfile, MutAnyOrigin]
    var qi_buf: UnsafePointer[Int8, MutAnyOrigin]
    var qi_biases: UnsafePointer[Float32, MutAnyOrigin]
    var running_o: UnsafePointer[Float32, MutAnyOrigin]
    var running_m: UnsafePointer[Float32, MutAnyOrigin]
    var running_l: UnsafePointer[Float32, MutAnyOrigin]
    var score_i32: UnsafePointer[Int32, MutAnyOrigin]
    var w_u8_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var k_vnni: UnsafePointer[UInt8, MutAnyOrigin]
    var v_vnni: UnsafePointer[Int8, MutAnyOrigin]


# ============================================================================
# Scratch sizing
# ============================================================================

def per_worker_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int]() -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    comptime padded_m = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime k_nt_pairs = BLOCK_N // TILE_N // 2
    comptime k_slices = head_dim // K_STEP
    comptime v_k_tiles = BLOCK_N // K_STEP
    comptime v_n_tiles = head_dim // TILE_N
    return (
        size_of[DecodeWorkerScratch]()
        + size_of[KernelProfile]()                       # profile
        + padded_m * head_dim                            # qi_buf i8
        + padded_m * size_of[Float32]()                  # qi_biases
        + padded_m * head_dim * size_of[Float32]()       # running_o f32
        + padded_m * size_of[Float32]() * 2              # running_m, running_l
        + padded_m * BLOCK_N * size_of[Int32]()          # score_i32
        + padded_m * BLOCK_N                             # w_u8_buf u8
        + k_nt_pairs * k_slices * 2 * TILE_BYTES         # k_vnni
        + v_k_tiles * v_n_tiles * TILE_BYTES             # v_vnni
    )

def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    num_workers: Int, max_workers_per_group: Int = 0,
) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    var wpg = max(1, num_workers // num_kv_heads)
    if max_workers_per_group > 0:
        wpg = min(wpg, max_workers_per_group)
    var total_workers = num_kv_heads * wpg
    return q_cols * size_of[Float32]() + total_workers * pw + size_of[CallerProfile]()


# ============================================================================
# Worker kernel — processes a context range for one KV group
# ============================================================================

def attn_decode_worker[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, ws_addr: Int, unused0: Int, unused1: Int,
    unused2: Int, unused3: Int,
):
    var t0 = tap()
    var ctx = UnsafePointer[AttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](unsafe_from_address=ws_addr)
    var prof = ws[].profile
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime hd_n_tiles = head_dim // TILE_N
    comptime k_slices = head_dim // K_STEP
    comptime K_PAIR_BYTES = k_slices * 2 * TILE_BYTES
    comptime padded_m = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP

    comptime HEAD_STRIDE = max_seq * head_dim

    var pos = ctx[].pos
    var score_scale = ctx[].score_scale
    var vagg_scale = ctx[].vagg_scale
    var g = ws[].group
    var ctx_start = ws[].ctx_start
    var ctx_end = ws[].ctx_end

    var qi_buf = ws[].qi_buf
    var qi_biases = ws[].qi_biases
    var running_o = ws[].running_o
    var running_m = ws[].running_m
    var running_l = ws[].running_l
    var score_i32 = ws[].score_i32
    var w_u8_buf = ws[].w_u8_buf
    var k_vnni = ws[].k_vnni
    var v_vnni = ws[].v_vnni

    var k_base = ctx[].k_base + g * HEAD_STRIDE
    var v_base = ctx[].v_base + g * HEAD_STRIDE

    var t1 = tap()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))
    prof[].amx_config = tap() - t1

    # =================================================================
    # Q prep — redundant per worker but cheap (gqa_factor rows)
    # =================================================================
    t1 = tap()
    var q_quant_inv = ctx[].q_quant_inv
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    var vq_inv = SIMD[DType.float32, fwht_w](q_quant_inv)
    comptime fwht_regs = head_dim // fwht_w
    comptime half_regs = fwht_regs // 2

    for hi in range(gqa_factor):
        var h = g * gqa_factor + hi
        var cos_row = ctx[].cos + pos * half
        var sin_row = ctx[].sin + pos * half
        var q_row = ctx[].q + h * head_dim

        var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
            fill=SIMD[DType.float32, fwht_w](0))
        for ri in range(half_regs):
            var j = ri * fwht_w
            var x_lo = (q_row + j).load[width=fwht_w]().cast[DType.float32]()
            var x_hi = (q_row + half + j).load[width=fwht_w]().cast[DType.float32]()
            var cv = (cos_row + j).load[width=fwht_w]()
            var sv = (sin_row + j).load[width=fwht_w]()
            r[ri] = x_lo * cv - x_hi * sv
            r[half_regs + ri] = x_hi * cv + x_lo * sv

        fwht_apply[DType.float32, head_dim](r)

        var qi_row = qi_buf + hi * head_dim
        var q_sum_acc = SIMD[DType.int32, fwht_w](0)
        for ri in range(fwht_regs):
            var qi = quantize_i8[fwht_w](r[ri], vq_inv)
            (qi_row + ri * fwht_w).store(qi)
            q_sum_acc += qi.cast[DType.int32]()
        qi_biases[hi] = Float32(q_sum_acc.reduce_add()) * Float32(128)
    prof[].q_prep = tap() - t1

    # =================================================================
    # Init running state
    # =================================================================
    t1 = tap()
    var vzero = SIMD[DType.float32, width](0)
    var vinf = SIMD[DType.float32, width](Float32(-1e30))
    var i = 0
    while i + width <= padded_m * head_dim:
        (running_o + i).store(vzero)
        i += width
    i = 0
    while i + width <= padded_m:
        (running_m + i).store(vinf)
        (running_l + i).store(vzero)
        i += width

    prof[].running_init = tap() - t1

    var my_ctx_len = ctx_end - ctx_start
    if my_ctx_len <= 0:
        prof[].total = tap() - t0
        return

    var vagg_arr = InlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    # =================================================================
    # KV-block loop — stream through this worker's context range
    # =================================================================
    for block_start in range(ctx_start, ctx_end, BLOCK_N):
        var block_len = min(BLOCK_N, ctx_end - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N

        # --- Pack K ---
        t1 = tap()
        for nt in range(0, n_tiles, 2):
            var k_pair_base = k_vnni + (nt // 2) * K_PAIR_BYTES
            var n_off = block_start + nt * TILE_N
            for ki in range(k_slices):
                var k_off = ki * K_STEP
                for ti in range(2):
                    var n_cols = max(0, min(TILE_N, block_len - (nt + ti) * TILE_N))
                    var k_rows = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
                        fill=SIMD[DType.uint32, TILE_N](0))
                    for col in range(n_cols):
                        k_rows[col] = (k_base + (n_off + ti * TILE_N + col) * head_dim + k_off).bitcast[UInt32]().load[width=TILE_N]()
                    transpose_rows[DType.uint32, TILE_N](
                        k_rows,
                        (k_pair_base + ki * 2 * TILE_BYTES + ti * TILE_BYTES).bitcast[UInt32](),
                        TILE_N)

        prof[].k_pack += tap() - t1

        # --- Pack V ---
        t1 = tap()
        for nt in range(hd_n_tiles):
            for kt in range(chunk_k_iters):
                pack_v_tile_vnni(v_base, head_dim,
                    block_start + kt * K_STEP, nt * TILE_N,
                    min(K_STEP, block_len - kt * K_STEP),
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

        prof[].v_pack += tap() - t1

        # --- Score GEMM ---
        t1 = tap()
        comptime m_iters = (padded_m + M_STEP - 1) // M_STEP
        for nt in range(0, n_tiles, 2):
            var k_nt_base = k_vnni + (nt // 2) * K_PAIR_BYTES
            for mi in range(m_iters):
                var m_off = mi * M_STEP
                amx_gemm_2x2[DType.int8, DType.uint8, BLOCK_N](
                    qi_buf + m_off * head_dim,
                    qi_buf + (m_off + TILE_M) * head_dim,
                    head_dim,
                    k_nt_base, k_nt_base + TILE_BYTES,
                    2 * TILE_BYTES, k_slices,
                    score_i32 + m_off * BLOCK_N + nt * TILE_N,
                )

        prof[].score_gemm += tap() - t1

        # --- Softmax ---
        t1 = tap()
        for hi in range(gqa_factor):
            var q_bi = qi_biases[hi]
            var q_sc = score_scale
            var si_row = score_i32 + hi * BLOCK_N

            var vmax = SIMD[DType.float32, width](Float32(-1e30))
            var t = 0
            while t + width <= block_len:
                var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bi) * q_sc
                vmax = max(vmax, dq)
                t += width
            var scalar_max = Float32(-1e30)
            while t < block_len:
                scalar_max = max(scalar_max, (Float32(si_row[t]) - q_bi) * q_sc)
                t += 1
            var row_max = max(vmax.reduce_max(), scalar_max)

            var m_old = running_m[hi]
            var m_new = max(m_old, row_max)
            running_m[hi] = m_new
            if m_new > m_old:
                var correction = exp_f32_fast[1](m_old - m_new)
                running_l[hi] = running_l[hi] * correction
                var ro = running_o + hi * head_dim
                var d = 0
                while d + width <= head_dim:
                    (ro + d).store((ro + d).load[width=width]() * correction)
                    d += width

            var l_acc = SIMD[DType.float32, width](0)
            var w_row = w_u8_buf + hi * padded_chunk
            t = 0
            while t + width <= block_len:
                var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bi) * q_sc
                var e = exp_f32_fast(dq - m_new)
                (w_row + t).store(roundeven(e * Float32(255)).cast[DType.uint8]())
                l_acc += e
                t += width
            var l_contrib = l_acc.reduce_add()
            while t < block_len:
                var dq = (Float32(si_row[t]) - q_bi) * q_sc
                var e = exp_f32_fast[1](dq - m_new)
                w_row[t] = roundeven[DType.float32, 1](e * Float32(255)).cast[DType.uint8]()
                l_contrib += e
                t += 1
            comptime u8w = simd_width_of[DType.uint8]()
            comptime u8zeros = SIMD[DType.uint8, u8w](0)
            while t + u8w <= padded_chunk:
                (w_row + t).store(u8zeros)
                t += u8w
            running_l[hi] += l_contrib

        prof[].softmax += tap() - t1

        # --- V-agg ---
        t1 = tap()
        for ns in range(head_dim // N_STEP):
            var d_off = ns * N_STEP
            var nt_lo = d_off // TILE_N
            for mi in range(m_iters):
                amx_gemm_2x2[DType.uint8, DType.int8, N_STEP](
                    w_u8_buf + (mi * M_STEP) * padded_chunk,
                    w_u8_buf + (mi * M_STEP + TILE_M) * padded_chunk,
                    padded_chunk,
                    v_vnni + nt_lo * chunk_k_iters * TILE_BYTES,
                    v_vnni + (nt_lo + 1) * chunk_k_iters * TILE_BYTES,
                    TILE_BYTES, chunk_k_iters,
                    vagg_i32,
                )
                for r in range(min(M_STEP, gqa_factor - mi * M_STEP)):
                    var ro_row = running_o + (mi * M_STEP + r) * head_dim + d_off
                    var c = 0
                    while c + width <= N_STEP:
                        var raw = (vagg_i32 + r * N_STEP + c).load[width=width]().cast[DType.float32]()
                        (ro_row + c).store((ro_row + c).load[width=width]() + raw)
                        c += width
        prof[].vagg += tap() - t1

    prof[].total = tap() - t0


# ============================================================================
# Public API
# ============================================================================

def decode[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
    QT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    q: DynView[QT],
    k_cache: KVCache[max_seq, head_dim, num_kv_heads],
    v_cache: KVCache[max_seq, head_dim, num_kv_heads],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    scratch: Int,
    pos: Int,
    q_layer_scale: Float32,
    k_layer_scale: Float32,
    v_layer_scale: Float32,
    mut pool: BurstPool[],
    max_workers_per_group: Int = 0,
) -> PoolFence:
    """Async AMX decode — context-split across workers, true GQA."""
    var tc0 = tap()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
    comptime padded_m = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime width = simd_width_of[DType.float32]()
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    var workers_per_group_val = max(1, pool.capacity // num_kv_heads)
    if max_workers_per_group > 0:
        workers_per_group_val = min(workers_per_group_val, max_workers_per_group)
    var total_jobs_val = num_kv_heads * workers_per_group_val
    var cp = UnsafePointer[CallerProfile, MutAnyOrigin](
        unsafe_from_address=scratch + WORKERS_OFF + total_jobs_val * pw)
    cp[] = CallerProfile()

    var ctx = AttnCtx(
        UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=q.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=k_cache.data_base),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=v_cache.data_base),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch),
        Float32(127.0) / q_layer_scale,
        q_layer_scale * k_layer_scale * inv_sqrt_hd / (Float32(127.0) * Float32(127.0)),
        v_layer_scale / (Float32(255.0) * Float32(127.0)),
        pos, 1,
    )
    var ctx_ptr = UnsafePointer(to=ctx)
    var vagg_scale = ctx.vagg_scale

    var workers_per_group = workers_per_group_val
    var total_jobs = total_jobs_val
    var context_len = pos
    cp[].setup = tap() - tc0

    comptime K_NT_PAIRS = BLOCK_N // TILE_N // 2
    comptime K_SLICES = head_dim // K_STEP
    comptime V_K_TILES = BLOCK_N // K_STEP
    comptime V_N_TILES = head_dim // TILE_N

    # Build per-worker scratch and assign context ranges
    for g in range(num_kv_heads):
        var chunk_size = (context_len + workers_per_group - 1) // workers_per_group
        # Align chunk boundaries to BLOCK_N for clean tiling
        chunk_size = ((chunk_size + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

        for w in range(workers_per_group):
            var job_idx = g * workers_per_group + w
            var c_start = min(w * chunk_size, context_len)
            var c_end = min(c_start + chunk_size, context_len)

            var ws_base = scratch + WORKERS_OFF + job_idx * pw
            var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
                unsafe_from_address=ws_base)
            var data = ws_base + size_of[DecodeWorkerScratch]()
            var off = 0
            ws[].group = g
            ws[].ctx_start = c_start
            ws[].ctx_end = c_end
            ws[].profile = UnsafePointer[KernelProfile, MutAnyOrigin](unsafe_from_address=data + off)
            ws[].profile[] = KernelProfile()
            off += size_of[KernelProfile]()
            ws[].qi_buf = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * head_dim
            ws[].qi_biases = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * size_of[Float32]()
            ws[].running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * head_dim * size_of[Float32]()
            ws[].running_m = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * size_of[Float32]()
            ws[].running_l = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * size_of[Float32]()
            ws[].score_i32 = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * BLOCK_N * size_of[Int32]()
            ws[].w_u8_buf = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_m * BLOCK_N
            ws[].k_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
            off += K_NT_PAIRS * K_SLICES * 2 * TILE_BYTES
            ws[].v_vnni = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=data + off)

            var pack = pool.args_base + job_idx
            pack[].arg0 = Int(ctx_ptr)
            pack[].arg1 = ws_base
            pack[].arg2 = 0
            pack[].arg3 = 0
            pack[].arg4 = 0
            pack[].arg5 = 0

    var tc1 = tap()
    pool.dispatch(
        attn_decode_worker[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, total_jobs,
    )
    cp[].dispatch = tap() - tc1

    tc1 = tap()
    pool.join()
    cp[].join = tap() - tc1

    # =================================================================
    # Merge partial results across workers per KV group
    # =================================================================
    tc1 = tap()
    var out = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch)

    for g in range(num_kv_heads):
        for hi in range(gqa_factor):
            var h = g * gqa_factor + hi

            # Start from worker 0's state
            var ws0_base = scratch + WORKERS_OFF + g * workers_per_group * pw
            var ws0 = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
                unsafe_from_address=ws0_base)
            var merged_m = ws0[].running_m[hi]
            var merged_l = ws0[].running_l[hi]
            var merged_o = ws0[].running_o + hi * head_dim

            # Merge remaining workers
            for w in range(1, workers_per_group):
                var wsw_base = scratch + WORKERS_OFF + (g * workers_per_group + w) * pw
                var wsw = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
                    unsafe_from_address=wsw_base)
                var w_m = wsw[].running_m[hi]
                var w_l = wsw[].running_l[hi]
                var w_o = wsw[].running_o + hi * head_dim

                var new_m = max(merged_m, w_m)
                var c_old = exp_f32_fast[1](merged_m - new_m)
                var c_new = exp_f32_fast[1](w_m - new_m)

                merged_l = merged_l * c_old + w_l * c_new
                var vc_old = SIMD[DType.float32, width](c_old)
                var vc_new = SIMD[DType.float32, width](c_new)
                var d = 0
                while d + width <= head_dim:
                    var o_merged = (merged_o + d).load[width=width]() * vc_old
                    var o_worker = (w_o + d).load[width=width]() * vc_new
                    (merged_o + d).store(o_merged + o_worker)
                    d += width
                merged_m = new_m

            # Final normalize and write output
            var final_scale = vagg_scale / merged_l
            var d = 0
            while d + width <= head_dim:
                (out + h * head_dim + d).store(
                    (merged_o + d).load[width=width]() * final_scale)
                d += width

    cp[].merge = tap() - tc1
    cp[].total = tap() - tc0

    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def read_caller_profile[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    scratch: Int, num_workers: Int, max_wpg: Int = 0,
) -> CallerProfile:
    """Read the CallerProfile stored at the end of scratch."""
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
    var wpg = max(1, num_workers // num_kv_heads)
    if max_wpg > 0: wpg = min(wpg, max_wpg)
    var total_workers = num_kv_heads * wpg
    var cp = UnsafePointer[CallerProfile, MutAnyOrigin](
        unsafe_from_address=scratch + WORKERS_OFF + total_workers * pw)
    return cp[]


def collect_profiles[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    scratch: Int,
    num_workers: Int,
    mut agg: ProfileAggregator,
    max_wpg: Int = 0,
):
    """Read worker profiles from scratch after join."""
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
    var wpg = max(1, num_workers // num_kv_heads)
    if max_wpg > 0: wpg = min(wpg, max_wpg)
    var total_workers = num_kv_heads * wpg
    for j in range(total_workers):
        var ws_base = scratch + WORKERS_OFF + j * pw
        var data = ws_base + size_of[DecodeWorkerScratch]()
        var prof = UnsafePointer[KernelProfile, MutAnyOrigin](unsafe_from_address=data)
        agg.add(prof[])
