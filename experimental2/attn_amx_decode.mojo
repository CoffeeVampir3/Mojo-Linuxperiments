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
from std.os.atomic import Atomic, Consistency
from threading import BurstPool
import linux.sys as linux

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven, exp_f32_fast, quantize_i8
from simd_math.matrixops import transpose_rows
from experimental.hadquant_impl import fwht_apply, fwht_width
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni
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

comptime AtomicInt32 = Atomic[DType.int32]


@fieldwise_init
struct DecodeWorkerScratch:
    var group: Int
    var ctx_start: Int
    var ctx_end: Int
    var worker_in_group: Int
    var workers_per_group: Int
    var group_state: Int           # raw address of per-group merge state
    var profile: UnsafePointer[KernelProfile, MutAnyOrigin]
    var qi_buf: UnsafePointer[Int8, MutAnyOrigin]
    var qi_biases: UnsafePointer[Float32, MutAnyOrigin]
    var running_o: UnsafePointer[Float32, MutAnyOrigin]
    var running_m: UnsafePointer[Float32, MutAnyOrigin]
    var running_l: UnsafePointer[Float32, MutAnyOrigin]
    var score_i32: UnsafePointer[Int32, MutAnyOrigin]
    var w_u8_buf: UnsafePointer[UInt8, MutAnyOrigin]


# Group merge state layout (per KV group, allocated in scratch):
#   [0]:                          counter (Int32, atomic) — workers remaining
#   [4]:                          merge_ready (Int32, atomic) — set by last worker
#   [8]:                          corrections[wpg * gqa_factor] (Float32)
#   [8 + wpg*gf*4]:              ro_ptrs[wpg] (Int) — running_o addresses
#   [8 + wpg*gf*4 + wpg*8]:      rm_ptrs[wpg] (Int) — running_m addresses
#   [8 + wpg*gf*4 + wpg*16]:     rl_ptrs[wpg] (Int) — running_l addresses
comptime MERGE_COUNTER_OFF = 0
comptime MERGE_READY_OFF = 4
comptime MERGE_CORR_OFF = 8

def group_state_bytes[num_heads: Int, num_kv_heads: Int](wpg: Int) -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    return 8 + wpg * gqa_factor * size_of[Float32]() + wpg * 3 * size_of[Int]()


# ============================================================================
# Scratch sizing
# ============================================================================

def per_worker_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int]() -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    comptime padded_m = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    return (
        size_of[DecodeWorkerScratch]()
        + size_of[KernelProfile]()                       # profile
        + padded_m * head_dim                            # qi_buf i8
        + padded_m * size_of[Float32]()                  # qi_biases
        + padded_m * head_dim * size_of[Float32]()       # running_o f32
        + padded_m * size_of[Float32]() * 2              # running_m, running_l
        + padded_m * BLOCK_N * size_of[Int32]()          # score_i32
        + padded_m * BLOCK_N                             # w_u8_buf u8
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
    var gs = group_state_bytes[num_heads, num_kv_heads](wpg)
    return q_cols * size_of[Float32]() + total_workers * pw + num_kv_heads * gs + size_of[CallerProfile]()


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
    comptime k_slices = head_dim // K_STEP
    comptime padded_m = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime hd_n_tiles = head_dim // TILE_N
    comptime VAGG_N = TILE_N * 3

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

    var vagg_arr = InlineArray[Int32, TILE_M * VAGG_N](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    # Stack buffer for fused VNNI packing (one tile, reused)
    var tile_buf = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
        fill=SIMD[DType.uint32, TILE_N](0))
    var tile_buf_addr = Int(UnsafePointer(to=tile_buf))

    # =================================================================
    # KV-block loop — stream through this worker's context range
    # =================================================================
    for block_start in range(ctx_start, ctx_end, BLOCK_N):
        var block_len = min(BLOCK_N, ctx_end - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N

        # --- Score GEMM (1-3-3 fused K pack) ---
        t1 = tap()
        for nt in range(0, n_tiles, 3):
            var nt_count = min(3, n_tiles - nt)
            tilezero[4](); tilezero[5](); tilezero[6]()
            if nt_count < 3:
                tilezero[3]()
            if nt_count < 2:
                tilezero[2]()
            for ki in range(k_slices):
                tileload[0](qi_buf + ki * K_STEP, head_dim)
                for ti in range(nt_count):
                    var n_cols = max(0, min(TILE_N, block_len - (nt + ti) * TILE_N))
                    var k_rows = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
                        fill=SIMD[DType.uint32, TILE_N](0))
                    for col in range(n_cols):
                        k_rows[col] = (k_base + (block_start + (nt + ti) * TILE_N + col) * head_dim + ki * K_STEP).bitcast[UInt32]().load[width=TILE_N]()
                    transpose_rows[DType.uint32, TILE_N](
                        k_rows,
                        UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=tile_buf_addr),
                        TILE_N)
                    if ti == 0:
                        tileload[1](UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                    elif ti == 1:
                        tileload[2](UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                    else:
                        tileload[3](UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                tile_dp[4, 0, 1, DType.int8, DType.uint8]()
                tile_dp[5, 0, 2, DType.int8, DType.uint8]()
                tile_dp[6, 0, 3, DType.int8, DType.uint8]()
            comptime score_stride = BLOCK_N * size_of[Int32]()
            var s_dst = score_i32 + nt * TILE_N
            tilestore[4](s_dst, score_stride)
            if nt_count >= 2:
                tilestore[5](s_dst + TILE_N, score_stride)
            if nt_count >= 3:
                tilestore[6](s_dst + 2 * TILE_N, score_stride)

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

        # --- V-agg (1-3-3 fused V pack) ---
        t1 = tap()
        comptime vagg_stride = VAGG_N * size_of[Int32]()
        for ns in range(0, hd_n_tiles, 3):
            var out_tiles = min(3, hd_n_tiles - ns)
            tilezero[4](); tilezero[5](); tilezero[6]()
            if out_tiles < 3:
                tilezero[3]()
            for kt in range(chunk_k_iters):
                tileload[0](w_u8_buf + kt * K_STEP, padded_chunk)
                var v_n_pos = min(K_STEP, block_len - kt * K_STEP)
                for ti in range(out_tiles):
                    pack_v_tile_vnni(v_base, head_dim,
                        block_start + kt * K_STEP, (ns + ti) * TILE_N,
                        v_n_pos,
                        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=tile_buf_addr))
                    if ti == 0:
                        tileload[1](UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                    elif ti == 1:
                        tileload[2](UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                    else:
                        tileload[3](UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=tile_buf_addr), TILE_N * VNNI_BLK)
                tile_dp[4, 0, 1, DType.uint8, DType.int8]()
                tile_dp[5, 0, 2, DType.uint8, DType.int8]()
                tile_dp[6, 0, 3, DType.uint8, DType.int8]()
            tilestore[4](vagg_i32, vagg_stride)
            tilestore[5](vagg_i32 + TILE_N, vagg_stride)
            if out_tiles >= 3:
                tilestore[6](vagg_i32 + 2 * TILE_N, vagg_stride)
            var d_off = ns * TILE_N
            var cols = out_tiles * TILE_N
            for r in range(gqa_factor):
                var ro_row = running_o + r * head_dim + d_off
                var c = 0
                while c + width <= cols:
                    var raw = (vagg_i32 + r * VAGG_N + c).load[width=width]().cast[DType.float32]()
                    (ro_row + c).store((ro_row + c).load[width=width]() + raw)
                    c += width
        prof[].vagg += tap() - t1

    # =================================================================
    # Fused merge — workers self-coordinate via atomic counter
    # =================================================================
    var gs = ws[].group_state
    var wpg = ws[].workers_per_group
    var wid = ws[].worker_in_group

    # Offsets into group state
    var corr_base = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=gs + MERGE_CORR_OFF)
    var ro_ptrs = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=gs + MERGE_CORR_OFF + wpg * gqa_factor * size_of[Float32]())
    var rm_ptrs = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(ro_ptrs) + wpg * size_of[Int]())
    var rl_ptrs = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(rm_ptrs) + wpg * size_of[Int]())

    var counter_p = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=gs + MERGE_COUNTER_OFF)
    var ready_p = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=gs + MERGE_READY_OFF)

    # Signal context phase complete
    var old = AtomicInt32.fetch_add[ordering=Consistency.ACQUIRE_RELEASE](counter_p, -1)

    if old == 1:
        # Last worker: compute scalar merge + precompute corrections
        var vagg_scale = ctx[].vagg_scale
        for hi in range(gqa_factor):
            # Find global max
            var m = Float32(-1e30)
            for wi in range(wpg):
                var rm = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rm_ptrs[wi])
                m = max(m, rm[hi])

            # Compute corrections and merged_l
            var l = Float32(0)
            for wi in range(wpg):
                var rm = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rm_ptrs[wi])
                var rl = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rl_ptrs[wi])
                var c = exp_f32_fast[1](rm[hi] - m)
                l += rl[hi] * c
                corr_base[wi * gqa_factor + hi] = c

            # Fold vagg_scale / merged_l into corrections
            var inv_l = vagg_scale / l
            for wi in range(wpg):
                corr_base[wi * gqa_factor + hi] *= inv_l

        # Signal merge ready
        AtomicInt32.store[ordering=Consistency.RELEASE](ready_p, 1)
    else:
        # Spin until last worker signals
        var sys = linux.linux_sys()
        while AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_p) == 0:
            sys.arch_cpu_relax()

    # All workers: pull-merge output for assigned heads
    var out = ctx[].row_f32 + g * gqa_factor * head_dim
    var total_out = gqa_factor * head_dim
    var my_start = wid * total_out // wpg
    var my_end = (wid + 1) * total_out // wpg
    # Align to SIMD width for clean vectorization
    my_start = (my_start // width) * width
    if wid + 1 < wpg:
        my_end = (my_end // width) * width

    var idx = my_start
    while idx + width <= my_end:
        var hi = idx // head_dim
        var d = idx % head_dim
        var c0 = corr_base[hi]   # correction for worker 0
        var ro0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=ro_ptrs[0])
        var acc = (ro0 + hi * head_dim + d).load[width=width]() * c0
        for wi in range(1, wpg):
            var c = corr_base[wi * gqa_factor + hi]
            var ro = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=ro_ptrs[wi])
            acc = acc + (ro + hi * head_dim + d).load[width=width]() * c
        (out + idx).store(acc)
        idx += width

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
    comptime padded_m = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime width = simd_width_of[DType.float32]()
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    var workers_per_group_val = max(1, pool.capacity // num_kv_heads)
    if max_workers_per_group > 0:
        workers_per_group_val = min(workers_per_group_val, max_workers_per_group)
    var total_jobs_val = num_kv_heads * workers_per_group_val
    var gs_size = group_state_bytes[num_heads, num_kv_heads](workers_per_group_val)
    var gs_base = scratch + WORKERS_OFF + total_jobs_val * pw
    var cp = UnsafePointer[CallerProfile, MutAnyOrigin](
        unsafe_from_address=gs_base + num_kv_heads * gs_size)
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

    # Build per-worker scratch, group merge state, and assign context ranges
    for g in range(num_kv_heads):
        var chunk_size = (context_len + workers_per_group - 1) // workers_per_group

        # Initialize per-group merge state
        var gs_addr = gs_base + g * gs_size
        var gs_counter = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=gs_addr + MERGE_COUNTER_OFF)
        var gs_ready = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=gs_addr + MERGE_READY_OFF)
        AtomicInt32.store[ordering=Consistency.RELEASE](gs_counter, Int32(workers_per_group))
        AtomicInt32.store[ordering=Consistency.RELEASE](gs_ready, 0)

        # Pointer arrays within group state
        var ro_ptrs_base = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=gs_addr + MERGE_CORR_OFF + workers_per_group * gqa_factor * size_of[Float32]())
        var rm_ptrs_base = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(ro_ptrs_base) + workers_per_group * size_of[Int]())
        var rl_ptrs_base = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(rm_ptrs_base) + workers_per_group * size_of[Int]())

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
            ws[].worker_in_group = w
            ws[].workers_per_group = workers_per_group
            ws[].group_state = gs_addr
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

            # Register this worker's pointers in group state
            ro_ptrs_base[w] = Int(ws[].running_o)
            rm_ptrs_base[w] = Int(ws[].running_m)
            rl_ptrs_base[w] = Int(ws[].running_l)

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

    # Merge is fused into worker kernels — output is ready
    cp[].merge = 0
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
    var gs = group_state_bytes[num_heads, num_kv_heads](wpg)
    var cp = UnsafePointer[CallerProfile, MutAnyOrigin](
        unsafe_from_address=scratch + WORKERS_OFF + total_workers * pw + num_kv_heads * gs)
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
