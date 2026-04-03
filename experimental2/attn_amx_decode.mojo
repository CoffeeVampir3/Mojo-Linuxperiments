"""AMX decode attention — 1-3-3 tiles, VNNI-primary K, multi-worker context split.

Workers within each KV group split the context range. Each worker processes
its block range independently, then decode_merge combines their online
softmax states and writes the final output.

Score:  Q_i8 × K_u8_vnni → i32      (tdpbsud, signed Q × unsigned K)
V-agg: W_u8 × V_i8_vnni → i32      (tdpbusd, unsigned W × signed V)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.isolated_burst_pool import IsolatedBurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt
from simd_math import exp_f32_fast
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_1x3, prep_q_row, softmax_row
from experimental2.kernel_profile import KernelProfile, tap, ProfileAggregator
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_133_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tile_dp,
)


comptime BLOCK_N = 512
comptime SCORE_N = ((BLOCK_N // TILE_N + 2) // 3) * 3 * TILE_N  # 528
comptime N_STEP_133 = 3 * TILE_N  # 48


# ============================================================================
# Per-worker scratch
# ============================================================================

@fieldwise_init
struct DecodeWorkerScratch:
    """Per-worker config + merge-accessible buffers.
    Kernel-local buffers (qi, scores, weights, V pack) are stack-allocated."""
    # Per-dispatch config
    var group: Int
    var ctx_start: Int
    var ctx_end: Int
    var pos: Int
    var q_quant_inv: Float32
    var score_scale: Float32
    var k_vnni_head: UnsafePointer[UInt8, MutAnyOrigin]
    # Merge-accessible (survive kernel for decode_merge + profiling)
    var running_o: UnsafePointer[Float32, MutAnyOrigin]
    var running_m: UnsafePointer[Float32, MutAnyOrigin]
    var running_l: UnsafePointer[Float32, MutAnyOrigin]
    var profile: UnsafePointer[KernelProfile, MutAnyOrigin]


# ============================================================================
# Scratch sizing
# ============================================================================

def per_worker_merge_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int]() -> Int:
    """Per-worker scratch that survives the kernel for merge + profiling."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime padded_q = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    return (
        size_of[DecodeWorkerScratch]()
        + padded_q * head_dim * size_of[Float32]()       # running_o f32
        + padded_q * size_of[Float32]() * 2              # running_m, running_l
        + size_of[KernelProfile]()                       # profile
    )


def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    num_workers: Int,
) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_merge_bytes[num_heads, num_kv_heads, head_dim]()
    return q_cols * size_of[Float32]() + size_of[AttnCtx]() + num_workers * pw


# ============================================================================
# Kernel — processes [ctx_start, ctx_end), no final normalize
# ============================================================================

def attn_decode[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, ws_addr: Int, unused0: Int, unused1: Int,
    unused2: Int, unused3: Int,
):
    var t_entry = tap()
    var ctx = UnsafePointer[AttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](unsafe_from_address=ws_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime half = head_dim // 2
    comptime hd_n_tiles = head_dim // TILE_N
    comptime hd_n_padded = ((hd_n_tiles + 2) // 3) * 3
    comptime k_slices = head_dim // K_STEP
    comptime K_TILE_K_BYTES = k_slices * TILE_BYTES
    comptime padded_q = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime m_iters = padded_q // TILE_M
    comptime V_HEAD_STRIDE = max_seq * head_dim

    comptime v_k_tiles = BLOCK_N // K_STEP

    var g = ws[].group
    var ctx_start = ws[].ctx_start
    var ctx_end = ws[].ctx_end
    var pos = ws[].pos
    var score_scale = ws[].score_scale
    var q_quant_inv = ws[].q_quant_inv

    var t0 = tap()

    var cfg = make_133_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    var t1 = tap()

    # Merge-accessible state (in scratch, survives kernel for decode_merge)
    var running_o = ws[].running_o
    var running_m = ws[].running_m
    var running_l = ws[].running_l
    var k_vnni_head = ws[].k_vnni_head

    # Stack-allocated working memory (all written before read — no init needed)
    var qi_arr = InlineArray[Int8, padded_q * head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Int8]()
    var bias_arr = InlineArray[Float32, padded_q](uninitialized=True)
    var qi_biases = UnsafePointer(to=bias_arr).bitcast[Float32]()
    var score_arr = InlineArray[Int32, padded_q * SCORE_N](uninitialized=True)
    var score_i32 = UnsafePointer(to=score_arr).bitcast[Int32]()
    var wu8_arr = InlineArray[UInt8, padded_q * BLOCK_N](uninitialized=True)
    var w_u8_buf = UnsafePointer(to=wu8_arr).bitcast[UInt8]()
    var vv_arr = InlineArray[Int8, hd_n_padded * v_k_tiles * TILE_BYTES](uninitialized=True)
    var v_vnni = UnsafePointer(to=vv_arr).bitcast[Int8]()

    var v_base = ctx[].v_base + g * V_HEAD_STRIDE
    for hi in range(gqa_factor):
        var h = g * gqa_factor + hi
        qi_biases[hi] = prep_q_row[head_dim](
            ctx[].q + h * head_dim,
            ctx[].cos + pos * half,
            ctx[].sin + pos * half,
            q_quant_inv,
            qi_buf + hi * head_dim,
        )

    var t2 = tap()

    # =================================================================
    # Init running state
    # =================================================================
    var vzero = SIMD[DType.float32, width](0)
    var vinf = SIMD[DType.float32, width](Float32(-1e30))
    var i = 0
    while i + width <= padded_q * head_dim:
        (running_o + i).store(vzero)
        i += width
    i = 0
    while i + width <= padded_q:
        (running_m + i).store(vinf)
        (running_l + i).store(vzero)
        i += width

    var t3 = tap()

    var vagg_arr = InlineArray[Int32, TILE_M * N_STEP_133](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    var ns_k_read = 0
    var ns_v_pack = 0
    var ns_score = 0
    var ns_softmax = 0
    var ns_vagg = 0

    # =================================================================
    # KV-outer loop over this worker's context range
    # =================================================================
    for block_start in range(ctx_start, ctx_end, BLOCK_N):
        var block_len = min(BLOCK_N, ctx_end - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N
        var n_groups = (n_tiles + 2) // 3
        var block_tile_base = (block_start // TILE_N) * K_TILE_K_BYTES

        var tb0 = tap()
        var tb1 = tap()

        # --- Pack V ---
        for kt in range(chunk_k_iters):
            var kt_n_pos = min(K_STEP, block_len - kt * K_STEP)
            var kt_pos_off = block_start + kt * K_STEP
            for nt in range(hd_n_tiles):
                pack_v_tile_vnni(v_base, head_dim,
                    kt_pos_off, nt * TILE_N,
                    kt_n_pos,
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)
            for nt in range(hd_n_tiles, hd_n_padded):
                pack_v_tile_vnni(v_base, head_dim,
                    kt_pos_off, 0, 0,
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

        var tb2 = tap()

        # --- Score GEMM ---
        for ng in range(n_groups):
            var nt_base = ng * 3
            var b0 = k_vnni_head + block_tile_base + nt_base * K_TILE_K_BYTES
            var b1 = k_vnni_head + block_tile_base + (nt_base + 1) * K_TILE_K_BYTES
            var b2 = k_vnni_head + block_tile_base + (nt_base + 2) * K_TILE_K_BYTES
            for mi in range(m_iters):
                amx_gemm_1x3[DType.int8, DType.uint8, SCORE_N](
                    qi_buf + mi * TILE_M * head_dim,
                    head_dim,
                    b0, b1, b2,
                    TILE_BYTES, k_slices,
                    score_i32 + mi * TILE_M * SCORE_N + nt_base * TILE_N,
                )

        var tb3 = tap()

        # --- Softmax ---
        for hi in range(gqa_factor):
            softmax_row[head_dim](
                score_i32 + hi * SCORE_N,
                qi_biases[hi], score_scale,
                block_len, padded_chunk,
                running_m + hi, running_l + hi,
                running_o + hi * head_dim,
                w_u8_buf + hi * padded_chunk,
            )

        var tb4 = tap()

        # --- V-agg ---
        for ng in range(hd_n_padded // 3):
            var d_off = ng * N_STEP_133
            var nt0 = ng * 3
            var nt1 = ng * 3 + 1
            var nt2 = ng * 3 + 2
            for mi in range(m_iters):
                amx_gemm_1x3[DType.uint8, DType.int8, N_STEP_133](
                    w_u8_buf + mi * TILE_M * padded_chunk,
                    padded_chunk,
                    v_vnni + nt0 * chunk_k_iters * TILE_BYTES,
                    v_vnni + nt1 * chunk_k_iters * TILE_BYTES,
                    v_vnni + nt2 * chunk_k_iters * TILE_BYTES,
                    TILE_BYTES, chunk_k_iters,
                    vagg_i32,
                )
                var valid_cols = min(N_STEP_133, head_dim - d_off)
                for r in range(min(TILE_M, gqa_factor - mi * TILE_M)):
                    var ro_row = running_o + (mi * TILE_M + r) * head_dim + d_off
                    var c = 0
                    while c + width <= valid_cols:
                        var raw = (vagg_i32 + r * N_STEP_133 + c).load[width=width]().cast[DType.float32]()
                        (ro_row + c).store((ro_row + c).load[width=width]() + raw)
                        c += width

        var tb5 = tap()

        ns_k_read += tb1 - tb0
        ns_v_pack += tb2 - tb1
        ns_score += tb3 - tb2
        ns_softmax += tb4 - tb3
        ns_vagg += tb5 - tb4

    # No final normalize — decode_merge handles it
    var t_end = tap()

    var prof = ws[].profile
    prof[].amx_config = t1 - t0
    prof[].q_prep = t2 - t1
    prof[].running_init = t3 - t2
    prof[].k_pack = ns_k_read
    prof[].v_pack = ns_v_pack
    prof[].score_gemm = ns_score
    prof[].softmax = ns_softmax
    prof[].vagg = ns_vagg
    prof[].merge_wait = 0
    prof[].merge_work = 0
    prof[].total = t_end - t0
    var t_final = tap()
    prof[].overhead = (t0 - t_entry) + (t_final - t_end)
    prof[].done_timestamp = t_final


# ============================================================================
# Merge — combine partial softmax states, write final output
# ============================================================================

def decode_merge[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    scratch: Int,
    workers_per_group: Int,
    vagg_scale: Float32,
):
    """Merge partial online softmax states from context-split workers."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_merge_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]() + size_of[AttnCtx]()
    comptime width = simd_width_of[DType.float32]()
    var output = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch)

    for g in range(num_kv_heads):
        var ws0_base = scratch + WORKERS_OFF + (g * workers_per_group) * pw
        var ws0 = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](unsafe_from_address=ws0_base)
        var o0 = ws0[].running_o
        var m0 = ws0[].running_m
        var l0 = ws0[].running_l

        # Merge subsequent workers into worker 0's state
        for w in range(1, workers_per_group):
            var wsw_base = scratch + WORKERS_OFF + (g * workers_per_group + w) * pw
            var wsw = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](unsafe_from_address=wsw_base)
            var ow = wsw[].running_o
            var mw = wsw[].running_m
            var lw = wsw[].running_l

            for hi in range(gqa_factor):
                var m_new = max(m0[hi], mw[hi])
                var c0 = exp_f32_fast[1](m0[hi] - m_new)
                var cw = exp_f32_fast[1](mw[hi] - m_new)
                m0[hi] = m_new
                l0[hi] = c0 * l0[hi] + cw * lw[hi]
                var ro0 = o0 + hi * head_dim
                var row = ow + hi * head_dim
                var vc0 = SIMD[DType.float32, width](c0)
                var vcw = SIMD[DType.float32, width](cw)
                var d = 0
                while d + width <= head_dim:
                    (ro0 + d).store(vc0 * (ro0 + d).load[width=width]() + vcw * (row + d).load[width=width]())
                    d += width

        # Final normalize
        for hi in range(gqa_factor):
            var h = g * gqa_factor + hi
            var final_scale = vagg_scale / l0[hi]
            var ro = o0 + hi * head_dim
            var out = output + h * head_dim
            var d = 0
            while d + width <= head_dim:
                (out + d).store((ro + d).load[width=width]() * final_scale)
                d += width


# ============================================================================
# Profile collection
# ============================================================================

def collect_profiles[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    scratch: Int,
    workers_per_group: Int,
) -> ProfileAggregator:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_merge_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]() + size_of[AttnCtx]()
    var total = num_kv_heads * workers_per_group
    var agg = ProfileAggregator()
    for j in range(total):
        var ws_base = scratch + WORKERS_OFF + j * pw
        var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](unsafe_from_address=ws_base)
        agg.add(ws[].profile[])
    return agg^


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
    workers_per_group: Int,
    mut pool: IsolatedBurstPool[],
) -> PoolFence:
    """Async AMX decode — multi-worker context split within each KV group."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime pw = per_worker_merge_bytes[num_heads, num_kv_heads, head_dim]()
    comptime padded_q = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime k_slices = head_dim // K_STEP
    comptime tile_k_bytes = k_slices * TILE_BYTES
    comptime max_tiles = (max_seq + TILE_N - 1) // TILE_N
    comptime k_head_stride = max_tiles * tile_k_bytes
    comptime hd_n_padded = ((head_dim // TILE_N + 2) // 3) * 3
    comptime v_k_tiles = BLOCK_N // K_STEP
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    # Derived per-layer constants
    var q_quant_inv = Float32(127.0) / q_layer_scale
    var score_scale = q_layer_scale * k_layer_scale * inv_sqrt_hd / (Float32(127.0) * Float32(127.0))

    # Write AttnCtx into scratch (after output buffer, before workers)
    var output_bytes = q_cols * size_of[Float32]()
    var ctx_ptr = UnsafePointer[AttnCtx, MutAnyOrigin](
        unsafe_from_address=scratch + output_bytes)
    ctx_ptr[] = AttnCtx(
        UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=q.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=k_cache.k_base),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=v_cache.v_base),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch),
    )
    var workers_off = output_bytes + size_of[AttnCtx]()

    var context = pos + 1
    var max_blocks = (context + BLOCK_N - 1) // BLOCK_N
    var max_wpg = pool.capacity // num_kv_heads
    var wpg = min(min(workers_per_group, max_blocks), max_wpg)
    var blocks_per_worker = (max_blocks + wpg - 1) // wpg
    var total_jobs = num_kv_heads * wpg

    for g in range(num_kv_heads):
        for w in range(wpg):
            var first_block = w * blocks_per_worker
            var last_block = min((w + 1) * blocks_per_worker, max_blocks)
            var ctx_s = first_block * BLOCK_N
            var ctx_e = min(last_block * BLOCK_N, context)

            var job_idx = g * wpg + w
            var ws_base = scratch + workers_off + job_idx * pw
            var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
                unsafe_from_address=ws_base)
            var data = ws_base + size_of[DecodeWorkerScratch]()
            var off = 0
            ws[].group = g
            ws[].ctx_start = ctx_s
            ws[].ctx_end = ctx_e
            ws[].pos = pos
            ws[].q_quant_inv = q_quant_inv
            ws[].score_scale = score_scale
            ws[].k_vnni_head = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=k_cache.k_base + g * k_head_stride)
            # Merge-accessible buffers (survive kernel for decode_merge)
            ws[].running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_q * head_dim * size_of[Float32]()
            ws[].running_m = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_q * size_of[Float32]()
            ws[].running_l = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
            off += padded_q * size_of[Float32]()
            ws[].profile = UnsafePointer[KernelProfile, MutAnyOrigin](unsafe_from_address=data + off)
            ws[].profile[] = KernelProfile()

            var pack = pool.args_base + job_idx
            pack[].arg0 = Int(ctx_ptr)
            pack[].arg1 = ws_base
            pack[].arg2 = 0
            pack[].arg3 = 0
            pack[].arg4 = 0
            pack[].arg5 = 0

    pool.dispatch(
        attn_decode[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, total_jobs,
    )
    return PoolFence(UnsafePointer[IsolatedBurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
