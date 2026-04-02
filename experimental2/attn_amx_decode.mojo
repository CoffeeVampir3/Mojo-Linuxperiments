"""AMX decode attention — 1-3-3 tile layout, specialized for SL=1.

Uses TMM0(A) × TMM1-3(B) → TMM4-6(C) for decode where gqa_factor ≤ 16
fits exactly in one A tile. Processes 3 N-tiles per A load — 50% more
throughput vs the 2-2-4 layout which wastes the second A tile.

Score:  Q_i8 × K_u8_vnni → i32      (tdpbsud, signed Q × unsigned K)
V-agg: W_u8 × V_i8_vnni → i32      (tdpbusd, unsigned W × signed V)
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
from simd_math import sqrt
from simd_math.matrixops import transpose_rows
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_1x3, prep_q_row, softmax_row
from experimental2.kernel_profile import KernelProfile, tap, ProfileAggregator
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_133_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tile_dp,
)


comptime BLOCK_N = 512
# Score buffer width padded to multiple of 3 tiles for 1-3-3 grouping
comptime SCORE_N = ((BLOCK_N // TILE_N + 2) // 3) * 3 * TILE_N  # 528
comptime N_STEP_133 = 3 * TILE_N  # 48


# ============================================================================
# Per-worker scratch — typed pointers produced by caller
# ============================================================================

@fieldwise_init
struct DecodeWorkerScratch:
    var group: Int
    var qi_buf: UnsafePointer[Int8, MutAnyOrigin]
    var qi_biases: UnsafePointer[Float32, MutAnyOrigin]
    var running_o: UnsafePointer[Float32, MutAnyOrigin]
    var running_m: UnsafePointer[Float32, MutAnyOrigin]
    var running_l: UnsafePointer[Float32, MutAnyOrigin]
    var score_i32: UnsafePointer[Int32, MutAnyOrigin]
    var w_u8_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var k_vnni: UnsafePointer[UInt8, MutAnyOrigin]
    var v_vnni: UnsafePointer[Int8, MutAnyOrigin]
    var profile: UnsafePointer[KernelProfile, MutAnyOrigin]


# ============================================================================
# Scratch sizing
# ============================================================================

def per_worker_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int]() -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    comptime padded_q = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime k_slices = head_dim // K_STEP
    comptime k_n_tiles_padded = ((BLOCK_N // TILE_N + 2) // 3) * 3
    comptime v_k_tiles = BLOCK_N // K_STEP
    comptime hd_n_padded = ((head_dim // TILE_N + 2) // 3) * 3
    return (
        size_of[DecodeWorkerScratch]()
        + padded_q * head_dim                            # qi_buf i8
        + gqa_factor * size_of[Float32]()                # qi_biases
        + padded_q * head_dim * size_of[Float32]()       # running_o f32
        + padded_q * size_of[Float32]() * 2              # running_m, running_l
        + padded_q * SCORE_N * size_of[Int32]()          # score_i32 (wider for 1-3-3)
        + padded_q * BLOCK_N                             # w_u8_buf u8
        + k_n_tiles_padded * k_slices * TILE_BYTES       # k_vnni (individual tiles)
        + hd_n_padded * v_k_tiles * TILE_BYTES           # v_vnni (padded for triple groups)
        + size_of[KernelProfile]()                       # profile
    )


def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    num_workers: Int,
) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    return q_cols * size_of[Float32]() + num_workers * pw


# ============================================================================
# Kernel — 1-3-3 tile layout, one worker per KV group
# ============================================================================

def attn_decode[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, ws_addr: Int, unused0: Int, unused1: Int,
    unused2: Int, unused3: Int,
):
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
    comptime HEAD_STRIDE = max_seq * head_dim

    var pos = ctx[].pos
    var score_scale = ctx[].score_scale
    var vagg_scale = ctx[].vagg_scale
    var g = ws[].group

    var t0 = tap()

    var cfg = make_133_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    var t1 = tap()

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

    # =================================================================
    # Q prep — RoPE → FWHT → quantize (gqa_factor rows, all at pos)
    # =================================================================
    var q_quant_inv = ctx[].q_quant_inv
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

    var context = pos + 1
    var vagg_arr = InlineArray[Int32, TILE_M * N_STEP_133](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    var ns_k_pack = 0
    var ns_v_pack = 0
    var ns_score = 0
    var ns_softmax = 0
    var ns_vagg = 0

    # =================================================================
    # KV-outer loop
    # =================================================================
    for block_start in range(0, context, BLOCK_N):
        var block_len = min(BLOCK_N, context - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N
        var n_groups = (n_tiles + 2) // 3
        var n_padded = n_groups * 3

        var tb0 = tap()

        # --- Pack K (individual tiles for 1-3-3 grouping) ---
        for nt in range(n_tiles):
            var k_tile_base = k_vnni + nt * K_TILE_K_BYTES
            var n_off = block_start + nt * TILE_N
            var n_cols = max(0, min(TILE_N, block_len - nt * TILE_N))
            for ki in range(k_slices):
                var k_off = ki * K_STEP
                var k_rows = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
                    fill=SIMD[DType.uint32, TILE_N](0))
                for col in range(n_cols):
                    k_rows[col] = (k_base + (n_off + col) * head_dim + k_off).bitcast[UInt32]().load[width=TILE_N]()
                transpose_rows[DType.uint32, TILE_N](
                    k_rows,
                    (k_tile_base + ki * TILE_BYTES).bitcast[UInt32](),
                    TILE_N)
        # Zero padding K tiles for complete triple groups
        comptime u8w = simd_width_of[DType.uint8]()
        comptime u8zeros = SIMD[DType.uint8, u8w](0)
        for nt in range(n_tiles, n_padded):
            var p = (k_vnni + nt * K_TILE_K_BYTES).bitcast[UInt8]()
            var bi = 0
            while bi + u8w <= K_TILE_K_BYTES:
                (p + bi).store(u8zeros)
                bi += u8w

        var tb1 = tap()

        # --- Pack V (with padding tile for triple groups) ---
        for nt in range(hd_n_tiles):
            for kt in range(chunk_k_iters):
                pack_v_tile_vnni(v_base, head_dim,
                    block_start + kt * K_STEP, nt * TILE_N,
                    min(K_STEP, block_len - kt * K_STEP),
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)
        # Zero padding V tiles
        for nt in range(hd_n_tiles, hd_n_padded):
            for kt in range(chunk_k_iters):
                pack_v_tile_vnni(v_base, head_dim,
                    block_start + kt * K_STEP, 0, 0,
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

        var tb2 = tap()

        # --- Score GEMM: i8 Q × u8 K → i32 (1-3-3 triple groups) ---
        for ng in range(n_groups):
            var nt_base = ng * 3
            var b0 = k_vnni + nt_base * K_TILE_K_BYTES
            var b1 = k_vnni + (nt_base + 1) * K_TILE_K_BYTES
            var b2 = k_vnni + (nt_base + 2) * K_TILE_K_BYTES
            for mi in range(m_iters):
                amx_gemm_1x3[DType.int8, DType.uint8, SCORE_N](
                    qi_buf + mi * TILE_M * head_dim,
                    head_dim,
                    b0, b1, b2,
                    TILE_BYTES, k_slices,
                    score_i32 + mi * TILE_M * SCORE_N + nt_base * TILE_N,
                )

        var tb3 = tap()

        # --- Softmax (stride = SCORE_N, all rows share causal_limit = block_len) ---
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

        # --- V-agg: u8 W × i8 V → i32 (1-3-3 triple groups) ---
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

        ns_k_pack += tb1 - tb0
        ns_v_pack += tb2 - tb1
        ns_score += tb3 - tb2
        ns_softmax += tb4 - tb3
        ns_vagg += tb5 - tb4

    # =================================================================
    # Final normalize
    # =================================================================
    var t_fn0 = tap()
    for hi in range(gqa_factor):
        var h = g * gqa_factor + hi
        var final_scale = vagg_scale / running_l[hi]
        var ro = running_o + hi * head_dim
        var out = ctx[].row_f32 + h * head_dim
        var d = 0
        while d + width <= head_dim:
            (out + d).store((ro + d).load[width=width]() * final_scale)
            d += width
    var t_fn1 = tap()

    # Write profile
    var prof = ws[].profile
    prof[].amx_config = t1 - t0
    prof[].q_prep = t2 - t1
    prof[].running_init = t3 - t2
    prof[].k_pack = ns_k_pack
    prof[].v_pack = ns_v_pack
    prof[].score_gemm = ns_score
    prof[].softmax = ns_softmax
    prof[].vagg = ns_vagg
    prof[].merge_wait = 0
    prof[].merge_work = t_fn1 - t_fn0
    prof[].total = t_fn1 - t0


# ============================================================================
# Profile collection
# ============================================================================

def collect_profiles[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    scratch: Int,
) -> ProfileAggregator:
    """Read per-worker profiles from scratch and aggregate."""
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
    var agg = ProfileAggregator()
    for g in range(num_kv_heads):
        var ws_base = scratch + WORKERS_OFF + g * pw
        var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
            unsafe_from_address=ws_base)
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
    mut pool: BurstPool[],
) -> PoolFence:
    """Async AMX decode — 1-3-3 layout, SL=1, one worker per KV group."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime padded_q = ((gqa_factor + TILE_M - 1) // TILE_M) * TILE_M
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
    comptime k_slices = head_dim // K_STEP
    comptime k_n_tiles_padded = ((BLOCK_N // TILE_N + 2) // 3) * 3
    comptime v_k_tiles = BLOCK_N // K_STEP
    comptime hd_n_padded = ((head_dim // TILE_N + 2) // 3) * 3
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

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

    for g in range(num_kv_heads):
        var ws_base = scratch + WORKERS_OFF + g * pw
        var ws = UnsafePointer[DecodeWorkerScratch, MutAnyOrigin](
            unsafe_from_address=ws_base)
        var data = ws_base + size_of[DecodeWorkerScratch]()
        var off = 0
        ws[].group = g
        ws[].qi_buf = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * head_dim
        ws[].qi_biases = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
        off += gqa_factor * size_of[Float32]()
        ws[].running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * head_dim * size_of[Float32]()
        ws[].running_m = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * size_of[Float32]()
        ws[].running_l = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * size_of[Float32]()
        ws[].score_i32 = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * SCORE_N * size_of[Int32]()
        ws[].w_u8_buf = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * BLOCK_N
        ws[].k_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
        off += k_n_tiles_padded * k_slices * TILE_BYTES
        ws[].v_vnni = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=data + off)
        off += hd_n_padded * v_k_tiles * TILE_BYTES
        ws[].profile = UnsafePointer[KernelProfile, MutAnyOrigin](unsafe_from_address=data + off)
        ws[].profile[] = KernelProfile()

        var pack = pool.args_base + g
        pack[].arg0 = Int(ctx_ptr)
        pack[].arg1 = ws_base
        pack[].arg2 = 0
        pack[].arg3 = 0
        pack[].arg4 = 0
        pack[].arg5 = 0

    pool.dispatch(
        attn_decode[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, num_kv_heads,
    )
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
