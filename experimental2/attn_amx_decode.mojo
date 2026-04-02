"""AMX decode attention — specialized for SL=1.

Same integer pipeline as prefill (zero per-token scales, uniform KV cache)
but optimized for the single-token case:
- All gqa_factor Q rows processed in one pass (no Q batching)
- Single causal limit for all rows (pos + 1)
- One worker per KV group

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
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_2x2, prep_q_row, softmax_row
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
    comptime padded_q = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime k_nt_pairs = BLOCK_N // TILE_N // 2
    comptime k_slices = head_dim // K_STEP
    comptime v_k_tiles = BLOCK_N // K_STEP
    comptime v_n_tiles = head_dim // TILE_N
    return (
        size_of[DecodeWorkerScratch]()
        + padded_q * head_dim                            # qi_buf i8
        + gqa_factor * size_of[Float32]()                # qi_biases
        + padded_q * head_dim * size_of[Float32]()       # running_o f32
        + padded_q * size_of[Float32]() * 2              # running_m, running_l
        + padded_q * BLOCK_N * size_of[Int32]()          # score_i32
        + padded_q * BLOCK_N                             # w_u8_buf
        + k_nt_pairs * k_slices * 2 * TILE_BYTES         # k_vnni
        + v_k_tiles * v_n_tiles * TILE_BYTES             # v_vnni
    )


def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    num_workers: Int,
) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    return q_cols * size_of[Float32]() + num_workers * pw


# ============================================================================
# Kernel — one worker per KV group
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
    comptime k_slices = head_dim // K_STEP
    comptime K_PAIR_BYTES = k_slices * 2 * TILE_BYTES
    comptime padded_q = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime m_iters = padded_q // M_STEP
    comptime HEAD_STRIDE = max_seq * head_dim

    var pos = ctx[].pos
    var score_scale = ctx[].score_scale
    var vagg_scale = ctx[].vagg_scale
    var g = ws[].group

    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

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

    var context = pos + 1
    var vagg_arr = InlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    # =================================================================
    # KV-outer loop
    # =================================================================
    for block_start in range(0, context, BLOCK_N):
        var block_len = min(BLOCK_N, context - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N

        # --- Pack K (all N-tile pairs) ---
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

        # --- Pack V ---
        for nt in range(hd_n_tiles):
            for kt in range(chunk_k_iters):
                pack_v_tile_vnni(v_base, head_dim,
                    block_start + kt * K_STEP, nt * TILE_N,
                    min(K_STEP, block_len - kt * K_STEP),
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

        # --- Score GEMM: i8 Q × u8 K → i32 ---
        for nt in range(0, n_tiles, 2):
            var k_nt_base = k_vnni + (nt // 2) * K_PAIR_BYTES
            for mi in range(m_iters):
                amx_gemm_2x2[DType.int8, DType.uint8, BLOCK_N](
                    qi_buf + mi * M_STEP * head_dim,
                    qi_buf + (mi * M_STEP + TILE_M) * head_dim,
                    head_dim,
                    k_nt_base, k_nt_base + TILE_BYTES,
                    2 * TILE_BYTES, k_slices,
                    score_i32 + mi * M_STEP * BLOCK_N + nt * TILE_N,
                )

        # --- Softmax (all rows share causal_limit = block_len) ---
        for hi in range(gqa_factor):
            softmax_row[head_dim](
                score_i32 + hi * BLOCK_N,
                qi_biases[hi], score_scale,
                block_len, padded_chunk,
                running_m + hi, running_l + hi,
                running_o + hi * head_dim,
                w_u8_buf + hi * padded_chunk,
            )

        # --- V-agg: u8 W × i8 V → i32 ---
        for ns in range(head_dim // N_STEP):
            var d_off = ns * N_STEP
            var nt_lo = d_off // TILE_N
            for mi in range(m_iters):
                amx_gemm_2x2[DType.uint8, DType.int8, N_STEP](
                    w_u8_buf + mi * M_STEP * padded_chunk,
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

    # =================================================================
    # Final normalize
    # =================================================================
    for hi in range(gqa_factor):
        var h = g * gqa_factor + hi
        var final_scale = vagg_scale / running_l[hi]
        var ro = running_o + hi * head_dim
        var out = ctx[].row_f32 + h * head_dim
        var d = 0
        while d + width <= head_dim:
            (out + d).store((ro + d).load[width=width]() * final_scale)
            d += width


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
    """Async AMX decode — SL=1, one worker per KV group."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim]()
    comptime padded_q = ((gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime WORKERS_OFF = q_cols * size_of[Float32]()
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

    comptime K_NT_PAIRS = BLOCK_N // TILE_N // 2
    comptime K_SLICES = head_dim // K_STEP
    comptime V_K_TILES = BLOCK_N // K_STEP
    comptime V_N_TILES = head_dim // TILE_N

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
        off += padded_q * BLOCK_N * size_of[Int32]()
        ws[].w_u8_buf = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
        off += padded_q * BLOCK_N
        ws[].k_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=data + off)
        off += K_NT_PAIRS * K_SLICES * 2 * TILE_BYTES
        ws[].v_vnni = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=data + off)

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
