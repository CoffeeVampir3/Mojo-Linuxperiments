"""AMX prefill attention — uniform integer pipeline, zero per-token scales.

K cache is u8 (i8 XOR 0x80) for tdpbsud. V cache is i8 directly for
tdpbusd. Per-layer fixed scales derived from weight norms and Hadamard
concentration. No per-token metadata. No per-row adaptive scaling.

Score:  Q_i8 × K_u8_vnni → i32      (tdpbsud, signed Q × unsigned K)
V-agg: W_u8 × V_i8_vnni → i32      (tdpbusd, unsigned W × signed V)

Softmax directly produces u8 weights: round(exp(s - max) * 255).
V-agg dequant is a single per-layer constant.

Parallelism: Q rows split across workers per KV group. Each worker
redundantly packs K/V (all cores active, no barriers).
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
from experimental.hadquant_impl import fwht_block
from experimental2.kv_cache import KVCache
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tdpbusd, tdpbsud,
)


comptime BLOCK_N = 512
comptime Q_BATCH = 32


# ============================================================================
# Context — passed to all workers
# ============================================================================

@fieldwise_init
struct AttnCtx:
    var q: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var cos: UnsafePointer[Float32, MutAnyOrigin]
    var sin: UnsafePointer[Float32, MutAnyOrigin]
    var k_data: Int
    var v_data: Int
    var row_f32: UnsafePointer[Float32, MutAnyOrigin]
    var q_quant_inv: Float32
    var score_scale: Float32
    var vagg_scale: Float32
    var pos: Int
    var seq_len: Int


# ============================================================================
# V VNNI pack — interleave i8 rows to [K/4,N,4]
# ============================================================================

@always_inline
def pack_v_tile_vnni(
    v_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    pos_off: Int, dim_off: Int, n_pos: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pack V[pos,dim] i8 → VNNI [K/4,N,4] via SIMD interleave."""
    comptime VNNI_GROUP_BYTES = TILE_N * VNNI_BLK
    comptime zeros = SIMD[DType.uint8, VNNI_GROUP_BYTES](0)
    var full_groups = n_pos // VNNI_BLK
    for kg in range(full_groups):
        var p = pos_off + kg * VNNI_BLK
        var r0 = (v_base + (p + 0) * head_dim + dim_off).load[width=TILE_N]()
        var r1 = (v_base + (p + 1) * head_dim + dim_off).load[width=TILE_N]()
        var r2 = (v_base + (p + 2) * head_dim + dim_off).load[width=TILE_N]()
        var r3 = (v_base + (p + 3) * head_dim + dim_off).load[width=TILE_N]()
        (dst + kg * VNNI_GROUP_BYTES).store(r0.interleave(r2).interleave(r1.interleave(r3)))
    for kg in range(full_groups, K_STEP // VNNI_BLK):
        (dst + kg * VNNI_GROUP_BYTES).store(zeros)


# ============================================================================
# Scratch sizing
# ============================================================================

def per_worker_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                     max_prefill: Int = 512]() -> Int:
    comptime gqa_factor = num_heads // num_kv_heads
    comptime total_q_max = ((max_prefill * gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime k_nt_pairs = BLOCK_N // TILE_N // 2
    comptime k_slices = head_dim // K_STEP
    comptime v_k_tiles = BLOCK_N // K_STEP
    comptime v_n_tiles = head_dim // TILE_N
    return (
        total_q_max * head_dim                              # qi_buf i8
        + total_q_max * size_of[Float32]()                  # qi_biases (128*sum)
        + total_q_max * head_dim * size_of[Float32]()       # running_o f32
        + total_q_max * size_of[Float32]() * 2              # running_m, running_l
        + Q_BATCH * BLOCK_N * size_of[Int32]()              # score_i32 (GEMM output)
        + Q_BATCH * BLOCK_N                                 # w_u8_buf u8
        + k_nt_pairs * k_slices * 2 * TILE_BYTES            # k_vnni (all N-tile pairs)
        + v_k_tiles * v_n_tiles * TILE_BYTES                # v_vnni
    )

def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                  max_prefill: Int = 512](num_workers: Int) -> Int:
    comptime q_cols = num_heads * head_dim
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim, max_prefill]()
    return max_prefill * q_cols * size_of[Float32]() + num_workers * pw


# ============================================================================
# Kernel
# ============================================================================

def attn_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int](
    ctx_addr: Int, group: Int, q_row_start: Int, q_row_end: Int,
    worker_scratch: Int, unused: Int,
):
    var ctx = UnsafePointer[AttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime hd_n_tiles = head_dim // TILE_N
    comptime k_slices = head_dim // K_STEP
    comptime K_PAIR_BYTES = k_slices * 2 * TILE_BYTES

    comptime HEAD_STRIDE = max_seq * head_dim

    var seq_len = ctx[].seq_len
    var pos = ctx[].pos
    var score_scale = ctx[].score_scale
    var vagg_scale = ctx[].vagg_scale
    var g = group
    var my_q_rows = q_row_end - q_row_start
    if my_q_rows <= 0:
        return
    var padded_q_rows = ((my_q_rows + M_STEP - 1) // M_STEP) * M_STEP

    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    # --- Scratch ---
    var off = 0
    var qi_buf = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += my_q_rows * head_dim
    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += my_q_rows * size_of[Float32]()
    var running_o = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * head_dim * size_of[Float32]()
    var running_m = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * size_of[Float32]()
    var running_l = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += padded_q_rows * size_of[Float32]()
    var score_i32 = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += Q_BATCH * BLOCK_N * size_of[Int32]()
    var w_u8_buf = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += Q_BATCH * BLOCK_N
    var k_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=worker_scratch + off)
    off += (BLOCK_N // TILE_N // 2) * K_PAIR_BYTES
    var v_vnni = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=worker_scratch + off)

    comptime score_stride_i32 = BLOCK_N * size_of[Int32]()

    var k_base = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=ctx[].k_data + g * HEAD_STRIDE)
    var v_base = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=ctx[].v_data + g * HEAD_STRIDE)

    # =================================================================
    # Q prep
    # =================================================================
    var q_quant_inv = ctx[].q_quant_inv
    var vq_inv = SIMD[DType.float32, width](q_quant_inv)

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
        var j = 0
        while j + width <= half:
            var x_lo = (q_row + j).load[width=width]().cast[DType.float32]()
            var x_hi = (q_row + half + j).load[width=width]().cast[DType.float32]()
            var cv = (cos_row + j).load[width=width]()
            var sv = (sin_row + j).load[width=width]()
            (qp + j).store(x_lo * cv - x_hi * sv)
            (qp + half + j).store(x_hi * cv + x_lo * sv)
            j += width
        fwht_block[DType.float32, head_dim](qp)
        var qi_row = qi_buf + local_idx * head_dim
        var q_sum_acc = SIMD[DType.int32, width](0)
        k = 0
        while k + width <= head_dim:
            var qi = quantize_i8((qp + k).load[width=width](), vq_inv)
            (qi_row + k).store(qi)
            q_sum_acc += qi.cast[DType.int32]()
            k += width
        qi_biases[local_idx] = Float32(q_sum_acc.reduce_add()) * Float32(128)

    # =================================================================
    # Init running state
    # =================================================================
    var vzero = SIMD[DType.float32, width](0)
    var vinf = SIMD[DType.float32, width](Float32(-1e30))
    var i = 0
    while i + width <= padded_q_rows * head_dim:
        (running_o + i).store(vzero)
        i += width
    i = 0
    while i + width <= padded_q_rows:
        (running_m + i).store(vinf)
        (running_l + i).store(vzero)
        i += width

    var max_context = pos + seq_len
    var vagg_arr = InlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()
    comptime vagg_stride = N_STEP * size_of[Int32]()

    # =================================================================
    # KV-outer, query-inner
    # =================================================================
    for block_start in range(0, max_context, BLOCK_N):
        var block_len = min(BLOCK_N, max_context - block_start)
        var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
        var chunk_k_iters = padded_chunk // K_STEP
        var n_tiles = (block_len + TILE_N - 1) // TILE_N

        # --- Pack K (all N-tile pairs) ---
        for nt in range(0, n_tiles, 2):
            var k_pair_base = k_vnni + (nt // 2) * K_PAIR_BYTES
            var n_off = block_start + nt * TILE_N
            for ki in range(k_slices):
                var k_off = ki * K_STEP
                # Tile 0 of pair
                var n0 = min(TILE_N, block_len - nt * TILE_N)
                var k_rows0 = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
                    fill=SIMD[DType.uint32, TILE_N](0))
                for col in range(n0):
                    k_rows0[col] = (k_base + (n_off + col) * head_dim + k_off).bitcast[Scalar[DType.uint32]]().load[width=TILE_N]()
                transpose_rows[DType.uint32, TILE_N](
                    k_rows0,
                    (k_pair_base + ki * 2 * TILE_BYTES).bitcast[Scalar[DType.uint32]](),
                    TILE_N)
                # Tile 1 of pair
                var n1 = max(0, min(TILE_N, block_len - (nt + 1) * TILE_N))
                var k_rows1 = InlineArray[SIMD[DType.uint32, TILE_N], TILE_N](
                    fill=SIMD[DType.uint32, TILE_N](0))
                for col in range(n1):
                    k_rows1[col] = (k_base + (n_off + TILE_N + col) * head_dim + k_off).bitcast[Scalar[DType.uint32]]().load[width=TILE_N]()
                transpose_rows[DType.uint32, TILE_N](
                    k_rows1,
                    (k_pair_base + ki * 2 * TILE_BYTES + TILE_BYTES).bitcast[Scalar[DType.uint32]](),
                    TILE_N)

        # --- Pack V (i8 → VNNI interleave) ---
        for nt in range(hd_n_tiles):
            for kt in range(chunk_k_iters):
                pack_v_tile_vnni(v_base, head_dim,
                    block_start + kt * K_STEP, nt * TILE_N,
                    min(K_STEP, block_len - kt * K_STEP),
                    v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

        # --- Query-inner ---
        for qb_start in range(0, my_q_rows, Q_BATCH):
            var qb_len = min(Q_BATCH, my_q_rows - qb_start)
            var qb_m_iters = (qb_len + M_STEP - 1) // M_STEP

            # Score GEMM
            for nt in range(0, n_tiles, 2):
                var k_nt_base = k_vnni + (nt // 2) * K_PAIR_BYTES
                for mi in range(qb_m_iters):
                    var m_off = qb_start + mi * M_STEP
                    tilezero[4](); tilezero[5](); tilezero[6](); tilezero[7]()
                    for ki in range(k_slices):
                        tileload[0](qi_buf + m_off * head_dim + ki * K_STEP, head_dim)
                        tileload[1](qi_buf + (m_off + TILE_M) * head_dim + ki * K_STEP, head_dim)
                        tileload[2](k_nt_base + ki * 2 * TILE_BYTES, TILE_N * VNNI_BLK)
                        tileload[3](k_nt_base + ki * 2 * TILE_BYTES + TILE_BYTES, TILE_N * VNNI_BLK)
                        tdpbsud[4, 0, 2](); tdpbsud[5, 0, 3]()
                        tdpbsud[6, 1, 2](); tdpbsud[7, 1, 3]()
                    var sb = score_i32 + (mi * M_STEP) * BLOCK_N + nt * TILE_N
                    tilestore[4](sb, score_stride_i32)
                    tilestore[5](sb + TILE_N, score_stride_i32)
                    tilestore[6](sb + TILE_M * BLOCK_N, score_stride_i32)
                    tilestore[7](sb + TILE_M * BLOCK_N + TILE_N, score_stride_i32)

            # Softmax
            for qi_local in range(qb_len):
                var qi_row = qb_start + qi_local
                var abs_row = q_row_start + qi_row
                var m_pos = abs_row // gqa_factor
                var causal_limit = min(pos + m_pos + 1 - block_start, block_len)
                var q_bi = qi_biases[qi_row]
                var q_sc = score_scale
                var si_row = score_i32 + qi_local * BLOCK_N

                # Max pass (i32 → dequant inline, no f32 store)
                var vmax = SIMD[DType.float32, width](Float32(-1e30))
                var t = 0
                while t + width <= causal_limit:
                    var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bi) * q_sc
                    vmax = max(vmax, dq)
                    t += width
                var scalar_max = Float32(-1e30)
                while t < causal_limit:
                    scalar_max = max(scalar_max, (Float32(si_row[t]) - q_bi) * q_sc)
                    t += 1
                var row_max = max(vmax.reduce_max(), scalar_max)

                # Correction
                var m_old = running_m[qi_row]
                var m_new = max(m_old, row_max)
                running_m[qi_row] = m_new
                if m_new > m_old:
                    var correction = exp_f32_fast[1](m_old - m_new)
                    running_l[qi_row] = running_l[qi_row] * correction
                    var ro = running_o + qi_row * head_dim
                    var d = 0
                    while d + width <= head_dim:
                        (ro + d).store((ro + d).load[width=width]() * correction)
                        d += width

                # Fused dequant + exp → u8 (re-reads i32, no score_buf)
                var l_acc = SIMD[DType.float32, width](0)
                var w_row = w_u8_buf + qi_local * padded_chunk
                t = 0
                while t + width <= causal_limit:
                    var dq = ((si_row + t).load[width=width]().cast[DType.float32]() - q_bi) * q_sc
                    var e = exp_f32_fast(dq - m_new)
                    (w_row + t).store(roundeven(e * Float32(255)).cast[DType.uint8]())
                    l_acc += e
                    t += width
                var l_contrib = l_acc.reduce_add()
                while t < causal_limit:
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
                while t < padded_chunk:
                    w_row[t] = UInt8(0)
                    t += 1
                running_l[qi_row] += l_contrib

            # V-agg
            for ns in range(head_dim // N_STEP):
                var d_off = ns * N_STEP
                for mi in range(qb_m_iters):
                    var m_off = qb_start + mi * M_STEP
                    tilezero[4](); tilezero[5](); tilezero[6](); tilezero[7]()
                    for kt in range(chunk_k_iters):
                        tileload[0](w_u8_buf + (mi * M_STEP) * padded_chunk + kt * K_STEP, padded_chunk)
                        tileload[1](w_u8_buf + (mi * M_STEP + TILE_M) * padded_chunk + kt * K_STEP, padded_chunk)
                        var nt_lo = d_off // TILE_N
                        tileload[2](v_vnni + (nt_lo * chunk_k_iters + kt) * TILE_BYTES, TILE_N * VNNI_BLK)
                        tileload[3](v_vnni + ((nt_lo + 1) * chunk_k_iters + kt) * TILE_BYTES, TILE_N * VNNI_BLK)
                        tdpbusd[4, 0, 2](); tdpbusd[5, 0, 3]()
                        tdpbusd[6, 1, 2](); tdpbusd[7, 1, 3]()
                    tilestore[4](vagg_i32, vagg_stride)
                    tilestore[5](vagg_i32 + TILE_N, vagg_stride)
                    tilestore[6](vagg_i32 + TILE_M * N_STEP, vagg_stride)
                    tilestore[7](vagg_i32 + TILE_M * N_STEP + TILE_N, vagg_stride)
                    # Accumulate i32→f32 (vagg_scale deferred to final normalize)
                    for r in range(min(M_STEP, qb_len - mi * M_STEP)):
                        var ro_row = running_o + (qb_start + mi * M_STEP + r) * head_dim + d_off
                        var c = 0
                        while c + width <= N_STEP:
                            var raw = (vagg_i32 + r * N_STEP + c).load[width=width]().cast[DType.float32]()
                            (ro_row + c).store((ro_row + c).load[width=width]() + raw)
                            c += width

    # =================================================================
    # Final normalize
    # =================================================================
    for local_idx in range(my_q_rows):
        var abs_row = q_row_start + local_idx
        var m = abs_row // gqa_factor
        var hi = abs_row % gqa_factor
        var h = g * gqa_factor + hi
        var final_scale = vagg_scale / running_l[local_idx]
        var ro = running_o + local_idx * head_dim
        var out = ctx[].row_f32 + m * q_cols + h * head_dim
        var d = 0
        while d + width <= head_dim:
            (out + d).store((ro + d).load[width=width]() * final_scale)
            d += width


# ============================================================================
# Public API
# ============================================================================

def prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
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
    """Async AMX prefill — zero per-token scales, uniform integer pipeline."""
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    comptime max_prefill = 512
    comptime pw = per_worker_bytes[num_heads, num_kv_heads, head_dim, max_prefill]()
    comptime WORKERS_OFF = max_prefill * q_cols * size_of[Float32]()
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    var ctx = AttnCtx(
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=q.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        k_cache.data_base,
        v_cache.data_base,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch),
        Float32(127.0) / q_layer_scale,
        q_layer_scale * k_layer_scale * inv_sqrt_hd / (Float32(127.0) * Float32(127.0)),
        v_layer_scale / (Float32(255.0) * Float32(127.0)),
        pos, q.seq_len,
    )
    var ctx_ptr = UnsafePointer(to=ctx)

    var workers_per_group = max(1, pool.capacity // num_kv_heads)
    var total_q_rows = q.seq_len * gqa_factor
    var rows_per_worker = (total_q_rows + workers_per_group - 1) // workers_per_group
    var total_jobs = num_kv_heads * workers_per_group

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
            pack[].arg4 = scratch + WORKERS_OFF + job_idx * pw
            pack[].arg5 = 0

    pool.dispatch(
        attn_prefill[num_heads, num_kv_heads, head_dim, max_seq],
        pool.args_base, total_jobs,
    )
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
