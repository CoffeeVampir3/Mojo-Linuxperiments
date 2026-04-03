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
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_2x2, prep_q_row, softmax_row
from experimental.amx import (
    TILE_M, TILE_K, TILE_N, VNNI_BLK, M_STEP, N_STEP, K_STEP, TILE_BYTES,
    TileConfig, make_224_i8_config,
    ldtilecfg, tilezero, tileload, tilestore, tile_dp,
)


comptime BLOCK_N = 512
comptime Q_BATCH = 32


# ============================================================================
# Per-worker scratch — typed pointers produced by caller
# ============================================================================

@fieldwise_init
struct WorkerScratch:
    """Per-worker dispatch config. All buffers are stack-allocated in the kernel."""
    var group: Int
    var q_row_start: Int
    var q_row_end: Int
    var pos: Int
    var seq_len: Int
    var q_quant_inv: Float32
    var score_scale: Float32
    var vagg_scale: Float32


# ============================================================================
# Scratch sizing
# ============================================================================

def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                  max_seq_len: Int, prefill_chunk: Int = 512](num_workers: Int) -> Int:
    """Output buffer + AttnCtx + worker configs. All working memory is stack-allocated."""
    comptime q_cols = num_heads * head_dim
    return (max_seq_len * q_cols * size_of[Float32]()
          + size_of[AttnCtx]()
          + num_workers * size_of[WorkerScratch]())


# ============================================================================
# Kernel
# ============================================================================

def attn_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                 max_seq: Int, prefill_chunk: Int = 512](
    ctx_addr: Int, ws_addr: Int, unused0: Int, unused1: Int,
    unused2: Int, unused3: Int,
):
    var ctx = UnsafePointer[AttnCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var ws = UnsafePointer[WorkerScratch, MutAnyOrigin](unsafe_from_address=ws_addr)
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = num_heads * head_dim
    comptime hd_n_tiles = head_dim // TILE_N
    comptime k_slices = head_dim // K_STEP
    comptime K_TILE_K_BYTES = k_slices * TILE_BYTES
    comptime MAX_TILES = (max_seq + TILE_N - 1) // TILE_N
    comptime K_HEAD_STRIDE = MAX_TILES * K_TILE_K_BYTES
    comptime V_HEAD_STRIDE = max_seq * head_dim
    comptime chunk_q_max = ((prefill_chunk * gqa_factor + M_STEP - 1) // M_STEP) * M_STEP
    comptime v_k_tiles = BLOCK_N // K_STEP

    var g = ws[].group
    var q_row_start = ws[].q_row_start
    var my_q_rows = ws[].q_row_end - q_row_start
    var pos = ws[].pos
    var seq_len = ws[].seq_len
    var score_scale = ws[].score_scale
    var vagg_scale = ws[].vagg_scale
    var q_quant_inv = ws[].q_quant_inv
    if my_q_rows <= 0:
        return

    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    var output = ctx[].output
    var k_vnni_head = ctx[].k_base + g * K_HEAD_STRIDE
    var v_base = ctx[].v_base + g * V_HEAD_STRIDE

    var qi_arr = InlineArray[Int8, chunk_q_max * head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Int8]()
    var bias_arr = InlineArray[Float32, chunk_q_max](uninitialized=True)
    var qi_biases = UnsafePointer(to=bias_arr).bitcast[Float32]()
    var rm_arr = InlineArray[Float32, chunk_q_max](uninitialized=True)
    var running_m = UnsafePointer(to=rm_arr).bitcast[Float32]()
    var rl_arr = InlineArray[Float32, chunk_q_max](uninitialized=True)
    var running_l = UnsafePointer(to=rl_arr).bitcast[Float32]()
    var off_arr = InlineArray[Int, chunk_q_max](uninitialized=True)
    var output_offsets = UnsafePointer(to=off_arr).bitcast[Int]()
    var score_arr = InlineArray[Int32, Q_BATCH * BLOCK_N](uninitialized=True)
    var score_i32 = UnsafePointer(to=score_arr).bitcast[Int32]()
    var wu8_arr = InlineArray[UInt8, Q_BATCH * BLOCK_N](uninitialized=True)
    var w_u8_buf = UnsafePointer(to=wu8_arr).bitcast[UInt8]()
    var vv_arr = InlineArray[Int8, hd_n_tiles * v_k_tiles * TILE_BYTES](uninitialized=True)
    var v_vnni = UnsafePointer(to=vv_arr).bitcast[Int8]()

    var vzero = SIMD[DType.float32, width](0)
    var vinf = SIMD[DType.float32, width](Float32(-1e30))
    var vagg_arr = InlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var vagg_i32 = UnsafePointer(to=vagg_arr).bitcast[Int32]()

    # =================================================================
    # Outer chunk loop — process Q rows in prefill_chunk-sized pieces
    # =================================================================
    for chunk_start in range(0, my_q_rows, chunk_q_max):
        var chunk_rows = min(chunk_q_max, my_q_rows - chunk_start)
        var padded_chunk_rows = ((chunk_rows + M_STEP - 1) // M_STEP) * M_STEP

        # --- Compute output offsets + zero output + Q prep ---
        for local_idx in range(chunk_rows):
            var abs_row = q_row_start + chunk_start + local_idx
            var m = abs_row // gqa_factor
            var hi = abs_row % gqa_factor
            var h = g * gqa_factor + hi
            var actual_pos = pos + m
            output_offsets[local_idx] = m * q_cols + h * head_dim
            # Zero output for this row
            var out_row = output + output_offsets[local_idx]
            var d = 0
            while d + width <= head_dim:
                (out_row + d).store(vzero)
                d += width
            # Q prep
            qi_biases[local_idx] = prep_q_row[head_dim](
                ctx[].q + m * q_cols + h * head_dim,
                ctx[].cos + actual_pos * half,
                ctx[].sin + actual_pos * half,
                q_quant_inv,
                qi_buf + local_idx * head_dim,
            )

        # --- Init running_m/l for this chunk ---
        var i = 0
        while i + width <= padded_chunk_rows:
            (running_m + i).store(vinf)
            (running_l + i).store(vzero)
            i += width

        # --- KV-outer, query-inner ---
        var max_context = pos + seq_len
        for block_start in range(0, max_context, BLOCK_N):
            var block_len = min(BLOCK_N, max_context - block_start)
            var padded_chunk = ((block_len + K_STEP - 1) // K_STEP) * K_STEP
            var chunk_k_iters = padded_chunk // K_STEP
            var n_tiles = (block_len + TILE_N - 1) // TILE_N

            var block_tile_base = (block_start // TILE_N) * K_TILE_K_BYTES

            # Pack V
            for nt in range(hd_n_tiles):
                for kt in range(chunk_k_iters):
                    pack_v_tile_vnni(v_base, head_dim,
                        block_start + kt * K_STEP, nt * TILE_N,
                        min(K_STEP, block_len - kt * K_STEP),
                        v_vnni + (nt * chunk_k_iters + kt) * TILE_BYTES)

            # Query-inner
            for qb_start in range(0, chunk_rows, Q_BATCH):
                var qb_len = min(Q_BATCH, chunk_rows - qb_start)
                var qb_m_iters = (qb_len + M_STEP - 1) // M_STEP

                # Score GEMM
                for nt in range(0, n_tiles, 2):
                    var b0 = k_vnni_head + block_tile_base + nt * K_TILE_K_BYTES
                    var b1 = k_vnni_head + block_tile_base + (nt + 1) * K_TILE_K_BYTES
                    for mi in range(qb_m_iters):
                        var m_off = qb_start + mi * M_STEP
                        amx_gemm_2x2[DType.int8, DType.uint8, BLOCK_N](
                            qi_buf + m_off * head_dim,
                            qi_buf + (m_off + TILE_M) * head_dim,
                            head_dim,
                            b0, b1,
                            TILE_BYTES, k_slices,
                            score_i32 + (mi * M_STEP) * BLOCK_N + nt * TILE_N,
                        )

                # Softmax — accumulates directly into output
                for qi_local in range(qb_len):
                    var qi_row = qb_start + qi_local
                    var abs_row = q_row_start + chunk_start + qi_row
                    var m_pos = abs_row // gqa_factor
                    var causal_limit = min(pos + m_pos + 1 - block_start, block_len)
                    softmax_row[head_dim](
                        score_i32 + qi_local * BLOCK_N,
                        qi_biases[qi_row], score_scale,
                        causal_limit, padded_chunk,
                        running_m + qi_row, running_l + qi_row,
                        output + output_offsets[qi_row],
                        w_u8_buf + qi_local * padded_chunk,
                    )

                # V-agg — accumulates directly into output
                for ns in range(head_dim // N_STEP):
                    var d_off = ns * N_STEP
                    var nt_lo = d_off // TILE_N
                    for mi in range(qb_m_iters):
                        amx_gemm_2x2[DType.uint8, DType.int8, N_STEP](
                            w_u8_buf + (mi * M_STEP) * padded_chunk,
                            w_u8_buf + (mi * M_STEP + TILE_M) * padded_chunk,
                            padded_chunk,
                            v_vnni + nt_lo * chunk_k_iters * TILE_BYTES,
                            v_vnni + (nt_lo + 1) * chunk_k_iters * TILE_BYTES,
                            TILE_BYTES, chunk_k_iters,
                            vagg_i32,
                        )
                        for r in range(min(M_STEP, qb_len - mi * M_STEP)):
                            var ro_row = output + output_offsets[qb_start + mi * M_STEP + r] + d_off
                            var c = 0
                            while c + width <= N_STEP:
                                var raw = (vagg_i32 + r * N_STEP + c).load[width=width]().cast[DType.float32]()
                                (ro_row + c).store((ro_row + c).load[width=width]() + raw)
                                c += width

        # --- Final normalize (in-place on output) ---
        for local_idx in range(chunk_rows):
            var final_scale = vagg_scale / running_l[local_idx]
            var out_row = output + output_offsets[local_idx]
            var d = 0
            while d + width <= head_dim:
                (out_row + d).store((out_row + d).load[width=width]() * final_scale)
                d += width


# ============================================================================
# Public API
# ============================================================================

def prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
    QT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped,
    P: BurstThreadPool, prefill_chunk: Int = 512](
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
    mut pool: P,
) -> PoolFence[P]:
    """Async AMX prefill — chunked Q processing, zero per-token scales.

    seq_len can exceed prefill_chunk: the kernel internally processes Q rows
    in chunks, reusing scratch buffers each iteration. Single dispatch.
    """
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))

    # Derived per-layer constants
    var q_quant_inv = Float32(127.0) / q_layer_scale
    var score_scale = q_layer_scale * k_layer_scale * inv_sqrt_hd / (Float32(127.0) * Float32(127.0))
    var vagg_scale = v_layer_scale / (Float32(255.0) * Float32(127.0))

    # Write AttnCtx into scratch (after output buffer, before workers)
    var output_bytes = q.seq_len * q_cols * size_of[Float32]()
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

    var workers_per_group = max(1, pool.get_capacity() // num_kv_heads)
    var total_q_rows = q.seq_len * gqa_factor
    var rows_per_worker = (total_q_rows + workers_per_group - 1) // workers_per_group
    var total_jobs = num_kv_heads * workers_per_group

    for g in range(num_kv_heads):
        for w in range(workers_per_group):
            var job_idx = g * workers_per_group + w
            var q_start = w * rows_per_worker
            var q_end = min(q_start + rows_per_worker, total_q_rows)

            var ws_base = scratch + workers_off + job_idx * size_of[WorkerScratch]()
            var ws = UnsafePointer[WorkerScratch, MutAnyOrigin](
                unsafe_from_address=ws_base)
            ws[].group = g
            ws[].q_row_start = q_start
            ws[].q_row_end = q_end
            ws[].pos = pos
            ws[].seq_len = q.seq_len
            ws[].q_quant_inv = q_quant_inv
            ws[].score_scale = score_scale
            ws[].vagg_scale = vagg_scale

            var pack = pool.get_args_base() + job_idx
            pack[].arg0 = Int(ctx_ptr)
            pack[].arg1 = ws_base
            pack[].arg2 = 0
            pack[].arg3 = 0
            pack[].arg4 = 0
            pack[].arg5 = 0

    pool.dispatch(
        attn_prefill[num_heads, num_kv_heads, head_dim, max_seq, prefill_chunk],
        pool.get_args_base(), total_jobs,
    )
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
