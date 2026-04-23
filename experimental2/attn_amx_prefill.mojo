"""AMX prefill attention — per-head dynamic Q/K scales, fixed V scale.

K cache is u8 (i8 XOR 0x80) for tdpbsud. V cache is i8 directly for
tdpbusd. Q and K use per-head dynamic absmax scales (stored in cache).
V uses a corrected per-layer fixed scale.

Score dequant: (raw - b_q) * S_Q[head] / (127^2 * sqrt(d_k)) * S_K[head, pos]
V-agg dequant: fixed per-layer constant from S_V.

Parallelism: Q rows split across workers per KV group. Each worker
redundantly packs K/V (all cores active, no barriers).
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Bound, DynView,
)
from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven
from experimental2.kv_cache import KVCache
from experimental2.helpers import AttnCtx, pack_v_tile_vnni, amx_gemm_2x2, prep_q_row, softmax_row
from experimental3.amx import (
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
    var vagg_scale: Float32


# ============================================================================
# Scratch sizing
# ============================================================================

def scratch_bytes[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                  max_seq_len: Int, prefill_chunk: Int = 512](num_workers: Int) -> Int:
    """f32 accumulation buffer + i8 output + AttnCtx + worker configs."""
    comptime q_cols = num_heads * head_dim
    return (max_seq_len * q_cols * size_of[Float32]()      # f32 V-agg accumulation
          + max_seq_len * q_cols * size_of[Scalar[DType.int8]]()  # i8 quantized output
          + size_of[AttnCtx]()
          + num_workers * size_of[WorkerScratch]())


# ============================================================================
# Kernel args + kernel
# ============================================================================

@fieldwise_init
struct PrefillKernelArgs(Copyable, ImplicitlyCopyable):
    var ctx_addr: Int
    var ws_addr: Int


def attn_prefill[num_heads: Int, num_kv_heads: Int, head_dim: Int,
                 max_seq: Int, prefill_chunk: Int = 512](
    args: PrefillKernelArgs,
):
    var ctx = UnsafePointer[AttnCtx, MutAnyOrigin](unsafe_from_address=args.ctx_addr)
    var ws = UnsafePointer[WorkerScratch, MutAnyOrigin](unsafe_from_address=args.ws_addr)
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
    var vagg_scale = ws[].vagg_scale
    var inv_sqrt_hd = ctx[].inv_sqrt_hd
    var inv_127sq = Float32(1.0) / (Float32(127.0) * Float32(127.0))
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
        # Per-row Q score partial: S_Q[head, pos] / (127^2 * sqrt(d_k))
        var q_partials_arr = InlineArray[Float32, chunk_q_max](uninitialized=True)
        var q_partials = UnsafePointer(to=q_partials_arr).bitcast[Float32]()

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
            # Q prep — returns (bias, scale)
            var qr = prep_q_row[head_dim](
                ctx[].q + m * ctx[].q_stride + h * head_dim,
                ctx[].cos + actual_pos * half,
                ctx[].sin + actual_pos * half,
                qi_buf + local_idx * head_dim,
            )
            qi_biases[local_idx] = qr[0]
            var q_scale = qr[1]
            # Write Q scale to cache
            (ctx[].q_scale_base + h * ctx[].max_seq + actual_pos)[] = q_scale
            # Precompute Q-partial for scoring: S_Q / (127^2 * sqrt(d_k))
            q_partials[local_idx] = q_scale * inv_127sq * inv_sqrt_hd

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
                # K scales for this KV group
                var k_scale_head = ctx[].k_scale_base + g * ctx[].max_seq
                for qi_local in range(qb_len):
                    var qi_row = qb_start + qi_local
                    var abs_row = q_row_start + chunk_start + qi_row
                    var m_pos = abs_row // gqa_factor
                    var causal_limit = min(pos + m_pos + 1 - block_start, block_len)
                    softmax_row[head_dim](
                        score_i32 + qi_local * BLOCK_N,
                        qi_biases[qi_row], q_partials[qi_row],
                        k_scale_head, block_start,
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

        # --- Final normalize + quantize to i8 ---
        comptime f32_lo = SIMD[DType.float32, width](-128.0)
        comptime f32_hi = SIMD[DType.float32, width](127.0)
        var qi_out = ctx[].qi_output
        var vqi_scale = SIMD[DType.float32, width](ctx[].qi_scale)
        for local_idx in range(chunk_rows):
            var final_scale = vagg_scale / running_l[local_idx]
            var accum_row = output + output_offsets[local_idx]
            var qi_row = qi_out + output_offsets[local_idx]
            var d = 0
            while d + width <= head_dim:
                var v = (accum_row + d).load[width=width]() * final_scale
                (qi_row + d).store(min(max(roundeven(v * vqi_scale), f32_lo), f32_hi).cast[DType.int8]())
                d += width


# ============================================================================
# Public API
# ============================================================================

def prefill[
    P: BurstThreadPool, origin: MutOrigin, //,
    num_heads: Int, num_kv_heads: Int, head_dim: Int, max_seq: Int,
    num_q_heads: Int,
    QT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped,
    prefill_chunk: Int = 512,
](
    q: DynView[QT],
    q_stride: Int,
    cache: KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    scratch: Int,
    pos: Int,
    v_layer_scale: Float32,
    ref [origin] pool: P,
) -> PoolFence[P, origin]:
    """Async AMX prefill — per-head dynamic Q/K scales, fixed V scale.

    Q/K scales are computed per-head at quantization time and stored in the
    cache. V uses v_layer_scale (corrected fixed scale).
    """
    comptime gqa_factor = num_heads // num_kv_heads
    comptime q_cols = QT.COLS
    var inv_sqrt_hd = Float32(1.0 / Float64(sqrt[DType.float32, 1](Float32(head_dim))))
    var vagg_scale = v_layer_scale / (Float32(255.0) * Float32(127.0))

    # Scratch layout: [f32 accum | i8 output | AttnCtx | worker configs]
    var accum_bytes = q.seq_len * q_cols * size_of[Float32]()
    var qi_bytes = q.seq_len * q_cols * size_of[Scalar[DType.int8]]()
    var qi_inv = Float32(127) / v_layer_scale
    var ctx_ptr = UnsafePointer[AttnCtx, MutAnyOrigin](
        unsafe_from_address=scratch + accum_bytes + qi_bytes)
    ctx_ptr[] = AttnCtx(
        UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=q.ptr),
        q_stride,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cache.k_base),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=cache.v_base),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scratch),
        UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=scratch + accum_bytes),
        qi_inv,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cache.k_scale_base),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cache.q_scale_base),
        max_seq,
        inv_sqrt_hd,
    )
    var workers_off = accum_bytes + qi_bytes + size_of[AttnCtx]()

    comptime MAX_POOL_CAPACITY = 128
    var workers_per_group = max(1, pool.get_capacity() // num_kv_heads)
    var total_q_rows = q.seq_len * gqa_factor
    var rows_per_worker = (total_q_rows + workers_per_group - 1) // workers_per_group
    var total_jobs = num_kv_heads * workers_per_group

    var jobs = InlineArray[PrefillKernelArgs, MAX_POOL_CAPACITY](
        fill=PrefillKernelArgs(0, 0))

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
            ws[].vagg_scale = vagg_scale

            jobs[job_idx] = PrefillKernelArgs(Int(ctx_ptr), ws_base)

    pool.dispatch[PrefillKernelArgs, attn_prefill[num_heads, num_kv_heads, head_dim, max_seq, prefill_chunk]](
        UnsafePointer(to=jobs[0]), total_jobs)
    return PoolFence[P, origin].over(pool)
