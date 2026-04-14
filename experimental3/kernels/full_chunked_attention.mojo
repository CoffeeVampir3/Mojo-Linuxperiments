"""Context-parallel chunked attention — multi-head data reuse for full attention.

Splits the KV cache into chunks distributed across BurstPool workers. Each
worker scores all Q heads against its chunk of position groups, reusing K and V
cache data across heads via L1 locality. After join, partial online-softmax
states are merged with log-sum-exp correction and the output is quantized to i8
for the O projection.

Data fusion:
  - K/V position group data loaded once per pg, scored against all Q heads
    while hot in L1 (heads_per_group × reuse, typically 8×).
  - Merge and absmax-quantize fused: merged f32 is normalized in registers
    and quantized without writing an intermediate f32 buffer to memory.
  - Per-position V scale folded into attention-weight u8 quantization
    (inherited from v_agg_group — zero extra cost in the inner loop).

Dispatch wrappers at the bottom: full_attn_prep_dispatch, chunked_attn_dispatch.

Usage:
  1. Dispatch full_attn_prep_kernel (1 job) — KV cache write + Q prep.
  2. Dispatch chunked_attn_kernel to pool — one worker per chunk, all Q heads.
  3. Join, then call merge_and_quantize on the collected partial states.
"""

from std.memory import UnsafePointer
from std.collections import InlineArray

from kernels.kernel_ops import PoolFence
from threading.threading_traits import BurstThreadPool
from std.sys.info import simd_width_of

from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.sliding_attention import score_group, v_agg_group
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.rope_and_kv_cache_write import (
    write_k_head_normed, write_v_head_normed,
)
from experimental3.helpers import prep_q_row_normed_partial
from simd_math import exp_f32


comptime WIDTH = CACHE_WIDTH


# ============================================================================
# Partial state layout
# ============================================================================
#
# Per Q head per chunk: [max: f32, sum: f32, v_acc: f32 × head_dim]
# Stride per head   = (2 + head_dim) f32s
# Stride per chunk  = heads_per_group × stride_per_head


@always_inline
def partial_head_stride[head_dim: Int]() -> Int:
    """F32 element stride between consecutive Q heads in a partial buffer."""
    return 2 + head_dim


@always_inline
def partial_chunk_stride[head_dim: Int, heads_per_group: Int]() -> Int:
    """F32 element stride between consecutive chunks in the partial buffer."""
    return heads_per_group * partial_head_stride[head_dim]()


@always_inline
def partial_chunk_bytes[head_dim: Int, heads_per_group: Int]() -> Int:
    """Byte size of one chunk's partial output (all Q heads)."""
    return partial_chunk_stride[head_dim, heads_per_group]() * 4


# ============================================================================
# Worker args
# ============================================================================


@fieldwise_init
struct ChunkedAttnArgs(Copyable, ImplicitlyCopyable):
    """Per-worker arguments for chunked attention scoring.

    All pointer fields are raw Int addresses.  The partial_out buffer must have
    room for heads_per_group × (2 + head_dim) f32 values.
    """
    var q_i8_base: Int
    var qi_biases_base: Int
    var q_scales_base: Int
    var cache_base: Int
    var kv_head: Int
    var start_pg: Int
    var end_pg: Int
    var partial_out: Int
    var context_len: Int

    def __init__(out self):
        self.q_i8_base = 0
        self.qi_biases_base = 0
        self.q_scales_base = 0
        self.cache_base = 0
        self.kv_head = 0
        self.start_pg = 0
        self.end_pg = 0
        self.partial_out = 0
        self.context_len = 0


# ============================================================================
# Prep kernel — KV cache write + Q prep (1 job, runs before chunk dispatch)
# ============================================================================


@fieldwise_init
struct FullAttnPrepArgs(Copyable, ImplicitlyCopyable):
    """Args for the single-job prep kernel: KV cache write + all-head Q prep."""
    var q_bf16_base: Int
    var k_bf16_ptr: Int
    var q_norm_ptr: Int
    var k_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var cache_pos: Int
    var kv_head: Int
    var eps: Float32
    var q_i8_out: Int
    var qi_biases_out: Int
    var q_scales_out: Int

    def __init__(out self):
        self.q_bf16_base = 0
        self.k_bf16_ptr = 0
        self.q_norm_ptr = 0
        self.k_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.cache_base = 0
        self.cache_pos = 0
        self.kv_head = 0
        self.eps = Float32(0)
        self.q_i8_out = 0
        self.qi_biases_out = 0
        self.q_scales_out = 0


def full_attn_prep_kernel[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
](args: FullAttnPrepArgs):
    """KV cache write + Q prep for all heads.  Dispatched as 1 job before chunks.

    K=V shared: both K and V cache entries are written from the K projection.
    Each Q head is normed, partial-RoPE'd, FWHT'd, and quantized to i8.
    """
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](
        args.cache_base)
    var cos = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.cos_ptr)
    var sin = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.sin_ptr)

    var work_arr = InlineArray[Float32, head_dim](uninitialized=True)
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](uninitialized=True)
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    write_k_head_normed[head_dim, rope_dims](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.k_bf16_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.k_norm_ptr),
        cos, sin, work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    write_v_head_normed[head_dim](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.k_bf16_ptr),
        work, qi_buf,
        cache, args.cache_pos, args.kv_head, args.eps)

    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.qi_biases_out)
    var q_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.q_scales_out)

    for qh in range(heads_per_group):
        var q_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=args.q_bf16_base + qh * head_dim * 2)
        var q_i8_dst = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=args.q_i8_out + qh * head_dim)
        var result = prep_q_row_normed_partial[head_dim, rope_dims](
            q_bf16.bitcast[BFloat16](),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=args.q_norm_ptr),
            cos, sin,
            q_i8_dst.bitcast[Int8](), args.eps)
        qi_biases[qh] = result[0]
        q_scales[qh] = result[1]


# ============================================================================
# Chunk kernel — one context chunk, all Q heads
# ============================================================================


def chunked_attn_kernel[
    head_dim: Int, max_seq: Int, num_kv_heads: Int, num_q_heads: Int,
    heads_per_group: Int,
](args: ChunkedAttnArgs):
    """Score one KV-cache chunk against all Q heads.

    Iterates position groups in [start_pg, end_pg).  For each group every Q
    head is scored against the same K data and V is aggregated, keeping K/V
    data hot in L1 across heads (heads_per_group × reuse).

    Output: partial online-softmax state per Q head written to partial_out.
    """
    var cache = Gemma4KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](
        args.cache_base)
    var k_scales = cache.k_scale_ptr(args.kv_head)
    var v_scales = cache.v_scale_ptr(args.kv_head)

    var qi_biases = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.qi_biases_base)
    var q_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.q_scales_base)

    # Per-head running softmax state.
    var running_max = InlineArray[Float32, heads_per_group](fill=Float32(-1e30))
    var running_sum = InlineArray[Float32, heads_per_group](fill=Float32(0))

    # V accumulator: heads_per_group × head_dim, zero-initialised.
    comptime V_ACC_TOTAL = heads_per_group * head_dim
    var v_acc_storage = InlineArray[Float32, V_ACC_TOTAL](fill=Float32(0))
    var v_acc_base = UnsafePointer(to=v_acc_storage).bitcast[Float32]()

    # Scratch for scores (overwritten by score_group every iteration).
    var scores_arr = InlineArray[Float32, WIDTH](uninitialized=True)
    var scores = UnsafePointer(to=scores_arr).bitcast[Float32]()

    # Pre-compute q_factor per head (matches single_pass_attention's division).
    comptime Q_DENOM = Float32(127) * Float32(127)
    var q_factors = InlineArray[Float32, heads_per_group](uninitialized=True)
    for qh in range(heads_per_group):
        q_factors[qh] = q_scales[qh] / Q_DENOM

    # -----------------------------------------------------------------
    # Main loop — position groups
    # -----------------------------------------------------------------
    for pg in range(args.start_pg, args.end_pg):
        var k_pg = cache.k_pg_ptr(args.kv_head, pg)
        var v_pg = cache.v_pg_ptr(args.kv_head, pg)
        var k_sc_pg = k_scales + pg * WIDTH
        var v_sc_pg = v_scales + pg * WIDTH
        var group_start = pg * WIDTH

        # All Q heads against this position group.  K data (4 KB at
        # head_dim=512) is pulled into L1 by the first head and stays
        # hot for the remaining heads.  V data likewise.
        for qh in range(heads_per_group):
            var q_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
                unsafe_from_address=args.q_i8_base + qh * head_dim)

            score_group[head_dim](
                q_i8, k_pg, qi_biases[qh], q_factors[qh], k_sc_pg, scores)

            # Mask positions past context_len in the last group.
            for p in range(WIDTH):
                if group_start + p >= args.context_len:
                    scores[p] = Float32(-1e30)

            var scores_vec = scores.load[width=WIDTH]()
            var group_max = scores_vec.reduce_max()
            var new_max = max(running_max[qh], group_max)

            var v_acc = v_acc_base + qh * head_dim

            # Rescale previous accumulator when the running max shifts.
            if running_sum[qh] > 0:
                var rescale = Float32(exp_f32[1](running_max[qh] - new_max))
                var d = 0
                while d + WIDTH <= head_dim:
                    (v_acc + d).store(
                        (v_acc + d).load[width=WIDTH]() * rescale)
                    d += WIDTH
                running_sum[qh] *= rescale

            running_max[qh] = new_max
            var exp_scores = exp_f32[WIDTH](scores_vec - new_max)

            for p in range(WIDTH):
                if group_start + p >= args.context_len:
                    exp_scores[p] = Float32(0)

            running_sum[qh] += exp_scores.reduce_add()
            v_agg_group[head_dim](exp_scores, v_sc_pg, v_pg, v_acc)

    # -----------------------------------------------------------------
    # Write partial states
    # -----------------------------------------------------------------
    var out = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=args.partial_out)
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()

    for qh in range(heads_per_group):
        var dst = out + qh * HEAD_STRIDE
        dst[] = running_max[qh]
        (dst + 1)[] = running_sum[qh]
        var src = v_acc_base + qh * head_dim
        var d = 0
        while d + WIDTH <= head_dim:
            (dst + 2 + d).store((src + d).load[width=WIDTH]())
            d += WIDTH


# ============================================================================
# Merge + quantize
# ============================================================================


def merge_and_quantize[head_dim: Int, heads_per_group: Int](
    partial_base: UnsafePointer[Float32, MutAnyOrigin],
    num_chunks: Int,
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    head_scales: UnsafePointer[Float32, MutAnyOrigin],
):
    """Merge partial softmax states from all chunks and quantize to i8.

    For each Q head: find global max across chunks, rescale and accumulate
    partial v_acc values with log-sum-exp correction, normalize by the total
    softmax denominator, then absmax-quantize to i8.  The merged f32 values
    live in a stack-local InlineArray (head_dim × 4 bytes, in L1) so the
    normalize → quantize path has no extra memory traffic.
    """
    comptime HEAD_STRIDE = partial_head_stride[head_dim]()
    comptime CHUNK_STRIDE = partial_chunk_stride[head_dim, heads_per_group]()
    comptime width = simd_width_of[DType.float32]()

    for qh in range(heads_per_group):
        # --- Pass 1: global max ---
        var global_max = Float32(-1e30)
        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            global_max = max(global_max, p[])

        # --- Pass 2: rescale + accumulate ---
        var total_sum = Float32(0)
        var merged = InlineArray[Float32, head_dim](fill=Float32(0))
        var merged_ptr = UnsafePointer(to=merged).bitcast[Float32]()

        for c in range(num_chunks):
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var chunk_max = p[]
            var chunk_sum = (p + 1)[]
            var v_acc = p + 2

            if chunk_sum <= 0:
                continue

            var rescale = Float32(exp_f32[1](chunk_max - global_max))
            total_sum += chunk_sum * rescale

            var d = 0
            while d + width <= head_dim:
                var acc = (merged_ptr + d).load[width=width]()
                var cv = (v_acc + d).load[width=width]()
                (merged_ptr + d).store(cv.fma(rescale, acc))
                d += width

        # --- Normalize: v / (127 × total_sum) ---
        var inv_sum = Float32(1.0) / (Float32(127) * total_sum)
        var d = 0
        while d + width <= head_dim:
            (merged_ptr + d).store(
                (merged_ptr + d).load[width=width]() * inv_sum)
            d += width

        # --- Quantize to i8 ---
        head_scales[qh] = absmax_quantize_i8[head_dim](
            merged_ptr, qi_out + qh * head_dim)


# ============================================================================
# Dispatch wrappers
# ============================================================================


def full_attn_prep_dispatch[
    head_dim: Int, rope_dims: Int, heads_per_group: Int,
    max_seq: Int, num_kv_heads: Int, num_q_heads: Int, P: BurstThreadPool,
](
    q_bf16_base: Int, k_bf16_ptr: Int,
    q_norm_ptr: Int, k_norm_ptr: Int,
    cos_ptr: Int, sin_ptr: Int,
    cache_base: Int, cache_pos: Int, kv_head: Int,
    eps: Float32,
    q_i8_out: Int, qi_biases_out: Int, q_scales_out: Int,
    mut pool: P,
) -> PoolFence[P]:
    var args = FullAttnPrepArgs(
        q_bf16_base=q_bf16_base, k_bf16_ptr=k_bf16_ptr,
        q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
        cos_ptr=cos_ptr, sin_ptr=sin_ptr,
        cache_base=cache_base, cache_pos=cache_pos, kv_head=kv_head,
        eps=eps,
        q_i8_out=q_i8_out, qi_biases_out=qi_biases_out, q_scales_out=q_scales_out)
    pool.dispatch[FullAttnPrepArgs,
        full_attn_prep_kernel[head_dim, rope_dims, heads_per_group,
            max_seq, num_kv_heads, num_q_heads]](
        UnsafePointer(to=args), 1)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


comptime MAX_CHUNKS = 32


def chunked_attn_dispatch[
    head_dim: Int, max_seq: Int,
    num_kv_heads: Int, num_q_heads: Int, heads_per_group: Int,
    P: BurstThreadPool,
](
    q_i8_base: Int, qi_biases_base: Int, q_scales_base: Int,
    cache_base: Int, kv_head: Int,
    context_len: Int, pool_capacity: Int,
    partial_out_base: Int,
    mut pool: P,
) -> PoolFence[P]:
    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
    var num_chunks = min(pool_capacity, MAX_CHUNKS)
    if num_chunks > num_pg:
        num_chunks = num_pg
    var pgs_per_chunk = (num_pg + num_chunks - 1) // num_chunks
    comptime CHUNK_F32_STRIDE = heads_per_group * (2 + head_dim)
    var chunk_args = InlineArray[ChunkedAttnArgs, MAX_CHUNKS](fill=ChunkedAttnArgs())
    for c in range(num_chunks):
        var start = c * pgs_per_chunk
        var end = min((c + 1) * pgs_per_chunk, num_pg)
        chunk_args[c] = ChunkedAttnArgs(
            q_i8_base=q_i8_base,
            qi_biases_base=qi_biases_base,
            q_scales_base=q_scales_base,
            cache_base=cache_base,
            kv_head=kv_head,
            start_pg=start,
            end_pg=end,
            partial_out=partial_out_base + c * CHUNK_F32_STRIDE * 4,
            context_len=context_len)
    pool.dispatch[ChunkedAttnArgs,
        chunked_attn_kernel[head_dim, max_seq, num_kv_heads, num_q_heads, heads_per_group]](
        UnsafePointer(to=chunk_args[0]), num_chunks)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))
