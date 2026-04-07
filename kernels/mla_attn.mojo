"""Multi-head Latent Attention kernels for DeepSeek-V2.

Weight-absorbed MLA fuses four steps per head into one kernel:
  1. Query transform: q_hat = W_UK[h]^T @ q_nope[h]
  2. Dual-path scoring: content (q_hat^T @ c_KV) + RoPE (q_pe^T @ k_R)
  3. Online-softmax latent aggregation over compressed KV cache
  4. Output projection: o[h] = W_UV[h] @ o_latent

All heads share a single c_KV + k_R cache per position — the core MLA
bandwidth advantage over standard MHA/GQA.

Also provides mla_kv_cache_write: fused split + RMSNorm + dual cache write
for the kv_a projection output.
"""

from std.math import sqrt
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import Encoding, Shaped, Bound, DynView, CacheView
from kernels.kernel_ops import PoolFence
from simd_math import exp_f32


# =============================================================================
# MLA attention kernel — BurstPool dispatch target
# =============================================================================


def mla_kernel[
    num_heads: Int,
    nope_dim: Int,
    rope_dim: Int,
    kv_lora_rank: Int,
    v_head_dim: Int,
    softmax_scale: Float64,
](
    qp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ckv_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    kr_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    kvb_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    packed: Int,
):
    """Weight-absorbed MLA attention.

    qp:    Q buffer     [seq, num_heads * (nope_dim + rope_dim)]
    dp:    output       [seq, num_heads * v_head_dim]
    ckv_p: c_KV cache   [capacity, kv_lora_rank]
    kr_p:  k_R cache    [capacity, rope_dim]
    kvb_p: kv_b_proj    [num_heads * (nope_dim + v_head_dim), kv_lora_rank]
    packed: start_head[63:56] | end_head[55:48] | pos[47:24] | seq_len[23:0]
    """
    comptime width = simd_width_of[DType.float32]()
    comptime q_head_dim = nope_dim + rope_dim
    comptime q_stride = num_heads * q_head_dim
    comptime o_stride = num_heads * v_head_dim
    comptime kvb_head_dim = nope_dim + v_head_dim
    comptime scale = Float32(softmax_scale)

    var start_head = (packed >> 56) & 0xFF
    var end_head = (packed >> 48) & 0xFF
    var pos = (packed >> 24) & 0xFFFFFF
    var seq_len = packed & 0xFFFFFF

    for m in range(seq_len):
        var causal_len = pos + m + 1
        var q_row = qp + m * q_stride
        var d_row = dp + m * o_stride

        for h in range(start_head, end_head):
            var q_nope_base = q_row + h * q_head_dim
            var q_pe_base = q_nope_base + nope_dim
            var wuk = kvb_p + h * kvb_head_dim * kv_lora_rank
            var wuv = kvb_p + (h * kvb_head_dim + nope_dim) * kv_lora_rank

            # ── Step 1: q_hat = W_UK^T @ q_nope ───────────────────
            #
            # Outer-product accumulation over nope_dim input elements.
            # Each iteration broadcasts one q_nope scalar and FMAs with
            # the corresponding 512-element W_UK row.
            var q_hat = InlineArray[Float32, kv_lora_rank](fill=Float32(0))
            var qhp = UnsafePointer(to=q_hat[0])

            for k in range(nope_dim):
                var s = SIMD[DType.float32, width](
                    (q_nope_base + k).load().cast[DType.float32]()
                )
                var w_row = wuk + k * kv_lora_rank
                for c in range(0, kv_lora_rank, width):
                    var acc = (qhp + c).load[width=width]()
                    var w = (w_row + c).load[width=width]().cast[DType.float32]()
                    (qhp + c).store(w.fma(s, acc))

            # ── Load q_pe to f32 stack buffer ──────────────────────
            var q_pe = InlineArray[Float32, rope_dim](fill=Float32(0))
            var qpp = UnsafePointer(to=q_pe[0])
            for c in range(0, rope_dim, width):
                (qpp + c).store(
                    (q_pe_base + c).load[width=width]().cast[DType.float32]()
                )

            # ── Steps 2-3: Dual-path score + online softmax ───────
            #
            # o_latent accumulates the weighted sum of c_KV latents.
            # Online softmax avoids materializing a score buffer.
            var o_latent = InlineArray[Float32, kv_lora_rank](fill=Float32(0))
            var olp = UnsafePointer(to=o_latent[0])
            var running_max = Float32(-1e30)
            var running_sum = Float32(0)

            for t in range(causal_len):
                var ckv_row = ckv_p + t * kv_lora_rank
                var kr_row = kr_p + t * rope_dim

                # Content score: q_hat^T @ c_KV[t]  (kv_lora_rank-dim dot)
                var content = SIMD[DType.float32, width](0)
                for c in range(0, kv_lora_rank, width):
                    var q = (qhp + c).load[width=width]()
                    var cv = (ckv_row + c).load[width=width]().cast[DType.float32]()
                    content = q.fma(cv, content)

                # RoPE score: q_pe^T @ k_R[t]  (rope_dim-dim dot)
                var rp = SIMD[DType.float32, width](0)
                for c in range(0, rope_dim, width):
                    var q = (qpp + c).load[width=width]()
                    var kr = (kr_row + c).load[width=width]().cast[DType.float32]()
                    rp = q.fma(kr, rp)

                var score = (content.reduce_add() + rp.reduce_add()) * scale

                # Online softmax
                var new_max = max(running_max, score)
                var correction = Float32(
                    exp_f32[1](SIMD[DType.float32, 1](running_max - new_max))
                )
                var w = Float32(
                    exp_f32[1](SIMD[DType.float32, 1](score - new_max))
                )

                # Rescale prior accumulator + add weighted latent.
                # c_KV data was just read for scoring so is hot in L1.
                var corr_v = SIMD[DType.float32, width](correction)
                var wv = SIMD[DType.float32, width](w)
                for c in range(0, kv_lora_rank, width):
                    var prior = (olp + c).load[width=width]()
                    var cv = (ckv_row + c).load[width=width]().cast[DType.float32]()
                    (olp + c).store(prior * corr_v + cv * wv)

                running_sum = correction * running_sum + w
                running_max = new_max

            # ── Normalize ──────────────────────────────────────────
            var inv = SIMD[DType.float32, width](1.0 / running_sum)
            for c in range(0, kv_lora_rank, width):
                (olp + c).store((olp + c).load[width=width]() * inv)

            # ── Step 4: o[h] = W_UV @ o_latent ────────────────────
            #
            # Per output element: 512-dim dot of W_UV row against
            # the normalized latent accumulator.
            var d_head = d_row + h * v_head_dim
            for k in range(v_head_dim):
                var w_row = wuv + k * kv_lora_rank
                var acc = SIMD[DType.float32, width](0)
                for c in range(0, kv_lora_rank, width):
                    var wval = (w_row + c).load[width=width]().cast[DType.float32]()
                    var o = (olp + c).load[width=width]()
                    acc = wval.fma(o, acc)
                (d_head + k).store(acc.reduce_add().cast[DType.bfloat16]())


# =============================================================================
# MLA attention — typed wrapper
# =============================================================================


def mla_attention[
    num_heads: Int,
    nope_dim: Int,
    rope_dim: Int,
    kv_lora_rank: Int,
    v_head_dim: Int,
    softmax_scale: Float64,
    QT: Encoding & Shaped,
    CkvCT: Encoding & Shaped,
    KrCT: Encoding & Shaped,
    KvBT: Encoding & Shaped,
    OutT: Encoding & Shaped,
    P: BurstThreadPool,
](
    q: DynView[QT],
    ckv_cache: CacheView[CkvCT],
    kr_cache: CacheView[KrCT],
    kv_b_proj: Bound[KvBT],
    output: DynView[OutT],
    pos: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Weight-absorbed MLA attention dispatched over BurstPool.

    Partitions query heads across workers. All heads read from the
    same shared c_KV and k_R caches.

    Q:          [seq, num_heads * (nope_dim + rope_dim)]
    c_KV cache: [capacity, kv_lora_rank]
    k_R cache:  [capacity, rope_dim]
    kv_b_proj:  [num_heads * (nope_dim + v_head_dim), kv_lora_rank]
    output:     [seq, num_heads * v_head_dim]
    """
    comptime q_head_dim = nope_dim + rope_dim
    comptime w = simd_width_of[DType.float32]()

    comptime assert QT.DTYPE == DType.bfloat16, "mla: Q must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "mla: output must be bf16"
    comptime assert CkvCT.DTYPE == DType.bfloat16, "mla: c_KV cache must be bf16"
    comptime assert KrCT.DTYPE == DType.bfloat16, "mla: k_R cache must be bf16"
    comptime assert KvBT.DTYPE == DType.bfloat16, "mla: kv_b must be bf16"
    comptime assert QT.COLS == num_heads * q_head_dim, "mla: Q cols mismatch"
    comptime assert OutT.COLS == num_heads * v_head_dim, "mla: output cols mismatch"
    comptime assert CkvCT.COLS == kv_lora_rank, "mla: c_KV cols mismatch"
    comptime assert KrCT.COLS == rope_dim, "mla: k_R cols mismatch"
    comptime assert KvBT.ROWS == num_heads * (nope_dim + v_head_dim), "mla: kv_b rows mismatch"
    comptime assert KvBT.COLS == kv_lora_rank, "mla: kv_b cols mismatch"
    comptime assert kv_lora_rank % w == 0, "mla: d_c must be f32-simd-aligned"
    comptime assert rope_dim % w == 0, "mla: d_R must be f32-simd-aligned"

    var seq_len = q.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(num_heads, pool.get_capacity())
    var heads_per_job = (num_heads + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * heads_per_job
        var end = min(start + heads_per_job, num_heads)
        var pack = pool.get_args_base() + i
        pack[].arg0 = q.ptr
        pack[].arg1 = output.ptr
        pack[].arg2 = ckv_cache.ptr
        pack[].arg3 = kr_cache.ptr
        pack[].arg4 = kv_b_proj.ptr
        pack[].arg5 = (start << 56) | (end << 48) | (pos << 24) | seq_len

    pool.dispatch(
        mla_kernel[num_heads, nope_dim, rope_dim, kv_lora_rank, v_head_dim, softmax_scale],
        pool.get_args_base(), num_jobs,
    )
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# =============================================================================
# KV-A split + RMSNorm + dual cache write — BurstPool dispatch target
# =============================================================================


def mla_kv_split_kernel[kv_lora_rank: Int, rope_dim: Int, eps: Float64](
    src_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    norm_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ckv_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    kr_p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_end: Int,
    pos: Int,
):
    """Split kv_a output, RMSNorm the c_KV portion, write both to caches.

    src_p:  kv_a buffer  [seq, kv_lora_rank + rope_dim]  (strided rows)
    norm_p: layernorm γ  [kv_lora_rank]
    ckv_p:  c_KV cache   [capacity, kv_lora_rank]
    kr_p:   k_R cache    [capacity, rope_dim]

    Assumes mla_rope_kr has already been applied to src in-place.
    """
    comptime stride = kv_lora_rank + rope_dim
    comptime f32w = simd_width_of[DType.float32]()
    comptime bf16w = simd_width_of[DType.bfloat16]()
    comptime eps_f = Float32(eps)

    var start_row = start_end >> 32
    var end_row = start_end & 0xFFFFFFFF

    for row in range(start_row, end_row):
        var src_row = src_p + row * stride

        # RMSNorm c_KV portion → write to cache
        var sq_acc = SIMD[DType.float32, f32w](0)
        for c in range(0, kv_lora_rank, f32w):
            var x = (src_row + c).load[width=f32w]().cast[DType.float32]()
            sq_acc = x.fma(x, sq_acc)
        var rms_scale = Float32(1.0) / sqrt(
            sq_acc.reduce_add() / Float32(kv_lora_rank) + eps_f
        )
        var sv = SIMD[DType.float32, f32w](rms_scale)

        var dst_ckv = ckv_p + (pos + row) * kv_lora_rank
        for c in range(0, kv_lora_rank, f32w):
            var x = (src_row + c).load[width=f32w]().cast[DType.float32]()
            var g = (norm_p + c).load[width=f32w]().cast[DType.float32]()
            (dst_ckv + c).store((x * sv * g).cast[DType.bfloat16]())

        # Copy k_R (already RoPE'd) → write to cache
        var dst_kr = kr_p + (pos + row) * rope_dim
        for c in range(0, rope_dim, bf16w):
            (dst_kr + c).store(
                (src_row + kv_lora_rank + c).load[width=bf16w]()
            )


# =============================================================================
# KV-A cache write — typed wrapper
# =============================================================================


def mla_kv_cache_write[
    kv_lora_rank: Int,
    rope_dim: Int,
    eps: Float64,
    SrcT: Encoding & Shaped,
    NormT: Encoding & Shaped,
    CkvCT: Encoding & Shaped,
    KrCT: Encoding & Shaped,
    P: BurstThreadPool,
](
    src: DynView[SrcT],
    norm: Bound[NormT],
    ckv_cache: CacheView[CkvCT],
    kr_cache: CacheView[KrCT],
    pos: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Split kv_a, RMSNorm c_KV, write both halves to separate caches.

    Call after mla_rope_kr has rotated k_R in the source buffer in-place.
    """
    comptime assert SrcT.DTYPE == DType.bfloat16, "kv_split: src must be bf16"
    comptime assert SrcT.COLS == kv_lora_rank + rope_dim, "kv_split: src cols mismatch"
    comptime assert CkvCT.COLS == kv_lora_rank, "kv_split: c_KV cols mismatch"
    comptime assert KrCT.COLS == rope_dim, "kv_split: k_R cols mismatch"
    comptime assert kv_lora_rank % simd_width_of[DType.float32]() == 0, "kv_split: d_c alignment"
    comptime assert rope_dim % simd_width_of[DType.bfloat16]() == 0, "kv_split: d_R alignment"

    var seq_len = src.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.get_args_base() + i
        pack[].arg0 = src.ptr
        pack[].arg1 = norm.ptr
        pack[].arg2 = ckv_cache.ptr
        pack[].arg3 = kr_cache.ptr
        pack[].arg4 = (start << 32) | end
        pack[].arg5 = pos

    pool.dispatch(
        mla_kv_split_kernel[kv_lora_rank, rope_dim, eps],
        pool.get_args_base(), num_jobs,
    )
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
