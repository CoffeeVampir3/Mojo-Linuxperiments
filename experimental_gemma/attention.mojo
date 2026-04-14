"""Gemma4 attention kernels.

Local (sliding window) attention:
  - 2:1 GQA (16Q / 8KV), head_dim=256
  - Window size = 1024 tokens
  - Scale = 1.0 (QK-norm replaces 1/sqrt(d))
  - Online softmax, single-pass (no score buffer)
  - SIMD register accumulator (head_dim=256 fits in registers)

Global (full causal) attention:
  - 8:1 GQA (16Q / 2KV), head_dim=512
  - Full causal (attend to all prior positions)
  - Scale = 1.0
  - Stack-array accumulator (head_dim=512 would exhaust registers)
  - Runtime loops to avoid instruction cache pressure from unrolling
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY, BF16Ptr
from threading.threading_traits import BurstThreadPool
from modeling.model_spec import Encoding, Shaped, DynView, CacheView
from simd_math import exp_f32


# =============================================================================
# Dispatch args
# =============================================================================


@fieldwise_init
struct AttentionHeadArgs(Copyable, ImplicitlyCopyable):
    var qp: BF16Ptr
    var dp: BF16Ptr
    var kp: BF16Ptr
    var vp: BF16Ptr
    var start_item: Int
    var end_item: Int
    var pos: Int


# =============================================================================
# Local (sliding window) attention kernel
# =============================================================================


def local_attention_kernel[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    kv_cols: Int, window_size: Int,
](args: AttentionHeadArgs):
    """Sliding-window GQA with scale=1.0 and online softmax.

    Each query attends only to KV positions within [causal_len - window_size, causal_len).
    Score = dot(q, k) with no 1/sqrt(d) scaling (QK-norm handles this).
    """
    var qp = args.qp
    var dp = args.dp
    var kp = args.kp
    var vp = args.vp
    comptime heads_per_group = num_heads // num_kv_heads
    comptime q_stride = num_heads * head_dim
    comptime kv_stride = kv_cols
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = head_dim // width

    for item in range(args.start_item, args.end_item):
        var m = item // num_heads
        var q_head_idx = item - m * num_heads
        var causal_len = args.pos + m + 1
        var window_start = max(0, causal_len - window_size)
        var kv_group = q_head_idx // heads_per_group
        var kv_head_offset = kv_group * head_dim
        var q_head = qp + m * q_stride + q_head_idx * head_dim
        var d_head = dp + m * q_stride + q_head_idx * head_dim

        var acc = SIMD[DType.float32, head_dim](0)
        var running_max = Float32(-1e30)
        var running_sum = Float32(0)

        for t in range(window_start, causal_len):
            # Score = dot(q, K[t]), no scaling
            var k_row = kp + t * kv_stride + kv_head_offset
            var dot_acc = SIMD[DType.float32, width](0)
            comptime for c in range(chunks):
                comptime off = c * width
                var qv = (q_head + off).load[width=width]().cast[DType.float32]()
                var kv = (k_row + off).load[width=width]().cast[DType.float32]()
                dot_acc = qv.fma(kv, dot_acc)
            var score = dot_acc.reduce_add()

            # Online softmax
            var new_max = max(running_max, score)
            var correction = Float32(
                exp_f32[1](SIMD[DType.float32, 1](running_max - new_max))
            )
            var w = Float32(
                exp_f32[1](SIMD[DType.float32, 1](score - new_max))
            )

            # V accumulation
            var v_row = vp + t * kv_stride + kv_head_offset
            comptime for c in range(chunks):
                comptime off = c * width
                var prior = acc.slice[width, offset=off]()
                var vv = (v_row + off).load[width=width]().cast[DType.float32]()
                acc = acc.insert[offset=off](prior * correction + vv * w)

            running_sum = correction * running_sum + w
            running_max = new_max

        # Normalize and store
        var inv_sum = 1.0 / running_sum
        comptime for c in range(chunks):
            comptime off = c * width
            var v = acc.slice[width, offset=off]() * inv_sum
            (d_head + off).store(v.cast[DType.bfloat16]())


# =============================================================================
# High-level dispatch
# =============================================================================


def local_attention[num_heads: Int, num_kv_heads: Int, head_dim: Int,
    window_size: Int,
    QT: Encoding & Shaped, KCT: Encoding & Shaped, VCT: Encoding & Shaped,
    OutT: Encoding & Shaped, P: BurstThreadPool](
    q: DynView[QT], k_cache: CacheView[KCT], v_cache: CacheView[VCT],
    output: DynView[OutT], pos: Int,
    mut pool: P,
) -> PoolFence[P] where KCT.DTYPE == DType.bfloat16:
    """Sliding-window GQA attention with scale=1.0.

    Q[M, H*D] attends over KV cache within a window of window_size tokens.
    Causal masked, online softmax. Partitioned by KV head group.
    """
    comptime assert QT.DTYPE == DType.bfloat16, "local_attention: Q must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "local_attention: output must be bf16"
    comptime assert VCT.DTYPE == DType.bfloat16, "local_attention: V cache must be bf16"
    comptime assert QT.COLS == num_heads * head_dim, "local_attention: Q cols != H*D"
    comptime assert OutT.COLS == QT.COLS, "local_attention: output cols != Q cols"
    comptime assert KCT.COLS == num_kv_heads * head_dim, "local_attention: K cache cols != Hkv*D"
    comptime assert VCT.COLS == KCT.COLS, "local_attention: V cache cols != K cache cols"
    comptime assert KCT.ROWS == VCT.ROWS, "local_attention: K/V capacity mismatch"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "local_attention: head_dim must be f32-simd-aligned"
    comptime assert num_heads % num_kv_heads == 0, "local_attention: heads must divide evenly for GQA"

    var seq_len = q.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var total_items = seq_len * num_heads
    var num_jobs = min(total_items, pool.get_capacity())
    var items_per_job = (total_items + num_jobs - 1) // num_jobs

    var qpp = q.as_ptr[DType.bfloat16]()
    var opp = output.as_ptr[DType.bfloat16]()
    var kpp = k_cache.as_ptr[DType.bfloat16]()
    var vpp = v_cache.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[AttentionHeadArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * items_per_job
        var end = min(start + items_per_job, total_items)
        jobs[i] = AttentionHeadArgs(qpp, opp, kpp, vpp, start, end, pos)

    pool.dispatch[AttentionHeadArgs,
        local_attention_kernel[num_heads, num_kv_heads, head_dim, KCT.COLS, window_size]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# =============================================================================
# Global (full causal) attention kernel
# =============================================================================


def global_attention_kernel[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    kv_cols: Int,
](args: AttentionHeadArgs):
    """Full-causal GQA with scale=1.0 and stack-array V accumulator.

    Uses stack-resident f32 array for V accumulation instead of
    SIMD[f32, head_dim] to avoid register exhaustion at head_dim=512.
    Runtime loops to keep instruction footprint manageable.
    """
    var qp = args.qp
    var dp = args.dp
    var kp = args.kp
    var vp = args.vp
    comptime heads_per_group = num_heads // num_kv_heads
    comptime q_stride = num_heads * head_dim
    comptime kv_stride = kv_cols
    comptime width = simd_width_of[DType.float32]()

    for item in range(args.start_item, args.end_item):
        var m = item // num_heads
        var q_head_idx = item - m * num_heads
        var causal_len = args.pos + m + 1
        var kv_group = q_head_idx // heads_per_group
        var kv_head_offset = kv_group * head_dim
        var q_head = qp + m * q_stride + q_head_idx * head_dim
        var d_head = dp + m * q_stride + q_head_idx * head_dim

        var acc_buf = InlineArray[Float32, head_dim](fill=Float32(0))
        var ap = UnsafePointer(to=acc_buf[0])
        var running_max = Float32(-1e30)
        var running_sum = Float32(0)

        for t in range(causal_len):
            # Score = dot(q, K[t]), no scaling
            var k_row = kp + t * kv_stride + kv_head_offset
            var dot_acc = SIMD[DType.float32, width](0)
            for c in range(0, head_dim, width):
                var qv = (q_head + c).load[width=width]().cast[DType.float32]()
                var kv = (k_row + c).load[width=width]().cast[DType.float32]()
                dot_acc = qv.fma(kv, dot_acc)
            var score = dot_acc.reduce_add()

            # Online softmax
            var new_max = max(running_max, score)
            var correction = Float32(
                exp_f32[1](SIMD[DType.float32, 1](running_max - new_max))
            )
            var w = Float32(
                exp_f32[1](SIMD[DType.float32, 1](score - new_max))
            )

            # V accumulation via stack array
            var v_row = vp + t * kv_stride + kv_head_offset
            for c in range(0, head_dim, width):
                var prior = (ap + c).load[width=width]()
                var vv = (v_row + c).load[width=width]().cast[DType.float32]()
                (ap + c).store(prior * correction + vv * w)

            running_sum = correction * running_sum + w
            running_max = new_max

        # Normalize and store
        var inv_sum = 1.0 / running_sum
        for c in range(0, head_dim, width):
            var v = (ap + c).load[width=width]() * inv_sum
            (d_head + c).store(v.cast[DType.bfloat16]())


# =============================================================================
# Global attention dispatch
# =============================================================================


def global_attention[num_heads: Int, num_kv_heads: Int, head_dim: Int,
    QT: Encoding & Shaped, KCT: Encoding & Shaped, VCT: Encoding & Shaped,
    OutT: Encoding & Shaped, P: BurstThreadPool](
    q: DynView[QT], k_cache: CacheView[KCT], v_cache: CacheView[VCT],
    output: DynView[OutT], pos: Int,
    mut pool: P,
) -> PoolFence[P] where KCT.DTYPE == DType.bfloat16:
    """Full-causal GQA attention with scale=1.0.

    Q[M, H*D] attends over full KV cache history [0..pos+M].
    Causal masked, online softmax. Partitioned by KV head group.
    """
    comptime assert QT.DTYPE == DType.bfloat16, "global_attention: Q must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "global_attention: output must be bf16"
    comptime assert VCT.DTYPE == DType.bfloat16, "global_attention: V cache must be bf16"
    comptime assert QT.COLS == num_heads * head_dim, "global_attention: Q cols != H*D"
    comptime assert OutT.COLS == QT.COLS, "global_attention: output cols != Q cols"
    comptime assert KCT.COLS == num_kv_heads * head_dim, "global_attention: K cache cols != Hkv*D"
    comptime assert VCT.COLS == KCT.COLS, "global_attention: V cache cols != K cache cols"
    comptime assert KCT.ROWS == VCT.ROWS, "global_attention: K/V capacity mismatch"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "global_attention: head_dim must be f32-simd-aligned"
    comptime assert num_heads % num_kv_heads == 0, "global_attention: heads must divide evenly for GQA"

    var seq_len = q.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var total_items = seq_len * num_heads
    var num_jobs = min(total_items, pool.get_capacity())
    var items_per_job = (total_items + num_jobs - 1) // num_jobs

    var qpp = q.as_ptr[DType.bfloat16]()
    var opp = output.as_ptr[DType.bfloat16]()
    var kpp = k_cache.as_ptr[DType.bfloat16]()
    var vpp = v_cache.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[AttentionHeadArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * items_per_job
        var end = min(start + items_per_job, total_items)
        jobs[i] = AttentionHeadArgs(qpp, opp, kpp, vpp, start, end, pos)

    pool.dispatch[AttentionHeadArgs,
        global_attention_kernel[num_heads, num_kv_heads, head_dim, KCT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
