"""RMSNorm variants for Gemma4.

rmsnorm_no_scale: output = input / rms(input)
  Used for V-norm and router-norm. No learnable weight.

rmsnorm_per_head: output = (input / rms(input)) * weight
  Used for Q-norm and K-norm. Operates on head_dim-width vectors
  with a shared weight vector broadcast across all heads.
  Input shape: (seq_len, num_heads * head_dim), norm applied per-head.
"""

from std.math import sqrt
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY


# =============================================================================
# Dispatch arg structs
# =============================================================================


@fieldwise_init
struct RMSNormNoScaleArgs(Copyable, ImplicitlyCopyable):
    var input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var start_row: Int
    var end_row: Int
    var eps: Float64


@fieldwise_init
struct RMSNormPerHeadArgs(Copyable, ImplicitlyCopyable):
    var input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var start_row: Int
    var end_row: Int
    var eps: Float64


# =============================================================================
# Kernel functions
# =============================================================================


def rmsnorm_no_scale_kernel[cols: Int](args: RMSNormNoScaleArgs):
    """RMSNorm without learnable scale. output = input / rms(input).
    Each row of width `cols` is normalized independently."""
    var ip = args.input
    var op = args.output
    comptime width = simd_width_of[DType.float32]()

    for row in range(args.start_row, args.end_row):
        var row_in = ip + row * cols
        var row_out = op + row * cols

        var acc = SIMD[DType.float32, width](0)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            acc = x.fma(x, acc)
        var sum_sq = acc.reduce_add()
        var scale = Float32(1.0) / sqrt(sum_sq / Float32(cols) + Float32(args.eps))

        var sv = SIMD[DType.float32, width](scale)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            (row_out + j).store((x * sv).cast[DType.bfloat16]())


def rmsnorm_per_head_kernel[head_dim: Int, num_heads: Int](args: RMSNormPerHeadArgs):
    """RMSNorm with learnable scale, applied per-head.

    Input row is (num_heads * head_dim) wide. Each head_dim segment
    is normalized independently, then scaled by the shared weight
    vector of length head_dim.
    """
    var ip = args.input
    var wp = args.weight
    var op = args.output
    comptime width = simd_width_of[DType.float32]()
    comptime row_stride = num_heads * head_dim

    for row in range(args.start_row, args.end_row):
        var row_base_in = ip + row * row_stride
        var row_base_out = op + row * row_stride

        for h in range(num_heads):
            var head_in = row_base_in + h * head_dim
            var head_out = row_base_out + h * head_dim

            var acc = SIMD[DType.float32, width](0)
            for j in range(0, head_dim, width):
                var x = (head_in + j).load[width=width]().cast[DType.float32]()
                acc = x.fma(x, acc)
            var sum_sq = acc.reduce_add()
            var scale = Float32(1.0) / sqrt(sum_sq / Float32(head_dim) + Float32(args.eps))

            var sv = SIMD[DType.float32, width](scale)
            for j in range(0, head_dim, width):
                var x = (head_in + j).load[width=width]().cast[DType.float32]()
                var w = (wp + j).load[width=width]().cast[DType.float32]()
                (head_out + j).store((x * sv * w).cast[DType.bfloat16]())


# =============================================================================
# High-level operations
# =============================================================================


def rmsnorm_no_scale[InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P]:
    """RMSNorm without learnable scale via BurstPool.
    output = input / rms(input). F32 accumulation, bf16 I/O."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_no_scale: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_no_scale: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm_no_scale: cols must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = tptr[Scalar[DType.bfloat16]](input.ptr)
    var op = tptr[Scalar[DType.bfloat16]](output.ptr)
    var jobs = InlineArray[RMSNormNoScaleArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormNoScaleArgs(ip, op, start, end, Float64(eps))

    pool.dispatch[RMSNormNoScaleArgs, rmsnorm_no_scale_kernel[InT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_per_head[head_dim: Int, num_heads: Int,
    W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: P,
    eps: Float32 = 1e-6,
) -> PoolFence[P] where W.DTYPE == DType.bfloat16:
    """Per-head RMSNorm with learnable scale via BurstPool.
    Each head_dim segment is normalized independently, then scaled
    by the shared weight vector. F32 accumulation, bf16 I/O."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm_per_head: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm_per_head: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm_per_head: input/output cols mismatch"
    comptime assert InT.COLS == head_dim * num_heads, "rmsnorm_per_head: cols != heads * dim"
    comptime assert W.ROWS * W.COLS == head_dim, "rmsnorm_per_head: weight size != head_dim"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "rmsnorm_per_head: head_dim must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = tptr[Scalar[DType.bfloat16]](input.ptr)
    var wp = tptr[Scalar[DType.bfloat16]](weight.ptr)
    var op = tptr[Scalar[DType.bfloat16]](output.ptr)
    var jobs = InlineArray[RMSNormPerHeadArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormPerHeadArgs(ip, wp, op, start, end, Float64(eps))

    pool.dispatch[RMSNormPerHeadArgs, rmsnorm_per_head_kernel[head_dim, num_heads]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
