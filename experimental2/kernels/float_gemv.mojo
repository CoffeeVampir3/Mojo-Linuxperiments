"""BF16 GEMV — column-parallel decode for tied-embedding LM head.

dst[1,N] = input[1,K] × weight[N,K]^T via BurstPool dispatch.
Workers split N (output columns), 4-wide column tiling per worker.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kernel_ops import PoolFence


@fieldwise_init
struct FloatGemvArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var weight_ptr: Int
    var out_ptr: Int
    var start_col: Int
    var end_col: Int


def float_gemv_kernel[K: Int, N: Int](args: FloatGemvArgs):
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var wp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.weight_ptr)
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.out_ptr)
    comptime width = simd_width_of[DType.float32]()
    comptime Nr = 4
    var start_col = args.start_col
    var end_col = args.end_col
    var n_full = end_col - ((end_col - start_col) % Nr)

    for n in range(start_col, n_full, Nr):
        var w0 = wp + n * K
        var w1 = wp + (n + 1) * K
        var w2 = wp + (n + 2) * K
        var w3 = wp + (n + 3) * K
        var acc0 = SIMD[DType.float32, width](0)
        var acc1 = SIMD[DType.float32, width](0)
        var acc2 = SIMD[DType.float32, width](0)
        var acc3 = SIMD[DType.float32, width](0)
        for k in range(0, K, width):
            var x = (ip + k).load[width=width]().cast[DType.float32]()
            acc0 = x.fma((w0 + k).load[width=width]().cast[DType.float32](), acc0)
            acc1 = x.fma((w1 + k).load[width=width]().cast[DType.float32](), acc1)
            acc2 = x.fma((w2 + k).load[width=width]().cast[DType.float32](), acc2)
            acc3 = x.fma((w3 + k).load[width=width]().cast[DType.float32](), acc3)
        dp[n] = acc0.reduce_add().cast[DType.bfloat16]()
        dp[n + 1] = acc1.reduce_add().cast[DType.bfloat16]()
        dp[n + 2] = acc2.reduce_add().cast[DType.bfloat16]()
        dp[n + 3] = acc3.reduce_add().cast[DType.bfloat16]()
    for n in range(n_full, end_col):
        var row_w = wp + n * K
        var acc = SIMD[DType.float32, width](0)
        for k in range(0, K, width):
            var x = (ip + k).load[width=width]().cast[DType.float32]()
            acc = x.fma((row_w + k).load[width=width]().cast[DType.float32](), acc)
        dp[n] = acc.reduce_add().cast[DType.bfloat16]()


def float_gemv[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: P,
) -> PoolFence[P] where W.DTYPE == DType.bfloat16:
    comptime assert InT.DTYPE == DType.bfloat16, "float_gemv: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "float_gemv: output must be bf16"
    comptime assert InT.COLS == W.COLS, "float_gemv: input K != weight K"
    comptime assert OutT.COLS == W.ROWS, "float_gemv: output N != weight N"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "float_gemv: K must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime N = W.ROWS
    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = pool.get_capacity()
    var cols_per_job = (N + num_jobs - 1) // num_jobs

    var jobs = InlineArray[FloatGemvArgs, MAX_POOL_CAPACITY](
        fill=FloatGemvArgs(0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * cols_per_job
        var end = min(start + cols_per_job, N)
        jobs[i] = FloatGemvArgs(input.ptr, weight.ptr, output.ptr, start, end)

    pool.dispatch[FloatGemvArgs, float_gemv_kernel[InT.COLS, N]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
