"""Gemma4 elementwise operations.

scaled_add: dst += src * scalar (pool-dispatched, for layer_scalar)
embed_lookup_scaled: table[token] * scale (pool-dispatched, for embed_scale)
logit_softcap: tanh(x/30) * 30 (inline, final output transform)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY, BF16Ptr
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr
from modeling.model_spec import Encoding, Shaped, Bound, DynView
from experimental_gemma.activations import tanh_f32


# =============================================================================
# Dispatch arg structs
# =============================================================================


@fieldwise_init
struct ScaledAddArgs(Copyable, ImplicitlyCopyable):
    var src: BF16Ptr
    var dst: BF16Ptr
    var scale: Float32
    var start_row: Int
    var end_row: Int


@fieldwise_init
struct ScaledEmbedArgs(Copyable, ImplicitlyCopyable):
    var table: BF16Ptr
    var tokens: UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
    var output: BF16Ptr
    var scale: Float32
    var start_row: Int
    var end_row: Int


# =============================================================================
# Kernel functions
# =============================================================================


def scaled_add_kernel[cols: Int](args: ScaledAddArgs):
    """dst[row, j] += src[row, j] * scale. F32 compute, bf16 I/O."""
    comptime width = simd_width_of[DType.float32]()
    var sp = args.src
    var dp = args.dst
    var sv = SIMD[DType.float32, width](args.scale)

    for row in range(args.start_row, args.end_row):
        var src_row = sp + row * cols
        var dst_row = dp + row * cols
        for j in range(0, cols, width):
            var s = (src_row + j).load[width=width]().cast[DType.float32]()
            var d = (dst_row + j).load[width=width]().cast[DType.float32]()
            (dst_row + j).store((d + s * sv).cast[DType.bfloat16]())


def embed_lookup_scaled_kernel[cols: Int](args: ScaledEmbedArgs):
    """out[row] = table[token[row]] * scale. F32 compute, bf16 I/O."""
    comptime width = simd_width_of[DType.float32]()
    var tp = args.table
    var tokens = args.tokens
    var dp = args.output
    var sv = SIMD[DType.float32, width](args.scale)

    for i in range(args.start_row, args.end_row):
        var src = tp + Int(tokens[i]) * cols
        var out = dp + i * cols
        for j in range(0, cols, width):
            var v = (src + j).load[width=width]().cast[DType.float32]()
            (out + j).store((v * sv).cast[DType.bfloat16]())


# =============================================================================
# High-level operations
# =============================================================================


def scaled_add[SrcT: Encoding & Shaped, DstT: Encoding & Shaped,
    P: BurstThreadPool](
    src: DynView[SrcT], dst: DynView[DstT],
    scale: Float32,
    mut pool: P,
) -> PoolFence[P]:
    """dst += src * scale. F32 compute, bf16 I/O. Partitioned by rows."""
    comptime assert SrcT.DTYPE == DType.bfloat16, "scaled_add: src must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "scaled_add: dst must be bf16"
    comptime assert SrcT.COLS == DstT.COLS, "scaled_add: src/dst cols mismatch"
    comptime assert SrcT.COLS % simd_width_of[DType.float32]() == 0, "scaled_add: cols must be f32-simd-aligned"

    var seq_len = src.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var sp = tptr[Scalar[DType.bfloat16]](src.ptr)
    var dp = tptr[Scalar[DType.bfloat16]](dst.ptr)
    var jobs = InlineArray[ScaledAddArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = ScaledAddArgs(sp, dp, scale, start, end)

    pool.dispatch[ScaledAddArgs, scaled_add_kernel[SrcT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def embed_lookup_scaled[W: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool](
    table: Bound[W], tokens: Int, output: DynView[OutT],
    scale: Float32,
    mut pool: P,
) -> PoolFence[P] where W.DTYPE == DType.bfloat16:
    """Gather + scale: for each token ID, output[row] = table[id] * scale."""
    comptime assert OutT.DTYPE == DType.bfloat16, "embed_scaled: output must be bf16"
    comptime assert W.COLS == OutT.COLS, "embed_scaled: table hidden != output hidden"

    var seq_len = output.seq_len
    if seq_len == 0:
        return PoolFence[P].completed()

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var tp = tptr[Scalar[DType.bfloat16]](table.ptr)
    var tkp = tptr[Scalar[DType.int32]](tokens)
    var op = tptr[Scalar[DType.bfloat16]](output.ptr)
    var jobs = InlineArray[ScaledEmbedArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = ScaledEmbedArgs(tp, tkp, op, scale, start, end)

    pool.dispatch[ScaledEmbedArgs, embed_lookup_scaled_kernel[W.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def elem_scale[T: Encoding & Shaped](dst: DynView[T], scale: Float32):
    """In-place scalar multiply: dst *= scale. F32 compute, bf16 I/O."""
    comptime assert T.DTYPE == DType.bfloat16, "elem_scale: must be bf16"
    comptime width = simd_width_of[DType.float32]()
    var sv = SIMD[DType.float32, width](scale)
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst.ptr)
    for i in range(0, dst.seq_len * T.COLS, width):
        var v = (dp + i).load[width=width]().cast[DType.float32]()
        (dp + i).store((v * sv).cast[DType.bfloat16]())


def logit_softcap[T: Encoding & Shaped](dst: DynView[T]):
    """In-place logit softcapping: dst = tanh(dst / 30) * 30."""
    comptime assert T.DTYPE == DType.bfloat16, "logit_softcap: must be bf16"
    comptime width = simd_width_of[DType.float32]()
    comptime cap = Float32(30.0)
    comptime inv_cap = Float32(1.0) / cap

    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst.ptr)
    for i in range(0, dst.seq_len * T.COLS, width):
        var v = (dp + i).load[width=width]().cast[DType.float32]()
        var capped = tanh_f32(v * inv_cap) * cap
        (dp + i).store(capped.cast[DType.bfloat16]())
