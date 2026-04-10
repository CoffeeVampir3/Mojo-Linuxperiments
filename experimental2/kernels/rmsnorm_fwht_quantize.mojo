"""RMSNorm + FWHT + dynamic-scale i8 quantization.

Per row: bf16 -> f32 -> divide by RMS -> block-diagonal FWHT -> compute
absmax -> quantize with per-row dynamic scale.

No gamma (absorbed into weights). Per-row absmax scale written to output
array for downstream GEMV dequantization.

    absmax[m] = max(|FWHT(x[m] / rms(x[m]))|)
    qi[m, k] = clamp(round(FWHT(x[m] / rms(x[m])) * 127 / absmax[m]), -128, 127)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.utils import IndexList
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from kernels.kernel_ops import PoolFence
from simd_math import sqrt
from simd_math.matrixops import log2
from experimental2.kernels.quantize import absmax_quantize_i8


# ============================================================================
# FWHT primitives (self-contained, no experimental dependency)
# ============================================================================


def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


def butterfly_partner[i: Int, stride: Int]() -> Int:
    return i ^ stride


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result


@always_inline
def fwht_apply[T: DType, block: Int](
    mut r: InlineArray[SIMD[T, fwht_width[T, block]()], block // fwht_width[T, block]()],
):
    """Butterfly stages + 1/sqrt(block) normalize on pre-loaded registers."""
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            comptime mask = butterfly_shuffle[width, stride]()
            var sign_buf = InlineArray[Scalar[T], width](fill=Scalar[T](1.0))
            comptime for k in range(width):
                comptime if (k >> stage) & 1 != 0:
                    sign_buf[k] = Scalar[T](-1.0)
            var sign = UnsafePointer(to=sign_buf).bitcast[Scalar[T]]().load[width=width]()
            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val

    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc


@always_inline
def fwht_block[block: Int](buf: UnsafePointer[Float32, MutAnyOrigin]):
    """In-register FWHT on one block of f32 elements."""
    comptime width = fwht_width[DType.float32, block]()
    comptime regs = block // width

    var r = InlineArray[SIMD[DType.float32, width], regs](fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


# ============================================================================
# Row kernel — one row: bf16 -> f32 -> RMSNorm -> FWHT -> quantize i8
# ============================================================================


def rmsnorm_gamma_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm * gamma -> FWHT -> dynamic i8.

    Gamma is applied during normalization (fused multiply).
    Computes per-row absmax after FWHT, writes it to scale_out, quantizes.
    """
    comptime width = simd_width_of[DType.float32]()

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        var g = (gamma + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x * g)
        k += width

    var inv_rms = Float32(1.0) / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(cols // block):
        fwht_block[block](work + b * block)

    scale_out[] = absmax_quantize_i8[cols](work, row_qi)


def rmsnorm_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm -> FWHT -> dynamic i8.

    Computes per-row absmax after FWHT, writes it to scale_out, quantizes.
    """
    comptime width = simd_width_of[DType.float32]()

    # Load bf16 -> f32 + RMS reduction
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x)
        k += width

    var rms = sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var inv_rms = Float32(1.0) / rms

    # Normalize in place
    var vinv_rms = SIMD[DType.float32, width](inv_rms)
    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    # Block-diagonal FWHT
    for b in range(cols // block):
        fwht_block[block](work + b * block)

    scale_out[] = absmax_quantize_i8[cols](work, row_qi)


# ============================================================================
# Worker args + kernel
# ============================================================================


@fieldwise_init
struct RmsNormFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var start_row: Int
    var end_row: Int


def rmsnorm_fwht_quantize_worker[cols: Int, block: Int](args: RmsNormFwhtArgs):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_fwht_quantize_row[cols, block](
            inp + m * cols, qi + m * cols, work, scales + m, 1e-5,
        )


@fieldwise_init
struct RmsNormGammaFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var gamma_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var start_row: Int
    var end_row: Int


def rmsnorm_gamma_fwht_quantize_worker[cols: Int, block: Int](args: RmsNormGammaFwhtArgs):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var gamma = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gamma_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_gamma_fwht_quantize_row[cols, block](
            inp + m * cols, gamma, qi + m * cols, work, scales + m, 1e-5,
        )


# ============================================================================
# Dispatch
# ============================================================================


def rmsnorm_gamma_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, gamma_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm * gamma + FWHT + dynamic-scale quantize.

    in_ptr:     bf16 [seq_len, cols]
    gamma_ptr:  bf16 [cols] — norm gamma, applied during normalization
    qi_ptr:     i8   [seq_len, cols] output
    work_ptr:   f32  [cols] scratch per worker
    scale_ptr:  f32  [seq_len] output — per-row absmax scales for GEMV dequant
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormGammaFwhtArgs, MAX_POOL_CAPACITY](
        fill=RmsNormGammaFwhtArgs(0, 0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormGammaFwhtArgs(
            in_ptr, gamma_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, start, end)

    pool.dispatch[RmsNormGammaFwhtArgs, rmsnorm_gamma_fwht_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def rmsnorm_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm + FWHT + dynamic-scale quantize.

    in_ptr:    bf16 [seq_len, cols]
    qi_ptr:    i8   [seq_len, cols] output
    work_ptr:  f32  [cols] scratch per worker
    scale_ptr: f32  [seq_len] output — per-row absmax scales for GEMV dequant
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormFwhtArgs, MAX_POOL_CAPACITY](
        fill=RmsNormFwhtArgs(0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtArgs(
            in_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, start, end)

    pool.dispatch[RmsNormFwhtArgs, rmsnorm_fwht_quantize_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# Blocked-scale variant — one absmax scale per FWHT block
# ============================================================================


@always_inline
def rmsnorm_fwht_quantize_row_blocked[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    blk_scale_out: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    """One row: bf16 -> RMSNorm -> FWHT -> dynamic i8 with per-block scales."""
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block

    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (row_in + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x)
        k += width

    var rms = sqrt[DType.float32, 1](vsum.reduce_add() / Float32(cols) + eps)
    var inv_rms = Float32(1.0) / rms
    var vinv_rms = SIMD[DType.float32, width](inv_rms)

    k = 0
    while k + width <= cols:
        (work + k).store((work + k).load[width=width]() * vinv_rms)
        k += width

    for b in range(num_blocks):
        fwht_block[block](work + b * block)
        blk_scale_out[b] = absmax_quantize_i8[block](work + b * block, row_qi + b * block)


@fieldwise_init
struct RmsNormFwhtBlockedArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var blk_scale_ptr: Int
    var start_row: Int
    var end_row: Int


def rmsnorm_fwht_quantize_blocked_worker[cols: Int, block: Int](
    args: RmsNormFwhtBlockedArgs,
):
    comptime num_blocks = cols // block

    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.in_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var blk_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.blk_scale_ptr)

    for m in range(args.start_row, args.end_row):
        rmsnorm_fwht_quantize_row_blocked[cols, block](
            inp + m * cols,
            qi + m * cols,
            work,
            blk_scales + m * num_blocks,
            1e-5,
        )


def rmsnorm_fwht_quantize_blocked[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int,
    qi_ptr: Int,
    work_ptr: Int,
    blk_scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm + FWHT + dynamic i8 with one scale per FWHT block."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[RmsNormFwhtBlockedArgs, MAX_POOL_CAPACITY](
        fill=RmsNormFwhtBlockedArgs(0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RmsNormFwhtBlockedArgs(
            in_ptr,
            qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            blk_scale_ptr,
            start,
            end)

    pool.dispatch[RmsNormFwhtBlockedArgs, rmsnorm_fwht_quantize_blocked_worker[cols, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
