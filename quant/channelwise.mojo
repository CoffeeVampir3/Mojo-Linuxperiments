"""Channelwise quantization pipeline stages.

Atomic stages and their compositions for channelwise int8 quantization.
Each stage conforms to PipelineFn and can be composed freely.

Pipelines:
    channelwise[source, target, precision]   — quantize + pack (pack from weight descriptor)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from threading import BurstPool

from .pipelines import PipelineFn, Timer
from simd_math import bf16_load_as, roundeven


# =============================================================================
# Atomic stage: channelwise scale computation
# =============================================================================


def channelwise_scales[source: DType, precision: DType](
    pool_ptr: UnsafePointer[BurstPool[], MutAnyOrigin],
    src: UnsafePointer[UInt8, MutAnyOrigin],
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Compute per-row abs-max scale. scale[row] = max(|src[row,:]|) / 127."""
    var num_jobs = min(rows, pool_ptr[].capacity)
    var rows_per_job = (rows + num_jobs - 1) // num_jobs
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, rows)
        var pack = pool_ptr[].args_base + i
        pack[].arg0 = Int(src)
        pack[].arg1 = Int(scale)
        pack[].arg2 = rows
        pack[].arg3 = cols
        pack[].arg4 = start
        pack[].arg5 = end
    pool_ptr[].dispatch(
        channelwise_scales_kernel[precision],
        pool_ptr[].args_base, num_jobs,
    )
    pool_ptr[].join()


def channelwise_scales_kernel[precision: DType](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[precision], MutAnyOrigin],
    rows: Int, cols: Int, start_row: Int, end_row: Int,
):
    comptime width = simd_width_of[precision]()
    for row in range(start_row, end_row):
        var row_ptr = src + row * cols
        var vmax = SIMD[precision, width](0)
        for k in range(0, cols, width):
            vmax = max(vmax, bf16_load_as[precision, width](row_ptr, k).__abs__())
        dst[row] = vmax.reduce_max() / Scalar[precision](127.0)


# =============================================================================
# Atomic stage: channelwise int8 quantization
# =============================================================================


def channelwise_quantize[source: DType, target: DType, precision: DType](
    pool_ptr: UnsafePointer[BurstPool[], MutAnyOrigin],
    src: UnsafePointer[UInt8, MutAnyOrigin],
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Quantize: weight[row,k] = clamp(round(src[row,k] / scale[row]), -128, 127)."""
    var num_jobs = min(rows, pool_ptr[].capacity)
    var rows_per_job = (rows + num_jobs - 1) // num_jobs
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, rows)
        var pack = pool_ptr[].args_base + i
        pack[].arg0 = Int(src)
        pack[].arg1 = Int(scale)
        pack[].arg2 = Int(weight)
        pack[].arg3 = cols
        pack[].arg4 = start
        pack[].arg5 = end
    pool_ptr[].dispatch(
        channelwise_quantize_kernel[target, precision],
        pool_ptr[].args_base, num_jobs,
    )
    pool_ptr[].join()


def channelwise_quantize_kernel[target: DType, precision: DType](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    scales: UnsafePointer[Scalar[precision], MutAnyOrigin],
    dst: UnsafePointer[Scalar[target], MutAnyOrigin],
    cols: Int, start_row: Int, end_row: Int,
):
    comptime width = simd_width_of[precision]()
    comptime lo = SIMD[precision, width](-128.0)
    comptime hi = SIMD[precision, width](127.0)

    for row in range(start_row, end_row):
        var scale = scales[row]
        var inv_scale = SIMD[precision, width](
            Scalar[precision](1.0) / scale if scale != Scalar[precision](0.0) else Scalar[precision](0.0)
        )
        var src_row = src + row * cols
        var dst_row = dst + row * cols
        for k in range(0, cols, width):
            var v = bf16_load_as[precision, width](src_row, k)
            var clamped = min(max(roundeven(v * inv_scale), lo), hi)
            (dst_row + k).store(clamped.cast[target]())


# =============================================================================
# Pipeline composition
# =============================================================================


def channelwise[source: DType, target: DType, precision: DType = DType.float32](
    pool_ptr: UnsafePointer[BurstPool[], MutAnyOrigin],
    src: UnsafePointer[UInt8, MutAnyOrigin],
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Channelwise quantization: scales -> quantize."""
    var t1 = Timer("scales")
    channelwise_scales[source, precision](pool_ptr, src, weight, scale, rows, cols)
    t1^.stop()

    var t2 = Timer("quantize")
    channelwise_quantize[source, target, precision](pool_ptr, src, weight, scale, rows, cols)
    t2^.stop()
