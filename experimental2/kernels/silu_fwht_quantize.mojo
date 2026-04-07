"""SiLU + FWHT + dynamic-scale i8 quantization (MLP domain exit).

Per row: bf16 gate, bf16 up -> f32 SiLU(gate) * up -> block-diagonal FWHT
-> compute absmax -> quantize with per-row dynamic scale.

Input is the combined gate+up bf16 GEMV output with row stride `stride`.
Gate occupies columns [0, cols), up occupies columns [cols, 2*cols).
Output is i8 [seq_len, cols] in the Hadamard domain, ready for int8_gemv
with the down projection. Per-row absmax scales written to output array.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr

from kernels.kernel_ops import PoolFence
from simd_math import exp_f32
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8


# ============================================================================
# Row kernel — one row: silu(gate) * up -> FWHT -> dynamic i8
# ============================================================================


def silu_fwht_quantize_row[cols: Int, block: Int](
    gate: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    up: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
):
    """One row: silu(gate) * up -> FWHT -> dynamic i8.

    Computes per-row absmax after FWHT, writes it to scale_out, quantizes.
    """
    comptime width = simd_width_of[DType.float32]()

    # Fused SiLU(gate) * up -> f32 work buffer
    var k = 0
    while k + width <= cols:
        var g = (gate + k).load[width=width]().cast[DType.float32]()
        var u = (up + k).load[width=width]().cast[DType.float32]()
        var sig = SIMD[DType.float32, width](1.0) / (SIMD[DType.float32, width](1.0) + exp_f32[width](-g))
        (work + k).store(g * sig * u)
        k += width

    # Block-diagonal FWHT
    for b in range(cols // block):
        fwht_block[block](work + b * block)

    scale_out[] = absmax_quantize_i8[cols](work, row_qi)


# ============================================================================
# Worker args + kernel
# ============================================================================


@fieldwise_init
struct SiluFwhtArgs(Copyable, ImplicitlyCopyable):
    var gate_up_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var start_row: Int
    var end_row: Int


def silu_fwht_quantize_worker[cols: Int, stride: Int, block: Int](args: SiluFwhtArgs):
    var gate_up = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.gate_up_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=args.qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.work_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=args.scale_ptr)

    for m in range(args.start_row, args.end_row):
        silu_fwht_quantize_row[cols, block](
            gate_up + m * stride,
            gate_up + m * stride + cols,
            qi + m * cols,
            work, scales + m,
        )


# ============================================================================
# Dispatch
# ============================================================================


def silu_fwht_quantize[cols: Int, stride: Int, block: Int, P: BurstThreadPool](
    gate_up_ptr: Int, qi_ptr: Int, work_ptr: Int,
    scale_ptr: Int,
    seq_len: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch SiLU + FWHT + dynamic-scale quantize across workers.

    gate_up_ptr: bf16 [seq_len, stride] — gate at [0:cols], up at [cols:2*cols]
    qi_ptr:      i8   [seq_len, cols] output
    work_ptr:    f32  [cols] scratch per worker
    scale_ptr:   f32  [seq_len] output — per-row absmax scales for down GEMV dequant
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[SiluFwhtArgs, MAX_POOL_CAPACITY](
        fill=SiluFwhtArgs(0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = SiluFwhtArgs(
            gate_up_ptr, qi_ptr,
            work_ptr + i * cols * size_of[Float32](),
            scale_ptr, start, end)

    pool.dispatch[SiluFwhtArgs, silu_fwht_quantize_worker[cols, stride, block]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
