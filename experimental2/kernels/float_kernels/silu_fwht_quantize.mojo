"""SiLU + FWHT + fixed-scale i8 quantization (MLP domain exit).

Per row: bf16 gate, bf16 up -> f32 SiLU(gate) * up -> block-diagonal FWHT
-> quantize with fixed per-layer scale S_post.

Input is the combined gate+up bf16 GEMV output with row stride `stride`.
Gate occupies columns [0, cols), up occupies columns [cols, 2*cols).
Output is i8 [seq_len, cols] in the Hadamard domain, ready for int8_gemv
with the down projection.

    qi[m, k] = clamp(round(FWHT(silu(gate[m]) * up[m]) * 127 / S_post), -128, 127)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from simd_math import exp_f32, roundeven
from experimental2.kernels.float_kernels.rmsnorm_fwht_quantize import fwht_block


# ============================================================================
# Row kernel — one row: silu(gate) * up -> FWHT -> fixed-scale i8
# ============================================================================


def silu_fwht_quantize_row[cols: Int, block: Int](
    gate: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    up: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    quant_inv: Float32,
):
    """One row: silu(gate) * up -> FWHT -> fixed-scale i8.

    quant_inv = 127.0 / S_post (precomputed by caller).
    work buffer must have at least cols f32 elements.
    """
    comptime width = simd_width_of[DType.float32]()

    # Fused SiLU(gate) * up -> f32 work buffer
    # SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
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

    # Fixed-scale quantize: qi = clamp(round(x * 127 / S_post), -128, 127)
    var vinv = SIMD[DType.float32, width](quant_inv)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= cols:
        var v = (work + k).load[width=width]()
        var qi = min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]()
        (row_qi + k).store(qi)
        k += width


# ============================================================================
# Worker kernel (BurstPool ABI: 6 Int args)
# ============================================================================


def silu_fwht_quantize_worker[cols: Int, stride: Int, block: Int](
    gate_up_ptr: Int, qi_ptr: Int, work_ptr: Int,
    quant_inv_bits: Int, start_row: Int, end_row: Int,
):
    var gate_up = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=gate_up_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=work_ptr)
    var qi_bits = Int32(quant_inv_bits)
    var quant_inv = UnsafePointer(to=qi_bits).bitcast[Float32]()[]

    for m in range(start_row, end_row):
        silu_fwht_quantize_row[cols, block](
            gate_up + m * stride,
            gate_up + m * stride + cols,
            qi + m * cols,
            work, quant_inv,
        )


# ============================================================================
# Dispatch
# ============================================================================


def silu_fwht_quantize[cols: Int, stride: Int, block: Int, P: BurstThreadPool](
    gate_up_ptr: Int, qi_ptr: Int, work_ptr: Int,
    seq_len: Int,
    s_post: Float32,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch SiLU + FWHT + fixed-scale quantize across workers.

    gate_up_ptr: bf16 [seq_len, stride] — gate at [0:cols], up at [cols:2*cols]
    qi_ptr:      i8   [seq_len, cols] output
    work_ptr:    f32  [cols] scratch per worker (caller provides cols * num_workers f32)
    s_post:      per-layer post-nonlinearity scale
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    var quant_inv = Float32(127) / s_post
    var qi_copy = quant_inv
    var quant_inv_bits = Int(UnsafePointer(to=qi_copy).bitcast[Int32]()[])

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.get_args_base() + i
        pack[].arg0 = gate_up_ptr
        pack[].arg1 = qi_ptr
        pack[].arg2 = work_ptr + i * cols * size_of[Float32]()
        pack[].arg3 = quant_inv_bits
        pack[].arg4 = start
        pack[].arg5 = end

    pool.dispatch(
        silu_fwht_quantize_worker[cols, stride, block],
        pool.get_args_base(), num_jobs,
    )
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
