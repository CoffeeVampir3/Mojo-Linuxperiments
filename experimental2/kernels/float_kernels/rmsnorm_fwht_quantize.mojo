"""RMSNorm + FWHT + fixed-scale i8 quantization.

Per row: bf16 -> f32 -> divide by RMS -> block-diagonal FWHT -> quantize
with fixed model-global scale S_act = C(n).

No gamma (absorbed into weights). No per-token adaptive scale.
Output is i8 in the Hadamard domain, ready for int8_gemv.

    qi[m, k] = clamp(round(FWHT(x[m] / rms(x[m])) * 127 / S_act), -128, 127)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.utils import IndexList
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from simd_math import sqrt, roundeven
from simd_math.matrixops import log2


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


def rmsnorm_fwht_quantize_row[cols: Int, block: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    quant_inv: Float32,
    eps: Float32,
):
    """One row: bf16 -> RMSNorm -> FWHT -> fixed-scale i8.

    quant_inv = 127.0 / S_act (precomputed by caller).
    work buffer must have at least cols f32 elements.
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

    # Fixed-scale quantize: qi = clamp(round(x * 127 / S_act), -128, 127)
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


def rmsnorm_fwht_quantize_worker[cols: Int, block: Int](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    quant_inv_bits: Int, start_row: Int, end_row: Int,
):
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=in_ptr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_ptr)
    var work = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=work_ptr)
    var qi_bits = Int32(quant_inv_bits)
    var quant_inv = UnsafePointer(to=qi_bits).bitcast[Float32]()[]

    for m in range(start_row, end_row):
        rmsnorm_fwht_quantize_row[cols, block](
            inp + m * cols, qi + m * cols, work, quant_inv, 1e-5,
        )


# ============================================================================
# Dispatch
# ============================================================================


def rmsnorm_fwht_quantize[cols: Int, block: Int, P: BurstThreadPool](
    in_ptr: Int, qi_ptr: Int, work_ptr: Int,
    seq_len: Int,
    s_act: Float32,
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch RMSNorm + FWHT + fixed-scale quantize.

    in_ptr:   bf16 [seq_len, cols]
    qi_ptr:   i8   [seq_len, cols] output
    work_ptr: f32  [cols] scratch per worker (caller provides cols * num_workers f32)
    s_act:    model-global activation scale C(n)
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    var quant_inv = Float32(127) / s_act
    var qi_copy = quant_inv
    var quant_inv_bits = Int(UnsafePointer(to=qi_copy).bitcast[Int32]()[])

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.get_args_base() + i
        pack[].arg0 = in_ptr
        pack[].arg1 = qi_ptr
        pack[].arg2 = work_ptr + i * cols * size_of[Float32]()
        pack[].arg3 = quant_inv_bits
        pack[].arg4 = start
        pack[].arg5 = end

    pool.dispatch(
        rmsnorm_fwht_quantize_worker[cols, block],
        pool.get_args_base(), num_jobs,
    )
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
