"""Self-contained rms_fwht_quantize for assembly inspection."""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from std.utils import IndexList
from std.benchmark import keep


# --- simd_math intrinsics ---

@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic["llvm.sqrt", SIMD[dtype, width], SIMD[dtype, width]](x)

@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic["llvm.nearbyint", SIMD[dtype, width], SIMD[dtype, width]](x)


# --- FWHT comptime helpers ---

def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()

def butterfly_partner[i: Int, stride: Int]() -> Int:
    return i ^ stride

def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result

def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


# --- FWHT ---

@always_inline
def fwht_block[T: DType, block: Int](
    buf: UnsafePointer[Scalar[T], MutAnyOrigin],
):
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    var r = InlineArray[SIMD[T, width], regs](fill=SIMD[T, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()

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

    comptime for i in range(regs):
        (buf + i * width).store(r[i])


@always_inline
def fwht_row[T: DType, block: Int](
    buf: UnsafePointer[Scalar[T], MutAnyOrigin], cols: Int,
):
    for b in range(cols // block):
        fwht_block[T, block](buf + b * block)


# --- rms_fwht_quantize: single row ---

@no_inline
def rms_fwht_quantize_row[block: Int, cols: Int](
    row_in: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    scale_out: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    eps: Float32,
):
    comptime width = simd_width_of[DType.float32]()

    # Load bf16 → f32
    var k = 0
    while k + width <= cols:
        (work + k).store(
            (row_in + k).load[width=width]().cast[DType.float32]()
        )
        k += width
    while k < cols:
        work[k] = Float32(row_in[k])
        k += 1

    # FWHT
    fwht_row[DType.float32, block](work, cols)

    # Dual reduction: sum-of-squares + absmax
    var vsum = SIMD[DType.float32, width](0)
    var vmax = SIMD[DType.float32, width](0)
    k = 0
    while k + width <= cols:
        var v = (work + k).load[width=width]()
        vsum = v.fma(v, vsum)
        vmax = max(vmax, v.__abs__())
        k += width
    var sum_sq = vsum.reduce_add()
    var absmax = vmax.reduce_max()
    while k < cols:
        var v = work[k]
        sum_sq += v * v
        var a = v if v >= 0 else -v
        if a > absmax:
            absmax = a
        k += 1

    # Scale: rms folded in
    var rms = sqrt[DType.float32, 1](sum_sq / Float32(cols) + eps)
    scale_out[0] = absmax / (rms * Float32(127.0))

    # Quantize: standard absmax int8
    var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
    var vinv = SIMD[DType.float32, width](inv)
    comptime vlo = SIMD[DType.float32, width](-128.0)
    comptime vhi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= cols:
        var v = (work + k).load[width=width]()
        var q = min(max(roundeven(v * vinv), vlo), vhi)
        (row_qi + k).store(q.cast[DType.int8]())
        k += width
    while k < cols:
        var v = roundeven[DType.float32, 1](work[k] * inv)
        var q = min(max(v, Float32(-128.0)), Float32(127.0))
        row_qi[k] = q.cast[DType.int8]()
        k += 1


# --- entry ---

def main():
    comptime COLS = 576
    comptime BLOCK = 64

    var bf16_in = alloc[Scalar[DType.bfloat16]](COLS)
    var qi_out = alloc[Scalar[DType.int8]](COLS)
    var scale = alloc[Float32](1)
    var work = alloc[Float32](COLS)

    for i in range(COLS):
        bf16_in[i] = Scalar[DType.bfloat16](Float32(i) / 576.0 - 0.5)

    rms_fwht_quantize_row[BLOCK, COLS](bf16_in, qi_out, scale, work, 1e-5)

    keep(scale[0])
    keep(qi_out[0])

    bf16_in.free()
    qi_out.free()
    scale.free()
    work.free()
