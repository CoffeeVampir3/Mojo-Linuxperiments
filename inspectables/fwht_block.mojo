"""Minimal FWHT SIMD assembly inspection. Self-contained, no imports from project."""

from std.memory.unsafe_pointer import alloc
from std.collections import InlineArray
from std.utils import IndexList
from std.sys.info import simd_width_of
from std.math import sqrt
from std.benchmark import keep


# --- comptime helpers (from simd_math/matrixops) ---

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


# --- FWHT core ---

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

    var sc = Scalar[T](1.0 / sqrt(Float64(block)))
    comptime for i in range(regs):
        r[i] = r[i] * sc

    comptime for i in range(regs):
        (buf + i * width).store(r[i])


# --- entry ---

def main():
    var buf = alloc[Float32](64)
    for i in range(64):
        buf[i] = Float32(i)

    fwht_block[DType.float32, 64](buf)

    keep(buf[0])
    buf.free()
