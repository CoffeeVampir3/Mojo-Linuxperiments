"""Investigate the +990-byte overhead in bf16-only unified paths.

Hypothesis: comptime branching leaves dead-code stubs that LLVM doesn't DCE.
If hot-loop instructions match, the overhead is cold-path / metadata only.
"""
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from compile import compile_info

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq, rms_reduce_bf16,
)


@always_inline
def existing_bf16_row[cols: Int, has_gamma: Bool, has_residual: Bool](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce_bf16[cols](src)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
        var v = (src + k).load[width=width]().cast[DType.float32]()
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            var normed = v * vinv * g
            comptime if has_residual:
                var x = (dst + k).load[width=width]().cast[DType.float32]()
                (dst + k).store((x + normed).cast[DType.bfloat16]())
            else:
                (dst + k).store(normed.cast[DType.bfloat16]())
        else:
            (dst + k).store((v * vinv).cast[DType.bfloat16]())
        k += width


def E7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    existing_bf16_row[cols, True, False](src, gamma, dst, Float32(1e-6))


def main():
    var asm = String(compile_info[E7[3072], emission_kind="asm"]().asm)
    print("== EXISTING bf16+gamma asm (length:", asm.byte_length(), ") ==")
    print(asm)
