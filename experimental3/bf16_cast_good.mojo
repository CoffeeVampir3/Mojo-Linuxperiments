"""Minimal repro: bf16→f32 via manual bit-shift.
bf16 is f32 with the low 16 bits truncated, so:
  zero-extend u16→u32 then shift left 16 recovers the f32 bits.
Generates: vpmovzxwd + vpslld (2 instructions)."""

from memory import UnsafePointer
from benchmark import keep


@no_inline
fn bf16_to_f32(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    var wide = ptr.bitcast[Scalar[DType.uint16]]().load[width=8]().cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=8]()


fn main():
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=0x1000000)
    keep(bf16_to_f32(p))
