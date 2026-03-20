"""Minimal repro: bf16→f32 cast generates scalar decomposition.
Expected: vpmovzxwd + vpslld (2 instructions).
Actual: ~35 scalar vpextrw/shl/vmovd/vinsertps instructions."""

from memory import UnsafePointer
from benchmark import keep


@no_inline
fn bf16_to_f32(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    return ptr.load[width=8]().cast[DType.float32]()


fn main():
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=0x1000000)
    keep(bf16_to_f32(p))
