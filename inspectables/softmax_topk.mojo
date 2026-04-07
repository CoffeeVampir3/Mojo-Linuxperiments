"""Assembly inspection for SIMD softmax + top-K router selection.

Self-contained. Inlines exp_f32 from simd_math to match kernels/moe_kernels.mojo exactly.
"""

from std.sys.info import simd_width_of
from std.benchmark import keep

comptime W = simd_width_of[DType.float32]()
comptime NUM_EXPERTS = 64
comptime K = 6
comptime CHUNKS = NUM_EXPERTS // W


# Inlined from simd_math/ops.mojo — must match exactly
@always_inline
def exp_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    comptime LN2_HI = Float32(0.693145751953125)
    comptime LN2_LO = Float32(1.4286068203094172e-06)
    comptime INV_LN2 = Float32(1.4426950408889634)
    comptime EXP_LO = Float32(-87.0)
    comptime EXP_HI = Float32(88.0)

    var lo_mask = ((x - EXP_LO).to_bits() >> 31) & 1
    var xc = x * (1 - lo_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_LO) * lo_mask.cast[DType.float32]()
    var hi_mask = ((EXP_HI - xc).to_bits() >> 31) & 1
    xc = xc * (1 - hi_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_HI) * hi_mask.cast[DType.float32]()

    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()

    var nf = n.cast[DType.float32]()
    var r = (xc - nf * LN2_HI) - nf * LN2_LO

    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999999995) + r * (
        Float32(0.5000000004) + r * (
        Float32(0.1666666456) + r * (
        Float32(0.04166685110) + r * (
        Float32(0.008333621758) + r * (
        Float32(0.001389404636)))))))

    var pow2n = SIMD[DType.float32, width](
        from_bits=(n + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n


@fieldwise_init
struct TopKResult(Movable):
    var indices: InlineArray[Int, K]
    var gates: InlineArray[Float32, K]


@no_inline
def softmax_topk_64_6(
    logits_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> TopKResult:
    comptime width = W

    # Load bf16 → f32 into stack-resident array
    var vals = InlineArray[Float32, NUM_EXPERTS](fill=Float32(0))
    var vp = UnsafePointer(to=vals[0])
    for c in range(CHUNKS):
        var v = (logits_ptr + c * width).load[width=width]().cast[DType.float32]()
        (vp + c * width).store(v)

    # Vectorized max
    var max_vec = vp.load[width=width]()
    for c in range(1, CHUNKS):
        max_vec = max(max_vec, (vp + c * width).load[width=width]())
    var max_val = max_vec.reduce_max()

    # Vectorized exp(x - max) and sum
    var bcast_max = SIMD[DType.float32, width](max_val)
    var sum_val = Float32(0)
    for c in range(CHUNKS):
        var p = vp + c * width
        var v = exp_f32(p.load[width=width]() - bcast_max)
        p.store(v)
        sum_val += v.reduce_add()

    # Vectorized normalize
    var inv_sum = SIMD[DType.float32, width](Float32(1.0) / sum_val)
    for c in range(CHUNKS):
        var p = vp + c * width
        p.store(p.load[width=width]() * inv_sum)

    # Top-K: scalar scan on the flat array — simple, branch-friendly, no SIMD overhead
    var result = TopKResult(
        indices=InlineArray[Int, K](fill=0),
        gates=InlineArray[Float32, K](fill=Float32(0)),
    )

    for sel in range(K):
        var best_idx = 0
        var best_val = vals[0]
        for i in range(1, NUM_EXPERTS):
            if vals[i] > best_val:
                best_val = vals[i]
                best_idx = i
        result.indices[sel] = best_idx
        result.gates[sel] = best_val
        vals[best_idx] = Float32(-1.0)

    return result^


def main():
    var logits = InlineArray[Scalar[DType.bfloat16], NUM_EXPERTS](fill=Scalar[DType.bfloat16](0))
    for i in range(NUM_EXPERTS):
        logits[i] = Scalar[DType.bfloat16](Float32(i) * 0.1 - 3.0)

    var ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=logits[0]))
    )
    var result = softmax_topk_64_6(ptr)

    keep(result.indices[0])
    keep(result.gates[0])
