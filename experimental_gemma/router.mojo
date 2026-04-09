"""Gemma4 MoE router: softmax, top-k, renormalize, per-expert scale.

Gemma4 routing (differs from DeepSeek V2):
  1. rms_norm_no_scale(input)
  2. Multiply by learnable scale * (1/sqrt(hidden))
  3. Project to num_experts logits
  4. Softmax over all experts
  5. Top-8 selection
  6. Renormalize top-8 weights to sum=1
  7. Multiply by per-expert learned scale

The router projection (step 3) uses the standard gemv kernel.
This module handles steps 4-7: softmax_topk_renorm.
Steps 1-2 are composed from existing kernels in the forward pass.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32


@fieldwise_init
struct Gemma4TopKResult[k: Int](Copyable, ImplicitlyCopyable, Movable):
    var indices: InlineArray[Int, Self.k]
    var weights: InlineArray[Float32, Self.k]


def softmax_topk_renorm[num_experts: Int, k: Int](
    logits_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    per_expert_scale_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Gemma4TopKResult[k]:
    """Gemma4 routing: softmax → top-k → renormalize → per-expert scale.

    logits_ptr: [num_experts] bf16 router output
    per_expert_scale_ptr: [num_experts] bf16 learned per-expert multiplier

    Returns top-k (index, weight) pairs where weight includes
    the per-expert scale and sums to per_expert_scale[selected].
    """
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = num_experts // width
    comptime assert num_experts % width == 0, "num_experts must be simd-aligned"
    comptime assert k <= num_experts, "k must be <= num_experts"

    # Load bf16 logits → f32 into stack array
    var vals = InlineArray[Float32, num_experts](fill=Float32(0))
    var vp = UnsafePointer(to=vals[0])
    for c in range(chunks):
        var v = (logits_ptr + c * width).load[width=width]().cast[DType.float32]()
        (vp + c * width).store(v)

    # Vectorized max
    var max_vec = vp.load[width=width]()
    for c in range(1, chunks):
        max_vec = max(max_vec, (vp + c * width).load[width=width]())
    var max_val = max_vec.reduce_max()

    # Vectorized exp(x - max) and sum
    var bcast_max = SIMD[DType.float32, width](max_val)
    var sum_val = Float32(0)
    for c in range(chunks):
        var p = vp + c * width
        var v = exp_f32(p.load[width=width]() - bcast_max)
        p.store(v)
        sum_val += v.reduce_add()

    # Vectorized normalize (softmax complete)
    var inv_sum = SIMD[DType.float32, width](Float32(1.0) / sum_val)
    for c in range(chunks):
        var p = vp + c * width
        p.store(p.load[width=width]() * inv_sum)

    # Top-K: greedy scan
    var result = Gemma4TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )
    for sel in range(k):
        var best_idx = 0
        var best_val = vals[0]
        for i in range(1, num_experts):
            if vals[i] > best_val:
                best_val = vals[i]
                best_idx = i
        result.indices[sel] = best_idx
        result.weights[sel] = best_val
        vals[best_idx] = Float32(-1.0)

    # Renormalize top-k weights to sum=1
    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv_topk = Float32(1.0) / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv_topk

    # Apply per-expert learned scale
    for sel in range(k):
        var eid = result.indices[sel]
        var scale = Float32(per_expert_scale_ptr[eid])
        result.weights[sel] *= scale

    return result^
