"""MiniMax M2.7 sigmoid router: sigmoid + correction bias + top-k + renormalize.

MiniMax routing (differs from Gemma4 softmax routing):
  1. sigmoid(logits) -> weights
  2. weights + correction_bias -> selection scores
  3. top-k from selection scores
  4. gather raw sigmoid weights at selected indices (NOT biased scores)
  5. renormalize selected weights to sum=1

No per-expert learned scale (unlike Gemma4).
"""

from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.common_math import F32Ptr
from minimax.kernels.activations import sigmoid_f32


@fieldwise_init
struct TopKResult[k: Int](Copyable, ImplicitlyCopyable, Movable):
    var indices: InlineArray[Int, Self.k]
    var weights: InlineArray[Float32, Self.k]


def sigmoid_topk_renorm[num_experts: Int, k: Int](
    logits_ptr: F32Ptr,
    correction_bias_ptr: F32Ptr,
) -> TopKResult[k]:
    """Sigmoid routing: sigmoid -> bias-shifted selection -> top-k -> renorm.

    logits_ptr:          f32[num_experts] router projection output
    correction_bias_ptr: f32[num_experts] learned bias for expert selection

    Returns top-k (index, weight) pairs where weights are the raw sigmoid
    values (without bias), renormalized to sum=1.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = num_experts // width
    comptime assert num_experts % width == 0, "num_experts must be simd-aligned"
    comptime assert k <= num_experts, "k must be <= num_experts"

    var weights = InlineArray[Float32, num_experts](fill=Float32(0))
    var scores = InlineArray[Float32, num_experts](fill=Float32(0))
    var wp = UnsafePointer(to=weights[0])
    var sp = UnsafePointer(to=scores[0])

    for c in range(chunks):
        var logits = (logits_ptr + c * width).load[width=width]()
        var w = sigmoid_f32(logits)
        (wp + c * width).store(w)
        var bias = (correction_bias_ptr + c * width).load[width=width]()
        (sp + c * width).store(w + bias)

    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )

    for sel in range(k):
        var best_idx = 0
        var best_val = scores[0]
        for i in range(1, num_experts):
            if scores[i] > best_val:
                best_val = scores[i]
                best_idx = i
        result.indices[sel] = best_idx
        result.weights[sel] = weights[best_idx]
        scores[best_idx] = Float32(-1.0)

    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv_topk = 1.0 / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv_topk

    return result^
