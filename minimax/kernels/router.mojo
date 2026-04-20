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
from simd_math.matrixops import reduce_top_k, fill_lane_iota
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

    Argmax selection uses simd_math.matrixops.reduce_top_k — a tagged
    butterfly reduction over a SIMD score bank, log2(regs)+log2(width)
    stages per extraction. Bit-identical to a scalar leftmost-wins scan
    on any input with nonzero top-k gaps (see router_simd_verification.mojo
    for the correctness/fuzz harness).
    """
    comptime width = simd_width_of[DType.float32]()
    comptime regs = num_experts // width
    comptime assert num_experts % width == 0, "num_experts must be simd-aligned"
    comptime assert k <= num_experts, "k must be <= num_experts"

    # Raw sigmoid weights retained in memory for scatter-gather after top-k.
    var weights = InlineArray[Float32, num_experts](uninitialized=True)
    var wp = UnsafePointer(to=weights[0])

    # Score bank (sigmoid + bias) lives in SIMD registers, tagged with the
    # per-lane global expert index.
    var score_regs = InlineArray[SIMD[DType.float32, width], regs](
        fill=SIMD[DType.float32, width](0))
    var index_regs = InlineArray[SIMD[DType.int32, width], regs](
        fill=SIMD[DType.int32, width](0))
    fill_lane_iota[width, regs](index_regs)

    comptime for r in range(regs):
        var logits = (logits_ptr + r * width).load[width=width]()
        var w = sigmoid_f32(logits)
        (wp + r * width).store(w)
        var bias = (correction_bias_ptr + r * width).load[width=width]()
        score_regs[r] = w + bias

    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        weights=InlineArray[Float32, k](fill=Float32(0)),
    )
    # reduce_top_k writes winners in descending-score order; values are
    # discarded (we renormalize the raw sigmoid weights, not the biased
    # scores).
    var topk_scores = InlineArray[Float32, k](fill=Float32(0))
    reduce_top_k[DType.float32, width, regs, k](
        score_regs, index_regs,
        Float32(-1e30),
        result.indices, topk_scores)

    for sel in range(k):
        result.weights[sel] = weights[result.indices[sel]]
    var topk_sum = Float32(0)
    for sel in range(k):
        topk_sum += result.weights[sel]
    var inv_topk = 1.0 / topk_sum
    for sel in range(k):
        result.weights[sel] *= inv_topk

    return result^
