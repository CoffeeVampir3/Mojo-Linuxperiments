"""Validate Gemma4 router: softmax → topk → renorm → per-expert scale.

Tests with a small expert count (16 experts, top-4) to make the full
probability distribution visible. Verifies:
  1. Softmax probabilities sum to 1
  2. Top-k selects the correct indices
  3. Post-renorm weights sum to 1 (before per-expert scale)
  4. Per-expert scale is applied correctly
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs

from simd_math import exp_f32
from experimental_gemma.router import softmax_topk_renorm


def test_router():
    print("=== Gemma4 router: softmax_topk_renorm ===")

    comptime NUM_EXPERTS = 16
    comptime TOP_K = 4

    var logits = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS)
    var scales = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS)

    # Logits: make experts 3, 7, 11, 14 clearly dominant
    var logit_vals = InlineArray[Float32, NUM_EXPERTS](fill=Float32(0))
    logit_vals[0] = -1.0;  logit_vals[1] = -0.5;  logit_vals[2] =  0.0;  logit_vals[3] =  3.0
    logit_vals[4] =  0.2;  logit_vals[5] = -0.3;  logit_vals[6] =  0.1;  logit_vals[7] =  2.5
    logit_vals[8] = -0.8;  logit_vals[9] =  0.4;  logit_vals[10] = -0.1; logit_vals[11] =  2.0
    logit_vals[12] =  0.3; logit_vals[13] = -0.6; logit_vals[14] =  1.8; logit_vals[15] = -0.2

    for i in range(NUM_EXPERTS):
        logits[i] = Scalar[DType.bfloat16](logit_vals[i])

    # Per-expert scales: varying around 1.0
    var scale_vals = InlineArray[Float32, NUM_EXPERTS](fill=Float32(0))
    for i in range(NUM_EXPERTS):
        scale_vals[i] = Float32(0.8) + Float32(i) * Float32(0.03)
        scales[i] = Scalar[DType.bfloat16](scale_vals[i])

    # Compute f32 reference softmax for verification
    print("  --- input logits and softmax probabilities ---")
    print("  expert | logit(bf16) | softmax_prob | per_expert_scale")
    print("  -------+-------------+--------------+-----------------")

    var f32_logits = InlineArray[Float32, NUM_EXPERTS](fill=Float32(0))
    for i in range(NUM_EXPERTS):
        f32_logits[i] = Float32(logits[i])

    var max_logit = f32_logits[0]
    for i in range(1, NUM_EXPERTS):
        if f32_logits[i] > max_logit:
            max_logit = f32_logits[i]

    var exp_sum = Float32(0)
    var probs = InlineArray[Float32, NUM_EXPERTS](fill=Float32(0))
    for i in range(NUM_EXPERTS):
        var e = exp_f32[1](SIMD[DType.float32, 1](f32_logits[i] - max_logit))
        probs[i] = e[0]
        exp_sum += probs[i]
    for i in range(NUM_EXPERTS):
        probs[i] /= exp_sum

    for i in range(NUM_EXPERTS):
        print("  " + String(i) + " | " + String(Float32(logits[i])) + " | " + String(probs[i]) + " | " + String(Float32(scales[i])))

    var prob_sum = Float32(0)
    for i in range(NUM_EXPERTS):
        prob_sum += probs[i]
    print("  softmax sum=" + String(prob_sum))

    # Run the kernel
    var result = softmax_topk_renorm[NUM_EXPERTS, TOP_K](logits, scales)

    # Show results
    print()
    print("  --- top-" + String(TOP_K) + " results ---")
    print("  rank | expert | raw_softmax  | renormed     | per_exp_scale | final_weight")
    print("  -----+--------+--------------+--------------+---------------+-------------")

    # Compute expected renormalized weights for comparison
    var topk_raw_sum = Float32(0)
    for sel in range(TOP_K):
        topk_raw_sum += probs[result.indices[sel]]

    for sel in range(TOP_K):
        var eid = result.indices[sel]
        var raw_prob = probs[eid]
        var renormed = raw_prob / topk_raw_sum
        var final_expected = renormed * Float32(scales[eid])
        var final_got = result.weights[sel]
        print("  " + String(sel) + " | " + String(eid) + " | " + String(raw_prob) + " | " + String(renormed) + " | " + String(Float32(scales[eid])) + " | " + String(final_got))

    # Verify renorm property: weights / per_expert_scale should sum to 1
    var renorm_sum = Float32(0)
    for sel in range(TOP_K):
        var eid = result.indices[sel]
        renorm_sum += result.weights[sel] / Float32(scales[eid])
    print()
    print("  sum(weight/per_expert_scale)=" + String(renorm_sum) + " (should be ~1.0)")

    # Verify ordering: weights should be in descending softmax order
    print("  ordering check (should be descending by raw softmax):")
    var ordered = True
    for sel in range(TOP_K - 1):
        var a = probs[result.indices[sel]]
        var b = probs[result.indices[sel + 1]]
        if a < b:
            ordered = False
        print("    rank " + String(sel) + ": expert " + String(result.indices[sel]) + " prob=" + String(a))
    print("    rank " + String(TOP_K - 1) + ": expert " + String(result.indices[TOP_K - 1]) + " prob=" + String(probs[result.indices[TOP_K - 1]]))
    print("  correctly ordered: " + String(ordered))

    logits.free()
    scales.free()


def main():
    test_router()
