"""Gemma 4 int8 MoE — router kernel, post-reduce combine.

Pool dispatchers (gemma4_moe_phase1, gemma4_moe_phase2, router_topk_dispatch)
live in experimental3.kernels.dispatch_kernels. This file holds the compute
bodies: the router kernel worker and the moe_combine post-reduce sequence.

Expert weights are block-sharded along the expert dimension: rank `r` owns
experts `[r*EPR, (r+1)*EPR)` where `EPR = num_experts // tp`. Local index
for expert `e` on its owning rank is `e - rank*EPR`. The flat
`[num_experts*M, K]` source tensor collapses to a contiguous row range
per rank, so the loader stores one contiguous slice per rank with no
strided reads. All memory is rank-local. Phase 1 inputs (act_i8/act_scale)
must be produced by the rank-local rmsnorm_dual_gamma_fwht_quantize.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import sqrt
from experimental_gemma.router import softmax_topk_renorm, Gemma4TopKResult
from experimental3.common_math import BF16Ptr
from experimental3.kernels.dispatch_args import RouterTopkArgs


# ============================================================================
# Router: softmax + top-k kernel
# ============================================================================


def router_topk_kernel[num_experts: Int, k: Int](args: RouterTopkArgs[k]):
    """Softmax → top-k → renormalize → per-expert scale."""
    var result = softmax_topk_renorm[num_experts, k](
        args.logits, args.per_expert_scale)
    args.result_ptr[] = result


# ============================================================================
# Post-allreduce combine
# ============================================================================


def moe_combine[hidden: Int](
    moe_out: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    moe_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dense_normed: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    combine_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    x_main: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    layer_scalar: Float32,
    eps: Float32,
):
    """Post-allreduce combine. All inline on 2816 elements.

    1. moe_normed = rmsnorm(moe_out, POST_FFN_NORM_2.γ)
    2. combined = moe_normed + dense_normed
    3. combined_normed = rmsnorm(combined, POST_FFN_NORM.γ)
    4. x_main = (x_main + combined_normed) * layer_scalar

    moe_out:       bf16[hidden] — allreduced MoE output (read, overwritten as scratch)
    dense_normed:  bf16[hidden] — pre-computed rmsnorm(dense_out, POST_FFN_NORM_1.γ)
    x_main:        bf16[hidden] — residual stream (read-modify-write)
    """
    comptime width = simd_width_of[DType.float32]()

    # Step 1: rmsnorm(moe_out, γ₂) → overwrite moe_out with moe_normed
    var sum_sq = SIMD[DType.float32, width](0)
    var i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (moe_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 2: combined = moe_normed + dense_normed (in-place into moe_out)
    i = 0
    while i + width <= hidden:
        var m = (moe_out + i).load[width=width]().cast[DType.float32]()
        var d = (dense_normed + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((m + d).cast[DType.bfloat16]())
        i += width

    # Step 3: rmsnorm(combined, γ₃)
    sum_sq = SIMD[DType.float32, width](0)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (combine_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 4: x_main = (x_main + combined_normed) * layer_scalar
    i = 0
    while i + width <= hidden:
        var x = (x_main + i).load[width=width]().cast[DType.float32]()
        var c = (moe_out + i).load[width=width]().cast[DType.float32]()
        (x_main + i).store(((x + c) * layer_scalar).cast[DType.bfloat16]())
        i += width
