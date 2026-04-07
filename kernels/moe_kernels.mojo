"""Mixture-of-Experts routing and dispatch kernels."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import exp_f32
from kernels.kernel_ops import gemv_kernel, GemmArgs


# =============================================================================
# Router selection — SIMD softmax + greedy top-K
# =============================================================================


@fieldwise_init
struct TopKResult[k: Int]:
    """Selected expert indices and their gate values."""
    var indices: InlineArray[Int, Self.k]
    var gates: InlineArray[Float32, Self.k]


def softmax_topk[num_experts: Int, k: Int](
    logits_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> TopKResult[k]:
    """Softmax over num_experts logits, then greedy top-K selection.

    V2-Lite routing: softmax scoring, no gate renormalization.
    Returns raw softmax probabilities as gate values.

    SIMD softmax (vectorized max, exp, sum, normalize) then scalar top-K
    scan on the flat array.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = num_experts // width
    comptime assert num_experts % width == 0, "num_experts must be simd-aligned"
    comptime assert k <= num_experts, "k must be <= num_experts"

    # Load bf16 → f32 into stack-resident array
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

    # Vectorized normalize
    var inv_sum = SIMD[DType.float32, width](Float32(1.0) / sum_val)
    for c in range(chunks):
        var p = vp + c * width
        p.store(p.load[width=width]() * inv_sum)

    # Top-K: scalar scan
    var result = TopKResult[k](
        indices=InlineArray[Int, k](fill=0),
        gates=InlineArray[Float32, k](fill=Float32(0)),
    )

    for sel in range(k):
        var best_idx = 0
        var best_val = vals[0]
        for i in range(1, num_experts):
            if vals[i] > best_val:
                best_val = vals[i]
                best_idx = i
        result.indices[sel] = best_idx
        result.gates[sel] = best_val
        vals[best_idx] = Float32(-1.0)

    return result^


# =============================================================================
# Expert FFN kernel — runs as a single BurstPool job per expert
# =============================================================================


def expert_ffn_kernel[intermediate: Int, hidden: Int](
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gate_up_weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    down_weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gate_val: Float64,
):
    """Complete SwiGLU expert FFN for decode (seq_len=1).

    gate_up_weight points to [gate_proj | up_proj] contiguous: [2*intermediate, hidden].
    Composes gemv_kernel for projections, inline silu_mul between.
    Called as a BurstPool job — one worker, one expert, full FFN.
    """
    comptime width = simd_width_of[DType.float32]()

    # Fused gate+up GEMV: gate and up weights are adjacent in memory
    # [2*intermediate] = [gate_w | up_w][2*intermediate, hidden] @ input[hidden]
    comptime fused = 2 * intermediate
    var fused_act = InlineArray[Scalar[DType.bfloat16], fused](uninitialized=True)
    var fused_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=fused_act[0]))
    )

    gemv_kernel[hidden, fused](GemmArgs(input, gate_up_weight, fused_ptr, 0, fused, 1))

    # SiLU(gate[:intermediate]) * up[intermediate:] → write back to first half
    var gate_ptr = fused_ptr
    var up_ptr = fused_ptr + intermediate
    for i in range(0, intermediate, width):
        var g = (gate_ptr + i).load[width=width]().cast[DType.float32]()
        var u = (up_ptr + i).load[width=width]().cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp_f32[width](-g))
        (gate_ptr + i).store((g * sig * u).cast[DType.bfloat16]())

    # Down GEMV: [hidden] = down_w[hidden, intermediate] @ silu_result[intermediate]
    gemv_kernel[intermediate, hidden](GemmArgs(gate_ptr, down_weight, output, 0, hidden, 1))

    # Gate scaling
    var scale = Float32(gate_val)
    for i in range(0, hidden, width):
        var v = (output + i).load[width=width]().cast[DType.float32]()
        (output + i).store((v * scale).cast[DType.bfloat16]())
