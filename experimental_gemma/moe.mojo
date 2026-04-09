"""Gemma4 MoE expert FFN kernel and dispatch.

Expert FFN per expert (GELU-tanh, not SiLU):
  1. gate_up = input @ gate_up_proj[e]^T → [2*intermediate]
  2. Split: gate = first half, up = second half
  3. activated = gelu_tanh(gate) * up → [intermediate]
  4. output = activated @ down_proj[e]^T * routing_weight → [hidden]

Dispatch: expert FFNs as individual BurstPool jobs, then accumulate.
No shared experts (dense MLP handled separately in forward pass).
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.kernel_ops import gemv_kernel, GemmArgs, BF16Ptr
from experimental_gemma.activations import gelu_tanh_f32
from experimental_gemma.router import Gemma4TopKResult
from threading.threading_traits import BurstThreadPool


# =============================================================================
# Dispatch arg struct
# =============================================================================


@fieldwise_init
struct Gemma4ExpertFFNArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var gate_up: BF16Ptr
    var down: BF16Ptr
    var output: BF16Ptr
    var routing_weight: Float32


# =============================================================================
# Expert FFN kernel
# =============================================================================


def gemma4_expert_ffn_kernel[intermediate: Int, hidden: Int](
    args: Gemma4ExpertFFNArgs,
):
    """GELU-tanh expert FFN for decode (seq_len=1).

    gate_up: [gate_proj | up_proj] contiguous [2*intermediate, hidden].
    First intermediate rows are gate, second intermediate rows are up.
    Writes routing-weight-scaled [hidden] output.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime gate_up_dim = 2 * intermediate

    var inp = args.input
    var gate_up_w = args.gate_up
    var down_w = args.down
    var out = args.output
    var scale = args.routing_weight

    # Gate+up GEMV: [2*intermediate] = gate_up_w @ input
    var fused_buf = InlineArray[Scalar[DType.bfloat16], gate_up_dim](uninitialized=True)
    var fused_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=fused_buf[0])))

    gemv_kernel[hidden, gate_up_dim](GemmArgs(inp, gate_up_w, fused_ptr, 0, gate_up_dim, 1))

    # Split + gelu_tanh(gate) * up → first half (in-place)
    var gate_ptr = fused_ptr
    var up_ptr = fused_ptr + intermediate
    for i in range(0, intermediate, width):
        var g = (gate_ptr + i).load[width=width]().cast[DType.float32]()
        var u = (up_ptr + i).load[width=width]().cast[DType.float32]()
        (gate_ptr + i).store((gelu_tanh_f32(g) * u).cast[DType.bfloat16]())

    # Down GEMV with routing weight scaling
    for n in range(hidden):
        var dw_row = down_w + n * intermediate
        var acc = SIMD[DType.float32, width](0)
        for k in range(0, intermediate, width):
            var x = (gate_ptr + k).load[width=width]().cast[DType.float32]()
            var w = (dw_row + k).load[width=width]().cast[DType.float32]()
            acc = x.fma(w, acc)
        out[n] = Scalar[DType.bfloat16](acc.reduce_add() * scale)


# =============================================================================
# MoE dispatch
# =============================================================================


def gemma4_moe_dispatch[
    num_experts: Int,
    top_k: Int,
    intermediate: Int,
    hidden: Int,
    P: BurstThreadPool,
](
    expert_input: BF16Ptr,
    routing: Gemma4TopKResult[top_k],
    expert_gate_up_base: BF16Ptr,
    expert_down_base: BF16Ptr,
    expert_out_buf: BF16Ptr,
    dst: BF16Ptr,
    mut pool: P,
):
    """Dispatch top-k expert FFNs and accumulate outputs.

    expert_input:         [hidden] normed input for expert computation
    routing:              top-k indices and weights from softmax_topk_renorm
    expert_gate_up_base:  base of folded [num_experts*2*intermediate, hidden]
    expert_down_base:     base of folded [num_experts*hidden, intermediate]
    expert_out_buf:       scratch [top_k * hidden] for per-expert outputs
    dst:                  [hidden] output (sum of weighted expert outputs)
    """
    comptime gate_up_dim = 2 * intermediate
    comptime gate_up_expert_stride = gate_up_dim * hidden
    comptime down_expert_stride = hidden * intermediate

    var jobs = InlineArray[Gemma4ExpertFFNArgs, top_k](uninitialized=True)
    for s in range(top_k):
        var eid = routing.indices[s]
        jobs[s] = Gemma4ExpertFFNArgs(
            expert_input,
            expert_gate_up_base + eid * gate_up_expert_stride,
            expert_down_base + eid * down_expert_stride,
            expert_out_buf + s * hidden,
            routing.weights[s],
        )

    pool.dispatch[Gemma4ExpertFFNArgs, gemma4_expert_ffn_kernel[intermediate, hidden]](
        UnsafePointer(to=jobs[0]), top_k)
    pool.join()

    # Accumulate expert outputs in f32, single pass
    comptime width = simd_width_of[DType.float32]()
    for i in range(0, hidden, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(top_k):
            acc += (expert_out_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
