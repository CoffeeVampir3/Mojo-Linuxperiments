"""Mixture-of-Experts routing and dispatch kernels.

Two-phase MoE dispatch for decode:
  Phase 1: Routed expert FFNs (parallel, per-expert output buffers)
           + shared expert gate/up GEMVs (sharded)
  Phase 2: Fused silu_mul + shared down_proj + routed buffer sum
           into x_residual (one dispatch, all NUMA-local)
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32
from kernels.kernel_ops import gemv_kernel, GemmArgs
from threading.threading_traits import BurstThreadPool

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# =============================================================================
# Dispatch arg structs
# =============================================================================


@fieldwise_init
struct ExpertFFNArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var gate_up: BF16Ptr
    var down: BF16Ptr
    var output: BF16Ptr
    var gate_val: Float32
    var intermediate: Int
    var hidden: Int


@fieldwise_init
struct MoECombineArgs(Copyable, ImplicitlyCopyable):
    var shared_gate: BF16Ptr
    var shared_up: BF16Ptr
    var shared_down_w: BF16Ptr
    var expert_bufs: BF16Ptr
    var dst: BF16Ptr
    var num_experts: Int
    var hidden: Int
    var shared_intermediate: Int


# =============================================================================
# Router selection — SIMD softmax + greedy top-K
# =============================================================================


@fieldwise_init
struct TopKResult[k: Int](Movable):
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


def expert_ffn_kernel[intermediate: Int, hidden: Int](args: ExpertFFNArgs):
    """Complete SwiGLU expert FFN for decode (seq_len=1).

    gate_up: [gate_proj | up_proj] contiguous [2*intermediate, hidden].
    Writes gate-scaled [hidden] output (overwrite, not accumulate).
    """
    comptime width = simd_width_of[DType.float32]()
    comptime fused = 2 * intermediate

    var inp = args.input
    var gate_up = args.gate_up
    var down = args.down
    var out = args.output
    var scale = args.gate_val

    # Fused gate+up GEMV: [2*intermediate] = [gate_w | up_w] @ input
    var fused_act = InlineArray[Scalar[DType.bfloat16], fused](uninitialized=True)
    var fused_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=fused_act[0])))

    gemv_kernel[hidden, fused](GemmArgs(inp, gate_up, fused_ptr, 0, fused, 1))

    # SiLU(gate) * up -> first half
    var gate_ptr = fused_ptr
    var up_ptr = fused_ptr + intermediate
    for i in range(0, intermediate, width):
        var g = (gate_ptr + i).load[width=width]().cast[DType.float32]()
        var u = (up_ptr + i).load[width=width]().cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp_f32[width](-g))
        (gate_ptr + i).store((g * sig * u).cast[DType.bfloat16]())

    # Down GEMV with fused gate scaling: out[n] = bf16(dot(silu, down_row_n) * scale)
    var sv = SIMD[DType.float32, width](scale)
    for n in range(hidden):
        var dw_row = down + n * intermediate
        var acc = SIMD[DType.float32, width](0)
        for k in range(0, intermediate, width):
            var x = (gate_ptr + k).load[width=width]().cast[DType.float32]()
            var w = (dw_row + k).load[width=width]().cast[DType.float32]()
            acc = x.fma(w, acc)
        out[n] = Scalar[DType.bfloat16](acc.reduce_add() * scale)


# =============================================================================
# Fused combine kernel — silu_mul + shared down_proj + routed buffer sum
# =============================================================================


def moe_combine_kernel[shared_intermediate: Int, hidden: Int](
    args: MoECombineArgs,
):
    """Phase 2 of MoE dispatch: combine shared expert and routed expert outputs.

    1. silu_mul on shared gate/up intermediates
    2. Shared down_proj GEMV (ColShard partial) -> write to dst
    3. Sum routed expert output buffers -> add to dst

    All data is NUMA-local.
    """
    comptime width = simd_width_of[DType.float32]()

    var sg = args.shared_gate
    var su = args.shared_up
    var sdw = args.shared_down_w
    var expert_bufs = args.expert_bufs
    var dst = args.dst
    var n_experts = args.num_experts

    # silu_mul in-place on shared gate buffer
    for i in range(0, shared_intermediate, width):
        var g = (sg + i).load[width=width]().cast[DType.float32]()
        var u = (su + i).load[width=width]().cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp_f32[width](-g))
        (sg + i).store((g * sig * u).cast[DType.bfloat16]())

    # Shared down_proj: dst = down_w @ silu_result (overwrite)
    gemv_kernel[shared_intermediate, hidden](
        GemmArgs(sg, sdw, dst, 0, hidden, 1))

    # Sum routed expert buffers into dst (accumulate)
    for e in range(n_experts):
        var ebuf = expert_bufs + e * hidden
        for i in range(0, hidden, width):
            var existing = (dst + i).load[width=width]().cast[DType.float32]()
            var expert_val = (ebuf + i).load[width=width]().cast[DType.float32]()
            (dst + i).store((existing + expert_val).cast[DType.bfloat16]())


# =============================================================================
# MoE block orchestration — two-phase dispatch
# =============================================================================


def moe_dispatch[
    num_experts: Int,
    top_k: Int,
    expert_intermediate: Int,
    shared_intermediate: Int,
    hidden: Int,
    tp: Int,
    P: BurstThreadPool,
](
    input: BF16Ptr,
    router_weight: BF16Ptr,
    expert_base: Int,
    expert_stride: Int,
    shared_gate_w: BF16Ptr,
    shared_up_w: BF16Ptr,
    shared_down_w: BF16Ptr,
    shared_gate_buf: BF16Ptr,
    shared_up_buf: BF16Ptr,
    expert_out_buf: BF16Ptr,
    dst: BF16Ptr,
    rank: Int,
    mut pool: P,
):
    """Two-phase MoE dispatch for decode (seq_len=1).

    input:           post-norm hidden state [hidden] (local replica)
    router_weight:   router centroids [num_experts, hidden]
    expert_base:     base address of this rank's expert weight region
    expert_stride:   bytes between consecutive experts
    shared_*_w:      shared expert weight pointers (RowShard/ColShard)
    shared_gate_buf: scratch for shared gate intermediate [shared_intermediate/tp]
    shared_up_buf:   scratch for shared up intermediate [shared_intermediate/tp]
    expert_out_buf:  scratch for per-expert outputs [top_k * hidden]
    dst:             x_residual [hidden] (accumulation target, overwritten)
    rank:            this rank's index
    """
    comptime width = simd_width_of[DType.float32]()
    comptime local_shared = shared_intermediate // tp

    # --- Router (inline, host-side) ---
    var router_out = InlineArray[Scalar[DType.bfloat16], num_experts](
        fill=Scalar[DType.bfloat16](0))
    var router_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=router_out[0])))
    gemv_kernel[hidden, num_experts](
        GemmArgs(input, router_weight, router_ptr, 0, num_experts, 1))

    var routing = softmax_topk[num_experts, top_k](router_ptr)

    # --- Group selected experts by rank ---
    var local_count = 0
    var local_expert_ids = InlineArray[Int, top_k](fill=0)
    var local_gates = InlineArray[Float32, top_k](fill=Float32(0))

    for s in range(top_k):
        var eid = routing.indices[s]
        if eid % tp == rank:
            local_expert_ids[local_count] = eid
            local_gates[local_count] = routing.gates[s]
            local_count += 1

    # --- Phase 1: Routed experts + shared gate/up ---

    # Dispatch routed expert FFNs (one job per selected local expert)
    comptime MAX_CAPACITY = 128
    var expert_jobs = InlineArray[ExpertFFNArgs, top_k](uninitialized=True)
    for j in range(local_count):
        var eid = local_expert_ids[j]
        var local_idx = eid // tp
        var ew_base = expert_base + local_idx * expert_stride
        var gate_up_ptr = BF16Ptr(unsafe_from_address=ew_base)
        var down_ptr = BF16Ptr(
            unsafe_from_address=ew_base + 2 * expert_intermediate * hidden * 2)
        expert_jobs[j] = ExpertFFNArgs(
            input, gate_up_ptr, down_ptr,
            expert_out_buf + j * hidden,
            local_gates[j],
            expert_intermediate, hidden,
        )

    if local_count > 0:
        pool.dispatch[ExpertFFNArgs, expert_ffn_kernel[expert_intermediate, hidden]](
            UnsafePointer(to=expert_jobs[0]), local_count)
        pool.join()

    # Shared expert gate+up GEMVs (RowShard: each rank computes its shard)
    var shared_jobs = InlineArray[GemmArgs, MAX_CAPACITY](uninitialized=True)
    var cap = pool.get_capacity()

    # Gate GEMV: shared_gate_buf = shared_gate_w @ input
    var cols_per_job = (local_shared + cap - 1) // cap
    var gate_jobs = min(local_shared, cap)
    for i in range(gate_jobs):
        var start = i * cols_per_job
        var end = min(start + cols_per_job, local_shared)
        shared_jobs[i] = GemmArgs(input, shared_gate_w, shared_gate_buf, start, end, 1)
    pool.dispatch[GemmArgs, gemv_kernel[hidden, local_shared]](
        UnsafePointer(to=shared_jobs[0]), gate_jobs)
    pool.join()

    # Up GEMV: shared_up_buf = shared_up_w @ input
    for i in range(gate_jobs):
        var start = i * cols_per_job
        var end = min(start + cols_per_job, local_shared)
        shared_jobs[i] = GemmArgs(input, shared_up_w, shared_up_buf, start, end, 1)
    pool.dispatch[GemmArgs, gemv_kernel[hidden, local_shared]](
        UnsafePointer(to=shared_jobs[0]), gate_jobs)
    pool.join()

    # --- Phase 2: Fused combine (dispatched for NUMA locality) ---
    var combine_args = MoECombineArgs(
        shared_gate_buf, shared_up_buf, shared_down_w,
        expert_out_buf, dst,
        local_count, hidden, local_shared,
    )
    pool.dispatch[MoECombineArgs, moe_combine_kernel[local_shared, hidden]](
        UnsafePointer(to=combine_args), 1)
    pool.join()
