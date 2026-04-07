"""POC: MoE expert dispatch over NUMA nodes via BurstPool.

Sets up N_EXPERTS toy experts distributed across NUMA nodes.
Router selects TOP_K experts, groups by node, dispatches to
per-node BurstPools. Each worker runs one expert FFN (dot product
with expert-specific weights). Results are gated and summed.
Verified against sequential reference.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray
from simd_math import exp_f32
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

comptime HIDDEN = 128
comptime INTERMEDIATE = 64
comptime N_EXPERTS = 16
comptime TOP_K = 4
comptime TP = 2
comptime ALIGNMENT = 64

# Per-expert weight layout: gate[INTERMEDIATE, HIDDEN] + up[INTERMEDIATE, HIDDEN] + down[HIDDEN, INTERMEDIATE]
comptime GATE_OFF = 0
comptime GATE_SIZE = INTERMEDIATE * HIDDEN * 2
comptime UP_OFF = GATE_SIZE
comptime UP_SIZE = INTERMEDIATE * HIDDEN * 2
comptime DOWN_OFF = UP_OFF + UP_SIZE
comptime DOWN_SIZE = HIDDEN * INTERMEDIATE * 2
comptime EXPERT_STRIDE = DOWN_OFF + DOWN_SIZE
comptime EXPERTS_PER_NODE = N_EXPERTS // TP
comptime NODE_ARENA_SIZE = EXPERTS_PER_NODE * EXPERT_STRIDE + ALIGNMENT

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@fieldwise_init
struct ExpertFFNArgs(Copyable, ImplicitlyCopyable):
    var input_ptr: BF16Ptr
    var expert_base: Int
    var output_ptr: BF16Ptr
    var gate_val: Float32
    var intermediate: Int
    var hidden: Int


def expert_ffn_kernel(args: ExpertFFNArgs):
    """Run one expert FFN: gate+up -> silu_mul -> down. Writes gated result."""
    comptime width = simd_width_of[DType.float32]()

    var inp = args.input_ptr
    var gate_w = BF16Ptr(unsafe_from_address=args.expert_base + GATE_OFF)
    var up_w = BF16Ptr(unsafe_from_address=args.expert_base + UP_OFF)
    var down_w = BF16Ptr(unsafe_from_address=args.expert_base + DOWN_OFF)
    var out = args.output_ptr
    var gate_val = args.gate_val
    var intermediate = args.intermediate
    var hidden = args.hidden

    # Phase 1: gate + up GEMVs, fused with silu_mul into intermediate buffer
    var inter = InlineArray[Float32, INTERMEDIATE](fill=Float32(0))

    for n in range(intermediate):
        var gate_acc = SIMD[DType.float32, width](0)
        var up_acc = SIMD[DType.float32, width](0)
        var gw_row = gate_w + n * hidden
        var uw_row = up_w + n * hidden
        for k in range(0, hidden, width):
            var x = (inp + k).load[width=width]().cast[DType.float32]()
            gate_acc = x.fma((gw_row + k).load[width=width]().cast[DType.float32](), gate_acc)
            up_acc = x.fma((uw_row + k).load[width=width]().cast[DType.float32](), up_acc)
        var g = gate_acc.reduce_add()
        var u = up_acc.reduce_add()
        var silu_g = g / (Float32(1.0) + Float32(exp_f32[1](SIMD[DType.float32, 1](-g))))
        inter[n] = silu_g * u

    # Phase 2: down GEMV + gate scaling
    for n in range(hidden):
        var acc = SIMD[DType.float32, width](0)
        var dw_row = down_w + n * intermediate
        for k in range(0, intermediate, width):
            var x = SIMD[DType.float32, width](0)
            for lane in range(width):
                x[lane] = inter[k + lane]
            acc = x.fma((dw_row + k).load[width=width]().cast[DType.float32](), acc)
        out[n] = Scalar[DType.bfloat16](acc.reduce_add() * gate_val)


def sequential_expert_ffn(
    inp: BF16Ptr, expert_base: Int, gate_val: Float32, dst: BF16Ptr,
):
    """Reference sequential expert FFN for verification."""
    var gate_w = BF16Ptr(unsafe_from_address=expert_base + GATE_OFF)
    var up_w = BF16Ptr(unsafe_from_address=expert_base + UP_OFF)
    var down_w = BF16Ptr(unsafe_from_address=expert_base + DOWN_OFF)

    var inter = InlineArray[Float32, INTERMEDIATE](fill=Float32(0))

    for n in range(INTERMEDIATE):
        var gate_acc = Float32(0)
        var up_acc = Float32(0)
        for k in range(HIDDEN):
            var x = Float32(inp[k])
            gate_acc += x * Float32(gate_w[n * HIDDEN + k])
            up_acc += x * Float32(up_w[n * HIDDEN + k])
        var silu_g = gate_acc / (Float32(1.0) + Float32(exp_f32[1](SIMD[DType.float32, 1](-gate_acc))))
        inter[n] = silu_g * up_acc

    for n in range(HIDDEN):
        var acc = Float32(0)
        for k in range(INTERMEDIATE):
            acc += inter[k] * Float32(down_w[n * INTERMEDIATE + k])
        dst[n] = Scalar[DType.bfloat16](acc * gate_val)


def main():
    print("=== MoE Dispatch POC ===")
    print("experts:", N_EXPERTS, "top_k:", TOP_K, "tp:", TP)
    print("hidden:", HIDDEN, "intermediate:", INTERMEDIATE)
    print()

    # --- Setup NUMA arenas and pools ---
    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)

    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](TP)
    for rank in range(TP):
        var arena = NumaArena[alignment=ALIGNMENT](topo[rank], NODE_ARENA_SIZE)
        if not arena:
            print("arena alloc failed for rank", rank)
            return
        arenas.push(arena^)

    var pools = HeapMoveArray[BurstPool[]](TP)
    for rank in range(TP):
        pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
    )
    for rank in range(TP):
        pool_ptrs[rank] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[rank]))
        )

    var arena_bases = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        arena_bases[rank] = Int(arenas[rank].base)

    print("pool capacities:", pool_ptrs[0][].get_capacity(), pool_ptrs[1][].get_capacity())

    # --- Fill expert weights with deterministic pattern ---
    for e in range(N_EXPERTS):
        var node = e % TP
        var local_idx = e // TP
        var base = arena_bases[node] + local_idx * EXPERT_STRIDE
        var wp = BF16Ptr(unsafe_from_address=base)
        var total_elems = EXPERT_STRIDE // 2
        for i in range(total_elems):
            wp[i] = Scalar[DType.bfloat16](Float32(e * 1000 + i) * 0.001 - 5.0)

    # --- Create input hidden state ---
    var input_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](fill=Scalar[DType.bfloat16](0))
    for i in range(HIDDEN):
        input_buf[i] = Scalar[DType.bfloat16](Float32(i) * 0.01 - 0.5)
    var input_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=input_buf[0])))

    # --- Simulate router: pick TOP_K experts ---
    var selected_ids = InlineArray[Int, TOP_K](fill=0)
    var selected_gates = InlineArray[Float32, TOP_K](fill=Float32(0))
    selected_ids[0] = 3
    selected_ids[1] = 7
    selected_ids[2] = 10
    selected_ids[3] = 14
    selected_gates[0] = Float32(0.35)
    selected_gates[1] = Float32(0.25)
    selected_gates[2] = Float32(0.22)
    selected_gates[3] = Float32(0.18)

    print("selected experts:", selected_ids[0], selected_ids[1], selected_ids[2], selected_ids[3])
    print("gates:", selected_gates[0], selected_gates[1], selected_gates[2], selected_gates[3])

    # --- Group by node ---
    var node_expert_count = InlineArray[Int, TP](fill=0)
    var node_expert_ids = InlineArray[InlineArray[Int, TOP_K], TP](
        fill=InlineArray[Int, TOP_K](fill=0)
    )
    var node_expert_gates = InlineArray[InlineArray[Float32, TOP_K], TP](
        fill=InlineArray[Float32, TOP_K](fill=Float32(0))
    )

    for s in range(TOP_K):
        var eid = selected_ids[s]
        var node = eid % TP
        var idx = node_expert_count[node]
        node_expert_ids[node][idx] = eid
        node_expert_gates[node][idx] = selected_gates[s]
        node_expert_count[node] += 1

    for node in range(TP):
        print("  node", node, ":", node_expert_count[node], "experts")

    # --- Allocate per-expert output buffers (on host stack) ---
    var expert_outputs = InlineArray[InlineArray[Scalar[DType.bfloat16], HIDDEN], TOP_K](
        fill=InlineArray[Scalar[DType.bfloat16], HIDDEN](fill=Scalar[DType.bfloat16](0))
    )

    # --- Dispatch to all nodes ---
    var t0 = perf_counter_ns()

    for node in range(TP):
        var count = node_expert_count[node]
        if count == 0:
            continue

        var job_args = InlineArray[ExpertFFNArgs, TOP_K](uninitialized=True)
        for j in range(count):
            var eid = node_expert_ids[node][j]
            var local_idx = eid // TP
            var expert_base = arena_bases[node] + local_idx * EXPERT_STRIDE

            # Find which global selection index this is (for output buffer)
            var out_idx = 0
            for s in range(TOP_K):
                if selected_ids[s] == eid:
                    out_idx = s
                    break

            job_args[j] = ExpertFFNArgs(
                input_ptr,
                expert_base,
                BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=expert_outputs[out_idx][0]))),
                node_expert_gates[node][j],
                INTERMEDIATE,
                HIDDEN,
            )

        pool_ptrs[node][].dispatch[ExpertFFNArgs, expert_ffn_kernel](
            UnsafePointer(to=job_args[0]), count)

    # Join all nodes
    for node in range(TP):
        if node_expert_count[node] > 0:
            pool_ptrs[node][].join()

    var dispatch_us = (perf_counter_ns() - t0) / 1000
    print("dispatch + join:", dispatch_us, "us")

    # --- Sum gated expert outputs ---
    var dispatched_result = InlineArray[Float32, HIDDEN](fill=Float32(0))
    for s in range(TOP_K):
        for h in range(HIDDEN):
            dispatched_result[h] += Float32(expert_outputs[s][h])

    # --- Sequential reference ---
    var ref_result = InlineArray[Float32, HIDDEN](fill=Float32(0))
    var ref_out_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](fill=Scalar[DType.bfloat16](0))

    for s in range(TOP_K):
        var eid = selected_ids[s]
        var node = eid % TP
        var local_idx = eid // TP
        var expert_base = arena_bases[node] + local_idx * EXPERT_STRIDE
        var ref_out = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_out_buf[0])))

        sequential_expert_ffn(input_ptr, expert_base, selected_gates[s], ref_out)

        for h in range(HIDDEN):
            ref_result[h] += Float32(ref_out_buf[h])

    # --- Compare ---
    var max_err = Float32(0)
    for h in range(HIDDEN):
        var err = abs(dispatched_result[h] - ref_result[h])
        if err > max_err:
            max_err = err

    print()
    print("result[:4] dispatched:", dispatched_result[0], dispatched_result[1], dispatched_result[2], dispatched_result[3])
    print("result[:4] reference: ", ref_result[0], ref_result[1], ref_result[2], ref_result[3])
    print("max error:", max_err)

    var bf16_ulp = Float32(0.0078125)
    if max_err <= bf16_ulp * 4:
        print("PASS (within 4 bf16 ULPs)")
    else:
        print("FAIL")

    _ = arenas
    _ = pools
    _ = input_buf
    _ = expert_outputs
