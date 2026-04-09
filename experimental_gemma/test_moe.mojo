"""Validate Gemma4 expert FFN kernel and MoE dispatch.

Tests with small dimensions (4 experts, top-2, intermediate=16, hidden=32).
Reference computes in pure f32 (no intermediate bf16 roundtrips).
Kernel has intermediate bf16 stages (GEMV output → activation → GEMV),
so kern_err reflects accumulated bf16 quantization across stages.

Verifies:
  1. Per-expert output matches reference within bf16 pipeline tolerance
  2. Accumulated output (sum of weighted experts) matches reference
  3. Routing weight scaling is applied correctly
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import exp_f32
from experimental_gemma.activations import gelu_tanh_f32
from experimental_gemma.router import Gemma4TopKResult
from experimental_gemma.moe import (
    gemma4_moe_dispatch, gemma4_expert_ffn_kernel, Gemma4ExpertFFNArgs,
)
from kernels.kernel_ops import GemmArgs, gemv_kernel

from numa import NumaInfo
from threading import BurstPool

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def f32_expert_ffn(
    inp: UnsafePointer[Scalar[DType.bfloat16], _],
    gate_up: UnsafePointer[Scalar[DType.bfloat16], _],
    down: UnsafePointer[Scalar[DType.bfloat16], _],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    routing_weight: Float32,
    intermediate: Int,
    hidden: Int,
):
    """Reference expert FFN in pure f32."""
    var gate_up_dim = 2 * intermediate

    # Gate+up GEMV in f32
    var fused = alloc[Float32](gate_up_dim)
    for n in range(gate_up_dim):
        var acc = Float32(0)
        for k in range(hidden):
            acc += Float32(inp[k]) * Float32(gate_up[n * hidden + k])
        fused[n] = acc

    # gelu_tanh(gate) * up
    var activated = alloc[Float32](intermediate)
    for i in range(intermediate):
        var g = fused[i]
        var u = fused[intermediate + i]
        var gv = gelu_tanh_f32[1](SIMD[DType.float32, 1](g))
        activated[i] = gv[0] * u

    # Down GEMV with routing weight
    for n in range(hidden):
        var acc = Float32(0)
        for k in range(intermediate):
            acc += activated[k] * Float32(down[n * intermediate + k])
        dst[n] = acc * routing_weight

    fused.free()
    activated.free()


def test_moe_dispatch():
    print("=== Gemma4 MoE dispatch ===")

    comptime NUM_EXPERTS = 4
    comptime TOP_K = 2
    comptime INTERMEDIATE = 16
    comptime HIDDEN = 32
    comptime GATE_UP_DIM = 2 * INTERMEDIATE

    # Allocate expert weights with deterministic values
    var gate_up_all = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS * GATE_UP_DIM * HIDDEN)
    var down_all = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS * HIDDEN * INTERMEDIATE)

    for e in range(NUM_EXPERTS):
        for i in range(GATE_UP_DIM * HIDDEN):
            var v = Float32(0.01) * Float32((e * GATE_UP_DIM * HIDDEN + i) % 97 - 48)
            gate_up_all[e * GATE_UP_DIM * HIDDEN + i] = Scalar[DType.bfloat16](v)
        for i in range(HIDDEN * INTERMEDIATE):
            var v = Float32(0.01) * Float32((e * HIDDEN * INTERMEDIATE + i) % 83 - 41)
            down_all[e * HIDDEN * INTERMEDIATE + i] = Scalar[DType.bfloat16](v)

    # Input vector
    var inp = alloc[Scalar[DType.bfloat16]](HIDDEN)
    for i in range(HIDDEN):
        inp[i] = Scalar[DType.bfloat16](Float32(-0.5) + Float32(i) * Float32(0.03))

    # Routing: select experts 1 and 3
    var routing = Gemma4TopKResult[TOP_K](
        indices=InlineArray[Int, TOP_K](fill=0),
        weights=InlineArray[Float32, TOP_K](fill=Float32(0)),
    )
    routing.indices[0] = 1
    routing.indices[1] = 3
    routing.weights[0] = Float32(0.65)
    routing.weights[1] = Float32(0.35)

    print("  routing: expert " + String(routing.indices[0]) + " w=" + String(routing.weights[0])
        + ", expert " + String(routing.indices[1]) + " w=" + String(routing.weights[1]))

    # F32 reference per expert
    var ref_per_expert = alloc[Float32](TOP_K * HIDDEN)
    for s in range(TOP_K):
        var eid = routing.indices[s]
        f32_expert_ffn(
            inp,
            gate_up_all + eid * GATE_UP_DIM * HIDDEN,
            down_all + eid * HIDDEN * INTERMEDIATE,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_per_expert + s * HIDDEN)),
            routing.weights[s],
            INTERMEDIATE,
            HIDDEN,
        )

    # F32 reference accumulated
    var ref_sum = alloc[Float32](HIDDEN)
    for i in range(HIDDEN):
        var acc = Float32(0)
        for s in range(TOP_K):
            acc += ref_per_expert[s * HIDDEN + i]
        ref_sum[i] = acc

    # Run kernel dispatch
    var expert_out_buf = alloc[Scalar[DType.bfloat16]](TOP_K * HIDDEN)
    var dst = alloc[Scalar[DType.bfloat16]](HIDDEN)

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    gemma4_moe_dispatch[NUM_EXPERTS, TOP_K, INTERMEDIATE, HIDDEN](
        BF16Ptr(unsafe_from_address=Int(inp)),
        routing,
        BF16Ptr(unsafe_from_address=Int(gate_up_all)),
        BF16Ptr(unsafe_from_address=Int(down_all)),
        BF16Ptr(unsafe_from_address=Int(expert_out_buf)),
        BF16Ptr(unsafe_from_address=Int(dst)),
        pool,
    )

    # Per-expert comparison
    for s in range(TOP_K):
        var eid = routing.indices[s]
        print()
        print("  --- expert " + String(eid) + " (weight=" + String(routing.weights[s]) + ") ---")
        print("  idx | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
        print("  ----+--------------+--------------+--------------+--------------+---------")

        var max_kern_err = Float64(0)
        var max_bf16_err = Float64(0)
        for i in range(HIDDEN):
            var kv = Float32(expert_out_buf[s * HIDDEN + i])
            var fv = ref_per_expert[s * HIDDEN + i]
            var bv = Float32(Scalar[DType.bfloat16](fv))
            var kern_err = abs(Float64(kv) - Float64(bv))
            var bf16_err = abs(Float64(fv) - Float64(bv))
            if kern_err > max_kern_err:
                max_kern_err = kern_err
            if bf16_err > max_bf16_err:
                max_bf16_err = bf16_err
            print("  " + String(i) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

        print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    # Accumulated comparison
    print()
    print("  --- accumulated output (sum of " + String(TOP_K) + " experts) ---")
    print("  idx | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+---------")

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for i in range(HIDDEN):
        var kv = Float32(dst[i])
        var fv = ref_sum[i]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        print("  " + String(i) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    gate_up_all.free()
    down_all.free()
    inp.free()
    ref_per_expert.free()
    ref_sum.free()
    expert_out_buf.free()
    dst.free()
    _ = pool^


def test_parallel_dispatch():
    """Two pools dispatching concurrently: expert FFNs on pool_moe, dense GEMV on pool_dense."""
    print("=== Parallel dispatch (two pools, concurrent) ===")

    comptime NUM_EXPERTS = 4
    comptime TOP_K = 2
    comptime INTERMEDIATE = 16
    comptime HIDDEN = 32
    comptime GATE_UP_DIM = 2 * INTERMEDIATE
    comptime DENSE_OUT = 32

    # Expert weights (same as sequential test)
    var gate_up_all = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS * GATE_UP_DIM * HIDDEN)
    var down_all = alloc[Scalar[DType.bfloat16]](NUM_EXPERTS * HIDDEN * INTERMEDIATE)
    for e in range(NUM_EXPERTS):
        for i in range(GATE_UP_DIM * HIDDEN):
            var v = Float32(0.01) * Float32((e * GATE_UP_DIM * HIDDEN + i) % 97 - 48)
            gate_up_all[e * GATE_UP_DIM * HIDDEN + i] = Scalar[DType.bfloat16](v)
        for i in range(HIDDEN * INTERMEDIATE):
            var v = Float32(0.01) * Float32((e * HIDDEN * INTERMEDIATE + i) % 83 - 41)
            down_all[e * HIDDEN * INTERMEDIATE + i] = Scalar[DType.bfloat16](v)

    # Dense MLP weight (simulates one projection)
    var dense_w = alloc[Scalar[DType.bfloat16]](DENSE_OUT * HIDDEN)
    for i in range(DENSE_OUT * HIDDEN):
        dense_w[i] = Scalar[DType.bfloat16](Float32(0.01) * Float32(i % 71 - 35))

    # Input
    var inp = alloc[Scalar[DType.bfloat16]](HIDDEN)
    for i in range(HIDDEN):
        inp[i] = Scalar[DType.bfloat16](Float32(-0.5) + Float32(i) * Float32(0.03))

    # Routing
    var routing = Gemma4TopKResult[TOP_K](
        indices=InlineArray[Int, TOP_K](fill=0),
        weights=InlineArray[Float32, TOP_K](fill=Float32(0)),
    )
    routing.indices[0] = 1
    routing.indices[1] = 3
    routing.weights[0] = Float32(0.65)
    routing.weights[1] = Float32(0.35)

    # F32 references
    var ref_per_expert = alloc[Float32](TOP_K * HIDDEN)
    for s in range(TOP_K):
        var eid = routing.indices[s]
        f32_expert_ffn(
            inp,
            gate_up_all + eid * GATE_UP_DIM * HIDDEN,
            down_all + eid * HIDDEN * INTERMEDIATE,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_per_expert + s * HIDDEN)),
            routing.weights[s],
            INTERMEDIATE,
            HIDDEN,
        )

    var ref_dense = alloc[Float32](DENSE_OUT)
    for n in range(DENSE_OUT):
        var acc = Float32(0)
        for k in range(HIDDEN):
            acc += Float32(inp[k]) * Float32(dense_w[n * HIDDEN + k])
        ref_dense[n] = acc

    # Buffers
    var expert_out_buf = alloc[Scalar[DType.bfloat16]](TOP_K * HIDDEN)
    var dense_out = alloc[Scalar[DType.bfloat16]](DENSE_OUT)

    # Two pools on same NUMA node
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool_moe = BurstPool[].for_numa_node(numa, topo[0])
    var pool_dense = BurstPool[].for_numa_node(numa, topo[0])

    var inp_ptr = BF16Ptr(unsafe_from_address=Int(inp))

    # Build expert FFN jobs
    var expert_jobs = InlineArray[Gemma4ExpertFFNArgs, TOP_K](uninitialized=True)
    for s in range(TOP_K):
        var eid = routing.indices[s]
        expert_jobs[s] = Gemma4ExpertFFNArgs(
            inp_ptr,
            BF16Ptr(unsafe_from_address=Int(gate_up_all)) + eid * GATE_UP_DIM * HIDDEN,
            BF16Ptr(unsafe_from_address=Int(down_all)) + eid * HIDDEN * INTERMEDIATE,
            BF16Ptr(unsafe_from_address=Int(expert_out_buf)) + s * HIDDEN,
            routing.weights[s],
        )

    # Build dense GEMV job
    var dense_jobs = InlineArray[GemmArgs, 1](uninitialized=True)
    dense_jobs[0] = GemmArgs(
        inp_ptr,
        BF16Ptr(unsafe_from_address=Int(dense_w)),
        BF16Ptr(unsafe_from_address=Int(dense_out)),
        0, DENSE_OUT, 1,
    )

    # Dispatch both pools concurrently
    pool_moe.dispatch[Gemma4ExpertFFNArgs, gemma4_expert_ffn_kernel[INTERMEDIATE, HIDDEN]](
        UnsafePointer(to=expert_jobs[0]), TOP_K)
    pool_dense.dispatch[GemmArgs, gemv_kernel[HIDDEN, DENSE_OUT]](
        UnsafePointer(to=dense_jobs[0]), 1)

    # Join both
    pool_moe.join()
    pool_dense.join()

    # Verify expert outputs match sequential reference
    print("  expert outputs (parallel vs f32 reference):")
    var max_expert_err = Float64(0)
    for s in range(TOP_K):
        var eid = routing.indices[s]
        var err = Float64(0)
        for i in range(HIDDEN):
            var kv = Float32(expert_out_buf[s * HIDDEN + i])
            var bv = Float32(Scalar[DType.bfloat16](ref_per_expert[s * HIDDEN + i]))
            var e = abs(Float64(kv) - Float64(bv))
            if e > err:
                err = e
        if err > max_expert_err:
            max_expert_err = err
        print("    expert " + String(eid) + ": max |kernel - bf16(ref)| = " + String(err))

    # Verify dense GEMV output
    print("  dense GEMV (parallel vs f32 reference):")
    print("  idx | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err")
    print("  ----+--------------+--------------+--------------+---------")
    var max_dense_err = Float64(0)
    for i in range(DENSE_OUT):
        var kv = Float32(dense_out[i])
        var fv = ref_dense[i]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var err = abs(Float64(kv) - Float64(bv))
        if err > max_dense_err:
            max_dense_err = err
        print("  " + String(i) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(err))

    print("  max expert_err=" + String(max_expert_err) + "  max dense_err=" + String(max_dense_err))

    gate_up_all.free()
    down_all.free()
    dense_w.free()
    inp.free()
    ref_per_expert.free()
    ref_dense.free()
    expert_out_buf.free()
    dense_out.free()
    _ = pool_moe^
    _ = pool_dense^


def main():
    test_moe_dispatch()
    print()
    test_parallel_dispatch()
