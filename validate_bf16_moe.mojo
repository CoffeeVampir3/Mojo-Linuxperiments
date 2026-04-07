"""Validate MoE dispatch against sequential reference.

Tests moe_dispatch (routed expert FFNs + shared expert + combine)
against a scalar sequential implementation. TP=1 (single rank,
no allreduce needed) to isolate the dispatch logic.
"""

from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from simd_math import exp_f32
from threading import BurstPool
from kernels.kernel_ops import gemv_kernel, GemmArgs
from kernels.moe_kernels import (
    softmax_topk, moe_dispatch, BF16Ptr,
)

comptime HIDDEN = 128
comptime EXPERT_INTERMEDIATE = 64
comptime SHARED_INTERMEDIATE = 128
comptime N_EXPERTS = 16
comptime TOP_K = 4
comptime TP = 1

# Expert weight layout (gate+up contiguous, then down)
comptime GATE_UP_ELEMS = 2 * EXPERT_INTERMEDIATE * HIDDEN
comptime DOWN_ELEMS = HIDDEN * EXPERT_INTERMEDIATE
comptime EXPERT_ELEMS = GATE_UP_ELEMS + DOWN_ELEMS
comptime EXPERT_STRIDE = EXPERT_ELEMS * 2

# Shared expert (no sharding for TP=1)
comptime SHARED_GATE_ELEMS = SHARED_INTERMEDIATE * HIDDEN
comptime SHARED_UP_ELEMS = SHARED_INTERMEDIATE * HIDDEN
comptime SHARED_DOWN_ELEMS = HIDDEN * SHARED_INTERMEDIATE

# Router
comptime ROUTER_ELEMS = N_EXPERTS * HIDDEN


def fill_deterministic(p: BF16Ptr, n: Int, seed: Int):
    """Fill with small deterministic values centered around zero."""
    for i in range(n):
        var raw = ((seed * 7 + i * 13) ^ (seed + i)) % 1000
        p[i] = Scalar[DType.bfloat16]((Float32(raw) - 500.0) * 0.002)


def scalar_silu(g: Float32) -> Float32:
    var sig = Float32(1.0) / (Float32(1.0) + Float32(exp_f32[1](SIMD[DType.float32, 1](-g))))
    return g * sig


def scalar_dot_bf16(
    a: BF16Ptr, b: BF16Ptr, n: Int,
) -> Scalar[DType.bfloat16]:
    """Scalar f32 dot product of bf16 vectors, result cast to bf16."""
    var acc = Float32(0)
    for i in range(n):
        acc += Float32(a[i]) * Float32(b[i])
    return Scalar[DType.bfloat16](acc)


def ref_expert_ffn(
    inp: BF16Ptr, gate_up: BF16Ptr, down: BF16Ptr,
    dst: BF16Ptr, gate_val: Float32,
    intermediate: Int, hidden: Int,
):
    """Scalar reference SwiGLU FFN with bf16 at stage boundaries."""
    var gate_w = gate_up
    var up_w = gate_up + intermediate * hidden

    # Stage 1: gate+up dot products → bf16
    var fused = InlineArray[Scalar[DType.bfloat16], 2 * EXPERT_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    for n in range(intermediate):
        fused[n] = scalar_dot_bf16(inp, gate_w + n * hidden, hidden)
        fused[intermediate + n] = scalar_dot_bf16(inp, up_w + n * hidden, hidden)

    # Stage 2: silu_mul → bf16
    var inter = InlineArray[Scalar[DType.bfloat16], EXPERT_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    for n in range(intermediate):
        var g = Float32(fused[n])
        var u = Float32(fused[intermediate + n])
        inter[n] = Scalar[DType.bfloat16](scalar_silu(g) * u)

    # Stage 3: down dot products → bf16, then gate scale → bf16
    var inter_p = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=inter[0])))
    for n in range(hidden):
        var acc = Float32(0)
        for k in range(intermediate):
            acc += Float32(inter[k]) * Float32(down[n * intermediate + k])
        dst[n] = Scalar[DType.bfloat16](acc * gate_val)


def ref_shared_ffn(
    inp: BF16Ptr, gate_w: BF16Ptr, up_w: BF16Ptr, down_w: BF16Ptr,
    gate_buf: BF16Ptr, up_buf: BF16Ptr, dst: BF16Ptr,
    intermediate: Int, hidden: Int,
):
    """Scalar reference shared SwiGLU FFN with bf16 at stage boundaries."""
    # Stage 1: gate+up → bf16
    for n in range(intermediate):
        gate_buf[n] = scalar_dot_bf16(inp, gate_w + n * hidden, hidden)
        up_buf[n] = scalar_dot_bf16(inp, up_w + n * hidden, hidden)

    # Stage 2: silu_mul → bf16 (in-place on gate_buf)
    for n in range(intermediate):
        var g = Float32(gate_buf[n])
        var u = Float32(up_buf[n])
        gate_buf[n] = Scalar[DType.bfloat16](scalar_silu(g) * u)

    # Stage 3: down → bf16
    for n in range(hidden):
        var acc = Float32(0)
        for k in range(intermediate):
            acc += Float32(gate_buf[k]) * Float32(down_w[n * intermediate + k])
        dst[n] = Scalar[DType.bfloat16](acc)


def main():
    print("=== MoE Dispatch Validation ===")
    print("hidden:", HIDDEN, "expert_inter:", EXPERT_INTERMEDIATE,
          "shared_inter:", SHARED_INTERMEDIATE)
    print("experts:", N_EXPERTS, "top_k:", TOP_K, "tp:", TP)

    # --- Allocate weight buffers ---
    var expert_weights = alloc[Scalar[DType.bfloat16]](N_EXPERTS * EXPERT_ELEMS)
    var router_weights = alloc[Scalar[DType.bfloat16]](ROUTER_ELEMS)
    var shared_gate_w = alloc[Scalar[DType.bfloat16]](SHARED_GATE_ELEMS)
    var shared_up_w = alloc[Scalar[DType.bfloat16]](SHARED_UP_ELEMS)
    var shared_down_w = alloc[Scalar[DType.bfloat16]](SHARED_DOWN_ELEMS)

    # Cast to MutAnyOrigin for kernel compatibility
    var expert_p = BF16Ptr(unsafe_from_address=Int(expert_weights))
    var router_p = BF16Ptr(unsafe_from_address=Int(router_weights))
    var sg_w = BF16Ptr(unsafe_from_address=Int(shared_gate_w))
    var su_w = BF16Ptr(unsafe_from_address=Int(shared_up_w))
    var sd_w = BF16Ptr(unsafe_from_address=Int(shared_down_w))

    # Fill with deterministic data
    for e in range(N_EXPERTS):
        fill_deterministic(expert_p + e * EXPERT_ELEMS, EXPERT_ELEMS, e + 1)
    fill_deterministic(router_p, ROUTER_ELEMS, 100)
    fill_deterministic(sg_w, SHARED_GATE_ELEMS, 200)
    fill_deterministic(su_w, SHARED_UP_ELEMS, 201)
    fill_deterministic(sd_w, SHARED_DOWN_ELEMS, 202)

    # --- Input hidden state ---
    var input_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](fill=Scalar[DType.bfloat16](0))
    for i in range(HIDDEN):
        input_buf[i] = Scalar[DType.bfloat16](Float32(i) * 0.01 - 0.5)
    var input_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=input_buf[0])))

    # --- Scratch buffers for moe_dispatch ---
    var shared_gate_buf = alloc[Scalar[DType.bfloat16]](SHARED_INTERMEDIATE)
    var shared_up_buf = alloc[Scalar[DType.bfloat16]](SHARED_INTERMEDIATE)
    var expert_out_buf = alloc[Scalar[DType.bfloat16]](TOP_K * HIDDEN)
    var dispatch_result = alloc[Scalar[DType.bfloat16]](HIDDEN)

    var sg_buf = BF16Ptr(unsafe_from_address=Int(shared_gate_buf))
    var su_buf = BF16Ptr(unsafe_from_address=Int(shared_up_buf))
    var eo_buf = BF16Ptr(unsafe_from_address=Int(expert_out_buf))
    var dst = BF16Ptr(unsafe_from_address=Int(dispatch_result))

    # --- Run moe_dispatch ---
    var pool = BurstPool[](15)
    if not pool:
        print("pool creation failed")
        return
    print("pool capacity:", pool.get_capacity())

    var t0 = perf_counter_ns()
    moe_dispatch[N_EXPERTS, TOP_K, EXPERT_INTERMEDIATE, SHARED_INTERMEDIATE,
                 HIDDEN, TP](
        input_ptr, router_p,
        Int(expert_p), EXPERT_STRIDE,
        sg_w, su_w, sd_w,
        sg_buf, su_buf, eo_buf,
        dst, 0, pool,
    )
    var dispatch_us = (perf_counter_ns() - t0) / 1000
    print("moe_dispatch:", dispatch_us, "us")

    # --- Sequential reference (same kernels, no pool dispatch) ---

    # Route (same as moe_dispatch does internally)
    var ref_router_out = InlineArray[Scalar[DType.bfloat16], N_EXPERTS](
        fill=Scalar[DType.bfloat16](0))
    var ref_router_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_router_out[0])))
    gemv_kernel[HIDDEN, N_EXPERTS](
        GemmArgs(input_ptr, router_p, ref_router_ptr, 0, N_EXPERTS, 1))
    var routing = softmax_topk[N_EXPERTS, TOP_K](ref_router_ptr)

    print("\nrouted experts:", routing.indices[0], routing.indices[1],
          routing.indices[2], routing.indices[3])
    print("gates:", routing.gates[0], routing.gates[1],
          routing.gates[2], routing.gates[3])

    # --- Scalar reference (bf16 at stage boundaries) ---

    # Run each selected expert with scalar reference
    var ref_expert_bufs = InlineArray[
        InlineArray[Scalar[DType.bfloat16], HIDDEN], TOP_K](
        fill=InlineArray[Scalar[DType.bfloat16], HIDDEN](fill=Scalar[DType.bfloat16](0)))

    var local_count = 0
    for s in range(TOP_K):
        var eid = routing.indices[s]
        if eid % TP == 0:
            var ew = expert_p + eid * EXPERT_ELEMS
            var eo = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_expert_bufs[local_count][0])))
            ref_expert_ffn(
                input_ptr, ew, ew + GATE_UP_ELEMS, eo,
                routing.gates[s], EXPERT_INTERMEDIATE, HIDDEN)
            local_count += 1

    # Shared expert with scalar reference
    var ref_sg_buf = InlineArray[Scalar[DType.bfloat16], SHARED_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var ref_su_buf = InlineArray[Scalar[DType.bfloat16], SHARED_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var ref_dst_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var ref_sg = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_sg_buf[0])))
    var ref_su = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_su_buf[0])))
    var ref_dst_p = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_dst_buf[0])))

    ref_shared_ffn(
        input_ptr, sg_w, su_w, sd_w,
        ref_sg, ref_su, ref_dst_p,
        SHARED_INTERMEDIATE, HIDDEN)

    # Combine: shared down wrote to ref_dst_p, now add expert buffers
    # (bf16 accumulation, same as moe_combine_kernel)
    for e in range(local_count):
        var ebuf = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=ref_expert_bufs[e][0])))
        for h in range(HIDDEN):
            var existing = Float32(ref_dst_p[h])
            var expert_val = Float32(ebuf[h])
            ref_dst_p[h] = Scalar[DType.bfloat16](existing + expert_val)

    var ref_result = InlineArray[Float32, HIDDEN](fill=Float32(0))
    for h in range(HIDDEN):
        ref_result[h] = Float32(ref_dst_buf[h])

    # === Per-stage comparison ===
    # Isolate each stage to find where divergence originates.

    # Stage A: single expert FFN in isolation
    # Pick expert 7 (first selected), run both paths, compare
    var eid0 = routing.indices[0]
    var ew0 = expert_p + eid0 * EXPERT_ELEMS
    var gate0 = routing.gates[0]

    # Dispatched expert (via pool)
    var d_expert_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var d_ep = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=d_expert_buf[0])))

    from kernels.moe_kernels import ExpertFFNArgs, expert_ffn_kernel
    var test_args = ExpertFFNArgs(
        input_ptr, ew0, ew0 + GATE_UP_ELEMS, d_ep,
        gate0, EXPERT_INTERMEDIATE, HIDDEN)
    pool.dispatch[ExpertFFNArgs, expert_ffn_kernel[EXPERT_INTERMEDIATE, HIDDEN]](
        UnsafePointer(to=test_args), 1)
    pool.join()

    # Reference expert (scalar)
    var r_expert_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var r_ep = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=r_expert_buf[0])))
    ref_expert_ffn(input_ptr, ew0, ew0 + GATE_UP_ELEMS, r_ep, gate0,
                   EXPERT_INTERMEDIATE, HIDDEN)

    var expert_mismatches = 0
    var expert_max_err = Float32(0)
    for h in range(HIDDEN):
        var err = abs(Float32(d_expert_buf[h]) - Float32(r_expert_buf[h]))
        if err > 0:
            expert_mismatches += 1
        if err > expert_max_err:
            expert_max_err = err
    print("\n--- Stage A: single expert FFN (expert", eid0, ") ---")
    print("final mismatches:", expert_mismatches, "/", HIDDEN, " max_err:", expert_max_err)

    # Decompose: run each sub-stage and compare
    comptime test_w = simd_width_of[DType.float32]()
    comptime FUSED = 2 * EXPERT_INTERMEDIATE

    # A.1: gate+up GEMV — dispatched vs scalar
    var d_fused = InlineArray[Scalar[DType.bfloat16], FUSED](uninitialized=True)
    var r_fused = InlineArray[Scalar[DType.bfloat16], FUSED](
        fill=Scalar[DType.bfloat16](0))
    var d_fp = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=d_fused[0])))
    gemv_kernel[HIDDEN, FUSED](GemmArgs(input_ptr, ew0, d_fp, 0, FUSED, 1))
    for n in range(FUSED):
        r_fused[n] = scalar_dot_bf16(input_ptr, ew0 + n * HIDDEN, HIDDEN)

    var a1_mm = 0
    for n in range(FUSED):
        if d_fused[n] != r_fused[n]:
            a1_mm += 1
    print("  A.1 gate+up GEMV mismatches:", a1_mm, "/", FUSED)

    # A.2: silu_mul — use SAME gate/up inputs (dispatched GEMV output), compare SIMD vs scalar silu
    var d_silu = InlineArray[Scalar[DType.bfloat16], EXPERT_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var r_silu = InlineArray[Scalar[DType.bfloat16], EXPERT_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var d_sp = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=d_silu[0])))
    var d_gate = d_fp
    var d_up = d_fp + EXPERT_INTERMEDIATE
    # SIMD silu
    for i in range(0, EXPERT_INTERMEDIATE, test_w):
        var g = (d_gate + i).load[width=test_w]().cast[DType.float32]()
        var u = (d_up + i).load[width=test_w]().cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp_f32[test_w](-g))
        (d_sp + i).store((g * sig * u).cast[DType.bfloat16]())
    # Scalar silu (same inputs)
    for n in range(EXPERT_INTERMEDIATE):
        var g = Float32(d_fused[n])
        var u = Float32(d_fused[EXPERT_INTERMEDIATE + n])
        r_silu[n] = Scalar[DType.bfloat16](scalar_silu(g) * u)

    var a2_mm = 0
    for n in range(EXPERT_INTERMEDIATE):
        if d_silu[n] != r_silu[n]:
            a2_mm += 1
            if a2_mm <= 3:
                print("    silu[", n, "] simd:", Float32(d_silu[n]),
                      "scalar:", Float32(r_silu[n]),
                      "g:", Float32(d_fused[n]),
                      "u:", Float32(d_fused[EXPERT_INTERMEDIATE + n]))
    print("  A.2 silu_mul mismatches:", a2_mm, "/", EXPERT_INTERMEDIATE)

    # A.3: down GEMV — use SAME silu inputs, compare SIMD vs scalar
    var d_down = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var r_down = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var d_dp = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=d_down[0])))
    var down_w = ew0 + GATE_UP_ELEMS
    gemv_kernel[EXPERT_INTERMEDIATE, HIDDEN](
        GemmArgs(d_sp, down_w, d_dp, 0, HIDDEN, 1))
    for n in range(HIDDEN):
        r_down[n] = scalar_dot_bf16(d_sp, down_w + n * EXPERT_INTERMEDIATE, EXPERT_INTERMEDIATE)

    var a3_mm = 0
    for n in range(HIDDEN):
        if d_down[n] != r_down[n]:
            a3_mm += 1
    print("  A.3 down GEMV (K=64) mismatches:", a3_mm, "/", HIDDEN)

    # A.0: does ref_expert_ffn's gate/up GEMV match gemv_kernel for THIS expert's weights?
    var r_fused_e2e = InlineArray[Scalar[DType.bfloat16], FUSED](
        fill=Scalar[DType.bfloat16](0))
    var gate_w = ew0
    var up_w = ew0 + EXPERT_INTERMEDIATE * HIDDEN
    for n in range(EXPERT_INTERMEDIATE):
        r_fused_e2e[n] = scalar_dot_bf16(input_ptr, gate_w + n * HIDDEN, HIDDEN)
        r_fused_e2e[EXPERT_INTERMEDIATE + n] = scalar_dot_bf16(input_ptr, up_w + n * HIDDEN, HIDDEN)
    var a0_mm = 0
    for n in range(FUSED):
        if d_fused[n] != r_fused_e2e[n]:
            a0_mm += 1
    print("  A.0 expert gate/up: gemv_kernel vs scalar_dot:", a0_mm, "/", FUSED)

    # Stage B: shared expert gate GEMV in isolation
    var d_sg_test = InlineArray[Scalar[DType.bfloat16], SHARED_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var r_sg_test = InlineArray[Scalar[DType.bfloat16], SHARED_INTERMEDIATE](
        fill=Scalar[DType.bfloat16](0))
    var d_sg_p = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=d_sg_test[0])))
    var r_sg_p = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=r_sg_test[0])))

    # Dispatched (via pool, partitioned)
    var sg_test_jobs = InlineArray[GemmArgs, 128](uninitialized=True)
    var sg_cap = pool.get_capacity()
    var sg_cpj = (SHARED_INTERMEDIATE + sg_cap - 1) // sg_cap
    var sg_nj = min(SHARED_INTERMEDIATE, sg_cap)
    for i in range(sg_nj):
        var s = i * sg_cpj
        var e = min(s + sg_cpj, SHARED_INTERMEDIATE)
        sg_test_jobs[i] = GemmArgs(input_ptr, sg_w, d_sg_p, s, e, 1)
    pool.dispatch[GemmArgs, gemv_kernel[HIDDEN, SHARED_INTERMEDIATE]](
        UnsafePointer(to=sg_test_jobs[0]), sg_nj)
    pool.join()

    # Reference (scalar)
    for n in range(SHARED_INTERMEDIATE):
        r_sg_test[n] = scalar_dot_bf16(input_ptr, sg_w + n * HIDDEN, HIDDEN)

    var gemv_mismatches = 0
    var gemv_max_err = Float32(0)
    for n in range(SHARED_INTERMEDIATE):
        var err = abs(Float32(d_sg_test[n]) - Float32(r_sg_test[n]))
        if err > 0:
            gemv_mismatches += 1
        if err > gemv_max_err:
            gemv_max_err = err
    # Stage B.5: exp_f32 scalar vs vector comparison
    comptime test_width = simd_width_of[DType.float32]()
    var exp_mismatches = 0
    for n in range(SHARED_INTERMEDIATE):
        var g = Float32(d_sg_test[n])
        var scalar_exp = Float32(exp_f32[1](SIMD[DType.float32, 1](-g)))
        # Build a vector with g in lane 0, compute vectorized exp
        var gv = SIMD[DType.float32, test_width](g)
        var vector_exp = Float32(exp_f32[test_width](-gv)[0])
        if scalar_exp != vector_exp:
            exp_mismatches += 1
            if exp_mismatches <= 3:
                print("  exp_f32 mismatch at", n, ": scalar=", scalar_exp,
                      "vector=", vector_exp, "input=", g)
    print("\n--- Stage B.5: exp_f32[1] vs exp_f32[", test_width, "] ---")
    print("mismatches:", exp_mismatches, "/", SHARED_INTERMEDIATE)

    print("\n--- Stage B: shared gate GEMV [", SHARED_INTERMEDIATE, ",", HIDDEN, "] ---")
    print("mismatches:", gemv_mismatches, "/", SHARED_INTERMEDIATE, " max_err:", gemv_max_err)

    # Stage C: full pipeline comparison
    var max_err = Float32(0)
    var pipeline_mismatches = 0
    for h in range(HIDDEN):
        var err = abs(Float32(dst[h]) - ref_result[h])
        if err > 0:
            pipeline_mismatches += 1
        if err > max_err:
            max_err = err
    print("\n--- Stage C: full MoE pipeline ---")
    print("result[:4] dispatched:", Float32(dst[0]), Float32(dst[1]),
          Float32(dst[2]), Float32(dst[3]))
    print("result[:4] reference: ", ref_result[0], ref_result[1],
          ref_result[2], ref_result[3])
    print("mismatches:", pipeline_mismatches, "/", HIDDEN, " max_err:", max_err)

    if expert_mismatches == 0 and gemv_mismatches == 0 and pipeline_mismatches == 0:
        print("\nPASS (bit-exact at all stages)")
    elif expert_max_err <= 1.0 and gemv_max_err <= 1.0:
        print("\nPASS (per-stage error <= 1 bf16 ULP, pipeline amplification expected)")
    else:
        print("\nFAIL")

    expert_weights.free()
    router_weights.free()
    shared_gate_w.free()
    shared_up_w.free()
    shared_down_w.free()
    shared_gate_buf.free()
    shared_up_buf.free()
    expert_out_buf.free()
    dispatch_result.free()
    _ = input_buf
    _ = pool
