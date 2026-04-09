"""MoE-focused microbenchmarks for the experimental Gemma kernels.

Benchmarks:
  1. Single expert FFN kernel, direct call on the main thread.
  2. Routed-expert dispatch + join + accumulate using BurstPool timestamps.
  3. Dense Gemma MLP decode branch, timed stage-by-stage.
  4. Side-by-side dense-branch + routed-expert overlap on two BurstPools.

Run on the remote AMX-capable host via:
  remote_build.fish experimental_gemma/benchmark_moe.mojo
"""

from std.collections import InlineArray
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numa import NumaInfo
from threading import BurstPool

from modeling.model_spec import BF16, Bound, DynView, Encoding, Replicated, Shaped, Slot
from kernels.kernel_ops import gemm
from experimental_gemma.activations import gelu_tanh_mul
from experimental_gemma.moe import (
    Gemma4ExpertFFNArgs,
    gemma4_expert_ffn_kernel,
)
from experimental_gemma.router import Gemma4TopKResult


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@fieldwise_init
struct DispatchTiming:
    var dispatch_ns: Int
    var wall_ns: Int
    var done_from_start_ns: Int
    var join_overhead_ns: Int
    var done_ts_abs: Int
    var end_ts_abs: Int


@fieldwise_init
struct MoETiming:
    var dispatch_ns: Int
    var worker_done_ns: Int
    var join_overhead_ns: Int
    var accumulate_ns: Int
    var total_ns: Int
    var done_ts_abs: Int
    var end_ts_abs: Int


@fieldwise_init
struct OverlapTiming:
    var moe_dispatch_ns: Int
    var moe_done_ns: Int
    var dense_done_ns: Int
    var dense_total_ns: Int
    var moe_accumulate_ns: Int
    var wall_ns: Int
    var done_spread_ns: Int
    var tail_after_last_worker_ns: Int


def checksum_bf16(ptr: BF16Ptr, count: Int) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += Float32(ptr[i])
    return sum


def print_simple_timing(label: String, iters: Int, elapsed_ns: Int, checksum: Float32):
    var per_iter_us = Float64(elapsed_ns) / Float64(iters) / 1000.0
    print(label + ": " + String(per_iter_us) + " us/iter  checksum=" + String(checksum))


def print_moe_timing(label: String, iters: Int, timing: MoETiming, checksum: Float32):
    var scale = Float64(iters) * 1000.0
    print(label
        + ": total=" + String(Float64(timing.total_ns) / scale) + " us"
        + "  dispatch=" + String(Float64(timing.dispatch_ns) / scale) + " us"
        + "  last_worker=" + String(Float64(timing.worker_done_ns) / scale) + " us"
        + "  join_oh=" + String(Float64(timing.join_overhead_ns) / Float64(iters)) + " ns"
        + "  accumulate=" + String(Float64(timing.accumulate_ns) / scale) + " us"
        + "  checksum=" + String(checksum))


def print_dense_timing(
    label: String,
    iters: Int,
    gate: DispatchTiming,
    up: DispatchTiming,
    act_ns: Int,
    down: DispatchTiming,
    total_ns: Int,
    checksum: Float32,
):
    var scale = Float64(iters) * 1000.0
    print(label
        + ": total=" + String(Float64(total_ns) / scale) + " us"
        + "  gate(last_worker)=" + String(Float64(gate.done_from_start_ns) / scale) + " us"
        + "  up(last_worker)=" + String(Float64(up.done_from_start_ns) / scale) + " us"
        + "  act=" + String(Float64(act_ns) / scale) + " us"
        + "  down(last_worker)=" + String(Float64(down.done_from_start_ns) / scale) + " us"
        + "  join_oh=" + String(Float64(gate.join_overhead_ns + up.join_overhead_ns + down.join_overhead_ns) / Float64(iters)) + " ns"
        + "  checksum=" + String(checksum))


def print_overlap_timing(label: String, iters: Int, timing: OverlapTiming, checksum: Float32):
    var scale = Float64(iters) * 1000.0
    print(label
        + ": wall=" + String(Float64(timing.wall_ns) / scale) + " us"
        + "  moe_dispatch=" + String(Float64(timing.moe_dispatch_ns) / scale) + " us"
        + "  moe_done=" + String(Float64(timing.moe_done_ns) / scale) + " us"
        + "  dense_done=" + String(Float64(timing.dense_done_ns) / scale) + " us"
        + "  dense_total=" + String(Float64(timing.dense_total_ns) / scale) + " us"
        + "  moe_accumulate=" + String(Float64(timing.moe_accumulate_ns) / scale) + " us"
        + "  done_spread=" + String(Float64(timing.done_spread_ns) / scale) + " us"
        + "  tail_after_last_worker=" + String(Float64(timing.tail_after_last_worker_ns) / Float64(iters)) + " ns"
        + "  checksum=" + String(checksum))


def fill_input[hidden: Int](ptr: BF16Ptr):
    for i in range(hidden):
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 29) - 14) * Float32(0.0625))


def fill_dense_weight[rows: Int, cols: Int](ptr: BF16Ptr, stride_bias: Int):
    for i in range(rows * cols):
        ptr[i] = Scalar[DType.bfloat16](Float32((i + stride_bias) % 97 - 48) * Float32(0.01))


def fill_moe_weights[used_experts: Int, intermediate: Int, hidden: Int](
    gate_up_all: BF16Ptr,
    down_all: BF16Ptr,
):
    var gate_up_dim = 2 * intermediate
    for e in range(used_experts):
        for i in range(gate_up_dim * hidden):
            gate_up_all[e * gate_up_dim * hidden + i] = Scalar[DType.bfloat16](
                Float32((e * 17 + i) % 101 - 50) * Float32(0.01)
            )
        for i in range(hidden * intermediate):
            down_all[e * hidden * intermediate + i] = Scalar[DType.bfloat16](
                Float32((e * 13 + i) % 89 - 44) * Float32(0.01)
            )


def make_routing[top_k: Int]() -> Gemma4TopKResult[top_k]:
    var routing = Gemma4TopKResult[top_k](
        indices=InlineArray[Int, top_k](fill=0),
        weights=InlineArray[Float32, top_k](fill=Float32(0)),
    )

    comptime assert top_k <= 8, "benchmark helper assumes top_k <= 8"

    for i in range(top_k):
        routing.indices[i] = i
        if i == 0:
            routing.weights[i] = Float32(0.23)
        elif i == 1:
            routing.weights[i] = Float32(0.18)
        elif i == 2:
            routing.weights[i] = Float32(0.15)
        elif i == 3:
            routing.weights[i] = Float32(0.12)
        elif i == 4:
            routing.weights[i] = Float32(0.11)
        elif i == 5:
            routing.weights[i] = Float32(0.08)
        elif i == 6:
            routing.weights[i] = Float32(0.07)
        else:
            routing.weights[i] = Float32(0.06)
    return routing^


def accumulate_expert_outputs[top_k: Int, hidden: Int](expert_out_buf: BF16Ptr, dst: BF16Ptr):
    comptime width = simd_width_of[DType.float32]()
    for i in range(0, hidden, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(top_k):
            acc += (expert_out_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())


def timed_decode_gemm[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped](
    input: DynView[InT],
    weight: Bound[W],
    output: DynView[OutT],
    mut pool: BurstPool[],
) -> DispatchTiming where W.DTYPE == DType.bfloat16:
    var t0 = Int(perf_counter_ns())
    var pool_ptr = gemm(input, weight, output, pool).take()
    var t1 = Int(perf_counter_ns())
    if pool_ptr:
        pool_ptr[].join()
    var t2 = Int(perf_counter_ns())
    var done_ts = pool.last_worker_timestamp()
    return DispatchTiming(
        dispatch_ns=t1 - t0,
        wall_ns=t2 - t0,
        done_from_start_ns=done_ts - t0,
        join_overhead_ns=t2 - done_ts,
        done_ts_abs=done_ts,
        end_ts_abs=t2,
    )


def timed_moe_dispatch[top_k: Int, intermediate: Int, hidden: Int](
    expert_input: BF16Ptr,
    routing: Gemma4TopKResult[top_k],
    expert_gate_up_base: BF16Ptr,
    expert_down_base: BF16Ptr,
    expert_out_buf: BF16Ptr,
    dst: BF16Ptr,
    mut pool: BurstPool[],
) -> MoETiming:
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

    var t0 = Int(perf_counter_ns())
    pool.dispatch[Gemma4ExpertFFNArgs, gemma4_expert_ffn_kernel[intermediate, hidden]](
        UnsafePointer(to=jobs[0]), top_k)
    var t1 = Int(perf_counter_ns())
    pool.join()
    var t2 = Int(perf_counter_ns())
    var done_ts = pool.last_worker_timestamp()

    var t3 = Int(perf_counter_ns())
    accumulate_expert_outputs[top_k, hidden](expert_out_buf, dst)
    var t4 = Int(perf_counter_ns())

    return MoETiming(
        dispatch_ns=t1 - t0,
        worker_done_ns=done_ts - t0,
        join_overhead_ns=t2 - done_ts,
        accumulate_ns=t4 - t3,
        total_ns=t4 - t0,
        done_ts_abs=done_ts,
        end_ts_abs=t4,
    )


def bench_expert_ffn_direct[intermediate: Int, hidden: Int](label: String, iters: Int):
    comptime gate_up_dim = 2 * intermediate

    var inp = alloc[Scalar[DType.bfloat16]](hidden)
    var gate_up = alloc[Scalar[DType.bfloat16]](gate_up_dim * hidden)
    var down = alloc[Scalar[DType.bfloat16]](hidden * intermediate)
    var out = alloc[Scalar[DType.bfloat16]](hidden)

    fill_input[hidden](BF16Ptr(unsafe_from_address=Int(inp)))
    fill_dense_weight[gate_up_dim, hidden](BF16Ptr(unsafe_from_address=Int(gate_up)), 3)
    fill_dense_weight[hidden, intermediate](BF16Ptr(unsafe_from_address=Int(down)), 11)

    var args = Gemma4ExpertFFNArgs(
        BF16Ptr(unsafe_from_address=Int(inp)),
        BF16Ptr(unsafe_from_address=Int(gate_up)),
        BF16Ptr(unsafe_from_address=Int(down)),
        BF16Ptr(unsafe_from_address=Int(out)),
        Float32(0.125),
    )

    for _ in range(3):
        gemma4_expert_ffn_kernel[intermediate, hidden](args)

    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        gemma4_expert_ffn_kernel[intermediate, hidden](args)
    var t1 = Int(perf_counter_ns())

    print_simple_timing(label, iters, t1 - t0, checksum_bf16(BF16Ptr(unsafe_from_address=Int(out)), hidden))

    inp.free()
    gate_up.free()
    down.free()
    out.free()


def bench_moe_dispatch[num_experts: Int, top_k: Int, intermediate: Int, hidden: Int](
    label: String,
    iters: Int,
    mut pool: BurstPool[],
):
    comptime used_experts = top_k

    var inp = alloc[Scalar[DType.bfloat16]](hidden)
    var gate_up_all = alloc[Scalar[DType.bfloat16]](used_experts * 2 * intermediate * hidden)
    var down_all = alloc[Scalar[DType.bfloat16]](used_experts * hidden * intermediate)
    var expert_out = alloc[Scalar[DType.bfloat16]](top_k * hidden)
    var dst = alloc[Scalar[DType.bfloat16]](hidden)

    fill_input[hidden](BF16Ptr(unsafe_from_address=Int(inp)))
    fill_moe_weights[used_experts, intermediate, hidden](
        BF16Ptr(unsafe_from_address=Int(gate_up_all)),
        BF16Ptr(unsafe_from_address=Int(down_all)),
    )
    var routing = make_routing[top_k]()

    for _ in range(3):
        _ = timed_moe_dispatch[top_k, intermediate, hidden](
            BF16Ptr(unsafe_from_address=Int(inp)),
            routing,
            BF16Ptr(unsafe_from_address=Int(gate_up_all)),
            BF16Ptr(unsafe_from_address=Int(down_all)),
            BF16Ptr(unsafe_from_address=Int(expert_out)),
            BF16Ptr(unsafe_from_address=Int(dst)),
            pool,
        )

    var total = MoETiming(0, 0, 0, 0, 0, 0, 0)
    for _ in range(iters):
        var t = timed_moe_dispatch[top_k, intermediate, hidden](
            BF16Ptr(unsafe_from_address=Int(inp)),
            routing,
            BF16Ptr(unsafe_from_address=Int(gate_up_all)),
            BF16Ptr(unsafe_from_address=Int(down_all)),
            BF16Ptr(unsafe_from_address=Int(expert_out)),
            BF16Ptr(unsafe_from_address=Int(dst)),
            pool,
        )
        total.dispatch_ns += t.dispatch_ns
        total.worker_done_ns += t.worker_done_ns
        total.join_overhead_ns += t.join_overhead_ns
        total.accumulate_ns += t.accumulate_ns
        total.total_ns += t.total_ns
    print_moe_timing(
        label,
        iters,
        total,
        checksum_bf16(BF16Ptr(unsafe_from_address=Int(dst)), hidden)
            + checksum_bf16(BF16Ptr(unsafe_from_address=Int(expert_out)), top_k * hidden),
    )

    inp.free()
    gate_up_all.free()
    down_all.free()
    expert_out.free()
    dst.free()


def bench_dense_branch[hidden: Int, intermediate: Int](
    label: String,
    iters: Int,
    mut pool: BurstPool[],
):
    comptime InView = Slot[BF16, Replicated, 1, hidden, 1]
    comptime MidView = Slot[BF16, Replicated, 1, intermediate, 1]
    comptime GateW = Slot[BF16, Replicated, intermediate, hidden, 1]
    comptime DownW = Slot[BF16, Replicated, hidden, intermediate, 1]

    var inp = alloc[Scalar[DType.bfloat16]](hidden)
    var gate_w = alloc[Scalar[DType.bfloat16]](intermediate * hidden)
    var up_w = alloc[Scalar[DType.bfloat16]](intermediate * hidden)
    var down_w = alloc[Scalar[DType.bfloat16]](hidden * intermediate)
    var gate = alloc[Scalar[DType.bfloat16]](intermediate)
    var up = alloc[Scalar[DType.bfloat16]](intermediate)
    var act = alloc[Scalar[DType.bfloat16]](intermediate)
    var out = alloc[Scalar[DType.bfloat16]](hidden)

    fill_input[hidden](BF16Ptr(unsafe_from_address=Int(inp)))
    fill_dense_weight[intermediate, hidden](BF16Ptr(unsafe_from_address=Int(gate_w)), 5)
    fill_dense_weight[intermediate, hidden](BF16Ptr(unsafe_from_address=Int(up_w)), 19)
    fill_dense_weight[hidden, intermediate](BF16Ptr(unsafe_from_address=Int(down_w)), 31)

    var inp_v = DynView[InView](Int(inp), 1)
    var gate_v = DynView[MidView](Int(gate), 1)
    var up_v = DynView[MidView](Int(up), 1)
    var act_v = DynView[MidView](Int(act), 1)
    var out_v = DynView[InView](Int(out), 1)
    var gate_b = Bound[GateW](Int(gate_w))
    var up_b = Bound[GateW](Int(up_w))
    var down_b = Bound[DownW](Int(down_w))

    for _ in range(2):
        _ = timed_decode_gemm[GateW, InView, MidView](inp_v, gate_b, gate_v, pool)
        _ = timed_decode_gemm[GateW, InView, MidView](inp_v, up_b, up_v, pool)
        gelu_tanh_mul(gate_v, up_v, act_v)
        _ = timed_decode_gemm[DownW, MidView, InView](act_v, down_b, out_v, pool)

    var gate_total = DispatchTiming(0, 0, 0, 0, 0, 0)
    var up_total = DispatchTiming(0, 0, 0, 0, 0, 0)
    var down_total = DispatchTiming(0, 0, 0, 0, 0, 0)
    var act_ns = 0
    var total_ns = 0
    for _ in range(iters):
        var branch_t0 = Int(perf_counter_ns())
        var gate_t = timed_decode_gemm[GateW, InView, MidView](inp_v, gate_b, gate_v, pool)
        var up_t = timed_decode_gemm[GateW, InView, MidView](inp_v, up_b, up_v, pool)
        var a0 = Int(perf_counter_ns())
        gelu_tanh_mul(gate_v, up_v, act_v)
        var a1 = Int(perf_counter_ns())
        var down_t = timed_decode_gemm[DownW, MidView, InView](act_v, down_b, out_v, pool)
        var branch_t1 = Int(perf_counter_ns())

        gate_total.dispatch_ns += gate_t.dispatch_ns
        gate_total.wall_ns += gate_t.wall_ns
        gate_total.done_from_start_ns += gate_t.done_from_start_ns
        gate_total.join_overhead_ns += gate_t.join_overhead_ns

        up_total.dispatch_ns += up_t.dispatch_ns
        up_total.wall_ns += up_t.wall_ns
        up_total.done_from_start_ns += up_t.done_from_start_ns
        up_total.join_overhead_ns += up_t.join_overhead_ns

        down_total.dispatch_ns += down_t.dispatch_ns
        down_total.wall_ns += down_t.wall_ns
        down_total.done_from_start_ns += down_t.done_from_start_ns
        down_total.join_overhead_ns += down_t.join_overhead_ns

        act_ns += a1 - a0
        total_ns += branch_t1 - branch_t0
    print_dense_timing(
        label,
        iters,
        gate_total,
        up_total,
        act_ns,
        down_total,
        total_ns,
        checksum_bf16(BF16Ptr(unsafe_from_address=Int(out)), hidden),
    )

    inp.free()
    gate_w.free()
    up_w.free()
    down_w.free()
    gate.free()
    up.free()
    act.free()
    out.free()


def run_overlap_once[top_k: Int, expert_intermediate: Int, hidden: Int, dense_intermediate: Int](
    expert_input: BF16Ptr,
    routing: Gemma4TopKResult[top_k],
    expert_gate_up_base: BF16Ptr,
    expert_down_base: BF16Ptr,
    expert_out_buf: BF16Ptr,
    moe_dst: BF16Ptr,
    inp_v: DynView[Slot[BF16, Replicated, 1, hidden, 1]],
    gate_b: Bound[Slot[BF16, Replicated, dense_intermediate, hidden, 1]],
    up_b: Bound[Slot[BF16, Replicated, dense_intermediate, hidden, 1]],
    down_b: Bound[Slot[BF16, Replicated, hidden, dense_intermediate, 1]],
    gate_v: DynView[Slot[BF16, Replicated, 1, dense_intermediate, 1]],
    up_v: DynView[Slot[BF16, Replicated, 1, dense_intermediate, 1]],
    act_v: DynView[Slot[BF16, Replicated, 1, dense_intermediate, 1]],
    dense_out_v: DynView[Slot[BF16, Replicated, 1, hidden, 1]],
    mut pool_moe: BurstPool[],
    mut pool_dense: BurstPool[],
) -> OverlapTiming:
    comptime gate_up_dim = 2 * expert_intermediate
    comptime gate_up_expert_stride = gate_up_dim * hidden
    comptime down_expert_stride = hidden * expert_intermediate

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

    var t0 = Int(perf_counter_ns())
    pool_moe.dispatch[Gemma4ExpertFFNArgs, gemma4_expert_ffn_kernel[expert_intermediate, hidden]](
        UnsafePointer(to=jobs[0]), top_k)
    var t1 = Int(perf_counter_ns())

    _ = timed_decode_gemm[
        Slot[BF16, Replicated, dense_intermediate, hidden, 1],
        Slot[BF16, Replicated, 1, hidden, 1],
        Slot[BF16, Replicated, 1, dense_intermediate, 1],
    ](inp_v, gate_b, gate_v, pool_dense)
    _ = timed_decode_gemm[
        Slot[BF16, Replicated, dense_intermediate, hidden, 1],
        Slot[BF16, Replicated, 1, hidden, 1],
        Slot[BF16, Replicated, 1, dense_intermediate, 1],
    ](inp_v, up_b, up_v, pool_dense)
    gelu_tanh_mul(gate_v, up_v, act_v)
    var down_t = timed_decode_gemm[
        Slot[BF16, Replicated, hidden, dense_intermediate, 1],
        Slot[BF16, Replicated, 1, dense_intermediate, 1],
        Slot[BF16, Replicated, 1, hidden, 1],
    ](act_v, down_b, dense_out_v, pool_dense)

    pool_moe.join()
    var moe_done_ts = pool_moe.last_worker_timestamp()

    var acc0 = Int(perf_counter_ns())
    accumulate_expert_outputs[top_k, hidden](expert_out_buf, moe_dst)
    var acc1 = Int(perf_counter_ns())

    var dense_done = down_t.done_ts_abs - t0
    var moe_done = moe_done_ts - t0
    var spread = moe_done - dense_done
    if spread < 0:
        spread = -spread

    var critical_done = down_t.done_ts_abs
    if moe_done_ts > critical_done:
        critical_done = moe_done_ts

    return OverlapTiming(
        moe_dispatch_ns=t1 - t0,
        moe_done_ns=moe_done,
        dense_done_ns=dense_done,
        dense_total_ns=down_t.end_ts_abs - t0,
        moe_accumulate_ns=acc1 - acc0,
        wall_ns=acc1 - t0,
        done_spread_ns=spread,
        tail_after_last_worker_ns=acc1 - critical_done,
    )


def bench_overlap_dense_and_moe[top_k: Int, expert_intermediate: Int, hidden: Int, dense_intermediate: Int](
    label: String,
    iters: Int,
    mut pool_moe: BurstPool[],
    mut pool_dense: BurstPool[],
):
    comptime InView = Slot[BF16, Replicated, 1, hidden, 1]
    comptime MidView = Slot[BF16, Replicated, 1, dense_intermediate, 1]
    comptime GateW = Slot[BF16, Replicated, dense_intermediate, hidden, 1]
    comptime DownW = Slot[BF16, Replicated, hidden, dense_intermediate, 1]
    comptime used_experts = top_k

    var inp = alloc[Scalar[DType.bfloat16]](hidden)
    var gate_w = alloc[Scalar[DType.bfloat16]](dense_intermediate * hidden)
    var up_w = alloc[Scalar[DType.bfloat16]](dense_intermediate * hidden)
    var down_w = alloc[Scalar[DType.bfloat16]](hidden * dense_intermediate)
    var gate = alloc[Scalar[DType.bfloat16]](dense_intermediate)
    var up = alloc[Scalar[DType.bfloat16]](dense_intermediate)
    var act = alloc[Scalar[DType.bfloat16]](dense_intermediate)
    var dense_out = alloc[Scalar[DType.bfloat16]](hidden)

    var gate_up_all = alloc[Scalar[DType.bfloat16]](used_experts * 2 * expert_intermediate * hidden)
    var down_all = alloc[Scalar[DType.bfloat16]](used_experts * hidden * expert_intermediate)
    var expert_out = alloc[Scalar[DType.bfloat16]](top_k * hidden)
    var moe_dst = alloc[Scalar[DType.bfloat16]](hidden)

    fill_input[hidden](BF16Ptr(unsafe_from_address=Int(inp)))
    fill_dense_weight[dense_intermediate, hidden](BF16Ptr(unsafe_from_address=Int(gate_w)), 5)
    fill_dense_weight[dense_intermediate, hidden](BF16Ptr(unsafe_from_address=Int(up_w)), 19)
    fill_dense_weight[hidden, dense_intermediate](BF16Ptr(unsafe_from_address=Int(down_w)), 31)
    fill_moe_weights[used_experts, expert_intermediate, hidden](
        BF16Ptr(unsafe_from_address=Int(gate_up_all)),
        BF16Ptr(unsafe_from_address=Int(down_all)),
    )
    var routing = make_routing[top_k]()

    var inp_v = DynView[InView](Int(inp), 1)
    var gate_v = DynView[MidView](Int(gate), 1)
    var up_v = DynView[MidView](Int(up), 1)
    var act_v = DynView[MidView](Int(act), 1)
    var dense_out_v = DynView[InView](Int(dense_out), 1)
    var gate_b = Bound[GateW](Int(gate_w))
    var up_b = Bound[GateW](Int(up_w))
    var down_b = Bound[DownW](Int(down_w))

    for _ in range(2):
        _ = run_overlap_once[top_k, expert_intermediate, hidden, dense_intermediate](
            BF16Ptr(unsafe_from_address=Int(inp)),
            routing,
            BF16Ptr(unsafe_from_address=Int(gate_up_all)),
            BF16Ptr(unsafe_from_address=Int(down_all)),
            BF16Ptr(unsafe_from_address=Int(expert_out)),
            BF16Ptr(unsafe_from_address=Int(moe_dst)),
            inp_v,
            gate_b,
            up_b,
            down_b,
            gate_v,
            up_v,
            act_v,
            dense_out_v,
            pool_moe,
            pool_dense,
        )

    var total = OverlapTiming(0, 0, 0, 0, 0, 0, 0, 0)
    for _ in range(iters):
        var t = run_overlap_once[top_k, expert_intermediate, hidden, dense_intermediate](
            BF16Ptr(unsafe_from_address=Int(inp)),
            routing,
            BF16Ptr(unsafe_from_address=Int(gate_up_all)),
            BF16Ptr(unsafe_from_address=Int(down_all)),
            BF16Ptr(unsafe_from_address=Int(expert_out)),
            BF16Ptr(unsafe_from_address=Int(moe_dst)),
            inp_v,
            gate_b,
            up_b,
            down_b,
            gate_v,
            up_v,
            act_v,
            dense_out_v,
            pool_moe,
            pool_dense,
        )
        total.moe_dispatch_ns += t.moe_dispatch_ns
        total.moe_done_ns += t.moe_done_ns
        total.dense_done_ns += t.dense_done_ns
        total.dense_total_ns += t.dense_total_ns
        total.moe_accumulate_ns += t.moe_accumulate_ns
        total.wall_ns += t.wall_ns
        total.done_spread_ns += t.done_spread_ns
        total.tail_after_last_worker_ns += t.tail_after_last_worker_ns
    print_overlap_timing(
        label,
        iters,
        total,
        checksum_bf16(BF16Ptr(unsafe_from_address=Int(dense_out)), hidden)
            + checksum_bf16(BF16Ptr(unsafe_from_address=Int(moe_dst)), hidden),
    )

    inp.free()
    gate_w.free()
    up_w.free()
    down_w.free()
    gate.free()
    up.free()
    act.free()
    dense_out.free()
    gate_up_all.free()
    down_all.free()
    expert_out.free()
    moe_dst.free()


def main():
    print("=== experimental_gemma MoE microbench ===")
    print("Gemma4 decode shapes: hidden=2816  dense_intermediate=2112  moe_intermediate=704  top_k=8")

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])
    var pool_moe = BurstPool[].for_numa_node(numa, topo[0])
    var pool_dense = BurstPool[].for_numa_node(numa, topo[0])
    print("pool workers: single=" + String(pool.get_capacity())
        + "  overlap=" + String(pool_moe.get_capacity()) + "+" + String(pool_dense.get_capacity()))

    bench_expert_ffn_direct[704, 2816]("expert_ffn direct hidden=2816 intermediate=704", 30)
    bench_moe_dispatch[128, 8, 704, 2816]("moe_dispatch top_k=8 hidden=2816 intermediate=704", 20, pool)
    bench_dense_branch[2816, 2112]("dense_mlp decode hidden=2816 intermediate=2112", 20, pool)
    bench_overlap_dense_and_moe[8, 704, 2816, 2112](
        "dense_mlp + moe overlap (two pools)",
        20,
        pool_moe,
        pool_dense,
    )

    _ = pool^
    _ = pool_moe^
    _ = pool_dense^
