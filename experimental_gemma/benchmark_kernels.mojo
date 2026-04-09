"""Microbenchmarks for the experimental Gemma kernels.

This is intended for quick remote runs on the AMX-capable server via
`remote_build.fish experimental_gemma/benchmark_kernels.mojo`.
"""

from std.collections import InlineArray
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns

from numa import NumaInfo
from threading import BurstPool

from modeling.model_spec import BF16, Slot, Replicated, DynView, Bound
from experimental_gemma.norms import rmsnorm_no_scale, rmsnorm_per_head
from experimental_gemma.activations import gelu_tanh_mul
from experimental_gemma.router import softmax_topk_renorm


def checksum_bf16(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += Float32(ptr[i])
    return sum


def print_timing(label: String, iters: Int, elapsed_ns: Int, checksum: Float32):
    var per_iter_us = Float64(elapsed_ns) / Float64(iters) / 1000.0
    print(label + ": " + String(per_iter_us) + " us/iter  checksum=" + String(checksum))


def bench_rmsnorm_no_scale[seq_len: Int, cols: Int](
    label: String,
    iters: Int,
    mut pool: BurstPool[],
):
    comptime View = Slot[BF16, Replicated, seq_len, cols, 1]

    var inp = alloc[Scalar[DType.bfloat16]](seq_len * cols)
    var out = alloc[Scalar[DType.bfloat16]](seq_len * cols)

    for i in range(seq_len * cols):
        inp[i] = Scalar[DType.bfloat16](Float32((i % 31) - 15) * Float32(0.0625))
        out[i] = Scalar[DType.bfloat16](0.0)

    var inp_v = DynView[View](Int(inp), seq_len)
    var out_v = DynView[View](Int(out), seq_len)

    for _ in range(5):
        rmsnorm_no_scale(inp_v, out_v, pool, eps=Float32(1e-6)).join()

    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        rmsnorm_no_scale(inp_v, out_v, pool, eps=Float32(1e-6)).join()
    var t1 = Int(perf_counter_ns())

    print_timing(label, iters, t1 - t0, checksum_bf16(out, seq_len * cols))

    inp.free()
    out.free()


def bench_rmsnorm_per_head[seq_len: Int, head_dim: Int, num_heads: Int](
    label: String,
    iters: Int,
    mut pool: BurstPool[],
):
    comptime cols = head_dim * num_heads
    comptime View = Slot[BF16, Replicated, seq_len, cols, 1]
    comptime WeightView = Slot[BF16, Replicated, head_dim, 1, 1]

    var inp = alloc[Scalar[DType.bfloat16]](seq_len * cols)
    var wt = alloc[Scalar[DType.bfloat16]](head_dim)
    var out = alloc[Scalar[DType.bfloat16]](seq_len * cols)

    for i in range(seq_len * cols):
        inp[i] = Scalar[DType.bfloat16](Float32((i % 29) - 14) * Float32(0.078125))
        out[i] = Scalar[DType.bfloat16](0.0)
    for i in range(head_dim):
        wt[i] = Scalar[DType.bfloat16](Float32(0.75) + Float32(i % 13) * Float32(0.03125))

    var inp_v = DynView[View](Int(inp), seq_len)
    var out_v = DynView[View](Int(out), seq_len)
    var wt_b = Bound[WeightView](Int(wt))

    for _ in range(5):
        rmsnorm_per_head[head_dim, num_heads](inp_v, wt_b, out_v, pool, eps=Float32(1e-6)).join()

    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        rmsnorm_per_head[head_dim, num_heads](inp_v, wt_b, out_v, pool, eps=Float32(1e-6)).join()
    var t1 = Int(perf_counter_ns())

    print_timing(label, iters, t1 - t0, checksum_bf16(out, seq_len * cols))

    inp.free()
    wt.free()
    out.free()


def bench_gelu_tanh_mul[seq_len: Int, cols: Int](label: String, iters: Int):
    comptime View = Slot[BF16, Replicated, seq_len, cols, 1]

    var gate = alloc[Scalar[DType.bfloat16]](seq_len * cols)
    var up = alloc[Scalar[DType.bfloat16]](seq_len * cols)
    var out = alloc[Scalar[DType.bfloat16]](seq_len * cols)

    for i in range(seq_len * cols):
        gate[i] = Scalar[DType.bfloat16](Float32((i % 23) - 11) * Float32(0.125))
        up[i] = Scalar[DType.bfloat16](Float32(0.5) + Float32(i % 17) * Float32(0.0625))
        out[i] = Scalar[DType.bfloat16](0.0)

    var gate_v = DynView[View](Int(gate), seq_len)
    var up_v = DynView[View](Int(up), seq_len)
    var out_v = DynView[View](Int(out), seq_len)

    for _ in range(5):
        gelu_tanh_mul(gate_v, up_v, out_v)

    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        gelu_tanh_mul(gate_v, up_v, out_v)
    var t1 = Int(perf_counter_ns())

    print_timing(label, iters, t1 - t0, checksum_bf16(out, seq_len * cols))

    gate.free()
    up.free()
    out.free()


def bench_router[num_experts: Int, k: Int](label: String, iters: Int):
    var logits = alloc[Scalar[DType.bfloat16]](num_experts)
    var scales = alloc[Scalar[DType.bfloat16]](num_experts)

    for i in range(num_experts):
        logits[i] = Scalar[DType.bfloat16](Float32((i % 19) - 9) * Float32(0.25))
        scales[i] = Scalar[DType.bfloat16](Float32(0.8) + Float32(i % 11) * Float32(0.03125))

    var checksum = Float32(0)
    for _ in range(20):
        var warm = softmax_topk_renorm[num_experts, k](logits, scales)
        checksum += warm.weights[0]

    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var result = softmax_topk_renorm[num_experts, k](logits, scales)
        checksum += result.weights[0]
        checksum += Float32(result.indices[0])
    var t1 = Int(perf_counter_ns())

    print_timing(label, iters, t1 - t0, checksum)

    logits.free()
    scales.free()


def main():
    print("=== experimental_gemma kernel microbench ===")

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    print("-- decode-shaped --")
    bench_rmsnorm_no_scale[1, 2816]("rmsnorm_no_scale seq=1 cols=2816", 400, pool)
    bench_rmsnorm_per_head[1, 256, 16]("rmsnorm_per_head seq=1 head=256 heads=16", 400, pool)
    bench_rmsnorm_per_head[1, 512, 16]("rmsnorm_per_head seq=1 head=512 heads=16", 200, pool)
    bench_gelu_tanh_mul[1, 2112]("gelu_tanh_mul seq=1 cols=2112", 800)
    bench_router[128, 8]("softmax_topk_renorm experts=128 topk=8", 200000)

    print("-- prefill-shaped --")
    bench_rmsnorm_no_scale[256, 2816]("rmsnorm_no_scale seq=256 cols=2816", 20, pool)
    bench_rmsnorm_per_head[256, 256, 16]("rmsnorm_per_head seq=256 head=256 heads=16", 20, pool)
    bench_gelu_tanh_mul[256, 2112]("gelu_tanh_mul seq=256 cols=2112", 40)

    _ = pool^
