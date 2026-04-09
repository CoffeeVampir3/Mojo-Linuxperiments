"""Validate rmsnorm_no_scale and rmsnorm_per_head.

Reference: rms_norm(x, eps) = x / sqrt(mean(x^2) + eps)
Per-head: same but applied per head_dim segment, then scaled by weight.

References compute in f32 to match the kernel's internal precision.
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs, sqrt
from std.sys.info import simd_width_of

from modeling.model_spec import BF16, Slot, Replicated, DynView, Bound
from experimental_gemma.norms import rmsnorm_no_scale, rmsnorm_per_head

from numa import NumaInfo
from threading import BurstPool


def f32_ref_no_scale(inp: UnsafePointer[Scalar[DType.bfloat16], _], dst: UnsafePointer[Float32, MutAnyOrigin], cols: Int, eps: Float32):
    """Reference rmsnorm_no_scale in f32, matching kernel precision."""
    var sum_sq = Float32(0)
    for j in range(cols):
        var v = Float32(inp[j])
        sum_sq += v * v
    var scale = Float32(1.0) / sqrt(sum_sq / Float32(cols) + eps)
    for j in range(cols):
        dst[j] = Float32(inp[j]) * scale


def f32_ref_per_head(
    inp: UnsafePointer[Scalar[DType.bfloat16], _],
    wt: UnsafePointer[Scalar[DType.bfloat16], _],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    head_dim: Int, num_heads: Int, eps: Float32,
):
    """Reference rmsnorm_per_head in f32, matching kernel precision."""
    for h in range(num_heads):
        var sum_sq = Float32(0)
        for j in range(head_dim):
            var v = Float32(inp[h * head_dim + j])
            sum_sq += v * v
        var scale = Float32(1.0) / sqrt(sum_sq / Float32(head_dim) + eps)
        for j in range(head_dim):
            var normed = Float32(inp[h * head_dim + j]) * scale
            var w = Float32(wt[j])
            dst[h * head_dim + j] = normed * w


def test_rmsnorm_no_scale():
    print("=== rmsnorm_no_scale error decomposition ===")

    comptime COLS = 32
    comptime View = Slot[BF16, Replicated, 1, COLS, 1]

    var inp = alloc[Scalar[DType.bfloat16]](COLS)
    var dst = alloc[Scalar[DType.bfloat16]](COLS)
    var f32_dst = alloc[Float32](COLS)

    for i in range(COLS):
        inp[i] = Scalar[DType.bfloat16](Float32(-1.5) + Float32(i) * Float32(0.1))
        dst[i] = Scalar[DType.bfloat16](0.0)

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    var inp_v = DynView[View](Int(inp), 1)
    var dst_v = DynView[View](Int(dst), 1)
    rmsnorm_no_scale(inp_v, dst_v, pool, eps=Float32(1e-6)).join()

    f32_ref_no_scale(inp, f32_dst, COLS, Float32(1e-6))

    print("  idx | input(bf16)  | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+--------------+---------")

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for i in range(COLS):
        var iv = Float32(inp[i])
        var kv = Float32(dst[i])
        var fv = f32_dst[i]
        var bv = Float32(Scalar[DType.bfloat16](fv))

        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err

        print("  " + String(i) + " | " + String(iv) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    # output rms should be ~1.0
    var sum_sq = Float32(0)
    for i in range(COLS):
        var v = Float32(dst[i])
        sum_sq += v * v
    var output_rms = sqrt(sum_sq / Float32(COLS))
    print("  output rms=" + String(output_rms) + " (should be ~1.0)")
    print()

    inp.free()
    dst.free()
    f32_dst.free()
    _ = pool^


def test_rmsnorm_per_head():
    print("=== rmsnorm_per_head error decomposition ===")

    comptime HEAD_DIM = 8
    comptime NUM_HEADS = 4
    comptime COLS = HEAD_DIM * NUM_HEADS
    comptime View = Slot[BF16, Replicated, 1, COLS, 1]
    comptime WeightView = Slot[BF16, Replicated, HEAD_DIM, 1, 1]

    var inp = alloc[Scalar[DType.bfloat16]](COLS)
    var wt = alloc[Scalar[DType.bfloat16]](HEAD_DIM)
    var dst = alloc[Scalar[DType.bfloat16]](COLS)
    var f32_dst = alloc[Float32](COLS)

    # Different magnitudes per head to test independent normalization
    for h in range(NUM_HEADS):
        var head_scale = Float32(0.5) + Float32(h) * Float32(1.5)
        for j in range(HEAD_DIM):
            inp[h * HEAD_DIM + j] = Scalar[DType.bfloat16](
                (Float32(-0.5) + Float32(j) * Float32(0.15)) * head_scale
            )
            dst[h * HEAD_DIM + j] = Scalar[DType.bfloat16](0.0)

    for j in range(HEAD_DIM):
        wt[j] = Scalar[DType.bfloat16](Float32(0.8) + Float32(j) * Float32(0.05))

    f32_ref_per_head(inp, wt, f32_dst, HEAD_DIM, NUM_HEADS, Float32(1e-6))

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    var inp_v = DynView[View](Int(inp), 1)
    var wt_b = Bound[WeightView](Int(wt))
    var dst_v = DynView[View](Int(dst), 1)
    rmsnorm_per_head[HEAD_DIM, NUM_HEADS](inp_v, wt_b, dst_v, pool, eps=Float32(1e-6)).join()

    print("  head | idx | input(bf16) | kernel(bf16) | bf16(f32ref) | f32ref      | kern_err    | bf16_err")
    print("  -----+-----+-------------+--------------+--------------+-------------+-------------+---------")

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for h in range(NUM_HEADS):
        for j in range(HEAD_DIM):
            var idx = h * HEAD_DIM + j
            var iv = Float32(inp[idx])
            var kv = Float32(dst[idx])
            var fv = f32_dst[idx]
            var bv = Float32(Scalar[DType.bfloat16](fv))

            var kern_err = abs(Float64(kv) - Float64(bv))
            var bf16_err = abs(Float64(fv) - Float64(bv))
            if kern_err > max_kern_err:
                max_kern_err = kern_err
            if bf16_err > max_bf16_err:
                max_bf16_err = bf16_err

            print("  " + String(h) + " | " + String(j) + " | " + String(iv) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    # Per-head rms check: divide out weight, rms should be ~1.0
    print("  per-head output rms (pre-weight):")
    for h in range(NUM_HEADS):
        var ss = Float32(0)
        for j in range(HEAD_DIM):
            var v = Float32(dst[h * HEAD_DIM + j])
            var w = Float32(wt[j])
            var normed = v / w
            ss += normed * normed
        print("    head " + String(h) + ": rms=" + String(sqrt(ss / Float32(HEAD_DIM))))

    inp.free()
    wt.free()
    dst.free()
    f32_dst.free()
    _ = pool^


def main():
    test_rmsnorm_no_scale()
    test_rmsnorm_per_head()
