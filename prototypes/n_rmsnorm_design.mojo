from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from std.math import sqrt

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs


# ============================================================================
# Emit spec: comptime descriptor for one output path.
# ============================================================================

@fieldwise_init
struct EmitSpec(Copyable, ImplicitlyCopyable):
    var mode: Int
    var has_gamma: Bool
    var fwht: Bool


comptime EMIT_BF16: Int = 0
comptime EMIT_BF16_RESIDUAL: Int = 1
comptime EMIT_I8_PER_ROW: Int = 2
comptime EMIT_I8_PER_BLOCK: Int = 3


# ============================================================================
# Per-spec scale stride at comptime.
# ============================================================================

@always_inline
def scale_stride[spec: EmitSpec, cols: Int, block: Int]() -> Int:
    comptime if spec.mode == EMIT_I8_PER_BLOCK:
        return cols // block
    return 1


# ============================================================================
# Pass-1: load src to f32 work buffer, accumulate sum(x^2).
# Identical to the existing load-without-gamma helper — gamma is NOT fused.
# ============================================================================

@always_inline
def load_to_work_and_reduce[cols: Int](src: BF16Ptr, work: F32Ptr) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width

    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work + off).store(x)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work + off).store(x)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work + k).store(x)
        k += width
    return tree_reduce_accs(accs)


# ============================================================================
# Pass-2: emit one output. All branches resolved at comptime by `spec`.
# ============================================================================

@always_inline
def emit_one[
    cols: Int, block: Int, spec: EmitSpec,
](
    work: F32Ptr,                    # raw x in f32 (shared across outputs)
    inv_rms: Float32,
    gamma: BF16Ptr,                  # ignored if !spec.has_gamma
    dst_addr: Int,                   # type-erased; reinterpreted per spec
    scale: F32Ptr,                   # length depends on spec.mode
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)

    comptime if spec.mode == EMIT_BF16 or spec.mode == EMIT_BF16_RESIDUAL:
        var dst_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        var k = 0
        while k < cols:
            var x = (work + k).load[width=width]()
            var normed = x * vinv
            comptime if spec.has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            comptime if spec.mode == EMIT_BF16_RESIDUAL:
                var prev = (dst_bf16 + k).load[width=width]().cast[DType.float32]()
                (dst_bf16 + k).store((prev + normed).cast[DType.bfloat16]())
            else:
                (dst_bf16 + k).store(normed.cast[DType.bfloat16]())
            k += width
        return

    # i8 modes: write through a private scratch buffer because the source
    # `work` is shared with sibling outputs and must stay as raw x.
    var local_arr = InlineArray[Float32, cols](uninitialized=True)
    var local = UnsafePointer(to=local_arr).bitcast[Float32]()

    var k = 0
    while k < cols:
        var x = (work + k).load[width=width]()
        var normed = x * vinv
        comptime if spec.has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            normed = normed * g
        (local + k).store(normed)
        k += width

    comptime if spec.fwht:
        for b in range(cols // block):
            fwht_block[block](local + b * block)

    var dst_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=dst_addr)
    comptime if spec.mode == EMIT_I8_PER_BLOCK:
        for b in range(cols // block):
            scale[b] = absmax_quantize_i8[block](
                local + b * block, dst_i8 + b * block)
    else:
        scale[0] = absmax_quantize_i8[cols](local, dst_i8)

    _ = local_arr


# ============================================================================
# N-output RMSNorm row body.
# ============================================================================

@always_inline
def n_rmsnorm_row[
    cols: Int, block: Int, N: Int,
    *specs: EmitSpec,
](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gamma_ptrs: InlineArray[BF16Ptr, N],
    dst_addrs: InlineArray[Int, N],
    scale_ptrs: InlineArray[F32Ptr, N],
):
    comptime assert len(specs) == N, "len(specs) must equal N"
    var sum_sq = load_to_work_and_reduce[cols](src, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    comptime for i in range(N):
        emit_one[cols, block, specs[i]](
            work, inv, gamma_ptrs[i], dst_addrs[i], scale_ptrs[i])


# ============================================================================
# End-to-end test: dual-output (i8 fwht per-row + bf16) — minimax dual_norm.
# ============================================================================

def main():
    comptime cols = 32
    comptime block = 16
    comptime N = 2

    var src_arr = InlineArray[Scalar[DType.bfloat16], cols](fill=Scalar[DType.bfloat16](0))
    var src = UnsafePointer(to=src_arr).bitcast[Scalar[DType.bfloat16]]()
    for i in range(cols):
        src[i] = Scalar[DType.bfloat16](Float32(i + 1) * 0.1)

    var split_gamma_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](0.5))
    var full_gamma_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](1.0))
    var split_gamma = UnsafePointer(to=split_gamma_arr).bitcast[Scalar[DType.bfloat16]]()
    var full_gamma = UnsafePointer(to=full_gamma_arr).bitcast[Scalar[DType.bfloat16]]()

    var work_arr = InlineArray[Float32, cols](uninitialized=True)
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()

    var qi_arr = InlineArray[Scalar[DType.int8], cols](fill=Scalar[DType.int8](0))
    var qi = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()
    var qi_scale_arr = InlineArray[Float32, 1](fill=Float32(0))
    var qi_scale = UnsafePointer(to=qi_scale_arr).bitcast[Float32]()

    var bf16_out_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](0))
    var bf16_out = UnsafePointer(to=bf16_out_arr).bitcast[Scalar[DType.bfloat16]]()
    var unused_scale_arr = InlineArray[Float32, 1](fill=Float32(0))
    var unused_scale = UnsafePointer(to=unused_scale_arr).bitcast[Float32]()

    var gamma_ptrs: InlineArray[BF16Ptr, N] = [split_gamma, full_gamma]
    var dst_addrs: InlineArray[Int, N] = [Int(qi), Int(bf16_out)]
    var scale_ptrs: InlineArray[F32Ptr, N] = [qi_scale, unused_scale]

    n_rmsnorm_row[cols, block, N,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
        EmitSpec(mode=EMIT_BF16, has_gamma=True, fwht=False),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)

    print("i8 scale =", qi_scale[0])
    print("first 4 i8 outputs:", Int(qi[0]), Int(qi[1]), Int(qi[2]), Int(qi[3]))
    print("first 4 bf16 outputs:",
        Float32(bf16_out[0]), Float32(bf16_out[1]),
        Float32(bf16_out[2]), Float32(bf16_out[3]))
