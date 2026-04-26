from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq, rms_reduce_bf16,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs


@fieldwise_init
struct Bf16Spec(Copyable, ImplicitlyCopyable):
    var has_gamma: Bool
    var has_residual: Bool


@fieldwise_init
struct I8Spec(Copyable, ImplicitlyCopyable):
    var has_gamma: Bool
    var fwht: Bool
    var per_block: Bool


@always_inline
def compute_inv_rms[cols: Int](src: BF16Ptr, eps: Float32) -> Float32:
    var sum_sq = rms_reduce_bf16[cols](src)
    return inv_rms_from_sum_sq(sum_sq, cols, eps)


@always_inline
def stage_x_compute_inv_rms[cols: Int](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
) -> Float32:
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
    return inv_rms_from_sum_sq(tree_reduce_accs(accs), cols, eps)


@always_inline
def write_bf16_normed[cols: Int, spec: Bf16Spec](
    src: BF16Ptr, inv_rms: Float32, gamma: BF16Ptr, dst: BF16Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)
    var k = 0
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var normed = x * vinv
        comptime if spec.has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            normed = normed * g
        comptime if spec.has_residual:
            var prev = (dst + k).load[width=width]().cast[DType.float32]()
            (dst + k).store((prev + normed).cast[DType.bfloat16]())
        else:
            (dst + k).store(normed.cast[DType.bfloat16]())
        k += width


@always_inline
def write_n_bf16_normed[cols: Int, N: Int, *specs: Bf16Spec](
    src: BF16Ptr, inv_rms: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[BF16Ptr, N],
):
    comptime assert len(specs) == N
    comptime for i in range(N):
        write_bf16_normed[cols, specs[i]](src, inv_rms, gammas[i], dsts[i])


@always_inline
def write_i8_quantized[
    cols: Int, block: Int, spec: I8Spec, in_place: Bool,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst: I8Ptr, scale: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)
    comptime if in_place:
        var k = 0
        while k < cols:
            var x = (work + k).load[width=width]()
            var normed = x * vinv
            comptime if spec.has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            (work + k).store(normed)
            k += width
        comptime if spec.fwht:
            for b in range(cols // block):
                fwht_block[block](work + b * block)
        comptime if spec.per_block:
            for b in range(cols // block):
                scale[b] = absmax_quantize_i8[block](
                    work + b * block, dst + b * block)
        else:
            scale[0] = absmax_quantize_i8[cols](work, dst)
    else:
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
        comptime if spec.per_block:
            for b in range(cols // block):
                scale[b] = absmax_quantize_i8[block](
                    local + b * block, dst + b * block)
        else:
            scale[0] = absmax_quantize_i8[cols](local, dst)
        _ = local_arr


@always_inline
def write_n_i8_quantized[cols: Int, block: Int, N: Int, *specs: I8Spec](
    work: F32Ptr, inv_rms: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[I8Ptr, N],
    scales: InlineArray[F32Ptr, N],
):
    comptime assert len(specs) == N
    comptime for i in range(N):
        comptime in_place = (i == N - 1)
        write_i8_quantized[cols, block, specs[i], in_place](
            work, inv_rms, gammas[i], dsts[i], scales[i])


@always_inline
def rmsnorm_n_bf16_row[cols: Int, N: Int, *specs: Bf16Spec](
    src: BF16Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[BF16Ptr, N],
):
    var inv = compute_inv_rms[cols](src, eps)
    write_n_bf16_normed[cols, N, *specs](src, inv, gammas, dsts)


@always_inline
def rmsnorm_n_i8_row[cols: Int, block: Int, N: Int, *specs: I8Spec](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[I8Ptr, N],
    scales: InlineArray[F32Ptr, N],
):
    var inv = stage_x_compute_inv_rms[cols](src, work, eps)
    write_n_i8_quantized[cols, block, N, *specs](
        work, inv, gammas, dsts, scales)


@always_inline
def rmsnorm_i8_and_bf16_row[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, scale: F32Ptr,
    normed_bf16: BF16Ptr,
    work: F32Ptr, eps: Float32,
):
    var inv = stage_x_compute_inv_rms[cols](src, work, eps)
    write_bf16_normed[cols, Bf16Spec(has_gamma=True, has_residual=False)](
        src, inv, full_gamma, normed_bf16)
    write_i8_quantized[cols, block,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
        in_place=True,
    ](work, inv, split_gamma, qi, scale)
