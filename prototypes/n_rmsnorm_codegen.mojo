"""Compare codegen between hand-rolled dual norm and the n-output design."""
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from compile import compile_info

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs

# Direct copies of the row bodies we want to compare. Both are inlined
# `comptime fn`-equivalents so we get a single function we can dump.

# ============================================================================
# Hand-rolled (current minimax) dual-output row.
# ============================================================================

@always_inline
def existing_dual_output_row[cols: Int, block: Int](
    src: BF16Ptr,
    split_gamma: BF16Ptr,
    full_gamma: BF16Ptr,
    qi: I8Ptr,
    work: F32Ptr,
    scale: F32Ptr,
    normed_bf16: BF16Ptr,
    eps: Float32,
):
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
    var sum_sq = tree_reduce_accs(accs)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)

    k = 0
    while k < cols:
        var x = (work + k).load[width=width]()
        var normed = x * vinv
        var sg = (split_gamma + k).load[width=width]().cast[DType.float32]()
        var fg = (full_gamma + k).load[width=width]().cast[DType.float32]()
        (work + k).store(normed * sg)
        (normed_bf16 + k).store((normed * fg).cast[DType.bfloat16]())
        k += width

    for b in range(cols // block):
        fwht_block[block](work + b * block)
    scale[0] = absmax_quantize_i8[cols](work, qi)


# ============================================================================
# N-output version, hard-instantiated for the same dual-output shape.
# ============================================================================

@fieldwise_init
struct EmitSpec(Copyable, ImplicitlyCopyable):
    var mode: Int
    var has_gamma: Bool
    var fwht: Bool

comptime EMIT_BF16: Int = 0
comptime EMIT_I8_PER_ROW: Int = 2

@always_inline
def emit_one[
    cols: Int, block: Int, spec: EmitSpec,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst_addr: Int, scale: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)

    comptime if spec.mode == EMIT_BF16:
        var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        var k = 0
        while k < cols:
            var x = (work + k).load[width=width]()
            var normed = x * vinv
            comptime if spec.has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            (dst + k).store(normed.cast[DType.bfloat16]())
            k += width
        return

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
    scale[0] = absmax_quantize_i8[cols](local, dst_i8)
    _ = local_arr


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
    comptime assert len(specs) == N
    var sum_sq = load_to_work_and_reduce[cols](src, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    comptime for i in range(N):
        emit_one[cols, block, specs[i]](
            work, inv, gamma_ptrs[i], dst_addrs[i], scale_ptrs[i])


# ============================================================================
# Top-level shims so we can dump LLVM for each.
# ============================================================================

def existing_shim[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    existing_dual_output_row[cols, block](
        src, split_gamma, full_gamma, qi, work, scale, normed,
        Float32(1e-6))


def n_shim[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [split_gamma, full_gamma]
    var dst_addrs: InlineArray[Int, 2] = [Int(qi), Int(normed)]
    var unused_scale_arr = InlineArray[Float32, 1](fill=Float32(0))
    var scale_ptrs: InlineArray[F32Ptr, 2] = [
        scale,
        UnsafePointer(to=unused_scale_arr).bitcast[Float32]()]
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
        EmitSpec(mode=EMIT_BF16, has_gamma=True, fwht=False),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused_scale_arr


@always_inline
def existing_single_i8_row[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scale: F32Ptr, eps: Float32,
):
    """Single-output i8 with FWHT (the attn_quantize path).
    Mirror of rmsnorm_fwht_quant_row[has_gamma=True, per_block=False]."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width

    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        var g = (gamma + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work + off).store(x * g)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            var g = (gamma + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work + off).store(x * g)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var g = (gamma + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work + k).store(x * g)
        k += width
    var sum_sq = tree_reduce_accs(accs)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    k = 0
    while k < cols:
        (work + k).store((work + k).load[width=width]() * vinv)
        k += width
    for b in range(cols // block):
        fwht_block[block](work + b * block)
    scale[0] = absmax_quantize_i8[cols](work, qi)


def existing_single_shim[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scale: F32Ptr,
):
    existing_single_i8_row[cols, block](src, gamma, qi, work, scale, Float32(1e-6))


def n_single_shim[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scale: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [gamma]
    var dst_addrs: InlineArray[Int, 1] = [Int(qi)]
    var scale_ptrs: InlineArray[F32Ptr, 1] = [scale]
    n_rmsnorm_row[cols, block, 1,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def report(label: String, asm_str: String):
    print("---", label, "---")
    print("  fmadd:        ", asm_str.count("vfmadd"))
    print("  movups:       ", asm_str.count("vmovups"))
    print("  mulps:        ", asm_str.count("vmulps"))
    print("  call:         ", asm_str.count("call"))
    print("  bytes:        ", asm_str.byte_length())


def main():
    var ex_dual = String(compile_info[
        existing_shim[3072, 128], emission_kind="asm"]().asm)
    var n_dual = String(compile_info[
        n_shim[3072, 128], emission_kind="asm"]().asm)
    var ex_single = String(compile_info[
        existing_single_shim[3072, 128], emission_kind="asm"]().asm)
    var n_single = String(compile_info[
        n_single_shim[3072, 128], emission_kind="asm"]().asm)

    print()
    print("== DUAL OUTPUT (i8 fwht + bf16) ==")
    report("existing dual_output_row", ex_dual)
    report("n_rmsnorm_row N=2", n_dual)

    print()
    print("== SINGLE OUTPUT (i8 fwht only) ==")
    report("existing rmsnorm_fwht_quant_row", ex_single)
    report("n_rmsnorm_row N=1", n_single)
