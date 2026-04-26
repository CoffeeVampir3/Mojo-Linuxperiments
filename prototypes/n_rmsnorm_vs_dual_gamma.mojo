"""Direct comparison: existing rmsnorm_dual_gamma vs unified N=2 dual i8.

The user's question: does the existing dual-gamma worker get a 2x efficiency
win from sharing pass 1, that the unified design loses?

Answer: no — both designs share pass 1. The difference is only in pass 2's
arithmetic ordering. Numbers below.
"""
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


# ============================================================================
# Faithful copy of experimental3.kernels.rmsnorm.rmsnorm_dual_gamma_fwht_quant_row
# ============================================================================

@always_inline
def existing_load_and_reduce_dual[cols: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, cols]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var x = (src + off).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
        accs[i] = x * x
        (work_a + off).store(x * ga)
        (work_b + off).store(x * gb)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            var ga = (gamma_a + off).load[width=width]().cast[DType.float32]()
            var gb = (gamma_b + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            (work_a + off).store(x * ga)
            (work_b + off).store(x * gb)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        var ga = (gamma_a + k).load[width=width]().cast[DType.float32]()
        var gb = (gamma_b + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        (work_a + k).store(x * ga)
        (work_b + k).store(x * gb)
        k += width
    return tree_reduce_accs(accs)


@always_inline
def existing_normalize_inplace[cols: Int](work: F32Ptr, inv: Float32):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
        (work + k).store((work + k).load[width=width]() * vinv)
        k += width


@always_inline
def existing_dual_gamma_row[cols: Int, block: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    qi_a: I8Ptr, qi_b: I8Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
    scale_a: F32Ptr, scale_b: F32Ptr,
    eps: Float32,
):
    var sum_sq = existing_load_and_reduce_dual[cols](
        src, gamma_a, gamma_b, work_a, work_b)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    existing_normalize_inplace[cols](work_a, inv)
    existing_normalize_inplace[cols](work_b, inv)
    for b in range(cols // block):
        fwht_block[block](work_a + b * block)
    for b in range(cols // block):
        fwht_block[block](work_b + b * block)
    scale_a[0] = absmax_quantize_i8[cols](work_a, qi_a)
    scale_b[0] = absmax_quantize_i8[cols](work_b, qi_b)


# ============================================================================
# Unified N=2 dual-i8 (with last-i8 in-place optimization)
# ============================================================================

@fieldwise_init
struct EmitSpec(Copyable, ImplicitlyCopyable):
    var mode: Int
    var has_gamma: Bool
    var fwht: Bool


comptime EMIT_BF16: Int = 0
comptime EMIT_I8_PER_ROW: Int = 2


@always_inline
def is_i8[spec: EmitSpec]() -> Bool:
    return spec.mode == EMIT_I8_PER_ROW


comptime LastI8Index[N: Int, *specs: EmitSpec]: Int = _last_i8_impl[N, *specs]()


@always_inline
def _last_i8_impl[N: Int, *specs: EmitSpec]() -> Int:
    var idx = N
    comptime for i in range(N):
        comptime if is_i8[specs[i]]():
            idx = i
    return idx


@always_inline
def emit_one_i8[
    cols: Int, block: Int, has_gamma: Bool, fwht: Bool, in_place: Bool,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst_addr: Int, scale: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)

    comptime if in_place:
        var k = 0
        while k < cols:
            var x = (work + k).load[width=width]()
            var normed = x * vinv
            comptime if has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            (work + k).store(normed)
            k += width
        comptime if fwht:
            for b in range(cols // block):
                fwht_block[block](work + b * block)
        var dst_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        scale[0] = absmax_quantize_i8[cols](work, dst_i8)
    else:
        var local_arr = InlineArray[Float32, cols](uninitialized=True)
        var local = UnsafePointer(to=local_arr).bitcast[Float32]()
        var k = 0
        while k < cols:
            var x = (work + k).load[width=width]()
            var normed = x * vinv
            comptime if has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            (local + k).store(normed)
            k += width
        comptime if fwht:
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
    cols: Int, block: Int, N: Int, *specs: EmitSpec,
](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gamma_ptrs: InlineArray[BF16Ptr, N],
    dst_addrs: InlineArray[Int, N],
    scale_ptrs: InlineArray[F32Ptr, N],
):
    comptime assert len(specs) == N
    comptime last_i8 = LastI8Index[N, *specs]
    var sum_sq = load_to_work_and_reduce[cols](src, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    comptime for i in range(N):
        comptime in_place = (i == last_i8)
        emit_one_i8[cols, block, specs[i].has_gamma, specs[i].fwht, in_place](
            work, inv, gamma_ptrs[i], dst_addrs[i], scale_ptrs[i])


# ============================================================================
# Shims for codegen comparison
# ============================================================================

def existing_dual_gamma_shim[cols: Int, block: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    qi_a: I8Ptr, qi_b: I8Ptr,
    work_a: F32Ptr, work_b: F32Ptr,
    scale_a: F32Ptr, scale_b: F32Ptr,
):
    existing_dual_gamma_row[cols, block](
        src, gamma_a, gamma_b, qi_a, qi_b,
        work_a, work_b, scale_a, scale_b, Float32(1e-6))


def n_dual_i8_shim[cols: Int, block: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    qi_a: I8Ptr, qi_b: I8Ptr, work: F32Ptr,
    scale_a: F32Ptr, scale_b: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [gamma_a, gamma_b]
    var dst_addrs: InlineArray[Int, 2] = [Int(qi_a), Int(qi_b)]
    var scale_ptrs: InlineArray[F32Ptr, 2] = [scale_a, scale_b]
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def report(label: String, asm_str: String):
    print("---", label, "---")
    print("  fmadd:        ", asm_str.count("vfmadd"))
    print("  movups:       ", asm_str.count("vmovups"))
    print("  movaps:       ", asm_str.count("vmovaps"))
    print("  load (mov+ld):", asm_str.count("vmovups") + asm_str.count("vmovaps"))
    print("  mulps:        ", asm_str.count("vmulps"))
    print("  cvtne(bf16):  ", asm_str.count("vcvtne"))
    print("  call:         ", asm_str.count("call"))
    print("  bytes:        ", asm_str.byte_length())


def main():
    var existing = String(compile_info[
        existing_dual_gamma_shim[3072, 128], emission_kind="asm"]().asm)
    var unified = String(compile_info[
        n_dual_i8_shim[3072, 128], emission_kind="asm"]().asm)

    print("== Direct comparison: existing dual-gamma vs unified N=2 dual-i8 ==")
    print()
    report("EXISTING rmsnorm_dual_gamma_fwht_quant_row", existing)
    print()
    report("UNIFIED  n_rmsnorm_row[N=2, i8+i8]", unified)
