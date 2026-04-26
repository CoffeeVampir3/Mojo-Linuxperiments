"""Test the 'last i8 output consumes work in-place' optimization."""
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


@fieldwise_init
struct EmitSpec(Copyable, ImplicitlyCopyable):
    var mode: Int
    var has_gamma: Bool
    var fwht: Bool


comptime EMIT_BF16: Int = 0
comptime EMIT_BF16_RESIDUAL: Int = 1
comptime EMIT_I8_PER_ROW: Int = 2
comptime EMIT_I8_PER_BLOCK: Int = 3


@always_inline
def is_i8_mode[spec: EmitSpec]() -> Bool:
    return spec.mode == EMIT_I8_PER_ROW or spec.mode == EMIT_I8_PER_BLOCK


# Comptime helper: for a given spec list, find the index of the last i8.
# Returns N if there are no i8 outputs.
comptime LastI8Index[N: Int, *specs: EmitSpec]: Int = _last_i8_impl[N, *specs]()


@always_inline
def _last_i8_impl[N: Int, *specs: EmitSpec]() -> Int:
    var idx = N
    comptime for i in range(N):
        comptime if is_i8_mode[specs[i]]():
            idx = i
    return idx


@always_inline
def emit_one[
    cols: Int, block: Int, spec: EmitSpec, in_place: Bool,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst_addr: Int, scale: F32Ptr,
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
                var prev = (dst_bf16 + k).load[width=width]().cast[
                    DType.float32]()
                (dst_bf16 + k).store((prev + normed).cast[DType.bfloat16]())
            else:
                (dst_bf16 + k).store(normed.cast[DType.bfloat16]())
            k += width
        return

    # i8 mode. If in_place, mutate `work` directly; otherwise local scratch.
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
        var dst_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        comptime if spec.mode == EMIT_I8_PER_BLOCK:
            for b in range(cols // block):
                scale[b] = absmax_quantize_i8[block](
                    work + b * block, dst_i8 + b * block)
        else:
            scale[0] = absmax_quantize_i8[cols](work, dst_i8)
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
        var dst_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        comptime if spec.mode == EMIT_I8_PER_BLOCK:
            for b in range(cols // block):
                scale[b] = absmax_quantize_i8[block](
                    local + b * block, dst_i8 + b * block)
        else:
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
    comptime last_i8 = LastI8Index[N, *specs]
    var sum_sq = load_to_work_and_reduce[cols](src, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    comptime for i in range(N):
        comptime in_place = (i == last_i8)
        emit_one[cols, block, specs[i], in_place](
            work, inv, gamma_ptrs[i], dst_addrs[i], scale_ptrs[i])


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


def n_dual_shim[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [full_gamma, split_gamma]
    var dst_addrs: InlineArray[Int, 2] = [Int(normed), Int(qi)]
    var unused_scale_arr = InlineArray[Float32, 1](fill=Float32(0))
    var unused_scale = UnsafePointer(to=unused_scale_arr).bitcast[Float32]()
    var scale_ptrs: InlineArray[F32Ptr, 2] = [unused_scale, scale]
    # Order: bf16 FIRST, i8 LAST so the last i8 takes the in-place path.
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_BF16, has_gamma=True, fwht=False),
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused_scale_arr


def report(label: String, asm_str: String):
    print("---", label, "---")
    print("  fmadd:        ", asm_str.count("vfmadd"))
    print("  movups:       ", asm_str.count("vmovups"))
    print("  mulps:        ", asm_str.count("vmulps"))
    print("  call:         ", asm_str.count("call"))
    print("  bytes:        ", asm_str.byte_length())


def correctness_check():
    """Run dual i8+bf16 and verify against a manual reference computation."""
    comptime cols = 32
    comptime block = 16

    # Set up data.
    var src_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](0))
    var src = UnsafePointer(to=src_arr).bitcast[Scalar[DType.bfloat16]]()
    for i in range(cols):
        src[i] = Scalar[DType.bfloat16](Float32(i + 1) * 0.1)

    var split_gamma_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](0.5))
    var full_gamma_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](1.0))
    var split_gamma = UnsafePointer(to=split_gamma_arr).bitcast[
        Scalar[DType.bfloat16]]()
    var full_gamma = UnsafePointer(to=full_gamma_arr).bitcast[
        Scalar[DType.bfloat16]]()

    var work_arr = InlineArray[Float32, cols](uninitialized=True)
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()

    var qi_arr = InlineArray[Scalar[DType.int8], cols](
        fill=Scalar[DType.int8](0))
    var qi = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()
    var qi_scale_arr = InlineArray[Float32, 1](fill=Float32(0))
    var qi_scale = UnsafePointer(to=qi_scale_arr).bitcast[Float32]()

    var bf16_out_arr = InlineArray[Scalar[DType.bfloat16], cols](
        fill=Scalar[DType.bfloat16](0))
    var bf16_out = UnsafePointer(to=bf16_out_arr).bitcast[
        Scalar[DType.bfloat16]]()

    n_dual_shim[cols, block](
        src, split_gamma, full_gamma, qi, work, qi_scale, bf16_out)

    print("dual i8+bf16:")
    print("  i8 scale =", qi_scale[0])
    print("  bf16 out[0..3] =",
        Float32(bf16_out[0]), Float32(bf16_out[1]),
        Float32(bf16_out[2]), Float32(bf16_out[3]))


def n_dual_i8_shim[cols: Int, block: Int](
    src: BF16Ptr, gamma_a: BF16Ptr, gamma_b: BF16Ptr,
    qi_a: I8Ptr, qi_b: I8Ptr, work: F32Ptr,
    scale_a: F32Ptr, scale_b: F32Ptr,
):
    """Gemma4-shaped dual-gamma dense+expert i8 outputs."""
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [gamma_a, gamma_b]
    var dst_addrs: InlineArray[Int, 2] = [Int(qi_a), Int(qi_b)]
    var scale_ptrs: InlineArray[F32Ptr, 2] = [scale_a, scale_b]
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def main():
    var single = String(compile_info[
        n_single_shim[3072, 128], emission_kind="asm"]().asm)
    var dual_mixed = String(compile_info[
        n_dual_shim[3072, 128], emission_kind="asm"]().asm)
    var dual_i8 = String(compile_info[
        n_dual_i8_shim[3072, 128], emission_kind="asm"]().asm)

    print("== N=1 single i8 (in_place=True) ==")
    report("n_rmsnorm_row N=1", single)
    print()
    print("== N=2 bf16 first + i8 last (minimax dual_norm shape) ==")
    report("n_rmsnorm_row N=2 mixed", dual_mixed)
    print()
    print("== N=2 i8 + i8 (Gemma4 dense+expert shape) ==")
    report("n_rmsnorm_row N=2 dual_i8", dual_i8)
    print()
    correctness_check()
