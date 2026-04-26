"""Decomposed n-rmsnorm: bf16 fanout and i8 fanout as distinct kernels.

The previous unified design merged two genuinely different operations
behind a comptime-if ladder. Splitting them gives:
  - one clean kernel per output dtype
  - shared inv_rms via the primitive (no kernel-level branching)
  - mixed cases (e.g. minimax dual_norm) are explicit compositions
"""
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from compile import compile_info

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq, rms_reduce_bf16,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs


# ============================================================================
# Task 1: Reduce. Two flavors based on whether i8 emits will follow.
# ============================================================================

@always_inline
def reduce_bf16_only[cols: Int](src: BF16Ptr, eps: Float32) -> Float32:
    """Pass 1 for bf16-only fanout. No work buffer."""
    var sum_sq = rms_reduce_bf16[cols](src)
    return inv_rms_from_sum_sq(sum_sq, cols, eps)


@always_inline
def reduce_with_work[cols: Int](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
) -> Float32:
    """Pass 1 for i8 fanout. Stores raw x to work + accumulates sum(x^2)."""
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


# ============================================================================
# Task 2: Emit bf16. ONE small spec, ONE streaming kernel.
# ============================================================================

@fieldwise_init
struct Bf16Spec(Copyable, ImplicitlyCopyable):
    """How a single bf16 output should be written.

    has_gamma:    apply per-channel scale before write.
    has_residual: dst += normed (vs dst = normed). Used by post_attn_norm.
    """
    var has_gamma: Bool
    var has_residual: Bool


@always_inline
def emit_one_bf16[cols: Int, spec: Bf16Spec](
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
def emit_n_bf16[cols: Int, N: Int, *specs: Bf16Spec](
    src: BF16Ptr, inv_rms: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[BF16Ptr, N],
):
    comptime assert len(specs) == N
    comptime for i in range(N):
        emit_one_bf16[cols, specs[i]](src, inv_rms, gammas[i], dsts[i])


# ============================================================================
# Task 3: Emit i8 (with FWHT + absmax). ONE small spec, ONE streaming kernel.
# ============================================================================

@fieldwise_init
struct I8Spec(Copyable, ImplicitlyCopyable):
    """How a single i8 output should be written.

    has_gamma: apply per-channel scale.
    fwht:      apply block-diagonal FWHT before quantize.
    per_block: emit one absmax per FWHT block (vs one per row).
    """
    var has_gamma: Bool
    var fwht: Bool
    var per_block: Bool


@always_inline
def emit_one_i8[
    cols: Int, block: Int, spec: I8Spec, in_place: Bool,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst: I8Ptr, scale: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)

    # Pick the buffer we operate on. If we own work (in_place), mutate it;
    # else copy through a stack-local scratch so siblings can still read work.
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
def emit_n_i8[cols: Int, block: Int, N: Int, *specs: I8Spec](
    work: F32Ptr, inv_rms: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[I8Ptr, N],
    scales: InlineArray[F32Ptr, N],
):
    comptime assert len(specs) == N
    # Last i8 takes work in-place (work is no longer needed by anyone after).
    comptime for i in range(N):
        comptime in_place = (i == N - 1)
        emit_one_i8[cols, block, specs[i], in_place](
            work, inv_rms, gammas[i], dsts[i], scales[i])


# ============================================================================
# Two clean row kernels.
# ============================================================================

@always_inline
def rmsnorm_n_bf16_row[cols: Int, N: Int, *specs: Bf16Spec](
    src: BF16Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[BF16Ptr, N],
):
    """Replaces rmsnorm_bf16_row{,no_scale,per_head,post_attn}_kernel."""
    var inv = reduce_bf16_only[cols](src, eps)
    emit_n_bf16[cols, N, *specs](src, inv, gammas, dsts)


@always_inline
def rmsnorm_n_i8_row[cols: Int, block: Int, N: Int, *specs: I8Spec](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[I8Ptr, N],
    scales: InlineArray[F32Ptr, N],
):
    """Replaces rmsnorm_fwht_quant_row + rmsnorm_dual_gamma_fwht_quant_row."""
    var inv = reduce_with_work[cols](src, work, eps)
    emit_n_i8[cols, block, N, *specs](work, inv, gammas, dsts, scales)


# ============================================================================
# The ONE mixed case: minimax dual_norm. Explicit composition; trivial body.
# ============================================================================

@always_inline
def rmsnorm_dual_output_row[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, scale: F32Ptr,
    normed_bf16: BF16Ptr,
    work: F32Ptr, eps: Float32,
):
    """Replaces rmsnorm_dual_output_worker. Composes the two kernels."""
    var inv = reduce_with_work[cols](src, work, eps)
    emit_one_bf16[cols, Bf16Spec(has_gamma=True, has_residual=False)](
        src, inv, full_gamma, normed_bf16)
    emit_one_i8[cols, block,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
        in_place=True,
    ](work, inv, split_gamma, qi, scale)


# ============================================================================
# Shims for codegen comparison vs the existing kernels.
# ============================================================================

# Existing reproductions (copied from previous file).
@always_inline
def existing_load_and_reduce[cols: Int, has_gamma: Bool](
    src: BF16Ptr, gamma: BF16Ptr, work: F32Ptr,
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
        comptime if has_gamma:
            var g = (gamma + off).load[width=width]().cast[DType.float32]()
            (work + off).store(x * g)
        else:
            (work + off).store(x)
    var k = step
    while k + step <= cols:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var x = (src + off).load[width=width]().cast[DType.float32]()
            accs[i] = x.fma(x, accs[i])
            comptime if has_gamma:
                var g = (gamma + off).load[width=width]().cast[DType.float32]()
                (work + off).store(x * g)
            else:
                (work + off).store(x)
        k += step
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        accs[0] = x.fma(x, accs[0])
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            (work + k).store(x * g)
        else:
            (work + k).store(x)
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
def existing_fwht_quant_row[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool,
](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr,
    work: F32Ptr, scales: F32Ptr, eps: Float32,
):
    var sum_sq = existing_load_and_reduce[cols, has_gamma](src, gamma, work)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    existing_normalize_inplace[cols](work, inv)
    for b in range(cols // block):
        fwht_block[block](work + b * block)
    comptime if per_block:
        for b in range(cols // block):
            scales[b] = absmax_quantize_i8[block](
                work + b * block, qi + b * block)
    else:
        scales[0] = absmax_quantize_i8[cols](work, qi)


@always_inline
def existing_bf16_row[cols: Int, has_gamma: Bool, has_residual: Bool](
    src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, eps: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce_bf16[cols](src)
    var inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < cols:
        var v = (src + k).load[width=width]().cast[DType.float32]()
        comptime if has_gamma:
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            var normed = v * vinv * g
            comptime if has_residual:
                var x = (dst + k).load[width=width]().cast[DType.float32]()
                (dst + k).store((x + normed).cast[DType.bfloat16]())
            else:
                (dst + k).store(normed.cast[DType.bfloat16]())
        else:
            (dst + k).store((v * vinv).cast[DType.bfloat16]())
        k += width


# Existing shims
def E1[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    existing_fwht_quant_row[cols, block, True, False](
        src, gamma, qi, work, scale, Float32(1e-6))


def E5[cols: Int, block: Int](
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    # Skip — existing dual_output uses a different body shape; just compare
    # the new split-design body.
    rmsnorm_dual_output_row[cols, block](
        src, sg, fg, qi, scale, normed, work, Float32(1e-6))


def E7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    existing_bf16_row[cols, True, False](src, gamma, dst, Float32(1e-6))


# Split-design shims
def S1[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[I8Ptr, 1] = [qi]
    var scales: InlineArray[F32Ptr, 1] = [scale]
    rmsnorm_n_i8_row[cols, block, 1,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
    ](src, work, Float32(1e-6), gammas, dsts, scales)


def S5[cols: Int, block: Int](
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    rmsnorm_dual_output_row[cols, block](
        src, sg, fg, qi, scale, normed, work, Float32(1e-6))


def S7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[BF16Ptr, 1] = [dst]
    rmsnorm_n_bf16_row[cols, 1,
        Bf16Spec(has_gamma=True, has_residual=False),
    ](src, Float32(1e-6), gammas, dsts)


@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable):
    var fmadd: Int
    var mulps: Int
    var bf16ld: Int
    var f32ldst: Int
    var stores: Int
    var calls: Int
    var bytes: Int


def stats_of(asm: String) -> Stats:
    return Stats(
        fmadd=asm.count("vfmadd"),
        mulps=asm.count("vmulps"),
        bf16ld=asm.count("vpmovzxwd"),
        f32ldst=asm.count("vmovups") + asm.count("vmovaps"),
        stores=asm.count("vmovdqu") + asm.count("vmovdqa"),
        calls=asm.count("call"),
        bytes=asm.byte_length(),
    )


def row(label: String, e: Stats, s: Stats):
    print(label)
    print("  existing: fmadd=", e.fmadd, " mul=", e.mulps,
          " bf16ld=", e.bf16ld, " f32ld/st=", e.f32ldst,
          " store=", e.stores, " call=", e.calls, " bytes=", e.bytes)
    print("  split:    fmadd=", s.fmadd, " mul=", s.mulps,
          " bf16ld=", s.bf16ld, " f32ld/st=", s.f32ldst,
          " store=", s.stores, " call=", s.calls, " bytes=", s.bytes)
    print("  Δ bytes=", s.bytes - e.bytes,
          " mul=", s.mulps - e.mulps,
          " bf16ld=", s.bf16ld - e.bf16ld)
    print()


def main():
    comptime cols = 3072
    comptime block = 128

    var e1 = stats_of(String(compile_info[E1[cols, block], emission_kind="asm"]().asm))
    var s1 = stats_of(String(compile_info[S1[cols, block], emission_kind="asm"]().asm))
    var e5 = stats_of(String(compile_info[E5[cols, block], emission_kind="asm"]().asm))
    var s5 = stats_of(String(compile_info[S5[cols, block], emission_kind="asm"]().asm))
    var e7 = stats_of(String(compile_info[E7[cols], emission_kind="asm"]().asm))
    var s7 = stats_of(String(compile_info[S7[cols], emission_kind="asm"]().asm))

    print("Split design (clean: i8 kernel, bf16 kernel, mixed composition)")
    print("=" * 72)
    print()
    row("S1 attn_quantize (i8+γ+FWHT)", e1, s1)
    row("S5 minimax dual_norm (i8 + bf16, explicit composition)", e5, s5)
    row("S7 standard bf16 norm (γ)", e7, s7)
