"""Comprehensive codegen comparison: existing vs unified for all 8 norm shapes.

Shapes covered:
  S1: i8 + gamma + FWHT + per-row     (attn_quantize, FFN quantize)
  S2: i8 + gamma + FWHT + per-block   (lm_head)
  S3: i8 + no gamma + FWHT + per-row  (smollm2-style)
  S4: dual i8 + i8 + gammas           (Gemma4 dense+expert)
  S5: dual i8 + bf16                  (minimax dual_norm)
  S6: bf16 only, no gamma             (V-norm, router-norm)
  S7: bf16 only, gamma                (standard rmsnorm)
  S8: bf16 only, gamma + residual     (post_attn_norm)

For each we report (fmadd, movups, mulps, cvtne, calls, bytes).
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
# Faithful reproductions of the 4 existing row kernels.
# ============================================================================

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
def existing_emit_quant[cols: Int, block: Int, per_block: Bool](
    work: F32Ptr, qi: I8Ptr, scales: F32Ptr,
):
    comptime if per_block:
        comptime num_blocks = cols // block
        for b in range(num_blocks):
            scales[b] = absmax_quantize_i8[block](
                work + b * block, qi + b * block)
    else:
        scales[0] = absmax_quantize_i8[cols](work, qi)


@always_inline
def existing_fwht_rotate[cols: Int, block: Int](work: F32Ptr):
    for b in range(cols // block):
        fwht_block[block](work + b * block)


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
    existing_fwht_rotate[cols, block](work)
    existing_emit_quant[cols, block, per_block](work, qi, scales)


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
    existing_fwht_rotate[cols, block](work_a)
    existing_fwht_rotate[cols, block](work_b)
    scale_a[0] = absmax_quantize_i8[cols](work_a, qi_a)
    scale_b[0] = absmax_quantize_i8[cols](work_b, qi_b)


@always_inline
def existing_dual_output_row[cols: Int, block: Int](
    src: BF16Ptr, split_gamma: BF16Ptr, full_gamma: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed_bf16: BF16Ptr,
    eps: Float32,
):
    """Minimax dual_output: i8(split_gamma+FWHT) + bf16(full_gamma)."""
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

    existing_fwht_rotate[cols, block](work)
    scale[0] = absmax_quantize_i8[cols](work, qi)


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


# ============================================================================
# Unified n_rmsnorm with smart pass-1 selection.
# - If any spec is i8, use load_to_work_and_reduce (writes f32 work).
# - If all specs are bf16, use rms_reduce_bf16 (no work store) and have
#   each emit reload from src.
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


@always_inline
def is_i8[spec: EmitSpec]() -> Bool:
    return spec.mode == EMIT_I8_PER_ROW or spec.mode == EMIT_I8_PER_BLOCK


comptime AnyI8[N: Int, *specs: EmitSpec]: Bool = _any_i8_impl[N, *specs]()


@always_inline
def _any_i8_impl[N: Int, *specs: EmitSpec]() -> Bool:
    var found = False
    comptime for i in range(N):
        comptime if is_i8[specs[i]]():
            found = True
    return found


comptime LastI8Index[N: Int, *specs: EmitSpec]: Int = _last_i8_impl[N, *specs]()


@always_inline
def _last_i8_impl[N: Int, *specs: EmitSpec]() -> Int:
    var idx = N
    comptime for i in range(N):
        comptime if is_i8[specs[i]]():
            idx = i
    return idx


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
def emit_one[
    cols: Int, block: Int, spec: EmitSpec,
    in_place: Bool, src_is_bf16: Bool,
](
    src_or_work: F32Ptr,    # f32 work; ignored if src_is_bf16
    src_bf16: BF16Ptr,      # used if src_is_bf16 (for bf16-only fast path)
    inv_rms: Float32, gamma: BF16Ptr,
    dst_addr: Int, scale: F32Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    var vinv = SIMD[DType.float32, width](inv_rms)

    comptime if spec.mode == EMIT_BF16 or spec.mode == EMIT_BF16_RESIDUAL:
        var dst_bf16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        var k = 0
        while k < cols:
            var x: SIMD[DType.float32, width]
            comptime if src_is_bf16:
                x = (src_bf16 + k).load[width=width]().cast[DType.float32]()
            else:
                x = (src_or_work + k).load[width=width]()
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

    # i8 mode (only reachable with src_is_bf16=False)
    comptime if in_place:
        var k = 0
        while k < cols:
            var x = (src_or_work + k).load[width=width]()
            var normed = x * vinv
            comptime if spec.has_gamma:
                var g = (gamma + k).load[width=width]().cast[DType.float32]()
                normed = normed * g
            (src_or_work + k).store(normed)
            k += width
        comptime if spec.fwht:
            for b in range(cols // block):
                fwht_block[block](src_or_work + b * block)
        var dst_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=dst_addr)
        comptime if spec.mode == EMIT_I8_PER_BLOCK:
            for b in range(cols // block):
                scale[b] = absmax_quantize_i8[block](
                    src_or_work + b * block, dst_i8 + b * block)
        else:
            scale[0] = absmax_quantize_i8[cols](src_or_work, dst_i8)
    else:
        var local_arr = InlineArray[Float32, cols](uninitialized=True)
        var local = UnsafePointer(to=local_arr).bitcast[Float32]()
        var k = 0
        while k < cols:
            var x = (src_or_work + k).load[width=width]()
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
def n_rmsnorm_row[
    cols: Int, block: Int, N: Int, *specs: EmitSpec,
](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gamma_ptrs: InlineArray[BF16Ptr, N],
    dst_addrs: InlineArray[Int, N],
    scale_ptrs: InlineArray[F32Ptr, N],
):
    comptime assert len(specs) == N
    comptime any_i8 = AnyI8[N, *specs]
    comptime last_i8 = LastI8Index[N, *specs]

    var inv: Float32
    comptime if any_i8:
        var sum_sq = load_to_work_and_reduce[cols](src, work)
        inv = inv_rms_from_sum_sq(sum_sq, cols, eps)
    else:
        var sum_sq = rms_reduce_bf16[cols](src)
        inv = inv_rms_from_sum_sq(sum_sq, cols, eps)

    comptime for i in range(N):
        comptime in_place = (i == last_i8)
        comptime src_is_bf16 = not any_i8
        emit_one[cols, block, specs[i], in_place, src_is_bf16](
            work, src, inv, gamma_ptrs[i], dst_addrs[i], scale_ptrs[i])


# ============================================================================
# Existing-kernel shims, one per shape.
# ============================================================================

def E1[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    existing_fwht_quant_row[cols, block, True, False](
        src, gamma, qi, work, scale, Float32(1e-6))


def E2[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    existing_fwht_quant_row[cols, block, True, True](
        src, gamma, qi, work, scale, Float32(1e-6))


def E3[cols: Int, block: Int](
    src: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    existing_fwht_quant_row[cols, block, False, False](
        src, BF16Ptr(unsafe_from_address=0), qi, work, scale, Float32(1e-6))


def E4[cols: Int, block: Int](
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr, qa: I8Ptr, qb: I8Ptr,
    wa: F32Ptr, wb: F32Ptr, sa: F32Ptr, sb: F32Ptr,
):
    existing_dual_gamma_row[cols, block](
        src, ga, gb, qa, qb, wa, wb, sa, sb, Float32(1e-6))


def E5[cols: Int, block: Int](
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    existing_dual_output_row[cols, block](
        src, sg, fg, qi, work, scale, normed, Float32(1e-6))


def E6[cols: Int](src: BF16Ptr, dst: BF16Ptr):
    existing_bf16_row[cols, False, False](
        src, BF16Ptr(unsafe_from_address=0), dst, Float32(1e-6))


def E7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    existing_bf16_row[cols, True, False](src, gamma, dst, Float32(1e-6))


def E8[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    existing_bf16_row[cols, True, True](src, gamma, dst, Float32(1e-6))


# ============================================================================
# Unified shims, one per shape.
# ============================================================================

def U1[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [gamma]
    var dst_addrs: InlineArray[Int, 1] = [Int(qi)]
    var scale_ptrs: InlineArray[F32Ptr, 1] = [scale]
    n_rmsnorm_row[cols, block, 1,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def U2[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [gamma]
    var dst_addrs: InlineArray[Int, 1] = [Int(qi)]
    var scale_ptrs: InlineArray[F32Ptr, 1] = [scale]
    n_rmsnorm_row[cols, block, 1,
        EmitSpec(mode=EMIT_I8_PER_BLOCK, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def U3[cols: Int, block: Int](
    src: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [BF16Ptr(unsafe_from_address=0)]
    var dst_addrs: InlineArray[Int, 1] = [Int(qi)]
    var scale_ptrs: InlineArray[F32Ptr, 1] = [scale]
    n_rmsnorm_row[cols, block, 1,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=False, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def U4[cols: Int, block: Int](
    src: BF16Ptr, ga: BF16Ptr, gb: BF16Ptr, qa: I8Ptr, qb: I8Ptr,
    work: F32Ptr, sa: F32Ptr, sb: F32Ptr,
):
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [ga, gb]
    var dst_addrs: InlineArray[Int, 2] = [Int(qa), Int(qb)]
    var scale_ptrs: InlineArray[F32Ptr, 2] = [sa, sb]
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)


def U5[cols: Int, block: Int](
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    """Order: bf16 first, i8 last (so i8 takes in-place path)."""
    var gamma_ptrs: InlineArray[BF16Ptr, 2] = [fg, sg]
    var dst_addrs: InlineArray[Int, 2] = [Int(normed), Int(qi)]
    var unused = InlineArray[Float32, 1](fill=Float32(0))
    var scale_ptrs: InlineArray[F32Ptr, 2] = [
        UnsafePointer(to=unused).bitcast[Float32](), scale,
    ]
    n_rmsnorm_row[cols, block, 2,
        EmitSpec(mode=EMIT_BF16, has_gamma=True, fwht=False),
        EmitSpec(mode=EMIT_I8_PER_ROW, has_gamma=True, fwht=True),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused


def U6[cols: Int](src: BF16Ptr, dst: BF16Ptr, work: F32Ptr):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [BF16Ptr(unsafe_from_address=0)]
    var dst_addrs: InlineArray[Int, 1] = [Int(dst)]
    var unused = InlineArray[Float32, 1](fill=Float32(0))
    var scale_ptrs: InlineArray[F32Ptr, 1] = [
        UnsafePointer(to=unused).bitcast[Float32](),
    ]
    n_rmsnorm_row[cols, 1, 1,
        EmitSpec(mode=EMIT_BF16, has_gamma=False, fwht=False),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused


def U7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, work: F32Ptr):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [gamma]
    var dst_addrs: InlineArray[Int, 1] = [Int(dst)]
    var unused = InlineArray[Float32, 1](fill=Float32(0))
    var scale_ptrs: InlineArray[F32Ptr, 1] = [
        UnsafePointer(to=unused).bitcast[Float32](),
    ]
    n_rmsnorm_row[cols, 1, 1,
        EmitSpec(mode=EMIT_BF16, has_gamma=True, fwht=False),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused


def U8[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr, work: F32Ptr):
    var gamma_ptrs: InlineArray[BF16Ptr, 1] = [gamma]
    var dst_addrs: InlineArray[Int, 1] = [Int(dst)]
    var unused = InlineArray[Float32, 1](fill=Float32(0))
    var scale_ptrs: InlineArray[F32Ptr, 1] = [
        UnsafePointer(to=unused).bitcast[Float32](),
    ]
    n_rmsnorm_row[cols, 1, 1,
        EmitSpec(mode=EMIT_BF16_RESIDUAL, has_gamma=True, fwht=False),
    ](src, work, Float32(1e-6), gamma_ptrs, dst_addrs, scale_ptrs)
    _ = unused


# ============================================================================
# Reporting
# ============================================================================

@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable):
    var fmadd: Int
    var mulps: Int
    var addps: Int
    var bf16_load: Int   # vpmovzxwd (bf16 unpack)
    var f32_load: Int    # vmovups + vmovaps (aligned/unaligned f32 load)
    var stores: Int      # vmovdqu + movdqu + vmovups when used as store
    var bf16_pack: Int   # vcvtne2ps2bf16 + vpslld (bf16 packing)
    var calls: Int
    var bytes: Int


def stats_of(asm_str: String) -> Stats:
    return Stats(
        fmadd=asm_str.count("vfmadd"),
        mulps=asm_str.count("vmulps"),
        addps=asm_str.count("vaddps"),
        bf16_load=asm_str.count("vpmovzxwd"),
        f32_load=asm_str.count("vmovups") + asm_str.count("vmovaps"),
        stores=asm_str.count("vmovdqu") + asm_str.count("vmovdqa"),
        bf16_pack=asm_str.count("vcvtne") + asm_str.count("vpslld"),
        calls=asm_str.count("call"),
        bytes=asm_str.byte_length(),
    )


def row(label: String, e: Stats, u: Stats):
    print(label)
    print("  existing:  fmadd=", e.fmadd,
          " mul=", e.mulps, " add=", e.addps,
          " bf16ld=", e.bf16_load, " f32ld/st=", e.f32_load,
          " store=", e.stores, " bf16op=", e.bf16_pack,
          " call=", e.calls, " bytes=", e.bytes)
    print("  unified:   fmadd=", u.fmadd,
          " mul=", u.mulps, " add=", u.addps,
          " bf16ld=", u.bf16_load, " f32ld/st=", u.f32_load,
          " store=", u.stores, " bf16op=", u.bf16_pack,
          " call=", u.calls, " bytes=", u.bytes)
    var dbytes = u.bytes - e.bytes
    var dfmadd = u.fmadd - e.fmadd
    var dmul = u.mulps - e.mulps
    var dbf = u.bf16_load - e.bf16_load
    var df32 = u.f32_load - e.f32_load
    var dst_ = u.stores - e.stores
    print("  delta:     bytes=", dbytes,
          " fmadd=", dfmadd, " mul=", dmul,
          " bf16ld=", dbf, " f32ld/st=", df32, " store=", dst_)
    print()


def main():
    comptime cols = 3072
    comptime block = 128

    var e1 = stats_of(String(compile_info[E1[cols, block], emission_kind="asm"]().asm))
    var u1 = stats_of(String(compile_info[U1[cols, block], emission_kind="asm"]().asm))
    var e2 = stats_of(String(compile_info[E2[cols, block], emission_kind="asm"]().asm))
    var u2 = stats_of(String(compile_info[U2[cols, block], emission_kind="asm"]().asm))
    var e3 = stats_of(String(compile_info[E3[cols, block], emission_kind="asm"]().asm))
    var u3 = stats_of(String(compile_info[U3[cols, block], emission_kind="asm"]().asm))
    var e4 = stats_of(String(compile_info[E4[cols, block], emission_kind="asm"]().asm))
    var u4 = stats_of(String(compile_info[U4[cols, block], emission_kind="asm"]().asm))
    var e5 = stats_of(String(compile_info[E5[cols, block], emission_kind="asm"]().asm))
    var u5 = stats_of(String(compile_info[U5[cols, block], emission_kind="asm"]().asm))
    var e6 = stats_of(String(compile_info[E6[cols], emission_kind="asm"]().asm))
    var u6 = stats_of(String(compile_info[U6[cols], emission_kind="asm"]().asm))
    var e7 = stats_of(String(compile_info[E7[cols], emission_kind="asm"]().asm))
    var u7 = stats_of(String(compile_info[U7[cols], emission_kind="asm"]().asm))
    var e8 = stats_of(String(compile_info[E8[cols], emission_kind="asm"]().asm))
    var u8 = stats_of(String(compile_info[U8[cols], emission_kind="asm"]().asm))

    print("=" * 72)
    print("Norm-row codegen comparison: cols=", cols, ", block=", block)
    print("=" * 72)
    print()
    row("S1: i8 + gamma + FWHT + per-row    (attn_quantize)",       e1, u1)
    row("S2: i8 + gamma + FWHT + per-block  (lm_head)",              e2, u2)
    row("S3: i8 + no_gamma + FWHT + per-row (smollm2)",              e3, u3)
    row("S4: dual i8 + i8 + gammas          (gemma4 dense+expert)",  e4, u4)
    row("S5: dual i8 + bf16                 (minimax dual_norm)",    e5, u5)
    row("S6: bf16 only, no gamma            (V-norm, router-norm)",  e6, u6)
    row("S7: bf16 only, gamma               (standard)",             e7, u7)
    row("S8: bf16 only, gamma + residual    (post_attn_norm)",       e8, u8)
