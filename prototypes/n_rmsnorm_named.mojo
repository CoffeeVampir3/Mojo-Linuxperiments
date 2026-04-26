"""Renamed split design. Same code, clearer names."""
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
# Pass 1: RMS reduction. Two flavors.
# ============================================================================

@always_inline
def compute_inv_rms[cols: Int](src: BF16Ptr, eps: Float32) -> Float32:
    """Streaming RMS reduction over bf16 src. Returns inv_rms scalar.

    No staging. Use when downstream emits all read directly from src
    (i.e., bf16-only fanout)."""
    var sum_sq = rms_reduce_bf16[cols](src)
    return inv_rms_from_sum_sq(sum_sq, cols, eps)


@always_inline
def stage_x_compute_inv_rms[cols: Int](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
) -> Float32:
    """RMS reduction + stage raw x to f32 work for downstream FWHT.

    Use when at least one downstream emit is i8 (i8 emits need f32
    staging because FWHT operates in-place on f32)."""
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
# Bf16 emission. Spec carries only fields meaningful for bf16 output.
# ============================================================================

@fieldwise_init
struct Bf16Spec(Copyable, ImplicitlyCopyable):
    """Per-output configuration for a normalized bf16 write."""
    var has_gamma: Bool       # multiply by per-channel gamma after inv_rms
    var has_residual: Bool    # dst += normed (vs dst = normed)


@always_inline
def write_bf16_normed[cols: Int, spec: Bf16Spec](
    src: BF16Ptr, inv_rms: Float32, gamma: BF16Ptr, dst: BF16Ptr,
):
    """Write one normalized bf16 output. Reads src directly (no work buffer).

    dst[k] = src[k] * inv_rms [* gamma[k]] [+ dst[k]]
    """
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
    """Fanout of N normalized bf16 writes sharing one inv_rms."""
    comptime assert len(specs) == N
    comptime for i in range(N):
        write_bf16_normed[cols, specs[i]](src, inv_rms, gammas[i], dsts[i])


# ============================================================================
# I8 emission. Spec carries only fields meaningful for i8 output.
# ============================================================================

@fieldwise_init
struct I8Spec(Copyable, ImplicitlyCopyable):
    """Per-output configuration for a quantized i8 write."""
    var has_gamma: Bool       # multiply by per-channel gamma after inv_rms
    var fwht: Bool            # apply block-diagonal FWHT before absmax
    var per_block: Bool       # one absmax per FWHT block (vs per-row)


@always_inline
def write_i8_quantized[
    cols: Int, block: Int, spec: I8Spec, in_place: Bool,
](
    work: F32Ptr, inv_rms: Float32, gamma: BF16Ptr,
    dst: I8Ptr, scale: F32Ptr,
):
    """Write one normalized + (FWHT) + absmax-quantized i8 output.

    in_place=True: mutate work directly (only safe for the LAST i8 emit
                   that consumes work).
    in_place=False: stage through a stack-local f32 buffer so siblings
                    can still read raw x from work.
    """
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
    """Fanout of N normalized + quantized i8 writes sharing one inv_rms.

    The final i8 emit takes the in-place path on `work` since `work` won't
    be re-read after the last quantization."""
    comptime assert len(specs) == N
    comptime for i in range(N):
        comptime in_place = (i == N - 1)
        write_i8_quantized[cols, block, specs[i], in_place](
            work, inv_rms, gammas[i], dsts[i], scales[i])


# ============================================================================
# Row-level workers (compose pass 1 + pass 2).
# ============================================================================

@always_inline
def rmsnorm_n_bf16_row[cols: Int, N: Int, *specs: Bf16Spec](
    src: BF16Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[BF16Ptr, N],
):
    """N bf16 outputs sharing one RMS reduction. No work buffer required.

    Replaces: rmsnorm_bf16_row, rmsnorm_no_scale_kernel,
              post_attn_norm_kernel, dense_norm_kernel."""
    var inv = compute_inv_rms[cols](src, eps)
    write_n_bf16_normed[cols, N, *specs](src, inv, gammas, dsts)


@always_inline
def rmsnorm_n_i8_row[cols: Int, block: Int, N: Int, *specs: I8Spec](
    src: BF16Ptr, work: F32Ptr, eps: Float32,
    gammas: InlineArray[BF16Ptr, N],
    dsts: InlineArray[I8Ptr, N],
    scales: InlineArray[F32Ptr, N],
):
    """N i8 outputs sharing one RMS reduction + f32 staging.

    Replaces: rmsnorm_fwht_quant_row, rmsnorm_dual_gamma_fwht_quant_row."""
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
    """One i8 + one bf16 sharing one RMS reduction (minimax dual_norm).

    Replaces: rmsnorm_dual_output_row.
    Composition: stage x for the i8 path, then write the bf16 output from
    src (cheaper than re-reading work), then quantize the i8 output
    in-place on work."""
    var inv = stage_x_compute_inv_rms[cols](src, work, eps)
    write_bf16_normed[cols, Bf16Spec(has_gamma=True, has_residual=False)](
        src, inv, full_gamma, normed_bf16)
    write_i8_quantized[cols, block,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
        in_place=True,
    ](work, inv, split_gamma, qi, scale)


# ============================================================================
# Verify codegen unchanged.
# ============================================================================

def shim_S1[cols: Int, block: Int](
    src: BF16Ptr, gamma: BF16Ptr, qi: I8Ptr, work: F32Ptr, scale: F32Ptr,
):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[I8Ptr, 1] = [qi]
    var scales: InlineArray[F32Ptr, 1] = [scale]
    rmsnorm_n_i8_row[cols, block, 1,
        I8Spec(has_gamma=True, fwht=True, per_block=False),
    ](src, work, Float32(1e-6), gammas, dsts, scales)


def shim_S5[cols: Int, block: Int](
    src: BF16Ptr, sg: BF16Ptr, fg: BF16Ptr,
    qi: I8Ptr, work: F32Ptr, scale: F32Ptr, normed: BF16Ptr,
):
    rmsnorm_i8_and_bf16_row[cols, block](
        src, sg, fg, qi, scale, normed, work, Float32(1e-6))


def shim_S7[cols: Int](src: BF16Ptr, gamma: BF16Ptr, dst: BF16Ptr):
    var gammas: InlineArray[BF16Ptr, 1] = [gamma]
    var dsts: InlineArray[BF16Ptr, 1] = [dst]
    rmsnorm_n_bf16_row[cols, 1,
        Bf16Spec(has_gamma=True, has_residual=False),
    ](src, Float32(1e-6), gammas, dsts)


def main():
    var s1 = String(compile_info[shim_S1[3072, 128], emission_kind="asm"]().asm)
    var s5 = String(compile_info[shim_S5[3072, 128], emission_kind="asm"]().asm)
    var s7 = String(compile_info[shim_S7[3072], emission_kind="asm"]().asm)
    print("renamed S1 (i8 single):    ", s1.byte_length(), "bytes  fmadd=",
          s1.count("vfmadd"), " mul=", s1.count("vmulps"))
    print("renamed S5 (i8 + bf16):    ", s5.byte_length(), "bytes  fmadd=",
          s5.count("vfmadd"), " mul=", s5.count("vmulps"))
    print("renamed S7 (bf16 single):  ", s7.byte_length(), "bytes  fmadd=",
          s7.count("vfmadd"), " mul=", s7.count("vmulps"))
