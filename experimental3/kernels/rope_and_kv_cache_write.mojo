"""Per-head norm + RoPE + FWHT + quantize + KV cache write.

Register-resident primitives. The head stays in a SIMD register bank from load
through quantize — no memory round-trips between stages. Helpers below are
composable: each kernel picks the sequence that matches its shape.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import quantize_i8
from experimental3.amx import VNNI_BLK
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.fwht import fwht_apply, fwht_width
from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr, inv_rms_from_sum_sq,
)


# ============================================================================
# In-register head primitives
# ============================================================================


@always_inline
def load_head_bf16[width: Int, regs: Int, //](
    src: BF16Ptr,
    mut r: InlineArray[SIMD[DType.float32, width], regs],
):
    """Load bf16[regs*width] into the f32 register bank."""
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()


@always_inline
def regs_sum_sq[width: Int, regs: Int, //](
    r: InlineArray[SIMD[DType.float32, width], regs],
) -> Float32:
    """Sum(x^2) over the register bank — horizontal reduce to scalar."""
    var acc = r[0] * r[0]
    comptime for ri in range(1, regs):
        acc = r[ri].fma(r[ri], acc)
    return acc.reduce_add()


@always_inline
def regs_scale[width: Int, regs: Int, //](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    s: Float32,
):
    """In-place r[:] *= s."""
    var vs = SIMD[DType.float32, width](s)
    comptime for ri in range(regs):
        r[ri] = r[ri] * vs


@always_inline
def regs_scale_with_gamma_bf16[width: Int, regs: Int, //](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    s: Float32,
    gamma: BF16Ptr,
):
    """In-place r[:] *= s * bf16_gamma[:]."""
    var vs = SIMD[DType.float32, width](s)
    comptime for ri in range(regs):
        var g = (gamma + ri * width).load[width=width]().cast[DType.float32]()
        r[ri] = r[ri] * vs * g


@always_inline
def regs_rope_partial[
    width: Int, regs: Int, //,
    rope_regs: Int, pair_reg_stride: Int,
](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    cos: F32Ptr, sin: F32Ptr,
):
    """In-register partial RoPE: rotate reg[ri] with reg[ri+pair_reg_stride]
    for ri in [0, rope_regs). cos/sin indexed by lane within the rope half.
    """
    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[pair_reg_stride + ri]
        var cv = (cos + ri * width).load[width=width]()
        var sv = (sin + ri * width).load[width=width]()
        r[ri] = x_lo * cv - x_hi * sv
        r[pair_reg_stride + ri] = x_hi * cv + x_lo * sv


@always_inline
def regs_absmax_quantize_i8[width: Int, regs: Int, //](
    r: InlineArray[SIMD[DType.float32, width], regs],
    qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    """Dynamic absmax quantize register bank → i8 memory.

    Returns (absmax, qi_sum). qi_sum is the lane-sum of stored int8 values —
    callers needing asymmetric bias use it, others discard.
    """
    var vmax = SIMD[DType.float32, width](0)
    comptime for ri in range(regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)
    var vq_inv = SIMD[DType.float32, width](Float32(127.0) / absmax)
    var qsum = SIMD[DType.int32, width](0)
    comptime for ri in range(regs):
        var qi = quantize_i8[width](r[ri], vq_inv)
        (qi_out + ri * width).store(qi)
        qsum += qi.cast[DType.int32]()
    return (absmax, qsum.reduce_add())


# ============================================================================
# K head: /rms -> full RoPE -> FWHT -> dynamic quantize -> VNNI cache
# ============================================================================


def write_k_head_normed[head_dim: Int, rope_dims: Int = head_dim, pair_stride: Int = head_dim // 2](
    src_bf16: BF16Ptr,
    k_norm: BF16Ptr,
    cos: F32Ptr, sin: F32Ptr,
    qi_buf: I8Ptr,
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int, head: Int, eps: Float32,
):
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = (rope_dims // 2) // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(src_bf16, r)
    var inv_rms = inv_rms_from_sum_sq(regs_sum_sq(r), head_dim, eps)
    regs_scale_with_gamma_bf16(r, inv_rms, k_norm)
    regs_rope_partial[rope_regs=rope_regs, pair_reg_stride=pair_reg_stride](r, cos, sin)
    fwht_apply[DType.float32, head_dim](r)
    var quant = regs_absmax_quantize_i8(r, qi_buf)
    cache.write_k(pos, head, qi_buf.bitcast[Int8]())
    cache.write_k_scale(pos, head, quant[0])


# ============================================================================
# V head: /rms (no gamma) -> FWHT -> dynamic absmax quantize -> VNNI cache
# ============================================================================


def write_v_head_normed[head_dim: Int](
    src_bf16: BF16Ptr,
    qi_buf: I8Ptr,
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int, head: Int, eps: Float32,
):
    """V head: bf16 -> f32 -> /rms -> FWHT -> per-token absmax quantize -> VNNI cache."""
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(src_bf16, r)
    var inv_rms = inv_rms_from_sum_sq(regs_sum_sq(r), head_dim, eps)
    regs_scale(r, inv_rms)
    fwht_apply[DType.float32, head_dim](r)
    var quant = regs_absmax_quantize_i8(r, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, quant[0])
