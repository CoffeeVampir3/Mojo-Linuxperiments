from std.memory import UnsafePointer
from std.collections import InlineArray

from experimental3.kernels.fwht import fwht_apply, fwht_width
from experimental3.kv_cache import Gemma4KVCache
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from experimental3.kernels.rope_and_kv_cache_write import (
    load_head_bf16,
    regs_scale_with_gamma_bf16,
    regs_rope_partial,
    regs_absmax_quantize_i8,
)


# ============================================================================
# Q head: inv_rms * gamma → RoPE(pair_stride) → FWHT → quantize → i8
# ============================================================================


@always_inline
def prep_q_head[head_dim: Int, rope_dim: Int, pair_stride: Int](
    q_bf16: BF16Ptr, gamma: BF16Ptr,
    cos: F32Ptr, sin: F32Ptr,
    inv_rms: Float32,
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
) -> Tuple[Float32, Float32]:
    """Per-head Q prep with pre-computed full-vector inv_rms.

    Returns (qi_bias, absmax).
    """
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = (rope_dim // 2) // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(q_bf16, r)
    regs_scale_with_gamma_bf16(r, inv_rms, gamma)
    regs_rope_partial[rope_regs=rope_regs, pair_reg_stride=pair_reg_stride](r, cos, sin)
    fwht_apply[DType.float32, head_dim](r)
    var quant = regs_absmax_quantize_i8(r, qi_out.bitcast[Scalar[DType.int8]]())
    return (Float32(quant[1]) * Float32(128.0), quant[0])


# ============================================================================
# K head: inv_rms * gamma → RoPE(pair_stride) → FWHT → quantize → VNNI cache
# ============================================================================


def write_k_head[head_dim: Int, rope_dim: Int, pair_stride: Int](
    k_bf16: BF16Ptr, gamma: BF16Ptr,
    cos: F32Ptr, sin: F32Ptr,
    inv_rms: Float32,
    qi_buf: I8Ptr,
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int, head: Int,
):
    """Per-head K prep with pre-computed full-vector inv_rms, writes to VNNI cache."""
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = (rope_dim // 2) // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(k_bf16, r)
    regs_scale_with_gamma_bf16(r, inv_rms, gamma)
    regs_rope_partial[rope_regs=rope_regs, pair_reg_stride=pair_reg_stride](r, cos, sin)
    fwht_apply[DType.float32, head_dim](r)
    var quant = regs_absmax_quantize_i8(r, qi_buf)
    cache.write_k(pos, head, qi_buf.bitcast[Int8]())
    cache.write_k_scale(pos, head, quant[0])


# ============================================================================
# V head: quantize → VNNI cache (no norm, no RoPE, no FWHT — two-sided)
# ============================================================================


def write_v_direct[head_dim: Int](
    v_bf16: BF16Ptr,
    qi_buf: I8Ptr,
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int, head: Int,
):
    """V cache write for two-sided weights — FWHT already baked into weights."""
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(v_bf16, r)
    var quant = regs_absmax_quantize_i8(r, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, quant[0])
