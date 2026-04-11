"""Per-head norm + RoPE + FWHT + KV cache write for Gemma 4.

Pipeline:
  K: bf16 -> f32 -> /rms(per-head) -> RoPE -> FWHT -> dynamic quantize -> VNNI scatter
  V: bf16 -> f32 -> /rms(no gamma) -> FWHT -> dynamic absmax quantize -> VNNI scatter

Both K and V use per-position dynamic absmax quantization. The V scale is
folded into the W (attention-weight) quantization in v_agg_group, so the
inner loop pays no extra cost for using full per-token int8 dynamic range.

Uses Gemma4KVCache which stores both K and V in VNNI tile format:
  K: [head][tile_idx][k_slice × TILE_BYTES] — u8 (XOR 0x80) for tdpbsud
  V: [head][hd_tile][k_group × TILE_BYTES] — i8, transposed for V-agg tileload

All buffers are passed in by the caller.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from experimental.amx import VNNI_BLK
from experimental3.kv_cache import Gemma4KVCache
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8
from simd_math import sqrt


# ============================================================================
# Per-head RMS-divide helper
# ============================================================================


@always_inline
def rms_divide[head_dim: Int](work: UnsafePointer[Float32, MutAnyOrigin], eps: Float32):
    """In-place: work[0:head_dim] /= rms(work). Gamma, if any, is applied by the caller."""
    comptime width = simd_width_of[DType.float32]()
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= head_dim:
        var x = (work + k).load[width=width]()
        vsum = x.fma(x, vsum)
        k += width
    var inv_rms = 1.0 / sqrt[DType.float32, 1](vsum.reduce_add() / Float32(head_dim) + eps)
    k = 0
    while k + width <= head_dim:
        (work + k).store((work + k).load[width=width]() * inv_rms)
        k += width


# ============================================================================
# K head: /rms -> full RoPE -> FWHT -> dynamic quantize -> VNNI cache
# ============================================================================


def write_k_head_normed[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k_norm: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
    eps: Float32,
):
    """K head for sliding attention: bf16 -> f32 -> /rms -> *k_norm -> full RoPE -> FWHT -> cache."""
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2

    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    rms_divide[head_dim](work, eps)

    k = 0
    while k + width <= head_dim:
        (work + k).store((work + k).load[width=width]() * (k_norm + k).load[width=width]().cast[DType.float32]())
        k += width

    k = 0
    while k + width <= half:
        var x_lo = (work + k).load[width=width]()
        var x_hi = (work + half + k).load[width=width]()
        var cv = (cos + k).load[width=width]()
        var sv = (sin + k).load[width=width]()
        (work + k).store(x_lo * cv - x_hi * sv)
        (work + half + k).store(x_hi * cv + x_lo * sv)
        k += width

    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_k(pos, head, qi_buf.bitcast[Int8]())
    cache.write_k_scale(pos, head, absmax)


# ============================================================================
# K head: /rms -> partial RoPE -> FWHT -> dynamic quantize -> VNNI cache
# ============================================================================


def write_k_head_normed_partial[head_dim: Int, rope_dims: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k_norm: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
    eps: Float32,
):
    """K head for full attention: /rms -> *k_norm -> partial RoPE (128 of 512 dims rotated)."""
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime rope_half = rope_dims // 2

    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    rms_divide[head_dim](work, eps)

    k = 0
    while k + width <= head_dim:
        (work + k).store((work + k).load[width=width]() * (k_norm + k).load[width=width]().cast[DType.float32]())
        k += width

    k = 0
    while k + width <= rope_half:
        var x_lo = (work + k).load[width=width]()
        var x_hi = (work + half + k).load[width=width]()
        var cv = (cos + k).load[width=width]()
        var sv = (sin + k).load[width=width]()
        (work + k).store(x_lo * cv - x_hi * sv)
        (work + half + k).store(x_hi * cv + x_lo * sv)
        k += width

    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_k(pos, head, qi_buf.bitcast[Int8]())
    cache.write_k_scale(pos, head, absmax)


# ============================================================================
# V head: /rms (no gamma) -> FWHT -> dynamic absmax quantize -> VNNI cache
# ============================================================================


def write_v_head_normed[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
    eps: Float32,
):
    """V head: bf16 -> f32 -> /rms -> FWHT -> per-token absmax quantize -> VNNI cache.

    The per-token scale is stored in the cache and folded into the attention
    weight quantization during v_agg_group. No extra dequant cost in the score
    or v-agg inner loop.
    """
    comptime width = simd_width_of[DType.float32]()

    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    rms_divide[head_dim](work, eps)
    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, absmax)
