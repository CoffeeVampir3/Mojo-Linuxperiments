"""Per-head norm + RoPE + FWHT + quantize + KV cache write."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from experimental3.amx import VNNI_BLK
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.fwht import fwht_block
from experimental3.common_math import rms_normalize_inplace
from experimental3.kernels.quantize import absmax_quantize_i8
from kernels.kv_rotors import rotate_pair


# ============================================================================
# K head: /rms -> full RoPE -> FWHT -> dynamic quantize -> VNNI cache
# ============================================================================


def write_k_head_normed[head_dim: Int, rope_dims: Int = head_dim, pair_stride: Int = head_dim // 2](
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
    comptime width = simd_width_of[DType.float32]()
    comptime rope_half = rope_dims // 2

    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    rms_normalize_inplace[head_dim](work, eps)

    k = 0
    while k + width <= head_dim:
        (work + k).store((work + k).load[width=width]() * (k_norm + k).load[width=width]().cast[DType.float32]())
        k += width

    k = 0
    while k + width <= rope_half:
        rotate_pair[DType.float32, width, pair_stride](work, cos, sin, k)
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

    rms_normalize_inplace[head_dim](work, eps)
    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, absmax)
