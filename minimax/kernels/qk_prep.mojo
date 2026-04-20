from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from simd_math import quantize_i8
from experimental3.kernels.fwht import fwht_apply, fwht_block, fwht_width
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kv_cache import Gemma4KVCache
from kernels.kv_rotors import rotate_pair


# ============================================================================
# Q head: inv_rms * gamma → RoPE(pair_stride) → FWHT → quantize → i8
# ============================================================================


@always_inline
def prep_q_head[head_dim: Int, rope_dim: Int, pair_stride: Int](
    q_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    inv_rms: Float32,
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
) -> Tuple[Float32, Float32]:
    """Per-head Q prep with pre-computed full-vector inv_rms.

    Returns (qi_bias, absmax).
    """
    comptime fwht_w = fwht_width[DType.float32, head_dim]()
    comptime fwht_regs = head_dim // fwht_w
    comptime pair_reg_stride = pair_stride // fwht_w
    comptime rope_half = rope_dim // 2
    comptime rope_regs = rope_half // fwht_w

    debug_assert(head_dim % fwht_w == 0, "head_dim must be a multiple of FWHT SIMD width")
    debug_assert(pair_stride % fwht_w == 0, "pair_stride must be a multiple of FWHT SIMD width")
    debug_assert(rope_dim % 2 == 0, "rope_dim must be even")
    debug_assert(rope_half % fwht_w == 0, "rope_dim/2 must be a multiple of FWHT SIMD width")
    debug_assert(rope_dim <= head_dim, "rope_dim must not exceed head_dim")

    var r = InlineArray[SIMD[DType.float32, fwht_w], fwht_regs](
        fill=SIMD[DType.float32, fwht_w](0))

    # Load bf16 → f32, apply inv_rms * gamma
    var vinv = SIMD[DType.float32, fwht_w](inv_rms)
    comptime for ri in range(fwht_regs):
        var x = (q_bf16 + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()
        var g = (gamma + ri * fwht_w).load[width=fwht_w]().cast[DType.float32]()
        r[ri] = x * vinv * g

    # Partial RoPE: rotate rope_dim elements at pair_stride
    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[pair_reg_stride + ri]
        var cv = (cos + ri * fwht_w).load[width=fwht_w]()
        var sv = (sin + ri * fwht_w).load[width=fwht_w]()
        r[ri] = x_lo * cv - x_hi * sv
        r[pair_reg_stride + ri] = x_hi * cv + x_lo * sv

    fwht_apply[DType.float32, head_dim](r)

    # Dynamic absmax + quantize
    var vmax = SIMD[DType.float32, fwht_w](0)
    comptime for ri in range(fwht_regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    var vq_inv = SIMD[DType.float32, fwht_w](127.0 / absmax)
    var q_sum_acc = SIMD[DType.int32, fwht_w](0)
    comptime for ri in range(fwht_regs):
        var qi = quantize_i8[fwht_w](r[ri], vq_inv)
        (qi_out + ri * fwht_w).store(qi)
        q_sum_acc += qi.cast[DType.int32]()

    var qi_bias = Float32(q_sum_acc.reduce_add()) * 128.0
    return (qi_bias, absmax)


# ============================================================================
# K head: inv_rms * gamma → RoPE(pair_stride) → FWHT → quantize → VNNI cache
# ============================================================================


def write_k_head[head_dim: Int, rope_dim: Int, pair_stride: Int](
    k_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    inv_rms: Float32,
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
):
    """Per-head K prep with pre-computed full-vector inv_rms, writes to VNNI cache."""
    comptime width = simd_width_of[DType.float32]()
    comptime rope_half = rope_dim // 2

    # Load bf16 → f32, apply inv_rms * gamma
    var k = 0
    while k + width <= head_dim:
        var x = (k_bf16 + k).load[width=width]().cast[DType.float32]()
        var g = (gamma + k).load[width=width]().cast[DType.float32]()
        (work + k).store(x * inv_rms * g)
        k += width

    # Partial RoPE at pair_stride
    k = 0
    while k + width <= rope_half:
        rotate_pair[DType.float32, width, pair_stride](work, cos, sin, k)
        k += width

    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_k(pos, head, qi_buf.bitcast[Int8]())
    cache.write_k_scale(pos, head, absmax)


# ============================================================================
# V head: quantize → VNNI cache (no norm, no RoPE, no FWHT — two-sided)
# ============================================================================


def write_v_direct[head_dim: Int](
    v_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
):
    """V cache write for two-sided weights — FWHT already baked into weights.

    Just bf16 → f32 → absmax quantize → VNNI cache write.
    """
    comptime width = simd_width_of[DType.float32]()

    var k = 0
    while k + width <= head_dim:
        (work + k).store((v_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, absmax)
