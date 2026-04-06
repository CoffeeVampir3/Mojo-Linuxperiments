"""RoPE + KV cache write — bf16 GEMV output to quantized KV cache.

K: bf16 -> f32 -> RoPE -> FWHT -> dynamic absmax quantize -> i8 -> cache
V: bf16 -> f32 -> FWHT -> quantize(fixed S_V) -> i8 -> cache (row-major)

K uses per-head dynamic scales: after FWHT, compute absmax of the head_dim
values, quantize with that absmax, and store the scale in the cache alongside
the i8 data. This eliminates clipping from fixed per-layer scales.

V uses a corrected fixed scale (adequate because V norms are small and uniform
across heads — see butterquant/test_hadamard_roundtrip.mojo for the proof).

Input is a slice of the combined QKV bf16 output with stride QKV_N per row.
Each KV head is head_dim elements wide within that row.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from simd_math import sqrt, roundeven
from experimental2.kv_cache import KVCache
from experimental2.kernels.float_kernels.rmsnorm_fwht_quantize import fwht_block


# ============================================================================
# Per-head write: K (RoPE + FWHT + quantize + cache write)
# ============================================================================


def write_k_head[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
):
    """One K head: bf16 -> f32 -> RoPE -> FWHT -> dynamic quantize -> cache write.

    Computes per-head absmax after FWHT and stores the scale in the cache.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2

    # Load bf16 -> f32
    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    # RoPE: [lo, hi] -> [lo*cos - hi*sin, hi*cos + lo*sin]
    k = 0
    while k + width <= half:
        var x_lo = (work + k).load[width=width]()
        var x_hi = (work + half + k).load[width=width]()
        var cv = (cos + k).load[width=width]()
        var sv = (sin + k).load[width=width]()
        (work + k).store(x_lo * cv - x_hi * sv)
        (work + half + k).store(x_hi * cv + x_lo * sv)
        k += width

    # FWHT (block = head_dim)
    fwht_block[head_dim](work)

    # Dynamic absmax
    var vmax = SIMD[DType.float32, width](0)
    k = 0
    while k + width <= head_dim:
        vmax = max(vmax, (work + k).load[width=width]().__abs__())
        k += width
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    # Quantize with dynamic scale
    var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= head_dim:
        var v = (work + k).load[width=width]()
        (qi_buf + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
        k += width

    cache.write_k(pos, head, qi_buf)
    cache.write_k_scale(pos, head, absmax)


# ============================================================================
# Per-head write: V (FWHT + quantize + cache write, no RoPE)
# ============================================================================


def write_v_head[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    quant_inv: Float32,
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
):
    """One V head: bf16 -> f32 -> FWHT -> fixed-scale quantize -> cache write."""
    comptime width = simd_width_of[DType.float32]()

    # Load bf16 -> f32
    var k = 0
    while k + width <= head_dim:
        (work + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
        k += width

    # FWHT (block = head_dim)
    fwht_block[head_dim](work)

    # Fixed-scale quantize
    var vinv = SIMD[DType.float32, width](quant_inv)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= head_dim:
        var v = (work + k).load[width=width]()
        (qi_buf + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
        k += width

    cache.write_v(pos, head, qi_buf)


# ============================================================================
# Combined K+V write for all heads and positions
# ============================================================================


def rope_and_kv_cache_write[head_dim: Int, num_kv_heads: Int, max_seq: Int, num_q_heads: Int](
    k_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    v_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k_stride: Int,
    v_stride: Int,
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    cache: KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    pos: Int,
    seq_len: Int,
    s_v: Float32,
):
    """Write K and V from bf16 GEMV output into quantized KV cache.

    K uses dynamic per-head absmax scales (computed here, stored in cache).
    V uses fixed corrected scale s_v.
    """
    comptime half = head_dim // 2
    var v_quant_inv = Float32(127) / s_v

    var work_arr = InlineArray[Float32, head_dim](fill=Float32(0))
    var work = UnsafePointer(to=work_arr).bitcast[Float32]()
    var qi_arr = InlineArray[Scalar[DType.int8], head_dim](fill=Scalar[DType.int8](0))
    var qi_buf = UnsafePointer(to=qi_arr).bitcast[Scalar[DType.int8]]()

    for m in range(seq_len):
        var actual_pos = pos + m
        var cos_row = cos + actual_pos * half
        var sin_row = sin + actual_pos * half

        for g in range(num_kv_heads):
            write_k_head[head_dim](
                k_bf16 + m * k_stride + g * head_dim,
                cos_row, sin_row,
                work, qi_buf,
                cache, actual_pos, g,
            )
            write_v_head[head_dim](
                v_bf16 + m * v_stride + g * head_dim,
                v_quant_inv, work, qi_buf,
                cache, actual_pos, g,
            )
