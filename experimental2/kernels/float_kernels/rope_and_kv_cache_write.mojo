"""RoPE + KV cache write — bf16 GEMV output to quantized KV cache.

K: bf16 -> f32 -> RoPE -> FWHT -> quantize(S_K) -> i8 -> cache (VNNI scatter)
V: bf16 -> f32 -> FWHT -> quantize(S_V) -> i8 -> cache (row-major)

RoPE is applied to K here (not in a separate pass) because the pipeline is
bf16 -> f32 -> RoPE -> FWHT -> quantize, and splitting RoPE out would require
materializing an f32 intermediate.

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
    quant_inv: Float32,
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: KVCache[_, head_dim, _],
    pos: Int,
    head: Int,
):
    """One K head: bf16 -> f32 -> RoPE -> FWHT -> quantize -> cache write."""
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

    # Fixed-scale quantize
    var vinv = SIMD[DType.float32, width](quant_inv)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= head_dim:
        var v = (work + k).load[width=width]()
        (qi_buf + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
        k += width

    cache.write_k(pos, head, qi_buf)


# ============================================================================
# Per-head write: V (FWHT + quantize + cache write, no RoPE)
# ============================================================================


def write_v_head[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    quant_inv: Float32,
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: KVCache[_, head_dim, _],
    pos: Int,
    head: Int,
):
    """One V head: bf16 -> f32 -> FWHT -> quantize -> cache write."""
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


def rope_and_kv_cache_write[head_dim: Int, num_kv_heads: Int, max_seq: Int](
    k_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    v_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    k_stride: Int,
    v_stride: Int,
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    cache: KVCache[max_seq, head_dim, num_kv_heads],
    pos: Int,
    seq_len: Int,
    s_k: Float32,
    s_v: Float32,
):
    """Write K and V from bf16 GEMV output into quantized KV cache.

    k_bf16/v_bf16: pointers to the K/V region within the QKV output.
    k_stride/v_stride: bf16 element stride between rows (= QKV_N for combined output).
    cos/sin: RoPE tables, [max_seq, head_dim/2].
    s_k/s_v: per-layer quantization scales.
    """
    comptime half = head_dim // 2
    var k_quant_inv = Float32(127) / s_k
    var v_quant_inv = Float32(127) / s_v

    # Stack-allocated working memory
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
                k_quant_inv, work, qi_buf,
                cache, actual_pos, g,
            )
            write_v_head[head_dim](
                v_bf16 + m * v_stride + g * head_dim,
                v_quant_inv, work, qi_buf,
                cache, actual_pos, g,
            )
