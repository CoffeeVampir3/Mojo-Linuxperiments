"""Per-head norm + RoPE + FWHT + KV cache write for Gemma 4.

Pipeline:
  K: bf16 -> f32 -> /rms(per-head) -> RoPE -> FWHT -> dynamic quantize -> VNNI scatter
  V: bf16 -> f32 -> /rms(no gamma) -> FWHT -> fixed quantize -> VNNI scatter

Uses Gemma4KVCache which stores both K and V in VNNI tile format:
  K: [head][tile_idx][k_slice × TILE_BYTES] — u8 (XOR 0x80) for tdpbsud
  V: [head][hd_tile][k_group × TILE_BYTES] — i8, transposed for V-agg tileload

All buffers passed in. Kernels use only stack InlineArrays for small temporaries.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental.amx import VNNI_BLK
from experimental3.kv_cache import Gemma4KVCache
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8, fixed_quantize_i8
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
# V head: /rms (no gamma) -> FWHT -> fixed quantize -> VNNI cache
# ============================================================================


def write_v_head_with_inv_rms[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
    inv_rms: Float32,
):
    """V head: bf16 -> f32 -> *inv_rms -> FWHT -> dynamic absmax quantize -> VNNI cache."""
    comptime width = simd_width_of[DType.float32]()

    var k = 0
    while k + width <= head_dim:
        (work + k).store(
            (src_bf16 + k).load[width=width]().cast[DType.float32]() * inv_rms
        )
        k += width

    fwht_block[head_dim](work)

    var absmax = absmax_quantize_i8[head_dim](work, qi_buf)
    cache.write_v(pos, head, qi_buf.bitcast[Int8]())
    cache.write_v_scale(pos, head, absmax)


def write_v_head_normed[head_dim: Int](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cache: Gemma4KVCache[_, head_dim, _, _],
    pos: Int,
    head: Int,
    eps: Float32,
):
    """V head: bf16 -> f32 -> /rms -> FWHT -> dynamic absmax quantize -> VNNI cache."""
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


# ============================================================================
# Validation
# ============================================================================


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * Float64(4.0) - Float64(2.0)


def scalar_rms_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var sum_sq = Float64(0)
    for i in range(n):
        sum_sq += buf[i] * buf[i]
    return Float64(sqrt[DType.float64, 1](sum_sq / Float64(n) + Float64(1e-6)))


def scalar_fwht_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var stride = 1
    while stride < n:
        var i = 0
        while i < n:
            for j in range(stride):
                var a = buf[i + j]
                var b = buf[i + j + stride]
                buf[i + j] = a + b
                buf[i + j + stride] = a - b
            i += stride * 2
        stride *= 2
    var sc = Float64(1.0) / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] = buf[i] * sc


def cosine_sim_f64(a: UnsafePointer[Float64, MutAnyOrigin], b: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (Float64(sqrt[DType.float64, 1](na)) * Float64(sqrt[DType.float64, 1](nb)))


def validate_k_normed[head_dim: Int]():
    """Validate K head pipeline: load -> /rms -> full rope -> fwht -> quantize -> VNNI cache roundtrip."""
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]
    var rng = UInt64(0xCAFEBABE12345678)

    var src = alloc[Scalar[DType.bfloat16]](head_dim)
    var k_norm = alloc[Scalar[DType.bfloat16]](head_dim)
    var cos_f32 = alloc[Float32](half)
    var sin_f32 = alloc[Float32](half)
    var expected = alloc[Float64](head_dim)
    var work = alloc[Float32](head_dim)
    var qi = alloc[Scalar[DType.int8]](head_dim)

    for i in range(head_dim):
        src[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        k_norm[i] = Scalar[DType.bfloat16](Float32(1.0))
    for i in range(half):
        var angle = xorshift64(rng) * 0.5
        cos_f32[i] = Float32(1.0 - angle * angle * 0.5)
        sin_f32[i] = Float32(angle)

    # f64 reference: load -> /rms -> rope -> fwht
    for i in range(head_dim):
        expected[i] = Float64(src[i])
    var rms = scalar_rms_f64(expected.bitcast[Float64](), head_dim)
    for i in range(head_dim):
        expected[i] /= rms
    var pre_fwht = alloc[Float64](head_dim)
    for j in range(half):
        var lo = expected[j]
        var hi = expected[half + j]
        expected[j] = lo * Float64(cos_f32[j]) - hi * Float64(sin_f32[j])
        expected[half + j] = hi * Float64(cos_f32[j]) + lo * Float64(sin_f32[j])
    for i in range(head_dim):
        pre_fwht[i] = expected[i]
    scalar_fwht_f64(expected.bitcast[Float64](), head_dim)

    # Kernel pipeline — writes into actual Gemma4KVCache
    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    comptime test_pos = 7
    comptime test_head = 1
    write_k_head_normed[head_dim](
        src, k_norm, cos_f32, sin_f32, work, qi,
        cache, test_pos, test_head, Float32(1e-6))

    # Pre-quantize accuracy: kernel f32 vs f64 reference
    var kernel_f64 = alloc[Float64](head_dim)
    for i in range(head_dim):
        kernel_f64[i] = Float64(work[i])
    var cos_pre = cosine_sim_f64(expected.bitcast[Float64](), kernel_f64.bitcast[Float64](), head_dim)
    print("  pre-quantize cosine (f32 vs f64):            " + String(cos_pre))

    # Round-trip: read K back from width-packed VNNI cache, dequant, inverse FWHT
    var stored_scale = cache.k_scale_ptr(test_head)[test_pos]
    comptime WIDTH = Cache.WIDTH

    var recovered_i8 = alloc[Scalar[DType.int8]](head_dim)
    var pos_group = test_pos // WIDTH
    var slot = test_pos % WIDTH
    var k_pg = cache.k_pg_ptr(test_head, pos_group)
    for kdg in range(Cache.K_DIM_GROUPS):
        var base = k_pg + kdg * WIDTH * VNNI_BLK + slot * VNNI_BLK
        for b in range(VNNI_BLK):
            recovered_i8[kdg * VNNI_BLK + b] = (base.bitcast[Scalar[DType.uint8]]()[b] ^ UInt8(0x80)).cast[DType.int8]()

    var recovered = alloc[Float64](head_dim)
    var dq = Float64(stored_scale) / 127.0
    for i in range(head_dim):
        recovered[i] = Float64(Int64(recovered_i8[i])) * dq
    scalar_fwht_f64(recovered.bitcast[Float64](), head_dim)
    var cos_rt = cosine_sim_f64(pre_fwht.bitcast[Float64](), recovered.bitcast[Float64](), head_dim)
    print("  round-trip cosine (VNNI cache->dequant->invFWHT): " + String(cos_rt))
    print("  stored scale:                                     " + String(stored_scale))

    src.free()
    k_norm.free()
    cos_f32.free()
    sin_f32.free()
    expected.free()
    pre_fwht.free()
    work.free()
    qi.free()
    kernel_f64.free()
    cache_buf.free()
    recovered_i8.free()
    recovered.free()


def validate_k_partial[head_dim: Int, rope_dims: Int]():
    """Validate partial RoPE K pipeline for full-attention layers."""
    comptime width = simd_width_of[DType.float32]()
    comptime half = head_dim // 2
    comptime rope_half = rope_dims // 2
    var rng = UInt64(0xABCD1234DEAD5678)

    var src = alloc[Scalar[DType.bfloat16]](head_dim)
    var k_norm = alloc[Scalar[DType.bfloat16]](head_dim)
    var cos_f32 = alloc[Float32](rope_half)
    var sin_f32 = alloc[Float32](rope_half)
    var expected = alloc[Float64](head_dim)
    var work = alloc[Float32](head_dim)
    var qi = alloc[Scalar[DType.int8]](head_dim)

    for i in range(head_dim):
        src[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
        k_norm[i] = Scalar[DType.bfloat16](Float32(1.0))
    for i in range(rope_half):
        var angle = xorshift64(rng) * 0.5
        cos_f32[i] = Float32(1.0 - angle * angle * 0.5)
        sin_f32[i] = Float32(angle)

    # f64 reference
    for i in range(head_dim):
        expected[i] = Float64(src[i])
    var rms = scalar_rms_f64(expected.bitcast[Float64](), head_dim)
    for i in range(head_dim):
        expected[i] /= rms
    for j in range(rope_half):
        var lo = expected[j]
        var hi = expected[half + j]
        expected[j] = lo * Float64(cos_f32[j]) - hi * Float64(sin_f32[j])
        expected[half + j] = hi * Float64(cos_f32[j]) + lo * Float64(sin_f32[j])
    var pre_fwht = alloc[Float64](head_dim)
    for i in range(head_dim):
        pre_fwht[i] = expected[i]
    scalar_fwht_f64(expected.bitcast[Float64](), head_dim)

    # Kernel pipeline — uses Gemma4KVCache
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]
    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    comptime test_pos = 3
    comptime test_head = 0
    write_k_head_normed_partial[head_dim, rope_dims](
        src, k_norm, cos_f32, sin_f32, work, qi,
        cache, test_pos, test_head, Float32(1e-6))

    var kernel_f64 = alloc[Float64](head_dim)
    for i in range(head_dim):
        kernel_f64[i] = Float64(work[i])
    var cos_pre = cosine_sim_f64(expected.bitcast[Float64](), kernel_f64.bitcast[Float64](), head_dim)
    print("  pre-quantize cosine (f32 vs f64):            " + String(cos_pre))

    # Round-trip via dequant of qi_buf (not from cache, just math check)
    var absmax = Float32(0)
    for i in range(head_dim):
        var a = qi[i].__abs__()
        if Float32(a) > absmax:
            absmax = Float32(a)
    # absmax should be 127 since we quantize to fill the range
    var recovered = alloc[Float64](head_dim)
    var k_scale_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=cache.k_scale_base + test_head * max_seq * 4 + test_pos * 4)
    var stored_scale = k_scale_ptr[]
    var dq = Float64(stored_scale) / 127.0
    for i in range(head_dim):
        recovered[i] = Float64(Int64(qi[i])) * dq
    scalar_fwht_f64(recovered.bitcast[Float64](), head_dim)
    var cos_rt = cosine_sim_f64(pre_fwht.bitcast[Float64](), recovered.bitcast[Float64](), head_dim)
    print("  round-trip cosine (quant->dequant->invFWHT):  " + String(cos_rt))

    src.free()
    k_norm.free()
    cos_f32.free()
    sin_f32.free()
    expected.free()
    pre_fwht.free()
    work.free()
    qi.free()
    kernel_f64.free()
    cache_buf.free()
    recovered.free()


def validate_v_normed[head_dim: Int]():
    """Validate V head pipeline: /rms -> FWHT -> fixed quantize -> VNNI cache roundtrip."""
    comptime width = simd_width_of[DType.float32]()
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]
    var rng = UInt64(0x1234ABCDDEAD5678)

    var src = alloc[Scalar[DType.bfloat16]](head_dim)
    var expected = alloc[Float64](head_dim)
    var normed_f64 = alloc[Float64](head_dim)
    var work = alloc[Float32](head_dim)
    var qi = alloc[Scalar[DType.int8]](head_dim)
    var recovered = alloc[Float64](head_dim)

    for i in range(head_dim):
        src[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

    # f64 reference: /rms -> fwht
    for i in range(head_dim):
        expected[i] = Float64(src[i])
    var rms = scalar_rms_f64(expected.bitcast[Float64](), head_dim)
    for i in range(head_dim):
        expected[i] /= rms
        normed_f64[i] = expected[i]
    scalar_fwht_f64(expected.bitcast[Float64](), head_dim)

    # Pre-quantize accuracy
    var k = 0
    while k + width <= head_dim:
        (work + k).store((src + k).load[width=width]().cast[DType.float32]())
        k += width
    rms_divide[head_dim](work, Float32(1e-6))
    fwht_block[head_dim](work)

    var kernel_f64 = alloc[Float64](head_dim)
    for i in range(head_dim):
        kernel_f64[i] = Float64(work[i])
    var cos_pre = cosine_sim_f64(expected.bitcast[Float64](), kernel_f64.bitcast[Float64](), head_dim)
    print("  pre-quantize cosine (f32 vs f64):            " + String(cos_pre))

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    comptime test_pos = 5
    comptime test_head = 0
    write_v_head_normed[head_dim](
        src, work, qi,
        cache, test_pos, test_head, Float32(1e-6))

    # Read V back from width-packed VNNI layout and verify values match qi_buf
    from experimental.amx import VNNI_BLK
    comptime WIDTH = Cache.WIDTH
    var mismatches = 0
    var pos_group = test_pos // WIDTH
    var sub_quad = (test_pos % WIDTH) // VNNI_BLK
    var vnni_slot = test_pos % VNNI_BLK
    var v_pg = cache.v_pg_ptr(test_head, pos_group)

    for cg in range(Cache.V_CHANNEL_GROUPS):
        var cg_base = v_pg + cg * Cache.V_CG_BYTES + sub_quad * WIDTH * VNNI_BLK
        for ci in range(WIDTH):
            var got = cg_base[ci * VNNI_BLK + vnni_slot]
            var expected_val = qi[cg * WIDTH + ci].cast[DType.int8]()
            if got != expected_val:
                mismatches += 1

    print("  V VNNI roundtrip mismatches:                  " + String(mismatches))

    # Dequant round-trip: qi -> inverse FWHT -> compare vs normed original
    var dq = Float64(cache.v_scale_ptr(test_head)[test_pos]) / 127.0
    for i in range(head_dim):
        recovered[i] = Float64(Int64(qi[i])) * dq
    scalar_fwht_f64(recovered.bitcast[Float64](), head_dim)
    var cos_rt = cosine_sim_f64(normed_f64.bitcast[Float64](), recovered.bitcast[Float64](), head_dim)
    print("  round-trip cosine (fixed quant->invFWHT):     " + String(cos_rt))

    src.free()
    expected.free()
    normed_f64.free()
    work.free()
    qi.free()
    kernel_f64.free()
    cache_buf.free()
    recovered.free()


def main():
    print("=== rope_and_kv_cache_write validation (Gemma4KVCache) ===")

    print("\nK head normed, head_dim=256 (sliding):")
    validate_k_normed[256]()

    print("\nK head normed partial, head_dim=512, rope_dims=128 (full):")
    validate_k_partial[512, 128]()

    print("\nV head normed, head_dim=256 (sliding):")
    validate_v_normed[256]()

    print("\nV head normed, head_dim=512 (full):")
    validate_v_normed[512]()
