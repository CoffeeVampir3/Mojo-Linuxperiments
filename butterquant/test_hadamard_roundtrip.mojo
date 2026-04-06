"""Hadamard quantization round-trip test.

Takes bf16 Q/K from layer 0 (proven cos=0.9999 vs reference), applies the
full attention quantization pipeline per head:

    RoPE → FWHT → quantize(S) → dequant → FWHT (inverse = self)

Compares the round-tripped vector against RoPE(original) to measure pure
quantization distortion of the Hadamard scheme on real Q/K vectors.
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.pathlib import Path

from modeling.smollm2_butterquant_tp import (
    SmolLM2ButterQuant, SmolLM2Config, concentration_constant, FWHT_BLOCK,
)
from modeling.model_spec import BF16
from tokenizer import load_tokenizer
from experimental.amx import init_intel_amx
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from simd_math import roundeven

comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"


def rope_head(
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    head_dim: Int,
):
    """bf16 → f32 + RoPE one head into dst."""
    comptime width = simd_width_of[DType.float32]()
    var half = head_dim // 2

    # Load bf16 → f32
    var k = 0
    while k + width <= head_dim:
        (dst + k).store((src + k).load[width=width]().cast[DType.float32]())
        k += width

    # RoPE: [lo, hi] -> [lo*cos - hi*sin, hi*cos + lo*sin]
    k = 0
    while k + width <= half:
        var x_lo = (dst + k).load[width=width]()
        var x_hi = (dst + half + k).load[width=width]()
        var cv = (cos + k).load[width=width]()
        var sv = (sin + k).load[width=width]()
        (dst + k).store(x_lo * cv - x_hi * sv)
        (dst + half + k).store(x_hi * cv + x_lo * sv)
        k += width


def fwht_quantize_dequant_fwht(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    head_dim: Int,
    scale: Float32,
):
    """In-place: FWHT → quantize(fixed S) → dequant → FWHT."""
    comptime width = simd_width_of[DType.float32]()

    fwht_block[C.HEAD_DIM](buf)

    var quant_inv = Float32(127) / scale
    var dequant_sc = scale / Float32(127)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    var k = 0
    while k + width <= head_dim:
        var v = (buf + k).load[width=width]()
        var qi = min(max(roundeven(v * quant_inv), lo), hi)
        (buf + k).store(qi * dequant_sc)
        k += width

    fwht_block[C.HEAD_DIM](buf)


def fwht_quantize_dequant_fwht_dynamic(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    head_dim: Int,
):
    """In-place: FWHT → quantize(absmax) → dequant → FWHT. Per-head dynamic scale."""
    comptime width = simd_width_of[DType.float32]()

    fwht_block[C.HEAD_DIM](buf)

    # Compute absmax of the FWHT'd values
    var vmax = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= head_dim:
        vmax = max(vmax, (buf + k).load[width=width]().__abs__())
        k += width
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    var quant_inv = Float32(127) / absmax
    var dequant_sc = absmax / Float32(127)
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= head_dim:
        var v = (buf + k).load[width=width]()
        var qi = min(max(roundeven(v * quant_inv), lo), hi)
        (buf + k).store(qi * dequant_sc)
        k += width

    fwht_block[C.HEAD_DIM](buf)


def compare_f32(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Returns (cos_sim, rel_l2, norm_a, norm_b)."""
    comptime width = simd_width_of[DType.float32]()
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    var dsq = Float64(0)

    var k = 0
    while k + width <= count:
        var va = (a + k).load[width=width]()
        var vb = (b + k).load[width=width]()
        var d = va - vb
        dot += (va * vb).cast[DType.float64]().reduce_add()
        na += (va * va).cast[DType.float64]().reduce_add()
        nb += (vb * vb).cast[DType.float64]().reduce_add()
        dsq += (d * d).cast[DType.float64]().reduce_add()
        k += width
    while k < count:
        var va = Float64((a + k)[])
        var vb = Float64((b + k)[])
        dot += va * vb; na += va * va; nb += vb * vb; dsq += (va - vb) * (va - vb)
        k += 1

    var cos = dot / (na.__pow__(0.5) * nb.__pow__(0.5) + 1e-30)
    var rel = dsq.__pow__(0.5) / (na.__pow__(0.5) + 1e-30)
    return (cos, rel, na.__pow__(0.5), nb.__pow__(0.5))


def main():
    _ = init_intel_amx()

    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("tokenizer load failed"); return
    var tok = tok_opt.take()

    var prompt = "The quick brown fox jumps over the lazy dog. " * 5
    var token_ids = tok.encode(prompt)
    var seq_len = len(token_ids)

    print("loading quant model...")
    var model_opt = SmolLM2ButterQuant[1].load(Path(QUANT_PATH))
    if not model_opt:
        print("model load failed"); return
    var model = model_opt.take()

    # Embed + get bf16 QKV from layer 0
    var tp = model.token_buffer()
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](token_ids[i])
    model.debug_embed(Int(tp), seq_len)

    var q_bf16 = alloc[Scalar[DType.bfloat16]](C.HIDDEN)
    var k_bf16 = alloc[Scalar[DType.bfloat16]](C.KV_HIDDEN)
    var v_bf16 = alloc[Scalar[DType.bfloat16]](C.KV_HIDDEN)
    model.debug_qkv(0, seq_len, q_bf16, k_bf16, v_bf16)

    # Get RoPE tables and layer scales
    var rv = model.rank(0)
    comptime HALF = C.HEAD_DIM // 2
    var last_pos = seq_len - 1
    var cos_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_cos().ptr) + last_pos * HALF
    var sin_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_sin().ptr) + last_pos * HALF
    var s_q = model.layer_scales[0].q_layer_scale
    var s_k = model.layer_scales[0].k_layer_scale
    var s_v = model.layer_scales[0].v_layer_scale

    print("tokens:", seq_len, " last_pos:", last_pos)
    print("S_Q=", s_q, " S_K=", s_k, " S_V=", s_v)
    print()

    # Working buffers
    var original = alloc[Float32](C.HEAD_DIM)
    var rt_fixed = alloc[Float32](C.HEAD_DIM)
    var rt_dynamic = alloc[Float32](C.HEAD_DIM)

    # Helper: load one head bf16 → f32 (no RoPE)
    def load_head(src_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
                  dst: UnsafePointer[Float32, MutAnyOrigin]):
        comptime width = simd_width_of[DType.float32]()
        var k = 0
        while k + width <= C.HEAD_DIM:
            (dst + k).store((src_bf16 + k).load[width=width]().cast[DType.float32]())
            k += width

    # === Q heads: fixed vs dynamic ===
    print("=== Q heads (" + String(C.NUM_HEADS) + " heads) ===")
    print("  head  |  fixed cos  fixed rel_l2  |  dynamic cos  dynamic rel_l2  | |orig|")
    var q_fixed_sum = Float64(0)
    var q_dyn_sum = Float64(0)
    for h in range(C.NUM_HEADS):
        rope_head(q_bf16 + h * C.HEAD_DIM, cos_ptr, sin_ptr, original, C.HEAD_DIM)

        memcpy(dest=rt_fixed, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht(rt_fixed, C.HEAD_DIM, s_q)
        var rf = compare_f32(original, rt_fixed, C.HEAD_DIM)

        memcpy(dest=rt_dynamic, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht_dynamic(rt_dynamic, C.HEAD_DIM)
        var rd = compare_f32(original, rt_dynamic, C.HEAD_DIM)

        q_fixed_sum += rf[0]
        q_dyn_sum += rd[0]
        print("  ", h, "  |", rf[0], rf[1], " |", rd[0], rd[1], " |", rf[2])
    print("  mean  |", q_fixed_sum / C.NUM_HEADS, "              |",
          q_dyn_sum / C.NUM_HEADS)
    print()

    # === K heads: fixed vs dynamic ===
    print("=== K heads (" + String(C.NUM_KV_HEADS) + " heads) ===")
    print("  head  |  fixed cos  fixed rel_l2  |  dynamic cos  dynamic rel_l2  | |orig|")
    var k_fixed_sum = Float64(0)
    var k_dyn_sum = Float64(0)
    for h in range(C.NUM_KV_HEADS):
        rope_head(k_bf16 + h * C.HEAD_DIM, cos_ptr, sin_ptr, original, C.HEAD_DIM)

        memcpy(dest=rt_fixed, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht(rt_fixed, C.HEAD_DIM, s_k)
        var rf = compare_f32(original, rt_fixed, C.HEAD_DIM)

        memcpy(dest=rt_dynamic, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht_dynamic(rt_dynamic, C.HEAD_DIM)
        var rd = compare_f32(original, rt_dynamic, C.HEAD_DIM)

        k_fixed_sum += rf[0]
        k_dyn_sum += rd[0]
        print("  ", h, "  |", rf[0], rf[1], " |", rd[0], rd[1], " |", rf[2])
    print("  mean  |", k_fixed_sum / C.NUM_KV_HEADS, "              |",
          k_dyn_sum / C.NUM_KV_HEADS)
    print()

    # === V heads: fixed vs dynamic (no RoPE) ===
    print("=== V heads (" + String(C.NUM_KV_HEADS) + " heads, no RoPE) ===")
    print("  head  |  fixed cos  fixed rel_l2  |  dynamic cos  dynamic rel_l2  | |orig|")
    var v_fixed_sum = Float64(0)
    var v_dyn_sum = Float64(0)
    for h in range(C.NUM_KV_HEADS):
        load_head(v_bf16 + h * C.HEAD_DIM, original)

        memcpy(dest=rt_fixed, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht(rt_fixed, C.HEAD_DIM, s_v)
        var rf = compare_f32(original, rt_fixed, C.HEAD_DIM)

        memcpy(dest=rt_dynamic, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht_dynamic(rt_dynamic, C.HEAD_DIM)
        var rd = compare_f32(original, rt_dynamic, C.HEAD_DIM)

        v_fixed_sum += rf[0]
        v_dyn_sum += rd[0]
        print("  ", h, "  |", rf[0], rf[1], " |", rd[0], rd[1], " |", rf[2])
    print("  mean  |", v_fixed_sum / C.NUM_KV_HEADS, "              |",
          v_dyn_sum / C.NUM_KV_HEADS)
    print()

    # === Dot product preservation: fixed vs dynamic ===
    print("=== Dot product preservation (Q head 0 vs K head 0) ===")
    var q_orig = alloc[Float32](C.HEAD_DIM)
    var k_orig = alloc[Float32](C.HEAD_DIM)
    var q_rt = alloc[Float32](C.HEAD_DIM)
    var k_rt = alloc[Float32](C.HEAD_DIM)

    rope_head(q_bf16, cos_ptr, sin_ptr, q_orig, C.HEAD_DIM)
    rope_head(k_bf16, cos_ptr, sin_ptr, k_orig, C.HEAD_DIM)

    # Fixed scale
    memcpy(dest=q_rt, src=q_orig, count=C.HEAD_DIM)
    memcpy(dest=k_rt, src=k_orig, count=C.HEAD_DIM)
    fwht_quantize_dequant_fwht(q_rt, C.HEAD_DIM, s_q)
    fwht_quantize_dequant_fwht(k_rt, C.HEAD_DIM, s_k)
    var dot_exact = Float64(0)
    var dot_fixed = Float64(0)
    for i in range(C.HEAD_DIM):
        dot_exact += Float64(q_orig[i]) * Float64(k_orig[i])
        dot_fixed += Float64(q_rt[i]) * Float64(k_rt[i])

    # Dynamic scale
    memcpy(dest=q_rt, src=q_orig, count=C.HEAD_DIM)
    memcpy(dest=k_rt, src=k_orig, count=C.HEAD_DIM)
    fwht_quantize_dequant_fwht_dynamic(q_rt, C.HEAD_DIM)
    fwht_quantize_dequant_fwht_dynamic(k_rt, C.HEAD_DIM)
    var dot_dynamic = Float64(0)
    for i in range(C.HEAD_DIM):
        dot_dynamic += Float64(q_rt[i]) * Float64(k_rt[i])

    print("  exact   =", dot_exact)
    print("  fixed   =", dot_fixed, " ratio=", dot_fixed / (dot_exact + 1e-30))
    print("  dynamic =", dot_dynamic, " ratio=", dot_dynamic / (dot_exact + 1e-30))

    # =================================================================
    # === PROPOSED DESIGN: dynamic Q/K + fixed V ===
    # =================================================================
    print()
    print("=== Proposed design: dynamic Q/K scales, fixed V scale ===")
    print("  Q/K: per-head absmax after FWHT")
    print("  V:   fixed S_V =", s_v, "(corrected formula)")
    print()

    # Q round-trip with dynamic scale
    print("  --- Q heads (dynamic) ---")
    var q_proposed_sum = Float64(0)
    for h in range(C.NUM_HEADS):
        rope_head(q_bf16 + h * C.HEAD_DIM, cos_ptr, sin_ptr, original, C.HEAD_DIM)
        memcpy(dest=rt_dynamic, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht_dynamic(rt_dynamic, C.HEAD_DIM)
        var rd = compare_f32(original, rt_dynamic, C.HEAD_DIM)
        q_proposed_sum += rd[0]
        print("    head", h, " cos=", rd[0], " rel_l2=", rd[1])
    print("    mean cos=", q_proposed_sum / Float64(C.NUM_HEADS))

    # K round-trip with dynamic scale
    print("  --- K heads (dynamic) ---")
    var k_proposed_sum = Float64(0)
    for h in range(C.NUM_KV_HEADS):
        rope_head(k_bf16 + h * C.HEAD_DIM, cos_ptr, sin_ptr, original, C.HEAD_DIM)
        memcpy(dest=rt_dynamic, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht_dynamic(rt_dynamic, C.HEAD_DIM)
        var rd = compare_f32(original, rt_dynamic, C.HEAD_DIM)
        k_proposed_sum += rd[0]
        print("    head", h, " cos=", rd[0], " rel_l2=", rd[1])
    print("    mean cos=", k_proposed_sum / Float64(C.NUM_KV_HEADS))

    # V round-trip with FIXED corrected scale
    print("  --- V heads (fixed S_V) ---")
    var v_proposed_sum = Float64(0)
    for h in range(C.NUM_KV_HEADS):
        load_head(v_bf16 + h * C.HEAD_DIM, original)
        memcpy(dest=rt_fixed, src=original, count=C.HEAD_DIM)
        fwht_quantize_dequant_fwht(rt_fixed, C.HEAD_DIM, s_v)
        var rf = compare_f32(original, rt_fixed, C.HEAD_DIM)
        v_proposed_sum += rf[0]
        print("    head", h, " cos=", rf[0], " rel_l2=", rf[1])
    print("    mean cos=", v_proposed_sum / Float64(C.NUM_KV_HEADS))

    # Dot product test: dynamic Q × dynamic K
    print("  --- Dot product (dynamic Q × dynamic K) ---")
    for qh in range(min(C.NUM_HEADS, 3)):
        for kh in range(C.NUM_KV_HEADS):
            rope_head(q_bf16 + qh * C.HEAD_DIM, cos_ptr, sin_ptr, q_orig, C.HEAD_DIM)
            rope_head(k_bf16 + kh * C.HEAD_DIM, cos_ptr, sin_ptr, k_orig, C.HEAD_DIM)
            memcpy(dest=q_rt, src=q_orig, count=C.HEAD_DIM)
            memcpy(dest=k_rt, src=k_orig, count=C.HEAD_DIM)
            fwht_quantize_dequant_fwht_dynamic(q_rt, C.HEAD_DIM)
            fwht_quantize_dequant_fwht_dynamic(k_rt, C.HEAD_DIM)
            var de = Float64(0)
            var dr = Float64(0)
            for i in range(C.HEAD_DIM):
                de += Float64(q_orig[i]) * Float64(k_orig[i])
                dr += Float64(q_rt[i]) * Float64(k_rt[i])
            var ratio = dr / (de + Float64(1e-30))
            print("    Q", qh, "× K", kh, " exact=", de, " quant=", dr, " ratio=", ratio)

    q_bf16.free(); k_bf16.free(); v_bf16.free()
    original.free(); rt_fixed.free(); rt_dynamic.free()
    q_orig.free(); k_orig.free(); q_rt.free(); k_rt.free()
    _ = model
