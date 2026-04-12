"""S_post round-trip test: prove dynamic absmax fixes the MLP quantization.

Loads the quantized model, runs layer 0 through attention (using the new
dynamic Q/K scales), then computes the MLP gate+up GEMV and silu(gate)*up
in f32. Tests the FWHT → quantize → dequant → FWHT round-trip on the actual
silu output with fixed S_post vs dynamic per-token absmax.

Same methodology as test_hadamard_roundtrip.mojo but for the MLP path.
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.pathlib import Path

from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant, SmolLM2Config, FWHT_BLOCK
from modeling.model_spec import BF16
from tokenizer import load_tokenizer
from experimental.amx import init_intel_amx
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from simd_math import roundeven, exp_f32

comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"
comptime GATE_ROWS = C.INTERMEDIATE  # tp=1


def fwht_quantize_dequant_fwht_fixed[block: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin], cols: Int, scale: Float32,
):
    """Block-diagonal FWHT → fixed-scale quantize → dequant → FWHT."""
    comptime width = simd_width_of[DType.float32]()

    for b in range(cols // block):
        fwht_block[block](buf + b * block)

    var vinv = SIMD[DType.float32, width](Float32(127) / scale)
    var dq = SIMD[DType.float32, width](scale / Float32(127))
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    var k = 0
    while k + width <= cols:
        var v = (buf + k).load[width=width]()
        (buf + k).store(min(max(roundeven(v * vinv), lo), hi) * dq)
        k += width

    for b in range(cols // block):
        fwht_block[block](buf + b * block)


def fwht_quantize_dequant_fwht_dynamic[block: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin], cols: Int,
):
    """Block-diagonal FWHT → dynamic absmax quantize → dequant → FWHT."""
    comptime width = simd_width_of[DType.float32]()

    for b in range(cols // block):
        fwht_block[block](buf + b * block)

    # Per-row absmax (one scale for the whole row)
    var vmax = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        vmax = max(vmax, (buf + k).load[width=width]().__abs__())
        k += width
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)

    var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
    var dq = SIMD[DType.float32, width](absmax / Float32(127))
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    k = 0
    while k + width <= cols:
        var v = (buf + k).load[width=width]()
        (buf + k).store(min(max(roundeven(v * vinv), lo), hi) * dq)
        k += width

    for b in range(cols // block):
        fwht_block[block](buf + b * block)


def compare_f32(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    count: Int, label: String,
):
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
    var cos = dot / (na.__pow__(0.5) * nb.__pow__(0.5) + Float64(1e-30))
    var rel = dsq.__pow__(0.5) / (na.__pow__(0.5) + Float64(1e-30))
    print(label, " cos=", cos, " rel_l2=", rel,
          " |a|=", na.__pow__(0.5), " |b|=", nb.__pow__(0.5))


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

    var tp = model.token_buffer()
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](token_ids[i])
    model.debug_embed(Int(tp), seq_len)
    model.debug_layer_attn(0, seq_len, 0)

    # Now x_main holds post-attention state for layer 0.
    # Run the MLP gate+up GEMV to get gate and up bf16 outputs.
    # Then compute silu(gate)*up in f32 manually.

    comptime M = SmolLM2ButterQuant[1].M
    comptime L = M.LAYER
    comptime QKV_N = L.QKV_N
    comptime GATE_UP_N = L.GATE_UP_N
    comptime WORK_F32 = C.HIDDEN * 64
    comptime ACT_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()

    var rv = model.rank(0)
    var sb = rv.scratch_base()
    var s_act = model.s_act
    var s_act_dequant = s_act / Float32(127)
    var ls = model.layer_scales[0]

    # RMSNorm + FWHT + quantize activation
    var act_work = model.scratch.borrow[UInt8, ACT_WORK_BYTES]()
    var act_i8_off = act_work.offset
    var work_off = act_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

    from experimental2.kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize
    from experimental2.kernels.int8_gemv import int8_gemv

    rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
        rv.x_main(seq_len).ptr, sb + act_i8_off, sb + work_off,
        s_act, Float32(1e-5), seq_len, model.pools[0],
    ).join()

    # Gate+up GEMV
    var gate_up_lease = model.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * GATE_UP_N]()
    int8_gemv[GATE_UP_N, C.HIDDEN](
        sb + act_i8_off,
        rv.layer_weight[L.GATE_PROJ](0).ptr,
        rv.layer_weight[L.GATE_COLSUM](0).ptr,
        rv.layer_weight[L.GATE_ROW_SCALE](0).ptr,
        sb + gate_up_lease.offset, seq_len, s_act_dequant,
        model.pools[0],
    ).join()
    act_work^.release()

    # Extract last token's gate and up bf16 outputs, compute silu(gate)*up in f32
    comptime GATE_COLS = C.INTERMEDIATE  # tp=1
    var gu_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=sb + gate_up_lease.offset)
    var last_row = gu_ptr + (seq_len - 1) * GATE_UP_N
    comptime width = simd_width_of[DType.float32]()

    var silu_output = alloc[Float32](GATE_COLS)
    var k = 0
    while k + width <= GATE_COLS:
        var g = (last_row + k).load[width=width]().cast[DType.float32]()
        var u = (last_row + GATE_COLS + k).load[width=width]().cast[DType.float32]()
        # SiLU(g) = g / (1 + exp(-g))
        var sig = SIMD[DType.float32, width](1.0) / (SIMD[DType.float32, width](1.0) + exp_f32[width](-g))
        (silu_output + k).store(g * sig * u)
        k += width

    gate_up_lease^.release()

    print("tokens:", seq_len)
    print("S_act =", s_act)
    print("S_post =", ls.post_layer_scale)

    # === S_act check: RMSNorm'd activation before gate+up GEMV ===
    # Read x_main (post-attention), apply RMSNorm in f32, check FWHT absmax
    print()
    print("=== S_act check (activation before gate+up GEMV) ===")
    var x_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=rv.x_main(seq_len).ptr)
    var last_x = x_ptr + (seq_len - 1) * C.HIDDEN
    var act_f32 = alloc[Float32](C.HIDDEN)

    # Load bf16 → f32
    k = 0
    while k + width <= C.HIDDEN:
        (act_f32 + k).store((last_x + k).load[width=width]().cast[DType.float32]())
        k += width

    # RMSNorm (no gamma — absorbed into weights)
    var rms_sum = Float64(0)
    for i in range(C.HIDDEN):
        rms_sum += Float64(act_f32[i]) * Float64(act_f32[i])
    var rms = Float32((rms_sum / Float64(C.HIDDEN) + Float64(1e-5)).__pow__(0.5))
    var inv_rms = Float32(1.0) / rms
    for i in range(C.HIDDEN):
        act_f32[i] = act_f32[i] * inv_rms

    # FWHT and check absmax
    var act_fwht = alloc[Float32](C.HIDDEN)
    memcpy(dest=act_fwht, src=act_f32, count=C.HIDDEN)
    for b in range(C.HIDDEN // FWHT_BLOCK):
        fwht_block[FWHT_BLOCK](act_fwht + b * FWHT_BLOCK)
    var act_fwht_max = Float32(0)
    for i in range(C.HIDDEN):
        var a = act_fwht[i]
        if a < Float32(0):
            a = -a
        if a > act_fwht_max:
            act_fwht_max = a
    print("RMSNorm'd activation: ||x||/sqrt(d) =", rms, " (should be ~1)")
    print("After FWHT: absmax=", act_fwht_max, " S_act covers=", s_act)
    print("Ratio actual/S_act=", Float64(act_fwht_max) / Float64(s_act))

    # Round-trip
    var act_rt_fixed = alloc[Float32](C.HIDDEN)
    memcpy(dest=act_rt_fixed, src=act_f32, count=C.HIDDEN)
    fwht_quantize_dequant_fwht_fixed[FWHT_BLOCK](act_rt_fixed, C.HIDDEN, s_act)
    compare_f32(act_f32, act_rt_fixed, C.HIDDEN, "S_act fixed   ")

    var act_rt_dyn = alloc[Float32](C.HIDDEN)
    memcpy(dest=act_rt_dyn, src=act_f32, count=C.HIDDEN)
    fwht_quantize_dequant_fwht_dynamic[FWHT_BLOCK](act_rt_dyn, C.HIDDEN)
    compare_f32(act_f32, act_rt_dyn, C.HIDDEN, "S_act dynamic ")

    act_f32.free()
    act_fwht.free()
    act_rt_fixed.free()
    act_rt_dyn.free()
    print()

    # Compute actual absmax of silu output
    var actual_max = Float32(0)
    for i in range(GATE_COLS):
        var a = silu_output[i]
        if a < Float32(0):
            a = -a
        if a > actual_max:
            actual_max = a
    print("Silu output: cols=", GATE_COLS, " absmax=", actual_max)

    # After FWHT, what's the absmax?
    var fwht_check = alloc[Float32](GATE_COLS)
    memcpy(dest=fwht_check, src=silu_output, count=GATE_COLS)
    for b in range(GATE_COLS // FWHT_BLOCK):
        fwht_block[FWHT_BLOCK](fwht_check + b * FWHT_BLOCK)
    var fwht_max = Float32(0)
    for i in range(GATE_COLS):
        var a = fwht_check[i]
        if a < Float32(0):
            a = -a
        if a > fwht_max:
            fwht_max = a
    print("After FWHT: absmax=", fwht_max, " S_post covers=", ls.post_layer_scale)
    print("Ratio actual/S_post=", Float64(fwht_max) / Float64(ls.post_layer_scale))
    fwht_check.free()

    print()
    print("=== Round-trip: fixed S_post vs dynamic absmax ===")

    # Fixed S_post round-trip
    var rt_fixed = alloc[Float32](GATE_COLS)
    memcpy(dest=rt_fixed, src=silu_output, count=GATE_COLS)
    fwht_quantize_dequant_fwht_fixed[FWHT_BLOCK](rt_fixed, GATE_COLS, ls.post_layer_scale)
    compare_f32(silu_output, rt_fixed, GATE_COLS, "fixed S_post  ")

    # Dynamic absmax round-trip
    var rt_dynamic = alloc[Float32](GATE_COLS)
    memcpy(dest=rt_dynamic, src=silu_output, count=GATE_COLS)
    fwht_quantize_dequant_fwht_dynamic[FWHT_BLOCK](rt_dynamic, GATE_COLS)
    compare_f32(silu_output, rt_dynamic, GATE_COLS, "dynamic absmax")

    silu_output.free()
    rt_fixed.free()
    rt_dynamic.free()
    _ = model
