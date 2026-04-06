"""Layer-by-layer stability comparison: bf16 reference vs ButterQuant int8.

Runs both models in lockstep through a prefill, comparing x_main (the
residual stream) after each layer. Reports cosine similarity and relative
L2 error at each of the 30 layer boundaries plus post-embed.

If cosine similarity decays rapidly across layers, quantization error is
compounding and the scheme is unstable. If it holds steady, the per-layer
error is bounded and the issue is elsewhere.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path

from modeling.smollm2_tp import SmolLM2TP, SmolLM2Config
from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant
from modeling.model_spec import BF16
from tokenizer import load_tokenizer
from experimental.amx import init_intel_amx

comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime BF16_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"


def compare_bf16(
    rp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    qp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int, label: String,
):
    """Compare two bf16 vectors element-wise."""
    comptime width = simd_width_of[DType.float32]()
    var dot = Float64(0)
    var norm_r = Float64(0)
    var norm_q = Float64(0)
    var diff_sq = Float64(0)

    var k = 0
    while k + width <= count:
        var r = (rp + k).load[width=width]().cast[DType.float32]()
        var q = (qp + k).load[width=width]().cast[DType.float32]()
        var d = r - q
        dot += (r * q).cast[DType.float64]().reduce_add()
        norm_r += (r * r).cast[DType.float64]().reduce_add()
        norm_q += (q * q).cast[DType.float64]().reduce_add()
        diff_sq += (d * d).cast[DType.float64]().reduce_add()
        k += width
    while k < count:
        var r = Float64((rp + k)[].cast[DType.float32]())
        var q = Float64((qp + k)[].cast[DType.float32]())
        dot += r * q; norm_r += r * r; norm_q += q * q; diff_sq += (r - q) * (r - q)
        k += 1

    var cos_sim = dot / (norm_r.__pow__(0.5) * norm_q.__pow__(0.5) + 1e-30)
    var rel_l2 = diff_sq.__pow__(0.5) / (norm_r.__pow__(0.5) + 1e-30)

    print(label, " cos=", cos_sim, " rel_l2=", rel_l2,
          " |ref|=", norm_r.__pow__(0.5), " |quant|=", norm_q.__pow__(0.5))


def compare_hidden(
    ref_ptr: Int, quant_ptr: Int, seq_len: Int, label: String,
):
    """Compare last token's hidden state (HIDDEN elements)."""
    var offset = (seq_len - 1) * C.HIDDEN
    compare_bf16(
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ref_ptr) + offset,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=quant_ptr) + offset,
        C.HIDDEN, label,
    )


def main():
    _ = init_intel_amx()

    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("tokenizer load failed"); return
    var tok = tok_opt.take()

    var prompt = "The quick brown fox jumps over the lazy dog. " * 5
    var token_ids = tok.encode(prompt)
    var seq_len = len(token_ids)
    print("tokens:", seq_len)

    # Load bf16 reference model
    print("loading bf16 model...")
    var reference_opt = SmolLM2TP[BF16, 1].load(Path(BF16_PATH))
    if not reference_opt:
        print("bf16 model load failed"); return
    var reference = reference_opt.take()

    # Load quantized model
    print("loading quant model...")
    var quant_opt = SmolLM2ButterQuant[1].load(Path(QUANT_PATH))
    if not quant_opt:
        print("quant model load failed"); return
    var quant = quant_opt.take()

    # Write tokens to both models
    var reference_tp = reference.token_buffer()
    var quant_tp = quant.token_buffer()
    for i in range(seq_len):
        reference_tp[i] = Scalar[DType.int32](token_ids[i])
        quant_tp[i] = Scalar[DType.int32](token_ids[i])

    # Embed
    reference.debug_embed(Int(reference_tp), seq_len)
    quant.debug_embed(Int(quant_tp), seq_len)
    compare_hidden(
        reference.debug_x_main_ptr(seq_len),
        quant.debug_x_main_ptr(seq_len),
        seq_len, "embed  ",
    )

    # Layer-by-layer: attn and mlp
    for layer in range(C.NUM_LAYERS):
        var lpad = " " + String(layer) if layer < 10 else String(layer)

        reference.debug_layer_attn(layer, seq_len, 0)
        quant.debug_layer_attn(layer, seq_len, 0)
        compare_hidden(
            reference.debug_x_main_ptr(seq_len),
            quant.debug_x_main_ptr(seq_len),
            seq_len, "layer " + lpad + " attn",
        )

        reference.debug_layer_mlp(layer, seq_len, 0)
        quant.debug_layer_mlp(layer, seq_len, 0)
        compare_hidden(
            reference.debug_x_main_ptr(seq_len),
            quant.debug_x_main_ptr(seq_len),
            seq_len, "layer " + lpad + "  mlp",
        )

    _ = reference
    _ = quant
