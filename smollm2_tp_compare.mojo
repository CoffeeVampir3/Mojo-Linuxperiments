"""Compare TP=1 and TP=3 SmolLM2 layer outputs on the same token.

This is a focused diagnostic for TP drift. It steps the model block-by-block
using the debug helpers in modeling/smollm2_tp.mojo and reports activation
error after each attention and MLP phase.
"""

from std.math import abs
from std.memory import UnsafePointer
from std.pathlib import Path

from modeling.model_spec import BF16
from modeling.smollm2_tp import SmolLM2Config, SmolLM2TP


comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime HIDDEN = SmolLM2Config.HIDDEN
comptime NUM_LAYERS = SmolLM2Config.NUM_LAYERS


@fieldwise_init
struct CompareStats(Copyable, ImplicitlyCopyable):
    var max_abs: Float32
    var max_idx: Int
    var gt_001: Int
    var gt_01: Int
    var gt_10: Int


def compare_bf16(ptr_a: Int, ptr_b: Int, elements: Int) -> CompareStats:
    var a = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr_a)
    var b = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr_b)

    var stats = CompareStats(
        max_abs=Float32(0),
        max_idx=-1,
        gt_001=0,
        gt_01=0,
        gt_10=0,
    )

    for i in range(elements):
        var av = Float32(a[i])
        var bv = Float32(b[i])
        var err = abs(av - bv)
        if err > stats.max_abs:
            stats.max_abs = err
            stats.max_idx = i
        if err > 0.001:
            stats.gt_001 += 1
        if err > 0.01:
            stats.gt_01 += 1
        if err > 0.1:
            stats.gt_10 += 1

    return stats


def print_stats(label: String, read stats: CompareStats, elements: Int):
    print(
        label,
        " max_abs=", stats.max_abs,
        " idx=", stats.max_idx,
        " >1e-3=", stats.gt_001, "/", elements,
        " >1e-2=", stats.gt_01,
        " >1e-1=", stats.gt_10,
    )


def main():
    var model1_opt = SmolLM2TP[BF16, 1].load(Path(MODEL_PATH))
    if not model1_opt:
        print("Failed to load TP=1 model")
        return
    var model3_opt = SmolLM2TP[BF16, 3].load(Path(MODEL_PATH))
    if not model3_opt:
        print("Failed to load TP=3 model")
        return

    var model1 = model1_opt.take()
    var model3 = model3_opt.take()

    var token = 42
    var seq_len = 1

    var buf1 = model1.token_buffer()
    var buf3 = model3.token_buffer()
    buf1[0] = Scalar[DType.int32](token)
    buf3[0] = Scalar[DType.int32](token)

    print("=== TP compare ===")
    print("token:", token, "seq_len:", seq_len)

    model1.debug_embed(Int(buf1), seq_len)
    model3.debug_embed(Int(buf3), seq_len)
    print_stats(
        "embed",
        compare_bf16(model1.debug_x_main_ptr(seq_len), model3.debug_x_main_ptr(seq_len), HIDDEN),
        HIDDEN,
    )

    for layer in range(NUM_LAYERS):
        model1.debug_layer_attn(layer, seq_len, 0)
        model3.debug_layer_attn(layer, seq_len, 0)
        print_stats(
            "layer " + String(layer) + " attn",
            compare_bf16(model1.debug_x_main_ptr(seq_len), model3.debug_x_main_ptr(seq_len), HIDDEN),
            HIDDEN,
        )

        model1.debug_layer_mlp(layer, seq_len, 0)
        model3.debug_layer_mlp(layer, seq_len, 0)
        print_stats(
            "layer " + String(layer) + " mlp ",
            compare_bf16(model1.debug_x_main_ptr(seq_len), model3.debug_x_main_ptr(seq_len), HIDDEN),
            HIDDEN,
        )
