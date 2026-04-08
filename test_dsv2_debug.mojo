"""Per-layer stats through full DSV2-Lite forward pass."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path

from modeling.deepseekv2_lite import DeepSeekV2Lite, DeepSeekV2LiteConfig
from modeling.model_spec import LogitsView

comptime MODEL_DIR = "checkpoints/deepseekv2-lite"
comptime C = DeepSeekV2LiteConfig
comptime VOCAB = C.VOCAB_SIZE


def bf16_stats(label: String, ptr: Int, n: Int):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=ptr)
    comptime width = simd_width_of[DType.float32]()
    var min_v = Float32(1e30)
    var max_v = Float32(-1e30)
    var sum_sq = Float64(0)
    for i in range(0, n, width):
        var v = (p + i).load[width=width]().cast[DType.float32]()
        for k in range(width):
            if v[k] < min_v: min_v = v[k]
            if v[k] > max_v: max_v = v[k]
            sum_sq += Float64(v[k]) * Float64(v[k])
    print(label, "| var=", Float32(sum_sq / Float64(n)),
          "range=[", min_v, ",", max_v, "]")


def main():
    var model_opt = DeepSeekV2Lite[1].load(Path(MODEL_DIR))
    if not model_opt:
        return
    var model = model_opt.take()

    comptime M = DeepSeekV2Lite[1].M
    var host = model.rank(0)

    # BOS + "The"
    var tp = model.token_buffer()
    tp[0] = Scalar[DType.int32](100000)
    tp[1] = Scalar[DType.int32](549)
    var seq_len = 2

    var logits = model.forward(Int(tp), seq_len, 0)

    # Print x_main after full forward (it holds the final-normed state)
    bf16_stats("final x_main", host.x_main(seq_len).ptr, seq_len * C.HIDDEN)

    # Top-5
    var top_vals = InlineArray[Float32, 5](fill=Float32(-1e30))
    var top_ids = InlineArray[Int, 5](fill=0)
    for j in range(VOCAB):
        var v = logits.load_f32[1](j)
        if v[0] > top_vals[4]:
            top_vals[4] = v[0]
            top_ids[4] = j
            for k in range(3, -1, -1):
                if top_vals[k + 1] > top_vals[k]:
                    var tv = top_vals[k]; top_vals[k] = top_vals[k+1]; top_vals[k+1] = tv
                    var ti = top_ids[k]; top_ids[k] = top_ids[k+1]; top_ids[k+1] = ti
    print("top-5:", end="")
    for i in range(5):
        print(" id=", top_ids[i], "v=", top_vals[i], end="")
    print()

    logits^.release()
    _ = model
