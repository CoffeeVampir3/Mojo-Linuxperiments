"""Quick DSV2-Lite inference test.

Tests both our tokenization and HF's (space-dropped) token IDs
on a simple prompt to determine which the model expects.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from tokenizer import load_tokenizer
from tokenizer.loader import load_tokenizer as load_tok
from modeling.deepseekv2_lite import DeepSeekV2Lite, DeepSeekV2LiteConfig
from modeling.model_spec import LogitsView
from tokenizer.tokenizer import BPETokenizer
from tokenizer.loader import AutoPreTokenizer
from tokenizer.gpt2 import GPT2ByteTransform


comptime TOKENIZER_PATH = "checkpoints/deepseekv2-lite/tokenizer.json"
comptime MODEL_DIR = "checkpoints/deepseekv2-lite"
comptime C = DeepSeekV2LiteConfig
comptime VOCAB = C.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 15

comptime Tok = BPETokenizer[AutoPreTokenizer, GPT2ByteTransform]


def greedy_argmax(read view: LogitsView[VOCAB]) -> Tuple[Int, Float32]:
    comptime width = simd_width_of[DType.float32]()
    var best_val = Float32(-1e30)
    var best_idx = 0
    for j in range(0, VOCAB, width):
        var v = view.load_f32[width](j)
        for k in range(width):
            if v[k] > best_val:
                best_val = v[k]
                best_idx = j + k
    return (best_idx, best_val)


def run_test(
    mut model: DeepSeekV2Lite[1],
    mut tok: Tok,
    label: String,
    ids: List[Int],
):
    var tp = model.token_buffer()
    var seq_len = len(ids)
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](ids[i])

    print("--- " + label + " ---")
    print("ids:", end="")
    for i in range(seq_len):
        print("", ids[i], end="")
    print()

    var t0 = perf_counter_ns()
    var logits = model.forward(Int(tp), seq_len, 0)
    var ms = (perf_counter_ns() - t0) / 1_000_000
    print("prefill:", ms, "ms")

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
    print("top-5 logits:")
    for i in range(5):
        var id_list = List[Int]()
        id_list.append(top_ids[i])
        print(" ", i, "id=", top_ids[i], "val=", top_vals[i], "tok=", repr(tok.decode(id_list)))

    # Greedy decode
    var result = greedy_argmax(logits)
    var next_id = result[0]
    logits^.release()

    var generated = List[Int]()
    generated.append(next_id)

    for step in range(1, MAX_NEW_TOKENS):
        tp[0] = Scalar[DType.int32](next_id)
        logits = model.forward(Int(tp), 1, seq_len + step - 1)
        result = greedy_argmax(logits)
        next_id = result[0]
        logits^.release()
        generated.append(next_id)

    print("generated ids:", end="")
    for i in range(len(generated)):
        print("", generated[i], end="")
    print()
    print("generated:", repr(tok.decode(generated)))
    print()


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("tokenizer failed")
        return
    var tok = tok_opt.take()

    var model_opt = DeepSeekV2Lite[1].load(Path(MODEL_DIR))
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()

    # Test A: our tokenizer (Ġ-prefixed, with BOS)
    var our_ids = tok.encode("The capital of France is")
    run_test(model, tok, "Our tokenizer (Ġ-prefixed)", our_ids)

    # Test B: HF token IDs (space-dropped, no BOS)
    var hf_ids = List[Int]()
    hf_ids.append(549)    # The
    hf_ids.append(42394)  # capital
    hf_ids.append(994)    # of
    hf_ids.append(36715)  # France
    hf_ids.append(262)    # is
    run_test(model, tok, "HF tokenizer (bare, no BOS)", hf_ids)

    # Test C: HF token IDs with BOS prepended
    var hf_bos_ids = List[Int]()
    hf_bos_ids.append(100000)  # BOS
    hf_bos_ids.append(549)
    hf_bos_ids.append(42394)
    hf_bos_ids.append(994)
    hf_bos_ids.append(36715)
    hf_bos_ids.append(262)
    run_test(model, tok, "HF tokenizer (bare, with BOS)", hf_bos_ids)

    _ = model
