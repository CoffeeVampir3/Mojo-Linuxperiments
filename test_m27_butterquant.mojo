"""Autoregressive generation test for MiniMax M2.7 ButterQuant (int8).

Loads model + tokenizer, processes prompt one token at a time, picks the next
token by greedy argmax on the LM-head logits (no softcap), reports timing and
generated text.

Run remote only (too large for local):   fish remote_build.fish test_m27_butterquant.mojo
"""

from std.memory import UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from tokenizer import load_tokenizer
from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config, MiniMaxM27ButterQuant,
)


comptime TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"
comptime MODEL_DIR = "quantized_models"
comptime VOCAB = MiniMaxM27Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 1
comptime TP = 4


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var prompt = "The capital of France"
    var token_ids = List[Int]()
    token_ids.append(1)  # <bos> — verify against tokenizer config
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    print("prompt:", repr(prompt))
    print("tokens:", len(token_ids), "ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    var t0 = perf_counter_ns()
    var model_opt = MiniMaxM27ButterQuant[TP].load(Path(MODEL_DIR))
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    var tp = model.token_buffer()
    var prompt_len = len(token_ids)
    model.reset_profile()

    var t1 = perf_counter_ns()
    for i in range(prompt_len - 1):
        tp[0] = Scalar[DType.int32](token_ids[i])
        _ = model.forward_decode(Int(tp), i)

    tp[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
    var next_id = Int(model.forward_decode(Int(tp), prompt_len - 1))
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000

    var generated = List[Int]()
    generated.append(next_id)

    print(
        "prompt  |", prompt_len, "tokens |",
        prefill_ms, "ms |",
        Int(Float64(prompt_len) / (Float64(prefill_ms) / 1000.0)), "t/s",
    )
    model.report_profile("prompt")
    model.reset_profile()

    var pos = prompt_len
    var decode_start = perf_counter_ns()

    for step in range(1, MAX_NEW_TOKENS):
        tp[0] = Scalar[DType.int32](next_id)
        next_id = Int(model.forward_decode(Int(tp), pos))
        generated.append(next_id)
        pos += 1

        if next_id == 1 or next_id == 2:
            break

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
    print(
        "decode  |", decode_tokens, "tokens |",
        decode_elapsed_ms, "ms |",
        Int(decode_tps), "t/s",
    )
    model.report_profile("decode")

    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    print("=== generated", len(generated), "tokens ===")
    print(full_text)
