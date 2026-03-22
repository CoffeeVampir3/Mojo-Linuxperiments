"""Autoregressive generation test: tokenize a prompt, prefill, then
decode 30 tokens greedily. Reports load time, per-token forward time,
and the final generated sequence."""

from memory import UnsafePointer
from sys.info import simd_width_of
from pathlib import Path
from time import perf_counter_ns

from tokenizer import load_tokenizer
from experimental4.smollm2 import LogitsView, SmolLM2Loaded, SmolLM2Config, MODEL_PATH, BF16


comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime VOCAB = SmolLM2Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 30


fn greedy_argmax(view: LogitsView[VOCAB]) -> Tuple[Int, Float32]:
    """Greedy decode: return (token_id, logit) with highest value in last row."""
    comptime width = simd_width_of[DType.float32]()
    var last = view.rows() - 1
    var best_val = Float32(-1e30)
    var best_idx = 0

    for j in range(0, VOCAB, width):
        var v = view.load_f32[width](last, j)
        for k in range(width):
            if v[k] > best_val:
                best_val = v[k]
                best_idx = j + k

    return (best_idx, best_val)


fn main():
    # --- Load tokenizer ---
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("Failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    # --- Encode prompt ---
    var prompt = "The capital of France is"
    var token_ids = tok.encode(prompt)
    print("prompt:", repr(prompt))
    print("tokens:", len(token_ids), "ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    # --- Load model ---
    var t0 = perf_counter_ns()
    var model_opt = SmolLM2Loaded[BF16, 1].load(Path(MODEL_PATH), 0)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    # --- Prefill: forward the full prompt ---
    var seq_len = len(token_ids)
    var tp = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=model.scratch_ptr()
    )
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](token_ids[i])

    print("--- prefill profile ---")
    var t1 = perf_counter_ns()
    var logits = model.forward(model.scratch_ptr(), seq_len, 0, profile=True)
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000
    var result = greedy_argmax(logits)
    var next_id = result[0]

    var generated = List[Int]()
    generated.append(next_id)

    var one_tok = List[Int]()
    one_tok.append(next_id)
    var decoded_tok = tok.decode(one_tok)
    print(
        "prefill |", seq_len, "tokens |",
        prefill_ms, "ms | next:", next_id,
        repr(decoded_tok),
    )

    # --- Decode: generate 29 more tokens one at a time ---
    var pos = seq_len  # KV cache already has [0, seq_len)

    for step in range(1, MAX_NEW_TOKENS):
        # Write the single token into scratch
        tp[0] = Scalar[DType.int32](next_id)

        # Profile the first decode step
        var do_profile = step == 1
        if do_profile:
            print("\n--- decode step 1 profile ---")

        var t2 = perf_counter_ns()
        logits = model.forward(model.scratch_ptr(), 1, pos, profile=do_profile)
        var decode_ms = (perf_counter_ns() - t2) / 1_000_000

        result = greedy_argmax(logits)
        next_id = result[0]
        generated.append(next_id)
        pos += 1

        one_tok[0] = next_id
        decoded_tok = tok.decode(one_tok)
        print(
            "step", step, "|",
            decode_ms, "ms | next:", next_id,
            repr(decoded_tok),
        )

    # --- Final output ---
    # Combine prompt tokens + generated tokens for full decode
    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    print("=== generated", MAX_NEW_TOKENS, "tokens ===")
    print(full_text)
