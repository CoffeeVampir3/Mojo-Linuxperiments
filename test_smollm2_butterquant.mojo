"""Autoregressive generation test for ButterQuant int8 model.

Tokenize a prompt, prefill, then decode greedily.
Reports load time, prefill/decode throughput, and generated text.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from tokenizer import load_tokenizer
from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant, SmolLM2Config
from modeling.model_spec import LogitsView
from experimental.amx import init_intel_amx

comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime MODEL_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"
comptime VOCAB = SmolLM2Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 512


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


def main():
    _ = init_intel_amx()

    # --- Load tokenizer ---
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("Failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    # --- Encode prompt ---
    var prompt = """Constantinos Daskalakis (Greek: Κωνσταντίνος Δασκαλάκης; born 29 April 1981) is a Greek theoretical computer scientist.[2] He is a professor at MIT's Electrical Engineering and Computer Science department and a member of the MIT Computer Science and Artificial Intelligence Laboratory.[3][4][5] He was awarded the Rolf Nevanlinna Prize and the Grace Murray Hopper Award in 2018.
Education and career

Daskalakis was born in Athens on 29 April 1981.[6] His grandparents originated from Crete, where he summered as a child. His parents were high school teachers of mathematics and literature.[7][8] He has a younger brother, Nikolaos Daskalakis, who is a neuroscientist and Boston University professor.[9][10] When Daskalakis was in third grade, his father bought an Amstrad CPC, which Daskalakis stayed up all night with, attempting to learn how it worked.[11]

He attended Varvakeio High School and received a Diploma in Electrical and Computer Engineering from the National Technical University of Athens in 2004, completing an undergraduate thesis supervised by Stathis Zachos. As an undergraduate, Daskalakis attained perfect scores in all but one of his classes, something which had not previously been achieved in the university's history.[11] He received a PhD in computer science from the University of California, Berkeley advised by Christos Papadimitriou.[12][1]

From 2008 to 2009, Daskalakis was a postdoctoral researcher at Microsoft Research mentored by Jennifer Chayes. He joined MIT in 2009 and was given tenure in 2015.[13]

He is a co-founder and chief scientist of Archimedes AI research center.[citation needed]
Research

Daskalakis works on the theory of computation and its interface with game theory, economics, probability theory, statistics and machine learning.[2] He is known for work on the computational complexity of Nash equilibria, the complexity of multi-item auctions, and the behavior of the expectation–maximization algorithm. He has worked on efficient methods for statistical hypothesis testing and learning in high dimensions, as well as concentration properties of high-dimensional distributions.
Awards and honors

Constantinos Daskalakis was awarded the 2008 ACM Doctoral Dissertation Award for "advancing our understanding of behavior in complex networks of interacting individuals."[14] He later co-authored the paper The Complexity of Computing a Nash Equilibrium[15] based on the same work with Christos Papadimitriou and Paul W. Goldberg, for which they were awarded the 2008 Kalai Game Theory and Computer Science Prize.[16]

In 2018, Daskalakis was awarded the Nevanlinna Prize for "transforming our understanding of the computational complexity of fundamental problems in markets, auctions, equilibria and other economic structures."[17] In the same year, he also received the Simons Foundation Investigator award in theoretical computer science.[18]

He was named to the 2022 class of ACM Fellows"""
    var token_ids = tok.encode(prompt)
    print("prompt:", repr(prompt))
    print("tokens:", len(token_ids))

    # --- Load model ---
    var t0 = perf_counter_ns()
    var model_opt = SmolLM2ButterQuant[3].load(Path(MODEL_PATH))
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print("layer 0 V scale:", model.layer_scales[0].v_layer_scale)
    print("(all other scales are dynamic per-row, computed at runtime)")
    print()

    # --- Write tokens ---
    var token_buf = model.token_buffer()
    var seq_len = len(token_ids)
    for i in range(seq_len):
        token_buf[i] = Scalar[DType.int32](token_ids[i])

    # --- Prefill ---
    var t1 = perf_counter_ns()
    var logits = model.forward(Int(token_buf), seq_len, 0)
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000

    # Top-5 logits diagnostic
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
    print("top-5 logits after prefill:")
    for i in range(5):
        var id_list = List[Int]()
        id_list.append(top_ids[i])
        print(" ", i, "id=", top_ids[i], "val=", top_vals[i], "tok=", repr(tok.decode(id_list)))

    var result = greedy_argmax(logits)
    var next_id = result[0]
    logits^.release()

    var generated = List[Int]()
    generated.append(next_id)

    var prefill_tps = Float64(seq_len) / (Float64(prefill_ms) / 1000.0)
    print(
        "prefill |", seq_len, "tokens |",
        prefill_ms, "ms |",
        Int(prefill_tps), "t/s",
    )

    # --- Decode ---
    var pos = seq_len
    var decode_start = perf_counter_ns()

    for step in range(1, MAX_NEW_TOKENS):
        token_buf[0] = Scalar[DType.int32](next_id)
        logits = model.forward(Int(token_buf), 1, pos)
        result = greedy_argmax(logits)
        next_id = result[0]
        logits^.release()
        generated.append(next_id)
        pos += 1

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = MAX_NEW_TOKENS - 1
    var decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
    print(
        "decode  |", decode_tokens, "tokens |",
        decode_elapsed_ms, "ms |",
        Int(decode_tps), "t/s",
    )

    # --- Profile report (averaged over all decode steps) ---
    print()
    model.report_profile()

    # --- Final output ---
    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    print(all_ids)
    var full_text = tok.decode(all_ids)
    print("\n=== generated", MAX_NEW_TOKENS, "tokens ===")
    print(full_text)

    _ = model
