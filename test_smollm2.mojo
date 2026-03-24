"""Autoregressive generation test: tokenize a prompt, prefill, then
decode tokens greedily. Reports load time, per-token forward time,
and the final generated sequence.

Uses the parametric SmolLM2TP with TP=1."""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from tokenizer import load_tokenizer
from modeling.smollm2_tp import LogitsView, SmolLM2Config, SmolLM2TP
from modeling.model_spec import BF16


comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime VOCAB = SmolLM2Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 512


def greedy_argmax(view: LogitsView[VOCAB]) -> Tuple[Int, Float32]:
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


def main():
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
    print("tokens:", len(token_ids), "ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    # --- Load model (parametric TP=1) ---
    var t0 = perf_counter_ns()
    var model_opt = SmolLM2TP[BF16, 1].load(Path(MODEL_PATH))
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    # --- Write tokens into rank 0's scratch area ---
    var rv = model.rank(0)
    var tokens_addr = rv.scratch_slot(0)
    var tp = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=tokens_addr
    )

    var seq_len = len(token_ids)
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](token_ids[i])

    # --- Prefill: forward the full prompt ---
    print("--- prefill profile ---")
    var t1 = perf_counter_ns()
    var logits = model.forward(tokens_addr, seq_len, 0, profile=True)
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000
    var result = greedy_argmax(logits)
    var next_id = result[0]

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
        tp[0] = Scalar[DType.int32](next_id)

        logits = model.forward(tokens_addr, 1, pos, profile=False)

        result = greedy_argmax(logits)
        next_id = result[0]
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

    # --- Final output ---
    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    print("=== generated", MAX_NEW_TOKENS, "tokens ===")
    print(full_text)
