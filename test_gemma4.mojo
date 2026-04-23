from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaInfo, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe import Gemma4Config, Gemma4
from modeling.model_spec import LogitsView


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime VOCAB = Gemma4Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 128


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


def load_and_run[
    P: BurstThreadPool, //,
](
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pool: P,
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4[1, P].load(Path(MODEL_DIR), numa, numa_topo, pool^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    var tp_ptr = model.token_buffer()
    var prompt_len = len(token_ids)

    var t1 = perf_counter_ns()
    for i in range(prompt_len - 1):
        tp_ptr[0] = Scalar[DType.int32](token_ids[i])
        var logits = model.forward(Int(tp_ptr), 1, i)
        logits^.release()

    tp_ptr[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
    var logits = model.forward(Int(tp_ptr), 1, prompt_len - 1)
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000

    var top_vals = InlineArray[Float32, 5](fill=Float32(-1e30))
    var top_ids = InlineArray[Int, 5](fill=0)
    for j in range(VOCAB):
        var v = logits.load_f32[1](j)
        if v[0] > top_vals[4]:
            top_vals[4] = v[0]
            top_ids[4] = j
            for k in range(3, -1, -1):
                if top_vals[k + 1] > top_vals[k]:
                    var tv = top_vals[k]; top_vals[k] = top_vals[k + 1]; top_vals[k + 1] = tv
                    var ti = top_ids[k]; top_ids[k] = top_ids[k + 1]; top_ids[k + 1] = ti
    print("top-5 logits after prompt:")
    for i in range(5):
        var id_list = List[Int]()
        id_list.append(top_ids[i])
        print(" ", i, "id=", top_ids[i], "val=", top_vals[i], "tok=", repr(tok.decode(id_list)))

    var result = greedy_argmax(logits)
    var next_id = result[0]
    logits^.release()

    var generated = List[Int]()
    generated.append(next_id)

    var prefill_tps = Float64(prompt_len) / (Float64(prefill_ms) / 1000.0)
    print(
        "prompt  |", prompt_len, "tokens |",
        prefill_ms, "ms |",
        Int(prefill_tps), "t/s",
    )

    var pos = prompt_len
    var decode_start = perf_counter_ns()

    for step in range(1, MAX_NEW_TOKENS):
        tp_ptr[0] = Scalar[DType.int32](next_id)

        logits = model.forward(Int(tp_ptr), 1, pos)

        result = greedy_argmax(logits)
        next_id = result[0]
        logits^.release()
        generated.append(next_id)
        pos += 1

        if next_id == 1:
            break

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
    print(
        "decode  |", decode_tokens, "tokens |",
        decode_elapsed_ms, "ms |",
        Int(decode_tps), "t/s",
    )

    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    print("=== generated", len(generated), "tokens ===")
    print(full_text)


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var prompt = "The capital of france is"
    var token_ids = List[Int]()
    token_ids.append(2)  # <bos>
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    print("prompt:", repr(prompt))
    print("tokens:", len(token_ids), "ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(1)

    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pool = IsolatedBurstPool[].for_topology(numa, numa_topo[0])
        load_and_run(numa, numa_topo, pool^, tok, token_ids)
    else:
        print("mode: cold (spin-backoff)")
        var pool = BurstPool[].for_topology(numa, numa_topo[0])
        load_and_run(numa, numa_topo, pool^, tok, token_ids)
