from std.memory import UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe_butterquant_tp import (
    Gemma4Config, Gemma4ButterQuant,
)


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "quantized_models"
comptime VOCAB = Gemma4Config.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 20
comptime TP = 1


def load_and_run[
    P: BurstThreadPool, //,
    tp: Int,
](
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4ButterQuant[tp, P].load(Path(MODEL_DIR), numa, numa_topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    var tp_ptr = model.token_buffer()
    var prompt_len = len(token_ids)
    model.reset_profile()

    var t1 = perf_counter_ns()
    for i in range(prompt_len - 1):
        tp_ptr[0] = Scalar[DType.int32](token_ids[i])
        _ = model.forward_decode(Int(tp_ptr), i)

    tp_ptr[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
    var next_id = Int(model.forward_decode(Int(tp_ptr), prompt_len - 1))
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
        tp_ptr[0] = Scalar[DType.int32](next_id)
        next_id = Int(model.forward_decode(Int(tp_ptr), pos))
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


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var prompt = "The capital of France"
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
    var numa_topo = numa.plan_topology(TP)

    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        load_and_run[TP](numa, numa_topo, pools^, tok, token_ids)
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        load_and_run[TP](numa, numa_topo, pools^, tok, token_ids)
