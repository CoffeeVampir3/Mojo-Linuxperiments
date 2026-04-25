from std.memory import UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config, MiniMaxM27ButterQuant, PREFILL_CHUNK_SIZE,
)


comptime TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"
comptime MODEL_DIR = "quantized_models"
comptime MAX_NEW_TOKENS = 32
comptime TP = 4


def load_and_run[
    P: BurstThreadPool, //,
    tp: Int,
](
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = MiniMaxM27ButterQuant[tp, P].load(
        Path(MODEL_DIR), numa_topo, pools^)
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
    var next_id = 0
    var prefilled = 0
    while prefilled < prompt_len:
        var chunk_len = min(PREFILL_CHUNK_SIZE, prompt_len - prefilled)
        for i in range(chunk_len):
            tp_ptr[i] = Scalar[DType.int32](token_ids[prefilled + i])
        var is_last_prefill_chunk = prefilled + chunk_len == prompt_len
        next_id = Int(model.forward(
            Int(tp_ptr), prefilled, chunk_len, is_last_prefill_chunk))
        prefilled += chunk_len
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

    if next_id != MiniMaxM27Config.EOS_TOKEN_ID:
        for step in range(1, MAX_NEW_TOKENS):
            tp_ptr[0] = Scalar[DType.int32](next_id)
            next_id = Int(model.forward(Int(tp_ptr), pos, 1))
            generated.append(next_id)
            pos += 1

            if next_id == MiniMaxM27Config.EOS_TOKEN_ID:
                break

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Float64(0.0)
    if decode_tokens > 0 and decode_elapsed_ms > 0:
        decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
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

    var generated_text = tok.decode(generated)
    var full_text = tok.decode(all_ids)
    print()
    print("generated ids:", end="")
    for i in range(len(generated)):
        print("", generated[i], end="")
    print()
    print("=== generated", len(generated), "tokens ===")
    print(generated_text)
    print()
    print("=== full text ===")
    print(full_text)


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var user_prompt = """prompt  | 42 tokens | 466 ms | 90 t/s
prompt forward profile
  samples:   1
  positions: 0..0
  wall / token
    avg      466.073 ms    stddev     0.000 ms
    p50      466.073 ms    p90      466.073 ms
    p99      466.073 ms    max      466.073 ms
  phase-sum / token
    avg      465.865 ms    overlap-counted   100.0% of wall
  hot path:  moe_phase1 40.8%  moe_phase2 17.1%  attn_proj 14.1%  o_proj 10.4%  kv_write 3.4%
  note: phase timings are local elapsed times; overlapped phases can sum above 100% of wall.
  phases
    phase                vs wall        avg        p99        max | dispatch avg/p99 | kernel avg/p99 | join avg/p99  [ms]
                        --------  ---------  ---------  --------- | ---------------- | -------------- | ------------
    moe_phase1             40.8%    189.984    189.984    189.984 |    0.000/   0.000 |  189.946/ 189.946 |    0.038/   0.038
    moe_phase2             17.1%     79.894     79.894     79.894 |    0.000/   0.000 |   79.819/  79.819 |    0.074/   0.074
    attn_proj              14.1%     65.803     65.803     65.803 |    0.000/   0.000 |   65.722/  65.722 |    0.080/   0.080
    o_proj                 10.4%     48.413     48.413     48.413 |    0.000/   0.000 |   48.357/  48.357 |    0.056/   0.056
    kv_write                3.4%     15.955     15.955     15.955 |    0.000/   0.000 |   15.923/  15.923 |    0.031/   0.031
    q_prep                  2.5%     11.757     11.757     11.757 |    0.000/   0.000 |   11.757/  11.757 |    0.000/   0.000
    attn_reduce             2.1%      9.840      9.840      9.840 |    0.000/   0.000 |    9.840/   9.840 |    0.000/   0.000
    ffn_reduce              2.0%      9.408      9.408      9.408 |    0.000/   0.000 |    9.408/   9.408 |    0.000/   0.000
    dual_norm               1.7%      8.134      8.134      8.134 |    0.027/   0.027 |    8.072/   8.072 |    0.034/   0.034
    attn_quantize           1.6%      7.259      7.259      7.259 |    0.013/   0.013 |    7.205/   7.205 |    0.040/   0.040
    moe_router              1.4%      6.506      6.506      6.506 |    0.000/   0.000 |    6.451/   6.451 |    0.055/   0.055
    lm_head                 0.9%      4.250      4.250      4.250 |    0.000/   0.000 |    4.249/   4.249 |    0.001/   0.001
    attention               0.7%      3.490      3.490      3.490 |    0.000/   0.000 |    3.441/   3.441 |    0.048/   0.048
    norm_prep               0.6%      2.926      2.926      2.926 |    0.000/   0.000 |    2.926/   2.926 |    0.000/   0.000
    moe_route_schedule      0.2%      0.941      0.941      0.941 |    0.000/   0.000 |    0.941/   0.941 |    0.000/   0.000
    moe_route_merge         0.2%      0.923      0.923      0.923 |    0.000/   0.000 |    0.923/   0.923 |    0.000/   0.000
    lm_argmax               0.1%      0.309      0.309      0.309 |    0.000/   0.000 |    0.309/   0.309 |    0.000/   0.000
    omitted tiny phases: broadcast, embed, final_norm, lm_act_bcast
  moe routing
    layer-events: 62  slots: 496  experts/layer: 256  top-k: 8  tp: 4
    max rank load/event: ideal 2.000  avg 3.451  p90 4.000  p99 5.000  max 5.000  active-ranks avg 3.596
    random occupancy baseline: max-load avg 3.512  p90 5  p99 6
    same-rank expert pairs/event: ideal 4.000  random 6.918  avg 6.774  p90 9.000  p99 11.000  max 11.000
    rank slot share: r0 23.0% r1 27.0% r2 25.2% r3 24.8%
    most imbalanced layers
      L 7 rank 1 avg-load 5.000 share 62.5%
      L24 rank 2 avg-load 5.000 share 62.5%
      L27 rank 1 avg-load 5.000 share 62.5%
      L42 rank 2 avg-load 5.000 share 62.5%
      L60 rank 2 avg-load 5.000 share 62.5%
    most concentrated layers
      L 0 top E 13 hit 100.0%  top8-cover 100.0%  max-load 4.000  same-rank-pairs 9.000
      L 1 top E  3 hit 100.0%  top8-cover 100.0%  max-load 4.000  same-rank-pairs 9.000
      L 2 top E 99 hit 100.0%  top8-cover 100.0%  max-load 4.000  same-rank-pairs 8.000
      L 3 top E  8 hit 100.0%  top8-cover 100.0%  max-load 3.000  same-rank-pairs 6.000
      L 4 top E  9 hit 100.0%  top8-cover 100.0%  max-load 3.000  same-rank-pairs 7.000
    hottest experts
      L 0:E 13 rank 0 hits 1/1 (100.0% of layer routes)
      L 0:E 46 rank 0 hits 1/1 (100.0% of layer routes)
      L 0:E 61 rank 0 hits 1/1 (100.0% of layer routes)
      L 0:E 63 rank 0 hits 1/1 (100.0% of layer routes)
      L 0:E129 rank 2 hits 1/1 (100.0% of layer routes)
      L 0:E193 rank 3 hits 1/1 (100.0% of layer routes)
      L 0:E197 rank 3 hits 1/1 (100.0% of layer routes)
      L 0:E206 rank 3 hits 1/1 (100.0% of layer routes)
"""
    var prompt = (
        "]~!b[]~b]system\n"
        + "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
        + "[e~[\n]~b]user\n"
        + user_prompt
        + "[e~[\n]~b]ai\n<think>\n"
    )
    var token_ids = tok.encode(prompt)
    print("prompt:", repr(user_prompt))
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
        load_and_run[TP](numa_topo, pools^, tok, token_ids)
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        load_and_run[TP](numa_topo, pools^, tok, token_ids)
