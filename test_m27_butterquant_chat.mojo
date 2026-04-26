from std.memory import Span
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
comptime MAX_NEW_TOKENS = 1024
comptime TP = 4
comptime SYSTEM_PROMPT = (
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
)


def tokens_per_second(tokens: Int, elapsed_ms: UInt) -> Int:
    if tokens <= 0 or elapsed_ms == 0:
        return 0
    return Int(Float64(tokens) / (Float64(elapsed_ms) / 1000.0))


def elapsed_ms_since(start_ns: UInt) -> UInt:
    return (perf_counter_ns() - start_ns) / 1_000_000


def chat_preamble() -> String:
    return "]~!b[]~b]system\n" + SYSTEM_PROMPT + "[e~[\n"


def byte_substring(text: String, start: Int, end: Int) -> String:
    if end <= start:
        return String("")
    var data = text.as_bytes()
    return String(
        unsafe_from_utf8=Span[Byte, _](
            ptr=data.unsafe_ptr() + start,
            length=end - start,
        )
    )


def strip_newlines(text: String, start: Int, end: Int) -> String:
    var data = text.as_bytes()
    var start_idx = start
    var end_idx = end
    while start_idx < end_idx and data[start_idx] == Byte(10):
        start_idx += 1
    while end_idx > start_idx and data[end_idx - 1] == Byte(10):
        end_idx -= 1
    return byte_substring(text, start_idx, end_idx)


def assistant_content_for_history(assistant_text: String) -> String:
    var think_end = assistant_text.rfind("</think>")
    if think_end < 0:
        return assistant_text
    return strip_newlines(
        assistant_text,
        think_end + String("</think>").byte_length(),
        assistant_text.byte_length(),
    )


def wake_model_workers[P: BurstThreadPool, //, tp: Int](
    mut model: MiniMaxM27ButterQuant[tp, P],
):
    for rank in range(tp):
        model.main_pools[rank].wake()


def sleep_model_workers[P: BurstThreadPool, //, tp: Int](
    mut model: MiniMaxM27ButterQuant[tp, P],
):
    for rank in range(tp):
        model.main_pools[rank].sleep()


@always_inline
def is_utf8_continuation_byte(b: Byte) -> Bool:
    return b >= Byte(0x80) and b <= Byte(0xBF)


def complete_utf8_prefix_len(bytes: List[Byte]) -> Int:
    var i = 0
    var n = len(bytes)
    while i < n:
        var b0 = bytes[i]
        var width: Int
        if b0 < Byte(0x80):
            width = 1
        elif b0 >= Byte(0xC2) and b0 <= Byte(0xDF):
            width = 2
        elif b0 >= Byte(0xE0) and b0 <= Byte(0xEF):
            width = 3
        elif b0 >= Byte(0xF0) and b0 <= Byte(0xF4):
            width = 4
        else:
            break

        if i + width > n:
            break

        if width >= 2 and not is_utf8_continuation_byte(bytes[i + 1]):
            break
        if width >= 3 and not is_utf8_continuation_byte(bytes[i + 2]):
            break
        if width >= 4 and not is_utf8_continuation_byte(bytes[i + 3]):
            break

        if width == 3:
            if b0 == Byte(0xE0) and bytes[i + 1] < Byte(0xA0):
                break
            if b0 == Byte(0xED) and bytes[i + 1] > Byte(0x9F):
                break
        elif width == 4:
            if b0 == Byte(0xF0) and bytes[i + 1] < Byte(0x90):
                break
            if b0 == Byte(0xF4) and bytes[i + 1] > Byte(0x8F):
                break

        i += width

    return i


struct GraphemeStream:
    var pending_bytes: List[Byte]
    var pending_text: String

    def __init__(out self):
        self.pending_bytes = List[Byte]()
        self.pending_text = String("")

    def push_bytes(mut self, bytes: List[Byte]):
        for i in range(len(bytes)):
            self.pending_bytes.append(bytes[i])

        var valid_len = complete_utf8_prefix_len(self.pending_bytes)
        if valid_len == 0:
            return

        var chunk = String(
            unsafe_from_utf8=Span[Byte, _](
                ptr=self.pending_bytes.unsafe_ptr(),
                length=valid_len,
            )
        )
        self.push_text(chunk^)

        var rest = List[Byte](capacity=len(self.pending_bytes) - valid_len)
        for i in range(valid_len, len(self.pending_bytes)):
            rest.append(self.pending_bytes[i])
        self.pending_bytes = rest^

    def push_text(mut self, text: String):
        if text.byte_length() == 0:
            return
        self.pending_text += text
        self.flush_graphemes(False)

    def flush_graphemes(mut self, force: Bool):
        if self.pending_text.byte_length() == 0:
            return

        var graphemes = List[String]()
        for grapheme in self.pending_text.graphemes():
            graphemes.append(String(grapheme))

        var count = len(graphemes)
        if count == 0:
            return

        var emit_count = count
        if not force:
            emit_count -= 1

        if emit_count > 0:
            var out = String("")
            for i in range(emit_count):
                out += graphemes[i]
            print(out, end="", flush=True)

        if force:
            self.pending_text = String("")
        else:
            self.pending_text = graphemes[count - 1]

    def finish(mut self):
        var valid_len = complete_utf8_prefix_len(self.pending_bytes)
        if valid_len > 0:
            var chunk = String(
                unsafe_from_utf8=Span[Byte, _](
                    ptr=self.pending_bytes.unsafe_ptr(),
                    length=valid_len,
                )
            )
            self.pending_text += chunk
            self.pending_bytes.resize(unsafe_uninit_length=0)

        self.flush_graphemes(True)


def stream_token(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut stream: GraphemeStream,
    token_id: Int,
):
    var token = tok.id_to_token(token_id)
    if not token:
        return

    var token_text = token.value()
    if tok.is_special_id(token_id):
        stream.push_text(token_text^)
        return

    var raw_bytes = tok.byte_transform.decode_bytes(token_text^)
    stream.push_bytes(raw_bytes)


struct HistoryCache:
    # Tracks the token prefix that is physically represented in the model KV
    # cache. A sampled token is not committed here until a later forward call
    # consumes that token as input and writes its K/V slot.
    var kv_committed_tokens: List[Int]

    def __init__(out self):
        self.kv_committed_tokens = List[Int]()

    def committed_len(self) -> Int:
        return len(self.kv_committed_tokens)

    def last_common_prefix(self, candidate: List[Int]) -> Int:
        var common = 0
        var limit = min(len(self.kv_committed_tokens), len(candidate))
        while common < limit:
            if self.kv_committed_tokens[common] != candidate[common]:
                break
            common += 1
        return common

    def replace_with(mut self, tokens: List[Int]):
        self.kv_committed_tokens = List[Int](capacity=len(tokens))
        for i in range(len(tokens)):
            self.kv_committed_tokens.append(tokens[i])

    def append_committed(mut self, token_id: Int):
        self.kv_committed_tokens.append(token_id)


def load_and_chat[
    P: BurstThreadPool, //,
    tp: Int,
](
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
    var tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    startup_start_ns: UInt,
    tokenizer_ms: UInt,
    numa_ms: UInt,
    pool_ms: UInt,
) raises:
    var t0 = perf_counter_ns()
    var model_opt = MiniMaxM27ButterQuant[tp, P].load(
        Path(MODEL_DIR), numa_topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var model_ms = elapsed_ms_since(t0)
    var ready_ms = elapsed_ms_since(startup_start_ns)
    print(
        "startup | tokenizer", tokenizer_ms, "ms |",
        "numa", numa_ms, "ms |",
        "pools", pool_ms, "ms |",
        "230GB weights load + initialize", model_ms, "ms |",
        "ready", ready_ms, "ms",
    )
    print()

    var tp_ptr = model.token_buffer()
    var history = chat_preamble()
    var cache = HistoryCache()
    print("MiniMax-M2.7 chat. Type /quit, quit, or exit to stop.")
    print()
    sleep_model_workers(model)

    while True:
        var user_text: String
        try:
            user_text = input("you> ")
        except:
            print()
            break

        if user_text == "/quit" or user_text == "quit" or user_text == "exit":
            break
        if user_text.byte_length() == 0:
            continue

        var user_block = "]~b]user\n" + user_text + "[e~[\n"
        var prompt = history + user_block + "]~b]ai\n<think>\n"
        var token_ids = tok.encode(prompt)
        var prompt_len = len(token_ids)
        if prompt_len >= MiniMaxM27Config.MAX_SEQ_LEN:
            print("context is full; restart the program for a fresh chat")
            continue

        wake_model_workers(model)
        var max_new_tokens = min(
            MAX_NEW_TOKENS, MiniMaxM27Config.MAX_SEQ_LEN - prompt_len)
        model.reset_profile()

        var reuse_tokens = cache.last_common_prefix(token_ids)
        var prefill_tokens = prompt_len - reuse_tokens
        var t1 = perf_counter_ns()
        var next_id = 0
        var prefilled = reuse_tokens
        while prefilled < prompt_len:
            var chunk_len = min(PREFILL_CHUNK_SIZE, prompt_len - prefilled)
            for i in range(chunk_len):
                tp_ptr[i] = Scalar[DType.int32](token_ids[prefilled + i])
            var is_last_prefill_chunk = prefilled + chunk_len == prompt_len
            next_id = Int(model.forward(
                Int(tp_ptr), prefilled, chunk_len, is_last_prefill_chunk))
            prefilled += chunk_len

        if prefill_tokens == 0:
            # KV contains the full prompt, but logits for the next token are not
            # cached. Re-run the final prompt token in-place to produce logits.
            tp_ptr[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
            next_id = Int(model.forward(Int(tp_ptr), prompt_len - 1, 1, True))

        cache.replace_with(token_ids)
        var prefill_ms = (perf_counter_ns() - t1) / 1_000_000
        var prefill_work_tokens = prefill_tokens
        if prefill_work_tokens == 0:
            prefill_work_tokens = 1

        print(
            "prefill |", prompt_len, "prompt tokens |",
            reuse_tokens, "reused |",
            prefill_tokens, "new |",
            prefill_work_tokens, "work tokens |",
            prefill_ms, "ms |",
            tokens_per_second(prefill_work_tokens, prefill_ms), "t/s",
        )
        model.reset_profile()

        var generated = List[Int]()
        var stream = GraphemeStream()
        print("assistant>")
        if next_id != MiniMaxM27Config.EOS_TOKEN_ID:
            generated.append(next_id)
            stream_token(tok, stream, next_id)

        var pos = prompt_len
        var decode_start = perf_counter_ns()
        var step = 1
        while (
            step < max_new_tokens
            and next_id != MiniMaxM27Config.EOS_TOKEN_ID
        ):
            var committed_id = next_id
            tp_ptr[0] = Scalar[DType.int32](committed_id)
            next_id = Int(model.forward(Int(tp_ptr), pos, 1))
            cache.append_committed(committed_id)
            pos += 1
            step += 1

            if next_id == MiniMaxM27Config.EOS_TOKEN_ID:
                break
            generated.append(next_id)
            stream_token(tok, stream, next_id)

        var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
        var decode_tokens = len(generated) - 1
        if decode_tokens < 0:
            decode_tokens = 0
        stream.finish()
        print()
        print(
            "decode  |", decode_tokens, "tokens |",
            decode_elapsed_ms, "ms |",
            tokens_per_second(decode_tokens, decode_elapsed_ms), "t/s",
        )
        model.reset_profile()

        var assistant_text = tok.decode(generated)
        print()

        var assistant_content = assistant_content_for_history(assistant_text)
        history = history + user_block + "]~b]ai\n" + assistant_content + "[e~[\n"
        sleep_model_workers(model)


def main() raises:
    var startup_start_ns = perf_counter_ns()

    var tokenizer_start_ns = perf_counter_ns()
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    var tokenizer_ms = elapsed_ms_since(tokenizer_start_ns)
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var numa_start_ns = perf_counter_ns()
    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)
    var numa_ms = elapsed_ms_since(numa_start_ns)

    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (explicit sleep)")
        var pool_start_ns = perf_counter_ns()
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        var pool_ms = elapsed_ms_since(pool_start_ns)
        load_and_chat[TP](
            numa_topo, pools^, tok^,
            startup_start_ns, tokenizer_ms, numa_ms, pool_ms,
        )
    else:
        print("mode: cold (spin-backoff)")
        var pool_start_ns = perf_counter_ns()
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        var pool_ms = elapsed_ms_since(pool_start_ns)
        load_and_chat[TP](
            numa_topo, pools^, tok^,
            startup_start_ns, tokenizer_ms, numa_ms, pool_ms,
        )
