from std.memory import Span
from std.pathlib import Path
from std.python import PythonObject

from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config,
    MiniMaxM27ButterQuant,
    PREFILL_CHUNK_SIZE,
)
from notstdcollections import HeapMoveArray
from numa import NumaInfo
from threading.isolated_burst_pool import IsolatedBurstPool
from tokenizer import (
    AutoByteTransform,
    AutoPreTokenizer,
    BPETokenizer,
    load_tokenizer,
)


comptime TP = 4
comptime DEFAULT_TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"
comptime DEFAULT_MODEL_DIR = "quantized_models"
comptime DEFAULT_MAX_NEW_TOKENS = 1024


def chat_preamble(system_prompt: String) -> String:
    return "]~!b[]~b]system\n" + system_prompt + "[e~[\n"


def user_block(user_text: String) -> String:
    return "]~b]user\n" + user_text + "[e~[\n"


def assistant_prefix() -> String:
    return "]~b]ai\n<think>\n"


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


struct StreamTextDecoder(Movable):
    var pending_bytes: List[Byte]
    var pending_text: String

    def __init__(out self):
        self.pending_bytes = List[Byte]()
        self.pending_text = String("")

    def push_bytes(mut self, bytes: List[Byte]) -> String:
        for i in range(len(bytes)):
            self.pending_bytes.append(bytes[i])

        var valid_len = complete_utf8_prefix_len(self.pending_bytes)
        if valid_len == 0:
            return String("")

        var chunk = String(
            unsafe_from_utf8=Span[Byte, _](
                ptr=self.pending_bytes.unsafe_ptr(),
                length=valid_len,
            )
        )
        var out = self.push_text(chunk^)

        var rest = List[Byte](capacity=len(self.pending_bytes) - valid_len)
        for i in range(valid_len, len(self.pending_bytes)):
            rest.append(self.pending_bytes[i])
        self.pending_bytes = rest^
        return out^

    def push_text(mut self, text: String) -> String:
        if text.byte_length() == 0:
            return String("")
        self.pending_text += text
        return self.flush_graphemes(False)

    def flush_graphemes(mut self, force: Bool) -> String:
        if self.pending_text.byte_length() == 0:
            return String("")

        var graphemes = List[String]()
        for grapheme in self.pending_text.graphemes():
            graphemes.append(String(grapheme))

        var count = len(graphemes)
        if count == 0:
            return String("")

        var emit_count = count
        if not force:
            emit_count -= 1

        var out = String("")
        for i in range(emit_count):
            out += graphemes[i]

        if force:
            self.pending_text = String("")
        else:
            self.pending_text = graphemes[count - 1]

        return out^

    def finish(mut self) -> String:
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

        return self.flush_graphemes(True)


def stream_token_text(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut decoder: StreamTextDecoder,
    token_id: Int,
) -> String:
    var token = tok.id_to_token(token_id)
    if not token:
        return String("")

    var token_text = token.value()
    if tok.is_special_id(token_id):
        return decoder.push_text(token_text^)

    var raw_bytes = tok.byte_transform.decode_bytes(token_text^)
    return decoder.push_bytes(raw_bytes)


struct HistoryCache(Movable):
    # This is the exact token prefix that has been written into the physical KV
    # cache. The token most recently sampled from logits is not committed until
    # a later forward call consumes it as input and writes its K/V slot.
    var kv_committed_tokens: List[Int]

    def __init__(out self):
        self.kv_committed_tokens = List[Int]()

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


struct M27Session(Movable, Writable):
    var tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform]
    var model: MiniMaxM27ButterQuant[TP, IsolatedBurstPool[]]
    var system_prompt: String
    var history: String
    var cache: HistoryCache
    var stream_active: Bool
    var stream_pos: Int
    var stream_step: Int
    var stream_limit: Int
    var stream_next_id: Int
    var stream_user_block: String
    var stream_generated: List[Int]
    var stream_decoder: StreamTextDecoder

    def __init__(
        out self,
        system_prompt: String,
        tokenizer_path: String = String(DEFAULT_TOKENIZER_PATH),
        model_dir: String = String(DEFAULT_MODEL_DIR),
    ) raises:
        var tok_opt = load_tokenizer(Path(tokenizer_path))
        if not tok_opt:
            raise Error("failed to load tokenizer from " + tokenizer_path)

        var numa = NumaInfo()
        if not numa.has_isolation():
            raise Error("M27Session currently requires isolated CPU workers")

        var numa_topo = numa.plan_topology(TP)
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))

        var model_opt = MiniMaxM27ButterQuant[
            TP, IsolatedBurstPool[]
        ].load(Path(model_dir), numa_topo, pools^)
        if not model_opt:
            raise Error("failed to load MiniMax-M2.7 model from " + model_dir)

        self.tok = tok_opt.take()
        self.model = model_opt.take()
        self.system_prompt = system_prompt
        self.history = chat_preamble(self.system_prompt)
        self.cache = HistoryCache()
        self.stream_active = False
        self.stream_pos = 0
        self.stream_step = 0
        self.stream_limit = 0
        self.stream_next_id = MiniMaxM27Config.EOS_TOKEN_ID
        self.stream_user_block = String("")
        self.stream_generated = List[Int]()
        self.stream_decoder = StreamTextDecoder()
        self.sleep_workers()

    @staticmethod
    def py_init(
        out self: M27Session, args: PythonObject, kwargs: PythonObject
    ) raises:
        # The Python binding layer passes a null kwargs object when no keyword
        # arguments are present, so this POC constructor is positional-only and
        # intentionally does not inspect kwargs.
        if len(args) > 3:
            raise Error(
                "M27Session(system_prompt, tokenizer_path?, model_dir?)"
            )

        var system_prompt = String(
            "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
        )
        var tokenizer_path = String(DEFAULT_TOKENIZER_PATH)
        var model_dir = String(DEFAULT_MODEL_DIR)

        if len(args) >= 1:
            system_prompt = String(args[0])
        if len(args) >= 2:
            tokenizer_path = String(args[1])
        if len(args) >= 3:
            model_dir = String(args[2])

        self = Self(system_prompt^, tokenizer_path^, model_dir^)

    def reset(mut self, system_prompt: String):
        self.system_prompt = system_prompt
        self.history = chat_preamble(self.system_prompt)
        self.cache = HistoryCache()
        self.stream_active = False
        self.stream_pos = 0
        self.stream_step = 0
        self.stream_limit = 0
        self.stream_next_id = MiniMaxM27Config.EOS_TOKEN_ID
        self.stream_user_block = String("")
        self.stream_generated = List[Int]()
        self.stream_decoder = StreamTextDecoder()
        self.sleep_workers()

    def wake_workers(mut self):
        for rank in range(TP):
            self.model.main_pools[rank].wake()

    def sleep_workers(mut self):
        for rank in range(TP):
            self.model.main_pools[rank].sleep()

    def start_turn(mut self, user_text: String, max_new_tokens: Int) raises:
        if self.stream_active:
            raise Error("stream already active")
        if max_new_tokens <= 0:
            raise Error("max_new_tokens must be positive")

        var ub = user_block(user_text)
        var prompt = self.history + ub + assistant_prefix()
        var token_ids = self.tok.encode(prompt)
        var prompt_len = len(token_ids)
        if prompt_len >= MiniMaxM27Config.MAX_SEQ_LEN:
            raise Error("context is full; create a new M27Session")

        var limit = min(
            max_new_tokens, MiniMaxM27Config.MAX_SEQ_LEN - prompt_len)
        var reuse_tokens = self.cache.last_common_prefix(token_ids)
        var prefill_tokens = prompt_len - reuse_tokens
        var next_id = 0
        var tp_ptr = self.model.token_buffer()

        self.wake_workers()
        var prefilled = reuse_tokens
        while prefilled < prompt_len:
            var chunk_len = min(PREFILL_CHUNK_SIZE, prompt_len - prefilled)
            for i in range(chunk_len):
                tp_ptr[i] = Scalar[DType.int32](token_ids[prefilled + i])
            var is_last_prefill_chunk = prefilled + chunk_len == prompt_len
            next_id = Int(self.model.forward(
                Int(tp_ptr), prefilled, chunk_len, is_last_prefill_chunk))
            prefilled += chunk_len

        if prefill_tokens == 0:
            # KV contains the prompt, but not logits for the next token.
            tp_ptr[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
            next_id = Int(self.model.forward(
                Int(tp_ptr), prompt_len - 1, 1, True))

        self.cache.replace_with(token_ids)
        self.stream_active = True
        self.stream_pos = prompt_len
        self.stream_step = 0
        self.stream_limit = limit
        self.stream_next_id = next_id
        self.stream_user_block = ub^
        self.stream_generated = List[Int]()
        self.stream_decoder = StreamTextDecoder()
        self.sleep_workers()

    def finish_turn(mut self) raises -> PythonObject:
        var tail = self.stream_decoder.finish()
        var assistant_text = self.tok.decode(self.stream_generated)
        var assistant_content = assistant_content_for_history(assistant_text)
        self.history = self.history + self.stream_user_block + "]~b]ai\n" + assistant_content + "[e~[\n"

        self.stream_active = False
        self.stream_pos = 0
        self.stream_step = 0
        self.stream_limit = 0
        self.stream_next_id = MiniMaxM27Config.EOS_TOKEN_ID
        self.stream_user_block = String("")
        self.stream_generated = List[Int]()
        self.sleep_workers()

        if tail.byte_length() > 0:
            return PythonObject(tail^)
        return PythonObject()

    def next_chunk(mut self) raises -> PythonObject:
        if not self.stream_active:
            return PythonObject()

        self.wake_workers()
        var tp_ptr = self.model.token_buffer()
        while True:
            if (
                self.stream_step >= self.stream_limit
                or self.stream_next_id == MiniMaxM27Config.EOS_TOKEN_ID
            ):
                return self.finish_turn()

            if self.stream_step > 0:
                var committed_id = self.stream_next_id
                tp_ptr[0] = Scalar[DType.int32](committed_id)
                self.stream_next_id = Int(
                    self.model.forward(Int(tp_ptr), self.stream_pos, 1)
                )
                self.cache.append_committed(committed_id)
                self.stream_pos += 1

                if self.stream_next_id == MiniMaxM27Config.EOS_TOKEN_ID:
                    return self.finish_turn()

            self.stream_generated.append(self.stream_next_id)
            self.stream_step += 1

            var out = stream_token_text(
                self.tok, self.stream_decoder, self.stream_next_id)
            if out.byte_length() > 0:
                self.sleep_workers()
                return PythonObject(out^)

    def generate(mut self, user_text: String, max_new_tokens: Int) raises -> String:
        if max_new_tokens <= 0:
            raise Error("max_new_tokens must be positive")

        var ub = user_block(user_text)
        var prompt = self.history + ub + assistant_prefix()
        var token_ids = self.tok.encode(prompt)
        var prompt_len = len(token_ids)
        if prompt_len >= MiniMaxM27Config.MAX_SEQ_LEN:
            raise Error("context is full; create a new M27Session")

        var limit = min(
            max_new_tokens, MiniMaxM27Config.MAX_SEQ_LEN - prompt_len)
        var reuse_tokens = self.cache.last_common_prefix(token_ids)
        var prefill_tokens = prompt_len - reuse_tokens
        var next_id = 0
        var tp_ptr = self.model.token_buffer()

        self.wake_workers()
        var prefilled = reuse_tokens
        while prefilled < prompt_len:
            var chunk_len = min(PREFILL_CHUNK_SIZE, prompt_len - prefilled)
            for i in range(chunk_len):
                tp_ptr[i] = Scalar[DType.int32](token_ids[prefilled + i])
            var is_last_prefill_chunk = prefilled + chunk_len == prompt_len
            next_id = Int(self.model.forward(
                Int(tp_ptr), prefilled, chunk_len, is_last_prefill_chunk))
            prefilled += chunk_len

        if prefill_tokens == 0:
            # The KV cache already contains the full prompt, but it does not
            # store logits. Re-run the final prompt token in place to recover
            # logits for the next token without changing the logical KV prefix.
            tp_ptr[0] = Scalar[DType.int32](token_ids[prompt_len - 1])
            next_id = Int(self.model.forward(
                Int(tp_ptr), prompt_len - 1, 1, True))

        self.cache.replace_with(token_ids)

        var generated = List[Int]()
        if next_id != MiniMaxM27Config.EOS_TOKEN_ID:
            generated.append(next_id)

        var pos = prompt_len
        var step = 1
        while step < limit and next_id != MiniMaxM27Config.EOS_TOKEN_ID:
            var committed_id = next_id
            tp_ptr[0] = Scalar[DType.int32](committed_id)
            next_id = Int(self.model.forward(Int(tp_ptr), pos, 1))
            self.cache.append_committed(committed_id)
            pos += 1
            step += 1

            if next_id == MiniMaxM27Config.EOS_TOKEN_ID:
                break
            generated.append(next_id)

        var assistant_text = self.tok.decode(generated)
        var assistant_content = assistant_content_for_history(assistant_text)
        self.history = self.history + ub + "]~b]ai\n" + assistant_content + "[e~[\n"
        self.sleep_workers()
        return assistant_text^

    @staticmethod
    def py_send(py_self: PythonObject, user_text: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var result = self_ptr[].generate(
            String(user_text), DEFAULT_MAX_NEW_TOKENS)
        return PythonObject(result^)

    @staticmethod
    def py_send_with_limit(
        py_self: PythonObject,
        user_text: PythonObject,
        max_new_tokens: PythonObject,
    ) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var result = self_ptr[].generate(
            String(user_text), Int(py=max_new_tokens))
        return PythonObject(result^)

    @staticmethod
    def py_start_turn(py_self: PythonObject, user_text: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].start_turn(String(user_text), DEFAULT_MAX_NEW_TOKENS)
        return PythonObject()

    @staticmethod
    def py_start_turn_with_limit(
        py_self: PythonObject,
        user_text: PythonObject,
        max_new_tokens: PythonObject,
    ) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].start_turn(String(user_text), Int(py=max_new_tokens))
        return PythonObject()

    @staticmethod
    def py_next_chunk(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return self_ptr[].next_chunk()

    @staticmethod
    def py_reset(py_self: PythonObject, system_prompt: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].reset(String(system_prompt))
        return PythonObject()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "M27Session(history_bytes=",
            self.history.byte_length(),
            ")",
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)
