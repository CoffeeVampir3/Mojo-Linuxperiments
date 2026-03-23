"""Core tokenizer module: traits, Unicode classification, shared utilities, and BPETokenizer."""

from std.collections import Dict
from std.memory import Span, UnsafePointer

from .unicode_props import (
    LETTER_RANGES,
    LETTER_PAIR_COUNT,
    LETTER_MIN,
    LETTER_MAX,
    NUMBER_RANGES,
    NUMBER_PAIR_COUNT,
    NUMBER_MIN,
    NUMBER_MAX,
    WHITESPACE_RANGES,
    WHITESPACE_PAIR_COUNT,
    WHITESPACE_MIN,
    WHITESPACE_MAX,
)
from .unicode_psm_props import (
    MARK_RANGES,
    MARK_PAIR_COUNT,
    MARK_MIN,
    MARK_MAX,
    PUNCT_SYMBOL_RANGES,
    PUNCT_SYMBOL_PAIR_COUNT,
    PUNCT_SYMBOL_MIN,
    PUNCT_SYMBOL_MAX,
)
from .bpe import (
    pack_pair_ids, split_merge_pair, bpe_merge_ids,
    PieceCache,
)


# =============================================================================
# Capability traits
# =============================================================================


def bytes_to_gpt2(data: Span[Byte, _]) -> String:
    var cp_table = materialize[BYTE_TO_CODEPOINT]()
    var out = List[Byte]()
    out.reserve(len(data) * 2)
    for i in range(len(data)):
        var cp = Codepoint(unsafe_unchecked_codepoint=UInt32(cp_table[Int(data[i])]))
        var needed = cp.utf8_byte_length()
        var base = len(out)
        out.resize(unsafe_uninit_length=base + needed)
        _ = cp.unsafe_write_utf8[True](out.unsafe_ptr() + base)
    return String(unsafe_from_utf8=Span(out))


def gpt2_to_bytes(text: String) -> List[Byte]:
    var cp_table = materialize[CODEPOINT_TO_BYTE]()
    var out = List[Byte]()
    out.reserve(text.byte_length())
    for cp in text.codepoints():
        var val = Int(cp.to_u32())
        if val < 324:
            var byte_val = cp_table[val]
            if byte_val >= 0:
                out.append(Byte(byte_val))
    return out^


trait ByteTransformCapability(Movable, ImplicitlyDestructible):
    def encode_bytes(self, data: Span[Byte, _]) -> String:
        return bytes_to_gpt2(data)

    def decode_bytes(self, text: String) -> List[Byte]:
        return gpt2_to_bytes(text)


trait PreTokenizerCapability(Movable, ImplicitlyDestructible):
    def pre_tokenize(self, text: String) -> List[String]:
        ...


trait Tokenizer(Movable):
    def encode(mut self, text: String) -> List[Int]: ...
    def decode(self, ids: List[Int]) -> String: ...
    def vocab_size(self) -> Int: ...
    def token_to_id(self, token: String) -> Optional[Int]: ...
    def id_to_token(self, id: Int) -> Optional[String]: ...


# =============================================================================
# GPT2 byte encoding tables
# =============================================================================


def make_byte_to_codepoint() -> InlineArray[Int, 256]:
    var table = InlineArray[Int, 256](fill=0)
    var n = 256
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
            table[b] = b
        else:
            table[b] = n
            n += 1
    return table^


def make_codepoint_to_byte() -> InlineArray[Int, 324]:
    var table = InlineArray[Int, 324](fill=-1)
    var fwd = make_byte_to_codepoint()
    for b in range(256):
        table[fwd[b]] = b
    return table^


comptime BYTE_TO_CODEPOINT = make_byte_to_codepoint()
comptime CODEPOINT_TO_BYTE = make_codepoint_to_byte()


# =============================================================================
# UnicodeContext — bundled range table pointers
# =============================================================================


struct UnicodeContext(TrivialRegisterPassable):
    """Bundled Unicode range table pointers. Materializes comptime tables
    once and provides all classification pointers in a single value."""
    var letters: UnsafePointer[UInt32, MutAnyOrigin]
    var numbers: UnsafePointer[UInt32, MutAnyOrigin]
    var whitespace: UnsafePointer[UInt32, MutAnyOrigin]
    var marks: UnsafePointer[UInt32, MutAnyOrigin]
    var punct_symbols: UnsafePointer[UInt32, MutAnyOrigin]

    def __init__(out self):
        self.letters = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(materialize[LETTER_RANGES]().unsafe_ptr()))
        self.numbers = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(materialize[NUMBER_RANGES]().unsafe_ptr()))
        self.whitespace = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(materialize[WHITESPACE_RANGES]().unsafe_ptr()))
        self.marks = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(materialize[MARK_RANGES]().unsafe_ptr()))
        self.punct_symbols = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(materialize[PUNCT_SYMBOL_RANGES]().unsafe_ptr()))


# =============================================================================
# Unicode classification
# =============================================================================


@always_inline
def is_ascii_letter(b: Byte) -> Bool:
    return ((b | Byte(0x20)) - Byte(97)) < Byte(26)


@always_inline
def is_ascii_digit(b: Byte) -> Bool:
    return (b - Byte(48)) < Byte(10)


@always_inline
def is_ascii_regex_space(b: Byte) -> Bool:
    return (b >= Byte(9) and b <= Byte(13)) or b == Byte(32)


@always_inline
def decode_utf8_codepoint(data: Span[Byte, _], pos: Int, n: Int) -> Tuple[UInt32, Int]:
    var b0 = data[pos]
    if b0 < Byte(0x80):
        return (UInt32(b0), 1)
    if b0 < Byte(0xE0):
        if pos + 1 < n:
            var cp = (UInt32(b0 & Byte(0x1F)) << UInt32(6)) | UInt32(data[pos + 1] & Byte(0x3F))
            return (cp, 2)
        return (UInt32(b0), 1)
    if b0 < Byte(0xF0):
        if pos + 2 < n:
            var cp = (
                (UInt32(b0 & Byte(0x0F)) << UInt32(12))
                | (UInt32(data[pos + 1] & Byte(0x3F)) << UInt32(6))
                | UInt32(data[pos + 2] & Byte(0x3F))
            )
            return (cp, 3)
        return (UInt32(b0), 1)
    if pos + 3 < n:
        var cp = (
            (UInt32(b0 & Byte(0x07)) << UInt32(18))
            | (UInt32(data[pos + 1] & Byte(0x3F)) << UInt32(12))
            | (UInt32(data[pos + 2] & Byte(0x3F)) << UInt32(6))
            | UInt32(data[pos + 3] & Byte(0x3F))
        )
        return (cp, 4)
    return (UInt32(b0), 1)


@always_inline
def in_unicode_ranges(cp: UInt32, ranges: UnsafePointer[UInt32, _], pair_count: Int) -> Bool:
    var lo = 0
    var hi = pair_count
    while lo < hi:
        var mid = (lo + hi) // 2
        var start = ranges[mid * 2]
        var end = ranges[mid * 2 + 1]
        if cp < start:
            hi = mid
        elif cp > end:
            lo = mid + 1
        else:
            return True
    return False


@always_inline
def is_unicode_letter_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_letter(Byte(cp))
    if cp < LETTER_MIN or cp > LETTER_MAX:
        return False
    return in_unicode_ranges(cp, ctx.letters, LETTER_PAIR_COUNT)


@always_inline
def is_unicode_number_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_digit(Byte(cp))
    if cp < NUMBER_MIN or cp > NUMBER_MAX:
        return False
    return in_unicode_ranges(cp, ctx.numbers, NUMBER_PAIR_COUNT)


@always_inline
def is_unicode_whitespace_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_regex_space(Byte(cp))
    if cp < WHITESPACE_MIN or cp > WHITESPACE_MAX:
        return False
    return in_unicode_ranges(cp, ctx.whitespace, WHITESPACE_PAIR_COUNT)


@always_inline
def is_ascii_punct_symbol(b: Byte) -> Bool:
    return (
        (b >= Byte(33) and b <= Byte(47))
        or (b >= Byte(58) and b <= Byte(64))
        or (b >= Byte(91) and b <= Byte(96))
        or (b >= Byte(123) and b <= Byte(126))
    )


@always_inline
def is_unicode_punct_symbol_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_punct_symbol(Byte(cp))
    if cp < PUNCT_SYMBOL_MIN or cp > PUNCT_SYMBOL_MAX:
        return False
    return in_unicode_ranges(cp, ctx.punct_symbols, PUNCT_SYMBOL_PAIR_COUNT)


@always_inline
def is_unicode_mark_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp < MARK_MIN or cp > MARK_MAX:
        return False
    return in_unicode_ranges(cp, ctx.marks, MARK_PAIR_COUNT)


@always_inline
def is_number_start_at(data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_number_cp(parsed[0], ctx)


@always_inline
def is_whitespace_start_at(data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_whitespace_cp(parsed[0], ctx)


# =============================================================================
# Shared utilities
# =============================================================================


@always_inline
def span_to_string(data: Span[Byte, _], start: Int, end: Int) -> String:
    return String(unsafe_from_utf8=Span[Byte, _](ptr=data.unsafe_ptr() + start, length=end - start))


comptime PRETOKENIZE_SIMD_WIDTH = 16


@always_inline
def simd_ascii_letters[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return ((block | Byte(0x20)) - Byte(97)).le(Byte(25))


@always_inline
def simd_ascii_digits[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(48)).le(Byte(9))


@always_inline
def simd_spaces[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(9)).le(Byte(4)) | block.eq(Byte(32))


@always_inline
def skip_while_matching[
    scalar_pred: def(Byte) -> Bool,
    simd_pred: def[w: Int](SIMD[DType.uint8, w]) -> SIMD[DType.bool, w],
    width: Int = PRETOKENIZE_SIMD_WIDTH,
](data: Span[Byte, _], pos: Int, n: Int) -> Int:
    var i = pos
    var data_ptr = data.unsafe_ptr()
    while i + width <= n:
        var block = (data_ptr + i).load[width=width]()
        var mask = simd_pred[width](block)
        if all(mask):
            i += width
            continue
        comptime for lane in range(width):
            if not mask[lane]:
                return i + lane
    while i < n and scalar_pred(data[i]):
        i += 1
    return i


@always_inline
def consume_codepoint_run[
    pred: def(UInt32, UnicodeContext) -> Bool,
](data: Span[Byte, _], start: Int, n: Int, ctx: UnicodeContext) -> Int:
    var i = start
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if not pred(parsed[0], ctx):
            break
        i += parsed[1]
    return i


def sort_strings_by_byte_length_desc(mut values: List[String]):
    for i in range(1, len(values)):
        var cur = values[i]
        var cur_len = cur.byte_length()
        var j = i
        while j > 0 and values[j - 1].byte_length() < cur_len:
            values[j] = values[j - 1]
            j -= 1
        values[j] = cur


@always_inline
def span_matches_at(data: Span[Byte, _], pos: Int, pattern: Span[Byte, _]) -> Bool:
    if pos + len(pattern) > len(data):
        return False
    for i in range(len(pattern)):
        if data[pos + i] != pattern[i]:
            return False
    return True


def find_added_token_match(
    text: Span[Byte, _],
    pos: Int,
    added_token_order: List[String],
    added_tokens: Dict[String, Int],
) -> Tuple[Int, Int]:
    for tok in added_token_order:
        var tok_bytes = tok.as_bytes()
        if span_matches_at(text, pos, tok_bytes):
            var found = added_tokens.get(tok)
            if found:
                return (found.value(), len(tok_bytes))
    return (-1, 0)


def split_numbers(
    piece: String, max_group: Int, ctx: UnicodeContext, mut out: List[String],
):
    """Split digit/number runs. max_group=1 for individual digits (GPT2),
    max_group=3 for DeepSeek-style 1-3 grouping."""
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        if is_number_start_at(data, i, n, ctx):
            if chunk_start < i:
                out.append(span_to_string(data, chunk_start, i))

            var j = i
            var count = 0
            while j < n and count < max_group and is_number_start_at(data, j, n, ctx):
                var parsed = decode_utf8_codepoint(data, j, n)
                j += parsed[1]
                count += 1

            out.append(span_to_string(data, i, j))
            i = j
            chunk_start = i
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]

    if chunk_start < n:
        out.append(span_to_string(data, chunk_start, n))


# =============================================================================
# BPETokenizer
# =============================================================================


struct BPETokenizer[
    pretokenizer_type: PreTokenizerCapability,
    byte_transform_type: ByteTransformCapability,
](Tokenizer):
    var vocab: Dict[String, Int]
    var vocab_rev: List[String]
    var merges: List[String]
    var merge_ranks: Dict[String, Int]
    var merge_pair_ranks: Dict[UInt64, Int]
    var merge_pair_out: Dict[UInt64, Int]
    var added_tokens: Dict[String, Int]
    var added_token_order: List[String]
    var special_tokens: Dict[String, Int]
    var special_ids: List[Int]
    var ignore_merges: Bool
    var fuse_unk: Bool
    var byte_fallback: Bool
    var unk_token: String
    var unk_token_id: Int
    var add_bos_token: Bool
    var add_eos_token: Bool
    var bos_token_id: Int
    var eos_token_id: Int
    var _vocab_size: Int
    var piece_cache: PieceCache
    var pretokenizer: Self.pretokenizer_type
    var byte_transform: Self.byte_transform_type

    def __init__(
        out self,
        var vocab: Dict[String, Int],
        var merges: List[String],
        var added_tokens: Dict[String, Int],
        var added_token_order: List[String],
        var special_tokens: Dict[String, Int],
        var special_ids: List[Int],
        ignore_merges: Bool,
        fuse_unk: Bool,
        byte_fallback: Bool,
        unk_token: String,
        add_bos_token: Bool,
        add_eos_token: Bool,
        bos_token_id: Int,
        eos_token_id: Int,
        vocab_size: Int,
        var pretokenizer: Self.pretokenizer_type,
        var byte_transform: Self.byte_transform_type,
    ):
        var vocab_rev = List[String](length=vocab_size, fill=String(""))
        for item in vocab.items():
            var id = item.value
            if id >= 0 and id < vocab_size:
                vocab_rev[id] = item.key.copy()

        var merge_ranks = Dict[String, Int]()
        var merge_pair_ranks = Dict[UInt64, Int]()
        var merge_pair_out = Dict[UInt64, Int]()
        for i in range(len(merges)):
            merge_ranks[merges[i].copy()] = i
            var split = split_merge_pair(merges[i])
            if not split:
                continue
            var left_tok = split.value()[0]
            var right_tok = split.value()[1]
            var left_id = vocab.get(left_tok)
            var right_id = vocab.get(right_tok)
            if not left_id or not right_id:
                continue
            var merged_tok = left_tok + right_tok
            var out_id = vocab.get(merged_tok)
            if not out_id:
                continue
            var key = pack_pair_ids(left_id.value(), right_id.value())
            merge_pair_ranks[key] = i
            merge_pair_out[key] = out_id.value()

        sort_strings_by_byte_length_desc(added_token_order)

        var unk_id = -1
        if unk_token.byte_length() > 0:
            var found = vocab.get(unk_token)
            if found:
                unk_id = found.value()

        self.vocab = vocab^
        self.vocab_rev = vocab_rev^
        self.merges = merges^
        self.merge_ranks = merge_ranks^
        self.merge_pair_ranks = merge_pair_ranks^
        self.merge_pair_out = merge_pair_out^
        self.added_tokens = added_tokens^
        self.added_token_order = added_token_order^
        self.special_tokens = special_tokens^
        self.special_ids = special_ids^
        self.ignore_merges = ignore_merges
        self.fuse_unk = fuse_unk
        self.byte_fallback = byte_fallback
        self.unk_token = unk_token
        self.unk_token_id = unk_id
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._vocab_size = vocab_size
        self.piece_cache = PieceCache()
        self.pretokenizer = pretokenizer^
        self.byte_transform = byte_transform^

    def vocab_size(self) -> Int:
        return self._vocab_size

    def token_to_id(self, token: String) -> Optional[Int]:
        var found = self.vocab.get(token)
        if found:
            return found.value()
        return None

    def id_to_token(self, id: Int) -> Optional[String]:
        if id >= 0 and id < self._vocab_size:
            var tok = self.vocab_rev[id]
            if tok.byte_length() > 0:
                return tok
        return None

    def is_special_token(self, token: String) -> Bool:
        return self.special_tokens.__contains__(token)

    def is_special_id(self, id: Int) -> Bool:
        for sid in self.special_ids:
            if sid == id:
                return True
        return False

    def merge_rank(self, pair: String) -> Int:
        var found = self.merge_ranks.get(pair)
        if found:
            return found.value()
        return -1

    def num_merges(self) -> Int:
        return len(self.merges)

    def num_special_tokens(self) -> Int:
        return len(self.special_tokens)

    def encode_piece(mut self, piece: String, mut ids: List[Int]):
        if piece.byte_length() == 0:
            return

        if self.piece_cache.get(piece, ids):
            return

        var transformed = self.byte_transform.encode_bytes(piece.as_bytes())
        var symbol_ids = List[Int]()
        for slice in transformed.codepoint_slices():
            var found = self.vocab.get(String(slice))
            if found:
                symbol_ids.append(found.value())
            elif self.unk_token_id >= 0:
                symbol_ids.append(self.unk_token_id)
        if not self.ignore_merges:
            symbol_ids = bpe_merge_ids(symbol_ids, self.merge_pair_ranks, self.merge_pair_out)

        self.piece_cache.put(piece.copy(), symbol_ids)

        for id in symbol_ids:
            ids.append(id)

    def encode_span(mut self, data: Span[Byte, _], start: Int, end: Int, mut ids: List[Int]):
        if end <= start:
            return
        var chunk = span_to_string(data, start, end)
        var pieces = self.pretokenizer.pre_tokenize(chunk)
        for piece in pieces:
            self.encode_piece(String(piece), ids)

    def encode(mut self, text: String) -> List[Int]:
        var ids = List[Int]()

        if self.add_bos_token and self.bos_token_id >= 0:
            ids.append(self.bos_token_id)

        if text.byte_length() > 0:
            var data = text.as_bytes()
            var n = len(data)
            var i = 0
            var chunk_start = 0
            while i < n:
                var matched = find_added_token_match(data, i, self.added_token_order, self.added_tokens)
                var tok_id = matched[0]
                var tok_len = matched[1]
                if tok_id >= 0 and tok_len > 0:
                    if chunk_start < i:
                        self.encode_span(data, chunk_start, i, ids)
                    ids.append(tok_id)
                    i += tok_len
                    chunk_start = i
                    continue
                i += 1
            if chunk_start < n:
                self.encode_span(data, chunk_start, n, ids)

        if self.add_eos_token and self.eos_token_id >= 0:
            ids.append(self.eos_token_id)

        return ids^

    def decode(self, ids: List[Int]) -> String:
        var encoded_parts = List[Byte]()
        var decoded = String("")
        for id in ids:
            if id < 0 or id >= self._vocab_size:
                continue

            var tok = self.vocab_rev[id]
            if self.is_special_id(id):
                if len(encoded_parts) > 0:
                    var raw_bytes = self.byte_transform.decode_bytes(
                        String(unsafe_from_utf8=Span(encoded_parts))
                    )
                    decoded = decoded + String(unsafe_from_utf8=Span(raw_bytes))
                    encoded_parts.resize(unsafe_uninit_length=0)
                decoded = decoded + tok.copy()
                continue

            var tok_data = tok.as_bytes()
            for j in range(len(tok_data)):
                encoded_parts.append(tok_data[j])

        if len(encoded_parts) > 0:
            var raw_bytes = self.byte_transform.decode_bytes(String(unsafe_from_utf8=Span(encoded_parts)))
            decoded = decoded + String(unsafe_from_utf8=Span(raw_bytes))
        return decoded^
