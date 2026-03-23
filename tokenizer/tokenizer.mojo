from std.collections import Dict
from std.memory import Span
from .capabilities import ByteTransformCapability, PreTokenizerCapability
from .auto import AutoPreTokenizer
from .gpt2 import GPT2ByteTransform
from .shared_capabilities import (
    span_to_string,
    sort_strings_by_byte_length_desc,
    span_matches_at,
    find_added_token_match,
)


trait Tokenizer(Movable):
    def encode(mut self, text: String) -> List[Int]:
        ...

    def decode(self, ids: List[Int]) -> String:
        ...

    def vocab_size(self) -> Int:
        ...

    def token_to_id(self, token: String) -> Optional[Int]:
        ...

    def id_to_token(self, id: Int) -> Optional[String]:
        ...


@always_inline
def pack_pair_ids(left: Int, right: Int) -> UInt64:
    return (UInt64(UInt32(left)) << UInt64(32)) | UInt64(UInt32(right))


@always_inline
def split_merge_pair(pair: String) -> Optional[Tuple[String, String]]:
    var bytes = pair.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == Byte(32):
            return (
                span_to_string(bytes, 0, i),
                span_to_string(bytes, i + 1, len(bytes)),
            )
    return None


struct MergeCandidate(Copyable, ImplicitlyCopyable):
    var rank: Int
    var left: Int
    var right: Int

    def __init__(out self, rank: Int, left: Int, right: Int):
        self.rank = rank
        self.left = left
        self.right = right


@always_inline
def candidate_less(a: MergeCandidate, b: MergeCandidate) -> Bool:
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.left < b.left


def heap_push(mut heap: List[MergeCandidate], cand: MergeCandidate):
    heap.append(cand)
    var idx = len(heap) - 1
    while idx > 0:
        var parent = (idx - 1) // 2
        if not candidate_less(heap[idx], heap[parent]):
            break
        var tmp = heap[parent]
        heap[parent] = heap[idx]
        heap[idx] = tmp
        idx = parent


def heap_pop_min(mut heap: List[MergeCandidate]) -> MergeCandidate:
    var top = heap[0]
    var n = len(heap)
    if n == 1:
        heap.resize(unsafe_uninit_length=0)
        return top
    heap[0] = heap[n - 1]
    heap.resize(unsafe_uninit_length=n - 1)

    var idx = 0
    var heap_n = len(heap)
    while True:
        var left = idx * 2 + 1
        if left >= heap_n:
            break
        var right = left + 1
        var best = left
        if right < heap_n and candidate_less(heap[right], heap[left]):
            best = right
        if not candidate_less(heap[best], heap[idx]):
            break
        var tmp = heap[idx]
        heap[idx] = heap[best]
        heap[best] = tmp
        idx = best
    return top


def bpe_merge_ids(
    mut symbols: List[Int],
    pair_ranks: Dict[UInt64, Int],
    pair_out: Dict[UInt64, Int],
) -> List[Int]:
    var n = len(symbols)
    if n <= 1:
        var out_short = List[Int]()
        for i in range(n):
            out_short.append(symbols[i])
        return out_short^

    var prev = List[Int](length=n, fill=-1)
    var next = List[Int](length=n, fill=-1)
    var alive = List[Bool](length=n, fill=True)
    for i in range(n):
        if i > 0:
            prev[i] = i - 1
        if i + 1 < n:
            next[i] = i + 1

    var heap = List[MergeCandidate]()
    heap.reserve(n)
    for i in range(n - 1):
        var key = pack_pair_ids(symbols[i], symbols[i + 1])
        var rank = pair_ranks.get(key)
        if rank:
            heap_push(heap, MergeCandidate(rank.value(), i, i + 1))

    while len(heap) > 0:
        var cand = heap_pop_min(heap)
        var l = cand.left
        var r = cand.right
        if l < 0 or r < 0:
            continue
        if not alive[l] or not alive[r]:
            continue
        if next[l] != r:
            continue

        var key = pack_pair_ids(symbols[l], symbols[r])
        var rank = pair_ranks.get(key)
        if not rank or rank.value() != cand.rank:
            continue
        var out = pair_out.get(key)
        if not out:
            continue

        symbols[l] = out.value()
        alive[r] = False
        var rr = next[r]
        next[l] = rr
        if rr >= 0:
            prev[rr] = l
        next[r] = -1
        prev[r] = -1

        var pl = prev[l]
        if pl >= 0 and alive[pl]:
            var k0 = pack_pair_ids(symbols[pl], symbols[l])
            var r0 = pair_ranks.get(k0)
            if r0:
                heap_push(heap, MergeCandidate(r0.value(), pl, l))

        if rr >= 0 and alive[rr]:
            var k1 = pack_pair_ids(symbols[l], symbols[rr])
            var r1 = pair_ranks.get(k1)
            if r1:
                heap_push(heap, MergeCandidate(r1.value(), l, rr))

    var head = -1
    for i in range(n):
        if alive[i] and prev[i] < 0:
            head = i
            break

    var out_ids = List[Int]()
    if head < 0:
        return out_ids^
    var cur = head
    while cur >= 0:
        if alive[cur]:
            out_ids.append(symbols[cur])
        cur = next[cur]
    return out_ids^


comptime PIECE_CACHE_MAX_ENTRIES = 65536
comptime PIECE_CACHE_MAX_IDS_PER_ENTRY = 128


struct PieceCache(Movable):
    var index: Dict[String, Int]
    var starts: List[Int]
    var lens: List[Int]
    var values: List[Int]

    def __init__(out self):
        self.index = Dict[String, Int]()
        self.starts = List[Int]()
        self.lens = List[Int]()
        self.values = List[Int]()

    def get(self, piece: String, mut ids: List[Int]) -> Bool:
        var cached_slot = self.index.get(piece)
        if not cached_slot:
            return False
        var slot = cached_slot.value()
        if slot < 0 or slot >= len(self.starts):
            return False
        var start = self.starts[slot]
        var count = self.lens[slot]
        for i in range(count):
            ids.append(self.values[start + i])
        return True

    def put(mut self, piece: String, symbol_ids: List[Int]):
        if len(symbol_ids) == 0 or len(symbol_ids) > PIECE_CACHE_MAX_IDS_PER_ENTRY:
            return
        if len(self.starts) >= PIECE_CACHE_MAX_ENTRIES:
            self.clear()
        var entry_start = len(self.values)
        for i in range(len(symbol_ids)):
            self.values.append(symbol_ids[i])
        var slot = len(self.starts)
        self.starts.append(entry_start)
        self.lens.append(len(symbol_ids))
        self.index[piece] = slot

    def clear(mut self):
        self.index = Dict[String, Int]()
        self.starts.resize(unsafe_uninit_length=0)
        self.lens.resize(unsafe_uninit_length=0)
        self.values.resize(unsafe_uninit_length=0)


struct BPETokenizer[
    pretokenizer_type: PreTokenizerCapability = AutoPreTokenizer,
    byte_transform_type: ByteTransformCapability = GPT2ByteTransform,
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
        for i in range(len(self.special_ids)):
            if self.special_ids[i] == id:
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

        for i in range(len(symbol_ids)):
            ids.append(symbol_ids[i])

    def encode_span(mut self, data: Span[Byte, _], start: Int, end: Int, mut ids: List[Int]):
        if end <= start:
            return
        var chunk = span_to_string(data, start, end)
        var pieces = self.pretokenizer.pre_tokenize(chunk)
        for i in range(len(pieces)):
            self.encode_piece(pieces[i], ids)

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
        for i in range(len(ids)):
            var id = ids[i]
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

    def print_summary(self):
        print("=== BPETokenizer Summary ===")
        print("Vocab size:      ", self._vocab_size)
        print("Merge rules:     ", len(self.merges))
        print("Special tokens:  ", len(self.special_tokens))
        print("Added tokens:    ", len(self.added_tokens))
        print("Ignore merges:   ", self.ignore_merges)
        print("Fuse unk:        ", self.fuse_unk)
        print("Byte fallback:   ", self.byte_fallback)
        print("Add BOS token:   ", self.add_bos_token, " (id=", self.bos_token_id, ")")
        print("Add EOS token:   ", self.add_eos_token, " (id=", self.eos_token_id, ")")
        print()

        print("Special Tokens:")
        for item in self.special_tokens.items():
            print("  id:", item.value, " token:", repr(item.key))
        print()

        print("First 5 Merge Rules:")
        var show = 5
        if len(self.merges) < show:
            show = len(self.merges)
        for i in range(show):
            print("  [", i, "]", repr(self.merges[i]))
        print()

        if len(self.merges) > 10:
            print("Last 5 Merge Rules:")
            for i in range(len(self.merges) - 5, len(self.merges)):
                print("  [", i, "]", repr(self.merges[i]))
            print()

        print("First 10 Vocab Entries (by ID):")
        var show_vocab = 10
        if self._vocab_size < show_vocab:
            show_vocab = self._vocab_size
        for i in range(show_vocab):
            print("  id:", i, " token:", repr(self.vocab_rev[i]))
        print()

        print("Sample Vocab Lookups:")
        var samples = List[String]()
        samples.append("hello")
        samples.append("Hello")
        samples.append("the")
        samples.append("Ġthe")
        samples.append("def")
        samples.append("Ġdef")
        samples.append(".")
        samples.append("Ċ")
        samples.append("0")
        samples.append("Ġ")
        for i in range(len(samples)):
            var found = self.vocab.get(samples[i])
            if found:
                print("  ", repr(samples[i]), "->", found.value())
            else:
                print("  ", repr(samples[i]), "-> NOT FOUND")
        print()
