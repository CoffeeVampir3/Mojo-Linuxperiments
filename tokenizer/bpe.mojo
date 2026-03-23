"""BPE merge algorithm internals: merge engine, min-heap, and piece cache."""

from std.collections import Dict
from std.memory import Span
from .tokenizer import span_to_string


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


@fieldwise_init
struct MergeCandidate(Copyable, ImplicitlyCopyable):
    var rank: Int
    var left: Int
    var right: Int


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
        for sym in symbols:
            out_short.append(sym)
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
        for id in symbol_ids:
            self.values.append(id)
        var slot = len(self.starts)
        self.starts.append(entry_start)
        self.lens.append(len(symbol_ids))
        self.index[piece] = slot

    def clear(mut self):
        self.index = Dict[String, Int]()
        self.starts.resize(unsafe_uninit_length=0)
        self.lens.resize(unsafe_uninit_length=0)
        self.values.resize(unsafe_uninit_length=0)
