from std.memory import Span
from ..tokenizer import decode_utf8_codepoint
from .nfc_tables import (
    DECOMP_KEYS, DECOMP_OFFSETS, DECOMP_FLAT,
    DECOMP_KEY_COUNT, DECOMP_MIN, DECOMP_MAX,
    CCC_KEYS, CCC_VALUES, CCC_KEY_COUNT, CCC_MIN, CCC_MAX,
    COMP_PACKED_KEYS, COMP_VALUES, COMP_KEY_COUNT,
)


comptime SBASE = UInt32(0xAC00)
comptime LBASE = UInt32(0x1100)
comptime VBASE = UInt32(0x1161)
comptime TBASE = UInt32(0x11A7)
comptime LCOUNT = UInt32(19)
comptime VCOUNT = UInt32(21)
comptime TCOUNT = UInt32(28)
comptime NCOUNT = UInt32(588)
comptime SCOUNT = UInt32(11172)


@always_inline
def is_hangul_syllable(cp: UInt32) -> Bool:
    return cp >= SBASE and cp < SBASE + SCOUNT


@always_inline
def lookup_ccc(cp: UInt32) -> UInt8:
    if cp < CCC_MIN or cp > CCC_MAX:
        return UInt8(0)
    var lo = 0
    var hi = CCC_KEY_COUNT
    while lo < hi:
        var mid = (lo + hi) // 2
        var k = CCC_KEYS[mid]
        if cp < k:
            hi = mid
        elif cp > k:
            lo = mid + 1
        else:
            return CCC_VALUES[mid]
    return UInt8(0)


@always_inline
def lookup_decomp_index(cp: UInt32) -> Int:
    if cp < DECOMP_MIN or cp > DECOMP_MAX:
        return -1
    var lo = 0
    var hi = DECOMP_KEY_COUNT
    while lo < hi:
        var mid = (lo + hi) // 2
        var k = DECOMP_KEYS[mid]
        if cp < k:
            hi = mid
        elif cp > k:
            lo = mid + 1
        else:
            return mid
    return -1


@always_inline
def lookup_composition(first: UInt32, second: UInt32) -> UInt32:
    if first >= LBASE and first < LBASE + LCOUNT:
        if second >= VBASE and second < VBASE + VCOUNT:
            var lindex = first - LBASE
            var vindex = second - VBASE
            return SBASE + (lindex * VCOUNT + vindex) * TCOUNT
    if is_hangul_syllable(first):
        var sindex = first - SBASE
        if sindex % TCOUNT == UInt32(0) and second > TBASE and second < TBASE + TCOUNT:
            return first + (second - TBASE)
    var key = (UInt64(first) << UInt64(32)) | UInt64(second)
    var lo = 0
    var hi = COMP_KEY_COUNT
    while lo < hi:
        var mid = (lo + hi) // 2
        var k = COMP_PACKED_KEYS[mid]
        if key < k:
            hi = mid
        elif key > k:
            lo = mid + 1
        else:
            return COMP_VALUES[mid]
    return UInt32(0)


def decompose_one(cp: UInt32, mut out: List[UInt32]):
    if is_hangul_syllable(cp):
        var sindex = cp - SBASE
        var l = LBASE + sindex // NCOUNT
        var v = VBASE + (sindex % NCOUNT) // TCOUNT
        var t = TBASE + sindex % TCOUNT
        out.append(l)
        out.append(v)
        if t != TBASE:
            out.append(t)
        return
    var idx = lookup_decomp_index(cp)
    if idx < 0:
        out.append(cp)
        return
    var start = Int(DECOMP_OFFSETS[idx])
    var end = Int(DECOMP_OFFSETS[idx + 1])
    for i in range(start, end):
        out.append(DECOMP_FLAT[i])


def canonical_reorder(mut cps: List[UInt32]):
    var n = len(cps)
    var i = 1
    while i < n:
        var ccc_i = lookup_ccc(cps[i])
        if ccc_i == UInt8(0):
            i += 1
            continue
        var ccc_prev = lookup_ccc(cps[i - 1])
        if ccc_prev != UInt8(0) and ccc_prev > ccc_i:
            var tmp = cps[i]
            cps[i] = cps[i - 1]
            cps[i - 1] = tmp
            if i > 1:
                i -= 1
                continue
        i += 1


def compose(var cps: List[UInt32]) -> List[UInt32]:
    var n = len(cps)
    var result = List[UInt32]()
    if n == 0:
        return result^
    var starter_idx = -1
    var max_ccc = UInt8(0)
    for i in range(n):
        var c = cps[i]
        var c_ccc = lookup_ccc(c)
        if starter_idx >= 0:
            var not_blocked = (c_ccc > max_ccc) or (c_ccc == UInt8(0) and max_ccc == UInt8(0))
            if not_blocked:
                var composed = lookup_composition(result[starter_idx], c)
                if composed != UInt32(0):
                    result[starter_idx] = composed
                    continue
        result.append(c)
        if c_ccc == UInt8(0):
            starter_idx = len(result) - 1
            max_ccc = UInt8(0)
        else:
            if c_ccc > max_ccc:
                max_ccc = c_ccc
    return result^


def encode_utf8(cp: UInt32, mut out: List[Byte]):
    if cp < UInt32(0x80):
        out.append(Byte(cp))
        return
    if cp < UInt32(0x800):
        out.append(Byte(UInt32(0xC0) | (cp >> UInt32(6))))
        out.append(Byte(UInt32(0x80) | (cp & UInt32(0x3F))))
        return
    if cp < UInt32(0x10000):
        out.append(Byte(UInt32(0xE0) | (cp >> UInt32(12))))
        out.append(Byte(UInt32(0x80) | ((cp >> UInt32(6)) & UInt32(0x3F))))
        out.append(Byte(UInt32(0x80) | (cp & UInt32(0x3F))))
        return
    out.append(Byte(UInt32(0xF0) | (cp >> UInt32(18))))
    out.append(Byte(UInt32(0x80) | ((cp >> UInt32(12)) & UInt32(0x3F))))
    out.append(Byte(UInt32(0x80) | ((cp >> UInt32(6)) & UInt32(0x3F))))
    out.append(Byte(UInt32(0x80) | (cp & UInt32(0x3F))))


@always_inline
def cp_may_need_normalization(cp: UInt32) -> Bool:
    if cp < UInt32(0x80):
        return False
    if is_hangul_syllable(cp):
        return True
    if cp >= DECOMP_MIN and cp <= DECOMP_MAX:
        return True
    if cp >= CCC_MIN and cp <= CCC_MAX:
        return True
    return False


def nfc_normalize(text: String) -> String:
    var bytes = text.as_bytes()
    var n = len(bytes)

    var any_nonascii = False
    for i in range(n):
        if bytes[i] >= Byte(0x80):
            any_nonascii = True
            break
    if not any_nonascii:
        return text

    var cps = List[UInt32](capacity=n)
    var i = 0
    var any_candidate = False
    while i < n:
        var parsed = decode_utf8_codepoint(bytes, i, n)
        cps.append(parsed[0])
        if not any_candidate and cp_may_need_normalization(parsed[0]):
            any_candidate = True
        i += parsed[1]

    if not any_candidate:
        return text

    var decomposed = List[UInt32](capacity=len(cps) * 2)
    for j in range(len(cps)):
        decompose_one(cps[j], decomposed)

    canonical_reorder(decomposed)
    var composed = compose(decomposed^)

    var out_bytes = List[Byte](capacity=n)
    for k in range(len(composed)):
        encode_utf8(composed[k], out_bytes)
    return String(unsafe_from_utf8=out_bytes^)
