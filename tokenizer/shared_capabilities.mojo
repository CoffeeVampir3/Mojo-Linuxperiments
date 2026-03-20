from memory import Span

from .unicode_props import (
    LETTER_PAIR_COUNT,
    LETTER_MIN,
    LETTER_MAX,
    NUMBER_PAIR_COUNT,
    NUMBER_MIN,
    NUMBER_MAX,
    WHITESPACE_PAIR_COUNT,
    WHITESPACE_MIN,
    WHITESPACE_MAX,
)


fn make_byte_to_codepoint() -> InlineArray[Int, 256]:
    var table = InlineArray[Int, 256](fill=0)
    var n = 256
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
            table[b] = b
        else:
            table[b] = n
            n += 1
    return table^


fn make_codepoint_to_byte() -> InlineArray[Int, 324]:
    var table = InlineArray[Int, 324](fill=-1)
    var fwd = make_byte_to_codepoint()
    for b in range(256):
        table[fwd[b]] = b
    return table^


comptime BYTE_TO_CODEPOINT = make_byte_to_codepoint()
comptime CODEPOINT_TO_BYTE = make_codepoint_to_byte()


fn bytes_to_gpt2(data: Span[Byte]) -> String:
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


fn gpt2_to_bytes(text: String) -> List[Byte]:
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


@always_inline
fn is_ascii_letter(b: Byte) -> Bool:
    return ((b | Byte(0x20)) - Byte(97)) < Byte(26)


@always_inline
fn is_ascii_digit(b: Byte) -> Bool:
    return (b - Byte(48)) < Byte(10)


@always_inline
fn is_ascii_regex_space(b: Byte) -> Bool:
    return (b >= Byte(9) and b <= Byte(13)) or b == Byte(32)


@always_inline
fn decode_utf8_codepoint(data: Span[Byte], pos: Int, n: Int) -> Tuple[UInt32, Int]:
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
fn in_unicode_ranges(cp: UInt32, ranges: UnsafePointer[UInt32], pair_count: Int) -> Bool:
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
fn is_unicode_letter_cp(cp: UInt32, letter_ranges: UnsafePointer[UInt32]) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_letter(Byte(cp))
    if cp < LETTER_MIN or cp > LETTER_MAX:
        return False
    return in_unicode_ranges(cp, letter_ranges, LETTER_PAIR_COUNT)


@always_inline
fn is_unicode_number_cp(cp: UInt32, number_ranges: UnsafePointer[UInt32]) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_digit(Byte(cp))
    if cp < NUMBER_MIN or cp > NUMBER_MAX:
        return False
    return in_unicode_ranges(cp, number_ranges, NUMBER_PAIR_COUNT)


@always_inline
fn is_unicode_whitespace_cp(cp: UInt32, whitespace_ranges: UnsafePointer[UInt32]) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_regex_space(Byte(cp))
    if cp < WHITESPACE_MIN or cp > WHITESPACE_MAX:
        return False
    return in_unicode_ranges(cp, whitespace_ranges, WHITESPACE_PAIR_COUNT)


@always_inline
fn is_number_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    number_ranges: UnsafePointer[UInt32],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_number_cp(parsed[0], number_ranges)


@always_inline
fn is_whitespace_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    whitespace_ranges: UnsafePointer[UInt32],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_whitespace_cp(parsed[0], whitespace_ranges)


@always_inline
fn span_to_string(data: Span[Byte], start: Int, end: Int) -> String:
    return String(unsafe_from_utf8=Span[Byte](ptr=data.unsafe_ptr() + start, length=end - start))
