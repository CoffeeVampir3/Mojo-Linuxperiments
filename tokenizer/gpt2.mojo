from std.memory import Span
from .capabilities import ByteTransformCapability, PreTokenizerCapability
from .shared_capabilities import (
    is_ascii_letter,
    is_ascii_digit,
    is_ascii_regex_space,
    decode_utf8_codepoint,
    is_unicode_letter_cp,
    is_unicode_number_cp,
    is_unicode_whitespace_cp,
    is_number_start_at,
    is_whitespace_start_at,
    span_to_string,
    skip_while_matching,
    simd_ascii_letters,
    simd_ascii_digits,
    simd_spaces,
    consume_codepoint_run,
)
from .unicode_props import (
    LETTER_RANGES,
    NUMBER_RANGES,
    WHITESPACE_RANGES,
)


@always_inline
def is_letter_start_at(
    data: Span[Byte, _],
    pos: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32, _],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_letter_cp(parsed[0], letter_ranges)


@always_inline
def is_symbol_start_at(
    data: Span[Byte, _],
    pos: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32, _],
    number_ranges: UnsafePointer[UInt32, _],
    whitespace_ranges: UnsafePointer[UInt32, _],
) -> Bool:
    var b = data[pos]
    if b < Byte(0x80):
        return not is_ascii_regex_space(b) and not is_ascii_letter(b) and not is_ascii_digit(b)
    var parsed = decode_utf8_codepoint(data, pos, n)
    var cp = parsed[0]
    if is_unicode_whitespace_cp(cp, whitespace_ranges):
        return False
    if is_unicode_letter_cp(cp, letter_ranges):
        return False
    if is_unicode_number_cp(cp, number_ranges):
        return False
    return True


@always_inline
def consume_letter_run(
    data: Span[Byte, _],
    start: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32, _],
) -> Int:
    var i = start
    while i < n:
        var b = data[i]
        if is_ascii_letter(b):
            i = skip_while_matching[is_ascii_letter, simd_ascii_letters](data, i, n)
            continue
        var parsed = decode_utf8_codepoint(data, i, n)
        if is_unicode_letter_cp(parsed[0], letter_ranges):
            i += parsed[1]
            continue
        break
    return i


@always_inline
def consume_number_run(
    data: Span[Byte, _],
    start: Int,
    n: Int,
    number_ranges: UnsafePointer[UInt32, _],
) -> Int:
    var i = start
    while i < n:
        if is_ascii_digit(data[i]):
            i = skip_while_matching[is_ascii_digit, simd_ascii_digits](data, i, n)
            continue
        var parsed = decode_utf8_codepoint(data, i, n)
        if is_unicode_number_cp(parsed[0], number_ranges):
            i += parsed[1]
            continue
        break
    return i


@always_inline
def consume_symbol_run(
    data: Span[Byte, _],
    start: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32, _],
    number_ranges: UnsafePointer[UInt32, _],
    whitespace_ranges: UnsafePointer[UInt32, _],
) -> Int:
    var i = start
    while i < n:
        if not is_symbol_start_at(data, i, n, letter_ranges, number_ranges, whitespace_ranges):
            break
        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]
    return i


@always_inline
def consume_whitespace_run(
    data: Span[Byte, _],
    start: Int,
    n: Int,
    whitespace_ranges: UnsafePointer[UInt32, _],
) -> Tuple[Int, Int, Int]:
    var i = start
    var last_cp_start = start
    var cp_count = 0

    while i < n:
        var b = data[i]
        if b < Byte(0x80):
            if not is_ascii_regex_space(b):
                break
            var ascii_end = skip_while_matching[is_ascii_regex_space, simd_spaces](data, i, n)
            if ascii_end <= i:
                break
            cp_count += ascii_end - i
            last_cp_start = ascii_end - 1
            i = ascii_end
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        if not is_unicode_whitespace_cp(parsed[0], whitespace_ranges):
            break
        last_cp_start = i
        cp_count += 1
        i += parsed[1]

    return (i, last_cp_start, cp_count)


def try_contraction(data: Span[Byte, _], pos: Int, n: Int) -> Int:
    if pos + 1 >= n:
        return 0
    var c = data[pos + 1]
    if c == Byte(115) or c == Byte(116) or c == Byte(100) or c == Byte(109):
        return 2
    if pos + 2 < n:
        var c2 = data[pos + 2]
        if c == Byte(108) and c2 == Byte(108): return 3
        if c == Byte(114) and c2 == Byte(101): return 3
        if c == Byte(118) and c2 == Byte(101): return 3
    return 0


def pre_tokenize_bytelevel_span(
    data: Span[Byte, _],
    start: Int,
    end: Int,
    letter_ranges: UnsafePointer[UInt32, _],
    number_ranges: UnsafePointer[UInt32, _],
    whitespace_ranges: UnsafePointer[UInt32, _],
    mut result: List[String],
):
    var i = start
    while i < end:
        var b = data[i]

        # 's|'t|'re|'ve|'m|'ll|'d
        if b == Byte(39):
            var clen = try_contraction(data, i, end)
            if clen > 0:
                result.append(span_to_string(data, i, i + clen))
                i += clen
                continue

        # ?\p{L}+
        if b == Byte(32) and i + 1 < end and is_letter_start_at(data, i + 1, end, letter_ranges):
            var letter_end = consume_letter_run(data, i + 1, end, letter_ranges)
            result.append(span_to_string(data, i, letter_end))
            i = letter_end
            continue
        if is_letter_start_at(data, i, end, letter_ranges):
            var letter_end = consume_letter_run(data, i, end, letter_ranges)
            result.append(span_to_string(data, i, letter_end))
            i = letter_end
            continue

        # ?\p{N}+
        if b == Byte(32) and i + 1 < end and is_number_start_at(data, i + 1, end, number_ranges):
            var number_end = consume_number_run(data, i + 1, end, number_ranges)
            result.append(span_to_string(data, i, number_end))
            i = number_end
            continue
        if is_number_start_at(data, i, end, number_ranges):
            var number_end = consume_number_run(data, i, end, number_ranges)
            result.append(span_to_string(data, i, number_end))
            i = number_end
            continue

        # ?[^\s\p{L}\p{N}]+
        if (
            b == Byte(32)
            and i + 1 < end
            and is_symbol_start_at(data, i + 1, end, letter_ranges, number_ranges, whitespace_ranges)
        ):
            var sym_end = consume_symbol_run(
                data, i + 1, end, letter_ranges, number_ranges, whitespace_ranges
            )
            result.append(span_to_string(data, i, sym_end))
            i = sym_end
            continue
        if is_symbol_start_at(data, i, end, letter_ranges, number_ranges, whitespace_ranges):
            var sym_end = consume_symbol_run(data, i, end, letter_ranges, number_ranges, whitespace_ranges)
            result.append(span_to_string(data, i, sym_end))
            i = sym_end
            continue

        # \s+(?!\S)
        if is_whitespace_start_at(data, i, end, whitespace_ranges):
            var ws_run = consume_whitespace_run(data, i, end, whitespace_ranges)
            var ws_end = ws_run[0]
            var ws_last_start = ws_run[1]
            var ws_count = ws_run[2]

            if ws_end == end:
                result.append(span_to_string(data, i, ws_end))
                i = ws_end
                continue
            if ws_count >= 2:
                result.append(span_to_string(data, i, ws_last_start))
                i = ws_last_start
                continue

            # \s+
            result.append(span_to_string(data, i, ws_end))
            i = ws_end
            continue

        # Safety fallback for malformed UTF-8 sequences.
        var parsed = decode_utf8_codepoint(data, i, end)
        result.append(span_to_string(data, i, i + parsed[1]))
        i += parsed[1]


def pre_tokenize(text: String) -> List[String]:
    var result = List[String]()
    var data = text.as_bytes()
    var n = len(data)
    if n == 0:
        return result^

    var letter_ranges = materialize[LETTER_RANGES]()
    var number_ranges = materialize[NUMBER_RANGES]()
    var whitespace_ranges = materialize[WHITESPACE_RANGES]()

    var letter_ptr = letter_ranges.unsafe_ptr()
    var number_ptr = number_ranges.unsafe_ptr()
    var whitespace_ptr = whitespace_ranges.unsafe_ptr()

    # Digits(individual_digits=true)
    var i = 0
    var chunk_start = 0
    while i < n:
        if is_number_start_at(data, i, n, number_ptr):
            var parsed = decode_utf8_codepoint(data, i, n)
            var cp_len = parsed[1]
            if chunk_start < i:
                pre_tokenize_bytelevel_span(
                    data, chunk_start, i,
                    letter_ptr, number_ptr, whitespace_ptr, result,
                )
            pre_tokenize_bytelevel_span(
                data, i, i + cp_len,
                letter_ptr, number_ptr, whitespace_ptr, result,
            )
            i += cp_len
            chunk_start = i
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]

    if chunk_start < n:
        pre_tokenize_bytelevel_span(
            data, chunk_start, n,
            letter_ptr, number_ptr, whitespace_ptr, result,
        )

    return result^

struct GPT2ByteTransform(ByteTransformCapability):
    def __init__(out self):
        pass


struct GPT2PreTokenizer(PreTokenizerCapability):
    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        return pre_tokenize(text)
