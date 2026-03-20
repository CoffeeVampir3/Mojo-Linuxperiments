from memory import Span

from .capabilities import ByteTransformCapability, PreTokenizerCapability
from .shared_capabilities import (
    bytes_to_gpt2,
    gpt2_to_bytes,
    is_ascii_letter,
    is_ascii_digit,
    is_ascii_regex_space,
    decode_utf8_codepoint,
    in_unicode_ranges,
    is_unicode_letter_cp,
    is_unicode_number_cp,
    is_unicode_whitespace_cp,
    is_number_start_at,
    is_whitespace_start_at,
    span_to_string,
)
from .unicode_props import (
    LETTER_RANGES,
    NUMBER_RANGES,
    WHITESPACE_RANGES,
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


@always_inline
fn is_ascii_punct_symbol(b: Byte) -> Bool:
    return (
        (b >= Byte(33) and b <= Byte(47))
        or (b >= Byte(58) and b <= Byte(64))
        or (b >= Byte(91) and b <= Byte(96))
        or (b >= Byte(123) and b <= Byte(126))
    )


@always_inline
fn is_newline_byte(b: Byte) -> Bool:
    return b == Byte(10) or b == Byte(13)


@always_inline
fn is_unicode_mark_cp(cp: UInt32, mark_ranges: UnsafePointer[UInt32]) -> Bool:
    if cp < MARK_MIN or cp > MARK_MAX:
        return False
    return in_unicode_ranges(cp, mark_ranges, MARK_PAIR_COUNT)


@always_inline
fn is_unicode_punct_symbol_cp(cp: UInt32, punct_symbol_ranges: UnsafePointer[UInt32]) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_punct_symbol(Byte(cp))
    if cp < PUNCT_SYMBOL_MIN or cp > PUNCT_SYMBOL_MAX:
        return False
    return in_unicode_ranges(cp, punct_symbol_ranges, PUNCT_SYMBOL_PAIR_COUNT)


@always_inline
fn is_cjk_japanese_cp(cp: UInt32) -> Bool:
    return (
        (cp >= UInt32(0x4E00) and cp <= UInt32(0x9FA5))
        or (cp >= UInt32(0x3040) and cp <= UInt32(0x309F))
        or (cp >= UInt32(0x30A0) and cp <= UInt32(0x30FF))
    )


@always_inline
fn is_letter_mark_cp(
    cp: UInt32,
    letter_ranges: UnsafePointer[UInt32],
    mark_ranges: UnsafePointer[UInt32],
) -> Bool:
    return is_unicode_letter_cp(cp, letter_ranges) or is_unicode_mark_cp(cp, mark_ranges)


@always_inline
fn is_letter_mark_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
    mark_ranges: UnsafePointer[UInt32],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_letter_mark_cp(parsed[0], letter_ranges, mark_ranges)


@always_inline
fn is_punct_symbol_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    punct_symbol_ranges: UnsafePointer[UInt32],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_punct_symbol_cp(parsed[0], punct_symbol_ranges)


@always_inline
fn consume_letter_mark_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
    mark_ranges: UnsafePointer[UInt32],
) -> Int:
    var i = start
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if not is_letter_mark_cp(parsed[0], letter_ranges, mark_ranges):
            break
        i += parsed[1]
    return i


@always_inline
fn consume_punct_symbol_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    punct_symbol_ranges: UnsafePointer[UInt32],
) -> Int:
    var i = start
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if not is_unicode_punct_symbol_cp(parsed[0], punct_symbol_ranges):
            break
        i += parsed[1]
    return i


fn split_numbers_piece(
    piece: String,
    number_ranges: UnsafePointer[UInt32],
    mut out: List[String],
):
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        if is_number_start_at(data, i, n, number_ranges):
            if chunk_start < i:
                out.append(span_to_string(data, chunk_start, i))

            var j = i
            var count = 0
            while j < n and count < 3 and is_number_start_at(data, j, n, number_ranges):
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


fn split_cjk_piece(piece: String, mut out: List[String]):
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if is_cjk_japanese_cp(parsed[0]):
            if chunk_start < i:
                out.append(span_to_string(data, chunk_start, i))

            var j = i + parsed[1]
            while j < n:
                var parsed_next = decode_utf8_codepoint(data, j, n)
                if not is_cjk_japanese_cp(parsed_next[0]):
                    break
                j += parsed_next[1]

            out.append(span_to_string(data, i, j))
            i = j
            chunk_start = i
            continue

        i += parsed[1]

    if chunk_start < n:
        out.append(span_to_string(data, chunk_start, n))


@always_inline
fn is_branch_b_prefix_cp(
    cp: UInt32,
    letter_ranges: UnsafePointer[UInt32],
    punct_symbol_ranges: UnsafePointer[UInt32],
) -> Bool:
    if cp == UInt32(10) or cp == UInt32(13):
        return False
    if is_unicode_letter_cp(cp, letter_ranges):
        return False
    if is_unicode_punct_symbol_cp(cp, punct_symbol_ranges):
        return False
    return True


fn try_match_main_pattern(
    data: Span[Byte],
    pos: Int,
    end: Int,
    letter_ranges: UnsafePointer[UInt32],
    mark_ranges: UnsafePointer[UInt32],
    punct_symbol_ranges: UnsafePointer[UInt32],
    whitespace_ranges: UnsafePointer[UInt32],
) -> Int:
    var b = data[pos]

    # [!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~][A-Za-z]+
    if is_ascii_punct_symbol(b) and pos + 1 < end and is_ascii_letter(data[pos + 1]):
        var j = pos + 2
        while j < end and is_ascii_letter(data[j]):
            j += 1
        return j

    # [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    var parsed = decode_utf8_codepoint(data, pos, end)
    var cp = parsed[0]
    var cp_len = parsed[1]

    if is_branch_b_prefix_cp(cp, letter_ranges, punct_symbol_ranges):
        var j = pos + cp_len
        if (
            j < end
            and is_letter_mark_start_at(data, j, end, letter_ranges, mark_ranges)
        ):
            return consume_letter_mark_run(data, j, end, letter_ranges, mark_ranges)

    if is_letter_mark_cp(cp, letter_ranges, mark_ranges):
        return consume_letter_mark_run(data, pos, end, letter_ranges, mark_ranges)

    #  ?[\p{P}\p{S}]+[\r\n]*
    var sym_start = pos
    if b == Byte(32):
        if pos + 1 < end and is_punct_symbol_start_at(data, pos + 1, end, punct_symbol_ranges):
            sym_start = pos + 1
        else:
            sym_start = -1
    else:
        if not is_punct_symbol_start_at(data, pos, end, punct_symbol_ranges):
            sym_start = -1

    if sym_start >= 0:
        var j = consume_punct_symbol_run(data, sym_start, end, punct_symbol_ranges)
        while j < end and is_newline_byte(data[j]):
            j += 1
        return j

    # \s*[\r\n]+ | \s+(?!\S) | \s+
    if is_whitespace_start_at(data, pos, end, whitespace_ranges):
        var ws_end = pos
        var last_newline_end = -1
        while ws_end < end:
            var wb = data[ws_end]
            if wb < Byte(0x80):
                if not is_ascii_regex_space(wb):
                    break
                ws_end += 1
                if is_newline_byte(wb):
                    last_newline_end = ws_end
                continue

            var ws_parsed = decode_utf8_codepoint(data, ws_end, end)
            if not is_unicode_whitespace_cp(ws_parsed[0], whitespace_ranges):
                break
            ws_end += ws_parsed[1]

        if last_newline_end >= 0:
            return last_newline_end
        if ws_end == end:
            return ws_end
        return ws_end

    return -1


fn split_main_piece(
    piece: String,
    letter_ranges: UnsafePointer[UInt32],
    mark_ranges: UnsafePointer[UInt32],
    punct_symbol_ranges: UnsafePointer[UInt32],
    whitespace_ranges: UnsafePointer[UInt32],
    mut out: List[String],
):
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        var match_end = try_match_main_pattern(
            data,
            i,
            n,
            letter_ranges,
            mark_ranges,
            punct_symbol_ranges,
            whitespace_ranges,
        )
        if match_end > i:
            if chunk_start < i:
                out.append(span_to_string(data, chunk_start, i))
            out.append(span_to_string(data, i, match_end))
            i = match_end
            chunk_start = i
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]

    if chunk_start < n:
        out.append(span_to_string(data, chunk_start, n))


struct DeepSeekV3ByteTransform(ByteTransformCapability):
    fn __init__(out self):
        pass

    fn encode_bytes(self, data: Span[Byte]) -> String:
        return bytes_to_gpt2(data)

    fn decode_bytes(self, text: String) -> List[Byte]:
        return gpt2_to_bytes(text)


struct DeepSeekV3PreTokenizer(PreTokenizerCapability):
    fn __init__(out self):
        pass

    fn pre_tokenize(self, text: String) -> List[String]:
        var result = List[String]()
        if text.byte_length() == 0:
            return result^

        var number_ranges = materialize[NUMBER_RANGES]()
        var letter_ranges = materialize[LETTER_RANGES]()
        var whitespace_ranges = materialize[WHITESPACE_RANGES]()
        var mark_ranges = materialize[MARK_RANGES]()
        var punct_symbol_ranges = materialize[PUNCT_SYMBOL_RANGES]()

        var number_ptr = number_ranges.unsafe_ptr()
        var letter_ptr = letter_ranges.unsafe_ptr()
        var whitespace_ptr = whitespace_ranges.unsafe_ptr()
        var mark_ptr = mark_ranges.unsafe_ptr()
        var punct_symbol_ptr = punct_symbol_ranges.unsafe_ptr()

        var stage1 = List[String]()
        split_numbers_piece(text, number_ptr, stage1)

        var stage2 = List[String]()
        for i in range(len(stage1)):
            split_cjk_piece(stage1[i], stage2)

        for i in range(len(stage2)):
            split_main_piece(
                stage2[i],
                letter_ptr,
                mark_ptr,
                punct_symbol_ptr,
                whitespace_ptr,
                result,
            )

        return result^
