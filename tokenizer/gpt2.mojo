from memory import Span
from .capabilities import ByteTransformCapability, PreTokenizerCapability
from .shared_capabilities import (
    bytes_to_gpt2,
    gpt2_to_bytes,
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
)
from .unicode_props import (
    LETTER_RANGES,
    NUMBER_RANGES,
    WHITESPACE_RANGES,
)

comptime PRETOKENIZE_SIMD_WIDTH = 16


@always_inline
fn simd_ascii_letters[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return ((block | Byte(0x20)) - Byte(97)).le(Byte(25))


@always_inline
fn simd_ascii_digits[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(48)).le(Byte(9))


@always_inline
fn simd_spaces[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(9)).le(Byte(4)) | block.eq(Byte(32))


@always_inline
fn is_letter_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_letter_cp(parsed[0], letter_ranges)


@always_inline
fn is_symbol_start_at(
    data: Span[Byte],
    pos: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
    number_ranges: UnsafePointer[UInt32],
    whitespace_ranges: UnsafePointer[UInt32],
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


fn sort_strings_by_byte_length_desc(mut values: List[String]):
    for i in range(1, len(values)):
        var cur = values[i]
        var cur_len = cur.byte_length()
        var j = i
        while j > 0 and values[j - 1].byte_length() < cur_len:
            values[j] = values[j - 1]
            j -= 1
        values[j] = cur


@always_inline
fn span_matches_at(data: Span[Byte], pos: Int, pattern: Span[Byte]) -> Bool:
    if pos + len(pattern) > len(data):
        return False
    for i in range(len(pattern)):
        if data[pos + i] != pattern[i]:
            return False
    return True


fn find_added_token_match(
    text: Span[Byte],
    pos: Int,
    added_token_order: List[String],
    added_tokens: Dict[String, Int],
) -> Tuple[Int, Int]:
    for i in range(len(added_token_order)):
        var tok = added_token_order[i]
        var tok_bytes = tok.as_bytes()
        if span_matches_at(text, pos, tok_bytes):
            var found = added_tokens.get(tok)
            if found:
                return (found.value(), len(tok_bytes))
    return (-1, 0)


@always_inline
fn skip_ascii_letters_simd(data: Span[Byte], pos: Int, n: Int) -> Int:
    var i = pos
    var ptr = data.unsafe_ptr()
    while i + PRETOKENIZE_SIMD_WIDTH <= n:
        var block = (ptr + i).load[width=PRETOKENIZE_SIMD_WIDTH]()
        var mask = simd_ascii_letters[PRETOKENIZE_SIMD_WIDTH](block)
        if all(mask):
            i += PRETOKENIZE_SIMD_WIDTH
            continue
        @parameter
        for lane in range(PRETOKENIZE_SIMD_WIDTH):
            if not mask[lane]:
                return i + lane
    while i < n and is_ascii_letter(data[i]):
        i += 1
    return i


@always_inline
fn skip_ascii_digits_simd(data: Span[Byte], pos: Int, n: Int) -> Int:
    var i = pos
    var ptr = data.unsafe_ptr()
    while i + PRETOKENIZE_SIMD_WIDTH <= n:
        var block = (ptr + i).load[width=PRETOKENIZE_SIMD_WIDTH]()
        var mask = simd_ascii_digits[PRETOKENIZE_SIMD_WIDTH](block)
        if all(mask):
            i += PRETOKENIZE_SIMD_WIDTH
            continue
        @parameter
        for lane in range(PRETOKENIZE_SIMD_WIDTH):
            if not mask[lane]:
                return i + lane
    while i < n and is_ascii_digit(data[i]):
        i += 1
    return i


@always_inline
fn skip_spaces_simd(data: Span[Byte], pos: Int, n: Int) -> Int:
    var i = pos
    var ptr = data.unsafe_ptr()
    while i + PRETOKENIZE_SIMD_WIDTH <= n:
        var block = (ptr + i).load[width=PRETOKENIZE_SIMD_WIDTH]()
        var mask = simd_spaces[PRETOKENIZE_SIMD_WIDTH](block)
        if all(mask):
            i += PRETOKENIZE_SIMD_WIDTH
            continue
        @parameter
        for lane in range(PRETOKENIZE_SIMD_WIDTH):
            if not mask[lane]:
                return i + lane
    while i < n and is_ascii_regex_space(data[i]):
        i += 1
    return i


@always_inline
fn consume_letter_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
) -> Int:
    var i = start
    while i < n:
        var b = data[i]
        if is_ascii_letter(b):
            i = skip_ascii_letters_simd(data, i, n)
            continue
        var parsed = decode_utf8_codepoint(data, i, n)
        if is_unicode_letter_cp(parsed[0], letter_ranges):
            i += parsed[1]
            continue
        break
    return i


@always_inline
fn consume_number_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    number_ranges: UnsafePointer[UInt32],
) -> Int:
    var i = start
    while i < n:
        if is_ascii_digit(data[i]):
            i = skip_ascii_digits_simd(data, i, n)
            continue
        var parsed = decode_utf8_codepoint(data, i, n)
        if is_unicode_number_cp(parsed[0], number_ranges):
            i += parsed[1]
            continue
        break
    return i


@always_inline
fn consume_symbol_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    letter_ranges: UnsafePointer[UInt32],
    number_ranges: UnsafePointer[UInt32],
    whitespace_ranges: UnsafePointer[UInt32],
) -> Int:
    var i = start
    while i < n:
        if not is_symbol_start_at(data, i, n, letter_ranges, number_ranges, whitespace_ranges):
            break
        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]
    return i


@always_inline
fn consume_whitespace_run(
    data: Span[Byte],
    start: Int,
    n: Int,
    whitespace_ranges: UnsafePointer[UInt32],
) -> Tuple[Int, Int, Int]:
    var i = start
    var last_cp_start = start
    var cp_count = 0

    while i < n:
        var b = data[i]
        if b < Byte(0x80):
            if not is_ascii_regex_space(b):
                break
            var ascii_end = skip_spaces_simd(data, i, n)
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


fn try_contraction(data: Span[Byte], pos: Int, n: Int) -> Int:
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


fn pre_tokenize_bytelevel_span(
    data: Span[Byte],
    start: Int,
    end: Int,
    letter_ranges: UnsafePointer[UInt32],
    number_ranges: UnsafePointer[UInt32],
    whitespace_ranges: UnsafePointer[UInt32],
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


fn pre_tokenize(text: String) -> List[String]:
    var result = List[String]()
    var data = text.as_bytes()
    var n = len(data)
    if n == 0:
        return result^

    # Match HF pre-tokenizer semantics exactly:
    # Sequence([Digits(individual_digits=true), ByteLevel(use_regex=true, add_prefix_space=false)])
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
                    data,
                    chunk_start,
                    i,
                    letter_ptr,
                    number_ptr,
                    whitespace_ptr,
                    result,
                )
            pre_tokenize_bytelevel_span(
                data,
                i,
                i + cp_len,
                letter_ptr,
                number_ptr,
                whitespace_ptr,
                result,
            )
            i += cp_len
            chunk_start = i
            continue

        var parsed = decode_utf8_codepoint(data, i, n)
        i += parsed[1]

    if chunk_start < n:
        pre_tokenize_bytelevel_span(
            data,
            chunk_start,
            n,
            letter_ptr,
            number_ptr,
            whitespace_ptr,
            result,
        )

    return result^

struct GPT2ByteTransform(ByteTransformCapability):
    fn __init__(out self):
        pass

    fn encode_bytes(self, data: Span[Byte]) -> String:
        return bytes_to_gpt2(data)

    fn decode_bytes(self, text: String) -> List[Byte]:
        return gpt2_to_bytes(text)


struct GPT2PreTokenizer(PreTokenizerCapability):
    fn __init__(out self):
        pass

    fn pre_tokenize(self, text: String) -> List[String]:
        return pre_tokenize(text)
