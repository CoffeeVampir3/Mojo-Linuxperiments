from std.memory import Span
from .tokenizer import (
    ByteTransformCapability, PreTokenizerCapability,
    UnicodeContext,
    is_ascii_letter,
    is_ascii_punct_symbol,
    is_ascii_regex_space,
    decode_utf8_codepoint,
    is_unicode_letter_cp,
    is_unicode_whitespace_cp,
    is_unicode_punct_symbol_cp,
    is_unicode_mark_cp,
    is_whitespace_start_at,
    span_to_string,
    consume_codepoint_run,
    split_numbers,
)


@always_inline
def is_newline_byte(b: Byte) -> Bool:
    return b == Byte(10) or b == Byte(13)


@always_inline
def is_cjk_japanese_cp(cp: UInt32) -> Bool:
    return (
        (cp >= UInt32(0x4E00) and cp <= UInt32(0x9FA5))
        or (cp >= UInt32(0x3040) and cp <= UInt32(0x309F))
        or (cp >= UInt32(0x30A0) and cp <= UInt32(0x30FF))
    )


@always_inline
def is_letter_mark_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    return is_unicode_letter_cp(cp, ctx) or is_unicode_mark_cp(cp, ctx)


@always_inline
def is_letter_mark_start_at(
    data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext,
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_letter_mark_cp(parsed[0], ctx)


@always_inline
def is_punct_symbol_start_at(
    data: Span[Byte, _], pos: Int, n: Int, ctx: UnicodeContext,
) -> Bool:
    var parsed = decode_utf8_codepoint(data, pos, n)
    return is_unicode_punct_symbol_cp(parsed[0], ctx)


@always_inline
def consume_letter_mark_run(
    data: Span[Byte, _], start: Int, n: Int, ctx: UnicodeContext,
) -> Int:
    var i = start
    while i < n:
        var parsed = decode_utf8_codepoint(data, i, n)
        if not is_letter_mark_cp(parsed[0], ctx):
            break
        i += parsed[1]
    return i


def consume_punct_symbol_run(
    data: Span[Byte, _], start: Int, n: Int, ctx: UnicodeContext,
) -> Int:
    return consume_codepoint_run[is_unicode_punct_symbol_cp](data, start, n, ctx)


def split_cjk_piece(piece: String, mut out: List[String]):
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
def is_branch_b_prefix_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    if cp == UInt32(10) or cp == UInt32(13):
        return False
    if is_unicode_letter_cp(cp, ctx):
        return False
    if is_unicode_punct_symbol_cp(cp, ctx):
        return False
    return True


def try_match_main_pattern(
    data: Span[Byte, _], pos: Int, end: Int, ctx: UnicodeContext,
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

    if is_branch_b_prefix_cp(cp, ctx):
        var j = pos + cp_len
        if j < end and is_letter_mark_start_at(data, j, end, ctx):
            return consume_letter_mark_run(data, j, end, ctx)

    if is_letter_mark_cp(cp, ctx):
        return consume_letter_mark_run(data, pos, end, ctx)

    #  ?[\p{P}\p{S}]+[\r\n]*
    var sym_start = pos
    if b == Byte(32):
        if pos + 1 < end and is_punct_symbol_start_at(data, pos + 1, end, ctx):
            sym_start = pos + 1
        else:
            sym_start = -1
    else:
        if not is_punct_symbol_start_at(data, pos, end, ctx):
            sym_start = -1

    if sym_start >= 0:
        var j = consume_punct_symbol_run(data, sym_start, end, ctx)
        while j < end and is_newline_byte(data[j]):
            j += 1
        return j

    # \s*[\r\n]+ | \s+(?!\S) | \s+
    if is_whitespace_start_at(data, pos, end, ctx):
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
            if not is_unicode_whitespace_cp(ws_parsed[0], ctx):
                break
            ws_end += ws_parsed[1]

        if last_newline_end >= 0:
            return last_newline_end
        if ws_end == end:
            return ws_end
        return ws_end

    return -1


def split_main_piece(
    piece: String, ctx: UnicodeContext, mut out: List[String],
):
    var data = piece.as_bytes()
    var n = len(data)
    if n == 0:
        return

    var i = 0
    var chunk_start = 0
    while i < n:
        var match_end = try_match_main_pattern(data, i, n, ctx)
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
    def __init__(out self):
        pass


struct DeepSeekV3PreTokenizer(PreTokenizerCapability):
    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        var result = List[String]()
        if text.byte_length() == 0:
            return result^

        var ctx = UnicodeContext()

        # Stage 1: Split numbers in groups of 1-3
        var stage1 = List[String]()
        split_numbers(text, 3, ctx, stage1)

        # Stage 2: Isolate CJK/Japanese runs
        var stage2 = List[String]()
        for piece in stage1:
            split_cjk_piece(String(piece), stage2)

        # Stage 3: Main pattern split
        for piece in stage2:
            split_main_piece(String(piece), ctx, result)

        return result^
