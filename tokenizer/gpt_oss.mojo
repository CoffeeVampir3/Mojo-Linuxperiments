from std.memory import Span
from .tokenizer import (
    ByteTransformCapability, PreTokenizerCapability,
    UnicodeContext,
    is_ascii_letter,
    is_ascii_digit,
    is_ascii_regex_space,
    decode_utf8_codepoint,
    is_unicode_letter_cp,
    is_unicode_number_cp,
    is_unicode_whitespace_cp,
    is_unicode_mark_cp,
    in_comptime_ranges,
    span_to_string,
)
from .unicode_case_props import (
    LOWERCASE_RANGES,
    LOWERCASE_PAIR_COUNT,
    LOWERCASE_MIN,
    LOWERCASE_MAX,
    UPPERCASE_RANGES,
    UPPERCASE_PAIR_COUNT,
    UPPERCASE_MIN,
    UPPERCASE_MAX,
)


# =============================================================================
# Case classification
# =============================================================================


struct CaseContext(TrivialRegisterPassable):
    def __init__(out self):
        pass


@always_inline
def is_ascii_lowercase(b: Byte) -> Bool:
    return (b - Byte(97)) < Byte(26)


@always_inline
def is_ascii_uppercase(b: Byte) -> Bool:
    return (b - Byte(65)) < Byte(26)


@always_inline
def is_unicode_lowercase_cp(cp: UInt32, cc: CaseContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_lowercase(Byte(cp))
    if cp < LOWERCASE_MIN or cp > LOWERCASE_MAX:
        return False
    return in_comptime_ranges[LOWERCASE_RANGES, LOWERCASE_PAIR_COUNT](cp)


@always_inline
def is_unicode_uppercase_cp(cp: UInt32, cc: CaseContext) -> Bool:
    if cp < UInt32(0x80):
        return is_ascii_uppercase(Byte(cp))
    if cp < UPPERCASE_MIN or cp > UPPERCASE_MAX:
        return False
    return in_comptime_ranges[UPPERCASE_RANGES, UPPERCASE_PAIR_COUNT](cp)


@always_inline
def is_upper_like_cp(cp: UInt32, ctx: UnicodeContext, cc: CaseContext) -> Bool:
    """Matches [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}] = letter_or_mark AND NOT Ll."""
    if cp < UInt32(0x80):
        return is_ascii_uppercase(Byte(cp))
    if not (is_unicode_letter_cp(cp, ctx) or is_unicode_mark_cp(cp, ctx)):
        return False
    return not is_unicode_lowercase_cp(cp, cc)


@always_inline
def is_lower_like_cp(cp: UInt32, ctx: UnicodeContext, cc: CaseContext) -> Bool:
    """Matches [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}] = letter_or_mark AND NOT (Lu|Lt)."""
    if cp < UInt32(0x80):
        return is_ascii_lowercase(Byte(cp))
    if not (is_unicode_letter_cp(cp, ctx) or is_unicode_mark_cp(cp, ctx)):
        return False
    return not is_unicode_uppercase_cp(cp, cc)


@always_inline
def is_letter_or_mark_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    return is_unicode_letter_cp(cp, ctx) or is_unicode_mark_cp(cp, ctx)


# =============================================================================
# Prefix and contraction matching
# =============================================================================


@always_inline
def is_prefix_cp(cp: UInt32, ctx: UnicodeContext) -> Bool:
    """Matches [^\\r\\n\\p{L}\\p{N}] — not newline, not letter, not number."""
    if cp == UInt32(13) or cp == UInt32(10):
        return False
    if is_unicode_letter_cp(cp, ctx):
        return False
    if is_unicode_number_cp(cp, ctx):
        return False
    return True


@always_inline
def is_newline_byte(b: Byte) -> Bool:
    return b == Byte(10) or b == Byte(13)


@always_inline
def is_newline_or_slash_byte(b: Byte) -> Bool:
    return b == Byte(10) or b == Byte(13) or b == Byte(47)


def try_contraction_ci(data: Span[Byte, _], pos: Int, n: Int) -> Int:
    """Case-insensitive contraction match: (?i:'s|'t|'re|'ve|'m|'ll|'d)."""
    if pos + 1 >= n:
        return 0
    var q = data[pos]
    if q != Byte(39) and q != Byte(0xE2):
        return 0

    var apostrophe_len = 1
    if q == Byte(0xE2):
        if pos + 3 >= n:
            return 0
        if data[pos + 1] != Byte(0x80) or data[pos + 2] != Byte(0x99):
            return 0
        apostrophe_len = 3

    if pos + apostrophe_len >= n:
        return 0

    var c = data[pos + apostrophe_len] | Byte(0x20)
    if c == Byte(115) or c == Byte(116) or c == Byte(100) or c == Byte(109):
        return apostrophe_len + 1
    if pos + apostrophe_len + 1 < n:
        var c2 = data[pos + apostrophe_len + 1] | Byte(0x20)
        if c == Byte(108) and c2 == Byte(108): return apostrophe_len + 2
        if c == Byte(114) and c2 == Byte(101): return apostrophe_len + 2
        if c == Byte(118) and c2 == Byte(101): return apostrophe_len + 2
    return 0


# =============================================================================
# Core matching: try each alternative at a position
# =============================================================================


def try_match_letter_alts(
    data: Span[Byte, _], pos: Int, end: Int,
    ctx: UnicodeContext, cc: CaseContext,
) -> Int:
    """Try Alt 1 and Alt 2 (letter alternatives with optional prefix).

    Alt 1: [^\\r\\n\\p{L}\\p{N}]? [upper_like]* [lower_like]+ contraction?
    Alt 2: [^\\r\\n\\p{L}\\p{N}]? [upper_like]+ [lower_like]* contraction?
    """
    var i = pos

    # Optional prefix: [^\r\n\p{L}\p{N}]?
    var prefix_end = pos
    var parsed = decode_utf8_codepoint(data, i, end)
    var cp = parsed[0]
    var cp_len = parsed[1]
    if is_prefix_cp(cp, ctx):
        prefix_end = i + cp_len
        i = prefix_end
        if i >= end:
            return -1
        parsed = decode_utf8_codepoint(data, i, end)
        cp = parsed[0]

    # The first char after prefix must be letter or mark
    if not is_letter_or_mark_cp(cp, ctx):
        return -1

    # Consume upper_like run
    var upper_end = i
    while upper_end < end:
        parsed = decode_utf8_codepoint(data, upper_end, end)
        if not is_upper_like_cp(parsed[0], ctx, cc):
            break
        upper_end += parsed[1]

    var upper_count = upper_end - i

    # Consume lower_like run after upper
    var lower_end = upper_end
    while lower_end < end:
        parsed = decode_utf8_codepoint(data, lower_end, end)
        if not is_lower_like_cp(parsed[0], ctx, cc):
            break
        lower_end += parsed[1]

    var lower_count = lower_end - upper_end

    # Alt 1: upper* lower+ (need at least one lower)
    if lower_count > 0:
        var match_end = lower_end
        var clen = try_contraction_ci(data, match_end, end)
        match_end += clen
        if match_end > prefix_end:
            return match_end

    # Alt 2: upper+ lower* (need at least one upper)
    if upper_count > 0:
        var match_end = lower_end
        var clen = try_contraction_ci(data, match_end, end)
        match_end += clen
        if match_end > prefix_end:
            return match_end

    return -1


def try_match_numbers(
    data: Span[Byte, _], pos: Int, end: Int, ctx: UnicodeContext,
) -> Int:
    """Alt 3: \\p{N}{1,3}"""
    var parsed = decode_utf8_codepoint(data, pos, end)
    if not is_unicode_number_cp(parsed[0], ctx):
        return -1
    var i = pos + parsed[1]
    var count = 1
    while i < end and count < 3:
        parsed = decode_utf8_codepoint(data, i, end)
        if not is_unicode_number_cp(parsed[0], ctx):
            break
        i += parsed[1]
        count += 1
    return i


def try_match_symbols(
    data: Span[Byte, _], pos: Int, end: Int, ctx: UnicodeContext,
) -> Int:
    """Alt 4:  ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"""
    var i = pos

    # Optional space prefix
    if data[i] == Byte(32):
        i += 1
        if i >= end:
            return -1

    # Need at least one [^\s\p{L}\p{N}]
    var parsed = decode_utf8_codepoint(data, i, end)
    var cp = parsed[0]
    if is_unicode_whitespace_cp(cp, ctx) or is_unicode_letter_cp(cp, ctx) or is_unicode_number_cp(cp, ctx):
        return -1

    i += parsed[1]

    # Consume more [^\s\p{L}\p{N}]
    while i < end:
        parsed = decode_utf8_codepoint(data, i, end)
        cp = parsed[0]
        if is_unicode_whitespace_cp(cp, ctx) or is_unicode_letter_cp(cp, ctx) or is_unicode_number_cp(cp, ctx):
            break
        i += parsed[1]

    # Trailing [\r\n/]*
    while i < end and is_newline_or_slash_byte(data[i]):
        i += 1

    if i > pos:
        return i
    return -1


def try_match_newlines(
    data: Span[Byte, _], pos: Int, end: Int, ctx: UnicodeContext,
) -> Int:
    """Alt 5: \\s*[\\r\\n]+"""
    var i = pos
    var found_newline = False

    # \s* then \r\n+
    while i < end:
        var b = data[i]
        if is_newline_byte(b):
            found_newline = True
            i += 1
            continue
        if found_newline:
            break
        # Before first newline, consume whitespace
        if b < Byte(0x80):
            if is_ascii_regex_space(b):
                i += 1
                continue
            break
        var parsed = decode_utf8_codepoint(data, i, end)
        if is_unicode_whitespace_cp(parsed[0], ctx):
            i += parsed[1]
            continue
        break

    if found_newline:
        return i
    return -1


def try_match_whitespace(
    data: Span[Byte, _], pos: Int, end: Int, ctx: UnicodeContext,
) -> Int:
    """Alt 6 + 7: \\s+(?!\\S) | \\s+

    \\s+(?!\\S) greedily consumes whitespace then backtracks until the
    next character is whitespace or end-of-string. This leaves the last
    whitespace char available as a prefix for the following word token.
    \\s+ is the fallback that consumes any remaining single whitespace.
    """
    var parsed = decode_utf8_codepoint(data, pos, end)
    if not is_unicode_whitespace_cp(parsed[0], ctx):
        return -1

    var i = pos + parsed[1]
    var last_ws_start = pos
    while i < end:
        parsed = decode_utf8_codepoint(data, i, end)
        if not is_unicode_whitespace_cp(parsed[0], ctx):
            break
        last_ws_start = i
        i += parsed[1]

    # At end of input: consume all whitespace
    if i == end:
        return i

    # Followed by non-whitespace: backtrack to leave last ws char
    # for the next token's prefix — but only if we have 2+ ws chars
    if last_ws_start > pos:
        return last_ws_start

    # Single whitespace char followed by non-whitespace: \s+ matches it
    return i


def try_match_at(
    data: Span[Byte, _], pos: Int, end: Int,
    ctx: UnicodeContext, cc: CaseContext,
) -> Int:
    """Try all alternatives at position, return match end or -1."""
    # Alt 1 + 2: letter alternatives
    var m = try_match_letter_alts(data, pos, end, ctx, cc)
    if m > pos:
        return m

    # Alt 3: numbers
    m = try_match_numbers(data, pos, end, ctx)
    if m > pos:
        return m

    # Alt 4: symbols
    m = try_match_symbols(data, pos, end, ctx)
    if m > pos:
        return m

    # Alt 5: newlines
    m = try_match_newlines(data, pos, end, ctx)
    if m > pos:
        return m

    # Alt 6 + 7: whitespace
    m = try_match_whitespace(data, pos, end, ctx)
    if m > pos:
        return m

    return -1


# =============================================================================
# Pre-tokenizer
# =============================================================================


def pre_tokenize_gpt_oss(text: String) -> List[String]:
    var result = List[String]()
    var data = text.as_bytes()
    var n = len(data)
    if n == 0:
        return result^

    var ctx = UnicodeContext()
    var cc = CaseContext()

    var i = 0
    while i < n:
        var match_end = try_match_at(data, i, n, ctx, cc)
        if match_end > i:
            result.append(span_to_string(data, i, match_end))
            i = match_end
            continue

        # Safety fallback for unmatched bytes
        var parsed = decode_utf8_codepoint(data, i, n)
        result.append(span_to_string(data, i, i + parsed[1]))
        i += parsed[1]

    return result^


struct GptOssByteTransform(ByteTransformCapability):
    def __init__(out self):
        pass


struct GptOssPreTokenizer(PreTokenizerCapability):
    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        return pre_tokenize_gpt_oss(text)
