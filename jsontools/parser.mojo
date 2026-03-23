from std.collections import Dict
from std.memory import Span, UnsafePointer
from std.bit import count_trailing_zeros

# --- JSON byte constants ---

comptime QUOTE = Byte(34)
comptime BACKSLASH = Byte(92)
comptime CHAR_SLASH = Byte(47)
comptime LBRACE = Byte(123)
comptime RBRACE = Byte(125)
comptime LBRACKET = Byte(91)
comptime RBRACKET = Byte(93)
comptime COLON = Byte(58)
comptime COMMA = Byte(44)
comptime MINUS = Byte(45)
comptime PLUS = Byte(43)
comptime DOT = Byte(46)
comptime DIGIT_0 = Byte(48)
comptime ASCII_a = Byte(97)
comptime CHAR_b = Byte(98)
comptime CHAR_f = Byte(102)
comptime CHAR_n = Byte(110)
comptime CHAR_r = Byte(114)
comptime CHAR_t = Byte(116)
comptime CHAR_u = Byte(117)
comptime CHAR_E = Byte(69)
comptime CHAR_e = Byte(101)

# --- Lookup tables ---

def make_escape_table() -> InlineArray[Byte, 256]:
    var table = InlineArray[Byte, 256](fill=0)
    table[Int(QUOTE)] = QUOTE
    table[Int(BACKSLASH)] = BACKSLASH
    table[Int(CHAR_SLASH)] = CHAR_SLASH
    table[Int(CHAR_b)] = Byte(8)
    table[Int(CHAR_f)] = Byte(12)
    table[Int(CHAR_n)] = Byte(10)
    table[Int(CHAR_r)] = Byte(13)
    table[Int(CHAR_t)] = Byte(9)
    return table^

def make_hex_table() -> InlineArray[Int8, 256]:
    var table = InlineArray[Int8, 256](fill=-1)
    for i in range(10):
        table[Int(DIGIT_0) + i] = Int8(i)
    for i in range(6):
        table[Int(ASCII_a) + i] = Int8(10 + i)
        table[Int(Byte(65)) + i] = Int8(10 + i)
    return table^

comptime ESCAPE_TABLE = make_escape_table()
comptime HEX_TABLE = make_hex_table()

comptime CHAR_WHITESPACE: Byte = 1
comptime CHAR_DIGIT: Byte = 2
comptime CHAR_NUMBER_START: Byte = 4

def make_char_class_table() -> InlineArray[Byte, 256]:
    var table = InlineArray[Byte, 256](fill=0)
    table[9] = CHAR_WHITESPACE
    table[10] = CHAR_WHITESPACE
    table[13] = CHAR_WHITESPACE
    table[32] = CHAR_WHITESPACE
    for i in range(10):
        table[Int(DIGIT_0) + i] = CHAR_DIGIT | CHAR_NUMBER_START
    table[Int(MINUS)] = CHAR_NUMBER_START
    return table^

comptime CHAR_CLASS = make_char_class_table()

# --- Error type ---

@fieldwise_init
struct ParseError(Copyable, Writable):
    var message: String
    var pos: Int

# --- Scalar classification helpers ---

@always_inline
def is_whitespace(b: Byte) -> Bool:
    return (materialize[CHAR_CLASS]()[Int(b)] & CHAR_WHITESPACE) != 0

@always_inline
def is_digit(b: Byte) -> Bool:
    return (materialize[CHAR_CLASS]()[Int(b)] & CHAR_DIGIT) != 0

@always_inline
def is_number_start(b: Byte) -> Bool:
    return (materialize[CHAR_CLASS]()[Int(b)] & CHAR_NUMBER_START) != 0

# --- SIMD classification helpers ---

@always_inline
def simd_whitespace[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(9)).le(Byte(4)) | block.eq(Byte(32))

@always_inline
def simd_digits[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - DIGIT_0).le(Byte(9))

@always_inline
def simd_any_of2[w: Int](
    block: SIMD[DType.uint8, w],
    a: Byte,
    b: Byte,
) -> SIMD[DType.bool, w]:
    return block.eq(a) | block.eq(b)

@always_inline
def first_true_index[w: Int](mask: SIMD[DType.bool, w]) -> Int:
    var packed: UInt64 = 0
    comptime for i in range(w):
        packed |= UInt64(mask[i]) << UInt64(i)
    if packed == 0:
        return w
    return Int(count_trailing_zeros(packed))

@always_inline
def append_block_prefix[w: Int](
    mut out: List[Byte],
    block: SIMD[DType.uint8, w],
    count: Int,
):
    comptime for i in range(w):
        if i < count:
            out.append(block[i])

# --- Scalar string helpers ---

@always_inline
def hex_value(b: Byte) -> Int:
    return Int(materialize[HEX_TABLE]()[Int(b)])

def append_utf8(mut out: List[Byte], codepoint: Int):
    var cp = Codepoint(unsafe_unchecked_codepoint=UInt32(codepoint))
    var needed = cp.utf8_byte_length()
    var base = len(out)
    out.resize(unsafe_uninit_length=base + needed)
    var dst = out.unsafe_ptr() + base
    _ = cp.unsafe_write_utf8[True](dst)

@always_inline
def escape_value(esc: Byte) -> Byte:
    return materialize[ESCAPE_TABLE]()[Int(esc)]

@always_inline
def match_literal_at[lit: StringLiteral](
    ptr: UnsafePointer[Byte, ImmutAnyOrigin],
    pos: Int,
    length: Int,
) -> Bool:
    if pos + len(lit) > length:
        return False
    comptime bytes = StringSlice(lit).as_bytes()
    comptime for i in range(len(lit)):
        if ptr[pos + i] != bytes[i]:
            return False
    return True

# --- Generic JSON parser ---

struct Parser[origin: Origin, simd_width: Int = 16]:
    var data: Span[Byte, Self.origin]
    var pos: Int

    def __init__(out self, data: Span[Byte, Self.origin]):
        self.data = data
        self.pos = 0

    @always_inline
    def remaining(self) -> Int:
        return len(self.data) - self.pos

    @always_inline
    def has_more(self) -> Bool:
        return self.pos < len(self.data)

    @always_inline
    def peek(self) -> Byte:
        return self.data[self.pos]

    @always_inline
    def advance(mut self) -> Byte:
        var b = self.data[self.pos]
        self.pos += 1
        return b

    @always_inline
    def consume(mut self, expected: Byte) -> Bool:
        if self.has_more() and self.peek() == expected:
            self.pos += 1
            return True
        return False

    def skip_while_simd[
        pred_scalar: def(Byte) -> Bool,
        pred_simd: def[width: Int](SIMD[DType.uint8, width]) -> SIMD[DType.bool, width],
    ](mut self) -> Int:
        var start = self.pos
        var ptr = self.data.unsafe_ptr()
        while self.remaining() >= Self.simd_width:
            var block = (ptr + self.pos).load[width=Self.simd_width]()
            var matches = pred_simd[Self.simd_width](block)
            if all(matches):
                self.pos += Self.simd_width
                continue
            comptime for i in range(Self.simd_width):
                if not matches[i]:
                    self.pos += i
                    return self.pos - start
        while self.has_more() and pred_scalar(self.peek()):
            self.pos += 1
        return self.pos - start

    def skip_whitespace(mut self):
        _ = self.skip_while_simd[is_whitespace, simd_whitespace]()

    def skip_digits(mut self) -> Int:
        return self.skip_while_simd[is_digit, simd_digits]()

    def try_consume[lit: StringLiteral](mut self) -> Bool:
        if not match_literal_at[lit](self.data.unsafe_ptr(), self.pos, len(self.data)):
            return False
        self.pos += len(lit)
        return True

    def delimited_next(mut self, close: Byte) raises ParseError -> Bool:
        """Returns True if more items follow, False if delimiter closed."""
        self.skip_whitespace()
        if self.consume(close):
            return False
        if not self.consume(COMMA):
            raise ParseError("expected ',' or closing delimiter", self.pos)
        self.skip_whitespace()
        return True

    def object_key(mut self) raises ParseError -> String:
        var key = self.parse_string()
        self.skip_whitespace()
        if not self.consume(COLON):
            raise ParseError("expected ':'", self.pos)
        self.skip_whitespace()
        return key

    def parse_hex4(mut self) raises ParseError -> Int:
        var v = 0
        for _ in range(4):
            if not self.has_more():
                raise ParseError("unexpected end in hex escape", self.pos)
            var digit = hex_value(self.advance())
            if digit < 0:
                raise ParseError("invalid hex digit", self.pos)
            v = (v << 4) + digit
        return v

    def append_escape(mut self, mut out: List[Byte]) raises ParseError:
        if not self.has_more():
            raise ParseError("unexpected end in escape sequence", self.pos)
        var esc = self.advance()
        var mapped = escape_value(esc)
        if mapped != 0:
            out.append(mapped)
            return
        if esc != CHAR_u:
            raise ParseError("invalid escape character", self.pos)
        var cp = self.parse_hex4()
        if cp >= 0xD800 and cp <= 0xDBFF:
            if not (self.has_more() and self.peek() == BACKSLASH):
                raise ParseError("expected surrogate pair continuation", self.pos)
            self.pos += 1
            if not (self.has_more() and self.peek() == CHAR_u):
                raise ParseError("expected \\u in surrogate pair", self.pos)
            self.pos += 1
            var low = self.parse_hex4()
            if low < 0xDC00 or low > 0xDFFF:
                raise ParseError("invalid low surrogate", self.pos)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
        append_utf8(out, cp)

    def parse_string(mut self) raises ParseError -> String:
        if not self.consume(QUOTE):
            raise ParseError("expected '\"'", self.pos)
        var out_bytes = List[Byte]()
        var ptr = self.data.unsafe_ptr()
        while self.has_more():
            if self.remaining() >= Self.simd_width:
                var block = (ptr + self.pos).load[width=Self.simd_width]()
                var hits = simd_any_of2[Self.simd_width](block, QUOTE, BACKSLASH)
                if not any(hits):
                    append_block_prefix[Self.simd_width](out_bytes, block, Self.simd_width)
                    self.pos += Self.simd_width
                    continue
                var idx = first_true_index[Self.simd_width](hits)
                append_block_prefix[Self.simd_width](out_bytes, block, idx)
                self.pos += idx
            var b = self.advance()
            if b == QUOTE:
                if len(out_bytes) == 0:
                    return String("")
                var out_ptr = out_bytes.unsafe_ptr()
                return String(unsafe_from_utf8=Span[Byte, _](ptr=out_ptr, length=len(out_bytes)))
            if b == BACKSLASH:
                self.append_escape(out_bytes)
            else:
                out_bytes.append(b)
        raise ParseError("unterminated string", self.pos)

    def skip_number(mut self) raises ParseError:
        _ = self.consume(MINUS)
        if not self.has_more():
            raise ParseError("unexpected end in number", self.pos)
        if self.peek() == DIGIT_0:
            self.pos += 1
        elif self.skip_digits() == 0:
            raise ParseError("expected digit", self.pos)
        if self.consume(DOT) and self.skip_digits() == 0:
            raise ParseError("expected digit after '.'", self.pos)
        if self.has_more() and (self.peek() == CHAR_e or self.peek() == CHAR_E):
            self.pos += 1
            _ = self.consume(PLUS) or self.consume(MINUS)
            if self.skip_digits() == 0:
                raise ParseError("expected digit in exponent", self.pos)

    def parse_uint(mut self) raises ParseError -> Int:
        if not self.has_more() or not is_digit(self.peek()):
            raise ParseError("expected unsigned integer", self.pos)
        var start = self.pos
        var count = self.skip_digits()
        if count == 0:
            raise ParseError("expected digit", self.pos)
        var v = 0
        for i in range(count):
            v = v * 10 + Int(self.data[start + i] - DIGIT_0)
        return v

    def skip_value(mut self) raises ParseError:
        self.skip_whitespace()
        if not self.has_more():
            raise ParseError("unexpected end of input", self.pos)
        var b = self.peek()
        if b == QUOTE:
            _ = self.parse_string()
            return
        if b == LBRACE:
            self.skip_object()
            return
        if b == LBRACKET:
            self.skip_array()
            return
        if self.try_consume[lit="true"]() or self.try_consume[lit="false"]() or self.try_consume[lit="null"]():
            return
        if is_number_start(b):
            self.skip_number()
            return
        raise ParseError("unexpected character", self.pos)

    def skip_array(mut self) raises ParseError:
        if not self.consume(LBRACKET):
            raise ParseError("expected '['", self.pos)
        self.skip_whitespace()
        if self.consume(RBRACKET):
            return
        while True:
            self.skip_value()
            if not self.delimited_next(RBRACKET):
                return

    def skip_object(mut self) raises ParseError:
        if not self.consume(LBRACE):
            raise ParseError("expected '{'", self.pos)
        self.skip_whitespace()
        if self.consume(RBRACE):
            return
        while True:
            _ = self.object_key()
            self.skip_value()
            if not self.delimited_next(RBRACE):
                return

    def parse_bool(mut self) raises ParseError -> Bool:
        if self.try_consume[lit="true"]():
            return True
        if self.try_consume[lit="false"]():
            return False
        raise ParseError("expected 'true' or 'false'", self.pos)

    def parse_string_array(mut self) raises ParseError -> List[String]:
        """Parse a JSON array of strings: ["a", "b", ...]"""
        if not self.consume(LBRACKET):
            raise ParseError("expected '['", self.pos)
        self.skip_whitespace()
        var result = List[String]()
        if self.consume(RBRACKET):
            return result^
        while True:
            result.append(self.parse_string())
            if not self.delimited_next(RBRACKET):
                break
        return result^

    def parse_uint_array(mut self) raises ParseError -> List[Int]:
        """Parse a JSON array of unsigned integers: [1, 2, ...]"""
        if not self.consume(LBRACKET):
            raise ParseError("expected '['", self.pos)
        self.skip_whitespace()
        var result = List[Int]()
        if self.consume(RBRACKET):
            return result^
        while True:
            result.append(self.parse_uint())
            if not self.delimited_next(RBRACKET):
                break
        return result^

    def parse_string_uint_dict(mut self) raises ParseError -> Dict[String, Int]:
        """Parse a JSON object mapping strings to unsigned ints: {"a": 1, ...}"""
        if not self.consume(LBRACE):
            raise ParseError("expected '{'", self.pos)
        self.skip_whitespace()
        var result = Dict[String, Int]()
        if self.consume(RBRACE):
            return result^
        while True:
            var key = self.object_key()
            var val = self.parse_uint()
            result[key^] = val
            if not self.delimited_next(RBRACE):
                break
        return result^
