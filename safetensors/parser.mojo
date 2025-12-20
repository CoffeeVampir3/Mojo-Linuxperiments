from collections import Dict
from memory import Span, UnsafePointer
from pathlib import Path
from bit import count_trailing_zeros
from sys import argv

comptime HEADER_LEN_BYTES = 8
comptime MAX_HEADER_SIZE = 100 * 1024 * 1024

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

fn make_escape_table() -> InlineArray[Byte, 256]:
    var table = InlineArray[Byte, 256](fill=0)
    table[Int(QUOTE)] = QUOTE
    table[Int(BACKSLASH)] = BACKSLASH
    table[Int(CHAR_SLASH)] = CHAR_SLASH
    table[Int(CHAR_b)] = Byte(8)
    table[Int(CHAR_f)] = Byte(12)
    table[Int(CHAR_n)] = Byte(10)
    table[Int(CHAR_r)] = Byte(13)
    table[Int(CHAR_t)] = Byte(9)
    return table

fn make_hex_table() -> InlineArray[Int8, 256]:
    var table = InlineArray[Int8, 256](fill=-1)
    for i in range(10):
        table[Int(DIGIT_0) + i] = Int8(i)
    for i in range(6):
        table[Int(ASCII_a) + i] = Int8(10 + i)
        table[Int(Byte(65)) + i] = Int8(10 + i)
    return table

comptime ESCAPE_TABLE = make_escape_table()
comptime HEX_TABLE = make_hex_table()

comptime CHAR_WHITESPACE: Byte = 1
comptime CHAR_DIGIT: Byte = 2
comptime CHAR_NUMBER_START: Byte = 4

fn make_char_class_table() -> InlineArray[Byte, 256]:
    var table = InlineArray[Byte, 256](fill=0)
    table[9] = CHAR_WHITESPACE
    table[10] = CHAR_WHITESPACE
    table[13] = CHAR_WHITESPACE
    table[32] = CHAR_WHITESPACE
    for i in range(10):
        table[Int(DIGIT_0) + i] = CHAR_DIGIT | CHAR_NUMBER_START
    table[Int(MINUS)] = CHAR_NUMBER_START
    return table

comptime CHAR_CLASS = make_char_class_table()

fn parse_dtype(s: String) -> DType:
    if s == "BOOL":
        return DType.bool
    if s == "U8":
        return DType.uint8
    if s == "I8":
        return DType.int8
    if s == "I16":
        return DType.int16
    if s == "U16":
        return DType.uint16
    if s == "F16":
        return DType.float16
    if s == "BF16":
        return DType.bfloat16
    if s == "I32":
        return DType.int32
    if s == "U32":
        return DType.uint32
    if s == "F32":
        return DType.float32
    if s == "F64":
        return DType.float64
    if s == "I64":
        return DType.int64
    if s == "U64":
        return DType.uint64
    return DType.invalid

struct TensorMeta(Movable, Copyable):
    var dtype: DType
    var shape: List[Int]
    var start: Int
    var end: Int

    fn __init__(out self, dtype: DType, var shape: List[Int], start: Int, end: Int):
        self.dtype = dtype
        self.shape = shape^
        self.start = start
        self.end = end

    fn __copyinit__(out self, existing: Self):
        self.dtype = existing.dtype
        self.shape = existing.shape.copy()
        self.start = existing.start
        self.end = existing.end

    fn __moveinit__(out self, deinit existing: Self):
        self.dtype = existing.dtype
        self.shape = existing.shape^
        self.start = existing.start
        self.end = existing.end

    fn byte_size(self) -> Int:
        return self.end - self.start

    fn numel(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n

@fieldwise_init
struct SafetensorsHeader(Movable):
    var path: Path
    var tensors: Dict[String, TensorMeta]
    var data_offset: Int
    var file_len: Int

@always_inline
fn is_whitespace(b: Byte) -> Bool:
    return (CHAR_CLASS[Int(b)] & CHAR_WHITESPACE) != 0

@always_inline
fn is_digit(b: Byte) -> Bool:
    return (CHAR_CLASS[Int(b)] & CHAR_DIGIT) != 0

@always_inline
fn is_number_start(b: Byte) -> Bool:
    return (CHAR_CLASS[Int(b)] & CHAR_NUMBER_START) != 0

@always_inline
fn simd_whitespace[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - Byte(9)).le(Byte(4)) | block.eq(Byte(32))

@always_inline
fn simd_digits[w: Int](block: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (block - DIGIT_0).le(Byte(9))

@always_inline
fn simd_any_of2[w: Int](
    block: SIMD[DType.uint8, w],
    a: Byte,
    b: Byte,
) -> SIMD[DType.bool, w]:
    return block.eq(a) | block.eq(b)

@always_inline
fn first_true_index[w: Int](mask: SIMD[DType.bool, w]) -> Int:
    var packed: UInt64 = 0
    @parameter
    for i in range(w):
        packed |= UInt64(mask[i]) << i
    if packed == 0:
        return w
    return Int(count_trailing_zeros(packed))

@always_inline
fn append_block_prefix[w: Int](
    mut out: List[Byte],
    block: SIMD[DType.uint8, w],
    count: Int,
):
    @parameter
    for i in range(w):
        if i < count:
            out.append(block[i])

@always_inline
fn hex_value(b: Byte) -> Int:
    return Int(HEX_TABLE[Int(b)])

fn append_utf8(mut out: List[Byte], codepoint: Int):
    var cp = Codepoint(unsafe_unchecked_codepoint=UInt32(codepoint))
    var needed = cp.utf8_byte_length()
    var base = len(out)
    out.resize(unsafe_uninit_length=base + needed)
    var dst = out.unsafe_ptr() + base
    _ = cp.unsafe_write_utf8[True](dst)

@always_inline
fn escape_value(esc: Byte) -> Byte:
    return ESCAPE_TABLE[Int(esc)]

@always_inline
fn match_literal_at[lit: StringLiteral](
    ptr: UnsafePointer[Byte, MutAnyOrigin],
    pos: Int,
    length: Int,
) -> Bool:
    if pos + len(lit) > length:
        return False
    comptime bytes = StringSlice(lit).as_bytes()
    @parameter
    for i in range(len(lit)):
        if ptr[pos + i] != bytes[i]:
            return False
    return True

struct Parser[simd_width: Int = 16]:
    var ptr: UnsafePointer[Byte, MutAnyOrigin]
    var len: Int
    var pos: Int

    fn __init__(out self, ptr: UnsafePointer[Byte, MutAnyOrigin], length: Int):
        self.ptr = ptr
        self.len = length
        self.pos = 0

    @always_inline
    fn remaining(self) -> Int:
        return self.len - self.pos

    @always_inline
    fn has_more(self) -> Bool:
        return self.pos < self.len

    @always_inline
    fn peek(self) -> Byte:
        return self.ptr[self.pos]

    @always_inline
    fn advance(mut self) -> Byte:
        var b = self.ptr[self.pos]
        self.pos += 1
        return b

    @always_inline
    fn consume(mut self, expected: Byte) -> Bool:
        if self.has_more() and self.peek() == expected:
            self.pos += 1
            return True
        return False

    fn skip_while_simd[
        pred_scalar: fn(Byte) -> Bool,
        pred_simd: fn[width: Int](SIMD[DType.uint8, width]) -> SIMD[DType.bool, width],
    ](mut self) -> Int:
        var start = self.pos
        while self.remaining() >= Self.simd_width:
            var block = (self.ptr + self.pos).load[width=Self.simd_width]()
            var matches = pred_simd[Self.simd_width](block)
            if all(matches):
                self.pos += Self.simd_width
                continue
            @parameter
            for i in range(Self.simd_width):
                if not matches[i]:
                    self.pos += i
                    return self.pos - start
        while self.has_more() and pred_scalar(self.peek()):
            self.pos += 1
        return self.pos - start

    fn skip_whitespace(mut self):
        _ = self.skip_while_simd[is_whitespace, simd_whitespace]()

    fn delimited_next(mut self, close: Byte) -> Optional[Bool]:
        self.skip_whitespace()
        if self.consume(close):
            return False
        if not self.consume(COMMA):
            return None
        self.skip_whitespace()
        return True

    fn object_key(mut self) -> Optional[String]:
        var key = self.parse_string()
        if not key:
            return None
        self.skip_whitespace()
        if not self.consume(COLON):
            return None
        self.skip_whitespace()
        return key.value()

    fn try_consume[lit: StringLiteral](mut self) -> Bool:
        if not match_literal_at[lit](self.ptr, self.pos, self.len):
            return False
        self.pos += len(lit)
        return True

    fn parse_hex4(mut self) -> Int:
        var v = 0
        for _ in range(4):
            if not self.has_more():
                return -1
            var digit = hex_value(self.advance())
            if digit < 0:
                return -1
            v = (v << 4) + digit
        return v

    fn append_escape(mut self, mut out: List[Byte]) -> Bool:
        if not self.has_more():
            return False
        var esc = self.advance()
        var mapped = escape_value(esc)
        if mapped != 0:
            out.append(mapped)
            return True
        if esc != CHAR_u:
            return False
        var cp = self.parse_hex4()
        if cp < 0:
            return False
        if cp >= 0xD800 and cp <= 0xDBFF:
            if not (self.has_more() and self.peek() == BACKSLASH):
                return False
            self.pos += 1
            if not (self.has_more() and self.peek() == CHAR_u):
                return False
            self.pos += 1
            var low = self.parse_hex4()
            if low < 0:
                return False
            if low < 0xDC00 or low > 0xDFFF:
                return False
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
        append_utf8(out, cp)
        return True

    fn parse_string(mut self) -> Optional[String]:
        if not self.consume(QUOTE):
            return None
        var out_bytes = List[Byte]()
        while self.has_more():
            if self.remaining() >= Self.simd_width:
                var block = (self.ptr + self.pos).load[width=Self.simd_width]()
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
                var ptr = out_bytes.unsafe_ptr()
                return String(bytes=Span[Byte](ptr=ptr, length=len(out_bytes)))
            if b == BACKSLASH:
                if not self.append_escape(out_bytes):
                    return None
            else:
                out_bytes.append(b)
        return None

    fn skip_number(mut self) -> Bool:
        _ = self.consume(MINUS)
        if not self.has_more():
            return False
        if self.peek() == DIGIT_0:
            self.pos += 1
        elif self.skip_digits() == 0:
            return False
        if self.consume(DOT) and self.skip_digits() == 0:
            return False
        if self.has_more() and (self.peek() == CHAR_e or self.peek() == CHAR_E):
            self.pos += 1
            _ = self.consume(PLUS) or self.consume(MINUS)
            if self.skip_digits() == 0:
                return False
        return True

    fn skip_digits(mut self) -> Int:
        return self.skip_while_simd[is_digit, simd_digits]()

    fn parse_uint(mut self) -> Optional[Int]:
        if not self.has_more() or not is_digit(self.peek()):
            return None
        var start = self.pos
        var count = self.skip_digits()
        if count == 0:
            return None
        var v = 0
        for i in range(count):
            v = v * 10 + Int(self.ptr[start + i] - DIGIT_0)
        return v

    fn skip_value(mut self) -> Bool:
        self.skip_whitespace()
        if not self.has_more():
            return False
        var b = self.peek()
        if b == QUOTE:
            var tmp = self.parse_string()
            return Bool(tmp)
        if b == LBRACE:
            return self.skip_object()
        if b == LBRACKET:
            return self.skip_array()
        if self.try_consume[lit="true"]() or self.try_consume[lit="false"]() or self.try_consume[lit="null"]():
            return True
        if is_number_start(b):
            return self.skip_number()
        return False

    fn skip_array(mut self) -> Bool:
        if not self.consume(LBRACKET):
            return False
        self.skip_whitespace()
        if self.consume(RBRACKET):
            return True
        while True:
            if not self.skip_value():
                return False
            var more = self.delimited_next(RBRACKET)
            if not more:
                return False
            if not more.value():
                return True

    fn skip_object(mut self) -> Bool:
        if not self.consume(LBRACE):
            return False
        self.skip_whitespace()
        if self.consume(RBRACE):
            return True
        while True:
            if not self.object_key():
                return False
            if not self.skip_value():
                return False
            var more = self.delimited_next(RBRACE)
            if not more:
                return False
            if not more.value():
                return True

    fn parse_offsets(mut self) -> Optional[Tuple[Int, Int]]:
        if not self.consume(LBRACKET):
            return None
        self.skip_whitespace()
        var start_val = self.parse_uint()
        if not start_val:
            return None
        var more = self.delimited_next(RBRACKET)
        if not more or not more.value():
            return None
        var end_val = self.parse_uint()
        if not end_val:
            return None
        self.skip_whitespace()
        if not self.consume(RBRACKET):
            return None
        return (start_val.value(), end_val.value())

    fn parse_shape(mut self) -> Optional[List[Int]]:
        if not self.consume(LBRACKET):
            return None
        self.skip_whitespace()
        var shape = List[Int]()
        if self.consume(RBRACKET):
            return shape^
        while True:
            var dim = self.parse_uint()
            if not dim:
                return None
            shape.append(dim.value())
            var more = self.delimited_next(RBRACKET)
            if not more:
                return None
            if not more.value():
                break
        return shape^

    fn parse_tensor(mut self) -> Optional[TensorMeta]:
        if not self.consume(LBRACE):
            return None
        self.skip_whitespace()
        if self.consume(RBRACE):
            return None
        var has_offsets = False
        var has_dtype = False
        var has_shape = False
        var start = 0
        var end = 0
        var dtype = DType.invalid
        var shape = List[Int]()
        while True:
            var key = self.object_key()
            if not key:
                return None
            var key_val = key.value()
            if key_val == "data_offsets":
                var offsets = self.parse_offsets()
                if not offsets:
                    return None
                var offs = offsets.value()
                start = offs[0]
                end = offs[1]
                has_offsets = True
            elif key_val == "dtype":
                var dtype_str = self.parse_string()
                if not dtype_str:
                    return None
                dtype = parse_dtype(dtype_str.value())
                has_dtype = True
            elif key_val == "shape":
                var shape_opt = self.parse_shape()
                if not shape_opt:
                    return None
                shape = shape_opt.take()
                has_shape = True
            else:
                if not self.skip_value():
                    return None
            var more = self.delimited_next(RBRACE)
            if not more:
                return None
            if not more.value():
                break
        if not has_offsets or not has_dtype or not has_shape:
            return None
        return TensorMeta(dtype, shape^, start, end)

    fn parse(mut self) -> Optional[Dict[String, TensorMeta]]:
        var tensors = Dict[String, TensorMeta]()
        self.skip_whitespace()
        if not self.consume(LBRACE):
            return None
        self.skip_whitespace()
        if self.consume(RBRACE):
            return tensors^
        while True:
            var key = self.object_key()
            if not key:
                return None
            var key_value = key.value()
            if key_value == "__metadata__":
                if not self.skip_value():
                    return None
            else:
                var tensor = self.parse_tensor()
                if not tensor:
                    return None
                var meta = tensor.take()
                if meta.end < meta.start:
                    return None
                tensors[key_value^] = meta^
            var more = self.delimited_next(RBRACE)
            if not more:
                return None
            if not more.value():
                break
        self.skip_whitespace()
        if self.has_more():
            return None
        return tensors^

fn read_u64_le(ptr: UnsafePointer[Byte, MutAnyOrigin]) -> UInt64:
    var v = UInt64(0)
    for i in range(HEADER_LEN_BYTES):
        v |= UInt64(ptr[i]) << UInt64(i * 8)
    return v

fn parse_safetensors_header[simd_width: Int = 16](path: Path) -> Optional[SafetensorsHeader]:
    var header_bytes: List[Byte]
    var header_size = 0
    var file_len: UInt64
    try:
        with open(path, "r") as f:
            file_len = f.seek(0, 2)
            _ = f.seek(0, 0)
            if file_len < UInt64(HEADER_LEN_BYTES):
                print("load: file too small")
                return None
            var header_len_bytes = f.read_bytes(size=HEADER_LEN_BYTES)
            if len(header_len_bytes) != HEADER_LEN_BYTES:
                print("load: file too small")
                return None
            var header_len = read_u64_le(header_len_bytes.unsafe_ptr())
            if header_len > UInt64(MAX_HEADER_SIZE):
                print("load: header too large")
                return None
            if header_len > file_len - UInt64(HEADER_LEN_BYTES):
                print("load: header length exceeds file")
                return None
            header_size = Int(header_len)
            header_bytes = List[Byte](unsafe_uninit_length=header_size)
            var bytes_read = f.read(Span(header_bytes))
            if bytes_read != header_size:
                print("load: header length exceeds file")
                return None
    except e:
        print("load: failed to read file:", e)
        return None
    var ptr = header_bytes.unsafe_ptr()
    var parser = Parser[simd_width](ptr, header_size)
    var tensors = parser.parse()
    _ = header_bytes  # Lifetime extension
    if tensors:
        return SafetensorsHeader(path, tensors.take(), HEADER_LEN_BYTES + header_size, Int(file_len))
    return None

fn main():
    var args = argv()
    if len(args) < 2:
        print("usage:", args[0], "<safetensors_file>")
        return
    var result = parse_safetensors_header(Path(args[1]))
    if result:
        var header = result.take()
        print("File:", header.path)
        print("File size:", header.file_len, "bytes")
        print("Tensors:", len(header.tensors))
        print("Data offset:", header.data_offset)
        for item in header.tensors.items():
            var name = item.key
            var meta = item.value.copy()
            var start = header.data_offset + meta.start
            var end = header.data_offset + meta.end
            print(" ", name, meta.dtype, meta.shape, "-", meta.byte_size(), "bytes @", start, "..", end)
    else:
        print("Parse failed")
