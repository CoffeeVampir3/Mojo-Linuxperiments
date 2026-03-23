from std.collections import Dict
from std.memory import Span, UnsafePointer
from std.pathlib import Path

from jsontools.parser import (
    Parser,
    ParseError,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
)

comptime HEADER_LEN_BYTES = 8
comptime MAX_HEADER_SIZE = 100 * 1024 * 1024

def parse_dtype(s: String) -> DType:
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

struct TensorMeta(Copyable, Writable):
    var dtype: DType
    var shape: List[Int]
    var start: Int
    var end: Int

    def __init__(out self, dtype: DType, var shape: List[Int], start: Int, end: Int):
        self.dtype = dtype
        self.shape = shape^
        self.start = start
        self.end = end

    def byte_size(self) -> Int:
        return self.end - self.start

    def numel(self) -> Int:
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

def parse_offsets(mut parser: Parser) raises ParseError -> Tuple[Int, Int]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for offsets", parser.pos)
    parser.skip_whitespace()
    var start_val = parser.parse_uint()
    if not parser.delimited_next(RBRACKET):
        raise ParseError("expected two offsets", parser.pos)
    var end_val = parser.parse_uint()
    parser.skip_whitespace()
    if not parser.consume(RBRACKET):
        raise ParseError("expected ']' after offsets", parser.pos)
    return (start_val, end_val)

def parse_shape(mut parser: Parser) raises ParseError -> List[Int]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for shape", parser.pos)
    parser.skip_whitespace()
    var shape = List[Int]()
    if parser.consume(RBRACKET):
        return shape^
    while True:
        shape.append(parser.parse_uint())
        if not parser.delimited_next(RBRACKET):
            break
    return shape^

def parse_tensor(mut parser: Parser) raises ParseError -> TensorMeta:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for tensor", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACE):
        raise ParseError("empty tensor object", parser.pos)
    var has_offsets = False
    var has_dtype = False
    var has_shape = False
    var start = 0
    var end = 0
    var dtype = DType.invalid
    var shape = List[Int]()
    while True:
        var key_val = parser.object_key()
        if key_val == "data_offsets":
            var offs = parse_offsets(parser)
            start = offs[0]
            end = offs[1]
            has_offsets = True
        elif key_val == "dtype":
            var dtype_str = parser.parse_string()
            dtype = parse_dtype(dtype_str)
            has_dtype = True
        elif key_val == "shape":
            shape = parse_shape(parser)
            has_shape = True
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break
    if not has_offsets or not has_dtype or not has_shape:
        raise ParseError("tensor missing required fields (dtype, shape, data_offsets)", parser.pos)
    return TensorMeta(dtype, shape^, start, end)

def parse_safetensors_dict(mut parser: Parser) raises ParseError -> Dict[String, TensorMeta]:
    var tensors = Dict[String, TensorMeta]()
    parser.skip_whitespace()
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' at start", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACE):
        return tensors^
    while True:
        var key_value = parser.object_key()
        if key_value == "__metadata__":
            parser.skip_value()
        else:
            var meta = parse_tensor(parser)
            if meta.end < meta.start:
                raise ParseError("invalid tensor offsets", parser.pos)
            tensors[key_value^] = meta^
        if not parser.delimited_next(RBRACE):
            break
    parser.skip_whitespace()
    if parser.has_more():
        raise ParseError("trailing content after root object", parser.pos)
    return tensors^

def read_u64_le(ptr: UnsafePointer[Byte, MutAnyOrigin]) -> UInt64:
    var v = UInt64(0)
    for i in range(HEADER_LEN_BYTES):
        v |= UInt64(ptr[i]) << UInt64(i * 8)
    return v

def parse_safetensors_header[simd_width: Int = 16](path: Path) -> Optional[SafetensorsHeader]:
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
    var parser = Parser(Span(header_bytes))
    try:
        var tensors = parse_safetensors_dict(parser)
        return SafetensorsHeader(path, tensors^, HEADER_LEN_BYTES + header_size, Int(file_len))
    except e:
        print("load: parse error at pos", e.pos, "-", e.message)
        return None
