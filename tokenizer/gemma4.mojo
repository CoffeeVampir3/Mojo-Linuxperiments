from std.memory import Span
from .tokenizer import ByteTransformCapability, PreTokenizerCapability, span_to_string


def gemma4_encode_bytes(data: Span[Byte, _]) -> String:
    return span_to_string(data, 0, len(data))


@always_inline
def hex_nibble_value(b: Byte) -> Int:
    if b >= Byte(48) and b <= Byte(57):
        return Int(b) - 48
    if b >= Byte(65) and b <= Byte(70):
        return Int(b) - 55
    if b >= Byte(97) and b <= Byte(102):
        return Int(b) - 87
    return -1


def gemma4_decode_bytes(text: String) -> List[Byte]:
    var data = text.as_bytes()
    var n = len(data)
    var out = List[Byte]()
    var i = 0
    while i < n:
        if (
            i + 5 < n
            and data[i] == Byte(0x3C)
            and data[i + 1] == Byte(0x30)
            and data[i + 2] == Byte(0x78)
            and data[i + 5] == Byte(0x3E)
        ):
            var hi = hex_nibble_value(data[i + 3])
            var lo = hex_nibble_value(data[i + 4])
            if hi >= 0 and lo >= 0:
                out.append(Byte(hi * 16 + lo))
                i += 6
                continue

        if (
            i + 2 < n
            and data[i] == Byte(0xE2)
            and data[i + 1] == Byte(0x96)
            and data[i + 2] == Byte(0x81)
        ):
            out.append(Byte(0x20))
            i += 3
            continue

        out.append(data[i])
        i += 1
    return out^


def pre_tokenize_gemma4(text: String) -> List[String]:
    var result = List[String]()
    if text.byte_length() == 0:
        return result^

    var data = text.as_bytes()
    var n = len(data)
    var buf = List[Byte]()
    for i in range(n):
        if data[i] == Byte(0x20):
            buf.append(Byte(0xE2))
            buf.append(Byte(0x96))
            buf.append(Byte(0x81))
        else:
            buf.append(data[i])
    result.append(String(unsafe_from_utf8=Span[Byte, _](ptr=buf.unsafe_ptr(), length=len(buf))))
    return result^


struct Gemma4ByteTransform(ByteTransformCapability):
    def __init__(out self):
        pass

    def encode_bytes(self, data: Span[Byte, _]) -> String:
        return gemma4_encode_bytes(data)

    def decode_bytes(self, text: String) -> List[Byte]:
        return gemma4_decode_bytes(text)


struct Gemma4PreTokenizer(PreTokenizerCapability):
    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        return pre_tokenize_gemma4(text)
