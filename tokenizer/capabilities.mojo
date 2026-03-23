from memory import Span

from .shared_capabilities import bytes_to_gpt2, gpt2_to_bytes


trait ByteTransformCapability(Movable, ImplicitlyDestructible):
    fn encode_bytes(self, data: Span[Byte]) -> String:
        return bytes_to_gpt2(data)

    fn decode_bytes(self, text: String) -> List[Byte]:
        return gpt2_to_bytes(text)


trait PreTokenizerCapability(Movable, ImplicitlyDestructible):
    fn pre_tokenize(self, text: String) -> List[String]:
        ...
