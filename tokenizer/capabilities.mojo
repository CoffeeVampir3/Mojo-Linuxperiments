from memory import Span


trait ByteTransformCapability(Movable, ImplicitlyDestructible):
    fn encode_bytes(self, data: Span[Byte]) -> String:
        ...

    fn decode_bytes(self, text: String) -> List[Byte]:
        ...


trait PreTokenizerCapability(Movable, ImplicitlyDestructible):
    fn pre_tokenize(self, text: String) -> List[String]:
        ...
