from .tokenizer import ByteTransformCapability, PreTokenizerCapability
from .gpt_oss import pre_tokenize_gpt_oss
from .NFC import nfc_normalize


struct MinimaxM27ByteTransform(ByteTransformCapability):
    def __init__(out self):
        pass


struct MinimaxM27PreTokenizer(PreTokenizerCapability):
    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        return pre_tokenize_gpt_oss(nfc_normalize(text))


def pre_tokenize_minimax_m27(text: String) -> List[String]:
    return pre_tokenize_gpt_oss(nfc_normalize(text))
