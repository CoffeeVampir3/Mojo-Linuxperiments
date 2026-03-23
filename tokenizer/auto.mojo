from .capabilities import PreTokenizerCapability
from .deepseek_v3 import DeepSeekV3PreTokenizer
from .gpt2 import pre_tokenize as gpt2_pre_tokenize


comptime TOKENIZER_FLAVOR_UNSUPPORTED = 0
comptime TOKENIZER_FLAVOR_GPT2 = 1
comptime TOKENIZER_FLAVOR_DEEPSEEK_V3 = 2


struct AutoPreTokenizer(PreTokenizerCapability):
    var flavor: Int
    var deepseek: DeepSeekV3PreTokenizer

    def __init__(out self, flavor: Int):
        self.flavor = flavor
        self.deepseek = DeepSeekV3PreTokenizer()

    def pre_tokenize(self, text: String) -> List[String]:
        if self.flavor == TOKENIZER_FLAVOR_GPT2:
            return gpt2_pre_tokenize(text)
        if self.flavor == TOKENIZER_FLAVOR_DEEPSEEK_V3:
            return self.deepseek.pre_tokenize(text)

        var out = List[String]()
        out.append(text.copy())
        return out^
