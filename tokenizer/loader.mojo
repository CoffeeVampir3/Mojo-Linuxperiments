from std.pathlib import Path
from std.memory import Span
from std.collections import Dict

from jsontools.parser import (
    Parser,
    ParseError,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    QUOTE,
)
from .tokenizer import (
    BPETokenizer, ByteTransformCapability, PreTokenizerCapability,
    span_to_string, bytes_to_gpt2, gpt2_to_bytes,
)
from .bpe import pack_pair_ids
from .deepseek_v3 import DeepSeekV3ByteTransform, DeepSeekV3PreTokenizer
from .gpt2 import GPT2ByteTransform, GPT2PreTokenizer, pre_tokenize as gpt2_pre_tokenize
from .gpt_oss import GptOssByteTransform, GptOssPreTokenizer, pre_tokenize_gpt_oss
from .gemma4 import (
    Gemma4ByteTransform, Gemma4PreTokenizer,
    pre_tokenize_gemma4, gemma4_encode_bytes, gemma4_decode_bytes,
)


comptime TOKENIZER_FLAVOR_UNSUPPORTED = 0
comptime TOKENIZER_FLAVOR_GPT2 = 1
comptime TOKENIZER_FLAVOR_DEEPSEEK_V3 = 2
comptime TOKENIZER_FLAVOR_GPT_OSS = 3
comptime TOKENIZER_FLAVOR_GEMMA4 = 4


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
        if self.flavor == TOKENIZER_FLAVOR_GPT_OSS:
            return pre_tokenize_gpt_oss(text)
        if self.flavor == TOKENIZER_FLAVOR_GEMMA4:
            return pre_tokenize_gemma4(text)

        var out = List[String]()
        out.append(text.copy())
        return out^


struct AutoByteTransform(ByteTransformCapability):
    var flavor: Int

    def __init__(out self, flavor: Int):
        self.flavor = flavor

    def encode_bytes(self, data: Span[Byte, _]) -> String:
        if self.flavor == TOKENIZER_FLAVOR_GEMMA4:
            return gemma4_encode_bytes(data)
        return bytes_to_gpt2(data)

    def decode_bytes(self, text: String) -> List[Byte]:
        if self.flavor == TOKENIZER_FLAVOR_GEMMA4:
            return gemma4_decode_bytes(text)
        return gpt2_to_bytes(text)


struct ModelOptions(Movable):
    var fuse_unk: Bool
    var byte_fallback: Bool
    var unk_token: String

    def __init__(out self):
        self.fuse_unk = False
        self.byte_fallback = False
        self.unk_token = String("")


struct TokenizerConfigOptions(Movable):
    var bos_token: String
    var eos_token: String

    def __init__(out self):
        self.bos_token = String("")
        self.eos_token = String("")


struct PreTokenizerStageSignature(Copyable, ImplicitlyCopyable):
    var stage_type: String
    var behavior: String
    var regex_pattern: String
    var use_regex: Bool
    var add_prefix_space: Bool
    var individual_digits: Bool

    def __init__(out self):
        self.stage_type = String("")
        self.behavior = String("")
        self.regex_pattern = String("")
        self.use_regex = False
        self.add_prefix_space = False
        self.individual_digits = False


def parse_optional_bool(mut parser: Parser, default_value: Bool) raises ParseError -> Bool:
    if parser.try_consume[lit="null"]():
        return default_value
    return parser.parse_bool()


def parse_regex_pattern(mut parser: Parser) raises ParseError -> String:
    parser.skip_whitespace()
    if parser.try_consume[lit="null"]():
        return String("")

    if not parser.consume(LBRACE):
        parser.skip_value()
        return String("")

    parser.skip_whitespace()
    var regex = String("")
    if parser.consume(RBRACE):
        return regex^

    while True:
        var key = parser.object_key()
        if key == "Regex" or key == "String":
            regex = parser.parse_string()
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break

    return regex^


def parse_pretokenizer_stage_signature(
    mut parser: Parser, mut stage: PreTokenizerStageSignature
) raises ParseError:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for pretokenizer stage", parser.pos)

    parser.skip_whitespace()
    if parser.consume(RBRACE):
        return

    while True:
        var key = parser.object_key()
        if key == "type":
            stage.stage_type = parser.parse_string()
        elif key == "behavior":
            stage.behavior = parser.parse_string()
        elif key == "use_regex":
            stage.use_regex = parse_optional_bool(parser, stage.use_regex)
        elif key == "add_prefix_space":
            stage.add_prefix_space = parse_optional_bool(parser, stage.add_prefix_space)
        elif key == "individual_digits":
            stage.individual_digits = parse_optional_bool(parser, stage.individual_digits)
        elif key == "pattern":
            stage.regex_pattern = parse_regex_pattern(parser)
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break


def parse_pretokenizer_signatures(
    mut parser: Parser, mut stages: List[PreTokenizerStageSignature]
) raises ParseError:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for pre_tokenizer", parser.pos)

    parser.skip_whitespace()
    if parser.consume(RBRACE):
        return

    var single_stage = PreTokenizerStageSignature()
    var found_pretokenizers = False

    while True:
        var key = parser.object_key()
        if key == "pretokenizers":
            found_pretokenizers = True
            if not parser.consume(LBRACKET):
                raise ParseError("expected '[' for pretokenizers", parser.pos)
            parser.skip_whitespace()
            if parser.consume(RBRACKET):
                pass
            else:
                while True:
                    var stage = PreTokenizerStageSignature()
                    parse_pretokenizer_stage_signature(parser, stage)
                    stages.append(stage^)
                    if not parser.delimited_next(RBRACKET):
                        break
        elif key == "type":
            single_stage.stage_type = parser.parse_string()
        elif key == "behavior":
            single_stage.behavior = parser.parse_string()
        elif key == "pattern":
            single_stage.regex_pattern = parse_regex_pattern(parser)
        elif key == "use_regex":
            single_stage.use_regex = parse_optional_bool(parser, single_stage.use_regex)
        elif key == "add_prefix_space":
            single_stage.add_prefix_space = parse_optional_bool(
                parser, single_stage.add_prefix_space
            )
        elif key == "individual_digits":
            single_stage.individual_digits = parse_optional_bool(
                parser, single_stage.individual_digits
            )
        else:
            parser.skip_value()

        if not parser.delimited_next(RBRACE):
            break

    if not found_pretokenizers and single_stage.stage_type.byte_length() > 0:
        stages.append(single_stage^)


def is_gpt2_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
    if len(stages) != 2:
        return False

    var s0 = stages[0]
    var s1 = stages[1]
    if s0.stage_type != "Digits" or not s0.individual_digits:
        return False
    if s1.stage_type != "ByteLevel":
        return False
    if not s1.use_regex:
        return False
    if s1.add_prefix_space:
        return False
    return True


def is_deepseek_v3_stage2_pattern(regex: String) -> Bool:
    if not ("[A-Za-z]+" in regex):
        return False
    if not ("\\p{M}" in regex):
        return False
    if not ("\\p{P}\\p{S}" in regex):
        return False
    if not ("\\s+(?!\\S)" in regex):
        return False
    return True


def is_deepseek_v3_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
    if len(stages) != 4:
        return False

    var s0 = stages[0]
    var s1 = stages[1]
    var s2 = stages[2]
    var s3 = stages[3]

    if s0.stage_type != "Split" or s0.behavior != "Isolated":
        return False
    if s0.regex_pattern != "\\p{N}{1,3}":
        return False

    if s1.stage_type != "Split" or s1.behavior != "Isolated":
        return False
    if not ("一-龥" in s1.regex_pattern):
        return False
    if not ("぀-ゟ" in s1.regex_pattern):
        return False
    if not ("゠-ヿ" in s1.regex_pattern):
        return False

    if s2.stage_type != "Split" or s2.behavior != "Isolated":
        return False
    if not is_deepseek_v3_stage2_pattern(s2.regex_pattern):
        return False

    if s3.stage_type != "ByteLevel":
        return False
    if s3.use_regex:
        return False
    if s3.add_prefix_space:
        return False

    return True


def is_gpt_oss_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
    if len(stages) != 2:
        return False

    var s0 = stages[0]
    var s1 = stages[1]

    # Stage 0: Split with Isolated behavior and the o200k regex
    if s0.stage_type != "Split" or s0.behavior != "Isolated":
        return False
    if not ("\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}" in s0.regex_pattern):
        return False
    if not ("\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}" in s0.regex_pattern):
        return False
    if not ("\\p{N}{1,3}" in s0.regex_pattern):
        return False

    # Stage 1: ByteLevel with use_regex=false
    if s1.stage_type != "ByteLevel":
        return False
    if s1.use_regex:
        return False
    if s1.add_prefix_space:
        return False

    return True


def is_gemma4_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
    if len(stages) != 1:
        return False
    var s0 = stages[0]
    if s0.stage_type != "Split":
        return False
    if s0.behavior != "MergedWithPrevious":
        return False
    return True


def detect_tokenizer_flavor(path: Path) -> Int:
    var file_bytes: List[Byte]
    try:
        file_bytes = path.read_bytes()
    except:
        return TOKENIZER_FLAVOR_UNSUPPORTED

    var parser = Parser(Span(file_bytes))
    var stages = List[PreTokenizerStageSignature]()

    try:
        parser.skip_whitespace()
        if not parser.consume(LBRACE):
            return TOKENIZER_FLAVOR_UNSUPPORTED
        parser.skip_whitespace()
        if parser.consume(RBRACE):
            return TOKENIZER_FLAVOR_UNSUPPORTED

        while True:
            var key = parser.object_key()
            if key == "pre_tokenizer":
                parse_pretokenizer_signatures(parser, stages)
            else:
                parser.skip_value()
            if not parser.delimited_next(RBRACE):
                break
    except:
        return TOKENIZER_FLAVOR_UNSUPPORTED

    if is_gpt2_pretokenizer_signature(stages):
        return TOKENIZER_FLAVOR_GPT2
    if is_deepseek_v3_pretokenizer_signature(stages):
        return TOKENIZER_FLAVOR_DEEPSEEK_V3
    if is_gpt_oss_pretokenizer_signature(stages):
        return TOKENIZER_FLAVOR_GPT_OSS
    if is_gemma4_pretokenizer_signature(stages):
        return TOKENIZER_FLAVOR_GEMMA4
    return TOKENIZER_FLAVOR_UNSUPPORTED


def parse_added_token_content(mut parser: Parser) raises ParseError -> String:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for AddedToken object", parser.pos)
    parser.skip_whitespace()
    var content = String("")
    if parser.consume(RBRACE):
        return content^

    while True:
        var key = parser.object_key()
        if key == "content":
            content = parser.parse_string()
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break
    return content^


def parse_token_string_value(mut parser: Parser) raises ParseError -> String:
    parser.skip_whitespace()
    if parser.try_consume[lit="null"]():
        return String("")

    if parser.has_more() and parser.peek() == QUOTE:
        return parser.parse_string()

    if parser.has_more() and parser.peek() == LBRACE:
        return parse_added_token_content(parser)

    parser.skip_value()
    return String("")


def parse_tokenizer_config(path: Path) -> TokenizerConfigOptions:
    var opts = TokenizerConfigOptions()
    var file_bytes: List[Byte]
    try:
        file_bytes = path.read_bytes()
    except:
        return opts^

    var parser = Parser(Span(file_bytes))
    try:
        parser.skip_whitespace()
        if not parser.consume(LBRACE):
            return opts^
        parser.skip_whitespace()
        if parser.consume(RBRACE):
            return opts^

        while True:
            var key = parser.object_key()
            if key == "bos_token":
                opts.bos_token = parse_token_string_value(parser)
            elif key == "eos_token":
                opts.eos_token = parse_token_string_value(parser)
            else:
                parser.skip_value()
            if not parser.delimited_next(RBRACE):
                break
    except:
        return opts^

    return opts^

def parse_added_token(mut parser: Parser) raises ParseError -> Tuple[Int, String, Bool]:
    """Parse one added_token object, returning (id, content, special)."""
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for added_token", parser.pos)
    parser.skip_whitespace()
    var id = 0
    var content = String("")
    var special = False
    while True:
        var key = parser.object_key()
        if key == "id":
            id = parser.parse_uint()
        elif key == "content":
            content = parser.parse_string()
        elif key == "special":
            special = parser.parse_bool()
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break
    return (id, content^, special)

def parse_added_tokens_array(
    mut parser: Parser,
    mut added_tokens: Dict[String, Int],
    mut added_token_order: List[String],
    mut special_tokens: Dict[String, Int],
    mut special_ids: List[Int],
) raises ParseError:
    """Parse added_tokens array, populating added + special token maps."""
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for added_tokens", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACKET):
        return
    while True:
        var result = parse_added_token(parser)
        var id = result[0]
        var content = result[1]
        var is_special = result[2]
        added_tokens[content.copy()] = id
        added_token_order.append(content.copy())
        if is_special:
            special_tokens[content^] = id
            special_ids.append(id)
        if not parser.delimited_next(RBRACKET):
            break

def split_merge_string(pair: String) -> Optional[Tuple[String, String]]:
    """Split a space-delimited merge string like 'a b' into ('a', 'b')."""
    var bytes = pair.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == Byte(32):
            return (
                span_to_string(bytes, 0, i),
                span_to_string(bytes, i + 1, len(bytes)),
            )
    return None


def parse_merge_pair(mut parser: Parser) raises ParseError -> Tuple[String, String]:
    """Parse a single merge entry: either a string 'a b' or an array ['a', 'b']."""
    parser.skip_whitespace()
    if parser.has_more() and parser.peek() == LBRACKET:
        if not parser.consume(LBRACKET):
            raise ParseError("expected '[' for merge pair", parser.pos)
        parser.skip_whitespace()
        var left = parser.parse_string()
        if not parser.delimited_next(RBRACKET):
            raise ParseError("expected second element in merge pair", parser.pos)
        var right = parser.parse_string()
        parser.skip_whitespace()
        if not parser.consume(RBRACKET):
            raise ParseError("expected ']' for merge pair", parser.pos)
        return (left^, right^)

    var merged = parser.parse_string()
    var split = split_merge_string(merged)
    if not split:
        raise ParseError("invalid merge string (no space delimiter)", parser.pos)
    return split.take()


def parse_merges(
    mut parser: Parser,
    vocab: Dict[String, Int],
    mut merge_pair_ranks: Dict[UInt64, Int],
    mut merge_pair_out: Dict[UInt64, Int],
) raises ParseError -> Int:
    """Parse the merges array, building pair rank/output dicts directly.
    Returns the number of merges parsed."""
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for merges", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACKET):
        return 0

    var count = 0
    while True:
        var pair = parse_merge_pair(parser)
        var left_tok = pair[0]
        var right_tok = pair[1]
        var left_id = vocab.get(left_tok)
        var right_id = vocab.get(right_tok)
        if left_id and right_id:
            var merged_tok = left_tok + right_tok
            var out_id = vocab.get(merged_tok)
            if out_id:
                var key = pack_pair_ids(left_id.value(), right_id.value())
                merge_pair_ranks[key] = count
                merge_pair_out[key] = out_id.value()
        count += 1
        if not parser.delimited_next(RBRACKET):
            break
    return count


def parse_model_section(
    mut parser: Parser,
    mut vocab: Dict[String, Int],
    mut merge_pair_ranks: Dict[UInt64, Int],
    mut merge_pair_out: Dict[UInt64, Int],
    mut merge_count: Int,
    mut opts: ModelOptions,
) raises ParseError:
    """Parse the 'model' object, extracting vocab and building merge dicts."""
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for model", parser.pos)
    parser.skip_whitespace()
    if parser.consume(RBRACE):
        return
    while True:
        var key = parser.object_key()
        if key == "vocab":
            vocab = parser.parse_string_uint_dict()
        elif key == "merges":
            merge_count = parse_merges(
                parser, vocab, merge_pair_ranks, merge_pair_out,
            )
        elif key == "fuse_unk":
            opts.fuse_unk = parser.parse_bool()
        elif key == "byte_fallback":
            opts.byte_fallback = parser.parse_bool()
        elif key == "unk_token":
            if parser.try_consume[lit="null"]():
                opts.unk_token = String("")
            else:
                opts.unk_token = parser.parse_string()
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break

def load_tokenizer_with_capabilities[
    pretokenizer_type: PreTokenizerCapability,
    byte_transform_type: ByteTransformCapability,
](
    path: Path,
    var pretokenizer: pretokenizer_type,
    var byte_transform: byte_transform_type,
    use_piece_cache: Bool = True,
) -> Optional[BPETokenizer[pretokenizer_type, byte_transform_type]]:
    """Load a BPETokenizer from tokenizer.json using injected capabilities."""
    var file_bytes: List[Byte]
    try:
        file_bytes = path.read_bytes()
    except e:
        print("tokenizer: failed to read file:", e)
        return None

    var parser = Parser(Span(file_bytes))

    var vocab = Dict[String, Int]()
    var merge_pair_ranks = Dict[UInt64, Int]()
    var merge_pair_out = Dict[UInt64, Int]()
    var merge_count = 0
    var added_tokens = Dict[String, Int]()
    var added_token_order = List[String]()
    var special_tokens = Dict[String, Int]()
    var special_ids = List[Int]()
    var model_opts = ModelOptions()
    var tokenizer_cfg_path = Path(
        String(path).replace("tokenizer.json", "tokenizer_config.json")
    )
    var tokenizer_cfg = parse_tokenizer_config(tokenizer_cfg_path)

    try:
        parser.skip_whitespace()
        if not parser.consume(LBRACE):
            print("tokenizer: expected '{' at start")
            return None
        parser.skip_whitespace()

        while True:
            var key = parser.object_key()
            if key == "added_tokens":
                parse_added_tokens_array(
                    parser,
                    added_tokens,
                    added_token_order,
                    special_tokens,
                    special_ids,
                )
            elif key == "model":
                parse_model_section(
                    parser, vocab,
                    merge_pair_ranks, merge_pair_out, merge_count,
                    model_opts,
                )
            else:
                parser.skip_value()
            if not parser.delimited_next(RBRACE):
                break
    except e:
        print("tokenizer: parse error at pos", e.pos, ":", e.message)
        return None

    var bos_token_id = -1
    if tokenizer_cfg.bos_token.byte_length() > 0:
        var bos_special = special_tokens.get(tokenizer_cfg.bos_token)
        if bos_special:
            bos_token_id = bos_special.value()
        else:
            var bos_vocab = vocab.get(tokenizer_cfg.bos_token)
            if bos_vocab:
                bos_token_id = bos_vocab.value()

    var eos_token_id = -1
    if tokenizer_cfg.eos_token.byte_length() > 0:
        var eos_special = special_tokens.get(tokenizer_cfg.eos_token)
        if eos_special:
            eos_token_id = eos_special.value()
        else:
            var eos_vocab = vocab.get(tokenizer_cfg.eos_token)
            if eos_vocab:
                eos_token_id = eos_vocab.value()

    var vocab_size = len(vocab)
    return BPETokenizer[pretokenizer_type, byte_transform_type](
        vocab^,
        merge_count,
        merge_pair_ranks^,
        merge_pair_out^,
        added_tokens^,
        added_token_order^,
        special_tokens^,
        special_ids^,
        model_opts.fuse_unk,
        model_opts.byte_fallback,
        model_opts.unk_token^,
        False,
        False,
        bos_token_id,
        eos_token_id,
        vocab_size,
        use_piece_cache,
        pretokenizer^,
        byte_transform^,
    )


def load_tokenizer(path: Path) -> Optional[BPETokenizer[AutoPreTokenizer, AutoByteTransform]]:
    """Load a BPETokenizer by auto-detecting supported pre-tokenizer semantics."""
    var flavor = detect_tokenizer_flavor(path)
    if flavor != TOKENIZER_FLAVOR_UNSUPPORTED:
        return load_tokenizer_with_capabilities(
            path,
            AutoPreTokenizer(flavor),
            AutoByteTransform(flavor),
            use_piece_cache=(flavor != TOKENIZER_FLAVOR_GEMMA4),
        )

    print("tokenizer: unsupported pre-tokenizer semantics in", path)
    return None


def load_gpt2_tokenizer(path: Path) -> Optional[
    BPETokenizer[GPT2PreTokenizer, GPT2ByteTransform]
]:
    """Load a BPETokenizer using GPT-2 pre-tokenizer semantics."""
    return load_tokenizer_with_capabilities(path, GPT2PreTokenizer(), GPT2ByteTransform())


def load_deepseek_v3_tokenizer(path: Path) -> Optional[
    BPETokenizer[DeepSeekV3PreTokenizer, DeepSeekV3ByteTransform]
]:
    """Load a BPETokenizer using DeepSeek V3 pre-tokenizer semantics."""
    return load_tokenizer_with_capabilities(
        path,
        DeepSeekV3PreTokenizer(),
        DeepSeekV3ByteTransform(),
    )


def load_gpt_oss_tokenizer(path: Path) -> Optional[
    BPETokenizer[GptOssPreTokenizer, GptOssByteTransform]
]:
    """Load a BPETokenizer using GPT-OSS pre-tokenizer semantics."""
    return load_tokenizer_with_capabilities(
        path,
        GptOssPreTokenizer(),
        GptOssByteTransform(),
    )


def load_gemma4_tokenizer(path: Path) -> Optional[
    BPETokenizer[Gemma4PreTokenizer, Gemma4ByteTransform]
]:
    """Load a BPETokenizer using Gemma 4 tokenizer semantics."""
    return load_tokenizer_with_capabilities(
        path,
        Gemma4PreTokenizer(),
        Gemma4ByteTransform(),
        use_piece_cache=False,
    )
