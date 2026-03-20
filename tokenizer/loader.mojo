from pathlib import Path
from memory import Span
from collections import Dict

from jsontools.parser import (
    Parser,
    ParseError,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    QUOTE,
)
from .tokenizer import BPETokenizer
from .auto import (
    AutoPreTokenizer,
    TOKENIZER_FLAVOR_UNSUPPORTED,
    TOKENIZER_FLAVOR_GPT2,
    TOKENIZER_FLAVOR_DEEPSEEK_V3,
)
from .capabilities import ByteTransformCapability, PreTokenizerCapability
from .deepseek_v3 import DeepSeekV3ByteTransform, DeepSeekV3PreTokenizer
from .gpt2 import GPT2ByteTransform, GPT2PreTokenizer


struct ModelOptions(Movable):
    var ignore_merges: Bool
    var fuse_unk: Bool
    var byte_fallback: Bool
    var unk_token: String

    fn __init__(out self):
        self.ignore_merges = False
        self.fuse_unk = False
        self.byte_fallback = False
        self.unk_token = String("")


struct TokenizerConfigOptions(Movable):
    var add_bos_token: Bool
    var add_eos_token: Bool
    var bos_token: String
    var eos_token: String

    fn __init__(out self):
        self.add_bos_token = False
        self.add_eos_token = False
        self.bos_token = String("")
        self.eos_token = String("")


struct PreTokenizerStageSignature(Copyable, ImplicitlyCopyable):
    var stage_type: String
    var behavior: String
    var regex_pattern: String
    var use_regex: Bool
    var add_prefix_space: Bool
    var individual_digits: Bool

    fn __init__(out self):
        self.stage_type = String("")
        self.behavior = String("")
        self.regex_pattern = String("")
        self.use_regex = False
        self.add_prefix_space = False
        self.individual_digits = False


fn parse_optional_bool(mut parser: Parser, default_value: Bool) raises ParseError -> Bool:
    if parser.try_consume[lit="null"]():
        return default_value
    return parser.parse_bool()


fn parse_regex_pattern(mut parser: Parser) raises ParseError -> String:
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
        if key == "Regex":
            regex = parser.parse_string()
        else:
            parser.skip_value()
        if not parser.delimited_next(RBRACE):
            break

    return regex^


fn parse_pretokenizer_stage_signature(
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


fn parse_pretokenizer_signatures(
    mut parser: Parser, mut stages: List[PreTokenizerStageSignature]
) raises ParseError:
    if not parser.consume(LBRACE):
        raise ParseError("expected '{' for pre_tokenizer", parser.pos)

    parser.skip_whitespace()
    if parser.consume(RBRACE):
        return

    while True:
        var key = parser.object_key()
        if key == "pretokenizers":
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
        else:
            parser.skip_value()

        if not parser.delimited_next(RBRACE):
            break


fn is_gpt2_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
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


fn is_deepseek_v3_stage2_pattern(regex: String) -> Bool:
    if not ("[A-Za-z]+" in regex):
        return False
    if not ("\\p{M}" in regex):
        return False
    if not ("\\p{P}\\p{S}" in regex):
        return False
    if not ("\\s+(?!\\S)" in regex):
        return False
    return True


fn is_deepseek_v3_pretokenizer_signature(stages: List[PreTokenizerStageSignature]) -> Bool:
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


fn detect_tokenizer_flavor(path: Path) -> Int:
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
    return TOKENIZER_FLAVOR_UNSUPPORTED


fn parse_added_token_content(mut parser: Parser) raises ParseError -> String:
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


fn parse_token_string_value(mut parser: Parser) raises ParseError -> String:
    parser.skip_whitespace()
    if parser.try_consume[lit="null"]():
        return String("")

    if parser.has_more() and parser.peek() == QUOTE:
        return parser.parse_string()

    if parser.has_more() and parser.peek() == LBRACE:
        return parse_added_token_content(parser)

    parser.skip_value()
    return String("")


fn parse_tokenizer_config(path: Path) -> TokenizerConfigOptions:
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
            if key == "add_bos_token":
                opts.add_bos_token = parse_optional_bool(parser, opts.add_bos_token)
            elif key == "add_eos_token":
                opts.add_eos_token = parse_optional_bool(parser, opts.add_eos_token)
            elif key == "bos_token":
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

fn parse_added_token(mut parser: Parser) raises ParseError -> Tuple[Int, String, Bool]:
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

fn parse_added_tokens_array(
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

fn parse_model_section(
    mut parser: Parser,
    mut vocab: Dict[String, Int],
    mut merges: List[String],
    mut opts: ModelOptions,
) raises ParseError:
    """Parse the 'model' object, extracting vocab/merges/options."""
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
            merges = parser.parse_string_array()
        elif key == "ignore_merges":
            opts.ignore_merges = parser.parse_bool()
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

fn load_tokenizer_with_capabilities[
    pretokenizer_type: PreTokenizerCapability,
    byte_transform_type: ByteTransformCapability,
](
    path: Path,
    var pretokenizer: pretokenizer_type,
    var byte_transform: byte_transform_type,
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
    var merges = List[String]()
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
                parse_model_section(parser, vocab, merges, model_opts)
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
        merges^,
        added_tokens^,
        added_token_order^,
        special_tokens^,
        special_ids^,
        model_opts.ignore_merges,
        model_opts.fuse_unk,
        model_opts.byte_fallback,
        model_opts.unk_token^,
        tokenizer_cfg.add_bos_token,
        tokenizer_cfg.add_eos_token,
        bos_token_id,
        eos_token_id,
        vocab_size,
        pretokenizer^,
        byte_transform^,
    )


fn load_tokenizer(path: Path) -> Optional[BPETokenizer]:
    """Load a BPETokenizer by auto-detecting supported pre-tokenizer semantics."""
    var flavor = detect_tokenizer_flavor(path)
    if flavor == TOKENIZER_FLAVOR_GPT2 or flavor == TOKENIZER_FLAVOR_DEEPSEEK_V3:
        return load_tokenizer_with_capabilities(
            path,
            AutoPreTokenizer(flavor),
            GPT2ByteTransform(),
        )

    print("tokenizer: unsupported pre-tokenizer semantics in", path)
    print("tokenizer: supported flavors are GPT-2 and DeepSeek V3 only")
    return None


fn load_gpt2_tokenizer(path: Path) -> Optional[
    BPETokenizer[GPT2PreTokenizer, GPT2ByteTransform]
]:
    """Load a BPETokenizer using GPT-2 pre-tokenizer semantics."""
    return load_tokenizer_with_capabilities(path, GPT2PreTokenizer(), GPT2ByteTransform())


fn load_deepseek_v3_tokenizer(path: Path) -> Optional[
    BPETokenizer[DeepSeekV3PreTokenizer, DeepSeekV3ByteTransform]
]:
    """Load a BPETokenizer using DeepSeek V3 pre-tokenizer semantics."""
    return load_tokenizer_with_capabilities(
        path,
        DeepSeekV3PreTokenizer(),
        DeepSeekV3ByteTransform(),
    )
