from std.pathlib import Path
from tokenizer import load_tokenizer, BPETokenizer


comptime TOKENIZER_PATH = "checkpoints/gpt-oss-20b/tokenizer.json"


def ids_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def print_ids(ids: List[Int]):
    var s = String("[")
    for i in range(len(ids)):
        if i > 0:
            s += ", "
        s += String(ids[i])
    s += "]"
    print(s)


def check(
    mut tok: BPETokenizer[...],
    text: String,
    expected: List[Int],
) -> Bool:
    var got = tok.encode(text)
    var decoded = tok.decode(got)
    if ids_equal(got, expected) and decoded == text:
        print("  PASS:", repr(text))
        return True
    print("  FAIL:", repr(text))
    if not ids_equal(got, expected):
        print("    expected:", end=" ")
        print_ids(expected)
        print("    got:     ", end=" ")
        print_ids(got)
    if decoded != text:
        print("    DECODE MISMATCH:", repr(decoded))
    return False


def main():
    print("Loading GPT-OSS tokenizer...")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("FAILED to load tokenizer")
        return
    var tok = tok_opt.take()
    print("Loaded. Vocab:", tok.vocab_size(), "Merges:", tok.num_merges())
    print("BOS:", tok.bos_token_id, "EOS:", tok.eos_token_id)
    print()

    var pass_count = 0
    var fail_count = 0

    print("=== Encode/decode tests ===")

    # Basic English
    if check(tok, "Hello, world!", [13225, 11, 2375, 0]): pass_count += 1
    else: fail_count += 1
    if check(tok, "The quick brown fox jumps over the lazy dog.",
        [976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13]): pass_count += 1
    else: fail_count += 1

    # CamelCase
    if check(tok, "camelCase httpClient parseHTTPResponse",
        [178067, 6187, 3958, 3510, 8420, 17893, 3186]): pass_count += 1
    else: fail_count += 1
    if check(tok, "CamelCase XMLParser HTMLElement",
        [137910, 6187, 22580, 9231, 97351]): pass_count += 1
    else: fail_count += 1
    if check(tok, "ALLCAPS lowercase MiXeD",
        [7011, 56928, 50, 90395, 13236, 148218, 35]): pass_count += 1
    else: fail_count += 1

    # Contractions
    if check(tok, "DON'T don't Can't can't",
        [134882, 51532, 4128, 58989, 8535]): pass_count += 1
    else: fail_count += 1

    # Numbers
    if check(tok, "12345 123 1 42 999",
        [7633, 2548, 220, 7633, 220, 16, 220, 4689, 220, 9130]): pass_count += 1
    else: fail_count += 1

    # Whitespace
    if check(tok, " leading space", [8117, 4918]): pass_count += 1
    else: fail_count += 1
    if check(tok, "multiple   spaces   here",
        [76466, 256, 18608, 256, 2105]): pass_count += 1
    else: fail_count += 1
    if check(tok, "line1\nline2\nline3",
        [1137, 16, 198, 1137, 17, 198, 1137, 18]): pass_count += 1
    else: fail_count += 1

    # Symbols / code
    if check(tok, "symbols: @#$%^&*()",
        [134245, 25, 759, 108156, 108254, 5, 9, 416]): pass_count += 1
    else: fail_count += 1
    if check(tok, "code: def foo(x): return x + 1",
        [3056, 25, 1056, 30551, 4061, 3127, 622, 1215, 659, 220, 16]): pass_count += 1
    else: fail_count += 1

    # Chinese
    if check(tok, "你好世界", [177519, 28428]): pass_count += 1
    else: fail_count += 1
    if check(tok, "今天天气很好，我想出去走走。",
        [10941, 1487, 25896, 148483, 40824, 18165, 190657, 11941, 11941, 788]): pass_count += 1
    else: fail_count += 1
    if check(tok, "深度学习是人工智能的一个分支。",
        [23052, 8913, 64550, 3221, 47243, 60319, 1616, 22912, 2957, 18904, 788]): pass_count += 1
    else: fail_count += 1
    if check(tok, "中英混合test测试123",
        [1404, 24309, 85591, 4377, 3190, 82843, 7633]): pass_count += 1
    else: fail_count += 1

    # Japanese
    if check(tok, "こんにちは世界", [95839, 28428]): pass_count += 1
    else: fail_count += 1
    if check(tok, "東京は日本の首都です。",
        [108713, 5205, 9048, 3385, 15425, 12232, 15121, 788]): pass_count += 1
    else: fail_count += 1
    if check(tok, "カタカナとひらがなの混合テスト",
        [14214, 12288, 14214, 27354, 5330, 60922, 8870, 6632, 172712, 85591, 4377, 16056, 38236]): pass_count += 1
    else: fail_count += 1
    if check(tok, "日本語English混在テキスト",
        [9048, 40909, 28881, 85591, 2178, 16056, 18368, 38236]): pass_count += 1
    else: fail_count += 1

    # Korean
    if check(tok, "안녕하세요 세계", [14307, 171731, 75755]): pass_count += 1
    else: fail_count += 1
    if check(tok, "한국어 테스트입니다", [114854, 5959, 169814, 27001]): pass_count += 1
    else: fail_count += 1

    # Arabic
    if check(tok, "مرحبا بالعالم", [158894, 26537, 101462, 12773]): pass_count += 1
    else: fail_count += 1

    # Russian
    if check(tok, "Привет мир", [23881, 131903, 37934]): pass_count += 1
    else: fail_count += 1

    # Thai
    if check(tok, "สวัสดีชาวโลก",
        [4406, 187986, 21883, 2293, 8247, 17359, 93469]): pass_count += 1
    else: fail_count += 1

    # Emoji
    if check(tok, "emoji: 👩‍💻 🚀 ❤️",
        [75339, 25, 61138, 102, 2524, 31446, 119, 169883, 222, 122205]): pass_count += 1
    else: fail_count += 1

    # Special tokens
    if check(tok, "<|startoftext|>", [199998]): pass_count += 1
    else: fail_count += 1
    if check(tok, "<|endoftext|>", [199999]): pass_count += 1
    else: fail_count += 1
    if check(tok, "<|start|>system<|message|>You are helpful.<|end|>",
        [200006, 17360, 200008, 3575, 553, 10297, 13, 200007]): pass_count += 1
    else: fail_count += 1

    print()
    print("Results:", pass_count, "passed,", fail_count, "failed")
