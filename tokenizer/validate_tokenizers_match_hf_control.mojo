from std.pathlib import Path
from tokenizer import load_tokenizer, BPETokenizer


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


def test_smollm2(mut tok: BPETokenizer[...]) -> Tuple[Int, Int]:
    var p = 0
    var f = 0

    # Basic English
    if check(tok, "Hello, world!", [19556, 28, 905, 17]): p += 1
    else: f += 1
    if check(tok, "The quick brown fox jumps over the lazy dog.",
        [504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767, 30]): p += 1
    else: f += 1

    # CamelCase
    if check(tok, "camelCase httpClient parseHTTPResponse",
        [12744, 299, 10802, 3622, 15866, 12865, 23432, 12463]): p += 1
    else: f += 1
    if check(tok, "CamelCase XMLParser HTMLElement",
        [51, 14548, 10802, 18391, 13157, 8627, 61, 4796, 1862]): p += 1
    else: f += 1
    if check(tok, "ALLCAPS lowercase MiXeD",
        [12631, 33467, 67, 37748, 11633, 72, 85, 52]): p += 1
    else: f += 1

    # Contractions
    if check(tok, "DON'T don't Can't can't",
        [52, 2154, 23, 68, 1326, 982, 1978, 982, 416, 982]): p += 1
    else: f += 1

    # Numbers
    if check(tok, "12345 123 1 42 999",
        [33, 34, 35, 36, 37, 216, 33, 34, 35, 216, 33, 216, 36, 34, 216, 41, 41, 41]): p += 1
    else: f += 1

    # Whitespace
    if check(tok, " leading space", [2899, 1898]): p += 1
    else: f += 1
    if check(tok, "multiple   spaces   here",
        [30404, 256, 5600, 256, 1535]): p += 1
    else: f += 1
    if check(tok, "line1\nline2\nline3",
        [1311, 33, 198, 1311, 34, 198, 1311, 35]): p += 1
    else: f += 1
    if check(tok, "tabs\there\ttoo",
        [100, 7366, 197, 1531, 197, 23034]): p += 1
    else: f += 1

    # Symbols / code
    if check(tok, "symbols: @#$%^&*()",
        [42332, 42, 3394, 19, 20, 21, 78, 22, 26, 1000]): p += 1
    else: f += 1
    if check(tok, "code: def foo(x): return x + 1",
        [4635, 42, 753, 29856, 24, 104, 727, 1003, 1792, 1232, 216, 33]): p += 1
    else: f += 1

    # Chinese
    if check(tok, "你好世界",
        [18645, 250, 48392, 138, 7906, 240, 178, 239, 230]): p += 1
    else: f += 1
    if check(tok, "中英混合test测试123",
        [28589, 179, 229, 126, 177, 132, 132, 16357, 226, 2129, 43222, 229, 16313, 239, 33, 34, 35]): p += 1
    else: f += 1

    # Japanese
    if check(tok, "こんにちは世界",
        [7365, 237, 10391, 237, 41152, 7365, 111, 7365, 124, 7906, 240, 178, 239, 230]): p += 1
    else: f += 1
    if check(tok, "東京は日本の首都です。",
        [30679, 126, 16736, 122, 7365, 124, 23274, 115, 40993, 26453, 180, 116, 240, 180, 38466, 7365, 117, 39958, 19076]): p += 1
    else: f += 1

    # Korean
    if check(tok, "안녕하세요 세계",
        [183, 239, 226, 182, 223, 239, 33085, 242, 183, 222, 133, 183, 244, 238, 18601, 222, 133, 181, 128, 222]): p += 1
    else: f += 1

    # Arabic
    if check(tok, "مرحبا بالعالم",
        [21706, 20602, 44626, 31462, 10805, 45726, 22208, 34966, 22208, 21706]): p += 1
    else: f += 1

    # Russian
    if check(tok, "Привет мир",
        [155, 249, 46879, 16464, 32547, 41337, 7872, 9175]): p += 1
    else: f += 1

    # Thai
    if check(tok, "สวัสดีชาวโลก",
        [10283, 120, 10283, 117, 10283, 126, 10283, 120, 10283, 238, 10283, 130, 10283, 228, 43662, 10283, 117, 31933, 220, 10283, 115, 10283, 219]): p += 1
    else: f += 1

    # Emoji
    if check(tok, "emoji: 👩‍💻 🚀 ❤️",
        [391, 33777, 42, 15107, 235, 119, 321, 231, 10813, 236, 136, 15107, 244, 218, 4636, 247, 114, 31752]): p += 1
    else: f += 1

    return (p, f)


def test_deepseek_v3(mut tok: BPETokenizer[...]) -> Tuple[Int, Int]:
    var p = 0
    var f = 0

    # Basic English
    if check(tok, "Hello, world!", [19923, 14, 2058, 3]): p += 1
    else: f += 1
    if check(tok, "The quick brown fox jumps over the lazy dog.",
        [671, 4787, 13769, 46012, 54994, 1060, 270, 41638, 6397, 16]): p += 1
    else: f += 1

    # CamelCase
    if check(tok, "camelCase httpClient parseHTTPResponse",
        [69, 31105, 15434, 7283, 13834, 27438, 45909, 11169]): p += 1
    else: f += 1
    if check(tok, "CamelCase XMLParser HTMLElement",
        [37, 31105, 15434, 31792, 40563, 12305, 47, 4392, 3662]): p += 1
    else: f += 1
    if check(tok, "ALLCAPS lowercase MiXeD",
        [2570, 11059, 79732, 64508, 21857, 58, 71, 38]): p += 1
    else: f += 1

    # Contractions
    if check(tok, "DON'T don't Can't can't",
        [38, 1964, 67322, 2090, 1664, 3721, 1664, 588, 1664]): p += 1
    else: f += 1

    # Numbers
    if check(tok, "12345 123 1 42 999",
        [6895, 1883, 223, 6895, 223, 19, 223, 3180, 223, 8834]): p += 1
    else: f += 1

    # Whitespace
    if check(tok, " leading space", [6646, 3987]): p += 1
    else: f += 1
    if check(tok, "multiple   spaces   here",
        [87372, 262, 13564, 262, 2155]): p += 1
    else: f += 1
    if check(tok, "line1\nline2\nline3",
        [1836, 19, 201, 1836, 20, 201, 1836, 21]): p += 1
    else: f += 1
    if check(tok, "tabs\there\ttoo",
        [86, 10284, 200, 1036, 200, 56255]): p += 1
    else: f += 1

    # Symbols / code
    if check(tok, "symbols: @#$%^&*()",
        [67297, 85, 28, 2390, 125463, 7, 64, 8, 12, 1393]): p += 1
    else: f += 1
    if check(tok, "code: def foo(x): return x + 1",
        [8308, 28, 1351, 52735, 4042, 2605, 1354, 1527, 940, 223, 19]): p += 1
    else: f += 1

    # Chinese
    if check(tok, "你好世界", [30594, 3427]): p += 1
    else: f += 1
    if check(tok, "今天天气很好，我想出去走走。",
        [5237, 16652, 12210, 303, 11458, 9772, 69787, 320]): p += 1
    else: f += 1
    if check(tok, "中英混合test测试123",
        [525, 3218, 14769, 7958, 10251, 6895]): p += 1
    else: f += 1

    # Japanese
    if check(tok, "こんにちは世界",
        [4549, 7245, 2298, 12457, 2841, 3427]): p += 1
    else: f += 1
    if check(tok, "東京は日本の首都です。",
        [66771, 2841, 82619, 41175, 8262, 320]): p += 1
    else: f += 1
    if check(tok, "カタカナとひらがなの混合テスト",
        [15961, 11767, 15961, 27071, 2495, 40259, 4970, 2936, 2942, 1576, 14769, 109288]): p += 1
    else: f += 1
    if check(tok, "日本語English混在テキスト",
        [88768, 17530, 5764, 445, 17383, 20367, 24552]): p += 1
    else: f += 1

    # Korean
    if check(tok, "안녕하세요 세계",
        [31404, 11939, 246, 4567, 73527, 76878]): p += 1
    else: f += 1

    # Arabic
    if check(tok, "مرحبا بالعالم",
        [10393, 2212, 53067, 9254, 1183, 14059]): p += 1
    else: f += 1

    # Russian
    if check(tok, "Привет мир", [24797, 8919, 74779]): p += 1
    else: f += 1

    # Thai
    if check(tok, "สวัสดีชาวโลก",
        [2952, 34189, 2952, 25581, 88176, 50889]): p += 1
    else: f += 1

    # Emoji
    if check(tok, "emoji: 👩‍💻 🚀 ❤️",
        [18872, 7063, 28, 52780, 105, 46088, 23903, 122, 7351, 251, 225, 53341, 100, 10759]): p += 1
    else: f += 1

    return (p, f)


def test_gpt_oss(mut tok: BPETokenizer[...]) -> Tuple[Int, Int]:
    var p = 0
    var f = 0

    # Basic English
    if check(tok, "Hello, world!", [13225, 11, 2375, 0]): p += 1
    else: f += 1
    if check(tok, "The quick brown fox jumps over the lazy dog.",
        [976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13]): p += 1
    else: f += 1

    # CamelCase
    if check(tok, "camelCase httpClient parseHTTPResponse",
        [178067, 6187, 3958, 3510, 8420, 17893, 3186]): p += 1
    else: f += 1
    if check(tok, "CamelCase XMLParser HTMLElement",
        [137910, 6187, 22580, 9231, 97351]): p += 1
    else: f += 1
    if check(tok, "ALLCAPS lowercase MiXeD",
        [7011, 56928, 50, 90395, 13236, 148218, 35]): p += 1
    else: f += 1

    # Contractions
    if check(tok, "DON'T don't Can't can't",
        [134882, 51532, 4128, 58989, 8535]): p += 1
    else: f += 1

    # Numbers
    if check(tok, "12345 123 1 42 999",
        [7633, 2548, 220, 7633, 220, 16, 220, 4689, 220, 9130]): p += 1
    else: f += 1

    # Whitespace
    if check(tok, " leading space", [8117, 4918]): p += 1
    else: f += 1
    if check(tok, "multiple   spaces   here",
        [76466, 256, 18608, 256, 2105]): p += 1
    else: f += 1
    if check(tok, "line1\nline2\nline3",
        [1137, 16, 198, 1137, 17, 198, 1137, 18]): p += 1
    else: f += 1
    if check(tok, "tabs\there\ttoo",
        [68999, 197, 19992, 197, 23657]): p += 1
    else: f += 1

    # Symbols / code
    if check(tok, "symbols: @#$%^&*()",
        [134245, 25, 759, 108156, 108254, 5, 9, 416]): p += 1
    else: f += 1
    if check(tok, "code: def foo(x): return x + 1",
        [3056, 25, 1056, 30551, 4061, 3127, 622, 1215, 659, 220, 16]): p += 1
    else: f += 1

    # Chinese
    if check(tok, "你好世界", [177519, 28428]): p += 1
    else: f += 1
    if check(tok, "中英混合test测试123",
        [1404, 24309, 85591, 4377, 3190, 82843, 7633]): p += 1
    else: f += 1

    # Japanese
    if check(tok, "こんにちは世界", [95839, 28428]): p += 1
    else: f += 1
    if check(tok, "東京は日本の首都です。",
        [108713, 5205, 9048, 3385, 15425, 12232, 15121, 788]): p += 1
    else: f += 1
    if check(tok, "カタカナとひらがなの混合テスト",
        [14214, 12288, 14214, 27354, 5330, 60922, 8870, 6632, 172712, 85591, 4377, 16056, 38236]): p += 1
    else: f += 1
    if check(tok, "日本語English混在テキスト",
        [9048, 40909, 28881, 85591, 2178, 16056, 18368, 38236]): p += 1
    else: f += 1

    # Korean
    if check(tok, "안녕하세요 세계", [14307, 171731, 75755]): p += 1
    else: f += 1

    # Arabic
    if check(tok, "مرحبا بالعالم", [158894, 26537, 101462, 12773]): p += 1
    else: f += 1

    # Russian
    if check(tok, "Привет мир", [23881, 131903, 37934]): p += 1
    else: f += 1

    # Thai
    if check(tok, "สวัสดีชาวโลก",
        [4406, 187986, 21883, 2293, 8247, 17359, 93469]): p += 1
    else: f += 1

    # Emoji
    if check(tok, "emoji: 👩‍💻 🚀 ❤️",
        [75339, 25, 61138, 102, 2524, 31446, 119, 169883, 222, 122205]): p += 1
    else: f += 1

    # Special tokens
    if check(tok, "<|startoftext|>", [199998]): p += 1
    else: f += 1
    if check(tok, "<|endoftext|>", [199999]): p += 1
    else: f += 1
    if check(tok, "<|start|>system<|message|>You are helpful.<|end|>",
        [200006, 17360, 200008, 3575, 553, 10297, 13, 200007]): p += 1
    else: f += 1

    return (p, f)


def main():
    var total_pass = 0
    var total_fail = 0

    # SmolLM2 (GPT-2 pre-tokenizer)
    print("=== SmolLM2 (GPT-2) ===")
    var smol_opt = load_tokenizer(Path("checkpoints/SmolLM2/tokenizer.json"))
    if not smol_opt:
        print("FAILED to load SmolLM2 tokenizer")
    else:
        var smol = smol_opt.take()
        print("Vocab:", smol.vocab_size(), "Merges:", smol.num_merges())
        var result = test_smollm2(smol)
        print("SmolLM2:", result[0], "passed,", result[1], "failed")
        total_pass += result[0]
        total_fail += result[1]

    print()

    # DeepSeek V3
    print("=== DeepSeek V3 ===")
    var ds_opt = load_tokenizer(Path("checkpoints/deepseekv3/tokenizer.json"))
    if not ds_opt:
        print("FAILED to load DeepSeek V3 tokenizer")
    else:
        var ds = ds_opt.take()
        print("Vocab:", ds.vocab_size(), "Merges:", ds.num_merges())
        var result = test_deepseek_v3(ds)
        print("DeepSeek V3:", result[0], "passed,", result[1], "failed")
        total_pass += result[0]
        total_fail += result[1]

    print()

    # GPT-OSS
    print("=== GPT-OSS ===")
    var gpt_opt = load_tokenizer(Path("checkpoints/gpt-oss-20b/tokenizer.json"))
    if not gpt_opt:
        print("FAILED to load GPT-OSS tokenizer")
    else:
        var gpt = gpt_opt.take()
        print("Vocab:", gpt.vocab_size(), "Merges:", gpt.num_merges())
        var result = test_gpt_oss(gpt)
        print("GPT-OSS:", result[0], "passed,", result[1], "failed")
        total_pass += result[0]
        total_fail += result[1]

    print()
    print("=== TOTAL:", total_pass, "passed,", total_fail, "failed ===")
