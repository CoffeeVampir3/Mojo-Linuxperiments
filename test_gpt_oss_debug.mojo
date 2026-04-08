from std.pathlib import Path
from tokenizer import load_tokenizer, pre_tokenize_gpt_oss
from tokenizer.bpe import pack_pair_ids
from tokenizer.tokenizer import (
    bytes_to_gpt2, UnicodeContext,
    is_unicode_letter_cp, is_unicode_number_cp,
    is_unicode_whitespace_cp, is_unicode_mark_cp,
)


def main():
    var tok_opt = load_tokenizer(Path("checkpoints/gpt-oss-20b/tokenizer.json"))
    if not tok_opt:
        print("FAILED")
        return
    var tok = tok_opt.take()

    # Step 1: Check what bytes the Mojo string actually contains
    var text = " ❤️"
    var raw = text.as_bytes()
    print("=== Raw bytes of ' ❤️' ===")
    print("byte count:", len(raw))
    for i in range(len(raw)):
        print("  byte", i, ":", Int(raw[i]), "hex:", hex(Int(raw[i])))

    # Expected: 0x20 0xE2 0x9D 0xA4 0xEF 0xB8 0x8F (7 bytes)

    # Step 2: Check pre-tokenizer output
    print()
    print("=== Pre-tokenizer pieces ===")
    var pieces = pre_tokenize_gpt_oss(text)
    print("piece count:", len(pieces))
    for i in range(len(pieces)):
        var p = pieces[i]
        var pb = p.as_bytes()
        print("  piece", i, ":", repr(p), "bytes:", len(pb))

    # Step 3: Check byte transform
    print()
    print("=== Byte transform ===")
    for i in range(len(pieces)):
        var p = pieces[i]
        var transformed = bytes_to_gpt2(p.as_bytes())
        print("  piece", i, "->", repr(transformed))
        # Show each codepoint's vocab lookup
        for slice in transformed.codepoint_slices():
            var s = String(slice)
            var found = tok.token_to_id(s)
            if found:
                print("    codepoint", repr(s), "-> id", found.value())
            else:
                print("    codepoint", repr(s), "-> NOT FOUND")

    # Step 4: Full encode
    print()
    print("=== Full encode ===")
    var ids = tok.encode(text)
    print("IDs:", end="")
    for i in range(len(ids)):
        print("", ids[i], end="")
    print()
    print("Decoded:", repr(tok.decode(ids)))

    # Step 5: Check Unicode classification for key codepoints
    print()
    print("=== Unicode classification ===")
    var ctx = UnicodeContext()
    var test_cps = List[UInt32]()
    var test_names = List[String]()
    test_cps.append(UInt32(0x2764)); test_names.append("U+2764 HEART")
    test_cps.append(UInt32(0xFE0F)); test_names.append("U+FE0F VS16")
    test_cps.append(UInt32(0x200D)); test_names.append("U+200D ZWJ")
    test_cps.append(UInt32(0x1F469)); test_names.append("U+1F469 WOMAN")
    test_cps.append(UInt32(0x1F4BB)); test_names.append("U+1F4BB COMPUTER")
    test_cps.append(UInt32(0x1F680)); test_names.append("U+1F680 ROCKET")
    for i in range(len(test_cps)):
        var cp = test_cps[i]
        print(
            " ", test_names[i],
            "letter:", is_unicode_letter_cp(cp, ctx),
            "number:", is_unicode_number_cp(cp, ctx),
            "ws:", is_unicode_whitespace_cp(cp, ctx),
            "mark:", is_unicode_mark_cp(cp, ctx),
        )

    # Step 6: Test the full emoji string
    print()
    print("=== Full emoji string ===")
    var full = "emoji: 👩‍💻 🚀 ❤️"
    var full_pieces = pre_tokenize_gpt_oss(full)
    print("piece count:", len(full_pieces))
    for i in range(len(full_pieces)):
        var p = full_pieces[i]
        print("  piece", i, ":", repr(p), "bytes:", len(p.as_bytes()))
