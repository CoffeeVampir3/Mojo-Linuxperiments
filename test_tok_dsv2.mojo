from std.pathlib import Path
from tokenizer import load_tokenizer

def main():
    var tok_opt = load_tokenizer(Path("checkpoints/deepseekv2-lite/tokenizer.json"))
    if not tok_opt:
        print("failed")
        return
    var tok = tok_opt.take()
    var text = "Hello, world! This is a test 123."
    var ids = tok.encode(text)
    print("our ids:", end="")
    for i in range(len(ids)):
        print("", ids[i], end="")
    print()
    print("our len:", len(ids))
    for i in range(len(ids)):
        var id_list = List[Int]()
        id_list.append(ids[i])
        print(" ", i, ":", ids[i], "->", repr(tok.decode(id_list)))
