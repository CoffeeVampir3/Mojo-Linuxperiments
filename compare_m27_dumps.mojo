from std.pathlib import Path


comptime DECODE_DIR = "m27_dump_decode"
comptime PREFILL_DIR = "m27_dump_prefill"
comptime MAX_PROMPT_POS = -1  # < 0 means compare every dumped prompt position.
comptime PROGRESS_INTERVAL = 20000


def basename(path: String) -> String:
    var parts = path.split("/")
    if len(parts) == 0:
        return path
    return String(parts[len(parts) - 1])


def prompt_pos(name: String) raises -> Int:
    if not name.startswith("p"):
        return -1
    var sep = name.find("_")
    if sep <= 1:
        return -1
    return atol(String(name[byte=1:sep]))


def should_compare(name: String) raises -> Bool:
    if not name.endswith(".bin"):
        return False
    var pos = prompt_pos(name)
    if pos < 0:
        return False
    comptime if MAX_PROMPT_POS < 0:
        return True
    else:
        return pos <= MAX_PROMPT_POS


def compare_one(name: String) raises -> Bool:
    var decode_path = Path(DECODE_DIR) / name
    var prefill_path = Path(PREFILL_DIR) / name

    if not decode_path.exists():
        print("missing decode file:", name)
        return False
    if not prefill_path.exists():
        print("missing prefill file:", name)
        return False

    var decode = decode_path.read_bytes()
    var prefill = prefill_path.read_bytes()

    if len(decode) != len(prefill):
        print(
            "size mismatch:",
            name,
            "decode",
            len(decode),
            "prefill",
            len(prefill),
        )
        return False

    for i in range(len(decode)):
        if decode[i] != prefill[i]:
            print(
                "byte mismatch:",
                name,
                "offset",
                i,
                "decode",
                decode[i],
                "prefill",
                prefill[i],
            )
            return False

    return True


def main() raises:
    var prefill_dir = Path(PREFILL_DIR)
    if not prefill_dir.exists():
        print("missing prefill dump dir:", PREFILL_DIR)
        return
    var decode_dir = Path(DECODE_DIR)
    if not decode_dir.exists():
        print("missing decode dump dir:", DECODE_DIR)
        return

    var compared = 0
    var skipped = 0
    for entry in prefill_dir.listdir():
        var name = basename(String(entry))
        if not should_compare(name):
            skipped += 1
            continue
        if not compare_one(name):
            print("compared before mismatch:", compared)
            print("skipped:", skipped)
            return
        compared += 1
        if compared % PROGRESS_INTERVAL == 0:
            print("progress:", compared, "files matched")

    print("compared:", compared)
    print("skipped:", skipped)
    print("mismatched:", 0)
