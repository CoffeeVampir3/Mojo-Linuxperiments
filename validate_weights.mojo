"""Compare raw weight dumps between TP=1 and TP=2.

TP=1 gate_proj L0 r0 should be [2112, 2816] = 5947392 bytes.
TP=2 gate_proj L0 r0 should be the first [1056, 2816] = 2973696 bytes of TP=1.
TP=2 gate_proj L0 r1 should be the next [1056, 2816] = 2973696 bytes of TP=1.
"""


comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112


def read_bytes(path: String) -> Optional[List[UInt8]]:
    try:
        var fh = FileHandle(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        var out = List[UInt8](capacity=len(raw))
        for i in range(len(raw)):
            out.append(raw[i])
        return out^
    except:
        return None


def compare_bytes(label: String, a: List[UInt8], b: List[UInt8], a_off: Int, count: Int):
    var mismatches = 0
    var first_mismatch = -1
    for i in range(count):
        if a[a_off + i] != b[i]:
            mismatches += 1
            if first_mismatch < 0:
                first_mismatch = i
    if mismatches == 0:
        print(label, " MATCH (", count, "bytes)")
    else:
        print(label, " MISMATCH:", mismatches, "/", count, "bytes differ, first at", first_mismatch)


def i8_stats(label: String, data: List[UInt8], count: Int):
    var sum_abs = 0
    var sum_val = 0
    for i in range(count):
        var raw = data[i]
        var v = Int(raw) if raw < 128 else Int(raw) - 256
        sum_val += v
        if v < 0:
            sum_abs -= v
        else:
            sum_abs += v
    print(label, " n:", count, " sum:", sum_val, " abs_sum:", sum_abs)


def main():
    var tp = INTERMEDIATE // 2
    var data_rows = tp
    var data_bytes = data_rows * HIDDEN

    print("Expected: TP=1 gate_proj =", INTERMEDIATE * HIDDEN, "bytes")
    print("Expected: TP=2 per-rank  =", data_bytes, "bytes")
    print()

    var tp1 = read_bytes("taps/weights_tp1/gate_proj_L0_r0.bin")
    var tp2_r0 = read_bytes("taps/weights_tp2/gate_proj_L0_r0.bin")
    var tp2_r1 = read_bytes("taps/weights_tp2/gate_proj_L0_r1.bin")

    if not tp1:
        print("missing taps/weights_tp1/gate_proj_L0_r0.bin")
        return
    if not tp2_r0:
        print("missing taps/weights_tp2/gate_proj_L0_r0.bin")
        return
    if not tp2_r1:
        print("missing taps/weights_tp2/gate_proj_L0_r1.bin")
        return

    var a = tp1.take()
    var b0 = tp2_r0.take()
    var b1 = tp2_r1.take()

    print("TP=1 file size:", len(a))
    print("TP=2 r0 file size:", len(b0))
    print("TP=2 r1 file size:", len(b1))
    print()

    i8_stats("TP=1 full      ", a, len(a))
    i8_stats("TP=1 first half", a, data_bytes)
    i8_stats("TP=2 r0        ", b0, len(b0))
    i8_stats("TP=2 r1        ", b1, len(b1))
    print()

    if len(a) >= 2 * data_bytes and len(b0) >= data_bytes and len(b1) >= data_bytes:
        compare_bytes("r0 vs tp1[0:half]", a, b0, 0, data_bytes)
        compare_bytes("r1 vs tp1[half:]", a, b1, data_bytes, data_bytes)
    else:
        print("unexpected sizes, cannot compare")
