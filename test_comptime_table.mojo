from tokenizer.unicode_props import (
    WHITESPACE_RANGES, WHITESPACE_PAIR_COUNT,
    WHITESPACE_MIN, WHITESPACE_MAX,
)
from tokenizer.tokenizer import in_unicode_ranges
from std.memory import UnsafePointer


def check_via_materialize(cp: UInt32):
    var table = materialize[WHITESPACE_RANGES]()
    var ptr = UnsafePointer[UInt32, MutAnyOrigin](
        unsafe_from_address=Int(table.unsafe_ptr()))
    var result = in_unicode_ranges(cp, ptr, WHITESPACE_PAIR_COUNT)
    print("  materialize:", result)


def check_via_comptime(cp: UInt32):
    var lo = 0
    var hi = Int(WHITESPACE_PAIR_COUNT)
    while lo < hi:
        var mid = (lo + hi) // 2
        var start = WHITESPACE_RANGES[mid * 2]
        var end = WHITESPACE_RANGES[mid * 2 + 1]
        if cp < start:
            hi = mid
        elif cp > end:
            lo = mid + 1
        else:
            print("  comptime direct:", True)
            return
    print("  comptime direct:", False)


def main():
    var test_cps = List[UInt32]()
    var test_names = List[String]()
    test_cps.append(UInt32(0x0020)); test_names.append("U+0020 SPACE")
    test_cps.append(UInt32(0x200D)); test_names.append("U+200D ZWJ")
    test_cps.append(UInt32(0x2764)); test_names.append("U+2764 HEART")
    test_cps.append(UInt32(0x2005)); test_names.append("U+2005 FOUR-PER-EM")
    test_cps.append(UInt32(0x200A)); test_names.append("U+200A HAIR SPACE")
    test_cps.append(UInt32(0x200B)); test_names.append("U+200B ZERO WIDTH")

    for i in range(len(test_cps)):
        print(test_names[i], "(", test_cps[i], "):")
        check_via_comptime(test_cps[i])
        check_via_materialize(test_cps[i])
