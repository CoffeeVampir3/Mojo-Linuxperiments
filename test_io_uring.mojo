"""Test: io_uring read/write round-trip.

Creates a file via register_files[ReadWriteMode], writes an initial zero,
then iterates N rounds of read-verify-write to confirm both directions work.
"""

from std.pathlib import Path
from std.memory import alloc
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion, ReadWriteMode,
)


comptime TEST_FILE = "/tmp/test_io_uring_rw.bin"
comptime ROUNDS = 10
comptime INCREMENT = 5
comptime VALUE_SIZE = 8


def main():
    var ring = IoRing[32]()
    if not ring:
        print("FAIL: io_uring setup failed")
        return

    var paths = List[Path]()
    paths.append(Path(TEST_FILE))
    try:
        _ = ring.register_files[ReadWriteMode](paths)
    except err:
        print("FAIL: register_files:", err.error_message())
        return

    var read_buf = alloc[Int](1)
    var write_buf = alloc[Int](1)

    # Write initial zero through io_uring
    write_buf[] = 0
    try:
        _ = ring.submit_one(WriteOp(file_idx=0, offset=0, length=VALUE_SIZE, src=write_buf, id=0))
        _ = ring.wait()
    except err:
        print("FAIL: initial write:", err.error_message())
        read_buf.free()
        write_buf.free()
        return

    var all_passed = True

    for round in range(ROUNDS):
        var expected_value = round * INCREMENT

        # Read + verify + write in one try block
        try:
            read_buf[] = -1
            _ = ring.submit_one(ReadOp(file_idx=0, offset=0, length=VALUE_SIZE, dest=read_buf, id=0))
            var completions = ring.wait()
            if len(completions) == 0 or completions[0].result != Int32(VALUE_SIZE):
                print("FAIL round", round, ": read completion bad")
                all_passed = False
                break

            var got = read_buf[]
            if got != expected_value:
                print("FAIL round", round, ": expected", expected_value, "got", got)
                all_passed = False
                break

            write_buf[] = got + INCREMENT
            _ = ring.submit_one(WriteOp(file_idx=0, offset=0, length=VALUE_SIZE, src=write_buf, id=1))
            completions = ring.wait()
            if len(completions) == 0 or completions[0].result != Int32(VALUE_SIZE):
                print("FAIL round", round, ": write completion bad")
                all_passed = False
                break
        except err:
            print("FAIL round", round, ":", err.error_message())
            all_passed = False
            break

    if all_passed:
        # Final verification read
        try:
            read_buf[] = -1
            _ = ring.submit_one(ReadOp(file_idx=0, offset=0, length=VALUE_SIZE, dest=read_buf, id=2))
            _ = ring.wait()
            var final_value = read_buf[]
            var expected_final = ROUNDS * INCREMENT
            if final_value == expected_final:
                print("PASS: all", ROUNDS, "rounds verified, final value =", final_value)
            else:
                print("FAIL: final value expected", expected_final, "got", final_value)
        except err:
            print("FAIL: final read:", err.error_message())

    read_buf.free()
    write_buf.free()
