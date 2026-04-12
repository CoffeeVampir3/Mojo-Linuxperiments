"""Minimal reproducer: Mojo compiler crash on codegen.

`pixi run mojo build repro_packfn.mojo` aborts — either with
"Operand is null" at an `insertvalue` (LLVM IR verifier) or a full
compiler SIGSEGV, depending on the surrounding context.

Required ingredients (remove any one and the crash disappears):

  1. A `thin -> None` function-type alias `PackFn`.
  2. A top-level `def pack_noop(...)` matching `PackFn`'s signature.
  3. A struct `Entry` with a `pack_fn: PackFn` field, `@fieldwise_init`.
  4. A helper `make_entry(pack: PackFn) -> Entry` that constructs the
     struct through the typed parameter boundary.
  5. A caller that binds `var nop = pack_noop` to a local, uses that
     local via the helper AND performs a direct inline kwarg
     construction with the bare function reference, into the same
     `List[Entry]` via `append`.
"""

from std.memory import UnsafePointer


comptime PackFn = def(
    UnsafePointer[UInt8, MutAnyOrigin],
    UnsafePointer[UInt8, MutAnyOrigin],
    Int, Int,
) thin -> None


def pack_noop(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    pass


@fieldwise_init
struct Entry(Copyable):
    var pack_fn: PackFn


def make_entry(pack: PackFn) -> Entry:
    return Entry(pack_fn=pack)


def build() -> List[Entry]:
    var out = List[Entry]()
    var nop = pack_noop
    out.append(make_entry(nop))                 # callee construction via `pack: PackFn`
    out.append(Entry(pack_fn=pack_noop))        # inline kwarg with bare function ref
    return out^


def main():
    var out = build()
    print("len:", len(out))
