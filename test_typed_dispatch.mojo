"""Prototype: typed dispatch via expanded mailbox data area.

Verifies Mojo lets us:
  1. Copy an arbitrary struct into a fixed-size Int-aligned data area
  2. Generate a comptime trampoline that reconstructs the struct and calls the kernel
  3. Extract a function pointer from the trampoline (non-capturing)
  4. Round-trip mixed field types (Int, Float32) without corruption
  5. Simulate multi-job dispatch with per-worker args
"""

from std.sys.info import size_of
from std.memory import UnsafePointer


# ============================================================================
# Expanded mailbox data area
# ============================================================================

comptime MAILBOX_DATA_SLOTS = 32  # 32 Int slots = 256 bytes
comptime MAILBOX_DATA_BYTES = 256

# New single-pointer ABI: kernel receives pointer to data area
comptime TypedKernelFn = def (Int) -> None


@fieldwise_init
struct SimMailbox(Copyable, ImplicitlyCopyable):
    var func_ptr: Int
    var data: InlineArray[Int, MAILBOX_DATA_SLOTS]

    def __init__(out self):
        self.func_ptr = 0
        self.data = InlineArray[Int, MAILBOX_DATA_SLOTS](fill=0)


# ============================================================================
# Trampoline — module-level, parameterized by Args + kernel
#
# Both are comptime parameters, so this is a plain non-capturing function.
# Each (Args, kernel) pair monomorphizes to a unique function pointer.
# ============================================================================


def typed_trampoline[
    Args: Copyable & ImplicitlyCopyable,
    kernel: def (Args) -> None,
](data_ptr: Int):
    kernel(UnsafePointer[Args, MutAnyOrigin](unsafe_from_address=data_ptr)[])


# ============================================================================
# dispatch_args — the new API surface
# ============================================================================


def dispatch_args[
    Args: Copyable & ImplicitlyCopyable,
    kernel: def (Args) -> None,
](mut mailbox: SimMailbox, args: Args):
    comptime assert size_of[Args]() <= MAILBOX_DATA_BYTES, "args exceed mailbox capacity"

    # Copy struct into the data area (same as mailbox pass-1 remote write)
    UnsafePointer(to=mailbox.data[0]).bitcast[Args]()[] = args

    # Store the monomorphized trampoline's function pointer
    var tramp = typed_trampoline[Args, kernel]
    mailbox.func_ptr = UnsafePointer(to=tramp).bitcast[Int]()[]


def invoke(mailbox: SimMailbox):
    """Simulate worker_main: call func_ptr with pointer to local data."""
    var data_ptr = Int(UnsafePointer(to=mailbox.data[0]))
    UnsafePointer(to=mailbox.func_ptr).bitcast[TypedKernelFn]()[](data_ptr)


# ============================================================================
# Test 1: Simple struct (24 bytes, fits in old ArgPack too)
# ============================================================================


@fieldwise_init
struct SimpleArgs(Copyable, ImplicitlyCopyable):
    var a: Int
    var b: Int
    var c: Int


def simple_kernel(args: SimpleArgs):
    var sum = args.a + args.b + args.c
    print("  a=" + String(args.a) + " b=" + String(args.b) + " c=" + String(args.c)
        + " sum=" + String(sum) + " " + ("PASS" if sum == 60 else "FAIL"))


# ============================================================================
# Test 2: MLA-sized struct (72 bytes — exceeds old 48-byte ArgPack)
# ============================================================================


@fieldwise_init
struct MLAArgs(Copyable, ImplicitlyCopyable):
    var q_ptr: Int
    var output_ptr: Int
    var ckv_ptr: Int
    var kr_ptr: Int
    var kvb_ptr: Int
    var start_head: Int
    var end_head: Int
    var pos: Int
    var seq_len: Int


def mla_kernel(args: MLAArgs):
    var ok = (
        args.q_ptr == 0xAAAA and args.output_ptr == 0xBBBB
        and args.ckv_ptr == 0xCCCC and args.kr_ptr == 0xDDDD
        and args.kvb_ptr == 0xEEEE
        and args.start_head == 0 and args.end_head == 8
        and args.pos == 42 and args.seq_len == 1
    )
    print("  9 fields intact: " + ("PASS" if ok else "FAIL"))


# ============================================================================
# Test 3: Mixed types (Float32 survives the round-trip)
# ============================================================================


@fieldwise_init
struct MixedArgs(Copyable, ImplicitlyCopyable):
    var ptr_val: Int
    var count: Int
    var scale: Float32
    var flag: Int


def mixed_kernel(args: MixedArgs):
    var scale_ok = args.scale > Float32(3.13) and args.scale < Float32(3.15)
    var count_ok = args.count == 100
    print("  scale=" + String(args.scale) + " count=" + String(args.count)
        + " " + ("PASS" if scale_ok and count_ok else "FAIL"))


# ============================================================================
# Test 4: Multi-job — different args per mailbox (simulates N workers)
# ============================================================================


def multi_kernel(args: SimpleArgs):
    print("  worker got a=" + String(args.a) + " b=" + String(args.b)
        + " c=" + String(args.c))


# ============================================================================
# Main
# ============================================================================


def main():
    print("=== typed dispatch prototype ===")
    print("mailbox data: " + String(MAILBOX_DATA_BYTES) + " bytes ("
        + String(MAILBOX_DATA_SLOTS) + " slots)")
    print("SimpleArgs: " + String(size_of[SimpleArgs]()) + " bytes")
    print("MLAArgs:    " + String(size_of[MLAArgs]()) + " bytes")
    print("MixedArgs:  " + String(size_of[MixedArgs]()) + " bytes")
    print()

    print("test 1: simple (3 fields)")
    var mb1 = SimMailbox()
    dispatch_args[SimpleArgs, simple_kernel](mb1, SimpleArgs(10, 20, 30))
    invoke(mb1)

    print("test 2: MLA-sized (9 fields, would not fit in old ArgPack)")
    var mb2 = SimMailbox()
    dispatch_args[MLAArgs, mla_kernel](mb2, MLAArgs(
        0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD, 0xEEEE, 0, 8, 42, 1,
    ))
    invoke(mb2)

    print("test 3: mixed types (Int + Float32)")
    var mb3 = SimMailbox()
    dispatch_args[MixedArgs, mixed_kernel](mb3, MixedArgs(0x1234, 100, Float32(3.14), 1))
    invoke(mb3)

    print("test 4: multi-job (4 workers, different args each)")
    var mailboxes = InlineArray[SimMailbox, 4](fill=SimMailbox())
    for i in range(4):
        dispatch_args[SimpleArgs, multi_kernel](
            mailboxes[i], SimpleArgs(i * 10, i * 20, i * 30),
        )
    for i in range(4):
        invoke(mailboxes[i])

    print()
    print("=== done ===")
