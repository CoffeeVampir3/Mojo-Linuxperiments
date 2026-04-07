from std.sys.info import size_of
from std.memory import UnsafePointer
from std.os.atomic import Atomic

comptime AtomicInt32 = Atomic[DType.int32]

# Single-pointer ABI: worker passes pointer to NUMA-local mailbox data area.
comptime KernelFn = def (Int) -> None

# Mailbox data area: 256 bytes (32 Int slots), stored inline in each worker's
# mailbox on the worker's NUMA node. Dispatch copies the caller's args struct
# into this area; the worker reads locally.
comptime MAILBOX_DATA_SLOTS = 32
comptime MAILBOX_DATA_BYTES = 256


# ============================================================================
# Typed dispatch trampoline
# ============================================================================


def typed_trampoline[
    Args: Copyable & ImplicitlyCopyable,
    kernel: def (Args) -> None,
](data_ptr: Int):
    """Reconstruct Args from mailbox data pointer, call kernel.

    Both Args and kernel are comptime parameters, so this is a plain
    non-capturing function. Each (Args, kernel) pair monomorphizes to
    a unique function pointer extractable by the dispatch machinery.
    """
    kernel(UnsafePointer[Args, MutAnyOrigin](unsafe_from_address=data_ptr)[])


# ============================================================================
# Join flag — lives on the main thread's NUMA node
# ============================================================================

@align(64)
struct JoinFlag:
    """Completion flag on the main thread's NUMA node.
    Main thread reads locally. Worker writes remotely once."""
    var done: AtomicInt32  # 0=running, 1=complete
    var timestamp: Int     # worker writes perf_counter_ns() before setting done

    def __init__(out self):
        self.done = AtomicInt32(0)
        self.timestamp = 0


# ============================================================================
# Stack / slot layout
# ============================================================================

struct SlotLayout(TrivialRegisterPassable):
    comptime TLS_SIZE = 256
    comptime TCB_SIZE = 64
    comptime TCB_SELF_OFFSET = 0x10
    comptime TCB = Self.TLS_SIZE
    comptime CHILD_TID = Self.TCB + Self.TCB_SIZE
    comptime WORKER_ID = Self.CHILD_TID + 8
    comptime WORKER_MAGIC = Self.WORKER_ID + 8
    comptime WORKER_MAGIC_VALUE = Int(0x4255525354574B52)  # "BURSTWKR"
    comptime WORKER_ID_FROM_FS = Self.WORKER_ID - Self.TCB
    comptime WORKER_MAGIC_FROM_FS = Self.WORKER_MAGIC - Self.TCB
    comptime HEADER = ((Self.WORKER_MAGIC + 8 + 4095) // 4096) * 4096
    comptime GUARD = 4096
    comptime ALTSTACK_SIZE = 64 * 1024
    comptime ALT_GUARD = Self.GUARD
    comptime DEFAULT_STACK = 2 * 1024 * 1024


def compute_slot_size(stack_size: Int) -> Int:
    debug_assert(stack_size >= SlotLayout.GUARD and stack_size % SlotLayout.GUARD == 0,
        "stack_size must be a multiple of 4096 (>= 4096)")
    var raw = (SlotLayout.HEADER + SlotLayout.GUARD + stack_size
             + SlotLayout.ALT_GUARD + SlotLayout.ALTSTACK_SIZE)
    return ((raw + SlotLayout.GUARD - 1) // SlotLayout.GUARD) * SlotLayout.GUARD


# ============================================================================
# Helpers
# ============================================================================

@always_inline
def ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)
