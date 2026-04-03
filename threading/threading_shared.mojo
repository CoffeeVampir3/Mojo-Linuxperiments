from std.sys.info import size_of
from std.memory import UnsafePointer
from std.os.atomic import Atomic

comptime AtomicInt32 = Atomic[DType.int32]
comptime KernelFn = def (Int, Int, Int, Int, Int, Int) -> None

# Aligned to avoid false sharing.


# ============================================================================
# Kernel call ABI
# ============================================================================

@fieldwise_init
struct ArgPack(Copyable, ImplicitlyCopyable):
    var arg0: Int
    var arg1: Int
    var arg2: Int
    var arg3: Int
    var arg4: Int
    var arg5: Int

    def __init__(out self):
        self.arg0 = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0
        self.arg4 = 0
        self.arg5 = 0


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
