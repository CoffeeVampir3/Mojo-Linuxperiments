from sys.info import size_of
from os.atomic import Atomic, Consistency

comptime AtomicInt32 = Atomic[DType.int32]

struct SharedPoolState:
    # Cache-line sizing to avoid false sharing.
    comptime DispatchPadBytes = 64 - (
        size_of[type_of(Self().work_available)]()
        + size_of[type_of(Self().shutdown)]()
        + size_of[type_of(Self().func_ptr)]()
    )
    comptime DonePadBytes = 64 - size_of[type_of(Self().work_done)]()

    comptime mew = size_of[Self().work_done]()

    var work_available: AtomicInt32  # Workers decrement to claim work
    var shutdown: AtomicInt32        # Shutdown signal
    var func_ptr: Int64              # Kernel entry bits
    var pad0: InlineArray[UInt8, Self.DispatchPadBytes]

    var work_done: AtomicInt32       # Workers increment when done
    var pad1: InlineArray[UInt8, Self.DonePadBytes]

    fn __init__(out self):
        self.work_available = AtomicInt32(0)
        self.shutdown = AtomicInt32(0)
        self.func_ptr = 0
        self.pad0 = InlineArray[UInt8, Self.DispatchPadBytes](uninitialized=True)
        self.work_done = AtomicInt32(0)
        self.pad1 = InlineArray[UInt8, Self.DonePadBytes](uninitialized=True)

fn main():
    pass