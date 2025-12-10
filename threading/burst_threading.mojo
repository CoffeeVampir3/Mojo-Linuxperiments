"""Persistent worker thread pool for low-latency burst dispatch.

Workers spawn once at pool creation and persist until destruction.
Hot mode: workers spin on PAUSE (zero syscall latency, burns CPU).
Cold mode: workers futex_wait (syscall on wake, power efficient).
"""

from sys.intrinsics import inlined_assembly
from sys.info import size_of
from memory import UnsafePointer, memcpy
from os.atomic import Atomic, Consistency
import threading.linux as linux
from notstdcollections import HeapMoveArray
from numa import NumaInfo, CpuMask

comptime AtomicInt32 = Atomic[DType.int32]

fn ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)

fn get_fs_base() -> Int:
    return Int(inlined_assembly["mov %fs:0, $0", Int64, constraints="=r"]())

fn pause():
    inlined_assembly["pause", NoneType, constraints="~{memory}"]()

# Memory layout per worker slot:
# [TLS 256B][TCB 64B][child_tid 4B][pad][WorkerState 128B aligned][Guard 4KB][Stack]
@register_passable("trivial")
struct SlotLayout:
    comptime TLS_SIZE = 256
    comptime TCB_SIZE = 64
    comptime TCB_SELF_OFFSET = 0x10
    comptime TCB = Self.TLS_SIZE
    comptime CHILD_TID = Self.TCB + Self.TCB_SIZE
    comptime WORKER_STATE = ((Self.CHILD_TID + 4 + 63) // 64) * 64
    comptime WORKER_STATE_SIZE = 128  # 2 cache lines to prevent false sharing
    comptime HEADER = ((Self.WORKER_STATE + Self.WORKER_STATE_SIZE + 4095) // 4096) * 4096
    comptime GUARD = 4096
    comptime DEFAULT_STACK = 64 * 1024

fn slot_size[stack_size: Int]() -> Int:
    return SlotLayout.HEADER + SlotLayout.GUARD + stack_size

# WorkerState uses SPSC (single-producer single-consumer) protocol:
# - Main thread is the only writer to ready/func_ptr/args (producer)
# - Worker is the only writer to done (consumer)
# - No atomic RMW needed, just proper memory ordering
#
# Cache line layout (128 bytes total, 2 cache lines):
# Line 1 (producer-written): ready, func_ptr, args - written by main, read by worker
# Line 2 (consumer-written): done - written by worker, read by main
# This prevents false sharing between main thread and worker
struct WorkerState:
    # === Cache line 1: Producer (main thread) writes, worker reads ===
    var ready: AtomicInt32      # Signal from main -> worker (SPSC flag)
    var _pad0: Int32            # Padding for alignment
    var func_ptr: Int64         # -1 = shutdown signal
    var arg0: Int64
    var arg1: Int64
    var arg2: Int64
    var arg3: Int64
    var arg4: Int64
    var arg5: Int64
    var _pad1: Int64            # Pad to 64 bytes

    # === Cache line 2: Consumer (worker) writes, main thread reads ===
    var done: AtomicInt32       # Signal from worker -> main (SPSC flag)
    var _pad2: InlineArray[UInt8, 60]  # Pad to 64 bytes

    fn __init__(out self):
        self.ready = AtomicInt32(0)
        self._pad0 = 0
        self.func_ptr = 0
        self.arg0 = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0
        self.arg4 = 0
        self.arg5 = 0
        self._pad1 = 0
        self.done = AtomicInt32(0)
        self._pad2 = InlineArray[UInt8, 60](uninitialized=True)

struct SharedPoolState:
    # === Cache line 1: Work dispatch (main writes, workers read) ===
    var work_available: AtomicInt32  # Workers decrement to claim work
    var shutdown: AtomicInt32        # -1 = shutdown signal
    var func_ptr: Int64
    var arg0: Int64
    var arg1: Int64
    var arg2: Int64
    var arg3: Int64
    var arg4: Int64
    var arg5: Int64

    # === Cache line 2: Completion tracking (workers write, main reads) ===
    var work_done: AtomicInt32       # Workers increment when done
    var _pad2: InlineArray[UInt8, 60]

    fn __init__(out self):
        self.work_available = AtomicInt32(0)
        self.shutdown = AtomicInt32(0)
        self.func_ptr = 0
        self.arg0 = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0
        self.arg4 = 0
        self.arg5 = 0
        self.work_done = AtomicInt32(0)
        self._pad2 = InlineArray[UInt8, 60](uninitialized=True)

struct WorkerSlot(Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var child_tid: UnsafePointer[Int32, MutAnyOrigin]
    var state: UnsafePointer[WorkerState, MutAnyOrigin]
    var stack_top: UnsafePointer[UInt8, MutAnyOrigin]

    fn __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.state = UnsafePointer[WorkerState, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.WORKER_STATE)
        self.stack_top = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)

    fn __moveinit__(out self, deinit other: Self):
        self.base = other.base
        self.child_tid = other.child_tid
        self.state = other.state
        self.stack_top = other.stack_top

    @always_inline
    fn is_alive(self) -> Bool:
        return self.child_tid[] != 0

struct WorkerStackHead[mask_size: Int]:
    var entry: Int64
    var slot_base: Int64              # Base address, state/child_tid derived from this
    var parent_fs: Int64              # Parent's FS base for TLS copy
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]  # Pool's shared state (not in slot)
    var futex_flags: Int64
    var cpu_mask: CpuMask[Self.mask_size]  # Embedded mask, copied to child's stack

    fn __init__(out self, entry: Int64, slot_base: Int64, parent_fs: Int64,
                shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
                futex_flags: Int64, var cpu_mask: CpuMask[Self.mask_size]):
        self.entry = entry
        self.slot_base = slot_base
        self.parent_fs = parent_fs
        self.shared = shared
        self.futex_flags = futex_flags
        self.cpu_mask = cpu_mask^

struct BurstPool[stack_size: Int = SlotLayout.DEFAULT_STACK, mask_size: Int = 128](Movable):
    comptime slot_size = slot_size[Self.stack_size]()
    var slots: HeapMoveArray[WorkerSlot]
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]
    var arena_base: Int
    var capacity: Int
    var cpu_mask: CpuMask[Self.mask_size]
    var numa_node: Optional[Int]
    var futex_flags: Int
    var pinned: Bool
    var workers_alive: Bool

    fn __init__(out self, capacity: Int, var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](), numa_node: Optional[Int] = None):
        self.capacity = capacity
        self.slots = HeapMoveArray[WorkerSlot](capacity)
        self.arena_base = 0
        self.shared = UnsafePointer[SharedPoolState, MutAnyOrigin]()
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.workers_alive = False

        # Use plain PRIVATE futexes (not NUMA-bucketed) to allow CHILD_CLEARTID to work
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE

        var arena_size = Self.slot_size * capacity + size_of[SharedPoolState]()
        self.arena_base = linux.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.NORESERVE
        ](0, arena_size)
        if self.arena_base < 0:
            return

        if numa_node is not None:
            var nodemask = UInt64(1) << numa_node.value()
            if linux.sys_mbind[policy=linux.Mempolicy.BIND](self.arena_base, arena_size, nodemask) < 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return

        self.shared = UnsafePointer[SharedPoolState, MutAnyOrigin](
            unsafe_from_address=self.arena_base + Self.slot_size * capacity
        )
        self.shared[] = SharedPoolState()

        for i in range(capacity):
            var slot_base = self.arena_base + i * Self.slot_size
            if linux.sys_mprotect(slot_base + SlotLayout.HEADER, SlotLayout.GUARD, linux.Prot.NONE) != 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return
            var slot = WorkerSlot(slot_base)
            slot.child_tid[] = 0
            slot.state[] = WorkerState()
            self.slots.push(slot^)

        self._spawn_workers()

    fn __moveinit__(out self, deinit other: Self):
        self.slots = other.slots^
        self.shared = other.shared
        self.arena_base = other.arena_base
        self.capacity = other.capacity
        self.cpu_mask = other.cpu_mask.copy()
        self.numa_node = other.numa_node
        self.futex_flags = other.futex_flags
        self.pinned = other.pinned
        self.workers_alive = other.workers_alive
        _ = other.arena_base
        _ = other.workers_alive

    fn __del__(deinit self):
        if self.arena_base == 0:
            return

        if self.workers_alive:
            # Signal shutdown and wake all workers
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=self.shared[].shutdown.value), 1)
            _ = linux.sys_futex_wake(Int(self.shared), self.capacity, self.futex_flags)

            # Wait for all workers to exit
            # CHILD_CLEARTID does legacy futex(FUTEX_WAKE) without PRIVATE flag,
            # so we must wait with shared (non-private) futex to match the hash bucket
            comptime shared_futex_flags = linux.Futex2.SIZE_U32
            for i in range(self.capacity):
                while self.slots[i][].is_alive():
                    _ = linux.sys_futex_wait(
                        Int(self.slots[i][].child_tid),
                        Int(self.slots[i][].child_tid[]),
                        shared_futex_flags)

        _ = linux.sys_munmap(self.arena_base, Self.slot_size * self.capacity + size_of[SharedPoolState]())

    fn __bool__(self) -> Bool:
        return self.arena_base != 0 and self.workers_alive

    fn __len__(self) -> Int:
        return self.capacity

    @staticmethod
    fn for_numa_node(numa: NumaInfo, node: Int) -> Self:
        return Self(numa.cpus_on_node(node), numa.get_node_mask[Self.mask_size](node), node)

    @staticmethod
    fn for_numa_node_excluding(numa: NumaInfo, node: Int, exclude_cpu: Int) -> Self:
        var mask = numa.get_node_mask[Self.mask_size](node)
        var cap = numa.cpus_on_node(node)
        if mask.test(exclude_cpu):
            mask.clear(exclude_cpu)
            cap -= 1
        return Self(cap, mask^, node)


    fn sync[F: AnyTrivialRegType, *Ts: ImplicitlyCopyable](mut self, func: F, *args: *Ts):
        """Dispatch func to all workers and wait for completion."""
        var packed_args = InlineArray[Int64, 6](0)
        @parameter
        for i in range(args.__len__()):
            comptime T = type_of(args[i])
            constrained[size_of[T]() == 8, "args must be 8 bytes"]()
            var arg = args[i]
            packed_args[i] = UnsafePointer(to=arg).bitcast[Int64]()[]

        var func_copy = func
        var func_ptr = UnsafePointer(to=func_copy).bitcast[Int64]()[]

        # Write work data to shared state
        self.shared[].func_ptr = func_ptr
        self.shared[].arg0 = packed_args[0]
        self.shared[].arg1 = packed_args[1]
        self.shared[].arg2 = packed_args[2]
        self.shared[].arg3 = packed_args[3]
        self.shared[].arg4 = packed_args[4]
        self.shared[].arg5 = packed_args[5]

        # Reset done counter, then publish work_available with release semantics
        AtomicInt32.store[ordering=Consistency.MONOTONIC](
            UnsafePointer(to=self.shared[].work_done.value), 0)
        AtomicInt32.store[ordering=Consistency.RELEASE](
            UnsafePointer(to=self.shared[].work_available.value), Int32(self.capacity))

        # Single futex wake to wake all sleeping workers
        _ = linux.sys_futex_wake(Int(self.shared), self.capacity, self.futex_flags)

        # Wait for all work to complete
        while AtomicInt32.load[ordering=Consistency.ACQUIRE](
                UnsafePointer(to=self.shared[].work_done.value)) < Int32(self.capacity):
            pause()

    fn _spawn_workers(mut self):
        var parent_fs = get_fs_base()

        # Build list of CPUs from mask for 1:1 pinning
        var cpu_list = InlineArray[Int, 1024](-1)
        var cpu_count = 0
        if self.pinned:
            for cpu in range(Self.mask_size * 8):
                if self.cpu_mask.test(cpu):
                    cpu_list[cpu_count] = cpu
                    cpu_count += 1

        for i in range(self.capacity):
            # Create per-worker mask with single CPU for 1:1 pinning
            var worker_mask = CpuMask[Self.mask_size]()
            if self.pinned and i < cpu_count:
                worker_mask.set(cpu_list[i])

            var stack_top_addr = Int(self.slots[i][].stack_top) + Self.stack_size
            var stack_head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](stack_head_addr)
            var worker_main_copy = worker_main[Self.mask_size]
            head[] = WorkerStackHead[Self.mask_size](
                entry=UnsafePointer(to=worker_main_copy).bitcast[Int64]()[],
                slot_base=Int64(Int(self.slots[i][].base)),
                parent_fs=Int64(parent_fs),
                shared=self.shared,
                futex_flags=Int64(self.futex_flags),
                cpu_mask=worker_mask^,
            )

            var tcb_addr = Int(self.slots[i][].base) + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i][].stack_top),
                stack_head_addr - Int(self.slots[i][].stack_top),
                tcb_addr,
                Int(self.slots[i][].child_tid)
            )
            var result = clone3_with_entry(Int(UnsafePointer(to=clone_args)), size_of[linux.Clone3Args]())
            _ = clone_args
            _ = head[]
            if result < 0:
                return

        self.workers_alive = True

fn clone3_with_entry(clone_args_ptr: Int, clone_args_size: Int) -> Int:
    # Child diverges via ret to entry point on stack head
    return Int(inlined_assembly[
        "mov $$435, %rax\nsyscall\ntest %rax, %rax\njnz 1f\nmov %rsp, %rdi\nret\n1:",
        Int64, Int64, Int64,
        constraints="={rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
    ](clone_args_ptr, clone_args_size))

fn worker_main[mask_size: Int](stack_head_ptr: Int):
    var head_ptr = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var futex_flags = Int(head_ptr[].futex_flags)
    var slot_base = Int(head_ptr[].slot_base)
    var shared = head_ptr[].shared

    # Derive addresses from slot_base
    var tcb_addr = slot_base + SlotLayout.TCB
    var child_tid_ptr = ptr[Int32](slot_base + SlotLayout.CHILD_TID)

    # TLS init must happen before any print/stdlib calls
    memcpy(dest=ptr[Int8](slot_base), src=ptr[Int8](Int(head_ptr[].parent_fs) - SlotLayout.TLS_SIZE), count=SlotLayout.TLS_SIZE)
    memcpy(dest=ptr[Int8](tcb_addr), src=ptr[Int8](Int(head_ptr[].parent_fs)), count=SlotLayout.TCB_SIZE)
    ptr[Int64](tcb_addr + SlotLayout.TCB_SELF_OFFSET)[] = Int64(tcb_addr)

    # Pin to CPU if mask has any bits set
    if head_ptr[].cpu_mask.count() > 0:
        var ret = linux.sys_sched_setaffinity(0, mask_size, Int(head_ptr[].cpu_mask.ptr()))
        if ret != 0:
            print("sched_setaffinity failed:", ret)

    comptime SPIN_LIMIT = 1000  # Spin iterations before sleeping

    while True:
        # Check for shutdown
        if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
            break

        # Try to claim work by atomically decrementing work_available
        var avail = shared[].work_available.load[ordering=Consistency.ACQUIRE]()

        if avail > 0:
            # Try to claim one unit of work
            var old = shared[].work_available.fetch_sub[ordering=Consistency.ACQUIRE_RELEASE](1)
            if old > 0:
                # Successfully claimed work - execute it
                var func_addr = shared[].func_ptr
                UnsafePointer(to=func_addr).bitcast[fn()]()[]()

                # Signal completion
                _ = shared[].work_done.fetch_add[ordering=Consistency.ACQUIRE_RELEASE](1)
                continue
            else:
                # Lost the race, undo our decrement
                _ = shared[].work_available.fetch_add[ordering=Consistency.MONOTONIC](1)

        # No work available - spin briefly then sleep
        var spins = 0
        while shared[].work_available.load[ordering=Consistency.MONOTONIC]() <= 0:
            if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                break
            if spins < SPIN_LIMIT:
                pause()
                spins += 1
            else:
                # Sleep on work_available address, expecting value <= 0
                _ = linux.sys_futex_wait(Int(shared), 0, futex_flags)
                spins = 0  # Reset spin count after wake

    # CHILD_CLEARTID handles clearing child_tid and futex wake automatically
    linux.sys_exit()
