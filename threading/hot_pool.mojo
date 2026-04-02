"""HotPool — per-worker mailbox dispatch with zero CAS contention.

Each worker spins on its own cache-line-aligned slot. The dispatcher
writes job info directly to each slot. No shared atomic counter,
no thundering herd, no CAS storms.

Designed for decode-style workloads: few jobs dispatched to many workers,
with begin_forward/end_forward lifecycle for sustained hot spinning.

Workers are pinned to cores with configurable headroom (default 2) to
prevent scheduler deadlock under sustained spin.

Dispatch: O(N) sequential stores to N worker slots.
Claim:    Each worker reads its own slot — zero cross-core traffic.
Join:     Spin on shared work_done counter (staggered decrements, no storm).
"""

from std.sys.info import size_of
from std.memory import UnsafePointer, memcpy
import linux.sys as linux
from numa import NumaInfo, CpuMask
from notstdcollections import HeapMoveArray


comptime AtomicInt32 = Atomic[DType.int32]
from std.os.atomic import Atomic, Consistency

comptime KernelFn = def(Int, Int, Int, Int, Int, Int)

@fieldwise_init
struct ArgPack(TrivialRegisterPassable):
    var arg0: Int
    var arg1: Int
    var arg2: Int
    var arg3: Int
    var arg4: Int
    var arg5: Int
    var pad0: Int
    var pad1: Int

    def __init__(out self):
        self.arg0 = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0
        self.arg4 = 0
        self.arg5 = 0
        self.pad0 = 0
        self.pad1 = 0


def ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)


# ============================================================================
# Per-worker mailbox — one cache line per worker
# ============================================================================

@align(64)
struct WorkerSlotMailbox:
    """Per-worker dispatch slot. Dispatcher writes, worker reads.

    job_ready: 0 = idle, 1 = job available. Worker spins on this.
    func_ptr:  Kernel function pointer (set before job_ready=1).
    pack:      Arguments for this worker's job.
    """
    var job_ready: AtomicInt32
    var func_ptr: Int
    var pack: ArgPack

    def __init__(out self):
        self.job_ready = AtomicInt32(0)
        self.func_ptr = 0
        self.pack = ArgPack()


# ============================================================================
# Shared state — minimal, separate cache lines
# ============================================================================

@align(64)
struct HotPoolShared:
    var work_done: AtomicInt32
    var shutdown: AtomicInt32
    var hot_mode: AtomicInt32

    def __init__(out self):
        self.work_done = AtomicInt32(0)
        self.shutdown = AtomicInt32(0)
        self.hot_mode = AtomicInt32(0)


# ============================================================================
# Stack layout (reuses BurstPool's slot structure)
# ============================================================================

struct SlotLayout(TrivialRegisterPassable):
    comptime TLS_SIZE = 256
    comptime TCB_SIZE = 64
    comptime TCB_SELF_OFFSET = 0x10
    comptime TCB = Self.TLS_SIZE
    comptime CHILD_TID = Self.TCB + Self.TCB_SIZE
    comptime WORKER_ID = Self.CHILD_TID + 8
    comptime WORKER_MAGIC = Self.WORKER_ID + 8
    comptime WORKER_MAGIC_VALUE = Int(0x484F54504F4F4C57)  # "HOTPOOLW"
    comptime WORKER_ID_FROM_FS = Self.WORKER_ID - Self.TCB
    comptime WORKER_MAGIC_FROM_FS = Self.WORKER_MAGIC - Self.TCB
    comptime HEADER = ((Self.WORKER_MAGIC + 8 + 4095) // 4096) * 4096
    comptime GUARD = 4096
    comptime ALTSTACK_SIZE = 64 * 1024
    comptime ALT_GUARD = Self.GUARD
    comptime DEFAULT_STACK = 64 * 1024

def slot_size[stack_size: Int]() -> Int:
    comptime assert stack_size >= SlotLayout.GUARD and stack_size % SlotLayout.GUARD == 0
    var raw = SlotLayout.HEADER + SlotLayout.GUARD + stack_size + SlotLayout.ALT_GUARD + SlotLayout.ALTSTACK_SIZE
    return ((raw + SlotLayout.GUARD - 1) // SlotLayout.GUARD) * SlotLayout.GUARD


struct WorkerSlotMem(Movable, ImplicitlyDestructible):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var child_tid: UnsafePointer[Int32, MutAnyOrigin]
    var stack_top: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.stack_top = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)

    @always_inline
    def is_alive(self) -> Bool:
        return self.child_tid[] != 0


struct WorkerStackHead[mask_size: Int]:
    var entry: Int
    var slot_base: Int
    var worker_id: Int
    var parent_fs: Int
    var shared: UnsafePointer[HotPoolShared, MutAnyOrigin]
    var mailbox: UnsafePointer[WorkerSlotMailbox, MutAnyOrigin]
    var futex_flags: Int
    var altstack_base: Int
    var altstack_size: Int
    var pinned: Int
    var cpu_mask: CpuMask[Self.mask_size]

    def __init__(out self, entry: Int, slot_base: Int, parent_fs: Int,
                worker_id: Int,
                shared: UnsafePointer[HotPoolShared, MutAnyOrigin],
                mailbox: UnsafePointer[WorkerSlotMailbox, MutAnyOrigin],
                futex_flags: Int,
                altstack_base: Int, altstack_size: Int, pinned: Int,
                var cpu_mask: CpuMask[Self.mask_size]):
        self.entry = entry
        self.slot_base = slot_base
        self.worker_id = worker_id
        self.parent_fs = parent_fs
        self.shared = shared
        self.mailbox = mailbox
        self.futex_flags = futex_flags
        self.altstack_base = altstack_base
        self.altstack_size = altstack_size
        self.pinned = pinned
        self.cpu_mask = cpu_mask^


# ============================================================================
# HotPool
# ============================================================================

struct HotPool[stack_size: Int = SlotLayout.DEFAULT_STACK, mask_size: Int = 128](Movable):
    comptime slot_size = slot_size[Self.stack_size]()
    var slots: HeapMoveArray[WorkerSlotMem]
    var shared: UnsafePointer[HotPoolShared, MutAnyOrigin]
    var mailboxes: UnsafePointer[WorkerSlotMailbox, MutAnyOrigin]
    var arena_base: Int
    var capacity: Int
    var cpu_mask: CpuMask[Self.mask_size]
    var numa_node: Optional[Int]
    var futex_flags: Int
    var pinned: Bool
    var workers_alive: Bool

    def __init__(out self, capacity: Int, var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](), numa_node: Optional[Int] = None):
        self.capacity = capacity
        self.slots = HeapMoveArray[WorkerSlotMem](capacity)
        self.arena_base = 0
        self.shared = UnsafePointer[HotPoolShared, MutAnyOrigin]()
        self.mailboxes = UnsafePointer[WorkerSlotMailbox, MutAnyOrigin]()
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.workers_alive = False
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE

        var sys = linux.linux_sys()
        var mailbox_bytes = capacity * size_of[WorkerSlotMailbox]()
        var arena_size = Self.slot_size * capacity + size_of[HotPoolShared]() + mailbox_bytes
        self.arena_base = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.NORESERVE | linux.MapFlag.POPULATE
        ](0, arena_size)
        if self.arena_base < 0:
            return

        if numa_node is not None:
            var nodemask = UInt64(1) << UInt64(numa_node.value())
            if sys.sys_mbind[policy=linux.Mempolicy.BIND](self.arena_base, arena_size, nodemask) < 0:
                _ = sys.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return

        var shared_addr = self.arena_base + Self.slot_size * capacity
        self.shared = UnsafePointer[HotPoolShared, MutAnyOrigin](unsafe_from_address=shared_addr)
        self.shared[] = HotPoolShared()

        var mailbox_addr = shared_addr + size_of[HotPoolShared]()
        self.mailboxes = UnsafePointer[WorkerSlotMailbox, MutAnyOrigin](unsafe_from_address=mailbox_addr)
        for i in range(capacity):
            (self.mailboxes + i)[] = WorkerSlotMailbox()

        for i in range(capacity):
            var slot_base = self.arena_base + i * Self.slot_size
            if sys.sys_mprotect(slot_base + SlotLayout.HEADER, SlotLayout.GUARD, linux.Prot.NONE) != 0:
                _ = sys.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return
            if sys.sys_mprotect(
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + Self.stack_size,
                SlotLayout.ALT_GUARD, linux.Prot.NONE,
            ) != 0:
                _ = sys.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return
            var slot = WorkerSlotMem(slot_base)
            slot.child_tid[] = 0
            self.slots.push(slot^)

        self.spawn_workers()

    def __del__(deinit self):
        if self.arena_base == 0:
            return

        var sys = linux.linux_sys()
        if self.workers_alive:
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=self.shared[].shutdown.value), 1)
            # Wake any sleeping workers via their mailbox futex
            for i in range(self.capacity):
                var ready_ptr = UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)
                AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 1)
                _ = sys.sys_futex_wake(Int(ready_ptr), 1, self.futex_flags)

            comptime shared_futex_flags = linux.Futex2.SIZE_U32
            for i in range(self.capacity):
                while self.slots[i].is_alive():
                    _ = sys.sys_futex_wait(
                        Int(self.slots[i].child_tid),
                        Int(self.slots[i].child_tid[]),
                        shared_futex_flags)

        var mailbox_bytes = self.capacity * size_of[WorkerSlotMailbox]()
        _ = sys.sys_munmap(
            self.arena_base,
            Self.slot_size * self.capacity + size_of[HotPoolShared]() + mailbox_bytes)

    def __bool__(self) -> Bool:
        return self.arena_base != 0 and self.workers_alive

    def __len__(self) -> Int:
        return self.capacity

    @staticmethod
    def for_numa_node(numa: NumaInfo, node: Int, headroom: Int = 2) -> Self:
        """Create pool with headroom cores reserved for OS/caller."""
        var mask = numa.get_node_mask[Self.mask_size](node)
        var total = numa.cpus_on_node(node)
        var cap = max(1, total - headroom)
        var removed = 0
        var bit = Self.mask_size * 64 - 1
        while removed < headroom and bit >= 0:
            if mask.test(bit):
                mask.clear(bit)
                removed += 1
            bit -= 1
        return Self(cap, mask^, node)

    def begin_forward(mut self):
        """Enter hot mode: wake all workers, keep them spinning."""
        AtomicInt32.store[ordering=Consistency.RELEASE](
            UnsafePointer(to=self.shared[].hot_mode.value), 1)
        # Wake any workers that are futex-sleeping on their mailbox
        var sys = linux.linux_sys()
        for i in range(self.capacity):
            var ready_ptr = UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)
            _ = sys.sys_futex_wake(Int(ready_ptr), 1, self.futex_flags)

    def end_forward(mut self):
        """Exit hot mode: workers drain back to futex_wait."""
        AtomicInt32.store[ordering=Consistency.RELEASE](
            UnsafePointer(to=self.shared[].hot_mode.value), 0)

    def dispatch[F: TrivialRegisterPassable](mut self, kernel: F, packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int = -1):
        """Dispatch jobs to workers via per-worker mailboxes.

        Each of the first num_jobs workers gets one job. Remaining workers
        stay idle. No shared CAS — each worker reads only its own mailbox.
        """
        var jobs = num_jobs if num_jobs >= 0 else self.capacity
        debug_assert(jobs <= self.capacity, "num_jobs must be <= pool capacity")
        if jobs <= 0:
            return

        comptime KernelType = type_of(kernel)
        comptime assert size_of[KernelType]() == 8, "kernel must be an 8-byte function pointer"

        var done_ptr = UnsafePointer(to=self.shared[].work_done.value)
        debug_assert(
            AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr) == 0,
            "previous dispatch still in flight; call join() first",
        )

        var kernel_copy = kernel
        var kernel_ptr = UnsafePointer(to=kernel_copy).bitcast[Int]()[]

        # Set work_done before making jobs visible
        AtomicInt32.store[ordering=Consistency.MONOTONIC](done_ptr, Int32(jobs))

        # Write to each worker's mailbox — no contention, each is a separate cache line
        var sys = linux.linux_sys()
        var hot = self.shared[].hot_mode.load[ordering=Consistency.MONOTONIC]() != 0
        for i in range(jobs):
            var mb = self.mailboxes + i
            mb[].func_ptr = kernel_ptr
            mb[].pack = (packs + i)[]
            # Release store: worker sees func_ptr and pack before job_ready
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=mb[].job_ready.value), 1)
            if not hot:
                _ = sys.sys_futex_wake(Int(UnsafePointer(to=mb[].job_ready.value)), 1, self.futex_flags)

    def join(mut self):
        """Wait for all dispatched jobs to complete."""
        var done_ptr = UnsafePointer(to=self.shared[].work_done.value)
        var sys = linux.linux_sys()
        while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr) > 0:
            sys.arch_cpu_relax()

    def spawn_workers(mut self):
        var sys = linux.linux_sys()
        var parent_fs = sys.arch_thread_pointer()

        for i in range(self.capacity):
            var worker_mask = self.cpu_mask.copy() if self.pinned else CpuMask[Self.mask_size]()

            var stack_top_addr = Int(self.slots[i].stack_top) + Self.stack_size
            var stack_head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](stack_head_addr)
            var worker_main_copy = worker_main[Self.mask_size]
            var slot_base = Int(self.slots[i].base)
            var altstack_base = (
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + Self.stack_size + SlotLayout.ALT_GUARD
            )
            head[] = WorkerStackHead[Self.mask_size](
                UnsafePointer(to=worker_main_copy).bitcast[Int]()[],
                slot_base,
                parent_fs,
                i,
                self.shared,
                self.mailboxes + i,
                self.futex_flags,
                altstack_base,
                SlotLayout.ALTSTACK_SIZE,
                Int(self.pinned),
                worker_mask^,
            )
            var tcb_addr = Int(self.slots[i].base) + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i].stack_top),
                stack_head_addr - Int(self.slots[i].stack_top),
                tcb_addr,
                Int(self.slots[i].child_tid)
            )

            var result = sys.sys_clone3_with_entry(UnsafePointer(to=clone_args), size_of[linux.Clone3Args]())
            if result < 0:
                return
        self.workers_alive = True


# ============================================================================
# Worker main loop
# ============================================================================

def worker_main[mask_size: Int](stack_head_ptr: Int):
    var head_ptr = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var sys = linux.linux_sys()

    # Set up alt stack for signal handling
    var altstack_base_val = head_ptr[].altstack_base
    var altstack_size_val = head_ptr[].altstack_size
    var ss = linux.StackT()
    ss.ss_sp = altstack_base_val
    ss.ss_size = UInt64(altstack_size_val)
    ss.ss_flags = 0
    _ = sys.sys_sigaltstack(UnsafePointer(to=ss))

    var futex_flags = head_ptr[].futex_flags
    var slot_base = head_ptr[].slot_base
    var worker_id = head_ptr[].worker_id
    var shared = head_ptr[].shared
    var mailbox = head_ptr[].mailbox

    # TLS setup
    var tcb_addr = slot_base + SlotLayout.TCB
    comptime TLS_TCB_SIZE = SlotLayout.TLS_SIZE + SlotLayout.TCB_SIZE
    memcpy(
        dest=ptr[Int8](slot_base),
        src=ptr[Int8](head_ptr[].parent_fs - SlotLayout.TLS_SIZE),
        count=TLS_TCB_SIZE,
    )
    ptr[Int](tcb_addr + SlotLayout.TCB_SELF_OFFSET)[] = tcb_addr
    ptr[Int](slot_base + SlotLayout.WORKER_ID)[] = worker_id
    ptr[Int](slot_base + SlotLayout.WORKER_MAGIC)[] = SlotLayout.WORKER_MAGIC_VALUE

    # Pin to CPU
    if head_ptr[].pinned != 0:
        var ret = sys.sys_sched_setaffinity(0, mask_size, Int(head_ptr[].cpu_mask.ptr()))
        if ret != 0:
            print("sched_setaffinity failed:", ret)

    var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)
    comptime COLD_SPIN_LIMIT = 1000

    while True:
        if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
            break

        # Check own mailbox
        if AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_ptr) != 0:
            # Job available — execute it
            var func_addr = mailbox[].func_ptr
            var p = mailbox[].pack

            # Clear mailbox before executing (dispatcher can see slot is consumed)
            AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 0)

            UnsafePointer(to=func_addr).bitcast[KernelFn]()[](
                p.arg0, p.arg1, p.arg2, p.arg3, p.arg4, p.arg5,
            )

            # Signal completion
            _ = shared[].work_done.fetch_sub[ordering=Consistency.ACQUIRE_RELEASE](1)
            continue

        # No work — spin locally on own mailbox
        var hot = shared[].hot_mode.load[ordering=Consistency.MONOTONIC]() != 0
        if hot:
            # Hot mode: pure spin on own cache line
            sys.arch_cpu_relax()
        else:
            # Cold mode: brief spin then futex_wait on own mailbox
            var spins = 0
            while AtomicInt32.load[ordering=Consistency.MONOTONIC](ready_ptr) == 0:
                if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                    break
                if shared[].hot_mode.load[ordering=Consistency.MONOTONIC]() != 0:
                    break  # Switched to hot while we were spinning
                if spins < COLD_SPIN_LIMIT:
                    sys.arch_cpu_relax()
                    spins += 1
                else:
                    _ = sys.sys_futex_wait(Int(ready_ptr), 0, futex_flags)
                    spins = 0

    sys.sys_exit()
