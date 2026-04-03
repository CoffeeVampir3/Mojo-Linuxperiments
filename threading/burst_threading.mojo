"""BurstPool — spin-backoff pool with dual-mailbox NUMA-aware dispatch.

Workers spin briefly on local mailboxes, then Dekker-sleep via futex.
Safe on any machine, any core count.

Memory layout — dual arena for NUMA-optimal access:

  Worker arena (mbind to worker's NUMA node):
    Thread stacks + WorkerMailboxes + SharedState
    Workers read mailboxes locally. Dispatcher writes remotely.

  Join arena (first-touch on main thread's NUMA node):
    JoinFlags + ArgPacks
    Main thread polls JoinFlags locally. Workers write done remotely once.

Dispatch: two-pass stores to N worker mailboxes + Dekker wake check.
Join:     poll local JoinFlags (zero cross-NUMA reads).
"""

from std.sys.info import size_of
from std.memory import UnsafePointer, memcpy
from std.time import perf_counter_ns
import linux.sys as linux
from std.os.atomic import Atomic, Consistency
from numa import NumaInfo, CpuMask
from notstdcollections import HeapMoveArray
from .isolated_burst_pool import ArgPack
from .threading_traits import BurstThreadPool

comptime AtomicInt32 = Atomic[DType.int32]
comptime KernelFn = def (Int, Int, Int, Int, Int, Int) -> None


# ============================================================================
# Per-worker mailbox — lives on the worker's NUMA node
# ============================================================================

@align(64)
struct WorkerMailbox:
    """Dispatch slot on the worker's NUMA node. Exactly one cache line.

    Worker reads locally. Dispatcher writes remotely.
    sleeping flag is for Dekker backoff protocol.
    """
    var job_ready: AtomicInt32  # 0=idle, 1=work available
    var sleeping: AtomicInt32   # Dekker: worker sets before futex_wait
    var func_ptr: Int
    var pack: ArgPack

    def __init__(out self):
        self.job_ready = AtomicInt32(0)
        self.sleeping = AtomicInt32(0)
        self.func_ptr = 0
        self.pack = ArgPack()


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
# Shared state — lives on worker's NUMA node
# ============================================================================

@align(64)
struct SharedState:
    var shutdown: AtomicInt32

    def __init__(out self):
        self.shutdown = AtomicInt32(0)


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
    comptime DEFAULT_STACK = 64 * 1024


def compute_slot_size(stack_size: Int) -> Int:
    debug_assert(stack_size >= SlotLayout.GUARD and stack_size % SlotLayout.GUARD == 0,
        "stack_size must be a multiple of 4096 (>= 4096)")
    var raw = (SlotLayout.HEADER + SlotLayout.GUARD + stack_size
             + SlotLayout.ALT_GUARD + SlotLayout.ALTSTACK_SIZE)
    return ((raw + SlotLayout.GUARD - 1) // SlotLayout.GUARD) * SlotLayout.GUARD


@always_inline
def ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)


# ============================================================================
# Worker ID from TLS
# ============================================================================

@always_inline
def current_worker_id() -> Int:
    """Return worker id when running in a BurstPool worker, else -1."""
    var sys = linux.linux_sys()
    var magic = sys.arch_tls_load_i64[offset=SlotLayout.WORKER_MAGIC_FROM_FS]()
    if magic != SlotLayout.WORKER_MAGIC_VALUE:
        return -1
    return sys.arch_tls_load_i64[offset=SlotLayout.WORKER_ID_FROM_FS]()


# ============================================================================
# SIGSEGV handler
# ============================================================================

def burst_sigsegv_handler(signo: Int32, info: Int, ucontext: Int):
    var sys = linux.linux_sys()
    var ctx = sys.arch_decode_sigsegv(info, ucontext)
    var worker = current_worker_id()
    var pid = sys.sys_getpid()
    var tid = sys.sys_gettid()

    print(
        "burst: SIGSEGV worker=", worker,
        "pid=", pid,
        "tid=", tid,
        "rip=", hex(ctx.ip),
        "rsp=", hex(ctx.sp),
        "addr=", hex(ctx.fault_addr),
    )

    _ = sys.sys_tgkill(pid, tid, linux.Signal.SEGV)
    sys.sys_exit_group(128 + Int(signo))

def install_burst_sigsegv_handler():
    var sys = linux.linux_sys()
    var handler_copy = burst_sigsegv_handler
    var handler_addr = UnsafePointer(to=handler_copy).bitcast[Int]()[]

    var act = linux.RtSigAction()
    act.handler = handler_addr
    act.flags = UInt64(linux.SigActionFlag.SIGINFO | linux.SigActionFlag.ONSTACK)
    act.mask = linux.SigSet64()

    _ = sys.sys_rt_sigaction(linux.Signal.SEGV, UnsafePointer(to=act))


# ============================================================================
# Worker slot — tracks one worker's memory region
# ============================================================================

struct WorkerSlot(Movable, ImplicitlyDestructible):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var child_tid: UnsafePointer[Int32, MutAnyOrigin]
    var stack_top: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.stack_top = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)

    @always_inline
    def is_alive(self) -> Bool:
        return self.child_tid[] != 0


# ============================================================================
# Worker stack head — passed to worker via stack pointer
# ============================================================================

@fieldwise_init
struct WorkerStackHead[mask_size: Int]:
    var entry: Int
    var slot_base: Int
    var parent_fs: Int
    var worker_id: Int
    var mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var join_flag: UnsafePointer[JoinFlag, MutAnyOrigin]
    var shared: UnsafePointer[SharedState, MutAnyOrigin]
    var futex_flags: Int
    var altstack_base: Int
    var altstack_size: Int
    var pinned: Int
    var cpu_mask: CpuMask[Self.mask_size]


# ============================================================================
# BurstPool
# ============================================================================

comptime SPIN_LIMIT = 1000

struct BurstPool[mask_size: Int = 128](BurstThreadPool):
    """Spin-backoff pool with dual-mailbox NUMA-aware dispatch.

    Workers spin on local mailboxes, then Dekker-sleep via futex.
    Join polls local JoinFlags. Zero cross-NUMA reads on the join path.
    """
    # Worker-side (on worker's NUMA node)
    var slots: HeapMoveArray[WorkerSlot]
    var mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var shared: UnsafePointer[SharedState, MutAnyOrigin]
    var worker_arena: Int
    var worker_arena_size: Int

    # Main-thread-side (on main's NUMA node)
    var join_flags: UnsafePointer[JoinFlag, MutAnyOrigin]
    var args_base: UnsafePointer[ArgPack, MutAnyOrigin]
    var join_arena: Int
    var join_arena_size: Int

    # State
    var capacity: Int
    var active_jobs: Int
    var stack_size: Int
    var slot_size: Int
    var cpu_mask: CpuMask[Self.mask_size]
    var numa_node: Optional[Int]
    var workers_alive: Bool
    var futex_flags: Int
    var pinned: Bool

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def __init__(out self, capacity: Int,
                 var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](),
                 numa_node: Optional[Int] = None,
                 stack_size: Int = SlotLayout.DEFAULT_STACK):
        self.capacity = capacity
        self.active_jobs = 0
        self.stack_size = stack_size
        self.slot_size = compute_slot_size(stack_size)
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.workers_alive = False
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE
        self.slots = HeapMoveArray[WorkerSlot](capacity)
        self.worker_arena = 0
        self.worker_arena_size = 0
        self.join_arena = 0
        self.join_arena_size = 0
        self.mailboxes = UnsafePointer[WorkerMailbox, MutAnyOrigin]()
        self.shared = UnsafePointer[SharedState, MutAnyOrigin]()
        self.join_flags = UnsafePointer[JoinFlag, MutAnyOrigin]()
        self.args_base = UnsafePointer[ArgPack, MutAnyOrigin]()

        install_burst_sigsegv_handler()

        var sys = linux.linux_sys()

        # --- Worker arena: stacks + shared + mailboxes (on worker's NUMA node) ---
        var mailbox_bytes = capacity * size_of[WorkerMailbox]()
        self.worker_arena_size = (self.slot_size * capacity
            + size_of[SharedState]() + mailbox_bytes)
        self.worker_arena = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS
                | linux.MapFlag.NORESERVE | linux.MapFlag.POPULATE
        ](0, self.worker_arena_size)
        if self.worker_arena < 0:
            self.worker_arena = 0
            return

        if numa_node is not None:
            var nodemask = UInt64(1) << UInt64(numa_node.value())
            if sys.sys_mbind[policy=linux.Mempolicy.BIND](
                self.worker_arena, self.worker_arena_size, nodemask
            ) < 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return

        var shared_addr = self.worker_arena + self.slot_size * capacity
        self.shared = ptr[SharedState](shared_addr)
        self.shared[] = SharedState()

        self.mailboxes = ptr[WorkerMailbox](shared_addr + size_of[SharedState]())
        for i in range(capacity):
            (self.mailboxes + i)[] = WorkerMailbox()

        # Guard pages per slot
        for i in range(capacity):
            var slot_base = self.worker_arena + i * self.slot_size
            if sys.sys_mprotect(slot_base + SlotLayout.HEADER,
                                SlotLayout.GUARD, linux.Prot.NONE) != 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return
            if sys.sys_mprotect(
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + self.stack_size,
                SlotLayout.ALT_GUARD, linux.Prot.NONE,
            ) != 0:
                _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
                self.worker_arena = 0
                return
            var slot = WorkerSlot(slot_base)
            slot.child_tid[] = 0
            self.slots.push(slot^)

        # --- Join arena: join flags + argpacks (on main thread's NUMA node) ---
        var args_bytes = capacity * size_of[ArgPack]()
        self.join_arena_size = capacity * size_of[JoinFlag]() + args_bytes
        self.join_arena = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.POPULATE
        ](0, self.join_arena_size)
        if self.join_arena < 0:
            _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
            self.worker_arena = 0
            self.join_arena = 0
            return

        # No mbind — stays on main thread's node by first-touch
        self.join_flags = ptr[JoinFlag](self.join_arena)
        for i in range(capacity):
            (self.join_flags + i)[] = JoinFlag()

        self.args_base = ptr[ArgPack](self.join_arena + capacity * size_of[JoinFlag]())

        self.spawn_workers()

    # ----------------------------------------------------------------
    # Bool / len
    # ----------------------------------------------------------------

    def __bool__(self) -> Bool:
        return self.worker_arena != 0 and self.join_arena != 0

    def __len__(self) -> Int:
        return self.capacity

    # ----------------------------------------------------------------
    # Dispatch — two-pass write + Dekker wake check
    # ----------------------------------------------------------------

    def dispatch[F: TrivialRegisterPassable](mut self, kernel: F,
        packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int = -1):
        var jobs = num_jobs if num_jobs >= 0 else self.capacity
        debug_assert(jobs <= self.capacity, "num_jobs must be <= pool capacity")
        if jobs <= 0:
            return

        comptime KernelType = type_of(kernel)
        comptime assert size_of[KernelType]() == 8, "kernel must be an 8-byte function pointer"

        debug_assert(self.active_jobs == 0,
            "previous dispatch still in flight; call join() first")

        var kernel_copy = kernel
        var kernel_ptr = UnsafePointer(to=kernel_copy).bitcast[Int]()[]

        self.active_jobs = jobs

        # Pass 1: write dispatch data to worker mailboxes (remote to worker's node)
        for i in range(jobs):
            var mb = self.mailboxes + i
            mb[].func_ptr = kernel_ptr
            mb[].pack = (packs + i)[]

        # Pass 2: set job_ready flags (remote writes, RELEASE ordering)
        for i in range(jobs):
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=(self.mailboxes + i)[].job_ready.value), 1)

        # Pass 3: Dekker wake — check sleeping, futex_wake if needed
        # If worker stored sleeping=1 but missed our job_ready store (x86 store-load
        # reordering), the futex_wait's atomic check catches it (sees job_ready=1,
        # returns EAGAIN). So this is an optimization, not a correctness requirement.
        var sys = linux.linux_sys()
        for i in range(jobs):
            if AtomicInt32.load[ordering=Consistency.ACQUIRE](
                UnsafePointer(to=(self.mailboxes + i)[].sleeping.value)
            ) != 0:
                _ = sys.sys_futex_wake(
                    Int(UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)),
                    1, self.futex_flags)

    # ----------------------------------------------------------------
    # Join — poll local JoinFlags
    # ----------------------------------------------------------------

    def join(mut self):
        """Wait for all dispatched jobs. Polls JoinFlags on main's NUMA node."""
        var sys = linux.linux_sys()
        for i in range(self.active_jobs):
            var done_ptr = UnsafePointer(to=(self.join_flags + i)[].done.value)
            while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr) == 0:
                sys.arch_cpu_relax()
            AtomicInt32.store[ordering=Consistency.MONOTONIC](done_ptr, 0)
        self.active_jobs = 0

    def get_capacity(self) -> Int:
        return self.capacity

    def get_args_base(self) -> UnsafePointer[ArgPack, MutAnyOrigin]:
        return self.args_base

    def last_worker_timestamp(self) -> Int:
        """Max completion timestamp across workers from the last dispatch.
        Call after join(). Workers write perf_counter_ns() before setting done."""
        var max_ts = 0
        for i in range(self.capacity):
            var ts = (self.join_flags + i)[].timestamp
            if ts > max_ts:
                max_ts = ts
        return max_ts

    # ----------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------

    def __del__(deinit self):
        if self.worker_arena == 0:
            return

        var sys = linux.linux_sys()
        if self.workers_alive:
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=self.shared[].shutdown.value), 1)
            # Wake all sleeping workers so they see the shutdown flag
            for i in range(self.capacity):
                var ready_ptr = UnsafePointer(to=(self.mailboxes + i)[].job_ready.value)
                AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 1)
                _ = sys.sys_futex_wake(Int(ready_ptr), 1, self.futex_flags)
            for i in range(self.capacity):
                while self.slots[i].is_alive():
                    sys.arch_cpu_relax()

        if self.worker_arena != 0:
            _ = sys.sys_munmap(self.worker_arena, self.worker_arena_size)
        if self.join_arena != 0:
            _ = sys.sys_munmap(self.join_arena, self.join_arena_size)

    # ----------------------------------------------------------------
    # Spawn workers
    # ----------------------------------------------------------------

    def spawn_workers(mut self):
        var sys = linux.linux_sys()
        var parent_fs = sys.arch_thread_pointer()

        for i in range(self.capacity):
            var worker_mask = CpuMask[Self.mask_size]()
            if self.pinned:
                worker_mask = self.cpu_mask.copy()

            var stack_top_addr = Int(self.slots[i].stack_top) + self.stack_size
            var head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](head_addr)
            var entry_fn = worker_main[Self.mask_size]
            var slot_base = Int(self.slots[i].base)
            var altstack_base = (
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD
                + self.stack_size + SlotLayout.ALT_GUARD
            )
            head[] = WorkerStackHead[Self.mask_size](
                UnsafePointer(to=entry_fn).bitcast[Int]()[],
                slot_base,
                parent_fs,
                i,
                self.mailboxes + i,
                self.join_flags + i,
                self.shared,
                self.futex_flags,
                altstack_base,
                SlotLayout.ALTSTACK_SIZE,
                Int(self.pinned),
                worker_mask^,
            )
            var tcb_addr = slot_base + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i].stack_top),
                head_addr - Int(self.slots[i].stack_top),
                tcb_addr,
                Int(self.slots[i].child_tid),
            )
            var result = sys.sys_clone3_with_entry(
                UnsafePointer(to=clone_args), size_of[linux.Clone3Args]())
            if result < 0:
                self.capacity = i
                if i > 0:
                    self.workers_alive = True
                return
        self.workers_alive = True

    # ----------------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------------

    @staticmethod
    def for_numa_node(numa: NumaInfo, node: Int, headroom: Int = 0,
                      stack_size: Int = SlotLayout.DEFAULT_STACK) -> Self:
        """Create pool for a NUMA node. headroom reserves cores for OS/caller."""
        var mask = numa.get_node_mask[Self.mask_size](node)
        var total = numa.cpus_on_node(node)
        var cap = max(1, total - headroom)
        if headroom > 0:
            var removed = 0
            var bit = Self.mask_size * 64 - 1
            while removed < headroom and bit >= 0:
                if mask.test(bit):
                    mask.clear(bit)
                    removed += 1
                bit -= 1
        return Self(cap, mask^, node, stack_size)

    @staticmethod
    def for_topology(numa: NumaInfo, node: Int,
                     stack_size: Int = SlotLayout.DEFAULT_STACK) -> Self:
        """Create pool from topology discovery. Uses isolated cores on the node
        if isolation is configured, otherwise all cores on the node."""
        var mask = numa.get_worker_mask[Self.mask_size](node)
        var cap = mask.count()
        if cap == 0:
            cap = 1
            mask = numa.get_node_mask[Self.mask_size](node)
        return Self(cap, mask^, node, stack_size)

    @staticmethod
    def for_numa_node_excluding(numa: NumaInfo, node: Int, exclude_cpu: Int,
                                stack_size: Int = SlotLayout.DEFAULT_STACK) -> Self:
        var mask = numa.get_node_mask[Self.mask_size](node)
        var cap = numa.cpus_on_node(node)
        if mask.test(exclude_cpu):
            mask.clear(exclude_cpu)
            cap -= 1
        return Self(cap, mask^, node, stack_size)


def make_node_pools[mask_size: Int = 128](
    numa: NumaInfo,
    stack_size: Int = SlotLayout.DEFAULT_STACK,
) -> HeapMoveArray[BurstPool[mask_size]]:
    """Create one BurstPool per NUMA node, sized from isolation topology."""
    var pools = HeapMoveArray[BurstPool[mask_size]](numa.num_nodes)
    for i in range(numa.num_nodes):
        pools.push(BurstPool[mask_size].for_topology(numa, numa.nodes[i].id, stack_size))
    return pools^


# ============================================================================
# Worker entry point
# ============================================================================

def worker_main[mask_size: Int](stack_head_ptr: Int):
    var head = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var sys = linux.linux_sys()

    # Signal stack for SIGSEGV handler
    var ss = linux.StackT()
    ss.ss_sp = head[].altstack_base
    ss.ss_size = UInt64(head[].altstack_size)
    ss.ss_flags = 0
    _ = sys.sys_sigaltstack(UnsafePointer(to=ss))

    # TLS setup
    var slot_base = head[].slot_base
    var tcb_addr = slot_base + SlotLayout.TCB
    comptime TLS_TCB_SIZE = SlotLayout.TLS_SIZE + SlotLayout.TCB_SIZE
    memcpy(
        dest=ptr[Int8](slot_base),
        src=ptr[Int8](head[].parent_fs - SlotLayout.TLS_SIZE),
        count=TLS_TCB_SIZE,
    )
    ptr[Int](tcb_addr + SlotLayout.TCB_SELF_OFFSET)[] = tcb_addr
    ptr[Int](slot_base + SlotLayout.WORKER_ID)[] = head[].worker_id
    ptr[Int](slot_base + SlotLayout.WORKER_MAGIC)[] = SlotLayout.WORKER_MAGIC_VALUE

    # Pin to assigned cores
    if head[].pinned != 0:
        _ = sys.sys_sched_setaffinity(0, mask_size * 8, Int(head[].cpu_mask.ptr()))

    var mailbox = head[].mailbox
    var join_flag = head[].join_flag
    var shared = head[].shared
    var futex_flags = head[].futex_flags
    var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)
    var sleeping_ptr = UnsafePointer(to=mailbox[].sleeping.value)
    var done_ptr = UnsafePointer(to=join_flag[].done.value)

    # --- Main loop: spin on local mailbox, backoff to futex_wait ---
    while True:
        if AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_ptr) != 0:
            # Read dispatch data (local reads from worker's NUMA node)
            var func_addr = mailbox[].func_ptr
            var p = mailbox[].pack
            # Clear job_ready (local write)
            AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 0)
            # Execute kernel
            UnsafePointer(to=func_addr).bitcast[KernelFn]()[](
                p.arg0, p.arg1, p.arg2, p.arg3, p.arg4, p.arg5,
            )
            # Signal completion (remote writes to main's NUMA node)
            join_flag[].timestamp = Int(perf_counter_ns())
            AtomicInt32.store[ordering=Consistency.RELEASE](done_ptr, 1)
            continue

        if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
            break

        # Spin phase — brief spin on local job_ready
        var spins = 0
        while AtomicInt32.load[ordering=Consistency.MONOTONIC](ready_ptr) == 0:
            if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                break
            if spins < SPIN_LIMIT:
                sys.arch_cpu_relax()
                spins += 1
            else:
                # Dekker sleep: publish sleeping=1, recheck job_ready.
                # Dispatcher publishes job_ready=1 then checks sleeping.
                # If both miss (x86 store-load reordering), futex_wait's
                # atomic check sees job_ready=1 and returns EAGAIN.
                AtomicInt32.store[ordering=Consistency.RELEASE](sleeping_ptr, 1)
                if AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_ptr) != 0:
                    AtomicInt32.store[ordering=Consistency.RELEASE](sleeping_ptr, 0)
                    break
                if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
                    AtomicInt32.store[ordering=Consistency.RELEASE](sleeping_ptr, 0)
                    break
                _ = sys.sys_futex_wait(Int(ready_ptr), 0, futex_flags)
                AtomicInt32.store[ordering=Consistency.RELEASE](sleeping_ptr, 0)
                spins = 0

    sys.sys_exit()
