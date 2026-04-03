"""BurstPool — per-worker mailbox dispatch with zero CAS contention.

Each worker spins on its own cache-line-aligned slot. The dispatcher
writes job info directly to each slot. No shared atomic counter,
no thundering herd, no CAS storms.

Two threading models, chosen at compile time via the Model parameter:

  ColdModel (default): workers spin briefly then futex_wait.
    Safe on any machine, any core count. Adds wake latency per dispatch.

  IsolatedModel: workers pure-spin, never sleep.
    Requires CPU isolation (isolcpus + nohz_full + rcu_nocbs).
    Zero-syscall dispatch path. Workers are never preempted.

Dispatch: O(N) sequential stores to N worker mailboxes.
Claim:    Each worker reads its own mailbox — zero cross-core traffic.
Join:     Scan per-worker done flags (no shared counter, no atomic RMW).
"""

from std.sys.info import size_of
from std.memory import UnsafePointer, memcpy
import linux.sys as linux
from std.os.atomic import Atomic, Consistency
from numa import NumaInfo, CpuMask
from notstdcollections import HeapMoveArray

comptime AtomicInt32 = Atomic[DType.int32]

# Uniform worker call ABI:
#   workers always invoke as (arg0..arg5) where each arg is a 64-bit integer-class
#   value (pointers or Ints).
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
# Per-worker mailbox
# ============================================================================

@align(64)
struct WorkerMailbox:
    """Per-worker dispatch slot.

    Cache line 1 (offset 0-63): dispatch path.
      job_ready, sleeping, func_ptr, pack args — written by dispatcher,
      read by worker.

    Cache line 2 (offset 64-127): padding.
      ArgPack tail (pad0/pad1) + explicit padding isolates dispatch
      data from the completion flag.

    Cache line 3 (offset 128+): completion path.
      done flag — written by worker, read by dispatcher during join.
      Fully isolated: no false sharing with dispatch or sleep paths.
    """
    # --- cache line 1: dispatch data ---
    var job_ready: AtomicInt32
    var sleeping: AtomicInt32  # used by ColdModel; padding for IsolatedModel
    var func_ptr: Int
    var pack: ArgPack
    # --- cache line 2: padding (isolates dispatch from completion) ---
    var line_pad0: Int
    var line_pad1: Int
    var line_pad2: Int
    var line_pad3: Int
    var line_pad4: Int
    var line_pad5: Int
    # --- cache line 3: completion flag ---
    var done: AtomicInt32

    def __init__(out self):
        self.job_ready = AtomicInt32(0)
        self.sleeping = AtomicInt32(0)
        self.func_ptr = 0
        self.pack = ArgPack()
        self.line_pad0 = 0
        self.line_pad1 = 0
        self.line_pad2 = 0
        self.line_pad3 = 0
        self.line_pad4 = 0
        self.line_pad5 = 0
        self.done = AtomicInt32(0)


# ============================================================================
# Shared state — minimal
# ============================================================================

@align(64)
struct SharedPoolState:
    var shutdown: AtomicInt32
    var parked: AtomicInt32  # 0=spinning, 1=parked. Doubles as futex guard value.

    def __init__(out self):
        self.shutdown = AtomicInt32(0)
        self.parked = AtomicInt32(0)


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

def slot_size[stack_size: Int]() -> Int:
    comptime assert stack_size >= SlotLayout.GUARD and stack_size % SlotLayout.GUARD == 0, "stack_size must be a multiple of 4096 (>= 4096)"
    var raw = SlotLayout.HEADER + SlotLayout.GUARD + stack_size + SlotLayout.ALT_GUARD + SlotLayout.ALTSTACK_SIZE
    return ((raw + SlotLayout.GUARD - 1) // SlotLayout.GUARD) * SlotLayout.GUARD

@always_inline
def current_worker_id() -> Int:
    """Return worker id when running in a BurstPool worker, else -1."""
    var sys = linux.linux_sys()
    var magic = sys.arch_tls_load_i64[offset=SlotLayout.WORKER_MAGIC_FROM_FS]()
    if magic != SlotLayout.WORKER_MAGIC_VALUE:
        return -1
    return sys.arch_tls_load_i64[offset=SlotLayout.WORKER_ID_FROM_FS]()

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


struct WorkerSlot(Movable, ImplicitlyDestructible):
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
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]
    var mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var futex_flags: Int
    var altstack_base: Int
    var altstack_size: Int
    var pinned: Int
    var cpu_mask: CpuMask[Self.mask_size]

    def __init__(out self, entry: Int, slot_base: Int, parent_fs: Int,
                worker_id: Int,
                shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
                mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin],
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
# ThreadingModel trait
# ============================================================================

trait ThreadingModel(ImplicitlyDestructible):
    """Defines how workers wait for work and how the dispatcher wakes them.

    ColdModel:     spin briefly → Dekker → futex_wait. Safe everywhere.
    IsolatedModel: pure PAUSE spin. Requires CPU isolation.
    """
    comptime PER_CORE_PIN: Bool

    def __init__(out self): ...

    def worker_loop(self,
        mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        futex_flags: Int,
    ): ...

    def on_dispatch(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        jobs: Int,
        futex_flags: Int,
    ): ...

    def on_shutdown(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ): ...

    def wake(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ): ...

    def sleep(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
    ): ...


# ============================================================================
# ColdModel — safe on any machine
# ============================================================================

struct ColdModel(ThreadingModel):
    """Workers spin for COLD_SPIN_LIMIT iterations, then Dekker-sleep via futex.

    Dispatch wakes sleeping workers with per-worker futex_wake (Dekker protocol
    ensures no missed wakes). Safe on any core count, any machine.
    """
    comptime PER_CORE_PIN = False
    comptime COLD_SPIN_LIMIT = 1000

    def __init__(out self):
        pass

    def worker_loop(self,
        mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        futex_flags: Int,
    ):
        var sys = linux.linux_sys()
        var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)

        while True:
            if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
                break

            if AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_ptr) != 0:
                var func_addr = mailbox[].func_ptr
                var p = mailbox[].pack
                AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 0)
                UnsafePointer(to=func_addr).bitcast[KernelFn]()[](
                    p.arg0, p.arg1, p.arg2, p.arg3, p.arg4, p.arg5,
                )
                AtomicInt32.store[ordering=Consistency.RELEASE](
                    UnsafePointer(to=mailbox[].done.value), 1)
                continue

            var spins = 0
            while AtomicInt32.load[ordering=Consistency.MONOTONIC](ready_ptr) == 0:
                if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                    break
                if spins < Self.COLD_SPIN_LIMIT:
                    sys.arch_cpu_relax()
                    spins += 1
                else:
                    # Dekker sleep: publish sleeping=1, recheck ready_ptr.
                    # Dispatcher publishes ready=1 then checks sleeping.
                    # TSO guarantees at least one side sees the other's store.
                    var sleeping_ptr = UnsafePointer(to=mailbox[].sleeping.value)
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

    def on_dispatch(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        jobs: Int,
        futex_flags: Int,
    ):
        """Dekker wake: check sleeping flag, futex_wake if needed."""
        var sys = linux.linux_sys()
        for i in range(jobs):
            var mb = mailboxes + i
            if AtomicInt32.load[ordering=Consistency.ACQUIRE](
                UnsafePointer(to=mb[].sleeping.value)
            ) != 0:
                _ = sys.sys_futex_wake(
                    Int(UnsafePointer(to=mb[].job_ready.value)), 1, futex_flags)

    def on_shutdown(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ):
        """Wake all sleeping workers so they see the shutdown flag."""
        var sys = linux.linux_sys()
        for i in range(capacity):
            var ready_ptr = UnsafePointer(to=(mailboxes + i)[].job_ready.value)
            AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 1)
            _ = sys.sys_futex_wake(Int(ready_ptr), 1, futex_flags)

    def wake(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ):
        """No-op: ColdModel workers manage their own sleep/wake."""
        pass

    def sleep(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
    ):
        """No-op: ColdModel workers manage their own sleep/wake."""
        pass


# ============================================================================
# IsolatedModel — zero-syscall dispatch on isolated cores
# ============================================================================

struct IsolatedModel(ThreadingModel):
    """Workers pure-spin on job_ready. No sleeping, no futex, no Dekker.

    Requires CPU isolation: isolcpus + nohz_full + rcu_nocbs on worker cores.
    Workers are never preempted, so spinning is safe and dispatch is
    zero-syscall (just RELEASE stores to mailboxes).

    Workers burn CPU while awake. Use wake()/sleep() to control when
    isolated cores are active. Between sleep() and wake(), workers park
    via futex_wait and consume no CPU.
    """
    comptime PER_CORE_PIN = True

    def __init__(out self):
        pass

    def worker_loop(self,
        mailbox: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        futex_flags: Int,
    ):
        var sys = linux.linux_sys()
        var ready_ptr = UnsafePointer(to=mailbox[].job_ready.value)

        while True:
            if AtomicInt32.load[ordering=Consistency.ACQUIRE](ready_ptr) != 0:
                var func_addr = mailbox[].func_ptr
                var p = mailbox[].pack
                AtomicInt32.store[ordering=Consistency.RELEASE](ready_ptr, 0)
                UnsafePointer(to=func_addr).bitcast[KernelFn]()[](
                    p.arg0, p.arg1, p.arg2, p.arg3, p.arg4, p.arg5,
                )
                AtomicInt32.store[ordering=Consistency.RELEASE](
                    UnsafePointer(to=mailbox[].done.value), 1)
            else:
                if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                    break
                if shared[].parked.load[ordering=Consistency.ACQUIRE]() != 0:
                    # Park via futex_wait on parked with expected=1.
                    # If wake() already set parked=0, futex_wait sees 0!=1 → EAGAIN.
                    var parked_ptr = UnsafePointer(to=shared[].parked.value)
                    _ = sys.sys_futex_wait(Int(parked_ptr), 1, futex_flags)
                else:
                    sys.arch_cpu_relax()

    def on_dispatch(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        jobs: Int,
        futex_flags: Int,
    ):
        """No-op: workers are spinning on isolated cores."""
        pass

    def on_shutdown(self,
        mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin],
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ):
        """Unpark workers so they see the shutdown flag."""
        var parked_ptr = UnsafePointer(to=shared[].parked.value)
        AtomicInt32.store[ordering=Consistency.RELEASE](parked_ptr, 0)
        var sys = linux.linux_sys()
        _ = sys.sys_futex_wake(Int(parked_ptr), capacity, futex_flags)

    def wake(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
        capacity: Int,
        futex_flags: Int,
    ):
        """Unpark workers. Sets parked=0, futex_wake unblocks sleepers.
        Late arrivals see parked=0!=1 via futex expected-value check → EAGAIN."""
        var parked_ptr = UnsafePointer(to=shared[].parked.value)
        AtomicInt32.store[ordering=Consistency.RELEASE](parked_ptr, 0)
        var sys = linux.linux_sys()
        _ = sys.sys_futex_wake(Int(parked_ptr), capacity, futex_flags)

    def sleep(self,
        shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
    ):
        """Park workers. They see parked=1 and futex_wait(parked, 1) blocks."""
        AtomicInt32.store[ordering=Consistency.RELEASE](
            UnsafePointer(to=shared[].parked.value), 1)


# ============================================================================
# BurstPool[Model]
# ============================================================================

struct BurstPool[Model: ThreadingModel = ColdModel, stack_size: Int = SlotLayout.DEFAULT_STACK, mask_size: Int = 128](Movable):
    comptime slot_size = slot_size[Self.stack_size]()
    var slots: HeapMoveArray[WorkerSlot]
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]
    var mailboxes: UnsafePointer[WorkerMailbox, MutAnyOrigin]
    var args_base: UnsafePointer[ArgPack, MutAnyOrigin]
    var arena_base: Int
    var capacity: Int
    var active_jobs: Int
    var cpu_mask: CpuMask[Self.mask_size]
    var numa_node: Optional[Int]
    var futex_flags: Int
    var pinned: Bool
    var workers_alive: Bool

    def __init__(out self, capacity: Int, var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](), numa_node: Optional[Int] = None):
        self.capacity = capacity
        self.active_jobs = 0
        self.slots = HeapMoveArray[WorkerSlot](capacity)
        self.arena_base = 0
        self.shared = UnsafePointer[SharedPoolState, MutAnyOrigin]()
        self.mailboxes = UnsafePointer[WorkerMailbox, MutAnyOrigin]()
        self.args_base = UnsafePointer[ArgPack, MutAnyOrigin]()
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.workers_alive = False
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE

        install_burst_sigsegv_handler()

        var sys = linux.linux_sys()
        var mailbox_bytes = capacity * size_of[WorkerMailbox]()
        var args_bytes = capacity * size_of[ArgPack]()
        var arena_size = Self.slot_size * capacity + size_of[SharedPoolState]() + mailbox_bytes + args_bytes
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
        self.shared = UnsafePointer[SharedPoolState, MutAnyOrigin](unsafe_from_address=shared_addr)
        self.shared[] = SharedPoolState()

        var mailbox_addr = shared_addr + size_of[SharedPoolState]()
        self.mailboxes = UnsafePointer[WorkerMailbox, MutAnyOrigin](unsafe_from_address=mailbox_addr)
        for i in range(capacity):
            (self.mailboxes + i)[] = WorkerMailbox()

        self.args_base = UnsafePointer[ArgPack, MutAnyOrigin](
            unsafe_from_address=mailbox_addr + mailbox_bytes)

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
            var slot = WorkerSlot(slot_base)
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
            var model = Self.Model()
            model.on_shutdown(self.mailboxes, self.shared, self.capacity, self.futex_flags)

            # Spin-wait for workers to exit.  Workers are already shutting
            # down so this completes in microseconds.  We avoid futex_wait
            # here because (a) a TOCTOU race between is_alive() and the
            # expected-value read can cause an infinite sleep, and (b) the
            # kernel's CHILD_CLEARTID uses old-style futex(FUTEX_WAKE)
            # which may not match our futex2 wait queue.
            for i in range(self.capacity):
                while self.slots[i].is_alive():
                    sys.arch_cpu_relax()

        var mailbox_bytes = self.capacity * size_of[WorkerMailbox]()
        var args_bytes = self.capacity * size_of[ArgPack]()
        _ = sys.sys_munmap(
            self.arena_base,
            Self.slot_size * self.capacity + size_of[SharedPoolState]() + mailbox_bytes + args_bytes)

    def __bool__(self) -> Bool:
        return self.arena_base != 0 and self.workers_alive

    def __len__(self) -> Int:
        return self.capacity

    def wake(mut self):
        """Unpark workers. IsolatedModel: resumes spinning. ColdModel: no-op."""
        var model = Self.Model()
        model.wake(self.shared, self.capacity, self.futex_flags)

    def sleep(mut self):
        """Park workers. IsolatedModel: futex_wait, zero CPU burn. ColdModel: no-op."""
        var model = Self.Model()
        model.sleep(self.shared)

    @staticmethod
    def for_numa_node(numa: NumaInfo, node: Int, headroom: Int = 0) -> Self:
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
        return Self(cap, mask^, node)

    @staticmethod
    def for_topology(numa: NumaInfo, node: Int) -> Self:
        """Create pool from topology discovery. Uses isolated cores on the node
        if isolation is configured, otherwise all cores on the node."""
        var mask = numa.get_worker_mask[Self.mask_size](node)
        var cap = mask.count()
        if cap == 0:
            cap = 1
            mask = numa.get_node_mask[Self.mask_size](node)
        return Self(cap, mask^, node)

    @staticmethod
    def for_numa_node_excluding(numa: NumaInfo, node: Int, exclude_cpu: Int) -> Self:
        var mask = numa.get_node_mask[Self.mask_size](node)
        var cap = numa.cpus_on_node(node)
        if mask.test(exclude_cpu):
            mask.clear(exclude_cpu)
            cap -= 1
        return Self(cap, mask^, node)

    def dispatch[F: TrivialRegisterPassable](mut self, kernel: F, packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int = -1):
        """Dispatch jobs to workers via per-worker mailboxes.

        Each of the first num_jobs workers gets one job. Remaining workers
        stay idle. Packs can be args_base or any ArgPack array.
        """
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

        for i in range(jobs):
            var mb = self.mailboxes + i
            mb[].func_ptr = kernel_ptr
            mb[].pack = (packs + i)[]
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=mb[].job_ready.value), 1)

        # Model-specific wake strategy (after all mailboxes written)
        var model = Self.Model()
        model.on_dispatch(self.mailboxes, jobs, self.futex_flags)

    def join(mut self):
        """Wait for all dispatched jobs to complete."""
        var sys = linux.linux_sys()
        for i in range(self.active_jobs):
            var done_ptr = UnsafePointer(to=(self.mailboxes + i)[].done.value)
            while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr) == 0:
                sys.arch_cpu_relax()
            # Reset for next round — cache line is already loaded from the read
            AtomicInt32.store[ordering=Consistency.MONOTONIC](done_ptr, 0)
        self.active_jobs = 0

    def spawn_workers(mut self):
        var sys = linux.linux_sys()
        var parent_fs = sys.arch_thread_pointer()

        for i in range(self.capacity):
            var worker_mask = CpuMask[Self.mask_size]()
            if self.pinned:
                comptime
                if Self.Model.PER_CORE_PIN:
                    # Per-worker pinning: worker i → the i-th set bit.
                    # Required on isolated cores where the load balancer is disabled.
                    var bit_count = 0
                    for bit in range(Self.mask_size * 8):
                        if self.cpu_mask.test(bit):
                            if bit_count == i:
                                worker_mask.set(bit)
                                break
                            bit_count += 1
                else:
                    # Shared mask: let the kernel load balancer spread workers.
                    worker_mask = self.cpu_mask.copy()

            var stack_top_addr = Int(self.slots[i].stack_top) + Self.stack_size
            var stack_head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](stack_head_addr)
            var worker_main_copy = worker_main[Self.Model, Self.mask_size]
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
                # Some workers may already be alive — shrink capacity so
                # __del__ shuts down only the ones that were spawned.
                self.capacity = i
                if i > 0:
                    self.workers_alive = True
                return
        self.workers_alive = True


def make_node_pools[Model: ThreadingModel = ColdModel, stack_size: Int = SlotLayout.DEFAULT_STACK, mask_size: Int = 128](
    numa: NumaInfo,
) -> HeapMoveArray[BurstPool[Model, stack_size, mask_size]]:
    """Create one BurstPool per NUMA node, sized from isolation topology."""
    var pools = HeapMoveArray[BurstPool[Model, stack_size, mask_size]](numa.num_nodes)
    for i in range(numa.num_nodes):
        pools.push(BurstPool[Model, stack_size, mask_size].for_topology(numa, numa.nodes[i].id))
    return pools^


# ============================================================================
# Worker entry point — parameterized on Model
# ============================================================================

def worker_main[Model: ThreadingModel, mask_size: Int](stack_head_ptr: Int):
    var head_ptr = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var sys = linux.linux_sys()

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

    if head_ptr[].pinned != 0:
        var ret = sys.sys_sched_setaffinity(0, mask_size * 8, Int(head_ptr[].cpu_mask.ptr()))
        if ret != 0:
            print("sched_setaffinity failed:", ret)

    # Delegate to model-specific loop
    var model = Model()
    model.worker_loop(mailbox, shared, futex_flags)

    # CHILD_CLEARTID handles clearing child_tid and futex wake automatically
    sys.sys_exit()
