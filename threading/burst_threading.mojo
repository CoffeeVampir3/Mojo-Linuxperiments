from sys.intrinsics import inlined_assembly
from sys.info import size_of
from memory import UnsafePointer, memcpy
from os.atomic import Atomic, Consistency
import linux.syscalls as linux
from notstdcollections import HeapMoveArray
from numa import NumaInfo, CpuMask

comptime AtomicInt32 = Atomic[DType.int32]
# Uniform worker call ABI:
#   workers always invoke as (arg0..arg5) where each arg is a 64-bit integer-class
#   value (pointers or Ints).
# User kernels may take any *prefix* of these arguments (0..6 total); extra
# trailing args are passed by the caller but ignored by the callee on x86-64 SysV.
#
# Per-worker metadata (like worker_id) is published via %fs-relative storage in
# the worker slot. Call `current_worker_id()` from inside a kernel if needed.
comptime KernelFn = fn(Int, Int, Int, Int, Int, Int)

@fieldwise_init
@register_passable("trivial")
struct ArgPack:
    var arg0: Int
    var arg1: Int
    var arg2: Int
    var arg3: Int
    var arg4: Int
    var arg5: Int
    var pad0: Int
    var pad1: Int

    fn __init__(out self):
        self.arg0 = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0
        self.arg4 = 0
        self.arg5 = 0
        self.pad0 = 0
        self.pad1 = 0

fn ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)

fn get_fs_base() -> Int:
    return Int(inlined_assembly["mov %fs:0, $0", Int, constraints="=r"]())

fn pause():
    inlined_assembly["pause", NoneType, constraints="~{memory}"]()

# Memory layout per worker slot (slot_base points at the start of the TLS block;
# FS base points at the TCB at slot_base + TCB):
# [TLS 256B][TCB 64B][child_tid 4B][pad 4B][worker_id 8B][magic 8B][pad..HEADER][Guard 4KB][Stack stack_size][AltGuard 4KB][AltStack altstack_size][pad..slot_end]
@register_passable("trivial")
struct SlotLayout:
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
    comptime ALTSTACK_SIZE = 64 * 1024 # Much bigger than MINSTK however it overflows frequently at that size.
    comptime ALT_GUARD = Self.GUARD
    comptime DEFAULT_STACK = 64 * 1024

fn slot_size[stack_size: Int]() -> Int:
    constrained[
        stack_size >= SlotLayout.GUARD and stack_size % SlotLayout.GUARD == 0,
        "stack_size must be a multiple of 4096 (>= 4096)",
    ]()
    # Must be page-aligned so each worker's guard page can be protected with mprotect.
    var raw = SlotLayout.HEADER + SlotLayout.GUARD + stack_size + SlotLayout.ALT_GUARD + SlotLayout.ALTSTACK_SIZE
    return ((raw + SlotLayout.GUARD - 1) // SlotLayout.GUARD) * SlotLayout.GUARD

@always_inline
fn current_worker_id() -> Int:
    """Return worker id when running in a BurstPool worker, else -1."""
    var magic = inlined_assembly[
        "mov %fs:" + String(SlotLayout.WORKER_MAGIC_FROM_FS) + ", $0",
        Int,
        constraints="=r",
    ]()
    if magic != SlotLayout.WORKER_MAGIC_VALUE:
        return -1
    return inlined_assembly[
        "mov %fs:" + String(SlotLayout.WORKER_ID_FROM_FS) + ", $0",
        Int,
        constraints="=r",
    ]()

@register_passable("trivial")
struct KernelSigInfo:
    # Minimal siginfo_t prefix + si_addr for SIGSEGV (x86_64).
    var si_signo: Int32
    var si_errno: Int32
    var si_code: Int32
    var pad0: Int32
    var si_addr: Int

fn burst_sigsegv_handler(signo: Int32, info: Int, ucontext: Int):
    comptime UCONTEXT_GREGS_OFFSET = 40
    comptime REG_RSP = 15
    comptime REG_RIP = 16
    var gregs = ucontext + UCONTEXT_GREGS_OFFSET
    var rsp = ptr[UInt64](gregs + REG_RSP * 8)[]
    var rip = ptr[UInt64](gregs + REG_RIP * 8)[]

    var worker = current_worker_id()
    var si_addr = UInt64(ptr[KernelSigInfo](info)[].si_addr)
    var pid = linux.sys_getpid()
    var tid = linux.sys_gettid()

    print(
        "burst: SIGSEGV worker=", worker,
        "pid=", pid,
        "tid=", tid,
        "rip=", hex(rip),
        "rsp=", hex(rsp),
        "addr=", hex(si_addr),
    )

    _ = linux.sys_tgkill(pid, tid, linux.Signal.SEGV)
    linux.sys_exit_group(128 + Int(signo))

fn sig_restorer():
    inlined_assembly[
        "mov $$15, %rax\nsyscall",
        NoneType,
        constraints="~{rax},~{rcx},~{r11},~{memory}",
    ]()

fn install_burst_sigsegv_handler():
    var handler_copy = burst_sigsegv_handler
    var handler_addr = UnsafePointer(to=handler_copy).bitcast[Int]()[]
    var restorer_copy = sig_restorer
    var restorer_addr = UnsafePointer(to=restorer_copy).bitcast[Int]()[]

    var act = linux.KernelSigAction()
    act.handler = handler_addr
    act.flags = UInt64(
        linux.SigActionFlag.SIGINFO | linux.SigActionFlag.ONSTACK |
        linux.SigActionFlag.RESTORER
    )
    act.restorer = restorer_addr
    act.mask = linux.KernelSigSet()

    _ = linux.sys_rt_sigaction(linux.Signal.SEGV, UnsafePointer(to=act))

struct SharedPoolState:
    # Cache line padding to avoid false sharing between dispatch and completion fields.
    # Cache line 1: Work dispatch (main writes, workers read)
    var work_available: AtomicInt32  # Workers decrement to claim work
    var shutdown: AtomicInt32        # Shutdown signal
    var func_ptr: Int               # Kernel entry bits

    # Pad the rest of the CL
    comptime DispatchPadBytes = 64 - (
        size_of[type_of(Self().work_available)]()
        + size_of[type_of(Self().shutdown)]()
        + size_of[type_of(Self().func_ptr)]()
    )
    var pad0: InlineArray[UInt8, Self.DispatchPadBytes]

    # Cache line 2: Completion tracking (workers write, main reads)
    var work_done: AtomicInt32       # Jobs remaining; workers decrement when done

    comptime DonePadBytes = 64 - size_of[type_of(Self().work_done)]()
    var pad1: InlineArray[UInt8, Self.DonePadBytes]

    fn __init__(out self):
        self.work_available = AtomicInt32(0)
        self.shutdown = AtomicInt32(0)
        self.func_ptr = 0
        self.pad0 = InlineArray[UInt8, Self.DispatchPadBytes](uninitialized=True)
        self.work_done = AtomicInt32(0)
        self.pad1 = InlineArray[UInt8, Self.DonePadBytes](uninitialized=True)

struct WorkerSlot(Movable, ImplicitlyDestructible):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var child_tid: UnsafePointer[Int32, MutAnyOrigin]
    var stack_top: UnsafePointer[UInt8, MutAnyOrigin]

    fn __init__(out self, slot_base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base)
        self.child_tid = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.CHILD_TID)
        self.stack_top = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_base + SlotLayout.HEADER + SlotLayout.GUARD)

    fn __moveinit__(out self, deinit other: Self):
        self.base = other.base
        self.child_tid = other.child_tid
        self.stack_top = other.stack_top

    @always_inline
    fn is_alive(self) -> Bool:
        return self.child_tid[] != 0

struct WorkerStackHead[mask_size: Int]:
    var entry: Int
    var slot_base: Int
    var worker_id: Int
    var parent_fs: Int
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]
    var args_base: UnsafePointer[ArgPack, MutAnyOrigin]
    var futex_flags: Int
    var altstack_base: Int
    var altstack_size: Int
    var pinned: Int
    var cpu_mask: CpuMask[Self.mask_size]

    fn __init__(out self, entry: Int, slot_base: Int, parent_fs: Int,
                worker_id: Int,
                shared: UnsafePointer[SharedPoolState, MutAnyOrigin],
                args_base: UnsafePointer[ArgPack, MutAnyOrigin],
                futex_flags: Int, altstack_base: Int, altstack_size: Int, pinned: Int, var cpu_mask: CpuMask[Self.mask_size]):
        self.entry = entry
        self.slot_base = slot_base
        self.worker_id = worker_id
        self.parent_fs = parent_fs
        self.shared = shared
        self.args_base = args_base
        self.futex_flags = futex_flags
        self.altstack_base = altstack_base
        self.altstack_size = altstack_size
        self.pinned = pinned
        self.cpu_mask = cpu_mask^

struct BurstPool[stack_size: Int = SlotLayout.DEFAULT_STACK, mask_size: Int = 128](Movable):
    comptime slot_size = slot_size[Self.stack_size]()
    var slots: HeapMoveArray[WorkerSlot]
    var shared: UnsafePointer[SharedPoolState, MutAnyOrigin]
    var args_base: UnsafePointer[ArgPack, MutAnyOrigin]
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
        self.args_base = UnsafePointer[ArgPack, MutAnyOrigin]()
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.workers_alive = False

        # Use plain PRIVATE futexes (not NUMA-bucketed) to allow CHILD_CLEARTID to work
        self.futex_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE

        install_burst_sigsegv_handler()

        var args_arena_size = capacity * size_of[ArgPack]()
        var arena_size = Self.slot_size * capacity + size_of[SharedPoolState]() + args_arena_size
        self.arena_base = linux.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.NORESERVE | linux.MapFlag.POPULATE
        ](0, arena_size)
        if self.arena_base < 0:
            return

        if numa_node is not None:
            var nodemask = UInt64(1) << numa_node.value()
            if linux.sys_mbind[policy=linux.Mempolicy.BIND](self.arena_base, arena_size, nodemask) < 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return

        var shared_addr = self.arena_base + Self.slot_size * capacity
        self.shared = UnsafePointer[SharedPoolState, MutAnyOrigin](unsafe_from_address=shared_addr)
        self.shared[] = SharedPoolState()
        self.args_base = UnsafePointer[ArgPack, MutAnyOrigin](
            unsafe_from_address=shared_addr + size_of[SharedPoolState]()
        )

        for i in range(capacity):
            var slot_base = self.arena_base + i * Self.slot_size
            if linux.sys_mprotect(slot_base + SlotLayout.HEADER, SlotLayout.GUARD, linux.Prot.NONE) != 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return
            if linux.sys_mprotect(
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + Self.stack_size,
                SlotLayout.ALT_GUARD,
                linux.Prot.NONE,
            ) != 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.arena_base = 0
                return
            var slot = WorkerSlot(slot_base)
            slot.child_tid[] = 0
            self.slots.push(slot^)

        self.spawn_workers()

    fn __moveinit__(out self, deinit other: Self):
        self.slots = other.slots^
        self.shared = other.shared
        self.args_base = other.args_base
        self.arena_base = other.arena_base
        self.capacity = other.capacity
        self.cpu_mask = other.cpu_mask.copy()
        self.numa_node = other.numa_node
        self.futex_flags = other.futex_flags
        self.pinned = other.pinned
        self.workers_alive = other.workers_alive

    fn __del__(deinit self):
        if self.arena_base == 0:
            return

        if self.workers_alive:
            # Signal shutdown and wake all workers
            AtomicInt32.store[ordering=Consistency.RELEASE](
                UnsafePointer(to=self.shared[].shutdown.value), 1)
            var workPtr = UnsafePointer(to=self.shared[].work_available.value)
            _ = linux.sys_futex_wake(Int(workPtr), self.capacity, self.futex_flags)

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

        _ = linux.sys_munmap(
            self.arena_base,
            Self.slot_size * self.capacity + size_of[SharedPoolState]() + self.capacity * size_of[ArgPack]()
        )

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

    fn dispatch[F: AnyTrivialRegType](mut self, kernel: F, packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int = -1):
        """Launch `num_jobs` packs to workers and return immediately.

        Each pack is 8×Int (64B). arg0..arg5 are user arguments; pad0/pad1 are reserved.
        Workers steal jobs via work_available and invoke the kernel using the uniform ABI
        `(arg0..arg5)`. The user kernel may accept any prefix of these arguments.
        Use `current_worker_id()` inside a kernel if you need the worker id.
        Call `join()` to wait for completion.
        """
        var jobs = num_jobs if num_jobs >= 0 else self.capacity
        debug_assert(jobs <= self.capacity, "num_jobs must be <= pool capacity")
        if jobs <= 0:
            return

        comptime KernelType = type_of(kernel)
        constrained[size_of[KernelType]() == 8, "kernel must be an 8-byte function pointer"]()

        var donePtr = UnsafePointer(to=self.shared[].work_done.value)
        debug_assert(
            AtomicInt32.load[ordering=Consistency.ACQUIRE](donePtr) == 0,
            "previous dispatch still in flight; call join() first",
        )

        # Copy per-job packs into fixed args arena (one ArgPack per job).
        for i in range(jobs):
            var src = packs + i
            var dst = self.args_base + i
            dst[] = src[]

        var kernel_copy = kernel
        var kernel_ptr = UnsafePointer(to=kernel_copy).bitcast[Int]()[]
        self.shared[].func_ptr = kernel_ptr

        var workPtr = UnsafePointer(to=self.shared[].work_available.value)

        # Publish jobs remaining and work available with release semantics.
        AtomicInt32.store[ordering=Consistency.MONOTONIC](donePtr, Int32(jobs))
        AtomicInt32.store[ordering=Consistency.RELEASE](workPtr, Int32(jobs))

        _ = linux.sys_futex_wake(Int(workPtr), jobs, self.futex_flags)

    fn join(mut self):
        """Wait for the most recent dispatch to complete."""
        var donePtr = UnsafePointer(to=self.shared[].work_done.value)
        while AtomicInt32.load[ordering=Consistency.ACQUIRE](donePtr) > 0:
            pause()

    fn spawn_workers(mut self):
        var parent_fs = get_fs_base()

        for i in range(self.capacity):
            var worker_mask = self.cpu_mask.copy() if self.pinned else CpuMask[Self.mask_size]()

            var stack_top_addr = Int(self.slots[i][].stack_top) + Self.stack_size
            var stack_head_addr = (stack_top_addr - size_of[WorkerStackHead[Self.mask_size]]()) & ~15
            var head = ptr[WorkerStackHead[Self.mask_size]](stack_head_addr)
            var worker_main_copy = worker_main[Self.mask_size]
            var slot_base = Int(self.slots[i][].base)
            var altstack_base = (
                slot_base + SlotLayout.HEADER + SlotLayout.GUARD + Self.stack_size + SlotLayout.ALT_GUARD
            )
            head[] = WorkerStackHead[Self.mask_size](
                UnsafePointer(to=worker_main_copy).bitcast[Int]()[],
                slot_base,
                parent_fs,
                i,
                self.shared,
                self.args_base,
                self.futex_flags,
                altstack_base,
                SlotLayout.ALTSTACK_SIZE,
                Int(self.pinned),
                worker_mask^,
            )
            var tcb_addr = Int(self.slots[i][].base) + SlotLayout.TCB
            var clone_args = linux.Clone3Args.thread(
                Int(self.slots[i][].stack_top),
                stack_head_addr - Int(self.slots[i][].stack_top),
                tcb_addr,
                Int(self.slots[i][].child_tid)
            )

            var result = clone3_with_entry(UnsafePointer(to=clone_args), size_of[linux.Clone3Args]())
            if result < 0:
                return
        self.workers_alive = True

fn clone3_with_entry(clone_args_ptr: UnsafePointer[linux.Clone3Args, MutAnyOrigin], clone_args_size: Int) -> Int:
    # Child diverges via ret to entry point on stack head.
    # Take a typed pointer so the compiler tracks the argument's lifetime.
    return Int(inlined_assembly[
        "mov $$435, %rax\nsyscall\ntest %rax, %rax\njnz 1f\nmov %rsp, %rdi\nret\n1:",
        Int, Int, Int,
        constraints="={rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
    ](Int(clone_args_ptr), clone_args_size))

fn worker_main[mask_size: Int](stack_head_ptr: Int):
    var head_ptr = ptr[WorkerStackHead[mask_size]](stack_head_ptr)
    var altstack_base_val = head_ptr[].altstack_base
    var altstack_size_val = head_ptr[].altstack_size
    var ss = linux.StackT()
    ss.ss_sp = altstack_base_val
    ss.ss_size = UInt64(altstack_size_val)
    ss.ss_flags = 0
    _ = linux.sys_sigaltstack(UnsafePointer(to=ss))
    var futex_flags = head_ptr[].futex_flags
    var slot_base = head_ptr[].slot_base
    var worker_id = head_ptr[].worker_id
    var shared = head_ptr[].shared
    var args_base = head_ptr[].args_base

    # Derive addresses from slot_base
    var tcb_addr = slot_base + SlotLayout.TCB
    # TLS init must happen before any print/stdlib calls.
    # Parent FS base points at its TCB and static TLS precedes it, so copy the
    # TLS+TCB block as one contiguous region.
    comptime TLS_TCB_SIZE = SlotLayout.TLS_SIZE + SlotLayout.TCB_SIZE
    memcpy(
        dest=ptr[Int8](slot_base),
        src=ptr[Int8](head_ptr[].parent_fs - SlotLayout.TLS_SIZE),
        count=TLS_TCB_SIZE,
    )
    ptr[Int](tcb_addr + SlotLayout.TCB_SELF_OFFSET)[] = tcb_addr

    # Publish worker_id into %fs-relative storage for kernel access.
    ptr[Int](slot_base + SlotLayout.WORKER_ID)[] = worker_id
    ptr[Int](slot_base + SlotLayout.WORKER_MAGIC)[] = SlotLayout.WORKER_MAGIC_VALUE

    # Pin to CPU if requested by the parent.
    if head_ptr[].pinned != 0:
        var ret = linux.sys_sched_setaffinity(0, mask_size, Int(head_ptr[].cpu_mask.ptr()))
        if ret != 0:
            print("sched_setaffinity failed:", ret)

    comptime SPIN_LIMIT = 1000  # Spin iterations before sleeping
    var workPtr = UnsafePointer(to=shared[].work_available.value)

    while True:
        # Check for shutdown
        if shared[].shutdown.load[ordering=Consistency.ACQUIRE]() != 0:
            break

        # Try to claim work by atomically decrementing work_available
        var avail = shared[].work_available.load[ordering=Consistency.MONOTONIC]()

        if avail > 0:
            # Try to claim one unit of work
            var old = shared[].work_available.fetch_sub[ordering=Consistency.ACQUIRE_RELEASE](1)
            if old > 0:
                # Successfully claimed work - execute the corresponding pack.
                var job_idx = Int(old - 1)
                var pack_ptr = args_base + job_idx
                var func_addr = shared[].func_ptr
                UnsafePointer(to=func_addr).bitcast[KernelFn]()[](
                    pack_ptr[].arg0,
                    pack_ptr[].arg1,
                    pack_ptr[].arg2,
                    pack_ptr[].arg3,
                    pack_ptr[].arg4,
                    pack_ptr[].arg5,
                )

                # Signal completion by decrementing jobs remaining.
                _ = shared[].work_done.fetch_sub[ordering=Consistency.ACQUIRE_RELEASE](1)
                continue
            else:
                # If we fetch <= 0 there's no more work, compare to lowest racing loser
                # and have the lowest racing loser set to 0 only if it's still that value,
                # to not screw up new dispatches.
                var expected = old - 1
                _ = shared[].work_available.compare_exchange(expected, 0)

        # A note, because it wasn't obvious to me. If you do not futex wait the scheduler
        # on linux can absolutely deadlock spinning threads by not scheduling
        # other (wanting to work) threads, leading to no-work on spinning threads and
        # never re-scheduling waiting threads, so the work effectively deadlocks.
        # (The threads look busy because they spin on the atomic.)
        # Basically, we could go faster but it runs into stochastic failure
        # unless we set up the system to specifically account for this number
        # of physical threads to be dispatched by us.
        # There's about a 6x penalty in latency cost (worst case) doing this VS pure spin
        # however it guarantees work will always attempt to complete (in bounded time)
        # and this doesn't need complex user/system configuration. But it's not optimal.

        # No work available - spin briefly then sleep
        var spins = 0
        while shared[].work_available.load[ordering=Consistency.MONOTONIC]() <= 0:
            if shared[].shutdown.load[ordering=Consistency.MONOTONIC]() != 0:
                break
            if spins < SPIN_LIMIT:
                pause()
                spins += 1
            else:
                # Sleep on work_available address, expecting value == 0
                _ = linux.sys_futex_wait(Int(workPtr), 0, futex_flags)
                spins = 0  # Reset spin count after wake

    # CHILD_CLEARTID handles clearing child_tid and futex wake automatically
    linux.sys_exit()
