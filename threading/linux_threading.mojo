from sys.intrinsics import inlined_assembly
from sys.info import size_of
from memory import UnsafePointer, MutUnsafePointer, memcpy
from os import Atomic
import threading.linux as linux
from numa import NumaInfo, CpuMask

# Words some people might care about:
# Not POSIX
# Linux-Kernel level threads
# Clone3
# Futex2
# Rseq
# Numa Aware (For both memory and futex)
# Parameterized stack size
# Parameterized numa cpu masking size (For large servers)
# No thread locals (our entire stack is thread local, user is in charge for heap allocations)
# Fake not good "argument passing" for 64 bit trivial types.
# Privately mapped with self waking


# Open questions:
# Numa placement/allocation -- memory becomes awkward. This becomes even more complex if concurrency factors
# in that gpus exist. Optimal thread usage is strongly tied to the allocation strategy so
# it's likely a production version of this would either take an allocator or take preallocated memory
# The largest concern I have is around how to allocate when the user cares and how to not make the API offensive
# when they don't.


# Design notes:
# ThreadPool(1) VS Thread:
#     Single threads are a thread pool. There's currently no 'Thread' abstraction because variadic forwarding is not
#     yet implemented in mojo. Otherwise 'Thread' could be a light wrapper around `ThreadPool(1)`
#     I think it's inoffensive enough at the moment that the increased code area is a worse proposition that the somewhat
#     weird temporary abstraction.


# Non-Mojo Allocations:
#     The mojo allocator is not very flexible and it's easier currently to just work around it.
#     In a production setting, it'd be ideal to have an allocator trait that we pass in and could do the
#     numa aware arena stuff. 

@fieldwise_init
struct ThreadPool[stack_size: Int = StackSize.DEFAULT, mask_size: Int = 128](Movable):
    alias slot_size = slot_total_size[Self.stack_size]()
    var slots: List[ThreadSlot]
    var cpu_mask: CpuMask[Self.mask_size]
    var arena_base: Int
    var capacity: Int
    var numa_node: Optional[Int]
    var pinned: Bool
    var valid: Bool
    var futex_flags: Int

    fn __init__(out self, capacity: Int, var cpu_mask: CpuMask[Self.mask_size] = CpuMask[Self.mask_size](), numa_node: Optional[Int] = None):
        self.capacity = capacity
        self.slots = List[ThreadSlot](capacity=capacity)
        self.pinned = cpu_mask.count() > 0
        self.cpu_mask = cpu_mask^
        self.numa_node = numa_node
        self.arena_base = 0
        self.valid = True
        alias base_flags = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE
        self.futex_flags = base_flags | linux.Futex2.NUMA if numa_node is not None else base_flags
        self.allocate_arena()

    # What should actually happen if we try to delete the threadpool while slots are in use?
    # Currently there's no means to shut threads down and it's also very suspicious practice
    # to stall what is likely the main thread on a destructor to wait for thread shutdown.
    fn __del__(deinit self):
        if self.valid:
            for i in range(len(self.slots)):
                if self.slots[i].in_use():
                    print("FATAL: ThreadPool destroyed with active threads")
                    from os import abort
                    abort()
            _ = linux.sys_munmap(self.arena_base, Self.slot_size * self.capacity)

    fn __bool__(self) -> Bool:
        return self.valid

    fn __iter__(self) -> PoolIter:
        return PoolIter(self.capacity)

    fn __len__(self) -> Int:
        return self.capacity

    # A thread pool for a whole numa node.
    @staticmethod
    fn for_numa_node(numa: NumaInfo, node: Int) -> Self:
        return Self(numa.cpus_on_node(node), numa.get_node_mask[Self.mask_size](node), node)

    # A thread pool for a whole numa node exluding a specific cpu/node. Typically for excluding the orchestration thread.
    @staticmethod
    fn for_numa_node_excluding(numa: NumaInfo, node: Int, exclude_cpu: Int) -> Self:
        var mask = numa.get_node_mask[Self.mask_size](node)
        var cap = numa.cpus_on_node(node)
        if mask.test(exclude_cpu):
            mask.clear(exclude_cpu)
            cap -= 1
        return Self(cap, mask^, node)

    fn launch[F: AnyTrivialRegType, *Ts: ImplicitlyCopyable](mut self, func: F, *args: *Ts) -> Bool:
        for i in range(len(self.slots)):
            if not self.slots[i].in_use():
                var packed_args = InlineArray[Int64, 6](0)
                @parameter
                for j in range(args.__len__()):
                    alias T = type_of(args[j])
                    constrained[size_of[T]() == 8, "args must be 8 bytes"]()
                    var arg = args[j]
                    packed_args[j] = UnsafePointer(to=arg).bitcast[Int64]()[]
                var func_copy = func
                var func_ptr = Int(UnsafePointer(to=func_copy).bitcast[Int64]()[])
                var mask_ptr = Int(self.cpu_mask.ptr()) if self.pinned else 0
                var mask_sz = Self.mask_size if self.pinned else 0
                var tid = spawn_on_slot[args.__len__(), Self.stack_size](self.slots[i], func_ptr, packed_args, mask_ptr, mask_sz, self.futex_flags)
                if tid < 0:
                    return False
                return True
        return False

    fn wait_all(mut self):
        for i in range(len(self.slots)):
            if self.slots[i].in_use():
                wait_for_thread(self.slots[i].child_tid_addr, self.futex_flags)

    fn wait_any(mut self) -> Int:
        return wait_for_any_thread(self.slots, self.futex_flags)

    fn allocate_arena(mut self):
        var arena_size = Self.slot_size * self.capacity
        self.arena_base = linux.sys_mmap[prot=linux.Prot.RW, flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.NORESERVE](0, arena_size)
        if self.arena_base < 0:
            self.valid = False
            return
        if self.numa_node is not None:
            var nodemask = UInt64(1) << self.numa_node.value()
            var bind_result = linux.sys_mbind[policy=linux.Mempolicy.BIND](self.arena_base, arena_size, nodemask)
            if bind_result < 0:
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.valid = False
                return
        var node_hint = Int32(-1) if self.numa_node is None else Int32(self.numa_node.value())
        for i in range(self.capacity):
            var slot_base = self.arena_base + i * Self.slot_size
            var guard_addr = slot_base + SlotLayout.HEADER
            if not setup_guard_page(guard_addr, PageSize.GUARD):
                _ = linux.sys_munmap(self.arena_base, arena_size)
                self.valid = False
                return
            var child_tid_addr = slot_base + SlotLayout.CHILD_TID
            ptr[linux.FutexNuma32](child_tid_addr)[] = linux.FutexNuma32(node_hint)
            self.slots.append(ThreadSlot(
                base=slot_base,
                child_tid_addr=child_tid_addr,
                stack_base=guard_addr + PageSize.GUARD,
            ))

fn set_thread_affinity[mask_size: Int](mask: CpuMask[mask_size]) -> Bool:
    """Pin calling thread to CPUs in mask."""
    return linux.sys_sched_setaffinity(0, mask_size, Int(mask.ptr())) == 0

fn get_cpu_and_node_rseq() -> Tuple[Int, Int]:
    """Get current CPU and NUMA node via rseq"""
    alias cpu_offset = SlotLayout.RSEQ_FROM_FS + RseqOffsets.CPU_ID
    alias node_offset = SlotLayout.RSEQ_FROM_FS + RseqOffsets.NODE_ID
    var cpu = Int(inlined_assembly["movl %fs:" + String(cpu_offset) + ", $0", UInt32, constraints="=r"]())
    var node = Int(inlined_assembly["movl %fs:" + String(node_offset) + ", $0", UInt32, constraints="=r"]())
    return (cpu, node)

alias MutexGatePtr = MutUnsafePointer[Int32, MutOrigin.external]

@fieldwise_init
@register_passable("trivial")
struct MutexGate:
    """Single-use futex-based synchronization token.
    One thread waits, another releases. Not reentrant - reset before reuse."""
    var ptr: MutexGatePtr

    @always_inline
    @staticmethod
    fn from_addr(addr: Int) -> Self:
        return Self(ptr=MutexGatePtr(unsafe_from_address=addr))

    @always_inline
    fn wait(self, flags: Int = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE):
        while self.ptr[] == 0:
            _ = linux.sys_futex_wait(Int(self.ptr), 0, flags)

    @always_inline
    fn release(self, flags: Int = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE):
        self.ptr[] = 1
        _ = linux.sys_futex_wake(Int(self.ptr), 1, flags)

    @always_inline
    fn reset(self):
        self.ptr[] = 0

# === Internal Implementation ===

@register_passable("trivial")
struct PageSize:
    alias STANDARD = 4096
    alias GUARD = Self.STANDARD

@register_passable("trivial")
struct StackSize:
    alias DEFAULT = 64 * 1024

@register_passable("trivial")
struct TLS:
    alias STATIC_SIZE = 256
    alias TCB_SIZE = 64
    alias TCB_SELF_OFFSET = 0x10

@register_passable("trivial")
struct SlotLayout:
    alias TCB = TLS.STATIC_SIZE
    alias CHILD_TID = Self.TCB + TLS.TCB_SIZE
    alias RSEQ = ((Self.CHILD_TID + 8 + 31) // 32) * 32
    alias RSEQ_FROM_FS = Self.RSEQ - Self.TCB
    alias HEADER = ((Self.CHILD_TID + 8 + PageSize.STANDARD - 1) // PageSize.STANDARD) * PageSize.STANDARD

@register_passable("trivial")
struct RseqOffsets:
    alias CPU_ID = 4
    alias NODE_ID = 20

alias RSEQ_CPU_ID_UNINITIALIZED = UInt32(0xffffffff)

@register_passable("trivial")
struct StackHead:
    var entry: Int64
    var slot_base: Int64
    var parent_fs: Int64
    var user_func: Int64
    var cpu_mask_ptr: Int64
    var cpu_mask_size: Int64
    var futex_flags: Int64
    var arg0: Int64
    var arg1: Int64
    var arg2: Int64
    var arg3: Int64
    var arg4: Int64
    var arg5: Int64
    alias ARGS_OFFSET = 56

    @always_inline
    fn __init__(out self, entry: Int64, slot_base: Int64, parent_fs: Int64, user_func: Int64,
                cpu_mask_ptr: Int64, cpu_mask_size: Int64, futex_flags: Int64, args: InlineArray[Int64, 6]):
        self.entry = entry
        self.slot_base = slot_base
        self.parent_fs = parent_fs
        self.user_func = user_func
        self.cpu_mask_ptr = cpu_mask_ptr
        self.cpu_mask_size = cpu_mask_size
        self.futex_flags = futex_flags
        self.arg0 = args[0]
        self.arg1 = args[1]
        self.arg2 = args[2]
        self.arg3 = args[3]
        self.arg4 = args[4]
        self.arg5 = args[5]

@fieldwise_init
@register_passable("trivial")
struct ThreadSlot:
    var base: Int
    var child_tid_addr: Int
    var stack_base: Int

    @always_inline
    fn in_use(self) -> Bool:
        return ptr[Int32](self.child_tid_addr)[] != 0

struct PoolIter:
    var current: Int
    var end: Int

    fn __init__(out self, size: Int):
        self.current = 0
        self.end = size

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Int:
        var result = self.current
        self.current += 1
        return result

    fn __has_next__(self) -> Bool:
        return self.current < self.end

    fn __len__(self) -> Int:
        return self.end - self.current

fn slot_total_size[stack_size: Int]() -> Int:
    alias raw = SlotLayout.HEADER + PageSize.GUARD + stack_size
    return ((raw + PageSize.STANDARD - 1) // PageSize.STANDARD) * PageSize.STANDARD

fn ptr[T: AnyType](addr: Int) -> UnsafePointer[T, MutOrigin.external]:
    return UnsafePointer[T, MutOrigin.external](unsafe_from_address=addr)

fn get_fs_base() -> Int:
    return Int(inlined_assembly["mov %fs:0, $0", Int64, constraints="=r"]())

fn child_entry[num_args: Int](stack_head_ptr: Int):
    """Child thread entry point. Receives pointer to StackHead via rdi."""
    var remote_head = UnsafePointer[StackHead, MutOrigin.external](unsafe_from_address=stack_head_ptr)
    var local_head = remote_head[]
    var slot_base = Int(local_head.slot_base)
    init_thread_tls(slot_base, Int(local_head.parent_fs))
    _ = register_rseq(slot_base)
    if local_head.cpu_mask_ptr != 0 and local_head.cpu_mask_size > 0:
        var tid = linux.sys_gettid()
        _ = linux.sys_sched_setaffinity(tid, Int(local_head.cpu_mask_size), Int(local_head.cpu_mask_ptr))

    # Load args from local copy and call user function
    var local_head_ptr = Int(UnsafePointer(to=local_head))
    alias arg_regs = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
    fn build_arg_loads() -> String:
        var s = String("")
        @parameter
        for i in range(num_args):
            s += "mov " + String(StackHead.ARGS_OFFSET + i * 8) + "(%rbx), %" + arg_regs[i] + "\n"
        return s
    alias arg_loads = build_arg_loads()

    var child_tid_addr = slot_base + SlotLayout.CHILD_TID
    var func_addr = local_head.user_func
    var futex_flags = local_head.futex_flags

    # Trampoline into call (user function), landing from trampoline then restores tid address and flags to tid=0 self wake
    var tid_addr_after = inlined_assembly[
        "mov %rsi, %rbx\nmov %rdx, %r12\nmov %rcx, %r13\nand $$-16, %rsp\n" + arg_loads + "call *%rax\nmov %r12, %rax",
        Int64, Int64, Int64, Int64, Int64,
        constraints="={rax},{rax},{rsi},{rdx},{rcx},~{rbx},~{r12},~{r13},~{rsp},~{rdi},~{r8},~{r9},~{r10},~{r11},~{memory}",
    ](func_addr, local_head_ptr, child_tid_addr, futex_flags)
    var flags_after = inlined_assembly["mov %r13, $0", Int64, constraints="={rax},~{r13}"]()

    # Clear tid and wake waiters
    ptr[Int32](Int(tid_addr_after))[] = 0
    _ = linux.sys_futex_wake(Int(tid_addr_after), 1, Int(flags_after))
    linux.sys_exit()

fn clone3_with_entry(clone_args_ptr: Int, clone_args_size: Int) -> Int:
    # Typical clone3 stack head ret trick (diverges main thread and cloned child, child ret's off main path)
    return Int(inlined_assembly[
        "mov $$435, %rax\nsyscall\ntest %rax, %rax\njnz 1f\nmov %rsp, %rdi\nret\n1:",
        Int64, Int64, Int64,
        constraints="={rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
    ](clone_args_ptr, clone_args_size))

fn spawn_on_slot[num_args: Int, stack_size: Int](slot: ThreadSlot, func_ptr: Int, args: InlineArray[Int64, 6], cpu_mask_ptr: Int = 0, cpu_mask_size: Int = 0, futex_flags: Int = linux.Futex2.SIZE_U32 | linux.Futex2.PRIVATE) -> Int:
    var child_entry_copy = child_entry[num_args]
    var child_entry_ptr = Int(UnsafePointer(to=child_entry_copy).bitcast[Int64]()[])
    var stack_head_addr = (slot.stack_base + stack_size - size_of[StackHead]()) & ~15
    var head = UnsafePointer[StackHead, MutOrigin.external](unsafe_from_address=stack_head_addr)
    head[] = StackHead(
        entry=Int64(child_entry_ptr),
        slot_base=Int64(slot.base),
        parent_fs=Int64(get_fs_base()),
        user_func=Int64(func_ptr),
        cpu_mask_ptr=Int64(cpu_mask_ptr),
        cpu_mask_size=Int64(cpu_mask_size),
        futex_flags=Int64(futex_flags),
        args=args,
    )
    var effective_size = stack_head_addr - slot.stack_base
    var tcb_addr = slot.base + SlotLayout.TCB
    var clone_args = linux.Clone3Args.thread(slot.stack_base, effective_size, tcb_addr, slot.child_tid_addr)
    var clone_args_ptr = UnsafePointer(to=clone_args)
    var result = clone3_with_entry(Int(clone_args_ptr), size_of[linux.Clone3Args]())
    _ = clone_args
    _ = head[]
    return result

fn init_thread_tls(slot_base: Int, parent_fs: Int):
    """Initialize child thread's TLS by copying from parent."""
    var tcb_addr = slot_base + SlotLayout.TCB
    memcpy(dest=ptr[Int8](slot_base), src=ptr[Int8](parent_fs - TLS.STATIC_SIZE), count=TLS.STATIC_SIZE)
    memcpy(dest=ptr[Int8](tcb_addr), src=ptr[Int8](parent_fs), count=TLS.TCB_SIZE)
    ptr[Int64](tcb_addr + TLS.TCB_SELF_OFFSET)[] = Int64(tcb_addr)

fn wait_for_thread(child_tid_addr: Int, flags: Int):
    while ptr[Int32](child_tid_addr)[] != 0:
        _ = linux.sys_futex_wait(child_tid_addr, Int(ptr[Int32](child_tid_addr)[]), flags)

fn wait_for_any_thread(slots: List[ThreadSlot], flags: Int) -> Int:
    while True:
        var waiters = List[linux.FutexWaitv](capacity=len(slots))
        var waiter_to_slot = List[Int](capacity=len(slots))
        for i in range(len(slots)):
            var tid = ptr[Int32](slots[i].child_tid_addr)[]
            if tid == 0:
                continue
            waiters.append(linux.FutexWaitv(UInt64(tid), UInt64(slots[i].child_tid_addr), UInt32(flags), 0))
            waiter_to_slot.append(i)
        if len(waiters) == 0:
            return -1
        var woken_idx = linux.sys_futex_waitv(waiters.unsafe_ptr(), len(waiters))
        if woken_idx >= 0 and woken_idx < len(waiter_to_slot):
            var slot_idx = waiter_to_slot[woken_idx]
            if not slots[slot_idx].in_use():
                return slot_idx

fn register_rseq(slot_base: Int) -> Bool:
    """Register rseq for current thread. Call from child thread."""
    var rseq_addr = slot_base + SlotLayout.RSEQ
    var rseq_ptr = UnsafePointer[linux.Rseq, MutOrigin.external](unsafe_from_address=rseq_addr)
    rseq_ptr[] = linux.Rseq()
    rseq_ptr[].cpu_id = RSEQ_CPU_ID_UNINITIALIZED
    rseq_ptr[].cpu_id_start = RSEQ_CPU_ID_UNINITIALIZED
    var result = linux.sys_rseq(rseq_addr, size_of[linux.Rseq](), 0, linux.RSEQ_SIG)
    return result == 0

fn setup_guard_page(addr: Int, size: Int) -> Bool:
    return linux.sys_mprotect(addr, size, linux.Prot.NONE) == 0

# Thread Setup Procedure
#
# Memory Layout (per slot):
#   (Where stack is the very TOP of the memory region, growing down towards the guard.)
#   [Static TLS (256B)][TCB (64B)][child_tid (8B)][rseq (32B aligned)][Guard Page (4kb)][<<- Stack <<-][Stack Head (copied args)]
#   The kernel sets the TCB addr as the fs base for the child thread, TLS is at a negative offset from it.
#
# Pool Creation:
#   mmap a contiguous arena for all slots memory layouts.
#   Given numa, we mbind the arena to the given node
#
# Thread Launch (parent side):
#   Parent writes StackHead to top of child's stack region (in arena)
#      - Contains: entry point, slot_base, parent_fs, user_func, cpu_mask_ptr, futex_flags, args
#      - Page faults on correct node (arena already mbind'd)
#   Parent calls clone3 with CLONE_SETTLS pointing to child's TCB
#   clone3 returns child's TID to parent; child begins executing
#
# Thread Startup (child side):
#   Child wakes at stack top, executes `ret` which pops entry address and jumps to child_entry
#   Child copies StackHead to local_head on stack (arena pages fault on correct node)
#   Child initializes TLS by copying from parent (remote read, local write)
#   Child registers rseq with kernel (required before sched_setaffinity due to rseq inheritance)
#   Child sets CPU affinity via sched_setaffinity
#   Child calls user function with args from local_head
#   On return, child clears child_tid and wakes futex with matching flags, then exits
#
# rseq:
#   The rseq struct lives at a fixed offset from the TCB region.
#   After registration, the kernel updates cpu_id/node_id before returning to userspace.
#   get_cpu_and_node_rseq() reads these directly from memory without a syscall.
#
# NUMA Memory Model:
#   When numa_node is set:
#     - Stack pages: faults on child's node (mbind)
#     - TLS region: memcopy to child
#     - rseq structure: kernel writes after registration to child
#     - child_tid: mbind child
#     - StackHead: mbind child
#   Remote Accesses (One time startup costs):
#     - cpu_mask read: child reads from parent's ThreadPool struct
#     - TLS copy: parent stages TLS (to kickoff thread), child copies the TLS region.
#   User data pointers passed as args are the user's responsibility to place correctly.
