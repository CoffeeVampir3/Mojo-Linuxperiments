from memory import UnsafePointer
from os import abort
from iter import Iterator
from sys import size_of
from numa import CpuMask, NumaInfo
from .x86_64_systhread import (
    syscall_mmap,
    syscall_mprotect,
    syscall_munmap,
    clone_trampoline_capture,
    syscall_futex_wait,
    get_fs_base,
    syscall_sched_setaffinity,
)

# Public API: Thread, ThreadPool, join_all
#
# Thread = ThreadPool[1] - 1-slot pool = single thread. Wow!
# ThreadPool[capacity, stack_size=64 * 1024, cpu_mask_size=0]() - pre-allocated reusable thread slots
#   .launch(func, *args) -> Bool - starts task on available slot, False if full
#   .join() / .join_all() - blocks until all spawned tasks complete, then slots are reusable
#   .for_node(numa, node) -> Self - creates pool pinned to a NUMA node (requires cpu_mask_size > 0)
#   iterable: yields 0..capacity-1
#
# wait_all(*threads) - joins multiple Thread/ThreadPools, blocking until all complete. This is consuming for now.

# stack_size is in bytes, so 64 * 1024 = 64KB of stack -- stack_size=64*1024. Defaults to 64kb

# cpu_mask_size is the size of the NUMA CpuMask in bytes. Defaults to 0 (no NUMA pinning).
#   When > 0, enables the cpu_mask keyword argument for Thread, or pinned mode for ThreadPool.
#   NumaThread/NumaThreadPool use cpu_mask_size=128, sufficient for up to 1024 CPUs.

# args: up to 6, must be 8-byte pointer-sized types (Int, UnsafePointer, etc)
# Similarly there's no return, so write to an out buffer.

# Thread pool was designed for symmetric use on full scatter (Allocate a pool, first TLS touch pins child on node, threads will
# naturally be reused because of list order as long as we use *all* threads.)

# Use NumaThread/NumaThreadPool if you need numa node pinning for the thread.
# EX pins all 4 threads to numa node 0
#   from numa import NumaInfo
#   var numa = NumaInfo()
#   var pool = NumaThreadPool[4].for_node(numa, 0)
#   for i in pool:
#       _ = pool.launch(worker_fn, i)
#   pool.join_all()

# Memory layout constants

# The default size of the NUMA CPU mask in bytes. It's 0 because by default this threading assumes your 
# not using numa by default. (Use NumaThread or NumaThreadPool)
alias DEFAULT_CPU_MASK_SIZE = 0 # This is parameterized. (cpu_mask_size)

alias PAGE_SIZE = 4096
alias DEFAULT_STACK_SIZE = 64 * 1024 # This is parameterized. (stack_size)
alias GUARD_SIZE = PAGE_SIZE

alias Thread[stack_size: Int = DEFAULT_STACK_SIZE, cpu_mask_size: Int = DEFAULT_CPU_MASK_SIZE] = ThreadPool[stack_size, cpu_mask_size]
alias NumaThread = Thread[DEFAULT_STACK_SIZE, 128]
alias NumaThreadPool[stack_size: Int = DEFAULT_STACK_SIZE] = ThreadPool[stack_size, 128]

fn set_thread_affinity[mask_size: Int](mask: CpuMask[mask_size]) -> Bool:
    var ret = syscall_sched_setaffinity(0, Int64(mask_size), mask.ptr())
    return ret == 0

fn set_thread_affinity(cpu_id: Int) -> Bool:
    var mask = CpuMask[128]()
    mask.set(cpu_id)
    return set_thread_affinity(mask)

struct ThreadPool[stack_size: Int = DEFAULT_STACK_SIZE, cpu_mask_size: Int = 0](Movable, Joinable):
    alias slot_size = slot_size[Self.stack_size, Self.cpu_mask_size]()
    alias user_args_offset = args_offset[Self.cpu_mask_size]()
    var slots: List[StackSlot]
    var cpu_mask: CpuMask[Self.cpu_mask_size]
    var capacity: Int
    var pinned: Bool
    var valid: Bool

    fn __init__(out self, capacity: Int = 1):
        self.capacity = capacity
        self.slots = List[StackSlot](capacity=capacity)
        self.cpu_mask = CpuMask[Self.cpu_mask_size]()
        self.pinned = False
        self.valid = True
        self.allocate_slots()

    fn __init__(out self, var cpu_mask: CpuMask[Self.cpu_mask_size], capacity: Int = 1):
        constrained[Self.cpu_mask_size > 0, "ThreadPool cpu_mask_size must be > 0 when using cpu_mask"]()
        self.capacity = capacity
        self.slots = List[StackSlot](capacity=capacity)
        self.cpu_mask = cpu_mask^
        self.pinned = True
        self.valid = True
        self.allocate_slots()

    fn allocate_slots(mut self):
        for _ in range(self.capacity):
            var slot = allocate_slot[Self.stack_size, Self.cpu_mask_size]()
            if slot.mmap_base == 0:
                for j in range(len(self.slots)):
                    _ = syscall_munmap(self.slots[j].mmap_base, Int64(Self.slot_size))
                self.valid = False
                return
            self.slots.append(slot^)

    fn __moveinit__(out self, deinit existing: Self):
        self.slots = existing.slots^
        self.cpu_mask = existing.cpu_mask^
        self.capacity = existing.capacity
        self.pinned = existing.pinned
        self.valid = existing.valid

    @staticmethod
    fn for_node(numa: NumaInfo, node: Int) -> Self:
        constrained[Self.cpu_mask_size > 0, "ThreadPool cpu_mask_size must be > 0 for for_node"]()
        var mask = numa.get_node_mask[Self.cpu_mask_size](node)
        return Self(mask^, numa.cpus_on_node(node))

    @staticmethod
    fn for_node_excluding(numa: NumaInfo, node: Int, exclude_cpu: Int) -> Self:
        constrained[Self.cpu_mask_size > 0, "ThreadPool cpu_mask_size must be > 0 for for_node_excluding"]()
        var mask = numa.get_node_mask[Self.cpu_mask_size](node)
        var capacity = numa.cpus_on_node(node)
        if mask.test(exclude_cpu):
            mask.clear(exclude_cpu)
            capacity -= 1
        return Self(mask^, capacity)

    fn clone_on_slot[num_args: Int](mut self, slot_idx: Int, func_ptr: Int64) -> Bool:
        var base = Int(self.slots[slot_idx].mmap_base)
        var init_copy = init_slot_tls_child
        var init_ptr = UnsafePointer(to=init_copy).bitcast[Int64]()[]
        var effective_mask_size = Int64(0)
        if self.pinned:
            self.cpu_mask.copy_to(ptr(base + CPU_MASK_OFFSET).bitcast[UInt8]())
            effective_mask_size = Int64(Self.cpu_mask_size)
        var tid = clone_trampoline_capture[num_args, Self.user_args_offset, CPU_MASK_OFFSET](
            Int64(THREAD_FLAGS),
            Int64(self.slots[slot_idx].stack_top),
            Int64(self.slots[slot_idx].child_tid_addr),
            Int64(self.slots[slot_idx].tcb_addr),
            init_ptr,
            Int64(self.slots[slot_idx].mmap_base),
            func_ptr,
            effective_mask_size,
        )
        if tid < 0:
            self.slots[slot_idx].in_use = 0
            return False
        return True

    # There's two launch methods for now because variadic args can't be forwarded right now.
    # So this is just a temporary hacky fix.
    fn launch[F: AnyTrivialRegType, *Ts: ImplicitlyCopyable](mut self, func: F, *args: *Ts) -> Bool:
        for i in range(len(self.slots)):
            if self.slots[i].in_use == 0:
                self.slots[i].in_use = 1
                var base = Int(self.slots[i].mmap_base)
                ptr(base + INIT_PARENT_FS_OFFSET)[] = get_fs_base()
                @parameter
                for j in range(args.__len__()):
                    alias T = type_of(args[j])
                    constrained[size_of[T]() == 8, "launch args must be 8 bytes (pointer-sized)"]()
                    var arg = args[j]
                    ptr(base + Self.user_args_offset + j * 8)[] = UnsafePointer(to=arg).bitcast[Int64]()[]
                var func_copy = func
                var func_ptr = UnsafePointer(to=func_copy).bitcast[Int64]()[]
                return self.clone_on_slot[args.__len__()](i, func_ptr)
        return False

    # Chaining launch that returns self, allows us to chain into launch which is usually how you use
    # a single thread.
    fn launch[F: AnyTrivialRegType, *Ts: ImplicitlyCopyable](var self, func: F, *args: *Ts) -> Self:
        for i in range(len(self.slots)):
            if self.slots[i].in_use == 0:
                self.slots[i].in_use = 1
                var base = Int(self.slots[i].mmap_base)
                ptr(base + INIT_PARENT_FS_OFFSET)[] = get_fs_base()
                @parameter
                for j in range(args.__len__()):
                    alias T = type_of(args[j])
                    constrained[size_of[T]() == 8, "launch args must be 8 bytes (pointer-sized)"]()
                    var arg = args[j]
                    ptr(base + Self.user_args_offset + j * 8)[] = UnsafePointer(to=arg).bitcast[Int64]()[]
                var func_copy = func
                var func_ptr = UnsafePointer(to=func_copy).bitcast[Int64]()[]
                if not self.clone_on_slot[args.__len__()](i, func_ptr):
                    self.valid = False
                return self^
        self.valid = False
        return self^

    fn __bool__(self) -> Bool:
        return self.valid

    fn __iter__(self) -> PoolIter:
        return PoolIter(self.capacity)

    fn join_all(mut self):
        for i in range(len(self.slots)):
            if self.slots[i].in_use != 0:
                wait_for_thread(self.slots[i].child_tid_addr)
                self.slots[i].in_use = 0

    fn join(mut self):
        self.join_all()

    fn __del__(deinit self):
        if not self.valid:
            return
        for i in range(len(self.slots)):
            if self.slots[i].in_use != 0:
                print("FATAL: ThreadPool destroyed with active threads, call join_all() first")
                abort()
            _ = syscall_munmap(self.slots[i].mmap_base, Int64(Self.slot_size))

fn wait_all[*Ts: Joinable](var *threads: *Ts):
    @parameter
    for i in range(threads.__len__()):
        threads[i].join()

# This is the current user facing api. From here be dragons.

trait Joinable:
    fn join(mut self): ...

struct PoolIter(Iterator):
    alias Element = Int
    var index: Int
    var capacity: Int

    fn __init__(out self, capacity: Int):
        self.index = 0
        self.capacity = capacity

    fn __next__(mut self) -> Self.Element:
        var i = self.index
        self.index += 1
        return i

    fn __has_next__(self) -> Bool:
        return self.index < self.capacity

fn wait_for_thread(child_tid_addr: Int):
    var p = ptr(child_tid_addr)
    while p[] != 0:
        _ = syscall_futex_wait(Int64(child_tid_addr), p[])

# Basic thread implementation for x86-64 Linux.
#
# References:
#   - glibc/nptl/pthread_create.c: clone_flags, __clone_internal call
#   - glibc/sysdeps/unix/sysv/linux/x86_64/clone.S: syscall wrapper
#   - glibc/sysdeps/x86_64/nptl/tls.h: tcbhead_t structure
#   - glibc/nptl/allocatestack.c: stack/TLS memory layout
#   - https://chao-tic.github.io/blog/2018/12/25/tls: TLS deep dive
#
# TLS Memory Layout (x86-64, TLS 2, TLS_TCB_AT_TP):
#   [Static TLS blocks] [TCB (tcbhead_t)] [struct pthread]
#   Static TLS access: mov %fs:-offset, %reg (negative offset from TCB)
#
# Key simplifications vs glibc NPTL:
#   - No signal mask setup during creation
#   - No mutex list
#   - (We) Copy parent's static TLS instead of reinitializing from .tdata

fn slot_size[stack_size: Int, cpu_mask_size: Int]() -> Int:
    alias tls_tcb_dtv = STATIC_TLS_SIZE + TCB_SIZE + (DTV_ENTRIES * DTV_ENTRY_SIZE)
    alias metadata_size = 8 + cpu_mask_size + 48  # parent_fs + cpu_mask + args (6 * 8)
    alias total_tls = tls_tcb_dtv + metadata_size
    alias tls_region = ((total_tls + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    return tls_region + GUARD_SIZE + stack_size

# Static TLS size estimate. Since Mojo has no thread-local facility yet,
# we only need enough for glibc internals (errno, locale).
alias STATIC_TLS_SIZE = 256

# TCB size - only need through stack_guard (0x28) + pointer_guard (0x30).
alias TCB_SIZE = 64

# Clone flags - identical to glibc NPTL pthread_create.c
alias CLONE_VM = 0x00000100
alias CLONE_FS = 0x00000200
alias CLONE_FILES = 0x00000400
alias CLONE_SIGHAND = 0x00000800
alias CLONE_THREAD = 0x00010000
alias CLONE_SYSVSEM = 0x00040000
alias CLONE_SETTLS = 0x00080000
alias CLONE_PARENT_SETTID = 0x00100000
alias CLONE_CHILD_CLEARTID = 0x00200000

alias THREAD_FLAGS = (CLONE_VM | CLONE_FS | CLONE_FILES |
                      CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM |
                      CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID)

# TCB offsets from glibc sysdeps/x86_64/nptl/tls.h tcbhead_t
alias TCB_OFFSET_TCB = 0x00
alias TCB_OFFSET_DTV = 0x08
alias TCB_OFFSET_SELF = 0x10

# DTV structure
alias DTV_ENTRIES = 4
alias DTV_ENTRY_SIZE = 16

# mmap flags and protections
alias PROT_NONE = 0x0
alias PROT_READ = 0x1
alias PROT_WRITE = 0x2
alias MAP_PRIVATE = 0x02
alias MAP_ANONYMOUS = 0x20
alias MAP_FAILED = -1

fn ptr(addr: Int) -> UnsafePointer[Int64, MutOrigin.external]:
    var p: UnsafePointer[Int64, MutOrigin.external]
    p = type_of(p)(unsafe_from_address=addr)
    return p

alias TLS_TCB_DTV_SIZE = STATIC_TLS_SIZE + TCB_SIZE + (DTV_ENTRIES * DTV_ENTRY_SIZE)

# Offset in mmap region to store parent_fs for child to read (in padding before guard)
alias INIT_PARENT_FS_OFFSET = TLS_TCB_DTV_SIZE + 16
# Offset for CPU affinity mask (after parent_fs)
alias CPU_MASK_OFFSET = INIT_PARENT_FS_OFFSET + 8

fn args_offset[cpu_mask_size: Int]() -> Int:
    return CPU_MASK_OFFSET + cpu_mask_size

@fieldwise_init
struct StackSlot(Copyable, Movable):
    var in_use: Int64
    var mmap_base: Int64
    var tcb_addr: Int
    var child_tid_addr: Int
    var stack_top: Int

    fn __init__(out self):
        self.in_use = 0
        self.mmap_base = 0
        self.tcb_addr = 0
        self.child_tid_addr = 0
        self.stack_top = 0

fn allocate_slot[stack_size: Int = DEFAULT_STACK_SIZE, cpu_mask_size: Int = 128]() -> StackSlot:
    alias sz = slot_size[stack_size, cpu_mask_size]()
    var block_addr = syscall_mmap(
        Int64(0), Int64(sz), Int64(PROT_READ | PROT_WRITE), Int64(MAP_PRIVATE | MAP_ANONYMOUS)
    )
    if block_addr == MAP_FAILED or block_addr < 0:
        return StackSlot()

    var static_tls_addr = Int(block_addr)
    var tcb_addr = static_tls_addr + STATIC_TLS_SIZE
    var dtv_addr = tcb_addr + TCB_SIZE
    var child_tid_addr = dtv_addr + (DTV_ENTRIES * DTV_ENTRY_SIZE)
    alias tls_region = slot_size[stack_size, cpu_mask_size]() - GUARD_SIZE - stack_size
    var guard_addr = tls_region + Int(block_addr)
    var stack_base = guard_addr + GUARD_SIZE
    var stack_top = (stack_base + stack_size) & ~15

    var mprotect_ret = syscall_mprotect(Int64(guard_addr), Int64(GUARD_SIZE), Int64(PROT_NONE))
    if mprotect_ret < 0:
        _ = syscall_munmap(block_addr, Int64(sz))
        return StackSlot()

    var slot = StackSlot()
    slot.mmap_base = block_addr
    slot.tcb_addr = tcb_addr
    slot.child_tid_addr = child_tid_addr
    slot.stack_top = stack_top
    return slot^

fn init_tls_core(parent_fs: Int, static_tls_addr: Int, tcb_addr: Int, dtv_addr: Int, child_tid_addr: Int):
    var parent_static_tls = parent_fs - STATIC_TLS_SIZE
    var i = 0
    while i < STATIC_TLS_SIZE:
        ptr(static_tls_addr + i)[] = ptr(parent_static_tls + i)[]
        i += 8
    i = 0
    while i < TCB_SIZE:
        ptr(tcb_addr + i)[] = ptr(parent_fs + i)[]
        i += 8
    ptr(tcb_addr + TCB_OFFSET_TCB)[] = Int64(tcb_addr)
    ptr(tcb_addr + TCB_OFFSET_SELF)[] = Int64(tcb_addr)
    var dtv_ptr_for_tcb = dtv_addr + DTV_ENTRY_SIZE
    ptr(dtv_addr)[] = Int64(DTV_ENTRIES - 2)
    ptr(dtv_addr + DTV_ENTRY_SIZE)[] = Int64(0)
    ptr(dtv_addr + 2 * DTV_ENTRY_SIZE)[] = Int64(static_tls_addr)
    ptr(tcb_addr + TCB_OFFSET_DTV)[] = Int64(dtv_ptr_for_tcb)
    ptr(child_tid_addr)[] = Int64(1)


fn init_slot_tls_child(mmap_base: Int):
    var parent_fs = Int(ptr(mmap_base + INIT_PARENT_FS_OFFSET)[])
    var tcb_addr = mmap_base + STATIC_TLS_SIZE
    var dtv_addr = tcb_addr + TCB_SIZE
    var child_tid_addr = dtv_addr + (DTV_ENTRIES * DTV_ENTRY_SIZE)
    init_tls_core(parent_fs, mmap_base, tcb_addr, dtv_addr, child_tid_addr)