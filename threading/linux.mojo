from sys.intrinsics import inlined_assembly
from memory import UnsafePointer

comptime KernelPtr = UInt64
comptime KernelFlags = UInt64
comptime KernelFlags32 = UInt32

@register_passable("trivial")
struct Syscall:
    comptime mmap = 9
    comptime munmap = 11
    comptime mprotect = 10
    comptime mbind = 237
    comptime move_pages = 279
    comptime madvise = 28
    comptime clone3 = 435
    comptime exit = 60
    comptime futex_waitv = 449
    comptime futex_wake = 454
    comptime futex_wait = 455
    comptime sched_setaffinity = 203
    comptime rseq = 334
    comptime gettid = 186

@register_passable("trivial")
struct CloneFlags:
    comptime VM = 0x00000100
    comptime FS = 0x00000200
    comptime FILES = 0x00000400
    comptime SIGHAND = 0x00000800
    comptime PIDFD = 0x00001000
    comptime PTRACE = 0x00002000
    comptime VFORK = 0x00004000
    comptime PARENT = 0x00008000
    comptime THREAD = 0x00010000
    comptime NEWNS = 0x00020000
    comptime SYSVSEM = 0x00040000
    comptime SETTLS = 0x00080000
    comptime PARENT_SETTID = 0x00100000
    comptime CHILD_CLEARTID = 0x00200000
    comptime DETACHED = 0x00400000
    comptime UNTRACED = 0x00800000
    comptime CHILD_SETTID = 0x01000000
    comptime NEWCGROUP = 0x02000000
    comptime NEWUTS = 0x04000000
    comptime NEWIPC = 0x08000000
    comptime NEWUSER = 0x10000000
    comptime NEWPID = 0x20000000
    comptime NEWNET = 0x40000000
    comptime IO = 0x80000000
    comptime THREAD_FLAGS = (
        Self.VM | Self.FS | Self.FILES | Self.SIGHAND |
        Self.THREAD | Self.SYSVSEM | Self.SETTLS |
        Self.PARENT_SETTID | Self.CHILD_CLEARTID
    )

@register_passable("trivial")
struct Futex2:
    comptime SIZE_U8 = 0x00
    comptime SIZE_U16 = 0x01
    comptime SIZE_U32 = 0x02
    comptime SIZE_U64 = 0x03
    comptime NUMA = 0x04
    comptime PRIVATE = 0x80

@register_passable("trivial")
struct FutexNuma32:
    var value: Int32
    var node: Int32

    fn __init__(out self, node: Int32 = -1):
        self.value = 0
        self.node = node

@fieldwise_init
@register_passable("trivial")
struct FutexWaitv:
    var val: UInt64
    var uaddr: KernelPtr
    var flags: KernelFlags32
    var __reserved: UInt32

@register_passable("trivial")
struct Clone3Args:
    var flags: KernelFlags
    var pidfd: KernelPtr
    var child_tid: KernelPtr
    var parent_tid: KernelPtr
    var exit_signal: UInt64
    var stack: KernelPtr
    var stack_size: UInt64
    var tls: KernelPtr
    var set_tid: KernelPtr
    var set_tid_size: UInt64
    var cgroup: UInt64

    fn __init__(out self):
        self.flags = 0
        self.pidfd = 0
        self.child_tid = 0
        self.parent_tid = 0
        self.exit_signal = 0
        self.stack = 0
        self.stack_size = 0
        self.tls = 0
        self.set_tid = 0
        self.set_tid_size = 0
        self.cgroup = 0

    @staticmethod
    fn thread(stack: Int, stack_size: Int, tls: Int, child_tid_addr: Int) -> Self:
        var args = Self()
        args.flags = CloneFlags.THREAD_FLAGS
        args.stack = UInt64(stack)
        args.stack_size = UInt64(stack_size)
        args.tls = UInt64(tls)
        args.child_tid = UInt64(child_tid_addr)
        args.parent_tid = UInt64(child_tid_addr)
        return args

@register_passable("trivial")
struct Rseq:
    var cpu_id_start: UInt32
    var cpu_id: UInt32
    var rseq_cs: KernelPtr
    var flags: KernelFlags32
    var node_id: UInt32
    var mm_cid: UInt32
    var padding: UInt32

    fn __init__(out self):
        self.cpu_id_start = 0
        self.cpu_id = 0
        self.rseq_cs = 0
        self.flags = 0
        self.node_id = 0
        self.mm_cid = 0
        self.padding = 0

comptime RSEQ_SIG = 0x53053053

@register_passable("trivial")
struct Prot:
    comptime NONE = 0x0
    comptime READ = 0x1
    comptime WRITE = 0x2
    comptime EXEC = 0x4
    comptime RW = Self.READ | Self.WRITE
    comptime RWX = Self.READ | Self.WRITE | Self.EXEC

@register_passable("trivial")
struct MapFlag:
    comptime SHARED = 0x01
    comptime PRIVATE = 0x02
    comptime FIXED = 0x10
    comptime ANONYMOUS = 0x20
    comptime NORESERVE = 0x4000
    comptime POPULATE = 0x8000
    comptime HUGETLB = 0x40000
    comptime HUGE_2MB = 21 << 26
    comptime HUGE_1GB = 30 << 26

@register_passable("trivial")
struct Mempolicy:
    comptime DEFAULT = 0
    comptime PREFERRED = 1
    comptime BIND = 2
    comptime INTERLEAVE = 3
    comptime LOCAL = 4

@register_passable("trivial")
struct Madvise:
    comptime NORMAL = 0
    comptime RANDOM = 1
    comptime SEQUENTIAL = 2
    comptime WILLNEED = 3
    comptime DONTNEED = 4
    comptime HUGEPAGE = 14
    comptime NOHUGEPAGE = 15

@register_passable("trivial")
struct PageSize:
    comptime STANDARD = 4096
    comptime THP_2MB = 2 * 1024 * 1024
    comptime EXPLICIT_2MB = -2
    comptime EXPLICIT_1GB = -1

fn syscall[count: Int](nr: Int64, *args: Int64) -> Int:
    comptime regs = ("", ",{rdi}", ",{rdi},{rsi}", ",{rdi},{rsi},{rdx}",
                  ",{rdi},{rsi},{rdx},{rcx}", ",{rdi},{rsi},{rdx},{rcx},{r8}",
                  ",{rdi},{rsi},{rdx},{rcx},{r8},{r9}")
    comptime asm = "mov %rcx, %r10\nsyscall" if count > 3 else "syscall"
    comptime constraints = "={rax},{rax}" + regs[count] + ",~{rcx},~{r10},~{r11},~{memory}"
    @parameter
    if count == 0:
        return Int(inlined_assembly[asm, Int64, Int64, constraints=constraints](nr))
    elif count == 1:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, constraints=constraints](nr, args[0]))
    elif count == 2:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, Int64, constraints=constraints](nr, args[0], args[1]))
    elif count == 3:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, Int64, Int64, constraints=constraints](nr, args[0], args[1], args[2]))
    elif count == 4:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, Int64, Int64, Int64, constraints=constraints](nr, args[0], args[1], args[2], args[3]))
    elif count == 5:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, Int64, Int64, Int64, Int64, constraints=constraints](nr, args[0], args[1], args[2], args[3], args[4]))
    elif count == 6:
        return Int(inlined_assembly[asm, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, constraints=constraints](nr, args[0], args[1], args[2], args[3], args[4], args[5]))
    else:
        constrained[False, "syscall supports 0-6 arguments"]()
        return 0

fn sys_mmap[
    prot: Int = Prot.RW,
    flags: Int = MapFlag.PRIVATE | MapFlag.ANONYMOUS,
](addr: Int, length: Int, fd: Int = -1, offset: Int = 0) -> Int:
    """Memory map with compile-time protection and flags.

    Parameters:
        prot: Protection flags
        flags: Mapping flags

    Args:
        addr: Hint address (0 for kernel choice).
        length: Size in bytes.
        fd: File descriptor (-1 for anonymous).
        offset: File offset.

    Returns:
        Mapped address, or negative errno on failure.
    """
    return syscall[6](Syscall.mmap, Int64(addr), Int64(length),
        Int64(prot), Int64(flags), Int64(fd), Int64(offset))

fn sys_munmap(addr: Int, length: Int) -> Int:
    """Unmap memory region. Returns 0 on success, negative errno on failure."""
    return syscall[2](Syscall.munmap, Int64(addr), Int64(length))

fn sys_mbind[
    policy: Int = Mempolicy.BIND,
    flags: Int = 0,
](addr: Int, length: Int, nodemask: UInt64, maxnode: Int = 64) -> Int:
    """Bind memory range to NUMA node(s).

    Parameters:
        policy: Memory policy
        flags: Optional flags.

    Args:
        addr: Start address of memory range.
        length: Length in bytes.
        nodemask: Bitmask of allowed nodes (bit N = node N).
        maxnode: Maximum node number (typically 64).

    Returns:
        0 on success, negative errno on failure.
    """
    var mask_storage = InlineArray[UInt64, 1](nodemask)
    var mask_ptr = UnsafePointer(to=mask_storage)
    var result = syscall[6](Syscall.mbind, Int64(addr), Int64(length),
        Int64(policy), Int64(Int(mask_ptr)), Int64(maxnode), Int64(flags))
    _ = mask_ptr[]
    return result

fn sys_madvise[advice: Int](addr: Int, length: Int) -> Int:
    """Advise kernel about memory usage patterns.

    Parameters:
        advice: Advice type

    Returns:
        0 on success, negative errno on failure.
    """
    return syscall[3](Syscall.madvise, Int64(addr), Int64(length), Int64(advice))

fn sys_move_pages_query(addr: Int) -> Int:
    """Query which NUMA node a page resides on.

    Args:
        addr: Address within the page to query.

    Returns:
        Node ID (>= 0), or negative errno on failure.
        Note: Returns -ENOENT if page not yet faulted.
    """
    var pages = InlineArray[Int64, 1](Int64(addr))
    var status = InlineArray[Int32, 1](Int32(-1))
    var pages_ptr = UnsafePointer(to=pages)
    var status_ptr = UnsafePointer(to=status)
    var result = syscall[6](Syscall.move_pages, 0, 1,
        Int64(Int(pages_ptr)), 0, Int64(Int(status_ptr)), 0)
    _ = pages_ptr[]
    _ = status_ptr[]
    if result < 0:
        return result
    return Int(status[0])

fn sys_mprotect(addr: Int, length: Int, prot: Int) -> Int:
    """Change protection on a region of memory."""
    return syscall[3](Syscall.mprotect, Int64(addr), Int64(length), Int64(prot))

# Lower 32 bits
comptime FUTEX_BITSET_MATCH_ANY: Int64 = 0xFFFFFFFF

fn sys_futex_wait(addr: Int, expected: Int, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
    """Wait on a futex. Returns 0 on wake, negative errno on failure."""
    return syscall[6](Syscall.futex_wait, Int64(addr), Int64(expected), FUTEX_BITSET_MATCH_ANY, Int64(flags), 0, 0)

fn sys_futex_waitv(waiters: UnsafePointer[FutexWaitv], nr_futexes: Int, flags: Int = 0, timeout: Int = 0, clockid: Int = 0) -> Int:
    """Wait on multiple futexes. Returns index of woken futex, or negative errno.
    timeout=0 means NULL (wait indefinitely). clockid: 0=CLOCK_REALTIME, 1=CLOCK_MONOTONIC."""
    return syscall[5](Syscall.futex_waitv, Int64(Int(waiters)), Int64(nr_futexes), Int64(flags), Int64(timeout), Int64(clockid))

fn sys_futex_wake(addr: Int, nr_wake: Int = 1, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
    """Wake waiters on a futex. Returns number of waiters woken, or negative errno."""
    return syscall[4](Syscall.futex_wake, Int64(addr), FUTEX_BITSET_MATCH_ANY, Int64(nr_wake), Int64(flags))

fn sys_exit(code: Int = 0):
    """Exit current thread."""
    _ = syscall[1](Syscall.exit, Int64(code))

fn sys_gettid() -> Int:
    """Get thread ID."""
    return syscall[0](Syscall.gettid)

fn sys_rseq(rseq_ptr: Int, len: Int, flags: Int, sig: Int) -> Int:
    """Register rseq for current thread."""
    return syscall[4](Syscall.rseq, Int64(rseq_ptr), Int64(len), Int64(flags), Int64(sig))

fn sys_sched_setaffinity(tid: Int, mask_size: Int, mask_ptr: Int) -> Int:
    """Set CPU affinity for thread. tid=0 -> current thread."""
    return syscall[3](Syscall.sched_setaffinity, Int64(tid), Int64(mask_size), Int64(mask_ptr))
