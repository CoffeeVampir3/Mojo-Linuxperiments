from sys.intrinsics import inlined_assembly
from memory import UnsafePointer

alias KernelPtr = UInt64
alias KernelFlags = UInt64
alias KernelFlags32 = UInt32

@register_passable("trivial")
struct Syscall:
    alias mmap = 9
    alias munmap = 11
    alias mprotect = 10
    alias mbind = 237
    alias move_pages = 279
    alias madvise = 28
    alias clone3 = 435
    alias exit = 60
    alias futex_waitv = 449
    alias futex_wake = 454
    alias futex_wait = 455
    alias sched_setaffinity = 203
    alias rseq = 334
    alias gettid = 186

@register_passable("trivial")
struct CloneFlags:
    alias VM = 0x00000100
    alias FS = 0x00000200
    alias FILES = 0x00000400
    alias SIGHAND = 0x00000800
    alias PIDFD = 0x00001000
    alias PTRACE = 0x00002000
    alias VFORK = 0x00004000
    alias PARENT = 0x00008000
    alias THREAD = 0x00010000
    alias NEWNS = 0x00020000
    alias SYSVSEM = 0x00040000
    alias SETTLS = 0x00080000
    alias PARENT_SETTID = 0x00100000
    alias CHILD_CLEARTID = 0x00200000
    alias DETACHED = 0x00400000
    alias UNTRACED = 0x00800000
    alias CHILD_SETTID = 0x01000000
    alias NEWCGROUP = 0x02000000
    alias NEWUTS = 0x04000000
    alias NEWIPC = 0x08000000
    alias NEWUSER = 0x10000000
    alias NEWPID = 0x20000000
    alias NEWNET = 0x40000000
    alias IO = 0x80000000
    alias THREAD_FLAGS = (
        Self.VM | Self.FS | Self.FILES | Self.SIGHAND |
        Self.THREAD | Self.SYSVSEM | Self.SETTLS |
        Self.PARENT_SETTID
    )

@register_passable("trivial")
struct Futex2:
    alias SIZE_U8 = 0x00
    alias SIZE_U16 = 0x01
    alias SIZE_U32 = 0x02
    alias SIZE_U64 = 0x03
    alias NUMA = 0x04
    alias PRIVATE = 0x80

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

alias RSEQ_SIG = 0x53053053

@register_passable("trivial")
struct Prot:
    alias NONE = 0x0
    alias READ = 0x1
    alias WRITE = 0x2
    alias EXEC = 0x4
    alias RW = Self.READ | Self.WRITE
    alias RWX = Self.READ | Self.WRITE | Self.EXEC

@register_passable("trivial")
struct MapFlag:
    alias SHARED = 0x01
    alias PRIVATE = 0x02
    alias FIXED = 0x10
    alias ANONYMOUS = 0x20
    alias NORESERVE = 0x4000
    alias POPULATE = 0x8000
    alias HUGETLB = 0x40000
    alias HUGE_2MB = 21 << 26
    alias HUGE_1GB = 30 << 26

@register_passable("trivial")
struct Mempolicy:
    alias DEFAULT = 0
    alias PREFERRED = 1
    alias BIND = 2
    alias INTERLEAVE = 3
    alias LOCAL = 4

@register_passable("trivial")
struct Madvise:
    alias NORMAL = 0
    alias RANDOM = 1
    alias SEQUENTIAL = 2
    alias WILLNEED = 3
    alias DONTNEED = 4
    alias HUGEPAGE = 14
    alias NOHUGEPAGE = 15

@register_passable("trivial")
struct PageSize:
    alias STANDARD = 4096
    alias THP_2MB = 2 * 1024 * 1024
    alias EXPLICIT_2MB = -2
    alias EXPLICIT_1GB = -1

fn syscall[count: Int](nr: Int64, *args: Int64) -> Int:
    alias regs = ("", ",{rdi}", ",{rdi},{rsi}", ",{rdi},{rsi},{rdx}",
                  ",{rdi},{rsi},{rdx},{rcx}", ",{rdi},{rsi},{rdx},{rcx},{r8}",
                  ",{rdi},{rsi},{rdx},{rcx},{r8},{r9}")
    alias asm = "mov %rcx, %r10\nsyscall" if count > 3 else "syscall"
    alias constraints = "={rax},{rax}" + regs[count] + ",~{rcx},~{r10},~{r11},~{memory}"
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
alias FUTEX_BITSET_MATCH_ANY: Int64 = 0xFFFFFFFF

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
