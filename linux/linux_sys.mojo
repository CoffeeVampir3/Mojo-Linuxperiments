from memory import UnsafePointer

# =============================================================================
# Linux syscall interface specification.
#
# This module defines:
# - Linux kernel UAPI structs/constants used by syscalls in this project.
# - A `LinuxSys` interface that architecture-specific implementations must
#   implement end-to-end (no shared default method bodies).
#
# The UAPI types here intentionally model the kernel syscall ABI (not libc).
# For 64-bit Linux targets (x86-64 and arm64), these layouts are compatible.
# =============================================================================

comptime KernelPtr = UInt64
comptime KernelFlags = UInt64
comptime KernelFlags32 = UInt32

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
struct Signal:
    comptime ABRT = 6
    comptime SEGV = 11

@register_passable("trivial")
struct SigActionFlag:
    # From asm-generic/signal-defs.h (Linux uapi)
    comptime SIGINFO = 0x00000004
    comptime ONSTACK = 0x08000000
    comptime RESTART = 0x10000000
    comptime NODEFER = 0x40000000
    comptime RESETHAND = 0x80000000

@register_passable("trivial")
struct SigSet64:
    # Portable 64-signal mask used by rt_sigaction on 64-bit Linux.
    var bits0: UInt64

    fn __init__(out self):
        self.bits0 = 0

@register_passable("trivial")
struct RtSigAction:
    # Portable rt_sigaction inputs/outputs.
    # Implementations may translate to the kernel's per-arch sigaction ABI.
    var handler: Int
    var flags: UInt64
    var mask: SigSet64

    fn __init__(out self):
        self.handler = 0
        self.flags = 0
        self.mask = SigSet64()

@register_passable("trivial")
struct SigSegvContext:
    # Portable subset of SIGSEGV context information.
    # Implementations decode per-arch kernel signal frame layouts.
    var ip: UInt64
    var sp: UInt64
    var fault_addr: UInt64

    fn __init__(out self):
        self.ip = 0
        self.sp = 0
        self.fault_addr = 0

@register_passable("trivial")
struct StackT:
    # stack_t: void *ss_sp; int ss_flags; size_t ss_size;
    var ss_sp: Int
    var ss_flags: Int32
    var pad: Int32
    var ss_size: UInt64

    fn __init__(out self):
        self.ss_sp = 0
        self.ss_flags = 0
        self.pad = 0
        self.ss_size = 0

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
    var reserved: UInt32

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

# =============================================================================
# io_uring types and constants
# Reference: https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h
# =============================================================================

@register_passable("trivial")
struct IoUringSetup:
    comptime IOPOLL = 1 << 0
    comptime SQPOLL = 1 << 1
    comptime SQ_AFF = 1 << 2
    comptime CQSIZE = 1 << 3
    comptime CLAMP = 1 << 4
    comptime ATTACH_WQ = 1 << 5
    comptime R_DISABLED = 1 << 6
    comptime SUBMIT_ALL = 1 << 7
    comptime COOP_TASKRUN = 1 << 8
    comptime TASKRUN_FLAG = 1 << 9
    comptime SQE128 = 1 << 10
    comptime CQE32 = 1 << 11
    comptime SINGLE_ISSUER = 1 << 12
    comptime DEFER_TASKRUN = 1 << 13
    comptime NO_MMAP = 1 << 14
    comptime REGISTERED_FD_ONLY = 1 << 15
    comptime NO_SQARRAY = 1 << 16

@register_passable("trivial")
struct IoUringEnter:
    comptime GETEVENTS = 1 << 0
    comptime SQ_WAKEUP = 1 << 1
    comptime SQ_WAIT = 1 << 2
    comptime EXT_ARG = 1 << 3
    comptime REGISTERED_RING = 1 << 4

@register_passable("trivial")
struct IoUringSqeFlags:
    comptime FIXED_FILE = 1 << 0
    comptime IO_DRAIN = 1 << 1
    comptime IO_LINK = 1 << 2
    comptime IO_HARDLINK = 1 << 3
    comptime ASYNC = 1 << 4
    comptime BUFFER_SELECT = 1 << 5
    comptime CQE_SKIP_SUCCESS = 1 << 6

@register_passable("trivial")
struct IoUringOp:
    comptime NOP = 0
    comptime READV = 1
    comptime WRITEV = 2
    comptime FSYNC = 3
    comptime READ_FIXED = 4
    comptime WRITE_FIXED = 5
    comptime POLL_ADD = 6
    comptime POLL_REMOVE = 7
    comptime SYNC_FILE_RANGE = 8
    comptime SENDMSG = 9
    comptime RECVMSG = 10
    comptime TIMEOUT = 11
    comptime TIMEOUT_REMOVE = 12
    comptime ACCEPT = 13
    comptime ASYNC_CANCEL = 14
    comptime LINK_TIMEOUT = 15
    comptime CONNECT = 16
    comptime FALLOCATE = 17
    comptime OPENAT = 18
    comptime CLOSE = 19
    comptime FILES_UPDATE = 20
    comptime STATX = 21
    comptime READ = 22
    comptime WRITE = 23
    comptime FADVISE = 24
    comptime MADVISE = 25
    comptime SEND = 26
    comptime RECV = 27
    comptime OPENAT2 = 28
    comptime EPOLL_CTL = 29
    comptime SPLICE = 30
    comptime PROVIDE_BUFFERS = 31
    comptime REMOVE_BUFFERS = 32

@register_passable("trivial")
struct IoUringRegisterOp:
    comptime REGISTER_BUFFERS = 0
    comptime UNREGISTER_BUFFERS = 1
    comptime REGISTER_FILES = 2
    comptime UNREGISTER_FILES = 3
    comptime REGISTER_EVENTFD = 4
    comptime UNREGISTER_EVENTFD = 5
    comptime REGISTER_FILES_UPDATE = 6
    comptime REGISTER_EVENTFD_ASYNC = 7
    comptime REGISTER_PROBE = 8
    comptime REGISTER_PERSONALITY = 9
    comptime UNREGISTER_PERSONALITY = 10

@register_passable("trivial")
struct IoUringCqeFlags:
    comptime BUFFER = 1 << 0
    comptime MORE = 1 << 1
    comptime SOCK_NONEMPTY = 1 << 2
    comptime NOTIF = 1 << 3

@register_passable("trivial")
struct SqRingOffsets:
    var head: UInt32
    var tail: UInt32
    var ring_mask: UInt32
    var ring_entries: UInt32
    var flags: UInt32
    var dropped: UInt32
    var array: UInt32
    var resv1: UInt32
    var user_addr: UInt64

    fn __init__(out self):
        self.head = 0
        self.tail = 0
        self.ring_mask = 0
        self.ring_entries = 0
        self.flags = 0
        self.dropped = 0
        self.array = 0
        self.resv1 = 0
        self.user_addr = 0

@register_passable("trivial")
struct CqRingOffsets:
    var head: UInt32
    var tail: UInt32
    var ring_mask: UInt32
    var ring_entries: UInt32
    var overflow: UInt32
    var cqes: UInt32
    var flags: UInt32
    var resv1: UInt32
    var user_addr: UInt64

    fn __init__(out self):
        self.head = 0
        self.tail = 0
        self.ring_mask = 0
        self.ring_entries = 0
        self.overflow = 0
        self.cqes = 0
        self.flags = 0
        self.resv1 = 0
        self.user_addr = 0

@register_passable("trivial")
struct IoUringParams:
    var sq_entries: UInt32
    var cq_entries: UInt32
    var flags: UInt32
    var sq_thread_cpu: UInt32
    var sq_thread_idle: UInt32
    var features: UInt32
    var wq_fd: UInt32
    var resv0: UInt32
    var resv1: UInt32
    var resv2: UInt32
    var sq_off: SqRingOffsets
    var cq_off: CqRingOffsets

    fn __init__(out self, sq_entries: UInt32 = 0, flags: UInt32 = 0):
        self.sq_entries = sq_entries
        self.cq_entries = 0
        self.flags = flags
        self.sq_thread_cpu = 0
        self.sq_thread_idle = 0
        self.features = 0
        self.wq_fd = 0
        self.resv0 = 0
        self.resv1 = 0
        self.resv2 = 0
        self.sq_off = SqRingOffsets()
        self.cq_off = CqRingOffsets()

@register_passable("trivial")
struct IoUringSqe:
    var opcode: UInt8
    var flags: UInt8
    var ioprio: UInt16
    var fd: Int32
    var off: UInt64       # File offset (union with addr2)
    var addr: UInt64      # Buffer address (union with splice_off_in)
    var len: UInt32       # Buffer size or iovec count
    var op_flags: UInt32  # Operation-specific flags (rw_flags, fsync_flags, etc.)
    var user_data: UInt64 # Passed back in CQE
    var buf_index: UInt16 # Index into registered buffers (union with buf_group)
    var personality: UInt16
    var splice_fd_in: Int32  # Union with file_index
    var addr3: UInt64
    var pad: UInt64

    fn __init__(out self):
        self.opcode = 0
        self.flags = 0
        self.ioprio = 0
        self.fd = 0
        self.off = 0
        self.addr = 0
        self.len = 0
        self.op_flags = 0
        self.user_data = 0
        self.buf_index = 0
        self.personality = 0
        self.splice_fd_in = 0
        self.addr3 = 0
        self.pad = 0

    @staticmethod
    fn read(fd: Int32, offset: UInt64, buf: UInt64, size: UInt32, user_data: UInt64) -> Self:
        var sqe = Self()
        sqe.opcode = IoUringOp.READ
        sqe.fd = fd
        sqe.off = offset
        sqe.addr = buf
        sqe.len = size
        sqe.user_data = user_data
        return sqe

    @staticmethod
    fn read_fixed(file_index: Int32, offset: UInt64, buf: UInt64, size: UInt32,
                  buf_index: UInt16, user_data: UInt64) -> Self:
        var sqe = Self()
        sqe.opcode = IoUringOp.READ
        sqe.flags = IoUringSqeFlags.FIXED_FILE
        sqe.fd = file_index
        sqe.off = offset
        sqe.addr = buf
        sqe.len = size
        sqe.buf_index = buf_index
        sqe.user_data = user_data
        return sqe

@register_passable("trivial")
struct IoUringCqe:
    var user_data: UInt64
    var res: Int32
    var flags: UInt32

    fn __init__(out self):
        self.user_data = 0
        self.res = 0
        self.flags = 0

@register_passable("trivial")
struct IoVec:
    var base: UInt64
    var len: UInt64

    fn __init__(out self, base: Int, length: Int):
        self.base = UInt64(base)
        self.len = UInt64(length)

comptime IORING_OFF_SQ_RING: Int = 0
comptime IORING_OFF_CQ_RING: Int = 0x8000000
comptime IORING_OFF_SQES: Int = 0x10000000

comptime AT_FDCWD: Int = -100

@register_passable("trivial")
struct OpenFlags:
    comptime RDONLY = 0
    comptime WRONLY = 1
    comptime RDWR = 2
    comptime CREAT = 0o100
    comptime EXCL = 0o200
    comptime TRUNC = 0o1000
    comptime APPEND = 0o2000
    comptime NONBLOCK = 0o4000
    comptime CLOEXEC = 0o2000000
    comptime DIRECT = 0o40000

@register_passable("trivial")
struct IoUringFeat:
    comptime SINGLE_MMAP = 1 << 0
    comptime NODROP = 1 << 1
    comptime SUBMIT_STABLE = 1 << 2
    comptime RW_CUR_POS = 1 << 3
    comptime CUR_PERSONALITY = 1 << 4
    comptime FAST_POLL = 1 << 5
    comptime POLL_32BITS = 1 << 6
    comptime SQPOLL_NONFIXED = 1 << 7
    comptime EXT_ARG = 1 << 8
    comptime NATIVE_WORKERS = 1 << 9
    comptime RSRC_TAGS = 1 << 10
    comptime CQE_SKIP = 1 << 11
    comptime LINKED_FILE = 1 << 12

# Lower 32 bits
comptime FUTEX_BITSET_MATCH_ANY: Int = 0xFFFFFFFF

trait LinuxSys:
    fn syscall[count: Int](self, nr: Int, *args: Int) -> Int: ...

    fn sys_mmap[
        prot: Int = Prot.RW,
        flags: Int = MapFlag.PRIVATE | MapFlag.ANONYMOUS,
    ](self, addr: Int, length: Int, fd: Int = -1, offset: Int = 0) -> Int: ...

    fn sys_munmap(self, addr: Int, length: Int) -> Int: ...

    fn sys_mbind[
        policy: Int = Mempolicy.BIND,
        flags: Int = 0,
    ](self, addr: Int, length: Int, nodemask: UInt64, maxnode: Int = 64) -> Int: ...

    fn sys_madvise[advice: Int](self, addr: Int, length: Int) -> Int: ...

    fn sys_move_pages_query(self, addr: Int) -> Int: ...

    fn sys_mprotect(self, addr: Int, length: Int, prot: Int) -> Int: ...

    fn sys_write(self, fd: Int, buf: Int, count: Int) -> Int: ...

    fn sys_rt_sigaction(
        self,
        signum: Int,
        act: UnsafePointer[RtSigAction, MutAnyOrigin],
        old: UnsafePointer[RtSigAction, MutAnyOrigin] = UnsafePointer[RtSigAction, MutAnyOrigin](),
    ) -> Int: ...

    fn sys_sigaltstack(
        self,
        ss: UnsafePointer[StackT, MutAnyOrigin],
        old: UnsafePointer[StackT, MutAnyOrigin] = UnsafePointer[StackT, MutAnyOrigin](),
    ) -> Int: ...

    fn sys_futex_wait(self, addr: Int, expected: Int, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int: ...

    fn sys_futex_waitv(
        self,
        waiters: UnsafePointer[FutexWaitv],
        nr_futexes: Int,
        flags: Int = 0,
        timeout: Int = 0,
        clockid: Int = 0,
    ) -> Int: ...

    fn sys_futex_wake(self, addr: Int, nr_wake: Int = 1, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int: ...

    fn sys_exit(self, code: Int = 0): ...

    fn sys_exit_group(self, code: Int = 0): ...

    fn sys_getpid(self) -> Int: ...

    fn sys_gettid(self) -> Int: ...

    fn sys_getcpu(self) -> Tuple[Int, Int]: ...

    fn sys_tgkill(self, pid: Int, tid: Int, sig: Int) -> Int: ...

    fn sys_rseq(self, rseq_ptr: Int, len: Int, flags: Int, sig: Int) -> Int: ...

    fn sys_sched_setaffinity(self, tid: Int, mask_size: Int, mask_ptr: Int) -> Int: ...

    fn sys_openat(self, dirfd: Int, mut pathname: String, flags: Int, mode: Int = 0) -> Int: ...

    fn sys_close(self, fd: Int) -> Int: ...

    fn sys_io_uring_setup(self, entries: UInt32, params: UnsafePointer[IoUringParams]) -> Int: ...

    fn sys_io_uring_enter(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
    ) -> Int: ...

    fn sys_io_uring_enter_sig(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
        sig: Int,
        sigsz: Int,
    ) -> Int: ...

    fn sys_io_uring_register(
        self,
        fd: Int,
        opcode: UInt32,
        arg: Int,
        nr_args: UInt32,
    ) -> Int: ...

    # -------------------------------------------------------------------------
    # Architecture / threading capabilities.
    # These are intentionally defined in the system interface so threading code
    # can be written without embedding arch-specific inline assembly at call
    # sites.
    # -------------------------------------------------------------------------

    fn arch_cpu_relax(self): ...

    fn arch_thread_pointer(self) -> Int: ...

    fn arch_tls_load_i64[offset: Int](self) -> Int: ...

    fn sys_clone3_with_entry(
        self,
        clone_args_ptr: UnsafePointer[Clone3Args, MutAnyOrigin],
        clone_args_size: Int,
    ) -> Int: ...

    fn arch_decode_sigsegv(self, siginfo: Int, ucontext: Int) -> SigSegvContext: ...
