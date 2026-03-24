"""io_uring ring management: setup, submission, completion, teardown.

Generic over operation type via the IoOp trait. Consumers define
concrete op types (ReadOp, WriteOp) that conform to IoOp and describe
how to fill an SQE. The ring handles lifecycle and scheduling.

Error types use trait-based conformance: each failure mode is a concrete
struct conforming to IoRingError and optional sub-traits (Fatal, Retryable,
ShortTransfer, SystemError). Consumers handle errors via typed raises
and compile-time conformance checks.
"""

from std.collections import Dict
from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import size_of
import linux.sys as linux


# =============================================================================
# File mode traits
# =============================================================================


trait FileMode:
    comptime OPEN_FLAGS: Int
    comptime CREATE_MODE: Int

trait IORead(FileMode): ...
trait IOWrite(FileMode): ...
trait IOReadWrite(IORead, IOWrite): ...
trait IOAppend(IOWrite): ...

struct ReadMode(IORead):
    comptime OPEN_FLAGS = linux.OpenFlags.RDONLY | linux.OpenFlags.CLOEXEC
    comptime CREATE_MODE = 0

struct WriteMode(IOWrite):
    comptime OPEN_FLAGS = linux.OpenFlags.WRONLY | linux.OpenFlags.CREAT | linux.OpenFlags.TRUNC | linux.OpenFlags.CLOEXEC
    comptime CREATE_MODE = 0o644

struct ReadWriteMode(IOReadWrite):
    comptime OPEN_FLAGS = linux.OpenFlags.RDWR | linux.OpenFlags.CREAT | linux.OpenFlags.CLOEXEC
    comptime CREATE_MODE = 0o644

struct AppendMode(IOAppend):
    comptime OPEN_FLAGS = linux.OpenFlags.WRONLY | linux.OpenFlags.CREAT | linux.OpenFlags.APPEND | linux.OpenFlags.CLOEXEC
    comptime CREATE_MODE = 0o644


# =============================================================================
# IoOp trait + concrete ops
# =============================================================================


trait IoOp(TrivialRegisterPassable):
    comptime OPCODE: UInt8
    comptime FLAGS: UInt8

    def sqe_fd(self) -> Int32: ...
    def sqe_offset(self) -> UInt64: ...
    def sqe_addr(self) -> UInt64: ...
    def sqe_len(self) -> UInt32: ...
    def op_id(self) -> Int: ...
    def expected_bytes(self) -> Int: ...


@always_inline
def fill_sqe[Op: IoOp](sqe: UnsafePointer[linux.IoUringSqe, MutAnyOrigin], op: Op):
    sqe[].opcode = Op.OPCODE
    sqe[].flags = Op.FLAGS
    sqe[].fd = op.sqe_fd()
    sqe[].off = op.sqe_offset()
    sqe[].addr = op.sqe_addr()
    sqe[].len = op.sqe_len()
    sqe[].user_data = UInt64(op.op_id())
    sqe[].ioprio = 0
    sqe[].buf_index = 0
    sqe[].personality = 0
    sqe[].splice_fd_in = 0
    sqe[].addr3 = 0
    sqe[].pad = 0
    sqe[].op_flags = 0


@fieldwise_init
struct ReadOp[T: AnyType = UInt8](IoOp, Writable):
    """Read from file into buffer at `dest` for `length` bytes."""
    comptime OPCODE = linux.IoUringOp.READ
    comptime FLAGS = linux.IoUringSqeFlags.FIXED_FILE
    var file_idx: Int
    var offset: Int
    var length: Int
    var dest: UnsafePointer[Self.T, MutAnyOrigin]
    var id: Int

    def sqe_fd(self) -> Int32: return Int32(self.file_idx)
    def sqe_offset(self) -> UInt64: return UInt64(self.offset)
    def sqe_addr(self) -> UInt64: return UInt64(Int(self.dest))
    def sqe_len(self) -> UInt32: return UInt32(self.length)
    def op_id(self) -> Int: return self.id
    def expected_bytes(self) -> Int: return self.length


@fieldwise_init
struct WriteOp[T: AnyType = UInt8](IoOp, Writable):
    """Write `length` bytes from buffer at `src` into file."""
    comptime OPCODE = linux.IoUringOp.WRITE
    comptime FLAGS = linux.IoUringSqeFlags.FIXED_FILE
    var file_idx: Int
    var offset: Int
    var length: Int
    var src: UnsafePointer[Self.T, MutAnyOrigin]
    var id: Int

    def sqe_fd(self) -> Int32: return Int32(self.file_idx)
    def sqe_offset(self) -> UInt64: return UInt64(self.offset)
    def sqe_addr(self) -> UInt64: return UInt64(Int(self.src))
    def sqe_len(self) -> UInt32: return UInt32(self.length)
    def op_id(self) -> Int: return self.id
    def expected_bytes(self) -> Int: return self.length


# =============================================================================
# Completion
# =============================================================================


@fieldwise_init
struct Completion(TrivialRegisterPassable, Writable):
    var id: Int
    var result: Int32


# =============================================================================
# Error traits
# =============================================================================


trait IoRingError(Writable, Copyable, ImplicitlyCopyable):
    def error_message(self) -> String: ...
    def error_op_id(self) -> Int: ...

# Severity
trait Fatal(IoRingError): ...
trait Retryable(IoRingError): ...

# Context
trait ShortTransfer(IoRingError):
    def transfer_expected(self) -> Int: ...
    def transfer_actual(self) -> Int: ...

trait SystemError(IoRingError):
    def sys_errno(self) -> Int: ...


# =============================================================================
# Concrete error types
# =============================================================================


@fieldwise_init
struct RingError(SystemError):
    """A ring operation failed. The context field identifies the operation."""
    var op_id: Int
    var errno: Int
    var context: StaticString

    def error_message(self) -> String:
        return String(self.context) + " failed (errno=" + String(self.errno) + ")"

    def error_op_id(self) -> Int: return self.op_id
    def sys_errno(self) -> Int: return self.errno


# =============================================================================
# Ring queues
# =============================================================================


struct SubmissionQueue(TrivialRegisterPassable):
    var ring: UnsafePointer[UInt8, MutAnyOrigin]
    var ring_size: Int
    var head: UnsafePointer[UInt32, MutAnyOrigin]
    var tail: UnsafePointer[UInt32, MutAnyOrigin]
    var mask: UInt32
    var array: UnsafePointer[UInt32, MutAnyOrigin]
    var entries: UnsafePointer[linux.IoUringSqe, MutAnyOrigin]
    var entries_size: Int

    def __init__(out self):
        self.ring = UnsafePointer[UInt8, MutAnyOrigin]()
        self.ring_size = 0
        self.head = UnsafePointer[UInt32, MutAnyOrigin]()
        self.tail = UnsafePointer[UInt32, MutAnyOrigin]()
        self.mask = 0
        self.array = UnsafePointer[UInt32, MutAnyOrigin]()
        self.entries = UnsafePointer[linux.IoUringSqe, MutAnyOrigin]()
        self.entries_size = 0

    def __bool__(self) -> Bool:
        return self.ring.__bool__()


struct CompletionQueue(TrivialRegisterPassable):
    var ring: UnsafePointer[UInt8, MutAnyOrigin]
    var ring_size: Int
    var head: UnsafePointer[UInt32, MutAnyOrigin]
    var tail: UnsafePointer[UInt32, MutAnyOrigin]
    var mask: UInt32
    var entries: UnsafePointer[linux.IoUringCqe, MutAnyOrigin]

    def __init__(out self):
        self.ring = UnsafePointer[UInt8, MutAnyOrigin]()
        self.ring_size = 0
        self.head = UnsafePointer[UInt32, MutAnyOrigin]()
        self.tail = UnsafePointer[UInt32, MutAnyOrigin]()
        self.mask = 0
        self.entries = UnsafePointer[linux.IoUringCqe, MutAnyOrigin]()

    def __bool__(self) -> Bool:
        return self.ring.__bool__()

    def ready(self) -> Int:
        return Int(self.tail[] - self.head[])


# =============================================================================
# IoRing
# =============================================================================


struct IoRing[queue_depth: Int = 2048](Movable):
    comptime MAX_WAIT_EMPTY_RETRIES = 8

    var ring_fd: Int
    var sq: SubmissionQueue
    var cq: CompletionQueue
    var max_entries: UInt32
    var pending_count: Int
    var file_fds: List[Int32]
    var single_mmap: Bool

    def __init__(out self):
        comptime assert (Self.queue_depth & (Self.queue_depth - 1)) == 0 and Self.queue_depth > 0, "queue_depth must be a power of 2"
        self.ring_fd = -1
        self.sq = SubmissionQueue()
        self.cq = CompletionQueue()
        self.max_entries = UInt32(Self.queue_depth)
        self.pending_count = 0
        self.file_fds = List[Int32]()
        self.single_mmap = False

        var sys = linux.linux_sys()
        var params = linux.IoUringParams()
        var params_ptr = UnsafePointer(to=params)
        var fd = sys.sys_io_uring_setup(self.max_entries, params_ptr)
        if fd < 0:
            return

        self.ring_fd = fd
        params = params_ptr[]
        self._map_rings(params)
        if self.ring_fd >= 0 and self.sq:
            var entries = Int(self.sq.mask) + 1
            if entries > 0:
                self.max_entries = UInt32(entries)

    def _map_rings(mut self, params: linux.IoUringParams):
        var sys = linux.linux_sys()
        self.sq.ring_size = Int(params.sq_off.array) + Int(params.sq_entries) * size_of[UInt32]()
        self.cq.ring_size = Int(params.cq_off.cqes) + Int(params.cq_entries) * size_of[linux.IoUringCqe]()

        self.single_mmap = (params.features & linux.IoUringFeat.SINGLE_MMAP) != 0
        if self.single_mmap:
            if self.cq.ring_size > self.sq.ring_size:
                self.sq.ring_size = self.cq.ring_size
            self.cq.ring_size = self.sq.ring_size

        var sq_addr = sys.sys_mmap[
            prot=linux.Prot.RW, flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
        ](0, self.sq.ring_size, self.ring_fd, linux.IORING_OFF_SQ_RING)
        if sq_addr < 0:
            _ = sys.sys_close(self.ring_fd)
            self.ring_fd = -1
            return

        self.sq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=sq_addr)

        if self.single_mmap:
            self.cq.ring = self.sq.ring
        else:
            var cq_addr = sys.sys_mmap[
                prot=linux.Prot.RW, flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
            ](0, self.cq.ring_size, self.ring_fd, linux.IORING_OFF_CQ_RING)
            if cq_addr < 0:
                _ = sys.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
                _ = sys.sys_close(self.ring_fd)
                self.ring_fd = -1
                return
            self.cq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cq_addr)

        self.sq.entries_size = Int(params.sq_entries) * size_of[linux.IoUringSqe]()
        var sqes_addr = sys.sys_mmap[
            prot=linux.Prot.RW, flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
        ](0, self.sq.entries_size, self.ring_fd, linux.IORING_OFF_SQES)
        if sqes_addr < 0:
            _ = sys.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
            if not self.single_mmap:
                _ = sys.sys_munmap(Int(self.cq.ring), self.cq.ring_size)
            _ = sys.sys_close(self.ring_fd)
            self.ring_fd = -1
            return

        self.sq.entries = UnsafePointer[linux.IoUringSqe, MutAnyOrigin](unsafe_from_address=sqes_addr)
        self.sq.head = (self.sq.ring + Int(params.sq_off.head)).bitcast[UInt32]()
        self.sq.tail = (self.sq.ring + Int(params.sq_off.tail)).bitcast[UInt32]()
        self.sq.mask = (self.sq.ring + Int(params.sq_off.ring_mask)).bitcast[UInt32]()[]
        self.sq.array = (self.sq.ring + Int(params.sq_off.array)).bitcast[UInt32]()
        self.cq.head = (self.cq.ring + Int(params.cq_off.head)).bitcast[UInt32]()
        self.cq.tail = (self.cq.ring + Int(params.cq_off.tail)).bitcast[UInt32]()
        self.cq.mask = (self.cq.ring + Int(params.cq_off.ring_mask)).bitcast[UInt32]()[]
        self.cq.entries = (self.cq.ring + Int(params.cq_off.cqes)).bitcast[linux.IoUringCqe]()

    def __del__(deinit self):
        var sys = linux.linux_sys()
        for fd in self.file_fds:
            if fd >= 0:
                _ = sys.sys_close(Int(fd))
        if self.ring_fd < 0:
            return
        if self.sq.entries:
            _ = sys.sys_munmap(Int(self.sq.entries), self.sq.entries_size)
        if self.cq.ring and not self.single_mmap:
            _ = sys.sys_munmap(Int(self.cq.ring), self.cq.ring_size)
        if self.sq.ring:
            _ = sys.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
        _ = sys.sys_close(self.ring_fd)

    def __bool__(self) -> Bool:
        return self.ring_fd >= 0

    def pending(self) -> Int:
        return self.pending_count

    def register_files[M: FileMode = ReadMode](mut self, paths: List[Path]) raises RingError -> Int:
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "register")

        var count = len(paths)
        if count == 0:
            return 0

        var sys = linux.linux_sys()
        self.file_fds = List[Int32](capacity=count)
        for path in paths:
            var path_str = String(path)
            var fd = sys.sys_openat(linux.AT_FDCWD, path_str, M.OPEN_FLAGS, M.CREATE_MODE)
            if fd < 0:
                for open_fd in self.file_fds:
                    _ = sys.sys_close(Int(open_fd))
                self.file_fds = List[Int32]()
                raise RingError(-1, fd, "open")
            self.file_fds.append(Int32(fd))

        var result = sys.sys_io_uring_register(
            self.ring_fd, linux.IoUringRegisterOp.REGISTER_FILES,
            Int(self.file_fds.unsafe_ptr()), UInt32(count),
        )
        if result < 0:
            for open_fd in self.file_fds:
                _ = sys.sys_close(Int(open_fd))
            self.file_fds = List[Int32]()
            raise RingError(-1, result, "register")

        return count

    def submit_one[Op: IoOp](mut self, op: Op) raises RingError -> Int:
        if self.ring_fd < 0:
            raise RingError(op.op_id(), self.ring_fd, "submit")

        var ring_entries = Int(self.sq.mask) + 1
        if ring_entries <= 0:
            raise RingError(op.op_id(), -1, "submit")

        var tail = self.sq.tail[]
        var head = self.sq.head[]
        if Int(tail - head) >= ring_entries:
            return 0

        var idx = tail & self.sq.mask
        fill_sqe[Op](self.sq.entries + Int(idx), op)
        self.sq.array[Int(idx)] = idx
        self.sq.tail[] = tail + 1

        var sys = linux.linux_sys()
        var result = sys.sys_io_uring_enter(self.ring_fd, 1, 0, 0)
        if result < 0:
            self.sq.tail[] = tail
            raise RingError(op.op_id(), result, "submit")
        if result != 1:
            raise RingError(op.op_id(), -1, "submit")

        self.pending_count += 1
        return 1

    def wait(mut self, min_complete: Int = 1) raises RingError -> List[Completion]:
        var completions = List[Completion]()
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "wait")

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        if head == tail and min_complete > 0:
            var sys = linux.linux_sys()
            for _ in range(Self.MAX_WAIT_EMPTY_RETRIES):
                var result = sys.sys_io_uring_enter(
                    self.ring_fd, 0, UInt32(min_complete), linux.IoUringEnter.GETEVENTS,
                )
                if result < 0:
                    raise RingError(-1, result, "wait")
                tail = self.cq.tail[]
                if head != tail:
                    break

        while head != tail:
            var idx = head & self.cq.mask
            var cqe = self.cq.entries[Int(idx)]
            completions.append(Completion(Int(cqe.user_data), cqe.res))
            head += 1
            self.pending_count -= 1

        self.cq.head[] = head
        return completions^

    def poll(mut self) -> List[Completion]:
        var completions = List[Completion]()
        if self.ring_fd < 0:
            return completions^

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        while head != tail:
            var idx = head & self.cq.mask
            var cqe = self.cq.entries[Int(idx)]
            completions.append(Completion(Int(cqe.user_data), cqe.res))
            head += 1
            self.pending_count -= 1

        self.cq.head[] = head
        return completions^
