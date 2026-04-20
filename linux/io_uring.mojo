"""io_uring ring management: setup, submission, completion, teardown.

Generic over operation type via the IoOp trait. Consumers define
concrete op types (ReadOp, WriteOp) that conform to IoOp and describe
how to fill an SQE. The ring handles lifecycle and scheduling.

Error types use trait-based conformance: each failure mode is a concrete
struct conforming to IoRingError and optional sub-traits (Fatal, Retryable,
ShortTransfer, SystemError). Consumers handle errors via typed raises
and compile-time conformance checks.
"""

from std.collections import InlineArray
from std.memory import UnsafePointer, Span
from std.pathlib import Path
from std.sys.info import size_of
import linux.sys as linux

from threading.burst_threading import BurstPool


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
        self.map_rings(params)
        if self.ring_fd >= 0 and self.sq:
            var entries = Int(self.sq.mask) + 1
            if entries > 0:
                self.max_entries = UInt32(entries)

    def map_rings(mut self, params: linux.IoUringParams):
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

    def register_fds(
        mut self, fds: Span[Int32, _],
    ) raises RingError -> Int:
        """Register pre-opened fds with this ring's fixed-file table.
        Does not take ownership — caller keeps the fds alive for the ring's
        lifetime and is responsible for closing them."""
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "register")

        var count = len(fds)
        if count == 0:
            return 0

        var sys = linux.linux_sys()
        var result = sys.sys_io_uring_register(
            self.ring_fd, linux.IoUringRegisterOp.REGISTER_FILES,
            Int(fds.unsafe_ptr()), UInt32(count),
        )
        if result < 0:
            raise RingError(-1, result, "register")

        return count

    def register_files[M: FileMode = ReadMode](
        mut self, paths: Span[Path, _],
    ) raises RingError -> Int:
        """Convenience: open the paths and register the resulting fds with
        this ring. The ring takes ownership of the opened fds (closes them
        on ring destruction). For multi-ring use cases prefer opening once
        via open_files_for_ring and calling register_fds per ring."""
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "register")

        var count = len(paths)
        if count == 0:
            return 0

        var sys = linux.linux_sys()
        self.file_fds = List[Int32](capacity=count)
        for i in range(count):
            var path_str = String(paths[i])
            var fd = sys.sys_openat(linux.AT_FDCWD, path_str, M.OPEN_FLAGS, M.CREATE_MODE)
            if fd < 0:
                for open_fd in self.file_fds:
                    _ = sys.sys_close(Int(open_fd))
                self.file_fds = List[Int32]()
                raise RingError(-1, fd, "open")
            self.file_fds.append(Int32(fd))

        var fds_span = Span[Int32, MutAnyOrigin](
            ptr=UnsafePointer[Int32, MutAnyOrigin](
                unsafe_from_address=Int(self.file_fds.unsafe_ptr())),
            length=len(self.file_fds),
        )
        try:
            _ = self.register_fds(fds_span)
        except err:
            for open_fd in self.file_fds:
                _ = sys.sys_close(Int(open_fd))
            self.file_fds = List[Int32]()
            raise err

        return count

    def sq_free(self) -> Int:
        if self.ring_fd < 0:
            return 0
        var ring_entries = Int(self.sq.mask) + 1
        return ring_entries - Int(self.sq.tail[] - self.sq.head[])

    def submit_many[Op: IoOp](
        mut self, ops: Span[Op, _], start: Int,
    ) raises RingError -> Int:
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "submit")

        var ring_entries = Int(self.sq.mask) + 1
        if ring_entries <= 0:
            raise RingError(-1, -1, "submit")

        var tail = self.sq.tail[]
        var head = self.sq.head[]
        var free_slots = ring_entries - Int(tail - head)
        var avail = len(ops) - start
        if free_slots <= 0 or avail <= 0:
            return 0
        var n = min(free_slots, avail)

        for i in range(n):
            var idx = (tail + UInt32(i)) & self.sq.mask
            fill_sqe[Op](self.sq.entries + Int(idx), ops[start + i])
            self.sq.array[Int(idx)] = idx
        self.sq.tail[] = tail + UInt32(n)

        var sys = linux.linux_sys()
        var submitted = 0
        while submitted < n:
            var remaining = n - submitted
            var result = sys.sys_io_uring_enter(
                self.ring_fd, UInt32(remaining), 0, 0)
            if result < 0:
                raise RingError(-1, result, "submit")
            if result == 0:
                raise RingError(-1, -1, "submit")
            submitted += Int(result)

        self.pending_count += n
        return n

    def submit_one[Op: IoOp](mut self, op: Op) raises RingError -> Int:
        var ops = InlineArray[Op, 1](uninitialized=True)
        ops[0] = op
        var span = Span[Op, MutAnyOrigin](
            ptr=UnsafePointer[Op, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=ops[0]))),
            length=1,
        )
        return self.submit_many[Op](span, 0)

    def drain[
        visitor: def(Completion) capturing -> None,
    ](mut self, min_complete: Int = 0) raises RingError -> Int:
        """Walk the CQ from head to tail, invoking visitor per CQE. Allocates
        nothing. If min_complete > 0 and the CQ is empty, blocks via
        io_uring_enter until at least that many are available. Returns the
        number of CQEs processed."""
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

        var count = 0
        while head != tail:
            var idx = head & self.cq.mask
            var cqe = self.cq.entries[Int(idx)]
            visitor(Completion(Int(cqe.user_data), cqe.res))
            head += 1
            self.pending_count -= 1
            count += 1

        self.cq.head[] = head
        return count

    def drain_one(mut self) raises RingError -> Completion:
        """Block until one CQE is available, return it on the stack."""
        if self.ring_fd < 0:
            raise RingError(-1, self.ring_fd, "wait")

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        if head == tail:
            var sys = linux.linux_sys()
            for _ in range(Self.MAX_WAIT_EMPTY_RETRIES):
                var result = sys.sys_io_uring_enter(
                    self.ring_fd, 0, 1, linux.IoUringEnter.GETEVENTS,
                )
                if result < 0:
                    raise RingError(-1, result, "wait")
                tail = self.cq.tail[]
                if head != tail:
                    break

        if head == tail:
            raise RingError(-1, -1, "wait")

        var idx = head & self.cq.mask
        var cqe = self.cq.entries[Int(idx)]
        var c = Completion(Int(cqe.user_data), cqe.res)
        self.cq.head[] = head + 1
        self.pending_count -= 1
        return c


# =============================================================================
# File opening for registered-file tables
# =============================================================================


def open_files_for_ring[M: FileMode = ReadMode](
    paths: Span[Path, _],
) raises RingError -> List[Int32]:
    """Open paths once on the calling thread, return fds. Caller owns
    the fds and is responsible for closing them (e.g. after all rings
    that registered them are destroyed)."""
    var sys = linux.linux_sys()
    var count = len(paths)
    var fds = List[Int32](capacity=count)
    for i in range(count):
        var path_str = String(paths[i])
        var fd = sys.sys_openat(linux.AT_FDCWD, path_str, M.OPEN_FLAGS, M.CREATE_MODE)
        if fd < 0:
            for open_fd in fds:
                _ = sys.sys_close(Int(open_fd))
            raise RingError(-1, fd, "open")
        fds.append(Int32(fd))
    return fds^


# =============================================================================
# Read-completion queue processing (module-private)
# =============================================================================


struct LoadError(Copyable, Writable):
    var msg: String

    def __init__(out self, msg: String):
        self.msg = msg

    @staticmethod
    def from_ring[E: IoRingError](err: E) -> Self:
        return Self("io_uring: op " + String(err.error_op_id()) + ": " + err.error_message())


def process_read_queue[
    on_complete: def(Completion) capturing -> None,
](
    mut ring: IoRing,
    ops: Span[ReadOp[], _],
) -> Optional[LoadError]:
    """Submit reads in batches filling the SQ to capacity, drain completions
    inline as they arrive. Requires op ids to be local indices in [0, total),
    so each CQE validates against ops[c.id] in O(1). The submit-drain loop
    allocates nothing."""
    var total = len(ops)
    if total == 0:
        return None

    if not ring:
        return LoadError("ring not initialized")

    for i in range(total):
        if ops[i].op_id() != i:
            return LoadError("op id must equal its local index; got id=" +
                String(ops[i].op_id()) + " at position " + String(i))
        if ops[i].expected_bytes() <= 0:
            return LoadError("invalid op length at position " + String(i))

    var err_msg = String("")

    @parameter
    def visit(c: Completion):
        if err_msg.byte_length() != 0:
            return
        var id = c.id
        if id < 0 or id >= total:
            err_msg = "unknown completion id: " + String(id)
            return
        if c.result < 0:
            err_msg = "CQE negative: op " + String(id) +
                " errno=" + String(Int(c.result))
            return
        var expected = ops[id].expected_bytes()
        var got = Int(c.result)
        if got != expected:
            err_msg = "short read: op " + String(id) +
                " got " + String(got) + "/" + String(expected)
            return
        on_complete(c)

    var submitted = 0
    var completed = 0

    while completed < total:
        if submitted < total and ring.sq_free() > 0:
            try:
                submitted += ring.submit_many[ReadOp[]](ops, submitted)
            except err:
                return LoadError.from_ring(err)

        var must_block = submitted == total or ring.sq_free() == 0
        try:
            completed += ring.drain[visit](1 if must_block else 0)
        except err:
            return LoadError.from_ring(err)

        if err_msg.byte_length() != 0:
            return LoadError(err_msg^)

    return None


# =============================================================================
# Multi-worker read dispatch
# =============================================================================


def close_fds(var fds: List[Int32]):
    """Close each fd. Takes ownership so callers can't reuse stale fds."""
    var sys = linux.linux_sys()
    for fd in fds:
        if fd >= 0:
            _ = sys.sys_close(Int(fd))


@fieldwise_init
struct LoadShardArgs(Copyable, ImplicitlyCopyable):
    """Private: carried across the BurstPool mailbox to a load worker.
    Both fields are Spans — POD under the hood, safe to bytewise-copy
    through the mailbox. Backing storage is owned by the caller of
    dispatch_reads and must live until that caller joins the pool."""
    var fds: Span[Int32, MutAnyOrigin]
    var ops: Span[ReadOp[], MutAnyOrigin]


def load_worker_kernel[queue_depth: Int](args: LoadShardArgs):
    """Worker body: construct a ring, register fds, fill-drain ops.
    Hard-aborts the process on any failure."""
    var sys = linux.linux_sys()

    var ring = IoRing[queue_depth]()
    if not ring:
        print("io_uring setup failed")
        sys.sys_exit_group(1)
        return

    try:
        _ = ring.register_fds(args.fds)
    except err:
        print("register_fds failed:", err.error_message())
        sys.sys_exit_group(1)
        return

    @parameter
    def on_complete(c: Completion):
        pass

    var err = process_read_queue[on_complete](ring, args.ops)
    if err:
        print("load error:", err.value().msg)
        sys.sys_exit_group(1)
        return


def make_load_args(
    fds: Span[Int32, _],
    ops: Span[ReadOp[], _],
) -> LoadShardArgs:
    var fds_wild = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(fds.unsafe_ptr()))
    var ops_wild = UnsafePointer[ReadOp[], MutAnyOrigin](
        unsafe_from_address=Int(ops.unsafe_ptr()))
    return LoadShardArgs(
        fds=Span[Int32, MutAnyOrigin](ptr=fds_wild, length=len(fds)),
        ops=Span[ReadOp[], MutAnyOrigin](ptr=ops_wild, length=len(ops)),
    )


def dispatch_reads[queue_depth: Int, mask_size: Int](
    mut pool: BurstPool[mask_size],
    mut args: LoadShardArgs,
):
    """Dispatch one worker on `pool` with the prepared args. The args
    backing storage must live until the caller joins `pool`."""
    pool.dispatch[LoadShardArgs, load_worker_kernel[queue_depth]](
        UnsafePointer(to=args), 1)


def run_reads_multi[queue_depth: Int, mask_size: Int](
    mut pools: Span[BurstPool[mask_size], MutAnyOrigin],
    paths: List[Path],
    ops_per_rank: List[List[ReadOp[]]],
):
    """Open the paths once on the calling thread, dispatch one worker per
    pool (each worker drives its own ring over its rank's ops slice), join
    all pools, close fds. Aborts on any failure."""
    var sys = linux.linux_sys()
    var n = len(pools)
    debug_assert(n == len(ops_per_rank),
        "pools and ops_per_rank must have matching length")

    var fds: List[Int32]
    try:
        fds = open_files_for_ring(Span(paths))
    except err:
        print("open_files_for_ring failed:", err.error_message())
        sys.sys_exit_group(1)
        return

    var args_list = List[LoadShardArgs](capacity=n)
    for r in range(n):
        args_list.append(make_load_args(Span(fds), Span(ops_per_rank[r])))

    var pool_base = pools.unsafe_ptr()
    for r in range(n):
        dispatch_reads[queue_depth, mask_size](
            (pool_base + r)[], args_list[r])
    for r in range(n):
        (pool_base + r)[].join()

    # Keep backing storage alive until after every worker has joined,
    # regardless of ASAP destruction's opinion about last-use points.
    _ = args_list^
    _ = ops_per_rank
    close_fds(fds^)
