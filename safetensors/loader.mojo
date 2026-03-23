import linux.sys as linux
from std.collections import Dict
from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import size_of

struct ReadOp(TrivialRegisterPassable, Writable):
    """A single read operation: file region → buffer."""
    var file_idx: Int32
    var offset: Int
    var length: Int32
    var dest: Int
    var id: Int  # Maps to io_uring SQE user_data

    def __init__(out self, file_idx: Int, offset: Int, length: Int, dest: Int, id: Int = 0):
        self.file_idx = Int32(file_idx)
        self.offset = offset
        self.length = Int32(length)
        self.dest = dest
        self.id = id

struct Completion(TrivialRegisterPassable, Writable):
    var id: Int  # Copied back from io_uring CQE user_data
    var result: Int32

    def __init__(out self, id: Int = 0, result: Int32 = 0):
        self.id = id
        self.result = result


struct IoLoadErrorTag(Copyable, TrivialRegisterPassable, Equatable, Writable):
    var raw: UInt8

    def __init__(out self, raw: UInt8):
        self.raw = raw


struct IoLoadErrorKind(TrivialRegisterPassable):
    comptime INVALID_RING_STATE = IoLoadErrorTag(1)
    comptime INVALID_OP_SPEC = IoLoadErrorTag(2)
    comptime DUPLICATE_OP_ID = IoLoadErrorTag(3)
    comptime SUBMIT_FAILED = IoLoadErrorTag(4)
    comptime SUBMIT_SHORT = IoLoadErrorTag(5)
    comptime WAIT_FAILED = IoLoadErrorTag(6)
    comptime UNKNOWN_COMPLETION_ID = IoLoadErrorTag(7)
    comptime DUPLICATE_COMPLETION_ID = IoLoadErrorTag(8)
    comptime CQE_NEGATIVE = IoLoadErrorTag(9)
    comptime SHORT_READ = IoLoadErrorTag(10)
    comptime INTERNAL_STATE = IoLoadErrorTag(11)


def io_load_error_name(tag: IoLoadErrorTag) -> String:
    if tag == IoLoadErrorKind.INVALID_RING_STATE:
        return "INVALID_RING_STATE"
    if tag == IoLoadErrorKind.INVALID_OP_SPEC:
        return "INVALID_OP_SPEC"
    if tag == IoLoadErrorKind.DUPLICATE_OP_ID:
        return "DUPLICATE_OP_ID"
    if tag == IoLoadErrorKind.SUBMIT_FAILED:
        return "SUBMIT_FAILED"
    if tag == IoLoadErrorKind.SUBMIT_SHORT:
        return "SUBMIT_SHORT"
    if tag == IoLoadErrorKind.WAIT_FAILED:
        return "WAIT_FAILED"
    if tag == IoLoadErrorKind.UNKNOWN_COMPLETION_ID:
        return "UNKNOWN_COMPLETION_ID"
    if tag == IoLoadErrorKind.DUPLICATE_COMPLETION_ID:
        return "DUPLICATE_COMPLETION_ID"
    if tag == IoLoadErrorKind.CQE_NEGATIVE:
        return "CQE_NEGATIVE"
    if tag == IoLoadErrorKind.SHORT_READ:
        return "SHORT_READ"
    if tag == IoLoadErrorKind.INTERNAL_STATE:
        return "INTERNAL_STATE"
    return "UNKNOWN_IO_LOAD_ERROR"


struct IoLoadError(Copyable, Writable):
    var kind: IoLoadErrorTag
    var op_id: Int
    var expected: Int
    var actual: Int
    var errno: Int

    def __init__(
        out self,
        kind: IoLoadErrorTag,
        op_id: Int = -1,
        expected: Int = 0,
        actual: Int = 0,
        errno: Int = 0,
    ):
        self.kind = kind
        self.op_id = op_id
        self.expected = expected
        self.actual = actual
        self.errno = errno


def print_io_load_error(err: IoLoadError):
    print(
        "io_uring load error:",
        io_load_error_name(err.kind),
        "op_id",
        err.op_id,
        "expected",
        err.expected,
        "actual",
        err.actual,
        "errno",
        err.errno,
    )


def validate_completion_checked(
    c: Completion,
    expected_by_id: Dict[Int, Int],
    mut seen_by_id: Dict[Int, Int],
) -> Optional[IoLoadError]:
    var expected_opt = expected_by_id.get(c.id)
    var seen_opt = seen_by_id.get(c.id)
    if not expected_opt or not seen_opt:
        return IoLoadError(
            kind=IoLoadErrorKind.UNKNOWN_COMPLETION_ID,
            op_id=c.id,
            actual=Int(c.result),
        )

    if seen_opt.value() != 0:
        return IoLoadError(
            kind=IoLoadErrorKind.DUPLICATE_COMPLETION_ID,
            op_id=c.id,
            actual=Int(c.result),
        )

    seen_by_id[c.id] = 1

    if c.result < 0:
        return IoLoadError(
            kind=IoLoadErrorKind.CQE_NEGATIVE,
            op_id=c.id,
            expected=expected_opt.value(),
            actual=Int(c.result),
            errno=Int(c.result),
        )

    var expected = expected_opt.value()
    var got = Int(c.result)
    if got != expected:
        return IoLoadError(
            kind=IoLoadErrorKind.SHORT_READ,
            op_id=c.id,
            expected=expected,
            actual=got,
        )

    return None


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

    def available(self, max_entries: UInt32) -> Int:
        return Int(max_entries - (self.tail[] - self.head[]))


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
        """Number of completions ready to be used."""
        return Int(self.tail[] - self.head[])


struct IoLoader[queue_depth: Int = 2048](Movable):
    comptime MAX_WAIT_EMPTY_RETRIES = 8

    var ring_fd: Int
    var sq: SubmissionQueue
    var cq: CompletionQueue
    var max_entries: UInt32
    var pending_count: Int
    var file_fds: List[Int32]
    var single_mmap: Bool
    var last_wait_result: Int
    var last_wait_errno: Int

    def __init__(out self):
        comptime assert (Self.queue_depth & (Self.queue_depth - 1)) == 0 and Self.queue_depth > 0, "queue_depth must be a power of 2"
        self.ring_fd = -1
        self.sq = SubmissionQueue()
        self.cq = CompletionQueue()
        self.max_entries = UInt32(Self.queue_depth)
        self.pending_count = 0
        self.file_fds = List[Int32]()
        self.single_mmap = False
        self.last_wait_result = 0
        self.last_wait_errno = 0

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
        """Map submission and completion queue rings after io_uring_setup."""
        var sys = linux.linux_sys()
        self.sq.ring_size = Int(params.sq_off.array) + Int(params.sq_entries) * size_of[UInt32]()
        self.cq.ring_size = Int(params.cq_off.cqes) + Int(params.cq_entries) * size_of[linux.IoUringCqe]()

        self.single_mmap = (params.features & linux.IoUringFeat.SINGLE_MMAP) != 0

        if self.single_mmap:
            if self.cq.ring_size > self.sq.ring_size:
                self.sq.ring_size = self.cq.ring_size
            self.cq.ring_size = self.sq.ring_size

        var sq_addr = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
        ](0, self.sq.ring_size, self.ring_fd, linux.IORING_OFF_SQ_RING)

        if sq_addr < 0:
            _ = sys.sys_close(self.ring_fd)
            self.ring_fd = -1
            return

        self.sq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=sq_addr)

        # Map completion queue ring (may share with submission queue if SINGLE_MMAP)
        if self.single_mmap:
            self.cq.ring = self.sq.ring
        else:
            var cq_addr = sys.sys_mmap[
                prot=linux.Prot.RW,
                flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
            ](0, self.cq.ring_size, self.ring_fd, linux.IORING_OFF_CQ_RING)

            if cq_addr < 0:
                _ = sys.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
                _ = sys.sys_close(self.ring_fd)
                self.ring_fd = -1
                return

            self.cq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cq_addr)

        self.sq.entries_size = Int(params.sq_entries) * size_of[linux.IoUringSqe]()
        var sqes_addr = sys.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
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
        for i in range(len(self.file_fds)):
            if self.file_fds[i] >= 0:
                _ = sys.sys_close(Int(self.file_fds[i]))

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

    def register_files(mut self, paths: List[Path]) -> Int:
        """Returns number of files registered, or negative errno on failure."""
        if self.ring_fd < 0:
            return -1

        var count = len(paths)
        if count == 0:
            return 0

        var sys = linux.linux_sys()
        self.file_fds = List[Int32](capacity=count)
        for i in range(count):
            var path_str = String(paths[i])
            var fd = sys.sys_openat(
                linux.AT_FDCWD,
                path_str,
                linux.OpenFlags.RDONLY | linux.OpenFlags.CLOEXEC,
            )
            if fd < 0:
                for k in range(len(self.file_fds)):
                    _ = sys.sys_close(Int(self.file_fds[k]))
                self.file_fds = List[Int32]()
                return fd

            self.file_fds.append(Int32(fd))

        var result = sys.sys_io_uring_register(
            self.ring_fd,
            linux.IoUringRegisterOp.REGISTER_FILES,
            Int(self.file_fds.unsafe_ptr()),
            UInt32(count),
        )

        if result < 0:
            for i in range(len(self.file_fds)):
                _ = sys.sys_close(Int(self.file_fds[i]))
            self.file_fds = List[Int32]()
            return result

        return count

    def submit(mut self, ops: List[ReadOp]) -> Int:
        """Non-blocking.
        Returns number of ops submitted (may be < len(ops) if queue full).
        """
        if self.ring_fd < 0:
            return -1

        var count = len(ops)
        if count == 0:
            return 0

        var ring_entries = Int(self.sq.mask) + 1
        if ring_entries <= 0:
            return -1

        var tail = self.sq.tail[]
        var head = self.sq.head[]
        var submitted = 0

        for i in range(count):
            if Int(tail - head) >= ring_entries:
                break

            var idx = tail & self.sq.mask
            var sqe = self.sq.entries + Int(idx)

            var op = ops[i]
            sqe[].opcode = linux.IoUringOp.READ
            sqe[].flags = linux.IoUringSqeFlags.FIXED_FILE
            sqe[].fd = op.file_idx
            sqe[].off = UInt64(op.offset)
            sqe[].addr = UInt64(op.dest)
            sqe[].len = UInt32(op.length)
            sqe[].user_data = UInt64(op.id)
            sqe[].ioprio = 0
            sqe[].buf_index = 0
            sqe[].personality = 0
            sqe[].splice_fd_in = 0
            sqe[].addr3 = 0
            sqe[].pad = 0
            sqe[].op_flags = 0

            self.sq.array[Int(idx)] = idx

            tail += 1
            submitted += 1

        if submitted == 0:
            return 0

        self.sq.tail[] = tail

        var sys = linux.linux_sys()
        var result = sys.sys_io_uring_enter(
            self.ring_fd,
            UInt32(submitted),
            0,
            0,
        )

        if result < 0:
            return result

        self.pending_count += Int(result)
        return Int(result)

    def wait(mut self, min_complete: Int = 1) -> List[Completion]:
        """Block until at least min_complete operations finish.
        Returns all available completions.
        """
        var completions = List[Completion]()
        if self.ring_fd < 0:
            return completions^

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        if head == tail and min_complete > 0:
            var sys = linux.linux_sys()
            self.last_wait_result = 0
            self.last_wait_errno = 0
            for _ in range(Self.MAX_WAIT_EMPTY_RETRIES):
                var result = sys.sys_io_uring_enter(
                    self.ring_fd,
                    0,
                    UInt32(min_complete),
                    linux.IoUringEnter.GETEVENTS,
                )
                self.last_wait_result = result
                if result < 0:
                    self.last_wait_errno = result
                    return completions^
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
        """Immediately returns whatever completions are ready."""
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

    def pending(self) -> Int:
        return self.pending_count

    def submit_one_checked(mut self, op: ReadOp) -> Int:
        if self.ring_fd < 0:
            return -1

        if Int(op.length) <= 0:
            return -1

        var ring_entries = Int(self.sq.mask) + 1
        if ring_entries <= 0:
            return -1

        var tail = self.sq.tail[]
        var head = self.sq.head[]
        if Int(tail - head) >= ring_entries:
            return 0

        var idx = tail & self.sq.mask
        var sqe = self.sq.entries + Int(idx)

        sqe[].opcode = linux.IoUringOp.READ
        sqe[].flags = linux.IoUringSqeFlags.FIXED_FILE
        sqe[].fd = op.file_idx
        sqe[].off = UInt64(op.offset)
        sqe[].addr = UInt64(op.dest)
        sqe[].len = UInt32(op.length)
        sqe[].user_data = UInt64(op.id)
        sqe[].ioprio = 0
        sqe[].buf_index = 0
        sqe[].personality = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3 = 0
        sqe[].pad = 0
        sqe[].op_flags = 0
        self.sq.array[Int(idx)] = idx

        self.sq.tail[] = tail + 1

        var sys = linux.linux_sys()
        var result = sys.sys_io_uring_enter(
            self.ring_fd,
            1,
            0,
            0,
        )
        if result < 0:
            self.sq.tail[] = tail
            return result
        if result != 1:
            return -1

        self.pending_count += 1
        return 1

    def process_queue_checked[
        on_complete: def(Completion) capturing -> None,
    ](mut self, ops: List[ReadOp], min_complete: Int = 1) -> Optional[IoLoadError]:
        var total = len(ops)
        if total == 0:
            return None

        if self.ring_fd < 0 or not self.sq or not self.cq:
            return IoLoadError(
                kind=IoLoadErrorKind.INVALID_RING_STATE,
                actual=self.ring_fd,
            )

        var ring_entries = Int(self.sq.mask) + 1
        if ring_entries <= 0:
            return IoLoadError(
                kind=IoLoadErrorKind.INVALID_RING_STATE,
                actual=ring_entries,
            )

        var expected_by_id = Dict[Int, Int]()
        var seen_by_id = Dict[Int, Int]()
        for i in range(total):
            var op = ops[i]
            var expected = Int(op.length)
            if expected <= 0:
                return IoLoadError(
                    kind=IoLoadErrorKind.INVALID_OP_SPEC,
                    op_id=op.id,
                    expected=1,
                    actual=expected,
                )
            var prior = expected_by_id.get(op.id)
            if prior:
                return IoLoadError(
                    kind=IoLoadErrorKind.DUPLICATE_OP_ID,
                    op_id=op.id,
                    expected=prior.value(),
                    actual=expected,
                )
            expected_by_id[op.id] = expected
            seen_by_id[op.id] = 0

        var submitted = 0
        var completed = 0
        while submitted < total:
            var submit_res = self.submit_one_checked(ops[submitted])
            if submit_res < 0:
                return IoLoadError(
                    kind=IoLoadErrorKind.SUBMIT_FAILED,
                    op_id=ops[submitted].id,
                    errno=submit_res,
                    actual=submit_res,
                )

            if submit_res == 0:
                var completions = self.wait(min_complete=min_complete)
                if len(completions) == 0:
                    return IoLoadError(
                        kind=IoLoadErrorKind.WAIT_FAILED,
                        op_id=ops[submitted].id,
                        expected=min_complete,
                        actual=self.last_wait_result,
                        errno=self.last_wait_errno,
                    )
                for c in completions:
                    var err = validate_completion_checked(c, expected_by_id, seen_by_id)
                    if err:
                        return err^
                    on_complete(c)
                    completed += 1
                continue

            submitted += 1

            var completions = self.wait(min_complete=min_complete)
            if len(completions) == 0:
                return IoLoadError(
                    kind=IoLoadErrorKind.WAIT_FAILED,
                    op_id=ops[submitted - 1].id,
                    expected=min_complete,
                    actual=self.last_wait_result,
                    errno=self.last_wait_errno,
                )
            for c in completions:
                var err = validate_completion_checked(c, expected_by_id, seen_by_id)
                if err:
                    return err^
                on_complete(c)
                completed += 1

        while self.pending_count > 0:
            var completions = self.wait(min_complete=min_complete)
            if len(completions) == 0:
                return IoLoadError(
                    kind=IoLoadErrorKind.WAIT_FAILED,
                    expected=min_complete,
                    actual=self.last_wait_result,
                    errno=self.last_wait_errno,
                )
            for c in completions:
                var err = validate_completion_checked(c, expected_by_id, seen_by_id)
                if err:
                    return err^
                on_complete(c)
                completed += 1

        if completed != total:
            return IoLoadError(
                kind=IoLoadErrorKind.INTERNAL_STATE,
                expected=total,
                actual=completed,
            )

        return None

    def process_queue[
        on_complete: def(Completion) capturing -> None,
    ](mut self, ops: List[ReadOp], min_complete: Int = 1) -> Int:
        var completed = 0

        @parameter
        def wrapped(c: Completion):
            completed += 1
            on_complete(c)

        var err = self.process_queue_checked[wrapped](ops, min_complete=min_complete)
        if err:
            return -1
        return completed
