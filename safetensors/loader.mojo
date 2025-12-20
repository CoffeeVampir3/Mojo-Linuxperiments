import linux.syscalls as linux
from memory import UnsafePointer
from pathlib import Path
from sys.info import size_of

@register_passable("trivial")
struct ReadOp:
    """A single read operation: file region → buffer."""
    var file_idx: Int32
    var offset: Int64
    var length: Int32
    var dest: Int64
    var user_data: Int64

    fn __init__(out self, file_idx: Int, offset: Int, length: Int, dest: Int, user_data: Int = 0):
        self.file_idx = Int32(file_idx)
        self.offset = Int64(offset)
        self.length = Int32(length)
        self.dest = Int64(dest)
        self.user_data = Int64(user_data)

@register_passable("trivial")
struct Completion:
    var user_data: Int64
    var result: Int32

    fn __init__(out self, user_data: Int64 = 0, result: Int32 = 0):
        self.user_data = user_data
        self.result = result


@register_passable("trivial")
struct SubmissionQueue:
    var ring: UnsafePointer[UInt8, MutAnyOrigin]
    var ring_size: Int
    var head: UnsafePointer[UInt32, MutAnyOrigin]
    var tail: UnsafePointer[UInt32, MutAnyOrigin]
    var mask: UInt32
    var array: UnsafePointer[UInt32, MutAnyOrigin]
    var entries: UnsafePointer[linux.IoUringSqe, MutAnyOrigin]
    var entries_size: Int

    fn __init__(out self):
        self.ring = UnsafePointer[UInt8, MutAnyOrigin]()
        self.ring_size = 0
        self.head = UnsafePointer[UInt32, MutAnyOrigin]()
        self.tail = UnsafePointer[UInt32, MutAnyOrigin]()
        self.mask = 0
        self.array = UnsafePointer[UInt32, MutAnyOrigin]()
        self.entries = UnsafePointer[linux.IoUringSqe, MutAnyOrigin]()
        self.entries_size = 0

    fn __bool__(self) -> Bool:
        return self.ring.__bool__()

    fn available(self, max_entries: UInt32) -> Int:
        return Int(max_entries - (self.tail[] - self.head[]))


@register_passable("trivial")
struct CompletionQueue:
    var ring: UnsafePointer[UInt8, MutAnyOrigin]
    var ring_size: Int
    var head: UnsafePointer[UInt32, MutAnyOrigin]
    var tail: UnsafePointer[UInt32, MutAnyOrigin]
    var mask: UInt32
    var entries: UnsafePointer[linux.IoUringCqe, MutAnyOrigin]

    fn __init__(out self):
        self.ring = UnsafePointer[UInt8, MutAnyOrigin]()
        self.ring_size = 0
        self.head = UnsafePointer[UInt32, MutAnyOrigin]()
        self.tail = UnsafePointer[UInt32, MutAnyOrigin]()
        self.mask = 0
        self.entries = UnsafePointer[linux.IoUringCqe, MutAnyOrigin]()

    fn __bool__(self) -> Bool:
        return self.ring.__bool__()

    fn ready(self) -> Int:
        """Number of completions ready to be used."""
        return Int(self.tail[] - self.head[])


struct IoLoader[queue_depth: Int = 2048](Movable):
    var ring_fd: Int
    var sq: SubmissionQueue
    var cq: CompletionQueue
    var max_entries: UInt32
    var pending_count: Int
    var file_fds: List[Int32]
    var single_mmap: Bool

    fn __init__(out self):
        constrained[
            (Self.queue_depth & (Self.queue_depth - 1)) == 0 and Self.queue_depth > 0,
            "queue_depth must be a power of 2"
        ]()
        self.ring_fd = -1
        self.sq = SubmissionQueue()
        self.cq = CompletionQueue()
        self.max_entries = UInt32(Self.queue_depth)
        self.pending_count = 0
        self.file_fds = List[Int32]()
        self.single_mmap = False

        var params = linux.IoUringParams()
        var params_ptr = UnsafePointer(to=params)
        var fd = linux.sys_io_uring_setup(self.max_entries, params_ptr)
        if fd < 0:
            return

        self.ring_fd = fd
        params = params_ptr[]

        self.map_rings(params)

    fn map_rings(mut self, params: linux.IoUringParams):
        """Map submission and completion queue rings after io_uring_setup."""
        self.sq.ring_size = Int(params.sq_off.array) + Int(params.sq_entries) * size_of[UInt32]()
        self.cq.ring_size = Int(params.cq_off.cqes) + Int(params.cq_entries) * size_of[linux.IoUringCqe]()

        self.single_mmap = (params.features & linux.IoUringFeat.SINGLE_MMAP) != 0

        if self.single_mmap:
            if self.cq.ring_size > self.sq.ring_size:
                self.sq.ring_size = self.cq.ring_size
            self.cq.ring_size = self.sq.ring_size

        var sq_addr = linux.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
        ](0, self.sq.ring_size, self.ring_fd, linux.IORING_OFF_SQ_RING)

        if sq_addr < 0:
            _ = linux.sys_close(self.ring_fd)
            self.ring_fd = -1
            return

        self.sq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=sq_addr)

        # Map completion queue ring (may share with submission queue if SINGLE_MMAP)
        if self.single_mmap:
            self.cq.ring = self.sq.ring
        else:
            var cq_addr = linux.sys_mmap[
                prot=linux.Prot.RW,
                flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
            ](0, self.cq.ring_size, self.ring_fd, linux.IORING_OFF_CQ_RING)

            if cq_addr < 0:
                _ = linux.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
                _ = linux.sys_close(self.ring_fd)
                self.ring_fd = -1
                return

            self.cq.ring = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cq_addr)

        self.sq.entries_size = Int(params.sq_entries) * size_of[linux.IoUringSqe]()
        var sqes_addr = linux.sys_mmap[
            prot=linux.Prot.RW,
            flags=linux.MapFlag.SHARED | linux.MapFlag.POPULATE,
        ](0, self.sq.entries_size, self.ring_fd, linux.IORING_OFF_SQES)

        if sqes_addr < 0:
            _ = linux.sys_munmap(Int(self.sq.ring), self.sq.ring_size)
            if not self.single_mmap:
                _ = linux.sys_munmap(Int(self.cq.ring), self.cq.ring_size)
            _ = linux.sys_close(self.ring_fd)
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

    fn __del__(deinit self):
        for i in range(len(self.file_fds)):
            if self.file_fds[i] >= 0:
                _ = linux.sys_close(Int(self.file_fds[i]))

        if self.ring_fd < 0:
            return

        if self.sq.entries:
            _ = linux.sys_munmap(Int(self.sq.entries), self.sq.entries_size)

        if self.cq.ring and not self.single_mmap:
            _ = linux.sys_munmap(Int(self.cq.ring), self.cq.ring_size)

        if self.sq.ring:
            _ = linux.sys_munmap(Int(self.sq.ring), self.sq.ring_size)

        _ = linux.sys_close(self.ring_fd)

    fn __bool__(self) -> Bool:
        return self.ring_fd >= 0

    fn register_files(mut self, paths: List[Path]) -> Int:
        """
        Returns number of files registered, or negative errno on failure.
        """
        if self.ring_fd < 0:
            return -1

        var count = len(paths)
        if count == 0:
            return 0

        self.file_fds = List[Int32](capacity=count)
        for i in range(count):
            var path_str = String(paths[i])
            var fd = linux.sys_openat(
                linux.AT_FDCWD,
                path_str,
                linux.OpenFlags.RDONLY | linux.OpenFlags.CLOEXEC,
            )
            if fd < 0:
                for k in range(len(self.file_fds)):
                    _ = linux.sys_close(Int(self.file_fds[k]))
                self.file_fds = List[Int32]()
                return fd

            self.file_fds.append(Int32(fd))

        var result = linux.sys_io_uring_register(
            self.ring_fd,
            linux.IoUringRegisterOp.REGISTER_FILES,
            Int(self.file_fds.unsafe_ptr()),
            UInt32(count),
        )

        if result < 0:
            for i in range(len(self.file_fds)):
                _ = linux.sys_close(Int(self.file_fds[i]))
            self.file_fds = List[Int32]()
            return result

        return count

    fn submit(mut self, ops: List[ReadOp]) -> Int:
        """Non-blocking.
        Returns number of ops submitted (may be < len(ops) if queue full).
        """
        if self.ring_fd < 0:
            return -1

        var count = len(ops)
        if count == 0:
            return 0

        var tail = self.sq.tail[]
        var head = self.sq.head[]
        var submitted = 0

        for i in range(count):
            if tail - head >= self.max_entries:
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
            sqe[].user_data = UInt64(op.user_data)
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

        var result = linux.sys_io_uring_enter(
            self.ring_fd,
            UInt32(submitted),
            0,
            0,
        )

        if result < 0:
            return result

        self.pending_count += submitted
        return submitted

    fn wait(mut self, min_complete: Int = 1) -> List[Completion]:
        """Block until at least min_complete operations finish. (No burn, kernel sleep)
        Returns all available completions.
        """
        var completions = List[Completion]()
        if self.ring_fd < 0:
            return completions^

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        if head == tail and min_complete > 0:
            var result = linux.sys_io_uring_enter(
                self.ring_fd,
                0,
                UInt32(min_complete),
                linux.IoUringEnter.GETEVENTS,
            )
            if result < 0:
                return completions^
            tail = self.cq.tail[]

        while head != tail:
            var idx = head & self.cq.mask
            var cqe = self.cq.entries[Int(idx)]
            completions.append(Completion(Int64(cqe.user_data), cqe.res))
            head += 1
            self.pending_count -= 1

        self.cq.head[] = head
        return completions^

    fn poll(mut self) -> List[Completion]:
        """Immediately returns whatever completions are ready."""
        var completions = List[Completion]()
        if self.ring_fd < 0:
            return completions^

        var head = self.cq.head[]
        var tail = self.cq.tail[]

        # Pointer traversal.
        while head != tail:
            var idx = head & self.cq.mask
            var cqe = self.cq.entries[Int(idx)]
            completions.append(Completion(Int64(cqe.user_data), cqe.res))
            head += 1
            self.pending_count -= 1

        self.cq.head[] = head
        return completions^

    fn pending(self) -> Int:
        return self.pending_count
