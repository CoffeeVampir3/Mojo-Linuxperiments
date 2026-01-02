from sys import CompilationTarget
from sys.info import is_64bit

from .linux_sys import *
from .x86_64_impl import X86_64LinuxSys

comptime IS_X86_64: Bool = is_64bit() and CompilationTarget.is_x86()


fn require_supported_target():
    constrained[
        IS_X86_64,
        "linux.sys: unsupported target (only x86_64 implemented); add an aarch64 implementation",
    ]()


struct LinuxSysImpl(LinuxSys):
    fn __init__(out self):
        pass

    fn syscall[count: Int](self, nr: Int, *args: Int) -> Int:
        require_supported_target()
        var sys = X86_64LinuxSys()
        @parameter
        if count == 0:
            return sys.syscall[0](nr)
        elif count == 1:
            return sys.syscall[1](nr, args[0])
        elif count == 2:
            return sys.syscall[2](nr, args[0], args[1])
        elif count == 3:
            return sys.syscall[3](nr, args[0], args[1], args[2])
        elif count == 4:
            return sys.syscall[4](nr, args[0], args[1], args[2], args[3])
        elif count == 5:
            return sys.syscall[5](nr, args[0], args[1], args[2], args[3], args[4])
        elif count == 6:
            return sys.syscall[6](nr, args[0], args[1], args[2], args[3], args[4], args[5])
        else:
            constrained[False, "syscall supports 0-6 arguments"]()
            return 0

    fn sys_mmap[
        prot: Int = Prot.RW,
        flags: Int = MapFlag.PRIVATE | MapFlag.ANONYMOUS,
    ](self, addr: Int, length: Int, fd: Int = -1, offset: Int = 0) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_mmap[prot=prot, flags=flags](addr, length, fd, offset)

    fn sys_munmap(self, addr: Int, length: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_munmap(addr, length)

    fn sys_mbind[
        policy: Int = Mempolicy.BIND,
        flags: Int = 0,
    ](self, addr: Int, length: Int, nodemask: UInt64, maxnode: Int = 64) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_mbind[policy=policy, flags=flags](addr, length, nodemask, maxnode)

    fn sys_madvise[advice: Int](self, addr: Int, length: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_madvise[advice](addr, length)

    fn sys_move_pages_query(self, addr: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_move_pages_query(addr)

    fn sys_mprotect(self, addr: Int, length: Int, prot: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_mprotect(addr, length, prot)

    fn sys_write(self, fd: Int, buf: Int, count: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_write(fd, buf, count)

    fn sys_rt_sigaction(
        self,
        signum: Int,
        act: UnsafePointer[RtSigAction, MutAnyOrigin],
        old: UnsafePointer[RtSigAction, MutAnyOrigin] = UnsafePointer[RtSigAction, MutAnyOrigin](),
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_rt_sigaction(signum, act, old)

    fn sys_sigaltstack(
        self,
        ss: UnsafePointer[StackT, MutAnyOrigin],
        old: UnsafePointer[StackT, MutAnyOrigin] = UnsafePointer[StackT, MutAnyOrigin](),
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_sigaltstack(ss, old)

    fn sys_futex_wait(self, addr: Int, expected: Int, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_futex_wait(addr, expected, flags)

    fn sys_futex_waitv(
        self,
        waiters: UnsafePointer[FutexWaitv],
        nr_futexes: Int,
        flags: Int = 0,
        timeout: Int = 0,
        clockid: Int = 0,
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_futex_waitv(waiters, nr_futexes, flags, timeout, clockid)

    fn sys_futex_wake(self, addr: Int, nr_wake: Int = 1, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_futex_wake(addr, nr_wake, flags)

    fn sys_exit(self, code: Int = 0):
        require_supported_target()
        X86_64LinuxSys().sys_exit(code)

    fn sys_exit_group(self, code: Int = 0):
        require_supported_target()
        X86_64LinuxSys().sys_exit_group(code)

    fn sys_getpid(self) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_getpid()

    fn sys_gettid(self) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_gettid()

    fn sys_tgkill(self, pid: Int, tid: Int, sig: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_tgkill(pid, tid, sig)

    fn sys_rseq(self, rseq_ptr: Int, len: Int, flags: Int, sig: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_rseq(rseq_ptr, len, flags, sig)

    fn sys_sched_setaffinity(self, tid: Int, mask_size: Int, mask_ptr: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_sched_setaffinity(tid, mask_size, mask_ptr)

    fn sys_openat(self, dirfd: Int, mut pathname: String, flags: Int, mode: Int = 0) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_openat(dirfd, pathname, flags, mode)

    fn sys_close(self, fd: Int) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_close(fd)

    fn sys_io_uring_setup(self, entries: UInt32, params: UnsafePointer[IoUringParams]) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_io_uring_setup(entries, params)

    fn sys_io_uring_enter(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_io_uring_enter(fd, to_submit, min_complete, flags)

    fn sys_io_uring_enter_sig(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
        sig: Int,
        sigsz: Int,
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_io_uring_enter_sig(fd, to_submit, min_complete, flags, sig, sigsz)

    fn sys_io_uring_register(
        self,
        fd: Int,
        opcode: UInt32,
        arg: Int,
        nr_args: UInt32,
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_io_uring_register(fd, opcode, arg, nr_args)

    fn arch_cpu_relax(self):
        require_supported_target()
        X86_64LinuxSys().arch_cpu_relax()

    fn arch_thread_pointer(self) -> Int:
        require_supported_target()
        return X86_64LinuxSys().arch_thread_pointer()

    fn arch_tls_load_i64[offset: Int](self) -> Int:
        require_supported_target()
        return X86_64LinuxSys().arch_tls_load_i64[offset]()

    fn sys_clone3_with_entry(
        self,
        clone_args_ptr: UnsafePointer[Clone3Args, MutAnyOrigin],
        clone_args_size: Int,
    ) -> Int:
        require_supported_target()
        return X86_64LinuxSys().sys_clone3_with_entry(clone_args_ptr, clone_args_size)

    fn arch_decode_sigsegv(self, siginfo: Int, ucontext: Int) -> SigSegvContext:
        require_supported_target()
        return X86_64LinuxSys().arch_decode_sigsegv(siginfo, ucontext)


fn linux_sys() -> LinuxSysImpl:
    return LinuxSysImpl()
