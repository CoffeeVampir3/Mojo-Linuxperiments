from sys.intrinsics import inlined_assembly
from sys.info import size_of
from memory import UnsafePointer

from .linux_sys import *

comptime SA_RESTORER: UInt64 = 0x04000000

fn rt_sigreturn_restorer():
    inlined_assembly[
        "mov $$15, %rax\nsyscall",
        NoneType,
        constraints="~{rax},~{rcx},~{r11},~{memory}",
    ]()

@register_passable("trivial")
struct KernelRtSigActionX86_64:
    var handler: Int
    var flags: UInt64
    var restorer: Int
    var mask: SigSet64

    fn __init__(out self):
        self.handler = 0
        self.flags = 0
        self.restorer = 0
        self.mask = SigSet64()

@register_passable("trivial")
struct KernelSigInfoX86_64:
    # Minimal siginfo_t prefix + si_addr for SIGSEGV (x86_64).
    var si_signo: Int32
    var si_errno: Int32
    var si_code: Int32
    var pad0: Int32
    var si_addr: Int


struct X86_64LinuxSys(LinuxSys):
    comptime NR_write = 1
    comptime NR_mmap = 9
    comptime NR_mprotect = 10
    comptime NR_munmap = 11
    comptime NR_rt_sigaction = 13
    comptime NR_sigaltstack = 131
    comptime NR_exit = 60
    comptime NR_getpid = 39
    comptime NR_gettid = 186
    comptime NR_sched_setaffinity = 203
    comptime NR_exit_group = 231
    comptime NR_tgkill = 234
    comptime NR_mbind = 237
    comptime NR_openat = 257
    comptime NR_close = 3
    comptime NR_move_pages = 279
    comptime NR_rseq = 334
    comptime NR_madvise = 28
    comptime NR_io_uring_setup = 425
    comptime NR_io_uring_enter = 426
    comptime NR_io_uring_register = 427
    comptime NR_clone3 = 435
    comptime NR_futex_waitv = 449
    comptime NR_futex_wake = 454
    comptime NR_futex_wait = 455

    fn __init__(out self):
        pass

    fn syscall[count: Int](self, nr: Int, *args: Int) -> Int:
        comptime regs = ("", ",{rdi}", ",{rdi},{rsi}", ",{rdi},{rsi},{rdx}",
                      ",{rdi},{rsi},{rdx},{rcx}", ",{rdi},{rsi},{rdx},{rcx},{r8}",
                      ",{rdi},{rsi},{rdx},{rcx},{r8},{r9}")
        comptime asm = "mov %rcx, %r10\nsyscall" if count > 3 else "syscall"
        comptime constraints = "={rax},{rax}" + regs[count] + ",~{rcx},~{r10},~{r11},~{memory}"
        @parameter
        if count == 0:
            return Int(inlined_assembly[asm, Int, Int, constraints=constraints](nr))
        elif count == 1:
            return Int(inlined_assembly[asm, Int, Int, Int, constraints=constraints](nr, args[0]))
        elif count == 2:
            return Int(
                inlined_assembly[asm, Int, Int, Int, Int, constraints=constraints](nr, args[0], args[1])
            )
        elif count == 3:
            return Int(
                inlined_assembly[asm, Int, Int, Int, Int, Int, constraints=constraints](
                    nr, args[0], args[1], args[2]
                )
            )
        elif count == 4:
            return Int(
                inlined_assembly[asm, Int, Int, Int, Int, Int, Int, constraints=constraints](
                    nr, args[0], args[1], args[2], args[3]
                )
            )
        elif count == 5:
            return Int(
                inlined_assembly[asm, Int, Int, Int, Int, Int, Int, Int, constraints=constraints](
                    nr, args[0], args[1], args[2], args[3], args[4]
                )
            )
        elif count == 6:
            return Int(
                inlined_assembly[asm, Int, Int, Int, Int, Int, Int, Int, Int, constraints=constraints](
                    nr, args[0], args[1], args[2], args[3], args[4], args[5]
                )
            )
        else:
            constrained[False, "syscall supports 0-6 arguments"]()
            return 0

    fn arch_cpu_relax(self):
        inlined_assembly["pause", NoneType, constraints="~{memory}"]()

    fn arch_thread_pointer(self) -> Int:
        # The thread pointer is stored at %fs:0 on x86-64 (tcb self-pointer).
        return Int(inlined_assembly["mov %fs:0, $0", Int, constraints="=r"]())

    fn arch_tls_load_i64[offset: Int](self) -> Int:
        comptime asm = "mov %fs:" + String(offset) + ", $0"
        return Int(inlined_assembly[asm, Int, constraints="=r"]())

    fn sys_mmap[
        prot: Int = Prot.RW,
        flags: Int = MapFlag.PRIVATE | MapFlag.ANONYMOUS,
    ](self, addr: Int, length: Int, fd: Int = -1, offset: Int = 0) -> Int:
        return self.syscall[6](Self.NR_mmap, addr, length, prot, flags, fd, offset)

    fn sys_munmap(self, addr: Int, length: Int) -> Int:
        return self.syscall[2](Self.NR_munmap, addr, length)

    fn sys_mbind[
        policy: Int = Mempolicy.BIND,
        flags: Int = 0,
    ](self, addr: Int, length: Int, nodemask: UInt64, maxnode: Int = 64) -> Int:
        var mask_storage = InlineArray[UInt64, 1](nodemask)
        var mask_ptr = UnsafePointer(to=mask_storage)
        var result = self.syscall[6](Self.NR_mbind, addr, length, policy, Int(mask_ptr), maxnode, flags)
        _ = mask_ptr[]
        return result

    fn sys_madvise[advice: Int](self, addr: Int, length: Int) -> Int:
        return self.syscall[3](Self.NR_madvise, addr, length, advice)

    fn sys_move_pages_query(self, addr: Int) -> Int:
        var pages = InlineArray[Int, 1](addr)
        var status = InlineArray[Int32, 1](Int32(-1))
        var pages_ptr = UnsafePointer(to=pages)
        var status_ptr = UnsafePointer(to=status)
        var result = self.syscall[6](Self.NR_move_pages, 0, 1, Int(pages_ptr), 0, Int(status_ptr), 0)
        _ = pages_ptr[]
        _ = status_ptr[]
        if result < 0:
            return result
        return Int(status[0])

    fn sys_mprotect(self, addr: Int, length: Int, prot: Int) -> Int:
        return self.syscall[3](Self.NR_mprotect, addr, length, prot)

    fn sys_write(self, fd: Int, buf: Int, count: Int) -> Int:
        return self.syscall[3](Self.NR_write, fd, buf, count)

    fn sys_rt_sigaction(
        self,
        signum: Int,
        act: UnsafePointer[RtSigAction, MutAnyOrigin],
        old: UnsafePointer[RtSigAction, MutAnyOrigin] = UnsafePointer[RtSigAction, MutAnyOrigin](),
    ) -> Int:
        var restorer_copy = rt_sigreturn_restorer
        var restorer_addr = UnsafePointer(to=restorer_copy).bitcast[Int]()[]

        var kact = KernelRtSigActionX86_64()
        kact.handler = act[].handler
        kact.flags = act[].flags | UInt64(SA_RESTORER)
        kact.restorer = restorer_addr
        kact.mask = act[].mask

        var kact_ptr = UnsafePointer(to=kact)
        if Int(old) != 0:
            var kold = KernelRtSigActionX86_64()
            var kold_ptr = UnsafePointer(to=kold)
            var result = self.syscall[4](
                Self.NR_rt_sigaction,
                signum,
                Int(kact_ptr),
                Int(kold_ptr),
                size_of[SigSet64](),
            )
            if result == 0:
                old[].handler = kold_ptr[].handler
                old[].flags = kold_ptr[].flags & ~UInt64(SA_RESTORER)
                old[].mask = kold_ptr[].mask
            _ = kold_ptr[]
            _ = kact_ptr[]
            return result
        else:
            var result = self.syscall[4](
                Self.NR_rt_sigaction,
                signum,
                Int(kact_ptr),
                0,
                size_of[SigSet64](),
            )
            _ = kact_ptr[]
            return result

    fn sys_sigaltstack(
        self,
        ss: UnsafePointer[StackT, MutAnyOrigin],
        old: UnsafePointer[StackT, MutAnyOrigin] = UnsafePointer[StackT, MutAnyOrigin](),
    ) -> Int:
        return self.syscall[2](Self.NR_sigaltstack, Int(ss), Int(old))

    fn sys_futex_wait(self, addr: Int, expected: Int, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
        return self.syscall[6](Self.NR_futex_wait, addr, expected, FUTEX_BITSET_MATCH_ANY, flags, 0, 0)

    fn sys_futex_waitv(
        self,
        waiters: UnsafePointer[FutexWaitv],
        nr_futexes: Int,
        flags: Int = 0,
        timeout: Int = 0,
        clockid: Int = 0,
    ) -> Int:
        return self.syscall[5](Self.NR_futex_waitv, Int(waiters), nr_futexes, flags, timeout, clockid)

    fn sys_futex_wake(self, addr: Int, nr_wake: Int = 1, flags: Int = Futex2.SIZE_U32 | Futex2.PRIVATE) -> Int:
        return self.syscall[4](Self.NR_futex_wake, addr, FUTEX_BITSET_MATCH_ANY, nr_wake, flags)

    fn sys_exit(self, code: Int = 0):
        _ = self.syscall[1](Self.NR_exit, code)

    fn sys_exit_group(self, code: Int = 0):
        _ = self.syscall[1](Self.NR_exit_group, code)

    fn sys_getpid(self) -> Int:
        return self.syscall[0](Self.NR_getpid)

    fn sys_gettid(self) -> Int:
        return self.syscall[0](Self.NR_gettid)

    fn sys_tgkill(self, pid: Int, tid: Int, sig: Int) -> Int:
        return self.syscall[3](Self.NR_tgkill, pid, tid, sig)

    fn sys_rseq(self, rseq_ptr: Int, len: Int, flags: Int, sig: Int) -> Int:
        return self.syscall[4](Self.NR_rseq, rseq_ptr, len, flags, sig)

    fn sys_sched_setaffinity(self, tid: Int, mask_size: Int, mask_ptr: Int) -> Int:
        return self.syscall[3](Self.NR_sched_setaffinity, tid, mask_size, mask_ptr)

    fn sys_openat(self, dirfd: Int, mut pathname: String, flags: Int, mode: Int = 0) -> Int:
        var cstr = pathname.as_c_string_slice()
        return self.syscall[4](Self.NR_openat, dirfd, Int(cstr.unsafe_ptr()), flags, mode)

    fn sys_clone3_with_entry(
        self,
        clone_args_ptr: UnsafePointer[Clone3Args, MutAnyOrigin],
        clone_args_size: Int,
    ) -> Int:
        # Child diverges via `ret` to the entry pointer placed at the top of the
        # new stack. The entry is called with arg0 = stack head pointer.
        comptime asm = (
            "mov $$" + String(Self.NR_clone3) + ", %rax\n"
            "syscall\n"
            "test %rax, %rax\n"
            "jnz 1f\n"
            "mov %rsp, %rdi\n"
            "ret\n"
            "1:"
        )
        return Int(
            inlined_assembly[
                asm,
                Int,
                Int,
                Int,
                constraints="={rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
            ](Int(clone_args_ptr), clone_args_size)
        )

    fn sys_close(self, fd: Int) -> Int:
        return self.syscall[1](Self.NR_close, fd)

    fn sys_io_uring_setup(self, entries: UInt32, params: UnsafePointer[IoUringParams]) -> Int:
        return self.syscall[2](Self.NR_io_uring_setup, Int(entries), Int(params))

    fn sys_io_uring_enter(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
    ) -> Int:
        return self.syscall[4](
            Self.NR_io_uring_enter,
            fd,
            Int(to_submit),
            Int(min_complete),
            Int(flags),
        )

    fn sys_io_uring_enter_sig(
        self,
        fd: Int,
        to_submit: UInt32,
        min_complete: UInt32,
        flags: UInt32,
        sig: Int,
        sigsz: Int,
    ) -> Int:
        return self.syscall[6](
            Self.NR_io_uring_enter,
            fd,
            Int(to_submit),
            Int(min_complete),
            Int(flags),
            sig,
            sigsz,
        )

    fn sys_io_uring_register(
        self,
        fd: Int,
        opcode: UInt32,
        arg: Int,
        nr_args: UInt32,
    ) -> Int:
        return self.syscall[4](
            Self.NR_io_uring_register,
            fd,
            Int(opcode),
            arg,
            Int(nr_args),
        )

    fn arch_decode_sigsegv(self, siginfo: Int, ucontext: Int) -> SigSegvContext:
        # x86-64 ucontext_t layout (glibc/kernel ABI):
        # gregs array is at offset 40 bytes and uses fixed indices.
        comptime UCONTEXT_GREGS_OFFSET = 40
        comptime REG_RSP = 15
        comptime REG_RIP = 16

        var ctx = SigSegvContext()
        var gregs = ucontext + UCONTEXT_GREGS_OFFSET
        ctx.sp = UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=gregs + REG_RSP * 8)[]
        ctx.ip = UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=gregs + REG_RIP * 8)[]
        ctx.fault_addr = UInt64(
            UnsafePointer[KernelSigInfoX86_64, MutAnyOrigin](unsafe_from_address=siginfo)[].si_addr
        )
        return ctx
