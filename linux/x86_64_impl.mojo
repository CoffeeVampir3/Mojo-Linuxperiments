from std.sys import inlined_assembly
from std.sys.info import size_of
from std.memory import UnsafePointer

from .linux_sys import *

# =============================================================================
# x86_64-specific types
# =============================================================================

comptime SA_RESTORER: UInt64 = 0x04000000

def rt_sigreturn_restorer():
    inlined_assembly[
        "mov $$15, %rax\nsyscall",
        NoneType,
        constraints="~{rax},~{rcx},~{r11},~{memory}",
    ]()

struct KernelRtSigActionX86_64(TrivialRegisterPassable):
    var handler: Int
    var flags: UInt64
    var restorer: Int
    var mask: SigSet64

    def __init__(out self):
        self.handler = 0
        self.flags = 0
        self.restorer = 0
        self.mask = SigSet64()

struct KernelSigInfoX86_64(TrivialRegisterPassable):
    var si_signo: Int32
    var si_errno: Int32
    var si_code: Int32
    var pad0: Int32
    var si_addr: Int

# =============================================================================
# X86_64LinuxSys — conforms to LinuxSys.
#
# Provides: syscall numbers, raw syscall mechanism, and arch-specific ops.
# Inherits: all shared syscall wrappers from LinuxSys defaults.
# =============================================================================

struct X86_64LinuxSys(LinuxSys):
    # Syscall numbers.
    comptime NR_write = 1
    comptime NR_mmap = 9
    comptime NR_mprotect = 10
    comptime NR_munmap = 11
    comptime NR_rt_sigaction = 13
    comptime NR_sigaltstack = 131
    comptime NR_exit = 60
    comptime NR_getpid = 39
    comptime NR_gettid = 186
    comptime NR_getcpu = 309
    comptime NR_sched_yield = 24
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

    def __init__(out self):
        pass

    # --- Raw syscall mechanism (x86_64 register ABI) ---

    def syscall[*Ts: Intable](self, nr: Int, *args: *Ts) -> Int:
        comptime count = args.__len__()
        comptime assert count <= 6, "syscall supports 0-6 arguments"

        var a0 = 0
        var a1 = 0
        var a2 = 0
        var a3 = 0
        var a4 = 0
        var a5 = 0
        comptime if count > 0: a0 = Int(args[0])
        comptime if count > 1: a1 = Int(args[1])
        comptime if count > 2: a2 = Int(args[2])
        comptime if count > 3: a3 = Int(args[3])
        comptime if count > 4: a4 = Int(args[4])
        comptime if count > 5: a5 = Int(args[5])

        return Int(inlined_assembly[
            "mov %rcx, %r10\nsyscall",
            Int, Int, Int, Int, Int, Int, Int, Int,
            constraints="={rax},{rax},{rdi},{rsi},{rdx},{rcx},{r8},{r9},~{rcx},~{r10},~{r11},~{memory}",
        ](nr, a0, a1, a2, a3, a4, a5))

    # --- Architecture-specific operations ---

    def arch_cpu_relax(self):
        inlined_assembly["pause", NoneType, constraints="~{memory}"]()

    def arch_thread_pointer(self) -> Int:
        return Int(inlined_assembly["mov %fs:0, $0", Int, constraints="=r"]())

    def arch_tls_load_i64[offset: Int](self) -> Int:
        comptime asm = "mov %fs:" + String(offset) + ", $0"
        return Int(inlined_assembly[asm, Int, constraints="=r"]())

    def sys_rt_sigaction(
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
            var result = self.syscall(
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
            var result = self.syscall(
                Self.NR_rt_sigaction,
                signum,
                Int(kact_ptr),
                0,
                size_of[SigSet64](),
            )
            _ = kact_ptr[]
            return result

    def sys_clone3_with_entry(
        self,
        clone_args_ptr: UnsafePointer[Clone3Args, MutAnyOrigin],
        clone_args_size: Int,
    ) -> Int:
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

    def arch_decode_sigsegv(self, siginfo: Int, ucontext: Int) -> SigSegvContext:
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
