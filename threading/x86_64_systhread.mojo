from sys.intrinsics import inlined_assembly

fn syscall_mmap(addr: Int64, length: Int64, prot: Int64, flags: Int64) -> Int64:
    return inlined_assembly[
        """
        mov $$9, %rax
        mov $$-1, %r8
        xor %r9d, %r9d
        syscall
        """,
        Int64,
        Int64, Int64, Int64, Int64,
        constraints = "={rax},{rdi},{rsi},{rdx},{r10},~{r8},~{r9},~{rcx},~{r11},~{memory}",
    ](addr, length, prot, flags)

fn syscall_sched_setaffinity(pid: Int64, cpusetsize: Int64, mask: UnsafePointer[UInt8, MutOrigin.external]) -> Int64:
    return inlined_assembly[
        """
        mov $$203, %rax
        syscall
        """,
        Int64,
        Int64, Int64, UnsafePointer[UInt8, MutOrigin.external],
        constraints = "={rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}",
    ](pid, cpusetsize, mask)

fn syscall_mprotect(addr: Int64, length: Int64, prot: Int64) -> Int64:
    # Not strictly neccesary but without mprotect we don't have fault protection for
    # thread stack overflow.
    return inlined_assembly[
        """
        mov $$10, %rax
        syscall
        """,
        Int64,
        Int64, Int64, Int64,
        constraints = "={rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}",
    ](addr, length, prot)

fn syscall_munmap(addr: Int64, length: Int64) -> Int64:
    return inlined_assembly[
        """
        mov $$11, %rax
        syscall
        """,
        Int64,
        Int64, Int64,
        constraints = "={rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
    ](addr, length)

fn syscall_futex_wait(addr: Int64, expected_val: Int64) -> Int64:
    """futex(addr, FUTEX_WAIT, expected, NULL, NULL, 0) - syscall 202"""
    return inlined_assembly[
        """
        mov $$202, %rax
        xor %esi, %esi
        xor %r10d, %r10d
        xor %r8d, %r8d
        xor %r9d, %r9d
        syscall
        """,
        Int64,
        Int64, Int64,
        constraints = "={rax},{rdi},{rdx},~{rsi},~{r10},~{r8},~{r9},~{rcx},~{r11},~{memory}",
    ](addr, expected_val)

fn get_fs_base() -> Int64:
    """Read FS segment base (glibc stores TCB self-pointer at FS:0)."""
    return inlined_assembly[
        "mov %fs:0, $0",
        Int64,
        constraints = "=r",
    ]()

fn arg_reg[i: Int]() -> String:
    constrained[i >= 0 and i < 6, "x86-64 only has 6 argument registers (0-5)"]()
    @parameter
    if i == 0: return "rdi"
    elif i == 1: return "rsi"
    elif i == 2: return "rdx"
    elif i == 3: return "rcx"
    elif i == 4: return "r8"
    else: return "r9"

fn build_arg_loads[num_args: Int, args_offset: Int]() -> String:
    var result = String("")
    @parameter
    for i in range(num_args):
        result += "mov " + String(args_offset + i * 8) + "(%r13), %" + arg_reg[i]() + "\n"
    return result

fn clone_trampoline_capture[num_args: Int, args_offset: Int, cpu_mask_offset: Int](
    flags: Int64,
    stack_top: Int64,
    child_tid_addr: Int64,
    tls_addr: Int64,
    init_func: Int64,
    init_arg: Int64,
    user_func: Int64,
    cpu_mask_size: Int64,
) -> Int64:
    """
    Clone with optional CPU affinity pinning.
    If cpu_mask_size > 0, calls sched_setaffinity before init_func.
    If cpu_mask_size == 0, skips affinity (degenerate unpinned case).
    Mask is read from mmap_base + cpu_mask_offset.
    r13 = init_arg = mmap_base, r15 = cpu_mask_size.
    """
    constrained[num_args >= 0 and num_args <= 6, "x86-64 supports max 6 register args"]()
    alias arg_loads = build_arg_loads[num_args, args_offset]()
    return inlined_assembly[
        """
        mov %rdx, %r10
        mov $$56, %rax
        syscall

        test %rax, %rax
        jnz 1f

        // Child: align stack
        and $$-16, %rsp

        // Pin to CPUs if cpu_mask_size > 0
        test %r15, %r15
        jz 2f
        mov $$203, %rax
        xor %edi, %edi
        mov %r15, %rsi
        lea """ + String(cpu_mask_offset) + """(%r13), %rdx
        syscall
    2:
        // Init TLS (first-touch on correct NUMA node)
        mov %r13, %rdi
        call *%r12

        // Load capture args from mmap region, then call user func
        """ + arg_loads + """
        call *%r14

        mov $$60, %rax
        xor %edi, %edi
        syscall
    1:
        """,
        Int64,
        Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64,
        constraints = "={rax},{rdi},{rsi},{rdx},{r8},{r12},{r13},{r14},{r15},~{r10},~{rcx},~{r11},~{memory}",
    ](flags, stack_top, child_tid_addr, tls_addr, init_func, init_arg, user_func, cpu_mask_size)
