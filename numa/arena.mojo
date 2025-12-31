from sys.info import size_of
from memory import UnsafePointer
import linux.syscalls as linux

@fieldwise_init
struct NumaArena[alignment: Int = 8, page_size: Int = linux.PageSize.THP_2MB](Movable):
    """NUMA-bound bump allocator.
    Memory is allocated via mmap and bound to the specified node via mbind(MPOL_BIND).
    Physical pages are allocated on first touch.

    Parameters:
        alignment: (Bytes)
        page_size: Page size
    """
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var size: Int
    var offset: Int
    var node: Int

    fn __init__(out self, node: Int, size: Int):
        """Args:
            node: NUMA node ID to bind memory to.
            size: Total arena size in bytes.
        """
        self.node = node
        self.size = size
        self.offset = 0
        self.base = mmap_numa[page_size=Self.page_size](size, node)
        if not self.base:
            self.size = 0

    fn __del__(deinit self):
        if self.base:
            _ = linux.sys_munmap(Int(self.base), self.size)

    fn __bool__(self) -> Bool:
        """ True if arena memory was allocated. """
        return self.base.__bool__()

    fn alloc[T: AnyType](mut self, count: Int = 1) -> UnsafePointer[T, MutAnyOrigin]:
        """Bump allocates a T aligned to arena's constraints

        Args:
            count: Number of T's to allocate.

        Returns:
            Pointer to uninitialized memory, or zero pointer if arena exhausted.
        """
        var bytes_needed = count * size_of[T]()
        var aligned_offset = align_up(self.offset, Self.alignment)
        if aligned_offset + bytes_needed > self.size:
            return UnsafePointer[T, MutAnyOrigin]()
        var ptr = self.base.offset(aligned_offset).bitcast[T]()
        self.offset = aligned_offset + bytes_needed
        return ptr

    fn mark(self) -> Int:
        """Current allocation offset in bytes."""
        return self.offset

    fn reset_to(mut self, watermark: Int):
        """Args:
            watermark: Offset (bytes)
        """
        if watermark >= 0 and watermark <= self.offset:
            self.offset = watermark

    fn reset(mut self):
        """Invalidate entire arena."""
        self.offset = 0

    fn remaining(self) -> Int:
        """Remaining (bytes)"""
        return self.size - self.offset

    fn used(self) -> Int:
        """Allocated (bytes)"""
        return self.offset

    fn verify_placement(self) -> Bool:
        """Debugging -> Verify first allocated page resides on expected NUMA node."""
        if not self.base or self.offset == 0:
            return True
        var status = linux.sys_move_pages_query(Int(self.base))
        if status < 0:
            return False
        return status == self.node

# === ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ===
# Implementation Details \/
# === ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ===

fn mmap_numa_impl[
    prot: Int, flags: Int, use_thp: Bool = False
](size: Int, node: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    var addr = linux.sys_mmap[prot=prot, flags=flags](0, size)
    if addr < 0:
        return UnsafePointer[UInt8, MutAnyOrigin]()
    @parameter
    if use_thp:
        _ = linux.sys_madvise[linux.Madvise.HUGEPAGE](addr, size)
    var nodemask = UInt64(1) << node
    var bind_result = linux.sys_mbind[policy=linux.Mempolicy.BIND](addr, size, nodemask)
    if bind_result < 0:
        _ = linux.sys_munmap(addr, size)
        return UnsafePointer[UInt8, MutAnyOrigin]()
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)

fn mmap_numa[
    prot: Int = linux.Prot.RW,
    page_size: Int = linux.PageSize.THP_2MB,
](size: Int, node: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Allocate anonymous memory bound to a specific NUMA node.

    Parameters:
        prot: Prot flags
        page_size: PageSize flags

    Args:
        size: (Bytes)
        node: Target NUMA node.

    Returns:
        Pointer to mapped memory, or zero pointer on failure.
    """
    comptime base_flags = linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS | linux.MapFlag.NORESERVE
    @parameter
    if page_size == linux.PageSize.EXPLICIT_2MB:
        comptime flags = base_flags | linux.MapFlag.HUGETLB | linux.MapFlag.HUGE_2MB
        return mmap_numa_impl[prot, flags](size, node)
    elif page_size == linux.PageSize.EXPLICIT_1GB:
        comptime flags = base_flags | linux.MapFlag.HUGETLB | linux.MapFlag.HUGE_1GB
        return mmap_numa_impl[prot, flags](size, node)
    elif page_size == linux.PageSize.THP_2MB:
        return mmap_numa_impl[prot, base_flags, use_thp=True](size, node)
    else:
        return mmap_numa_impl[prot, base_flags](size, node)

fn align_up(value: Int, alignment: Int) -> Int:
    return (value + alignment - 1) & ~(alignment - 1)
