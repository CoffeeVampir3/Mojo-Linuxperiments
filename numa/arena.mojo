from std.sys.info import size_of
from std.math import align_up
from std.memory import UnsafePointer
import linux.sys as linux

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

    def __init__(out self, node: Int, size: Int):
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

    def __del__(deinit self):
        if self.base:
            var sys = linux.linux_sys()
            _ = sys.sys_munmap(Int(self.base), self.size)

    def __bool__(self) -> Bool:
        """True if arena memory was allocated."""
        return self.base.__bool__()

    def alloc[T: AnyType](mut self, count: Int = 1) -> UnsafePointer[T, MutAnyOrigin]:
        """Bump allocates a T aligned to arena's constraints.

        Args:
            count: Number of T's to allocate.

        Returns:
            Pointer to uninitialized memory, or zero pointer if arena exhausted.
        """
        var bytes_needed = count * size_of[T]()
        var aligned_offset = align_up(self.offset, Self.alignment)
        if aligned_offset + bytes_needed > self.size:
            return UnsafePointer[T, MutAnyOrigin]()
        var ptr = (self.base + aligned_offset).bitcast[T]()
        self.offset = aligned_offset + bytes_needed
        return ptr

    def mark(self) -> Int:
        """Current allocation offset in bytes."""
        return self.offset

    def reset_to(mut self, watermark: Int):
        """Args:
            watermark: Offset (bytes)
        """
        if watermark >= 0 and watermark <= self.offset:
            self.offset = watermark

    def reset(mut self):
        """Invalidate entire arena."""
        self.offset = 0

    def prefault(self, offset: Int = 0, length: Int = -1) -> Bool:
        """Pre-fault pages via MADV_POPULATE_WRITE so first access doesn't page-fault.

        Args:
            offset: Byte offset into the arena to start prefaulting.
            length: Bytes to prefault. -1 (default) means from offset to end of arena.

        Returns:
            True on success, False on failure.
        """
        if not self.base:
            return False
        var actual_len = length if length >= 0 else self.size - offset
        if actual_len <= 0 or offset + actual_len > self.size:
            return False
        var sys = linux.linux_sys()
        var result = sys.sys_madvise[linux.Madvise.POPULATE_WRITE](
            Int(self.base) + offset, actual_len,
        )
        return result == 0

    def remaining(self) -> Int:
        """Remaining (bytes)."""
        return self.size - self.offset

    def used(self) -> Int:
        """Allocated (bytes)."""
        return self.offset

    def verify_placement(self) -> Bool:
        """Verify first allocated page resides on expected NUMA node."""
        if not self.base or self.offset == 0:
            return True
        var sys = linux.linux_sys()
        var status = sys.sys_move_pages_query(Int(self.base))
        if status < 0:
            return False
        return status == self.node

# === ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ===
# Implementation Details \/
# === ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ===

def mmap_numa_impl[
    prot: Int, flags: Int, use_thp: Bool = False
](size: Int, node: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    var sys = linux.linux_sys()
    var addr = sys.sys_mmap[prot=prot, flags=flags](0, size)
    if addr < 0:
        return UnsafePointer[UInt8, MutAnyOrigin]()
    comptime if use_thp:
        _ = sys.sys_madvise[linux.Madvise.HUGEPAGE](addr, size)
    var nodemask = UInt64(1) << UInt64(node)
    var bind_result = sys.sys_mbind[policy=linux.Mempolicy.BIND](addr, size, nodemask)
    if bind_result < 0:
        _ = sys.sys_munmap(addr, size)
        return UnsafePointer[UInt8, MutAnyOrigin]()
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)

def mmap_numa[
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
    comptime if page_size == linux.PageSize.EXPLICIT_2MB:
        comptime flags = base_flags | linux.MapFlag.HUGETLB | linux.MapFlag.HUGE_2MB
        return mmap_numa_impl[prot, flags](size, node)
    elif page_size == linux.PageSize.EXPLICIT_1GB:
        comptime flags = base_flags | linux.MapFlag.HUGETLB | linux.MapFlag.HUGE_1GB
        return mmap_numa_impl[prot, flags](size, node)
    elif page_size == linux.PageSize.THP_2MB:
        return mmap_numa_impl[prot, base_flags, use_thp=True](size, node)
    else:
        return mmap_numa_impl[prot, base_flags](size, node)
