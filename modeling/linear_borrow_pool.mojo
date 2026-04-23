from std.sys.info import size_of
from std.os import abort

from modeling.model_spec import Encoding, Shaped, Shape, ShapeLike, ScratchView


comptime SCRATCH_LEASE_ALIGNMENT = 64


@always_inline
def scratch_block_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_LEASE_ALIGNMENT - 1) // SCRATCH_LEASE_ALIGNMENT) * SCRATCH_LEASE_ALIGNMENT


@always_inline
def scratch_lease_bytes[T: AnyType, count: Int]() -> Int:
    comptime raw_byte_size = count * size_of[T]()
    return scratch_block_bytes[raw_byte_size]()


@explicit_destroy
struct ScratchLease(Movable):
    """Linear token representing a borrowed offset range in the scratch pool.

    offset: byte offset from the scratch base (same for every rank).
    Must be consumed via ^.release().
    """
    var offset: Int
    var byte_size: Int
    var pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, offset: Int, byte_size: Int, pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]):
        self.offset = offset
        self.byte_size = byte_size
        self.pool_offset_ptr = pool_offset_ptr

    def release(deinit self):
        """Return the offset range to the pool. LIFO — must be top of stack."""
        var top = self.pool_offset_ptr[]
        if self.offset + self.byte_size != top:
            print("ScratchPool: non-LIFO release detected. lease_offset=",
                  self.offset, " byte_size=", self.byte_size,
                  " expected_top=", self.offset + self.byte_size,
                  " actual_top=", top)
            abort("ScratchPool: non-LIFO release")
        self.pool_offset_ptr[] -= self.byte_size

    @always_inline
    def as_ptr[
        o: MutOrigin, //,
        T: AnyType,
    ](
        ref [o] self, scratch_base: Int, element_offset: Int = 0,
    ) -> UnsafePointer[T, o]:
        """Typed pointer into the lease's region.

        Origin is inferred from the lease reference, so release-then-use
        on any pointer derived from this call trips the compile-time
        lifetime checker. Used for packed-struct buffers (router
        candidates, top-K results, etc.) where a 2D view makes no sense
        but typed pointer access is still appropriate.
        """
        return UnsafePointer[T, o](
            unsafe_from_address=scratch_base + self.offset
                + element_offset * size_of[T]())

    @always_inline
    def view[
        o: MutOrigin, //,
        E: Encoding, S: ShapeLike,
    ](
        ref [o] self, scratch_base: Int, seq_len: Int,
        element_offset: Int = 0,
    ) -> ScratchView[E, S, o]:
        """Bridge from scratch lease to typed view.

        Origin comes from the lease reference: release-then-use on any
        derived view fails at compile time.

        element_offset: element-unit offset into the lease (e.g., to split
        a [K; V] buffer into separate K and V views from one lease).
        """
        return ScratchView[E, S, o](
            UnsafePointer[Scalar[E.DTYPE], o](
                unsafe_from_address=(scratch_base + self.offset
                    + element_offset * E.ELEMENT_BYTES)),
            seq_len)


struct ScratchPool(Movable):
    """Offset-only bump allocator. One per model, not per rank.

    borrow[T, count]() returns a ScratchLease with a 64-byte-aligned offset.
    Each rank materializes a pointer: rv.scratch_base() + lease.offset.
    Cumulative overflow aborts. high_water canary prints on new peaks.

    TODO: if Mojo gains comptime counting, replace runtime cumulative
    check with a static assert.
    """
    var capacity: Int
    var offset: Int
    var high_water: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.offset = 0
        self.high_water = 0

    def borrow[T: AnyType, count: Int](mut self) -> ScratchLease:
        """Borrow `count` elements of type T. Returns a 64B block lease."""
        comptime byte_size = scratch_lease_bytes[T, count]()
        var lease_offset = self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("ScratchPool: cumulative borrows exceed capacity")
        if self.offset > self.high_water:
            self.high_water = self.offset
            print("scratch: new peak " + String(self.high_water)
                  + " / " + String(self.capacity) + " bytes")
        return ScratchLease(
            offset=lease_offset, byte_size=byte_size,
            pool_offset_ptr=UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))
            ),
        )
