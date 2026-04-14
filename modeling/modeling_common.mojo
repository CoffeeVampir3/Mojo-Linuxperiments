"""Shared atomic modeling primitives.

Typed refs carry static tensor semantics through layout/build time.
Repeated[T] carries placement topology.
SectionBuilder emits typed state/aux refs and derives byte counts from
reservation side effects.
"""

from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, Shaped,
    Shape, ShapeLike, Mat, Bound, DynView, CacheView,
    I8, F32,
    DEFAULT_ALIGNMENT,
)
from experimental.linear_borrow_pool import ScratchLease, ScratchPool


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


comptime StaticTensorView[E: Encoding, S: ShapeLike] = Bound[Mat[E, S.N, S.M]]
comptime DynamicTensorView[E: Encoding, S: ShapeLike] = DynView[Mat[E, S.N, S.M]]


@always_inline
def static_tensor_view[E: Encoding, S: ShapeLike](
    base: Int, slot: TensorRef[E, S],
) -> StaticTensorView[E, S]:
    return StaticTensorView[E, S](base + slot.offset)


@always_inline
def dynamic_tensor_view[E: Encoding, S: ShapeLike](
    base: Int, slot: TensorRef[E, S], seq_len: Int,
) -> DynamicTensorView[E, S]:
    return DynamicTensorView[E, S](base + slot.offset, seq_len)


@always_inline
def scratch_tensor_view[E: Encoding, rows: Int, cols: Int](
    scratch_base: Int, read lease: ScratchLease, seq_len: Int,
) -> DynamicTensorView[E, Shape[rows, cols]]:
    return DynamicTensorView[E, Shape[rows, cols]](
        scratch_base + lease.offset, seq_len)


@always_inline
def scratch_ptr[T: AnyType](
    scratch_base: Int, read lease: ScratchLease,
) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](
        unsafe_from_address=scratch_base + lease.offset)


@explicit_destroy
struct BorrowedScratchTensor[E: Encoding, rows: Int, cols: Int](Movable):
    var ptr: UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin]
    var seq_len: Int
    var lease: ScratchLease

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin],
        seq_len: Int,
        var lease: ScratchLease,
    ):
        self.ptr = ptr
        self.seq_len = seq_len
        self.lease = lease^

    @always_inline
    def view(self) -> DynamicTensorView[Self.E, Shape[Self.rows, Self.cols]]:
        return DynamicTensorView[Self.E, Shape[Self.rows, Self.cols]](
            Int(self.ptr), self.seq_len)

    def release(deinit self):
        self.lease^.release()


@always_inline
def borrow_scratch_tensor[E: Encoding, rows: Int, cols: Int](
    mut pool: ScratchPool,
    scratch_base: Int,
    seq_len: Int,
) -> BorrowedScratchTensor[E, rows, cols]:
    var lease = pool.borrow[Scalar[E.DTYPE], rows * cols]()
    return BorrowedScratchTensor[E, rows, cols](
        UnsafePointer[Scalar[E.DTYPE], MutAnyOrigin](
            unsafe_from_address=scratch_base + lease.offset
        ),
        seq_len,
        lease^,
    )


@fieldwise_init
struct TensorRef[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Typed offset for one materialized tensor family."""
    var offset: Int

    @always_inline
    def bound(self, base: Int) -> Bound[Mat[Self.E, Self.S.N, Self.S.M]]:
        return Bound[Mat[Self.E, Self.S.N, Self.S.M]](base + self.offset)

    @always_inline
    def dyn(self, base: Int, seq_len: Int) -> DynView[Mat[Self.E, Self.S.N, Self.S.M]]:
        return DynView[Mat[Self.E, Self.S.N, Self.S.M]](base + self.offset, seq_len)

    @always_inline
    def cache(self, base: Int) -> CacheView[Mat[Self.E, Self.S.N, Self.S.M]]:
        return CacheView[Mat[Self.E, Self.S.N, Self.S.M]](base + self.offset)

    @always_inline
    def addr(self, base: Int) -> Int:
        return base + self.offset


comptime SlotOffset[E: Encoding, S: ShapeLike] = TensorRef[E, S]


@fieldwise_init
struct QOffset[DS: ShapeLike, SS: ShapeLike](Copyable, ImplicitlyCopyable):
    """Quantized weight atoms: i8 data + f32 scale."""
    var data: SlotOffset[I8, Self.DS]
    var scale: SlotOffset[F32, Self.SS]


@fieldwise_init
struct OpaqueSlot[bytes: Int](Copyable, ImplicitlyCopyable):
    """Addressable non-matrix region with a fixed byte extent."""
    var offset: Int

    @always_inline
    def addr(self, base: Int) -> Int:
        return base + self.offset


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable](Copyable, ImplicitlyCopyable):
    """Repeated topology: proto identity + off/stride/count placement."""
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    @always_inline
    def base(self, arena_base: Int, idx: Int) -> Int:
        return arena_base + self.off + idx * self.stride


struct SectionBuilder:
    """Typed cursor allocator for state and persistent aux sections."""
    var cursor: Int

    def __init__(out self):
        self.cursor = 0

    @always_inline
    def align(mut self, alignment: Int = DEFAULT_ALIGNMENT):
        self.cursor = align_up(self.cursor, alignment)

    @always_inline
    def reserve[E: Encoding, S: ShapeLike](mut self) -> SlotOffset[E, S]:
        self.align()
        comptime size = S.bytes_for[E.ELEMENT_BYTES]()
        var off = self.cursor
        self.cursor += size
        return SlotOffset[E, S](off)

    @always_inline
    def reserve_opaque[bytes: Int](mut self) -> OpaqueSlot[bytes]:
        self.align()
        var off = self.cursor
        self.cursor += bytes
        return OpaqueSlot[bytes](off)

    @always_inline
    def reserve_bytes(mut self, nbytes: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
        self.align(alignment)
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def bytes(self) -> Int:
        return self.cursor
