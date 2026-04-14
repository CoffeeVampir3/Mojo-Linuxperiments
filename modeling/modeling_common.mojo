"""Shared atomic modeling primitives.

Typed refs carry static tensor semantics through layout/build time.
Repeated[T] carries placement topology.
SectionBuilder emits typed state/aux refs and derives byte counts from
reservation side effects.
"""

from modeling.model_spec import (
    Encoding, Shaped,
    ShapeLike, Mat, Bound, DynView, CacheView,
    I8, F32,
    DEFAULT_ALIGNMENT,
)


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


@fieldwise_init
struct SlotView[E: Encoding, S: ShapeLike](Encoding, Shaped, Copyable, Movable):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.N
    comptime COLS = Self.S.M
    var ptr: Int

    @always_inline
    def as_bound(self) -> Bound[Mat[Self.E, Self.S.N, Self.S.M]]:
        return Bound[Mat[Self.E, Self.S.N, Self.S.M]](self.ptr)

    @always_inline
    def as_cache(self) -> CacheView[Mat[Self.E, Self.S.N, Self.S.M]]:
        return CacheView[Mat[Self.E, Self.S.N, Self.S.M]](self.ptr)


@fieldwise_init
struct SlotOffset[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Typed offset for one materialized tensor family."""
    var offset: Int

    @always_inline
    def bind(self, base: Int) -> SlotView[Self.E, Self.S]:
        return SlotView[Self.E, Self.S](base + self.offset)

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


@fieldwise_init
struct QOffset[DS: ShapeLike, SS: ShapeLike](Copyable, ImplicitlyCopyable):
    """Quantized weight atoms: i8 data + f32 scale."""
    var data: SlotOffset[I8, Self.DS]
    var scale: SlotOffset[F32, Self.SS]


@fieldwise_init
struct QOffset3[DS: ShapeLike, SS: ShapeLike, CS: ShapeLike](Copyable, ImplicitlyCopyable):
    """Quantized weight atoms: i8 data + f32 scale + f32 colsum."""
    var data: SlotOffset[I8, Self.DS]
    var scale: SlotOffset[F32, Self.SS]
    var colsum: SlotOffset[F32, Self.CS]


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
