"""Shared atomic modeling primitives.

Typed refs carry static tensor semantics through layout/build time.
Repeated[T] carries placement topology.
SectionBuilder emits typed state/aux refs and derives byte counts from
reservation side effects.
"""

from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, StaticView, DynamicView, CacheView, ScratchView,
    DEFAULT_ALIGNMENT, WeightDesc, HOST_RANK, DISTRIBUTED,
)
from modeling.linear_borrow_pool import ScratchLease, ScratchPool


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


@fieldwise_init
struct TensorRef[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Typed offset for one materialized tensor family."""
    var offset: Int

    @always_inline
    def bound(self, base: Int) -> StaticView[Self.E, Self.S]:
        return StaticView[Self.E, Self.S](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))

    @always_inline
    def bound_dyn(self, base: Int, seq_len: Int) -> DynamicView[Self.E, Self.S]:
        return DynamicView[Self.E, Self.S](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset),
            seq_len)

    @always_inline
    def bound_cache(self, base: Int) -> CacheView[Self.E, Self.S]:
        return CacheView[Self.E, Self.S](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))

    @always_inline
    def bound_row(self, base: Int, row: Int) -> StaticView[Self.E, Shape[1, Self.S.M]]:
        """View a single row at runtime offset. Used for per-position table
        lookups (rope cos/sin tables, etc.) where the row index is dynamic
        but the column width is comptime-fixed by the slot's Shape."""
        return StaticView[Self.E, Shape[1, Self.S.M]](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset
                    + row * Self.S.M * Self.E.ELEMENT_BYTES))

    @always_inline
    def addr(self, base: Int) -> Int:
        return base + self.offset


comptime SlotOffset[E: Encoding, S: ShapeLike] = TensorRef[E, S]


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
        comptime size = S.bytes[E]()
        var off = self.cursor
        self.cursor += size
        return SlotOffset[E, S](off)

    @always_inline
    def reserve_bytes(mut self, nbytes: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
        self.align(alignment)
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def bytes(self) -> Int:
        return self.cursor


# =============================================================================
# Layer builder — cursor-based weight catalog emitter
# =============================================================================


@fieldwise_init
struct LayerBuilder(Movable):
    var tp: Int
    var cursor: Int
    var layer_prefix: String
    var layer_base: Int

    def __init__(out self, tp: Int, prefix: String, layer_base: Int):
        self.tp = tp
        self.cursor = 0
        self.layer_prefix = prefix
        self.layer_base = layer_base

    @always_inline
    def emit_shape[E: Encoding, S: ShapeLike](mut self,
            mut entries: List[WeightDesc],
            suffix: String,
            quantizable: Bool = False,
            target_rank: Int = DISTRIBUTED) -> TensorRef[E, S]:
        comptime alloc = S.bytes[E]()
        var off = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=E.DTYPE, element_bytes=E.ELEMENT_BYTES,
            global_rows=S.GLOBAL_N, global_cols=S.GLOBAL_M,
            local_rows=S.N, local_cols=S.M,
            data_rows=S.DATA_N, data_cols=S.DATA_M,
            quantizable=quantizable, absorbed=False,
            target_rank=target_rank,
        ))
        return TensorRef[E, S](off)

    @always_inline
    def qs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                        target_rank: Int = DISTRIBUTED) -> TensorRef[I8, S]:
        return self.emit_shape[I8, S](entries, suffix, True, target_rank)

    @always_inline
    def fs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                        target_rank: Int = DISTRIBUTED) -> TensorRef[F32, S]:
        return self.emit_shape[F32, S](entries, suffix, False, target_rank)

    @always_inline
    def bfs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                         target_rank: Int = DISTRIBUTED) -> TensorRef[BF16, S]:
        return self.emit_shape[BF16, S](entries, suffix, False, target_rank)

    @always_inline
    def colsum_slot[E: Encoding, S: ShapeLike](mut self) -> SlotOffset[E, S]:
        """Typed colsum reservation — returns a SlotOffset for view-based
        access."""
        comptime nbytes = S.bytes[E]()
        var off = self.cursor
        self.cursor += nbytes
        return SlotOffset[E, S](off)
