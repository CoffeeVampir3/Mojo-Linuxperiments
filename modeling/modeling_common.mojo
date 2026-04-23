from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, StaticView, DynamicView, ScratchView,
    DEFAULT_ALIGNMENT, WeightDesc, HOST_RANK, DISTRIBUTED,
    align_up,
)
from modeling.linear_borrow_pool import ScratchLease, ScratchPool


@fieldwise_init
struct ArenaLayout(Copyable, ImplicitlyCopyable):
    """Common arena metadata shared by every model topology.

    `base` is the per-rank arena start address. The sizing fields describe
    the layout the loader/runtime expects: distributed (weights) +
    state (activations, KV cache, rope, scratch) form the main arena;
    host is a separate single-rank region for shared artifacts like embed.
    """
    var base: Int
    var distributed_bytes: Int
    var state_bytes: Int
    var host_bytes: Int
    var scratch_off: Int
    var scratch_capacity: Int

    def bind(self, new_base: Int) -> Self:
        var t = self
        t.base = new_base
        return t

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    @always_inline
    def scratch_base(self) -> Int:
        return self.base + self.scratch_off


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
    def reserve[E: Encoding, S: ShapeLike](mut self) -> TensorRef[E, S]:
        self.align()
        comptime size = S.bytes[E]()
        var off = self.cursor
        self.cursor += size
        return TensorRef[E, S](off)

    @always_inline
    def reserve_bytes(mut self, nbytes: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
        self.align(alignment)
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def bytes(self) -> Int:
        return self.cursor


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
        var off = align_up(self.cursor)
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=E.DTYPE, element_bytes=E.ELEMENT_BYTES,
            global_rows=S.GLOBAL_N, global_cols=S.GLOBAL_M,
            local_rows=S.N, local_cols=S.M,
            data_rows=S.DATA_N, data_cols=S.DATA_M,
            quantizable=quantizable,
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
    def colsum_slot[E: Encoding, S: ShapeLike](mut self) -> TensorRef[E, S]:
        """Typed colsum reservation — returns a TensorRef for view-based
        access."""
        comptime nbytes = S.bytes[E]()
        var off = self.cursor
        self.cursor += nbytes
        return TensorRef[E, S](off)

    @always_inline
    def reserve_block(mut self, nbytes: Int) -> Int:
        """Reserve an aligned byte block for sub-slab emission. Used when a
        single contiguous region is later populated by multiple per-element
        WeightDescs (e.g. per-expert MoE weights in one arena region)."""
        var off = align_up(self.cursor)
        self.cursor = off + nbytes
        return off
