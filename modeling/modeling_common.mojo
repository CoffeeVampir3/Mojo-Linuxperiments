"""Shared atomic modeling primitives.

Typed refs carry static tensor semantics through layout/build time.
Repeated[T] carries placement topology.
SectionBuilder emits typed state/aux refs and derives byte counts from
reservation side effects.
"""

from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, Shaped,
    Shape, ShapeLike, Mat, StaticView, DynamicView, ScratchView,
    DEFAULT_ALIGNMENT, WeightDesc, HOST_RANK, DISTRIBUTED,
)
from modeling.linear_borrow_pool import ScratchLease, ScratchPool


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


comptime StaticTensorView[E: Encoding, S: ShapeLike] = StaticView[Mat[E, S.N, S.M]]
comptime DynamicTensorView[E: Encoding, S: ShapeLike] = DynamicView[Mat[E, S.N, S.M]]


@always_inline
def static_tensor_view[E: Encoding, S: ShapeLike](
    base: Int, slot: TensorRef[E, S],
) -> StaticTensorView[E, S]:
    return StaticTensorView[E, S](
        UnsafePointer[Scalar[E.DTYPE], MutAnyOrigin](
            unsafe_from_address=base + slot.offset))


@always_inline
def dynamic_tensor_view[E: Encoding, S: ShapeLike](
    base: Int, slot: TensorRef[E, S], seq_len: Int,
) -> DynamicTensorView[E, S]:
    return DynamicTensorView[E, S](
        UnsafePointer[Scalar[E.DTYPE], MutAnyOrigin](
            unsafe_from_address=base + slot.offset),
        seq_len)


@explicit_destroy
struct BorrowedScratchTensor[E: Encoding, rows: Int, cols: Int](Movable):
    """Linear wrapper: lease + base + seq_len. .view() returns a MutAnyOrigin
    DynamicView for convenience. Release consumes the wrapper linearly."""
    var lease: ScratchLease
    var base: Int
    var seq_len: Int

    def __init__(out self, var lease: ScratchLease, base: Int, seq_len: Int):
        self.lease = lease^
        self.base = base
        self.seq_len = seq_len

    @always_inline
    def view(self) -> DynamicView[Mat[Self.E, Self.rows, Self.cols]]:
        return DynamicView[Mat[Self.E, Self.rows, Self.cols]](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=self.base + self.lease.offset),
            self.seq_len)

    @always_inline
    def addr(self) -> Int:
        return self.base + self.lease.offset

    def release(deinit self):
        self.lease^.release()


@always_inline
def borrow_scratch_tensor[E: Encoding, rows: Int, cols: Int](
    mut pool: ScratchPool, base: Int, seq_len: Int,
) -> BorrowedScratchTensor[E, rows, cols]:
    var lease = pool.borrow[Scalar[E.DTYPE], rows * cols]()
    return BorrowedScratchTensor[E, rows, cols](lease^, base, seq_len)


@always_inline
def scratch_ptr[T: AnyType](
    scratch_base: Int, read lease: ScratchLease,
) -> UnsafePointer[T, MutAnyOrigin]:
    """Lift a lease to a typed pointer; used where the caller needs a raw
    UnsafePointer rather than a view (e.g., pass-through to LogitsView or
    to sub-kernels that take raw pointers)."""
    return UnsafePointer[T, MutAnyOrigin](
        unsafe_from_address=scratch_base + lease.offset)


@fieldwise_init
struct TensorRef[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Typed offset for one materialized tensor family."""
    var offset: Int

    @always_inline
    def bound(self, base: Int) -> StaticView[Mat[Self.E, Self.S.N, Self.S.M]]:
        return StaticView[Mat[Self.E, Self.S.N, Self.S.M]](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))

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
        comptime size = S.bytes_for[E.ELEMENT_BYTES]()
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
# Layer sharding modes
# =============================================================================


struct LayerShard:
    comptime ROW  = 0
    comptime COL  = 1
    comptime REPL = 2
    comptime HOST = 3


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
    def emit(mut self,
            mut entries: List[WeightDesc],
            suffix: String,
            global_rows: Int, global_cols: Int,
            dtype: DType, element_bytes: Int,
            shard: Int, quantizable: Bool = False,
            target_rank: Int = DISTRIBUTED) -> Int:
        var local_rows = global_rows // self.tp if shard == LayerShard.ROW else global_rows
        var local_cols = global_cols // self.tp if shard == LayerShard.COL else global_cols
        var effective_target = target_rank
        if shard == LayerShard.HOST:
            effective_target = HOST_RANK
        var alloc = local_rows * local_cols * element_bytes
        var off = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=dtype, element_bytes=element_bytes,
            global_rows=global_rows, global_cols=global_cols,
            local_rows=local_rows, local_cols=local_cols,
            data_rows=local_rows, data_cols=local_cols,
            quantizable=quantizable, absorbed=False,
            target_rank=effective_target,
        ))
        return off

    @always_inline
    def emit_shape[S: ShapeLike, element_bytes: Int](mut self,
            mut entries: List[WeightDesc],
            suffix: String,
            dtype: DType,
            quantizable: Bool = False,
            target_rank: Int = DISTRIBUTED) -> Int:
        comptime alloc = S.bytes_for[element_bytes]()
        var off = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=dtype, element_bytes=element_bytes,
            global_rows=S.GLOBAL_N, global_cols=S.GLOBAL_M,
            local_rows=S.N, local_cols=S.M,
            data_rows=S.DATA_N, data_cols=S.DATA_M,
            quantizable=quantizable, absorbed=False,
            target_rank=target_rank,
        ))
        return off

    @always_inline
    def qs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                        target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit_shape[S, 1](entries, suffix, DType.int8, True, target_rank)

    @always_inline
    def fs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                        target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit_shape[S, 4](entries, suffix, DType.float32, False, target_rank)

    @always_inline
    def bfs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                         target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit_shape[S, 2](entries, suffix, DType.bfloat16, False, target_rank)

    @always_inline
    def colsum(mut self, nbytes: Int) -> Int:
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def colsum_slot[E: Encoding, S: ShapeLike](mut self) -> SlotOffset[E, S]:
        """Typed colsum reservation — returns a SlotOffset for view-based
        access. Replaces `colsum(nbytes)` at sites that pass the result to
        trait-composed dispatch kernels."""
        comptime nbytes = S.bytes_for[E.ELEMENT_BYTES]()
        var off = self.cursor
        self.cursor += nbytes
        return SlotOffset[E, S](off)

    @always_inline
    def q(mut self, mut entries: List[WeightDesc], suffix: String,
          rows: Int, cols: Int, shard: Int,
          target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.int8, 1, shard, True, target_rank)

    @always_inline
    def f(mut self, mut entries: List[WeightDesc], suffix: String,
          rows: Int, cols: Int, shard: Int,
          target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.float32, 4, shard, False, target_rank)

    @always_inline
    def bf(mut self, mut entries: List[WeightDesc], suffix: String,
           rows: Int, cols: Int, shard: Int,
           target_rank: Int = DISTRIBUTED) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.bfloat16, 2, shard, False, target_rank)
