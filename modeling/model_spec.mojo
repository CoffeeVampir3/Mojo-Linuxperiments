from std.memory import UnsafePointer
from modeling.linear_borrow_pool import ScratchLease


# =============================================================================
# Packing strategy — must be defined first as Placed references PackFn
# =============================================================================

comptime PackFn = def(
    UnsafePointer[UInt8, MutAnyOrigin],  # src: row-major source
    UnsafePointer[UInt8, MutAnyOrigin],  # dst: packed destination
    Int, Int,                            # rows, cols
) thin -> None

def pack_noop(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    pass

trait PackingStrategy:
    comptime PACK_FN: PackFn

struct Unpacked(PackingStrategy):
    comptime PACK_FN = pack_noop


# =============================================================================
# Kernel tiling — composable per-dimension constraints
#
# Atomic traits: each declares one axis of the kernel's tile structure.
# Composed structs: Kernel2DTiling (row + col), Kernel3DTiling (+ panel).
# Consumers take trait bounds on the axes they need.
# New dimensions are additive — one trait per axis.
# =============================================================================

trait RowTiled:
    """Kernel tiles the row (N/output) dimension in steps of ROW_TILE."""
    comptime ROW_TILE: Int

trait ColTiled:
    """Kernel tiles the col (K/reduction) dimension in steps of COL_TILE."""
    comptime COL_TILE: Int

trait PanelHeight:
    """Kernel processes the activation (M) dimension in panels of PANEL."""
    comptime PANEL: Int

struct Kernel2DTiling[row_tile: Int, col_tile: Int](RowTiled, ColTiled, PanelHeight):
    comptime ROW_TILE = Self.row_tile
    comptime COL_TILE = Self.col_tile
    comptime PANEL = 1

struct Kernel3DTiling[row_tile: Int, col_tile: Int, panel: Int](RowTiled, ColTiled, PanelHeight):
    comptime ROW_TILE = Self.row_tile
    comptime COL_TILE = Self.col_tile
    comptime PANEL = Self.panel

comptime Untiled = Kernel2DTiling[1, 1]


# =============================================================================
# Weight classification
#
# Lifecycle traits — a weight tag declares the full pipeline disposition:
#   Quantizable      : quantized (FWHT + int8), per-row scale in output
#   Gamma             : quantized with gamma absorption from preceding norm
#   Passthrough       : copied through quantizer unchanged, loaded, used
#   Absorbed          : consumed during quantization (gamma source), absent
# =============================================================================

trait WeightTag: ...
trait Quantizable: ...
trait Gamma: ...
trait PassthroughTag: ...
trait Absorbed: ...

struct IsQuantizable(WeightTag, Quantizable): ...
struct IsGammaQuantizable(WeightTag, Quantizable, Gamma): ...
struct IsPassthrough(WeightTag, PassthroughTag): ...
struct IsAbsorbed(WeightTag, Absorbed): ...


# =============================================================================
# Core traits
# =============================================================================

trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait Placed:
    comptime OFFSET: Int
    comptime GLOBAL_ROWS: Int
    comptime GLOBAL_COLS: Int
    comptime PACK_FN: PackFn
    comptime TARGET_RANK: Int

trait Named:
    comptime NAME: StaticString

trait Dynamic:
    ...


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1


# =============================================================================
# Shape — single source of truth for tensor geometry under TP
#
# Encodes global dimensions, sharding, alignment, and tp degree. All derived
# quantities (local dims, data dims, padding, byte counts) are comptime.
# Downstream code (LayerBuilder, loader, kernels) reads from Shape instead
# of recomputing dimensions independently.
# =============================================================================

def _align_up(val: Int, a: Int) -> Int:
    return ((val + a - 1) // a) * a

trait ShapeLike:
    comptime GLOBAL_N: Int
    comptime GLOBAL_M: Int
    comptime DATA_N: Int
    comptime DATA_M: Int
    comptime N: Int
    comptime M: Int
    comptime ELEMS: Int

    @staticmethod
    def bytes_for[elem_bytes: Int]() -> Int: ...
    @staticmethod
    def row_bytes_for[elem_bytes: Int]() -> Int: ...
    @staticmethod
    def col_bytes_for[elem_bytes: Int]() -> Int: ...

struct Shape[
    global_n: Int, global_m: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    tp: Int = 1,
    align_n: Int = 1, align_m: Int = 1,
](ShapeLike):
    comptime GLOBAL_N = Self.global_n
    comptime GLOBAL_M = Self.global_m
    comptime DATA_N = Self.global_n // Self.tp if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.tp if Self.shard_m else Self.global_m
    comptime N = _align_up(Self.DATA_N, Self.align_n)
    comptime M = _align_up(Self.DATA_M, Self.align_m)
    comptime PAD_N = Self.N - Self.DATA_N
    comptime PAD_M = Self.M - Self.DATA_M
    comptime ELEMS = Self.N * Self.M

    @staticmethod
    def bytes_for[elem_bytes: Int]() -> Int:
        return Self.ELEMS * elem_bytes

    @staticmethod
    def row_bytes_for[elem_bytes: Int]() -> Int:
        """Bytes for one row: M elements."""
        return Self.M * elem_bytes

    @staticmethod
    def col_bytes_for[elem_bytes: Int]() -> Int:
        """Bytes for one column: N elements."""
        return Self.N * elem_bytes


# =============================================================================
# Mat — lightweight Encoding & Shaped for Bound / DynView / CacheView
#
# Carries only the local dimensions the kernel needs (ROWS, COLS) plus the
# dtype.  No sharding, no tp, no placement — those live in LayerBuilder /
# Shape / WeightDesc where the layout is computed.  Mat is what you hand to
# a kernel via Bound[Mat[...]] or DynView[Mat[...]].
# =============================================================================

struct Mat[E: Encoding, rows: Int, cols: Int](Encoding, Shaped):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.rows
    comptime COLS = Self.cols


# =============================================================================
# Legacy DimStrategy / ShardStrategy — used by SmolLM2 PlacedSlot chain
# =============================================================================

trait DimStrategy:
    @staticmethod
    def local(d: Int, tp: Int) -> Int: ...

struct Divide(DimStrategy):
    @staticmethod
    def local(d: Int, tp: Int) -> Int:
        return d // tp

struct Keep(DimStrategy):
    @staticmethod
    def local(d: Int, tp: Int) -> Int:
        return d


trait ShardStrategy:
    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int: ...
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int: ...

struct Shard2D[Row: DimStrategy, Col: DimStrategy](ShardStrategy):
    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int:
        return Self.Row.local(r, tp)
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int:
        return Self.Col.local(c, tp)

comptime RowShard = Shard2D[Divide, Keep]
comptime ColShard = Shard2D[Keep, Divide]
comptime Replicated = Shard2D[Keep, Keep]


# =============================================================================
# Placement locality
#
# target_rank on a PlacedSlot says where the loader is allowed to write the
# weight. The default (DISTRIBUTED) means every rank participates: replicated
# slots get a copy on each rank, sharded slots get their slice. A non-negative
# target_rank pins the slot to a single rank's arena — used for host-only
# weights (final norm, embed, lm head) that only one rank ever reads, and for
# per-expert MoE sharding where the rank is determined by expert ID.
# =============================================================================

comptime DISTRIBUTED = -1
comptime HOST_RANK = 0


struct Slot[E: Encoding, S: ShardStrategy, rows: Int, cols: Int, tp: Int](
    Encoding, Shaped
):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)

struct PlacedSlot[
    E: Encoding, S: ShardStrategy,
    rows: Int, cols: Int, tp: Int, offset: Int,
    name: StringLiteral,
    Tag: WeightTag = IsPassthrough,
    Packing: PackingStrategy = Unpacked,
    Tiling: RowTiled & ColTiled & PanelHeight = Untiled,
    target_rank: Int = DISTRIBUTED,
](
    Encoding, Shaped, Placed, Named, ShardStrategy,
    Quantizable where conforms_to(Tag, Quantizable),
    Gamma where conforms_to(Tag, Gamma),
    PassthroughTag where conforms_to(Tag, PassthroughTag),
    Absorbed where conforms_to(Tag, Absorbed),
):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)
    comptime OFFSET = Self.offset
    comptime GLOBAL_ROWS = Self.rows
    comptime GLOBAL_COLS = Self.cols
    comptime NAME: StaticString = Self.name
    comptime PACK_FN = Self.Packing.PACK_FN
    comptime TARGET_RANK = Self.target_rank

    @staticmethod
    def shard_rows(r: Int, n: Int) -> Int:
        return Self.S.shard_rows(r, n)
    @staticmethod
    def shard_cols(c: Int, n: Int) -> Int:
        return Self.S.shard_cols(c, n)


@fieldwise_init
struct Bound[T: Encoding & Shaped](Encoding, Shaped):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin](
                unsafe_from_address=self.ptr))

@fieldwise_init
struct DynView[T: Encoding & Shaped](Encoding, Shaped, Dynamic):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int
    var seq_len: Int

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin](
                unsafe_from_address=self.ptr))

@fieldwise_init
struct CacheView[T: Encoding & Shaped](Encoding, Shaped):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin](
                unsafe_from_address=self.ptr))

def bind[T: Encoding & Shaped & Placed & Named](base: Int) -> Bound[T]:
    return Bound[T](base + T.OFFSET)


def byte_count[T: Encoding & Shaped]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES

comptime DEFAULT_ALIGNMENT = 64

def next_offset[T: Encoding & Shaped & Placed, alignment: Int = DEFAULT_ALIGNMENT]() -> Int:
    comptime aligned = ((T.OFFSET + alignment - 1) // alignment) * alignment
    return aligned + byte_count[T]()


@fieldwise_init
struct WeightDesc(Copyable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_rows: Int
    var local_cols: Int
    var data_rows: Int
    var data_cols: Int
    var quantizable: Bool
    var absorbed: Bool
    var target_rank: Int

def weight_desc[T: Encoding & Shaped & Placed & Named](
    prefix: String = "", base: Int = 0,
) -> WeightDesc:
    comptime is_quantizable = conforms_to(T, Quantizable)
    comptime is_absorbed = conforms_to(T, Absorbed)
    return WeightDesc(
        name=prefix + String(T.NAME), arena_offset=base + T.OFFSET,
        dtype=T.DTYPE, element_bytes=T.ELEMENT_BYTES,
        global_rows=T.GLOBAL_ROWS, global_cols=T.GLOBAL_COLS,
        local_rows=T.ROWS, local_cols=T.COLS,
        data_rows=T.ROWS, data_cols=T.COLS,
        quantizable=is_quantizable,
        absorbed=is_absorbed,
        target_rank=T.TARGET_RANK,
    )


trait WeightIterable:
    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](): ...


# =============================================================================
# Quantizer tasks — one concrete variant per real operation.
#
# The quantizer's task list is a heterogeneous `List[Task]` where `Task`
# is the `Variant` over the five structs below. Each variant is
# self-describing: read the fields at the call site, you know the full
# scope of what hits disk for that weight. No `to_op()` indirection, no
# flag-bag struct, no "is this combination legitimate" check at runtime.
#
# Adding a variant is one new struct here + one arm in the dispatch
# match in quant/butterquant.mojo. Nothing else moves.
# =============================================================================


struct SourceFormat:
    """Vendor-side disk encoding for a quantizable tensor.

    A runtime tag that picks the `Converter` used to dequant source bytes
    into f32. The converter traits live in `quant/source_format.mojo`;
    this module only exposes the tag so modeling files can declare
    source formats without depending on the quantizer pipeline.
    """
    comptime BF16 = 0
    comptime F32 = 1
    comptime FP8_E4M3_BLOCK128 = 2


# --- Concrete task variants -------------------------------------------------


@fieldwise_init
struct Passthrough(Copyable, Movable, ImplicitlyCopyable):
    """Copy bytes from source to output unchanged.

    `expected_dtype` is a plan-time sanity gate: the planner fails hard
    if the on-disk tensor's dtype doesn't match. Catches the "thought
    this was bf16 but it's f32" class of mistake without coupling the
    passthrough path to any decode logic.
    """
    var name: String
    var expected_dtype: DType


@fieldwise_init
struct ButterquantI8PerRow(Copyable, Movable, ImplicitlyCopyable):
    """FWHT rotate at `block`, emit int8 weight + f32 per-row scale.

    Output entries: `<name>` int8, `<name>_scale` f32 shape (rows,).
    """
    var name: String
    var source: Int   # SourceFormat tag
    var block: Int    # FWHT rotation block (power of 2)


@fieldwise_init
struct ButterquantI8PerRowAbsorbed(Copyable, Movable, ImplicitlyCopyable):
    """Same as PerRow but first multiplies the weight row-wise by
    sqrt(|gamma|) where gamma is loaded from the tensor named by
    `gamma_src`. The runtime RMSNorm consuming this weight can then
    skip the gamma multiply.
    """
    var name: String
    var source: Int
    var block: Int
    var gamma_src: String


@fieldwise_init
struct ButterquantI8PerBlock(Copyable, Movable, ImplicitlyCopyable):
    """FWHT rotate at `block`, emit int8 weight + f32 per-block scale
    grid. One scale per (row, col_block) pair.

    Output entries: `<name>` int8, `<name>_scale` f32 shape (rows, cols/block).
    """
    var name: String
    var source: Int
    var block: Int


@fieldwise_init
struct ButterquantI8PerBlockAbsorbed(Copyable, Movable, ImplicitlyCopyable):
    """Gamma-absorbed PerBlock — absorbs sqrt(|gamma|) from `gamma_src`
    before FWHT + per-block absmax i8.
    """
    var name: String
    var source: Int
    var block: Int
    var gamma_src: String


# --- Task variant umbrella --------------------------------------------------


from std.utils.variant import Variant

comptime Task = Variant[
    Passthrough,
    ButterquantI8PerRow,
    ButterquantI8PerRowAbsorbed,
    ButterquantI8PerBlock,
    ButterquantI8PerBlockAbsorbed,
]


trait Dims:
    comptime HIDDEN: Int
    comptime NUM_LAYERS: Int

trait Attention:
    comptime NUM_HEADS: Int
    comptime HEAD_DIM: Int

trait GQA:
    comptime NUM_KV_HEADS: Int
    comptime KV_HIDDEN: Int
    comptime GQA_FACTOR: Int

trait FFN:
    comptime INTERMEDIATE: Int

trait Vocab:
    comptime VOCAB_SIZE: Int
    comptime TIE_EMBEDDINGS: Bool

trait Sequence:
    comptime MAX_SEQ_LEN: Int

trait RoPEConfig:
    comptime ROPE_THETA: Float64

trait RMSNormConfig:
    comptime RMS_NORM_EPS: Float64


# =============================================================================
# Logits view — non-owning read-only access to model output
# =============================================================================


@explicit_destroy
struct LogitsView[vocab: Int, dtype: DType = DType.bfloat16](Movable):
    """Owning view of one logit vector (VOCAB elements).

    Holds a ScratchLease — the scratch offset is reserved until this
    view is dropped. The caller must drop the previous LogitsView
    before calling forward() again.
    """
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var lease: ScratchLease

    def __init__(out self, ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin], var lease: ScratchLease):
        self.ptr = ptr
        self.lease = lease^

    def load_f32[width: Int](self, offset: Int) -> SIMD[DType.float32, width]:
        return (self.ptr + offset).load[width=width]().cast[DType.float32]()

    def release(deinit self):
        """Drop the view, returning the scratch offset to the pool."""
        self.lease^.release()
