from std.memory import UnsafePointer
from experimental.linear_borrow_pool import ScratchLease


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
#   PerBlockQuantizable : quantized (FWHT + int8), per-FWHT-block scale in
#                        output. Used by the LM head and other consumers that
#                        want the tighter per-block dynamic range at the cost
#                        of an 11×-ish larger scale array.
#   Gamma             : quantized with gamma absorption from preceding norm
#   Passthrough       : copied through quantizer unchanged, loaded, used
#   Absorbed          : consumed during quantization (gamma source), absent
# =============================================================================

trait WeightTag: ...
trait Quantizable: ...
trait PerBlockQuantizable: ...
trait Gamma: ...
trait Passthrough: ...
trait Absorbed: ...

struct IsQuantizable(WeightTag, Quantizable): ...
struct IsGammaQuantizable(WeightTag, Quantizable, Gamma): ...
struct IsPerBlockQuantizable(WeightTag, PerBlockQuantizable): ...
struct IsPassthrough(WeightTag, Passthrough): ...
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

struct F16(Encoding):
    comptime DTYPE = DType.float16
    comptime ELEMENT_BYTES = 2

struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1


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
    PerBlockQuantizable where conforms_to(Tag, PerBlockQuantizable),
    Gamma where conforms_to(Tag, Gamma),
    Passthrough where conforms_to(Tag, Passthrough),
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

@fieldwise_init
struct DynView[T: Encoding & Shaped](Encoding, Shaped, Dynamic):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int
    var seq_len: Int

@fieldwise_init
struct CacheView[T: Encoding & Shaped](Encoding, Shaped):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int

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
    var quantizable: Bool
    var absorbed: Bool
    var target_rank: Int
    # NOTE: no `pack_fn` field. The loader never reads it; packing is done
    # by model-specific init code (e.g. `init_sliding_layer` / `init_full_layer`
    # in gemma_4_moe_butterquant_tp.mojo) via explicit `pack_at(...)` calls
    # after the raw bytes have been loaded. Storing a runtime function
    # pointer here was both dead code and triggered a Mojo backend codegen
    # crash (see repro_packfn.mojo) when constructed inline.

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
# Runtime quantizer task — one work unit per source weight.
#
# Replaces the WeightIterable-driven template dispatch in the quantizer.
# Each task is self-contained: the kind tells the driver which processing
# path to take, and gamma_src names the norm tensor to absorb into this
# weight (empty string for no absorption). The old ABSORBED / GAMMA_QUANTIZE
# two-task handshake is gone — gamma is an explicit edge from consumer to
# source, loaded lazily by the driver's one-slot cache.
# =============================================================================


# --- Quantization scheme variants ---

from std.utils import Variant


@fieldwise_init
struct QuantPassthrough(Copyable, Movable, ImplicitlyCopyable):
    """Weight copied unchanged at source dtype."""
    var tag: Int
    def __init__(out self):
        self.tag = 0


@fieldwise_init
struct RowQuantized(Copyable, Movable, ImplicitlyCopyable):
    """FWHT rotation + single absmax scale per row."""
    var rotation: Int


@fieldwise_init
struct BlockQuantized(Copyable, Movable, ImplicitlyCopyable):
    """FWHT rotation + per-block absmax scales."""
    var rotation: Int
    var scale_blk: Int


@fieldwise_init
struct SmoothBlockQuantized(Copyable, Movable):
    """Smooth split (sqrt(|gamma|)) + FWHT rotation + per-block absmax scales."""
    var rotation: Int
    var scale_blk: Int
    var smooth_src: String


comptime QuantScheme = Variant[
    QuantPassthrough,
    RowQuantized,
    BlockQuantized,
    SmoothBlockQuantized,
]


def quant_is_quantized(read s: QuantScheme) -> Bool:
    return not s.isa[QuantPassthrough]()


def quant_rotation(read s: QuantScheme) -> Int:
    if s.isa[RowQuantized]():
        return s[RowQuantized].rotation
    elif s.isa[BlockQuantized]():
        return s[BlockQuantized].rotation
    elif s.isa[SmoothBlockQuantized]():
        return s[SmoothBlockQuantized].copy().rotation
    return 0


def quant_scale_blocks(read s: QuantScheme, cols: Int) -> Int:
    """0 for passthrough, 1 for per-row, cols/blk for per-block."""
    if s.isa[RowQuantized]():
        return 1
    elif s.isa[BlockQuantized]():
        return cols // s[BlockQuantized].scale_blk
    elif s.isa[SmoothBlockQuantized]():
        return cols // s[SmoothBlockQuantized].copy().scale_blk
    return 0


def quant_smooth_source(read s: QuantScheme) -> String:
    if s.isa[SmoothBlockQuantized]():
        return s[SmoothBlockQuantized].copy().smooth_src
    return ""


@fieldwise_init
struct QuantizeTask(Copyable, Movable):
    var name: String
    var scheme: QuantScheme


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
