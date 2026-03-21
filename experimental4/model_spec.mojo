# ===----------------------------------------------------------------------=== #
# model_spec.mojo — Composable trait vocabulary for model structure
#
# Two carrier traits (Encoding, Shaped) and two producer traits
# (DimStrategy, ShardStrategy) compose into Slot — the universal
# buffer descriptor. Any function chooses its trait bound to see
# exactly the information it needs, nothing more.
# ===----------------------------------------------------------------------=== #


# ===--- Carrier traits ---=== #

trait Encoding:
    """What is stored: dtype and element width."""
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Shaped:
    """How much is stored: rows and cols (post-sharding local dims)."""
    comptime ROWS: Int
    comptime COLS: Int


# ===--- Encoding instances ---=== #

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


# ===--- Dimension strategy atoms ---=== #

trait DimStrategy:
    """How one dimension maps under tensor parallelism."""
    @staticmethod
    fn local(d: Int, tp: Int) -> Int: ...

struct Divide(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int:
        return d // tp

struct Keep(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int:
        return d


# ===--- Composed sharding ---=== #

trait ShardStrategy:
    """How a 2D buffer is distributed across TP ranks."""
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: ...
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: ...

struct Shard2D[Row: DimStrategy, Col: DimStrategy](ShardStrategy):
    """Composes two DimStrategy atoms into a full 2D strategy."""
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int:
        return Self.Row.local(r, tp)
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int:
        return Self.Col.local(c, tp)

comptime RowShard = Shard2D[Divide, Keep]
comptime ColShard = Shard2D[Keep, Divide]
comptime Replicated = Shard2D[Keep, Keep]


# ===--- Composition hub ---=== #

struct Slot[E: Encoding, S: ShardStrategy, rows: Int, cols: Int, tp: Int](
    Encoding, Shaped
):
    """Composes encoding + shape + sharding + tp into local dims.
    Conforms to Encoding & Shaped — the universal sizing interface."""
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)


# ===--- Placed weight slot ---=== #

trait Placed:
    """Arena offset and pre-sharding global dims for a weight slot."""
    comptime OFFSET: Int
    comptime GLOBAL_ROWS: Int
    comptime GLOBAL_COLS: Int

trait Named:
    """Tensor name (or suffix) for checkpoint lookup."""
    comptime NAME: StaticString

struct PlacedSlot[
    E: Encoding, S: ShardStrategy,
    rows: Int, cols: Int, tp: Int, offset: Int,
    name: StringLiteral,
](Encoding, Shaped, Placed, Named):
    """Self-describing weight slot: encoding, sharding, placement, and name."""
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)
    comptime OFFSET = Self.offset
    comptime GLOBAL_ROWS = Self.rows
    comptime GLOBAL_COLS = Self.cols
    comptime NAME: StaticString = Self.name


# ===--- Role marker traits ---=== #

trait Scale(Encoding, Shaped):
    """Marker: this buffer is a dequantization scale."""
    ...

struct ScaleSlot[E: Encoding, S: ShardStrategy, rows: Int, cols: Int, tp: Int](
    Scale
):
    """A Slot that is marked as a scale. Identical mechanics, distinct role."""
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)


# ===--- Free functions on trait bounds ---=== #

fn byte_count[T: Encoding & Shaped]() -> Int:
    """Universal byte count for any Encoding & Shaped."""
    return T.ROWS * T.COLS * T.ELEMENT_BYTES


# ===--- Config traits: independent architectural concerns ---=== #

trait Dims:
    """Core transformer dimensions."""
    comptime HIDDEN: Int
    comptime NUM_LAYERS: Int

trait Attention:
    """Multi-head attention geometry."""
    comptime NUM_HEADS: Int
    comptime HEAD_DIM: Int

trait GQA:
    """Grouped query attention."""
    comptime NUM_KV_HEADS: Int
    comptime KV_HIDDEN: Int
    comptime GQA_FACTOR: Int

trait FFN:
    """Feed-forward network width."""
    comptime INTERMEDIATE: Int

trait Vocab:
    """Token vocabulary."""
    comptime VOCAB_SIZE: Int
    comptime TIE_EMBEDDINGS: Bool

trait Sequence:
    """Context window."""
    comptime MAX_SEQ_LEN: Int

trait RoPEConfig:
    """Rotary position embedding parameters."""
    comptime ROPE_THETA: Float64

trait RMSNormConfig:
    """RMS normalization parameters."""
    comptime RMS_NORM_EPS: Float64
