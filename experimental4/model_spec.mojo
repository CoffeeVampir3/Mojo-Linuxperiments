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
    fn local(d: Int, tp: Int) -> Int: ...

struct Divide(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int:
        return d // tp

struct Keep(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int:
        return d


trait ShardStrategy:
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: ...
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: ...

struct Shard2D[Row: DimStrategy, Col: DimStrategy](ShardStrategy):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int:
        return Self.Row.local(r, tp)
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int:
        return Self.Col.local(c, tp)

comptime RowShard = Shard2D[Divide, Keep]
comptime ColShard = Shard2D[Keep, Divide]
comptime Replicated = Shard2D[Keep, Keep]

trait NodeLocal(ShardStrategy):
    ...

struct PrincipleNodeLocal(NodeLocal):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int:
        return r
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int:
        return c


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
](Encoding, Shaped, Placed, Named, ShardStrategy):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)
    comptime OFFSET = Self.offset
    comptime GLOBAL_ROWS = Self.rows
    comptime GLOBAL_COLS = Self.cols
    comptime NAME: StaticString = Self.name

    @staticmethod
    fn shard_rows(r: Int, n: Int) -> Int:
        return Self.S.shard_rows(r, n)
    @staticmethod
    fn shard_cols(c: Int, n: Int) -> Int:
        return Self.S.shard_cols(c, n)


# TODO: Bound currently only carries T (Encoding & Shaped). Once conditional struct
# conformance is supported, PlacedSlot will conditionally conform to NodeLocal,
# and S can fold back into T — eliminating the separate S parameter in callbacks.
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

fn bind[T: Encoding & Shaped & Placed & Named](base: Int) -> Bound[T]:
    return Bound[T](base + T.OFFSET)


fn byte_count[T: Encoding & Shaped]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES

comptime DEFAULT_ALIGNMENT = 64

fn offset_after[T: Encoding & Shaped, base: Int, alignment: Int = DEFAULT_ALIGNMENT]() -> Int:
    comptime aligned = ((base + alignment - 1) // alignment) * alignment
    return aligned + byte_count[T]()

fn next_offset[T: Encoding & Shaped & Placed, alignment: Int = DEFAULT_ALIGNMENT]() -> Int:
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

fn weight_desc[T: Encoding & Shaped & Placed & Named](
    prefix: String = "", base: Int = 0,
) -> WeightDesc:
    return WeightDesc(
        name=prefix + String(T.NAME), arena_offset=base + T.OFFSET,
        dtype=T.DTYPE, element_bytes=T.ELEMENT_BYTES,
        global_rows=T.GLOBAL_ROWS, global_cols=T.GLOBAL_COLS,
        local_rows=T.ROWS, local_cols=T.COLS,
    )


# TODO: S is passed separately until conditional struct conformance is supported,
# at which point PlacedSlot will conditionally conform to NodeLocal and S folds into T.
trait WeightIterable:
    @staticmethod
    fn for_each_weight[
        func: fn[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](): ...


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
