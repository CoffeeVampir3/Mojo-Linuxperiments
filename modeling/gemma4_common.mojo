"""Gemma 4 shared infrastructure — config, layout builder, typed view helpers."""

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32,
    HOST_RANK, DISTRIBUTED,
    Mat, Bound, DynView, CacheView,
    ShapeLike, WeightDesc,
    DEFAULT_ALIGNMENT,
)


# =============================================================================
# Architecture config — shared across bf16 and ButterQuant models
# =============================================================================


struct Gemma4BaseConfig:
    comptime HIDDEN = 2816
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 16

    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime Q_DIM_SLIDING = 4096
    comptime KV_DIM_SLIDING = 2048

    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime Q_DIM_FULL = 8192
    comptime KV_DIM_FULL = 1024

    comptime INTERMEDIATE = 2112
    comptime MOE_GATE_UP_FUSED = 1408
    comptime MOE_INTERMEDIATE = 704
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 262144
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime LOGIT_SOFTCAP = 30.0


# =============================================================================
# Layer routing
# =============================================================================


@always_inline
def is_full_layer(layer_idx: Int) -> Bool:
    return (layer_idx + 1) % 6 == 0


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
            shard: Int, quantizable: Bool = False) -> Int:
        var local_rows = global_rows // self.tp if shard == LayerShard.ROW else global_rows
        var local_cols = global_cols // self.tp if shard == LayerShard.COL else global_cols
        var target_rank = HOST_RANK if shard == LayerShard.HOST else DISTRIBUTED
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
            target_rank=target_rank,
        ))
        return off

    @always_inline
    def emit_shape[S: ShapeLike, element_bytes: Int](mut self,
            mut entries: List[WeightDesc],
            suffix: String,
            dtype: DType,
            quantizable: Bool = False) -> Int:
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
            target_rank=DISTRIBUTED,
        ))
        return off

    @always_inline
    def qs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String) -> Int:
        return self.emit_shape[S, 1](entries, suffix, DType.int8, True)

    @always_inline
    def fs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String) -> Int:
        return self.emit_shape[S, 4](entries, suffix, DType.float32, False)

    @always_inline
    def bfs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String) -> Int:
        return self.emit_shape[S, 2](entries, suffix, DType.bfloat16, False)

    @always_inline
    def colsum(mut self, nbytes: Int) -> Int:
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def q(mut self, mut entries: List[WeightDesc], suffix: String,
          rows: Int, cols: Int, shard: Int) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.int8, 1, shard, True)

    @always_inline
    def f(mut self, mut entries: List[WeightDesc], suffix: String,
          rows: Int, cols: Int, shard: Int) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.float32, 4, shard, False)

    @always_inline
    def bf(mut self, mut entries: List[WeightDesc], suffix: String,
           rows: Int, cols: Int, shard: Int) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.bfloat16, 2, shard, False)


# =============================================================================
# Typed view helpers — eliminate Mat alias boilerplate
# =============================================================================


@always_inline
def bound_mat[E: Encoding, S: ShapeLike](addr: Int) -> Bound[Mat[E, S.N, S.M]]:
    return Bound[Mat[E, S.N, S.M]](addr)


@always_inline
def bound_vec[E: Encoding, dim: Int](addr: Int) -> Bound[Mat[E, dim, 1]]:
    return Bound[Mat[E, dim, 1]](addr)


@always_inline
def dyn_mat[E: Encoding, rows: Int, cols: Int](addr: Int, seq_len: Int) -> DynView[Mat[E, rows, cols]]:
    return DynView[Mat[E, rows, cols]](addr, seq_len)


@always_inline
def cache_mat[E: Encoding, rows: Int, cols: Int](addr: Int) -> CacheView[Mat[E, rows, cols]]:
    return CacheView[Mat[E, rows, cols]](addr)
