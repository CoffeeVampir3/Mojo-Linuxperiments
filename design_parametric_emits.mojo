"""Design reproduction: parametric emit.

LayerBuilder today exposes six near-identical methods — bf/f/q and
bfs/fs/qs — that differ only in (dtype, element_bytes, default_quantizable)
and in whether rows/cols come from runtime values or from a compile-time
Shape. The shard dimension is carried as a runtime Int with a switch
inside emit.

Two orthogonal axes hide under those six methods:

  Encoding  : BF16 / F32 / I8        -> (dtype, element_bytes, default
                                         "this weight flows through the
                                         quantizer pipeline" flag)
  Sharding  : RowShard / ColShard /  -> (rows, cols, tp) -> (local_rows,
              Replicated / HostOnly     local_cols) + default target rank

The bf/f/q surface is the Cartesian product of those axes, spelled as a
hand-rolled method per (Encoding, Sharding) combination. That's 3 * 4 = 12
methods if we enumerate everything — we only have six because the current
API carries shard as a runtime Int and elides the HOST/REPL distinction
into it. The runtime branch is the "missing structure": it's a switch we
could push into the type system.

Collapsing the product:

  def emit[E: Encoding, S: Sharding](self, ..., rows, cols) -> Int

Every call is monomorphized per (E, S) pair. No bf/f/q spellings, no
`shard: Int` parameter, no runtime branch, and each weight's dtype /
byte-width / quantizer disposition / per-rank geometry is derived by the
compiler from the parameter types.

This file is a standalone reproduction. It redefines minimal local
versions of Encoding / WeightDesc / DEFAULT_ALIGNMENT so it compiles and
runs on its own with `pixi run mojo design_parametric_emits.mojo`; the
intent is to shape the abstraction before pulling it into
modeling_common.
"""


comptime DEFAULT_ALIGNMENT = 64
comptime DISTRIBUTED = -1
comptime HOST_RANK   = 0


# =============================================================================
# Local WeightDesc mirror — just enough fields to see the abstraction shape.
# =============================================================================


@fieldwise_init
struct WeightDesc(Copyable, Movable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_rows: Int
    var local_cols: Int
    var quantizable: Bool
    var target_rank: Int


# =============================================================================
# Encoding — add DEFAULT_QUANTIZABLE to the existing contract. The flag is
# an intrinsic property of the storage, not of the pipeline, and every call
# site currently re-asserts the same mapping (I8 -> quantizable, others ->
# not). Hoisting it to the type removes that repetition.
# =============================================================================


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int
    comptime DEFAULT_QUANTIZABLE: Bool


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2
    comptime DEFAULT_QUANTIZABLE = False


struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4
    comptime DEFAULT_QUANTIZABLE = False


struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1
    comptime DEFAULT_QUANTIZABLE = True


# =============================================================================
# Sharding — (rows, cols, tp) -> (local_rows, local_cols) + default target.
#
# Each variant is a tiny struct with comptime members and two static funcs.
# That's all the per-rank geometry the emitter needs. No runtime switch.
# =============================================================================


trait Sharding:
    comptime DEFAULT_TARGET: Int

    @staticmethod
    def local_rows(rows: Int, tp: Int) -> Int: ...
    @staticmethod
    def local_cols(cols: Int, tp: Int) -> Int: ...


struct RowShard(Sharding):
    """Output-rows partitioned across ranks. Matches ROW in the old API."""
    comptime DEFAULT_TARGET = DISTRIBUTED

    @staticmethod
    def local_rows(rows: Int, tp: Int) -> Int:
        return rows // tp

    @staticmethod
    def local_cols(cols: Int, tp: Int) -> Int:
        return cols


struct ColShard(Sharding):
    """Contraction-cols partitioned across ranks. Matches COL."""
    comptime DEFAULT_TARGET = DISTRIBUTED

    @staticmethod
    def local_rows(rows: Int, tp: Int) -> Int:
        return rows

    @staticmethod
    def local_cols(cols: Int, tp: Int) -> Int:
        return cols // tp


struct Replicated(Sharding):
    """Every rank holds the full tensor. Matches REPL."""
    comptime DEFAULT_TARGET = DISTRIBUTED

    @staticmethod
    def local_rows(rows: Int, tp: Int) -> Int:
        return rows

    @staticmethod
    def local_cols(cols: Int, tp: Int) -> Int:
        return cols


struct HostOnly(Sharding):
    """Only the host rank materializes this weight. Matches HOST.
    Local shape equals global shape on that rank; other ranks skip.
    """
    comptime DEFAULT_TARGET = HOST_RANK

    @staticmethod
    def local_rows(rows: Int, tp: Int) -> Int:
        return rows

    @staticmethod
    def local_cols(cols: Int, tp: Int) -> Int:
        return cols


# =============================================================================
# Shape — minimal ShapeLike for the compile-time emit_shape variant.
# =============================================================================


trait ShapeLike:
    comptime GLOBAL_N: Int
    comptime GLOBAL_M: Int


struct Shape[global_n: Int, global_m: Int](ShapeLike):
    comptime GLOBAL_N = Self.global_n
    comptime GLOBAL_M = Self.global_m


# =============================================================================
# Parametric LayerBuilder
# =============================================================================


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


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

    # -------------------------------------------------------------------------
    # Runtime rows/cols
    # -------------------------------------------------------------------------

    @always_inline
    def emit[E: Encoding, S: Sharding](mut self,
        mut entries: List[WeightDesc],
        suffix: String,
        rows: Int, cols: Int,
        quantizable: Bool = E.DEFAULT_QUANTIZABLE,
        target_rank: Int = S.DEFAULT_TARGET,
    ) -> Int:
        var local_rows = S.local_rows(rows, self.tp)
        var local_cols = S.local_cols(cols, self.tp)
        var alloc = local_rows * local_cols * E.ELEMENT_BYTES
        var off = align_up(self.cursor)
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=E.DTYPE, element_bytes=E.ELEMENT_BYTES,
            global_rows=rows, global_cols=cols,
            local_rows=local_rows, local_cols=local_cols,
            quantizable=quantizable,
            target_rank=target_rank,
        ))
        return off

    # -------------------------------------------------------------------------
    # Compile-time rows/cols via Shape[...] — same body, Shape supplies the
    # dims. Replaces bfs/fs/qs.
    # -------------------------------------------------------------------------

    @always_inline
    def emit_shape[E: Encoding, S: Sharding, Sh: ShapeLike](mut self,
        mut entries: List[WeightDesc],
        suffix: String,
        quantizable: Bool = E.DEFAULT_QUANTIZABLE,
        target_rank: Int = S.DEFAULT_TARGET,
    ) -> Int:
        return self.emit[E, S](
            entries, suffix, Sh.GLOBAL_N, Sh.GLOBAL_M,
            quantizable=quantizable, target_rank=target_rank,
        )

    # -------------------------------------------------------------------------
    # Side-car allocations — orthogonal to the (E, S) family.
    # -------------------------------------------------------------------------

    @always_inline
    def colsum(mut self, nbytes: Int) -> Int:
        var off = self.cursor
        self.cursor += nbytes
        return off


# =============================================================================
# Demo — one MiniMax-M2.7 attention + MoE body emitted through the
# parametric surface. Compare against the `bf / f / q + shard: Int`
# spellings in the current emit_attn / emit_body.
# =============================================================================


def demo_emit():
    comptime HIDDEN = 3072
    comptime Q_DIM  = 6144
    comptime KV_DIM = 1024
    comptime NE     = 256
    comptime MI     = 1536

    var descs = List[WeightDesc]()
    var b = LayerBuilder(tp=2, prefix="model.layers.0.", layer_base=0)

    # Butterquant int8 projection weights, row-sharded on output.
    _ = b.emit[I8, RowShard](descs, "self_attn.q_proj.weight", Q_DIM,  HIDDEN)
    _ = b.emit[I8, RowShard](descs, "self_attn.k_proj.weight", KV_DIM, HIDDEN)
    _ = b.emit[I8, RowShard](descs, "self_attn.v_proj.weight", KV_DIM, HIDDEN)

    # Per-row F32 scales for each butterquantized weight (same shard axis).
    _ = b.emit[F32, RowShard](descs, "self_attn.q_proj.weight_scale", Q_DIM,  1)
    _ = b.emit[F32, RowShard](descs, "self_attn.k_proj.weight_scale", KV_DIM, 1)
    _ = b.emit[F32, RowShard](descs, "self_attn.v_proj.weight_scale", KV_DIM, 1)

    # O projection — contraction-dim sharded; its scale is replicated.
    _ = b.emit[I8,  ColShard](descs,  "self_attn.o_proj.weight",       HIDDEN, Q_DIM)
    _ = b.emit[F32, Replicated](descs, "self_attn.o_proj.weight_scale", HIDDEN, 1)

    # Full-vector Q/K RMSNorm weights — replicated BF16.
    _ = b.emit[BF16, Replicated](descs, "self_attn.q_norm.weight", Q_DIM,  1)
    _ = b.emit[BF16, Replicated](descs, "self_attn.k_norm.weight", KV_DIM, 1)

    # Layer norms — replicated BF16.
    _ = b.emit[BF16, Replicated](descs, "input_layernorm.weight",          HIDDEN, 1)
    _ = b.emit[BF16, Replicated](descs, "post_attention_layernorm.weight", HIDDEN, 1)

    # Router — F32 passthrough, replicated.
    _ = b.emit[F32, Replicated](descs, "block_sparse_moe.gate.weight",               NE, HIDDEN)
    _ = b.emit[F32, Replicated](descs, "block_sparse_moe.e_score_correction_bias",    NE, 1)

    # Experts — one desc per expert would go here with target_rank overrides;
    # this demo uses the stacked-virtual-tensor form for brevity.
    _ = b.emit[I8,  RowShard](descs, "block_sparse_moe.experts.w1_stacked", NE * MI, HIDDEN)
    _ = b.emit[F32, RowShard](descs, "block_sparse_moe.experts.w1_scale",   NE * MI, 1)

    _ = b.emit[I8,  RowShard](descs, "block_sparse_moe.experts.w3_stacked", NE * MI, HIDDEN)
    _ = b.emit[F32, RowShard](descs, "block_sparse_moe.experts.w3_scale",   NE * MI, 1)

    _ = b.emit[I8,  RowShard](descs, "block_sparse_moe.experts.w2_stacked", NE * HIDDEN, MI)
    _ = b.emit[F32, RowShard](descs, "block_sparse_moe.experts.w2_scale",   NE * HIDDEN, 1)

    print("emitted", len(descs), "descs, cursor =", b.cursor, "bytes")

    # A couple of invariants worth spot-checking at runtime. These would be
    # compile-time asserts once the reproduction moves into modeling_common.
    var q = descs[0].copy()
    if q.local_rows != Q_DIM // 2:
        print("FAIL: row-shard local_rows =", q.local_rows, "expected", Q_DIM // 2)
    if q.dtype != DType.int8 or q.element_bytes != 1:
        print("FAIL: I8 encoding metadata wrong")
    if not q.quantizable:
        print("FAIL: I8 default_quantizable should be True")

    var qnorm = descs[8].copy()
    if qnorm.local_rows != Q_DIM or qnorm.dtype != DType.bfloat16:
        print("FAIL: BF16/Replicated Q_NORM geometry wrong")
    if qnorm.target_rank != DISTRIBUTED:
        print("FAIL: Replicated default target should be DISTRIBUTED")


def main():
    demo_emit()
