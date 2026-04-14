"""Final model design proposal — comprehensive, no hand-waving.

Covers: typed refs with static and dynamic binding, quantized multi-atom
families, opaque non-matrix refs, arena section addressing, alignment,
TP assertions, both flavors with full topology, quant host aux.
"""

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, Mat, Bound, DynView,
    WeightDesc, DEFAULT_ALIGNMENT,
)
from modeling.gemma4_common import (
    LayerShard, LayerBuilder, is_full_layer,
)


# =============================================================================
# Architecture — TP divisibility requirements documented
# =============================================================================


struct Arch:
    comptime HIDDEN = 256
    comptime NUM_HEADS = 4
    comptime HEAD_DIM_SLIDING = 64
    comptime NUM_KV_HEADS_SLIDING = 2
    comptime Q_DIM_SLIDING = 256
    comptime KV_DIM_SLIDING = 128
    comptime HEAD_DIM_FULL = 128
    comptime NUM_KV_HEADS_FULL = 1
    comptime Q_DIM_FULL = 512
    comptime KV_DIM_FULL = 128
    comptime INTERMEDIATE = 512
    comptime NUM_EXPERTS = 8
    comptime TOP_K = 2
    comptime MOE_GATE_UP = 256
    comptime MOE_INTERMEDIATE = 128
    comptime VOCAB_SIZE = 1024
    comptime NUM_LAYERS = 6
    comptime NUM_SLIDING = 5
    comptime NUM_FULL = 1
    comptime MAX_SEQ_LEN = 128
    comptime SLIDING_WINDOW = 64
    comptime SLIDING_ROPE_HALF = Self.HEAD_DIM_SLIDING // 2
    comptime FULL_ROPE_HALF = 64
    comptime FWHT_BLK = 64
    comptime LM_HEAD_FWHT_BLK = 64

comptime A = Arch


def assert_tp[tp: Int]():
    debug_assert(A.NUM_HEADS % tp == 0, "NUM_HEADS must divide by tp")
    debug_assert(A.NUM_KV_HEADS_SLIDING % tp == 0, "NUM_KV_HEADS_SLIDING must divide by tp")
    debug_assert(A.Q_DIM_SLIDING % tp == 0, "Q_DIM_SLIDING must divide by tp")
    debug_assert(A.KV_DIM_SLIDING % tp == 0, "KV_DIM_SLIDING must divide by tp")
    debug_assert(A.INTERMEDIATE % tp == 0, "INTERMEDIATE must divide by tp")
    debug_assert(A.NUM_EXPERTS % tp == 0, "NUM_EXPERTS must divide by tp")


# =============================================================================
# Core Algebra
# =============================================================================


@fieldwise_init
struct SlotView[E: Encoding, S: ShapeLike](Encoding, Shaped, Copyable, Movable):
    """Terminal: typed pointer. Static binding for weights, norms, tables."""
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.N
    comptime COLS = Self.S.M
    var ptr: Int


@fieldwise_init
struct SlotOffset[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Ref: typed offset. Two bind modes — static and dynamic."""
    var offset: Int

    def bind(self, base: Int) -> SlotView[Self.E, Self.S]:
        """Static bind: weights, norms, tables. Returns SlotView."""
        return SlotView[Self.E, Self.S](base + self.offset)

    def dyn(self, base: Int, seq_len: Int) -> DynView[Mat[Self.E, Self.S.N, Self.S.M]]:
        """Dynamic bind: activations, scratch. Returns DynView with runtime seq_len."""
        return DynView[Mat[Self.E, Self.S.N, Self.S.M]](base + self.offset, seq_len)

    def addr(self, base: Int) -> Int:
        """Raw address: for init-time writes, collectives, raw pointer access."""
        return base + self.offset


@fieldwise_init
struct QOffset[DS: ShapeLike, SS: ShapeLike](Copyable, ImplicitlyCopyable):
    """Quantized weight: i8 data + f32 per-row scale."""
    var data: SlotOffset[I8, Self.DS]
    var scale: SlotOffset[F32, Self.SS]


@fieldwise_init
struct QOffset3[DS: ShapeLike, SS: ShapeLike, CS: ShapeLike](Copyable, ImplicitlyCopyable):
    """Quantized weight with colsum: data + scale + colsum."""
    var data: SlotOffset[I8, Self.DS]
    var scale: SlotOffset[F32, Self.SS]
    var colsum: SlotOffset[F32, Self.CS]


@fieldwise_init
struct OpaqueSlot[bytes: Int](Copyable, ImplicitlyCopyable):
    """Non-matrix ref: opaque byte region (e.g., packed KV cache).
    The slot knows its size but not its internal layout."""
    var offset: Int

    def addr(self, base: Int) -> Int:
        return base + self.offset


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable](Copyable, ImplicitlyCopyable):
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    def base(self, arena: Int, idx: Int) -> Int:
        return arena + self.off + idx * self.stride


struct SectionBuilder:
    """Cursor allocator producing typed refs. Bytes fall out of reserves."""
    var cursor: Int

    def __init__(out self):
        self.cursor = 0

    def align(mut self):
        self.cursor = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT

    def reserve[E: Encoding, S: ShapeLike](mut self) -> SlotOffset[E, S]:
        self.align()
        comptime size = S.bytes_for[E.ELEMENT_BYTES]()
        var off = self.cursor
        self.cursor += size
        return SlotOffset[E, S](off)

    def reserve_opaque[bytes: Int](mut self) -> OpaqueSlot[bytes]:
        self.align()
        var off = self.cursor
        self.cursor += bytes
        return OpaqueSlot[bytes](off)

    def reserve_bytes(mut self, nbytes: Int) -> Int:
        var off = self.cursor
        self.cursor += nbytes
        return off

    def bytes(self) -> Int:
        return self.cursor


# =============================================================================
# BF16 Flavor
# =============================================================================


struct BF16Shapes[tp: Int]:
    comptime SlidingQ  = Shape[A.Q_DIM_SLIDING, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingKV = Shape[A.KV_DIM_SLIDING, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingO  = Shape[A.HIDDEN, A.Q_DIM_SLIDING, shard_m=True, tp=Self.tp]
    comptime FullQ     = Shape[A.Q_DIM_FULL, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullK     = Shape[A.KV_DIM_FULL, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullO     = Shape[A.HIDDEN, A.Q_DIM_FULL, shard_m=True, tp=Self.tp]
    comptime GateUp    = Shape[A.INTERMEDIATE, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime Down      = Shape[A.HIDDEN, A.INTERMEDIATE, shard_m=True, tp=Self.tp]
    comptime ExpertsGU = Shape[A.NUM_EXPERTS * A.MOE_GATE_UP, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime ExpertsDown = Shape[A.NUM_EXPERTS * A.HIDDEN, A.MOE_INTERMEDIATE, shard_n=True, tp=Self.tp]


@fieldwise_init
struct BF16SlidingAttn[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = BF16Shapes[Self.tp]
    var q_proj: SlotOffset[BF16, Self.S.SlidingQ]
    var k_proj: SlotOffset[BF16, Self.S.SlidingKV]
    var v_proj: SlotOffset[BF16, Self.S.SlidingKV]
    var o_proj: SlotOffset[BF16, Self.S.SlidingO]
    var q_norm: SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]]
    var k_norm: SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]]


@fieldwise_init
struct BF16FullAttn[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = BF16Shapes[Self.tp]
    var q_proj: SlotOffset[BF16, Self.S.FullQ]
    var k_proj: SlotOffset[BF16, Self.S.FullK]
    var o_proj: SlotOffset[BF16, Self.S.FullO]
    var q_norm: SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]]
    var k_norm: SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]]


@fieldwise_init
struct BF16Body[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = BF16Shapes[Self.tp]
    var input_norm:      SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_attn_norm:  SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var pre_ffn_norm:    SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var pre_ffn_norm_2:  SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm_1: SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm_2: SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm:   SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var gate_proj:       SlotOffset[BF16, Self.S.GateUp]
    var up_proj:         SlotOffset[BF16, Self.S.GateUp]
    var down_proj:       SlotOffset[BF16, Self.S.Down]
    var router_proj:     SlotOffset[BF16, Shape[A.NUM_EXPERTS, A.HIDDEN]]
    var router_scale:    SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var router_pes:      SlotOffset[BF16, Shape[A.NUM_EXPERTS, 1]]
    var experts_gate_up: SlotOffset[BF16, Self.S.ExpertsGU]
    var experts_down:    SlotOffset[BF16, Self.S.ExpertsDown]
    var layer_scalar:    SlotOffset[BF16, Shape[1, 1]]


@fieldwise_init
struct BF16SlidingLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: BF16SlidingAttn[Self.tp]
    var body: BF16Body[Self.tp]

@fieldwise_init
struct BF16FullLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: BF16FullAttn[Self.tp]
    var body: BF16Body[Self.tp]


# =============================================================================
# Quant Flavor — VNNI alignment, multi-atom weights, colsums, host aux
# =============================================================================


comptime VNNI_ALIGN = 64

struct QuantShapes[tp: Int]:
    comptime SlidingQ    = Shape[A.Q_DIM_SLIDING, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingQSc  = Shape[A.Q_DIM_SLIDING, 1, shard_n=True, tp=Self.tp]
    comptime SlidingKV   = Shape[A.KV_DIM_SLIDING, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingKVSc = Shape[A.KV_DIM_SLIDING, 1, shard_n=True, tp=Self.tp]
    comptime SlidingO    = Shape[A.HIDDEN, A.Q_DIM_SLIDING, shard_m=True, tp=Self.tp]
    comptime FullQ       = Shape[A.Q_DIM_FULL, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullQSc     = Shape[A.Q_DIM_FULL, 1, shard_n=True, tp=Self.tp]
    comptime FullK       = Shape[A.KV_DIM_FULL, A.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullKSc     = Shape[A.KV_DIM_FULL, 1, shard_n=True, tp=Self.tp]
    comptime FullO       = Shape[A.HIDDEN, A.Q_DIM_FULL, shard_m=True, tp=Self.tp]
    comptime GateUp      = Shape[A.INTERMEDIATE, A.HIDDEN, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    comptime GateUpSc    = Shape[A.INTERMEDIATE, 1, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    comptime Down        = Shape[A.HIDDEN, A.INTERMEDIATE, shard_m=True, tp=Self.tp, align_m=VNNI_ALIGN]
    comptime DownSc      = Shape[A.HIDDEN, 1]
    # Colsum shapes: derived from weight shapes + block size
    comptime GUColsum    = Shape[Self.GateUp.N * 2, 1]
    comptime DownColsum  = Shape[A.HIDDEN, Self.Down.M // A.FWHT_BLK]


# Quant attention: loaded projections (QOffset) + init-time colsums (SlotOffset)
@fieldwise_init
struct QuantSlidingAttn[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = QuantShapes[Self.tp]
    var q_proj:  QOffset[Self.S.SlidingQ, Self.S.SlidingQSc]
    var k_proj:  QOffset[Self.S.SlidingKV, Self.S.SlidingKVSc]
    var v_proj:  QOffset[Self.S.SlidingKV, Self.S.SlidingKVSc]
    var o_proj:  QOffset[Self.S.SlidingO, Shape[A.HIDDEN, 1]]
    var q_norm:  SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]]
    var k_norm:  SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]]
    var qkv_colsum: SlotOffset[F32, Shape[(A.Q_DIM_SLIDING + 2 * A.KV_DIM_SLIDING) // Self.tp, 1]]
    var o_colsum:   SlotOffset[F32, Shape[A.HIDDEN, A.NUM_HEADS // Self.tp]]


@fieldwise_init
struct QuantFullAttn[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = QuantShapes[Self.tp]
    var q_proj:  QOffset[Self.S.FullQ, Self.S.FullQSc]
    var k_proj:  QOffset[Self.S.FullK, Self.S.FullKSc]
    var o_proj:  QOffset[Self.S.FullO, Shape[A.HIDDEN, 1]]
    var q_norm:  SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]]
    var k_norm:  SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]]
    var qk_colsum: SlotOffset[F32, Shape[(A.Q_DIM_FULL + A.KV_DIM_FULL) // Self.tp, 1]]
    var o_colsum:  SlotOffset[F32, Shape[A.HIDDEN, A.NUM_HEADS // Self.tp]]


# Quant body: data+scale for projections, separate colsum slots, norms bf16
@fieldwise_init
struct QuantBody[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = QuantShapes[Self.tp]
    var input_norm:      SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_attn_norm:  SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var pre_ffn_norm:    SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var pre_ffn_norm_2:  SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm_1: SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm_2: SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var post_ffn_norm:   SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var gate_proj:       QOffset[Self.S.GateUp, Self.S.GateUpSc]
    var up_proj:         QOffset[Self.S.GateUp, Self.S.GateUpSc]
    var down_proj:       QOffset[Self.S.Down, Self.S.DownSc]
    var router_proj:     QOffset[Shape[A.NUM_EXPERTS, A.HIDDEN], Shape[A.NUM_EXPERTS, 1]]
    var router_scale:    SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var router_pes:      SlotOffset[BF16, Shape[A.NUM_EXPERTS, 1]]
    var experts_gate_up: QOffset[Shape[A.NUM_EXPERTS * A.MOE_GATE_UP, A.HIDDEN], Shape[A.NUM_EXPERTS * A.MOE_GATE_UP, 1]]
    var experts_down:    QOffset[Shape[A.NUM_EXPERTS * A.HIDDEN, A.MOE_INTERMEDIATE], Shape[A.NUM_EXPERTS * A.HIDDEN, 1]]
    var layer_scalar:    SlotOffset[BF16, Shape[1, 1]]
    var gu_colsum:       SlotOffset[F32, Self.S.GUColsum]
    var down_colsum:     SlotOffset[F32, Self.S.DownColsum]
    var router_colsum:   SlotOffset[F32, Shape[A.NUM_EXPERTS, 1]]


@fieldwise_init
struct QuantSlidingLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: QuantSlidingAttn[Self.tp]
    var body: QuantBody[Self.tp]

@fieldwise_init
struct QuantFullLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: QuantFullAttn[Self.tp]
    var body: QuantBody[Self.tp]


# Quant host: embed is quantized + gamma-derived aux tensors
comptime VOCAB_NUM_BLOCKS = A.HIDDEN // A.LM_HEAD_FWHT_BLK

@fieldwise_init
struct QuantHostSlots(Copyable, ImplicitlyCopyable):
    var final_norm:     SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var embed_data:     SlotOffset[I8, Shape[A.VOCAB_SIZE, A.HIDDEN]]
    var embed_scale:    SlotOffset[F32, Shape[A.VOCAB_SIZE, VOCAB_NUM_BLOCKS]]
    var embed_colsum:   SlotOffset[F32, Shape[A.VOCAB_SIZE, VOCAB_NUM_BLOCKS]]
    var sqrt_gamma:     SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var inv_sqrt_gamma: SlotOffset[F32, Shape[A.HIDDEN, 1]]


# =============================================================================
# State Families
# =============================================================================


@fieldwise_init
struct BF16KVSlots(Copyable, ImplicitlyCopyable):
    var k: SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_SLIDING]]
    var v: SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_SLIDING]]

@fieldwise_init
struct BF16FullKVSlots(Copyable, ImplicitlyCopyable):
    var k: SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_FULL]]
    var v: SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_FULL]]

# Packed KV cache for quant: opaque internal layout
comptime PACKED_SLIDING_KV_BYTES = 2 * A.SLIDING_WINDOW * A.KV_DIM_SLIDING * 2
comptime PACKED_FULL_KV_BYTES = 2 * A.MAX_SEQ_LEN * A.KV_DIM_FULL * 2

@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: SlotOffset[F32, Shape[A.MAX_SEQ_LEN, Self.half]]
    var sin: SlotOffset[F32, Shape[A.MAX_SEQ_LEN, Self.half]]

@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main:     SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]]
    var x_residual: SlotOffset[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]]

@fieldwise_init
struct BF16HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm: SlotOffset[BF16, Shape[A.HIDDEN, 1]]
    var embed:      SlotOffset[BF16, Shape[A.VOCAB_SIZE, A.HIDDEN]]


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct BF16Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var sliding: Repeated[BF16SlidingLayer[Self.tp]]
    var full: Repeated[BF16FullLayer[Self.tp]]
    var distributed_bytes: Int
    var sliding_kv: Repeated[BF16KVSlots]
    var full_kv: Repeated[BF16FullKVSlots]
    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_rope: RopeSlots[A.SLIDING_ROPE_HALF]
    var full_rope: RopeSlots[A.FULL_ROPE_HALF]
    var state_bytes: Int
    var host: BF16HostSlots

    def state_base(self, arena: Int) -> Int:
        return arena + self.distributed_bytes


@fieldwise_init
struct QuantTopology[tp: Int](Copyable, ImplicitlyCopyable):
    var sliding: Repeated[QuantSlidingLayer[Self.tp]]
    var full: Repeated[QuantSlidingLayer[Self.tp]]
    var distributed_bytes: Int
    var sliding_kv: Repeated[OpaqueSlot[PACKED_SLIDING_KV_BYTES]]
    var full_kv: Repeated[OpaqueSlot[PACKED_FULL_KV_BYTES]]
    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_rope: RopeSlots[A.SLIDING_ROPE_HALF]
    var full_rope: RopeSlots[A.FULL_ROPE_HALF]
    var state_bytes: Int
    var host: QuantHostSlots

    def state_base(self, arena: Int) -> Int:
        return arena + self.distributed_bytes


# =============================================================================
# Emit helpers
# =============================================================================


def qoff[DS: ShapeLike, SS: ShapeLike](
    mut b: LayerBuilder, mut e: List[WeightDesc],
    data_sfx: String, scale_sfx: String,
) -> QOffset[DS, SS]:
    return QOffset[DS, SS](
        data=SlotOffset[I8, DS](b.qs[DS](e, data_sfx)),
        scale=SlotOffset[F32, SS](b.fs[SS](e, scale_sfx)))


def bf16_emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BF16Body[tp]:
    comptime S = BF16Shapes[tp]
    comptime R = LayerShard.REPL
    comptime H = A.HIDDEN
    comptime NE = A.NUM_EXPERTS
    return BF16Body[tp](
        input_norm      = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "input_layernorm.weight", H, 1, R)),
        post_attn_norm  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_attention_layernorm.weight", H, 1, R)),
        pre_ffn_norm    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm.weight", H, 1, R)),
        pre_ffn_norm_2  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm_2.weight", H, 1, R)),
        post_ffn_norm_1 = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_1.weight", H, 1, R)),
        post_ffn_norm_2 = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_2.weight", H, 1, R)),
        post_ffn_norm   = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm.weight", H, 1, R)),
        gate_proj       = SlotOffset[BF16, S.GateUp](b.bfs[S.GateUp](e, "mlp.gate_proj.weight")),
        up_proj         = SlotOffset[BF16, S.GateUp](b.bfs[S.GateUp](e, "mlp.up_proj.weight")),
        down_proj       = SlotOffset[BF16, S.Down](b.bfs[S.Down](e, "mlp.down_proj.weight")),
        router_proj     = SlotOffset[BF16, Shape[NE, H]](b.bf(e, "router.proj.weight", NE, H, R)),
        router_scale    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "router.scale", H, 1, R)),
        router_pes      = SlotOffset[BF16, Shape[NE, 1]](b.bf(e, "router.per_expert_scale", NE, 1, R)),
        experts_gate_up = SlotOffset[BF16, S.ExpertsGU](b.bfs[S.ExpertsGU](e, "experts.gate_up_proj")),
        experts_down    = SlotOffset[BF16, S.ExpertsDown](b.bfs[S.ExpertsDown](e, "experts.down_proj")),
        layer_scalar    = SlotOffset[BF16, Shape[1, 1]](b.bf(e, "layer_scalar", 1, 1, R)))


def bf16_emit_sliding[tp: Int](pfx: String, base: Int, mut e: List[WeightDesc]) -> Tuple[BF16SlidingLayer[tp], Int]:
    var b = LayerBuilder(tp, pfx, base)
    comptime S = BF16Shapes[tp]
    comptime R = LayerShard.REPL
    var attn = BF16SlidingAttn[tp](
        q_proj=SlotOffset[BF16, S.SlidingQ](b.bfs[S.SlidingQ](e, "self_attn.q_proj.weight")),
        k_proj=SlotOffset[BF16, S.SlidingKV](b.bfs[S.SlidingKV](e, "self_attn.k_proj.weight")),
        v_proj=SlotOffset[BF16, S.SlidingKV](b.bfs[S.SlidingKV](e, "self_attn.v_proj.weight")),
        o_proj=SlotOffset[BF16, S.SlidingO](b.bfs[S.SlidingO](e, "self_attn.o_proj.weight")),
        q_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.q_norm.weight", A.HEAD_DIM_SLIDING, 1, R)),
        k_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.k_norm.weight", A.HEAD_DIM_SLIDING, 1, R)))
    var body = bf16_emit_body[tp](b, e)
    return (BF16SlidingLayer[tp](attn=attn, body=body), b.cursor)


def bf16_emit_full[tp: Int](pfx: String, base: Int, mut e: List[WeightDesc]) -> Tuple[BF16FullLayer[tp], Int]:
    var b = LayerBuilder(tp, pfx, base)
    comptime S = BF16Shapes[tp]
    comptime R = LayerShard.REPL
    var attn = BF16FullAttn[tp](
        q_proj=SlotOffset[BF16, S.FullQ](b.bfs[S.FullQ](e, "self_attn.q_proj.weight")),
        k_proj=SlotOffset[BF16, S.FullK](b.bfs[S.FullK](e, "self_attn.k_proj.weight")),
        o_proj=SlotOffset[BF16, S.FullO](b.bfs[S.FullO](e, "self_attn.o_proj.weight")),
        q_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.q_norm.weight", A.HEAD_DIM_FULL, 1, R)),
        k_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.k_norm.weight", A.HEAD_DIM_FULL, 1, R)))
    var body = bf16_emit_body[tp](b, e)
    return (BF16FullLayer[tp](attn=attn, body=body), b.cursor)


def quant_emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> QuantBody[tp]:
    comptime S = QuantShapes[tp]
    comptime R = LayerShard.REPL
    comptime ROW = LayerShard.ROW
    comptime H = A.HIDDEN
    comptime NE = A.NUM_EXPERTS
    comptime GU = A.MOE_GATE_UP
    comptime MI = A.MOE_INTERMEDIATE
    return QuantBody[tp](
        input_norm      = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "input_layernorm.weight", H, 1, R)),
        post_attn_norm  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_attention_layernorm.weight", H, 1, R)),
        pre_ffn_norm    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm.weight", H, 1, R)),
        pre_ffn_norm_2  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm_2.weight", H, 1, R)),
        post_ffn_norm_1 = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_1.weight", H, 1, R)),
        post_ffn_norm_2 = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_2.weight", H, 1, R)),
        post_ffn_norm   = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm.weight", H, 1, R)),
        gate_proj       = qoff[S.GateUp, S.GateUpSc](b, e, "mlp.gate_proj.weight", "mlp.gate_proj.weight_scale"),
        up_proj         = qoff[S.GateUp, S.GateUpSc](b, e, "mlp.up_proj.weight", "mlp.up_proj.weight_scale"),
        down_proj       = qoff[S.Down, S.DownSc](b, e, "mlp.down_proj.weight", "mlp.down_proj.weight_scale"),
        router_proj     = qoff[Shape[NE, H], Shape[NE, 1]](b, e, "router.proj.weight", "router.proj.weight_scale"),
        router_scale    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "router.scale", H, 1, R)),
        router_pes      = SlotOffset[BF16, Shape[NE, 1]](b.bf(e, "router.per_expert_scale", NE, 1, R)),
        experts_gate_up = qoff[Shape[NE * GU, H], Shape[NE * GU, 1]](b, e, "experts.gate_up_proj", "experts.gate_up_proj_scale"),
        experts_down    = qoff[Shape[NE * H, MI], Shape[NE * H, 1]](b, e, "experts.down_proj", "experts.down_proj_scale"),
        layer_scalar    = SlotOffset[BF16, Shape[1, 1]](b.bf(e, "layer_scalar", 1, 1, R)),
        gu_colsum       = SlotOffset[F32, S.GUColsum](b.colsum(S.GUColsum.N * S.GUColsum.M * 4)),
        down_colsum     = SlotOffset[F32, S.DownColsum](b.colsum(S.DownColsum.N * S.DownColsum.M * 4)),
        router_colsum   = SlotOffset[F32, Shape[NE, 1]](b.colsum(NE * 4)))


def quant_emit_sliding[tp: Int](pfx: String, base: Int, mut e: List[WeightDesc]) -> Tuple[QuantSlidingLayer[tp], Int]:
    var b = LayerBuilder(tp, pfx, base)
    comptime S = QuantShapes[tp]
    comptime R = LayerShard.REPL
    comptime H = A.HIDDEN
    var attn = QuantSlidingAttn[tp](
        q_proj=qoff[S.SlidingQ, S.SlidingQSc](b, e, "self_attn.q_proj.weight", "self_attn.q_proj.weight_scale"),
        k_proj=qoff[S.SlidingKV, S.SlidingKVSc](b, e, "self_attn.k_proj.weight", "self_attn.k_proj.weight_scale"),
        v_proj=qoff[S.SlidingKV, S.SlidingKVSc](b, e, "self_attn.v_proj.weight", "self_attn.v_proj.weight_scale"),
        o_proj=qoff[S.SlidingO, Shape[H, 1]](b, e, "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale"),
        q_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.q_norm.weight", A.HEAD_DIM_SLIDING, 1, R)),
        k_norm=SlotOffset[BF16, Shape[A.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.k_norm.weight", A.HEAD_DIM_SLIDING, 1, R)),
        qkv_colsum=SlotOffset[F32, Shape[(A.Q_DIM_SLIDING + 2 * A.KV_DIM_SLIDING) // tp, 1]](
            b.colsum((A.Q_DIM_SLIDING + 2 * A.KV_DIM_SLIDING) // tp * 4)),
        o_colsum=SlotOffset[F32, Shape[H, A.NUM_HEADS // tp]](b.colsum(H * (A.NUM_HEADS // tp) * 4)))
    var body = quant_emit_body[tp](b, e)
    return (QuantSlidingLayer[tp](attn=attn, body=body), b.cursor)


# =============================================================================
# Build: SectionBuilder for state, LayerBuilder for weights/host
# =============================================================================


def bf16_build[tp: Int]() -> Tuple[BF16Topology[tp], List[WeightDesc]]:
    assert_tp[tp]()
    var descs = List[WeightDesc]()
    var scratch_descs = List[WeightDesc]()
    var sl_r = bf16_emit_sliding[tp]("", 0, scratch_descs)
    var fl_r = bf16_emit_full[tp]("", 0, scratch_descs)
    var sl_proto = sl_r[0]
    var sl_stride = sl_r[1]
    var fl_proto = fl_r[0]
    var fl_stride = fl_r[1]

    var sl_off = 0
    var fl_off = sl_off + A.NUM_SLIDING * sl_stride
    var dist = fl_off + A.NUM_FULL * fl_stride

    var si = 0
    var fi = 0
    for i in range(A.NUM_LAYERS):
        var pfx = "layers." + String(i) + "."
        if is_full_layer(i):
            _ = bf16_emit_full[tp](pfx, fl_off + fi * fl_stride, descs)
            fi += 1
        else:
            _ = bf16_emit_sliding[tp](pfx, sl_off + si * sl_stride, descs)
            si += 1

    # State via SectionBuilder — bytes fall out of reserves
    var state = SectionBuilder()

    var skv_sb = SectionBuilder()
    var skv_proto = BF16KVSlots(
        k=skv_sb.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_SLIDING]](),
        v=skv_sb.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_SLIDING]]())
    var sliding_kv = Repeated[BF16KVSlots](skv_proto, state.cursor, skv_sb.bytes(), A.NUM_SLIDING)
    _ = state.reserve_bytes(A.NUM_SLIDING * skv_sb.bytes())

    var fkv_sb = SectionBuilder()
    var fkv_proto = BF16FullKVSlots(
        k=fkv_sb.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_FULL]](),
        v=fkv_sb.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.KV_DIM_FULL]]())
    var full_kv = Repeated[BF16FullKVSlots](fkv_proto, state.cursor, fkv_sb.bytes(), A.NUM_FULL)
    _ = state.reserve_bytes(A.NUM_FULL * fkv_sb.bytes())

    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]]())

    comptime Sh = BF16Shapes[tp]
    comptime scratch_cap = A.MAX_SEQ_LEN * Sh.GateUp.N * BF16.ELEMENT_BYTES * 2
    var scratch_off = state.reserve_bytes(scratch_cap)

    var sliding_rope = RopeSlots[A.SLIDING_ROPE_HALF](
        cos=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.SLIDING_ROPE_HALF]](),
        sin=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.SLIDING_ROPE_HALF]]())
    var full_rope = RopeSlots[A.FULL_ROPE_HALF](
        cos=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.FULL_ROPE_HALF]](),
        sin=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.FULL_ROPE_HALF]]())

    # Host via LayerBuilder (needs WeightDescs for loader)
    var host_off = ((dist + state.bytes() + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    comptime HOST = LayerShard.HOST
    var host = BF16HostSlots(
        final_norm=SlotOffset[BF16, Shape[A.HIDDEN, 1]](hb.bf(descs, "norm.weight", A.HIDDEN, 1, HOST)),
        embed=SlotOffset[BF16, Shape[A.VOCAB_SIZE, A.HIDDEN]](hb.bf(descs, "embed_tokens.weight", A.VOCAB_SIZE, A.HIDDEN, HOST)))

    return (BF16Topology[tp](
        sliding=Repeated[BF16SlidingLayer[tp]](sl_proto, sl_off, sl_stride, A.NUM_SLIDING),
        full=Repeated[BF16FullLayer[tp]](fl_proto, fl_off, fl_stride, A.NUM_FULL),
        distributed_bytes=dist,
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        state_bytes=state.bytes(), host=host), descs^)


def quant_build[tp: Int]() -> Tuple[QuantTopology[tp], List[WeightDesc]]:
    assert_tp[tp]()
    var descs = List[WeightDesc]()
    var scratch_descs = List[WeightDesc]()
    var sl_r = quant_emit_sliding[tp]("", 0, scratch_descs)
    var sl_proto = sl_r[0]
    var sl_stride = sl_r[1]

    # Quant full emit would go here; reuse sliding proto for mock
    var fl_proto = sl_proto
    var fl_stride = sl_stride

    var sl_off = 0
    var fl_off = sl_off + A.NUM_SLIDING * sl_stride
    var dist = fl_off + A.NUM_FULL * fl_stride

    var si = 0
    var fi = 0
    for i in range(A.NUM_LAYERS):
        var pfx = "layers." + String(i) + "."
        if is_full_layer(i):
            _ = quant_emit_sliding[tp](pfx, fl_off + fi * fl_stride, descs)
            fi += 1
        else:
            _ = quant_emit_sliding[tp](pfx, sl_off + si * sl_stride, descs)
            si += 1

    var state = SectionBuilder()

    # Packed KV caches: opaque slots
    var sliding_kv = Repeated[OpaqueSlot[PACKED_SLIDING_KV_BYTES]](
        OpaqueSlot[PACKED_SLIDING_KV_BYTES](0), state.cursor, PACKED_SLIDING_KV_BYTES, A.NUM_SLIDING)
    _ = state.reserve_bytes(A.NUM_SLIDING * PACKED_SLIDING_KV_BYTES)

    var full_kv = Repeated[OpaqueSlot[PACKED_FULL_KV_BYTES]](
        OpaqueSlot[PACKED_FULL_KV_BYTES](0), state.cursor, PACKED_FULL_KV_BYTES, A.NUM_FULL)
    _ = state.reserve_bytes(A.NUM_FULL * PACKED_FULL_KV_BYTES)

    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[A.MAX_SEQ_LEN, A.HIDDEN]]())

    comptime scratch_cap = A.MAX_SEQ_LEN * A.HIDDEN * 4
    var scratch_off = state.reserve_bytes(scratch_cap)

    var sliding_rope = RopeSlots[A.SLIDING_ROPE_HALF](
        cos=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.SLIDING_ROPE_HALF]](),
        sin=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.SLIDING_ROPE_HALF]]())
    var full_rope = RopeSlots[A.FULL_ROPE_HALF](
        cos=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.FULL_ROPE_HALF]](),
        sin=state.reserve[F32, Shape[A.MAX_SEQ_LEN, A.FULL_ROPE_HALF]]())

    # Quant host with aux tensors
    var host_off = ((dist + state.bytes() + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    comptime HOST = LayerShard.HOST
    comptime R = LayerShard.REPL
    var host_sb = SectionBuilder()
    host_sb.cursor = host_off
    var host = QuantHostSlots(
        final_norm=SlotOffset[BF16, Shape[A.HIDDEN, 1]](hb.bf(descs, "norm.weight", A.HIDDEN, 1, HOST)),
        embed_data=SlotOffset[I8, Shape[A.VOCAB_SIZE, A.HIDDEN]](hb.q(descs, "embed_tokens.weight", A.VOCAB_SIZE, A.HIDDEN, HOST)),
        embed_scale=SlotOffset[F32, Shape[A.VOCAB_SIZE, VOCAB_NUM_BLOCKS]](
            hb.f(descs, "embed_tokens.weight_scale", A.VOCAB_SIZE, VOCAB_NUM_BLOCKS, HOST)),
        embed_colsum=SlotOffset[F32, Shape[A.VOCAB_SIZE, VOCAB_NUM_BLOCKS]](hb.colsum(A.VOCAB_SIZE * VOCAB_NUM_BLOCKS * 4)),
        sqrt_gamma=SlotOffset[BF16, Shape[A.HIDDEN, 1]](hb.colsum(A.HIDDEN * 2)),
        inv_sqrt_gamma=SlotOffset[F32, Shape[A.HIDDEN, 1]](hb.colsum(A.HIDDEN * 4)))

    return (QuantTopology[tp](
        sliding=Repeated[QuantSlidingLayer[tp]](sl_proto, sl_off, sl_stride, A.NUM_SLIDING),
        full=Repeated[QuantSlidingLayer[tp]](fl_proto, fl_off, fl_stride, A.NUM_FULL),
        distributed_bytes=dist,
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        state_bytes=state.bytes(), host=host), descs^)


# =============================================================================
# Mock kernels + forward
# =============================================================================


def mock_gemm[W: Encoding & Shaped](w: W, seq: Int):
    print("  gemm [" + String(W.ROWS) + "x" + String(W.COLS) + "]")

def mock_norm[N: Encoding & Shaped](n: N):
    print("  norm d=" + String(N.ROWS))

def mock_i8_gemv[D: Encoding & Shaped, S: Encoding & Shaped](d: D, s: S):
    print("  i8_gemv [" + String(D.ROWS) + "x" + String(D.COLS) + "] sc[" + String(S.ROWS) + "]")

def mock_kv[C: Encoding & Shaped](c: C):
    print("  kv [" + String(C.ROWS) + "x" + String(C.COLS) + "]")


def bf16_forward_layer[tp: Int](topo: BF16Topology[tp], arena: Int, layer: Int, si: Int, fi: Int, seq: Int):
    var sb = topo.state_base(arena)
    if is_full_layer(layer):
        var lb = topo.full.base(arena, fi)
        var fl = topo.full.proto
        print("layer " + String(layer) + " (full):")
        mock_norm(fl.body.input_norm.bind(lb))
        mock_gemm(fl.attn.q_proj.bind(lb), seq)
        mock_gemm(fl.attn.k_proj.bind(lb), seq)
        print("  k_to_v_copy")
        mock_gemm(fl.attn.o_proj.bind(lb), seq)
        mock_norm(fl.body.post_attn_norm.bind(lb))
        mock_gemm(fl.body.gate_proj.bind(lb), seq)
        mock_gemm(fl.body.down_proj.bind(lb), seq)
        # DynView for activations:
        var x = topo.activations.x_main.dyn(sb, seq)
        debug_assert(x.seq_len == seq, "dyn carries seq_len")
    else:
        var lb = topo.sliding.base(arena, si)
        var sl = topo.sliding.proto
        print("layer " + String(layer) + " (sliding):")
        mock_norm(sl.body.input_norm.bind(lb))
        mock_gemm(sl.attn.q_proj.bind(lb), seq)
        mock_gemm(sl.attn.k_proj.bind(lb), seq)
        mock_gemm(sl.attn.v_proj.bind(lb), seq)
        mock_gemm(sl.attn.o_proj.bind(lb), seq)
        # KV cache access via typed ref
        var skv = topo.sliding_kv.base(sb, si)
        mock_kv(topo.sliding_kv.proto.k.bind(skv))
        mock_kv(topo.sliding_kv.proto.v.bind(skv))
        mock_norm(sl.body.post_attn_norm.bind(lb))
        mock_gemm(sl.body.gate_proj.bind(lb), seq)
        mock_gemm(sl.body.down_proj.bind(lb), seq)
        mock_gemm(sl.body.experts_gate_up.bind(lb), seq)


def quant_forward_layer[tp: Int](topo: QuantTopology[tp], arena: Int, si: Int, seq: Int):
    var lb = topo.sliding.base(arena, si)
    var sl = topo.sliding.proto
    var sb = topo.state_base(arena)
    print("quant layer " + String(si) + ":")

    mock_norm(sl.body.input_norm.bind(lb))
    # Quantized projections: access data and scale atoms independently
    mock_i8_gemv(sl.attn.q_proj.data.bind(lb), sl.attn.q_proj.scale.bind(lb))
    mock_i8_gemv(sl.attn.k_proj.data.bind(lb), sl.attn.k_proj.scale.bind(lb))
    # Colsum access: typed ref, not raw arithmetic
    print("  colsum at off=" + String(sl.attn.qkv_colsum.addr(lb)))
    mock_i8_gemv(sl.attn.o_proj.data.bind(lb), sl.attn.o_proj.scale.bind(lb))
    print("  o_colsum at off=" + String(sl.attn.o_colsum.addr(lb)))

    mock_norm(sl.body.post_attn_norm.bind(lb))
    mock_i8_gemv(sl.body.gate_proj.data.bind(lb), sl.body.gate_proj.scale.bind(lb))
    mock_i8_gemv(sl.body.down_proj.data.bind(lb), sl.body.down_proj.scale.bind(lb))
    print("  gu_colsum at off=" + String(sl.body.gu_colsum.addr(lb)))
    print("  down_colsum at off=" + String(sl.body.down_colsum.addr(lb)))

    # Init-time write targets: .addr() gives writable address
    print("  init: router_scale bake at " + String(sl.body.router_scale.addr(lb)))

    # Opaque KV cache: just an address, internal layout is cache-specific
    var cache_addr = topo.sliding_kv.base(sb, si)
    print("  packed_kv at " + String(cache_addr))

    # Activation with dynamic seq_len
    var x = topo.activations.x_main.dyn(sb, seq)
    debug_assert(x.seq_len == seq, "dyn propagates seq_len")


# =============================================================================
# Verification
# =============================================================================


def main():
    print("=== Final Design Proposal ===\n")

    # Core algebra
    comptime S = BF16Shapes[1]
    var off = SlotOffset[BF16, S.SlidingQ](100)
    var sv = off.bind(1000)
    debug_assert(sv.ptr == 1100, "static bind")
    debug_assert(sv.ROWS == A.Q_DIM_SLIDING and sv.COLS == A.HIDDEN, "type flows")
    var dv = off.dyn(1000, 42)
    debug_assert(dv.ptr == 1100 and dv.seq_len == 42, "dynamic bind")
    debug_assert(off.addr(1000) == 1100, "raw addr")
    var opq = OpaqueSlot[4096](200)
    debug_assert(opq.addr(1000) == 1200, "opaque addr")
    print("algebra: ok")

    # BF16 build
    var bf16_r = bf16_build[1]()
    var bf16_topo = bf16_r[0]
    print("\nbf16: " + String(len(bf16_r[1])) + " descs, dist=" +
          String(bf16_topo.distributed_bytes) + ", state=" + String(bf16_topo.state_bytes))

    var arena = 0x10000
    bf16_forward_layer[1](bf16_topo, arena, 0, 0, 0, 1)
    bf16_forward_layer[1](bf16_topo, arena, 5, 4, 0, 1)

    # Quant build
    var qt_r = quant_build[1]()
    var qt_topo = qt_r[0]
    print("\nquant: " + String(len(qt_r[1])) + " descs, dist=" +
          String(qt_topo.distributed_bytes) + ", state=" + String(qt_topo.state_bytes))
    print("quant host: embed_colsum.offset=" + String(qt_topo.host.embed_colsum.offset))
    print("quant host: inv_sqrt_gamma.offset=" + String(qt_topo.host.inv_sqrt_gamma.offset))

    quant_forward_layer[1](qt_topo, arena, 0, 1)

    # TP=2
    assert_tp[2]()
    comptime S2 = BF16Shapes[2]
    debug_assert(S2.SlidingQ.N == A.Q_DIM_SLIDING // 2, "tp2 shard")
    comptime QS2 = QuantShapes[2]
    comptime expected = ((A.INTERMEDIATE // 2 + VNNI_ALIGN - 1) // VNNI_ALIGN) * VNNI_ALIGN
    debug_assert(QS2.GateUp.N == expected, "tp2 VNNI align")
    print("\ntp=2: ok")

    print("\n=== all passed ===")
