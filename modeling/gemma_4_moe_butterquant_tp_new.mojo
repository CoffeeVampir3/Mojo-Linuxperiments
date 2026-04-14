"""Gemma 4 26B-A4B ButterQuant — int8 MoE, atomic layout primitives.

Everything up to the forward boundary: architecture, storage shapes,
family products with QOffset, topology with packed KV caches,
emit with colsums, build, multi-rank load, init (colsum + VNNI pack +
gamma + router baking), quantizer task emission. No forward pass.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import size_of, simd_width_of
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, BF16, F32, I8,
    Shape, ShapeLike, WeightDesc,
    DEFAULT_ALIGNMENT, HOST_RANK,
    QuantizeTask, NoQuant, Rotated, SmoothPerBlock,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, LayerShard, LayerBuilder, is_full_layer,
)
from modeling.modeling_common import (
    SlotOffset, QOffset, OpaqueSlot, Repeated, SectionBuilder, align_up,
)
from modeling.loader import discover_shards, load_weights_from_descs
from experimental3.init_weights import (
    colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at,
)
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.common_math import I8Ptr, F32Ptr, BF16Ptr
from experimental3.profiler import ForwardLogger
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables
from experimental.linear_borrow_pool import ScratchPool
from simd_math import sqrt


comptime Gemma4Config = Gemma4BaseConfig
comptime C = Gemma4Config
comptime FWHT_BLK = 64
comptime FWHT_BLK_HIDDEN = 256
comptime LM_HEAD_FWHT_BLK = 64
comptime DENSE_NUM_BLOCKS = C.INTERMEDIATE // FWHT_BLK
comptime MOE_NUM_BLOCKS = C.MOE_INTERMEDIATE // FWHT_BLK
comptime EMBED_SCALE = 53.0
comptime VNNI_ALIGN = 64


# =============================================================================
# Storage shapes — VNNI-aligned for int8 kernels
# =============================================================================


struct Gemma4Shapes[tp: Int]:
    alias GateUp      = Shape[C.INTERMEDIATE, C.HIDDEN, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    alias GateUpScale = Shape[C.INTERMEDIATE, 1, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    alias Down        = Shape[C.HIDDEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp, align_m=VNNI_ALIGN]

    alias SlidingQKV  = Shape[C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING, 1, shard_n=True, tp=Self.tp]
    alias SlidingQ    = Shape[C.Q_DIM_SLIDING, 1, shard_n=True, tp=Self.tp]
    alias FullQK      = Shape[C.Q_DIM_FULL + C.KV_DIM_FULL, 1, shard_n=True, tp=Self.tp]
    alias FullQ       = Shape[C.Q_DIM_FULL, 1, shard_n=True, tp=Self.tp]

    comptime DBLK = 64
    comptime DENSE_INT_LOCAL = Self.GateUp.N
    comptime DOWN_K = Self.Down.M
    comptime DOWN_NUM_BLK = Self.DOWN_K // Self.DBLK
    comptime FULL_HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL


# =============================================================================
# Family products — typed refs with QOffset for quantized weights
# =============================================================================


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    alias S = Gemma4Shapes[Self.tp]
    var input_norm:      SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm:    SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var gate_proj:       QOffset[Self.S.GateUp, Self.S.GateUpScale]
    var up_proj:         QOffset[Self.S.GateUp, Self.S.GateUpScale]
    var down_proj:       QOffset[Self.S.Down, Shape[C.HIDDEN, 1]]
    var router_scale:    SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var router_proj:     QOffset[Shape[C.NUM_EXPERTS, C.HIDDEN], Shape[C.NUM_EXPERTS, 1]]
    var router_pes:      SlotOffset[BF16, Shape[C.NUM_EXPERTS, 1]]
    var pre_ffn_norm_2:  SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var experts_gate_up: QOffset[Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN], Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, 1]]
    var experts_down:    QOffset[Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE], Shape[C.NUM_EXPERTS * C.HIDDEN, 1]]
    var post_attn_norm:  SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_1: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_2_rt: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm:   SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var layer_scalar:    SlotOffset[BF16, Shape[1, 1]]


@fieldwise_init
struct BodyColsumRefs[tp: Int](Copyable, ImplicitlyCopyable):
    alias S = Gemma4Shapes[Self.tp]
    var o_colsum:            SlotOffset[F32, Shape[C.HIDDEN, C.NUM_HEADS // Self.tp]]
    var gu_colsum:           SlotOffset[F32, Shape[Self.S.DENSE_INT_LOCAL * 2, 1]]
    var down_colsum:         SlotOffset[F32, Shape[C.HIDDEN, Self.S.DOWN_NUM_BLK]]
    var router_colsum:       SlotOffset[F32, Shape[C.NUM_EXPERTS, 1]]
    var experts_gu_colsum:   SlotOffset[F32, Shape[C.NUM_EXPERTS // Self.tp * C.MOE_GATE_UP_FUSED, 1]]
    var experts_down_colsum: SlotOffset[F32, Shape[C.NUM_EXPERTS // Self.tp * C.HIDDEN, MOE_NUM_BLOCKS]]


@fieldwise_init
struct SlidingAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:    QOffset[Shape[C.Q_DIM_SLIDING, C.HIDDEN], Shape[C.Q_DIM_SLIDING, 1]]
    var k_proj:    QOffset[Shape[C.KV_DIM_SLIDING, C.HIDDEN], Shape[C.KV_DIM_SLIDING, 1]]
    var v_proj:    QOffset[Shape[C.KV_DIM_SLIDING, C.HIDDEN], Shape[C.KV_DIM_SLIDING, 1]]
    var o_proj:    QOffset[Shape[C.HIDDEN, C.Q_DIM_SLIDING], Shape[C.HIDDEN, 1]]
    var q_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var k_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var qkv_colsum: SlotOffset[F32, Shape[(C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // Self.tp, 1]]


@fieldwise_init
struct FullAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:    QOffset[Shape[C.Q_DIM_FULL, C.HIDDEN], Shape[C.Q_DIM_FULL, 1]]
    var k_proj:    QOffset[Shape[C.KV_DIM_FULL, C.HIDDEN], Shape[C.KV_DIM_FULL, 1]]
    var o_proj:    QOffset[Shape[C.HIDDEN, C.Q_DIM_FULL], Shape[C.HIDDEN, 1]]
    var q_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var k_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var qk_colsum: SlotOffset[F32, Shape[(C.Q_DIM_FULL + C.KV_DIM_FULL) // Self.tp, 1]]


@fieldwise_init
struct SlidingLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: SlidingAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]
    var colsums: BodyColsumRefs[Self.tp]


@fieldwise_init
struct FullLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: FullAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]
    var colsums: BodyColsumRefs[Self.tp]


# =============================================================================
# State + host families
# =============================================================================


comptime SLIDING_CACHE_BYTES = Gemma4KVCache[
    C.SLIDING_WINDOW, C.HEAD_DIM_SLIDING,
    C.NUM_KV_HEADS_SLIDING, C.NUM_HEADS].TOTAL_BYTES

comptime FULL_CACHE_BYTES = Gemma4KVCache[
    C.MAX_SEQ_LEN, C.HEAD_DIM_FULL,
    C.NUM_KV_HEADS_FULL, C.NUM_HEADS].TOTAL_BYTES

@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]
    var sin: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]


@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main:     SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_HEAD_FWHT_BLK

@fieldwise_init
struct HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm:     SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var embed_data:     SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var embed_scale:    SlotOffset[F32, Shape[C.VOCAB_SIZE, VOCAB_NUM_BLOCKS]]
    var embed_colsum:   SlotOffset[F32, Shape[C.VOCAB_SIZE, VOCAB_NUM_BLOCKS]]
    var sqrt_gamma:     SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var inv_sqrt_gamma: SlotOffset[F32, Shape[C.HIDDEN, 1]]


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct Gemma4Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var sliding: Repeated[SlidingLayerRefs[Self.tp]]
    var full: Repeated[FullLayerRefs[Self.tp]]
    var distributed_bytes: Int

    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_rope: RopeSlots[C.HEAD_DIM_SLIDING // 2]
    var full_rope: RopeSlots[64]
    var sliding_cache_off: Int
    var sliding_cache_stride: Int
    var full_cache_off: Int
    var full_cache_stride: Int
    var state_bytes: Int

    var host: HostSlots
    var host_bytes: Int

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    def state_base(self, arena_base: Int) -> Int:
        return arena_base + self.distributed_bytes

    def scratch_base(self, arena_base: Int) -> Int:
        return self.state_base(arena_base) + self.scratch_off


# =============================================================================
# Emit helpers
# =============================================================================


def emit_qoff[DS: ShapeLike, SS: ShapeLike](
    mut b: LayerBuilder, mut e: List[WeightDesc],
    data_sfx: String, scale_sfx: String,
    data_rows: Int, data_cols: Int, scale_rows: Int, scale_cols: Int,
    shard: Int,
) -> QOffset[DS, SS]:
    return QOffset[DS, SS](
        data=SlotOffset[I8, DS](b.q(e, data_sfx, data_rows, data_cols, shard)),
        scale=SlotOffset[F32, SS](b.f(e, scale_sfx, scale_rows, scale_cols,
            LayerShard.REPL if shard == LayerShard.COL else shard)))


def emit_qoff_shaped[DS: ShapeLike, SS: ShapeLike](
    mut b: LayerBuilder, mut e: List[WeightDesc],
    data_sfx: String, scale_sfx: String,
) -> QOffset[DS, SS]:
    return QOffset[DS, SS](
        data=SlotOffset[I8, DS](b.qs[DS](e, data_sfx)),
        scale=SlotOffset[F32, SS](b.fs[SS](e, scale_sfx)))


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime ROW = LayerShard.ROW
    comptime COL = LayerShard.COL
    comptime REPL = LayerShard.REPL
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    comptime GU = C.MOE_GATE_UP_FUSED
    comptime MI = C.MOE_INTERMEDIATE
    alias S = Gemma4Shapes[tp]

    return BodyRefs[tp](
        input_norm      = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "input_layernorm.weight", H, 1, REPL)),
        pre_ffn_norm    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm.weight", H, 1, REPL)),
        gate_proj       = emit_qoff_shaped[S.GateUp, S.GateUpScale](b, e, "mlp.gate_proj.weight", "mlp.gate_proj.weight_scale"),
        up_proj         = emit_qoff_shaped[S.GateUp, S.GateUpScale](b, e, "mlp.up_proj.weight", "mlp.up_proj.weight_scale"),
        down_proj       = QOffset[S.Down, Shape[H, 1]](
            data=SlotOffset[I8, S.Down](b.qs[S.Down](e, "mlp.down_proj.weight")),
            scale=SlotOffset[F32, Shape[H, 1]](b.f(e, "mlp.down_proj.weight_scale", H, 1, REPL))),
        router_scale    = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "router.scale", H, 1, REPL)),
        router_proj     = emit_qoff[Shape[NE, H], Shape[NE, 1]](b, e,
            "router.proj.weight", "router.proj.weight_scale", NE, H, REPL, NE, 1),
        router_pes      = SlotOffset[BF16, Shape[NE, 1]](b.bf(e, "router.per_expert_scale", NE, 1, REPL)),
        pre_ffn_norm_2  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "pre_feedforward_layernorm_2.weight", H, 1, REPL)),
        experts_gate_up = emit_qoff[Shape[NE * GU, H], Shape[NE * GU, 1]](b, e,
            "experts.gate_up_proj", "experts.gate_up_proj_scale", NE * GU, H, ROW, NE * GU, 1),
        experts_down    = emit_qoff[Shape[NE * H, MI], Shape[NE * H, 1]](b, e,
            "experts.down_proj", "experts.down_proj_scale", NE * H, MI, ROW, NE * H, 1),
        post_attn_norm  = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_attention_layernorm.weight", H, 1, REPL)),
        post_ffn_norm_1 = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_1.weight", H, 1, REPL)),
        post_ffn_norm_2_rt = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm_2.weight", H, 1, REPL)),
        post_ffn_norm   = SlotOffset[BF16, Shape[H, 1]](b.bf(e, "post_feedforward_layernorm.weight", H, 1, REPL)),
        layer_scalar    = SlotOffset[BF16, Shape[1, 1]](b.bf(e, "layer_scalar", 1, 1, REPL)),
    )


def emit_body_colsums[tp: Int](mut b: LayerBuilder) -> BodyColsumRefs[tp]:
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    comptime GU = C.MOE_GATE_UP_FUSED
    comptime o_num_blk = C.NUM_HEADS // tp
    comptime experts_local = NE // tp
    alias S = Gemma4Shapes[tp]
    return BodyColsumRefs[tp](
        o_colsum            = SlotOffset[F32, Shape[H, o_num_blk]](b.colsum(H * o_num_blk * 4)),
        gu_colsum           = SlotOffset[F32, Shape[S.DENSE_INT_LOCAL * 2, 1]](b.colsum(S.DENSE_INT_LOCAL * 2 * 4)),
        down_colsum         = SlotOffset[F32, Shape[H, S.DOWN_NUM_BLK]](b.colsum(H * S.DOWN_NUM_BLK * 4)),
        router_colsum       = SlotOffset[F32, Shape[NE, 1]](b.colsum(NE * 4)),
        experts_gu_colsum   = SlotOffset[F32, Shape[experts_local * GU, 1]](b.colsum(experts_local * GU * 4)),
        experts_down_colsum = SlotOffset[F32, Shape[experts_local * H, MOE_NUM_BLOCKS]](
            b.colsum(experts_local * H * MOE_NUM_BLOCKS * 4)),
    )


def emit_sliding[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[SlidingLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW = LayerShard.ROW
    comptime COL = LayerShard.COL
    comptime REPL = LayerShard.REPL
    comptime H = C.HIDDEN

    var attn = SlidingAttnRefs[tp](
        q_proj  = emit_qoff[Shape[C.Q_DIM_SLIDING, H], Shape[C.Q_DIM_SLIDING, 1]](b, e,
            "self_attn.q_proj.weight", "self_attn.q_proj.weight_scale", C.Q_DIM_SLIDING, H, ROW, C.Q_DIM_SLIDING, 1),
        k_proj  = emit_qoff[Shape[C.KV_DIM_SLIDING, H], Shape[C.KV_DIM_SLIDING, 1]](b, e,
            "self_attn.k_proj.weight", "self_attn.k_proj.weight_scale", C.KV_DIM_SLIDING, H, ROW, C.KV_DIM_SLIDING, 1),
        v_proj  = emit_qoff[Shape[C.KV_DIM_SLIDING, H], Shape[C.KV_DIM_SLIDING, 1]](b, e,
            "self_attn.v_proj.weight", "self_attn.v_proj.weight_scale", C.KV_DIM_SLIDING, H, ROW, C.KV_DIM_SLIDING, 1),
        o_proj  = emit_qoff[Shape[H, C.Q_DIM_SLIDING], Shape[H, 1]](b, e,
            "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale", H, C.Q_DIM_SLIDING, COL, H, 1),
        q_norm  = SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.q_norm.weight", C.HEAD_DIM_SLIDING, 1, REPL)),
        k_norm  = SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.k_norm.weight", C.HEAD_DIM_SLIDING, 1, REPL)),
        qkv_colsum = SlotOffset[F32, Shape[(C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // tp, 1]](
            b.colsum((C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // tp * 4)),
    )
    var body = emit_body[tp](b, e)
    var colsums = emit_body_colsums[tp](b)
    return (SlidingLayerRefs[tp](attn=attn, body=body, colsums=colsums), b.cursor)


def emit_full[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW = LayerShard.ROW
    comptime COL = LayerShard.COL
    comptime REPL = LayerShard.REPL
    comptime H = C.HIDDEN

    var attn = FullAttnRefs[tp](
        q_proj  = emit_qoff[Shape[C.Q_DIM_FULL, H], Shape[C.Q_DIM_FULL, 1]](b, e,
            "self_attn.q_proj.weight", "self_attn.q_proj.weight_scale", C.Q_DIM_FULL, H, ROW, C.Q_DIM_FULL, 1),
        k_proj  = emit_qoff[Shape[C.KV_DIM_FULL, H], Shape[C.KV_DIM_FULL, 1]](b, e,
            "self_attn.k_proj.weight", "self_attn.k_proj.weight_scale", C.KV_DIM_FULL, H, ROW, C.KV_DIM_FULL, 1),
        o_proj  = emit_qoff[Shape[H, C.Q_DIM_FULL], Shape[H, 1]](b, e,
            "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale", H, C.Q_DIM_FULL, COL, H, 1),
        q_norm  = SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.q_norm.weight", C.HEAD_DIM_FULL, 1, REPL)),
        k_norm  = SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.k_norm.weight", C.HEAD_DIM_FULL, 1, REPL)),
        qk_colsum = SlotOffset[F32, Shape[(C.Q_DIM_FULL + C.KV_DIM_FULL) // tp, 1]](
            b.colsum((C.Q_DIM_FULL + C.KV_DIM_FULL) // tp * 4)),
    )
    var body = emit_body[tp](b, e)
    var colsums = emit_body_colsums[tp](b)
    return (FullLayerRefs[tp](attn=attn, body=body, colsums=colsums), b.cursor)


# =============================================================================
# Scratch budget
# =============================================================================


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16 = 2
    comptime i8 = 1
    comptime f32 = 4
    comptime topk_bytes = size_of[Scalar[DType.int32]]() * 8 + 8 * 4
    alias S = Gemma4Shapes[tp]

    comptime persistent = f32 + S.DOWN_NUM_BLK * f32

    comptime sliding_attn_peak = persistent + (
        C.HIDDEN * i8 + C.HIDDEN * f32 + f32
        + S.SlidingQKV.col_bytes_for[bf16]()
        + S.SlidingQ.col_bytes_for[i8]()
        + (S.SlidingQ.N // C.HEAD_DIM_SLIDING) * f32)

    comptime full_attn_phase1 = persistent + (
        S.FullQK.col_bytes_for[bf16]()
        + C.HIDDEN * i8 + C.HIDDEN * f32 + f32)
    comptime full_attn_phase2 = persistent + (
        S.FullQK.col_bytes_for[bf16]()
        + S.FullQ.col_bytes_for[i8]()
        + (S.FullQ.N // C.HEAD_DIM_FULL) * f32
        + S.FULL_HPG * C.HEAD_DIM_FULL * i8
        + S.FULL_HPG * f32 * 2
        + 32 * S.FULL_HPG * (2 + C.HEAD_DIM_FULL) * f32)
    comptime full_attn_peak = full_attn_phase1 if full_attn_phase1 > full_attn_phase2 else full_attn_phase2

    comptime ffn_peak = persistent + (
        C.HIDDEN * i8 + C.HIDDEN * f32 + C.HIDDEN * f32
        + C.HIDDEN * i8 + f32
        + C.NUM_EXPERTS * bf16
        + topk_bytes
        + C.TOP_K * C.MOE_INTERMEDIATE * i8
        + C.TOP_K * MOE_NUM_BLOCKS * f32
        + C.TOP_K * C.HIDDEN * bf16
        + size_of[Int32]()
        + S.GateUp.col_bytes_for[i8]()
        + C.HIDDEN * bf16 * 2)

    comptime layer_peak = sliding_attn_peak if sliding_attn_peak > full_attn_peak else full_attn_peak
    comptime decode_peak = ffn_peak if ffn_peak > layer_peak else layer_peak

    comptime lm_head_peak = (
        C.HIDDEN * i8 + VOCAB_NUM_BLOCKS * f32
        + C.HIDDEN * f32 + C.VOCAB_SIZE * bf16)
    return lm_head_peak if lm_head_peak > decode_peak else decode_peak


# =============================================================================
# Build plan
# =============================================================================


@fieldwise_init
struct Gemma4LoadPlan[tp: Int](Movable):
    var topology: Gemma4Topology[Self.tp]
    var descs: List[WeightDesc]


def build_gemma4_plan[tp: Int]() -> Gemma4LoadPlan[tp]:
    debug_assert(C.NUM_HEADS % tp == 0, "NUM_HEADS % tp")
    debug_assert(C.NUM_KV_HEADS_SLIDING % tp == 0, "NUM_KV_HEADS_SLIDING % tp")
    debug_assert(C.NUM_KV_HEADS_FULL % tp == 0, "NUM_KV_HEADS_FULL % tp")
    debug_assert(C.INTERMEDIATE % tp == 0, "INTERMEDIATE % tp")
    debug_assert(C.NUM_EXPERTS % tp == 0, "NUM_EXPERTS % tp")

    var descs = List[WeightDesc]()
    var probe = List[WeightDesc]()
    var sl_r = emit_sliding[tp]("", 0, probe)
    var fl_r = emit_full[tp]("", 0, probe)
    var sl_proto = sl_r[0]
    var sl_stride = sl_r[1]
    var fl_proto = fl_r[0]
    var fl_stride = fl_r[1]

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    var si = 0
    var fi = 0
    for i in range(C.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if is_full_layer(i):
            _ = emit_full[tp](prefix, fl_off + fi * fl_stride, descs)
            fi += 1
        else:
            _ = emit_sliding[tp](prefix, sl_off + si * sl_stride, descs)
            si += 1

    # State
    var state = SectionBuilder()

    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]())

    var scratch_cap = calculate_peak_scratch[tp]()
    var scratch_off = state.reserve_bytes(scratch_cap)

    var sliding_rope = RopeSlots[C.HEAD_DIM_SLIDING // 2](
        cos=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM_SLIDING // 2]](),
        sin=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM_SLIDING // 2]]())
    var full_rope = RopeSlots[64](
        cos=state.reserve[F32, Shape[C.MAX_SEQ_LEN, 64]](),
        sin=state.reserve[F32, Shape[C.MAX_SEQ_LEN, 64]]())

    comptime sliding_cache_stride = Gemma4KVCache[
        C.SLIDING_WINDOW, C.HEAD_DIM_SLIDING,
        C.NUM_KV_HEADS_SLIDING // tp, C.NUM_HEADS // tp].TOTAL_BYTES
    comptime full_cache_stride = Gemma4KVCache[
        C.MAX_SEQ_LEN, C.HEAD_DIM_FULL,
        C.NUM_KV_HEADS_FULL // tp, C.NUM_HEADS // tp].TOTAL_BYTES
    var sliding_cache_off = state.reserve_bytes(C.NUM_SLIDING_LAYERS * sliding_cache_stride)
    var full_cache_off = state.reserve_bytes(C.NUM_FULL_LAYERS * full_cache_stride)

    # Host
    var host_off = align_up(distributed + state.bytes())
    comptime HOST = LayerShard.HOST
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var final_norm_off = hb.bf(descs, "model.language_model.norm.weight", C.HIDDEN, 1, HOST)
    var embed_data_off = hb.q(descs, "model.language_model.embed_tokens.weight", C.VOCAB_SIZE, C.HIDDEN, HOST)
    var embed_scale_off = hb.f(descs, "model.language_model.embed_tokens.weight_scale", C.VOCAB_SIZE, VOCAB_NUM_BLOCKS, HOST)
    var embed_colsum_off = hb.colsum(C.VOCAB_SIZE * VOCAB_NUM_BLOCKS * 4)
    var sqrt_gamma_off = hb.colsum(C.HIDDEN * 2)
    var inv_sqrt_gamma_off = hb.colsum(C.HIDDEN * 4)

    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](final_norm_off),
        embed_data=SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]](embed_data_off),
        embed_scale=SlotOffset[F32, Shape[C.VOCAB_SIZE, VOCAB_NUM_BLOCKS]](embed_scale_off),
        embed_colsum=SlotOffset[F32, Shape[C.VOCAB_SIZE, VOCAB_NUM_BLOCKS]](embed_colsum_off),
        sqrt_gamma=SlotOffset[BF16, Shape[C.HIDDEN, 1]](sqrt_gamma_off),
        inv_sqrt_gamma=SlotOffset[F32, Shape[C.HIDDEN, 1]](inv_sqrt_gamma_off))

    var topo = Gemma4Topology[tp](
        sliding=Repeated[SlidingLayerRefs[tp]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[tp]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        distributed_bytes=distributed,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        sliding_cache_off=sliding_cache_off, sliding_cache_stride=sliding_cache_stride,
        full_cache_off=full_cache_off, full_cache_stride=full_cache_stride,
        state_bytes=state.bytes(),
        host=host, host_bytes=hb.cursor)
    return Gemma4LoadPlan[tp](topo, descs^)


# =============================================================================
# Init: colsum, VNNI pack, gamma, router baking
# =============================================================================


def init_layer_body[tp: Int](
    arena_base: Int, layer_base: Int,
    body: BodyRefs[tp], colsums: BodyColsumRefs[tp],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    comptime experts_local = C.NUM_EXPERTS // tp
    alias S = Gemma4Shapes[tp]

    colsum_at(arena_base, body.gate_proj.data.addr(layer_base), colsums.gu_colsum.addr(layer_base),
        S.DENSE_INT_LOCAL * 2, C.HIDDEN)
    block_colsum_at(arena_base, body.down_proj.data.addr(layer_base), colsums.down_colsum.addr(layer_base),
        C.HIDDEN, S.DOWN_K, S.DBLK)
    colsum_at(arena_base, body.router_proj.data.addr(layer_base), colsums.router_colsum.addr(layer_base),
        C.NUM_EXPERTS, C.HIDDEN)
    colsum_at(arena_base, body.experts_gate_up.data.addr(layer_base), colsums.experts_gu_colsum.addr(layer_base),
        experts_local * C.MOE_GATE_UP_FUSED, C.HIDDEN)
    for e in range(experts_local):
        block_colsum_at(arena_base,
            body.experts_down.data.addr(layer_base) + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            colsums.experts_down_colsum.addr(layer_base) + e * C.HIDDEN * MOE_NUM_BLOCKS * 4,
            C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK)

    pack_at(arena_base, body.gate_proj.data.addr(layer_base), S.DENSE_INT_LOCAL * 2, C.HIDDEN, scratch)
    pack_at(arena_base, body.down_proj.data.addr(layer_base), C.HIDDEN, S.DOWN_K, scratch)
    pack_at(arena_base, body.router_proj.data.addr(layer_base), C.NUM_EXPERTS, C.HIDDEN, scratch)
    for e in range(experts_local):
        pack_at(arena_base, body.experts_gate_up.data.addr(layer_base) + e * C.MOE_GATE_UP_FUSED * C.HIDDEN,
            C.MOE_GATE_UP_FUSED, C.HIDDEN, scratch)
        pack_at(arena_base, body.experts_down.data.addr(layer_base) + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            C.HIDDEN, C.MOE_INTERMEDIATE, scratch)


def init_sliding_layer[tp: Int](
    arena_base: Int, layer_base: Int,
    layer: SlidingLayerRefs[tp],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    comptime qkv_n_local = (C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // tp
    comptime q_local = C.Q_DIM_SLIDING // tp

    colsum_at(arena_base, layer.attn.q_proj.data.addr(layer_base), layer.attn.qkv_colsum.addr(layer_base),
        qkv_n_local, C.HIDDEN)
    block_colsum_at(arena_base, layer.attn.o_proj.data.addr(layer_base), layer.colsums.o_colsum.addr(layer_base),
        C.HIDDEN, q_local, C.HEAD_DIM_SLIDING)

    init_layer_body[tp](arena_base, layer_base, layer.body, layer.colsums, scratch)

    pack_at(arena_base, layer.attn.q_proj.data.addr(layer_base), qkv_n_local, C.HIDDEN, scratch)
    pack_at(arena_base, layer.attn.o_proj.data.addr(layer_base), C.HIDDEN, q_local, scratch)


def init_full_layer[tp: Int](
    arena_base: Int, layer_base: Int,
    layer: FullLayerRefs[tp],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    comptime qk_n_local = (C.Q_DIM_FULL + C.KV_DIM_FULL) // tp
    comptime q_local = C.Q_DIM_FULL // tp

    colsum_at(arena_base, layer.attn.q_proj.data.addr(layer_base), layer.attn.qk_colsum.addr(layer_base),
        qk_n_local, C.HIDDEN)
    block_colsum_at(arena_base, layer.attn.o_proj.data.addr(layer_base), layer.colsums.o_colsum.addr(layer_base),
        C.HIDDEN, q_local, C.HEAD_DIM_FULL)

    init_layer_body[tp](arena_base, layer_base, layer.body, layer.colsums, scratch)

    pack_at(arena_base, layer.attn.q_proj.data.addr(layer_base), qk_n_local, C.HIDDEN, scratch)
    pack_at(arena_base, layer.attn.o_proj.data.addr(layer_base), C.HIDDEN, q_local, scratch)


# =============================================================================
# Model
# =============================================================================


struct Gemma4ButterQuant[tp: Int](Movable):
    comptime MAX_PACK_BYTES = (C.Q_DIM_FULL + C.KV_DIM_FULL) * C.HIDDEN

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var profile: ForwardLogger
    var topology: Gemma4Topology[Self.tp]

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[BurstPool[]],
        var scratch: ScratchPool,
        bases: InlineArray[Int, Self.tp],
        topology: Gemma4Topology[Self.tp],
    ):
        self.arenas = arenas^
        self.main_pools = pools^
        self.scratch = scratch^
        self.bases = bases
        self.profile = ForwardLogger()
        self.topology = topology

    def init_state(mut self):
        var topo = self.topology
        var pack_scratch = alloc[UInt8](Self.MAX_PACK_BYTES)

        for rank in range(Self.tp):
            var base = self.bases[rank]
            var sb = topo.state_base(base)

            init_sliding_rope_tables(
                topo.sliding_rope.cos.bound(sb),
                topo.sliding_rope.sin.bound(sb))
            init_full_rope_tables(
                topo.full_rope.cos.bound(sb),
                topo.full_rope.sin.bound(sb))

            var si = 0
            var fi = 0
            for i in range(C.NUM_LAYERS):
                if is_full_layer(i):
                    var lb = topo.full.base(base, fi)
                    init_full_layer[Self.tp](base, lb, topo.full.proto, pack_scratch)
                    fi += 1
                else:
                    var lb = topo.sliding.base(base, si)
                    init_sliding_layer[Self.tp](base, lb, topo.sliding.proto, pack_scratch)
                    si += 1

            if rank == HOST_RANK:
                block_colsum_row_major_at(base,
                    topo.host.embed_data.addr(base),
                    topo.host.embed_colsum.addr(base),
                    C.VOCAB_SIZE, C.HIDDEN, LM_HEAD_FWHT_BLK)

                var fn_gamma = BF16Ptr(unsafe_from_address=topo.host.final_norm.addr(base))
                compute_sqrt_gamma[C.HIDDEN](fn_gamma,
                    BF16Ptr(unsafe_from_address=topo.host.sqrt_gamma.addr(base)))
                compute_inv_sqrt_gamma[C.HIDDEN](fn_gamma,
                    F32Ptr(unsafe_from_address=topo.host.inv_sqrt_gamma.addr(base)))

            comptime inv_sqrt_h = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
            si = 0
            fi = 0
            for i in range(C.NUM_LAYERS):
                var sc_addr: Int
                if is_full_layer(i):
                    var lb = topo.full.base(base, fi)
                    sc_addr = topo.full.proto.body.router_proj.scale.addr(lb)
                    fi += 1
                else:
                    var lb = topo.sliding.base(base, si)
                    sc_addr = topo.sliding.proto.body.router_proj.scale.addr(lb)
                    si += 1
                var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sc_addr)
                for n in range(C.NUM_EXPERTS):
                    sc_ptr[n] *= inv_sqrt_h

        pack_scratch.free()
        print("state initialized")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_gemma4_plan[Self.tp]()
        var topo = plan.topology

        var numa = NumaInfo()
        var nodes = numa.plan_topology(Self.tp)

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = topo.host_arena_bytes() if rank == HOST_RANK else topo.arena_bytes()
            print("rank", rank, "node", nodes[rank], "allocating", size // (1024 * 1024), "MB")
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](nodes[rank], size)
            if not arena:
                print("arena allocation failed for rank", rank)
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_weights_from_descs(plan.descs, shards, arena_bases)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(topo.distributed_bytes, topo.state_bytes)

        var pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool[].for_numa_node(numa, nodes[rank]))

        var bases = InlineArray[Int, Self.tp](fill=0)
        for rank in range(Self.tp):
            bases[rank] = Int(arenas[rank].base)

        var scratch = ScratchPool(topo.scratch_capacity)
        var model = Self(arenas^, pools^, scratch^, bases, topo)
        model.init_state()
        return model^

    @staticmethod
    def build_quantizer_tasks() -> List[QuantizeTask]:
        var tasks = List[QuantizeTask]()
        comptime HB = FWHT_BLK_HIDDEN
        var pt = NoQuant().to_op()
        var rot_hb = Rotated(HB).to_op()

        for i in range(C.NUM_LAYERS):
            var p = "model.language_model.layers." + String(i) + "."
            var full = is_full_layer(i)

            tasks.append(QuantizeTask(p + "self_attn.q_proj.weight", rot_hb))
            tasks.append(QuantizeTask(p + "self_attn.k_proj.weight", rot_hb))
            if not full:
                tasks.append(QuantizeTask(p + "self_attn.v_proj.weight", rot_hb))
            tasks.append(QuantizeTask(p + "self_attn.o_proj.weight",
                Rotated(C.HEAD_DIM_FULL).to_op() if full else rot_hb))

            tasks.append(QuantizeTask(p + "mlp.gate_proj.weight", rot_hb))
            tasks.append(QuantizeTask(p + "mlp.up_proj.weight", rot_hb))
            tasks.append(QuantizeTask(p + "mlp.down_proj.weight", Rotated(FWHT_BLK).to_op()))
            tasks.append(QuantizeTask(p + "router.proj.weight", rot_hb))
            tasks.append(QuantizeTask(p + "experts.gate_up_proj", Rotated(HB).to_op()))
            tasks.append(QuantizeTask(p + "experts.down_proj", Rotated(FWHT_BLK).to_op()))

            tasks.append(QuantizeTask(p + "input_layernorm.weight", pt))
            tasks.append(QuantizeTask(p + "post_attention_layernorm.weight", pt))
            tasks.append(QuantizeTask(p + "pre_feedforward_layernorm.weight", pt))
            tasks.append(QuantizeTask(p + "pre_feedforward_layernorm_2.weight", pt))
            tasks.append(QuantizeTask(p + "post_feedforward_layernorm.weight", pt))
            tasks.append(QuantizeTask(p + "post_feedforward_layernorm_1.weight", pt))
            tasks.append(QuantizeTask(p + "post_feedforward_layernorm_2.weight", pt))
            tasks.append(QuantizeTask(p + "self_attn.q_norm.weight", pt))
            tasks.append(QuantizeTask(p + "self_attn.k_norm.weight", pt))
            tasks.append(QuantizeTask(p + "router.scale", pt))
            tasks.append(QuantizeTask(p + "router.per_expert_scale", pt))
            tasks.append(QuantizeTask(p + "layer_scalar", pt))

        tasks.append(QuantizeTask("model.language_model.norm.weight", pt))
        tasks.append(QuantizeTask("model.language_model.embed_tokens.weight",
            SmoothPerBlock(LM_HEAD_FWHT_BLK, "model.language_model.norm.weight").to_op()))
        return tasks^


def main():
    print("gemma_4_moe_butterquant_tp_new: module")
