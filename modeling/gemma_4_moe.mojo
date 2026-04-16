"""Gemma 4 26B-A4B — bf16 model, atomic layout primitives.

Architecture, storage shapes, family products, topology, emit, build,
load, init, and forward — all bound directly from topology with no
RankView or legacy layout bags.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math import sqrt

from numa import NumaArena, NumaInfo
from threading import BurstPool
from modeling.linear_borrow_pool import ScratchPool, ScratchLease

from modeling.model_spec import (
    Encoding, BF16, F32,
    Shape, ShapeLike, Mat, DynView, CacheView, WeightDesc,
    DEFAULT_ALIGNMENT, LogitsView,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, LayerShard, LayerBuilder, is_full_layer,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    StaticTensorView, DynamicTensorView,
    static_tensor_view, dynamic_tensor_view,
    borrow_scratch_tensor, scratch_tensor_view, scratch_ptr,
)
from modeling.loader import discover_shards, load_weights_from_descs
from kernels.kernel_ops import (
    gemm, rmsnorm, elem_add, kv_cache_write,
    gemv_kernel, GemmArgs,
    BF16Ptr,
)
from kernels.kv_rotors import rope

from experimental_gemma.activations import gelu_tanh_mul
from experimental3.kernels.dispatch_kernels import rmsnorm_no_scale, rmsnorm_per_head
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables, apply_full_rope
from experimental_gemma.router import softmax_topk_renorm
from experimental_gemma.moe import gemma4_moe_dispatch
from experimental_gemma.attention import local_attention, global_attention
from experimental_gemma.ops import embed_lookup_scaled, logit_softcap, elem_scale


comptime Gemma4Config = Gemma4BaseConfig
comptime C = Gemma4Config
comptime EMBED_SCALE = sqrt[DType.float32, 1](C.HIDDEN)


# =============================================================================
# Storage shapes
# =============================================================================


struct Gemma4Shapes[tp: Int]:
    comptime GateUp      = Shape[C.INTERMEDIATE, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime Down        = Shape[C.HIDDEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp]
    comptime SlidingQ    = Shape[C.Q_DIM_SLIDING, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingKV   = Shape[C.KV_DIM_SLIDING, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingO    = Shape[C.HIDDEN, C.Q_DIM_SLIDING, shard_m=True, tp=Self.tp]
    comptime FullQ       = Shape[C.Q_DIM_FULL, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullK       = Shape[C.KV_DIM_FULL, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullO       = Shape[C.HIDDEN, C.Q_DIM_FULL, shard_m=True, tp=Self.tp]
    comptime ExpertsGateUp = Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime ExpertsDown   = Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE, shard_n=True, tp=Self.tp]


# =============================================================================
# Family products — typed refs, no stride, no views
# =============================================================================


@fieldwise_init
struct SlidingAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.tp]
    var q_proj: SlotOffset[BF16, Self.S.SlidingQ]
    var k_proj: SlotOffset[BF16, Self.S.SlidingKV]
    var v_proj: SlotOffset[BF16, Self.S.SlidingKV]
    var o_proj: SlotOffset[BF16, Self.S.SlidingO]
    var q_norm: SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var k_norm: SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]


@fieldwise_init
struct FullAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.tp]
    var q_proj: SlotOffset[BF16, Self.S.FullQ]
    var k_proj: SlotOffset[BF16, Self.S.FullK]
    var o_proj: SlotOffset[BF16, Self.S.FullO]
    var q_norm: SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var k_norm: SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.tp]
    var input_norm:      SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm:  SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm:    SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm_2:  SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_1: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_2: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm:   SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var gate_proj:       SlotOffset[BF16, Self.S.GateUp]
    var up_proj:         SlotOffset[BF16, Self.S.GateUp]
    var down_proj:       SlotOffset[BF16, Self.S.Down]
    var router_proj:     SlotOffset[BF16, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    var router_scale:    SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var router_pes:      SlotOffset[BF16, Shape[C.NUM_EXPERTS, 1]]
    var experts_gate_up: SlotOffset[BF16, Self.S.ExpertsGateUp]
    var experts_down:    SlotOffset[BF16, Self.S.ExpertsDown]
    var layer_scalar:    SlotOffset[BF16, Shape[1, 1]]


@fieldwise_init
struct SlidingLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: SlidingAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


@fieldwise_init
struct FullLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: FullAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


# =============================================================================
# State + host families
# =============================================================================


@fieldwise_init
struct SlidingKVSlots(Copyable, ImplicitlyCopyable):
    var k: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]]
    var v: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]]


@fieldwise_init
struct FullKVSlots(Copyable, ImplicitlyCopyable):
    var k: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_FULL]]
    var v: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_FULL]]


@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]
    var sin: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]


@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main:     SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


@fieldwise_init
struct HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var embed:      SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct Gemma4Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena_base: Int
    var sliding: Repeated[SlidingLayerRefs[Self.tp]]
    var full: Repeated[FullLayerRefs[Self.tp]]
    var distributed_bytes: Int

    var sliding_kv: Repeated[SlidingKVSlots]
    var full_kv: Repeated[FullKVSlots]
    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_rope: RopeSlots[C.HEAD_DIM_SLIDING // 2]
    var full_rope: RopeSlots[64]
    var state_bytes: Int

    var host: HostSlots
    var host_bytes: Int

    def bind(self, base: Int) -> Self:
        var t = self
        t.arena_base = base
        return t

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    @always_inline
    def sliding_base(self, idx: Int) -> Int:
        return self.arena_base + self.sliding.off + idx * self.sliding.stride

    @always_inline
    def full_base(self, idx: Int) -> Int:
        return self.arena_base + self.full.off + idx * self.full.stride

    @always_inline
    def state_base(self) -> Int:
        return self.arena_base + self.distributed_bytes

    @always_inline
    def scratch_base(self) -> Int:
        return self.state_base() + self.scratch_off

    @always_inline
    def scratch_addr(self, read lease: ScratchLease) -> Int:
        return self.scratch_base() + lease.offset

    @always_inline
    def x_main(self, seq_len: Int) -> DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]:
        return dynamic_tensor_view(self.state_base(), self.activations.x_main, seq_len)

    @always_inline
    def x_residual(self, seq_len: Int) -> DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]:
        return dynamic_tensor_view(self.state_base(), self.activations.x_residual, seq_len)

    @always_inline
    def sliding_k_cache(self, sliding_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]]:
        var kb = self.sliding_kv.base(self.state_base(), sliding_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]](
            kb + self.sliding_kv.proto.k.offset)

    @always_inline
    def sliding_v_cache(self, sliding_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]]:
        var kb = self.sliding_kv.base(self.state_base(), sliding_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]](
            kb + self.sliding_kv.proto.v.offset)

    @always_inline
    def full_k_cache(self, full_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]]:
        var kb = self.full_kv.base(self.state_base(), full_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]](
            kb + self.full_kv.proto.k.offset)

    @always_inline
    def full_v_cache(self, full_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]]:
        var kb = self.full_kv.base(self.state_base(), full_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]](
            kb + self.full_kv.proto.v.offset)


# =============================================================================
# Emit
# =============================================================================


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime S = Gemma4Shapes[tp]
    comptime R = LayerShard.REPL
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    return BodyRefs[tp](
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
        experts_gate_up = SlotOffset[BF16, S.ExpertsGateUp](b.bfs[S.ExpertsGateUp](e, "experts.gate_up_proj")),
        experts_down    = SlotOffset[BF16, S.ExpertsDown](b.bfs[S.ExpertsDown](e, "experts.down_proj")),
        layer_scalar    = SlotOffset[BF16, Shape[1, 1]](b.bf(e, "layer_scalar", 1, 1, R)),
    )


def emit_sliding[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[SlidingLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime S = Gemma4Shapes[tp]
    comptime R = LayerShard.REPL
    var attn = SlidingAttnRefs[tp](
        q_proj = SlotOffset[BF16, S.SlidingQ](b.bfs[S.SlidingQ](e, "self_attn.q_proj.weight")),
        k_proj = SlotOffset[BF16, S.SlidingKV](b.bfs[S.SlidingKV](e, "self_attn.k_proj.weight")),
        v_proj = SlotOffset[BF16, S.SlidingKV](b.bfs[S.SlidingKV](e, "self_attn.v_proj.weight")),
        o_proj = SlotOffset[BF16, S.SlidingO](b.bfs[S.SlidingO](e, "self_attn.o_proj.weight")),
        q_norm = SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.q_norm.weight", C.HEAD_DIM_SLIDING, 1, R)),
        k_norm = SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]](b.bf(e, "self_attn.k_norm.weight", C.HEAD_DIM_SLIDING, 1, R)),
    )
    return (SlidingLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def emit_full[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime S = Gemma4Shapes[tp]
    comptime R = LayerShard.REPL
    var attn = FullAttnRefs[tp](
        q_proj = SlotOffset[BF16, S.FullQ](b.bfs[S.FullQ](e, "self_attn.q_proj.weight")),
        k_proj = SlotOffset[BF16, S.FullK](b.bfs[S.FullK](e, "self_attn.k_proj.weight")),
        o_proj = SlotOffset[BF16, S.FullO](b.bfs[S.FullO](e, "self_attn.o_proj.weight")),
        q_norm = SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.q_norm.weight", C.HEAD_DIM_FULL, 1, R)),
        k_norm = SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]](b.bf(e, "self_attn.k_norm.weight", C.HEAD_DIM_FULL, 1, R)),
    )
    return (FullLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


# =============================================================================
# Scratch budget
# =============================================================================


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16 = BF16.ELEMENT_BYTES
    comptime seq = C.MAX_SEQ_LEN
    comptime S = Gemma4Shapes[tp]

    # Match the actual live set in forward:
    # q+k+v are live together, then k/v are released before attn_out is borrowed.
    comptime full_q = seq * S.FullQ.N * bf16
    comptime full_kv = seq * S.FullK.N * bf16
    comptime full_attn = (
        full_q + full_kv + full_kv
        if full_q + full_kv + full_kv > full_q + full_q
        else full_q + full_q
    )
    comptime sliding_q = seq * S.SlidingQ.N * bf16
    comptime sliding_kv = seq * S.SlidingKV.N * bf16
    comptime sliding_attn = (
        sliding_q + sliding_kv + sliding_kv
        if sliding_q + sliding_kv + sliding_kv > sliding_q + sliding_q
        else sliding_q + sliding_q
    )
    comptime ffn_dense = seq * S.GateUp.N * bf16 * 2
    comptime ffn_moe = seq * C.HIDDEN * bf16 * 2 + C.TOP_K * C.HIDDEN * bf16
    comptime ffn_peak = ffn_dense if ffn_dense > ffn_moe else ffn_moe
    comptime attn_peak = full_attn if full_attn > sliding_attn else sliding_attn
    return ffn_peak if ffn_peak > attn_peak else attn_peak


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

    # Probe strides
    var probe = List[WeightDesc]()
    var sl_r = emit_sliding[tp]("", 0, probe)
    var fl_r = emit_full[tp]("", 0, probe)
    var sl_proto = sl_r[0]
    var sl_stride = sl_r[1]
    var fl_proto = fl_r[0]
    var fl_stride = fl_r[1]

    # Weight sections
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

    # State via SectionBuilder
    var state = SectionBuilder()

    var skv_sb = SectionBuilder()
    var skv_proto = SlidingKVSlots(
        k=skv_sb.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]](),
        v=skv_sb.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]]())
    var sliding_kv = Repeated[SlidingKVSlots](
        skv_proto, state.cursor, skv_sb.bytes(), C.NUM_SLIDING_LAYERS)
    _ = state.reserve_bytes(C.NUM_SLIDING_LAYERS * skv_sb.bytes())

    var fkv_sb = SectionBuilder()
    var fkv_proto = FullKVSlots(
        k=fkv_sb.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_FULL]](),
        v=fkv_sb.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_FULL]]())
    var full_kv = Repeated[FullKVSlots](
        fkv_proto, state.cursor, fkv_sb.bytes(), C.NUM_FULL_LAYERS)
    _ = state.reserve_bytes(C.NUM_FULL_LAYERS * fkv_sb.bytes())

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

    # Host section
    var host_off = align_up(distributed + state.bytes())
    comptime HOST = LayerShard.HOST
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](
            hb.bf(descs, "model.language_model.norm.weight", C.HIDDEN, 1, HOST)),
        embed=SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]](
            hb.bf(descs, "model.language_model.embed_tokens.weight", C.VOCAB_SIZE, C.HIDDEN, HOST)))

    var topo = Gemma4Topology[tp](
        arena_base=0,
        sliding=Repeated[SlidingLayerRefs[tp]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[tp]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        distributed_bytes=distributed,
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        state_bytes=state.bytes(),
        host=host, host_bytes=hb.cursor)
    return Gemma4LoadPlan[tp](topo, descs^)


# =============================================================================
# Model — arena, pool, scratch, topology.
# =============================================================================


struct Gemma4[tp: Int](Movable):
    var arena: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool: BurstPool[]
    var scratch: ScratchPool
    var topology: Gemma4Topology[Self.tp]

    def __init__(out self,
        var arena: NumaArena[alignment=DEFAULT_ALIGNMENT],
        var pool: BurstPool[],
        topology: Gemma4Topology[Self.tp],
    ):
        self.arena = arena^
        self.pool = pool^
        self.topology = topology.bind(Int(self.arena.base))
        self.scratch = ScratchPool(topology.scratch_capacity)

    def init_state(mut self):
        var topo = self.topology
        var sb = topo.state_base()

        # RoPE tables
        init_sliding_rope_tables(
            topo.sliding_rope.cos.bound(sb),
            topo.sliding_rope.sin.bound(sb))
        init_full_rope_tables(
            topo.full_rope.cos.bound(sb),
            topo.full_rope.sin.bound(sb))

        # Router scale baking: fold 1/sqrt(hidden) into router_scale weights
        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        comptime width = simd_width_of[DType.float32]()
        var si = 0
        var fi = 0
        for i in range(C.NUM_LAYERS):
            var scale_addr: Int
            if is_full_layer(i):
                var lb = topo.full_base(fi)
                scale_addr = topo.full.proto.body.router_scale.addr(lb)
                fi += 1
            else:
                var lb = topo.sliding_base(si)
                scale_addr = topo.sliding.proto.body.router_scale.addr(lb)
                si += 1
            var p = BF16Ptr(unsafe_from_address=scale_addr)
            for j in range(0, C.HIDDEN, width):
                var v = (p + j).load[width=width]().cast[DType.float32]()
                (p + j).store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        print("state initialized: rope tables + baked router constants")

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
        var nodes = numa.plan_topology(1)

        var size = topo.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB (" +
              String(topo.distributed_bytes // (1024 * 1024)) + " MB weights + " +
              String(topo.state_bytes // (1024 * 1024)) + " MB state)")
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](nodes[0], size)
        if not arena:
            print("arena allocation failed")
            return None

        var arena_bases = List[Int]()
        arena_bases.append(Int(arena.base))

        var load_result = load_weights_from_descs(plan.descs, shards, arena_bases)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        _ = arena.prefault(topo.distributed_bytes, topo.state_bytes)

        var pool = BurstPool[].for_numa_node(numa, nodes[0])
        var model = Self(arena^, pool^, topo)
        model.init_state()
        return model^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topology.scratch_base())

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime S = Gemma4Shapes[Self.tp]
        var topo = self.topology
        var sb = topo.state_base()
        var scb = topo.scratch_base()

        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        var x_main = topo.x_main(seq_len)
        var x_residual = topo.x_residual(seq_len)
        var embed = static_tensor_view(topo.arena_base, topo.host.embed)
        var final_norm = static_tensor_view(topo.arena_base, topo.host.final_norm)
        var sliding_cos = static_tensor_view(sb, topo.sliding_rope.cos)
        var sliding_sin = static_tensor_view(sb, topo.sliding_rope.sin)
        var full_cos = static_tensor_view(sb, topo.full_rope.cos)
        var full_sin = static_tensor_view(sb, topo.full_rope.sin)

        embed_lookup_scaled(embed, tokens_ptr, x_main, EMBED_SCALE, self.pool).join()

        var sliding_idx = 0
        var full_idx = 0

        for layer_idx in range(C.NUM_LAYERS):
            var full = is_full_layer(layer_idx)
            var lb = topo.full_base(full_idx) if full else topo.sliding_base(sliding_idx)
            var body = topo.full.proto.body if full else topo.sliding.proto.body

            if full:
                self.attention_full(
                    topo, x_main, x_residual,
                    full_cos, full_sin,
                    full_idx, seq_len, pos)
            else:
                self.attention_sliding(
                    topo, x_main, x_residual,
                    sliding_cos, sliding_sin,
                    sliding_idx, seq_len, pos)

            rmsnorm(x_residual, static_tensor_view(lb, body.post_attn_norm),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()
            elem_add(x_main, x_residual, x_main)

            rmsnorm(x_main, static_tensor_view(lb, body.pre_ffn_norm),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var gate = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.GateUp.N](
                self.scratch, scb, seq_len)
            var up = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.GateUp.N](
                self.scratch, scb, seq_len)

            gemm(x_residual, static_tensor_view(lb, body.gate_proj),
                gate.view(), self.pool).join()
            gemm(x_residual, static_tensor_view(lb, body.up_proj),
                up.view(), self.pool).join()

            gelu_tanh_mul(gate.view(), up.view(), gate.view())
            up^.release()

            gemm(gate.view(), static_tensor_view(lb, body.down_proj),
                x_residual, self.pool).join()
            gate^.release()

            var dense = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, C.HIDDEN](
                self.scratch, scb, seq_len)
            rmsnorm(x_residual, static_tensor_view(lb, body.post_ffn_norm_1),
                dense.view(), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var router = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, C.HIDDEN](
                self.scratch, scb, seq_len)
            rmsnorm(x_main, static_tensor_view(lb, body.router_scale),
                router.view(), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var router_logits_buf = InlineArray[Scalar[DType.bfloat16], C.NUM_EXPERTS](
                fill=Scalar[DType.bfloat16](0))
            var router_logits_ptr = BF16Ptr(
                unsafe_from_address=Int(UnsafePointer(to=router_logits_buf[0])))
            var router_input_ptr = router.view().as_ptr[DType.bfloat16]()

            gemv_kernel[C.HIDDEN, C.NUM_EXPERTS](GemmArgs(
                router_input_ptr,
                static_tensor_view(lb, body.router_proj).as_ptr[DType.bfloat16](),
                router_logits_ptr,
                0, C.NUM_EXPERTS, 1))

            var routing = softmax_topk_renorm[C.NUM_EXPERTS, C.TOP_K](
                router_logits_ptr,
                static_tensor_view(lb, body.router_pes).as_ptr[DType.bfloat16]())

            router^.release()

            rmsnorm(x_main, static_tensor_view(lb, body.pre_ffn_norm_2),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var moe = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, C.HIDDEN](
                self.scratch, scb, seq_len)
            var expert_buf_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()

            gemma4_moe_dispatch[C.NUM_EXPERTS, C.TOP_K, C.MOE_INTERMEDIATE, C.HIDDEN](
                x_residual.as_ptr[DType.bfloat16](),
                routing,
                static_tensor_view(lb, body.experts_gate_up).as_ptr[DType.bfloat16](),
                static_tensor_view(lb, body.experts_down).as_ptr[DType.bfloat16](),
                scratch_ptr[Scalar[DType.bfloat16]](scb, expert_buf_lease),
                moe.view().as_ptr[DType.bfloat16](),
                self.pool)

            expert_buf_lease^.release()

            rmsnorm(moe.view(), static_tensor_view(lb, body.post_ffn_norm_2),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()
            moe^.release()

            elem_add(dense.view(), x_residual, x_residual)
            dense^.release()

            rmsnorm(x_residual, static_tensor_view(lb, body.post_ffn_norm),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var layer_scalar = Float32(static_tensor_view(lb, body.layer_scalar).as_ptr[DType.bfloat16]()[])
            elem_add(x_main, x_residual, x_main)
            elem_scale(x_main, layer_scalar)

            if full:
                full_idx += 1
            else:
                sliding_idx += 1

        rmsnorm(x_main, final_norm, x_main, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var last_row_off = (seq_len - 1) * C.HIDDEN * BF16.ELEMENT_BYTES
        var last_hidden = DynView[Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]](x_main.ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = scratch_tensor_view[BF16, 1, C.VOCAB_SIZE](scb, logit_lease, 1)
        gemm(last_hidden, embed, logit_view, self.pool).join()
        logit_softcap(logit_view)

        return LogitsView[C.VOCAB_SIZE](
            scratch_ptr[Scalar[DType.bfloat16]](scb, logit_lease), logit_lease^)

    def attention_sliding(mut self,
        topo: Gemma4Topology[Self.tp],
        x_main: DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]],
        x_residual: DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]],
        sliding_cos: StaticTensorView[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM_SLIDING // 2]],
        sliding_sin: StaticTensorView[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM_SLIDING // 2]],
        sliding_idx: Int, seq_len: Int, pos: Int,
    ):
        comptime S = Gemma4Shapes[Self.tp]
        var lb = topo.sliding_base(sliding_idx)
        var sl = topo.sliding.proto
        var scb = topo.scratch_base()

        rmsnorm(x_main, static_tensor_view(lb, sl.body.input_norm),
            x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var q = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](
            self.scratch, scb, seq_len)
        var k = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](
            self.scratch, scb, seq_len)
        var v = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](
            self.scratch, scb, seq_len)

        gemm(x_residual, static_tensor_view(lb, sl.attn.q_proj),
            q.view(), self.pool).join()
        gemm(x_residual, static_tensor_view(lb, sl.attn.k_proj),
            k.view(), self.pool).join()
        gemm(x_residual, static_tensor_view(lb, sl.attn.v_proj),
            v.view(), self.pool).join()

        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            q.view(),
            static_tensor_view(lb, sl.attn.q_norm),
            q.view(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            k.view(),
            static_tensor_view(lb, sl.attn.k_norm),
            k.view(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(v.view(), v.view(), self.pool, Float32(C.RMS_NORM_EPS)).join()

        rope[C.HEAD_DIM_SLIDING, C.NUM_HEADS](q.view(), sliding_cos, sliding_sin, pos)
        rope[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](k.view(), sliding_cos, sliding_sin, pos)

        kv_cache_write(k.view(), topo.sliding_k_cache(sliding_idx), pos)
        kv_cache_write(v.view(), topo.sliding_v_cache(sliding_idx), pos)

        v^.release()
        k^.release()

        var attn_out = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](
            self.scratch, scb, seq_len)

        local_attention[C.NUM_HEADS, C.NUM_KV_HEADS_SLIDING, C.HEAD_DIM_SLIDING, C.SLIDING_WINDOW](
            q.view(),
            topo.sliding_k_cache(sliding_idx),
            topo.sliding_v_cache(sliding_idx),
            attn_out.view(), pos, self.pool).join()

        gemm(attn_out.view(), static_tensor_view(lb, sl.attn.o_proj),
            x_residual, self.pool).join()

        attn_out^.release()
        q^.release()

    def attention_full(mut self,
        topo: Gemma4Topology[Self.tp],
        x_main: DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]],
        x_residual: DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]],
        full_cos: StaticTensorView[F32, Shape[C.MAX_SEQ_LEN, 64]],
        full_sin: StaticTensorView[F32, Shape[C.MAX_SEQ_LEN, 64]],
        full_idx: Int, seq_len: Int, pos: Int,
    ):
        comptime S = Gemma4Shapes[Self.tp]
        var lb = topo.full_base(full_idx)
        var fl = topo.full.proto
        var scb = topo.scratch_base()

        rmsnorm(x_main, static_tensor_view(lb, fl.body.input_norm),
            x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var q = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.FullQ.N](
            self.scratch, scb, seq_len)
        var k = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.FullK.N](
            self.scratch, scb, seq_len)
        var v = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.FullK.N](
            self.scratch, scb, seq_len)

        gemm(x_residual, static_tensor_view(lb, fl.attn.q_proj),
            q.view(), self.pool).join()
        gemm(x_residual, static_tensor_view(lb, fl.attn.k_proj),
            k.view(), self.pool).join()

        var kp = k.view().as_ptr[DType.bfloat16]()
        var vp = v.view().as_ptr[DType.bfloat16]()
        comptime copy_width = simd_width_of[DType.bfloat16]()
        for j in range(0, S.FullK.N * seq_len, copy_width):
            (vp + j).store((kp + j).load[width=copy_width]())

        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_HEADS](
            q.view(),
            static_tensor_view(lb, fl.attn.q_norm),
            q.view(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_KV_HEADS_FULL](
            k.view(),
            static_tensor_view(lb, fl.attn.k_norm),
            k.view(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(v.view(), v.view(), self.pool, Float32(C.RMS_NORM_EPS)).join()

        apply_full_rope[C.NUM_HEADS](q.view(), full_cos, full_sin, pos)
        apply_full_rope[C.NUM_KV_HEADS_FULL](k.view(), full_cos, full_sin, pos)

        kv_cache_write(k.view(), topo.full_k_cache(full_idx), pos)
        kv_cache_write(v.view(), topo.full_v_cache(full_idx), pos)

        v^.release()
        k^.release()

        var attn_out = borrow_scratch_tensor[BF16, C.MAX_SEQ_LEN, S.FullQ.N](
            self.scratch, scb, seq_len)

        global_attention[C.NUM_HEADS, C.NUM_KV_HEADS_FULL, C.HEAD_DIM_FULL](
            q.view(),
            topo.full_k_cache(full_idx),
            topo.full_v_cache(full_idx),
            attn_out.view(), pos, self.pool).join()

        gemm(attn_out.view(), static_tensor_view(lb, fl.attn.o_proj),
            x_residual, self.pool).join()

        attn_out^.release()
        q^.release()


def main():
    var model_opt = Gemma4[1].load(Path("checkpoints/gemma-4-26B-A4B"))
    if not model_opt:
        print("failed to load model")
        return
    print("model loaded successfully")
