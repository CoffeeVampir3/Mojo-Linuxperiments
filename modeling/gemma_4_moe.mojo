from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math import sqrt

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from modeling.linear_borrow_pool import ScratchPool, scratch_block_bytes

from modeling.model_spec import (
    BF16, F32,
    Shape, DynamicView, WeightDesc,
    DynamicTensor, StaticTensor,
    DEFAULT_ALIGNMENT, LogitsView,
    HOST_RANK,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, is_full_layer,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    LayerBuilder,
)
from modeling.loader import discover_shards, load_weights_from_descs
from kernels.kernel_ops import (
    gemm, rmsnorm, elem_add, kv_cache_write,
    gemv_kernel, GemmArgs,
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
    def scratch_base(self) -> Int:
        return self.arena_base + self.scratch_off


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime S = Gemma4Shapes[tp]
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    return BodyRefs[tp](
        input_norm      = b.bfs[Shape[H, 1]](e, "input_layernorm.weight"),
        post_attn_norm  = b.bfs[Shape[H, 1]](e, "post_attention_layernorm.weight"),
        pre_ffn_norm    = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm.weight"),
        pre_ffn_norm_2  = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm_2.weight"),
        post_ffn_norm_1 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_1.weight"),
        post_ffn_norm_2 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_2.weight"),
        post_ffn_norm   = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm.weight"),
        gate_proj       = b.bfs[S.GateUp](e, "mlp.gate_proj.weight"),
        up_proj         = b.bfs[S.GateUp](e, "mlp.up_proj.weight"),
        down_proj       = b.bfs[S.Down](e, "mlp.down_proj.weight"),
        router_proj     = b.bfs[Shape[NE, H]](e, "router.proj.weight"),
        router_scale    = b.bfs[Shape[H, 1]](e, "router.scale"),
        router_pes      = b.bfs[Shape[NE, 1]](e, "router.per_expert_scale"),
        experts_gate_up = b.bfs[S.ExpertsGateUp](e, "experts.gate_up_proj"),
        experts_down    = b.bfs[S.ExpertsDown](e, "experts.down_proj"),
        layer_scalar    = b.bfs[Shape[1, 1]](e, "layer_scalar"),
    )


def emit_sliding[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[SlidingLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime S = Gemma4Shapes[tp]
    var attn = SlidingAttnRefs[tp](
        q_proj = b.bfs[S.SlidingQ](e, "self_attn.q_proj.weight"),
        k_proj = b.bfs[S.SlidingKV](e, "self_attn.k_proj.weight"),
        v_proj = b.bfs[S.SlidingKV](e, "self_attn.v_proj.weight"),
        o_proj = b.bfs[S.SlidingO](e, "self_attn.o_proj.weight"),
        q_norm = b.bfs[Shape[C.HEAD_DIM_SLIDING, 1]](e, "self_attn.q_norm.weight"),
        k_norm = b.bfs[Shape[C.HEAD_DIM_SLIDING, 1]](e, "self_attn.k_norm.weight"),
    )
    return (SlidingLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def emit_full[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime S = Gemma4Shapes[tp]
    var attn = FullAttnRefs[tp](
        q_proj = b.bfs[S.FullQ](e, "self_attn.q_proj.weight"),
        k_proj = b.bfs[S.FullK](e, "self_attn.k_proj.weight"),
        o_proj = b.bfs[S.FullO](e, "self_attn.o_proj.weight"),
        q_norm = b.bfs[Shape[C.HEAD_DIM_FULL, 1]](e, "self_attn.q_norm.weight"),
        k_norm = b.bfs[Shape[C.HEAD_DIM_FULL, 1]](e, "self_attn.k_norm.weight"),
    )
    return (FullLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16 = BF16.ELEMENT_BYTES
    comptime seq = C.MAX_SEQ_LEN
    comptime S = Gemma4Shapes[tp]

    # Match the actual live set in forward:
    # q+k+v are live together, then k/v are released before attn_out is borrowed.
    comptime full_q = scratch_block_bytes[seq * S.FullQ.N * bf16]()
    comptime full_kv = scratch_block_bytes[seq * S.FullK.N * bf16]()
    comptime full_attn = (
        full_q + full_kv + full_kv
        if full_q + full_kv + full_kv > full_q + full_q
        else full_q + full_q
    )
    comptime sliding_q = scratch_block_bytes[seq * S.SlidingQ.N * bf16]()
    comptime sliding_kv = scratch_block_bytes[seq * S.SlidingKV.N * bf16]()
    comptime sliding_attn = (
        sliding_q + sliding_kv + sliding_kv
        if sliding_q + sliding_kv + sliding_kv > sliding_q + sliding_q
        else sliding_q + sliding_q
    )
    comptime ffn_dense = (
        scratch_block_bytes[seq * S.GateUp.N * bf16]()
        + scratch_block_bytes[seq * S.GateUp.N * bf16]()
    )
    comptime ffn_moe = (
        scratch_block_bytes[seq * C.HIDDEN * bf16]()
        + scratch_block_bytes[seq * C.HIDDEN * bf16]()
        + scratch_block_bytes[C.TOP_K * C.HIDDEN * bf16]()
    )
    comptime ffn_peak = ffn_dense if ffn_dense > ffn_moe else ffn_moe
    comptime attn_peak = full_attn if full_attn > sliding_attn else sliding_attn
    return ffn_peak if ffn_peak > attn_peak else attn_peak


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

    # State via SectionBuilder — pre-seeded with `distributed` so every slot
    # carries an arena-absolute offset.
    var state = SectionBuilder()
    state.cursor = distributed

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
    var host_off = align_up(state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var host = HostSlots(
        final_norm=hb.bfs[Shape[C.HIDDEN, 1]](descs, "model.language_model.norm.weight", target_rank=HOST_RANK),
        embed=hb.bfs[Shape[C.VOCAB_SIZE, C.HIDDEN]](descs, "model.language_model.embed_tokens.weight", target_rank=HOST_RANK))

    var topo = Gemma4Topology[tp](
        arena_base=0,
        sliding=Repeated[SlidingLayerRefs[tp]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[tp]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        distributed_bytes=distributed,
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        state_bytes=state.bytes() - distributed,
        host=host, host_bytes=hb.cursor)
    return Gemma4LoadPlan[tp](topo, descs^)


struct Gemma4[tp: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arena: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool: Self.Pool
    var scratch: ScratchPool
    var topology: Gemma4Topology[Self.tp]

    def __init__(out self,
        var arena: NumaArena[alignment=DEFAULT_ALIGNMENT],
        var pool: Self.Pool,
        topology: Gemma4Topology[Self.tp],
    ):
        self.arena = arena^
        self.pool = pool^
        self.topology = topology.bind(Int(self.arena.base))
        self.scratch = ScratchPool(topology.scratch_capacity)

    def init_state(mut self):
        var topo = self.topology

        init_sliding_rope_tables(
            topo.sliding_rope.cos.bound(topo.arena_base),
            topo.sliding_rope.sin.bound(topo.arena_base))
        init_full_rope_tables(
            topo.full_rope.cos.bound(topo.arena_base),
            topo.full_rope.sin.bound(topo.arena_base))

        # Router scale baking: fold 1/sqrt(hidden) into router_scale weights
        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        comptime width = simd_width_of[DType.float32]()
        var si = 0
        var fi = 0
        for i in range(C.NUM_LAYERS):
            var p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
            if is_full_layer(i):
                var lb = topo.full.base(topo.arena_base, fi)
                p = topo.full.proto.body.router_scale.bound(lb).as_ptr()
                fi += 1
            else:
                var lb = topo.sliding.base(topo.arena_base, si)
                p = topo.sliding.proto.body.router_scale.bound(lb).as_ptr()
                si += 1
            for j in range(0, C.HIDDEN, width):
                var v = (p + j).load[width=width]().cast[DType.float32]()
                (p + j).store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        print("state initialized: rope tables + baked router constants")

    @staticmethod
    def load(
        dir_path: Path,
        numa: NumaInfo,
        numa_topo: NumaTopology,
        var pool: Self.Pool,
    ) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_gemma4_plan[Self.tp]()
        var topo = plan.topology

        var size = topo.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB (" +
              String(topo.distributed_bytes // (1024 * 1024)) + " MB weights + " +
              String(topo.state_bytes // (1024 * 1024)) + " MB state)")
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa_topo[0], size)
        if not arena:
            print("arena allocation failed")
            return None

        var arena_bases = List[Int]()
        arena_bases.append(Int(arena.base))

        var load_result = load_weights_from_descs(plan.descs, shards, arena_bases, numa_topo)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        _ = arena.prefault(topo.distributed_bytes, topo.state_bytes)

        var model = Self(arena^, pool^, topo)
        model.init_state()
        return model^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topology.scratch_base())

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime S = Gemma4Shapes[Self.tp]
        comptime XShape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
        var topo = self.topology
        var scb = topo.scratch_base()

        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        var x_main = topo.activations.x_main.bound_dyn(topo.arena_base, seq_len)
        var x_residual = topo.activations.x_residual.bound_dyn(topo.arena_base, seq_len)
        var embed = topo.host.embed.bound(topo.arena_base)
        var final_norm = topo.host.final_norm.bound(topo.arena_base)
        var sliding_cos = topo.sliding_rope.cos.bound(topo.arena_base)
        var sliding_sin = topo.sliding_rope.sin.bound(topo.arena_base)
        var full_cos = topo.full_rope.cos.bound(topo.arena_base)
        var full_sin = topo.full_rope.sin.bound(topo.arena_base)

        embed_lookup_scaled(embed, tokens_ptr, x_main, EMBED_SCALE, self.pool).join()

        var sliding_idx = 0
        var full_idx = 0

        for layer_idx in range(C.NUM_LAYERS):
            var full = is_full_layer(layer_idx)
            var lb = topo.full.base(topo.arena_base, full_idx) if full else topo.sliding.base(topo.arena_base, sliding_idx)
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

            rmsnorm(x_residual, body.post_attn_norm.bound(lb),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()
            elem_add(x_main, x_residual, x_main)

            rmsnorm(x_main, body.pre_ffn_norm.bound(lb),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var gate_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.GateUp.N]()
            var up_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.GateUp.N]()

            gemm(x_residual, body.gate_proj.bound(lb),
                gate_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len),
                self.pool).join()
            gemm(x_residual, body.up_proj.bound(lb),
                up_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len),
                self.pool).join()

            gelu_tanh_mul(
                gate_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len).any(),
                up_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len).any(),
                gate_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len).any())
            up_lease^.release()

            gemm(gate_lease.view[BF16, C.MAX_SEQ_LEN, S.GateUp.N](scb, seq_len),
                body.down_proj.bound(lb),
                x_residual, self.pool).join()
            gate_lease^.release()

            var dense_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            rmsnorm(x_residual, body.post_ffn_norm_1.bound(lb),
                dense_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len),
                self.pool, Float32(C.RMS_NORM_EPS)).join()

            var router_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            rmsnorm(x_main, body.router_scale.bound(lb),
                router_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len),
                self.pool, Float32(C.RMS_NORM_EPS)).join()

            var router_logits_buf = InlineArray[Scalar[DType.bfloat16], C.NUM_EXPERTS](
                fill=Scalar[DType.bfloat16](0))
            var router_logits_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                UnsafePointer(to=router_logits_buf[0]))
            var router_input_ptr = router_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len).as_ptr[DType.bfloat16]()

            gemv_kernel[C.HIDDEN, C.NUM_EXPERTS](GemmArgs(
                router_input_ptr,
                body.router_proj.bound(lb).as_ptr[DType.bfloat16](),
                router_logits_ptr,
                0, C.NUM_EXPERTS, 1))

            var routing = softmax_topk_renorm[C.NUM_EXPERTS, C.TOP_K](
                router_logits_ptr,
                body.router_pes.bound(lb).as_ptr[DType.bfloat16]())

            router_lease^.release()

            rmsnorm(x_main, body.pre_ffn_norm_2.bound(lb),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var moe_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            var expert_buf_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()

            gemma4_moe_dispatch[C.NUM_EXPERTS, C.TOP_K, C.MOE_INTERMEDIATE, C.HIDDEN](
                x_residual.as_ptr[DType.bfloat16](),
                routing,
                body.experts_gate_up.bound(lb).as_ptr[DType.bfloat16](),
                body.experts_down.bound(lb).as_ptr[DType.bfloat16](),
                expert_buf_lease.as_ptr[Scalar[DType.bfloat16]](scb),
                moe_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len).as_ptr[DType.bfloat16](),
                self.pool)

            expert_buf_lease^.release()

            rmsnorm(moe_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len),
                body.post_ffn_norm_2.bound(lb),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()
            moe_lease^.release()

            elem_add(
                dense_lease.view[BF16, C.MAX_SEQ_LEN, C.HIDDEN](scb, seq_len),
                x_residual, x_residual)
            dense_lease^.release()

            rmsnorm(x_residual, body.post_ffn_norm.bound(lb),
                x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

            var layer_scalar = Float32(body.layer_scalar.bound(lb).as_ptr[DType.bfloat16]()[])
            elem_add(x_main, x_residual, x_main)
            elem_scale(x_main, layer_scalar)

            if full:
                full_idx += 1
            else:
                sliding_idx += 1

        rmsnorm(x_main, final_norm, x_main, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var last_row_off = (seq_len - 1) * C.HIDDEN
        var last_hidden = DynamicView[BF16, XShape](
            x_main.as_ptr[DType.bfloat16]() + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = logit_lease.view[BF16, 1, C.VOCAB_SIZE](scb, 1)
        gemm(last_hidden, embed, logit_view, self.pool).join()
        logit_softcap(logit_view)

        var logit_ptr = logit_lease.as_ptr[Scalar[DType.bfloat16]](scb)
        return LogitsView[C.VOCAB_SIZE](logit_ptr, logit_lease^)

    def attention_sliding[
        XMainT: DynamicTensor, XResT: DynamicTensor,
        CosT: StaticTensor, SinT: StaticTensor,
    ](mut self,
        topo: Gemma4Topology[Self.tp],
        x_main: XMainT,
        x_residual: XResT,
        sliding_cos: CosT,
        sliding_sin: SinT,
        sliding_idx: Int, seq_len: Int, pos: Int,
    ) where CosT.DTYPE == DType.float32:
        comptime S = Gemma4Shapes[Self.tp]
        var lb = topo.sliding.base(topo.arena_base, sliding_idx)
        var sl = topo.sliding.proto
        var scb = topo.scratch_base()
        var kv_base = topo.sliding_kv.base(topo.arena_base, sliding_idx)

        rmsnorm(x_main, sl.body.input_norm.bound(lb),
            x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var q_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingQ.N]()
        var k_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingKV.N]()
        var v_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingKV.N]()

        gemm(x_residual, sl.attn.q_proj.bound(lb),
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len),
            self.pool).join()
        gemm(x_residual, sl.attn.k_proj.bound(lb),
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len),
            self.pool).join()
        gemm(x_residual, sl.attn.v_proj.bound(lb),
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len),
            self.pool).join()

        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len).any(),
            sl.attn.q_norm.bound(lb),
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len).any(),
            sl.attn.k_norm.bound(lb),
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len).any(),
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()

        rope[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len),
            sliding_cos, sliding_sin, pos)
        rope[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len),
            sliding_cos, sliding_sin, pos)

        kv_cache_write(
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len),
            topo.sliding_kv.proto.k.bound_cache(kv_base), pos)
        kv_cache_write(
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingKV.N](scb, seq_len),
            topo.sliding_kv.proto.v.bound_cache(kv_base), pos)

        v_lease^.release()
        k_lease^.release()

        var attn_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingQ.N]()

        local_attention[C.NUM_HEADS, C.NUM_KV_HEADS_SLIDING, C.HEAD_DIM_SLIDING, C.SLIDING_WINDOW](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len),
            topo.sliding_kv.proto.k.bound_cache(kv_base),
            topo.sliding_kv.proto.v.bound_cache(kv_base),
            attn_out_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len),
            pos, self.pool).join()

        gemm(attn_out_lease.view[BF16, C.MAX_SEQ_LEN, S.SlidingQ.N](scb, seq_len),
            sl.attn.o_proj.bound(lb),
            x_residual, self.pool).join()

        attn_out_lease^.release()
        q_lease^.release()

    def attention_full[
        XMainT: DynamicTensor, XResT: DynamicTensor,
        CosT: StaticTensor, SinT: StaticTensor,
    ](mut self,
        topo: Gemma4Topology[Self.tp],
        x_main: XMainT,
        x_residual: XResT,
        full_cos: CosT,
        full_sin: SinT,
        full_idx: Int, seq_len: Int, pos: Int,
    ) where CosT.DTYPE == DType.float32:
        comptime S = Gemma4Shapes[Self.tp]
        var lb = topo.full.base(topo.arena_base, full_idx)
        var fl = topo.full.proto
        var scb = topo.scratch_base()
        var kv_base = topo.full_kv.base(topo.arena_base, full_idx)

        rmsnorm(x_main, fl.body.input_norm.bound(lb),
            x_residual, self.pool, Float32(C.RMS_NORM_EPS)).join()

        var q_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullQ.N]()
        var k_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullK.N]()
        var v_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullK.N]()

        gemm(x_residual, fl.attn.q_proj.bound(lb),
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len),
            self.pool).join()
        gemm(x_residual, fl.attn.k_proj.bound(lb),
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len),
            self.pool).join()

        var kp = k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).as_ptr[DType.bfloat16]()
        var vp = v_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).as_ptr[DType.bfloat16]()
        comptime copy_width = simd_width_of[DType.bfloat16]()
        for j in range(0, S.FullK.N * seq_len, copy_width):
            (vp + j).store((kp + j).load[width=copy_width]())

        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_HEADS](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len).any(),
            fl.attn.q_norm.bound(lb),
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_KV_HEADS_FULL](
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).any(),
            fl.attn.k_norm.bound(lb),
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).any(),
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len).any(),
            self.pool, Float32(C.RMS_NORM_EPS)).join()

        apply_full_rope[C.NUM_HEADS](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len),
            full_cos, full_sin, pos)
        apply_full_rope[C.NUM_KV_HEADS_FULL](
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len),
            full_cos, full_sin, pos)

        kv_cache_write(
            k_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len),
            topo.full_kv.proto.k.bound_cache(kv_base), pos)
        kv_cache_write(
            v_lease.view[BF16, C.MAX_SEQ_LEN, S.FullK.N](scb, seq_len),
            topo.full_kv.proto.v.bound_cache(kv_base), pos)

        v_lease^.release()
        k_lease^.release()

        var attn_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullQ.N]()

        global_attention[C.NUM_HEADS, C.NUM_KV_HEADS_FULL, C.HEAD_DIM_FULL](
            q_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len),
            topo.full_kv.proto.k.bound_cache(kv_base),
            topo.full_kv.proto.v.bound_cache(kv_base),
            attn_out_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len),
            pos, self.pool).join()

        gemm(attn_out_lease.view[BF16, C.MAX_SEQ_LEN, S.FullQ.N](scb, seq_len),
            fl.attn.o_proj.bound(lb),
            x_residual, self.pool).join()

        attn_out_lease^.release()
        q_lease^.release()


def main():
    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(1)
    var pool = BurstPool[].for_topology(numa, numa_topo[0])
    var model_opt = Gemma4[1].load(
        Path("checkpoints/gemma-4-26B-A4B"), numa, numa_topo, pool^)
    if not model_opt:
        print("failed to load model")
        return
    print("model loaded successfully")
