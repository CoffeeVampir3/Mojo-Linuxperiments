"""MiniMax-M2.7 ButterQuant — int8 MoE, NUMA-aware tensor parallel.

62 uniform decoder layers: full causal GQA (48 Q heads, 8 KV heads, head_dim 128),
partial RoPE (64 of 128 dims), full-vector Q/K RMSNorm, sigmoid-routed MoE
(256 experts, top-8, SwiGLU). Untied embeddings.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, Mat, DynView, Bound,
    WeightDesc, DEFAULT_ALIGNMENT, LogitsView,
    Task, SourceFormat,
    Passthrough, ButterquantI8PerRow, ButterquantI8PerRowAbsorbed,
    ButterquantI8PerBlock, ButterquantI8PerBlockAbsorbed,
    HOST_RANK,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    StaticTensorView, DynamicTensorView,
    static_tensor_view, dynamic_tensor_view,
    scratch_tensor_view, scratch_ptr,
    LayerShard, LayerBuilder,
)
from modeling.linear_borrow_pool import ScratchPool, ScratchLease
from experimental3.profiler import ForwardLogger


# =============================================================================
# Config
# =============================================================================


struct MiniMaxM27Config:
    comptime HIDDEN = 3072
    comptime NUM_LAYERS = 62

    comptime NUM_HEADS = 48
    comptime NUM_KV_HEADS = 8
    comptime HEAD_DIM = 128
    comptime Q_DIM = 6144
    comptime KV_DIM = 1024
    comptime HPG = 6
    comptime ROPE_DIM = 64
    comptime ROPE_THETA = 5_000_000
    comptime MAX_POS = 196608

    comptime MOE_INTERMEDIATE = 1536
    comptime NUM_EXPERTS = 256
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 200064
    comptime RMS_NORM_EPS = 1e-6

    comptime MAX_SEQ_LEN = 4096


comptime C = MiniMaxM27Config
comptime FWHT_BLK = 128
comptime FWHT_BLK_HIDDEN = 128
comptime FWHT_BLK_MOE_DOWN = 128
comptime LM_HEAD_FWHT_BLK = 64
comptime VNNI_ALIGN = 64
comptime MOE_DOWN_NUM_BLK = C.MOE_INTERMEDIATE // FWHT_BLK_MOE_DOWN


# =============================================================================
# Per-TP shape aliases
# =============================================================================


struct MiniMaxShapes[tp: Int]:
    comptime Q_LOCAL = C.Q_DIM // Self.tp
    comptime KV_LOCAL = C.KV_DIM // Self.tp
    comptime NUM_HEADS_LOCAL = C.NUM_HEADS // Self.tp
    comptime NUM_KV_HEADS_LOCAL = C.NUM_KV_HEADS // Self.tp
    comptime EXPERTS_LOCAL = C.NUM_EXPERTS // Self.tp


# =============================================================================
# Typed refs
# =============================================================================


@fieldwise_init
struct AttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:      SlotOffset[I8,  Shape[C.Q_DIM,  C.HIDDEN]]
    var k_proj:      SlotOffset[I8,  Shape[C.KV_DIM, C.HIDDEN]]
    var v_proj:      SlotOffset[I8,  Shape[C.KV_DIM, C.HIDDEN]]
    var o_proj:      SlotOffset[I8,  Shape[C.HIDDEN, C.Q_DIM]]
    var q_proj_sc:   SlotOffset[F32, Shape[C.Q_DIM,  1]]
    var k_proj_sc:   SlotOffset[F32, Shape[C.KV_DIM, 1]]
    var v_proj_sc:   SlotOffset[F32, Shape[C.KV_DIM, 1]]
    var o_proj_sc:   SlotOffset[F32, Shape[C.HIDDEN, 1]]

    # Full-vector RMSNorm over concatenated Q (6144) / K (1024).
    # Under TP the RMS denominator needs a cross-rank sum-of-squares reduce.
    var q_norm:      SlotOffset[BF16, Shape[C.Q_DIM,  1]]
    var k_norm:      SlotOffset[BF16, Shape[C.KV_DIM, 1]]

    var q_colsum:    Int
    var k_colsum:    Int
    var v_colsum:    Int
    var o_colsum:    Int


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = MiniMaxShapes[Self.tp]

    var input_norm:     SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]

    # F32 router — sigmoid + correction-bias + top-k is magnitude-sensitive.
    var router_proj: SlotOffset[F32, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    var router_bias: SlotOffset[F32, Shape[C.NUM_EXPERTS, 1]]

    # SwiGLU experts: w1=gate, w3=up, w2=down.
    # Arena stores experts_local contiguous experts per rank (ROW-sharded
    # over the NE*rows axis). Per-expert checkpoint tensors are concatenated
    # by the quantizer pipeline into these contiguous blocks.
    var experts_w1:       SlotOffset[I8,  Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN]]
    var experts_w1_sc:    SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w3:       SlotOffset[I8,  Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN]]
    var experts_w3_sc:    SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w2:       SlotOffset[I8,  Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE]]
    var experts_w2_sc:    SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.HIDDEN, 1]]

    var experts_w1_colsum: Int
    var experts_w3_colsum: Int
    var experts_w2_colsum: Int


@fieldwise_init
struct LayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: AttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


# =============================================================================
# State
# =============================================================================


@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main:     SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]
    var sin: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]


@fieldwise_init
struct HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]

    # Untied: embed_tokens is plain BF16 (lookup table only, no GEMV).
    # lm_head is butterquant int8 per-block, absorbing final_norm gamma.
    var embed:      SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_head:    SlotOffset[I8,   Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_head_sc: SlotOffset[F32,  Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]

    var lm_head_colsum_off: Int
    var sqrt_gamma_off:     Int
    var inv_sqrt_gamma_off: Int


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct MiniMaxM27Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena_base: Int
    var layers: Repeated[LayerRefs[Self.tp]]
    var distributed_bytes: Int

    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var rope: RopeSlots[C.ROPE_DIM // 2]

    # Placeholder — real layout needs the typed cache struct (VNNI tiles,
    # per-head scale arrays) once the forward is implemented.
    var kv_cache_off: Int
    var kv_cache_stride: Int
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
    def layer_base(self, idx: Int) -> Int:
        return self.arena_base + self.layers.off + idx * self.layers.stride

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
    def kv_cache_base(self, idx: Int) -> Int:
        return self.state_base() + self.kv_cache_off + idx * self.kv_cache_stride

    @always_inline
    def rope_cos_row(self, pos: Int) -> Int:
        return self.state_base() + self.rope.cos.offset + pos * (C.ROPE_DIM // 2) * size_of[Float32]()

    @always_inline
    def rope_sin_row(self, pos: Int) -> Int:
        return self.state_base() + self.rope.sin.offset + pos * (C.ROPE_DIM // 2) * size_of[Float32]()


# =============================================================================
# Emit
# =============================================================================


def emit_layer[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[LayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H  = C.HIDDEN
    comptime MI = C.MOE_INTERMEDIATE
    comptime NE = C.NUM_EXPERTS
    comptime S  = MiniMaxShapes[tp]
    comptime experts_local = S.EXPERTS_LOCAL
    comptime q_n_loc  = C.Q_DIM // tp
    comptime kv_n_loc = C.KV_DIM // tp
    comptime o_num_blk = C.NUM_HEADS // tp

    # --- Attention projections ---
    var q_proj = SlotOffset[I8, Shape[C.Q_DIM, H]](
        b.q(e, "self_attn.q_proj.weight", C.Q_DIM, H, ROW))
    var q_proj_sc = SlotOffset[F32, Shape[C.Q_DIM, 1]](
        b.f(e, "self_attn.q_proj.weight_scale", C.Q_DIM, 1, ROW))
    var k_proj = SlotOffset[I8, Shape[C.KV_DIM, H]](
        b.q(e, "self_attn.k_proj.weight", C.KV_DIM, H, ROW))
    var k_proj_sc = SlotOffset[F32, Shape[C.KV_DIM, 1]](
        b.f(e, "self_attn.k_proj.weight_scale", C.KV_DIM, 1, ROW))
    var v_proj = SlotOffset[I8, Shape[C.KV_DIM, H]](
        b.q(e, "self_attn.v_proj.weight", C.KV_DIM, H, ROW))
    var v_proj_sc = SlotOffset[F32, Shape[C.KV_DIM, 1]](
        b.f(e, "self_attn.v_proj.weight_scale", C.KV_DIM, 1, ROW))
    var o_proj = SlotOffset[I8, Shape[H, C.Q_DIM]](
        b.q(e, "self_attn.o_proj.weight", H, C.Q_DIM, COL))
    var o_proj_sc = SlotOffset[F32, Shape[H, 1]](
        b.f(e, "self_attn.o_proj.weight_scale", H, 1, REPL))

    var q_norm = SlotOffset[BF16, Shape[C.Q_DIM, 1]](
        b.bf(e, "self_attn.q_norm.weight", C.Q_DIM, 1, REPL))
    var k_norm = SlotOffset[BF16, Shape[C.KV_DIM, 1]](
        b.bf(e, "self_attn.k_norm.weight", C.KV_DIM, 1, REPL))

    var q_colsum = b.colsum(q_n_loc * 4)
    var k_colsum = b.colsum(kv_n_loc * 4)
    var v_colsum = b.colsum(kv_n_loc * 4)
    var o_colsum = b.colsum(H * o_num_blk * 4)

    var attn = AttnRefs[tp](
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
        q_proj_sc=q_proj_sc, k_proj_sc=k_proj_sc, v_proj_sc=v_proj_sc,
        o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm,
        q_colsum=q_colsum, k_colsum=k_colsum,
        v_colsum=v_colsum, o_colsum=o_colsum,
    )

    # --- Norms ---
    var input_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "input_layernorm.weight", H, 1, REPL))
    var post_attn_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "post_attention_layernorm.weight", H, 1, REPL))

    # --- Router (F32, not butterquantized) ---
    var router_proj = SlotOffset[F32, Shape[NE, H]](
        b.f(e, "block_sparse_moe.gate.weight", NE, H, REPL))
    var router_bias = SlotOffset[F32, Shape[NE, 1]](
        b.f(e, "block_sparse_moe.e_score_correction_bias", NE, 1, REPL))

    # --- Experts (w1=gate, w3=up, w2=down) ---
    # ROW-sharding over the NE*rows axis distributes whole experts across ranks.
    var experts_w1 = SlotOffset[I8, Shape[NE * MI, H]](
        b.q(e, "block_sparse_moe.w1", NE * MI, H, ROW))
    var experts_w1_sc = SlotOffset[F32, Shape[experts_local * MI, 1]](
        b.f(e, "block_sparse_moe.w1_scale", NE * MI, 1, ROW))
    var experts_w3 = SlotOffset[I8, Shape[NE * MI, H]](
        b.q(e, "block_sparse_moe.w3", NE * MI, H, ROW))
    var experts_w3_sc = SlotOffset[F32, Shape[experts_local * MI, 1]](
        b.f(e, "block_sparse_moe.w3_scale", NE * MI, 1, ROW))
    var experts_w2 = SlotOffset[I8, Shape[NE * H, MI]](
        b.q(e, "block_sparse_moe.w2", NE * H, MI, ROW))
    var experts_w2_sc = SlotOffset[F32, Shape[experts_local * H, 1]](
        b.f(e, "block_sparse_moe.w2_scale", NE * H, 1, ROW))

    var experts_w1_colsum = b.colsum(experts_local * MI * 4)
    var experts_w3_colsum = b.colsum(experts_local * MI * 4)
    var experts_w2_colsum = b.colsum(experts_local * H * MOE_DOWN_NUM_BLK * 4)

    var body = BodyRefs[tp](
        input_norm=input_norm, post_attn_norm=post_attn_norm,
        router_proj=router_proj, router_bias=router_bias,
        experts_w1=experts_w1, experts_w1_sc=experts_w1_sc,
        experts_w3=experts_w3, experts_w3_sc=experts_w3_sc,
        experts_w2=experts_w2, experts_w2_sc=experts_w2_sc,
        experts_w1_colsum=experts_w1_colsum,
        experts_w3_colsum=experts_w3_colsum,
        experts_w2_colsum=experts_w2_colsum,
    )

    return (LayerRefs[tp](attn=attn, body=body), b.cursor)


# =============================================================================
# Scratch budget
# =============================================================================


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime S = MiniMaxShapes[tp]
    comptime bf16 = 2
    comptime f32  = 4
    comptime i8   = 1

    comptime attn_peak = (
        C.HIDDEN * i8 + C.HIDDEN * f32 + f32
        + S.Q_LOCAL * bf16 + 2 * S.KV_LOCAL * bf16
        + S.Q_LOCAL * i8
        + S.NUM_HEADS_LOCAL * f32 * 2
        + S.NUM_HEADS_LOCAL * C.HEAD_DIM * f32
    )

    comptime moe_peak = (
        C.HIDDEN * i8 + C.HIDDEN * f32 + f32
        + C.NUM_EXPERTS * f32 + C.NUM_EXPERTS * f32
        + C.TOP_K * 2 * f32
        + C.TOP_K * C.MOE_INTERMEDIATE * i8
        + C.TOP_K * MOE_DOWN_NUM_BLK * f32
        + C.TOP_K * C.HIDDEN * bf16
    )

    comptime lm_head_peak = (
        C.HIDDEN * i8
        + (C.HIDDEN // LM_HEAD_FWHT_BLK) * f32
        + C.HIDDEN * f32
        + C.VOCAB_SIZE * bf16
    )

    comptime layer_peak = attn_peak if attn_peak > moe_peak else moe_peak
    return lm_head_peak if lm_head_peak > layer_peak else layer_peak


# =============================================================================
# Build plan
# =============================================================================


@fieldwise_init
struct MiniMaxM27LoadPlan[tp: Int](Movable):
    var topology: MiniMaxM27Topology[Self.tp]
    var descs: List[WeightDesc]


def build_minimax_plan[tp: Int]() -> MiniMaxM27LoadPlan[tp]:
    var descs = List[WeightDesc]()

    var probe = List[WeightDesc]()
    var lr = emit_layer[tp]("", 0, probe)
    var layer_proto = lr[0]
    var layer_stride = lr[1]

    var layers_off = 0
    var distributed = layers_off + C.NUM_LAYERS * layer_stride

    for i in range(C.NUM_LAYERS):
        var prefix = "model.layers." + String(i) + "."
        _ = emit_layer[tp](prefix, layers_off + i * layer_stride, descs)

    comptime bf16 = size_of[Scalar[DType.bfloat16]]()
    comptime f32  = size_of[Float32]()

    var state = SectionBuilder()
    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](),
    )
    var scratch_cap = calculate_peak_scratch[tp]()
    var scratch_off = state.reserve_bytes(scratch_cap)

    var rope = RopeSlots[C.ROPE_DIM // 2](
        cos=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.ROPE_DIM // 2]](),
        sin=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.ROPE_DIM // 2]](),
    )

    # KV cache: placeholder byte estimate. The real layout depends on the
    # butterquant cache struct (VNNI-tiled K, row-major V, per-position
    # K/Q scale arrays) and will change when the forward is implemented.
    comptime KV_CACHE_PER_POS = (
        C.NUM_KV_HEADS * C.HEAD_DIM * 1
        + C.NUM_KV_HEADS * C.HEAD_DIM * 1
        + C.NUM_KV_HEADS * f32
        + C.NUM_HEADS * f32
    )
    comptime kv_cache_stride = align_up(C.MAX_SEQ_LEN * KV_CACHE_PER_POS)
    var kv_cache_off = state.reserve_bytes(C.NUM_LAYERS * kv_cache_stride)

    # Host section
    comptime HOST = LayerShard.HOST
    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    var host_off = align_up(distributed + state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off

    var final_norm_off = hb.bf(descs, "model.norm.weight", C.HIDDEN, 1, HOST)

    var embed_off = hb.bf(descs, "model.embed_tokens.weight",
                         C.VOCAB_SIZE, C.HIDDEN, HOST)

    var lm_head_off = hb.q(descs, "lm_head.weight",
                          C.VOCAB_SIZE, C.HIDDEN, HOST)
    var lm_head_sc_off = hb.f(descs, "lm_head.weight_scale",
                             C.VOCAB_SIZE, vocab_num_blocks, HOST)

    var lm_head_colsum_bytes = C.VOCAB_SIZE * vocab_num_blocks * f32
    var lm_head_colsum_off = hb.colsum(lm_head_colsum_bytes)
    var sqrt_gamma_off = hb.cursor
    hb.cursor += C.HIDDEN * bf16
    var inv_sqrt_gamma_off = hb.cursor
    hb.cursor += C.HIDDEN * f32
    var host_bytes = hb.cursor

    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](final_norm_off),
        embed=SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]](embed_off),
        lm_head=SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]](lm_head_off),
        lm_head_sc=SlotOffset[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]](lm_head_sc_off),
        lm_head_colsum_off=lm_head_colsum_off,
        sqrt_gamma_off=sqrt_gamma_off,
        inv_sqrt_gamma_off=inv_sqrt_gamma_off,
    )

    var topo = MiniMaxM27Topology[tp](
        arena_base=0,
        layers=Repeated[LayerRefs[tp]](layer_proto, layers_off, layer_stride, C.NUM_LAYERS),
        distributed_bytes=distributed,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        rope=rope,
        kv_cache_off=kv_cache_off, kv_cache_stride=kv_cache_stride,
        state_bytes=state.bytes(),
        host=host, host_bytes=host_bytes,
    )
    return MiniMaxM27LoadPlan[tp](topo, descs^)


# =============================================================================
# Model struct
# =============================================================================


struct MiniMaxM27ButterQuant[tp: Int](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var topos: InlineArray[MiniMaxM27Topology[Self.tp], Self.tp]
    var profile: ForwardLogger

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[BurstPool[]],
        var sc: ScratchPool,
        topos: InlineArray[MiniMaxM27Topology[Self.tp], Self.tp],
    ):
        self.arenas = arenas^
        self.main_pools = mp^
        self.scratch = sc^
        self.topos = topos
        self.profile = ForwardLogger()

    def pool_ptrs(self) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
        for r in range(Self.tp):
            ptrs[r] = UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.main_pools[r])))
        return ptrs^

    def x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.topos[r].x_main(seq_len).ptr
        return ptrs^

    def x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.topos[r].x_residual(seq_len).ptr
        return ptrs^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topos[0].scratch_base())

    @staticmethod
    def build_quantizer_tasks() -> List[Task]:
        var tasks = List[Task]()

        for i in range(C.NUM_LAYERS):
            var p = "model.layers." + String(i) + "."

            tasks.append(ButterquantI8PerRow(p + "self_attn.q_proj.weight",
                SourceFormat.FP8_E4M3, FWHT_BLK_HIDDEN))
            tasks.append(ButterquantI8PerRow(p + "self_attn.k_proj.weight",
                SourceFormat.FP8_E4M3, FWHT_BLK_HIDDEN))
            tasks.append(ButterquantI8PerRow(p + "self_attn.v_proj.weight",
                SourceFormat.FP8_E4M3, FWHT_BLK_HIDDEN))
            tasks.append(ButterquantI8PerRow(p + "self_attn.o_proj.weight",
                SourceFormat.FP8_E4M3, C.HEAD_DIM))

            for j in range(C.NUM_EXPERTS):
                var ep = p + "block_sparse_moe.experts." + String(j) + "."
                tasks.append(ButterquantI8PerRow(ep + "w1.weight",
                    SourceFormat.FP8_E4M3, FWHT_BLK_HIDDEN))
                tasks.append(ButterquantI8PerRow(ep + "w3.weight",
                    SourceFormat.FP8_E4M3, FWHT_BLK_HIDDEN))
                tasks.append(ButterquantI8PerRow(ep + "w2.weight",
                    SourceFormat.FP8_E4M3, FWHT_BLK_MOE_DOWN))

            tasks.append(Passthrough(p + "input_layernorm.weight", DType.bfloat16))
            tasks.append(Passthrough(p + "post_attention_layernorm.weight", DType.bfloat16))
            tasks.append(Passthrough(p + "self_attn.q_norm.weight", DType.bfloat16))
            tasks.append(Passthrough(p + "self_attn.k_norm.weight", DType.bfloat16))
            tasks.append(Passthrough(p + "block_sparse_moe.gate.weight", DType.float32))
            tasks.append(Passthrough(p + "block_sparse_moe.e_score_correction_bias",
                DType.float32))

        tasks.append(Passthrough("model.norm.weight", DType.bfloat16))
        tasks.append(Passthrough("model.embed_tokens.weight", DType.bfloat16))
        tasks.append(ButterquantI8PerBlockAbsorbed("lm_head.weight",
            SourceFormat.BF16, LM_HEAD_FWHT_BLK, "model.norm.weight"))
        return tasks^

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    def init_state(mut self):
        print("MiniMaxM27ButterQuant.init_state: not implemented")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        print("MiniMaxM27ButterQuant.load: not implemented")
        return None

    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> Int32:
        _ = tokens_ptr
        _ = pos
        print("MiniMaxM27ButterQuant.forward_decode: not implemented")
        return Int32(0)
