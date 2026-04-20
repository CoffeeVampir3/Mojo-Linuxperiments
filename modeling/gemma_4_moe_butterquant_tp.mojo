"""Gemma 4 26B-A4B ButterQuant — int8 MoE, NUMA-aware tensor parallel.

Topology-based forward: typed SlotOffset refs, bound topology per rank,
no RankView / Gemma4ModelLayout indirection. Topology carries arena_base
and resolves all addresses — same pattern for tp=1 and tp=N.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, Mat, DynView, Bound,
    WeightDesc, DEFAULT_ALIGNMENT, LogitsView,
    TaskVisitor, PerRow, PerBlockAbsorbed,
)
from quant.source_format import Bf16Converter
from modeling.gemma4_common import (
    Gemma4BaseConfig, is_full_layer,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    StaticTensorView, DynamicTensorView,
    static_tensor_view, dynamic_tensor_view,
    scratch_tensor_view, scratch_ptr,
    LayerShard, LayerBuilder,
)
from kernels.kernel_ops import PoolFence, BF16Ptr
from kernels.reductions import ring_allreduce, small_allreduce, ring_broadcast, ring_allgather
from modeling.linear_borrow_pool import ScratchPool, ScratchLease

from experimental3.kernels.dispatch_kernels import (
    rmsnorm_gamma_fwht_quantize,
    rmsnorm_dual_gamma_fwht_quantize,
    rmsnorm_gamma_fwht_per_block_quantize,
    post_attn_norm_dispatch,
    expert_sum_dispatch,
    dense_norm_dispatch,
    post_reduce_dispatch,
    int8_gemv,
    fused_gu_gelu_tanh, fused_gu_gelu_tanh_wa,
    int8_gemv_blocked, int8_gemv_blocked_wa,
    lm_head_gemv,
    gemma4_moe_phase1, gemma4_moe_phase2, router_topk_dispatch,
    sliding_attn_dispatch,
    cp_attn_prep_dispatch, cp_chunked_attn_dispatch,
    merge_local_chunks_dispatch, cp_gather_dispatch,
)
from experimental3.profiler import (
    PhaseTiming, phase_timing_from_points, finish_single_pool_fence,
    ForwardSample, ForwardLogger,
)
from experimental3.init_weights import (
    colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at,
)
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr
from experimental3.kernels.full_chunked_attention_fused import (
    MAX_CP_RANKS,
    cp_local_pos, cp_owning_rank, cp_local_context_len,
)
from experimental_gemma.router import Gemma4TopKResult
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables
from experimental_gemma.ops import embed_lookup_blocked, logit_softcap
from modeling.loader import discover_shards, load_weights_from_descs
from modeling.model_spec import HOST_RANK
from simd_math import sqrt
from experimental3.tensor_dump import Dumper



# =============================================================================
# Config
# =============================================================================


comptime Gemma4Config = Gemma4BaseConfig
comptime C = Gemma4Config
comptime FWHT_BLK = 64
comptime FWHT_BLK_HIDDEN = 256
comptime FWHT_BLK_DENSE_DOWN = 16
comptime LM_HEAD_FWHT_BLK = 64
comptime MOE_NUM_BLOCKS = C.MOE_INTERMEDIATE // FWHT_BLK
comptime EMBED_SCALE = 53.0
comptime VNNI_ALIGN = 64
comptime DUMP_ATTENTION = False


# =============================================================================
# Per-TP shape aliases
# =============================================================================


struct Gemma4Shapes[tp: Int]:
    comptime GateUp      = Shape[C.INTERMEDIATE, C.HIDDEN, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    comptime GateUpScale = Shape[C.INTERMEDIATE, 1, shard_n=True, tp=Self.tp, align_n=VNNI_ALIGN]
    comptime Down        = Shape[C.HIDDEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp, align_m=VNNI_ALIGN]

    comptime SlidingQ    = Shape[C.Q_DIM_SLIDING, 1, shard_n=True, tp=Self.tp]
    comptime FullQ       = Shape[C.Q_DIM_FULL, 1, shard_n=True, tp=Self.tp]

    comptime DENSE_INT_LOCAL = Self.GateUp.N
    comptime DENSE_DOWN_NUM_BLK = Self.DENSE_INT_LOCAL // FWHT_BLK_DENSE_DOWN
    comptime FULL_HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL


# =============================================================================
# Family products — typed refs for quantized weights
# =============================================================================


@fieldwise_init
struct SlidingAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:    SlotOffset[I8, Shape[C.Q_DIM_SLIDING, C.HIDDEN]]
    var k_proj:    SlotOffset[I8, Shape[C.KV_DIM_SLIDING, C.HIDDEN]]
    var v_proj:    SlotOffset[I8, Shape[C.KV_DIM_SLIDING, C.HIDDEN]]
    var q_proj_sc: SlotOffset[F32, Shape[C.Q_DIM_SLIDING, 1]]
    var k_proj_sc: SlotOffset[F32, Shape[C.KV_DIM_SLIDING, 1]]
    var v_proj_sc: SlotOffset[F32, Shape[C.KV_DIM_SLIDING, 1]]
    var o_proj:    SlotOffset[I8, Shape[C.HIDDEN, C.Q_DIM_SLIDING]]
    var o_proj_sc: SlotOffset[F32, Shape[C.HIDDEN, 1]]
    var q_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var k_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var q_colsum: Int
    var kv_colsum: Int
    var o_colsum: Int


@fieldwise_init
struct FullAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:    SlotOffset[I8, Shape[C.Q_DIM_FULL, C.HIDDEN]]
    var k_proj:    SlotOffset[I8, Shape[C.KV_DIM_FULL, C.HIDDEN]]
    var q_proj_sc: SlotOffset[F32, Shape[C.Q_DIM_FULL, 1]]
    var k_proj_sc: SlotOffset[F32, Shape[C.KV_DIM_FULL, 1]]
    var o_proj:    SlotOffset[I8, Shape[C.HIDDEN, C.Q_DIM_FULL]]
    var o_proj_sc: SlotOffset[F32, Shape[C.HIDDEN, 1]]
    var q_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var k_norm:    SlotOffset[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var q_colsum: Int
    var k_colsum: Int
    var o_colsum: Int


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
    var layer_scalar:    SlotOffset[BF16, Shape[1, 1]]
    var gate_proj:       SlotOffset[I8, Self.S.GateUp]
    var gate_proj_sc:    SlotOffset[F32, Self.S.GateUpScale]
    var up_proj:         SlotOffset[I8, Self.S.GateUp]
    var up_proj_sc:      SlotOffset[F32, Self.S.GateUpScale]
    var down_proj:       SlotOffset[I8, Self.S.Down]
    var down_proj_sc:    SlotOffset[F32, Shape[C.HIDDEN, 1]]
    var router_scale:    SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var router_proj:     SlotOffset[I8, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    var router_proj_sc:  SlotOffset[F32, Shape[C.NUM_EXPERTS, 1]]
    var router_pes:      SlotOffset[BF16, Shape[C.NUM_EXPERTS, 1]]
    var experts_gate_up:    SlotOffset[I8, Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN]]
    var experts_gate_up_sc: SlotOffset[F32, Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, 1]]
    var experts_down:       SlotOffset[I8, Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE]]
    var experts_down_sc:    SlotOffset[F32, Shape[C.NUM_EXPERTS * C.HIDDEN, 1]]
    var gu_colsum: Int
    var down_colsum: Int
    var router_colsum: Int
    var experts_gu_colsum: Int
    var experts_down_colsum: Int


@fieldwise_init
struct SlidingLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: SlidingAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


@fieldwise_init
struct FullLayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: FullAttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


# =============================================================================
# State families
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
    var embed:      SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var embed_sc:   SlotOffset[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]
    var embed_colsum_off: Int
    var sqrt_gamma_off: Int
    var inv_sqrt_gamma_off: Int


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct Gemma4Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena_base: Int
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
    def sliding_cache_base(self, idx: Int) -> Int:
        return self.state_base() + self.sliding_cache_off + idx * self.sliding_cache_stride

    @always_inline
    def full_cache_base(self, idx: Int) -> Int:
        return self.state_base() + self.full_cache_off + idx * self.full_cache_stride

    @always_inline
    def sliding_cos_row(self, pos: Int) -> Int:
        return self.state_base() + self.sliding_rope.cos.offset + pos * (C.HEAD_DIM_SLIDING // 2) * size_of[Float32]()

    @always_inline
    def sliding_sin_row(self, pos: Int) -> Int:
        return self.state_base() + self.sliding_rope.sin.offset + pos * (C.HEAD_DIM_SLIDING // 2) * size_of[Float32]()


# =============================================================================
# Emit
# =============================================================================


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    comptime GU = C.MOE_GATE_UP_FUSED
    comptime MI = C.MOE_INTERMEDIATE
    comptime S = Gemma4Shapes[tp]
    comptime o_num_blk = C.NUM_HEADS // tp
    comptime experts_local = NE // tp

    # Order is not arbitrary (contiguousness assumed by some ops)
    var input_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "input_layernorm.weight", H, 1, REPL))
    var post_attn_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "post_attention_layernorm.weight", H, 1, REPL))
    var pre_ffn_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "pre_feedforward_layernorm.weight", H, 1, REPL))
    var pre_ffn_norm_2 = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "pre_feedforward_layernorm_2.weight", H, 1, REPL))
    var post_ffn_norm_1 = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "post_feedforward_layernorm_1.weight", H, 1, REPL))
    var post_ffn_norm_2 = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "post_feedforward_layernorm_2.weight", H, 1, REPL))
    var post_ffn_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "post_feedforward_layernorm.weight", H, 1, REPL))
    var layer_scalar = SlotOffset[BF16, Shape[1, 1]](
        b.bf(e, "layer_scalar", 1, 1, REPL))

    var gate_proj = SlotOffset[I8, S.GateUp](
        b.qs[S.GateUp](e, "mlp.gate_proj.weight"))
    var up_proj = SlotOffset[I8, S.GateUp](
        b.qs[S.GateUp](e, "mlp.up_proj.weight"))
    var gate_proj_sc = SlotOffset[F32, S.GateUpScale](
        b.fs[S.GateUpScale](e, "mlp.gate_proj.weight_scale"))
    var up_proj_sc = SlotOffset[F32, S.GateUpScale](
        b.fs[S.GateUpScale](e, "mlp.up_proj.weight_scale"))

    var down_proj = SlotOffset[I8, S.Down](
        b.qs[S.Down](e, "mlp.down_proj.weight"))
    var down_proj_sc = SlotOffset[F32, Shape[H, 1]](
        b.f(e, "mlp.down_proj.weight_scale", H, 1, REPL))
    var router_scale = SlotOffset[BF16, Shape[H, 1]](
        b.bf(e, "router.scale", H, 1, REPL))
    var router_proj = SlotOffset[I8, Shape[NE, H]](
        b.q(e, "router.proj.weight", NE, H, REPL))
    var router_proj_sc = SlotOffset[F32, Shape[NE, 1]](
        b.f(e, "router.proj.weight_scale", NE, 1, REPL))
    var router_pes = SlotOffset[BF16, Shape[NE, 1]](
        b.bf(e, "router.per_expert_scale", NE, 1, REPL))
    var experts_gate_up = SlotOffset[I8, Shape[NE * GU, H]](
        b.q(e, "experts.gate_up_proj", NE * GU, H, ROW))
    var experts_gate_up_sc = SlotOffset[F32, Shape[NE * GU, 1]](
        b.f(e, "experts.gate_up_proj_scale", NE * GU, 1, ROW))
    var experts_down = SlotOffset[I8, Shape[NE * H, MI]](
        b.q(e, "experts.down_proj", NE * H, MI, ROW))
    var experts_down_sc = SlotOffset[F32, Shape[NE * H, 1]](
        b.f(e, "experts.down_proj_scale", NE * H, 1, ROW))
    var gu_colsum = b.colsum(S.DENSE_INT_LOCAL * 2 * 4)
    var down_colsum = b.colsum(H * S.DENSE_DOWN_NUM_BLK * 4)
    var router_colsum = b.colsum(NE * 4)
    var experts_gu_colsum = b.colsum(experts_local * GU * 4)
    var experts_down_colsum = b.colsum(experts_local * H * MOE_NUM_BLOCKS * 4)

    return BodyRefs[tp](
        input_norm=input_norm,
        post_attn_norm=post_attn_norm,
        pre_ffn_norm=pre_ffn_norm,
        pre_ffn_norm_2=pre_ffn_norm_2,
        post_ffn_norm_1=post_ffn_norm_1,
        post_ffn_norm_2=post_ffn_norm_2,
        post_ffn_norm=post_ffn_norm,
        layer_scalar=layer_scalar,
        gate_proj=gate_proj,
        gate_proj_sc=gate_proj_sc,
        up_proj=up_proj,
        up_proj_sc=up_proj_sc,
        down_proj=down_proj,
        down_proj_sc=down_proj_sc,
        router_scale=router_scale,
        router_proj=router_proj,
        router_proj_sc=router_proj_sc,
        router_pes=router_pes,
        experts_gate_up=experts_gate_up,
        experts_gate_up_sc=experts_gate_up_sc,
        experts_down=experts_down,
        experts_down_sc=experts_down_sc,
        gu_colsum=gu_colsum,
        down_colsum=down_colsum,
        router_colsum=router_colsum,
        experts_gu_colsum=experts_gu_colsum,
        experts_down_colsum=experts_down_colsum,
    )


def emit_sliding[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[SlidingLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H = C.HIDDEN
    comptime HDS = C.HEAD_DIM_SLIDING
    comptime o_num_blk = C.NUM_HEADS // tp
    comptime q_n_loc = C.Q_DIM_SLIDING // tp
    comptime kv_n = 2 * (C.KV_DIM_SLIDING // tp)
    var attn = SlidingAttnRefs[tp](
        q_proj    = SlotOffset[I8, Shape[C.Q_DIM_SLIDING, H]](b.q(e, "self_attn.q_proj.weight", C.Q_DIM_SLIDING, H, ROW)),
        k_proj    = SlotOffset[I8, Shape[C.KV_DIM_SLIDING, H]](b.q(e, "self_attn.k_proj.weight", C.KV_DIM_SLIDING, H, ROW)),
        v_proj    = SlotOffset[I8, Shape[C.KV_DIM_SLIDING, H]](b.q(e, "self_attn.v_proj.weight", C.KV_DIM_SLIDING, H, ROW)),
        q_proj_sc = SlotOffset[F32, Shape[C.Q_DIM_SLIDING, 1]](b.f(e, "self_attn.q_proj.weight_scale", C.Q_DIM_SLIDING, 1, ROW)),
        k_proj_sc = SlotOffset[F32, Shape[C.KV_DIM_SLIDING, 1]](b.f(e, "self_attn.k_proj.weight_scale", C.KV_DIM_SLIDING, 1, ROW)),
        v_proj_sc = SlotOffset[F32, Shape[C.KV_DIM_SLIDING, 1]](b.f(e, "self_attn.v_proj.weight_scale", C.KV_DIM_SLIDING, 1, ROW)),
        o_proj    = SlotOffset[I8, Shape[H, C.Q_DIM_SLIDING]](b.q(e, "self_attn.o_proj.weight", H, C.Q_DIM_SLIDING, COL)),
        o_proj_sc = SlotOffset[F32, Shape[H, 1]](b.f(e, "self_attn.o_proj.weight_scale", H, 1, REPL)),
        q_norm    = SlotOffset[BF16, Shape[HDS, 1]](b.bf(e, "self_attn.q_norm.weight", HDS, 1, REPL)),
        k_norm    = SlotOffset[BF16, Shape[HDS, 1]](b.bf(e, "self_attn.k_norm.weight", HDS, 1, REPL)),
        q_colsum  = b.colsum(q_n_loc * 4),
        kv_colsum = b.colsum(kv_n * 4),
        o_colsum  = b.colsum(H * o_num_blk * 4),
    )
    return (SlidingLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def emit_full[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H = C.HIDDEN
    comptime HDF = C.HEAD_DIM_FULL
    comptime o_num_blk = C.NUM_HEADS // tp
    comptime q_n_loc = C.Q_DIM_FULL // tp
    var attn = FullAttnRefs[tp](
        q_proj    = SlotOffset[I8, Shape[C.Q_DIM_FULL, H]](b.q(e, "self_attn.q_proj.weight", C.Q_DIM_FULL, H, ROW)),
        k_proj    = SlotOffset[I8, Shape[C.KV_DIM_FULL, H]](b.q(e, "self_attn.k_proj.weight", C.KV_DIM_FULL, H, REPL)),
        q_proj_sc = SlotOffset[F32, Shape[C.Q_DIM_FULL, 1]](b.f(e, "self_attn.q_proj.weight_scale", C.Q_DIM_FULL, 1, ROW)),
        k_proj_sc = SlotOffset[F32, Shape[C.KV_DIM_FULL, 1]](b.f(e, "self_attn.k_proj.weight_scale", C.KV_DIM_FULL, 1, REPL)),
        o_proj    = SlotOffset[I8, Shape[H, C.Q_DIM_FULL]](b.q(e, "self_attn.o_proj.weight", H, C.Q_DIM_FULL, COL)),
        o_proj_sc = SlotOffset[F32, Shape[H, 1]](b.f(e, "self_attn.o_proj.weight_scale", H, 1, REPL)),
        q_norm    = SlotOffset[BF16, Shape[HDF, 1]](b.bf(e, "self_attn.q_norm.weight", HDF, 1, REPL)),
        k_norm    = SlotOffset[BF16, Shape[HDF, 1]](b.bf(e, "self_attn.k_norm.weight", HDF, 1, REPL)),
        q_colsum  = b.colsum(q_n_loc * 4),
        k_colsum  = b.colsum(C.KV_DIM_FULL * 4),
        o_colsum  = b.colsum(H * o_num_blk * 4),
    )
    return (FullLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


# =============================================================================
# Scratch budget
# =============================================================================


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16 = 2
    comptime i8 = 1
    comptime f32 = 4
    comptime topk_bytes = size_of[Gemma4TopKResult[C.TOP_K]]()
    comptime S = Gemma4Shapes[tp]

    comptime persistent = f32 + S.DENSE_DOWN_NUM_BLK * f32

    comptime sliding_attn_peak = persistent + (
        C.HIDDEN * i8 + C.HIDDEN * f32 + f32
        + S.SlidingQ.N * bf16
        + 2 * (C.KV_DIM_SLIDING // tp) * bf16
        + S.SlidingQ.N * i8
        + (S.SlidingQ.N // C.HEAD_DIM_SLIDING) * f32
    )

    comptime full_q_bf16 = S.FullQ.N * bf16
    comptime full_k_bf16 = C.KV_DIM_FULL * bf16
    comptime FULL_ATTN_MAX_CHUNKS = 32
    comptime full_attn_phase1 = persistent + (
        full_q_bf16 + full_k_bf16
        + C.HIDDEN * i8 + C.HIDDEN * f32 + f32
    )
    comptime full_attn_cp_extra = (
        C.Q_DIM_FULL * bf16
        + C.NUM_HEADS * f32 * 2
        + C.NUM_HEADS * C.HEAD_DIM_FULL * f32
    )
    comptime full_attn_phase2 = persistent + (
        full_q_bf16 + full_k_bf16
        + S.FullQ.N * i8
        + (S.FullQ.N // C.HEAD_DIM_FULL) * f32
        + S.FULL_HPG * C.HEAD_DIM_FULL * i8
        + S.FULL_HPG * f32 * 2
        + FULL_ATTN_MAX_CHUNKS * S.FULL_HPG * (2 + C.HEAD_DIM_FULL) * f32
        + full_attn_cp_extra
    )
    comptime full_attn_peak = (
        full_attn_phase1 if full_attn_phase1 > full_attn_phase2 else full_attn_phase2
    )

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
        + C.HIDDEN * bf16
        + C.HIDDEN * bf16
    )

    comptime layer_peak = (
        sliding_attn_peak if sliding_attn_peak > full_attn_peak else full_attn_peak
    )
    comptime decode_peak = ffn_peak if ffn_peak > layer_peak else layer_peak

    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    comptime lm_head_peak = (
        C.HIDDEN * i8 + vocab_num_blocks * f32
        + C.HIDDEN * f32 + C.VOCAB_SIZE * bf16
    )
    return lm_head_peak if lm_head_peak > decode_peak else decode_peak


# =============================================================================
# Build plan
# =============================================================================


@fieldwise_init
struct Gemma4LoadPlan[tp: Int](Movable):
    var topology: Gemma4Topology[Self.tp]
    var descs: List[WeightDesc]


def build_gemma4_plan[tp: Int]() -> Gemma4LoadPlan[tp]:
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

    # State block
    comptime bf16 = size_of[Scalar[DType.bfloat16]]()
    comptime f32 = size_of[Float32]()
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
    comptime LOCAL_MAX_SEQ_FULL = (C.MAX_SEQ_LEN + tp - 1) // tp
    comptime full_cache_stride = Gemma4KVCache[
        LOCAL_MAX_SEQ_FULL, C.HEAD_DIM_FULL,
        C.NUM_KV_HEADS_FULL, C.NUM_HEADS].TOTAL_BYTES
    var sliding_cache_off = state.reserve_bytes(C.NUM_SLIDING_LAYERS * sliding_cache_stride)
    var full_cache_off = state.reserve_bytes(C.NUM_FULL_LAYERS * full_cache_stride)

    # Host section
    var host_off = align_up(distributed + state.bytes())
    comptime HOST = LayerShard.HOST
    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var final_norm_off = hb.bf(descs, "model.language_model.norm.weight", C.HIDDEN, 1, HOST)
    var embed_off = hb.q(descs, "model.language_model.embed_tokens.weight", C.VOCAB_SIZE, C.HIDDEN, HOST)
    var embed_sc_off = hb.f(descs, "model.language_model.embed_tokens.weight_scale", C.VOCAB_SIZE, vocab_num_blocks, HOST)
    var embed_colsum_bytes = C.VOCAB_SIZE * vocab_num_blocks * f32
    var embed_colsum_off = hb.colsum(embed_colsum_bytes)
    var sqrt_gamma_off = embed_colsum_off + embed_colsum_bytes
    var inv_sqrt_gamma_off = sqrt_gamma_off + C.HIDDEN * bf16
    var host_bytes = inv_sqrt_gamma_off + C.HIDDEN * f32

    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](final_norm_off),
        embed=SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]](embed_off),
        embed_sc=SlotOffset[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]](embed_sc_off),
        embed_colsum_off=embed_colsum_off,
        sqrt_gamma_off=sqrt_gamma_off,
        inv_sqrt_gamma_off=inv_sqrt_gamma_off)

    var topo = Gemma4Topology[tp](
        arena_base=0,
        sliding=Repeated[SlidingLayerRefs[tp]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[tp]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        distributed_bytes=distributed,
        activations=activations,
        scratch_off=scratch_off, scratch_capacity=scratch_cap,
        sliding_rope=sliding_rope, full_rope=full_rope,
        sliding_cache_off=sliding_cache_off,
        sliding_cache_stride=sliding_cache_stride,
        full_cache_off=full_cache_off,
        full_cache_stride=full_cache_stride,
        state_bytes=state.bytes(),
        host=host, host_bytes=host_bytes)
    return Gemma4LoadPlan[tp](topo, descs^)


# =============================================================================
# Init helpers — colsums + VNNI packing for shared body weights
# =============================================================================


def zero_row_padding(base: Int, weight_off: Int, actual_rows: Int, padded_rows: Int, cols: Int):
    """Zero the alignment padding rows at the end of a weight matrix."""
    if actual_rows >= padded_rows:
        return
    var start = base + weight_off + actual_rows * cols
    var nbytes = (padded_rows - actual_rows) * cols
    comptime width = simd_width_of[DType.uint8]()
    var ptr = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](unsafe_from_address=start)
    var i = 0
    while i + width <= nbytes:
        (ptr + i).store(SIMD[DType.uint8, width](0))
        i += width
    while i < nbytes:
        ptr[i] = 0
        i += 1


def init_layer_body_padding[tp: Int](arena_base: Int, layer_off: Int, body: BodyRefs[tp]):
    comptime S = Gemma4Shapes[tp]
    comptime actual = C.INTERMEDIATE // tp
    comptime padded = S.DENSE_INT_LOCAL
    zero_row_padding(arena_base, layer_off + body.gate_proj.offset, actual, padded, C.HIDDEN)
    zero_row_padding(arena_base, layer_off + body.up_proj.offset, actual, padded, C.HIDDEN)


def init_layer_body_colsums[tp: Int](arena_base: Int, layer_off: Int, body: BodyRefs[tp]):
    comptime experts_local = C.NUM_EXPERTS // tp
    comptime S = Gemma4Shapes[tp]
    colsum_at(arena_base, layer_off + body.gate_proj.offset, layer_off + body.gu_colsum,
        S.DENSE_INT_LOCAL * 2, C.HIDDEN)
    block_colsum_at(arena_base, layer_off + body.down_proj.offset, layer_off + body.down_colsum,
        C.HIDDEN, S.DENSE_INT_LOCAL, FWHT_BLK_DENSE_DOWN)
    colsum_at(arena_base, layer_off + body.router_proj.offset, layer_off + body.router_colsum,
        C.NUM_EXPERTS, C.HIDDEN)
    colsum_at(arena_base, layer_off + body.experts_gate_up.offset, layer_off + body.experts_gu_colsum,
        experts_local * C.MOE_GATE_UP_FUSED, C.HIDDEN)
    for e in range(experts_local):
        block_colsum_at(arena_base,
            layer_off + body.experts_down.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            layer_off + body.experts_down_colsum + e * C.HIDDEN * MOE_NUM_BLOCKS * 4,
            C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK)


def init_layer_body_pack[tp: Int](arena_base: Int, layer_off: Int, body: BodyRefs[tp],
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    comptime experts_local = C.NUM_EXPERTS // tp
    comptime S = Gemma4Shapes[tp]
    pack_at(arena_base, layer_off + body.gate_proj.offset, S.DENSE_INT_LOCAL * 2, C.HIDDEN, scratch)
    pack_at(arena_base, layer_off + body.down_proj.offset, C.HIDDEN, S.DENSE_INT_LOCAL, scratch)
    pack_at(arena_base, layer_off + body.router_proj.offset, C.NUM_EXPERTS, C.HIDDEN, scratch)
    for e in range(experts_local):
        pack_at(arena_base, layer_off + body.experts_gate_up.offset + e * C.MOE_GATE_UP_FUSED * C.HIDDEN,
            C.MOE_GATE_UP_FUSED, C.HIDDEN, scratch)
        pack_at(arena_base, layer_off + body.experts_down.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            C.HIDDEN, C.MOE_INTERMEDIATE, scratch)


# =============================================================================
# Multi-rank dispatch
# =============================================================================


def tp_parallel[tp: Int,
    body: def[rank: Int](Gemma4Topology[tp], mut BurstPool[]) capturing -> PoolFence[BurstPool[]],
](
    topos: InlineArray[Gemma4Topology[tp], tp],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
) -> PhaseTiming:
    var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    var active = InlineArray[Bool, tp](fill=False)
    var t0 = Int(perf_counter_ns())
    comptime for rank in range(tp):
        ptrs[rank] = body[rank](topos[rank], pool_ptrs[rank][]).take()
    var t1 = Int(perf_counter_ns())
    for i in range(tp):
        if ptrs[i]:
            active[i] = True
            ptrs[i][].join()
    var t2 = Int(perf_counter_ns())

    var max_done_ns = 0
    var any_active = False
    for i in range(tp):
        if active[i]:
            any_active = True
            var ts = ptrs[i][].last_worker_timestamp()
            if ts > max_done_ns:
                max_done_ns = ts
    return phase_timing_from_points(t0, t1, max_done_ns, t1, t2, any_active)


# =============================================================================
# Model struct
# =============================================================================


struct Gemma4ButterQuant[tp: Int](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var topos: InlineArray[Gemma4Topology[Self.tp], Self.tp]
    var profile: ForwardLogger

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[BurstPool[]],
        var sc: ScratchPool,
        topos: InlineArray[Gemma4Topology[Self.tp], Self.tp],
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
    def describe_quantization[V: TaskVisitor](mut visitor: V) -> Bool:
        comptime Bf16 = Bf16Converter
        comptime HB = FWHT_BLK_HIDDEN
        comptime LB = LM_HEAD_FWHT_BLK
        var ok = True

        for i in range(C.NUM_LAYERS):
            var p = "model.language_model.layers." + String(i) + "."
            var is_full = is_full_layer(i)

            ok &= visitor.quantize(PerRow[Bf16](p + "self_attn.q_proj.weight", HB))
            ok &= visitor.quantize(PerRow[Bf16](p + "self_attn.k_proj.weight", HB))
            if not is_full:
                ok &= visitor.quantize(PerRow[Bf16](p + "self_attn.v_proj.weight", HB))
            if is_full:
                ok &= visitor.quantize(PerRow[Bf16](p + "self_attn.o_proj.weight", 512))
            else:
                ok &= visitor.quantize(PerRow[Bf16](p + "self_attn.o_proj.weight", HB))

            ok &= visitor.quantize(PerRow[Bf16](p + "mlp.gate_proj.weight", HB))
            ok &= visitor.quantize(PerRow[Bf16](p + "mlp.up_proj.weight", HB))
            ok &= visitor.quantize(PerRow[Bf16](p + "mlp.down_proj.weight", FWHT_BLK_DENSE_DOWN))

            ok &= visitor.quantize(PerRow[Bf16](p + "router.proj.weight", HB))
            ok &= visitor.quantize(PerRow[Bf16](p + "experts.gate_up_proj", HB))
            ok &= visitor.quantize(PerRow[Bf16](p + "experts.down_proj", FWHT_BLK))

            ok &= visitor.passthrough(p + "input_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "post_attention_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "pre_feedforward_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "pre_feedforward_layernorm_2.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "post_feedforward_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "post_feedforward_layernorm_1.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "post_feedforward_layernorm_2.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "self_attn.q_norm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "self_attn.k_norm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "router.scale", DType.bfloat16)
            ok &= visitor.passthrough(p + "router.per_expert_scale", DType.bfloat16)
            ok &= visitor.passthrough(p + "layer_scalar", DType.bfloat16)

        ok &= visitor.passthrough("model.language_model.norm.weight", DType.bfloat16)
        ok &= visitor.quantize(PerBlockAbsorbed[Bf16](
            "model.language_model.embed_tokens.weight", LB,
            "model.language_model.norm.weight"))
        return ok

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    def init_state(mut self):
        comptime MAX_PACK_BYTES = C.Q_DIM_FULL * C.HIDDEN
        var numa = NumaInfo()
        var pack_arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa.plan_topology(1)[0], MAX_PACK_BYTES)
        var pack_scratch = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(pack_arena.base))

        for rank in range(Self.tp):
            var topo = self.topos[rank]
            var sb = topo.state_base()
            init_sliding_rope_tables(
                topo.sliding_rope.cos.bound(sb),
                topo.sliding_rope.sin.bound(sb))
            init_full_rope_tables(
                topo.full_rope.cos.bound(sb),
                topo.full_rope.sin.bound(sb))

            var base = topo.arena_base
            var sliding_idx = 0
            var full_idx = 0
            for i in range(C.NUM_LAYERS):
                if is_full_layer(i):
                    var lb = topo.full_base(full_idx)
                    var fl = topo.full.proto
                    comptime q_local = C.Q_DIM_FULL // Self.tp
                    # Q projection colsum + pack (sharded)
                    colsum_at(base, lb - base + fl.attn.q_proj.offset, lb - base + fl.attn.q_colsum,
                        q_local, C.HIDDEN)
                    pack_at(base, lb - base + fl.attn.q_proj.offset, q_local, C.HIDDEN, pack_scratch)
                    # K projection colsum + pack (replicated)
                    colsum_at(base, lb - base + fl.attn.k_proj.offset, lb - base + fl.attn.k_colsum,
                        C.KV_DIM_FULL, C.HIDDEN)
                    pack_at(base, lb - base + fl.attn.k_proj.offset, C.KV_DIM_FULL, C.HIDDEN, pack_scratch)
                    # O projection colsum + pack
                    block_colsum_at(base, lb - base + fl.attn.o_proj.offset, lb - base + fl.attn.o_colsum,
                        C.HIDDEN, q_local, C.HEAD_DIM_FULL)
                    pack_at(base, lb - base + fl.attn.o_proj.offset, C.HIDDEN, q_local, pack_scratch)
                    init_layer_body_padding[Self.tp](base, lb - base, fl.body)
                    init_layer_body_colsums[Self.tp](base, lb - base, fl.body)
                    init_layer_body_pack[Self.tp](base, lb - base, fl.body, pack_scratch)
                    full_idx += 1
                else:
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    comptime q_local = C.Q_DIM_SLIDING // Self.tp
                    comptime kv_n = 2 * (C.KV_DIM_SLIDING // Self.tp)
                    # Q projection colsum + pack (sharded)
                    colsum_at(base, lb - base + sl.attn.q_proj.offset, lb - base + sl.attn.q_colsum,
                        q_local, C.HIDDEN)
                    pack_at(base, lb - base + sl.attn.q_proj.offset, q_local, C.HIDDEN, pack_scratch)
                    # KV projection colsum + pack (sharded, k+v contiguous)
                    colsum_at(base, lb - base + sl.attn.k_proj.offset, lb - base + sl.attn.kv_colsum,
                        kv_n, C.HIDDEN)
                    pack_at(base, lb - base + sl.attn.k_proj.offset, kv_n, C.HIDDEN, pack_scratch)
                    # O projection colsum + pack
                    block_colsum_at(base, lb - base + sl.attn.o_proj.offset, lb - base + sl.attn.o_colsum,
                        C.HIDDEN, q_local, C.HEAD_DIM_SLIDING)
                    pack_at(base, lb - base + sl.attn.o_proj.offset, C.HIDDEN, q_local, pack_scratch)
                    init_layer_body_padding[Self.tp](base, lb - base, sl.body)
                    init_layer_body_colsums[Self.tp](base, lb - base, sl.body)
                    init_layer_body_pack[Self.tp](base, lb - base, sl.body, pack_scratch)
                    sliding_idx += 1

            if rank == HOST_RANK:
                block_colsum_row_major_at(base,
                    topo.host.embed.offset, topo.host.embed_colsum_off,
                    C.VOCAB_SIZE, C.HIDDEN, LM_HEAD_FWHT_BLK)
                var fn_gamma = BF16Ptr(unsafe_from_address=base + topo.host.final_norm.offset)
                compute_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    BF16Ptr(unsafe_from_address=base + topo.host.sqrt_gamma_off))
                compute_inv_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    F32Ptr(unsafe_from_address=base + topo.host.inv_sqrt_gamma_off))

            # Fold 1/sqrt(hidden) into router weight scales
            comptime inv_sqrt_h = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
            sliding_idx = 0
            full_idx = 0
            for i in range(C.NUM_LAYERS):
                var sc_ptr: UnsafePointer[Float32, MutAnyOrigin]
                if is_full_layer(i):
                    var lb = topo.full_base(full_idx)
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=topo.full.proto.body.router_proj_sc.addr(lb))
                    full_idx += 1
                else:
                    var lb = topo.sliding_base(sliding_idx)
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=topo.sliding.proto.body.router_proj_sc.addr(lb))
                    sliding_idx += 1
                for n in range(C.NUM_EXPERTS):
                    sc_ptr[n] *= inv_sqrt_h

        print("state initialized")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        if C.INTERMEDIATE % (Self.tp * FWHT_BLK_DENSE_DOWN) != 0:
            print(
                "unsupported TP=", Self.tp,
                ": dense FFN sharding requires C.INTERMEDIATE / tp aligned to FWHT_BLK_DENSE_DOWN;",
                " got", C.INTERMEDIATE, "/", Self.tp, "=",
                C.INTERMEDIATE // Self.tp, "with FWHT_BLK_DENSE_DOWN=", FWHT_BLK_DENSE_DOWN,
            )
            return None

        if C.NUM_KV_HEADS_SLIDING % Self.tp != 0:
            print(
                "unsupported TP=", Self.tp,
                ": sliding attention requires NUM_KV_HEADS_SLIDING divisible by tp;",
                " got NUM_KV_HEADS_SLIDING=", C.NUM_KV_HEADS_SLIDING,
            )
            return None

        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_gemma4_plan[Self.tp]()

        var numa = NumaInfo()
        var numa_topo = numa.plan_topology(Self.tp)

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = plan.topology.host_arena_bytes() if rank == HOST_RANK else plan.topology.arena_bytes()
            print("rank", rank, "node", numa_topo[rank], "allocating", size // (1024 * 1024), "MB")
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa_topo[rank], size)
            if not arena:
                print("arena allocation failed for rank", rank)
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_weights_from_descs(plan.descs, shards, arena_bases, numa_topo)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(plan.topology.distributed_bytes, plan.topology.state_bytes)

        var main_pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            main_pools.push(BurstPool[].for_numa_node(numa, numa_topo[rank], headroom=2))

        var topos = InlineArray[Gemma4Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))

        var scratch = ScratchPool(plan.topology.scratch_capacity)
        var model = Self(arenas^, main_pools^, scratch^, topos)
        model.init_state()
        return model^

    # =========================================================================
    # Forward — decode (seq_len=1)
    # =========================================================================

    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> Int32:
        comptime S = Gemma4Shapes[Self.tp]
        comptime Q_DIM_LOCAL_SLIDING = C.Q_DIM_SLIDING // Self.tp
        comptime KV_DIM_LOCAL_SLIDING = C.KV_DIM_SLIDING // Self.tp
        comptime Q_DIM_LOCAL_FULL = C.Q_DIM_FULL // Self.tp
        comptime EPS = Float32(C.RMS_NORM_EPS)
        comptime seq_len = 1
        comptime DENSE_INT_LOCAL = S.DENSE_INT_LOCAL
        comptime DENSE_DOWN_NUM_BLK = S.DENSE_DOWN_NUM_BLK
        comptime FULL_HPG = S.FULL_HPG
        comptime ROPE_DIMS_FULL = 128
        comptime FULL_ATTN_MAX_CHUNKS = 32
        comptime X_SLOT = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]
        comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_HEAD_FWHT_BLK

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(pos)
        var topos = self.topos
        var host = topos[0]
        var mp = self.pool_ptrs()
        # --- Embed ---
        var t_embed0 = Int(perf_counter_ns())
        var embed_fence = embed_lookup_blocked[fwht_blk = LM_HEAD_FWHT_BLK](
            host.host.embed.bound(host.arena_base),
            host.host.embed_sc.bound(host.arena_base),
            host.arena_base + host.host.inv_sqrt_gamma_off,
            tokens_ptr,
            host.x_main(seq_len), Float32(EMBED_SCALE),
            self.main_pools[0])
        var t_embed1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("embed"), finish_single_pool_fence(t_embed0, t_embed1, embed_fence^))

        var t_bcast0 = Int(perf_counter_ns())
        ring_broadcast[X_SLOT, Self.tp](
            host.x_main(seq_len).ptr, self.x_main_ptrs(seq_len), seq_len, mp)
        sample.add(self.profile.phase("broadcast"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_bcast0))

        var act_scale_lease = self.scratch.borrow[Float32, 1]()
        var post_blk_scale_lease = self.scratch.borrow[Float32, DENSE_DOWN_NUM_BLK]()

        var sliding_idx = 0
        var full_idx = 0
        for layer_idx in range(C.NUM_LAYERS):
            var is_full = is_full_layer(layer_idx)

            # =============================================================
            # ATTENTION BLOCK
            # =============================================================

            if not is_full:
                var attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
                var attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
                var attn_scale_lease = self.scratch.borrow[Float32, 1]()

                @parameter
                def do_attn_quantize[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                        topo.x_main(seq_len).ptr, sl.body.input_norm.addr(lb),
                        topo.scratch_addr(attn_i8_lease),
                        topo.scratch_addr(attn_work_lease), topo.scratch_addr(attn_scale_lease),
                        EPS, seq_len, pool)
                sample.add(self.profile.phase("local_attn_quantize"), tp_parallel[Self.tp, do_attn_quantize](topos, mp))

                var q_lease = self.scratch.borrow[Scalar[DType.bfloat16], Q_DIM_LOCAL_SLIDING]()
                var kv_lease = self.scratch.borrow[Scalar[DType.bfloat16], 2 * KV_DIM_LOCAL_SLIDING]()

                @parameter
                def do_q_gemv[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    return int8_gemv[Q_DIM_LOCAL_SLIDING, C.HIDDEN](
                        topo.scratch_addr(attn_i8_lease),
                        sl.attn.q_proj.addr(lb),
                        lb + sl.attn.q_colsum,
                        sl.attn.q_proj_sc.addr(lb),
                        topo.scratch_addr(q_lease),
                        seq_len, topo.scratch_addr(attn_scale_lease), pool)
                sample.add(self.profile.phase("local_attn_proj"), tp_parallel[Self.tp, do_q_gemv](topos, mp))

                @parameter
                def do_kv_gemv[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    return int8_gemv[2 * KV_DIM_LOCAL_SLIDING, C.HIDDEN](
                        topo.scratch_addr(attn_i8_lease),
                        sl.attn.k_proj.addr(lb),
                        lb + sl.attn.kv_colsum,
                        sl.attn.k_proj_sc.addr(lb),
                        topo.scratch_addr(kv_lease),
                        seq_len, topo.scratch_addr(attn_scale_lease), pool)
                sample.add(self.profile.phase("local_attn_proj"), tp_parallel[Self.tp, do_kv_gemv](topos, mp))

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_SLIDING]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, Q_DIM_LOCAL_SLIDING // C.HEAD_DIM_SLIDING]()

                @parameter
                def do_sliding_attn[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
                    comptime NKV = C.NUM_KV_HEADS_SLIDING // Self.tp
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    return sliding_attn_dispatch[
                        C.HEAD_DIM_SLIDING, HPG, C.SLIDING_WINDOW, NKV, C.NUM_HEADS // Self.tp](
                        topo.scratch_addr(q_lease),
                        topo.scratch_addr(kv_lease),
                        topo.scratch_addr(kv_lease) + KV_DIM_LOCAL_SLIDING * 2,
                        sl.attn.q_norm.addr(lb), sl.attn.k_norm.addr(lb),
                        topo.sliding_cos_row(pos), topo.sliding_sin_row(pos),
                        topo.sliding_cache_base(sliding_idx),
                        pos % C.SLIDING_WINDOW, min(pos + 1, C.SLIDING_WINDOW),
                        topo.scratch_addr(attn_qi_lease), topo.scratch_addr(attn_head_sc_lease),
                        EPS, pool)
                sample.add(self.profile.phase("local_attention"), tp_parallel[Self.tp, do_sliding_attn](topos, mp))

                @parameter
                def do_o_proj[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.sliding_base(sliding_idx)
                    var sl = topo.sliding.proto
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_SLIDING, C.HEAD_DIM_SLIDING](
                        I8Ptr(unsafe_from_address=topo.scratch_addr(attn_qi_lease)),
                        U8Ptr(unsafe_from_address=sl.attn.o_proj.addr(lb)),
                        F32Ptr(unsafe_from_address=topo.scratch_addr(attn_head_sc_lease)),
                        sl.attn.o_proj_sc.bound(lb).as_ptr(),
                        F32Ptr(unsafe_from_address=lb + sl.attn.o_colsum),
                        topo.x_residual(seq_len).as_ptr(),
                        seq_len, pool)
                sample.add(self.profile.phase("local_o_proj"), tp_parallel[Self.tp, do_o_proj](topos, mp))
                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                kv_lease^.release()
                q_lease^.release()
                attn_scale_lease^.release()
                attn_work_lease^.release()
                attn_i8_lease^.release()
            else:
                # Full attention — context parallel only (not head-parallel TP/CP
                # flash decode as in M27). Gemma4 has only 2 full-attention KV heads,
                # so head-sharding caps at tp=2 and gives zero benefit beyond that.
                # CP position-sharding scales to any TP with negligible coordination
                # cost: 16KB Q allgather + scalar remote reads for cross-rank merge.
                comptime LOCAL_MAX_SEQ_FULL = (C.MAX_SEQ_LEN + Self.tp - 1) // Self.tp
                comptime HEADS_PER_RANK = C.NUM_HEADS // Self.tp
                comptime Q_GROUP_BF16 = FULL_HPG * C.HEAD_DIM_FULL * 2
                comptime K_HEAD_BF16 = C.HEAD_DIM_FULL * 2

                var full_q_lease = self.scratch.borrow[Scalar[DType.bfloat16], Q_DIM_LOCAL_FULL]()
                var full_k_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.KV_DIM_FULL]()

                var full_attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
                var full_attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
                var full_attn_scale_lease = self.scratch.borrow[Float32, 1]()

                @parameter
                def do_full_attn_quantize[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.full_base(full_idx)
                    var fl = topo.full.proto
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                        topo.x_main(seq_len).ptr, fl.body.input_norm.addr(lb),
                        topo.scratch_addr(full_attn_i8_lease),
                        topo.scratch_addr(full_attn_work_lease), topo.scratch_addr(full_attn_scale_lease),
                        EPS, seq_len, pool)
                sample.add(self.profile.phase("global_attn_quantize"), tp_parallel[Self.tp, do_full_attn_quantize](topos, mp))

                @parameter
                def do_full_q_gemv[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.full_base(full_idx)
                    var fl = topo.full.proto
                    return int8_gemv[Q_DIM_LOCAL_FULL, C.HIDDEN](
                        topo.scratch_addr(full_attn_i8_lease),
                        fl.attn.q_proj.addr(lb),
                        lb + fl.attn.q_colsum,
                        fl.attn.q_proj_sc.addr(lb),
                        topo.scratch_addr(full_q_lease),
                        seq_len, topo.scratch_addr(full_attn_scale_lease), pool)
                sample.add(self.profile.phase("global_attn_proj"), tp_parallel[Self.tp, do_full_q_gemv](topos, mp))

                @parameter
                def do_full_k_gemv[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.full_base(full_idx)
                    var fl = topo.full.proto
                    return int8_gemv[C.KV_DIM_FULL, C.HIDDEN](
                        topo.scratch_addr(full_attn_i8_lease),
                        fl.attn.k_proj.addr(lb),
                        lb + fl.attn.k_colsum,
                        fl.attn.k_proj_sc.addr(lb),
                        topo.scratch_addr(full_k_lease),
                        seq_len, topo.scratch_addr(full_attn_scale_lease), pool)
                sample.add(self.profile.phase("global_attn_proj"), tp_parallel[Self.tp, do_full_k_gemv](topos, mp))
                full_attn_scale_lease^.release()
                full_attn_work_lease^.release()
                full_attn_i8_lease^.release()

                # Q all-gather: each rank pulls all Q shards locally
                var full_q_all_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.Q_DIM_FULL]()
                var q_shard_ptrs = InlineArray[Int, Self.tp](fill=0)
                var q_dst_ptrs = InlineArray[Int, Self.tp](fill=0)
                for r in range(Self.tp):
                    q_shard_ptrs[r] = topos[r].scratch_addr(full_q_lease)
                    q_dst_ptrs[r] = topos[r].scratch_addr(full_q_all_lease)
                ring_allgather[Self.tp](
                    q_shard_ptrs, q_dst_ptrs, Q_DIM_LOCAL_FULL * 2, mp)

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_FULL]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, HEADS_PER_RANK]()
                var q_i8_prep_lease = self.scratch.borrow[Scalar[DType.int8], FULL_HPG * C.HEAD_DIM_FULL]()
                var qi_biases_lease = self.scratch.borrow[Float32, FULL_HPG]()
                var q_scales_lease = self.scratch.borrow[Float32, FULL_HPG]()
                comptime PARTIAL_F32S = FULL_ATTN_MAX_CHUNKS * FULL_HPG * (2 + C.HEAD_DIM_FULL)
                var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()
                var cp_m_lease = self.scratch.borrow[Float32, C.NUM_HEADS]()
                var cp_l_lease = self.scratch.borrow[Float32, C.NUM_HEADS]()
                var cp_v_lease = self.scratch.borrow[Float32, C.NUM_HEADS * C.HEAD_DIM_FULL]()

                for kv in range(C.NUM_KV_HEADS_FULL):
                    @parameter
                    def do_cp_attn_prep[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                        var lb = topo.full_base(full_idx)
                        var fl = topo.full.proto
                        var lp = cp_local_pos(pos, Self.tp)
                        var is_owner = cp_owning_rank(pos, Self.tp) == rank
                        return cp_attn_prep_dispatch[
                            C.HEAD_DIM_FULL, ROPE_DIMS_FULL, FULL_HPG,
                            LOCAL_MAX_SEQ_FULL, C.NUM_KV_HEADS_FULL, C.NUM_HEADS](
                            topo.scratch_addr(full_q_all_lease) + kv * Q_GROUP_BF16,
                            topo.scratch_addr(full_k_lease) + kv * K_HEAD_BF16,
                            fl.attn.q_norm.addr(lb), fl.attn.k_norm.addr(lb),
                            topo.state_base() + topo.full_rope.cos.offset + pos * 64 * size_of[Float32](),
                            topo.state_base() + topo.full_rope.sin.offset + pos * 64 * size_of[Float32](),
                            topo.full_cache_base(full_idx), lp, kv,
                            EPS, is_owner,
                            topo.scratch_addr(q_i8_prep_lease),
                            topo.scratch_addr(qi_biases_lease),
                            topo.scratch_addr(q_scales_lease),
                            pool)
                    sample.add(self.profile.phase("global_attention"), tp_parallel[Self.tp, do_cp_attn_prep](topos, mp))

                    @parameter
                    def do_cp_chunk_attn[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                        var local_ctx = cp_local_context_len(pos + 1, rank, Self.tp)
                        return cp_chunked_attn_dispatch[
                            C.HEAD_DIM_FULL, LOCAL_MAX_SEQ_FULL,
                            C.NUM_KV_HEADS_FULL, C.NUM_HEADS, FULL_HPG](
                            topo.scratch_addr(q_i8_prep_lease),
                            topo.scratch_addr(qi_biases_lease),
                            topo.scratch_addr(q_scales_lease),
                            topo.full_cache_base(full_idx), kv,
                            local_ctx, Int(pool.capacity),
                            topo.scratch_addr(partial_lease),
                            pool)
                    sample.add(self.profile.phase("global_attention"), tp_parallel[Self.tp, do_cp_chunk_attn](topos, mp))

                    @parameter
                    def do_merge_chunks[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                        var local_ctx = cp_local_context_len(pos + 1, rank, Self.tp)
                        var num_pg = (local_ctx + CACHE_WIDTH - 1) // CACHE_WIDTH
                        var nc = min(Int(pool.capacity), FULL_ATTN_MAX_CHUNKS)
                        if nc > num_pg:
                            nc = num_pg
                        return merge_local_chunks_dispatch[C.HEAD_DIM_FULL, FULL_HPG](
                            topo.scratch_addr(partial_lease), nc,
                            topo.scratch_addr(cp_m_lease) + kv * FULL_HPG * 4,
                            topo.scratch_addr(cp_l_lease) + kv * FULL_HPG * 4,
                            topo.scratch_addr(cp_v_lease) + kv * FULL_HPG * C.HEAD_DIM_FULL * 4,
                            pool)
                    sample.add(self.profile.phase("global_attention"), tp_parallel[Self.tp, do_merge_chunks](topos, mp))

                # Cross-rank gather + quantize: each rank's pool merges V
                # from all ranks (remote reads) and writes qi/scales locally
                var all_m_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                var all_l_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                var all_v_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                for r in range(Self.tp):
                    all_m_addrs[r] = topos[r].scratch_addr(cp_m_lease)
                    all_l_addrs[r] = topos[r].scratch_addr(cp_l_lease)
                    all_v_addrs[r] = topos[r].scratch_addr(cp_v_lease)

                @parameter
                def do_cp_gather[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    return cp_gather_dispatch[C.HEAD_DIM_FULL, C.NUM_HEADS, Self.tp](
                        rank, all_m_addrs, all_l_addrs, all_v_addrs,
                        topo.scratch_addr(attn_qi_lease),
                        topo.scratch_addr(attn_head_sc_lease),
                        rank * HEADS_PER_RANK, HEADS_PER_RANK, pool)
                sample.add(self.profile.phase("global_attention"), tp_parallel[Self.tp, do_cp_gather](topos, mp))

                cp_v_lease^.release()
                cp_l_lease^.release()
                cp_m_lease^.release()
                partial_lease^.release()
                q_scales_lease^.release()
                qi_biases_lease^.release()
                q_i8_prep_lease^.release()

                @parameter
                def do_full_o_proj[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var lb = topo.full_base(full_idx)
                    var fl = topo.full.proto
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_FULL, C.HEAD_DIM_FULL](
                        I8Ptr(unsafe_from_address=topo.scratch_addr(attn_qi_lease)),
                        U8Ptr(unsafe_from_address=fl.attn.o_proj.addr(lb)),
                        F32Ptr(unsafe_from_address=topo.scratch_addr(attn_head_sc_lease)),
                        fl.attn.o_proj_sc.bound(lb).as_ptr(),
                        F32Ptr(unsafe_from_address=lb + fl.attn.o_colsum),
                        topo.x_residual(seq_len).as_ptr(),
                        seq_len, pool)
                sample.add(self.profile.phase("global_o_proj"), tp_parallel[Self.tp, do_full_o_proj](topos, mp))

                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                full_q_all_lease^.release()
                full_k_lease^.release()
                full_q_lease^.release()

            # Allreduce + post-attn norm
            var t_attn_reduce0 = Int(perf_counter_ns())
            small_allreduce[X_SLOT, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len, mp)
            if is_full:
                sample.add(self.profile.phase("global_attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))
            else:
                sample.add(self.profile.phase("local_attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))

            @parameter
            def do_post_attn_norm[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return post_attn_norm_dispatch[C.HIDDEN](
                    topo.x_residual(seq_len).ptr,
                    body.post_attn_norm.addr(lb),
                    topo.x_main(seq_len).ptr, EPS, pool)
            sample.add(self.profile.phase("post_attn_norm"), tp_parallel[Self.tp, do_post_attn_norm](topos, mp))

            # =============================================================
            # FFN BLOCK
            # =============================================================

            var act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var act_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var act_work2_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var expert_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var expert_act_scale_lease = self.scratch.borrow[Float32, 1]()

            @parameter
            def do_router_quantize[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                    topo.x_main(seq_len).ptr, body.router_scale.addr(lb),
                    topo.scratch_addr(act_i8_lease),
                    topo.scratch_addr(act_work_lease), topo.scratch_addr(act_scale_lease),
                    EPS, seq_len, pool)
            sample.add(self.profile.phase("router_quantize"), tp_parallel[Self.tp, do_router_quantize](topos, mp))

            var router_logits_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.NUM_EXPERTS]()
            var routing_lease = self.scratch.borrow[Gemma4TopKResult[C.TOP_K], 1]()

            @parameter
            def do_router_gemv[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return int8_gemv[C.NUM_EXPERTS, C.HIDDEN](
                    topo.scratch_addr(act_i8_lease),
                    body.router_proj.addr(lb),
                    lb + body.router_colsum,
                    body.router_proj_sc.addr(lb),
                    topo.scratch_addr(router_logits_lease),
                    seq_len, topo.scratch_addr(act_scale_lease), pool)
            sample.add(self.profile.phase("router_proj"), tp_parallel[Self.tp, do_router_gemv](topos, mp))

            @parameter
            def do_router_topk[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return router_topk_dispatch[C.NUM_EXPERTS, C.TOP_K](
                    BF16Ptr(unsafe_from_address=topo.scratch_addr(router_logits_lease)),
                    body.router_pes.bound(lb).as_ptr(),
                    topo.scratch_addr(routing_lease), pool)
            sample.add(self.profile.phase("router_topk"), tp_parallel[Self.tp, do_router_topk](topos, mp))

            var expert_qi_lease = self.scratch.borrow[Scalar[DType.int8], C.TOP_K * C.MOE_INTERMEDIATE]()
            var expert_blk_scale_lease = self.scratch.borrow[Float32, C.TOP_K * MOE_NUM_BLOCKS]()
            var expert_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
            var local_count_lease = self.scratch.borrow[Int32, 1]()

            @parameter
            def do_ffn_quantize[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return rmsnorm_dual_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                    topo.x_main(seq_len).ptr,
                    body.pre_ffn_norm.addr(lb),
                    body.pre_ffn_norm_2.addr(lb),
                    topo.scratch_addr(act_i8_lease),
                    topo.scratch_addr(expert_act_i8_lease),
                    topo.scratch_addr(act_work_lease),
                    topo.scratch_addr(act_work2_lease),
                    topo.scratch_addr(act_scale_lease),
                    topo.scratch_addr(expert_act_scale_lease),
                    EPS, seq_len, pool)
            sample.add(self.profile.phase("ffn_quantize"), tp_parallel[Self.tp, do_ffn_quantize](topos, mp))

            @parameter
            def do_expert_phase1[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var routing = UnsafePointer[Gemma4TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=topo.scratch_addr(routing_lease))[]
                comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
                var expert_base = rank * experts_per_rank
                var lc = 0
                for s in range(C.TOP_K):
                    var eid = routing.indices[s]
                    if eid >= expert_base and eid < expert_base + experts_per_rank:
                        lc += 1
                UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=topo.scratch_addr(local_count_lease))[] = Int32(lc)
                return gemma4_moe_phase1[
                    C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(expert_act_i8_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(expert_act_scale_lease)),
                    routing,
                    body.experts_gate_up.addr(lb),
                    C.MOE_GATE_UP_FUSED * C.HIDDEN,
                    body.experts_gate_up_sc.addr(lb),
                    C.MOE_GATE_UP_FUSED * 4,
                    lb + body.experts_gu_colsum,
                    C.MOE_GATE_UP_FUSED * 4,
                    I8Ptr(unsafe_from_address=topo.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(expert_blk_scale_lease)),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase1"), tp_parallel[Self.tp, do_expert_phase1](topos, mp))

            var dense_post_i8_lease = self.scratch.borrow[Scalar[DType.int8], DENSE_INT_LOCAL]()

            @parameter
            def do_dense_phase1[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return fused_gu_gelu_tanh_wa[DENSE_INT_LOCAL, C.HIDDEN, FWHT_BLK_DENSE_DOWN](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(act_i8_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(act_scale_lease)),
                    U8Ptr(unsafe_from_address=body.gate_proj.addr(lb)),
                    body.gate_proj_sc.bound(lb).as_ptr(),
                    F32Ptr(unsafe_from_address=lb + body.gu_colsum),
                    I8Ptr(unsafe_from_address=topo.scratch_addr(dense_post_i8_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(post_blk_scale_lease)),
                    seq_len, pool)
            sample.add(self.profile.phase("dense_phase1"), tp_parallel[Self.tp, do_dense_phase1](topos, mp))

            @parameter
            def do_expert_phase2[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var routing = UnsafePointer[Gemma4TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=topo.scratch_addr(routing_lease))[]
                return gemma4_moe_phase2[
                    C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(expert_blk_scale_lease)),
                    routing,
                    body.experts_down.addr(lb),
                    C.HIDDEN * C.MOE_INTERMEDIATE,
                    body.experts_down_sc.addr(lb),
                    C.HIDDEN * 4,
                    lb + body.experts_down_colsum,
                    C.HIDDEN * MOE_NUM_BLOCKS * 4,
                    BF16Ptr(unsafe_from_address=topo.scratch_addr(expert_out_lease)),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase2"), tp_parallel[Self.tp, do_expert_phase2](topos, mp))

            var dense_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_dense_phase2[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return int8_gemv_blocked_wa[C.HIDDEN, DENSE_INT_LOCAL, FWHT_BLK_DENSE_DOWN](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(dense_post_i8_lease)),
                    U8Ptr(unsafe_from_address=body.down_proj.addr(lb)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(post_blk_scale_lease)),
                    body.down_proj_sc.bound(lb).as_ptr(),
                    F32Ptr(unsafe_from_address=lb + body.down_colsum),
                    BF16Ptr(unsafe_from_address=topo.scratch_addr(dense_out_lease)),
                    seq_len, pool)
            sample.add(self.profile.phase("dense_phase2"), tp_parallel[Self.tp, do_dense_phase2](topos, mp))

            var dense_normed_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_expert_sum[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lc = Int(UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=topo.scratch_addr(local_count_lease))[] )
                return expert_sum_dispatch[C.HIDDEN](
                    topo.scratch_addr(expert_out_lease), lc,
                    topo.x_residual(seq_len).ptr, pool)
            sample.add(self.profile.phase("pre_reduce"), tp_parallel[Self.tp, do_expert_sum](topos, mp))

            var t_dense_reduce0 = Int(perf_counter_ns())
            var dense_out_ptrs = InlineArray[Int, Self.tp](fill=0)
            for r in range(Self.tp):
                dense_out_ptrs[r] = topos[r].scratch_addr(dense_out_lease)
            small_allreduce[X_SLOT, Self.tp](dense_out_ptrs, 1, mp)
            sample.add(self.profile.phase("mlp_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_dense_reduce0))

            @parameter
            def do_dense_norm[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return dense_norm_dispatch[C.HIDDEN](
                    topo.scratch_addr(dense_out_lease),
                    body.post_ffn_norm_1.addr(lb),
                    topo.scratch_addr(dense_normed_lease), EPS, pool)
            sample.add(self.profile.phase("pre_reduce"), tp_parallel[Self.tp, do_dense_norm](topos, mp))

            var t_mlp_reduce0 = Int(perf_counter_ns())
            small_allreduce[X_SLOT, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len, mp)
            sample.add(self.profile.phase("mlp_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_mlp_reduce0))

            @parameter
            def do_post_reduce[rank: Int](topo: Gemma4Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.full_base(full_idx) if is_full else topo.sliding_base(sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var ls = body.layer_scalar.bound(lb).as_ptr()
                return post_reduce_dispatch[C.HIDDEN](
                    topo.x_residual(seq_len).ptr,
                    body.post_ffn_norm_2.addr(lb),
                    topo.scratch_addr(dense_normed_lease),
                    body.post_ffn_norm.addr(lb),
                    topo.x_main(seq_len).ptr, Float32(ls[]), EPS, pool)
            sample.add(self.profile.phase("post_reduce"), tp_parallel[Self.tp, do_post_reduce](topos, mp))

            dense_normed_lease^.release()
            dense_out_lease^.release()
            dense_post_i8_lease^.release()
            local_count_lease^.release()
            expert_out_lease^.release()
            expert_blk_scale_lease^.release()
            expert_qi_lease^.release()
            routing_lease^.release()
            router_logits_lease^.release()
            expert_act_scale_lease^.release()
            expert_act_i8_lease^.release()
            act_work2_lease^.release()
            act_work_lease^.release()
            act_i8_lease^.release()

            if is_full:
                full_idx += 1
            else:
                sliding_idx += 1

        post_blk_scale_lease^.release()
        act_scale_lease^.release()

        # --- Final norm + LM head ---
        var last_hidden_bf16 = host.x_main(seq_len).ptr
        var lm_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
        var lm_act_blk_scale_lease = self.scratch.borrow[Float32, VOCAB_NUM_BLOCKS]()
        var lm_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
        var t_final0 = Int(perf_counter_ns())
        var final_fence = rmsnorm_gamma_fwht_per_block_quantize[
            C.HIDDEN, LM_HEAD_FWHT_BLK](
            last_hidden_bf16,
            host.arena_base + host.host.sqrt_gamma_off,
            host.scratch_addr(lm_act_i8_lease),
            host.scratch_addr(lm_work_lease),
            host.scratch_addr(lm_act_blk_scale_lease),
            EPS, 1, self.main_pools[0])
        var t_final1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("final_norm"), finish_single_pool_fence(t_final0, t_final1, final_fence^))

        # lm_work is dead after final_norm, so release it before placing logits
        # above the remaining LM-head input leases.
        lm_work_lease^.release()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = scratch_tensor_view[BF16, 1, C.VOCAB_SIZE](host.scratch_base(), logit_lease, 1)
        var t_lm0 = Int(perf_counter_ns())
        var lm_fence = lm_head_gemv[
            C.VOCAB_SIZE, C.HIDDEN, LM_HEAD_FWHT_BLK](
            host.scratch_addr(lm_act_i8_lease),
            host.host.embed.addr(host.arena_base),
            host.scratch_addr(lm_act_blk_scale_lease),
            host.host.embed_sc.addr(host.arena_base),
            host.arena_base + host.host.embed_colsum_off,
            logit_view.ptr,
            self.main_pools[0])
        var t_lm1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("lm_head"), finish_single_pool_fence(t_lm0, t_lm1, lm_fence^))
        var t_softcap0 = Int(perf_counter_ns())
        logit_softcap(logit_view)
        sample.add(self.profile.phase("softcap"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_softcap0))

        comptime width = simd_width_of[DType.float32]()
        var logits = LogitsView[C.VOCAB_SIZE](
            scratch_ptr[Scalar[DType.bfloat16]](host.scratch_base(), logit_lease), logit_lease^)
        var best_val = Float32(-1e30)
        var best_idx = Int32(0)
        for j in range(0, C.VOCAB_SIZE, width):
            var v = logits.load_f32[width](j)
            for k in range(width):
                if v[k] > best_val:
                    best_val = v[k]
                    best_idx = Int32(j + k)
        logits^.release()
        lm_act_blk_scale_lease^.release()
        lm_act_i8_lease^.release()
        sample.wall_ns = Int(perf_counter_ns()) - t_forward0
        self.profile.record(sample)
        return best_idx


def main():
    print("gemma_4_moe_butterquant_tp_new: module")
