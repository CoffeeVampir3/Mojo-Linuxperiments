from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of, simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, DynamicView, StaticView,
    WeightDesc, DEFAULT_ALIGNMENT, LogitsView,
    TaskVisitor, PerRow, PerBlockAbsorbed,
)
from quant.source_format import Bf16Converter
from modeling.gemma4_common import (
    Gemma4BaseConfig, is_full_layer,
)
from modeling.modeling_common import (
    ArenaLayout, TensorRef, Repeated, SectionBuilder, align_up,
    LayerBuilder,
)
from kernels.kernel_ops import PoolFence
from kernels.reductions import ring_allreduce, small_allreduce, ring_broadcast, ring_allgather
from modeling.linear_borrow_pool import ScratchPool, ScratchLease, scratch_block_bytes

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
    timed_tp_parallel, ForwardSample, ForwardLogger,
)
from experimental3.init_weights import (
    PackColsumTask, make_pack_colsum_task, dispatch_pack_colsum_tasks,
    colsum_at, zero_pad_tail,
)
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
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
    var q_proj:    TensorRef[I8, Shape[C.Q_DIM_SLIDING, C.HIDDEN]]
    var k_proj:    TensorRef[I8, Shape[C.KV_DIM_SLIDING, C.HIDDEN]]
    var v_proj:    TensorRef[I8, Shape[C.KV_DIM_SLIDING, C.HIDDEN]]
    var q_proj_sc: TensorRef[F32, Shape[C.Q_DIM_SLIDING, 1]]
    var k_proj_sc: TensorRef[F32, Shape[C.KV_DIM_SLIDING, 1]]
    var v_proj_sc: TensorRef[F32, Shape[C.KV_DIM_SLIDING, 1]]
    var o_proj:    TensorRef[I8, Shape[C.HIDDEN, C.Q_DIM_SLIDING]]
    var o_proj_sc: TensorRef[F32, Shape[C.HIDDEN, 1]]
    var q_norm:    TensorRef[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var k_norm:    TensorRef[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var q_colsum:  TensorRef[F32, Shape[C.Q_DIM_SLIDING // Self.tp, 1]]
    var kv_colsum: TensorRef[F32, Shape[2 * (C.KV_DIM_SLIDING // Self.tp), 1]]
    var o_colsum:  TensorRef[F32, Shape[C.HIDDEN * (C.NUM_HEADS // Self.tp), 1]]


@fieldwise_init
struct FullAttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var q_proj:    TensorRef[I8, Shape[C.Q_DIM_FULL, C.HIDDEN]]
    var k_proj:    TensorRef[I8, Shape[C.KV_DIM_FULL, C.HIDDEN]]
    var q_proj_sc: TensorRef[F32, Shape[C.Q_DIM_FULL, 1]]
    var k_proj_sc: TensorRef[F32, Shape[C.KV_DIM_FULL, 1]]
    var o_proj:    TensorRef[I8, Shape[C.HIDDEN, C.Q_DIM_FULL]]
    var o_proj_sc: TensorRef[F32, Shape[C.HIDDEN, 1]]
    var q_norm:    TensorRef[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var k_norm:    TensorRef[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var q_colsum:  TensorRef[F32, Shape[C.Q_DIM_FULL // Self.tp, 1]]
    var k_colsum:  TensorRef[F32, Shape[C.KV_DIM_FULL, 1]]
    var o_colsum:  TensorRef[F32, Shape[C.HIDDEN * (C.NUM_HEADS // Self.tp), 1]]


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.tp]
    var input_norm:      TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm:  TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm:    TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm_2:  TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_1: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_2: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm:   TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var layer_scalar:    TensorRef[BF16, Shape[1, 1]]
    var gate_proj:       TensorRef[I8, Self.S.GateUp]
    var gate_proj_sc:    TensorRef[F32, Self.S.GateUpScale]
    var up_proj:         TensorRef[I8, Self.S.GateUp]
    var up_proj_sc:      TensorRef[F32, Self.S.GateUpScale]
    var down_proj:       TensorRef[I8, Self.S.Down]
    var down_proj_sc:    TensorRef[F32, Shape[C.HIDDEN, 1]]
    var router_scale:    TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var router_proj:     TensorRef[I8, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    var router_proj_sc:  TensorRef[F32, Shape[C.NUM_EXPERTS, 1]]
    var router_pes:      TensorRef[BF16, Shape[C.NUM_EXPERTS, 1]]
    var experts_gate_up:    TensorRef[I8, Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN]]
    var experts_gate_up_sc: TensorRef[F32, Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, 1]]
    var experts_down:       TensorRef[I8, Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE]]
    var experts_down_sc:    TensorRef[F32, Shape[C.NUM_EXPERTS * C.HIDDEN, 1]]
    var gu_colsum:           TensorRef[F32, Shape[Self.S.DENSE_INT_LOCAL * 2, 1]]
    var down_colsum:         TensorRef[F32, Shape[C.HIDDEN * Self.S.DENSE_DOWN_NUM_BLK, 1]]
    var router_colsum:       TensorRef[F32, Shape[C.NUM_EXPERTS, 1]]
    var experts_gu_colsum:   TensorRef[F32, Shape[(C.NUM_EXPERTS // Self.tp) * C.MOE_GATE_UP_FUSED, 1]]
    var experts_down_colsum: TensorRef[F32, Shape[(C.NUM_EXPERTS // Self.tp) * C.HIDDEN * MOE_NUM_BLOCKS, 1]]


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
    var x_main:     TensorRef[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: TensorRef[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: TensorRef[F32, Shape[C.MAX_SEQ_LEN, Self.half]]
    var sin: TensorRef[F32, Shape[C.MAX_SEQ_LEN, Self.half]]


@fieldwise_init
struct HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var embed:      TensorRef[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var embed_sc:   TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]
    var embed_colsum:    TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]
    var sqrt_gamma:      TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var inv_sqrt_gamma:  TensorRef[F32, Shape[C.HIDDEN, 1]]


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct Gemma4Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs[Self.tp]]
    var full: Repeated[FullLayerRefs[Self.tp]]

    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.HEAD_DIM_SLIDING // 2]
    var full_rope: RopeSlots[64]

    var sliding_cache_off: Int
    var sliding_cache_stride: Int
    var full_cache_off: Int
    var full_cache_stride: Int

    var host: HostSlots

    @always_inline
    def bind(self, base: Int) -> Self:
        var t = self
        t.arena = t.arena.bind(base)
        return t


# =============================================================================
# Emit
# =============================================================================


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    comptime GU = C.MOE_GATE_UP_FUSED
    comptime MI = C.MOE_INTERMEDIATE
    comptime S = Gemma4Shapes[tp]
    comptime experts_local = NE // tp

    # Order is not arbitrary (contiguousness assumed by some ops)
    var input_norm = b.bfs[Shape[H, 1]](e, "input_layernorm.weight")
    var post_attn_norm = b.bfs[Shape[H, 1]](e, "post_attention_layernorm.weight")
    var pre_ffn_norm = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm.weight")
    var pre_ffn_norm_2 = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm_2.weight")
    var post_ffn_norm_1 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_1.weight")
    var post_ffn_norm_2 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_2.weight")
    var post_ffn_norm = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm.weight")
    var layer_scalar = b.bfs[Shape[1, 1]](e, "layer_scalar")

    var gate_proj = b.qs[S.GateUp](e, "mlp.gate_proj.weight")
    var up_proj = b.qs[S.GateUp](e, "mlp.up_proj.weight")
    var gate_proj_sc = b.fs[S.GateUpScale](e, "mlp.gate_proj.weight_scale")
    var up_proj_sc = b.fs[S.GateUpScale](e, "mlp.up_proj.weight_scale")

    var down_proj = b.qs[S.Down](e, "mlp.down_proj.weight")
    var down_proj_sc = b.fs[Shape[H, 1]](e, "mlp.down_proj.weight_scale")
    var router_scale = b.bfs[Shape[H, 1]](e, "router.scale")
    var router_proj = b.qs[Shape[NE, H]](e, "router.proj.weight")
    var router_proj_sc = b.fs[Shape[NE, 1]](e, "router.proj.weight_scale")
    var router_pes = b.bfs[Shape[NE, 1]](e, "router.per_expert_scale")
    var experts_gate_up = TensorRef[I8, Shape[NE * GU, H]](
        b.qs[Shape[NE * GU, H, shard_n=True, tp=tp]](e, "experts.gate_up_proj").offset)
    var experts_gate_up_sc = TensorRef[F32, Shape[NE * GU, 1]](
        b.fs[Shape[NE * GU, 1, shard_n=True, tp=tp]](e, "experts.gate_up_proj_scale").offset)
    var experts_down = TensorRef[I8, Shape[NE * H, MI]](
        b.qs[Shape[NE * H, MI, shard_n=True, tp=tp]](e, "experts.down_proj").offset)
    var experts_down_sc = TensorRef[F32, Shape[NE * H, 1]](
        b.fs[Shape[NE * H, 1, shard_n=True, tp=tp]](e, "experts.down_proj_scale").offset)
    var gu_colsum = b.colsum_slot[F32, Shape[S.DENSE_INT_LOCAL * 2, 1]]()
    var down_colsum = b.colsum_slot[F32, Shape[H * S.DENSE_DOWN_NUM_BLK, 1]]()
    var router_colsum = b.colsum_slot[F32, Shape[NE, 1]]()
    var experts_gu_colsum = b.colsum_slot[F32, Shape[experts_local * GU, 1]]()
    var experts_down_colsum = b.colsum_slot[F32, Shape[experts_local * H * MOE_NUM_BLOCKS, 1]]()

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
    comptime H = C.HIDDEN
    comptime HDS = C.HEAD_DIM_SLIDING
    var attn = SlidingAttnRefs[tp](
        q_proj    = TensorRef[I8, Shape[C.Q_DIM_SLIDING, H]](b.qs[Shape[C.Q_DIM_SLIDING, H, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight").offset),
        k_proj    = TensorRef[I8, Shape[C.KV_DIM_SLIDING, H]](b.qs[Shape[C.KV_DIM_SLIDING, H, shard_n=True, tp=tp]](e, "self_attn.k_proj.weight").offset),
        v_proj    = TensorRef[I8, Shape[C.KV_DIM_SLIDING, H]](b.qs[Shape[C.KV_DIM_SLIDING, H, shard_n=True, tp=tp]](e, "self_attn.v_proj.weight").offset),
        q_proj_sc = TensorRef[F32, Shape[C.Q_DIM_SLIDING, 1]](b.fs[Shape[C.Q_DIM_SLIDING, 1, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight_scale").offset),
        k_proj_sc = TensorRef[F32, Shape[C.KV_DIM_SLIDING, 1]](b.fs[Shape[C.KV_DIM_SLIDING, 1, shard_n=True, tp=tp]](e, "self_attn.k_proj.weight_scale").offset),
        v_proj_sc = TensorRef[F32, Shape[C.KV_DIM_SLIDING, 1]](b.fs[Shape[C.KV_DIM_SLIDING, 1, shard_n=True, tp=tp]](e, "self_attn.v_proj.weight_scale").offset),
        o_proj    = TensorRef[I8, Shape[H, C.Q_DIM_SLIDING]](b.qs[Shape[H, C.Q_DIM_SLIDING, shard_m=True, tp=tp]](e, "self_attn.o_proj.weight").offset),
        o_proj_sc = b.fs[Shape[H, 1]](e, "self_attn.o_proj.weight_scale"),
        q_norm    = b.bfs[Shape[HDS, 1]](e, "self_attn.q_norm.weight"),
        k_norm    = b.bfs[Shape[HDS, 1]](e, "self_attn.k_norm.weight"),
        q_colsum  = b.colsum_slot[F32, Shape[C.Q_DIM_SLIDING // tp, 1]](),
        kv_colsum = b.colsum_slot[F32, Shape[2 * (C.KV_DIM_SLIDING // tp), 1]](),
        o_colsum  = b.colsum_slot[F32, Shape[H * (C.NUM_HEADS // tp), 1]](),
    )
    return (SlidingLayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def emit_full[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime H = C.HIDDEN
    comptime HDF = C.HEAD_DIM_FULL
    var attn = FullAttnRefs[tp](
        q_proj    = TensorRef[I8, Shape[C.Q_DIM_FULL, H]](b.qs[Shape[C.Q_DIM_FULL, H, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight").offset),
        k_proj    = b.qs[Shape[C.KV_DIM_FULL, H]](e, "self_attn.k_proj.weight"),
        q_proj_sc = TensorRef[F32, Shape[C.Q_DIM_FULL, 1]](b.fs[Shape[C.Q_DIM_FULL, 1, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight_scale").offset),
        k_proj_sc = b.fs[Shape[C.KV_DIM_FULL, 1]](e, "self_attn.k_proj.weight_scale"),
        o_proj    = TensorRef[I8, Shape[H, C.Q_DIM_FULL]](b.qs[Shape[H, C.Q_DIM_FULL, shard_m=True, tp=tp]](e, "self_attn.o_proj.weight").offset),
        o_proj_sc = b.fs[Shape[H, 1]](e, "self_attn.o_proj.weight_scale"),
        q_norm    = b.bfs[Shape[HDF, 1]](e, "self_attn.q_norm.weight"),
        k_norm    = b.bfs[Shape[HDF, 1]](e, "self_attn.k_norm.weight"),
        q_colsum  = b.colsum_slot[F32, Shape[C.Q_DIM_FULL // tp, 1]](),
        k_colsum  = b.colsum_slot[F32, Shape[C.KV_DIM_FULL, 1]](),
        o_colsum  = b.colsum_slot[F32, Shape[H * (C.NUM_HEADS // tp), 1]](),
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
    comptime int32_bytes = size_of[Int32]()
    comptime S = Gemma4Shapes[tp]
    comptime gateup_col_bytes = S.GateUp.col_bytes[I8]()

    comptime persistent = (
        scratch_block_bytes[f32]()
        + scratch_block_bytes[S.DENSE_DOWN_NUM_BLK * f32]()
    )

    comptime sliding_attn_peak = persistent + (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[f32]()
        + scratch_block_bytes[S.SlidingQ.N * bf16]()
        + scratch_block_bytes[2 * (C.KV_DIM_SLIDING // tp) * bf16]()
        + scratch_block_bytes[S.SlidingQ.N * i8]()
        + scratch_block_bytes[(S.SlidingQ.N // C.HEAD_DIM_SLIDING) * f32]()
    )

    comptime full_q_bf16 = S.FullQ.N * bf16
    comptime full_k_bf16 = C.KV_DIM_FULL * bf16
    comptime full_attn_phase1 = persistent + (
        scratch_block_bytes[full_q_bf16]()
        + scratch_block_bytes[full_k_bf16]()
        + scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[f32]()
    )
    comptime full_attn_cp_extra = (
        scratch_block_bytes[C.Q_DIM_FULL * bf16]()
        + scratch_block_bytes[C.NUM_HEADS * f32]()
        + scratch_block_bytes[C.NUM_HEADS * f32]()
        + scratch_block_bytes[C.NUM_HEADS * C.HEAD_DIM_FULL * f32]()
    )
    comptime full_attn_phase2 = persistent + (
        scratch_block_bytes[full_q_bf16]()
        + scratch_block_bytes[full_k_bf16]()
        + scratch_block_bytes[S.FullQ.N * i8]()
        + scratch_block_bytes[(S.FullQ.N // C.HEAD_DIM_FULL) * f32]()
        + scratch_block_bytes[S.FULL_HPG * C.HEAD_DIM_FULL * i8]()
        + scratch_block_bytes[S.FULL_HPG * f32]()
        + scratch_block_bytes[S.FULL_HPG * f32]()
        + scratch_block_bytes[C.FULL_ATTN_MAX_CHUNKS * S.FULL_HPG * (2 + C.HEAD_DIM_FULL) * f32]()
        + full_attn_cp_extra
    )
    comptime full_attn_peak = (
        full_attn_phase1 if full_attn_phase1 > full_attn_phase2 else full_attn_phase2
    )

    comptime ffn_peak = persistent + (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[f32]()
        + scratch_block_bytes[C.NUM_EXPERTS * bf16]()
        + scratch_block_bytes[topk_bytes]()
        + scratch_block_bytes[C.TOP_K * C.MOE_INTERMEDIATE * i8]()
        + scratch_block_bytes[C.TOP_K * MOE_NUM_BLOCKS * f32]()
        + scratch_block_bytes[C.TOP_K * C.HIDDEN * bf16]()
        + scratch_block_bytes[int32_bytes]()
        + scratch_block_bytes[gateup_col_bytes]()
        + scratch_block_bytes[C.HIDDEN * bf16]()
        + scratch_block_bytes[C.HIDDEN * bf16]()
    )

    comptime layer_peak = (
        sliding_attn_peak if sliding_attn_peak > full_attn_peak else full_attn_peak
    )
    comptime decode_peak = ffn_peak if ffn_peak > layer_peak else layer_peak

    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    comptime lm_head_peak = (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[vocab_num_blocks * f32]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[C.VOCAB_SIZE * bf16]()
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

    # State block — pre-seed cursor with `distributed` so every slot carries
    # an arena-absolute offset.
    var state = SectionBuilder()
    state.cursor = distributed
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
    var host_off = align_up(state.bytes())
    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var final_norm_off = hb.bfs[Shape[C.HIDDEN, 1]](descs, "model.language_model.norm.weight", target_rank=HOST_RANK)
    var embed_off = hb.qs[Shape[C.VOCAB_SIZE, C.HIDDEN]](descs, "model.language_model.embed_tokens.weight", target_rank=HOST_RANK)
    var embed_sc_off = hb.fs[Shape[C.VOCAB_SIZE, vocab_num_blocks]](descs, "model.language_model.embed_tokens.weight_scale", target_rank=HOST_RANK)
    var embed_colsum = hb.colsum_slot[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]]()
    var sqrt_gamma = hb.colsum_slot[BF16, Shape[C.HIDDEN, 1]]()
    var inv_sqrt_gamma = hb.colsum_slot[F32, Shape[C.HIDDEN, 1]]()
    var host_bytes = hb.cursor

    var host = HostSlots(
        final_norm=final_norm_off,
        embed=embed_off,
        embed_sc=embed_sc_off,
        embed_colsum=embed_colsum,
        sqrt_gamma=sqrt_gamma,
        inv_sqrt_gamma=inv_sqrt_gamma)

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state.bytes() - distributed,
        host_bytes=host_bytes,
        scratch_off=scratch_off,
        scratch_capacity=scratch_cap)

    var topo = Gemma4Topology[tp](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[tp]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[tp]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        sliding_cache_off=sliding_cache_off,
        sliding_cache_stride=sliding_cache_stride,
        full_cache_off=full_cache_off,
        full_cache_stride=full_cache_stride,
        host=host)
    return Gemma4LoadPlan[tp](topo, descs^)


# =============================================================================
# Init helpers — colsums + VNNI packing for shared body weights
# =============================================================================


def init_layer_body_padding[tp: Int](arena_base: Int, layer_off: Int, body: BodyRefs[tp]):
    zero_pad_tail(arena_base + layer_off, body.gate_proj)
    zero_pad_tail(arena_base + layer_off, body.up_proj)


def append_layer_body_tasks[tp: Int](
    arena_base: Int, layer_off: Int, body: BodyRefs[tp],
    mut tasks: List[PackColsumTask],
):
    comptime experts_local = C.NUM_EXPERTS // tp
    comptime S = Gemma4Shapes[tp]
    var lb = arena_base + layer_off

    var gu_slab = TensorRef[I8, Shape[S.DENSE_INT_LOCAL * 2, C.HIDDEN]](
        offset=body.gate_proj.offset)
    tasks.append(make_pack_colsum_task(lb, gu_slab, body.gu_colsum))

    tasks.append(make_pack_colsum_task(
        lb, body.down_proj, body.down_colsum,
        block_cols=FWHT_BLK_DENSE_DOWN, colsum_row_major=False))

    tasks.append(make_pack_colsum_task(lb, body.router_proj, body.router_colsum))

    for e in range(experts_local):
        var expert_gu = TensorRef[I8, Shape[C.MOE_GATE_UP_FUSED, C.HIDDEN]](
            offset=body.experts_gate_up.offset + e * C.MOE_GATE_UP_FUSED * C.HIDDEN)
        var expert_gu_colsum = TensorRef[F32, Shape[C.MOE_GATE_UP_FUSED, 1]](
            offset=body.experts_gu_colsum.offset + e * C.MOE_GATE_UP_FUSED * 4)
        tasks.append(make_pack_colsum_task(lb, expert_gu, expert_gu_colsum))

        var expert_down = TensorRef[I8, Shape[C.HIDDEN, C.MOE_INTERMEDIATE]](
            offset=body.experts_down.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE)
        var expert_down_colsum = TensorRef[F32, Shape[C.HIDDEN * MOE_NUM_BLOCKS, 1]](
            offset=body.experts_down_colsum.offset + e * C.HIDDEN * MOE_NUM_BLOCKS * 4)
        tasks.append(make_pack_colsum_task(
            lb, expert_down, expert_down_colsum,
            block_cols=FWHT_BLK, colsum_row_major=False))


# =============================================================================
# Multi-rank dispatch
# =============================================================================




# =============================================================================
# Model struct
# =============================================================================


struct Gemma4ButterQuant[tp: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[Self.Pool]
    var scratch: ScratchPool
    var topos: InlineArray[Gemma4Topology[Self.tp], Self.tp]
    var profile: ForwardLogger

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[Self.Pool],
        var sc: ScratchPool,
        topos: InlineArray[Gemma4Topology[Self.tp], Self.tp],
    ):
        self.arenas = arenas^
        self.main_pools = mp^
        self.scratch = sc^
        self.topos = topos
        self.profile = ForwardLogger()


    def x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            var topo = self.topos[r]
            ptrs[r] = topo.activations.x_main.addr(topo.arena.base)
        return ptrs^

    def x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            var topo = self.topos[r]
            ptrs[r] = topo.activations.x_residual.addr(topo.arena.base)
        return ptrs^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topos[0].arena.scratch_base())

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
        comptime Q_LOCAL_FULL = C.Q_DIM_FULL // Self.tp
        comptime Q_LOCAL_SLIDING = C.Q_DIM_SLIDING // Self.tp
        comptime KV_N_SLIDING = 2 * (C.KV_DIM_SLIDING // Self.tp)

        for rank in range(Self.tp):
            var topo = self.topos[rank]
            var base = topo.arena.base
            var numa_node = self.arenas[rank].node

            init_sliding_rope_tables(
                topo.sliding_rope.cos.bound(base),
                topo.sliding_rope.sin.bound(base))
            init_full_rope_tables(
                topo.full_rope.cos.bound(base),
                topo.full_rope.sin.bound(base))

            var sliding_idx = 0
            var full_idx = 0
            for i in range(C.NUM_LAYERS):
                if is_full_layer(i):
                    var lb = topo.full.base(base, full_idx)
                    init_layer_body_padding[Self.tp](base, lb - base, topo.full.proto.body)
                    full_idx += 1
                else:
                    var lb = topo.sliding.base(base, sliding_idx)
                    init_layer_body_padding[Self.tp](base, lb - base, topo.sliding.proto.body)
                    sliding_idx += 1

            var tasks = List[PackColsumTask]()
            sliding_idx = 0
            full_idx = 0
            for i in range(C.NUM_LAYERS):
                if is_full_layer(i):
                    var lb = topo.full.base(base, full_idx)
                    var fl = topo.full.proto
                    var q_slot = TensorRef[I8, Shape[Q_LOCAL_FULL, C.HIDDEN]](
                        offset=fl.attn.q_proj.offset)
                    tasks.append(make_pack_colsum_task(lb, q_slot, fl.attn.q_colsum))
                    var k_slot = TensorRef[I8, Shape[C.KV_DIM_FULL, C.HIDDEN]](
                        offset=fl.attn.k_proj.offset)
                    tasks.append(make_pack_colsum_task(lb, k_slot, fl.attn.k_colsum))
                    var o_slot = TensorRef[I8, Shape[C.HIDDEN, Q_LOCAL_FULL]](
                        offset=fl.attn.o_proj.offset)
                    tasks.append(make_pack_colsum_task(
                        lb, o_slot, fl.attn.o_colsum,
                        block_cols=C.HEAD_DIM_FULL, colsum_row_major=False))
                    append_layer_body_tasks[Self.tp](base, lb - base, fl.body, tasks)
                    full_idx += 1
                else:
                    var lb = topo.sliding.base(base, sliding_idx)
                    var sl = topo.sliding.proto
                    var q_slot = TensorRef[I8, Shape[Q_LOCAL_SLIDING, C.HIDDEN]](
                        offset=sl.attn.q_proj.offset)
                    tasks.append(make_pack_colsum_task(lb, q_slot, sl.attn.q_colsum))
                    var kv_slab = TensorRef[I8, Shape[KV_N_SLIDING, C.HIDDEN]](
                        offset=sl.attn.k_proj.offset)
                    tasks.append(make_pack_colsum_task(lb, kv_slab, sl.attn.kv_colsum))
                    var o_slot = TensorRef[I8, Shape[C.HIDDEN, Q_LOCAL_SLIDING]](
                        offset=sl.attn.o_proj.offset)
                    tasks.append(make_pack_colsum_task(
                        lb, o_slot, sl.attn.o_colsum,
                        block_cols=C.HEAD_DIM_SLIDING, colsum_row_major=False))
                    append_layer_body_tasks[Self.tp](base, lb - base, sl.body, tasks)
                    sliding_idx += 1

            dispatch_pack_colsum_tasks(self.main_pools[rank], numa_node, tasks)

            if rank == HOST_RANK:
                colsum_at(base,
                    topo.host.embed, topo.host.embed_colsum,
                    block_cols=LM_HEAD_FWHT_BLK, colsum_row_major=True)
                var fn_gamma = topo.host.final_norm.bound(base).as_ptr()
                compute_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    topo.host.sqrt_gamma.bound(base).as_ptr())
                compute_inv_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    topo.host.inv_sqrt_gamma.bound(base).as_ptr())

            comptime inv_sqrt_h = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
            sliding_idx = 0
            full_idx = 0
            for i in range(C.NUM_LAYERS):
                var sc_ptr: UnsafePointer[Float32, MutAnyOrigin]
                if is_full_layer(i):
                    var lb = topo.full.base(base, full_idx)
                    sc_ptr = topo.full.proto.body.router_proj_sc.bound(lb).as_ptr()
                    full_idx += 1
                else:
                    var lb = topo.sliding.base(base, sliding_idx)
                    sc_ptr = topo.sliding.proto.body.router_proj_sc.bound(lb).as_ptr()
                    sliding_idx += 1
                for n in range(C.NUM_EXPERTS):
                    sc_ptr[n] *= inv_sqrt_h

        print("state initialized")

    @staticmethod
    def load(
        dir_path: Path,
        numa: NumaInfo,
        numa_topo: NumaTopology,
        var main_pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
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

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = plan.topology.arena.host_arena_bytes() if rank == HOST_RANK else plan.topology.arena.arena_bytes()
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
            _ = arenas[rank].prefault(plan.topology.arena.distributed_bytes, plan.topology.arena.state_bytes)

        var topos = InlineArray[Gemma4Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))

        var scratch = ScratchPool(plan.topology.arena.scratch_capacity)
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
        comptime XShape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
        comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_HEAD_FWHT_BLK

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(pos)
        var topos = self.topos
        var host = topos[0]
        # --- Embed ---
        var t_embed0 = Int(perf_counter_ns())
        var embed_fence = embed_lookup_blocked[fwht_blk = LM_HEAD_FWHT_BLK](
            host.host.embed.bound(host.arena.base),
            host.host.embed_sc.bound(host.arena.base),
            host.arena.base + host.host.inv_sqrt_gamma.offset,
            tokens_ptr,
            host.activations.x_main.bound_dyn(host.arena.base, seq_len), Float32(EMBED_SCALE),
            self.main_pools[0])
        var t_embed1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("embed"), finish_single_pool_fence(t_embed0, t_embed1, embed_fence^))

        var t_bcast0 = Int(perf_counter_ns())
        ring_broadcast[BF16, XShape, Self.tp](
            host.activations.x_main.addr(host.arena.base), self.x_main_ptrs(seq_len), seq_len, self.main_pools)
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
                def do_attn_quantize[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.sliding.base(topo.arena.base, sliding_idx)
                    var sl = topo.sliding.proto
                    var sb = topo.arena.scratch_base()
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                        topo.activations.x_main.bound_dyn(topo.arena.base, seq_len), sl.body.input_norm.bound(lb),
                        attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        attn_work_lease.view[F32, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        EPS, pool)
                sample.add(self.profile.phase("local_attn_quantize"), timed_tp_parallel[Self.tp,do_attn_quantize](topos, self.main_pools))

                var q_lease = self.scratch.borrow[Scalar[DType.bfloat16], Q_DIM_LOCAL_SLIDING]()
                var kv_lease = self.scratch.borrow[Scalar[DType.bfloat16], 2 * KV_DIM_LOCAL_SLIDING]()

                @parameter
                def do_q_gemv[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.sliding.base(topo.arena.base, sliding_idx)
                    var sl = topo.sliding.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv[Q_DIM_LOCAL_SLIDING, C.HIDDEN](
                        attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        sl.attn.q_proj.bound(lb),
                        sl.attn.q_colsum.bound(lb),
                        sl.attn.q_proj_sc.bound(lb),
                        q_lease.view[BF16, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_SLIDING]](sb, seq_len),
                        attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        pool)
                sample.add(self.profile.phase("local_attn_proj"), timed_tp_parallel[Self.tp,do_q_gemv](topos, self.main_pools))

                @parameter
                def do_kv_gemv[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.sliding.base(topo.arena.base, sliding_idx)
                    var sl = topo.sliding.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv[2 * KV_DIM_LOCAL_SLIDING, C.HIDDEN](
                        attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        sl.attn.k_proj.bound(lb),
                        sl.attn.kv_colsum.bound(lb),
                        sl.attn.k_proj_sc.bound(lb),
                        kv_lease.view[BF16, Shape[C.MAX_SEQ_LEN, 2 * KV_DIM_LOCAL_SLIDING]](sb, seq_len),
                        attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        pool)
                sample.add(self.profile.phase("local_attn_proj"), timed_tp_parallel[Self.tp,do_kv_gemv](topos, self.main_pools))

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_SLIDING]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, Q_DIM_LOCAL_SLIDING // C.HEAD_DIM_SLIDING]()

                @parameter
                def do_sliding_attn[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
                    comptime NKV = C.NUM_KV_HEADS_SLIDING // Self.tp
                    var lb = topo.sliding.base(topo.arena.base, sliding_idx)
                    var sl = topo.sliding.proto
                    var sb = topo.arena.scratch_base()
                    var cos_row = topo.sliding_rope.cos.bound_row(topo.arena.base, pos)
                    var sin_row = topo.sliding_rope.sin.bound_row(topo.arena.base, pos)
                    return sliding_attn_dispatch[
                        C.HEAD_DIM_SLIDING, HPG, C.SLIDING_WINDOW, NKV, C.NUM_HEADS // Self.tp](
                        q_lease.view[BF16, Shape[1, Q_DIM_LOCAL_SLIDING]](sb, 1),
                        kv_lease.view[BF16, Shape[1, KV_DIM_LOCAL_SLIDING]](sb, 1).any(),
                        kv_lease.view[BF16, Shape[1, KV_DIM_LOCAL_SLIDING]](sb, 1, element_offset=KV_DIM_LOCAL_SLIDING).any(),
                        sl.attn.q_norm.bound(lb), sl.attn.k_norm.bound(lb),
                        cos_row, sin_row,
                        topo.arena.base + topo.sliding_cache_off + sliding_idx * topo.sliding_cache_stride,
                        pos % C.SLIDING_WINDOW, min(pos + 1, C.SLIDING_WINDOW),
                        attn_qi_lease.view[I8, Shape[1, Q_DIM_LOCAL_SLIDING]](sb, 1),
                        attn_head_sc_lease.view[F32, Shape[1, Q_DIM_LOCAL_SLIDING // C.HEAD_DIM_SLIDING]](sb, 1),
                        EPS, pool)
                sample.add(self.profile.phase("local_attention"), timed_tp_parallel[Self.tp,do_sliding_attn](topos, self.main_pools))

                @parameter
                def do_o_proj[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.sliding.base(topo.arena.base, sliding_idx)
                    var sl = topo.sliding.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_SLIDING, C.HEAD_DIM_SLIDING](
                        attn_qi_lease.view[I8, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_SLIDING]](sb, seq_len),
                        sl.attn.o_proj.bound(lb),
                        attn_head_sc_lease.view[F32, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_SLIDING // C.HEAD_DIM_SLIDING]](sb, seq_len),
                        sl.attn.o_proj_sc.bound(lb),
                        sl.attn.o_colsum.bound(lb),
                        topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len),
                        pool)
                sample.add(self.profile.phase("local_o_proj"), timed_tp_parallel[Self.tp,do_o_proj](topos, self.main_pools))
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
                def do_full_attn_quantize[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.full.base(topo.arena.base, full_idx)
                    var fl = topo.full.proto
                    var sb = topo.arena.scratch_base()
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                        topo.activations.x_main.bound_dyn(topo.arena.base, seq_len), fl.body.input_norm.bound(lb),
                        full_attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        full_attn_work_lease.view[F32, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        full_attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        EPS, pool)
                sample.add(self.profile.phase("global_attn_quantize"), timed_tp_parallel[Self.tp,do_full_attn_quantize](topos, self.main_pools))

                @parameter
                def do_full_q_gemv[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.full.base(topo.arena.base, full_idx)
                    var fl = topo.full.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv[Q_DIM_LOCAL_FULL, C.HIDDEN](
                        full_attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        fl.attn.q_proj.bound(lb),
                        fl.attn.q_colsum.bound(lb),
                        fl.attn.q_proj_sc.bound(lb),
                        full_q_lease.view[BF16, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_FULL]](sb, seq_len),
                        full_attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        pool)
                sample.add(self.profile.phase("global_attn_proj"), timed_tp_parallel[Self.tp,do_full_q_gemv](topos, self.main_pools))

                @parameter
                def do_full_k_gemv[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.full.base(topo.arena.base, full_idx)
                    var fl = topo.full.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv[C.KV_DIM_FULL, C.HIDDEN](
                        full_attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                        fl.attn.k_proj.bound(lb),
                        fl.attn.k_colsum.bound(lb),
                        fl.attn.k_proj_sc.bound(lb),
                        full_k_lease.view[BF16, Shape[C.MAX_SEQ_LEN, C.KV_DIM_FULL]](sb, seq_len),
                        full_attn_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                        pool)
                sample.add(self.profile.phase("global_attn_proj"), timed_tp_parallel[Self.tp,do_full_k_gemv](topos, self.main_pools))
                full_attn_scale_lease^.release()
                full_attn_work_lease^.release()
                full_attn_i8_lease^.release()

                # Q all-gather: each rank pulls all Q shards locally
                var full_q_all_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.Q_DIM_FULL]()
                var q_shard_ptrs = InlineArray[Int, Self.tp](fill=0)
                var q_dst_ptrs = InlineArray[Int, Self.tp](fill=0)
                for r in range(Self.tp):
                    q_shard_ptrs[r] = topos[r].arena.scratch_base() + full_q_lease.offset
                    q_dst_ptrs[r] = topos[r].arena.scratch_base() + full_q_all_lease.offset
                ring_allgather[Self.tp](
                    q_shard_ptrs, q_dst_ptrs, Q_DIM_LOCAL_FULL * 2, self.main_pools)

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_FULL]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, HEADS_PER_RANK]()
                var q_i8_prep_lease = self.scratch.borrow[Scalar[DType.int8], FULL_HPG * C.HEAD_DIM_FULL]()
                var qi_biases_lease = self.scratch.borrow[Float32, FULL_HPG]()
                var q_scales_lease = self.scratch.borrow[Float32, FULL_HPG]()
                comptime PARTIAL_F32S = C.FULL_ATTN_MAX_CHUNKS * FULL_HPG * (2 + C.HEAD_DIM_FULL)
                var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()
                var cp_m_lease = self.scratch.borrow[Float32, C.NUM_HEADS]()
                var cp_l_lease = self.scratch.borrow[Float32, C.NUM_HEADS]()
                var cp_v_lease = self.scratch.borrow[Float32, C.NUM_HEADS * C.HEAD_DIM_FULL]()

                for kv in range(C.NUM_KV_HEADS_FULL):
                    @parameter
                    def do_cp_attn_prep[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                        var lb = topo.full.base(topo.arena.base, full_idx)
                        var fl = topo.full.proto
                        var sb = topo.arena.scratch_base()
                        var lp = cp_local_pos(pos, Self.tp)
                        var is_owner = cp_owning_rank(pos, Self.tp) == rank
                        comptime Q_GROUP_ELEMS = FULL_HPG * C.HEAD_DIM_FULL
                        comptime K_HEAD_ELEMS = C.HEAD_DIM_FULL
                        var cos_row = topo.full_rope.cos.bound_row(topo.arena.base, pos)
                        var sin_row = topo.full_rope.sin.bound_row(topo.arena.base, pos)
                        return cp_attn_prep_dispatch[
                            C.HEAD_DIM_FULL, ROPE_DIMS_FULL, FULL_HPG,
                            LOCAL_MAX_SEQ_FULL, C.NUM_KV_HEADS_FULL, C.NUM_HEADS](
                            full_q_all_lease.view[BF16, Shape[1, Q_GROUP_ELEMS]](
                                sb, 1, element_offset=kv * Q_GROUP_ELEMS),
                            full_k_lease.view[BF16, Shape[1, K_HEAD_ELEMS]](
                                sb, 1, element_offset=kv * K_HEAD_ELEMS),
                            fl.attn.q_norm.bound(lb), fl.attn.k_norm.bound(lb),
                            cos_row, sin_row,
                            topo.arena.base + topo.full_cache_off + full_idx * topo.full_cache_stride, lp, kv,
                            EPS, is_owner,
                            q_i8_prep_lease.view[I8, Shape[1, Q_GROUP_ELEMS]](sb, 1),
                            qi_biases_lease.view[F32, Shape[1, FULL_HPG]](sb, 1),
                            q_scales_lease.view[F32, Shape[1, FULL_HPG]](sb, 1),
                            pool)
                    sample.add(self.profile.phase("global_attention"), timed_tp_parallel[Self.tp,do_cp_attn_prep](topos, self.main_pools))

                    @parameter
                    def do_cp_chunk_attn[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                        var local_ctx = cp_local_context_len(pos + 1, rank, Self.tp)
                        return cp_chunked_attn_dispatch[
                            C.HEAD_DIM_FULL, LOCAL_MAX_SEQ_FULL,
                            C.NUM_KV_HEADS_FULL, C.NUM_HEADS, FULL_HPG,
                            C.FULL_ATTN_MAX_CHUNKS](
                            topo.arena.scratch_base() + q_i8_prep_lease.offset,
                            topo.arena.scratch_base() + qi_biases_lease.offset,
                            topo.arena.scratch_base() + q_scales_lease.offset,
                            topo.arena.base + topo.full_cache_off + full_idx * topo.full_cache_stride, kv,
                            local_ctx, Int(pool.get_capacity()),
                            topo.arena.scratch_base() + partial_lease.offset,
                            pool)
                    sample.add(self.profile.phase("global_attention"), timed_tp_parallel[Self.tp,do_cp_chunk_attn](topos, self.main_pools))

                    @parameter
                    def do_merge_chunks[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                        var local_ctx = cp_local_context_len(pos + 1, rank, Self.tp)
                        var num_pg = (local_ctx + CACHE_WIDTH - 1) // CACHE_WIDTH
                        var nc = min(Int(pool.get_capacity()), C.FULL_ATTN_MAX_CHUNKS)
                        if nc > num_pg:
                            nc = num_pg
                        return merge_local_chunks_dispatch[
                            C.HEAD_DIM_FULL, FULL_HPG, C.FULL_ATTN_MAX_CHUNKS](
                            topo.arena.scratch_base() + partial_lease.offset, nc,
                            topo.arena.scratch_base() + cp_m_lease.offset + kv * FULL_HPG * 4,
                            topo.arena.scratch_base() + cp_l_lease.offset + kv * FULL_HPG * 4,
                            topo.arena.scratch_base() + cp_v_lease.offset + kv * FULL_HPG * C.HEAD_DIM_FULL * 4,
                            pool)
                    sample.add(self.profile.phase("global_attention"), timed_tp_parallel[Self.tp,do_merge_chunks](topos, self.main_pools))

                # Cross-rank gather + quantize: each rank's pool merges V
                # from all ranks (remote reads) and writes qi/scales locally
                var all_m_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                var all_l_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                var all_v_addrs = InlineArray[Int, MAX_CP_RANKS](fill=0)
                for r in range(Self.tp):
                    all_m_addrs[r] = topos[r].arena.scratch_base() + cp_m_lease.offset
                    all_l_addrs[r] = topos[r].arena.scratch_base() + cp_l_lease.offset
                    all_v_addrs[r] = topos[r].arena.scratch_base() + cp_v_lease.offset

                @parameter
                def do_cp_gather[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    return cp_gather_dispatch[C.HEAD_DIM_FULL, C.NUM_HEADS, Self.tp](
                        rank, all_m_addrs, all_l_addrs, all_v_addrs,
                        topo.arena.scratch_base() + attn_qi_lease.offset,
                        topo.arena.scratch_base() + attn_head_sc_lease.offset,
                        rank * HEADS_PER_RANK, HEADS_PER_RANK, pool)
                sample.add(self.profile.phase("global_attention"), timed_tp_parallel[Self.tp,do_cp_gather](topos, self.main_pools))

                cp_v_lease^.release()
                cp_l_lease^.release()
                cp_m_lease^.release()
                partial_lease^.release()
                q_scales_lease^.release()
                qi_biases_lease^.release()
                q_i8_prep_lease^.release()

                @parameter
                def do_full_o_proj[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var lb = topo.full.base(topo.arena.base, full_idx)
                    var fl = topo.full.proto
                    var sb = topo.arena.scratch_base()
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_FULL, C.HEAD_DIM_FULL](
                        attn_qi_lease.view[I8, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_FULL]](sb, seq_len),
                        fl.attn.o_proj.bound(lb),
                        attn_head_sc_lease.view[F32, Shape[C.MAX_SEQ_LEN, Q_DIM_LOCAL_FULL // C.HEAD_DIM_FULL]](sb, seq_len),
                        fl.attn.o_proj_sc.bound(lb),
                        fl.attn.o_colsum.bound(lb),
                        topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len),
                        pool)
                sample.add(self.profile.phase("global_o_proj"), timed_tp_parallel[Self.tp,do_full_o_proj](topos, self.main_pools))

                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                full_q_all_lease^.release()
                full_k_lease^.release()
                full_q_lease^.release()

            # Allreduce + post-attn norm
            var t_attn_reduce0 = Int(perf_counter_ns())
            small_allreduce[BF16, XShape, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len)
            if is_full:
                sample.add(self.profile.phase("global_attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))
            else:
                sample.add(self.profile.phase("local_attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))

            @parameter
            def do_post_attn_norm[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                return post_attn_norm_dispatch[C.HIDDEN](
                    topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len),
                    body.post_attn_norm.bound(lb),
                    topo.activations.x_main.bound_dyn(topo.arena.base, seq_len), EPS, pool)
            sample.add(self.profile.phase("post_attn_norm"), timed_tp_parallel[Self.tp,do_post_attn_norm](topos, self.main_pools))

            # =============================================================
            # FFN BLOCK
            # =============================================================

            var act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var act_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var act_work2_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var expert_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var expert_act_scale_lease = self.scratch.borrow[Float32, 1]()

            @parameter
            def do_router_quantize[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return rmsnorm_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                    topo.activations.x_main.bound_dyn(topo.arena.base, seq_len), body.router_scale.bound(lb),
                    act_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_work_lease.view[F32, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    EPS, pool)
            sample.add(self.profile.phase("router_quantize"), timed_tp_parallel[Self.tp,do_router_quantize](topos, self.main_pools))

            var router_logits_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.NUM_EXPERTS]()
            var routing_lease = self.scratch.borrow[Gemma4TopKResult[C.TOP_K], 1]()

            @parameter
            def do_router_gemv[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return int8_gemv[C.NUM_EXPERTS, C.HIDDEN](
                    act_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    body.router_proj.bound(lb),
                    body.router_colsum.bound(lb),
                    body.router_proj_sc.bound(lb),
                    router_logits_lease.view[BF16, Shape[C.MAX_SEQ_LEN, C.NUM_EXPERTS]](sb, seq_len),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    pool)
            sample.add(self.profile.phase("router_proj"), timed_tp_parallel[Self.tp,do_router_gemv](topos, self.main_pools))

            @parameter
            def do_router_topk[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return router_topk_dispatch[C.NUM_EXPERTS, C.TOP_K](
                    router_logits_lease.view[BF16, Shape[1, C.NUM_EXPERTS]](sb, 1),
                    body.router_pes.bound(lb),
                    routing_lease.as_ptr[Gemma4TopKResult[C.TOP_K]](sb),
                    pool)
            sample.add(self.profile.phase("router_topk"), timed_tp_parallel[Self.tp,do_router_topk](topos, self.main_pools))

            var expert_qi_lease = self.scratch.borrow[Scalar[DType.int8], C.TOP_K * C.MOE_INTERMEDIATE]()
            var expert_blk_scale_lease = self.scratch.borrow[Float32, C.TOP_K * MOE_NUM_BLOCKS]()
            var expert_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
            var local_count_lease = self.scratch.borrow[Int32, 1]()

            @parameter
            def do_ffn_quantize[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return rmsnorm_dual_gamma_fwht_quantize[C.HIDDEN, FWHT_BLK_HIDDEN](
                    topo.activations.x_main.bound_dyn(topo.arena.base, seq_len),
                    body.pre_ffn_norm.bound(lb),
                    body.pre_ffn_norm_2.bound(lb),
                    act_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    expert_act_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_work_lease.view[F32, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_work2_lease.view[F32, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    expert_act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    EPS, pool)
            sample.add(self.profile.phase("ffn_quantize"), timed_tp_parallel[Self.tp,do_ffn_quantize](topos, self.main_pools))

            @parameter
            def do_expert_phase1[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                var routing = routing_lease.as_ptr[Gemma4TopKResult[C.TOP_K]](sb)[]
                comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
                var expert_base = rank * experts_per_rank
                var lc = 0
                for s in range(C.TOP_K):
                    var eid = routing.indices[s]
                    if eid >= expert_base and eid < expert_base + experts_per_rank:
                        lc += 1
                local_count_lease.as_ptr[Int32](sb)[] = Int32(lc)
                return gemma4_moe_phase1[
                    C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    expert_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1),
                    expert_act_scale_lease.view[F32, Shape[1, 1]](sb, 1),
                    routing,
                    body.experts_gate_up.bound(lb), C.MOE_GATE_UP_FUSED * C.HIDDEN,
                    body.experts_gate_up_sc.bound(lb), C.MOE_GATE_UP_FUSED * 4,
                    body.experts_gu_colsum.bound(lb), C.MOE_GATE_UP_FUSED * 4,
                    expert_qi_lease.view[I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                    expert_blk_scale_lease.view[F32, Shape[C.TOP_K, MOE_NUM_BLOCKS]](sb, C.TOP_K),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase1"), timed_tp_parallel[Self.tp,do_expert_phase1](topos, self.main_pools))

            var dense_post_i8_lease = self.scratch.borrow[Scalar[DType.int8], DENSE_INT_LOCAL]()

            @parameter
            def do_dense_phase1[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return fused_gu_gelu_tanh_wa[DENSE_INT_LOCAL, C.HIDDEN, FWHT_BLK_DENSE_DOWN](
                    act_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    body.gate_proj.bound(lb),
                    body.gate_proj_sc.bound(lb),
                    body.gu_colsum.bound(lb),
                    dense_post_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, DENSE_INT_LOCAL]](sb, seq_len),
                    post_blk_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, DENSE_DOWN_NUM_BLK]](sb, seq_len),
                    pool)
            sample.add(self.profile.phase("dense_phase1"), timed_tp_parallel[Self.tp,do_dense_phase1](topos, self.main_pools))

            @parameter
            def do_expert_phase2[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                var routing = routing_lease.as_ptr[Gemma4TopKResult[C.TOP_K]](sb)[]
                return gemma4_moe_phase2[
                    C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    expert_qi_lease.view[I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                    expert_blk_scale_lease.view[F32, Shape[C.TOP_K, MOE_NUM_BLOCKS]](sb, C.TOP_K),
                    routing,
                    body.experts_down.bound(lb), C.HIDDEN * C.MOE_INTERMEDIATE,
                    body.experts_down_sc.bound(lb), C.HIDDEN * 4,
                    body.experts_down_colsum.bound(lb), C.HIDDEN * MOE_NUM_BLOCKS * 4,
                    expert_out_lease.view[BF16, Shape[C.TOP_K, C.HIDDEN]](sb, C.TOP_K),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase2"), timed_tp_parallel[Self.tp,do_expert_phase2](topos, self.main_pools))

            var dense_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_dense_phase2[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return int8_gemv_blocked_wa[C.HIDDEN, DENSE_INT_LOCAL, FWHT_BLK_DENSE_DOWN](
                    dense_post_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, DENSE_INT_LOCAL]](sb, seq_len),
                    body.down_proj.bound(lb),
                    post_blk_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, DENSE_DOWN_NUM_BLK]](sb, seq_len),
                    body.down_proj_sc.bound(lb),
                    body.down_colsum.bound(lb),
                    dense_out_lease.view[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    pool)
            sample.add(self.profile.phase("dense_phase2"), timed_tp_parallel[Self.tp,do_dense_phase2](topos, self.main_pools))

            var dense_normed_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_expert_sum[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var sb = topo.arena.scratch_base()
                var lc = Int(local_count_lease.as_ptr[Int32](sb)[])
                return expert_sum_dispatch[C.HIDDEN, C.TOP_K](
                    expert_out_lease.view[BF16, Shape[C.TOP_K, C.HIDDEN]](sb, C.TOP_K),
                    lc,
                    topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len), pool)
            sample.add(self.profile.phase("pre_reduce"), timed_tp_parallel[Self.tp,do_expert_sum](topos, self.main_pools))

            var t_dense_reduce0 = Int(perf_counter_ns())
            var dense_out_ptrs = InlineArray[Int, Self.tp](fill=0)
            for r in range(Self.tp):
                dense_out_ptrs[r] = topos[r].arena.scratch_base() + dense_out_lease.offset
            small_allreduce[BF16, XShape, Self.tp](dense_out_ptrs, 1)
            sample.add(self.profile.phase("mlp_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_dense_reduce0))

            @parameter
            def do_dense_norm[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                return dense_norm_dispatch[C.HIDDEN](
                    dense_out_lease.view[BF16, Shape[1, C.HIDDEN]](sb, 1),
                    body.post_ffn_norm_1.bound(lb),
                    dense_normed_lease.view[BF16, Shape[1, C.HIDDEN]](sb, 1),
                    EPS, pool)
            sample.add(self.profile.phase("pre_reduce"), timed_tp_parallel[Self.tp,do_dense_norm](topos, self.main_pools))

            var t_mlp_reduce0 = Int(perf_counter_ns())
            small_allreduce[BF16, XShape, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len)
            sample.add(self.profile.phase("mlp_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_mlp_reduce0))

            @parameter
            def do_post_reduce[rank: Int, origin: MutOrigin](topo: Gemma4Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.full.base(topo.arena.base, full_idx) if is_full else topo.sliding.base(topo.arena.base, sliding_idx)
                var body = topo.full.proto.body if is_full else topo.sliding.proto.body
                var sb = topo.arena.scratch_base()
                var ls = body.layer_scalar.bound(lb).as_ptr()
                return post_reduce_dispatch[C.HIDDEN](
                    topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len),
                    body.post_ffn_norm_2.bound(lb),
                    dense_normed_lease.view[BF16, Shape[1, C.HIDDEN]](sb, 1),
                    body.post_ffn_norm.bound(lb),
                    topo.activations.x_main.bound_dyn(topo.arena.base, seq_len), Float32(ls[]), EPS, pool)
            sample.add(self.profile.phase("post_reduce"), timed_tp_parallel[Self.tp,do_post_reduce](topos, self.main_pools))

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
        var lm_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
        var lm_act_blk_scale_lease = self.scratch.borrow[Float32, VOCAB_NUM_BLOCKS]()
        var lm_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
        var t_final0 = Int(perf_counter_ns())
        var final_fence = rmsnorm_gamma_fwht_per_block_quantize[
            C.HIDDEN, LM_HEAD_FWHT_BLK](
            host.activations.x_main.bound_dyn(host.arena.base, seq_len),
            host.host.sqrt_gamma.bound(host.arena.base),
            lm_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            lm_work_lease.view[F32, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            lm_act_blk_scale_lease.view[F32, Shape[1, VOCAB_NUM_BLOCKS]](host.arena.scratch_base(), 1),
            EPS, self.main_pools[0])
        var t_final1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("final_norm"), finish_single_pool_fence(t_final0, t_final1, final_fence^))

        # lm_work is dead after final_norm, so release it before placing logits
        # above the remaining LM-head input leases.
        lm_work_lease^.release()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = logit_lease.view[BF16, Shape[1, C.VOCAB_SIZE]](host.arena.scratch_base(), 1)
        var t_lm0 = Int(perf_counter_ns())
        var lm_fence = lm_head_gemv[
            C.VOCAB_SIZE, C.HIDDEN, LM_HEAD_FWHT_BLK](
            lm_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            host.host.embed.bound(host.arena.base),
            lm_act_blk_scale_lease.view[F32, Shape[1, C.HIDDEN // LM_HEAD_FWHT_BLK]](host.arena.scratch_base(), 1),
            host.host.embed_sc.bound(host.arena.base),
            host.host.embed_colsum.bound(host.arena.base),
            logit_view,
            self.main_pools[0])
        var t_lm1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("lm_head"), finish_single_pool_fence(t_lm0, t_lm1, lm_fence^))
        var t_softcap0 = Int(perf_counter_ns())
        logit_softcap(logit_view)
        sample.add(self.profile.phase("softcap"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_softcap0))

        comptime width = simd_width_of[DType.float32]()
        var logits = LogitsView[C.VOCAB_SIZE](
            logit_view.as_ptr[DType.bfloat16](), logit_lease^)
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
