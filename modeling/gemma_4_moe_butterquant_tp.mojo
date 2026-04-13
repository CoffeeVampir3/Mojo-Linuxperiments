"""Gemma 4 26B-A4B ButterQuant — int8 MoE, NUMA-aware tensor parallel."""

from std.pathlib import Path
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import size_of, simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import (
    Encoding, Shaped, Named, BF16, F32, I8,
    Replicated, HOST_RANK, DISTRIBUTED,
    Slot, Bound, DynView,
    WeightDesc,
    DEFAULT_ALIGNMENT,
    LogitsView,
    QuantizeTask, QuantScheme,
    QuantPassthrough, RowQuantized, BlockQuantized,
    SmoothRowQuantized, SmoothBlockQuantized,
)
from kernels.kernel_ops import PoolFence
from kernels.reductions import ring_allreduce, ring_broadcast
from experimental.linear_borrow_pool import ScratchPool, ScratchLease
from experimental3.kernels.rmsnorm import (
    rmsnorm_gamma_fwht_quantize,
    rmsnorm_dual_gamma_fwht_quantize,
    rmsnorm_gamma_fwht_per_block_quantize,
    PostAttnNormArgs, post_attn_norm_kernel,
    PreReduceArgs, pre_reduce_kernel,
    ExpertSumArgs, expert_sum_kernel,
    DenseNormArgs, dense_norm_kernel,
    PostReduceArgs, post_reduce_kernel,
)
from experimental3.profiler import (
    PhaseTiming, phase_timing_from_points, finish_single_pool_fence,
    ForwardSample, ForwardLogger,
)
from experimental3.init_weights import (
    colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at,
)
from experimental3.kernels.int8_gemv import int8_gemv
from experimental3.kernels.float_gemv import float_gemv
from experimental3.moe import (
    gemma4_moe_phase1, gemma4_moe_phase2,
)
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.dense_ffn import (
    fused_gu_gelu_tanh,
    int8_gemv_blocked,
    RouterTopkArgs, router_topk_kernel,
)
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.kernels.sliding_attention import (
    AttnGroupArgs, sliding_attn_group_kernel,
)
from experimental3.kernels.full_attention import (
    full_attn_group_kernel,
)
from experimental3.kernels.full_chunked_attention import (
    ChunkedAttnArgs, chunked_attn_kernel, merge_and_quantize,
    FullAttnPrepArgs, full_attn_prep_kernel,
    partial_chunk_bytes, partial_chunk_stride,
)
from experimental3.kernels.lm_head import lm_head_gemv
from experimental_gemma.router import Gemma4TopKResult
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables
from experimental_gemma.ops import embed_lookup_scaled, embed_lookup_blocked, logit_softcap
from modeling.loader import load_weights, discover_shards, load_weights_from_descs
from simd_math import sqrt


# =============================================================================
# Config
# =============================================================================


struct Gemma4Config:
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
    comptime FWHT_BLK = 64
    comptime FWHT_BLK_HIDDEN = 256
    comptime LM_HEAD_FWHT_BLK = 64
    comptime DENSE_NUM_BLOCKS = Self.INTERMEDIATE // Self.FWHT_BLK
    comptime MOE_NUM_BLOCKS = Self.MOE_INTERMEDIATE // Self.FWHT_BLK
    comptime VOCAB_SIZE = 262144
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5
    comptime RMS_NORM_EPS = 1e-6
    comptime EMBED_SCALE = 53.0
    comptime LOGIT_SOFTCAP = 30.0

comptime C = Gemma4Config


# =============================================================================
# Runtime layout
# =============================================================================


struct LayerShard:
    comptime ROW  = 0   # split rows across ranks (local_rows = global_rows // tp)
    comptime COL  = 1   # split cols across ranks (local_cols = global_cols // tp)
    comptime REPL = 2   # full copy on every rank
    comptime HOST = 3   # full copy, pinned to HOST_RANK only


@fieldwise_init
struct LayerBuilder(Movable):
    """Builds weight catalog entries and byte offsets in a single pass."""
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
        # Shard kind fully determines local dims AND target rank.
        var local_rows = global_rows // self.tp if shard == LayerShard.ROW else global_rows
        var local_cols = global_cols // self.tp if shard == LayerShard.COL else global_cols
        var target_rank = HOST_RANK if shard == LayerShard.HOST else DISTRIBUTED
        var bytes = local_rows * local_cols * element_bytes
        var off = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
        self.cursor = off + bytes
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=dtype, element_bytes=element_bytes,
            global_rows=global_rows, global_cols=global_cols,
            local_rows=local_rows, local_cols=local_cols,
            quantizable=quantizable, absorbed=False,
            target_rank=target_rank,
        ))
        return off

    @always_inline
    def colsum(mut self, nbytes: Int) -> Int:
        """Reserve unaligned colsum region (computed at init, not loaded)."""
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


@fieldwise_init
struct LayerBodyWeights(Copyable, ImplicitlyCopyable, Movable):
    """Offsets shared between sliding and full layers."""
    var input_norm: Int
    # Dense MLP
    var pre_ffn_norm: Int
    var gate_proj: Int
    var up_proj: Int
    var gate_proj_sc: Int
    var up_proj_sc: Int
    var down_proj: Int
    var down_proj_sc: Int
    # Router
    var router_scale: Int
    var router_proj: Int
    var router_proj_sc: Int
    var router_pes: Int
    # MoE experts (block-sharded along the expert dim)
    var pre_ffn_norm_2: Int
    var experts_gate_up: Int
    var experts_gate_up_sc: Int
    var experts_down: Int
    var experts_down_sc: Int
    # Non-absorbable norms + per-layer scalar
    var post_attn_norm: Int
    var post_ffn_norm_1: Int
    var post_ffn_norm_2_rt: Int
    var post_ffn_norm: Int
    var layer_scalar: Int


@fieldwise_init
struct LayerBodyColsums(Copyable, ImplicitlyCopyable, Movable):
    var o_colsum: Int
    var gu_colsum: Int
    var down_colsum: Int
    var router_colsum: Int
    var experts_gu_colsum: Int
    var experts_down_colsum: Int


@fieldwise_init
struct SlidingLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    """Sliding-attention layer offsets."""
    var q_proj: Int
    var k_proj: Int
    var v_proj: Int
    var q_proj_sc: Int
    var k_proj_sc: Int
    var v_proj_sc: Int
    var o_proj: Int
    var o_proj_sc: Int
    var q_norm: Int
    var k_norm: Int
    var body: LayerBodyWeights
    var qkv_colsum: Int
    var colsums: LayerBodyColsums
    var stride: Int


@fieldwise_init
struct FullLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    """Full-attention layer offsets (K=V shared)."""
    var q_proj: Int
    var k_proj: Int
    var q_proj_sc: Int
    var k_proj_sc: Int
    var o_proj: Int
    var o_proj_sc: Int
    var q_norm: Int
    var k_norm: Int
    var body: LayerBodyWeights
    var qk_colsum: Int
    var colsums: LayerBodyColsums
    var stride: Int


def emit_layer_body[tp: Int](
    mut b: LayerBuilder, mut entries: List[WeightDesc],
) -> LayerBodyWeights:
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime INT = C.INTERMEDIATE
    comptime NE  = C.NUM_EXPERTS
    comptime GU  = C.MOE_GATE_UP_FUSED
    comptime MI  = C.MOE_INTERMEDIATE

    var input_norm     = b.bf(entries, "input_layernorm.weight",                H, 1,   REPL)
    var pre_ffn_norm   = b.bf(entries, "pre_feedforward_layernorm.weight",      H, 1,   REPL)
    var gate_proj      = b.q (entries, "mlp.gate_proj.weight",                  INT, H, ROW)
    var up_proj        = b.q (entries, "mlp.up_proj.weight",                    INT, H, ROW)
    var gate_proj_sc   = b.f (entries, "mlp.gate_proj.weight_scale",            INT, 1, ROW)
    var up_proj_sc     = b.f (entries, "mlp.up_proj.weight_scale",              INT, 1, ROW)
    var down_proj      = b.q (entries, "mlp.down_proj.weight",                  H, INT, COL)
    var down_proj_sc   = b.f (entries, "mlp.down_proj.weight_scale",            H, 1,   REPL)
    var router_scale   = b.bf(entries, "router.scale",                          H, 1,   REPL)
    var router_proj    = b.q (entries, "router.proj.weight",                    NE, H,  REPL)
    var router_proj_sc = b.f (entries, "router.proj.weight_scale",              NE, 1,  REPL)
    var router_pes     = b.bf(entries, "router.per_expert_scale",               NE, 1,  REPL)
    var pre_ffn_norm_2     = b.bf(entries, "pre_feedforward_layernorm_2.weight", H, 1,        REPL)
    var experts_gate_up    = b.q (entries, "experts.gate_up_proj",               NE * GU, H,  ROW)
    var experts_gate_up_sc = b.f (entries, "experts.gate_up_proj_scale",         NE * GU, 1,  ROW)
    var experts_down       = b.q (entries, "experts.down_proj",                  NE * H,  MI, ROW)
    var experts_down_sc    = b.f (entries, "experts.down_proj_scale",            NE * H,  1,  ROW)
    var post_attn_norm     = b.bf(entries, "post_attention_layernorm.weight",    H, 1, REPL)
    var post_ffn_norm_1    = b.bf(entries, "post_feedforward_layernorm_1.weight", H, 1, REPL)
    var post_ffn_norm_2_rt = b.bf(entries, "post_feedforward_layernorm_2.weight", H, 1, REPL)
    var post_ffn_norm      = b.bf(entries, "post_feedforward_layernorm.weight",  H, 1, REPL)
    var layer_scalar       = b.bf(entries, "layer_scalar",                       1, 1, REPL)

    return LayerBodyWeights(
        input_norm=input_norm,
        pre_ffn_norm=pre_ffn_norm,
        gate_proj=gate_proj, up_proj=up_proj,
        gate_proj_sc=gate_proj_sc, up_proj_sc=up_proj_sc,
        down_proj=down_proj, down_proj_sc=down_proj_sc,
        router_scale=router_scale, router_proj=router_proj,
        router_proj_sc=router_proj_sc, router_pes=router_pes,
        pre_ffn_norm_2=pre_ffn_norm_2,
        experts_gate_up=experts_gate_up, experts_gate_up_sc=experts_gate_up_sc,
        experts_down=experts_down, experts_down_sc=experts_down_sc,
        post_attn_norm=post_attn_norm,
        post_ffn_norm_1=post_ffn_norm_1,
        post_ffn_norm_2_rt=post_ffn_norm_2_rt,
        post_ffn_norm=post_ffn_norm,
        layer_scalar=layer_scalar,
    )


def emit_layer_colsums[tp: Int](mut b: LayerBuilder) -> LayerBodyColsums:
    comptime H   = C.HIDDEN
    comptime INT = C.INTERMEDIATE
    comptime NE  = C.NUM_EXPERTS
    comptime GU  = C.MOE_GATE_UP_FUSED
    comptime o_num_blk     = C.NUM_HEADS // tp
    comptime experts_local = NE // tp
    comptime INT_LOCAL     = INT // tp
    comptime DBLK          = C.FWHT_BLK if tp == 1 else 16
    comptime DOWN_BLK_LOCAL = INT_LOCAL // DBLK

    var o_colsum            = b.colsum(H * o_num_blk * 4)
    var gu_colsum           = b.colsum(INT_LOCAL * 2 * 4)
    var down_colsum         = b.colsum(H * DOWN_BLK_LOCAL * 4)
    var router_colsum       = b.colsum(NE * 4)
    var experts_gu_colsum   = b.colsum(experts_local * GU * 4)
    var experts_down_colsum = b.colsum(experts_local * H * C.MOE_NUM_BLOCKS * 4)
    return LayerBodyColsums(
        o_colsum=o_colsum, gu_colsum=gu_colsum, down_colsum=down_colsum,
        router_colsum=router_colsum,
        experts_gu_colsum=experts_gu_colsum,
        experts_down_colsum=experts_down_colsum,
    )


def sliding_layer_spec[tp: Int](
    prefix: String, layer_base: Int, mut entries: List[WeightDesc],
) -> SlidingLayerOffsets:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime HDS = C.HEAD_DIM_SLIDING

    # Attention
    var q_proj    = b.q (entries, "self_attn.q_proj.weight",        C.Q_DIM_SLIDING,  H,  ROW)
    var k_proj    = b.q (entries, "self_attn.k_proj.weight",        C.KV_DIM_SLIDING, H,  ROW)
    var v_proj    = b.q (entries, "self_attn.v_proj.weight",        C.KV_DIM_SLIDING, H,  ROW)
    var q_proj_sc = b.f (entries, "self_attn.q_proj.weight_scale",  C.Q_DIM_SLIDING,  1,  ROW)
    var k_proj_sc = b.f (entries, "self_attn.k_proj.weight_scale",  C.KV_DIM_SLIDING, 1,  ROW)
    var v_proj_sc = b.f (entries, "self_attn.v_proj.weight_scale",  C.KV_DIM_SLIDING, 1,  ROW)
    var o_proj    = b.q (entries, "self_attn.o_proj.weight",        H, C.Q_DIM_SLIDING, COL)
    var o_proj_sc = b.f (entries, "self_attn.o_proj.weight_scale",  H, 1,              REPL)
    var q_norm    = b.bf(entries, "self_attn.q_norm.weight",        HDS, 1,            REPL)
    var k_norm    = b.bf(entries, "self_attn.k_norm.weight",        HDS, 1,            REPL)

    # Shared body
    var body = emit_layer_body[tp](b, entries)

    # Colsums
    comptime qkv_n_loc = (C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // tp
    var qkv_colsum = b.colsum(qkv_n_loc * 4)
    var colsums = emit_layer_colsums[tp](b)

    return SlidingLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        q_proj_sc=q_proj_sc, k_proj_sc=k_proj_sc, v_proj_sc=v_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm,
        body=body,
        qkv_colsum=qkv_colsum, colsums=colsums,
        stride=b.cursor,
    )


def full_layer_spec[tp: Int](
    prefix: String, layer_base: Int, mut entries: List[WeightDesc],
) -> FullLayerOffsets:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime ROW, COL, REPL = LayerShard.ROW, LayerShard.COL, LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime HDF = C.HEAD_DIM_FULL

    # Attention
    var q_proj    = b.q (entries, "self_attn.q_proj.weight",       C.Q_DIM_FULL,  H, ROW)
    var k_proj    = b.q (entries, "self_attn.k_proj.weight",       C.KV_DIM_FULL, H, ROW)
    var q_proj_sc = b.f (entries, "self_attn.q_proj.weight_scale", C.Q_DIM_FULL,  1, ROW)
    var k_proj_sc = b.f (entries, "self_attn.k_proj.weight_scale", C.KV_DIM_FULL, 1, ROW)
    var o_proj    = b.q (entries, "self_attn.o_proj.weight",       H, C.Q_DIM_FULL, COL)
    var o_proj_sc = b.f (entries, "self_attn.o_proj.weight_scale", H, 1,            REPL)
    var q_norm    = b.bf(entries, "self_attn.q_norm.weight",       HDF, 1,          REPL)
    var k_norm    = b.bf(entries, "self_attn.k_norm.weight",       HDF, 1,          REPL)

    # Shared body
    var body = emit_layer_body[tp](b, entries)

    # Colsums
    comptime qk_n_loc = (C.Q_DIM_FULL + C.KV_DIM_FULL) // tp
    var qk_colsum = b.colsum(qk_n_loc * 4)
    var colsums = emit_layer_colsums[tp](b)

    return FullLayerOffsets(
        q_proj=q_proj, k_proj=k_proj,
        q_proj_sc=q_proj_sc, k_proj_sc=k_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm,
        body=body,
        qk_colsum=qk_colsum, colsums=colsums,
        stride=b.cursor,
    )


@fieldwise_init
struct Gemma4ModelLayout(Copyable, ImplicitlyCopyable, Movable):
    # Per-layer-kind offsets (relative to the layer's base address)
    var sliding: SlidingLayerOffsets
    var full: FullLayerOffsets
    # Layer bases in the weight block (relative to arena base)
    var sliding_off: Int
    var sliding_stride: Int
    var full_off: Int
    var full_stride: Int
    var distributed_bytes: Int
    # State block offsets (relative to state_base = arena_base + distributed_bytes)
    var x_main_off: Int
    var x_residual_off: Int
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_rope_half: Int
    var sliding_cos_off: Int
    var sliding_sin_off: Int
    var full_rope_half: Int
    var full_cos_off: Int
    var full_sin_off: Int
    var sliding_cache_off: Int
    var sliding_cache_stride: Int
    var full_cache_off: Int
    var full_cache_stride: Int
    var state_bytes: Int
    # Host-only (absolute offsets in the host arena)
    var host_only_off: Int
    var final_norm_off: Int
    var embed_off: Int
    var embed_sc_off: Int
    var vocab_num_blocks: Int
    var embed_colsum_off: Int
    var embed_colsum_bytes: Int
    var sqrt_gamma_off: Int       # bf16 [HIDDEN] — sqrt(|final_norm.weight|)
    var inv_sqrt_gamma_off: Int   # f32  [HIDDEN] — 1/sqrt(|final_norm.weight|)

    @always_inline
    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    @always_inline
    def host_arena_bytes(self) -> Int:
        return self.inv_sqrt_gamma_off + C.HIDDEN * 4


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16_bytes = size_of[Scalar[DType.bfloat16]]()
    comptime i8_bytes = size_of[Scalar[DType.int8]]()
    comptime f32_bytes = size_of[Float32]()
    comptime topk_bytes = size_of[Gemma4TopKResult[C.TOP_K]]()

    comptime DENSE_INT_LOCAL = C.INTERMEDIATE // tp
    comptime DBLK = C.FWHT_BLK if tp == 1 else 16
    comptime DBLK_LOCAL = DENSE_INT_LOCAL // DBLK
    comptime persistent = (
        f32_bytes                          # act_scale_lease
        + DBLK_LOCAL * f32_bytes           # post_blk_scale_lease
    )

    comptime qkv_n_local = C.Q_DIM_SLIDING // tp + 2 * (C.KV_DIM_SLIDING // tp)
    comptime sliding_attn_peak = persistent + (
        C.HIDDEN * i8_bytes
        + C.HIDDEN * f32_bytes
        + f32_bytes
        + qkv_n_local * bf16_bytes
        + (C.Q_DIM_SLIDING // tp) * i8_bytes
        + ((C.Q_DIM_SLIDING // tp) // C.HEAD_DIM_SLIDING) * f32_bytes
    )

    comptime qk_n_local = C.Q_DIM_FULL // tp + C.KV_DIM_FULL // tp
    comptime full_attn_phase1 = persistent + (
        qk_n_local * bf16_bytes
        + C.HIDDEN * i8_bytes
        + C.HIDDEN * f32_bytes
        + f32_bytes
    )
    comptime FULL_HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
    comptime FULL_ATTN_MAX_CHUNKS = 32
    comptime full_attn_phase2 = persistent + (
        qk_n_local * bf16_bytes
        + (C.Q_DIM_FULL // tp) * i8_bytes
        + ((C.Q_DIM_FULL // tp) // C.HEAD_DIM_FULL) * f32_bytes
        + FULL_HPG * C.HEAD_DIM_FULL * i8_bytes
        + FULL_HPG * f32_bytes * 2
        + FULL_ATTN_MAX_CHUNKS * FULL_HPG * (2 + C.HEAD_DIM_FULL) * f32_bytes
    )
    comptime full_attn_peak = (
        full_attn_phase1 if full_attn_phase1 > full_attn_phase2 else full_attn_phase2
    )

    comptime ffn_peak = persistent + (
        C.HIDDEN * i8_bytes
        + C.HIDDEN * f32_bytes
        + C.HIDDEN * f32_bytes
        + C.HIDDEN * i8_bytes
        + f32_bytes
        + C.NUM_EXPERTS * bf16_bytes
        + topk_bytes
        + C.TOP_K * C.MOE_INTERMEDIATE * i8_bytes
        + C.TOP_K * C.MOE_NUM_BLOCKS * f32_bytes
        + C.TOP_K * C.HIDDEN * bf16_bytes
        + size_of[Int32]()
        + DENSE_INT_LOCAL * i8_bytes
        + C.HIDDEN * bf16_bytes
        + C.HIDDEN * bf16_bytes
    )

    comptime layer_peak = (
        sliding_attn_peak if sliding_attn_peak > full_attn_peak else full_attn_peak
    )
    comptime decode_peak = ffn_peak if ffn_peak > layer_peak else layer_peak

    comptime vocab_num_blocks = C.HIDDEN // C.LM_HEAD_FWHT_BLK
    comptime lm_head_peak = (
        C.HIDDEN * i8_bytes
        + vocab_num_blocks * f32_bytes
        + C.HIDDEN * f32_bytes
        + C.VOCAB_SIZE * bf16_bytes
    )
    return lm_head_peak if lm_head_peak > decode_peak else decode_peak


@fieldwise_init
struct Gemma4LoadPlan(Movable):
    var layout: Gemma4ModelLayout
    var descs: List[WeightDesc]


def build_gemma4_load_plan[tp: Int]() -> Gemma4LoadPlan:
    var descs = List[WeightDesc]()

    # Probe strides from a template pass
    var scratch = List[WeightDesc]()
    var sliding_offsets = sliding_layer_spec[tp]("", 0, scratch)
    var full_offsets    = full_layer_spec[tp]("", 0, scratch)
    var sliding_stride  = sliding_offsets.stride
    var full_stride     = full_offsets.stride

    var sliding_off       = 0
    var full_off          = sliding_off + C.NUM_SLIDING_LAYERS * sliding_stride
    var distributed_bytes = full_off + C.NUM_FULL_LAYERS * full_stride

    # Emit catalog entries for all layers
    var sliding_idx = 0
    var full_idx = 0
    for i in range(C.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if (i + 1) % 6 == 0:
            var base = full_off + full_idx * full_stride
            _ = full_layer_spec[tp](prefix, base, descs)
            full_idx += 1
        else:
            var base = sliding_off + sliding_idx * sliding_stride
            _ = sliding_layer_spec[tp](prefix, base, descs)
            sliding_idx += 1

    # State block layout
    comptime bf16 = size_of[Scalar[DType.bfloat16]]()
    comptime f32  = size_of[Float32]()
    var x_main_off = 0
    var x_main_bytes = C.MAX_SEQ_LEN * C.HIDDEN * bf16
    var x_residual_off = x_main_off + x_main_bytes
    var scratch_off = x_residual_off + x_main_bytes
    var scratch_capacity = calculate_peak_scratch[tp]()

    comptime sliding_rope_half = C.HEAD_DIM_SLIDING // 2
    comptime full_rope_half = 64
    var sliding_cos_off = scratch_off + scratch_capacity
    var sliding_cos_bytes = C.MAX_SEQ_LEN * sliding_rope_half * f32
    var sliding_sin_off = sliding_cos_off + sliding_cos_bytes
    var full_cos_off = sliding_sin_off + sliding_cos_bytes
    var full_cos_bytes = C.MAX_SEQ_LEN * full_rope_half * f32
    var full_sin_off = full_cos_off + full_cos_bytes

    comptime sliding_cache_stride = Gemma4KVCache[
        C.SLIDING_WINDOW, C.HEAD_DIM_SLIDING,
        C.NUM_KV_HEADS_SLIDING // tp, C.NUM_HEADS // tp
    ].TOTAL_BYTES
    comptime full_cache_stride = Gemma4KVCache[
        C.MAX_SEQ_LEN, C.HEAD_DIM_FULL,
        C.NUM_KV_HEADS_FULL // tp, C.NUM_HEADS // tp
    ].TOTAL_BYTES
    var sliding_cache_off = full_sin_off + full_cos_bytes
    var full_cache_off    = sliding_cache_off + C.NUM_SLIDING_LAYERS * sliding_cache_stride
    var state_bytes       = full_cache_off + C.NUM_FULL_LAYERS * full_cache_stride

    # Host-only section
    var host_only_off = ((distributed_bytes + state_bytes + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime vocab_num_blocks = C.HIDDEN // C.LM_HEAD_FWHT_BLK
    comptime HOST = LayerShard.HOST
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_only_off
    var final_norm_off = hb.bf(descs, "model.language_model.norm.weight",
                                C.HIDDEN, 1,               HOST)
    var embed_off      = hb.q (descs, "model.language_model.embed_tokens.weight",
                                C.VOCAB_SIZE, C.HIDDEN,    HOST)
    var embed_sc_off   = hb.f (descs, "model.language_model.embed_tokens.weight_scale",
                                C.VOCAB_SIZE, vocab_num_blocks, HOST)
    # Embed colsums (computed at init)
    var embed_colsum_off = hb.colsum(C.VOCAB_SIZE * vocab_num_blocks * 4)
    var embed_colsum_bytes = C.VOCAB_SIZE * vocab_num_blocks * 4
    # Smooth-split gamma (computed at init)
    var sqrt_gamma_off = embed_colsum_off + embed_colsum_bytes
    var inv_sqrt_gamma_off = sqrt_gamma_off + C.HIDDEN * 2  # sqrt_gamma is bf16

    var layout = Gemma4ModelLayout(
        sliding=sliding_offsets, full=full_offsets,
        sliding_off=sliding_off, sliding_stride=sliding_stride,
        full_off=full_off, full_stride=full_stride,
        distributed_bytes=distributed_bytes,
        x_main_off=x_main_off, x_residual_off=x_residual_off,
        scratch_off=scratch_off, scratch_capacity=scratch_capacity,
        sliding_rope_half=sliding_rope_half,
        sliding_cos_off=sliding_cos_off, sliding_sin_off=sliding_sin_off,
        full_rope_half=full_rope_half,
        full_cos_off=full_cos_off, full_sin_off=full_sin_off,
        sliding_cache_off=sliding_cache_off, sliding_cache_stride=sliding_cache_stride,
        full_cache_off=full_cache_off, full_cache_stride=full_cache_stride,
        state_bytes=state_bytes,
        host_only_off=host_only_off,
        final_norm_off=final_norm_off,
        embed_off=embed_off,
        embed_sc_off=embed_sc_off,
        vocab_num_blocks=vocab_num_blocks,
        embed_colsum_off=embed_colsum_off,
        embed_colsum_bytes=embed_colsum_bytes,
        sqrt_gamma_off=sqrt_gamma_off,
        inv_sqrt_gamma_off=inv_sqrt_gamma_off,
    )
    return Gemma4LoadPlan(layout, descs^)


# =============================================================================
# Rank view
# =============================================================================


@fieldwise_init
struct RankView[tp: Int](Copyable, Movable):
    # Shape aliases for Bound[T] / DynView[T]
    comptime X_MAIN     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime SLIDING_ROPE_HALF = C.HEAD_DIM_SLIDING // 2
    comptime SLIDING_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime SLIDING_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime FULL_ROPE_HALF = 64
    comptime FULL_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF, Self.tp]
    comptime FULL_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF, Self.tp]
    # Host-only shapes
    comptime EMBED      = Slot[I8, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp]
    comptime VOCAB_NUM_BLOCKS = C.HIDDEN // C.LM_HEAD_FWHT_BLK
    comptime EMBED_SC   = Slot[F32, Replicated, C.VOCAB_SIZE, Self.VOCAB_NUM_BLOCKS, Self.tp]
    comptime FINAL_NORM = Slot[BF16, Replicated, C.HIDDEN, 1, Self.tp]
    comptime LOGITS     = Slot[BF16, Replicated, 1, C.VOCAB_SIZE, Self.tp]

    var base: Int
    var layout: UnsafePointer[Gemma4ModelLayout, MutAnyOrigin]

    @always_inline
    def L(self) -> ref [MutAnyOrigin] Gemma4ModelLayout:
        return self.layout[]

    def weight_base(self) -> Int:
        return self.base
    def state_base(self) -> Int:
        return self.base + self.L().distributed_bytes
    def scratch_base(self) -> Int:
        return self.state_base() + self.L().scratch_off
    def scratch_addr(self, read lease: ScratchLease) -> Int:
        return self.scratch_base() + lease.offset
    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)
    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        return DynView[V](self.scratch_base() + lease.offset, seq_len)

    def x_main(self, seq_len: Int) -> DynView[Self.X_MAIN]:
        return DynView[Self.X_MAIN](self.state_base() + self.L().x_main_off, seq_len)
    def x_residual(self, seq_len: Int) -> DynView[Self.X_RESIDUAL]:
        return DynView[Self.X_RESIDUAL](self.state_base() + self.L().x_residual_off, seq_len)

    def sliding_cos(self) -> Bound[Self.SLIDING_COS]:
        return Bound[Self.SLIDING_COS](self.state_base() + self.L().sliding_cos_off)
    def sliding_sin(self) -> Bound[Self.SLIDING_SIN]:
        return Bound[Self.SLIDING_SIN](self.state_base() + self.L().sliding_sin_off)
    def full_cos(self) -> Bound[Self.FULL_COS]:
        return Bound[Self.FULL_COS](self.state_base() + self.L().full_cos_off)
    def full_sin(self) -> Bound[Self.FULL_SIN]:
        return Bound[Self.FULL_SIN](self.state_base() + self.L().full_sin_off)

    def sliding_cos_row(self, pos: Int) -> Int:
        return self.state_base() + self.L().sliding_cos_off + pos * Self.SLIDING_ROPE_HALF * size_of[Float32]()
    def sliding_sin_row(self, pos: Int) -> Int:
        return self.state_base() + self.L().sliding_sin_off + pos * Self.SLIDING_ROPE_HALF * size_of[Float32]()

    def sliding_cache_base(self, layer_idx: Int) -> Int:
        return self.state_base() + self.L().sliding_cache_off + layer_idx * self.L().sliding_cache_stride
    def full_cache_base(self, full_layer_idx: Int) -> Int:
        return self.state_base() + self.L().full_cache_off + full_layer_idx * self.L().full_cache_stride

    def sliding_layer_base(self, sliding_idx: Int) -> Int:
        return self.weight_base() + self.L().sliding_off + sliding_idx * self.L().sliding_stride
    def full_layer_base(self, full_idx: Int) -> Int:
        return self.weight_base() + self.L().full_off + full_idx * self.L().full_stride

    # Host-only accessors
    def embed_table(self) -> Bound[Self.EMBED]:
        return Bound[Self.EMBED](self.weight_base() + self.L().embed_off)
    def embed_scale(self) -> Bound[Self.EMBED_SC]:
        return Bound[Self.EMBED_SC](self.weight_base() + self.L().embed_sc_off)
    def final_norm(self) -> Bound[Self.FINAL_NORM]:
        return Bound[Self.FINAL_NORM](self.weight_base() + self.L().final_norm_off)


# =============================================================================
# Ranks
# =============================================================================


@fieldwise_init
struct Ranks[tp: Int]:
    var bases: InlineArray[Int, Self.tp]
    var layout: UnsafePointer[Gemma4ModelLayout, MutAnyOrigin]

    def view(self, r: Int) -> RankView[Self.tp]:
        return RankView[Self.tp](self.bases[r], self.layout)

    def timed_parallel[body: def[rank: Int](RankView[Self.tp], mut BurstPool[]) capturing -> PoolFence[BurstPool[]]](
        self, pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp],
    ) -> PhaseTiming:
        var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
        var active = InlineArray[Bool, Self.tp](fill=False)
        var t0 = Int(perf_counter_ns())
        comptime for rank in range(Self.tp):
            ptrs[rank] = body[rank](self.view(rank), pool_ptrs[rank][]).take()
        var t1 = Int(perf_counter_ns())
        for i in range(Self.tp):
            if ptrs[i]:
                active[i] = True
                ptrs[i][].join()
        var t2 = Int(perf_counter_ns())

        var max_done_ns = 0
        var any_active = False
        for i in range(Self.tp):
            if active[i]:
                any_active = True
                var ts = ptrs[i][].last_worker_timestamp()
                if ts > max_done_ns:
                    max_done_ns = ts
        return phase_timing_from_points(t0, t1, max_done_ns, t1, t2, any_active)

    def x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_residual(seq_len).ptr
        return ptrs^

    def x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_main(seq_len).ptr
        return ptrs^


# =============================================================================
# Load-time: column sums + VNNI packing
# =============================================================================


def init_layer_body[tp: Int](arena_base: Int, layer_base: Int,
    body: LayerBodyWeights, colsums: LayerBodyColsums,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """Colsum + VNNI pack for everything shared between sliding and full
    layers (dense MLP, router, per-rank experts)."""
    comptime experts_local = C.NUM_EXPERTS // tp
    comptime ne_gu_local = experts_local * C.MOE_GATE_UP_FUSED
    comptime INT_LOCAL = C.INTERMEDIATE // tp
    comptime DBLK = C.FWHT_BLK if tp == 1 else 16
    # Dense MLP colsums (sharded: gate/up ROW, down COL).
    colsum_at(arena_base, layer_base + body.gate_proj, layer_base + colsums.gu_colsum,
        INT_LOCAL * 2, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + body.down_proj, layer_base + colsums.down_colsum,
        C.HIDDEN, INT_LOCAL, DBLK)
    # Router + expert colsums.
    colsum_at(arena_base, layer_base + body.router_proj, layer_base + colsums.router_colsum,
        C.NUM_EXPERTS, C.HIDDEN)
    colsum_at(arena_base, layer_base + body.experts_gate_up, layer_base + colsums.experts_gu_colsum,
        ne_gu_local, C.HIDDEN)
    for e in range(experts_local):
        block_colsum_at(arena_base,
            layer_base + body.experts_down + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            layer_base + colsums.experts_down_colsum + e * C.HIDDEN * C.MOE_NUM_BLOCKS * 4,
            C.HIDDEN, C.MOE_INTERMEDIATE, C.FWHT_BLK)
    # VNNI packing — colsums above must be done first (pack reorders bytes).
    pack_at(arena_base, layer_base + body.gate_proj,   INT_LOCAL * 2, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + body.down_proj,   C.HIDDEN, INT_LOCAL,     scratch)
    pack_at(arena_base, layer_base + body.router_proj, C.NUM_EXPERTS, C.HIDDEN,      scratch)
    for e in range(experts_local):
        pack_at(arena_base, layer_base + body.experts_gate_up + e * C.MOE_GATE_UP_FUSED * C.HIDDEN,
            C.MOE_GATE_UP_FUSED, C.HIDDEN, scratch)
        pack_at(arena_base, layer_base + body.experts_down + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            C.HIDDEN, C.MOE_INTERMEDIATE, scratch)


def init_sliding_layer[tp: Int](arena_base: Int, layer_base: Int,
    sl: SlidingLayerOffsets,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """Per-rank colsum + VNNI pack for one sliding layer's local block.

    The attention QKV/O block is the only kind-specific section; everything
    shared runs through init_layer_body.
    """
    comptime qkv_n_local = (C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING) // tp
    comptime q_local     = C.Q_DIM_SLIDING // tp
    # Attention colsums (kind-specific: [Q|K|V], then O per-head-block).
    colsum_at(arena_base, layer_base + sl.q_proj, layer_base + sl.qkv_colsum,
        qkv_n_local, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + sl.o_proj, layer_base + sl.colsums.o_colsum,
        C.HIDDEN, q_local, C.HEAD_DIM_SLIDING)
    # Shared body.
    init_layer_body[tp](arena_base, layer_base, sl.body, sl.colsums, scratch)
    # Attention VNNI pack (kind-specific).
    pack_at(arena_base, layer_base + sl.q_proj, qkv_n_local, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + sl.o_proj, C.HIDDEN, q_local, scratch)


def init_full_layer[tp: Int](arena_base: Int, layer_base: Int,
    fl: FullLayerOffsets,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """Per-rank colsum + VNNI pack for one full-attention layer's local block."""
    comptime qk_n_local = (C.Q_DIM_FULL + C.KV_DIM_FULL) // tp
    comptime q_local    = C.Q_DIM_FULL // tp
    # Attention colsums (kind-specific: [Q|K], then O per-head-block).
    colsum_at(arena_base, layer_base + fl.q_proj, layer_base + fl.qk_colsum,
        qk_n_local, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + fl.o_proj, layer_base + fl.colsums.o_colsum,
        C.HIDDEN, q_local, C.HEAD_DIM_FULL)
    # Shared body.
    init_layer_body[tp](arena_base, layer_base, fl.body, fl.colsums, scratch)
    # Attention VNNI pack (kind-specific).
    pack_at(arena_base, layer_base + fl.q_proj, qk_n_local, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + fl.o_proj, C.HIDDEN, q_local, scratch)


# =============================================================================
# Model struct
# =============================================================================


struct Gemma4ButterQuant[tp: Int](Movable):
    # Peak scratch bytes for VNNI packing — sized for the widest quantized
    # weight that pack_at re-copies into scratch. Full-attn Q+K stacked is
    # the biggest contiguous row-major source in the layout.
    comptime MAX_PACK_BYTES = (C.Q_DIM_FULL + C.KV_DIM_FULL) * C.HIDDEN

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var profile: ForwardLogger
    var layout: Gemma4ModelLayout

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[BurstPool[]],
        var sc: ScratchPool,
        bases: InlineArray[Int, Self.tp],
        layout: Gemma4ModelLayout,
    ):
        self.arenas = arenas^
        self.main_pools = mp^
        self.scratch = sc^
        self.bases = bases
        self.profile = ForwardLogger()
        self.layout = layout

    def make_pool_ptrs(self, pools: HeapMoveArray[BurstPool[]]) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
        for r in range(Self.tp):
            ptrs[r] = UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=pools[r])))
        return ptrs^

    def ranks(mut self) -> Ranks[Self.tp]:
        return Ranks[Self.tp](self.bases,
            UnsafePointer[Gemma4ModelLayout, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.layout))))
    def main_ptrs(self) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        return self.make_pool_ptrs(self.main_pools)

    # =========================================================================
    # Load + init
    # =========================================================================

    def init_state(mut self):
        var pack_scratch = alloc[UInt8](Self.MAX_PACK_BYTES)
        var L = self.layout
        for rank in range(Self.tp):
            var rv = self.ranks().view(rank)
            init_sliding_rope_tables(rv.sliding_cos(), rv.sliding_sin())
            init_full_rope_tables(rv.full_cos(), rv.full_sin())

            var base = Int(self.arenas[rank].base)
            var sliding_idx = 0
            var full_idx = 0
            for i in range(C.NUM_LAYERS):
                if (i + 1) % 6 == 0:
                    init_full_layer[Self.tp](base,
                        L.full_off + full_idx * L.full_stride,
                        L.full, pack_scratch)
                    full_idx += 1
                else:
                    init_sliding_layer[Self.tp](base,
                        L.sliding_off + sliding_idx * L.sliding_stride,
                        L.sliding, pack_scratch)
                    sliding_idx += 1

            # LM head colsums + smooth-split gamma (host rank only)
            if rank == HOST_RANK:
                block_colsum_row_major_at(
                    base,
                    L.embed_off,
                    L.embed_colsum_off,
                    C.VOCAB_SIZE, C.HIDDEN, C.LM_HEAD_FWHT_BLK)

                var fn_gamma = BF16Ptr(unsafe_from_address=base + L.final_norm_off)
                compute_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    BF16Ptr(unsafe_from_address=base + L.sqrt_gamma_off))
                compute_inv_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    F32Ptr(unsafe_from_address=base + L.inv_sqrt_gamma_off))

            # Fold 1/sqrt(hidden) into router weight scales
            comptime inv_sqrt_h = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
            sliding_idx = 0
            full_idx = 0
            for i in range(C.NUM_LAYERS):
                var sc_ptr: UnsafePointer[Float32, MutAnyOrigin]
                if (i + 1) % 6 == 0:
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=base + L.full_off + full_idx * L.full_stride + L.full.body.router_proj_sc)
                    full_idx += 1
                else:
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=base + L.sliding_off + sliding_idx * L.sliding_stride + L.sliding.body.router_proj_sc)
                    sliding_idx += 1
                for n in range(C.NUM_EXPERTS):
                    sc_ptr[n] *= inv_sqrt_h

        pack_scratch.free()

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_gemma4_load_plan[Self.tp]()

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = plan.layout.host_arena_bytes() if rank == HOST_RANK else plan.layout.arena_bytes()
            print("rank", rank, "node", topo[rank], "allocating", size // (1024 * 1024), "MB")
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
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
            _ = arenas[rank].prefault(plan.layout.distributed_bytes, plan.layout.state_bytes)

        var main_pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            main_pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var bases = InlineArray[Int, Self.tp](fill=0)
        for rank in range(Self.tp):
            bases[rank] = Int(arenas[rank].base)

        var scratch = ScratchPool(plan.layout.scratch_capacity)
        var model = Self(arenas^, main_pools^, scratch^, bases, plan.layout)
        model.init_state()
        print("state initialized")
        return model^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.ranks().view(0).scratch_base())

    @staticmethod
    def build_quantizer_tasks() -> List[QuantizeTask]:
        var tasks = List[QuantizeTask]()
        comptime HB = C.FWHT_BLK_HIDDEN
        comptime LB = C.LM_HEAD_FWHT_BLK

        var pt = QuantScheme(QuantPassthrough())
        var row_hb = QuantScheme(RowQuantized(rotation=HB))
        # Channelwise (no rotation). FWHT is applied to the activation at
        # runtime, not baked into the weight. These weights are TP-safe under
        # any row/col shard — no block alignment constraint.
        var row_no_rot = QuantScheme(RowQuantized(rotation=0))

        for i in range(C.NUM_LAYERS):
            var prefix = "model.language_model.layers." + String(i) + "."
            var is_full = (i + 1) % 6 == 0

            # Attention projections
            tasks.append(QuantizeTask(prefix + "self_attn.q_proj.weight", row_hb.copy()))
            tasks.append(QuantizeTask(prefix + "self_attn.k_proj.weight", row_hb.copy()))
            if not is_full:
                tasks.append(QuantizeTask(prefix + "self_attn.v_proj.weight", row_hb.copy()))
            if is_full:
                tasks.append(QuantizeTask(prefix + "self_attn.o_proj.weight",
                    QuantScheme(RowQuantized(rotation=512))))
            else:
                tasks.append(QuantizeTask(prefix + "self_attn.o_proj.weight", row_hb.copy()))

            # Dense MLP — gate/up rotate on K=HIDDEN (unsplit), TP-safe under ROW shard.
            # down_proj is channelwise (no rotation), TP-safe under COL shard.
            tasks.append(QuantizeTask(prefix + "mlp.gate_proj.weight", row_hb.copy()))
            tasks.append(QuantizeTask(prefix + "mlp.up_proj.weight", row_hb.copy()))
            tasks.append(QuantizeTask(prefix + "mlp.down_proj.weight", row_no_rot.copy()))

            # Router + experts — channelwise, no rotation on weights.
            tasks.append(QuantizeTask(prefix + "router.proj.weight", row_hb.copy()))
            tasks.append(QuantizeTask(prefix + "experts.gate_up_proj", row_no_rot.copy()))
            tasks.append(QuantizeTask(prefix + "experts.down_proj", row_no_rot.copy()))

            # Norms + scalars — passthrough
            tasks.append(QuantizeTask(prefix + "input_layernorm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "post_attention_layernorm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "pre_feedforward_layernorm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "pre_feedforward_layernorm_2.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "post_feedforward_layernorm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "post_feedforward_layernorm_1.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "post_feedforward_layernorm_2.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "self_attn.q_norm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "self_attn.k_norm.weight", pt.copy()))
            tasks.append(QuantizeTask(prefix + "router.scale", pt.copy()))
            tasks.append(QuantizeTask(prefix + "router.per_expert_scale", pt.copy()))
            tasks.append(QuantizeTask(prefix + "layer_scalar", pt.copy()))

        # Host-only
        tasks.append(QuantizeTask("model.language_model.norm.weight", pt.copy()))
        tasks.append(QuantizeTask("model.language_model.embed_tokens.weight",
            QuantScheme(SmoothBlockQuantized(
                rotation=LB, scale_blk=LB,
                smooth_src="model.language_model.norm.weight"))))
        return tasks^

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    # =========================================================================
    # Forward — decode (seq_len=1)
    # =========================================================================


    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        # Shape-carrying type aliases the kernels want at comptime.
        comptime RV = RankView[Self.tp]
        comptime QKV_N_SLIDING = C.Q_DIM_SLIDING + 2 * C.KV_DIM_SLIDING
        comptime QK_N_FULL = C.Q_DIM_FULL + C.KV_DIM_FULL
        comptime Q_DIM_LOCAL_SLIDING = C.Q_DIM_SLIDING // Self.tp
        comptime KV_DIM_LOCAL_SLIDING = C.KV_DIM_SLIDING // Self.tp
        comptime Q_DIM_LOCAL_FULL = C.Q_DIM_FULL // Self.tp
        comptime EPS = Float32(C.RMS_NORM_EPS)
        comptime seq_len = 1

        # Runtime offsets — read-only views onto the layout on `self`.
        # Closures below capture these by reference; no per-closure copies.
        var L = self.layout
        var sl = self.layout.sliding
        var fl = self.layout.full

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(pos)
        var rnks = self.ranks()
        var host = rnks.view(0)
        var mp = self.main_ptrs()

        # --- Embed ---
        var t_embed0 = Int(perf_counter_ns())
        var embed_fence = embed_lookup_blocked[fwht_blk = C.LM_HEAD_FWHT_BLK](
            host.embed_table(),
            host.embed_scale(),
            host.weight_base() + L.inv_sqrt_gamma_off,
            tokens_ptr,
            host.x_main(seq_len), Float32(C.EMBED_SCALE),
            self.main_pools[0])
        var t_embed1 = Int(perf_counter_ns())
        sample.embed = finish_single_pool_fence(t_embed0, t_embed1, embed_fence^)

        var t_bcast0 = Int(perf_counter_ns())
        ring_broadcast[RV.X_MAIN, Self.tp](
            host.x_main(seq_len).ptr, rnks.x_main_ptrs(seq_len), seq_len, mp)
        sample.broadcast = PhaseTiming.opaque(Int(perf_counter_ns()) - t_bcast0)

        var act_scale_lease = self.scratch.borrow[Float32, 1]()
        comptime DENSE_INT_LOCAL = C.INTERMEDIATE // Self.tp
        comptime DBLK = C.FWHT_BLK if Self.tp == 1 else 16
        comptime DBLK_LOCAL = DENSE_INT_LOCAL // DBLK
        var post_blk_scale_lease = self.scratch.borrow[Float32, DBLK_LOCAL]()

        var sliding_idx = 0
        var full_idx = 0
        for layer_idx in range(C.NUM_LAYERS):
            var is_full = (layer_idx + 1) % 6 == 0

            # =============================================================
            # ATTENTION BLOCK
            # =============================================================

            if not is_full:
                var attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
                var attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
                var attn_scale_lease = self.scratch.borrow[Float32, 1]()

                @parameter
                def do_attn_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var slb = rv.sliding_layer_base(sliding_idx)
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                        rv.x_main(seq_len).ptr, slb + sl.body.input_norm,
                        rv.scratch_addr(attn_i8_lease),
                        rv.scratch_addr(attn_work_lease), rv.scratch_addr(attn_scale_lease),
                        EPS,
                        seq_len, pool)
                sample.attn_quantize.add(rnks.timed_parallel[do_attn_quantize](mp))

                var qkv_lease = self.scratch.borrow[Scalar[DType.bfloat16], QKV_N_SLIDING // Self.tp]()

                @parameter
                def do_qkv_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var slb = rv.sliding_layer_base(sliding_idx)
                    return int8_gemv[QKV_N_SLIDING // Self.tp, C.HIDDEN](
                        rv.scratch_addr(attn_i8_lease),
                        slb + sl.q_proj,
                        slb + sl.qkv_colsum,
                        slb + sl.q_proj_sc,
                        rv.scratch_addr(qkv_lease),
                        seq_len, rv.scratch_addr(attn_scale_lease), pool)
                sample.attn_proj.add(rnks.timed_parallel[do_qkv_gemv](mp))

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_SLIDING]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, Q_DIM_LOCAL_SLIDING // C.HEAD_DIM_SLIDING]()

                @parameter
                def do_sliding_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
                    comptime NKV = C.NUM_KV_HEADS_SLIDING // Self.tp
                    var qkv_base = rv.scratch_addr(qkv_lease)
                    var q_base = qkv_base
                    var k_base = qkv_base + Q_DIM_LOCAL_SLIDING * 2
                    var v_base = k_base + KV_DIM_LOCAL_SLIDING * 2
                    var cache_pos = pos % C.SLIDING_WINDOW
                    var context_len = min(pos + 1, C.SLIDING_WINDOW)
                    var slb = rv.sliding_layer_base(sliding_idx)
                    var q_norm_addr = slb + sl.q_norm
                    var k_norm_addr = slb + sl.k_norm
                    var jobs = InlineArray[AttnGroupArgs, 8](
                        fill=AttnGroupArgs())
                    for g in range(NKV):
                        jobs[g] = AttnGroupArgs(
                            q_base + g * HPG * C.HEAD_DIM_SLIDING * 2,
                            k_base + g * C.HEAD_DIM_SLIDING * 2,
                            v_base + g * C.HEAD_DIM_SLIDING * 2,
                            q_norm_addr, k_norm_addr,
                            rv.sliding_cos_row(pos), rv.sliding_sin_row(pos),
                            rv.sliding_cache_base(sliding_idx), g,
                            cache_pos, context_len,
                            rv.scratch_addr(attn_qi_lease) + g * HPG * C.HEAD_DIM_SLIDING,
                            rv.scratch_addr(attn_head_sc_lease) + g * HPG * 4,
                            EPS)
                    pool.dispatch[AttnGroupArgs,
                        sliding_attn_group_kernel[C.HEAD_DIM_SLIDING, HPG,
                            C.SLIDING_WINDOW, C.NUM_KV_HEADS_SLIDING // Self.tp, C.NUM_HEADS // Self.tp]](
                        UnsafePointer(to=jobs[0]), NKV)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                sample.attention.add(rnks.timed_parallel[do_sliding_attn](mp))

                @parameter
                def do_o_proj[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var slb = rv.sliding_layer_base(sliding_idx)
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_SLIDING, C.HEAD_DIM_SLIDING](
                        I8Ptr(unsafe_from_address=rv.scratch_addr(attn_qi_lease)),
                        U8Ptr(unsafe_from_address=slb + sl.o_proj),
                        F32Ptr(unsafe_from_address=rv.scratch_addr(attn_head_sc_lease)),
                        F32Ptr(unsafe_from_address=slb + sl.o_proj_sc),
                        F32Ptr(unsafe_from_address=slb + sl.colsums.o_colsum),
                        BF16Ptr(unsafe_from_address=rv.x_residual(seq_len).ptr),
                        seq_len, pool)
                sample.o_proj.add(rnks.timed_parallel[do_o_proj](mp))
                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                qkv_lease^.release()
                attn_scale_lease^.release()
                attn_work_lease^.release()
                attn_i8_lease^.release()
            else:
                # Full attention
                comptime FULL_HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
                comptime FULL_NKV = C.NUM_KV_HEADS_FULL // Self.tp
                comptime ROPE_DIMS_FULL = 128
                var qk_lease = self.scratch.borrow[Scalar[DType.bfloat16], QK_N_FULL // Self.tp]()

                var full_attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
                var full_attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
                var full_attn_scale_lease = self.scratch.borrow[Float32, 1]()

                @parameter
                def do_full_attn_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                        rv.x_main(seq_len).ptr, flb + fl.body.input_norm,
                        rv.scratch_addr(full_attn_i8_lease),
                        rv.scratch_addr(full_attn_work_lease), rv.scratch_addr(full_attn_scale_lease),
                        EPS,
                        seq_len, pool)
                sample.attn_quantize.add(rnks.timed_parallel[do_full_attn_quantize](mp))

                @parameter
                def do_full_qk_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return int8_gemv[QK_N_FULL // Self.tp, C.HIDDEN](
                        rv.scratch_addr(full_attn_i8_lease),
                        flb + fl.q_proj,
                        flb + fl.qk_colsum,
                        flb + fl.q_proj_sc,
                        rv.scratch_addr(qk_lease),
                        seq_len, rv.scratch_addr(full_attn_scale_lease), pool)
                sample.attn_proj.add(rnks.timed_parallel[do_full_qk_gemv](mp))
                full_attn_scale_lease^.release()
                full_attn_work_lease^.release()
                full_attn_i8_lease^.release()

                # Scoring + V-agg (context-parallel chunked)
                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_DIM_LOCAL_FULL]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, Q_DIM_LOCAL_FULL // C.HEAD_DIM_FULL]()
                var q_i8_prep_lease = self.scratch.borrow[Scalar[DType.int8], FULL_HPG * C.HEAD_DIM_FULL]()
                var qi_biases_lease = self.scratch.borrow[Float32, FULL_HPG]()
                var q_scales_lease = self.scratch.borrow[Float32, FULL_HPG]()
                comptime FULL_ATTN_MAX_CHUNKS = 32
                comptime PARTIAL_F32S = FULL_ATTN_MAX_CHUNKS * FULL_HPG * (2 + C.HEAD_DIM_FULL)
                var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()

                # Phase 1: KV cache write + Q prep (1 job per rank)
                @parameter
                def do_full_attn_prep[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var qk_base = rv.scratch_addr(qk_lease)
                    var q_base = qk_base
                    var k_base = qk_base + Q_DIM_LOCAL_FULL * 2
                    var cos_addr = rv.state_base() + L.full_cos_off + pos * RV.FULL_ROPE_HALF * size_of[Float32]()
                    var sin_addr = rv.state_base() + L.full_sin_off + pos * RV.FULL_ROPE_HALF * size_of[Float32]()
                    var flb = rv.full_layer_base(full_idx)
                    comptime FULL_NKV_LOCAL = C.NUM_KV_HEADS_FULL // Self.tp
                    var jobs = InlineArray[FullAttnPrepArgs, 1](fill=FullAttnPrepArgs())
                    jobs[0] = FullAttnPrepArgs(
                        q_bf16_base=q_base,
                        k_bf16_ptr=k_base,
                        q_norm_ptr=flb + fl.q_norm,
                        k_norm_ptr=flb + fl.k_norm,
                        cos_ptr=cos_addr,
                        sin_ptr=sin_addr,
                        cache_base=rv.full_cache_base(full_idx),
                        cache_pos=pos,
                        kv_head=0,
                        eps=EPS,
                        q_i8_out=rv.scratch_addr(q_i8_prep_lease),
                        qi_biases_out=rv.scratch_addr(qi_biases_lease),
                        q_scales_out=rv.scratch_addr(q_scales_lease))
                    pool.dispatch[FullAttnPrepArgs,
                        full_attn_prep_kernel[C.HEAD_DIM_FULL, ROPE_DIMS_FULL, FULL_HPG,
                            C.MAX_SEQ_LEN, FULL_NKV_LOCAL, C.NUM_HEADS // Self.tp]](
                        UnsafePointer(to=jobs[0]), 1)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                sample.attention.add(rnks.timed_parallel[do_full_attn_prep](mp))

                # Phase 2: Chunked scoring (pool.capacity jobs per rank)
                @parameter
                def do_full_chunk_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var context_len = pos + 1
                    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
                    var num_chunks = min(Int(pool.capacity), FULL_ATTN_MAX_CHUNKS)
                    if num_chunks > num_pg:
                        num_chunks = num_pg
                    var pgs_per_chunk = (num_pg + num_chunks - 1) // num_chunks
                    comptime FULL_NKV_LOCAL = C.NUM_KV_HEADS_FULL // Self.tp
                    var chunk_args = InlineArray[ChunkedAttnArgs, FULL_ATTN_MAX_CHUNKS](
                        fill=ChunkedAttnArgs())
                    comptime CHUNK_F32_STRIDE = FULL_HPG * (2 + C.HEAD_DIM_FULL)
                    for c in range(num_chunks):
                        var start = c * pgs_per_chunk
                        var end = min((c + 1) * pgs_per_chunk, num_pg)
                        chunk_args[c] = ChunkedAttnArgs(
                            q_i8_base=rv.scratch_addr(q_i8_prep_lease),
                            qi_biases_base=rv.scratch_addr(qi_biases_lease),
                            q_scales_base=rv.scratch_addr(q_scales_lease),
                            cache_base=rv.full_cache_base(full_idx),
                            kv_head=0,
                            start_pg=start,
                            end_pg=end,
                            partial_out=rv.scratch_addr(partial_lease) + c * CHUNK_F32_STRIDE * 4,
                            context_len=context_len)
                    pool.dispatch[ChunkedAttnArgs,
                        chunked_attn_kernel[C.HEAD_DIM_FULL, C.MAX_SEQ_LEN,
                            FULL_NKV_LOCAL, C.NUM_HEADS // Self.tp, FULL_HPG]](
                        UnsafePointer(to=chunk_args[0]), num_chunks)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                sample.attention.add(rnks.timed_parallel[do_full_chunk_attn](mp))

                # Phase 3: Merge partials + quantize (inline per rank)
                for rank in range(Self.tp):
                    var rv = rnks.view(rank)
                    var context_len = pos + 1
                    var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
                    var num_chunks = min(Int(self.main_pools[rank].capacity), FULL_ATTN_MAX_CHUNKS)
                    if num_chunks > num_pg:
                        num_chunks = num_pg
                    merge_and_quantize[C.HEAD_DIM_FULL, FULL_HPG](
                        F32Ptr(unsafe_from_address=rv.scratch_addr(partial_lease)),
                        num_chunks,
                        I8Ptr(unsafe_from_address=rv.scratch_addr(attn_qi_lease)),
                        F32Ptr(unsafe_from_address=rv.scratch_addr(attn_head_sc_lease)))

                partial_lease^.release()
                q_scales_lease^.release()
                qi_biases_lease^.release()
                q_i8_prep_lease^.release()

                # O projection
                @parameter
                def do_full_o_proj[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return int8_gemv_blocked[C.HIDDEN, Q_DIM_LOCAL_FULL, C.HEAD_DIM_FULL](
                        I8Ptr(unsafe_from_address=rv.scratch_addr(attn_qi_lease)),
                        U8Ptr(unsafe_from_address=flb + fl.o_proj),
                        F32Ptr(unsafe_from_address=rv.scratch_addr(attn_head_sc_lease)),
                        F32Ptr(unsafe_from_address=flb + fl.o_proj_sc),
                        F32Ptr(unsafe_from_address=flb + fl.colsums.o_colsum),
                        BF16Ptr(unsafe_from_address=rv.x_residual(seq_len).ptr),
                        seq_len, pool)
                sample.o_proj.add(rnks.timed_parallel[do_full_o_proj](mp))
                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                qk_lease^.release()

            var body = fl.body if is_full else sl.body
            var cs   = fl.colsums if is_full else sl.colsums

            # Allreduce + post-attn norm
            var t_attn_reduce0 = Int(perf_counter_ns())
            ring_allreduce[RV.X_RESIDUAL, Self.tp](
                rnks.x_residual_ptrs(seq_len), seq_len, mp)
            sample.attn_reduce.add(PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))

            @parameter
            def do_post_attn_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var nlb = rv.sliding_layer_base(sliding_idx) if not is_full else rv.full_layer_base(full_idx)
                var args = InlineArray[PostAttnNormArgs, 1](fill=PostAttnNormArgs(
                    rv.x_residual(seq_len).ptr,
                    nlb + body.post_attn_norm,
                    rv.x_main(seq_len).ptr, EPS))
                pool.dispatch[PostAttnNormArgs, post_attn_norm_kernel[C.HIDDEN]](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            sample.post_attn_norm.add(rnks.timed_parallel[do_post_attn_norm](mp))

            # =============================================================
            # FFN BLOCK
            # =============================================================

            var act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var act_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var act_work2_lease = self.scratch.borrow[Float32, C.HIDDEN]()
            var expert_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var expert_act_scale_lease = self.scratch.borrow[Float32, 1]()

            @parameter
            def do_router_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                return rmsnorm_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                    rv.x_main(seq_len).ptr, lb + body.router_scale,
                    rv.scratch_addr(act_i8_lease),
                    rv.scratch_addr(act_work_lease), rv.scratch_addr(act_scale_lease),
                    EPS,
                    seq_len, pool)
            sample.router_quantize.add(rnks.timed_parallel[do_router_quantize](mp))

            var router_logits_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.NUM_EXPERTS]()
            var routing_lease = self.scratch.borrow[Gemma4TopKResult[C.TOP_K], 1]()

            @parameter
            def do_router_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                return int8_gemv[C.NUM_EXPERTS, C.HIDDEN](
                    rv.scratch_addr(act_i8_lease),
                    lb + body.router_proj,
                    lb + cs.router_colsum,
                    lb + body.router_proj_sc,
                    rv.scratch_addr(router_logits_lease),
                    seq_len, rv.scratch_addr(act_scale_lease), pool)
            sample.router_proj.add(rnks.timed_parallel[do_router_gemv](mp))

            @parameter
            def do_router_topk[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var args = InlineArray[RouterTopkArgs, 1](fill=RouterTopkArgs(
                    BF16Ptr(unsafe_from_address=rv.scratch_addr(router_logits_lease)),
                    BF16Ptr(unsafe_from_address=lb + body.router_pes),
                    rv.scratch_addr(routing_lease)))
                pool.dispatch[RouterTopkArgs, router_topk_kernel[C.NUM_EXPERTS, C.TOP_K]](
                    UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            sample.router_topk.add(rnks.timed_parallel[do_router_topk](mp))

            var expert_qi_lease = self.scratch.borrow[Scalar[DType.int8], C.TOP_K * C.MOE_INTERMEDIATE]()
            var expert_blk_scale_lease = self.scratch.borrow[Float32, C.TOP_K * C.MOE_NUM_BLOCKS]()
            var expert_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
            var local_count_lease = self.scratch.borrow[Int32, 1]()

            # Dual-gamma quantize: dense + expert activations in one pass
            @parameter
            def do_ffn_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                return rmsnorm_dual_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                    rv.x_main(seq_len).ptr,
                    lb + body.pre_ffn_norm,
                    lb + body.pre_ffn_norm_2,
                    rv.scratch_addr(act_i8_lease),
                    rv.scratch_addr(expert_act_i8_lease),
                    rv.scratch_addr(act_work_lease),
                    rv.scratch_addr(act_work2_lease),
                    rv.scratch_addr(act_scale_lease),
                    rv.scratch_addr(expert_act_scale_lease),
                    EPS,
                    seq_len, pool)
            sample.ffn_quantize.add(rnks.timed_parallel[do_ffn_quantize](mp))

            # Expert phase 1
            @parameter
            def do_expert_phase1[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var routing = UnsafePointer[Gemma4TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(routing_lease))[]

                # Count local experts for this rank
                comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
                var expert_base = rank * experts_per_rank
                var lc = 0
                for s in range(C.TOP_K):
                    var eid = routing.indices[s]
                    if eid >= expert_base and eid < expert_base + experts_per_rank:
                        lc += 1
                UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(local_count_lease))[] = Int32(lc)

                return gemma4_moe_phase1[
                    C.MOE_INTERMEDIATE, C.HIDDEN, C.FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=rv.scratch_addr(expert_act_i8_lease)),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(expert_act_scale_lease)),
                    routing,
                    lb + body.experts_gate_up,
                    C.MOE_GATE_UP_FUSED * C.HIDDEN,
                    lb + body.experts_gate_up_sc,
                    C.MOE_GATE_UP_FUSED * 4,
                    lb + cs.experts_gu_colsum,
                    C.MOE_GATE_UP_FUSED * 4,
                    I8Ptr(unsafe_from_address=rv.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(expert_blk_scale_lease)),
                    rank, pool)
            sample.expert_phase1.add(rnks.timed_parallel[do_expert_phase1](mp))

            var dense_post_i8_lease = self.scratch.borrow[Scalar[DType.int8], DENSE_INT_LOCAL]()

            @parameter
            def do_dense_phase1[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                return fused_gu_gelu_tanh[DENSE_INT_LOCAL, C.HIDDEN, DBLK](
                    I8Ptr(unsafe_from_address=rv.scratch_addr(act_i8_lease)),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(act_scale_lease)),
                    U8Ptr(unsafe_from_address=lb + body.gate_proj),
                    F32Ptr(unsafe_from_address=lb + body.gate_proj_sc),
                    F32Ptr(unsafe_from_address=lb + cs.gu_colsum),
                    I8Ptr(unsafe_from_address=rv.scratch_addr(dense_post_i8_lease)),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(post_blk_scale_lease)),
                    seq_len, pool)
            sample.dense_phase1.add(rnks.timed_parallel[do_dense_phase1](mp))

            # Expert phase 2
            @parameter
            def do_expert_phase2[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var routing = UnsafePointer[Gemma4TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(routing_lease))[]
                return gemma4_moe_phase2[
                    C.HIDDEN, C.MOE_INTERMEDIATE, C.FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=rv.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(expert_blk_scale_lease)),
                    routing,
                    lb + body.experts_down,
                    C.HIDDEN * C.MOE_INTERMEDIATE,
                    lb + body.experts_down_sc,
                    C.HIDDEN * 4,
                    lb + cs.experts_down_colsum,
                    C.HIDDEN * C.MOE_NUM_BLOCKS * 4,
                    BF16Ptr(unsafe_from_address=rv.scratch_addr(expert_out_lease)),
                    rank, pool)
            sample.expert_phase2.add(rnks.timed_parallel[do_expert_phase2](mp))

            var dense_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_dense_phase2[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                return int8_gemv_blocked[C.HIDDEN, DENSE_INT_LOCAL, DBLK](
                    I8Ptr(unsafe_from_address=rv.scratch_addr(dense_post_i8_lease)),
                    U8Ptr(unsafe_from_address=lb + body.down_proj),
                    F32Ptr(unsafe_from_address=rv.scratch_addr(post_blk_scale_lease)),
                    F32Ptr(unsafe_from_address=lb + body.down_proj_sc),
                    F32Ptr(unsafe_from_address=lb + cs.down_colsum),
                    BF16Ptr(unsafe_from_address=rv.scratch_addr(dense_out_lease)),
                    seq_len, pool)
            sample.dense_phase2.add(rnks.timed_parallel[do_dense_phase2](mp))

            var dense_normed_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            # Expert sum → x_residual (before expert allreduce)
            @parameter
            def do_expert_sum[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lc = Int(UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(local_count_lease))[] )
                var args = InlineArray[ExpertSumArgs, 1](fill=ExpertSumArgs(
                    rv.scratch_addr(expert_out_lease), lc,
                    rv.x_residual(seq_len).ptr))
                pool.dispatch[ExpertSumArgs, expert_sum_kernel[C.HIDDEN]](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            sample.pre_reduce.add(rnks.timed_parallel[do_expert_sum](mp))

            # Dense allreduce (partial dense_out → full dense_out)
            var t_dense_reduce0 = Int(perf_counter_ns())
            var dense_out_ptrs = InlineArray[Int, Self.tp](fill=0)
            for r in range(Self.tp):
                dense_out_ptrs[r] = rnks.view(r).scratch_addr(dense_out_lease)
            ring_allreduce[RV.X_RESIDUAL, Self.tp](dense_out_ptrs, 1, mp)
            sample.mlp_reduce.add(PhaseTiming.opaque(Int(perf_counter_ns()) - t_dense_reduce0))

            # Dense norm (after allreduce, dense_out is now the full sum)
            @parameter
            def do_dense_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var args = InlineArray[DenseNormArgs, 1](fill=DenseNormArgs(
                    rv.scratch_addr(dense_out_lease),
                    lb + body.post_ffn_norm_1,
                    rv.scratch_addr(dense_normed_lease), EPS))
                pool.dispatch[DenseNormArgs, dense_norm_kernel[C.HIDDEN]](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            sample.pre_reduce.add(rnks.timed_parallel[do_dense_norm](mp))

            # Expert allreduce (expert sum across ranks)
            var t_mlp_reduce0 = Int(perf_counter_ns())
            ring_allreduce[RV.X_RESIDUAL, Self.tp](
                rnks.x_residual_ptrs(seq_len), seq_len, mp)
            sample.mlp_reduce.add(PhaseTiming.opaque(Int(perf_counter_ns()) - t_mlp_reduce0))

            # Post-reduce: norm experts + combine with dense_normed + residual
            @parameter
            def do_post_reduce[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var ls = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=lb + body.layer_scalar)
                var args = InlineArray[PostReduceArgs, 1](fill=PostReduceArgs(
                    rv.x_residual(seq_len).ptr,
                    lb + body.post_ffn_norm_2_rt,
                    rv.scratch_addr(dense_normed_lease),
                    lb + body.post_ffn_norm,
                    rv.x_main(seq_len).ptr, Float32(ls[]), EPS))
                pool.dispatch[PostReduceArgs, post_reduce_kernel[C.HIDDEN]](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            sample.post_reduce.add(rnks.timed_parallel[do_post_reduce](mp))
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

        # --- Final norm (fused with FWHT + per-block quantize) + LM head + softcap ---
        var last_hidden_bf16 = host.x_main(seq_len).ptr
        var lm_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
        var lm_act_blk_scale_lease = self.scratch.borrow[Float32, RV.VOCAB_NUM_BLOCKS]()
        var lm_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
        var t_final0 = Int(perf_counter_ns())
        var final_fence = rmsnorm_gamma_fwht_per_block_quantize[
            C.HIDDEN, C.LM_HEAD_FWHT_BLK](
            last_hidden_bf16,
            host.weight_base() + L.sqrt_gamma_off,
            host.scratch_addr(lm_act_i8_lease),
            host.scratch_addr(lm_work_lease),
            host.scratch_addr(lm_act_blk_scale_lease),
            EPS, 1, self.main_pools[0])
        var t_final1 = Int(perf_counter_ns())
        sample.final_norm = finish_single_pool_fence(t_final0, t_final1, final_fence^)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = host.scratch_view[RV.LOGITS](logit_lease, 1)
        var t_lm0 = Int(perf_counter_ns())
        var lm_fence = lm_head_gemv[
            C.VOCAB_SIZE, C.HIDDEN, C.LM_HEAD_FWHT_BLK](
            host.scratch_addr(lm_act_i8_lease),
            host.weight_base() + L.embed_off,
            host.scratch_addr(lm_act_blk_scale_lease),
            host.weight_base() + L.embed_sc_off,
            host.weight_base() + L.embed_colsum_off,
            logit_view.ptr,
            self.main_pools[0])
        var t_lm1 = Int(perf_counter_ns())
        sample.lm_head = finish_single_pool_fence(t_lm0, t_lm1, lm_fence^)
        lm_work_lease^.release()
        lm_act_blk_scale_lease^.release()
        lm_act_i8_lease^.release()
        var t_softcap0 = Int(perf_counter_ns())
        logit_softcap(logit_view)
        sample.softcap = PhaseTiming.opaque(Int(perf_counter_ns()) - t_softcap0)
        sample.wall_ns = Int(perf_counter_ns()) - t_forward0
        self.profile.record(sample)
        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^)


def main():
    print("gemma_4_moe_butterquant_tp: module")
