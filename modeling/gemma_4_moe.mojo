"""Gemma 4 26B-A4B — bf16 reference model.

Text decoder weights only (vision/audio encoder weights ignored).

Layout: [25 sliding layers][5 full layers][state][host-only: final_norm, embed]
State:  [sliding KV caches][full KV caches][x_main][x_residual][scratch][rope tables]
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from simd_math import sqrt
from numa import NumaArena, NumaInfo
from threading import BurstPool
from experimental.linear_borrow_pool import ScratchPool, ScratchLease

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32,
    HOST_RANK, DISTRIBUTED,
    Mat, Bound, DynView, CacheView,
    Shape, ShapeLike, WeightDesc,
    DEFAULT_ALIGNMENT, LogitsView,
)
from modeling.loader import discover_shards, load_weights_from_descs

from kernels.kernel_ops import (
    gemm, rmsnorm, elem_add, kv_cache_write,
    gemv_kernel, GemmArgs,
    BF16Ptr,
)
from kernels.kv_rotors import rope
from threading.threading_shared import ptr as tptr

from experimental_gemma.activations import gelu_tanh_mul
from experimental_gemma.norms import rmsnorm_no_scale, rmsnorm_per_head
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables, apply_full_rope
from experimental_gemma.router import softmax_topk_renorm
from experimental_gemma.moe import gemma4_moe_dispatch
from experimental_gemma.attention import local_attention, global_attention
from experimental_gemma.ops import embed_lookup_scaled, logit_softcap, elem_scale


# =============================================================================
# Config
# =============================================================================


struct Gemma4Config:
    comptime HIDDEN = 2816
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 16

    # Sliding attention geometry
    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime Q_DIM_SLIDING = 4096
    comptime KV_DIM_SLIDING = 2048

    # Full/global attention geometry
    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime Q_DIM_FULL = 8192
    comptime KV_DIM_FULL = 1024

    # Dense MLP
    comptime INTERMEDIATE = 2112

    # MoE
    comptime MOE_INTERMEDIATE = 704
    comptime MOE_GATE_UP_FUSED = 1408
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    # Vocab
    comptime VOCAB_SIZE = 262144

    # Layer counts by type
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5

    # Sequence and numerics
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime EMBED_SCALE = sqrt[DType.float32, 1](Self.HIDDEN)
    comptime LOGIT_SOFTCAP = 30.0


comptime C = Gemma4Config


# =============================================================================
# Per-TP shape aliases
# =============================================================================


struct Gemma4Shapes[tp: Int]:
    comptime GateUp    = Shape[C.INTERMEDIATE, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime Down      = Shape[C.HIDDEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp]

    comptime SlidingQ  = Shape[C.Q_DIM_SLIDING, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingKV = Shape[C.KV_DIM_SLIDING, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime SlidingO  = Shape[C.HIDDEN, C.Q_DIM_SLIDING, shard_m=True, tp=Self.tp]

    comptime FullQ     = Shape[C.Q_DIM_FULL, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullK     = Shape[C.KV_DIM_FULL, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime FullO     = Shape[C.HIDDEN, C.Q_DIM_FULL, shard_m=True, tp=Self.tp]

    comptime ExpertsGateUp = Shape[C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime ExpertsDown   = Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE, shard_n=True, tp=Self.tp]


# =============================================================================
# Runtime layout
# =============================================================================


struct LayerShard:
    comptime ROW  = 0
    comptime COL  = 1
    comptime REPL = 2
    comptime HOST = 3


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
            shard: Int) -> Int:
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
            quantizable=False, absorbed=False,
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
    def bfs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String) -> Int:
        return self.emit_shape[S, 2](entries, suffix, DType.bfloat16, False)

    @always_inline
    def bf(mut self, mut entries: List[WeightDesc], suffix: String,
           rows: Int, cols: Int, shard: Int) -> Int:
        return self.emit(entries, suffix, rows, cols, DType.bfloat16, 2, shard)


@fieldwise_init
struct LayerBodyWeights(Copyable, ImplicitlyCopyable, Movable):
    var input_norm: Int
    var post_attn_norm: Int
    var pre_ffn_norm: Int
    var pre_ffn_norm_2: Int
    var post_ffn_norm_1: Int
    var post_ffn_norm_2: Int
    var post_ffn_norm: Int
    var gate_proj: Int
    var up_proj: Int
    var down_proj: Int
    var router_proj: Int
    var router_scale: Int
    var router_pes: Int
    var experts_gate_up: Int
    var experts_down: Int
    var layer_scalar: Int


@fieldwise_init
struct SlidingLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    var q_proj: Int
    var k_proj: Int
    var v_proj: Int
    var o_proj: Int
    var q_norm: Int
    var k_norm: Int
    var body: LayerBodyWeights
    var stride: Int


@fieldwise_init
struct FullLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    var q_proj: Int
    var k_proj: Int
    var o_proj: Int
    var q_norm: Int
    var k_norm: Int
    var body: LayerBodyWeights
    var stride: Int


def emit_layer_body[tp: Int](
    mut b: LayerBuilder, mut entries: List[WeightDesc],
) -> LayerBodyWeights:
    comptime REPL = LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime NE  = C.NUM_EXPERTS
    comptime S = Gemma4Shapes[tp]

    var input_norm      = b.bf(entries, "input_layernorm.weight",               H, 1, REPL)
    var post_attn_norm  = b.bf(entries, "post_attention_layernorm.weight",      H, 1, REPL)
    var pre_ffn_norm    = b.bf(entries, "pre_feedforward_layernorm.weight",     H, 1, REPL)
    var pre_ffn_norm_2  = b.bf(entries, "pre_feedforward_layernorm_2.weight",   H, 1, REPL)
    var post_ffn_norm_1 = b.bf(entries, "post_feedforward_layernorm_1.weight",  H, 1, REPL)
    var post_ffn_norm_2 = b.bf(entries, "post_feedforward_layernorm_2.weight",  H, 1, REPL)
    var post_ffn_norm   = b.bf(entries, "post_feedforward_layernorm.weight",    H, 1, REPL)
    var gate_proj       = b.bfs[S.GateUp](entries, "mlp.gate_proj.weight")
    var up_proj         = b.bfs[S.GateUp](entries, "mlp.up_proj.weight")
    var down_proj       = b.bfs[S.Down](entries, "mlp.down_proj.weight")
    var router_proj     = b.bf(entries, "router.proj.weight",      NE, H, REPL)
    var router_scale    = b.bf(entries, "router.scale",            H,  1, REPL)
    var router_pes      = b.bf(entries, "router.per_expert_scale", NE, 1, REPL)
    var experts_gate_up = b.bfs[S.ExpertsGateUp](entries, "experts.gate_up_proj")
    var experts_down    = b.bfs[S.ExpertsDown](entries, "experts.down_proj")
    var layer_scalar    = b.bf(entries, "layer_scalar",            1,  1, REPL)

    return LayerBodyWeights(
        input_norm=input_norm, post_attn_norm=post_attn_norm,
        pre_ffn_norm=pre_ffn_norm, pre_ffn_norm_2=pre_ffn_norm_2,
        post_ffn_norm_1=post_ffn_norm_1, post_ffn_norm_2=post_ffn_norm_2,
        post_ffn_norm=post_ffn_norm,
        gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj,
        router_proj=router_proj, router_scale=router_scale, router_pes=router_pes,
        experts_gate_up=experts_gate_up, experts_down=experts_down,
        layer_scalar=layer_scalar,
    )


def sliding_layer_spec[tp: Int](
    prefix: String, layer_base: Int, mut entries: List[WeightDesc],
) -> SlidingLayerOffsets:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime REPL = LayerShard.REPL
    comptime S = Gemma4Shapes[tp]

    var q_proj = b.bfs[S.SlidingQ](entries, "self_attn.q_proj.weight")
    var k_proj = b.bfs[S.SlidingKV](entries, "self_attn.k_proj.weight")
    var v_proj = b.bfs[S.SlidingKV](entries, "self_attn.v_proj.weight")
    var o_proj = b.bfs[S.SlidingO](entries, "self_attn.o_proj.weight")
    var q_norm = b.bf(entries, "self_attn.q_norm.weight", C.HEAD_DIM_SLIDING, 1, REPL)
    var k_norm = b.bf(entries, "self_attn.k_norm.weight", C.HEAD_DIM_SLIDING, 1, REPL)
    var body = emit_layer_body[tp](b, entries)

    return SlidingLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
        q_norm=q_norm, k_norm=k_norm,
        body=body, stride=b.cursor,
    )


def full_layer_spec[tp: Int](
    prefix: String, layer_base: Int, mut entries: List[WeightDesc],
) -> FullLayerOffsets:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime REPL = LayerShard.REPL
    comptime S = Gemma4Shapes[tp]

    var q_proj = b.bfs[S.FullQ](entries, "self_attn.q_proj.weight")
    var k_proj = b.bfs[S.FullK](entries, "self_attn.k_proj.weight")
    var o_proj = b.bfs[S.FullO](entries, "self_attn.o_proj.weight")
    var q_norm = b.bf(entries, "self_attn.q_norm.weight", C.HEAD_DIM_FULL, 1, REPL)
    var k_norm = b.bf(entries, "self_attn.k_norm.weight", C.HEAD_DIM_FULL, 1, REPL)
    var body = emit_layer_body[tp](b, entries)

    return FullLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, o_proj=o_proj,
        q_norm=q_norm, k_norm=k_norm,
        body=body, stride=b.cursor,
    )


# =============================================================================
# Whole-model runtime layout
# =============================================================================


@fieldwise_init
struct Gemma4ModelLayout(Copyable, ImplicitlyCopyable, Movable):
    var sliding: SlidingLayerOffsets
    var full: FullLayerOffsets
    var sliding_off: Int
    var sliding_stride: Int
    var full_off: Int
    var full_stride: Int
    var distributed_bytes: Int
    var sliding_kv_off: Int
    var sliding_kv_stride: Int
    var full_kv_off: Int
    var full_kv_stride: Int
    var x_main_off: Int
    var x_residual_off: Int
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_cos_off: Int
    var sliding_sin_off: Int
    var full_cos_off: Int
    var full_sin_off: Int
    var state_bytes: Int
    var host_only_off: Int
    var final_norm_off: Int
    var embed_off: Int

    @always_inline
    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    @always_inline
    def host_arena_bytes(self) -> Int:
        return self.embed_off + C.VOCAB_SIZE * C.HIDDEN * BF16.ELEMENT_BYTES


@fieldwise_init
struct Gemma4LoadPlan(Movable):
    var layout: Gemma4ModelLayout
    var descs: List[WeightDesc]


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime bf16 = BF16.ELEMENT_BYTES
    comptime S = C.MAX_SEQ_LEN
    comptime Sh = Gemma4Shapes[tp]

    comptime full_attn_borrows = (
        S * Sh.FullQ.N * bf16 +
        S * Sh.FullK.N * bf16 +
        S * Sh.FullK.N * bf16 +
        S * Sh.FullQ.N * bf16
    )
    comptime sliding_attn_borrows = (
        S * Sh.SlidingQ.N * bf16 +
        S * Sh.SlidingKV.N * bf16 +
        S * Sh.SlidingKV.N * bf16 +
        S * Sh.SlidingQ.N * bf16
    )
    comptime ffn_borrows_dense = S * Sh.GateUp.N * bf16 * 2
    comptime ffn_borrows_moe = (
        S * C.HIDDEN * bf16 +
        S * C.HIDDEN * bf16 +
        C.TOP_K * C.HIDDEN * bf16
    )
    comptime ffn_peak = ffn_borrows_dense if ffn_borrows_dense > ffn_borrows_moe else ffn_borrows_moe
    comptime attn_peak = full_attn_borrows if full_attn_borrows > sliding_attn_borrows else sliding_attn_borrows
    return ffn_peak if ffn_peak > attn_peak else attn_peak


def build_gemma4_load_plan[tp: Int]() -> Gemma4LoadPlan:
    var descs = List[WeightDesc]()

    # Template pass: probe strides and canonical per-layer offsets.
    var scratch = List[WeightDesc]()
    var sliding_offsets = sliding_layer_spec[tp]("", 0, scratch)
    var full_offsets    = full_layer_spec[tp]("", 0, scratch)
    var sliding_stride  = sliding_offsets.stride
    var full_stride     = full_offsets.stride

    var sliding_off       = 0
    var full_off          = sliding_off + C.NUM_SLIDING_LAYERS * sliding_stride
    var distributed_bytes = full_off + C.NUM_FULL_LAYERS * full_stride

    # Real pass: emit entries for every layer in layer-index order.
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

    # State block layout: [sliding KV][full KV][x_main][x_residual][scratch][rope].
    comptime bf16 = size_of[Scalar[DType.bfloat16]]()
    comptime f32  = size_of[Float32]()
    comptime KV_S  = C.MAX_SEQ_LEN * C.KV_DIM_SLIDING * bf16   # k or v cache, one of
    comptime KV_F  = C.MAX_SEQ_LEN * C.KV_DIM_FULL    * bf16
    comptime sliding_kv_stride = 2 * KV_S                      # k + v per layer
    comptime full_kv_stride    = 2 * KV_F
    var sliding_kv_off = 0
    var full_kv_off    = sliding_kv_off + C.NUM_SLIDING_LAYERS * sliding_kv_stride
    var x_main_off     = full_kv_off + C.NUM_FULL_LAYERS * full_kv_stride
    var x_main_bytes   = C.MAX_SEQ_LEN * C.HIDDEN * bf16
    var x_residual_off = x_main_off + x_main_bytes
    var scratch_off    = x_residual_off + x_main_bytes
    var scratch_capacity = calculate_peak_scratch[tp]()

    comptime sliding_rope_half = C.HEAD_DIM_SLIDING // 2
    comptime full_rope_half = 64
    var sliding_cos_off = scratch_off + scratch_capacity
    var sliding_cos_bytes = C.MAX_SEQ_LEN * sliding_rope_half * f32
    var sliding_sin_off = sliding_cos_off + sliding_cos_bytes
    var full_cos_off    = sliding_sin_off + sliding_cos_bytes
    var full_cos_bytes  = C.MAX_SEQ_LEN * full_rope_half * f32
    var full_sin_off    = full_cos_off + full_cos_bytes
    var state_bytes     = full_sin_off + full_cos_bytes

    # Host-only: final_norm + tied embed table, pinned to rank 0.
    var host_only_off = ((distributed_bytes + state_bytes + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime HOST = LayerShard.HOST
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_only_off
    var final_norm_off = hb.bf(descs, "model.language_model.norm.weight",
                               C.HIDDEN, 1,               HOST)
    var embed_off      = hb.bf(descs, "model.language_model.embed_tokens.weight",
                               C.VOCAB_SIZE, C.HIDDEN,    HOST)

    var layout = Gemma4ModelLayout(
        sliding=sliding_offsets, full=full_offsets,
        sliding_off=sliding_off, sliding_stride=sliding_stride,
        full_off=full_off, full_stride=full_stride,
        distributed_bytes=distributed_bytes,
        sliding_kv_off=sliding_kv_off, sliding_kv_stride=sliding_kv_stride,
        full_kv_off=full_kv_off, full_kv_stride=full_kv_stride,
        x_main_off=x_main_off, x_residual_off=x_residual_off,
        scratch_off=scratch_off, scratch_capacity=scratch_capacity,
        sliding_cos_off=sliding_cos_off, sliding_sin_off=sliding_sin_off,
        full_cos_off=full_cos_off, full_sin_off=full_sin_off,
        state_bytes=state_bytes,
        host_only_off=host_only_off,
        final_norm_off=final_norm_off,
        embed_off=embed_off,
    )
    return Gemma4LoadPlan(layout, descs^)


# =============================================================================
# Rank view
# =============================================================================


@fieldwise_init
struct RankView[tp: Int](Copyable, Movable):
    comptime S = Gemma4Shapes[Self.tp]

    # Activations / state — always full size.
    comptime X_MAIN          = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]
    comptime X_RESIDUAL      = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]
    comptime HIDDEN_VIEW     = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]
    comptime LOGITS_VIEW     = Mat[BF16, C.MAX_SEQ_LEN, C.VOCAB_SIZE]

    # Dense MLP scratch — local intermediate dim.
    comptime MLP_VIEW        = Mat[BF16, C.MAX_SEQ_LEN, Self.S.GateUp.N]

    # RoPE tables.
    comptime SLIDING_ROPE_HALF = C.HEAD_DIM_SLIDING // 2
    comptime FULL_ROPE_HALF    = 64
    comptime SLIDING_COS     = Mat[F32, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF]
    comptime SLIDING_SIN     = Mat[F32, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF]
    comptime FULL_COS        = Mat[F32, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF]
    comptime FULL_SIN        = Mat[F32, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF]

    # KV caches.
    comptime SLIDING_K_CACHE = Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]
    comptime SLIDING_V_CACHE = Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING]
    comptime FULL_K_CACHE    = Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]
    comptime FULL_V_CACHE    = Mat[BF16, C.MAX_SEQ_LEN, C.KV_DIM_FULL]
    comptime SLIDING_KV_BYTES = C.MAX_SEQ_LEN * C.KV_DIM_SLIDING * BF16.ELEMENT_BYTES
    comptime FULL_KV_BYTES    = C.MAX_SEQ_LEN * C.KV_DIM_FULL * BF16.ELEMENT_BYTES

    # Scratch-view shapes — projection outputs, Shape-derived.
    comptime Q_SLIDING_VIEW  = Mat[BF16, C.MAX_SEQ_LEN, Self.S.SlidingQ.N]
    comptime KV_SLIDING_VIEW = Mat[BF16, C.MAX_SEQ_LEN, Self.S.SlidingKV.N]
    comptime Q_FULL_VIEW     = Mat[BF16, C.MAX_SEQ_LEN, Self.S.FullQ.N]
    comptime KV_FULL_VIEW    = Mat[BF16, C.MAX_SEQ_LEN, Self.S.FullK.N]

    # Weight shapes — norms (replicated).
    comptime NORM_W          = Mat[BF16, C.HIDDEN, 1]
    comptime SLIDING_HEAD_NORM = Mat[BF16, C.HEAD_DIM_SLIDING, 1]
    comptime FULL_HEAD_NORM  = Mat[BF16, C.HEAD_DIM_FULL, 1]
    comptime FINAL_NORM      = Mat[BF16, C.HIDDEN, 1]

    # Weight shapes — projections, Shape-derived.
    comptime FFN_GATE_W      = Mat[BF16, Self.S.GateUp.N, Self.S.GateUp.M]
    comptime FFN_UP_W        = Mat[BF16, Self.S.GateUp.N, Self.S.GateUp.M]
    comptime FFN_DOWN_W      = Mat[BF16, Self.S.Down.N, Self.S.Down.M]
    comptime SLIDING_Q_PROJ  = Mat[BF16, Self.S.SlidingQ.N, Self.S.SlidingQ.M]
    comptime SLIDING_KV_PROJ = Mat[BF16, Self.S.SlidingKV.N, Self.S.SlidingKV.M]
    comptime SLIDING_O_PROJ  = Mat[BF16, Self.S.SlidingO.N, Self.S.SlidingO.M]
    comptime FULL_Q_PROJ     = Mat[BF16, Self.S.FullQ.N, Self.S.FullQ.M]
    comptime FULL_K_PROJ     = Mat[BF16, Self.S.FullK.N, Self.S.FullK.M]
    comptime FULL_O_PROJ     = Mat[BF16, Self.S.FullO.N, Self.S.FullO.M]

    # Host-only.
    comptime EMBED           = Mat[BF16, C.VOCAB_SIZE, C.HIDDEN]

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

    # Activations.
    def x_main(self, seq_len: Int) -> DynView[Self.X_MAIN]:
        return DynView[Self.X_MAIN](self.state_base() + self.L().x_main_off, seq_len)
    def x_residual(self, seq_len: Int) -> DynView[Self.X_RESIDUAL]:
        return DynView[Self.X_RESIDUAL](self.state_base() + self.L().x_residual_off, seq_len)

    # Scratch views.
    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        return DynView[V](self.scratch_base() + lease.offset, seq_len)
    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)

    # KV caches. K cache first, V cache right after (stride is 2*kv_bytes per layer).
    def sliding_k_cache(self, sliding_idx: Int) -> CacheView[Self.SLIDING_K_CACHE]:
        return CacheView[Self.SLIDING_K_CACHE](
            self.state_base() + self.L().sliding_kv_off + sliding_idx * self.L().sliding_kv_stride)
    def sliding_v_cache(self, sliding_idx: Int) -> CacheView[Self.SLIDING_V_CACHE]:
        return CacheView[Self.SLIDING_V_CACHE](
            self.state_base() + self.L().sliding_kv_off + sliding_idx * self.L().sliding_kv_stride
            + Self.SLIDING_KV_BYTES)
    def full_k_cache(self, full_idx: Int) -> CacheView[Self.FULL_K_CACHE]:
        return CacheView[Self.FULL_K_CACHE](
            self.state_base() + self.L().full_kv_off + full_idx * self.L().full_kv_stride)
    def full_v_cache(self, full_idx: Int) -> CacheView[Self.FULL_V_CACHE]:
        return CacheView[Self.FULL_V_CACHE](
            self.state_base() + self.L().full_kv_off + full_idx * self.L().full_kv_stride
            + Self.FULL_KV_BYTES)

    # RoPE tables.
    def sliding_cos(self) -> Bound[Self.SLIDING_COS]:
        return Bound[Self.SLIDING_COS](self.state_base() + self.L().sliding_cos_off)
    def sliding_sin(self) -> Bound[Self.SLIDING_SIN]:
        return Bound[Self.SLIDING_SIN](self.state_base() + self.L().sliding_sin_off)
    def full_cos(self) -> Bound[Self.FULL_COS]:
        return Bound[Self.FULL_COS](self.state_base() + self.L().full_cos_off)
    def full_sin(self) -> Bound[Self.FULL_SIN]:
        return Bound[Self.FULL_SIN](self.state_base() + self.L().full_sin_off)

    # Per-layer weight base addresses.
    def sliding_base(self, sliding_idx: Int) -> Int:
        return self.weight_base() + self.L().sliding_off + sliding_idx * self.L().sliding_stride
    def full_base(self, full_idx: Int) -> Int:
        return self.weight_base() + self.L().full_off + full_idx * self.L().full_stride

    # Host-only.
    def final_norm(self) -> Bound[Self.FINAL_NORM]:
        return Bound[Self.FINAL_NORM](self.weight_base() + self.L().final_norm_off)
    def embed_table(self) -> Bound[Self.EMBED]:
        return Bound[Self.EMBED](self.weight_base() + self.L().embed_off)


# =============================================================================
# Loaded model
# =============================================================================


struct Gemma4[tp: Int](Movable):
    var arena: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool: BurstPool[]
    var scratch: ScratchPool
    var base: Int
    var layout: Gemma4ModelLayout

    def __init__(out self,
        var arena: NumaArena[alignment=DEFAULT_ALIGNMENT],
        var pool: BurstPool[],
        layout: Gemma4ModelLayout,
    ):
        self.base = Int(arena.base)
        self.arena = arena^
        self.pool = pool^
        self.layout = layout
        self.scratch = ScratchPool(layout.scratch_capacity)

    def view(mut self) -> RankView[Self.tp]:
        return RankView[Self.tp](self.base,
            UnsafePointer[Gemma4ModelLayout, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.layout))))

    def init_state(mut self):
        """Initialize RoPE tables and bake router constants."""
        var s = self.view()
        var L = self.layout

        # RoPE tables.
        init_sliding_rope_tables(s.sliding_cos(), s.sliding_sin())
        init_full_rope_tables(s.full_cos(), s.full_sin())

        # Bake router constant: router_scale *= 1/sqrt(hidden). One-time
        # rewrite of the bf16 gamma so the router-input rmsnorm absorbs
        # the 1/sqrt(H) without paying for it on every forward.
        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        comptime width = simd_width_of[DType.float32]()
        var sliding_idx = 0
        var full_idx = 0
        for i in range(C.NUM_LAYERS):
            var scale_ptr: BF16Ptr
            if (i + 1) % 6 == 0:
                scale_ptr = BF16Ptr(unsafe_from_address=s.full_base(full_idx) + L.full.body.router_scale)
                full_idx += 1
            else:
                scale_ptr = BF16Ptr(unsafe_from_address=s.sliding_base(sliding_idx) + L.sliding.body.router_scale)
                sliding_idx += 1
            for j in range(0, C.HIDDEN, width):
                var v = (scale_ptr + j).load[width=width]().cast[DType.float32]()
                (scale_ptr + j).store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        print("state initialized: rope tables + baked router constants")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        # Build the whole-model runtime layout and weight catalog in one
        # pass. Single source of truth for offsets + descs — they can't
        # drift because each sliding/full spec emit produces both.
        var plan = build_gemma4_load_plan[Self.tp]()

        var numa = NumaInfo()
        var topo = numa.plan_topology(1)

        var size = plan.layout.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB (" +
              String(plan.layout.distributed_bytes // (1024 * 1024)) + " MB weights + " +
              String(plan.layout.state_bytes // (1024 * 1024)) + " MB state)")
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[0], size)
        if not arena:
            print("arena allocation failed")
            return None

        var arena_bases = List[Int]()
        arena_bases.append(Int(arena.base))

        var result = load_weights_from_descs(plan.descs, shards, arena_bases)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        _ = arena.prefault(plan.layout.distributed_bytes, plan.layout.state_bytes)

        var pool = BurstPool[].for_numa_node(numa, topo[0])
        var model = Self(arena^, pool^, plan.layout)
        model.init_state()
        return model^

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.view().scratch_base())

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime SV = RankView[Self.tp]
        comptime S = Gemma4Shapes[Self.tp]
        var s = self.view()
        var L = self.layout
        var sl = L.sliding
        var fl = L.full

        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        # --- Embed ---
        embed_lookup_scaled(s.embed_table(), tokens_ptr, s.x_main(seq_len),
            C.EMBED_SCALE, self.pool).join()

        # --- Layer loop ---
        var sliding_idx = 0
        var full_idx = 0

        for layer_idx in range(C.NUM_LAYERS):
            var is_full = (layer_idx + 1) % 6 == 0

            # Layer weight base (kind-specific stride).
            var lb = s.full_base(full_idx) if is_full else s.sliding_base(sliding_idx)
            # Shared-body offsets: bound once per layer, avoids per-field ternaries
            # in every emit below.
            var body = fl.body if is_full else sl.body

            # =====================
            # ATTENTION BLOCK
            # =====================

            if is_full:
                self.attention_full(s, full_idx, seq_len, pos)
            else:
                self.attention_sliding(s, sliding_idx, seq_len, pos)

            # =====================
            # FEEDFORWARD BLOCK
            # =====================

            # --- Post-attention norm + residual add ---
            rmsnorm(s.x_residual(seq_len), Bound[SV.NORM_W](lb + body.post_attn_norm),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
            elem_add(s.x_main(seq_len), s.x_residual(seq_len), s.x_main(seq_len))

            # --- Dense MLP path ---
            rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + body.pre_ffn_norm),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var gate_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.GateUp.N]()
            var up_lease   = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.GateUp.N]()

            gemm(s.x_residual(seq_len), Bound[SV.FFN_GATE_W](lb + body.gate_proj),
                s.scratch_view[SV.MLP_VIEW](gate_lease, seq_len), self.pool).join()
            gemm(s.x_residual(seq_len), Bound[SV.FFN_UP_W](lb + body.up_proj),
                s.scratch_view[SV.MLP_VIEW](up_lease, seq_len), self.pool).join()

            gelu_tanh_mul(s.scratch_view[SV.MLP_VIEW](gate_lease, seq_len),
                s.scratch_view[SV.MLP_VIEW](up_lease, seq_len),
                s.scratch_view[SV.MLP_VIEW](gate_lease, seq_len))

            up_lease^.release()

            gemm(s.scratch_view[SV.MLP_VIEW](gate_lease, seq_len),
                Bound[SV.FFN_DOWN_W](lb + body.down_proj),
                s.x_residual(seq_len), self.pool).join()

            gate_lease^.release()

            # dense_normed = rmsnorm(dense_out, post_ffn_norm_1)
            var dense_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            rmsnorm(s.x_residual(seq_len), Bound[SV.NORM_W](lb + body.post_ffn_norm_1),
                s.scratch_view[SV.HIDDEN_VIEW](dense_lease, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            # --- MoE path ---
            # Router: rmsnorm with baked scale (already includes 1/sqrt(hidden)).
            var router_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + body.router_scale),
                s.scratch_view[SV.HIDDEN_VIEW](router_lease, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            # Router projection → softmax_topk_renorm (per-token, inline for decode).
            var router_logits_buf = InlineArray[Scalar[DType.bfloat16], C.NUM_EXPERTS](fill=Scalar[DType.bfloat16](0))
            var router_logits_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=router_logits_buf[0])))
            var router_input_ptr  = s.scratch_ptr[Scalar[DType.bfloat16]](router_lease)

            gemv_kernel[C.HIDDEN, C.NUM_EXPERTS](GemmArgs(
                router_input_ptr,
                BF16Ptr(unsafe_from_address=lb + body.router_proj),
                router_logits_ptr, 0, C.NUM_EXPERTS, 1))

            var routing = softmax_topk_renorm[C.NUM_EXPERTS, C.TOP_K](
                router_logits_ptr,
                BF16Ptr(unsafe_from_address=lb + body.router_pes))

            router_lease^.release()

            # MoE FFN: norm → expert dispatch.
            rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + body.pre_ffn_norm_2),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var moe_lease        = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            var expert_buf_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()

            gemma4_moe_dispatch[C.NUM_EXPERTS, C.TOP_K, C.MOE_INTERMEDIATE, C.HIDDEN](
                tptr[Scalar[DType.bfloat16]](s.x_residual(seq_len).ptr),
                routing,
                BF16Ptr(unsafe_from_address=lb + body.experts_gate_up),
                BF16Ptr(unsafe_from_address=lb + body.experts_down),
                s.scratch_ptr[Scalar[DType.bfloat16]](expert_buf_lease),
                s.scratch_ptr[Scalar[DType.bfloat16]](moe_lease),
                self.pool)

            expert_buf_lease^.release()

            # moe_normed = rmsnorm(moe_out, post_ffn_norm_2)
            rmsnorm(s.scratch_view[SV.HIDDEN_VIEW](moe_lease, seq_len),
                Bound[SV.NORM_W](lb + body.post_ffn_norm_2),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            moe_lease^.release()

            # --- Combine dense + MoE, final post-ffn norm, layer scalar ---
            elem_add(s.scratch_view[SV.HIDDEN_VIEW](dense_lease, seq_len),
                s.x_residual(seq_len), s.x_residual(seq_len))
            dense_lease^.release()

            rmsnorm(s.x_residual(seq_len), Bound[SV.NORM_W](lb + body.post_ffn_norm),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var layer_scalar = Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=lb + body.layer_scalar)[])
            elem_add(s.x_main(seq_len), s.x_residual(seq_len), s.x_main(seq_len))
            elem_scale(s.x_main(seq_len), layer_scalar)

            if is_full:
                full_idx += 1
            else:
                sliding_idx += 1

        # --- Final norm + LM head ---
        rmsnorm(s.x_main(seq_len), s.final_norm(),
            s.x_main(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        var last_row_off = (seq_len - 1) * C.HIDDEN * SV.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[SV.X_MAIN](s.x_main(seq_len).ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = s.scratch_view[SV.LOGITS_VIEW](logit_lease, 1)
        gemm(last_hidden, s.embed_table(), logit_view, self.pool).join()

        logit_softcap(logit_view)

        return LogitsView[C.VOCAB_SIZE](
            s.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^)

    # =========================================================================
    # Attention helpers (separate methods to avoid massive forward body)
    # =========================================================================

    def attention_sliding(mut self, s: RankView[Self.tp], sliding_idx: Int, seq_len: Int, pos: Int):
        comptime SV = RankView[Self.tp]
        comptime S = Gemma4Shapes[Self.tp]
        var lb = s.sliding_base(sliding_idx)
        var sl = self.layout.sliding

        # Input norm → x_residual
        rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + sl.body.input_norm),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K, V projections.
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingQ.N]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingKV.N]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingKV.N]()

        gemm(s.x_residual(seq_len), Bound[SV.SLIDING_Q_PROJ](lb + sl.q_proj),
            s.scratch_view[SV.Q_SLIDING_VIEW](q, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), Bound[SV.SLIDING_KV_PROJ](lb + sl.k_proj),
            s.scratch_view[SV.KV_SLIDING_VIEW](k, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), Bound[SV.SLIDING_KV_PROJ](lb + sl.v_proj),
            s.scratch_view[SV.KV_SLIDING_VIEW](v, seq_len), self.pool).join()

        # Per-head norms.
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            s.scratch_view[SV.Q_SLIDING_VIEW](q, seq_len),
            Bound[SV.SLIDING_HEAD_NORM](lb + sl.q_norm),
            s.scratch_view[SV.Q_SLIDING_VIEW](q, seq_len),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            s.scratch_view[SV.KV_SLIDING_VIEW](k, seq_len),
            Bound[SV.SLIDING_HEAD_NORM](lb + sl.k_norm),
            s.scratch_view[SV.KV_SLIDING_VIEW](k, seq_len),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(s.scratch_view[SV.KV_SLIDING_VIEW](v, seq_len),
            s.scratch_view[SV.KV_SLIDING_VIEW](v, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # RoPE (Q and K only).
        rope[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            s.scratch_view[SV.Q_SLIDING_VIEW](q, seq_len), s.sliding_cos(), s.sliding_sin(), pos)
        rope[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            s.scratch_view[SV.KV_SLIDING_VIEW](k, seq_len), s.sliding_cos(), s.sliding_sin(), pos)

        # KV cache write.
        kv_cache_write(s.scratch_view[SV.KV_SLIDING_VIEW](k, seq_len), s.sliding_k_cache(sliding_idx), pos)
        kv_cache_write(s.scratch_view[SV.KV_SLIDING_VIEW](v, seq_len), s.sliding_v_cache(sliding_idx), pos)

        v^.release()
        k^.release()

        # Attention.
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.SlidingQ.N]()
        local_attention[C.NUM_HEADS, C.NUM_KV_HEADS_SLIDING, C.HEAD_DIM_SLIDING, C.SLIDING_WINDOW](
            s.scratch_view[SV.Q_SLIDING_VIEW](q, seq_len),
            s.sliding_k_cache(sliding_idx), s.sliding_v_cache(sliding_idx),
            s.scratch_view[SV.Q_SLIDING_VIEW](attn_out, seq_len), pos, self.pool).join()

        # O projection → x_residual.
        gemm(s.scratch_view[SV.Q_SLIDING_VIEW](attn_out, seq_len),
            Bound[SV.SLIDING_O_PROJ](lb + sl.o_proj),
            s.x_residual(seq_len), self.pool).join()

        attn_out^.release()
        q^.release()

    def attention_full(mut self, s: RankView[Self.tp], full_idx: Int, seq_len: Int, pos: Int):
        comptime SV = RankView[Self.tp]
        comptime S = Gemma4Shapes[Self.tp]
        var lb = s.full_base(full_idx)
        var fl = self.layout.full

        # Input norm → x_residual.
        rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + fl.body.input_norm),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K projections (K=V shared).
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullQ.N]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullK.N]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullK.N]()

        gemm(s.x_residual(seq_len), Bound[SV.FULL_Q_PROJ](lb + fl.q_proj),
            s.scratch_view[SV.Q_FULL_VIEW](q, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), Bound[SV.FULL_K_PROJ](lb + fl.k_proj),
            s.scratch_view[SV.KV_FULL_VIEW](k, seq_len), self.pool).join()
        # K=V sharing: copy K projection output to V scratch before norms diverge.
        var kp = s.scratch_ptr[Scalar[DType.bfloat16]](k)
        var vp = s.scratch_ptr[Scalar[DType.bfloat16]](v)
        comptime copy_width = simd_width_of[DType.bfloat16]()
        for j in range(0, S.FullK.N * seq_len, copy_width):
            (vp + j).store((kp + j).load[width=copy_width]())

        # Per-head norms (K gets scale, V doesn't).
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_HEADS](
            s.scratch_view[SV.Q_FULL_VIEW](q, seq_len),
            Bound[SV.FULL_HEAD_NORM](lb + fl.q_norm),
            s.scratch_view[SV.Q_FULL_VIEW](q, seq_len),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_KV_HEADS_FULL](
            s.scratch_view[SV.KV_FULL_VIEW](k, seq_len),
            Bound[SV.FULL_HEAD_NORM](lb + fl.k_norm),
            s.scratch_view[SV.KV_FULL_VIEW](k, seq_len),
            self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(s.scratch_view[SV.KV_FULL_VIEW](v, seq_len),
            s.scratch_view[SV.KV_FULL_VIEW](v, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # RoPE (Q and K only, partial rotation for full attention).
        apply_full_rope[C.NUM_HEADS](s.scratch_view[SV.Q_FULL_VIEW](q, seq_len),
            s.full_cos(), s.full_sin(), pos)
        apply_full_rope[C.NUM_KV_HEADS_FULL](s.scratch_view[SV.KV_FULL_VIEW](k, seq_len),
            s.full_cos(), s.full_sin(), pos)

        # KV cache write.
        kv_cache_write(s.scratch_view[SV.KV_FULL_VIEW](k, seq_len), s.full_k_cache(full_idx), pos)
        kv_cache_write(s.scratch_view[SV.KV_FULL_VIEW](v, seq_len), s.full_v_cache(full_idx), pos)

        v^.release()
        k^.release()

        # Attention.
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.FullQ.N]()
        global_attention[C.NUM_HEADS, C.NUM_KV_HEADS_FULL, C.HEAD_DIM_FULL](
            s.scratch_view[SV.Q_FULL_VIEW](q, seq_len),
            s.full_k_cache(full_idx), s.full_v_cache(full_idx),
            s.scratch_view[SV.Q_FULL_VIEW](attn_out, seq_len), pos, self.pool).join()

        # O projection → x_residual.
        gemm(s.scratch_view[SV.Q_FULL_VIEW](attn_out, seq_len),
            Bound[SV.FULL_O_PROJ](lb + fl.o_proj),
            s.x_residual(seq_len), self.pool).join()

        attn_out^.release()
        q^.release()

    # =========================================================================
    # Variance reporting (unchanged from initial bring-up)
    # =========================================================================

    @staticmethod
    def bf16_variance(ptr: Int, n: Int) -> Float64:
        var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
        comptime width = simd_width_of[DType.float32]()
        var acc = SIMD[DType.float64, 1](0)
        for i in range(0, n, width):
            var v = p.load[width=width](offset=i).cast[DType.float64]()
            acc += (v * v).reduce_add()
        return acc / Float64(n)

    def report_weight_variance(self):
        var L = self.layout
        var sl = L.sliding
        var fl = L.full
        var base = self.base

        comptime S = Gemma4Shapes[Self.tp]
        comptime q_sliding_n  = S.SlidingQ.ELEMS
        comptime k_sliding_n  = S.SlidingKV.ELEMS
        comptime q_full_n     = S.FullQ.ELEMS
        comptime k_full_n     = S.FullK.ELEMS

        var sliding_idx = 0
        var full_idx = 0
        for i in range(C.NUM_LAYERS):
            if (i + 1) % 6 == 0:
                var layer_base = base + L.full_off + full_idx * L.full_stride
                var q_ptr = layer_base + fl.q_proj
                var k_ptr = layer_base + fl.k_proj
                print("layer", i, "(full)   | q:", Self.bf16_variance(q_ptr, q_full_n),
                    "| k:", Self.bf16_variance(k_ptr, k_full_n))
                full_idx += 1
            else:
                var layer_base = base + L.sliding_off + sliding_idx * L.sliding_stride
                var q_ptr = layer_base + sl.q_proj
                var k_ptr = layer_base + sl.k_proj
                print("layer", i, "(slide)  | q:", Self.bf16_variance(q_ptr, q_sliding_n),
                    "| k:", Self.bf16_variance(k_ptr, k_sliding_n))
                sliding_idx += 1


def main():
    var model_opt = Gemma4[1].load(Path("checkpoints/gemma-4-26B-A4B"))
    if not model_opt:
        print("failed to load model")
        return
    var model = model_opt.take()
    model.report_weight_variance()
