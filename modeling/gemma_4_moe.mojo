"""Gemma 4 26B-A4B — weight loading, state management, and forward pass.

Text decoder weights only (vision/audio encoder weights ignored).
TP=1 for initial bring-up.

Layout: [25 sliding layers][5 full layers][state][host-only: final_norm, embed]
State:  [sliding KV caches][full KV caches][x_main][x_residual][scratch][rope tables]
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.math import sqrt
from numa import NumaArena, NumaInfo
from threading import BurstPool
from experimental.linear_borrow_pool import ScratchPool, ScratchLease

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32,
    Replicated, HOST_RANK, DISTRIBUTED,
    Slot, Bound, DynView, CacheView,
    byte_count, WeightDesc,
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
    comptime GATE_UP_DIM = 1408
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    # Vocab
    comptime VOCAB_SIZE = 262144
    comptime TIE_EMBEDDINGS = True

    # Layer counts by type
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5

    # Sequence and numerics
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime EMBED_SCALE = 53.0  # sqrt(2816) ≈ 53.04, cast to bf16


comptime C = Gemma4Config


# =============================================================================
# Runtime layout — plain-Int offset structs built once at load time.
#
# Each Gemma4 layer has a short kind-specific attention block (Q/K/V/O and
# per-head norms, sliding vs full differ only in shape and whether V exists)
# followed by an identical body (norms + dense MLP + router + MoE experts
# + layer_scalar). LayerBuilder.emit is the one place that knows how a
# shard kind maps to (local_rows, local_cols, target_rank); spec functions
# read as `(name, shape, placement)` schematics.
# =============================================================================


struct LayerShard:
    """Placement kinds a weight catalog entry can express.

    Gemma4 stores experts as a flat [NUM_EXPERTS*M, K] tensor, so
    expert-block sharding (rank r holds experts [r*N/tp, (r+1)*N/tp))
    collapses cleanly onto ROW — the flat row range is one contiguous
    slice. No dedicated EXPERT kind needed.
    """
    comptime ROW  = 0   # split rows across ranks (local_rows = global_rows // tp)
    comptime COL  = 1   # split cols across ranks (local_cols = global_cols // tp)
    comptime REPL = 2   # full copy on every rank
    comptime HOST = 3   # full copy, pinned to HOST_RANK only


@fieldwise_init
struct LayerBuilder(Movable):
    """Single source of truth for one layer's catalog AND offsets.

    One `emit()` per weight records both (a) the WeightDesc the loader
    consumes, and (b) the aligned byte offset the forward path reads.
    Spec functions collect return values into a named-fields struct so
    forward code can do `body.gate_proj` instead of template-indexing a
    list.
    """
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
        var bytes = local_rows * local_cols * element_bytes
        var off = ((self.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
        self.cursor = off + bytes
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=dtype, element_bytes=element_bytes,
            global_rows=global_rows, global_cols=global_cols,
            local_rows=local_rows, local_cols=local_cols,
            quantizable=False, absorbed=False,
            target_rank=target_rank,
        ))
        return off

    @always_inline
    def bf(mut self, mut entries: List[WeightDesc], suffix: String,
           rows: Int, cols: Int, shard: Int) -> Int:
        """bf16 weight — every weight in gemma_4_moe is bf16."""
        return self.emit(entries, suffix, rows, cols, DType.bfloat16, 2, shard)


@fieldwise_init
struct LayerBodyWeights(Copyable, ImplicitlyCopyable, Movable):
    """Fields identical between sliding and full attention layers.

    Sliding and full only differ in the attention block (Q/K/V/O and
    per-head norms with kind-specific HEAD_DIM). Everything from
    `input_norm` onwards is shared and lives here. Both layer-offset
    structs embed one of these so forward code can bind a single `body`
    local per layer instead of branching field-by-field.
    """
    # Runtime norms.
    var input_norm: Int
    var post_attn_norm: Int
    var pre_ffn_norm: Int
    var pre_ffn_norm_2: Int
    var post_ffn_norm_1: Int
    var post_ffn_norm_2: Int
    var post_ffn_norm: Int
    # Dense MLP.
    var gate_proj: Int
    var up_proj: Int
    var down_proj: Int
    # Router.
    var router_proj: Int
    var router_scale: Int
    var router_pes: Int
    # MoE experts (block-sharded along expert dim).
    var experts_gate_up: Int
    var experts_down: Int
    # Per-layer scalar.
    var layer_scalar: Int


@fieldwise_init
struct SlidingLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    """Sliding-attention layer (25 of 30) — has V projection + HEAD_DIM_SLIDING."""
    # Attention — kind-specific.
    var q_proj: Int
    var k_proj: Int
    var v_proj: Int
    var o_proj: Int
    var q_norm: Int
    var k_norm: Int
    # Shared body (input_norm → layer_scalar).
    var body: LayerBodyWeights
    var stride: Int


@fieldwise_init
struct FullLayerOffsets(Copyable, ImplicitlyCopyable, Movable):
    """Full-attention layer (5 of 30) — K=V shared, HEAD_DIM_FULL."""
    # Attention — kind-specific.
    var q_proj: Int
    var k_proj: Int
    var o_proj: Int
    var q_norm: Int
    var k_norm: Int
    # Shared body.
    var body: LayerBodyWeights
    var stride: Int


def emit_layer_body[tp: Int](
    mut b: LayerBuilder, mut entries: List[WeightDesc],
) -> LayerBodyWeights:
    """Emit every weight shared between sliding and full layers.

    Called from both spec functions after the attention block; ordering
    mirrors the original single-pass layout (all runtime norms first,
    then dense MLP, router, experts, scalar).
    """
    comptime ROW, REPL = LayerShard.ROW, LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime INT = C.INTERMEDIATE
    comptime NE  = C.NUM_EXPERTS
    comptime GU  = C.GATE_UP_DIM
    comptime MI  = C.MOE_INTERMEDIATE

    # Runtime norms (not absorbed into any quantized projection).
    var input_norm      = b.bf(entries, "input_layernorm.weight",               H, 1, REPL)
    var post_attn_norm  = b.bf(entries, "post_attention_layernorm.weight",      H, 1, REPL)
    var pre_ffn_norm    = b.bf(entries, "pre_feedforward_layernorm.weight",     H, 1, REPL)
    var pre_ffn_norm_2  = b.bf(entries, "pre_feedforward_layernorm_2.weight",   H, 1, REPL)
    var post_ffn_norm_1 = b.bf(entries, "post_feedforward_layernorm_1.weight",  H, 1, REPL)
    var post_ffn_norm_2 = b.bf(entries, "post_feedforward_layernorm_2.weight",  H, 1, REPL)
    var post_ffn_norm   = b.bf(entries, "post_feedforward_layernorm.weight",    H, 1, REPL)
    # Dense MLP — currently replicated; Megatron row/col would need kernel rewiring.
    var gate_proj = b.bf(entries, "mlp.gate_proj.weight", INT, H,   REPL)
    var up_proj   = b.bf(entries, "mlp.up_proj.weight",   INT, H,   REPL)
    var down_proj = b.bf(entries, "mlp.down_proj.weight", H,   INT, REPL)
    # Router — replicated (every rank reproduces the same top-k).
    var router_proj  = b.bf(entries, "router.proj.weight",       NE, H, REPL)
    var router_scale = b.bf(entries, "router.scale",             H, 1,  REPL)
    var router_pes   = b.bf(entries, "router.per_expert_scale",  NE, 1, REPL)
    # MoE experts — block-sharded along expert dim. Flat [NE*M, K] → ROW
    # gives rank r a contiguous block of NE/tp experts.
    var experts_gate_up = b.bf(entries, "experts.gate_up_proj", NE * GU, H,  ROW)
    var experts_down    = b.bf(entries, "experts.down_proj",    NE * H,  MI, ROW)
    # Per-layer scalar.
    var layer_scalar    = b.bf(entries, "layer_scalar",         1, 1,       REPL)

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
    """Spec for one sliding-attention layer (25 of 30 in Gemma4-26B)."""
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime REPL = LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime HDS = C.HEAD_DIM_SLIDING

    # Attention (kind-specific: has V projection, uses HEAD_DIM_SLIDING).
    var q_proj = b.bf(entries, "self_attn.q_proj.weight", C.Q_DIM_SLIDING,  H, REPL)
    var k_proj = b.bf(entries, "self_attn.k_proj.weight", C.KV_DIM_SLIDING, H, REPL)
    var v_proj = b.bf(entries, "self_attn.v_proj.weight", C.KV_DIM_SLIDING, H, REPL)
    var o_proj = b.bf(entries, "self_attn.o_proj.weight", H, C.Q_DIM_SLIDING,  REPL)
    var q_norm = b.bf(entries, "self_attn.q_norm.weight", HDS, 1,              REPL)
    var k_norm = b.bf(entries, "self_attn.k_norm.weight", HDS, 1,              REPL)
    # Shared body.
    var body = emit_layer_body[tp](b, entries)

    return SlidingLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
        q_norm=q_norm, k_norm=k_norm,
        body=body,
        stride=b.cursor,
    )


def full_layer_spec[tp: Int](
    prefix: String, layer_base: Int, mut entries: List[WeightDesc],
) -> FullLayerOffsets:
    """Spec for one full-attention layer (5 of 30 in Gemma4-26B)."""
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime REPL = LayerShard.REPL
    comptime H   = C.HIDDEN
    comptime HDF = C.HEAD_DIM_FULL

    # Attention (kind-specific: K=V shared so no V projection, HEAD_DIM_FULL).
    var q_proj = b.bf(entries, "self_attn.q_proj.weight", C.Q_DIM_FULL,  H, REPL)
    var k_proj = b.bf(entries, "self_attn.k_proj.weight", C.KV_DIM_FULL, H, REPL)
    var o_proj = b.bf(entries, "self_attn.o_proj.weight", H, C.Q_DIM_FULL, REPL)
    var q_norm = b.bf(entries, "self_attn.q_norm.weight", HDF, 1,          REPL)
    var k_norm = b.bf(entries, "self_attn.k_norm.weight", HDF, 1,          REPL)
    # Shared body.
    var body = emit_layer_body[tp](b, entries)

    return FullLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, o_proj=o_proj,
        q_norm=q_norm, k_norm=k_norm,
        body=body,
        stride=b.cursor,
    )


# =============================================================================
# Whole-model runtime layout
# =============================================================================


@fieldwise_init
struct Gemma4ModelLayout(Copyable, ImplicitlyCopyable, Movable):
    # Per-layer-kind offsets (relative to each layer's base address).
    var sliding: SlidingLayerOffsets
    var full: FullLayerOffsets
    # Layer bases in the weight block (relative to arena base).
    var sliding_off: Int
    var sliding_stride: Int
    var full_off: Int
    var full_stride: Int
    var distributed_bytes: Int
    # State block layout (relative to state_base = arena_base + distributed_bytes).
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
    # Host-only (absolute offsets in the host arena).
    var host_only_off: Int
    var final_norm_off: Int
    var embed_off: Int

    @always_inline
    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    @always_inline
    def host_arena_bytes(self) -> Int:
        return self.embed_off + C.VOCAB_SIZE * C.HIDDEN * 2


@fieldwise_init
struct Gemma4LoadPlan(Movable):
    """Both the runtime layout and the weight catalog from one pass."""
    var layout: Gemma4ModelLayout
    var descs: List[WeightDesc]


def calculate_peak_scratch() -> Int:
    """Runtime mirror of the old Gemma4Model.calculate_peak_scratch."""
    comptime S = C.MAX_SEQ_LEN
    # Full attention phase peak: q + k + v (then release k,v; borrow attn_out).
    comptime full_attn_borrows = (
        S * C.Q_DIM_FULL * 2 +
        S * C.KV_DIM_FULL * 2 +
        S * C.KV_DIM_FULL * 2 +
        S * C.Q_DIM_FULL * 2
    )
    comptime sliding_attn_borrows = (
        S * C.Q_DIM_SLIDING * 2 +
        S * C.KV_DIM_SLIDING * 2 +
        S * C.KV_DIM_SLIDING * 2 +
        S * C.Q_DIM_SLIDING * 2
    )
    comptime ffn_borrows_dense = S * C.INTERMEDIATE * 2 * 2              # gate + up
    comptime ffn_borrows_moe = (
        S * C.HIDDEN * 2 +                                                # dense_normed
        S * C.HIDDEN * 2 +                                                # moe_out
        C.TOP_K * C.HIDDEN * 2                                            # expert_out_buf
    )
    comptime ffn_peak = ffn_borrows_dense if ffn_borrows_dense > ffn_borrows_moe else ffn_borrows_moe
    comptime attn_peak = full_attn_borrows if full_attn_borrows > sliding_attn_borrows else sliding_attn_borrows
    return ffn_peak if ffn_peak > attn_peak else attn_peak


def build_gemma4_load_plan[tp: Int]() -> Gemma4LoadPlan:
    """Build layout + catalog in one pass.

    Calls the spec helpers once with empty prefix to grab canonical
    offsets and strides (entries discarded), then walks all 30 layers
    in layer-index order emitting real entries with correct prefixes
    and absolute arena offsets. State block and host-only region are
    laid out afterwards from the model-level layout math.
    """
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
    var scratch_capacity = calculate_peak_scratch()

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
    # Shape-carrying type aliases used by DynView / Bound / CacheView returns.
    # These are compile-time type-only; offsets come from the runtime layout.
    comptime X_MAIN           = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL       = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime SLIDING_ROPE_HALF = C.HEAD_DIM_SLIDING // 2
    comptime SLIDING_COS      = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime SLIDING_SIN      = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime FULL_COS         = Slot[F32, Replicated, C.MAX_SEQ_LEN, 64, Self.tp]
    comptime FULL_SIN         = Slot[F32, Replicated, C.MAX_SEQ_LEN, 64, Self.tp]
    comptime SLIDING_K_CACHE  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]
    comptime SLIDING_V_CACHE  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]
    comptime FULL_K_CACHE     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]
    comptime FULL_V_CACHE     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]
    comptime MLP_VIEW         = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime HIDDEN_VIEW      = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime LOGITS_VIEW      = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]
    comptime Q_SLIDING_VIEW   = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.Q_DIM_SLIDING, Self.tp]
    comptime KV_SLIDING_VIEW  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]
    comptime Q_FULL_VIEW      = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.Q_DIM_FULL, Self.tp]
    comptime KV_FULL_VIEW     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]
    comptime FINAL_NORM       = Slot[BF16, Replicated, C.HIDDEN, 1, Self.tp]
    comptime EMBED            = Slot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp]
    comptime FFN_GATE_W       = Slot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN, Self.tp]
    comptime FFN_UP_W         = Slot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN, Self.tp]
    comptime FFN_DOWN_W       = Slot[BF16, Replicated, C.HIDDEN, C.INTERMEDIATE, Self.tp]
    comptime NORM_W           = Slot[BF16, Replicated, C.HIDDEN, 1, Self.tp]
    # Attention projection shapes — sliding layers.
    comptime SLIDING_Q_PROJ   = Slot[BF16, Replicated, C.Q_DIM_SLIDING,  C.HIDDEN, Self.tp]
    comptime SLIDING_KV_PROJ  = Slot[BF16, Replicated, C.KV_DIM_SLIDING, C.HIDDEN, Self.tp]
    comptime SLIDING_O_PROJ   = Slot[BF16, Replicated, C.HIDDEN, C.Q_DIM_SLIDING,  Self.tp]
    comptime SLIDING_HEAD_NORM = Slot[BF16, Replicated, C.HEAD_DIM_SLIDING, 1, Self.tp]
    # Attention projection shapes — full layers (K=V shared).
    comptime FULL_Q_PROJ      = Slot[BF16, Replicated, C.Q_DIM_FULL,  C.HIDDEN, Self.tp]
    comptime FULL_K_PROJ      = Slot[BF16, Replicated, C.KV_DIM_FULL, C.HIDDEN, Self.tp]
    comptime FULL_O_PROJ      = Slot[BF16, Replicated, C.HIDDEN, C.Q_DIM_FULL,  Self.tp]
    comptime FULL_HEAD_NORM   = Slot[BF16, Replicated, C.HEAD_DIM_FULL, 1, Self.tp]

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
            + byte_count[Self.SLIDING_K_CACHE]())
    def full_k_cache(self, full_idx: Int) -> CacheView[Self.FULL_K_CACHE]:
        return CacheView[Self.FULL_K_CACHE](
            self.state_base() + self.L().full_kv_off + full_idx * self.L().full_kv_stride)
    def full_v_cache(self, full_idx: Int) -> CacheView[Self.FULL_V_CACHE]:
        return CacheView[Self.FULL_V_CACHE](
            self.state_base() + self.L().full_kv_off + full_idx * self.L().full_kv_stride
            + byte_count[Self.FULL_K_CACHE]())

    # RoPE tables.
    def sliding_cos(self) -> Bound[Self.SLIDING_COS]:
        return Bound[Self.SLIDING_COS](self.state_base() + self.L().sliding_cos_off)
    def sliding_sin(self) -> Bound[Self.SLIDING_SIN]:
        return Bound[Self.SLIDING_SIN](self.state_base() + self.L().sliding_sin_off)
    def full_cos(self) -> Bound[Self.FULL_COS]:
        return Bound[Self.FULL_COS](self.state_base() + self.L().full_cos_off)
    def full_sin(self) -> Bound[Self.FULL_SIN]:
        return Bound[Self.FULL_SIN](self.state_base() + self.L().full_sin_off)

    # Per-layer weight base addresses (add the per-field offset from sliding/full).
    def sliding_base(self, sliding_idx: Int) -> Int:
        return self.weight_base() + self.L().sliding_off + sliding_idx * self.L().sliding_stride
    def full_base(self, full_idx: Int) -> Int:
        return self.weight_base() + self.L().full_off + full_idx * self.L().full_stride

    # Host-only accessors — direct typed binds for final_norm + tied embed.
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

    def sv(mut self) -> StateView[Self.tp]:
        return StateView[Self.tp](self.base,
            UnsafePointer[Gemma4ModelLayout, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.layout))))

    def init_state(mut self):
        """Initialize RoPE tables and bake router constants."""
        var s = self.sv()
        var L = self.layout

        # RoPE tables.
        init_sliding_rope_tables(s.sliding_cos(), s.sliding_sin())
        init_full_rope_tables(s.full_cos(), s.full_sin())

        # Bake router constant: router_scale *= 1/sqrt(hidden). One-time
        # rewrite of the bf16 gamma so the router-input rmsnorm absorbs
        # the 1/sqrt(H) without paying for it on every forward.
        comptime inv_sqrt_hidden = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
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
            unsafe_from_address=self.sv().scratch_base())

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime SV = StateView[Self.tp]
        var s = self.sv()
        var L = self.layout
        var sl = L.sliding
        var fl = L.full

        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        # --- Embed ---
        embed_lookup_scaled(s.embed_table(), tokens_ptr, s.x_main(seq_len),
            Float32(C.EMBED_SCALE), self.pool).join()

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

            var gate_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.INTERMEDIATE]()
            var up_lease   = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.INTERMEDIATE]()

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

    def attention_sliding(mut self, s: StateView[Self.tp], sliding_idx: Int, seq_len: Int, pos: Int):
        comptime SV = StateView[Self.tp]
        var lb = s.sliding_base(sliding_idx)
        var sl = self.layout.sliding

        # Input norm → x_residual
        rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + sl.body.input_norm),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K, V projections.
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_SLIDING]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_SLIDING]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_SLIDING]()

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
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_SLIDING]()
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

    def attention_full(mut self, s: StateView[Self.tp], full_idx: Int, seq_len: Int, pos: Int):
        comptime SV = StateView[Self.tp]
        var lb = s.full_base(full_idx)
        var fl = self.layout.full

        # Input norm → x_residual.
        rmsnorm(s.x_main(seq_len), Bound[SV.NORM_W](lb + fl.body.input_norm),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K projections (K=V shared).
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_FULL]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_FULL]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_FULL]()

        gemm(s.x_residual(seq_len), Bound[SV.FULL_Q_PROJ](lb + fl.q_proj),
            s.scratch_view[SV.Q_FULL_VIEW](q, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), Bound[SV.FULL_K_PROJ](lb + fl.k_proj),
            s.scratch_view[SV.KV_FULL_VIEW](k, seq_len), self.pool).join()
        # K=V sharing: copy K projection output to V scratch before norms diverge.
        var kp = s.scratch_ptr[Scalar[DType.bfloat16]](k)
        var vp = s.scratch_ptr[Scalar[DType.bfloat16]](v)
        comptime copy_width = simd_width_of[DType.bfloat16]()
        for j in range(0, C.KV_DIM_FULL * seq_len, copy_width):
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
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_FULL]()
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

        # Per-rank Q/K byte counts differ by flavor.
        comptime q_sliding_n  = (C.Q_DIM_SLIDING  // Self.tp) * C.HIDDEN
        comptime k_sliding_n  = (C.KV_DIM_SLIDING // Self.tp) * C.HIDDEN
        comptime q_full_n     = (C.Q_DIM_FULL     // Self.tp) * C.HIDDEN
        comptime k_full_n     = (C.KV_DIM_FULL    // Self.tp) * C.HIDDEN

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
