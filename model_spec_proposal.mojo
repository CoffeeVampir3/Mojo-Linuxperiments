"""Proposal: runtime weight catalogs + plain layer-offset structs.

Replaces the `PlacedSlot[...]`/`SlidingLayer[tp]`/`FullLayer[tp]` comptime
cascade. Three layers, one responsibility each:

  1. `Gemma4Dims` — bare comptime scalars kernels want at compile time.
  2. `WeightEntry` + `build_*_layer_plan` — runtime catalogs for the
     loader/quantizer, plus the aligned byte offsets.
  3. `SlidingLayerOffsets` / `FullLayerOffsets` — plain-Int structs the
     forward path reads (`slb + offs.q_proj`) with zero template friction.

Nothing in this file is parameterized on `tp` at the type level. `tp` is a
single comptime parameter on `Gemma4ButterQuant[tp]` (unchanged), carried
into kernel call sites via `Gemma4Dims.*` scalars. Layer offsets are built
once at model init and stored as fields.

Sketch only — not plumbed into loader.mojo / forward_decode yet.
"""

from std.memory import UnsafePointer
from std.sys.info import size_of


# =============================================================================
# Pack function signature — unchanged from model_spec.mojo
# =============================================================================


comptime PackFn = def(
    UnsafePointer[UInt8, MutAnyOrigin],  # src
    UnsafePointer[UInt8, MutAnyOrigin],  # dst
    Int, Int,                            # rows, cols
)

def pack_noop(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    pass


# =============================================================================
# Runtime enums — one Int each, no trait dispatch
# =============================================================================


struct ShardKind:
    comptime ROW = 0          # split rows across ranks
    comptime COL = 1          # split cols across ranks
    comptime REPLICATED = 2   # full copy on every rank

struct WeightTag:
    comptime PASSTHROUGH = 0
    comptime QUANTIZABLE = 1
    comptime GAMMA_QUANTIZABLE = 2   # absorbs a preceding norm's gamma
    comptime PER_BLOCK_QUANTIZABLE = 3
    comptime ABSORBED = 4            # consumed by quantizer, never allocated

comptime DISTRIBUTED = -1
comptime HOST_RANK = 0
comptime DEFAULT_ALIGNMENT = 64


def dtype_bytes(dtype: DType) -> Int:
    if dtype == DType.bfloat16 or dtype == DType.float16:
        return 2
    if dtype == DType.float32 or dtype == DType.int32:
        return 4
    if dtype == DType.int8 or dtype == DType.uint8:
        return 1
    return 0


def shard_local_rows(global_rows: Int, shard: Int, tp: Int) -> Int:
    if shard == ShardKind.ROW:
        return global_rows // tp
    return global_rows

def shard_local_cols(global_cols: Int, shard: Int, tp: Int) -> Int:
    if shard == ShardKind.COL:
        return global_cols // tp
    return global_cols

def align_up(x: Int, a: Int) -> Int:
    return ((x + a - 1) // a) * a


# =============================================================================
# WeightEntry — the one record the loader and quantizer need
# =============================================================================


@fieldwise_init
struct WeightEntry(Copyable):
    var name: String            # full tensor name, e.g. "model.language_model.layers.0.self_attn.q_proj.weight"
    var arena_offset: Int       # byte offset into the rank arena
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_rows: Int
    var local_cols: Int
    var shard: Int              # ShardKind
    var tag: Int                # WeightTag
    var pack_fn: PackFn
    var target_rank: Int        # DISTRIBUTED or a specific rank


# =============================================================================
# Gemma4Dims — the ONLY comptime scalars. Everything kernel-shaped.
# =============================================================================


struct Gemma4Dims:
    comptime HIDDEN = 2816
    comptime NUM_LAYERS = 30
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5

    comptime NUM_HEADS = 16
    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime Q_DIM_SLIDING = 4096
    comptime KV_DIM_SLIDING = 2048
    comptime QKV_DIM_SLIDING = Self.Q_DIM_SLIDING + 2 * Self.KV_DIM_SLIDING

    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime Q_DIM_FULL = 8192
    comptime KV_DIM_FULL = 1024
    comptime QK_DIM_FULL = Self.Q_DIM_FULL + Self.KV_DIM_FULL

    comptime INTERMEDIATE = 2112
    comptime MOE_GATE_UP_FUSED = 1408
    comptime MOE_INTERMEDIATE = 704
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    comptime FWHT_BLK = 64
    comptime FWHT_BLK_HIDDEN = 256
    comptime DENSE_NUM_BLOCKS = Self.INTERMEDIATE // Self.FWHT_BLK
    comptime MOE_NUM_BLOCKS = Self.MOE_INTERMEDIATE // Self.FWHT_BLK
    comptime VOCAB_NUM_BLOCKS = Self.HIDDEN // Self.FWHT_BLK_HIDDEN

    comptime VOCAB_SIZE = 262144
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime EMBED_SCALE = 53.0
    comptime LOGIT_SOFTCAP = 30.0


comptime D = Gemma4Dims


# =============================================================================
# Layer offsets — plain Int fields, runtime-built, forward path reads these
# =============================================================================


@fieldwise_init
struct SlidingLayerOffsets(Copyable):
    # Attention
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
    # MoE experts
    var pre_ffn_norm_2: Int
    var experts_gate_up: Int
    var experts_gate_up_sc: Int
    var experts_down: Int
    var experts_down_sc: Int
    # Post-block norms + layer scalar
    var post_attn_norm: Int
    var post_ffn_norm_1: Int
    var post_ffn_norm_2_rt: Int
    var post_ffn_norm: Int
    var layer_scalar: Int
    # Column sums (computed at init, not loaded)
    var qkv_colsum: Int
    var o_colsum: Int
    var gu_colsum: Int
    var down_colsum: Int
    var router_colsum: Int
    var experts_gu_colsum: Int
    var experts_down_colsum: Int
    var stride: Int


@fieldwise_init
struct FullLayerOffsets(Copyable):
    # Attention — note: no V_PROJ, K=V shared
    var q_proj: Int
    var k_proj: Int
    var q_proj_sc: Int
    var k_proj_sc: Int
    var o_proj: Int
    var o_proj_sc: Int
    var q_norm: Int
    var k_norm: Int
    var input_norm: Int
    # Dense MLP (identical to sliding)
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
    # MoE experts
    var pre_ffn_norm_2: Int
    var experts_gate_up: Int
    var experts_gate_up_sc: Int
    var experts_down: Int
    var experts_down_sc: Int
    # Post-block norms + layer scalar
    var post_attn_norm: Int
    var post_ffn_norm_1: Int
    var post_ffn_norm_2_rt: Int
    var post_ffn_norm: Int
    var layer_scalar: Int
    # Column sums
    var qk_colsum: Int
    var o_colsum: Int
    var gu_colsum: Int
    var down_colsum: Int
    var router_colsum: Int
    var experts_gu_colsum: Int
    var experts_down_colsum: Int
    var stride: Int


# =============================================================================
# Layer plan — one builder produces both the catalog (for loader/quantizer)
# and the offsets struct (for forward path) in one pass. Builder is private;
# users call plan_sliding_layer / plan_full_layer.
# =============================================================================


struct LayerPlanBuilder:
    var tp: Int
    var cursor: Int                      # running byte offset, relative to layer base
    var entries: List[WeightEntry]       # appended in-order
    var layer_name_prefix: String        # "model.language_model.layers.N."

    def __init__(out self, tp: Int, prefix: String):
        self.tp = tp
        self.cursor = 0
        self.entries = List[WeightEntry]()
        self.layer_name_prefix = prefix

    def emit(mut self, name_suffix: String, global_rows: Int, global_cols: Int,
             dtype: DType, shard: Int, tag: Int, pack_fn: PackFn,
             target_rank: Int = DISTRIBUTED) -> Int:
        """Assign the next aligned offset, record a WeightEntry, advance cursor.

        Absorbed tags still reserve a slot in the catalog so the quantizer
        can find the source tensor, but emit nothing in the arena layout.
        """
        var lr = shard_local_rows(global_rows, shard, self.tp)
        var lc = shard_local_cols(global_cols, shard, self.tp)
        var eb = dtype_bytes(dtype)
        var off = align_up(self.cursor, DEFAULT_ALIGNMENT)
        self.entries.append(WeightEntry(
            name=self.layer_name_prefix + name_suffix,
            arena_offset=off,
            dtype=dtype, element_bytes=eb,
            global_rows=global_rows, global_cols=global_cols,
            local_rows=lr, local_cols=lc,
            shard=shard, tag=tag, pack_fn=pack_fn,
            target_rank=target_rank,
        ))
        if tag != WeightTag.ABSORBED:
            self.cursor = off + lr * lc * eb
        return off

    def reserve(mut self, bytes: Int) -> Int:
        """Carve out raw bytes (e.g. colsum buffers, no associated tensor)."""
        var off = align_up(self.cursor, DEFAULT_ALIGNMENT)
        self.cursor = off + bytes
        return off


@fieldwise_init
struct SlidingLayerPlan(Copyable):
    var offsets: SlidingLayerOffsets
    var entries: List[WeightEntry]

@fieldwise_init
struct FullLayerPlan(Copyable):
    var offsets: FullLayerOffsets
    var entries: List[WeightEntry]


def plan_sliding_layer(tp: Int, prefix: String, vnni_pack: PackFn) -> SlidingLayerPlan:
    var b = LayerPlanBuilder(tp, prefix)
    comptime Q = WeightTag.QUANTIZABLE
    comptime P = WeightTag.PASSTHROUGH

    # --- Attention i8: [Q|K|V] then [Q|K|V]_scales ---
    var q_proj    = b.emit("self_attn.q_proj.weight", D.Q_DIM_SLIDING,  D.HIDDEN, DType.int8,    ShardKind.ROW, Q, vnni_pack)
    var k_proj    = b.emit("self_attn.k_proj.weight", D.KV_DIM_SLIDING, D.HIDDEN, DType.int8,    ShardKind.ROW, Q, vnni_pack)
    var v_proj    = b.emit("self_attn.v_proj.weight", D.KV_DIM_SLIDING, D.HIDDEN, DType.int8,    ShardKind.ROW, Q, vnni_pack)
    var q_proj_sc = b.emit("self_attn.q_proj.weight_scale", D.Q_DIM_SLIDING,  1, DType.float32, ShardKind.ROW, P, pack_noop)
    var k_proj_sc = b.emit("self_attn.k_proj.weight_scale", D.KV_DIM_SLIDING, 1, DType.float32, ShardKind.ROW, P, pack_noop)
    var v_proj_sc = b.emit("self_attn.v_proj.weight_scale", D.KV_DIM_SLIDING, 1, DType.float32, ShardKind.ROW, P, pack_noop)
    # --- O proj ---
    var o_proj    = b.emit("self_attn.o_proj.weight", D.HIDDEN, D.Q_DIM_SLIDING, DType.int8,    ShardKind.COL,        Q, vnni_pack)
    var o_proj_sc = b.emit("self_attn.o_proj.weight_scale", D.HIDDEN, 1,         DType.float32, ShardKind.REPLICATED, P, pack_noop)
    # --- Per-head norms ---
    var q_norm     = b.emit("self_attn.q_norm.weight", D.HEAD_DIM_SLIDING, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var k_norm     = b.emit("self_attn.k_norm.weight", D.HEAD_DIM_SLIDING, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var input_norm = b.emit("input_layernorm.weight",   D.HIDDEN, 1,           DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    # --- Dense MLP ---
    var pre_ffn_norm = b.emit("pre_feedforward_layernorm.weight", D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var gate_proj    = b.emit("mlp.gate_proj.weight", D.INTERMEDIATE, D.HIDDEN,  DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var up_proj      = b.emit("mlp.up_proj.weight",   D.INTERMEDIATE, D.HIDDEN,  DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var gate_proj_sc = b.emit("mlp.gate_proj.weight_scale", D.INTERMEDIATE, 1,   DType.float32, ShardKind.REPLICATED, P, pack_noop)
    var up_proj_sc   = b.emit("mlp.up_proj.weight_scale",   D.INTERMEDIATE, 1,   DType.float32, ShardKind.REPLICATED, P, pack_noop)
    var down_proj    = b.emit("mlp.down_proj.weight", D.HIDDEN, D.INTERMEDIATE,  DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var down_proj_sc = b.emit("mlp.down_proj.weight_scale", D.HIDDEN, 1,         DType.float32, ShardKind.REPLICATED, P, pack_noop)
    # --- Router ---
    var router_scale   = b.emit("router.scale", D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var router_proj    = b.emit("router.proj.weight", D.NUM_EXPERTS, D.HIDDEN, DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var router_proj_sc = b.emit("router.proj.weight_scale", D.NUM_EXPERTS, 1,  DType.float32, ShardKind.REPLICATED, P, pack_noop)
    var router_pes     = b.emit("router.per_expert_scale",  D.NUM_EXPERTS, 1,  DType.bfloat16,ShardKind.REPLICATED, P, pack_noop)
    # --- Expert pre-norm + packed-2D experts ---
    var pre_ffn_norm_2 = b.emit("pre_feedforward_layernorm_2.weight", D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var experts_gate_up    = b.emit("experts.gate_up_proj",       D.NUM_EXPERTS * D.MOE_GATE_UP_FUSED, D.HIDDEN,         DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var experts_gate_up_sc = b.emit("experts.gate_up_proj_scale", D.NUM_EXPERTS * D.MOE_GATE_UP_FUSED, 1,                 DType.float32, ShardKind.REPLICATED, P, pack_noop)
    var experts_down       = b.emit("experts.down_proj",          D.NUM_EXPERTS * D.HIDDEN,             D.MOE_INTERMEDIATE, DType.int8,    ShardKind.REPLICATED, Q, vnni_pack)
    var experts_down_sc    = b.emit("experts.down_proj_scale",    D.NUM_EXPERTS * D.HIDDEN,             1,                 DType.float32, ShardKind.REPLICATED, P, pack_noop)
    # --- Non-absorbable norms + scalar ---
    var post_attn_norm    = b.emit("post_attention_layernorm.weight",    D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var post_ffn_norm_1   = b.emit("post_feedforward_layernorm_1.weight", D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var post_ffn_norm_2_rt = b.emit("post_feedforward_layernorm_2.weight", D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var post_ffn_norm     = b.emit("post_feedforward_layernorm.weight",  D.HIDDEN, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var layer_scalar      = b.emit("layer_scalar",                        1, 1,       DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)

    # --- Column sums: raw-byte reservations, no associated tensor ---
    var qkv_n   = D.Q_DIM_SLIDING + 2 * D.KV_DIM_SLIDING
    var o_nblk  = (D.Q_DIM_SLIDING // tp) // D.HEAD_DIM_SLIDING
    var qkv_colsum          = b.reserve(qkv_n * 4)
    var o_colsum            = b.reserve(D.HIDDEN * o_nblk * 4)
    var gu_colsum           = b.reserve(D.INTERMEDIATE * 2 * 4)
    var down_colsum         = b.reserve(D.HIDDEN * D.DENSE_NUM_BLOCKS * 4)
    var router_colsum       = b.reserve(D.NUM_EXPERTS * 4)
    var experts_gu_colsum   = b.reserve(D.NUM_EXPERTS * D.MOE_GATE_UP_FUSED * 4)
    var experts_down_colsum = b.reserve(D.NUM_EXPERTS * D.HIDDEN * D.MOE_NUM_BLOCKS * 4)

    var offs = SlidingLayerOffsets(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        q_proj_sc=q_proj_sc, k_proj_sc=k_proj_sc, v_proj_sc=v_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm, input_norm=input_norm,
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
        qkv_colsum=qkv_colsum, o_colsum=o_colsum, gu_colsum=gu_colsum,
        down_colsum=down_colsum, router_colsum=router_colsum,
        experts_gu_colsum=experts_gu_colsum, experts_down_colsum=experts_down_colsum,
        stride=b.cursor,
    )
    return SlidingLayerPlan(offs, b.entries^)


def plan_full_layer(tp: Int, prefix: String, vnni_pack: PackFn) -> FullLayerPlan:
    """Same shape, minus V_PROJ (K=V shared in full-attention layers)."""
    var b = LayerPlanBuilder(tp, prefix)
    comptime Q = WeightTag.QUANTIZABLE
    comptime P = WeightTag.PASSTHROUGH

    var q_proj    = b.emit("self_attn.q_proj.weight", D.Q_DIM_FULL,  D.HIDDEN, DType.int8,    ShardKind.ROW, Q, vnni_pack)
    var k_proj    = b.emit("self_attn.k_proj.weight", D.KV_DIM_FULL, D.HIDDEN, DType.int8,    ShardKind.ROW, Q, vnni_pack)
    var q_proj_sc = b.emit("self_attn.q_proj.weight_scale", D.Q_DIM_FULL,  1, DType.float32, ShardKind.ROW, P, pack_noop)
    var k_proj_sc = b.emit("self_attn.k_proj.weight_scale", D.KV_DIM_FULL, 1, DType.float32, ShardKind.ROW, P, pack_noop)
    var o_proj    = b.emit("self_attn.o_proj.weight", D.HIDDEN, D.Q_DIM_FULL,  DType.int8,    ShardKind.COL,        Q, vnni_pack)
    var o_proj_sc = b.emit("self_attn.o_proj.weight_scale", D.HIDDEN, 1,       DType.float32, ShardKind.REPLICATED, P, pack_noop)
    var q_norm     = b.emit("self_attn.q_norm.weight", D.HEAD_DIM_FULL, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var k_norm     = b.emit("self_attn.k_norm.weight", D.HEAD_DIM_FULL, 1, DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)
    var input_norm = b.emit("input_layernorm.weight",   D.HIDDEN, 1,        DType.bfloat16, ShardKind.REPLICATED, P, pack_noop)

    # Dense MLP / router / experts / post-norms: identical to sliding, elided for brevity.
    # (In the real file this is ~25 more b.emit(...) calls copied from plan_sliding_layer.)
    var pre_ffn_norm = 0
    var gate_proj = 0
    var up_proj = 0
    var gate_proj_sc = 0
    var up_proj_sc = 0
    var down_proj = 0
    var down_proj_sc = 0
    var router_scale = 0
    var router_proj = 0
    var router_proj_sc = 0
    var router_pes = 0
    var pre_ffn_norm_2 = 0
    var experts_gate_up = 0
    var experts_gate_up_sc = 0
    var experts_down = 0
    var experts_down_sc = 0
    var post_attn_norm = 0
    var post_ffn_norm_1 = 0
    var post_ffn_norm_2_rt = 0
    var post_ffn_norm = 0
    var layer_scalar = 0

    var qk_n = D.Q_DIM_FULL + D.KV_DIM_FULL
    var o_nblk = (D.Q_DIM_FULL // tp) // D.HEAD_DIM_FULL
    var qk_colsum           = b.reserve(qk_n * 4)
    var o_colsum            = b.reserve(D.HIDDEN * o_nblk * 4)
    var gu_colsum           = b.reserve(D.INTERMEDIATE * 2 * 4)
    var down_colsum         = b.reserve(D.HIDDEN * D.DENSE_NUM_BLOCKS * 4)
    var router_colsum       = b.reserve(D.NUM_EXPERTS * 4)
    var experts_gu_colsum   = b.reserve(D.NUM_EXPERTS * D.MOE_GATE_UP_FUSED * 4)
    var experts_down_colsum = b.reserve(D.NUM_EXPERTS * D.HIDDEN * D.MOE_NUM_BLOCKS * 4)

    var offs = FullLayerOffsets(
        q_proj=q_proj, k_proj=k_proj,
        q_proj_sc=q_proj_sc, k_proj_sc=k_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm, input_norm=input_norm,
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
        qk_colsum=qk_colsum, o_colsum=o_colsum, gu_colsum=gu_colsum,
        down_colsum=down_colsum, router_colsum=router_colsum,
        experts_gu_colsum=experts_gu_colsum, experts_down_colsum=experts_down_colsum,
        stride=b.cursor,
    )
    return FullLayerPlan(offs, b.entries^)


# =============================================================================
# Whole-model plan — aggregates both layer kinds, state, rope, KV, host-only.
# Everything the old `Gemma4Model[tp]` comptime namespace exposed, as plain Int
# fields. Built once during load, stored on `Gemma4ButterQuant`.
# =============================================================================


@fieldwise_init
struct Gemma4ModelLayout(Copyable):
    # Weights
    var sliding_off: Int
    var sliding_stride: Int
    var full_off: Int
    var full_stride: Int
    var distributed_bytes: Int
    # State
    var x_main_off: Int
    var x_residual_off: Int
    var scratch_off: Int
    var scratch_capacity: Int
    var sliding_cos_off: Int
    var sliding_sin_off: Int
    var full_cos_off: Int
    var full_sin_off: Int
    var sliding_cache_off: Int
    var sliding_cache_stride: Int
    var full_cache_off: Int
    var full_cache_stride: Int
    var state_bytes: Int
    # Host-only
    var host_only_off: Int
    var final_norm_off: Int
    var embed_off: Int
    var embed_sc_off: Int
    var embed_colsum_off: Int
    var embed_colsum_bytes: Int
    # Per-layer offsets, picked by is_full at forward time
    var sliding_layer: SlidingLayerOffsets
    var full_layer: FullLayerOffsets
    # Catalogs consumed by the loader / quantizer — one list across the whole
    # model. Entry `name` includes the full "model.language_model.layers.N."
    # prefix so the loader resolves by exact tensor name.
    var entries: List[WeightEntry]

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.embed_colsum_off + self.embed_colsum_bytes


def layer_is_full(layer_idx: Int) -> Bool:
    return (layer_idx + 1) % 6 == 0


def build_gemma4_layout(tp: Int, vnni_pack: PackFn, scratch_capacity: Int) -> Gemma4ModelLayout:
    """Compose per-layer plans into a whole-model layout.

    One pass emits the full weight catalog for all 30 layers + host-only
    tensors. The returned layout is everything the loader, quantizer, init
    code, and forward path need — no templates, no `Self.tp`.
    """
    # Per-layer offsets only need to be planned ONCE per kind: every sliding
    # layer has identical offsets relative to its own base, likewise for full.
    var sliding_template = plan_sliding_layer(tp, "" , vnni_pack)   # prefix irrelevant here
    var full_template    = plan_full_layer(tp, "", vnni_pack)
    var sliding_stride = sliding_template.offsets.stride
    var full_stride    = full_template.offsets.stride

    var sliding_off = 0
    var full_off = sliding_off + D.NUM_SLIDING_LAYERS * sliding_stride
    var distributed_bytes = full_off + D.NUM_FULL_LAYERS * full_stride

    # Build the catalog by walking all 30 layers in order and offsetting
    # each layer-local entry by its layer base.
    var entries = List[WeightEntry]()
    var sliding_idx = 0
    var full_idx = 0
    for i in range(D.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if layer_is_full(i):
            var base = full_off + full_idx * full_stride
            var p = plan_full_layer(tp, prefix, vnni_pack)
            for e in p.entries:
                var shifted = e
                shifted.arena_offset += base
                entries.append(shifted^)
            full_idx += 1
        else:
            var base = sliding_off + sliding_idx * sliding_stride
            var p = plan_sliding_layer(tp, prefix, vnni_pack)
            for e in p.entries:
                var shifted = e
                shifted.arena_offset += base
                entries.append(shifted^)
            sliding_idx += 1

    # State: x_main / x_residual / scratch / rope / kv caches.
    var bf16_bytes_per_hidden_row = D.HIDDEN * 2
    var x_main_off = 0
    var x_main_bytes = D.MAX_SEQ_LEN * bf16_bytes_per_hidden_row
    var x_residual_off = x_main_off + x_main_bytes
    var x_residual_bytes = x_main_bytes
    var scratch_off = x_residual_off + x_residual_bytes

    var sliding_rope_half = D.HEAD_DIM_SLIDING // 2
    var full_rope_half = 64
    var rope_bytes_sliding = D.MAX_SEQ_LEN * sliding_rope_half * 4
    var rope_bytes_full    = D.MAX_SEQ_LEN * full_rope_half * 4
    var sliding_cos_off = scratch_off + scratch_capacity
    var sliding_sin_off = sliding_cos_off + rope_bytes_sliding
    var full_cos_off    = sliding_sin_off + rope_bytes_sliding
    var full_sin_off    = full_cos_off + rope_bytes_full

    # KV-cache byte totals left abstract here — defer to the existing
    # Gemma4KVCache.TOTAL_BYTES helper at the call site; we just record the
    # offsets. Caller passes strides in (omitted from this sketch).
    var sliding_cache_stride = 0    # = Gemma4KVCache[...].TOTAL_BYTES
    var full_cache_stride    = 0    # same
    var sliding_cache_off = full_sin_off + rope_bytes_full
    var full_cache_off    = sliding_cache_off + D.NUM_SLIDING_LAYERS * sliding_cache_stride
    var state_bytes       = full_cache_off + D.NUM_FULL_LAYERS * full_cache_stride

    # Host-only: final norm + tied embed + per-block colsums.
    var host_only_off = align_up(distributed_bytes + state_bytes, DEFAULT_ALIGNMENT)
    var final_norm_off = host_only_off
    var embed_off      = align_up(final_norm_off + D.HIDDEN * 2, DEFAULT_ALIGNMENT)
    var embed_sc_off   = align_up(embed_off + D.VOCAB_SIZE * D.HIDDEN, DEFAULT_ALIGNMENT)
    var embed_sc_bytes = D.VOCAB_SIZE * D.VOCAB_NUM_BLOCKS * 4
    var embed_colsum_off = align_up(embed_sc_off + embed_sc_bytes, DEFAULT_ALIGNMENT)
    var embed_colsum_bytes = D.VOCAB_SIZE * D.VOCAB_NUM_BLOCKS * 4

    # Final three host-only entries get appended after the per-layer ones.
    entries.append(WeightEntry(
        name="model.language_model.norm.weight",
        arena_offset=final_norm_off, dtype=DType.bfloat16, element_bytes=2,
        global_rows=D.HIDDEN, global_cols=1, local_rows=D.HIDDEN, local_cols=1,
        shard=ShardKind.REPLICATED, tag=WeightTag.PASSTHROUGH, pack_fn=pack_noop,
        target_rank=HOST_RANK,
    ))
    entries.append(WeightEntry(
        name="model.language_model.embed_tokens.weight",
        arena_offset=embed_off, dtype=DType.int8, element_bytes=1,
        global_rows=D.VOCAB_SIZE, global_cols=D.HIDDEN,
        local_rows=D.VOCAB_SIZE, local_cols=D.HIDDEN,
        shard=ShardKind.REPLICATED, tag=WeightTag.PER_BLOCK_QUANTIZABLE,
        pack_fn=pack_noop, target_rank=HOST_RANK,
    ))
    entries.append(WeightEntry(
        name="model.language_model.embed_tokens.weight_scale",
        arena_offset=embed_sc_off, dtype=DType.float32, element_bytes=4,
        global_rows=D.VOCAB_SIZE, global_cols=D.VOCAB_NUM_BLOCKS,
        local_rows=D.VOCAB_SIZE, local_cols=D.VOCAB_NUM_BLOCKS,
        shard=ShardKind.REPLICATED, tag=WeightTag.PASSTHROUGH, pack_fn=pack_noop,
        target_rank=HOST_RANK,
    ))

    return Gemma4ModelLayout(
        sliding_off=sliding_off, sliding_stride=sliding_stride,
        full_off=full_off, full_stride=full_stride,
        distributed_bytes=distributed_bytes,
        x_main_off=x_main_off, x_residual_off=x_residual_off,
        scratch_off=scratch_off, scratch_capacity=scratch_capacity,
        sliding_cos_off=sliding_cos_off, sliding_sin_off=sliding_sin_off,
        full_cos_off=full_cos_off, full_sin_off=full_sin_off,
        sliding_cache_off=sliding_cache_off, sliding_cache_stride=sliding_cache_stride,
        full_cache_off=full_cache_off, full_cache_stride=full_cache_stride,
        state_bytes=state_bytes,
        host_only_off=host_only_off,
        final_norm_off=final_norm_off, embed_off=embed_off, embed_sc_off=embed_sc_off,
        embed_colsum_off=embed_colsum_off, embed_colsum_bytes=embed_colsum_bytes,
        sliding_layer=sliding_template.offsets,
        full_layer=full_template.offsets,
        entries=entries^,
    )


# =============================================================================
# Sketch: what the forward path looks like with this shape
#
# The `Gemma4ButterQuant[tp]` struct gains one field:
#     var layout: Gemma4ModelLayout
# and forward_decode picks per-layer offsets once per iteration:
#
#     for layer_idx in range(D.NUM_LAYERS):
#         var is_full = layer_is_full(layer_idx)
#         if is_full:
#             var layer_offs = self.layout.full_layer      # plain copy, 38 Ints
#             var lb = rv.full_layer_base(full_idx)
#             # closures capture layer_offs, lb — no comptime slot types.
#             @parameter
#             def do_full_qk_gemv[rank: Int](rv, mut pool) -> PoolFence[BurstPool[]]:
#                 return int8_gemv[D.QK_DIM_FULL // tp, D.HIDDEN](
#                     rv.scratch_addr(full_attn_i8_lease),
#                     lb + layer_offs.q_proj,
#                     lb + layer_offs.qk_colsum,
#                     lb + layer_offs.q_proj_sc,
#                     rv.scratch_addr(qk_lease),
#                     seq_len, rv.scratch_addr(full_attn_scale_lease), pool)
#         else:
#             var layer_offs = self.layout.sliding_layer
#             var lb = rv.sliding_layer_base(sliding_idx)
#             # ... same shape
#
# Note: kernel shape params (`D.QK_DIM_FULL // tp`, `D.HIDDEN`) stay comptime
# because `tp` is comptime on `Gemma4ButterQuant[tp]` and `D.*` are comptime
# scalars. Offset reads (`lb + layer_offs.q_proj`) are pure runtime Int math.
# No type elaborated per slot. No `comptime for`.
#
# The `FL.X if is_full else SL.X` ternaries disappear — the branch happens
# once per layer where `layer_offs` is bound.
#
# Loader becomes:
#     load_weights(layout.entries, arena_bases)    # plain function, no traits
#
# Quantizer becomes:
#     quantize_weights(layout.entries, ...)        # iterates by tag
# =============================================================================


# =============================================================================
# Quantizer task — self-contained work unit.
#
# Each task carries its own gamma link (empty for no-absorption). The old
# design had a separate ABSORBED task that set a `has_gamma` latch, and a
# following GAMMA_QUANTIZE task that consumed it implicitly — a stateful,
# order-dependent two-task handshake. Here the link is explicit per task
# and processing order is free.
# =============================================================================


@fieldwise_init
struct QuantizeTask(Copyable):
    var kind: Int              # WeightTag.QUANTIZABLE / GAMMA_QUANTIZABLE /
                               # PER_BLOCK_QUANTIZABLE / PASSTHROUGH.
                               # ABSORBED never appears — gamma linkage is
                               # carried by gamma_src below.
    var src_name: String       # source safetensors tensor name (full, prefixed)
    var gamma_src: String      # "" if no gamma; else the norm tensor name to
                               # read + absorb into this weight


def build_sliding_quantizer_tasks(prefix: String, out: mut List[QuantizeTask]):
    """Emit one layer's quantizer tasks. Ordering is free — each task is
    self-contained. Gemma4 currently uses runtime norms (no absorption),
    so gamma_src is always empty; included here as the shape smollm2 will
    use once ported.
    """
    comptime Q = WeightTag.QUANTIZABLE
    comptime P = WeightTag.PASSTHROUGH

    # Attention projections. Gemma4 applies input_layernorm at runtime, so
    # these do not absorb γ. For smollm2, gamma_src="input_layernorm.weight".
    out.append(QuantizeTask(Q, prefix + "self_attn.q_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "self_attn.k_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "self_attn.v_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "self_attn.o_proj.weight", ""))
    # Dense MLP
    out.append(QuantizeTask(Q, prefix + "mlp.gate_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "mlp.up_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "mlp.down_proj.weight", ""))
    # Router + experts
    out.append(QuantizeTask(Q, prefix + "router.proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "experts.gate_up_proj", ""))
    out.append(QuantizeTask(Q, prefix + "experts.down_proj", ""))
    # Norms + scalars kept in the source as passthrough bf16.
    out.append(QuantizeTask(P, prefix + "input_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_attention_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "pre_feedforward_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "pre_feedforward_layernorm_2.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm_1.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm_2.weight", ""))
    out.append(QuantizeTask(P, prefix + "self_attn.q_norm.weight", ""))
    out.append(QuantizeTask(P, prefix + "self_attn.k_norm.weight", ""))
    out.append(QuantizeTask(P, prefix + "router.scale", ""))
    out.append(QuantizeTask(P, prefix + "router.per_expert_scale", ""))
    out.append(QuantizeTask(P, prefix + "layer_scalar", ""))


def build_full_quantizer_tasks(prefix: String, out: mut List[QuantizeTask]):
    """Full-attention layer — same as sliding, minus V_PROJ (K=V shared)."""
    comptime Q = WeightTag.QUANTIZABLE
    comptime P = WeightTag.PASSTHROUGH
    out.append(QuantizeTask(Q, prefix + "self_attn.q_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "self_attn.k_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "self_attn.o_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "mlp.gate_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "mlp.up_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "mlp.down_proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "router.proj.weight", ""))
    out.append(QuantizeTask(Q, prefix + "experts.gate_up_proj", ""))
    out.append(QuantizeTask(Q, prefix + "experts.down_proj", ""))
    out.append(QuantizeTask(P, prefix + "input_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_attention_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "pre_feedforward_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "pre_feedforward_layernorm_2.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm_1.weight", ""))
    out.append(QuantizeTask(P, prefix + "post_feedforward_layernorm_2.weight", ""))
    out.append(QuantizeTask(P, prefix + "self_attn.q_norm.weight", ""))
    out.append(QuantizeTask(P, prefix + "self_attn.k_norm.weight", ""))
    out.append(QuantizeTask(P, prefix + "router.scale", ""))
    out.append(QuantizeTask(P, prefix + "router.per_expert_scale", ""))
    out.append(QuantizeTask(P, prefix + "layer_scalar", ""))


def build_gemma4_quantizer_tasks() -> List[QuantizeTask]:
    """All 30 layers + host-only (embed, final norm), in any order.

    Ordering is genuinely free: each task carries its own gamma link. Here
    we walk layers sequentially because it makes the resulting output file
    easy to read, not because the quantizer requires it.
    """
    var tasks = List[QuantizeTask]()
    for i in range(D.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if layer_is_full(i):
            build_full_quantizer_tasks(prefix, tasks)
        else:
            build_sliding_quantizer_tasks(prefix, tasks)

    comptime PBQ = WeightTag.PER_BLOCK_QUANTIZABLE
    comptime P = WeightTag.PASSTHROUGH
    tasks.append(QuantizeTask(P,   "model.language_model.norm.weight", ""))
    tasks.append(QuantizeTask(PBQ, "model.language_model.embed_tokens.weight", ""))
    return tasks^


# =============================================================================
# Validation stub — a sanity pass build_gemma4_layout can call to catch
# layout bugs early. Runtime, so failures print a diff instead of emitting
# a 200-line comptime trace.
# =============================================================================


def validate_layout(layout: Gemma4ModelLayout):
    """Cheap runtime invariant checks."""
    # Catalog offsets are strictly non-decreasing within each layer.
    var prev_off = -1
    var prev_layer = -1
    for e in layout.entries:
        # Skip host-only entries for the monotonicity check.
        if e.target_rank == HOST_RANK:
            continue
        if e.arena_offset <= prev_off:
            print("layout bug: non-monotonic offset for", e.name,
                  "at", e.arena_offset, "after", prev_off)
        prev_off = e.arena_offset
    # Total bytes account for every layer slot exactly once.
    var expected = D.NUM_SLIDING_LAYERS * layout.sliding_stride + D.NUM_FULL_LAYERS * layout.full_stride
    if layout.distributed_bytes != expected:
        print("layout bug: distributed_bytes", layout.distributed_bytes,
              "expected", expected)
