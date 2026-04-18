"""MiniMax-M2.7 ButterQuant — int8 MoE, NUMA-aware tensor parallel.

Modeling skeleton. All 62 layers are full causal GQA with partial RoPE on
the first 64 of each head's 128 dims; no sliding attention, no dense MLP,
no shared expert, untied embeddings. On-disk weights are FP8 E4M3 with
128x128 F32 block scales for the large linears. The runtime representation
is butterquant int8 + per-row F32 scales; the FP8 -> BF16 -> butterquant
pipeline is not implemented here — it is a stub consumed by load().

Forward, load, quantizer task list, and init_state are stubbed. The
structural parts (config, shapes, typed refs, topology, emit, build plan)
are complete.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, BF16, F32, I8,
    Shape, ShapeLike, Mat, DynView, Bound,
    WeightDesc, DEFAULT_ALIGNMENT,
    QuantizeTask, NoQuant, Rotated, SmoothPerBlock,
    HOST_RANK, DISTRIBUTED,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    StaticTensorView, DynamicTensorView,
    static_tensor_view, dynamic_tensor_view,
    scratch_tensor_view, scratch_ptr,
    LayerShard, LayerBuilder,
)
from modeling.linear_borrow_pool import ScratchPool
from experimental3.profiler import ForwardLogger


# =============================================================================
# Config
# =============================================================================


struct MiniMaxM27Config:
    comptime HIDDEN = 3072
    comptime NUM_LAYERS = 62

    # Attention: GQA with KV expansion factor 6, partial RoPE on first 64 dims.
    comptime NUM_HEADS = 48
    comptime NUM_KV_HEADS = 8
    comptime HEAD_DIM = 128
    comptime Q_DIM = 6144       # NUM_HEADS * HEAD_DIM
    comptime KV_DIM = 1024      # NUM_KV_HEADS * HEAD_DIM
    comptime HPG = 6            # NUM_HEADS / NUM_KV_HEADS
    comptime ROPE_DIM = 64      # partial rotation: first 64 of each head
    comptime ROPE_THETA = 5_000_000
    comptime MAX_POS = 196608

    # MoE: 256 experts, top-8 sigmoid routing with additive correction bias.
    comptime MOE_INTERMEDIATE = 1536
    comptime NUM_EXPERTS = 256
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 200064
    comptime RMS_NORM_EPS = 1e-6
    comptime TIE_EMBEDDINGS = False

    comptime MAX_SEQ_LEN = 4096


comptime C = MiniMaxM27Config

# ButterQuant FWHT block sizes — power-of-2, divide the contraction dim,
# and align with head_dim at the attention boundaries.
comptime FWHT_BLK = 128              # HEAD_DIM
comptime FWHT_BLK_HIDDEN = 128
comptime FWHT_BLK_MOE_DOWN = 128     # divides MOE_INTERMEDIATE = 1536
comptime LM_HEAD_FWHT_BLK = 64
comptime VNNI_ALIGN = 64
comptime MOE_DOWN_NUM_BLK = C.MOE_INTERMEDIATE // FWHT_BLK_MOE_DOWN


# =============================================================================
# Per-TP shape aliases
# =============================================================================


struct MiniMaxShapes[tp: Int]:
    # Attention projections — Q/K/V shard output rows across ranks,
    # O shards the input contraction dim across ranks.
    comptime QProj  = Shape[C.Q_DIM,  C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime KProj  = Shape[C.KV_DIM, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime VProj  = Shape[C.KV_DIM, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime OProj  = Shape[C.HIDDEN, C.Q_DIM,  shard_m=True, tp=Self.tp]

    # Experts — ROW-shard over the virtual NE*rows axis yields
    # experts_local = NE/tp whole experts per rank.
    comptime ExpertsW1 = Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN,
                               shard_n=True, tp=Self.tp]
    comptime ExpertsW2 = Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE,
                               shard_n=True, tp=Self.tp]
    comptime ExpertsW3 = Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN,
                               shard_n=True, tp=Self.tp]

    # Router — replicated. Float32 on disk; butterquantized to int8 at load.
    comptime RouterProj = Shape[C.NUM_EXPERTS, C.HIDDEN]

    # Local derived values.
    comptime EXPERTS_LOCAL = C.NUM_EXPERTS // Self.tp
    comptime Q_LOCAL  = C.Q_DIM  // Self.tp
    comptime KV_LOCAL = C.KV_DIM // Self.tp
    comptime NUM_HEADS_LOCAL    = C.NUM_HEADS    // Self.tp
    comptime NUM_KV_HEADS_LOCAL = C.NUM_KV_HEADS // Self.tp


# =============================================================================
# Typed refs — SlotOffset families for one layer
# =============================================================================


@fieldwise_init
struct AttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    # Butterquant int8 weight slots + per-row F32 scale slots.
    var q_proj:      SlotOffset[I8,  Shape[C.Q_DIM,  C.HIDDEN]]
    var k_proj:      SlotOffset[I8,  Shape[C.KV_DIM, C.HIDDEN]]
    var v_proj:      SlotOffset[I8,  Shape[C.KV_DIM, C.HIDDEN]]
    var o_proj:      SlotOffset[I8,  Shape[C.HIDDEN, C.Q_DIM]]
    var q_proj_sc:   SlotOffset[F32, Shape[C.Q_DIM,  1]]
    var k_proj_sc:   SlotOffset[F32, Shape[C.KV_DIM, 1]]
    var v_proj_sc:   SlotOffset[F32, Shape[C.KV_DIM, 1]]
    var o_proj_sc:   SlotOffset[F32, Shape[C.HIDDEN, 1]]

    # Full-vector Q/K RMSNorm weights (BF16 on disk). Full-vector norm over
    # the concatenated projected Q (6144) / K (1024). Replicated for now;
    # sharded-Q forward will need a small sum-of-squares allreduce.
    var q_norm:      SlotOffset[BF16, Shape[C.Q_DIM,  1]]
    var k_norm:      SlotOffset[BF16, Shape[C.KV_DIM, 1]]

    # Butterquant bookkeeping.
    var q_colsum:    Int
    var k_colsum:    Int
    var v_colsum:    Int
    var o_colsum:    Int


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = MiniMaxShapes[Self.tp]

    var input_norm:     SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]

    # Router. gate weight stays F32 on disk and F32 in the arena — the
    # sigmoid + correction-bias + top-k path is magnitude-sensitive under
    # DeepSeek-style auxiliary-free routing, so we don't butterquantize it.
    var router_proj:    SlotOffset[F32, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    # e_score_correction_bias — F32 on disk, passthrough.
    var router_bias:    SlotOffset[F32, Shape[C.NUM_EXPERTS, 1]]

    # Experts — SwiGLU (w1 = gate, w3 = up, w2 = down). The checkpoint stores
    # w1 and w3 as separate per-expert tensors; they could be concat-fused at
    # quantize time but are left separate here to keep the load path 1:1 with
    # the on-disk layout.
    var experts_w1:    SlotOffset[I8,  Self.S.ExpertsW1]
    var experts_w1_sc: SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w2:    SlotOffset[I8,  Self.S.ExpertsW2]
    var experts_w2_sc: SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.HIDDEN, 1]]
    var experts_w3:    SlotOffset[I8,  Self.S.ExpertsW3]
    var experts_w3_sc: SlotOffset[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]

    var experts_w1_colsum: Int
    var experts_w3_colsum: Int
    var experts_w2_colsum: Int


@fieldwise_init
struct LayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: AttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


# =============================================================================
# State — activations, RoPE tables, KV cache, host-side globals
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
    # Final norm gamma. Absorbed into lm_head at quantize time; kept here for
    # reference / debug / non-absorbed paths.
    var final_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]

    # Untied: embed_tokens and lm_head are separate tensors.
    # embed_tokens is butterquantized per-block (no absorption) for fast lookup.
    # lm_head is butterquantized per-block and absorbs final_norm's gamma.
    var embed:      SlotOffset[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var embed_sc:   SlotOffset[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]
    var lm_head:    SlotOffset[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_head_sc: SlotOffset[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_HEAD_FWHT_BLK]]

    var embed_colsum_off:   Int
    var lm_head_colsum_off: Int
    var sqrt_gamma_off:     Int
    var inv_sqrt_gamma_off: Int


# =============================================================================
# Topology — base + offsets + stride for every per-rank view
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
# FP8 weight spec — on-disk FP8 E4M3 + 128x128 F32 block scales, to be
# dequantized to BF16 and re-quantized into the butterquant arena slot.
#
# The int8_offset and row_scale_offset name where the FINAL butterquant
# layout lives. The block/per_block_scale fields describe the butterquant
# target format, not the FP8 source.
# =============================================================================


@fieldwise_init
struct Fp8WeightSpec(Copyable, Movable):
    var weight_name: String
    var scale_name: String
    var int8_offset: Int
    var row_scale_offset: Int
    var global_rows: Int
    var global_cols: Int
    var local_rows: Int
    var local_cols: Int
    var block: Int
    var per_block_scale: Bool
    var target_rank: Int
    var shard_mode: Int


# =============================================================================
# Emit — LayerBuilder extensions for FP8 and per-expert stacks
#
# LayerBuilder.bf/f/q emit one WeightDesc that the standard loader can
# consume. FP8 matmul weights live in a parallel Fp8WeightSpec list and
# get picked up by the (stubbed) dequant-then-butterquant pipeline.
# =============================================================================


def emit_fp8[tp: Int](
    mut b: LayerBuilder,
    mut fp8_specs: List[Fp8WeightSpec],
    weight_name: String, scale_name: String,
    rows: Int, cols: Int,
    shard: Int, block: Int,
    per_block_scale: Bool = False,
    target_rank: Int = DISTRIBUTED,
) -> Tuple[Int, Int]:
    var local_rows = rows // tp if shard == LayerShard.ROW else rows
    var local_cols = cols // tp if shard == LayerShard.COL else cols
    var int8_off = align_up(b.cursor)
    b.cursor = int8_off + local_rows * local_cols
    var scale_off = align_up(b.cursor)
    b.cursor = scale_off + local_rows * size_of[Float32]()
    fp8_specs.append(Fp8WeightSpec(
        weight_name=b.layer_prefix + weight_name,
        scale_name=b.layer_prefix + scale_name,
        int8_offset=int8_off, row_scale_offset=scale_off,
        global_rows=rows, global_cols=cols,
        local_rows=local_rows, local_cols=local_cols,
        block=block, per_block_scale=per_block_scale,
        target_rank=target_rank, shard_mode=shard,
    ))
    return (int8_off, scale_off)


def emit_fp8_per_expert[tp: Int](
    mut b: LayerBuilder,
    mut fp8_specs: List[Fp8WeightSpec],
    name_prefix: String, weight_suffix: String, scale_suffix: String,
    per_expert_rows: Int, per_expert_cols: Int,
    block: Int,
) -> Tuple[Int, Int]:
    comptime experts_local = C.NUM_EXPERTS // tp
    var int8_per = per_expert_rows * per_expert_cols
    var scale_per = per_expert_rows * size_of[Float32]()
    var int8_block_off = align_up(b.cursor)
    b.cursor = int8_block_off + experts_local * int8_per
    var scale_block_off = align_up(b.cursor)
    b.cursor = scale_block_off + experts_local * scale_per
    for i in range(C.NUM_EXPERTS):
        var rank = i // experts_local
        var k = i % experts_local
        fp8_specs.append(Fp8WeightSpec(
            weight_name=b.layer_prefix + name_prefix + String(i) + weight_suffix,
            scale_name=b.layer_prefix + name_prefix + String(i) + scale_suffix,
            int8_offset=int8_block_off + k * int8_per,
            row_scale_offset=scale_block_off + k * scale_per,
            global_rows=per_expert_rows, global_cols=per_expert_cols,
            local_rows=per_expert_rows, local_cols=per_expert_cols,
            block=block, per_block_scale=False,
            target_rank=rank, shard_mode=LayerShard.REPL,
        ))
    return (int8_block_off, scale_block_off)


def emit_attn[tp: Int](
    mut b: LayerBuilder,
    mut descs: List[WeightDesc],
    mut fp8_specs: List[Fp8WeightSpec],
) -> AttnRefs[tp]:
    comptime REPL = LayerShard.REPL
    comptime ROW  = LayerShard.ROW
    comptime COL  = LayerShard.COL
    comptime H    = C.HIDDEN
    comptime BLK  = FWHT_BLK_HIDDEN

    var q_refs = emit_fp8[tp](b, fp8_specs,
        "self_attn.q_proj.weight", "self_attn.q_proj.weight_scale_inv",
        C.Q_DIM, H, ROW, BLK)
    var k_refs = emit_fp8[tp](b, fp8_specs,
        "self_attn.k_proj.weight", "self_attn.k_proj.weight_scale_inv",
        C.KV_DIM, H, ROW, BLK)
    var v_refs = emit_fp8[tp](b, fp8_specs,
        "self_attn.v_proj.weight", "self_attn.v_proj.weight_scale_inv",
        C.KV_DIM, H, ROW, BLK)
    var o_refs = emit_fp8[tp](b, fp8_specs,
        "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale_inv",
        H, C.Q_DIM, COL, C.HEAD_DIM)

    var q_norm = SlotOffset[BF16, Shape[C.Q_DIM, 1]](
        b.bf(descs, "self_attn.q_norm.weight", C.Q_DIM, 1, REPL))
    var k_norm = SlotOffset[BF16, Shape[C.KV_DIM, 1]](
        b.bf(descs, "self_attn.k_norm.weight", C.KV_DIM, 1, REPL))

    comptime q_n_loc  = C.Q_DIM  // tp
    comptime kv_n_loc = C.KV_DIM // tp
    comptime o_num_blk = C.NUM_HEADS // tp
    var q_colsum = b.colsum(q_n_loc * 4)
    var k_colsum = b.colsum(kv_n_loc * 4)
    var v_colsum = b.colsum(kv_n_loc * 4)
    var o_colsum = b.colsum(H * o_num_blk * 4)

    return AttnRefs[tp](
        q_proj=SlotOffset[I8,  Shape[C.Q_DIM,  H]](q_refs[0]),
        k_proj=SlotOffset[I8,  Shape[C.KV_DIM, H]](k_refs[0]),
        v_proj=SlotOffset[I8,  Shape[C.KV_DIM, H]](v_refs[0]),
        o_proj=SlotOffset[I8,  Shape[H, C.Q_DIM]](o_refs[0]),
        q_proj_sc=SlotOffset[F32, Shape[C.Q_DIM,  1]](q_refs[1]),
        k_proj_sc=SlotOffset[F32, Shape[C.KV_DIM, 1]](k_refs[1]),
        v_proj_sc=SlotOffset[F32, Shape[C.KV_DIM, 1]](v_refs[1]),
        o_proj_sc=SlotOffset[F32, Shape[H, 1]](o_refs[1]),
        q_norm=q_norm, k_norm=k_norm,
        q_colsum=q_colsum, k_colsum=k_colsum,
        v_colsum=v_colsum, o_colsum=o_colsum,
    )


def emit_body[tp: Int](
    mut b: LayerBuilder,
    mut descs: List[WeightDesc],
    mut fp8_specs: List[Fp8WeightSpec],
) -> BodyRefs[tp]:
    comptime REPL = LayerShard.REPL
    comptime H    = C.HIDDEN
    comptime MI   = C.MOE_INTERMEDIATE
    comptime NE   = C.NUM_EXPERTS
    comptime S    = MiniMaxShapes[tp]
    comptime experts_local = S.EXPERTS_LOCAL

    var input_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(descs, "input_layernorm.weight", H, 1, REPL))
    var post_attn_norm = SlotOffset[BF16, Shape[H, 1]](
        b.bf(descs, "post_attention_layernorm.weight", H, 1, REPL))

    # Router gate weight — F32 passthrough; the sigmoid+bias+top-k path is
    # magnitude-sensitive so no butterquantization.
    var router_proj = SlotOffset[F32, Shape[NE, H]](
        b.f(descs, "block_sparse_moe.gate.weight", NE, H, REPL))
    # e_score_correction_bias — F32 passthrough, shape (NE,).
    var router_bias = SlotOffset[F32, Shape[NE, 1]](
        b.f(descs, "block_sparse_moe.e_score_correction_bias", NE, 1, REPL))

    # Experts. w1 = gate, w3 = up, w2 = down. Per-expert entries target a
    # specific rank; the arena layout stacks experts_local experts per rank.
    var w1_refs = emit_fp8_per_expert[tp](b, fp8_specs,
        "block_sparse_moe.experts.", ".w1.weight", ".w1.weight_scale_inv",
        MI, H, FWHT_BLK_HIDDEN)
    var w3_refs = emit_fp8_per_expert[tp](b, fp8_specs,
        "block_sparse_moe.experts.", ".w3.weight", ".w3.weight_scale_inv",
        MI, H, FWHT_BLK_HIDDEN)
    var w2_refs = emit_fp8_per_expert[tp](b, fp8_specs,
        "block_sparse_moe.experts.", ".w2.weight", ".w2.weight_scale_inv",
        H, MI, FWHT_BLK_MOE_DOWN)

    var experts_w1_colsum = b.colsum(experts_local * MI * 4)
    var experts_w3_colsum = b.colsum(experts_local * MI * 4)
    var experts_w2_colsum = b.colsum(experts_local * H * MOE_DOWN_NUM_BLK * 4)

    return BodyRefs[tp](
        input_norm=input_norm, post_attn_norm=post_attn_norm,
        router_proj=router_proj,
        router_bias=router_bias,
        experts_w1=SlotOffset[I8, S.ExpertsW1](w1_refs[0]),
        experts_w1_sc=SlotOffset[F32, Shape[experts_local * MI, 1]](w1_refs[1]),
        experts_w2=SlotOffset[I8, S.ExpertsW2](w2_refs[0]),
        experts_w2_sc=SlotOffset[F32, Shape[experts_local * H, 1]](w2_refs[1]),
        experts_w3=SlotOffset[I8, S.ExpertsW3](w3_refs[0]),
        experts_w3_sc=SlotOffset[F32, Shape[experts_local * MI, 1]](w3_refs[1]),
        experts_w1_colsum=experts_w1_colsum,
        experts_w3_colsum=experts_w3_colsum,
        experts_w2_colsum=experts_w2_colsum,
    )


def emit_layer[tp: Int](
    prefix: String, layer_base: Int,
    mut descs: List[WeightDesc],
    mut fp8_specs: List[Fp8WeightSpec],
) -> Tuple[LayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    var attn = emit_attn[tp](b, descs, fp8_specs)
    var body = emit_body[tp](b, descs, fp8_specs)
    return (LayerRefs[tp](attn=attn, body=body), b.cursor)


# =============================================================================
# Scratch budget
#
# Forward is stubbed; this returns a conservative placeholder sized for the
# worst of attention prep, expert dispatch, and LM head. Tighten once the
# forward path is written.
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
        + C.NUM_EXPERTS * f32 + C.NUM_EXPERTS * f32  # router logits + sigmoid
        + C.TOP_K * 2 * f32                           # topk weights + indices
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
# Build plan — emits one prototype layer for stride, then 62 layers with
# their correct prefix, then host section, then the topology.
# =============================================================================


@fieldwise_init
struct MiniMaxM27LoadPlan[tp: Int](Movable):
    var topology: MiniMaxM27Topology[Self.tp]
    var descs: List[WeightDesc]         # non-FP8 weights, standard loader path
    var fp8_specs: List[Fp8WeightSpec]  # FP8 weights, dequant-requant pipeline


def build_minimax_plan[tp: Int]() -> MiniMaxM27LoadPlan[tp]:
    var descs = List[WeightDesc]()
    var fp8_specs = List[Fp8WeightSpec]()

    # Probe the layer stride with a throwaway emit.
    var probe_descs = List[WeightDesc]()
    var probe_fp8   = List[Fp8WeightSpec]()
    var probe = emit_layer[tp]("", 0, probe_descs, probe_fp8)
    var layer_proto  = probe[0]
    var layer_stride = probe[1]

    var layers_off = 0
    var distributed = layers_off + C.NUM_LAYERS * layer_stride

    for i in range(C.NUM_LAYERS):
        var prefix = "model.layers." + String(i) + "."
        _ = emit_layer[tp](prefix, layers_off + i * layer_stride, descs, fp8_specs)

    # State block — activations, scratch, RoPE, KV cache.
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

    # KV cache: GQA with 8 KV heads, head_dim 128. Layout is left as a
    # stubbed placeholder — the real shape depends on the butterquant cache
    # format (u8 K, i8 V, per-position F32 K and Q scales) and is deferred
    # to the forward implementation. Reserve a conservative byte count so
    # the arena plan is correct-sized.
    comptime KV_CACHE_PER_POS = (
        C.NUM_KV_HEADS * C.HEAD_DIM * 1   # K u8
        + C.NUM_KV_HEADS * C.HEAD_DIM * 1 # V i8
        + C.NUM_KV_HEADS * f32            # K scales
        + C.NUM_HEADS   * f32             # Q scales
    )
    comptime kv_cache_stride = align_up(C.MAX_SEQ_LEN * KV_CACHE_PER_POS)
    var kv_cache_off = state.reserve_bytes(C.NUM_LAYERS * kv_cache_stride)

    # Host section — final_norm, embed, lm_head, colsums, gamma tables.
    comptime HOST = LayerShard.HOST
    comptime vocab_num_blocks = C.HIDDEN // LM_HEAD_FWHT_BLK
    var host_off = align_up(distributed + state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off

    var final_norm_off = hb.bf(descs, "model.norm.weight", C.HIDDEN, 1, HOST)

    # embed_tokens — BF16 on disk. Butterquantized (per-block, no absorption)
    # at load; the int8 + F32 scale grid is what lives in host arena.
    var embed_off = hb.q(descs, "model.embed_tokens.weight",
                        C.VOCAB_SIZE, C.HIDDEN, HOST)
    var embed_sc_off = hb.f(descs, "model.embed_tokens.weight_scale",
                           C.VOCAB_SIZE, vocab_num_blocks, HOST)

    # lm_head — BF16 on disk. Butterquantized with SmoothPerBlock absorbing
    # model.norm.weight.
    var lm_head_off = hb.q(descs, "lm_head.weight",
                          C.VOCAB_SIZE, C.HIDDEN, HOST)
    var lm_head_sc_off = hb.f(descs, "lm_head.weight_scale",
                             C.VOCAB_SIZE, vocab_num_blocks, HOST)

    var embed_colsum_bytes = C.VOCAB_SIZE * vocab_num_blocks * f32
    var embed_colsum_off = hb.colsum(embed_colsum_bytes)
    var lm_head_colsum_off = hb.colsum(embed_colsum_bytes)
    var sqrt_gamma_off     = hb.cursor
    hb.cursor += C.HIDDEN * bf16
    var inv_sqrt_gamma_off = hb.cursor
    hb.cursor += C.HIDDEN * f32
    var host_bytes = hb.cursor

    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](final_norm_off),
        embed=SlotOffset[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]](embed_off),
        embed_sc=SlotOffset[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]](embed_sc_off),
        lm_head=SlotOffset[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]](lm_head_off),
        lm_head_sc=SlotOffset[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]](lm_head_sc_off),
        embed_colsum_off=embed_colsum_off,
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
    return MiniMaxM27LoadPlan[tp](topo, descs^, fp8_specs^)


# =============================================================================
# Model struct
#
# All runtime methods below are stubbed. The structural plan — config, shapes,
# typed refs, topology, emit, build plan — is the working surface. Loader,
# quantizer, init_state, and forward land in follow-up passes.
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

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    # --- Stubs --------------------------------------------------------------

    @staticmethod
    def build_quantizer_tasks() -> List[QuantizeTask]:
        # STUB — returns empty. The real list pairs every butterquantized
        # weight with its rotation block + optional gamma-source name. The
        # FP8 source weights need a new quantizer path (dequant then
        # butterquant); the task schema may need a new op kind for that.
        return List[QuantizeTask]()

    def init_state(mut self):
        # STUB — RoPE tables, colsums, VNNI packing, sqrt_gamma for lm_head,
        # and the per-rank 1/sqrt(HIDDEN) fold into router proj scales all
        # belong here. See gemma_4_moe_butterquant_tp.Gemma4ButterQuant
        # .init_state for the pattern.
        print("MiniMaxM27ButterQuant.init_state: not implemented")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        # STUB — plan + allocate + standard-loader for non-FP8 weights +
        # FP8 dequant-requant pipeline for the matmul weights + init_state.
        print("MiniMaxM27ButterQuant.load: not implemented (path =",
              String(dir_path) + ")")
        return None

    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> Int32:
        # STUB — 62 layers of: input_norm+butterquant-quantize, Q/K/V GEMV,
        # full-vector Q/K RMSNorm (TP reduce), partial RoPE (first 64 of
        # each head), KV cache write, causal attention, O projection,
        # post-attention-norm+quantize, router GEMV, sigmoid+bias+top8,
        # expert dispatch (gate=w1, up=w3, SiLU*up, down=w2), residual add,
        # then final norm + lm_head + sample.
        _ = tokens_ptr
        _ = pos
        print("MiniMaxM27ButterQuant.forward_decode: not implemented")
        return Int32(0)
