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
    ButterquantI8PerBlockAbsorbed,
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
from modeling.loader import discover_shards, load_weights_from_descs
from experimental3.profiler import ForwardLogger
from experimental3.init_weights import colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.common_math import BF16Ptr, F32Ptr
from kernels.kv_rotors import init_rope_tables


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
    # MiniMax uses standard rotate_half: pair stride = ROPE_DIM // 2 = 32.
    # Gemma4 proportional RoPE uses HEAD_DIM // 2 = 256. The pair_stride
    # parameter on write_k_head_normed / rope_apply_partial controls this.
    # inv_freq[i] = 1 / (ROPE_THETA ^ (2i / ROPE_DIM)) for i in 0..31.
    comptime ROPE_PAIR_STRIDE = ROPE_DIM // 2

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
comptime LM_OUTPUT_HEAD_FWHT_BLK = 64
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
    var input_norm_sqrt_off: Int
    var post_attn_norm_sqrt_off: Int

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

    # Source checkpoint semantics:
    # - embed_tokens.weight is an untied BF16 lookup table.
    # - lm_head.weight is an untied BF16 source tensor.
    # Runtime semantics:
    # - lm_output_head is the ButterQuant output projection derived from
    #   source lm_head.weight, keeping the fast per-block GEMV path.
    var embed:      SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_output_head:    SlotOffset[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_output_head_sc: SlotOffset[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK]]
    var lm_output_head_colsum_off: Int
    var lm_output_head_sqrt_gamma_off: Int


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

    # Placeholder — real layout needs the typed cache struct (VNNI-tiled K,
    # row-major V, and per-KV-head K/V scale arrays) once forward lands.
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


def emit_expert_block(
    mut b: LayerBuilder,
    mut e: List[WeightDesc],
    layer_prefix: String,
    weight_suffix: String,
    scale_suffix: String,
    num_experts: Int,
    rows_per_expert: Int,
    cols: Int,
    experts_local: Int,
    tp: Int,
) -> Tuple[Int, Int]:
    """Reserve contiguous expert blocks and emit per-expert WeightDescs.

    Returns (weight_offset, scale_offset) relative to layer_base.
    """
    var weight_bytes = experts_local * rows_per_expert * cols
    var w_off = ((b.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    b.cursor = w_off + weight_bytes

    var scale_bytes = experts_local * rows_per_expert * 4
    var sc_off = ((b.cursor + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    b.cursor = sc_off + scale_bytes

    for j in range(num_experts):
        var local_j = j % experts_local
        var owning_rank = j // experts_local

        var ep = layer_prefix + "block_sparse_moe.experts." + String(j) + "."
        e.append(WeightDesc(
            name=ep + weight_suffix,
            arena_offset=b.layer_base + w_off + local_j * rows_per_expert * cols,
            dtype=DType.int8, element_bytes=1,
            global_rows=rows_per_expert, global_cols=cols,
            local_rows=rows_per_expert, local_cols=cols,
            data_rows=rows_per_expert, data_cols=cols,
            quantizable=True, absorbed=False,
            target_rank=owning_rank if tp > 1 else -1,
        ))
        e.append(WeightDesc(
            name=ep + weight_suffix + scale_suffix,
            arena_offset=b.layer_base + sc_off + local_j * rows_per_expert * 4,
            dtype=DType.float32, element_bytes=4,
            global_rows=rows_per_expert, global_cols=1,
            local_rows=rows_per_expert, local_cols=1,
            data_rows=rows_per_expert, data_cols=1,
            quantizable=False, absorbed=False,
            target_rank=owning_rank if tp > 1 else -1,
        ))

    return (w_off, sc_off)


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
    comptime bf16 = size_of[Scalar[DType.bfloat16]]()

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
    var input_norm_sqrt_off = b.colsum(H * bf16)
    var post_attn_norm_sqrt_off = b.colsum(H * bf16)

    # --- Router (F32, not butterquantized) ---
    var router_proj = SlotOffset[F32, Shape[NE, H]](
        b.f(e, "block_sparse_moe.gate.weight", NE, H, REPL))
    var router_bias = SlotOffset[F32, Shape[NE, 1]](
        b.f(e, "block_sparse_moe.e_score_correction_bias", NE, 1, REPL))

    # --- Experts (w1=gate, w3=up, w2=down) ---
    # Arena stores experts contiguously (ROW-sharded over the NE*rows axis).
    # The quantizer writes per-expert tensors (experts.J.w1.weight), so we
    # reserve the contiguous block then emit per-expert WeightDescs that the
    # loader can resolve individually. Each expert's data lands at the correct
    # offset within the contiguous block.
    var experts_w1_off = emit_expert_block(
        b, e, prefix, "w1.weight", "_scale", NE, MI, H, experts_local, tp)
    var experts_w1 = SlotOffset[I8, Shape[NE * MI, H]](experts_w1_off[0])
    var experts_w1_sc = SlotOffset[F32, Shape[experts_local * MI, 1]](experts_w1_off[1])

    var experts_w3_off = emit_expert_block(
        b, e, prefix, "w3.weight", "_scale", NE, MI, H, experts_local, tp)
    var experts_w3 = SlotOffset[I8, Shape[NE * MI, H]](experts_w3_off[0])
    var experts_w3_sc = SlotOffset[F32, Shape[experts_local * MI, 1]](experts_w3_off[1])

    var experts_w2_off = emit_expert_block(
        b, e, prefix, "w2.weight", "_scale", NE, H, MI, experts_local, tp)
    var experts_w2 = SlotOffset[I8, Shape[NE * H, MI]](experts_w2_off[0])
    var experts_w2_sc = SlotOffset[F32, Shape[experts_local * H, 1]](experts_w2_off[1])

    var experts_w1_colsum = b.colsum(experts_local * MI * 4)
    var experts_w3_colsum = b.colsum(experts_local * MI * 4)
    var experts_w2_colsum = b.colsum(experts_local * H * MOE_DOWN_NUM_BLK * 4)

    var body = BodyRefs[tp](
        input_norm=input_norm, post_attn_norm=post_attn_norm,
        input_norm_sqrt_off=input_norm_sqrt_off,
        post_attn_norm_sqrt_off=post_attn_norm_sqrt_off,
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

    comptime lm_output_head_peak = (
        C.HIDDEN * i8
        + (C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK) * f32
        + C.HIDDEN * f32
        + C.VOCAB_SIZE * bf16
    )

    comptime layer_peak = attn_peak if attn_peak > moe_peak else moe_peak
    return lm_output_head_peak if lm_output_head_peak > layer_peak else layer_peak


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
    # K/V scale arrays) and will change when the forward is implemented.
    comptime KV_CACHE_PER_POS = (
        C.NUM_KV_HEADS * C.HEAD_DIM * 1
        + C.NUM_KV_HEADS * C.HEAD_DIM * 1
        + C.NUM_KV_HEADS * f32
        + C.NUM_KV_HEADS * f32
    )
    comptime kv_cache_stride = align_up(C.MAX_SEQ_LEN * KV_CACHE_PER_POS)
    var kv_cache_off = state.reserve_bytes(C.NUM_LAYERS * kv_cache_stride)

    # Host section
    comptime HOST = LayerShard.HOST
    comptime vocab_num_blocks = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
    var host_off = align_up(distributed + state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off

    var final_norm_off = hb.bf(descs, "model.norm.weight", C.HIDDEN, 1, HOST)

    var embed_off = hb.bf(descs, "model.embed_tokens.weight",
                         C.VOCAB_SIZE, C.HIDDEN, HOST)

    var lm_output_head_off = hb.q(descs, "lm_head.weight",
                                 C.VOCAB_SIZE, C.HIDDEN, HOST)
    var lm_output_head_sc_off = hb.f(descs, "lm_head.weight_scale",
                                    C.VOCAB_SIZE, vocab_num_blocks, HOST)
    var lm_output_head_colsum_bytes = C.VOCAB_SIZE * vocab_num_blocks * f32
    var lm_output_head_colsum_off = hb.colsum(lm_output_head_colsum_bytes)
    var lm_output_head_sqrt_gamma_off = hb.cursor
    hb.cursor += C.HIDDEN * bf16
    var host_bytes = hb.cursor

    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](final_norm_off),
        embed=SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]](embed_off),
        lm_output_head=SlotOffset[I8, Shape[C.VOCAB_SIZE, C.HIDDEN]](
            lm_output_head_off),
        lm_output_head_sc=SlotOffset[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]](
            lm_output_head_sc_off),
        lm_output_head_colsum_off=lm_output_head_colsum_off,
        lm_output_head_sqrt_gamma_off=lm_output_head_sqrt_gamma_off,
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
        comptime FP8_BLOCK = SourceFormat.FP8_E4M3_BLOCK128

        for i in range(C.NUM_LAYERS):
            var p = "model.layers." + String(i) + "."
            var input_gamma = p + "input_layernorm.weight"
            var post_attn_gamma = p + "post_attention_layernorm.weight"

            tasks.append(ButterquantI8PerRowAbsorbed(p + "self_attn.q_proj.weight",
                FP8_BLOCK, FWHT_BLK_HIDDEN, input_gamma))
            tasks.append(ButterquantI8PerRowAbsorbed(p + "self_attn.k_proj.weight",
                FP8_BLOCK, FWHT_BLK_HIDDEN, input_gamma))
            tasks.append(ButterquantI8PerRowAbsorbed(p + "self_attn.v_proj.weight",
                FP8_BLOCK, FWHT_BLK_HIDDEN, input_gamma))
            tasks.append(ButterquantI8PerRow(p + "self_attn.o_proj.weight",
                FP8_BLOCK, C.HEAD_DIM))

            for j in range(C.NUM_EXPERTS):
                var ep = p + "block_sparse_moe.experts." + String(j) + "."
                tasks.append(ButterquantI8PerRowAbsorbed(ep + "w1.weight",
                    FP8_BLOCK, FWHT_BLK_HIDDEN, post_attn_gamma))
                tasks.append(ButterquantI8PerRowAbsorbed(ep + "w3.weight",
                    FP8_BLOCK, FWHT_BLK_HIDDEN, post_attn_gamma))
                tasks.append(ButterquantI8PerRow(ep + "w2.weight",
                    FP8_BLOCK, FWHT_BLK_MOE_DOWN))

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
            SourceFormat.BF16, LM_OUTPUT_HEAD_FWHT_BLK, "model.norm.weight"))
        return tasks^

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    def init_state(mut self):
        comptime MAX_PACK_BYTES = C.Q_DIM * C.HIDDEN
        var numa = NumaInfo()
        var pack_arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa.plan_topology(1)[0], MAX_PACK_BYTES)
        var pack_scratch = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(pack_arena.base))

        comptime bf16 = size_of[Scalar[DType.bfloat16]]()
        comptime experts_local = MiniMaxShapes[Self.tp].EXPERTS_LOCAL
        comptime q_local = C.Q_DIM // Self.tp
        comptime kv_local = C.KV_DIM // Self.tp
        comptime o_num_blk = C.NUM_HEADS // Self.tp

        for rank in range(Self.tp):
            var topo = self.topos[rank]
            var sb = topo.state_base()
            init_rope_tables(
                topo.rope.cos.bound(sb),
                topo.rope.sin.bound(sb),
                theta=Float64(C.ROPE_THETA))

            var base = topo.arena_base
            for i in range(C.NUM_LAYERS):
                var lb = topo.layer_base(i)
                var layer = topo.layers.proto

                colsum_at(base, lb - base + layer.attn.q_proj.offset, lb - base + layer.attn.q_colsum,
                    q_local, C.HIDDEN)
                pack_at(base, lb - base + layer.attn.q_proj.offset, q_local, C.HIDDEN, pack_scratch)

                colsum_at(base, lb - base + layer.attn.k_proj.offset, lb - base + layer.attn.k_colsum,
                    kv_local, C.HIDDEN)
                pack_at(base, lb - base + layer.attn.k_proj.offset, kv_local, C.HIDDEN, pack_scratch)

                colsum_at(base, lb - base + layer.attn.v_proj.offset, lb - base + layer.attn.v_colsum,
                    kv_local, C.HIDDEN)
                pack_at(base, lb - base + layer.attn.v_proj.offset, kv_local, C.HIDDEN, pack_scratch)

                block_colsum_at(base, lb - base + layer.attn.o_proj.offset, lb - base + layer.attn.o_colsum,
                    C.HIDDEN, q_local, C.HEAD_DIM)
                pack_at(base, lb - base + layer.attn.o_proj.offset, C.HIDDEN, q_local, pack_scratch)

                colsum_at(base, lb - base + layer.body.experts_w1.offset, lb - base + layer.body.experts_w1_colsum,
                    experts_local * C.MOE_INTERMEDIATE, C.HIDDEN)
                colsum_at(base, lb - base + layer.body.experts_w3.offset, lb - base + layer.body.experts_w3_colsum,
                    experts_local * C.MOE_INTERMEDIATE, C.HIDDEN)
                for e in range(experts_local):
                    pack_at(base, lb - base + layer.body.experts_w1.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN,
                        C.MOE_INTERMEDIATE, C.HIDDEN, pack_scratch)
                    pack_at(base, lb - base + layer.body.experts_w3.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN,
                        C.MOE_INTERMEDIATE, C.HIDDEN, pack_scratch)
                    block_colsum_at(base,
                        lb - base + layer.body.experts_w2.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE,
                        lb - base + layer.body.experts_w2_colsum + e * C.HIDDEN * MOE_DOWN_NUM_BLK * 4,
                        C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK_MOE_DOWN)
                    pack_at(base, lb - base + layer.body.experts_w2.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE,
                        C.HIDDEN, C.MOE_INTERMEDIATE, pack_scratch)

                var in_gamma = BF16Ptr(unsafe_from_address=layer.body.input_norm.addr(lb))
                compute_sqrt_gamma[C.HIDDEN](
                    in_gamma,
                    BF16Ptr(unsafe_from_address=lb + layer.body.input_norm_sqrt_off))

                var pa_gamma = BF16Ptr(unsafe_from_address=layer.body.post_attn_norm.addr(lb))
                compute_sqrt_gamma[C.HIDDEN](
                    pa_gamma,
                    BF16Ptr(unsafe_from_address=lb + layer.body.post_attn_norm_sqrt_off))

            if rank == HOST_RANK:
                block_colsum_row_major_at(base,
                    topo.host.lm_output_head.offset, topo.host.lm_output_head_colsum_off,
                    C.VOCAB_SIZE, C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK)
                var fn_gamma = BF16Ptr(unsafe_from_address=base + topo.host.final_norm.offset)
                compute_sqrt_gamma[C.HIDDEN](
                    fn_gamma,
                    BF16Ptr(unsafe_from_address=base + topo.host.lm_output_head_sqrt_gamma_off))

        print("state initialized")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        if C.NUM_KV_HEADS % Self.tp != 0:
            print(
                "unsupported TP=", Self.tp,
                ": KV heads must be divisible by tp; got", C.NUM_KV_HEADS,
            )
            return None

        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_minimax_plan[Self.tp]()

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

        var result = load_weights_from_descs(plan.descs, shards, arena_bases)
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

        var topos = InlineArray[MiniMaxM27Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))

        var scratch = ScratchPool(plan.topology.scratch_capacity)
        var model = Self(arenas^, main_pools^, scratch^, topos)
        model.init_state()
        return model^

    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> Int32:
        _ = tokens_ptr
        _ = pos
        print("MiniMaxM27ButterQuant.forward_decode: not implemented")
        return Int32(0)
