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
    TaskVisitor, QuantizeSpec,
    PerRow, PerRowAbsorbed, PerBlockAbsorbed, TwoSidedAbsorbed,
    HOST_RANK,
)
from quant.source_format import Bf16Converter, Fp8E4M3Block128Converter
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    StaticTensorView, DynamicTensorView,
    static_tensor_view, dynamic_tensor_view,
    scratch_tensor_view, scratch_ptr,
    LayerShard, LayerBuilder,
)
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY, embed_lookup
from kernels.reductions import small_allreduce, ring_broadcast
from kernels.kv_rotors import init_rope_tables
from modeling.linear_borrow_pool import ScratchPool, ScratchLease
from modeling.loader import discover_shards, load_weights_from_descs
from simd_math import sqrt

from experimental3.profiler import (
    PhaseTiming, phase_timing_from_points, finish_single_pool_fence,
    ForwardSample, ForwardLogger,
)
from experimental3.init_weights import colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.common_math import BF16Ptr, F32Ptr, I8Ptr, U8Ptr, rms_reduce_bf16, inv_rms_from_sum_sq
from experimental3.kv_cache import Gemma4KVCache, CACHE_WIDTH
from experimental3.kernels.dispatch_kernels import (
    rmsnorm_gamma_fwht_per_block_quantize,
    int8_gemv, int8_gemv_blocked,
    lm_head_gemv,
)
from experimental3.kernels.dispatch_args import RmsNormFwhtQuantArgs
from experimental3.kernels.rmsnorm import (
    rmsnorm_fwht_quant_worker, accumulate_expert_outputs,
)

from minimax.kernels.router import TopKResult, router_merge_and_renorm
from minimax.kernels.dispatch_args import (
    RouterCandidate, RouterMergeArgs, RmsNormDualOutputArgs, AttnGroupArgs,
)
from minimax.kernels.dispatch_kernels import (
    router_fused_dispatch,
    kv_write_dispatch,
    q_prep_kernel,
    chunked_score_dispatch,
    minimax_moe_phase1,
    minimax_moe_phase2,
)
from minimax.kernels.rmsnorm import rmsnorm_dual_output_worker
from minimax.kernels.attention import merge_and_quantize_kernel


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
    comptime ROPE_PAIR_STRIDE = Self.ROPE_DIM // 2

    comptime MOE_INTERMEDIATE = 1536
    comptime NUM_EXPERTS = 256
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 200064
    comptime BOS_TOKEN_ID = 200034
    comptime ROLE_TOKEN_ID = 200019
    comptime EOS_TOKEN_ID = 200020
    comptime RMS_NORM_EPS = 1e-6

    comptime MAX_SEQ_LEN = 4096

    # Comptime ceiling on how many parallel workers the chunked-attention
    # dispatcher will fan out to for a single KV group. Sizes stack arrays in
    # the dispatcher, merge kernel, and cross-chunk scratch buffer. Must be
    # >= any pool_capacity we will ever see at runtime.
    comptime MAX_ATTN_CHUNKS = 32


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
    comptime QKV_LOCAL = Self.Q_LOCAL + 2 * Self.KV_LOCAL
    comptime NUM_HEADS_LOCAL = C.NUM_HEADS // Self.tp
    comptime NUM_KV_HEADS_LOCAL = C.NUM_KV_HEADS // Self.tp
    comptime EXPERTS_LOCAL = C.NUM_EXPERTS // Self.tp


# =============================================================================
# Typed refs
# =============================================================================


@fieldwise_init
struct AttnRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = MiniMaxShapes[Self.tp]

    var qkv_proj:    SlotOffset[I8,  Shape[C.Q_DIM + 2 * C.KV_DIM, C.HIDDEN]]
    var qkv_proj_sc: SlotOffset[F32, Shape[C.Q_DIM + 2 * C.KV_DIM, 1]]
    var o_proj:      SlotOffset[I8,  Shape[C.HIDDEN, C.Q_DIM]]
    var o_proj_sc:   SlotOffset[F32, Shape[C.HIDDEN, 1]]

    var q_norm:      SlotOffset[BF16, Shape[C.Q_DIM,  1]]
    var k_norm:      SlotOffset[BF16, Shape[C.KV_DIM, 1]]

    var qkv_colsum:  Int
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

    # --- Attention projections (fused QKV: weights contiguous, scales contiguous) ---
    comptime qkv_n_loc = q_n_loc + 2 * kv_n_loc
    var qkv_proj_off = b.q(e, "self_attn.q_proj.weight", C.Q_DIM, H, ROW)
    _ = b.q(e, "self_attn.k_proj.weight", C.KV_DIM, H, ROW)
    _ = b.q(e, "self_attn.v_proj.weight", C.KV_DIM, H, ROW)
    var qkv_proj = SlotOffset[I8, Shape[C.Q_DIM + 2 * C.KV_DIM, H]](qkv_proj_off)

    var qkv_sc_off = b.f(e, "self_attn.q_proj.weight_scale", C.Q_DIM, 1, ROW)
    _ = b.f(e, "self_attn.k_proj.weight_scale", C.KV_DIM, 1, ROW)
    _ = b.f(e, "self_attn.v_proj.weight_scale", C.KV_DIM, 1, ROW)
    var qkv_proj_sc = SlotOffset[F32, Shape[C.Q_DIM + 2 * C.KV_DIM, 1]](qkv_sc_off)

    var o_proj = SlotOffset[I8, Shape[H, C.Q_DIM]](
        b.q(e, "self_attn.o_proj.weight", H, C.Q_DIM, COL))
    var o_proj_sc = SlotOffset[F32, Shape[H, 1]](
        b.f(e, "self_attn.o_proj.weight_scale", H, 1, REPL))

    var q_norm = SlotOffset[BF16, Shape[C.Q_DIM, 1]](
        b.bf(e, "self_attn.q_norm.weight", C.Q_DIM, 1, ROW))
    var k_norm = SlotOffset[BF16, Shape[C.KV_DIM, 1]](
        b.bf(e, "self_attn.k_norm.weight", C.KV_DIM, 1, ROW))

    var qkv_colsum = b.colsum(qkv_n_loc * 4)
    var o_colsum = b.colsum(H * o_num_blk * 4)

    var attn = AttnRefs[tp](
        qkv_proj=qkv_proj, qkv_proj_sc=qkv_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm,
        qkv_colsum=qkv_colsum, o_colsum=o_colsum,
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

    comptime HPG = C.HPG
    comptime PARTIAL_F32S = C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM)
    comptime attn_peak = (
        f32
        + S.QKV_LOCAL * bf16
        + HPG * C.HEAD_DIM * i8
        + HPG * f32
        + HPG * f32
        + S.Q_LOCAL * i8
        + S.NUM_HEADS_LOCAL * f32
        + PARTIAL_F32S * f32
    )

    comptime router_candidate_bytes = 16   # RouterCandidate: i32 + f32 + f32 + pad
    comptime topk_result_bytes = C.TOP_K * (8 + f32)   # Int + Float32 per slot
    comptime moe_peak = (
        C.HIDDEN * i8 + C.HIDDEN * f32 + f32
        + MAX_POOL_CAPACITY * C.TOP_K * router_candidate_bytes
        + topk_result_bytes
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

    comptime KV_HEADS_LOCAL = C.NUM_KV_HEADS // tp
    comptime HEADS_LOCAL = C.NUM_HEADS // tp
    comptime kv_cache_stride = Gemma4KVCache[
        C.MAX_SEQ_LEN, C.HEAD_DIM, KV_HEADS_LOCAL, HEADS_LOCAL].TOTAL_BYTES
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
# TP dispatch helper
# =============================================================================


def tp_parallel[tp: Int,
    body: def[rank: Int](MiniMaxM27Topology[tp], mut BurstPool[]) capturing -> PoolFence[BurstPool[]],
](
    topos: InlineArray[MiniMaxM27Topology[tp], tp],
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
    @staticmethod
    def describe_quantization[V: TaskVisitor](mut visitor: V) -> Bool:
        comptime Fp8 = Fp8E4M3Block128Converter
        comptime Bf16 = Bf16Converter
        var ok = True

        for i in range(C.NUM_LAYERS):
            var p = "model.layers." + String(i) + "."
            var input_gamma = p + "input_layernorm.weight"
            var post_attn_gamma = p + "post_attention_layernorm.weight"

            ok &= visitor.quantize(PerRowAbsorbed[Fp8](p + "self_attn.q_proj.weight",
                FWHT_BLK_HIDDEN, input_gamma))
            ok &= visitor.quantize(PerRowAbsorbed[Fp8](p + "self_attn.k_proj.weight",
                FWHT_BLK_HIDDEN, input_gamma))
            ok &= visitor.quantize(TwoSidedAbsorbed[Fp8](p + "self_attn.v_proj.weight",
                FWHT_BLK_HIDDEN, input_gamma, C.HEAD_DIM))
            ok &= visitor.quantize(PerRow[Fp8](p + "self_attn.o_proj.weight",
                C.HEAD_DIM))

            for j in range(C.NUM_EXPERTS):
                var ep = p + "block_sparse_moe.experts." + String(j) + "."
                ok &= visitor.quantize(PerRowAbsorbed[Fp8](ep + "w1.weight",
                    FWHT_BLK_HIDDEN, post_attn_gamma))
                ok &= visitor.quantize(PerRowAbsorbed[Fp8](ep + "w3.weight",
                    FWHT_BLK_HIDDEN, post_attn_gamma))
                ok &= visitor.quantize(PerRow[Fp8](ep + "w2.weight",
                    FWHT_BLK_MOE_DOWN))

            ok &= visitor.passthrough(p + "input_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "post_attention_layernorm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "self_attn.q_norm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "self_attn.k_norm.weight", DType.bfloat16)
            ok &= visitor.passthrough(p + "block_sparse_moe.gate.weight", DType.float32)
            ok &= visitor.passthrough(p + "block_sparse_moe.e_score_correction_bias",
                DType.float32)

        ok &= visitor.passthrough("model.norm.weight", DType.bfloat16)
        ok &= visitor.passthrough("model.embed_tokens.weight", DType.bfloat16)
        ok &= visitor.quantize(PerBlockAbsorbed[Bf16]("lm_head.weight",
            LM_OUTPUT_HEAD_FWHT_BLK, "model.norm.weight"))
        return ok

    def report_profile(self, label: String):
        self.profile.report(label)

    def reset_profile(mut self):
        self.profile.clear()

    def init_state(mut self):
        comptime MAX_PACK_BYTES = (C.Q_DIM + 2 * C.KV_DIM) * C.HIDDEN
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

                comptime qkv_local = q_local + 2 * kv_local
                colsum_at(base, lb - base + layer.attn.qkv_proj.offset, lb - base + layer.attn.qkv_colsum,
                    qkv_local, C.HIDDEN)
                pack_at(base, lb - base + layer.attn.qkv_proj.offset, qkv_local, C.HIDDEN, pack_scratch)

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

        var result = load_weights_from_descs(plan.descs, shards, arena_bases, numa_topo)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        print("DBG: prefault begin")
        for rank in range(Self.tp):
            _ = arenas[rank].prefault(plan.topology.distributed_bytes, plan.topology.state_bytes)
        print("DBG: prefault done")

        var main_pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            main_pools.push(BurstPool[].for_numa_node(numa, numa_topo[rank], headroom=2))
        print("DBG: main_pools done")

        var topos = InlineArray[MiniMaxM27Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))
        print("DBG: topos bind done")

        var scratch = ScratchPool(plan.topology.scratch_capacity)
        print("DBG: scratch pool done")
        var model = Self(arenas^, main_pools^, scratch^, topos)
        print("DBG: model ctor done, calling init_state")
        model.init_state()
        print("DBG: init_state returned")
        return model^

    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> Int32:
        comptime S = MiniMaxShapes[Self.tp]
        comptime EPS = Float32(C.RMS_NORM_EPS)
        comptime seq_len = 1
        comptime Q_LOCAL = S.Q_LOCAL
        comptime KV_LOCAL = S.KV_LOCAL
        comptime HPG = C.HPG
        comptime HEADS_PER_RANK = S.NUM_HEADS_LOCAL
        comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
        comptime X_SLOT = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]
        comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
        comptime Q_GROUP_BF16 = HPG * C.HEAD_DIM * 2
        comptime K_HEAD_BF16 = C.HEAD_DIM * 2
        comptime V_HEAD_BF16 = C.HEAD_DIM * 2
        comptime PARTIAL_F32S = C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM)

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(pos)
        var topos = self.topos
        var host = topos[0]
        var mp = self.pool_ptrs()

        # --- Embed (host rank) ---
        var t_embed0 = Int(perf_counter_ns())
        var embed_fence = embed_lookup(
            host.host.embed.bound(host.arena_base),
            tokens_ptr, host.x_main(seq_len),
            self.main_pools[0])
        var t_embed1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("embed"), finish_single_pool_fence(t_embed0, t_embed1, embed_fence^))

        var t_bcast0 = Int(perf_counter_ns())
        ring_broadcast[X_SLOT, Self.tp](
            host.x_main(seq_len).ptr, self.x_main_ptrs(seq_len), seq_len, mp)
        sample.add(self.profile.phase("broadcast"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_bcast0))

        var act_scale_lease = self.scratch.borrow[Float32, 1]()

        for layer_idx in range(C.NUM_LAYERS):

            # =============================================================
            # ATTENTION BLOCK
            # =============================================================

            # Phase 1: input norm + quantize.
            # qkv_lease is the longest-lived attention lease (held through the
            # entire attention block). Borrow it first so transient Phase-1
            # leases can release in LIFO order after Phase 2.
            comptime QKV_LOCAL = S.QKV_LOCAL
            var qkv_lease = self.scratch.borrow[Scalar[DType.bfloat16], QKV_LOCAL]()
            var attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()

            var t_attn_quant0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var lb = topos[r].layer_base(layer_idx)
                var layer = topos[r].layers.proto
                var args = RmsNormFwhtQuantArgs(
                    BF16Ptr(unsafe_from_address=topos[r].x_main(seq_len).ptr),
                    BF16Ptr(unsafe_from_address=lb + layer.body.input_norm_sqrt_off),
                    I8Ptr(unsafe_from_address=topos[r].scratch_addr(attn_i8_lease)),
                    F32Ptr(unsafe_from_address=topos[r].scratch_addr(attn_work_lease)),
                    F32Ptr(unsafe_from_address=topos[r].scratch_addr(act_scale_lease)),
                    EPS, 0, seq_len)
                rmsnorm_fwht_quant_worker[C.HIDDEN, FWHT_BLK_HIDDEN, True, False](args)
            sample.add(self.profile.phase("attn_quantize"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_quant0))

            # Phase 2: fused QKV projection (contiguous output)

            @parameter
            def do_qkv_gemv[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                return int8_gemv[QKV_LOCAL, C.HIDDEN](
                    topo.scratch_addr(attn_i8_lease),
                    layer.attn.qkv_proj.addr(lb),
                    lb + layer.attn.qkv_colsum,
                    layer.attn.qkv_proj_sc.addr(lb),
                    topo.scratch_addr(qkv_lease),
                    seq_len, topo.scratch_addr(act_scale_lease), pool)
            sample.add(self.profile.phase("attn_proj"), tp_parallel[Self.tp, do_qkv_gemv](topos, mp))

            attn_work_lease^.release()
            attn_i8_lease^.release()

            # Offsets into local QKV buffer
            comptime Q_OFF = 0
            comptime K_OFF = Q_LOCAL * 2
            comptime V_OFF = (Q_LOCAL + KV_LOCAL) * 2

            # Phase 3: scalar sum_sq allreduce for full-vector Q/K norm
            var t_norm_prep0 = Int(perf_counter_ns())
            var global_q_ss = Float32(0)
            var global_k_ss = Float32(0)
            for r in range(Self.tp):
                var q_ptr = BF16Ptr(unsafe_from_address=topos[r].scratch_addr(qkv_lease) + Q_OFF)
                var k_ptr = BF16Ptr(unsafe_from_address=topos[r].scratch_addr(qkv_lease) + K_OFF)
                global_q_ss += rms_reduce_bf16[Q_LOCAL](q_ptr)
                global_k_ss += rms_reduce_bf16[KV_LOCAL](k_ptr)
            sample.add(self.profile.phase("norm_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_norm_prep0))
            var inv_rms_q = inv_rms_from_sum_sq(global_q_ss, C.Q_DIM, EPS)
            var inv_rms_k = inv_rms_from_sum_sq(global_k_ss, C.KV_DIM, EPS)

            # Phase 4: K/V cache write (NKV_LOCAL parallel jobs per rank)
            @parameter
            def do_kv_write[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                return kv_write_dispatch[
                    C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
                    C.MAX_SEQ_LEN, KV_PER_RANK](
                    topo.scratch_addr(qkv_lease) + Q_OFF,
                    topo.scratch_addr(qkv_lease) + K_OFF,
                    topo.scratch_addr(qkv_lease) + V_OFF,
                    layer.attn.q_norm.addr(lb),
                    layer.attn.k_norm.addr(lb),
                    topo.rope_cos_row(pos), topo.rope_sin_row(pos),
                    inv_rms_q, inv_rms_k,
                    topo.kv_cache_base(layer_idx), pos,
                    pool)
            sample.add(self.profile.phase("kv_write"), tp_parallel[Self.tp, do_kv_write](topos, mp))

            # Phase 5: per-KV-group Q prep + chunked scoring + merge/quantize.
            # LIFO order: persistent leases (live through o_proj) at the bottom,
            # transient per-iteration leases on top so they release in reverse.
            var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_LOCAL]()
            var attn_head_sc_lease = self.scratch.borrow[Float32, HEADS_PER_RANK]()
            var q_i8_lease = self.scratch.borrow[Scalar[DType.int8], HPG * C.HEAD_DIM]()
            var qi_biases_lease = self.scratch.borrow[Float32, HPG]()
            var q_scales_lease = self.scratch.borrow[Float32, HPG]()
            var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()

            var context_len = pos + 1

            for kv in range(KV_PER_RANK):
                var t_qprep0 = Int(perf_counter_ns())
                for r in range(Self.tp):
                    var lb = topos[r].layer_base(layer_idx)
                    var layer = topos[r].layers.proto
                    var qp_args = AttnGroupArgs()
                    qp_args.q_bf16_base = BF16Ptr(unsafe_from_address=topos[r].scratch_addr(qkv_lease) + Q_OFF + kv * Q_GROUP_BF16)
                    qp_args.q_norm_ptr = BF16Ptr(unsafe_from_address=layer.attn.q_norm.addr(lb) + kv * HPG * C.HEAD_DIM * 2)
                    qp_args.cos_ptr = F32Ptr(unsafe_from_address=topos[r].rope_cos_row(pos))
                    qp_args.sin_ptr = F32Ptr(unsafe_from_address=topos[r].rope_sin_row(pos))
                    qp_args.inv_rms_q = inv_rms_q
                    qp_args.qi_out = I8Ptr(unsafe_from_address=topos[r].scratch_addr(q_i8_lease))
                    qp_args.head_scale_ptr = F32Ptr(unsafe_from_address=topos[r].scratch_addr(qi_biases_lease))
                    qp_args.context_len = topos[r].scratch_addr(q_scales_lease)
                    q_prep_kernel[C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE, HPG](qp_args)
                sample.add(self.profile.phase("q_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_qprep0))

                @parameter
                def do_chunk_score[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    return chunked_score_dispatch[
                        C.HEAD_DIM, HPG, C.MAX_SEQ_LEN, KV_PER_RANK, C.MAX_ATTN_CHUNKS](
                        topo.scratch_addr(q_i8_lease),
                        topo.scratch_addr(qi_biases_lease),
                        topo.scratch_addr(q_scales_lease),
                        topo.kv_cache_base(layer_idx), kv,
                        context_len, Int(pool.capacity),
                        topo.scratch_addr(partial_lease),
                        pool)
                sample.add(self.profile.phase("attention"), tp_parallel[Self.tp, do_chunk_score](topos, mp))

                var t_merge_q0 = Int(perf_counter_ns())
                var num_pg = (context_len + CACHE_WIDTH - 1) // CACHE_WIDTH
                for r in range(Self.tp):
                    var nc = min(Int(self.main_pools[r].capacity), C.MAX_ATTN_CHUNKS)
                    if nc > num_pg:
                        nc = num_pg
                    if nc > 0:
                        merge_and_quantize_kernel[C.HEAD_DIM, HPG, C.MAX_ATTN_CHUNKS](
                            F32Ptr(unsafe_from_address=topos[r].scratch_addr(partial_lease)),
                            nc,
                            I8Ptr(unsafe_from_address=topos[r].scratch_addr(attn_qi_lease) + kv * HPG * C.HEAD_DIM),
                            F32Ptr(unsafe_from_address=topos[r].scratch_addr(attn_head_sc_lease) + kv * HPG * 4))
                sample.add(self.profile.phase("merge_quant"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_merge_q0))

            partial_lease^.release()
            q_scales_lease^.release()
            qi_biases_lease^.release()
            q_i8_lease^.release()

            # Phase 6: O projection
            @parameter
            def do_o_proj[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                return int8_gemv_blocked[C.HIDDEN, Q_LOCAL, C.HEAD_DIM](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(attn_qi_lease)),
                    U8Ptr(unsafe_from_address=layer.attn.o_proj.addr(lb)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(attn_head_sc_lease)),
                    layer.attn.o_proj_sc.bound(lb).as_ptr(),
                    F32Ptr(unsafe_from_address=lb + layer.attn.o_colsum),
                    topo.x_residual(seq_len).as_ptr(),
                    seq_len, pool)
            sample.add(self.profile.phase("o_proj"), tp_parallel[Self.tp, do_o_proj](topos, mp))

            attn_head_sc_lease^.release()
            attn_qi_lease^.release()
            qkv_lease^.release()

            # Allreduce O-proj + fused residual add: x_main += sum(x_residual)
            var t_attn_reduce0 = Int(perf_counter_ns())
            small_allreduce[X_SLOT, Self.tp, residual_add=True](
                self.x_residual_ptrs(seq_len), seq_len, mp,
                self.x_main_ptrs(seq_len))
            sample.add(self.profile.phase("attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))

            # =============================================================
            # FFN BLOCK
            # =============================================================

            # Phase 7: dual-output norm (split-gamma i8 + full-gamma bf16).
            # LIFO ordering: borrow in reverse order of release-time. Long-lived
            # leases (used through Phase 9/10/11) sit at the bottom; transient
            # scratch (moe_work, only needed during the dual-norm dispatch) is
            # borrowed last so it can be released first.
            var moe_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var moe_scale_lease = self.scratch.borrow[Float32, 1]()
            var normed_bf16_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()
            var moe_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()

            var t_dual_norm0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var lb = topos[r].layer_base(layer_idx)
                var layer = topos[r].layers.proto
                var args = RmsNormDualOutputArgs(
                    BF16Ptr(unsafe_from_address=topos[r].x_main(seq_len).ptr),
                    BF16Ptr(unsafe_from_address=lb + layer.body.post_attn_norm_sqrt_off),
                    BF16Ptr(unsafe_from_address=layer.body.post_attn_norm.addr(lb)),
                    I8Ptr(unsafe_from_address=topos[r].scratch_addr(moe_i8_lease)),
                    F32Ptr(unsafe_from_address=topos[r].scratch_addr(moe_work_lease)),
                    F32Ptr(unsafe_from_address=topos[r].scratch_addr(moe_scale_lease)),
                    BF16Ptr(unsafe_from_address=topos[r].scratch_addr(normed_bf16_lease)),
                    EPS, 0, seq_len)
                rmsnorm_dual_output_worker[C.HIDDEN, FWHT_BLK_HIDDEN](args)
            sample.add(self.profile.phase("dual_norm"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_dual_norm0))

            moe_work_lease^.release()

            # Phase 8: fused router (f32 GEMV + sigmoid + bias + local top-K
            # per worker) + merge. routing outlives candidates (Phase 9/10
            # dereference routing), so routing is borrowed first; candidates
            # is transient to Phase 8.
            var routing_lease = self.scratch.borrow[TopKResult[C.TOP_K], 1]()
            var candidates_lease = self.scratch.borrow[
                RouterCandidate, MAX_POOL_CAPACITY * C.TOP_K]()

            @parameter
            def do_router_fused[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                return router_fused_dispatch[C.NUM_EXPERTS, C.HIDDEN, C.TOP_K](
                    BF16Ptr(unsafe_from_address=topo.scratch_addr(normed_bf16_lease)),
                    layer.body.router_proj.bound(lb).as_ptr(),
                    layer.body.router_bias.bound(lb).as_ptr(),
                    U8Ptr(unsafe_from_address=topo.scratch_addr(candidates_lease)),
                    pool)
            sample.add(self.profile.phase("router_proj"), tp_parallel[Self.tp, do_router_fused](topos, mp))

            var t_merge0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var num_workers = min(C.NUM_EXPERTS // C.TOP_K, Int(self.main_pools[r].capacity))
                var merge_args = RouterMergeArgs(
                    U8Ptr(unsafe_from_address=topos[r].scratch_addr(candidates_lease)),
                    U8Ptr(unsafe_from_address=topos[r].scratch_addr(routing_lease)),
                    num_workers * C.TOP_K)
                router_merge_and_renorm[C.TOP_K](merge_args)
            sample.add(self.profile.phase("router_topk"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_merge0))

            candidates_lease^.release()
            # normed_bf16 and routing remain live: routing is read by Phase 9/10;
            # normed_bf16 is dead after Phase 8a but held to keep the stack LIFO
            # (it sits below routing). Released at the end of the layer.

            # Phase 9: expert gate+up (w1/w3 + SiLU)
            var expert_qi_lease = self.scratch.borrow[Scalar[DType.int8], C.TOP_K * C.MOE_INTERMEDIATE]()
            var expert_blk_scale_lease = self.scratch.borrow[Float32, C.TOP_K * MOE_DOWN_NUM_BLK]()
            var expert_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
            var local_count_lease = self.scratch.borrow[Int32, 1]()

            @parameter
            def do_expert_phase1[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                var routing = UnsafePointer[TopKResult[C.TOP_K], MutAnyOrigin](
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
                return minimax_moe_phase1[
                    C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(moe_i8_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(moe_scale_lease)),
                    routing,
                    layer.body.experts_w1.addr(lb),
                    C.MOE_INTERMEDIATE * C.HIDDEN,
                    layer.body.experts_w1_sc.addr(lb),
                    C.MOE_INTERMEDIATE * 4,
                    lb + layer.body.experts_w1_colsum,
                    C.MOE_INTERMEDIATE * 4,
                    layer.body.experts_w3.addr(lb),
                    C.MOE_INTERMEDIATE * C.HIDDEN,
                    layer.body.experts_w3_sc.addr(lb),
                    C.MOE_INTERMEDIATE * 4,
                    lb + layer.body.experts_w3_colsum,
                    C.MOE_INTERMEDIATE * 4,
                    I8Ptr(unsafe_from_address=topo.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(expert_blk_scale_lease)),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase1"), tp_parallel[Self.tp, do_expert_phase1](topos, mp))

            # Phase 10: expert down (w2)
            @parameter
            def do_expert_phase2[rank: Int](topo: MiniMaxM27Topology[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = topo.layer_base(layer_idx)
                var layer = topo.layers.proto
                var routing = UnsafePointer[TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=topo.scratch_addr(routing_lease))[]
                return minimax_moe_phase2[
                    C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK_MOE_DOWN,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    I8Ptr(unsafe_from_address=topo.scratch_addr(expert_qi_lease)),
                    F32Ptr(unsafe_from_address=topo.scratch_addr(expert_blk_scale_lease)),
                    routing,
                    layer.body.experts_w2.addr(lb),
                    C.HIDDEN * C.MOE_INTERMEDIATE,
                    layer.body.experts_w2_sc.addr(lb),
                    C.HIDDEN * 4,
                    lb + layer.body.experts_w2_colsum,
                    C.HIDDEN * MOE_DOWN_NUM_BLK * 4,
                    BF16Ptr(unsafe_from_address=topo.scratch_addr(expert_out_lease)),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase2"), tp_parallel[Self.tp, do_expert_phase2](topos, mp))

            # Phase 11: expert reduce + fused allreduce + residual add
            var t_expert_sum0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var lc = Int(UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=topos[r].scratch_addr(local_count_lease))[])
                accumulate_expert_outputs[C.HIDDEN, C.TOP_K](
                    BF16Ptr(unsafe_from_address=topos[r].scratch_addr(expert_out_lease)),
                    lc,
                    BF16Ptr(unsafe_from_address=topos[r].x_residual(seq_len).ptr))
            sample.add(self.profile.phase("expert_sum"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_expert_sum0))

            var t_ffn_reduce0 = Int(perf_counter_ns())
            small_allreduce[X_SLOT, Self.tp, residual_add=True](
                self.x_residual_ptrs(seq_len), seq_len, mp,
                self.x_main_ptrs(seq_len))
            sample.add(self.profile.phase("ffn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_ffn_reduce0))

            local_count_lease^.release()
            expert_out_lease^.release()
            expert_blk_scale_lease^.release()
            expert_qi_lease^.release()
            routing_lease^.release()
            normed_bf16_lease^.release()
            moe_scale_lease^.release()
            moe_i8_lease^.release()

        act_scale_lease^.release()

        # --- Final norm + LM head (host rank only) ---
        var lm_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
        var lm_act_blk_scale_lease = self.scratch.borrow[Float32, VOCAB_NUM_BLOCKS]()
        var lm_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
        var t_final0 = Int(perf_counter_ns())
        var final_fence = rmsnorm_gamma_fwht_per_block_quantize[
            C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK](
            host.x_main(seq_len).ptr,
            host.arena_base + host.host.lm_output_head_sqrt_gamma_off,
            host.scratch_addr(lm_act_i8_lease),
            host.scratch_addr(lm_work_lease),
            host.scratch_addr(lm_act_blk_scale_lease),
            EPS, 1, self.main_pools[0])
        var t_final1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("final_norm"), finish_single_pool_fence(t_final0, t_final1, final_fence^))

        # lm_work is dead after final_norm; release now so logit_lease (next
        # borrow) lands on top of a LIFO-clean stack.
        lm_work_lease^.release()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = scratch_tensor_view[BF16, 1, C.VOCAB_SIZE](host.scratch_base(), logit_lease, 1)
        var t_lm0 = Int(perf_counter_ns())
        var lm_fence = lm_head_gemv[
            C.VOCAB_SIZE, C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK](
            host.scratch_addr(lm_act_i8_lease),
            host.host.lm_output_head.addr(host.arena_base),
            host.scratch_addr(lm_act_blk_scale_lease),
            host.host.lm_output_head_sc.addr(host.arena_base),
            host.arena_base + host.host.lm_output_head_colsum_off,
            logit_view.ptr,
            self.main_pools[0])
        var t_lm1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("lm_head"), finish_single_pool_fence(t_lm0, t_lm1, lm_fence^))

        # Argmax (no softcap in MiniMax)
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
