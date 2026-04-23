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
    TaskVisitor, QuantizeSpec,
    PerRow, PerRowAbsorbed, PerBlockAbsorbed, TwoSidedAbsorbed,
    HOST_RANK,
)
from quant.source_format import Bf16Converter, Fp8E4M3Block128Converter
from modeling.modeling_common import (
    TensorRef, Repeated, SectionBuilder, align_up,
    LayerBuilder, ArenaLayout,
)
from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY, embed_lookup
from kernels.reductions import small_allreduce, ring_broadcast
from kernels.kv_rotors import init_rope_tables
from simd_math import set_subnormal_zeroing
from modeling.linear_borrow_pool import ScratchPool, ScratchLease, scratch_block_bytes
from modeling.loader import discover_shards, load_weights_from_descs
from simd_math import sqrt

from experimental3.profiler import (
    PhaseTiming, phase_timing_from_points, finish_single_pool_fence,
    timed_tp_parallel, ForwardSample, ForwardLogger,
)
from experimental3.init_weights import colsum_at, block_colsum_at, block_colsum_row_major_at, pack_at
from experimental3.gamma import compute_sqrt_gamma, compute_inv_sqrt_gamma
from experimental3.common_math import rms_reduce_bf16, inv_rms_from_sum_sq
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
    chunked_score_dispatch_multi,
    attn_chunk_count,
    minimax_moe_phase1,
    minimax_moe_phase2,
)
from minimax.kernels.rmsnorm import rmsnorm_dual_output_worker
from kernels.worker_init import worker_init_dispatch
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

    var qkv_proj:    TensorRef[I8,  Shape[C.Q_DIM + 2 * C.KV_DIM, C.HIDDEN]]
    var qkv_proj_sc: TensorRef[F32, Shape[C.Q_DIM + 2 * C.KV_DIM, 1]]
    var o_proj:      TensorRef[I8,  Shape[C.HIDDEN, C.Q_DIM]]
    var o_proj_sc:   TensorRef[F32, Shape[C.HIDDEN, 1]]

    var q_norm:      TensorRef[BF16, Shape[C.Q_DIM,  1]]
    var k_norm:      TensorRef[BF16, Shape[C.KV_DIM, 1]]

    var qkv_colsum:  TensorRef[F32, Shape[(C.Q_DIM // Self.tp) + 2 * (C.KV_DIM // Self.tp), 1]]
    var o_colsum:    TensorRef[F32, Shape[C.HIDDEN * (C.NUM_HEADS // Self.tp), 1]]


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = MiniMaxShapes[Self.tp]

    var input_norm:      TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm:  TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var input_norm_sqrt: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm_sqrt: TensorRef[BF16, Shape[C.HIDDEN, 1]]

    # F32 router — sigmoid + correction-bias + top-k is magnitude-sensitive.
    var router_proj: TensorRef[F32, Shape[C.NUM_EXPERTS, C.HIDDEN]]
    var router_bias: TensorRef[F32, Shape[C.NUM_EXPERTS, 1]]

    # SwiGLU experts: w1=gate, w3=up, w2=down.
    # Arena stores experts_local contiguous experts per rank (ROW-sharded
    # over the NE*rows axis). Per-expert checkpoint tensors are concatenated
    # by the quantizer pipeline into these contiguous blocks.
    var experts_w1:       TensorRef[I8,  Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN]]
    var experts_w1_sc:    TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w3:       TensorRef[I8,  Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN]]
    var experts_w3_sc:    TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w2:       TensorRef[I8,  Shape[C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE]]
    var experts_w2_sc:    TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.HIDDEN, 1]]

    var experts_w1_colsum: TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w3_colsum: TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1]]
    var experts_w2_colsum: TensorRef[F32, Shape[Self.S.EXPERTS_LOCAL * C.HIDDEN * MOE_DOWN_NUM_BLK, 1]]


@fieldwise_init
struct LayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: AttnRefs[Self.tp]
    var body: BodyRefs[Self.tp]


# =============================================================================
# State
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

    # Source checkpoint semantics:
    # - embed_tokens.weight is an untied BF16 lookup table.
    # - lm_head.weight is an untied BF16 source tensor.
    # Runtime semantics:
    # - lm_output_head is the ButterQuant output projection derived from
    #   source lm_head.weight, keeping the fast per-block GEMV path.
    var embed:      TensorRef[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_output_head:         TensorRef[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_output_head_sc:      TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK]]
    var lm_output_head_colsum:  TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK]]
    var lm_output_head_sqrt_gamma: TensorRef[BF16, Shape[C.HIDDEN, 1]]


# =============================================================================
# Topology
# =============================================================================


@fieldwise_init
struct MiniMaxM27Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var layers: Repeated[LayerRefs[Self.tp]]

    var activations: ActivationSlots
    var rope: RopeSlots[C.ROPE_DIM // 2]

    var kv_cache_off: Int
    var kv_cache_stride: Int

    var host: HostSlots

    @always_inline
    def bind(self, base: Int) -> Self:
        var t = self
        t.arena = t.arena.bind(base)
        return t


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
    var w_off = b.reserve_block(experts_local * rows_per_expert * cols)
    var sc_off = b.reserve_block(experts_local * rows_per_expert * 4)

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
            quantizable=True,
            target_rank=owning_rank if tp > 1 else -1,
        ))
        e.append(WeightDesc(
            name=ep + weight_suffix + scale_suffix,
            arena_offset=b.layer_base + sc_off + local_j * rows_per_expert * 4,
            dtype=DType.float32, element_bytes=4,
            global_rows=rows_per_expert, global_cols=1,
            local_rows=rows_per_expert, local_cols=1,
            data_rows=rows_per_expert, data_cols=1,
            quantizable=False,
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
    comptime H  = C.HIDDEN
    comptime MI = C.MOE_INTERMEDIATE
    comptime NE = C.NUM_EXPERTS
    comptime S  = MiniMaxShapes[tp]
    comptime experts_local = S.EXPERTS_LOCAL
    comptime q_n_loc  = C.Q_DIM // tp
    comptime kv_n_loc = C.KV_DIM // tp
    comptime o_num_blk = C.NUM_HEADS // tp

    comptime qkv_n_loc = q_n_loc + 2 * kv_n_loc
    var qkv_proj_off = b.qs[Shape[C.Q_DIM, H, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight")
    _ = b.qs[Shape[C.KV_DIM, H, shard_n=True, tp=tp]](e, "self_attn.k_proj.weight")
    _ = b.qs[Shape[C.KV_DIM, H, shard_n=True, tp=tp]](e, "self_attn.v_proj.weight")
    var qkv_proj = TensorRef[I8, Shape[C.Q_DIM + 2 * C.KV_DIM, H]](qkv_proj_off.offset)

    var qkv_sc_off = b.fs[Shape[C.Q_DIM, 1, shard_n=True, tp=tp]](e, "self_attn.q_proj.weight_scale")
    _ = b.fs[Shape[C.KV_DIM, 1, shard_n=True, tp=tp]](e, "self_attn.k_proj.weight_scale")
    _ = b.fs[Shape[C.KV_DIM, 1, shard_n=True, tp=tp]](e, "self_attn.v_proj.weight_scale")
    var qkv_proj_sc = TensorRef[F32, Shape[C.Q_DIM + 2 * C.KV_DIM, 1]](qkv_sc_off.offset)

    var o_proj = TensorRef[I8, Shape[H, C.Q_DIM]](
        b.qs[Shape[H, C.Q_DIM, shard_m=True, tp=tp]](e, "self_attn.o_proj.weight").offset)
    var o_proj_sc = b.fs[Shape[H, 1]](e, "self_attn.o_proj.weight_scale")

    var q_norm = TensorRef[BF16, Shape[C.Q_DIM, 1]](
        b.bfs[Shape[C.Q_DIM, 1, shard_n=True, tp=tp]](e, "self_attn.q_norm.weight").offset)
    var k_norm = TensorRef[BF16, Shape[C.KV_DIM, 1]](
        b.bfs[Shape[C.KV_DIM, 1, shard_n=True, tp=tp]](e, "self_attn.k_norm.weight").offset)

    var qkv_colsum = b.colsum_slot[F32, Shape[qkv_n_loc, 1]]()
    var o_colsum = b.colsum_slot[F32, Shape[H * o_num_blk, 1]]()

    var attn = AttnRefs[tp](
        qkv_proj=qkv_proj, qkv_proj_sc=qkv_proj_sc,
        o_proj=o_proj, o_proj_sc=o_proj_sc,
        q_norm=q_norm, k_norm=k_norm,
        qkv_colsum=qkv_colsum, o_colsum=o_colsum,
    )

    var input_norm = b.bfs[Shape[H, 1]](e, "input_layernorm.weight")
    var post_attn_norm = b.bfs[Shape[H, 1]](e, "post_attention_layernorm.weight")
    var input_norm_sqrt = b.colsum_slot[BF16, Shape[H, 1]]()
    var post_attn_norm_sqrt = b.colsum_slot[BF16, Shape[H, 1]]()

    var router_proj = b.fs[Shape[NE, H]](e, "block_sparse_moe.gate.weight")
    var router_bias = b.fs[Shape[NE, 1]](e, "block_sparse_moe.e_score_correction_bias")

    # --- Experts (w1=gate, w3=up, w2=down) ---
    # Arena stores experts contiguously (ROW-sharded over the NE*rows axis).
    # The quantizer writes per-expert tensors (experts.J.w1.weight), so we
    # reserve the contiguous block then emit per-expert WeightDescs that the
    # loader can resolve individually. Each expert's data lands at the correct
    # offset within the contiguous block.
    var experts_w1_off = emit_expert_block(
        b, e, prefix, "w1.weight", "_scale", NE, MI, H, experts_local, tp)
    var experts_w1 = TensorRef[I8, Shape[NE * MI, H]](experts_w1_off[0])
    var experts_w1_sc = TensorRef[F32, Shape[experts_local * MI, 1]](experts_w1_off[1])

    var experts_w3_off = emit_expert_block(
        b, e, prefix, "w3.weight", "_scale", NE, MI, H, experts_local, tp)
    var experts_w3 = TensorRef[I8, Shape[NE * MI, H]](experts_w3_off[0])
    var experts_w3_sc = TensorRef[F32, Shape[experts_local * MI, 1]](experts_w3_off[1])

    var experts_w2_off = emit_expert_block(
        b, e, prefix, "w2.weight", "_scale", NE, H, MI, experts_local, tp)
    var experts_w2 = TensorRef[I8, Shape[NE * H, MI]](experts_w2_off[0])
    var experts_w2_sc = TensorRef[F32, Shape[experts_local * H, 1]](experts_w2_off[1])

    var experts_w1_colsum = b.colsum_slot[F32, Shape[experts_local * MI, 1]]()
    var experts_w3_colsum = b.colsum_slot[F32, Shape[experts_local * MI, 1]]()
    var experts_w2_colsum = b.colsum_slot[F32, Shape[experts_local * H * MOE_DOWN_NUM_BLK, 1]]()

    var body = BodyRefs[tp](
        input_norm=input_norm, post_attn_norm=post_attn_norm,
        input_norm_sqrt=input_norm_sqrt,
        post_attn_norm_sqrt=post_attn_norm_sqrt,
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
    comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
    comptime PARTIAL_F32S = (
        KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM))
    comptime act_scale_bytes = scratch_block_bytes[f32]()
    comptime qkv_bytes = scratch_block_bytes[S.QKV_LOCAL * bf16]()
    comptime attn_phase1_peak = (
        act_scale_bytes
        + qkv_bytes
        + scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
    )
    comptime attn_phase2_peak = (
        act_scale_bytes
        + qkv_bytes
        + scratch_block_bytes[S.Q_LOCAL * i8]()
        + scratch_block_bytes[S.NUM_HEADS_LOCAL * f32]()
        + scratch_block_bytes[KV_PER_RANK * HPG * C.HEAD_DIM * i8]()
        + scratch_block_bytes[KV_PER_RANK * HPG * f32]()
        + scratch_block_bytes[KV_PER_RANK * HPG * f32]()
        + scratch_block_bytes[PARTIAL_F32S * f32]()
    )
    comptime attn_peak = (
        attn_phase1_peak if attn_phase1_peak > attn_phase2_peak else attn_phase2_peak
    )

    comptime router_candidate_bytes = 16   # RouterCandidate: i32 + f32 + f32 + pad
    comptime topk_result_bytes = C.TOP_K * (8 + f32)   # Int + Float32 per slot
    comptime moe_peak = (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[f32]()
        + scratch_block_bytes[MAX_POOL_CAPACITY * C.TOP_K * router_candidate_bytes]()
        + scratch_block_bytes[topk_result_bytes]()
        + scratch_block_bytes[C.TOP_K * 2 * f32]()
        + scratch_block_bytes[C.TOP_K * C.MOE_INTERMEDIATE * i8]()
        + scratch_block_bytes[C.TOP_K * MOE_DOWN_NUM_BLK * f32]()
        + scratch_block_bytes[C.TOP_K * C.HIDDEN * bf16]()
    )

    comptime lm_output_head_peak = (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[(C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK) * f32]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[C.VOCAB_SIZE * bf16]()
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
    state.cursor = distributed
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

    comptime vocab_num_blocks = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
    var host_off = align_up(state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off

    var final_norm_off = hb.bfs[Shape[C.HIDDEN, 1]](descs, "model.norm.weight", target_rank=HOST_RANK)

    var embed_off = hb.bfs[Shape[C.VOCAB_SIZE, C.HIDDEN]](descs, "model.embed_tokens.weight", target_rank=HOST_RANK)

    var lm_output_head_off = hb.qs[Shape[C.VOCAB_SIZE, C.HIDDEN]](descs, "lm_head.weight", target_rank=HOST_RANK)
    var lm_output_head_sc_off = hb.fs[Shape[C.VOCAB_SIZE, vocab_num_blocks]](descs, "lm_head.weight_scale", target_rank=HOST_RANK)
    var lm_output_head_colsum = hb.colsum_slot[F32, Shape[C.VOCAB_SIZE, vocab_num_blocks]]()
    var lm_output_head_sqrt_gamma = hb.colsum_slot[BF16, Shape[C.HIDDEN, 1]]()
    var host_bytes = hb.cursor

    var host = HostSlots(
        final_norm=final_norm_off,
        embed=embed_off,
        lm_output_head=lm_output_head_off,
        lm_output_head_sc=lm_output_head_sc_off,
        lm_output_head_colsum=lm_output_head_colsum,
        lm_output_head_sqrt_gamma=lm_output_head_sqrt_gamma,
    )

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state.bytes() - distributed,
        host_bytes=host_bytes,
        scratch_off=scratch_off,
        scratch_capacity=scratch_cap,
    )
    var topo = MiniMaxM27Topology[tp](
        arena=arena,
        layers=Repeated[LayerRefs[tp]](layer_proto, layers_off, layer_stride, C.NUM_LAYERS),
        activations=activations,
        rope=rope,
        kv_cache_off=kv_cache_off, kv_cache_stride=kv_cache_stride,
        host=host,
    )
    return MiniMaxM27LoadPlan[tp](topo, descs^)


# =============================================================================
# TP dispatch helper
# =============================================================================




# =============================================================================
# Model struct
# =============================================================================


struct MiniMaxM27ButterQuant[tp: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[Self.Pool]
    var scratch: ScratchPool
    var topos: InlineArray[MiniMaxM27Topology[Self.tp], Self.tp]
    var profile: ForwardLogger

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[Self.Pool],
        var sc: ScratchPool,
        topos: InlineArray[MiniMaxM27Topology[Self.tp], Self.tp],
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
        set_subnormal_zeroing()
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
            var base = topo.arena.base
            init_rope_tables(
                topo.rope.cos.bound(base),
                topo.rope.sin.bound(base),
                theta=Float64(C.ROPE_THETA))

            for i in range(C.NUM_LAYERS):
                var lb = topo.layers.base(base, i)
                var layer = topo.layers.proto

                comptime qkv_local = q_local + 2 * kv_local
                var qkv_slot = TensorRef[I8, Shape[qkv_local, C.HIDDEN]](
                    offset=layer.attn.qkv_proj.offset)
                colsum_at(lb, qkv_slot, layer.attn.qkv_colsum)
                pack_at(lb, qkv_slot, pack_scratch)

                var o_slot = TensorRef[I8, Shape[C.HIDDEN, q_local]](
                    offset=layer.attn.o_proj.offset)
                block_colsum_at(lb, o_slot, layer.attn.o_colsum, C.HEAD_DIM)
                pack_at(lb, o_slot, pack_scratch)

                var experts_w1_slab = TensorRef[I8, Shape[experts_local * C.MOE_INTERMEDIATE, C.HIDDEN]](
                    offset=layer.body.experts_w1.offset)
                colsum_at(lb, experts_w1_slab, layer.body.experts_w1_colsum)
                var experts_w3_slab = TensorRef[I8, Shape[experts_local * C.MOE_INTERMEDIATE, C.HIDDEN]](
                    offset=layer.body.experts_w3.offset)
                colsum_at(lb, experts_w3_slab, layer.body.experts_w3_colsum)
                for e in range(experts_local):
                    var expert_w1 = TensorRef[I8, Shape[C.MOE_INTERMEDIATE, C.HIDDEN]](
                        offset=layer.body.experts_w1.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN)
                    pack_at(lb, expert_w1, pack_scratch)
                    var expert_w3 = TensorRef[I8, Shape[C.MOE_INTERMEDIATE, C.HIDDEN]](
                        offset=layer.body.experts_w3.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN)
                    pack_at(lb, expert_w3, pack_scratch)
                    var expert_w2 = TensorRef[I8, Shape[C.HIDDEN, C.MOE_INTERMEDIATE]](
                        offset=layer.body.experts_w2.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE)
                    var expert_w2_colsum = TensorRef[F32, Shape[C.HIDDEN * MOE_DOWN_NUM_BLK, 1]](
                        offset=layer.body.experts_w2_colsum.offset + e * C.HIDDEN * MOE_DOWN_NUM_BLK * 4)
                    block_colsum_at(lb, expert_w2, expert_w2_colsum, FWHT_BLK_MOE_DOWN)
                    pack_at(lb, expert_w2, pack_scratch)

                compute_sqrt_gamma[C.HIDDEN](
                    layer.body.input_norm.bound(lb).as_ptr(),
                    layer.body.input_norm_sqrt.bound(lb).as_ptr())

                compute_sqrt_gamma[C.HIDDEN](
                    layer.body.post_attn_norm.bound(lb).as_ptr(),
                    layer.body.post_attn_norm_sqrt.bound(lb).as_ptr())

            if rank == HOST_RANK:
                block_colsum_row_major_at(base,
                    topo.host.lm_output_head, topo.host.lm_output_head_colsum,
                    LM_OUTPUT_HEAD_FWHT_BLK)
                compute_sqrt_gamma[C.HIDDEN](
                    topo.host.final_norm.bound(base).as_ptr(),
                    topo.host.lm_output_head_sqrt_gamma.bound(base).as_ptr())

        for rank in range(Self.tp):
            worker_init_dispatch[C.HPG](self.main_pools[rank]).join()
        print("state initialized")

    @staticmethod
    def load(
        dir_path: Path,
        numa: NumaInfo,
        numa_topo: NumaTopology,
        var main_pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
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

        print("DBG: prefault begin")
        for rank in range(Self.tp):
            _ = arenas[rank].prefault(plan.topology.arena.distributed_bytes, plan.topology.arena.state_bytes)
        print("DBG: prefault done")

        var topos = InlineArray[MiniMaxM27Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))
        print("DBG: topos bind done")

        var scratch = ScratchPool(plan.topology.arena.scratch_capacity)
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
        comptime XShape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
        comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
        comptime Q_GROUP_BF16 = HPG * C.HEAD_DIM * 2
        comptime K_HEAD_BF16 = C.HEAD_DIM * 2
        comptime V_HEAD_BF16 = C.HEAD_DIM * 2
        comptime PARTIAL_F32S = (
            KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM))
        comptime ROPE_HALF = C.ROPE_DIM // 2

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(pos)
        var topos = self.topos
        var host = topos[0]

        # --- Embed (host rank) ---
        var t_embed0 = Int(perf_counter_ns())
        var embed_fence = embed_lookup(
            host.host.embed.bound(host.arena.base),
            tokens_ptr,
            host.activations.x_main.bound_dyn(host.arena.base, seq_len),
            self.main_pools[0])
        var t_embed1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("embed"), finish_single_pool_fence(t_embed0, t_embed1, embed_fence^))

        var t_bcast0 = Int(perf_counter_ns())
        ring_broadcast[BF16, XShape, Self.tp](
            host.activations.x_main.addr(host.arena.base),
            self.x_main_ptrs(seq_len), seq_len, self.main_pools)
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
                var topo_r = topos[r]
                var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                var layer = topo_r.layers.proto
                var sb = topo_r.arena.scratch_base()
                var args = RmsNormFwhtQuantArgs(
                    topo_r.activations.x_main.bound_dyn(topo_r.arena.base, seq_len).as_ptr[DType.bfloat16](),
                    layer.body.input_norm_sqrt.bound(lb).as_ptr[DType.bfloat16](),
                    attn_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.int8](),
                    attn_work_lease.view[F32, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.float32](),
                    act_scale_lease.view[F32, Shape[1, 1]](sb, 1).as_ptr[DType.float32](),
                    EPS, 0, seq_len)
                rmsnorm_fwht_quant_worker[C.HIDDEN, FWHT_BLK_HIDDEN, True, False](args)
            sample.add(self.profile.phase("attn_quantize"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_quant0))

            # Phase 2: fused QKV projection (contiguous output)

            @parameter
            def do_qkv_gemv[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                return int8_gemv[QKV_LOCAL, C.HIDDEN](
                    attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                    layer.attn.qkv_proj.bound(lb),
                    layer.attn.qkv_colsum.bound(lb),
                    layer.attn.qkv_proj_sc.bound(lb),
                    qkv_lease.view[BF16, Shape[C.MAX_SEQ_LEN, QKV_LOCAL]](sb, seq_len),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                    pool)
            sample.add(self.profile.phase("attn_proj"), timed_tp_parallel[Self.tp,do_qkv_gemv](topos, self.main_pools))

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
                var sb = topos[r].arena.scratch_base()
                var q_view = qkv_lease.view[BF16, Shape[1, Q_LOCAL]](sb, 1)
                var k_view = qkv_lease.view[BF16, Shape[1, KV_LOCAL]](sb, 1, element_offset=Q_LOCAL)
                global_q_ss += rms_reduce_bf16[Q_LOCAL](q_view.as_ptr[DType.bfloat16]())
                global_k_ss += rms_reduce_bf16[KV_LOCAL](k_view.as_ptr[DType.bfloat16]())
            sample.add(self.profile.phase("norm_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_norm_prep0))
            var inv_rms_q = inv_rms_from_sum_sq(global_q_ss, C.Q_DIM, EPS)
            var inv_rms_k = inv_rms_from_sum_sq(global_k_ss, C.KV_DIM, EPS)

            # Phase 4: K/V cache write (NKV_LOCAL parallel jobs per rank)
            @parameter
            def do_kv_write[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                var cos_row = topo.rope.cos.bound_row(topo.arena.base, pos)
                var sin_row = topo.rope.sin.bound_row(topo.arena.base, pos)
                return kv_write_dispatch[
                    C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
                    C.MAX_SEQ_LEN, KV_PER_RANK](
                    qkv_lease.view[BF16, Shape[1, Q_LOCAL]](sb, 1).any(),
                    qkv_lease.view[BF16, Shape[1, KV_LOCAL]](sb, 1, element_offset=Q_LOCAL).any(),
                    qkv_lease.view[BF16, Shape[1, KV_LOCAL]](sb, 1, element_offset=Q_LOCAL + KV_LOCAL).any(),
                    layer.attn.q_norm.bound(lb),
                    layer.attn.k_norm.bound(lb),
                    cos_row, sin_row,
                    inv_rms_q, inv_rms_k,
                    topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride, pos,
                    pool)
            sample.add(self.profile.phase("kv_write"), timed_tp_parallel[Self.tp,do_kv_write](topos, self.main_pools))

            # Phase 5: per-KV-group Q prep + chunked scoring + merge/quantize.
            # LIFO order: persistent leases (live through o_proj) at the bottom,
            # transient per-iteration leases on top so they release in reverse.
            var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], Q_LOCAL]()
            var attn_head_sc_lease = self.scratch.borrow[Float32, HEADS_PER_RANK]()
            var q_i8_lease = self.scratch.borrow[
                Scalar[DType.int8], KV_PER_RANK * HPG * C.HEAD_DIM]()
            var qi_biases_lease = self.scratch.borrow[
                Float32, KV_PER_RANK * HPG]()
            var q_scales_lease = self.scratch.borrow[
                Float32, KV_PER_RANK * HPG]()
            var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()

            var context_len = pos + 1

            var t_qprep0 = Int(perf_counter_ns())
            for kv in range(KV_PER_RANK):
                for r in range(Self.tp):
                    var topo_r = topos[r]
                    var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                    var layer = topo_r.layers.proto
                    var sb = topo_r.arena.scratch_base()
                    var q_view = qkv_lease.view[BF16, Shape[1, HPG * C.HEAD_DIM]](
                        sb, 1, element_offset=kv * HPG * C.HEAD_DIM)
                    var q_norm_off_elems = kv * HPG * C.HEAD_DIM
                    var q_norm_ptr = layer.attn.q_norm.bound(lb).as_ptr[DType.bfloat16]() + q_norm_off_elems
                    var qi_out_view = q_i8_lease.view[I8, Shape[1, HPG * C.HEAD_DIM]](
                        sb, 1, element_offset=kv * HPG * C.HEAD_DIM)
                    var qi_biases_view = qi_biases_lease.view[F32, Shape[1, HPG]](
                        sb, 1, element_offset=kv * HPG)
                    var q_scales_view = q_scales_lease.view[F32, Shape[1, HPG]](
                        sb, 1, element_offset=kv * HPG)
                    var qp_args = AttnGroupArgs()
                    qp_args.q_bf16_base = q_view.as_ptr[DType.bfloat16]()
                    qp_args.q_norm_ptr = q_norm_ptr
                    qp_args.cos_ptr = topo_r.rope.cos.bound_row(topo_r.arena.base, pos).as_ptr[DType.float32]()
                    qp_args.sin_ptr = topo_r.rope.sin.bound_row(topo_r.arena.base, pos).as_ptr[DType.float32]()
                    qp_args.inv_rms_q = inv_rms_q
                    qp_args.qi_out = qi_out_view.as_ptr[DType.int8]()
                    qp_args.head_scale_ptr = qi_biases_view.as_ptr[DType.float32]()
                    qp_args.context_len = Int(q_scales_view.as_ptr[DType.float32]())
                    q_prep_kernel[C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE, HPG](qp_args)
            sample.add(self.profile.phase("q_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_qprep0))

            @parameter
            def do_chunk_score_all[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var sb = topo.arena.scratch_base()
                return chunked_score_dispatch_multi[
                    C.HEAD_DIM, HPG, C.MAX_SEQ_LEN, KV_PER_RANK,
                    C.MAX_ATTN_CHUNKS](
                    sb + q_i8_lease.offset,
                    sb + qi_biases_lease.offset,
                    sb + q_scales_lease.offset,
                    topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
                    0, KV_PER_RANK,
                    context_len, Int(pool.get_capacity()),
                    sb + partial_lease.offset,
                    pool)
            sample.add(self.profile.phase("attention"), timed_tp_parallel[Self.tp,do_chunk_score_all](topos, self.main_pools))

            var t_merge_q0 = Int(perf_counter_ns())
            comptime CHUNK_F32_STRIDE = HPG * (2 + C.HEAD_DIM)
            for kv in range(KV_PER_RANK):
                for r in range(Self.tp):
                    var nc = attn_chunk_count(
                        context_len, Int(self.main_pools[r].get_capacity()),
                        C.MAX_ATTN_CHUNKS)
                    if nc > 0:
                        var sb = topos[r].arena.scratch_base()
                        var partial = partial_lease.view[F32, Shape[1, CHUNK_F32_STRIDE]](
                            sb, 1, element_offset=kv * nc * CHUNK_F32_STRIDE
                        ).as_ptr[DType.float32]()
                        var qi_out = attn_qi_lease.view[I8, Shape[1, HPG * C.HEAD_DIM]](
                            sb, 1, element_offset=kv * HPG * C.HEAD_DIM
                        ).as_ptr[DType.int8]()
                        var head_sc = attn_head_sc_lease.view[F32, Shape[1, HPG]](
                            sb, 1, element_offset=kv * HPG
                        ).as_ptr[DType.float32]()
                        merge_and_quantize_kernel[C.HEAD_DIM, HPG, C.MAX_ATTN_CHUNKS](
                            partial, nc, qi_out, head_sc)
            sample.add(self.profile.phase("merge_quant"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_merge_q0))

            partial_lease^.release()
            q_scales_lease^.release()
            qi_biases_lease^.release()
            q_i8_lease^.release()

            # Phase 6: O projection
            @parameter
            def do_o_proj[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                return int8_gemv_blocked[C.HIDDEN, Q_LOCAL, C.HEAD_DIM](
                    attn_qi_lease.view[I8, Shape[C.MAX_SEQ_LEN, Q_LOCAL]](sb, seq_len),
                    layer.attn.o_proj.bound(lb),
                    attn_head_sc_lease.view[F32, Shape[C.MAX_SEQ_LEN, Q_LOCAL // C.HEAD_DIM]](sb, seq_len),
                    layer.attn.o_proj_sc.bound(lb),
                    layer.attn.o_colsum.bound(lb),
                    topo.activations.x_residual.bound_dyn(topo.arena.base, seq_len),
                    pool)
            sample.add(self.profile.phase("o_proj"), timed_tp_parallel[Self.tp,do_o_proj](topos, self.main_pools))

            attn_head_sc_lease^.release()
            attn_qi_lease^.release()
            qkv_lease^.release()

            # Allreduce O-proj + fused residual add: x_main += sum(x_residual)
            var t_attn_reduce0 = Int(perf_counter_ns())
            small_allreduce[BF16, XShape, Self.tp, residual_add=True](
                self.x_residual_ptrs(seq_len), seq_len,
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
                var topo_r = topos[r]
                var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                var layer = topo_r.layers.proto
                var sb = topo_r.arena.scratch_base()
                var args = RmsNormDualOutputArgs(
                    topo_r.activations.x_main.bound_dyn(topo_r.arena.base, seq_len).as_ptr[DType.bfloat16](),
                    layer.body.post_attn_norm_sqrt.bound(lb).as_ptr[DType.bfloat16](),
                    layer.body.post_attn_norm.bound(lb).as_ptr[DType.bfloat16](),
                    moe_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.int8](),
                    moe_work_lease.view[F32, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.float32](),
                    moe_scale_lease.view[F32, Shape[1, 1]](sb, 1).as_ptr[DType.float32](),
                    normed_bf16_lease.view[BF16, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.bfloat16](),
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
            def do_router_fused[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                return router_fused_dispatch[C.NUM_EXPERTS, C.HIDDEN, C.TOP_K](
                    normed_bf16_lease.view[BF16, Shape[1, C.HIDDEN]](sb, 1),
                    layer.body.router_proj.bound(lb),
                    layer.body.router_bias.bound(lb),
                    candidates_lease.as_ptr[RouterCandidate](sb),
                    pool)
            sample.add(self.profile.phase("router_proj"), timed_tp_parallel[Self.tp,do_router_fused](topos, self.main_pools))

            var t_merge0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var sb_r = topos[r].arena.scratch_base()
                var num_workers = min(C.NUM_EXPERTS // C.TOP_K, Int(self.main_pools[r].get_capacity()))
                var merge_args = RouterMergeArgs[C.TOP_K](
                    candidates_lease.as_ptr[RouterCandidate](sb_r),
                    routing_lease.as_ptr[TopKResult[C.TOP_K]](sb_r),
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
            def do_expert_phase1[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                var routing = routing_lease.as_ptr[TopKResult[C.TOP_K]](sb)[]
                comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
                var expert_base = rank * experts_per_rank
                var lc = 0
                for s in range(C.TOP_K):
                    var eid = routing.indices[s]
                    if eid >= expert_base and eid < expert_base + experts_per_rank:
                        lc += 1
                local_count_lease.as_ptr[Int32](sb)[] = Int32(lc)
                return minimax_moe_phase1[
                    C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    moe_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1),
                    moe_scale_lease.view[F32, Shape[1, 1]](sb, 1),
                    routing,
                    layer.body.experts_w1.bound(lb),
                    C.MOE_INTERMEDIATE * C.HIDDEN,
                    layer.body.experts_w1_sc.bound(lb),
                    C.MOE_INTERMEDIATE * 4,
                    layer.body.experts_w1_colsum.bound(lb),
                    C.MOE_INTERMEDIATE * 4,
                    layer.body.experts_w3.bound(lb),
                    C.MOE_INTERMEDIATE * C.HIDDEN,
                    layer.body.experts_w3_sc.bound(lb),
                    C.MOE_INTERMEDIATE * 4,
                    layer.body.experts_w3_colsum.bound(lb),
                    C.MOE_INTERMEDIATE * 4,
                    expert_qi_lease.view[I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                    expert_blk_scale_lease.view[F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](sb, C.TOP_K),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase1"), timed_tp_parallel[Self.tp,do_expert_phase1](topos, self.main_pools))

            # Phase 10: expert down (w2)
            @parameter
            def do_expert_phase2[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                var routing = routing_lease.as_ptr[TopKResult[C.TOP_K]](sb)[]
                return minimax_moe_phase2[
                    C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK_MOE_DOWN,
                    C.TOP_K, C.NUM_EXPERTS, Self.tp](
                    expert_qi_lease.view[I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                    expert_blk_scale_lease.view[F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](sb, C.TOP_K),
                    routing,
                    layer.body.experts_w2.bound(lb),
                    C.HIDDEN * C.MOE_INTERMEDIATE,
                    layer.body.experts_w2_sc.bound(lb),
                    C.HIDDEN * 4,
                    layer.body.experts_w2_colsum.bound(lb),
                    C.HIDDEN * MOE_DOWN_NUM_BLK * 4,
                    expert_out_lease.view[BF16, Shape[C.TOP_K, C.HIDDEN]](sb, C.TOP_K),
                    rank, pool)
            sample.add(self.profile.phase("expert_phase2"), timed_tp_parallel[Self.tp,do_expert_phase2](topos, self.main_pools))

            # Phase 11: expert reduce + fused allreduce + residual add
            var t_expert_sum0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var topo_r = topos[r]
                var sb = topo_r.arena.scratch_base()
                var lc = Int(local_count_lease.as_ptr[Int32](sb)[])
                accumulate_expert_outputs[C.HIDDEN, C.TOP_K](
                    expert_out_lease.view[BF16, Shape[C.TOP_K, C.HIDDEN]](sb, C.TOP_K).as_ptr[DType.bfloat16](),
                    lc,
                    topo_r.activations.x_residual.bound_dyn(topo_r.arena.base, seq_len).as_ptr[DType.bfloat16]())
            sample.add(self.profile.phase("expert_sum"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_expert_sum0))

            var t_ffn_reduce0 = Int(perf_counter_ns())
            small_allreduce[BF16, XShape, Self.tp, residual_add=True](
                self.x_residual_ptrs(seq_len), seq_len,
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
            host.activations.x_main.bound_dyn(host.arena.base, seq_len),
            host.host.lm_output_head_sqrt_gamma.bound(host.arena.base),
            lm_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            lm_work_lease.view[F32, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            lm_act_blk_scale_lease.view[F32, Shape[1, VOCAB_NUM_BLOCKS]](host.arena.scratch_base(), 1),
            EPS, self.main_pools[0])
        var t_final1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("final_norm"), finish_single_pool_fence(t_final0, t_final1, final_fence^))

        # lm_work is dead after final_norm; release now so logit_lease (next
        # borrow) lands on top of a LIFO-clean stack.
        lm_work_lease^.release()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = logit_lease.view[BF16, Shape[1, C.VOCAB_SIZE]](host.arena.scratch_base(), 1)
        var t_lm0 = Int(perf_counter_ns())
        var lm_fence = lm_head_gemv[
            C.VOCAB_SIZE, C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK](
            lm_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](host.arena.scratch_base(), 1),
            host.host.lm_output_head.bound(host.arena.base),
            lm_act_blk_scale_lease.view[F32, Shape[1, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK]](host.arena.scratch_base(), 1),
            host.host.lm_output_head_sc.bound(host.arena.base),
            host.host.lm_output_head_colsum.bound(host.arena.base),
            logit_view,
            self.main_pools[0])
        var t_lm1 = Int(perf_counter_ns())
        sample.add(self.profile.phase("lm_head"), finish_single_pool_fence(t_lm0, t_lm1, lm_fence^))

        # Argmax (no softcap in MiniMax)
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
