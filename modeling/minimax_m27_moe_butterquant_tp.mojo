from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaTopology
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import (
    BF16, F32, I8,
    Shape, DynamicView,
    WeightDesc, DEFAULT_ALIGNMENT,
    TaskVisitor,
    PerRow, PerRowAbsorbed, PerBlockAbsorbed, TwoSidedAbsorbed,
    HOST_RANK,
)
from quant.source_format import Bf16Converter, Fp8E4M3Block128Converter
from modeling.modeling_common import (
    TensorRef, Repeated, SectionBuilder, align_up,
    LayerBuilder, ArenaLayout,
)
from kernels.kernel_ops import PoolFence, embed_lookup
from kernels.reductions import small_allreduce, parallel_allreduce, ring_broadcast
from kernels.kv_rotors import init_rope_tables
from simd_math import set_subnormal_zeroing
from modeling.linear_borrow_pool import (
    ScratchLease, ScratchPool, scratch_block_bytes,
)
from modeling.loader import discover_shards, load_weights_from_descs

from experimental3.profiler import (
    PhaseTiming, finish_single_pool_fence, timed_tp_parallel,
    ForwardSample, ForwardLogger,
)
from experimental3.small_phase_dispatch import run_tp_single_job_phase
from experimental3.init_weights import (
    PackColsumTask, make_pack_colsum_task, dispatch_pack_colsum_tasks, colsum_at,
)
from experimental3.gamma import compute_sqrt_gamma
from experimental3.common_math import rms_reduce_bf16, inv_rms_from_sum_sq
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.dispatch_kernels import (
    rmsnorm_gamma_fwht_per_block_quantize,
    int8_gemv, int8_gemv_blocked,
    lm_head_gemv,
)
from experimental3.kernels.dispatch_args import RmsNormFwhtQuantArgs
from experimental3.kernels.rmsnorm import (
    rmsnorm_fwht_quant_worker, accumulate_expert_outputs,
)

from minimax.kernels.router import (
    TopKResult, router_merge_multi_and_renorm, build_sparse_route_schedule,
)
from minimax.kernels.dispatch_args import (
    RouterCandidate, RmsNormDualOutputArgs,
    MergeQuantArgs, SparseRoute,
)
from minimax.kernels.dispatch_kernels import (
    router_fused_dispatch,
    kv_write_dispatch,
    q_prep_batch_dispatch,
    chunked_score_dispatch_multi,
    prefill_attn_dispatch,
    attn_chunk_count,
    minimax_moe_phase1,
    minimax_moe_phase2,
    minimax_sparse_moe_phase1,
    minimax_sparse_moe_phase2,
)
from minimax.kernels.rmsnorm import rmsnorm_dual_output_worker
from kernels.worker_init import worker_init_dispatch
from minimax.kernels.amx_attention import amx_prefill_config_kernel, amx_config_kernel, AmxConfigArgs
from minimax.kernels.attention import merge_quant_worker


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
    # MiniMax uses standard rotate_half: pair stride = ROPE_DIM // 2 = 32.
    # Gemma4 proportional RoPE uses HEAD_DIM // 2 = 256. The pair_stride
    # parameter on write_k_head_normed / rope_apply_partial controls this.
    # inv_freq[i] = 1 / (ROPE_THETA ^ (2i / ROPE_DIM)) for i in 0..31.
    comptime ROPE_PAIR_STRIDE = Self.ROPE_DIM // 2

    comptime MOE_INTERMEDIATE = 1536
    comptime NUM_EXPERTS = 256
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 200064
    # tokenizer.json defines IDs 0..200053. The checkpoint exposes ten
    # additional LM-head rows, but those IDs have no token text and must not be
    # emitted during generation.
    comptime GENERATION_VOCAB_SIZE = 200054
    comptime EOS_TOKEN_ID = 200020
    comptime RMS_NORM_EPS = 1e-6

    comptime MAX_SEQ_LEN = 1024 * 128

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
comptime MOE_DOWN_NUM_BLK = C.MOE_INTERMEDIATE // FWHT_BLK_MOE_DOWN
comptime PREFILL_CHUNK_SIZE = 1024


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

    # BF16 centered router weights plus a replicated gauge vector. Runtime
    # reconstructs logits as x @ centered_row + x @ gauge before sigmoid+bias.
    var router_proj: TensorRef[BF16, Shape[C.NUM_EXPERTS, C.HIDDEN, shard_n=True, tp=Self.tp]]
    var router_gauge: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var router_bias: TensorRef[F32, Shape[C.NUM_EXPERTS, 1, shard_n=True, tp=Self.tp]]

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
struct HostSlots[tp: Int](Copyable, ImplicitlyCopyable):
    var final_norm: TensorRef[BF16, Shape[C.HIDDEN, 1]]

    # Source checkpoint semantics:
    # - embed_tokens.weight is an untied BF16 lookup table.
    # - lm_head.weight is an untied BF16 source tensor.
    # Runtime semantics:
    # - lm_output_head is the row-sharded ButterQuant output projection
    #   derived from source lm_head.weight, keeping the fast per-block GEMV
    #   path while distributing the vocab scan across TP ranks.
    var embed:      TensorRef[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]
    var lm_output_head:         TensorRef[I8,  Shape[C.VOCAB_SIZE, C.HIDDEN, shard_n=True, tp=Self.tp]]
    var lm_output_head_sc:      TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK, shard_n=True, tp=Self.tp]]
    var lm_output_head_colsum:  TensorRef[F32, Shape[C.VOCAB_SIZE, C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK, shard_n=True, tp=Self.tp]]
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

    var host: HostSlots[Self.tp]

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
    comptime qkv_n_loc = S.QKV_LOCAL
    comptime o_num_blk = S.NUM_HEADS_LOCAL
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

    var router_proj = b.bfs[Shape[NE, H, shard_n=True, tp=tp]](
        e, "block_sparse_moe.gate.weight")
    var router_gauge = b.bfs[Shape[H, 1]](
        e, "block_sparse_moe.gate.weight_gauge")
    var router_bias = b.fs[Shape[NE, 1, shard_n=True, tp=tp]](e, "block_sparse_moe.e_score_correction_bias")

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
        router_proj=router_proj, router_gauge=router_gauge,
        router_bias=router_bias,
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
    comptime i32  = size_of[Int32]()
    comptime P = PREFILL_CHUNK_SIZE

    comptime HPG = C.HPG
    comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
    comptime PARTIAL_F32S = (
        KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM))
    comptime act_scale_bytes = scratch_block_bytes[P * f32]()
    comptime qkv_bytes = scratch_block_bytes[P * S.QKV_LOCAL * bf16]()
    comptime attn_phase1_peak = (
        act_scale_bytes
        + qkv_bytes
        + scratch_block_bytes[P * C.HIDDEN * i8]()
        + scratch_block_bytes[C.HIDDEN * f32]()
    )
    comptime attn_phase2_peak = (
        act_scale_bytes
        + qkv_bytes
        + scratch_block_bytes[P * S.Q_LOCAL * i8]()
        + scratch_block_bytes[P * S.NUM_HEADS_LOCAL * f32]()
        + scratch_block_bytes[P * KV_PER_RANK * HPG * C.HEAD_DIM * i8]()
        + scratch_block_bytes[P * KV_PER_RANK * HPG * f32]()
        + scratch_block_bytes[P * KV_PER_RANK * HPG * f32]()
        + scratch_block_bytes[PARTIAL_F32S * f32]()
    )
    comptime attn_peak = (
        attn_phase1_peak if attn_phase1_peak > attn_phase2_peak else attn_phase2_peak
    )

    comptime router_candidate_bytes = size_of[RouterCandidate]()
    comptime topk_result_bytes = size_of[TopKResult[C.TOP_K]]()
    comptime sparse_route_bytes = size_of[SparseRoute]()
    comptime moe_persistent = (
        act_scale_bytes
        + scratch_block_bytes[P * C.HIDDEN * i8]()
        + scratch_block_bytes[P * f32]()
        + scratch_block_bytes[P * C.HIDDEN * bf16]()
    )
    comptime moe_dual_norm_peak = (
        moe_persistent + scratch_block_bytes[C.HIDDEN * f32]()
    )
    comptime decode_router_peak = (
        moe_persistent
        + scratch_block_bytes[topk_result_bytes]()
        + scratch_block_bytes[C.TOP_K * router_candidate_bytes]()
    )
    comptime decode_expert_peak = (
        moe_persistent
        + scratch_block_bytes[topk_result_bytes]()
        + scratch_block_bytes[C.TOP_K * C.MOE_INTERMEDIATE * i8]()
        + scratch_block_bytes[C.TOP_K * MOE_DOWN_NUM_BLK * f32]()
        + scratch_block_bytes[C.TOP_K * C.HIDDEN * bf16]()
        + scratch_block_bytes[i32]()
    )
    comptime prefill_router_peak = (
        moe_persistent
        + scratch_block_bytes[P * topk_result_bytes]()
        + scratch_block_bytes[P * C.TOP_K * router_candidate_bytes]()
    )
    comptime prefill_sparse_expert_peak = (
        moe_persistent
        + scratch_block_bytes[P * topk_result_bytes]()
        + scratch_block_bytes[S.EXPERTS_LOCAL * i32]()
        + scratch_block_bytes[(S.EXPERTS_LOCAL + 1) * i32]()
        + scratch_block_bytes[S.EXPERTS_LOCAL * i32]()
        + scratch_block_bytes[P * C.TOP_K * sparse_route_bytes]()
        + scratch_block_bytes[P * C.TOP_K * C.MOE_INTERMEDIATE * i8]()
        + scratch_block_bytes[P * C.TOP_K * MOE_DOWN_NUM_BLK * f32]()
        + scratch_block_bytes[P * C.HIDDEN * f32]()
    )
    comptime moe_decode_peak = (
        decode_router_peak
        if decode_router_peak > decode_expert_peak else decode_expert_peak
    )
    comptime moe_prefill_peak = (
        prefill_router_peak
        if prefill_router_peak > prefill_sparse_expert_peak
        else prefill_sparse_expert_peak
    )
    comptime moe_decode_or_prefill_peak = (
        moe_decode_peak
        if moe_decode_peak > moe_prefill_peak else moe_prefill_peak
    )
    comptime moe_peak = (
        moe_dual_norm_peak
        if moe_dual_norm_peak > moe_decode_or_prefill_peak
        else moe_decode_or_prefill_peak
    )

    comptime lm_output_head_peak = (
        scratch_block_bytes[C.HIDDEN * i8]()
        + scratch_block_bytes[(C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK) * f32]()
        + scratch_block_bytes[C.HIDDEN * f32]()
        + scratch_block_bytes[(C.VOCAB_SIZE // tp) * bf16]()
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

    comptime vocab_num_blocks = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
    var global_builder = LayerBuilder(tp, "", 0)
    global_builder.cursor = distributed
    var lm_output_head_off = global_builder.qs[
        Shape[C.VOCAB_SIZE, C.HIDDEN, shard_n=True, tp=tp]](
        descs, "lm_head.weight")
    var lm_output_head_sc_off = global_builder.fs[
        Shape[C.VOCAB_SIZE, vocab_num_blocks, shard_n=True, tp=tp]](
        descs, "lm_head.weight_scale")
    var lm_output_head_colsum = global_builder.colsum_slot[
        F32, Shape[C.VOCAB_SIZE, vocab_num_blocks, shard_n=True, tp=tp]]()
    distributed = global_builder.cursor

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

    var host_off = align_up(state.bytes())
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off

    var final_norm_off = hb.bfs[Shape[C.HIDDEN, 1]](descs, "model.norm.weight", target_rank=HOST_RANK)

    var embed_off = hb.bfs[Shape[C.VOCAB_SIZE, C.HIDDEN]](descs, "model.embed_tokens.weight", target_rank=HOST_RANK)

    var lm_output_head_sqrt_gamma = hb.colsum_slot[BF16, Shape[C.HIDDEN, 1]]()
    var host_bytes = hb.cursor

    var host = HostSlots[tp](
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
# Model struct
# =============================================================================


struct MiniMaxM27ButterQuant[tp: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[Self.Pool]
    var scratch: ScratchPool
    var topos: InlineArray[MiniMaxM27Topology[Self.tp], Self.tp]
    var profile: ForwardLogger
    var amx_prefill_mode: Bool

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
        self.amx_prefill_mode = False


    def x_main_ptrs(self) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            var topo = self.topos[r]
            ptrs[r] = topo.activations.x_main.addr(topo.arena.base)
        return ptrs^

    def x_residual_ptrs(self) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            var topo = self.topos[r]
            ptrs[r] = topo.activations.x_residual.addr(topo.arena.base)
        return ptrs^

    def residual_add_allreduce(mut self, seq_len: Int):
        """x_main += allreduce(x_residual), replicated across ranks.

        Decode is latency-shaped and small enough for the main-thread reducer.
        Prefill is bandwidth-shaped, so reduce and gather with worker pools.
        """
        var residual_ptrs = self.x_residual_ptrs()
        var main_ptrs = self.x_main_ptrs()
        if seq_len == 1:
            small_allreduce[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN], Self.tp,
                residual_add=True](
                residual_ptrs, seq_len, main_ptrs)
        else:
            parallel_allreduce[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN], Self.tp,
                residual_add=True](
                residual_ptrs, seq_len, self.main_pools, main_ptrs)

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topos[0].arena.scratch_base())

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
            ok &= visitor.router_gauge_bf16(p + "block_sparse_moe.gate.weight")
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
        comptime experts_local = MiniMaxShapes[Self.tp].EXPERTS_LOCAL
        comptime q_local = C.Q_DIM // Self.tp
        comptime kv_local = C.KV_DIM // Self.tp
        comptime qkv_local = q_local + 2 * kv_local

        for rank in range(Self.tp):
            var topo = self.topos[rank]
            var base = topo.arena.base
            var numa_node = self.arenas[rank].node

            init_rope_tables(
                topo.rope.cos.bound(base),
                topo.rope.sin.bound(base),
                theta=Float64(C.ROPE_THETA))

            var tasks = List[PackColsumTask]()
            for i in range(C.NUM_LAYERS):
                var lb = topo.layers.base(base, i)
                var layer = topo.layers.proto

                var qkv_slot = TensorRef[I8, Shape[qkv_local, C.HIDDEN]](
                    offset=layer.attn.qkv_proj.offset)
                tasks.append(make_pack_colsum_task(
                    lb, qkv_slot, layer.attn.qkv_colsum))

                var o_slot = TensorRef[I8, Shape[C.HIDDEN, q_local]](
                    offset=layer.attn.o_proj.offset)
                tasks.append(make_pack_colsum_task(
                    lb, o_slot, layer.attn.o_colsum,
                    block_cols=C.HEAD_DIM, colsum_row_major=False))

                for e in range(experts_local):
                    var expert_w1 = TensorRef[I8, Shape[C.MOE_INTERMEDIATE, C.HIDDEN]](
                        offset=layer.body.experts_w1.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN)
                    var expert_w1_colsum = TensorRef[F32, Shape[C.MOE_INTERMEDIATE, 1]](
                        offset=layer.body.experts_w1_colsum.offset + e * C.MOE_INTERMEDIATE * 4)
                    tasks.append(make_pack_colsum_task(lb, expert_w1, expert_w1_colsum))

                    var expert_w3 = TensorRef[I8, Shape[C.MOE_INTERMEDIATE, C.HIDDEN]](
                        offset=layer.body.experts_w3.offset + e * C.MOE_INTERMEDIATE * C.HIDDEN)
                    var expert_w3_colsum = TensorRef[F32, Shape[C.MOE_INTERMEDIATE, 1]](
                        offset=layer.body.experts_w3_colsum.offset + e * C.MOE_INTERMEDIATE * 4)
                    tasks.append(make_pack_colsum_task(lb, expert_w3, expert_w3_colsum))

                    var expert_w2 = TensorRef[I8, Shape[C.HIDDEN, C.MOE_INTERMEDIATE]](
                        offset=layer.body.experts_w2.offset + e * C.HIDDEN * C.MOE_INTERMEDIATE)
                    var expert_w2_colsum = TensorRef[F32, Shape[C.HIDDEN * MOE_DOWN_NUM_BLK, 1]](
                        offset=layer.body.experts_w2_colsum.offset + e * C.HIDDEN * MOE_DOWN_NUM_BLK * 4)
                    tasks.append(make_pack_colsum_task(
                        lb, expert_w2, expert_w2_colsum,
                        block_cols=FWHT_BLK_MOE_DOWN, colsum_row_major=False))

            dispatch_pack_colsum_tasks(self.main_pools[rank], numa_node, tasks)

            for i in range(C.NUM_LAYERS):
                var lb = topo.layers.base(base, i)
                var layer = topo.layers.proto
                compute_sqrt_gamma[C.HIDDEN](
                    layer.body.input_norm.bound(lb).as_ptr(),
                    layer.body.input_norm_sqrt.bound(lb).as_ptr())
                compute_sqrt_gamma[C.HIDDEN](
                    layer.body.post_attn_norm.bound(lb).as_ptr(),
                    layer.body.post_attn_norm_sqrt.bound(lb).as_ptr())

            colsum_at(base,
                topo.host.lm_output_head, topo.host.lm_output_head_colsum,
                block_cols=LM_OUTPUT_HEAD_FWHT_BLK, colsum_row_major=True)

            if rank == HOST_RANK:
                compute_sqrt_gamma[C.HIDDEN](
                    topo.host.final_norm.bound(base).as_ptr(),
                    topo.host.lm_output_head_sqrt_gamma.bound(base).as_ptr())

        for rank in range(Self.tp):
            worker_init_dispatch[C.HPG](self.main_pools[rank]).join()
        print("state initialized")

    def moe_phase_decode[
        moe_i8_origin: MutOrigin,
        moe_scale_origin: MutOrigin,
        normed_origin: MutOrigin,
    ](
        mut self,
        mut sample: ForwardSample,
        layer_idx: Int,
        seq_len: Int,
        ref [moe_i8_origin] moe_i8_lease: ScratchLease,
        ref [moe_scale_origin] moe_scale_lease: ScratchLease,
        ref [normed_origin] normed_bf16_lease: ScratchLease,
    ):
        """Decode-shaped MoE: route one token, run selected expert GEMVs, combine."""
        var topos = self.topos
        var routing_lease = self.scratch.borrow[TopKResult[C.TOP_K], 1]()
        var candidates_lease = self.scratch.borrow[
            RouterCandidate, C.TOP_K]()

        @parameter
        def do_router_fused[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var lb = topo.layers.base(topo.arena.base, layer_idx)
            var layer = topo.layers.proto
            var sb = topo.arena.scratch_base()
            comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
            var expert_base = rank * experts_per_rank
            return router_fused_dispatch[experts_per_rank, C.HIDDEN, C.TOP_K](
                normed_bf16_lease.view[
                    BF16, Shape[1, C.HIDDEN]](sb, 1),
                layer.body.router_proj.bound(lb),
                layer.body.router_gauge.bound(lb),
                layer.body.router_bias.bound(lb),
                candidates_lease.as_ptr[RouterCandidate](sb),
                pool,
                expert_base)
        sample.add(
            self.profile.phase("moe_router"),
            timed_tp_parallel[Self.tp,do_router_fused](
                topos, self.main_pools))

        var t_merge0 = Int(perf_counter_ns())
        var candidate_ptrs = InlineArray[
            UnsafePointer[RouterCandidate, MutAnyOrigin], Self.tp](
            uninitialized=True)
        var candidate_counts = InlineArray[Int, Self.tp](uninitialized=True)
        for r in range(Self.tp):
            var sb_r = topos[r].arena.scratch_base()
            candidate_ptrs[r] = candidates_lease.as_ptr[RouterCandidate](sb_r)
            candidate_counts[r] = C.TOP_K
        var host_sb = topos[HOST_RANK].arena.scratch_base()
        router_merge_multi_and_renorm[C.TOP_K, Self.tp](
            candidate_ptrs, candidate_counts,
            routing_lease.as_ptr[TopKResult[C.TOP_K]](host_sb))
        var routing = routing_lease.as_ptr[
            TopKResult[C.TOP_K]](host_sb)[]
        self.profile.record_moe_route[C.TOP_K](
            layer_idx, C.NUM_LAYERS, C.NUM_EXPERTS, Self.tp,
            routing.indices, routing.weights)
        for r in range(Self.tp):
            var sb_r = topos[r].arena.scratch_base()
            routing_lease.as_ptr[TopKResult[C.TOP_K]](sb_r)[] = routing
        sample.add(
            self.profile.phase("moe_route_merge"),
            PhaseTiming.opaque(Int(perf_counter_ns()) - t_merge0))

        candidates_lease^.release()

        var expert_qi_lease = self.scratch.borrow[
            Scalar[DType.int8], C.TOP_K * C.MOE_INTERMEDIATE]()
        var expert_blk_scale_lease = self.scratch.borrow[
            Float32, C.TOP_K * MOE_DOWN_NUM_BLK]()
        var expert_out_lease = self.scratch.borrow[
            Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
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
                expert_qi_lease.view[
                    I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                expert_blk_scale_lease.view[
                    F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](sb, C.TOP_K),
                rank, pool)
        sample.add(
            self.profile.phase("moe_phase1"),
            timed_tp_parallel[Self.tp,do_expert_phase1](
                topos, self.main_pools))

        @parameter
        def do_expert_phase2[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var lb = topo.layers.base(topo.arena.base, layer_idx)
            var layer = topo.layers.proto
            var sb = topo.arena.scratch_base()
            var routing = routing_lease.as_ptr[TopKResult[C.TOP_K]](sb)[]
            return minimax_moe_phase2[
                C.HIDDEN, C.MOE_INTERMEDIATE, FWHT_BLK_MOE_DOWN,
                C.TOP_K, C.NUM_EXPERTS, Self.tp](
                expert_qi_lease.view[
                    I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
                expert_blk_scale_lease.view[
                    F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](sb, C.TOP_K),
                routing,
                layer.body.experts_w2.bound(lb),
                C.HIDDEN * C.MOE_INTERMEDIATE,
                layer.body.experts_w2_sc.bound(lb),
                C.HIDDEN * 4,
                layer.body.experts_w2_colsum.bound(lb),
                C.HIDDEN * MOE_DOWN_NUM_BLK * 4,
                expert_out_lease.view[
                    BF16, Shape[C.TOP_K, C.HIDDEN]](sb, C.TOP_K),
                rank, pool)
        sample.add(
            self.profile.phase("moe_phase2"),
            timed_tp_parallel[Self.tp,do_expert_phase2](
                topos, self.main_pools))

        var t_accum0 = Int(perf_counter_ns())
        for r in range(Self.tp):
            var topo_r = topos[r]
            var sb = topo_r.arena.scratch_base()
            var lc = Int(local_count_lease.as_ptr[Int32](sb)[])
            accumulate_expert_outputs[C.HIDDEN, C.TOP_K](
                expert_out_lease.view[
                    BF16, Shape[C.TOP_K, C.HIDDEN]](
                    sb, C.TOP_K).as_ptr[DType.bfloat16](),
                lc,
                topo_r.activations.x_residual.bound_dyn(
                    topo_r.arena.base, seq_len
                ).as_ptr[DType.bfloat16]())
        sample.add(
            self.profile.phase("moe_output_accum"),
            PhaseTiming.opaque(Int(perf_counter_ns()) - t_accum0))

        local_count_lease^.release()
        expert_out_lease^.release()
        expert_blk_scale_lease^.release()
        expert_qi_lease^.release()
        routing_lease^.release()

    def moe_phase_prefill[
        moe_i8_origin: MutOrigin,
        moe_scale_origin: MutOrigin,
        normed_origin: MutOrigin,
    ](
        mut self,
        mut sample: ForwardSample,
        layer_idx: Int,
        seq_len: Int,
        ref [moe_i8_origin] moe_i8_lease: ScratchLease,
        ref [moe_scale_origin] moe_scale_lease: ScratchLease,
        ref [normed_origin] normed_bf16_lease: ScratchLease,
    ):
        """Prefill-shaped MoE: route the chunk, bucket local routes, sparse compute."""
        var topos = self.topos
        var routing_lease = self.scratch.borrow[
            TopKResult[C.TOP_K], PREFILL_CHUNK_SIZE]()
        var candidates_lease = self.scratch.borrow[
            RouterCandidate, PREFILL_CHUNK_SIZE * C.TOP_K]()

        @parameter
        def do_router_fused[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var lb = topo.layers.base(topo.arena.base, layer_idx)
            var layer = topo.layers.proto
            var sb = topo.arena.scratch_base()
            comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
            var expert_base = rank * experts_per_rank
            return router_fused_dispatch[experts_per_rank, C.HIDDEN, C.TOP_K](
                normed_bf16_lease.view[
                    BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                layer.body.router_proj.bound(lb),
                layer.body.router_gauge.bound(lb),
                layer.body.router_bias.bound(lb),
                candidates_lease.as_ptr[RouterCandidate](sb),
                pool,
                expert_base)
        sample.add(
            self.profile.phase("moe_router"),
            timed_tp_parallel[Self.tp,do_router_fused](
                topos, self.main_pools))

        var t_merge0 = Int(perf_counter_ns())
        for token_idx in range(seq_len):
            var candidate_ptrs = InlineArray[
                UnsafePointer[RouterCandidate, MutAnyOrigin], Self.tp](
                uninitialized=True)
            var candidate_counts = InlineArray[Int, Self.tp](uninitialized=True)
            for r in range(Self.tp):
                var sb_r = topos[r].arena.scratch_base()
                var candidates = candidates_lease.as_ptr[RouterCandidate](sb_r)
                candidate_ptrs[r] = candidates + token_idx * C.TOP_K
                candidate_counts[r] = C.TOP_K
            var host_sb = topos[HOST_RANK].arena.scratch_base()
            router_merge_multi_and_renorm[C.TOP_K, Self.tp](
                candidate_ptrs, candidate_counts,
                routing_lease.as_ptr[TopKResult[C.TOP_K]](
                    host_sb, token_idx))
            var routing = routing_lease.as_ptr[
                TopKResult[C.TOP_K]](host_sb, token_idx)[]
            self.profile.record_moe_route[C.TOP_K](
                layer_idx, C.NUM_LAYERS, C.NUM_EXPERTS, Self.tp,
                routing.indices, routing.weights)
            for r in range(Self.tp):
                var sb_r = topos[r].arena.scratch_base()
                routing_lease.as_ptr[
                    TopKResult[C.TOP_K]](sb_r, token_idx)[] = routing
        sample.add(
            self.profile.phase("moe_route_merge"),
            PhaseTiming.opaque(Int(perf_counter_ns()) - t_merge0))

        candidates_lease^.release()

        comptime experts_per_rank = C.NUM_EXPERTS // Self.tp
        var route_counts_lease = self.scratch.borrow[Int32, experts_per_rank]()
        var route_offsets_lease = self.scratch.borrow[
            Int32, experts_per_rank + 1]()
        var route_cursors_lease = self.scratch.borrow[Int32, experts_per_rank]()
        var routes_lease = self.scratch.borrow[
            SparseRoute, PREFILL_CHUNK_SIZE * C.TOP_K]()
        var expert_qi_lease = self.scratch.borrow[
            Scalar[DType.int8],
            PREFILL_CHUNK_SIZE * C.TOP_K * C.MOE_INTERMEDIATE]()
        var expert_blk_scale_lease = self.scratch.borrow[
            Float32, PREFILL_CHUNK_SIZE * C.TOP_K * MOE_DOWN_NUM_BLK]()
        var moe_accum_lease = self.scratch.borrow[
            Float32, PREFILL_CHUNK_SIZE * C.HIDDEN]()

        var t_schedule0 = Int(perf_counter_ns())
        for r in range(Self.tp):
            var sb = topos[r].arena.scratch_base()
            _ = build_sparse_route_schedule[C.TOP_K, experts_per_rank](
                routing_lease.as_ptr[TopKResult[C.TOP_K]](sb),
                seq_len,
                r,
                route_counts_lease.as_ptr[Int32](sb),
                route_offsets_lease.as_ptr[Int32](sb),
                route_cursors_lease.as_ptr[Int32](sb),
                routes_lease.as_ptr[SparseRoute](sb),
            )
        sample.add(
            self.profile.phase("moe_route_schedule"),
            PhaseTiming.opaque(Int(perf_counter_ns()) - t_schedule0))

        @parameter
        def do_sparse_expert_phase1[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var lb = topo.layers.base(topo.arena.base, layer_idx)
            var layer = topo.layers.proto
            var sb = topo.arena.scratch_base()
            return minimax_sparse_moe_phase1[
                experts_per_rank, C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK](
                moe_i8_lease.view[
                    I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
                moe_scale_lease.view[
                    F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
                route_offsets_lease.as_ptr[Int32](sb),
                routes_lease.as_ptr[SparseRoute](sb),
                layer.body.experts_w1.bound(lb),
                C.MOE_INTERMEDIATE * C.HIDDEN,
                layer.body.experts_w1_sc.bound(lb),
                C.MOE_INTERMEDIATE,
                layer.body.experts_w3.bound(lb),
                C.MOE_INTERMEDIATE * C.HIDDEN,
                layer.body.experts_w3_sc.bound(lb),
                C.MOE_INTERMEDIATE,
                expert_qi_lease.view[
                    I8,
                    Shape[
                        PREFILL_CHUNK_SIZE * C.TOP_K,
                        C.MOE_INTERMEDIATE,
                    ],
                ](sb, PREFILL_CHUNK_SIZE * C.TOP_K),
                expert_blk_scale_lease.view[
                    F32,
                    Shape[
                        PREFILL_CHUNK_SIZE * C.TOP_K,
                        MOE_DOWN_NUM_BLK,
                    ],
                ](sb, PREFILL_CHUNK_SIZE * C.TOP_K),
                pool)
        sample.add(
            self.profile.phase("moe_phase1"),
            timed_tp_parallel[Self.tp,do_sparse_expert_phase1](
                topos, self.main_pools))

        @parameter
        def do_sparse_expert_phase2[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var lb = topo.layers.base(topo.arena.base, layer_idx)
            var layer = topo.layers.proto
            var sb = topo.arena.scratch_base()
            return minimax_sparse_moe_phase2[
                experts_per_rank, C.HIDDEN,
                C.MOE_INTERMEDIATE, FWHT_BLK_MOE_DOWN](
                route_offsets_lease.as_ptr[Int32](sb),
                routes_lease.as_ptr[SparseRoute](sb),
                expert_qi_lease.view[
                    I8,
                    Shape[
                        PREFILL_CHUNK_SIZE * C.TOP_K,
                        C.MOE_INTERMEDIATE,
                    ],
                ](sb, PREFILL_CHUNK_SIZE * C.TOP_K),
                expert_blk_scale_lease.view[
                    F32,
                    Shape[
                        PREFILL_CHUNK_SIZE * C.TOP_K,
                        MOE_DOWN_NUM_BLK,
                    ],
                ](sb, PREFILL_CHUNK_SIZE * C.TOP_K),
                layer.body.experts_w2.bound(lb),
                C.HIDDEN * C.MOE_INTERMEDIATE,
                layer.body.experts_w2_sc.bound(lb),
                C.HIDDEN,
                moe_accum_lease.view[
                    F32,
                    Shape[
                        PREFILL_CHUNK_SIZE,
                        C.HIDDEN,
                    ],
                ](sb, seq_len),
                topo.activations.x_residual.bound_dyn(
                    topo.arena.base, seq_len),
                pool)
        sample.add(
            self.profile.phase("moe_phase2"),
            timed_tp_parallel[Self.tp,do_sparse_expert_phase2](
                topos, self.main_pools))

        moe_accum_lease^.release()
        expert_blk_scale_lease^.release()
        expert_qi_lease^.release()
        routes_lease^.release()
        route_cursors_lease^.release()
        route_offsets_lease^.release()
        route_counts_lease^.release()
        routing_lease^.release()

    @staticmethod
    def load(
        dir_path: Path,
        numa_topo: NumaTopology,
        var main_pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
        if C.NUM_KV_HEADS % Self.tp != 0:
            print(
                "unsupported TP=", Self.tp,
                ": KV heads must be divisible by tp; got", C.NUM_KV_HEADS,
            )
            return None
        if C.VOCAB_SIZE % Self.tp != 0:
            print(
                "unsupported TP=", Self.tp,
                ": vocab size must be divisible by tp; got", C.VOCAB_SIZE,
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

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(plan.topology.arena.distributed_bytes, plan.topology.arena.state_bytes)

        var topos = InlineArray[MiniMaxM27Topology[Self.tp], Self.tp](fill=plan.topology)
        for rank in range(Self.tp):
            topos[rank] = plan.topology.bind(Int(arenas[rank].base))

        var scratch = ScratchPool(plan.topology.arena.scratch_capacity)
        var model = Self(arenas^, main_pools^, scratch^, topos)
        model.init_state()
        return model^

    def forward(
        mut self,
        tokens_ptr: Int,
        start_pos: Int,
        seq_len: Int,
        produce_next_token: Bool = True,
    ) -> Int32:
        comptime S = MiniMaxShapes[Self.tp]
        comptime EPS = Float32(C.RMS_NORM_EPS)
        comptime Q_LOCAL = S.Q_LOCAL
        comptime KV_LOCAL = S.KV_LOCAL
        comptime HPG = C.HPG
        comptime HEADS_PER_RANK = S.NUM_HEADS_LOCAL
        comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
        comptime XShape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
        comptime VOCAB_NUM_BLOCKS = C.HIDDEN // LM_OUTPUT_HEAD_FWHT_BLK
        comptime VOCAB_LOCAL = C.VOCAB_SIZE // Self.tp
        comptime PARTIAL_F32S = (
            KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + C.HEAD_DIM))
        comptime ROPE_HALF = C.ROPE_DIM // 2

        var t_forward0 = Int(perf_counter_ns())
        var sample = ForwardSample(start_pos, seq_len, produce_next_token)
        var topos = self.topos
        var host = topos[HOST_RANK]

        var want_prefill = seq_len > 1
        if want_prefill != self.amx_prefill_mode:
            self.amx_prefill_mode = want_prefill
            for rank in range(Self.tp):
                var cfg_args = InlineArray[AmxConfigArgs, 128](
                    fill=AmxConfigArgs())
                var cap = self.main_pools[rank].get_capacity()
                if want_prefill:
                    self.main_pools[rank].dispatch[AmxConfigArgs,
                        amx_prefill_config_kernel](
                        UnsafePointer(to=cfg_args[0]), cap)
                else:
                    self.main_pools[rank].dispatch[AmxConfigArgs,
                        amx_config_kernel[C.HPG]](
                        UnsafePointer(to=cfg_args[0]), cap)
                self.main_pools[rank].join()

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
            self.x_main_ptrs(), seq_len, self.main_pools)
        sample.add(self.profile.phase("broadcast"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_bcast0))

        var act_scale_lease = self.scratch.borrow[Float32, PREFILL_CHUNK_SIZE]()

        for layer_idx in range(C.NUM_LAYERS):
            # =============================================================
            # ATTENTION BLOCK
            # =============================================================

            comptime QKV_LOCAL = S.QKV_LOCAL
            var qkv_lease = self.scratch.borrow[Scalar[DType.bfloat16], PREFILL_CHUNK_SIZE * QKV_LOCAL]()
            var attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], PREFILL_CHUNK_SIZE * C.HIDDEN]()
            var attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()

            var attn_quant_jobs = InlineArray[
                RmsNormFwhtQuantArgs, Self.tp](uninitialized=True)
            for r in range(Self.tp):
                var topo_r = topos[r]
                var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                var layer = topo_r.layers.proto
                var sb = topo_r.arena.scratch_base()
                attn_quant_jobs[r] = RmsNormFwhtQuantArgs(
                    topo_r.activations.x_main.bound_dyn(topo_r.arena.base, seq_len).as_ptr[DType.bfloat16](),
                    layer.body.input_norm_sqrt.bound(lb).as_ptr[DType.bfloat16](),
                    attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len).as_ptr[DType.int8](),
                    attn_work_lease.view[F32, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.float32](),
                    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len).as_ptr[DType.float32](),
                    EPS, 0, seq_len)
            sample.add(
                self.profile.phase("attn_quantize"),
                run_tp_single_job_phase[
                    Self.tp,
                    rmsnorm_fwht_quant_worker[
                        C.HIDDEN, FWHT_BLK_HIDDEN, True, False],
                ](UnsafePointer(to=attn_quant_jobs[0]), self.main_pools),
            )

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

            var t_norm_prep0 = Int(perf_counter_ns())
            var inv_rms_q_arr = InlineArray[Float32, PREFILL_CHUNK_SIZE](
                fill=Float32(0))
            var inv_rms_k_arr = InlineArray[Float32, PREFILL_CHUNK_SIZE](
                fill=Float32(0))
            for row in range(seq_len):
                var q_ss = Float32(0)
                var k_ss = Float32(0)
                for r in range(Self.tp):
                    var sb = topos[r].arena.scratch_base()
                    var qkv_ptr = qkv_lease.as_ptr[
                        Scalar[DType.bfloat16]](sb, row * QKV_LOCAL)
                    q_ss += rms_reduce_bf16[Q_LOCAL](qkv_ptr)
                    k_ss += rms_reduce_bf16[KV_LOCAL](qkv_ptr + Q_LOCAL)
                inv_rms_q_arr[row] = inv_rms_from_sum_sq(q_ss, C.Q_DIM, EPS)
                inv_rms_k_arr[row] = inv_rms_from_sum_sq(k_ss, C.KV_DIM, EPS)
            sample.add(self.profile.phase("norm_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_norm_prep0))

            @parameter
            def do_kv_write[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                var lb = topo.layers.base(topo.arena.base, layer_idx)
                var layer = topo.layers.proto
                var sb = topo.arena.scratch_base()
                var qkv_ptr = qkv_lease.as_ptr[Scalar[DType.bfloat16]](sb)
                return kv_write_dispatch[
                    C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
                    C.MAX_SEQ_LEN, KV_PER_RANK, QKV_LOCAL,
                    Q_LOCAL, ROPE_HALF](
                    qkv_ptr,
                    layer.attn.k_norm.bound(lb).as_ptr[DType.bfloat16](),
                    topo.rope.cos.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
                    topo.rope.sin.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
                    UnsafePointer(to=inv_rms_k_arr[0]).bitcast[Float32]().as_any_origin(),
                    topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
                    start_pos, seq_len, pool)
            sample.add(self.profile.phase("kv_write"), timed_tp_parallel[Self.tp,do_kv_write](topos, self.main_pools))

            var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], PREFILL_CHUNK_SIZE * Q_LOCAL]()
            var attn_head_sc_lease = self.scratch.borrow[Float32, PREFILL_CHUNK_SIZE * HEADS_PER_RANK]()
            var q_i8_lease = self.scratch.borrow[
                Scalar[DType.int8], PREFILL_CHUNK_SIZE * KV_PER_RANK * HPG * C.HEAD_DIM]()
            var qi_biases_lease = self.scratch.borrow[
                Float32, PREFILL_CHUNK_SIZE * KV_PER_RANK * HPG]()
            var q_scales_lease = self.scratch.borrow[
                Float32, PREFILL_CHUNK_SIZE * KV_PER_RANK * HPG]()
            var partial_lease = self.scratch.borrow[Float32, PARTIAL_F32S]()

            var context_len = start_pos + seq_len

            var t_qprep0 = Int(perf_counter_ns())
            for r in range(Self.tp):
                var topo_r = topos[r]
                var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                var layer = topo_r.layers.proto
                var sb = topo_r.arena.scratch_base()
                var qkv_ptr = qkv_lease.as_ptr[Scalar[DType.bfloat16]](sb)
                q_prep_batch_dispatch[
                    C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
                    HPG, KV_PER_RANK, QKV_LOCAL, ROPE_HALF](
                    qkv_ptr,
                    layer.attn.q_norm.bound(lb).as_ptr[DType.bfloat16](),
                    topo_r.rope.cos.bound_row(topo_r.arena.base, 0).as_ptr[DType.float32](),
                    topo_r.rope.sin.bound_row(topo_r.arena.base, 0).as_ptr[DType.float32](),
                    UnsafePointer(to=inv_rms_q_arr[0]).bitcast[Float32]().as_any_origin(),
                    q_i8_lease.as_ptr[Scalar[DType.int8]](sb),
                    qi_biases_lease.as_ptr[Float32](sb),
                    q_scales_lease.as_ptr[Float32](sb),
                    start_pos, seq_len)
            sample.add(self.profile.phase("q_prep"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_qprep0))

            if seq_len == 1:
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

                var merge_quant_jobs = InlineArray[
                    MergeQuantArgs, Self.tp](uninitialized=True)
                for r in range(Self.tp):
                    var sb = topos[r].arena.scratch_base()
                    merge_quant_jobs[r] = MergeQuantArgs(
                        partial_lease.view[F32, Shape[1, PARTIAL_F32S]](
                            sb, 1).as_ptr[DType.float32](),
                        attn_chunk_count(context_len, C.MAX_ATTN_CHUNKS),
                        attn_qi_lease.view[
                            I8, Shape[1, KV_PER_RANK * HPG * C.HEAD_DIM]](
                            sb, 1).as_ptr[DType.int8](),
                        attn_head_sc_lease.view[
                            F32, Shape[1, KV_PER_RANK * HPG]](
                            sb, 1).as_ptr[DType.float32](),
                    )
                sample.add(
                    self.profile.phase("merge_quant"),
                    run_tp_single_job_phase[
                        Self.tp,
                        merge_quant_worker[
                            C.HEAD_DIM, HPG, C.MAX_ATTN_CHUNKS, KV_PER_RANK],
                    ](UnsafePointer(to=merge_quant_jobs[0]), self.main_pools),
                )
            else:
                @parameter
                def do_prefill_attn[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
                    var sb = topo.arena.scratch_base()
                    return prefill_attn_dispatch[
                        C.HEAD_DIM, HPG, C.MAX_SEQ_LEN, KV_PER_RANK,
                        C.NUM_HEADS, Q_LOCAL, HEADS_PER_RANK](
                        q_i8_lease.as_ptr[Scalar[DType.int8]](sb),
                        qi_biases_lease.as_ptr[Float32](sb),
                        q_scales_lease.as_ptr[Float32](sb),
                        topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
                        start_pos, seq_len,
                        attn_qi_lease.as_ptr[Scalar[DType.int8]](sb),
                        attn_head_sc_lease.as_ptr[Float32](sb),
                        pool)
                sample.add(self.profile.phase("attention"), timed_tp_parallel[Self.tp,do_prefill_attn](topos, self.main_pools))

            partial_lease^.release()
            q_scales_lease^.release()
            qi_biases_lease^.release()
            q_i8_lease^.release()

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
            self.residual_add_allreduce(seq_len)
            sample.add(self.profile.phase("attn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_attn_reduce0))

            # =============================================================
            # FFN BLOCK
            # =============================================================

            # Dual-output norm: split-gamma i8 plus full-gamma bf16.
            # LIFO ordering: borrow in reverse order of release-time. Long-lived
            # leases used by the MoE dispatch sit at the bottom; transient
            # scratch (moe_work, only needed during the dual-norm dispatch) is
            # borrowed last so it can be released first.
            var moe_i8_lease = self.scratch.borrow[Scalar[DType.int8], PREFILL_CHUNK_SIZE * C.HIDDEN]()
            var moe_scale_lease = self.scratch.borrow[Float32, PREFILL_CHUNK_SIZE]()
            var normed_bf16_lease = self.scratch.borrow[Scalar[DType.bfloat16], PREFILL_CHUNK_SIZE * C.HIDDEN]()
            var moe_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()

            var dual_norm_jobs = InlineArray[
                RmsNormDualOutputArgs, Self.tp](uninitialized=True)
            for r in range(Self.tp):
                var topo_r = topos[r]
                var lb = topo_r.layers.base(topo_r.arena.base, layer_idx)
                var layer = topo_r.layers.proto
                var sb = topo_r.arena.scratch_base()
                dual_norm_jobs[r] = RmsNormDualOutputArgs(
                    topo_r.activations.x_main.bound_dyn(topo_r.arena.base, seq_len).as_ptr[DType.bfloat16](),
                    layer.body.post_attn_norm_sqrt.bound(lb).as_ptr[DType.bfloat16](),
                    layer.body.post_attn_norm.bound(lb).as_ptr[DType.bfloat16](),
                    moe_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len).as_ptr[DType.int8](),
                    moe_work_lease.view[F32, Shape[1, C.HIDDEN]](sb, 1).as_ptr[DType.float32](),
                    moe_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len).as_ptr[DType.float32](),
                    normed_bf16_lease.view[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len).as_ptr[DType.bfloat16](),
                    EPS, 0, seq_len)
            sample.add(
                self.profile.phase("dual_norm"),
                run_tp_single_job_phase[
                    Self.tp,
                    rmsnorm_dual_output_worker[C.HIDDEN, FWHT_BLK_HIDDEN],
                ](UnsafePointer(to=dual_norm_jobs[0]), self.main_pools),
            )

            moe_work_lease^.release()

            if seq_len == 1:
                self.moe_phase_decode(
                    sample, layer_idx, seq_len,
                    moe_i8_lease, moe_scale_lease, normed_bf16_lease)
            else:
                self.moe_phase_prefill(
                    sample, layer_idx, seq_len,
                    moe_i8_lease, moe_scale_lease, normed_bf16_lease)

            var t_ffn_reduce0 = Int(perf_counter_ns())
            self.residual_add_allreduce(seq_len)
            sample.add(self.profile.phase("ffn_reduce"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_ffn_reduce0))

            normed_bf16_lease^.release()
            moe_scale_lease^.release()
            moe_i8_lease^.release()

        act_scale_lease^.release()

        if not produce_next_token:
            sample.wall_ns = Int(perf_counter_ns()) - t_forward0
            self.profile.record(sample)
            return Int32(-1)

        # --- Final norm + row-sharded LM head ---
        var lm_act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
        var lm_act_blk_scale_lease = self.scratch.borrow[Float32, VOCAB_NUM_BLOCKS]()
        var lm_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
        var t_final0 = Int(perf_counter_ns())
        comptime last_row_shape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
        var last_row_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=host.arena.base + host.activations.x_main.offset
                + (seq_len - 1) * C.HIDDEN * 2)
        var last_row_view = DynamicView[BF16, last_row_shape](last_row_ptr, 1)
        var final_fence = rmsnorm_gamma_fwht_per_block_quantize[
            C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK](
            last_row_view,
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

        var lm_act_ptrs = InlineArray[Int, Self.tp](fill=0)
        var lm_scale_ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            var sb_r = topos[r].arena.scratch_base()
            lm_act_ptrs[r] = Int(lm_act_i8_lease.as_ptr[Scalar[DType.int8]](sb_r))
            lm_scale_ptrs[r] = Int(lm_act_blk_scale_lease.as_ptr[Float32](sb_r))

        var t_lm_bcast0 = Int(perf_counter_ns())
        ring_broadcast[I8, Shape[1, C.HIDDEN], Self.tp](
            lm_act_ptrs[0], lm_act_ptrs, 1, self.main_pools)
        ring_broadcast[F32, Shape[1, VOCAB_NUM_BLOCKS], Self.tp](
            lm_scale_ptrs[0], lm_scale_ptrs, 1, self.main_pools)
        sample.add(self.profile.phase("lm_act_bcast"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_lm_bcast0))

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], VOCAB_LOCAL]()

        @parameter
        def do_lm_head[rank: Int, origin: MutOrigin](topo: MiniMaxM27Topology[Self.tp], ref [origin] pool: Self.Pool) -> PoolFence[Self.Pool, origin]:
            var sb = topo.arena.scratch_base()
            return lm_head_gemv[
                VOCAB_LOCAL, C.HIDDEN, LM_OUTPUT_HEAD_FWHT_BLK](
                lm_act_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1),
                topo.host.lm_output_head.bound(topo.arena.base),
                lm_act_blk_scale_lease.view[F32, Shape[1, VOCAB_NUM_BLOCKS]](sb, 1),
                topo.host.lm_output_head_sc.bound(topo.arena.base),
                topo.host.lm_output_head_colsum.bound(topo.arena.base),
                logit_lease.view[BF16, Shape[1, VOCAB_LOCAL]](sb, 1),
                pool)
        sample.add(self.profile.phase("lm_head"), timed_tp_parallel[Self.tp, do_lm_head](topos, self.main_pools))

        # Global greedy over row-sharded logits (no softcap in MiniMax).
        var t_argmax0 = Int(perf_counter_ns())
        comptime width = simd_width_of[DType.float32]()
        var best_val = Float32(-1e30)
        var best_idx = Int32(0)
        for r in range(Self.tp):
            var sb_r = topos[r].arena.scratch_base()
            var local_logits = logit_lease.as_ptr[Scalar[DType.bfloat16]](sb_r)
            var vocab_base = r * VOCAB_LOCAL
            for j in range(0, VOCAB_LOCAL, width):
                var v = (local_logits + j).load[width=width]().cast[DType.float32]()
                for k in range(width):
                    var global_idx = vocab_base + j + k
                    if global_idx < C.GENERATION_VOCAB_SIZE and v[k] > best_val:
                        best_val = v[k]
                        best_idx = Int32(global_idx)
        sample.add(self.profile.phase("lm_argmax"), PhaseTiming.opaque(Int(perf_counter_ns()) - t_argmax0))
        logit_lease^.release()
        lm_act_blk_scale_lease^.release()
        lm_act_i8_lease^.release()
        sample.wall_ns = Int(perf_counter_ns()) - t_forward0
        self.profile.record(sample)
        return best_idx
