"""Gemma 4 26B-A4B ButterQuant — int8 MoE with NUMA-aware sparse expert dispatch.

Layout: [25 sliding layers][5 full layers][state][host-only: final_norm, embed]
State:  [x_main][x_residual][scratch][rope tables][KV caches]

Weight layout per quantized projection:
  i8 weight (contiguous across fused groups like Q|K|V)
  f32 per-row scale
Column sums and VNNI packing computed at load time.

3 pools per rank: main_pool (all cores), expert_pool (half), dense_pool (half).
ranks.parallel[body](pool_ptrs) dispatches body to all ranks.
Body threads never compute — dispatch and return only.
"""

from std.pathlib import Path
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import size_of, simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32, I8,
    RowShard, ColShard, Replicated,
    IsQuantizable, IsPassthrough,
    Slot, PlacedSlot, Bound, DynView, bind, byte_count,
    WeightIterable, next_offset,
    DEFAULT_ALIGNMENT,
    Kernel3DTiling, LogitsView,
)
from kernels.vnni import VnniPacked
from kernels.kernel_ops import PoolFence, parallel_for, BF16Ptr, rmsnorm
from kernels.reductions import ring_allreduce, ring_broadcast
from kernels.kv_rotors import init_rope_tables
from experimental.linear_borrow_pool import ScratchPool, ScratchLease
from experimental2.kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize, rmsnorm_gamma_fwht_quantize
from experimental2.kernels.int8_gemv import int8_gemv
from experimental2.kernels.float_gemv import float_gemv
from experimental3.moe import (
    gemma4_moe_dispatch_local, moe_combine,
)
from experimental3.kv_cache import Gemma4KVCache
from experimental3.kernels.dense_ffn import (
    fused_gu_gelu_tanh,
    int8_gemv_blocked,
    RouterTopkArgs, router_topk_kernel,
)
from experimental3.kernels.sliding_attention import (
    SlidingAttnGroupArgs, sliding_attn_group_kernel,
)
from experimental3.kernels.full_attention import (
    FullAttnGroupArgs, full_attn_group_kernel,
)
from experimental_gemma.router import Gemma4TopKResult
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables
from experimental_gemma.ops import embed_lookup_scaled, logit_softcap
from modeling.loader import load_weights
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
# Sliding attention layer (25 of 30) — has Q, K, V projections
# =============================================================================


struct SlidingLayer[tp: Int]:
    # --- Attention i8: contiguous [Q|K|V], then scales [Q_sc|K_sc|V_sc] ---
    comptime Q_PROJ    = PlacedSlot[I8, RowShard, C.Q_DIM_SLIDING,  C.HIDDEN, Self.tp, 0,                            "self_attn.q_proj.weight", IsQuantizable, VnniPacked]
    comptime K_PROJ    = PlacedSlot[I8, RowShard, C.KV_DIM_SLIDING, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](),    "self_attn.k_proj.weight", IsQuantizable, VnniPacked]
    comptime V_PROJ    = PlacedSlot[I8, RowShard, C.KV_DIM_SLIDING, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](),    "self_attn.v_proj.weight", IsQuantizable, VnniPacked]
    comptime Q_PROJ_SC = PlacedSlot[F32, RowShard, C.Q_DIM_SLIDING, 1,        Self.tp, next_offset[Self.V_PROJ](),    "self_attn.q_proj.weight_scale"]
    comptime K_PROJ_SC = PlacedSlot[F32, RowShard, C.KV_DIM_SLIDING, 1,       Self.tp, next_offset[Self.Q_PROJ_SC](), "self_attn.k_proj.weight_scale"]
    comptime V_PROJ_SC = PlacedSlot[F32, RowShard, C.KV_DIM_SLIDING, 1,       Self.tp, next_offset[Self.K_PROJ_SC](), "self_attn.v_proj.weight_scale"]
    # --- O projection ---
    comptime O_PROJ    = PlacedSlot[I8, ColShard, C.HIDDEN, C.Q_DIM_SLIDING,  Self.tp, next_offset[Self.V_PROJ_SC](), "self_attn.o_proj.weight", IsQuantizable, VnniPacked]
    comptime O_PROJ_SC = PlacedSlot[F32, Replicated, C.HIDDEN, 1,             Self.tp, next_offset[Self.O_PROJ](),    "self_attn.o_proj.weight_scale"]
    # --- Per-head norms (runtime, not absorbed) ---
    comptime Q_NORM     = PlacedSlot[BF16, Replicated, C.HEAD_DIM_SLIDING, 1, Self.tp, next_offset[Self.O_PROJ_SC](), "self_attn.q_norm.weight"]
    comptime K_NORM     = PlacedSlot[BF16, Replicated, C.HEAD_DIM_SLIDING, 1, Self.tp, next_offset[Self.Q_NORM](),    "self_attn.k_norm.weight"]
    # --- Runtime norms (gamma applied to activation, not absorbed into weights) ---
    comptime INPUT_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,           Self.tp, next_offset[Self.K_NORM](),    "input_layernorm.weight"]
    # --- Dense MLP: contiguous [gate|up], then scales ---
    comptime PRE_FFN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,            Self.tp, next_offset[Self.INPUT_NORM](),   "pre_feedforward_layernorm.weight"]
    comptime GATE_PROJ    = PlacedSlot[I8, Replicated, C.INTERMEDIATE, C.HIDDEN,  Self.tp, next_offset[Self.PRE_FFN_NORM](), "mlp.gate_proj.weight", IsQuantizable, VnniPacked]
    comptime UP_PROJ      = PlacedSlot[I8, Replicated, C.INTERMEDIATE, C.HIDDEN,  Self.tp, next_offset[Self.GATE_PROJ](),    "mlp.up_proj.weight", IsQuantizable, VnniPacked]
    comptime GATE_PROJ_SC = PlacedSlot[F32, Replicated, C.INTERMEDIATE, 1,        Self.tp, next_offset[Self.UP_PROJ](),      "mlp.gate_proj.weight_scale"]
    comptime UP_PROJ_SC   = PlacedSlot[F32, Replicated, C.INTERMEDIATE, 1,        Self.tp, next_offset[Self.GATE_PROJ_SC](), "mlp.up_proj.weight_scale"]
    comptime DOWN_PROJ    = PlacedSlot[I8, Replicated, C.HIDDEN, C.INTERMEDIATE,  Self.tp, next_offset[Self.UP_PROJ_SC](),   "mlp.down_proj.weight", IsQuantizable, VnniPacked]
    comptime DOWN_PROJ_SC = PlacedSlot[F32, Replicated, C.HIDDEN, 1,              Self.tp, next_offset[Self.DOWN_PROJ](),    "mlp.down_proj.weight_scale"]
    # --- Router ---
    comptime ROUTER_SCALE   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,           Self.tp, next_offset[Self.DOWN_PROJ_SC](),   "router.scale"]
    comptime ROUTER_PROJ    = PlacedSlot[I8, Replicated, C.NUM_EXPERTS, C.HIDDEN,  Self.tp, next_offset[Self.ROUTER_SCALE](),   "router.proj.weight", IsQuantizable, VnniPacked]
    comptime ROUTER_PROJ_SC = PlacedSlot[F32, Replicated, C.NUM_EXPERTS, 1,        Self.tp, next_offset[Self.ROUTER_PROJ](),    "router.proj.weight_scale"]
    comptime ROUTER_PES     = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, 1,       Self.tp, next_offset[Self.ROUTER_PROJ_SC](), "router.per_expert_scale"]
    # --- Expert pre-norm + weights (packed 2D) ---
    comptime PRE_FFN_NORM_2     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,                                    Self.tp, next_offset[Self.ROUTER_PES](),          "pre_feedforward_layernorm_2.weight"]
    comptime EXPERTS_GATE_UP    = PlacedSlot[I8, Replicated, C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN,    Self.tp, next_offset[Self.PRE_FFN_NORM_2](),      "experts.gate_up_proj", IsQuantizable, VnniPacked]
    comptime EXPERTS_GATE_UP_SC = PlacedSlot[F32, Replicated, C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, 1,          Self.tp, next_offset[Self.EXPERTS_GATE_UP](),      "experts.gate_up_proj_scale"]
    comptime EXPERTS_DOWN       = PlacedSlot[I8, Replicated, C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE,     Self.tp, next_offset[Self.EXPERTS_GATE_UP_SC](),   "experts.down_proj", IsQuantizable, VnniPacked]
    comptime EXPERTS_DOWN_SC    = PlacedSlot[F32, Replicated, C.NUM_EXPERTS * C.HIDDEN, 1,                     Self.tp, next_offset[Self.EXPERTS_DOWN](),         "experts.down_proj_scale"]
    # --- Non-absorbable norms + scalar ---
    comptime POST_ATTN_NORM    = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.EXPERTS_DOWN_SC](),  "post_attention_layernorm.weight"]
    comptime POST_FFN_NORM_1   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_ATTN_NORM](),   "post_feedforward_layernorm_1.weight"]
    comptime POST_FFN_NORM_2_RT = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_1](), "post_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_2_RT](), "post_feedforward_layernorm.weight"]
    comptime LAYER_SCALAR      = PlacedSlot[BF16, Replicated, 1, 1,         Self.tp, next_offset[Self.POST_FFN_NORM](),    "layer_scalar"]
    # --- Column sums (computed at load time, raw byte offsets) ---
    comptime QKV_N = C.Q_DIM_SLIDING + C.KV_DIM_SLIDING + C.KV_DIM_SLIDING
    comptime O_NUM_BLOCKS = (C.Q_DIM_SLIDING // Self.tp) // C.HEAD_DIM_SLIDING
    comptime QKV_COLSUM_OFF = next_offset[Self.LAYER_SCALAR]()
    comptime O_COLSUM_OFF   = Self.QKV_COLSUM_OFF + Self.QKV_N * 4
    comptime GU_COLSUM_OFF  = Self.O_COLSUM_OFF + C.HIDDEN * Self.O_NUM_BLOCKS * 4
    comptime DOWN_COLSUM_OFF = Self.GU_COLSUM_OFF + C.INTERMEDIATE * 2 * 4
    comptime ROUTER_COLSUM_OFF = Self.DOWN_COLSUM_OFF + C.HIDDEN * C.DENSE_NUM_BLOCKS * 4
    comptime EXPERTS_GU_COLSUM_OFF = Self.ROUTER_COLSUM_OFF + C.NUM_EXPERTS * 4
    comptime EXPERTS_DOWN_COLSUM_OFF = Self.EXPERTS_GU_COLSUM_OFF + C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED * 4
    comptime STRIDE = Self.EXPERTS_DOWN_COLSUM_OFF + C.NUM_EXPERTS * C.HIDDEN * C.MOE_NUM_BLOCKS * 4

    comptime Q_DIM_LOCAL = C.Q_DIM_SLIDING // Self.tp
    comptime KV_DIM_LOCAL = C.KV_DIM_SLIDING // Self.tp
    comptime HEADS_PER_GROUP = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
    comptime NUM_KV_LOCAL = C.NUM_KV_HEADS_SLIDING // Self.tp

    @staticmethod
    def for_each_weight[func: def[T: Encoding & Shaped & Placed & Named](String, Int, Int) capturing -> None](prefix: String, base: Int):
        # Logical quantizer order: runtime full-attn input norm first, then the weights it feeds.
        func[Self.INPUT_NORM](prefix, base, -1)
        func[Self.Q_PROJ](prefix, base, -1)
        func[Self.K_PROJ](prefix, base, -1)
        func[Self.V_PROJ](prefix, base, -1)
        func[Self.Q_PROJ_SC](prefix, base, -1)
        func[Self.K_PROJ_SC](prefix, base, -1)
        func[Self.V_PROJ_SC](prefix, base, -1)
        func[Self.O_PROJ](prefix, base, -1)
        func[Self.O_PROJ_SC](prefix, base, -1)
        func[Self.Q_NORM](prefix, base, -1)
        func[Self.K_NORM](prefix, base, -1)
        func[Self.PRE_FFN_NORM](prefix, base, -1)
        func[Self.GATE_PROJ](prefix, base, -1)
        func[Self.UP_PROJ](prefix, base, -1)
        func[Self.GATE_PROJ_SC](prefix, base, -1)
        func[Self.UP_PROJ_SC](prefix, base, -1)
        func[Self.DOWN_PROJ](prefix, base, -1)
        func[Self.DOWN_PROJ_SC](prefix, base, -1)
        func[Self.ROUTER_SCALE](prefix, base, -1)
        func[Self.ROUTER_PROJ](prefix, base, -1)
        func[Self.ROUTER_PROJ_SC](prefix, base, -1)
        func[Self.ROUTER_PES](prefix, base, -1)
        func[Self.PRE_FFN_NORM_2](prefix, base, -1)
        func[Self.EXPERTS_GATE_UP](prefix, base, -1)
        func[Self.EXPERTS_GATE_UP_SC](prefix, base, -1)
        func[Self.EXPERTS_DOWN](prefix, base, -1)
        func[Self.EXPERTS_DOWN_SC](prefix, base, -1)
        func[Self.POST_ATTN_NORM](prefix, base, -1)
        func[Self.POST_FFN_NORM_1](prefix, base, -1)
        func[Self.POST_FFN_NORM_2_RT](prefix, base, -1)
        func[Self.POST_FFN_NORM](prefix, base, -1)
        func[Self.LAYER_SCALAR](prefix, base, -1)


# =============================================================================
# Full attention layer (5 of 30) — bf16 Q/K silo, K=V shared, no V_PROJ
# =============================================================================


struct FullLayer[tp: Int]:
    # --- Attention i8: contiguous [Q|K] (no V — K=V shared), then scales ---
    comptime Q_PROJ    = PlacedSlot[I8, RowShard, C.Q_DIM_FULL,  C.HIDDEN, Self.tp, 0,                            "self_attn.q_proj.weight", IsQuantizable, VnniPacked]
    comptime K_PROJ    = PlacedSlot[I8, RowShard, C.KV_DIM_FULL, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](),    "self_attn.k_proj.weight", IsQuantizable, VnniPacked]
    comptime Q_PROJ_SC = PlacedSlot[F32, RowShard, C.Q_DIM_FULL, 1,        Self.tp, next_offset[Self.K_PROJ](),    "self_attn.q_proj.weight_scale"]
    comptime K_PROJ_SC = PlacedSlot[F32, RowShard, C.KV_DIM_FULL, 1,       Self.tp, next_offset[Self.Q_PROJ_SC](), "self_attn.k_proj.weight_scale"]
    # --- O projection ---
    comptime O_PROJ    = PlacedSlot[I8, ColShard, C.HIDDEN, C.Q_DIM_FULL,  Self.tp, next_offset[Self.K_PROJ_SC](), "self_attn.o_proj.weight", IsQuantizable, VnniPacked]
    comptime O_PROJ_SC = PlacedSlot[F32, Replicated, C.HIDDEN, 1,          Self.tp, next_offset[Self.O_PROJ](),    "self_attn.o_proj.weight_scale"]
    # --- Per-head norms (runtime, not absorbed) ---
    comptime Q_NORM     = PlacedSlot[BF16, Replicated, C.HEAD_DIM_FULL, 1, Self.tp, next_offset[Self.O_PROJ_SC](), "self_attn.q_norm.weight"]
    comptime K_NORM     = PlacedSlot[BF16, Replicated, C.HEAD_DIM_FULL, 1, Self.tp, next_offset[Self.Q_NORM](),    "self_attn.k_norm.weight"]
    # --- Runtime norms ---
    comptime INPUT_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,        Self.tp, next_offset[Self.K_NORM](),    "input_layernorm.weight"]
    # --- Dense MLP (identical to sliding) ---
    comptime PRE_FFN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,            Self.tp, next_offset[Self.INPUT_NORM](),   "pre_feedforward_layernorm.weight"]
    comptime GATE_PROJ    = PlacedSlot[I8, Replicated, C.INTERMEDIATE, C.HIDDEN,  Self.tp, next_offset[Self.PRE_FFN_NORM](), "mlp.gate_proj.weight", IsQuantizable, VnniPacked]
    comptime UP_PROJ      = PlacedSlot[I8, Replicated, C.INTERMEDIATE, C.HIDDEN,  Self.tp, next_offset[Self.GATE_PROJ](),    "mlp.up_proj.weight", IsQuantizable, VnniPacked]
    comptime GATE_PROJ_SC = PlacedSlot[F32, Replicated, C.INTERMEDIATE, 1,        Self.tp, next_offset[Self.UP_PROJ](),      "mlp.gate_proj.weight_scale"]
    comptime UP_PROJ_SC   = PlacedSlot[F32, Replicated, C.INTERMEDIATE, 1,        Self.tp, next_offset[Self.GATE_PROJ_SC](), "mlp.up_proj.weight_scale"]
    comptime DOWN_PROJ    = PlacedSlot[I8, Replicated, C.HIDDEN, C.INTERMEDIATE,  Self.tp, next_offset[Self.UP_PROJ_SC](),   "mlp.down_proj.weight", IsQuantizable, VnniPacked]
    comptime DOWN_PROJ_SC = PlacedSlot[F32, Replicated, C.HIDDEN, 1,              Self.tp, next_offset[Self.DOWN_PROJ](),    "mlp.down_proj.weight_scale"]
    # --- Router ---
    comptime ROUTER_SCALE   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,           Self.tp, next_offset[Self.DOWN_PROJ_SC](),   "router.scale"]
    comptime ROUTER_PROJ    = PlacedSlot[I8, Replicated, C.NUM_EXPERTS, C.HIDDEN,  Self.tp, next_offset[Self.ROUTER_SCALE](),   "router.proj.weight", IsQuantizable, VnniPacked]
    comptime ROUTER_PROJ_SC = PlacedSlot[F32, Replicated, C.NUM_EXPERTS, 1,        Self.tp, next_offset[Self.ROUTER_PROJ](),    "router.proj.weight_scale"]
    comptime ROUTER_PES     = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, 1,       Self.tp, next_offset[Self.ROUTER_PROJ_SC](), "router.per_expert_scale"]
    # --- Expert pre-norm + weights (packed 2D) ---
    comptime PRE_FFN_NORM_2     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1,                                    Self.tp, next_offset[Self.ROUTER_PES](),          "pre_feedforward_layernorm_2.weight"]
    comptime EXPERTS_GATE_UP    = PlacedSlot[I8, Replicated, C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN,    Self.tp, next_offset[Self.PRE_FFN_NORM_2](),      "experts.gate_up_proj", IsQuantizable, VnniPacked]
    comptime EXPERTS_GATE_UP_SC = PlacedSlot[F32, Replicated, C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, 1,          Self.tp, next_offset[Self.EXPERTS_GATE_UP](),      "experts.gate_up_proj_scale"]
    comptime EXPERTS_DOWN       = PlacedSlot[I8, Replicated, C.NUM_EXPERTS * C.HIDDEN, C.MOE_INTERMEDIATE,     Self.tp, next_offset[Self.EXPERTS_GATE_UP_SC](),   "experts.down_proj", IsQuantizable, VnniPacked]
    comptime EXPERTS_DOWN_SC    = PlacedSlot[F32, Replicated, C.NUM_EXPERTS * C.HIDDEN, 1,                     Self.tp, next_offset[Self.EXPERTS_DOWN](),         "experts.down_proj_scale"]
    # --- Non-absorbable norms + scalar ---
    comptime POST_ATTN_NORM    = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.EXPERTS_DOWN_SC](),  "post_attention_layernorm.weight"]
    comptime POST_FFN_NORM_1   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_ATTN_NORM](),   "post_feedforward_layernorm_1.weight"]
    comptime POST_FFN_NORM_2_RT = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_1](), "post_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_2_RT](), "post_feedforward_layernorm.weight"]
    comptime LAYER_SCALAR      = PlacedSlot[BF16, Replicated, 1, 1,         Self.tp, next_offset[Self.POST_FFN_NORM](),    "layer_scalar"]
    # --- Column sums (computed at load time, raw byte offsets) ---
    comptime QK_N = C.Q_DIM_FULL + C.KV_DIM_FULL
    comptime O_NUM_BLOCKS = (C.Q_DIM_FULL // Self.tp) // C.HEAD_DIM_FULL
    comptime QK_COLSUM_OFF  = next_offset[Self.LAYER_SCALAR]()
    comptime O_COLSUM_OFF   = Self.QK_COLSUM_OFF + Self.QK_N * 4
    comptime GU_COLSUM_OFF  = Self.O_COLSUM_OFF + C.HIDDEN * Self.O_NUM_BLOCKS * 4
    comptime DOWN_COLSUM_OFF = Self.GU_COLSUM_OFF + C.INTERMEDIATE * 2 * 4
    comptime ROUTER_COLSUM_OFF = Self.DOWN_COLSUM_OFF + C.HIDDEN * C.DENSE_NUM_BLOCKS * 4
    comptime EXPERTS_GU_COLSUM_OFF = Self.ROUTER_COLSUM_OFF + C.NUM_EXPERTS * 4
    comptime EXPERTS_DOWN_COLSUM_OFF = Self.EXPERTS_GU_COLSUM_OFF + C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED * 4
    comptime STRIDE = Self.EXPERTS_DOWN_COLSUM_OFF + C.NUM_EXPERTS * C.HIDDEN * C.MOE_NUM_BLOCKS * 4

    comptime Q_DIM_LOCAL = C.Q_DIM_FULL // Self.tp
    comptime KV_DIM_LOCAL = C.KV_DIM_FULL // Self.tp
    comptime HEADS_PER_GROUP = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
    comptime NUM_KV_LOCAL = C.NUM_KV_HEADS_FULL

    @staticmethod
    def for_each_weight[func: def[T: Encoding & Shaped & Placed & Named](String, Int, Int) capturing -> None](prefix: String, base: Int):
        # Logical quantizer order: norms before the projections they feed.
        func[Self.INPUT_NORM](prefix, base, -1)
        func[Self.Q_PROJ](prefix, base, -1)
        func[Self.K_PROJ](prefix, base, -1)
        func[Self.Q_PROJ_SC](prefix, base, -1)
        func[Self.K_PROJ_SC](prefix, base, -1)
        func[Self.O_PROJ](prefix, base, -1)
        func[Self.O_PROJ_SC](prefix, base, -1)
        func[Self.Q_NORM](prefix, base, -1)
        func[Self.K_NORM](prefix, base, -1)
        func[Self.PRE_FFN_NORM](prefix, base, -1)
        func[Self.GATE_PROJ](prefix, base, -1)
        func[Self.UP_PROJ](prefix, base, -1)
        func[Self.GATE_PROJ_SC](prefix, base, -1)
        func[Self.UP_PROJ_SC](prefix, base, -1)
        func[Self.DOWN_PROJ](prefix, base, -1)
        func[Self.DOWN_PROJ_SC](prefix, base, -1)
        func[Self.ROUTER_SCALE](prefix, base, -1)
        func[Self.ROUTER_PROJ](prefix, base, -1)
        func[Self.ROUTER_PROJ_SC](prefix, base, -1)
        func[Self.ROUTER_PES](prefix, base, -1)
        func[Self.PRE_FFN_NORM_2](prefix, base, -1)
        func[Self.EXPERTS_GATE_UP](prefix, base, -1)
        func[Self.EXPERTS_GATE_UP_SC](prefix, base, -1)
        func[Self.EXPERTS_DOWN](prefix, base, -1)
        func[Self.EXPERTS_DOWN_SC](prefix, base, -1)
        func[Self.POST_ATTN_NORM](prefix, base, -1)
        func[Self.POST_FFN_NORM_1](prefix, base, -1)
        func[Self.POST_FFN_NORM_2_RT](prefix, base, -1)
        func[Self.POST_FFN_NORM](prefix, base, -1)
        func[Self.LAYER_SCALAR](prefix, base, -1)


# =============================================================================
# Model layout
# =============================================================================


struct Gemma4Model[tp: Int](WeightIterable):
    comptime SLIDING_STRIDE = SlidingLayer[1].STRIDE
    comptime FULL_STRIDE = FullLayer[1].STRIDE

    comptime SLIDING_OFF = 0
    comptime FULL_OFF = Self.SLIDING_OFF + C.NUM_SLIDING_LAYERS * Self.SLIDING_STRIDE
    comptime DISTRIBUTED_BYTES = Self.FULL_OFF + C.NUM_FULL_LAYERS * Self.FULL_STRIDE

    # State
    comptime X_MAIN_OFF = 0
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL_OFF = byte_count[Self.X_MAIN]()
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        """Peak ScratchPool bytes for the current decode-only forward path.

        `forward_decode` is hard-coded to `seq_len = 1`, so the true scratch
        requirement is the maximum cumulative borrow footprint across:

        - sliding attention
        - full attention
        - FFN/router/expert+dense path
        - final host logits buffer

        This is separate from loader/packing scratch. The previous value was a
        stale VNNI-pack target and substantially overestimated the live decode
        scratch requirement.
        """
        comptime bf16_bytes = size_of[Scalar[DType.bfloat16]]()
        comptime i8_bytes = size_of[Scalar[DType.int8]]()
        comptime f32_bytes = size_of[Float32]()
        comptime topk_bytes = size_of[Gemma4TopKResult[C.TOP_K]]()

        comptime persistent = (
            f32_bytes  # act_scale_lease
            + C.DENSE_NUM_BLOCKS * f32_bytes  # post_blk_scale_lease
        )

        comptime sliding_attn_peak = persistent + (
            C.HIDDEN * i8_bytes                       # attn_i8_lease
            + C.HIDDEN * f32_bytes                    # attn_work_lease
            + f32_bytes                               # attn_scale_lease
            + (SlidingLayer[1].QKV_N // Self.tp) * bf16_bytes
            + (C.Q_DIM_SLIDING // Self.tp) * i8_bytes
            + ((C.Q_DIM_SLIDING // Self.tp) // C.HEAD_DIM_SLIDING) * f32_bytes
        )

        comptime full_attn_phase1 = persistent + (
            (FullLayer[1].QK_N // Self.tp) * bf16_bytes
            + C.HIDDEN * i8_bytes
            + C.HIDDEN * f32_bytes
            + f32_bytes
        )
        comptime full_attn_phase2 = persistent + (
            (FullLayer[1].QK_N // Self.tp) * bf16_bytes
            + (C.Q_DIM_FULL // Self.tp) * i8_bytes
            + ((C.Q_DIM_FULL // Self.tp) // C.HEAD_DIM_FULL) * f32_bytes
        )
        comptime full_attn_peak = full_attn_phase1 if full_attn_phase1 > full_attn_phase2 else full_attn_phase2

        comptime ffn_peak = persistent + (
            C.HIDDEN * i8_bytes                       # act_i8_lease
            + C.HIDDEN * f32_bytes                    # act_work_lease
            + C.NUM_EXPERTS * bf16_bytes              # router_logits_lease
            + topk_bytes                              # routing_lease
            + C.TOP_K * C.HIDDEN * bf16_bytes         # expert_out_lease
            + size_of[Int32]()                        # local_count_lease
            + C.INTERMEDIATE * i8_bytes               # dense_post_i8_lease
            + C.HIDDEN * bf16_bytes                   # dense_out_lease
            + C.HIDDEN * bf16_bytes                   # dense_normed_lease
        )

        comptime layer_peak = (
            sliding_attn_peak if sliding_attn_peak > full_attn_peak else full_attn_peak
        )
        comptime decode_peak = ffn_peak if ffn_peak > layer_peak else layer_peak

        comptime logits_peak = C.VOCAB_SIZE * bf16_bytes
        comptime final_peak = logits_peak if logits_peak > decode_peak else decode_peak
        return final_peak

    # RoPE tables
    comptime SLIDING_ROPE_HALF = C.HEAD_DIM_SLIDING // 2
    comptime SLIDING_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime SLIDING_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_ROPE_HALF, Self.tp]
    comptime SLIDING_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime SLIDING_SIN_OFF = Self.SLIDING_COS_OFF + byte_count[Self.SLIDING_COS]()
    comptime FULL_ROPE_HALF = 64
    comptime FULL_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF, Self.tp]
    comptime FULL_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.FULL_ROPE_HALF, Self.tp]
    comptime FULL_COS_OFF = Self.SLIDING_SIN_OFF + byte_count[Self.SLIDING_SIN]()
    comptime FULL_SIN_OFF = Self.FULL_COS_OFF + byte_count[Self.FULL_COS]()

    # KV caches
    comptime SLIDING_CACHE = Gemma4KVCache[C.SLIDING_WINDOW, C.HEAD_DIM_SLIDING,
        C.NUM_KV_HEADS_SLIDING // Self.tp, C.NUM_HEADS // Self.tp]
    comptime SLIDING_CACHE_OFF = Self.FULL_SIN_OFF + byte_count[Self.FULL_SIN]()
    comptime SLIDING_CACHE_STRIDE = Self.SLIDING_CACHE.TOTAL_BYTES
    comptime FULL_CACHE = Gemma4KVCache[C.MAX_SEQ_LEN, C.HEAD_DIM_FULL,
        C.NUM_KV_HEADS_FULL, C.NUM_HEADS // Self.tp]
    comptime FULL_CACHE_OFF = Self.SLIDING_CACHE_OFF + C.NUM_SLIDING_LAYERS * Self.SLIDING_CACHE_STRIDE
    comptime FULL_CACHE_STRIDE = Self.FULL_CACHE.TOTAL_BYTES

    comptime STATE_BYTES = Self.FULL_CACHE_OFF + C.NUM_FULL_LAYERS * Self.FULL_CACHE_STRIDE

    # Host-only
    comptime HOST_ONLY_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, Self.HOST_ONLY_OFF, "model.language_model.norm.weight"]
    comptime EMBED = PlacedSlot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.language_model.embed_tokens.weight"]
    comptime LOGITS = Slot[BF16, Replicated, 1, C.VOCAB_SIZE, Self.tp]

    @staticmethod
    def host_arena_bytes() -> Int:
        return next_offset[Self.EMBED]()

    @staticmethod
    def arena_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    def for_each_weight[func: def[T: Encoding & Shaped & Placed & Named](String, Int, Int) capturing -> None]():
        var sliding_idx = 0
        var full_idx = 0
        comptime for i in range(C.NUM_LAYERS):
            var prefix = "model.language_model.layers." + String(i) + "."
            comptime if (i + 1) % 6 == 0:
                var base = Self.FULL_OFF + full_idx * Self.FULL_STRIDE
                FullLayer[1].for_each_weight[func](prefix, base)
                full_idx += 1
            else:
                var base = Self.SLIDING_OFF + sliding_idx * Self.SLIDING_STRIDE
                SlidingLayer[1].for_each_weight[func](prefix, base)
                sliding_idx += 1
        func[Self.FINAL_NORM]("", 0, -1)
        func[Self.EMBED]("", 0, -1)


# =============================================================================
# Rank view
# =============================================================================


@fieldwise_init
struct RankView[tp: Int]:
    comptime M = Gemma4Model[Self.tp]
    comptime SL = SlidingLayer[1]
    comptime FL = FullLayer[1]
    var base: Int

    def weight_base(self) -> Int:
        return self.base
    def state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES
    def scratch_base(self) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF
    def scratch_addr(self, read lease: ScratchLease) -> Int:
        return self.scratch_base() + lease.offset
    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)
    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        return DynView[V](self.scratch_base() + lease.offset, seq_len)

    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)
    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    def sliding_cos(self) -> Bound[Self.M.SLIDING_COS]:
        return Bound[Self.M.SLIDING_COS](self.state_base() + Self.M.SLIDING_COS_OFF)
    def sliding_sin(self) -> Bound[Self.M.SLIDING_SIN]:
        return Bound[Self.M.SLIDING_SIN](self.state_base() + Self.M.SLIDING_SIN_OFF)
    def full_cos(self) -> Bound[Self.M.FULL_COS]:
        return Bound[Self.M.FULL_COS](self.state_base() + Self.M.FULL_COS_OFF)
    def full_sin(self) -> Bound[Self.M.FULL_SIN]:
        return Bound[Self.M.FULL_SIN](self.state_base() + Self.M.FULL_SIN_OFF)

    def sliding_cos_row(self, pos: Int) -> Int:
        return self.state_base() + Self.M.SLIDING_COS_OFF + pos * Self.M.SLIDING_ROPE_HALF * size_of[Float32]()
    def sliding_sin_row(self, pos: Int) -> Int:
        return self.state_base() + Self.M.SLIDING_SIN_OFF + pos * Self.M.SLIDING_ROPE_HALF * size_of[Float32]()

    def sliding_cache_base(self, layer_idx: Int) -> Int:
        return self.state_base() + Self.M.SLIDING_CACHE_OFF + layer_idx * Self.M.SLIDING_CACHE_STRIDE
    def full_cache_base(self, full_layer_idx: Int) -> Int:
        return self.state_base() + Self.M.FULL_CACHE_OFF + full_layer_idx * Self.M.FULL_CACHE_STRIDE

    def sliding_layer_base(self, sliding_idx: Int) -> Int:
        return self.weight_base() + Self.M.SLIDING_OFF + sliding_idx * Self.M.SLIDING_STRIDE
    def full_layer_base(self, full_idx: Int) -> Int:
        return self.weight_base() + Self.M.FULL_OFF + full_idx * Self.M.FULL_STRIDE

    def host_weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())


# =============================================================================
# Ranks
# =============================================================================


@fieldwise_init
struct Ranks[tp: Int]:
    var bases: InlineArray[Int, Self.tp]

    def view(self, r: Int) -> RankView[Self.tp]:
        return RankView[Self.tp](self.bases[r])

    def parallel[body: def[rank: Int](RankView[Self.tp], mut BurstPool[]) capturing -> PoolFence[BurstPool[]]](
        self, pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp],
    ):
        @parameter
        def dispatch[rank: Int]() -> PoolFence[BurstPool[]]:
            return body[rank](self.view(rank), pool_ptrs[rank][])
        parallel_for[BurstPool[], Self.tp, dispatch]()

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
# Dispatched kernels
# =============================================================================


@fieldwise_init
struct PostAttnNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: Int
    var norm_w_ptr: Int
    var x_main_ptr: Int
    var eps: Float32

def post_attn_norm_kernel(args: PostAttnNormArgs):
    """Post-attention: rmsnorm + residual add."""
    comptime width = simd_width_of[DType.float32]()
    comptime h = C.HIDDEN
    var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.src_ptr)
    var norm_w = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.norm_w_ptr)
    var x_main = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.x_main_ptr)
    var sum_sq = SIMD[DType.float32, width](0)
    for i in range(0, h, width):
        var v = (src + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(h) + args.eps)
    for i in range(0, h, width):
        var v = (src + i).load[width=width]().cast[DType.float32]()
        var g = (norm_w + i).load[width=width]().cast[DType.float32]()
        var x = (x_main + i).load[width=width]().cast[DType.float32]()
        (x_main + i).store((x + v * inv_rms * g).cast[DType.bfloat16]())


@fieldwise_init
struct PreReduceArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: Int
    var local_count: Int
    var dst_ptr: Int
    var dense_ptr: Int
    var norm_w_ptr: Int
    var normed_ptr: Int
    var hidden: Int
    var eps: Float32

def pre_reduce_kernel(args: PreReduceArgs):
    """Accumulate local experts + rmsnorm(dense_out, POST_FFN_NORM_1)."""
    comptime width = simd_width_of[DType.float32]()
    var h = args.hidden
    var expert_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.expert_out_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dst_ptr)
    var dense = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dense_ptr)
    var norm_w = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.norm_w_ptr)
    var normed = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.normed_ptr)
    for i in range(0, h, width):
        var acc = SIMD[DType.float32, width](0)
        for e in range(args.local_count):
            acc += (expert_buf + e * h + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
    var sum_sq = SIMD[DType.float32, width](0)
    for i in range(0, h, width):
        var v = (dense + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(h) + args.eps)
    for i in range(0, h, width):
        var v = (dense + i).load[width=width]().cast[DType.float32]()
        var g = (norm_w + i).load[width=width]().cast[DType.float32]()
        (normed + i).store((v * inv_rms * g).cast[DType.bfloat16]())


@always_inline
def bf16_sum_sq[numel: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = SIMD[DType.float32, width](0)
    var i = 0
    while i + width <= numel:
        var v = (src + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    var out = sum_sq.reduce_add()
    while i < numel:
        var v = Float32(src[i])
        out += v * v
        i += 1
    return out


@fieldwise_init
struct PostReduceArgs(Copyable, ImplicitlyCopyable):
    var moe_out_ptr: Int
    var moe_norm_w_ptr: Int
    var dense_normed_ptr: Int
    var combine_norm_w_ptr: Int
    var x_main_ptr: Int
    var layer_scalar: Float32
    var eps: Float32

def post_reduce_kernel(args: PostReduceArgs):
    """Post-allreduce: norms + combine + residual + scalar."""
    moe_combine[C.HIDDEN](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.moe_out_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.moe_norm_w_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dense_normed_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.combine_norm_w_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.x_main_ptr),
        args.layer_scalar, args.eps)


def comptime_sqrt(x: Float64) -> Float64:
    if x <= Float64(0):
        return Float64(0)
    var g = x
    for _ in range(60):
        g = (g + x / g) * Float64(0.5)
    return g


def concentration_constant[n: Int, num_samples: Int = 10000]() -> Float64:
    var state = UInt64(0xDEADBEEF12345678)
    var total = Float64(0)
    var sqrt_n = comptime_sqrt(Float64(n))
    var rsqrt_n = Float64(1) / sqrt_n

    for _ in range(num_samples):
        var vec = InlineArray[Float64, n](fill=Float64(0))
        var norm_sq = Float64(0)
        for i in range(n):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var u = Float64(Int64(state & 0xFFFFFFFF)) / Float64(0xFFFFFFFF)
            var z = Float64(2) * u - Float64(1)
            vec[i] = z
            norm_sq += z * z

        var inv_norm = Float64(1) / comptime_sqrt(norm_sq)
        for i in range(n):
            vec[i] *= inv_norm

        var stride = 1
        while stride < n:
            var i = 0
            while i < n:
                for j in range(stride):
                    var a = vec[i + j]
                    var b = vec[i + j + stride]
                    vec[i + j] = a + b
                    vec[i + j + stride] = a - b
                i += 2 * stride
            stride *= 2
        for i in range(n):
            vec[i] *= rsqrt_n

        var max_abs = Float64(0)
        for i in range(n):
            var a = vec[i].__abs__()
            if a > max_abs:
                max_abs = a
        total += sqrt_n * max_abs

    return total / Float64(num_samples)


def frobenius_from_quantized[
    W: Encoding & Shaped & Placed & Named,
    S: Encoding & Shaped & Placed & Named,
](
    arena_base: Int, layer_base: Int,
) -> Float64:
    """Recover ||W'||_F^2 from quantized i8 rows and their per-row scales."""
    var w_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + W.OFFSET)
    var s_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + S.OFFSET)
    var frob_sq = Float64(0)
    for n in range(W.ROWS):
        var row_sq = Float64(0)
        for k in range(W.COLS):
            var v = Float64(w_ptr[n * W.COLS + k])
            row_sq += v * v
        var s = Float64(s_ptr[n])
        frob_sq += s * s * row_sq
    return frob_sq


def derive_sliding_v_scales[tp: Int](
    arena_bases: List[Int],
) -> InlineArray[Float32, C.NUM_SLIDING_LAYERS]:
    """V is /rms-normed per head before FWHT + cache write.

    After /rms each head has ||x||=sqrt(d_k), so max|FWHT(x)| ≈ C(d_k).
    Weight norm is irrelevant — /rms normalizes it away.
    """
    var cn = Float32(concentration_constant[C.HEAD_DIM_SLIDING]())
    var scales = InlineArray[Float32, C.NUM_SLIDING_LAYERS](fill=cn)
    return scales^


def derive_full_v_scale() -> Float32:
    """Full attention V is also /rms-normed per head. Same logic: S_V = C(d_k)."""
    return Float32(concentration_constant[C.HEAD_DIM_FULL]())


# =============================================================================
# Load-time: column sums + VNNI packing
# =============================================================================


def colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int, rows: Int, cols: Int):
    """Compute per-row column sum: colsum[n] = Σ_k i8_weight[n, k]."""
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + weight_off)
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum_off)
    for n in range(rows):
        var acc = Int(0)
        for k in range(cols):
            acc += Int(wp[n * cols + k])
        cp[n] = Float32(acc)


def block_colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int,
    rows: Int, cols: Int, block_cols: Int):
    """Per-block column sums in [num_blocks, N] layout for SIMD-friendly access.

    colsum[blk * rows + n] = Σ_{k in block} W_i8[n, k].
    Transposed so consecutive output rows are contiguous per block.
    """
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + weight_off)
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum_off)
    var num_blocks = cols // block_cols
    for n in range(rows):
        for blk in range(num_blocks):
            var acc = Int(0)
            var k0 = blk * block_cols
            for k in range(block_cols):
                acc += Int(wp[n * cols + k0 + k])
            cp[blk * rows + n] = Float32(acc)


def pack_at(arena_base: Int, weight_off: Int, rows: Int, cols: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """VNNI-pack weight in-place using scratch buffer."""
    from kernels.vnni import pack_vnni
    var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=arena_base + weight_off)
    memcpy(dest=scratch, src=src, count=rows * cols)
    pack_vnni(scratch, src, rows, cols)


def init_sliding_layer(arena_base: Int, layer_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    comptime SL = SlidingLayer[1]
    # Column sums
    colsum_at(arena_base, layer_base + SL.Q_PROJ.OFFSET, layer_base + SL.QKV_COLSUM_OFF,
        SL.QKV_N, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + SL.O_PROJ.OFFSET, layer_base + SL.O_COLSUM_OFF,
        C.HIDDEN, C.Q_DIM_SLIDING, C.HEAD_DIM_SLIDING)
    colsum_at(arena_base, layer_base + SL.GATE_PROJ.OFFSET, layer_base + SL.GU_COLSUM_OFF,
        C.INTERMEDIATE * 2, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + SL.DOWN_PROJ.OFFSET, layer_base + SL.DOWN_COLSUM_OFF,
        C.HIDDEN, C.INTERMEDIATE, C.FWHT_BLK)
    colsum_at(arena_base, layer_base + SL.ROUTER_PROJ.OFFSET, layer_base + SL.ROUTER_COLSUM_OFF,
        C.NUM_EXPERTS, C.HIDDEN)
    colsum_at(arena_base, layer_base + SL.EXPERTS_GATE_UP.OFFSET, layer_base + SL.EXPERTS_GU_COLSUM_OFF,
        C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN)
    for e in range(C.NUM_EXPERTS):
        block_colsum_at(arena_base,
            layer_base + SL.EXPERTS_DOWN.OFFSET + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            layer_base + SL.EXPERTS_DOWN_COLSUM_OFF + e * C.HIDDEN * C.MOE_NUM_BLOCKS * 4,
            C.HIDDEN, C.MOE_INTERMEDIATE, C.FWHT_BLK)
    # VNNI packing (colsums must be computed before packing reorders the data)
    pack_at(arena_base, layer_base + SL.Q_PROJ.OFFSET, SL.QKV_N, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + SL.O_PROJ.OFFSET, C.HIDDEN, C.Q_DIM_SLIDING, scratch)
    pack_at(arena_base, layer_base + SL.GATE_PROJ.OFFSET, C.INTERMEDIATE * 2, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + SL.DOWN_PROJ.OFFSET, C.HIDDEN, C.INTERMEDIATE, scratch)
    pack_at(arena_base, layer_base + SL.ROUTER_PROJ.OFFSET, C.NUM_EXPERTS, C.HIDDEN, scratch)
    for e in range(C.NUM_EXPERTS):
        pack_at(arena_base, layer_base + SL.EXPERTS_GATE_UP.OFFSET + e * C.MOE_GATE_UP_FUSED * C.HIDDEN, C.MOE_GATE_UP_FUSED, C.HIDDEN, scratch)
        pack_at(arena_base, layer_base + SL.EXPERTS_DOWN.OFFSET + e * C.HIDDEN * C.MOE_INTERMEDIATE, C.HIDDEN, C.MOE_INTERMEDIATE, scratch)


def init_full_layer(arena_base: Int, layer_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    comptime FL = FullLayer[1]
    # Column sums
    colsum_at(arena_base, layer_base + FL.Q_PROJ.OFFSET, layer_base + FL.QK_COLSUM_OFF,
        FL.QK_N, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + FL.O_PROJ.OFFSET, layer_base + FL.O_COLSUM_OFF,
        C.HIDDEN, C.Q_DIM_FULL, C.HEAD_DIM_FULL)
    colsum_at(arena_base, layer_base + FL.GATE_PROJ.OFFSET, layer_base + FL.GU_COLSUM_OFF,
        C.INTERMEDIATE * 2, C.HIDDEN)
    block_colsum_at(arena_base, layer_base + FL.DOWN_PROJ.OFFSET, layer_base + FL.DOWN_COLSUM_OFF,
        C.HIDDEN, C.INTERMEDIATE, C.FWHT_BLK)
    colsum_at(arena_base, layer_base + FL.ROUTER_PROJ.OFFSET, layer_base + FL.ROUTER_COLSUM_OFF,
        C.NUM_EXPERTS, C.HIDDEN)
    colsum_at(arena_base, layer_base + FL.EXPERTS_GATE_UP.OFFSET, layer_base + FL.EXPERTS_GU_COLSUM_OFF,
        C.NUM_EXPERTS * C.MOE_GATE_UP_FUSED, C.HIDDEN)
    for e in range(C.NUM_EXPERTS):
        block_colsum_at(arena_base,
            layer_base + FL.EXPERTS_DOWN.OFFSET + e * C.HIDDEN * C.MOE_INTERMEDIATE,
            layer_base + FL.EXPERTS_DOWN_COLSUM_OFF + e * C.HIDDEN * C.MOE_NUM_BLOCKS * 4,
            C.HIDDEN, C.MOE_INTERMEDIATE, C.FWHT_BLK)
    # VNNI packing
    pack_at(arena_base, layer_base + FL.Q_PROJ.OFFSET, FL.QK_N, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + FL.O_PROJ.OFFSET, C.HIDDEN, C.Q_DIM_FULL, scratch)
    pack_at(arena_base, layer_base + FL.GATE_PROJ.OFFSET, C.INTERMEDIATE * 2, C.HIDDEN, scratch)
    pack_at(arena_base, layer_base + FL.DOWN_PROJ.OFFSET, C.HIDDEN, C.INTERMEDIATE, scratch)
    pack_at(arena_base, layer_base + FL.ROUTER_PROJ.OFFSET, C.NUM_EXPERTS, C.HIDDEN, scratch)
    for e in range(C.NUM_EXPERTS):
        pack_at(arena_base, layer_base + FL.EXPERTS_GATE_UP.OFFSET + e * C.MOE_GATE_UP_FUSED * C.HIDDEN, C.MOE_GATE_UP_FUSED, C.HIDDEN, scratch)
        pack_at(arena_base, layer_base + FL.EXPERTS_DOWN.OFFSET + e * C.HIDDEN * C.MOE_INTERMEDIATE, C.HIDDEN, C.MOE_INTERMEDIATE, scratch)


# =============================================================================
# Model struct
# =============================================================================


struct Gemma4ButterQuant[tp: Int](Movable):
    comptime M = Gemma4Model[Self.tp]
    comptime SL = SlidingLayer[1]
    comptime FL = FullLayer[1]
    comptime MAX_PACK_BYTES = Self.FL.QK_N * C.HIDDEN

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var main_pools: HeapMoveArray[BurstPool[]]
    var expert_pools: HeapMoveArray[BurstPool[]]
    var dense_pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var sliding_v_scales: InlineArray[Float32, C.NUM_SLIDING_LAYERS]
    var full_v_scale: Float32

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var mp: HeapMoveArray[BurstPool[]],
        var ep: HeapMoveArray[BurstPool[]],
        var dp: HeapMoveArray[BurstPool[]],
        var sc: ScratchPool,
        bases: InlineArray[Int, Self.tp],
        sliding_v_scales: InlineArray[Float32, C.NUM_SLIDING_LAYERS],
        full_v_scale: Float32,
    ):
        self.arenas = arenas^
        self.main_pools = mp^
        self.expert_pools = ep^
        self.dense_pools = dp^
        self.scratch = sc^
        self.bases = bases
        self.sliding_v_scales = sliding_v_scales
        self.full_v_scale = full_v_scale

    def make_pool_ptrs(self, pools: HeapMoveArray[BurstPool[]]) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
        for r in range(Self.tp):
            ptrs[r] = UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=pools[r])))
        return ptrs^

    def ranks(self) -> Ranks[Self.tp]:
        return Ranks[Self.tp](self.bases)
    def main_ptrs(self) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        return self.make_pool_ptrs(self.main_pools)
    def expert_ptrs(self) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        return self.make_pool_ptrs(self.expert_pools)
    def dense_ptrs(self) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]:
        return self.make_pool_ptrs(self.dense_pools)

    # =========================================================================
    # Load + init
    # =========================================================================

    @staticmethod
    def discover_shards(dir_path: Path) -> List[Path]:
        var shards = List[Path]()
        try:
            for entry in dir_path.listdir():
                var name = String(entry)
                if name.endswith(".safetensors"):
                    shards.append(dir_path / name)
        except:
            pass
        for i in range(len(shards)):
            for j in range(i + 1, len(shards)):
                if String(shards[j]) < String(shards[i]):
                    var tmp = shards[i]
                    shards[i] = shards[j]
                    shards[j] = tmp
        return shards^

    def init_state(mut self):
        var pack_scratch = alloc[UInt8](Self.MAX_PACK_BYTES)
        for rank in range(Self.tp):
            var rv = self.ranks().view(rank)
            init_sliding_rope_tables(rv.sliding_cos(), rv.sliding_sin())
            init_full_rope_tables(rv.full_cos(), rv.full_sin())

            var base = Int(self.arenas[rank].base)
            var sliding_idx = 0
            var full_idx = 0
            for i in range(C.NUM_LAYERS):
                if (i + 1) % 6 == 0:
                    init_full_layer(base, Self.M.FULL_OFF + full_idx * Self.M.FULL_STRIDE, pack_scratch)
                    full_idx += 1
                else:
                    init_sliding_layer(base, Self.M.SLIDING_OFF + sliding_idx * Self.M.SLIDING_STRIDE, pack_scratch)
                    sliding_idx += 1

            # Bake 1/sqrt(hidden) into router weight scales (one-time)
            comptime inv_sqrt_h = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
            sliding_idx = 0
            full_idx = 0
            for i in range(C.NUM_LAYERS):
                var sc_ptr: UnsafePointer[Float32, MutAnyOrigin]
                if (i + 1) % 6 == 0:
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=base + Self.M.FULL_OFF + full_idx * Self.M.FULL_STRIDE + Self.FL.ROUTER_PROJ_SC.OFFSET)
                    full_idx += 1
                else:
                    sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
                        unsafe_from_address=base + Self.M.SLIDING_OFF + sliding_idx * Self.M.SLIDING_STRIDE + Self.SL.ROUTER_PROJ_SC.OFFSET)
                    sliding_idx += 1
                for n in range(C.NUM_EXPERTS):
                    sc_ptr[n] *= inv_sqrt_h

        pack_scratch.free()

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        var shards = Self.discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)
        comptime host_rank = 0

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == host_rank else Self.M.arena_bytes()
            print("rank", rank, "node", topo[rank], "allocating", size // (1024 * 1024), "MB")
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("arena allocation failed for rank", rank)
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_weights[Self.M](shards, arena_bases, host_index=host_rank)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        # Derive fixed V scales from the unpacked quantized weights before VNNI packing.
        var sliding_v_scales = derive_sliding_v_scales[Self.tp](arena_bases)
        var full_v_scale = derive_full_v_scale()

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        var main_pools = HeapMoveArray[BurstPool[]](Self.tp)
        var expert_pools = HeapMoveArray[BurstPool[]](Self.tp)
        var dense_pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            main_pools.push(BurstPool[].for_numa_node(numa, topo[rank]))
            expert_pools.push(BurstPool[].for_numa_node(numa, topo[rank]))
            dense_pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var bases = InlineArray[Int, Self.tp](fill=0)
        for rank in range(Self.tp):
            bases[rank] = Int(arenas[rank].base)

        var scratch = ScratchPool(Self.M.SCRATCH_CAPACITY)
        var model = Self(
            arenas^, main_pools^, expert_pools^, dense_pools^, scratch^, bases,
            sliding_v_scales, full_v_scale)
        model.init_state()
        print("v scales: sliding[0]=", model.sliding_v_scales[0], " full=", model.full_v_scale)
        print("state initialized")
        return model^

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.ranks().view(0).scratch_base())

    # =========================================================================
    # Forward — decode (seq_len=1), sliding layers only for now
    # =========================================================================


    def forward_decode(mut self, tokens_ptr: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime M = Self.M
        comptime SL = Self.SL
        comptime FL = Self.FL
        comptime EPS = Float32(C.RMS_NORM_EPS)
        comptime seq_len = 1

        var rnks = self.ranks()
        var host = rnks.view(0)
        var mp = self.main_ptrs()
        var ep = self.expert_ptrs()
        var dp = self.dense_ptrs()

        # --- Embed ---
        embed_lookup_scaled(
            host.host_weight[M.EMBED](), tokens_ptr,
            host.x_main(seq_len), Float32(C.EMBED_SCALE),
            self.main_pools[0]).join()
        ring_broadcast[M.X_MAIN, Self.tp](
            host.x_main(seq_len).ptr, rnks.x_main_ptrs(seq_len), seq_len, mp)

        var act_scale_lease = self.scratch.borrow[Float32, 1]()
        var post_blk_scale_lease = self.scratch.borrow[Float32, C.DENSE_NUM_BLOCKS]()

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
                        rv.x_main(seq_len).ptr, slb + SL.INPUT_NORM.OFFSET,
                        rv.scratch_addr(attn_i8_lease),
                        rv.scratch_addr(attn_work_lease), rv.scratch_addr(attn_scale_lease),
                        seq_len, pool)
                rnks.parallel[do_attn_quantize](mp)

                var qkv_lease = self.scratch.borrow[Scalar[DType.bfloat16], SL.QKV_N // Self.tp]()

                @parameter
                def do_qkv_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var slb = rv.sliding_layer_base(sliding_idx)
                    return int8_gemv[SL.QKV_N // Self.tp, C.HIDDEN](
                        rv.scratch_addr(attn_i8_lease),
                        slb + SL.Q_PROJ.OFFSET,
                        slb + SL.QKV_COLSUM_OFF,
                        slb + SL.Q_PROJ_SC.OFFSET,
                        rv.scratch_addr(qkv_lease),
                        seq_len, rv.scratch_addr(attn_scale_lease), pool)
                rnks.parallel[do_qkv_gemv](mp)

                var v_sum_sq = Float32(0)
                for r in range(Self.tp):
                    var rv = rnks.view(r)
                    var qkv_base = rv.scratch_addr(qkv_lease)
                    var v_base = qkv_base + (SL.Q_DIM_LOCAL + SL.KV_DIM_LOCAL) * 2
                    v_sum_sq += bf16_sum_sq[SL.KV_DIM_LOCAL](
                        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=v_base))
                var v_inv_rms = 1.0 / sqrt[DType.float32, 1](
                    v_sum_sq / Float32(C.KV_DIM_SLIDING) + EPS)

                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], SL.Q_DIM_LOCAL]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, SL.Q_DIM_LOCAL // C.HEAD_DIM_SLIDING]()

                @parameter
                def do_sliding_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
                    comptime NKV = C.NUM_KV_HEADS_SLIDING // Self.tp
                    var qkv_base = rv.scratch_addr(qkv_lease)
                    var q_base = qkv_base
                    var k_base = qkv_base + SL.Q_DIM_LOCAL * 2
                    var v_base = k_base + SL.KV_DIM_LOCAL * 2
                    var cache_pos = pos % C.SLIDING_WINDOW
                    var context_len = min(pos + 1, C.SLIDING_WINDOW)
                    var slb = rv.sliding_layer_base(sliding_idx)
                    var q_norm_addr = slb + SL.Q_NORM.OFFSET
                    var k_norm_addr = slb + SL.K_NORM.OFFSET
                    var jobs = InlineArray[SlidingAttnGroupArgs, 8](
                        fill=SlidingAttnGroupArgs(0,0,0,0,0,0,0,0,0,0,0,0,0,Float32(0),Float32(0)))
                    for g in range(NKV):
                        jobs[g] = SlidingAttnGroupArgs(
                            q_base + g * HPG * C.HEAD_DIM_SLIDING * 2,
                            k_base + g * C.HEAD_DIM_SLIDING * 2,
                            v_base + g * C.HEAD_DIM_SLIDING * 2,
                            q_norm_addr, k_norm_addr,
                            rv.sliding_cos_row(pos), rv.sliding_sin_row(pos),
                            rv.sliding_cache_base(sliding_idx), g,
                            cache_pos, context_len,
                            rv.scratch_addr(attn_qi_lease) + g * HPG * C.HEAD_DIM_SLIDING,
                            rv.scratch_addr(attn_head_sc_lease) + g * HPG * 4,
                            v_inv_rms,
                            EPS)
                    pool.dispatch[SlidingAttnGroupArgs,
                        sliding_attn_group_kernel[C.HEAD_DIM_SLIDING, HPG,
                            C.SLIDING_WINDOW, C.NUM_KV_HEADS_SLIDING // Self.tp, C.NUM_HEADS // Self.tp]](
                        UnsafePointer(to=jobs[0]), NKV)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                rnks.parallel[do_sliding_attn](mp)

                @parameter
                def do_o_proj[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var slb = rv.sliding_layer_base(sliding_idx)
                    return int8_gemv_blocked[C.HIDDEN, SL.Q_DIM_LOCAL, C.HEAD_DIM_SLIDING](
                        rv.scratch_addr(attn_qi_lease), slb + SL.O_PROJ.OFFSET,
                        rv.scratch_addr(attn_head_sc_lease), slb + SL.O_PROJ_SC.OFFSET,
                        slb + SL.O_COLSUM_OFF,
                        rv.x_residual(seq_len).ptr, seq_len, pool)
                rnks.parallel[do_o_proj](mp)
                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                qkv_lease^.release()
                attn_scale_lease^.release()
                attn_work_lease^.release()
                attn_i8_lease^.release()
            else:
                # Full attention — int8 Q/K projection, quantized cache/O/FFN unchanged
                comptime FL = FullLayer[1]
                comptime FULL_HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
                comptime FULL_Q_LOCAL = C.Q_DIM_FULL // Self.tp
                comptime FULL_NKV = C.NUM_KV_HEADS_FULL
                comptime ROPE_DIMS_FULL = 128
                var qk_lease = self.scratch.borrow[Scalar[DType.bfloat16], FL.QK_N // Self.tp]()

                var full_attn_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
                var full_attn_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()
                var full_attn_scale_lease = self.scratch.borrow[Float32, 1]()

                @parameter
                def do_full_attn_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return rmsnorm_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                        rv.x_main(seq_len).ptr, flb + FL.INPUT_NORM.OFFSET,
                        rv.scratch_addr(full_attn_i8_lease),
                        rv.scratch_addr(full_attn_work_lease), rv.scratch_addr(full_attn_scale_lease),
                        seq_len, pool)
                rnks.parallel[do_full_attn_quantize](mp)

                @parameter
                def do_full_qk_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return int8_gemv[FL.QK_N // Self.tp, C.HIDDEN](
                        rv.scratch_addr(full_attn_i8_lease),
                        flb + FL.Q_PROJ.OFFSET,
                        flb + FL.QK_COLSUM_OFF,
                        flb + FL.Q_PROJ_SC.OFFSET,
                        rv.scratch_addr(qk_lease),
                        seq_len, rv.scratch_addr(full_attn_scale_lease), pool)
                rnks.parallel[do_full_qk_gemv](mp)
                full_attn_scale_lease^.release()
                full_attn_work_lease^.release()
                full_attn_i8_lease^.release()

                var full_v_sum_sq = Float32(0)
                for r in range(Self.tp):
                    var rv = rnks.view(r)
                    var qk_base = rv.scratch_addr(qk_lease)
                    var k_base = qk_base + FULL_Q_LOCAL * 2
                    full_v_sum_sq += bf16_sum_sq[FL.KV_DIM_LOCAL](
                        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=k_base))
                var full_v_inv_rms = 1.0 / sqrt[DType.float32, 1](
                    full_v_sum_sq / Float32(C.KV_DIM_FULL) + EPS)

                # Scoring + V-agg: 2 workers (one per KV group, GQA 8:1)
                var attn_qi_lease = self.scratch.borrow[Scalar[DType.int8], FULL_Q_LOCAL]()
                var attn_head_sc_lease = self.scratch.borrow[Float32, FULL_Q_LOCAL // C.HEAD_DIM_FULL]()

                @parameter
                def do_full_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var qk_base = rv.scratch_addr(qk_lease)
                    var q_base = qk_base
                    var k_base = qk_base + FULL_Q_LOCAL * 2
                    var cos_addr = rv.state_base() + Self.M.FULL_COS_OFF + pos * Self.M.FULL_ROPE_HALF * size_of[Float32]()
                    var sin_addr = rv.state_base() + Self.M.FULL_SIN_OFF + pos * Self.M.FULL_ROPE_HALF * size_of[Float32]()

                    var flb = rv.full_layer_base(full_idx)
                    var q_norm_addr = flb + FL.Q_NORM.OFFSET
                    var k_norm_addr = flb + FL.K_NORM.OFFSET
                    var jobs = InlineArray[FullAttnGroupArgs, 4](
                        fill=FullAttnGroupArgs(0,0,0,0,0,0,0,0,0,0,0,0,Float32(0),Float32(0)))
                    for g in range(FULL_NKV):
                        jobs[g] = FullAttnGroupArgs(
                            q_base + g * FULL_HPG * C.HEAD_DIM_FULL * 2,
                            k_base + g * C.HEAD_DIM_FULL * 2,
                            q_norm_addr, k_norm_addr,
                            cos_addr, sin_addr,
                            rv.full_cache_base(full_idx), g,
                            pos, pos + 1,
                            rv.scratch_addr(attn_qi_lease) + g * FULL_HPG * C.HEAD_DIM_FULL,
                            rv.scratch_addr(attn_head_sc_lease) + g * FULL_HPG * 4,
                            full_v_inv_rms,
                            EPS)
                    pool.dispatch[FullAttnGroupArgs,
                        full_attn_group_kernel[C.HEAD_DIM_FULL, ROPE_DIMS_FULL, FULL_HPG,
                            C.MAX_SEQ_LEN, C.NUM_KV_HEADS_FULL, C.NUM_HEADS // Self.tp]](
                        UnsafePointer(to=jobs[0]), FULL_NKV)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                rnks.parallel[do_full_attn](mp)

                # O projection
                @parameter
                def do_full_o_proj[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                    var flb = rv.full_layer_base(full_idx)
                    return int8_gemv_blocked[C.HIDDEN, FULL_Q_LOCAL, C.HEAD_DIM_FULL](
                        rv.scratch_addr(attn_qi_lease), flb + FL.O_PROJ.OFFSET,
                        rv.scratch_addr(attn_head_sc_lease), flb + FL.O_PROJ_SC.OFFSET,
                        flb + FL.O_COLSUM_OFF,
                        rv.x_residual(seq_len).ptr, seq_len, pool)
                rnks.parallel[do_full_o_proj](mp)
                attn_head_sc_lease^.release()
                attn_qi_lease^.release()
                qk_lease^.release()

            # Allreduce + post-attn norm (both layer types)
            ring_allreduce[M.X_RESIDUAL, Self.tp](
                rnks.x_residual_ptrs(seq_len), seq_len, mp)

            @parameter
            def do_post_attn_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var nlb = rv.sliding_layer_base(sliding_idx) if not is_full else rv.full_layer_base(full_idx)
                var post_attn_norm_off = FL.POST_ATTN_NORM.OFFSET if is_full else SL.POST_ATTN_NORM.OFFSET
                var args = InlineArray[PostAttnNormArgs, 1](fill=PostAttnNormArgs(
                    rv.x_residual(seq_len).ptr,
                    nlb + post_attn_norm_off,
                    rv.x_main(seq_len).ptr, EPS))
                pool.dispatch[PostAttnNormArgs, post_attn_norm_kernel](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            rnks.parallel[do_post_attn_norm](mp)

            # =============================================================
            # FFN BLOCK (identical for sliding and full layers)
            # =============================================================

            var act_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.HIDDEN]()
            var act_work_lease = self.scratch.borrow[Float32, C.HIDDEN]()

            @parameter
            def do_router_quantize[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var router_scale_off = FL.ROUTER_SCALE.OFFSET if is_full else SL.ROUTER_SCALE.OFFSET
                return rmsnorm_gamma_fwht_quantize[C.HIDDEN, C.FWHT_BLK_HIDDEN](
                    rv.x_main(seq_len).ptr, lb + router_scale_off,
                    rv.scratch_addr(act_i8_lease),
                    rv.scratch_addr(act_work_lease), rv.scratch_addr(act_scale_lease),
                    seq_len, pool)
            rnks.parallel[do_router_quantize](mp)

            var router_logits_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.NUM_EXPERTS]()
            var routing_lease = self.scratch.borrow[Gemma4TopKResult[C.TOP_K], 1]()

            @parameter
            def do_router_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var router_proj_off = FL.ROUTER_PROJ.OFFSET if is_full else SL.ROUTER_PROJ.OFFSET
                var router_colsum_off = FL.ROUTER_COLSUM_OFF if is_full else SL.ROUTER_COLSUM_OFF
                var router_proj_sc_off = FL.ROUTER_PROJ_SC.OFFSET if is_full else SL.ROUTER_PROJ_SC.OFFSET
                return int8_gemv[C.NUM_EXPERTS, C.HIDDEN](
                    rv.scratch_addr(act_i8_lease),
                    lb + router_proj_off,
                    lb + router_colsum_off,
                    lb + router_proj_sc_off,
                    rv.scratch_addr(router_logits_lease),
                    seq_len, rv.scratch_addr(act_scale_lease), pool)
            rnks.parallel[do_router_gemv](mp)

            @parameter
            def do_router_topk[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var router_pes_off = FL.ROUTER_PES.OFFSET if is_full else SL.ROUTER_PES.OFFSET
                var args = InlineArray[RouterTopkArgs, 1](fill=RouterTopkArgs(
                    rv.scratch_addr(router_logits_lease),
                    lb + router_pes_off,
                    rv.scratch_addr(routing_lease)))
                pool.dispatch[RouterTopkArgs, router_topk_kernel[C.NUM_EXPERTS, C.TOP_K]](
                    UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            rnks.parallel[do_router_topk](mp)

            var expert_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()
            var local_count_lease = self.scratch.borrow[Int32, 1]()

            @parameter
            def do_expert_dispatch[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var routing = UnsafePointer[Gemma4TopKResult[C.TOP_K], MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(routing_lease))[]
                var pre_ffn_norm_2_off = FL.PRE_FFN_NORM_2.OFFSET if is_full else SL.PRE_FFN_NORM_2.OFFSET
                var experts_gate_up_off = FL.EXPERTS_GATE_UP.OFFSET if is_full else SL.EXPERTS_GATE_UP.OFFSET
                var experts_gate_up_sc_off = FL.EXPERTS_GATE_UP_SC.OFFSET if is_full else SL.EXPERTS_GATE_UP_SC.OFFSET
                var experts_gu_colsum_off = FL.EXPERTS_GU_COLSUM_OFF if is_full else SL.EXPERTS_GU_COLSUM_OFF
                var experts_down_off = FL.EXPERTS_DOWN.OFFSET if is_full else SL.EXPERTS_DOWN.OFFSET
                var experts_down_sc_off = FL.EXPERTS_DOWN_SC.OFFSET if is_full else SL.EXPERTS_DOWN_SC.OFFSET
                var experts_down_colsum_off = FL.EXPERTS_DOWN_COLSUM_OFF if is_full else SL.EXPERTS_DOWN_COLSUM_OFF
                var lc = gemma4_moe_dispatch_local[
                    C.NUM_EXPERTS, C.TOP_K, C.MOE_INTERMEDIATE, C.HIDDEN,
                    C.FWHT_BLK, C.FWHT_BLK_HIDDEN, Self.tp](
                    rv.x_main(seq_len).ptr, lb + pre_ffn_norm_2_off, routing,
                    lb + experts_gate_up_off,
                    C.MOE_GATE_UP_FUSED * C.HIDDEN,
                    lb + experts_gate_up_sc_off,
                    C.MOE_GATE_UP_FUSED * 4,
                    lb + experts_gu_colsum_off,
                    C.MOE_GATE_UP_FUSED * 4,
                    lb + experts_down_off,
                    C.HIDDEN * C.MOE_INTERMEDIATE,
                    lb + experts_down_sc_off,
                    C.HIDDEN * 4,
                    lb + experts_down_colsum_off,
                    C.HIDDEN * C.MOE_NUM_BLOCKS * 4,
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                        unsafe_from_address=rv.scratch_addr(expert_out_lease)),
                    rank, pool)
                UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(local_count_lease))[] = Int32(lc)
                return PoolFence[BurstPool[]].completed()
            rnks.parallel[do_expert_dispatch](ep)

            var dense_post_i8_lease = self.scratch.borrow[Scalar[DType.int8], C.INTERMEDIATE]()

            @parameter
            def do_dense_phase1[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var pre_ffn_norm_off = FL.PRE_FFN_NORM.OFFSET if is_full else SL.PRE_FFN_NORM.OFFSET
                var gate_proj_off = FL.GATE_PROJ.OFFSET if is_full else SL.GATE_PROJ.OFFSET
                var gate_proj_sc_off = FL.GATE_PROJ_SC.OFFSET if is_full else SL.GATE_PROJ_SC.OFFSET
                var gu_colsum_off = FL.GU_COLSUM_OFF if is_full else SL.GU_COLSUM_OFF
                return fused_gu_gelu_tanh[C.INTERMEDIATE, C.HIDDEN, C.FWHT_BLK, C.FWHT_BLK_HIDDEN](
                    rv.x_main(seq_len).ptr, lb + pre_ffn_norm_off,
                    lb + gate_proj_off,
                    lb + gate_proj_sc_off,
                    lb + gu_colsum_off,
                    rv.scratch_addr(dense_post_i8_lease), rv.scratch_addr(post_blk_scale_lease),
                    seq_len, pool)
            rnks.parallel[do_dense_phase1](dp)

            @parameter
            def do_dense_join1[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                pool.join()
                return PoolFence[BurstPool[]].completed()
            rnks.parallel[do_dense_join1](dp)

            var dense_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_dense_phase2[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var down_proj_off = FL.DOWN_PROJ.OFFSET if is_full else SL.DOWN_PROJ.OFFSET
                var down_proj_sc_off = FL.DOWN_PROJ_SC.OFFSET if is_full else SL.DOWN_PROJ_SC.OFFSET
                var down_colsum_off = FL.DOWN_COLSUM_OFF if is_full else SL.DOWN_COLSUM_OFF
                return int8_gemv_blocked[C.HIDDEN, C.INTERMEDIATE, C.FWHT_BLK](
                    rv.scratch_addr(dense_post_i8_lease),
                    lb + down_proj_off,
                    rv.scratch_addr(post_blk_scale_lease),
                    lb + down_proj_sc_off,
                    lb + down_colsum_off,
                    rv.scratch_addr(dense_out_lease), seq_len, pool)
            rnks.parallel[do_dense_phase2](dp)

            @parameter
            def do_join_dense[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                pool.join()
                return PoolFence[BurstPool[]].completed()
            rnks.parallel[do_join_dense](dp)
            @parameter
            def do_join_expert[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                pool.join()
                return PoolFence[BurstPool[]].completed()
            rnks.parallel[do_join_expert](ep)

            var dense_normed_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.HIDDEN]()

            @parameter
            def do_pre_reduce[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var post_ffn_norm_1_off = FL.POST_FFN_NORM_1.OFFSET if is_full else SL.POST_FFN_NORM_1.OFFSET
                var lc = Int(UnsafePointer[Int32, MutAnyOrigin](
                    unsafe_from_address=rv.scratch_addr(local_count_lease))[] )
                var args = InlineArray[PreReduceArgs, 1](fill=PreReduceArgs(
                    rv.scratch_addr(expert_out_lease), lc,
                    rv.x_residual(seq_len).ptr, rv.scratch_addr(dense_out_lease),
                    lb + post_ffn_norm_1_off,
                    rv.scratch_addr(dense_normed_lease), C.HIDDEN, EPS))
                pool.dispatch[PreReduceArgs, pre_reduce_kernel](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            rnks.parallel[do_pre_reduce](mp)

            ring_allreduce[M.X_RESIDUAL, Self.tp](
                rnks.x_residual_ptrs(seq_len), seq_len, mp)

            @parameter
            def do_post_reduce[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var lb = rv.full_layer_base(full_idx) if is_full else rv.sliding_layer_base(sliding_idx)
                var layer_scalar_off = FL.LAYER_SCALAR.OFFSET if is_full else SL.LAYER_SCALAR.OFFSET
                var post_ffn_norm_2_rt_off = FL.POST_FFN_NORM_2_RT.OFFSET if is_full else SL.POST_FFN_NORM_2_RT.OFFSET
                var post_ffn_norm_off = FL.POST_FFN_NORM.OFFSET if is_full else SL.POST_FFN_NORM.OFFSET
                var ls = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=lb + layer_scalar_off)
                var args = InlineArray[PostReduceArgs, 1](fill=PostReduceArgs(
                    rv.x_residual(seq_len).ptr,
                    lb + post_ffn_norm_2_rt_off,
                    rv.scratch_addr(dense_normed_lease),
                    lb + post_ffn_norm_off,
                    rv.x_main(seq_len).ptr, Float32(ls[]), EPS))
                pool.dispatch[PostReduceArgs, post_reduce_kernel](UnsafePointer(to=args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            rnks.parallel[do_post_reduce](mp)
            dense_normed_lease^.release()
            dense_out_lease^.release()
            dense_post_i8_lease^.release()
            local_count_lease^.release()
            expert_out_lease^.release()
            routing_lease^.release()
            router_logits_lease^.release()
            act_work_lease^.release()
            act_i8_lease^.release()

            if is_full:
                full_idx += 1
            else:
                sliding_idx += 1

        post_blk_scale_lease^.release()
        act_scale_lease^.release()

        # --- Final norm + LM head + softcap ---
        var last_hidden = DynView[M.X_MAIN](host.x_main(seq_len).ptr, 1)
        rmsnorm(last_hidden, host.host_weight[M.FINAL_NORM](), last_hidden,
            self.main_pools[0]).join()
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = host.scratch_view[M.LOGITS](logit_lease, 1)
        float_gemv(last_hidden, host.host_weight[M.EMBED](), logit_view,
            self.main_pools[0]).join()
        logit_softcap(logit_view)
        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^)


def main():
    print("gemma_4_moe_butterquant_tp: structural declarations")
    print("  sliding stride:  " + String(SlidingLayer[1].STRIDE) + " bytes")
    print("  full stride:     " + String(FullLayer[1].STRIDE) + " bytes")
    print("  distributed:     " + String(Gemma4Model[1].DISTRIBUTED_BYTES // (1024 * 1024)) + " MB")
    print("  state per rank:  " + String(Gemma4Model[1].STATE_BYTES // (1024 * 1024)) + " MB")
    print("  scratch:         " + String(Gemma4Model[1].SCRATCH_CAPACITY // 1024) + " KB")
