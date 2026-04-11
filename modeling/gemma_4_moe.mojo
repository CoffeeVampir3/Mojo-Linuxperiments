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
from notstdcollections import HeapMoveArray
from threading import BurstPool
from experimental.linear_borrow_pool import ScratchPool, ScratchLease

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32,
    RowShard, ColShard, Replicated, HOST_RANK,
    Slot, PlacedSlot, Bound, DynView, CacheView,
    byte_count, bind, WeightIterable, next_offset,
    DEFAULT_ALIGNMENT, LogitsView,
)
from modeling.loader import load_weights

from kernels.kernel_ops import (
    gemm, rmsnorm, elem_add, kv_cache_write,
    gemv_kernel, GemmArgs,
    PoolFence, BF16Ptr,
)
from kernels.kv_rotors import rope
from threading.threading_shared import ptr as tptr

from experimental_gemma.activations import gelu_tanh_mul
from experimental_gemma.norms import rmsnorm_no_scale, rmsnorm_per_head
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables, apply_full_rope
from experimental_gemma.router import softmax_topk_renorm
from experimental_gemma.moe import gemma4_moe_dispatch
from experimental_gemma.attention import local_attention, global_attention
from experimental_gemma.ops import scaled_add, embed_lookup_scaled, logit_softcap, elem_scale


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
# Sliding attention layer (25 of 30)
# =============================================================================


struct SlidingLayer[tp: Int]:
    # --- Attention (sliding geometry) ---
    comptime Q_PROJ = PlacedSlot[BF16, Replicated, C.Q_DIM_SLIDING,  C.HIDDEN,         Self.tp, 0,                             "self_attn.q_proj.weight"]
    comptime K_PROJ = PlacedSlot[BF16, Replicated, C.KV_DIM_SLIDING, C.HIDDEN,         Self.tp, next_offset[Self.Q_PROJ](),     "self_attn.k_proj.weight"]
    comptime V_PROJ = PlacedSlot[BF16, Replicated, C.KV_DIM_SLIDING, C.HIDDEN,         Self.tp, next_offset[Self.K_PROJ](),     "self_attn.v_proj.weight"]
    comptime O_PROJ = PlacedSlot[BF16, Replicated, C.HIDDEN,         C.Q_DIM_SLIDING,  Self.tp, next_offset[Self.V_PROJ](),     "self_attn.o_proj.weight"]
    comptime Q_NORM = PlacedSlot[BF16, Replicated, C.HEAD_DIM_SLIDING, 1,              Self.tp, next_offset[Self.O_PROJ](),     "self_attn.q_norm.weight"]
    comptime K_NORM = PlacedSlot[BF16, Replicated, C.HEAD_DIM_SLIDING, 1,              Self.tp, next_offset[Self.Q_NORM](),     "self_attn.k_norm.weight"]

    # --- Norms ---
    comptime INPUT_NORM      = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.K_NORM](),         "input_layernorm.weight"]
    comptime POST_ATTN_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](),     "post_attention_layernorm.weight"]
    comptime PRE_FFN_NORM    = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_ATTN_NORM](), "pre_feedforward_layernorm.weight"]
    comptime PRE_FFN_NORM_2  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.PRE_FFN_NORM](),   "pre_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM_1 = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.PRE_FFN_NORM_2](), "post_feedforward_layernorm_1.weight"]
    comptime POST_FFN_NORM_2 = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_1](), "post_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_2](), "post_feedforward_layernorm.weight"]

    # --- Dense MLP ---
    comptime GATE_PROJ = PlacedSlot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN,       Self.tp, next_offset[Self.POST_FFN_NORM](), "mlp.gate_proj.weight"]
    comptime UP_PROJ   = PlacedSlot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN,       Self.tp, next_offset[Self.GATE_PROJ](),     "mlp.up_proj.weight"]
    comptime DOWN_PROJ = PlacedSlot[BF16, Replicated, C.HIDDEN,       C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](),       "mlp.down_proj.weight"]

    # --- Router ---
    comptime ROUTER_PROJ             = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, C.HIDDEN, Self.tp, next_offset[Self.DOWN_PROJ](),               "router.proj.weight"]
    comptime ROUTER_SCALE            = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,        Self.tp, next_offset[Self.ROUTER_PROJ](),             "router.scale"]
    comptime ROUTER_PER_EXPERT_SCALE = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, 1,        Self.tp, next_offset[Self.ROUTER_SCALE](),            "router.per_expert_scale"]

    # --- Expert weights (3D folded to 2D) ---
    comptime EXPERTS_GATE_UP = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS * C.GATE_UP_DIM, C.HIDDEN,          Self.tp, next_offset[Self.ROUTER_PER_EXPERT_SCALE](), "experts.gate_up_proj"]
    comptime EXPERTS_DOWN    = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS * C.HIDDEN,      C.MOE_INTERMEDIATE, Self.tp, next_offset[Self.EXPERTS_GATE_UP](),          "experts.down_proj"]

    # --- Layer scalar ---
    comptime LAYER_SCALAR = PlacedSlot[BF16, Replicated, 1, 1, Self.tp, next_offset[Self.EXPERTS_DOWN](), "layer_scalar"]

    comptime STRIDE = next_offset[Self.LAYER_SCALAR]()

    # --- KV caches (in state, not weights) ---
    comptime K_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]
    comptime V_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named](String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base)
        func[Self.K_PROJ](prefix, base)
        func[Self.V_PROJ](prefix, base)
        func[Self.O_PROJ](prefix, base)
        func[Self.Q_NORM](prefix, base)
        func[Self.K_NORM](prefix, base)
        func[Self.INPUT_NORM](prefix, base)
        func[Self.POST_ATTN_NORM](prefix, base)
        func[Self.PRE_FFN_NORM](prefix, base)
        func[Self.PRE_FFN_NORM_2](prefix, base)
        func[Self.POST_FFN_NORM_1](prefix, base)
        func[Self.POST_FFN_NORM_2](prefix, base)
        func[Self.POST_FFN_NORM](prefix, base)
        func[Self.GATE_PROJ](prefix, base)
        func[Self.UP_PROJ](prefix, base)
        func[Self.DOWN_PROJ](prefix, base)
        func[Self.ROUTER_PROJ](prefix, base)
        func[Self.ROUTER_SCALE](prefix, base)
        func[Self.ROUTER_PER_EXPERT_SCALE](prefix, base)
        func[Self.EXPERTS_GATE_UP](prefix, base)
        func[Self.EXPERTS_DOWN](prefix, base)
        func[Self.LAYER_SCALAR](prefix, base)


# =============================================================================
# Full/global attention layer (5 of 30)
# =============================================================================


struct FullLayer[tp: Int]:
    # --- Attention (full geometry, K=V shared) ---
    comptime Q_PROJ = PlacedSlot[BF16, Replicated, C.Q_DIM_FULL,  C.HIDDEN,       Self.tp, 0,                             "self_attn.q_proj.weight"]
    comptime K_PROJ = PlacedSlot[BF16, Replicated, C.KV_DIM_FULL, C.HIDDEN,       Self.tp, next_offset[Self.Q_PROJ](),     "self_attn.k_proj.weight"]
    comptime O_PROJ = PlacedSlot[BF16, Replicated, C.HIDDEN,      C.Q_DIM_FULL,   Self.tp, next_offset[Self.K_PROJ](),     "self_attn.o_proj.weight"]
    comptime Q_NORM = PlacedSlot[BF16, Replicated, C.HEAD_DIM_FULL, 1,            Self.tp, next_offset[Self.O_PROJ](),     "self_attn.q_norm.weight"]
    comptime K_NORM = PlacedSlot[BF16, Replicated, C.HEAD_DIM_FULL, 1,            Self.tp, next_offset[Self.Q_NORM](),     "self_attn.k_norm.weight"]

    # --- Norms ---
    comptime INPUT_NORM      = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.K_NORM](),         "input_layernorm.weight"]
    comptime POST_ATTN_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](),     "post_attention_layernorm.weight"]
    comptime PRE_FFN_NORM    = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_ATTN_NORM](), "pre_feedforward_layernorm.weight"]
    comptime PRE_FFN_NORM_2  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.PRE_FFN_NORM](),   "pre_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM_1 = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.PRE_FFN_NORM_2](), "post_feedforward_layernorm_1.weight"]
    comptime POST_FFN_NORM_2 = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_1](), "post_feedforward_layernorm_2.weight"]
    comptime POST_FFN_NORM   = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.POST_FFN_NORM_2](), "post_feedforward_layernorm.weight"]

    # --- Dense MLP ---
    comptime GATE_PROJ = PlacedSlot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN,       Self.tp, next_offset[Self.POST_FFN_NORM](), "mlp.gate_proj.weight"]
    comptime UP_PROJ   = PlacedSlot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN,       Self.tp, next_offset[Self.GATE_PROJ](),     "mlp.up_proj.weight"]
    comptime DOWN_PROJ = PlacedSlot[BF16, Replicated, C.HIDDEN,       C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](),       "mlp.down_proj.weight"]

    # --- Router ---
    comptime ROUTER_PROJ             = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, C.HIDDEN, Self.tp, next_offset[Self.DOWN_PROJ](),               "router.proj.weight"]
    comptime ROUTER_SCALE            = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,        Self.tp, next_offset[Self.ROUTER_PROJ](),             "router.scale"]
    comptime ROUTER_PER_EXPERT_SCALE = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS, 1,        Self.tp, next_offset[Self.ROUTER_SCALE](),            "router.per_expert_scale"]

    # --- Expert weights ---
    comptime EXPERTS_GATE_UP = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS * C.GATE_UP_DIM, C.HIDDEN,          Self.tp, next_offset[Self.ROUTER_PER_EXPERT_SCALE](), "experts.gate_up_proj"]
    comptime EXPERTS_DOWN    = PlacedSlot[BF16, Replicated, C.NUM_EXPERTS * C.HIDDEN,      C.MOE_INTERMEDIATE, Self.tp, next_offset[Self.EXPERTS_GATE_UP](),          "experts.down_proj"]

    # --- Layer scalar ---
    comptime LAYER_SCALAR = PlacedSlot[BF16, Replicated, 1, 1, Self.tp, next_offset[Self.EXPERTS_DOWN](), "layer_scalar"]

    comptime STRIDE = next_offset[Self.LAYER_SCALAR]()

    # --- KV caches ---
    comptime K_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]
    comptime V_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named](String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base)
        func[Self.K_PROJ](prefix, base)
        func[Self.O_PROJ](prefix, base)
        func[Self.Q_NORM](prefix, base)
        func[Self.K_NORM](prefix, base)
        func[Self.INPUT_NORM](prefix, base)
        func[Self.POST_ATTN_NORM](prefix, base)
        func[Self.PRE_FFN_NORM](prefix, base)
        func[Self.PRE_FFN_NORM_2](prefix, base)
        func[Self.POST_FFN_NORM_1](prefix, base)
        func[Self.POST_FFN_NORM_2](prefix, base)
        func[Self.POST_FFN_NORM](prefix, base)
        func[Self.GATE_PROJ](prefix, base)
        func[Self.UP_PROJ](prefix, base)
        func[Self.DOWN_PROJ](prefix, base)
        func[Self.ROUTER_PROJ](prefix, base)
        func[Self.ROUTER_SCALE](prefix, base)
        func[Self.ROUTER_PER_EXPERT_SCALE](prefix, base)
        func[Self.EXPERTS_GATE_UP](prefix, base)
        func[Self.EXPERTS_DOWN](prefix, base)
        func[Self.LAYER_SCALAR](prefix, base)


# =============================================================================
# Model spec — weight layout + state layout
# =============================================================================


struct Gemma4Model[tp: Int](WeightIterable):
    comptime SLIDING = SlidingLayer[Self.tp]
    comptime FULL    = FullLayer[Self.tp]

    # Weight layout: [sliding layers][full layers]
    comptime SLIDING_OFF    = 0
    comptime FULL_OFF       = C.NUM_SLIDING_LAYERS * Self.SLIDING.STRIDE
    comptime DISTRIBUTED_BYTES = Self.FULL_OFF + C.NUM_FULL_LAYERS * Self.FULL.STRIDE

    # Scratch view types (unplaced — used with absolute addresses from ScratchPool)
    comptime Q_SLIDING_VIEW  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.Q_DIM_SLIDING, Self.tp]
    comptime KV_SLIDING_VIEW = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_SLIDING, Self.tp]
    comptime Q_FULL_VIEW     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.Q_DIM_FULL, Self.tp]
    comptime KV_FULL_VIEW    = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.tp]
    comptime MLP_VIEW        = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime HIDDEN_VIEW     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime LOGITS_VIEW     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    # FFN weight view types (used with runtime-computed absolute addresses)
    comptime FFN_GATE_W  = Slot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN, Self.tp]
    comptime FFN_UP_W    = Slot[BF16, Replicated, C.INTERMEDIATE, C.HIDDEN, Self.tp]
    comptime FFN_DOWN_W  = Slot[BF16, Replicated, C.HIDDEN, C.INTERMEDIATE, Self.tp]
    comptime NORM_W      = Slot[BF16, Replicated, C.HIDDEN, 1, Self.tp]
    comptime ROUTER_W    = Slot[BF16, Replicated, C.NUM_EXPERTS, C.HIDDEN, Self.tp]
    comptime EXPERT_SC_W = Slot[BF16, Replicated, C.NUM_EXPERTS, 1, Self.tp]

    # Scratch capacity
    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        comptime S = C.MAX_SEQ_LEN

        # Full attention phase peak: q + k + v (then release k,v and borrow attn_out)
        comptime full_attn_borrows = (
            S * C.Q_DIM_FULL * 2 +
            S * C.KV_DIM_FULL * 2 +
            S * C.KV_DIM_FULL * 2 +
            S * C.Q_DIM_FULL * 2
        )

        # Sliding attention phase peak
        comptime sliding_attn_borrows = (
            S * C.Q_DIM_SLIDING * 2 +
            S * C.KV_DIM_SLIDING * 2 +
            S * C.KV_DIM_SLIDING * 2 +
            S * C.Q_DIM_SLIDING * 2
        )

        # FFN phase peak: gate + up (then release, borrow dense_normed + moe_out + expert_buf)
        comptime ffn_borrows_dense = S * C.INTERMEDIATE * 2 * 2  # gate + up
        comptime ffn_borrows_moe = (
            S * C.HIDDEN * 2 +           # dense_normed
            S * C.HIDDEN * 2 +           # moe_out
            C.TOP_K * C.HIDDEN * 2       # expert_out_buf (flat, per-token)
        )
        comptime ffn_peak = max(ffn_borrows_dense, ffn_borrows_moe)

        comptime attn_peak = max(full_attn_borrows, sliding_attn_borrows)
        return max(attn_peak, ffn_peak)

    # Activation slots
    comptime X_MAIN     = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]

    # RoPE table slots
    comptime SLIDING_HALF = C.HEAD_DIM_SLIDING // 2
    comptime SLIDING_COS  = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_HALF, Self.tp]
    comptime SLIDING_SIN  = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.SLIDING_HALF, Self.tp]
    comptime FULL_COS     = Slot[F32, Replicated, C.MAX_SEQ_LEN, 64, Self.tp]
    comptime FULL_SIN     = Slot[F32, Replicated, C.MAX_SEQ_LEN, 64, Self.tp]

    # State layout
    comptime SLIDING_KV_STRIDE = Self.SLIDING.cache_bytes()
    comptime FULL_KV_STRIDE    = Self.FULL.cache_bytes()

    comptime SLIDING_KV_OFF = 0
    comptime FULL_KV_OFF    = Self.SLIDING_KV_OFF + C.NUM_SLIDING_LAYERS * Self.SLIDING_KV_STRIDE
    comptime X_MAIN_OFF     = Self.FULL_KV_OFF + C.NUM_FULL_LAYERS * Self.FULL_KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF    = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SLIDING_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime SLIDING_SIN_OFF = Self.SLIDING_COS_OFF + byte_count[Self.SLIDING_COS]()
    comptime FULL_COS_OFF    = Self.SLIDING_SIN_OFF + byte_count[Self.SLIDING_SIN]()
    comptime FULL_SIN_OFF    = Self.FULL_COS_OFF + byte_count[Self.FULL_COS]()
    comptime STATE_BYTES     = Self.FULL_SIN_OFF + byte_count[Self.FULL_SIN]()

    # Host-only weights (after distributed weights + state)
    comptime HOST_ONLY_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM    = PlacedSlot[BF16, Replicated, C.HIDDEN,     1,        Self.tp, Self.HOST_ONLY_OFF,             "model.language_model.norm.weight", target_rank=HOST_RANK]
    comptime EMBED         = PlacedSlot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.language_model.embed_tokens.weight", target_rank=HOST_RANK]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named](String, Int) capturing -> None,
    ]():
        var sliding_idx = 0
        var full_idx = 0
        comptime for i in range(C.NUM_LAYERS):
            var prefix = "model.language_model.layers." + String(i) + "."
            comptime if (i + 1) % 6 == 0:
                var base = Self.FULL_OFF + full_idx * Self.FULL.STRIDE
                Self.FULL.for_each_weight[func](prefix, base)
                full_idx += 1
            else:
                var base = Self.SLIDING_OFF + sliding_idx * Self.SLIDING.STRIDE
                Self.SLIDING.for_each_weight[func](prefix, base)
                sliding_idx += 1

        func[Self.FINAL_NORM]("", 0)
        func[Self.EMBED]("", 0)

    @staticmethod
    def host_arena_bytes() -> Int:
        return next_offset[Self.EMBED]()


# =============================================================================
# State accessor
# =============================================================================


@fieldwise_init
struct StateView[tp: Int]:
    comptime M = Gemma4Model[Self.tp]
    comptime SL = SlidingLayer[Self.tp]
    comptime FL = FullLayer[Self.tp]
    var base: Int

    def weight_base(self) -> Int:
        return self.base

    def state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES

    def scratch_base(self) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF

    # Activations
    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    # Scratch views
    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        return DynView[V](self.scratch_base() + lease.offset, seq_len)

    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)

    # KV caches
    def sliding_k_cache(self, sliding_idx: Int) -> CacheView[Self.SL.K_CACHE]:
        return CacheView[Self.SL.K_CACHE](self.state_base() + Self.M.SLIDING_KV_OFF + sliding_idx * Self.M.SLIDING_KV_STRIDE)

    def sliding_v_cache(self, sliding_idx: Int) -> CacheView[Self.SL.V_CACHE]:
        return CacheView[Self.SL.V_CACHE](
            self.state_base() + Self.M.SLIDING_KV_OFF + sliding_idx * Self.M.SLIDING_KV_STRIDE + byte_count[Self.SL.K_CACHE]())

    def full_k_cache(self, full_idx: Int) -> CacheView[Self.FL.K_CACHE]:
        return CacheView[Self.FL.K_CACHE](self.state_base() + Self.M.FULL_KV_OFF + full_idx * Self.M.FULL_KV_STRIDE)

    def full_v_cache(self, full_idx: Int) -> CacheView[Self.FL.V_CACHE]:
        return CacheView[Self.FL.V_CACHE](
            self.state_base() + Self.M.FULL_KV_OFF + full_idx * Self.M.FULL_KV_STRIDE + byte_count[Self.FL.K_CACHE]())

    # RoPE tables
    def sliding_cos(self) -> Bound[Self.M.SLIDING_COS]:
        return Bound[Self.M.SLIDING_COS](self.state_base() + Self.M.SLIDING_COS_OFF)

    def sliding_sin(self) -> Bound[Self.M.SLIDING_SIN]:
        return Bound[Self.M.SLIDING_SIN](self.state_base() + Self.M.SLIDING_SIN_OFF)

    def full_cos(self) -> Bound[Self.M.FULL_COS]:
        return Bound[Self.M.FULL_COS](self.state_base() + Self.M.FULL_COS_OFF)

    def full_sin(self) -> Bound[Self.M.FULL_SIN]:
        return Bound[Self.M.FULL_SIN](self.state_base() + Self.M.FULL_SIN_OFF)

    # Weight access — sliding layer
    def sliding_bind[T: Encoding & Shaped & Placed & Named](self, sliding_idx: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.SLIDING_OFF + sliding_idx * Self.SL.STRIDE)

    # Weight access — full layer
    def full_bind[T: Encoding & Shaped & Placed & Named](self, full_idx: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.FULL_OFF + full_idx * Self.FL.STRIDE)

    # Weight access — host-only
    def host_bind[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())

    # Raw weight address for a layer (for FFN weight views)
    def sliding_base(self, sliding_idx: Int) -> Int:
        return self.weight_base() + Self.M.SLIDING_OFF + sliding_idx * Self.SL.STRIDE

    def full_base(self, full_idx: Int) -> Int:
        return self.weight_base() + Self.M.FULL_OFF + full_idx * Self.FL.STRIDE


# =============================================================================
# Loaded model
# =============================================================================


struct Gemma4[tp: Int](Movable):
    comptime M = Gemma4Model[Self.tp]
    comptime SL = SlidingLayer[Self.tp]
    comptime FL = FullLayer[Self.tp]

    var arena: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool: BurstPool[]
    var scratch: ScratchPool
    var base: Int

    def __init__(out self, var arena: NumaArena[alignment=DEFAULT_ALIGNMENT], var pool: BurstPool[]):
        self.base = Int(arena.base)
        self.arena = arena^
        self.pool = pool^
        self.scratch = ScratchPool(Self.M.SCRATCH_CAPACITY)

    def sv(self) -> StateView[Self.tp]:
        return StateView[Self.tp](self.base)

    @staticmethod
    def discover_shards(dir_path: Path) -> List[Path]:
        var shards = List[Path]()
        try:
            for entry in dir_path.listdir():
                var name = String(entry)
                if name.endswith(".safetensors") and name.startswith("model-"):
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
        """Initialize RoPE tables and bake router constants."""
        var s = self.sv()

        # RoPE tables
        init_sliding_rope_tables(s.sliding_cos(), s.sliding_sin())
        init_full_rope_tables(s.full_cos(), s.full_sin())

        # Bake router constant: router_scale *= 1/sqrt(hidden)
        comptime inv_sqrt_hidden = Float32(1.0 / sqrt(Float64(C.HIDDEN)))
        comptime width = simd_width_of[DType.float32]()
        var sliding_idx = 0
        var full_idx = 0
        for i in range(C.NUM_LAYERS):
            var scale_ptr: BF16Ptr
            if (i + 1) % 6 == 0:
                scale_ptr = BF16Ptr(unsafe_from_address=s.full_base(full_idx) + Self.FL.ROUTER_SCALE.OFFSET)
                full_idx += 1
            else:
                scale_ptr = BF16Ptr(unsafe_from_address=s.sliding_base(sliding_idx) + Self.SL.ROUTER_SCALE.OFFSET)
                sliding_idx += 1
            for j in range(0, C.HIDDEN, width):
                var v = (scale_ptr + j).load[width=width]().cast[DType.float32]()
                (scale_ptr + j).store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        print("state initialized: rope tables + baked router constants")

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        var shards = Self.discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var numa = NumaInfo()
        var topo = numa.plan_topology(1)

        var size = Self.M.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB (" +
              String(Self.M.DISTRIBUTED_BYTES // (1024 * 1024)) + " MB weights + " +
              String(Self.M.STATE_BYTES // (1024 * 1024)) + " MB state)")
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[0], size)
        if not arena:
            print("arena allocation failed")
            return None

        var arena_bases = List[Int]()
        arena_bases.append(Int(arena.base))

        var result = load_weights[Self.M](shards, arena_bases)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        _ = arena.prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        var pool = BurstPool[].for_numa_node(numa, topo[0])
        var model = Self(arena^, pool^)
        model.init_state()
        return model^

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.sv().scratch_base())

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        comptime M = Self.M
        var s = self.sv()
        var pool_ptr = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=self.pool)))

        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        # --- Embed ---
        embed_lookup_scaled(s.host_bind[M.EMBED](), tokens_ptr, s.x_main(seq_len),
            Float32(C.EMBED_SCALE), self.pool).join()

        _ = pos

        # --- Layer loop ---
        var sliding_idx = 0
        var full_idx = 0

        for layer_idx in range(C.NUM_LAYERS):
            var is_full = (layer_idx + 1) % 6 == 0

            # Compute layer weight base for FFN (common weights at different offsets)
            var lb: Int
            if is_full:
                lb = s.full_base(full_idx)
            else:
                lb = s.sliding_base(sliding_idx)

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
            var post_attn_off = Self.FL.POST_ATTN_NORM.OFFSET if is_full else Self.SL.POST_ATTN_NORM.OFFSET
            rmsnorm(s.x_residual(seq_len), Bound[M.NORM_W](lb + post_attn_off),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
            elem_add(s.x_main(seq_len), s.x_residual(seq_len), s.x_main(seq_len))

            # --- Dense MLP path ---
            var pre_ffn_off = Self.FL.PRE_FFN_NORM.OFFSET if is_full else Self.SL.PRE_FFN_NORM.OFFSET
            rmsnorm(s.x_main(seq_len), Bound[M.NORM_W](lb + pre_ffn_off),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var gate_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.INTERMEDIATE]()
            var up_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.INTERMEDIATE]()

            var gate_off = Self.FL.GATE_PROJ.OFFSET if is_full else Self.SL.GATE_PROJ.OFFSET
            var up_off = Self.FL.UP_PROJ.OFFSET if is_full else Self.SL.UP_PROJ.OFFSET
            gemm(s.x_residual(seq_len), Bound[M.FFN_GATE_W](lb + gate_off),
                s.scratch_view[M.MLP_VIEW](gate_lease, seq_len), self.pool).join()
            gemm(s.x_residual(seq_len), Bound[M.FFN_UP_W](lb + up_off),
                s.scratch_view[M.MLP_VIEW](up_lease, seq_len), self.pool).join()

            gelu_tanh_mul(s.scratch_view[M.MLP_VIEW](gate_lease, seq_len),
                s.scratch_view[M.MLP_VIEW](up_lease, seq_len),
                s.scratch_view[M.MLP_VIEW](gate_lease, seq_len))

            up_lease^.release()

            var down_off = Self.FL.DOWN_PROJ.OFFSET if is_full else Self.SL.DOWN_PROJ.OFFSET
            gemm(s.scratch_view[M.MLP_VIEW](gate_lease, seq_len), Bound[M.FFN_DOWN_W](lb + down_off),
                s.x_residual(seq_len), self.pool).join()

            gate_lease^.release()

            # dense_normed = rmsnorm(dense_out, POST_FFN_NORM_1)
            var dense_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            var post_ffn1_off = Self.FL.POST_FFN_NORM_1.OFFSET if is_full else Self.SL.POST_FFN_NORM_1.OFFSET
            rmsnorm(s.x_residual(seq_len), Bound[M.NORM_W](lb + post_ffn1_off),
                s.scratch_view[M.HIDDEN_VIEW](dense_lease, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            # --- MoE path ---
            # Router: rmsnorm with baked scale (contains 1/sqrt(hidden))
            var router_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            var router_scale_off = Self.FL.ROUTER_SCALE.OFFSET if is_full else Self.SL.ROUTER_SCALE.OFFSET
            rmsnorm(s.x_main(seq_len), Bound[M.NORM_W](lb + router_scale_off),
                s.scratch_view[M.HIDDEN_VIEW](router_lease, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            # Router projection → softmax_topk_renorm (per-token, inline for decode)
            var router_proj_off = Self.FL.ROUTER_PROJ.OFFSET if is_full else Self.SL.ROUTER_PROJ.OFFSET
            var per_expert_off = Self.FL.ROUTER_PER_EXPERT_SCALE.OFFSET if is_full else Self.SL.ROUTER_PER_EXPERT_SCALE.OFFSET

            var router_logits_buf = InlineArray[Scalar[DType.bfloat16], C.NUM_EXPERTS](fill=Scalar[DType.bfloat16](0))
            var router_logits_ptr = BF16Ptr(unsafe_from_address=Int(UnsafePointer(to=router_logits_buf[0])))
            var router_input_ptr = s.scratch_ptr[Scalar[DType.bfloat16]](router_lease)

            gemv_kernel[C.HIDDEN, C.NUM_EXPERTS](GemmArgs(
                router_input_ptr,
                BF16Ptr(unsafe_from_address=lb + router_proj_off),
                router_logits_ptr, 0, C.NUM_EXPERTS, 1))

            var routing = softmax_topk_renorm[C.NUM_EXPERTS, C.TOP_K](
                router_logits_ptr,
                BF16Ptr(unsafe_from_address=lb + per_expert_off))

            router_lease^.release()

            # MoE FFN: norm → expert dispatch
            var pre_ffn2_off = Self.FL.PRE_FFN_NORM_2.OFFSET if is_full else Self.SL.PRE_FFN_NORM_2.OFFSET
            rmsnorm(s.x_main(seq_len), Bound[M.NORM_W](lb + pre_ffn2_off),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var moe_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
            var expert_buf_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.TOP_K * C.HIDDEN]()

            var experts_gu_off = Self.FL.EXPERTS_GATE_UP.OFFSET if is_full else Self.SL.EXPERTS_GATE_UP.OFFSET
            var experts_d_off = Self.FL.EXPERTS_DOWN.OFFSET if is_full else Self.SL.EXPERTS_DOWN.OFFSET

            gemma4_moe_dispatch[C.NUM_EXPERTS, C.TOP_K, C.MOE_INTERMEDIATE, C.HIDDEN](
                tptr[Scalar[DType.bfloat16]](s.x_residual(seq_len).ptr),
                routing,
                BF16Ptr(unsafe_from_address=lb + experts_gu_off),
                BF16Ptr(unsafe_from_address=lb + experts_d_off),
                s.scratch_ptr[Scalar[DType.bfloat16]](expert_buf_lease),
                s.scratch_ptr[Scalar[DType.bfloat16]](moe_lease),
                self.pool)

            expert_buf_lease^.release()

            # moe_normed = rmsnorm(moe_out, POST_FFN_NORM_2)
            var post_ffn2_off = Self.FL.POST_FFN_NORM_2.OFFSET if is_full else Self.SL.POST_FFN_NORM_2.OFFSET
            rmsnorm(s.scratch_view[M.HIDDEN_VIEW](moe_lease, seq_len), Bound[M.NORM_W](lb + post_ffn2_off),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            moe_lease^.release()

            # --- Combine ---
            elem_add(s.scratch_view[M.HIDDEN_VIEW](dense_lease, seq_len), s.x_residual(seq_len), s.x_residual(seq_len))
            dense_lease^.release()

            var post_ffn_off = Self.FL.POST_FFN_NORM.OFFSET if is_full else Self.SL.POST_FFN_NORM.OFFSET
            rmsnorm(s.x_residual(seq_len), Bound[M.NORM_W](lb + post_ffn_off),
                s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

            var ls_off = Self.FL.LAYER_SCALAR.OFFSET if is_full else Self.SL.LAYER_SCALAR.OFFSET
            var layer_scalar = Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=lb + ls_off)[])
            elem_add(s.x_main(seq_len), s.x_residual(seq_len), s.x_main(seq_len))
            elem_scale(s.x_main(seq_len), layer_scalar)

            if is_full:
                full_idx += 1
            else:
                sliding_idx += 1

        # --- Final norm + LM head ---
        rmsnorm(s.x_main(seq_len), s.host_bind[M.FINAL_NORM](),
            s.x_main(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        _ = pos

        var last_row_off = (seq_len - 1) * C.HIDDEN * M.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[M.X_MAIN](s.x_main(seq_len).ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = s.scratch_view[M.LOGITS_VIEW](logit_lease, 1)
        gemm(last_hidden, s.host_bind[M.EMBED](), logit_view, self.pool).join()

        logit_softcap(logit_view)

        return LogitsView[C.VOCAB_SIZE](
            s.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^)

    # =========================================================================
    # Attention helpers (separate methods to avoid massive forward body)
    # =========================================================================

    def attention_sliding(mut self, s: StateView[Self.tp], sliding_idx: Int, seq_len: Int, pos: Int):
        comptime M = Self.M

        # Input norm → x_residual
        rmsnorm(s.x_main(seq_len), s.sliding_bind[Self.SL.INPUT_NORM](sliding_idx),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K, V projections
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_SLIDING]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_SLIDING]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_SLIDING]()

        gemm(s.x_residual(seq_len), s.sliding_bind[Self.SL.Q_PROJ](sliding_idx),
            s.scratch_view[M.Q_SLIDING_VIEW](q, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), s.sliding_bind[Self.SL.K_PROJ](sliding_idx),
            s.scratch_view[M.KV_SLIDING_VIEW](k, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), s.sliding_bind[Self.SL.V_PROJ](sliding_idx),
            s.scratch_view[M.KV_SLIDING_VIEW](v, seq_len), self.pool).join()

        # Per-head norms
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            s.scratch_view[M.Q_SLIDING_VIEW](q, seq_len), s.sliding_bind[Self.SL.Q_NORM](sliding_idx),
            s.scratch_view[M.Q_SLIDING_VIEW](q, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            s.scratch_view[M.KV_SLIDING_VIEW](k, seq_len), s.sliding_bind[Self.SL.K_NORM](sliding_idx),
            s.scratch_view[M.KV_SLIDING_VIEW](k, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(s.scratch_view[M.KV_SLIDING_VIEW](v, seq_len),
            s.scratch_view[M.KV_SLIDING_VIEW](v, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # RoPE (Q and K only)
        rope[C.HEAD_DIM_SLIDING, C.NUM_HEADS](
            s.scratch_view[M.Q_SLIDING_VIEW](q, seq_len), s.sliding_cos(), s.sliding_sin(), pos)
        rope[C.HEAD_DIM_SLIDING, C.NUM_KV_HEADS_SLIDING](
            s.scratch_view[M.KV_SLIDING_VIEW](k, seq_len), s.sliding_cos(), s.sliding_sin(), pos)

        # KV cache write
        kv_cache_write(s.scratch_view[M.KV_SLIDING_VIEW](k, seq_len), s.sliding_k_cache(sliding_idx), pos)
        kv_cache_write(s.scratch_view[M.KV_SLIDING_VIEW](v, seq_len), s.sliding_v_cache(sliding_idx), pos)

        v^.release()
        k^.release()

        # Attention
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_SLIDING]()
        local_attention[C.NUM_HEADS, C.NUM_KV_HEADS_SLIDING, C.HEAD_DIM_SLIDING, C.SLIDING_WINDOW](
            s.scratch_view[M.Q_SLIDING_VIEW](q, seq_len),
            s.sliding_k_cache(sliding_idx), s.sliding_v_cache(sliding_idx),
            s.scratch_view[M.Q_SLIDING_VIEW](attn_out, seq_len), pos, self.pool).join()

        # O projection → x_residual
        gemm(s.scratch_view[M.Q_SLIDING_VIEW](attn_out, seq_len),
            s.sliding_bind[Self.SL.O_PROJ](sliding_idx),
            s.x_residual(seq_len), self.pool).join()

        attn_out^.release()
        q^.release()

    def attention_full(mut self, s: StateView[Self.tp], full_idx: Int, seq_len: Int, pos: Int):
        comptime M = Self.M

        # Input norm → x_residual
        rmsnorm(s.x_main(seq_len), s.full_bind[Self.FL.INPUT_NORM](full_idx),
            s.x_residual(seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # Q, K projections (V shares K projection)
        var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_FULL]()
        var k = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_FULL]()
        var v = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.KV_DIM_FULL]()

        gemm(s.x_residual(seq_len), s.full_bind[Self.FL.Q_PROJ](full_idx),
            s.scratch_view[M.Q_FULL_VIEW](q, seq_len), self.pool).join()
        gemm(s.x_residual(seq_len), s.full_bind[Self.FL.K_PROJ](full_idx),
            s.scratch_view[M.KV_FULL_VIEW](k, seq_len), self.pool).join()
        # K=V sharing: copy K projection output to V scratch before norms diverge
        var kp = s.scratch_ptr[Scalar[DType.bfloat16]](k)
        var vp = s.scratch_ptr[Scalar[DType.bfloat16]](v)
        comptime copy_width = simd_width_of[DType.bfloat16]()
        for j in range(0, C.KV_DIM_FULL * seq_len, copy_width):
            (vp + j).store((kp + j).load[width=copy_width]())

        # Per-head norms (norms diverge: K gets scale, V doesn't)
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_HEADS](
            s.scratch_view[M.Q_FULL_VIEW](q, seq_len), s.full_bind[Self.FL.Q_NORM](full_idx),
            s.scratch_view[M.Q_FULL_VIEW](q, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_per_head[C.HEAD_DIM_FULL, C.NUM_KV_HEADS_FULL](
            s.scratch_view[M.KV_FULL_VIEW](k, seq_len), s.full_bind[Self.FL.K_NORM](full_idx),
            s.scratch_view[M.KV_FULL_VIEW](k, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()
        rmsnorm_no_scale(s.scratch_view[M.KV_FULL_VIEW](v, seq_len),
            s.scratch_view[M.KV_FULL_VIEW](v, seq_len), self.pool, Float32(C.RMS_NORM_EPS)).join()

        # RoPE (Q and K only, partial rotation for full attention)
        apply_full_rope[C.NUM_HEADS](s.scratch_view[M.Q_FULL_VIEW](q, seq_len),
            s.full_cos(), s.full_sin(), pos)
        apply_full_rope[C.NUM_KV_HEADS_FULL](s.scratch_view[M.KV_FULL_VIEW](k, seq_len),
            s.full_cos(), s.full_sin(), pos)

        # KV cache write
        kv_cache_write(s.scratch_view[M.KV_FULL_VIEW](k, seq_len), s.full_k_cache(full_idx), pos)
        kv_cache_write(s.scratch_view[M.KV_FULL_VIEW](v, seq_len), s.full_v_cache(full_idx), pos)

        v^.release()
        k^.release()

        # Attention
        var attn_out = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.Q_DIM_FULL]()
        global_attention[C.NUM_HEADS, C.NUM_KV_HEADS_FULL, C.HEAD_DIM_FULL](
            s.scratch_view[M.Q_FULL_VIEW](q, seq_len),
            s.full_k_cache(full_idx), s.full_v_cache(full_idx),
            s.scratch_view[M.Q_FULL_VIEW](attn_out, seq_len), pos, self.pool).join()
        # O projection → x_residual
        gemm(s.scratch_view[M.Q_FULL_VIEW](attn_out, seq_len),
            s.full_bind[Self.FL.O_PROJ](full_idx),
            s.x_residual(seq_len), self.pool).join()

        attn_out^.release()
        q^.release()

    # =========================================================================
    # Variance reporting (unchanged from initial bring-up)
    # =========================================================================

    @staticmethod
    def bf16_rms(ptr: Int, n: Int) -> Float64:
        var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
        comptime width = simd_width_of[DType.float32]()
        var acc = Float64(0)
        for i in range(0, n, width):
            var v = p.load[width=width](offset=i).cast[DType.float64]()
            acc += (v * v).reduce_add()
        return sqrt(acc / Float64(n))

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
        comptime SL = Self.M.SLIDING
        comptime FL = Self.M.FULL
        var base = self.base

        var sliding_idx = 0
        var full_idx = 0
        for i in range(C.NUM_LAYERS):
            if (i + 1) % 6 == 0:
                var layer_base = base + Self.M.FULL_OFF + full_idx * FL.STRIDE
                var q_ptr = layer_base + FL.Q_PROJ.OFFSET
                var q_n = FL.Q_PROJ.ROWS * FL.Q_PROJ.COLS
                var k_ptr = layer_base + FL.K_PROJ.OFFSET
                var k_n = FL.K_PROJ.ROWS * FL.K_PROJ.COLS
                print("layer", i, "(full)   | q:", Self.bf16_variance(q_ptr, q_n),
                    "| k:", Self.bf16_variance(k_ptr, k_n))
                full_idx += 1
            else:
                var layer_base = base + Self.M.SLIDING_OFF + sliding_idx * SL.STRIDE
                var q_ptr = layer_base + SL.Q_PROJ.OFFSET
                var q_n = SL.Q_PROJ.ROWS * SL.Q_PROJ.COLS
                var k_ptr = layer_base + SL.K_PROJ.OFFSET
                var k_n = SL.K_PROJ.ROWS * SL.K_PROJ.COLS
                print("layer", i, "(slide)  | q:", Self.bf16_variance(q_ptr, q_n),
                    "| k:", Self.bf16_variance(k_ptr, k_n))
                sliding_idx += 1


def main():
    var model_opt = Gemma4[1].load(Path("checkpoints/gemma-4-26B-A4B"))
    if not model_opt:
        print("failed to load model")
        return
    var model = model_opt.take()
    model.report_weight_variance()
