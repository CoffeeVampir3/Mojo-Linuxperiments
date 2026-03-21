# ===----------------------------------------------------------------------=== #
# smollm2.mojo — SmolLM2-135M model
#
# Self-describing model. Each PlacedSlot carries its encoding, sharding,
# local/global dims, arena offset, and tensor name. The model provides
# its structure via Loadable traits. The loader drives iteration.
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import linux.sys as linux

from experimental4.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F16, F32, I8,
    ShardStrategy, RowShard, ColShard, Replicated,
    Slot, PlacedSlot, byte_count,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
)
from experimental4.loader import (
    Loadable, WeightDesc, weight_desc, LoadResult, load_safetensors,
    offset_after, next_offset,
)


# ===--- Architecture ---=== #

struct SmolLM2Config(Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig):
    comptime HIDDEN = 576
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 9
    comptime NUM_KV_HEADS = 3
    comptime INTERMEDIATE = 1536
    comptime VOCAB_SIZE = 49152
    comptime MAX_SEQ_LEN = 8192
    comptime ROPE_THETA = 100000.0
    comptime RMS_NORM_EPS = 1e-5
    comptime TIE_EMBEDDINGS = True

    comptime HEAD_DIM = Self.HIDDEN // Self.NUM_HEADS
    comptime KV_HIDDEN = Self.NUM_KV_HEADS * Self.HEAD_DIM
    comptime GQA_FACTOR = Self.NUM_HEADS // Self.NUM_KV_HEADS


# ===--- Layer ---=== #

struct SmolLM2Layer[E: Encoding, tp: Int]:
    comptime C = SmolLM2Config

    # Each PlacedSlot is self-describing: encoding, sharding, dims, offset, name
    comptime Q_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.HIDDEN, Self.C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight"]
    comptime K_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.KV_HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight"]
    comptime V_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.KV_HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight"]
    comptime O_PROJ      = PlacedSlot[Self.E, ColShard, Self.C.HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight"]
    comptime GATE_PROJ   = PlacedSlot[Self.E, RowShard, Self.C.INTERMEDIATE, Self.C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight"]
    comptime UP_PROJ     = PlacedSlot[Self.E, RowShard, Self.C.INTERMEDIATE, Self.C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight"]
    comptime DOWN_PROJ   = PlacedSlot[Self.E, ColShard, Self.C.HIDDEN, Self.C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight"]
    comptime INPUT_NORM  = PlacedSlot[F32, Replicated, Self.C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[F32, Replicated, Self.C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    # KV cache — runtime state, not loaded from file
    comptime K_CACHE = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.KV_HIDDEN, Self.tp]

    @staticmethod
    fn weight_bytes() -> Int:
        return Self.STRIDE

    @staticmethod
    fn cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()


# ===--- Full model ---=== #

struct SmolLM2[E: Encoding, tp: Int](Loadable):
    comptime C = SmolLM2Config
    comptime LAYER = SmolLM2Layer[Self.E, Self.tp]

    # Global weights — full tensor name in the type
    comptime EMBED      = PlacedSlot[Self.E, Replicated, Self.C.VOCAB_SIZE, Self.C.HIDDEN, Self.tp, 0, "model.embed_tokens.weight"]
    comptime FINAL_NORM = PlacedSlot[F32, Replicated, Self.C.HIDDEN, 1, Self.tp, next_offset[Self.EMBED](), "model.norm.weight"]

    # Structural constants for loader iteration
    comptime LAYERS_OFF   = next_offset[Self.FINAL_NORM]()
    comptime NUM_LAYERS   = Self.C.NUM_LAYERS
    comptime LAYER_STRIDE = Self.LAYER.STRIDE

    # State — runtime buffers, not loaded
    comptime ROPE_HALF = Self.C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, Self.C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, Self.C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.HIDDEN, Self.tp]
    comptime SCRATCH = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.INTERMEDIATE, Self.tp]
    comptime SCRATCH_COUNT = 3

    # State arena offsets (relative to state base)
    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + Self.C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_STRIDE = byte_count[Self.SCRATCH]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()

    # --- Sizing ---

    @staticmethod
    fn total_weight_bytes() -> Int:
        return Self.LAYERS_OFF + Self.C.NUM_LAYERS * Self.LAYER.STRIDE

    @staticmethod
    fn total_state_bytes() -> Int:
        return Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    @staticmethod
    fn total_arena_bytes() -> Int:
        return Self.total_weight_bytes() + Self.total_state_bytes()

    @staticmethod
    fn kv_cache_bytes() -> Int:
        return Self.C.NUM_LAYERS * Self.KV_STRIDE

    @staticmethod
    fn activation_bytes() -> Int:
        return byte_count[Self.X_MAIN]() + byte_count[Self.X_RESIDUAL]() + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE

    @staticmethod
    fn precomputed_bytes() -> Int:
        return byte_count[Self.ROPE_COS]() + byte_count[Self.ROPE_SIN]()

    # --- Loadable: weight descriptions ---

    @staticmethod
    fn global_weights() -> List[WeightDesc]:
        var w = List[WeightDesc]()
        w.append(weight_desc[Self.EMBED]())
        w.append(weight_desc[Self.FINAL_NORM]())
        return w^

    @staticmethod
    fn layer_weights(prefix: String, base: Int) -> List[WeightDesc]:
        comptime L = Self.LAYER
        var w = List[WeightDesc]()
        w.append(weight_desc[L.Q_PROJ](prefix, base))
        w.append(weight_desc[L.K_PROJ](prefix, base))
        w.append(weight_desc[L.V_PROJ](prefix, base))
        w.append(weight_desc[L.O_PROJ](prefix, base))
        w.append(weight_desc[L.GATE_PROJ](prefix, base))
        w.append(weight_desc[L.UP_PROJ](prefix, base))
        w.append(weight_desc[L.DOWN_PROJ](prefix, base))
        w.append(weight_desc[L.INPUT_NORM](prefix, base))
        w.append(weight_desc[L.POST_ATTN_NORM](prefix, base))
        return w^


# ===--- Lifetime-owning loaded model ---=== #

struct SmolLM2Loaded[E: Encoding, tp: Int](Movable):
    """Owns the arena memory for a loaded SmolLM2 model."""
    comptime M = SmolLM2[Self.E, Self.tp]

    var arena_ptr: Int
    var arena_size: Int

    fn __init__(out self, arena_ptr: Int, arena_size: Int):
        self.arena_ptr = arena_ptr
        self.arena_size = arena_size

    fn __moveinit__(out self, deinit other: Self):
        self.arena_ptr = other.arena_ptr
        self.arena_size = other.arena_size

    fn __del__(deinit self):
        if self.arena_ptr != 0 and self.arena_size > 0:
            var sys = linux.linux_sys()
            _ = sys.sys_munmap(self.arena_ptr, self.arena_size)

    @staticmethod
    fn load(path: Path, rank: Int = 0) -> Optional[Self]:
        var result = load_safetensors[Self.M](path, rank)
        if not result:
            return None
        var r = result.take()
        return Self(r.arena_ptr, r.arena_size)

    fn weight_base(self) -> Int:
        return self.arena_ptr

    fn state_base(self) -> Int:
        return self.arena_ptr + Self.M.total_weight_bytes()

    fn weight_ptr[T: Placed](self) -> Int:
        return self.weight_base() + T.OFFSET

    fn layer_weight_ptr[T: Placed](self, layer: Int) -> Int:
        return self.weight_base() + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE + T.OFFSET

    # State pointers
    fn kv_cache_ptr(self, layer: Int) -> Int:
        return self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    fn x_main_ptr(self) -> Int:
        return self.state_base() + Self.M.X_MAIN_OFF

    fn x_residual_ptr(self) -> Int:
        return self.state_base() + Self.M.X_RESIDUAL_OFF

    fn scratch_ptr(self, index: Int) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF + index * Self.M.SCRATCH_STRIDE

    fn rope_cos_ptr(self) -> Int:
        return self.state_base() + Self.M.ROPE_COS_OFF

    fn rope_sin_ptr(self) -> Int:
        return self.state_base() + Self.M.ROPE_SIN_OFF
