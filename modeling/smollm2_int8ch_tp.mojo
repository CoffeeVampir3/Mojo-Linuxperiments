"""SmolLM2-135M int8 channelwise with VNNI-packed weights.

Model spec for the quantized model. Projection weights are I8 with
per-shape VNNI packing. Norms and embeddings remain BF16.

Packing parameters are chosen per-shape for optimal cache utilization:
  - K_BLOCK = K (full K in one block — no redundant outer iterations)
  - N_BLOCK = largest multiple of 32 that divides N and keeps tile in L2

Kernel ops are stubbed — this file defines the model layout and weight
descriptors. The forward pass will be hand-rolled here when kernels
are implemented.
"""

from std.pathlib import Path

from quant.quantizer import quantize as quantize_impl
from quant.channelwise import channelwise

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32, I8,
    RowShard, ColShard, Replicated,
    PrincipleNodeLocal,
    IsQuantizable, IsPassthrough, Unpacked,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
)
from kernels.vnni import VnniPacking


# =============================================================================
# Model config (same as bf16 version)
# =============================================================================

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

comptime C = SmolLM2Config


# =============================================================================
# Int8 channelwise layer spec — VNNI packed projections
#
# Per-shape packing:
#   q/o_proj   [576, 576]   -> VnniPacking[288, 576]   (162KB tile)
#   k/v_proj   [192, 576]   -> VnniPacking[192, 576]   (108KB tile)
#   gate/up    [1536, 576]  -> VnniPacking[384, 576]   (216KB tile)
#   down_proj  [576, 1536]  -> VnniPacking[96, 1536]   (144KB tile)
# =============================================================================

comptime PACK_QO   = VnniPacking[288, 576]
comptime PACK_KV   = VnniPacking[192, 576]
comptime PACK_GATE = VnniPacking[384, 576]
comptime PACK_DOWN = VnniPacking[96, 1536]


struct Int8TPLayer[tp: Int]:
    # Projection weights: I8, quantizable, VNNI packed
    comptime Q_PROJ      = PlacedSlot[I8, RowShard, C.HIDDEN, C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight", IsQuantizable, PACK_QO]
    comptime K_PROJ      = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight", IsQuantizable, PACK_KV]
    comptime V_PROJ      = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight", IsQuantizable, PACK_KV]
    comptime O_PROJ      = PlacedSlot[I8, ColShard, C.HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight", IsQuantizable, PACK_QO]
    comptime GATE_PROJ   = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight", IsQuantizable, PACK_GATE]
    comptime UP_PROJ     = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight", IsQuantizable, PACK_GATE]
    comptime DOWN_PROJ   = PlacedSlot[I8, ColShard, C.HIDDEN, C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight", IsQuantizable, PACK_DOWN]

    # Scale slots: F32, one per row, per quantized weight
    comptime Q_SCALE     = PlacedSlot[F32, RowShard, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "self_attn.q_proj.weight_scale"]
    comptime K_SCALE     = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.Q_SCALE](), "self_attn.k_proj.weight_scale"]
    comptime V_SCALE     = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.K_SCALE](), "self_attn.v_proj.weight_scale"]
    comptime O_SCALE     = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.V_SCALE](), "self_attn.o_proj.weight_scale"]
    comptime GATE_SCALE  = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.O_SCALE](), "mlp.gate_proj.weight_scale"]
    comptime UP_SCALE    = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.GATE_SCALE](), "mlp.up_proj.weight_scale"]
    comptime DOWN_SCALE  = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.UP_SCALE](), "mlp.down_proj.weight_scale"]

    # Norms: BF16, passthrough, unpacked
    comptime INPUT_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_SCALE](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    # KV cache (bf16, same as source model)
    comptime K_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base)
        func[Self.K_PROJ](prefix, base)
        func[Self.V_PROJ](prefix, base)
        func[Self.O_PROJ](prefix, base)
        func[Self.GATE_PROJ](prefix, base)
        func[Self.UP_PROJ](prefix, base)
        func[Self.DOWN_PROJ](prefix, base)
        func[Self.Q_SCALE](prefix, base)
        func[Self.K_SCALE](prefix, base)
        func[Self.V_SCALE](prefix, base)
        func[Self.O_SCALE](prefix, base)
        func[Self.GATE_SCALE](prefix, base)
        func[Self.UP_SCALE](prefix, base)
        func[Self.DOWN_SCALE](prefix, base)
        func[Self.INPUT_NORM](prefix, base)
        func[Self.POST_ATTN_NORM](prefix, base)

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()


struct Int8TPModel[tp: Int](WeightIterable):
    comptime LAYER = Int8TPLayer[Self.tp]

    comptime LAYERS_OFF = 0
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = C.NUM_LAYERS * Self.LAYER.STRIDE

    comptime LOCAL_HEADS = C.NUM_HEADS // Self.tp
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp

    # Activation slots (bf16 — activations are not quantized)
    comptime ROPE_HALF = C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    comptime SCRATCH = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime SCRATCH_COUNT = 3

    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime KV_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]

    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_STRIDE = byte_count[Self.SCRATCH]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # NodeLocal weights (host arena)
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, C.HIDDEN, 1, Self.tp, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[BF16, PrincipleNodeLocal, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ]():
        comptime for i in range(C.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.LAYERS_OFF + i * Self.LAYER_STRIDE
            Self.LAYER.for_each_weight[func](prefix, base)
        func[Self.FINAL_NORM]("", 0)
        func[Self.EMBED]("", 0)

    @staticmethod
    def arena_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    def host_arena_bytes() -> Int:
        return next_offset[Self.EMBED]()

    @staticmethod
    def quantize(source_path: Path, output_path: Path, num_workers: Int = 4) -> Bool:
        return quantize_impl[Self](
            pipeline=channelwise[source=DType.bfloat16, target=DType.int8],
            weight_dtype=DType.int8,
            weight_element_bits=8,
            scale_dtype=DType.float32,
            scale_element_bits=32,
            source_path=source_path,
            output_path=output_path,
            num_workers=num_workers,
        )
