struct Gemma4BaseConfig:
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

    comptime VOCAB_SIZE = 262144
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime LOGIT_SOFTCAP = 30.0

    # Comptime ceiling on chunked-attention fan-out for full-attention layers.
    # Sizes dispatcher/merge stack arrays + cross-chunk partials buffer.
    # Must be >= any pool_capacity we will ever see at runtime.
    comptime FULL_ATTN_MAX_CHUNKS = 32


# =============================================================================
# Layer routing
# =============================================================================


@always_inline
def is_full_layer(layer_idx: Int) -> Bool:
    return (layer_idx + 1) % 6 == 0
