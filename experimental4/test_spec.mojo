from experimental4.model_spec import (
    Encoding, Shaped, BF16, I8, byte_count,
    Dims, Attention, GQA,
)
from experimental4.smollm2 import SmolLM2, SmolLM2Config


fn describe[T: Encoding & Shaped](name: String):
    print(name + ":", T.ROWS, "x", T.COLS, "=", byte_count[T](), "bytes", "(", T.DTYPE, ")")


fn print_attention_config[C: Dims & Attention & GQA]():
    """Only sees what it asks for — Dims, Attention, GQA."""
    print("Config:", C.HIDDEN, "hidden,", C.NUM_LAYERS, "layers,",
          C.NUM_HEADS, "heads, head_dim:", C.HEAD_DIM,
          "kv_hidden:", C.KV_HIDDEN, "gqa:", C.GQA_FACTOR)


fn print_model[E: Encoding, tp: Int]():
    comptime M = SmolLM2[E, tp]
    comptime L = M.LAYER

    print_attention_config[M.C]()
    print()

    print("Per-layer weights:")
    describe[L.Q_PROJ]("  q_proj")
    describe[L.K_PROJ]("  k_proj")
    describe[L.V_PROJ]("  v_proj")
    describe[L.O_PROJ]("  o_proj")
    describe[L.GATE_PROJ]("  gate_proj")
    describe[L.UP_PROJ]("  up_proj")
    describe[L.DOWN_PROJ]("  down_proj")
    describe[L.INPUT_NORM]("  input_norm")
    describe[L.POST_ATTN_NORM]("  post_attn_norm")
    print()

    print("Per-layer KV cache:")
    describe[L.K_CACHE]("  k_cache")
    describe[L.V_CACHE]("  v_cache")
    print()

    print("Globals:")
    describe[M.EMBED]("  embed")
    describe[M.FINAL_NORM]("  final_norm")
    print()

    print("Precomputed:")
    describe[M.ROPE_COS]("  rope_cos")
    describe[M.ROPE_SIN]("  rope_sin")
    print()

    print("Activations:")
    describe[M.X_MAIN]("  x_main")
    describe[M.X_RESIDUAL]("  x_residual")
    describe[M.SCRATCH]("  scratch (x" + String(M.SCRATCH_COUNT) + ")")
    print()

    print("--- Arena breakdown ---")
    print("  Weight bytes:      ", M.total_weight_bytes())
    print("  Precomputed bytes: ", M.precomputed_bytes())
    print("  KV cache bytes:    ", M.kv_cache_bytes())
    print("  Activation bytes:  ", M.activation_bytes())
    print("  Total arena bytes: ", M.total_arena_bytes())
    print()


fn main():
    print("========== SmolLM2-135M BF16 tp=1 ==========")
    print_model[BF16, 1]()

    print("========== SmolLM2-135M BF16 tp=2 ==========")
    print_model[BF16, 2]()

    print("========== SmolLM2-135M I8 tp=1 ==========")
    print_model[I8, 1]()
