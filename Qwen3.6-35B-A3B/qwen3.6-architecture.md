# Qwen3.6-35B-A3B Architecture Reference

This document describes the shipped Qwen3.6-35B-A3B release artifacts as they are packaged: the serialized model config, the safetensors weight index, the tokenizer and chat template, and the converging behavior across the four reference implementations (the HuggingFace repo, the first-party Transformers `qwen3_5_moe` module, vLLM's `qwen3_5` / `qwen3_5_mtp` integration, and SGLang's `qwen3_5` / `qwen3_5_mtp` integration). It separates exact observed behavior from metadata that is present in the release but not enforced by every runtime.

The shipped checkpoint reuses the Qwen3.5 modeling code (`model_type: "qwen3_5_moe"`, architectures `["Qwen3_5MoeForConditionalGeneration"]`); the "3.6" version label is a release/post-training distinction, not a new model class. Every cross-reference in this document points at the `qwen3_5_moe` / `qwen3_5` modules in upstream Transformers, and at the `qwen3_5*.py` files in vLLM and SGLang.

## 1. Model Summary

Qwen3.6-35B-A3B is a multimodal causal language model with a sparse Mixture-of-Experts text decoder (35B total / ~3B activated parameters per token) and a dense ViT-style vision encoder. The text decoder is a hybrid attention stack interleaving Gated DeltaNet (linear attention + SSM-style recurrent state) layers with gated softmax-attention (GQA) layers. A single Multi-Token Prediction module is shipped after the main stack.

- 40 base text decoder layers in a fixed pattern of `3 × linear_attention -> 1 × full_attention`, repeated 10 times. Layers 3, 7, 11, ..., 39 are full attention (10 layers). Layers 0, 1, 2, 4, 5, 6, ..., 38 are linear attention (30 layers).
- 1 MTP layer appended after the main stack at prefix `mtp.*`. Its embedded decoder runs full attention regardless of the position-in-pattern rule.
- Residual width 2048.
- Full-attention layers: 16 query heads, 2 KV heads (GQA factor 8), per-head `head_dim = 256`. Q projection is twice as wide as K/V because each Q head emits a 256-wide query plus a 256-wide sigmoid output gate (the Qwen3-Next "gated attention" pattern).
- Partial RoPE on the leading 64 dimensions (`partial_rotary_factor = 0.25`) of each query / key head. RoPE is multimodal (M-RoPE) with three position axes (text/temporal/height/width) and `mrope_section = [11, 11, 10]`.
- Linear-attention layers: Gated DeltaNet with 32 value heads and 16 K/Q heads (V is broadcast / Q,K are repeated 2x to match), per-head `key_dim = value_dim = 128`. A depthwise causal Conv1d of kernel size 4 sits between the input projections and the QKV split. Per-head time-step parameters `A_log` and `dt_bias` discretize the SSM update.
- Sparse MoE on every layer (linear and full): 256 routed experts plus 1 always-on shared expert, top-8 routing, softmax-then-renormalize routing weights (no auxiliary bias), `moe_intermediate_size = 512`, `shared_expert_intermediate_size = 512`. The shared expert is itself sigmoid-gated by a learned 1-wide projection.
- Routed experts are stored as 3D fused tensors per layer: `gate_up_proj` of shape `(256, 1024, 2048)` and `down_proj` of shape `(256, 2048, 512)`. There are no per-expert sub-modules on disk.
- Untied token embedding and LM head; vocabulary 248,320 tokens. RMSNorm uses the `(1 + weight)` zero-centered convention (the GLM/Qwen3-style "GemmaRMSNorm").
- Vision encoder: 27-block ViT with `hidden_size = 1152`, intermediate 4304, 16 heads, GELU-pytorch-tanh activation, patch size 16, temporal patch size 2, spatial merge size 2, merger output 2048 (so vision tokens land directly in the text residual width).
- Native context length 262,144 tokens; the README quotes extension up to ~1,010,000 tokens via runtime RoPE scaling (no scaling tensors shipped).
- Released weights are bfloat16 throughout. There is no quantization config in this release.

## 2. Serialized `config.json`

The shipped `config.json` is a nested config (`Qwen3_5MoeConfig`) with `text_config` (`Qwen3_5MoeTextConfig`) and `vision_config` (`Qwen3_5MoeVisionConfig`) sub-blocks.

### Top-level fields

| Field | Value |
| --- | --- |
| `architectures` | `["Qwen3_5MoeForConditionalGeneration"]` |
| `model_type` | `"qwen3_5_moe"` |
| `image_token_id` | `248056` |
| `video_token_id` | `248057` |
| `vision_start_token_id` | `248053` |
| `vision_end_token_id` | `248054` |
| `tie_word_embeddings` | `false` |
| `transformers_version` | `"4.57.1"` |

### `text_config` fields

| Field | Value |
| --- | --- |
| `model_type` | `"qwen3_5_moe_text"` |
| `dtype` | `"bfloat16"` |
| `vocab_size` | `248320` |
| `hidden_size` | `2048` |
| `num_hidden_layers` | `40` |
| `num_attention_heads` | `16` |
| `num_key_value_heads` | `2` |
| `head_dim` | `256` |
| `attn_output_gate` | `true` |
| `attention_bias` | `false` |
| `attention_dropout` | `0.0` |
| `hidden_act` | `"silu"` |
| `rms_norm_eps` | `1e-6` |
| `max_position_embeddings` | `262144` |
| `partial_rotary_factor` | `0.25` |
| `rope_parameters.rope_type` | `"default"` |
| `rope_parameters.rope_theta` | `10000000` |
| `rope_parameters.partial_rotary_factor` | `0.25` |
| `rope_parameters.mrope_interleaved` | `true` |
| `rope_parameters.mrope_section` | `[11, 11, 10]` |
| `full_attention_interval` | `4` |
| `layer_types` | 40-entry list, see section 3 |
| `linear_conv_kernel_dim` | `4` |
| `linear_key_head_dim` | `128` |
| `linear_value_head_dim` | `128` |
| `linear_num_key_heads` | `16` |
| `linear_num_value_heads` | `32` |
| `mamba_ssm_dtype` | `"float32"` |
| `num_experts` | `256` |
| `num_experts_per_tok` | `8` |
| `moe_intermediate_size` | `512` |
| `shared_expert_intermediate_size` | `512` |
| `output_router_logits` | `false` |
| `router_aux_loss_coef` | `0.001` |
| `mtp_num_hidden_layers` | `1` |
| `mtp_use_dedicated_embeddings` | `false` |
| `tie_word_embeddings` | `false` |
| `bos_token_id` | `248044` |
| `eos_token_id` | `248044` |
| `pad_token_id` | `null` |
| `use_cache` | `true` |

Derived shape facts from the text config:

- Full-attention `q_proj`: `2048 -> 16 * 256 * 2 = 8192` (twice the head-dim because of the output gate; chunk(2) into `query_states` and `gate`).
- Full-attention `k_proj`: `2048 -> 2 * 256 = 512`.
- Full-attention `v_proj`: `2048 -> 2 * 256 = 512`.
- Full-attention `o_proj`: `16 * 256 = 4096 -> 2048`.
- Full-attention RoPE: leading `0.25 * 256 = 64` dims of each Q/K head are rotated; the trailing 192 dims are unrotated.
- Full-attention per-head Q/K RMSNorm over `head_dim = 256`.
- Linear attention `key_dim = 16 * 128 = 2048` and `value_dim = 32 * 128 = 4096`. After QKV split through the conv, Q and K are repeated 2x along the head axis (`num_v_heads // num_k_heads = 2`) to match the value head count.
- Linear attention `in_proj_qkv`: `2048 -> 2 * 2048 + 4096 = 8192` (then conv1d over the 8192-wide sequence and split into Q, K, V of widths 2048, 2048, 4096).
- Linear attention `in_proj_z`: `2048 -> 4096`. The `z` stream feeds the gated RMSNorm.
- Linear attention `in_proj_b`, `in_proj_a`: `2048 -> num_v_heads = 32` each (per-head SSM gate inputs).
- Linear attention `conv1d`: depthwise (`groups = conv_dim = 8192`), kernel 4, bias-free.
- Linear attention `out_proj`: `value_dim = 4096 -> 2048`.
- Routed-expert `gate_up_proj` (3D): `(256 experts, 2 * 512 = 1024, 2048)`.
- Routed-expert `down_proj` (3D): `(256 experts, 2048, 512)`.
- Shared-expert `gate_proj` and `up_proj`: `(512, 2048)` each. `down_proj`: `(2048, 512)`.
- Router `gate.weight`: `(256 experts, 2048)`. No bias term.
- `shared_expert_gate.weight`: `(1, 2048)`, biasless.

### `vision_config` fields

| Field | Value |
| --- | --- |
| `model_type` | `"qwen3_5_moe"` |
| `depth` | `27` |
| `hidden_size` | `1152` |
| `intermediate_size` | `4304` |
| `num_heads` | `16` |
| `out_hidden_size` | `2048` |
| `hidden_act` | `"gelu_pytorch_tanh"` |
| `in_channels` | `3` |
| `patch_size` | `16` |
| `temporal_patch_size` | `2` |
| `spatial_merge_size` | `2` |
| `num_position_embeddings` | `2304` |
| `deepstack_visual_indexes` | `[]` |

Derived vision shapes:

- `patch_embed.proj`: `Conv3d(in=3, out=1152, kernel=(2, 16, 16), stride=(2, 16, 16))` (temporal patch size 2 over height-width 16).
- `pos_embed`: `nn.Embedding(2304, 1152)` for the 2D learned positional table that the model interpolates against the actual token grid.
- Per vision block (27 of them):
  - `norm1` and `norm2`: `LayerNorm(1152)` each, weight + bias.
  - `attn.qkv`: `Linear(1152 -> 3 * 1152, bias=true)`.
  - `attn.proj`: `Linear(1152 -> 1152, bias=true)`.
  - `mlp.linear_fc1`: `Linear(1152 -> 4304, bias=true)`.
  - `mlp.linear_fc2`: `Linear(4304 -> 1152, bias=true)`.
- `merger.norm`: `LayerNorm(1152 * spatial_merge_size**2 = 4608)`, weight + bias.
- `merger.linear_fc1`: `Linear(4608 -> 4608, bias=true)`.
- `merger.linear_fc2`: `Linear(4608 -> out_hidden_size = 2048, bias=true)`.

`deepstack_visual_indexes = []` disables Qwen3-VL's deepstack feature merge (the per-layer feature taps that some other Qwen3-VL variants use). The Qwen3.5 Transformers implementation explicitly deletes the `deepstack_*` machinery from the parent class.

## 3. Layer Pattern

The shipped `text_config.layer_types` is a 40-entry list:

```
linear_attention, linear_attention, linear_attention, full_attention,    # block 0
linear_attention, linear_attention, linear_attention, full_attention,    # block 1
...
linear_attention, linear_attention, linear_attention, full_attention,    # block 9
```

Equivalently, layer `i` is `full_attention` iff `(i + 1) % full_attention_interval == 0` with `full_attention_interval = 4`. There are 10 full-attention layers (indices 3, 7, 11, ..., 39) and 30 linear-attention layers (everything else).

If the field were absent, `Qwen3_5MoeTextConfig.__post_init__` would synthesize the same list from `full_attention_interval = 4`. The shipped checkpoint pins both, so either path produces the same pattern.

## 4. Top-Level Module Structure

The packaged conditional-generation entry point has this structure (matching the on-disk tensor names):

1. `model.visual.*`: 27-block ViT vision encoder (patch_embed, pos_embed, 27 blocks, merger).
2. `model.language_model.embed_tokens.weight`: `(248320, 2048)` token embedding.
3. `model.language_model.layers`: 40 base decoder layers (a mix of linear-attn and full-attn variants).
4. `model.language_model.norm.weight`: final RMSNorm gamma.
5. `lm_head.weight`: `(248320, 2048)` output projection. Untied from `embed_tokens`.
6. `mtp.*`: Multi-Token Prediction module (one decoder layer plus three RMSNorms and a fusion projection).

`tie_word_embeddings` is false at every nesting level. `lm_head.weight` is its own bf16 tensor.

The `Qwen3_5MoeForConditionalGeneration` wrapper has:

- `self.model = Qwen3_5MoeModel(config)` containing both `visual` and `language_model`.
- `self.lm_head = ParallelLMHead(...)` (vLLM/SGLang) or `nn.Linear(hidden, vocab, bias=False)` (Transformers).

The Transformers `Qwen3_5MoeForCausalLM` (text-only convenience class) declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`. The MTP and vision tensors are present on disk but are silently dropped when the user constructs the text-only causal LM rather than the full conditional-generation wrapper.

Each base decoder layer has two pre-norm residual sub-blocks:

1. `input_layernorm` -> token-mixer (`linear_attn` for linear-attention layers, `self_attn` for full-attention layers) -> residual add.
2. `post_attention_layernorm` -> sparse MoE block (`mlp.gate`, `mlp.experts`, `mlp.shared_expert`, `mlp.shared_expert_gate`) -> residual add.

There is no dense MLP fallback for the early layers; every layer is MoE.

## 5. Full-Attention Block (Gated GQA + M-RoPE)

Full-attention layers (indices 3, 7, 11, ..., 39 plus the MTP layer) use grouped-query softmax attention with a sigmoid output gate.

### Projection Shapes

- `self_attn.q_proj.weight`: `(8192, 2048)`, biasless. The 8192-wide output is `chunk(2, dim=-1)`-split into `query_states` and `gate`, each `(B, S, 16, 256)`.
- `self_attn.k_proj.weight`: `(512, 2048)`, biasless. Reshapes to `(B, S, 2, 256)`.
- `self_attn.v_proj.weight`: `(512, 2048)`, biasless.
- `self_attn.o_proj.weight`: `(2048, 4096)`, biasless.
- `self_attn.q_norm.weight`: RMSNorm over the per-head dimension (256). Applied to each query head independently.
- `self_attn.k_norm.weight`: RMSNorm over the per-head dimension (256). Applied to each key head independently.

Q/K RMSNorm is per-head (not per-token), and uses the `(1 + weight)` Qwen-style centering (`weight` is initialized to zero so the norm starts as the identity). The `head_dim` in the config is `256`, distinct from the `partial_rotary_factor`-derived rotary subspace size of `64`.

### Multimodal RoPE (M-RoPE)

- `rope_theta = 10_000_000`.
- `partial_rotary_factor = 0.25`, so the leading `64 = 0.25 * 256` dimensions of each Q/K head are rotated; the trailing 192 dims pass through unchanged.
- M-RoPE uses three position axes (temporal, height, width) plus an implicit textual axis. The `mrope_section = [11, 11, 10]` partitions the 32 rotary half-frequencies (= 64 / 2) across the three axes. With `mrope_interleaved = true`, the per-axis cosines/sines are interleaved across the rotary dims rather than concatenated as contiguous blocks.
- For text-only inputs, the text axis carries the integer position; for image / video inputs, the temporal/height/width axes are filled by the model's `compute_3d_position_ids` helper based on `image_grid_thw` / `video_grid_thw` from the processor.
- Inverse frequencies are computed once in float32; cosines and sines are cast back to the hidden-state dtype before application.

### Attention Math

For each layer and token block:

1. Project Q to `(B, S, 16 heads, 256)` query and `(B, S, 16, 256)` gate halves; split via `chunk(2, dim=-1)`.
2. Per-head RMSNorm `q_norm` on the query halves.
3. Project K to `(B, S, 2, 256)` and apply per-head RMSNorm `k_norm`.
4. Project V to `(B, S, 2, 256)`.
5. Apply M-RoPE to the leading 64 dims of Q and K. Trailing 192 dims pass through.
6. Update the KV cache (full-precision standard cache; no sliding window).
7. Repeat-interleave K and V from 2 KV heads to 16 attention heads (GQA factor 8). FlashAttention / SDPA backends consume the GQA layout directly.
8. Compute scaled-dot-product attention with scaling `1 / sqrt(256)` and a causal mask.
9. Reshape attention output to `(B, S, 4096)`.
10. Multiply elementwise by `sigmoid(gate)` (the Qwen3-Next gated-attention output gate).
11. Project through `o_proj` back to width 2048.
12. Add the result to the residual stream.

The shipped configuration does not enable sliding-window attention. There is no attention sink, no DSA-style indexer, no K/V LoRA, and no MLA latent compression.

## 6. Linear-Attention Block (Gated DeltaNet)

Linear-attention layers (the 30 non-multiple-of-4 layer indices) use Qwen3.5 / Qwen3-Next's `Gated DeltaNet`: a Mamba2-style SSM with a depthwise causal conv front-end, learned per-head decay, and a sigmoid-gated multiplicative beta term.

### Projection Shapes

- `linear_attn.in_proj_qkv.weight`: `(8192, 2048)`, biasless. Stacks Q (2048), K (2048), V (4096) for joint conv processing.
- `linear_attn.in_proj_z.weight`: `(4096, 2048)`, biasless. The z stream is the gate input to `RMSNormGated`.
- `linear_attn.in_proj_b.weight`: `(32, 2048)`, biasless. Per-V-head pre-sigmoid beta.
- `linear_attn.in_proj_a.weight`: `(32, 2048)`, biasless. Per-V-head pre-softplus a (combines with `dt_bias` for the SSM time step).
- `linear_attn.conv1d.weight`: `(8192, 1, 4)` depthwise conv (`groups = 8192`, kernel 4, padding 3, biasless on disk; `bias` exists in the module but is None / zero for this checkpoint).
- `linear_attn.A_log`: `(32,)` float32 per-head learned log decay.
- `linear_attn.dt_bias`: `(32,)` float32 per-head time-step bias.
- `linear_attn.norm.weight`: `(128,)` per-head-dim RMSNorm gamma for the gated norm `RMSNormGated(head_v_dim=128)`. Uses `(1 + weight)` centering.
- `linear_attn.out_proj.weight`: `(2048, 4096)`, biasless.

The Qwen3.5 implementation deliberately splits the four input projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) instead of the fused `in_proj_qkvz` + `in_proj_ba` used by Qwen3-Next. The on-disk tensor names match this split.

### Forward Pass

For each layer and token block (training / prefill path):

1. Apply the attention mask to padding positions to zero out their hidden states (`apply_mask_to_padding_states`).
2. `mixed_qkv = in_proj_qkv(x)`, transpose to `(B, 8192, S)` for the conv.
3. `z = in_proj_z(x).reshape(B, S, num_v_heads = 32, head_v_dim = 128)`.
4. `b = in_proj_b(x)`, `a = in_proj_a(x)`. Both shape `(B, S, 32)`.
5. Run depthwise causal Conv1d (kernel 4, SiLU activation) over `mixed_qkv`. Use `causal_conv1d_fn` if available; otherwise fall back to `F.silu(F.conv1d(x, weight, ...)[:, :, :S])`.
6. Transpose back and split `mixed_qkv` along the channel axis into `query (2048)`, `key (2048)`, `value (4096)`.
7. Reshape: `query` to `(B, S, 16, 128)`, `key` to `(B, S, 16, 128)`, `value` to `(B, S, 32, 128)`. Repeat-interleave Q and K along the head axis by `num_v_heads / num_k_heads = 2` to align with V's 32 heads.
8. `beta = sigmoid(b)`.
9. `g = -exp(A_log) * softplus(a + dt_bias)` — the per-token, per-head SSM gate.
10. Run the chunked gated delta-rule kernel (`chunk_gated_delta_rule`) with `use_qk_l2norm_in_kernel=True`. The kernel internally applies L2 normalization to Q and K along the head dimension before the recurrence, scales Q by `1 / sqrt(head_k_dim)`, processes 64-token chunks, and produces both the per-position output and the final recurrent state.
11. Apply `RMSNormGated(out, z)` on a per-head-dim basis (the gated norm consumes both the SSM output and the z stream). The norm operates over the 128-wide head-V dimension and applies the SiLU-gated multiplication that the kernel name implies.
12. Reshape to `(B, S, 4096)` and project through `out_proj` back to `2048`.
13. Add the result to the residual stream.

For decode (single-token, with cache):

- Reuse the previous conv state via `causal_conv1d_update` (no full Conv1d evaluation).
- Reuse the previous recurrent state via `recurrent_gated_delta_rule` (one-token recurrent step instead of the chunked path).
- Update both states in the per-layer cache slot.

The cache shape per layer is the conv state plus a `(num_v_heads = 32, head_k_dim = 128, head_v_dim = 128)` recurrent state — i.e. fixed-size in sequence length, unlike the standard attention KV cache.

The Qwen3.5 reference deletes Qwen3-Next's `fix_query_key_value_ordering` helper (it is not needed when the four input projections are split rather than fused), and it switches the Mamba norm to the FLA fused gated norm if available.

## 7. Sparse MoE Block

Every base layer (linear and full attention alike) and the MTP layer's embedded decoder use the same sparse MoE block. The block has three components: a router, a 3D-fused expert table, a gated shared expert.

### Router

- Router input width: 2048.
- Router output width: 256 experts.
- `mlp.gate.weight` shape: `(256, 2048)`, dtype bf16 on disk.
- No router bias term and no `e_score_correction_bias` (this is the Qwen3-VL-MoE "TopKRouter" recipe, not the DeepSeek `noaux_tc` recipe).

The routing logic is:

1. Reshape hidden states to `(B*S, 2048)`.
2. `router_logits = F.linear(hidden_states, gate.weight)`.
3. `router_logits = softmax(router_logits, dim=-1, dtype=float32)`.
4. `router_top_value, router_indices = topk(router_logits, k = num_experts_per_tok = 8)`.
5. Renormalize: `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)` (always; there is no `norm_topk_prob` switch to disable it).
6. Cast `router_top_value` back to the hidden-state dtype.
7. Return `(router_logits, router_scores = router_top_value, router_indices)`.

The Qwen3.5 MoE class (`Qwen3_5MoeSparseMoeBlock` extending `Qwen3NextSparseMoeBlock`) overrides the inherited router class with `Qwen3VLMoeTextTopKRouter` (softmax + always-renormalize) instead of Qwen3-Next's optional-renormalize router. Routing is in float32; experts run in bf16.

### Routed Experts (3D-fused)

Each layer ships routed-expert weights as two 3D tensors rather than 256 separate sub-modules:

- `mlp.experts.gate_up_proj`: `(256 experts, 2 * 512 = 1024, 2048)`.
- `mlp.experts.down_proj`: `(256 experts, 2048, 512)`.

Per-token expert computation for selected expert `e`:

```
gate, up = chunk(linear(x, gate_up_proj[e]), 2, dim=-1)   # both (..., 512)
y_e = linear(silu(gate) * up, down_proj[e])               # (..., 2048)
```

The expert loop iterates over the experts that were selected by at least one token in the batch, gathers those tokens, runs the expert MLP, multiplies by the per-token routing weight, and `index_add`-scatters back into the output buffer. vLLM and SGLang use their `FusedMoE` kernels and split the gate/up halves at load time (see section 12).

### Shared Expert

Every MoE block carries one shared SwiGLU MLP with the same intermediate width as a single routed expert:

- `mlp.shared_expert.gate_proj.weight`: `(512, 2048)`.
- `mlp.shared_expert.up_proj.weight`: `(512, 2048)`.
- `mlp.shared_expert.down_proj.weight`: `(2048, 512)`.

And a sigmoid mixing gate:

- `mlp.shared_expert_gate.weight`: `(1, 2048)`, biasless.

Combination:

```
shared = sigmoid(shared_expert_gate(x)) * shared_expert(x)
y = sum_i routing_weights[i] * routed_expert[indices[i]](x) + shared
```

The shared expert always runs; the sigmoid scalar gate gives the model a way to suppress its contribution when the routed mixture suffices. This differs from DeepSeek-V3 and GLM-5.1, where the shared expert is unconditionally added without a gate.

### Auxiliary Loss

`router_aux_loss_coef = 0.001` is present in the config; `output_router_logits = false` by default. The router-logits auxiliary load-balancing loss is only assembled by the Transformers causal-LM wrapper when the user opts in. Inference servers ignore it.

## 8. Multi-Token Prediction Module

The release ships one MTP module at on-disk prefix `mtp.*` (note: not `model.layers.40` and not `model.mtp` — it lives directly at the top of the safetensors namespace).

Tensors:

- `mtp.fc.weight`: `(2048, 4096)`. Combines the previous-token hidden stream and the next-token embedding stream after concatenation.
- `mtp.pre_fc_norm_embedding.weight`: RMSNorm gamma over the 2048-wide next-token embedding before the concat.
- `mtp.pre_fc_norm_hidden.weight`: RMSNorm gamma over the 2048-wide previous-token hidden before the concat.
- `mtp.norm.weight`: RMSNorm gamma applied after the embedded decoder layer, before the (shared) LM head.
- `mtp.layers.0.*`: a complete decoder layer with `self_attn` (full-attention; same shapes as a regular full-attention layer) and `mlp` (a complete sparse MoE block: 256 routed experts, 1 shared expert, router gate, shared-expert gate). Both layernorms (`input_layernorm`, `post_attention_layernorm`) are present.

The MTP layer is full-attention regardless of the linear/full pattern in `text_config.layer_types`. Both vLLM and SGLang force this when constructing the MTP wrapper (vLLM passes `layer_type="full_attention"`; SGLang sets `config.full_attention_interval = 1` before calling `Qwen3_5ForCausalLM(is_nextn=True)`).

`mtp_use_dedicated_embeddings = false` means the MTP shares `model.language_model.embed_tokens` with the main model. There is no `mtp.embed_tokens.weight` on disk. The MTP also reuses the global `lm_head` for its speculative-token logits.

Forward pass (one MTP step):

1. Embed the candidate next-token id through the shared `embed_tokens` to get `e` of shape `(B, S, 2048)`.
2. `e = pre_fc_norm_embedding(e)`; `h = pre_fc_norm_hidden(prev_hidden_state)`.
3. `concat = cat([e, h], dim=-1)` of shape `(B, S, 4096)`.
4. `hidden = fc(concat)` back to `(B, S, 2048)`.
5. Run through the embedded full-attention + MoE decoder layer (uses the shared rotary embedding cache and the running KV cache).
6. `hidden = norm(hidden)`.
7. `logits = lm_head(hidden)`.

Loading behavior:

- The first-party Transformers `Qwen3_5MoeForCausalLM` declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`. Constructing the text-only causal LM silently drops both the MTP and vision tensors.
- The Transformers `Qwen3_5MoeForConditionalGeneration` (the multimodal wrapper) keeps the vision tensors but does not contain an MTP submodule; the MTP weights are still dropped.
- vLLM exposes the MTP through a separate `Qwen3_5MultiTokenPredictor` class (`vllm/model_executor/models/qwen3_5_mtp.py`) and instantiates it on demand for speculative decoding. The constructor sets `mtp_start_layer_idx = num_hidden_layers` and reads `mtp_num_hidden_layers` (= 1 here) for how many MTP slots to allocate.
- SGLang exposes it through `Qwen3_5ForCausalLMMTP` (`python/sglang/srt/models/qwen3_5_mtp.py`). It also has special handling for nvfp4-quantized checkpoints that store `mtp.fc` in bf16 instead of fp4 (the same workaround vLLM uses).

The MTP weights are real and consume real storage; whether they are exercised at inference is a runtime decision.

## 9. Vision Encoder

The vision encoder follows the Qwen3-VL ViT pattern with a single-stage 27-block transformer plus a spatial-merge MLP that produces text-residual-width embeddings.

### Patch Embedding

`patch_embed.proj` is a `Conv3d(in_channels=3, out_channels=1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))` with bias. Each `(2, 16, 16)` chunk of the input video tensor produces one 1152-wide token. Single-image inputs are handled by repeating the image along the temporal axis to match `temporal_patch_size = 2`.

### Positional Embedding

`pos_embed` is a learned `nn.Embedding(2304, 1152)` table. The forward pass interpolates this table over the actual `(grid_t, grid_h, grid_w)` shape of each image / video using `fast_pos_embed_interpolate`.

### Vision Block

Each of the 27 blocks is a standard pre-norm transformer block:

1. `norm1` (LayerNorm 1152, weight + bias) -> attention.
2. Attention: `qkv` linear `(1152 -> 3 * 1152)` with bias; reshape to 16 heads of dim 72; rotary positional embedding produced from `Qwen3_5MoeVisionRotaryEmbedding`; concatenated cu_seqlens-based attention; `proj` linear `(1152 -> 1152)` with bias.
3. Residual add.
4. `norm2` (LayerNorm 1152, weight + bias) -> MLP.
5. MLP: `linear_fc1 (1152 -> 4304)` -> `gelu_pytorch_tanh` -> `linear_fc2 (4304 -> 1152)`, both with bias.
6. Residual add.

The vision rotary embedding uses a fixed base frequency, computed at construction time and stored as `inv_freq`. Vision RoPE is *not* M-RoPE; the multimodal axes are introduced by the text-side `Qwen3_5MoeTextRotaryEmbedding`, which consumes 3D position ids covering vision-token spans.

### Merger

The final visual hidden state is reshaped from per-patch `(N, 1152)` to per-spatial-merge-cell `(N / spatial_merge_size**2, 1152 * spatial_merge_size**2 = 4608)`, run through `merger.norm` (LayerNorm 4608) -> `merger.linear_fc1 (4608 -> 4608)` -> `gelu_pytorch_tanh` -> `merger.linear_fc2 (4608 -> 2048)`. The `2048` width matches the text residual stream so the merged vision tokens scatter directly into `inputs_embeds` at the `image_token_id` / `video_token_id` placeholder positions.

### DeepStack

`deepstack_visual_indexes = []` disables the per-layer feature merger that some other Qwen3-VL variants use. The Qwen3.5 vision class explicitly deletes `deepstack_visual_indexes` and `deepstack_merger_list` from the parent `Qwen3VLVisionModel`. There are no `deepstack_*` tensors on disk.

## 10. Forward Pass Summary

For input shape `(B, T)` (text-only, no images / videos):

1. Token lookup produces hidden states of shape `(B, T, 2048)`.
2. `compute_3d_position_ids(input_ids, ...)` produces `(4, B, T)` position ids covering the text axis plus three multimodal axes (the multimodal axes degenerate to zeros for text-only input).
3. `Qwen3_5MoeTextRotaryEmbedding` produces M-RoPE `(cos, sin)` tensors from those position ids.
4. The hidden states pass through 40 base decoder layers in pattern order. Each linear-attention layer reads / writes its own conv + recurrent state in the cache; each full-attention layer reads / writes its own `(K, V)` cache slot.
5. Each layer performs:
   - input RMSNorm -> token mixer (linear attn or gated-GQA full attn) -> residual add.
   - post-attention RMSNorm -> sparse MoE (router + top-8 routed experts + sigmoid-gated shared expert) -> residual add.
6. Final RMSNorm produces the last hidden state.
7. `lm_head` projects to logits of shape `(B, T, 248320)`.
8. (Optional) The MTP module at `mtp.*` is invoked separately by speculative-decoding wrappers in vLLM and SGLang to produce a one-step-ahead logit head. The Transformers `Qwen3_5MoeForCausalLM` skips it.

For multimodal input:

1. Vision tokens are ingested through `model.visual` (patch embed -> 27 blocks -> merger), producing a `(N_vision, 2048)` tensor that is `masked_scatter`'d into `inputs_embeds` at the positions marked by `image_token_id = 248056` or `video_token_id = 248057`.
2. The multimodal axes of the position ids are filled in based on `image_grid_thw` / `video_grid_thw` produced by the processor.
3. The text decoder runs as above on the merged `inputs_embeds`.

## 11. Quantization Story

The shipped `dtype` is `bfloat16`. There is no `quantization_config` block in `config.json`. All large linear weights, expert tensors, conv weights, and normalization gammas are stored as bf16. The float-32 fields are limited to:

- `linear_attn.A_log`: float32 per-head log decay.
- `linear_attn.dt_bias`: float32 per-head time-step bias.
- `mamba_ssm_dtype = "float32"`: the SSM internal state runs in float32 in the FLA kernel even when the surrounding linear weights are bf16.
- Router scoring is computed in float32 by both the Transformers and vLLM/SGLang implementations (`softmax(..., dtype=torch.float32)`), then cast back.
- RMSNorm forward computes in float32 then casts back.

The vLLM MTP wrapper has a workaround for nvfp4-quantized variants that places `mtp.fc` in bf16 (forced unquantized) because the quant-config exclude list omits it. This release does not exercise that path because the stock checkpoint is bf16 throughout.

## 12. Weight Inventory

The safetensors weight index reports:

- `metadata.total_size`: `71_903_645_408` bytes (~67 GiB at bf16).
- `1045` named tensors across 26 safetensors shards.

Per-component tensor counts (verified directly from `model.safetensors.index.json`):

| Component | Count |
| --- | --- |
| `model.language_model.*` | 692 |
| `model.visual.*` | 333 |
| `mtp.*` | 19 |
| `lm_head.weight` | 1 |
| **Total** | **1,045** |

Per-layer text-decoder breakdown:

| Layer family | Layers | Tensors per layer | Subtotal |
| --- | --- | --- | --- |
| Linear attention + MoE | 30 | 18 | 540 |
| Full attention + MoE | 10 | 15 | 150 |
| Top-level (`embed_tokens`, `norm`) | - | 2 | 2 |
| **Language-model subtotal** | - | - | **692** |

A linear-attention layer's 18 tensors:

- 1 `input_layernorm.weight`, 1 `post_attention_layernorm.weight`.
- 9 GDN tensors: `linear_attn.{A_log, conv1d.weight, dt_bias, in_proj_a.weight, in_proj_b.weight, in_proj_qkv.weight, in_proj_z.weight, norm.weight, out_proj.weight}`.
- 7 MoE tensors: `mlp.{gate.weight, experts.gate_up_proj, experts.down_proj, shared_expert.gate_proj.weight, shared_expert.up_proj.weight, shared_expert.down_proj.weight, shared_expert_gate.weight}`.

A full-attention layer's 15 tensors:

- 1 `input_layernorm.weight`, 1 `post_attention_layernorm.weight`.
- 6 attention tensors: `self_attn.{q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight, q_norm.weight, k_norm.weight}`.
- 7 MoE tensors (same set as the linear-attention layer).

The MTP module's 19 tensors:

- 4 fusion / norm tensors at `mtp.{fc.weight, pre_fc_norm_embedding.weight, pre_fc_norm_hidden.weight, norm.weight}`.
- 15 layer tensors at `mtp.layers.0.*` matching the full-attention + MoE layer layout described above.

The vision encoder's 333 tensors:

- 3 top-level: `model.visual.patch_embed.proj.{weight, bias}`, `model.visual.pos_embed.weight`.
- 6 merger tensors: `model.visual.merger.{norm.{weight, bias}, linear_fc1.{weight, bias}, linear_fc2.{weight, bias}}`.
- 27 blocks × 12 tensors each = 324: `norm1.{weight, bias}`, `norm2.{weight, bias}`, `attn.qkv.{weight, bias}`, `attn.proj.{weight, bias}`, `mlp.linear_fc1.{weight, bias}`, `mlp.linear_fc2.{weight, bias}`.

`lm_head.weight`: 1 tensor.

Verification: `30*18 + 10*15 + 2 + 19 + 333 + 1 = 540 + 150 + 2 + 19 + 333 + 1 = 1045`. ✓

The "35B total / 3B activated" claim in the README maps to: 1045 named tensors at bf16 covering ~35.95B parameters total; per-token activation includes the embedding lookup, all attention or linear-attention weights for one layer, exactly 8 of the 256 routed experts in each layer plus the always-on shared expert, the gate weights, and the LM head.

## 13. Tokenizer, Special Tokens, and Prompt Formatting

Qwen3.6-35B-A3B ships a GPT-2-style BPE tokenizer:

- `tokenizer.json` (single Tokenizers-backend blob).
- `tokenizer_config.json`.
- `vocab.json`, `merges.txt` (legacy GPT-2 representation, also shipped).
- `chat_template.jinja`.

### Tokenizer Metadata

- `tokenizer_class`: `Qwen2Tokenizer` (the same BPE class that Qwen2/Qwen3 has used since the 2024 line; the rust `tokenizers` backend is preferred).
- `model_max_length`: `262_144` (matches `text_config.max_position_embeddings`).
- `pad_token`: `<|endoftext|>` (id 248044).
- `eos_token`: `<|im_end|>` (id 248046).
- `bos_token`: `null` (no canonical BOS string at the tokenizer level; `add_bos_token = false`).
- `padding_side`: not pinned in this file; framework default applies.
- `pretokenize_regex`: the standard Qwen pretokenizer regex (`(?i:'s|'t|...)|...|\\s+`).

### Special Tokens

The added-tokens table covers:

- Control tokens: `<|endoftext|>` (248044), `<|im_start|>` (248045), `<|im_end|>` (248046), `<tool_call>`, `</tool_call>`.
- Object reference / bounding-box / quad markers: `<|object_ref_{start, end}|>`, `<|box_{start, end}|>`, `<|quad_{start, end}|>`.
- Vision: `<|vision_{start, end, pad}|>` (248053–248055), `<|image_pad|>` (248056), `<|video_pad|>` (248057).
- Audio: `<|audio_{start, end, pad}|>` (audio tokens reserved through 248076 in the added-tokens table).

The model itself is image-text-to-text (`pipeline_tag: image-text-to-text` in the README header). Audio markers exist in the vocabulary but are not used by this checkpoint's vision-only multimodal head.

### Generation Metadata

The shipped `generation_config.json` contains:

- `do_sample = true`
- `temperature = 1.0`
- `top_k = 20`
- `top_p = 0.95`
- `bos_token_id = 248044`
- `eos_token_id = [248046, 248044]`
- `pad_token_id = 248044`

### Prompt Template Behavior

The shipped `chat_template.jinja` uses `<|im_start|>` / `<|im_end|>` role markers, supports tool-calling with a custom `<tool_call><function=name><parameter=...>` syntax (instead of JSON), preserves a `<think>` / `</think>` reasoning block on the assistant turn, and provides a `preserve_thinking` flag plus `enable_thinking` flag for retaining or eliding the reasoning trace across turns. Vision messages emit `<|vision_start|><|image_pad|><|vision_end|>` for images and `<|vision_start|><|video_pad|><|vision_end|>` for videos; the renderer counts images / videos and optionally prepends `Picture N: ` / `Video N: ` when `add_vision_id` is true.

The chat template raises if a system message contains images or videos, if there is no user query, or if a system message appears anywhere other than the first position.

### Important Metadata Mismatches

The release artifacts are largely self-consistent on prompt bootstrap, with two footnotes:

1. The `generation_config.json` declares `bos_token_id = 248044`, but the tokenizer's `bos_token` is null and `add_bos_token = false`. The chat template never inserts `<|endoftext|>` at the start, so the BOS field is effectively cosmetic for the chat-format path.
2. `eos_token_id` is two ids in `generation_config.json` (`[248046, 248044]`) but a single id in `text_config` (`248044`). The chat template ends turns with `<|im_end|>` (248046). Multi-id EOS is honored by Transformers' `GenerationMixin` and by vLLM / SGLang generation loops, so both ids correctly stop generation; the inconsistency is informational.

### Multimodal Preprocessor

- `preprocessor_config.json`: `Qwen2VLImageProcessorFast` with `image_mean = image_std = [0.5, 0.5, 0.5]`, patch size 16, temporal patch size 2, merge size 2, `longest_edge = 16_777_216`, `shortest_edge = 65_536` (longest_edge / shortest_edge are token-budget caps over the H*W flat patch grid, not pixel sizes).
- `video_preprocessor_config.json`: `Qwen3VLVideoProcessor` with the same per-pixel statistics, `longest_edge = 25_165_824`, `shortest_edge = 4_096`.

`processor_class = "Qwen3VLProcessor"` in both files; the multimodal pipeline matches the Qwen3-VL family.

## 14. Implementation Surfaces

The four reference points consulted for this document express Qwen3.6-35B-A3B with different levels of dedication:

| Reference | Path | Level of dedication |
| --- | --- | --- |
| Hugging Face repo | `Qwen/Qwen3.6-35B-A3B` | First-party config (nested), tokenizer (vocab + merges + tokenizer.json), chat template, image / video preprocessor configs, safetensors. No `*.py` modeling code; the repo relies on the upstream Transformers `qwen3_5_moe` integration. |
| Transformers | `transformers/models/qwen3_5_moe/` (`configuration_qwen3_5_moe.py`, `modular_qwen3_5_moe.py`, `modeling_qwen3_5_moe.py`) | First-class architecture. `Qwen3_5MoeConfig` extends `Qwen3VLConfig`; `Qwen3_5MoeTextConfig` extends `Qwen3NextConfig`. `Qwen3_5MoeDecoderLayer` reuses `Qwen3NextDecoderLayer`, `Qwen3_5MoeAttention` reuses `Qwen3NextAttention` (gated GQA), `Qwen3_5MoeGatedDeltaNet` reuses `Qwen3_5GatedDeltaNet` (split input projections), `Qwen3_5MoeTopKRouter` reuses `Qwen3VLMoeTextTopKRouter` (softmax + always-renormalize). `Qwen3_5MoeForCausalLM` ignores `mtp.*` and `model.visual.*` weights via `_keys_to_ignore_on_load_unexpected`. The full multimodal entry point is `Qwen3_5MoeForConditionalGeneration` (extending `Qwen3VLMoeForConditionalGeneration`). |
| vLLM | `vllm/model_executor/models/qwen3_5.py` plus `qwen3_5_mtp.py` | First-class. `Qwen3_5MoeForConditionalGeneration` extends `Qwen3VLForConditionalGeneration`; `Qwen3_5DecoderLayer` extends `Qwen3NextDecoderLayer` and uses `GemmaRMSNorm` aliased as `Qwen3_5RMSNorm`. The full-attention path comes from `Qwen3NextAttention` with `attn_output_gate=True`. The Gated DeltaNet path comes from `vllm.model_executor.layers.mamba.gdn_linear_attn.GatedDeltaNetAttention`. The MoE path uses vLLM's `FusedMoE` and the `Qwen3NextSparseMoeBlock`. The MTP wrapper is a separate registered model `Qwen3_5MultiTokenPredictor` that constructs one `Qwen3_5DecoderLayer` with `layer_type="full_attention"`. |
| SGLang | `python/sglang/srt/models/qwen3_5.py` plus `qwen3_5_mtp.py` | First-class. `EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]`. The decoder layer is split into `Qwen3_5LinearDecoderLayer` and `Qwen3_5AttentionDecoderLayer` (selected by `layer_type`). The attention layer reads `attn_output_gate` from config (default true) and widens `qkv_proj` accordingly. The MoE path uses SGLang's `FusedMoE`. The MTP wrapper sets `config.full_attention_interval = 1` and `config.num_hidden_layers = 1` before constructing `Qwen3_5ForCausalLM(is_nextn=True)` so the inner module instantiates a single full-attention MoE layer. SGLang also handles the `model.language_model.` prefix stripping at load time. |

## 15. Metadata Present in the Release but Not Used as Hard Switches by the Checkpoint-Local Transformers Model

| Field | Stored Value | Transformers Behavior |
| --- | --- | --- |
| `attn_output_gate` | `true` | Honored. Q projection is allocated at `2 * num_heads * head_dim`; the second half is sigmoided and multiplied into the attention output. vLLM and SGLang also honor this and adjust `qkv_proj` width. |
| `mrope_interleaved` | `true` | Honored by the Transformers `Qwen3VLTextRotaryEmbedding` superclass: rotary frequencies for the temporal/height/width axes are interleaved across the rotary dims. vLLM / SGLang consult the same flag in their kernels. |
| `mrope_section` | `[11, 11, 10]` | Honored. Partitions the 32 rotary half-frequencies (= 64/2) across the three multimodal axes. |
| `partial_rotary_factor` | `0.25` | Honored end-to-end. Note the field appears both at `text_config.partial_rotary_factor` and at `text_config.rope_parameters.partial_rotary_factor`; both carry the same value. |
| `mamba_ssm_dtype` | `"float32"` | Honored: the FLA kernel for chunked / recurrent gated delta-rule keeps the SSM internal state in float32. |
| `output_router_logits` | `false` | Honored as a wrapper-level switch (the auxiliary loss is only assembled when true). |
| `router_aux_loss_coef` | `0.001` | Used only by the training auxiliary loss path; structurally inert at inference. |
| `mtp_num_hidden_layers` | `1` | Honored by vLLM and SGLang for spec decoding. The Transformers `Qwen3_5MoeForCausalLM` ignores `mtp.*` tensors via `_keys_to_ignore_on_load_unexpected`. The `Qwen3_5MoeForConditionalGeneration` wrapper also has no MTP path. |
| `mtp_use_dedicated_embeddings` | `false` | Documented expectation: the MTP shares `embed_tokens` with the main model. There is no `mtp.embed_tokens.weight` on disk; vLLM's MTP wrapper allocates its own `embed_tokens` parameter that gets populated from the shared embedding at load time (or stays randomly initialized if loading fails — the absence of dedicated MTP embeddings on disk means the weight loader has to wire this up). |
| `full_attention_interval` | `4` | Used only as a synthesizer for `layer_types` when the latter is absent. The shipped config pins `layer_types` explicitly, so this field is informational. |
| `attn_implementation` family | flash / sdpa | The Qwen3.5 and Qwen3-Next reference declares `_supports_flash_attn = True`, `_supports_sdpa = True`. Linear-attention layers have their own preferred kernel path (`causal_conv1d_fn` + `chunk_gated_delta_rule` from FLA); falling back to torch implementations is functionally correct but slow. |

## 16. Implementation Notes for Engineers

A checkpoint-matching implementation reproduces these facts:

- 40 base text-decoder layers in pattern `[L, L, L, F] * 10` (L = linear attention, F = full attention), plus 1 MTP layer at on-disk prefix `mtp.*` whose embedded decoder is full-attention.
- Hidden size 2048; vocabulary size 248320; `tie_word_embeddings = false`; lm_head is its own bf16 tensor of shape `(248320, 2048)`.
- Full-attention layers: 16 query heads, 2 KV heads, head_dim 256, `attn_output_gate = true`. The Q projection emits `2 * num_heads * head_dim = 8192` and is split into query (4096) and gate (4096); the gate is sigmoided and multiplied into the post-attention output before `o_proj`. Per-head Q/K RMSNorm over the 256-dim head; partial rotary on the leading 64 dims.
- M-RoPE with three multimodal axes plus a text axis, `mrope_section = [11, 11, 10]`, `mrope_interleaved = true`, `rope_theta = 10_000_000`.
- Linear-attention layers: Gated DeltaNet with 32 V heads, 16 K/Q heads (Q,K repeated 2x to align), head dim 128. Depthwise causal Conv1d over an 8192-wide channel axis with kernel 4 and SiLU; fused QKV input projection (`in_proj_qkv`) plus separate `in_proj_z`, `in_proj_b`, `in_proj_a`. Per-head `A_log` and `dt_bias` discretize the SSM step. Output through `RMSNormGated(out, z)` then `out_proj`.
- Sparse MoE on every layer: 256 routed experts plus 1 always-on shared expert, top-8 routing, softmax + always-renormalize routing weights (no bias term, no auxiliary correction). Routed-expert weights stored as 3D fused tensors `gate_up_proj (256, 1024, 2048)` and `down_proj (256, 2048, 512)`. Shared expert is a (512, 2048) SwiGLU MLP with a sigmoid scalar gate (`shared_expert_gate.weight: (1, 2048)`).
- bf16 weights throughout; float32 for `A_log`, `dt_bias`, the SSM internal state (per `mamba_ssm_dtype`), the router scoring, and RMSNorm internal computation.
- RMSNorm uses the `(1 + weight)` zero-centered convention (Gemma / Qwen3-style), with weight initialized to zero so the layer starts as an identity. `rms_norm_eps = 1e-6`.
- One MTP module's worth of weights at `mtp.*` (extra `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm`, plus a single full-attention + MoE layer at `mtp.layers.0.*`). The MTP shares the global `embed_tokens` and `lm_head`. Whether to load and run it is a runtime choice.
- 27-block ViT vision encoder (`model.visual.*`) with `hidden_size = 1152`, `intermediate_size = 4304`, 16 heads, GELU-pytorch-tanh activation, `patch_size = 16`, `temporal_patch_size = 2`, `spatial_merge_size = 2`, `out_hidden_size = 2048`. LayerNorm with bias on every block; learned `pos_embed (2304, 1152)` interpolated against the actual `(t, h, w)` grid; `merger` produces 2048-wide tokens that scatter directly into the text residual at `image_token_id` / `video_token_id` positions. No deepstack feature merging in this checkpoint.
- Native context 262144; the README quotes runtime extension up to ~1,010,000 tokens (no scaling tensors shipped; this requires a runtime-side YaRN/NTK-style RoPE rescale that is not part of the config).

Exact packaging parity also requires honoring:

- The `model.language_model.` prefix on text-decoder tensors (vLLM and SGLang both strip it; Transformers consumes the prefix natively because the conditional-generation wrapper has a `language_model` submodule).
- The `mtp.*` prefix sits at the safetensors root, not inside `model.*`. Implementations that load the MTP need to look for it there.
- The 3D fused expert layout. vLLM's and SGLang's `FusedMoE` weight loaders split the `(num_experts, 2 * intermediate, hidden)` `gate_up_proj` into separate gate / up halves at load time; the `(num_experts, hidden, intermediate)` `down_proj` is loaded as-is.
- The split GDN input projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`). SGLang remaps these into Qwen3-Next's fused `in_proj_qkvz` / `in_proj_ba` parameters at load time via its `stacked_params_mapping`; vLLM keeps them split internally.

## 17. Bibliography

- Hugging Face repo: `Qwen/Qwen3.6-35B-A3B` (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `preprocessor_config.json`, `video_preprocessor_config.json`, `model.safetensors.index.json`, 26 safetensors shards, `README.md`, `LICENSE`).
- Qwen blog post: `Qwen3.6-35B-A3B` (linked from the model card at `https://qwen.ai/blog?id=qwen3.6-35b-a3b`).
- Transformers: `src/transformers/models/qwen3_5_moe/{configuration,modular,modeling}_qwen3_5_moe.py`; the parent classes in `qwen3_5/`, `qwen3_next/`, and `qwen3_vl_moe/`; the vision encoder in `qwen3_vl/`.
- vLLM: `vllm/model_executor/models/qwen3_5.py` (`Qwen3_5DecoderLayer`, `Qwen3_5Model`, `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`), `qwen3_5_mtp.py` (`Qwen3_5MultiTokenPredictor`), `qwen3_next.py` (`Qwen3NextAttention` for the gated GQA path, `Qwen3NextSparseMoeBlock`), and `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (`GatedDeltaNetAttention` for the linear-attention path).
- SGLang: `python/sglang/srt/models/qwen3_5.py` (`Qwen3_5GatedDeltaNet`, `Qwen3_5LinearDecoderLayer`, `Qwen3_5AttentionDecoderLayer`, `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`), `qwen3_5_mtp.py` (`Qwen3_5ForCausalLMMTP`).
