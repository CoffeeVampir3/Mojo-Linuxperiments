# GLM-5.1 Architecture Reference

This document describes the shipped GLM-5.1 release artifacts as they are packaged: the serialized model config, the tokenizer, the generation config, the safetensors weight index, and the converging behavior across the four reference implementations (the HuggingFace repo, the first-party Transformers `glm_moe_dsa` module, vLLM's `deepseek_v2` integration, and SGLang's `deepseek_v2` integration). It separates exact observed behavior from metadata that is present in the release but not enforced by every runtime.

## 1. Model Summary

GLM-5.1 is a decoder-only causal language model with Multi-head Latent Attention (MLA), DeepSeek Sparse Attention (DSA) on every layer, sparse MoE feed-forward blocks, and a single Multi-Token Prediction (MTP) head appended to the end of the layer stack.

- 78 base decoder layers plus 1 MTP layer (79 layer slots, indices 0 through 78).
- Layers 0, 1, 2 are dense MLPs. Layers 3 through 77 are MoE. Layer 78 is the MTP module.
- Residual width 6144.
- 64 query heads and 64 key/value heads (no grouped-query reduction in the head count).
- MLA latent compression: `q_lora_rank=2048`, `kv_lora_rank=512`.
- Per-head dimensions: `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`. Concatenated `qk_head_dim` is 256.
- Partial RoPE on the trailing 64 dimensions of each Q and K head; the leading 192 (Q/K-nope) and 256 (V) dimensions are unrotated.
- 256 routed experts per MoE layer plus 1 shared expert. Top-8 routing with sigmoid scoring and a learned selection bias (`noaux_tc`).
- SwiGLU experts with intermediate size 2048. SwiGLU dense MLPs with intermediate size 12288.
- DSA indexer present on every base layer: 32 indexer heads, head dimension 128, top-2048 token selection.
- Untied token embedding and LM head.
- Released weights are bfloat16 (with FP32 router weights and bias, and FP32 indexer weights). A separate `zai-org/GLM-5.1-FP8` repository ships an FP8-block-quantized variant.

## 2. Serialized `config.json`

These values are explicitly stored in the shipped `config.json`.

| Field | Value |
| --- | --- |
| `architectures` | `["GlmMoeDsaForCausalLM"]` |
| `model_type` | `"glm_moe_dsa"` |
| `dtype` | `"bfloat16"` |
| `vocab_size` | `154880` |
| `hidden_size` | `6144` |
| `intermediate_size` | `12288` |
| `moe_intermediate_size` | `2048` |
| `num_hidden_layers` | `78` |
| `num_attention_heads` | `64` |
| `num_key_value_heads` | `64` |
| `head_dim` | `64` |
| `qk_head_dim` | `256` |
| `qk_nope_head_dim` | `192` |
| `qk_rope_head_dim` | `64` |
| `v_head_dim` | `256` |
| `q_lora_rank` | `2048` |
| `kv_lora_rank` | `512` |
| `hidden_act` | `"silu"` |
| `max_position_embeddings` | `202752` |
| `rope_parameters` | `{"rope_theta": 1000000, "rope_type": "default"}` |
| `rope_interleave` | `true` |
| `rms_norm_eps` | `1e-5` |
| `tie_word_embeddings` | `false` |
| `n_routed_experts` | `256` |
| `num_experts_per_tok` | `8` |
| `n_shared_experts` | `1` |
| `routed_scaling_factor` | `2.5` |
| `scoring_func` | `"sigmoid"` |
| `topk_method` | `"noaux_tc"` |
| `n_group` | `1` |
| `topk_group` | `1` |
| `norm_topk_prob` | `true` |
| `first_k_dense_replace` | `3` |
| `moe_layer_freq` | `1` |
| `index_topk` | `2048` |
| `index_head_dim` | `128` |
| `index_n_heads` | `32` |
| `indexer_rope_interleave` | `true` |
| `num_nextn_predict_layers` | `1` |
| `attention_bias` | `false` |
| `attention_dropout` | `0.0` |
| `initializer_range` | `0.02` |
| `pretraining_tp` | `1` |
| `ep_size` | `1` |
| `pad_token_id` | `154820` |
| `eos_token_id` | `[154820, 154827, 154829]` |
| `transformers_version` | `"5.4.0"` |
| `use_cache` | `true` |

Derived shape facts from the serialized config:

- Q-LoRA path: `q_a_proj` projects `6144 -> 2048`; `q_b_proj` projects `2048 -> 64 * 256 = 16384`.
- KV-LoRA path: `kv_a_proj_with_mqa` projects `6144 -> kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`; `kv_b_proj` projects `512 -> 64 * (qk_nope_head_dim + v_head_dim) = 64 * 448 = 28672`.
- Output projection: `o_proj` projects `64 * 256 = 16384 -> 6144`.
- Indexer projections: `wq_b` projects `2048 -> 32 * 128 = 4096`; `wk` projects `6144 -> 128`; `weights_proj` projects `6144 -> 32`; `k_norm` is a `LayerNorm(128, eps=1e-6)` with both weight and bias.
- Dense MLP (first three layers): `gate_proj` and `up_proj` are `6144 -> 12288`; `down_proj` is `12288 -> 6144`.
- Sparse MoE expert: `gate_proj` and `up_proj` are `6144 -> 2048`; `down_proj` is `2048 -> 6144`.
- Shared expert: same shapes as a single routed expert (intermediate width is `moe_intermediate_size * n_shared_experts = 2048 * 1 = 2048`).
- Router: `gate.weight` is `(256, 6144)` in float32 with no bias term in the linear projection; `gate.e_score_correction_bias` is `(256,)` in float32.

## 3. Top-Level Module Structure

The packaged causal LM entry point has this structure:

1. `embed_tokens`: embedding matrix of shape `(154880, 6144)`.
2. `layers`: 78 base decoder layers plus 1 MTP layer.
3. `norm`: final RMSNorm over the residual stream.
4. `lm_head`: output projection of shape `(154880, 6144)`.

`tie_word_embeddings` is false, so `lm_head.weight` is a separate tensor and is not tied to `embed_tokens.weight`. (The HF Transformers class does declare `lm_head.weight` as a tied-weights candidate, but the runtime tying is gated on the config flag and is therefore inactive for this checkpoint.)

Each base decoder layer has two pre-norm residual sub-blocks:

1. RMSNorm -> MLA self-attention with DSA indexer -> residual add.
2. RMSNorm -> MLP -> residual add. The MLP is a dense SwiGLU for `layer_idx < first_k_dense_replace = 3`, otherwise a sparse MoE block with one shared expert.

The trailing layer at index 78 is a MTP module, described in section 7.

## 4. Attention Block (MLA)

Each base layer uses Multi-head Latent Attention with a low-rank query path, a low-rank key/value path, and a single shared rope-only key stream (MQA-style for the rope component).

### Projection Shapes

- `q_a_proj`: `(6144 -> 2048)`, biasless (`attention_bias=false`).
- `q_a_layernorm`: RMSNorm over the 2048-dimensional Q residual.
- `q_b_proj`: `(2048 -> 16384)`, biasless. Reshapes to `(64 heads, 256 head_dim)`.
- `kv_a_proj_with_mqa`: `(6144 -> 576)`, biasless. Splits to `(512 latent kv, 64 rope k_pe)`.
- `kv_a_layernorm`: RMSNorm over the 512-dimensional KV latent.
- `kv_b_proj`: `(512 -> 28672)`, biasless. Reshapes to `(64 heads, 448 = 192 + 256)` and splits to `(k_nope, v)`.
- `o_proj`: `(16384 -> 6144)`, biasless.

The MLA construction follows DeepSeek-V3 / V3.2 conventions. There is no separate `q_proj` because `q_lora_rank` is set; the LoRA path is always used.

### Rotary Position Embeddings

- RoPE base is `1_000_000`.
- Maximum configured position span is `202_752`.
- Only the trailing `qk_rope_head_dim = 64` dimensions of each Q head and the dedicated 64-dimensional `k_pe` stream are rotated. The leading `qk_nope_head_dim = 192` of Q and K and the entire `v_head_dim = 256` value vector pass through unchanged.
- The K-rope stream is single-headed (one shared 64-dimensional vector per token), then expanded to 64 heads after rotation by repeating along the head dimension.
- RoPE inverse frequencies are computed once in float32 and the cosines/sines are cast back to the hidden-state dtype before application.
- `rope_interleave=true` and `indexer_rope_interleave=true` are present in the serialized config. The vLLM and SGLang DeepSeek-V2 paths consult these fields to switch between interleaved and split-half (NeoX/Llama) RoPE for the main attention and the indexer respectively. The first-party Transformers `modular_glm_moe_dsa.py` always uses split-half (NeoX/Llama) `rotate_half` and disregards these switches; the docstring notes the divergence explicitly.

### Attention Math

For each layer and token block:

1. Compute the Q residual `q_resid = q_a_layernorm(q_a_proj(x))` of shape `(B, S, 2048)`. Project to `q = q_b_proj(q_resid)` of shape `(B, S, 64, 256)`.
2. Project the KV latent `compressed_kv = kv_a_proj_with_mqa(x)` of shape `(B, S, 576)`, split into `(k_compressed, k_pe)` of widths 512 and 64, and apply `kv_a_layernorm` to `k_compressed`.
3. Expand `kv = kv_b_proj(k_compressed)` to `(B, S, 64, 448)` and split into `k_nope` `(B, S, 64, 192)` and `v` `(B, S, 64, 256)`.
4. Split Q along the head dim into `q_nope` `(..., 192)` and `q_pe` `(..., 64)`. Apply RoPE to `q_pe`.
5. Apply RoPE to the single-headed `k_pe` `(B, S, 1, 64)`, then broadcast to all 64 heads.
6. Concatenate to assemble `Q = [q_nope; q_pe]` and `K = [k_nope; k_pe]` of shape `(B, 64, S, 256)`. The value tensor stays at `(B, 64, S, 256)`.
7. Update the KV cache. Both Transformers and vLLM expand the full `(K, V)` into the cache for compatibility with standard attention backends. vLLM additionally exposes a compressed-cache MLA path (`DeepseekV2MLAAttention`) that avoids the `kv_b_proj` expansion at decode time.
8. Run the DSA indexer (section 5) to obtain `topk_indices` of shape `(B, S, 2048)`.
9. Build a combined sparse-plus-causal mask: `-inf` everywhere except at the top-2048 selected token positions, then add the causal mask on top.
10. Compute scaled-dot-product attention with scaling factor `1 / sqrt(qk_head_dim) = 1 / sqrt(256)`. When `v_head_dim != qk_head_dim`, FlashAttention paths zero-pad V to `qk_head_dim`, run attention, and slice back to `v_head_dim`.
11. Reshape to `(B, S, 16384)` and project through `o_proj` back to width 6144.
12. Add the attention output to the residual stream.

The shipped configuration does not enable sliding-window attention.

## 5. DeepSeek Sparse Attention Indexer

Every base decoder layer has its own `Indexer` module. The indexer is structurally separate from MLA: it has its own projections (`wq_b`, `wk`, `k_norm`, `weights_proj`), its own RoPE stream, and its own key cache.

### Projections

- `indexer.wq_b`: `(q_lora_rank=2048 -> n_heads * head_dim = 32 * 128 = 4096)`, biasless. Consumes the Q residual produced by `q_a_layernorm(q_a_proj(x))`.
- `indexer.wk`: `(hidden_size=6144 -> head_dim=128)`, biasless. Single shared key vector per token, computed directly from the layer input. (vLLM fuses `wk` and `weights_proj` into a single `MergedColumnParallelLinear` named `wk_weights_proj`, but the on-disk tensors and the Transformers reference are split.)
- `indexer.k_norm`: `LayerNorm(128, eps=1e-6)` with both `weight` and `bias` parameters. This is a real LayerNorm rather than RMSNorm, which is why the safetensors index contains both `indexer.k_norm.weight` and `indexer.k_norm.bias`.
- `indexer.weights_proj`: `(hidden_size=6144 -> n_heads=32)`, biasless. Produces a per-head scalar weight for each token.

The reference implementation keeps `indexer.weights_proj` in float32 even when the surrounding linear weights are FP8-quantized. The Transformers `_keep_in_fp32_modules = ["indexer.weights_proj"]` directive prevents FP8 conversion for this module.

### Indexer Forward Pass

For each token block:

1. Compute queries `q = wq_b(q_resid)` of shape `(B, S, 32, 128)` and split into rotary half `q_pe` `(..., 64)` and non-rotary half `q_nope` `(..., 64)`. Apply RoPE to `q_pe` and concatenate.
2. Compute keys `k = k_norm(wk(x))` of shape `(B, S, 128)` and split into `k_pe` `(..., 64)` and `k_nope` `(..., 64)`. Apply RoPE to `k_pe` and concatenate.
3. Append `k` to the indexer's own key cache (separate from the MLA `DynamicCache`, since `DynamicCache` is sized to exactly `num_hidden_layers` attention layers).
4. Compute per-token, per-head weights `w = weights_proj(x) * n_heads^{-1/2}`.
5. Compute scores `s[b, s, h, t] = (q[b, s, h, :] . k[b, t, :]) * (1 / sqrt(head_dim))`, apply ReLU, then aggregate across heads `score[b, s, t] = sum_h s[b, s, h, t] * w[b, s, h]`. Add the causal mask.
6. Take `topk(score, k=index_topk=2048)` to obtain the indices used by the MLA attention as a sparse mask.

The vLLM implementation runs the scoring kernel in FP8 (`SparseAttnIndexer`, `DeepseekV32IndexerCache`) with per-token-group quantization (`quant_block_size=128`, `scale_fmt="ue8m0"`). The reference Transformers implementation runs the scoring in float32 / bfloat16 and remarks that the Hadamard rotation used in DeepSeek-V3.2's reference indexer is mathematically a no-op on dot products and is therefore omitted.

### Per-Layer Skip Pattern

The Transformers config exposes `indexer_types: list[str] | None`. By default, layer 0 receives `"full"`, every `index_topk_freq`-th layer receives `"full"`, and the remaining layers receive `"shared"`. A `"shared"` layer reuses the previous layer's `topk_indices` instead of recomputing them. The decoder layer threads `prev_topk_indices` through the layer stack so that consecutive layers can share an index selection.

The shipped `config.json` does not pin `indexer_types` or `index_topk_freq` explicitly, so the default behavior depends on the runtime: with `index_topk_freq=1`, every layer is `"full"` and computes a fresh index; with a larger `freq`, blocks of layers reuse the same index. The shipped indexer weights exist on every base layer regardless, because the parameter set is decided at construction time.

## 6. Sparse MoE Block

For `layer_idx >= first_k_dense_replace = 3`, the feed-forward block is a sparse MoE.

### Router

- Router input width: 6144.
- Router output width: 256 experts.
- `gate.weight` shape: `(256, 6144)`, dtype float32.
- No router bias term in the linear projection.
- `gate.e_score_correction_bias`: shape `(256,)`, dtype float32.

The routing logic is the DeepSeek-V3 `noaux_tc` recipe:

1. Flatten hidden states to `(B*S, 6144)`.
2. Compute router logits in float32 via `F.linear(x.float(), gate.weight.float())`.
3. Apply elementwise sigmoid to obtain `scores`.
4. Form selection scores `scores_sel = scores + e_score_correction_bias`.
5. Group routing: reshape to `(B*S, n_group, n_routed_experts // n_group) = (B*S, 1, 256)` (with `n_group=1`, this is a no-op pass-through). Compute the sum of the top-2 expert scores within each group, take `topk_group=1` groups, mask out unselected groups, then take the top-`num_experts_per_tok=8` experts within the selected groups.
6. Gather the original sigmoid scores (without the bias) for the selected 8 experts.
7. If `norm_topk_prob`, divide each token's 8 weights by their sum (with a small epsilon).
8. Multiply the 8 weights by `routed_scaling_factor=2.5` and use them to combine the expert outputs.

`e_score_correction_bias` only influences expert selection; it does not contribute to the mixture weights themselves.

### Experts

Each routed expert is a SwiGLU MLP:

- `gate_proj`: `(2048, 6144)`
- `up_proj`: `(2048, 6144)`
- `down_proj`: `(6144, 2048)`

Per-token expert computation is `down_proj(silu(gate_proj(x)) * up_proj(x))`.

The reference Transformers implementation enumerates 256 separate experts per layer and dispatches only to the experts hit by at least one token in the current batch. vLLM uses its `FusedMoE` kernel; SGLang uses its analogous fused dispatcher.

### Shared Expert

Every MoE layer also contains a single shared SwiGLU MLP at `mlp.shared_experts`:

- `gate_proj`: `(moe_intermediate_size * n_shared_experts, hidden_size) = (2048, 6144)`
- `up_proj`: `(2048, 6144)`
- `down_proj`: `(6144, 2048)`

The shared expert receives the un-routed input and its output is added to the routed-expert sum to form the MoE block output:

```
y = sum_i topk_weight[i] * expert[topk_index[i]](x) + shared_experts(x)
```

## 7. Multi-Token Prediction Module

Layer index 78 is a single MTP module (`num_nextn_predict_layers = 1`). The released weight map contains a complete tensor set for it, distinct from the 78 base layers:

- `model.layers.78.input_layernorm.weight`, `model.layers.78.post_attention_layernorm.weight`: standard pre-norms used by the embedded decoder block.
- `model.layers.78.self_attn.*`: a complete MLA + DSA attention block with the same shapes as base layers.
- `model.layers.78.mlp.*`: a complete MoE block (router with `e_score_correction_bias`, 256 routed experts, 1 shared expert) with the same shapes as base MoE layers.
- `model.layers.78.enorm.weight`: RMSNorm applied to the next-token-position embedding stream.
- `model.layers.78.hnorm.weight`: RMSNorm applied to the previous-token hidden-state stream.
- `model.layers.78.eh_proj.weight`: linear projection of shape `(hidden_size, 2 * hidden_size) = (6144, 12288)`. Concatenates the normalized embedding and hidden streams and projects them back to `hidden_size`.
- `model.layers.78.shared_head.norm.weight`: final RMSNorm applied before the shared LM head produces the speculative logits.

Loading behavior diverges across runtimes:

- The first-party Transformers `GlmMoeDsaPreTrainedModel` declares `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]`. The MTP weights are present on disk but are silently dropped when loading a `GlmMoeDsaForCausalLM`; the loaded model has 78 layers and no speculative decoding capability.
- vLLM and SGLang detect `num_nextn_predict_layers > 0` and route the MTP weights into a separate speculative-decoding wrapper. SGLang's `model_config.py` contains an explicit rewrite from `GlmMoeDsaForCausalLM` to `DeepseekV3ForCausalLMNextN` for the draft-model path. vLLM's `get_spec_layer_idx_from_weight_name` helper detects MTP-layer tensors by checking whether `weight_name` starts with `model.layers.{num_hidden_layers + i}.` for `i in range(num_nextn_predict_layers)`.

The MTP tensors are real and consume real storage; whether they are exercised at inference is a runtime decision.

## 8. Forward Pass Summary

For input shape `(B, S)`:

1. Token lookup produces hidden states of shape `(B, S, 6144)`.
2. The shared `GlmMoeDsaRotaryEmbedding` produces `(cos, sin)` tensors covering all positions in float32, then casts them to the hidden-state dtype.
3. The hidden states pass through 78 base decoder layers, with `prev_topk_indices` threaded between layers so that `"shared"` indexer layers reuse the previous full layer's selection.
4. Each base decoder layer performs:
   - input RMSNorm -> MLA + DSA self-attention -> residual add
   - post-attention RMSNorm -> dense SwiGLU (layers 0-2) or sparse MoE with shared expert (layers 3-77) -> residual add
5. Final RMSNorm produces the last hidden state.
6. `lm_head` projects to logits of shape `(B, S, 154880)`.
7. The MTP module at layer 78 is invoked separately by speculative-decoding wrappers in vLLM and SGLang. It is skipped entirely by the first-party Transformers `GlmMoeDsaForCausalLM`.

## 9. Quantization Story

### Default release (`zai-org/GLM-5.1`)

The shipped `dtype` is `bfloat16`. Most parameters are stored bf16 with the following exceptions enforced by the Transformers implementation:

- `gate.weight`: float32.
- `gate.e_score_correction_bias`: float32 (kept via `_keep_in_fp32_modules_strict`).
- `indexer.weights_proj.weight`: stored as bf16 in this checkpoint, but `_keep_in_fp32_modules` keeps the surrounding `nn.Linear` from being FP8-converted by quantization wrappers.

### FP8 release (`zai-org/GLM-5.1-FP8`)

A separate FP8 variant of the same architecture exists. Its quantization conventions are not part of `zai-org/GLM-5.1` and are not described by the `config.json` reproduced in section 2. The DSA indexer's K cache in vLLM is FP8 (`scale_fmt="ue8m0"`, `quant_block_size=128`) regardless of the main weight dtype, because it is a runtime cache rather than a stored parameter.

### MXFP4 release (`amd/GLM-5.1-MXFP4`)

A community MXFP4 variant exists for AMD platforms. It is referenced from external recipes; it is not part of the upstream release.

## 10. Weight Inventory

The safetensors weight index reports a total of `59870` named tensors and `metadata.total_size` of `1_507_728_316_928` bytes (~1.51 TB), distributed across 282 safetensors shards.

Per-layer tensor counts measured directly from `model.safetensors.index.json`:

| Layer family | Layers | Tensors per layer | Subtotal |
| --- | --- | --- | --- |
| Dense MLP (0, 1, 2) | 3 | 17 | 51 |
| MoE (3 through 77) | 75 | 787 | 59,025 |
| MTP (78) | 1 | 791 | 791 |
| Top-level (`embed_tokens`, `norm`, `lm_head`) | - | - | 3 |
| **Total** | **79** | - | **59,870** |

A dense layer's 17 tensors:

- 1 `input_layernorm.weight`, 1 `post_attention_layernorm.weight`.
- 12 self-attention tensors: 5 indexer (`wq_b.weight`, `wk.weight`, `weights_proj.weight`, `k_norm.weight`, `k_norm.bias`), 7 MLA (`q_a_proj.weight`, `q_a_layernorm.weight`, `q_b_proj.weight`, `kv_a_proj_with_mqa.weight`, `kv_a_layernorm.weight`, `kv_b_proj.weight`, `o_proj.weight`).
- 3 dense MLP tensors: `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`.

A MoE layer's 787 tensors:

- 14 attention + layernorm tensors (same as dense).
- 2 router tensors: `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`.
- 3 shared-expert tensors: `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`.
- 768 routed-expert tensors: `mlp.experts.{0..255}.{gate_proj,up_proj,down_proj}.weight`.

The MTP layer's 791 tensors:

- All 787 tensors of a regular MoE layer.
- Plus 4 MTP-specific tensors: `enorm.weight`, `hnorm.weight`, `eh_proj.weight`, `shared_head.norm.weight`.

Top-level tensors:

- `model.embed_tokens.weight`
- `model.norm.weight`
- `lm_head.weight`

Assuming the bf16 weight footprint dominates, `1_507_728_316_928 / 2 ≈ 753.9B` parameters. Marketing materials quote both `754B` (full checkpoint including MTP) and `744B` (active main-stack parameters, MTP excluded). Both numbers describe the same artifact.

## 11. Tokenizer, Special Tokens, and Prompt Formatting

GLM-5.1 ships with a Tokenizers-backed BPE tokenizer in the form of a single `tokenizer.json` blob (no separate `vocab.json` / `merges.txt` files). The tokenizer artifacts include:

- `tokenizer_config.json`
- `chat_template.jinja`

### Tokenizer Metadata

- `tokenizer_class`: `TokenizersBackend`.
- `model_max_length`: `202_752` (matches `max_position_embeddings`).
- `pad_token`: `<|endoftext|>`.
- `eos_token`: `<|endoftext|>`.
- `padding_side`: `left`.

The added special tokens cover both control roles and multimodal markers, including `<|endoftext|>`, `[MASK]`, `[gMASK]`, `[sMASK]`, `<sop>`, `<eop>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`, and beg/end markers for image, video, audio, and transcription. The model itself is text-only at the language-model level; the multimodal markers are reserved in the vocabulary for downstream multimodal variants.

### Generation Metadata

The shipped `generation_config.json` contains:

- `temperature=1.0`
- `top_p=0.95`
- `pad_token_id=154820`
- `eos_token_id=[154820, 154827, 154829]` (matches `config.json`)

### Prompt Template Behavior

The shipped chat template uses the `<|system|>`, `<|user|>`, `<|assistant|>`, and `<|observation|>` role markers, surrounds tool-call payloads with `<tool_call>` / `</tool_call>` blocks, and supports an optional thinking section bounded by `<think>` / `</think>`.

### Important Metadata Mismatches

The release artifacts are largely self-consistent on prompt bootstrap. Two minor footnotes:

1. The stored `config.json` does not declare a BOS token id. The Transformers `GlmMoeDsaConfig` defaults `bos_token_id=0` and `eos_token_id=1` as placeholders, but neither overrides the tokenizer's actual special tokens.
2. `eos_token_id` is a list of three ids in `config.json` and `generation_config.json`, but the tokenizer surfaces a single `<|endoftext|>` as the canonical EOS string. Multi-id EOS is honored by Transformers' `GenerationMixin` and by vLLM/SGLang generation loops.

## 12. Implementation Surfaces

The four reference points consulted for this document express GLM-5.1 with different levels of dedication:

| Reference | Path | Level of dedication |
| --- | --- | --- |
| Hugging Face repo | `zai-org/GLM-5.1` | First-party config, tokenizer, chat template, safetensors. No `*.py` modeling code; relies on first-party Transformers integration. |
| Transformers | `transformers/models/glm_moe_dsa/` (`configuration_glm_moe_dsa.py`, `modular_glm_moe_dsa.py`, `modeling_glm_moe_dsa.py`) | First-class architecture. `GlmMoeDsaConfig` extends `Glm4MoeLiteConfig`. The decoder layer, MoE block, and `GlmMoeDsaForCausalLM` reuse `Glm4Moe*` parents; the `Indexer` and `GlmMoeDsaAttention` are bespoke. Does not load MTP. |
| vLLM | `vllm/model_executor/models/deepseek_v2.py` | `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass` — the GLM-5.1 entry is a thin alias. All architecture lives in the shared `DeepseekV2*` classes (`DeepseekV2MLAAttention`, `Indexer`, `SparseAttnIndexer`, `DeepseekV32IndexerCache`, `DeepseekV2MoE`). MTP is handled via the `next-N` speculative path. The DeepseekV2 attention path consults `indexer_rope_interleave` to choose RoPE style for the indexer. |
| SGLang | `python/sglang/srt/configs/model_config.py` plus `python/sglang/srt/models/deepseek_v2.py` | `GlmMoeDsaForCausalLM` is treated as a DeepSeek-V3-compatible architecture by `model_config.py` for the purposes of NSA detection, MLA `head_dim=256` derivation, draft-model remap to `DeepseekV3ForCausalLMNextN`, and inclusion in `piecewise_cuda_graph_disabled_model_archs`. The actual modeling reuses `DeepseekV2*` classes. |

## 13. Metadata Present in the Release but Not Used as Hard Switches by the Checkpoint-Local Transformers Model

| Field | Stored Value | Transformers Behavior |
| --- | --- | --- |
| `scoring_func` | `"sigmoid"` | Matches the implementation, but routing is hardcoded to sigmoid in `GlmMoeDsaMoE.route_tokens_to_experts`. |
| `topk_method` | `"noaux_tc"` | Matches the implementation, but the routing path is hardcoded; the field is informational. |
| `n_group` / `topk_group` | `1` / `1` | Honored. With these values, the group-mask step degenerates into a pass-through over all 256 experts. |
| `rope_interleave` | `true` | Ignored by the Transformers implementation, which uses split-half (NeoX/Llama) `rotate_half` unconditionally. Honored by vLLM and SGLang. |
| `indexer_rope_interleave` | `true` | Same as above; honored by vLLM and SGLang, ignored by Transformers. |
| `num_nextn_predict_layers` | `1` | Honored by vLLM and SGLang for spec decoding. The Transformers `GlmMoeDsaForCausalLM` ignores all `model.layers.78.*` tensors via `_keys_to_ignore_on_load_unexpected`. |
| `attn_implementation` family | flash / sdpa / flex | The Transformers reference declares `_supports_flash_attn = False`, `_supports_sdpa = True`, `_supports_flex_attn = False`. A `flash-mla` kernel is the recommended community implementation but is not the default. |
| `pretraining_tp` / `ep_size` | `1` / `1` | Used only by tensor-parallel and expert-parallel dispatch in vLLM/SGLang; structurally inert in the Transformers reference. |

## 14. Implementation Notes for Engineers

A checkpoint-matching implementation reproduces these facts:

- 78 base decoder layers plus 1 MTP module at layer index 78.
- Hidden size 6144, vocabulary size 154880.
- 64 attention heads with MLA: `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`.
- Partial RoPE on the trailing 64 dimensions of each Q head and on the dedicated single-head 64-dim K-rope stream; theta = `1_000_000`; max position = `202_752`.
- DSA indexer on every layer with 32 heads, head dim 128, `index_topk=2048`. Indexer uses LayerNorm (with bias) on K, RMSNorm-free Q path, FP32-friendly `weights_proj`, and an indexer-private K cache.
- Per-layer `indexer_types` switch controls whether each layer recomputes its top-k or reuses the previous layer's selection.
- First three layers (`first_k_dense_replace=3`) are dense SwiGLU MLPs with intermediate size 12288. The remaining 75 base layers are MoE with `num_experts_per_tok=8` of `n_routed_experts=256` plus 1 shared expert at `moe_intermediate_size=2048`.
- MoE routing is sigmoid scoring + `e_score_correction_bias` + group routing (`n_group=1`, `topk_group=1`, effectively no-op) + `norm_topk_prob` + `routed_scaling_factor=2.5`.
- Untied embeddings and LM head; `lm_head` is its own bf16 tensor of shape `(154880, 6144)`.
- bf16 weights for everything except the float32 router weights, the float32 router selection bias, and the float32 indexer `weights_proj` (the latter is bf16 on disk in this release but kept in float32 by the Transformers loader).
- One MTP module's worth of weights at `model.layers.78.*` (extra `enorm`, `hnorm`, `eh_proj`, `shared_head.norm`) is on disk; whether to load and run it is a runtime choice.

Exact packaging parity also requires honoring the runtime expectations described above: vLLM and SGLang consult `rope_interleave` and `indexer_rope_interleave` to choose RoPE styles, while the first-party Transformers reference uses split-half RoPE unconditionally.

## 15. Bibliography

- Hugging Face repo: `zai-org/GLM-5.1` (config.json, tokenizer_config.json, generation_config.json, chat_template.jinja, model.safetensors.index.json, README.md).
- Hugging Face repo: `zai-org/GLM-5.1-FP8` (FP8 variant; weights only).
- Transformers: `src/transformers/models/glm_moe_dsa/{configuration,modular,modeling}_glm_moe_dsa.py`.
- vLLM: `vllm/model_executor/models/deepseek_v2.py` (`GlmMoeDsaForCausalLM`, `DeepseekV2MLAAttention`, `Indexer`, `SparseAttnIndexer`, `DeepseekV32IndexerCache`) and `vllm/model_executor/models/registry.py`.
- SGLang: `python/sglang/srt/configs/model_config.py` and `python/sglang/srt/models/deepseek_v2.py`.
- Z.ai blog post and GLM-5 technical report (referenced from the model card).
