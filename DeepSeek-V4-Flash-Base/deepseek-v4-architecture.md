# DeepSeek-V4-Flash-Base Architecture Reference

This document describes the shipped DeepSeek-V4-Flash-Base release artifacts as they are packaged: the serialized model config, the safetensors weight index, the tokenizer, and the converging (and frequently diverging) behavior across the available reference implementations.

Unlike most modern open-weights releases, the four reference points for DeepSeek-V4 do not converge on a single source of truth. The Hugging Face repository at `deepseek-ai/DeepSeek-V4-Flash-Base` ships only weights, a Hugging Face-style `config.json`, and a tokenizer; it contains no Python modeling code. The Hugging Face Transformers main branch has no `deepseek_v4` module yet. The vLLM and SGLang implementations are open pull requests as of the time of writing (vLLM PR #40760, SGLang PR #23600), neither merged into their respective main branches. The only canonical implementation is the `inference/` directory shipped under `deepseek-ai/DeepSeek-V4-Pro/inference/`, which the paper itself ([DeepSeek-V4 tech report, footnote on page 9](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)) names as the open-source implementation that "specifies more details unambiguously".

The released safetensors do not use Hugging Face tensor naming. They use the reference implementation's naming (no `model.` prefix; `attn` instead of `self_attn`; `ffn` instead of `mlp`; `wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b` / `w1` / `w2` / `w3` instead of `q_a_proj` / `q_b_proj` / etc.). The shipped `config.json` advertises `"architectures": ["DeepseekV4ForCausalLM"]` but no first-party Transformers class with that name exists, and the included `convert.py` is the bridge between HF-style names and reference-style names.

## 1. Model Summary

DeepSeek-V4-Flash-Base is a decoder-only causal language model with a hybrid attention stack (sliding-window + Compressed Sparse Attention + Heavily Compressed Attention), Manifold-Constrained Hyper-Connections instead of plain residuals, sparse Mixture-of-Experts feed-forward blocks (with hash routing on the first three layers and noaux_tc routing thereafter), and one Multi-Token Prediction module appended after the main stack.

- 43 base transformer blocks plus 1 MTP block.
- Residual width 4096, hyper-connection multiplier 4 (so the residual stream stores `hc_mult * dim = 16384` per token).
- 64 query heads with a single shared key/value head (MQA), per-head `head_dim = 512`, of which `rope_head_dim = 64` are rotated.
- Q low-rank decomposition: `q_lora_rank = 1024`. Output low-rank decomposition: `o_lora_rank = 1024` split across `o_groups = 8` groups.
- Layer attention pattern (`compress_ratios`): `[0, 0, 4, 128, 4, 128, ..., 4, 0]` (44 entries). Layers 0 and 1 are sliding-window only (window 128). Layers 2 through 42 alternate CSA (compress 4) and HCA (compress 128). The trailing entry at index 43 is the MTP block, which runs sliding-window only.
- DSA-style "lightning indexer" present only on CSA layers: 64 indexer heads of dim 128, top-512 selection over compressed blocks.
- 256 routed experts per layer plus 1 shared expert, `num_experts_per_tok = 6`. Routed-expert intermediate size 2048.
- First three layers use hash routing keyed on token id (a `tid2eid` table). Remaining layers use the DeepSeek `noaux_tc` recipe with a learned `bias` for selection.
- Routing scoring function is `sqrtsoftplus`: `sqrt(softplus(logits))`. `routed_scaling_factor = 2.5` (in the HF config; the inference reference uses `route_scale`, with the same effect).
- SwiGLU experts with a clipped activation: `swiglu_limit = 10.0` clamps both gate and up projections.
- Untied token embedding and LM head; no final RMSNorm scale beyond an all-ones gamma.
- Released weights are FP8 (E4M3) for attention/output linear modules and FP4 (E2M1) for routed-expert MLPs, with FP8-E8M0-format scales (`ue8m0`) on `(128, 128)` blocks for FP8 weights and on `(128, 32)` blocks for FP4 weights. Embedding, RMSNorm, hyper-connection mixers, gate weights, and bias tensors stay in BF16 / FP32.

The blog and tech report quote 284B total parameters with 13B activated per token. The on-disk `total_size` is `294_673_469_000` bytes across 69,189 named tensors. Most of that mass is FP4 routed-expert weight (1 byte covers two FP4 elements via `float4_e2m1fn_x2`).

## 2. Serialized `config.json`

These values are explicitly stored in the shipped `config.json` at `deepseek-ai/DeepSeek-V4-Flash-Base`.

| Field | Value |
| --- | --- |
| `architectures` | `["DeepseekV4ForCausalLM"]` |
| `model_type` | `"deepseek_v4"` |
| `torch_dtype` | `"bfloat16"` |
| `vocab_size` | `129280` |
| `hidden_size` | `4096` |
| `num_hidden_layers` | `43` |
| `num_attention_heads` | `64` |
| `num_key_value_heads` | `1` |
| `head_dim` | `512` |
| `qk_rope_head_dim` | `64` |
| `q_lora_rank` | `1024` |
| `o_lora_rank` | `1024` |
| `o_groups` | `8` |
| `hidden_act` | `"silu"` |
| `swiglu_limit` | `10.0` |
| `moe_intermediate_size` | `2048` |
| `n_routed_experts` | `256` |
| `n_shared_experts` | `1` |
| `num_experts_per_tok` | `6` |
| `norm_topk_prob` | `true` |
| `routed_scaling_factor` | `1.5` |
| `scoring_func` | `"sqrtsoftplus"` |
| `topk_method` | `"noaux_tc"` |
| `num_hash_layers` | `3` |
| `num_nextn_predict_layers` | `1` |
| `sliding_window` | `128` |
| `compress_ratios` | 44 entries: `[0, 0, 4, 128, 4, 128, ..., 4, 128, 4, 0]` |
| `index_topk` | `512` |
| `index_head_dim` | `128` |
| `index_n_heads` | `64` |
| `indexer_rope_interleave` | (not in this config; appears in V4-Pro variants) |
| `compress_rope_theta` | `160000` |
| `rope_theta` | `10000` |
| `rope_scaling` | `{"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}` |
| `max_position_embeddings` | `1048576` |
| `hc_mult` | `4` |
| `hc_sinkhorn_iters` | `20` |
| `hc_eps` | `1e-6` |
| `rms_norm_eps` | `1e-6` |
| `attention_bias` | `false` |
| `attention_dropout` | `0.0` |
| `tie_word_embeddings` | `false` |
| `use_cache` | `true` |
| `bos_token_id` | `0` |
| `eos_token_id` | `1` |
| `quantization_config.quant_method` | `"fp8"` |
| `quantization_config.fmt` | `"e4m3"` |
| `quantization_config.scale_fmt` | `"ue8m0"` |
| `quantization_config.weight_block_size` | `[128, 128]` |
| `quantization_config.activation_scheme` | `"dynamic"` |

A few important notes on these values:

- `head_dim = 512` is the per-head dimension in the original (unrotated) Q/V space. It is *much* larger than V3-family head dims (128 or 192+64). The K side is shared across all heads — `num_key_value_heads = 1`, so K is a single `head_dim = 512` vector per token, of which the trailing 64 dims are rotary.
- The HF `config.json` does *not* declare a `kv_lora_rank` or a separate `qk_nope_head_dim` / `v_head_dim`. The model is not MLA in the V3 sense. SGLang's wrapper config invents `kv_lora_rank=512`, `qk_nope_head_dim=448`, `v_head_dim=512`, and `n_group=8` to bridge into its existing DeepSeek code paths; those values are derived, not stored.
- `compress_ratios` is a per-layer attention selector: `0` means sliding-window only; `4` means CSA (compress KV by 4x along the sequence and apply DSA-style top-k indexer); `128` means HCA (compress KV by 128x and run dense attention on the compressed block stream). The 44-entry length covers the 43 base layers (`compress_ratios[0..42]`) plus the MTP layer (`compress_ratios[43]`).
- The Flash variant differs from V4-Pro in the leading layers. V4-Pro starts `[128, 128, 4, 128, ...]`; V4-Flash-Base starts `[0, 0, 4, 128, ...]`. The blog post's "layers 0–1 are HCA" description is V4-Pro-specific.
- `swiglu_limit = 10.0` is a clipped SwiGLU. The expert clamps both `gate_proj(x)` (max 10) and `up_proj(x)` (between -10 and +10) before computing `silu(gate) * up`. The shared expert in the inference reference runs without the limit.
- `routed_scaling_factor = 1.5` in this HF config; the inference reference's `inference/config.json` uses `route_scale = 2.5` for V4-Pro. Both serve the same role (post-normalization scaling on the top-k mixture weights).

## 3. Top-Level Module Structure

The packaged model has this structure (using the on-disk reference naming):

1. `embed.weight`: token embedding of shape `(vocab_size, dim) = (129280, 4096)`.
2. `layers`: 43 base blocks at prefixes `layers.0.*` through `layers.42.*`.
3. `mtp.0.*`: a single MTP block (architecturally identical to a sliding-only base block plus an embedding-projection front-end and a private `norm`).
4. `norm.weight`: final RMSNorm gamma over the residual stream (separate from the MTP's own `mtp.0.norm`).
5. `head.weight`: LM head of shape `(vocab_size, dim) = (129280, 4096)`.
6. `hc_head_base`, `hc_head_fn`, `hc_head_scale`: parameters of the final hyper-connection collapse from the `hc_mult`-wide residual stream down to a single hidden vector before the head.

`tie_word_embeddings` is false. `head.weight` is a separate tensor.

The base transformer block (`Block` in the reference) has two sub-blocks, each wrapped in its own pre/post hyper-connection mixing rather than a plain residual:

1. `hc_pre(attn)` -> `attn_norm` (RMSNorm) -> `Attention` -> `hc_post`.
2. `hc_pre(ffn)` -> `ffn_norm` (RMSNorm) -> `MoE` -> `hc_post`.

Each block carries six mHC parameters (`hc_attn_base`, `hc_attn_fn`, `hc_attn_scale`, `hc_ffn_base`, `hc_ffn_fn`, `hc_ffn_scale`).

The MTP block adds three pieces on top of a regular block: an embedding projection (`enorm`, `e_proj`), a hidden-state projection (`hnorm`, `h_proj`), and its own `norm` plus its own private final hyper-connection collapse parameters (`hc_head_fn`, `hc_head_base`, `hc_head_scale`). It reuses the global `embed` and `head`.

## 4. Manifold-Constrained Hyper-Connections (mHC)

DeepSeek-V4 replaces every plain residual with a Manifold-Constrained Hyper-Connection (mHC). The residual stream is widened by `hc_mult = 4`, so each token actually carries `hc_mult * dim = 16384` activations between layers. Each block performs a learned weighted collapse to one `dim`-wide vector before the sub-block, runs the sub-block, and then expands back to `hc_mult` copies via a learned combination matrix. The combination matrix is constrained to the Birkhoff polytope (doubly stochastic) via 20 Sinkhorn-Knopp iterations per forward pass.

### Per-Layer Parameters

For each of the two sub-blocks (attention and FFN) and for the final head collapse:

- `hc_*_fn`: `(mix_hc, hc_dim)` where `mix_hc = (2 + hc_mult) * hc_mult = 24` and `hc_dim = hc_mult * dim = 16384`. This is the dynamic-component generator weight matrix.
- `hc_*_base`: `(mix_hc,)` static bias.
- `hc_*_scale`: `(3,)` learnable gating factors initialized to small values.

All three are stored in float32.

### hc_pre (collapse `(b, s, hc_mult, dim)` -> `(b, s, dim)` plus dynamic `post`/`comb`)

1. Flatten and float-cast the per-token `(hc_mult, dim)` slab to `(hc_mult * dim,)`.
2. Compute its inverse RMS norm.
3. Project through `hc_*_fn` to get a `(mix_hc,)`-wide mix vector, multiplied by the inverse RMS norm.
4. Decode that mix vector into three pieces via `hc_split_sinkhorn`:
   - `pre`: `(hc_mult,)` weights for collapsing the residual into a single vector. Computed via `2 * sigmoid(scale * mix + base)` (non-negative, bounded by 2).
   - `post`: `(hc_mult,)` weights for re-broadcasting the sub-block output into the residual.
   - `comb`: `(hc_mult, hc_mult)` doubly-stochastic combination matrix, projected to the Birkhoff polytope by `hc_sinkhorn_iters = 20` iterations of alternating row and column normalization on `exp(raw)`.
5. Sum-reduce across `hc_mult` weighted copies to yield the `(b, s, dim)` input to the sub-block.

### hc_post (re-broadcast)

1. Multiply the sub-block output by the dynamic `post` to get `hc_mult` per-stream contributions.
2. Apply the `comb` doubly-stochastic combination to the original residual.
3. Sum.

### Final Head Collapse

The final head uses a simpler collapse: only `pre` weights, no `comb` (since there is no further block). The `hc_head_fn`, `hc_head_base`, `hc_head_scale` parameters at the model top level produce the final `(hc_mult,)` weights via `2 * sigmoid(scale * mix + base)` (with the inference reference using `sigmoid(...) + hc_eps` rather than `2 * sigmoid(...)`; the `head` collapse formula is intentionally distinct from the per-block collapse).

The `_eps`-tagged `hc_head_*` parameter triplet duplicates inside the MTP block, because the MTP block reuses the global `embed` and `head` but needs its own collapse weights.

The mHC math is the source of all `hc_*` tensors in the safetensors index. There are:

- 6 mHC tensors per base block (3 for attention, 3 for FFN): `hc_attn_{base, fn, scale}`, `hc_ffn_{base, fn, scale}`.
- 3 mHC tensors at the model top level: `hc_head_{base, fn, scale}`.
- 3 additional mHC tensors inside the MTP block.

## 5. Attention Block

DeepSeek-V4 attention is *not* MLA in the DeepSeek-V3 sense. There is no `kv_lora_rank` and no separate `kv_b_proj` expansion. The KV path is a single full-width projection plus per-layer KV compression for the long-range path.

### Persistent Per-Layer Tensors

Every base block carries the following attention tensors:

- `attn.attn_sink`: `(num_attention_heads,)` float32 per-head learnable attention-sink logits, added to the softmax denominator (and never to the numerator). Lets each head independently down-weight all real positions to "near zero".
- `attn.q_norm.weight`: RMSNorm gamma over the `q_lora_rank = 1024` Q residual.
- `attn.kv_norm.weight`: RMSNorm gamma over the `head_dim = 512` shared KV vector.
- `attn.wq_a.weight + .scale`: Q LoRA down `(dim, q_lora_rank) = (4096, 1024)`, FP8.
- `attn.wq_b.weight + .scale`: Q LoRA up `(q_lora_rank, num_attention_heads * head_dim) = (1024, 32768)`, FP8.
- `attn.wkv.weight + .scale`: shared K/V projection `(dim, head_dim) = (4096, 512)`, FP8. There is *no* separate K and V projection — the same vector is used for both, and the model relies on its 512-wide capacity to do the work that V3-style separate K and V projections used to do.
- `attn.wo_a.weight + .scale`: grouped output LoRA down. Stored as `(num_attention_heads * head_dim // o_groups, o_groups * o_lora_rank) = (4096, 8192)`, FP8. Implemented as 8 independent `(num_attention_heads * head_dim // o_groups, o_lora_rank) = (4096, 1024)` GEMMs in the reference.
- `attn.wo_b.weight + .scale`: output LoRA up `(o_groups * o_lora_rank, dim) = (8192, 4096)`, FP8.
- `attn_norm.weight`: input RMSNorm gamma for the attention sub-block (applied after the mHC pre-collapse).

### Sliding-Only Layers (`compress_ratio = 0`, layers 0, 1, and the MTP block)

These layers have no `compressor` and no `indexer` modules. Each query attends only to the trailing `window_size = 128` tokens via a windowed cache plus the learnable per-head attention-sink slot. RoPE is applied to the trailing `rope_head_dim = 64` of each Q head and of the shared K vector. RoPE for sliding-only layers uses `rope_theta = 10_000` directly (YaRN scaling is disabled for these layers in the reference).

### CSA Layers (`compress_ratio = 4`, even-indexed layers from 2 to 42)

Each CSA layer adds:

- `attn.compressor.{ape, norm.weight, wgate.weight, wkv.weight}`: a "Compressor" sub-module whose role is to fold every 4 input tokens into a single compressed KV entry via a learned softmax-gated weighted sum.
  - `ape`: `(compress_ratio, 2 * head_dim) = (4, 1024)` float32 absolute positional bias added to the gate.
  - `wkv`: `(dim, 2 * head_dim) = (4096, 1024)` projection of the input hidden state to the compressed KV space; BF16/FP32 weight (no FP8 scale on disk).
  - `wgate`: `(dim, 2 * head_dim) = (4096, 1024)` projection that produces the gating logits used to softmax-pool 4 tokens into 1.
  - `norm.weight`: RMSNorm gamma over the compressed `head_dim = 512` vector.
  - The `2 *` multiplier on the head_dim comes from "overlap" — when `compress_ratio == 4`, the compressor uses overlapping windows (each compressed block consumes 8 tokens with stride 4) to soften block boundaries.
- `attn.indexer.compressor.{ape, norm.weight, wgate.weight, wkv.weight}`: a separate Compressor for the lightning indexer's own compressed-KV stream. Same shapes as the main compressor but its own parameters. The indexer compressor uses Hadamard rotation on its keys before FP4 simulation, while the main compressor stays in BF16.
- `attn.indexer.weights_proj.weight`: `(dim, index_n_heads) = (4096, 64)`. Per-head weights used in the indexer scoring kernel.
- `attn.indexer.wq_b.weight + .scale`: `(q_lora_rank, index_n_heads * index_head_dim) = (1024, 8192)`, FP8. The indexer reuses the post-norm Q residual (`q_norm(wq_a(x))`) — its own Q comes from a single matrix multiply on that residual, no separate down-projection.

The CSA forward pass is:

1. Compute Q (LoRA path: `wq_a` -> `q_norm` -> `wq_b`, then per-head RMSNorm on Q activations, then RoPE on the trailing 64 dims).
2. Compute the shared KV vector `wkv(x)`, RMSNorm it, RoPE the trailing 64 dims.
3. Run the lightning indexer (section 6) to obtain `index_topk = 512` compressed-block indices per query.
4. Run the main `compressor` to add new compressed-KV entries to the long-range cache (compressing every 4 input tokens via overlapping softmax-gated pooling).
5. Construct the per-query attended set as the union of: the recent `window_size = 128` tokens (sliding window), the top-512 indexer-selected compressed blocks, and the per-head attention-sink slot.
6. Sparse-attend (`sparse_attn` kernel) using the attended set, with softmax scale `head_dim^(-1/2) = 512^(-1/2)`.
7. Apply *inverse* RoPE to the attention output's trailing 64 dims (because the value vector — same as K — carried position embeddings, and the attention output's positional content needs to be undone).
8. Run the grouped output LoRA: split the `num_attention_heads * head_dim = 32768` attention output into 8 groups of width 4096, apply per-group `wo_a` to get 8 group-local 1024-wide intermediates, concatenate to 8192, and project through `wo_b` back to `dim = 4096`.

### HCA Layers (`compress_ratio = 128`, odd-indexed layers from 3 to 41)

Same as CSA, but:

- The main `compressor` uses `compress_ratio = 128` with no overlap and no indexer. KV entries are compressed 128:1 along the sequence.
- There is no indexer, no `indexer.*` tensors. Instead, every query attends densely over all (heavily-compressed) blocks.
- The per-query attended set is the union of the recent `window_size = 128` tokens and *all* compressed blocks, plus the attention-sink slot.

### YaRN

For CSA and HCA layers, RoPE on the indexer Q and on the compressed K stream uses `compress_rope_theta = 160_000` (or `40_000` in the V4-Pro inference reference) instead of the base `rope_theta = 10_000`. YaRN scaling is enabled with `factor = 16`, `original_max_position_embeddings = 65_536` (giving an effective max of `65_536 * 16 = 1_048_576`), `beta_fast = 32`, `beta_slow = 1`.

For sliding-only layers, RoPE uses the base `rope_theta = 10_000` and YaRN is disabled (`original_seq_len = 0` in the reference's `precompute_freqs_cis`).

### Attention Sink

The `attn_sink` parameter is a per-head learnable scalar that is exponentiated and added to the softmax denominator (only). Concretely, for each head `h`, the attention scores `s[h, :, :]` are softmax-normalized as `softmax(s) / (1 + exp(attn_sink[h]) / sum_k exp(s[h,:,k]))`, equivalent to inserting a virtual sink position whose logit is `attn_sink[h]`. This lets a head down-weight its real positions arbitrarily (the sink absorbs the leftover probability mass).

## 6. Lightning Indexer (DSA-style, CSA layers only)

The indexer is the same general design as DeepSeek-V3.2's DSA indexer, but applied to *compressed* blocks rather than raw tokens, and quantized to FP4 instead of FP8.

For each CSA layer:

1. The indexer's own `compressor` builds an FP4-quantized compressed key stream of length `seq_len / 4` (one entry per 4 input tokens). Hadamard rotation is applied to the compressed keys before FP4 quantization to spread entropy across dimensions.
2. The query for the indexer comes from `wq_b(q_residual)` where `q_residual = q_norm(wq_a(x))`. Reshape to `(batch, seq, index_n_heads = 64, index_head_dim = 128)`. RoPE the trailing 64 dims. Hadamard-rotate. FP4-quantize.
3. Compute per-head per-token scoring: `score[b, s, t] = sum_h ReLU(q[b, s, h, :] . k_compressed[b, t, :]) * weights_proj(x)[b, s, h] * (head_dim^(-1/2)) * (n_heads^(-1/2))`.
4. Top-`index_topk = 512` over the compressed-block axis to obtain the per-query selected compressed-block indices.

Because the indexer scores compressed blocks, the indexer's search space is `4x` shorter than the raw sequence. With `index_topk = 512` compressed blocks selected, the effective look-back window is up to `512 * 4 = 2_048` raw tokens of high-resolution context plus the full HCA pass on every layer.

The indexer K cache is private to the indexer (separate from the main attention cache), and it is sized to `max_seq_len / compress_ratio` entries.

## 7. MoE Block

Every base block has a sparse MoE feed-forward (no dense MLPs at all — `first_k_dense_replace` is effectively 0). Each block carries:

- `ffn.gate.weight`: `(n_routed_experts, dim) = (256, 4096)`, BF16 / FP32. The router projection.
- `ffn.gate.bias` (layers 3..42 and the MTP block) or `ffn.gate.tid2eid` (layers 0, 1, 2): the routing decision auxiliary.
- `ffn.experts.{0..255}.{w1, w2, w3}.weight + .scale`: 256 routed experts. Each expert is a SwiGLU MLP with `w1: (dim, moe_intermediate_size) = (4096, 2048)`, `w3: (dim, 2048)`, `w2: (2048, 4096)`. Stored as FP4 (`float4_e2m1fn_x2`) with FP8-E8M0 scales on `(128, 32)` blocks.
- `ffn.shared_experts.{w1, w2, w3}.weight + .scale`: a single shared SwiGLU MLP with the same per-expert shapes. Stored FP8 (not FP4) so it can run without expert-dispatch overhead and without the FP4 clip from `swiglu_limit`.
- `ffn_norm.weight`: RMSNorm gamma applied to the FFN sub-block input (after mHC pre-collapse).

### Routing

The router computes per-token logits via `linear(x.float(), gate.weight.float())`. The scoring function is `sqrtsoftplus`: `scores = sqrt(softplus(logits))`. Two routing modes coexist:

1. **Hash routing (layers 0, 1, 2)**: the gate has no `bias`. Instead, it has a `tid2eid` table of shape `(vocab_size, num_experts_per_tok) = (129280, 6)` int32. The expert indices for each token are looked up directly by token id: `indices = tid2eid[input_ids]`. The mixture weights still come from the `sqrtsoftplus`-scored router output, gathered at the predetermined indices and renormalized to sum to 1, then multiplied by `route_scale = 2.5` (or `1.5` in this HF config).
2. **Score routing (layers 3..42 and the MTP block)**: standard noaux_tc routing. Selection scores are `sqrtsoftplus(logits) + bias`. Take the top `num_experts_per_tok = 6` indices. The mixture weights are gathered from the original `sqrtsoftplus(logits)` (without the bias). Renormalize to sum to 1 and scale by `route_scale`.

Tooling tip: the conversion script collapses `e_score_correction_bias` (the V3 name) to `bias`. The shipped tensor is named `ffn.gate.bias` on disk.

### Expert Computation (Clipped SwiGLU)

For routed experts:

```
gate_pre = w1(x).float()
up_pre   = w3(x).float()
gate     = clamp(gate_pre, max=swiglu_limit)
up       = clamp(up_pre, min=-swiglu_limit, max=swiglu_limit)
y        = w2( silu(gate) * up * topk_weight )
```

The shared expert runs without `swiglu_limit` clipping. The clip prevents FP4 expert weights from amplifying activation outliers into overflow.

### Shared Expert Combine

The final FFN output is `sum_routed + shared_experts(x)`, where `sum_routed` is the renormalized weighted sum over the 6 selected routed experts.

## 8. MTP Block

The MTP module is appended *outside* the `layers` list (it lives at prefix `mtp.0.*`, not `layers.43.*`). It is structurally a sliding-only base block (with its own `attn.compressor`-free attention) plus three additional pieces:

- `mtp.0.enorm.weight`, `mtp.0.e_proj.weight + .scale`: RMSNorm + linear projection over the next-position embedding stream.
- `mtp.0.hnorm.weight`, `mtp.0.h_proj.weight + .scale`: RMSNorm + linear projection over the hidden-state stream from the main model.
- `mtp.0.norm.weight`: a private final RMSNorm before the (shared) LM head.
- `mtp.0.hc_head_{base, fn, scale}`: private final mHC collapse parameters.

Inputs are merged as `e_proj(enorm(emb)).unsqueeze(2) + h_proj(hnorm(h))`, then run through a regular Block forward, then collapsed via the private `hc_head_*` collapse, then through the (shared) `head`. The MTP block reuses the global `embed` and `head` weights.

The MTP attention runs sliding-window only (its `compress_ratio` is the trailing entry of `compress_ratios`, which is `0` for V4-Flash-Base).

## 9. Forward Pass Summary (Reference Implementation)

For input `(B, S)`:

1. `h = embed(input_ids)` of shape `(B, S, dim)`.
2. Expand to `hc_mult` copies: `h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)` of shape `(B, S, hc_mult, dim)`.
3. For each of 43 base blocks:
   a. `hc_pre(attn)` collapses the residual to `(B, S, dim)`.
   b. `attn_norm`.
   c. Attention (sliding-only / CSA / HCA depending on `compress_ratios[layer]`).
   d. `hc_post(attn)` re-expands and combines with the wide residual.
   e. `hc_pre(ffn)` collapses again.
   f. `ffn_norm`.
   g. MoE (hash for layers 0–2, noaux_tc for layers 3+).
   h. `hc_post(ffn)`.
4. Final `hc_head_*` collapse on the wide residual.
5. `norm` (RMSNorm with all-ones gamma, since no gamma is shipped at the top level — the parameter exists at `norm.weight` but the file ships it as the default ones tensor in the inference convention).
6. `head(hidden)` projects to logits of shape `(B, S, vocab_size)`.

The MTP block can run at the end of step 5 to produce a speculative next-position logit by re-mixing the hidden state with the next-position embedding through its private projections, then running its own block, then projecting through the shared head.

## 10. Quantization

The release stores most large linear weights in FP8 E4M3, with a small but important set in FP4 E2M1:

### FP4 (E2M1) — routed experts only

- `ffn.experts.*.{w1, w2, w3}.weight`: FP4, stored as `int8` packed two-per-byte. Shape on disk is `(out_features, in_features // 2)`.
- Companion scale tensor: `ffn.experts.*.{w1, w2, w3}.scale` of shape `(out_features, in_features // 32)` in FP8-E8M0 (`ue8m0`).

The `convert.py` script can losslessly upcast FP4 to FP8 via `cast_e2m1fn_to_e4m3fn` if you want to run on hardware without FP4 GEMM support.

### FP8 (E4M3) — most other linear weights

- All attention weights (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, `indexer.wq_b`).
- Shared-expert `w1` / `w2` / `w3` (these are *not* FP4).
- MTP block `e_proj`, `h_proj`.
- `weight_block_size = (128, 128)`. Each FP8 weight has a companion scale of shape `(out / 128, in / 128)` in FP8-E8M0 (`ue8m0`).

### BF16 / FP32

- `embed.weight`, `head.weight` (BF16 on disk; the inference reference upcasts `head.weight` to FP32 in memory).
- `norm.weight`, all `*_norm.weight` (RMSNorm gammas; BF16 on disk, FP32 in memory).
- All `hc_*` mHC parameters (FP32).
- `attn.attn_sink` (FP32).
- `ffn.gate.weight` (BF16; the router runs in FP32).
- `ffn.gate.bias` (FP32) and `ffn.gate.tid2eid` (int32).
- All compressor tensors (`compressor.ape`, `compressor.wkv`, `compressor.wgate`, `compressor.norm.weight`) — BF16 on disk, FP32 in memory.

The indexer scoring kernel runs activations and queries in FP4 simulation (Hadamard rotation + FP4 quantize), but the indexer's `weights_proj.weight` and `wq_b.weight` are stored as BF16 / FP8 respectively.

## 11. Weight Inventory

The safetensors index reports `metadata.total_size = 294_673_469_000` bytes across `69_189` named tensors and 46 safetensors shards.

Per-layer tensor counts (verified directly from `model.safetensors.index.json`):

| Layer family | Layers | Per-layer non-expert tensors | Per-layer expert tensors | Tensors per layer |
| --- | --- | --- | --- | --- |
| Sliding-only + hash routing (0, 1) | 2 | 23 | 1542 (256 routed × 6 + 6 shared) | 1565 |
| CSA + hash routing (2) | 1 | 34 | 1542 | 1576 |
| HCA + noaux_tc routing (3, 5, ..., 41) | 20 | 27 | 1542 | 1569 |
| CSA + noaux_tc routing (4, 6, ..., 42) | 20 | 34 | 1542 | 1576 |
| MTP block (mtp.0) | 1 | 35 (sliding + e_proj + h_proj + enorm + hnorm + own norm + own hc_head_*) | 1542 | 1577 |
| Top-level (`embed`, `head`, `norm`, `hc_head_{base, fn, scale}`) | - | 6 | - | 6 |
| **Total** | **44** | - | - | **69_189** |

Verification: `2*1565 + 1576 + 20*1569 + 20*1576 + 1577 + 6 = 3130 + 1576 + 31380 + 31520 + 1577 + 6 = 69189`.

Per-block component counts (matching the actual on-disk layout):

- Sliding-only attention (layers 0, 1, MTP-attn): 14 attn tensors = `attn_sink (1)` + `q_norm.weight (1)` + `kv_norm.weight (1)` + `wq_a.{weight, scale} (2)` + `wq_b.{weight, scale} (2)` + `wkv.{weight, scale} (2)` + `wo_a.{weight, scale} (2)` + `wo_b.{weight, scale} (2)` + `attn_norm.weight (1)`.
- CSA attention adds: `compressor.{ape, norm.weight, wgate.weight, wkv.weight} (4)` + `indexer.compressor.{ape, norm.weight, wgate.weight, wkv.weight} (4)` + `indexer.weights_proj.weight (1)` + `indexer.wq_b.{weight, scale} (2)` = 11 extra tensors.
- HCA attention adds: `compressor.{ape, norm.weight, wgate.weight, wkv.weight} (4)` only (no indexer) = 4 extra tensors.
- FFN block on every layer: `ffn_norm.weight (1)` + `ffn.gate.weight (1)` + `ffn.gate.bias OR tid2eid (1)` + 256 × 6 routed-expert tensors + 6 shared-expert tensors = 1545 tensors per layer.
- mHC per block: `hc_attn_{base, fn, scale} (3)` + `hc_ffn_{base, fn, scale} (3)` = 6 tensors.
- MTP-only extras: `enorm.weight (1)` + `hnorm.weight (1)` + `e_proj.{weight, scale} (2)` + `h_proj.{weight, scale} (2)` + `norm.weight (1)` + `hc_head_{base, fn, scale} (3)` + 6 mHC for the embedded block = `enorm.weight, hnorm.weight, e_proj.{w,s}, h_proj.{w,s}, norm.weight, hc_head_{base,fn,scale}` ≈ 12 extra tensors total over a sliding-only base.

The 13B-active claim corresponds to the parameters touched per token: the embedding, all attention weights, all routed-expert weights for the 6 chosen experts in each layer (plus the always-on shared expert), all gate weights, and the LM head.

## 12. Tokenizer and Prompt Formatting

DeepSeek-V4-Flash-Base ships only `tokenizer.json` and `tokenizer_config.json`; there is no chat template, no generation config, and no README in this repo (the model card on Hugging Face's web view is "No model card"). This is a base model — the post-trained variants ship at `deepseek-ai/DeepSeek-V4-Flash` and `deepseek-ai/DeepSeek-V4-Pro`.

- Tokenizer class: HuggingFace tokenizers Rust backend (`tokenizers` library), packaged as a single `tokenizer.json` blob.
- Vocabulary size: 129,280 tokens (matches `config.json`).
- BOS / EOS: declared in `config.json` as ids `0` and `1` respectively. Specific token strings are inside `tokenizer.json`.

For prompt formatting, refer to the post-trained variants.

## 13. Implementation Surfaces and Their Disagreements

The four reference points consulted for this document do *not* agree on a single representation of the model. The differences are not cosmetic — they affect tensor names, derived hyperparameters, and even what the model is "called".

### 13.1 The released checkpoint vs. the HF naming convention

The shipped safetensors tensor names follow the inference reference's naming, not Hugging Face Transformers' naming. Examples:

| HF Transformers name (typical) | DeepSeek-V4 on-disk name |
| --- | --- |
| `model.embed_tokens.weight` | `embed.weight` |
| `model.layers.0.input_layernorm.weight` | `layers.0.attn_norm.weight` |
| `model.layers.0.post_attention_layernorm.weight` | `layers.0.ffn_norm.weight` |
| `model.layers.0.self_attn.q_a_proj.weight` | `layers.0.attn.wq_a.weight` |
| `model.layers.0.self_attn.q_a_layernorm.weight` | `layers.0.attn.q_norm.weight` |
| `model.layers.0.self_attn.q_b_proj.weight` | `layers.0.attn.wq_b.weight` |
| `model.layers.0.self_attn.kv_a_proj_with_mqa.weight` | `layers.0.attn.wkv.weight` (and the V3 distinction between `kv_a` and `kv_b` does not exist in V4) |
| `model.layers.0.self_attn.o_proj.weight` | `layers.0.attn.wo_a.weight` + `layers.0.attn.wo_b.weight` (grouped LoRA, two tensors) |
| `model.layers.0.mlp.gate.weight` | `layers.0.ffn.gate.weight` |
| `model.layers.0.mlp.gate.e_score_correction_bias` | `layers.0.ffn.gate.bias` |
| `model.layers.0.mlp.experts.0.gate_proj.weight` | `layers.0.ffn.experts.0.w1.weight` |
| `model.layers.0.mlp.experts.0.up_proj.weight` | `layers.0.ffn.experts.0.w3.weight` |
| `model.layers.0.mlp.experts.0.down_proj.weight` | `layers.0.ffn.experts.0.w2.weight` |
| `lm_head.weight` | `head.weight` |
| (no equivalent) | `attn_sink`, `compressor.*`, `indexer.*`, `hc_*` |

Anyone loading the shipped weights with a standard HF Transformers `AutoModelForCausalLM` will fail outright until either (a) Transformers gains a `deepseek_v4` module or (b) the user runs `inference/convert.py` against a freshly HF-renamed checkpoint. The `convert.py` script in the inference reference is *bidirectionally* idempotent on already-renamed names because all of its `replace`s have no effect when the target string is already absent.

### 13.2 Hugging Face Transformers

There is no `transformers/models/deepseek_v4/` module on the main branch as of this writing. The shipped `config.json` declares `transformers_version = "4.57.1"` and `architectures = ["DeepseekV4ForCausalLM"]`, but the corresponding class does not exist in the upstream Transformers package. Any code that does `AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-Base")` will succeed (it loads the JSON dict); any code that does `AutoModelForCausalLM.from_pretrained(...)` will fail with an unknown architecture error.

### 13.3 vLLM (PR #40760, branch `zyongye/vllm:dsv4`)

vLLM's PR #40760 adds first-class `DeepseekV4ForCausalLM` support. Key files:

- `vllm/model_executor/models/deepseek_v4.py`: top-level model definition.
- `vllm/model_executor/layers/deepseek_v4_attention.py` (1062 lines): MLA-style attention with the V4-specific extensions. Uses `from transformers import DeepseekV2Config, DeepseekV3Config` (V4 has no first-party Transformers config).
- `vllm/model_executor/layers/deepseek_compressor.py` (436 lines): the CSA / HCA Compressor.
- `vllm/model_executor/layers/mhc.py` (436 lines): Manifold-Constrained Hyper-Connections.
- `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (477 lines), `csrc/moe/topk_softplus_sqrt_kernels.cu` (715 lines): hand-fused CUDA kernels for the Q-norm + RoPE + KV-insert path and for the `topk_softplus_sqrt` router.
- Registry (`vllm/model_executor/models/registry.py`): `"DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM")` and `"DeepSeekV4MTPModel": ("deepseek_v4_mtp", "DeepSeekV4MTP")`.

The PR description warns: *"This model implementation is highly optimized. All the component is coupled. Lot of manually fused kernel. Please consult @WoosukKwon @zyongye @ivanium before making any changes."* This is an unusually tight coupling for a vLLM model implementation, and not all functionality is exposed through the standard `AttentionImpl` interface.

vLLM's attention layer organizes its parameters as `DeepseekV4MLAModules`: `fused_wqa_wkv` (a single fused linear that produces both the Q-LoRA-down output and the shared KV vector in one GEMM), `q_norm`, `wq_b`, `kv_norm`, `wo_a`, `wo_b`, `attn_sink`, `rotary_emb`, `indexer`, `indexer_rotary_emb`. The fused `wqa_wkv` is a vLLM-side fusion; on disk the weights remain `wq_a` and `wkv` separate.

### 13.4 SGLang (PR #23600, branch `sgl-project/sglang:deepseek_v4`)

SGLang's PR #23600 adds dedicated DeepSeek-V4 support across many layers:

- `python/sglang/srt/configs/deepseek_v4.py` (72 lines): a standalone `DeepSeekV4Config` dataclass that *invents* MLA-style fields (`kv_lora_rank=512`, `qk_nope_head_dim=448`, `qk_rope_head_dim=64`, `v_head_dim=512`, `n_group=8`, `topk_group=8`, `intermediate_size=2048`) to bridge into SGLang's existing DeepSeek-V2/V3 code paths. The `model_type` it advertises is `"deepseek_ref"`, not `"deepseek_v4"`.
- `python/sglang/srt/layers/mhc.py` (643 lines): the mHC implementation.
- `python/sglang/srt/layers/deepseek_v4_rope.py` (196 lines): the dual-RoPE setup (sliding-only `rope_theta = 10_000` vs. compressed `compress_rope_theta`).
- `python/sglang/srt/layers/moe/deepseek_v4_topk.py` (250 lines): the `sqrtsoftplus` + hash-routing top-k logic.
- `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` (1333 lines): the attention backend with sliding + sparse + heavy-compressed paths.
- `python/sglang/srt/layers/attention/nsa/index_buf_accessor_v4.py`, `quant_k_cache_v4.py`: NSA (the SGLang umbrella name for sparse attention) plumbing.
- `python/sglang/srt/models/deepseek_v4.py` (2098 lines), `deepseek_v4_nextn.py` (248 lines): the model + MTP module.
- `python/sglang/srt/mem_cache/deepseekv4_memory_pool.py` (810 lines): KV cache pool tuned for sliding + compressed entries.
- `python/sglang/srt/layers/quantization/mxfp4_deepseek.py` (477 lines): FP4 expert quantization support.
- 19 `.cuh` CUDA kernel headers under `python/sglang/jit_kernel/csrc/deepseek_v4/` (compressors for ratio 4 and 128, fused norm-rope, hash-topk, paged MQA metadata, RMSNorm, RoPE, SiLU+mul masked post quant, top-k variants).

The SGLang config's invented MLA-style fields are derived from the actual on-disk shapes:
- `kv_lora_rank = 512` is `head_dim` — there is no LoRA, but SGLang treats the single shared KV vector as a "rank-512 KV latent".
- `qk_nope_head_dim = 448` and `qk_rope_head_dim = 64` correspond to `head_dim - rope_head_dim = 512 - 64 = 448` and the rope partition.
- `v_head_dim = 512` corresponds to the K = V vector width.
- `n_group = 8`, `topk_group = 8` — both equal `o_groups`. Since `topk_group == n_group`, the V3-style group-mask routing degenerates into a pure top-k.

These derivations are necessary because SGLang reuses MLA primitives. They are not part of the model's actual specification.

### 13.5 The official inference reference (`deepseek-ai/DeepSeek-V4-Pro/inference/`)

Six Python files (`model.py` 38.6 kB, `kernel.py` 22.2 kB, `convert.py` 7.1 kB, `generate.py` 6.3 kB, plus `config.json`, `requirements.txt`, `README.md`). This is the implementation that the paper itself names as the source of truth for ambiguous architectural details.

The shipped `inference/config.json` is for V4-Pro (`n_layers=61`, `dim=7168`, `n_heads=128`, `n_routed_experts=384`, `n_activated_experts=6`, `q_lora_rank=1536`, `head_dim=512`, `o_groups=16`, `compress_ratios` starts with `[128, 128, 4, 128, ...]`). It is not the V4-Flash-Base config. To run the reference against V4-Flash-Base weights, edit `inference/config.json` to: `n_layers=43`, `dim=4096`, `n_heads=64`, `n_routed_experts=256`, `n_activated_experts=6`, `n_hash_layers=3`, `n_shared_experts=1`, `q_lora_rank=1024`, `o_lora_rank=1024`, `o_groups=8`, `moe_inter_dim=2048`, `swiglu_limit=10.0`, `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`, `compress_ratios=[0,0,4,128,...,4,0]`, `compress_rope_theta=160000`. The exact field naming differs from the HF `config.json` (the inference reference uses `n_layers` not `num_hidden_layers`, `dim` not `hidden_size`, etc.).

The reference implementation does not consume the HF `config.json`. It consumes its own JSON whose fields match the `ModelArgs` dataclass keys.

### 13.6 Summary of disagreements

| Topic | HF `config.json` | Inference `inference/config.json` | vLLM PR #40760 | SGLang PR #23600 |
| --- | --- | --- | --- | --- |
| Field naming | HF style (`num_hidden_layers`, `hidden_size`, ...) | Reference style (`n_layers`, `dim`, ...) | HF style + `DeepseekV2Config` reuse | HF style + invented MLA fields |
| Tensor naming | (declares HF style by `architectures` field) | Reference style (`wq_a`, `wkv`, `w1`, ...) | HF style internally; converts on load | HF style internally; converts on load |
| Routing function name | `"sqrtsoftplus"` (in HF config) | `"sqrtsoftplus"` (named `score_func`) | `topk_softplus_sqrt` (kernel name) | `"sqrtsoftplus"` |
| MTP storage location | `mtp.0.*` on disk; `num_nextn_predict_layers=1` declares its existence | `Transformer.mtp[0]` (separate from `Transformer.layers`) | `DeepSeekV4MTPModel` (separate registry entry) | `deepseek_v4_nextn.py` (separate model file) |
| K/V latent | (not declared) | (not declared; KV is `head_dim`-wide direct) | Fused `wqa_wkv` | Invented `kv_lora_rank=512`, `qk_nope_head_dim=448`, `v_head_dim=512` |
| `compress_ratios[-1]` (the MTP entry) | `0` for V4-Flash-Base | (matches HF) | (consumed by the MTP model class) | (consumed by the nextn model class) |
| Final model `norm.weight` | (present at top level on disk) | All-ones default; weight is shipped but only as a pass-through | (loaded as RMSNorm) | (loaded as RMSNorm) |

## 14. Implementation Notes for Engineers

A checkpoint-matching implementation reproduces these facts:

- 43 base blocks plus 1 MTP block at prefix `mtp.0.*`. No `model.` prefix on any tensor.
- Per-block residual width 4096 expanded to `hc_mult * dim = 16384` for hyper-connection routing. Two mHC mixers per block (one for attention sub-block, one for FFN sub-block); each mixer carries `hc_*_fn` `(24, 16384)`, `hc_*_base` `(24,)`, `hc_*_scale` `(3,)` in float32.
- mHC pre-collapse uses `hc_split_sinkhorn` which produces `pre`, `post`, and a `(hc_mult, hc_mult)` doubly-stochastic combination matrix via 20 Sinkhorn-Knopp iterations. The Birkhoff-polytope projection is the "manifold-constrained" part of mHC.
- Final head collapse uses a different formula (`sigmoid(scale*mix + base) + hc_eps`) than per-block collapse (`2 * sigmoid(scale*mix + base)`).
- 64 query heads with a single shared K/V head, `head_dim = 512`, of which the trailing 64 dims are rotated. RoPE base is `10_000` for sliding-only layers and `160_000` for CSA/HCA layers. YaRN scaling is enabled for CSA/HCA paths only, with `factor = 16` and `original_max_position_embeddings = 65_536`.
- Q path is LoRA: `wq_a` `(4096, 1024)` -> `q_norm` (RMSNorm) -> `wq_b` `(1024, 32768)`. Per-head RMSNorm is applied to Q activations after `wq_b` (an additional normalization beyond `q_norm`).
- KV path is a single shared projection `wkv` `(4096, 512)` followed by `kv_norm` (RMSNorm). RoPE goes on the trailing 64 dims of the same vector. There is no separate K vs V projection.
- `attn_sink` is a per-head learnable float32 logit added only to the softmax denominator (acts as a virtual position with logit `attn_sink[h]`).
- `compress_ratios[layer] in {0, 4, 128}` selects sliding-only, CSA, or HCA respectively. The MTP block's value is the trailing entry of the array.
- CSA layers carry both a main `compressor` and an `indexer.compressor` (separate parameters). The indexer scores compressed blocks with FP4 simulation (Hadamard rotation + FP4 quantize) and selects top-512 compressed blocks per query. HCA layers carry only the main `compressor` and run dense attention over all compressed blocks.
- Output projection is grouped LoRA: split the attention output into 8 groups of 4096 along the `num_attention_heads * head_dim = 32768` axis, apply per-group `wo_a` to get 8 group-local 1024-wide intermediates, project all 8192 through `wo_b` back to 4096.
- After attention, apply *inverse* RoPE to the trailing 64 dims of the attention output (because the value vector — equal to K — carried position embeddings, and the output's positional content needs to be undone before the residual is added back).
- 256 routed experts per layer + 1 shared expert. `num_experts_per_tok = 6`. Routed experts are FP4 (E2M1) packed two-per-byte with FP8-E8M0 scales on `(out, in/32)` blocks. Shared expert is FP8 (E4M3) with `(out/128, in/128)` scales.
- Routing scoring is `sqrt(softplus(logits))`. First three layers (`num_hash_layers = 3`) use a `tid2eid` `(vocab_size, num_experts_per_tok)` int32 table for expert selection (looked up by token id), with mixture weights still coming from `sqrtsoftplus`. Remaining layers use noaux_tc with a learned `bias` for selection.
- Routed experts use a clipped SwiGLU: clamp `gate` to `<= 10.0`, clamp `up` to `[-10.0, 10.0]`. Shared expert is unclipped.
- Token embedding and LM head are untied. There is a top-level `norm.weight` that the inference reference defaults to all-ones if not loaded.
- The model is *not* an MLA model in the V3 sense. There is no `kv_lora_rank`, no `kv_b_proj`, no separate `qk_nope_head_dim` and `v_head_dim`. SGLang's wrapper invents these for code reuse; do not be misled.

Exact packaging parity also requires accepting that there is no Hugging Face Transformers integration yet, that vLLM and SGLang implementations are open PRs (not main-branch code) at the time of this writing, and that the `inference/` reference at `deepseek-ai/DeepSeek-V4-Pro/inference/` is the only authoritative source.

## 15. Bibliography

- Hugging Face repo: `deepseek-ai/DeepSeek-V4-Flash-Base` — `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json`, 46 safetensors shards. No README, no chat template, no generation config, no Python modeling code.
- Inference reference: `deepseek-ai/DeepSeek-V4-Pro/inference/` — `model.py`, `kernel.py`, `convert.py`, `generate.py`, `config.json`, `README.md`, `requirements.txt`. The paper points to this as the authoritative implementation for unambiguous details.
- Tech report: `deepseek-ai/DeepSeek-V4-Pro/DeepSeek_V4.pdf` — "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence".
- vLLM PR #40760 (branch `zyongye/vllm:dsv4`) — "[New Model] Support DeepseekV4". Open as of this writing.
- SGLang PR #23600 (branch `sgl-project/sglang:deepseek_v4`) — "DeepSeek V4". Open as of this writing.
- vLLM blog post: "DeepSeek V4 in vLLM: Efficient Long-context Attention".
- SGLang cookbook: `sgl-project/sgl-cookbook/docs/autoregressive/DeepSeek/DeepSeek-V4.md`.
- HuggingFace blog: "DeepSeek-V4: a million-token context that agents can actually use".
