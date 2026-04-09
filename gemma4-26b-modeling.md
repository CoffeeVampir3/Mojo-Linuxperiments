# Gemma 4 26B-A4B — Implementation Reference

Model: `google/gemma-4-26B-A4B`
Architecture class: `Gemma4ForConditionalGeneration` (we implement text decoder only)
Total params: ~25.2B, Active params per token: ~3.8B (MoE with 128 experts, top-8)
Weights dtype: BF16 (all weights)
Tied embeddings: Yes (`lm_head.weight` = `embed_tokens.weight`)


## Global Structure

```
ScaledEmbedding(262144, 2816, scale=sqrt(2816))
  → 30x DecoderLayer
  → RMSNorm(2816, eps=1e-6)
LM Head: tied to embedding (no separate weight)
  → Logit Softcapping: tanh(logits / 30.0) * 30.0
```


## Dimensions

| Parameter                  | Value       |
|----------------------------|-------------|
| vocab_size                 | 262144      |
| hidden_size                | 2816        |
| num_hidden_layers          | 30          |
| num_attention_heads        | 16          |
| head_dim (sliding)         | 256         |
| global_head_dim (full)     | 512         |
| num_kv_heads (sliding)     | 8           |
| num_kv_heads (full/global) | 2           |
| intermediate_size (dense)  | 2112        |
| moe_intermediate_size      | 704         |
| num_experts                | 128         |
| top_k_experts              | 8           |
| sliding_window             | 1024        |
| max_position_embeddings    | 262144      |
| rms_norm_eps               | 1e-6        |
| final_logit_softcapping    | 30.0        |


## Layer Pattern

5:1 ratio of sliding to full attention. 30 layers total:

```
[ 0] sliding    [ 6] sliding    [12] sliding    [18] sliding    [24] sliding
[ 1] sliding    [ 7] sliding    [13] sliding    [19] sliding    [25] sliding
[ 2] sliding    [ 8] sliding    [14] sliding    [20] sliding    [26] sliding
[ 3] sliding    [ 9] sliding    [15] sliding    [21] sliding    [27] sliding
[ 4] sliding    [10] sliding    [16] sliding    [22] sliding    [28] sliding
[ 5] full       [11] full       [17] full       [23] full       [29] full
```

Full attention layers: {5, 11, 17, 23, 29}


## Embedding

Scaled word embedding: output = `Embedding(input_ids) * sqrt(2816)`.

The scale factor `sqrt(2816) ≈ 53.066` is stored as a buffer (not a learned param) but
gets cast to BF16 at runtime, which rounds it to `53.5`. This is a known Gemma behavior.
The lm_head shares the same weight matrix (tied embeddings).


## Decoder Layer

Each layer has the following structure:

```
residual = x

x = input_layernorm(x)
x = attention(x)
x = post_attention_layernorm(x)
x = residual + x

residual = x

x = pre_feedforward_layernorm(x)
x = dense_mlp(x)

# MoE branch (parallel to dense MLP, fed from the RESIDUAL)
x1 = post_feedforward_layernorm_1(x)              # norm dense MLP output
_, weights, indices = router(residual)              # route from pre-MLP hidden states
x2 = pre_feedforward_layernorm_2(residual)          # norm input to experts
x2 = experts(x2, indices, weights)
x2 = post_feedforward_layernorm_2(x2)              # norm expert output
x = x1 + x2                                        # combine dense + sparse

x = post_feedforward_layernorm(x)
x = residual + x

x = x * layer_scalar                               # per-layer learned scalar (init=1.0)
```

Key detail: the dense MLP and MoE experts receive the **same input** (pre-MLP hidden states)
through separate norms. Their outputs are summed before the final post-feedforward norm.

The layer has 7 RMSNorm instances (all with learnable scale):
- `input_layernorm`
- `post_attention_layernorm`
- `pre_feedforward_layernorm`
- `post_feedforward_layernorm`
- `post_feedforward_layernorm_1` (MoE: after dense)
- `pre_feedforward_layernorm_2` (MoE: before experts)
- `post_feedforward_layernorm_2` (MoE: after experts)

Plus `layer_scalar`: a single learned float per layer, multiplied into the final output.


## RMSNorm

Two variants: with and without a learnable scale parameter.

```
def rms_norm(x, weight=None, eps=1e-6):
    x_f32 = x.float()
    mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True) + eps
    normed = x_f32 * pow(mean_sq, -0.5)      # use pow, not rsqrt (JAX compat)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(x.dtype)
```

All computation in FP32, cast back to input dtype at the end.
Uses `pow(x, -0.5)` rather than `rsqrt(x)` for cross-framework reproducibility.


## Attention

Two attention configurations depending on layer type:

### Sliding Attention (25 layers)

```
Q proj:  Linear(2816 → 4096, no bias)     = 16 heads × 256 dim
K proj:  Linear(2816 → 2048, no bias)     = 8 heads × 256 dim
V proj:  Linear(2816 → 2048, no bias)     = 8 heads × 256 dim
O proj:  Linear(4096 → 2816, no bias)

GQA ratio: 2 (each KV head serves 2 Q heads)
Sliding window: 1024 tokens
RoPE: default, theta=10000, full 256-dim rotation
```

### Full (Global) Attention (5 layers)

```
Q proj:  Linear(2816 → 8192, no bias)     = 16 heads × 512 dim
K proj:  Linear(2816 → 1024, no bias)     = 2 heads × 512 dim
V proj:  None (K=V, key projection output reused as values)
O proj:  Linear(8192 → 2816, no bias)

GQA ratio: 8 (each KV head serves 8 Q heads)
No sliding window (full causal attention)
RoPE: proportional, theta=1000000, partial rotation (128 of 512 dims)
```

### QKV Norms

All attention layers apply per-head RMSNorm to Q, K, and V **before** RoPE and attention:

```
q_norm: RMSNorm(head_dim, eps=1e-6, with_scale=True)    # learnable scale
k_norm: RMSNorm(head_dim, eps=1e-6, with_scale=True)    # learnable scale
v_norm: RMSNorm(head_dim, eps=1e-6, with_scale=False)   # no learnable scale
```

For full attention layers where K=V: both `k_norm` and `v_norm` are applied to the same
projection output, but they produce different results because `k_norm` has a learnable
scale and K additionally gets RoPE applied. V gets neither learned scale nor RoPE.

### Attention Scaling

`scaling = 1.0` (NOT `1/sqrt(head_dim)`). The QK norms replace the need for standard scaling.

### Attention Computation

```
QK = (Q @ K^T) * 1.0                   # scaling = 1.0
QK = QK + causal_mask                  # -inf for masked positions
attn_weights = softmax(QK, dim=-1)     # upcast to FP32 for softmax
output = attn_weights @ V
```

No softcapping in attention (only on final logits).


## RoPE (Rotary Position Embedding)

Uses standard `rotate_half` formulation (Llama-style):

```
def rotate_half(x):
    x1 = x[..., :dim//2]
    x2 = x[..., dim//2:]
    return cat(-x2, x1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

Two separate RoPE configurations, one per layer type:

### Sliding Layers
- theta = 10,000
- Full rotation: all 256 dims of head_dim participate
- inv_freq = 1 / (10000 ^ (arange(0, 256, 2) / 256)), giving 128 frequencies
- cos/sin computed as cat(freqs, freqs) → 256 dims

### Full Attention Layers
- theta = 1,000,000
- Partial rotation via proportional RoPE: only 128 of 512 dims get position encoding
- `partial_rotary_factor = 0.25` → `rope_angles = floor(0.25 * 512 / 2) = 64`
- inv_freq has 64 real frequencies, zero-padded to 256 total:
  - `inv_freq_rotated = 1 / (1000000 ^ (arange(0, 128, 2) / 512))` — 64 values
  - `inv_freq = cat(inv_freq_rotated, zeros(192))` — 256 values total
- cos/sin = cat(freqs, freqs) → 512 dims
- Zero-padded frequencies produce cos=1, sin=0 → those dims pass through unchanged
- The frequency denominator uses the **full** head_dim (512), not the rotary dim (128)

Both are precomputed once and reused across layers of the same type.


## Dense MLP (every layer)

Standard gated MLP with GELU activation:

```
gate = gate_proj(x)          # Linear(2816 → 2112, no bias)
up   = up_proj(x)            # Linear(2816 → 2112, no bias)
out  = down_proj(gelu(gate) * up)  # Linear(2112 → 2816, no bias)
```

Activation: `gelu_pytorch_tanh` = GELU with tanh approximation:
```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```


## MoE (every layer, parallel to dense MLP)

### Router

The router has a unique normalization structure:

```
def router(x):
    x = rms_norm(x, scale=None)               # RMSNorm without learnable scale
    x = x * self.scale * (1 / sqrt(2816))     # learnable per-dim scale + constant
    logits = linear(x, weight, bias=None)      # Linear(2816 → 128, no bias)

    probs = softmax(logits, dim=-1)            # softmax over all 128 experts
    weights, indices = topk(probs, k=8)        # select top-8

    weights = weights / weights.sum(dim=-1)    # renormalize to sum=1
    weights = weights * per_expert_scale[indices]  # learnable per-expert scale

    return weights, indices
```

Parameters:
- `norm`: RMSNorm(2816, no scale)
- `proj.weight`: (128, 2816)
- `scale`: (2816,) — per-dimension learnable scale, init=1.0
- `per_expert_scale`: (128,) — per-expert learnable scale, init=1.0

### Experts

128 experts, 8 active per token. Standard gated MLP per expert:

```
gate_up = x @ gate_up_proj[expert]        # (2816 → 1408)
gate, up = chunk(gate_up, 2, dim=-1)      # each 704
hidden = gelu(gate) * up
output = hidden @ down_proj[expert]       # (704 → 2816)
output = output * routing_weight
```

Weight shapes (stored as 3D tensors):
- `gate_up_proj`: (128, 1408, 2816) — note: (num_experts, 2*moe_intermediate, hidden)
- `down_proj`: (128, 2816, 704) — note: (num_experts, hidden, moe_intermediate)

Activation: same `gelu_pytorch_tanh` as the dense MLP.

No bias on any expert weight.

### Expert aggregation

Weighted sum of all 8 active expert outputs per token, using the renormalized
router weights with per-expert scaling applied.


## Logit Softcapping

Applied to the final logits before loss/sampling:

```
logits = lm_head(hidden_states)
logits = tanh(logits / 30.0) * 30.0
```

This bounds logits to the range (-30, 30).


## Special Tokens

| Token             | ID     |
|-------------------|--------|
| `<pad>`           | 0      |
| `<eos>`           | 1      |
| `<bos>`           | 2      |
| `<boi>` (image)   | 255999 |
| `<boa>` (audio)   | 256000 |
| `<image_token>`   | 258880 |
| `<audio_token>`   | 258881 |
| `<eoi>` (image)   | 258882 |
| `<eoa>` (audio)   | 258883 |


## Weight Names (HuggingFace safetensors format)

Per layer (N = 0..29):

### Attention (sliding layers: N not in {5,11,17,23,29})
```
model.layers.N.self_attn.q_proj.weight    (4096, 2816)
model.layers.N.self_attn.k_proj.weight    (2048, 2816)
model.layers.N.self_attn.v_proj.weight    (2048, 2816)
model.layers.N.self_attn.o_proj.weight    (2816, 4096)
model.layers.N.self_attn.q_norm.weight    (256,)
model.layers.N.self_attn.k_norm.weight    (256,)
```

### Attention (full layers: N in {5,11,17,23,29})
```
model.layers.N.self_attn.q_proj.weight    (8192, 2816)
model.layers.N.self_attn.k_proj.weight    (1024, 2816)
                                          (no v_proj — K=V)
model.layers.N.self_attn.o_proj.weight    (2816, 8192)
model.layers.N.self_attn.q_norm.weight    (512,)
model.layers.N.self_attn.k_norm.weight    (512,)
```

### Norms (all layers)
```
model.layers.N.input_layernorm.weight                  (2816,)
model.layers.N.post_attention_layernorm.weight          (2816,)
model.layers.N.pre_feedforward_layernorm.weight         (2816,)
model.layers.N.post_feedforward_layernorm.weight        (2816,)
model.layers.N.post_feedforward_layernorm_1.weight      (2816,)
model.layers.N.pre_feedforward_layernorm_2.weight       (2816,)
model.layers.N.post_feedforward_layernorm_2.weight      (2816,)
model.layers.N.layer_scalar                             (1,)
```

### Dense MLP (all layers)
```
model.layers.N.mlp.gate_proj.weight       (2112, 2816)
model.layers.N.mlp.up_proj.weight         (2112, 2816)
model.layers.N.mlp.down_proj.weight       (2816, 2112)
```

### MoE (all layers)
```
model.layers.N.router.proj.weight         (128, 2816)
model.layers.N.router.scale               (2816,)
model.layers.N.router.per_expert_scale    (128,)
model.layers.N.experts.gate_up_proj       (128, 1408, 2816)
model.layers.N.experts.down_proj          (128, 2816, 704)
```

### Global
```
model.embed_tokens.weight                 (262144, 2816)
model.norm.weight                         (2816,)
lm_head.weight                            tied to model.embed_tokens.weight
```


## Features NOT used by 26B-A4B

These exist in the Gemma4 architecture but are disabled for this specific model:

- `hidden_size_per_layer_input = 0` — per-layer token embeddings are disabled
- `num_kv_shared_layers = 0` — no KV cache sharing across layers
- `use_double_wide_mlp = False` — no double-width MLP
- `attention_bias = False` — no bias on any Q/K/V/O projection
- Vision encoder and audio encoder exist in the multimodal wrapper but are not needed for text-only inference


## Tokenizer

Uses the Gemma 4 tokenizer (SentencePiece-based, 262144 vocab). This is the same tokenizer
family as Gemma 2/3. The tokenizer is NOT yet implemented in this codebase and will need to
be investigated separately.
