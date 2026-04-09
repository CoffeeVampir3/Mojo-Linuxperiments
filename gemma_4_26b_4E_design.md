# Gemma 4 26B-A4B — Full Implementation Specification

Model: `google/gemma-4-26B-A4B`
Total parameters: ~25.2B. Active per token: ~3.8B.
All weights: BF16. Compute dtype: BF16 (with FP32 for norms and softmax).

This document specifies the text decoder only. The vision/audio encoders
are separate modules not covered here.


---


## 1. Notation and Conventions

Throughout this document:
- `B` = batch size
- `S` = sequence length
- Shapes are written as `(dim0, dim1, ...)`, outermost first
- `@` means matrix multiply
- `*` means element-wise multiply
- "Linear(in, out)" means `output = input @ W^T` where W has shape `(out, in)`.
  This is standard PyTorch convention: the weight matrix is stored as (out_features, in_features).
- All weight shapes given are the stored/checkpoint shapes.


---


## 2. Constants

```
VOCAB_SIZE              = 262144
HIDDEN                  = 2816
NUM_LAYERS              = 30
NUM_HEADS               = 16
HEAD_DIM_SLIDING        = 256
HEAD_DIM_GLOBAL         = 512
NUM_KV_HEADS_SLIDING    = 8
NUM_KV_HEADS_GLOBAL     = 2
GQA_RATIO_SLIDING       = 2        (= NUM_HEADS / NUM_KV_HEADS_SLIDING)
GQA_RATIO_GLOBAL        = 8        (= NUM_HEADS / NUM_KV_HEADS_GLOBAL)
INTERMEDIATE            = 2112
MOE_INTERMEDIATE        = 704
NUM_EXPERTS             = 128
TOP_K_EXPERTS           = 8
SLIDING_WINDOW          = 1024
MAX_POSITION            = 262144
RMS_EPS                 = 1e-6
LOGIT_SOFTCAP           = 30.0
EMBED_SCALE             = sqrt(2816) ≈ 53.066
```


---


## 3. Layer Schedule

30 layers in a 5:1 sliding-to-full pattern:

```
Layer  0: sliding      Layer 10: sliding      Layer 20: sliding
Layer  1: sliding      Layer 11: full         Layer 21: sliding
Layer  2: sliding      Layer 12: sliding      Layer 22: sliding
Layer  3: sliding      Layer 13: sliding      Layer 23: full
Layer  4: sliding      Layer 14: sliding      Layer 24: sliding
Layer  5: full         Layer 15: sliding      Layer 25: sliding
Layer  6: sliding      Layer 16: sliding      Layer 26: sliding
Layer  7: sliding      Layer 17: full         Layer 27: sliding
Layer  8: sliding      Layer 18: sliding      Layer 28: sliding
Layer  9: sliding      Layer 19: sliding      Layer 29: full
```

Full attention layers: indices {5, 11, 17, 23, 29}.
The rule: `layer_type = "full" if (i+1) % 6 == 0 else "sliding"`.


---


## 4. Full Forward Pass (Inference)

Given `token_ids` of shape `(B, S)`:

```
Step 1:  hidden = embed(token_ids) * EMBED_SCALE            → (B, S, 2816)
Step 2:  Precompute RoPE cos/sin for sliding and full (see §8)
Step 3:  Precompute causal masks for sliding and full (see §9)
Step 4:  For layer_idx in 0..29:
             hidden = decoder_layer(hidden, layer_idx)       → (B, S, 2816)
Step 5:  hidden = rms_norm(hidden, final_norm_weight)        → (B, S, 2816)
Step 6:  logits = hidden @ embed_weight^T                    → (B, S, 262144)
Step 7:  logits = tanh(logits / 30.0) * 30.0                → (B, S, 262144)
```

Notes on Step 1: The embedding scale `sqrt(2816)` is cast to BF16 before
multiplication, which rounds ~53.066 to ~53.5. This is intentional and
required for numerical reproducibility with the reference implementation.

Notes on Step 6: The lm_head weight is **tied** to the embedding weight.
There is no separate lm_head weight tensor. The operation is
`hidden @ embed_tokens.weight^T`.


---


## 5. Decoder Layer

Each layer receives `hidden` of shape `(B, S, 2816)` and produces the same shape.

```
def decoder_layer(x, layer_idx):

    # ---- Attention block ----
    residual = x
    x = rms_norm(x, input_layernorm.weight)
    x = attention(x, layer_idx)                          # see §6 or §7
    x = rms_norm(x, post_attention_layernorm.weight)
    x = residual + x

    # ---- Feedforward block ----
    residual = x
    x_normed = rms_norm(x, pre_feedforward_layernorm.weight)

    # Dense MLP path
    dense_out = dense_mlp(x_normed)                      # see §10

    # MoE path (operates on the RESIDUAL, not on x_normed)
    x_flat = residual.reshape(B*S, 2816)
    _, weights, indices = router(x_flat)                  # see §11
    expert_input = rms_norm(x_flat, pre_feedforward_layernorm_2.weight)
    moe_out = experts(expert_input, indices, weights)     # see §12
    moe_out = moe_out.reshape(B, S, 2816)

    # Combine
    dense_normed = rms_norm(dense_out, post_feedforward_layernorm_1.weight)
    moe_normed = rms_norm(moe_out, post_feedforward_layernorm_2.weight)
    combined = dense_normed + moe_normed

    combined = rms_norm(combined, post_feedforward_layernorm.weight)
    x = residual + combined

    x = x * layer_scalar                                 # scalar (1,), loaded from checkpoint

    return x
```

**Critical: `layer_scalar` is multiplicative damping on the entire residual stream.**
It scales `x` AFTER the residual addition — both the carried-forward residual
and the new FFN contribution are scaled together. This is NOT equivalent to
`x = residual + combined * layer_scalar` (which would only scale the FFN
output). The distinction matters because the scalar (~0.1 at early layers,
~0.8 at later layers) controls how quickly the residual stream grows across
layers. Without whole-state scaling, hidden state RMS grows ~3x too fast
and logits saturate the softcap.

Critical detail: The dense MLP and MoE experts receive the **same pre-MLP**
hidden states (the residual), but through **different norms**:
- Dense MLP receives: `rms_norm(residual, pre_feedforward_layernorm.weight)`
- MoE experts receive: `rms_norm(residual, pre_feedforward_layernorm_2.weight)`
- Router receives: `residual` (it applies its own internal norm)


---


## 6. Sliding Attention (layers not in {5, 11, 17, 23, 29})

Input: `x` of shape `(B, S, 2816)`, already layer-normed.

### 6.1 Projections

```
Q = x @ q_proj.weight^T                    → (B, S, 4096)
K = x @ k_proj.weight^T                    → (B, S, 2048)
V = x @ v_proj.weight^T                    → (B, S, 2048)
```

Reshape to per-head form:
```
Q = Q.reshape(B, S, 16, 256)
K = K.reshape(B, S,  8, 256)
V = V.reshape(B, S,  8, 256)
```

### 6.2 Per-head norms

Applied independently to each head (norm over the last dim, which is head_dim):
```
Q = rms_norm(Q, q_norm.weight)             per-head, weight shape (256,)
K = rms_norm(K, k_norm.weight)             per-head, weight shape (256,)
V = rms_norm_no_scale(V)                   per-head, no learnable weight
```

### 6.3 RoPE

Using sliding RoPE (theta=10000, full 256-dim rotation):
```
cos, sin = sliding_rope_cache              shape (1, S, 256) — see §8
Q = rope(Q, cos, sin)                      (B, S, 16, 256) — broadcast over heads
K = rope(K, cos, sin)                      (B, S,  8, 256)
```

V does NOT get RoPE.

### 6.4 Transpose to attention layout

```
Q = Q.transpose(1, 2)                      → (B, 16, S, 256)
K = K.transpose(1, 2)                      → (B,  8, S, 256)
V = V.transpose(1, 2)                      → (B,  8, S, 256)
```

### 6.5 KV cache update

```
K, V = kv_cache.update(K, V, layer_idx)    → K, V may now have S_kv > S
```

### 6.6 GQA expansion

Repeat each KV head to serve its group of Q heads:
```
K = repeat_kv(K, n_rep=2)                  (B, 8, S_kv, 256) → (B, 16, S_kv, 256)
V = repeat_kv(V, n_rep=2)                  (B, 8, S_kv, 256) → (B, 16, S_kv, 256)
```

`repeat_kv` inserts a repeat dimension and reshapes:
```
def repeat_kv(x, n_rep):
    # x: (B, H_kv, S, D)
    if n_rep == 1: return x
    B, H, S, D = x.shape
    x = x.unsqueeze(2).expand(B, H, n_rep, S, D)
    return x.reshape(B, H * n_rep, S, D)
```

### 6.7 Attention scores

```
scores = (Q @ K^T) * 1.0                   (B, 16, S, S_kv) — scaling is 1.0
scores = scores + sliding_mask              apply causal + sliding window mask
probs  = softmax(scores, dim=-1)            MUST upcast to FP32 for softmax, cast back after
output = probs @ V                          (B, 16, S, 256)
```

### 6.8 Output projection

```
output = output.transpose(1, 2)            → (B, S, 16, 256)
output = output.reshape(B, S, 4096)
output = output @ o_proj.weight^T           → (B, S, 2816)
```


---


## 7. Full (Global) Attention (layers in {5, 11, 17, 23, 29})

Same structure as §6, but with different dimensions and K=V sharing.

### 7.1 Projections

```
Q = x @ q_proj.weight^T                    → (B, S, 8192)
K = x @ k_proj.weight^T                    → (B, S, 1024)
V = K                                       K and V share the same projection output
```

There is **no v_proj weight** for these layers. The raw K projection output
is used as both the K and V input, before their respective norms diverge them.

Reshape:
```
Q = Q.reshape(B, S, 16, 512)
K = K.reshape(B, S,  2, 512)
V = V.reshape(B, S,  2, 512)               same tensor as K at this point
```

### 7.2 Per-head norms

```
Q = rms_norm(Q, q_norm.weight)             weight shape (512,)
K = rms_norm(K, k_norm.weight)             weight shape (512,)
V = rms_norm_no_scale(V)                   no learnable weight
```

After norms, K and V are now different tensors even though they started from
the same projection. K has a learnable scale; V does not.

### 7.3 RoPE (partial rotation)

Using full-attention RoPE (theta=1000000, 128 of 512 dims rotated):
```
cos, sin = full_rope_cache                 shape (1, S, 512) — see §8
Q = rope(Q, cos, sin)
K = rope(K, cos, sin)
```

The cos/sin vectors are 512-dim. The first 128 dims encode real frequencies.
The remaining 384 dims have cos=1, sin=0, so those dims pass through unchanged.
See §8 for how this is constructed.

V does NOT get RoPE.

### 7.4 Transpose, cache, GQA, attention

Same as §6.4–6.8 but with:
- GQA ratio = 8 (each of 2 KV heads serves 8 Q heads)
- Head dim = 512
- No sliding window (full causal mask)
- Output reshape: `(B, S, 16, 512)` → `(B, S, 8192)`
- o_proj: `(B, S, 8192)` → `(B, S, 2816)`


---


## 8. RoPE (Rotary Position Embedding)

### 8.1 Rotation formula

Uses the "rotate_half" convention (same as Llama):
```
def rotate_half(x):
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    return concat(-x2, x1, dim=-1)

def rope(x, cos, sin):
    # cos, sin: (1, S, D) or (B, S, D) — broadcast over heads via unsqueeze
    cos = cos.unsqueeze(2)                  → (B, S, 1, D)
    sin = sin.unsqueeze(2)                  → (B, S, 1, D)
    return (x * cos) + (rotate_half(x) * sin)
```

Note: RoPE is applied **before** the transpose to attention layout. The tensor
shapes at application time are `(B, S, H, D)`, so `unsqueeze_dim=2` broadcasts
cos/sin over the head dimension.

### 8.2 Sliding RoPE (for sliding attention layers)

Parameters: `theta = 10000`, `head_dim = 256`

```
dim = 256
base = 10000.0
i = arange(0, dim, 2)                      → [0, 2, 4, ..., 254]  — 128 values
inv_freq = 1.0 / (base ^ (i / dim))        → 128 inverse frequencies

# For positions 0..S-1:
freqs = outer(positions, inv_freq)          → (S, 128)
emb = concat(freqs, freqs, dim=-1)          → (S, 256)
cos = cos(emb)                              → (S, 256)
sin = sin(emb)                              → (S, 256)
```

All 256 dims participate in rotation. The `concat(freqs, freqs)` duplication
is required by the `rotate_half` formula.

### 8.3 Full-attention RoPE (for global attention layers)

Parameters: `theta = 1000000`, `global_head_dim = 512`, `partial_rotary_factor = 0.25`

```
head_dim = 512
rope_proportion = 0.25
rope_angles = floor(rope_proportion * head_dim / 2) = 64
nope_angles = head_dim / 2 - rope_angles = 192

# Compute frequencies only for the rotary portion
i = arange(0, 2 * rope_angles, 2)          → [0, 2, 4, ..., 126]  — 64 values
inv_freq_rot = 1.0 / (1000000 ^ (i / head_dim))

# IMPORTANT: denominator uses FULL head_dim (512), not rotary dim (128).
# This spreads the frequencies across a wider range.

# Zero-pad to full half-dim
inv_freq = concat(inv_freq_rot, zeros(192)) → 256 values total

# For positions 0..S-1:
freqs = outer(positions, inv_freq)          → (S, 256)
emb = concat(freqs, freqs, dim=-1)          → (S, 512)
cos = cos(emb)                              → (S, 512)
sin = sin(emb)                              → (S, 512)
```

The zero-padded frequencies produce `cos(0)=1` and `sin(0)=0`, which means
the `rotate_half` formula simplifies to identity for those dimensions:
`x * 1 + rotate_half(x) * 0 = x`. So dims 128..511 pass through unchanged.

### 8.4 Precomputation

Both RoPE tables are computed once at model initialization and reused across
all layers of the corresponding type. They only depend on position indices,
not on layer-specific parameters.


---


## 9. Causal Masks

### 9.1 Full causal mask

Standard lower-triangular mask. Position `q` can attend to position `k` iff `k <= q`.
Masked positions get `-inf` added to attention scores (before softmax).

### 9.2 Sliding window causal mask

Position `q` can attend to position `k` iff:
- `k <= q` (causal: can't look forward)
- `q - k < 1024` (sliding window: can't look more than 1024 tokens back)

Equivalently: `max(0, q - 1023) <= k <= q`.

### 9.3 With KV cache

During autoregressive generation, the query length is typically 1 (single new token),
and the key/value length grows with the cache. The mask logic remains the same but
operates on the current query position vs. all cached key positions.

For sliding attention with KV cache, only the most recent 1024 KV entries need
to be kept. Older entries can be evicted.

For full attention, the entire KV history must be retained.


---


## 10. Dense MLP

Input: `x` of shape `(B, S, 2816)`, already normed.

```
gate = x @ gate_proj.weight^T              → (B, S, 2112)
up   = x @ up_proj.weight^T                → (B, S, 2112)
out  = gelu_tanh(gate) * up                 → (B, S, 2112)
out  = out @ down_proj.weight^T             → (B, S, 2816)
```

No bias on any projection.

### GELU with tanh approximation

```
def gelu_tanh(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
```

Where `sqrt(2/pi) ≈ 0.7978845608`.


---


## 11. Router

Input: `x_flat` of shape `(T, 2816)` where `T = B * S`. This is the **pre-MLP
residual**, not the normed version.

```
def router(x):
    # Step 1: Normalize (no learnable scale)
    x = rms_norm_no_scale(x)                → (T, 2816)

    # Step 2: Apply learnable per-dim scale with constant factor
    x = x * scale * (1.0 / sqrt(2816))      → (T, 2816)
    # scale is a (2816,) learned parameter
    # 1/sqrt(2816) ≈ 0.01885 is a constant

    # Step 3: Project to expert scores
    scores = x @ proj.weight^T              → (T, 128)

    # Step 4: Softmax over ALL experts
    probs = softmax(scores, dim=-1)         → (T, 128)

    # Step 5: Select top-8
    top_k_weights, top_k_indices = topk(probs, k=8, dim=-1)
    # top_k_weights: (T, 8)
    # top_k_indices: (T, 8)

    # Step 6: Renormalize top-k to sum to 1
    top_k_weights = top_k_weights / sum(top_k_weights, dim=-1, keepdim=True)

    # Step 7: Apply per-expert learned scale
    top_k_weights = top_k_weights * per_expert_scale[top_k_indices]
    # per_expert_scale is a (128,) learned parameter

    return top_k_weights, top_k_indices
```

Weights:
```
router.norm          — RMSNorm(2816), no learnable scale (just the operation)
router.proj.weight   — (128, 2816)
router.scale         — (2816,)
router.per_expert_scale — (128,)
```


---


## 12. Sparse Experts (MoE)

Input: `x` of shape `(T, 2816)`, already normed by `pre_feedforward_layernorm_2`.
`top_k_indices` of shape `(T, 8)` and `top_k_weights` of shape `(T, 8)` from the router.

### 12.1 Weight shapes

```
experts.gate_up_proj    (128, 1408, 2816)   — (num_experts, 2*moe_intermediate, hidden)
experts.down_proj       (128, 2816,  704)   — (num_experts, hidden, moe_intermediate)
```

No biases.

### 12.2 Per-expert computation

For each expert `e` that has at least one token routed to it:

```
# Gather: find which tokens are routed to expert e
token_indices = tokens where top_k_indices contains e
routing_weights_for_e = corresponding weights from top_k_weights

# Expert MLP
current = x[token_indices]                  → (T_e, 2816)
gate_up = current @ gate_up_proj[e]^T       → (T_e, 1408)
gate = gate_up[..., :704]                   → (T_e, 704)   — first half
up   = gate_up[..., 704:]                   → (T_e, 704)   — second half
hidden = gelu_tanh(gate) * up               → (T_e, 704)
out = hidden @ down_proj[e]^T               → (T_e, 2816)

# Weight by routing score
out = out * routing_weights_for_e.unsqueeze(-1)

# Scatter-add back
output[token_indices] += out
```

The gate/up split is a simple chunk (first half / second half), NOT interleaved.

### 12.3 Implementation notes

The `gate_up_proj[e]` has shape `(1408, 2816)`. Using `F.linear(input, weight)`
computes `input @ weight^T`, i.e., `(T_e, 2816) @ (2816, 1408)` → `(T_e, 1408)`.

A token routed to multiple experts gets contributions from each, accumulated via
scatter-add (index_add). The output tensor is initialized to zeros.

The activation function is the same `gelu_tanh` as the dense MLP (§10).


---


## 13. RMS Normalization

Two variants used throughout the model.

### 13.1 RMSNorm with learnable scale

Used for: all layer norms, q_norm, k_norm.

```
def rms_norm(x, weight, eps=1e-6):
    x_f32 = x.to(float32)
    mean_sq = mean(x_f32 ^ 2, dim=-1, keepdim=True) + eps
    normed = x_f32 * pow(mean_sq, -0.5)
    normed = normed * weight.to(float32)
    return normed.to(x.dtype)
```

Key details:
- Input is cast to FP32 before computation.
- Uses `pow(x, -0.5)` NOT `rsqrt(x)`. This is for JAX/Torch numerical parity.
- Weight is also cast to FP32 for the multiply.
- Result is cast back to input dtype (BF16).

### 13.2 RMSNorm without scale

Used for: v_norm, router.norm.

```
def rms_norm_no_scale(x, eps=1e-6):
    x_f32 = x.to(float32)
    mean_sq = mean(x_f32 ^ 2, dim=-1, keepdim=True) + eps
    normed = x_f32 * pow(mean_sq, -0.5)
    return normed.to(x.dtype)
```

Same as above but with no learnable weight parameter. These norms have no
stored weights in the checkpoint.


---


## 14. Embedding

```
class ScaledEmbedding:
    weight: (262144, 2816)       — BF16

    def forward(token_ids):
        embeds = lookup(weight, token_ids)       → (B, S, 2816)
        scale = to_bf16(sqrt(2816))              ≈ 53.5 after BF16 cast
        return embeds * scale
```

The scale is stored as a buffer initialized to `float(sqrt(2816))`, then cast
to BF16 for the multiply. The BF16 rounding (53.066 → ~53.5) is a known
Gemma behavior and must be matched for numerical reproducibility.


---


## 15. Logit Softcapping

Applied after the lm_head projection, before loss or sampling:

```
logits = hidden @ embed_weight^T            → (B, S, 262144)
logits = tanh(logits / 30.0) * 30.0
```

This smoothly clamps logits to the range `(-30, +30)`. It prevents extreme
logit values while remaining differentiable.


---


## 16. Softmax Precision

Attention softmax **must** be computed in FP32 even when the rest of the model
runs in BF16:

```
scores_f32 = softmax(scores.to(float32), dim=-1)
probs = scores_f32.to(bf16)
```

This is explicitly required by the reference implementation.


---


## 17. Weight Inventory

### 17.1 Global weights

| Weight | Shape | Count |
|--------|-------|-------|
| `model.embed_tokens.weight` | (262144, 2816) | 738,197,504 |
| `model.norm.weight` | (2816,) | 2,816 |
| `lm_head.weight` | tied to embed_tokens | 0 (shared) |

Embedding scale buffer: `model.embed_tokens.embed_scale` = scalar


### 17.2 Per-layer weights — Sliding attention (25 layers)

| Weight | Shape | Per-layer |
|--------|-------|-----------|
| `self_attn.q_proj.weight` | (4096, 2816) | 11,534,336 |
| `self_attn.k_proj.weight` | (2048, 2816) | 5,767,168 |
| `self_attn.v_proj.weight` | (2048, 2816) | 5,767,168 |
| `self_attn.o_proj.weight` | (2816, 4096) | 11,534,336 |
| `self_attn.q_norm.weight` | (256,) | 256 |
| `self_attn.k_norm.weight` | (256,) | 256 |

v_norm has **no weight** (scale-free normalization).


### 17.3 Per-layer weights — Full attention (5 layers)

| Weight | Shape | Per-layer |
|--------|-------|-----------|
| `self_attn.q_proj.weight` | (8192, 2816) | 23,068,672 |
| `self_attn.k_proj.weight` | (1024, 2816) | 2,883,584 |
| (no v_proj) | — | — |
| `self_attn.o_proj.weight` | (2816, 8192) | 23,068,672 |
| `self_attn.q_norm.weight` | (512,) | 512 |
| `self_attn.k_norm.weight` | (512,) | 512 |


### 17.4 Per-layer weights — Dense MLP (all 30 layers)

| Weight | Shape | Per-layer |
|--------|-------|-----------|
| `mlp.gate_proj.weight` | (2112, 2816) | 5,947,392 |
| `mlp.up_proj.weight` | (2112, 2816) | 5,947,392 |
| `mlp.down_proj.weight` | (2816, 2112) | 5,947,392 |


### 17.5 Per-layer weights — MoE (all 30 layers)

| Weight | Shape | Per-layer |
|--------|-------|-----------|
| `router.proj.weight` | (128, 2816) | 360,448 |
| `router.scale` | (2816,) | 2,816 |
| `router.per_expert_scale` | (128,) | 128 |
| `experts.gate_up_proj` | (128, 1408, 2816) | 506,986,496 |
| `experts.down_proj` | (128, 2816, 704) | 253,493,248 |


### 17.6 Per-layer weights — Norms and scalar (all 30 layers)

| Weight | Shape | Per-layer |
|--------|-------|-----------|
| `input_layernorm.weight` | (2816,) | 2,816 |
| `post_attention_layernorm.weight` | (2816,) | 2,816 |
| `pre_feedforward_layernorm.weight` | (2816,) | 2,816 |
| `post_feedforward_layernorm.weight` | (2816,) | 2,816 |
| `post_feedforward_layernorm_1.weight` | (2816,) | 2,816 |
| `pre_feedforward_layernorm_2.weight` | (2816,) | 2,816 |
| `post_feedforward_layernorm_2.weight` | (2816,) | 2,816 |
| `layer_scalar` | (1,) | 1 |


### 17.7 Safetensors key names

All weights are prefixed with `model.layers.{N}.` for per-layer weights.
Full key examples for layer 0:

```
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight          ← absent for layers {5,11,17,23,29}
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.pre_feedforward_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.post_feedforward_layernorm_1.weight
model.layers.0.router.proj.weight
model.layers.0.router.scale
model.layers.0.router.per_expert_scale
model.layers.0.pre_feedforward_layernorm_2.weight
model.layers.0.experts.gate_up_proj
model.layers.0.experts.down_proj
model.layers.0.post_feedforward_layernorm_2.weight
model.layers.0.post_feedforward_layernorm.weight
model.layers.0.layer_scalar
```


---


## 18. KV Cache

### 18.1 Sliding attention layers

Each sliding attention layer maintains its own KV cache. Since the sliding
window is 1024, only the most recent 1024 KV entries are needed. Older entries
can be evicted.

Cache entry shapes per layer:
```
K: (B, 8, S_cached, 256)
V: (B, 8, S_cached, 256)
```

### 18.2 Full attention layers

Each full attention layer maintains its own KV cache with the full history.
No entries are evicted.

Cache entry shapes per layer:
```
K: (B, 2, S_cached, 512)
V: (B, 2, S_cached, 512)
```

### 18.3 Cache stores post-processed KV

The cache stores K and V **after** their respective norms and (for K) RoPE.
This means:
- Cached K = rms_norm(k_proj(x), k_norm.weight) then RoPE applied
- Cached V = rms_norm_no_scale(v_proj(x)) — or rms_norm_no_scale(k_proj(x)) for K=V layers

This avoids recomputing norms and RoPE on cached entries.


---


## 19. Features Disabled in 26B-A4B

The Gemma 4 architecture supports features that are NOT used in this model:

| Feature | Config value | Effect |
|---------|-------------|--------|
| Per-layer input embeddings | `hidden_size_per_layer_input = 0` | Disabled. No `embed_tokens_per_layer` or related weights exist. |
| KV sharing across layers | `num_kv_shared_layers = 0` | Disabled. Each layer has independent KV projections. |
| Double-wide MLP | `use_double_wide_mlp = False` | Disabled. MLP intermediate size is always 2112. |
| Attention bias | `attention_bias = False` | No bias on Q/K/V/O projections. |
| Vision/audio encoders | N/A | Present in the multimodal wrapper but not needed for text-only. |

These can be safely ignored for this specific model.


---


## 20. Special Tokens

| Token | ID | Usage |
|-------|----|-------|
| `<pad>` | 0 | Padding |
| `<eos>` | 1 | End of sequence / generation stop |
| `<bos>` | 2 | Beginning of sequence |

For text-only inference, prepend `<bos>` (id=2) to the input. Stop generation
on `<eos>` (id=1).


---


## 21. Numerical Checklist

For an implementation to match the reference:

1. RMSNorm: compute in FP32, use `pow(x, -0.5)` not `rsqrt(x)`
2. Attention softmax: compute in FP32, cast result back to BF16
3. Embedding scale: cast `sqrt(2816)` to BF16 before multiplying embeddings
4. Attention scaling: use `1.0`, not `1/sqrt(head_dim)`
5. RoPE full-attention freq denominator: use full `head_dim=512`, not rotary dim 128
6. Router constant: `1/sqrt(hidden_size)` = `1/sqrt(2816)`, computed at init
7. Logit softcapping: `tanh(logits / 30) * 30`, applied after lm_head
8. Expert gate/up split: first half / second half (chunk), not interleaved
9. No bias on any linear projection in the text model
10. Tied embedding weights: lm_head reuses embed_tokens.weight
11. **Layer scalar is whole-state multiplicative**: `x = (residual + ffn_out) * layer_scalar`,
    NOT `x = residual + ffn_out * layer_scalar`. The scalar damps the entire residual
    stream, not just the current layer's contribution. Getting this wrong causes hidden
    state RMS to grow ~3x too fast (reference: 1→3 over 30 layers; incorrect: 1→10).
