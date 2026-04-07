# DeepSeek-V2/V3 Architecture for Inference

Reference implementation target: **DeepSeek-V2-Lite** (15.7B total, 2.4B active).
This document covers everything needed to implement forward inference, and notes
where V3 (671B) diverges. All shapes and behaviors are verified against the
actual checkpoint files and reference `modeling_deepseek.py`.

## Model Dimensions (V2-Lite)

| Parameter | Symbol | V2-Lite | V3 |
|---|---|---|---|
| Layers | L | 27 | 61 |
| Hidden dim | d | 2048 | 7168 |
| Attention heads | n_h | 16 | 128 |
| Per-head key dim (content) | d_h | 128 | 128 |
| Per-head value dim | v_h | 128 | 128 |
| KV compression dim | d_c | 512 | 512 |
| Query compression dim | d'_c | N/A (null) | 1536 |
| Decoupled RoPE dim (per-head) | d_R_h | 64 | 64 |
| Shared experts | N_s | 2 (fused as 1 MLP) | 1 |
| Routed experts | N_r | 64 | 256 |
| Activated routed experts | K_r | 6 | 8 |
| Expert intermediate dim | - | 1408 | 2048 |
| Shared expert intermediate dim | - | 2816 (= 1408 * 2) | 2048 (= 2048 * 1) |
| Dense FFN intermediate dim | - | 10944 | 18432 |
| Dense FFN layers | - | Layer 0 only | Layers 0-2 |
| Vocab size | V | 102400 | 129280 |
| Max context | - | 163840 (YaRN from 4096) | 163840 (YaRN from 4096) |
| RoPE theta | - | 10000 | 10000 |
| RMSNorm eps | - | 1e-6 | 1e-6 |
| Tie embeddings | - | false | false |

## High-Level Forward Pass

```
for each token position t:
    h_t = embed_tokens(token_t)        # [V, d] lookup

    for each layer:
        # Attention block
        u_t = RMSNorm(h_t)             # input_layernorm
        u_t = MLA(u_t)
        h_t = h_t + u_t

        # FFN block
        u_t = RMSNorm(h_t)             # post_attention_layernorm
        if layer is dense:
            u_t = DenseFFN(u_t)         # SwiGLU MLP
        else:
            u_t = DeepSeekMoE(u_t)      # shared + routed experts
        h_t = h_t + u_t

    h_t = RMSNorm(h_t)                 # model.norm (final)
    logits = lm_head(h_t)              # separate [V, d] weight, NOT tied
```

## Multi-head Latent Attention (MLA)

MLA replaces standard MHA. Instead of caching full K and V tensors per head,
it caches a single compressed latent vector c_KV plus a small decoupled RoPE key k_R.

### Checkpoint Weight Names and Shapes

Per layer, the attention weights in the checkpoint are:

| Checkpoint name | Shape (V2-Lite) | Paper equivalent |
|---|---|---|
| `self_attn.q_proj` | [3072, 2048] | Fused W_UQ ∥ W_QR (no query compression) |
| `self_attn.kv_a_proj_with_mqa` | [576, 2048] | Fused W_DKV ∥ W_KR |
| `self_attn.kv_a_layernorm` | [512] | RMSNorm gamma for c_KV |
| `self_attn.kv_b_proj` | [4096, 512] | Fused W_UK ∥ W_UV |
| `self_attn.o_proj` | [2048, 2048] | W_O |

Key shape derivations:
- `q_proj`: 3072 = n_h * (d_h + d_R_h) = 16 * (128 + 64) = 16 * 192
- `kv_a_proj_with_mqa`: 576 = d_c + d_R_h = 512 + 64
- `kv_b_proj`: 4096 = n_h * (d_h + v_h) = 16 * (128 + 128) = 16 * 256
- `o_proj`: 2048 = n_h * v_h = 16 * 128

**V3 adds:** `q_a_proj` [1536, 7168], `q_a_layernorm` [1536], and `q_b_proj` [24576, 1536].
`q_b_proj` output = 128 * (128 + 64) = 24576. The `q_proj` single weight is replaced by this
down-up pair with a layernorm between.

### KV Cache Contents

Per token per layer, the cache stores:
- `c_KV_t`: d_c = 512 elements (the compressed KV latent, after RMSNorm)
- `k_R_t`: d_R_h = 64 elements (the decoupled RoPE key, after RoPE)

Total: **576 elements** per token per layer. Shared across all heads.
Compare standard MHA: 2 * n_h * d_h * L = 4096 per token per layer → **7x reduction**.

### Query Computation

**V2-Lite (no query compression, `q_lora_rank: null`):**
```
q_full = q_proj @ h_t                      # [3072] from [3072, 2048] @ [2048]
# reshape to [n_h, d_h + d_R_h] = [16, 192]
q_nope = q_full[:, :128]                    # per-head content query [16, 128]
q_pe   = q_full[:, 128:]                    # per-head RoPE query [16, 64]
q_pe   = RoPE(q_pe)                         # apply rotary position embedding
```

**V3 (with query compression, `q_lora_rank: 1536`):**
```
c_Q = q_a_proj @ h_t                       # [1536] down-project
c_Q = RMSNorm(c_Q)                         # q_a_layernorm
q_full = q_b_proj @ c_Q                    # [24576] up-project
# reshape to [128, 192], split same as above
```

### KV Computation (same structure for V2-Lite and V3)

```
kv_a = kv_a_proj_with_mqa @ h_t            # [576] from [576, 2048] @ [2048]
c_KV = kv_a[:512]                           # compressed KV latent
k_R  = kv_a[512:]                           # decoupled RoPE key [64]

c_KV = RMSNorm(c_KV)                       # kv_a_layernorm (THEN cache c_KV)
k_R  = RoPE(k_R)                           # apply rotary position embedding (THEN cache k_R)
```

Both c_KV (post-norm) and k_R (post-RoPE) are stored in the KV cache.

The up-projection `kv_b_proj` is NOT applied during caching. It is deferred to
attention time (and can be absorbed into query/output projections).

### Naive Attention Computation (without absorption)

If we explicitly up-project from the cache:
```
kv_up = kv_b_proj @ c_KV_j                 # [4096] from [4096, 512] @ [512]
# reshape to [n_h, d_h + v_h] = [16, 256]
k_nope_j = kv_up[:, :128]                  # per-head content key [16, 128]
v_j      = kv_up[:, 128:]                  # per-head value [16, 128]

# Concatenate content and RoPE components
q_i = [q_nope_i ; q_pe_i]                  # [d_h + d_R_h] = [192]
k_i = [k_nope_j_i ; k_R_j]                # [d_h + d_R_h] = [192] (k_R shared across heads)

# Scaled dot-product attention
score = (q_i^T @ k_i) * softmax_scale
o_t_i = sum_j softmax(score_j) * v_j_i
```

### Softmax Scale (YaRN mscale correction)

The attention scale is NOT simply `1/sqrt(d_h + d_R_h)`. YaRN applies an mscale correction:

```
base_scale = (d_h + d_R_h)^(-0.5)          # 1/sqrt(192) ≈ 0.0722
mscale = 0.1 * mscale_all_dim * ln(yarn_factor) + 1.0

V2-Lite: mscale_all_dim = 0.707, factor = 40
  mscale = 0.1 * 0.707 * ln(40) + 1.0 ≈ 1.2608
  softmax_scale = base_scale * mscale^2 ≈ 0.1148

V3: mscale_all_dim = 1.0, factor = 40
  mscale = 0.1 * 1.0 * ln(40) + 1.0 ≈ 1.3689
  softmax_scale = base_scale * mscale^2 ≈ 0.1353
```

### Weight Absorption Optimization

The key inference optimization: the content portions of W_UK and W_UV (packed in
`kv_b_proj`) can be absorbed into the query and output projections respectively.

**Score computation (absorb W_UK into query):**
```
# Instead of: q_nope_i^T @ (W_UK[i] @ c_KV_j)
# Precompute: q_hat_i = W_UK[i]^T @ q_nope_i    (applied once per query)
# Then:       score_C = q_hat_i^T @ c_KV_j       (dot product with cached latent)
```

The content score becomes a d_c=512 dot product against the raw cached latent.

**Value aggregation (absorb W_UV into output projection):**
```
# Instead of: o_i = sum_j w_j * (W_UV[i] @ c_KV_j)
# Compute:    o_latent_i = sum_j w_j * c_KV_j    (aggregate latents directly)
# Then:       o_i = W_UV[i] @ o_latent_i          (project once after aggregation)
```

**RoPE score is separate** (position-dependent, cannot be absorbed):
```
score_R = q_pe_i^T @ k_R_j                 # d_R_h = 64 dot product
score_total = (score_C + score_R) * softmax_scale
```

### Practical Decode Attention (with absorption)

Per head, single-token decode:
1. Transform query: `q_hat = W_UK[i]^T @ q_nope_i` (d_c=512 output, once)
2. For each cached position j:
   - Content score: `q_hat^T @ c_KV_j` (512-dim dot product)
   - RoPE score: `q_pe_i^T @ k_R_j` (64-dim dot product)
   - Combined: `(content + rope) * softmax_scale`
3. Softmax over positions
4. Aggregate: `o_latent = sum_j w_j * c_KV_j` (weighted sum of 512-dim vectors)
5. Project: `o = W_UV[i] @ o_latent` then through W_O

The KV cache read per position is d_c + d_R_h = 576 elements, but only ONE set
shared across all heads (vs n_h sets for MHA).

### RoPE Implementation

Standard YaRN-extended RoPE applied to the decoupled components only.
- Base frequency: theta = 10000
- Dimension: d_R_h = 64 (32 frequency pairs)
- Frequencies: `inv_freq[i] = 1.0 / (theta^(2i/dim))` for i in 0..dim/2
- YaRN interpolates between original and scaled frequencies using a linear ramp

The reference implementation uses an **interleaved-to-half permutation** before
applying rotate_half style RoPE:
```
# Input layout: [r0, i0, r1, i1, ...]  (interleaved real/imaginary)
# Permute to:   [r0, r1, ..., i0, i1, ...] (half-split)
x = x.view(d // 2, 2).transpose(1, 0).reshape(d)
# Then apply: x * cos + rotate_half(x) * sin
```

## DeepSeekMoE (FFN)

### Structure

Each MoE layer has:
- **N_s shared experts** fused into a single larger SwiGLU MLP
- **N_r routed experts** (sparse, top-K_r selected per token)

Each expert (shared or routed) is a standard SwiGLU FFN:
```
Expert(x) = (SiLU(gate_proj @ x) * (up_proj @ x)) @ down_proj
```

**Shared expert fusion**: the N_s shared experts are implemented as ONE MLP with
intermediate_size = moe_intermediate_size * N_s. For V2-Lite: 1408 * 2 = 2816.
For V3: 2048 * 1 = 2048.

### Checkpoint Weight Names (MoE layers 1-26)

| Checkpoint name | Shape (V2-Lite) | Notes |
|---|---|---|
| `mlp.gate` | [64, 2048] | Router centroids (one row per routed expert) |
| `mlp.shared_experts.gate_proj` | [2816, 2048] | Shared (fused) SwiGLU gate |
| `mlp.shared_experts.up_proj` | [2816, 2048] | Shared (fused) SwiGLU up |
| `mlp.shared_experts.down_proj` | [2048, 2816] | Shared (fused) SwiGLU down |
| `mlp.experts.{i}.gate_proj` | [1408, 2048] | Routed expert i gate |
| `mlp.experts.{i}.up_proj` | [1408, 2048] | Routed expert i up |
| `mlp.experts.{i}.down_proj` | [2048, 1408] | Routed expert i down |

### Checkpoint Weight Names (Dense layer 0)

| Checkpoint name | Shape (V2-Lite) |
|---|---|
| `mlp.gate_proj` | [10944, 2048] |
| `mlp.up_proj` | [10944, 2048] |
| `mlp.down_proj` | [2048, 10944] |

### Routing

**V2-Lite** (`scoring_func: "softmax"`, `norm_topk_prob: false`, `topk_method: "greedy"`):
```
# Compute affinity scores
logits_i = x^T @ e_i                        # for each routed expert i
s = softmax(logits)                          # over all 64 experts

# Select top-K_r experts (greedy)
selected = top_k(s, K_r=6)

# Gate values: raw softmax scores (NOT renormalized, * routed_scaling_factor=1.0)
g_i = s_i    for i in selected

# Compute output
output = routed_sum + shared_expert(x)
routed_sum = sum(g_i * Expert_routed_i(x) for i in selected)
```

**V3** (`scoring_func: "sigmoid"`, `norm_topk_prob: true`, `topk_method: "noaux_tc"`):
```
s_i = sigmoid(x^T @ e_i)                    # independent per-expert scores

# Group-limited selection: n_group=8 groups of 32 experts each
# Select top topk_group=4 groups, then top-8 experts within those groups
# (noaux_tc adds a learnable bias for selection only, not for gate values)

# Normalize gates among selected experts to sum to 1
g_i = s_i / sum(s_j for j in selected)

# Compute output (same structure)
```

### MoE Output Assembly

```
# The MoE block returns (no residual -- residual is added by the layer):
output = routed_weighted_sum + shared_experts(input)
```

## RMSNorm

Standard RMSNorm without learnable bias, with learnable scale gamma:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```
eps = 1e-6. Applied pre-norm (before attention and before FFN) in each layer.

Additional RMSNorm applied inside MLA:
- `kv_a_layernorm` [d_c=512]: applied to c_KV after down-projection, before caching/up-projection
- V3 also has `q_a_layernorm` [d'_c=1536]: applied to c_Q between query down/up-projections

## Complete Weight Layout (V2-Lite Checkpoint)

### Global weights

| Checkpoint name | Shape | dtype |
|---|---|---|
| `model.embed_tokens.weight` | [102400, 2048] | BF16 |
| `lm_head.weight` | [102400, 2048] | BF16 |
| `model.norm.weight` | [2048] | BF16 |

### Per layer (all 27 layers) -- Attention

| Checkpoint name | Shape | dtype |
|---|---|---|
| `self_attn.q_proj.weight` | [3072, 2048] | BF16 |
| `self_attn.kv_a_proj_with_mqa.weight` | [576, 2048] | BF16 |
| `self_attn.kv_a_layernorm.weight` | [512] | BF16 |
| `self_attn.kv_b_proj.weight` | [4096, 512] | BF16 |
| `self_attn.o_proj.weight` | [2048, 2048] | BF16 |
| `input_layernorm.weight` | [2048] | BF16 |
| `post_attention_layernorm.weight` | [2048] | BF16 |

### Layer 0 only -- Dense FFN

| Checkpoint name | Shape | dtype |
|---|---|---|
| `mlp.gate_proj.weight` | [10944, 2048] | BF16 |
| `mlp.up_proj.weight` | [10944, 2048] | BF16 |
| `mlp.down_proj.weight` | [2048, 10944] | BF16 |

### Layers 1-26 -- MoE FFN

| Checkpoint name | Shape | Count | dtype |
|---|---|---|---|
| `mlp.gate.weight` | [64, 2048] | 1 | BF16 |
| `mlp.shared_experts.gate_proj.weight` | [2816, 2048] | 1 | BF16 |
| `mlp.shared_experts.up_proj.weight` | [2816, 2048] | 1 | BF16 |
| `mlp.shared_experts.down_proj.weight` | [2048, 2816] | 1 | BF16 |
| `mlp.experts.{0..63}.gate_proj.weight` | [1408, 2048] | 64 | BF16 |
| `mlp.experts.{0..63}.up_proj.weight` | [1408, 2048] | 64 | BF16 |
| `mlp.experts.{0..63}.down_proj.weight` | [2048, 1408] | 64 | BF16 |

Total checkpoint size: ~31.4 GB (BF16).

## NUMA / Expert Placement Strategy

With 4 NUMA nodes and 64 routed experts per layer, a natural mapping is
16 experts per node. The shared expert (fused MLP with intermediate=2816) is
small enough to replicate on all nodes (always active).

For decode, the router selects 6 of 64 experts. If those 6 land on 2-3 nodes,
only those nodes need to execute expert FFNs. The hidden state is sent to the
relevant nodes, each node runs its selected experts locally, and results are
summed back. No allreduce needed -- expert outputs are independent and the
summation can happen at the destination.

This is fundamentally different from dense TP where every weight matrix is
sharded and requires allreduce. MoE expert parallelism replaces allreduce
with point-to-point dispatch/combine, which scales much better.

## Differences Between V2-Lite and V3 (Inference)

| Aspect | V2-Lite | V3 |
|---|---|---|
| Query compression | None (`q_lora_rank: null`) | W_DQ + RMSNorm + W_UQ (`q_lora_rank: 1536`) |
| Scoring function | `softmax` | `sigmoid` |
| Gate normalization | None (`norm_topk_prob: false`, scale=1.0) | Normalize to sum=1 (`norm_topk_prob: true`) |
| Routing method | `greedy` (simple top-K) | `noaux_tc` (group-limited, `n_group=8`, `topk_group=4`) |
| Shared experts | 2 (fused as 1 MLP, intermediate=2816) | 1 (intermediate=2048) |
| Routed experts | 64 (top-6) | 256 (top-8) |
| Dense FFN layers | Layer 0 (intermediate=10944) | Layers 0-2 (intermediate=18432) |
| Dense FFN count | 1 | 3 |
| MTP module | None | 1 (`num_nextn_predict_layers: 1`) |
| Vocab size | 102400 | 129280 |
| Embedding tying | false | false |
| YaRN mscale_all_dim | 0.707 | 1.0 |
| Softmax scale | ≈ 0.1148 | ≈ 0.1353 |
| Weight dtype | BF16 | BF16 |

All other inference mechanics (MLA cache structure, RoPE decoupling, kv_a/kv_b
split, expert FFN SwiGLU structure, RMSNorm placement, weight absorption) are
identical between V2-Lite and V3.

## ButterQuant Considerations for MLA + MoE

### Fused Projections

Q_PROJ [3072, 2048] and KV_A_PROJ [576, 2048] share the same input (h_t) and
can be fused into a single Q_KVA GEMV [3648, 2048]. This is the MLA equivalent
of the fused QKV GEMV in standard GQA. Gamma absorption from input_layernorm
applies identically.

KV_B_PROJ [4096, 512] cannot join this fused GEMV — different input dimension
(512 vs 2048) and a data-dependent kv_a_layernorm in between. With weight
absorption, KV_B_PROJ is not applied to the cache at all: W_UK is absorbed into
the query side and W_UV into the output side, both as small per-head transforms.

### Dual-Copy KV Cache

In standard GQA ButterQuant, K and V caches use different quantization:
- K: dynamic per-position scale (scoring accuracy)
- V: fixed scale (clean AMX accumulation without per-position dequant)

In MLA, c_KV is one tensor serving both roles. The resolution is to store two
int8 copies of c_KV with different scales:
- Scoring copy: dynamic per-position scale S_cKV (like K)
- Aggregation copy: fixed scale (like V)

Cache budget per position per layer:

| Layout | Bytes |
|---|---|
| bf16 c_KV + k_R | (512 + 64) * 2 = 1152 |
| Dual-copy int8 | 512 (scoring) + 512 (aggregation) + 64 (k_R) + ~8 (scales) ≈ 1096 |
| Standard MHA bf16 | 2 * 16 * 128 * 2 = 8192 |

Same memory as bf16 MLA, 7.5x less than MHA, but both scoring and aggregation
operate on int8 — half the bandwidth per element. No quality compromise on either
operation. Both copies are written during cache insertion from the same source
(post-norm c_KV), just quantized with different scales.

### Router

The MoE router [64, 2048] is a small matmul followed by softmax and top-K
selection. Keep this in bf16 — the matmul is tiny and top-K selection is
sensitive to small score differences that quantization could flip.

### Expert FFNs

Each expert is a standard SwiGLU (gate/up/down), same structure as dense FFN,
same quantization scheme. Gamma absorption from post_attention_layernorm into
gate+up applies identically. Per-expert fixed V-scale (for down_proj input)
derived from each expert's weight norms at load time — 64 scales per MoE layer.

### FWHT Block Sizes

| Dimension | Size | Power of 2? | Block size |
|---|---|---|---|
| Hidden | 2048 | Yes | 2048 (single block) |
| Latent (c_KV) | 512 | Yes | 512 (single block) |
| RoPE (k_R) | 64 | Yes | 64 (single block) |
| Expert intermediate | 1408 | No | 128 (11 blocks per row, 1408/128 = 11) |
| Shared intermediate | 2816 | No | 128 (22 blocks per row, 2816/128 = 22) |
| Dense intermediate | 10944 | No | 128 (85.5 — does not divide cleanly) |

Dense intermediate 10944 is not divisible by 128 (10944/128 = 85.5). Possible
block sizes: 64 (10944/64 = 171), 32 (10944/32 = 342). Block size 64 works.
Only layer 0 uses the dense FFN.

### kv_a_layernorm

The RMSNorm between KV_A output and cache write applies to the 512-dim latent.
Its gamma cannot be absorbed into a downstream weight (the latent is cached, not
immediately projected). The gamma multiply must remain at runtime, applied before
FWHT+quantize during cache write. This is a 512-element pointwise multiply —
negligible cost.
