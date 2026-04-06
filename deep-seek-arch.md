# DeepSeek-V2/V3 Architecture for Inference

Reference implementation target: **DeepSeek-V2-Lite** (15.7B total, 2.4B active).
This document covers everything needed to implement forward inference, and notes
where V3 (671B) diverges.

## Model Dimensions (V2-Lite)

| Parameter | Symbol | V2-Lite | V3 |
|---|---|---|---|
| Layers | L | 27 | 61 |
| Hidden dim | d | 2048 | 7168 |
| Attention heads | n_h | 16 | 128 |
| Per-head dim | d_h | 128 | 128 |
| KV compression dim | d_c | 512 | 512 |
| Query compression dim | d'_c | N/A | 1536 |
| Decoupled RoPE dim (per-head) | d_R_h | 64 | 64 |
| Shared experts | N_s | 2 | 1 |
| Routed experts | N_r | 64 | 256 |
| Activated routed experts | K_r | 6 | 8 |
| Expert intermediate dim | - | 1408 | 2048 |
| Dense FFN layers | - | Layer 0 only | Layers 0-2 |
| Vocab size | V | 100K | 128K |
| Max context | - | 32K (base) | 128K |

## High-Level Forward Pass

```
for each token position t:
    h_t = embed(token_t)

    for each layer:
        # Attention block
        u_t = RMSNorm(h_t)
        u_t = MLA(u_t)
        h_t = h_t + u_t

        # FFN block
        u_t = RMSNorm(h_t)
        if layer is dense:
            u_t = DenseFFN(u_t)      # SwiGLU MLP
        else:
            u_t = DeepSeekMoE(u_t)   # shared + routed experts
        h_t = h_t + u_t

    h_t = RMSNorm(h_t)
    logits = LMHead(h_t)             # tied with embedding in V2-Lite
```

## Multi-head Latent Attention (MLA)

MLA replaces standard MHA. Instead of caching full K and V tensors per head,
it caches a single compressed latent vector c_KV plus a small decoupled RoPE key k_R.

### KV Cache Contents

Per token, the cache stores:
- `c_KV_t`: d_c elements (512 for both V2-Lite and V3)
- `k_R_t`: d_R_h elements (64)

Total cache per token per layer: **576 elements** (vs 2 * n_h * d_h = 4096 for standard MHA).

### Query Computation

**V2-Lite (no query compression):**
```
q_C_t = W_UQ @ h_t                         # [n_h * d_h, d] -> full query
q_R_t = RoPE(W_QR @ h_t)                   # [n_h * d_R_h, d] -> decoupled RoPE queries
```
W_UQ projects directly from the hidden state. No down-projection.

**V3 (with query compression):**
```
c_Q_t = W_DQ @ h_t                         # [d'_c, d] -> compressed query latent
q_C_t = W_UQ @ c_Q_t                       # [n_h * d_h, d'_c] -> full query
q_R_t = RoPE(W_QR @ c_Q_t)                 # [n_h * d_R_h, d'_c] -> decoupled RoPE queries
```
The extra down-projection W_DQ reduces activation memory during training.
For inference, the two projections (W_DQ then W_UQ) could be fused into one matrix,
but keeping them separate matches the checkpoint layout.

### KV Computation (same for V2-Lite and V3)

```
c_KV_t = W_DKV @ h_t                       # [d_c, d] -> compressed KV latent (CACHED)
k_R_t  = RoPE(W_KR @ h_t)                  # [d_R_h, d] -> decoupled RoPE key (CACHED)
```

Note: W_DKV produces a single vector shared across all heads. The up-projections
W_UK and W_UV are NOT applied during KV storage -- they are deferred to attention time.

### Attention Computation

For each query head i at position t, attending to cached position j:

```
# Recover per-head keys and values from the cached latent
k_C_j = W_UK[i] @ c_KV_j                   # [d_h] from [d_h, d_c] @ [d_c]
v_C_j = W_UV[i] @ c_KV_j                   # [d_h] from [d_h, d_c] @ [d_c]

# Concatenate content and RoPE components
q_i = [q_C_t_i ; q_R_t_i]                  # [d_h + d_R_h]
k_i = [k_C_j_i ; k_R_j]                    # [d_h + d_R_h] (k_R is shared across heads)

# Standard scaled dot-product attention
score = (q_i^T @ k_i) / sqrt(d_h + d_R_h)
o_t_i = sum_j softmax(score_j) * v_C_j_i
```

### Weight Absorption Optimization

The key inference optimization: W_UK can be absorbed into the query projection,
and W_UV can be absorbed into the output projection. This avoids materializing
full K and V tensors from the latent.

**Score computation (absorb W_UK into query):**
```
# Instead of: q_C_t_i^T @ (W_UK[i] @ c_KV_j)
# Precompute: q_hat_i = W_UK[i]^T @ q_C_t_i    (applied once per query)
# Then:       score_C = q_hat_i^T @ c_KV_j      (dot product with cached latent directly)
```

The content score becomes a dot product of a transformed query against the raw
cached latent -- no per-position key materialization needed.

**Value aggregation (absorb W_UV into output projection):**
```
# Instead of: o_t_i = sum_j w_j * (W_UV[i] @ c_KV_j)
# Compute:    o_latent_i = sum_j w_j * c_KV_j   (aggregate latents directly)
# Then:       o_t_i = W_UV[i] @ o_latent_i       (project once after aggregation)
```

Or equivalently, absorb W_UV into W_O so the latent aggregation feeds directly
into the output projection.

**RoPE score is separate** and cannot be absorbed (position-dependent):
```
score_R = q_R_t_i^T @ k_R_j                # standard RoPE dot product
score_total = (score_C + score_R) / sqrt(d_h + d_R_h)
```

### Practical Decode Attention

With absorption, decode attention per head becomes:
1. Transform query: `q_hat = W_UK^T @ q_C` (d_c output, applied once)
2. For each cached position j:
   - Content score: `q_hat^T @ c_KV_j` (d_c dot product)
   - RoPE score: `q_R^T @ k_R_j` (d_R_h dot product)
   - Combined: `(content + rope) / sqrt(d_h + d_R_h)`
3. Softmax over positions
4. Aggregate: `o_latent = sum_j w_j * c_KV_j` (weighted sum of d_c vectors)
5. Project: `o = W_UV @ o_latent` then through W_O

The KV cache read is now d_c + d_R_h = 576 elements per position instead of 2 * d_h = 256.
Wait -- that's larger per element, but there's only ONE set shared across all heads
(vs n_h sets for MHA). Total cache: 576 * L vs 2 * n_h * d_h * L = 4096 * L for MHA.
So 7x reduction.

## DeepSeekMoE (FFN)

### Structure

Each MoE layer has:
- **N_s shared experts** (always active, applied to every token)
- **N_r routed experts** (sparse, top-K_r selected per token)

Each expert is a standard SwiGLU FFN:
```
Expert(x) = (SiLU(W_gate @ x) * (W_up @ x)) @ W_down
```
with intermediate dimension 1408 (V2-Lite) or 2048 (V3).

### Routing

```
# Compute affinity scores
s_i = sigmoid(x^T @ e_i)         # for each routed expert i

# Select top-K_r experts
selected = top_k(s, K_r)

# Normalize gates among selected experts
g_i = s_i / sum(s_j for j in selected)   # only over selected experts

# Compute output
output = x + sum(Expert_shared_i(x) for i in 1..N_s)
           + sum(g_i * Expert_routed_i(x) for i in selected)
```

**V2-Lite uses softmax** for affinity scores instead of sigmoid.
**V3 uses sigmoid** with normalization over selected experts (as shown above).
This is a minor difference in the gating computation.

**V3 auxiliary-loss-free routing** adds a bias term b_i to affinity scores for
top-K selection only. The bias does not affect the gate value used for weighting.
This is a training concern and does not change the inference computation -- the
trained weights already reflect the routing patterns. At inference, just compute
sigmoid affinities and select top-K.

### Dense Layers

Layer 0 (V2-Lite) or layers 0-2 (V3) use a standard dense SwiGLU FFN instead
of MoE. Same activation function, just one large expert instead of many small ones.

## RMSNorm

Standard RMSNorm without learnable bias, with learnable scale gamma:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```
eps = 1e-6. Applied before attention and before FFN in each layer (pre-norm).

Additional RMSNorm layers are applied after compressed latent vectors in MLA
(after c_KV and c_Q). These are noted in the DeepSeek papers as stabilization
for the low-rank bottlenecks.

## Weight Layout Summary (V2-Lite)

### Per layer -- Attention (MLA)

| Weight | Shape | Notes |
|---|---|---|
| W_UQ | [n_h * d_h, d] | Query up-projection (V2-Lite: no down-proj) |
| W_QR | [n_h * d_R_h, d] | Decoupled RoPE query projection |
| W_DKV | [d_c, d] | KV down-projection |
| W_KR | [d_R_h, d] | Decoupled RoPE key projection |
| W_UK | [n_h * d_h, d_c] | Key up-projection (can be absorbed) |
| W_UV | [n_h * d_h, d_c] | Value up-projection (can be absorbed) |
| W_O | [d, n_h * d_h] | Output projection |
| kv_norm | [d_c] | RMSNorm scale for c_KV |

V2-Lite shapes: W_UQ is [2048, 2048], W_DKV is [512, 2048], W_UK and W_UV are [2048, 512],
W_KR is [64, 2048], W_QR is [1024, 2048], W_O is [2048, 2048].

**V3 adds:** W_DQ [d'_c, d] = [1536, 7168] and q_norm [d'_c] for query compression.
W_UQ becomes [n_h * d_h, d'_c] = [16384, 1536] instead of [n_h * d_h, d].

### Per layer -- Dense FFN (layer 0)

| Weight | Shape |
|---|---|
| W_gate | [intermediate, d] |
| W_up | [intermediate, d] |
| W_down | [d, intermediate] |

V2-Lite dense intermediate is 5632 (typical ~2.75x hidden for SwiGLU).

### Per layer -- MoE FFN (layers 1-26)

| Weight | Shape | Count |
|---|---|---|
| gate_proj | [1408, 2048] | per expert (2 shared + 64 routed = 66 total) |
| up_proj | [1408, 2048] | per expert |
| down_proj | [2048, 1408] | per expert |
| router.e | [2048, 64] | routing centroids (one per routed expert) |

### Global

| Weight | Shape |
|---|---|
| embed_tokens | [V, d] = [100000, 2048] |
| final_norm | [d] = [2048] |
| lm_head | tied with embed_tokens |

## NUMA / Expert Placement Strategy

With 4 NUMA nodes and 64 routed experts per layer, a natural mapping is
16 experts per node. The 2 shared experts can be replicated on all nodes
(they're always active and small relative to the routed set).

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
| Query compression | None (direct projection) | W_DQ down-projects first |
| Gating function | Softmax | Sigmoid + normalize |
| Shared experts | 2 | 1 |
| Routed experts | 64 (top-6) | 256 (top-8) |
| Dense layers | Layer 0 | Layers 0-2 |
| MTP module | None | 1 additional depth (discardable) |
| Node-limited routing | Not applicable | Top-K constrained to M=4 nodes |
| Vocab / tokenizer | 100K BBPE | 128K BBPE |

All other inference mechanics (MLA, expert FFN structure, RMSNorm placement,
RoPE, weight absorption) are identical.
