# Gated DeltaNet: Mathematical Reference

This document is a from-first-principles engineering reference for the Gated Delta Rule and its hardware-efficient chunkwise implementation, which together constitute the linear-attention layer used by Qwen3-Next and Qwen3.5 / Qwen3.6. It is built up from the original sources: the Yang–Kautz–Hatamizadeh paper (ICLR 2025, arXiv 2412.06464), the upstream Flash Linear Attention (FLA) library implementation, the Yang et al. (2024) "Parallelizing Linear Transformers with the Delta Rule" paper that supplies the chunked WY decomposition, the Schlag–Irie–Schmidhuber (2021) "Linear Transformers are Secretly Fast Weight Programmers" paper that introduced the delta rule for linear attention, and the Mamba2 / SSD paper (Dao & Gu, 2024) for the data-dependent decay term. Each formula is cross-referenced to the actual FLA / vLLM / Transformers source files in `references/`.

The intended audience is someone who is *implementing* the kernel — either a chunkwise WY-form on a GPU, or a from-scratch SIMD reference for verification — not someone who is just calling it.

## 1. Lineage

Gated DeltaNet is a synthesis of three earlier ideas:

| Year | Author | Idea | Update Rule |
| --- | --- | --- | --- |
| 2020 | Katharopoulos et al. | Linear attention as RNN with outer-product state | `S_t = S_{t-1} + v_t k_t^T` |
| 2021 | Schlag, Irie, Schmidhuber | Delta rule replaces additive write with linear interpolation against the current state | `S_t = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T` |
| 2024a | Dao & Gu (Mamba2 / SSD) | Data-dependent scalar decay for state retention | `S_t = α_t S_{t-1} + v_t k_t^T` |
| 2024b | Yang, Wang, Zhang, Shen, Kim | Chunked parallel WY-form for the delta rule | (parallelizes 2021's recurrence) |
| 2024c | Yang, Kautz, Hatamizadeh | **Gated Delta Rule** (the topic of this doc) | `S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T` |

The motivating observation of the GDN paper is that the two scalar gates address different failure modes:

- **Decay `α_t` (Mamba2-style)** uniformly forgets *all* key-value associations at each step. It enables rapid memory erasure but is too coarse: it cannot selectively forget one association without forgetting others.
- **Delta rule `β_t` (DeltaNet-style)** modifies *only* the association at the current key. It supports targeted updates but provides no global clearing mechanism, so stale associations linger forever unless they happen to be revisited.

Combining them gives a state update with both adaptive global decay and adaptive per-key writing. Setting `α_t → 0` clears state instantly; setting `α_t → 1` recovers pure DeltaNet.

## 2. The Underlying Object: a Matrix-Valued State

Every model in this family maintains a per-head, per-layer *state matrix* `S_t ∈ R^{d_k × d_v}` rather than a stack of past keys/values. The state is the cumulative outer product of past values and (transformed) keys; the query "reads" the state by left-multiplying:

```
o_t = S_t^T q_t           # (d_v,)        if state is (d_k, d_v)
```

Equivalently, in the form used by the GDN paper (state shape `R^{d_v × d_k}`):

```
o_t = S_t q_t             # (d_v,)        if state is (d_v, d_k)
```

The two conventions differ only by a transpose. The FLA library and the GDN paper use the `(d_v, d_k)`-or-equivalent ordering with `o_t = S_t q_t`; the Qwen3-Next reference / Schlag formulation tends to use `o_t = S_t^T q_t`. This doc adopts the GDN paper's convention `S ∈ R^{d_v × d_k}` so that the equations match the source paper byte-for-byte. When reading the FLA Triton kernels, watch for the `einsum('bhd,bhdm->bhm', q, S)` pattern (state is `(d_k, d_v)` there, with `o = S^T q`); both formulations are mathematically equivalent.

The state is per-head: there are `num_v_heads` independent `(d_v, d_k) = (head_v_dim, head_k_dim)` states per layer. For Qwen3.6-35B-A3B that is `32 × (128, 128) = 32` independent 128×128 states per linear-attention layer, totaling `32 × 128 × 128 × 4 = 2 MB` of float32 state per linear-attention layer per token slot in the cache (small and constant in sequence length — the headline win of linear attention).

## 3. The Four Recurrences, Side by Side

Let `q_t, k_t ∈ R^{d_k}`, `v_t ∈ R^{d_v}`. All use a single attention head; the multi-head version replicates per head.

```
Linear Attention (Katharopoulos 2020):
    S_t = S_{t-1} + v_t k_t^T                                       (eq. LA)
    o_t = S_t q_t

Mamba2 / SSD (Dao & Gu 2024):
    S_t = α_t S_{t-1} + v_t k_t^T                  α_t ∈ (0, 1)     (eq. M2)
    o_t = S_t q_t

DeltaNet (Schlag 2021, Yang 2024b):
    v_t^old = S_{t-1} k_t                                           # current state's prediction at this key
    v_t^new = β_t v_t + (1 - β_t) v_t^old                           # convex blend of new and old
    S_t = S_{t-1} - v_t^old k_t^T + v_t^new k_t^T
        = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T               (eq. DN)
    o_t = S_t q_t                                  β_t ∈ (0, 1)

Gated DeltaNet (Yang–Kautz–Hatamizadeh 2024, eq. 10):
    S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T
        = α_t S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T           (eq. GDN)
    o_t = S_t q_t                                  α_t ∈ (0, 1), β_t ∈ (0, 1)
```

The transition operator `T_t := α_t (I - β_t k_t k_t^T)` is what the recurrence multiplies the previous state by. It is a *generalized Householder*: rank-1 perturbation of a scaled identity.

Three sanity checks:

1. **`β_t = 0` collapses to Mamba2.** `T_t = α_t I` and `S_t = α_t S_{t-1} + 0`. The state still receives `β_t v_t k_t^T = 0`, so this corresponds to "no write this step, just decay."
2. **`α_t = 1` collapses to DeltaNet.** `T_t = I - β_t k_t k_t^T`, and the recurrence is exactly Schlag's delta rule.
3. **`α_t = 0` clears the state.** `S_t = 0 + β_t v_t k_t^T`, i.e. the state is reset to a single-key-value association.

## 4. Online-Learning Interpretation

Every recurrence in this family is the closed form of one step of stochastic gradient descent on an online objective. Following Liu et al. (2024) (the "Longhorn" paper) and Sun et al. (2024) (the TTT paper), think of `S` as a fast-weight matrix whose job is to memorize the current key-value association. Each row of the GDN paper's Table 1 is one (objective, update-rule) pair:

| Method | Online objective `L(S_t)` | Resulting closed-form update |
| --- | --- | --- |
| Linear Attention | `‖S_t − S_{t−1}‖_F^2 − 2⟨S_t k_t, v_t⟩` | `S_t = S_{t−1} + v_t k_t^T` |
| Mamba2 | `‖S_t − α_t S_{t−1}‖_F^2 − 2⟨S_t k_t, v_t⟩` | `S_t = α_t S_{t−1} + v_t k_t^T` |
| Longhorn | `‖S_t − S_{t−1}‖_F^2 + β_t ‖S_t k_t − v_t‖^2` | implicit (closed-form globally optimal) |
| DeltaNet | `‖S_t − S_{t−1}‖_F^2 − 2⟨S_t k_t, β_t (v_t − S_{t−1} k_t)⟩` | `S_t = S_{t−1} (I − β_t k_t k_t^T) + β_t v_t k_t^T` |
| **Gated DeltaNet** | `‖S_t − α_t S_{t−1}‖_F^2 − 2⟨S_t k_t, β_t (v_t − α_t S_{t−1} k_t)⟩` | `S_t = S_{t−1} (α_t (I − β_t k_t k_t^T)) + β_t v_t k_t^T` |

The DeltaNet / Gated DeltaNet rows admit a cleaner interpretation. Define the per-step regression loss

```
L(S) = (1/2) ‖S k_t − v_t‖^2
```

Its gradient with respect to `S` is `(S k_t − v_t) k_t^T`, so an SGD step with learning rate `β_t` gives

```
S_{t+1} = S_t − β_t (S_t k_t − v_t) k_t^T
        = S_t (I − β_t k_t k_t^T) + β_t v_t k_t^T
```

— exactly DeltaNet. Pre-multiplying the previous state by `α_t` before the SGD step is the *adaptive weight decay* that turns DeltaNet into Gated DeltaNet:

```
S_{t+1} = α_t S_t − β_t (α_t S_t k_t − v_t) k_t^T
        = α_t S_t (I − β_t k_t k_t^T) + β_t v_t k_t^T
```

Practical implication: `β_t` is the per-token learning rate of the inner online learner, and `α_t` is its weight decay. The model can independently choose how aggressively to write at the current key (`β_t`) and how aggressively to forget the running memory (`α_t`).

## 5. Vector and Matrix Forms (Sequence Level)

Unrolling the recurrence (eq. GDN) from `t = 1` and assuming `S_0 = 0`:

```
S_t = sum_{i=1}^{t} γ_t^i β_i v_i k_i^T · ∏_{j=i+1}^{t} (I − β_j k_j k_j^T)            (eq. GDN-unrolled)
```

where

```
γ_t^i := ∏_{j=i+1}^{t} α_j                                   (cumulative decay from i+1 to t)
```

In practice it is convenient to work with `α_t` parameterized on log-scale: define `g_t = log α_t < 0` and the cumulative log-decay `G_j = sum_{i=1}^{j} g_i`. Then `γ_t^i = exp(G_t − G_i)`, and `γ_t^i / γ_t^j = exp(G_j − G_i)` are the only differences-of-cumulative-sums the kernel needs.

Reading at time `t`:

```
o_t = S_t q_t = sum_{i=1}^{t} (γ_t^i / γ_t^t) · β_i v_i (k_i^T q_t)' · [Householder corrections]
```

— the output is a weighted sum over past values, where the weight of past step `i` is the dot product `k_i^T q_t` adjusted by (a) the cumulative scalar decay since step `i`, (b) the cumulative product of generalized Householder matrices since step `i`. The Householder correction is what distinguishes DeltaNet-family recurrences from plain decayed linear attention: the weights are not just decayed dot products but also include the "pushback" from intervening writes at correlated keys.

## 6. Why a Naive Implementation is Hopeless

The recurrence is sequential: `S_t` depends on `S_{t−1}`. Naively training on a sequence of length `L` requires `L` sequential matrix-multiplies of `(d_v, d_k)` states by `(d_k, d_k)` transition matrices. For `L = 8192`, `d_k = d_v = 128`, you get `8192 × (128² + 128² · 128) = 8192 × 2.1M = 17B FLOPs/head/layer` arranged as a serial chain — nothing for tensor cores to chew on.

Two problems must be solved together:

1. **Dependency-breaking parallelism over the sequence axis** (so the GPU can do multiple chunks in parallel and run high-arithmetic-intensity matmuls).
2. **A closed-form expression for the cumulative product `∏ T_j`** that does not require materializing each transition matrix.

The DeltaNet 2024b paper introduced the key insight: a chunked WY decomposition turns the cumulative product of generalized Householders inside one chunk into a single triangular solve plus a few matrix-multiplies. The GDN paper extends this to include the gating scalars.

## 7. Chunked Parallel Form: Overview

Partition the sequence of length `L` into `N = L / C` chunks of size `C` (canonical: `C = 64` in FLA / Qwen3-Next; `C = 32` in some naive references; `C` is a power of two for kernel friendliness).

Notation: `[t]` indexes chunks; `Q_{[t]} ∈ R^{C × d_k}` is the chunk-`t` queries stacked rowwise (and similarly for `K, V, β, g`). `S_{[t]}` denotes the running state at the *start* of chunk `t` (so `S_{[0]} = S_0`, the initial state).

The chunk-level recurrence has two terms:

```
S_{[t+1]} = (decay factor over chunk) · S_{[t]} · P_{[t]}    +    H_{[t]}                    (eq. CP-1)
```

- `P_{[t]} := ∏_{i=1}^{C} (I − β_{[t],i} k_{[t],i} k_{[t],i}^T)`, the cumulative *Householder* product within the chunk (no decay).
- `H_{[t]} := sum_{i=1}^{C} β_{[t],i} v_{[t],i} k_{[t],i}^T · ∏_{j=i+1}^{C} (I − β_{[t],j} k_{[t],j} k_{[t],j}^T)`, the new contributions from this chunk's writes.
- Decay factors are the cumulative `α_t` products folded back in. The GDN paper (eq. 10 in §3.3) writes the gated equivalent of `P_{[t]}` and `H_{[t]}` as `F_{[t]}^r` and `G_{[t]}^r` with the `α` factors threaded through.

The output at any position `r` within chunk `t` is:

```
o_{[t]}^r = q_{[t]}^r · S_{[t]} · (γ-decay to position r) + (chunk-local correction with U, W)            (eq. CP-2)
```

Equivalently, vectorized over the chunk:

```
O_{[t]} = Q̃_{[t]} S_{[t]}^T + (Q_{[t]} K_{[t]}^T ⊙ Γ_{[t]} ⊙ M) · (Ũ_{[t]} − W̃_{[t]} S_{[t]}^T)         (eq. CP-3)
```

where `M` is a strict upper-triangular causal mask (`M_{ij} = 1` iff `i ≥ j`), `Γ_{[t]}` is a decay-aware mask (entry `(i, j)` = `γ_i / γ_j` if `i ≥ j`, else 0), and `(Ũ, W̃)` are the WY-decomposition tensors described next. The arrows on `Q̃, K̃, S̃` denote "decayed to a particular reference position within the chunk," consistent with §2.1 of the GDN paper.

Eq. CP-3 is what the kernel actually computes: two matmuls per chunk plus a triangular solve per chunk to build `(Ũ, W̃)`. The work is `O(L · C · d_v + L · d_v · d_k)`, with the inner matmuls being chunk-shaped (`C × d_k`, `C × d_v`, `d_k × d_v`) — friendly to tensor cores.

## 8. The WY Representation (and the UT Trick)

The cumulative product `P_{[t]} = ∏_{i=1}^{C} (I − β_i k_i k_i^T)` is a product of generalized Householder matrices (`I − u u^T` with `u = √β_i k_i`). The classical WY representation (Bischof & Loan, 1985) writes any such product as a single rank-`C` perturbation:

```
P_{[t]} = I − W_{[t]}^T K_{[t]}                            W_{[t]} ∈ R^{C × d_k}, K_{[t]} ∈ R^{C × d_k}    (eq. WY-1)
```

Computing `W` row-by-row gives a triangular recurrence (eq. 4 in the GDN paper):

```
w_r = β_r (k_r − sum_{i=1}^{r-1} w_i (k_i^T k_r))                                                          (eq. WY-2)
```

— each `w_r` is `β_r` times the residual of `k_r` after subtracting its projection onto the span of previous `w_i`. This is *exactly* the Gram–Schmidt-flavored `O(C^2)` triangular solve that the FLA `naive_chunk_gated_delta_rule` implements as the inner `for i in range(1, chunk_size)` loop in `references/fla/naive.py:131-133` — followed by adding `I` (line 133):

```python
attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, ...)
```

That `attn` matrix at the end of the loop is `T_{[t]} := [I + strictLower(diag(β) K K^T)]^{-1} · diag(β)`, the *UT transform* of Joffrain et al. (2006). It is the inverse-of-strict-lower-triangular-plus-diagonal — and Joffrain et al. showed it is exactly the matrix you get by running the Gram–Schmidt-flavored loop. Concretely, once `T` is in hand:

```
W_{[t]} = T_{[t]} K_{[t]}                                   ∈ R^{C × d_k}
U_{[t]} = T_{[t]} V_{[t]}                                   ∈ R^{C × d_v}
```

(eq. 7 of the paper). Now `H_{[t]} = U_{[t]}^T K_{[t]}` (eq. 5), and the cumulative-product/cumulative-write decomposition becomes:

```
S_{[t+1]} = S_{[t]} · (I − W_{[t]}^T K_{[t]}) + U_{[t]}^T K_{[t]}
          = S_{[t]} + (U_{[t]} − W_{[t]} S_{[t]}^T)^T K_{[t]}                                              (eq. WY-3)

O_{[t]} = Q_{[t]} S_{[t]}^T + (Q_{[t]} K_{[t]}^T ⊙ M) (U_{[t]} − W_{[t]} S_{[t]}^T)                        (eq. WY-4)
```

(equations 8 and 9 in the paper). Two important observations:

- **Only `(U − W S^T)` appears in both equations**, so the kernel computes it once per chunk and reuses it. FLA names this quantity `v_new` (it is the "actual write applied this chunk, after subtracting what the running state would already say"). See `references/fla/naive.py:148-153`:

  ```python
  v_prime = (k_cumdecay[:, :, i]) @ S
  v_new = v_i - v_prime
  o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
  o[:, :, i] = o_inter + attn @ v_new
  S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (...)).transpose(-1, -2) @ v_new
  ```

- **Triangular solve becomes a matrix prefix scan over the chunk axis.** The for-loop in eq. WY-2 is `O(C^2)` work but `O(log C)` depth; on GPU it is unrolled into a chunk-shared-memory routine. For `C = 64` and `d_k = 128` this is ~$2^{16}$ FLOPs per chunk per head — negligible compared to the matmuls.

## 9. Adding Decay: From DeltaNet to Gated DeltaNet

The above is for pure DeltaNet. To get Gated DeltaNet we thread the cumulative `α` factors through. Define:

```
G_j := log α_j                                  (per-step log-decay)
γ_j^[i:r] := exp(G_r − G_i) = ∏_{m=i+1}^{r} α_m  (decay from position i to r within chunk)
```

The GDN paper §3.3 shows that the same WY trick still works once decay is included, with the substitutions:

- `K_{[t]}` is replaced by `K̃_{[t]}` where row `r` is rescaled by `γ_{[t]}^C / γ_{[t]}^r` (the decay from the row's position to the *end* of the chunk).
- `Q_{[t]}` is replaced by `Q̃_{[t]}` where row `r` is rescaled by `γ_{[t]}^r` (the decay from the start of the chunk to the row's position).
- The state `S_{[t]}` itself is decayed by `γ_{[t]}^C` (the full chunk's decay) before being added to.
- The local mask `M` is replaced by `Γ_{[t]} ⊙ M` where `(Γ_{[t]})_{ij} = γ_{[t]}^i / γ_{[t]}^j` for `i ≥ j`.
- The triangular system uses `[I + strictLower(diag(β) (Γ_{[t]} ⊙ K K^T))]^{-1} diag(β)`.

The kernel implementations precompute `decay = g.cumsum(-1)` per chunk (so `decay[..., r]` is `G_{[t]}^r − G_{[t]}^0`, the chunk-relative log-cumulative-decay), then build `L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()` which is exactly `Γ_{[t]} ⊙ M` (see `references/fla/naive.py:127-129`). The state-carry update gets a multiplicative `decay[:, :, i, -1, None, None].exp()` factor (see line 152) which is `γ_{[t]}^C`.

End-to-end, the gated chunk update is:

```
T_{[t]} = [I + strictLower(diag(β) · (Γ_{[t]} ⊙ K_{[t]} K_{[t]}^T))]^{-1} · diag(β)         (eq. GDN-T)
Ũ_{[t]} = T_{[t]} V_{[t]}                                                                   (eq. GDN-U)
W̃_{[t]} = T_{[t]} K_{[t]}                                                                   (eq. GDN-W)

V_new   = Ũ_{[t]} − decay-adjusted(W̃_{[t]} · S_{[t]}^T)                                     (eq. GDN-Vnew)

O_{[t]} = decay-adjusted(Q_{[t]} S_{[t]}^T) + (Q_{[t]} K_{[t]}^T ⊙ Γ_{[t]} ⊙ M) · V_new     (eq. GDN-O)
S_{[t+1]} = γ_{[t]}^C · S_{[t]} + decay-adjusted(K_{[t]})^T · V_new                          (eq. GDN-S)
```

In `references/fla/naive.py:135-153` you can read this exact computation.

## 10. The Recurrent Form (Decode Path)

Eq. CP-1 collapses to single-token form when `C = 1`:

```
S_t   = α_t · S_{t-1}
v_pred = S_t k_t                                        # what the (already-decayed) state predicts at this key
v_new  = β_t (v_t − v_pred)                             # the SGD-with-weight-decay residual
S_t    = S_t + k_t v_new^T                              # apply the write
o_t    = S_t q_t                                        # read
```

This is `naive_recurrent_gated_delta_rule` in `references/fla/naive.py:50-59`:

```python
for i in range(T):
    b_q = q[:, :, i]
    b_k = k[:, :, i]
    b_v = v[:, :, i].clone()
    h = h.clone() * g[:, :, i].exp()[..., None, None]      # decay
    b_beta = beta[:, :, i]
    b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)       # subtract current prediction
    b_v = b_v * b_beta[..., None]                          # scale by β (learning rate)
    h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)  # rank-1 update
    o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)     # read
```

This is what the FLA `fused_recurrent_gated_delta_rule` Triton kernel implements per-token, used during decode (`q_len <= 64 and not self.training` per `gated_deltanet_layer.py:219`). The arithmetic per step is `O(d_k · d_v)` — one rank-1 outer-product update of the `(d_k, d_v)` state — and it is bandwidth-bound rather than compute-bound at the small batch sizes typical of decode.

## 11. Block-Level Architecture

The "linear-attention layer" delivered to a transformer block is more than the GDN recurrence. A canonical Gated DeltaNet block (paper §3.4, FLA `gated_deltanet.py`, Qwen3-Next `Qwen3NextGatedDeltaNet`) wraps the recurrence in five additional pieces:

```
                hidden_state x ∈ R^{B×T×D}
                        │
                        ▼
                ┌───────────────┐
                │ input layernorm│   (RMSNorm, applied outside the GDN module by the decoder layer)
                └───────────────┘
                        │
              ┌─────────┴────────────────────────────┐
              ▼                  ▼                ▼  ▼
    in_proj_qkv(x)       in_proj_z(x)       in_proj_b(x)  in_proj_a(x)
    (D → 2·k_dim+v_dim)  (D → v_dim)        (D → H_v)     (D → H_v)
              │                  │                 │       │
              ▼                  │                 ▼       ▼
       depthwise causal           │           β = σ(b)   g = −exp(A_log)·softplus(a + dt_bias)
       Conv1d(kernel=4) + SiLU    │
              │                   │
              ▼                   │
       split → q, k, v            │
       (k_dim, k_dim, v_dim)      │
              │                   │
              ▼                   │
       L2-norm q, L2-norm k       │
       (inside the kernel)        │
              │                   │
              ▼                   │
       chunk_gated_delta_rule(    │
           q, k, v, g, β,         │
           use_qk_l2norm_in_kernel│
       )                          │
              │                   │
              └───────► o ────────┴────► RMSNormGated(o, z) ───► o_norm output
                                                                   │
                                                                   ▼
                                                            out_proj(D → D)
```

There is one Mamba-style depthwise causal Conv1d *over the channel axis after the input projection* and *before the QKV split*. It mixes a small local window (kernel 4) into each channel of the `key_dim·2 + value_dim` stream. This shortconv is ablated in the original DeltaNet paper to be load-bearing — without it the linear-attention layer struggles on local-token tasks.

Q and K are L2-normalized along the head dimension before the recurrence (`use_qk_l2norm_in_kernel=True`). The L2 norm bounds `‖k‖₂ = 1`, which keeps `‖I − β k k^T‖₂ ≤ 1` (since `k k^T` is a rank-1 projection with eigenvalues 0 and `‖k‖² = 1`); this ensures the recurrence is non-expansive in the operator norm and prevents the state from blowing up over long sequences. It is mandatory for stable training of Gated DeltaNet at 32K+ contexts.

The output passes through a fused gated RMSNorm (`FusedRMSNormGated`): `RMSNorm(o) ⊙ SiLU(z)` — element-wise multiplied by a SiLU-activated z stream. This is the gating on the *output* (not the recurrence), modelled after Mamba's selective output gate. The `z` stream is a separate linear projection (`in_proj_z`); it is *not* the same as the SSM gate `α_t`.

Finally `out_proj` projects from the value space back to the residual width.

## 12. The Two Gates: a, b and Their Parameterizations

The two scalar gates per step per head are produced by separate low-rank linears:

```
a_t = in_proj_a(x_t)             ∈ R^{H_v}              # raw, real-valued
b_t = in_proj_b(x_t)             ∈ R^{H_v}              # raw, real-valued
```

**β (per-step write strength).** Parameterized as `β_t = sigmoid(b_t)`, so `β ∈ (0, 1)`. In the canonical FLA formulation `β` is a per-head scalar (one value per `H_v` axis), broadcast across the `head_v_dim`. Some variants (e.g. Longhorn) use a vector-valued `β`, but the canonical Qwen3-Next / FLA / GDN-paper version is scalar-per-head. There is an `allow_neg_eigval` flag in FLA (`gated_deltanet_layer.py:262`) that multiplies `β` by 2 (so `β ∈ (0, 2)`); this lets the eigenvalues of `(I − β k k^T)` go negative, which Grazzi et al. (2024) showed is necessary for state-tracking. Qwen3-Next leaves it off.

**α (per-step decay).** Parameterized through Mamba2's discretization recipe:

```
Δ_t = softplus(a_t + dt_bias)              # discrete-time time step, > 0
α_t = exp(−Δ_t · exp(A_log))               # ∈ (0, 1)
g_t = log α_t = −Δ_t · exp(A_log)          # < 0
```

`A_log` is a per-head learned parameter holding the log of the *continuous-time* decay rate `A > 0`. Initialized by sampling `A ~ Uniform(0, 16)` and storing `log A` (see `references/fla/gated_deltanet_layer.py:150-152` and `references/transformers/qwen3_next_gdn.py:534-535`). `dt_bias` initializes to give `Δ ∈ [dt_min, dt_max] = [0.001, 0.1]` at the start of training (FLA chooses a uniform sample on the log-scale and inverts softplus, lines 156-164 of `gated_deltanet_layer.py`).

The kernel takes `g_t = log α_t` as input and exponentiates inside (so the chunkwise `decay.cumsum(-1).exp()` works directly on log-decays). In the Qwen3-Next reference forward (`references/transformers/qwen3_next_gdn.py:494`):

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

— exactly the parameterization above, with `g` a tensor of shape `(B, T, H_v)` of negative log-decays.

## 13. Numerical Stability Tricks

Implementations that work in production differ from the math in a handful of places. None of them change the equations; they keep them numerically tractable.

- **All kernel internals are float32.** Even when the surrounding model is bf16/fp16, the `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` cast `q, k, v, β, g` to float32 before entering the loop and cast `o` back to the input dtype on exit. The state `S` is float32 throughout (`mamba_ssm_dtype: "float32"` in the Qwen3.6 config). This is because `α_t = exp(−Δ · exp(A_log))` can underflow rapidly in fp16 — once `α^t` rounds to zero you have lost the entire history. The Qwen3.6-35B-A3B `config.text_config.mamba_ssm_dtype = "float32"` pins this contract on disk.
- **`A_log.float()` before `.exp()`.** Even with mixed-precision the `A_log` parameter is force-cast to float32 before exponentiation to avoid `exp(-inf) → 0` from a bf16 round-trip on extreme values. The Qwen3-Next forward at line 494 puts the `.float()` explicitly: `-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`.
- **Q/K L2-norm inside the kernel.** As described in §11, this is non-optional for stability over long contexts. The `use_qk_l2norm_in_kernel=True` flag instructs the Triton kernel to apply `l2norm(x, dim=-1, eps=1e-6)` to Q and K right before the recurrence, so the norm runs in float32 and avoids the round-trip dance of doing it outside in bf16.
- **Cumulative-decay differences instead of raw cumulatives.** The kernel never materializes `γ_t^i = exp(G_t − G_i)` directly; it cumulative-sums `g` (log-decays) per chunk and constructs the `Γ` mask via `(decay[..., None] - decay[..., None, :]).tril().exp()`. This avoids producing tiny near-zero numbers that would underflow to zero before reaching the matmul.
- **Causal masking *before* the chunkwise `attn = -K_β K^T` softmax-equivalent step.** The strict-upper-triangular mask is applied with `masked_fill_(mask, 0)`, not by zeroing after the matmul; this avoids accumulating noise from positions that are about to be discarded.
- **Initial state.** When `cache_params is None` (training / first prefill chunk), `S = zeros(B, H_v, d_k, d_v)`. When loading from cache, the previous final state is re-used. There is no special initialization per layer; the model learns to start from zero.
- **Pad to a multiple of `chunk_size`.** Sequences that are not a multiple of `C` are right-padded with zeros (`F.pad(q, (0, 0, 0, pad_len))` etc. in `naive_chunk_gated_delta_rule`). The padded positions still consume compute but contribute nothing to the state because their `β` and `v` are zero.

## 14. Cache Layout and the Decode Path

Each linear-attention layer's per-token state has two parts:

- **Conv state**: `(conv_dim, conv_size − 1)` per layer. For Qwen3.6, `conv_dim = key_dim·2 + value_dim = 4096 + 4096 = 8192` and `conv_size = 4`, so `8192 × 3 = 24,576 float values` per layer. This is the trailing few tokens needed to compute the next conv output without re-reading history.
- **Recurrent state**: `(num_v_heads, head_k_dim, head_v_dim)` per layer = `(32, 128, 128) = 524,288 float32 values = 2 MB` per layer. 30 linear-attention layers = `60 MB` of state per sequence. Independent of sequence length.

Compare with a full-attention KV cache: at `head_dim = 256`, `2 KV heads`, `bf16`, sequence length 32K, that's `32K × 2 × 256 × 2 = 32 MB` per *layer* and grows linearly. So at 32K context the linear-attention state is already 16× smaller per layer; at 256K it is 128× smaller; at 1M tokens it is 512× smaller. This is the headline win of GDN as a long-context primitive.

Decode dispatch (`gated_deltanet_layer.py:219`):

```python
mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
```

- `mode == 'chunk'`: training and prefill (chunked WY-form, tensor-core friendly).
- `mode == 'fused_recurrent'`: decode (single-token rank-1 update).

The recurrent path takes the previous `S` and `conv_state` from the cache, applies the conv update via `causal_conv1d_update` (which slides one token through and writes back the new tail), then runs one rank-1 outer-product update on `S` and reads `o = S q`.

## 15. Hybrid Architectures

The original GDN paper proposes two hybrid variants; Qwen3-Next / Qwen3.5 / Qwen3.6 use a third (also a hybrid):

- **GatedDeltaNet-H1**: GDN + sliding-window attention (SWA), interleaved per layer.
- **GatedDeltaNet-H2**: GDN + Mamba2 + SWA, interleaved.
- **Qwen3.5/3.6 layout**: GDN ×3 + full-attention ×1, repeated. No SWA layer; the full-attention layer is a standard gated-GQA softmax attention with M-RoPE.

The motivation is the same in all three: GDN handles long-context retention with constant-memory state, but its rank-`d_k` write capacity caps the number of independent key-value associations it can store before collisions. Periodic full-attention (or windowed attention) layers provide an explicit O(L)-memory primitive that the GDN layers can offload high-resolution local recall to. In Qwen3.6's case, 1 in 4 layers is full-attention (10 of 40 layers), so 75% of the parameter budget gets the linear-time primitive and 25% gets the quadratic-time one.

This hybrid pattern is also what makes the linear-attention layers' state size manageable: at full sequence length you only need `30 × 60 MB / 30 = 60 MB` of GDN state per sequence, plus `10 × ` (KV cache for the full-attention layers) — far less than 40 layers of full attention would consume.

## 16. Where the Math Maps to Code

Quick map from the eqs in this doc to specific lines in the references:

| Equation / concept | File | Lines |
| --- | --- | --- |
| Recurrent rule (eq. GDN) | `references/fla/naive.py` | 50–59 (state update + read) |
| Per-chunk WY triangular system (eq. GDN-T) | `references/fla/naive.py` | 121–133 |
| `Γ`-mask construction | `references/fla/naive.py` | 127–129 (`L_mask`) |
| Per-chunk `(Ũ, W̃)` (eq. GDN-U/W) | `references/fla/naive.py` | 135–137 (`k_cumsum`, `k_cumdecay`) |
| `V_new` (eq. GDN-Vnew) | `references/fla/naive.py` | 148–149 |
| Output (eq. GDN-O) | `references/fla/naive.py` | 150–151 |
| State carry (eq. GDN-S) | `references/fla/naive.py` | 152–153 |
| Conv1d front-end | `references/fla/gated_deltanet_layer.py` | 169–193 |
| `α` parameterization | `references/transformers/qwen3_next_gdn.py` | 494 |
| `β` parameterization | `references/transformers/qwen3_next_gdn.py` | 492 |
| Q/K L2-norm in-kernel | `references/fla/gated_deltanet_layer.py` | 275 (kwarg) |
| Output gated RMSNorm | `references/fla/gated_deltanet_layer.py` | 306–310 |
| Decode dispatch | `references/fla/gated_deltanet_layer.py` | 219 |
| Triton chunkwise kernel | `references/fla/chunk.py`, `chunk_fwd.py`, `wy_fast.py` | (production kernel; see file headers) |
| Cache layout | `references/transformers/qwen3_next_gdn.py` | 532–547, 596–698 |

## 17. Suggested First-Principles Implementation Order

If you are writing this from scratch (e.g. a Mojo or SIMD reference), the following order minimizes the time between "code compiles" and "code is verifiable":

1. **Naive recurrent form** (eq. GDN unrolled token-by-token). 50 lines of code. `naive_recurrent_gated_delta_rule` in `references/fla/naive.py` is the spec. Compare token-by-token outputs against the FLA reference at `chunk_size = 1` (which collapses chunk to recurrent).
2. **Block-level wrapper** (in_proj_qkv/z/b/a, conv1d, sigmoid β, softplus·exp α parameterization, RMSNormGated, out_proj). Verify forward parity against `Qwen3NextGatedDeltaNet.forward` from `references/transformers/qwen3_next_gdn.py` on a fixed seed. This catches projection-shape and ordering bugs cheaply.
3. **Naive chunkwise form** (`naive_chunk_gated_delta_rule`). 70 lines. Verify it matches the recurrent form for any sequence length and any chunk size that divides it.
4. **WY triangular solve**. The for-loop on lines 131–133 of `naive.py` is the only nontrivial control flow. Compute `T = [I + strictLower(diag(β)·(Γ ⊙ K K^T))]^{-1} · diag(β)` and check it produces the same `attn @ v` as the for-loop.
5. **Vectorize the chunk**. Replace the per-token Python loop in eq. GDN-S with the matrix form (eq. WY-3, with decay factors threaded through). At this point you have a correct chunkwise reference that should match a tensor-core-friendly Triton kernel modulo numerics.
6. **Cache-aware decode path**. Plumb the `S` and conv state through a per-layer cache object. Verify that decoding `T` tokens one at a time produces the same logits as prefilling them in one shot.

Skip the Triton kernel entirely for a verification implementation. The chunkwise PyTorch reference is the algorithmic specification; the Triton kernel is just a faster matmul-and-mask-friendly translation.

## 18. Common Gotchas

- **Convention for `S`'s shape.** Some references write `S ∈ R^{d_v × d_k}` and read with `o = S q`; others write `S ∈ R^{d_k × d_v}` and read with `o = S^T q`. The actual rank-1 update is the same up to a transpose, but every kernel comment must match its convention. FLA uses `S ∈ R^{d_k × d_v}` (read `o = S^T q`); Qwen3-Next reference uses `S ∈ R^{d_v × d_k}` (read `o = S q`). Translating between them requires transposing `S` only — *not* swapping `k` and `v`.
- **Chunk-relative vs. global cumulative decay.** The `decay` tensor inside the chunk loop is *chunk-relative*: it's `cumsum` over the chunk axis, not over the full sequence. The state-carry on line 152 of `naive.py` multiplies `S` by `exp(decay[:, :, i, -1])` (the chunk's total log-decay), which threads the global accumulation through. Mixing chunk-local and global decays is a classic bug.
- **`β · k_t k_t^T` vs. `(β k_t) k_t^T` vs. `k_t (β k_t)^T`.** The rank-1 perturbation `β k k^T` is symmetric in `k` only if `β` is a scalar; if `β` is per-head you must keep the broadcast right. FLA premultiplies `k` by `β` once (`k_beta = k * beta[..., None]`) and `v` by `β` once (`v = v * beta[..., None]`) at the top of the chunk function (lines 117–118 of `naive.py`) and never multiplies by `β` again.
- **Causal masking with strict-vs-non-strict upper triangular.** Two different masks appear on lines 122 and 144 of `naive.py`: the first (`diagonal=0`) is for the WY triangular solve (the diagonal is masked because we want only entries strictly *below* the diagonal in the inversion), the second (`diagonal=1`) is for the output computation (the diagonal is *un*-masked because position `r` reads from itself plus all earlier positions). Conflating them produces silent off-by-one errors that are very hard to debug.
- **L2-norm before vs. after RoPE.** GDN does not use RoPE on Q/K (it doesn't need positional encoding because the recurrence is implicitly causal and the gate `α_t` provides time-decay positional information). Standard transformer attention applies RoPE to Q/K then sometimes Q/K-RMSNorm; GDN applies L2-norm and that's it.
- **Sigmoid on `b`, softplus-exp on `a`.** Two separate gates with two separate parameterizations. The FLA layer uses `a_proj` for the SSM gate and `b_proj` for the delta-rule gate; the Qwen3-Next reference (and on-disk weight names) reverses this by calling them `in_proj_a` and `in_proj_b` but they correspond to the same scalars (`a` → `g`/decay, `b` → `β`/write-strength). The on-disk Qwen3.6 names match the Qwen3-Next reference: `linear_attn.in_proj_a.weight` produces `α`, `linear_attn.in_proj_b.weight` produces `β`.

## 19. References

### Primary

- **Songlin Yang, Jan Kautz, Ali Hatamizadeh.** *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. arXiv 2412.06464. The canonical equation (eq. 10 in §3.1), the chunkwise gated WY decomposition (§3.3), and the online-learning interpretation (§3.1 Table 1) all come from this paper. PDF in `references/paper/gated_delta_networks_2412.06464.pdf`.
- **Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim.** *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. arXiv 2406.06484. The chunkwise WY-form for the *un-gated* delta rule. PDF in `references/related/parallelizing_deltanet_2406.06484.pdf`.
- **Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber.** *Linear Transformers are Secretly Fast Weight Programmers.* ICML 2021. arXiv 2102.11174. The original delta rule for linear attention. PDF in `references/related/linear_transformers_fast_weight_2102.11174.pdf`.
- **Tri Dao, Albert Gu.** *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. arXiv 2405.21060. The Mamba2 / SSD construction; the source of the data-dependent scalar decay `α_t`. PDF in `references/related/mamba2_ssd_2405.21060.pdf`.

### Implementations

- **FLA library.** `https://github.com/fla-org/flash-linear-attention`. The canonical reference implementation. Files of interest:
  - `fla/ops/gated_delta_rule/naive.py` (recurrent + chunkwise reference, ~160 lines, in `references/fla/naive.py`).
  - `fla/ops/gated_delta_rule/chunk.py`, `chunk_fwd.py` (production Triton chunkwise kernel).
  - `fla/ops/gated_delta_rule/fused_recurrent.py` (production Triton decode kernel).
  - `fla/ops/gated_delta_rule/wy_fast.py` (the WY triangular solve as a Triton kernel).
  - `fla/ops/gated_delta_rule/gate.py` (g/α parameterization).
  - `fla/layers/gated_deltanet.py` (block-level layer wrapping the recurrence).
- **Transformers (HuggingFace).** `transformers/models/qwen3_next/modeling_qwen3_next.py` (`Qwen3NextGatedDeltaNet`, `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `Qwen3NextRMSNormGated`). The eager fallback mirrors FLA's naive ops 1:1; the fast path imports `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` from FLA. Copied to `references/transformers/qwen3_next_gdn.py`.
- **vLLM.** `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (`GatedDeltaNetAttention`). Production-quality wrapper with paged conv-state cache, FlashAttention-style varlen support, and TP plumbing. Copied to `references/vllm/gdn_linear_attn.py`.

### Adjacent / Extension Work

- **Bo Liu et al.** *Longhorn: State Space Models are Amortized Online Learners.* arXiv 2407.14207. The implicit-online-learning view that the GDN paper extends.
- **Yu Sun et al.** *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv 2407.04620. The TTT framework that interprets the state matrix as a fast weight matrix being trained at test time.
- **Riccardo Grazzi et al.** *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* arXiv 2411.12537. The motivation for `allow_neg_eigval=True` (β ∈ (0, 2) instead of (0, 1)).
- **Julien Siems et al.** *DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders.* arXiv 2502.10297. Multiple Householder transitions per token, extending GDN's rank-1 transition to rank-`k`.
- **Christian H. Bischof, Charles Van Loan.** *The WY representation for products of Householder matrices.* SIAM 1985. The original WY decomposition.
- **Thierry Joffrain et al.** *Accumulating Householder transformations, revisited.* ACM TOMS 2006. The UT trick that gives the closed-form `[I + strictLower(...)]^{-1} · diag(β)`.
