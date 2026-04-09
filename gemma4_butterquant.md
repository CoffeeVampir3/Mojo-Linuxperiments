# Gemma 4 ButterQuant — Design Specification

## AMX-Native Int8 Inference for Gemma 4 26B-A4B via Hadamard-Rotated Quantization

Target: `experimental3/` module.
Prerequisite reading: `butterquant.md` (scheme), `gemma_4_26b_4E_design.md` (model).

---

## 0. Core Architectural Insight

ButterQuant's original framing presents the Hadamard transform as a
quantization aid — spreading outlier energy for better int8 round-trip
fidelity. The `experimental2/` implementation reveals the actual role:
the FWHT is a **structural orthogonal rotation**, the same kind of
operation as RoPE. In the attention path, RoPE and FWHT are fused into
a single register-resident pipeline:

```
Q_bf16 → f32 → RoPE → FWHT → quantize → i8     (prep_q_row)
K_bf16 → f32 → RoPE → FWHT → quantize → cache   (write_k_head)
```

The FWHT does not "fix" quantization — it puts Q and K into a domain
where int8 dot products produce correct attention scores (Parseval). The
dynamic absmax scales handle quantization quality. The Hadamard's
energy-spreading property is a bonus, not the mechanism.

In the GEMM path, the FWHT serves the algebraic role from `butterquant.md`
§2.1: single-sided weight rotation on the contraction dimension cancels
with the activation FWHT, returning the GEMV output to the original domain.
Again, the rotation enables the algebra — dynamic scales handle quality.

This distinction matters for Gemma 4 because:

1. Per-head QK-norms already constrain Q/K magnitudes, making the attention
   path cleaner than a standard transformer (the FWHT's energy-spreading
   role is even less important).
2. The MoE expert GEMV path is where the real win lives — 128 experts per
   layer dominate memory bandwidth, and int8 halves it.
3. Non-power-of-2 intermediate dimensions (2112, 704) force smaller FWHT
   blocks in the GEMM path, but this only affects the energy-spreading
   bonus, not the algebraic correctness.


---


## 1. Dimension Inventory and FWHT Block Sizes

Every FWHT block must be a power of 2 that divides the target dimension.

### 1.1 Attention Dimensions (Powers of 2 — Clean)

| Dimension | Value | FWHT Block | Blocks | Context |
|-----------|-------|------------|--------|---------|
| HEAD_DIM_SLIDING | 256 | 256 | 1 | Per-head Q/K/V, sliding layers |
| HEAD_DIM_FULL | 512 | 512 | 1 | Per-head Q/K/V, full layers |

One FWHT block per head. Entire head vector fits in registers. This is
the same setup as `experimental2/` — the attention path transfers directly.

### 1.2 GEMM Contraction Dimensions

| Dimension | Value | Factorization | FWHT Block | Blocks | Context |
|-----------|-------|---------------|------------|--------|---------|
| HIDDEN | 2816 | 256 x 11 | 256 | 11 | QKV, gate/up, expert gate_up, router, O proj input |
| Q_DIM_SLIDING | 4096 | 256 x 16 | 256 | 16 | O proj (sliding) |
| Q_DIM_FULL | 8192 | 256 x 32 | 256 | 32 | O proj (full) |
| INTERMEDIATE | 2112 | 64 x 33 | 64 | 33 | Dense down proj |
| MOE_INTERMEDIATE | 704 | 64 x 11 | 64 | 11 | Expert down proj |
| GATE_UP_DIM | 1408 | 64 x 22 | 64 | 22 | Post-nonlinearity → down |

Two block sizes in the system: **256** for hidden-dim paths, **64** for
intermediate-dim paths. All attention head dims are single-block (256 or
512). The FWHT infrastructure (`fwht_block[block]`, `fwht_apply`) is
already parameterized on block size.

The block=64 paths serve the dense and expert down projections. With 6
butterfly stages (vs 8 for 256), the per-block FWHT is cheaper. The
reduced energy spreading is compensated by dynamic absmax scales at
every quantization point — the `butterquant.md` measurements show dynamic
scales dominate quality regardless of block size.


---


## 2. Gamma Absorption Map

Offline absorption: fold RMSNorm learnable scale into downstream weight
matrices on the contraction dimension. Runtime RMSNorm reduces to
division by RMS (no gamma multiply).

### 2.1 Absorbable Norms (3 per layer)

| Norm | Absorbed Into | Contraction Dim |
|------|--------------|-----------------|
| `input_layernorm.γ` | Q, K, V projections (sliding) or Q, K (full) | HIDDEN=2816 |
| `pre_feedforward_layernorm.γ` | gate_proj, up_proj (dense MLP) | HIDDEN=2816 |
| `pre_feedforward_layernorm_2.γ` | All 128 expert gate_up_proj | HIDDEN=2816 |

For full attention layers (no V projection), `input_layernorm.γ` absorbs
into Q and K only. This is correct because V is derived from K's raw
projection output — the absorption acts on the contraction dimension
(HIDDEN), not the output dimension, so K=V sharing is unaffected.

### 2.2 Per-Head Norm Gamma Absorption (Q and K)

The per-head norms (`q_norm`, `k_norm`) apply on the **output** side of
the projection, with a per-head-dim learnable scale. These can be
absorbed into the projection weight rows on a per-head-block basis:

```
For head h in [0, num_heads):
  Q_proj'[h*head_dim : (h+1)*head_dim, :] *= q_norm.weight[:]
```

This is row-wise scaling on blocks of the output dimension. The
contraction dimension (HIDDEN) is unchanged, so FWHT weight rotation
is unaffected. At runtime, the per-head norm reduces to scale-free
RMS-divide only (no gamma multiply).

This absorption applies to Q and K norms for both sliding and full
layers. V norm has no learnable scale — it is always scale-free
RMS-divide at runtime.

### 2.3 Router Scale Absorption

The router uses: `rms_norm_no_scale(x) * router.scale * (1/sqrt(2816))`.
The baked `router.scale * inv_sqrt_hidden` functions as a gamma for
absorption into `router.proj.weight`. After absorption, the router
GEMV input is just `x / rms(x)`.

### 2.4 Non-Absorbable Norms (4 per layer)

| Norm | Why Not Absorbable |
|------|-------------------|
| `post_attention_layernorm.γ` | Followed by residual addition |
| `post_feedforward_layernorm_1.γ` | Followed by dense+MoE combine |
| `post_feedforward_layernorm_2.γ` | Followed by dense+MoE combine |
| `post_feedforward_layernorm.γ` | Followed by residual addition |

These remain bf16/f32 runtime operations. They are element-wise on
(1, 2816) vectors during decode — negligible cost.


---


## 3. Offline Weight Quantization

Extends `butterquant.md` §IV for the Gemma 4 weight inventory.

### 3.1 Per-Weight Quantization Recipe

For each weight matrix W with preceding absorbable norm gamma γ:

1. **Gamma absorption:** `W'[n, k] = W[n, k] * γ[k]`
2. **Per-head norm absorption** (Q/K projections only):
   `W'[n, k] *= head_norm_gamma[n % head_dim]`
3. **FWHT rotation** on contraction dim K per row:
   `W_rot[n, :] = block_diag_FWHT(W'[n, :], block_size)`
4. **Per-row int8 quantization:**
   `s_w[n] = absmax(W_rot[n,:]) / 127`
   `W_i8[n, k] = clamp(round(W_rot[n,k] / s_w[n] * 127), -128, 127)`
5. **VNNI packing:** Rearrange i8 weight into 6D blocked layout for
   `vpdpbusd` / `tdpbusd`. Layout:
   `[N/N_BLK, K/K_BLK, N_BLK/N_STEP, K_BLK/K_STEP, N_STEP/16, K_STEP/4, 16, 4]`
   Uses `pack_vnni()` from `kernels/vnni.mojo`. Byte count is identical to
   row-major i8 (N×K bytes), but layout is transposed within 16×64 tiles.

Each quantized weight matrix produces **three** runtime buffers:

| Buffer | Type | Shape | Purpose |
|--------|------|-------|---------|
| `weight` | i8 (VNNI-packed) | N × K bytes | Packed for vpdpbusd tile loads |
| `weight_scale` | f32 | N × 1 | Per-row absmax / 127, for GEMV dequant |
| `weight_colsum` | f32 | N × 1 | `sum_k weight_i8[n,k]`, for VNNI bias correction |

The GEMV dequant formula (from `butterquant.md` §4.1):
```
y[m, n] = (raw[m,n] - 128 * colsum[n]) * (S_act[m] / 127) * scale[n]
```

The `-128 * colsum` term corrects for the i8→u8 XOR 0x80 bias in the
`vpdpbusd` unsigned×signed dot product.

**Column sums** are computed at load time from the packed i8 weights
(not stored in safetensors). The offline quantizer writes only the i8
weights and f32 row scales. See `init_column_sums` in
`smollm2_butterquant_tp.mojo` for the pattern.

### 3.2 Weight Fusion for Decode

After gamma absorption, multiple weight matrices sharing the same input
activation can be **fused** (vertically stacked) into a single larger
GEMV. This eliminates redundant activation loads and dispatch overhead.

**Shared activation insight:** After gamma absorption, the dense MLP,
router, and expert gate_up paths all normalize the same input:
```
dense:   rms_norm(x_main, pre_ffn_norm.γ)     → absorb γ → x_main / rms(x_main)
router:  rms_norm(x_main, router.scale*inv√h)  → absorb   → x_main / rms(x_main)
expert:  rms_norm(x_main, pre_ffn_norm_2.γ)   → absorb γ → x_main / rms(x_main)
```
One `rmsnorm_fwht_quantize` produces a single i8 activation for all
three downstream GEMV categories.

**Fused weight groups** (placed contiguously in the weight layout):

| Fusion | Components | Fused Shape (N, K) |
|--------|-----------|-------------------|
| Sliding QKV | Q(4096) + K(2048) + V(2048) | (8192, 2816) |
| Full QK | Q(8192) + K(1024) | (9216, 2816) |
| Dense gate+up | gate(2112) + up(2112) | (4224, 2816) |

Scales and colsums for fused groups are contiguous too — the GEMV
reads them as a single `f32[fused_N]` array.

Experts are NOT fused with the dense/router weights because expert
selection depends on the router output. Expert gate_up weights are
already fused internally: `(1408, 2816) = gate(704) + up(704)`.

### 3.3 Weight Table

Each weight entry consists of the triplet: i8 VNNI-packed weight,
f32 row scale, f32 column sum (load-time computed).

| Weight | Shape | Block | Absorptions |
|--------|-------|-------|-------------|
| **Sliding Attention (25 layers) — fused QKV** ||||
| QKV_proj | (8192, 2816) | 256 | Q: input_ln.γ + q_norm.γ; K: input_ln.γ + k_norm.γ; V: input_ln.γ |
| O_proj | (2816, 4096) | 256 | None |
| **Full Attention (5 layers) — fused QK** ||||
| QK_proj | (9216, 2816) | 256 | Q: input_ln.γ + q_norm.γ; K: input_ln.γ + k_norm.γ |
| O_proj | (2816, 8192) | 256 | None |
| **Dense MLP (30 layers) — fused gate+up** ||||
| gate_up_proj | (4224, 2816) | 256 | pre_feedforward_layernorm.γ |
| down_proj | (2816, 2112) | 64 | None |
| **Router (30 layers)** ||||
| router.proj | (128, 2816) | 256 | router.scale * inv_sqrt_hidden |
| **MoE Experts (30 layers x 128 experts)** ||||
| expert gate_up | (1408, 2816) | 256 | pre_feedforward_layernorm_2.γ |
| expert down | (2816, 704) | 64 | None |

### 3.3 Passthrough Weights (Not Quantized)

| Weight | Shape | Reason |
|--------|-------|--------|
| embed_tokens | (262144, 2816) | LM head uses bf16 matmul (no int8 logits) |
| model.norm | (2816,) | Final norm, scalar ops |
| post_attention_layernorm | (2816,) per layer | Runtime norm, not absorbed |
| post_feedforward_layernorm_1/2 | (2816,) per layer | Runtime norm |
| post_feedforward_layernorm | (2816,) per layer | Runtime norm |
| router.per_expert_scale | (128,) per layer | Applied after softmax, not a GEMM input |
| layer_scalar | (1,) per layer | Scalar multiply |

### 3.4 V Scale Derivation

**Sliding layers** (V_proj exists):
```
S_V = ||W'_V||_F / sqrt(KV_DIM_SLIDING) * C(HEAD_DIM_SLIDING)
    = ||W'_V||_F / sqrt(2048) * C(256)
```

**Full layers** (K=V shared, V has scale-free norm):
Scale-free RMSNorm makes every head vector have `rms = 1`, so
`||V_head|| = sqrt(head_dim)` for every head, every position.
```
S_V_full = sqrt(HEAD_DIM_FULL) * C(HEAD_DIM_FULL)
         = sqrt(512) * C(512)
```
This is independent of weight norms — it depends only on the head
dimension and FWHT concentration constant. Compute C(512) via Monte
Carlo at compile time.

### 3.5 O Projection and Attention Output Scale

The attention output (post V-agg, post normalize) is quantized with S_V
for the O projection GEMV. Same as `butterquant.md` §3.2: `S_attn_out = S_V`.

For the O projection, the activation scale used in GEMV dequant is:
```
act_scale = S_V / 127      (fixed, per layer)
```


---


## 4. Attention Path — Register-Fused Pipeline

### 4.1 Sliding Attention (25 layers)

Geometry: 16 Q heads, 8 KV heads, head_dim=256, GQA ratio=2, window=1024.
Attention scale = 1.0 (QK-norms absorb scaling).
RoPE: theta=10000, full 256-dim rotation, half-dim table of 128 entries.

#### Q Preparation (per head)

```
Input: Q_bf16[head_dim] from int8 GEMV output (gamma-absorbed, per-head-norm-absorbed)
Pipeline:
  1. Load bf16 → f32 registers
  2. RMS-divide: r[i] /= rms(r)       ← residual per-head norm (gamma absorbed into weight)
  3. RoPE: rotate_half with cos/sin table
  4. FWHT_256: in-register butterfly stages + 1/sqrt(256) normalize
  5. Dynamic absmax → quantize → i8
Output: qi[head_dim] i8, qi_bias (128 * sum(qi)), q_scale (absmax)
```

This extends `experimental2/helpers.mojo:prep_q_row` with step 2 (RMS-divide).
The gamma from `q_norm.weight` is already absorbed into Q_proj rows, so
runtime only computes the scale-free normalization.

#### K Cache Write (per head)

```
Input: K_bf16[head_dim] from int8 GEMV output (gamma-absorbed, per-head-norm-absorbed)
Pipeline:
  1. Load bf16 → f32
  2. RMS-divide: same residual per-head norm
  3. RoPE: rotate_half with cos/sin table
  4. FWHT_256: in-register butterflies
  5. Dynamic absmax → quantize → i8
  6. XOR 0x80 → u8, scatter into VNNI tile format
Output: u8 in K cache, k_scale (absmax) stored in cache
```

Extends `experimental2/kernels/rope_and_kv_cache_write.mojo:write_k_head`
with step 2.

#### V Cache Write (per head)

```
Input: V_bf16[head_dim] from int8 GEMV output (input_layernorm gamma absorbed, no per-head gamma)
Pipeline:
  1. Load bf16 → f32
  2. Scale-free RMS-divide: v[i] /= rms(v)    ← V has no learnable scale
  3. FWHT_256: in-register butterflies
  4. Fixed-scale quantize (S_V) → i8
  5. Store row-major in V cache
Output: i8 in V cache
```

Extends `experimental2/kernels/rope_and_kv_cache_write.mojo:write_v_head`
with step 2.

#### Scoring and V-Aggregation

Identical to `experimental2/attn_amx_decode.mojo` with one change:

```
Score dequant: (raw - b_q) * S_Q[h] / 127^2 * S_K[g, t]
```

The `1/sqrt(d_k)` factor from `experimental2` is removed. Gemma 4 uses
attention scale = 1.0 because QK-norms handle scaling. The `inv_sqrt_hd`
field in `AttnCtx` becomes 1.0 (or is removed entirely, folding into the
q_partial computation).

V-agg, online softmax, and decode merge are unchanged from `experimental2`.

### 4.2 Full/Global Attention (5 layers)

Geometry: 16 Q heads, 2 KV heads, head_dim=512, GQA ratio=8, no window.
RoPE: theta=1000000, partial rotation — 128 of 512 dims rotated.
Compact cos/sin table: 64 entries (half of 128 rotated dims).

#### Q Preparation (per head) — Partial RoPE Variant

```
Input: Q_bf16[512] from int8 GEMV
Pipeline:
  1. Load bf16 → f32 registers (32 SIMD regs for 512 f32 values at width=16)
  2. RMS-divide: r[i] /= rms(r)
  3. Partial RoPE: rotate only dims [0..63] and [256..319] using 64-entry table
     Dims [64..255] and [320..511] pass through unchanged.
  4. FWHT_512: in-register butterfly stages (9 stages) + 1/sqrt(512) normalize
  5. Dynamic absmax → quantize → i8
Output: qi[512] i8, qi_bias, q_scale
```

The partial RoPE differs from sliding: only 128 of 512 dims are rotated,
using a compact 64-entry cos/sin table. The FWHT operates on the full
512-dim vector regardless — unrotated dims still participate in the
Hadamard transform. Parseval holds for the full composition.

New function: `prep_q_row_partial_rope[head_dim, rope_dims]`.

#### K Cache Write (per head) — Partial RoPE + K=V Fork

```
Input: K_bf16[512] from int8 GEMV (one projection, used for both K and V)
Pipeline for K:
  1-4. Same as Q partial RoPE pipeline (RMS-divide with k_norm gamma absorbed, partial RoPE, FWHT_512)
  5. Dynamic absmax → i8 → u8 XOR → VNNI cache

Pipeline for V (from same GEMV output):
  1. Load bf16 → f32 (same source as K)
  2. Scale-free RMS-divide (no gamma — V norm has no learnable scale)
  3. FWHT_512 (no RoPE for V)
  4. Fixed-scale quantize (S_V_full) → i8
  5. Store row-major in V cache
```

K and V are prepared from the same GEMV output but through divergent
pipelines (K gets gamma+RoPE, V gets scale-free norm only). The K
projection's input_layernorm gamma is absorbed into the weight's
contraction dimension, so the GEMV output is already gamma-absorbed
on the input side — the per-head K norm gamma is absorbed into the
output-side rows. Both K and V paths receive the correct intermediate
representation.

#### Scoring

Same as sliding but with head_dim=512 and no window mask:

```
Score dequant: (raw - b_q) * S_Q[h] / 127^2 * S_K[g, t]
```

Full causal mask: position q attends to all k <= q. The V-agg
accumulator uses stack arrays (not register arrays) because 512 f32
values exceed comfortable register residence — same strategy as
`experimental_gemma/attention.mojo:global_attention_kernel`.


---


## 5. GEMM Path — Projection GEMVs

### 5.1 Activation Quantization (rmsnorm_fwht_quantize)

Three distinct quantization points produce int8 activations for
downstream GEMVs:

| Quantization Point | Input Norm | Block | Output Feeds |
|-------------------|-----------|-------|-------------|
| Pre-attention | input_layernorm (γ absorbed) | 256 | QKV GEMVs |
| Pre-dense-MLP | pre_feedforward_layernorm (γ absorbed) | 256 | gate, up GEMVs |
| Pre-expert-MLP | pre_feedforward_layernorm_2 (γ absorbed) | 256 | 128 expert gate_up GEMVs |

Additionally, one quantization point for the router:

| Quantization Point | Input Norm | Block | Output Feeds |
|-------------------|-----------|-------|-------------|
| Pre-router | router.scale * inv_sqrt_h (absorbed) | 256 | router.proj GEMV |

All four use the same kernel: `rmsnorm_fwht_quantize[cols=2816, block=256]`.
Gamma is absorbed into downstream weights; runtime norm is scale-free
RMS-divide → block-diagonal FWHT → dynamic absmax → i8.

Reuse `experimental2/kernels/rmsnorm_fwht_quantize.mojo` directly.

### 5.2 Post-Nonlinearity Quantization

Two paths, same kernel: `gelu_tanh_fwht_quantize[cols, block=64]`.

**Dense MLP:** cols=2112, 33 FWHT blocks.
**Expert MLP:** cols=704, 11 FWHT blocks.

#### 5.2.1 Per-Block Scales

The post-nonlinearity quantization uses **per-FWHT-block dynamic scales**
rather than the per-row scales used elsewhere. This is necessary because
the block-diagonal FWHT (block=64) produces independently-distributed
64-element blocks. A single per-row absmax is dominated by the worst
block, wasting int8 range for the others.

Measured: per-block scales have a 1.8–2.2x range across blocks within
one row. Per-block quantization reduces NRMSE from 1.16% to 0.87%.

The downstream GEMV handles per-block activation scales during
K-iteration, applying a different scale per K_STEP=64 block. This
aligns naturally with the VNNI K-stepping.

#### 5.2.2 DC Component Correction

The FWHT's element 0 (the DC component) is a systematic outlier in the
post-nonlinearity quantization. Measured across 1100 FWHT blocks:
**element 0 is the block absmax 98% of the time**, at 1.64x the
second-largest element.

This is structural, not statistical. GELU-tanh maps negative inputs
to ~0 and positive inputs to ~x, producing a positive-biased signal.
The DC component `y[0] = (1/√64) × Σ(block)` captures this bias and
dominates the absmax, wasting int8 range for the 63 AC components.

The fix is a compile-time **DC scale factor** (DC_SCALE = 0.5):

1. **Activation quantize** (in-register): after FWHT, multiply element 0
   of each block by DC_SCALE before computing absmax and quantizing.
   One multiply per 64 elements. DC_SCALE = 0.5 is a bit shift.

2. **Weight quantize** (offline): after FWHT-rotating weight rows, scale
   columns at positions [0, 64, 128, ...] by 1/DC_SCALE = 2.0. These
   are the weight elements that contract with the DC position.

3. **GEMV** (runtime): unchanged. The factors cancel exactly:
   `W_rot'[n,0] × x_rot'[0] = (W/DC_SCALE) × (x×DC_SCALE) = W×x`.

Measured NRMSE improvement (averaged over 50 trials, random bf16 data):

| Approach | NRMSE (cols=704) | NRMSE (cols=2112) |
|----------|-----------------|------------------|
| Per-row absmax (original) | 0.882% | 0.886% |
| Per-block absmax | 0.671% | 0.693% |
| Per-block + DC_SCALE=0.5 | **0.532%** | **0.534%** |

The DC correction reduces total round-trip NRMSE by 40% vs the original
scheme, at zero GEMV runtime cost and one bit-shift per 64 elements
during activation quantize.

**Why DC_SCALE = 0.5:** Sweep over [0.4, 0.5, 0.55, 0.6, ..., 1.0]
shows NRMSE flattens at DC_SCALE ≈ 0.5. Below 0.5, the AC components
become the new absmax and further DC reduction gives no benefit. 0.5 is
also a power-of-two reciprocal (bit shift, no floating divide).

**Why this only applies here:** The pre-attention and pre-MLP activation
quantization uses block=256 on HIDDEN=2816. With larger blocks, the
Hadamard transform spreads energy across more dimensions, and the DC
component is less dominant relative to the AC. The post-nonlinearity
point is unique because (a) block=64 concentrates DC energy more and
(b) GELU-tanh produces stronger positive bias than RMSNorm'd activations.

### 5.3 Int8 GEMV Instantiations

With weight fusion (§3.2), the actual GEMV dispatches during decode:

| GEMV | Output N | Contraction K | Block | Act Scale | Notes |
|------|----------|--------------|-------|-----------|-------|
| Sliding QKV (fused) | 8192 | 2816 | 256 | Dynamic S_act | Split: Q[0:4096], K[4096:6144], V[6144:8192] |
| Sliding O proj | 2816 | 4096 | 256 | Fixed S_V | |
| Full QK (fused) | 9216 | 2816 | 256 | Dynamic S_act | Split: Q[0:8192], K[8192:9216]; V from K output |
| Full O proj | 2816 | 8192 | 256 | Fixed S_V_full | |
| Dense gate+up (fused) | 4224 | 2816 | 256 | Dynamic S_act | Split: gate[0:2112], up[2112:4224] |
| Dense down | 2816 | 2112 | 64 | Per-block S_post (33 scales) | DC-corrected (§5.2.2) |
| Router proj | 128 | 2816 | 256 | Dynamic S_act | Shares activation with gate+up and experts |
| Expert gate_up | 1408 | 2816 | 256 | Dynamic S_act | Same activation as router and dense |
| Expert down | 2816 | 704 | 64 | Per-block S_post (11 scales) | DC-corrected (§5.2.2) |

Each GEMV reads: VNNI-packed i8 weight, f32 row scale, f32 column sum.
For down projections, the activation has per-K-block scales instead of
per-row. The GEMV K-iteration applies each block's scale independently,
dequanting to f32 before cross-block accumulation. Weight columns at
DC positions (0, 64, 128, ...) are pre-scaled by 1/DC_SCALE = 2.0.

Standard GEMV dequant (per-row activation scale):
```
y[n] = (raw[n] - 128 * colsum[n]) * act_dequant * weight_scale[n]
```

Per-block GEMV dequant (down projections):
```
y[n] = Σ_b (raw_b[n] - 128 * colsum_b[n]) * (S_post[b] / 127) * weight_scale[n]
```

### 5.4 Fused Expert Kernel

For MoE, the hot path per active expert per token is:

```
activation_i8[2816] (from rmsnorm_fwht_quantize, shared across experts)
  → expert gate_up GEMV [1408, 2816] → f32 (stack buffer, not materialized to memory)
  → split gate[704] / up[704]
  → gelu_tanh(gate) * up → f32[704]
  → FWHT_64 (11 blocks) → dynamic absmax → i8[704]
  → expert down GEMV [2816, 704] → bf16
  → * routing_weight → accumulate into output
```

This is the expert analog of `fused_gu_silu_worker` from
`experimental2/kernels/int8_gemv.mojo`. New variant:
`fused_expert_gelu_tanh_worker[GATE_UP=1408, HIDDEN=2816, INTERMEDIATE=704, FWHT_BLK=64]`.

The gate_up output is 1408 = 2 x 704, split into gate[704] and up[704].
Both GEMVs and the fused nonlinearity happen in stack f32 buffers — no
memory traffic for intermediate activations.

Dispatch: `gemma4_moe_dispatch` creates top_k (8) jobs, each running
the fused expert kernel. BurstPool parallelizes across workers.


---


## 6. Decoder Layer Pipeline (Decode, seq_len=1)

Columns: step, operation, scale, domain, dtype.
`act_i8` denotes the pre-quantized activation buffer reused across GEMVs
sharing the same input norm.

```
                                                       Scale            Domain     Dtype
ATTENTION BLOCK
 1. x_n = x / rms(x)                                  —                Original   f32
 2. act_i8 = Q_S(FWHT_256(x_n))                       Dynamic S_act    Hadamard   i8
 3. Q = int8_gemv(act_i8, W'_Q)                        S_act * s_w      Original   bf16
 4. K = int8_gemv(act_i8, W'_K)                        S_act * s_w      Original   bf16
 5. V = int8_gemv(act_i8, W'_V)  [or K output if full] S_act * s_w     Original   bf16
 6. Q: /rms per head → RoPE → FWHT → Q_S_Q → i8       Dynamic S_Q      Hadamard   i8
 7. K: /rms per head → RoPE → FWHT → Q_S_K → cache     Dynamic S_K      Hadamard   u8
 8. V: /rms per head → FWHT → Q_S_V → cache            Fixed S_V        Hadamard   i8
 9. Score: int8 dot → (raw-b_q)*S_Q/127^2*S_K          Per-position     —          f32
10. Softmax → u8 weights                               255              —          u8
11. V-agg: u8 x i8_vnni → i32                          Deferred         Hadamard   i32
12. Normalize: * sigma_v / ell                          Fixed S_V        Hadamard   f32
13. Quantize attn output: Q_S_V(output)                 Fixed S_V        Hadamard   i8
14. O GEMV: int8_gemv(attn_i8, W'_O)                   S_V * s_w        Original   bf16
─── post_attention_layernorm (bf16, γ NOT absorbed)     —                Original   bf16
─── residual: x = x_main + normed_attn_out             —                Original   f32

DENSE MLP PATH
15. r_n = r / rms(r)                                   —                Original   f32
16. act_i8 = Q_S(FWHT_256(r_n))                        Dynamic S_act    Hadamard   i8
17. gate = int8_gemv(act_i8, W'_gate)                  S_act * s_w      Original   bf16→f32
18. up   = int8_gemv(act_i8, W'_up)                    S_act * s_w      Original   bf16→f32
19. phi = gelu_tanh(gate) * up                          —                Original   f32
20. phi_i8 = Q_S(FWHT_64(phi))                         Dynamic S_post   Hadamard   i8
21. dense_out = int8_gemv(phi_i8, W'_down)              S_post * s_w     Original   bf16
─── dense_normed = rmsnorm(dense_out, POST_FFN_NORM_1)  —               Original   bf16

MOE PATH (parallel with dense)
22. router_n = x_main / rms(x_main)                    —                Original   f32
23. router_i8 = Q_S(FWHT_256(router_n))                Dynamic S_act    Hadamard   i8
24. logits = int8_gemv(router_i8, W'_router)            S_act * s_w     Original   bf16
25. routing = softmax → topk → renorm → per_expert_scale                           f32
26. expert_n = x_main / rms_2(x_main)                  —                Original   f32
27. expert_i8 = Q_S(FWHT_256(expert_n))                Dynamic S_act    Hadamard   i8
    For each active expert e in top-8:
28.   gate_up = int8_gemv(expert_i8, W'_expert_gu[e])   S_act * s_w     Original   f32 (stack)
29.   split gate[704] / up[704]
30.   phi_e = gelu_tanh(gate) * up                      —                Original   f32
31.   phi_e_i8 = Q_S(FWHT_64(phi_e))                   Dynamic S_post   Hadamard   i8
32.   expert_out = int8_gemv(phi_e_i8, W'_expert_d[e])  S_post * s_w    Original   bf16
33.   output += expert_out * routing_weight[e]
─── moe_normed = rmsnorm(moe_out, POST_FFN_NORM_2)      —               Original   bf16

COMBINE
34. combined = dense_normed + moe_normed                —                Original   bf16
35. combined = rmsnorm(combined, POST_FFN_NORM)         —                Original   bf16
36. x = (x_main + combined) * layer_scalar              —                Original   f32
```


---


## 7. KV Cache Format

Reuse `experimental2/kv_cache.mojo:KVCache` with two instantiations:

### 7.1 Sliding Cache

```
KVCache[max_seq=1024, head_dim=256, num_kv_heads=8, num_q_heads=16]
```

Layout per layer: `[K VNNI u8 | V i8 row-major | K scales f32 | Q scales f32]`

Eviction: circular buffer, oldest entries overwritten at position % 1024.
Only 1024 positions retained (sliding window).

### 7.2 Full Cache

```
KVCache[max_seq=MAX_SEQ_LEN, head_dim=512, num_kv_heads=2, num_q_heads=16]
```

Same layout, no eviction. Full history retained.

### 7.3 Scale Storage

K scales: `f32[num_kv_heads][max_seq]` — dynamic, written at K cache write.
Q scales: `f32[num_q_heads][max_seq]` — dynamic, written at Q prep.

V scale: `f32` per layer — fixed, computed from weight norms at load time.
Not stored in cache. Passed as parameter to attention and O-projection
kernels.


---


## 8. Module Structure — experimental3/

```
experimental3/
  __init__.mojo                  — empty (direct imports)
  kv_cache.mojo                  — reexport or alias of experimental2/kv_cache.mojo
  kernel_profile.mojo            — reexport of experimental2/kernel_profile.mojo
  helpers.mojo                   — AttnCtx (Gemma 4 variant), prep_q_row variants,
                                   softmax_row, pack_v_tile_vnni, amx_gemm_1x3
  attn_sliding_decode.mojo       — AMX decode for sliding layers (head_dim=256, scale=1.0)
  attn_full_decode.mojo          — AMX decode for full layers (head_dim=512, scale=1.0)
  attn_sliding_prefill.mojo      — AMX prefill for sliding layers
  attn_full_prefill.mojo         — AMX prefill for full layers
  kernels/
    __init__.mojo                — empty
    quantize.mojo                — reexport of experimental2/kernels/quantize.mojo
    rmsnorm_fwht_quantize.mojo   — reexport of experimental2 (unchanged: cols=2816, block=256)
    gelu_tanh_fwht_quantize.mojo — NEW: gelu_tanh(gate)*up → FWHT → dynamic i8
    int8_gemv.mojo               — reexport of experimental2 (unchanged: parameterized [N, K])
    rope_and_kv_cache_write.mojo — NEW: per-head norm + RoPE + FWHT + cache write
                                   Variants for full/partial RoPE and K=V sharing
    float_gemv.mojo              — reexport of experimental2 (for LM head, bf16 matmul)
```

### 8.1 New Kernels

#### `gelu_tanh_fwht_quantize.mojo`

Clone of `experimental2/kernels/silu_fwht_quantize.mojo`, replacing:

```python
# SiLU: g * sigmoid(g) * u
var sig = 1.0 / (1.0 + exp(-g))
work[k] = g * sig * u

# GELU-tanh: 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715*g^3))) * u
var inner = 0.7978845608 * (g + 0.044715 * g * g * g)
var t = tanh(inner)    # via 2*sigmoid(2*inner) - 1
work[k] = 0.5 * g * (1.0 + t) * u
```

Parameterized as `gelu_tanh_fwht_quantize[cols, block]` with dispatch.

Also: fused variant `fused_gu_gelu_tanh_worker[GATE_ROWS, K, FWHT_BLK]`
in `int8_gemv.mojo` (or a Gemma4-specific file), following the pattern
of `fused_gu_silu_worker`.

#### `rope_and_kv_cache_write.mojo` (Gemma 4 variants)

**`write_k_head_normed[head_dim]`**: Adds per-head RMS-divide before RoPE.
```
bf16 → f32 → /rms → RoPE → FWHT → dynamic quantize → u8 cache
```

**`write_k_head_normed_partial_rope[head_dim, rope_dims]`**: For full
attention layers. Partial rotation on first `rope_dims` of `head_dim`.
```
bf16 → f32 → /rms → partial RoPE (128 of 512) → FWHT_512 → dynamic quantize → u8 cache
```

**`write_v_head_normed[head_dim]`**: Adds scale-free RMS-divide before FWHT.
```
bf16 → f32 → /rms (no gamma) → FWHT → fixed quantize → i8 cache
```

#### `helpers.mojo` — Modified AttnCtx

```mojo
struct Gemma4AttnCtx:
    var q: UnsafePointer[BFloat16, MutAnyOrigin]
    var q_stride: Int
    var cos: UnsafePointer[Float32, MutAnyOrigin]
    var sin: UnsafePointer[Float32, MutAnyOrigin]
    var k_base: UnsafePointer[UInt8, MutAnyOrigin]
    var v_base: UnsafePointer[Int8, MutAnyOrigin]
    var qi_output: UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
    var qi_scale: Float32           # 127 / S_V (fixed)
    var k_scale_base: UnsafePointer[Float32, MutAnyOrigin]
    var q_scale_base: UnsafePointer[Float32, MutAnyOrigin]
    var max_seq: Int
    # No inv_sqrt_hd — Gemma 4 uses scale=1.0
```

**`prep_q_row_normed[head_dim]`**: `prep_q_row` + RMS-divide at step 2.

**`prep_q_row_normed_partial[head_dim, rope_dims]`**: Partial RoPE variant.

### 8.2 Reused From experimental2/ (No Modification)

- `kv_cache.mojo` — KVCache struct, write_k, write_v, write_k_scale
- `kernel_profile.mojo` — KernelProfile, ProfileAggregator, tap()
- `kernels/quantize.mojo` — absmax_quantize_i8, fixed_quantize_i8
- `kernels/rmsnorm_fwht_quantize.mojo` — fwht_block, fwht_apply, row kernel, dispatch
- `kernels/int8_gemv.mojo` — VNNI dot, gemv_row, int8_gemv dispatch
- `kernels/float_gemv.mojo` — bf16 GEMV for LM head
- `helpers.mojo` — softmax_row, pack_v_tile_vnni, amx_gemm_1x3, amx_gemm_2x2


---


## 9. Memory Budget

### 9.1 Weight Memory (Int8 vs BF16)

Per quantized weight: i8 data (N×K) + f32 row_scale (N×4) + f32 colsum (N×4).
The colsum overhead is 8 bytes per output row — negligible vs N×K weight bytes.

| Component | BF16 (bytes) | Int8 + scales + colsums | Ratio |
|-----------|-------------|------------------------|-------|
| Sliding attention (25 layers) | 1,735 MB | 873 MB | 0.50x |
| Full attention (5 layers) | 490 MB | 247 MB | 0.50x |
| Dense MLP (30 layers) | 1,020 MB | 515 MB | 0.50x |
| Router (30 layers) | 21 MB | 11 MB | 0.50x |
| **MoE experts (30 x 128)** | **43,670 MB** | **21,890 MB** | **0.50x** |
| Embed + norms (passthrough) | 1,406 MB | 1,406 MB | 1.00x |
| **Total** | **~48.3 GB** | **~24.9 GB** | **0.52x** |

Expert weights dominate: 90% of model parameters. The 2x reduction on
experts is the primary win. Weight layout byte counts are identical for
row-major i8 and VNNI-packed i8 (same N×K bytes, different arrangement).

### 9.2 KV Cache Memory

Sliding (25 layers): `25 * KVCache[1024, 256, 8, 16].TOTAL_BYTES`
  - K: 25 * 8 * 1024 * 256 = 50 MB (u8) + 25 * 8 * 1024 * 4 = 0.8 MB (K scales)
  - V: 25 * 8 * 1024 * 256 = 50 MB (i8)
  - Q scales: 25 * 16 * 1024 * 4 = 1.6 MB

Full (5 layers at max_seq=4096): `5 * KVCache[4096, 512, 2, 16].TOTAL_BYTES`
  - K: 5 * 2 * ~4096 * 512 = 20 MB (u8) + scales
  - V: 5 * 2 * 4096 * 512 = 20 MB (i8)

Total KV cache: ~143 MB (int8) vs ~286 MB (bf16). Another 2x.

### 9.3 Activation Scratch

Peak scratch is dominated by attention Q+K+V+attn_out buffers during
full attention: `4096 * (8192 + 1024 + 1024 + 8192) * 2 = ~143 MB` in
bf16. With int8 activations, the quantized buffers are half: `~72 MB`.
Dynamic scales add `4 * 4096 = 16 KB` per quantization point — negligible.


---


## 10. NUMA Considerations

### 10.1 Expert Weight Placement

Expert weights are the dominant memory footprint. With 128 experts and
top-8 routing, access patterns are data-dependent and unpredictable.

Strategy: stripe experts across NUMA nodes by expert index. With 2 nodes:
experts [0..63] on node 0, [64..127] on node 1. BurstPool dispatch routes
expert jobs to the node holding that expert's weights (frequent-reader
principle from CLAUDE.md: data lives closest to the most-frequent
operation). ~50% of active experts will be node-local on average.

Int8 weights halve the remote-read bandwidth cost when routing crosses
nodes. This is the secondary NUMA win beyond the raw memory reduction.

### 10.2 KV Cache Placement

KV caches are read sequentially (streaming over context positions) and
written once per token. Place on the same node as the attention workers.
The BurstPool's per-KV-group dispatch naturally achieves this when cache
and workers are co-located.

### 10.3 Activation Buffers

Scratch buffers (Q, K, V projections, MLP intermediates) are short-lived.
Allocate on the NUMA node of the dispatching thread. The ScratchPool
borrow/release pattern from `gemma_4_moe.mojo` naturally supports this.


---


## 11. Implementation Order

### Phase 1: Offline Quantizer

Extend `quant/butterquant.mojo` for Gemma 4's weight inventory:
- Add `Quantizable`, `Gamma`, `Absorbed` traits to Gemma4Model weight slots
- Per-head-norm gamma absorption (Q, K row blocks)
- Router scale absorption
- Expert weight iteration (128 experts x 30 layers)
- Two FWHT block sizes (256 for hidden-dim, 64 for intermediate-dim)
- V scale computation (sliding: from W_V norms, full: from head_dim)

Deliverable: `gemma4_quantize.mojo` that reads bf16 safetensors and
writes int8 safetensors with VNNI-packed weights and per-row scales.

### Phase 2: Attention Kernels

- `prep_q_row_normed` (sliding) and `prep_q_row_normed_partial` (full)
- `write_k_head_normed` / `write_v_head_normed` with RMS-divide
- `attn_sliding_decode.mojo` — adapt experimental2 decode with scale=1.0
- `attn_full_decode.mojo` — adapt for head_dim=512, GQA=8, partial RoPE

Validation: cosine similarity of attention output vs bf16 reference,
per layer, sliding and full separately.

### Phase 3: GEMM Kernels

- `gelu_tanh_fwht_quantize` kernel
- `fused_expert_gelu_tanh_worker` for MoE experts
- Wire int8_gemv for all projection dimensions

Validation: per-projection cosine similarity vs bf16 reference.

### Phase 4: Forward Pass Integration

- Gemma4 model struct with int8 weight layout and quantized KV caches
- Forward pass using experimental3 kernels
- Layer-by-layer cosine similarity validation (target: cos > 0.999
  through all 30 layers)

### Phase 5: Performance

- BurstPool dispatch tuning for expert parallelism
- NUMA-aware expert placement
- AMX tile config optimization for head_dim=512
- Prefill kernels for sliding and full attention
