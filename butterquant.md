# ButterQuant

## AMX-Native Int8 Quantization via Hadamard Rotation and Dynamic Scales

ButterQuant is a hardware-optimized implementation of Hadamard-rotated channelwise int8 quantization, targeting Intel AMX (VNNI/`tdpbusd`/`tdpbsud`) for both weight-activation GEMVs and quantized attention scoring.

The quantization scheme follows the same principle as TurboQuant (Zandieh et al., 2025) and QuaRot (Ashkboos et al., 2024): apply an orthogonal rotation before quantization to spread energy uniformly across coordinates, then quantize each coordinate with a scalar quantizer. Where TurboQuant optimizes for minimal bit width (achieving quality neutrality at 3.5 bits via Lloyd-Max codebooks and QJL residual correction), ButterQuant optimizes for integer throughput at 8 bits — using the structured Hadamard transform (O(n log n) vs dense rotation's O(n²)), uniform symmetric int8 quantization (native to VNNI dot products), and per-row dynamic absmax scales (channelwise quantization). At 8-bit precision, uniform quantization with dynamic scales is already near the information-theoretic optimum, so the Lloyd-Max codebook advantage is negligible; the performance gain comes from executing the entire pipeline — activation quantization, projection GEMVs, attention scoring, and V aggregation — in int8 on AMX hardware.

**Key design choices and their motivation:**

- **Hadamard rotation** rather than random orthogonal: O(n log n) structured transform, self-inverse, implemented as in-register SIMD butterfly stages. Required for GEMM algebra (single-sided weight rotation cancels with activation rotation on the contraction dimension) and attention domain consistency (Parseval preserves dot products). Does not eliminate per-head magnitude variance — dynamic scales handle that.

- **Dynamic per-row absmax scales** at all activation quantization points (S_act, S_Q, S_K, S_post): empirically measured cos=0.879 with fixed analytical scales vs cos=0.9999 with dynamic absmax. The analytical scale derivation (from weight Frobenius norms) underestimates actual magnitudes by 1.4–3.3× because it assumes isotropic inputs, which real activations are not. Dynamic absmax eliminates this gap at negligible cost (one SIMD reduction per row, fused into the quantize loop).

- **Fixed scale for V only**: V projection outputs have small, uniform per-head norms (max/min ratio ~1.5×) because V magnitude doesn't affect attention distribution — only Q/K do. Per-position V scales would require dequantization inside the AMX tile accumulation, breaking the hardware pipeline. Measured round-trip cos=0.993 with fixed V scale, adequate for the convex-combination output.

- **Layer-by-layer accuracy**: measured cos>0.999 between bf16 reference and ButterQuant hidden states through all 30 layers of SmolLM2-135M, with no degradation trend. Output quality is indistinguishable from the bf16 model.

---

# I. Mathematical Foundations

## 1.1 The Hadamard Matrix

Let $H_n$ denote the $n \times n$ normalized Hadamard matrix satisfying:

$$H_n^T H_n = H_n H_n^T = I_n, \quad H_n^{-1} = H_n^T = H_n$$

with entries $\pm 1/\sqrt{n}$. Application cost is $O(n \log n)$ via the Fast Walsh-Hadamard Transform (FWHT).

**FWHT algorithm.** For a vector of length $n = 2^s$, the FWHT performs $s$ butterfly stages. At stage $k$ ($k = 0, \ldots, s-1$), the vector is partitioned into pairs at stride $2^k$, and each pair $(a, b)$ is replaced by $(a + b,\; a - b)$. After all stages, the result is scaled by $1/\sqrt{n}$. The scaling is applied once at the end, not distributed across stages. The total cost is $n \log_2 n$ additions plus $n$ multiplications (the final scaling). The transform is self-inverse: applying FWHT twice recovers the original vector.

**Block-diagonal extension.** For dimension $d = n \cdot B$, define:

$$\mathcal{H}_d = \text{diag}(\underbrace{H_n, \ldots, H_n}_{B})$$

$\mathcal{H}_d$ is orthonormal, block-local, and costs $O(d \log n)$. It is well-defined iff $n \mid d$.

## 1.2 Norm Preservation (Parseval)

For any orthonormal $H$ and any vectors $u, v$:

$$(i)\; \|Hu\| = \|u\|, \quad (ii)\; \langle Hu, Hv \rangle = \langle u, v \rangle, \quad (iii)\; \|Hu - Hv\| = \|u - v\|$$

**Corollary (MSE Invariance).** Quantization error in the Hadamard domain equals reconstruction error in the original domain:

$$\|x - H^T \hat{\tilde{x}}\|^2 = \|\tilde{x} - \hat{\tilde{x}}\|^2$$

## 1.3 Role of the Hadamard Transform

The FWHT serves two distinct roles in ButterQuant:

**1. GEMM algebra (essential).** The activation FWHT and the single-sided weight FWHT cancel on the contraction dimension during int8 GEMV (see §2.1). This is the core mechanism that enables int8 matrix-vector products to produce correct results in the original domain.

**2. Attention domain consistency (structural).** Q and K must both be in the Hadamard domain for $\langle \text{FWHT}(q), \text{FWHT}(k) \rangle = \langle q, k \rangle$ (Parseval). The FWHT applied to Q/K/V before cache write and attention scoring is required for the attention dot product to be mathematically correct, not as a quantization aid.

**Note on isotropization.** The Hadamard spreads energy uniformly within each block, which helps quantization when the input has outlier components. However, for GEMM projection outputs (Q, K, gate, up), the per-head/per-row magnitude variation exceeds what a single fixed scale can cover. ButterQuant uses per-row dynamic absmax scales at these quantization points rather than relying on isotropization alone.

## 1.4 The Concentration Constant $C(n)$

**Definition.** For block size $n$, define:

$$C(n) = \mathbb{E}_{x \sim \text{Uniform}(S^{n-1})} \left[\sqrt{n} \cdot \max_{i=1}^{n} |[H_n x]_i|\right]$$

$C(n)$ is a deterministic function of $n$ alone, computed via Monte Carlo at compile time. It is used in the derivation of the V projection scale (the only remaining fixed scale — see §3.2).

---

# II. Transformer Algebra in the Hadamard Domain

## 2.1 Linear Projection

Let $y = Wx$. In the Hadamard domain:

$$\tilde{y} = \tilde{W}\,\tilde{x}, \quad \text{where } \tilde{W} = \mathcal{H}_M W \mathcal{H}_K^T$$

**Proof.** $\mathcal{H}_M y = \mathcal{H}_M W (\mathcal{H}_K^T \mathcal{H}_K) x = (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x) = \tilde{W}\,\tilde{x}$. $\square$

**Domain of the output.** ButterQuant uses single-sided rotation: weights are FWHT-rotated on the contraction dimension $K$ only. The FWHT on the activation cancels with the weight's rotation on $K$, returning the GEMM output to the original domain.

## 2.2 Residual Connection

$\tilde{x} + \tilde{z} = \mathcal{H}(x + z)$ by linearity. The residual stream stays in f32 in the original domain throughout.

## 2.3 Scalar Multiplication

$\alpha\,\tilde{x} = \mathcal{H}(\alpha x)$ by linearity. Covers $1/\sqrt{d_k}$ scaling and RMSNorm division.

## 2.4 RMSNorm

**Definition.** $\text{rms}(x) = \sqrt{d^{-1}\|x\|^2}$.

By Parseval: $\text{rms}(\tilde{x}) = \text{rms}(x)$. The norm is computable in either domain.

**Gain absorption.** The elementwise gain $\gamma$ does not commute with $\mathcal{H}$. Resolution: define $W' = W \cdot \text{diag}(\gamma)$ offline. Runtime RMSNorm reduces to division by RMS (no gamma multiply).

**Absorption mapping:**

| Gamma source | Absorbed into |
|-------------|---------------|
| input\_layernorm.$\gamma$ | $W_Q$, $W_K$, $W_V$ (shared input) |
| post\_attention\_layernorm.$\gamma$ | $W_{\text{gate}}$, $W_{\text{up}}$ (shared input) |
| $W_O$, $W_{\text{down}}$ | Nothing (no norm precedes them) |

## 2.5 Attention Scores

For per-head query and key vectors $q_h, k_h \in \mathbb{R}^{d_k}$:

$$q_h^T k_h = \tilde{q}_h^T \tilde{k}_h$$

by Parseval. The $1/\sqrt{d_k}$ scaling is a scalar multiply. Softmax operates over the sequence dimension, orthogonal to the feature-space Hadamard.

## 2.6 Value Aggregation

$$\tilde{y}_h = \sum_i a_i\,\tilde{v}_{h,i}$$

by linearity, where $a_i$ are scalar attention weights.

## 2.7 Nonlinear Activations (Domain Exit)

Elementwise nonlinearities $\phi$ (SiLU, GELU) do not commute with $\mathcal{H}$:

$$\tilde{x} \xrightarrow{\mathcal{H}^{-1}} x \xrightarrow{\phi} \phi(x) \xrightarrow{\mathcal{H}} \widetilde{\phi(x)}$$

In ButterQuant with single-sided weight rotation, GEMM outputs are already in the original domain. The SiLU operates on original-domain values directly. The FWHT in the post-nonlinearity quantize is the domain re-entry for the subsequent down-projection.

---

# III. ButterQuant Quantization Scheme

## 3.1 Quantization Operator

The quantization operator $Q_S : \mathbb{R} \to \{-128, \ldots, 127\}$ is:

$$Q_S(x) = \text{clamp}\!\left(\text{round}\!\left(x \cdot \frac{127}{S}\right),\; -128,\; 127\right)$$

where $S \in \mathbb{R}^+$ is the scale parameter. Dequantization:

$$\hat{x} = Q_S(x) \cdot \frac{S}{127}$$

## 3.2 Scale Strategy

ButterQuant uses **dynamic per-row absmax scales** at all activation quantization points, with one exception: V projection uses a fixed scale derived from weight norms.

### Dynamic scales (runtime)

At each quantization boundary, after FWHT, compute the absmax of the transformed values and use it as the scale:

$$S[m] = \max_k |\tilde{x}[m, k]|$$

This is one SIMD reduction per row, fused into the quantize loop. The scale is written to an output array and passed to the downstream GEMV for dequantization.

**Where used:**
- $S_{\text{act}}[m]$: RMSNorm + FWHT output, per sequence position. Used by QKV and gate+up GEMVs.
- $S_Q[h, m]$: Q head after RoPE + FWHT, per head per position. Stored in cache, used for score dequant.
- $S_K[h, m]$: K head after RoPE + FWHT, per head per position. Stored in cache, used for score dequant.
- $S_{\text{post}}[m]$: silu(gate)×up after FWHT, per sequence position. Used by down GEMV.

### Fixed scale (V projection only)

$$\boxed{S_V = \frac{\|W'_V\|_F}{\sqrt{M_V}} \cdot C(d_k)}$$

where $M_V$ is the output dimension of the V projection (= KV\_HIDDEN) and $C(d_k)$ is the concentration constant for the FWHT block size.

V uses a fixed scale because:
1. V projection outputs have small, uniform per-head norms (measured max/min ratio ~1.5× vs Q's ~2.1×)
2. Per-position V scales would require dequantization inside the V-aggregation GEMM summation, breaking the AMX accumulation pattern
3. Measured round-trip quality with fixed V scale: cos ≈ 0.993, adequate for the convex-combination output

The attention output is also quantized with $S_V$ for the O projection (Category D: $S_{\text{attn\_out}} = S_V$, bounded by convexity of softmax-weighted sum).

### Scale summary

| Scale | Type | Granularity | Computed |
|-------|------|-------------|----------|
| $S_{\text{act}}$ | Dynamic absmax | Per row | Runtime (fused with FWHT+quantize) |
| $S_Q$ | Dynamic absmax | Per head, per position | Runtime (in Q prep) |
| $S_K$ | Dynamic absmax | Per head, per position | Runtime (in K cache write) |
| $S_V$ | Fixed | Per layer | Load time (from weight Frobenius norm) |
| $S_{\text{post}}$ | Dynamic absmax | Per row | Runtime (fused with SiLU+FWHT+quantize) |
| $S_{\text{attn\_out}}$ | Fixed ($= S_V$) | Per layer | Load time |

## 3.3 Storage Convention

Int8 values in the K cache use the u8 convention for VNNI/AMX hardware compatibility:

$$x_{\text{u8}} = Q_S(x) \oplus \texttt{0x80}$$

where $\oplus$ is bitwise XOR. V cache stores i8 directly (signed B operand of `tdpbusd`).

---

# IV. Weight Quantization (Offline)

Weights use per-row symmetric int8 quantization with Hadamard rotation on the contraction dimension.

For each weight matrix $W$:

**Step 1. Gamma absorption** (if preceded by RMSNorm):

$$W'_{n,k} = W_{n,k} \cdot \gamma_k$$

**Step 2. FWHT rotation** (contraction dimension $K$, per row):

$$W_{\text{rot}}[n,:] = \mathcal{H}_K(W'[n,:])$$

**Step 3. Per-row quantization:**

$$s_w[n] = \frac{\max_k |W_{\text{rot}}[n,k]|}{127}, \quad W_{\text{i8}}[n,k] = Q_{s_w[n]}(W_{\text{rot}}[n,k])$$

**Step 4. VNNI packing and column sum precomputation.**

Column sums for bias correction:

$$\text{colsum}[n] = \sum_{k} W_{\text{i8}}[n, k]$$

## 4.1 Int8 GEMV

The int8 GEMV computes $y = W' x_{\text{normed}}$:

$$\text{raw}[m, n] = \sum_k x_{\text{u8}}[m, k] \cdot W_{\text{i8}}[n, k]$$

where $x_{\text{u8}} = x_{\text{i8}} \oplus \texttt{0x80}$. The dequantized output is:

$$\boxed{y[m, n] = \left(\text{raw}[m, n] - 128 \cdot \text{colsum}[n]\right) \cdot \frac{S_{\text{act}}[m]}{127} \cdot s_w[n]}$$

Note: $S_{\text{act}}[m]$ is per-row (dynamic), not per-model. For the O projection, $S_{\text{act}}[m] = S_V$ (fixed). For the down projection, $S_{\text{act}}[m] = S_{\text{post}}[m]$ (dynamic).

---

# V. Attention Cache

## 5.1 Format

The cache stores K data (VNNI-formatted u8), V data (row-major i8), and per-head dynamic scales for K and Q:

| Region | Layout | Element type |
|--------|--------|-------------|
| K data | $[\text{num\_kv\_heads}][\text{tiles}][\text{k\_slices} \times \text{TILE\_BYTES}]$ | u8 (VNNI) |
| V data | $[\text{num\_kv\_heads}][\text{max\_seq}][\text{head\_dim}]$ | i8 |
| K scales | $[\text{num\_kv\_heads}][\text{max\_seq}]$ | f32 |
| Q scales | $[\text{num\_q\_heads}][\text{max\_seq}]$ | f32 |

Per-position storage overhead from scales: $(\text{num\_kv\_heads} + \text{num\_q\_heads}) \times 4$ bytes.

## 5.2 Cache Write (K)

For each KV head $g$ at position $\text{pos}$:

$$k = K_{\text{bf16}}[g \cdot d_k : (g+1) \cdot d_k]$$
$$k_{\text{rope}} = \text{RoPE}(k, \text{pos})$$
$$\tilde{k} = \text{FWHT}(k_{\text{rope}})$$
$$S_K[g, \text{pos}] = \max_i |\tilde{k}[i]|$$
$$\texttt{cache}[\text{pos}, i] = Q_{S_K[g, \text{pos}]}(\tilde{k}[i]) \oplus \texttt{0x80}$$

The scale $S_K[g, \text{pos}]$ is stored in the cache for use during score dequantization.

## 5.3 Cache Write (V)

Identical to K without RoPE, using fixed scale $S_V$:

$$v = V_{\text{bf16}}[g \cdot d_k : (g+1) \cdot d_k]$$
$$\tilde{v} = \text{FWHT}(v)$$
$$\texttt{cache}[\text{pos}, i] = Q_{S_V}(\tilde{v}[i])$$

V is stored as i8 directly (no XOR — signed B operand of `tdpbusd`).

## 5.4 RoPE Convention

Rotary Position Embedding applies a 2D rotation per dimension pair:

$$\begin{pmatrix} x'_j \\ x'_{j + d_k/2} \end{pmatrix} = \begin{pmatrix} \cos(p \cdot \theta_j) & -\sin(p \cdot \theta_j) \\ \sin(p \cdot \theta_j) & \cos(p \cdot \theta_j) \end{pmatrix} \begin{pmatrix} x_j \\ x_{j + d_k/2} \end{pmatrix}$$

RoPE is applied in the original domain (before FWHT). By Parseval, $\langle \text{FWHT}(\text{RoPE}(q, p_q)),\; \text{FWHT}(\text{RoPE}(k, p_k)) \rangle = \langle \text{RoPE}(q, p_q),\; \text{RoPE}(k, p_k) \rangle$.

## 5.5 Cache Read

No dequantization at read time. K bytes are loaded directly into AMX tile registers as u8. V bytes are loaded as i8 for VNNI packing. Dequantization is deferred to the scoring epilogue (K) and final normalize (V).

---

# VI. Attention Kernel

## 6.1 Grouped Query Attention (GQA)

With $N_h$ query heads and $N_{kv}$ KV heads, the GQA factor is $G = N_h / N_{kv}$. Each KV head serves $G$ query heads.

## 6.2 Q Preparation

For each query head $h$ at position $\text{pos}$:

$$q = \text{f32}(Q_{\text{bf16}}[h \cdot d_k : (h+1) \cdot d_k])$$
$$q_{\text{rope}} = \text{RoPE}(q, \text{pos})$$
$$\tilde{q} = \text{FWHT}(q_{\text{rope}})$$
$$S_Q[h, \text{pos}] = \max_i |\tilde{q}[i]|$$
$$q_{\text{i8}}[i] = Q_{S_Q[h, \text{pos}]}(\tilde{q}[i])$$
$$b_q = 128 \cdot \sum_{i=0}^{d_k-1} q_{\text{i8}}[i]$$

Returns: $(b_q, S_Q)$. The scale is stored in the cache; $b_q$ is used for scoring bias correction.

## 6.3 Scoring

$$r_t = \sum_{i=0}^{d_k-1} q_{\text{i8}}[i] \cdot k_{\text{u8}}[t, i]$$

The true attention score dequantizes as:

$$\boxed{s_t = (r_t - b_q) \cdot \frac{S_Q[h, \text{pos}_q]}{127^2 \cdot \sqrt{d_k}} \cdot S_K[g, t]}$$

The Q factor $S_Q[h, \text{pos}_q] / (127^2 \cdot \sqrt{d_k})$ is constant for a given query row and can be hoisted out of the position loop. The K factor $S_K[g, t]$ is per-position, loaded from the cache's K scale array (L1-resident).

## 6.4 Softmax

Online softmax with two passes. Identical to the previous design except the score dequant includes the per-position K scale factor:

**Pass 1 (max):** $(r_t - b_q) \cdot q_{\text{partial}} \cdot S_K[g, t]$, track max.

**Pass 2 (exp → u8):** $w_t = \text{round}(\exp(s_t - m) \cdot 255)$.

## 6.5 V Aggregation

$$p_d = \sum_{t} w_t \cdot v_{\text{i8}}[t, d]$$

Unchanged from previous design. V uses fixed scale; no per-position dequant needed inside the sum.

## 6.6 Final Normalize

$$\boxed{\text{output}_d = o_d \cdot \frac{\sigma_v}{\ell}, \quad \sigma_v = \frac{S_V}{255 \cdot 127}}$$

Unchanged. The attention output is then quantized with $S_V$ for the O projection.

---

# VII. Kernel Constants Summary

| Constant | Definition | Scope |
|----------|-----------|-------|
| $q_{\text{partial}} = S_Q / (127^2 \cdot \sqrt{d_k})$ | Q-side score dequant | Per query head, per position |
| $S_K[g, t]$ | K-side score dequant | Per KV head, per position (from cache) |
| $\sigma_v = S_V / (255 \cdot 127)$ | V-agg dequant | Per layer (fixed) |
| $127 / S_V$ | Attention output quantize | Per layer (fixed) |

Per-query-row runtime quantities: $b_q$ (bias correction), $S_Q$ (Q scale). Per-position: $S_K$ (from cache).

---

# VIII. One-Layer Forward Pass

| Step | Operation | Scale | Domain |
|------|-----------|-------|--------|
| 1 | $x_n = x^{(\ell)} / \text{rms}(x^{(\ell)})$ | — | Original, f32 |
| 2 | $x_{\text{i8}} = Q_{S_{\text{act}}[m]}(\text{FWHT}(x_n))$ | Dynamic absmax | Hadamard, i8 |
| 3–5 | QKV GEMV: $x_{\text{i8}} \times W'_{Q/K/V}$ | $S_{\text{act}}[m] / 127 \times s_w[n]$ | Original, bf16 |
| 6 | Cache $K$: RoPE + FWHT + $Q_{S_K[g,m]}$ + store | Dynamic absmax | Hadamard, u8 |
| 7 | Cache $V$: FWHT + $Q_{S_V}$ + store | Fixed $S_V$ | Hadamard, i8 |
| 8 | Q prep: RoPE + FWHT + $Q_{S_Q[h,m]}$ + $b_q$ | Dynamic absmax | Hadamard, i8 |
| 9 | Score: $(r_t - b_q) \cdot q_{\text{partial}} \cdot S_K[g,t]$ | Per-position | f32 |
| 10 | Softmax → u8 weights | 255 | u8 |
| 11 | V-agg: $o_d = \sum_t w_t \cdot v_{\text{i8}}[t,d]$ | Deferred | i32 → f32 |
| 12 | Normalize: $o_d \cdot \sigma_v / \ell$ | Fixed $S_V$ | Hadamard, f32 |
| 13 | Quantize attn output: $Q_{S_V}(\text{out})$ | Fixed $S_V$ | Hadamard, i8 |
| 14 | O GEMV: $\text{out}_{\text{i8}} \times W'_O$ | $S_V / 127 \times s_w[n]$ | Original, bf16 |
| 15 | $r = x^{(\ell)} + z_{\text{attn}}$ | — | Original, f32 |
| 16 | $r_n = r / \text{rms}(r)$ | — | Original, f32 |
| 17 | $r_{\text{i8}} = Q_{S_{\text{act}}[m]}(\text{FWHT}(r_n))$ | Dynamic absmax | Hadamard, i8 |
| 18–19 | Gate+Up GEMV | $S_{\text{act}}[m] / 127 \times s_w[n]$ | Original, bf16 |
| 20 | $\phi = \text{silu}(h_g) \odot h_u$ | — | Original, f32 |
| 21 | $\phi_{\text{i8}} = Q_{S_{\text{post}}[m]}(\text{FWHT}(\phi))$ | Dynamic absmax | Hadamard, i8 |
| 22 | Down GEMV: $\phi_{\text{i8}} \times W'_{\text{down}}$ | $S_{\text{post}}[m] / 127 \times s_w[n]$ | Original, bf16 |
| 23 | $x^{(\ell+1)} = r + z_{\text{mlp}}$ | — | Original, f32 |

---

# IX. AMX Tile Configuration

Both scoring (`tdpbsud`) and V-agg (`tdpbusd`) use identical tile dimensions:

| Tile | Role | Dimensions | Bytes |
|------|------|-----------|-------|
| 0, 1 | A operand | 16 rows $\times$ 64 cols (i8 or u8) | 1024 |
| 2, 3 | B operand (VNNI) | 64 rows $\times$ 16 cols, packed as [16, 16, 4] | 1024 |
| 4, 5, 6, 7 | C accumulator | 16 rows $\times$ 16 cols (i32) | 1024 |

---

# X. Measured Accuracy

**Layer-by-layer cosine similarity** (bf16 reference vs ButterQuant, SmolLM2-135M, 51-token prefill):

- Layer 0 attention: cos = 0.9994
- Layer 0 MLP: cos = 0.9998
- Layer 29 MLP: cos = 0.9990

Cosine similarity remains above 0.999 through all 30 layers with no degradation trend. Hidden state norms track within 1-2% of the bf16 reference throughout.

**Int8 GEMV accuracy** (activation quantize → GEMV → dequant, isolated):

- QKV GEMV output: cos > 0.9999 vs bf16 GEMM

**Quantization round-trip** (FWHT → quantize → dequant → FWHT, per head):

| Quantization point | Fixed-scale cos | Dynamic-scale cos |
|-------------------|----------------|------------------|
| Q heads | 0.894–0.980 | 0.99998 |
| K heads | 0.897–0.969 | 0.99999 |
| V heads (fixed) | 0.984–0.999 | — |
| Silu output | 0.879 | 0.9999 |
| RMSNorm activation | 0.997 | 0.9999 |

---

# XI. Design Notes

## Why dynamic scales for Q/K but not V

Q and K projection outputs exhibit high per-head norm variance (measured 2.1× max/min ratio for Q, 1.8× for K in SmolLM2). This is inherent to the attention mechanism: Q/K magnitudes control attention score sharpness, and different heads specialize in different patterns, producing divergent norms. A single fixed scale clips the large heads severely (measured 45% magnitude loss, dot product ratio 0.53).

V projection outputs have small, uniform norms (measured 1.5× max/min ratio). V carries content weighted by softmax; its magnitude affects only the residual update scale, not the attention distribution. The model has no incentive to make V large. A fixed scale derived from $\|W'_V\|_F / \sqrt{M_V} \cdot C(d_k)$ produces cos ≈ 0.993, adequate for the convex-combination output.

Additionally, per-position V scales would require dequantization inside the V-aggregation AMX GEMM summation ($S_V[t]$ varies per position inside $\sum_t w_t \cdot v_{\text{i8}}[t,d] \cdot S_V[t]$), breaking the tile accumulation pattern.

## Why dynamic scales for S_act and S_post

The fixed $S_{\text{act}} = C(n)$ derivation assumes isotropic per-component magnitudes after RMSNorm + FWHT. Measured ratio of actual absmax to $C(n)$ is ~1.4× — marginal but improvable. Dynamic absmax raises cos from 0.997 to 0.9999 for essentially zero compute cost (one SIMD reduction fused into the existing quantize loop).

The fixed $S_{\text{post}}$ prediction was 3.26× too small for actual silu(gate)×up values, causing cos = 0.879. This is because the analytical formula $S_{\text{post}} = \sqrt{M_2(\text{silu}, \sigma_{\text{gate}}) \cdot \sigma_{\text{up}}^2} \cdot C(n)$ assumes isotropic input to the gate/up projections, which does not hold for real activations. Dynamic absmax eliminates the prediction entirely.

## V scale derivation

The corrected V scale formula uses $\sqrt{M_V}$ (output dimension) as the denominator, not $\sqrt{K}$ (input dimension). The general formula for per-component RMS of $y = W'x$ with isotropic input:

$$\text{RMS}(y) = \frac{\|W'\|_F}{\sqrt{M}}$$

This equals $\|W'\|_F / \sqrt{K}$ only when $M = K$ (e.g., Q projection where $M = K = \text{HIDDEN}$). For V with $M = \text{KV\_HIDDEN} \neq K$, the distinction matters by a factor of $\sqrt{K/M} = \sqrt{\text{GQA\_FACTOR}}$.

---

# XII. Prerequisites and Constraints

1. **RMSNorm required.** $\|x / \text{rms}(x)\| = \sqrt{d}$ is exact. LayerNorm's mean subtraction does not commute with $\mathcal{H}$.

2. **Gamma absorption required.** Offline: $W' = W \cdot \text{diag}(\gamma)$. See §2.4 for the mapping.

3. **Block size = head\_dim.** Per-head attention requires FWHT blocks to align with head boundaries.

4. **Block size is a power of 2.** Must divide all tensor dimensions including under tensor parallelism.

5. **V scale derived from checkpoint.** $\|W'_V\|_F$ from Frobenius norm + $C(n)$. All other scales are runtime dynamic.

6. **Tensor parallelism and V scale.** Under TP, each rank holds a V weight shard. The full norm is recovered by all-reduce at load time.

---

# XIII. Boundary Layers

## 13.1 Embedding (First Layer Input)

The embedding output enters layer 0. Dynamic $S_{\text{act}}$ handles non-isotropic embedding vectors naturally — no special case needed.

## 13.2 LM Head (Final Layer Output)

The LM head projection uses the embedding matrix (tied weights) with a bf16 matmul. No int8 quantization — logits feed the sampler directly.

---

# XIV. Architecture Compatibility

| Architecture | Compatible | Notes |
|-------------|-----------|-------|
| LLaMA / Mistral / DeepSeek | Yes | RMSNorm, standard GQA |
| Mixture of Experts | Yes | Per-expert V scales; dynamic scales adapt per-token |
| Multi-head Latent Attention | Yes | Block size = latent dim |
| Sliding window attention | Yes | Cache format unchanged (scales indexed by position) |
| GPT-2 / BERT (LayerNorm) | No | Mean subtraction breaks commutativity |
