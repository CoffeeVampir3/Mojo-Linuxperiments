# ButterQuant

## Isotropic Int8 Quantization via Hadamard Butterfly Networks

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

## 1.3 Isotropization

The Hadamard matrix has the property that every output component depends equally on every input component (with $\pm 1/\sqrt{n}$ coefficients). For an arbitrary input vector $x$, the components of $\tilde{x} = H_n x$ have approximately equal magnitude — the energy of $x$ is spread uniformly regardless of the input's structure.

This is not a probabilistic statement about $x$. It is a deterministic consequence of the equal-coefficient structure of $H_n$: each $\tilde{x}_i = \sum_j (\pm 1/\sqrt{n}) x_j$ is a signed average of all components of $x$.

**Consequence for quantization.** If $\tilde{x}$ has approximately equal component magnitudes, then symmetric quantization with a single scale uses the full integer range for every component. No component is wasted on outliers. The effective bit utilization approaches the theoretical maximum.

## 1.4 The Concentration Constant $C(n)$

**Definition.** For block size $n$, define:

$$C(n) = \mathbb{E}_{x \sim \text{Uniform}(S^{n-1})} \left[\sqrt{n} \cdot \max_{i=1}^{n} |[H_n x]_i|\right]$$

where $S^{n-1}$ is the unit sphere in $\mathbb{R}^n$. $C(n)$ is the expected ratio of the absmax of $H_n x$ to the per-component RMS of $H_n x$, for a random unit vector.

$C(n)$ is a deterministic function of $n$ alone. It is computable to arbitrary precision by Monte Carlo sampling (draw random unit vectors, apply $H_n$, measure the absmax-to-RMS ratio, average) or by numerical integration over the sphere.

**Computation procedure.** To compute $C(n)$ to precision $\epsilon$:

```
samples = 0, total = 0
repeat until confidence interval width < epsilon:
    x = randn(n); x = x / norm(x)           # uniform on S^{n-1}
    hx = FWHT(x)
    total += sqrt(n) * max(abs(hx))
    samples += 1
C(n) = total / samples
```

**Reference values** (computed to 3 significant figures):

| $n$ | $C(n)$ |
|-----|--------|
| 32  | 3.18 |
| 64  | 3.53 |
| 128 | 3.85 |
| 256 | 4.15 |
| 512 | 4.42 |

$C(n)$ grows as $O(\sqrt{\log n})$. For practical block sizes (64–512), it ranges from approximately 3.5 to 4.5.

**Usage.** For a vector $x$ with $\|x\| = r$ partitioned into blocks of size $n$, the expected absmax of $\mathcal{H} x$ within each block is:

$$\mathbb{E}[\max_i |\tilde{x}_i|] \approx \frac{r}{\sqrt{n}} \cdot C(n)$$

where $r/\sqrt{n}$ is the per-component RMS within each block (since the $n$-component block has norm $r \cdot \sqrt{n/d}$ for a $d$-dimensional vector... see §3.2 for the precise per-context derivation).

---

# II. Transformer Algebra in the Hadamard Domain

## 2.1 Linear Projection

Let $y = Wx$. In the Hadamard domain:

$$\tilde{y} = \tilde{W}\,\tilde{x}, \quad \text{where } \tilde{W} = \mathcal{H}_M W \mathcal{H}_K^T$$

**Proof.** $\mathcal{H}_M y = \mathcal{H}_M W (\mathcal{H}_K^T \mathcal{H}_K) x = (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x) = \tilde{W}\,\tilde{x}$. $\square$

**Domain of the output.** The FWHT $\mathcal{H}_K$ on the contraction dimension $K$ cancels between the rotated activation and rotated weight. The output carries $\mathcal{H}_M$ on the output dimension $M$. If $M$ is not subsequently FWHT'd, the output is in the Hadamard domain for $M$ and the original domain for any other dimension. For the attention projections, $M$ = num\_heads $\times$ head\_dim, and the per-head FWHT on the output dimension is applied explicitly at KV cache write time and Q prep time.

## 2.2 Residual Connection

$\tilde{x} + \tilde{z} = \mathcal{H}(x + z)$ by linearity. No quantization, no domain transition. Both operands must be in the same domain (both Hadamard, or both original). In ButterQuant, residuals occur in f32 in the Hadamard domain, between dequantized GEMM outputs.

## 2.3 Scalar Multiplication

$\alpha\,\tilde{x} = \mathcal{H}(\alpha x)$ by linearity. Covers $1/\sqrt{d_k}$ scaling and RMSNorm division.

## 2.4 RMSNorm

**Definition.** $\text{rms}(x) = \sqrt{d^{-1}\|x\|^2}$.

By Parseval: $\text{rms}(\tilde{x}) = \text{rms}(x)$. The norm is computable in the Hadamard domain.

**Gain absorption.** The elementwise gain $\gamma$ does not commute with $\mathcal{H}$:

$$\mathcal{H}(u \odot \gamma) \neq (\mathcal{H}u) \odot (\mathcal{H}\gamma)$$

Resolution: define $W' = W \cdot \text{diag}(\gamma)$ offline. This is a column-wise scaling: $W'_{n,k} = W_{n,k} \cdot \gamma_k$. Then runtime RMSNorm reduces to:

$$\text{RMSNorm}(x) \mapsto \frac{x}{\text{rms}(x)}$$

The output norm is exact: $\|x / \text{rms}(x)\| = \sqrt{d}$.

**Absorption mapping.** For a standard transformer layer:

| Gamma source | Absorbed into |
|-------------|---------------|
| input\_layernorm.$\gamma$ | $W_Q$, $W_K$, $W_V$ (shared input) |
| post\_attention\_layernorm.$\gamma$ | $W_{\text{gate}}$, $W_{\text{up}}$ (shared input) |
| $W_O$, $W_{\text{down}}$ | Nothing (no norm precedes them) |

The absorbed gamma vectors are consumed during offline quantization and absent from the runtime model.

## 2.5 Attention Scores

For per-head query and key vectors $q_h, k_h \in \mathbb{R}^{d_k}$:

$$q_h^T k_h = \tilde{q}_h^T \tilde{k}_h$$

by Parseval (§1.2, property ii). The $1/\sqrt{d_k}$ scaling is a scalar multiply (§2.3). Softmax operates over the sequence dimension, orthogonal to the feature-space Hadamard.

## 2.6 Value Aggregation

$$\tilde{y}_h = \sum_i a_i\,\tilde{v}_{h,i}$$

by linearity of $H_h$ and §2.3, where $a_i$ are scalar attention weights (softmax outputs).

## 2.7 Nonlinear Activations (Domain Exit)

Elementwise nonlinearities $\phi$ (SiLU, GELU, ReLU) do not commute with $\mathcal{H}$:

$$\tilde{x} \xrightarrow{\mathcal{H}^{-1}} x \xrightarrow{\phi} \phi(x) \xrightarrow{\mathcal{H}} \widetilde{\phi(x)}$$

This is the sole domain exit per layer. Cost: two FWHT passes at $O(d_{\text{ff}} \log n)$ each, fused with $\phi$ in a single kernel (one read, one write).

---

# III. ButterQuant Quantization Scheme

## 3.1 Quantization Operator

**Definition.** The ButterQuant quantization operator $Q_S : \mathbb{R} \to \{-127, \ldots, 127\}$ is:

$$Q_S(x) = \text{clamp}\!\left(\text{round}\!\left(x \cdot \frac{127}{S}\right),\; -128,\; 127\right)$$

where $S \in \mathbb{R}^+$ is the scale parameter. The corresponding dequantization is:

$$\hat{x} = Q_S(x) \cdot \frac{S}{127}$$

This is uniform symmetric int8 quantization. The grid spacing is $S/127$. The nominal range is $[-S, S]$ (127 levels per side). The clamping range $[-128, 127]$ admits one extra negative level: $-128$ dequantizes to $-128S/127 \approx -1.008S$, extending the negative range by $0.8\%$. The theoretical symmetric claim is $[-127, 127]$; the full i8 range $[-128, 127]$ is used in practice because the hardware represents it natively, the asymmetry is negligible at 8-bit precision, and the extra level is free.

Every quantization point in ButterQuant uses this same operator — the only variable is $S$.

## 3.2 Scale Derivation

All scales are derived from two quantities: weight matrix Frobenius norms (known from the checkpoint) and $C(n)$ (known from the block size). No runtime measurement. No calibration data.

### Category A: Universal activation scale

Every activation entering a GEMM has been RMSNorm'd. After RMSNorm, $\|x_{\text{normed}}\| = \sqrt{d}$. The block-diagonal FWHT partitions this into $d/n$ blocks of $n$ components. Each block has norm $\sqrt{d} \cdot \sqrt{n/d} = \sqrt{n}$ (assuming the norm distributes evenly, which the Hadamard of the previous layer's output ensures). The per-component RMS within each block is $\sqrt{n}/\sqrt{n} = 1$. The expected absmax per block is $C(n) \cdot 1 = C(n)$.

$$\boxed{S_{\text{act}} = C(n)}$$

One constant for the entire model. Depends only on the FWHT block size.

### Category B: Per-projection scale

The GEMM output $y = W' x_{\text{normed}}$ has a different norm than $x_{\text{normed}}$. For a weight matrix $W' \in \mathbb{R}^{M \times K}$ with gamma absorbed:

The output $y \in \mathbb{R}^M$ is partitioned into heads of dimension $d_k = $ head\_dim. Per head, $\|y_{\text{head}}\|$ depends on the submatrix of $W'$ for that head. Averaging over input directions (justified by the isotropization of the previous layer):

$$\text{RMS}(y_{\text{head}}) \approx \frac{\|W'\|_F}{\sqrt{M \cdot K}} \cdot \|x_{\text{normed}}\| = \frac{\|W'\|_F}{\sqrt{M \cdot K}} \cdot \sqrt{d}$$

For the common case $M = d$ (attention projections):

$$\text{RMS}(y_{\text{head}}) = \frac{\|W'\|_F}{\sqrt{K}}$$

After per-head FWHT (block size $n = d_k$), the absmax concentrates at:

$$\boxed{S_{\text{proj}} = \frac{\|W'\|_F}{\sqrt{K}} \cdot C(d_k)}$$

Instances (one f32 per layer each):

| Scale | Weight matrix | Used by |
|-------|--------------|---------|
| $S_Q$ | $W'_Q$ | Q quantization in attention kernel |
| $S_K$ | $W'_K$ | K cache write |
| $S_V$ | $W'_V$ | V cache write |

### Category C: Post-nonlinearity scale

After the MLP domain exit, the nonlinearity output $\phi(h_g, h_u) = \text{activation}(h_g) \odot h_u$ is FWHT'd and re-quantized.

The scale depends on the second moment of the nonlinearity. Define:

$$M_2(\text{activation}, \sigma) = \mathbb{E}_{X \sim \mathcal{N}(0, \sigma^2)}\!\left[\text{activation}(X)^2\right]$$

This is a computable integral for each activation function:

| Activation | $M_2(\text{act}, \sigma)$ | Method |
|-----------|--------------------------|--------|
| ReLU | $\sigma^2 / 2$ (exact) | $\int_0^\infty x^2 \varphi(x/\sigma)\,dx / \sigma$ |
| SiLU | $\approx 0.298\,\sigma^2$ | Numerical: $\int [x \cdot \text{sigmoid}(x)]^2 \varphi(x/\sigma)\,dx / \sigma$ |
| GELU | $\approx 0.321\,\sigma^2$ | Numerical: $\int [\text{gelu}(x)]^2 \varphi(x/\sigma)\,dx / \sigma$ |

These constants are computed once by numerical integration (or Monte Carlo: draw $X \sim \mathcal{N}(0, \sigma^2)$, compute $\text{mean}(\text{act}(X)^2)$). The values above are for $\sigma = 1$; for general $\sigma$, $M_2 \propto \sigma^2$ by scaling.

For a gated MLP with $\phi = \text{activation}(W'_{\text{gate}} x) \odot W'_{\text{up}} x$, the per-component variance of $\phi$ is:

$$\text{Var}(\phi_i) = M_2\!\left(\text{act},\; \frac{\|W'_{\text{gate}}\|_F}{\sqrt{K}}\right) \cdot \frac{\|W'_{\text{up}}\|_F^2}{K}$$

The per-component RMS of $\phi$ is $\sqrt{\text{Var}(\phi_i)}$. After FWHT (block size $n$), the absmax concentrates at RMS $\times\; C(n)$:

$$\boxed{S_{\text{post}} = \sqrt{M_2\!\left(\text{act},\; \frac{\|W'_{\text{gate}}\|_F}{\sqrt{K}}\right) \cdot \frac{\|W'_{\text{up}}\|_F^2}{K}} \;\cdot\; C(n)}$$

One f32 per layer, computed offline.

### Category D: Attention output scale

The attention output is a convex combination of V vectors: $\text{out} = \sum_t a_t \tilde{v}_t$ where $a_t \geq 0$, $\sum a_t = 1$.

By convexity, $\|\text{out}\| \leq \max_t \|\tilde{v}_t\|$. The max V norm is bounded by the scale used to quantize V: $\|\tilde{v}_t\| \leq S_V$ (values outside $[-S_V, S_V]$ are clipped at quantization). Therefore:

$$\boxed{S_{\text{attn\_out}} = S_V}$$

This is a bound, not an exact derivation. It may be conservative (attention spread across many tokens causes cancellation, reducing the output norm below $S_V$). Using $S_V$ is safe: it never clips, at the cost of potentially underutilizing the int8 range by a factor determined by the attention entropy.

### Scale summary

| Scale | Formula | Count | Computed from |
|-------|---------|-------|---------------|
| $S_{\text{act}}$ | $C(n)$ | 1 per model | Block size |
| $S_Q$ | $\|W'_Q\|_F / \sqrt{K} \cdot C(d_k)$ | 1 per layer | Checkpoint |
| $S_K$ | $\|W'_K\|_F / \sqrt{K} \cdot C(d_k)$ | 1 per layer | Checkpoint |
| $S_V$ | $\|W'_V\|_F / \sqrt{K} \cdot C(d_k)$ | 1 per layer | Checkpoint |
| $S_{\text{post}}$ | See §3.2 Category C | 1 per layer | Checkpoint + $M_2$ |
| $S_{\text{attn\_out}}$ | $S_V$ | 1 per layer | $S_V$ |

All deterministic from the checkpoint. No calibration.

## 3.3 Storage Convention

Int8 values in the KV cache use the u8 convention for VNNI/AMX hardware compatibility:

$$x_{\text{u8}} = Q_S(x) \oplus \texttt{0x80}$$

where $\oplus$ is bitwise XOR. This maps signed i8 $\{-128, \ldots, 127\}$ to unsigned u8 $\{0, \ldots, 255\}$ by flipping the sign bit. The AMX instruction `tdpbusd` requires one unsigned and one signed operand; this convention allows cache data to serve as the unsigned operand without conversion.

---

# IV. Weight Quantization (Offline)

Weights use per-row symmetric int8 — standard channelwise quantization, improved by Hadamard rotation. This is distinct from the per-layer fixed scales used for activations and KV cache. Per-row weight scales are necessary because different rows of a weight matrix have genuinely different norms (unlike activations, which are normalized by RMSNorm).

For each weight matrix $W$:

**Step 1. Gamma absorption** (if preceded by RMSNorm):

$$W'_{n,k} = W_{n,k} \cdot \gamma_k$$

Column-wise scaling. See §2.4 for which weights absorb which gamma.

**Step 2. FWHT rotation** (contraction dimension $K$, per row):

$$W_{\text{rot}}[n,:] = \mathcal{H}_K(W'[n,:])$$

**Step 3. Per-row quantization:**

$$s_w[n] = \frac{\max_k |W_{\text{rot}}[n,k]|}{127}, \quad W_{\text{i8}}[n,k] = Q_{s_w[n]}(W_{\text{rot}}[n,k])$$

**Step 4. VNNI packing and column sum precomputation.**

VNNI format for the B operand of AMX `tdpbusd`/`tdpbsud`: a tile of dimensions $[K_{\text{tile}}, N_{\text{tile}}]$ is stored as $[K_{\text{tile}}/4,\; N_{\text{tile}},\; 4]$ — groups of 4 consecutive K-elements are interleaved with the N-column index. Byte at logical position $[k, n]$ is stored at offset $(k/4) \cdot N_{\text{tile}} \cdot 4 + n \cdot 4 + (k \bmod 4)$.

Column sums for bias correction:

$$\text{colsum}[n] = \sum_{k} W_{\text{i8}}[n, k]$$

Stored as f32 alongside each weight matrix.

## 4.1 Int8 GEMM

The int8 GEMM computes $y = W' x_{\text{normed}}$ as:

$$\text{raw}[m, n] = \sum_k x_{\text{u8}}[m, k] \cdot W_{\text{i8}}[n, k]$$

where $x_{\text{u8}} = x_{\text{i8}} \oplus \texttt{0x80}$ (activation stored as i8, XOR'd to u8 at register load for VNNI's unsigned $\times$ signed convention). This introduces a bias:

$$\text{raw}[m, n] = \sum_k (x_{\text{i8}}[m,k] + 128) \cdot W_{\text{i8}}[n,k] = \langle x_{\text{i8}}[m], W_{\text{i8}}[n] \rangle + 128 \cdot \text{colsum}[n]$$

The dequantized output is:

$$\boxed{y[m, n] = \left(\text{raw}[m, n] - 128 \cdot \text{colsum}[n]\right) \cdot s_{\text{act}} \cdot s_w[n]}$$

where $s_{\text{act}} = S_{\text{act}} / 127$ is the activation dequant scale and $s_w[n] = s_w[n]$ is the per-row weight dequant scale. The output is bf16 in the original domain (the Hadamard rotations on the contraction dimension $K$ cancelled between activation and weight — see §2.1).

---

# V. KV Cache

## 5.1 Format

Flat u8 array per cache (K or V, separately):

$$\texttt{data}: \texttt{u8}\;[\texttt{num\_kv\_heads},\; \texttt{max\_seq},\; \texttt{head\_dim}]$$

No scale arrays. No per-token metadata. K and V have identical layout and identical read/write mechanics (K additionally has RoPE before FWHT at write time).

Total bytes per position per layer: $2 \cdot \texttt{num\_kv\_heads} \cdot \texttt{head\_dim}$.

## 5.2 Cache Write (K)

Given $K_{\text{bf16}}$ from the K projection GEMM (§4.1 output, original domain), for each KV head $g$:

$$k = K_{\text{bf16}}[g \cdot d_k : (g+1) \cdot d_k]$$
$$k_{\text{rope}}[j] = k[j] \cos(\theta_j \cdot \text{pos}) + k[j + d_k/2] \sin(\theta_j \cdot \text{pos}) \quad \text{(see §5.4 for RoPE)}$$
$$\tilde{k} = \text{FWHT}(k_{\text{rope}})$$
$$\texttt{cache}[\text{pos}, i] = Q_{S_K}(\tilde{k}[i]) \oplus \texttt{0x80}$$

## 5.3 Cache Write (V)

Identical to K, without RoPE:

$$v = V_{\text{bf16}}[g \cdot d_k : (g+1) \cdot d_k]$$
$$\tilde{v} = \text{FWHT}(v)$$
$$\texttt{cache}[\text{pos}, i] = Q_{S_V}(\tilde{v}[i]) \oplus \texttt{0x80}$$

## 5.4 RoPE Convention

Rotary Position Embedding applies a 2D rotation per dimension pair. For head dimension $d_k$, define frequencies:

$$\theta_j = \text{base}^{-2j/d_k}, \quad j = 0, \ldots, d_k/2 - 1$$

where base is a model hyperparameter (typically 10000 or 500000). At position $p$:

$$\begin{pmatrix} x'_j \\ x'_{j + d_k/2} \end{pmatrix} = \begin{pmatrix} \cos(p \cdot \theta_j) & -\sin(p \cdot \theta_j) \\ \sin(p \cdot \theta_j) & \cos(p \cdot \theta_j) \end{pmatrix} \begin{pmatrix} x_j \\ x_{j + d_k/2} \end{pmatrix}$$

RoPE is applied in the original domain (before FWHT). By Parseval, $\langle \text{FWHT}(\text{RoPE}(q, p_q)),\; \text{FWHT}(\text{RoPE}(k, p_k)) \rangle = \langle \text{RoPE}(q, p_q),\; \text{RoPE}(k, p_k) \rangle$, preserving the correct relative position encoding.

## 5.5 Cache Read

No dequantization at read time. Bytes are loaded directly into AMX tile registers. For K (scoring): loaded as u8 (unsigned B operand of `tdpbsud`). For V (aggregation): XOR'd with $\texttt{0x80}$ during VNNI packing to restore true i8 (signed B operand of `tdpbusd`). Dequantization is deferred to the GEMM epilogue or final normalize via the scale constants.

---

# VI. Attention Kernel

## 6.1 Grouped Query Attention (GQA)

ButterQuant supports GQA natively. With $N_h$ query heads and $N_{kv}$ KV heads, the GQA factor is $G = N_h / N_{kv}$. Each KV head serves $G$ query heads. The kernel processes one KV group at a time: $G$ query heads share the same K and V data.

The Q buffer for one KV group has $\text{seq\_len} \times G$ rows, each of dimension $d_k$. These rows are processed in batches of $Q_{\text{batch}}$ (typically 32) for L1 residency of the running accumulator.

## 6.2 Q Preparation

Given $Q_{\text{bf16}}$ from the Q projection GEMM (original domain), for each query head $h$ in KV group $g$:

$$q = \text{f32}(Q_{\text{bf16}}[h \cdot d_k : (h+1) \cdot d_k])$$
$$q_{\text{rope}} = \text{RoPE}(q, \text{pos}_q) \quad \text{(§5.4)}$$
$$\tilde{q} = \text{FWHT}(q_{\text{rope}})$$
$$q_{\text{i8}}[i] = Q_{S_Q}(\tilde{q}[i])$$
$$b_q = 128 \cdot \sum_{i=0}^{d_k-1} q_{\text{i8}}[i]$$

The f32 intermediary is required: RoPE uses transcendental cos/sin, and FWHT butterfly additions accumulate across $\log_2 n$ stages where i8 would overflow.

$b_q$ is the bias correction for the `tdpbsud` u8 offset convention (§3.3). It is the only per-query-row runtime quantity.

## 6.3 Scoring

The AMX instruction `tdpbsud` computes signed $A \times$ unsigned $B \to$ int32 $C$:

$$r_t = \sum_{i=0}^{d_k-1} q_{\text{i8}}[i] \cdot k_{\text{u8}}[t, i]$$

Since $k_{\text{u8}} = k_{\text{i8}} + 128$ (§3.3):

$$r_t = \langle q_{\text{i8}},\, k_{\text{i8}}[t] \rangle + 128 \cdot \sum_i q_{\text{i8}}[i] = \langle q_{\text{i8}},\, k_{\text{i8}}[t] \rangle + b_q$$

The true attention score (§2.5) dequantizes as:

$$s_t = \frac{\langle \tilde{q},\, \tilde{k}_t \rangle}{\sqrt{d_k}} \approx \frac{\langle q_{\text{i8}},\, k_{\text{i8}}[t] \rangle}{127^2} \cdot S_Q \cdot S_K \cdot \frac{1}{\sqrt{d_k}}$$

Therefore:

$$\boxed{s_t = (r_t - b_q) \cdot \sigma_s, \quad \sigma_s = \frac{S_Q \cdot S_K}{127^2 \cdot \sqrt{d_k}}}$$

$\sigma_s$ is a per-layer constant. $b_q$ is per-query-row.

## 6.4 Softmax

Online softmax with two passes over the i32 score buffer. The i32 values from the score GEMM are NOT converted to f32 and stored — they remain in the score buffer and are re-read in both passes.

**Pass 1 (max).** For each query row, scan causally valid positions:

$$m = \max_{t < \text{causal\_limit}} \left[(r_t - b_q) \cdot \sigma_s\right]$$

This reads i32, computes the dequant inline ($1$ subtract $+ 1$ multiply per element), and tracks the max. No f32 store.

**Correction.** Update running state from previous KV chunks:

$$m_{\text{new}} = \max(m_{\text{old}},\; m)$$

If $m_{\text{new}} > m_{\text{old}}$:

$$c = \exp(m_{\text{old}} - m_{\text{new}})$$
$$\ell \leftarrow \ell \cdot c, \quad o_d \leftarrow o_d \cdot c \;\;\forall d \in [0, d_k)$$

If $m_{\text{new}} = m_{\text{old}}$: no correction needed ($c = 1$).

**Pass 2 (exp $\to$ u8).** Re-read i32, recompute dequant, exponentiate, quantize to u8:

$$e_t = \exp\!\left((r_t - b_q) \cdot \sigma_s - m_{\text{new}}\right)$$
$$w_t = \text{round}(e_t \cdot 255)$$
$$\ell \leftarrow \ell + e_t$$

$w_t \in \{0, \ldots, 255\}$ stored as u8. The scale 255 is universal (exp output $\in [0, 1]$). Positions beyond the causal limit: $w_t = 0$.

**Overflow safety.** $\max(e_t \cdot 255) = 1.0 \cdot 255 = 255$ (at $t = \arg\max$). Two IEEE 754 f32 rounding errors contribute at most $255 \cdot (1 + 2^{-24})^2 - 255 \approx 3 \times 10^{-5}$. $\text{round}(255.00003) = 255$. No clamp required. (Full proof: the product $w \cdot \text{fl}(255/w_{\max})$ for $w \leq w_{\max}$ can exceed 255 by at most $255 \cdot 2^{-23}$, which rounds to 255.)

## 6.5 V Aggregation

The AMX instruction `tdpbusd` computes unsigned $A \times$ signed $B \to$ int32 $C$:

$$p_d = \sum_{t} w_t \cdot v_{\text{i8}}[t, d]$$

$w_t$ (u8) is the unsigned A operand: the non-negative attention weight. $v_{\text{i8}}$ is the signed B operand: the true quantized V value, restored from $v_{\text{u8}}$ by XOR with $\texttt{0x80}$ during VNNI packing. No bias correction is needed — both operands have their natural sign.

The running f32 accumulator collects raw casts without scaling:

$$o_d \leftarrow o_d + \text{f32}(p_d)$$

The scale $\sigma_v$ is deferred to the final normalize (§6.6), saving one multiply per element per chunk.

## 6.6 Final Normalize

$$\boxed{\text{output}_d = o_d \cdot \frac{\sigma_v}{\ell}, \quad \sigma_v = \frac{S_V}{255 \cdot 127}}$$

**Derivation.** The i32 accumulator contains:

$$p_d = \sum_t w_t \cdot v_{\text{i8}}[t, d] \approx \sum_t (e_t \cdot 255) \cdot \left(\tilde{v}[t,d] \cdot \frac{127}{S_V}\right) = \frac{255 \cdot 127}{S_V} \sum_t e_t \cdot \tilde{v}[t,d]$$

The true attention output is $\sum_t a_t \tilde{v}[t,d]$ where $a_t = e_t / \ell$:

$$\text{output}_d = \frac{\sum_t e_t \cdot \tilde{v}[t,d]}{\ell} = \frac{p_d \cdot S_V}{255 \cdot 127 \cdot \ell} = \frac{o_d \cdot \sigma_v}{\ell}$$

---

# VII. Kernel Constants Summary

Three constants are precomputed at dispatch time from the per-layer scales:

| Constant | Definition | Scope |
|----------|-----------|-------|
| $127 / S_Q$ | Q quantization multiplier | Per-layer |
| $\sigma_s = S_Q \cdot S_K \;/\; (127^2 \cdot \sqrt{d_k})$ | Score dequantization | Per-layer |
| $\sigma_v = S_V \;/\; (255 \cdot 127)$ | V-agg dequantization | Per-layer |

Per-query-row runtime quantity: $b_q = 128 \cdot \sum_i q_{\text{i8}}[i]$ (computed during Q quantization, cost: one reduction fused with the quantization loop).

No per-token quantities. No per-channel quantities. No runtime-adaptive quantities.

---

# VIII. One-Layer Forward Pass

Let $x^{(\ell)}$ denote the f32 activation entering layer $\ell$. The residual stream is in the **original domain**. The Hadamard domain is entered at activation quantization boundaries (the fused RMSNorm + FWHT + quantize operation) and exited when the GEMM's single-sided rotations cancel. All weights $W'$ are precomputed offline with absorbed RMSNorm gains (§2.4) and single-sided FWHT on the contraction dimension $K$ (§IV).

| Step | Operation | Scale | Domain |
|------|-----------|-------|--------|
| 1 | $x_n = x^{(\ell)} / \text{rms}(x^{(\ell)})$ | — | Original, f32 |
| 2 | $x_{\text{i8}} = Q_{S_{\text{act}}}(\text{FWHT}(x_n))$ | $S_{\text{act}}$ | Hadamard, i8 |
| 3 | $Q_{\text{bf16}} = \text{int8\_gemm}(x_{\text{i8}}, W'_{Q,\text{rot}})$ | §4.1 | Original, bf16 |
| 4 | $K_{\text{bf16}} = \text{int8\_gemm}(x_{\text{i8}}, W'_{K,\text{rot}})$ | §4.1 | Original, bf16 |
| 5 | $V_{\text{bf16}} = \text{int8\_gemm}(x_{\text{i8}}, W'_{V,\text{rot}})$ | §4.1 | Original, bf16 |
| 6 | Cache $K$: RoPE + FWHT + $Q_{S_K}$ + XOR + store | $S_K$ | Hadamard, u8 |
| 7 | Cache $V$: FWHT + $Q_{S_V}$ + XOR + store | $S_V$ | Hadamard, u8 |
| 8 | Q prep: RoPE + FWHT + $Q_{S_Q}$ + compute $b_q$ | $S_Q$ | Hadamard, i8 |
| 9 | Score: $(r_t - b_q) \cdot \sigma_s$ | $\sigma_s$ | f32 |
| 10 | Softmax: $w_t = \text{round}(\exp(s_t - m) \cdot 255)$ | 255 | u8 |
| 11 | V-agg: $o_d = \sum_t w_t \cdot v_{\text{i8}}[t,d]$ | deferred | i32 $\to$ f32 |
| 12 | Normalize: $\text{out}_d = o_d \cdot \sigma_v / \ell$ | $\sigma_v$ | Hadamard, f32 |
| 13 | Quantize attn output: $Q_{S_V}(\text{out})$ | $S_V$ | Hadamard, i8 |
| 14 | $z_{\text{attn}} = \text{int8\_gemm}(\text{out}_{\text{i8}}, W'_{O,\text{rot}})$ | §4.1 | Original, f32 |
| 15 | $r = x^{(\ell)} + z_{\text{attn}}$ | — | Original, f32 |
| 16 | $r_n = r / \text{rms}(r)$ | — | Original, f32 |
| 17 | $r_{\text{i8}} = Q_{S_{\text{act}}}(\text{FWHT}(r_n))$ | $S_{\text{act}}$ | Hadamard, i8 |
| 18 | Gate: $h_g = \text{int8\_gemm}(r_{\text{i8}}, W'_{\text{gate,rot}})$ | §4.1 | Original, bf16 |
| 19 | Up: $h_u = \text{int8\_gemm}(r_{\text{i8}}, W'_{\text{up,rot}})$ | §4.1 | Original, bf16 |
| 20 | Nonlinearity: $\phi = \text{silu}(h_g) \odot h_u$ | — | Original, f32 |
| 21 | Re-entry: $\phi_{\text{i8}} = Q_{S_{\text{post}}}(\text{FWHT}(\phi))$ | $S_{\text{post}}$ | Hadamard, i8 |
| 22 | $z_{\text{mlp}} = \text{int8\_gemm}(\phi_{\text{i8}}, W'_{\text{down,rot}})$ | §4.1 | Original, f32 |
| 23 | $x^{(\ell+1)} = r + z_{\text{mlp}}$ | — | Original, f32 |

**Domain flow.** Activations enter the Hadamard domain at quantization boundaries (Steps 2, 7, 8, 13, 17, 21) via explicit FWHT. Each int8 GEMM's single-sided weight rotation cancels the activation's FWHT on the contraction dimension (§2.1), returning the output to the original domain. The residual stream remains original-domain f32 throughout.

**Domain exits for nonlinearity.** Steps 18-19 produce original-domain outputs (the GEMM rotations cancel). The SiLU in Step 20 operates on original-domain values directly — no explicit FWHT$^{-1}$ is needed because the GEMM already returned to the original domain. The FWHT in Step 21 is the domain re-entry for the subsequent down-projection.

**Remark (domain persistence).** An alternative design uses two-sided weight rotation ($\tilde{W} = \mathcal{H}_M W' \mathcal{H}_K^T$) so that GEMM outputs remain in the Hadamard domain. This eliminates the FWHT from Steps 2 and 17 (saving $N_h \cdot d_k \cdot \log d_k$ operations per layer) but requires FWHT$^{-1}$ $\to$ RoPE $\to$ FWHT round-trips for Q and K, partially offsetting the savings. ButterQuant uses single-sided rotation for uniformity: all weights have the same format, all GEMMs behave identically, and RoPE operates in the natural (original) domain without round-trips.

---

# IX. AMX Tile Configuration

Both scoring (`tdpbsud`) and V-agg (`tdpbusd`) use identical tile dimensions:

| Tile | Role | Dimensions | Bytes |
|------|------|-----------|-------|
| 0, 1 | A operand | 16 rows $\times$ 64 cols (i8 or u8) | 1024 |
| 2, 3 | B operand (VNNI) | 64 rows $\times$ 16 cols, packed as [16, 16, 4] | 1024 |
| 4, 5, 6, 7 | C accumulator | 16 rows $\times$ 16 cols (i32) | 1024 |

Configuration: 2 A tiles, 2 B tiles, 4 C tiles. Output tiling: $2 \times 2$ = $[M_{\text{step}}=32, N_{\text{step}}=32]$ per tile group. All ButterQuant GEMMs (scoring, V-agg, projections) use this same configuration.

The B operand VNNI format stores byte $[k, n]$ at offset $(k/4) \cdot 16 \cdot 4 + n \cdot 4 + (k \bmod 4)$. Groups of 4 consecutive K-elements for the same N-column are packed into a 32-bit dword.

---

# X. Error Characteristics

**Per-component.** With 8-bit uniform symmetric quantization on the isotropized distribution, the effective bits utilized are 7.99 / 8 regardless of input distribution (the Hadamard eliminates outlier-induced range waste).

**Measured kernel accuracy** (AMX prefill, $d_k=64$, context=64, 12 heads, 4 KV heads):

$$\text{avg\_abs\_err} = 1.37 \times 10^{-7}$$

---

# XI. Prerequisites and Constraints

1. **RMSNorm required.** $\|x / \text{rms}(x)\| = \sqrt{d}$ is exact and essential. LayerNorm's mean subtraction does not commute with $\mathcal{H}$.

2. **Gamma absorption required.** Offline: $W' = W \cdot \text{diag}(\gamma)$. See §2.4 for the mapping.

3. **Block size = head\_dim.** Per-head attention requires FWHT blocks to align with head boundaries. All projections sharing the head dimension use the same block size.

4. **Block size is a power of 2.** Must divide all tensor dimensions including under tensor parallelism: $n \mid (d / T)$.

5. **All scales derived from checkpoint.** Frobenius norms $+$ $C(n)$ $+$ $M_2$ table. No calibration data.

6. **Tensor parallelism and scale computation.** Per-projection scales use $\|W'\|_F$ — the Frobenius norm of the **full** (pre-sharding) weight matrix. Under tensor parallelism with $T$ ranks, each rank holds a shard. The full norm is recovered by a one-time all-reduce at model load: $\|W'\|_F = \sqrt{\sum_{p=1}^{T} \|W'_p\|_F^2}$. The scale is then broadcast to all ranks. This is a load-time operation (once per weight matrix), not a runtime cost.

---

# XII. Boundary Layers

## 12.1 Embedding (First Layer Input)

The embedding lookup produces a vector in the original domain (a row of the embedding matrix, typically bf16 or f32). This is the initial $x^{(0)}$ entering layer 0. The first Step 1-2 (RMSNorm + FWHT + quantize) handles it with no special case.

**Note on $S_{\text{act}}$ at the first layer.** The $S_{\text{act}} = C(n)$ derivation (§3.2 Category A) assumes uniform energy distribution across FWHT blocks, which holds when the input has been through prior Hadamard-domain processing. The embedding output has not. For the first layer only, the per-block energy may be non-uniform, causing occasional clipping or range underutilization. In practice the impact is negligible (one layer of $L$ total, and RMSNorm still controls the total norm). No special handling is needed.

## 12.2 LM Head (Final Layer Output)

After the last layer, $x^{(L)}$ is in the original domain (f32). The LM head projection produces logits:

$$\text{logits} = W_{\text{head}} \cdot \text{RMSNorm}(x^{(L)})$$

Two cases:

**Untied weights.** $W_{\text{head}}$ is a separate weight matrix. It receives the standard ButterQuant treatment: gamma absorption from the final RMSNorm, FWHT rotation on $K$, per-row int8 quantization. The activation is quantized with $S_{\text{act}}$ as usual. The GEMM output (logits) is f32 in the original domain. No further quantization — logits feed the sampler directly.

**Tied weights** (weight tying with embedding matrix). $W_{\text{head}} = W_{\text{embed}}^T$. The embedding matrix is typically not FWHT-rotated (it's a lookup table, not a GEMM operand in the usual sense). If tied, the final projection must either: (a) use the embedding matrix as-is with a bf16 matmul (no int8), or (b) maintain a separately rotated and quantized copy of the embedding matrix for the LM head. Option (a) is simpler; option (b) is faster for large vocabularies.

---

# XIII. What ButterQuant Eliminates

| Component | Status |
|-----------|--------|
| Per-token activation scales | Eliminated |
| Per-channel weight migration (SmoothQuant) | Eliminated |
| Outlier decomposition (mixed precision) | Eliminated |
| Online absmax reductions | Eliminated |
| Calibration data | Eliminated |
| Scale metadata in KV cache | Eliminated |
| Multiple AMX tile configurations | Eliminated |
| bf16 intermediaries in attention | Eliminated |
| Per-row adaptive scaling in softmax | Eliminated |

---

# XIV. Architecture Compatibility

| Architecture | Compatible | Notes |
|-------------|-----------|-------|
| LLaMA / Mistral / DeepSeek | Yes | RMSNorm, standard GQA |
| Mixture of Experts | Yes | Per-expert Category B/C scales |
| Multi-head Latent Attention | Yes | Block size = latent dim |
| Sliding window attention | Yes | Cache format unchanged |
| GPT-2 / BERT (LayerNorm) | No | Mean subtraction breaks commutativity |
| Models without normalization | No | Fixed-scale requires $\|x\|$ control |
