# Hadamard-Domain Transformer Arithmetic

## A Mathematical Treatment

---

# I. Foundations

## 1.1 The Hadamard Matrix

Let $H_n$ denote the $n \times n$ normalized Hadamard matrix, satisfying:

$$H_n^T H_n = H_n H_n^T = I_n$$

$$H_n^{-1} = H_n^T$$

with entries $\pm\, 1/\sqrt{n}$ and application cost $O(n \log n)$ via the fast Walsh-Hadamard transform. For notational brevity, we write $H$ where the dimension is clear from context.

**Block-diagonal extension.** For a vector space of dimension $d = n \cdot B$, we define the block-diagonal Hadamard:

$$\mathcal{H}_d = \text{diag}(\underbrace{H_n, H_n, \ldots, H_n}_{B \text{ blocks}})$$

$\mathcal{H}_d$ is orthonormal, block-local, and costs $O(d \log n)$ to apply.

**Divisibility constraint.** $\mathcal{H}_d$ is well-defined if and only if $n \mid d$. All subsequent results assume this holds. For a tensor of dimension $d$ sharded across $T$ devices, the per-shard condition is $n \mid (d / T)$.

## 1.2 The Distributional Guarantee

**Theorem (Hadamard Isotropization).** Let $x \in \mathbb{R}^n$ be an arbitrary vector with $\|x\| = r$. Define $\tilde{x} = H_n x$. Then each component $\tilde{x}_i$, after normalization to $[0,1]$, is distributed according to $\text{Beta}(\alpha, \beta)$ with parameters determined by $n$. For large $n$, the density is log-concave (when $\alpha \geq 1, \beta \geq 1$), unimodal, and concentrates as $n$ grows.

**Corollary.** The distribution of $\tilde{x}_i$ is independent of $x$ up to the scale $\|x\|$. The shape depends only on the block dimension $n$.

## 1.3 Optimal Quantization

Let $Q_b : \mathbb{R} \to \mathcal{C}_b$ denote the optimal $2^b$-level scalar quantizer for the Beta$(\alpha, \beta)$ density, where $\mathcal{C}_b = \{c_1, \ldots, c_{2^b}\}$ is the codebook of reconstruction levels.

**Theorem (Lloyd-Max Optimality).** When the source density is log-concave, the Lloyd-Max algorithm converges to the globally optimal codebook $\mathcal{C}_b^*$ minimizing:

$$\delta^2(b) = \mathbb{E}\!\left[(X - Q_b(X))^2\right] = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i)^2 f_{\text{Beta}}(x)\, dx$$

where $\{b_0, \ldots, b_{2^b}\}$ are the optimal decision boundaries.

**Universality.** Since the post-Hadamard distribution depends only on $n$ (not on the data), the codebook $\mathcal{C}_b^*$ is computed once and applies to every component of every vector transformed by $H_n$. No per-element, per-block, or per-tensor metadata is required.

## 1.4 Norm Preservation

**Theorem (Parseval).** For any orthonormal $H$ and any vectors $u, v$:

$$(i)\quad \|Hu\| = \|u\|$$

$$(ii)\quad \langle Hu, Hv \rangle = \langle u, v \rangle$$

$$(iii)\quad \|Hu - Hv\| = \|u - v\|$$

**Corollary (MSE Invariance).** If $\tilde{x} = Hx$ and $\hat{\tilde{x}} = Q_b(\tilde{x})$ componentwise, then:

$$\|x - H^T \hat{\tilde{x}}\|^2 = \|Hx - \hat{\tilde{x}}\|^2 = \|\tilde{x} - \hat{\tilde{x}}\|^2 = \sum_i ({\tilde{x}}_i - Q_b(\tilde{x}_i))^2$$

Quantization error in the Hadamard domain equals reconstruction error in the original domain.

---

# II. Domain Definitions

## 2.1 Weight Encoding

Given a weight matrix $W \in \mathbb{R}^{M \times K}$, define its Hadamard-domain representation:

$$\tilde{W} = Q_b\!\left(\mathcal{H}_M \; W \; \mathcal{H}_K^T\right)$$

where $Q_b$ is applied elementwise. $\tilde{W}$ is computed offline and stored. Each entry is an element of the codebook $\mathcal{C}_b^*$.

## 2.2 Activation Encoding

Given an activation vector $x \in \mathbb{R}^K$, define:

$$\tilde{x} = Q_b\!\left(\mathcal{H}_K \, x\right)$$

computed online. The quantization uses the same universal codebook $\mathcal{C}_b^*$.

## 2.3 Domain Transition Operators

**Entry:** $\mathcal{E}(x) = Q_b(\mathcal{H}\, x)$ — rotate and quantize into the Hadamard domain.

**Exit:** $\mathcal{E}^{-1}(\tilde{x}) = \mathcal{H}^T \tilde{x}$ — rotate back to the original domain (dequantized values).

---

# III. Linear Operations in the Hadamard Domain

## 3.1 Linear Projection

**Claim.** Let $y = Wx$. Then in the Hadamard domain:

$$\tilde{y} = \tilde{W}\,\tilde{x}$$

**Proof.** Insert $I = \mathcal{H}_K^T \mathcal{H}_K$:

$$\mathcal{H}_M\, y = \mathcal{H}_M\, W\, x = \mathcal{H}_M\, W\, (\mathcal{H}_K^T \mathcal{H}_K)\, x = (\mathcal{H}_M\, W\, \mathcal{H}_K^T)(\mathcal{H}_K\, x)$$

Up to quantization:

$$\tilde{y} = \tilde{W}\,\tilde{x} \qquad \blacksquare$$

**Remark.** The quantization of $W$ and $x$ introduces error. The exact relation is:

$$\tilde{W}\,\tilde{x} = \mathcal{H}_M W \mathcal{H}_K^T \cdot \mathcal{H}_K x + \epsilon_W \tilde{x} + \tilde{W}\epsilon_x + \epsilon_W \epsilon_x$$

where $\epsilon_W = \tilde{W} - \mathcal{H}_M W \mathcal{H}_K^T$ and $\epsilon_x = \tilde{x} - \mathcal{H}_K x$ are the quantization residuals, bounded by $\delta^2(b)$ per component.

## 3.2 Residual Connection

**Claim.** Let $y = x + z$. Then:

$$\tilde{y} = \tilde{x} + \tilde{z}$$

**Proof.** By linearity of $\mathcal{H}$:

$$\mathcal{H}(x + z) = \mathcal{H}x + \mathcal{H}z \qquad \blacksquare$$

**Remark.** Addition requires no quantization, no domain transition, and no metadata. If both operands are already in the Hadamard domain, the sum remains in the Hadamard domain.

## 3.3 Scalar-Vector Multiplication

**Claim.** Let $y = \alpha x$ for scalar $\alpha \in \mathbb{R}$. Then:

$$\tilde{y} = \alpha\, \tilde{x}$$

**Proof.** $\mathcal{H}(\alpha x) = \alpha\, \mathcal{H}x$. $\qquad \blacksquare$

**Remark.** This covers the $1/\sqrt{d_k}$ scaling in attention and the RMSNorm division.

---

# IV. Attention in the Hadamard Domain

## 4.1 Attention Projections

Let $x$ be the input activation. The query, key, and value projections are linear:

$$q = W_Q x, \quad k = W_K x, \quad v = W_V x$$

By §3.1, in the Hadamard domain:

$$\tilde{q} = \tilde{W}_Q\, \tilde{x}, \quad \tilde{k} = \tilde{W}_K\, \tilde{x}, \quad \tilde{v} = \tilde{W}_V\, \tilde{x}$$

## 4.2 Attention Scores

**Claim.** For per-head query and key vectors $q_h, k_h \in \mathbb{R}^{d_k}$, with per-head Hadamard $H_h$:

$$q_h^T k_h = \tilde{q}_h^T \tilde{k}_h$$

**Proof.** By Parseval (§1.4, property ii):

$$\langle H_h q_h,\; H_h k_h \rangle = \langle q_h,\; k_h \rangle \qquad \blacksquare$$

**Corollary.** The attention score matrix $A = \text{softmax}(Q K^T / \sqrt{d_k})$ can be computed entirely from Hadamard-domain queries and keys. The $1/\sqrt{d_k}$ scaling is a scalar multiplication (§3.3). Softmax operates over the sequence dimension, which is orthogonal to the feature-space Hadamard. No domain exit is required.

## 4.3 Attention-Weighted Sum

**Claim.** Let $y_h = \sum_i a_i v_{h,i}$ where $a_i$ are scalar attention weights. Then:

$$\tilde{y}_h = \sum_i a_i\, \tilde{v}_{h,i}$$

**Proof.** By linearity of $H_h$ and §3.3:

$$H_h y_h = H_h \sum_i a_i v_{h,i} = \sum_i a_i\, H_h v_{h,i} = \sum_i a_i\, \tilde{v}_{h,i} \qquad \blacksquare$$

**Remark.** The attention weights $a_i$ are scalars derived from softmax over the sequence dimension. They are not in the feature-space Hadamard domain and require no transformation.

## 4.4 Output Projection

$$y = W_O \cdot \text{concat}(\tilde{y}_1, \ldots, \tilde{y}_H)$$

By §3.1:

$$\tilde{y} = \tilde{W}_O \cdot \text{concat}(\tilde{y}_1, \ldots, \tilde{y}_H)$$

The concatenation of Hadamard-domain head outputs is itself a Hadamard-domain vector (each head's block is independently transformed).

---

# V. Normalization

## 5.1 RMSNorm — Norm Computation

**Definition.**

$$\text{rms}(x) = \sqrt{\frac{1}{d}\, \|x\|^2}$$

**Claim.**

$$\text{rms}(\tilde{x}) = \text{rms}(x)$$

**Proof.** By Parseval (§1.4, property i):

$$\|\tilde{x}\|^2 = \|\mathcal{H}x\|^2 = \|x\|^2 \qquad \blacksquare$$

**Remark.** The norm is computable entirely in the Hadamard domain. Under tensor parallelism, partial norms $\|\tilde{x}_p\|^2$ are accumulated via all-reduce, identically to the original domain since $\|\tilde{x}_p\| = \|x_p\|$ per shard.

## 5.2 RMSNorm — Division

**Claim.** Let $\bar{x} = x / \text{rms}(x)$. Then:

$$\tilde{\bar{x}} = \tilde{x}\, /\, \text{rms}(\tilde{x})$$

**Proof.** By §3.3 with $\alpha = 1/\text{rms}(x)$, and §5.1. $\qquad \blacksquare$

## 5.3 RMSNorm — Elementwise Gain

**Definition.** The full RMSNorm with learnable gain $\gamma \in \mathbb{R}^d$:

$$\text{RMSNorm}(x) = \frac{x}{\text{rms}(x)} \odot \gamma$$

**Problem.** The Hadamard does not distribute over elementwise (Hadamard/Schur) products:

$$\mathcal{H}(u \odot v) \neq (\mathcal{H}u) \odot (\mathcal{H}v)$$

**Resolution by conjugation.** Define:

$$\tilde{\Gamma} = \mathcal{H}\, \text{diag}(\gamma)\, \mathcal{H}^T$$

Then:

$$\mathcal{H}\!\left(\frac{x}{\text{rms}(x)} \odot \gamma\right) = \tilde{\Gamma} \cdot \frac{\tilde{x}}{\text{rms}(\tilde{x})}$$

$\tilde{\Gamma}$ is precomputed offline but is in general a dense $d \times d$ matrix, costing $O(d^2)$ to apply.

**Resolution by absorption.** If $W$ is the linear projection immediately following RMSNorm, define:

$$W' = W\, \text{diag}(\gamma)$$

This is an offline reparameterization. The RMSNorm then reduces to division by the norm (§5.2), and $\gamma$ is absorbed into the weight matrix which is subsequently Hadamard-transformed and quantized as usual:

$$\tilde{W}' = Q_b(\mathcal{H}_M\, W'\, \mathcal{H}_K^T)$$

**Result.** After absorption, RMSNorm in the Hadamard domain is:

$$\boxed{\text{RMSNorm}: \quad \tilde{x} \;\mapsto\; \frac{\tilde{x}}{\text{rms}(\tilde{x})}}$$

No domain exit. No additional parameters. The gain $\gamma$ is folded into $\tilde{W}'$.

---

# VI. Nonlinear Activations

## 6.1 The Domain Exit

Let $\phi : \mathbb{R} \to \mathbb{R}$ be a nonlinear activation applied elementwise (GELU, SiLU, ReLU, etc.). In general:

$$\mathcal{H}\, \phi(x) \neq \phi(\mathcal{H}\, x)$$

Elementwise nonlinearities do not commute with the Hadamard transform. A domain exit is required:

$$\tilde{x} \;\xrightarrow{\mathcal{H}^T}\; x \;\xrightarrow{\phi}\; \phi(x) \;\xrightarrow{\mathcal{H}}\; \widetilde{\phi(x)}$$

**Cost.** Two Hadamard transforms at $O(d \log n)$ each, fused with the elementwise $\phi$ in a single pass over the activation vector.

## 6.2 MLP Block

The standard transformer MLP:

$$\text{MLP}(x) = W_2\, \phi(W_1 x)$$

In the Hadamard domain:

$$\tilde{h} = \tilde{W}_1\, \tilde{x} \quad \text{(§3.1, Hadamard-domain matmul)}$$

$$h = \mathcal{H}^T \tilde{h} \quad \text{(exit)}$$

$$\phi(h) \quad \text{(nonlinearity in original domain)}$$

$$\widetilde{\phi(h)} = \mathcal{E}(\phi(h)) \quad \text{(re-enter)}$$

$$\tilde{y} = \tilde{W}_2\, \widetilde{\phi(h)} \quad \text{(§3.1, Hadamard-domain matmul)}$$

The domain exit-reentry is localized between the two linear projections, fused with the nonlinearity.

---

# VII. Tensor Parallelism

## 7.1 Shard Compatibility

**Condition.** Let $d$ be the tensor dimension and $T$ the parallelism degree. The block-diagonal Hadamard $\mathcal{H}_d$ is shard-compatible if:

$$n \;\Big|\; \frac{d}{T}$$

When satisfied, $\mathcal{H}_d$ decomposes as:

$$\mathcal{H}_d = \text{diag}(\mathcal{H}_{\text{shard}_1}, \ldots, \mathcal{H}_{\text{shard}_T})$$

Each shard's Hadamard is independent. No cross-device communication for the transform.

## 7.2 All-Reduce Commutativity

**Claim.** For partial results $\{y_p\}_{p=1}^T$ computed on $T$ devices:

$$\widetilde{\sum_p y_p} = \sum_p \tilde{y}_p$$

**Proof.** By linearity of $\mathcal{H}$:

$$\mathcal{H}\!\left(\sum_p y_p\right) = \sum_p \mathcal{H}\, y_p = \sum_p \tilde{y}_p \qquad \blacksquare$$

**Remark.** The all-reduce operates on Hadamard-domain partial sums. The communication pattern, volume, and semantics are identical to standard tensor parallelism.

## 7.3 Norm Computation Under Sharding

$$\|x\|^2 = \sum_p \|x_p\|^2 = \sum_p \|\tilde{x}_p\|^2 = \|\tilde{x}\|^2$$

The per-shard norm equality $\|\tilde{x}_p\| = \|x_p\|$ follows from shard-local orthonormality. The all-reduce for norm accumulation is identical in both domains.

---

# VIII. Inner Product Structure

## 8.1 Theorem

Every operation in the transformer forward pass decomposes into:

- **(a)** Inner products $\langle u, v \rangle$ (linear projections, attention scores)
- **(b)** Scalar-vector products $\alpha v$ (attention weighting, normalization scaling)
- **(c)** Elementwise additions $u + v$ (residual connections)
- **(d)** Elementwise nonlinearities $\phi(x)$ (activation functions, softmax)
- **(e)** Reductions $\sum_i f(x_i)$ (norms, softmax denominator)

Categories (a–c) commute with the Hadamard transform by orthonormality and linearity. Category (d) requires domain exit (§VI). Category (e) either preserves under orthonormality (L2 norms) or operates on a dimension orthogonal to the feature-space Hadamard (softmax over sequence length).

## 8.2 Corollary: No Outer Products Required

No operation in the standard transformer requires an outer product $uv^T$. Consequently, the quantization quality of the scheme depends only on the distribution along contraction (inner product) dimensions, which is precisely what the Hadamard controls.

## 8.3 Corollary: Single-Sided Sufficiency

If all operations are inner products, only the Hadamard on the contraction dimension $K$ is required for quantization quality:

$$W' = Q_b(W \mathcal{H}_K^T), \quad \tilde{x} = \mathcal{H}_K x$$

$$W' \tilde{x} \approx W \mathcal{H}_K^T \mathcal{H}_K x = Wx$$

The output $y$ is in the original domain. The left Hadamard $\mathcal{H}_M$ is needed only if one wishes to maintain the Hadamard domain for subsequent layers (§VIII.4).

## 8.4 Domain Persistence

Applying $\mathcal{H}_M$ on the output dimension keeps the result in the Hadamard domain:

$$\tilde{y} = \tilde{W}\, \tilde{x} = (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x)$$

This avoids re-rotation between consecutive linear layers. Domain exit is required only at nonlinearities (§VI). The two-sided transform is therefore an optimization for domain persistence, not a requirement for quantization quality.

---

# IX. Quantization Error Bound

## 9.1 Per-Component Error

Let $\delta^2(b, n)$ denote the MSE of the optimal $2^b$-level Lloyd-Max quantizer on Beta$(\alpha(n), \beta(n))$:

$$\delta^2(b, n) = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i^*)^2\, f_{\text{Beta}}(x; \alpha(n), \beta(n))\, dx$$

For large $n$ and $b$ bits, the high-resolution approximation gives:

$$\delta^2(b, n) \approx \frac{1}{12} \cdot 2^{-2b} \cdot \left(\int_0^1 f(x)^{1/3}\, dx\right)^3$$

## 9.2 Per-Vector Error

For a vector $x \in \mathbb{R}^d$ quantized through the block-diagonal Hadamard scheme:

$$\mathbb{E}\!\left[\|x - \hat{x}\|^2\right] = d \cdot \delta^2(b, n)$$

This holds for any $x$, independent of its distribution, because the error is determined entirely by the post-Hadamard Beta distribution.

## 9.3 Signal-to-Quantization-Noise Ratio

$$\text{SQNR} = \frac{\|x\|^2}{d \cdot \delta^2(b, n)}$$

The numerator is data-dependent (signal energy). The denominator is fixed (noise floor determined by bit depth $b$ and block size $n$).

## 9.4 Robustness

For Beta$(\alpha, \beta)$ with $\alpha \geq 1, \beta \geq 1$ (log-concave regime):

$$\frac{\delta^2(b, n)}{\sigma^2_{\text{Beta}}} \in [0.004, \, 0.01] \quad \text{at } b = 4$$

The relative variance loss is bounded between 0.4% and 1% across all log-concave Beta parameters. The scheme degrades gracefully: sub-1 Beta parameters (non-log-concave) increase relative error to 2–5%, but block sizes $n \geq 64$ ensure $\alpha, \beta \geq 1$.

---

# X. KV Cache Compression via Fast JL Transform

## 10.1 The FJLT

Define the Fast Johnson-Lindenstrauss projection $\Phi : \mathbb{R}^d \to \mathbb{R}^m$:

$$\Phi = S \cdot H \cdot D$$

where:

- $D = \text{diag}(\sigma_1, \ldots, \sigma_d)$, $\sigma_i \sim \text{Rademacher}$ (random sign flips)
- $H = \mathcal{H}_d$ (Hadamard transform)
- $S \in \mathbb{R}^{m \times d}$ selects $m$ coordinates, scaled by $\sqrt{d/m}$

**Cost:** $O(d \log d)$, dominated by $H$.

## 10.2 Distance Preservation

**Theorem (JL).** For any set of $N$ points and $m = O(\log N / \varepsilon^2)$:

$$(1 - \varepsilon)\|u - v\|^2 \leq \|\Phi u - \Phi v\|^2 \leq (1 + \varepsilon)\|u - v\|^2$$

with high probability over the random choices in $D$ and $S$.

**Corollary.** Inner products are preserved:

$$\left|\langle \Phi u, \Phi v \rangle - \langle u, v \rangle\right| \leq \varepsilon\, \|u\|\, \|v\|$$

## 10.3 Compressed Cache Attention

**Storage.** For each token $t$, store:

$$\hat{k}_t = Q_b(\Phi\, k_t) \in \mathcal{C}_b^m, \quad \hat{v}_t = Q_b(\Phi\, v_t) \in \mathcal{C}_b^m$$

requiring $m \cdot b$ bits per vector instead of $d_k \cdot p$ bits (where $p$ is the original precision).

**Score computation.** For a new query $q$:

$$s_t = \frac{q^T k_t}{\sqrt{d_k}} \approx \frac{(\Phi q)^T \hat{k}_t}{\sqrt{d_k}}$$

$\Phi q$ is computed once per query at cost $O(d_k \log d_k)$.

**Value aggregation.** After softmax:

$$y = \sum_t a_t\, v_t \approx \sum_t a_t\, \hat{v}_t$$

The result is in $\mathbb{R}^m$. If full-dimensional output is needed, a pseudo-inverse or learned projection maps back to $\mathbb{R}^{d_k}$.

## 10.4 Error Decomposition

The total approximation error in the attention score has two independent sources:

$$\left|q^T k_t - (\Phi q)^T \hat{k}_t\right| \leq \underbrace{\varepsilon\, \|q\|\, \|k_t\|}_{\text{JL subsampling}} + \underbrace{O(\sqrt{m}\, \delta(b))}_{\text{quantization}}$$

These are independent and additive.

## 10.5 Optimal Bit Budget

For a fixed total budget of $B = m \cdot b$ bits per cached vector, the optimal allocation minimizes:

$$\varepsilon_{\text{total}}(m, b) = \frac{c_1\, r^2}{m} + m \cdot \delta^2(b)$$

where $r^2 = \|k\|^2$ is the typical key norm. Minimizing over $m$ at fixed $b$:

$$m^* = r \sqrt{\frac{c_1}{\delta^2(b)}}$$

The optimal dimension grows with signal energy and shrinks as quantization precision improves.

## 10.6 Modularity

The FJLT cache compression is independent of the weight quantization scheme. If the weight scheme uses a Hadamard transform, the key vectors are already isotropized, and the FJLT simplifies to:

$$\hat{k}_t = Q_b(S\, \tilde{k}_t)$$

The sign-flip $D$ and Hadamard $H$ stages of the FJLT are redundant with the weight scheme's Hadamard. Only the subsampling $S$ is needed.

---

# XI. Complete Forward Pass

## 11.1 Notation

Let superscript $\ell$ denote layer index. Let $\tilde{x}^{(\ell)}$ denote the activation in the Hadamard domain entering layer $\ell$. All weight matrices $\tilde{W}^{(\ell)}_*$ are precomputed offline with absorbed RMSNorm gains.

## 11.2 One Transformer Layer

**Input:** $\tilde{x}^{(\ell)} \in \tilde{\mathbb{R}}^d$ (Hadamard domain)

**Step 1 — Pre-attention norm:**

$$\tilde{x}_{\text{normed}} = \frac{\tilde{x}^{(\ell)}}{\text{rms}(\tilde{x}^{(\ell)})}$$

*(§5.1–5.2: norm computed in Hadamard domain; gain absorbed into $\tilde{W}_Q, \tilde{W}_K, \tilde{W}_V$)*

**Step 2 — Attention projections:**

$$\tilde{q} = \tilde{W}_Q^{(\ell)}\, \tilde{x}_{\text{normed}}, \quad \tilde{k} = \tilde{W}_K^{(\ell)}\, \tilde{x}_{\text{normed}}, \quad \tilde{v} = \tilde{W}_V^{(\ell)}\, \tilde{x}_{\text{normed}}$$

*(§3.1: Hadamard-domain matmuls)*

**Step 3 — Attention scores (per head $h$):**

$$s_{h,t} = \frac{\tilde{q}_h^T \tilde{k}_{h,t}}{\sqrt{d_k}} \quad \forall\, t \in \{1, \ldots, L\}$$

*(§4.2: inner products preserved; §3.3: scalar division)*

**Step 4 — Attention weights:**

$$a_{h,t} = \text{softmax}_t(s_{h,t})$$

*(Softmax over sequence dimension, orthogonal to feature-space Hadamard)*

**Step 5 — Attention output (per head):**

$$\tilde{y}_h = \sum_t a_{h,t}\, \tilde{v}_{h,t}$$

*(§4.3: scalar-weighted linear combination)*

**Step 6 — Output projection:**

$$\tilde{z}_{\text{attn}} = \tilde{W}_O^{(\ell)}\, \text{concat}(\tilde{y}_1, \ldots, \tilde{y}_H)$$

*(§3.1)*

**Step 7 — Residual:**

$$\tilde{r}^{(\ell)} = \tilde{x}^{(\ell)} + \tilde{z}_{\text{attn}}$$

*(§3.2)*

**Step 8 — Pre-MLP norm:**

$$\tilde{r}_{\text{normed}} = \frac{\tilde{r}^{(\ell)}}{\text{rms}(\tilde{r}^{(\ell)})}$$

*(§5.1–5.2; gain absorbed into $\tilde{W}_1$)*

**Step 9 — MLP up-projection:**

$$\tilde{h} = \tilde{W}_1^{(\ell)}\, \tilde{r}_{\text{normed}}$$

*(§3.1)*

**Step 10 — Nonlinearity (domain exit):**

$$h = \mathcal{H}^T \tilde{h}$$

$$\phi(h) = \text{activation}(h)$$

$$\widetilde{\phi(h)} = \mathcal{E}(\phi(h))$$

*(§6.1: the sole domain exit in the layer)*

**Step 11 — MLP down-projection:**

$$\tilde{z}_{\text{mlp}} = \tilde{W}_2^{(\ell)}\, \widetilde{\phi(h)}$$

*(§3.1)*

**Step 12 — Residual:**

$$\tilde{x}^{(\ell+1)} = \tilde{r}^{(\ell)} + \tilde{z}_{\text{mlp}}$$

*(§3.2)*

**Output:** $\tilde{x}^{(\ell+1)} \in \tilde{\mathbb{R}}^d$ (Hadamard domain, input to layer $\ell + 1$)

---

## 11.3 Domain Exit Count

Per layer: **exactly one domain exit** at the MLP nonlinearity (Step 10), consisting of two Hadamard transforms fused with the elementwise activation. All other operations — five matmuls, two norms, two residual additions, attention scoring, and attention weighting — remain in the Hadamard domain throughout.
