# Isotropic Quantization: Hadamard-Domain Arithmetic for Transformer Inference

---

## Abstract

We present a unified quantization strategy for transformer inference that exploits the isotropizing properties of the Hadamard transform to achieve provably optimal scalar quantization of both weights and activations, with no per-channel scales, no online calibration, and no outlier-handling machinery. The scheme reduces a transformer's linear operations to integer matmuls against a universal precomputed codebook, with exactly one domain transition per layer (at the MLP nonlinearity). We show that the entire design follows from a single algebraic primitive — the orthonormal Hadamard rotation — applied consistently across the computation graph, and that the resulting system is compatible with block-diagonal tiling, tensor parallelism, and standard integer arithmetic pipelines. The quantization error is bounded at less than 1% of source variance at 4 bits per weight, with the bound holding uniformly across all layers and independent of data distribution.

---

## 1. Motivation

Transformer inference at scale is dominated by two costs: memory bandwidth for loading weight matrices, and arithmetic throughput for matrix multiplication. Quantizing weights to lower precision addresses both — fewer bits per weight means less data movement and, on hardware with native low-precision instructions, higher arithmetic throughput.

The practical difficulty is that weight and activation distributions in trained transformers are poorly behaved for naive quantization. Weights exhibit varying dynamic range across channels. Activations contain outlier channels with magnitudes orders of magnitude larger than the median. These phenomena force existing quantization schemes into increasingly complex compensatory mechanisms: per-channel scales, per-group scales, per-token activation scales, mixed-precision decomposition of outlier channels, and calibration-dependent importance weighting.

Each mechanism adds metadata, complicates kernel design, and introduces failure modes tied to distributional assumptions about the data. The per-channel scale must be loaded alongside the weight tile. The online activation quantization requires a reduction (absmax) before the matmul can begin. The mixed-precision outlier decomposition requires identifying and segregating outlier channels at calibration time, with no guarantee that the calibration distribution matches deployment.

We propose eliminating these mechanisms entirely by addressing their root cause. The outlier problem, the dynamic range variation, and the distributional heterogeneity across channels are all symptoms of the same underlying issue: the weight and activation values, as stored in the standard basis, have non-uniform and data-dependent statistical properties. If these properties were uniform and known, a single fixed quantization scheme would be optimal everywhere, with no adaptation required.

The Hadamard transform achieves exactly this. It is an orthonormal rotation that spreads energy uniformly across all components, converting any input distribution into a known, well-behaved form. Once the distribution is known, the optimal quantizer is known. Once the quantizer is universal, all per-channel and per-block metadata vanishes.

---

## 2. Core Construction

### 2.1 The Transform

Let $H_n$ denote the $n \times n$ normalized Hadamard matrix, with entries $\pm 1/\sqrt{n}$, satisfying $H^T H = I$. For tensor dimensions $d > n$, we use the block-diagonal form $\mathcal{H}_d = \text{diag}(H_n, \ldots, H_n)$, tiling $d/n$ independent blocks along the diagonal. We require $n \mid d$.

The block size $n$ is a design parameter. Larger $n$ gives stronger distributional guarantees; smaller $n$ gives finer compatibility with tensor sharding. In practice, $n = 512$ provides an effective tradeoff: the distributional convergence is tight, the fast Walsh-Hadamard transform costs $O(n \log n) = O(512 \times 9) \approx 4600$ operations per block, and the block size divides all common transformer dimensions at all common tensor-parallelism degrees.

The Hadamard matrix has the property that every output component depends equally on every input component (with $\pm 1/\sqrt{n}$ coefficients). This maximal mixing ensures that, regardless of the input's structure, the output components are approximately identically distributed according to a Beta distribution whose parameters depend only on the block dimension $n$. For $n \geq 64$, this Beta distribution is log-concave (shape parameters $\alpha, \beta \geq 1$), which is the technical condition required for the quantizer optimality result that follows.

### 2.2 The Quantizer

Given the known Beta distribution of post-Hadamard components, we compute the optimal scalar quantizer via the Lloyd-Max algorithm. For $b$ bits (i.e., $2^b$ reconstruction levels), this produces a codebook $\mathcal{C}_b^* = \{c_1, \ldots, c_{2^b}\}$ and decision boundaries $\{b_0, \ldots, b_{2^b}\}$ minimizing the mean squared error:

$$\delta^2(b) = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i)^2 \, f_{\text{Beta}}(x)\, dx$$

Because the source distribution is log-concave, Lloyd-Max converges to the global optimum. Because the distribution depends only on the block dimension $n$ (not on the data), the codebook is computed once and applies universally to every component of every weight and activation tensor in the model.

At $b = 4$ bits, the per-component MSE satisfies $\delta^2 / \sigma^2_{\text{Beta}} \in [0.004, 0.01]$ across all log-concave Beta parameterizations — that is, less than 1% of the source variance is lost to quantization, uniformly. By Parseval's theorem ($\|Hx - H\hat{x}\| = \|x - \hat{x}\|$), this MSE is identical in the original and Hadamard domains. The reconstruction error in the original space is exactly equal to the quantization noise in the rotated space.

### 2.3 What is Stored

A weight matrix $W \in \mathbb{R}^{M \times K}$ is stored as:

$$\tilde{W} = Q_b(\mathcal{H}_M \, W \, \mathcal{H}_K^T)$$

Each entry of $\tilde{W}$ is an element of the codebook $\mathcal{C}_b^*$, stored as a $b$-bit index. There are no accompanying scales, offsets, zero-points, or any other metadata. The codebook itself is a global constant — $2^b$ values shared across the entire model.

For comparison, a standard per-channel int8 scheme stores one float16 scale per output channel. For a 4096 × 4096 weight matrix, that is 4096 additional float16 values — 8 KB of metadata per matrix. Across a 32-layer transformer with multiple weight matrices per layer, this accumulates to hundreds of kilobytes of scale metadata that must be loaded, stored in registers, and applied in the matmul epilogue. The Hadamard scheme eliminates all of it.

---

## 3. Offline Preparation

### 3.1 RMSNorm Gain Absorption

RMSNorm, as used in modern transformer architectures, applies an elementwise learned gain $\gamma \in \mathbb{R}^d$ after normalization:

$$\text{RMSNorm}(x) = \frac{x}{\text{rms}(x)} \odot \gamma$$

The elementwise product $\odot \gamma$ does not commute with the Hadamard transform, which would force a domain exit at every normalization point — twice per layer. We eliminate this by absorbing $\gamma$ into the weight matrix of the immediately following linear projection.

If the projection following the norm is $y = W \cdot \text{RMSNorm}(x)$, we define:

$$W' = W \, \text{diag}(\gamma)$$

This is computed once, offline. The modified weight matrix $W'$ is then Hadamard-transformed and quantized as usual:

$$\tilde{W}' = Q_b(\mathcal{H}_M \, W' \, \mathcal{H}_K^T)$$

The RMSNorm operation reduces to division by a scalar:

$$\text{RMSNorm}(x) \mapsto \frac{x}{\text{rms}(x)}$$

This is exact — no approximation is involved. The gain $\gamma$ is not discarded; it is folded into the weight matrix where it participates in the same Hadamard rotation and quantization as the weights themselves. The operation is a standard offline reparameterization that changes nothing about the model's mathematical behavior.

### 3.2 Weight Quantization

For each weight matrix in the model:

1. Apply $\gamma$ absorption if the matrix follows a normalization layer.
2. Compute $\mathcal{H}_M W' \mathcal{H}_K^T$ (two block-diagonal Hadamard transforms, applied to the rows and columns respectively).
3. Quantize each entry to the nearest codebook level: $\tilde{W}'_{ij} = Q_b([\mathcal{H}_M W' \mathcal{H}_K^T]_{ij})$.
4. Store the $b$-bit indices.

The cost is negligible — two fast Hadamard transforms over the weight matrix, done once at model load time or export time.

### 3.3 Per-Layer Bit Allocation

The quantization signal-to-noise ratio for a weight matrix is:

$$\text{SQNR} = \frac{\|W\|_F^2 / (MK)}{\delta^2(b)}$$

The numerator — mean component energy — is a single scalar per weight matrix. The denominator is a known constant for each bit width $b$. This gives an instant, zero-cost predictor of quantization quality per layer.

For mixed-precision deployment, sort layers by $\|W\|_F^2 / (MK)$ and assign bit widths to meet a target average rate. Layers with smaller mean energy (and thus lower SQNR at a given bit width) receive more bits; layers with larger mean energy can tolerate fewer bits. This requires no calibration data, no trial quantization, and no forward pass evaluation — just the Frobenius norms, which are available from the checkpoint.

---

## 4. Inference Procedure

### 4.1 Activation Quantization

When an activation vector $x \in \mathbb{R}^K$ arrives at a linear layer:

1. Apply the block-diagonal Hadamard: $\tilde{x} = \mathcal{H}_K x$. Cost: $O(K \log n)$.
2. Quantize each component to the universal codebook: $\hat{\tilde{x}}_i = Q_b(\tilde{x}_i)$. Cost: $O(K)$.

There is no absmax reduction, no per-token scale computation, no dynamic range estimation. The quantization grid is fixed at compile time. Each component is independently mapped to its nearest codebook level via a fixed lookup (or equivalently, a fixed scale-and-round, since the codebook boundaries are precomputed constants).

For stochastic quantization (recommended for activations): instead of rounding to the nearest codebook level, randomly round up or down with probability proportional to the distance to each adjacent level. This makes the quantization error unbiased ($\mathbb{E}[\hat{x}] = x$) and independent across components. The per-component MSE increases slightly relative to deterministic Lloyd-Max, but the bias elimination provides tighter error concentration for inner products: the error in $\langle \hat{u}, \hat{v} \rangle$ concentrates as $O(1/\sqrt{d})$ rather than being worst-case $O(1)$.

### 4.2 Matrix Multiplication

The quantized activation $\hat{\tilde{x}}$ (in integer codebook indices) is multiplied against the stored quantized weight matrix $\tilde{W}$ (also in integer codebook indices) using native integer matmul instructions.

The critical property: both operands are integers from the same codebook, stored as plain $b$-bit values with no associated metadata. The matmul kernel loads weight tiles, loads activation tiles, and multiplies. There is no dequantization step, no scale multiplication in the epilogue, no metadata to track per tile. The kernel is strictly simpler than a standard per-channel quantized matmul.

The matmul output (in the integer accumulator) is the Hadamard-domain result. If the next operation is another linear projection, the output remains in the Hadamard domain and is requantized for the next layer. If the next operation is a nonlinearity, a domain exit is performed (§4.4).

### 4.3 RMSNorm at Inference

After $\gamma$ absorption, the runtime RMSNorm is:

1. Compute $\|\tilde{x}\|^2$ (a dot product of $\tilde{x}$ with itself). Cost: $O(d)$.
2. Divide the matmul output by $\sqrt{d^{-1} \|\tilde{x}\|^2}$.

The norm is computed in the Hadamard domain — it gives the identical value as in the original domain, by Parseval's theorem. No domain exit, no elementwise gain.

The norm computation and the subsequent matmul operate on the same input and are independent until the matmul output is ready. The norm is $O(d)$; the matmul is $O(Md)$. The norm computation is entirely hidden behind the matmul latency. In a pipelined implementation, the norm result is available (as a single scalar) before the matmul completes, and is applied as a scalar multiply to the output as it streams from the accumulator. The effective cost of RMSNorm is zero additional latency.

### 4.4 Nonlinear Activations

The MLP block requires a nonlinear activation (GELU, SiLU, or similar) between the up-projection and down-projection. This is the one operation that does not commute with the Hadamard transform and requires a domain exit.

The procedure:

1. Compute the up-projection in the Hadamard domain: $\tilde{h} = \tilde{W}_1 \hat{\tilde{x}}$.
2. Inverse Hadamard: $h = \mathcal{H}^T \tilde{h}$. Cost: $O(d_{\text{ff}} \log n)$.
3. Apply the nonlinearity: $\phi(h)$.
4. Forward Hadamard: $\widetilde{\phi(h)} = \mathcal{H}\, \phi(h)$. Cost: $O(d_{\text{ff}} \log n)$.
5. Requantize: $\hat{\tilde{z}} = Q_b(\widetilde{\phi(h)})$.
6. Down-projection: $\tilde{y} = \tilde{W}_2 \hat{\tilde{z}}$.

Steps 2–5 are fused into a single kernel: read the up-projection output, apply inverse Hadamard butterfly stages, apply the elementwise nonlinearity, apply forward Hadamard butterfly stages, quantize, write the result. The data is touched once. The two Hadamard transforms add $O(d_{\text{ff}} \log n)$ arithmetic to a kernel that already performs $O(d_{\text{ff}})$ work for the nonlinearity, representing a modest overhead of $\log n$ (approximately 9× for $n = 512$) on the elementwise pass.

This is the only point in the entire layer where the Hadamard domain is exited and re-entered.

### 4.5 Attention

The full attention mechanism operates in the Hadamard domain without exit:

**Projections.** Query, key, and value projections are linear — standard Hadamard-domain matmuls (§4.2). These may be fused into a single QKV projection.

**Scores.** The attention score $\tilde{q}_h^T \tilde{k}_{h,t}$ is an inner product of Hadamard-domain vectors. By orthonormality, this equals the original-domain inner product $q_h^T k_{h,t}$. The $1/\sqrt{d_k}$ scaling is a scalar multiply. Softmax operates over the sequence dimension, which is orthogonal to the feature-space Hadamard.

**Value aggregation.** The weighted sum $\sum_t a_t \tilde{v}_{h,t}$ is a linear combination with scalar (attention) weights. By linearity of the Hadamard transform, this equals $\mathcal{H}(\sum_t a_t v_{h,t})$ — the result remains in the Hadamard domain.

**Output projection.** The concatenation of per-head outputs, followed by the output projection matrix, is a standard Hadamard-domain matmul.

No operation in the attention mechanism requires a domain exit. The cached keys and values are stored in the Hadamard domain.

---

## 5. Tensor Parallelism

The block-diagonal Hadamard $\mathcal{H}_d = \text{diag}(H_n, \ldots, H_n)$ is compatible with tensor parallelism provided the shard dimension is divisible by the block size:

$$n \mid (d / T)$$

where $T$ is the parallelism degree. Under this condition, each shard's Hadamard is a contiguous subset of the block-diagonal — the transform is shard-local with no cross-device communication.

All inter-device communication in tensor-parallel transformers occurs through all-reduce operations (summing partial matmul outputs across shards). All-reduce computes $\sum_p y_p$, which commutes with the Hadamard transform by linearity:

$$\mathcal{H}\!\left(\sum_p y_p\right) = \sum_p \mathcal{H}\, y_p$$

The all-reduce operates on Hadamard-domain partial sums. The communication pattern, message sizes, and synchronization points are identical to standard tensor parallelism. No modification to the communication topology or schedule is required.

For the norm computation under tensor parallelism: each shard computes its local partial norm $\|\tilde{x}_p\|^2 = \|x_p\|^2$ (by shard-local orthonormality), and the partial norms are summed via all-reduce. This is identical to the standard-domain procedure.

The divisibility condition $n \mid (d/T)$ is naturally satisfied for power-of-two block sizes, power-of-two (or power-of-two-divisible) model dimensions, and power-of-two parallelism degrees — which is the universal case in practice.

---

## 6. Memory Layout and Hardware Interface

### 6.1 Weight Storage

Each entry of the quantized weight matrix $\tilde{W}$ is a $b$-bit integer representing a codebook index. These integers are stored in a packed format (e.g., two 4-bit values per byte). There is no interleaving of metadata with weight data — the weight tensor is a contiguous block of $b$-bit integers.

This is the simplest possible weight format. Any hardware-specific memory layout transformation (such as VNNI packing for Intel AMX, or tile-major reordering for NVIDIA tensor cores) is applied as a permutation of the stored integers at model load time. Because there is no per-block metadata that must track with specific weight elements, the packing is a pure physical-layer optimization that does not interact with the quantization scheme. The kernel loads packed integers, unpacks them, and feeds them to the arithmetic units. No scale lookup, no metadata association, no conditional logic based on block boundaries.

### 6.2 Activation Handling

Activations flow through the network as Hadamard-domain values. Between layers, they are requantized to $b$ bits using the universal codebook. The requantization is a fixed-function operation (compare against precomputed boundaries, select the nearest level) that can be fused into the matmul epilogue or the subsequent layer's prologue.

For stochastic quantization: the only additional requirement is a source of random bits. One random bit per component per quantization event determines whether to round up or down. This can be generated from a PRNG seeded per-layer, adding negligible cost.

---

## 7. Error Analysis

### 7.1 Per-Layer Error

The quantization introduces error at two points in each matmul: the weight quantization (offline) and the activation quantization (online). For a matmul $y = Wx$ with quantized $\tilde{W}$ and $\hat{\tilde{x}}$:

$$\tilde{W}\hat{\tilde{x}} = (\mathcal{H}W\mathcal{H}^T + E_W)(\mathcal{H}x + \epsilon_x)$$

$$= \mathcal{H}Wx + E_W \mathcal{H}x + \mathcal{H}W\mathcal{H}^T \epsilon_x + E_W \epsilon_x$$

The first-order error terms are $E_W \mathcal{H}x$ (weight noise) and $\mathcal{H}W\mathcal{H}^T \epsilon_x$ (activation noise). The second-order term $E_W \epsilon_x$ is negligible.

Because the Hadamard has isotropized both the weights and the activations, the entries of $E_W$ and $\epsilon_x$ are approximately independent with common variance $\delta^2(b)$. The cross terms do not constructively interfere — their expected energy distributes uniformly across output dimensions.

The per-component output MSE is approximately:

$$\text{MSE}_{\text{output}} \approx K \cdot \delta^2_W \cdot \frac{\|x\|^2}{K} + K \cdot \frac{\|W\|_F^2}{MK} \cdot \delta^2_x = \delta^2_W \|x\|^2/K \cdot K + \frac{\|W\|_F^2}{M} \delta^2_x$$

$$= \delta^2_W \|x\|^2 + \frac{\|W\|_F^2}{M} \delta^2_x$$

Both terms are proportional to the quantization noise $\delta^2$ and the signal energy ($\|x\|^2$ or $\|W\|_F^2/M$), giving an SQNR that depends only on the bit depth, not on distributional properties of the data.

### 7.2 Depth Accumulation

Through a transformer with $L$ layers, the quantization errors accumulate additively through the residual stream. Each layer contributes an independent error (the Hadamard isotropization ensures errors across layers are approximately independent in direction). RMSNorm resets the magnitude at each sublayer, preventing exponential amplification.

The total error after $L$ layers scales as $O(\sqrt{L})$ in RMS (by independence) rather than $O(L)$ (worst-case additive) or $O(\sigma^L)$ (exponential). For $L = 32$ and per-layer SQNR of 22 dB, the end-to-end SQNR degrades to approximately $22 - 10\log_{10}(\sqrt{32}) \approx 22 - 7.5 = 14.5$ dB, which remains in the regime where model quality is empirically preserved.

### 7.3 Stochastic Quantization Benefit

With stochastic quantization of activations, the error $\epsilon_x$ is zero-mean with independent components. For the attention score $s = \hat{\tilde{q}}^T \hat{\tilde{k}}$, the error is:

$$\hat{s} - s = \epsilon_q^T \tilde{k} + \tilde{q}^T \epsilon_k + \epsilon_q^T \epsilon_k$$

The first two terms are sums of $d_k$ independent zero-mean random variables, each bounded by $O(\delta \cdot \|k\|_\infty)$ and $O(\delta \cdot \|q\|_\infty)$ respectively. Post-Hadamard, $\|k\|_\infty \approx \|k\| / \sqrt{d_k}$ (energy is spread uniformly), so each term concentrates as:

$$|\epsilon_q^T \tilde{k}| = O\!\left(\delta \cdot \|q\| \cdot \|k\| / \sqrt{d_k}\right)$$

The relative error in the attention score is $O(\delta / \sqrt{d_k})$, which shrinks with head dimension. For $d_k = 128$ and $\delta \sim 0.01$, this is approximately $0.1\%$ — negligible relative to the softmax's temperature.

---

## 8. What This Eliminates

The following components of standard quantized inference pipelines are entirely absent from the proposed scheme:

**Per-channel weight scales.** Not needed. The universal codebook replaces all per-channel dynamic range adaptation.

**Per-token activation scales.** Not needed. The Hadamard transform normalizes the activation distribution, making a fixed quantization grid optimal for all tokens.

**Online absmax reductions.** Not needed. The activation quantization uses precomputed, fixed boundaries. No data-dependent reduction precedes the matmul.

**Outlier decomposition.** Not needed. The Hadamard spreads outlier energy uniformly across all components. There are no outliers in the Hadamard domain.

**Calibration data.** Not needed for quantization itself. The codebook is derived from the Beta distribution, which is determined by the block size, not by the data. (Calibration may still be useful for mixed-precision bit allocation refinement, but is not required for the basic scheme.)

**Per-block metadata in weight storage.** Not needed. Weights are plain integer arrays with no interleaved scales or offsets.

**Scale arithmetic in the matmul epilogue.** Not needed. The matmul output requires no per-element correction. The only post-matmul operation is the RMSNorm scalar (§4.3), which is a single multiply applied uniformly to the entire output vector.

**SmoothQuant-style channel migration.** Not needed. SmoothQuant redistributes outlier energy from activations to weights via a per-channel diagonal scaling. The Hadamard achieves the same redistribution but uniformly and optimally, without a calibration-dependent diagonal that must be tuned per model.

---

## 9. Implementation Summary

### 9.1 Offline (Once Per Model)

1. For each normalization layer, absorb $\gamma$ into the subsequent weight matrix: $W' = W \text{diag}(\gamma)$.
2. For each weight matrix, compute $\mathcal{H}_M W' \mathcal{H}_K^T$ via block-diagonal fast Walsh-Hadamard transforms.
3. Quantize each entry to the nearest codebook level. Store as packed $b$-bit integers.
4. Compute $\|W'\|_F$ for each matrix (for optional mixed-precision bit allocation).
5. Apply any hardware-specific memory layout transformation (VNNI packing, tile reordering) to the stored integer arrays.

### 9.2 Online (Per Token)

**At each linear layer:**
1. Hadamard-transform the input activation: $\tilde{x} = \mathcal{H}_K x$. Fused butterfly operations, $O(K \log n)$.
2. Quantize: $\hat{\tilde{x}} = Q_b(\tilde{x})$. Fixed-boundary comparison, $O(K)$.
3. Launch matmul: $\tilde{y} = \tilde{W}' \hat{\tilde{x}}$. Integer arithmetic, $O(MK)$.
4. Simultaneously compute norm: $\|\tilde{x}\|^2$. Reduction, $O(K)$, hidden behind matmul latency.
5. Apply norm correction to matmul output: $\tilde{y} / \text{rms}(\tilde{x})$. Scalar multiply, $O(M)$.

**At the MLP nonlinearity:**
6. Inverse Hadamard of up-projection output: $h = \mathcal{H}^T \tilde{h}$. $O(d_{\text{ff}} \log n)$.
7. Apply activation function: $\phi(h)$. $O(d_{\text{ff}})$.
8. Forward Hadamard: $\widetilde{\phi(h)} = \mathcal{H}\, \phi(h)$. $O(d_{\text{ff}} \log n)$.
9. Quantize and proceed to down-projection.

Steps 6–9 fuse into a single kernel with one read and one write of the intermediate activation.

**At attention:**
10. QKV projection (Hadamard-domain matmul).
11. Scores via inner products in Hadamard domain (identical values to original domain).
12. Softmax (domain-independent).
13. Value aggregation (linear combination in Hadamard domain).
14. Output projection (Hadamard-domain matmul).

No domain exit within attention.

### 9.3 Domain Exits Per Layer

One. At the MLP nonlinearity. All other operations — five matmuls, two norms, two residual additions, attention scoring, attention weighting — remain in the Hadamard domain.

---

## 10. Discussion

The strategy presented here is not a collection of independent optimizations but a single consistent application of one principle: rotate into a basis where the statistics are known, then exploit the known statistics everywhere.

The Hadamard transform is the rotation. The Beta distribution is the known statistics. The Lloyd-Max codebook is the exploitation. Every design decision — $\gamma$ absorption, norm-matmul parallelism, stochastic activation quantization, the Frobenius-norm SQNR predictor — is a consequence of operating in a domain where the distribution is uniform and known. In the standard basis, each of these would require a separate mechanism with separate assumptions. In the Hadamard basis, they are all corollaries of the same distributional guarantee.

The constraints of the approach are clear and narrow. The block size must divide the tensor dimensions (satisfied universally in practice). The nonlinear activations force one domain exit per layer (unavoidable without approximation). The codebook is optimal only in the log-concave regime of the Beta distribution (guaranteed for block sizes $\geq 64$). Within these constraints, the scheme provides tight, data-independent error bounds with minimal implementation complexity.

The resulting system has fewer moving parts than any existing quantization pipeline of comparable quality. It requires no calibration, no per-model tuning, no outlier handling, and no metadata beyond a single global codebook. The kernel is simpler (no epilogue scale correction), the memory layout is simpler (plain integer arrays), and the error analysis is simpler (one known distribution, one known codebook, one known MSE).

---

## Appendix A: Codebook Computation

The Lloyd-Max codebook for Beta$(\alpha, \beta)$ at $b$ bits is computed by the following iterative procedure:

1. Initialize $2^b$ reconstruction levels uniformly on $[0, 1]$.
2. Compute decision boundaries: $b_i = (c_i + c_{i+1})/2$.
3. Update reconstruction levels: $c_i = \mathbb{E}[X \mid b_{i-1} \leq X < b_i] = \int_{b_{i-1}}^{b_i} x \, f(x)\, dx \,\Big/\, \int_{b_{i-1}}^{b_i} f(x)\, dx$.
4. Repeat until convergence.

For Beta$(\alpha, \beta)$, the integrals in step 3 involve the regularized incomplete beta function, computable via standard numerical libraries. Convergence is typically achieved in 10–20 iterations. The computation runs once per (block size, bit depth) pair and produces a lookup table of $2^b$ values.

For practical use, an equivalent approach is to use the Gaussian codebook (tabulated for all common bit depths) with a first-order correction from the third and fourth cumulants of the Beta distribution. The correction is $O(1/n)$ in magnitude and can be ignored for $n \geq 256$ with negligible impact on MSE.

## Appendix B: Fast Walsh-Hadamard Transform

The $n \times n$ Hadamard matrix (for $n = 2^s$) is applied via $s = \log_2 n$ stages of butterfly operations. At stage $k$ ($k = 0, \ldots, s-1$), the vector is partitioned into pairs of elements at stride $2^k$, and each pair $(a, b)$ is replaced by $(a + b, \; a - b)$. The final result is scaled by $1/\sqrt{n}$.

Each stage touches all $n$ elements once. The total cost is $n \cdot s = n \log_2 n$ additions (no multiplications except the final scaling). For $n = 512$: 4608 additions per block, or approximately 9 additions per element. On hardware with fused multiply-add units, this is fewer operations than a single dot product of length 16.

The inverse transform is identical to the forward transform (up to scaling), since $H = H^T = H^{-1}$ for the normalized Hadamard. The domain exit and re-entry at the MLP nonlinearity thus use the same code path.
