# Hadamard Arithmetic Reference

## Mathematical Identities for Block-Hadamard Transforms in Transformer Algebra

---

# I. Foundations

## 1.1 The Hadamard Matrix

Let $H_n$ denote the $n \times n$ normalized Hadamard matrix. Its entries are
$\pm 1/\sqrt{n}$, and it satisfies

$$H_n^T H_n = H_n H_n^T = I_n,$$

$$H_n^{-1} = H_n^T = H_n.$$

For $n$ a power of two, $H_n x$ can be applied in $O(n \log n)$ operations via
the fast Walsh-Hadamard transform.

## 1.2 Block-Diagonal Extension

For dimension $d = n \cdot B$, define the block-diagonal Hadamard operator

$$\mathcal{H}_d = \operatorname{diag}(\underbrace{H_n, H_n, \ldots, H_n}_{B \text{ blocks}}).$$

Then $\mathcal{H}_d$ is orthonormal:

$$\mathcal{H}_d^T \mathcal{H}_d = I_d.$$

It is well-defined if and only if $n \mid d$.

## 1.3 Deterministic Coordinate Bounds

For any $x \in \mathbb{R}^n$ and any coordinate $i$,

$$|(H_n x)_i| = |\langle h_i, x \rangle| \le \|h_i\|_2 \, \|x\|_2 = \|x\|_2,$$

where $h_i$ is the $i$-th row of $H_n$.

Since each row has entries $\pm 1/\sqrt{n}$,

$$|(H_n x)_i| \le \frac{\|x\|_1}{\sqrt{n}}.$$

These bounds apply blockwise to $\mathcal{H}_d$.

## 1.4 Parseval Identities

For any orthonormal $H$ and any vectors $u, v$,

$$\|Hu\|_2 = \|u\|_2,$$

$$\langle Hu, Hv \rangle = \langle u, v \rangle,$$

$$\|Hu - Hv\|_2 = \|u - v\|_2.$$

For a block-diagonal Hadamard $\mathcal{H}_d$, the same statements hold with
$H$ replaced by $\mathcal{H}_d$.

## 1.5 MSE Invariance Under Orthogonal Reconstruction

Let $\tilde{x} = Hx$, and let $\hat{\tilde{x}} \in \mathbb{R}^n$ be any
approximation to $\tilde{x}$. Then

$$\|x - H^T \hat{\tilde{x}}\|_2^2 = \|Hx - \hat{\tilde{x}}\|_2^2.$$

Thus approximation error in transformed coordinates equals reconstruction error
in original coordinates.

---

# II. Quantized Transformed Coordinates

## 2.1 Coordinatewise Quantization-Reconstruction Operator

Let

$$\mathcal{Q} : \mathbb{R}^d \to \mathbb{R}^d$$

be any coordinatewise quantization-reconstruction operator. No specific codebook
or scale rule is assumed in this document.

## 2.2 Activation Encoding

Given $x \in \mathbb{R}^K$, define its transformed-coordinate encoding by

$$\hat{\tilde{x}} = \mathcal{Q}(\mathcal{H}_K x).$$

The corresponding reconstruction in original coordinates is

$$\hat{x} = \mathcal{H}_K^T \hat{\tilde{x}}.$$

## 2.3 Weight Encoding

Given $W \in \mathbb{R}^{M \times K}$, define its two-sided transformed form

$$\tilde{W} = \mathcal{H}_M W \mathcal{H}_K^T.$$

Its quantized transformed representation is

$$\hat{\tilde{W}} = \mathcal{Q}(\tilde{W}),$$

where $\mathcal{Q}$ acts entrywise.

## 2.4 Entry and Exit Maps

Define the transformed-coordinate entry map

$$\mathcal{E}(x) = \mathcal{Q}(\mathcal{H}x),$$

and the exact orthogonal exit map

$$\mathcal{R}(\tilde{x}) = \mathcal{H}^T \tilde{x}.$$

When $\tilde{x}$ is already reconstructed in $\mathbb{R}^d$, $\mathcal{R}$ is
just the inverse Hadamard transform.

---

# III. Linear Algebra in Hadamard Coordinates

## 3.1 Two-Sided Linear Identity

Let $y = Wx$, where $W \in \mathbb{R}^{M \times K}$ and $x \in \mathbb{R}^K$.
Then

$$\mathcal{H}_M y = (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x).$$

Proof:

$$\mathcal{H}_M W x
= \mathcal{H}_M W (\mathcal{H}_K^T \mathcal{H}_K) x
= (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x).$$

## 3.2 Single-Sided Linear Identity

If only the contraction dimension is transformed, define

$$W^\sharp = W \mathcal{H}_K^T, \qquad \tilde{x} = \mathcal{H}_K x.$$

Then

$$W^\sharp \tilde{x} = W \mathcal{H}_K^T \mathcal{H}_K x = Wx.$$

Thus single-sided transformed coordinates suffice to recover the original-domain
linear output exactly in the absence of quantization error.

## 3.3 Residual Addition

For $y = x + z$,

$$\mathcal{H}(x + z) = \mathcal{H}x + \mathcal{H}z.$$

Residual addition is therefore preserved by linearity.

## 3.4 Scalar Multiplication

For $\alpha \in \mathbb{R}$,

$$\mathcal{H}(\alpha x) = \alpha \mathcal{H}x.$$

This covers scalar normalization factors such as $1/\sqrt{d_k}$.

## 3.5 Concatenation Across Independent Blocks

If a vector is decomposed as a direct sum

$$x = x_1 \oplus x_2 \oplus \cdots \oplus x_m,$$

and each block carries its own Hadamard transform $H_i$, then

$$\left(\bigoplus_{i=1}^m H_i\right) x
= H_1 x_1 \oplus H_2 x_2 \oplus \cdots \oplus H_m x_m.$$

---

# IV. Attention Identities

## 4.1 Query, Key, and Value Projections

Let

$$q = W_Q x, \qquad k = W_K x, \qquad v = W_V x.$$

Under two-sided transformed coordinates,

$$\tilde{q} = \tilde{W}_Q \tilde{x}, \qquad
\tilde{k} = \tilde{W}_K \tilde{x}, \qquad
\tilde{v} = \tilde{W}_V \tilde{x},$$

where $\tilde{x} = \mathcal{H}x$ and
$\tilde{W}_* = \mathcal{H} W_* \mathcal{H}^T$ with dimensions chosen
appropriately per projection.

## 4.2 Attention Score Invariance

For any per-head query and key vectors $q_h, k_h \in \mathbb{R}^{d_k}$,

$$q_h^T k_h = (H q_h)^T (H k_h).$$

This is a direct application of Parseval:

$$\langle Hq_h, Hk_h \rangle = \langle q_h, k_h \rangle.$$

Consequently,

$$\frac{q_h^T k_h}{\sqrt{d_k}}
= \frac{(H q_h)^T (H k_h)}{\sqrt{d_k}}.$$

## 4.3 Attention-Weighted Sums

Let $a_i \in \mathbb{R}$ be scalar attention weights, and let

$$y_h = \sum_i a_i v_{h,i}.$$

Then by linearity,

$$H y_h = \sum_i a_i H v_{h,i}.$$

Hence

$$\widetilde{y}_h = \sum_i a_i \widetilde{v}_{h,i}.$$

## 4.4 Orthogonal Preprocessing Before Attention

If $R_p$ is any orthogonal map applied before scoring, then

$$\langle R_p q, R_t k \rangle
= \langle H R_p q, H R_t k \rangle.$$

Thus orthogonal feature-space preprocessing composes with Hadamard transforms
without changing inner products.

---

# V. RMS-Based Normalization

## 5.1 RMS Invariance

Define

$$\operatorname{rms}(x) = \sqrt{\frac{1}{d}\|x\|_2^2}.$$

By Parseval,

$$\operatorname{rms}(\mathcal{H}x) = \operatorname{rms}(x).$$

## 5.2 Division by RMS

Let

$$\bar{x} = \frac{x}{\operatorname{rms}(x)}.$$

Then

$$\mathcal{H}\bar{x}
= \frac{\mathcal{H}x}{\operatorname{rms}(\mathcal{H}x)}.$$

## 5.3 Elementwise Gain

For a gain vector $\gamma \in \mathbb{R}^d$,

$$\operatorname{RMSNorm}_\gamma(x)
= \frac{x}{\operatorname{rms}(x)} \odot \gamma
= \operatorname{diag}(\gamma)\frac{x}{\operatorname{rms}(x)}.$$

The Hadamard transform conjugates this diagonal operator into

$$\tilde{\Gamma} = \mathcal{H}\operatorname{diag}(\gamma)\mathcal{H}^T,$$

so that

$$\mathcal{H}\operatorname{RMSNorm}_\gamma(x)
= \tilde{\Gamma}\frac{\mathcal{H}x}{\operatorname{rms}(\mathcal{H}x)}.$$

## 5.4 Gain Absorption Into a Following Linear Map

If a linear map $W$ follows $\operatorname{RMSNorm}_\gamma$, define

$$W' = W \operatorname{diag}(\gamma).$$

Then

$$W \left(\frac{x}{\operatorname{rms}(x)} \odot \gamma\right)
= W' \frac{x}{\operatorname{rms}(x)}.$$

This is an exact reparameterization.

---

# VI. Nonlinearities

## 6.1 Non-Commutativity

For a nonlinear scalar function $\phi : \mathbb{R} \to \mathbb{R}$ applied
coordinatewise, one generally does not have

$$\mathcal{H}\phi(x) = \phi(\mathcal{H}x).$$

## 6.2 Exact Exit-Reentry Identity

If $\tilde{x} = \mathcal{H}x$, then the transformed representation of
$\phi(x)$ is

$$\widetilde{\phi(x)} = \mathcal{H}\phi(\mathcal{H}^T \tilde{x}).$$

This identity is exact and does not assume any approximation.

## 6.3 Two-Layer MLP Composition

For

$$\operatorname{MLP}(x) = W_2 \phi(W_1 x),$$

an exact transformed-coordinate expression is

$$\widetilde{h} = (\mathcal{H}W_1\mathcal{H}^T)(\mathcal{H}x),$$

$$\widetilde{\phi(h)} = \mathcal{H}\phi(\mathcal{H}^T \widetilde{h}),$$

$$\widetilde{y} = (\mathcal{H}W_2\mathcal{H}^T)\widetilde{\phi(h)}.$$

In a single-sided formulation on contraction dimensions,

$$h = (W_1 \mathcal{H}^T)(\mathcal{H}x),$$

$$y = (W_2 \mathcal{H}^T)\mathcal{H}\phi(h).$$

---

# VII. Tensor Parallel and Direct-Sum Structure

## 7.1 Shard Compatibility

Let $d$ be a transformed dimension and let $T$ be the number of shards. If

$$n \mid \frac{d}{T},$$

then the block-Hadamard decomposes across shards:

$$\mathcal{H}_d
= \mathcal{H}_{d/T}^{(1)} \oplus \mathcal{H}_{d/T}^{(2)} \oplus \cdots \oplus \mathcal{H}_{d/T}^{(T)}.$$

## 7.2 Linearity Across Partial Sums

For partial outputs $y_1, \ldots, y_T$,

$$\mathcal{H}\left(\sum_{p=1}^T y_p\right)
= \sum_{p=1}^T \mathcal{H}y_p.$$

## 7.3 Norm Decomposition Across Shards

If $x = x_1 \oplus \cdots \oplus x_T$, then

$$\|x\|_2^2 = \sum_{p=1}^T \|x_p\|_2^2
= \sum_{p=1}^T \|\mathcal{H}x_p\|_2^2.$$

---

# VIII. Structural Classification of Transformer Operations

Let feature-space Hadamard transforms act on contraction dimensions. Then the
standard transformer operations fall into the following mathematical classes:

- Inner products $\langle u, v \rangle$
- Scalar-vector products $\alpha v$
- Vector additions $u + v$
- Coordinatewise nonlinearities $\phi(u)$
- Reductions such as $\sum_i f(u_i)$

The first three are preserved by orthogonality and linearity. Coordinatewise
nonlinearities require the exit-reentry identity of Section VI. Norm reductions
are preserved by Parseval. Sequence-axis softmax acts on an index axis distinct
from the feature axis on which $\mathcal{H}$ acts.

## 8.1 Single-Sided and Two-Sided Representations

Two-sided transformed coordinates represent outputs in transformed form:

$$\mathcal{H}_M y = (\mathcal{H}_M W \mathcal{H}_K^T)(\mathcal{H}_K x).$$

Single-sided transformed coordinates represent the same output in original form:

$$y = (W \mathcal{H}_K^T)(\mathcal{H}_K x).$$

These are distinct representations of the same linear map.

---

# IX. Quantization Error Algebra

## 9.1 Reconstruction Error in Transformed Coordinates

Let

$$\hat{\tilde{x}} = \mathcal{Q}(\tilde{x}), \qquad e_x = \hat{\tilde{x}} - \tilde{x}.$$

Then

$$\|x - \mathcal{H}^T \hat{\tilde{x}}\|_2 = \|e_x\|_2.$$

## 9.2 Two-Sided Linear Perturbation Decomposition

Let

$$\tilde{W} = \mathcal{H}_M W \mathcal{H}_K^T, \qquad
\hat{\tilde{W}} = \tilde{W} + E_W,$$

$$\tilde{x} = \mathcal{H}_K x, \qquad
\hat{\tilde{x}} = \tilde{x} + e_x.$$

Then the transformed output perturbation is

$$\hat{\tilde{y}} - \tilde{y}
= \tilde{W} e_x + E_W \tilde{x} + E_W e_x,$$

where

$$\tilde{y} = \tilde{W}\tilde{x}, \qquad
\hat{\tilde{y}} = \hat{\tilde{W}}\hat{\tilde{x}}.$$

## 9.3 Single-Sided Linear Perturbation Decomposition

Let

$$W^\sharp = W \mathcal{H}_K^T, \qquad \hat{W}^\sharp = W^\sharp + E_W,$$

$$\tilde{x} = \mathcal{H}_K x, \qquad \hat{\tilde{x}} = \tilde{x} + e_x.$$

Then

$$\hat{y} - y
= W^\sharp e_x + E_W \tilde{x} + E_W e_x,$$

with

$$y = Wx, \qquad \hat{y} = \hat{W}^\sharp \hat{\tilde{x}}.$$

Since right multiplication by an orthogonal matrix preserves spectral norm,

$$\|W^\sharp\|_2 = \|W\|_2,$$

and since $\|\tilde{x}\|_2 = \|x\|_2$, one obtains the bound

$$\|\hat{y} - y\|_2
\le \|W\|_2 \, \|e_x\|_2 + \|E_W\|_2 \, \|x\|_2 + \|E_W\|_2 \, \|e_x\|_2.$$

## 9.4 Attention Score Perturbation

Let $\hat{q} = q + e_q$ and $\hat{k} = k + e_k$. Then

$$\hat{q}^T \hat{k} - q^T k
= q^T e_k + e_q^T k + e_q^T e_k.$$

If the approximations are formed in transformed coordinates and reconstructed
orthogonally, the same identity holds after inserting Hadamard transforms, since
inner products are preserved.
