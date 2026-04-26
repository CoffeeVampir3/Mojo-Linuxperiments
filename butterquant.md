# ButterQuant Reference

## Int8 Encoding and Dequantization Identities for Hadamard-Domain Transformer Operators

This document specifies the int8 quantization scheme that operates on top of the Hadamard-domain transformer algebra of the Hadamard Arithmetic Reference (HAR). HAR establishes which transformer operations preserve which Hadamard bases. This document specifies the int8 encoding of activations and weights, the dequantization identities at operator boundaries, and the recipes that match each operator class.

The conventions of HAR are inherited. $\mathcal{H}_d$ is the normalized block-diagonal Hadamard with block size $n$ such that $n \mid d$. $\mathcal{Q}$ is a coordinatewise quantization-reconstruction operator. $\tilde x = \mathcal{H}_d x$.

---

# I. The Symmetric Int8 Quantizer

## 1.1 Definition

For $S > 0$, the symmetric int8 quantizer $Q_S : \mathbb{R} \to \{-128, \ldots, 127\}$ is

$$Q_S(x) = \mathrm{clamp}(\mathrm{round}(127 x / S),\ -128,\ 127).$$

The coordinatewise reconstruction is $\hat x = Q_S(x) \cdot S / 127$.

## 1.2 Reconstruction error

For $|x| \le S$,

$$|\hat x - x| \le S / 254$$

under round-to-nearest-even. For $|x| > S$, the reconstruction saturates at $\mathrm{sign}(x) \cdot S$ and the error is $|x| - S$.

## 1.3 Dynamic absmax

Let $\Omega \subseteq \{1, \ldots, n\}$ be a coordinate support. The dynamic absmax over $\Omega$ for $x \in \mathbb{R}^n$ is

$$S_\Omega(x) = \max_{i \in \Omega} |x_i|.$$

For any $S' < S_\Omega$, at least one coordinate $|x_i|$ with $i \in \Omega$ saturates under $Q_{S'}$. The dynamic absmax is the smallest scale that admits no saturation over $\Omega$.

A positive floor $\epsilon_S$ may be applied: $S = \max(S_\Omega, \epsilon_S)$. The floor is inactive when $S_\Omega > \epsilon_S$ and avoids division by zero when $S_\Omega = 0$.

## 1.4 The u8/i8 affine relation

For $i \in \{-128, \ldots, 127\}$,

$$u = i + 128 = i \oplus \mathtt{0x80} \in \{0, \ldots, 255\}.$$

For $W \in \{-128, \ldots, 127\}^{N \times K}$ and $i \in \{-128, \ldots, 127\}^K$,

$$\sum_k (i_k + 128) \cdot W_{n,k} = \sum_k i_k \cdot W_{n,k} + 128 \cdot \sum_k W_{n,k}.$$

Define the colsum

$$\mathrm{cs}[n] = \sum_k W_{n,k}.$$

Then

$$\sum_k i_k \cdot W_{n,k} = \sum_k (i_k + 128) \cdot W_{n,k} - 128 \cdot \mathrm{cs}[n].$$

The signed-signed dot product is recoverable from the unsigned-signed dot product via subtraction of one per-output scalar. For per-block colsum on block $b$ with support $\Omega_b \subseteq \{1, \ldots, K\}$,

$$\mathrm{cs}[n, b] = \sum_{k \in \Omega_b} W_{n,k}.$$

## 1.5 Hardware operand constraints

Different SIMD/tile dot-product instructions impose different operand-sign requirements:

| Instruction | A operand | B operand |
|---|---|---|
| `vpdpbusd` (AVX-512 VNNI) | u8 | i8 |
| `vpdpbssd` (AVX-VNNI-INT8) | i8 | i8 |
| `tdpbsud` (AMX) | i8 (tile) | u8 (tile) |
| `tdpbssd` (AMX) | i8 (tile) | i8 (tile) |

Where the dot product instruction takes one unsigned operand and that operand encodes signed data via the affine relation of §1.4, the signed-signed inner product is recovered by subtracting $128$ times the sum of the operand that remains signed, taken over the contraction index. The location of this correction depends on which operand is unsigned: in a linear projection where the activation is encoded as $u_8$ and the weight stays $i_8$, the correction is $128 \cdot \sum_k W_{i_8}[n, k]$ (one value per output row, the weight colsum); in attention scoring where the cached K is encoded as $u_8$ and the query stays $i_8$, the correction is $128 \cdot \sum_k q_{i_8}[k]$ (one value per query head per query position, the query int8 sum).

Where the unsigned operand encodes non-negative data directly (without the $+128$ offset, e.g., a u8 attention weight in $[0, 255]$ representing a non-negative softmax product), no such correction is applied. Where the instruction admits a signed-signed form, both operands are consumed as $i_8$ and no correction is applied.

The dequantized identity is the same in all forms. The choice of operand sign at each kernel site is a property of the available instruction and the data range of each operand.

---

# II. Encodings

## 2.1 Per-row activation encoding

For $x \in \mathbb{R}^K$, the per-row encoding stores

$$x_{i_8}[k] = Q_{S_a}(x_k), \qquad S_a = \max_k |x_k|.$$

Output: one i8 vector of length $K$ and one f32 scalar.

## 2.2 Per-block activation encoding

Partition $\{1, \ldots, K\}$ into $K/B$ blocks of size $B$ with $B \mid K$. For block $b$ with support $\Omega_b$,

$$x_{i_8}[k] = Q_{S_a[b]}(x_k) \text{ for } k \in \Omega_b, \qquad S_a[b] = \max_{k \in \Omega_b} |x_k|.$$

Output: one i8 vector of length $K$ and $K/B$ f32 scales.

Since $\Omega_b \subseteq \{1, \ldots, K\}$, $S_a[b] \le S_a$ for every $b$. The per-block grid step $S_a[b] / 127$ is therefore at most as coarse as the per-row grid step $S_a / 127$ over the same coordinates. The relative magnitudes of the per-block grid steps within a row depend on the within-row energy distribution and are not bounded above by a fixed function of $B$ and $K$. The cost is $K/B$ scales per row in place of one.

## 2.3 Weight encodings

For $W \in \mathbb{R}^{N \times K}$, the per-row encoding stores

$$W_{i_8}[n, k] = Q_{S_w[n]}(W_{n, k}), \qquad S_w[n] = \max_k |W_{n, k}|,$$

and the per-block encoding stores

$$W_{i_8}[n, k] = Q_{S_w[n, b]}(W_{n, k}) \text{ for } k \in \Omega_b, \qquad S_w[n, b] = \max_{k \in \Omega_b} |W_{n, k}|.$$

## 2.4 Storage conventions for scales

The stored value of a weight scale is pre-divided by 127:

$$\bar S_w[n] = S_w[n] / 127.$$

The reconstruction $\hat W_{n, k} = W_{i_8}[n, k] \cdot \bar S_w[n]$ is then one multiplication.

The stored value of an activation scale is the raw $S_a$ without pre-division. The consumer applies the $/127$ factor at the dequantization step.

This asymmetry is a storage convention. The dequantized identity is unchanged.

## 2.5 Colsum companion

For each int8 weight consumed by an unsigned-signed GEMV, the colsum companion is computed once after quantization. The colsum's K-axis structure matches the K-block partitioning of the dequantization step, which is determined by the activation scale granularity at the consumer, not by the weight scale granularity:

$$\mathrm{cs}[n] = \sum_k W_{i_8}[n, k] \quad \text{(consumer uses per-row activation scale)},$$

$$\mathrm{cs}[n, b] = \sum_{k \in \Omega_b} W_{i_8}[n, k] \quad \text{(consumer uses per-K-block activation scale)}.$$

A weight with per-row weight scale that feeds a per-block GEMV (per-block activation scale) still requires per-K-block colsums, since the dequantization tail subtracts $128 \cdot \mathrm{cs}[n, b]$ inside the block sum (§7.2).

Kernel sites using a signed-signed dot product produce the same dequantized output without consuming the colsum. The colsum storage is allocated when the int8 weight is allocated; signed-signed paths read the int8 weight without reading the colsum.

---

# III. Hadamard-Rotated Encodings

## 3.1 K-rotated weight, single-sided

For $W \in \mathbb{R}^{N \times K}$, define

$$W^\sharp = W \mathcal{H}_K^T.$$

By HAR §3.2, $W^\sharp \mathcal{H}_K x = W x$ in exact arithmetic.

The per-row or per-block int8 encoding of $W^\sharp$ is computed by applying $\mathcal{H}_K^T$ to the rows of $W$ before quantization. The dequantized matmul output is in the original M-coordinate basis.

## 3.2 K- and M-rotated weight, two-sided

For $W \in \mathbb{R}^{N \times K}$ with $\mathcal{H}_M$ block-diagonal of block size $d$ such that $d \mid N$, define

$$\tilde W = \mathcal{H}_M W \mathcal{H}_K^T.$$

By HAR §3.1, $\tilde W \mathcal{H}_K x = \mathcal{H}_M W x$ in exact arithmetic. The dequantized matmul output is in the M-rotated basis.

The two-sided encoding applies $\mathcal{H}_K^T$ to the rows of $W$ and $\mathcal{H}_M$ to the columns of $W$ (per output-axis block) before quantization.

## 3.3 Equivalence at the K boundary

The runtime activation FWHT for the K axis is identical in single-sided and two-sided forms. The two encodings differ at the M-axis output: single-sided produces $W x$ directly; two-sided produces $\mathcal{H}_M W x$.

The two-sided form moves one runtime FWHT — the M-axis rotation that the next consumer would otherwise perform — into the offline weight encoding. The single-sided form leaves that FWHT at runtime.

## 3.4 Block-size constraint

When the M-axis rotation $\mathcal{H}_M$ has block size $d$, the next consumer that operates in the rotated basis must have a K-axis encoding with the same block size $d$. If the block sizes differ, the runtime must apply a basis-change FWHT, which is the same FWHT the single-sided encoding would have placed at runtime; the offline rotation is then redundant.

For per-head consumers (where the next operation is structured per head of dimension $d_{\text{head}}$), $d = d_{\text{head}}$.

## 3.5 Effect on the per-row absmax

By Parseval, $\|H_n x\|_2 = \|x\|_2$. The per-coordinate distribution differs in general. By HAR §1.3, $|(H_n x)_i| \le \|x\|_2$ and $|(H_n x)_i| \le \|x\|_1 / \sqrt n$.

The relationship between the rotated absmax $\max_k |(H_n x)_k|$ and the original absmax $\max_k |x_k|$ is not monotone. For $x$ aligned with a single coordinate ($x = c \cdot e_i$), the rotated absmax is $|c|/\sqrt n$, smaller than the original $|c|$ by factor $\sqrt n$. For $x$ aligned with one row of $H_n$ ($x = c \cdot h_i^T$), the rotated absmax is $|c|$, larger than the original $|c|/\sqrt n$ by factor $\sqrt n$. For inputs in between, the rotated absmax falls in between.

The rotation does not eliminate per-row magnitude variance across rows. Per-row dynamic scales handle inter-row variance independently of the rotation.

---

# IV. Gain Split

## 4.1 Decomposition

For $\gamma \in \mathbb{R}^d$ and $\gamma_k \ne 0$,

$$\gamma_k = \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|} \cdot \sqrt{|\gamma_k|}.$$

For $\gamma_k = 0$, the right-hand side is $0$ when $\mathrm{sign}(0)$ is taken as $0$.

## 4.2 Application to RMSNorm followed by linear map

Let $W \in \mathbb{R}^{N \times K}$ follow $\mathrm{RMSNorm}_\gamma$ in the operator chain $W \cdot \mathrm{RMSNorm}_\gamma(x)$. Define

$$W'_{n, k} = W_{n, k} \cdot \sqrt{|\gamma_k|},$$

$$x'_k = \frac{x_k}{\mathrm{rms}(x)} \cdot \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|}.$$

Then

$$\sum_k W'_{n, k} \cdot x'_k = \sum_k W_{n, k} \cdot \frac{x_k}{\mathrm{rms}(x)} \cdot \gamma_k = W \cdot \mathrm{RMSNorm}_\gamma(x).$$

In exact arithmetic, the split factorization equals absorption (HAR §5.4) with $W'' = W \mathrm{diag}(\gamma)$ and no runtime gain. The split is a reparameterization of absorption.

## 4.3 Effect on quantization grids

Under absorption, the offline quantizer $Q_{S_{w''}}$ for $W'' = W \mathrm{diag}(\gamma)$ sees per-row dynamic ranges scaled by $\max_k |\gamma_k|$ relative to $W$. The runtime quantizer sees $x / \mathrm{rms}(x)$ with no gain factor.

Under split, the offline quantizer for $W' = W \mathrm{diag}(\sqrt{|\gamma|})$ sees per-row dynamic ranges scaled by $\max_k \sqrt{|\gamma_k|}$. The runtime quantizer sees $x / \mathrm{rms}(x) \cdot \mathrm{sign}(\gamma) \sqrt{|\gamma|}$, scaled by $\max_k \sqrt{|\gamma_k|}$.

The split distributes $\sqrt{|\gamma|}$ symmetrically across the two quantization grids. Absorption concentrates the full $|\gamma|$ on the offline grid.

## 4.4 Stability floor

The runtime activation factor may be evaluated as

$$\sigma_\gamma[k] = \mathrm{sign}(\gamma_k) \sqrt{\max(|\gamma_k|, \epsilon_\gamma)}.$$

For $|\gamma_k| > \epsilon_\gamma$, $\sigma_\gamma[k] = \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|}$ as in §4.1, and the offline-runtime product equals $\gamma_k$.

For $|\gamma_k| \le \epsilon_\gamma$, the product is $\sqrt{|\gamma_k| \cdot \epsilon_\gamma}$ when the offline factor uses $\sqrt{|\gamma_k|}$ without floor. The mismatch is bounded by $\sqrt{\epsilon_\gamma \cdot \max(|\gamma_k|, \epsilon_\gamma)}$ and vanishes at $\gamma_k = 0$ when $\mathrm{sign}(0) = 0$.

## 4.5 Availability

The split is defined when an RMSNorm with gain $\gamma$ immediately precedes a linear map. The decomposition has no meaning otherwise: when no preceding RMSNorm exists, no $\gamma$ is available to split, and the linear map's weight encoding is the un-absorbed form.

---

# V. Single-Sided and Two-Sided Weight Forms

## 5.1 Two encodings of one linear map

For $y = Wx$ with $W \in \mathbb{R}^{N \times K}$, two int8 encodings are available:

- Single-sided: $W^\sharp = W \mathcal{H}_K^T$, dequantized output $y$ in original basis.
- Two-sided: $\tilde W = \mathcal{H}_M W \mathcal{H}_K^T$, dequantized output $\mathcal{H}_M y$ in M-rotated basis.

Both are exact reformulations of $y = Wx$. The runtime cost of the K-axis FWHT on the activation is identical. The output basis differs.

## 5.2 Operations preserving the M-rotated basis

The following operations on $y$ commute with the M-axis Hadamard $\mathcal{H}_M$ and preserve the M-rotated basis through to their output:

- Scalar multiplication: $\mathcal{H}_M(\alpha y) = \alpha \mathcal{H}_M y$.
- Vector addition with another vector in the same M-rotated basis: $\mathcal{H}_M(y + z) = \mathcal{H}_M y + \mathcal{H}_M z$.
- Inner product with a vector in the same M-rotated basis: $\langle \mathcal{H}_M y, \mathcal{H}_M z \rangle = \langle y, z \rangle$.
- Sum over an index axis disjoint from the M axis: $\mathcal{H}_M \sum_i y_i = \sum_i \mathcal{H}_M y_i$.
- A subsequent linear map $W_2$ with K-axis encoding rotated by the same $\mathcal{H}_M$: $W_2^\sharp = W_2 \mathcal{H}_M^T$ accepts $\mathcal{H}_M y$ as its rotated input.

## 5.3 Operations not preserving the M-rotated basis

The following operations do not in general commute with $\mathcal{H}_M$:

- Coordinatewise nonlinearities $\phi(y)$: HAR §6.1, $\mathcal{H}_M \phi(y) \ne \phi(\mathcal{H}_M y)$ in general.
- Coordinatewise gain $\mathrm{diag}(\beta) y$ when $\beta$ is not constant per Hadamard block: $\mathcal{H}_M \mathrm{diag}(\beta) \mathcal{H}_M^T$ is dense.
- Position-dependent rotations applied in the original coordinate basis (e.g., RoPE on a partial-RoPE layout): $\mathcal{H}_M R_p \mathcal{H}_M^T$ is dense.
- Reductions whose statistic is not norm-based.
- Residual addition into a vector in the original basis.
- Vocabulary logits.

After such an operation, a downstream linear map's encoding starts from a fresh K-axis rotation if int8 is to be re-entered.

## 5.4 Legality of the two-sided form

The two-sided form for $W$ is legal — meaning the encoding requires no basis-change FWHT beyond what is already in the operator chain — if and only if every operation between $W$ and the next operation requiring original coordinates is in the class of §5.2.

When the two-sided form is legal, the M-axis FWHT is applied once offline at weight quantization, in place of one runtime FWHT per token after the matmul output. When the two-sided form is not legal, using it requires inserting an explicit inverse FWHT before the next operation that requires original coordinates.

The single-sided form requires no basis-change FWHT in either case; it leaves the M-axis FWHT at runtime when one is required by the next consumer, and inserts no FWHT when one is not.

## 5.5 Specific operator chains

The following classifications follow from §5.2 and §5.3 applied to standard transformer block structures.

V projection, then V-aggregation, then O projection. V-aggregation $\sum_t a_t v_{g, t}$ is in §5.2 (scalar multiplication and sum over a disjoint axis). The O projection consumes its input on a K axis that can be K-rotated to match V's M-axis rotation. The two-sided form for V is available with $\mathcal{H}_M$ block size $d_{\text{head}}$ and is legal in the §5.4 sense (no additional basis conversion is required). The V cache stores $\tilde v$ in the rotated basis. The V-aggregation kernel applies no FWHT at runtime. The O projection is single-sided with K-axis block size $d_{\text{head}}$.

Q projection, then full-vector RMSNorm, then per-head gain, then RoPE, then per-head FWHT, then quantization. The full-vector RMS denominator $\alpha_Q = 1/\sqrt{\|q\|_2^2/D_Q + \epsilon}$ is norm-based and is preserved by Parseval (HAR §10.2): the same scalar $\alpha_Q$ can be collected from $q$ in any orthonormal basis. The per-head gain $\gamma_Q$ and the RoPE rotation $R_p$ applied in the original coordinate basis do not commute with the per-head Hadamard $H_{d_{\text{head}}}$ (HAR §10.1): $H_{d_{\text{head}}} \mathrm{diag}(\gamma_Q) H_{d_{\text{head}}}^T$ and $H_{d_{\text{head}}} R_p H_{d_{\text{head}}}^T$ are dense in general. A two-sided Q encoding with $\mathcal{H}_M = \bigoplus H_{d_{\text{head}}}$ would produce projection output in the per-head rotated basis; applying $\gamma_Q$ and $R_p$ then requires an explicit basis conversion (one inverse FWHT per head) before the gain and a re-entry rotation (one forward FWHT per head) after RoPE. The single-sided encoding places one runtime FWHT per head after $R_p$; the two-sided encoding would place two. K projection is identical.

FFN gate and up projections, then $\mathrm{silu}(\text{gate}) \odot \text{up}$, then down projection. The pointwise nonlinearity is in §5.3. A two-sided encoding of gate or up would produce output in a rotated basis and require an inverse FWHT before the nonlinearity. The single-sided encoding of gate and up places the K-axis FWHT in the offline weight only; the activation feeding the down projection is freshly quantized after the nonlinearity (a runtime FWHT entry into a per-block rotated basis). The down projection is single-sided because its output enters the residual stream in the original basis (residual addition is in §5.2 only when both operands share a basis, and the pre-existing residual is in original).

LM head. The matmul output is vocabulary logits, consumed in the vocabulary coordinate basis. A two-sided encoding would produce logits in a rotated basis and require an inverse FWHT before sampling. The single-sided encoding produces logits directly in the sampler's basis.

---

# VI. Per-Row GEMV Dequantization

## 6.1 Inputs

For one activation row $i \in \mathbb{R}^K$ and a per-row int8 weight encoding of $W \in \mathbb{R}^{N \times K}$:

| Quantity | Type | Shape | Storage |
|---|---|---|---|
| $x_{i_8}$ | int8 | $(K,)$ | activation int8 |
| $S_a$ | f32 | $()$ | activation absmax (raw) |
| $W_{i_8}$ | int8 | $(N, K)$ | weight int8 |
| $\bar S_w$ | f32 | $(N,)$ | $S_w / 127$ |
| $\mathrm{cs}$ | f32 | $(N,)$ | per-row weight colsum |

## 6.2 Unsigned-signed form

Define the unsigned-signed dot product

$$r[n] = \sum_k (x_{i_8}[k] + 128) \cdot W_{i_8}[n, k].$$

The dequantized matmul output is

$$\hat y[n] = (r[n] - 128 \cdot \mathrm{cs}[n]) \cdot \frac{S_a}{127} \cdot \bar S_w[n].$$

In exact arithmetic and absent quantization error, $\hat y[n] = (Wx)[n]$ for the K-rotated reformulation of §3.1.

## 6.3 Signed-signed form

Define

$$r'[n] = \sum_k x_{i_8}[k] \cdot W_{i_8}[n, k].$$

The dequantized output is

$$\hat y[n] = r'[n] \cdot \frac{S_a}{127} \cdot \bar S_w[n].$$

The colsum is not consumed.

## 6.4 Equivalence

In exact arithmetic, the two forms produce identical $\hat y[n]$. The choice is determined by the available dot-product instruction at the kernel site.

---

# VII. Per-Block GEMV Dequantization

## 7.1 Inputs

For one activation row with per-block scales over $K/B$ blocks of size $B$, and a per-row or per-block weight encoding:

| Quantity | Type | Shape | Storage |
|---|---|---|---|
| $x_{i_8}$ | int8 | $(K,)$ | activation int8 |
| $S_a$ | f32 | $(K/B,)$ | per-block activation absmax (raw) |
| $W_{i_8}$ | int8 | $(N, K)$ | weight int8 |
| $\bar S_w$ | f32 | $(N,)$ or $(N, K/B)$ | $S_w / 127$ |
| $\mathrm{cs}$ | f32 | $(N, K/B)$ | per-block weight colsum |

## 7.2 Unsigned-signed form, per-row weight scale

For each K-block $b$, define

$$r[n, b] = \sum_{k \in \Omega_b} (x_{i_8}[k] + 128) \cdot W_{i_8}[n, k].$$

The dequantized output is

$$\hat y[n] = \bar S_w[n] \cdot \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127}.$$

## 7.3 Unsigned-signed form, per-block weight scale

When the weight scale is per-block,

$$\hat y[n] = \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127} \cdot \bar S_w[n, b].$$

## 7.4 Signed-signed form

The signed-signed instantiation drops the colsum subtraction in the inner sum:

$$r'[n, b] = \sum_{k \in \Omega_b} x_{i_8}[k] \cdot W_{i_8}[n, k],$$

$$\hat y[n] = \bar S_w[n] \cdot \sum_b r'[n, b] \cdot \frac{S_a[b]}{127}$$

(per-row weight scale form; the per-block weight scale form analogous to §7.3 is obtained by multiplying inside the sum).

## 7.5 Output scale folding

The per-block GEMV admits a per-call output scalar $\beta \in \mathbb{R}$:

$$\hat y[n] = \beta \cdot \bar S_w[n] \cdot \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127}.$$

The scalar $\beta$ folds into the bf16 cast at the kernel output. This admits per-call scaling factors such as a routing weight in MoE down projections.

---

# VIII. KV Cache Encoding

## 8.1 Cache contents

For one cached position $t$ at KV head $g$:

| Quantity | Type | Storage |
|---|---|---|
| $K_{i_8}[g, t]$ | int8 | $(d_{\text{head}},)$ in u8 or i8 form |
| $V_{i_8}[g, t]$ | int8 | $(d_{\text{head}},)$ in i8 form |
| $K_{\text{scale}}[g, t]$ | f32 | $()$ |
| $V_{\text{scale}}[g, t]$ | f32 | $()$ |

## 8.2 K cache content

The K cache stores the per-head FWHT of the post-RoPE per-head normalized K vector:

$$\tilde k_{g, t} = H_{d_{\text{head}}} \, R_t \, (\gamma_K \odot \alpha_K \cdot k_{g, t})$$

where $R_t$ is RoPE at position $t$, $\gamma_K$ is the per-head K norm gain, and $\alpha_K$ is the full-vector K inverse RMS scalar. Then

$$K_{i_8}[g, t, k] = Q_{S_K[g, t]}(\tilde k_{g, t, k}), \qquad K_{\text{scale}}[g, t] = S_K[g, t] = \max_k |\tilde k_{g, t, k}|.$$

The cache may store $K_{i_8}$ in i8 form or in u8 form ($K_{i_8} \oplus \mathtt{0x80}$). The choice is determined by the operand convention of the score dot instruction. The dequantized score is identical under both storage choices.

## 8.3 V cache content, single-sided V

When the V projection uses the single-sided encoding (§3.1), the V cache stores the per-head FWHT of the projected V:

$$\tilde v_{g, t} = H_{d_{\text{head}}} v_{g, t},$$

$$V_{i_8}[g, t, k] = Q_{S_V[g, t]}(\tilde v_{g, t, k}), \qquad V_{\text{scale}}[g, t] = S_V[g, t] = \max_k |\tilde v_{g, t, k}|.$$

The per-head FWHT is applied at cache write time.

## 8.4 V cache content, two-sided V

When the V projection uses the two-sided encoding (§3.2), the projection output $v_{g, t}$ is already in the head-rotated basis. The cache stores it directly:

$$V_{i_8}[g, t, k] = Q_{S_V[g, t]}(v_{g, t, k}), \qquad V_{\text{scale}}[g, t] = \max_k |v_{g, t, k}|.$$

No FWHT is applied at cache write time. The cache layout, scale storage, and downstream V-aggregation are identical to §8.3.

## 8.5 Storage layout

The cache is indexed by position group of $W$ consecutive positions per group, where $W$ is the i32 SIMD lane count of the score dot instruction. $B_{\text{vnni}}$ is the inner VNNI dot width (4 for current AVX-512 and AMX VNNI).

K layout per head per pos_group:

$$[d_{\text{head}} / B_{\text{vnni}}] \times [W \times B_{\text{vnni}}] \quad \text{u8 or i8 bytes}.$$

Each $W \times B_{\text{vnni}}$ tile holds $W$ positions × $B_{\text{vnni}}$ K-axis values.

V layout per head per pos_group:

$$[d_{\text{head}} / W] \times [W / B_{\text{vnni}}] \times [W \times B_{\text{vnni}}] \quad \text{i8 bytes}.$$

The V layout is transposed relative to K: $W$ is the channel axis (the lane index of the V-aggregation dot), and $B_{\text{vnni}}$ is the position-within-group axis (the inner VNNI axis).

The total bytes per pos_group are $d_{\text{head}} \cdot W$ for both K and V.

---

# IX. Q Preparation

## 9.1 Recipe

Per query head $h$ at position $p$, with optional per-head gain $\gamma_Q$ and optional full-vector inverse RMS scalar $\alpha_Q$:

1. Load bf16 query head into f32 SIMD registers.
2. Apply gain and inverse RMS: $q_k \leftarrow q_k \cdot \alpha_Q \cdot \gamma_Q[k]$.
3. Apply RoPE in the rotary prefix.
4. Apply $H_{d_{\text{head}}}$ in-register.
5. $S_Q[h, p] = \max_k |q_k|$.
6. $q_{i_8}[h, p, k] = Q_{S_Q[h, p]}(q_k)$.

Steps 2 and 3 are skipped if QK normalization or RoPE is absent in the architecture.

## 9.2 Stored auxiliary scalars

Per query head per position:

$$b_q[h, p] = 128 \cdot \sum_k q_{i_8}[h, p, k],$$

$$f_Q[h, p] = \frac{S_Q[h, p]}{\sqrt{d_{\text{head}}}}.$$

The $1/\sqrt{d_{\text{head}}}$ is folded into $f_Q$ so the score loop consumes one per-position factor instead of two.

## 9.3 Position-dependent operations

The full-vector RMSNorm scalar $\alpha_Q = 1/\sqrt{(\sum_i q_i^2) / D_Q + \epsilon}$ depends on the projected Q values. By HAR §10.2, the squared sum can be accumulated in the projection epilogue. One cross-rank scalar reduction provides the global $\alpha_Q$.

The per-head RoPE is position-dependent and applied before the per-head FWHT. By HAR §10.1, the conjugated operator $H_{d_{\text{head}}} R_p H_{d_{\text{head}}}^T$ is dense for the partial-RoPE layout, so the runtime exit/re-entry around RoPE is required. The RoPE-then-FWHT order is the entry into the rotated cache basis.

---

# X. Attention Score Identity

## 10.1 Raw score

For one query head $h$ at position $p$ and one cached KV head $g$ at position $t$,

$$r[h, p, g, t] = \sum_k q_{i_8}[h, p, k] \cdot K_{u_8}[g, t, k]$$

where $K_{u_8}[g, t, k] = K_{i_8}[g, t, k] + 128$ (or $K_{u_8}$ stored directly as u8).

## 10.2 Dequantized score

The dequantized attention score is

$$s[h, p, g, t] = (r[h, p, g, t] - b_q[h, p]) \cdot \frac{f_Q[h, p]}{127^2} \cdot K_{\text{scale}}[g, t].$$

Derivation: $r - b_q$ is the signed-signed inner product of $q_{i_8}$ and $K_{i_8}$ by §1.4. Dequantization applies $S_Q/127$ on the Q side and $K_{\text{scale}}/127$ on the K side, contributing $S_Q \cdot K_{\text{scale}} / 127^2$. The score normalization $1/\sqrt{d_{\text{head}}}$ is folded into $f_Q$ at §9.2.

## 10.3 Causal and validity masking

For positions outside the valid context (causal or padding), $s[h, p, g, t] \leftarrow -\infty$. The masking is applied after dequantization, before the softmax.

---

# XI. Online Softmax with Folded V Scale

## 11.1 Per-position folded weight

For each cached position $t$ in a position group, the unnormalized attention weight is $a_t = \exp(s[h, p, g, t] - m)$ where $m$ is the running max maintained across position groups by the online softmax.

The folded weight is

$$w[t] = a_t \cdot V_{\text{scale}}[g, t].$$

## 11.2 Per-group u8 quantization

For one position group with positions $t \in \mathrm{grp}$,

$$m_w = \max_{t \in \mathrm{grp}} w[t], \qquad w_{u_8}[t] = \mathrm{clamp}(\mathrm{round}(255 \cdot w[t] / m_w), 0, 255).$$

The u8 grid range $[0, 255]$ is used because $w[t] \ge 0$ (the product of a non-negative attention weight and a non-negative V scale).

The per-group dequantization scalar is $w_{\text{scale}} = m_w / 255$.

## 11.3 V-aggregation contribution per group

The contribution of one position group to the V-aggregation accumulator is

$$\Delta y[h, p, d] = \left(\sum_{t \in \mathrm{grp}} w_{u_8}[t] \cdot V_{i_8}[g, t, d]\right) \cdot w_{\text{scale}}.$$

Substituting $w_{u_8}[t] = \mathrm{round}(255 \cdot w[t] / m_w)$ and $w_{\text{scale}} = m_w / 255$:

$$\Delta y[h, p, d] = \sum_{t \in \mathrm{grp}} w[t] \cdot V_{i_8}[g, t, d] + O(\text{rounding of } w_{u_8}).$$

The per-group $m_w$ cancels exactly to within the $w_{u_8}$ rounding. No cross-group bookkeeping is required for $m_w$.

## 11.4 Final normalization

After all position groups for one query position have been accumulated, let $\ell[h, p] = \sum_t a_t$ be the softmax denominator. The dequantized attention output is

$$y[h, p, d] = \frac{1}{127 \cdot \ell[h, p]} \sum_t a_t \cdot V_{\text{scale}}[g, t] \cdot V_{i_8}[g, t, d].$$

The factor $1/127$ is the V dequantization. The factor $1/\ell$ is the softmax normalization. The per-group $m_w$ scalars do not appear because they cancelled within each group during V-aggregation.

## 11.5 Precision budget

The 8-bit u8 grid for $w$ is shared between the softmax probability $a_t$ and the V scale $V_{\text{scale}}[g, t]$ within one position group. For uniform $V_{\text{scale}}$ within a group, the full 256-level grid is available for $a_t$. For $V_{\text{scale}}$ varying by factor $f$ within a group, the effective grid for $a_t$ is reduced to $256/f$.

The position-group structure determines the locality of this tradeoff: small groups (one pos_group of $W$ positions) keep $V_{\text{scale}}$ variation local and admit a tighter $w_{u_8}$ grid for $a_t$.

---

# XII. Norm-Output Kernels

## 12.1 Single-output norm

The single-output RMSNorm + γ-split + FWHT + int8 kernel produces, per row $x \in \mathbb{R}^d$:

1. $\sigma^2 = \sum_k x_k^2$ in f32, with the bf16-to-f32 cast performed in the same pass.
2. $\alpha = 1/\sqrt{\sigma^2/d + \epsilon}$.
3. $w_k = x_k \cdot \alpha \cdot \sigma_\gamma[k]$ where $\sigma_\gamma$ is the activation-side split-gain (§4.4).
4. $w \leftarrow \mathcal{H}_d w$.
5. $S_a = \max_k |w_k|$, optionally floored.
6. $x_{i_8}[k] = Q_{S_a}(w_k)$.

The per-block variant replaces step 5 with per-block absmax and step 6 with per-block quantization, producing $S_a[b]$ and $x_{i_8}[k]$ for $k \in \Omega_b$.

The output dtype is i8 for $x_{i_8}$ and f32 for $S_a$ (or $S_a[\cdot]$).

## 12.2 Dual-output norm

For an operator chain in which the same RMS reduction supports two simultaneous output encodings — one bf16 normalized output (with the original $\gamma$, no FWHT) and one int8 quantized output (with $\sigma_\gamma$ and FWHT) — the dual-output kernel computes the RMS reduction once and emits both outputs:

1. $\sigma^2$ and $\alpha$ as in §12.1.
2. bf16 output: $y_{\text{bf16}}[k] = \mathrm{bf16}(x_k \cdot \alpha \cdot \gamma_k)$ using the original $\gamma$.
3. int8 output: as in §12.1 steps 3–6 using $\sigma_\gamma$.

The reduction is the only shared computation; the two outputs feed independent downstream consumers.

The dual-output form has no precision benefit over running two independent norm kernels. It avoids one read of the input $x$ and one $\sigma^2$ reduction.

---

# XIII. Router Encodings

## 13.1 F32 router

For a router weight $W \in \mathbb{R}^{E \times d}$ and bias $b \in \mathbb{R}^E$, per-token routing computes:

1. $\delta_e = \sum_k x_k \cdot W_{e, k}$ in f32.
2. $r_e = \sigma(\delta_e)$ where $\sigma$ is the score activation (commonly sigmoid).
3. $s_e = r_e + b_e$.

Top-K selection by $s_e$. Mixture weights are renormalized $r_e$ values for the selected $e$:

$$\mu_e = \frac{r_e}{\sum_{e' \in \mathrm{top}\text{-}K} r_{e'}}.$$

## 13.2 Centered bf16 router with gauge

For a router weight $W \in \mathbb{R}^{E \times d}$, define the column gauge

$$g[k] = \frac{1}{E} \sum_e W[e, k]$$

and the centered weight

$$C[e, k] = W[e, k] - g[k].$$

The stored encoding:

| Tensor | Type | Shape |
|---|---|---|
| $C$ | bf16 | $(E, d)$ |
| $g$ | bf16 | $(d,)$ |
| $b$ | f32 | $(E,)$ |

The runtime computation:

1. $p = \sum_k x_k \cdot g[k]$ (one bf16 dot per token).
2. $c_e = \sum_k x_k \cdot C[e, k]$ (one bf16 dot per expert per token).
3. $\delta_e = c_e + p$.
4. $r_e = \sigma(\delta_e)$.
5. $s_e = r_e + b_e$.

In exact arithmetic, $\delta_e = \sum_k x_k \cdot W[e, k]$, since $c_e + p = \sum_k x_k C[e, k] + \sum_k x_k g[k] = \sum_k x_k (C[e, k] + g[k])$.

Top-K selection and mixture weights as in §13.1.

## 13.3 Pivot requirement

The selection score $s_e = \sigma(c_e + p) + b_e$ is not shift-invariant in $c_e$ when $\sigma$ is non-linear. For two experts $e_1, e_2$ with $c_{e_1} > c_{e_2}$, the ordering of $s_{e_1}, s_{e_2}$ depends on the values of $\sigma(c + p) + b$ at both $c$ values, not just on their difference.

Omitting $p$ replaces $\sigma(c_e + p)$ with $\sigma(c_e)$, shifting the input to $\sigma$ by $-p$. For sigmoid, this changes which output regime each expert occupies (saturated vs. linear), and the bias $b_e$ added after $\sigma$ interacts differently with each regime.

The pivot $p$ reconstructs $\delta_e$ from $c_e$ exactly. Centering $W \to C$ can reduce the bf16 cast error of the stored matrix when the column-mean component dominates per-element magnitudes; the bf16 round-trip relative error scales with element magnitude, and the gauge $g$ is stored at full bf16 precision separately. The pivot is the cost of recovering the exact argument to $\sigma$ from the centered representation.

## 13.4 Bias dtype constraint

The bias $b$ is added after $\sigma$ and is compared directly against the score margin between adjacent selection ranks. Its dtype determines the noise floor of the selection.

If the score margin between ranks $K$ and $K+1$ falls below the bf16 floor on a non-trivial fraction of decisions, $b$ stored as bf16 introduces selection mismatches at that fraction. Storing $b$ as f32 keeps the bias above the bf16 noise floor.

---

# XIV. Layer Composition

## 14.1 Operator-to-recipe mapping

The encoding of each operator in a standard decoder layer follows from the basis-preservation classification of §5.5:

| Operator | Weight encoding | Activation source |
|---|---|---|
| Q projection | Single-sided per-row, split-gain on input layernorm γ | §12.1 single-output norm |
| K projection | Single-sided per-row, split-gain on input layernorm γ | §12.1 single-output norm |
| V projection | Single-sided per-row split-gain, or two-sided per-row split-gain with M-axis block $d_{\text{head}}$ matching the O projection's K-axis rotation | §12.1 single-output norm |
| Q prep | §9 | QKV projection output, partitioned by head |
| K cache write | §8.2 | QKV projection output, partitioned by head |
| V cache write, single-sided | §8.3 | QKV projection output, partitioned by head |
| V cache write, two-sided | §8.4 | QKV projection output, partitioned by head |
| Attention scoring | §10 | Q prep, K cache |
| Online softmax + V-fold + V-agg | §11 | Score, V cache |
| Per-head attention output | Per-head dynamic absmax | V-aggregation output, normalized |
| O projection | Single-sided per-row, no gain split, K-block size $d_{\text{head}}$ | Per-head attention output |
| Pre-FFN norm | §12.1 (when no bf16 router branch) or §12.2 (with bf16 router) | Residual stream |
| Router | §13.1 or §13.2 | bf16 norm output |
| Gate / up projections | Single-sided per-row, split-gain on post-attention layernorm γ | int8 norm output |
| Post-nonlinearity int8 | Per-block dynamic absmax over an FWHT block | $\mathrm{silu}(\text{gate}) \odot \text{up}$ output, FWHT |
| Down projection | Single-sided per-row, no gain split, K-block size matching post-nonlinearity FWHT block | Post-nonlinearity int8 output |
| Final norm | §12.1 per-block | Last residual row |
| LM head | Single-sided per-block, split-gain on final norm γ | Final norm output |

## 14.2 Tensor-parallel sharding

Each operator admits sharding along one axis:

| Operator | Sharded axis |
|---|---|
| QKV projection | Output rows |
| O projection | Input columns |
| Gate / up projections | Output rows |
| Down projection | Input columns |
| MoE experts | Expert index (block-sharded) |
| Router (centered) | Output rows (expert index) |
| Router (gauge, bias) | Replicated |
| LM head | Output rows |

Norms and elementwise residual operations are replicated across ranks. Allreduce inserts at the boundary between operators sharded on different axes.

## 14.3 Boundary norm reductions

The full-vector QK normalization reduces over the full Q vector across all heads (and all TP ranks if Q is row-sharded). By HAR §10.2, the squared sum can be accumulated in the projection epilogue from local sums-of-squares, with one cross-rank scalar reduction supplying the global inverse RMS.

The post-attention residual addition is a vector add of the rank-local attention output (after the O projection allreduce) into the residual stream.

The final norm is a single-rank reduction over the last residual row before the LM head.
