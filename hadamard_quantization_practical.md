# Hadamard-Rotated Int8 Quantization: Practical Implementation Guide

## Overview

This document describes a concrete int8 quantization strategy for transformer
inference that uses the Fast Walsh-Hadamard Transform (FWHT) to improve
quantization quality without changing the matmul kernel, the weight packing
format, or the integer instruction set. The rotation is a pre-processing step
on activations (online) and weights (offline). Everything downstream — VNNI
packing, vpdpbusd/tdpbssd accumulation, per-row scales, epilogue rescale — is
identical to standard channelwise int8.

The strategy has been validated with working Mojo implementations at
`design_analysis/` covering:

- `int8ch.mojo` — baseline channelwise int8 matmul with distribution analysis
- `hadamard.mojo` — side-by-side channelwise vs Hadamard+int8 comparison
- `sweep.mojo` — distribution sensitivity across 4 outlier severity levels
- `block_sweep.mojo` — FWHT block size vs quality tradeoff
- `hadamard_layer.mojo` — full single-layer operational flow (projections only)
- `hadamard_layer_int8.mojo` — full single-layer flow including int8 attention

---

## 1. The Core Idea

Standard per-row int8 quantization sets the scale from the row's absmax. When
a row contains outlier channels (magnitudes 10-1000x larger than the median),
the scale is dominated by the outliers, and normal channels use only a fraction
of the int8 range.

Measured on synthetic data with 5% of channels at 30x magnitude:

```
                      Channelwise    Hadamard+Int8
Activation SQNR:      31.04 dB       41.58 dB
Matmul SQNR:          30.70 dB       38.48 dB
Effective bits:        4.06 / 8       7.99 / 8
qi=0 collapse:         23.99%         1.10%
Column RMS ratio:      47.09          2.13
```

The FWHT is an orthonormal rotation that spreads energy uniformly across all
components within a block. After rotation, every component has similar
magnitude regardless of the original channel structure. Per-row absmax
quantization then uses the full int8 range for every channel.

The rotation is exact and self-inverse: H * H = I (normalized). The matmul
result is mathematically identical to the unrotated matmul — the rotation
cancels between the activation and weight:

```
(H * x)^T * (H * w) = x^T * H^T * H * w = x^T * w
```

Quantization error is the only source of difference between the rotated and
unrotated paths.

---

## 2. The Fast Walsh-Hadamard Transform

### 2.1 Algorithm

For a vector of length n (power of 2), the FWHT performs log2(n) butterfly
stages. At each stage, pairs of elements at a given stride are replaced by
their sum and difference. A final scaling by 1/sqrt(n) makes the transform
orthonormal.

```
def fwht_inplace(buf, n):
    half = 1
    while half < n:
        for i in range(0, n, half * 2):
            for j in range(half):
                a, b = buf[i+j], buf[i+j+half]
                buf[i+j]      = a + b
                buf[i+j+half] = a - b
        half *= 2
    scale = 1 / sqrt(n)
    for i in range(n):
        buf[i] *= scale
```

Cost: n * log2(n) additions + n multiplies (the final scaling).

The transform is applied block-diagonally: for a row of K elements with block
size n, partition into K/n independent blocks and apply the FWHT to each.

### 2.2 Block Size Selection

Measured matmul SQNR (M=64, K=2048, N=2048, 5% outliers at 30x):

```
block  ops/element  SQNR     gain over channelwise
 none      0        30.80 dB    —
    2      2        33.23 dB   +2.4 dB
    4      3        34.41 dB   +3.6 dB
    8      4        36.28 dB   +5.5 dB
   16      5        37.14 dB   +6.3 dB
   32      6        37.40 dB   +6.6 dB
   64      7        38.19 dB   +7.4 dB
  128      8        38.18 dB   +7.4 dB
  256      9        38.55 dB   +7.8 dB
  512     10        38.63 dB   +7.8 dB
 1024     11        38.62 dB   +7.8 dB
 2048     12        38.52 dB   +7.7 dB
```

Diminishing returns set in around block=64. The sweet spot is 16-64: almost
all of the SQNR benefit at 5-7 additions per element. For context, the matmul
that follows is O(N*K) ops per activation row — the rotation at O(K * log(n))
is a fraction of a percent of the matmul compute.

### 2.3 Block Size Constraints

The block size must:

1. Be a power of 2
2. Divide the K dimension of the matmul it precedes
3. Divide K/tp under tensor parallelism (for shard-local rotation)
4. Match between the activation rotation and the weight rotation for that
   matmul (so H cancels)

For int8 attention (per-head rotation), block size = head_dim. This constrains
the O projection's weight rotation to also use block = head_dim for its K
dimension.

A practical default: block = min(64, head_dim) for all operations. This
satisfies all constraints for standard architectures and captures nearly all
of the quality benefit.

---

## 3. Offline Weight Preparation

Three steps, applied once per weight matrix at model load or export time.

### 3.1 RMSNorm Gamma Absorption

Standard RMSNorm applies a learned elementwise gain gamma after normalization:

```
RMSNorm(x) = (x / rms(x)) * gamma
```

The elementwise product with gamma does not commute with the Hadamard
transform. Absorbing gamma into the subsequent weight matrix eliminates it
from the online path:

```
W' = W * diag(gamma)     # W'[n,k] = W[n,k] * gamma[k]
```

This is a column-wise scaling — multiply each column k of W by the scalar
gamma[k]. Computed once, offline.

Which weights absorb which gamma:

```
input_layernorm.gamma     → Q, K, V          (share the same normed input)
post_attn_layernorm.gamma → GATE, UP         (share the same normed input)
O, DOWN                   → nothing          (no norm precedes them)
```

After absorption, runtime RMSNorm reduces to scalar division:

```
x_normed = x / rms(x)     # no gamma, no elementwise product
```

### 3.2 FWHT Rotation

Apply the block-diagonal FWHT to each row of the weight matrix (the K
dimension):

```
for each row n:
    W_rot[n, :] = block_diagonal_fwht(W'[n, :])
```

This rotates the weight's contraction dimension to match the activation
rotation that will be applied online.

### 3.3 Int8 Quantization + Packing

Standard per-row absmax quantization of the rotated weights:

```
scale[n] = max(|W_rot[n, :]|) / 127
qi[n, k] = clamp(round(W_rot[n, k] / scale[n]), -128, 127)
```

Followed by hardware-specific packing (VNNI 6D layout, AMX tile layout, etc.).
The packing is a permutation of int8 values — identical to channelwise. The
fact that the values were Hadamard-rotated before quantization is invisible to
the packing and to the matmul kernel.

### 3.4 Tensor Parallelism

Gamma absorption is shard-local. The gamma vector has dimension K = HIDDEN,
which is the non-sharded dimension for RowShard projections (Q, K, V, GATE,
UP). Every rank holds all K columns and applies the full gamma independently.
ColShard projections (O, DOWN) don't absorb gamma.

The FWHT is block-diagonal on K. For RowShard weights, K is not sharded — each
rank rotates its full K dimension. For ColShard weights, K is sharded, so each
rank rotates its local K/tp columns using block size that divides K/tp.

No cross-rank communication is needed for any offline preparation step.

---

## 4. Online Layer Execution

### 4.1 Fused Quantization Kernels

Four fused kernels replace the separate quantization passes. Each reads bf16,
performs the rotation in registers, and writes int8.

**rms_fwht_quantize** — 2 per layer (before QKV, before GATE/UP)

```
input:  bf16 x[M, K]
output: int8 qi[M, K], f32 scale[M]

Per row:
  1. FWHT(x[m, :]) in blocks → x_rot
  2. Dual reduction: sum(x_rot^2) → rms, max(|x_rot|) → absmax
  3. scale[m] = absmax / (rms * 127)
  4. qi[m, k] = round(x_rot[k] * 127 / absmax)
```

The rms cancels in the qi values and is encoded in the scale. The int8_gemm
epilogue applies `acc * act_scale * w_scale`, which includes the 1/rms factor
via act_scale. The result equals `W' * (x / rms)` = `W * gamma * (x / rms)`
= `W * RMSNorm(x)`.

Fusion structure: 1 read (bf16 activation), butterfly in registers/L1 per
block, dual reduction (sum-of-squares + absmax in one pass), quantize, 1 write
(int8). For block=512, the entire block (2KB f32) fits in the AVX-512 register
file (32 x 16 floats = 512 floats).

**fwht_quantize** — 1 per layer (before O_PROJ)

```
input:  bf16 x[M, K]
output: int8 qi[M, K], f32 scale[M]

Same as rms_fwht_quantize but without the rms computation.
scale[m] = max(|FWHT(x[m])|) / 127
```

For the O projection specifically: when int8 attention is used, the attention
output is already in the per-head rotated domain. This kernel becomes a plain
quantize (no FWHT) since the rotation was already done inside the attention.

**silu_fwht_quantize** — 1 per layer (before DOWN)

```
input:  bf16 gate[M, K], bf16 up[M, K]
output: int8 qi[M, K], f32 scale[M]

Per row:
  1. silu_out[k] = gate[k] * sigmoid(gate[k]) * up[k]
  2. FWHT(silu_out) in blocks → silu_rot
  3. quantize silu_rot → qi, scale
```

Fusion: reads gate + up, computes SiLU elementwise, butterfly in registers,
absmax + quantize, writes int8. The nonlinearity and the rotation share a
single pass.

### 4.2 Int8 GEMM

Identical to channelwise int8:

```
acc_i32[m, n] = sum_k int8(act[m, k]) * int8(weight[n, k])
out_bf16[m, n] = float(acc_i32) * act_scale[m] * weight_scale[n]
```

Maps to vpdpbusd (AVX-512 VNNI) or tdpbusd/tdpbssd (AMX). Same VNNI 6D
packing. Same tiling contracts. Same epilogue. The kernel does not know
the values were Hadamard-rotated.

For the u8/i8 signedness convention with vpdpbusd: activations are clamped
to [-128, 127] then shifted by +128 to [0, 255] for u8 storage. This
clamp-and-shift happens after the FWHT, in the quantization step. Unchanged
from channelwise.

### 4.3 Single-Layer Flow (Projections Only)

```
 1. rms_fwht_quantize(x)         → x_i8, x_sc
 2. int8_gemm(x_i8, Q')         → q             ┐
 3. int8_gemm(x_i8, K')         → k             ├ shared x_i8
 4. int8_gemm(x_i8, V')         → v             ┘
 5. rope(q), rope(k)                              bf16, unchanged
 6. kv_cache_write(k, v)                          bf16 → cache
 7. attention(q, k_cache, v_cache) → attn         bf16, unchanged
 8. fwht_quantize(attn)         → attn_i8, a_sc  (or plain quantize if int8 attn)
 9. int8_gemm(attn_i8, O')      → o_out
10. elem_add(x, o_out)          → x              residual
11. rms_fwht_quantize(x)        → x_i8, x_sc
12. int8_gemm(x_i8, GATE')      → gate           ┐ shared x_i8
13. int8_gemm(x_i8, UP')        → up             ┘
14. silu_fwht_quantize(gate, up) → silu_i8, s_sc
15. int8_gemm(silu_i8, DOWN)    → down_out
16. elem_add(x, down_out)       → x              residual
```

Allreduce slots between steps 9-10 and 15-16 for TP > 1.

Per layer: 4 fused quantization kernels, 7 int8 GEMMs, standard bf16 ops
(rope, attention, cache write, residual add, allreduce) unchanged.

### 4.4 Int8 Attention (Prefill Optimization)

For prefill (large M), the attention matmuls (Q*K^T scoring, weights*V
aggregation) are large enough to benefit from AMX int8. The approach:

**Scoring (Q * K^T):**

Q and K arrive as bf16 from the projection int8_gemms (original domain).
Apply per-head FWHT over head_dim, then quantize to int8. The rotated inner
products equal the original inner products (Parseval). Int8 dot products
produce scores, rescaled to f32 for softmax.

**Aggregation (weights * V):**

V cache entries are FWHT-rotated per kv_head over head_dim and quantized.
The softmax attention weights (f32) have the per-entry V scales absorbed
into them before quantizing to int8. This absorption enables clean i32
accumulation:

```
for each t:
    w_absorbed[t] = softmax_weight[t] * v_scale[t]
quantize(w_absorbed) → w_i8, w_sc

for each d:
    acc = sum_t int(w_i8[t]) * int(v_i8[t, d])
    out[d] = float(acc) * w_sc
```

Without scale absorption, the per-entry V scales would vary across the
contraction dimension (T), preventing factorization out of the i32
accumulator. The absorption folds the varying scales into the attention
weights (which are being quantized anyway), leaving a single scalar w_sc
for the epilogue.

**Output domain:**

The attention output is in the per-head rotated domain. It feeds directly into
the O projection quantization step — no FWHT needed, just a plain quantize.
The O projection weight is rotated with block = head_dim on its K dimension,
matching the per-head rotation. H cancels in the matmul.

**Decode vs prefill split:**

For decode (M=1), attention is GEMV — AMX tile setup overhead dominates, and
int8 scoring provides minimal benefit. Use bf16 attention for decode, int8
attention for prefill. This is the same split KTransformers uses (AMX for
prefill, AVX-512 VNNI for decode), just with the Hadamard rotation added to
the quantization boundary.

The layer flow with int8 attention differs at steps 7-8:

```
 7. int8_gqa_attention(q, k_cache, v_cache) → attn_rot
      fwht+quantize Q per head
      fwht+quantize K cache per kv_head
      int8 scoring (Parseval)
      softmax (f32)
      fwht+quantize V cache per kv_head
      absorb V scales, quantize weights
      int8 aggregation → output in rotated domain
 8. quantize(attn_rot)          → attn_i8    (no FWHT, already rotated)
```

### 4.5 Block Size Compatibility

Using block = head_dim for all operations guarantees compatibility between
the per-head attention rotation and the projection weight rotations. The
block boundaries align with head boundaries in the concatenated Q/K/V/O
tensors.

For SmolLM2 (head_dim=64), block=64 lands at the plateau of the quality
curve. For architectures with head_dim=128, block=128 is slightly beyond
the plateau but still near-optimal.

If int8 attention is not used (bf16 attention path), the block size for
projections can be chosen independently of head_dim. Larger blocks (256,
512) may be used for the activation rotations, giving slightly better
isotropization.

---

## 5. Distribution Sensitivity

Measured across 4 activation distributions (weights N(0, 0.02), M=64,
K=2048, N=2048, block=512):

```
                          Uniform    5%@5x     5%@100x    0.5%@1000x
Column RMS ratio
  Before rotation         1.91       9.02      180.45     1594.64
  After rotation          1.80       1.86      1.84       1.58

Activation SQNR (dB)
  Channelwise             42.18      33.55     31.13      36.63
  Hadamard                41.62      41.69     42.50      45.99
  Delta                   -0.56      +8.14     +11.38     +9.36

Matmul SQNR (dB)
  Channelwise             39.09      33.11     30.80      35.48
  Hadamard                38.53      38.52     39.05      40.02
  Delta                   -0.57      +5.42     +8.26      +4.54

Effective bits (/8)
  Channelwise             7.99       6.52      2.25       1.50
  Hadamard                7.99       7.99      7.99       7.99

qi=0 fraction (%)
  Channelwise             1.02       4.14      70.39      99.99
  Hadamard                1.04       1.11      1.05       0.64
```

Key observations from the data:

- On uniform data (no outliers), channelwise is 0.57 dB better. The rotation
  adds floating point noise to values that didn't need redistribution.
- The Hadamard rows are flat across all distributions. Channelwise degrades
  with outlier severity.
- Effective bits: channelwise drops to 1.5/8 at extreme outliers (99.99% of
  non-outlier values collapse to zero). Hadamard maintains 7.99/8 regardless.
- The column RMS ratio after rotation is consistently ~1.6-2.1 regardless of
  the input ratio (which ranges from 1.9 to 1595).

---

## 6. Architectural Generality

### 6.1 Standard Dense Transformers

Fully covered. The 7 projection matrices per layer (Q, K, V, O, GATE, UP,
DOWN) are the only learned weights in the layer. All become int8 GEMMs with
Hadamard-rotated inputs. The 2 norm gains are absorbed offline. Attention,
RoPE, residual connections, and allreduce are unchanged.

### 6.2 Mixture of Experts (MoE)

No structural issues. The router is a small linear projection — same
int8_gemm treatment. Each expert MLP (gate, up, down) gets the same offline
preparation and online flow independently. Gamma absorption from the pre-MLP
norm applies to the router weight and all expert gate/up weights.

Token routing (top-k selection, dispatch, combine) operates on bf16 values
produced by int8_gemm. The FWHT and quantization are per-row (per-token),
so routing tokens to different experts after quantization preserves
correctness — each token's qi and scale travel together.

### 6.3 Grouped Query Attention / Multi-Query Attention

Fully covered. GQA and MQA change the number of KV heads but not the
structure. The per-head FWHT in int8 attention operates on head_dim
regardless of how many heads share a KV group.

### 6.4 Alternative Nonlinearities

SiLU, GELU, ReLU, etc. change one line in the silu_fwht_quantize kernel
(the elementwise function). The FWHT and quantization steps are identical.
The fusion structure (read bf16, compute nonlinearity, butterfly, quantize,
write int8) is the same for any elementwise activation.

### 6.5 Alternative Position Encodings

RoPE operates in the original domain (bf16) between the QKV projections and
attention. It is unchanged. ALiBi (additive bias on attention scores) is also
unchanged — it operates on scalar scores, orthogonal to the feature-space
rotation. Absolute position embeddings are added before the first layer and
are unchanged.

---

## 7. Hardware Interface

### 7.1 AVX-512 VNNI (vpdpbusd)

The int8_gemm uses u8 activations x i8 weights → i32 accumulation. The VNNI
6D packing layout is unchanged. Weight packing is applied to the Hadamard-
rotated, quantized int8 values — the packing is a permutation of bytes and
doesn't interact with the rotation.

The FWHT in the fused quantization kernels uses f32 butterfly operations in
AVX-512 registers. For block=512, the block (2KB as f32) fits exactly in the
32-register AVX-512 file. Smaller blocks use fewer registers.

### 7.2 AMX (tdpbusd / tdpbssd)

AMX tiles operate on int8 data loaded from memory. The FWHT must complete
before the data enters the AMX pipeline — it cannot be fused into tile
operations. The FWHT runs in AVX-512 registers as a pre-pass, writes int8
to memory, then AMX loads the int8 tiles.

Tile configuration (LDTILECFG) is unchanged. Tile dimensions, the
accumulation datatype (i32), and the epilogue rescale are all identical
to channelwise.

For prefill with int8 attention, AMX handles the Q*K^T and weights*V
matmuls. For decode, the attention path remains bf16 (GEMV, not GEMM).

### 7.3 Epilogue

```
out_bf16[m, n] = float(acc_i32[m, n]) * act_scale[m] * weight_scale[n]
```

Two f32 multiplies per output element, identical to channelwise. For
rms_fwht_quantize, the act_scale encodes both the quantization range and
the rms normalization factor:

```
act_scale = max(|FWHT(x)|) / (rms(x) * 127)
```

The kernel does not distinguish this from a standard per-row scale.

---

## 8. Memory Layout

### 8.1 Weights

Stored as int8 with per-row f32 scales, same as channelwise. The Hadamard
rotation and gamma absorption are applied before quantization and do not
affect the stored format. VNNI packing is applied at load time to the int8
values, same as channelwise.

The norm gain vectors (input_layernorm.weight, post_attention_layernorm.weight)
are consumed during offline preparation and are not needed at runtime. They
can be dropped from the runtime memory layout.

### 8.2 Activations

All activations between operations are bf16, same as channelwise. The int8
quantized activations are transient — produced by a fused kernel and consumed
by the immediately following int8_gemm. They live in scratch memory and are
overwritten each layer.

### 8.3 KV Cache

Stored as bf16 in the original domain (after RoPE, before any Hadamard
rotation). For int8 attention, the cache entries are FWHT-rotated and
quantized on the fly during the attention computation.

Optionally, the cache could store pre-rotated int8 values to avoid repeated
FWHT on the same entries. This trades cache memory (int8 vs bf16, but with
per-entry scales) for compute. For long contexts with prefill, the savings
from avoiding redundant rotation may be significant.

---

## 9. Comparison with Channelwise Int8

What changes:

- 4 fused quantization kernels gain a butterfly pass (5-10 additions/element)
- Offline weight preparation adds gamma absorption + FWHT (one-time cost)
- Norm gain vectors are eliminated from runtime layout

What does not change:

- Int8 GEMM kernel (same instruction, same packing, same epilogue)
- Weight storage format (int8 + per-row f32 scale)
- Activation storage format (bf16 between ops)
- Tensor parallelism (same sharding, same allreduce positions)
- Attention mechanism (bf16 path identical; int8 path is additive)
- RoPE, residual connections, allreduce

The operational overhead of the FWHT is bounded by `K * (log2(block) + 1)`
additions per activation row per quantization point. For K=4096 and block=64,
that is 7 additions per element, applied 4 times per layer. The 7 int8 GEMMs
per layer each do K*N multiply-accumulates per row. The rotation is a fraction
of a percent of the matmul compute at any practical model size.

---

## 10. Implementation Checklist

### Offline (model export / load time)

1. For each norm: extract gamma vector
2. For each projection following a norm: `W'[n,k] = W[n,k] * gamma[k]`
3. For each projection: FWHT each row (K dimension) with chosen block size
4. Per-row absmax quantize to int8
5. VNNI/AMX pack as usual
6. Store int8 weights + f32 per-row scales (same format as channelwise)

### Online (per layer, per token)

1. `rms_fwht_quantize(x)` — fused norm + rotation + quantize
2. 3x `int8_gemm` for Q, K, V (shared quantized activation)
3. RoPE on bf16 Q, K
4. KV cache write (bf16)
5. Attention (bf16 for decode, optionally int8 for prefill)
6. `fwht_quantize(attn_out)` or `quantize(attn_rot)` if int8 attention
7. `int8_gemm` for O
8. Allreduce (TP > 1), residual add
9. `rms_fwht_quantize(x)` — fused norm + rotation + quantize
10. 2x `int8_gemm` for GATE, UP (shared quantized activation)
11. `silu_fwht_quantize(gate, up)` — fused nonlinearity + rotation + quantize
12. `int8_gemm` for DOWN
13. Allreduce (TP > 1), residual add

### New code required

- `fwht_inplace(buf, n)` — ~15 lines, pure additions + final scale
- `fwht_rows(mat, rows, cols, block)` — 5 lines, calls fwht_inplace per block
- `absorb_gamma(weight, gamma, rows, cols)` — 3 lines, column-wise multiply
- 3 fused quantization kernels (rms_fwht_quantize, fwht_quantize,
  silu_fwht_quantize) — each wraps FWHT + existing quantize logic
- Optionally: int8_gqa_attention with per-head FWHT and scale absorption

### Code that does not change

- int8_gemm kernel
- VNNI packing
- Weight loading / safetensors I/O (format is identical)
- Epilogue rescale
- Attention (bf16 path)
- RoPE, KV cache, residual add, allreduce
- BurstPool dispatch, NUMA arena layout, TP sharding logic
