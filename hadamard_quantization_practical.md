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
- `hadamard_layer_int8_kv.mojo` — int8 KV cache with RoPE, numerical validation

The offline quantization pipeline is implemented in `quant/` and validated
against a Python reference (`validation/validate_hadquant.py`): 422/422 tensors
pass, 100% exact match on int8 weights, zero relative error on scales.

The model spec, execution interface, and loader are implemented in
`modeling/smollm2_hadquant_tp.mojo`.

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

For SmolLM2 (head_dim=64), block=64 is set as a model-level constant
(`FWHT_BLOCK = HEAD_DIM`) and used for all operations.

---

## 3. Offline Weight Preparation

Three steps, applied once per weight matrix at quantization time. Implemented
in `quant/ops.mojo` and orchestrated by `quant/engine.mojo`.

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

The norm weights are marked `IsAbsorbed` in the model spec — consumed during
quantization, absent from the output file, not loaded at runtime.

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

Two fused kernels handle activation quantization. Each reads bf16, performs
the rotation in registers, and writes int8.

**rms_fwht_quantize** — 2 per layer (before QKV, before GATE/UP)

```
input:  bf16 x[M, K]
output: int8 qi[M, K], f32 scale[M]

Implementation (fused single-pass):
  1. FWHT(x[m, :]) in blocks → x_rot
  2. Dual reduction: sum(x_rot^2) → rms, max(|x_rot|) → absmax
  3. scale[m] = absmax / (rms * 127)
  4. qi[m, k] = round(x_rot[k] * rms * 127 / absmax)
```

The FWHT preserves norms (Parseval), so rms(FWHT(x)) = rms(x). The rms and
absmax are computed in a single dual reduction over the rotated values. The
rms factor is folded into the scale so the int8_gemm epilogue produces
`W * gamma * (x / rms)` = `W * RMSNorm(x)` without materializing a normalized
bf16 intermediate.

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

```
raw_acc[m,n] = sum_k u8(act[m,k] ^ 0x80) * i8(weight[n,k])
corrected    = float(raw_acc[m,n]) - 128.0 * weight_colsum[n]
output[m,n]  = bf16(corrected * act_scale[m] * weight_scale[n])
```

Maps to vpdpbusd (AVX-512 VNNI) or tdpbusd (AMX). Activations are stored as
i8 and converted to u8 via sign-bit XOR at register load for VNNI's u8 x i8
convention. This introduces a per-output-channel bias of `128 * colsum[n]`
where `colsum[n] = sum_k weight[n,k]`. The column sums are precomputed at load
time from the packed i8 weights and stored in the arena alongside each weight.
The bias is subtracted in the epilogue before scale multiplication.

Same VNNI 6D packing. Same tiling contracts. The kernel does not know the
values were Hadamard-rotated.

### 4.3 Int8 KV Cache

The KV cache stores pre-rotated int8 values with per-head f32 scales. This
eliminates repeated FWHT rotation of the entire context at every attention
call.

**At K cache write — `rope_fwht_quantize_k_write` (once per token per layer):**

K arrives as bf16 from int8_gemm (original domain). RoPE, FWHT, and int8
quantize are fused in a single per-head loop — one read, one write:

```
For each kv_head g:
    k_head = K[g*head_dim : (g+1)*head_dim]
    k_roped = RoPE(k_head, pos)
    k_rot = FWHT(k_roped)
    k_scale[pos, g] = max(|k_rot|) / 127
    k_qi[pos, g, :] = round(k_rot / k_scale)
```

This eliminates the separate RoPE pass on K. RoPE and FWHT both operate
per-head, so they share the same loop.

**At V cache write — `fwht_quantize_v_write` (once per token per layer):**

V arrives as bf16 from int8_gemm (original domain). No RoPE.

```
For each kv_head g:
    v_rot = FWHT(V[g*head_dim : (g+1)*head_dim])
    v_scale[pos, g] = max(|v_rot|) / 127
    v_qi[pos, g, :] = round(v_rot / v_scale)
```

**Cache layout per layer:**

```
K_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN]
V_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN]
K_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS]
V_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS]
```

For SmolLM2 (KV_HIDDEN=192, NUM_KV_HEADS=3, MAX_SEQ_LEN=8192): 408 bytes per
position vs 768 bytes for bf16 — 47% reduction, 89 MB saved across 30 layers.

**Correctness with RoPE:**

RoPE is applied to K in the original domain before cache write. The FWHT
rotation is applied after RoPE. At attention time, Q is also FWHT-rotated
per head (after RoPE). Parseval guarantees:

```
<FWHT(RoPE(Q)), FWHT(RoPE(K))> = <RoPE(Q), RoPE(K)>
```

Validated numerically at SmolLM2 dimensions (HEAD_DIM=64, CONTEXT=256):
bf16 KV and int8 KV paths produce bit-identical outputs (999 dB difference).
Int8 attention SQNR is 30 dB vs f32 reference.

### 4.4 Int8 GQA Attention

Reads pre-rotated int8 K and V from cache. Q arrives as bf16 without RoPE —
RoPE is applied inside the attention kernel per head, fused with the FWHT.

```
For each query head h (kv group g = h // GQA_FACTOR):

  1. RoPE(Q_head, pos)
  2. FWHT + quantize Q per head → qi_Q, q_scale
  3. Int8 scoring: score[t] = float(qi_Q · qi_K[t]) * q_scale * k_scale[t,g] / sqrt(d)
  4. Softmax in f32
  5. Absorb V scales: w_absorbed[t] = softmax[t] * v_scale[t, g]
  6. Quantize absorbed weights → qi_w, w_scale
  7. Int8 aggregation: out[d] = float(sum_t qi_w[t] * qi_V[t,d]) * w_scale
  8. Quantize output → int8 directly (absmax + round from f32, no bf16 step)

Output is int8 + f32 scale in per-head rotated domain.
```

V scale absorption (step 5) folds the per-entry V scales into the attention
weights before quantizing them. This enables clean i32 accumulation — without
absorption, the varying V scales across the contraction dimension (T) would
prevent factoring them out of the accumulator.

The output is quantized to int8 directly from the f32 aggregation result —
no bf16 intermediate. The int8 output feeds the O projection's int8_gemm
directly. The O weight is rotated with block = head_dim, so H cancels.

### 4.5 Single-Layer Flow

```
 1. rms_fwht_quantize(x)              → act_i8, act_sc
 2. int8_gemm(act_i8, Q')             → q (bf16, only projection materialized)
 3. int8_gemm_k_to_cache(act_i8, K')  → K cache (fused gemm + RoPE + FWHT + quant)
 4. int8_gemm_v_to_cache(act_i8, V')  → V cache (fused gemm + FWHT + quant)
 5. int8_gqa_attention(q, kv_cache)   → act_i8, act_sc (RoPE on Q fused, int8 out)
 6. int8_gemm(act_i8, O')            → o_out
 7. allreduce, residual add
 8. rms_fwht_quantize(x')            → act_i8, act_sc
 9. int8_gemm_gate_up(act_i8, G', U') → gate, up (one activation read)
10. silu_fwht_quantize(gate, up)      → act_i8, act_sc
11. int8_gemm(act_i8, DOWN')          → down_out
12. allreduce, residual add
```

Per layer: 12 steps. 2 fused activation quantize kernels, 3 standard int8
GEMMs (Q, O, DOWN), 2 gemm-to-cache kernels (K with fused RoPE, V), 1 fused
gate+up gemm, 1 int8 attention (fused RoPE on Q, int8 output), 1 fused SiLU
quantize. bf16 intermediates: only Q (slot 0) and gate/up (slots 0, 1). K and
V never materialize as bf16.

Validated numerically: the 12-step fused flow produces bit-identical output
to the unfused 15-step flow (999 dB, `design_analysis/optimal_flow.mojo`).

### 4.6 Block Size Compatibility

Using block = head_dim for all operations guarantees compatibility between
the per-head attention rotation and the projection weight rotations. The
block boundaries align with head boundaries in the concatenated Q/K/V/O
tensors.

For SmolLM2 (head_dim=64), block=64 lands at the plateau of the quality
curve. For architectures with head_dim=128, block=128 is slightly beyond
the plateau but still near-optimal.

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

---

## 6. Architectural Generality

### 6.1 Standard Dense Transformers

Fully covered. The 7 projection matrices per layer (Q, K, V, O, GATE, UP,
DOWN) are the only learned weights in the layer. All become int8 GEMMs with
Hadamard-rotated inputs. The 2 norm gains are absorbed offline. RoPE and
residual connections are unchanged. Allreduce positions are unchanged.

### 6.2 Mixture of Experts (MoE)

No structural issues. The router is a small linear projection — same
int8_gemm treatment. Each expert MLP (gate, up, down) gets the same offline
preparation and online flow independently. Gamma absorption from the pre-MLP
norm applies to the router weight and all expert gate/up weights.

### 6.3 Grouped Query Attention / Multi-Query Attention

Fully covered. GQA and MQA change the number of KV heads but not the
structure. The per-head FWHT in the KV cache and attention operates on
head_dim regardless of how many heads share a KV group.

### 6.4 Alternative Nonlinearities

SiLU, GELU, ReLU, etc. change one line in the silu_fwht_quantize kernel
(the elementwise function). The FWHT and quantization steps are identical.

### 6.5 Alternative Position Encodings

RoPE is applied to Q in bf16 as a separate step. For K, RoPE is fused into
the K cache write kernel (`rope_fwht_quantize_k_write`) — applied per head
before the FWHT rotation and int8 quantize, in a single pass. Parseval
guarantees the rotated dot products are correct:
`<FWHT(RoPE(Q)), FWHT(RoPE(K))> = <RoPE(Q), RoPE(K)>`.
ALiBi and absolute position embeddings are unchanged.

---

## 7. Hardware Interface

### 7.1 AVX-512 VNNI (vpdpbusd)

The int8_gemm uses u8 activations x i8 weights → i32 accumulation.
Activations are stored as i8 and XOR'd with 0x80 at register load to produce
u8. This introduces a bias of `128 * column_sum[n]` per output channel,
corrected in the epilogue. Column sums are precomputed at load time and stored
in the arena as f32 per weight row.

The VNNI 6D packing layout is unchanged. Weight packing is applied to the
Hadamard-rotated, quantized int8 values — the packing is a permutation of
bytes and doesn't interact with the rotation.

### 7.2 AMX (tdpbusd / tdpbssd)

AMX tiles operate on int8 data loaded from memory. The FWHT must complete
before the data enters the AMX pipeline — it cannot be fused into tile
operations. The FWHT runs in AVX-512 registers as a pre-pass, writes int8
to memory, then AMX loads the int8 tiles.

Tile configuration (LDTILECFG) is unchanged. Tile dimensions, the
accumulation datatype (i32), and the epilogue rescale are all identical.

### 7.3 Epilogue

```
raw_acc    = vpdpbusd(u8_act, i8_weight)
corrected  = float(raw_acc) - 128.0 * float(colsum[n])
output[m,n] = bf16(corrected * act_scale[m] * weight_scale[n])
```

Three f32 ops per output element (subtract, multiply, multiply). For
rms_fwht_quantize, the act_scale encodes both the quantization range and
the rms normalization factor:

```
act_scale = max(|FWHT(x)|) / (rms(x) * 127)
```

The kernel does not distinguish this from a standard per-row scale.

---

## 8. Memory Layout

### 8.1 Weights

Stored as int8 with per-row f32 scales, same format as channelwise. The
Hadamard rotation and gamma absorption are applied before quantization and do
not affect the stored format. VNNI packing is applied at load time.

Per-row f32 column sums are computed at load time from the packed i8 weights
for the VNNI u8/i8 bias correction.

The norm gain vectors (input_layernorm.weight, post_attention_layernorm.weight)
are consumed during quantization and absent from the output file. They are
marked `IsAbsorbed` in the model spec — the loader skips them.

### 8.2 Activations

All activations between operations are bf16. The int8 quantized activations
are transient — produced by a fused kernel and consumed by the immediately
following int8_gemm. They live in scratch memory (overlaid on scratch slot 2)
and are overwritten each phase.

### 8.3 KV Cache

Stored as int8 with per-head f32 scales, pre-rotated via per-head FWHT at
cache write time. K has RoPE applied before rotation. V is rotated directly.

This eliminates repeated FWHT rotation of the entire context at every
attention call. At attention time, only Q needs rotation (once per token).

Cache layout per layer:

```
K_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN]        # pre-rotated
V_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN]        # pre-rotated
K_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS]     # per-head scale
V_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS]     # per-head scale
```

Memory savings vs bf16 cache: 47% per position (408 vs 768 bytes for SmolLM2).

---

## 9. Memory Savings Summary (SmolLM2-135M, TP=1)

```
                    bf16        hadquant    savings
Weights (dist):     212 MB      107 MB      105 MB  (int8 + scales)
State (KV+act):     285 MB      197 MB       88 MB  (int8 KV cache)
Total arena:        497 MB      304 MB      193 MB  (39% reduction)
```

The int8 activation scratch is overlaid on existing bf16 scratch slot 2 at
zero additional cost. Column sums add ~622 KB to the weight stride (negligible).

---

## 10. Implementation Checklist

### Offline (quantization)

1. For each norm: read gamma vector from source (marked `IsAbsorbed`)
2. For each projection following a norm: `W'[n,k] = W[n,k] * gamma[k]`
3. For each projection: FWHT each row (K dimension) with block = FWHT_BLOCK
4. Per-row absmax quantize to int8
5. Store int8 weights + f32 per-row scales (norms absent from output)

Orchestrated by `quant/engine.mojo`. The model spec drives dispatch via
`for_each_weight` with `conforms_to` gating on `Absorbed`, `Gamma`,
`Quantizable`, and `Passthrough` traits.

### Load time

1. Load int8 weights + f32 scales from quantized safetensors (loader skips
   `IsAbsorbed` entries)
2. VNNI-pack weights in-place using scratch as temp
3. Compute per-row column sums from packed i8 weights (`init_column_sums`)
4. Initialize RoPE cos/sin tables

### Online (per layer, per token)

```
 1. rms_fwht_quantize(x)              → act_i8, act_sc
 2. int8_gemm(act_i8, Q')             → q (bf16)
 3. int8_gemm_k_to_cache(act_i8, K')  → K cache (fused gemm + RoPE + FWHT + quant)
 4. int8_gemm_v_to_cache(act_i8, V')  → V cache (fused gemm + FWHT + quant)
 5. int8_gqa_attention(q, kv_cache)   → act_i8, act_sc (RoPE on Q fused, int8 out)
 6. int8_gemm(act_i8, O')            → o_out
 7. allreduce, residual add
 8. rms_fwht_quantize(x')            → act_i8, act_sc
 9. int8_gemm_gate_up(act_i8, G', U') → gate, up (one activation read)
10. silu_fwht_quantize(gate, up)      → act_i8, act_sc
11. int8_gemm(act_i8, DOWN')          → down_out
12. allreduce, residual add
```

### New kernels

- `rms_fwht_quantize` — fused norm + FWHT rotation + dual reduction + int8 quantize
- `silu_fwht_quantize` — fused SiLU + FWHT rotation + int8 quantize
- `int8_gemm` — VNNI/AMX with column sum bias correction
- `int8_gemm_k_to_cache` — fused K gemm epilogue + RoPE + FWHT + int8 cache write
- `int8_gemm_v_to_cache` — fused V gemm epilogue + FWHT + int8 cache write
- `int8_gemm_gate_up` — fused GATE+UP gemm (one activation read, two outputs)
- `int8_gqa_attention` — fused RoPE on Q + int8 scoring + V scale absorption + int8 output

### Unchanged from bf16

- Residual connections
- Allreduce (same positions, same data)
- Embedding lookup
- Final norm (gamma-free: x / rms(x))
- BurstPool dispatch, NUMA arena layout, TP sharding logic
