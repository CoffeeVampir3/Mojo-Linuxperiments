# Quantized Hadamard Implementation Guide

This document provides the theoretical foundation, algebraic structure, and
implementation-relevant insights for the Hadamard-rotated int8 inference
kernels. It is written for the implementer — the person writing the SIMD
kernels, the attention logic, and the fused operations. It does not prescribe
implementations but lays out what must hold and why.

The model spec, slot layout, forward pass wiring, and quantizer are already
implemented. What remains is the kernel implementations behind the stub
signatures in `modeling/smollm2_hadquant_tp.mojo`.

---

## 1. The Algebraic Invariant

Every quantized matmul in the system has the same structure:

```
output[n] = <Q(H · a), Q(H · W[n])> · scales ≈ <H · a, H · W[n]> = <a, W[n]>
```

H is the block-diagonal Walsh-Hadamard matrix (block = HEAD_DIM = 64). Q(·)
is the int8 quantize-dequantize operation. The rotation H enters from both
sides and cancels via Parseval's theorem: `<Hx, Hy> = <x, y>` for any
orthonormal H.

The net computation is the standard unrotated matmul. The rotation exists
solely to improve how well int8 represents the intermediate values — it
equalizes magnitudes across components, making the uniform quantization step
near-optimal.

Three identities power the entire design:

1. **Parseval**: `<Hx, Hy> = <x, y>` — rotation cancels in dot products
2. **Norm preservation**: `rms(Hx) = rms(x)` — enables fused norm + rotation
3. **Distributive law**: `W · diag(γ) · x = (W · diag(γ)) · x` — gamma absorption

---

## 2. The Fast Walsh-Hadamard Transform

### 2.1 The Butterfly

For a vector of length n (power of 2), the FWHT performs log2(n) butterfly
stages. At each stage s (s = 0, 1, ..., log2(n)-1), pairs of elements at
stride 2^s are replaced by their sum and difference:

```
for stride in [1, 2, 4, ..., n/2]:
    for each pair (i, i+stride):
        a, b = buf[i], buf[i+stride]
        buf[i]        = a + b
        buf[i+stride] = a - b
```

A final scaling by 1/sqrt(n) makes the transform orthonormal and self-inverse:
H · H = I.

Cost: n · log2(n) additions + n multiplies (the final scaling).

For block=64: 6 butterfly stages, 64 additions per stage = 384 additions + 64
multiplies = 448 ops per block. A 576-element row has 9 independent blocks.

### 2.2 SIMD Structure

For block=64 with AVX-512 (16 f32 lanes), 4 ZMM registers hold the block:

```
Stage 0 (stride 1):   in-register shuffle + add/sub
Stage 1 (stride 2):   in-register shuffle + add/sub
Stage 2 (stride 4):   in-register shuffle + add/sub
Stage 3 (stride 8):   in-register shuffle + add/sub
Stage 4 (stride 16):  full-vector add/sub (stride = SIMD width, no shuffle)
Stage 5 (stride 32):  full-vector add/sub (register pairs)
```

The first 4 stages require shuffle instructions to rearrange elements within
registers. The interleave patterns for these shuffles follow the same butterfly
network structure as the transpose in `simd_math/matrixops.mojo`. The last 2
stages are pure SIMD add/sub at maximum throughput.

The 1/sqrt(64) = 0.125 final scaling is a single multiply per register.

### 2.3 Consistency Requirement

The offline quantizer (`quant/ops.mojo`) and the runtime kernels must produce
identical FWHT results for the rotation cancellation to hold. Both must use:
- The same butterfly ordering (stride doubling: 1, 2, 4, 8, 16, 32)
- The same normalization (1/sqrt(n) applied after all stages)
- The same floating-point evaluation order within each stage

If the SIMD kernel reorders operations for efficiency, the quantizer's scalar
code must produce the same result up to float32 rounding. In practice, the
butterfly is commutative and associative in exact arithmetic — the only
concern is the accumulation of f32 rounding errors across 6 stages, which is
negligible.

---

## 3. Quantization

### 3.1 Per-Row Absmax Int8

The fundamental quantization operation:

```
scale = max(|row|) / 127
qi[k] = clamp(round(row[k] / scale), -128, 127)
```

Dequantization: `row[k] ≈ qi[k] · scale`

The scale is one f32 scalar per row. The int8 values use the symmetric range
[-127, 127] (not [-128, 127]) to keep the step size uniform around zero.
The value -128 is used only for clamping, not as a target.

### 3.2 The VNNI u8/i8 Convention

The AVX-512 VNNI instruction `vpdpbusd` computes u8 × i8 → i32. Activations
must be unsigned, weights signed. Activations are stored as i8 and converted
to u8 at register load by XORing the sign bit:

```
u8_val = i8_val ^ 0x80
```

This maps: -128 → 0, 0 → 128, 127 → 255.

The shift introduces a bias per output element:

```
raw_acc[m,n] = Σ_k u8(act[m,k]) · i8(weight[n,k])
             = Σ_k (i8_act[m,k] + 128) · i8_weight[n,k]
             = Σ_k i8_act · i8_weight  +  128 · Σ_k weight[n,k]
```

The bias term `128 · colsum[n]` is per output row of the weight, independent
of the activation. It is precomputed at load time as f32 per weight row and
stored in the arena alongside each weight (`*_COLSUM` slots).

### 3.3 The Int8 GEMM Epilogue

```
corrected = float(raw_acc[m,n]) - 128.0 · colsum[n]
output[m,n] = bf16(corrected · act_scale[m] · weight_scale[n])
```

Three f32 operations per output element: subtract bias, multiply activation
scale, multiply weight scale, cast to bf16.

For `rms_fwht_quantize` outputs, the activation scale encodes both the
quantization range and the rms normalization:

```
act_scale = absmax(Hx) / (rms(x) · 127)
```

The kernel does not distinguish this from a standard scale.

---

## 4. Fused Kernel Contracts

### 4.1 rms_fwht_quantize

**Purpose**: Fused RMSNorm + FWHT rotation + int8 quantization. Replaces
separate norm → bf16 → rotate → quantize with a single-pass operation.

**Inputs**: bf16 activation [M, K], eps
**Outputs**: int8 qi [M, K], f32 scale [M]

**Computation per row m**:

```
x_rot = FWHT(x[m])                          rotate raw (unnormalized) input
rms = sqrt(Σ x_rot[k]² / K + eps)           Parseval: rms(Hx) = rms(x)
absmax = max(|x_rot[k]|)
scale[m] = absmax / (rms · 127)             normalization folded into scale
qi[m,k] = round(x_rot[k] · rms · 127 / absmax)
```

The key insight: rotate FIRST, then compute both rms and absmax in a single
pass over the rotated values. No intermediate normalized vector is
materialized. The rms factor ends up in the scale, and the int8_gemm epilogue
reconstructs `W · γ · (x / rms)` = `W · RMSNorm(x)`.

The FWHT and the dual reduction (sum-of-squares + absmax) should share the
same register file. Load bf16, convert to f32, butterfly in 4 registers,
reduce, quantize, store int8. One bf16 read, one int8 write per block.

### 4.2 silu_fwht_quantize

**Purpose**: Fused SiLU activation + FWHT + int8 quantization.

**Inputs**: bf16 gate [M, K], bf16 up [M, K]
**Outputs**: int8 qi [M, K], f32 scale [M]

**Computation per row m**:

```
silu[k] = gate[k] · σ(gate[k]) · up[k]      SiLU elementwise
silu_rot = FWHT(silu)                          rotate
scale[m] = max(|silu_rot|) / 127
qi[m,k] = round(silu_rot[k] / scale[m])
```

Two bf16 reads (gate + up), one int8 write. The SiLU involves exp() — a fast
f32 approximation is available in `simd_math/ops.mojo` (`exp_f32`). The
sigmoid is `1 / (1 + exp(-x))`.

### 4.3 int8_gemm

**Purpose**: Int8 matrix multiply with VNNI u8/i8 accumulation and per-row
scale epilogue.

**Inputs**: int8 activation [M, K], f32 act_scale [M], int8 weight [N, K]
(VNNI-packed), f32 weight_scale [N], f32 weight_colsum [N]
**Outputs**: bf16 output [M, N]

**Computation**:

```
For each (m, n):
    raw = Σ_k u8(act[m,k] ^ 0x80) · i8(weight[n,k])   vpdpbusd
    corrected = float(raw) - 128.0 · colsum[n]
    output[m,n] = bf16(corrected · act_scale[m] · weight_scale[n])
```

Maps to `vpdpbusd` (AVX-512 VNNI) or `tdpbusd` (AMX). The weight is
VNNI 6D packed — the byte layout within tiles is determined by the
`VnniPacked` packing strategy and the `Kernel3DTiling[32, 64, 32]` tiling
contract on each weight slot.

The activation is stored as i8 in row-major layout. The u8 conversion (XOR
0x80) happens at register load — no memory format change.

### 4.4 int8_gemm_k_to_cache

**Purpose**: Fused K projection + RoPE + FWHT + quantize + cache write. K
never materializes as bf16.

**Inputs**: int8 activation [M, K], f32 act_scale [M], int8 K weight
[KV_HIDDEN, K] (VNNI-packed), f32 weight_scale, f32 colsum,
f32 cos_table, f32 sin_table, pos
**Outputs**: int8 K cache [M entries at pos..pos+M-1], f32 K cache scale

**Computation per row m, per head g**:

```
For each output element d in head g:
    raw = Σ_k u8(act[m,k] ^ 0x80) · i8(K_weight[g·HEAD_DIM+d, k])
    k_bf16[d] = (float(raw) - 128·colsum) · act_scale · weight_scale

RoPE(k_bf16, HEAD_DIM, pos + m)             position-encode
k_rot = FWHT(k_bf16)                        per-head rotation
k_scale = max(|k_rot|) / 127
k_qi = round(k_rot / k_scale)
write k_qi → K_CACHE[pos+m, g]
write k_scale → K_CACHE_SCALE[pos+m, g]
```

The tile epilogue processes HEAD_DIM output elements (one complete head), then
applies RoPE + FWHT + quantize before moving to the next head. The head-sized
intermediate lives in registers (64 floats = 4 ZMM registers).

For M > 1 (prefill), each row m gets RoPE at position `pos + m`.

### 4.5 int8_gemm_v_to_cache

**Purpose**: Same as K but without RoPE. V is not position-encoded.

**Inputs**: Same as K minus cos/sin tables.
**Outputs**: int8 V cache, f32 V cache scale.

### 4.6 int8_gemm_gate_up

**Purpose**: Fused GATE + UP projection. Reads the quantized activation once
and produces both bf16 outputs, halving activation bandwidth.

**Inputs**: int8 activation [M, K], f32 act_scale [M], GATE weight + scale +
colsum, UP weight + scale + colsum
**Outputs**: bf16 gate [M, INTERMEDIATE/tp], bf16 up [M, INTERMEDIATE/tp]

Each output element computes two independent dot products against the same
activation row — one for GATE, one for UP. The implementation should maximize
reuse of loaded activation tiles across both weight matrices.

### 4.7 int8_gqa_attention

**Purpose**: Full int8 GQA attention with fused RoPE on Q, pre-rotated int8
KV cache, V scale absorption, and direct int8 output.

**Inputs**: bf16 Q [M, HIDDEN/tp], int8 K cache + scale, int8 V cache + scale,
f32 cos/sin tables, pos
**Outputs**: int8 qi [M, HIDDEN/tp], f32 scale [M]

This is the most complex kernel. It operates per query head with the following
data flow:

```
For each query row m, for each head h (kv group g = h / GQA_FACTOR):

    q_head = Q[m, h·HEAD_DIM : (h+1)·HEAD_DIM]        bf16 → f32
    RoPE(q_head, pos + m)                                position encode
    FWHT(q_head)                                         rotate
    qi_q, q_sc = quantize(q_head)                        int8

    context = pos + m + 1                                causal boundary
    for t in 0..context:
        score[t] = Σ_d qi_q[d] · qi_K[t,g,d]            int8 dot
        score[t] *= q_sc · k_scale[t,g] / √HEAD_DIM     rescale

    w = softmax(score[0..context])                       f32, numerically stable

    for t in 0..context:
        w_abs[t] = w[t] · v_scale[t,g]                  V scale absorption

    qi_w, w_sc = quantize(w_abs)                         int8 weights

    for d in 0..HEAD_DIM:
        out[d] = Σ_t qi_w[t] · qi_V[t,g,d]              int8 aggregation
        out[d] *= w_sc                                    rescale (f32)

    qi_out[m, h·HEAD_DIM..], sc_out[m] = quantize(out)   int8 output
```

**Key implementation notes**:

- **Causal masking**: For prefill (M > 1), query m can attend to cache
  positions 0..pos+m. The context length varies per query row. For decode
  (M=1), all positions 0..pos are visible.

- **V scale absorption**: The softmax weights (summing to 1) are multiplied by
  the per-entry V scales BEFORE quantizing to int8. This folds the varying V
  scales into the attention weights, enabling a single scalar w_sc for the
  aggregation epilogue. Without absorption, each V entry's scale would vary
  across the contraction dimension T, preventing clean i32 accumulation.

- **Output domain**: The aggregation computes a weighted sum of FWHT-rotated V
  entries. The output is in the per-head rotated domain. It feeds the O
  projection, whose weight is rotated with block = HEAD_DIM. The rotation
  cancels: `<H·attn_out, H·O_weight[n]> = <attn_out, O_weight[n]>`.

- **RoPE on Q**: Applied per head BEFORE the FWHT. RoPE and FWHT do not
  commute pointwise, but Parseval guarantees the inner product is preserved:
  `<FWHT(RoPE(Q)), FWHT(RoPE(K))> = <RoPE(Q), RoPE(K)>`. This was validated
  numerically at full SmolLM2 dimensions.

- **Decode vs prefill**: For decode (M=1), the scoring is a GEMV per head —
  one query against T keys. For prefill (M=large), the scoring is a triangular
  GEMM per head. These have different optimal implementations but the same
  interface.

### 4.8 rms_norm_no_gamma

**Purpose**: RMSNorm without gamma. Used for the final norm only (gamma is
not absorbed — the final norm has no downstream projection).

**Inputs**: bf16 input [M, K], eps
**Outputs**: bf16 output [M, K]

```
output[m,k] = input[m,k] / rms(input[m])
```

Standard scalar-division normalization. No FWHT, no gamma, no quantization.

---

## 5. KV Cache

### 5.1 Format

The KV cache stores pre-rotated int8 values with per-head f32 scales:

```
K_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN/tp]
V_CACHE:       int8  [MAX_SEQ_LEN, KV_HIDDEN/tp]
K_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS/tp]
V_CACHE_SCALE: f32   [MAX_SEQ_LEN, NUM_KV_HEADS/tp]
```

Each cache entry at position t contains KV_HIDDEN/tp int8 values organized as
NUM_KV_HEADS/tp heads of HEAD_DIM elements each. The per-head scale
reconstructs the original magnitude: `k_float[d] ≈ k_qi[d] · k_scale`.

### 5.2 Write Path

K and V are written by the fused gemm-to-cache kernels (steps 3, 4). The gemm
epilogue produces f32 values one head at a time, applies any per-head
operations (RoPE for K, nothing for V), FWHT-rotates, quantizes, and writes
directly to the cache. No bf16 intermediate exists.

For prefill (M > 1), M positions are written to the cache in sequence. Each
position gets its own RoPE angle (for K) and its own per-head scale.

### 5.3 Read Path

K and V cache entries are read by the attention kernel (step 5). They are
already rotated — no FWHT is applied at read time. The int8 values are used
directly in the scoring dot products (K) and the aggregation weighted sums (V).

The K scale factors into the scoring epilogue. The V scale is absorbed into
the attention weights before the aggregation.

### 5.4 Why Pre-Rotated

With bf16 cache, every attention call would FWHT-rotate the entire context
(T entries × KV_HIDDEN elements × log2(HEAD_DIM) ops per element) — repeated
every layer, every token. With int8 pre-rotated cache, the rotation happens
once at write time. At attention time, only Q needs rotation (HEAD_DIM elements
per head, once per token).

The pre-rotated format also halves cache memory (int8 vs bf16), which matters
for long contexts.

Numerically, pre-rotating at write time produces bit-identical results to
rotating on the fly at read time. This was validated at full SmolLM2 dimensions
(HEAD_DIM=64, CONTEXT=256, 9 heads, 3 KV heads).

---

## 6. Domain Transitions

At any point in the layer, each tensor is in one of two domains:

- **Original domain**: the standard representation (bf16 between operations)
- **Rotated domain**: FWHT-rotated, quantized to int8 (transient)

The transitions:

```
Original → Rotated:  rms_fwht_quantize, silu_fwht_quantize, KV cache write,
                     Q rotation inside attention
Rotated → Original:  int8_gemm (Parseval cancellation in the epilogue)
```

The rotated domain exists only between a quantize operation and its consuming
int8_gemm. The one exception is the KV cache, where rotated values persist
across tokens. The attention output is also transiently rotated (it comes from
a weighted sum of rotated V entries) but is quantized to int8 immediately and
consumed by the O projection gemm.

The residual stream (x_main) is always in the original domain. Every branch
departs (rotate + quantize), computes (int8 matmul), and returns (Parseval
cancellation) before the residual add.

---

## 7. Gamma Absorption

RMSNorm gamma is absorbed into projection weights offline:

```
W'[n,k] = W[n,k] · γ[k]
```

Column-wise scaling, computed once by the quantizer. After absorption, the
runtime norm is just `x / rms(x)` — no elementwise gamma multiply.

Which weights absorb which gamma:

```
input_layernorm.weight     → Q, K, V     (share the same normed input)
post_attention_layernorm.weight → GATE, UP (share the same normed input)
O, DOWN                    → nothing     (no preceding norm)
```

The norm weights are marked `IsAbsorbed` in the model spec. They are read
from the source bf16 file during quantization, used for column-wise scaling,
and not written to the output. At runtime they do not exist.

---

## 8. Memory Layout

### 8.1 Weights in the Arena (per layer)

```
Q_PROJ      int8 [HIDDEN/tp, HIDDEN]           VNNI-packed
K_PROJ      int8 [KV_HIDDEN/tp, HIDDEN]        VNNI-packed
V_PROJ      int8 [KV_HIDDEN/tp, HIDDEN]        VNNI-packed
O_PROJ      int8 [HIDDEN, HIDDEN/tp]            VNNI-packed
GATE_PROJ   int8 [INTERMEDIATE/tp, HIDDEN]      VNNI-packed
UP_PROJ     int8 [INTERMEDIATE/tp, HIDDEN]      VNNI-packed
DOWN_PROJ   int8 [HIDDEN, INTERMEDIATE/tp]      VNNI-packed
*_SCALE     f32  [rows, 1]                       per weight (7 total)
*_COLSUM    f32  [rows, 1]                       per weight (7 total, init at load)
```

Column sums are computed from the row-major weights BEFORE VNNI packing.
After packing, the byte layout is permuted and a naive row-major scan would
produce incorrect sums.

### 8.2 Activation Scratch

Three bf16 scratch slots at INTERMEDIATE width:

```
Slot 0:  Q output (attention block), gate output (MLP block)
Slot 1:  up output (MLP block)
Slot 2:  int8 activation overlay (fully available — K and V go to cache directly)
```

K and V never materialize as bf16 — they go directly from the gemm epilogue to
the int8 cache. This frees slot 2 entirely for the int8 activation + scale.

The int8 activation at HIDDEN width (4.7 MB for SmolLM2) and at INTERMEDIATE
width (12 MB) both fit in slot 2 (24 MB). The per-row f32 activation scale
(32 KB) is placed after the int8 data.

### 8.3 RoPE Tables

Precomputed at model load time. Shared across layers. Passed to the K cache
write kernel and the attention kernel.

### 8.4 Column Sums

Computed at model load time from the row-major int8 weights BEFORE VNNI
packing. One f32 per weight row. Same arena offset chain and per-layer
stride as the weight scales.

---

## 9. Tensor Parallelism

Megatron-style sharding:

```
RowShard (Q, K, V, GATE, UP):   output dimension sharded, K dimension full
ColShard (O, DOWN):              K dimension sharded, output dimension full
Replicated (x_main, act_i8):    all ranks identical
```

Communication pattern per layer:
- Allreduce after O projection (sum ColShard partial results)
- Allreduce after DOWN projection (same)
- Embedding broadcast at the start (once, outside layer loop)

The int8 KV cache is ColShard — each rank stores its local KV heads. The
attention is per-rank, per-head. No cross-rank communication for attention.

The FWHT block size (64) divides all sharded K dimensions. For SmolLM2:
HIDDEN/tp=576/tp, KV_HIDDEN/tp=192/tp, INTERMEDIATE/tp=1536/tp. For tp=1
and tp=3, all divisions by 64 are exact.

The column sum bias correction is per-shard. Each rank corrects its local
partial sum using its local column sums. After allreduce, the corrections
sum to the full correction.

---

## 10. Layer Flow (12 Steps)

```
 1. rms_fwht_quantize(x)              → act_i8, act_sc
 2. int8_gemm(act_i8, Q')             → q (bf16, slot 0)
 3. int8_gemm_k_to_cache(act_i8, K')  → K cache (fused gemm + RoPE + FWHT + quant)
 4. int8_gemm_v_to_cache(act_i8, V')  → V cache (fused gemm + FWHT + quant)
 5. int8_gqa_attention(q, kv_cache)   → act_i8, act_sc (fused RoPE on Q, int8 out)
 6. int8_gemm(act_i8, O')            → o_out (bf16)
 7. allreduce(o_out), x = x + o_out
 8. rms_fwht_quantize(x)             → act_i8, act_sc
 9. int8_gemm_gate_up(act_i8, G', U') → gate, up (bf16, slots 0, 1)
10. silu_fwht_quantize(gate, up)      → act_i8, act_sc
11. int8_gemm(act_i8, DOWN')          → down_out (bf16)
12. allreduce(down_out), x = x + down_out
```

After all layers: `rms_norm_no_gamma(x)` → bf16 gemm against embed table → logits.

This flow was validated numerically against a 15-step unfused flow and an f32
reference, producing bit-identical results between fused and unfused paths
(999 dB SQNR, `design_analysis/optimal_flow.mojo`).

---

## 11. Numerical Validation Summary

All numerical claims are backed by executable Mojo code in `design_analysis/`:

| Test | Result | File |
|------|--------|------|
| Hadamard rotation preserves dot products | exact (f32 precision) | `hadamard.mojo` |
| Block=64 quality plateau | 38.19 dB SQNR, +7.4 dB over channelwise | `block_sweep.mojo` |
| Stochastic vs deterministic rounding at 8 bits | deterministic wins | `sr_scaling.mojo`, `sr_depth.mojo` |
| Int8 KV cache vs bf16 KV cache | bit-identical (999 dB) | `hadamard_layer_int8_kv.mojo` |
| RoPE + FWHT correctness (Parseval) | validated at SmolLM2 dims | `hadamard_layer_int8_kv.mojo` |
| 12-step fused vs 15-step unfused | bit-identical (999 dB) | `optimal_flow.mojo` |
| Mojo quantizer vs Python reference | 422/422 pass, 100% exact | `validation/validate_hadquant.py` |

---

## 12. Memory Savings (SmolLM2-135M, TP=1)

```
                    bf16        hadquant    savings
Weights:            212 MB      107 MB      105 MB
State (KV+act):     285 MB      197 MB       88 MB
Total arena:        497 MB      304 MB      193 MB  (39%)
```
