# Gamma Split Investigation Summary

## Background

Gemma 4's ButterQuant pipeline quantizes activations and weights to i8 after
block-diagonal FWHT rotation. The RMSNorm learnable gain (gamma) creates
inter-block energy imbalance that wastes i8 precision. The "smooth split"
decomposes gamma as:

    act_side = RMSNorm(x) * sign(g) * sqrt(|g|)
    wt_side  = W * sqrt(|g|)

Both sides get FWHT-rotated and i8-quantized. The identity
`sign(g) * sqrt(|g|) * sqrt(|g|) = g` preserves the exact dot product.

## Current State

Gamma splitting is applied only to the lm_head (final_norm + embed_tokens
weight). It uses per-block quantization on both sides with a dedicated
`lm_head_gemv` kernel. Layer projections use per-row (channelwise)
quantization with the standard `int8_gemv` kernel.

## Findings

### 1. The split is universally beneficial under per-block quantization

Tested on real Gemma 4 norm weights from the bf16 checkpoint:

| Norm | mean\|g\| | max\|g\| | max/mean | per-block A/B | benefit |
|---|---|---|---|---|---|
| L0 input_norm | 4.50 | 30.6 | 6.8 | 1.04x | 4% |
| L0 pre_ffn | 2.43 | 23.3 | 9.6 | 1.24x | 19% |
| L15 input_norm | 4.79 | 72.5 | 15.2 | 1.24x | 19% |
| L15 pre_ffn | 0.39 | 6.6 | 17.1 | 1.65x | 39% |
| L29 input_norm | 9.29 | 316.0 | 34.0 | 4.77x | **79%** |
| L29 pre_ffn | 2.52 | 59.8 | 23.7 | 3.89x | **74%** |
| final_norm | 29.22 | 588.0 | 20.1 | 3.02x | **67%** |

A/B = error without split / error with split. Every norm benefits; later
layers with higher outlier ratios benefit dramatically.

### 2. Per-row quantization does not universally benefit

The decomposed error analysis (act-only, wt-only, combined) at different alpha
values reveals that per-row quantization creates a conflict:

| Norm | per-row alpha=0.5 | per-row alpha=1.0 | per-block alpha=0.5 | per-block alpha=1.0 |
|---|---|---|---|---|
| L0 input_norm | **5.94%** | 7.27% | **4.87%** | 5.05% |
| L0 pre_ffn | **5.16%** | 6.30% | **4.76%** | 5.85% |
| L15 input_norm | 23.69% | **23.18%** | **9.78%** | 12.10% |
| L15 pre_ffn | **5.00%** | 11.95% | **5.46%** | 9.00% |
| L29 input_norm | **1.65%** | 8.45% | **1.35%** | 6.46% |
| L29 pre_ffn | **4.79%** | 12.23% | **2.72%** | 10.56% |
| final_norm | **2.62%** | 8.22% | **1.69%** | 5.11% |

L15 input_norm is worse at alpha=0.5 under per-row (23.69% vs 23.18%). All
others improve. Per-block is consistently better at every layer.

The mechanism: absorbing sqrt(|gamma|) into the weight changes the per-block
energy distribution of the weight rows. Per-row quantization uses a single
absmax for the entire row, so blocks with amplified energy dominate the scale
and other blocks lose precision. Per-block quantization adapts per-block and
avoids this.

### 3. Full gamma absorption (alpha=0) is not viable

Alpha=0 eliminates gamma from the forward path entirely, which would allow
RMSNorm to stay in the Hadamard domain (pure scalar division). However:

- Error is comparable to alpha=1 (no split) for all layers
- The weight side absorbs the full gamma magnitude, creating severe
  quantization distortion
- Only alpha=0.5 consistently reduces error on both sides simultaneously

### 4. The conjugated gamma operator is full-rank within FWHT blocks

The gamma operation in the Hadamard domain is the conjugated operator
`H * diag(sqrt_gamma) * H^T`. Per-block spectral analysis shows:

| Norm | CV within blocks | rank@95% per block | rank@99% per block |
|---|---|---|---|
| L0 input_norm | 15-20% | 126 / 256 | 185 / 256 |
| L15 input_norm | 28-36% | 155 / 256 | 198 / 256 |
| L29 input_norm | 189-214% | 72 / 256 | 197 / 256 |
| final_norm | 91-110% | 67 / 256 | 194 / 256 |

At 99% variance capture, every norm requires ~194-198 of 256 dimensions per
block. The operator is essentially full-rank. Low-rank approximation of the
conjugated gamma is not viable.

### 5. The split is not rotationally invariant

The elementwise multiply `x * sqrt_gamma` does not commute with the Hadamard
transform. The booklet's Section 5.3 proves this:

    H(u * v) != (Hu) * (Hv)

Only full absorption (alpha=0) allows RMSNorm to stay in the Hadamard domain
(reduces to scalar division). The split (alpha=0.5) requires a domain exit for
the gamma multiply — identical in cost to the current approach.

The conjugated operator `H * diag(sqrt_gamma) * H^T` has butterfly-diagonal-butterfly
structure, but its application cost is O(n log n) — equal to exit + multiply +
re-enter. There is no cheaper representation.

### 6. Tensor parallelism is not a constraint

All norms that feed FWHT-quantized projections (input_layernorm,
pre_feedforward_layernorm, router.scale) are REPL (replicated across ranks).
The contraction dimension (HIDDEN) is full on each rank for column-parallel
projections. The gamma split decomposes element-wise on the contraction
dimension and is rank-independent regardless of sharding strategy.

### 7. Per-block quantization is AMX-unfriendly

Moving layer projections from per-row to per-block quantization would require
the GEMV to accumulate per-block and apply per-block scales in the inner loop.
This is computationally cheap (~1.5% overhead) but breaks AMX tile accumulation
patterns. Channelwise (per-row) scales are trivially fusible as a post-multiply;
per-block scales require restructuring the accumulation.

## Conclusion

The gamma split is mathematically sound and produces large improvements under
per-block quantization. However, layer projections use per-row quantization
(for AMX compatibility), and under per-row, the split is not universally
beneficial — some layers get slightly worse when absorbing sqrt(|gamma|)
creates energy imbalance on the weight side that per-row scales cannot
accommodate.

The split remains in production for the lm_head only, where per-block
quantization and the dedicated kernel make it unconditionally beneficial.
Extension to layer projections requires either:

1. Per-block GEMV kernel (AMX implications, more scale/colsum storage)
2. Per-layer selectivity (only split where the gamma profile benefits per-row)
3. Larger FWHT blocks (reduces the between-block variation but increases
   within-block quantization error)

The infrastructure (`experimental3/gamma.mojo`, `SmoothRowQuantized` in
model_spec) is in place for future use.

## Validation Files

- `quantization_analysis.mojo` — per-block split benefit on real norm weights
- `gamma_absorb_validation.mojo` — decomposed error analysis (act/wt/combined) across alpha values
- `gamma_spectral_analysis.mojo` — within-block rank analysis of the conjugated gamma operator
