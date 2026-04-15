# Quant Analysis

This directory contains small end-to-end mocks for the live non-head
quantization schemes in `gemma_4_moe_butterquant_tp.mojo`.

Excluded:
- Tied embedding / LM-head smooth-split scheme

Included:
- `router.mojo`
  Representative of the generic rotated rowwise linear scheme:
  `rmsnorm * gamma -> FWHT -> rowwise absmax i8 -> corrected int GEMV`
  Also reports a `philox_fwht` variant using a full Philox-derived random sign
  diagonal before the FWHT.
- `dense_ffn.mojo`
  Representative of the post-nonlinearity blocked scheme:
  `entry quant -> gate/up GEMV -> gelu_tanh * up -> FWHT(64) -> per-block i8 -> down GEMV`
  Also reports a `philox_fwht` variant on both the entry and post-activation
  re-entry transforms.
- `sliding_attention.mojo`
  Representative of the attention subsystem before `o_proj`:
  `Q/K/V prep -> quantized score -> quantized V aggregation -> final head quantization`
- `dense_ffn_stack.mojo`
  Multi-layer rollout for the hidden-state-preserving dense FFN path.
  Reports both teacher-forced local error and closed-loop accumulated drift
  for `fwht`, `philox_fwht`, and an alternating reflected-antithetic
  `A/B/A/B` pairing schedule.
- `dense_ffn_stack_300.mojo`
  Long-rollout version of the same harness with `300` layers and reduced
  trials, printing early layers plus periodic checkpoints.
- `dense_ffn_pair_cancellation.mojo`
  Direct paired-cancellation test for the dense FFN path. Reports the cosine
  between two paired error vectors and the RMSE of their average against the
  bf16 baseline.
- `dense_ffn_stack_covariance.mojo`
  Ten-layer closed-loop covariance analysis across `fwht`, `philox_fwht`, and
  the alternating antithetic schedule.
- `dense_ffn_block_autocovariance.mojo`
  Local transformed-block autocovariance of quantization error before the next
  matched linear contraction, for both the entry (`256`) and post-activation
  (`64`) FFN quantization sites.
- `dense_ffn_stack_terminal_direction.mojo`
  Medium-depth (`24` layer) directional analysis. For each scheme, forms the
  terminal drift direction `v = eta_L / ||eta_L||`, then reports per-layer
  alignment `<v, eta_l> / ||eta_l||` and the mean/variance of the clean-input
  injection projection `<v, xi_l>`.

Each script compares the quantized path to an equivalent bf16 baseline and
reports:
- RMSE
- Cosine similarity

Synthetic activation-like tensors are sampled from a skewed heavy-tail
distribution with rare positive outliers rather than a plain Gaussian.

`dense_ffn_stack.mojo` currently uses a stronger stress profile than the
single-layer mocks so accumulated drift is easier to observe.

The active randomized variant uses a full sign diagonal recovered from Philox
bits, not the degenerate Walsh/Kronecker sign family. For a 256-wide FWHT
block, this needs 256 random sign bits, i.e. only two `Philox.step()` calls,
because each call returns 128 random bits.

Run:

```bash
pixi run mojo build -I . quant_analysis/router.mojo
pixi run mojo build -I . quant_analysis/dense_ffn.mojo
pixi run mojo build -I . quant_analysis/sliding_attention.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_stack.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_stack_300.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_pair_cancellation.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_stack_covariance.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_block_autocovariance.mojo
pixi run mojo build -I . quant_analysis/dense_ffn_stack_terminal_direction.mojo

./router
./dense_ffn
./sliding_attention
./dense_ffn_stack
./dense_ffn_stack_300
./dense_ffn_pair_cancellation
./dense_ffn_stack_covariance
./dense_ffn_block_autocovariance
./dense_ffn_stack_terminal_direction
```
