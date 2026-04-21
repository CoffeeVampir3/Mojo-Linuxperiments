# MiniMax M2.7 router — f32 is load-bearing

The router projection `block_sparse_moe.gate.weight` is stored and computed
in f32. This is not an oversight and not a dtype holdover. It is load-bearing
for routing correctness.

## What the weights look like

Across all 62 layers (48,758,784 elements):

- absmax: 0.0773
- std: 0.0333
- abs_mean: 0.0271
- no zeros, no subnormals, no outliers
- 99th percentile of `|w|` is 6.32e-2

The values occupy roughly `[2^-11, 2^-4]`. bf16 round-trips this range with
max relative error 3.9e-3 (one bf16 ULP), matching the dtype's theoretical
floor. The weights themselves are bf16-friendly.

## Why f32 is required anyway

A live probe at the router/attention intersection (rank 0 `normed_bf16`,
real f32 weights, 4 decode tokens × 62 layers = 248 events) recomputed the
router pipeline in parallel with f32 weights and bf16-roundtripped weights.
For each event we measured:

- `gap89`: biased-score margin between f32 pick #8 and pick #9 (selection
  safety margin)
- `noise`: max `|score_f32 − score_bf16|` over the f32-picked top-8

`gap89` spans four decades: min 1.81e-5, median ~3e-3, max 0.28. `noise` is
tightly bounded at 1e-4 to 1.8e-3 — bf16's representational floor.

Whenever `gap89 < noise`, the bf16 path selects a different top-8 set.
Across 248 events:

- 8 selection changes (3.2% of decisions)
- worst case: 3 of 8 experts swapped (layer 3, token 0)
- worst renormalized routing-weight drift on a flipped pick: 2.5%
- multiple events with `gap89 < 1e-4`, i.e. selection decided at or below
  the bf16 noise floor

## Mechanism

The router is a discrete-selection function layered on top of continuous
scoring. Expert scores are frequently near-tied at the 8/9 boundary — the
biased-score distribution produces `gap89` values routinely in the 1e-5
to 1e-4 range. bf16 weight rounding perturbs scores by 1e-4 to 1e-3. The
two distributions overlap, and every overlap is a flipped decision.

The weight *distribution* is bf16-friendly in isolation. The *decision*
built on top of that distribution is not. f32 keeps the score perturbation
below the selection margin across all observed layers.

## Consequences

- `block_sparse_moe.gate.weight` stays f32.
- `e_score_correction_bias` stays f32 for the same reason (added directly
  to the score).
- Router bandwidth reduction must come from algorithmic changes (row-shard
  by expert ownership across TP ranks — bit-exact), not from dtype
  reduction.
- Any future attempt to quantize the router must be validated against the
  same probe: f32 and quantized paths run in parallel, measured by
  `gap89 vs noise` ratio across many tokens and layers.
