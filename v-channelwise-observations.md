# V Channelwise Quantization Observations

## Measurement Context

MiniMax-M2.7 (62 layers, 8 KV heads, HEAD_DIM=128) with ButterQuant int8
quantization. V projection weights use two-sided Hadamard rotation (FWHT on
both contraction and output dimensions at offline quantization time). V cache
entries are written as i8 with a dynamic absmax scale.

Two V scale granularities were compared:

- **Per-position dynamic absmax** (current): one scale per KV head per
  sequence position, computed at cache-write time as the absmax of the
  128-element V vector for that position.
- **Per-channel fixed scale** (proposed): one scale per channel across all
  positions, derived from the global absmax of each channel over the measured
  context window.

Measurements were taken over a 42-position context across all 62 layers (20,336
total V vectors analyzed).

## Cosine Similarity

| Granularity | cos min | cos mean |
|-------------|---------|----------|
| Per-position dynamic absmax | 0.99992 | 0.99998 |
| Per-channel fixed scale | 0.99455 | 0.99994 |

Mean cosine similarity degrades by approximately 0.00004. Worst-case cosine
similarity degrades from 0.99992 to 0.99455, a shift of 0.00537.

## Scale Utilization

Per-channel quantization achieves a mean utilization of 28.6% across all layers.
Utilization is defined as the ratio of a position's absmax to the channel-global
absmax; values below 1.0 indicate that the quantized representation uses a
fraction of the available integer range for that position.

| Layer | Mean utilization |
|-------|-----------------|
| 0 | 0.346 |
| 9 | 0.264 |
| 19 | 0.286 |
| 29 | 0.276 |
| 39 | 0.269 |
| 49 | 0.277 |
| 59 | 0.284 |
| 61 | 0.286 |

Layer 0 exhibits the highest utilization. Middle and later layers are grouped
between 0.26 and 0.29 with no clear trend.

The per-position dynamic absmax scale range spans 0.0085 to 36.5 (ratio 4272:1),
indicating large cross-position magnitude variation. This ratio is the primary
driver of low channelwise utilization: the channel-global scale must accommodate
the highest-magnitude position, leaving lower-magnitude positions quantized into
a small fraction of the int8 range.

At 28.6% mean utilization the effective precision loss is approximately
$\log_2(1 / 0.286) \approx 1.81$ bits relative to the per-position scheme.

## Per-Head Worst Cases

The five heads with the lowest per-channel cosine similarity (excluding BOS):

| Layer | Head | cos min | Position | Scale ratio |
|-------|------|---------|----------|-------------|
| 3 | 7 | 0.99455 | 20 | 41.0 |
| 3 | 2 | 0.99879 | 28 | 23.0 |
| 3 | 6 | 0.99886 | 19 | 23.8 |
| 3 | 4 | 0.99899 | 19 | 24.8 |
| 35 | 5 | 0.99895 | 6 | 18.4 |

Four of the five worst cases are in layer 3. Layer 35 head 5 operates at
notably higher absolute magnitudes (scale range 0.85 to 15.6) compared to the
layer 3 heads (scale ranges starting below 0.03), suggesting a different
distribution shape rather than the same outlier mechanism.

## Layer-by-Layer Cosine Detail

| Layer | Per-pos cos min | Per-pos cos mean | Per-ch cos min | Per-ch cos mean |
|-------|----------------|-----------------|---------------|----------------|
| 0 | 0.99996 | 0.99998 | 0.99934 | 0.99996 |
| 9 | 0.99994 | 0.99998 | 0.99455 | 0.99989 |
| 19 | 0.99994 | 0.99998 | 0.99455 | 0.99993 |
| 29 | 0.99994 | 0.99998 | 0.99455 | 0.99993 |
| 39 | 0.99992 | 0.99998 | 0.99455 | 0.99992 |
| 49 | 0.99992 | 0.99998 | 0.99455 | 0.99993 |
| 59 | 0.99992 | 0.99998 | 0.99455 | 0.99994 |
| 61 | 0.99992 | 0.99998 | 0.99455 | 0.99994 |

The per-channel cos min saturates at 0.99455 from layer 9 onward, indicating
the same head (layer 3 head 7) remains the global worst case as layers
accumulate.

## Observations

1. Per-channel V quantization produces measurably lower fidelity than
   per-position dynamic absmax. The gap is concentrated in cosine similarity
   worst cases and in utilization; mean cosine similarity remains above 0.9999
   across all layers.

2. The 28.6% mean utilization reflects cross-position magnitude variation that
   the feature-axis Hadamard rotation does not address. The Hadamard equalizes
   energy across the HEAD_DIM channels within each position but does not affect
   the distribution of magnitudes across positions.

3. For MiniMax-M2.7, channelwise V quantization was measured as lossier than
   positionwise but not to a degree that supports a conclusion of harm to model
   output quality at the granularity of these measurements. End-to-end output
   evaluation would be required to determine whether the observed fidelity
   reduction is consequential.

4. The two-sided Hadamard rotation on V weights (FWHT on both contraction and
   output dimensions) is orthogonal to the channelwise-vs-positionwise question.
   The output-side rotation operates within each position's feature vector and
   does not influence cross-position magnitude variation.
