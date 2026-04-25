# Router Gauge Correction

This note documents the motivation and measured tradeoff for using a gauge
correction when storing MiniMax M27 router weights in bf16.

## Problem

The M27 router currently uses fp32 router weights. The router score used for
top-k selection is:

```text
score[e] = sigmoid(x @ W[e]) + correction_bias[e]
```

The expert weights are already quantized, but the router weights being fp32 are
expensive because every token/chunk must evaluate the full expert distribution.
The goal is to avoid the fp32 router matrix path while keeping routing behavior
close to the fp32 baseline.

Naively casting router weights to bf16 is already close, but not exact. On the
current full prompt dump surface:

```text
events: 2604 layer/position router decisions
candidate scores: 666624

naive bf16(W):
  exact top8 order: 2485 / 2604 = 95.43%
  exact top8 set:   2567 / 2604 = 98.58%
  fp32 top8 contained in bf16 top12: 2602 / 2604
```

The remaining mismatches are mostly near the top-k boundary. Many fp32
rank-8/rank-9 gaps are very small, so exact fp32 agreement is partly preserving
a numerical tie-break convention rather than a clearly meaningful routing
preference.

## Gauge Transform

For each router layer, compute the mean router weight vector across experts:

```text
g[k] = mean_e W[e, k]
```

Then store:

```text
W_center[e, k] = bf16(W[e, k] - g[k])
g_bf16[k]      = bf16(g[k])
```

At runtime:

```text
delta[e] = x @ W_center[e]
pivot    = x @ g_bf16
score[e] = sigmoid(delta[e] + pivot) + correction_bias[e]
```

This avoids the fp32 router matrix. The additional runtime work is one bf16 dot
per row for the shared pivot, not a second full router GEMM.

The extra storage is one hidden-size bf16 vector per router layer.

## Why The Pivot Matters

For plain logit ranking, subtracting a shared token-dependent value would not
change top-k:

```text
argsort(z) == argsort(z - c)
```

But M27 does not rank raw logits. It ranks:

```text
sigmoid(logit) + correction_bias
```

That expression is not shift-invariant. Dropping the pivot changes the relative
strength of the sigmoid term versus the correction bias and badly damages
routing quality.

Measured variants:

```text
naive bf16(W):        exact top8 set 2567 / 2604 = 98.58%
gauge no pivot:       exact top8 set 1122 / 2604 = 43.09%
gauge constant pivot: exact top8 set 1582 / 2604 = 60.75%
gauge bf16 pivot:     exact top8 set 2577 / 2604 = 98.96%
gauge fp32 pivot:     exact top8 set 2577 / 2604 = 98.96%
```

The useful version is therefore the bf16-pivot form. It preserves the router
score semantics without reintroducing the fp32 router matrix.

## Observed Benefit

Compared with naive bf16:

```text
naive bf16(W):
  exact top8 set: 2567 / 2604
  set mismatches: 37
  fp32 top8 contained in bf16 top12: 2602 / 2604

gauge bf16 pivot:
  exact top8 set: 2577 / 2604
  set mismatches: 27
  fp32 top8 contained in bf16 top12: 2604 / 2604
```

This is a small absolute gain, but it removes about 27% of the remaining
top8-set mismatches versus naive bf16 on this dump surface. It also made top12
containment complete in the measured run.

## Why Stop Here

More complex residual schemes are possible, such as low-rank residual
correction:

```text
R = W - Wq
R ~= U @ V
correction[e] = (x @ U) @ V[e]
```

However, this adds runtime and storage complexity to chase mostly near-tie
behavior. Since the router itself often has very small boundary gaps, matching
fp32 exactly is not obviously equivalent to a semantically better routing
decision.

The gauge correction is simple, cheap, and targeted:

- It removes the fp32 router matrix path.
- It adds only one bf16 pivot dot per row.
- It requires only one extra bf16 hidden-size vector per layer.
- It measurably improves routing agreement over naive bf16.
- It avoids more elaborate residual machinery whose correctness value is hard
  to justify.

## Intended Implementation Direction

For router quantization/conversion:

1. For each layer, compute `g[k] = mean_e W[e, k]` in fp32 offline.
2. Store centered router weights as `bf16(W[e, k] - g[k])`.
3. Store `g` as bf16.

For router execution:

1. Compute the router GEMM/GEMV over centered bf16 weights.
2. Compute one bf16 pivot dot per row.
3. Add the pivot to every expert logit for that row.
4. Apply the existing `sigmoid(logit) + correction_bias` scoring.
5. Run the existing top-k selection.

Decode is the `row_count = 1` case. Prefill is the same operation over a chunk
of rows.
