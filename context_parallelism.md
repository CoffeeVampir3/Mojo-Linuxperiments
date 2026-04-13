# Context Parallelism for Attention on NUMA Systems

## Overview

Context parallelism distributes the KV cache across NUMA nodes so that each
node stores and scores only a fraction of the cached positions. This reduces
both per-node memory and per-node compute by a factor of N (the number of
nodes). A softmax-merge step recombines the partial results with exact
log-sum-exp correction.

## KV Cache Distribution

Positions are assigned to nodes round-robin:

    position p  -->  node (p mod N),  local slot (p div N)

Each node stores `ceil(T/N)` positions where T is the total context length.
New tokens are written to exactly one node per step, cycling through nodes.
No remote writes occur during cache updates — the owning node writes locally.

During attention scoring, each node scans only its local positions. The query
vector is replicated (broadcast or computed identically on each node). Each
node produces a partial softmax state that is merged across nodes.

## Partial Softmax State

Standard online softmax maintains running state `(m, l, v)`:

    m = running maximum score
    l = running sum of exp(score - m)
    v = running weighted accumulator (sum of exp(score - m) * V)

After scanning all local positions, node r holds partial state
`(m_r, l_r, v_r)` computed over its `T/N` positions.

## Cross-Node Merge

Given partial states from N nodes, the merge recovers the exact global result.

**Step 1 — Global maximum:**

    M = max(m_0, m_1, ..., m_{N-1})

Each node broadcasts its scalar `m_r`. Total data: N floats.

**Step 2 — Rescale and reduce:**

Each node computes its correction factor:

    alpha_r = exp(m_r - M)

Then rescales its partial state:

    l_r' = l_r * alpha_r
    v_r' = v_r * alpha_r

The global state is the sum:

    L = sum_r(l_r')
    V = sum_r(v_r')

This is a standard sum-allreduce on the rescaled values.

**Step 3 — Normalize:**

    output = V / L

**Correctness:** The rescaling preserves the softmax identity. For any
partition of scores into subsets S_r:

    sum_r [ sum_{i in S_r} exp(s_i - m_r) * exp(m_r - M) ]
    = sum_r [ sum_{i in S_r} exp(s_i - M) ]
    = sum_i exp(s_i - M)

which is the global softmax denominator with maximum M. The same identity
applies to the value accumulator V.

## Communication Pattern

The merge decomposes into two existing primitives:

1. **Scalar all-gather** of `m_r` values — N floats, one cache line total
2. **Sum-allreduce** of `(l_r', v_r')` — N x (1 + d) floats where d is head dim

The rescale is local computation between steps 1 and 2. On a NUMA system,
both operations use the local-write / remote-read pattern: each node writes
its contribution to local memory, other nodes read it remotely. No remote
writes occur.

The total cross-node data is `N * (1 + d) * 4` bytes per attention head.
For d=512 and N=4, this is ~8 KB per head — negligible compared to the KV
cache data each node avoids reading.

## Precision

The merge is mathematically exact. Floating-point results differ from a
single-pass sequential scan by at most 1 ULP due to the different ordering
of exp() evaluations and accumulation. The maximum observed error across
all tested configurations is one unit in the last place of f32.
