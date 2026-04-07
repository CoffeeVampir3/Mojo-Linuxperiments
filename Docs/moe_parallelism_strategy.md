# MoE Parallelism Strategy

## Expert Placement

Routed experts are distributed round-robin across NUMA nodes: expert `e` lives on
node `e % tp`. With 64 routed experts and TP=4, each node owns 16 experts.

Shared expert weights are RowShard (gate, up) / ColShard (down), identical to
dense FFN sharding. Each rank holds its shard.

## MoE Block Execution (decode)

Input: `x_residual` (post-norm hidden state), replicated and in-sync across ranks.

### Step 1: Router (host, inline)

Small GEMV `[64, 2048] @ hidden -> [64]` logits, then `softmax_topk[64, 6]`
yields 6 `(expert_id, gate_value)` pairs. Negligible cost, no pool dispatch needed.

### Step 2: Parallel dispatch (all ranks)

Two workloads dispatched concurrently to each rank's pool:

- **Routed experts** (1 thread per expert): Each selected expert where
  `expert_id % tp == rank` runs as a BurstPool job. Kernel executes a complete
  SwiGLU FFN and overwrites a per-expert output buffer with the gate-scaled
  `[hidden]` result. Expert intermediates are stack-local (`InlineArray` on
  worker stacks). Per-expert output buffers: `top_k * hidden * 2` bytes from
  scratch (24KB for V2-Lite, 112KB for V3). Workers that have no assigned
  expert are not dispatched.

- **Shared expert gate+up** (all ranks, sharded): `gemm(gate_proj)` and
  `gemm(up_proj)` dispatched as standard RowShard GEMVs into scratch
  intermediates.

### Step 3: Join

### Step 4: Fused dispatch (all ranks)

Single job per rank combining:
1. `silu_mul` on shared expert gate/up intermediates
2. Shared expert `gemm(down_proj)` (ColShard) -> write to `x_residual`
3. SIMD sum of per-expert routed output buffers -> add to `x_residual`

All operations are NUMA-local: shared intermediates, routed output buffers, and
`x_residual` all reside on the rank's NUMA node. No extra dispatch phase needed
for the routed buffer sum — it piggybacks on the shared expert's down projection.

### Step 5: Join

### Step 6: Allreduce

`ring_allreduce(x_residual)` across all ranks. One reduction collects:
- Shared expert ColShard partial sums (every rank contributes)
- Routed expert local sums (sparse: only ranks with selected experts contribute
  nonzero, others contribute only shared)

### Step 7: Residual

`elem_add(x_main, x_residual) -> x_main`

## Why One Allreduce Covers Both

Allreduce is a sum. Each rank's `x_residual` contains:
- Its shared expert ColShard partial (from step 4.2)
- Its local routed expert sum (from step 4.3)

Summing across ranks yields the complete MoE output. Ranks with no routed
experts contribute only their shared expert shard.

## Memory

**Per-expert output buffers**: `top_k * hidden * sizeof(bf16)` from scratch,
uniform borrow across all ranks. Worst case (all experts on one node) is the
same as the uniform allocation. V2-Lite: 24KB. V3: 112KB. Negligible.

**Shared expert intermediates**: `seq_len * (SHARED_INTERMEDIATE / tp) * 2 * 2`
from scratch. For decode: negligible.

**Expert FFN intermediates**: stack-local in the worker kernel. No scratch.

## Expert FFN Kernel

Each routed expert runs as a BurstPool job via typed struct dispatch:

```
@fieldwise_init
struct ExpertFFNArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var gate_up: BF16Ptr
    var down: BF16Ptr
    var output: BF16Ptr       # per-expert buffer (overwrite)
    var gate_val: Float32
    var intermediate: Int
    var hidden: Int
```

Kernel runs a complete SwiGLU FFN, writes gate-scaled `[hidden]` to its own
output buffer. No accumulation in the kernel — the post-join fused step sums
the buffers.

## Fused Down + Sum Kernel

Per-rank kernel combining shared expert completion with routed buffer collection:

```
@fieldwise_init
struct MoECombineArgs(Copyable, ImplicitlyCopyable):
    var shared_gate: BF16Ptr    # shared gate intermediate
    var shared_up: BF16Ptr      # shared up intermediate
    var shared_down_w: BF16Ptr  # shared down_proj weight (ColShard)
    var expert_bufs: BF16Ptr    # base of per-expert output buffers
    var dst: BF16Ptr            # x_residual
    var num_experts: Int         # how many routed expert buffers to sum
    var hidden: Int
    var shared_intermediate: Int
```

## Dispatch Phases: 2

The MoE block uses exactly two dispatch-join cycles:
1. Routed expert FFNs + shared gate/up (parallel compute)
2. Fused silu_mul + shared down + routed sum (combine + write)

Followed by one allreduce and one residual add.
