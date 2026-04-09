# Gemma 4 26B-A4B ButterQuant — Forward Pass Specification

Model: `google/gemma-4-26B-A4B`. Int8 quantized via ButterQuant.
Decode only (seq_len=1). Prefill follows the same structure with
multi-row dispatch.

---

## 1. Memory Model

### 1.1 Allocation

All memory is statically allocated at model load time. No runtime
allocations. Kernels use only stack `InlineArray` for temporaries.

Per rank: one `NumaArena` on the rank's NUMA node containing weights,
state, and scratch. One `ScratchPool` manages offsets across all ranks
— a single borrow reserves the same relative offset in every rank's
arena. Pointers materialized per-rank via `rv.scratch_base() + lease.offset`.

### 1.2 Distribution

| Category | Strategy | Contents |
|----------|----------|----------|
| Replicated | Full copy per rank | x_main, x_residual, scratch, KV caches, dense MLP weights, attention weights, router weights, norms, layer scalars, embedding |
| Distributed | Round-robin by expert ID | Expert weights: expert `e` on rank `e % tp`, local index `e // tp` |
| Sharded | Split by dimension | Attention Q/K/V (RowShard), O (ColShard) |

### 1.3 Cross-Node Communication

Exclusively `ring_allreduce`. Two per layer:
1. After attention O projection (ColShard partial sums)
2. After MoE local expert accumulation (partial expert sums)

No other cross-node reads or writes. Body threads access only their
rank's arena via `RankView`.

---

## 2. Pools

Three `BurstPool` instances per rank, partitioning the NUMA node's cores:

| Pool | Cores | Usage |
|------|-------|-------|
| main_pool | all | Sequential phases: attention, norms, activation quantize, router, pre-reduce, post-reduce |
| expert_pool | first half | Routed expert FFN kernels during overlap |
| dense_pool | second half | Dense MLP during overlap |

Main pool and expert+dense pools are mutually exclusive in time.

`ranks.parallel[body](pool_ptrs)` dispatches a body closure to all
ranks. `pool_ptrs` is the pool pointer array for the target pool,
passed at the call site. `Ranks` holds only arena bases.

---

## 3. Weight Layout

### 3.1 Quantized Projections

Each quantized weight produces three buffers:

| Buffer | Type | Shape | Notes |
|--------|------|-------|-------|
| weight | i8 VNNI-packed | N × K bytes | 6D tiled layout for vpdpbusd |
| weight_scale | f32 | N | Per-row absmax / 127 |
| weight_colsum | f32 | N (or N × num_blocks) | Per-row (or per-block) sum of i8 values |

RMSNorm gamma absorbed into downstream weights offline. Per-head Q/K
norm gamma absorbed into projection row blocks offline. Router scale ×
inv_sqrt_hidden absorbed into router projection offline.

DC correction: down projection weight columns at positions
[0, 64, 128, ...] pre-scaled by 1/DC_SCALE = 2.0 offline.

### 3.2 Per-Layer Weight Table

| Weight | Shape | Sharding | Block | Gamma Source |
|--------|-------|----------|-------|-------------|
| **Attention** |||||
| QKV (fused, sliding) | (8192, 2816) | RowShard | 256 | input_layernorm + per-head q/k norm |
| QK (fused, full) | (9216, 2816) | RowShard | 256 | input_layernorm + per-head q/k norm |
| O projection (sliding) | (2816, 4096) | ColShard | 256 | none |
| O projection (full) | (2816, 8192) | ColShard | 256 | none |
| **Dense MLP** |||||
| gate+up (fused) | (4224, 2816) | Replicated | 256 | pre_feedforward_layernorm |
| down | (2816, 2112) | Replicated | 64 | none (DC-corrected columns) |
| **Router** |||||
| router.proj | (128, 2816) | Replicated | 256 | router.scale × inv_sqrt_hidden |
| per_expert_scale | (128,) bf16 | Replicated | — | — |
| **Expert** (×128, distributed) |||||
| gate_up (fused) | (1408, 2816) | Replicated | 256 | pre_feedforward_layernorm_2 |
| down | (2816, 704) | Replicated | 64 | none (DC-corrected columns) |
| **Norms** (bf16, runtime) |||||
| post_attention_layernorm | (2816,) | Replicated | — | not absorbed |
| post_feedforward_layernorm_1 | (2816,) | Replicated | — | not absorbed |
| post_feedforward_layernorm_2 | (2816,) | Replicated | — | not absorbed |
| post_feedforward_layernorm | (2816,) | Replicated | — | not absorbed |
| layer_scalar | (1,) bf16 | Replicated | — | — |

### 3.3 Expert Weight Addressing

Expert `e` lives on rank `e % tp` at local index `e // tp`.
Address: `rv.expert_weight_base(layer) + (e // tp) * EXPERT_STRIDE + field_offset`.

All expert fields (gate_up, gate_up_scale, gate_up_colsum, down,
down_scale, down_block_colsum) at fixed offsets within the stride.

### 3.4 Down Projection Block Colsums

Dense down: f32[2816, 33] — 33 per-block colsums per output row (K=2112, block=64).
Expert down: f32[2816, 11] — 11 per-block colsums per output row (K=704, block=64).
Computed at load time from packed i8 weights. Used for per-block VNNI bias correction.

---

## 4. Scratch Layout

One `ScratchPool`, LIFO borrow/release. Offsets identical across ranks.

### 4.1 Forward-Lifetime Borrows

| Lease | Type | Count | Lifetime |
|-------|------|-------|----------|
| act_scale | f32 | 1 | Entire forward |
| post_blk_scale | f32 | DENSE_NUM_BLOCKS (33) | Entire forward |

### 4.2 Per-Layer Borrows

Borrowed and released within the layer loop, LIFO order.

| Lease | Type | Count | Alive During |
|-------|------|-------|-------------|
| act_i8 | i8 | HIDDEN (2816) | Steps 1–8 |
| act_work | f32 | HIDDEN (2816) | Step 1 only |
| router_logits | bf16 | NUM_EXPERTS (128) | Step 2 only |
| routing | Gemma4TopKResult[8] | 1 | Steps 2–8 |
| expert_out | bf16 | TOP_K × HIDDEN (22528) | Steps 3–8 |
| local_count | i32 | 1 | Steps 3–8 |
| dense_post_i8 | i8 | INTERMEDIATE (2112) | Steps 4–7 |
| dense_out | bf16 | HIDDEN (2816) | Steps 6–8 |
| dense_normed | bf16 | HIDDEN (2816) | Steps 8–10 |

Peak: all borrows from act_i8 through dense_normed alive simultaneously
during the overlap phase. ~56KB per rank.

---

## 5. Forward Pass — Per Layer

Input: x_main bf16[2816], identical on all ranks.
Output: x_main bf16[2816] updated, identical on all ranks.

### Attention Block (stubbed)

Steps use main_pool. Full specification deferred to attention kernel
implementation. The operations are:

| Step | Input | Output | Kernel |
|------|-------|--------|--------|
| Attention norm | x_main | act_i8[2816] + f32 scale | rmsnorm_fwht_quantize[2816, 256] |
| QKV GEMV | act_i8 | bf16[QKV_N] | int8_gemv[QKV_N, 2816] |
| Per-head norm + RoPE + FWHT + cache write | QKV bf16 | KV cache (VNNI i8) + Q i8 | write_k_head_normed, write_v_head_normed, prep_q_row_normed |
| Attention scoring + V-agg | Q i8, K/V cache | attn_out i8 | AMX decode (tdpbsud scoring, tdpbusd V-agg) |
| O GEMV | attn_out i8 | x_residual bf16[2816] partial | int8_gemv[2816, Q_DIM] |
| Allreduce | x_residual | x_residual (full) | ring_allreduce (ColShard O) |
| Post-attention norm + residual | x_residual, x_main | x_main | dispatched: rmsnorm(x_residual, POST_ATTN_NORM.γ); x_main += result |

After the attention block, x_main contains the residual + normed
attention output, identical on all ranks.

### FFN Block

Ten steps. Each dispatched to a pool via `ranks.parallel[body](pool_ptrs)`.
Body threads build args from rank-local data, dispatch, return.

#### Step 1: Activation Quantize [main_pool]

| | |
|---|---|
| Input | x_main bf16[2816] |
| Output | act_i8 i8[2816] at scratch(act_i8_lease), act_scale f32 at scratch(act_scale_lease) |
| Kernel | rmsnorm_fwht_quantize[cols=2816, block=256] |
| Operation | x_main / rms(x_main) → block-diagonal FWHT(256) → dynamic absmax → i8 |

Gamma from input_layernorm, pre_feedforward_layernorm, pre_feedforward_layernorm_2,
and router.scale × inv_sqrt_hidden are all absorbed into their respective
downstream weights. The runtime activation is `x / rms(x)` for all four
consumers. One quantization, one i8 result, shared by steps 2, 3, 4.

#### Step 2: Router [main_pool]

Two dispatches:

**Router GEMV:**

| | |
|---|---|
| Input | act_i8 i8[2816], act_scale f32, router weight i8 VNNI[128, 2816] |
| Output | router_logits bf16[128] at scratch(router_logits_lease) |
| Kernel | int8_gemv[N=128, K=2816] |

**Softmax + Top-K:**

| | |
|---|---|
| Input | router_logits bf16[128], per_expert_scale bf16[128] |
| Output | Gemma4TopKResult[8] at scratch(routing_lease) |
| Kernel | router_topk_kernel: softmax → greedy top-8 → renormalize → per-expert scale |

Routing result is identical on all ranks (Replicated router weights,
replicated input). Each rank reads its own copy from local scratch.

#### Step 3: Expert Dispatch [expert_pool]

| | |
|---|---|
| Input | act_i8, act_scale, routing result, expert weights (local to rank) |
| Output | expert_out bf16[local_count × 2816] at scratch(expert_out_lease), local_count i32 at scratch(local_count_lease) |
| Kernel | gemma4_expert_i8_kernel[intermediate=704, hidden=2816, fwht_blk=64] × local_count |

Body reads routing from local scratch. Filters `eid % tp == rank`.
Builds one job arg per local expert. Dispatches `local_count` jobs to
expert_pool. Returns `PoolFence.completed()` — workers start, body returns.

Each expert kernel (one BurstPool job, one worker):

| Phase | Operation | Domain |
|-------|-----------|--------|
| 1 | gate_up int8 GEMV: act_i8[2816] × W[1408, 2816] → f32[1408] stack | Original |
| 2 | split gate[704] / up[704], gelu_tanh(gate) × up → f32[704] stack | Original |
| 3 | FWHT(block=64) + DC_SCALE(0.5) on element 0 + per-block absmax → i8[704] stack, f32[11] block scales stack | Hadamard |
| 4 | down int8 GEMV blocked: i8[704] × W[2816, 704] → f32[2816] stack. Per-block dequant aligned with K_STEP=64. | Original |
| 5 | × routing_weight → bf16[2816] written to expert_out slot | — |

Stack per worker: ~30KB (gate_up f32 + qi i8 + down f32 + block scales).

Expert weight addressing: `rv.expert_weight_base(layer) + (eid // tp) * EXPERT_STRIDE`.

Writes local_count to local scratch for pre_reduce to read.

#### Step 4: Dense Phase 1 [dense_pool]

| | |
|---|---|
| Input | act_i8 i8[2816], act_scale f32, fused gate+up weight i8 VNNI[4224, 2816] |
| Output | dense_post_i8 i8[2112] at scratch(dense_post_i8_lease), per-block scales f32[33] at scratch(post_blk_scale_lease) |
| Kernel | fused_gu_gelu_tanh[cols=2112, K=2816, fwht_blk=64] (multi-worker) |

Fused kernel per worker (workers partition activation rows for prefill,
single row for decode):

| Phase | Operation |
|-------|-----------|
| 1 | gate+up int8 GEMV: i8[2816] × W[4224, 2816] → f32[4224] stack |
| 2 | split gate[2112] / up[2112], gelu_tanh(gate) × up → f32[2112] |
| 3 | FWHT(block=64, 33 blocks) + DC_SCALE(0.5) on element 0 per block |
| 4 | per-block absmax → i8[2112] + f32[33] block scales |

Expert pool workers run concurrently during this dispatch.

#### Step 5: Dense Phase 1 Join [dense_pool]

Body calls `pool.join()`. Returns `PoolFence.completed()`.
Expert pool workers may still be running.

#### Step 6: Dense Phase 2 [dense_pool]

| | |
|---|---|
| Input | dense_post_i8 i8[2112], per-block scales f32[33], down weight i8 VNNI[2816, 2112] + block colsums f32[2816, 33] |
| Output | dense_out bf16[2816] at scratch(dense_out_lease) |
| Kernel | int8_gemv_blocked[N=2816, K=2112, block=64] (multi-worker) |

Per-block dequant in the GEMV K-iteration:

```
for each K-block b (0..32):
    i32_partial = VNNI_dot(act_i8[b*64:(b+1)*64], weight[n, b*64:(b+1)*64])
    f32_acc += (i32_partial - 128 * block_colsum[n, b]) * (block_scale[b] / 127)
y[n] = f32_acc * weight_scale[n]
```

Expert pool workers still running concurrently.

#### Step 7a: Dense Phase 2 Join [dense_pool]

Body calls `pool.join()`. Dense path complete.

#### Step 7b: Expert Join [expert_pool]

Body calls `pool.join()`. Expert path complete.

All overlap work finished. Main pool now available.

#### Step 8: Pre-Reduce [main_pool]

| | |
|---|---|
| Input | expert_out bf16[local_count × 2816], local_count i32, dense_out bf16[2816], POST_FFN_NORM_1.γ bf16[2816] |
| Output | x_residual bf16[2816] (partial expert sum), dense_normed bf16[2816] |
| Kernel | pre_reduce_kernel (single dispatch) |

Two operations in one kernel:

1. **Expert accumulate**: sum `local_count` expert output slots → x_residual.
   ```
   x_residual[i] = Σ_{e=0}^{local_count-1} expert_out[e * HIDDEN + i]
   ```
   Routing weights already baked into expert outputs.

2. **Dense norm**: rmsnorm(dense_out, POST_FFN_NORM_1.γ, eps) → dense_normed.
   Dense output is identical on all ranks (Replicated weights).
   dense_normed is written to local scratch for step 10.

#### Step 9: Allreduce [cross-rank]

| | |
|---|---|
| Input | x_residual bf16[2816] per rank (partial expert sums) |
| Output | x_residual bf16[2816] per rank (full expert sum, identical on all ranks) |
| Kernel | ring_allreduce[X_RESIDUAL, tp] |

After this, x_residual = Σ (all 8 routing-weighted expert outputs).

#### Step 10: Post-Reduce [main_pool]

| | |
|---|---|
| Input | x_residual bf16[2816] (allreduced), dense_normed bf16[2816], POST_FFN_NORM_2.γ, POST_FFN_NORM.γ, layer_scalar, x_main |
| Output | x_main bf16[2816] (updated for next layer) |
| Kernel | post_reduce_kernel wrapping moe_combine[2816] |

Sequential operations on 2816 elements:

```
moe_normed = x_residual / rms(x_residual) * POST_FFN_NORM_2.γ
combined = moe_normed + dense_normed
combined_normed = combined / rms(combined) * POST_FFN_NORM.γ
x_main = (x_main + combined_normed) * layer_scalar
```

x_main is now ready for the next layer.

---

## 6. Final Layers

After 30 layer iterations:

| Step | Input | Output | Pool |
|------|-------|--------|------|
| Final norm | x_main[last_token] | normed bf16[2816] | main_pool (host rank) |
| LM head | normed × embed_weight^T | logits bf16[262144] | main_pool (host rank, bf16 GEMV) |
| Logit softcap | logits | tanh(logits/30) × 30 | main_pool (host rank) |

Embedding weight is tied (lm_head reuses embed_tokens.weight).
LM head uses bf16 float_gemv, not int8.
Logits written to scratch, returned via `LogitsView` which holds the lease.

---

## 7. Quantization Details

### 7.1 Activation Quantize (rmsnorm_fwht_quantize)

```
Input:  bf16[cols]
Step 1: bf16 → f32, compute rms, divide by rms (gamma absorbed into weights)
Step 2: block-diagonal FWHT (block=256 for HIDDEN, block=64 for INTERMEDIATE)
Step 3: compute per-row absmax, quantize to i8 with dynamic scale
Output: i8[cols], f32 scale
```

One scale per row. Used for pre-attention, pre-FFN activations, and router input.

### 7.2 Post-Nonlinearity Quantize (fused_gu_gelu_tanh)

```
Input:  i8 activation[K], f32 act_scale, VNNI gate+up weight[2*intermediate, K]
Step 1: int8 GEMV → f32 gate[intermediate] + f32 up[intermediate]
Step 2: gelu_tanh(gate) × up → f32[intermediate]
Step 3: FWHT (block=64), DC correction (element 0 × 0.5), per-block absmax → i8
Output: i8[intermediate], f32[intermediate/64] block scales
```

Per-block scales (one per FWHT block of 64 elements). DC correction reduces
NRMSE from 1.16% to 0.49%. Weight columns at DC positions (0, 64, 128, ...)
pre-scaled by 2.0 at offline quantize time — cancels in GEMV.

### 7.3 Blocked Down GEMV (int8_gemv_blocked)

```
Input:  i8 activation[K], f32[K/64] block scales, VNNI weight[N, K], f32[N] weight scales, f32[N, K/64] block colsums
Step 1: for each K-block b:
          i32_partial = VNNI_dot(act[b*64:(b+1)*64], weight[n, b*64:(b+1)*64])
          f32_acc += (i32_partial - 128 * block_colsum[n, b]) * (block_scale[b] / 127)
Step 2: y[n] = f32_acc * weight_scale[n]
Output: f32[N] (or bf16[N])
```

K_STEP = FWHT block = VNNI_K_STEP = 64. Per-block dequant aligned with
the VNNI K-iteration. i32 accumulation within each block, f32
accumulation across blocks.

### 7.4 Per-Head Norm + RoPE + FWHT Pipeline

For Q (prep_q_row_normed):
```
bf16[head_dim] → f32 registers → /rms (gamma absorbed) → RoPE → FWHT → dynamic absmax → i8
Returns: i8[head_dim], qi_bias (128 × Σqi), q_scale (absmax)
```

For K (write_k_head_normed):
```
bf16[head_dim] → f32 → /rms (gamma absorbed) → RoPE → FWHT → dynamic absmax → i8 → XOR 0x80 → u8 VNNI cache
```

For V (write_v_head_normed):
```
bf16[head_dim] → f32 → /rms (no gamma) → FWHT → fixed scale quantize → i8 VNNI cache
```

Sliding layers: full RoPE (theta=10000, head_dim=256, 128-entry cos/sin table).
Full layers: partial RoPE (theta=1000000, 128 of 512 dims rotated, 64-entry compact table).

Entire pipeline stays in registers for Q. K and V use a caller-provided
f32[head_dim] work buffer.

### 7.5 Attention Scoring

```
Score dequant: (raw - b_q) × S_Q[h] / 127² × S_K[g, t]
```

Scale = 1.0 (QK-norms absorb 1/√d_k). No inv_sqrt_hd term.

K stored as u8 (XOR 0x80) in VNNI tile format for tdpbsud (signed Q × unsigned K).
V stored as i8 in VNNI tile format for tdpbusd (unsigned W × signed V).
Both pre-packed at cache write time. Zero transformation during scoring scan.

### 7.6 KV Cache (Gemma4KVCache)

K: VNNI tile format `[head][tile_idx][k_slice × TILE_BYTES]`. Scatter-write at
cache write time. Direct tileload at scoring time.

V: VNNI tile format `[head][hd_tile][k_group × TILE_BYTES]`. Scatter-write at
cache write time (transposed from K — positions along K-dimension for V-agg).
Direct tileload at V-agg time. No runtime pack_v_tile_vnni.

Scales: f32 K scales `[num_kv_heads][max_seq]`, f32 Q scales `[num_q_heads][max_seq]`.
V uses a fixed per-layer scale derived from weight Frobenius norms.

---

## 8. Sync Points Per Layer

| # | Point | Type | Between |
|---|-------|------|---------|
| 1–N | Attention phase pool joins | Local | main_pool dispatches within attention block |
| N+1 | Attention O allreduce | Cross-rank | ColShard O partial sums |
| N+2 | Activation quantize join | Local | main_pool |
| N+3 | Router GEMV join | Local | main_pool |
| N+4 | Router top-k join | Local | main_pool |
| N+5 | Dense phase 1 join | Local | dense_pool |
| N+6 | Dense phase 2 join | Local | dense_pool |
| N+7 | Expert join | Local | expert_pool |
| N+8 | Pre-reduce join | Local | main_pool |
| N+9 | MoE allreduce | Cross-rank | Expert partial sums |
| N+10 | Post-reduce join | Local | main_pool |

Two cross-rank allreduces. All other sync is local pool joins.
Dense phase 1 join (N+5) and dense phase 2 dispatch overlap with
expert_pool execution.

---

## 9. Overlap Structure

```
time →

expert_pool:  [====== expert kernels (local_count jobs) ======]
dense_pool:   [== gate_up+gelu ==][join][== down ==][join]
                                                          [expert join]
main_pool:                                                 [pre_reduce]
allreduce:                                                             [allreduce]
main_pool:                                                                        [post_reduce]
```

Expert kernels and dense phases run concurrently on separate core partitions.
Dense has one internal sync (between fused gate_up+nonlinearity and down GEMV).
Expert join waits for all local experts after both dense phases complete.
The fused gate_up+nonlinearity phase is lighter than the expert pipeline,
so the dense join does not block the critical path.
