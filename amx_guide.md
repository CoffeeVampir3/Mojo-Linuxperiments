# AMX Integration Guide for Hadamard-Quantized Attention

## Overview

This document covers the strategy for integrating Intel AMX (Advanced Matrix Extensions)
into the Hadamard-quantized int8 inference pipeline. The target is Sapphire Rapids / Granite
Rapids server CPUs with AMX-INT8 support. The current codebase uses AVX-512 VNNI
(`vpdpbusd`) for all int8 dot products; AMX replaces these with tile-level matrix operations
that deliver ~256x more multiply-accumulates per instruction.

---

## AMX Tile Registers and Configuration

AMX provides 8 tile registers (TMM0-TMM7). Each tile is up to 16 rows x 64 bytes.
For int8 GEMM with VNNI packing (4 bytes per i32 lane):

| Tile Use | Rows | Cols (bytes) | Interpretation |
|----------|------|-------------|----------------|
| A (i8/u8 activations) | 16 | 64 | 16 rows x 64 elements |
| B (i8/u8 weights, VNNI-packed) | 16 | 64 | (K/4) rows x (N*4) cols |
| C (i32 accumulators) | 16 | 64 | 16 rows x 16 i32 elements |

The tile configuration is set via `_tile_loadconfig` (or `ldtilecfg`) which writes a 64-byte
config block specifying rows/cols for each tile. **This instruction is expensive** (~100s of
cycles, pipeline drain) and should be called as rarely as possible.

### The 2-2-4 Configuration (Recommended Default)

```
TMM0, TMM1: A tiles  (16x64 each)  -> M_STEP = 32
TMM2, TMM3: B tiles  (16x64 each)  -> N_STEP = 32
TMM4-TMM7:  C tiles  (16x64 each)  -> 2x2 output grid = [32, 32] i32
```

One tile multiply: `_tile_dpbusd(C, A, B)` computes:
```
C[i,j] += sum_{k=0..3} A_u8[i, 4j+k] * B_i8[4j+k, col_j]
```
producing a 16x16 i32 result from 16x64 u8 and 16x64 i8 inputs.

The 2-2-4 inner loop processes a 32x32 output block:
```
_tile_dpbusd(4, 0, 2)  // C[0:16, 0:16]  += A[0:16, :] * B[:, 0:16]
_tile_dpbusd(5, 0, 3)  // C[0:16, 16:32] += A[0:16, :] * B[:, 16:32]
_tile_dpbusd(6, 1, 2)  // C[16:32, 0:16] += A[16:32,:] * B[:, 0:16]
_tile_dpbusd(7, 1, 3)  // C[16:32,16:32] += A[16:32,:] * B[:, 16:32]
```

4 tile multiplies = 4 * 16,384 = 65,536 MACs from loading 2 A tiles + 2 B tiles (512 bytes total).

### Why 2-2-4 for Everything

The same tile config serves all operations in the model:

| Operation | M | K | N | Notes |
|-----------|---|---|---|-------|
| Linear projections (Q,K,V,O,gate,up,down) | seq_len | hidden | proj_dim | Standard GEMM |
| Attention scoring (Q*K^T) | gqa_factor | head_dim | context_block | Per KV group |
| V aggregation (P*V) | gqa_factor | context_block | head_dim | Per KV group |
| RMSNorm+FWHT+Quant | seq_len | hidden | - | Not a GEMM |

For decode (seq_len=1, gqa_factor=16), M < M_STEP=32 so only the first A tile is populated.
This wastes half the A capacity but avoids reconfiguration. The alternative 1-3-3 layout
(M_STEP=48, N_STEP=16) gives better B-tile reuse for small M but requires a separate
tile config.

**Recommended strategy:** Configure 2-2-4 once per worker thread at startup. Accept the
minor decode inefficiency. If profiling shows decode attention is the bottleneck, add a
second config (1-3-3) that is set at the decode/prefill boundary -- at most 2 config
calls per forward pass, not per layer.

### Tile Config in Practice

```
# Pseudocode for 2-2-4 int8 config
tile_config = TileConfig()
for i in range(2):     # A tiles
    tile_config.set(i, rows=16, cols=64)
for i in range(2, 4):  # B tiles (VNNI packed)
    tile_config.set(i, rows=16, cols=64)
for i in range(4, 8):  # C tiles (i32)
    tile_config.set(i, rows=16, cols=64)  # 16 rows x 16 i32 = 64 bytes
tile_config.apply()  # ldtilecfg -- expensive, do once
```

After this, all `_tile_loadd`, `_tile_stored`, `_tile_dpbusd`, `_tile_zero` calls
use the configured dimensions implicitly.

---

## Data Layout Requirements

### VNNI Packing for B Tiles

AMX int8 multiply (`_tile_dpbusd`) uses the same VNNI packing as `vpdpbusd`:
4 consecutive K-dimension bytes are grouped per i32 lane. For a B matrix of shape
[K, N], the packed layout interleaves K values in groups of 4:

```
Original B[k, n]:  row-major, K rows x N cols
Packed B_vnni:     (K/4) rows x (N*4) cols (byte-level)

For tile j, lane i:
  B_vnni[j, 4*i+0..3] = B[4*j+0..3, i]   (4 consecutive K values for output col i)
```

This is the same transpose-and-interleave used by the existing `pack_vnni` in
`kernels/vnni.mojo`. The packing is done once when weights are loaded (for projections)
or per-block during attention (for the quantized attention weights P).

### A Matrix Layout

A tiles are loaded in row-major order with stride = K (in bytes). No special packing
needed. `_tile_loadd(tmm, ptr, stride)` loads 16 rows of 64 bytes each.

### C Matrix Layout

C tiles are row-major i32 with stride = N * sizeof(i32). Loaded/stored via
`_tile_loadd`/`_tile_stored` with byte stride.

---

## Mapping to Attention Operations

### Current Architecture (VNNI)

The kernel processes KV in blocks of B_c=64 timesteps with online softmax:

```
for each KV block (B_c timesteps):
    1. Score:    for each t in block: i8_dot_row_major(K[t], Q[hi]) -> scores
    2. Softmax:  online update (f32 SIMD)
    3. Quantize: absorbed weights -> w_i8
    4. V agg:    for each d: i8_dot_row_major(V_transposed[d], w_i8) -> output
```

Steps 1 and 4 are the VNNI hot paths. Step 2 remains f32 SIMD regardless.

### AMX Attention (Proposed)

Both scoring and V agg become tile-level GEMMs within the blocked loop:

**Scoring (per block):**
```
# Q_i8:  [gqa_factor, head_dim]    = [16, 128]   (A operand)
# K_u8:  [B_c, head_dim]           = [64, 128]   (B operand, needs VNNI pack)
# S_i32: [gqa_factor, B_c]         = [16, 64]    (C output)

# With 2-2-4: M_STEP=32, K_STEP=64, N_STEP=32
# M=16 < M_STEP: one M-tile iteration
# K=128: 2 K-step iterations
# N=64: 2 N-step iterations
# Total: 2 * 2 * 4 = 16 tile_dp calls
```

**V Aggregation (per block):**
```
# W_i8:  [gqa_factor, B_c]         = [16, 64]    (A operand, quantized weights)
# V_u8:  [B_c, head_dim]           = [64, 128]   (B operand, needs VNNI pack)
# O_i32: [gqa_factor, head_dim]    = [16, 128]   (C output)

# M=16 < M_STEP: one M-tile iteration
# K=64: 1 K-step iteration (B_c fits in one K_STEP!)
# N=128: 4 N-step iterations
# Total: 1 * 4 * 4 = 16 tile_dp calls
```

Compare to current VNNI: 128 `i8_dot_row_major` calls per head x 16 heads = 2048 calls,
each with a 16-lane reduce. AMX does 16 tile_dp calls total, no reduces.

### Online Softmax Between Tile Operations

The blocked online softmax (from INT-FlashAttention) sits between scoring and V agg:

```
for each block b:
    S_i32 = tile_gemm(Q_i8, K_u8_block)           # AMX scoring
    S_f32 = dequantize(S_i32, q_scales, k_scales)  # f32 conversion
    update running max, rescale accumulator          # f32 SIMD
    W_absorbed = exp(S_f32 - max) * v_scale          # f32 SIMD
    W_i8 = quantize(W_absorbed)                      # f32 -> i8
    O_i32 += tile_gemm(W_i8, V_u8_block)            # AMX V agg
```

The f32 softmax work is small relative to the GEMMs. The key design point:
the C accumulator for V agg persists across blocks (running O), while scoring
C is consumed immediately.

### Prefill vs Decode

**Decode (seq_len=1):** M=gqa_factor=16. Only half the A tile is used.
Each KV group processes independently. The bottleneck is memory bandwidth
(streaming KV cache), not compute.

**Prefill (seq_len>1):** M=seq_len*gqa_factor or seq_len (depending on
how rows are batched). Tiles are fully utilized. This is where AMX
throughput dominates. The causal mask means each row attends to a
different context length, which complicates tiling -- rows can be grouped
by similar context length, or the mask applied post-multiply.

---

## Quantization Strategy: Hadamard Rotation + Absmax i8

### The Invariant

The Walsh-Hadamard Transform (FWHT) is a fixed orthogonal rotation applied
block-diagonally (block_size = head_dim). By Parseval's theorem:

```
<H*x, H*y> = <x, y>
```

Inner products are preserved through rotation. This means:
1. Rotate activations and weights by H before quantization
2. Quantize to int8 (absmax per-tensor/per-token)
3. Compute dot products in the rotated domain
4. The result equals the unrotated dot product (up to quantization error)

No inverse rotation is needed between layers -- the O projection weight
is stored pre-rotated, so the output stays in the rotated domain through
the residual stream.

### Why Rotation Helps Quantization

From TurboQuant (Zandieh et al., 2025): after random orthogonal rotation,
each coordinate of a unit-norm vector follows a Beta distribution that
converges to Gaussian in high dimensions. Coordinates become nearly
independent, allowing per-coordinate scalar quantization to achieve
near-optimal distortion (within 2.7x of the Shannon lower bound).

FWHT achieves the same distributional spreading as random rotation but is
O(d log d) instead of O(d^2) and requires no random matrix storage.

At 8-bit quantization, the MSE distortion per coordinate is ~4e-5 for
unit-norm vectors. This is negligible -- uniform absmax quantization is
essentially optimal at this bit width. The Lloyd-Max vs uniform distinction
only matters at 2-4 bits.

### Per-Token vs Per-Tensor Scaling

The INT-FlashAttention paper uses per-token scales for Q and K (S_Q, S_K
vectors), and a single tensor-level scale for V (S_V scalar). In our
implementation:

- **Q:** per-head absmax after FWHT (q_scales[hi])
- **K cache:** per-position absmax, stored alongside u8 data
- **V cache:** per-position absmax, stored alongside u8 data
- **Attention weights (P):** per-head-per-block absmax after V-scale absorption

The bias correction for u8 storage (i8 XOR 0x80):
```
true_dot = raw_u8xi8_dot - 128 * sum(i8_operand)
```
is one scalar per head, computed once and applied after the tile GEMM.

For AMX tile operations, the bias correction applies to the full C tile:
```
C_i32[m, n] = tile_dpbusd(A_u8, B_i8)   # raw u8 x i8
# Bias correct: subtract 128 * sum(B_i8_row) from each C column
# Or: subtract 128 * sum(A_u8_col) from each C row
# Depends on which operand is u8
```

---

## KV Cache Layout

### K Cache: Row-Major [head][pos][dim]

K entries are accessed during scoring as `K[pos, :]` -- a full head_dim
vector per timestep. Row-major layout gives contiguous reads for each entry.
For AMX, B_c consecutive K entries form a `[B_c, head_dim]` matrix that
needs VNNI packing into the B tile format.

### V Cache: Transposed [head][dim][pos]

V entries are accessed during aggregation as a reduction over the position
dimension. The transposed layout makes `V[:, d]` contiguous for a fixed
dimension, enabling direct VNNI dot products over timesteps.

For AMX V agg, the B operand is `V[B_c, head_dim]`. With transposed storage,
this requires gathering B_c values from each of head_dim columns. The VNNI
packing step naturally handles this rearrangement -- it's a transpose anyway.

Write path: `write_head_transposed()` scatters each byte to its
`[dim][pos]` location, XOR'd to u8. This is slower than row-major write
but amortized over many reads (write once per token generation, read every
subsequent decode step).

---

## BurstPool Dispatch and Parallelism

### Current Architecture

Workers are dispatched via `BurstPool.dispatch(kernel_fn, arg_packs, num_jobs)`.
The kernel function signature is `def(Int, Int, Int, Int, Int, Int)` -- 6 integer
arguments passed through `ArgPack`. Complex state is passed by pointer to a
context struct (`HadAttnCtx`).

### Parallelism Strategy

**Decode attention:** Parallelize over KV groups. Each of the 8 KV groups
(for DeepSeek-V3) runs independently on a separate worker. Each worker has
its own scratch region (running_o, block_scores, qi_qs) and writes to a
disjoint slice of the shared row_f32 output buffer. Output quantization
runs on the caller after `pool.join()`.

**Prefill attention:** Additionally parallelize over sequence length blocks
or context blocks within each KV group. The online softmax structure allows
splitting the context dimension into chunks, with a merge step for the
running max/sum.

**Projections:** Parallelize over the N dimension (output columns) of the
GEMM. Each worker processes a contiguous N_BLOCK of output columns.

### Scratch Memory

Pre-allocated from `NumaArena`, sized at init time via
`attn_scratch_bytes[..., num_workers]()`. Layout:

```
[0]                              row_f32       (shared across workers)
[row_f32_size + 0*PER_WORKER]    worker 0 scratch
[row_f32_size + 1*PER_WORKER]    worker 1 scratch
...
```

Each worker's scratch holds: running_o (f32), block_scores (f32), qi_qs (i8).
Small buffers (q_scales, q_biases, running_m, running_l, w_block) live on the
worker's stack as InlineArray.

The `ScratchPool` / `ScratchLease` pattern in the model code manages scratch
lifetime with LIFO semantics. Leases hold byte offsets (not pointers), and
each rank materializes pointers by adding its arena base address.

---

## Performance Expectations

### VNNI Baseline (Current, Measured)

DeepSeek-V3-like config: 128 heads, 8 KV heads, head_dim=128, context=4096.

| Configuration | Wall Time |
|---------------|-----------|
| Original (main, single-thread) | ~5250 us |
| Blocked + VNNI + transposed V | ~3520 us |
| + Parallel over 8 KV groups | ~1170 us |

### AMX Projections (Theoretical)

Per-group V agg compute (AMX vs VNNI):
- **VNNI:** 128 dims * 64 blocks * 1 vpdpbusd + reduce = ~8192 ops, ~400 us/group
- **AMX:** 4 N-steps * 1 K-step * 4 tile_dp * 64 blocks = ~1024 tile_dp, ~10 us/group

The 40x compute reduction may not translate to 40x wall time improvement
because memory bandwidth (streaming V cache at 512KB per group) becomes
the dominant bottleneck. Expected wall time with AMX: limited by
memory bandwidth to roughly the time to stream 4MB of V data (8 groups),
which at ~50 GB/s single-socket is ~80 us. Practical estimate with
overheads: 100-300 us total.

---

## Mojo Implementation Notes

### Inline Assembly for AMX

Mojo can call LLVM intrinsics directly via `llvm_intrinsic`. The AMX tile
instructions map to:

```
# Tile load: _tile_loadd(tmm, ptr, stride)
llvm_intrinsic["llvm.x86.tileloadd64.internal", ...]

# Tile store: _tile_stored(tmm, ptr, stride)
llvm_intrinsic["llvm.x86.tilestored64.internal", ...]

# Tile zero: _tile_zero(tmm)
llvm_intrinsic["llvm.x86.tilezero.internal", ...]

# Tile multiply: _tile_dpbusd(C, A, B)
llvm_intrinsic["llvm.x86.tdpbusd.internal", ...]

# Tile config: _tile_loadconfig(ptr)
llvm_intrinsic["llvm.x86.ldtilecfg.internal", ...]
```

The exact intrinsic signatures need verification against the Mojo/LLVM
version in use. The `CompilationTarget` module may expose AMX feature
detection similar to `has_vnni()`.

### Comptime Parameters

The tile dimensions (16, 64, 32) and step sizes are all comptime constants.
Use `comptime` for all GEMM tiling parameters. The blocked attention loop
bounds (B_c, num_blocks) are runtime but the tile operations within each
block are fully comptime-unrolled.

### InlineArray for Tile Buffers

VNNI packing buffers for B tiles are small (64 bytes per tile, ~1KB for
a full N_STEP block) and fit on the stack:

```
var b_packed = InlineArray[UInt8, TILE_N * TILE_K](fill=UInt8(0))
```

The 64-byte alignment requirement for tile loads can be ensured via the
arena allocator (which aligns to 8 bytes by default -- may need
`alignment=64` for AMX tile buffers).

### Function Dispatch

The kernel function for BurstPool dispatch must be `def(Int, Int, ...) -> None`
with 6 Int parameters. Context is passed by pointer through arg0. The
AMX tile config should be set once in each worker's initialization, not
per-dispatch. If BurstPool worker threads persist across dispatches (which
they do), the config survives.

### Fallback Path

All AMX code paths should have a VNNI fallback gated on
`CompilationTarget.has_amx()` (or equivalent). The existing VNNI kernel
serves as the fallback for non-AMX hardware. The kernel selection is
comptime, producing zero overhead in the non-AMX binary.

---

## Key References

- **INT-FlashAttention** (Chen et al., 2024): Blocked online softmax with int8 GEMM
  for both Q*K^T and P*V. Token-level quantization. Algorithm 1 is the blueprint
  for fused attention.

- **TurboQuant** (Zandieh et al., 2025): Proves rotation + scalar quantization is
  within 2.7x of Shannon's distortion-rate bound. Validates FWHT + absmax i8 as
  near-optimal. Provides roadmap for sub-8-bit KV cache if needed.

- **Intel AMX Programming Reference**: Tile register semantics, `ldtilecfg`
  format, `tdpbusd`/`tdpbssd`/`tdpbsud`/`tdpbuud` instruction variants.
  Note: `tdpbusd` = unsigned A x signed B (matches our u8 cache x i8 weights).

- The C++ AMX kernels in the codebase (GemmKernel133, GemmKernel224) demonstrate
  tile allocation strategies and VNNI packing patterns that translate directly
  to Mojo.
