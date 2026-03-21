# Atomic Decomposition of the Model Loading Problem

## Thesis

Every tensor in a model checkpoint — regardless of encoding, quantization strategy, or sharding policy — decomposes into N independent **atoms**. Each atom is a `(shape, byte_width, shard_strategy, placement)` tuple. The loader processes every atom with a single, universal algorithm. No atom's loading depends on any other atom's properties. The loader is a **map** over a flat list of atoms.

All non-flat concerns (scale shape derivation, shard axis propagation, block divisibility constraints, consumer pairing) are resolved **upstream** at config/skeleton definition time. By the time atoms reach the loader, the problem is purely flat.

The **execution layer** sits downstream and has its own decomposition — three levels from always-composable contracts to never-composable fused kernels. But the loader boundary is absolute: atoms enter, buffers exit.

---

## 1. The Atom

An atom is the indivisible unit of the loading problem. It is described by **carrier traits** — traits that hold data:

| Trait | Properties | Role |
|-------|-----------|------|
| `Encoding` | `DTYPE: DType`, `ELEMENT_BYTES: Int` | What type of data |
| `Shaped` | `ROWS: Int`, `COLS: Int` | How big |

And **producer traits** — types that carry computation:

| Trait | Role | Concrete Types |
|-------|------|----------------|
| `ShardStrategy` | How to divide across TP | `RowShard`, `ColShard`, `Replicated` |
| `Placement` | Where to materialize | `HostNode`, `AllNodes`, `ShardedNodes` |

Plus two runtime values from the manifest:

| Property | Type | Meaning |
|----------|------|---------|
| `file_offset` | Int | Byte position in checkpoint |
| `label` | String | Tensor name for manifest match |

This is **everything** the loader needs. The loader has no knowledge of:
- Whether the atom is a weight, a scale, a bias, or a zero-point
- Which quantization strategy produced it
- What other atoms exist in the model
- What the forward pass will do with the loaded data
- Which atoms are logically paired (e.g. weight + its scale)
- What compute kernel will consume the loaded buffer
- What role traits tag the atom's executor-facing descriptor

The atom is the **curried form** of the loading problem — all parameters are bound, all questions are answered.

---

## 2. Carrier vs Producer Traits

The trait architecture separates two fundamentally different kinds of traits.

### Carrier Traits: Data Without Behavior

```mojo
trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int
```

Carrier traits are **lookup tables** in the type system. `BF16.ELEMENT_BYTES` is always `2`. There is no computation, no strategy, no decision — just data. They are atomic (irreducible) because `DTYPE` determines `ELEMENT_BYTES` with no independent variation.

### Producer Traits: Types That Carry Computation

```mojo
trait ShardStrategy:
    @staticmethod
    fn shard_rows(global_rows: Int, tp: Int) -> Int: ...
    @staticmethod
    fn shard_cols(global_cols: Int, tp: Int) -> Int: ...

struct RowShard(ShardStrategy):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: return r // tp
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: return c
```

Producer traits are **computation** in the type system. The strategy IS a type. It doesn't encode a decision as data to be interpreted later — it carries the computation directly. This replaces integer-encoded dispatch (`SHARD_AXIS: Int`) with type-level strategy selection.

### The Key Pattern: Comptime Members via Type Parameter Methods

The critical capability: **Mojo allows calling static methods on trait-bound type parameters in comptime member initialization.**

```mojo
struct Slot[E: Encoding, S: ShardStrategy, rows: Int, cols: Int, tp: Int = 1](
    Encoding, Shaped,
):
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
```

`Self.S.shard_rows(...)` is resolved at compile time. The strategy type parameter determines which formula computes `ROWS`. The Slot struct has one definition; different strategy types produce different results through the same mechanism.

### Composable Consumption via Trait Intersection

Free functions consume trait intersections without knowing concrete types:

```mojo
fn slot_bytes[T: Encoding & Shaped]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES
```

This function works for any type that satisfies both `Encoding` and `Shaped` — it doesn't know or care whether `ROWS` was derived by `RowShard`, `ColShard`, or `Replicated`. This is the payoff: carrier traits provide the data, producer traits derive it, and consumers see only the carrier interface.

---

## 3. DimStrategy: The True Atom of Sharding

Shard strategy itself decomposes. The true atom is the **per-dimension strategy**:

```mojo
trait DimStrategy:
    @staticmethod
    fn local(global_size: Int, tp: Int) -> Int: ...

struct Divide(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int: return d // tp

struct Keep(DimStrategy):
    @staticmethod
    fn local(d: Int, tp: Int) -> Int: return d
```

Two atoms. Every 2D shard strategy is a composition:

```mojo
struct Shard2D[D0: DimStrategy, D1: DimStrategy](ShardStrategy):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: return Self.D0.local(r, tp)
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: return Self.D1.local(c, tp)

comptime RowShard   = Shard2D[Divide, Keep]
comptime ColShard   = Shard2D[Keep, Divide]
comptime Replicated = Shard2D[Keep, Keep]
```

This is algebraic: 2 atoms compose into all useful strategies. `Shard2D` is a **producer trait that composes producer traits** — higher-order composition through the type system.

### Placement Is Orthogonal

`HostOnly` and `Replicated` produce identical shapes (both `Shard2D[Keep, Keep]`) but differ in **where** the buffer materializes. This was conflated in the integer encoding (`SHARD_AXIS = -1` vs `-2`). Separation is correct:

```mojo
trait Placement:
    @staticmethod
    fn on_node(node_idx: Int, host_idx: Int, tp: Int) -> Bool: ...

struct HostNode(Placement):
    @staticmethod
    fn on_node(node_idx: Int, host_idx: Int, tp: Int) -> Bool:
        return node_idx == host_idx

struct AllNodes(Placement):
    @staticmethod
    fn on_node(node_idx: Int, host_idx: Int, tp: Int) -> Bool:
        return True
```

Shape derivation (ShardStrategy) and materialization location (Placement) are independent axes. The Slot composes both without coupling.

---

## 4. The Universal Loader Algorithm

Given an atom and a tensor-parallelism degree `tp`, the loader executes:

```
local_rows = S.shard_rows(rows, tp)
local_cols = S.shard_cols(cols, tp)
byte_count = local_rows * local_cols * elem_bytes

if strategy is col-sharded:
    fragments = local_rows strided reads (one per row)
else:
    fragments = 1 contiguous read

arena_bytes = align_up(byte_count, alignment)
```

This algorithm uses **only** the atom's own properties plus `tp` and `alignment`. It contains:
- **Flat arithmetic** (multiplication, alignment rounding)
- **One bounded dispatch** (contiguous vs. strided — determined by strategy type, not a switch)

With producer traits, even the dispatch can be carried by the strategy type:

```mojo
trait FullShardStrategy:
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: ...
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: ...
    @staticmethod
    fn fragments_per_shard[T: Shaped]() -> Int: ...
    @staticmethod
    fn needs_allreduce() -> Bool: ...
```

Fragment counting and allreduce signaling are **definitional** — bound to strategy identity, not computed from external data.

---

## 5. Encoding Groups: Multi-Buffer Decomposition

Every quantization strategy decomposes into a fixed number of atoms at config definition time. The decomposition is performed by **encoding groups** in the skeleton, not the loader.

### 5.1 Decomposition Table

| Encoding | Atom Count | Atoms |
|----------|-----------|-------|
| BF16 (plain) | 1 | weight(rows, cols, 2) |
| Int8 channelwise | 2 | weight(rows, cols, 1) + scale(rows, 1, 4) |
| Int8 channelwise+bias | 3 | weight + scale(rows, 1, 4) + bias(rows, 1, 4) |
| Int8 block[B] | 2 | weight(rows, cols, 1) + scale(rows, cols/B, 4) |
| Q4 block[B] | 3 | packed(rows, cols/2, 1) + scale(rows, cols/B, 2) + zero(rows, cols/B, 2) |

### 5.2 Encoding Group Types

The skeleton uses typed groups that bridge loader and executor:

```mojo
struct DenseGroup[E, S, P, rows, cols, suffix, tp]:
    comptime Weight = Slot[Self.E, Self.S, Self.P, Self.rows, Self.cols, Self.suffix, Self.tp]
    comptime Loader = BF16Loader
    @staticmethod fn atom_count() -> Int: return 1

struct QuantGroupV2[WeightEnc, ScaleEnc, S, ScaleS, P, rows, cols, block_size, ...]:
    comptime Weight = Slot[Self.WeightEnc, Self.S, Self.P, Self.rows, Self.cols, ...]
    comptime SCALE_COLS = Self.cols // Self.block_size
    comptime Scale = Slot[Self.ScaleEnc, Self.ScaleS, Self.P, Self.rows, Self.SCALE_COLS, ...]
    comptime Loader = I8ChanLoader

    # Role-tagged descriptors for the executor
    comptime WeightOp = WeightDesc[Self.WeightEnc, Self.Weight.ROWS, Self.Weight.COLS]
    comptime ScaleOp = ScaleDesc[Self.ScaleEnc, Self.Scale.ROWS, Self.Scale.COLS, Self.block_size]

struct TripleGroup[WeightEnc, ScaleEnc, ZeroEnc, S, P, rows, cols, pack_ratio, block_size, ...]:
    comptime PACKED_COLS = Self.cols // Self.pack_ratio
    comptime Weight = Slot[Self.WeightEnc, Self.S, Self.P, Self.rows, Self.PACKED_COLS, ...]
    comptime SCALE_COLS = Self.cols // Self.block_size
    comptime Scale = Slot[Self.ScaleEnc, Self.S, Self.P, Self.rows, Self.SCALE_COLS, ...]
    comptime Zero = Slot[Self.ZeroEnc, Self.S, Self.P, Self.rows, Self.SCALE_COLS, ...]
```

Three views of the same data:
- **Loader view**: flat `Slot` atoms, each satisfying `Encoding & Shaped`
- **Executor view**: role-tagged descriptors via `WeightOperand & Encoding & Shaped`, `ScaleOperand & Encoding & Shaped`
- **Kernel dispatch**: actionable trait (`Loader` member) or monolithic kernel identity

### 5.3 Scale Strategy Separation

Key finding: **scale strategy must be a separate type parameter.** A col-sharded weight with channelwise quantization produces `scale_cols = 1`. Dividing 1 by `tp = 2` gives 0. The scale must use `Replicated` while the weight uses `ColShard`.

`QuantGroupV2` takes `ScaleS` as a separate type parameter, making this a type-level decision in the skeleton rather than a runtime edge case in the loader.

### 5.4 The Blocked Topology Unification

All scale topologies are instances of a single formula:

```
scale_cols = weight_cols / block_size
```

| Strategy | block_size | scale_cols (cols=576) | Scale Shape |
|----------|-----------|----------------------|-------------|
| Channelwise | cols (576) | 1 | (576, 1) |
| Block[64] | 64 | 9 | (576, 9) |
| Block[32] | 32 | 18 | (576, 18) |
| Per-element | 1 | 576 | (576, 576) |

**Channelwise quantization is not a separate concept.** It is blocked quantization with `block_size = cols`.

### 5.5 Concrete Examples (Q_PROJ, 576x576)

**Replicated (tp=1):**

| Strategy | Atoms | Total Bytes |
|----------|-------|-------------|
| BF16 | 1 | 663,552 |
| Int8 chan | 2 | 334,080 |
| Int8 chan+bias | 3 | 336,384 |
| Int8 blk[32] | 2 | 373,248 |
| Q4 blk[32] | 3 | 207,360 |

**Row-sharded (tp=2):**

| Strategy | Atoms | Local Bytes | Fragments |
|----------|-------|-------------|-----------|
| BF16 | 1 | 331,776 | 1 |
| Int8 chan | 2 | 167,040 | 2 |
| Int8 blk[32] | 2 | 186,624 | 2 |
| Q4 blk[32] | 3 | 103,680 | 3 |

Every entry was produced by the same `plan_atom()` function. No special-casing.

---

## 6. Role Traits and Trait Intersection Constraints

### 6.1 Role Traits

Marker traits tag operand descriptors with their semantic role:

```mojo
trait Operand: pass
trait WeightOperand(Operand): pass
trait ActivationOperand(Operand): pass
trait ScaleOperand(Operand):
    comptime BLOCK_SIZE: Int
trait ZeroOperand(Operand):
    comptime BLOCK_SIZE: Int
```

These let kernel signatures express requirements as trait intersections rather than concrete type bundles:

```mojo
fn dense_linear[
    A: Encoding & Shaped & ActivationOperand,
    W: Encoding & Shaped & WeightOperand,
    O: Encoding & Shaped & ActivationOperand,
](seq_len: Int): ...

fn dequant_linear[
    A: Encoding & Shaped & ActivationOperand,
    W: Encoding & Shaped & WeightOperand,
    S: Encoding & Shaped & ScaleOperand,
    O: Encoding & Shaped & ActivationOperand,
](seq_len: Int): ...
```

### 6.2 Dimensional Validation

Kernel contracts compose as pure functions of trait properties:

```mojo
fn validate_linear_dims[
    A: Encoding & Shaped,
    W: Encoding & Shaped,
    O: Encoding & Shaped,
]():
    constrained[A.COLS == W.COLS, "K mismatch"]()
    constrained[O.COLS == W.ROWS, "N mismatch"]()
    constrained[O.ROWS == A.ROWS, "M mismatch"]()
```

This validation is Level 1 infrastructure — it works for bf16, f16, i8, any standard-layout encoding. Q4 packed weight (`cols / pack_ratio`) breaks `validate_linear_dims` because `A.COLS != W.COLS` — the packing ratio is an **atomic dimensional relationship** that cannot be composed away.

### 6.3 Groups Provide Role-Tagged Descriptors

The encoding group bridges atoms (for the loader) and role-tagged descriptors (for the executor):

```mojo
struct QuantGroupV2[...]:
    # For the loader: flat atoms
    comptime Weight = Slot[...]
    comptime Scale = Slot[...]

    # For the executor: role-tagged
    comptime WeightOp = WeightDesc[WeightEnc, Weight.ROWS, Weight.COLS]
    comptime ScaleOp = ScaleDesc[ScaleEnc, Scale.ROWS, Scale.COLS, block_size]
```

The loader sees `Weight` and `Scale` as generic atoms. The executor sees `WeightOp` and `ScaleOp` as role-tagged descriptors that can be consumed by trait-intersection-constrained kernel signatures.

---

## 7. Topology of the Problem Domain

The full system has a 5-level topological structure.

### Level 1: Flat Composition (Loader)

Independent contributions combined by arithmetic. No interaction terms.

| Relationship | Operation | Dependencies |
|-------------|-----------|-------------|
| dtype -> byte_width | 1:1 lookup (carrier trait) | none |
| elements x byte_width -> bytes | multiplication | independent |
| strategy.local(dim, tp) -> local_dim | producer method call | per-axis, independent |
| byte_count -> aligned_bytes | rounding | post-hoc |

### Level 2: Bounded Dispatch (Loader)

Strategy-determined dispatch. Permanently bounded to 2-3 cases.

With producer traits, this is no longer a switch — the strategy type carries the dispatch:
- `RowShard.fragments_per_shard[T]()` -> 1 (contiguous)
- `ColShard.fragments_per_shard[T]()` -> `T.ROWS` (strided)

### Level 3: Config-Time Derivation (Skeleton)

Non-flat relationships resolved once at model configuration time.

| Derivation | Inputs | Resolution |
|-----------|--------|-----------|
| Architecture dims -> slot shapes | Model config | `Slot[E, S, P, C.HIDDEN, C.HIDDEN, ...]` |
| Quant strategy -> scale shape | block_size, weight shape | `scale_cols = cols / block_size` |
| Weight strategy -> scale strategy | weight shard, scale_cols | Separate `ScaleS` type parameter |
| Encoding -> group type | Encoding family | `DenseGroup` vs `QuantGroupV2` vs `TripleGroup` |

### Level 4: Constraints (Skeleton, compile-time)

Irreducible coupling between atoms.

```
block_size must divide (cols / tp)
```

Validated by compile-time assertion. The loader never encounters an invalid combination.

Additional constraint: Q4 packing ratio creates atomic dimensional relationship (`packed_cols = cols / pack_ratio`), breaking standard `validate_linear_dims`.

### Level 5: Execution Dispatch (Executor, three-level decomposition)

The executor is not a single monolithic dispatch. It has internal structure:

**Level 5a: Always Composable** — contracts, sizing, role tagging. Shared by all kernels.

| Concern | Mechanism |
|---------|-----------|
| Buffer sizing | `T.ROWS * T.COLS * T.ELEMENT_BYTES` |
| Dimensional contracts | `constrained[A.COLS == W.COLS]` |
| Role identification | `WeightOperand`, `ScaleOperand` marker traits |
| Dispatch routing | `Group.Loader` type parameter |

**Level 5b: Conditionally Composable** — per-element load/dequant via actionable traits. Valid when structural assumptions hold (standard layout, sequential access, separated accumulation, no cross-element fusion, ISA-independent loop).

```mojo
trait WeightLoader:
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64: ...

fn generic_gemv[L: WeightLoader, K: Int, N: Int](...):
    for n in range(N):
        var acc = Float64(0)
        for k in range(K):
            acc += load_act(act_ptr, k) * L.load(weight_ptr, n, k, K, aux0, aux1)
        store_out(out_ptr, n, acc)
```

Standard dtypes (bf16, f16, f32) always satisfy the structural assumptions. Combinatorial reduction: `E x K` -> `E + K`.

**Level 5c: Never Composable** — optimized kernel bodies where load, dequant, tiling, and ISA intrinsics are co-designed. Five coupling axes: Load<->Tiling, Dequant<->Compute, Accumulator<->K-blocking, ISA<->Loop, Layout<->Kernel identity.

---

## 8. Independence Proof

### 8.1 The Test

10 atoms from 4 different quantization strategies, with mixed shard strategies:

```
[0] q_proj        (576 x 576) x 1B   RowShard   -> 165,888 B   1 frag
[1] q_proj_scale  (576 x 1)   x 4B   RowShard   ->   1,152 B   1 frag
[2] k_proj        (576 x 192) x 1B   RowShard   ->  55,296 B   1 frag
[3] k_proj_scale  (576 x 6)   x 4B   RowShard   ->   6,912 B   1 frag
[4] gate          (1536 x 288) x 1B  ColShard   -> 221,184 B  1536 frag
[5] gate_scale    (1536 x 18)  x 2B  ColShard   ->  27,648 B  1536 frag
[6] gate_zero     (1536 x 18)  x 2B  ColShard   ->  27,648 B  1536 frag
[7] up            (1536 x 576) x 1B  ColShard   -> 442,368 B  1536 frag
[8] up_scale      (1536 x 1)   x 4B  Replicated ->   6,144 B     1 frag
[9] up_bias       (1536 x 1)   x 4B  Replicated ->   6,144 B     1 frag
```

Every atom was processed by the same `plan_atom()` function. The function received no context about which atoms are weights vs. scales, which belong together, or what quantization produced them.

### 8.2 Aggregation

Cross-atom computations are limited to simple summation:
- **Total arena bytes:** sum of each atom's `arena_bytes`
- **Total IO fragments:** sum of each atom's `fragment_count`
- **Arena offsets:** cumulative prefix sum of `arena_bytes`

No atom's **size** depends on any other atom. Only **placement** (arena offset) depends on preceding atoms' sizes, which is inherent to sequential allocation.

---

## 9. What Does Not Decompose

Six concerns are non-flat. The first four are resolved upstream of the loader. The fifth and sixth are downstream.

### 9.1 Scale Shape Derivation (Skeleton)

**Problem:** `scale_cols = weight_cols / block_size` is a different formula per quantization strategy.

**Resolution:** The skeleton computes the formula once at config time via the encoding group and produces concrete atoms with explicit shapes.

### 9.2 Scale Shard Strategy Derivation (Skeleton)

**Problem:** A col-sharded weight with a channelwise scale (scale_cols=1) produces a replicated scale — the scale cannot be col-sharded along a single column.

**Resolution:** `QuantGroupV2` takes `ScaleS` as a separate type parameter. The skeleton makes the type-level decision: scale uses `Replicated` while weight uses `ColShard`.

### 9.3 Block Divisibility Constraint (Skeleton)

**Problem:** `block_size` must divide `cols / tp`. Irreducible 4-way coupling.

**Resolution:** Compile-time assertion. Invalid configurations are rejected before loading begins.

### 9.4 Consumer Pairing (Skeleton)

**Problem:** The forward pass needs to know that atom X is the weight and atom Y is its scale.

**Resolution:** The encoding group defines named slots and provides role-tagged descriptors. `Group.WeightOp` and `Group.ScaleOp` carry `WeightOperand` and `ScaleOperand` marker traits for the executor.

### 9.5 Execution Dispatch — Level 5a+5b (Executor, Conditionally Composable)

**Problem:** Kernel needs to match encoding for load+dequant.

**Resolution for standard dtypes:** Actionable traits (`WeightLoader`) carry per-element dequant. Kernel is generic over the loader type. The group provides `Loader = BF16Loader` or `Loader = I8ChanLoader`. Monomorphized at compile time — zero overhead.

**Structural assumptions required:** Standard layout, sequential access, separated accumulation, no cross-element fusion, ISA-independent loop structure. Standard dtypes (bf16, f16, f32) always satisfy these.

### 9.6 Execution Dispatch — Level 5c (Executor, Never Composable)

**Problem:** Optimized packed kernels fuse load, dequant, tiling, and ISA intrinsics.

**Resolution:** Selected as monolithic units by Level 5a dispatch. The type system chooses WHICH kernel to call, not how to compose it. Five coupling axes make decomposition impossible:
1. Load <-> Tiling (repacked buffer layout)
2. Dequant <-> Compute (interleaved mask/shift/dpbusd)
3. Accumulator <-> K-blocking (position-dependent init)
4. ISA <-> Loop structure (AVX vs AMX = different programming models)
5. Data layout <-> Kernel identity (buffer IS the access pattern)

**This is irreducible.** The optimization that makes these kernels fast IS the coupling. The design accommodates them through the same skeleton infrastructure — the group carries a kernel identity instead of (or alongside) an actionable trait.

---

## 10. The Three-Layer Architecture

```
+---------------------------------------------------------+
|  SKELETON (architecture-specific, compile-time)          |
|  * Composes: Encoding x Strategy x Placement             |
|  * Encoding groups bridge loader/executor/kernel         |
|  * Produces flat atoms for loader                        |
|  * Provides role-tagged descriptors for executor         |
|  * Carries actionable traits OR kernel identities        |
|  * Validates constraints                                 |
+---------------------------------------------------------+
|  LOADER (universal, algebraic)                           |
|  * Maps plan_atom() over atoms                           |
|  * Flat arithmetic + bounded dispatch                    |
|  * Zero encoding/execution/role awareness                |
+---------------------------------------------------------+
|  EXECUTOR (three-level decomposition)                    |
|  * Level 1: contracts, sizing, role tags (always)        |
|  * Level 2: actionable traits (when assumptions hold)    |
|  * Level 3: monolithic kernels (fused, peak perf)        |
+---------------------------------------------------------+
```

The loader and executor never interact. They receive independent outputs from the skeleton. This separation is absolute — neither layer knows about the other's concerns.

---

## 11. Design Implications

### 11.1 Adding a New Encoding

To add `Int8Channelwise`:
1. Define `I8` encoding type (carrier trait): `comptime ELEMENT_BYTES = 1`
2. Define `I8ChanLoader` (actionable trait): load + multiply by channelwise scale
3. Define decomposition in skeleton via `QuantGroupV2`: weight slot + scale slot with appropriate shapes
4. The group carries `Loader = I8ChanLoader` for Level 2 dispatch

**Changes to the loader: zero.**

### 11.2 Adding a New Shard Strategy

To add diagonal sharding:
1. Define `DiagonalShard(ShardStrategy)` with appropriate static methods
2. Use it in skeleton slot definitions

**Changes to the loader: zero.** Changes to existing strategies: zero.

### 11.3 Adding a New Architecture

To support GPT-2 instead of Llama:
1. Define `GPT2Arch` with its dimension traits
2. Define `GPT2Skeleton` with its specific slots and strategies
3. Reuse all existing encodings, strategies, placements, encoding groups

**Changes to the loader: zero.** Changes to existing traits: zero.

### 11.4 Adding an Optimized Kernel

To add a hand-tuned int4 GEMM:
1. The kernel, its buffer layout, its tiling, and its intrinsics are a monolithic unit
2. The skeleton's group carries the kernel identity for Level 3 dispatch
3. Level 1 contracts (validation, sizing) still compose — shared infrastructure
4. The forward pass routes to the optimized kernel via the group's dispatch

**Changes to the loader: zero.** Level 1 infrastructure is reused.

### 11.5 The Cost of a New Concern

| Concern | Where | Cost |
|---------|-------|------|
| New encoding | Skeleton + Executor | O(1) — carrier type + actionable trait + encoding group |
| New shard strategy | Skeleton | O(1) — one producer type |
| New placement | Skeleton | O(1) — one producer type |
| New architecture | Skeleton | O(N) — one skeleton with N slots |
| New quantization scheme | Skeleton + Executor | O(1) — group decomposition + loader or kernel |
| New optimized kernel | Executor L3 | O(1) — monolithic unit, selected by dispatch |

No concern has combinatorial cost. No concern requires modifying unrelated components.

---

## 12. Summary

The model description problem has clean algebraic structure across its three layers, with the execution layer revealing internal decomposition:

**Loader:** Fully compositional. Every atom processed identically by `plan_atom()`. No encoding awareness, no execution awareness, no role awareness. Universal.

**Skeleton:** Architecture-specific but built from algebraic components (carrier traits, producer traits, trait intersections, encoding groups). The only irreducibly model-specific layer. Provides three views of each tensor group: flat atoms for the loader, role-tagged descriptors for the executor, and dispatch information for the kernel.

**Executor:** Three-level decomposition. Level 1 (contracts, sizing, role tags) always composes. Level 2 (actionable traits for per-element ops) composes when structural assumptions hold — always true for standard dtypes, sometimes true for simple quantization. Level 3 (fused kernel bodies) never composes — monolithic atoms selected by dispatch.

The atom is the boundary between skeleton and loader. It is the curried form of the loading problem. Above it: derivation, composition, architectural decisions. Below it: flat arithmetic over independent values.

The execution boundary is the boundary between skeleton and executor. Above it: type-level strategy selection. Below it: a gradient from composable contracts to monolithic kernels, determined by structural properties of the data and the optimization level.

**The loader is invariant to encoding, quantization, architecture, execution strategy, and kernel optimization level. The executor is invariant to loading, sharding, placement, and architecture. The skeleton bridges both, composing general components into specific configurations and providing the right view to each consumer.**

---

## 13. Related Analysis

- **design_algebra.md** — Generality classification, compositionality gradient, producer trait patterns
- **kernel_analysis.md** — Full kernel decomposition: actionable traits, five coupling axes, structural assumptions, Level 2 vs Level 3 selection criteria
- **design_exploration3.mojo** — Encoding group working code with concrete numbers
- **design_exploration4.mojo** — Role traits and trait intersection constraints
- **design_exploration5.mojo** — Actionable traits, combinatorial reduction, enriched loaders
