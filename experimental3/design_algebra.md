# Design Algebra: Generality Analysis

## Thesis

The model description problem separates cleanly into three domains with fundamentally different algebraic character:

1. **Configuration & Loading** — fully compositional. Every concept decomposes into independent atoms that compose via flat arithmetic.
2. **Materialization (Sharding, Placement)** — algebraic via type-level producer traits. Shard strategy decomposes into per-dimension atoms (`Divide`, `Keep`). Placement is orthogonal.
3. **Execution (Compute Kernels)** — **not universally compositional**, but not monolithically atomic either. Decomposes into three levels: contracts (always compose), per-element operations (compose when structural assumptions hold), and fused kernel bodies (never compose).

The boundary between compositional and non-compositional is the **execution boundary**: the point where data description meets hardware dispatch. Everything upstream of this boundary is algebraic. At and below it, compositionality is a gradient determined by structural properties of the data and the kernel.

---

## 1. The Generality Map

### Algebraic (General, Composable, Reusable)

| Concept | Nature | Why It's Algebraic |
|---------|--------|--------------------|
| `byte_count` | `prod(dims) * elem_bytes` | Universal multiplication — works for any encoding, any shape |
| `DimStrategy` | `Divide` or `Keep` per dimension | Two atoms compose to describe any 2D shard policy |
| `ShardStrategy` | `Shard2D[D0, D1]` | Compositional — row, col, replicated all emerge from same mechanism |
| `Placement` | `HostNode`, `AllNodes` | Orthogonal to shape — *where*, not *how* |
| Fragment formula | Fixed per strategy type | Parametrically algebraic — the strategy type IS the parameter |
| Allreduce | Bound to strategy identity | Definitional — `ColShard` implies allreduce, `RowShard` doesn't |
| Naming instantiation | `suffix + layer_idx` | Formulaic within a family |
| Loader algorithm | `plan_atom()` | Universal — same function for every atom regardless of origin |
| Dimensional contracts | `constrained[A.COLS == W.COLS]` | Level 1 kernel infrastructure — shared by ALL kernels |
| Role tagging | `WeightOperand`, `ScaleOperand` | Marker traits — compose regardless of kernel optimization level |

### Atomic (Irreducible, Concrete)

| Concept | Nature | Why It's Atomic |
|---------|--------|-----------------|
| `Encoding` | `DTYPE -> ELEMENT_BYTES` | 1:1 lookup — no independent variation to generalize |
| `Shape` | `(ROWS, COLS)` | 2D is the actual structure for Llama family tensors |
| Naming convention | `model.layers.{idx}.{suffix}` | Family-specific standard, not derivable |
| Skeleton structure | Which slots exist, which strategies they use | Architecture-specific by definition |
| Architecture dims | `HIDDEN`, `N_HEADS`, `INTERMEDIATE` | Model-specific constants |
| Packing ratio | `cols / pack_ratio` (Q4 packed weight) | Dimensional relationship that breaks standard validation |

### Conditionally Compositional (Depends on Structural Assumptions)

| Concept | Nature | When It Composes | When It Doesn't |
|---------|--------|------------------|-----------------|
| Per-element dequant | `WeightLoader.load()` | Standard layout, sequential access | Repacked tiled layouts |
| Output casting | `OutputWriter.store()` | Standard output layout | Fused accumulate-and-store |
| Activation loading | `ActLoader.load()` | Standard input layout | Interleaved multi-pass |

### Non-Compositional (Forced Enumeration)

| Concept | Nature | Why It Doesn't Compose |
|---------|--------|------------------------|
| Fused kernel body | Tiling + intrinsics + pipeline fusion | Five coupling axes bind them (see Section 6.2) |
| Mixed-precision matmul | `bf16 x bf16 -> bf16`, `i8 x i8 -> i32`, etc. | Finite set of supported operations, not a product space |
| SIMD dispatch | ISA-specific intrinsics | `AVX2`, `AVX-512`, `NEON` have different valid operations |
| Buffer repacking | Kernel-specific layout transform | Co-designed with tiling — not separable |

---

## 2. The Execution Boundary

### Why Execution Doesn't Compose Universally

Consider matmul type combinations. If types composed freely, any `(A, B) -> C` would be valid:

```
bf16 x bf16 -> bf16    Yes  (native)
bf16 x bf16 -> f32     Yes  (accumulate wider)
i8   x i8   -> i32     Yes  (VNNI/int8 matmul)
f16  x f16  -> f32     Yes  (native)
i8   x bf16  -> ???    No   (no hardware support)
i4   x bf16  -> ???    No   (requires dequant pipeline)
f32  x i8   -> bf16    No   (nonsensical)
```

The valid set is not `types x types -> types`. It is a **finite lookup table** determined by:
- Hardware ISA (AVX2 vs AVX-512 vs NEON)
- Available intrinsics (VNNI, BF16 dot product, etc.)
- Numerical correctness constraints (accumulator precision)

However, this non-compositionality has internal structure. See Section 6 for the three-level decomposition.

### The Three-Layer Architecture

```
+---------------------------------------------------------+
|  SKELETON (architecture-specific, compile-time)          |
|  * Names slots, assigns strategies & placements          |
|  * Knows encoding -> decomposes to atoms                 |
|  * Provides three views: loader, executor, kernel        |
|  * Validates constraints (block divisibility, etc.)      |
+---------------------------------------------------------+
|  LOADER (universal, algebraic)                           |
|  * Receives atoms, maps plan_atom() over them            |
|  * No knowledge of encoding, quantization, execution     |
|  * Flat arithmetic + bounded IO dispatch                 |
+---------------------------------------------------------+
|  EXECUTOR (three-level decomposition, ISA-aware)         |
|  * Level 1: contracts, sizing, dispatch (always compose) |
|  * Level 2: per-element ops via actionable traits        |
|  * Level 3: monolithic kernels (fused, never compose)    |
+---------------------------------------------------------+
```

The skeleton sits at the top because it is the only layer that knows both the data description (which the loader needs) and the execution requirements (which the executor needs). It produces atoms for the loader and selects kernels for the executor. These two outputs are independent — loading and execution don't interact.

---

## 3. Producer Traits: Extending the Compute Graph via Types

The most powerful pattern discovered in this exploration is the **producer trait** — a trait whose types carry computation, not just data. This replaces integer-encoded dispatch with type-level strategy selection.

### 3.1 The Problem: Integer Shard Axis

The original design encoded shard strategy as an integer:

```mojo
# Integer encoding — an inverted enum
trait Sharded:
    comptime SHARD_AXIS: Int   # -1 = replicated, 0 = row, 1 = col
```

This externalizes computation. The consumer must switch on the integer:

```mojo
# The consumer does the work
var lr = rows // tp if SHARD_AXIS == 0 else rows
var lc = cols // tp if SHARD_AXIS == 1 else cols
```

The meaning of `0` and `1` is implicit. Adding a new strategy (e.g., diagonal sharding) requires modifying every consumer.

### 3.2 The Solution: Strategy Types

The strategy IS a type. It carries its own computation:

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

struct ColShard(ShardStrategy):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: return r
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: return c // tp
```

Now the computation lives WITH the strategy. No consumer switch. Adding a new strategy adds a new type — zero changes to consumers.

### 3.3 The Key Pattern: Comptime Members via Type Parameter Methods

The critical discovery: **Mojo allows calling static methods on trait-bound type parameters in comptime member initialization.**

```mojo
struct Slot[E: Encoding, S: ShardStrategy, rows: Int, cols: Int, tp: Int = 1](
    Encoding, Shaped,
):
    # The type parameter S PRODUCES the comptime value
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)

    # Carrier trait delegation (flat forwarding)
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
```

This is the mechanism that extends the compute graph through types. `Self.S.shard_rows(...)` is not a runtime call — it is resolved at compile time. The strategy type parameter determines which formula computes `ROWS`. Different strategies produce different formulas, but the Slot struct has **one definition**.

### 3.4 Composable Consumers via Trait Intersection

Free functions consume trait intersections without knowing concrete types:

```mojo
fn slot_bytes[T: Encoding & Shaped]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES
```

This function works for:
- `Slot[BF16, RowShard, 576, 576, 2]` -> `288 * 576 * 2 = 331,776`
- `Slot[I8, ColShard, 576, 576, 2]` -> `576 * 288 * 1 = 165,888`
- Any future Slot parameterization

No switch, no special-casing. The function sees `ROWS`, `COLS`, `ELEMENT_BYTES` — it doesn't know or care how they were derived. This is the payoff of the carrier/producer separation.

### 3.5 Higher-Order Producers: Traits Consuming Traits

A producer trait's methods can be parameterized on carrier traits:

```mojo
trait FragmentCounter:
    @staticmethod
    fn count[T: Shaped](tp: Int) -> Int: ...

struct ContiguousRead(FragmentCounter):
    @staticmethod
    fn count[T: Shaped](tp: Int) -> Int: return 1

struct StridedRead(FragmentCounter):
    @staticmethod
    fn count[T: Shaped](tp: Int) -> Int: return T.ROWS
```

Here the producer (`StridedRead`) consumes a carrier (`Shaped`) to produce a value. The carrier provides the data (`ROWS`), the producer provides the formula. This is **higher-order composition** — the producer's computation is parameterized by the carrier's data.

### 3.6 Multi-Concern Strategies

A single strategy type can carry multiple related concerns:

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

struct ColShardFull(FullShardStrategy):
    @staticmethod
    fn shard_rows(r: Int, tp: Int) -> Int: return r
    @staticmethod
    fn shard_cols(c: Int, tp: Int) -> Int: return c // tp
    @staticmethod
    fn fragments_per_shard[T: Shaped]() -> Int: return T.ROWS
    @staticmethod
    fn needs_allreduce() -> Bool: return True
```

Shape derivation, fragment counting, and allreduce signaling are all bound to the strategy identity. They travel together because they are facets of the same concept. This is not a monolithic trait — each method is independently callable. But they are **definitionally co-located** because they describe facets of the same semantic object.

---

## 4. DimStrategy: The True Atom of Sharding

### The Decomposition

Sharding decomposes further than `RowShard`/`ColShard`. The true atom is the **per-dimension strategy**:

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

comptime RowShard   = Shard2D[Divide, Keep]     # divide rows, keep cols
comptime ColShard   = Shard2D[Keep, Divide]      # keep rows, divide cols
comptime Replicated = Shard2D[Keep, Keep]         # keep both
```

This is algebraic: 2 atoms compose into 4 strategies (2^2), of which 3 are useful (the 4th, `Shard2D[Divide, Divide]`, divides both dimensions — valid but rarely used).

### The HostOnly Discovery

With integer encoding, `HostOnly` was `SHARD_AXIS = -2` — a special value distinct from `Replicated` (`SHARD_AXIS = -1`). But `HostOnly` and `Replicated` produce **identical shapes**: both are `Shard2D[Keep, Keep]`.

The difference is **placement**: HostOnly lives on one node, Replicated lives on all nodes. These were conflated in the integer encoding. Separating them is correct:

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

Now the Slot carries both independently:

```mojo
struct Slot[E: Encoding, S: ShardStrategy, P: Placement, rows: Int, cols: Int,
            suffix: StringLiteral, tp: Int = 1](Encoding, Shaped):
    comptime ROWS = Self.S.shard_rows(Self.rows, Self.tp)
    comptime COLS = Self.S.shard_cols(Self.cols, Self.tp)
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES

    @staticmethod
    fn on_node(node_idx: Int, host_idx: Int) -> Bool:
        return Self.P.on_node(node_idx, host_idx, Self.tp)
```

Shape and placement are orthogonal axes. The Slot composes them without coupling.

---

## 5. The Skeleton as Bridge

The skeleton is the **only** architecture-specific layer. It is irreducible — you cannot generalize "which slots a Llama model has" without losing the specificity that makes it a Llama model. But its components are all general:

```mojo
struct Skeleton[C: Arch, E: Encoding, tp: Int = 1]:
    comptime KV_HIDDEN = Self.C.N_KV_HEADS * Self.C.HEAD_DIM

    # Each slot composes: Encoding x Strategy x Placement x Shape
    comptime Embed = Slot[BF16, Replicated, HostNode,
        Self.C.VOCAB, Self.C.HIDDEN, "embed_tokens.weight"]

    comptime QProj = Slot[Self.E, RowShard, ShardedNodes,
        Self.C.HIDDEN, Self.C.HIDDEN, "self_attn.q_proj.weight", Self.tp]

    comptime OProj = Slot[Self.E, ColShard, ShardedNodes,
        Self.C.HIDDEN, Self.C.HIDDEN, "self_attn.o_proj.weight", Self.tp]
```

The skeleton's decisions:
- **Embed** uses `BF16` (always full precision), `Replicated` (same data everywhere), `HostNode` (only on host)
- **QProj** uses parameterized encoding `E`, `RowShard`, `ShardedNodes`
- **OProj** uses `ColShard` (implies allreduce after matmul)

These are architectural decisions. They cannot be derived from first principles. They are the definition of the model.

### 5.1 Encoding Groups: Multi-Buffer Composition

The skeleton uses encoding groups to handle multi-buffer quantized encodings while maintaining self-descriptiveness:

```mojo
struct DenseGroup[E, S, P, rows, cols, suffix, tp]:
    comptime Weight = Slot[Self.E, Self.S, Self.P, Self.rows, Self.cols, Self.suffix, Self.tp]
    comptime Loader = BF16Loader        # Level 2 actionable trait
    @staticmethod fn atom_count() -> Int: return 1

struct QuantGroupV2[WeightEnc, ScaleEnc, S, ScaleS, P, rows, cols, block_size, ...]:
    comptime Weight = Slot[Self.WeightEnc, Self.S, Self.P, Self.rows, Self.cols, ...]
    comptime SCALE_COLS = Self.cols // Self.block_size
    comptime Scale = Slot[Self.ScaleEnc, Self.ScaleS, Self.P, Self.rows, Self.SCALE_COLS, ...]
    comptime Loader = I8ChanLoader      # Level 2 actionable trait

    # Role-tagged descriptors for executor
    comptime WeightOp = WeightDesc[Self.WeightEnc, Self.Weight.ROWS, Self.Weight.COLS]
    comptime ScaleOp = ScaleDesc[Self.ScaleEnc, Self.Scale.ROWS, Self.Scale.COLS, Self.block_size]
```

Key finding: **scale strategy must be a separate type parameter.** ColShard + channelwise produces `scale_cols=1`; dividing 1 by `tp=2` gives 0. The scale must use `Replicated` while the weight uses `ColShard`. The group makes this a type-level decision.

---

## 6. Where Composition Breaks — And Where It Partially Holds

### The Nature of the Break

The data description layer composes because `bytes = rows * cols * elem_bytes` is a product of independent terms. There are no invalid combinations — any `(rows, cols, elem_bytes)` triple is a valid buffer.

The execution layer does NOT compose universally because the valid `(A_dtype, B_dtype, accumulator, output)` tuples are constrained by hardware. The product space has holes:

| A | B | Acc | Valid? | Reason |
|---|---|-----|--------|--------|
| bf16 | bf16 | f32 | Yes | Native BF16 dot product |
| i8 | i8 | i32 | Yes | VNNI support |
| f16 | f16 | f32 | Yes | Native FP16 |
| bf16 | i8 | ??? | No | No mixed bf16xi8 hardware op |
| i8 | bf16 | ??? | No | Same — asymmetric not supported |
| i4 | i4 | i32 | Maybe | Only with specific hardware |

However, the execution layer is not monolithically non-compositional. It decomposes into **three levels** with different algebraic character.

### 6.1 Three Levels of Kernel Decomposition

**Level 1: Always Composable** — contracts, sizing, dispatch selection, role tagging. These describe WHAT, never HOW. They compose regardless of kernel optimization level.

| Concern | Mechanism |
|---------|-----------|
| Buffer sizing | `T.ROWS * T.COLS * T.ELEMENT_BYTES` via `Encoding & Shaped` |
| Dimensional contracts | `constrained[A.COLS == W.COLS, "K mismatch"]()` |
| Scale topology | `constrained[W.COLS == S.COLS * S.BLOCK_SIZE]` |
| Role tagging | `WeightOperand`, `ScaleOperand` marker traits |
| Dispatch selection | `Group.Loader` type parameter on the skeleton |
| Allreduce decision | `Loader.NEEDS_ALLREDUCE` comptime property |
| Shape arithmetic | `output_cols[W]()`, `contraction_dim[W]()` — pure functions of trait properties |

These are the trait intersection constraints from design_exploration4. They compose regardless of whether the kernel body is a naive loop or hand-tuned intrinsics.

**Level 2: Conditionally Composable** — per-element operations factored into actionable traits. Valid when data layout is standard and the pipeline is sequential.

```mojo
trait WeightLoader:
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64: ...

struct BF16Loader(WeightLoader):
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64:
        return bf16_to_f32(weight_ptr, row * stride + col)  # aux unused

struct I8ChanLoader(WeightLoader):
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64:
        return i8_to_f32(weight_ptr, row * stride + col) * f32_load(aux0, row)

fn generic_gemv[L: WeightLoader, K: Int, N: Int](
    act_ptr: Int, weight_ptr: Int, out_ptr: Int, aux0: Int, aux1: Int):
    for n in range(N):
        var acc = Float64(0)
        for k in range(K):
            acc += load_act(act_ptr, k) * L.load(weight_ptr, n, k, K, aux0, aux1)
        store_out(out_ptr, n, acc)
```

This gives `E + K` items instead of `E x K`. The combinatorial reduction is real:

| Scenario | Without | With | Reduction |
|----------|---------|------|-----------|
| 4 enc x 3 shapes | 12 | 7 | 42% |
| 4 enc x 3 shapes x 2 writers | 24 | 9 | 63% |
| 6 enc x 5 shapes x 2 writers | 60 | 13 | 78% |

Each new axis adds O(1), not O(N). Mojo monomorphizes `L.load()` at compile time — zero overhead.

**Level 3: Never Composable** — optimized kernel bodies where load, dequant, tiling, and ISA intrinsics are co-designed into a single fused unit. Selected by Level 1 dispatch, not decomposed by traits.

```
GemmKernel224Int4_1_LowKGroup
    +-- BufferA layout (co-designed)
    +-- BufferB layout (co-designed)
    +-- AVX inner loop (fused mask + shift + dpbusd)
    +-- AMX inner loop (tile load + run_tile)
    +-- K-block accumulator management
    +-- Lo/Hi nibble processing order
```

These are selected as a unit. The type system's role here is choosing WHICH monolithic kernel to call, not composing the kernel from parts.

### 6.2 Structural Assumptions for Level 2

The actionable trait pattern holds when ALL of these are true:

| Assumption | When it holds | When it breaks |
|------------|---------------|----------------|
| Standard layout | bf16/f16/f32 row-major | Repacked tiled layouts |
| Sequential access | Naive inner loops | Interleaved lo/hi nibble passes |
| Separated accumulation | Simple dot products | K-blocked reductions with position-dependent init |
| No cross-element fusion | Independent element ops | Shared packed byte for nibble pairs |
| ISA-independent loop | Same loop, different width | AVX vs AMX fundamentally different |

Standard dtypes (bf16, f16, f32) satisfy all five. The only difference between a bf16 and f16 kernel is the cast: `bf16_to_f32` vs `f16_to_f32`. An actionable trait carrying this cast generalizes perfectly. This is what CUTLASS/CuTe achieve with macros, and it works because the operations are genuinely independent.

Packed/quantized kernels with repacked buffers violate most or all. Five coupling axes bind them:
1. **Load <-> Tiling** — repacked buffer layout co-designed with kernel tiling
2. **Dequant <-> Compute** — mask/shift/dpbusd interleaved, not sequential
3. **Accumulator <-> K-blocking** — position-dependent init within reduction dim
4. **ISA <-> Loop structure** — AVX explicit loops vs AMX tile ops are different programming models
5. **Data layout <-> Kernel identity** — the buffer IS the kernel's access pattern

### 6.3 The Compositionality Is a Gradient, Not a Wall

```
FULLY COMPOSITIONAL ---------------------------------------- NOT COMPOSITIONAL

Level 1              Level 2               Level 3
Contracts            Per-element ops       Fused kernel body
Sizing               Actionable traits     Tiling + intrinsics
Dispatch selection   (when assumptions     Buffer repacking
Role tagging          hold)                Pipeline fusion
```

The design exploits every level:
- Level 1 infrastructure (trait intersections, role tags, validation) is shared by ALL kernels
- Level 2 actionable traits cover standard dtypes and prototyping
- Level 3 monolithic dispatch handles peak-performance packed kernels

### 6.4 The Encoding Group as Bridge

The skeleton's encoding group bridges all three levels:

```mojo
struct QuantGroupV2[...]:
    # Level 1: flat atoms for loader, role-tagged descriptors for executor
    comptime Weight = Slot[WeightEnc, S, P, rows, cols, suffix, tp]
    comptime WeightOp = WeightDesc[WeightEnc, Weight.ROWS, Weight.COLS]
    comptime ScaleOp = ScaleDesc[ScaleEnc, Scale.ROWS, Scale.COLS, block_size]

    # Level 2 or 3: kernel dispatch
    comptime Loader = I8ChanLoader   # actionable trait (Level 2)
    # or: kernel identity for monolithic dispatch (Level 3)
```

| Layer | Knows About | Doesn't Know About |
|-------|-------------|---------------------|
| Loader | Atom shape, bytes, shard geometry | Encoding semantics, execution |
| Skeleton | Everything — it's the bridge | — |
| Executor | Encoding semantics, kernel level | Loading, shard geometry |

The forward pass reads from the group:

```mojo
# Level 2 (actionable trait — reference/standard-dtype kernel):
generic_gemv[Skel.QProj.Loader, K, N](ptrs...)

# Level 3 (monolithic — optimized packed kernel):
optimized_int4_gemm(ptrs..., Skel.QProj.Weight.ROWS, Skel.QProj.Weight.COLS)
```

Both paths consume the same skeleton. The choice between Level 2 and Level 3 is a performance/development tradeoff, not an architectural one.

---

## 7. Classification Summary

### The Algebra Table

| Concept | Layer | Algebraic? | Reason |
|---------|-------|------------|--------|
| DimStrategy (Divide/Keep) | Config | **Yes** | Two atoms compose to describe all 2D shard policies |
| Shard2D[D0, D1] | Config | **Yes** | Composition of DimStrategies |
| Placement (Host/All) | Config | **Yes** | Orthogonal to shape, independent axis |
| byte_count = dims x elem_bytes | Loader | **Yes** | Universal multiplication |
| Fragment generation | Loader | **Yes** | Parametrically algebraic per strategy type |
| Dimensional contracts | Executor L1 | **Yes** | Trait intersection constraints, shared by all kernels |
| Role tagging | Executor L1 | **Yes** | Marker traits compose freely |
| Allreduce signaling | Executor L1 | **Yes** | Definitional — bound to strategy identity |
| Per-element dequant | Executor L2 | **Conditional** | Composes when structural assumptions hold (standard dtypes) |
| Actionable trait loaders | Executor L2 | **Conditional** | E+K reduction, but requires standard layout |
| Encoding (DTYPE -> bytes) | Config | **No** | 1:1 lookup, irreducible |
| Shape (ROWS, COLS) | Config | **No** | Concrete data, not derivable |
| Skeleton structure | Config | **No** | Architecture-specific by definition |
| Packing ratio (Q4) | Config | **No** | Atomic dimensional relationship |
| Fused kernel body | Executor L3 | **No** | Five coupling axes, irreducibly hardware-specific |
| Dequantization pipeline | Executor L3 | **No** | Concrete recipe per quantization, when fused |
| SIMD intrinsics | Executor L3 | **No** | ISA-specific, not abstractable |
| Buffer repacking | Executor L3 | **No** | Co-designed with tiling, not separable |

### The Compositionality Gradient

```
FULLY COMPOSITIONAL ---------------------------------------------- NOT COMPOSITIONAL

DimStrategy    Placement    byte_count    Contracts    Loaders     Fused kernels
   |              |            |             |            |              |
   |   Compose    |   Compose  |   Compose   |  Compose   | Conditional  |   Enumerate
   |   freely     |   freely   |   freely    |  always    | (std dtypes) |   concretely
   |              |            |             |            |              |
   +-------- LOADER ----------+             +-- EXECUTOR L1+L2 ---------+
                                                                   L3 --+
                                      SKELETON (bridge)
```

The design achieves maximal compositionality where it's possible (loading, configuration), exploits conditional compositionality where structural assumptions hold (standard-dtype kernels via actionable traits), and cleanly isolates forced enumeration where it's necessary (optimized packed kernels). The skeleton is the bridge — it lives in all three worlds.

---

## 8. Related Analysis

- **atomic_analysis.md** — Detailed decomposition of the loading problem, atom independence proof, carrier/producer trait architecture
- **kernel_analysis.md** — Full kernel decomposition: actionable trait pattern, five coupling axes, structural assumptions, practical architecture for Level 2 vs Level 3 selection
- **design_exploration2.mojo** — DimStrategy, Placement, Shard2D working code
- **design_exploration3.mojo** — Encoding groups (DenseGroup, QuantGroupV2, TripleGroup) with concrete numbers
- **design_exploration4.mojo** — Role traits, trait intersection constraints, kernel interface composition
- **design_exploration5.mojo** — Actionable traits (WeightLoader), combinatorial reduction, enriched loaders
