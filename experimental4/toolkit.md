# Toolkit: What You Need to Know

This document collects observed facts, measured properties, and verified constraints relevant to building a static CPU inference engine for SmolLM2-135M in Mojo. It does not prescribe a design. It is a reference — things that are true about the problem, the hardware, the data, and the language, independent of how you choose to structure the solution.

---

## 1. The Model

SmolLM2-135M is a LLaMA-family decoder-only causal language model.

### 1.1 Architecture Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` | 576 | config.json |
| `num_hidden_layers` | 30 | config.json |
| `num_attention_heads` | 9 | config.json |
| `num_key_value_heads` | 3 | config.json (GQA) |
| `intermediate_size` | 1536 | config.json |
| `max_position_embeddings` | 8192 | config.json |
| `rope_theta` | 100000 | config.json |
| `vocab_size` | 49152 | config.json |
| `tie_word_embeddings` | true | config.json |
| `rms_norm_eps` | 1e-5 | config.json |
| `hidden_act` | silu | config.json |

Derived:
- `head_dim = 576 / 9 = 64`
- `kv_hidden = 3 * 64 = 192`
- GQA repetition factor: `9 / 3 = 3`

### 1.2 Forward Pass Structure

```
embed_tokens(input_ids) -> [B, T, 576]

for each of 30 layers:
    h = RMSNorm(x)
    Q = q_proj(h)      # [B, T, 576]
    K = k_proj(h)      # [B, T, 192]
    V = v_proj(h)      # [B, T, 192]
    apply RoPE to Q, K
    repeat_kv(K, V, factor=3)
    attn = softmax(Q @ K^T / sqrt(64) + causal_mask) @ V
    x = x + o_proj(attn)

    h = RMSNorm(x)
    x = x + down_proj(silu(gate_proj(h)) * up_proj(h))

final RMSNorm
logits = x @ embed_tokens.weight^T    (tied weights)
```

### 1.3 The Tensor Inventory

Per layer, 9 weight matrices:

| Tensor | Rows | Cols | Notes |
|--------|------|------|-------|
| `q_proj.weight` | 576 | 576 | Full hidden -> full hidden |
| `k_proj.weight` | 192 | 576 | KV hidden -> full hidden |
| `v_proj.weight` | 192 | 576 | KV hidden -> full hidden |
| `o_proj.weight` | 576 | 576 | Full hidden -> full hidden |
| `gate_proj.weight` | 1536 | 576 | Hidden -> intermediate |
| `up_proj.weight` | 1536 | 576 | Hidden -> intermediate |
| `down_proj.weight` | 576 | 1536 | Intermediate -> hidden |
| `input_layernorm.weight` | 576 | — | 1D, per-element scale |
| `post_attention_layernorm.weight` | 576 | — | 1D, per-element scale |

Global tensors:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `embed_tokens.weight` | 49152 x 576 | Embedding + tied LM head |
| `norm.weight` | 576 | Final RMSNorm |

All weight matrices in the safetensors file are 2D. The head structure (9 query heads, 3 KV heads, head_dim=64) is a reshaping/indexing concern in the forward pass, not a storage concern.

### 1.4 Naming Convention

```
model.embed_tokens.weight
model.layers.{0..29}.self_attn.{q,k,v,o}_proj.weight
model.layers.{0..29}.mlp.{gate,up,down}_proj.weight
model.layers.{0..29}.input_layernorm.weight
model.layers.{0..29}.post_attention_layernorm.weight
model.norm.weight
```

### 1.5 Matmul Dimension Relationships

For a linear layer `output = input @ weight^T`:
- Input shape: `[B, T, K]`
- Weight shape: `[N, K]` (stored as N rows, K cols)
- Output shape: `[B, T, N]`

The contraction dimension K equals `weight.cols`. The output dimension N equals `weight.rows`.

Attention projections:
- q_proj: K=576, N=576
- k_proj: K=576, N=192
- v_proj: K=576, N=192
- o_proj: K=576, N=576

MLP:
- gate_proj: K=576, N=1536
- up_proj: K=576, N=1536
- down_proj: K=1536, N=576

---

## 2. Encodings and Quantization

### 2.1 Encoding Properties

An encoding determines the byte width and dtype of stored elements. These are 1:1 lookups with no independent variation — `DType` determines `ELEMENT_BYTES` uniquely.

| Encoding | DType | Element Bytes |
|----------|-------|---------------|
| BF16 | bfloat16 | 2 |
| F16 | float16 | 2 |
| F32 | float32 | 4 |
| I8 | int8 | 1 |
| Q4 (packed) | uint8 | 1 (but 2 values per byte) |

### 2.2 Buffer Decomposition Per Encoding

Every quantization strategy produces a fixed number of independent buffers. This count is a property of the encoding family, not the tensor or the model.

| Encoding Family | Buffer Count | Buffers |
|-----------------|-------------|---------|
| Dense (BF16/F16/F32) | 1 | weight |
| Int8 channelwise | 2 | weight + scale |
| Int8 channelwise + bias | 3 | weight + scale + bias |
| Int8 blockwise[B] | 2 | weight + scale |
| Q4 blockwise[B] | 3 | packed_weight + scale + zero |

Each buffer has its own shape, dtype, and byte width. The decomposition is performed once per encoding family — it is not per-tensor or per-layer.

### 2.3 Scale Shape Derivation

All scale shapes follow one formula:

```
scale_cols = weight_cols / block_size
```

| Strategy | block_size | scale_cols (cols=576) |
|----------|-----------|----------------------|
| Channelwise | cols (576) | 1 |
| Block[64] | 64 | 9 |
| Block[32] | 32 | 18 |
| Per-element | 1 | 576 |

Channelwise quantization is not a separate concept. It is blockwise quantization with `block_size = weight_cols`. This means a single parameterized description covers all cases.

Scale dtype is typically float32 (for int8) or bfloat16/float16 (for Q4). It is independent of the weight dtype.

### 2.4 Q4 Packing

Q4 stores two 4-bit values per byte. This creates a packing ratio:

```
packed_cols = logical_cols / 2
```

The stored buffer has shape `(rows, packed_cols)` while the logical weight has shape `(rows, logical_cols)`. This breaks any dimensional validation that assumes `stored_cols == logical_cols`. It is an irreducible relationship — you cannot make it disappear through abstraction.

### 2.5 Block Divisibility Constraint

For blockwise quantization:

```
block_size must divide (cols / tp)
```

This is a 4-way coupling between block_size, cols, tp, and the quantization strategy. There is no way to check this per-variable — it involves all four simultaneously. It must be validated before loading begins, not at load time.

### 2.6 Concrete Memory Footprints (q_proj, 576x576)

| Strategy | Buffers | Total Bytes | vs BF16 |
|----------|---------|-------------|---------|
| BF16 | 1 | 663,552 | 1.00x |
| Int8 channelwise | 2 | 334,080 | 0.50x |
| Int8 chan + bias | 3 | 336,384 | 0.51x |
| Int8 block[32] | 2 | 373,248 | 0.56x |
| Q4 block[32] | 3 | 207,360 | 0.31x |

---

## 3. Tensor Parallelism and Sharding

### 3.1 Per-Dimension Nature

Sharding is a per-dimension operation. Each dimension of a 2D tensor is independently either divided by tp or kept whole. This gives 2^2 = 4 combinations, of which 3 are useful:

| Strategy | Rows | Cols | Use Case |
|----------|------|------|----------|
| Row-shard | divide | keep | Output projection, attention projections |
| Col-shard | keep | divide | Input projection when output needs allreduce |
| Replicated | keep | keep | Shared/broadcast tensors, embeddings |
| Both-divide | divide | divide | Theoretically valid, rarely used |

### 3.2 Which Tensors Get Which Strategy

In the LLaMA architecture:

| Tensor | Strategy | Reason |
|--------|----------|--------|
| q_proj, k_proj, v_proj | Row-shard | Split output heads across TP ranks |
| o_proj | Col-shard | Recombine head outputs, needs allreduce |
| gate_proj, up_proj | Row-shard | Split intermediate dim across ranks |
| down_proj | Col-shard | Recombine from intermediate, needs allreduce |
| embed_tokens | Replicated or row-shard | Depends on implementation |
| layernorm weights | Replicated | 1D, small, same on all ranks |

Col-sharding implies an allreduce after the matmul. Row-sharding does not.

### 3.3 IO Implications

- **Row-sharded**: local data is contiguous in the file. One read per buffer.
- **Col-sharded**: local data is strided. Each row is a separate read of `local_cols * elem_bytes` bytes, separated by `global_cols * elem_bytes` stride. This means `local_rows` separate reads.
- **Replicated**: contiguous. One read per buffer. All ranks read the same data.

### 3.4 Scale Strategy Independence

The scale buffer's shard strategy may differ from the weight buffer's. Specifically:

- Col-sharded weight with channelwise quantization: `scale_cols = 1`. Dividing 1 by `tp = 2` gives 0.
- The scale must be replicated, even though the weight is col-sharded.

This means weight strategy and scale strategy are separate parameters, not one parameter that applies uniformly. Blockwise quantization with `block_size < cols` does not have this problem because `scale_cols` is large enough to divide.

### 3.5 Placement Is Orthogonal

Where a buffer materializes (which nodes have it) is independent of how it's shaped.

Two cases observed:
- **Host-only**: buffer exists only on the host node (e.g., embedding table for CPU-only lookup)
- **All nodes**: buffer exists on every TP rank (replicated) or each rank has its shard (sharded)

Placement and shape were previously conflated in integer encodings (`SHARD_AXIS = -1` for replicated, `-2` for host-only). They produce identical shapes but different materialization behavior.

### 3.6 Concrete Sharded Shapes (q_proj, 576x576, tp=2)

| Strategy | Local Shape | Local Bytes (BF16) | Fragments |
|----------|------------|-------------------|-----------|
| Row-shard | 288 x 576 | 331,776 | 1 |
| Col-shard | 576 x 288 | 331,776 | 576 |
| Replicated | 576 x 576 | 663,552 | 1 |

Same total bytes for row/col shard. Fragment count differs dramatically.

---

## 4. The Loading Problem

### 4.1 What's Needed Per Buffer

To load one buffer from a safetensors file, you need:
1. **Shape**: rows, cols (after sharding)
2. **Byte width**: element_bytes (from encoding)
3. **File offset**: byte position in the checkpoint file
4. **Shard strategy**: determines contiguous vs strided read
5. **TP rank**: which shard to read (for sharded buffers)

That's it. The loader does not need to know:
- Whether this buffer is a weight, scale, bias, or zero-point
- Which quantization strategy produced it
- What other buffers exist
- What computation will consume it
- Which layer it belongs to

### 4.2 The Loading Algorithm

```
local_rows = strategy.shard_rows(global_rows, tp)
local_cols = strategy.shard_cols(global_cols, tp)
byte_count = local_rows * local_cols * elem_bytes

if col-sharded:
    for each row:
        read(local_cols * elem_bytes) from (offset + row * global_cols * elem_bytes + rank * local_cols * elem_bytes)
else:
    read(byte_count) from (offset + rank * byte_count)

arena_bytes = align_up(byte_count, alignment)
```

This is flat arithmetic plus one bounded dispatch (contiguous vs strided). The dispatch is determined by the strategy, not computed from the data.

### 4.3 Buffer Independence

Every buffer's byte count is computed from its own properties alone. No buffer's size depends on any other buffer's size. Cross-buffer aggregation is limited to:
- **Total arena bytes**: sum of aligned sizes
- **Arena offsets**: prefix sum of aligned sizes (inherent to sequential allocation)
- **Total IO operations**: sum of fragment counts

### 4.4 Alignment

Arena placement typically aligns each buffer to a power-of-two boundary (e.g., 64 bytes). The aligned size is:

```
aligned = (raw_bytes + alignment - 1) // alignment * alignment
```

This is a post-hoc rounding operation. It does not affect the data, only the placement.

---

## 5. The Execution Problem

### 5.1 Valid Matmul Dtype Combinations

The set of valid `(A_dtype, B_dtype) -> accumulator` combinations is a finite lookup table determined by hardware, not a product space.

| A | B | Accumulator | Valid on AVX2? |
|---|---|-------------|---------------|
| bf16 | bf16 | f32 | Yes (via cast-and-fma) |
| f16 | f16 | f32 | Yes (via cast-and-fma) |
| f32 | f32 | f32 | Yes (native) |
| i8 | i8 | i32 | Yes (VNNI if available) |
| i8 | bf16 | ??? | No |
| bf16 | i8 | ??? | No |
| i4 | i4 | i32 | Requires specific support |

Mixed-precision combinations that require dequantization (e.g., i8 weight * bf16 activation) are implemented as: dequantize i8 → f32, then f32 * f32. The dequantization is part of the kernel, not the type system.

### 5.2 Five Structural Assumptions for Decomposable Kernels

Per-element operations (load one element, dequantize it, multiply, accumulate) can be factored out of a kernel when ALL of these hold:

| Assumption | Description |
|------------|-------------|
| Standard layout | Buffer is row-major, elements at predictable offsets |
| Sequential access | Elements accessed in index order (no reordering) |
| Separated accumulation | Accumulator lifecycle is simple zero-then-add |
| No cross-element fusion | Element [n,k] processing doesn't share state with [n,k+1] |
| ISA-independent loop | Same loop structure regardless of SIMD width |

Standard dtypes (bf16, f16, f32) satisfy all five. The only thing that varies between them is the element-level cast function.

### 5.3 When the Assumptions Break

Optimized packed/quantized kernels violate these assumptions. Five coupling axes bind their internals:

1. **Load <-> Tiling**: Buffer data is in a kernel-specific repacked layout co-designed with the tiling strategy. A `load(row, col)` interface assumes standard layout, but the optimization IS the non-standard layout.

2. **Dequant <-> Compute**: Mask, shift, and multiply operations are interleaved with accumulation (`_mm512_dpbusd_epi32`), not executed sequentially. Separating them reintroduces the latency that fusing eliminates.

3. **Accumulator <-> K-blocking**: Whether to zero-initialize or load-and-continue depends on position within the reduction dimension. This is part of the tiling, not separable from it.

4. **ISA <-> Loop structure**: AVX uses explicit `set1 + dpbusd` loops. AMX uses `load_a, load_b, run_tile` — a fundamentally different programming model. The loop structure IS the ISA choice.

5. **Data layout <-> Kernel identity**: The repacked buffer format IS the kernel's access pattern. They are co-designed.

These couplings are not incidental. The performance gain comes FROM the coupling. An int4 kernel that fuses nibble extraction with dot product is fast precisely because it doesn't go through a generic element-load interface.

### 5.4 Three Levels of Kernel Internals

Not all parts of a kernel have the same composition properties:

**Level 1 — Always independent of kernel body:**
- Buffer sizing (`rows * cols * elem_bytes`)
- Dimensional contracts (`A.cols == W.cols` for matmul compatibility)
- Role identification (which buffer is weight, which is scale)
- Dispatch routing (which kernel to call)
- Allreduce decisions (col-shard implies allreduce)

These are properties of the WHAT, not the HOW. They hold for any kernel — naive loop, SIMD-optimized, hand-tuned intrinsics.

**Level 2 — Independent when structural assumptions hold:**
- Per-element load + dequantize
- Per-element output cast + store
- Per-element activation load

For standard dtypes, these are genuinely independent operations. Factoring them out gives `E + K` implementations instead of `E * K` (where E = encoding count, K = kernel shape count).

| Scenario | Without factoring | With factoring | Reduction |
|----------|------------------|----------------|-----------|
| 4 enc x 3 shapes | 12 | 7 | 42% |
| 4 enc x 3 shapes x 2 writers | 24 | 9 | 63% |
| 6 enc x 5 shapes x 2 writers | 60 | 13 | 78% |

**Level 3 — Never independent:**
- The fused kernel body: tiling + intrinsics + pipeline + buffer layout
- Selected as a monolithic unit. The type system's role is choosing WHICH kernel, not composing it from parts.

### 5.5 The Execution Boundary

Everything upstream of execution (shape derivation, byte counting, sharding, placement) is flat arithmetic over independent values. At the execution boundary, the problem transitions from "compute the buffer description" to "use the buffer contents." This is where hardware-specific constraints enter and composition becomes partial.

### 5.6 Dimensional Contracts

For a standard linear layer:
```
contraction: activation.cols == weight.cols       (K dimension)
output_cols: output.cols == weight.rows            (N dimension)
batch:       output.rows == activation.rows        (M dimension)
```

For quantized with scale:
```
scale_topology: weight.cols == scale.cols * block_size
```

For Q4 packed:
```
packed: stored.cols == logical.cols / pack_ratio
```

The Q4 constraint breaks the standard dimensional contract. `activation.cols != stored_weight.cols` — the packing ratio is an additional relationship that must be tracked.

---

## 6. Mojo's Type System

### 6.1 What Traits Express

Traits in Mojo can carry:
- **Comptime members**: `comptime DTYPE: DType` — compile-time constants
- **Static methods**: `@staticmethod fn f() -> T` — compile-time callable functions
- **Default implementations**: methods with bodies that conformers inherit
- **Required conformance**: subtrait relationships via `trait A(B):`

A type conforms to a trait by providing all required comptime members and methods. Conformance is structural — if it has the right members, it conforms.

### 6.2 Trait Composition

`fn f[T: A & B]()` requires T to satisfy both A and B. Inside the function, all members of both A and B are visible. This is the primary mechanism for combining concerns.

Trait composition is a product/intersection: `A & B` gives access to the union of A's and B's members. This is the meet in the trait lattice — the most specific bound that includes both.

### 6.3 Comptime Resolution

Static methods on trait-bound type parameters resolve at compile time:

```mojo
struct Foo[S: SomeStrategy, rows: Int]:
    comptime RESULT = Self.S.compute(Self.rows)
```

`Self.S.compute(Self.rows)` is not a runtime call. The concrete type bound to `S` determines which implementation of `compute` is called, and the result is a compile-time constant. This means type parameters can carry computation that produces comptime values in the containing struct.

### 6.4 Parameterized Comptime

Comptime aliases can be parameterized to create type families or value families:

```mojo
# Type family: produces a type from parameters
comptime WeightSlot[E: Encoding, dim: Int] = SomeStruct[E, dim, dim]

# Value family: produces a comptime Int from parameters
comptime AlignedBytes[E: Encoding, r: Int, c: Int, align: Int = 64]: Int =
    ((r * c * E.ELEMENT_BYTES + align - 1) // align) * align
```

Type families are used with `[params]` in type position (no parens). Value families are used with `[params]` in value position (also no parens — they are not callable).

### 6.5 Hard Limits

**No trait specialization.** If `fn f[E: Encoding]()` and `fn f[E: Encoding & HasScale]()` both exist, and a type satisfies both, calling `f` is an ambiguous call error. Mojo does not resolve overloads by trait specificity. You must use different function names.

**No coproducts.** There is no `T: A | B` (either-or) bound. If you need to handle "weight or scale" generically, you need a common supertrait or a tagged comptime member, not a union type.

**No dependent traits.** You cannot require `constrained[T.COLS % T.BLOCK_SIZE == 0]` as part of a trait definition. Divisibility constraints must be checked at the use site with `constrained[]`.

**No negative bounds.** There is no `T: Encoding & !HasScale` to match types that lack a trait. If you need different behavior for types with and without a trait, use different function names.

**No higher-kinded types.** You cannot abstract over type constructors themselves (e.g., "any type that takes an Encoding and produces a Shaped"). Type constructors are used by convention, not by abstraction.

**trait_downcast works on values, not types.** `trait_downcast[TargetTrait](some_value)` downcasts a runtime value. It cannot be used for compile-time type-level dispatch. For type-level conditional behavior, use trait-bounded free functions.

**conforms_to is runtime.** `conforms_to[SomeTrait](some_value)` returns a runtime Bool. It's useful for runtime branching but not for compile-time specialization.

### 6.6 Patterns That Work

**Trait defaults reduce boilerplate.** A trait can provide default method implementations using `Self.MEMBER` — conformers only need to provide the varying comptime members. A conformer can be as short as 2 lines.

```mojo
trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int
    @staticmethod
    fn byte_count(rows: Int, cols: Int) -> Int:
        return rows * cols * Self.ELEMENT_BYTES

struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2
    # byte_count inherited
```

**Trait-bounded free functions for conditional behavior.** Instead of overloading by specificity, use distinct function names with different trait bounds:

```mojo
fn format_dense[E: Encoding]() -> String: ...
fn format_quant[E: Encoding & HasScale]() -> String: ...
```

A type that lacks `HasScale` simply cannot call `format_quant` — compile error.

**Types that carry computation.** A struct with only static methods and no fields can serve as a "strategy object" that is purely a type-level computation carrier. The type parameter selects which formula to use; no runtime value is ever created.

**Self. prefix for struct parameters.** All references to a struct's type parameters inside the struct body must use `Self.ParamName`.

**@fieldwise_init, not @value.** Mojo's memberwise initializer decorator is `@fieldwise_init`.

---

## 7. Observed Composition Properties

### 7.1 The Carrier/Actionable Spectrum

Traits observed in this problem occupy a spectrum:

| Category | Data | Computation | Example |
|----------|------|-------------|---------|
| Pure carrier | Yes | None | `Encoding` (DTYPE, ELEMENT_BYTES) |
| Pure actionable | None | Yes | `DimStrategy` (local() method only) |
| Hybrid | Yes | Derived from data | `ShardStrategy` (D0, D1 data → shard_rows/cols computation) |

Pure carriers compose freely via `&` — they are just data dimensions. Pure actionable traits compose as function composition. Hybrids compose when the derivation is flat (no coupling between the data members).

### 7.2 Where Independence Holds

| Domain | Independent Quantity | Independence Means |
|--------|---------------------|-------------------|
| Loading | Each buffer's byte count | No buffer's size depends on another's |
| Sharding | Each dimension's strategy | Row-shard and col-shard are per-dimension |
| Encoding | DTYPE → ELEMENT_BYTES | 1:1 lookup, no interaction |
| Placement | Where vs how | Shape is independent of materialization location |
| Level 1 kernel | Dimensional contracts | Hold regardless of kernel optimization level |

### 7.3 Where Coupling Is Irreducible

| Coupling | What's Coupled | Why It Can't Decompose |
|----------|---------------|----------------------|
| Block divisibility | block_size, cols, tp, quant strategy | 4-way interaction |
| Scale strategy | weight shard strategy, scale_cols | Col-shard + channelwise → replicated scale |
| Q4 packing | stored_cols, logical_cols, pack_ratio | Dimensional relationship |
| Fused kernels | load, dequant, tiling, ISA, layout | Performance comes from the fusion |
| Valid matmul dtypes | A_dtype, B_dtype, accumulator, ISA | Hardware-determined finite set |

### 7.4 The Compositionality Gradient

```
FULLY INDEPENDENT ─────────────────────────────── FULLY COUPLED

Buffer sizing    Placement    Contracts    Per-elem ops    Fused kernels
     |               |            |             |               |
  no inter-       orthog. to   hold for      hold when      five coupling
  action          shape        all kernels   5 assumptions   axes bind all
  between                                   are met         internals
  buffers
```

This is a gradient, not a wall. The boundary between "independent" and "coupled" is determined by structural properties of the data and the kernel, not by an architectural choice.

### 7.5 Encoding → Buffer Decomposition Is Finite

The number of encoding families is small and known at build time. Each family produces a fixed buffer count (1, 2, or 3). This decomposition is not compositional — you cannot derive "int8 channelwise produces 2 buffers" from first principles. But it is a small, closed set that changes rarely.

Currently relevant families:
- Dense: 1 buffer
- Quantized + scale: 2 buffers
- Quantized + scale + aux (bias or zero): 3 buffers

### 7.6 Trait Bounds as Visibility Control

A function with bound `[T: Encoding]` sees DTYPE and ELEMENT_BYTES. A function with `[T: Shaped]` sees ROWS and COLS. A function with `[T: Encoding & Shaped]` sees all four.

This is enforced by the compiler — you cannot accidentally access a member you didn't ask for. The practical consequence: a function that only needs byte width literally cannot be affected by a change to how shapes are computed.

Two types with different construction histories but the same trait conformances produce identical results through any trait-bounded function. The function sees only the interface, not the provenance.

---

## 8. Concrete Numbers: Full Model

### 8.1 Per-Layer Memory (BF16, tp=1)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| q_proj | 576 x 576 | 663,552 |
| k_proj | 192 x 576 | 221,184 |
| v_proj | 192 x 576 | 221,184 |
| o_proj | 576 x 576 | 663,552 |
| gate_proj | 1536 x 576 | 1,769,472 |
| up_proj | 1536 x 576 | 1,769,472 |
| down_proj | 576 x 1536 | 1,769,472 |
| input_layernorm | 576 | 2,304 (f32) |
| post_attn_layernorm | 576 | 2,304 (f32) |
| **Layer total** | | **7,082,496** |

### 8.2 Full Model Memory

| Component | Bytes (BF16) |
|-----------|-------------|
| Embedding | 49152 x 576 x 2 = 56,623,104 |
| 30 layers | 30 x 7,082,496 = 212,474,880 |
| Final norm | 576 x 4 = 2,304 |
| **Total** | **~269 MB** |

(Tied weights: LM head reuses embedding, so no additional memory.)

### 8.3 Full Model Memory by Encoding

| Encoding | Approx Total | vs BF16 |
|----------|-------------|---------|
| BF16 | 269 MB | 1.00x |
| Int8 channelwise | ~141 MB | 0.52x |
| Q4 block[32] | ~92 MB | 0.34x |

(Approximate: layernorm weights always f32, embedding may stay bf16.)

### 8.4 Sharded Memory (tp=2, BF16)

Per-layer sharded weights only (excluding layernorm):

| Tensor | Strategy | Local Shape | Local Bytes |
|--------|----------|------------|-------------|
| q_proj | RowShard | 288 x 576 | 331,776 |
| k_proj | RowShard | 96 x 576 | 110,592 |
| v_proj | RowShard | 96 x 576 | 110,592 |
| o_proj | ColShard | 576 x 288 | 331,776 |
| gate_proj | RowShard | 768 x 576 | 884,736 |
| up_proj | RowShard | 768 x 576 | 884,736 |
| down_proj | ColShard | 576 x 768 | 884,736 |

Layer total (sharded): ~3,538,944 bytes (vs 7,080,192 unsharded = exactly half).

---

## 9. The Target Hardware

### 9.1 CPU: 12th Gen Intel i7-12700KF

- **ISA**: AVX2 (256-bit SIMD). No AVX-512.
- **Cores**: 8P + 4E (12 total, 20 threads with hyperthreading)
- **L1 cache**: 48KB per P-core, 32KB per E-core (data)
- **L2 cache**: 1.25MB per P-core, 2MB shared per E-core cluster
- **L3 cache**: 25MB shared
- **Memory**: DDR4 or DDR5 depending on board

### 9.2 AVX2-Relevant Operations

| Operation | Intrinsic Family | Width |
|-----------|-----------------|-------|
| f32 FMA | `_mm256_fmadd_ps` | 8 floats |
| f32 multiply | `_mm256_mul_ps` | 8 floats |
| i8 -> i32 extend | `_mm256_cvtepi8_epi32` | 8 elements |
| bf16 -> f32 | shift left 16 bits | 8 elements |
| i32 dot product | `_mm256_dpbusd_epi32` (if VNNI) | — |

No native bf16 dot product on AVX2. BF16 matmul is: load bf16 pair → cast each to f32 → f32 FMA.

### 9.3 Memory Bandwidth Considerations

For a 135M parameter model, the entire weight set fits in L3 cache for BF16 (~269 MB > 25 MB, so it doesn't) but individual layers do (~7 MB fits in L3 comfortably). Single-token inference (seq_len=1) is memory-bandwidth-bound: the matmul reads the entire weight matrix for a single row of output.

For batch=1, seq_len=1 (autoregressive decoding):
- Each linear layer reads `rows * cols * elem_bytes` bytes of weights
- And produces `rows` output values
- Arithmetic intensity: `cols` FMAs per `cols * elem_bytes` bytes read = `1 / elem_bytes` ops/byte
- BF16: 0.5 ops/byte. Int8: 1 op/byte. Q4: 2 ops/byte.
- All are firmly memory-bandwidth-bound on any modern CPU.

---

## 10. The Safetensors Format

### 10.1 File Structure

```
[8 bytes: header_size as little-endian u64]
[header_size bytes: JSON header]
[remaining bytes: raw tensor data]
```

The JSON header maps tensor names to `{dtype, shape, data_offsets: [start, end]}`. Offsets are relative to the end of the header.

### 10.2 Properties

- Tensors are stored contiguously in row-major order
- No compression — raw bytes
- Multiple tensors packed sequentially in the data section
- Offset arithmetic: `absolute_offset = 8 + header_size + data_offsets[0]`
- dtype strings: `"BF16"`, `"F16"`, `"F32"`, `"I8"`, `"U8"`, etc.

### 10.3 For Sharded Loading

Row-sharded: contiguous sub-range. `start + rank * shard_bytes` to `start + (rank+1) * shard_bytes`.

Col-sharded: strided. For each row, read `local_cols * elem_bytes` starting at `row_start + rank * local_cols * elem_bytes`.

The safetensors format makes row-sharding cheap (one read) and col-sharding expensive (one read per row). This is a property of row-major storage, not of the format specifically.

---

## 11. Activation Memory and Intermediate Buffers

### 11.1 Per-Token Activation Sizes

For batch=1, seq_len=1 (single token decode):

| Buffer | Shape | Bytes (f32) |
|--------|-------|-------------|
| Input embedding lookup | [1, 576] | 2,304 |
| Post-RMSNorm | [1, 576] | 2,304 |
| Q projection output | [1, 576] | 2,304 |
| K projection output | [1, 192] | 768 |
| V projection output | [1, 192] | 768 |
| Attention output | [1, 576] | 2,304 |
| Post-attn residual | [1, 576] | 2,304 |
| Gate projection output | [1, 1536] | 6,144 |
| Up projection output | [1, 1536] | 6,144 |
| SiLU(gate) * up | [1, 1536] | 6,144 |
| Down projection output | [1, 576] | 2,304 |
| Logits | [1, 49152] | 196,608 |

Activation memory is negligible compared to weight memory. The logits buffer (196 KB) is the largest single activation.

### 11.2 KV Cache

For autoregressive decoding, each layer stores K and V for all previous positions:

| Per layer | Shape | Bytes per position (f32) |
|-----------|-------|-------------------------|
| K cache | [max_seq, 3, 64] | 768 |
| V cache | [max_seq, 3, 64] | 768 |
| **Per layer total** | | 1,536 per position |

For 30 layers at max context (8192 positions): `30 * 8192 * 1536 = 377 MB` in f32. In bf16: ~189 MB. The KV cache can dominate total memory at long sequences.

---

## 12. RoPE (Rotary Position Embeddings)

### 12.1 The Computation

For each pair of adjacent elements `(x_2i, x_{2i+1})` in a head:

```
cos_θ = cos(position * θ_i)
sin_θ = sin(position * θ_i)

x'_2i   = x_2i * cos_θ - x_{2i+1} * sin_θ
x'_{2i+1} = x_{2i+1} * cos_θ + x_2i * sin_θ
```

Where `θ_i = rope_theta^(-2i/head_dim)` for `i = 0, 1, ..., head_dim/2 - 1`.

### 12.2 Properties

- Applied to Q and K, not V
- head_dim = 64 → 32 rotation pairs per head
- The cos/sin table can be precomputed for all positions up to max_seq_len
- Table shape: `[max_seq_len, head_dim/2]` = `[8192, 32]`
- Table bytes: `8192 * 32 * 4 * 2 = 2,097,152` bytes (2 MB for both cos and sin in f32)

---

## 13. RMSNorm

```
RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
```

- Per-element operation after computing the RMS
- Weight shape: [hidden_size] = [576]
- eps = 1e-5
- Applied twice per layer: before attention, before MLP
- No bias term

---

## 14. SwiGLU MLP

```
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

- `silu(x) = x * sigmoid(x)`
- Element-wise multiplication between gate and up branches
- The gate and up projections can be computed independently, then combined
- The down projection depends on the combined result

This means gate_proj and up_proj are embarrassingly parallel, but down_proj must wait for both.

---

## 15. Summary of Relationships

### Independent (no interaction between these):
- Each buffer's byte count and every other buffer's byte count
- Row-dimension strategy and column-dimension strategy
- Encoding (dtype/bytes) and sharding strategy
- Sharding strategy and placement
- Level 1 kernel concerns and Level 3 kernel internals
- Loading and execution

### Derived (one determines the other):
- DType → element_bytes (1:1)
- Encoding family → buffer count (1:1)
- Block_size + weight_cols → scale_cols (formula)
- Shard strategy + global_shape + tp → local_shape (formula)
- Col-shard → requires allreduce (implication)
- Col-shard → strided IO (implication)

### Coupled (must be considered together):
- block_size, cols, tp, quantization → divisibility constraint
- Weight shard strategy + scale_cols → scale shard strategy
- Q4 pack_ratio + logical_cols → stored_cols
- Fused kernel: load + dequant + tiling + ISA + layout (five axes)

### Finite enumeration (small closed sets):
- Encoding families: dense, quant+scale, quant+scale+aux
- Shard strategies: row, col, replicated (3 useful from 2^2)
- Placements: host-only, all-nodes
- Valid matmul dtype combinations: hardware-determined table
