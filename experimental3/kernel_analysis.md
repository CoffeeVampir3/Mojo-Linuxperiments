# Kernel Decomposition Analysis

## Thesis

The compute kernel is not a monolithic atom. Parts of it decompose into independent, trait-expressible concerns. Other parts are irreducibly fused by optimization pressure. The boundary between these is determined by **structural assumptions** that hold for some kernel classes and not others.

The decomposition has three levels:
1. **Always composable** — buffer sizing, dimensional contracts, dispatch selection, role tagging
2. **Conditionally composable** — per-element load/dequant/store, when data layout is standard and the pipeline is sequential
3. **Never composable** — optimized kernel bodies where load, dequant, tiling, and ISA intrinsics are co-designed

Standard-dtype kernels (bf16, f16, f32) satisfy the structural assumptions for Level 2. Packed/quantized kernels with repacked buffers and fused pipelines operate at Level 3. Both consume the Level 1 infrastructure.

---

## 1. The Actionable Trait Pattern

### 1.1 The Concept

Instead of treating the kernel as indivisible, factor out per-element operations into **actionable traits** — traits whose types carry computation that the kernel calls generically:

```mojo
trait WeightLoader:
    @staticmethod
    fn load(
        weight_ptr: Int, row: Int, col: Int, stride: Int,
        aux0: Int, aux1: Int,
    ) -> Float64: ...
```

Each encoding implements the load+dequant operation:

```mojo
struct BF16Loader(WeightLoader):
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64:
        # bf16 → f32 cast. aux0/aux1 unused.
        return bf16_to_f32(weight_ptr, row * stride + col)

struct I8ChanLoader(WeightLoader):
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64:
        # i8 load, multiply by channelwise scale from aux0.
        return i8_to_f32(weight_ptr, row * stride + col) * f32_load(aux0, row)
```

The kernel becomes generic over the loader:

```mojo
fn generic_gemv[L: WeightLoader, K: Int, N: Int](
    act_ptr: Int, weight_ptr: Int, out_ptr: Int,
    aux0: Int, aux1: Int,
):
    for n in range(N):
        var acc = Float64(0)
        for k in range(K):
            acc += load_act(act_ptr, k) * L.load(weight_ptr, n, k, K, aux0, aux1)
        store_out(out_ptr, n, acc)
```

Mojo monomorphizes `L.load()` at compile time. `generic_gemv[BF16Loader]` produces identical machine code to a hand-written `bf16_gemv`. Zero overhead.

### 1.2 Combinatorial Reduction

Without factoring: `E × K` concrete functions (encodings × kernel shapes).
With factoring: `E + K` items (encodings + kernel shapes).

| Scenario | Without | With | Reduction |
|----------|---------|------|-----------|
| 4 enc × 3 shapes | 12 | 7 | 42% |
| 4 enc × 3 shapes × 2 writers | 24 | 9 | 63% |
| 6 enc × 5 shapes × 2 writers | 60 | 13 | 78% |

The savings are additive, not multiplicative. Each new axis adds O(1), not O(N).

### 1.3 Multiple Actionable Trait Axes

The pattern extends to output writing and activation loading:

```mojo
fn triply_generic_gemv[
    A: ActLoader,       # how to load activations
    L: WeightLoader,    # how to load + dequant weights
    W: OutputWriter,    # how to cast + store outputs
    K: Int, N: Int,
](act_ptr: Int, weight_ptr: Int, out_ptr: Int, aux0: Int, aux1: Int):
    for n in range(N):
        var acc = Float64(0)
        for k in range(K):
            acc += A.load(act_ptr, 0, k, K) * L.load(weight_ptr, n, k, K, aux0, aux1)
        W.store(out_ptr, 0, n, N, acc)
```

Each concern is an independent trait axis. Adding a new output format = 1 new Writer, 0 kernel changes.

### 1.4 Enriched Loaders: Comptime Properties + Methods

The loader can carry BOTH computation and metadata:

```mojo
trait EnrichedLoader:
    @staticmethod
    fn load(weight_ptr: Int, row: Int, col: Int, stride: Int,
            aux0: Int, aux1: Int) -> Float64: ...
    comptime NEEDS_ALLREDUCE: Bool
    comptime ACC_BITS: Int
```

The kernel reads properties at compile time:

```mojo
fn enriched_gemv[L: EnrichedLoader, K: Int, N: Int](...):
    # ... inner loop using L.load() ...
    @parameter
    if L.NEEDS_ALLREDUCE:
        allreduce(out_ptr, N)
```

The loader is both a carrier (metadata) and a producer (computation). The kernel is generic over both.

---

## 2. Where the Pattern Breaks: Optimized Kernels

### 2.1 A Real Kernel

Consider an int4 GEMM kernel (from a production C++ codebase):

```cpp
using BufferA = BufferAWithSumKGroupImpl<GemmKernel224Int4_1_LowKGroup>;
using BufferB = BufferBInt4WithZeroLowKGroupImpl<GemmKernel224Int4_1_LowKGroup>;

static void avx_kernel(int m, int n, int k, int m_begin, int n_begin,
                       int k_block_begin, int32_t* int_c,
                       BufferA* ba, BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4_1_LowKGroup;
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
        // Process lo nibble
        __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin);
        for (int m_i = 0; m_i < m; m_i++)
            for (int k_i = 0; k_i < 16; k_i++) {
                __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                    __m512i b512_lo = _mm512_and_si512(K::lo_mask(), b512[...]);
                    c512[...] = _mm512_dpbusd_epi32(c512[...], b512_lo, ma_lo);
                }
            }
    } else {
        // Process hi nibble — different mask, different shift
        // ... _mm512_srli_epi32(..., 4) ...
    }
}
```

This kernel violates every structural assumption of the actionable trait pattern.

### 2.2 The Five Coupling Axes

**1. Load ↔ Tiling.** `bb->get_submat()` returns data in a kernel-specific repacked layout. `BufferBInt4WithZeroLowKGroupImpl` is co-designed with the kernel's tiling strategy. A `WeightLoader.load(row, col)` assumes standard layout — but the optimized kernel's entire value proposition is that data is NOT in standard layout.

**2. Dequant ↔ Compute fusion.** The lo/hi nibble processing isn't "load → dequant → fma" sequentially. The mask (`K::lo_mask()`), the shift (`_mm512_srli_epi32(..., 4)`), and the `dpbusd` are interleaved into a single fused pipeline. Separating them reintroduces the latency that fusing eliminates.

**3. Accumulator ↔ K-blocking.** `if (k_block_begin % k_group_size == 0) { zero } else { load }` — the accumulator management depends on position within the reduction dimension. This is part of the tiling strategy, not separable from it.

**4. ISA ↔ Loop structure.** The AVX version uses explicit `set1 + dpbusd` loops. The AMX version uses `load_a, load_b_lo, run_tile` — a completely different programming model. The loop structure IS the ISA choice.

**5. Data layout ↔ Kernel identity.** `BufferA`, `BufferB`, `BufferC` are not just "how to read" — they define "how the data was PREPARED" for this kernel's access pattern. The repacking and the kernel are a co-designed unit.

### 2.3 Structural Assumptions for Level 2

The actionable trait pattern holds when ALL of the following are true:

| Assumption | Description | When it holds | When it breaks |
|------------|-------------|---------------|----------------|
| Standard layout | Weight buffer is row-major or universally understood | bf16/f16/f32 weights | Repacked tiled layouts |
| Sequential access | Elements accessed in predictable order | Naive inner loops | Interleaved lo/hi nibble passes |
| Separated accumulation | Accumulator lifecycle managed by harness | Simple dot products | K-blocked reductions with position-dependent init |
| No cross-element fusion | Element [n,k] doesn't share intermediates with [n,k+1] | Independent element ops | Shared packed byte for nibble pairs |
| ISA-independent loop | Same loop structure regardless of hardware | Scalar/simple SIMD | AVX vs AMX fundamentally different |

Standard-dtype kernels (bf16×bf16, f16×f16, f32×f32) satisfy all five. The data is in standard layout, elements are accessed sequentially, there is no dequantization to fuse, and the loop structure is the same across ISAs (only the SIMD width changes).

Quantized kernels with repacked buffers violate most or all of these. The optimization that makes them fast IS the violation — repacking, fusing, and interleaving are the optimizations.

---

## 3. Three Levels of Decomposition

### Level 1: Always Composable

These concerns are independent of kernel optimization level. They describe WHAT, never HOW.

| Concern | Mechanism | Example |
|---------|-----------|---------|
| Buffer sizing | `operand_bytes[T: Encoding & Shaped]()` | `T.ROWS * T.COLS * T.ELEMENT_BYTES` |
| Dimensional contracts | `validate_linear_dims[A, W, O]()` | `constrained[A.COLS == W.COLS]` |
| Scale topology | `validate_scale_dims[W, S]()` | `constrained[W.COLS == S.COLS * S.BLOCK_SIZE]` |
| Dispatch selection | `Group.Loader` type parameter | Skeleton carries which kernel to call |
| Role tagging | `WeightOperand`, `ScaleOperand` traits | Operand descriptors carry identity |
| Allreduce decision | `Loader.NEEDS_ALLREDUCE` | Comptime property on the loader type |
| Shape arithmetic | `output_cols[W]()`, `contraction_dim[W]()` | Pure functions of trait properties |

These are the trait intersection constraints from design_exploration4. They compose regardless of whether the kernel body is a naive loop or a hand-tuned AVX intrinsic sequence.

### Level 2: Conditionally Composable

Per-element operations factored into actionable traits. Valid when structural assumptions hold.

| Concern | Trait | Assumption required |
|---------|-------|---------------------|
| Weight load + dequant | `WeightLoader.load()` | Standard layout, sequential access |
| Output cast + store | `OutputWriter.store()` | Standard output layout |
| Activation load | `ActLoader.load()` | Standard input layout |

**Where this works well:**
- Reference/correctness kernels for any encoding
- Production kernels for standard dtypes (bf16, f16, f32) where data is not repacked
- Prototyping new encodings before hand-optimizing

**Where this breaks:**
- Int4 kernels with repacked buffers and fused nibble processing
- AMX tile operations (fundamentally different programming model)
- Any kernel where the buffer layout is co-designed with the tiling

### Level 3: Never Composable

The optimized kernel body, its buffer layout, its tiling, and its ISA intrinsics form an indivisible unit.

```
GemmKernel224Int4_1_LowKGroup
    ├── BufferA layout (co-designed)
    ├── BufferB layout (co-designed)
    ├── AVX inner loop (fused mask + shift + dpbusd)
    ├── AMX inner loop (tile load + run_tile)
    ├── K-block accumulator management
    └── Lo/Hi nibble processing order
```

These are selected as a unit by the dispatch mechanism from Level 1. The type system's role here is choosing WHICH monolithic kernel to call, not composing the kernel from parts.

---

## 4. The Topology Is Consistent

The decomposition boundary follows the same principle across all layers of the system:

| Layer | Independent (composable) | Coupled (atomic) |
|-------|-------------------------|-------------------|
| Config | Per-slot traits (Encoding, Strategy, Placement) | Skeleton structure (which slots exist) |
| Loader | Per-atom `plan_atom()` | Cross-atom aggregation (prefix sums) |
| Kernel | Per-element load/dequant (when layout is standard) | Loop structure, tiling, pipeline fusion |

The pattern is: **independent contributions compose, coupled structures don't.**

- In the config layer, each slot's properties are independent of other slots → composable.
- In the loader, each atom's byte count is independent of other atoms → composable.
- In the kernel, each element's dequantized value is independent of other elements → composable (when the structural assumptions hold).

When the kernel fuses element-level operations with the loop structure (repacked buffers, interleaved nibble processing, position-dependent accumulation), the per-element independence is lost, and the kernel becomes atomic.

---

## 5. The Encoding Group as Bridge

The skeleton's encoding group (`DenseGroup`, `QuantGroupV2`, `TripleGroup`) bridges configuration, loading, and execution:

```mojo
struct QuantGroupV2[WeightEnc, ScaleEnc, S, ScaleS, P, rows, cols, block_size, ...]:
    # For the LOADER: flat atom descriptors
    comptime Weight = Slot[WeightEnc, S, P, rows, cols, suffix, tp]
    comptime Scale  = Slot[ScaleEnc, ScaleS, P, rows, SCALE_COLS, scale_suffix, tp]

    # For the EXECUTOR: role-tagged operand descriptors
    comptime WeightOp = WeightDesc[WeightEnc, Weight.ROWS, Weight.COLS]
    comptime ScaleOp  = ScaleDesc[ScaleEnc, Scale.ROWS, Scale.COLS, block_size]

    # For the KERNEL DISPATCH: actionable trait (Level 2) or kernel identity (Level 3)
    comptime Loader = I8ChanLoader
```

Three views of the same data:
- **Loader view**: flat atoms, each satisfying `Encoding & Shaped`
- **Executor view**: role-tagged descriptors via trait intersection (`WeightOperand & Encoding & Shaped`)
- **Kernel dispatch**: either a Level 2 actionable trait or a Level 3 kernel identity

The forward pass reads from the group:

```mojo
# Level 2 (actionable trait — reference kernel):
generic_gemv[Skel.QProj.Loader, K, N](ptrs...)

# Level 3 (monolithic dispatch — optimized kernel):
optimized_int4_gemm(ptrs..., Skel.QProj.Weight.ROWS, Skel.QProj.Weight.COLS)
```

Both paths consume the same skeleton. The choice between Level 2 and Level 3 is a performance/development tradeoff, not an architectural one.

---

## 6. What Standard Dtypes Get Right

The actionable trait pattern works particularly well for standard dtypes (bf16, f16, f32) because they satisfy all structural assumptions:

| Assumption | bf16/f16/f32 | int4 repacked |
|------------|-------------|---------------|
| Standard layout | Row-major ✓ | Kernel-specific ✗ |
| Sequential access | Left-to-right ✓ | Lo/hi nibble passes ✗ |
| Separated accumulation | Zero-then-add ✓ | Position-dependent ✗ |
| No cross-element fusion | Independent casts ✓ | Shared packed byte ✗ |
| ISA-independent loop | Same loop, different width ✓ | AVX vs AMX ✗ |

For bf16 and f16, the only difference is the cast: `bf16_to_f32` vs `f16_to_f32`. A `WeightLoader` that carries this cast generalizes perfectly. This is exactly what CUTLASS/CuTe do (with macros rather than traits), and it works because the underlying operations are genuinely independent.

The lesson: actionable traits are not speculative. They are a real, proven decomposition for the class of kernels where per-element operations are independent. The question is always: does this specific kernel satisfy the structural assumptions?

---

## 7. Practical Architecture

### When to use Level 2 (actionable traits):

- Standard dtype matmul (bf16, f16, f32) — the structural assumptions hold
- Reference/correctness implementations for any encoding — correctness first, optimize later
- Prototyping new quantization schemes — fast iteration, defer optimization
- Simple quantized inference where data is in standard layout (channelwise dequant without repacking)

### When to use Level 3 (monolithic kernels):

- Int4/Q4 kernels with repacked buffers
- AMX tile operations
- Any kernel where the buffer layout is co-designed with the tiling
- Any kernel where dequantization is fused into the compute pipeline
- Peak-performance production kernels

### The architecture supports both:

```
Skeleton → Group
              ├── .Weight, .Scale (Level 1: always available)
              ├── .WeightOp, .ScaleOp (Level 1: role-tagged descriptors)
              ├── .Loader (Level 2: actionable trait, when applicable)
              └── kernel identity (Level 3: monolithic dispatch)
```

Level 1 is the foundation. Level 2 and Level 3 are alternative implementations of the same mathematical operation, selected based on performance requirements and structural constraints.

---

## 8. Summary

The kernel is not a binary: fully composable or fully atomic. It has internal structure.

**Per-element operations** (load, dequant, cast, store) are independent when the data layout is standard and the pipeline is sequential. These factor into actionable traits. The combinatorial reduction is real: `E × K` → `E + K`.

**Loop structure** (tiling, work partition, SIMD width, prefetch, pipeline fusion) is coupled. It stays in the kernel. When the loop structure absorbs per-element operations (repacked buffers, fused dequant, interleaved nibble processing), the kernel becomes atomic.

**The structural question** for any specific kernel: does the per-element operation depend on the loop structure? If no → Level 2 (actionable traits). If yes → Level 3 (monolithic).

Standard dtypes say no — they always decompose. Packed/repacked quantized formats say yes — they are monolithic atoms. The design accommodates both through the same skeleton infrastructure, choosing the right level per kernel.
