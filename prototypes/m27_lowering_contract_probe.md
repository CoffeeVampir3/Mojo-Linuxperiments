# M27 Lowering Contract Probe

This is a design prototype for cleaning `modeling/minimax_m27_moe_butterquant_tp.mojo`
without preserving the current adapter shape. The useful boundary is not a generic
ButterQuant ABI object. The useful boundary is a model-specific lowering contract:
forward code works with typed M27 concepts, and those concepts lower once into the
pointer form the kernels already need.

## What Must Improve

An abstraction is worth keeping only if it removes at least one repeated decision
from the forward body:

- which arena base applies
- which layer base applies
- which scratch base applies
- which shape and runtime row count applies
- which companion scale and colsum tensor belongs to a packed int8 weight
- which expert stride belongs to a packed expert slab
- which KV cache address belongs to a layer
- where pointer erasure to `MutAnyOrigin` happens

If a proposed type just stores the same pointer list that the call already had, it
is noise.

## Current Ceremony

The repeated pattern is usually:

```mojo
var lb = topo.layers.base(topo.arena.base, layer_idx)
var layer = topo.layers.proto
var sb = topo.arena.scratch_base()

some_dispatch[
    ... shape constants ...
](
    scratch_lease.view[I8, Shape[...]](sb, seq_len),
    layer.some_weight.bound(lb),
    layer.some_colsum.bound(lb),
    layer.some_scale.bound(lb),
    other_lease.view[BF16, Shape[...]](sb, seq_len),
    scale_lease.view[F32, Shape[...]](sb, seq_len),
    pool,
)
```

That is not only long. It makes every call site re-decide the same contracts.
The win should come from moving those decisions into named lowerable concepts.

## Prototype Rule

Forward code should be allowed to create these values:

```mojo
var rank = M27RankLayer[Self.tp](topo, layer_idx)
var attn = M27AttentionScratch[...].bind(..., rank.scratch_base(), seq_len)
var moe = bind_dense_moe_scratch(..., rank.scratch_base())
```

Forward code should not repeatedly call:

```mojo
.bound(lb)
.bound_row(...)
.view[...](sb, seq_len)
.as_ptr[...]
UnsafePointer(...).as_any_origin()
unsafe_from_address=...
```

Those calls are still necessary, but the prototype boundary says they belong to
the model-specific contract methods that lower an operation.

## Rank And Layer Scope

This is the smallest useful shared object. It does not represent an ABI. It
represents the facts every rank-local operation needs and currently recomputes.

```mojo
@fieldwise_init
struct M27RankLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var topo: MiniMaxM27Topology[tp]
    var layer_idx: Int

    @always_inline
    def arena_base(self) -> Int:
        return self.topo.arena.base

    @always_inline
    def scratch_base(self) -> Int:
        return self.topo.arena.scratch_base()

    @always_inline
    def layer_base(self) -> Int:
        return self.topo.layers.base(self.topo.arena.base, self.layer_idx)

    @always_inline
    def layer(self) -> LayerRefs[tp]:
        return self.topo.layers.proto

    @always_inline
    def kv_cache_base(self) -> Int:
        return (
            self.topo.arena.base
            + self.topo.kv_cache_off
            + self.layer_idx * self.topo.kv_cache_stride
        )
```

This removes repeated arena/layer/scratch/cache derivation from each nested
`@parameter` callback. The callback still receives `topo` and `pool`, but it
immediately binds a rank-layer scope and stops recomputing bases.

## Bound Scratch Workspaces

Scratch leases are already close to the right design. The missing piece is that
call sites repeatedly choose the same shape and `seq_len`. A bound workspace
packages those choices without owning the lease.

```mojo
@fieldwise_init
struct M27AttentionScratch[
    qkv_local: Int,
    q_local: Int,
    heads_per_rank: Int,
    kv_per_rank: Int,
    hpg: Int,
    max_seq_len: Int,
    hidden: Int,
    origin: MutOrigin,
](Copyable, ImplicitlyCopyable):
    var qkv: ScratchView[BF16, Shape[max_seq_len, qkv_local], origin]
    var attn_i8: ScratchView[I8, Shape[max_seq_len, hidden], origin]
    var act_scale: ScratchView[F32, Shape[max_seq_len, 1], origin]
    var q_i8: UnsafePointer[Scalar[DType.int8], origin]
    var qi_biases: UnsafePointer[Float32, origin]
    var q_scales: UnsafePointer[Float32, origin]
    var attn_qi: ScratchView[I8, Shape[max_seq_len, q_local], origin]
    var attn_head_sc: ScratchView[F32, Shape[max_seq_len, heads_per_rank], origin]
    var partial: UnsafePointer[Float32, origin]
```

The constructor would be the only place spelling:

```mojo
qkv_lease.view[BF16, Shape[C.MAX_SEQ_LEN, QKV_LOCAL]](sb, seq_len)
attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len)
act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len)
q_i8_lease.as_ptr[Scalar[DType.int8]](sb)
```

The gain is not the struct. The gain is that every later attention phase consumes
the same typed workspace, so the forward body cannot accidentally bind the same
lease with a different shape, row count, or element type.

## Packed Int8 Weight Contracts

The repeated int8 linear pattern has three persistent tensors:

```mojo
packed i8 weight
f32 scale
f32 colsum
```

A model-specific contract should keep those companions together.

```mojo
@fieldwise_init
struct M27I8Linear[
    out_features: Int,
    in_features: Int,
    max_rows: Int,
](Copyable, ImplicitlyCopyable):
    var packed: StaticView[I8, Shape[out_features, in_features]]
    var scale: StaticView[F32, Shape[out_features, 1]]
    var colsum: StaticView[F32, Shape[out_features, 1]]

    @always_inline
    def dispatch[
        P: BurstThreadPool,
        in_origin: MutOrigin,
        out_origin: MutOrigin,
        scale_origin: MutOrigin,
        pool_origin: MutOrigin,
    ](
        self,
        act: ScratchView[I8, Shape[max_rows, in_features], in_origin],
        act_scale: ScratchView[F32, Shape[max_rows, 1], scale_origin],
        out: ScratchView[BF16, Shape[max_rows, out_features], out_origin],
        ref [pool_origin] pool: P,
    ) -> PoolFence[P, pool_origin]:
        return int8_gemv[out_features, in_features](
            act.any(), self.packed, self.colsum, self.scale, out.any(),
            act_scale.any(), pool)
```

This is a stronger abstraction than a raw ABI wrapper because it binds `packed`,
`scale`, and `colsum` as one quantized weight. A call site can no longer pass an
`o_proj` colsum to a `qkv_proj` packed tensor by adjacency accident.

## QKV Projection Before And After

Current shape:

```mojo
return int8_gemv[QKV_LOCAL, C.HIDDEN](
    attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](sb, seq_len),
    layer.attn.qkv_proj.bound(lb),
    layer.attn.qkv_colsum.bound(lb),
    layer.attn.qkv_proj_sc.bound(lb),
    qkv_lease.view[BF16, Shape[C.MAX_SEQ_LEN, QKV_LOCAL]](sb, seq_len),
    act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](sb, seq_len),
    pool)
```

Prototype shape:

```mojo
var rank = M27RankLayer[Self.tp](topo, layer_idx)
var attn = M27AttentionScratch[...].bind(..., rank.scratch_base(), seq_len)
return rank.qkv_projection().dispatch(attn.attn_i8, attn.act_scale, attn.qkv, pool)
```

Facts removed from the call site:

- `QKV_LOCAL`
- `C.HIDDEN`
- `lb`
- `sb`
- `Shape[C.MAX_SEQ_LEN, C.HIDDEN]`
- `Shape[C.MAX_SEQ_LEN, QKV_LOCAL]`
- qkv scale/colsum adjacency

## KV And Q Prep Contracts

The attention prep code repeats two kinds of ceremony:

```mojo
topo.rope.cos.bound_row(topo.arena.base, 0).as_ptr[DType.float32]()
topo.rope.sin.bound_row(topo.arena.base, 0).as_ptr[DType.float32]()
UnsafePointer(to=inv_rms_arr[0]).bitcast[Float32]().as_any_origin()
topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride
```

Those are not ButterQuant abstractions. They are M27 attention lowering facts.

```mojo
@fieldwise_init
struct M27RopeTables(Copyable, ImplicitlyCopyable):
    var cos0: StaticView[F32, Shape[1, ROPE_HALF]]
    var sin0: StaticView[F32, Shape[1, ROPE_HALF]]

    @always_inline
    def cos_ptr(self) -> F32Ptr:
        return self.cos0.as_ptr[DType.float32]()

    @always_inline
    def sin_ptr(self) -> F32Ptr:
        return self.sin0.as_ptr[DType.float32]()


@fieldwise_init
struct M27InvRmsRows[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Float32, origin]

    @always_inline
    def erased(self) -> F32Ptr:
        return self.ptr.as_any_origin()


@fieldwise_init
struct M27KVCacheWindow(Copyable, ImplicitlyCopyable):
    var base: Int
    var start_pos: Int
    var seq_len: Int
```

Prototype call shape:

```mojo
return rank.kv_writer(attn.qkv, inv_rms.k, rank.rope(), rank.kv_cache_window(start_pos, seq_len))
    .dispatch(pool)
```

The benefit is direct: the forward path stops mixing three unrelated address
derivations inside one kernel call.

## Expert Slab Contracts

MoE is where the stride arguments are obviously not adding information. They are
derived from the M27 expert layout:

```mojo
C.MOE_INTERMEDIATE * C.HIDDEN
C.MOE_INTERMEDIATE * 4
C.HIDDEN * C.MOE_INTERMEDIATE
C.HIDDEN * 4
C.HIDDEN * MOE_DOWN_NUM_BLK * 4
```

Those should be methods or comptime members of expert slab contracts.

```mojo
@fieldwise_init
struct M27GateUpExpertSlab[
    num_experts: Int,
    experts_local: Int,
    intermediate: Int,
    hidden: Int,
](Copyable, ImplicitlyCopyable):
    var w1: StaticView[I8, Shape[num_experts * intermediate, hidden]]
    var w1_scale: StaticView[F32, Shape[experts_local * intermediate, 1]]
    var w1_colsum: StaticView[F32, Shape[experts_local * intermediate, 1]]
    var w3: StaticView[I8, Shape[num_experts * intermediate, hidden]]
    var w3_scale: StaticView[F32, Shape[experts_local * intermediate, 1]]
    var w3_colsum: StaticView[F32, Shape[experts_local * intermediate, 1]]

    comptime weight_stride = intermediate * hidden
    comptime aux_stride = intermediate * F32.ELEMENT_BYTES
```

```mojo
@fieldwise_init
struct M27DownExpertSlab[
    num_experts: Int,
    experts_local: Int,
    hidden: Int,
    intermediate: Int,
    down_blocks: Int,
](Copyable, ImplicitlyCopyable):
    var w2: StaticView[I8, Shape[num_experts * hidden, intermediate]]
    var w2_scale: StaticView[F32, Shape[experts_local * hidden, 1]]
    var w2_colsum: StaticView[F32, Shape[experts_local * hidden * down_blocks, 1]]

    comptime weight_stride = hidden * intermediate
    comptime scale_stride = hidden * F32.ELEMENT_BYTES
    comptime colsum_stride = hidden * down_blocks * F32.ELEMENT_BYTES
```

Current phase1 call:

```mojo
return minimax_moe_phase1[
    C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
    C.TOP_K, C.NUM_EXPERTS, Self.tp](
    moe_i8_lease.view[I8, Shape[1, C.HIDDEN]](sb, 1),
    moe_scale_lease.view[F32, Shape[1, 1]](sb, 1),
    routing,
    layer.body.experts_w1.bound(lb),
    C.MOE_INTERMEDIATE * C.HIDDEN,
    layer.body.experts_w1_sc.bound(lb),
    C.MOE_INTERMEDIATE * 4,
    layer.body.experts_w1_colsum.bound(lb),
    C.MOE_INTERMEDIATE * 4,
    layer.body.experts_w3.bound(lb),
    C.MOE_INTERMEDIATE * C.HIDDEN,
    layer.body.experts_w3_sc.bound(lb),
    C.MOE_INTERMEDIATE * 4,
    layer.body.experts_w3_colsum.bound(lb),
    C.MOE_INTERMEDIATE * 4,
    expert_qi_lease.view[I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](sb, C.TOP_K),
    expert_blk_scale_lease.view[F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](sb, C.TOP_K),
    rank, pool)
```

Prototype phase1 call:

```mojo
return rank.gate_up_experts().dispatch(
    moe.input_i8,
    moe.input_scale,
    routing,
    moe.expert_qi,
    moe.expert_block_scale,
    rank_id,
    pool,
)
```

Facts removed from the call site:

- all six expert stride arguments
- all six `bound(lb)` calls
- two scratch shape spellings
- the implicit pairing of w1/w3 packed, scale, and colsum tensors

This is a real reduction because the phase function no longer accepts stride
parameters that can disagree with its own compile-time layout.

## Lowered Pointer Forms Still Exist

Kernel workers should still receive compact `Copyable, ImplicitlyCopyable`
argument structs with raw pointers. The change is where those structs are
constructed.

```mojo
@fieldwise_init
struct GateUpLowered(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var w1_packed: I8Ptr
    var w1_scale: F32Ptr
    var w1_colsum: F32Ptr
    var w3_packed: I8Ptr
    var w3_scale: F32Ptr
    var w3_colsum: F32Ptr
    var qi_out: I8Ptr
    var block_scale: F32Ptr
```

`GateUpLowered` is not a forward ABI. It is the final lowered kernel payload.
The forward code should never assemble it directly. The expert slab dispatch
method should assemble it from a coherent `M27GateUpExpertSlab`.

## First Code Probe

`prototypes/m27_moe_phase1_lowering_probe.mojo` sketches the smallest
production-facing experiment: MoE phase1 only.

The probe is not sufficient if it lands as a wrapper over the current
stride-bearing function. It is only a call-shape test. If the shape survives,
the production rewrite should change the phase1 dispatch signature to accept the
expert slab contract directly and delete the loose stride arguments.

1. Add `M27RankLayer`, `M27DenseMoeScratch`, and `M27GateUpExpertSlab` in a prototype
   `.mojo` file or a temporary production file.
2. Rewrite only the dense `do_expert_phase1` call through those contracts.
3. Compare the call site before and after.
4. Keep the kernel signature unchanged at first.
5. If the call site only got shorter by hiding a same-shaped parameter list,
   delete the attempt.

MoE phase1 is the right probe because it contains obvious removable facts: expert
strides and bound weight companions. Attention should come second because the
KV/Q prep contract is also useful, but it is entangled with more phases.

## API Probe Results

The Mojo API experiments changed the design in a useful way.

`StaticView` and `ScratchView` should not be stored directly in fieldwise
contract structs. A prototype that did this failed because those view types do
not expose copy/move conformance. The better contract stores copyable model refs,
bases, and safe pointers to leases, then materializes views inside the lowering
method.

`Pointer[T, origin]` works as a safe contract carrier for stack or lease-backed
objects. `prototypes/pointer_contract_api_probe.mojo` validates that a copyable
contract can store a `Pointer` to a non-copyable token and later call methods on
the pointee through the preserved origin.

`prototypes/m27_moe_phase1_lowering_probe.mojo` now uses that pattern:

```mojo
struct M27DenseMoeScratch[
    input_origin: MutOrigin,
    scale_origin: MutOrigin,
    qi_origin: MutOrigin,
    block_scale_origin: MutOrigin,
](Copyable, ImplicitlyCopyable):
    var scratch_base: Int
    var input_i8_lease: Pointer[ScratchLease, Self.input_origin]
    var input_scale_lease: Pointer[ScratchLease, Self.scale_origin]
    var expert_qi_lease: Pointer[ScratchLease, Self.qi_origin]
    var expert_block_scale_lease: Pointer[ScratchLease, Self.block_scale_origin]
```

That keeps scratch lifetime tied to the lease values while still giving the
contract a compact copyable shape.

`Span[T, origin]` works cleanly for stack row buffers. This is a better fit for
`inv_rms_q_arr` and `inv_rms_k_arr` than manual pointer construction:

```mojo
@fieldwise_init
struct F32Rows[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var rows: Span[Float32, Self.origin]

    @always_inline
    def ptr(self) -> F32Ptr:
        return self.rows.unsafe_ptr()
```

`prototypes/span_lowering_api_probe.mojo` validates this. For attention, this
suggests replacing:

```mojo
UnsafePointer(to=inv_rms_k_arr[0]).bitcast[Float32]().as_any_origin()
```

with a named contract:

```mojo
var inv_rms_k = F32Rows(Span(inv_rms_k_arr))
...
inv_rms_k.ptr()
```

`prototypes/m27_attention_lowering_probe.mojo` applies the same idea to the M27
KV-write path. Its lowered call shape is:

```mojo
return kv_write_dispatch[...](
    qkv.base_ptr(),
    rank_layer.k_norm(),
    rank_layer.rope_cos0(),
    rank_layer.rope_sin0(),
    inv_rms_k.ptr(),
    rank_layer.kv_cache_base(),
    start_pos,
    seq_len,
    pool,
)
```

This is still calling the existing kernel wrapper, but the forward-side ceremony
has a better division:

- `M27AttentionLayer` owns layer base, rope row binding, norm binding, and KV
  cache base math.
- `M27QkvScratch` owns qkv lease-to-pointer lowering.
- `M27InvRmsRows` owns stack row-span-to-pointer lowering.

The fully generic associated-type trait form was less useful. A trait that says
"I lower to some pointer type" is too broad for a generic caller to pass the
result into a concrete pointer ABI. A trait that lowers directly to an erased
`MutAnyOrigin` pointer compiles, but that makes the trait itself the erasure
boundary. That can still be useful for tiny leaf APIs, but the stronger pattern
for M27 forward is concrete contracts with named `phase` or `dispatch` methods.

## Rejected Shape

This is the adapter shape that should not come back:

```mojo
struct ButterQuantActivation:
    var data: I8Ptr
    var scale: F32Ptr

struct ButterQuantWeight:
    var packed: I8Ptr
    var scale: F32Ptr
    var colsum: F32Ptr
```

That only renames the pile of pointers. It does not know the M27 layer, scratch
base, runtime row count, expert layout, or companion tensor provenance. It cannot
remove the binding ceremony that is making the forward code noisy.

## Decision Criteria

Keep a contract if it makes an invalid call harder to write and removes repeated
facts from the forward body.

Delete a contract if its constructor receives the same loose values that the
kernel would have received.

The target is not fewer characters. The target is fewer manually repeated facts.
