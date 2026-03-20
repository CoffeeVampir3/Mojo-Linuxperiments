# experimental3

Static deployment prototype for CPU-only NUMA loading/execution.

`experimental3` is Linux-only and requires `io_uring`. There is no fallback
loader path, and there is no runtime checkpoint discovery.

This version treats the deployment target as part of the model definition:

- `smollm2_tp1.mojo`
  - Defines a fully static `TP=1` SmolLM2 descriptor.
  - Owns:
    - tensor slots
    - tensor names
    - tensor shapes
    - placement / shard-axis
    - node topology
    - runtime-state layout (`kv_cache + workspace`)
  - Binds a loaded package into per-node typed weights plus per-node runtime state.
  - Executes a static decode without any string lookup in the executor.

- `core.mojo`
  - Generic static descriptor contract.
  - Shared helpers for shapes, alignment, and validation.

- `tensor_contracts.mojo`
  - Shared trait-based tensor contracts for formats, layouts, distributions, and
    bound tensors.
  - Keeps semantic tensor typology out of the executor and descriptor call
    sites.

- `package_manifest.mojo`
  - Parses the static package manifest consumed by the loader.
  - Carries only disk layout: checkpoint file paths and per-slot file offsets in
    descriptor order.
  - Model structure is intentionally not duplicated here.

- `loader.mojo`
  - Generic static distributor + loader.
  - The descriptor supplies the exact tensor table, topology, and state sizing.
  - The package manifest supplies only the static disk offsets for those slots.
  - Replicated weights are materialized once per node.
  - Produces slot-indexed `LoadedTensorRecord`s for binding.

- `executor_ops.mojo`
  - Node-aware executor operation types and tracing stubs.

The key change from `experimental2` is that `experimental3` no longer parses
`config.json` or the safetensors header at runtime. The executor never rebuilds
model structure from tensor names; it consumes a bound model assembled once from
the descriptor.

## Current specialization

Only `TP=1` is implemented here:

- one NUMA node
- replicated weights loaded once on that node
- column/row-sharded slots still keep their placement tags, but degenerate to a
  single shard

The generic loader core assumes `TP == node_count` for CPU NUMA sharding. A
future `TP=4` target should be added as a separate descriptor/executor
specialization, reusing the same static loader core.

## Run

```bash
pixi run mojo run -I . experimental3/smollm2_tp1.mojo
```

This entrypoint loads the package through `io_uring`, binds the static model
view, and runs the bound executor. If `io_uring` is unavailable, startup fails.

## Regenerate Manifest

```bash
pixi run mojo run -I . experimental3/generate_smollm2_tp1_manifest.mojo
```

This is an offline packaging step. Runtime loading consumes only the generated
package manifest.
