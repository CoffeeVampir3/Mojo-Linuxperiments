from collections import Dict
from memory import UnsafePointer, alloc
from pathlib import Path

from numa import NumaArena
from safetensors.loader import IoLoader, ReadOp, Completion, print_io_load_error

from experimental3.core import (
    StaticModelDescriptor,
    align_up_int,
    shape_num_bytes,
    dtype_bytes,
    AXIS_NONE,
    AXIS_HOST,
)
from experimental3.package_manifest import (
    StaticPackageManifest,
    parse_static_package_manifest,
)


comptime DEFAULT_ALIGNMENT = 64
comptime DEFAULT_IO_QUEUE_DEPTH = 2048


@fieldwise_init
struct ReadFragmentPlan(Copyable, Writable):
    var file_offset: Int
    var shard_offset: Int
    var length_bytes: Int


@fieldwise_init
struct ShardGeometry(Movable):
    var local_shape: List[Int]
    var global_offset: List[Int]
    var num_bytes: Int
    var fragments: List[ReadFragmentPlan]


@fieldwise_init
struct ShardLoadPlan(Copyable, Writable):
    var slot_id: UInt16
    var slot_name: String
    var layer_idx: Int
    var tensor_name: String
    var file_idx: Int
    var dtype: DType
    var shard_axis: Int
    var global_shape: List[Int]
    var node_index: Int
    var node_id: Int
    var shard_index: Int
    var shard_count: Int
    var local_shape: List[Int]
    var global_offset: List[Int]
    var num_bytes: Int
    var arena_offset: Int
    var fragments: List[ReadFragmentPlan]


@fieldwise_init
struct NodeArenaPlan(Copyable, Writable):
    var node_id: Int
    var weight_bytes: Int
    var state_bytes: Int
    var total_bytes: Int


@fieldwise_init
struct StaticLoadPlan(Movable):
    var node_ids: List[Int]
    var nodes: List[NodeArenaPlan]
    var shards: List[ShardLoadPlan]
    var alignment: Int
    var total_weight_bytes: Int
    var total_state_bytes: Int
    var total_bytes: Int


@fieldwise_init
struct TensorShardView(Copyable, Writable):
    var arena_id: Int
    var node_id: Int
    var ptr: Int
    var num_bytes: Int
    var local_shape: List[Int]
    var global_offset: List[Int]
    var shard_index: Int
    var shard_count: Int


@fieldwise_init
struct LoadedTensorRecord(Copyable, Writable):
    var slot_id: UInt16
    var slot_name: String
    var layer_idx: Int
    var tensor_name: String
    var dtype: DType
    var shard_axis: Int
    var shape: List[Int]
    var shards: List[TensorShardView]


fn make_slot_key(layer_idx: Int, slot_id: UInt16) -> Int:
    return (layer_idx + 2) * 65536 + Int(slot_id)


@fieldwise_init
struct NodeArenaRuntime(Copyable, Writable):
    var node_id: Int
    var arena_ptr: Int
    var base_ptr: Int
    var weight_base: Int
    var weight_bytes: Int
    var state_base: Int
    var state_bytes: Int
    var total_bytes: Int


@fieldwise_init
struct LoadedStaticPackage(Movable):
    var manifest: StaticPackageManifest
    var plan: StaticLoadPlan
    var arena_ptrs: List[Int]
    var nodes: List[NodeArenaRuntime]
    var records: Dict[Int, LoadedTensorRecord]
    var bytes_loaded: Int

    fn __del__(deinit self):
        for i in range(len(self.arena_ptrs)):
            if self.arena_ptrs[i] != 0:
                destroy_arena_ptr(self.arena_ptrs[i])
                self.arena_ptrs[i] = 0


fn arena_ptr_from_addr[
    alignment: Int = DEFAULT_ALIGNMENT,
](
    addr: Int
) -> UnsafePointer[NumaArena[alignment=alignment], MutAnyOrigin]:
    return UnsafePointer[NumaArena[alignment=alignment], MutAnyOrigin](
        unsafe_from_address=addr
    )


fn make_arena_ptr[
    alignment: Int = DEFAULT_ALIGNMENT,
](node_id: Int, total_bytes: Int) -> Int:
    var ptr = alloc[NumaArena[alignment=alignment]](1)
    var arena = NumaArena[alignment=alignment](node_id, total_bytes)
    ptr.init_pointee_move(arena^)
    if not ptr[]:
        ptr.destroy_pointee()
        ptr.free()
        return 0
    return Int(ptr)


fn destroy_arena_ptr[
    alignment: Int = DEFAULT_ALIGNMENT,
](addr: Int):
    if addr == 0:
        return
    var ptr = arena_ptr_from_addr[alignment](addr)
    ptr.destroy_pointee()
    ptr.free()


fn cleanup_all_arenas[
    alignment: Int = DEFAULT_ALIGNMENT,
](mut arena_ptrs: List[Int]):
    for i in range(len(arena_ptrs)):
        if arena_ptrs[i] != 0:
            destroy_arena_ptr[alignment](arena_ptrs[i])
            arena_ptrs[i] = 0


fn sort_shards_by_index(mut shards: List[TensorShardView]):
    for i in range(1, len(shards)):
        var cur = shards[i].copy()
        var j = i
        while j > 0 and shards[j - 1].shard_index > cur.shard_index:
            shards[j] = shards[j - 1].copy()
            j -= 1
        shards[j] = cur^


fn build_replicated_geometry(
    tensor_name: String,
    expected_shape: List[Int],
    dtype: DType,
    file_offset: Int,
) -> Optional[ShardGeometry]:
    var total_bytes = shape_num_bytes(expected_shape, dtype)
    if total_bytes <= 0:
        print("Unsupported dtype for", tensor_name, "dtype", dtype)
        return None
    var global_offset = List[Int](length=len(expected_shape), fill=0)
    var fragments = List[ReadFragmentPlan]()
    fragments.append(ReadFragmentPlan(
        file_offset=file_offset,
        shard_offset=0,
        length_bytes=total_bytes,
    ))
    return ShardGeometry(
        local_shape=expected_shape.copy(),
        global_offset=global_offset^,
        num_bytes=total_bytes,
        fragments=fragments^,
    )


fn build_sharded_geometry(
    tensor_name: String,
    expected_shape: List[Int],
    dtype: DType,
    file_offset: Int,
    shard_axis: Int,
    shard_index: Int,
    shard_count: Int,
) -> Optional[ShardGeometry]:
    if len(expected_shape) != 2:
        print("Sharding currently supports rank-2 tensors only:", tensor_name, "shape", expected_shape)
        return None

    var elem_bytes = dtype_bytes(dtype)
    if elem_bytes <= 0:
        print("Unsupported dtype for sharding:", tensor_name, dtype)
        return None

    var rows = expected_shape[0]
    var cols = expected_shape[1]

    if shard_axis == 0:
        if rows % shard_count != 0:
            print("Axis-0 shard mismatch:", tensor_name, "rows", rows, "shards", shard_count)
            return None

        var local_rows = rows // shard_count
        var row_start = shard_index * local_rows
        var shard_bytes = local_rows * cols * elem_bytes

        var local_shape = List[Int]()
        local_shape.append(local_rows)
        local_shape.append(cols)

        var global_offset = List[Int]()
        global_offset.append(row_start)
        global_offset.append(0)

        var fragments = List[ReadFragmentPlan]()
        fragments.append(ReadFragmentPlan(
            file_offset=file_offset + row_start * cols * elem_bytes,
            shard_offset=0,
            length_bytes=shard_bytes,
        ))

        return ShardGeometry(
            local_shape=local_shape^,
            global_offset=global_offset^,
            num_bytes=shard_bytes,
            fragments=fragments^,
        )

    if shard_axis == 1:
        if cols % shard_count != 0:
            print("Axis-1 shard mismatch:", tensor_name, "cols", cols, "shards", shard_count)
            return None

        var local_cols = cols // shard_count
        var col_start = shard_index * local_cols

        var local_shape = List[Int]()
        local_shape.append(rows)
        local_shape.append(local_cols)

        var global_offset = List[Int]()
        global_offset.append(0)
        global_offset.append(col_start)

        var row_bytes = local_cols * elem_bytes
        var fragments = List[ReadFragmentPlan](capacity=rows)
        for r in range(rows):
            var src = file_offset + (r * cols + col_start) * elem_bytes
            var dst = r * row_bytes
            fragments.append(ReadFragmentPlan(
                file_offset=src,
                shard_offset=dst,
                length_bytes=row_bytes,
            ))

        return ShardGeometry(
            local_shape=local_shape^,
            global_offset=global_offset^,
            num_bytes=rows * row_bytes,
            fragments=fragments^,
        )

    print("Unsupported shard axis", shard_axis, "for tensor", tensor_name)
    return None


fn build_shard_geometry(
    tensor_name: String,
    expected_shape: List[Int],
    dtype: DType,
    file_offset: Int,
    shard_axis: Int,
    shard_index: Int,
    shard_count: Int,
) -> Optional[ShardGeometry]:
    if shard_axis == AXIS_NONE or shard_axis == AXIS_HOST:
        return build_replicated_geometry(tensor_name, expected_shape, dtype, file_offset)

    if shard_count <= 0:
        print("Invalid shard count for", tensor_name, shard_count)
        return None

    return build_sharded_geometry(
        tensor_name,
        expected_shape,
        dtype,
        file_offset,
        shard_axis,
        shard_index,
        shard_count,
    )


fn build_static_distribution_plan[
    desc_t: StaticModelDescriptor,
](
    manifest: StaticPackageManifest,
    alignment: Int = DEFAULT_ALIGNMENT,
) -> Optional[StaticLoadPlan]:
    var node_ids = desc_t.node_ids()
    if len(node_ids) == 0:
        print("Descriptor has zero nodes")
        return None

    if len(node_ids) != desc_t.NODE_COUNT:
        print("Descriptor node-count mismatch, trait", desc_t.NODE_COUNT, "method", len(node_ids))
        return None

    if len(node_ids) != desc_t.TP:
        print("Descriptor TP/node mismatch, TP", desc_t.TP, "nodes", len(node_ids))
        return None

    var state_per_node = desc_t.state_bytes_per_node(alignment)
    var node_weight_bytes = List[Int](length=len(node_ids), fill=0)
    var shards = List[ShardLoadPlan]()
    var specs = desc_t.tensor_specs()

    if len(manifest.slot_layout) != len(specs):
        print(
            "Package manifest slot count mismatch, manifest",
            len(manifest.slot_layout),
            "descriptor",
            len(specs),
        )
        return None

    for i in range(len(specs)):
        var spec = specs[i].copy()
        var layout = manifest.slot_layout[i].copy()
        if layout.file_idx < 0 or layout.file_idx >= len(manifest.files):
            print("Package manifest file index out of range for", spec.tensor_name, "index", layout.file_idx)
            return None
        var dtype = spec.dtype
        if dtype == DType.invalid:
            print("Unsupported dtype for", spec.tensor_name)
            return None

        if spec.shard_axis != AXIS_NONE and spec.shard_axis != AXIS_HOST and spec.shard_axis != 0 and spec.shard_axis != 1:
            print("Unsupported shard axis for", spec.tensor_name, spec.shard_axis)
            return None

        # AXIS_HOST: single copy on host node only
        var is_host_only = spec.shard_axis == AXIS_HOST
        var host_idx = desc_t.host_node_index()
        var shard_count = 1 if is_host_only else len(node_ids)
        for shard_idx in range(shard_count):
            var node_idx = host_idx if is_host_only else shard_idx
            var node_id = node_ids[node_idx]

            var geom_opt = build_shard_geometry(
                spec.tensor_name,
                spec.expected_shape,
                dtype,
                layout.file_offset,
                spec.shard_axis,
                shard_idx,
                shard_count,
            )
            if not geom_opt:
                return None
            var geom = geom_opt.take()

            var arena_off = align_up_int(node_weight_bytes[node_idx], alignment)
            node_weight_bytes[node_idx] = arena_off + geom.num_bytes

            shards.append(ShardLoadPlan(
                slot_id=spec.slot_id,
                slot_name=spec.slot_name.copy(),
                layer_idx=spec.layer_idx,
                tensor_name=spec.tensor_name.copy(),
                file_idx=layout.file_idx,
                dtype=dtype,
                shard_axis=spec.shard_axis,
                global_shape=spec.expected_shape.copy(),
                node_index=node_idx,
                node_id=node_id,
                shard_index=shard_idx,
                shard_count=shard_count,
                local_shape=geom.local_shape.copy(),
                global_offset=geom.global_offset.copy(),
                num_bytes=geom.num_bytes,
                arena_offset=arena_off,
                fragments=geom.fragments.copy(),
            ))

    var nodes = List[NodeArenaPlan](capacity=len(node_ids))
    var total_weight_bytes = 0
    var total_state_bytes = 0

    for n in range(len(node_ids)):
        var weight_bytes = node_weight_bytes[n]
        var total_bytes = weight_bytes + state_per_node
        total_weight_bytes += weight_bytes
        total_state_bytes += state_per_node
        nodes.append(NodeArenaPlan(
            node_id=node_ids[n],
            weight_bytes=weight_bytes,
            state_bytes=state_per_node,
            total_bytes=total_bytes,
        ))

    return StaticLoadPlan(
        node_ids=node_ids^,
        nodes=nodes^,
        shards=shards^,
        alignment=alignment,
        total_weight_bytes=total_weight_bytes,
        total_state_bytes=total_state_bytes,
        total_bytes=total_weight_bytes + total_state_bytes,
    )


fn build_records_from_plan(
    plan: StaticLoadPlan,
    node_weight_bases: List[Int],
) -> Dict[Int, LoadedTensorRecord]:
    var records = Dict[Int, LoadedTensorRecord]()

    for i in range(len(plan.shards)):
        var shard = plan.shards[i].copy()
        var shard_base = node_weight_bases[shard.node_index] + shard.arena_offset
        var shard_view = TensorShardView(
            arena_id=shard.node_index,
            node_id=shard.node_id,
            ptr=shard_base,
            num_bytes=shard.num_bytes,
            local_shape=shard.local_shape.copy(),
            global_offset=shard.global_offset.copy(),
            shard_index=shard.shard_index,
            shard_count=shard.shard_count,
        )

        var key = make_slot_key(shard.layer_idx, shard.slot_id)
        var existing_opt = records.get(key)
        if existing_opt:
            var record = existing_opt.value().copy()
            record.shards.append(shard_view.copy())
            records[key] = record^
        else:
            var shard_list = List[TensorShardView]()
            shard_list.append(shard_view^)
            records[key] = LoadedTensorRecord(
                slot_id=shard.slot_id,
                slot_name=shard.slot_name.copy(),
                layer_idx=shard.layer_idx,
                tensor_name=shard.tensor_name.copy(),
                dtype=shard.dtype,
                shard_axis=shard.shard_axis,
                shape=shard.global_shape.copy(),
                shards=shard_list^,
            )
    var keys = List[Int]()
    for item in records.items():
        keys.append(item.key)
    for i in range(len(keys)):
        var key = keys[i]
        var record_opt = records.get(key)
        if not record_opt:
            continue
        var record = record_opt.value().copy()
        sort_shards_by_index(record.shards)
        records[key] = record^
    return records^


fn load_static_package[
    desc_t: StaticModelDescriptor,
    io_depth: Int = DEFAULT_IO_QUEUE_DEPTH,
    alignment: Int = DEFAULT_ALIGNMENT,
](
    package_manifest_path: String,
) -> Optional[LoadedStaticPackage]:
    var manifest_opt = parse_static_package_manifest(Path(package_manifest_path))
    if not manifest_opt:
        return None
    var manifest = manifest_opt.take()

    if manifest.model_id != desc_t.model_id():
        print("Package manifest model_id mismatch:", manifest.model_id, "expected", desc_t.model_id())
        return None

    var plan_opt = build_static_distribution_plan[desc_t](manifest, alignment)
    if not plan_opt:
        return None
    var plan = plan_opt.take()

    var node_count = len(plan.nodes)
    var arena_ptrs = List[Int](length=node_count, fill=0)
    var node_weight_bases = List[Int](length=node_count, fill=0)
    var nodes = List[NodeArenaRuntime](capacity=node_count)

    for i in range(node_count):
        var node_plan = plan.nodes[i].copy()
        var arena_addr = make_arena_ptr[alignment](node_plan.node_id, node_plan.total_bytes)
        if arena_addr == 0:
            print("Failed to allocate NUMA arena for node", node_plan.node_id)
            cleanup_all_arenas[alignment](arena_ptrs)
            return None
        arena_ptrs[i] = arena_addr

        var arena = arena_ptr_from_addr[alignment](arena_addr)
        var base_ptr = Int(arena[].base)

        var weight_base = base_ptr
        if node_plan.weight_bytes > 0:
            var weight_ptr = arena[].alloc[UInt8](node_plan.weight_bytes)
            if not weight_ptr:
                print("Failed to reserve weight bytes on node", node_plan.node_id)
                cleanup_all_arenas[alignment](arena_ptrs)
                return None
            weight_base = Int(weight_ptr)

        var state_base = weight_base + node_plan.weight_bytes
        if node_plan.state_bytes > 0:
            var state_ptr = arena[].alloc[UInt8](node_plan.state_bytes)
            if not state_ptr:
                print("Failed to reserve runtime-state bytes on node", node_plan.node_id)
                cleanup_all_arenas[alignment](arena_ptrs)
                return None
            state_base = Int(state_ptr)

        node_weight_bases[i] = weight_base

        nodes.append(NodeArenaRuntime(
            node_id=node_plan.node_id,
            arena_ptr=arena_addr,
            base_ptr=base_ptr,
            weight_base=weight_base,
            weight_bytes=node_plan.weight_bytes,
            state_base=state_base,
            state_bytes=node_plan.state_bytes,
            total_bytes=node_plan.total_bytes,
        ))

    var ops = List[ReadOp]()
    var op_id = 0
    for i in range(len(plan.shards)):
        var shard = plan.shards[i].copy()
        var shard_base = node_weight_bases[shard.node_index] + shard.arena_offset
        for f in range(len(shard.fragments)):
            var frag = shard.fragments[f].copy()
            ops.append(ReadOp(
                file_idx=shard.file_idx,
                offset=frag.file_offset,
                length=frag.length_bytes,
                dest=shard_base + frag.shard_offset,
                id=op_id,
            ))
            op_id += 1

    var loader = IoLoader[io_depth]()
    if not loader:
        print("experimental3 requires Linux io_uring; no fallback loader is implemented")
        cleanup_all_arenas[alignment](arena_ptrs)
        return None

    var paths = List[Path]()
    for i in range(len(manifest.files)):
        paths.append(Path(manifest.files[i]))
    var registered = loader.register_files(paths)
    if registered < 0:
        print("Failed to register package files, errno:", registered)
        cleanup_all_arenas[alignment](arena_ptrs)
        return None

    var bytes_loaded = 0

    @parameter
    fn on_complete(c: Completion):
        bytes_loaded += Int(c.result)

    var err = loader.process_queue_checked[on_complete](ops)
    if err:
        print_io_load_error(err.value())
        cleanup_all_arenas[alignment](arena_ptrs)
        return None

    var records = build_records_from_plan(plan, node_weight_bases)

    return LoadedStaticPackage(
        manifest=manifest^,
        plan=plan^,
        arena_ptrs=arena_ptrs^,
        nodes=nodes^,
        records=records^,
        bytes_loaded=bytes_loaded,
    )
