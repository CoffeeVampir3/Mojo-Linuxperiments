from std.pathlib import Path
from std.memory import UnsafePointer

from modeling.model_spec import WeightDesc
from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from linux.io_uring import IoRing, ReadOp, run_reads_multi
from notstdcollections import HeapMoveArray
from threading.burst_threading import BurstPool
from numa import NumaInfo, NumaTopology


comptime DEFAULT_IO_DEPTH = 2048
comptime DEFAULT_MASK_SIZE = 128


def discover_shards(path: Path) -> List[Path]:
    """Enumerate safetensors shard paths, sorted by name.

    Accepts either:
      - a directory containing one or more *.safetensors files
      - a direct *.safetensors file path

    This covers multi-shard HF checkpoints (model-00001-of-000NN.safetensors),
    single-file HF checkpoints (model.safetensors), and single-file quantizer
    outputs.
    """
    var shards = List[Path]()
    var path_str = String(path)
    if path_str.endswith(".safetensors"):
        shards.append(path)
        return shards^

    try:
        for entry in path.listdir():
            var name = String(entry)
            if name.endswith(".safetensors"):
                shards.append(path / name)
    except:
        pass
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            if String(shards[j]) < String(shards[i]):
                var tmp = shards[i]
                shards[i] = shards[j]
                shards[j] = tmp
    return shards^


def validate_weight(
    desc: WeightDesc, found_dtype: DType, found_shape: List[Int],
) -> Bool:
    if desc.dtype != found_dtype:
        print("dtype mismatch for", desc.name + ":",
            "expected", desc.dtype, "got", found_dtype)
        return False

    if len(found_shape) == 1:
        var expected = desc.global_rows * desc.global_cols
        if expected != found_shape[0]:
            print("shape mismatch for", desc.name + ":",
                "expected [" + String(expected) + "]",
                "got [" + String(found_shape[0]) + "]")
            return False
    elif len(found_shape) == 2:
        if desc.global_rows != found_shape[0] or desc.global_cols != found_shape[1]:
            print("shape mismatch for", desc.name + ":",
                "expected [" + String(desc.global_rows) + ", " + String(desc.global_cols) + "]",
                "got [" + String(found_shape[0]) + ", " + String(found_shape[1]) + "]")
            return False
    elif len(found_shape) == 3:
        var folded_rows = found_shape[0] * found_shape[1]
        if desc.global_rows != folded_rows or desc.global_cols != found_shape[2]:
            print("shape mismatch for", desc.name + ":",
                "expected [" + String(desc.global_rows) + ", " + String(desc.global_cols) + "]",
                "got [" + String(found_shape[0]) + ", " + String(found_shape[1]) + ", " + String(found_shape[2]) + "]")
            return False
    else:
        print("unexpected rank for", desc.name + ":", len(found_shape))
        return False

    return True


def emit_reads(
    desc: WeightDesc,
    file_idx: Int,
    file_data_start: Int,
    arena_base: Int,
    rank: Int,
    mut ops: List[ReadOp[]],
):
    """Append ReadOps for the given weight into `ops`. Each op's id is its
    local index within `ops` — the ring uses id as a direct lookup index."""
    var dest = arena_base + desc.arena_offset

    if desc.data_rows == desc.global_rows and desc.data_cols == desc.global_cols:
        var data_bytes = desc.data_rows * desc.data_cols * desc.element_bytes
        ops.append(ReadOp(
            file_idx=file_idx, offset=file_data_start, length=data_bytes,
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=dest),
            id=len(ops),
        ))
    elif desc.data_rows != desc.global_rows:
        var row_start = rank * desc.data_rows
        var data_bytes = desc.data_rows * desc.global_cols * desc.element_bytes
        var file_off = file_data_start + row_start * desc.global_cols * desc.element_bytes
        ops.append(ReadOp(
            file_idx=file_idx, offset=file_off, length=data_bytes,
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=dest),
            id=len(ops),
        ))
    else:
        var file_cols = desc.data_cols
        var stride_cols = desc.local_cols
        var col_start = rank * file_cols
        var file_row_bytes = file_cols * desc.element_bytes
        var stride_bytes = stride_cols * desc.element_bytes
        for r in range(desc.data_rows):
            var src = file_data_start + (r * desc.global_cols + col_start) * desc.element_bytes
            var dst = dest + r * stride_bytes
            ops.append(ReadOp(
                file_idx=file_idx, offset=src, length=file_row_bytes,
                dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
                id=len(ops),
            ))


def find_tensor(
    name: String,
    ref headers: HeapMoveArray[SafetensorsHeader],
) -> Optional[Tuple[Int, TensorMeta]]:
    for i in range(len(headers)):
        var meta_opt = headers[i].tensors.get(name)
        if meta_opt:
            return (i, meta_opt.value().copy())
    return None


def resolve_and_emit(
    w: WeightDesc,
    ref headers: HeapMoveArray[SafetensorsHeader],
    arena_bases: List[Int],
    ranks: List[Int],
    mut ops_per_rank: List[List[ReadOp[]]],
) -> Bool:
    var found = find_tensor(w.name, headers)
    if not found:
        print("missing tensor:", w.name)
        return False
    var shard_idx = found.value()[0]
    var meta = found.value()[1].copy()
    if not validate_weight(w, meta.dtype, meta.shape):
        return False
    var data_start = headers[shard_idx].data_offset + meta.start
    for i in range(len(ranks)):
        var r = ranks[i]
        emit_reads(w, shard_idx, data_start, arena_bases[r], r, ops_per_rank[r])
    return True


@fieldwise_init
struct LoadResult(Movable):
    var bytes_loaded: Int
    var num_ops: Int


def load_weights_from_descs[
    io_depth: Int = DEFAULT_IO_DEPTH,
    mask_size: Int = DEFAULT_MASK_SIZE,
](
    descs: List[WeightDesc],
    paths: List[Path],
    arena_bases: List[Int],
    numa_topo: NumaTopology,
) -> Optional[LoadResult]:
    """Runtime variant — takes a prebuilt List[WeightDesc]."""
    var headers = HeapMoveArray[SafetensorsHeader](len(paths))
    for i in range(len(paths)):
        var header_opt = parse_safetensors_header(paths[i])
        if not header_opt:
            print("failed to parse:", String(paths[i]))
            return None
        headers.push(header_opt.take())

    var targeted_weights = List[WeightDesc]()
    var distributed_weights = List[WeightDesc]()
    for i in range(len(descs)):
        var d = descs[i].copy()
        if d.absorbed:
            continue
        if d.target_rank >= 0:
            targeted_weights.append(d^)
        else:
            distributed_weights.append(d^)

    var tp = len(arena_bases)
    var ops_per_rank = List[List[ReadOp[]]]()
    for _ in range(tp):
        ops_per_rank.append(List[ReadOp[]]())

    var all_ranks = List[Int]()
    for r in range(tp):
        all_ranks.append(r)

    for w in targeted_weights:
        var ranks = List[Int]()
        ranks.append(w.target_rank % tp)
        if not resolve_and_emit(w, headers, arena_bases, ranks, ops_per_rank):
            return None
    for w in distributed_weights:
        if not resolve_and_emit(w, headers, arena_bases, all_ranks, ops_per_rank):
            return None

    return run_load[io_depth, mask_size](paths, numa_topo, ops_per_rank^)


def run_load[
    io_depth: Int,
    mask_size: Int,
](
    paths: List[Path],
    numa_topo: NumaTopology,
    var ops_per_rank: List[List[ReadOp[]]],
) -> Optional[LoadResult]:
    """Build transient load pools (one 1-capacity pool per NUMA node in
    numa_topo), run the multi-pool read dispatch, tally bytes/ops. The
    load pools are destroyed on return; the inference-time pools are a
    separate concern of the caller."""
    var total_bytes = 0
    var total_ops = 0
    for r in range(len(ops_per_rank)):
        for i in range(len(ops_per_rank[r])):
            total_bytes += ops_per_rank[r][i].length
            total_ops += 1

    var numa = NumaInfo()
    var tp = len(ops_per_rank)
    var load_pools = HeapMoveArray[BurstPool[mask_size]](tp)
    for r in range(tp):
        var mask = numa.get_node_mask[mask_size](numa_topo[r])
        load_pools.push(BurstPool[mask_size](
            capacity=1, cpu_mask=mask^, numa_node=numa_topo[r]))
        if not load_pools[r]:
            print("load pool setup failed for rank", r)
            return None

    var pools_span = Span[BurstPool[mask_size], MutAnyOrigin](
        ptr=load_pools.ptr, length=len(load_pools))
    run_reads_multi[io_depth, mask_size](pools_span, paths, ops_per_rank)

    return LoadResult(total_bytes, total_ops)
