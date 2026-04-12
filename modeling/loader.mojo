from std.pathlib import Path
from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, byte_count,
    Absorbed,
    WeightIterable, WeightDesc, weight_desc,
)
from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from linux.io_uring import IoRing, ReadOp, Completion
from safetensors.loader import process_read_queue, LoadError
from notstdcollections import HeapMoveArray


comptime DEFAULT_IO_DEPTH = 2048


def discover_shards(dir_path: Path) -> List[Path]:
    """Enumerate safetensors files in a directory, sorted by name.

    Matches any *.safetensors — this covers multi-shard HF checkpoints
    (model-00001-of-000NN.safetensors), single-file HF checkpoints
    (model.safetensors), and single-file quantizer outputs. Callers are
    expected to point at a directory that only contains the intended
    checkpoint's tensor files.
    """
    var shards = List[Path]()
    try:
        for entry in dir_path.listdir():
            var name = String(entry)
            if name.endswith(".safetensors"):
                shards.append(dir_path / name)
    except:
        pass
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            if String(shards[j]) < String(shards[i]):
                var tmp = shards[i]
                shards[i] = shards[j]
                shards[j] = tmp
    return shards^


@fieldwise_init
struct ReadFragment(Copyable):
    var file_idx: Int
    var file_offset: Int
    var dest: Int
    var length: Int


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
    mut ops: List[ReadFragment],
):
    var dest = arena_base + desc.arena_offset
    var local_bytes = desc.local_rows * desc.local_cols * desc.element_bytes

    if desc.local_rows == desc.global_rows and desc.local_cols == desc.global_cols:
        ops.append(ReadFragment(
            file_idx=file_idx, file_offset=file_data_start, dest=dest, length=local_bytes,
        ))
    elif desc.local_rows != desc.global_rows:
        var row_start = rank * desc.local_rows
        var file_off = file_data_start + row_start * desc.global_cols * desc.element_bytes
        ops.append(ReadFragment(
            file_idx=file_idx, file_offset=file_off, dest=dest, length=local_bytes,
        ))
    else:
        var col_start = rank * desc.local_cols
        var local_row_bytes = desc.local_cols * desc.element_bytes
        for r in range(desc.local_rows):
            var src = file_data_start + (r * desc.global_cols + col_start) * desc.element_bytes
            var dst = dest + r * local_row_bytes
            ops.append(ReadFragment(
                file_idx=file_idx, file_offset=src, dest=dst, length=local_row_bytes,
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
    mut fragments: List[ReadFragment],
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
        emit_reads(w, shard_idx, data_start, arena_bases[ranks[i]], ranks[i], fragments)
    return True


@fieldwise_init
struct LoadResult(Movable):
    var bytes_loaded: Int
    var num_ops: Int


def load_weights[
    M: WeightIterable,
    io_depth: Int = DEFAULT_IO_DEPTH,
](
    paths: List[Path],
    arena_bases: List[Int],
) -> Optional[LoadResult]:
    """Load weights from one or more safetensors files into pre-allocated arenas.

    Supports sharded checkpoints: each tensor is looked up across all shard
    headers, and reads are issued from the correct file.
    """
    var headers = HeapMoveArray[SafetensorsHeader](len(paths))
    for i in range(len(paths)):
        var header_opt = parse_safetensors_header(paths[i])
        if not header_opt:
            print("failed to parse:", String(paths[i]))
            return None
        headers.push(header_opt.take())

    var targeted_weights = List[WeightDesc]()
    var distributed_weights = List[WeightDesc]()

    @parameter
    def collect[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        comptime if conforms_to(T, Absorbed):
            pass
        else:
            var desc = weight_desc[T](prefix, base)
            if desc.target_rank >= 0:
                targeted_weights.append(desc^)
            else:
                distributed_weights.append(desc^)

    M.for_each_weight[collect]()

    var fragments = List[ReadFragment]()
    var tp = len(arena_bases)

    var all_ranks = List[Int]()
    for r in range(tp):
        all_ranks.append(r)

    for w in targeted_weights:
        var ranks = List[Int]()
        ranks.append(w.target_rank % tp)
        if not resolve_and_emit(w, headers, arena_bases, ranks, fragments):
            return None

    for w in distributed_weights:
        if not resolve_and_emit(w, headers, arena_bases, all_ranks, fragments):
            return None

    var num_fragments = len(fragments)
    var ops = List[ReadOp[]](capacity=num_fragments)
    for i in range(num_fragments):
        var frag = fragments[i].copy()
        ops.append(ReadOp(
            file_idx=frag.file_idx,
            offset=frag.file_offset,
            length=frag.length,
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=frag.dest),
            id=i,
        ))

    var ring = IoRing[io_depth]()
    if not ring:
        print("io_uring setup failed")
        return None

    try:
        _ = ring.register_files(paths)
    except err:
        print("register_files failed:", err.error_message())
        return None

    var bytes_loaded = 0

    @parameter
    def on_complete(c: Completion):
        bytes_loaded += Int(c.result)

    var err = process_read_queue[on_complete](ring, ops)
    if err:
        print("load error:", err.value().msg)
        return None

    return LoadResult(bytes_loaded, num_fragments)


def load_safetensors[
    M: WeightIterable,
    io_depth: Int = DEFAULT_IO_DEPTH,
](
    path: Path,
    arena_bases: List[Int],
) -> Optional[LoadResult]:
    """Load weights from a single safetensors file."""
    var paths = List[Path]()
    paths.append(path)
    return load_weights[M, io_depth](paths, arena_bases)


def load_weights_from_descs[
    io_depth: Int = DEFAULT_IO_DEPTH,
](
    descs: List[WeightDesc],
    paths: List[Path],
    arena_bases: List[Int],
) -> Optional[LoadResult]:
    """Runtime variant of load_weights — takes a prebuilt List[WeightDesc].

    Intended for models that build their weight catalog at runtime rather
    than through the WeightIterable / for_each_weight comptime template.
    Shares all the io_uring + validation plumbing with load_weights[M];
    differs only in how the desc list is produced.
    """
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

    var fragments = List[ReadFragment]()
    var tp = len(arena_bases)

    var all_ranks = List[Int]()
    for r in range(tp):
        all_ranks.append(r)

    for w in targeted_weights:
        var ranks = List[Int]()
        ranks.append(w.target_rank % tp)
        if not resolve_and_emit(w, headers, arena_bases, ranks, fragments):
            return None

    for w in distributed_weights:
        if not resolve_and_emit(w, headers, arena_bases, all_ranks, fragments):
            return None

    var num_fragments = len(fragments)
    var ops = List[ReadOp[]](capacity=num_fragments)
    for i in range(num_fragments):
        var frag = fragments[i].copy()
        ops.append(ReadOp(
            file_idx=frag.file_idx,
            offset=frag.file_offset,
            length=frag.length,
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=frag.dest),
            id=i,
        ))

    var ring = IoRing[io_depth]()
    if not ring:
        print("io_uring setup failed")
        return None

    try:
        _ = ring.register_files(paths)
    except err:
        print("register_files failed:", err.error_message())
        return None

    var bytes_loaded = 0

    @parameter
    def on_complete(c: Completion):
        bytes_loaded += Int(c.result)

    var err = process_read_queue[on_complete](ring, ops)
    if err:
        print("load error:", err.value().msg)
        return None

    return LoadResult(bytes_loaded, num_fragments)
