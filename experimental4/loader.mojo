from std.pathlib import Path

from experimental4.model_spec import (
    Encoding, Shaped, Placed, Named, byte_count,
    NodeLocal,
    WeightIterable, WeightDesc, weight_desc,
)
from safetensors.parser import parse_safetensors_header
from safetensors.loader import IoLoader, ReadOp, Completion, print_io_load_error


comptime DEFAULT_IO_DEPTH = 2048


@fieldwise_init
struct ReadFragment(Copyable):
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
    else:
        print("unexpected rank for", desc.name + ":", len(found_shape))
        return False

    return True


def emit_reads(
    desc: WeightDesc,
    file_data_start: Int,
    arena_base: Int,
    rank: Int,
    mut ops: List[ReadFragment],
):
    var dest = arena_base + desc.arena_offset
    var local_bytes = desc.local_rows * desc.local_cols * desc.element_bytes

    if desc.local_rows == desc.global_rows and desc.local_cols == desc.global_cols:
        ops.append(ReadFragment(
            file_offset=file_data_start, dest=dest, length=local_bytes,
        ))
    elif desc.local_rows != desc.global_rows:
        var row_start = rank * desc.local_rows
        var file_off = file_data_start + row_start * desc.global_cols * desc.element_bytes
        ops.append(ReadFragment(
            file_offset=file_off, dest=dest, length=local_bytes,
        ))
    else:
        var col_start = rank * desc.local_cols
        var local_row_bytes = desc.local_cols * desc.element_bytes
        for r in range(desc.local_rows):
            var src = file_data_start + (r * desc.global_cols + col_start) * desc.element_bytes
            var dst = dest + r * local_row_bytes
            ops.append(ReadFragment(
                file_offset=src, dest=dst, length=local_row_bytes,
            ))


@fieldwise_init
struct LoadResult(Movable):
    var bytes_loaded: Int
    var num_ops: Int


def load_safetensors[
    M: WeightIterable,
    io_depth: Int = DEFAULT_IO_DEPTH,
](
    path: Path,
    arena_bases: List[Int],
    host_index: Int = 0,
) -> Optional[LoadResult]:
    """Load weights from a safetensors file into pre-allocated arenas.

    Args:
        path: Path to the safetensors file.
        arena_bases: Base address of each rank's arena (len == tp).
        host_index: Index into arena_bases for NodeLocal weights.
    """
    var header_opt = parse_safetensors_header(path)
    if not header_opt:
        return None
    var header = header_opt.take()

    var node_local_weights = List[WeightDesc]()
    var distributed_weights = List[WeightDesc]()

    @parameter
    def collect[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        comptime if conforms_to(T, NodeLocal):
            node_local_weights.append(weight_desc[T](prefix, base))
        else:
            distributed_weights.append(weight_desc[T](prefix, base))

    M.for_each_weight[collect]()

    var fragments = List[ReadFragment]()
    var tp = len(arena_bases)

    for i in range(len(node_local_weights)):
        var w = node_local_weights[i].copy()
        var meta_opt = header.tensors.get(w.name)
        if not meta_opt:
            print("missing tensor:", w.name)
            return None
        var meta = meta_opt.value().copy()
        if not validate_weight(w, meta.dtype, meta.shape):
            return None
        emit_reads(w, header.data_offset + meta.start, arena_bases[host_index], host_index, fragments)

    for i in range(len(distributed_weights)):
        var w = distributed_weights[i].copy()
        var meta_opt = header.tensors.get(w.name)
        if not meta_opt:
            print("missing tensor:", w.name)
            return None
        var meta = meta_opt.value().copy()
        if not validate_weight(w, meta.dtype, meta.shape):
            return None
        for rank in range(tp):
            emit_reads(w, header.data_offset + meta.start, arena_bases[rank], rank, fragments)

    var num_fragments = len(fragments)
    var ops = List[ReadOp](capacity=num_fragments)
    for i in range(num_fragments):
        var frag = fragments[i].copy()
        ops.append(ReadOp(
            file_idx=0,
            offset=frag.file_offset,
            length=frag.length,
            dest=frag.dest,
            id=i,
        ))

    var loader = IoLoader[io_depth]()
    if not loader:
        print("io_uring setup failed")
        return None

    var paths = List[Path]()
    paths.append(path)
    var registered = loader.register_files(paths)
    if registered < 0:
        print("register_files failed, errno:", registered)
        return None

    var bytes_loaded = 0

    @parameter
    def on_complete(c: Completion):
        bytes_loaded += Int(c.result)

    var err = loader.process_queue_checked[on_complete](ops)
    if err:
        print_io_load_error(err.value())
        return None

    return LoadResult(bytes_loaded, num_fragments)
