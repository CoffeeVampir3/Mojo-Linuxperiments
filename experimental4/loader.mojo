# ===----------------------------------------------------------------------=== #
# loader.mojo — General weight loading system with io_uring execution
#
# The model spec describes its weights via weight_map() -> List[WeightDesc].
# The loader takes that description and handles everything: header parsing,
# validation, shard-aware read geometry, arena allocation, io_uring batch
# execution. The model has zero loading logic.
#
# WeightDesc is the runtime interchange — it captures the values that
# comptime Slot types produce (dtype, element_bytes, local/global dims)
# so the loader can operate without knowing the model's type structure.
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import linux.sys as linux

from experimental4.model_spec import Encoding, Shaped, Placed, Named, byte_count
from safetensors.parser import SafetensorsHeader, TensorMeta, parse_safetensors_header
from safetensors.loader import IoLoader, ReadOp, Completion, print_io_load_error


comptime DEFAULT_ALIGNMENT = 64
comptime DEFAULT_IO_DEPTH = 2048


# ===--- Comptime offset computation ---=== #

@always_inline
fn align_up(value: Int, alignment: Int) -> Int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


fn offset_after[T: Encoding & Shaped, base: Int, alignment: Int = DEFAULT_ALIGNMENT]() -> Int:
    """Comptime: returns the next arena offset after placing T at aligned base."""
    comptime aligned = ((base + alignment - 1) // alignment) * alignment
    return aligned + byte_count[T]()


fn next_offset[T: Encoding & Shaped & Placed, alignment: Int = DEFAULT_ALIGNMENT]() -> Int:
    """Comptime: next arena offset after a PlacedSlot, using its own OFFSET as base."""
    comptime aligned = ((T.OFFSET + alignment - 1) // alignment) * alignment
    return aligned + byte_count[T]()


# ===--- Weight description ---=== #

@fieldwise_init
struct WeightDesc(Copyable):
    """Runtime description of a single weight tensor.
    Bridges comptime Slot info to the loader."""
    var name: String         # safetensors tensor name
    var arena_offset: Int    # byte offset within weight arena
    var dtype: DType         # expected dtype
    var element_bytes: Int   # bytes per element
    var global_rows: Int     # full tensor shape (for validation + shard geometry)
    var global_cols: Int
    var local_rows: Int      # post-sharding local shape
    var local_cols: Int


fn weight_desc[T: Encoding & Shaped & Placed & Named](
    prefix: String = "", base: Int = 0,
) -> WeightDesc:
    """Build a WeightDesc entirely from the PlacedSlot's comptime info.
    prefix is prepended to T.NAME (e.g. layer prefix for per-layer weights)."""
    return WeightDesc(
        name=prefix + String(T.NAME), arena_offset=base + T.OFFSET,
        dtype=T.DTYPE, element_bytes=T.ELEMENT_BYTES,
        global_rows=T.GLOBAL_ROWS, global_cols=T.GLOBAL_COLS,
        local_rows=T.ROWS, local_cols=T.COLS,
    )


# ===--- Loadable trait ---=== #

trait Loadable:
    """Model descriptor for loading. The model describes its structure
    via traits and comptime members. The loader drives iteration and
    handles validation, shard geometry, allocation, and io_uring."""

    comptime NUM_LAYERS: Int
    comptime LAYERS_OFF: Int
    comptime LAYER_STRIDE: Int

    @staticmethod
    fn global_weights() -> List[WeightDesc]: ...

    @staticmethod
    fn layer_weights(prefix: String, base: Int) -> List[WeightDesc]: ...

    @staticmethod
    fn total_weight_bytes() -> Int: ...

    @staticmethod
    fn total_arena_bytes() -> Int: ...


# ===--- Loader internals ---=== #

@fieldwise_init
struct ReadFragment(Copyable):
    """One contiguous file-to-memory transfer."""
    var file_offset: Int
    var dest: Int
    var length: Int


fn validate_weight(
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


fn emit_reads(
    desc: WeightDesc,
    file_data_start: Int,
    arena_base: Int,
    rank: Int,
    mut ops: List[ReadFragment],
):
    """Emit read fragments for a weight based on local vs global dims."""
    var dest = arena_base + desc.arena_offset
    var local_bytes = desc.local_rows * desc.local_cols * desc.element_bytes

    if desc.local_rows == desc.global_rows and desc.local_cols == desc.global_cols:
        # Replicated — single contiguous read
        ops.append(ReadFragment(
            file_offset=file_data_start, dest=dest, length=local_bytes,
        ))
    elif desc.local_rows != desc.global_rows:
        # Row shard — contiguous slice
        var row_start = rank * desc.local_rows
        var file_off = file_data_start + row_start * desc.global_cols * desc.element_bytes
        ops.append(ReadFragment(
            file_offset=file_off, dest=dest, length=local_bytes,
        ))
    else:
        # Col shard — strided reads, one per row
        var col_start = rank * desc.local_cols
        var local_row_bytes = desc.local_cols * desc.element_bytes
        for r in range(desc.local_rows):
            var src = file_data_start + (r * desc.global_cols + col_start) * desc.element_bytes
            var dst = dest + r * local_row_bytes
            ops.append(ReadFragment(
                file_offset=src, dest=dst, length=local_row_bytes,
            ))


# ===--- Load result ---=== #

@fieldwise_init
struct LoadResult(Movable):
    """Result of a successful load."""
    var arena_ptr: Int
    var arena_size: Int
    var bytes_loaded: Int
    var num_ops: Int


# ===--- Full loading pipeline ---=== #

fn load_safetensors[
    M: Loadable,
    io_depth: Int = DEFAULT_IO_DEPTH,
](
    path: Path,
    rank: Int = 0,
) -> Optional[LoadResult]:
    """Load a model from a safetensors file.

    Takes the model's weight_map, handles validation, shard geometry,
    arena allocation, and io_uring batch execution.
    Caller owns the returned arena memory.
    """
    # Parse safetensors header
    var header_opt = parse_safetensors_header(path)
    if not header_opt:
        return None
    var header = header_opt.take()

    # Allocate arena
    var arena_size = M.total_arena_bytes()
    var sys = linux.linux_sys()
    var arena_ptr = sys.sys_mmap[
        prot=linux.Prot.RW,
        flags=linux.MapFlag.PRIVATE | linux.MapFlag.ANONYMOUS,
    ](0, arena_size)
    if arena_ptr < 0:
        print("mmap failed, errno:", arena_ptr)
        return None

    # Build read plan — loader drives iteration using model's structural traits
    var weights = M.global_weights()
    for layer in range(M.NUM_LAYERS):
        var prefix = "model.layers." + String(layer) + "."
        var base = M.LAYERS_OFF + layer * M.LAYER_STRIDE
        var lw = M.layer_weights(prefix, base)
        for j in range(len(lw)):
            weights.append(lw[j].copy())

    var fragments = List[ReadFragment]()

    for i in range(len(weights)):
        var w = weights[i].copy()
        var meta_opt = header.tensors.get(w.name)
        if not meta_opt:
            print("missing tensor:", w.name)
            _ = sys.sys_munmap(arena_ptr, arena_size)
            return None
        var meta = meta_opt.value().copy()
        if not validate_weight(w, meta.dtype, meta.shape):
            _ = sys.sys_munmap(arena_ptr, arena_size)
            return None
        emit_reads(w, header.data_offset + meta.start, arena_ptr, rank, fragments)

    # Convert to ReadOps
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

    # io_uring execute
    var loader = IoLoader[io_depth]()
    if not loader:
        print("io_uring setup failed")
        _ = sys.sys_munmap(arena_ptr, arena_size)
        return None

    var paths = List[Path]()
    paths.append(path)
    var registered = loader.register_files(paths)
    if registered < 0:
        print("register_files failed, errno:", registered)
        _ = sys.sys_munmap(arena_ptr, arena_size)
        return None

    var bytes_loaded = 0

    @parameter
    fn on_complete(c: Completion):
        bytes_loaded += Int(c.result)

    var err = loader.process_queue_checked[on_complete](ops)
    if err:
        print_io_load_error(err.value())
        _ = sys.sys_munmap(arena_ptr, arena_size)
        return None

    return LoadResult(
        arena_ptr=arena_ptr,
        arena_size=arena_size,
        bytes_loaded=bytes_loaded,
        num_ops=num_fragments,
    )
