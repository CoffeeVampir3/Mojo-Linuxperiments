from safetensors.parser import parse_safetensors_header, TensorMeta
from safetensors.loader import IoLoader, ReadOp, Completion
from pathlib import Path
from memory import UnsafePointer, alloc

fn validate_f32_tensor(buf: UnsafePointer[UInt8, MutAnyOrigin], meta: TensorMeta, name: String) -> Bool:
    """Validate tensor values are sequential starting from base."""
    var ptr = buf.offset(meta.start).bitcast[Float32]()

    var total = 1
    for i in range(len(meta.shape)):
        total *= meta.shape[i]

    var base = ptr[0]
    for i in range(total):
        var expected = base + Float32(i)
        if ptr[i] != expected:
            print("  FAIL:", name, "index", i)
            return False

    print("  OK:", name, "shape", meta.shape, "base =", Int(base))
    return True

fn main():
    var path = Path("test_models/example_model.safetensors")

    # Parse header
    var header_opt = parse_safetensors_header(path)
    if not header_opt:
        print("Failed to parse header")
        return
    var header = header_opt.take()
    print("Parsed:", len(header.tensors), "tensors")

    var data_size = header.file_len - header.data_offset

    # Create loader
    var loader = IoLoader[64]()
    if not loader:
        print("Failed to create io_uring")
        return

    # Register file
    var paths = List[Path]()
    paths.append(path)
    var count = loader.register_files(paths)
    if count < 0:
        print("Failed to register files, errno:", count)
        return

    # Allocate buffer for entire data section
    var buf: UnsafePointer[UInt8, MutAnyOrigin] = alloc[UInt8](data_size)

    # Build per-tensor read ops and print/validate as each completes.
    var tensor_count = len(header.tensors)
    if tensor_count == 0:
        print("No tensors")
        buf.free()
        return

    var ops = List[ReadOp](capacity=tensor_count)
    var names = List[String](capacity=tensor_count)
    var metas = List[TensorMeta](capacity=tensor_count)

    var op_idx = 0
    for item in header.tensors.items():
        var name = item.key
        var meta = item.value.copy()
        var length = meta.byte_size()
        ops.append(ReadOp(
            file_idx=0,
            offset=header.data_offset + meta.start,
            length=length,
            dest=Int(buf) + meta.start,
            id=op_idx,
        ))
        names.append(name.copy())
        metas.append(meta^)
        op_idx += 1

    print("Submitting", len(ops), "tensor reads...\n")

    var passed = 0
    var failed = 0

    @parameter
    fn on_tensor_complete(c: Completion):
        var idx = Int(c.id)
        if idx < 0 or idx >= len(names):
            print("Completion with unknown id:", c.id, "res", c.result)
            return

        var meta = metas[idx].copy()
        var name = names[idx].copy()

        if c.result < 0:
            print("IO FAIL:", name, "-", meta.dtype, meta.shape, "errno:", c.result)
            if meta.dtype == DType.float32:
                failed += 1
            return

        print("IO DONE:", name, "-", meta.dtype, meta.shape, "-", c.result, "bytes")
        if meta.dtype == DType.float32:
            if validate_f32_tensor(buf, meta, name):
                passed += 1
            else:
                failed += 1
        else:
            print("  (unsupported dtype)")

    var done = loader.process_queue[on_tensor_complete](ops)
    if done < 0:
        print("Load failed")
        buf.free()
        return

    print()
    print("Validation:", passed, "passed,", failed, "failed")

    buf.free()
