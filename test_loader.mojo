from safetensors.parser import parse_safetensors_header, TensorMeta
from safetensors.loader import IoLoader, ReadOp, Completion
from pathlib import Path
from memory import UnsafePointer, alloc

fn validate_f32_tensor(buf: UnsafePointer[UInt8], meta: TensorMeta, name: String) -> Bool:
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
    var buf = alloc[UInt8](data_size)

    # Load entire data section
    var ops = List[ReadOp]()
    ops.append(ReadOp(
        file_idx=0,
        offset=header.data_offset,
        length=data_size,
        dest=Int(buf),
        user_data=0,
    ))
    _ = loader.submit(ops)
    var completions = loader.wait(min_complete=1)

    if len(completions) == 0 or completions[0].result < 0:
        print("Load failed")
        buf.free()
        return

    print("Loaded", completions[0].result, "bytes\n")

    # Validate tensor values
    var passed = 0
    var failed = 0
    for item in header.tensors.items():
        var name = item.key
        var meta = item.value.copy()
        print(name, "-", meta.dtype, meta.shape)
        if meta.dtype == DType.float32:
            if validate_f32_tensor(buf, meta, name):
                passed += 1
            else:
                failed += 1
        else:
            print("  (unsupported dtype)")

    print()
    print("Validation:", passed, "passed,", failed, "failed")

    buf.free()
