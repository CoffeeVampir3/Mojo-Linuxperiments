"""Offline quantization pipeline.

A pipeline is a function that transforms a source buffer into quantized
weight + scale buffers. Composition is just calling atomic operations
in sequence — the pipeline function IS the specification.

The orchestrator handles safetensors I/O, double-buffered streaming,
and header construction. It receives a pipeline function and calls it
per tensor. It knows nothing about what the pipeline does internally.

Usage:
    from modeling.smollm2_int8ch_tp import Int8TPModel

    var ok = Int8TPModel[1].quantize(source_path, output_path)
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path
from std.time import perf_counter_ns

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader, TensorMeta,
    HEADER_LEN_BYTES,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion, ReadWriteMode, RingError,
)
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, Quantizable,
    WeightIterable, WeightDesc, weight_desc,
)
from .pipelines import PipelineFn, Timer


# =============================================================================
# Safetensors writer utilities
# =============================================================================


def dtype_to_st_string(dt: DType) -> String:
    if dt == DType.int8:
        return "I8"
    if dt == DType.float32:
        return "F32"
    if dt == DType.bfloat16:
        return "BF16"
    if dt == DType.float16:
        return "F16"
    if dt == DType.uint8:
        return "U8"
    if dt == DType.int32:
        return "I32"
    if dt == DType.int64:
        return "I64"
    if dt == DType.float64:
        return "F64"
    return "UNKNOWN"


@fieldwise_init
struct OutputTensor(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape: List[Int]
    var data_start: Int
    var data_end: Int
    var src_file_offset: Int
    var src_length: Int
    var quantize: Bool
    var is_scale: Bool


def build_header_json(entries: List[OutputTensor]) -> String:
    var s = String("{")
    for i in range(len(entries)):
        if i > 0:
            s += ","
        var e = entries[i].copy()
        s += '"' + e.name + '":{"dtype":"' + dtype_to_st_string(e.dtype) + '","shape":['
        for j in range(len(e.shape)):
            if j > 0:
                s += ","
            s += String(e.shape[j])
        s += '],"data_offsets":[' + String(e.data_start) + "," + String(e.data_end) + "]}"
    s += "}"
    return s^


def write_u64_le(value: UInt64) -> InlineArray[UInt8, 8]:
    var buf = InlineArray[UInt8, 8](fill=0)
    for i in range(8):
        buf[i] = UInt8((value >> UInt64(i * 8)) & 0xFF)
    return buf


# =============================================================================
# Output layout computation
# =============================================================================


def compute_output_layout(
    weights: List[WeightDesc],
    header: SafetensorsHeader,
    weight_dtype: DType, weight_element_bits: Int,
    scale_dtype: DType, scale_element_bits: Int,
) -> List[OutputTensor]:
    """Build output manifest from model spec weights + safetensors header.

    The model spec (via WeightDesc.quantizable) determines which tensors
    to quantize. The safetensors header provides source file offsets.
    """
    var entries = List[OutputTensor]()
    var offset = 0

    for i in range(len(weights)):
        var w = weights[i].copy()
        var meta_opt = header.tensors.get(w.name)
        if not meta_opt:
            continue
        var meta = meta_opt.value().copy()

        if w.quantizable:
            var rows = meta.shape[0]
            var cols = meta.shape[1]

            var weight_bytes = rows * cols * weight_element_bits // 8
            entries.append(OutputTensor(
                name=w.name,
                dtype=weight_dtype, shape=meta.shape.copy(),
                data_start=offset, data_end=offset + weight_bytes,
                src_file_offset=header.data_offset + meta.start,
                src_length=meta.end - meta.start,
                quantize=True, is_scale=False,
            ))
            offset += weight_bytes

            var sc_bytes = rows * scale_element_bits // 8
            var shape_list = List[Int]()
            shape_list.append(rows)
            shape_list.append(1)
            entries.append(OutputTensor(
                name=w.name + "_scale",
                dtype=scale_dtype, shape=shape_list^,
                data_start=offset, data_end=offset + sc_bytes,
                src_file_offset=-1, src_length=0,
                quantize=False, is_scale=True,
            ))
            offset += sc_bytes
        else:
            var byte_size = meta.end - meta.start
            entries.append(OutputTensor(
                name=w.name,
                dtype=meta.dtype, shape=meta.shape.copy(),
                data_start=offset, data_end=offset + byte_size,
                src_file_offset=header.data_offset + meta.start,
                src_length=byte_size,
                quantize=False, is_scale=False,
            ))
            offset += byte_size

    return entries^


# =============================================================================
# I/O helpers
# =============================================================================


def submit_write(mut ring: IoRing, op: WriteOp) raises RingError:
    _ = ring.submit_one(op)
    var completions = ring.wait()
    if len(completions) == 0:
        raise RingError(op.id, -1, "write: no completion")
    if completions[0].result < 0:
        raise RingError(op.id, Int(completions[0].result), "write")


def submit_read(mut ring: IoRing, op: ReadOp) raises RingError:
    _ = ring.submit_one(op)
    var completions = ring.wait()
    if len(completions) == 0:
        raise RingError(op.id, -1, "read: no completion")
    if completions[0].result < 0:
        raise RingError(op.id, Int(completions[0].result), "read")
    var got = Int(completions[0].result)
    if got != op.expected_bytes():
        raise RingError(op.id, -1, "short read")


# =============================================================================
# Orchestrator
# =============================================================================


def quantize[M: WeightIterable](
    pipeline: PipelineFn,
    weight_dtype: DType, weight_element_bits: Int,
    scale_dtype: DType, scale_element_bits: Int,
    source_path: Path, output_path: Path,
    num_workers: Int = 4,
) -> Bool:

    # --- Phase 1: collect weights from model spec, parse source, compute layout ---

    var weights = List[WeightDesc]()

    @parameter
    def collect[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        weights.append(weight_desc[T](prefix, base))

    M.for_each_weight[collect]()

    var header_opt = parse_safetensors_header(source_path)
    if not header_opt:
        print("quantize: failed to parse source header")
        return False
    var header = header_opt.take()

    var entries = compute_output_layout(
        weights, header, weight_dtype, weight_element_bits, scale_dtype, scale_element_bits,
    )
    print(
        "quantize: " + String(len(header.tensors)) + " source tensors -> "
        + String(len(entries)) + " output tensors"
    )

    # --- Phase 2: build and write output header ---

    var json = build_header_json(entries)
    var json_len = len(json)
    var header_prefix = write_u64_le(UInt64(json_len))

    var header_buf_size = HEADER_LEN_BYTES + json_len
    var header_buf = alloc[UInt8](header_buf_size)
    for i in range(HEADER_LEN_BYTES):
        header_buf[i] = header_prefix[i]
    var json_bytes = json.as_bytes()
    memcpy(dest=header_buf + HEADER_LEN_BYTES, src=json_bytes.unsafe_ptr(), count=json_len)

    var data_start = HEADER_LEN_BYTES + json_len

    var read_ring = IoRing[]()
    var write_ring = IoRing[]()
    if not read_ring or not write_ring:
        print("quantize: io_uring setup failed")
        header_buf.free()
        return False

    var paths = List[Path]()
    paths.append(source_path)
    paths.append(output_path)
    try:
        _ = read_ring.register_files[ReadWriteMode](paths)
        _ = write_ring.register_files[ReadWriteMode](paths)
    except err:
        print("quantize: register_files failed:", err)
        header_buf.free()
        return False

    try:
        submit_write(write_ring, WriteOp(
            file_idx=1, offset=0, length=header_buf_size, src=header_buf, id=0,
        ))
    except err:
        print("quantize: header write failed:", err)
        header_buf.free()
        return False
    header_buf.free()

    # --- Phase 3: stream tensors through pipeline ---

    var max_src_bytes = 0
    var max_weight_bytes = 0
    var max_scale_bytes = 0
    for i in range(len(entries)):
        if entries[i].src_length > max_src_bytes:
            max_src_bytes = entries[i].src_length
        if entries[i].quantize:
            max_weight_bytes = max(max_weight_bytes, entries[i].data_end - entries[i].data_start)
        if entries[i].is_scale:
            max_scale_bytes = max(max_scale_bytes, entries[i].data_end - entries[i].data_start)

    var buf_a = alloc[UInt8](max(max_src_bytes, 1))
    var buf_b = alloc[UInt8](max(max_src_bytes, 1))
    var weight_buf = alloc[UInt8](max(max_weight_bytes, 1))
    var scale_buf = alloc[UInt8](max(max_scale_bytes, 1))

    var pool = BurstPool[](num_workers)
    if not pool:
        print("quantize: BurstPool creation failed")
        return False

    var pool_ptr = UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    )

    var io_entries = List[Int]()
    for i in range(len(entries)):
        if not entries[i].is_scale:
            io_entries.append(i)

    var op_id = 1
    var ok = True
    var quant_count = 0
    var quant_ns = Int(0)
    var quant_bytes = Int(0)
    var total_start = perf_counter_ns()

    print("quantize: streaming " + String(len(io_entries)) + " tensors")

    # Prime: read first tensor
    if len(io_entries) > 0:
        var first = entries[io_entries[0]].copy()
        try:
            submit_read(read_ring, ReadOp(
                file_idx=0, offset=first.src_file_offset,
                length=first.src_length, dest=buf_a, id=op_id,
            ))
        except err:
            print("quantize: initial read failed:", err)
            ok = False
        op_id += 1

    var cur_buf = buf_a
    var next_buf = buf_b

    for idx in range(len(io_entries)):
        if not ok:
            break

        var ei = io_entries[idx]
        var entry = entries[ei].copy()

        # Submit async read for next tensor
        var read_pending = False
        if idx + 1 < len(io_entries):
            var next_entry = entries[io_entries[idx + 1]].copy()
            try:
                _ = read_ring.submit_one(ReadOp(
                    file_idx=0, offset=next_entry.src_file_offset,
                    length=next_entry.src_length, dest=next_buf, id=op_id,
                ))
                read_pending = True
            except err:
                print("quantize: read submit failed:", err)
                ok = False
                break
            op_id += 1

        var tensor_t0 = perf_counter_ns()

        if entry.quantize:
            var rows = entry.shape[0]
            var cols = entry.shape[1]

            # Run the pipeline
            var pipe_t0 = perf_counter_ns()
            pipeline(pool_ptr, cur_buf, weight_buf, scale_buf, rows, cols)
            var pipe_ns = Int(perf_counter_ns() - pipe_t0)
            var pipe_us = pipe_ns // 1_000
            quant_ns += pipe_ns
            quant_count += 1
            quant_bytes += entry.src_length

            # Write weight
            try:
                submit_write(write_ring, WriteOp(
                    file_idx=1, offset=data_start + entry.data_start,
                    length=entry.data_end - entry.data_start,
                    src=weight_buf, id=op_id,
                ))
            except err:
                print("quantize: weight write failed:", err)
                ok = False
                break
            op_id += 1

            # Write scale
            var scale_entry = entries[ei + 1].copy()
            try:
                submit_write(write_ring, WriteOp(
                    file_idx=1, offset=data_start + scale_entry.data_start,
                    length=scale_entry.data_end - scale_entry.data_start,
                    src=scale_buf, id=op_id,
                ))
            except err:
                print("quantize: scale write failed:", err)
                ok = False
                break
            op_id += 1

            var total_us = Int(perf_counter_ns() - tensor_t0) // 1_000
            print(
                "  [" + String(idx + 1) + "/" + String(len(io_entries)) + "]"
                + " quantize " + entry.name
                + " [" + String(rows) + "x" + String(cols) + "]"
                + " pipe=" + String(pipe_us) + "us"
                + " total=" + String(total_us) + "us"
            )
        else:
            # Passthrough
            try:
                submit_write(write_ring, WriteOp(
                    file_idx=1, offset=data_start + entry.data_start,
                    length=entry.data_end - entry.data_start,
                    src=cur_buf, id=op_id,
                ))
            except err:
                print("quantize: passthrough write failed:", err)
                ok = False
                break
            op_id += 1

            var total_us = Int(perf_counter_ns() - tensor_t0) // 1_000
            var size_kb = (entry.data_end - entry.data_start) // 1024
            print(
                "  [" + String(idx + 1) + "/" + String(len(io_entries)) + "]"
                + " passthrough " + entry.name
                + " " + String(size_kb) + "KB"
                + " total=" + String(total_us) + "us"
            )

        # Wait for next read before swapping
        if read_pending:
            try:
                var completions = read_ring.wait()
                if len(completions) == 0 or completions[0].result < 0:
                    print("quantize: prefetch read failed")
                    ok = False
                    break
            except err:
                print("quantize: read wait failed:", err)
                ok = False
                break

        var tmp = cur_buf
        cur_buf = next_buf
        next_buf = tmp

    buf_a.free()
    buf_b.free()
    weight_buf.free()
    scale_buf.free()

    var total_ns = Int(perf_counter_ns() - total_start)

    if ok:
        # Validate output file size
        var last_entry = entries[len(entries) - 1].copy()
        var expected_size = data_start + last_entry.data_end
        try:
            with open(output_path, "r") as f:
                var actual_size = Int(f.seek(0, 2))
                if actual_size == expected_size:
                    print("quantize: size ok (" + String(actual_size) + " bytes)")
                else:
                    print("quantize: SIZE MISMATCH expected=" + String(expected_size) + " actual=" + String(actual_size))
                    ok = False
        except:
            print("quantize: could not verify output size")

        var total_ms = total_ns // 1_000_000
        var avg_us = quant_ns // max(quant_count, 1) // 1_000
        var throughput_mbs = quant_bytes * 1_000 // max(quant_ns, 1)
        print(
            "quantize: " + String(quant_count) + " layers quantized"
            + ", avg " + String(avg_us) + "us/layer"
            + ", " + String(throughput_mbs) + " MB/s"
            + ", total " + String(total_ms) + "ms"
        )
        print("quantize: done, output at " + String(output_path))
    return ok
