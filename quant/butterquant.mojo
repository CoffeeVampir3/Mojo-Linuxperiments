"""ButterQuant offline weight quantizer.

Per quantizable weight:
  1. Gamma absorption (if preceding norm): W'[n,k] = W[n,k] * gamma[k]
  2. FWHT rotation on contraction dimension K per row (block = largest pow2 factor of K, cap 256)
  3. DC correction for post-nonlinearity weights (block < 256): element 0 of each block *= DC_SCALE
  4. Per-row symmetric i8 quantization: s[n] = absmax(row)/127

Supports multi-shard checkpoints. The for_each_weight interface uses the
3-arg signature (prefix, base, target_rank).
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader, TensorMeta,
    HEADER_LEN_BYTES,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion, ReadWriteMode, ReadMode, RingError,
)
from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Quantizable, Gamma, Absorbed,
    WeightIterable,
)
from notstdcollections import HeapMoveArray
from experimental.hadquant_impl import fwht_row
from simd_math import roundeven

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.float32]()
comptime DC_SCALE = Float32(0.5)
comptime MAX_FWHT_BLOCK = 256

comptime ABSORBED = 0
comptime GAMMA_QUANTIZE = 1
comptime QUANTIZE = 2
comptime PASSTHROUGH = 3


# =============================================================================
# Weight task — runtime descriptor collected from comptime dispatch
# =============================================================================


@fieldwise_init
struct WeightTask(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var kind: Int
    var cols: Int


# =============================================================================
# Output entry — one tensor in the output safetensors
# =============================================================================


@fieldwise_init
struct OutputEntry(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var dtype: DType
    var rows: Int
    var cols: Int
    var data_start: Int
    var data_end: Int

    def byte_size(self) -> Int:
        return self.data_end - self.data_start


# =============================================================================
# Safetensors header builder
# =============================================================================


def dtype_string(dt: DType) -> String:
    if dt == DType.int8: return "I8"
    if dt == DType.float32: return "F32"
    if dt == DType.bfloat16: return "BF16"
    if dt == DType.float16: return "F16"
    if dt == DType.uint8: return "U8"
    return "UNKNOWN"


def build_header(entries: List[OutputEntry]) -> List[UInt8]:
    var json = String("{")
    for i in range(len(entries)):
        if i > 0:
            json += ","
        var e = entries[i]
        json += '"' + e.name + '":{"dtype":"' + dtype_string(e.dtype)
        json += '","shape":[' + String(e.rows)
        if e.cols > 0:
            json += "," + String(e.cols)
        json += '],"data_offsets":[' + String(e.data_start)
        json += "," + String(e.data_end) + "]}"
    json += "}"

    var json_bytes = json.as_bytes()
    var json_len = len(json_bytes)

    var buf = List[UInt8]()
    var header_len = UInt64(json_len)
    for i in range(8):
        buf.append(UInt8((header_len >> UInt64(i * 8)) & 0xFF))
    for i in range(json_len):
        buf.append(json_bytes[i])
    return buf^


# =============================================================================
# FWHT block size computation
# =============================================================================


def fwht_block_for_cols(cols: Int) -> Int:
    """Largest power-of-2 factor of cols, capped at MAX_FWHT_BLOCK."""
    var block = 1
    var c = cols
    while c % 2 == 0 and block < MAX_FWHT_BLOCK:
        block *= 2
        c //= 2
    return block


# =============================================================================
# Core transforms
# =============================================================================


def absorb_gamma(work: PtrF32, gamma_buf: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        var row = work + r * cols
        var k = 0
        while k + WIDTH <= cols:
            (row + k).store(
                (row + k).load[width=WIDTH]() * (gamma_buf + k).load[width=WIDTH]()
            )
            k += WIDTH
        while k < cols:
            row[k] = row[k] * gamma_buf[k]
            k += 1


def fwht_rotate(work: PtrF32, rows: Int, cols: Int, block: Int):
    for r in range(rows):
        if block == 256:
            fwht_row[DType.float32, 256](work + r * cols, cols)
        elif block == 128:
            fwht_row[DType.float32, 128](work + r * cols, cols)
        elif block == 64:
            fwht_row[DType.float32, 64](work + r * cols, cols)
        else:
            fwht_row[DType.float32, 32](work + r * cols, cols)


def apply_dc_correction(work: PtrF32, rows: Int, cols: Int, block: Int):
    """Scale element 0 of each FWHT block by DC_SCALE. Applied to post-nonlinearity
    weights (down projections) where the DC component is a systematic outlier."""
    var num_blocks = cols // block
    for r in range(rows):
        var row = work + r * cols
        for b in range(num_blocks):
            row[b * block] *= DC_SCALE


def quantize_rows(work: PtrF32, qi: PtrI8, scales_buf: PtrF32, rows: Int, cols: Int):
    comptime lo = SIMD[DType.float32, WIDTH](-128.0)
    comptime hi = SIMD[DType.float32, WIDTH](127.0)

    for r in range(rows):
        var src = work + r * cols
        var dst = qi + r * cols

        var vmax = SIMD[DType.float32, WIDTH](0)
        var k = 0
        while k + WIDTH <= cols:
            vmax = max(vmax, (src + k).load[width=WIDTH]().__abs__())
            k += WIDTH
        var amax = vmax.reduce_max()
        while k < cols:
            var a = src[k]
            if a < Float32(0):
                a = -a
            if a > amax:
                amax = a
            k += 1

        scales_buf[r] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        var vinv = SIMD[DType.float32, WIDTH](inv)

        k = 0
        while k + WIDTH <= cols:
            var v = (src + k).load[width=WIDTH]()
            (dst + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
            k += WIDTH
        while k < cols:
            var v = roundeven[DType.float32, 1](src[k] * inv)
            dst[k] = min(max(v, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()
            k += 1


def bf16_to_f32(src: PtrU8, dst: PtrF32, count: Int):
    comptime w = simd_width_of[DType.bfloat16]()
    var bp = src.bitcast[Scalar[DType.bfloat16]]()
    var k = 0
    while k + w <= count:
        (dst + k).store((bp + k).load[width=w]().cast[DType.float32]())
        k += w
    while k < count:
        dst[k] = Float32(bp[k])
        k += 1


# =============================================================================
# I/O wrapper
# =============================================================================


struct RingIO(Movable):
    var ring: IoRing[]

    def __init__(out self):
        self.ring = IoRing[]()

    def read(mut self, file_idx: Int, offset: Int,
             dest: PtrU8, length: Int) -> Bool:
        try:
            _ = self.ring.submit_one(ReadOp(
                file_idx=file_idx, offset=offset, length=length,
                dest=dest, id=0,
            ))
            var completions = self.ring.wait()
            return len(completions) > 0 and Int(completions[0].result) == length
        except:
            return False

    def write(mut self, file_idx: Int, offset: Int,
              src: PtrU8, length: Int) -> Bool:
        try:
            _ = self.ring.submit_one(WriteOp(
                file_idx=file_idx, offset=offset, length=length,
                src=src, id=0,
            ))
            var completions = self.ring.wait()
            return len(completions) > 0 and Int(completions[0].result) == length
        except:
            return False


# =============================================================================
# Multi-shard tensor lookup
# =============================================================================


def find_tensor(headers: HeapMoveArray[SafetensorsHeader], name: String) -> Tuple[Int, TensorMeta]:
    """Find a tensor across multiple shard headers. Returns (shard_index, meta)."""
    for i in range(len(headers)):
        var opt = headers[i].tensors.get(name)
        if opt:
            return (i, opt.value().copy())
    return (-1, TensorMeta(DType.uint8, List[Int](), 0, 0))


def fold_shape(shape: List[Int]) -> Tuple[Int, Int]:
    """Fold 3D shape [a, b, c] → (a*b, c). Pass through 2D and 1D."""
    if len(shape) == 3:
        return (shape[0] * shape[1], shape[2])
    elif len(shape) == 2:
        return (shape[0], shape[1])
    else:
        return (shape[0], 1)


def discover_shards(dir_path: Path) -> List[Path]:
    var shards = List[Path]()
    try:
        for entry in dir_path.listdir():
            var name = String(entry)
            if name.endswith(".safetensors") and name.startswith("model-"):
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


# =============================================================================
# Quantizer
# =============================================================================


def quantize[M: WeightIterable](
    source_dir: Path, output_path: Path,
) -> Bool:
    """Quantize a multi-shard checkpoint directory to a single output safetensors.

    Discovers all model-*.safetensors in source_dir, parses headers, then
    processes each weight according to its trait tags. Per-weight FWHT block
    is computed from column count. DC correction applied for small blocks.
    """
    var t0 = Int(perf_counter_ns())

    # Discover and parse all source shards.
    var shard_paths = discover_shards(source_dir)
    if len(shard_paths) == 0:
        print("quantize: no shards found in " + String(source_dir))
        return False
    print("found " + String(len(shard_paths)) + " shard(s)")

    var headers = HeapMoveArray[SafetensorsHeader](len(shard_paths))
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            print("quantize: failed to parse " + String(shard_paths[i]))
            return False
        headers.push(h.take())

    # --- Phase 1: collect weight tasks via comptime dispatch ---
    var tasks = List[WeightTask]()

    @parameter
    def collect[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int, target_rank: Int):
        var name = prefix + String(T.NAME)
        comptime cols = T.COLS
        comptime if conforms_to(T, Absorbed):
            tasks.append(WeightTask(name, ABSORBED, cols))
        elif conforms_to(T, Gamma):
            tasks.append(WeightTask(name, GAMMA_QUANTIZE, cols))
        elif conforms_to(T, Quantizable):
            tasks.append(WeightTask(name, QUANTIZE, cols))
        else:
            tasks.append(WeightTask(name, PASSTHROUGH, cols))

    M.for_each_weight[collect]()

    # --- Build output layout ---
    var entries = List[OutputEntry]()
    var offset = 0
    var max_elements = 0
    var max_cols = 0

    for i in range(len(tasks)):
        var task = tasks[i]
        if task.kind == ABSORBED:
            continue

        if task.kind == GAMMA_QUANTIZE or task.kind == QUANTIZE:
            var result = find_tensor(headers, task.name)
            if result[0] < 0:
                print("quantize: missing weight " + task.name)
                return False
            var meta = result[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            var weight_bytes = rows * cols
            entries.append(OutputEntry(task.name, DType.int8, rows, cols, offset, offset + weight_bytes))
            offset += weight_bytes
            var scale_bytes = rows * 4
            entries.append(OutputEntry(task.name + "_scale", DType.float32, rows, 1, offset, offset + scale_bytes))
            offset += scale_bytes
            if rows * cols > max_elements:
                max_elements = rows * cols
            if cols > max_cols:
                max_cols = cols

        elif task.kind == PASSTHROUGH:
            var result = find_tensor(headers, task.name)
            if result[0] < 0:
                continue
            var meta = result[1].copy()
            var byte_size = meta.end - meta.start
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            entries.append(OutputEntry(task.name, meta.dtype, rows, cols, offset, offset + byte_size))
            offset += byte_size
            if byte_size > max_elements:
                max_elements = byte_size

    # --- Write header ---
    var header_bytes = build_header(entries)
    var data_start = len(header_bytes)

    var rio = RingIO()
    var all_paths = List[Path]()
    for i in range(len(shard_paths)):
        all_paths.append(shard_paths[i])
    all_paths.append(output_path)
    var output_file_idx = len(shard_paths)
    try:
        _ = rio.ring.register_files[ReadWriteMode](all_paths)
    except:
        print("quantize: failed to register files")
        return False

    if not rio.write(output_file_idx, 0, PtrU8(unsafe_from_address=Int(UnsafePointer(to=header_bytes[0]))), len(header_bytes)):
        print("quantize: failed to write header")
        return False
    print("header: " + String(len(header_bytes)) + " bytes, " + String(len(entries)) + " entries")

    # --- Phase 2: process weights ---
    var read_buf = alloc[UInt8](max_elements * 2)
    var work = alloc[Scalar[DType.float32]](max_elements)
    var qi = alloc[Scalar[DType.int8]](max_elements)
    var scales_buf = alloc[Scalar[DType.float32]](max_elements)
    var gamma_buf = alloc[Scalar[DType.float32]](max_cols)
    var has_gamma = False
    var entry_idx = 0
    var total_bytes = 0
    var num_quantized = 0

    for i in range(len(tasks)):
        var task = tasks[i]

        if task.kind == ABSORBED:
            var result = find_tensor(headers, task.name)
            var shard_idx = result[0]
            var meta = result[1].copy()
            var byte_size = meta.end - meta.start
            if not rio.read(shard_idx, headers[shard_idx].data_offset + meta.start,
                    read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False
            bf16_to_f32(read_buf.bitcast[UInt8](), gamma_buf, meta.shape[0])
            has_gamma = True
            print("  absorbed: " + task.name)

        elif task.kind == GAMMA_QUANTIZE or task.kind == QUANTIZE:
            var result = find_tensor(headers, task.name)
            var shard_idx = result[0]
            var meta = result[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            var byte_size = meta.end - meta.start

            if not rio.read(shard_idx, headers[shard_idx].data_offset + meta.start,
                    read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False

            bf16_to_f32(read_buf.bitcast[UInt8](), work, rows * cols)

            if task.kind == GAMMA_QUANTIZE and has_gamma:
                absorb_gamma(work, gamma_buf, rows, cols)

            var block = fwht_block_for_cols(cols)
            fwht_rotate(work, rows, cols, block)

            if block < MAX_FWHT_BLOCK:
                apply_dc_correction(work, rows, cols, block)
                print("  quantized: " + task.name + " [" + String(rows) + "x" + String(cols)
                    + "] block=" + String(block) + " DC-corrected")
            else:
                print("  quantized: " + task.name + " [" + String(rows) + "x" + String(cols)
                    + "] block=" + String(block))

            quantize_rows(work, qi, scales_buf, rows, cols)

            var we = entries[entry_idx]
            if not rio.write(output_file_idx, data_start + we.data_start, qi.bitcast[UInt8](), we.byte_size()):
                print("quantize: failed to write " + task.name)
                return False
            entry_idx += 1
            total_bytes += we.byte_size()

            var se = entries[entry_idx]
            if not rio.write(output_file_idx, data_start + se.data_start, scales_buf.bitcast[UInt8](), se.byte_size()):
                print("quantize: failed to write " + task.name + "_scale")
                return False
            entry_idx += 1
            total_bytes += se.byte_size()

            num_quantized += 1

            if task.kind == QUANTIZE:
                has_gamma = False

        elif task.kind == PASSTHROUGH:
            var result = find_tensor(headers, task.name)
            if result[0] < 0:
                continue
            var shard_idx = result[0]
            var meta = result[1].copy()
            var byte_size = meta.end - meta.start

            if not rio.read(shard_idx, headers[shard_idx].data_offset + meta.start,
                    read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False

            var pe = entries[entry_idx]
            if not rio.write(output_file_idx, data_start + pe.data_start, read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to write " + task.name)
                return False
            entry_idx += 1
            total_bytes += byte_size
            print("  passthrough: " + task.name)

    read_buf.free()
    work.free()
    qi.free()
    scales_buf.free()
    gamma_buf.free()

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("quantize: " + String(num_quantized) + " weights, "
        + String(total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    return True
