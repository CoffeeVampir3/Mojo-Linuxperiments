"""ButterQuant offline weight quantizer.

Implements butterquant.md Section IV exactly:
  1. Gamma absorption: W'[n,k] = W[n,k] * gamma[k]
  2. FWHT rotation on contraction dimension K per row
  3. Per-row symmetric i8 quantization: s[n] = absmax(row)/127

Phase 1 uses for_each_weight to collect weight metadata into a runtime list.
Phase 2 iterates that list in a normal loop — no closures, no capture issues.
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
from experimental.hadquant_impl import fwht_row
from simd_math import roundeven

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.float32]()

# Weight task kinds.
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
# Safetensors header
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


def fwht_rotate[block: Int](work: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        fwht_row[DType.float32, block](work + r * cols, cols)


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
# Quantizer
# =============================================================================


def quantize[M: WeightIterable, block: Int](
    source_path: Path, output_path: Path,
) -> Bool:
    var t0 = Int(perf_counter_ns())

    # Parse source header.
    var header_opt = parse_safetensors_header(source_path)
    if not header_opt:
        print("quantize: failed to parse source header")
        return False
    var header = header_opt.take()

    # --- Phase 1: collect weight tasks via comptime dispatch ---
    var tasks = List[WeightTask]()

    @parameter
    def collect[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        var name = prefix + String(T.NAME)
        comptime if conforms_to(T, Absorbed):
            tasks.append(WeightTask(name, ABSORBED))
        elif conforms_to(T, Gamma):
            tasks.append(WeightTask(name, GAMMA_QUANTIZE))
        elif conforms_to(T, Quantizable):
            tasks.append(WeightTask(name, QUANTIZE))
        else:
            tasks.append(WeightTask(name, PASSTHROUGH))

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
            var meta_opt = header.tensors.get(task.name)
            if not meta_opt:
                print("quantize: missing weight " + task.name)
                return False
            var meta = meta_opt.value().copy()
            var rows = meta.shape[0]
            var cols = meta.shape[1] if len(meta.shape) > 1 else 1
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
            var meta_opt = header.tensors.get(task.name)
            if not meta_opt:
                continue  # not in source (e.g. row scales) — skip
            var meta = meta_opt.value().copy()
            var byte_size = meta.end - meta.start
            var rows = meta.shape[0]
            var cols = meta.shape[1] if len(meta.shape) > 1 else 1
            entries.append(OutputEntry(task.name, meta.dtype, rows, cols, offset, offset + byte_size))
            offset += byte_size
            if byte_size > max_elements:
                max_elements = byte_size

    # --- Write header ---
    var header_bytes = build_header(entries)
    var data_start = len(header_bytes)

    var rio = RingIO()
    var paths = List[Path]()
    paths.append(source_path)
    paths.append(output_path)
    try:
        _ = rio.ring.register_files[ReadWriteMode](paths)
    except:
        print("quantize: failed to register files")
        return False

    if not rio.write(1, 0, PtrU8(unsafe_from_address=Int(UnsafePointer(to=header_bytes[0]))), len(header_bytes)):
        print("quantize: failed to write header")
        return False
    print("header: " + String(len(header_bytes)) + " bytes, " + String(len(entries)) + " entries")

    # --- Phase 2: process weights in a normal loop ---
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
            var meta = header.tensors.get(task.name).value().copy()
            var byte_size = meta.end - meta.start
            if not rio.read(0, header.data_offset + meta.start, read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False
            bf16_to_f32(read_buf.bitcast[UInt8](), gamma_buf, meta.shape[0])
            has_gamma = True
            print("  absorbed: " + task.name)

        elif task.kind == GAMMA_QUANTIZE or task.kind == QUANTIZE:
            var meta = header.tensors.get(task.name).value().copy()
            var rows = meta.shape[0]
            var cols = meta.shape[1] if len(meta.shape) > 1 else 1
            var byte_size = meta.end - meta.start

            if not rio.read(0, header.data_offset + meta.start, read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False

            bf16_to_f32(read_buf.bitcast[UInt8](), work, rows * cols)

            if task.kind == GAMMA_QUANTIZE and has_gamma:
                absorb_gamma(work, gamma_buf, rows, cols)

            fwht_rotate[block](work, rows, cols)
            quantize_rows(work, qi, scales_buf, rows, cols)

            # Write I8 weight.
            var we = entries[entry_idx]
            if not rio.write(1, data_start + we.data_start, qi.bitcast[UInt8](), we.byte_size()):
                print("quantize: failed to write " + task.name)
                return False
            entry_idx += 1
            total_bytes += we.byte_size()

            # Write F32 scales.
            var se = entries[entry_idx]
            if not rio.write(1, data_start + se.data_start, scales_buf.bitcast[UInt8](), se.byte_size()):
                print("quantize: failed to write " + task.name + "_scale")
                return False
            entry_idx += 1
            total_bytes += se.byte_size()

            num_quantized += 1
            print("  quantized: " + task.name + " [" + String(rows) + "x" + String(cols) + "]")

            if task.kind == QUANTIZE:
                has_gamma = False

        elif task.kind == PASSTHROUGH:
            var meta_opt = header.tensors.get(task.name)
            if not meta_opt:
                continue
            var meta = meta_opt.value().copy()
            var byte_size = meta.end - meta.start

            if not rio.read(0, header.data_offset + meta.start, read_buf.bitcast[UInt8](), byte_size):
                print("quantize: failed to read " + task.name)
                return False

            var pe = entries[entry_idx]
            if not rio.write(1, data_start + pe.data_start, read_buf.bitcast[UInt8](), byte_size):
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
