"""ButterQuant offline weight quantizer.

Two-pass visitor design:
  1. Plan: model calls `describe_quantization(planner)` — the planner validates
     each task against safetensors headers, computes output offsets, emits entries.
  2. Execute: model calls `describe_quantization(executor)` — the executor reads,
     converts, quantizes, and writes each weight using typed Converter access.

Both passes use the `TaskVisitor` trait. Each `visitor.quantize[T: QuantizeSpec](task)`
call is statically dispatched — source format flows through comptime trait members,
no runtime switches anywhere.
"""

from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns
from std.math import max
from std.os import abort

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, ReadWriteMode, Completion,
)
from modeling.model_spec import QuantizeSpec, TaskVisitor
from modeling.loader import discover_shards
from std.memory.unsafe_pointer import alloc
from experimental3.kernels.fwht import fwht_block, fwht_row
from simd_math import quantize_i8, quantize_i8_scalar, sqrt as simd_sqrt
from quant.source_format import Converter

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.float32]()
comptime DEFAULT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK_BYTES = 16 * 1024 * 1024


# =============================================================================
# Output entry + safetensors header builder
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


def dtype_string(dt: DType) -> String:
    if dt == DType.int8: return "I8"
    if dt == DType.float32: return "F32"
    if dt == DType.bfloat16: return "BF16"
    if dt == DType.float16: return "F16"
    if dt == DType.uint8: return "U8"
    if dt == DType.float8_e4m3fn: return "F8_E4M3"
    return "UNKNOWN"


def build_header(entries: List[OutputEntry]) -> List[UInt8]:
    var json = String("{")
    for i in range(len(entries)):
        if i > 0:
            json += ","
        ref e = entries[i]
        json += '"' + e.name + '":{"dtype":"' + dtype_string(e.dtype)
        json += '","shape":[' + String(e.rows)
        if e.cols > 0:
            json += "," + String(e.cols)
        json += '],"data_offsets":[' + String(e.data_start)
        json += "," + String(e.data_end) + "]}"
    json += "}"

    var json_bytes = json.as_bytes()
    var json_len = len(json_bytes)

    var buf = List[UInt8](capacity=8 + json_len)
    var header_len = UInt64(json_len)
    for i in range(8):
        buf.append(UInt8((header_len >> UInt64(i * 8)) & 0xFF))
    for i in range(json_len):
        buf.append(json_bytes[i])
    return buf^


# =============================================================================
# io_uring read/write wrapper
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
            var c = self.ring.drain_one()
            return Int(c.result) == length
        except:
            return False

    def write(mut self, file_idx: Int, offset: Int,
              src: PtrU8, length: Int) -> Bool:
        try:
            _ = self.ring.submit_one(WriteOp(
                file_idx=file_idx, offset=offset, length=length,
                src=src, id=0,
            ))
            var c = self.ring.drain_one()
            return Int(c.result) == length
        except:
            return False


# =============================================================================
# Safetensors lookup
# =============================================================================


@fieldwise_init
struct TensorLocation(Copyable, Movable, ImplicitlyCopyable):
    var shard: Int
    var start: Int
    var end: Int
    var dtype: DType
    var rows: Int
    var cols: Int

    def byte_size(self) -> Int:
        return self.end - self.start


def fold_shape(shape: List[Int]) -> Tuple[Int, Int]:
    if len(shape) == 3:
        return (shape[0] * shape[1], shape[2])
    elif len(shape) == 2:
        return (shape[0], shape[1])
    else:
        return (shape[0], 1)


def find_tensor(ref headers: List[SafetensorsHeader],
                name: String) -> Optional[TensorLocation]:
    for i in range(len(headers)):
        try:
            ref m = headers[i].tensors[name]
            var rc = fold_shape(m.shape)
            return TensorLocation(i, m.start, m.end, m.dtype, rc[0], rc[1])
        except:
            pass
    return None


@always_inline
def u8_addr(mut buf: List[UInt8]) -> PtrU8:
    return PtrU8(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def f32_addr(mut buf: List[Float32]) -> PtrF32:
    return PtrF32(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def i8_addr(mut buf: List[Scalar[DType.int8]]) -> PtrI8:
    return PtrI8(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def bf16_addr(mut buf: List[Scalar[DType.bfloat16]]) -> PtrBF16:
    return PtrBF16(unsafe_from_address=Int(buf.unsafe_ptr()))


# =============================================================================
# Core transforms
# =============================================================================


@always_inline
def apply_gamma_in_place(work_row: PtrF32, gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        (work_row + k).store(
            (work_row + k).load[width=WIDTH]() *
            (gamma + k).load[width=WIDTH]())
        k += WIDTH


@always_inline
def row_absmax(work_row: PtrF32, cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, WIDTH](0)
    var k = 0
    while k + WIDTH <= cols:
        vmax = max(vmax, (work_row + k).load[width=WIDTH]().__abs__())
        k += WIDTH
    return vmax.reduce_max()


@always_inline
def quantize_inv(work: PtrF32, qi: PtrI8, inv: Float32, n: Int):
    var vinv = SIMD[DType.float32, WIDTH](inv)
    var k = 0
    while k + WIDTH <= n:
        var v = (work + k).load[width=WIDTH]()
        (qi + k).store(quantize_i8[WIDTH](v, vinv))
        k += WIDTH


def fwht_rotate_rows[block: Int](work: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        fwht_row[block](work + r * cols, cols)


def fwht_rotate_columns[head_dim: Int](work: PtrF32, rows: Int, cols: Int):
    """Apply FWHT along the row dimension per head block of `head_dim` rows.

    Row-major layout [rows, cols]. For each head block and each column,
    gathers the column into contiguous scratch, applies FWHT, scatters back.
    Offline quantizer only — not performance-critical.
    """
    var scratch = alloc[Float32](head_dim)
    var num_heads = rows // head_dim
    for h in range(num_heads):
        var base = h * head_dim
        for c in range(cols):
            for r in range(head_dim):
                (scratch + r).store((work + (base + r) * cols + c).load())
            fwht_block[head_dim](scratch)
            for r in range(head_dim):
                (work + (base + r) * cols + c).store((scratch + r).load())


def quant_rows_per_row(work: PtrF32, qi: PtrI8, scales: PtrF32,
                       rows: Int, cols: Int):
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var amax = row_absmax(work_row, cols)
        scales[r] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        quantize_inv(work_row, qi_row, inv, cols)


def quant_rows_per_block[block: Int](work: PtrF32, qi: PtrI8, scales: PtrF32,
                                     rows: Int, cols: Int):
    var num_blocks = cols // block
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var scale_row = scales + r * num_blocks
        for b in range(num_blocks):
            var off = b * block
            var amax = row_absmax(work_row + off, block)
            scale_row[b] = amax / Float32(127.0)
            var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
            quantize_inv(work_row + off, qi_row + off, inv, block)


def rotate_and_quant_per_row[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    fwht_rotate_rows[block](work, rows, cols)
    quant_rows_per_row(work, qi, scales, rows, cols)


def rotate_and_quant_per_block[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    fwht_rotate_rows[block](work, rows, cols)
    quant_rows_per_block[block](work, qi, scales, rows, cols)


def rotate_and_quant[per_block: Bool](
    block: Int, work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int, two_sided_head_dim: Int = 0,
):
    @parameter
    def go[b: Int]():
        fwht_rotate_rows[b](work, rows, cols)
        if two_sided_head_dim == 128:
            fwht_rotate_columns[128](work, rows, cols)
        comptime if per_block:
            quant_rows_per_block[b](work, qi, scales, rows, cols)
        else:
            quant_rows_per_row(work, qi, scales, rows, cols)
    if block == 512: go[512]()
    elif block == 256: go[256]()
    elif block == 128: go[128]()
    elif block == 64: go[64]()
    elif block == 32: go[32]()
    elif block == 16: go[16]()
    else:
        abort("rotate_and_quant: unsupported block size")


def bf16_to_f32(src: PtrU8, dst: PtrF32, count: Int):
    var bp = src.bitcast[Scalar[DType.bfloat16]]()
    var k = 0
    while k + WIDTH <= count:
        (dst + k).store((bp + k).load[width=WIDTH]().cast[DType.float32]())
        k += WIDTH


# =============================================================================
# GammaCache — one-slot, name-keyed. Pre-applies sqrt(|·|) on load so
# sibling absorbed tasks reuse the sqrt'd values.
# =============================================================================


struct GammaCache(Movable):
    var last_name: String
    var last_cols: Int
    var buf: List[Float32]

    def __init__(out self, max_cols: Int):
        self.last_name = ""
        self.last_cols = 0
        self.buf = List[Float32](length=max(1, max_cols), fill=Float32(0))

    @always_inline
    def ptr(mut self) -> PtrF32:
        return f32_addr(self.buf)

    def ensure(mut self, name: String, cols: Int,
               ref headers: List[SafetensorsHeader],
               mut rio: RingIO) -> Bool:
        if name == self.last_name:
            if cols != self.last_cols:
                print("quantize: gamma cache shape conflict for " + name
                    + " cached cols=" + String(self.last_cols)
                    + ", requested cols=" + String(cols))
                return False
            return True
        var loc_opt = find_tensor(headers, name)
        if not loc_opt:
            print("quantize: missing gamma source " + name)
            return False
        var loc = loc_opt.value()
        if loc.dtype != DType.bfloat16:
            print("quantize: gamma dtype mismatch for " + name
                + " expected BF16, found " + dtype_string(loc.dtype))
            return False
        var numel = loc.rows * loc.cols
        if numel != cols:
            print("quantize: gamma shape mismatch for " + name
                + " expected " + String(cols) + " elements, found "
                + String(numel))
            return False
        var byte_size = loc.byte_size()
        if byte_size != cols * 2:
            print("quantize: gamma byte size mismatch for " + name
                + " expected " + String(cols * 2) + ", found "
                + String(byte_size))
            return False
        var raw = List[UInt8](length=byte_size, fill=UInt8(0))
        if not rio.read(loc.shard, headers[loc.shard].data_offset + loc.start,
                u8_addr(raw), byte_size):
            print("quantize: failed to read gamma " + name)
            return False
        var dst = self.ptr()
        bf16_to_f32(u8_addr(raw), dst, cols)
        var k = 0
        while k + WIDTH <= cols:
            var v = (dst + k).load[width=WIDTH]().__abs__()
            (dst + k).store(simd_sqrt(v))
            k += WIDTH
        self.last_name = name
        self.last_cols = cols
        return True


def is_supported_fwht_block(block: Int) -> Bool:
    return block == 512 or block == 256 or block == 128 \
        or block == 64 or block == 32 or block == 16


# =============================================================================
# Planner — TaskVisitor that validates tasks and computes output offsets.
# =============================================================================


struct Planner(TaskVisitor, Movable):
    var headers: List[SafetensorsHeader]
    var entries: List[OutputEntry]
    var offset: Int
    var max_gamma_cols: Int
    var ok: Bool
    var count: Int

    def __init__(out self, var headers: List[SafetensorsHeader]):
        self.headers = headers^
        self.entries = List[OutputEntry]()
        self.offset = 0
        self.max_gamma_cols = 0
        self.ok = True
        self.count = 0

    def quantize[T: QuantizeSpec](mut self, task: T) -> Bool:
        if not self.ok:
            return False
        self.count += 1
        var name = task.weight_name()
        var block = task.fwht_block()
        var per_block = task.is_per_block()

        comptime elem_bytes = T.SOURCE_ELEMENT_BYTES

        var loc_opt = find_tensor(self.headers, name)
        if not loc_opt:
            print("quantize: missing weight " + name)
            self.ok = False
            return False
        var loc = loc_opt.value()
        if loc.dtype != T.SOURCE_DTYPE:
            print("quantize: source dtype mismatch for " + name
                + " expected " + dtype_string(T.SOURCE_DTYPE)
                + ", found " + dtype_string(loc.dtype))
            self.ok = False
            return False
        var rows = loc.rows
        var cols = loc.cols

        if rows <= 0 or cols <= 0:
            print("quantize: invalid shape for " + name)
            self.ok = False
            return False
        if not is_supported_fwht_block(block):
            print("quantize: unsupported FWHT block for " + name)
            self.ok = False
            return False
        if block % WIDTH != 0 or cols % WIDTH != 0 or cols % block != 0:
            print("quantize: alignment error for " + name)
            self.ok = False
            return False

        var expected_bytes = rows * cols * elem_bytes
        if loc.byte_size() != expected_bytes:
            print("quantize: byte size mismatch for " + name)
            self.ok = False
            return False

        comptime if T.AUX_SUFFIX != StaticString(""):
            var aux_name = name + String(T.AUX_SUFFIX)
            var aux_opt = find_tensor(self.headers, aux_name)
            if not aux_opt:
                print("quantize: missing companion " + aux_name)
                self.ok = False
                return False
            var aux = aux_opt.value()
            if aux.dtype != T.AUX_DTYPE:
                print("quantize: companion dtype mismatch for " + aux_name)
                self.ok = False
                return False
            var expected_aux_bytes = T.aux_bytes_for(rows, cols)
            if aux.byte_size() != expected_aux_bytes:
                print("quantize: companion size mismatch for " + aux_name)
                self.ok = False
                return False
            comptime if T.AUX_ROW_BLOCK > 0:
                if rows % T.AUX_ROW_BLOCK != 0 or cols % T.AUX_ROW_BLOCK != 0:
                    print("quantize: companion layout mismatch for " + name)
                    self.ok = False
                    return False

        var gamma = task.gamma_source()
        if gamma != "" and cols > self.max_gamma_cols:
            self.max_gamma_cols = cols

        var num_blocks = cols // block if per_block else 1
        var weight_bytes = rows * cols
        var scale_bytes = rows * num_blocks * 4
        var weight_off = self.offset
        var scale_off = self.offset + weight_bytes
        self.offset = scale_off + scale_bytes

        self.entries.append(OutputEntry(
            name, DType.int8, rows, cols,
            weight_off, weight_off + weight_bytes))
        self.entries.append(OutputEntry(
            name + "_scale", DType.float32, rows, num_blocks,
            scale_off, scale_off + scale_bytes))
        return True

    def passthrough(mut self, name: String, expected_dtype: DType) -> Bool:
        if not self.ok:
            return False
        self.count += 1

        var loc_opt = find_tensor(self.headers, name)
        if not loc_opt:
            print("quantize: missing passthrough tensor " + name)
            self.ok = False
            return False
        var loc = loc_opt.value()
        if loc.dtype != expected_dtype:
            print("quantize: dtype mismatch for " + name
                + " — expected " + dtype_string(expected_dtype)
                + ", found " + dtype_string(loc.dtype))
            self.ok = False
            return False
        var byte_size = loc.byte_size()
        var weight_off = self.offset
        self.offset += byte_size
        self.entries.append(OutputEntry(
            name, loc.dtype, loc.rows, loc.cols,
            weight_off, weight_off + byte_size))
        return True

    def router_gauge_bf16(mut self, name: String) -> Bool:
        if not self.ok:
            return False
        self.count += 1

        var loc_opt = find_tensor(self.headers, name)
        if not loc_opt:
            print("quantize: missing router tensor " + name)
            self.ok = False
            return False
        var loc = loc_opt.value()
        if loc.dtype != DType.float32:
            print("quantize: router dtype mismatch for " + name
                + " expected F32, found " + dtype_string(loc.dtype))
            self.ok = False
            return False
        var rows = loc.rows
        var cols = loc.cols
        if rows <= 0 or cols <= 0:
            print("quantize: invalid router shape for " + name)
            self.ok = False
            return False
        if cols % WIDTH != 0:
            print("quantize: router cols must be f32-simd-aligned for " + name)
            self.ok = False
            return False
        var expected_bytes = rows * cols * 4
        if loc.byte_size() != expected_bytes:
            print("quantize: router byte size mismatch for " + name)
            self.ok = False
            return False

        var centered_bytes = rows * cols * 2
        var gauge_bytes = cols * 2
        var centered_off = self.offset
        var gauge_off = centered_off + centered_bytes
        self.offset = gauge_off + gauge_bytes

        self.entries.append(OutputEntry(
            name, DType.bfloat16, rows, cols,
            centered_off, centered_off + centered_bytes))
        self.entries.append(OutputEntry(
            name + "_gauge", DType.bfloat16, cols, 1,
            gauge_off, gauge_off + gauge_bytes))
        return True


# =============================================================================
# Executor — TaskVisitor that reads, converts, quantizes, and writes.
# =============================================================================


struct Executor(TaskVisitor):
    var planner: Planner
    var rio: RingIO
    var gamma_cache: GammaCache
    var data_start: Int
    var output_file_idx: Int
    var panel_rows: Int
    var copy_chunk_bytes: Int
    var entry_idx: Int
    var ok: Bool
    var total_bytes: Int
    var num_quantized: Int
    var num_passthrough: Int
    var task_idx: Int
    var total_tasks: Int

    def __init__(out self,
        var planner: Planner,
        var rio: RingIO,
        var gamma_cache: GammaCache,
        data_start: Int, output_file_idx: Int,
        panel_rows: Int, copy_chunk_bytes: Int,
    ):
        self.total_tasks = planner.count
        self.planner = planner^
        self.rio = rio^
        self.gamma_cache = gamma_cache^
        self.data_start = data_start
        self.output_file_idx = output_file_idx
        self.panel_rows = panel_rows
        self.copy_chunk_bytes = copy_chunk_bytes
        self.entry_idx = 0
        self.ok = True
        self.total_bytes = 0
        self.num_quantized = 0
        self.num_passthrough = 0
        self.task_idx = 0

    def quantize[T: QuantizeSpec](mut self, task: T) -> Bool:
        if not self.ok:
            return False
        var name = task.weight_name()
        ref entry = self.planner.entries[self.entry_idx]
        var weight_out_off = entry.data_start
        var weight_out_bytes = entry.byte_size()
        ref scale_entry = self.planner.entries[self.entry_idx + 1]
        var scale_out_off = scale_entry.data_start
        var scale_out_bytes = scale_entry.byte_size()
        self.entry_idx += 2

        var rows = entry.rows
        var cols = entry.cols
        var block = task.fwht_block()
        var per_block = task.is_per_block()
        var gamma_src = task.gamma_source()
        var two_sided = task.two_sided_head_dim()

        var prefix = "  [" + String(self.task_idx + 1) + "/" + String(self.total_tasks) + "] "
        var mode = "per-block" if per_block else "per-row"
        var num_blocks = cols // block if per_block else 1
        print(prefix + "butterquant " + mode + ": " + name
            + " [" + String(rows) + "x" + String(cols) + "]"
            + " dtype=" + dtype_string(T.SOURCE_DTYPE)
            + " block=" + String(block))
        self.task_idx += 1


        comptime src_elem_bytes = T.SOURCE_ELEMENT_BYTES
        comptime aux_row_block = T.AUX_ROW_BLOCK

        var companion_shard = -1
        var companion_start = 0
        var companion_bytes = 0
        comptime if T.AUX_SUFFIX != StaticString(""):
            var aux_name = name + String(T.AUX_SUFFIX)
            var aux_opt = find_tensor(self.planner.headers, aux_name)
            var aux = aux_opt.value()
            companion_shard = aux.shard
            companion_start = self.planner.headers[aux.shard].data_offset + aux.start
            companion_bytes = aux.byte_size()

        var loc = find_tensor(self.planner.headers, name).value()
        var primary_shard = loc.shard
        var primary_start = self.planner.headers[loc.shard].data_offset + loc.start

        var scale_stride = num_blocks * 4

        var aux_elems = max(1, companion_bytes // max(1, size_of[Scalar[T.AUX_DTYPE]]()))
        var aux_buf = List[Scalar[T.AUX_DTYPE]](
            length=aux_elems, fill=Scalar[T.AUX_DTYPE](0))
        var aux_base = UnsafePointer[Scalar[T.AUX_DTYPE], MutAnyOrigin](
            unsafe_from_address=Int(aux_buf.unsafe_ptr()))
        if companion_shard >= 0:
            if not self.rio.read(companion_shard, companion_start,
                    aux_base.bitcast[UInt8](), companion_bytes):
                print("quantize: failed to read companion for " + name)
                self.ok = False
                return False
        var aux_row_stride = 0
        comptime if aux_row_block > 0:
            if self.panel_rows % aux_row_block != 0:
                print("quantize: panel_rows must be a multiple of aux block")
                self.ok = False
                return False
            aux_row_stride = (cols // aux_row_block) * size_of[Scalar[T.AUX_DTYPE]]()

        var gamma_ptr = PtrF32()
        if gamma_src != "":
            if not self.gamma_cache.ensure(gamma_src, cols, self.planner.headers, self.rio):
                self.ok = False
                return False
            gamma_ptr = self.gamma_cache.ptr()

        var src_buf = List[Scalar[T.SOURCE_DTYPE]](
            length=self.panel_rows * cols, fill=Scalar[T.SOURCE_DTYPE](0))
        var work_buf = List[Float32](
            length=self.panel_rows * cols, fill=Float32(0))
        var qi_buf = List[Scalar[DType.int8]](
            length=self.panel_rows * cols, fill=Scalar[DType.int8](0))
        var scales_buf = List[Float32](
            length=self.panel_rows * num_blocks, fill=Float32(0))

        var src = UnsafePointer[Scalar[T.SOURCE_DTYPE], MutAnyOrigin](
            unsafe_from_address=Int(src_buf.unsafe_ptr()))
        var work = f32_addr(work_buf)
        var qi = i8_addr(qi_buf)
        var scales = f32_addr(scales_buf)
        var qi_u8 = qi.bitcast[UInt8]()
        var scales_u8 = scales.bitcast[UInt8]()

        var rows_done = 0
        while rows_done < rows:
            var panel = min(self.panel_rows, rows - rows_done)
            var panel_bytes = panel * cols * src_elem_bytes

            if not self.rio.read(primary_shard,
                    primary_start + rows_done * cols * src_elem_bytes,
                    src.bitcast[UInt8](), panel_bytes):
                print("quantize: failed to read panel for " + name)
                self.ok = False
                return False

            var aux_panel = aux_base
            comptime if aux_row_block > 0:
                aux_panel = UnsafePointer[Scalar[T.AUX_DTYPE], MutAnyOrigin](
                    unsafe_from_address=Int(aux_base) + (rows_done // aux_row_block) * aux_row_stride)

            T.convert[DType.float32](src, aux_panel, work, panel, cols)

            if gamma_ptr:
                for r in range(panel):
                    apply_gamma_in_place(work + r * cols, gamma_ptr, cols)

            if per_block:
                rotate_and_quant[True](block, work, qi, scales, panel, cols, two_sided)
            else:
                rotate_and_quant[False](block, work, qi, scales, panel, cols, two_sided)

            var dst_w = self.data_start + weight_out_off + rows_done * cols
            if not self.rio.write(self.output_file_idx, dst_w, qi_u8, panel * cols):
                self.ok = False
                return False
            var dst_s = self.data_start + scale_out_off + rows_done * scale_stride
            if not self.rio.write(self.output_file_idx, dst_s, scales_u8, panel * scale_stride):
                self.ok = False
                return False
            rows_done += panel

        _ = src_buf^
        _ = work_buf^
        _ = qi_buf^
        _ = scales_buf^
        _ = aux_buf^

        self.total_bytes += weight_out_bytes + scale_out_bytes
        self.num_quantized += 1
        return True

    def passthrough(mut self, name: String, expected_dtype: DType) -> Bool:
        if not self.ok:
            return False
        ref entry = self.planner.entries[self.entry_idx]
        self.entry_idx += 1

        var prefix = "  [" + String(self.task_idx + 1) + "/" + String(self.total_tasks) + "] "
        print(prefix + "passthrough: " + name
            + " [" + String(entry.rows) + "x" + String(entry.cols) + "]"
            + " dtype=" + dtype_string(expected_dtype)
            + " bytes=" + String(entry.byte_size()))
        self.task_idx += 1


        var loc = find_tensor(self.planner.headers, name).value()
        var primary_shard = loc.shard
        var primary_start = self.planner.headers[loc.shard].data_offset + loc.start
        var primary_bytes = loc.byte_size()
        var weight_out_off = entry.data_start

        var buf = List[UInt8](length=self.copy_chunk_bytes, fill=UInt8(0))
        var buf_ptr = u8_addr(buf)
        var copied = 0
        while copied < primary_bytes:
            var chunk = min(self.copy_chunk_bytes, primary_bytes - copied)
            if not self.rio.read(primary_shard, primary_start + copied,
                    buf_ptr, chunk):
                self.ok = False
                return False
            if not self.rio.write(self.output_file_idx,
                    self.data_start + weight_out_off + copied,
                    buf_ptr, chunk):
                self.ok = False
                return False
            copied += chunk
        _ = buf^

        self.total_bytes += primary_bytes
        self.num_passthrough += 1
        return True

    def router_gauge_bf16(mut self, name: String) -> Bool:
        if not self.ok:
            return False
        ref centered_entry = self.planner.entries[self.entry_idx]
        ref gauge_entry = self.planner.entries[self.entry_idx + 1]
        self.entry_idx += 2

        var rows = centered_entry.rows
        var cols = centered_entry.cols
        var prefix = "  [" + String(self.task_idx + 1) + "/" + String(self.total_tasks) + "] "
        print(prefix + "router gauge bf16: " + name
            + " [" + String(rows) + "x" + String(cols) + "]"
            + " gauge=[" + String(gauge_entry.rows) + "x"
            + String(gauge_entry.cols) + "]")
        self.task_idx += 1

        var loc = find_tensor(self.planner.headers, name).value()
        var primary_shard = loc.shard
        var primary_start = self.planner.headers[loc.shard].data_offset + loc.start
        var primary_bytes = loc.byte_size()
        var total = rows * cols

        var src_buf = List[Float32](length=total, fill=Float32(0))
        var gauge_buf = List[Float32](length=cols, fill=Float32(0))
        var centered_buf = List[Scalar[DType.bfloat16]](
            length=total, fill=Scalar[DType.bfloat16](0))
        var gauge_bf16_buf = List[Scalar[DType.bfloat16]](
            length=cols, fill=Scalar[DType.bfloat16](0))

        var src = f32_addr(src_buf)
        var gauge = f32_addr(gauge_buf)
        var centered = bf16_addr(centered_buf)
        var gauge_bf16 = bf16_addr(gauge_bf16_buf)

        if not self.rio.read(primary_shard, primary_start,
                src.bitcast[UInt8](), primary_bytes):
            print("quantize: failed to read router tensor for " + name)
            self.ok = False
            return False

        var r = 0
        while r < rows:
            var row = src + r * cols
            var k = 0
            while k + WIDTH <= cols:
                (gauge + k).store(
                    (gauge + k).load[width=WIDTH]()
                    + (row + k).load[width=WIDTH]())
                k += WIDTH
            r += 1

        var inv_rows = SIMD[DType.float32, WIDTH](
            Float32(1.0) / Float32(rows))
        var k = 0
        while k + WIDTH <= cols:
            var g = (gauge + k).load[width=WIDTH]() * inv_rows
            (gauge + k).store(g)
            (gauge_bf16 + k).store(g.cast[DType.bfloat16]())
            k += WIDTH

        r = 0
        while r < rows:
            var row = src + r * cols
            var out = centered + r * cols
            k = 0
            while k + WIDTH <= cols:
                var v = (
                    (row + k).load[width=WIDTH]()
                    - (gauge + k).load[width=WIDTH]()
                )
                (out + k).store(v.cast[DType.bfloat16]())
                k += WIDTH
            r += 1

        if not self.rio.write(
                self.output_file_idx,
                self.data_start + centered_entry.data_start,
                centered.bitcast[UInt8](),
                centered_entry.byte_size()):
            self.ok = False
            return False
        if not self.rio.write(
                self.output_file_idx,
                self.data_start + gauge_entry.data_start,
                gauge_bf16.bitcast[UInt8](),
                gauge_entry.byte_size()):
            self.ok = False
            return False

        _ = src_buf^
        _ = gauge_buf^
        _ = centered_buf^
        _ = gauge_bf16_buf^

        self.total_bytes += centered_entry.byte_size() + gauge_entry.byte_size()
        self.num_quantized += 1
        return True


# =============================================================================
# Top-level driver
# =============================================================================


def parse_source_headers(source_dir: Path) -> Optional[List[SafetensorsHeader]]:
    var shard_paths = discover_shards(source_dir)
    if len(shard_paths) == 0:
        print("quantize: no shards found in " + String(source_dir))
        return None
    print("found " + String(len(shard_paths)) + " shard(s)")
    var headers = List[SafetensorsHeader]()
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            print("quantize: failed to parse " + String(shard_paths[i]))
            return None
        headers.append(h.take())
    return headers^
