"""ButterQuant offline weight quantizer.

Walks a `List[Task]`, resolves each against source safetensors headers,
and produces a single quantized output file. Each task becomes a
`QuantizePlan` — a flat struct carrying all scalars needed for both
planning and execution. No Variant on the plan, no re-extraction at
execute time.

Per quantizable weight:
  1. Read source panel through SourceFormat converter → f32
  2. Optional gamma absorption: row *= sqrt(|gamma|)
  3. FWHT rotation at comptime block size
  4. Per-row or per-block symmetric absmax i8 quantization
  5. Write int8 weights + f32 scales to output

Gamma is resolved through a one-slot cache keyed by tensor name.
Sibling absorbed tasks sharing the same gamma hit the cache.
"""

from std.memory import UnsafePointer
from std.pathlib import Path
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.math import max
from std.os import abort

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, ReadWriteMode,
)
from modeling.model_spec import (
    Task, SourceFormat,
    Passthrough, ButterquantI8PerRow, ButterquantI8PerRowAbsorbed,
    ButterquantI8PerBlock, ButterquantI8PerBlockAbsorbed,
)
from modeling.loader import discover_shards
from experimental3.kernels.fwht import fwht_row
from simd_math import quantize_i8, quantize_i8_scalar, sqrt as simd_sqrt
from quant.source_format import (
    source_dtype, source_element_bytes,
    source_aux_dtype, source_aux_bytes, source_aux_name,
    source_aux_row_block, convert_to_f32,
)

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
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


def source_format_string(source: Int) -> String:
    if source == SourceFormat.BF16: return "BF16"
    if source == SourceFormat.F32: return "F32"
    if source == SourceFormat.FP8_E4M3_BLOCK128: return "FP8_E4M3_BLOCK128"
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


def rotate_and_quant_per_row[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        fwht_row[block](work_row, cols)
        var amax = row_absmax(work_row, cols)
        scales[r] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        quantize_inv(work_row, qi_row, inv, cols)


def rotate_and_quant_per_block[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    var num_blocks = cols // block
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var scale_row = scales + r * num_blocks
        fwht_row[block](work_row, cols)
        for b in range(num_blocks):
            var off = b * block
            var amax = row_absmax(work_row + off, block)
            scale_row[b] = amax / Float32(127.0)
            var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
            quantize_inv(work_row + off, qi_row + off, inv, block)


def rotate_and_quant[per_block: Bool](
    block: Int, work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    @parameter
    def go[b: Int]():
        comptime if per_block:
            rotate_and_quant_per_block[b](work, qi, scales, rows, cols)
        else:
            rotate_and_quant_per_row[b](work, qi, scales, rows, cols)
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


# =============================================================================
# QuantizePlan — flat, self-contained. No Variant, no re-extraction.
#
# is_passthrough() == True  → copy bytes unchanged, ignore quant fields.
# is_passthrough() == False → FWHT rotate + absmax i8, using source/block/etc.
# =============================================================================


@fieldwise_init
struct QuantizePlan(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var rows: Int
    var cols: Int

    var source: Int
    var block: Int
    var per_block: Bool
    var gamma_src: String
    var passthrough_dtype: DType

    var primary_shard: Int
    var primary_start: Int
    var primary_bytes: Int

    var companion_shard: Int
    var companion_start: Int
    var companion_bytes: Int

    var weight_out_off: Int
    var weight_out_bytes: Int
    var scale_out_off: Int
    var scale_out_bytes: Int

    @always_inline
    def is_passthrough(self) -> Bool:
        return self.scale_out_off < 0


# =============================================================================
# Planning
# =============================================================================


@fieldwise_init
struct PlanBundle(Movable):
    var plans: List[QuantizePlan]
    var entries: List[OutputEntry]
    var max_gamma_cols: Int


def is_supported_fwht_block(block: Int) -> Bool:
    return block == 512 or block == 256 or block == 128 \
        or block == 64 or block == 32 or block == 16


def plan_passthrough(
    name: String, expected_dtype: DType, mut offset: Int,
    ref headers: List[SafetensorsHeader],
    mut plans: List[QuantizePlan],
    mut entries: List[OutputEntry],
) -> Bool:
    var loc_opt = find_tensor(headers, name)
    if not loc_opt:
        print("quantize: missing passthrough tensor " + name)
        return False
    var loc = loc_opt.value()
    if loc.dtype != expected_dtype:
        print("quantize: dtype mismatch for " + name
            + " — expected " + dtype_string(expected_dtype)
            + ", found " + dtype_string(loc.dtype))
        return False
    var byte_size = loc.byte_size()
    var weight_off = offset
    offset += byte_size
    entries.append(OutputEntry(
        name, loc.dtype, loc.rows, loc.cols,
        weight_off, weight_off + byte_size))
    plans.append(QuantizePlan(
        name=name, rows=loc.rows, cols=loc.cols,
        source=0, block=0, per_block=False,
        gamma_src=String(""), passthrough_dtype=loc.dtype,
        primary_shard=loc.shard,
        primary_start=headers[loc.shard].data_offset + loc.start,
        primary_bytes=byte_size,
        companion_shard=-1, companion_start=0, companion_bytes=0,
        weight_out_off=weight_off, weight_out_bytes=byte_size,
        scale_out_off=-1, scale_out_bytes=0,
    ))
    return True


def plan_butterquant(
    name: String, source: Int, block: Int, per_block: Bool,
    gamma_src: String, mut offset: Int,
    ref headers: List[SafetensorsHeader],
    mut plans: List[QuantizePlan],
    mut entries: List[OutputEntry],
) -> Bool:
    var expected_dtype = source_dtype(source)
    var elem_bytes = source_element_bytes(source)
    if expected_dtype == DType.invalid or elem_bytes <= 0:
        print("quantize: unsupported source format "
            + source_format_string(source) + " for " + name)
        return False

    var loc_opt = find_tensor(headers, name)
    if not loc_opt:
        print("quantize: missing weight " + name)
        return False
    var loc = loc_opt.value()
    if loc.dtype != expected_dtype:
        print("quantize: source dtype mismatch for " + name
            + " expected " + dtype_string(expected_dtype)
            + ", found " + dtype_string(loc.dtype))
        return False
    var rows = loc.rows
    var cols = loc.cols

    if rows <= 0 or cols <= 0:
        print("quantize: invalid shape for " + name
            + " rows=" + String(rows) + ", cols=" + String(cols))
        return False
    if not is_supported_fwht_block(block):
        print("quantize: unsupported FWHT block for " + name
            + " block=" + String(block))
        return False
    if block % WIDTH != 0:
        print("quantize: FWHT block not SIMD-aligned for " + name)
        return False
    if cols % WIDTH != 0:
        print("quantize: cols not SIMD-aligned for " + name)
        return False
    if cols % block != 0:
        print("quantize: cols must be a multiple of block for " + name)
        return False

    var expected_bytes = rows * cols * elem_bytes
    if loc.byte_size() != expected_bytes:
        print("quantize: byte size mismatch for " + name
            + " expected " + String(expected_bytes)
            + ", found " + String(loc.byte_size()))
        return False

    var primary_shard = loc.shard
    var primary_start = headers[loc.shard].data_offset + loc.start

    var companion_shard = -1
    var companion_start = 0
    var companion_bytes = 0
    var aux_name = source_aux_name(source, name)
    if aux_name != "":
        var aux_opt = find_tensor(headers, aux_name)
        if not aux_opt:
            print("quantize: missing companion tensor " + aux_name)
            return False
        var aux = aux_opt.value()
        var expected_aux_dtype = source_aux_dtype(source)
        if expected_aux_dtype == DType.invalid:
            print("quantize: unsupported companion for " + name)
            return False
        if aux.dtype != expected_aux_dtype:
            print("quantize: companion dtype mismatch for " + aux_name
                + " expected " + dtype_string(expected_aux_dtype)
                + ", found " + dtype_string(aux.dtype))
            return False
        var expected_aux_bytes = source_aux_bytes(source, rows, cols)
        if aux.byte_size() != expected_aux_bytes:
            print("quantize: companion size mismatch for " + aux_name
                + " expected " + String(expected_aux_bytes)
                + ", found " + String(aux.byte_size()))
            return False
        var aux_rb = source_aux_row_block(source)
        if aux_rb > 0 and (rows % aux_rb != 0 or cols % aux_rb != 0):
            print("quantize: companion layout mismatch for " + name)
            return False
        companion_shard = aux.shard
        companion_start = headers[aux.shard].data_offset + aux.start
        companion_bytes = expected_aux_bytes

    var num_blocks = cols // block if per_block else 1
    var weight_bytes = rows * cols
    var scale_bytes = rows * num_blocks * 4

    var weight_off = offset
    var scale_off = offset + weight_bytes
    offset = scale_off + scale_bytes

    entries.append(OutputEntry(
        name, DType.int8, rows, cols,
        weight_off, weight_off + weight_bytes))
    entries.append(OutputEntry(
        name + "_scale", DType.float32, rows, num_blocks,
        scale_off, scale_off + scale_bytes))

    plans.append(QuantizePlan(
        name=name, rows=rows, cols=cols,
        source=source, block=block, per_block=per_block,
        gamma_src=gamma_src, passthrough_dtype=DType.invalid,
        primary_shard=primary_shard,
        primary_start=primary_start,
        primary_bytes=expected_bytes,
        companion_shard=companion_shard,
        companion_start=companion_start,
        companion_bytes=companion_bytes,
        weight_out_off=weight_off, weight_out_bytes=weight_bytes,
        scale_out_off=scale_off, scale_out_bytes=scale_bytes,
    ))
    return True


def plan_task(
    task: Task, mut offset: Int,
    ref headers: List[SafetensorsHeader],
    mut plans: List[QuantizePlan],
    mut entries: List[OutputEntry],
) -> Bool:
    if task.isa[Passthrough]():
        var t = task[Passthrough]
        return plan_passthrough(t.name, t.expected_dtype, offset,
            headers, plans, entries)
    if task.isa[ButterquantI8PerRow]():
        var t = task[ButterquantI8PerRow]
        return plan_butterquant(t.name, t.source, t.block, False,
            String(""), offset, headers, plans, entries)
    if task.isa[ButterquantI8PerRowAbsorbed]():
        var t = task[ButterquantI8PerRowAbsorbed]
        return plan_butterquant(t.name, t.source, t.block, False,
            t.gamma_src, offset, headers, plans, entries)
    if task.isa[ButterquantI8PerBlock]():
        var t = task[ButterquantI8PerBlock]
        return plan_butterquant(t.name, t.source, t.block, True,
            String(""), offset, headers, plans, entries)
    if task.isa[ButterquantI8PerBlockAbsorbed]():
        var t = task[ButterquantI8PerBlockAbsorbed]
        return plan_butterquant(t.name, t.source, t.block, True,
            t.gamma_src, offset, headers, plans, entries)
    print("quantize: unknown task variant")
    return False


def plan_quantization(
    tasks: List[Task],
    ref headers: List[SafetensorsHeader],
) -> Optional[PlanBundle]:
    var plans = List[QuantizePlan](capacity=len(tasks))
    var entries = List[OutputEntry]()
    var offset = 0

    for t_idx in range(len(tasks)):
        var task = tasks[t_idx].copy()
        if not plan_task(task, offset, headers, plans, entries):
            return None

    var max_gamma_cols = 0
    for i in range(len(plans)):
        if plans[i].gamma_src != "" and plans[i].cols > max_gamma_cols:
            max_gamma_cols = plans[i].cols

    return PlanBundle(plans^, entries^, max_gamma_cols)


# =============================================================================
# Execution
# =============================================================================


def execute_passthrough(
    plan: QuantizePlan, data_start: Int,
    output_file_idx: Int, mut rio: RingIO, copy_chunk_bytes: Int,
) -> Bool:
    var buf = List[UInt8](length=copy_chunk_bytes, fill=UInt8(0))
    var buf_ptr = u8_addr(buf)
    var copied = 0
    while copied < plan.primary_bytes:
        var chunk = min(copy_chunk_bytes, plan.primary_bytes - copied)
        if not rio.read(plan.primary_shard, plan.primary_start + copied,
                buf_ptr, chunk):
            return False
        if not rio.write(output_file_idx,
                data_start + plan.weight_out_off + copied,
                buf_ptr, chunk):
            return False
        copied += chunk
    _ = buf^
    return True


def execute_butterquant(
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, mut gamma_cache: GammaCache,
    ref headers: List[SafetensorsHeader],
    panel_rows: Int,
) -> Bool:
    var src_elem_bytes = source_element_bytes(plan.source)
    if src_elem_bytes <= 0:
        print("quantize: unsupported source for " + plan.name)
        return False
    var num_blocks = plan.cols // plan.block if plan.per_block else 1
    var scale_stride = num_blocks * 4

    var companion_buf = List[UInt8](
        length=max(1, plan.companion_bytes), fill=UInt8(0))
    var companion_base = u8_addr(companion_buf)
    if plan.companion_shard >= 0:
        if not rio.read(plan.companion_shard, plan.companion_start,
                companion_base, plan.companion_bytes):
            print("quantize: failed to read companion for " + plan.name)
            return False
    var companion_rb = source_aux_row_block(plan.source)
    var companion_row_stride = 0
    if companion_rb > 0:
        if panel_rows % companion_rb != 0:
            print("quantize: panel_rows must be a multiple of aux block for "
                + plan.name + " panel_rows=" + String(panel_rows)
                + ", aux_block=" + String(companion_rb))
            return False
        companion_row_stride = (plan.cols // companion_rb) * 4

    var gamma_ptr = PtrF32()
    if plan.gamma_src != "":
        if not gamma_cache.ensure(plan.gamma_src, plan.cols, headers, rio):
            return False
        gamma_ptr = gamma_cache.ptr()

    var src_buf = List[UInt8](
        length=panel_rows * plan.cols * src_elem_bytes, fill=UInt8(0))
    var work_buf = List[Float32](
        length=panel_rows * plan.cols, fill=Float32(0))
    var qi_buf = List[Scalar[DType.int8]](
        length=panel_rows * plan.cols, fill=Scalar[DType.int8](0))
    var scales_buf = List[Float32](
        length=panel_rows * num_blocks, fill=Float32(0))

    var src = u8_addr(src_buf)
    var work = f32_addr(work_buf)
    var qi = i8_addr(qi_buf)
    var scales = f32_addr(scales_buf)
    var qi_u8 = qi.bitcast[UInt8]()
    var scales_u8 = scales.bitcast[UInt8]()

    var rows_done = 0
    while rows_done < plan.rows:
        var panel = min(panel_rows, plan.rows - rows_done)
        var panel_bytes = panel * plan.cols * src_elem_bytes

        if not rio.read(plan.primary_shard,
                plan.primary_start + rows_done * plan.cols * src_elem_bytes,
                src, panel_bytes):
            print("quantize: failed to read panel for " + plan.name)
            return False

        var companion_panel = companion_base
        if companion_rb > 0:
            companion_panel = companion_base + (rows_done // companion_rb) * companion_row_stride

        convert_to_f32(plan.source, src, companion_panel, work, panel, plan.cols)

        if gamma_ptr:
            for r in range(panel):
                apply_gamma_in_place(work + r * plan.cols, gamma_ptr, plan.cols)

        if plan.per_block:
            rotate_and_quant[True](plan.block, work, qi, scales, panel, plan.cols)
        else:
            rotate_and_quant[False](plan.block, work, qi, scales, panel, plan.cols)

        var dst_w = data_start + plan.weight_out_off + rows_done * plan.cols
        if not rio.write(output_file_idx, dst_w, qi_u8, panel * plan.cols):
            return False
        var dst_s = data_start + plan.scale_out_off + rows_done * scale_stride
        if not rio.write(output_file_idx, dst_s, scales_u8, panel * scale_stride):
            return False
        rows_done += panel

    _ = src_buf^
    _ = work_buf^
    _ = qi_buf^
    _ = scales_buf^
    _ = companion_buf^
    return True


def execute_task(
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, mut gamma_cache: GammaCache,
    ref headers: List[SafetensorsHeader],
    panel_rows: Int, copy_chunk_bytes: Int,
) -> Bool:
    if plan.is_passthrough():
        return execute_passthrough(plan, data_start, output_file_idx,
            rio, copy_chunk_bytes)
    return execute_butterquant(plan, data_start, output_file_idx,
        rio, gamma_cache, headers, panel_rows)


def print_plan_progress(plan: QuantizePlan, plan_idx: Int, total_plans: Int):
    var prefix = "  [" + String(plan_idx + 1) + "/" + String(total_plans) + "] "
    if plan.is_passthrough():
        print(prefix + "passthrough: " + plan.name
            + " [" + String(plan.rows) + "x" + String(plan.cols) + "]"
            + " dtype=" + dtype_string(plan.passthrough_dtype)
            + " bytes=" + String(plan.weight_out_bytes))
        return

    var mode = "per-block" if plan.per_block else "per-row"
    var num_blocks = plan.cols // plan.block if plan.per_block else 1
    var msg = prefix + "butterquant " + mode + ": " + plan.name
        + " [" + String(plan.rows) + "x" + String(plan.cols) + "]"
        + " source=" + source_format_string(plan.source)
        + " block=" + String(plan.block)
        + " scales/row=" + String(num_blocks)
    if plan.gamma_src != "":
        msg += " gamma=" + plan.gamma_src
    if plan.companion_shard >= 0:
        msg += " companion_bytes=" + String(plan.companion_bytes)
    print(msg)


# =============================================================================
# Top-level driver
# =============================================================================


def run_quantizer[
    panel_rows: Int = DEFAULT_PANEL_ROWS,
    copy_chunk_bytes: Int = DEFAULT_COPY_CHUNK_BYTES,
](
    tasks: List[Task],
    source_dir: Path,
    output_path: Path,
) -> Bool:
    comptime assert panel_rows > 0, "panel_rows must be positive"
    comptime assert copy_chunk_bytes > 0, "copy_chunk_bytes must be positive"

    var t0 = Int(perf_counter_ns())

    var shard_paths = discover_shards(source_dir)
    if len(shard_paths) == 0:
        print("quantize: no shards found in " + String(source_dir))
        return False
    print("found " + String(len(shard_paths)) + " shard(s)")

    var headers = List[SafetensorsHeader]()
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            print("quantize: failed to parse " + String(shard_paths[i]))
            return False
        headers.append(h.take())

    var bundle_opt = plan_quantization(tasks, headers)
    if not bundle_opt:
        return False
    var bundle = bundle_opt.take()

    var header_bytes = build_header(bundle.entries)
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

    if not rio.write(output_file_idx, 0,
            PtrU8(unsafe_from_address=Int(UnsafePointer(to=header_bytes[0]))),
            len(header_bytes)):
        print("quantize: failed to write header")
        return False
    print("header: " + String(len(header_bytes)) + " bytes, "
        + String(len(bundle.entries)) + " entries, "
        + String(len(bundle.plans)) + " plans")

    var gamma_cache = GammaCache(bundle.max_gamma_cols)

    var total_bytes = 0
    var num_quantized = 0
    var num_passthrough = 0
    for p_idx in range(len(bundle.plans)):
        var plan = bundle.plans[p_idx].copy()
        print_plan_progress(plan, p_idx, len(bundle.plans))
        if not execute_task(plan, data_start, output_file_idx,
                rio, gamma_cache, headers, panel_rows, copy_chunk_bytes):
            return False
        total_bytes += plan.weight_out_bytes + max(plan.scale_out_bytes, 0)
        if plan.is_passthrough():
            num_passthrough += 1
        else:
            num_quantized += 1

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("quantize: " + String(num_quantized) + " quantized, "
        + String(num_passthrough) + " passthrough, "
        + String(total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    return True
