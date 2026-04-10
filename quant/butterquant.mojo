"""ButterQuant offline weight quantizer.

Per quantizable weight:
  1. Gamma absorption (if preceding norm): W'[n,k] = W[n,k] * gamma[k]
  2. FWHT rotation on contraction dimension K per row (block = largest pow2 factor of K, cap 256)
  3. Per-row symmetric i8 quantization: s[n] = absmax(row)/127

Supports multi-shard checkpoints. The for_each_weight interface uses the
3-arg signature (prefix, base, target_rank).

The quantizer is panelized rather than whole-tensor buffered: scratch is sized
for a configurable row panel and reused across weights. Large tensors are
processed panel-by-panel, and row ranges within a panel are dispatched through
BurstPool workers when available.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.math import max, align_up

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader, TensorMeta,
    HEADER_LEN_BYTES,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion, ReadWriteMode, ReadMode, RingError,
)
from numa import NumaArena, NumaInfo
from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Quantizable, Gamma, Absorbed,
    WeightIterable,
)
from notstdcollections import HeapMoveArray
from experimental.hadquant_impl import fwht_row
from simd_math import roundeven
from threading import BurstPool

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.float32]()
comptime MAX_FWHT_BLOCK = 256
comptime FULL_O_PROJ_FWHT_BLOCK = 512
comptime DEFAULT_QUANT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK_BYTES = 16 * 1024 * 1024
comptime DEFAULT_ARENA_ALIGNMENT = 64

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


@fieldwise_init
struct QuantPanelJob(Copyable, ImplicitlyCopyable, Movable):
    var src_ptr: Int
    var work_ptr: Int
    var qi_ptr: Int
    var scales_ptr: Int
    var gamma_ptr: Int
    var cols: Int
    var row_start: Int
    var row_count: Int
    var apply_gamma: Bool


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


def fwht_block_for_weight(name: String, cols: Int) -> Int:
    """Per-weight FWHT block selection.

    Gemma4 full-attention O projection is consumed one 512-dim head at a time
    with one activation scale per head. Its offline rotation must use the same
    512-wide basis; the generic 256-wide cap is still correct elsewhere.
    """
    if cols == 8192 and name.endswith("self_attn.o_proj.weight"):
        return FULL_O_PROJ_FWHT_BLOCK
    return fwht_block_for_cols(cols)


# =============================================================================
# Core transforms
# =============================================================================


def quantize_panel_rows[block: Int](job: QuantPanelJob):
    comptime lo = SIMD[DType.float32, WIDTH](-128.0)
    comptime hi = SIMD[DType.float32, WIDTH](127.0)

    var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=job.src_ptr)
    var work = PtrF32(unsafe_from_address=job.work_ptr)
    var qi = PtrI8(unsafe_from_address=job.qi_ptr)
    var scales = PtrF32(unsafe_from_address=job.scales_ptr)
    var gamma = PtrF32(unsafe_from_address=job.gamma_ptr)
    var cols = job.cols

    for r in range(job.row_start, job.row_start + job.row_count):
        var src_row = src + r * cols
        var work_row = work + r * cols
        var qi_row = qi + r * cols

        var k = 0
        while k + WIDTH <= cols:
            var x = (src_row + k).load[width=WIDTH]().cast[DType.float32]()
            if job.apply_gamma:
                x *= (gamma + k).load[width=WIDTH]()
            (work_row + k).store(x)
            k += WIDTH
        while k < cols:
            var x = Float32(src_row[k])
            if job.apply_gamma:
                x *= gamma[k]
            work_row[k] = x
            k += 1

        fwht_row[DType.float32, block](work_row, cols)

        var vmax = SIMD[DType.float32, WIDTH](0)
        k = 0
        while k + WIDTH <= cols:
            vmax = max(vmax, (work_row + k).load[width=WIDTH]().__abs__())
            k += WIDTH
        var amax = vmax.reduce_max()
        while k < cols:
            var a = work_row[k]
            if a < Float32(0):
                a = -a
            if a > amax:
                amax = a
            k += 1

        scales[r] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        var vinv = SIMD[DType.float32, WIDTH](inv)

        k = 0
        while k + WIDTH <= cols:
            var v = (work_row + k).load[width=WIDTH]()
            (qi_row + k).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
            k += WIDTH
        while k < cols:
            var v = roundeven[DType.float32, 1](work_row[k] * inv)
            qi_row[k] = min(max(v, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()
            k += 1


def quantize_panel_dispatch[mask_size: Int, block: Int](
    src_ptr: Int,
    work_ptr: Int,
    qi_ptr: Int,
    scales_ptr: Int,
    gamma_ptr: Int,
    rows: Int,
    cols: Int,
    apply_gamma: Bool,
    mut pool: BurstPool[mask_size],
):
    if rows <= 0:
        return
    if pool and pool.get_capacity() > 1 and rows > 1:
        var num_jobs = min(rows, pool.get_capacity())
        var rows_per_job = (rows + num_jobs - 1) // num_jobs
        var jobs = alloc[QuantPanelJob](num_jobs)
        for i in range(num_jobs):
            var row_start = i * rows_per_job
            var row_count = min(rows_per_job, rows - row_start)
            jobs[i] = QuantPanelJob(
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr,
                cols, row_start, row_count, apply_gamma)
        pool.dispatch[QuantPanelJob, quantize_panel_rows[block]](jobs, num_jobs)
        pool.join()
        jobs.free()
    else:
        quantize_panel_rows[block](QuantPanelJob(
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr,
            cols, 0, rows, apply_gamma))


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


def quantize[M: WeightIterable,
    panel_rows: Int = DEFAULT_QUANT_PANEL_ROWS,
    copy_chunk_bytes: Int = DEFAULT_COPY_CHUNK_BYTES,
    mask_size: Int = 128,
    arena_alignment: Int = DEFAULT_ARENA_ALIGNMENT,
](
    source_dir: Path, output_path: Path,
) -> Bool:
    """Quantize a multi-shard checkpoint directory to a single output safetensors.

    Discovers all model-*.safetensors in source_dir, parses headers, then
    processes each weight according to its trait tags. Per-weight FWHT block
    is computed from column count. Quantizable tensors are processed in
    row panels of `panel_rows`, while passthrough tensors are copied in
    `copy_chunk_bytes` chunks.
    """
    comptime assert panel_rows > 0, "panel_rows must be positive"
    comptime assert copy_chunk_bytes > 0, "copy_chunk_bytes must be positive"

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
    var max_quant_cols = 0

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
            if cols > max_quant_cols:
                max_quant_cols = cols

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

    # --- Phase 2: panel scratch + worker pool ---
    var panel_elems = panel_rows * max_quant_cols
    var quant_panel_bytes = panel_elems * 2
    var work_elems = panel_elems
    var qi_elems = panel_elems
    var scale_rows = panel_rows
    var gamma_cols = max(1, max_quant_cols)

    # Copy-only and quantization phases are disjoint. Lay out the compute
    # buffers immediately after the true quant input panel, not after the
    # larger copy chunk. This lets one arena cover max(copy_peak, quant_peak).
    var work_off = align_up(quant_panel_bytes, arena_alignment)
    var work_bytes = work_elems * 4
    var qi_off = align_up(work_off + work_bytes, arena_alignment)
    var qi_bytes = qi_elems
    var scales_off = align_up(qi_off + qi_bytes, arena_alignment)
    var scales_bytes = scale_rows * 4
    var gamma_off = align_up(scales_off + scales_bytes, arena_alignment)
    var gamma_bytes = gamma_cols * 4
    var quant_peak_bytes = gamma_off + gamma_bytes
    var scratch_bytes = max(copy_chunk_bytes, quant_peak_bytes)

    var numa = NumaInfo()
    var node = 0
    if numa.num_nodes > 0:
        node = numa.plan_topology(1)[0]

    var arena = NumaArena[alignment=arena_alignment](node, scratch_bytes)
    if not arena:
        print("quantize: failed to allocate panel scratch arena")
        return False
    _ = arena.prefault()

    var scratch_base = Int(arena.base)
    var io_buf = PtrU8(unsafe_from_address=scratch_base)
    var work = PtrF32(unsafe_from_address=scratch_base + work_off)
    var qi = PtrI8(unsafe_from_address=scratch_base + qi_off)
    var scales_buf = PtrF32(unsafe_from_address=scratch_base + scales_off)
    var gamma_buf = PtrF32(unsafe_from_address=scratch_base + gamma_off)

    var pool = BurstPool[mask_size].for_topology(numa, node)
    if pool and pool.get_capacity() > 1:
        print("quantize: panel_rows=" + String(panel_rows) + ", workers=" + String(pool.get_capacity())
            + ", node=" + String(node))
    else:
        print("quantize: panel_rows=" + String(panel_rows) + ", running serial panel path")

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
                    io_buf, byte_size):
                print("quantize: failed to read " + task.name)
                return False
            bf16_to_f32(io_buf, gamma_buf, meta.shape[0])
            has_gamma = True
            print("  absorbed: " + task.name)

        elif task.kind == GAMMA_QUANTIZE or task.kind == QUANTIZE:
            var result = find_tensor(headers, task.name)
            var shard_idx = result[0]
            var meta = result[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]

            var block = fwht_block_for_weight(task.name, cols)
            print("  quantized: " + task.name + " [" + String(rows) + "x" + String(cols)
                + "] block=" + String(block))

            var we = entries[entry_idx]
            var se = entries[entry_idx + 1]
            var rows_done = 0
            while rows_done < rows:
                var panel = min(panel_rows, rows - rows_done)
                var panel_bytes = panel * cols * 2
                var src_off = headers[shard_idx].data_offset + meta.start + rows_done * cols * 2
                if not rio.read(shard_idx, src_off, io_buf, panel_bytes):
                    print("quantize: failed to read panel for " + task.name)
                    return False

                if block == 512:
                    quantize_panel_dispatch[mask_size, 512](
                        Int(io_buf), Int(work), Int(qi), Int(scales_buf), Int(gamma_buf),
                        panel, cols, task.kind == GAMMA_QUANTIZE and has_gamma, pool)
                elif block == 256:
                    quantize_panel_dispatch[mask_size, 256](
                        Int(io_buf), Int(work), Int(qi), Int(scales_buf), Int(gamma_buf),
                        panel, cols, task.kind == GAMMA_QUANTIZE and has_gamma, pool)
                elif block == 128:
                    quantize_panel_dispatch[mask_size, 128](
                        Int(io_buf), Int(work), Int(qi), Int(scales_buf), Int(gamma_buf),
                        panel, cols, task.kind == GAMMA_QUANTIZE and has_gamma, pool)
                elif block == 64:
                    quantize_panel_dispatch[mask_size, 64](
                        Int(io_buf), Int(work), Int(qi), Int(scales_buf), Int(gamma_buf),
                        panel, cols, task.kind == GAMMA_QUANTIZE and has_gamma, pool)
                else:
                    quantize_panel_dispatch[mask_size, 32](
                        Int(io_buf), Int(work), Int(qi), Int(scales_buf), Int(gamma_buf),
                        panel, cols, task.kind == GAMMA_QUANTIZE and has_gamma, pool)

                var dst_weight_off = data_start + we.data_start + rows_done * cols
                if not rio.write(output_file_idx, dst_weight_off, qi.bitcast[UInt8](), panel * cols):
                    print("quantize: failed to write panel for " + task.name)
                    return False
                var dst_scale_off = data_start + se.data_start + rows_done * 4
                if not rio.write(output_file_idx, dst_scale_off, scales_buf.bitcast[UInt8](), panel * 4):
                    print("quantize: failed to write panel scales for " + task.name)
                    return False
                total_bytes += panel * cols + panel * 4
                rows_done += panel

            entry_idx += 2
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

            var pe = entries[entry_idx]
            var copied = 0
            while copied < byte_size:
                var chunk = min(copy_chunk_bytes, byte_size - copied)
                if not rio.read(shard_idx, headers[shard_idx].data_offset + meta.start + copied, io_buf, chunk):
                    print("quantize: failed to read " + task.name)
                    return False
                if not rio.write(output_file_idx, data_start + pe.data_start + copied, io_buf, chunk):
                    print("quantize: failed to write " + task.name)
                    return False
                copied += chunk
            entry_idx += 1
            total_bytes += byte_size
            print("  passthrough: " + task.name)

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("quantize: " + String(num_quantized) + " weights, "
        + String(total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    return True
