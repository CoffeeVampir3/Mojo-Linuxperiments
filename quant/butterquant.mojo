"""ButterQuant offline weight quantizer.

Driver is a plain function over `List[QuantizeTask]`. Callers pass the task
list they want quantized; each task is self-contained (kind + src_name +
optional gamma_src). The driver turns tasks into `QuantizePlan`s in one pass,
writing the output header, then processes each plan independently. Ordering
of the task list does not matter — gamma absorption is carried per-plan and
resolved by a one-slot `GammaCache`.

Per quantizable weight:
  1. Smooth/gamma absorption (if scheme has smooth_src): row *= sqrt(|gamma[k]|)
  2. FWHT rotation on the contraction dim per row (block sized by column
     count, capped at MAX_FWHT_BLOCK, plus a full-attn o_proj special case)
  3. Per-row (or per-block) symmetric i8 quantization

Panelized — scratch is sized for one row panel at the widest quantizable
column count and reused across plans.
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
    QuantizeTask, QuantScheme,
    QuantPassthrough, RowQuantized, BlockQuantized, SmoothBlockQuantized,
    quant_is_quantized, quant_rotation, quant_scale_blocks, quant_smooth_source,
)
from modeling.loader import discover_shards
from notstdcollections import HeapMoveArray
from experimental.hadquant_impl import fwht_row
from simd_math import roundeven, sqrt as simd_sqrt
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

    The embed table uses 64-wide blocks (matching FWHT_BLK_HIDDEN) for finer
    per-block scale adaptation — smaller blocks reduce quantization drift in
    the lm_head output.
    """
    if cols == 8192 and name.endswith("self_attn.o_proj.weight"):
        return FULL_O_PROJ_FWHT_BLOCK
    if name.endswith("embed_tokens.weight"):
        return 64
    return fwht_block_for_cols(cols)


# =============================================================================
# Core transforms
# =============================================================================


def quantize_panel_rows[block: Int, per_block: Bool](job: QuantPanelJob):
    """FWHT + absmax i8 quantize, with optional per-column scaling.

    When per_block=False: one absmax scale per row (scales layout: [rows]).
    When per_block=True:  one absmax per FWHT block (scales layout: [rows, num_blocks]).
    If job.apply_gamma: multiply each element by gamma_ptr[k] before FWHT.
    """
    comptime lo = SIMD[DType.float32, WIDTH](-128.0)
    comptime hi = SIMD[DType.float32, WIDTH](127.0)

    var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=job.src_ptr)
    var work = PtrF32(unsafe_from_address=job.work_ptr)
    var qi = PtrI8(unsafe_from_address=job.qi_ptr)
    var scales = PtrF32(unsafe_from_address=job.scales_ptr)
    var gamma = PtrF32(unsafe_from_address=job.gamma_ptr)
    var cols = job.cols
    comptime num_blocks = 1 if not per_block else 0
    var rt_num_blocks = cols // block if per_block else 1

    for r in range(job.row_start, job.row_start + job.row_count):
        var src_row = src + r * cols
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var scale_row = scales + (r * rt_num_blocks if per_block else r)

        # Load bf16 → f32, optionally apply per-column scaling
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

        # Block-diagonal FWHT
        fwht_row[DType.float32, block](work_row, cols)

        # Absmax + quantize — per-row or per-block
        comptime if per_block:
            for blk in range(rt_num_blocks):
                var blk_off = blk * block
                var blk_work = work_row + blk_off
                var blk_qi = qi_row + blk_off

                var vmax = SIMD[DType.float32, WIDTH](0)
                var bk = 0
                while bk + WIDTH <= block:
                    vmax = max(vmax, (blk_work + bk).load[width=WIDTH]().__abs__())
                    bk += WIDTH
                var amax = vmax.reduce_max()
                while bk < block:
                    var a = blk_work[bk]
                    if a < Float32(0):
                        a = -a
                    if a > amax:
                        amax = a
                    bk += 1

                scale_row[blk] = amax / Float32(127.0)
                var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
                var vinv = SIMD[DType.float32, WIDTH](inv)

                bk = 0
                while bk + WIDTH <= block:
                    var v = (blk_work + bk).load[width=WIDTH]()
                    (blk_qi + bk).store(min(max(roundeven(v * vinv), lo), hi).cast[DType.int8]())
                    bk += WIDTH
                while bk < block:
                    var v = roundeven[DType.float32, 1](blk_work[bk] * inv)
                    blk_qi[bk] = min(max(v, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()
                    bk += 1
        else:
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


def quantize_panel_dispatch[mask_size: Int, block: Int, per_block: Bool](
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
        pool.dispatch[QuantPanelJob, quantize_panel_rows[block, per_block]](jobs, num_jobs)
        pool.join()
        jobs.free()
    else:
        quantize_panel_rows[block, per_block](QuantPanelJob(
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr,
            cols, 0, rows, apply_gamma))


def run_panel[mask_size: Int](
    block: Int,
    per_block: Bool,
    src_ptr: Int, work_ptr: Int, qi_ptr: Int, scales_ptr: Int, gamma_ptr: Int,
    rows: Int, cols: Int, apply_gamma: Bool,
    mut pool: BurstPool[mask_size],
):
    """Unified dispatch for all quantize panel variants.

    per_block: per-FWHT-block scales (True) or per-row scales (False).
    apply_gamma: multiply by gamma_ptr per element before FWHT.
    """
    if per_block:
        if block == 512:
            quantize_panel_dispatch[mask_size, 512, True](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 256:
            quantize_panel_dispatch[mask_size, 256, True](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 128:
            quantize_panel_dispatch[mask_size, 128, True](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 64:
            quantize_panel_dispatch[mask_size, 64, True](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        else:
            quantize_panel_dispatch[mask_size, 32, True](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
    else:
        if block == 512:
            quantize_panel_dispatch[mask_size, 512, False](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 256:
            quantize_panel_dispatch[mask_size, 256, False](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 128:
            quantize_panel_dispatch[mask_size, 128, False](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        elif block == 64:
            quantize_panel_dispatch[mask_size, 64, False](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
        else:
            quantize_panel_dispatch[mask_size, 32, False](
                src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)


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


def find_tensor(ref headers: HeapMoveArray[SafetensorsHeader], name: String) -> Tuple[Int, TensorMeta]:
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


# =============================================================================
# QuantizePlan — one task resolved against the source checkpoint
# =============================================================================


struct PlanKind:
    comptime QUANTIZE = 0
    comptime PASSTHROUGH = 1
    comptime MISSING = 2


@fieldwise_init
struct QuantizePlan(Copyable, Movable):
    var kind: Int                   # PlanKind.QUANTIZE / PASSTHROUGH / MISSING
    var scheme: QuantScheme         # the full scheme (only meaningful when kind=QUANTIZE)

    var src_name: String
    var src_shard: Int
    var src_start: Int
    var src_bytes: Int
    var src_dtype: DType

    var rows: Int
    var cols: Int
    var block: Int                  # FWHT rotation block (from scheme)
    var num_blocks: Int             # scale blocks per row (from scheme)

    # Output layout — offsets relative to data_start
    var weight_out_off: Int
    var weight_out_bytes: Int
    var scale_out_off: Int
    var scale_out_bytes: Int


# =============================================================================
# GammaCache — one-slot name-keyed cache for absorbed norms
#
# Shared gammas (q/k/v off a single input_layernorm) read the source once
# when the plans that reference them come in any order. Hit test is a string
# compare; on miss, re-read via the same RingIO the driver already has.
# =============================================================================


struct GammaCache(Movable):
    var last_name: String
    var buf: PtrF32

    def __init__(out self, buf: PtrF32):
        self.last_name = ""
        self.buf = buf

    def ensure(mut self, name: String, mut rio: RingIO,
               ref headers: HeapMoveArray[SafetensorsHeader],
               io_buf: PtrU8) -> Bool:
        if name == self.last_name:
            return True
        var located = find_tensor(headers, name)
        if located[0] < 0:
            print("quantize: missing gamma source " + name)
            return False
        var shard_idx = located[0]
        var meta = located[1].copy()
        var byte_size = meta.end - meta.start
        if not rio.read(shard_idx, headers[shard_idx].data_offset + meta.start,
                io_buf, byte_size):
            return False
        bf16_to_f32(io_buf, self.buf, meta.shape[0])
        self.last_name = name
        return True


# =============================================================================
# Planning — tasks + headers → self-contained plans + output entries
# =============================================================================


@fieldwise_init
struct PlanBundle(Movable):
    var plans: List[QuantizePlan]
    var entries: List[OutputEntry]
    var max_quant_cols: Int
    var max_num_blocks: Int


def plan_quantization(
    tasks: List[QuantizeTask],
    ref headers: HeapMoveArray[SafetensorsHeader],
) -> Optional[PlanBundle]:
    var plans = List[QuantizePlan](capacity=len(tasks))
    var entries = List[OutputEntry]()
    var offset = 0
    var max_quant_cols = 0
    var max_num_blocks = 1

    for t_idx in range(len(tasks)):
        var t = tasks[t_idx].copy()
        var s = t.scheme.copy()

        if quant_is_quantized(s):
            var located = find_tensor(headers, t.name)
            if located[0] < 0:
                print("quantize: missing weight " + t.name)
                return None
            var shard_idx = located[0]
            var meta = located[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            var rot = quant_rotation(s)
            var block = rot if rot > 0 else fwht_block_for_cols(cols)
            var num_blocks = quant_scale_blocks(s, cols)
            if num_blocks == 0:
                num_blocks = 1
            var weight_bytes = rows * cols
            var scale_bytes = rows * num_blocks * 4

            var weight_off = offset
            offset += weight_bytes
            var scale_off = offset
            offset += scale_bytes

            entries.append(OutputEntry(
                t.name, DType.int8, rows, cols,
                weight_off, weight_off + weight_bytes))
            entries.append(OutputEntry(
                t.name + "_scale", DType.float32, rows, num_blocks,
                scale_off, scale_off + scale_bytes))

            plans.append(QuantizePlan(
                kind=PlanKind.QUANTIZE,
                scheme=t.scheme.copy(),
                src_name=t.name, src_shard=shard_idx,
                src_start=headers[shard_idx].data_offset + meta.start,
                src_bytes=weight_bytes * 2,
                src_dtype=meta.dtype,
                rows=rows, cols=cols, block=block, num_blocks=num_blocks,
                weight_out_off=weight_off, weight_out_bytes=weight_bytes,
                scale_out_off=scale_off, scale_out_bytes=scale_bytes,
            ))
            if cols > max_quant_cols:
                max_quant_cols = cols
            if num_blocks > max_num_blocks:
                max_num_blocks = num_blocks

        else:
            # Passthrough
            var located = find_tensor(headers, t.name)
            if located[0] < 0:
                plans.append(QuantizePlan(
                    kind=PlanKind.MISSING,
                    scheme=t.scheme.copy(),
                    src_name=t.name, src_shard=-1, src_start=0,
                    src_bytes=0, src_dtype=DType.uint8,
                    rows=0, cols=0, block=0, num_blocks=0,
                    weight_out_off=-1, weight_out_bytes=0,
                    scale_out_off=-1, scale_out_bytes=0,
                ))
                continue
            var shard_idx = located[0]
            var meta = located[1].copy()
            var byte_size = meta.end - meta.start
            var rc = fold_shape(meta.shape)
            var weight_off = offset
            offset += byte_size
            entries.append(OutputEntry(
                t.name, meta.dtype, rc[0], rc[1],
                weight_off, weight_off + byte_size))
            plans.append(QuantizePlan(
                kind=PlanKind.PASSTHROUGH,
                scheme=t.scheme.copy(),
                src_name=t.name, src_shard=shard_idx,
                src_start=headers[shard_idx].data_offset + meta.start,
                src_bytes=byte_size, src_dtype=meta.dtype,
                rows=rc[0], cols=rc[1], block=0, num_blocks=0,
                weight_out_off=weight_off, weight_out_bytes=byte_size,
                scale_out_off=-1, scale_out_bytes=0,
            ))

    return PlanBundle(plans^, entries^, max_quant_cols, max_num_blocks)


# =============================================================================
# Block-size dispatch — one cascade per kernel family, called once per plan
# =============================================================================




# =============================================================================
# Panel scratch — five pointers + the row panel budget, packed together
# =============================================================================


@fieldwise_init
struct PanelScratch(Copyable):
    var io_buf: PtrU8
    var work: PtrF32
    var qi: PtrI8
    var scales: PtrF32
    var gamma: PtrF32
    var panel_rows: Int


@fieldwise_init
struct ScratchLayout(Copyable):
    var total_bytes: Int
    var work_off: Int
    var qi_off: Int
    var scales_off: Int
    var gamma_off: Int


def compute_scratch_layout(
    max_quant_cols: Int, max_num_blocks: Int,
    panel_rows: Int, copy_chunk_bytes: Int, arena_alignment: Int,
) -> ScratchLayout:
    var panel_elems = panel_rows * max_quant_cols
    var quant_panel_bytes = panel_elems * 2
    var work_off = align_up(quant_panel_bytes, arena_alignment)
    var work_bytes = panel_elems * 4
    var qi_off = align_up(work_off + work_bytes, arena_alignment)
    var qi_bytes = panel_elems
    var scales_off = align_up(qi_off + qi_bytes, arena_alignment)
    var scales_bytes = panel_rows * max_num_blocks * 4
    var gamma_off = align_up(scales_off + scales_bytes, arena_alignment)
    var gamma_bytes = max(1, max_quant_cols) * 4
    var quant_peak = gamma_off + gamma_bytes
    return ScratchLayout(
        total_bytes=max(copy_chunk_bytes, quant_peak),
        work_off=work_off, qi_off=qi_off,
        scales_off=scales_off, gamma_off=gamma_off,
    )


# =============================================================================
# Per-plan processors — pure functions of plan + scratch + io + pool
# =============================================================================


def process_quantize_plan[mask_size: Int](
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, mut gamma_cache: GammaCache,
    ref headers: HeapMoveArray[SafetensorsHeader],
    scratch: PanelScratch, mut pool: BurstPool[mask_size],
) -> Bool:
    """Unified quantization processor for all QUANTIZE/GAMMA_QUANTIZE/PER_BLOCK/SMOOTH variants.

    Handles gamma loading, optional sqrt transform (smooth split),
    per-row vs per-block scale output, and panel I/O.
    """
    var smooth_src = quant_smooth_source(plan.scheme)
    var apply_gamma = smooth_src != ""
    var is_smooth = apply_gamma
    var per_block = plan.num_blocks > 1

    if apply_gamma:
        if not gamma_cache.ensure(smooth_src, rio, headers, scratch.io_buf):
            print("quantize: failed to load gamma/smooth source " + smooth_src)
            return False

    # For smooth split: transform gamma → sqrt(|gamma|) in-place.
    if is_smooth:
        comptime w = simd_width_of[DType.float32]()
        var gp = scratch.gamma
        var k = 0
        while k + w <= plan.cols:
            var v = (gp + k).load[width=w]().__abs__()
            (gp + k).store(simd_sqrt(v))
            k += w
        while k < plan.cols:
            var v = gp[k]
            if v < Float32(0):
                v = -v
            gp[k] = simd_sqrt(v)
            k += 1

    var scale_stride = plan.num_blocks * 4
    var rows_done = 0
    while rows_done < plan.rows:
        var panel = min(scratch.panel_rows, plan.rows - rows_done)
        var panel_bytes = panel * plan.cols * 2
        var src_off = plan.src_start + rows_done * plan.cols * 2
        if not rio.read(plan.src_shard, src_off, scratch.io_buf, panel_bytes):
            print("quantize: failed to read panel for " + plan.src_name)
            return False

        run_panel[mask_size](
            plan.block, per_block,
            Int(scratch.io_buf), Int(scratch.work), Int(scratch.qi),
            Int(scratch.scales), Int(scratch.gamma),
            panel, plan.cols, apply_gamma, pool)

        var dst_w = data_start + plan.weight_out_off + rows_done * plan.cols
        if not rio.write(output_file_idx, dst_w,
                scratch.qi.bitcast[UInt8](), panel * plan.cols):
            return False
        var dst_s = data_start + plan.scale_out_off + rows_done * scale_stride
        if not rio.write(output_file_idx, dst_s,
                scratch.scales.bitcast[UInt8](), panel * scale_stride):
            return False
        rows_done += panel
    return True



def process_passthrough_plan(
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, scratch: PanelScratch, copy_chunk_bytes: Int,
) -> Bool:
    var copied = 0
    while copied < plan.src_bytes:
        var chunk = min(copy_chunk_bytes, plan.src_bytes - copied)
        if not rio.read(plan.src_shard, plan.src_start + copied,
                scratch.io_buf, chunk):
            return False
        if not rio.write(output_file_idx,
                data_start + plan.weight_out_off + copied,
                scratch.io_buf, chunk):
            return False
        copied += chunk
    return True


# =============================================================================
# Top-level driver
# =============================================================================


def run_quantizer[
    panel_rows: Int = DEFAULT_QUANT_PANEL_ROWS,
    copy_chunk_bytes: Int = DEFAULT_COPY_CHUNK_BYTES,
    mask_size: Int = 128,
    arena_alignment: Int = DEFAULT_ARENA_ALIGNMENT,
](
    tasks: List[QuantizeTask],
    source_dir: Path,
    output_path: Path,
) -> Bool:
    """Quantize a source directory into a single output safetensors file.

    Driver is model-agnostic: caller provides the task list, which each
    model's own build_quantizer_tasks() function generates. Processing is
    stateless across tasks and order-independent — gamma absorption is
    carried on the task and resolved by a one-slot cache.
    """
    comptime assert panel_rows > 0, "panel_rows must be positive"
    comptime assert copy_chunk_bytes > 0, "copy_chunk_bytes must be positive"

    var t0 = Int(perf_counter_ns())

    # --- Source discovery + header parsing ---
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

    # --- Phase A: plan every task in one pass ---
    var bundle_opt = plan_quantization(tasks, headers)
    if not bundle_opt:
        return False
    var bundle = bundle_opt.take()

    # --- Write output header ---
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

    # --- Allocate panel scratch ---
    var layout = compute_scratch_layout(
        bundle.max_quant_cols, bundle.max_num_blocks,
        panel_rows, copy_chunk_bytes, arena_alignment)

    var numa = NumaInfo()
    var node = 0
    if numa.num_nodes > 0:
        node = numa.plan_topology(1)[0]

    var arena = NumaArena[alignment=arena_alignment](node, layout.total_bytes)
    if not arena:
        print("quantize: failed to allocate panel scratch arena")
        return False
    _ = arena.prefault()

    var scratch_base = Int(arena.base)
    var scratch = PanelScratch(
        io_buf=PtrU8(unsafe_from_address=scratch_base),
        work=PtrF32(unsafe_from_address=scratch_base + layout.work_off),
        qi=PtrI8(unsafe_from_address=scratch_base + layout.qi_off),
        scales=PtrF32(unsafe_from_address=scratch_base + layout.scales_off),
        gamma=PtrF32(unsafe_from_address=scratch_base + layout.gamma_off),
        panel_rows=panel_rows,
    )

    var pool = BurstPool[mask_size].for_topology(numa, node)
    if pool and pool.get_capacity() > 1:
        print("quantize: panel_rows=" + String(panel_rows)
            + ", workers=" + String(pool.get_capacity())
            + ", node=" + String(node))
    else:
        print("quantize: panel_rows=" + String(panel_rows)
            + ", running serial panel path")

    # --- Phase B: process each plan independently ---
    var gamma_cache = GammaCache(scratch.gamma)
    var total_bytes = 0
    var num_quantized = 0
    var num_missing = 0

    for p_idx in range(len(bundle.plans)):
        var plan = bundle.plans[p_idx].copy()
        if plan.kind == PlanKind.QUANTIZE:
            var per_block = plan.num_blocks > 1
            var smooth_src = quant_smooth_source(plan.scheme)
            var label = "per-block quantized" if per_block else "quantized"
            print("  " + label + ": " + plan.src_name
                + " [" + String(plan.rows) + "x" + String(plan.cols)
                + "] block=" + String(plan.block)
                + " num_blocks=" + String(plan.num_blocks)
                + (" smooth=" + smooth_src if smooth_src != "" else ""))
            if not process_quantize_plan[mask_size](
                    plan, data_start, output_file_idx, rio, gamma_cache,
                    headers, scratch, pool):
                return False
            total_bytes += plan.weight_out_bytes + plan.scale_out_bytes
            num_quantized += 1
        elif plan.kind == PlanKind.PASSTHROUGH:
            print("  passthrough: " + plan.src_name)
            if not process_passthrough_plan(
                    plan, data_start, output_file_idx, rio,
                    scratch, copy_chunk_bytes):
                return False
            total_bytes += plan.weight_out_bytes
        else:
            # PlanKind.MISSING — loader-only reservation, no source tensor.
            num_missing += 1

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("quantize: " + String(num_quantized) + " weights, "
        + String(total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    if num_missing > 0:
        print("quantize: " + String(num_missing) + " missing passthrough task(s) skipped")

    # Force the panel arena to live until here. Otherwise Mojo's ASAP
    # destruction frees it right after we cache `scratch.io_buf`, leaving
    # the raw pointer dangling for the entire processing loop and earning
    # us EFAULT on every io_uring read.
    _ = arena^
    return True
