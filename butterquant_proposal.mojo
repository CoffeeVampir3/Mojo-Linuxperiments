"""Proposal: cleaner butterquant driver on top of List[QuantizeTask].

Fixes, relative to quant/butterquant.mojo today:

  1. No `has_gamma` latch. Each task names its own gamma source (or empty);
     processing is stateless across tasks and order-independent.
  2. No `entry_idx` lockstep counter. Each QuantizePlan carries its own
     output offsets, so phase B is a pure `for plan in plans: process(plan)`.
  3. Block-size dispatch cascade (`if block == 512: elif 256: ...`) lives
     in exactly one helper per kernel family instead of being inlined at
     every call site.
  4. ABSORBED is not a task kind. Gamma is a named edge, loaded lazily with
     a one-slot cache so shared gammas (q/k/v off one input_layernorm) read
     the source once.
  5. No dead `cols` field. Rows/cols live on the plan, resolved from source
     metadata during planning.
  6. PASSTHROUGH tasks that don't exist in the source become an explicit
     "missing" plan kind instead of a silent `continue` mid-loop.

Structure:
  QuantizeTask         (from model_spec_proposal — input)
  QuantizePlan         (self-contained unit, carries output offsets)
  GammaCache           (one-slot LRU by name)
  plan_quantization()  (tasks + headers → plans + output layout)
  run_quantizer()      (top-level: plan → write header → process plans)
  process_*_plan()     (one per kind, pure functions of plan + scratch)

Kernel primitives (`quantize_panel_dispatch`, `per_block_quantize_panel_dispatch`,
`fwht_block_for_weight`, `fwht_block_for_cols`, `bf16_to_f32`, `RingIO`) are
unchanged and imported from quant/butterquant.mojo.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.math import max, align_up

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader, TensorMeta,
)
from linux.io_uring import ReadWriteMode
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from model_spec_proposal import (
    QuantizeTask, WeightTag, build_gemma4_quantizer_tasks,
)
from quant.butterquant import (
    OutputEntry, build_header, dtype_string,
    RingIO, PtrU8, PtrF32, PtrI8,
    quantize_panel_dispatch, per_block_quantize_panel_dispatch,
    fwht_block_for_weight, bf16_to_f32,
    DEFAULT_QUANT_PANEL_ROWS, DEFAULT_COPY_CHUNK_BYTES, DEFAULT_ARENA_ALIGNMENT,
    MAX_FWHT_BLOCK,
)


# =============================================================================
# QuantizePlan — one task, fully resolved against the source checkpoint
# =============================================================================


struct PlanKind:
    comptime QUANTIZE = 0
    comptime PER_BLOCK_QUANTIZE = 1
    comptime PASSTHROUGH = 2
    comptime MISSING = 3          # passthrough task, tensor not in source


@fieldwise_init
struct QuantizePlan(Copyable):
    var kind: Int

    # Source (zero-filled for MISSING)
    var src_name: String
    var src_shard: Int
    var src_start: Int            # file offset of data (header.data_offset + meta.start)
    var src_bytes: Int            # raw passthrough byte count, or rows*cols*elem for quant
    var src_dtype: DType

    # Dimensions (1 and 1 for plain passthrough where shape is preserved verbatim)
    var rows: Int
    var cols: Int
    var block: Int                # FWHT block, 0 for passthrough
    var num_blocks: Int           # 1 for row-wise, cols/block for per-block, 0 for passthrough

    # Gamma linkage
    var gamma_src: String         # "" if none

    # Output layout — offsets relative to data_start
    var weight_out_off: Int
    var weight_out_bytes: Int
    var scale_out_off: Int         # -1 if not applicable
    var scale_out_bytes: Int


# =============================================================================
# GammaCache — one-slot name-keyed cache for the absorbed norm tensor
#
# The quantizer processes plans in order; shared gammas (q/k/v referencing
# "input_layernorm.weight") come as a contiguous run. A one-slot cache
# collapses the three reads into one.
# =============================================================================


struct GammaCache:
    var last_name: String
    var buf: PtrF32
    var cap_cols: Int

    def __init__(out self, buf: PtrF32, cap_cols: Int):
        self.last_name = ""
        self.buf = buf
        self.cap_cols = cap_cols

    def ensure(mut self, name: String, mut rio: RingIO,
               headers: HeapMoveArray[SafetensorsHeader],
               io_buf: PtrU8) -> Bool:
        """Load `name` into gamma buf if not already cached."""
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
# Planning: tasks + source headers → list of self-contained plans
# =============================================================================


def find_tensor(headers: HeapMoveArray[SafetensorsHeader],
                name: String) -> Tuple[Int, TensorMeta]:
    for i in range(len(headers)):
        var opt = headers[i].tensors.get(name)
        if opt:
            return (i, opt.value().copy())
    return (-1, TensorMeta(DType.uint8, List[Int](), 0, 0))


def fold_shape(shape: List[Int]) -> Tuple[Int, Int]:
    if len(shape) == 3:
        return (shape[0] * shape[1], shape[2])
    elif len(shape) == 2:
        return (shape[0], shape[1])
    else:
        return (shape[0], 1)


@fieldwise_init
struct PlanBundle(Movable):
    var plans: List[QuantizePlan]
    var entries: List[OutputEntry]
    var max_quant_cols: Int
    var max_num_blocks: Int


def plan_quantization(
    tasks: List[QuantizeTask],
    headers: HeapMoveArray[SafetensorsHeader],
) -> Optional[PlanBundle]:
    """Resolve every task against the source checkpoint, computing output
    offsets in a single pass. Returns a bundle the caller walks twice:
    once to write the header, once to process.
    """
    var plans = List[QuantizePlan](capacity=len(tasks))
    var entries = List[OutputEntry]()
    var offset = 0
    var max_quant_cols = 0
    var max_num_blocks = 1

    for i in range(len(tasks)):
        var t = tasks[i]

        if t.kind == WeightTag.QUANTIZABLE or t.kind == WeightTag.GAMMA_QUANTIZABLE:
            var located = find_tensor(headers, t.src_name)
            if located[0] < 0:
                print("quantize: missing weight " + t.src_name)
                return None
            var shard_idx = located[0]
            var meta = located[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            var block = fwht_block_for_weight(t.src_name, cols)
            var weight_bytes = rows * cols
            var scale_bytes = rows * 4

            var weight_off = offset
            offset += weight_bytes
            var scale_off = offset
            offset += scale_bytes

            entries.append(OutputEntry(
                t.src_name, DType.int8, rows, cols,
                weight_off, weight_off + weight_bytes))
            entries.append(OutputEntry(
                t.src_name + "_scale", DType.float32, rows, 1,
                scale_off, scale_off + scale_bytes))

            plans.append(QuantizePlan(
                kind=PlanKind.QUANTIZE,
                src_name=t.src_name, src_shard=shard_idx,
                src_start=headers[shard_idx].data_offset + meta.start,
                src_bytes=weight_bytes * 2,    # bf16 source
                src_dtype=meta.dtype,
                rows=rows, cols=cols, block=block, num_blocks=1,
                gamma_src=t.gamma_src,
                weight_out_off=weight_off, weight_out_bytes=weight_bytes,
                scale_out_off=scale_off, scale_out_bytes=scale_bytes,
            ))
            if cols > max_quant_cols:
                max_quant_cols = cols

        elif t.kind == WeightTag.PER_BLOCK_QUANTIZABLE:
            var located = find_tensor(headers, t.src_name)
            if located[0] < 0:
                print("quantize: missing weight " + t.src_name)
                return None
            var shard_idx = located[0]
            var meta = located[1].copy()
            var rc = fold_shape(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            var block = fwht_block_for_weight(t.src_name, cols)
            var num_blocks = cols // block
            var weight_bytes = rows * cols
            var scale_bytes = rows * num_blocks * 4

            var weight_off = offset
            offset += weight_bytes
            var scale_off = offset
            offset += scale_bytes

            entries.append(OutputEntry(
                t.src_name, DType.int8, rows, cols,
                weight_off, weight_off + weight_bytes))
            entries.append(OutputEntry(
                t.src_name + "_scale", DType.float32, rows, num_blocks,
                scale_off, scale_off + scale_bytes))

            plans.append(QuantizePlan(
                kind=PlanKind.PER_BLOCK_QUANTIZE,
                src_name=t.src_name, src_shard=shard_idx,
                src_start=headers[shard_idx].data_offset + meta.start,
                src_bytes=weight_bytes * 2,
                src_dtype=meta.dtype,
                rows=rows, cols=cols, block=block, num_blocks=num_blocks,
                gamma_src="",
                weight_out_off=weight_off, weight_out_bytes=weight_bytes,
                scale_out_off=scale_off, scale_out_bytes=scale_bytes,
            ))
            if cols > max_quant_cols:
                max_quant_cols = cols
            if num_blocks > max_num_blocks:
                max_num_blocks = num_blocks

        elif t.kind == WeightTag.PASSTHROUGH:
            var located = find_tensor(headers, t.src_name)
            if located[0] < 0:
                # Explicit MISSING — loader-only reservations (e.g., scale
                # slots) that aren't in the pre-quantized source. Recorded
                # so reports see them; skipped during processing.
                plans.append(QuantizePlan(
                    kind=PlanKind.MISSING,
                    src_name=t.src_name, src_shard=-1, src_start=0,
                    src_bytes=0, src_dtype=DType.uint8,
                    rows=0, cols=0, block=0, num_blocks=0,
                    gamma_src="",
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
                t.src_name, meta.dtype, rc[0], rc[1],
                weight_off, weight_off + byte_size))
            plans.append(QuantizePlan(
                kind=PlanKind.PASSTHROUGH,
                src_name=t.src_name, src_shard=shard_idx,
                src_start=headers[shard_idx].data_offset + meta.start,
                src_bytes=byte_size, src_dtype=meta.dtype,
                rows=rc[0], cols=rc[1], block=0, num_blocks=0,
                gamma_src="",
                weight_out_off=weight_off, weight_out_bytes=byte_size,
                scale_out_off=-1, scale_out_bytes=0,
            ))

    return PlanBundle(plans^, entries^, max_quant_cols, max_num_blocks)


# =============================================================================
# Dispatch helpers — one place for the block-size cascade per kernel family.
# =============================================================================


def run_quant_panel[mask_size: Int](
    block: Int,
    src_ptr: Int, work_ptr: Int, qi_ptr: Int, scales_ptr: Int, gamma_ptr: Int,
    rows: Int, cols: Int, apply_gamma: Bool,
    mut pool: BurstPool[mask_size],
):
    if block == 512:
        quantize_panel_dispatch[mask_size, 512](
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
    elif block == 256:
        quantize_panel_dispatch[mask_size, 256](
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
    elif block == 128:
        quantize_panel_dispatch[mask_size, 128](
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
    elif block == 64:
        quantize_panel_dispatch[mask_size, 64](
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)
    else:
        quantize_panel_dispatch[mask_size, 32](
            src_ptr, work_ptr, qi_ptr, scales_ptr, gamma_ptr, rows, cols, apply_gamma, pool)


def run_per_block_panel[mask_size: Int](
    block: Int,
    src_ptr: Int, work_ptr: Int, qi_ptr: Int, scales_ptr: Int,
    rows: Int, cols: Int,
    mut pool: BurstPool[mask_size],
):
    if block == 512:
        per_block_quantize_panel_dispatch[mask_size, 512](
            src_ptr, work_ptr, qi_ptr, scales_ptr, rows, cols, pool)
    elif block == 256:
        per_block_quantize_panel_dispatch[mask_size, 256](
            src_ptr, work_ptr, qi_ptr, scales_ptr, rows, cols, pool)
    elif block == 128:
        per_block_quantize_panel_dispatch[mask_size, 128](
            src_ptr, work_ptr, qi_ptr, scales_ptr, rows, cols, pool)
    elif block == 64:
        per_block_quantize_panel_dispatch[mask_size, 64](
            src_ptr, work_ptr, qi_ptr, scales_ptr, rows, cols, pool)
    else:
        per_block_quantize_panel_dispatch[mask_size, 32](
            src_ptr, work_ptr, qi_ptr, scales_ptr, rows, cols, pool)


# =============================================================================
# Scratch layout — unchanged algebra, reified into a struct so we can pass
# one argument instead of six raw pointers.
# =============================================================================


@fieldwise_init
struct PanelScratch:
    var io_buf: PtrU8
    var work: PtrF32
    var qi: PtrI8
    var scales: PtrF32
    var gamma: PtrF32
    var panel_rows: Int

    def total_bytes(max_quant_cols: Int, max_num_blocks: Int,
                    panel_rows: Int, copy_chunk_bytes: Int,
                    arena_alignment: Int) -> Int:
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
        return max(copy_chunk_bytes, quant_peak)


# =============================================================================
# Per-plan processors — pure functions of plan + scratch + io.
# No cross-plan state; no `has_gamma` latch; no `entry_idx` counter.
# =============================================================================


def process_quantize_plan[mask_size: Int](
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, mut gamma_cache: GammaCache,
    headers: HeapMoveArray[SafetensorsHeader],
    scratch: PanelScratch, mut pool: BurstPool[mask_size],
    panel_rows: Int,
) -> Bool:
    var apply_gamma = plan.gamma_src != ""
    if apply_gamma:
        if not gamma_cache.ensure(plan.gamma_src, rio, headers, scratch.io_buf):
            return False

    var rows_done = 0
    while rows_done < plan.rows:
        var panel = min(panel_rows, plan.rows - rows_done)
        var panel_bytes = panel * plan.cols * 2
        var src_off = plan.src_start + rows_done * plan.cols * 2
        if not rio.read(plan.src_shard, src_off, scratch.io_buf, panel_bytes):
            print("quantize: failed to read panel for " + plan.src_name)
            return False

        run_quant_panel[mask_size](
            plan.block,
            Int(scratch.io_buf), Int(scratch.work), Int(scratch.qi),
            Int(scratch.scales), Int(scratch.gamma),
            panel, plan.cols, apply_gamma, pool)

        var dst_w = data_start + plan.weight_out_off + rows_done * plan.cols
        if not rio.write(output_file_idx, dst_w,
                scratch.qi.bitcast[UInt8](), panel * plan.cols):
            return False
        var dst_s = data_start + plan.scale_out_off + rows_done * 4
        if not rio.write(output_file_idx, dst_s,
                scratch.scales.bitcast[UInt8](), panel * 4):
            return False
        rows_done += panel
    return True


def process_per_block_plan[mask_size: Int](
    plan: QuantizePlan, data_start: Int, output_file_idx: Int,
    mut rio: RingIO, scratch: PanelScratch,
    mut pool: BurstPool[mask_size], panel_rows: Int,
) -> Bool:
    var rows_done = 0
    while rows_done < plan.rows:
        var panel = min(panel_rows, plan.rows - rows_done)
        var panel_bytes = panel * plan.cols * 2
        var src_off = plan.src_start + rows_done * plan.cols * 2
        if not rio.read(plan.src_shard, src_off, scratch.io_buf, panel_bytes):
            return False

        run_per_block_panel[mask_size](
            plan.block,
            Int(scratch.io_buf), Int(scratch.work), Int(scratch.qi),
            Int(scratch.scales),
            panel, plan.cols, pool)

        var dst_w = data_start + plan.weight_out_off + rows_done * plan.cols
        if not rio.write(output_file_idx, dst_w,
                scratch.qi.bitcast[UInt8](), panel * plan.cols):
            return False
        var dst_s = data_start + plan.scale_out_off + rows_done * plan.num_blocks * 4
        if not rio.write(output_file_idx, dst_s,
                scratch.scales.bitcast[UInt8](), panel * plan.num_blocks * 4):
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
    """Generic driver. Gemma4 entrypoint is a 3-liner:
        return run_quantizer[...](build_gemma4_quantizer_tasks(), src, dst)
    """
    var t0 = Int(perf_counter_ns())

    var shard_paths = discover_shards(source_dir)
    if len(shard_paths) == 0:
        print("quantize: no shards in " + String(source_dir))
        return False

    var headers = HeapMoveArray[SafetensorsHeader](len(shard_paths))
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
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
    for p in shard_paths:
        all_paths.append(p)
    all_paths.append(output_path)
    var output_file_idx = len(shard_paths)
    try:
        _ = rio.ring.register_files[ReadWriteMode](all_paths)
    except:
        return False

    if not rio.write(output_file_idx, 0,
            PtrU8(unsafe_from_address=Int(UnsafePointer(to=header_bytes[0]))),
            len(header_bytes)):
        return False

    # --- Phase B: allocate scratch, spin pool, process plans ---
    var scratch_bytes = PanelScratch.total_bytes(
        bundle.max_quant_cols, bundle.max_num_blocks,
        panel_rows, copy_chunk_bytes, arena_alignment)

    var numa = NumaInfo()
    var node = 0 if numa.num_nodes == 0 else numa.plan_topology(1)[0]
    var arena = NumaArena[alignment=arena_alignment](node, scratch_bytes)
    if not arena:
        return False
    _ = arena.prefault()

    var scratch_base = Int(arena.base)
    # Layout matches PanelScratch.total_bytes.
    var panel_elems = panel_rows * bundle.max_quant_cols
    var work_off = align_up(panel_elems * 2, arena_alignment)
    var qi_off = align_up(work_off + panel_elems * 4, arena_alignment)
    var scales_off = align_up(qi_off + panel_elems, arena_alignment)
    var gamma_off = align_up(
        scales_off + panel_rows * bundle.max_num_blocks * 4, arena_alignment)

    var scratch = PanelScratch(
        io_buf=PtrU8(unsafe_from_address=scratch_base),
        work=PtrF32(unsafe_from_address=scratch_base + work_off),
        qi=PtrI8(unsafe_from_address=scratch_base + qi_off),
        scales=PtrF32(unsafe_from_address=scratch_base + scales_off),
        gamma=PtrF32(unsafe_from_address=scratch_base + gamma_off),
        panel_rows=panel_rows,
    )

    var pool = BurstPool[mask_size].for_topology(numa, node)
    var gamma_cache = GammaCache(scratch.gamma, bundle.max_quant_cols)

    var num_quantized = 0
    var total_bytes = 0
    for plan in bundle.plans:
        if plan.kind == PlanKind.QUANTIZE:
            if not process_quantize_plan[mask_size](
                    plan, data_start, output_file_idx, rio, gamma_cache,
                    headers, scratch, pool, panel_rows):
                return False
            total_bytes += plan.weight_out_bytes + plan.scale_out_bytes
            num_quantized += 1
        elif plan.kind == PlanKind.PER_BLOCK_QUANTIZE:
            if not process_per_block_plan[mask_size](
                    plan, data_start, output_file_idx, rio,
                    scratch, pool, panel_rows):
                return False
            total_bytes += plan.weight_out_bytes + plan.scale_out_bytes
            num_quantized += 1
        elif plan.kind == PlanKind.PASSTHROUGH:
            if not process_passthrough_plan(
                    plan, data_start, output_file_idx, rio,
                    scratch, copy_chunk_bytes):
                return False
            total_bytes += plan.weight_out_bytes
        # PlanKind.MISSING: intentional no-op; plan exists for reporting.

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("quantize: " + String(num_quantized) + " weights, "
        + String(total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    return True


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
# Gemma4 entrypoint
# =============================================================================


def quantize_gemma4(source_dir: Path, output_path: Path) -> Bool:
    return run_quantizer(build_gemma4_quantizer_tasks(), source_dir, output_path)
