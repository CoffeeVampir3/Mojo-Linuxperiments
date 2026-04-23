from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from kernels.vnni import pack_and_colsum_vnni, L2_TARGET
from numa import NumaArena
from threading.threading_traits import BurstThreadPool

from modeling.model_spec import Encoding, ShapeLike, DEFAULT_ALIGNMENT
from modeling.modeling_common import TensorRef


@always_inline
def pack_and_colsum_at[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    arena_base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    block_cols: Int = S.M,
    colsum_row_major: Bool = True,
):
    var src = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=arena_base + weight.offset)
    var colsum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum.offset)
    pack_and_colsum_vnni(
        src, src, scratch,
        S.N, S.M, block_cols,
        colsum_ptr, colsum_row_major,
    )


@always_inline
def colsum_at[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    arena_base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
    block_cols: Int = S.M,
    colsum_row_major: Bool = True,
):
    """Per-row per-block int8 sum without VNNI packing. Used for weights
    that are consumed in their native row-major layout (e.g., the lm_head
    output projection)."""
    debug_assert(block_cols > 0 and S.M % block_cols == 0,
        "colsum_at: block_cols must divide S.M")

    comptime simd_i8 = simd_width_of[DType.int8]()
    debug_assert(block_cols % simd_i8 == 0,
        "colsum_at: block_cols must be a multiple of SIMD int8 width")

    var src = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=arena_base + weight.offset)
    var colsum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum.offset)
    var num_blocks = S.M // block_cols

    for row in range(S.N):
        var src_row = src + row * S.M
        for block_idx in range(num_blocks):
            var acc = SIMD[DType.int32, simd_i8](0)
            var base = block_idx * block_cols
            for k in range(0, block_cols, simd_i8):
                acc += (src_row + base + k).load[width=simd_i8]().cast[DType.int32]()
            var sum = Float32(Int(acc.reduce_add()))
            if colsum_row_major:
                colsum_ptr[row * num_blocks + block_idx] = sum
            else:
                colsum_ptr[block_idx * S.N + row] = sum


@fieldwise_init
struct PackColsumTask(Copyable, ImplicitlyCopyable):
    """One pack+colsum unit of work. Offsets are arena-relative so the task
    list is NUMA-node-independent; the worker adds `arena_base` locally."""
    var arena_base: Int
    var weight_off: Int
    var colsum_off: Int
    var rows: Int
    var cols: Int
    var block_cols: Int
    var colsum_row_major: Bool


@fieldwise_init
struct PackColsumWorkerArgs(Copyable, ImplicitlyCopyable):
    var tasks: UnsafePointer[PackColsumTask, MutAnyOrigin]
    var start: Int
    var end: Int
    var scratch: UnsafePointer[UInt8, MutAnyOrigin]


def pack_colsum_worker_kernel(args: PackColsumWorkerArgs):
    for i in range(args.start, args.end):
        var task = args.tasks[i]
        var src = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=task.arena_base + task.weight_off)
        var colsum_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=task.arena_base + task.colsum_off)
        pack_and_colsum_vnni(
            src, src, args.scratch,
            task.rows, task.cols, task.block_cols,
            colsum_ptr, task.colsum_row_major,
        )


@always_inline
def make_pack_colsum_task[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    arena_base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
    block_cols: Int = S.M,
    colsum_row_major: Bool = True,
) -> PackColsumTask:
    return PackColsumTask(
        arena_base=arena_base,
        weight_off=weight.offset,
        colsum_off=colsum.offset,
        rows=S.N,
        cols=S.M,
        block_cols=block_cols,
        colsum_row_major=colsum_row_major,
    )


def dispatch_pack_colsum_tasks[P: BurstThreadPool](
    mut pool: P,
    numa_node: Int,
    tasks: List[PackColsumTask],
):
    """Partition `tasks` across `pool`'s workers, hand each a NUMA-local
    L2_TARGET scratch, run, join. Scratch is allocated on `numa_node` for
    the duration of the dispatch and freed on return."""
    var num_workers = pool.get_capacity()
    var num_tasks = len(tasks)
    if num_tasks == 0 or num_workers == 0:
        return

    var scratch_arena = NumaArena[alignment=DEFAULT_ALIGNMENT](
        numa_node, num_workers * L2_TARGET)
    debug_assert(scratch_arena.__bool__(),
        "dispatch_pack_colsum_tasks: scratch allocation failed")

    var tasks_ptr = UnsafePointer[PackColsumTask, MutAnyOrigin](
        unsafe_from_address=Int(tasks.unsafe_ptr()))

    var args_list = List[PackColsumWorkerArgs](capacity=num_workers)
    for w in range(num_workers):
        var start = (num_tasks * w) // num_workers
        var end = (num_tasks * (w + 1)) // num_workers
        var scratch = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(scratch_arena.base) + w * L2_TARGET)
        args_list.append(PackColsumWorkerArgs(
            tasks=tasks_ptr, start=start, end=end, scratch=scratch))

    var args_ptr = UnsafePointer[PackColsumWorkerArgs, MutAnyOrigin](
        unsafe_from_address=Int(args_list.unsafe_ptr()))
    pool.dispatch[PackColsumWorkerArgs, pack_colsum_worker_kernel](
        args_ptr, num_workers)
    pool.join()
    _ = args_list^
    _ = tasks
    _ = scratch_arena^


@always_inline
def zero_pad_tail[E: Encoding, S: ShapeLike](base: Int, slot: TensorRef[E, S]):
    """Zero the alignment padding rows at the end of a weight matrix.
    Uses S.DATA_N (logical rows) and S.N (padded rows). No-op when unpadded."""
    if S.DATA_N >= S.N:
        return
    var start = base + slot.offset + S.DATA_N * S.M * E.ELEMENT_BYTES
    var nbytes = (S.N - S.DATA_N) * S.M * E.ELEMENT_BYTES
    comptime width = simd_width_of[DType.uint8]()
    var ptr = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](unsafe_from_address=start)
    var i = 0
    while i + width <= nbytes:
        (ptr + i).store(SIMD[DType.uint8, width](0))
        i += width
    while i < nbytes:
        ptr[i] = 0
        i += 1
