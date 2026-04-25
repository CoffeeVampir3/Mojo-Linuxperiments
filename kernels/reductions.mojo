from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray
from std.sys.info import simd_width_of
from std.atomic import Atomic, Ordering
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr
from notstdcollections import HeapMoveArray
import linux.sys as linux

from modeling.model_spec import Encoding, Shaped, ShapeLike

comptime AtomicInt32 = Atomic[DType.int32]

# Per-rank completion state for fused allreduce.
# Each rank's state is at base + rank * 64 (cache-line isolated).
#   offset 0: remaining workers counter (Int32, atomic)
#   offset 8: done flag (Int32, atomic)
comptime RANK_STATE_STRIDE = 64
comptime COUNTER_OFF = 0
comptime DONE_OFF = 8
comptime TRACE_PARALLEL_ALLREDUCE = False


@always_inline
def counter_ptr(state_base: Int, rank: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=state_base + rank * RANK_STATE_STRIDE + COUNTER_OFF
    )


@always_inline
def done_ptr(state_base: Int, rank: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=state_base + rank * RANK_STATE_STRIDE + DONE_OFF
    )


def small_allreduce[
    E: Encoding, S: ShapeLike, tp: Int, residual_add: Bool = False,
](
    ptrs: InlineArray[Int, tp],
    seq_len: Int,
    dst_ptrs: InlineArray[Int, tp] = InlineArray[Int, tp](fill=0),
):
    """Replicated allreduce for small tensors. Main thread reduces all
    sources into ptrs[0], then parallel broadcast to remaining ranks.

    residual_add=True: fused allreduce + residual add. Reduces ptrs[0..tp],
    adds to dst_ptrs (x_main), broadcasts updated dst_ptrs[0] to other ranks.
    Saves one full read+write pass of the reduced buffer.
    """
    var total = seq_len * S.M
    if total <= 0 or tp <= 1:
        comptime if residual_add:
            if total > 0 and tp == 1:
                comptime width = simd_width_of[DType.bfloat16]()
                var src = tptr[Scalar[DType.bfloat16]](ptrs[0])
                var dst = tptr[Scalar[DType.bfloat16]](dst_ptrs[0])
                for i in range(0, total, width):
                    var s = src.load[width=width](i).cast[DType.float32]()
                    var d = dst.load[width=width](i).cast[DType.float32]()
                    dst.store(i, (d + s).cast[DType.bfloat16]())
        return

    comptime width = simd_width_of[DType.bfloat16]()

    comptime if residual_add:
        var dst = tptr[Scalar[DType.bfloat16]](dst_ptrs[0])
        for i in range(0, total, width):
            var acc = tptr[Scalar[DType.bfloat16]](ptrs[0]).load[width=width](i).cast[DType.float32]()
            for r in range(1, tp):
                acc += tptr[Scalar[DType.bfloat16]](ptrs[r]).load[width=width](i).cast[DType.float32]()
            var old = dst.load[width=width](i).cast[DType.float32]()
            dst.store(i, (old + acc).cast[DType.bfloat16]())
    else:
        var dst = tptr[Scalar[DType.bfloat16]](ptrs[0])
        for i in range(0, total, width):
            var acc = dst.load[width=width](i).cast[DType.float32]()
            for r in range(1, tp):
                acc += tptr[Scalar[DType.bfloat16]](ptrs[r]).load[width=width](i).cast[DType.float32]()
            dst.store(i, acc.cast[DType.bfloat16]())

    var total_bytes = total * E.ELEMENT_BYTES
    comptime if residual_add:
        for r in range(1, tp):
            memcpy(
                dest=tptr[Byte](dst_ptrs[r]),
                src=tptr[Byte](dst_ptrs[0]),
                count=total_bytes)
    else:
        for r in range(1, tp):
            memcpy(
                dest=tptr[Byte](ptrs[r]),
                src=tptr[Byte](ptrs[0]),
                count=total_bytes)


@fieldwise_init
struct MemcpyArgs(Copyable, ImplicitlyCopyable):
    var dst: Int
    var src: Int
    var count: Int


@fieldwise_init
struct FusedReduceGatherArgs(Copyable, ImplicitlyCopyable):
    var config_addr: Int
    var start_element: Int
    var end_element: Int
    var my_rank: Int
    var worker_idx: Int
    var num_workers: Int


@fieldwise_init
struct ParallelReduceArgs[tp: Int](Copyable, ImplicitlyCopyable):
    var src_ptrs: InlineArray[Int, Self.tp]
    var dst_ptrs: InlineArray[Int, Self.tp]
    var chunk: Int
    var rem: Int
    var start_element: Int
    var end_element: Int
    var my_rank: Int
    var worker_idx: Int
    var num_workers: Int


def make_parallel_reduce_args[tp: Int](
    ptrs: InlineArray[Int, tp],
    dst_ptrs: InlineArray[Int, tp],
    chunk: Int,
    rem: Int,
    start_element: Int,
    end_element: Int,
    my_rank: Int,
    worker_idx: Int,
    num_workers: Int,
) -> ParallelReduceArgs[tp]:
    comptime assert tp > 0, "parallel allreduce requires at least one rank"
    return ParallelReduceArgs[tp](
        ptrs, dst_ptrs, chunk, rem,
        start_element, end_element, my_rank, worker_idx, num_workers,
    )


def memcpy_kernel(args: MemcpyArgs):
    memcpy(
        dest=tptr[Byte](args.dst),
        src=tptr[Byte](args.src),
        count=args.count,
    )


def fused_reduce_gather_kernel(args: FusedReduceGatherArgs):
    fused_reduce_gather_impl[False](args)


def fused_reduce_gather_add_kernel(args: FusedReduceGatherArgs):
    fused_reduce_gather_impl[True](args)


def reduce_chunk_kernel[tp: Int](args: ParallelReduceArgs[tp]):
    reduce_chunk_impl[False, tp](args)


def reduce_chunk_add_kernel[tp: Int](args: ParallelReduceArgs[tp]):
    reduce_chunk_impl[True, tp](args)


def gather_chunks_kernel[tp: Int](args: ParallelReduceArgs[tp]):
    gather_chunks_impl[False, tp](args)


def gather_chunks_add_kernel[tp: Int](args: ParallelReduceArgs[tp]):
    gather_chunks_impl[True, tp](args)


def reduce_chunk_impl[residual_add: Bool, tp: Int](args: ParallelReduceArgs[tp]):
    """Worker-parallel reduce phase without worker-side cross-rank waits."""
    var my_rank = args.my_rank
    var my_out_addr = args.src_ptrs[my_rank]
    comptime if residual_add:
        my_out_addr = args.dst_ptrs[my_rank]

    var dst = tptr[Scalar[DType.bfloat16]](my_out_addr)
    comptime width = simd_width_of[DType.float32]()

    var i = args.start_element
    while i < args.end_element and i % width != 0:
        var acc = Float32(tptr[Scalar[DType.bfloat16]](args.src_ptrs[0])[i])
        for r in range(1, tp):
            acc += Float32(tptr[Scalar[DType.bfloat16]](args.src_ptrs[r])[i])
        comptime if residual_add:
            acc += Float32(dst[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1
    while i + width <= args.end_element:
        var acc = tptr[Scalar[DType.bfloat16]](args.src_ptrs[0]).load[
            width=width](i).cast[DType.float32]()
        for r in range(1, tp):
            acc += tptr[Scalar[DType.bfloat16]](args.src_ptrs[r]).load[
                width=width](i).cast[DType.float32]()
        comptime if residual_add:
            acc += (dst + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
        i += width
    while i < args.end_element:
        var acc = Float32(tptr[Scalar[DType.bfloat16]](args.src_ptrs[0])[i])
        for r in range(1, tp):
            acc += Float32(tptr[Scalar[DType.bfloat16]](args.src_ptrs[r])[i])
        comptime if residual_add:
            acc += Float32(dst[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1


def gather_chunks_impl[residual_add: Bool, tp: Int](args: ParallelReduceArgs[tp]):
    """Worker-parallel gather phase after the main thread joins reduction."""
    var my_rank = args.my_rank
    var my_out_addr = args.src_ptrs[my_rank]
    comptime if residual_add:
        my_out_addr = args.dst_ptrs[my_rank]

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        var src_chunk_start = src_rank * args.chunk
        var src_chunk_count = args.chunk
        if src_rank == tp - 1:
            src_chunk_count += args.rem
        var src_chunk_end = src_chunk_start + src_chunk_count

        var copy_start = max(args.start_element, src_chunk_start)
        var copy_end = min(args.end_element, src_chunk_end)
        if copy_start < copy_end:
            var src_addr = args.src_ptrs[src_rank]
            comptime if residual_add:
                src_addr = args.dst_ptrs[src_rank]
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](
                    unsafe_from_address=my_out_addr + copy_start * 2),
                src=UnsafePointer[Byte, MutAnyOrigin](
                    unsafe_from_address=src_addr + copy_start * 2),
                count=(copy_end - copy_start) * 2,
            )


def fused_reduce_gather_impl[residual_add: Bool](args: FusedReduceGatherArgs):
    """Each BurstPool worker: reduce slice -> signal -> pull from completed ranks.

    residual_add=True: reduce writes (dst + acc) to dst instead of acc to src.
    The gather phase copies from dst (x_main) instead of src (x_residual).
    """
    var config_addr = args.config_addr
    var start_element = args.start_element
    var end_element = args.end_element
    var my_rank = args.my_rank
    var worker_idx = args.worker_idx
    var num_workers = args.num_workers

    var cfg = tptr[FusedConfig](config_addr)
    var ptrs = tptr[Int](cfg[].ptrs_addr)
    var state_base = cfg[].state_base
    var chunk = cfg[].chunk
    var rem = cfg[].rem
    var tp = cfg[].tp
    var sys = linux.linux_sys()

    var my_out_addr = ptrs[my_rank]
    var gather_src_addr = cfg[].ptrs_addr
    comptime if residual_add:
        var dst_ptrs_raw = tptr[Int](cfg[].dst_ptrs_addr)
        my_out_addr = dst_ptrs_raw[my_rank]
        gather_src_addr = cfg[].dst_ptrs_addr
    var dst = tptr[Scalar[DType.bfloat16]](my_out_addr)
    var gather_src = tptr[Int](gather_src_addr)
    comptime width = simd_width_of[DType.float32]()

    var src0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
    var i = start_element
    while i + width <= end_element:
        var acc = (src0 + i).load[width=width]().cast[DType.float32]()
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            acc += (src_r + i).load[width=width]().cast[DType.float32]()
        comptime if residual_add:
            acc += (dst + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
        i += width
    while i < end_element:
        var acc = Float32(src0[i])
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            acc += Float32(src_r[i])
        comptime if residual_add:
            acc += Float32(dst[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1

    var old = AtomicInt32.fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
        counter_ptr(state_base, my_rank), -1
    )
    if old == 1:
        AtomicInt32.store[ordering=Ordering.RELEASE](
            done_ptr(state_base, my_rank), 1
        )

    var workers = num_workers if num_workers > 0 else 1

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        while AtomicInt32.load[ordering=Ordering.ACQUIRE](done_ptr(state_base, src_rank)) == 0:
            sys.arch_cpu_relax()

        var src_chunk_start = src_rank * chunk
        var src_chunk_count = chunk + (rem if src_rank == tp - 1 else 0)

        var copy_per_worker = (src_chunk_count + workers - 1) // workers
        var copy_start = src_chunk_start + worker_idx * copy_per_worker
        var copy_end = min(copy_start + copy_per_worker, src_chunk_start + src_chunk_count)

        if copy_start < copy_end:
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=my_out_addr + copy_start * 2),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=gather_src[src_rank] + copy_start * 2),
                count=(copy_end - copy_start) * 2,
            )


struct FusedConfig:
    """Shared configuration for fused allreduce workers.
    Allocated on the caller's stack, accessed by workers via raw pointer.
    Lifetime guaranteed by the caller blocking on pool join."""
    var ptrs_addr: Int
    var dst_ptrs_addr: Int
    var state_base: Int
    var chunk: Int
    var rem: Int
    var tp: Int

    def __init__(out self):
        self.ptrs_addr = 0
        self.dst_ptrs_addr = 0
        self.state_base = 0
        self.chunk = 0
        self.rem = 0
        self.tp = 0


def ring_broadcast[
    P: BurstThreadPool, //,
    E: Encoding, S: ShapeLike, tp: Int,
](
    src_ptr: Int,
    dst_ptrs: InlineArray[Int, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    """Parallel pull broadcast. All destination ranks memcpy from source
    simultaneously via per-node workers. ~26 GB/s aggregate on 4 NUMA nodes.
    """
    var total_bytes = S.rows_bytes[E](seq_len)
    if total_bytes <= 0 or tp <= 1:
        return

    if src_ptr != dst_ptrs[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )

    var mcpy_jobs = InlineArray[MemcpyArgs, tp](
        fill=MemcpyArgs(0, 0, 0)
    )
    for r in range(1, tp):
        mcpy_jobs[r] = MemcpyArgs(dst_ptrs[r], dst_ptrs[0], total_bytes)
        pools[r].dispatch[MemcpyArgs, memcpy_kernel](
            UnsafePointer(to=mcpy_jobs[r]), 1)
    for r in range(1, tp):
        pools[r].join()


def ring_allreduce[
    P: BurstThreadPool, //,
    E: Encoding, S: ShapeLike, tp: Int, residual_add: Bool = False,
](
    ptrs: InlineArray[Int, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
    dst_ptrs: InlineArray[Int, tp] = InlineArray[Int, tp](fill=0),
):
    """Fused allreduce. Each node's full BurstPool reduces its chunk from
    all sources, signals completion, then all workers pull completed chunks
    from other ranks. Single dispatch per node, no sync between phases.
    ~25 GB/s on 4 NUMA nodes.

    residual_add=True: fused allreduce + residual add. Reduces ptrs[0..tp],
    adds to dst_ptrs (x_main), gathers from dst_ptrs. x_main must be
    identical across ranks before the call (replicated residual stream).
    """
    comptime assert E.DTYPE == DType.bfloat16, "ring_allreduce: only bf16 tensors are supported"
    comptime assert E.ELEMENT_BYTES == 2, "ring_allreduce: bf16 byte width mismatch"
    comptime cols = S.M
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        comptime if residual_add:
            if total_elements > 0 and tp == 1:
                comptime width = simd_width_of[DType.bfloat16]()
                var src = tptr[Scalar[DType.bfloat16]](ptrs[0])
                var dst = tptr[Scalar[DType.bfloat16]](dst_ptrs[0])
                comptime fwidth = simd_width_of[DType.float32]()
                var j = 0
                while j + fwidth <= total_elements:
                    var s = (src + j).load[width=fwidth]().cast[DType.float32]()
                    var d = (dst + j).load[width=fwidth]().cast[DType.float32]()
                    (dst + j).store((d + s).cast[DType.bfloat16]())
                    j += fwidth
        return

    var chunk = total_elements // tp
    var rem = total_elements - chunk * tp

    var state_mem = InlineArray[Int64, tp * (RANK_STATE_STRIDE // 8)](fill=0)
    var state_base = Int(UnsafePointer(to=state_mem))

    for r in range(tp):
        var num_workers = pools[r].get_capacity()
        AtomicInt32.store[ordering=Ordering.RELEASE](
            counter_ptr(state_base, r), Int32(num_workers)
        )

    var cfg = FusedConfig()
    cfg.ptrs_addr = Int(UnsafePointer(to=ptrs))
    comptime if residual_add:
        cfg.dst_ptrs_addr = Int(UnsafePointer(to=dst_ptrs))
    cfg.state_base = state_base
    cfg.chunk = chunk
    cfg.rem = rem
    cfg.tp = tp

    var config_addr = Int(UnsafePointer(to=cfg))

    var jobs = InlineArray[FusedReduceGatherArgs, 128](
        fill=FusedReduceGatherArgs(0, 0, 0, 0, 0, 0)
    )
    for r in range(tp):
        var rank_start = r * chunk
        var rank_count = chunk + (rem if r == tp - 1 else 0)
        var num_workers = pools[r].get_capacity()
        var rows_per_worker = (rank_count + num_workers - 1) // num_workers

        for w in range(num_workers):
            var w_start = rank_start + w * rows_per_worker
            var w_end = min(w_start + rows_per_worker, rank_start + rank_count)
            if w_start >= rank_start + rank_count:
                w_start = rank_start + rank_count
                w_end = w_start
            jobs[w] = FusedReduceGatherArgs(
                config_addr, w_start, w_end, r, w, num_workers
            )

        comptime if residual_add:
            pools[r].dispatch[FusedReduceGatherArgs, fused_reduce_gather_add_kernel](
                UnsafePointer(to=jobs[0]), num_workers)
        else:
            pools[r].dispatch[FusedReduceGatherArgs, fused_reduce_gather_kernel](
                UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        pools[r].join()


def parallel_allreduce[
    P: BurstThreadPool, //,
    E: Encoding, S: ShapeLike, tp: Int, residual_add: Bool = False,
](
    ptrs: InlineArray[Int, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
    dst_ptrs: InlineArray[Int, tp] = InlineArray[Int, tp](fill=0),
):
    """Two-stage worker-parallel allreduce.

    This uses the same shard ownership as ring_allreduce but keeps the barrier
    on the main thread: dispatch reduce chunks, join all ranks, then dispatch
    gather copies. That avoids worker-side cross-rank spin waits, which are
    hard to debug and can hang the entire pool if one rank fails to signal.
    """
    comptime assert E.DTYPE == DType.bfloat16, "parallel_allreduce: only bf16 tensors are supported"
    comptime assert E.ELEMENT_BYTES == 2, "parallel_allreduce: bf16 byte width mismatch"
    comptime cols = S.M
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        comptime if residual_add:
            if total_elements > 0 and tp == 1:
                comptime width = simd_width_of[DType.float32]()
                var src = tptr[Scalar[DType.bfloat16]](ptrs[0])
                var dst = tptr[Scalar[DType.bfloat16]](dst_ptrs[0])
                var j = 0
                while j + width <= total_elements:
                    var s = (src + j).load[width=width]().cast[DType.float32]()
                    var d = (dst + j).load[width=width]().cast[DType.float32]()
                    (dst + j).store((d + s).cast[DType.bfloat16]())
                    j += width
                while j < total_elements:
                    dst[j] = Scalar[DType.bfloat16](
                        Float32(dst[j]) + Float32(src[j]))
                    j += 1
        return

    var chunk = total_elements // tp
    var rem = total_elements - chunk * tp

    comptime if TRACE_PARALLEL_ALLREDUCE:
        print(
            "DBG parallel_allreduce begin total=", total_elements,
            " chunk=", chunk, " rem=", rem)

    var jobs = InlineArray[ParallelReduceArgs[tp], 128](
        fill=make_parallel_reduce_args[tp](
            ptrs, dst_ptrs, chunk, rem, 0, 0, 0, 0, 1)
    )

    for r in range(tp):
        var rank_start = r * chunk
        var rank_count = chunk + (rem if r == tp - 1 else 0)
        var num_workers = pools[r].get_capacity()
        var elems_per_worker = (rank_count + num_workers - 1) // num_workers
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print(
                "DBG parallel_allreduce reduce dispatch r=", r,
                " workers=", num_workers,
                " rank_start=", rank_start,
                " rank_count=", rank_count,
                " elems_per_worker=", elems_per_worker)
        for w in range(num_workers):
            var w_start = rank_start + w * elems_per_worker
            var w_end = min(w_start + elems_per_worker, rank_start + rank_count)
            if w_start >= rank_start + rank_count:
                w_start = rank_start + rank_count
                w_end = w_start
            jobs[w] = make_parallel_reduce_args[tp](
                ptrs, dst_ptrs, chunk, rem,
                w_start, w_end, r, w, num_workers)
        comptime if residual_add:
            pools[r].dispatch[ParallelReduceArgs[tp], reduce_chunk_add_kernel[tp]](
                UnsafePointer(to=jobs[0]), num_workers)
        else:
            pools[r].dispatch[ParallelReduceArgs[tp], reduce_chunk_kernel[tp]](
                UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print("DBG parallel_allreduce reduce join begin r=", r)
        pools[r].join()
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print("DBG parallel_allreduce reduce join done r=", r)

    for r in range(tp):
        var num_workers = pools[r].get_capacity()
        var elems_per_worker = (
            total_elements + num_workers - 1) // num_workers
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print(
                "DBG parallel_allreduce gather dispatch r=", r,
                " workers=", num_workers,
                " elems_per_worker=", elems_per_worker)
        for w in range(num_workers):
            var w_start = w * elems_per_worker
            var w_end = min(w_start + elems_per_worker, total_elements)
            if w_start >= total_elements:
                w_start = total_elements
                w_end = w_start
            jobs[w] = make_parallel_reduce_args[tp](
                ptrs, dst_ptrs, chunk, rem,
                w_start, w_end, r, w, num_workers)
        comptime if residual_add:
            pools[r].dispatch[ParallelReduceArgs[tp], gather_chunks_add_kernel[tp]](
                UnsafePointer(to=jobs[0]), num_workers)
        else:
            pools[r].dispatch[ParallelReduceArgs[tp], gather_chunks_kernel[tp]](
                UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print("DBG parallel_allreduce gather join begin r=", r)
        pools[r].join()
        comptime if TRACE_PARALLEL_ALLREDUCE:
            print("DBG parallel_allreduce gather join done r=", r)

    comptime if TRACE_PARALLEL_ALLREDUCE:
        print("DBG parallel_allreduce done")


struct AllGatherConfig:
    var src_ptrs_addr: Int
    var dst_ptrs_addr: Int
    var shard_bytes: Int
    var tp: Int

    def __init__(out self):
        self.src_ptrs_addr = 0
        self.dst_ptrs_addr = 0
        self.shard_bytes = 0
        self.tp = 0


@fieldwise_init
struct AllGatherArgs(Copyable, ImplicitlyCopyable):
    var config_addr: Int
    var my_rank: Int
    var worker_idx: Int
    var num_workers: Int


def allgather_kernel(args: AllGatherArgs):
    """Each worker copies its slice of every shard into the local dst buffer."""
    var cfg = tptr[AllGatherConfig](args.config_addr)
    var src_ptrs = tptr[Int](cfg[].src_ptrs_addr)
    var dst_ptrs = tptr[Int](cfg[].dst_ptrs_addr)
    var shard_bytes = cfg[].shard_bytes
    var tp = cfg[].tp
    var my_dst = dst_ptrs[args.my_rank]
    var workers = args.num_workers if args.num_workers > 0 else 1

    var bytes_per_worker = (shard_bytes + workers - 1) // workers
    var start = args.worker_idx * bytes_per_worker
    var end = min(start + bytes_per_worker, shard_bytes)
    if start >= end:
        return

    var count = end - start
    for src in range(tp):
        memcpy(
            dest=tptr[Byte](my_dst + src * shard_bytes + start),
            src=tptr[Byte](src_ptrs[src] + start),
            count=count)


def ring_allgather[
    P: BurstThreadPool, //,
    tp: Int,
](
    src_ptrs: InlineArray[Int, tp],
    dst_ptrs: InlineArray[Int, tp],
    shard_bytes: Int,
    mut pools: HeapMoveArray[P],
):
    """Parallel pull allgather. Each rank's pool workers copy all shards
    into the local concatenated buffer. Shard s is placed at
    dst[rank] + s * shard_bytes. All writes are NUMA-local.

    Caller must ensure source data is ready (e.g. via tp_parallel join)
    before calling — no cross-rank synchronization is performed.
    """
    if shard_bytes <= 0:
        return
    if tp <= 1:
        if src_ptrs[0] != dst_ptrs[0]:
            memcpy(
                dest=tptr[Byte](dst_ptrs[0]),
                src=tptr[Byte](src_ptrs[0]),
                count=shard_bytes)
        return

    var cfg = AllGatherConfig()
    cfg.src_ptrs_addr = Int(UnsafePointer(to=src_ptrs))
    cfg.dst_ptrs_addr = Int(UnsafePointer(to=dst_ptrs))
    cfg.shard_bytes = shard_bytes
    cfg.tp = tp
    var config_addr = Int(UnsafePointer(to=cfg))

    var jobs = InlineArray[AllGatherArgs, 128](
        fill=AllGatherArgs(0, 0, 0, 0))
    for r in range(tp):
        var num_workers = pools[r].get_capacity()
        for w in range(num_workers):
            jobs[w] = AllGatherArgs(config_addr, r, w, num_workers)
        pools[r].dispatch[AllGatherArgs, allgather_kernel](
            UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        pools[r].join()
