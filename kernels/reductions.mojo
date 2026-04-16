"""NUMA-aware collective operations for tensor parallelism.

Broadcast: parallel pull — all destination ranks memcpy from source
simultaneously via per-node BurstPool workers.

Allreduce: fused multi-core reduce + flag-signaled parallel pull.
Each node's full BurstPool reduces its chunk from all sources, the
last worker signals completion via atomic flag, then all workers
immediately pull completed chunks from other ranks. Single dispatch
per node, no synchronization between reduce and gather phases.
"""

from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray
from std.sys.info import simd_width_of
from std.os.atomic import Atomic, Consistency
from threading import BurstPool
from threading.threading_shared import ptr as tptr
import linux.sys as linux

from modeling.model_spec import Encoding, Shaped

comptime AtomicInt32 = Atomic[DType.int32]

# Per-rank completion state for fused allreduce.
# Each rank's state is at base + rank * 64 (cache-line isolated).
#   offset 0: remaining workers counter (Int32, atomic)
#   offset 8: done flag (Int32, atomic)
comptime RANK_STATE_STRIDE = 64
comptime COUNTER_OFF = 0
comptime DONE_OFF = 8


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


# =============================================================================
# Small-tensor replicated reduce
# =============================================================================
#
# For small tensors the multi-worker allreduce is dispatch-dominated.
# Each rank dispatches a single worker that reduces the full tensor from
# all sources. Every rank gets the complete result with no gather phase,
# no atomics, and no cross-rank synchronization.


struct SmallReduceConfig:
    var ptrs_addr: Int
    var total_elements: Int
    var tp: Int

    def __init__(out self):
        self.ptrs_addr = 0
        self.total_elements = 0
        self.tp = 0


@fieldwise_init
struct SmallReduceArgs(Copyable, ImplicitlyCopyable):
    var config_addr: Int
    var my_rank: Int


def small_reduce_kernel(args: SmallReduceArgs):
    """Replicated reduce: single worker reduces the full tensor from all ranks."""
    var cfg = tptr[SmallReduceConfig](args.config_addr)
    var ptrs = tptr[Int](cfg[].ptrs_addr)
    var total_elements = cfg[].total_elements
    var tp = cfg[].tp
    var dst = tptr[Scalar[DType.bfloat16]](ptrs[args.my_rank])
    comptime width = simd_width_of[DType.float32]()

    var src0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=ptrs[0])
    var i = 0
    while i + width <= total_elements:
        var acc = (src0 + i).load[width=width]().cast[DType.float32]()
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ptrs[r])
            acc += (src_r + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
        i += width
    while i < total_elements:
        var acc = Float32(src0[i])
        for r in range(1, tp):
            acc += Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ptrs[r])[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1


def small_allreduce[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp],
    seq_len: Int,
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
):
    """Replicated allreduce for small tensors. Each rank's single worker
    reduces the full tensor from all sources — no chunking, no atomics,
    no gather phase. All writes NUMA-local.

    Same signature as ring_allreduce for easy substitution at call sites.
    """
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var sr_cfg = SmallReduceConfig()
    sr_cfg.ptrs_addr = Int(UnsafePointer(to=ptrs))
    sr_cfg.total_elements = total_elements
    sr_cfg.tp = tp
    var sr_config_addr = Int(UnsafePointer(to=sr_cfg))

    var sr_args = SmallReduceArgs(0, 0)
    for r in range(tp):
        sr_args = SmallReduceArgs(sr_config_addr, r)
        pool_ptrs[r][].dispatch[SmallReduceArgs, small_reduce_kernel](
            UnsafePointer(to=sr_args), 1)
    for r in range(tp):
        pool_ptrs[r][].join()


# =============================================================================
# Dispatch args structs
# =============================================================================


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


# =============================================================================
# Dispatch kernels
# =============================================================================


def memcpy_kernel(args: MemcpyArgs):
    memcpy(
        dest=tptr[Byte](args.dst),
        src=tptr[Byte](args.src),
        count=args.count,
    )


def fused_reduce_gather_kernel(args: FusedReduceGatherArgs):
    """Each BurstPool worker: reduce slice -> signal -> pull from completed ranks.

    Reads FusedConfig from config_addr for buffer pointers, completion state,
    and chunk layout. The reduce reads from all tp source buffers and writes
    locally. After the last worker on a rank finishes, it sets the done flag.
    All workers then pull completed chunks from other ranks, dividing the
    copy work among themselves.
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

    var my_buf = ptrs[my_rank]
    var dst = tptr[Scalar[DType.bfloat16]](my_buf)
    comptime width = simd_width_of[DType.float32]()

    # --- Reduce my slice from all tp sources ---
    # First source as base, accumulate the rest.
    var src0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
    var i = start_element
    while i + width <= end_element:
        var acc = (src0 + i).load[width=width]().cast[DType.float32]()
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            acc += (src_r + i).load[width=width]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())
        i += width
    while i < end_element:
        var acc = Float32(src0[i])
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            acc += Float32(src_r[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1

    # --- Signal completion ---
    var old = AtomicInt32.fetch_add[ordering=Consistency.ACQUIRE_RELEASE](
        counter_ptr(state_base, my_rank), -1
    )
    if old == 1:
        AtomicInt32.store[ordering=Consistency.RELEASE](
            done_ptr(state_base, my_rank), 1
        )

    # --- Pull from other ranks as they complete ---
    var workers = num_workers if num_workers > 0 else 1

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr(state_base, src_rank)) == 0:
            sys.arch_cpu_relax()

        var src_chunk_start = src_rank * chunk
        var src_chunk_count = chunk + (rem if src_rank == tp - 1 else 0)

        var copy_per_worker = (src_chunk_count + workers - 1) // workers
        var copy_start = src_chunk_start + worker_idx * copy_per_worker
        var copy_end = min(copy_start + copy_per_worker, src_chunk_start + src_chunk_count)

        if copy_start < copy_end:
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=my_buf + copy_start * 2),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[src_rank] + copy_start * 2),
                count=(copy_end - copy_start) * 2,
            )


struct FusedConfig:
    """Shared configuration for fused allreduce workers.
    Allocated on the caller's stack, accessed by workers via raw pointer.
    Lifetime guaranteed by the caller blocking on pool join."""
    var ptrs_addr: Int
    var state_base: Int
    var chunk: Int
    var rem: Int
    var tp: Int

    def __init__(out self):
        self.ptrs_addr = 0
        self.state_base = 0
        self.chunk = 0
        self.rem = 0
        self.tp = 0


# =============================================================================
# Broadcast: parallel pull
# =============================================================================


def ring_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int,
    dst_ptrs: InlineArray[Int, tp],
    seq_len: Int,
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
):
    """Parallel pull broadcast. All destination ranks memcpy from source
    simultaneously via per-node workers. ~26 GB/s aggregate on 4 NUMA nodes.
    """
    var total_bytes = seq_len * T.COLS * T.ELEMENT_BYTES
    if total_bytes <= 0 or tp <= 1:
        return

    # Ensure rank 0 has the data.
    if src_ptr != dst_ptrs[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )

    # Dispatch: each destination rank pulls from rank 0.
    var mcpy_jobs = InlineArray[MemcpyArgs, tp](
        fill=MemcpyArgs(0, 0, 0)
    )
    for r in range(1, tp):
        mcpy_jobs[r] = MemcpyArgs(dst_ptrs[r], dst_ptrs[0], total_bytes)
        pool_ptrs[r][].dispatch[MemcpyArgs, memcpy_kernel](
            UnsafePointer(to=mcpy_jobs[r]), 1)
    for r in range(1, tp):
        pool_ptrs[r][].join()


# =============================================================================
# Allreduce: fused multi-core reduce + flag-signaled pull
# =============================================================================


def ring_allreduce[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp],
    seq_len: Int,
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
):
    """Fused allreduce. Each node's full BurstPool reduces its chunk from
    all sources, signals completion, then all workers pull completed chunks
    from other ranks. Single dispatch per node, no sync between phases.
    ~25 GB/s on 4 NUMA nodes.

    Workers partition the chunk into row slices. Each worker reduces its
    slice by reading from all tp source buffers with f32 SIMD accumulation.
    The last worker to finish atomically sets a done flag. All workers then
    transition to the allgather: polling other ranks' done flags and copying
    their chunks locally, with the copy work divided among all workers.
    """
    comptime assert T.DTYPE == DType.bfloat16, "ring_allreduce: only bf16 tensors are supported"
    comptime assert T.ELEMENT_BYTES == 2, "ring_allreduce: bf16 byte width mismatch"
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var total_bytes = total_elements * T.ELEMENT_BYTES

    var chunk = total_elements // tp
    var rem = total_elements - chunk * tp

    # Per-rank completion state (cache-line padded, stack-allocated).
    # Atomic counters/flags need stronger-than-byte alignment.
    var state_mem = InlineArray[Int64, tp * (RANK_STATE_STRIDE // 8)](fill=0)
    var state_base = Int(UnsafePointer(to=state_mem))

    # Initialize counters.
    for r in range(tp):
        var num_workers = pool_ptrs[r][].capacity
        AtomicInt32.store[ordering=Consistency.RELEASE](
            counter_ptr(state_base, r), Int32(num_workers)
        )

    # Shared config.
    var cfg = FusedConfig()
    cfg.ptrs_addr = Int(UnsafePointer(to=ptrs))
    cfg.state_base = state_base
    cfg.chunk = chunk
    cfg.rem = rem
    cfg.tp = tp

    var config_addr = Int(UnsafePointer(to=cfg))

    # Dispatch: each rank's full pool, workers partition the chunk.
    var jobs = InlineArray[FusedReduceGatherArgs, 128](
        fill=FusedReduceGatherArgs(0, 0, 0, 0, 0, 0)
    )
    for r in range(tp):
        var rank_start = r * chunk
        var rank_count = chunk + (rem if r == tp - 1 else 0)
        var num_workers = pool_ptrs[r][].capacity
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

        pool_ptrs[r][].dispatch[FusedReduceGatherArgs, fused_reduce_gather_kernel](
            UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        pool_ptrs[r][].join()


# =============================================================================
# Allgather: parallel pull — each rank assembles all shards locally
# =============================================================================


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


def ring_allgather[tp: Int](
    src_ptrs: InlineArray[Int, tp],
    dst_ptrs: InlineArray[Int, tp],
    shard_bytes: Int,
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
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
        var num_workers = pool_ptrs[r][].capacity
        for w in range(num_workers):
            jobs[w] = AllGatherArgs(config_addr, r, w, num_workers)
        pool_ptrs[r][].dispatch[AllGatherArgs, allgather_kernel](
            UnsafePointer(to=jobs[0]), num_workers)

    for r in range(tp):
        pool_ptrs[r][].join()
