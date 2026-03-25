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
import linux.sys as linux

from modeling.model_spec import Encoding, Shaped
from simd_math import bf16_load_as

comptime AtomicInt32 = Atomic[DType.int32]

# Below this byte threshold, skip pool dispatch and do work inline.
# Dispatch overhead (~10-20μs) dominates for small tensors.
comptime SMALL_THRESHOLD = 64 * 1024  # 64KB

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
# Dispatch kernels (BurstPool ABI: 6 Int args)
# =============================================================================


def memcpy_kernel(
    dst: Int, src: Int, count: Int,
    n3: Int, n4: Int, n5: Int,
):
    memcpy(
        dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst),
        src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src),
        count=count,
    )


def fused_reduce_gather_kernel(
    config_addr: Int, start_element: Int, end_element: Int,
    my_rank: Int, n4: Int, n5: Int,
):
    """Each BurstPool worker: reduce slice → signal → pull from completed ranks.

    Reads FusedConfig from config_addr for buffer pointers, completion state,
    and chunk layout. The reduce reads from all tp source buffers and writes
    locally. After the last worker on a rank finishes, it sets the done flag.
    All workers then pull completed chunks from other ranks, dividing the
    copy work among themselves.
    """
    var cfg = UnsafePointer[FusedConfig, MutAnyOrigin](unsafe_from_address=config_addr)
    var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=cfg[].ptrs_addr)
    var state_base = cfg[].state_base
    var chunk = cfg[].chunk
    var rem = cfg[].rem
    var tp = cfg[].tp
    var sys = linux.linux_sys()

    var my_buf = ptrs[my_rank]
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=my_buf)
    comptime width = simd_width_of[DType.float32]()

    # --- Reduce my slice from all tp sources ---
    # First source as base, accumulate the rest.
    var src0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
    var i = start_element
    while i + width <= end_element:
        var acc = bf16_load_as[DType.float32, width](src0, i)
        for r in range(1, tp):
            var src_r = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            acc += bf16_load_as[DType.float32, width](src_r, i)
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
    var worker_slice = end_element - start_element
    var worker_idx = 0
    if worker_slice > 0:
        worker_idx = (start_element - my_rank * chunk) // worker_slice
    var num_workers = (chunk + worker_slice - 1) // worker_slice if worker_slice > 0 else 1

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr(state_base, src_rank)) == 0:
            sys.arch_cpu_relax()

        var src_chunk_start = src_rank * chunk
        var src_chunk_count = chunk + (rem if src_rank == tp - 1 else 0)

        var copy_per_worker = (src_chunk_count + num_workers - 1) // num_workers
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

    # Small-tensor fast path: sequential memcpy, no dispatch overhead.
    if total_bytes < SMALL_THRESHOLD:
        for r in range(1, tp):
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[r]),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[0]),
                count=total_bytes,
            )
        return

    # Dispatch: each destination rank pulls from rank 0.
    for r in range(1, tp):
        var pack = pool_ptrs[r][].args_base
        pack[].arg0 = dst_ptrs[r]
        pack[].arg1 = dst_ptrs[0]
        pack[].arg2 = total_bytes
        pool_ptrs[r][].dispatch(memcpy_kernel, pool_ptrs[r][].args_base, 1)
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
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var total_bytes = total_elements * T.ELEMENT_BYTES

    # Small-tensor fast path: single-threaded reduce + broadcast.
    if total_bytes < SMALL_THRESHOLD:
        comptime width = simd_width_of[DType.float32]()
        var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
        for r in range(1, tp):
            var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[r])
            var i = 0
            while i + width <= total_elements:
                var d = bf16_load_as[DType.float32, width](dst, i)
                var s = bf16_load_as[DType.float32, width](src, i)
                (dst + i).store((d + s).cast[DType.bfloat16]())
                i += width
            while i < total_elements:
                dst[i] = Scalar[DType.bfloat16](Float32(dst[i]) + Float32(src[i]))
                i += 1
        for r in range(1, tp):
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[r]),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[0]),
                count=total_bytes,
            )
        return

    var chunk = total_elements // tp
    var rem = total_elements - chunk * tp

    # Per-rank completion state (cache-line padded, stack-allocated).
    var state_mem = InlineArray[UInt8, tp * RANK_STATE_STRIDE](fill=0)
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
            var pack = pool_ptrs[r][].args_base + w
            pack[].arg0 = config_addr
            pack[].arg1 = w_start
            pack[].arg2 = w_end
            pack[].arg3 = r

        pool_ptrs[r][].dispatch(fused_reduce_gather_kernel, pool_ptrs[r][].args_base, num_workers)

    for r in range(tp):
        pool_ptrs[r][].join()
