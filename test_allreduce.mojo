"""Allreduce: fused reduce+gather in a single dispatch per node.

Each node's full BurstPool is dispatched once. Every worker:
  1. Reduces its slice of the local chunk from all 4 sources
  2. Atomically decrements a per-rank completion counter
  3. The last worker to finish (counter hits 0) sets a "done" flag
  4. All workers then poll the other 3 ranks' done flags
  5. As each flag appears, pull that rank's chunk into local buffer

No pool-level join between reduce and gather. No barriers.
Each rank transitions from reduce to gather independently.
Fast ranks start pulling before slow ranks finish.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.os.atomic import Atomic, Consistency
from std.collections import InlineArray
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from simd_math import bf16_load_as
import linux.sys as linux

comptime PoolPtr = UnsafePointer[BurstPool[], MutAnyOrigin]
comptime AtomicInt32 = Atomic[DType.int32]

comptime TP = 4
comptime COLS = 4096 * 4
comptime SEQ_LEN = 256
comptime TOTAL_BYTES = SEQ_LEN * COLS * 2
comptime TOTAL_ELEMENTS = SEQ_LEN * COLS

# Per-rank completion state: padded to cache lines to avoid false sharing.
# Layout: rank r's state at base + r * 64
#   bytes [0..4): remaining workers counter (AtomicInt32, starts at num_workers)
#   bytes [8..12): done flag (AtomicInt32, 0 = not done, 1 = done)
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
# Fused kernel: reduce my slice, signal completion, pull from others
# =============================================================================


def fused_reduce_gather_kernel(
    config_addr: Int, start_element: Int, end_element: Int,
    my_rank: Int, n4: Int, n5: Int,
):
    """Each BurstPool worker runs this. Reduces its assigned slice,
    then participates in the allgather by pulling from completed ranks.

    Config layout (passed by address):
      [0..32): ptrs array (4 x Int = 4 x 8 bytes)
      [32..36): chunk (Int as 4 bytes... actually everything is Int = 8 bytes)

    Actually, let's just pass everything via the config struct address.
    """
    var cfg = UnsafePointer[FusedConfig, MutAnyOrigin](unsafe_from_address=config_addr)
    var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cfg[].ptrs)))
    var state_base = cfg[].state_base
    var chunk = cfg[].chunk
    var rem = cfg[].rem
    var tp = cfg[].tp
    var total_elements = cfg[].total_elements
    var sys = linux.linux_sys()

    var my_buf = ptrs[my_rank]
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=my_buf)
    comptime width = simd_width_of[DType.float32]()

    # --- Step 1: Reduce my slice from all 4 sources ---
    var src0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
    var src1 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[1])
    var src2 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[2])
    var src3 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[3])

    var i = start_element
    while i + width <= end_element:
        var acc = bf16_load_as[DType.float32, width](src0, i)
        acc += bf16_load_as[DType.float32, width](src1, i)
        acc += bf16_load_as[DType.float32, width](src2, i)
        acc += bf16_load_as[DType.float32, width](src3, i)
        (dst + i).store(acc.cast[DType.bfloat16]())
        i += width
    while i < end_element:
        var acc = Float32(src0[i]) + Float32(src1[i]) + Float32(src2[i]) + Float32(src3[i])
        dst[i] = Scalar[DType.bfloat16](acc)
        i += 1

    # --- Step 2: Signal my rank's reduce is (partially) done ---
    # Atomically decrement the counter. If I'm the last worker, set done flag.
    var old = AtomicInt32.fetch_add[ordering=Consistency.ACQUIRE_RELEASE](
        counter_ptr(state_base, my_rank), -1
    )
    if old == 1:
        # I'm the last worker — all slices of my rank's chunk are reduced.
        AtomicInt32.store[ordering=Consistency.RELEASE](
            done_ptr(state_base, my_rank), 1
        )

    # --- Step 3: Pull from other ranks as they complete ---
    # Each worker pulls a share of each remote chunk. We divide the copy
    # work among all workers on this node.
    var my_worker_id = start_element - my_rank * chunk  # offset within my rank's chunk
    var num_workers_approx = (chunk + ((end_element - start_element) - 1)) // (end_element - start_element)
    # Simpler: just use start/end to derive worker index
    var worker_slice = end_element - start_element
    var worker_idx = 0
    if worker_slice > 0:
        worker_idx = (start_element - my_rank * chunk) // worker_slice

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        # Poll until src_rank's reduction is complete
        while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr(state_base, src_rank)) == 0:
            sys.arch_cpu_relax()

        # Copy my share of src_rank's chunk into my buffer.
        # src_rank owns chunk [src_rank * chunk, src_rank * chunk + chunk_size).
        var src_chunk_start = src_rank * chunk
        var src_chunk_count = chunk + (rem if src_rank == tp - 1 else 0)

        # Divide this chunk among all workers on my node.
        # Use same row-partitioning as the reduce phase.
        var total_workers = num_workers_approx if num_workers_approx > 0 else 1
        var copy_per_worker = (src_chunk_count + total_workers - 1) // total_workers
        var copy_start = src_chunk_start + worker_idx * copy_per_worker
        var copy_end = min(copy_start + copy_per_worker, src_chunk_start + src_chunk_count)

        if copy_start < copy_end:
            var byte_start = copy_start * 2
            var byte_count = (copy_end - copy_start) * 2
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=my_buf + byte_start),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[src_rank] + byte_start),
                count=byte_count,
            )


struct FusedConfig:
    var ptrs: InlineArray[Int, 4]
    var state_base: Int
    var chunk: Int
    var rem: Int
    var tp: Int
    var total_elements: Int

    def __init__(out self):
        self.ptrs = InlineArray[Int, 4](fill=0)
        self.state_base = 0
        self.chunk = 0
        self.rem = 0
        self.tp = 0
        self.total_elements = 0


def fused_allreduce(
    bases: InlineArray[Int, TP],
    total_elements: Int,
    pool_ptrs: InlineArray[PoolPtr, TP],
):
    var chunk = total_elements // TP
    var rem = total_elements - chunk * TP

    # Per-rank completion state (cache-line padded)
    var state_mem = InlineArray[UInt8, TP * RANK_STATE_STRIDE](fill=0)
    var state_base = Int(UnsafePointer(to=state_mem))

    # Initialize counters to num_workers per rank
    for r in range(TP):
        var num_workers = pool_ptrs[r][].capacity
        AtomicInt32.store[ordering=Consistency.RELEASE](
            counter_ptr(state_base, r), Int32(num_workers)
        )
        AtomicInt32.store[ordering=Consistency.RELEASE](
            done_ptr(state_base, r), 0
        )

    # Config struct
    var cfg = FusedConfig()
    for r in range(TP):
        cfg.ptrs[r] = bases[r]
    cfg.state_base = state_base
    cfg.chunk = chunk
    cfg.rem = rem
    cfg.tp = TP
    cfg.total_elements = total_elements

    var config_addr = Int(UnsafePointer(to=cfg))

    # Dispatch: each rank's full pool, each worker gets a slice of the rank's chunk
    for r in range(TP):
        var rank_start = r * chunk
        var rank_count = chunk + (rem if r == TP - 1 else 0)
        var num_workers = pool_ptrs[r][].capacity
        var rows_per_worker = (rank_count + num_workers - 1) // num_workers

        for w in range(num_workers):
            var w_start = rank_start + w * rows_per_worker
            var w_end = min(w_start + rows_per_worker, rank_start + rank_count)
            if w_start >= rank_start + rank_count:
                # Fewer elements than workers — give this worker nothing
                # Still need to dispatch it so it decrements the counter
                w_start = rank_start + rank_count
                w_end = w_start
            var pack = pool_ptrs[r][].args_base + w
            pack[].arg0 = config_addr
            pack[].arg1 = w_start
            pack[].arg2 = w_end
            pack[].arg3 = r  # my_rank

        pool_ptrs[r][].dispatch(fused_reduce_gather_kernel, pool_ptrs[r][].args_base, num_workers)

    for r in range(TP):
        pool_ptrs[r][].join()


# =============================================================================
# Baselines
# =============================================================================


def naive_allreduce(bases: InlineArray[Int, TP], total_elements: Int):
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=bases[0])
    comptime width = simd_width_of[DType.float32]()
    for r in range(1, TP):
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=bases[r])
        var i = 0
        while i + width <= total_elements:
            var d = bf16_load_as[DType.float32, width](dst, i)
            var s = bf16_load_as[DType.float32, width](src, i)
            (dst + i).store((d + s).cast[DType.bfloat16]())
            i += width
        while i < total_elements:
            dst[i] = Scalar[DType.bfloat16](Float32(dst[i]) + Float32(src[i]))
            i += 1
    var total_bytes = total_elements * 2
    for r in range(1, TP):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[r]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=total_bytes,
        )


def scatter_gather_1w(
    bases: InlineArray[Int, TP],
    total_elements: Int,
    pool_ptrs_1w: InlineArray[PoolPtr, TP],
):
    var chunk = total_elements // TP
    var rem = total_elements - chunk * TP

    def reduce_rows_kernel(
        ptrs_addr: Int, dst_ptr: Int, start_element: Int,
        end_element: Int, tp: Int, n5: Int,
    ):
        var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=ptrs_addr)
        var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst_ptr)
        comptime w = simd_width_of[DType.float32]()
        var s0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])
        var s1 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[1])
        var s2 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[2])
        var s3 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[3])
        var i = start_element
        while i + w <= end_element:
            var acc = bf16_load_as[DType.float32, w](s0, i)
            acc += bf16_load_as[DType.float32, w](s1, i)
            acc += bf16_load_as[DType.float32, w](s2, i)
            acc += bf16_load_as[DType.float32, w](s3, i)
            (dst + i).store(acc.cast[DType.bfloat16]())
            i += w
        while i < end_element:
            dst[i] = Scalar[DType.bfloat16](
                Float32(s0[i]) + Float32(s1[i]) + Float32(s2[i]) + Float32(s3[i])
            )
            i += 1

    for r in range(TP):
        var start = r * chunk
        var count = chunk + (rem if r == TP - 1 else 0)
        var pack = pool_ptrs_1w[r][].args_base
        pack[].arg0 = Int(UnsafePointer(to=bases))
        pack[].arg1 = bases[r]
        pack[].arg2 = start
        pack[].arg3 = start + count
        pack[].arg4 = TP
        pool_ptrs_1w[r][].dispatch(reduce_rows_kernel, pool_ptrs_1w[r][].args_base, 1)
    for r in range(TP):
        pool_ptrs_1w[r][].join()

    def allgather_kernel(
        rank: Int, ptrs_addr: Int, chk: Int,
        rm: Int, tp: Int, n5: Int,
    ):
        var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=ptrs_addr)
        var my_buf = ptrs[rank]
        for r in range(tp):
            if r == rank:
                continue
            var start = r * chk
            var count = chk + (rm if r == tp - 1 else 0)
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=my_buf + start * 2),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[r] + start * 2),
                count=count * 2,
            )

    for r in range(TP):
        var pack = pool_ptrs_1w[r][].args_base
        pack[].arg0 = r
        pack[].arg1 = Int(UnsafePointer(to=bases))
        pack[].arg2 = chunk
        pack[].arg3 = rem
        pack[].arg4 = TP
        pool_ptrs_1w[r][].dispatch(allgather_kernel, pool_ptrs_1w[r][].args_base, 1)
    for r in range(TP):
        pool_ptrs_1w[r][].join()


# =============================================================================
# Helpers
# =============================================================================


def fill_value(ptr: Int, elements: Int, value: Float32):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    var v = Scalar[DType.bfloat16](value)
    for i in range(elements):
        p[i] = v


def read_f32(ptr: Int, idx: Int) -> Float32:
    return Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)[idx])


def buffers_match(a: Int, b: Int, elements: Int) -> Bool:
    var pa = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=a)
    var pb = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=b)
    for i in range(elements):
        if pa[i] != pb[i]:
            return False
    return True


def sort_times(mut times: List[Int]):
    for i in range(1, len(times)):
        var c = times[i]
        var j = i
        while j > 0 and times[j - 1] > c:
            times[j] = times[j - 1]
            j -= 1
        times[j] = c


def verify_allreduce(
    name: String,
    bases: InlineArray[Int, TP],
    expected_bf16: Float32,
) -> Bool:
    var ok = True
    for r in range(TP):
        var spots = InlineArray[Int, 3](fill=0)
        spots[0] = 0
        spots[1] = TOTAL_ELEMENTS // 2
        spots[2] = TOTAL_ELEMENTS - 1
        for ci in range(3):
            var v = read_f32(bases[r], spots[ci])
            if v != expected_bf16:
                print("FAIL", name, "rank", r, "idx", spots[ci],
                      "got", v, "expected", expected_bf16)
                ok = False
    if ok:
        for r in range(1, TP):
            if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
                print("FAIL", name, "rank", r, "differs from rank 0")
                ok = False
    print(name + ":", "PASS" if ok else "FAIL", "( sum =", expected_bf16, ")")
    return ok


def bench[
    run_fn: def(InlineArray[Int, TP], Int) capturing [_] -> None,
](
    name: String,
    bases: InlineArray[Int, TP],
    vals: InlineArray[Float32, TP],
    warmup: Int = 5,
    iters: Int = 50,
):
    for _ in range(warmup):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        run_fn(bases, TOTAL_ELEMENTS)

    var times = List[Int]()
    for _ in range(iters):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        var t0 = perf_counter_ns()
        run_fn(bases, TOTAL_ELEMENTS)
        var t1 = perf_counter_ns()
        times.append(Int(t1 - t0))

    sort_times(times)
    var med = times[iters // 2]
    var med_us = med // 1000
    var bw = Float64(TOTAL_BYTES * TP) / Float64(med)
    print("  ", name, ":", med_us, "us  (", TOTAL_BYTES, "B/rank,", bw, "GB/s )")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=== Allreduce: Fused Full-Node (TP=" + String(TP) + ", "
          + String(TOTAL_BYTES) + " B) ===")
    print()

    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)
    print("NUMA:", numa.num_nodes, "nodes")
    print("Ranks:", end=" ")
    for r in range(TP):
        print(topo[r], end=" ")
    print()
    if numa.num_nodes > 1:
        print("Distances:")
        for i in range(numa.num_nodes):
            for j in range(numa.num_nodes):
                print(numa.distance(i, j), end=" ")
            print()
    print()

    var arenas = HeapMoveArray[NumaArena[alignment=64]](TP)
    var bases = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        var arena = NumaArena[alignment=64](topo[rank], TOTAL_BYTES)
        if not arena:
            print("FATAL: arena alloc failed rank", rank)
            return
        bases[rank] = Int(arena.base)
        arenas.push(arena^)
    for rank in range(TP):
        _ = arenas[rank].prefault()

    print("Memory:")
    for rank in range(TP):
        var ok = arenas[rank].verify_placement()
        print("  rank", rank, "-> node", topo[rank], ":", "OK" if ok else "MISS")
    print()

    # 1-worker pools (for scatter_gather baseline)
    var pools_1w = HeapMoveArray[BurstPool[]](TP)
    for rank in range(TP):
        var pool = BurstPool[](1, numa.get_node_mask[128](topo[rank]), topo[rank])
        if not pool:
            print("FATAL: 1w pool failed rank", rank)
            return
        pools_1w.push(pool^)
    var pp_1w = InlineArray[PoolPtr, TP](fill=PoolPtr())
    for rank in range(TP):
        pp_1w[rank] = PoolPtr(unsafe_from_address=Int(UnsafePointer(to=pools_1w[rank])))

    # Full-node pools (for fused reduce+gather)
    var pools_full = HeapMoveArray[BurstPool[]](TP)
    for rank in range(TP):
        var pool = BurstPool[].for_numa_node(numa, topo[rank])
        if not pool:
            print("FATAL: full pool failed rank", rank)
            return
        pools_full.push(pool^)
    var pp_full = InlineArray[PoolPtr, TP](fill=PoolPtr())
    for rank in range(TP):
        pp_full[rank] = PoolPtr(unsafe_from_address=Int(UnsafePointer(to=pools_full[rank])))

    print("Pools:")
    for rank in range(TP):
        print("  rank", rank, ":", pools_full[rank].capacity, "cores on node", topo[rank])
    print()

    var vals = InlineArray[Float32, TP](fill=Float32(0))
    vals[0] = Float32(1.0)
    vals[1] = Float32(2.0)
    vals[2] = Float32(3.0)
    vals[3] = Float32(4.0)

    var expected = Float32(0)
    for r in range(TP):
        expected += Float32(Scalar[DType.bfloat16](vals[r]))
    var expected_bf16 = Float32(Scalar[DType.bfloat16](expected))

    # =================================================================
    # Correctness
    # =================================================================
    print("--- Correctness ---")

    for r in range(TP):
        fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
    naive_allreduce(bases, TOTAL_ELEMENTS)
    _ = verify_allreduce("naive", bases, expected_bf16)

    for r in range(TP):
        fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
    scatter_gather_1w(bases, TOTAL_ELEMENTS, pp_1w)
    _ = verify_allreduce("scatter_gather_1w", bases, expected_bf16)

    for r in range(TP):
        fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
    fused_allreduce(bases, TOTAL_ELEMENTS, pp_full)
    _ = verify_allreduce("fused", bases, expected_bf16)

    print()

    # =================================================================
    # Benchmark
    # =================================================================
    print("--- Benchmark: " + String(TOTAL_BYTES) + " B ---")

    @parameter
    def run_naive(b: InlineArray[Int, TP], n: Int):
        naive_allreduce(b, n)

    @parameter
    def run_1w(b: InlineArray[Int, TP], n: Int):
        scatter_gather_1w(b, n, pp_1w)

    @parameter
    def run_fused(b: InlineArray[Int, TP], n: Int):
        fused_allreduce(b, n, pp_full)

    bench[run_naive]("naive", bases, vals)
    bench[run_1w]("scatter_gather_1w", bases, vals)
    bench[run_fused]("fused", bases, vals)

    print()
    print("=== Done ===")

    _ = arenas
    _ = pools_1w
    _ = pools_full
