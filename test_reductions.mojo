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

comptime RANK_STATE_STRIDE = 64
comptime COUNTER_OFF = 0
comptime DONE_OFF = 8


# =============================================================================
# Atomic helpers
# =============================================================================


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
# Broadcast: parallel pull
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


def parallel_pull_broadcast(
    src_ptr: Int,
    bases: InlineArray[Int, TP],
    total_bytes: Int,
    pool_ptrs_1w: InlineArray[PoolPtr, TP],
):
    """All destination ranks pull from source simultaneously."""
    if src_ptr != bases[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )
    for r in range(1, TP):
        pool_ptrs_1w[r][].args_base[].arg0 = bases[r]
        pool_ptrs_1w[r][].args_base[].arg1 = bases[0]
        pool_ptrs_1w[r][].args_base[].arg2 = total_bytes
        pool_ptrs_1w[r][].dispatch(memcpy_kernel, pool_ptrs_1w[r][].args_base, 1)
    for r in range(1, TP):
        pool_ptrs_1w[r][].join()


# =============================================================================
# Allreduce: fused multi-core reduce + flag-signaled pull
# =============================================================================


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


def fused_reduce_gather_kernel(
    config_addr: Int, start_element: Int, end_element: Int,
    my_rank: Int, n4: Int, n5: Int,
):
    """Each worker: reduce slice → signal → pull from completed ranks."""
    var cfg = UnsafePointer[FusedConfig, MutAnyOrigin](unsafe_from_address=config_addr)
    var ptrs = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cfg[].ptrs)))
    var state_base = cfg[].state_base
    var chunk = cfg[].chunk
    var rem = cfg[].rem
    var tp = cfg[].tp
    var sys = linux.linux_sys()

    var my_buf = ptrs[my_rank]
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=my_buf)
    comptime width = simd_width_of[DType.float32]()

    # --- Reduce my slice from all 4 sources ---
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
    var num_workers_approx = (chunk + worker_slice - 1) // worker_slice if worker_slice > 0 else 1

    for src_rank in range(tp):
        if src_rank == my_rank:
            continue

        while AtomicInt32.load[ordering=Consistency.ACQUIRE](done_ptr(state_base, src_rank)) == 0:
            sys.arch_cpu_relax()

        var src_chunk_start = src_rank * chunk
        var src_chunk_count = chunk + (rem if src_rank == tp - 1 else 0)

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


def fused_allreduce(
    bases: InlineArray[Int, TP],
    total_elements: Int,
    pool_ptrs_full: InlineArray[PoolPtr, TP],
):
    var chunk = total_elements // TP
    var rem = total_elements - chunk * TP

    var state_mem = InlineArray[UInt8, TP * RANK_STATE_STRIDE](fill=0)
    var state_base = Int(UnsafePointer(to=state_mem))

    for r in range(TP):
        var num_workers = pool_ptrs_full[r][].capacity
        AtomicInt32.store[ordering=Consistency.RELEASE](
            counter_ptr(state_base, r), Int32(num_workers)
        )
        AtomicInt32.store[ordering=Consistency.RELEASE](
            done_ptr(state_base, r), 0
        )

    var cfg = FusedConfig()
    for r in range(TP):
        cfg.ptrs[r] = bases[r]
    cfg.state_base = state_base
    cfg.chunk = chunk
    cfg.rem = rem
    cfg.tp = TP
    cfg.total_elements = total_elements

    var config_addr = Int(UnsafePointer(to=cfg))

    for r in range(TP):
        var rank_start = r * chunk
        var rank_count = chunk + (rem if r == TP - 1 else 0)
        var num_workers = pool_ptrs_full[r][].capacity
        var rows_per_worker = (rank_count + num_workers - 1) // num_workers

        for w in range(num_workers):
            var w_start = rank_start + w * rows_per_worker
            var w_end = min(w_start + rows_per_worker, rank_start + rank_count)
            if w_start >= rank_start + rank_count:
                w_start = rank_start + rank_count
                w_end = w_start
            var pack = pool_ptrs_full[r][].args_base + w
            pack[].arg0 = config_addr
            pack[].arg1 = w_start
            pack[].arg2 = w_end
            pack[].arg3 = r

        pool_ptrs_full[r][].dispatch(fused_reduce_gather_kernel, pool_ptrs_full[r][].args_base, num_workers)

    for r in range(TP):
        pool_ptrs_full[r][].join()


# =============================================================================
# Naive baselines
# =============================================================================


def naive_broadcast(bases: InlineArray[Int, TP], total_bytes: Int):
    for r in range(1, TP):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[r]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=bases[0]),
            count=total_bytes,
        )


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


# =============================================================================
# Helpers
# =============================================================================


def fill_value(ptr: Int, elements: Int, value: Float32):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    var v = Scalar[DType.bfloat16](value)
    for i in range(elements):
        p[i] = v


def fill_pattern(ptr: Int, elements: Int, seed: Int):
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(elements):
        p[i] = UInt16((seed + i * 7 + 13) & 0xFFFF)


def read_f32(ptr: Int, idx: Int) -> Float32:
    return Float32(UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)[idx])


def buffers_match(a: Int, b: Int, elements: Int) -> Bool:
    var pa = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=a)
    var pb = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=b)
    for i in range(elements):
        if pa[i] != pb[i]:
            return False
    return True


def clear_buffer(ptr: Int, total_bytes: Int):
    var p = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_bytes):
        p[i] = 0


def sort_times(mut times: List[Int]):
    for i in range(1, len(times)):
        var c = times[i]
        var j = i
        while j > 0 and times[j - 1] > c:
            times[j] = times[j - 1]
            j -= 1
        times[j] = c


# =============================================================================
# Main
# =============================================================================


def main():
    print("=== NUMA Reductions (TP=" + String(TP) + ", " + String(TOTAL_BYTES) + " B) ===")
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

    # Arenas
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

    # 1-worker pools (broadcast)
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

    # Full-node pools (allreduce)
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

    # =================================================================
    # Correctness: Broadcast
    # =================================================================
    print("--- Correctness: Broadcast ---")

    # Naive
    fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
    for r in range(1, TP):
        clear_buffer(bases[r], TOTAL_BYTES)
    naive_broadcast(bases, TOTAL_BYTES)
    var naive_bc_ok = True
    for r in range(1, TP):
        if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
            print("FAIL naive_broadcast rank", r)
            naive_bc_ok = False
    print("naive_broadcast:", "PASS" if naive_bc_ok else "FAIL")

    # Parallel pull
    fill_pattern(bases[0], TOTAL_ELEMENTS, 77)
    for r in range(1, TP):
        clear_buffer(bases[r], TOTAL_BYTES)
    parallel_pull_broadcast(bases[0], bases, TOTAL_BYTES, pp_1w)
    var pull_bc_ok = True
    for r in range(1, TP):
        if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
            print("FAIL parallel_pull_broadcast rank", r)
            pull_bc_ok = False
    print("parallel_pull_broadcast:", "PASS" if pull_bc_ok else "FAIL")

    print()

    # =================================================================
    # Correctness: Allreduce
    # =================================================================
    print("--- Correctness: Allreduce ---")

    var vals = InlineArray[Float32, TP](fill=Float32(0))
    vals[0] = Float32(1.0)
    vals[1] = Float32(2.0)
    vals[2] = Float32(3.0)
    vals[3] = Float32(4.0)
    var expected = Float32(0)
    for r in range(TP):
        expected += Float32(Scalar[DType.bfloat16](vals[r]))
    var expected_bf16 = Float32(Scalar[DType.bfloat16](expected))

    # Naive
    for r in range(TP):
        fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
    naive_allreduce(bases, TOTAL_ELEMENTS)
    var naive_ar_ok = True
    for r in range(TP):
        var spots = InlineArray[Int, 3](fill=0)
        spots[0] = 0
        spots[1] = TOTAL_ELEMENTS // 2
        spots[2] = TOTAL_ELEMENTS - 1
        for ci in range(3):
            var v = read_f32(bases[r], spots[ci])
            if v != expected_bf16:
                print("FAIL naive_allreduce rank", r, "idx", spots[ci],
                      "got", v, "expected", expected_bf16)
                naive_ar_ok = False
    if naive_ar_ok:
        for r in range(1, TP):
            if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
                print("FAIL naive_allreduce rank", r, "differs from rank 0")
                naive_ar_ok = False
    print("naive_allreduce:", "PASS" if naive_ar_ok else "FAIL",
          "( sum =", expected_bf16, ")")

    # Fused
    for r in range(TP):
        fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
    fused_allreduce(bases, TOTAL_ELEMENTS, pp_full)
    var fused_ar_ok = True
    for r in range(TP):
        var spots = InlineArray[Int, 3](fill=0)
        spots[0] = 0
        spots[1] = TOTAL_ELEMENTS // 2
        spots[2] = TOTAL_ELEMENTS - 1
        for ci in range(3):
            var v = read_f32(bases[r], spots[ci])
            if v != expected_bf16:
                print("FAIL fused_allreduce rank", r, "idx", spots[ci],
                      "got", v, "expected", expected_bf16)
                fused_ar_ok = False
    if fused_ar_ok:
        for r in range(1, TP):
            if not buffers_match(bases[0], bases[r], TOTAL_ELEMENTS):
                print("FAIL fused_allreduce rank", r, "differs from rank 0")
                fused_ar_ok = False
    print("fused_allreduce:", "PASS" if fused_ar_ok else "FAIL",
          "( sum =", expected_bf16, ")")

    print()

    # =================================================================
    # Benchmark: Broadcast
    # =================================================================
    print("--- Benchmark: Broadcast (" + String(TOTAL_BYTES) + " B) ---")

    # Warmup
    for _ in range(5):
        fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
        naive_broadcast(bases, TOTAL_BYTES)
    for _ in range(5):
        fill_pattern(bases[0], TOTAL_ELEMENTS, 42)
        parallel_pull_broadcast(bases[0], bases, TOTAL_BYTES, pp_1w)

    # Naive broadcast
    var naive_bc_times = List[Int]()
    for _ in range(50):
        var t0 = perf_counter_ns()
        naive_broadcast(bases, TOTAL_BYTES)
        naive_bc_times.append(Int(perf_counter_ns() - t0))
    sort_times(naive_bc_times)
    var nbc_med = naive_bc_times[25]
    print("  naive         :", nbc_med // 1000, "us,",
          Float64(TOTAL_BYTES * (TP - 1)) / Float64(nbc_med), "GB/s aggregate")

    # Parallel pull broadcast
    var pull_bc_times = List[Int]()
    for _ in range(50):
        var t0 = perf_counter_ns()
        parallel_pull_broadcast(bases[0], bases, TOTAL_BYTES, pp_1w)
        pull_bc_times.append(Int(perf_counter_ns() - t0))
    sort_times(pull_bc_times)
    var pbc_med = pull_bc_times[25]
    print("  parallel_pull :", pbc_med // 1000, "us,",
          Float64(TOTAL_BYTES * (TP - 1)) / Float64(pbc_med), "GB/s aggregate")

    print()

    # =================================================================
    # Benchmark: Allreduce
    # =================================================================
    print("--- Benchmark: Allreduce (" + String(TOTAL_BYTES) + " B) ---")

    # Warmup
    for _ in range(5):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        naive_allreduce(bases, TOTAL_ELEMENTS)
    for _ in range(5):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        fused_allreduce(bases, TOTAL_ELEMENTS, pp_full)

    # Naive allreduce
    var naive_ar_times = List[Int]()
    for _ in range(50):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        var t0 = perf_counter_ns()
        naive_allreduce(bases, TOTAL_ELEMENTS)
        naive_ar_times.append(Int(perf_counter_ns() - t0))
    sort_times(naive_ar_times)
    var nar_med = naive_ar_times[25]
    print("  naive         :", nar_med // 1000, "us,",
          Float64(TOTAL_BYTES * TP) / Float64(nar_med), "GB/s")

    # Fused allreduce
    var fused_ar_times = List[Int]()
    for _ in range(50):
        for r in range(TP):
            fill_value(bases[r], TOTAL_ELEMENTS, vals[r])
        var t0 = perf_counter_ns()
        fused_allreduce(bases, TOTAL_ELEMENTS, pp_full)
        fused_ar_times.append(Int(perf_counter_ns() - t0))
    sort_times(fused_ar_times)
    var far_med = fused_ar_times[25]
    print("  fused         :", far_med // 1000, "us,",
          Float64(TOTAL_BYTES * TP) / Float64(far_med), "GB/s")

    print()
    print("=== Done ===")

    _ = arenas
    _ = pools_1w
    _ = pools_full
