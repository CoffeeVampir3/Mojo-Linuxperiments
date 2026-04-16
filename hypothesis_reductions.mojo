"""Hypothesis: small_allreduce or ring_allreduce introduces non-determinism at TP>1.

Test: write known bf16 patterns to per-rank NUMA-local buffers, run the reduction
many times, and check whether the output is bit-identical across iterations.

If the reduction is deterministic, every iteration produces the same bits.
If it drifts, we've found the source.
"""

from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from kernels.reductions import small_allreduce, ring_allreduce
from modeling.model_spec import BF16, Mat


comptime TP = 4
comptime COLS = 2816
comptime ELEMENTS = COLS
comptime BYTES = ELEMENTS * 2
comptime ITERATIONS = 200
comptime ALIGNMENT = 64

# ring_allreduce test uses a larger tensor to exercise chunked reduce+gather
comptime BIG_ROWS = 4
comptime BIG_ELEMENTS = COLS * BIG_ROWS
comptime BIG_BYTES = BIG_ELEMENTS * 2

# Per-rank arena: big enough for the largest test buffer
comptime RANK_ARENA_BYTES = BIG_BYTES + 4096

# Main-thread scratch arena on node 0: 4 snapshot slots for comparison
comptime SNAP_ARENA_BYTES = BIG_BYTES * 4


def bf16ptr(addr: Int) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=addr)


def byteptr(addr: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)


def fill_rank_pattern(base: Int, rank: Int, elements: Int):
    var ptr = bf16ptr(base)
    for i in range(elements):
        var val = Float32(rank * 1000 + i) * 0.01
        ptr[i] = Scalar[DType.bfloat16](val)


def snap(src: Int, dst: Int, nbytes: Int):
    memcpy(dest=byteptr(dst), src=byteptr(src), count=nbytes)


def diff_bytes(a: Int, b: Int, nbytes: Int) -> Int:
    var pa = byteptr(a)
    var pb = byteptr(b)
    var diffs = 0
    for i in range(nbytes):
        if pa[i] != pb[i]:
            diffs += 1
    return diffs


def make_pool_ptrs(
    mut pools: HeapMoveArray[BurstPool[]],
) -> InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP]:
    var pp = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for r in range(TP):
        pp[r] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[r])))
    return pp^


def make_buf_ptrs(
    arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
) -> InlineArray[Int, TP]:
    var bp = InlineArray[Int, TP](fill=0)
    for r in range(TP):
        bp[r] = Int(arenas[r].base)
    return bp^


# ============================================================================
# Test 1: small_allreduce determinism
# ============================================================================


def test_small_allreduce(
    arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut pools: HeapMoveArray[BurstPool[]],
    scratch: NumaArena[alignment=ALIGNMENT],
):
    print("--- test: small_allreduce determinism ---")

    var buf_ptrs = make_buf_ptrs(arenas)
    var pool_ptrs = make_pool_ptrs(pools)

    # Scratch layout on node 0: [ref_snap | cur_snap] each BYTES
    var ref_snap = Int(scratch.base)
    var cur_snap = ref_snap + BYTES
    var max_diffs = 0
    var drift_count = 0

    for iteration in range(ITERATIONS):
        for r in range(TP):
            fill_rank_pattern(buf_ptrs[r], r, ELEMENTS)

        small_allreduce[Mat[BF16, 1, COLS], TP](buf_ptrs, 1, pool_ptrs)

        if iteration == 0:
            snap(buf_ptrs[0], ref_snap, BYTES)
        else:
            snap(buf_ptrs[0], cur_snap, BYTES)
            var diffs = diff_bytes(ref_snap, cur_snap, BYTES)
            if diffs > 0:
                drift_count += 1
                if diffs > max_diffs:
                    max_diffs = diffs

    if drift_count == 0:
        print("  PASS: " + String(ITERATIONS) + " iterations, all bit-identical")
    else:
        print("  FAIL: " + String(drift_count) + "/" + String(ITERATIONS - 1)
            + " iterations drifted, max " + String(max_diffs) + " differing bytes")

    # Cross-rank consistency: all ranks should produce identical output
    for r in range(TP):
        fill_rank_pattern(buf_ptrs[r], r, ELEMENTS)
    small_allreduce[Mat[BF16, 1, COLS], TP](buf_ptrs, 1, pool_ptrs)

    var cross_rank_ok = True
    for r in range(1, TP):
        var diffs = diff_bytes(buf_ptrs[0], buf_ptrs[r], BYTES)
        if diffs > 0:
            print("  FAIL: rank 0 vs rank " + String(r) + ": " + String(diffs) + " differing bytes")
            cross_rank_ok = False
    if cross_rank_ok:
        print("  PASS: all ranks produce identical results")


# ============================================================================
# Test 2: ring_allreduce determinism
# ============================================================================


def test_ring_allreduce(
    arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut pools: HeapMoveArray[BurstPool[]],
    scratch: NumaArena[alignment=ALIGNMENT],
):
    print("--- test: ring_allreduce determinism ---")

    var buf_ptrs = make_buf_ptrs(arenas)
    var pool_ptrs = make_pool_ptrs(pools)

    var ref_snap = Int(scratch.base)
    var cur_snap = ref_snap + BIG_BYTES
    var max_diffs = 0
    var drift_count = 0

    for iteration in range(ITERATIONS):
        for r in range(TP):
            fill_rank_pattern(buf_ptrs[r], r, BIG_ELEMENTS)

        ring_allreduce[Mat[BF16, BIG_ROWS, COLS], TP](buf_ptrs, BIG_ROWS, pool_ptrs)

        if iteration == 0:
            snap(buf_ptrs[0], ref_snap, BIG_BYTES)
        else:
            snap(buf_ptrs[0], cur_snap, BIG_BYTES)
            var diffs = diff_bytes(ref_snap, cur_snap, BIG_BYTES)
            if diffs > 0:
                drift_count += 1
                if diffs > max_diffs:
                    max_diffs = diffs

    if drift_count == 0:
        print("  PASS: " + String(ITERATIONS) + " iterations, all bit-identical")
    else:
        print("  FAIL: " + String(drift_count) + "/" + String(ITERATIONS - 1)
            + " iterations drifted, max " + String(max_diffs) + " differing bytes")

    # Cross-rank consistency
    for r in range(TP):
        fill_rank_pattern(buf_ptrs[r], r, BIG_ELEMENTS)
    ring_allreduce[Mat[BF16, BIG_ROWS, COLS], TP](buf_ptrs, BIG_ROWS, pool_ptrs)

    var cross_rank_ok = True
    for r in range(1, TP):
        var diffs = diff_bytes(buf_ptrs[0], buf_ptrs[r], BIG_BYTES)
        if diffs > 0:
            print("  FAIL: rank 0 vs rank " + String(r) + ": " + String(diffs) + " differing bytes")
            cross_rank_ok = False
    if cross_rank_ok:
        print("  PASS: all ranks produce identical results")


# ============================================================================
# Test 3: cross-rank visibility after dispatch+join
# ============================================================================


@fieldwise_init
struct FillArgs(Copyable, ImplicitlyCopyable):
    var buf: Int
    var pattern: Scalar[DType.bfloat16]
    var count: Int


def fill_kernel(args: FillArgs):
    var ptr = bf16ptr(args.buf)
    for i in range(args.count):
        ptr[i] = args.pattern


@fieldwise_init
struct CheckArgs(Copyable, ImplicitlyCopyable):
    var buf: Int
    var expected: Scalar[DType.bfloat16]
    var count: Int
    var result_addr: Int


def check_kernel(args: CheckArgs):
    var ptr = bf16ptr(args.buf)
    var mismatches = 0
    for i in range(args.count):
        if ptr[i] != args.expected:
            mismatches += 1
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=args.result_addr)[] = mismatches


def test_dispatch_join_fence(
    arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut pools: HeapMoveArray[BurstPool[]],
    scratch: NumaArena[alignment=ALIGNMENT],
):
    """Rank B's workers write a pattern. After join, rank A's workers read
    rank B's buffer. Repeat many times, check for stale reads."""
    print("--- test: cross-rank visibility after dispatch+join ---")

    var pool_ptrs = make_pool_ptrs(pools)

    # Each rank uses its own arena: first BYTES for the write buffer
    var write_bufs = InlineArray[Int, TP](fill=0)
    for r in range(TP):
        write_bufs[r] = Int(arenas[r].base)

    # Result slot lives on scratch arena (main thread reads it)
    var result_addr = Int(scratch.base)
    var stale_count = 0

    for iteration in range(ITERATIONS):
        var pattern = Scalar[DType.bfloat16](Float32(iteration + 1) * 0.5)

        # Phase 1: each rank's pool writes a pattern to its NUMA-local buffer
        for r in range(TP):
            var args = FillArgs(write_bufs[r], pattern, ELEMENTS)
            pool_ptrs[r][].dispatch[FillArgs, fill_kernel](
                UnsafePointer(to=args), 1)
        for r in range(TP):
            pool_ptrs[r][].join()

        # Phase 2: each rank's pool reads from every OTHER rank's buffer
        for reader in range(TP):
            for writer in range(TP):
                if writer == reader:
                    continue
                var args = CheckArgs(write_bufs[writer], pattern, ELEMENTS, result_addr)
                pool_ptrs[reader][].dispatch[CheckArgs, check_kernel](
                    UnsafePointer(to=args), 1)
                pool_ptrs[reader][].join()

                var mismatches = UnsafePointer[Int, MutAnyOrigin](
                    unsafe_from_address=result_addr)[]
                if mismatches > 0:
                    stale_count += 1
                    if stale_count <= 5:
                        print("  stale read: iter=" + String(iteration)
                            + " reader=rank" + String(reader)
                            + " writer=rank" + String(writer)
                            + " mismatches=" + String(mismatches))

    if stale_count == 0:
        print("  PASS: " + String(ITERATIONS) + " iterations, no stale cross-rank reads")
    else:
        print("  FAIL: " + String(stale_count) + " stale read events")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=== Reduction Determinism Tests (TP=" + String(TP) + ") ===")
    print("")

    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)
    print("NUMA topology: " + String(TP) + " nodes selected")
    for r in range(TP):
        print("  rank " + String(r) + " -> node " + String(topo[r]))

    # Per-rank NUMA-local arenas — one per NUMA node, just like the model
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](TP)
    for r in range(TP):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[r], RANK_ARENA_BYTES))
        if not arenas[r]:
            print("arena alloc failed for rank " + String(r))
            return

    # Per-rank burst pools — workers pinned to each NUMA node
    var pools = HeapMoveArray[BurstPool[]](TP)
    for r in range(TP):
        pools.push(BurstPool[].for_numa_node(numa, topo[r], headroom=2))
        print("  rank " + String(r) + " pool capacity: " + String(pools[r].capacity))

    # Main-thread scratch on node 0 for snapshot comparison
    var scratch = NumaArena[alignment=ALIGNMENT](topo[0], SNAP_ARENA_BYTES)

    print("")
    test_small_allreduce(arenas, pools, scratch)
    print("")
    test_ring_allreduce(arenas, pools, scratch)
    print("")
    test_dispatch_join_fence(arenas, pools, scratch)
    print("")
    print("=== done ===")
