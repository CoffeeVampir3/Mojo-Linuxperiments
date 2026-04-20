"""Validate small_allreduce and ring_allreduce with residual_add parameter.

Tests both the original (residual_add=False) and fused (residual_add=True)
paths on real NUMA topology with BurstPools. Verifies correctness by
comparing against scalar reference computation.

Run: fish remote_build.fish test_allreduce_residual.mojo
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from numa import NumaInfo
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import BF16, Shape, Mat

from kernels.reductions import small_allreduce, ring_allreduce


comptime COLS = 3072
comptime X_SLOT = Mat[BF16, 1, COLS]


def fill_bf16(base: Int, count: Int, value: Float32):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=base)
    for i in range(count):
        p[i] = Scalar[DType.bfloat16](value)


def fill_bf16_pattern(base: Int, count: Int, rank: Int):
    """Fill with rank-dependent pattern: value = rank * 0.1 + element * 0.001."""
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=base)
    for i in range(count):
        p[i] = Scalar[DType.bfloat16](Float32(rank) * 0.1 + Float32(i) * 0.001)


def read_bf16(base: Int, idx: Int) -> Float32:
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=base)
    return Float32(p[idx])


def check_close(actual: Float32, expected: Float32, tol: Float32, label: String, idx: Int) -> Bool:
    var diff = (actual - expected).__abs__()
    if diff > tol:
        print("FAIL", label, "idx=", idx, "actual=", actual, "expected=", expected, "diff=", diff)
        return False
    return True


def test_small_allreduce_original[tp: Int](
    arenas: InlineArray[UnsafePointer[NumaArena[], MutAnyOrigin], tp],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
) -> Bool:
    """Test small_allreduce with residual_add=False (original behavior)."""
    var ptrs = InlineArray[Int, tp](fill=0)
    for r in range(tp):
        var buf = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        ptrs[r] = Int(buf)
        fill_bf16_pattern(ptrs[r], COLS, r)

    small_allreduce[X_SLOT, tp](ptrs, 1, pool_ptrs)

    var ok = True
    for i in range(COLS):
        var expected = Float32(0)
        for r in range(tp):
            expected += Float32(r) * 0.1 + Float32(i) * 0.001
        if not check_close(read_bf16(ptrs[0], i), expected, 0.05, "small_orig", i):
            ok = False
            if i > 5:
                break

    for r in range(1, tp):
        if read_bf16(ptrs[r], 0) != read_bf16(ptrs[0], 0):
            print("FAIL small_orig: rank", r, "not broadcast")
            ok = False

    return ok


def test_small_allreduce_fused[tp: Int](
    arenas: InlineArray[UnsafePointer[NumaArena[], MutAnyOrigin], tp],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
) -> Bool:
    """Test small_allreduce with residual_add=True (fused)."""
    var src_ptrs = InlineArray[Int, tp](fill=0)
    var dst_ptrs = InlineArray[Int, tp](fill=0)

    for r in range(tp):
        var src = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        var dst = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        src_ptrs[r] = Int(src)
        dst_ptrs[r] = Int(dst)
        fill_bf16_pattern(src_ptrs[r], COLS, r)
        fill_bf16(dst_ptrs[r], COLS, 1.0)

    small_allreduce[X_SLOT, tp, residual_add=True](
        src_ptrs, 1, pool_ptrs, dst_ptrs)

    var ok = True
    for i in range(COLS):
        var reduced = Float32(0)
        for r in range(tp):
            reduced += Float32(r) * 0.1 + Float32(i) * 0.001
        var expected = 1.0 + reduced
        if not check_close(read_bf16(dst_ptrs[0], i), expected, 0.05, "small_fused", i):
            ok = False
            if i > 5:
                break

    for r in range(1, tp):
        if read_bf16(dst_ptrs[r], 0) != read_bf16(dst_ptrs[0], 0):
            print("FAIL small_fused: rank", r, "not broadcast")
            ok = False

    return ok


def test_ring_allreduce_original[tp: Int](
    arenas: InlineArray[UnsafePointer[NumaArena[], MutAnyOrigin], tp],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
) -> Bool:
    """Test ring_allreduce with residual_add=False (original behavior)."""
    var ptrs = InlineArray[Int, tp](fill=0)
    for r in range(tp):
        var buf = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        ptrs[r] = Int(buf)
        fill_bf16_pattern(ptrs[r], COLS, r)

    ring_allreduce[X_SLOT, tp](ptrs, 1, pool_ptrs)

    var ok = True
    for i in range(COLS):
        var expected = Float32(0)
        for r in range(tp):
            expected += Float32(r) * 0.1 + Float32(i) * 0.001
        if not check_close(read_bf16(ptrs[0], i), expected, 0.05, "ring_orig", i):
            ok = False
            if i > 5:
                break

    for r in range(1, tp):
        var mismatch = 0
        for i in range(COLS):
            if (read_bf16(ptrs[r], i) - read_bf16(ptrs[0], i)).__abs__() > 0.01:
                mismatch += 1
        if mismatch > 0:
            print("FAIL ring_orig: rank", r, "has", mismatch, "mismatches vs rank 0")
            ok = False

    return ok


def test_ring_allreduce_fused[tp: Int](
    arenas: InlineArray[UnsafePointer[NumaArena[], MutAnyOrigin], tp],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp],
) -> Bool:
    """Test ring_allreduce with residual_add=True (fused)."""
    var src_ptrs = InlineArray[Int, tp](fill=0)
    var dst_ptrs = InlineArray[Int, tp](fill=0)

    for r in range(tp):
        var src = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        var dst = arenas[r][].alloc[Scalar[DType.bfloat16]](COLS)
        src_ptrs[r] = Int(src)
        dst_ptrs[r] = Int(dst)
        fill_bf16_pattern(src_ptrs[r], COLS, r)
        fill_bf16(dst_ptrs[r], COLS, 2.0)

    ring_allreduce[X_SLOT, tp, residual_add=True](
        src_ptrs, 1, pool_ptrs, dst_ptrs)

    var ok = True
    for i in range(COLS):
        var reduced = Float32(0)
        for r in range(tp):
            reduced += Float32(r) * 0.1 + Float32(i) * 0.001
        var expected = 2.0 + reduced
        if not check_close(read_bf16(dst_ptrs[0], i), expected, 0.05, "ring_fused", i):
            ok = False
            if i > 5:
                break

    for r in range(1, tp):
        var mismatch = 0
        for i in range(COLS):
            if (read_bf16(dst_ptrs[r], i) - read_bf16(dst_ptrs[0], i)).__abs__() > 0.01:
                mismatch += 1
        if mismatch > 0:
            print("FAIL ring_fused: rank", r, "has", mismatch, "mismatches vs rank 0")
            ok = False

    return ok


def main():
    print("=== Allreduce + residual_add validation ===")

    var numa = NumaInfo()
    var num_nodes = numa.num_nodes
    print("NUMA nodes:", num_nodes)

    if num_nodes < 2:
        print("Need >= 2 NUMA nodes for TP allreduce test, skipping multi-node tests")

    comptime tp = 4
    var arena_size = COLS * 4 * size_of[Scalar[DType.bfloat16]]()

    if num_nodes < tp:
        print("Need", tp, "NUMA nodes, have", num_nodes, "— skipping")
        return

    var arenas = HeapMoveArray[NumaArena[]](tp)
    var pools = HeapMoveArray[BurstPool[]](tp)
    for r in range(tp):
        arenas.push(NumaArena[](r, arena_size))
        if not arenas[r]:
            print("Failed to allocate arena on node", r)
            return
        _ = arenas[r].prefault()
        pools.push(BurstPool[].for_numa_node(numa, r, headroom=2))

    var arena_ptrs = InlineArray[UnsafePointer[NumaArena[], MutAnyOrigin], tp](
        fill=UnsafePointer[NumaArena[], MutAnyOrigin]())
    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for r in range(tp):
        arena_ptrs[r] = UnsafePointer(to=arenas[r])
        pool_ptrs[r] = UnsafePointer(to=pools[r])

    print()
    print("--- small_allreduce (original) ---")
    var ok1 = test_small_allreduce_original[tp](arena_ptrs, pool_ptrs)
    print("PASS" if ok1 else "FAIL")

    print()
    print("--- small_allreduce (fused residual_add) ---")
    for r in range(tp):
        arenas[r].reset()
    var ok2 = test_small_allreduce_fused[tp](arena_ptrs, pool_ptrs)
    print("PASS" if ok2 else "FAIL")

    print()
    print("--- ring_allreduce (original) ---")
    for r in range(tp):
        arenas[r].reset()
    var ok3 = test_ring_allreduce_original[tp](arena_ptrs, pool_ptrs)
    print("PASS" if ok3 else "FAIL")

    print()
    print("--- ring_allreduce (fused residual_add) ---")
    for r in range(tp):
        arenas[r].reset()
    var ok4 = test_ring_allreduce_fused[tp](arena_ptrs, pool_ptrs)
    print("PASS" if ok4 else "FAIL")

    print()
    if ok1 and ok2 and ok3 and ok4:
        print("=== ALL PASS ===")
    else:
        print("=== FAILURES ===")
