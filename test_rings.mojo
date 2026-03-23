"""Benchmark and correctness tests for ring communication primitives.

Tests broadcast and allreduce across NUMA boundaries with three implementations:
1. Naive (star memcpy / scalar sum)
2. Ring topology (sequential hops / ring reduce-scatter + allgather)
3. Ring + SIMD (same ring topology, SIMD-accelerated accumulation)

Validates correctness of all three, then benchmarks at realistic sizes.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from std.collections import InlineArray

from experimental4.model_spec import (
    Encoding, Shaped, BF16, Slot, Replicated, byte_count,
)


# =============================================================================
# Slot types for testing
# =============================================================================

comptime COLS_SMALL = 8
comptime COLS_MEDIUM = 576     # SmolLM2 hidden dim
comptime COLS_LARGE = 4096

comptime MAX_SEQ = 1024
comptime TP = 3

comptime SlotSmall = Slot[BF16, Replicated, MAX_SEQ, COLS_SMALL, TP]
comptime SlotMedium = Slot[BF16, Replicated, MAX_SEQ, COLS_MEDIUM, TP]
comptime SlotLarge = Slot[BF16, Replicated, MAX_SEQ, COLS_LARGE, TP]


# =============================================================================
# Broadcast implementations
# =============================================================================

def naive_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int, dst_ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Star broadcast: host copies directly to every other rank."""
    var total_bytes = seq_len * T.COLS * T.ELEMENT_BYTES
    if total_bytes <= 0 or tp <= 1:
        return
    for rank in range(1, tp):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[rank]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )


def ring_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int, dst_ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Ring broadcast: forward along ring, one hop per step."""
    var total_bytes = seq_len * T.COLS * T.ELEMENT_BYTES
    if total_bytes <= 0 or tp <= 1:
        return
    if src_ptr != dst_ptrs[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )
    for step in range(tp - 1):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[step + 1]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[step]),
            count=total_bytes,
        )


# =============================================================================
# Allreduce implementations
# =============================================================================

def naive_allreduce[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Naive allreduce: scalar sum all ranks into rank 0, then broadcast."""
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[0])

    # Accumulate all other ranks into rank 0 (scalar).
    for rank in range(1, tp):
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[rank])
        for i in range(total_elements):
            var s = Float32(src[i])
            var d = Float32(dst[i])
            dst[i] = Scalar[DType.bfloat16](s + d)

    # Broadcast rank 0 to all others.
    var total_bytes = total_elements * T.ELEMENT_BYTES
    for rank in range(1, tp):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[rank]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[0]),
            count=total_bytes,
        )


def ring_allreduce_scalar[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Ring allreduce with scalar accumulation."""
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var chunk_size = total_elements // tp
    var remainder = total_elements - chunk_size * tp

    @always_inline
    def chunk_start(c: Int) -> Int:
        return c * chunk_size

    @always_inline
    def chunk_len(c: Int) -> Int:
        if c == tp - 1:
            return chunk_size + remainder
        return chunk_size

    # Reduce-scatter.
    for step in range(tp - 1):
        for rank in range(tp):
            var c = (rank - step + tp) % tp
            var next_rank = (rank + 1) % tp
            var cs = chunk_start(c)
            var cl = chunk_len(c)
            var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[rank])
            var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptrs[next_rank])
            for i in range(cl):
                var s = Float32(src[cs + i])
                var d = Float32(dst[cs + i])
                dst[cs + i] = Scalar[DType.bfloat16](s + d)

    # Allgather.
    for step in range(tp - 1):
        for rank in range(tp):
            var c = (rank - step + 1 + tp) % tp
            var next_rank = (rank + 1) % tp
            var cs = chunk_start(c)
            var cl = chunk_len(c)
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[next_rank] + cs * T.ELEMENT_BYTES),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[rank] + cs * T.ELEMENT_BYTES),
                count=cl * T.ELEMENT_BYTES,
            )


def ring_allreduce_simd[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Ring allreduce with SIMD-accelerated accumulation."""
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    var chunk_size = total_elements // tp
    var remainder = total_elements - chunk_size * tp
    comptime width = simd_width_of[DType.float32]()

    @always_inline
    def chunk_start(c: Int) -> Int:
        return c * chunk_size

    @always_inline
    def chunk_len(c: Int) -> Int:
        if c == tp - 1:
            return chunk_size + remainder
        return chunk_size

    @always_inline
    def accumulate(dst_ptr: Int, src_ptr: Int, start: Int, length: Int):
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=src_ptr)
        var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst_ptr)
        var i = 0
        while i + width <= length:
            var s_raw = (src + start + i).bitcast[Scalar[DType.uint16]]().load[width=width]()
            var d_raw = (dst + start + i).bitcast[Scalar[DType.uint16]]().load[width=width]()
            var s_f32 = SIMD[DType.float32, width](from_bits=s_raw.cast[DType.uint32]() << 16)
            var d_f32 = SIMD[DType.float32, width](from_bits=d_raw.cast[DType.uint32]() << 16)
            (dst + start + i).store((s_f32 + d_f32).cast[DType.bfloat16]())
            i += width
        while i < length:
            var s = Float32(src[start + i])
            var d = Float32(dst[start + i])
            dst[start + i] = Scalar[DType.bfloat16](s + d)
            i += 1

    # Reduce-scatter.
    for step in range(tp - 1):
        for rank in range(tp):
            var c = (rank - step + tp) % tp
            var next_rank = (rank + 1) % tp
            accumulate(ptrs[next_rank], ptrs[rank], chunk_start(c), chunk_len(c))

    # Allgather.
    for step in range(tp - 1):
        for rank in range(tp):
            var c = (rank - step + 1 + tp) % tp
            var next_rank = (rank + 1) % tp
            var cs = chunk_start(c)
            var cl = chunk_len(c)
            memcpy(
                dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[next_rank] + cs * T.ELEMENT_BYTES),
                src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[rank] + cs * T.ELEMENT_BYTES),
                count=cl * T.ELEMENT_BYTES,
            )


# =============================================================================
# Helpers
# =============================================================================

def fill_bf16_value(ptr: Int, total_elements: Int, value: Float32):
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    var bf_val = Scalar[DType.bfloat16](value)
    for i in range(total_elements):
        p[i] = bf_val


def fill_pattern(ptr: Int, total_elements: Int, seed: UInt16):
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_elements):
        p[i] = UInt16((Int(seed) + i * 7 + 13) & 0xFFFF)


def read_bf16_as_f32(ptr: Int, idx: Int) -> Float32:
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    return Float32(p[idx])


def verify_match(ptr_a: Int, ptr_b: Int, total_elements: Int) -> Tuple[Bool, Int]:
    var a = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr_a)
    var b = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr_b)
    for i in range(total_elements):
        if a[i] != b[i]:
            return (False, i)
    return (True, -1)


def clear_buffer(ptr: Int, total_bytes: Int):
    var p = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_bytes):
        p[i] = 0


# =============================================================================
# Correctness validation
# =============================================================================

def validate_allreduce[
    T: Encoding & Shaped,
    allreduce_fn: def(InlineArray[Int, TP], Int) capturing [_] -> None,
](
    name: String,
    seq_len: Int,
    arenas: List[Int],
    rank_values: InlineArray[Float32, TP],
) -> Bool:
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    var total_bytes = total_elements * T.ELEMENT_BYTES

    for rank in range(TP):
        clear_buffer(arenas[rank], total_bytes)
        fill_bf16_value(arenas[rank], total_elements, rank_values[rank])

    var ptrs = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        ptrs[rank] = arenas[rank]

    allreduce_fn(ptrs, seq_len)

    var expected_sum = Float32(0)
    for rank in range(TP):
        expected_sum += Float32(Scalar[DType.bfloat16](rank_values[rank]))
    var expected_bf16 = Float32(Scalar[DType.bfloat16](expected_sum))

    for rank in range(TP):
        var check_indices = List[Int]()
        check_indices.append(0)
        check_indices.append(total_elements // 2)
        check_indices.append(total_elements - 1)
        for ci in range(len(check_indices)):
            var idx = check_indices[ci]
            var v = read_bf16_as_f32(arenas[rank], idx)
            if v != expected_bf16:
                print("FAIL", name, "- rank", rank, "element", idx, "got", v, "expected", expected_bf16)
                return False

    for rank in range(1, TP):
        var result = verify_match(arenas[0], arenas[rank], total_elements)
        if not result[0]:
            print("FAIL", name, "- rank", rank, "differs from rank 0 at element", result[1])
            return False

    print("PASS", name, "( sum =", expected_bf16, ",", total_elements, "elements )")
    return True


# =============================================================================
# Benchmark harness
# =============================================================================

def bench_allreduce[
    T: Encoding & Shaped,
    allreduce_fn: def(InlineArray[Int, TP], Int) capturing [_] -> None,
](
    name: String,
    seq_len: Int,
    arenas: List[Int],
    rank_values: InlineArray[Float32, TP],
    warmup: Int = 3,
    iters: Int = 20,
) -> Int:
    """Returns median time in microseconds."""
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    var total_bytes = total_elements * T.ELEMENT_BYTES

    var ptrs = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        ptrs[rank] = arenas[rank]

    # Warmup.
    for _ in range(warmup):
        for rank in range(TP):
            fill_bf16_value(arenas[rank], total_elements, rank_values[rank])
        allreduce_fn(ptrs, seq_len)

    # Timed runs.
    var times = List[Int]()
    for _ in range(iters):
        for rank in range(TP):
            fill_bf16_value(arenas[rank], total_elements, rank_values[rank])
        var t0 = perf_counter_ns()
        allreduce_fn(ptrs, seq_len)
        var t1 = perf_counter_ns()
        times.append(Int(t1 - t0))

    # Sort for median.
    for i in range(1, len(times)):
        var cur = times[i]
        var j = i
        while j > 0 and times[j - 1] > cur:
            times[j] = times[j - 1]
            j -= 1
        times[j] = cur

    var median_ns = times[len(times) // 2]
    var median_us = median_ns // 1000
    var total_data = total_bytes * TP
    var bw_gbps = Float64(total_data) / Float64(median_ns)  # GB/s
    print("  ", name, ":", median_us, "us  (", total_bytes, "B/rank,", bw_gbps, "GB/s effective )")
    return median_us


def bench_broadcast[
    T: Encoding & Shaped,
    broadcast_fn: def(Int, InlineArray[Int, TP], Int) capturing [_] -> None,
](
    name: String,
    seq_len: Int,
    arenas: List[Int],
    warmup: Int = 3,
    iters: Int = 20,
) -> Int:
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    var total_bytes = total_elements * T.ELEMENT_BYTES

    var ptrs = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        ptrs[rank] = arenas[rank]

    for _ in range(warmup):
        fill_pattern(arenas[0], total_elements, UInt16(42))
        broadcast_fn(arenas[0], ptrs, seq_len)

    var times = List[Int]()
    for _ in range(iters):
        fill_pattern(arenas[0], total_elements, UInt16(42))
        var t0 = perf_counter_ns()
        broadcast_fn(arenas[0], ptrs, seq_len)
        var t1 = perf_counter_ns()
        times.append(Int(t1 - t0))

    for i in range(1, len(times)):
        var cur = times[i]
        var j = i
        while j > 0 and times[j - 1] > cur:
            times[j] = times[j - 1]
            j -= 1
        times[j] = cur

    var median_ns = times[len(times) // 2]
    var median_us = median_ns // 1000
    var total_data = total_bytes * (TP - 1)
    var bw_gbps = Float64(total_data) / Float64(median_ns)
    print("  ", name, ":", median_us, "us  (", total_bytes, "B/rank,", bw_gbps, "GB/s effective )")
    return median_us


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Ring Communication Benchmark ===")
    print()

    var numa = NumaInfo()
    var topo = numa.plan_topology(TP)
    print("NUMA topology:", numa.num_nodes, "nodes")
    print("Selected nodes for TP=3:", end=" ")
    for r in range(TP):
        print(topo[r], end=" ")
    print()
    if numa.num_nodes > 1:
        print("Distance matrix:")
        for i in range(numa.num_nodes):
            for j in range(numa.num_nodes):
                print(numa.distance(i, j), end=" ")
            print()
    print()

    var arena_size = MAX_SEQ * COLS_LARGE * 2
    var arena_bases = List[Int]()
    var arenas_alive = HeapMoveArray[NumaArena[alignment=64]](TP)

    for rank in range(TP):
        var arena = NumaArena[alignment=64](topo[rank], arena_size)
        if not arena:
            print("FATAL: arena allocation failed for rank", rank)
            return
        arena_bases.append(Int(arena.base))
        arenas_alive.push(arena^)

    for rank in range(TP):
        _ = arenas_alive[rank].prefault()

    print("NUMA placement:")
    for rank in range(TP):
        var placed = arenas_alive[rank].verify_placement()
        print("  rank", rank, "-> node", topo[rank], "placement:", "OK" if placed else "UNKNOWN")
    print()

    var vals = InlineArray[Float32, TP](fill=Float32(0))
    vals[0] = Float32(1.0)
    vals[1] = Float32(2.0)
    vals[2] = Float32(3.0)

    var passed = 0
    var failed = 0

    # =================================================================
    # Correctness: validate all three allreduce implementations agree
    # =================================================================
    print("--- Correctness: Allreduce ---")

    @parameter
    def wrap_naive(ptrs: InlineArray[Int, TP], seq_len: Int):
        naive_allreduce[SlotMedium, TP](ptrs, seq_len)

    @parameter
    def wrap_ring_scalar(ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_allreduce_scalar[SlotMedium, TP](ptrs, seq_len)

    @parameter
    def wrap_ring_simd(ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_allreduce_simd[SlotMedium, TP](ptrs, seq_len)

    if validate_allreduce[SlotMedium, wrap_naive]("naive_seq64", 64, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    if validate_allreduce[SlotMedium, wrap_ring_scalar]("ring_scalar_seq64", 64, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    if validate_allreduce[SlotMedium, wrap_ring_simd]("ring_simd_seq64", 64, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    # Large scale.
    @parameter
    def wrap_naive_lg(ptrs: InlineArray[Int, TP], seq_len: Int):
        naive_allreduce[SlotLarge, TP](ptrs, seq_len)

    @parameter
    def wrap_ring_scalar_lg(ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_allreduce_scalar[SlotLarge, TP](ptrs, seq_len)

    @parameter
    def wrap_ring_simd_lg(ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_allreduce_simd[SlotLarge, TP](ptrs, seq_len)

    if validate_allreduce[SlotLarge, wrap_naive_lg]("naive_large_seq256", 256, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    if validate_allreduce[SlotLarge, wrap_ring_scalar_lg]("ring_scalar_large_seq256", 256, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    if validate_allreduce[SlotLarge, wrap_ring_simd_lg]("ring_simd_large_seq256", 256, arena_bases, vals):
        passed += 1
    else:
        failed += 1

    print()

    # =================================================================
    # Benchmark: Broadcast
    # =================================================================
    print("--- Benchmark: Broadcast (SmolLM2 hidden=576, seq=64) ---")

    @parameter
    def wrap_naive_bcast(src: Int, ptrs: InlineArray[Int, TP], seq_len: Int):
        naive_broadcast[SlotMedium, TP](src, ptrs, seq_len)

    @parameter
    def wrap_ring_bcast(src: Int, ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_broadcast[SlotMedium, TP](src, ptrs, seq_len)

    _ = bench_broadcast[SlotMedium, wrap_naive_bcast]("naive_star", 64, arena_bases)
    _ = bench_broadcast[SlotMedium, wrap_ring_bcast]("ring", 64, arena_bases)

    print()
    print("--- Benchmark: Broadcast (large=4096, seq=256) ---")

    @parameter
    def wrap_naive_bcast_lg(src: Int, ptrs: InlineArray[Int, TP], seq_len: Int):
        naive_broadcast[SlotLarge, TP](src, ptrs, seq_len)

    @parameter
    def wrap_ring_bcast_lg(src: Int, ptrs: InlineArray[Int, TP], seq_len: Int):
        ring_broadcast[SlotLarge, TP](src, ptrs, seq_len)

    _ = bench_broadcast[SlotLarge, wrap_naive_bcast_lg]("naive_star", 256, arena_bases)
    _ = bench_broadcast[SlotLarge, wrap_ring_bcast_lg]("ring", 256, arena_bases)

    print()

    # =================================================================
    # Benchmark: Allreduce
    # =================================================================
    print("--- Benchmark: Allreduce (SmolLM2 hidden=576, seq=1 decode) ---")
    _ = bench_allreduce[SlotMedium, wrap_naive]("naive", 1, arena_bases, vals)
    _ = bench_allreduce[SlotMedium, wrap_ring_scalar]("ring_scalar", 1, arena_bases, vals)
    _ = bench_allreduce[SlotMedium, wrap_ring_simd]("ring_simd", 1, arena_bases, vals)

    print()
    print("--- Benchmark: Allreduce (SmolLM2 hidden=576, seq=64 prefill) ---")
    _ = bench_allreduce[SlotMedium, wrap_naive]("naive", 64, arena_bases, vals)
    _ = bench_allreduce[SlotMedium, wrap_ring_scalar]("ring_scalar", 64, arena_bases, vals)
    _ = bench_allreduce[SlotMedium, wrap_ring_simd]("ring_simd", 64, arena_bases, vals)

    print()
    print("--- Benchmark: Allreduce (large=4096, seq=256) ---")
    _ = bench_allreduce[SlotLarge, wrap_naive_lg]("naive", 256, arena_bases, vals)
    _ = bench_allreduce[SlotLarge, wrap_ring_scalar_lg]("ring_scalar", 256, arena_bases, vals)
    _ = bench_allreduce[SlotLarge, wrap_ring_simd_lg]("ring_simd", 256, arena_bases, vals)

    print()
    print("=== Correctness:", passed, "passed,", failed, "failed ===")

    _ = arenas_alive
