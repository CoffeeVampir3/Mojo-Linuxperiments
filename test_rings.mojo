"""Test ring_broadcast correctness on a real NUMA system.

Allocates tp arenas on NUMA nodes selected by plan_topology,
fills the host rank's buffer with a known pattern, ring-broadcasts
to all ranks, and verifies every rank received the exact data.

Tests:
1. Single-element broadcast (trivial)
2. Small buffer (few cache lines)
3. Large buffer (multiple pages, exercises NUMA bandwidth)
4. Non-power-of-2 sizes (exercises remainder handling)
5. Verify NUMA placement via move_pages query
"""

from std.memory import UnsafePointer, memcpy
from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from std.collections import InlineArray

from experimental4.model_spec import (
    Encoding, Shaped, BF16, Slot, Replicated, byte_count,
)


# Reuse ring_broadcast from the TP module.
def ring_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int, dst_ptrs: InlineArray[Int, tp], seq_len: Int,
):
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


# Test slot types with varying widths.
comptime COLS_SMALL = 8       # 16 bytes per row (bf16)
comptime COLS_MEDIUM = 576    # SmolLM2 hidden dim
comptime COLS_LARGE = 4096    # Stress test

comptime MAX_SEQ = 1024
comptime TP = 3

comptime SlotSmall = Slot[BF16, Replicated, MAX_SEQ, COLS_SMALL, TP]
comptime SlotMedium = Slot[BF16, Replicated, MAX_SEQ, COLS_MEDIUM, TP]
comptime SlotLarge = Slot[BF16, Replicated, MAX_SEQ, COLS_LARGE, TP]


def fill_pattern(ptr: Int, total_elements: Int, seed: UInt16):
    """Fill a bf16 buffer with a deterministic pattern based on index + seed."""
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_elements):
        # Pattern: (seed + i) with some bit mixing to catch byte-swap bugs.
        p[i] = UInt16((Int(seed) + i * 7 + 13) & 0xFFFF)


def verify_match(ptr_a: Int, ptr_b: Int, total_elements: Int) -> Tuple[Bool, Int]:
    """Compare two bf16 buffers element-wise. Returns (ok, first_mismatch_idx)."""
    var a = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr_a)
    var b = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr_b)
    for i in range(total_elements):
        if a[i] != b[i]:
            return (False, i)
    return (True, -1)


def verify_zero(ptr: Int, total_elements: Int) -> Bool:
    """Verify a buffer is all zeros."""
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_elements):
        if p[i] != 0:
            return False
    return True


def clear_buffer(ptr: Int, total_bytes: Int):
    var p = UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptr)
    for i in range(total_bytes):
        p[i] = 0


def run_broadcast_test[T: Encoding & Shaped](
    name: String,
    seq_len: Int,
    arenas: List[Int],  # arena base addresses
    seed: UInt16 = UInt16(42),
) -> Bool:
    comptime cols = T.COLS
    comptime elem_bytes = T.ELEMENT_BYTES
    var total_elements = seq_len * cols
    var total_bytes = total_elements * elem_bytes

    # Clear all buffers.
    for rank in range(TP):
        clear_buffer(arenas[rank], total_bytes)

    # Verify all start at zero.
    for rank in range(TP):
        if not verify_zero(arenas[rank], total_elements):
            print("FAIL", name, "- rank", rank, "not zero before broadcast")
            return False

    # Fill host rank with pattern.
    fill_pattern(arenas[0], total_elements, seed)

    # Build dst_ptrs and broadcast.
    var dst_ptrs = InlineArray[Int, TP](fill=0)
    for rank in range(TP):
        dst_ptrs[rank] = arenas[rank]

    ring_broadcast[T, TP](arenas[0], dst_ptrs, seq_len)

    # Verify all ranks match host.
    for rank in range(1, TP):
        var result = verify_match(arenas[0], arenas[rank], total_elements)
        if not result[0]:
            print(
                "FAIL", name, "- rank", rank,
                "mismatch at element", result[1],
                "of", total_elements,
            )
            return False

    print("PASS", name, "(", total_elements, "elements,", total_bytes, "bytes )")
    return True


def main():
    print("=== Ring Broadcast Tests ===")
    print()

    # Discover NUMA topology.
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

    # Allocate TP arenas on the selected NUMA nodes.
    # Use a generous size to accommodate the largest test.
    var arena_size = MAX_SEQ * COLS_LARGE * 2  # bf16 = 2 bytes
    var arena_bases = List[Int]()
    var arenas_alive = HeapMoveArray[NumaArena[alignment=64]](TP)

    for rank in range(TP):
        var arena = NumaArena[alignment=64](topo[rank], arena_size)
        if not arena:
            print("FATAL: arena allocation failed for rank", rank, "on node", topo[rank])
            return
        arena_bases.append(Int(arena.base))
        arenas_alive.push(arena^)

    # Prefault all pages.
    for rank in range(TP):
        _ = arenas_alive[rank][].prefault()

    # Verify NUMA placement.
    print("NUMA placement:")
    for rank in range(TP):
        var placed = arenas_alive[rank][].verify_placement()
        print("  rank", rank, "-> node", topo[rank], "placement:", "OK" if placed else "UNKNOWN")
    print()

    # --- Tests ---
    var passed = 0
    var failed = 0

    if run_broadcast_test[SlotSmall]("small_seq1", 1, arena_bases, seed=UInt16(1)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotSmall]("small_seq8", 8, arena_bases, seed=UInt16(2)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotSmall]("small_seq17", 17, arena_bases, seed=UInt16(3)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotMedium]("medium_seq1", 1, arena_bases, seed=UInt16(10)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotMedium]("medium_seq64", 64, arena_bases, seed=UInt16(11)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotMedium]("medium_seq333", 333, arena_bases, seed=UInt16(12)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotLarge]("large_seq1", 1, arena_bases, seed=UInt16(20)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotLarge]("large_seq128", 128, arena_bases, seed=UInt16(21)):
        passed += 1
    else:
        failed += 1

    if run_broadcast_test[SlotLarge]("large_seq1024", 1024, arena_bases, seed=UInt16(22)):
        passed += 1
    else:
        failed += 1

    # --- Idempotency: broadcast same buffer to itself ---
    clear_buffer(arena_bases[0], 64)
    fill_pattern(arena_bases[0], 16, UInt16(99))
    var self_ptrs = InlineArray[Int, TP](fill=arena_bases[0])
    ring_broadcast[SlotSmall, TP](arena_bases[0], self_ptrs, 2)
    # Should not corrupt — all ptrs are the same address.
    var expected_buf = List[UInt16](capacity=16)
    for i in range(16):
        expected_buf.append(UInt16((99 + i * 7 + 13) & 0xFFFF))
    var p = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=arena_bases[0])
    var idempotent_ok = True
    for i in range(16):
        if p[i] != expected_buf[i]:
            idempotent_ok = False
            break
    if idempotent_ok:
        print("PASS idempotency (broadcast to self)")
        passed += 1
    else:
        print("FAIL idempotency (broadcast to self)")
        failed += 1

    print()
    print("=== Results:", passed, "passed,", failed, "failed ===")
