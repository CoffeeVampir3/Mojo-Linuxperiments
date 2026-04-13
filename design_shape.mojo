"""Design: Shape as the single source of truth for tensor geometry.

Shape[N, M, shard_n, shard_m, tp, align_n, align_m] knows:
  - Global dims (N, M before sharding)
  - Local dims (after shard + alignment)
  - Data dims (what's in the file, no padding)
  - Allocation size for any dtype

Everything downstream — Slot, WeightDesc, loader, runtime — derives from Shape.

Run: pixi run mojo build -I . design_shape.mojo && ./design_shape
"""


# =============================================================================
# Shape
# =============================================================================

def _align_up(val: Int, a: Int) -> Int:
    return ((val + a - 1) // a) * a

struct Shape[
    global_n: Int, global_m: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    tp: Int = 1,
    align_n: Int = 1, align_m: Int = 1,
]:
    # Data dims — what exists in the file per rank
    comptime DATA_N = Self.global_n // Self.tp if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.tp if Self.shard_m else Self.global_m

    # Local dims — allocated per rank (padded)
    comptime N = _align_up(Self.DATA_N, Self.align_n)
    comptime M = _align_up(Self.DATA_M, Self.align_m)

    # Padding
    comptime PAD_N = Self.N - Self.DATA_N
    comptime PAD_M = Self.M - Self.DATA_M

    @staticmethod
    def alloc_bytes[elem_bytes: Int]() -> Int:
        return Self.N * Self.M * elem_bytes

    @staticmethod
    def data_bytes[elem_bytes: Int]() -> Int:
        return Self.DATA_N * Self.DATA_M * elem_bytes

    @staticmethod
    def row_stride[elem_bytes: Int]() -> Int:
        return Self.M * elem_bytes


# =============================================================================
# How this replaces current Slot + shard strategy
# =============================================================================

# Current approach:
#   comptime RowShard = Shard2D[Divide, Keep]
#   Slot[I8, RowShard, 2112, 2816, tp=4]
#     → ROWS = 528, COLS = 2816
#
# New approach:
#   alias GateShape = Shape[2112, 2816, shard_n=True, tp=4]
#     → N = 528, M = 2816, DATA_N = 528
#     → alloc_bytes[1]() = 528 * 2816 = 1,486,848
#
# With alignment:
#   alias DownShape = Shape[2816, 2112, shard_m=True, tp=4, align_m=64]
#     → N = 2816, M = 576, DATA_M = 528
#     → alloc_bytes[1]() = 2816 * 576 = 1,621,696


# =============================================================================
# Thin Slot — just Shape + placement metadata
# =============================================================================

struct TensorSlot[S: Shape, offset: Int, elem_bytes: Int]:
    comptime N = Self.S.N
    comptime M = Self.S.M
    comptime DATA_N = Self.S.DATA_N
    comptime DATA_M = Self.S.DATA_M
    comptime OFFSET = Self.offset
    comptime ALLOC = Self.S.alloc_bytes[Self.elem_bytes]()

    @staticmethod
    def next_offset[alignment: Int = 64]() -> Int:
        comptime aligned_off = _align_up(Self.OFFSET, alignment)
        return aligned_off + Self.ALLOC


# =============================================================================
# Demo: Gemma4 dense FFN layout
# =============================================================================

def demo_gemma4[tp: Int]():
    print("--- Gemma4 dense FFN shapes, tp=" + String(tp) + " ---")

    alias H = 2816
    alias INT = 2112

    # gate/up: ROW shard (divide N), K=HIDDEN stays full
    # N needs VNNI_N_STEP=32 alignment for the GEMV output
    alias GateUp = Shape[INT, H, shard_n=True, tp=tp, align_n=32]
    print("gate/up [" + String(INT) + ", " + String(H) + "] ROW shard, align_n=32:")
    print("  N=" + String(GateUp.N) + " M=" + String(GateUp.M)
        + " DATA_N=" + String(GateUp.DATA_N)
        + " alloc_i8=" + String(GateUp.alloc_bytes[1]()) + " bytes")

    # gate/up scale: ROW shard, same N alignment, 1 col
    alias GateUpScale = Shape[INT, 1, shard_n=True, tp=tp, align_n=32]
    print("gate/up_scale [" + String(INT) + ", 1] ROW shard, align_n=32:")
    print("  N=" + String(GateUpScale.N)
        + " DATA_N=" + String(GateUpScale.DATA_N)
        + " alloc_f32=" + String(GateUpScale.alloc_bytes[4]()) + " bytes")

    # down: COL shard (divide M), N=HIDDEN stays full
    # M needs VNNI_K_STEP=64 alignment for the GEMV input
    alias Down = Shape[H, INT, shard_m=True, tp=tp, align_m=64]
    print("down [" + String(H) + ", " + String(INT) + "] COL shard, align_m=64:")
    print("  N=" + String(Down.N) + " M=" + String(Down.M)
        + " DATA_M=" + String(Down.DATA_M)
        + " pad=" + String(Down.PAD_M)
        + " alloc_i8=" + String(Down.alloc_bytes[1]()) + " bytes")

    # down scale: replicated [H, 1] — no shard, no align
    alias DownScale = Shape[H, 1]
    print("down_scale [" + String(H) + ", 1] replicated:")
    print("  alloc_f32=" + String(DownScale.alloc_bytes[4]()) + " bytes")

    # Colsum for down: per-block [H, K/block] where K is the padded M
    # The shape system gives us Down.M (padded), so block count is automatic
    comptime DBLK = 64 if tp == 1 else 16
    comptime DOWN_NUM_BLK = Down.M // DBLK
    print("down_colsum: blocks_per_row=" + String(DOWN_NUM_BLK)
        + " total_f32=" + String(H * DOWN_NUM_BLK))

    # Scratch lease for intermediate activation: needs Down.M elements
    # (padded, so the gap between real data and padding stays zero)
    print("intermediate scratch: " + String(Down.M) + " i8 elements"
        + " (" + String(Down.DATA_M) + " real + " + String(Down.PAD_M) + " pad)")

    print()


def demo_consistency():
    print("--- Consistency check: gate/up N == down M ---")

    alias H = 2816
    alias INT = 2112

    # At tp=4 with alignment, gate/up and down must agree on the
    # intermediate dimension.
    alias GateUp4 = Shape[INT, H, shard_n=True, tp=4, align_n=32]
    alias Down4   = Shape[H, INT, shard_m=True, tp=4, align_m=64]

    print("gate/up N (output dim): " + String(GateUp4.N))
    print("down M (input dim):     " + String(Down4.M))
    print("gate/up DATA_N:         " + String(GateUp4.DATA_N))
    print("down DATA_M:            " + String(Down4.DATA_M))

    # DATA dims match (528), but padded dims might differ!
    # gate/up aligns to 32 → align_up(528, 32) = 544
    # down aligns to 64 → align_up(528, 64) = 576
    # This is a PROBLEM — the intermediate buffer between them
    # needs a single consistent size.
    print()
    print("NOTE: gate/up padded N=" + String(GateUp4.N) + " != down padded M=" + String(Down4.M))
    print("The intermediate buffer must use max(544, 576) = 576.")
    print("Or: use the same alignment on both (align=64 on gate/up too).")
    print()

    # Fix: align both to 64
    alias GateUp4_fixed = Shape[INT, H, shard_n=True, tp=4, align_n=64]
    alias Down4_fixed   = Shape[H, INT, shard_m=True, tp=4, align_m=64]
    print("Fixed (both align to 64):")
    print("  gate/up N=" + String(GateUp4_fixed.N) + " down M=" + String(Down4_fixed.M)
        + " — consistent")

    print()


def demo_loader_contract():
    print("--- Loader contract ---")
    print()
    print("Shape provides everything emit_reads needs:")
    print()

    alias Down4 = Shape[2816, 2112, shard_m=True, tp=4, align_m=64]

    print("  COL shard path (down_proj tp=4):")
    print("    data_cols = DATA_M = " + String(Down4.DATA_M) + " (bytes to read per row)")
    print("    stride    = M      = " + String(Down4.M) + " (row pitch in arena)")
    print("    rows      = N      = " + String(Down4.N))
    print("    col_start = rank * DATA_M")
    print()
    print("  For each row r:")
    print("    src = file_data_start + (r * global_cols + rank * DATA_M) * elem_bytes")
    print("    dst = arena_base + offset + r * M * elem_bytes")
    print("    len = DATA_M * elem_bytes")
    print()
    print("  The gap (M - DATA_M) = " + String(Down4.PAD_M) + " bytes per row")
    print("  stays zero from mmap. No explicit zeroing needed.")
    print()


def main():
    demo_gemma4[1]()
    demo_gemma4[2]()
    demo_gemma4[4]()
    demo_consistency()
    demo_loader_contract()
