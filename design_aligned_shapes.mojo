"""Design exploration: aligned shape abstractions for TP padding.

Three proposals for making padding a first-class part of the shape system
so that allocation, loading, and runtime all derive correct dimensions from
a single source of truth.

The problem: INTERMEDIATE=2112, tp=4 → 528 per rank. 528 % 64 ≠ 0.
We need allocation at 576, file reads at 528, GEMV at 576, and colsums/packing
at 576. Currently these are computed independently → error-prone.

Run: pixi run mojo build -I . design_aligned_shapes.mojo && ./design_aligned_shapes
"""


def align_up(val: Int, a: Int) -> Int:
    return ((val + a - 1) // a) * a


# =============================================================================
# Proposal A: Dim carries alignment
#
# The dimension itself knows how to align after sharding. Most composable —
# each dimension is independently typed. But introduces a new abstraction.
# =============================================================================

struct Dim[global_val: Int, tp_divide: Bool, a: Int = 1]:
    @staticmethod
    def data[tp: Int]() -> Int:
        @parameter
        if Self.tp_divide:
            return Self.global_val // tp
        else:
            return Self.global_val

    @staticmethod
    def local[tp: Int]() -> Int:
        comptime d = Self.data[tp]()
        return align_up(d, Self.a)


def demo_proposal_a():
    print("=== Proposal A: Dim carries alignment ===")

    # down_proj: [HIDDEN, INTERMEDIATE], COL shard, align cols to 64
    print("down_proj [2816, 2112] COL shard, col_align=64:")
    alias R = Dim[2816, False]
    alias C = Dim[2112, True, 64]
    print("  tp=1: rows=" + String(R.local[1]()) + " cols=" + String(C.local[1]())
        + " data_cols=" + String(C.data[1]()))
    print("  tp=2: rows=" + String(R.local[2]()) + " cols=" + String(C.local[2]())
        + " data_cols=" + String(C.data[2]()))
    print("  tp=4: rows=" + String(R.local[4]()) + " cols=" + String(C.local[4]())
        + " data_cols=" + String(C.data[4]()))
    print()


# =============================================================================
# Proposal B: Shape[N, M, ...] — all-in-one, tp baked in
#
# Everything in one type. Most explicit — complete description in the
# signature. But verbose and you need a different type per tp degree.
# =============================================================================

struct ShapeB[
    global_n: Int, global_m: Int,
    shard_n: Bool, shard_m: Bool,
    tp: Int,
    align_n: Int = 1, align_m: Int = 1,
]:
    comptime DATA_N = Self.global_n // Self.tp if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.tp if Self.shard_m else Self.global_m
    comptime N = align_up(Self.DATA_N, Self.align_n)
    comptime M = align_up(Self.DATA_M, Self.align_m)
    comptime PAD_N = Self.N - Self.DATA_N
    comptime PAD_M = Self.M - Self.DATA_M


def demo_proposal_b():
    print("=== Proposal B: Shape[N, M, shard, tp, align] ===")

    alias Down4 = ShapeB[2816, 2112, False, True, 4, 1, 64]
    print("down_proj tp=4: N=" + String(Down4.N) + " M=" + String(Down4.M)
        + " DATA_M=" + String(Down4.DATA_M) + " PAD_M=" + String(Down4.PAD_M))

    alias Down1 = ShapeB[2816, 2112, False, True, 1, 1, 64]
    print("down_proj tp=1: N=" + String(Down1.N) + " M=" + String(Down1.M)
        + " DATA_M=" + String(Down1.DATA_M) + " PAD_M=" + String(Down1.PAD_M))

    alias Gate4 = ShapeB[2112, 2816, True, False, 4]
    print("gate_proj tp=4: N=" + String(Gate4.N) + " M=" + String(Gate4.M))
    print()


# =============================================================================
# Proposal C: Shard strategy owns alignment (minimal diff from current)
#
# Extends the existing DimStrategy trait with data(). Adds one new struct.
# Smallest change to existing code. Alignment is a property of the shard
# strategy, which is how the model already thinks about sharding.
# =============================================================================

# Current trait, extended with data()
trait DimC:
    @staticmethod
    def local(d: Int, tp: Int) -> Int: ...
    @staticmethod
    def data(d: Int, tp: Int) -> Int: ...

struct DivideC(DimC):
    @staticmethod
    def local(d: Int, tp: Int) -> Int:
        return d // tp
    @staticmethod
    def data(d: Int, tp: Int) -> Int:
        return d // tp

struct KeepC(DimC):
    @staticmethod
    def local(d: Int, tp: Int) -> Int:
        return d
    @staticmethod
    def data(d: Int, tp: Int) -> Int:
        return d

struct DivideAlignedC[a: Int](DimC):
    @staticmethod
    def local(d: Int, tp: Int) -> Int:
        return align_up(d // tp, Self.a)
    @staticmethod
    def data(d: Int, tp: Int) -> Int:
        return d // tp


def demo_proposal_c():
    print("=== Proposal C: Shard strategy owns alignment ===")

    # Direct usage — no Slot wrapper needed to demonstrate
    print("down_proj [2816, 2112] COL=DivideAligned[64]:")
    alias ColStrat = DivideAlignedC[64]
    alias RowStrat = KeepC
    print("  tp=1: rows=" + String(RowStrat.local(2816, 1))
        + " cols=" + String(ColStrat.local(2112, 1))
        + " data_cols=" + String(ColStrat.data(2112, 1)))
    print("  tp=4: rows=" + String(RowStrat.local(2816, 4))
        + " cols=" + String(ColStrat.local(2112, 4))
        + " data_cols=" + String(ColStrat.data(2112, 4)))
    print()


# =============================================================================
# Analysis
# =============================================================================

def analysis():
    print("=== Analysis ===")
    print()
    print("All three produce the same numbers:")
    print("  tp=4 down_proj: alloc_cols=576, data_cols=528, pad=48")
    print()
    print("Proposal A (Dim carries alignment):")
    print("  + Most composable — Dim[2112, True, 64] is reusable")
    print("  + Shape is just ShapeA[RowDim, ColDim]")
    print("  - New abstraction layer (Dim replaces raw Int)")
    print("  - Mojo struggles with parametric trait conformance on")
    print("    structs with many params (ShapeA[R,C] can't easily")
    print("    conform to ShardStrategy trait)")
    print()
    print("Proposal B (all-in-one Shape):")
    print("  + Most explicit — everything in one place")
    print("  + All derived quantities are comptime fields")
    print("  - Verbose: 7 parameters per shape")
    print("  - tp baked in → need different type per tp degree")
    print("  - Doesn't compose with existing trait system")
    print()
    print("Proposal C (shard strategy + data()):")
    print("  + Smallest diff from current code")
    print("  + Fits existing trait system exactly")
    print("  + DivideAligned[64] is a drop-in alongside Divide/Keep")
    print("  - Alignment lives in strategy, not shape — fine for")
    print("    our use case (alignment is always tied to sharding)")
    print("  - Needs data_rows/data_cols on WeightDesc + loader fix")
    print()
    print("Recommendation: Proposal C.")
    print("  It's the smallest change that solves the problem.")
    print("  The emit system already uses ShardStrategy to compute")
    print("  local dims — adding data() is natural. The loader")
    print("  already has the COL shard path; it just needs to use")
    print("  data_cols for file reads and local_cols for stride.")
    print("  No new abstractions, no trait system fights.")
    print()

    # Concrete impact on the codebase
    print("=== Concrete changes for Proposal C ===")
    print()
    print("model_spec.mojo:")
    print("  1. Add data() to DimStrategy trait")
    print("  2. Add DivideAligned[align] struct")
    print("  3. Add data_rows/data_cols to WeightDesc")
    print("  4. Update weight_desc[] to populate them")
    print()
    print("loader.mojo:")
    print("  5. emit_reads COL path: read data_cols, stride local_cols")
    print()
    print("gemma_4_moe_butterquant_tp.mojo:")
    print("  6. down_proj uses COL_ALIGNED_64 shard mode")
    print("  7. All downstream dims (colsums, pack, GEMV) use COLS")
    print("     which is now 576 — automatically correct")
    print()
    print("That's it. 7 small changes, no new abstractions.")


def main():
    demo_proposal_a()
    demo_proposal_b()
    demo_proposal_c()
    analysis()
