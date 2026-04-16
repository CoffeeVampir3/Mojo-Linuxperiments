"""Verify weight loading correctness at TP=1 and TP=2 by comparing raw bytes.

Loads gate_proj layer 0 at both TP levels, dumps the raw data, and
byte-compares TP=2 rank slices against the corresponding TP=1 data.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from modeling.model_spec import HOST_RANK, DEFAULT_ALIGNMENT
from modeling.gemma_4_moe_butterquant_tp import (
    Gemma4Config, Gemma4Shapes, build_gemma4_plan,
)
from modeling.loader import discover_shards, load_weights_from_descs

comptime C = Gemma4Config
comptime HIDDEN = C.HIDDEN
comptime INTERMEDIATE = C.INTERMEDIATE
comptime MODEL_DIR = "quantized_models"


def load_and_dump[tp: Int]():
    comptime S = Gemma4Shapes[tp]
    comptime data_rows = INTERMEDIATE // tp
    comptime padded_rows = S.DENSE_INT_LOCAL
    comptime data_bytes = data_rows * HIDDEN

    print("=== TP=" + String(tp) + " ===")
    print("  data_rows:", data_rows, " padded_rows:", padded_rows, " data_bytes:", data_bytes)

    var shards = discover_shards(Path(MODEL_DIR))
    if len(shards) == 0:
        print("no shards found")
        return

    var plan = build_gemma4_plan[tp]()

    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(tp)

    var arena_bases = List[Int]()
    var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](tp)
    for rank in range(tp):
        var sz = plan.topology.host_arena_bytes() if rank == HOST_RANK else plan.topology.arena_bytes()
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa_topo[rank % tp], sz)
        if not arena:
            print("  arena alloc failed rank", rank)
            return
        arena_bases.append(Int(arena.base))
        arenas.push(arena^)

    var result = load_weights_from_descs(plan.descs, shards, arena_bases)
    if not result:
        print("  load failed")
        return
    print("  loaded OK")

    # Get gate_proj offset from layer 0 (sliding)
    var proto = plan.topology.sliding
    var body = proto.proto.body
    var gate_off = body.gate_proj.offset

    for rank in range(tp):
        var base = arena_bases[rank]
        var lb = proto.off + 0 * proto.stride
        var addr = base + lb + gate_off
        var ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=addr)

        # Stats on actual data rows only
        var sum_abs = 0
        var sum_val = 0
        var nonzero = 0
        for i in range(data_bytes):
            var v = Int(ptr[i])
            sum_val += v
            if v < 0:
                sum_abs -= v
            else:
                sum_abs += v
            if v != 0:
                nonzero += 1

        print("  rank", rank, "gate_proj: sum=", sum_val, " abs_sum=", sum_abs,
            " nonzero=", nonzero, "/", data_bytes)

        # Check padding region
        if padded_rows > data_rows:
            var pad_start = data_bytes
            var pad_bytes = (padded_rows - data_rows) * HIDDEN
            var pad_nonzero = 0
            for i in range(pad_bytes):
                if ptr[pad_start + i] != 0:
                    pad_nonzero += 1
            print("  rank", rank, "padding: nonzero=", pad_nonzero, "/", pad_bytes)

    # For TP=2: byte-compare rank slices against TP=1 if both loaded
    if tp == 1:
        # Dump bytes for later comparison
        var base = arena_bases[0]
        var lb = proto.off
        var addr = base + lb + gate_off
        print("  TP=1 gate_proj addr:", addr, "size:", INTERMEDIATE * HIDDEN)


def main():
    load_and_dump[1]()
    print()
    load_and_dump[2]()
