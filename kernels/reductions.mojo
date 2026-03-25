"""Tree-topology collective operations for tensor parallelism.

Binary tree structure maximizes parallel memory controller utilization.
Each phase's transfers are independent — they touch disjoint source and
destination NUMA domains, so hardware can overlap them across controllers.

ring_allreduce: tree reduce to rank 0, then tree broadcast.
ring_broadcast: binary tree fanout from rank 0.

Rank ordering assumes maximal adjacency (as provided by NumaInfo.plan_topology).
"""

from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding, Shaped
from simd_math import bf16_load_as


def ring_allreduce[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Tree allreduce: reduce to rank 0, then broadcast.

    Phase 1 — tree reduce (leaves → root):
      Each phase pairs ranks at distance `stride`. Independent pairs
      accumulate in parallel across separate memory controllers.
      tp=4: {1→0, 3→2}, then {2→0}. Two phases, first fully parallel.

    Phase 2 — tree broadcast (root → leaves):
      Each phase doubles the set of ranks holding the result.
      tp=4: {0→1}, then {0→2, 1→3}. Two phases, second fully parallel.

    bf16 data with f32 SIMD accumulation.
    """
    comptime cols = T.COLS
    var total_elements = seq_len * cols
    if total_elements <= 0 or tp <= 1:
        return

    comptime width = simd_width_of[DType.float32]()
    var total_bytes = total_elements * T.ELEMENT_BYTES

    @always_inline
    def accumulate(dst_ptr: Int, src_ptr: Int, count: Int):
        """dst[:count] += src[:count] in bf16 with f32 SIMD."""
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=src_ptr)
        var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst_ptr)
        var i = 0
        while i + width <= count:
            var s_f32 = bf16_load_as[DType.float32, width](src, i)
            var d_f32 = bf16_load_as[DType.float32, width](dst, i)
            (dst + i).store((s_f32 + d_f32).cast[DType.bfloat16]())
            i += width
        while i < count:
            var s = Float32(src[i])
            var d = Float32(dst[i])
            dst[i] = Scalar[DType.bfloat16](s + d)
            i += 1

    # --- Phase 1: Tree reduce to rank 0 ---
    # stride=1: pairs (0,1), (2,3), ... — accumulate odd into even
    # stride=2: pairs (0,2), (4,6), ... — accumulate into lower
    # stride=4: pairs (0,4), ...
    # Each phase's pairs are independent (disjoint memory controllers).
    var stride = 1
    while stride < tp:
        for dst in range(0, tp, stride * 2):
            var src = dst + stride
            if src < tp:
                accumulate(ptrs[dst], ptrs[src], total_elements)
        stride *= 2

    # --- Phase 2: Tree broadcast from rank 0 ---
    # stride=1: 0→1
    # stride=2: 0→2, 1→3
    # stride=4: 0→4, 1→5, 2→6, 3→7
    # Each phase's copies are independent (disjoint destinations).
    stride = 1
    while stride < tp:
        for src in range(stride):
            var dst = src + stride
            if dst < tp:
                memcpy(
                    dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[dst]),
                    src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=ptrs[src]),
                    count=total_bytes,
                )
        stride *= 2


def ring_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int, dst_ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Tree broadcast: binary fanout from rank 0.

    Each phase doubles the number of ranks holding the data.
    Within a phase, all copies are independent — they read from
    distinct source ranks and write to distinct destination ranks,
    maximizing parallel memory controller utilization.

    tp=4: phase 0: {0→1}, phase 1: {0→2, 1→3}.
    ceil(log2(tp)) phases total vs tp-1 for sequential forwarding.
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

    # Binary tree fanout: each phase doubles the populated set.
    # stride=1: 0→1
    # stride=2: 0→2, 1→3
    # stride=4: 0→4, 1→5, 2→6, 3→7
    var stride = 1
    while stride < tp:
        for src in range(stride):
            var dst = src + stride
            if dst < tp:
                memcpy(
                    dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[dst]),
                    src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[src]),
                    count=total_bytes,
                )
        stride *= 2
