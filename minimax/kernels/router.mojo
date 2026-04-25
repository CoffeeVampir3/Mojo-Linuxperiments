from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of

from minimax.kernels.activations import sigmoid_f32
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs
from minimax.kernels.dispatch_args import (
    RouterCandidate, TopKResult, SparseRoute, RouterFusedArgs,
)


@always_inline
def bf16_dot_row[hidden: Int](
    lhs: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rhs: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, hidden]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var a = (lhs + off).load[width=width]().cast[DType.float32]()
        var b = (rhs + off).load[width=width]().cast[DType.float32]()
        accs[i] = a * b
    var k = step
    while k + step <= hidden:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var a = (lhs + off).load[width=width]().cast[DType.float32]()
            var b = (rhs + off).load[width=width]().cast[DType.float32]()
            accs[i] = a.fma(b, accs[i])
        k += step
    while k + width <= hidden:
        var a = (lhs + k).load[width=width]().cast[DType.float32]()
        var b = (rhs + k).load[width=width]().cast[DType.float32]()
        accs[0] = a.fma(b, accs[0])
        k += width
    return tree_reduce_accs(accs)


def router_fused_worker[hidden: Int, k: Int](args: RouterFusedArgs):
    """Phase 1: centered bf16 GEMV + gauge pivot + sigmoid + local top-K.

    For every assigned activation row and local expert row:
      pivot = bf16[hidden] · bf16[hidden] gauge
      dot   = bf16[hidden] · bf16[hidden] centered_row + pivot
      raw   = sigmoid(dot)
      score = raw + bias[e]
    Maintain a K-slot sorted buffer (descending by score, lowest-eid
    wins ties), writing one K-candidate row per activation row.
    """
    for row in range(args.row_count):
        var act_row = args.act_bf16 + row * hidden
        var candidate_row = args.candidates + row * args.candidate_row_stride
        var eid_buf = InlineArray[Int32, k](fill=Int32(-1))
        var score_buf = InlineArray[Float32, k](fill=Float32(-1e30))
        var raw_buf = InlineArray[Float32, k](fill=Float32(0))
        var pivot = bf16_dot_row[hidden](act_row, args.gauge_bf16)

        for n in range(args.n_count):
            var local_row = args.n_start + n
            var eid = args.eid_base + local_row
            var dot = bf16_dot_row[hidden](
                act_row, args.weight_bf16 + local_row * hidden) + pivot
            var raw = sigmoid_f32[1](SIMD[DType.float32, 1](dot))[0]
            var score = raw + args.bias_f32[local_row]
            if score > score_buf[k - 1]:
                var slot = k - 1
                while slot > 0 and score > score_buf[slot - 1]:
                    eid_buf[slot] = eid_buf[slot - 1]
                    score_buf[slot] = score_buf[slot - 1]
                    raw_buf[slot] = raw_buf[slot - 1]
                    slot -= 1
                eid_buf[slot] = Int32(eid)
                score_buf[slot] = score
                raw_buf[slot] = raw

        for s in range(k):
            candidate_row[s] = RouterCandidate(
                eid_buf[s], score_buf[s], raw_buf[s])


def router_merge_multi_and_renorm[k: Int, num_sources: Int](
    candidate_ptrs: InlineArray[UnsafePointer[RouterCandidate, MutAnyOrigin], num_sources],
    candidate_counts: InlineArray[Int, num_sources],
    result_ptr: UnsafePointer[TopKResult[k], MutAnyOrigin],
):
    """Merge rank-local router candidates into one global TopKResult[k]."""
    var eid_buf = InlineArray[Int32, k](fill=Int32(-1))
    var score_buf = InlineArray[Float32, k](fill=Float32(-1e30))
    var raw_buf = InlineArray[Float32, k](fill=Float32(0))

    for src in range(num_sources):
        var candidates = candidate_ptrs[src]
        for c in range(candidate_counts[src]):
            var score = candidates[c].score
            if score > score_buf[k - 1]:
                var eid = candidates[c].eid
                var raw = candidates[c].raw
                var slot = k - 1
                while slot > 0 and score > score_buf[slot - 1]:
                    eid_buf[slot] = eid_buf[slot - 1]
                    score_buf[slot] = score_buf[slot - 1]
                    raw_buf[slot] = raw_buf[slot - 1]
                    slot -= 1
                eid_buf[slot] = eid
                score_buf[slot] = score
                raw_buf[slot] = raw

    var sum_raw = Float32(0)
    for s in range(k):
        sum_raw += raw_buf[s]
    var inv = Float32(1.0) / sum_raw

    var result = TopKResult[k](
        indices=InlineArray[Int, k](uninitialized=True),
        weights=InlineArray[Float32, k](uninitialized=True),
    )
    for s in range(k):
        result.indices[s] = Int(eid_buf[s])
        result.weights[s] = raw_buf[s] * inv

    result_ptr[] = result


def build_sparse_route_schedule[k: Int, experts_per_rank: Int](
    routing: UnsafePointer[TopKResult[k], MutAnyOrigin],
    seq_len: Int,
    rank: Int,
    counts: UnsafePointer[Int32, MutAnyOrigin],
    offsets: UnsafePointer[Int32, MutAnyOrigin],
    cursors: UnsafePointer[Int32, MutAnyOrigin],
    routes: UnsafePointer[SparseRoute, MutAnyOrigin],
) -> Int:
    """Counting-sort final routes into rank-local expert buckets.

    Static MiniMax placement makes ownership arithmetic:
      expert_base = rank * experts_per_rank
      local_expert = eid - expert_base

    Outputs:
      counts[e]  = route count for local expert e
      offsets[e] = prefix start for local expert e
      routes[offsets[e] : offsets[e + 1]] are that expert's token routes

    Each stored route only needs the source token and normalized gate weight;
    the enclosing bucket identifies the local expert.
    Returns the total number of local routes for this rank.
    """
    var expert_base = rank * experts_per_rank
    for e in range(experts_per_rank):
        counts[e] = Int32(0)

    for token in range(seq_len):
        var r = routing[token]
        for slot in range(k):
            var eid = r.indices[slot]
            if eid >= expert_base and eid < expert_base + experts_per_rank:
                var local_expert = eid - expert_base
                counts[local_expert] = counts[local_expert] + Int32(1)

    offsets[0] = Int32(0)
    var total = 0
    for e in range(experts_per_rank):
        total += Int(counts[e])
        offsets[e + 1] = Int32(total)
        cursors[e] = offsets[e]

    for token in range(seq_len):
        var r = routing[token]
        for slot in range(k):
            var eid = r.indices[slot]
            if eid >= expert_base and eid < expert_base + experts_per_rank:
                var local_expert = eid - expert_base
                var dst = Int(cursors[local_expert])
                routes[dst] = SparseRoute(
                    Int32(token), r.weights[slot])
                cursors[local_expert] = cursors[local_expert] + Int32(1)

    return total
