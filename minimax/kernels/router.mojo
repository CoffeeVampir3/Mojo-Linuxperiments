from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.common_math import F32Ptr, BF16Ptr
from minimax.kernels.activations import sigmoid_f32
from minimax.kernels.gemm import f32_gemv_row
from minimax.kernels.dispatch_args import (
    RouterCandidate, TopKResult, RouterFusedArgs, RouterMergeArgs,
)


def router_fused_worker[hidden: Int, k: Int](args: RouterFusedArgs):
    """Phase 1: f32 GEMV + sigmoid + bias + local top-K.

    For each assigned expert row e in [n_start, n_start + n_count):
      dot   = bf16[hidden] · f32[hidden]
      raw   = sigmoid(dot)
      score = raw + bias[e]
    Maintain a K-slot sorted buffer (descending by score, lowest-eid
    wins ties), write it to candidates[0..K).
    """
    var eid_buf = InlineArray[Int32, k](fill=Int32(-1))
    var score_buf = InlineArray[Float32, k](fill=Float32(-1e30))
    var raw_buf = InlineArray[Float32, k](fill=Float32(0))

    for n in range(args.n_count):
        var eid = args.n_start + n
        var dot = f32_gemv_row[hidden](
            args.act_bf16, args.weight_f32 + eid * hidden)
        var raw = sigmoid_f32[1](SIMD[DType.float32, 1](dot))[0]
        var score = raw + args.bias_f32[eid]
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
        args.candidates[s] = RouterCandidate(eid_buf[s], score_buf[s], raw_buf[s])


def router_merge_and_renorm[k: Int](args: RouterMergeArgs[k]):
    """Phase 2: merge worker candidates → TopKResult[k], renormalize.

    Scalar insert each candidate into a K-slot sorted buffer, then
    normalize the raw sigmoid weights of the winners to sum = 1.
    """
    var eid_buf = InlineArray[Int32, k](fill=Int32(-1))
    var score_buf = InlineArray[Float32, k](fill=Float32(-1e30))
    var raw_buf = InlineArray[Float32, k](fill=Float32(0))

    for c in range(args.num_candidates):
        var score = args.candidates[c].score
        if score > score_buf[k - 1]:
            var eid = args.candidates[c].eid
            var raw = args.candidates[c].raw
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

    args.result_ptr[] = result
