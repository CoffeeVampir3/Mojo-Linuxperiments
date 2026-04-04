"""Validate ButterQuant's fixed-scale assumption on real SmolLM2 activations.

Loads SmolLM2-135M bf16, reads embedding rows, applies RMSNorm (no gamma)
+ block FWHT, and measures whether per-block absmax concentrates near C(n).

The embedding layer is the adversarial case — it hasn't been through prior
Hadamard processing, so block energy may be non-uniform.
"""

from std.pathlib import Path
from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from simd_math import sqrt

from modeling.smollm2_tp import SmolLM2TP, SmolLM2Config
from modeling.model_spec import BF16
from modeling.smollm2_butterquant_tp import concentration_constant, comptime_sqrt
from experimental.hadquant_impl import fwht_row

comptime C = SmolLM2Config
comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime HIDDEN = C.HIDDEN
comptime HD = C.HEAD_DIM
comptime BLOCKS = HIDDEN // HD
comptime NUM_TOKENS = 64


def main():
    var cn = concentration_constant[HD]()

    var model_opt = SmolLM2TP[BF16, 1].load(Path(MODEL_PATH))
    if not model_opt:
        print("Failed to load model")
        return
    var model = model_opt.take()

    comptime M = SmolLM2TP[BF16, 1].M
    var rv = model.rank(0)
    var embed_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=rv.weight[M.EMBED]().ptr)

    var tokens = InlineArray[Int, NUM_TOKENS](fill=0)
    for i in range(NUM_TOKENS):
        tokens[i] = (i * (C.VOCAB_SIZE // NUM_TOKENS)) + 7

    var work = alloc[Scalar[DType.float32]](HIDDEN)

    # Per-block statistics
    var total_blocks = 0
    var total_elements = 0
    var clipped_elements = 0
    var ratio_sum = Float64(0)
    var ratio_max = Float64(0)
    var ratio_min = Float64(1e9)

    # Per-block energy uniformity: track block norms
    var block_norm_sum = InlineArray[Float64, BLOCKS](fill=Float64(0))
    var block_norm_max = InlineArray[Float64, BLOCKS](fill=Float64(0))

    # Histogram: ratio buckets [0-0.5, 0.5-0.75, 0.75-1.0, 1.0-1.25, 1.25-1.5, 1.5+]
    var hist = InlineArray[Int, 6](fill=0)

    var cn_f32 = Float32(cn)

    for t_idx in range(NUM_TOKENS):
        var tid = tokens[t_idx]
        var row = embed_ptr + tid * HIDDEN

        for i in range(HIDDEN):
            work[i] = Float32(row[i])

        # RMSNorm: x / rms(x)
        var sum_sq = Float32(0)
        for i in range(HIDDEN):
            sum_sq += work[i] * work[i]
        var rms = sqrt[DType.float32, 1](sum_sq / Float32(HIDDEN))
        for i in range(HIDDEN):
            work[i] /= rms

        # Block-diagonal FWHT
        fwht_row[DType.float32, HD](work, HIDDEN)

        _ = t_idx

        for b in range(BLOCKS):
            var absmax = Float32(0)
            var block_sq = Float64(0)
            for d in range(HD):
                var v = work[b * HD + d]
                block_sq += Float64(v * v)
                if v < Float32(0):
                    v = -v
                if v > absmax:
                    absmax = v
                if v > cn_f32:
                    clipped_elements += 1
                total_elements += 1

            var block_norm = comptime_sqrt(block_sq)
            block_norm_sum[b] += block_norm
            if block_norm > block_norm_max[b]:
                block_norm_max[b] = block_norm

            var ratio = Float64(absmax) / cn
            ratio_sum += ratio
            total_blocks += 1
            if ratio > ratio_max:
                ratio_max = ratio
            if ratio < ratio_min:
                ratio_min = ratio

            if ratio < 0.5:
                hist[0] += 1
            elif ratio < 0.75:
                hist[1] += 1
            elif ratio < 1.0:
                hist[2] += 1
            elif ratio < 1.25:
                hist[3] += 1
            elif ratio < 1.5:
                hist[4] += 1
            else:
                hist[5] += 1

    work.free()

    var avg_ratio = ratio_sum / Float64(total_blocks)
    var clip_pct = Float64(clipped_elements) / Float64(total_elements) * Float64(100)

    print("ButterQuant fixed-scale validation (embedding layer, " + String(NUM_TOKENS) + " tokens)")
    print("C(" + String(HD) + ") = " + String(cn))
    print("")
    print("absmax / C(n) distribution across " + String(total_blocks) + " blocks:")
    print("  < 0.50 : " + String(hist[0]) + "   (severely underutilized)")
    print("  0.50-0.75: " + String(hist[1]))
    print("  0.75-1.00: " + String(hist[2]) + "   (ideal range)")
    print("  1.00-1.25: " + String(hist[3]) + "   (mild clipping)")
    print("  1.25-1.50: " + String(hist[4]) + "   (moderate clipping)")
    print("  > 1.50 : " + String(hist[5]) + "   (severe clipping)")
    print("")
    print("avg absmax/C(n) = " + String(avg_ratio))
    print("min absmax/C(n) = " + String(ratio_min))
    print("max absmax/C(n) = " + String(ratio_max))
    print("element clip rate = " + String(clip_pct) + "%")
    print("")

    # Block energy uniformity
    var ideal_norm = comptime_sqrt(Float64(HD))
    print("per-block energy (ideal norm = " + String(ideal_norm) + "):")
    for b in range(BLOCKS):
        var avg_norm = block_norm_sum[b] / Float64(NUM_TOKENS)
        print("  block " + String(b) + ": avg=" + String(avg_norm)
            + "  max=" + String(block_norm_max[b])
            + "  ratio=" + String(avg_norm / ideal_norm))

    _ = model
