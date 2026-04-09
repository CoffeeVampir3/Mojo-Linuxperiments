"""Within-block outlier analysis for FWHT'd gelu_tanh*up output.

Is the outlier the DC component (index 0) systematically, or random?
What fraction of the absmax does the second-largest element represent?
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from simd_math import exp_f32, sqrt
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block


@always_inline
def gelu_tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    var inner = Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)
    var sig = 1.0 / (1.0 + exp_f32[width](-2.0 * inner))
    return 0.5 * x * (1.0 + 2.0 * sig - 1.0)


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def analyze[cols: Int, block: Int, num_trials: Int]():
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block
    var rng = UInt64(0xFEEDCAFEDEAD1234)

    # Counters
    var dc_is_max_count = Int(0)
    var total_blocks_seen = Int(0)
    var sum_second_ratio = Float64(0)
    var sum_absmax_over_rms = Float64(0)
    # Track how often each index is the outlier
    var max_index_hist = alloc[Int32](block)
    for i in range(block):
        max_index_hist[i] = Int32(0)

    var work = alloc[Float32](cols)

    for trial in range(num_trials):
        var gate = alloc[Scalar[DType.bfloat16]](cols)
        var up = alloc[Scalar[DType.bfloat16]](cols)
        for i in range(cols):
            gate[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
            up[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

        var k = 0
        while k + width <= cols:
            var g = (gate + k).load[width=width]().cast[DType.float32]()
            var u = (up + k).load[width=width]().cast[DType.float32]()
            (work + k).store(gelu_tanh_f32[width](g) * u)
            k += width

        for b in range(num_blocks):
            fwht_block[block](work + b * block)

            # Find absmax and second-largest within this block
            var blk_ptr = work + b * block
            var max_val = Float32(0)
            var second_val = Float32(0)
            var max_idx = 0
            var sum_sq = Float64(0)
            for i in range(block):
                var a = blk_ptr[i].__abs__()
                sum_sq += Float64(blk_ptr[i]) * Float64(blk_ptr[i])
                if a > max_val:
                    second_val = max_val
                    max_val = a
                    max_idx = i
                elif a > second_val:
                    second_val = a

            max_index_hist[max_idx] += Int32(1)
            if max_idx == 0:
                dc_is_max_count += 1
            total_blocks_seen += 1
            if max_val > Float32(0):
                sum_second_ratio += Float64(second_val) / Float64(max_val)
            var blk_rms = Float64(sqrt[DType.float64, 1](sum_sq / Float64(block)))
            if blk_rms > Float64(0):
                sum_absmax_over_rms += Float64(max_val) / blk_rms

        gate.free()
        up.free()

    print("  total blocks analyzed: " + String(total_blocks_seen))
    print("  DC (index 0) is outlier: " + String(dc_is_max_count) + "/" + String(total_blocks_seen)
        + " (" + String(Float64(dc_is_max_count) / Float64(total_blocks_seen) * 100.0) + "%)")
    print("  avg second/max ratio:    " + String(sum_second_ratio / Float64(total_blocks_seen))
        + "  (1.0 = no outlier, 0.5 = outlier is 2x next)")
    print("  avg absmax/rms:          " + String(sum_absmax_over_rms / Float64(total_blocks_seen)))

    # Top-5 outlier indices
    print("  outlier index histogram (top 5):")
    for rank in range(5):
        var best_idx = 0
        var best_count = Int32(0)
        for i in range(block):
            if max_index_hist[i] > best_count:
                best_count = max_index_hist[i]
                best_idx = i
        print("    index " + String(best_idx) + ": " + String(best_count) + " times ("
            + String(Float64(Int(best_count)) / Float64(total_blocks_seen) * 100.0) + "%)")
        max_index_hist[best_idx] = Int32(0)

    max_index_hist.free()
    work.free()


def main():
    print("=== within-block outlier analysis ===")

    print("\ncols=704, block=64, 100 trials:")
    analyze[704, 64, 100]()

    print("\ncols=2112, block=64, 100 trials:")
    analyze[2112, 64, 100]()
