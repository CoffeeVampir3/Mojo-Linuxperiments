"""DC scale sweep: measure NRMSE vs DC_SCALE factor.

For each DC_SCALE in [0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0]:
  - Apply FWHT, scale element 0 of each block by DC_SCALE
  - Per-block absmax quantize
  - Dequant (without DC correction — measuring raw quantization quality)
  - Inverse FWHT, compute NRMSE vs reference
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from simd_math import exp_f32, sqrt, roundeven
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8


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


def scalar_fwht_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var stride = 1
    while stride < n:
        var i = 0
        while i < n:
            for j in range(stride):
                var a = buf[i + j]
                var b = buf[i + j + stride]
                buf[i + j] = a + b
                buf[i + j + stride] = a - b
            i += stride * 2
        stride *= 2
    var sc = 1.0 / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] *= sc


def sweep[cols: Int, block: Int, num_trials: Int]():
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = cols // block

    # DC_SCALE values to test
    var scales = InlineArray[Float32, 8](fill=Float32(0))
    scales[0] = 0.40
    scales[1] = 0.50
    scales[2] = 0.55
    scales[3] = 0.60
    scales[4] = 0.65
    scales[5] = 0.70
    scales[6] = 0.80
    scales[7] = 1.00  # baseline (no correction)
    comptime num_scales = 8

    var sum_nrmse = InlineArray[Float64, 8](fill=Float64(0))
    var sum_cosine = InlineArray[Float64, 8](fill=Float64(0))

    var work = alloc[Float32](cols)
    var work_dc = alloc[Float32](cols)
    var qi = alloc[Scalar[DType.int8]](cols)
    var blk_scales = alloc[Float32](num_blocks)
    var recovered = alloc[Float64](cols)

    for trial in range(num_trials):
        var rng = UInt64(0xDEADBEEFCAFE1234 + UInt64(trial) * 0x100000001)

        var gate = alloc[Scalar[DType.bfloat16]](cols)
        var up_buf = alloc[Scalar[DType.bfloat16]](cols)
        for i in range(cols):
            gate[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))
            up_buf[i] = Scalar[DType.bfloat16](Float32(xorshift64(rng)))

        # f64 reference
        var expected = alloc[Float64](cols)
        for i in range(cols):
            var g = Float64(gate[i])
            var u = Float64(up_buf[i])
            # Scalar gelu_tanh
            var inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
            var e = Float64(exp_f32[1](Float32(-2.0 * inner)))
            var t = (1.0 - e) / (1.0 + e)
            expected[i] = 0.5 * g * (1.0 + t) * u

        # Compute gelu_tanh*up in f32
        var k = 0
        while k + width <= cols:
            var g = (gate + k).load[width=width]().cast[DType.float32]()
            var u = (up_buf + k).load[width=width]().cast[DType.float32]()
            (work + k).store(gelu_tanh_f32[width](g) * u)
            k += width

        for si in range(num_scales):
            var dc_scale = scales[si]
            var dc_inv = 1.0 / dc_scale

            # Copy work, apply FWHT + DC scaling
            for i in range(cols):
                work_dc[i] = work[i]
            for b in range(num_blocks):
                fwht_block[block](work_dc + b * block)
                work_dc[b * block] *= dc_scale  # scale DC down

            # Per-block quantize
            for b in range(num_blocks):
                blk_scales[b] = absmax_quantize_i8[block](work_dc + b * block, qi + b * block)

            # Dequant with DC correction + inverse FWHT
            for b in range(num_blocks):
                var dq = Float64(blk_scales[b]) / 127.0
                for j in range(block):
                    recovered[b * block + j] = Float64(Int64(qi[b * block + j])) * dq
                # Undo DC scaling on element 0
                recovered[b * block] *= Float64(dc_inv)
            for b in range(num_blocks):
                scalar_fwht_f64(recovered + b * block, block)

            # NRMSE and cosine
            var dot = Float64(0)
            var ne = Float64(0)
            var nr = Float64(0)
            var se = Float64(0)
            for i in range(cols):
                dot += expected[i] * recovered[i]
                ne += expected[i] * expected[i]
                nr += recovered[i] * recovered[i]
                se += (expected[i] - recovered[i]) * (expected[i] - recovered[i])
            var cosine = dot / (Float64(sqrt[DType.float64, 1](ne)) * Float64(sqrt[DType.float64, 1](nr)))
            var nrmse = Float64(sqrt[DType.float64, 1](se / ne))
            sum_nrmse[si] += nrmse
            sum_cosine[si] += cosine

        gate.free()
        up_buf.free()
        expected.free()

    # Report
    print("  DC_SCALE | avg NRMSE     | avg cosine")
    print("  ---------+--------------+----------------")
    for si in range(num_scales):
        var avg_nrmse = sum_nrmse[si] / Float64(num_trials)
        var avg_cos = sum_cosine[si] / Float64(num_trials)
        print("  " + String(scales[si]) + "     | " + String(avg_nrmse) + " | " + String(avg_cos))

    work.free()
    work_dc.free()
    qi.free()
    blk_scales.free()
    recovered.free()


def main():
    print("=== DC scale sweep ===")

    print("\ncols=704, block=64, 50 trials:")
    sweep[704, 64, 50]()

    print("\ncols=2112, block=64, 50 trials:")
    sweep[2112, 64, 50]()
