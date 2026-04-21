from std.sys.info import simd_width_of
from std.collections import InlineArray

from experimental3.kernels.gemv import gemv_row
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import F32Ptr, BF16Ptr
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs
from minimax.kernels.activations import silu_mul
from minimax.kernels.dispatch_args import FusedW1W3SiluArgs


def fused_w1_w3_silu_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    """N-tiled gate(w1) + up(w3) GEMV -> SiLU -> FWHT -> per-block i8.

    Processes row_count activation rows, each over N-range
    [n_start, n_start + n_count). Tiles in fwht_blk-sized chunks:
      w1 GEMV[fwht_blk, K] + w3 GEMV[fwht_blk, K] -> silu(gate)*up
      -> FWHT -> absmax quantize -> i8.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "fused_w1_w3_silu: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "fused_w1_w3_silu: K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.w1_packed + n_off * K,
                dequant,
                args.w1_scale + n_off,
                args.w1_colsum + n_off,
                gate)

            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()
            gemv_row[fwht_blk, K, DType.float32](
                act_i8,
                args.w3_packed + n_off * K,
                dequant,
                args.w3_scale + n_off,
                args.w3_colsum + n_off,
                up)

            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width

            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)

            local_n += fwht_blk


# ============================================================================
# f32 GEMV — bf16 activation × f32 weight → f32 output (router projection)
# ============================================================================


@always_inline
def f32_gemv_row[K: Int](
    act: BF16Ptr, weight_row: F32Ptr,
) -> Float32:
    """Dot product: bf16[K] · f32[K] → f32 scalar."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, K]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var a = (act + off).load[width=width]().cast[DType.float32]()
        var w = (weight_row + off).load[width=width]()
        accs[i] = a * w
    var k = step
    while k + step <= K:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var a = (act + off).load[width=width]().cast[DType.float32]()
            var w = (weight_row + off).load[width=width]()
            accs[i] = a.fma(w, accs[i])
        k += step
    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width]()
        accs[0] = a.fma(w, accs[0])
        k += width
    return tree_reduce_accs(accs)


