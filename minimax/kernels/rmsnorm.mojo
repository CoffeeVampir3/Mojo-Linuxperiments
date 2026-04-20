"""MiniMax RMSNorm and residual kernels.

Dual-output norm (post-attention phase 7):
  Path A (MoE):    x/rms · split_gamma → FWHT → per-row i8
  Path B (router): x/rms · full_gamma  → bf16

Residual add (post-FFN phase 11):
  x_main += moe_out. No norm, no dense path, no layer scalar.
"""

from std.sys.info import simd_width_of

from experimental3.common_math import (
    F32Ptr, BF16Ptr, I8Ptr,
    inv_rms_from_sum_sq,
)
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.rmsnorm import fwht_rotate, emit_quant
from minimax.kernels.dispatch_args import RmsNormDualOutputArgs


@always_inline
def rmsnorm_dual_output_row[cols: Int, block: Int](
    src: BF16Ptr,
    split_gamma: BF16Ptr,
    full_gamma: BF16Ptr,
    qi: I8Ptr,
    work: F32Ptr,
    scale: F32Ptr,
    normed_bf16: BF16Ptr,
    eps: Float32,
):
    """Single-row dual-output RMSNorm.

    Pass 1: load bf16 x → f32 work, accumulate sum(x²).
    Pass 2: read work, write split-gamma to work + full-gamma to bf16.
    Then FWHT + per-row quantize on work → i8.
    """
    comptime width = simd_width_of[DType.float32]()

    # Pass 1: load x, store raw f32 to work, reduce sum_sq
    var vsum = SIMD[DType.float32, width](0)
    var k = 0
    while k < cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        vsum = x.fma(x, vsum)
        (work + k).store(x)
        k += width

    var inv = inv_rms_from_sum_sq(vsum.reduce_add(), cols, eps)
    var vinv = SIMD[DType.float32, width](inv)

    # Pass 2: apply inv_rms with both gammas from the L1-hot work buffer
    k = 0
    while k < cols:
        var x = (work + k).load[width=width]()
        var normed = x * vinv
        var sg = (split_gamma + k).load[width=width]().cast[DType.float32]()
        var fg = (full_gamma + k).load[width=width]().cast[DType.float32]()
        (work + k).store(normed * sg)
        (normed_bf16 + k).store((normed * fg).cast[DType.bfloat16]())
        k += width

    fwht_rotate[cols, block](work)
    emit_quant[cols, block, False](work, qi, scale)


def rmsnorm_dual_output_worker[cols: Int, block: Int](
    args: RmsNormDualOutputArgs,
):
    for m in range(args.start_row, args.end_row):
        rmsnorm_dual_output_row[cols, block](
            args.src_ptr + m * cols,
            args.split_gamma_ptr,
            args.full_gamma_ptr,
            args.qi_ptr + m * cols,
            args.work_ptr,
            args.scale_ptr + m,
            args.normed_bf16_ptr + m * cols,
            args.eps)
