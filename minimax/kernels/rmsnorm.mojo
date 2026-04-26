from kernels.rmsnorm import rmsnorm_i8_and_bf16_row
from minimax.kernels.dispatch_args import RmsNormDualOutputArgs


def rmsnorm_dual_output_worker[cols: Int, block: Int](
    args: RmsNormDualOutputArgs,
):
    for m in range(args.start_row, args.end_row):
        rmsnorm_i8_and_bf16_row[cols, block](
            args.src_ptr + m * cols,
            args.split_gamma_ptr,
            args.full_gamma_ptr,
            args.qi_ptr + m * cols,
            args.scale_ptr + m,
            args.normed_bf16_ptr + m * cols,
            args.work_ptr,
            args.eps)
