from std.sys.info import simd_width_of

from experimental3.moe import moe_combine
from std.collections import InlineArray
from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from simd_math.matrixops import tree_merge_accs, port_unroll_for
from experimental3.kernels.dispatch_args import (
    RmsNormFwhtQuantArgs, RmsNormDualGammaFwhtArgs,
    RMSNormNoScaleArgs, RMSNormPerHeadArgs,
    PostAttnNormArgs, ExpertSumArgs, DenseNormArgs, PostReduceArgs,
)
from kernels.rmsnorm import (
    Bf16Spec, I8Spec, rmsnorm_n_bf16_row, rmsnorm_n_i8_row,
)


@always_inline
def accumulate_expert_outputs[hidden: Int, max_local: Int](
    expert_buf: BF16Ptr,
    local_count: Int,
    dst: BF16Ptr,
):
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = port_unroll_for[max_local]()
    debug_assert(hidden % width == 0, "hidden must be f32-simd-aligned")
    debug_assert(local_count <= max_local, "local_count exceeds comptime max_local")

    for i in range(0, hidden, width):
        var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
            fill=SIMD[DType.float32, width](0))
        var e = 0
        while e + port_unroll <= local_count:
            comptime for u in range(port_unroll):
                accs[u] += (expert_buf + (e + u) * hidden + i).load[width=width]().cast[DType.float32]()
            e += port_unroll
        while e < local_count:
            accs[0] += (expert_buf + e * hidden + i).load[width=width]().cast[DType.float32]()
            e += 1
        (dst + i).store(tree_merge_accs(accs).cast[DType.bfloat16]())


def rmsnorm_fwht_quant_worker[cols: Int, block: Int,
    has_gamma: Bool, per_block: Bool](
    args: RmsNormFwhtQuantArgs,
):
    var inp = args.in_ptr
    var gamma = args.gamma_ptr
    var qi = args.qi_ptr
    var work = args.work_ptr
    var scales = args.scale_ptr

    comptime scale_stride = (cols // block) if per_block else 1
    for m in range(args.start_row, args.end_row):
        var gammas: InlineArray[BF16Ptr, 1] = [gamma]
        var dsts: InlineArray[I8Ptr, 1] = [qi + m * cols]
        var scales_arr: InlineArray[F32Ptr, 1] = [scales + m * scale_stride]
        rmsnorm_n_i8_row[cols, block, 1,
            I8Spec(has_gamma=has_gamma, fwht=True, per_block=per_block),
        ](inp + m * cols, work, args.eps, gammas, dsts, scales_arr)


def rmsnorm_dual_gamma_fwht_quant_worker[cols: Int, block: Int](
    args: RmsNormDualGammaFwhtArgs,
):
    var inp = args.in_ptr
    var gamma_a = args.gamma_a_ptr
    var gamma_b = args.gamma_b_ptr
    var qi_a = args.qi_a_ptr
    var qi_b = args.qi_b_ptr
    var work = args.work_ptr
    var scale_a = args.scale_a_ptr
    var scale_b = args.scale_b_ptr

    for m in range(args.start_row, args.end_row):
        var gammas: InlineArray[BF16Ptr, 2] = [gamma_a, gamma_b]
        var dsts: InlineArray[I8Ptr, 2] = [qi_a + m * cols, qi_b + m * cols]
        var scales: InlineArray[F32Ptr, 2] = [scale_a + m, scale_b + m]
        rmsnorm_n_i8_row[cols, block, 2,
            I8Spec(has_gamma=True, fwht=True, per_block=False),
            I8Spec(has_gamma=True, fwht=True, per_block=False),
        ](inp + m * cols, work, args.eps, gammas, dsts, scales)


def rmsnorm_no_scale_kernel[cols: Int](args: RMSNormNoScaleArgs):
    for row in range(args.start_row, args.end_row):
        var gammas: InlineArray[BF16Ptr, 1] = [BF16Ptr(unsafe_from_address=0)]
        var dsts: InlineArray[BF16Ptr, 1] = [args.output + row * cols]
        rmsnorm_n_bf16_row[cols, 1,
            Bf16Spec(has_gamma=False, has_residual=False),
        ](args.input + row * cols, args.eps, gammas, dsts)


def rmsnorm_per_head_kernel[head_dim: Int, num_heads: Int](args: RMSNormPerHeadArgs):
    comptime row_stride = num_heads * head_dim
    for row in range(args.start_row, args.end_row):
        var row_in = args.input + row * row_stride
        var row_out = args.output + row * row_stride
        for h in range(num_heads):
            var gammas: InlineArray[BF16Ptr, 1] = [args.weight]
            var dsts: InlineArray[BF16Ptr, 1] = [row_out + h * head_dim]
            rmsnorm_n_bf16_row[head_dim, 1,
                Bf16Spec(has_gamma=True, has_residual=False),
            ](row_in + h * head_dim, args.eps, gammas, dsts)


def post_attn_norm_kernel[hidden: Int](args: PostAttnNormArgs):
    var gammas: InlineArray[BF16Ptr, 1] = [args.norm_w_ptr]
    var dsts: InlineArray[BF16Ptr, 1] = [args.x_main_ptr]
    rmsnorm_n_bf16_row[hidden, 1,
        Bf16Spec(has_gamma=True, has_residual=True),
    ](args.src_ptr, args.eps, gammas, dsts)


@fieldwise_init
struct PreReduceArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: BF16Ptr
    var local_count: Int
    var dst_ptr: BF16Ptr
    var dense_ptr: BF16Ptr
    var norm_w_ptr: BF16Ptr
    var normed_ptr: BF16Ptr
    var eps: Float32

    def __init__(out self):
        self.expert_out_ptr = BF16Ptr()
        self.local_count = 0
        self.dst_ptr = BF16Ptr()
        self.dense_ptr = BF16Ptr()
        self.norm_w_ptr = BF16Ptr()
        self.normed_ptr = BF16Ptr()
        self.eps = Float32(0)


def pre_reduce_kernel[hidden: Int, max_local: Int](args: PreReduceArgs):
    var expert_buf = args.expert_out_ptr
    var dst = args.dst_ptr
    accumulate_expert_outputs[hidden, max_local](expert_buf, args.local_count, dst)
    var gammas: InlineArray[BF16Ptr, 1] = [args.norm_w_ptr]
    var dsts: InlineArray[BF16Ptr, 1] = [args.normed_ptr]
    rmsnorm_n_bf16_row[hidden, 1,
        Bf16Spec(has_gamma=True, has_residual=False),
    ](args.dense_ptr, args.eps, gammas, dsts)


def expert_sum_kernel[hidden: Int, max_local: Int](args: ExpertSumArgs):
    var expert_buf = args.expert_out_ptr
    var dst = args.dst_ptr
    accumulate_expert_outputs[hidden, max_local](expert_buf, args.local_count, dst)


def dense_norm_kernel[hidden: Int](args: DenseNormArgs):
    var gammas: InlineArray[BF16Ptr, 1] = [args.norm_w_ptr]
    var dsts: InlineArray[BF16Ptr, 1] = [args.dst_ptr]
    rmsnorm_n_bf16_row[hidden, 1,
        Bf16Spec(has_gamma=True, has_residual=False),
    ](args.src_ptr, args.eps, gammas, dsts)


def post_reduce_kernel[hidden: Int](args: PostReduceArgs):
    moe_combine[hidden](
        args.moe_out_ptr,
        args.moe_norm_w_ptr,
        args.dense_normed_ptr,
        args.combine_norm_w_ptr,
        args.x_main_ptr,
        args.layer_scalar, args.eps)
