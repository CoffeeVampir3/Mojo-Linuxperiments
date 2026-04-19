"""Argument structs for MiniMax pool-dispatched workers.

Mirrors the pattern in experimental3/kernels/dispatch_args.mojo: each struct
is Copyable + ImplicitlyCopyable for InlineArray packing through BurstPool.
"""

from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr


@fieldwise_init
struct FusedW1W3SiluArgs(Copyable, ImplicitlyCopyable):
    """Args for fused gate(w1) + up(w3) + SiLU + FWHT + per-block i8.

    Separate w1/w3 weight pointers — MiniMax experts store gate and up
    projections as independent [intermediate, hidden] matrices, not the
    fused [2*intermediate, hidden] layout used by Gemma4.
    """
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var w1_packed: U8Ptr
    var w1_scale: F32Ptr
    var w1_colsum: F32Ptr
    var w3_packed: U8Ptr
    var w3_scale: F32Ptr
    var w3_colsum: F32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var n_count: Int
    var row_count: Int

    def __init__(out self):
        self.act_i8 = I8Ptr()
        self.act_scale = F32Ptr()
        self.w1_packed = U8Ptr()
        self.w1_scale = F32Ptr()
        self.w1_colsum = F32Ptr()
        self.w3_packed = U8Ptr()
        self.w3_scale = F32Ptr()
        self.w3_colsum = F32Ptr()
        self.qi_out = I8Ptr()
        self.blk_scale = F32Ptr()
        self.n_start = 0
        self.n_count = 0
        self.row_count = 0


@fieldwise_init
struct RmsNormDualOutputArgs(Copyable, ImplicitlyCopyable):
    """Args for dual-output RMSNorm: split-gamma i8 + full-gamma bf16.

    One RMS reduction, two outputs from the same norm:
      - qi_ptr/scale_ptr: sign(γ)√|γ| → FWHT → i8 (for MoE experts)
      - normed_bf16_ptr:  γ → bf16 (for router projection)
    """
    var src_ptr: BF16Ptr
    var split_gamma_ptr: BF16Ptr
    var full_gamma_ptr: BF16Ptr
    var qi_ptr: I8Ptr
    var work_ptr: F32Ptr
    var scale_ptr: F32Ptr
    var normed_bf16_ptr: BF16Ptr
    var eps: Float32
    var start_row: Int
    var end_row: Int

    def __init__(out self):
        self.src_ptr = BF16Ptr()
        self.split_gamma_ptr = BF16Ptr()
        self.full_gamma_ptr = BF16Ptr()
        self.qi_ptr = I8Ptr()
        self.work_ptr = F32Ptr()
        self.scale_ptr = F32Ptr()
        self.normed_bf16_ptr = BF16Ptr()
        self.eps = Float32(0)
        self.start_row = 0
        self.end_row = 0
