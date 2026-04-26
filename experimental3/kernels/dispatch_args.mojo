"""Argument structs for every pool-dispatched worker.

Centralized home for the Args plain-data types that dispatchers in
dispatch_kernels.mojo fill and workers in the various kernel files consume.
Keeping them in one place makes the dispatcher/worker boundary explicit: the
kernel file owns compute, this file owns the arg schema, and dispatch_kernels
is the single producer.

Every struct here is Copyable + ImplicitlyCopyable so it can be placed into an
InlineArray and passed by value through BurstThreadPool.dispatch.
"""

from std.collections import InlineArray

from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from experimental_gemma.router import Gemma4TopKResult


# ============================================================================
# Shared sizing constants
# ============================================================================


comptime MAX_CP_RANKS = 8


# ============================================================================
# GEMM-shaped workers (gemm.mojo)
# ============================================================================


@fieldwise_init
struct WorkerConfig(Copyable, ImplicitlyCopyable, Defaultable):
    var act_ptr: I8Ptr
    var wpacked_ptr: I8Ptr
    var colsum_ptr: F32Ptr
    var weight_scale_ptr: F32Ptr
    var dst_ptr: BF16Ptr
    var act_scale_ptr: F32Ptr
    var start: Int
    var count: Int

    def __init__(out self):
        self.act_ptr = I8Ptr()
        self.wpacked_ptr = I8Ptr()
        self.colsum_ptr = F32Ptr()
        self.weight_scale_ptr = F32Ptr()
        self.dst_ptr = BF16Ptr()
        self.act_scale_ptr = F32Ptr()
        self.start = 0
        self.count = 0


@fieldwise_init
struct Int8GemvBlockedArgs(Copyable, ImplicitlyCopyable, Defaultable):
    var act: I8Ptr
    var wpacked: I8Ptr
    var blk_scale: F32Ptr
    var wscale: F32Ptr
    var blk_colsum: F32Ptr
    var dst: BF16Ptr
    var output_scale: Float32
    var n_out: Int
    var colsum_stride: Int
    var row_count: Int

    def __init__(out self):
        self.act = I8Ptr()
        self.wpacked = I8Ptr()
        self.blk_scale = F32Ptr()
        self.wscale = F32Ptr()
        self.blk_colsum = F32Ptr()
        self.dst = BF16Ptr()
        self.output_scale = Float32(0)
        self.n_out = 0
        self.colsum_stride = 0
        self.row_count = 0


@fieldwise_init
struct FusedGuGeluTanhArgs(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var wpacked: I8Ptr
    var wscale: F32Ptr
    var wcolsum: F32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var n_count: Int
    var row_count: Int

    def __init__(out self):
        self.act_i8 = I8Ptr()
        self.act_scale = F32Ptr()
        self.wpacked = I8Ptr()
        self.wscale = F32Ptr()
        self.wcolsum = F32Ptr()
        self.qi_out = I8Ptr()
        self.blk_scale = F32Ptr()
        self.n_start = 0
        self.n_count = 0
        self.row_count = 0


@fieldwise_init
struct LmHeadArgs(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var weight: I8Ptr
    var act_blk_scales: F32Ptr
    var w_blk_scales: F32Ptr
    var w_blk_colsums: F32Ptr
    var dst: BF16Ptr
    var n_start: Int
    var n_count: Int

    def __init__(out self):
        self.act = I8Ptr()
        self.weight = I8Ptr()
        self.act_blk_scales = F32Ptr()
        self.w_blk_scales = F32Ptr()
        self.w_blk_colsums = F32Ptr()
        self.dst = BF16Ptr()
        self.n_start = 0
        self.n_count = 0


# ============================================================================
# MoE router (moe.mojo)
# ============================================================================


@fieldwise_init
struct RouterTopkArgs[k: Int](Copyable, ImplicitlyCopyable):
    var logits: BF16Ptr
    var per_expert_scale: BF16Ptr
    var result_ptr: UnsafePointer[Gemma4TopKResult[Self.k], MutAnyOrigin]

    def __init__(out self):
        self.logits = BF16Ptr()
        self.per_expert_scale = BF16Ptr()
        self.result_ptr = UnsafePointer[Gemma4TopKResult[Self.k], MutAnyOrigin]()


# ============================================================================
# Sliding attention (sliding_attention.mojo)
# ============================================================================


@fieldwise_init
struct AttnGroupArgs(Copyable, ImplicitlyCopyable):
    """Shared args for both sliding and full attention group kernels.

    For full attention (K=V shared), set v_bf16_ptr = k_bf16_ptr.
    """
    var q_bf16_ptr: BF16Ptr
    var k_bf16_ptr: BF16Ptr
    var v_bf16_ptr: BF16Ptr
    var q_norm_ptr: BF16Ptr
    var k_norm_ptr: BF16Ptr
    var cos_ptr: F32Ptr
    var sin_ptr: F32Ptr
    var cache_base: U8Ptr
    var kv_head: Int
    var cache_pos: Int
    var context_len: Int
    var qi_out_ptr: I8Ptr
    var head_scale_ptr: F32Ptr
    var eps: Float32

    def __init__(out self):
        self.q_bf16_ptr = BF16Ptr()
        self.k_bf16_ptr = BF16Ptr()
        self.v_bf16_ptr = BF16Ptr()
        self.q_norm_ptr = BF16Ptr()
        self.k_norm_ptr = BF16Ptr()
        self.cos_ptr = F32Ptr()
        self.sin_ptr = F32Ptr()
        self.cache_base = U8Ptr()
        self.kv_head = 0
        self.cache_pos = 0
        self.context_len = 0
        self.qi_out_ptr = I8Ptr()
        self.head_scale_ptr = F32Ptr()
        self.eps = Float32(0)


# ============================================================================
# Context-parallel full attention (full_chunked_attention_fused.mojo)
# ============================================================================


@fieldwise_init
struct ChunkedAttnArgs(Copyable, ImplicitlyCopyable):
    """Per-worker arguments for chunked attention scoring."""
    var q_i8_base: I8Ptr
    var qi_biases_base: F32Ptr
    var q_scales_base: F32Ptr
    var cache_base: U8Ptr
    var kv_head: Int
    var start_pg: Int
    var end_pg: Int
    var partial_out: F32Ptr
    var context_len: Int

    def __init__(out self):
        self.q_i8_base = I8Ptr()
        self.qi_biases_base = F32Ptr()
        self.q_scales_base = F32Ptr()
        self.cache_base = U8Ptr()
        self.kv_head = 0
        self.start_pg = 0
        self.end_pg = 0
        self.partial_out = F32Ptr()
        self.context_len = 0


@fieldwise_init
struct CpAttnPrepArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_base: BF16Ptr
    var k_bf16_ptr: BF16Ptr
    var q_norm_ptr: BF16Ptr
    var k_norm_ptr: BF16Ptr
    var cos_ptr: F32Ptr
    var sin_ptr: F32Ptr
    var cache_base: U8Ptr
    var cache_pos: Int
    var kv_head: Int
    var eps: Float32
    var q_i8_out: I8Ptr
    var qi_biases_out: F32Ptr
    var q_scales_out: F32Ptr
    var write_kv: Int32

    def __init__(out self):
        self.q_bf16_base = BF16Ptr()
        self.k_bf16_ptr = BF16Ptr()
        self.q_norm_ptr = BF16Ptr()
        self.k_norm_ptr = BF16Ptr()
        self.cos_ptr = F32Ptr()
        self.sin_ptr = F32Ptr()
        self.cache_base = U8Ptr()
        self.cache_pos = 0
        self.kv_head = 0
        self.eps = Float32(0)
        self.q_i8_out = I8Ptr()
        self.qi_biases_out = F32Ptr()
        self.q_scales_out = F32Ptr()
        self.write_kv = Int32(0)


@fieldwise_init
struct MergeChunksArgs(Copyable, ImplicitlyCopyable):
    var partial_base: F32Ptr
    var num_chunks: Int
    var out_m: F32Ptr
    var out_l: F32Ptr
    var out_v: F32Ptr

    def __init__(out self):
        self.partial_base = F32Ptr()
        self.num_chunks = 0
        self.out_m = F32Ptr()
        self.out_l = F32Ptr()
        self.out_v = F32Ptr()


@fieldwise_init
struct CpGatherArgs(Copyable, ImplicitlyCopyable):
    var rank: Int
    var head_start: Int
    var head_count: Int
    var qi_out: I8Ptr
    var head_scales: F32Ptr
    var all_m: InlineArray[F32Ptr, MAX_CP_RANKS]
    var all_l: InlineArray[F32Ptr, MAX_CP_RANKS]
    var all_v: InlineArray[F32Ptr, MAX_CP_RANKS]

    def __init__(out self):
        self.rank = 0
        self.head_start = 0
        self.head_count = 0
        self.qi_out = I8Ptr()
        self.head_scales = F32Ptr()
        self.all_m = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())
        self.all_l = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())
        self.all_v = InlineArray[F32Ptr, MAX_CP_RANKS](fill=F32Ptr())


# ============================================================================
# RMSNorm family (rmsnorm.mojo)
# ============================================================================


@fieldwise_init
struct RmsNormFwhtQuantArgs(Copyable, ImplicitlyCopyable):
    """Unified args for all single-lane FWHT+quantize variants.

    gamma_ptr is null when has_gamma=False (never dereferenced).
    scale_ptr points to 1 scale per row (per-row) or cols/block per row (per-block).
    """
    var in_ptr: BF16Ptr
    var gamma_ptr: BF16Ptr
    var qi_ptr: I8Ptr
    var work_ptr: F32Ptr
    var scale_ptr: F32Ptr
    var eps: Float32
    var start_row: Int
    var end_row: Int

    def __init__(out self):
        self.in_ptr = BF16Ptr()
        self.gamma_ptr = BF16Ptr()
        self.qi_ptr = I8Ptr()
        self.work_ptr = F32Ptr()
        self.scale_ptr = F32Ptr()
        self.eps = 0.0
        self.start_row = 0
        self.end_row = 0


@fieldwise_init
struct RmsNormDualGammaFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: BF16Ptr
    var gamma_a_ptr: BF16Ptr
    var gamma_b_ptr: BF16Ptr
    var qi_a_ptr: I8Ptr
    var qi_b_ptr: I8Ptr
    var work_ptr: F32Ptr
    var scale_a_ptr: F32Ptr
    var scale_b_ptr: F32Ptr
    var eps: Float32
    var start_row: Int
    var end_row: Int

    def __init__(out self):
        self.in_ptr = BF16Ptr()
        self.gamma_a_ptr = BF16Ptr()
        self.gamma_b_ptr = BF16Ptr()
        self.qi_a_ptr = I8Ptr()
        self.qi_b_ptr = I8Ptr()
        self.work_ptr = F32Ptr()
        self.scale_a_ptr = F32Ptr()
        self.scale_b_ptr = F32Ptr()
        self.eps = 0.0
        self.start_row = 0
        self.end_row = 0


@fieldwise_init
struct RMSNormNoScaleArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float32

    def __init__(out self):
        self.input = BF16Ptr()
        self.output = BF16Ptr()
        self.start_row = 0
        self.end_row = 0
        self.eps = Float32(0)


@fieldwise_init
struct RMSNormPerHeadArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float32

    def __init__(out self):
        self.input = BF16Ptr()
        self.weight = BF16Ptr()
        self.output = BF16Ptr()
        self.start_row = 0
        self.end_row = 0
        self.eps = Float32(0)


@fieldwise_init
struct PostAttnNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: BF16Ptr
    var norm_w_ptr: BF16Ptr
    var x_main_ptr: BF16Ptr
    var eps: Float32

    def __init__(out self):
        self.src_ptr = BF16Ptr()
        self.norm_w_ptr = BF16Ptr()
        self.x_main_ptr = BF16Ptr()
        self.eps = Float32(0)


@fieldwise_init
struct ExpertSumArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: BF16Ptr
    var local_count: Int
    var dst_ptr: BF16Ptr

    def __init__(out self):
        self.expert_out_ptr = BF16Ptr()
        self.local_count = 0
        self.dst_ptr = BF16Ptr()


@fieldwise_init
struct DenseNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: BF16Ptr
    var norm_w_ptr: BF16Ptr
    var dst_ptr: BF16Ptr
    var eps: Float32

    def __init__(out self):
        self.src_ptr = BF16Ptr()
        self.norm_w_ptr = BF16Ptr()
        self.dst_ptr = BF16Ptr()
        self.eps = Float32(0)


@fieldwise_init
struct PostReduceArgs(Copyable, ImplicitlyCopyable):
    var moe_out_ptr: BF16Ptr
    var moe_norm_w_ptr: BF16Ptr
    var dense_normed_ptr: BF16Ptr
    var combine_norm_w_ptr: BF16Ptr
    var x_main_ptr: BF16Ptr
    var layer_scalar: Float32
    var eps: Float32

    def __init__(out self):
        self.moe_out_ptr = BF16Ptr()
        self.moe_norm_w_ptr = BF16Ptr()
        self.dense_normed_ptr = BF16Ptr()
        self.combine_norm_w_ptr = BF16Ptr()
        self.x_main_ptr = BF16Ptr()
        self.layer_scalar = Float32(0)
        self.eps = Float32(0)
