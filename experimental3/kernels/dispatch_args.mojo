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


# ============================================================================
# Shared sizing constants
# ============================================================================


comptime MAX_CP_RANKS = 8


# ============================================================================
# GEMM-shaped workers (gemm.mojo)
# ============================================================================


@fieldwise_init
struct WorkerConfig(Copyable, ImplicitlyCopyable):
    var act_ptr: Int
    var wpacked_ptr: Int
    var colsum_ptr: Int
    var weight_scale_ptr: Int
    var dst_ptr: Int
    var act_scale_ptr: Int
    var start_row: Int
    var row_count: Int


@fieldwise_init
struct Int8GemvBlockedArgs(Copyable, ImplicitlyCopyable):
    var act: I8Ptr
    var wpacked: U8Ptr
    var blk_scale: F32Ptr
    var wscale: F32Ptr
    var blk_colsum: F32Ptr
    var dst: BF16Ptr
    var output_scale: Float32


@fieldwise_init
struct FusedGuGeluTanhArgs(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var wpacked: U8Ptr
    var wscale: F32Ptr
    var wcolsum: F32Ptr
    var qi_out: I8Ptr
    var blk_scale: F32Ptr
    var n_start: Int
    var n_count: Int
    var row_count: Int


@fieldwise_init
struct LmHeadArgs(Copyable, ImplicitlyCopyable):
    var act: Int
    var weight: Int
    var act_blk_scales: Int
    var w_blk_scales: Int
    var w_blk_colsums: Int
    var dst: Int
    var n_start: Int
    var n_count: Int


# ============================================================================
# MoE router (moe.mojo)
# ============================================================================


@fieldwise_init
struct RouterTopkArgs(Copyable, ImplicitlyCopyable):
    var logits: BF16Ptr
    var per_expert_scale: BF16Ptr
    var result_ptr: Int


# ============================================================================
# Sliding attention (sliding_attention.mojo)
# ============================================================================


@fieldwise_init
struct AttnGroupArgs(Copyable, ImplicitlyCopyable):
    """Shared args for both sliding and full attention group kernels.

    For full attention (K=V shared), set v_bf16_ptr = k_bf16_ptr.
    """
    var q_bf16_ptr: Int
    var k_bf16_ptr: Int
    var v_bf16_ptr: Int
    var q_norm_ptr: Int
    var k_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var kv_head: Int
    var cache_pos: Int
    var context_len: Int
    var qi_out_ptr: Int
    var head_scale_ptr: Int
    var eps: Float32

    def __init__(out self):
        self.q_bf16_ptr = 0
        self.k_bf16_ptr = 0
        self.v_bf16_ptr = 0
        self.q_norm_ptr = 0
        self.k_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.cache_base = 0
        self.kv_head = 0
        self.cache_pos = 0
        self.context_len = 0
        self.qi_out_ptr = 0
        self.head_scale_ptr = 0
        self.eps = Float32(0)


# ============================================================================
# Context-parallel full attention (full_chunked_attention_fused.mojo)
# ============================================================================


@fieldwise_init
struct ChunkedAttnArgs(Copyable, ImplicitlyCopyable):
    """Per-worker arguments for chunked attention scoring."""
    var q_i8_base: Int
    var qi_biases_base: Int
    var q_scales_base: Int
    var cache_base: Int
    var kv_head: Int
    var start_pg: Int
    var end_pg: Int
    var partial_out: Int
    var context_len: Int

    def __init__(out self):
        self.q_i8_base = 0
        self.qi_biases_base = 0
        self.q_scales_base = 0
        self.cache_base = 0
        self.kv_head = 0
        self.start_pg = 0
        self.end_pg = 0
        self.partial_out = 0
        self.context_len = 0


@fieldwise_init
struct CpAttnPrepArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_base: Int
    var k_bf16_ptr: Int
    var q_norm_ptr: Int
    var k_norm_ptr: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var cache_pos: Int
    var kv_head: Int
    var eps: Float32
    var q_i8_out: Int
    var qi_biases_out: Int
    var q_scales_out: Int
    var write_kv: Int32

    def __init__(out self):
        self.q_bf16_base = 0
        self.k_bf16_ptr = 0
        self.q_norm_ptr = 0
        self.k_norm_ptr = 0
        self.cos_ptr = 0
        self.sin_ptr = 0
        self.cache_base = 0
        self.cache_pos = 0
        self.kv_head = 0
        self.eps = Float32(0)
        self.q_i8_out = 0
        self.qi_biases_out = 0
        self.q_scales_out = 0
        self.write_kv = Int32(0)


@fieldwise_init
struct MergeChunksArgs(Copyable, ImplicitlyCopyable):
    var partial_base: Int
    var num_chunks: Int
    var out_m: Int
    var out_l: Int
    var out_v: Int

    def __init__(out self):
        self.partial_base = 0
        self.num_chunks = 0
        self.out_m = 0
        self.out_l = 0
        self.out_v = 0


@fieldwise_init
struct CpGatherArgs(Copyable, ImplicitlyCopyable):
    var rank: Int
    var head_start: Int
    var head_count: Int
    var qi_out: Int
    var head_scales: Int
    var all_m: InlineArray[Int, MAX_CP_RANKS]
    var all_l: InlineArray[Int, MAX_CP_RANKS]
    var all_v: InlineArray[Int, MAX_CP_RANKS]

    def __init__(out self):
        self.rank = 0
        self.head_start = 0
        self.head_count = 0
        self.qi_out = 0
        self.head_scales = 0
        self.all_m = InlineArray[Int, MAX_CP_RANKS](fill=0)
        self.all_l = InlineArray[Int, MAX_CP_RANKS](fill=0)
        self.all_v = InlineArray[Int, MAX_CP_RANKS](fill=0)


# ============================================================================
# RMSNorm family (rmsnorm.mojo)
# ============================================================================


@fieldwise_init
struct RmsNormFwhtQuantArgs(Copyable, ImplicitlyCopyable):
    """Unified args for all single-lane FWHT+quantize variants.

    gamma_ptr is 0 when has_gamma=False (never dereferenced).
    scale_ptr points to 1 scale per row (per-row) or cols/block per row (per-block).
    """
    var in_ptr: Int
    var gamma_ptr: Int
    var qi_ptr: Int
    var work_ptr: Int
    var scale_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int

    def __init__(out self):
        self.in_ptr = 0
        self.gamma_ptr = 0
        self.qi_ptr = 0
        self.work_ptr = 0
        self.scale_ptr = 0
        self.eps = 0.0
        self.start_row = 0
        self.end_row = 0


@fieldwise_init
struct RmsNormDualGammaFwhtArgs(Copyable, ImplicitlyCopyable):
    var in_ptr: Int
    var gamma_a_ptr: Int
    var gamma_b_ptr: Int
    var qi_a_ptr: Int
    var qi_b_ptr: Int
    var work_a_ptr: Int
    var work_b_ptr: Int
    var scale_a_ptr: Int
    var scale_b_ptr: Int
    var eps: Float32
    var start_row: Int
    var end_row: Int

    def __init__(out self):
        self.in_ptr = 0
        self.gamma_a_ptr = 0
        self.gamma_b_ptr = 0
        self.qi_a_ptr = 0
        self.qi_b_ptr = 0
        self.work_a_ptr = 0
        self.work_b_ptr = 0
        self.scale_a_ptr = 0
        self.scale_b_ptr = 0
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


@fieldwise_init
struct RMSNormPerHeadArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float32


@fieldwise_init
struct PostAttnNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: Int
    var norm_w_ptr: Int
    var x_main_ptr: Int
    var eps: Float32


@fieldwise_init
struct ExpertSumArgs(Copyable, ImplicitlyCopyable):
    var expert_out_ptr: Int
    var local_count: Int
    var dst_ptr: Int


@fieldwise_init
struct DenseNormArgs(Copyable, ImplicitlyCopyable):
    var src_ptr: Int
    var norm_w_ptr: Int
    var dst_ptr: Int
    var eps: Float32


@fieldwise_init
struct PostReduceArgs(Copyable, ImplicitlyCopyable):
    var moe_out_ptr: Int
    var moe_norm_w_ptr: Int
    var dense_normed_ptr: Int
    var combine_norm_w_ptr: Int
    var x_main_ptr: Int
    var layer_scalar: Float32
    var eps: Float32
