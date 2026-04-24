from std.collections import InlineArray

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
    var w1_packed: I8Ptr
    var w1_scale: F32Ptr
    var w1_colsum: F32Ptr
    var w3_packed: I8Ptr
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
        self.w1_packed = I8Ptr()
        self.w1_scale = F32Ptr()
        self.w1_colsum = F32Ptr()
        self.w3_packed = I8Ptr()
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


@fieldwise_init
struct AttnGroupArgs(Copyable, ImplicitlyCopyable):
    """Args for MiniMax head-local attention group kernel.

    One KV group: write K/V to cache, prep each Q head, score against
    full local cache, quantize output. Caller pre-offsets pointers to
    this KV group's slice.
    """
    var q_bf16_base: BF16Ptr
    var k_bf16_ptr: BF16Ptr
    var v_bf16_ptr: BF16Ptr
    var q_norm_ptr: BF16Ptr
    var k_norm_ptr: BF16Ptr
    var cos_ptr: F32Ptr
    var sin_ptr: F32Ptr
    var inv_rms_q: Float32
    var inv_rms_k: Float32
    var cache_base: U8Ptr
    var cache_pos: Int
    var kv_head: Int
    var context_len: Int
    var qi_out: I8Ptr
    var head_scale_ptr: F32Ptr

    def __init__(out self):
        self.q_bf16_base = BF16Ptr()
        self.k_bf16_ptr = BF16Ptr()
        self.v_bf16_ptr = BF16Ptr()
        self.q_norm_ptr = BF16Ptr()
        self.k_norm_ptr = BF16Ptr()
        self.cos_ptr = F32Ptr()
        self.sin_ptr = F32Ptr()
        self.inv_rms_q = Float32(0)
        self.inv_rms_k = Float32(0)
        self.cache_base = U8Ptr()
        self.cache_pos = 0
        self.kv_head = 0
        self.context_len = 0
        self.qi_out = I8Ptr()
        self.head_scale_ptr = F32Ptr()


@fieldwise_init
struct KVWriteBatchArgs(Copyable, ImplicitlyCopyable):
    var k_bf16_base: BF16Ptr
    var v_bf16_base: BF16Ptr
    var qkv_row_stride: Int
    var k_norm_ptr: BF16Ptr
    var cos_base: F32Ptr
    var sin_base: F32Ptr
    var rope_row_elems: Int
    var inv_rms_k_arr: F32Ptr
    var cache_base: U8Ptr
    var start_pos: Int
    var pos_count: Int
    var kv_head: Int

    def __init__(out self):
        self.k_bf16_base = BF16Ptr()
        self.v_bf16_base = BF16Ptr()
        self.qkv_row_stride = 0
        self.k_norm_ptr = BF16Ptr()
        self.cos_base = F32Ptr()
        self.sin_base = F32Ptr()
        self.rope_row_elems = 0
        self.inv_rms_k_arr = F32Ptr()
        self.cache_base = U8Ptr()
        self.start_pos = 0
        self.pos_count = 0
        self.kv_head = 0


@fieldwise_init
struct QPrepBatchArgs(Copyable, ImplicitlyCopyable):
    var q_bf16_base: BF16Ptr
    var qkv_row_stride: Int
    var q_norm_ptr: BF16Ptr
    var cos_base: F32Ptr
    var sin_base: F32Ptr
    var rope_row_elems: Int
    var inv_rms_q_arr: F32Ptr
    var qi_out: I8Ptr
    var qi_out_head_stride: Int
    var qi_biases_out: F32Ptr
    var qi_biases_head_stride: Int
    var q_scales_out: F32Ptr
    var q_scales_head_stride: Int
    var start_pos: Int
    var pos_count: Int
    var kv_head: Int

    def __init__(out self):
        self.q_bf16_base = BF16Ptr()
        self.qkv_row_stride = 0
        self.q_norm_ptr = BF16Ptr()
        self.cos_base = F32Ptr()
        self.sin_base = F32Ptr()
        self.rope_row_elems = 0
        self.inv_rms_q_arr = F32Ptr()
        self.qi_out = I8Ptr()
        self.qi_out_head_stride = 0
        self.qi_biases_out = F32Ptr()
        self.qi_biases_head_stride = 0
        self.q_scales_out = F32Ptr()
        self.q_scales_head_stride = 0
        self.start_pos = 0
        self.pos_count = 0
        self.kv_head = 0


@fieldwise_init
struct PrefillAttnArgs(Copyable, ImplicitlyCopyable):
    var q_i8: I8Ptr
    var qi_biases: F32Ptr
    var q_factors: F32Ptr
    var cache_base: U8Ptr
    var kv_head: Int
    var q_start: Int
    var q_count: Int
    var context_len: Int
    var qi_out: I8Ptr
    var qi_out_row_stride: Int
    var head_sc_out: F32Ptr
    var head_sc_row_stride: Int
    var head_col_offset: Int
    var pos_count: Int

    def __init__(out self):
        self.q_i8 = I8Ptr()
        self.qi_biases = F32Ptr()
        self.q_factors = F32Ptr()
        self.cache_base = U8Ptr()
        self.kv_head = 0
        self.q_start = 0
        self.q_count = 0
        self.context_len = 0
        self.qi_out = I8Ptr()
        self.qi_out_row_stride = 0
        self.head_sc_out = F32Ptr()
        self.head_sc_row_stride = 0
        self.head_col_offset = 0
        self.pos_count = 0


@fieldwise_init
struct MergeQuantArgs(Copyable, ImplicitlyCopyable):
    var partial_base: F32Ptr
    var num_chunks: Int
    var qi_out: I8Ptr
    var head_scale_ptr: F32Ptr

    def __init__(out self):
        self.partial_base = F32Ptr()
        self.num_chunks = 0
        self.qi_out = I8Ptr()
        self.head_scale_ptr = F32Ptr()


@fieldwise_init
struct RouterCandidate(Copyable, ImplicitlyCopyable):
    """One (expert_id, score, raw) entry in a worker-local top-K buffer.

    score = raw + correction_bias (used for selection).
    raw   = sigmoid(dot) pre-bias (used for renormalization).
    """
    var eid: Int32
    var score: Float32
    var raw: Float32

    def __init__(out self):
        self.eid = Int32(-1)
        self.score = Float32(-1e30)
        self.raw = Float32(0)


@fieldwise_init
struct TopKResult[k: Int](Copyable, ImplicitlyCopyable, Movable):
    var indices: InlineArray[Int, Self.k]
    var weights: InlineArray[Float32, Self.k]


@fieldwise_init
struct RouterFusedArgs(Copyable, ImplicitlyCopyable):
    """Args for fused router worker: f32 GEMV + sigmoid + local top-K.

    Each worker owns rows [n_start, n_start + n_count) and writes K
    candidates to candidates[0..K) in descending score order.
    """
    var act_bf16: BF16Ptr
    var weight_f32: F32Ptr
    var bias_f32: F32Ptr
    var candidates: UnsafePointer[RouterCandidate, MutAnyOrigin]
    var eid_base: Int
    var n_start: Int
    var n_count: Int

    def __init__(out self):
        self.act_bf16 = BF16Ptr()
        self.weight_f32 = F32Ptr()
        self.bias_f32 = F32Ptr()
        self.candidates = UnsafePointer[RouterCandidate, MutAnyOrigin]()
        self.eid_base = 0
        self.n_start = 0
        self.n_count = 0
