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
struct F32GemvArgs(Copyable, ImplicitlyCopyable):
    """Args for bf16 × f32 GEMV (router projection)."""
    var act_bf16: BF16Ptr
    var weight_f32: F32Ptr
    var dst_f32: F32Ptr
    var n_start: Int
    var n_count: Int

    def __init__(out self):
        self.act_bf16 = BF16Ptr()
        self.weight_f32 = F32Ptr()
        self.dst_f32 = F32Ptr()
        self.n_start = 0
        self.n_count = 0


@fieldwise_init
struct RouterTopkArgs(Copyable, ImplicitlyCopyable):
    """Args for MiniMax sigmoid router top-k."""
    var logits: F32Ptr
    var correction_bias: F32Ptr
    var result_ptr: U8Ptr

    def __init__(out self):
        self.logits = F32Ptr()
        self.correction_bias = F32Ptr()
        self.result_ptr = U8Ptr()


@fieldwise_init
struct NormPrepArgs(Copyable, ImplicitlyCopyable):
    """Args for full-vector Q/K norm preparation.

    Reads bf16 Q and K buffers, computes inv_rms for each, writes two
    f32 scalars to dst (dst[0] = inv_rms_q, dst[1] = inv_rms_k).
    """
    var q_ptr: BF16Ptr
    var k_ptr: BF16Ptr
    var dst: F32Ptr
    var eps: Float32

    def __init__(out self):
        self.q_ptr = BF16Ptr()
        self.k_ptr = BF16Ptr()
        self.dst = F32Ptr()
        self.eps = Float32(0)
