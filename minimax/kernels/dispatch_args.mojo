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
struct PrefillAttnTileArgs(Copyable, ImplicitlyCopyable):
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
struct PrefillAttnArgs(Copyable, ImplicitlyCopyable):
    var q_i8_base: I8Ptr
    var qi_biases_base: F32Ptr
    var q_factors_base: F32Ptr
    var cache_base: U8Ptr
    var start_pos: Int
    var seq_len: Int
    var qi_out_base: I8Ptr
    var qi_out_row_stride: Int
    var head_sc_out_base: F32Ptr
    var head_sc_row_stride: Int
    var worker_id: Int
    var num_workers: Int

    def __init__(out self):
        self.q_i8_base = I8Ptr()
        self.qi_biases_base = F32Ptr()
        self.q_factors_base = F32Ptr()
        self.cache_base = U8Ptr()
        self.start_pos = 0
        self.seq_len = 0
        self.qi_out_base = I8Ptr()
        self.qi_out_row_stride = 0
        self.head_sc_out_base = F32Ptr()
        self.head_sc_row_stride = 0
        self.worker_id = 0
        self.num_workers = 0


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
struct ChunkedScoreMultiArgs(Copyable, ImplicitlyCopyable):
    var q_i8_base: I8Ptr
    var qi_biases_base: F32Ptr
    var q_scales_base: F32Ptr
    var cache_base: U8Ptr
    var partial_out_base: F32Ptr
    var kv_start: Int
    var kv_count: Int
    var context_len: Int
    var num_chunks: Int
    var blocks_per_chunk: Int
    var extra_blocks: Int
    var worker_id: Int
    var num_workers: Int

    def __init__(out self):
        self.q_i8_base = I8Ptr()
        self.qi_biases_base = F32Ptr()
        self.q_scales_base = F32Ptr()
        self.cache_base = U8Ptr()
        self.partial_out_base = F32Ptr()
        self.kv_start = 0
        self.kv_count = 0
        self.context_len = 0
        self.num_chunks = 0
        self.blocks_per_chunk = 0
        self.extra_blocks = 0
        self.worker_id = 0
        self.num_workers = 0


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
struct SparseRoute(Copyable, ImplicitlyCopyable):
    """One final routed token inside a rank-local expert bucket.

    Phase1 consumes routes through expert bucket ranges:
      routes[offsets[expert] : offsets[expert + 1]]

    Phase2 consumes the same expert buckets and scatters weighted rows back to
    the owning token. The bucket range itself identifies the local expert.
    """
    var token: Int32
    var weight: Float32

    def __init__(out self):
        self.token = Int32(-1)
        self.weight = Float32(0)


@fieldwise_init
struct SparseMoePhase1Args(Copyable, ImplicitlyCopyable):
    """Persistent sparse MoE phase1 worker contract.

    Workers stride over local experts, consume bucketed routes, and write
    route-major intermediate activations:
      expert_qi[route_idx, intermediate]
      expert_blk_scale[route_idx, intermediate / fwht_blk]
    """
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var offsets: UnsafePointer[Int32, MutAnyOrigin]
    var routes: UnsafePointer[SparseRoute, MutAnyOrigin]
    var w1_packed: I8Ptr
    var w1_scale: F32Ptr
    var w3_packed: I8Ptr
    var w3_scale: F32Ptr
    var expert_qi: I8Ptr
    var expert_blk_scale: F32Ptr
    var expert_stride: Int  # i8 elements per local expert
    var scale_stride: Int   # f32 elements per local expert
    var worker_id: Int
    var num_workers: Int

    def __init__(out self):
        self.act_i8 = I8Ptr()
        self.act_scale = F32Ptr()
        self.offsets = UnsafePointer[Int32, MutAnyOrigin]()
        self.routes = UnsafePointer[SparseRoute, MutAnyOrigin]()
        self.w1_packed = I8Ptr()
        self.w1_scale = F32Ptr()
        self.w3_packed = I8Ptr()
        self.w3_scale = F32Ptr()
        self.expert_qi = I8Ptr()
        self.expert_blk_scale = F32Ptr()
        self.expert_stride = 0
        self.scale_stride = 0
        self.worker_id = 0
        self.num_workers = 0


@fieldwise_init
struct SparseMoePhase2Args(Copyable, ImplicitlyCopyable):
    """Bucketed sparse MoE phase2 worker contract.

    Workers own a disjoint hidden-column stripe. They zero a f32 accumulator
    for that stripe, walk rank-local expert buckets, add weighted route outputs
    into accumulator[token, hidden], then cast once to bf16. Hidden ownership
    keeps the scatter/add race-free without atomics.
    """
    var offsets: UnsafePointer[Int32, MutAnyOrigin]
    var routes: UnsafePointer[SparseRoute, MutAnyOrigin]
    var expert_qi: I8Ptr
    var expert_blk_scale: F32Ptr
    var down_packed: I8Ptr
    var down_scale: F32Ptr
    var accum: F32Ptr
    var dst: BF16Ptr
    var expert_stride: Int  # i8 elements per local expert
    var scale_stride: Int   # f32 elements per local expert
    var seq_len: Int
    var hidden_start: Int
    var hidden_count: Int

    def __init__(out self):
        self.offsets = UnsafePointer[Int32, MutAnyOrigin]()
        self.routes = UnsafePointer[SparseRoute, MutAnyOrigin]()
        self.expert_qi = I8Ptr()
        self.expert_blk_scale = F32Ptr()
        self.down_packed = I8Ptr()
        self.down_scale = F32Ptr()
        self.accum = F32Ptr()
        self.dst = BF16Ptr()
        self.expert_stride = 0
        self.scale_stride = 0
        self.seq_len = 0
        self.hidden_start = 0
        self.hidden_count = 0


@fieldwise_init
struct RouterFusedArgs(Copyable, ImplicitlyCopyable):
    """Args for fused router worker: centered bf16 GEMV + gauge pivot + local top-K.

    Each worker owns rows [n_start, n_start + n_count) and writes K
    candidates to candidates[0..K) in descending score order.
    """
    var act_bf16: BF16Ptr
    var weight_bf16: BF16Ptr
    var gauge_bf16: BF16Ptr
    var bias_f32: F32Ptr
    var candidates: UnsafePointer[RouterCandidate, MutAnyOrigin]
    var eid_base: Int
    var n_start: Int
    var n_count: Int
    var row_count: Int
    var candidate_row_stride: Int

    def __init__(out self):
        self.act_bf16 = BF16Ptr()
        self.weight_bf16 = BF16Ptr()
        self.gauge_bf16 = BF16Ptr()
        self.bias_f32 = F32Ptr()
        self.candidates = UnsafePointer[RouterCandidate, MutAnyOrigin]()
        self.eid_base = 0
        self.n_start = 0
        self.n_count = 0
        self.row_count = 0
        self.candidate_row_stride = 0
