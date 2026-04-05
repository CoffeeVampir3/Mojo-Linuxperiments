"""SmolLM2-135M ButterQuant — isotropic int8 via Hadamard butterfly networks.

Weights are FWHT-rotated on the contraction dimension and quantized to I8
with per-row F32 scales. RMSNorm gamma is absorbed into projection weights
offline. Activation quantization uses a single model-global scale S_act = C(n).
KV cache is flat u8/i8 with per-layer fixed scales derived from weight norms.

Weight layout per layer (distributed across ranks):
  - 7 I8 projections (VNNI-packed at load time)
  - 7 F32 per-row weight scales (from quantized checkpoint)
  - 7 F32 column sums (computed at load time for VNNI bias correction)

Per-layer runtime constants (derived at load time from weight Frobenius norms):
  - S_Q, S_K, S_V: projection scales for attention quantization
  - S_post: post-nonlinearity scale for MLP re-quantization
  - S_attn_out = S_V (see butterquant.md Section 3.2 Category D)

Model-global constant: S_act = C(n) where n = FWHT_BLOCK = HEAD_DIM.

See butterquant.md for the full specification.
"""

from std.pathlib import Path
from std.memory import UnsafePointer, memcpy
from std.sys.info import size_of
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32, I8,
    RowShard, ColShard, Replicated,
    PrincipleNodeLocal,
    IsQuantizable, IsGammaQuantizable, IsPassthrough, IsAbsorbed, Quantizable,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
    Kernel3DTiling,
    LogitsView,
)
from kernels.vnni import VnniPacked, pack_vnni
from kernels.kernel_ops import (
    embed_lookup, elem_add, rmsnorm,
    init_rope_tables,
    PoolFence, parallel_for,
)
from experimental2.kernels.float_kernels.float_gemv import float_gemv
from experimental2.kernels.float_kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize
from experimental2.kernels.int_kernels.int8_gemv import int8_gemv, WorkerConfig
from experimental2.kernels.float_kernels.rope_and_kv_cache_write import rope_and_kv_cache_write
from experimental2.kernels.float_kernels.silu_fwht_quantize import silu_fwht_quantize
from experimental2.attn_amx_decode import (
    decode as amx_decode, decode_merge,
)
from experimental2.attn_amx_prefill import prefill as amx_prefill
from kernels.reductions import ring_allreduce, ring_broadcast
from modeling.loader import load_safetensors
from quant import quantize as butterquant_quantize
from experimental.linear_borrow_pool import ScratchPool, ScratchLease
from experimental.hadquant_impl import compute_column_sum
from experimental2.kv_cache import KVCache


# =============================================================================
# Model config
# =============================================================================


struct SmolLM2Config(Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig):
    comptime HIDDEN = 576
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 9
    comptime NUM_KV_HEADS = 3
    comptime INTERMEDIATE = 1536
    comptime VOCAB_SIZE = 49152
    comptime MAX_SEQ_LEN = 8192
    comptime ROPE_THETA = 100000.0
    comptime RMS_NORM_EPS = 1e-5
    comptime TIE_EMBEDDINGS = True

    comptime HEAD_DIM = Self.HIDDEN // Self.NUM_HEADS
    comptime KV_HIDDEN = Self.NUM_KV_HEADS * Self.HEAD_DIM
    comptime GQA_FACTOR = Self.NUM_HEADS // Self.NUM_KV_HEADS

comptime C = SmolLM2Config
comptime FWHT_BLOCK = C.HEAD_DIM


# =============================================================================
# Per-layer derived scales
# =============================================================================


struct LayerScales(Copyable, ImplicitlyCopyable):
    """Per-layer fixed scales derived from weight Frobenius norms at load time.

    S_Q = ||W'_Q||_F / sqrt(K) * C(d_k)
    S_K = ||W'_K||_F / sqrt(K) * C(d_k)
    S_V = ||W'_V||_F / sqrt(K) * C(d_k)
    S_post = sqrt(M_2(silu, ||W'_gate||_F / sqrt(K)) * ||W'_up||_F^2 / K) * C(n)
    """
    var q_layer_scale: Float32
    var k_layer_scale: Float32
    var v_layer_scale: Float32
    var post_layer_scale: Float32

    def __init__(out self):
        self.q_layer_scale = Float32(0)
        self.k_layer_scale = Float32(0)
        self.v_layer_scale = Float32(0)
        self.post_layer_scale = Float32(0)


# =============================================================================
# Concentration constant C(n) — comptime Monte Carlo
# =============================================================================

# C(n) = E[sqrt(n) * max_i |[H_n x]_i|] for x uniform on S^{n-1}.
# Computed via Monte Carlo with a deterministic PRNG seed at compile time.
# 10000 samples gives ~4 significant figures.


def comptime_sqrt(x: Float64) -> Float64:
    if x <= Float64(0):
        return Float64(0)
    var g = x
    for _ in range(60):
        g = (g + x / g) * Float64(0.5)
    return g


def concentration_constant[n: Int, num_samples: Int = 10000]() -> Float64:
    var state = UInt64(0xDEADBEEF12345678)
    var total = Float64(0)
    var sqrt_n = comptime_sqrt(Float64(n))
    var rsqrt_n = Float64(1) / sqrt_n

    for _ in range(num_samples):
        var vec = InlineArray[Float64, n](fill=Float64(0))
        var norm_sq = Float64(0)
        for i in range(n):
            # Irwin-Hall: sum of 12 uniforms - 6 approximates N(0,1)
            var val = Float64(0)
            for _ in range(12):
                state = state * 6364136223846793005 + 1442695040888963407
                val += Float64(state >> 11) / Float64(UInt64(1) << 53)
            val -= Float64(6)
            vec[i] = val
            norm_sq += val * val

        var inv_norm = Float64(1) / comptime_sqrt(norm_sq)
        for i in range(n):
            vec[i] *= inv_norm

        # FWHT in-place
        var h = 1
        while h < n:
            var i = 0
            while i < n:
                for j in range(h):
                    var a = vec[i + j]
                    var b = vec[i + j + h]
                    vec[i + j] = a + b
                    vec[i + j + h] = a - b
                i += h * 2
            h *= 2
        # Scale by 1/sqrt(n)
        var absmax = Float64(0)
        for i in range(n):
            var a = vec[i] * rsqrt_n
            if a < Float64(0):
                a = -a
            if a > absmax:
                absmax = a

        total += sqrt_n * absmax

    return total / Float64(num_samples)


# =============================================================================
# Layer spec
# =============================================================================


struct ButterQuantTPLayer[tp: Int]:

    # Norms: absorbed during quantization, absent from quantized file.
    # Placed before the projections they feed for quantizer ordering.
    comptime INPUT_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, 0, "input_layernorm.weight", IsAbsorbed]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, 0, "post_attention_layernorm.weight", IsAbsorbed]

    # I8 projection weights (VNNI-packed at load time).
    comptime Q_PROJ    = PlacedSlot[I8, RowShard, C.HIDDEN, C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime K_PROJ    = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime V_PROJ    = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime O_PROJ    = PlacedSlot[I8, ColShard, C.HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight", IsQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime GATE_PROJ = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime UP_PROJ   = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime DOWN_PROJ = PlacedSlot[I8, ColShard, C.HIDDEN, C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight", IsQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]

    # Per-row F32 weight scales (from quantized checkpoint).
    comptime Q_ROW_SCALE    = PlacedSlot[F32, RowShard, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "self_attn.q_proj.weight_scale"]
    comptime K_ROW_SCALE    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.Q_ROW_SCALE](), "self_attn.k_proj.weight_scale"]
    comptime V_ROW_SCALE    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.K_ROW_SCALE](), "self_attn.v_proj.weight_scale"]
    comptime O_ROW_SCALE    = PlacedSlot[F32, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.V_ROW_SCALE](), "self_attn.o_proj.weight_scale"]
    comptime GATE_ROW_SCALE = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.O_ROW_SCALE](), "mlp.gate_proj.weight_scale"]
    comptime UP_ROW_SCALE   = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.GATE_ROW_SCALE](), "mlp.up_proj.weight_scale"]
    comptime DOWN_ROW_SCALE = PlacedSlot[F32, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.UP_ROW_SCALE](), "mlp.down_proj.weight_scale"]

    # Column sums: computed at load time from packed i8 weights.
    # colsum[n] = sum_k weight_i8[n,k]. Used by int8_gemm for VNNI bias correction.
    comptime Q_COLSUM    = PlacedSlot[F32, RowShard, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_ROW_SCALE](), "self_attn.q_proj.weight_colsum"]
    comptime K_COLSUM    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.Q_COLSUM](), "self_attn.k_proj.weight_colsum"]
    comptime V_COLSUM    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.K_COLSUM](), "self_attn.v_proj.weight_colsum"]
    comptime O_COLSUM    = PlacedSlot[F32, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.V_COLSUM](), "self_attn.o_proj.weight_colsum"]
    comptime GATE_COLSUM = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.O_COLSUM](), "mlp.gate_proj.weight_colsum"]
    comptime UP_COLSUM   = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.GATE_COLSUM](), "mlp.up_proj.weight_colsum"]
    comptime DOWN_COLSUM = PlacedSlot[F32, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.UP_COLSUM](), "mlp.down_proj.weight_colsum"]

    comptime STRIDE = next_offset[Self.DOWN_COLSUM]()

    # Combined dimensions for fused GEMV dispatch.
    # The PlacedSlot chain places Q, K, V weights (and their row_scale and
    # colsum arrays) contiguously in memory with no gaps. This lets us
    # VNNI-pack Q+K+V as a single [QKV_N, K] matrix and run one GEMV that
    # produces all three outputs contiguously. The caller slices Q, K, V
    # by known row counts. Same principle applies to gate+up.
    comptime QKV_N = Self.Q_PROJ.ROWS + Self.K_PROJ.ROWS + Self.V_PROJ.ROWS
    comptime GATE_UP_N = Self.GATE_PROJ.ROWS + Self.UP_PROJ.ROWS

    # KV cache: flat u8/i8, zero per-token metadata.
    # One KVCache instance per layer holds both K (VNNI) and V (row-major i8).
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp
    comptime KVC = KVCache[C.MAX_SEQ_LEN, C.HEAD_DIM, Self.LOCAL_KV_HEADS]

    # Weight iteration: norms before the projections they feed.
    # Column sums are excluded (computed at load time, not loaded from file).
    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.INPUT_NORM](prefix, base)
        func[Self.Q_PROJ](prefix, base)
        func[Self.K_PROJ](prefix, base)
        func[Self.V_PROJ](prefix, base)
        func[Self.O_PROJ](prefix, base)
        func[Self.POST_ATTN_NORM](prefix, base)
        func[Self.GATE_PROJ](prefix, base)
        func[Self.UP_PROJ](prefix, base)
        func[Self.DOWN_PROJ](prefix, base)
        func[Self.Q_ROW_SCALE](prefix, base)
        func[Self.K_ROW_SCALE](prefix, base)
        func[Self.V_ROW_SCALE](prefix, base)
        func[Self.O_ROW_SCALE](prefix, base)
        func[Self.GATE_ROW_SCALE](prefix, base)
        func[Self.UP_ROW_SCALE](prefix, base)
        func[Self.DOWN_ROW_SCALE](prefix, base)

    @staticmethod
    def cache_bytes() -> Int:
        return Self.KVC.TOTAL_BYTES


# =============================================================================
# Model spec
# =============================================================================


struct ButterQuantTPModel[tp: Int = 1](WeightIterable):
    comptime LAYER = ButterQuantTPLayer[Self.tp]

    comptime LAYERS_OFF = 0
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = C.NUM_LAYERS * Self.LAYER.STRIDE

    comptime LOCAL_HEADS = C.NUM_HEADS // Self.tp
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp

    # Activation slots.
    comptime ROPE_HALF = C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    # Typed scratch views.
    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime KV_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime ACT_I8_HIDDEN = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime ACT_I8_INTERMEDIATE = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]

    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        """Peak scratch bytes across attention and MLP phases of one layer.

        LIFO borrow pattern — inner borrows are released before the next
        is allocated, so temporally non-overlapping regions share space.

        Attention phase (LIFO nesting: q_buf > qkv > act_work, then qkv > attn):
            q_buf:       MAX_SEQ_LEN * Q_COLS * 2         (bf16, contiguous Q for attention)
            qkv:         MAX_SEQ_LEN * QKV_N * 2          (bf16, combined GEMV output)
            act_work:    MAX_SEQ_LEN * HIDDEN + work       (i8 + f32 rmsnorm scratch)
            attn_scratch: MAX_SEQ_LEN * Q_COLS * 5 + overhead  (attention kernel internal)

        MLP phase (LIFO nesting: gate_up > mlp_work, then gate_up > post_work):
            gate_up:     MAX_SEQ_LEN * GATE_UP_N * 2      (bf16, combined GEMV output)
            mlp_work:    MAX_SEQ_LEN * HIDDEN + work       (i8 + f32 rmsnorm scratch)
            post_work:   MAX_SEQ_LEN * GATE_ROWS + work    (i8 + f32 silu scratch)
        """
        comptime S = C.MAX_SEQ_LEN
        comptime H = C.HIDDEN
        comptime I = C.INTERMEDIATE
        comptime TP = Self.tp
        comptime Q_COLS = H // TP
        comptime QKV_N = (H + 2 * C.KV_HIDDEN) // TP
        comptime GATE_UP_N = 2 * (I // TP)
        comptime GATE_ROWS = I // TP
        comptime MAX_WORKERS = 64
        comptime WORK_OVERHEAD = H * MAX_WORKERS * 4  # f32 work buffer for rmsnorm
        comptime SILU_WORK = GATE_ROWS * MAX_WORKERS * 4  # f32 work buffer for silu

        # Attention: q_buf + max(qkv + act_work, attn_scratch)
        comptime q_buf = S * Q_COLS * 2
        comptime qkv_act = S * QKV_N * 2 + S * H + WORK_OVERHEAD
        comptime attn_scratch = S * Q_COLS * 5 + 1024 * 1024  # generous overhead
        comptime attn_inner = attn_scratch if attn_scratch > qkv_act else qkv_act
        comptime attn_peak = q_buf + attn_inner

        # MLP: gate_up + max(mlp_work, post_work)
        comptime mlp_work = S * H + WORK_OVERHEAD
        comptime post_work = S * GATE_ROWS + SILU_WORK
        comptime mlp_inner = post_work if post_work > mlp_work else mlp_work
        comptime mlp_peak = S * GATE_UP_N * 2 + mlp_inner

        comptime if attn_peak > mlp_peak:
            return attn_peak
        else:
            return mlp_peak

    # State layout.
    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # NodeLocal weights (host arena only).
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, C.HIDDEN, 1, Self.tp, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[BF16, PrincipleNodeLocal, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ]():
        comptime for i in range(C.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.LAYERS_OFF + i * Self.LAYER_STRIDE
            Self.LAYER.for_each_weight[func](prefix, base)
        func[Self.FINAL_NORM]("", 0)
        func[Self.EMBED]("", 0)

    @staticmethod
    def arena_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    def host_arena_bytes() -> Int:
        return next_offset[Self.EMBED]()

    @staticmethod
    def quantize(source_path: Path, output_path: Path) -> Bool:
        return butterquant_quantize[Self, FWHT_BLOCK](source_path, output_path)


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankView[tp: Int]:
    comptime M = ButterQuantTPModel[Self.tp]
    comptime L = Self.M.LAYER
    var base: Int

    def __init__(out self, arena_base: Int):
        self.base = arena_base

    def weight_base(self) -> Int:
        return self.base

    def state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES

    def layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE)

    def weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())

    # KV cache: one instance per layer holds both K (VNNI) and V (row-major i8).
    comptime KVC = Self.L.KVC

    def kv_base(self, layer: Int) -> Int:
        return self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    def kv_cache(self, layer: Int) -> Self.KVC:
        return Self.KVC(self.kv_base(layer))

    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    def scratch_base(self) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF

    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        return DynView[V](self.scratch_base() + lease.offset, seq_len)

    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)

    def rope_cos(self) -> Bound[Self.M.ROPE_COS]:
        return Bound[Self.M.ROPE_COS](self.state_base() + Self.M.ROPE_COS_OFF)

    def rope_sin(self) -> Bound[Self.M.ROPE_SIN]:
        return Bound[Self.M.ROPE_SIN](self.state_base() + Self.M.ROPE_SIN_OFF)


# =============================================================================
# Ranks — dispatch helper
# =============================================================================


@fieldwise_init
struct Ranks[tp: Int]:
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def view(self, r: Int) -> RankView[Self.tp]:
        return RankView[Self.tp](self.bases[r])

    def parallel[body: def[rank: Int] (RankView[Self.tp], mut BurstPool[]) capturing -> PoolFence[BurstPool[]]](self):
        @parameter
        def dispatch[rank: Int]() -> PoolFence[BurstPool[]]:
            var rv = RankView[Self.tp](self.bases[rank])
            return body[rank](rv, self.pool_ptrs[rank][])
        parallel_for[BurstPool[], Self.tp, dispatch]()

    def each[body: def (RankView[Self.tp]) capturing -> None](self):
        for r in range(Self.tp):
            body(self.view(r))

    def x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_main(seq_len).ptr
        return ptrs^

    def x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_residual(seq_len).ptr
        return ptrs^


# =============================================================================
# Weight packing + column sums
# =============================================================================


def pack_combined[S1: Encoding & Shaped & Placed & Named,
                   S2: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    """VNNI-pack two contiguous weight matrices as one combined [N1+N2, K] block."""
    comptime combined_n = S1.ROWS + S2.ROWS
    comptime k = S1.COLS
    comptime assert S2.COLS == k, "pack_combined: K mismatch"
    comptime combined_bytes = combined_n * k
    var src = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + S1.OFFSET)
    memcpy(dest=scratch, src=src, count=combined_bytes)
    pack_vnni(scratch, src, combined_n, k)


def pack_combined3[S1: Encoding & Shaped & Placed & Named,
                    S2: Encoding & Shaped & Placed & Named,
                    S3: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    """VNNI-pack three contiguous weight matrices as one combined [N1+N2+N3, K] block."""
    comptime combined_n = S1.ROWS + S2.ROWS + S3.ROWS
    comptime k = S1.COLS
    comptime assert S2.COLS == k, "pack_combined3: K mismatch (S2)"
    comptime assert S3.COLS == k, "pack_combined3: K mismatch (S3)"
    comptime combined_bytes = combined_n * k
    var src = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + S1.OFFSET)
    memcpy(dest=scratch, src=src, count=combined_bytes)
    pack_vnni(scratch, src, combined_n, k)


def pack_single[S: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    """VNNI-pack a single weight matrix."""
    comptime weight_bytes = S.ROWS * S.COLS
    var src = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + S.OFFSET)
    memcpy(dest=scratch, src=src, count=weight_bytes)
    pack_vnni(scratch, src, S.ROWS, S.COLS)


def pack_weights[tp: Int](arena_base: Int, scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """VNNI-pack all projection weights per layer.

    Because the PlacedSlot chain places weights contiguously (Q then K then V,
    gate then up), we can pack multiple matrices as a single combined block.
    This produces one VNNI tiling across the full combined N dimension, so a
    single linear scan at runtime covers all sub-matrices. No extra memory —
    the packed output occupies the same bytes as the original row-major data.

    Q+K+V packed as one [QKV_N, K] block.
    Gate+Up packed as one [GateUp_N, K] block.
    O and Down packed individually.
    """
    comptime L = ButterQuantTPLayer[tp]
    comptime M = ButterQuantTPModel[tp]
    for layer in range(C.NUM_LAYERS):
        var lb = M.LAYERS_OFF + layer * M.LAYER_STRIDE
        pack_combined3[L.Q_PROJ, L.K_PROJ, L.V_PROJ](arena_base, lb, scratch)
        pack_single[L.O_PROJ](arena_base, lb, scratch)
        pack_combined[L.GATE_PROJ, L.UP_PROJ](arena_base, lb, scratch)
        pack_single[L.DOWN_PROJ](arena_base, lb, scratch)


def init_column_sums[tp: Int](arena_base: Int):
    comptime L = ButterQuantTPLayer[tp]
    comptime M = ButterQuantTPModel[tp]
    for layer in range(C.NUM_LAYERS):
        var layer_base = M.LAYERS_OFF + layer * M.LAYER_STRIDE
        compute_column_sum[L.Q_PROJ, L.Q_COLSUM](arena_base, layer_base)
        compute_column_sum[L.K_PROJ, L.K_COLSUM](arena_base, layer_base)
        compute_column_sum[L.V_PROJ, L.V_COLSUM](arena_base, layer_base)
        compute_column_sum[L.O_PROJ, L.O_COLSUM](arena_base, layer_base)
        compute_column_sum[L.GATE_PROJ, L.GATE_COLSUM](arena_base, layer_base)
        compute_column_sum[L.UP_PROJ, L.UP_COLSUM](arena_base, layer_base)
        compute_column_sum[L.DOWN_PROJ, L.DOWN_COLSUM](arena_base, layer_base)


# =============================================================================
# GEMM epilogue table precomputation
# =============================================================================


def precompute_gemm_bias[CS: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int,
):
    """Transform colsum[n] → 128 * colsum[n] in place."""
    var p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + CS.OFFSET)
    for n in range(CS.ROWS):
        p[n] = Float32(128) * p[n]


def precompute_gemm_scale[RS: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int, act_dequant: Float32,
):
    """Transform row_scale[n] → row_scale[n] * act_dequant in place."""
    var p = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + RS.OFFSET)
    for n in range(RS.ROWS):
        p[n] = p[n] * act_dequant


def precompute_gemm_tables[tp: Int](
    arena_base: Int,
    layer_scales: InlineArray[LayerScales, C.NUM_LAYERS],
):
    """Transform colsum → bias and row_scale → combined_scale in place.

    After this, the GEMM epilogue is: (raw_i32 - bias[n]) * scale[n].
    """
    comptime L = ButterQuantTPLayer[tp]
    comptime M = ButterQuantTPModel[tp]
    var s_act_dequant = Float32(concentration_constant[FWHT_BLOCK]()) / Float32(127)

    for layer in range(C.NUM_LAYERS):
        var lb = M.LAYERS_OFF + layer * M.LAYER_STRIDE
        var s_v_dequant = layer_scales[layer].v_layer_scale / Float32(127)
        var s_post_dequant = layer_scales[layer].post_layer_scale / Float32(127)

        # Bias: colsum → 128 * colsum
        precompute_gemm_bias[L.Q_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.K_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.V_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.O_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.GATE_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.UP_COLSUM](arena_base, lb)
        precompute_gemm_bias[L.DOWN_COLSUM](arena_base, lb)

        # Scale: row_scale → row_scale * (activation_scale / 127)
        # Q/K/V/gate/up: activated with S_act
        precompute_gemm_scale[L.Q_ROW_SCALE](arena_base, lb, s_act_dequant)
        precompute_gemm_scale[L.K_ROW_SCALE](arena_base, lb, s_act_dequant)
        precompute_gemm_scale[L.V_ROW_SCALE](arena_base, lb, s_act_dequant)
        precompute_gemm_scale[L.GATE_ROW_SCALE](arena_base, lb, s_act_dequant)
        precompute_gemm_scale[L.UP_ROW_SCALE](arena_base, lb, s_act_dequant)
        # O: activated with S_V (attention output quantized with S_V)
        precompute_gemm_scale[L.O_ROW_SCALE](arena_base, lb, s_v_dequant)
        # Down: activated with S_post (silu output quantized with S_post)
        precompute_gemm_scale[L.DOWN_ROW_SCALE](arena_base, lb, s_post_dequant)


# =============================================================================
# Per-layer scale derivation
# =============================================================================


def frobenius_from_quantized[W: Encoding & Shaped & Placed & Named,
                             S: Encoding & Shaped & Placed & Named](
    arena_base: Int, layer_base: Int,
) -> Float64:
    """Recover ||W'||_F from quantized i8 weights and per-row f32 scales.

    ||W'||_F^2 = sum_n s_w[n]^2 * sum_k W_i8[n,k]^2
    """
    var w_ptr = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + W.OFFSET)
    var s_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + S.OFFSET)
    var frob_sq = Float64(0)
    for n in range(W.ROWS):
        var row_sq = Float64(0)
        for k in range(W.COLS):
            var v = Float64(w_ptr[n * W.COLS + k])
            row_sq += v * v
        var s = Float64(s_ptr[n])
        frob_sq += s * s * row_sq
    return frob_sq


def derive_layer_scales[tp: Int](arena_base: Int) -> InlineArray[LayerScales, C.NUM_LAYERS]:
    """Compute per-layer fixed scales from weight Frobenius norms.

    See butterquant.md Sections 3.2 (Categories B, C).
    """
    comptime L = ButterQuantTPLayer[tp]
    comptime M = ButterQuantTPModel[tp]
    var cn = concentration_constant[FWHT_BLOCK]()
    var sqrt_k = Float64(C.HIDDEN).__pow__(0.5)
    comptime SILU_M2_COEFF = 0.298  # M_2(silu, 1) / 1^2, see spec Section 3.2C

    var scales = InlineArray[LayerScales, C.NUM_LAYERS](fill=LayerScales())
    for layer in range(C.NUM_LAYERS):
        var layer_base = M.LAYERS_OFF + layer * M.LAYER_STRIDE

        var q_frob_sq = frobenius_from_quantized[L.Q_PROJ, L.Q_ROW_SCALE](arena_base, layer_base)
        var k_frob_sq = frobenius_from_quantized[L.K_PROJ, L.K_ROW_SCALE](arena_base, layer_base)
        var v_frob_sq = frobenius_from_quantized[L.V_PROJ, L.V_ROW_SCALE](arena_base, layer_base)
        var gate_frob_sq = frobenius_from_quantized[L.GATE_PROJ, L.GATE_ROW_SCALE](arena_base, layer_base)
        var up_frob_sq = frobenius_from_quantized[L.UP_PROJ, L.UP_ROW_SCALE](arena_base, layer_base)

        # Category B: S_proj = ||W'||_F / sqrt(K) * C(d_k)
        scales[layer].q_layer_scale = Float32(q_frob_sq.__pow__(0.5) / sqrt_k * cn)
        scales[layer].k_layer_scale = Float32(k_frob_sq.__pow__(0.5) / sqrt_k * cn)
        scales[layer].v_layer_scale = Float32(v_frob_sq.__pow__(0.5) / sqrt_k * cn)

        # Category C: S_post = sqrt(M_2(silu, sigma_gate) * sigma_up^2) * C(n)
        # where sigma = ||W'||_F / sqrt(K)
        var sigma_gate = gate_frob_sq.__pow__(0.5) / sqrt_k
        var sigma_up_sq = up_frob_sq / Float64(C.HIDDEN)
        var m2 = SILU_M2_COEFF * sigma_gate * sigma_gate
        scales[layer].post_layer_scale = Float32((m2 * sigma_up_sq).__pow__(0.5) * cn)

    return scales^


# =============================================================================
# Loaded model
# =============================================================================


struct SmolLM2ButterQuant[tp: Int](Movable):
    comptime M = ButterQuantTPModel[Self.tp]

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]
    var layer_scales: InlineArray[LayerScales, C.NUM_LAYERS]
    var s_act: Float32

    def __init__(out self, var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
                var pools: HeapMoveArray[BurstPool[]],
                var layer_scales: InlineArray[LayerScales, C.NUM_LAYERS]):
        self.bases = InlineArray[Int, Self.tp](fill=0)
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
        )
        self.scratch = ScratchPool(Self.M.SCRATCH_CAPACITY)
        self.layer_scales = layer_scales^
        self.s_act = Float32(concentration_constant[FWHT_BLOCK]())
        self.arenas = arenas^
        self.pools = pools^
        for rank in range(Self.tp):
            self.bases[rank] = Int(self.arenas[rank].base)
            self.pool_ptrs[rank] = UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.pools[rank]))
            )

    def rank(self, r: Int) -> RankView[Self.tp]:
        return RankView[Self.tp](self.bases[r])

    @staticmethod
    def print_memory():
        comptime arena_per_rank = Self.M.arena_bytes()
        comptime host_arena = Self.M.host_arena_bytes()
        comptime total = host_arena + (Self.tp - 1) * arena_per_rank
        print("SmolLM2 ButterQuant TP=" + String(Self.tp) + ": "
            + String(total // (1024 * 1024)) + " MB total")
        comptime if Self.tp == 1:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
        else:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
            comptime for r in range(1, Self.tp):
                print("  rank " + String(r) + ":        " + String(arena_per_rank // (1024 * 1024)) + " MB")

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.rank(0).state_base() + Self.M.SCRATCH_OFF
        )

    @staticmethod
    def load(path: Path) -> Optional[Self]:
        comptime assert C.NUM_HEADS % Self.tp == 0, "TP must evenly divide NUM_HEADS"
        comptime assert C.NUM_KV_HEADS % Self.tp == 0, "TP must evenly divide NUM_KV_HEADS"
        comptime assert C.INTERMEDIATE % Self.tp == 0, "TP must evenly divide INTERMEDIATE"

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)
        comptime host_rank = 0

        # Allocate NUMA arenas.
        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == host_rank else Self.M.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("butterquant: arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        # Load quantized weights from safetensors.
        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_safetensors[Self.M](path, arena_bases, host_index=host_rank)
        if not result:
            print("butterquant: weight loading failed")
            return None

        # Prefault arenas.
        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        # Derive per-layer scales from weight Frobenius norms (before VNNI packing).
        var layer_scales = derive_layer_scales[Self.tp](Int(arenas[host_rank].base))

        # Column sums + VNNI packing (column sums computed before packing
        # since packing reorders the weight data).
        for rank in range(Self.tp):
            var base = Int(arenas[rank].base)
            init_column_sums[Self.tp](base)
            var scratch_ptr = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=base + Self.M.DISTRIBUTED_BYTES + Self.M.SCRATCH_OFF
            )
            pack_weights[Self.tp](base, scratch_ptr)

        # Create BurstPools per NUMA node.
        var pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var model = Self(arenas^, pools^, layer_scales^)

        # Init RoPE tables.
        for rank in range(Self.tp):
            var rv = model.rank(rank)
            init_rope_tables(rv.rope_cos(), rv.rope_sin(), Float64(C.ROPE_THETA))

        return model^


    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
    ) -> LogitsView[C.VOCAB_SIZE]:
        comptime M = Self.M
        comptime L = M.LAYER
        var s_act = self.s_act
        var s_act_dequant = s_act / Float32(127)
        comptime QKV_N = L.QKV_N
        comptime Q_ROWS = C.HIDDEN // Self.tp
        comptime KV_ROWS = C.KV_HIDDEN // Self.tp
        comptime Q_COLS = Q_ROWS  # = LOCAL_HEADS * HEAD_DIM
        comptime GATE_UP_N = L.GATE_UP_N
        comptime GATE_ROWS = C.INTERMEDIATE // Self.tp
        comptime O_K = Q_COLS
        comptime DOWN_K = GATE_ROWS
        comptime MAX_WORKERS = 64
        comptime WORK_F32 = C.HIDDEN * MAX_WORKERS  # f32 elems for rmsnorm work
        comptime SILU_WORK_F32 = GATE_ROWS * MAX_WORKERS  # f32 elems for silu work
        comptime ACT_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
        comptime MLP_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
        comptime POST_WORK_BYTES = C.MAX_SEQ_LEN * GATE_ROWS + SILU_WORK_F32 * size_of[Float32]()
        comptime ATTN_SCRATCH_BYTES = C.MAX_SEQ_LEN * Q_COLS * (size_of[Float32]() + 1) + 1024 * 1024

        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)

        # --- Embed (host rank, then broadcast) ---
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), self.pools[0]).join()
        ring_broadcast[M.X_MAIN, Self.tp](host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        for layer_idx in range(C.NUM_LAYERS):
            var ls = self.layer_scales[layer_idx]
            var s_v_dequant = ls.v_layer_scale / Float32(127)
            var s_post_dequant = ls.post_layer_scale / Float32(127)

            # =============================================================
            # Attention block
            # =============================================================
            # LIFO: q_buf > qkv > act_work (release) > attn (release) > (release qkv) > (release q_buf)

            var q_buf = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Q_COLS]()
            var qkv = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * QKV_N]()
            var act_work = self.scratch.borrow[UInt8, ACT_WORK_BYTES]()
            var act_i8_off = act_work.offset
            var work_off = act_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                var sb = rv.scratch_base()

                # RMSNorm + FWHT + quantize: x_main → act_i8
                rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                    rv.x_main(seq_len).ptr, sb + act_i8_off, sb + work_off,
                    seq_len, s_act, self.pools[rank],
                ).join()

                # int8_gemv QKV: act_i8 → qkv
                var qkv_configs = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
                int8_gemv[QKV_N, C.HIDDEN](
                    sb + act_i8_off,
                    rv.layer_weight[L.Q_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.Q_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.Q_ROW_SCALE](layer_idx).ptr,
                    sb + qkv.offset, seq_len, s_act_dequant,
                    qkv_configs, self.pools[rank],
                ).join()
                _ = qkv_configs

            act_work^.release()

            # KV cache write + Q copy to contiguous buffer
            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                var sb = rv.scratch_base()
                var kvc = rv.kv_cache(layer_idx)
                var qkv_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=sb + qkv.offset)

                rope_and_kv_cache_write[C.HEAD_DIM, M.LOCAL_KV_HEADS, C.MAX_SEQ_LEN](
                    qkv_ptr + Q_ROWS,              # k_bf16 (after Q columns)
                    qkv_ptr + Q_ROWS + KV_ROWS,    # v_bf16 (after K columns)
                    QKV_N, QKV_N,                   # k_stride, v_stride (element stride)
                    UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_cos().ptr),
                    UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_sin().ptr),
                    kvc, pos, seq_len,
                    ls.k_layer_scale, ls.v_layer_scale,
                )

                # Copy Q from strided QKV to contiguous q_buf
                var q_dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=sb + q_buf.offset)
                for m in range(seq_len):
                    memcpy(dest=q_dst + m * Q_COLS, src=qkv_ptr + m * QKV_N, count=Q_COLS)

            qkv^.release()

            # Attention → i8 output (within attn_scratch)
            var attn = self.scratch.borrow[UInt8, ATTN_SCRATCH_BYTES]()

            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                var sb = rv.scratch_base()
                var kvc = rv.kv_cache(layer_idx)
                var q_view = DynView[M.Q_VIEW](sb + q_buf.offset, seq_len)

                if seq_len == 1:
                    var wpg_hint = self.pools[rank].get_capacity() // M.LOCAL_KV_HEADS
                    amx_decode[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN](
                        q_view, kvc, kvc,
                        rv.rope_cos(), rv.rope_sin(),
                        sb + attn.offset, pos,
                        ls.q_layer_scale, ls.k_layer_scale, ls.v_layer_scale,
                        wpg_hint, self.pools[rank],
                    ).join()
                    # Replicate decode's internal wpg capping for decode_merge
                    comptime DECODE_BLOCK_N = 512
                    var context = pos + 1
                    var max_blocks = (context + DECODE_BLOCK_N - 1) // DECODE_BLOCK_N
                    var wpg = min(min(wpg_hint, max_blocks),
                                  self.pools[rank].get_capacity() // M.LOCAL_KV_HEADS)
                    var vagg_scale = ls.v_layer_scale / (Float32(255) * Float32(127))
                    decode_merge[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM](
                        sb + attn.offset, wpg, vagg_scale,
                    )
                else:
                    amx_prefill[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN](
                        q_view, kvc, kvc,
                        rv.rope_cos(), rv.rope_sin(),
                        sb + attn.offset, pos,
                        ls.q_layer_scale, ls.k_layer_scale, ls.v_layer_scale,
                        self.pools[rank],
                    ).join()

                # i8 output location within attn scratch
                var attn_i8_addr = sb + attn.offset
                if seq_len > 1:
                    attn_i8_addr += seq_len * Q_COLS * size_of[Float32]()

                # int8_gemv O: attn_i8 → x_residual
                var o_configs = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
                int8_gemv[C.HIDDEN, O_K](
                    attn_i8_addr,
                    rv.layer_weight[L.O_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.O_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.O_ROW_SCALE](layer_idx).ptr,
                    rv.x_residual(seq_len).ptr, seq_len, s_v_dequant,
                    o_configs, self.pools[rank],
                ).join()
                _ = o_configs

            attn^.release()
            q_buf^.release()

            # Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)
            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))

            # =============================================================
            # MLP block
            # =============================================================
            # LIFO: gate_up > mlp_work (release) > post_work (release) > (release gate_up)

            var gate_up = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * GATE_UP_N]()
            var mlp_work = self.scratch.borrow[UInt8, MLP_WORK_BYTES]()
            var mlp_i8_off = mlp_work.offset
            var mlp_work_off = mlp_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                var sb = rv.scratch_base()

                # RMSNorm + FWHT + quantize: x_main → mlp_i8
                rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                    rv.x_main(seq_len).ptr, sb + mlp_i8_off, sb + mlp_work_off,
                    seq_len, s_act, self.pools[rank],
                ).join()

                # int8_gemv gate+up: mlp_i8 → gate_up
                var gu_configs = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
                int8_gemv[GATE_UP_N, C.HIDDEN](
                    sb + mlp_i8_off,
                    rv.layer_weight[L.GATE_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.GATE_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.GATE_ROW_SCALE](layer_idx).ptr,
                    sb + gate_up.offset, seq_len, s_act_dequant,
                    gu_configs, self.pools[rank],
                ).join()
                _ = gu_configs

            mlp_work^.release()

            # SiLU + FWHT + quantize: gate_up → post_i8, then down projection
            var post_work = self.scratch.borrow[UInt8, POST_WORK_BYTES]()
            var post_i8_off = post_work.offset
            var silu_work_off = post_work.offset + C.MAX_SEQ_LEN * GATE_ROWS

            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                var sb = rv.scratch_base()

                silu_fwht_quantize[GATE_ROWS, GATE_UP_N, FWHT_BLOCK](
                    sb + gate_up.offset, sb + post_i8_off, sb + silu_work_off,
                    seq_len, ls.post_layer_scale, self.pools[rank],
                ).join()

                # int8_gemv down: post_i8 → x_residual
                var dn_configs = InlineArray[WorkerConfig, 128](fill=WorkerConfig(Float32(0), 0))
                int8_gemv[C.HIDDEN, DOWN_K](
                    sb + post_i8_off,
                    rv.layer_weight[L.DOWN_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.DOWN_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.DOWN_ROW_SCALE](layer_idx).ptr,
                    rv.x_residual(seq_len).ptr, seq_len, s_post_dequant,
                    dn_configs, self.pools[rank],
                ).join()
                _ = dn_configs

            post_work^.release()
            gate_up^.release()

            # Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)
            for rank in range(Self.tp):
                var rv = ranks.view(rank)
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))

            _ = layer_idx

        # --- Final norm + LM head (host rank only, tied embeddings) ---
        var last_row_off = (seq_len - 1) * C.HIDDEN * M.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[M.X_MAIN](host.x_main(seq_len).ptr + last_row_off, 1)
        rmsnorm(last_hidden, host.weight[M.FINAL_NORM](), last_hidden, self.pools[0]).join()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = host.scratch_view[M.LOGITS](logit_lease, 1)
        float_gemv(last_hidden, host.weight[M.EMBED](), logit_view, self.pools[0]).join()

        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^,
        )
