"""SmolLM2-135M ButterQuant — Hadamard-rotated channelwise int8 quantization.

Weights are FWHT-rotated on the contraction dimension and quantized to I8
with per-row F32 scales. RMSNorm gamma is absorbed into projection weights
offline. All activation scales are dynamic per-row absmax, except V which
uses a fixed per-layer scale derived from weight Frobenius norms.

Weight layout per layer (distributed across ranks):
  - 7 I8 projections (VNNI-packed at load time)
  - 7 F32 per-row weight scales (from quantized checkpoint)
  - 7 F32 column sums (computed at load time for VNNI bias correction)

See butterquant.md for the full specification.
"""

from std.pathlib import Path
from std.memory import UnsafePointer, memcpy
from std.sys.info import size_of, simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_shared import ptr as tptr

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32, I8,
    RowShard, ColShard, Replicated, HOST_RANK,
    IsQuantizable, IsGammaQuantizable, IsAbsorbed,
    Slot, PlacedSlot, Bound, DynView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
    Kernel3DTiling,
    LogitsView,
)
from kernels.vnni import VnniPacked, pack_vnni
from kernels.kernel_ops import (
    embed_lookup, rmsnorm,
    PoolFence, parallel_for,
)
from kernels.kv_rotors import init_rope_tables
from experimental2.kernels.float_gemv import float_gemv
from experimental2.kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize
from experimental2.kernels.int8_gemv import int8_gemv, fused_gu_silu
from experimental2.kernels.rope_and_kv_cache_write import rope_and_kv_cache_write
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
# Forward pass profiling
# =============================================================================


@fieldwise_init
struct ForwardProfile(Copyable, Movable):
    var count: Int
    var embed_ns: Int
    var attn_norm_ns: Int
    var qkv_gemv_ns: Int
    var kv_write_ns: Int
    var attention_ns: Int
    var o_gemv_ns: Int
    var attn_reduce_ns: Int
    var mlp_norm_ns: Int
    var gu_gemv_ns: Int
    var silu_ns: Int
    var down_gemv_ns: Int
    var mlp_reduce_ns: Int
    var final_ns: Int

    def __init__(out self):
        self.count = 0
        self.embed_ns = 0
        self.attn_norm_ns = 0
        self.qkv_gemv_ns = 0
        self.kv_write_ns = 0
        self.attention_ns = 0
        self.o_gemv_ns = 0
        self.attn_reduce_ns = 0
        self.mlp_norm_ns = 0
        self.gu_gemv_ns = 0
        self.silu_ns = 0
        self.down_gemv_ns = 0
        self.mlp_reduce_ns = 0
        self.final_ns = 0

    def accumulate(mut self, mut other: Self):
        self.count += 1
        self.embed_ns += other.embed_ns
        self.attn_norm_ns += other.attn_norm_ns
        self.qkv_gemv_ns += other.qkv_gemv_ns
        self.kv_write_ns += other.kv_write_ns
        self.attention_ns += other.attention_ns
        self.o_gemv_ns += other.o_gemv_ns
        self.attn_reduce_ns += other.attn_reduce_ns
        self.mlp_norm_ns += other.mlp_norm_ns
        self.gu_gemv_ns += other.gu_gemv_ns
        self.silu_ns += other.silu_ns
        self.down_gemv_ns += other.down_gemv_ns
        self.mlp_reduce_ns += other.mlp_reduce_ns
        self.final_ns += other.final_ns

    def report(self):
        if self.count == 0:
            print("(no forward passes profiled)")
            return
        var n = self.count
        var total = (self.embed_ns + self.attn_norm_ns + self.qkv_gemv_ns
            + self.kv_write_ns + self.attention_ns + self.o_gemv_ns
            + self.attn_reduce_ns + self.mlp_norm_ns + self.gu_gemv_ns
            + self.silu_ns + self.down_gemv_ns + self.mlp_reduce_ns
            + self.final_ns)
        print("forward profile (" + String(n) + " calls, avg " + String(total // n // 1000) + " us):")
        Self.phase("  embed      ", self.embed_ns, n)
        Self.phase("  attn_norm  ", self.attn_norm_ns, n)
        Self.phase("  qkv_gemv   ", self.qkv_gemv_ns, n)
        Self.phase("  kv_write   ", self.kv_write_ns, n)
        Self.phase("  attention  ", self.attention_ns, n)
        Self.phase("  o_gemv     ", self.o_gemv_ns, n)
        Self.phase("  attn_reduce", self.attn_reduce_ns, n)
        Self.phase("  mlp_norm   ", self.mlp_norm_ns, n)
        Self.phase("  gu_gemv    ", self.gu_gemv_ns, n)
        Self.phase("  silu       ", self.silu_ns, n)
        Self.phase("  down_gemv  ", self.down_gemv_ns, n)
        Self.phase("  mlp_reduce ", self.mlp_reduce_ns, n)
        Self.phase("  final      ", self.final_ns, n)

    @staticmethod
    def phase(name: String, total_ns: Int, count: Int):
        print(name + String(total_ns // count // 1000) + " us")


# =============================================================================
# Per-layer derived scales
# =============================================================================


struct LayerScales(Copyable, ImplicitlyCopyable):
    """Per-layer fixed scale for V projection only.

    All other scales are dynamic (per-row absmax, computed at runtime):
      S_act: per-row in rmsnorm_fwht_quantize
      Q/K:   per-head in attention kernel
      S_post: per-row in silu_fwht_quantize

    V uses a corrected fixed scale because V norms are small and uniform
    across heads, and per-position V scales would complicate the V-agg kernel.
      S_V = ||W'_V||_F / sqrt(KV_HIDDEN) * C(d_k)
    """
    var v_layer_scale: Float32

    def __init__(out self):
        self.v_layer_scale = Float32(0)


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

    # Attention cache: K (VNNI), V (row-major i8), per-head dynamic Q/K scales.
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp
    comptime LOCAL_Q_HEADS = C.NUM_HEADS // Self.tp
    comptime KVC = KVCache[C.MAX_SEQ_LEN, C.HEAD_DIM, Self.LOCAL_KV_HEADS, Self.LOCAL_Q_HEADS]

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

        Persistent borrows (live across all layers):
            act_scales:  MAX_SEQ_LEN * 4    (f32 per-row scales from rmsnorm_fwht_quantize)
            post_scales: MAX_SEQ_LEN * 4    (f32 per-row scales from silu_fwht_quantize)
            v_scales:    MAX_SEQ_LEN * 4    (f32 constant V scale for O GEMV)

        Attention phase (LIFO nesting: qkv > act_work, then qkv + attn):
            qkv:         MAX_SEQ_LEN * QKV_N * 2          (bf16, stays live for Q during attn)
            act_work:    MAX_SEQ_LEN * HIDDEN + work       (i8 + f32 rmsnorm scratch)
            attn_scratch: MAX_SEQ_LEN * Q_COLS * 5 + overhead

        MLP phase (fused gate+up GEMV → SiLU → FWHT → i8):
            mlp_work:    MAX_SEQ_LEN * HIDDEN + work       (i8 + f32 rmsnorm scratch)
            post_i8:     MAX_SEQ_LEN * GATE_ROWS           (i8 fused output)
            Both live during fused kernel (input i8 must not alias output i8).
        """
        comptime S = C.MAX_SEQ_LEN
        comptime H = C.HIDDEN
        comptime I = C.INTERMEDIATE
        comptime TP = Self.tp
        comptime Q_COLS = H // TP
        comptime QKV_N = (H + 2 * C.KV_HIDDEN) // TP
        comptime GATE_ROWS = I // TP
        comptime MAX_WORKERS = 64
        comptime WORK_OVERHEAD = H * MAX_WORKERS * 4  # f32 work buffer for rmsnorm

        # Persistent scale arrays (3 × MAX_SEQ_LEN f32)
        comptime scale_arrays = 3 * S * 4

        # Attention: qkv + max(act_work, attn_scratch)
        comptime qkv_buf = S * QKV_N * 2
        comptime act_work = S * H + WORK_OVERHEAD
        comptime attn_scratch = S * Q_COLS * 5 + 1024 * 1024
        comptime attn_inner = attn_scratch if attn_scratch > act_work else act_work
        comptime attn_peak = qkv_buf + attn_inner

        # MLP: mlp_work + post_i8 (both live during fused kernel)
        comptime mlp_work = S * H + WORK_OVERHEAD
        comptime post_i8_bytes = S * GATE_ROWS
        comptime mlp_peak = mlp_work + post_i8_bytes

        comptime layer_peak = attn_peak if attn_peak > mlp_peak else mlp_peak
        return scale_arrays + layer_peak

    # State layout.
    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # Host-only weights (host arena only).
    comptime HOST_ONLY_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, Self.HOST_ONLY_OFF, "model.norm.weight", target_rank=HOST_RANK]
    comptime EMBED = PlacedSlot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight", target_rank=HOST_RANK]

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
        pack_single[L.GATE_PROJ](arena_base, lb, scratch)
        pack_single[L.UP_PROJ](arena_base, lb, scratch)
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


def derive_layer_scales[tp: Int](arena_bases: List[Int]) -> InlineArray[LayerScales, C.NUM_LAYERS]:
    """Compute per-layer fixed V scale from the full TP-sharded V matrix.

    V is row-sharded across TP ranks, so the global Frobenius norm is:

      ||W'_V||_F^2 = sum_r ||W'_{V,r}||_F^2

    The fixed scale must be derived from that full-matrix norm; using one
    local shard with the global KV_HIDDEN denominator underestimates S_V.

      S_V = ||W'_V||_F / sqrt(KV_HIDDEN) * C(d_k)

    All other activation scales (S_act, Q, K, S_post) are dynamic per-row.
    """
    comptime L = ButterQuantTPLayer[tp]
    comptime M = ButterQuantTPModel[tp]
    var cn = concentration_constant[FWHT_BLOCK]()
    var sqrt_kv_hidden = Float64(C.KV_HIDDEN).__pow__(0.5)

    var scales = InlineArray[LayerScales, C.NUM_LAYERS](fill=LayerScales())
    for layer in range(C.NUM_LAYERS):
        var layer_base = M.LAYERS_OFF + layer * M.LAYER_STRIDE
        var v_frob_sq = Float64(0)
        for rank in range(tp):
            v_frob_sq += frobenius_from_quantized[L.V_PROJ, L.V_ROW_SCALE](
                arena_bases[rank], layer_base)
        scales[layer].v_layer_scale = Float32(v_frob_sq.__pow__(0.5) / sqrt_kv_hidden * cn)

    return scales^


# =============================================================================
# NUMA-local dispatch kernels (typed dispatch ABI)
# =============================================================================


@fieldwise_init
struct KVWriteConfig(Copyable, ImplicitlyCopyable):
    var k_ptr: Int
    var v_ptr: Int
    var k_stride: Int
    var v_stride: Int
    var cos_ptr: Int
    var sin_ptr: Int
    var cache_base: Int
    var pos: Int
    var seq_len: Int
    var s_v: Float32


def kv_write_worker[head_dim: Int, num_kv_heads: Int, max_seq: Int, num_q_heads: Int](
    cfg: KVWriteConfig,
):
    rope_and_kv_cache_write[head_dim, num_kv_heads, max_seq, num_q_heads](
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=cfg.k_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=cfg.v_ptr),
        cfg.k_stride, cfg.v_stride,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.cos_ptr),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.sin_ptr),
        KVCache[max_seq, head_dim, num_kv_heads, num_q_heads](cfg.cache_base),
        cfg.pos, cfg.seq_len, cfg.s_v,
    )


@fieldwise_init
struct ElemAddArgs(Copyable, ImplicitlyCopyable):
    var a_ptr: Int
    var b_ptr: Int
    var dst_ptr: Int
    var count: Int


def elem_add_worker(args: ElemAddArgs):
    var ap = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.a_ptr)
    var bp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.b_ptr)
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=args.dst_ptr)
    comptime width = simd_width_of[DType.float32]()
    for i in range(0, args.count, width):
        var av = (ap + i).load[width=width]().cast[DType.float32]()
        var bv = (bp + i).load[width=width]().cast[DType.float32]()
        (dp + i).store((av + bv).cast[DType.bfloat16]())


@fieldwise_init
struct DecodeMergeArgs(Copyable, ImplicitlyCopyable):
    var scratch: Int
    var wpg: Int
    var vagg_scale_bits: Int


def decode_merge_worker[num_heads: Int, num_kv_heads: Int, head_dim: Int](
    args: DecodeMergeArgs,
):
    var vagg_scale = Float32(from_bits=UInt32(args.vagg_scale_bits))
    decode_merge[num_heads, num_kv_heads, head_dim](args.scratch, args.wpg, vagg_scale)


# =============================================================================
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
    var profile: ForwardProfile

    def __init__(out self, var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
                var pools: HeapMoveArray[BurstPool[]],
                var layer_scales: InlineArray[LayerScales, C.NUM_LAYERS]):
        self.bases = InlineArray[Int, Self.tp](fill=0)
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
        )
        self.scratch = ScratchPool(Self.M.SCRATCH_CAPACITY)
        self.layer_scales = layer_scales^
        self.profile = ForwardProfile()
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

    def report_profile(self):
        self.profile.report()

    def reset_profile(mut self):
        self.profile = ForwardProfile()

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

        # Allocate NUMA arenas.
        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == HOST_RANK else Self.M.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("butterquant: arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        # Load quantized weights from safetensors.
        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_safetensors[Self.M](path, arena_bases)
        if not result:
            print("butterquant: weight loading failed")
            return None

        # Prefault arenas.
        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        # Derive per-layer scales from weight Frobenius norms (before VNNI packing).
        var layer_scales = derive_layer_scales[Self.tp](arena_bases)

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
        comptime QKV_N = L.QKV_N
        comptime Q_ROWS = C.HIDDEN // Self.tp
        comptime KV_ROWS = C.KV_HIDDEN // Self.tp
        comptime Q_COLS = Q_ROWS
        comptime GATE_ROWS = C.INTERMEDIATE // Self.tp
        comptime O_K = Q_COLS
        comptime DOWN_K = GATE_ROWS
        comptime MAX_WORKERS = 64
        comptime WORK_F32 = C.HIDDEN * MAX_WORKERS
        comptime ACT_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
        comptime MLP_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
        comptime ATTN_SCRATCH_BYTES = C.MAX_SEQ_LEN * Q_COLS * (size_of[Float32]() + 1) + 1024 * 1024

        var fp = ForwardProfile()
        var t0 = Int(perf_counter_ns())

        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)

        # --- Embed (host rank, then broadcast) ---
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), self.pools[0]).join()
        ring_broadcast[M.X_MAIN, Self.tp](host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        var t1 = Int(perf_counter_ns())
        fp.embed_ns = t1 - t0

        var act_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
        var post_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
        var v_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()

        var kv_cfg_buf = InlineArray[KVWriteConfig, Self.tp](
            fill=KVWriteConfig(0, 0, 0, 0, 0, 0, 0, 0, 0, Float32(0)))
        var kv_cfg_base = Int(UnsafePointer(to=kv_cfg_buf).bitcast[KVWriteConfig]())

        for layer_idx in range(C.NUM_LAYERS):
            var ls = self.layer_scales[layer_idx]

            # =============================================================
            # Attention block
            # =============================================================

            var qkv = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * QKV_N]()
            var act_work = self.scratch.borrow[UInt8, ACT_WORK_BYTES]()
            var act_i8_off = act_work.offset
            var work_off = act_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

            var ta = Int(perf_counter_ns())

            @parameter
            def do_attn_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                return rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                    rv.x_main(seq_len).ptr, sb + act_i8_off, sb + work_off,
                    sb + act_scale_lease.offset, Float32(C.RMS_NORM_EPS), seq_len, pool)
            ranks.parallel[do_attn_norm]()

            var tb = Int(perf_counter_ns())

            @parameter
            def do_qkv_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                return int8_gemv[QKV_N, C.HIDDEN](
                    sb + act_i8_off,
                    rv.layer_weight[L.Q_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.Q_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.Q_ROW_SCALE](layer_idx).ptr,
                    sb + qkv.offset, seq_len,
                    sb + act_scale_lease.offset,
                    pool)
            ranks.parallel[do_qkv_gemv]()

            act_work^.release()

            var tc = Int(perf_counter_ns())

            @parameter
            def do_kv_write[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                var cfg_ptr = UnsafePointer[KVWriteConfig, MutAnyOrigin](
                    unsafe_from_address=kv_cfg_base + rank * size_of[KVWriteConfig]())
                cfg_ptr[] = KVWriteConfig(
                    sb + qkv.offset + Q_ROWS * 2,
                    sb + qkv.offset + (Q_ROWS + KV_ROWS) * 2,
                    QKV_N, QKV_N,
                    rv.rope_cos().ptr, rv.rope_sin().ptr,
                    rv.kv_base(layer_idx),
                    pos, seq_len, ls.v_layer_scale)
                pool.dispatch[KVWriteConfig,
                    kv_write_worker[C.HEAD_DIM, M.LOCAL_KV_HEADS, C.MAX_SEQ_LEN, M.LOCAL_HEADS]](
                    cfg_ptr, 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            ranks.parallel[do_kv_write]()

            var td = Int(perf_counter_ns())

            var attn = self.scratch.borrow[UInt8, ATTN_SCRATCH_BYTES]()

            @parameter
            def do_attention[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                var kvc = rv.kv_cache(layer_idx)
                var q_view = DynView[M.Q_VIEW](sb + qkv.offset, seq_len)
                if seq_len == 1:
                    var wpg_hint = pool.get_capacity() // M.LOCAL_KV_HEADS
                    amx_decode[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN, M.LOCAL_HEADS](
                        q_view, QKV_N, kvc,
                        rv.rope_cos(), rv.rope_sin(),
                        sb + attn.offset, pos, ls.v_layer_scale,
                        wpg_hint, pool).join()
                    comptime DECODE_BLOCK_N = 512
                    var context = pos + 1
                    var max_blocks = (context + DECODE_BLOCK_N - 1) // DECODE_BLOCK_N
                    var wpg = min(min(wpg_hint, max_blocks), pool.get_capacity() // M.LOCAL_KV_HEADS)
                    var vagg_scale = ls.v_layer_scale / (Float32(255) * Float32(127))
                    var merge_args = InlineArray[DecodeMergeArgs, 1](
                        fill=DecodeMergeArgs(sb + attn.offset, wpg, Int(vagg_scale.to_bits())))
                    pool.dispatch[DecodeMergeArgs,
                        decode_merge_worker[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM]](
                        UnsafePointer(to=merge_args[0]), 1)
                    return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                        unsafe_from_address=Int(UnsafePointer(to=pool))))
                else:
                    return amx_prefill[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN, M.LOCAL_HEADS](
                        q_view, QKV_N, kvc,
                        rv.rope_cos(), rv.rope_sin(),
                        sb + attn.offset, pos, ls.v_layer_scale, pool)
            ranks.parallel[do_attention]()

            var te = Int(perf_counter_ns())

            @parameter
            def do_o_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                var v_sc = UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=sb + v_scale_lease.offset)
                for m in range(seq_len):
                    v_sc[m] = ls.v_layer_scale
                var attn_i8_addr = sb + attn.offset
                if seq_len > 1:
                    attn_i8_addr += seq_len * Q_COLS * size_of[Float32]()
                return int8_gemv[C.HIDDEN, O_K](
                    attn_i8_addr,
                    rv.layer_weight[L.O_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.O_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.O_ROW_SCALE](layer_idx).ptr,
                    rv.x_residual(seq_len).ptr, seq_len,
                    sb + v_scale_lease.offset,
                    pool)
            ranks.parallel[do_o_gemv]()

            qkv^.release()
            attn^.release()

            var tf = Int(perf_counter_ns())

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_attn_res_add[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var add_args = InlineArray[ElemAddArgs, 1](
                    fill=ElemAddArgs(
                        rv.x_main(seq_len).ptr,
                        rv.x_residual(seq_len).ptr,
                        rv.x_main(seq_len).ptr,
                        seq_len * C.HIDDEN))
                pool.dispatch[ElemAddArgs, elem_add_worker](
                    UnsafePointer(to=add_args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            ranks.parallel[do_attn_res_add]()

            var tg = Int(perf_counter_ns())

            # =============================================================
            # MLP block
            # =============================================================

            var mlp_work = self.scratch.borrow[UInt8, MLP_WORK_BYTES]()
            var mlp_i8_off = mlp_work.offset
            var mlp_work_off = mlp_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

            @parameter
            def do_mlp_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                return rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                    rv.x_main(seq_len).ptr, sb + mlp_i8_off, sb + mlp_work_off,
                    sb + act_scale_lease.offset, Float32(C.RMS_NORM_EPS), seq_len, pool)
            ranks.parallel[do_mlp_norm]()

            var th = Int(perf_counter_ns())

            # Fused gate+up GEMV → SiLU → FWHT → i8 (no bf16 intermediate)
            # mlp_work stays alive: fused kernel reads i8 input from mlp_i8_off
            var post_i8 = self.scratch.borrow[Scalar[DType.int8], C.MAX_SEQ_LEN * GATE_ROWS]()

            @parameter
            def do_fused_gu_silu[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                return fused_gu_silu[GATE_ROWS, C.HIDDEN, FWHT_BLOCK](
                    sb + mlp_i8_off,
                    rv.layer_weight[L.GATE_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.GATE_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.GATE_ROW_SCALE](layer_idx).ptr,
                    rv.layer_weight[L.UP_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.UP_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.UP_ROW_SCALE](layer_idx).ptr,
                    sb + post_i8.offset,
                    sb + post_scale_lease.offset,
                    seq_len, sb + act_scale_lease.offset,
                    pool)
            ranks.parallel[do_fused_gu_silu]()

            mlp_work^.release()

            var ti = Int(perf_counter_ns())
            var tj = ti  # silu is fused, no separate phase

            @parameter
            def do_down_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var sb = rv.scratch_base()
                return int8_gemv[C.HIDDEN, DOWN_K](
                    sb + post_i8.offset,
                    rv.layer_weight[L.DOWN_PROJ](layer_idx).ptr,
                    rv.layer_weight[L.DOWN_COLSUM](layer_idx).ptr,
                    rv.layer_weight[L.DOWN_ROW_SCALE](layer_idx).ptr,
                    rv.x_residual(seq_len).ptr, seq_len,
                    sb + post_scale_lease.offset,
                    pool)
            ranks.parallel[do_down_gemv]()

            post_i8^.release()

            var tk = Int(perf_counter_ns())

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_mlp_res_add[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
                var add_args = InlineArray[ElemAddArgs, 1](
                    fill=ElemAddArgs(
                        rv.x_main(seq_len).ptr,
                        rv.x_residual(seq_len).ptr,
                        rv.x_main(seq_len).ptr,
                        seq_len * C.HIDDEN))
                pool.dispatch[ElemAddArgs, elem_add_worker](
                    UnsafePointer(to=add_args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            ranks.parallel[do_mlp_res_add]()

            var tl = Int(perf_counter_ns())

            fp.attn_norm_ns += tb - ta
            fp.qkv_gemv_ns += tc - tb
            fp.kv_write_ns += td - tc
            fp.attention_ns += te - td
            fp.o_gemv_ns += tf - te
            fp.attn_reduce_ns += tg - tf
            fp.mlp_norm_ns += th - tg
            fp.gu_gemv_ns += ti - th
            fp.silu_ns += tj - ti
            fp.down_gemv_ns += tk - tj
            fp.mlp_reduce_ns += tl - tk

            _ = layer_idx

        v_scale_lease^.release()
        post_scale_lease^.release()
        act_scale_lease^.release()

        var tfinal = Int(perf_counter_ns())

        # --- Final norm + LM head (host rank only, tied embeddings) ---
        var last_row_off = (seq_len - 1) * C.HIDDEN * M.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[M.X_MAIN](host.x_main(seq_len).ptr + last_row_off, 1)
        rmsnorm(last_hidden, host.weight[M.FINAL_NORM](), last_hidden, self.pools[0]).join()

        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = host.scratch_view[M.LOGITS](logit_lease, 1)
        float_gemv(last_hidden, host.weight[M.EMBED](), logit_view, self.pools[0]).join()

        fp.final_ns = Int(perf_counter_ns()) - tfinal
        self.profile.accumulate(fp)

        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^,
        )

    # =========================================================================
    # === DEBUG === Stepped forward pass for layer-by-layer comparison
    # =========================================================================

    def debug_embed(mut self, tokens_ptr: Int, seq_len: Int):
        debug_assert(seq_len >= 0 and seq_len <= C.MAX_SEQ_LEN,
            "debug_embed: seq_len exceeds MAX_SEQ_LEN")
        comptime M = Self.M
        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), self.pools[0]).join()
        ring_broadcast[M.X_MAIN, Self.tp](
            host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)

    def debug_layer_attn(mut self, layer_idx: Int, seq_len: Int, pos: Int):
        debug_assert(seq_len >= 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "debug_layer_attn: sequence range exceeds MAX_SEQ_LEN")
        comptime M = Self.M
        comptime L = M.LAYER
        comptime QKV_N = L.QKV_N
        comptime Q_ROWS = C.HIDDEN // Self.tp
        comptime KV_ROWS = C.KV_HIDDEN // Self.tp
        comptime Q_COLS = Q_ROWS
        comptime O_K = Q_COLS
        comptime MAX_WORKERS = 64
        comptime WORK_F32 = C.HIDDEN * MAX_WORKERS
        comptime ACT_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
        comptime ATTN_SCRATCH_BYTES = C.MAX_SEQ_LEN * Q_COLS * (size_of[Float32]() + 1) + 1024 * 1024

        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var act_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
        var v_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
        var ls = self.layer_scales[layer_idx]
        var kv_cfg_buf = InlineArray[KVWriteConfig, Self.tp](
            fill=KVWriteConfig(0, 0, 0, 0, 0, 0, 0, 0, 0, Float32(0)))
        var kv_cfg_base = Int(UnsafePointer(to=kv_cfg_buf).bitcast[KVWriteConfig]())

        var qkv = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * QKV_N]()
        var act_work = self.scratch.borrow[UInt8, ACT_WORK_BYTES]()
        var act_i8_off = act_work.offset
        var work_off = act_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

        @parameter
        def do_attn_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            return rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                rv.x_main(seq_len).ptr, sb + act_i8_off, sb + work_off,
                sb + act_scale_lease.offset, Float32(C.RMS_NORM_EPS), seq_len, pool)
        ranks.parallel[do_attn_norm]()

        @parameter
        def do_qkv_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            return int8_gemv[QKV_N, C.HIDDEN](
                sb + act_i8_off,
                rv.layer_weight[L.Q_PROJ](layer_idx).ptr,
                rv.layer_weight[L.Q_COLSUM](layer_idx).ptr,
                rv.layer_weight[L.Q_ROW_SCALE](layer_idx).ptr,
                sb + qkv.offset, seq_len,
                sb + act_scale_lease.offset,
                pool)
        ranks.parallel[do_qkv_gemv]()

        act_work^.release()

        @parameter
        def do_kv_write[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            var cfg_ptr = UnsafePointer[KVWriteConfig, MutAnyOrigin](
                unsafe_from_address=kv_cfg_base + rank * size_of[KVWriteConfig]())
            cfg_ptr[] = KVWriteConfig(
                sb + qkv.offset + Q_ROWS * 2,
                sb + qkv.offset + (Q_ROWS + KV_ROWS) * 2,
                QKV_N, QKV_N,
                rv.rope_cos().ptr, rv.rope_sin().ptr,
                rv.kv_base(layer_idx),
                pos, seq_len, ls.v_layer_scale)
            pool.dispatch[KVWriteConfig,
                kv_write_worker[C.HEAD_DIM, M.LOCAL_KV_HEADS, C.MAX_SEQ_LEN, M.LOCAL_HEADS]](
                cfg_ptr, 1)
            return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=pool))))
        ranks.parallel[do_kv_write]()

        var attn = self.scratch.borrow[UInt8, ATTN_SCRATCH_BYTES]()

        @parameter
        def do_attention[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            var kvc = rv.kv_cache(layer_idx)
            var q_view = DynView[M.Q_VIEW](sb + qkv.offset, seq_len)
            if seq_len == 1:
                var wpg_hint = pool.get_capacity() // M.LOCAL_KV_HEADS
                amx_decode[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN, M.LOCAL_HEADS](
                    q_view, QKV_N, kvc,
                    rv.rope_cos(), rv.rope_sin(),
                    sb + attn.offset, pos, ls.v_layer_scale,
                    wpg_hint, pool).join()
                comptime DECODE_BLOCK_N = 512
                var context = pos + 1
                var max_blocks = (context + DECODE_BLOCK_N - 1) // DECODE_BLOCK_N
                var wpg = min(min(wpg_hint, max_blocks), pool.get_capacity() // M.LOCAL_KV_HEADS)
                var vagg_scale = ls.v_layer_scale / (Float32(255) * Float32(127))
                var merge_args = InlineArray[DecodeMergeArgs, 1](
                    fill=DecodeMergeArgs(sb + attn.offset, wpg, Int(vagg_scale.to_bits())))
                pool.dispatch[DecodeMergeArgs,
                    decode_merge_worker[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM]](
                    UnsafePointer(to=merge_args[0]), 1)
                return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=pool))))
            else:
                return amx_prefill[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN, M.LOCAL_HEADS](
                    q_view, QKV_N, kvc,
                    rv.rope_cos(), rv.rope_sin(),
                    sb + attn.offset, pos, ls.v_layer_scale, pool)
        ranks.parallel[do_attention]()

        @parameter
        def do_o_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            var v_sc = UnsafePointer[Float32, MutAnyOrigin](
                unsafe_from_address=sb + v_scale_lease.offset)
            for m in range(seq_len):
                v_sc[m] = ls.v_layer_scale
            var attn_i8_addr = sb + attn.offset
            if seq_len > 1:
                attn_i8_addr += seq_len * Q_COLS * size_of[Float32]()
            return int8_gemv[C.HIDDEN, O_K](
                attn_i8_addr,
                rv.layer_weight[L.O_PROJ](layer_idx).ptr,
                rv.layer_weight[L.O_COLSUM](layer_idx).ptr,
                rv.layer_weight[L.O_ROW_SCALE](layer_idx).ptr,
                rv.x_residual(seq_len).ptr, seq_len,
                sb + v_scale_lease.offset,
                pool)
        ranks.parallel[do_o_gemv]()

        qkv^.release()
        attn^.release()

        ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        @parameter
        def do_attn_res_add[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var add_args = InlineArray[ElemAddArgs, 1](
                fill=ElemAddArgs(
                    rv.x_main(seq_len).ptr,
                    rv.x_residual(seq_len).ptr,
                    rv.x_main(seq_len).ptr,
                    seq_len * C.HIDDEN))
            pool.dispatch[ElemAddArgs, elem_add_worker](
                UnsafePointer(to=add_args[0]), 1)
            return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=pool))))
        ranks.parallel[do_attn_res_add]()

        v_scale_lease^.release()
        act_scale_lease^.release()

    def debug_layer_mlp(mut self, layer_idx: Int, seq_len: Int, pos: Int):
        debug_assert(seq_len >= 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "debug_layer_mlp: sequence range exceeds MAX_SEQ_LEN")
        comptime M = Self.M
        comptime L = M.LAYER
        comptime GATE_ROWS = C.INTERMEDIATE // Self.tp
        comptime DOWN_K = GATE_ROWS
        comptime MAX_WORKERS = 64
        comptime WORK_F32 = C.HIDDEN * MAX_WORKERS
        comptime MLP_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()

        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var act_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
        var post_scale_lease = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()

        var mlp_work = self.scratch.borrow[UInt8, MLP_WORK_BYTES]()
        var mlp_i8_off = mlp_work.offset
        var mlp_work_off = mlp_work.offset + C.MAX_SEQ_LEN * C.HIDDEN

        @parameter
        def do_mlp_norm[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            return rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
                rv.x_main(seq_len).ptr, sb + mlp_i8_off, sb + mlp_work_off,
                sb + act_scale_lease.offset, Float32(C.RMS_NORM_EPS), seq_len, pool)
        ranks.parallel[do_mlp_norm]()

        var post_i8 = self.scratch.borrow[Scalar[DType.int8], C.MAX_SEQ_LEN * GATE_ROWS]()

        @parameter
        def do_fused_gu_silu[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            return fused_gu_silu[GATE_ROWS, C.HIDDEN, FWHT_BLOCK](
                sb + mlp_i8_off,
                rv.layer_weight[L.GATE_PROJ](layer_idx).ptr,
                rv.layer_weight[L.GATE_COLSUM](layer_idx).ptr,
                rv.layer_weight[L.GATE_ROW_SCALE](layer_idx).ptr,
                rv.layer_weight[L.UP_PROJ](layer_idx).ptr,
                rv.layer_weight[L.UP_COLSUM](layer_idx).ptr,
                rv.layer_weight[L.UP_ROW_SCALE](layer_idx).ptr,
                sb + post_i8.offset,
                sb + post_scale_lease.offset,
                seq_len, sb + act_scale_lease.offset,
                pool)
        ranks.parallel[do_fused_gu_silu]()

        mlp_work^.release()

        @parameter
        def do_down_gemv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var sb = rv.scratch_base()
            return int8_gemv[C.HIDDEN, DOWN_K](
                sb + post_i8.offset,
                rv.layer_weight[L.DOWN_PROJ](layer_idx).ptr,
                rv.layer_weight[L.DOWN_COLSUM](layer_idx).ptr,
                rv.layer_weight[L.DOWN_ROW_SCALE](layer_idx).ptr,
                rv.x_residual(seq_len).ptr, seq_len,
                sb + post_scale_lease.offset,
                pool)
        ranks.parallel[do_down_gemv]()

        post_i8^.release()

        ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        @parameter
        def do_mlp_res_add[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence[BurstPool[]]:
            var add_args = InlineArray[ElemAddArgs, 1](
                fill=ElemAddArgs(
                    rv.x_main(seq_len).ptr,
                    rv.x_residual(seq_len).ptr,
                    rv.x_main(seq_len).ptr,
                    seq_len * C.HIDDEN))
            pool.dispatch[ElemAddArgs, elem_add_worker](
                UnsafePointer(to=add_args[0]), 1)
            return PoolFence[BurstPool[]](UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=pool))))
        ranks.parallel[do_mlp_res_add]()

        post_scale_lease^.release()
        act_scale_lease^.release()

    def debug_x_main_ptr(self, seq_len: Int) -> Int:
        return RankView[Self.tp](self.bases[0]).x_main(seq_len).ptr

    def debug_set_x_main(mut self, src_ptr: Int, seq_len: Int):
        debug_assert(seq_len >= 0 and seq_len <= C.MAX_SEQ_LEN,
            "debug_set_x_main: seq_len exceeds MAX_SEQ_LEN")
        comptime M = Self.M
        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](
                unsafe_from_address=host.x_main(seq_len).ptr),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=seq_len * C.HIDDEN * 2)
        ring_broadcast[M.X_MAIN, Self.tp](
            host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)
