"""SmolLM2-135M Hadamard-rotated int8 quantization.

Projection weights are FWHT-rotated and quantized to I8 with per-row F32
scales. RMSNorm gamma is absorbed into projection weights offline — norm
weights are consumed during quantization and absent from the quantized
file. At runtime, RMSNorm is just x / rms(x).

KV cache stores pre-rotated u8 (for native vpdpbusd) with per-head F32 scales. K and V are
written directly from the projection gemm epilogue — no bf16 intermediate.
RoPE on K is fused into the cache write. RoPE on Q is fused into the
attention kernel.

Layer flow (12 steps):
     1. rms_fwht_quantize(x)              → act_i8, act_sc
     2. int8_gemm(act_i8, Q')             → q (bf16, only projection materialized)
     3. int8_gemm_k_to_cache(act_i8, K')  → K cache (fused gemm + RoPE + FWHT + quant)
     4. int8_gemm_v_to_cache(act_i8, V')  → V cache (fused gemm + FWHT + quant)
     5. int8_gqa_attention(q, kv_cache)   → act_i8, act_sc (RoPE on Q fused, int8 out)
     6. int8_gemm(act_i8, O')            → o_out
     7. allreduce, residual add
     8. rms_fwht_quantize(x')            → act_i8, act_sc
     9. int8_gemm_gate_up(act_i8, G', U') → gate, up (one activation read)
    10. silu_fwht_quantize(gate, up)      → act_i8, act_sc
    11. int8_gemm(act_i8, DOWN')          → down_out
    12. allreduce, residual add

Usage:
    from modeling.smollm2_hadquant_tp import HadQuantTPModel, SmolLM2HadQuant

    # Quantize (offline, once)
    HadQuantTPModel[1].quantize(source_path, output_path)

    # Load + run (runtime)
    var model = SmolLM2HadQuant[1].load(quantized_path)
    var logits = model.forward(tokens_ptr, seq_len, pos)
"""

from std.pathlib import Path
from std.memory import UnsafePointer, memcpy
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
from kernels.vnni import VnniPacked
from kernels.kernel_ops import (
    elem_add, embed_lookup,
    init_rope_tables,
    PoolFence, parallel_for,
)
from kernels.reductions import ring_allreduce, ring_broadcast
from kernels.profiler import Profiler
from modeling.loader import load_safetensors
from quant.engine import quantize as quantize_impl
from experimental.linear_borrow_pool import ScratchPool, ScratchLease
from experimental.hadquant_impl import (
    rms_fwht_quantize, silu_fwht_quantize,
    int8_gemm, int8_gemm_k_to_cache, int8_gemm_v_to_cache,
    int8_gemm_gate_up, rms_norm_no_gamma,
    compute_column_sum,
)
from experimental.hadquant_attn import int8_gqa_attention
from experimental.hadquant_kv_cache import HadQuantKVCache


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
# Layer spec
# =============================================================================


struct HadQuantTPLayer[tp: Int]:

    # --- Norms: absorbed during quantization, absent from output ---

    comptime INPUT_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, 0, "input_layernorm.weight", IsAbsorbed]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, 0, "post_attention_layernorm.weight", IsAbsorbed]

    # --- Projection weights: I8, VNNI-packed at load time ---

    comptime Q_PROJ    = PlacedSlot[I8, RowShard, C.HIDDEN, C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime K_PROJ    = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime V_PROJ    = PlacedSlot[I8, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime O_PROJ    = PlacedSlot[I8, ColShard, C.HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight", IsQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime GATE_PROJ = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime UP_PROJ   = PlacedSlot[I8, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight", IsGammaQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime DOWN_PROJ = PlacedSlot[I8, ColShard, C.HIDDEN, C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight", IsQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]

    # --- Scales: F32, one per row ---

    comptime Q_SCALE    = PlacedSlot[F32, RowShard, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "self_attn.q_proj.weight_scale"]
    comptime K_SCALE    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.Q_SCALE](), "self_attn.k_proj.weight_scale"]
    comptime V_SCALE    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.K_SCALE](), "self_attn.v_proj.weight_scale"]
    comptime O_SCALE    = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.V_SCALE](), "self_attn.o_proj.weight_scale"]
    comptime GATE_SCALE = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.O_SCALE](), "mlp.gate_proj.weight_scale"]
    comptime UP_SCALE   = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.GATE_SCALE](), "mlp.up_proj.weight_scale"]
    comptime DOWN_SCALE = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.UP_SCALE](), "mlp.down_proj.weight_scale"]

    # --- Column sums: F32, one per row, for VNNI u8/i8 bias correction ---
    #     Computed at load time from packed i8 weights: colsum[n] = sum_k weight[n,k]
    #     Used by int8_gemm epilogue: corrected = raw_acc - 128 * colsum[n]

    comptime Q_COLSUM    = PlacedSlot[F32, RowShard, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_SCALE](), "self_attn.q_proj.weight_colsum"]
    comptime K_COLSUM    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.Q_COLSUM](), "self_attn.k_proj.weight_colsum"]
    comptime V_COLSUM    = PlacedSlot[F32, RowShard, C.KV_HIDDEN, 1, Self.tp, next_offset[Self.K_COLSUM](), "self_attn.v_proj.weight_colsum"]
    comptime O_COLSUM    = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.V_COLSUM](), "self_attn.o_proj.weight_colsum"]
    comptime GATE_COLSUM = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.O_COLSUM](), "mlp.gate_proj.weight_colsum"]
    comptime UP_COLSUM   = PlacedSlot[F32, RowShard, C.INTERMEDIATE, 1, Self.tp, next_offset[Self.GATE_COLSUM](), "mlp.up_proj.weight_colsum"]
    comptime DOWN_COLSUM = PlacedSlot[F32, ColShard, C.HIDDEN, 1, Self.tp, next_offset[Self.UP_COLSUM](), "mlp.down_proj.weight_colsum"]

    comptime STRIDE = next_offset[Self.DOWN_COLSUM]()

    # --- KV cache: per-head contiguous via HadQuantKVCache ---

    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp
    comptime KV_CACHE = HadQuantKVCache[C.MAX_SEQ_LEN, C.HEAD_DIM, Self.LOCAL_KV_HEADS]

    # --- Iteration: norms before the projections they feed ---

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
        func[Self.Q_SCALE](prefix, base)
        func[Self.K_SCALE](prefix, base)
        func[Self.V_SCALE](prefix, base)
        func[Self.O_SCALE](prefix, base)
        func[Self.GATE_SCALE](prefix, base)
        func[Self.UP_SCALE](prefix, base)
        func[Self.DOWN_SCALE](prefix, base)

    @staticmethod
    def cache_bytes() -> Int:
        """Total KV cache bytes per layer: K cache + V cache."""
        return 2 * Self.KV_CACHE.TOTAL_BYTES


# =============================================================================
# Model spec
# =============================================================================


struct HadQuantTPModel[tp: Int](WeightIterable):
    comptime LAYER = HadQuantTPLayer[Self.tp]

    comptime LAYERS_OFF = 0
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = C.NUM_LAYERS * Self.LAYER.STRIDE

    comptime LOCAL_HEADS = C.NUM_HEADS // Self.tp
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp

    # --- Activation slots ---

    comptime ROPE_HALF = C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    # Typed DynView slots — used to construct views over borrowed scratch.
    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime ACT_I8_HIDDEN = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime ACT_I8_INTERMEDIATE = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime ACT_SCALE = Slot[F32, Replicated, C.MAX_SEQ_LEN, 1, Self.tp]

    # Scratch capacity: derived from the peak phase (MLP > attention).
    # The pool is a bump allocator — cumulative sum of all borrows in a phase.
    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        """Peak scratch bytes across both phases of one layer.

        Each phase creates a fresh ScratchPool. The pool is a bump
        allocator (no reclaim), so the peak is the cumulative sum of
        all borrows within the largest phase.

        Attention phase borrows (in order):
            act_i8:      MAX_SEQ_LEN * HIDDEN         (int8)
            act_scale:   MAX_SEQ_LEN * 4               (f32, 1 per row)
            rms_work:    HIDDEN * 4                     (f32, per-row temp)
            q:           MAX_SEQ_LEN * HIDDEN/tp * 2   (bf16)
            kv_scratch:  KV_HIDDEN/tp * 4              (f32, gemm intermediate)

        MLP phase borrows (in order):
            act_i8:      MAX_SEQ_LEN * HIDDEN          (int8)
            act_scale:   MAX_SEQ_LEN * 4                (f32)
            rms_work:    HIDDEN * 4                      (f32, per-row temp)
            gate:        MAX_SEQ_LEN * INTERMEDIATE/tp * 2  (bf16)
            up:          MAX_SEQ_LEN * INTERMEDIATE/tp * 2  (bf16)
            act_i8_inter: MAX_SEQ_LEN * INTERMEDIATE   (int8)
            act_scale_inter: MAX_SEQ_LEN * 4            (f32)
            silu_work:   INTERMEDIATE * 4               (f32, per-row temp)
        """
        comptime S = C.MAX_SEQ_LEN
        comptime H = C.HIDDEN
        comptime I = C.INTERMEDIATE
        comptime KV = C.KV_HIDDEN
        comptime TP = Self.tp

        comptime attn_peak = (
            S * H              # act_i8
            + S * 4            # act_scale
            + H * 4            # rms_work
            + S * (H // TP) * 2  # q (bf16)
            + (KV // TP) * 4   # kv_scratch
        )

        comptime mlp_peak = (
            S * H              # act_i8
            + S * 4            # act_scale
            + H * 4            # rms_work
            + S * (I // TP) * 2  # gate (bf16)
            + S * (I // TP) * 2  # up (bf16)
            + S * I            # act_i8_inter
            + S * 4            # act_scale_inter
            + I * 4            # silu_work
        )

        comptime if attn_peak > mlp_peak:
            return attn_peak
        else:
            return mlp_peak

    # --- State layout ---

    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # NodeLocal weights (host arena)
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
        return quantize_impl[Self, DType.float32, DType.int8, FWHT_BLOCK](
            source_path, output_path,
        )


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankView[tp: Int]:
    comptime M = HadQuantTPModel[Self.tp]
    comptime L = Self.M.LAYER
    var base: Int

    def __init__(out self, arena_base: Int):
        self.base = arena_base

    def weight_base(self) -> Int:
        return self.base

    def state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES

    # --- Layer weights + scales ---

    def layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE)

    def weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())

    # --- KV cache ---

    def kv_base(self, layer: Int) -> Int:
        return self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    comptime KVCache = Self.L.KV_CACHE

    def k_cache(self, layer: Int) -> Self.KVCache:
        """K cache for this layer."""
        return Self.KVCache(self.kv_base(layer))

    def v_cache(self, layer: Int) -> Self.KVCache:
        """V cache for this layer (after K cache in memory)."""
        return Self.KVCache(self.kv_base(layer) + Self.KVCache.TOTAL_BYTES)

    # --- bf16 activation views ---

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

    # --- RoPE tables ---

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

    def parallel[body: def[rank: Int] (RankView[Self.tp], mut BurstPool[]) capturing -> PoolFence](self):
        @parameter
        def dispatch[rank: Int]() -> PoolFence:
            var rv = RankView[Self.tp](self.bases[rank])
            return body[rank](rv, self.pool_ptrs[rank][])
        parallel_for[Self.tp, dispatch]()

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
# Weight packing
# =============================================================================


def pack_weights[M: WeightIterable](
    arena_base: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    @parameter
    def pack_if_needed[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        comptime if conforms_to(T, Quantizable):
            comptime weight_bytes = T.ROWS * T.COLS * T.ELEMENT_BYTES
            var arena_slot = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=arena_base + base + T.OFFSET
            )
            memcpy(dest=scratch, src=arena_slot, count=weight_bytes)
            T.PACK_FN(scratch, arena_slot, T.ROWS, T.COLS)

    M.for_each_weight[pack_if_needed]()


def init_column_sums[tp: Int](arena_base: Int):
    """Compute column sums for all projection weights across all layers."""
    comptime L = HadQuantTPLayer[tp]
    comptime M = HadQuantTPModel[tp]
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
# Logits view
# =============================================================================
# Loaded model + forward pass
# =============================================================================


struct SmolLM2HadQuant[tp: Int](Movable):
    comptime M = HadQuantTPModel[Self.tp]

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def __init__(out self, var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
                var pools: HeapMoveArray[BurstPool[]]):
        self.bases = InlineArray[Int, Self.tp](fill=0)
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
        )
        self.scratch = ScratchPool(Self.M.SCRATCH_CAPACITY)
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
    def load(path: Path) -> Optional[Self]:
        comptime assert C.NUM_HEADS % Self.tp == 0, "TP must evenly divide NUM_HEADS"
        comptime assert C.NUM_KV_HEADS % Self.tp == 0, "TP must evenly divide NUM_KV_HEADS"
        comptime assert C.INTERMEDIATE % Self.tp == 0, "TP must evenly divide INTERMEDIATE"

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)
        comptime host_rank = 0

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == host_rank else Self.M.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("hadquant: arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_safetensors[Self.M](path, arena_bases, host_index=host_rank)
        if not result:
            print("hadquant: weight loading failed")
            return None

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        for rank in range(Self.tp):
            var base = Int(arenas[rank].base)
            var scratch = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=base + Self.M.DISTRIBUTED_BYTES + Self.M.SCRATCH_OFF
            )
            init_column_sums[Self.tp](base)  # before packing — weights still row-major
            pack_weights[Self.M](base, scratch)

        var pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var model = Self(arenas^, pools^)

        for rank in range(Self.tp):
            var rv = model.rank(rank)
            init_rope_tables(rv.rope_cos(), rv.rope_sin(), Float64(C.ROPE_THETA))

        return model^

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------

    def forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[C.VOCAB_SIZE]:
        comptime M = Self.M
        comptime L = M.LAYER
        var prof = Profiler(profile)

        var ranks = Ranks[Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)

        # --- Embed (host rank, then broadcast) ---
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), ranks.pool_ptrs[0][]).join()
        ring_broadcast[M.X_MAIN, Self.tp](host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        for layer_idx in range(C.NUM_LAYERS):

            # === Attention block ===

            # Borrow scratch offsets — same for every rank
            var act_i8 = self.scratch.borrow[Scalar[DType.int8], C.MAX_SEQ_LEN * C.HIDDEN]()
            var act_scale = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
            var rms_work = self.scratch.borrow[Float32, C.HIDDEN]()

            # 1. rms_fwht_quantize
            @parameter
            def do_rms_fwht_qkv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rms_fwht_quantize[FWHT_BLOCK](
                    rv.x_main(seq_len),
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.scratch_ptr[Float32](rms_work),
                    pool, Float32(C.RMS_NORM_EPS),
                )
            ranks.parallel[do_rms_fwht_qkv]()
            rms_work^.release()

            # 2. Q gemm → bf16
            var q = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * M.Q_VIEW.COLS]()

            @parameter
            def do_q[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.Q_PROJ, L.Q_SCALE, L.Q_COLSUM, M.ACT_I8_HIDDEN, M.ACT_SCALE, M.Q_VIEW](
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.layer_weight[L.Q_PROJ](layer_idx), rv.layer_weight[L.Q_SCALE](layer_idx), rv.layer_weight[L.Q_COLSUM](layer_idx),
                    rv.scratch_view[M.Q_VIEW](q, seq_len), pool,
                )
            ranks.parallel[do_q]()

            # 3-4. K/V gemm → cache
            var kv_work = self.scratch.borrow[Float32, M.LOCAL_KV_HEADS * C.HEAD_DIM]()

            @parameter
            def do_k_to_cache[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_k_to_cache[FWHT_BLOCK, C.HEAD_DIM, M.LOCAL_KV_HEADS, C.MAX_SEQ_LEN](
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.layer_weight[L.K_PROJ](layer_idx), rv.layer_weight[L.K_SCALE](layer_idx), rv.layer_weight[L.K_COLSUM](layer_idx),
                    rv.k_cache(layer_idx),
                    rv.rope_cos(), rv.rope_sin(),
                    pos, rv.scratch_ptr[Float32](kv_work), pool,
                )
            ranks.parallel[do_k_to_cache]()

            @parameter
            def do_v_to_cache[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_v_to_cache[FWHT_BLOCK, C.HEAD_DIM, M.LOCAL_KV_HEADS, C.MAX_SEQ_LEN](
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.layer_weight[L.V_PROJ](layer_idx), rv.layer_weight[L.V_SCALE](layer_idx), rv.layer_weight[L.V_COLSUM](layer_idx),
                    rv.v_cache(layer_idx),
                    pos, rv.scratch_ptr[Float32](kv_work), pool,
                )
            ranks.parallel[do_v_to_cache]()
            kv_work^.release()

            # 5. Attention → int8 output (reuses act_i8/act_scale)
            @parameter
            def do_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gqa_attention[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN](
                    rv.scratch_view[M.Q_VIEW](q, seq_len),
                    rv.k_cache(layer_idx), rv.v_cache(layer_idx),
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.rope_cos(), rv.rope_sin(),
                    pos, pool,
                )
            ranks.parallel[do_attn]()
            q^.release()

            # 6. O projection
            @parameter
            def do_o[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.O_PROJ, L.O_SCALE, L.O_COLSUM, M.ACT_I8_HIDDEN, M.ACT_SCALE, M.X_RESIDUAL](
                    rv.scratch_view[M.ACT_I8_HIDDEN](act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale, seq_len),
                    rv.layer_weight[L.O_PROJ](layer_idx), rv.layer_weight[L.O_SCALE](layer_idx), rv.layer_weight[L.O_COLSUM](layer_idx),
                    rv.x_residual(seq_len), pool,
                )
            ranks.parallel[do_o]()
            act_scale^.release()
            act_i8^.release()

            # 7. Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_res_add(rv: RankView[Self.tp]):
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))
            ranks.each[do_res_add]()

            # === MLP block ===

            # 8. rms_fwht_quantize
            var mlp_act_i8 = self.scratch.borrow[Scalar[DType.int8], C.MAX_SEQ_LEN * C.HIDDEN]()
            var mlp_act_scale = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
            var mlp_rms_work = self.scratch.borrow[Float32, C.HIDDEN]()

            @parameter
            def do_rms_fwht_mlp[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rms_fwht_quantize[FWHT_BLOCK](
                    rv.x_main(seq_len),
                    rv.scratch_view[M.ACT_I8_HIDDEN](mlp_act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](mlp_act_scale, seq_len),
                    rv.scratch_ptr[Float32](mlp_rms_work),
                    pool, Float32(C.RMS_NORM_EPS),
                )
            ranks.parallel[do_rms_fwht_mlp]()
            mlp_rms_work^.release()

            # 9. Gate+Up gemm → bf16
            var gate = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * M.MLP_VIEW.COLS]()
            var up = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * M.MLP_VIEW.COLS]()

            @parameter
            def do_gate_up[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_gate_up(
                    rv.scratch_view[M.ACT_I8_HIDDEN](mlp_act_i8, seq_len),
                    rv.scratch_view[M.ACT_SCALE](mlp_act_scale, seq_len),
                    rv.layer_weight[L.GATE_PROJ](layer_idx), rv.layer_weight[L.GATE_SCALE](layer_idx), rv.layer_weight[L.GATE_COLSUM](layer_idx),
                    rv.layer_weight[L.UP_PROJ](layer_idx), rv.layer_weight[L.UP_SCALE](layer_idx), rv.layer_weight[L.UP_COLSUM](layer_idx),
                    rv.scratch_view[M.MLP_VIEW](gate, seq_len),
                    rv.scratch_view[M.MLP_VIEW](up, seq_len),
                    pool,
                )
            ranks.parallel[do_gate_up]()
            mlp_act_scale^.release()
            mlp_act_i8^.release()

            # 10. silu_fwht_quantize(gate, up) → act_i8, act_sc
            var act_i8_inter = self.scratch.borrow[Scalar[DType.int8], C.MAX_SEQ_LEN * C.INTERMEDIATE]()
            var act_scale_inter = self.scratch.borrow[Float32, C.MAX_SEQ_LEN]()
            var silu_work = self.scratch.borrow[Float32, C.INTERMEDIATE]()

            @parameter
            def do_silu_fwht[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return silu_fwht_quantize[FWHT_BLOCK](
                    rv.scratch_view[M.MLP_VIEW](gate, seq_len),
                    rv.scratch_view[M.MLP_VIEW](up, seq_len),
                    rv.scratch_view[M.ACT_I8_INTERMEDIATE](act_i8_inter, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale_inter, seq_len),
                    rv.scratch_ptr[Float32](silu_work),
                    pool,
                )
            ranks.parallel[do_silu_fwht]()
            silu_work^.release()
            up^.release()
            gate^.release()

            # 11. DOWN gemm
            @parameter
            def do_down[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.DOWN_PROJ, L.DOWN_SCALE, L.DOWN_COLSUM, M.ACT_I8_INTERMEDIATE, M.ACT_SCALE, M.X_RESIDUAL](
                    rv.scratch_view[M.ACT_I8_INTERMEDIATE](act_i8_inter, seq_len),
                    rv.scratch_view[M.ACT_SCALE](act_scale_inter, seq_len),
                    rv.layer_weight[L.DOWN_PROJ](layer_idx), rv.layer_weight[L.DOWN_SCALE](layer_idx), rv.layer_weight[L.DOWN_COLSUM](layer_idx),
                    rv.x_residual(seq_len), pool,
                )
            ranks.parallel[do_down]()
            act_scale_inter^.release()
            act_i8_inter^.release()

            # 12. Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)
            ranks.each[do_res_add]()

            _ = layer_idx

        # --- Final norm + LM head ---
        rms_norm_no_gamma(host.x_main(seq_len), host.x_main(seq_len), ranks.pool_ptrs[0][]).join()

        from kernels.kernel_ops import gemm
        var last_row_off = (seq_len - 1) * C.HIDDEN * M.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[M.X_MAIN](host.x_main(seq_len).ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        gemm(last_hidden, host.weight[M.EMBED](), host.scratch_view[M.LOGITS](logit_lease, 1), ranks.pool_ptrs[0][]).join()

        prof.finish()
        prof.report()
        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^,
        )
