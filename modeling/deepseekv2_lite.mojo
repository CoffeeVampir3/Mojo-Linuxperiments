"""DeepSeek-V2-Lite with BF16 weights.

15.7B total parameters, 2.4B active per token.
27 layers: 1 dense FFN (layer 0) + 26 MoE FFN (layers 1-26).
MLA attention with compressed KV cache (576 elements per token per layer).
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32,
    RowShard, ColShard, Replicated,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    LogitsView,
)
from modeling.loader import load_weights, LoadResult
from kernels.kv_rotors import init_rope_tables
from experimental.linear_borrow_pool import ScratchPool, ScratchLease


# =============================================================================
# Model configuration
# =============================================================================


struct DeepSeekV2LiteConfig:
    # Core dimensions
    comptime HIDDEN = 2048
    comptime NUM_LAYERS = 27
    comptime NUM_HEADS = 16

    # MLA head dimensions
    comptime QK_NOPE_HEAD_DIM = 128
    comptime QK_ROPE_HEAD_DIM = 64
    comptime V_HEAD_DIM = 128
    comptime KV_LORA_RANK = 512

    # FFN dimensions
    comptime DENSE_INTERMEDIATE = 10944
    comptime MOE_INTERMEDIATE = 1408
    comptime N_SHARED_EXPERTS = 2
    comptime SHARED_INTERMEDIATE = Self.MOE_INTERMEDIATE * Self.N_SHARED_EXPERTS
    comptime N_ROUTED_EXPERTS = 64
    comptime N_EXPERTS_PER_TOK = 6
    comptime FIRST_K_DENSE = 1

    # Global
    comptime VOCAB_SIZE = 102400
    comptime MAX_SEQ_LEN = 4096
    comptime ROPE_THETA = 10000.0
    comptime RMS_NORM_EPS = 1e-6
    comptime TIE_EMBEDDINGS = False

    # Derived MLA dimensions
    comptime Q_HEAD_DIM = Self.QK_NOPE_HEAD_DIM + Self.QK_ROPE_HEAD_DIM
    comptime Q_PROJ_DIM = Self.NUM_HEADS * Self.Q_HEAD_DIM
    comptime KV_A_DIM = Self.KV_LORA_RANK + Self.QK_ROPE_HEAD_DIM
    comptime KV_B_DIM = Self.NUM_HEADS * (Self.QK_NOPE_HEAD_DIM + Self.V_HEAD_DIM)
    comptime O_PROJ_DIM = Self.NUM_HEADS * Self.V_HEAD_DIM
    comptime KV_CACHE_DIM = Self.KV_LORA_RANK + Self.QK_ROPE_HEAD_DIM
    comptime ROPE_HALF = Self.QK_ROPE_HEAD_DIM // 2

    # YaRN scaling parameters
    comptime YARN_FACTOR = 40.0
    comptime YARN_ORIGINAL_MAX_POS = 4096
    comptime YARN_BETA_FAST = 32.0
    comptime YARN_BETA_SLOW = 1.0
    comptime YARN_MSCALE_ALL_DIM = 0.707


comptime C = DeepSeekV2LiteConfig


# =============================================================================
# Per-expert weight spec (offsets relative to expert base)
# =============================================================================


struct ExpertWeights[tp: Int]:
    comptime GATE_PROJ = PlacedSlot[BF16, Replicated, C.MOE_INTERMEDIATE, C.HIDDEN, Self.tp, 0, "gate_proj.weight"]
    comptime UP_PROJ   = PlacedSlot[BF16, Replicated, C.MOE_INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "up_proj.weight"]
    comptime DOWN_PROJ = PlacedSlot[BF16, Replicated, C.HIDDEN, C.MOE_INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "down_proj.weight"]
    comptime STRIDE    = next_offset[Self.DOWN_PROJ]()

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int, Int) capturing -> None,
    ](prefix: String, base: Int, target_rank: Int):
        func[Self.GATE_PROJ](prefix, base, target_rank)
        func[Self.UP_PROJ](prefix, base, target_rank)
        func[Self.DOWN_PROJ](prefix, base, target_rank)


# =============================================================================
# Dense layer (layer 0): MLA attention + dense SwiGLU FFN
# =============================================================================


struct DenseLayer[tp: Int]:
    # --- MLA attention ---
    comptime Q_PROJ         = PlacedSlot[BF16, RowShard,   C.Q_PROJ_DIM,  C.HIDDEN,       Self.tp, 0,                              "self_attn.q_proj.weight"]
    comptime KV_A_PROJ      = PlacedSlot[BF16, Replicated, C.KV_A_DIM,    C.HIDDEN,       Self.tp, next_offset[Self.Q_PROJ](),      "self_attn.kv_a_proj_with_mqa.weight"]
    comptime KV_A_NORM      = PlacedSlot[BF16, Replicated, C.KV_LORA_RANK, 1,             Self.tp, next_offset[Self.KV_A_PROJ](),   "self_attn.kv_a_layernorm.weight"]
    comptime KV_B_PROJ      = PlacedSlot[BF16, RowShard,   C.KV_B_DIM,    C.KV_LORA_RANK, Self.tp, next_offset[Self.KV_A_NORM](),   "self_attn.kv_b_proj.weight"]
    comptime O_PROJ         = PlacedSlot[BF16, ColShard,   C.HIDDEN,      C.O_PROJ_DIM,   Self.tp, next_offset[Self.KV_B_PROJ](),   "self_attn.o_proj.weight"]
    comptime INPUT_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,              Self.tp, next_offset[Self.O_PROJ](),      "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,              Self.tp, next_offset[Self.INPUT_NORM](),  "post_attention_layernorm.weight"]

    # --- Dense FFN ---
    comptime GATE_PROJ = PlacedSlot[BF16, RowShard, C.DENSE_INTERMEDIATE, C.HIDDEN,             Self.tp, next_offset[Self.POST_ATTN_NORM](), "mlp.gate_proj.weight"]
    comptime UP_PROJ   = PlacedSlot[BF16, RowShard, C.DENSE_INTERMEDIATE, C.HIDDEN,             Self.tp, next_offset[Self.GATE_PROJ](),      "mlp.up_proj.weight"]
    comptime DOWN_PROJ = PlacedSlot[BF16, ColShard, C.HIDDEN,             C.DENSE_INTERMEDIATE,  Self.tp, next_offset[Self.UP_PROJ](),        "mlp.down_proj.weight"]

    comptime STRIDE = next_offset[Self.DOWN_PROJ]()

    # MLA KV cache: compressed latent + RoPE key per position
    comptime CKV_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_LORA_RANK,    Self.tp]
    comptime KR_CACHE  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.QK_ROPE_HEAD_DIM, Self.tp]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base, -1)
        func[Self.KV_A_PROJ](prefix, base, -1)
        func[Self.KV_A_NORM](prefix, base, -1)
        func[Self.KV_B_PROJ](prefix, base, -1)
        func[Self.O_PROJ](prefix, base, -1)
        func[Self.INPUT_NORM](prefix, base, -1)
        func[Self.POST_ATTN_NORM](prefix, base, -1)
        func[Self.GATE_PROJ](prefix, base, -1)
        func[Self.UP_PROJ](prefix, base, -1)
        func[Self.DOWN_PROJ](prefix, base, -1)

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.CKV_CACHE]() + byte_count[Self.KR_CACHE]()


# =============================================================================
# MoE layer (layers 1-26): MLA attention + router + shared + routed experts
# =============================================================================


struct MoELayer[tp: Int]:
    # --- MLA attention (same projections as DenseLayer) ---
    comptime Q_PROJ         = PlacedSlot[BF16, RowShard,   C.Q_PROJ_DIM,  C.HIDDEN,       Self.tp, 0,                              "self_attn.q_proj.weight"]
    comptime KV_A_PROJ      = PlacedSlot[BF16, Replicated, C.KV_A_DIM,    C.HIDDEN,       Self.tp, next_offset[Self.Q_PROJ](),      "self_attn.kv_a_proj_with_mqa.weight"]
    comptime KV_A_NORM      = PlacedSlot[BF16, Replicated, C.KV_LORA_RANK, 1,             Self.tp, next_offset[Self.KV_A_PROJ](),   "self_attn.kv_a_layernorm.weight"]
    comptime KV_B_PROJ      = PlacedSlot[BF16, RowShard,   C.KV_B_DIM,    C.KV_LORA_RANK, Self.tp, next_offset[Self.KV_A_NORM](),   "self_attn.kv_b_proj.weight"]
    comptime O_PROJ         = PlacedSlot[BF16, ColShard,   C.HIDDEN,      C.O_PROJ_DIM,   Self.tp, next_offset[Self.KV_B_PROJ](),   "self_attn.o_proj.weight"]
    comptime INPUT_NORM     = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,              Self.tp, next_offset[Self.O_PROJ](),      "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN,      1,              Self.tp, next_offset[Self.INPUT_NORM](),  "post_attention_layernorm.weight"]

    # --- MoE FFN ---
    comptime ROUTER      = PlacedSlot[BF16, Replicated, C.N_ROUTED_EXPERTS,  C.HIDDEN,             Self.tp, next_offset[Self.POST_ATTN_NORM](), "mlp.gate.weight"]
    comptime SHARED_GATE = PlacedSlot[BF16, Replicated, C.SHARED_INTERMEDIATE, C.HIDDEN,           Self.tp, next_offset[Self.ROUTER](),         "mlp.shared_experts.gate_proj.weight"]
    comptime SHARED_UP   = PlacedSlot[BF16, Replicated, C.SHARED_INTERMEDIATE, C.HIDDEN,           Self.tp, next_offset[Self.SHARED_GATE](),    "mlp.shared_experts.up_proj.weight"]
    comptime SHARED_DOWN = PlacedSlot[BF16, Replicated, C.HIDDEN,             C.SHARED_INTERMEDIATE, Self.tp, next_offset[Self.SHARED_UP](),     "mlp.shared_experts.down_proj.weight"]

    # Routed experts: 64 experts, each with gate/up/down projections
    comptime EXPERTS_OFF = next_offset[Self.SHARED_DOWN]()
    comptime EXPERT      = ExpertWeights[Self.tp]
    comptime STRIDE      = Self.EXPERTS_OFF + C.N_ROUTED_EXPERTS * Self.EXPERT.STRIDE

    # MLA KV cache (same layout as DenseLayer)
    comptime CKV_CACHE = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.KV_LORA_RANK,    Self.tp]
    comptime KR_CACHE  = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.QK_ROPE_HEAD_DIM, Self.tp]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base, -1)
        func[Self.KV_A_PROJ](prefix, base, -1)
        func[Self.KV_A_NORM](prefix, base, -1)
        func[Self.KV_B_PROJ](prefix, base, -1)
        func[Self.O_PROJ](prefix, base, -1)
        func[Self.INPUT_NORM](prefix, base, -1)
        func[Self.POST_ATTN_NORM](prefix, base, -1)
        func[Self.ROUTER](prefix, base, -1)
        func[Self.SHARED_GATE](prefix, base, -1)
        func[Self.SHARED_UP](prefix, base, -1)
        func[Self.SHARED_DOWN](prefix, base, -1)
        comptime for e in range(C.N_ROUTED_EXPERTS):
            var expert_prefix = prefix + "mlp.experts." + String(e) + "."
            var expert_base = base + Self.EXPERTS_OFF + e * Self.EXPERT.STRIDE
            Self.EXPERT.for_each_weight[func](expert_prefix, expert_base, e % Self.tp)

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.CKV_CACHE]() + byte_count[Self.KR_CACHE]()


# =============================================================================
# Model spec — memory layout for weights, state, and scratch
# =============================================================================


struct DSV2Model[tp: Int](WeightIterable):
    comptime DENSE = DenseLayer[Self.tp]
    comptime MOE   = MoELayer[Self.tp]

    # Weight memory: layer 0 (dense) then layers 1-26 (MoE), contiguous
    comptime LAYERS_OFF      = 0
    comptime DENSE_STRIDE    = Self.DENSE.STRIDE
    comptime MOE_LAYERS_OFF  = Self.DENSE_STRIDE
    comptime MOE_STRIDE      = Self.MOE.STRIDE
    comptime DISTRIBUTED_BYTES = Self.DENSE_STRIDE + (C.NUM_LAYERS - C.FIRST_K_DENSE) * Self.MOE_STRIDE

    # Per-rank head counts
    comptime LOCAL_HEADS = C.NUM_HEADS // Self.tp

    # Per-rank activation slots
    comptime ROPE_COS    = Slot[F32, Replicated, C.MAX_SEQ_LEN, C.ROPE_HALF, Self.tp]
    comptime ROPE_SIN    = Slot[F32, Replicated, C.MAX_SEQ_LEN, C.ROPE_HALF, Self.tp]
    comptime X_MAIN      = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN,   Self.tp]
    comptime X_RESIDUAL   = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN,   Self.tp]

    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        comptime S  = C.MAX_SEQ_LEN
        comptime TP = Self.tp

        # Attention phase: q + kv_a (latent) + attn_out
        comptime attn_peak = (
            S * (C.Q_PROJ_DIM // TP) * 2
            + S * C.KV_A_DIM * 2
            + S * (C.O_PROJ_DIM // TP) * 2
        )

        # MoE phase: shared gate+up (always active)
        comptime moe_peak = S * C.SHARED_INTERMEDIATE * 2 * 2

        # Logits (post-loop)
        comptime logit_bytes = C.VOCAB_SIZE * 2

        comptime layer_peak = attn_peak if attn_peak > moe_peak else moe_peak
        return layer_peak if layer_peak > logit_bytes else logit_bytes

    # KV cache layout (same shape for all layers)
    comptime KV_CACHE_STRIDE = Self.DENSE.cache_bytes()
    comptime KV_OFF          = 0
    comptime X_MAIN_OFF      = Self.KV_OFF + C.NUM_LAYERS * Self.KV_CACHE_STRIDE
    comptime X_RESIDUAL_OFF  = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF     = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime ROPE_COS_OFF    = Self.SCRATCH_OFF + Self.SCRATCH_CAPACITY
    comptime ROPE_SIN_OFF    = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES     = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # Host-only weights (host arena only)
    comptime HOST_ONLY_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN,     1,        Self.tp, Self.HOST_ONLY_OFF,             "model.norm.weight"]
    comptime EMBED      = PlacedSlot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]
    comptime LM_HEAD    = PlacedSlot[BF16, Replicated, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.EMBED](),      "lm_head.weight"]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int, Int) capturing -> None,
    ]():
        # Layer 0 (dense)
        Self.DENSE.for_each_weight[func]("model.layers.0.", Self.LAYERS_OFF)

        # Layers 1-26 (MoE)
        comptime for i in range(C.FIRST_K_DENSE, C.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.MOE_LAYERS_OFF + (i - C.FIRST_K_DENSE) * Self.MOE_STRIDE
            Self.MOE.for_each_weight[func](prefix, base)

        # Host-only
        func[Self.FINAL_NORM]("", 0, 0)
        func[Self.EMBED]("", 0, 0)
        func[Self.LM_HEAD]("", 0, 0)

    @staticmethod
    def arena_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    def host_arena_bytes() -> Int:
        return next_offset[Self.LM_HEAD]()


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankView[tp: Int]:
    comptime M = DSV2Model[Self.tp]
    var base: Int

    def __init__(out self, arena_base: Int):
        self.base = arena_base

    def weight_base(self) -> Int:
        return self.base

    def state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES

    def dense_layer_weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.LAYERS_OFF)

    def moe_layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.MOE_LAYERS_OFF + (layer - C.FIRST_K_DENSE) * Self.M.MOE_STRIDE)

    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    def scratch_base(self) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF

    def rope_cos(self) -> Bound[Self.M.ROPE_COS]:
        return Bound[Self.M.ROPE_COS](self.state_base() + Self.M.ROPE_COS_OFF)

    def rope_sin(self) -> Bound[Self.M.ROPE_SIN]:
        return Bound[Self.M.ROPE_SIN](self.state_base() + Self.M.ROPE_SIN_OFF)


# =============================================================================
# Loaded model
# =============================================================================


struct DeepSeekV2Lite[tp: Int](Movable):
    comptime M = DSV2Model[Self.tp]

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[BurstPool[]]
    var scratch: ScratchPool
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def __init__(
        out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[BurstPool[]],
    ):
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
    def print_memory():
        comptime host_arena = Self.M.host_arena_bytes()
        comptime arena_per_rank = Self.M.arena_bytes()
        comptime distributed = Self.M.DISTRIBUTED_BYTES
        comptime state = Self.M.STATE_BYTES

        print("DeepSeek-V2-Lite TP=" + String(Self.tp))
        print("  distributed weights:", distributed // (1024 * 1024), "MB per rank")
        print("  state (kv cache + activations):", state // (1024 * 1024), "MB per rank")
        print("  host arena (rank 0):", host_arena // (1024 * 1024), "MB")
        comptime if Self.tp > 1:
            print("  non-host arena:", arena_per_rank // (1024 * 1024), "MB each")

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.rank(0).state_base() + Self.M.SCRATCH_OFF
        )

    @staticmethod
    def discover_shards(dir_path: Path) -> List[Path]:
        """Find safetensors shard files in a directory, sorted by name."""
        var shards = List[Path]()
        try:
            for entry in dir_path.listdir():
                var name = String(entry)
                if name.endswith(".safetensors") and name.startswith("model-"):
                    shards.append(dir_path / name)
        except:
            pass

        # Sort by filename to ensure consistent ordering
        for i in range(len(shards)):
            for j in range(i + 1, len(shards)):
                if String(shards[j]) < String(shards[i]):
                    var tmp = shards[i]
                    shards[i] = shards[j]
                    shards[j] = tmp
        return shards^

    @staticmethod
    def load(dir_path: Path) -> Optional[Self]:
        """Load from a directory containing sharded safetensors files."""
        var shards = Self.discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)
        comptime host_rank = 0

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == host_rank else Self.M.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_weights[Self.M](shards, arena_bases, host_index=host_rank)
        if not result:
            print("weight loading failed")
            return None
        var loaded = result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        var pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var model = Self(arenas^, pools^)

        for rank in range(Self.tp):
            var rv = model.rank(rank)
            init_rope_tables(
                rv.rope_cos(), rv.rope_sin(),
                theta=Float64(C.ROPE_THETA),
                factor=Float64(C.YARN_FACTOR),
                original_max_pos=C.YARN_ORIGINAL_MAX_POS,
                beta_fast=Float64(C.YARN_BETA_FAST),
                beta_slow=Float64(C.YARN_BETA_SLOW),
            )

        return model^

    def forward(mut self, tokens_ptr: Int, seq_len: Int, pos: Int) -> LogitsView[C.VOCAB_SIZE]:
        """Stub forward — returns zeroed logits without computation."""
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var host = self.rank(0)
        var logit_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=host.scratch_base() + logit_lease.offset
        )
        for i in range(C.VOCAB_SIZE):
            logit_ptr[i] = Scalar[DType.bfloat16](0)
        return LogitsView[C.VOCAB_SIZE](logit_ptr, logit_lease^)

    @staticmethod
    def bf16_variance(ptr: Int, n: Int) -> Float64:
        var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
        comptime width = simd_width_of[DType.float32]()
        var acc = SIMD[DType.float64, 1](0)
        for i in range(0, n, width):
            var v = p.load[width=width](offset=i).cast[DType.float64]()
            acc += (v * v).reduce_add()
        return acc / Float64(n)

    def report_weight_variance(self):
        comptime M = Self.M
        var host = self.rank(0)
        var base = host.weight_base()

        # Layer 0 (dense)
        var q0 = bind[M.DENSE.Q_PROJ](base + M.LAYERS_OFF)
        var kva0 = bind[M.DENSE.KV_A_PROJ](base + M.LAYERS_OFF)
        var gate0 = bind[M.DENSE.GATE_PROJ](base + M.LAYERS_OFF)
        print(
            "layer  0 (dense) | q_proj var:",
            Self.bf16_variance(q0.ptr, M.DENSE.Q_PROJ.ROWS * M.DENSE.Q_PROJ.COLS),
            "| kv_a var:",
            Self.bf16_variance(kva0.ptr, M.DENSE.KV_A_PROJ.ROWS * M.DENSE.KV_A_PROJ.COLS),
            "| gate var:",
            Self.bf16_variance(gate0.ptr, M.DENSE.GATE_PROJ.ROWS * M.DENSE.GATE_PROJ.COLS),
        )

        # MoE layers
        for layer in range(C.FIRST_K_DENSE, C.NUM_LAYERS):
            var layer_base = base + M.MOE_LAYERS_OFF + (layer - C.FIRST_K_DENSE) * M.MOE_STRIDE

            var q = bind[M.MOE.Q_PROJ](layer_base)
            var kva = bind[M.MOE.KV_A_PROJ](layer_base)
            var router = bind[M.MOE.ROUTER](layer_base)
            var shared_gate = bind[M.MOE.SHARED_GATE](layer_base)

            # Sample one routed expert (expert 0)
            var e0_gate_ptr = layer_base + M.MOE.EXPERTS_OFF + 0 * M.MOE.EXPERT.STRIDE + M.MOE.EXPERT.GATE_PROJ.OFFSET
            var e0_n = M.MOE.EXPERT.GATE_PROJ.ROWS * M.MOE.EXPERT.GATE_PROJ.COLS

            print(
                "layer", layer, "(moe)   | q_proj var:",
                Self.bf16_variance(q.ptr, M.MOE.Q_PROJ.ROWS * M.MOE.Q_PROJ.COLS),
                "| kv_a var:",
                Self.bf16_variance(kva.ptr, M.MOE.KV_A_PROJ.ROWS * M.MOE.KV_A_PROJ.COLS),
                "| router var:",
                Self.bf16_variance(router.ptr, M.MOE.ROUTER.ROWS * M.MOE.ROUTER.COLS),
                "| shared var:",
                Self.bf16_variance(shared_gate.ptr, M.MOE.SHARED_GATE.ROWS * M.MOE.SHARED_GATE.COLS),
                "| expert0 var:",
                Self.bf16_variance(e0_gate_ptr, e0_n),
            )
