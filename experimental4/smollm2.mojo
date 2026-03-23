from std.pathlib import Path

from std.memory import UnsafePointer
from numa import NumaArena, NumaInfo
from threading import BurstPool

from experimental4.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32,
    ShardStrategy, RowShard, ColShard, Replicated,
    PrincipleNodeLocal,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
)
from experimental4.kernel_ops import (
    gemm, rmsnorm, embed_lookup, silu_mul, elem_add, rope, kv_cache_write,
    attention, init_rope_tables,
)
from experimental4.loader import load_safetensors
from experimental4.profiler import Profiler


trait LogitAccess:
    """Read-only access to model output logits.
    VOCAB is comptime for SIMD loop bounds. rows() is runtime
    (1 for decode, N for prefill). load_f32 upcasts from storage
    dtype so consumers always work in f32."""
    comptime DTYPE: DType
    comptime VOCAB: Int
    def rows(self) -> Int: ...
    def load_f32[width: Int](self, row: Int, offset: Int) -> SIMD[DType.float32, width]: ...


@fieldwise_init
struct LogitsView[vocab: Int, dtype: DType = DType.bfloat16](LogitAccess):
    """Non-owning, read-only view of logits in arena scratch memory.
    Valid only while the backing arena is alive."""
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab
    var ptr: Int
    var seq_len: Int

    def rows(self) -> Int:
        return self.seq_len

    def load_f32[width: Int](self, row: Int, offset: Int) -> SIMD[DType.float32, width]:
        var p = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=self.ptr
        )
        return (p + row * Self.vocab + offset).load[width=width]().cast[DType.float32]()


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


struct SmolLM2Layer[E: Encoding, tp: Int]:
    comptime C = SmolLM2Config

    comptime Q_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.HIDDEN, Self.C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight"]
    comptime K_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.KV_HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight"]
    comptime V_PROJ      = PlacedSlot[Self.E, RowShard, Self.C.KV_HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight"]
    comptime O_PROJ      = PlacedSlot[Self.E, ColShard, Self.C.HIDDEN, Self.C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight"]
    comptime GATE_PROJ   = PlacedSlot[Self.E, RowShard, Self.C.INTERMEDIATE, Self.C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight"]
    comptime UP_PROJ     = PlacedSlot[Self.E, RowShard, Self.C.INTERMEDIATE, Self.C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight"]
    comptime DOWN_PROJ   = PlacedSlot[Self.E, ColShard, Self.C.HIDDEN, Self.C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight"]
    comptime INPUT_NORM  = PlacedSlot[BF16, Replicated, Self.C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, Self.C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    comptime K_CACHE = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.KV_HIDDEN, Self.tp]

    @staticmethod
    def for_each_weight[
        func: def[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[RowShard, Self.Q_PROJ](prefix, base)
        func[RowShard, Self.K_PROJ](prefix, base)
        func[RowShard, Self.V_PROJ](prefix, base)
        func[ColShard, Self.O_PROJ](prefix, base)
        func[RowShard, Self.GATE_PROJ](prefix, base)
        func[RowShard, Self.UP_PROJ](prefix, base)
        func[ColShard, Self.DOWN_PROJ](prefix, base)
        func[Replicated, Self.INPUT_NORM](prefix, base)
        func[Replicated, Self.POST_ATTN_NORM](prefix, base)

    @staticmethod
    def weight_bytes() -> Int:
        return Self.STRIDE

    @staticmethod
    def cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()


struct SmolLM2[E: Encoding, tp: Int](WeightIterable):
    comptime C = SmolLM2Config
    comptime LAYER = SmolLM2Layer[Self.E, Self.tp]

    # --- Distributed weights (all arenas) ---
    comptime LAYERS_OFF   = 0
    comptime NUM_LAYERS   = Self.C.NUM_LAYERS
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = Self.C.NUM_LAYERS * Self.LAYER.STRIDE

    # --- State (all arenas, offsets relative to state_base) ---
    comptime ROPE_HALF = Self.C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, Self.C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, Self.C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.HIDDEN, Self.tp]
    comptime SCRATCH = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.INTERMEDIATE, Self.tp]
    comptime SCRATCH_COUNT = 3

    comptime LOGITS = Slot[BF16, Replicated, Self.C.MAX_SEQ_LEN, Self.C.VOCAB_SIZE, Self.tp]

    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + Self.C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_STRIDE = byte_count[Self.SCRATCH]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # --- NodeLocal weights (host arena only, offset past state) ---
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, Self.C.HIDDEN, 1, Self.tp, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[Self.E, PrincipleNodeLocal, Self.C.VOCAB_SIZE, Self.C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

    @staticmethod
    def for_each_weight[
        func: def[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ]():
        comptime for i in range(Self.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.LAYERS_OFF + i * Self.LAYER_STRIDE
            Self.LAYER.for_each_weight[func](prefix, base)

        func[PrincipleNodeLocal, Self.FINAL_NORM]("", 0)
        func[PrincipleNodeLocal, Self.EMBED]("", 0)

    @staticmethod
    def distributed_weight_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES

    @staticmethod
    def node_local_weight_bytes() -> Int:
        return byte_count[Self.FINAL_NORM]() + byte_count[Self.EMBED]()

    @staticmethod
    def total_state_bytes() -> Int:
        return Self.STATE_BYTES

    @staticmethod
    def arena_bytes() -> Int:
        """Non-host arena: distributed weights + state."""
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    def host_arena_bytes() -> Int:
        """Host arena: distributed weights + state + node-local weights."""
        return next_offset[Self.EMBED]()

    @staticmethod
    def kv_cache_bytes() -> Int:
        return Self.C.NUM_LAYERS * Self.KV_STRIDE

    @staticmethod
    def activation_bytes() -> Int:
        return byte_count[Self.X_MAIN]() + byte_count[Self.X_RESIDUAL]() + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE

    @staticmethod
    def precomputed_bytes() -> Int:
        return byte_count[Self.ROPE_COS]() + byte_count[Self.ROPE_SIN]()


@fieldwise_init
struct SmolLM2Loaded[E: Encoding, tp: Int](Movable):
    comptime M = SmolLM2[Self.E, Self.tp]

    var arena: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool: BurstPool[]

    @staticmethod
    def load(path: Path, node_id: Int) -> Optional[Self]:
        """Load SmolLM2 from a safetensors file. Handles all setup:
        arena allocation, weight loading, RoPE init, and thread pool creation."""
        var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](node_id, Self.M.host_arena_bytes())
        if not arena:
            print("NumaArena allocation failed for node", node_id)
            return None
        var bases = List[Int]()
        bases.append(Int(arena.base))
        var result = load_safetensors[Self.M](path, bases, host_index=0)
        if not result:
            return None
        _ = arena.prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)
        var numa = NumaInfo()
        var pool = BurstPool[].for_numa_node(numa, node_id)
        var model = Self(arena^, pool^)
        init_rope_tables(
            Bound[Self.M.ROPE_COS](model.rope_cos_ptr()),
            Bound[Self.M.ROPE_SIN](model.rope_sin_ptr()),
            Float64(Self.M.C.ROPE_THETA),
        )
        return model^

    def weight_base(self) -> Int:
        return Int(self.arena.base)

    def state_base(self) -> Int:
        return Int(self.arena.base) + Self.M.DISTRIBUTED_BYTES

    def weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())

    def layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE)

    def k_cache_ptr(self, layer: Int) -> Int:
        return self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    def v_cache_ptr(self, layer: Int) -> Int:
        return self.k_cache_ptr(layer) + byte_count[Self.M.LAYER.K_CACHE]()

    def x_main_ptr(self) -> Int:
        return self.state_base() + Self.M.X_MAIN_OFF

    def x_residual_ptr(self) -> Int:
        return self.state_base() + Self.M.X_RESIDUAL_OFF

    def scratch_ptr(self) -> Int:
        """Address of scratch memory for writing token IDs before forward().
        Safe to use as tokens_ptr — consumed by embed_lookup before
        scratch is reused for intermediates."""
        return self.state_base() + Self.M.SCRATCH_OFF

    def scratch_slot(self, index: Int) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF + index * Self.M.SCRATCH_STRIDE

    def rope_cos_ptr(self) -> Int:
        return self.state_base() + Self.M.ROPE_COS_OFF

    def rope_sin_ptr(self) -> Int:
        return self.state_base() + Self.M.ROPE_SIN_OFF

    def forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[Self.M.C.VOCAB_SIZE]
        where Self.E.DTYPE == DType.bfloat16:
        comptime M = Self.M
        comptime L = M.LAYER
        comptime C = M.C
        var prof = Profiler(profile)

        var x = DynView[M.X_MAIN](self.x_main_ptr(), seq_len)
        var x_res = DynView[M.X_RESIDUAL](self.x_residual_ptr(), seq_len)
        var rope_cos = Bound[M.ROPE_COS](self.rope_cos_ptr())
        var rope_sin = Bound[M.ROPE_SIN](self.rope_sin_ptr())

        prof.section("embed")
        embed_lookup(self.weight[M.EMBED](), tokens_ptr, x, self.pool).join()

        for layer_idx in range(C.NUM_LAYERS):
            var k_cache = CacheView[L.K_CACHE](self.k_cache_ptr(layer_idx))
            var v_cache = CacheView[L.V_CACHE](self.v_cache_ptr(layer_idx))

            prof.section("rmsnorm")
            rmsnorm(x, self.layer_weight[L.INPUT_NORM](layer_idx), x_res, self.pool).join()

            var q = DynView[M.X_MAIN](self.scratch_slot(0), seq_len)
            var k = DynView[L.K_CACHE](self.scratch_slot(1), seq_len)
            var v = DynView[L.V_CACHE](self.scratch_slot(2), seq_len)

            prof.section("gemm_qkv")
            gemm(x_res, self.layer_weight[L.Q_PROJ](layer_idx), q, self.pool).join()
            gemm(x_res, self.layer_weight[L.K_PROJ](layer_idx), k, self.pool).join()
            gemm(x_res, self.layer_weight[L.V_PROJ](layer_idx), v, self.pool).join()

            prof.section("rope")
            rope[C.HEAD_DIM, C.NUM_HEADS](q, rope_cos, rope_sin, pos)
            rope[C.HEAD_DIM, C.NUM_KV_HEADS](k, rope_cos, rope_sin, pos)

            prof.section("kv_write")
            kv_cache_write(k, k_cache, pos)
            kv_cache_write(v, v_cache, pos)

            var attn_out = DynView[M.X_MAIN](self.scratch_slot(1), seq_len)

            prof.section("attention")
            attention[C.NUM_HEADS, C.NUM_KV_HEADS, C.HEAD_DIM](
                q, k_cache, v_cache, attn_out, pos, self.pool).join()

            prof.section("gemm_o")
            gemm(attn_out, self.layer_weight[L.O_PROJ](layer_idx), x_res, self.pool).join()

            prof.section("elem_add")
            elem_add(x, x_res, x)

            var gate = DynView[M.SCRATCH](self.scratch_slot(0), seq_len)
            var up = DynView[M.SCRATCH](self.scratch_slot(1), seq_len)

            prof.section("rmsnorm")
            rmsnorm(x, self.layer_weight[L.POST_ATTN_NORM](layer_idx), x_res, self.pool).join()

            prof.section("gemm_gate_up")
            gemm(x_res, self.layer_weight[L.GATE_PROJ](layer_idx), gate, self.pool).join()
            gemm(x_res, self.layer_weight[L.UP_PROJ](layer_idx), up, self.pool).join()

            prof.section("silu_mul")
            silu_mul(gate, up, gate)

            prof.section("gemm_down")
            gemm(gate, self.layer_weight[L.DOWN_PROJ](layer_idx), x_res, self.pool).join()

            prof.section("elem_add")
            elem_add(x, x_res, x)

        prof.section("final_norm")
        rmsnorm(x, self.weight[M.FINAL_NORM](), x, self.pool).join()

        var logits = DynView[M.LOGITS](self.scratch_slot(0), seq_len)

        prof.section("lm_head")
        gemm(x, self.weight[M.EMBED](), logits, self.pool).join()

        prof.finish()
        prof.report()

        return LogitsView[M.C.VOCAB_SIZE](logits.ptr, seq_len)


comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"


def main():
    var model_opt = SmolLM2Loaded[BF16, 1].load(Path(MODEL_PATH), 0)
    if not model_opt:
        return
    var model = model_opt.take()

    var tp = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=model.scratch_ptr()
    )
    tp[0] = Scalar[DType.int32](42)
    var logits = model.forward(model.scratch_ptr(), 1, 0, profile=True)
    _ = logits
