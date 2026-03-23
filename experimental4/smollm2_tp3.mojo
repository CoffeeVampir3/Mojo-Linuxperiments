"""SmolLM2-135M with tensor parallelism = 3.

TP=3 cleanly divides: NUM_HEADS=9 (3 per rank), NUM_KV_HEADS=3 (1 per rank),
INTERMEDIATE=1536 (512 per rank). Each rank has its own NUMA arena, BurstPool,
and activation buffers. Allreduce is needed after ColShard matmuls (o_proj,
down_proj) to sum partial results across ranks.

This file is self-contained — each TP configuration is a separate universe
so it can be tuned for true optimality without entangling distribution concerns.
"""

from pathlib import Path

from memory import UnsafePointer
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
    PoolFence, parallel,
)
from experimental4.loader import load_safetensors
from experimental4.profiler import Profiler
from experimental4.smollm2 import SmolLM2Config, LogitsView, LogitAccess


# =============================================================================
# TP=3 constants
# =============================================================================

comptime TP = 3
comptime C = SmolLM2Config

# Per-rank derived constants.
comptime LOCAL_HEADS = C.NUM_HEADS // TP          # 3
comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // TP    # 1
comptime LOCAL_Q_DIM = LOCAL_HEADS * C.HEAD_DIM    # 192
comptime LOCAL_KV_DIM = LOCAL_KV_HEADS * C.HEAD_DIM  # 64
comptime LOCAL_INTERMEDIATE = C.INTERMEDIATE // TP  # 512


# =============================================================================
# Layer and model weight layout (per-rank arena)
# =============================================================================

struct TP3Layer[E: Encoding]:
    comptime Q_PROJ      = PlacedSlot[Self.E, RowShard, C.HIDDEN, C.HIDDEN, TP, 0, "self_attn.q_proj.weight"]
    comptime K_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, TP, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight"]
    comptime V_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, TP, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight"]
    comptime O_PROJ      = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.HIDDEN, TP, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight"]
    comptime GATE_PROJ   = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, TP, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight"]
    comptime UP_PROJ     = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, TP, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight"]
    comptime DOWN_PROJ   = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.INTERMEDIATE, TP, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight"]
    comptime INPUT_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, TP, next_offset[Self.DOWN_PROJ](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, TP, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    # Per-rank KV cache: each rank stores its local KV heads.
    comptime K_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, TP]  # [MAX_SEQ, 64]
    comptime V_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, TP]  # [MAX_SEQ, 64]

    @staticmethod
    fn for_each_weight[
        func: fn[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
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
    fn weight_bytes() -> Int:
        return Self.STRIDE

    @staticmethod
    fn cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()


struct TP3Model[E: Encoding](WeightIterable):
    comptime LAYER = TP3Layer[Self.E]

    # --- Distributed weights (all rank arenas) ---
    comptime LAYERS_OFF   = 0
    comptime NUM_LAYERS   = C.NUM_LAYERS
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = C.NUM_LAYERS * Self.LAYER.STRIDE

    # --- Per-rank activation slots (correct local dimensions) ---
    comptime ROPE_HALF = C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, TP]
    comptime ROPE_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, TP]

    # Full hidden width — each rank needs the complete hidden state
    # (input to sharded matmuls, output of allreduce).
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, TP]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, TP]

    # Scratch for sharded Q output: [seq, LOCAL_Q_DIM=192]
    comptime Q_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, TP]

    # Scratch for sharded K/V output: [seq, LOCAL_KV_DIM=64]
    comptime KV_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, TP]

    # Scratch for sharded gate/up output: [seq, LOCAL_INTERMEDIATE=512]
    comptime MLP_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, TP]

    # Logits (full vocab, only on host rank).
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, TP]

    # --- Per-rank state layout ---
    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime Q_SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime KV_SCRATCH_0_OFF = Self.Q_SCRATCH_OFF + byte_count[Self.Q_SCRATCH]()
    comptime KV_SCRATCH_1_OFF = Self.KV_SCRATCH_0_OFF + byte_count[Self.KV_SCRATCH]()
    comptime MLP_SCRATCH_0_OFF = Self.KV_SCRATCH_1_OFF + byte_count[Self.KV_SCRATCH]()
    comptime MLP_SCRATCH_1_OFF = Self.MLP_SCRATCH_0_OFF + byte_count[Self.MLP_SCRATCH]()
    comptime ROPE_COS_OFF = Self.MLP_SCRATCH_1_OFF + byte_count[Self.MLP_SCRATCH]()
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # --- NodeLocal weights (host arena only, after distributed + state) ---
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, C.HIDDEN, 1, TP, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[Self.E, PrincipleNodeLocal, C.VOCAB_SIZE, C.HIDDEN, TP, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

    @staticmethod
    fn for_each_weight[
        func: fn[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ]():
        @parameter
        for i in range(Self.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.LAYERS_OFF + i * Self.LAYER_STRIDE
            Self.LAYER.for_each_weight[func](prefix, base)

        func[PrincipleNodeLocal, Self.FINAL_NORM]("", 0)
        func[PrincipleNodeLocal, Self.EMBED]("", 0)

    @staticmethod
    fn arena_bytes() -> Int:
        """Non-host rank arena: distributed weights + state."""
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    fn host_arena_bytes() -> Int:
        """Host rank arena: distributed weights + state + node-local weights."""
        return next_offset[Self.EMBED]()


# =============================================================================
# Stubs for missing kernels
#
# Each stub documents the exact signature and semantics needed. Implement
# these one at a time — the forward pass calls them and will work once
# the stubs are replaced with real implementations.
# =============================================================================


fn allreduce_sum_missing_stub[T: Encoding & Shaped](
    buf0: DynView[T], buf1: DynView[T], buf2: DynView[T],
):
    """Megatron-style allreduce: sum element-wise across 3 rank buffers,
    result written back to ALL rank buffers.

    After this call: buf0[i] == buf1[i] == buf2[i] == original(buf0[i] + buf1[i] + buf2[i]).

    Each buffer is [seq_len, T.COLS] bf16. For TP=3 on a single node:
    load 3 bf16 vectors → upcast to f32 → sum → downcast to bf16 → store
    to all 3 destinations. Should be BurstPool-parallelized over rows.

    STUB: currently a no-op. Replace with real implementation.
    """
    pass


fn parallel_for_ranks_missing_stub[
    body: fn(Int) capturing -> None,
](num_ranks: Int):
    """Execute body(rank) for each rank in parallel, wait for all to complete.

    This is the TP orchestration primitive. On a single node it can use
    std threading or dispatch each rank's work to its BurstPool. On
    multi-node it would involve network communication.

    For single-node TP=3: launch 3 threads (or use the main thread for
    rank 0 and 2 helper threads for ranks 1,2), each calling body(rank).
    Barrier at the end.

    STUB: currently runs sequentially. Replace with real implementation.
    """
    for rank in range(num_ranks):
        body(rank)


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankState[E: Encoding]:
    """Non-owning view into one rank's arena. Provides typed accessors
    for weights, activations, KV cache, and scratch buffers."""
    comptime M = TP3Model[Self.E]
    comptime L = Self.M.LAYER

    var weight_base: Int
    var state_base: Int

    fn __init__(out self, arena_base: Int):
        self.weight_base = arena_base
        self.state_base = arena_base + Self.M.DISTRIBUTED_BYTES

    fn layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE)

    fn weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base)

    fn k_cache_ptr(self, layer: Int) -> Int:
        return self.state_base + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    fn v_cache_ptr(self, layer: Int) -> Int:
        return self.k_cache_ptr(layer) + byte_count[Self.L.K_CACHE]()

    fn x_main_ptr(self) -> Int:
        return self.state_base + Self.M.X_MAIN_OFF

    fn x_residual_ptr(self) -> Int:
        return self.state_base + Self.M.X_RESIDUAL_OFF

    fn q_scratch_ptr(self) -> Int:
        return self.state_base + Self.M.Q_SCRATCH_OFF

    fn kv_scratch_ptr(self, index: Int) -> Int:
        """index 0 = K scratch, index 1 = V scratch."""
        if index == 0:
            return self.state_base + Self.M.KV_SCRATCH_0_OFF
        return self.state_base + Self.M.KV_SCRATCH_1_OFF

    fn mlp_scratch_ptr(self, index: Int) -> Int:
        """index 0 = gate scratch, index 1 = up scratch."""
        if index == 0:
            return self.state_base + Self.M.MLP_SCRATCH_0_OFF
        return self.state_base + Self.M.MLP_SCRATCH_1_OFF

    fn rope_cos_ptr(self) -> Int:
        return self.state_base + Self.M.ROPE_COS_OFF

    fn rope_sin_ptr(self) -> Int:
        return self.state_base + Self.M.ROPE_SIN_OFF


# =============================================================================
# Loaded model: manages 3 arenas + 3 pools
# =============================================================================


@fieldwise_init
struct SmolLM2TP3Loaded[E: Encoding](Movable):
    comptime M = TP3Model[Self.E]

    var arena0: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var arena1: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var arena2: NumaArena[alignment=DEFAULT_ALIGNMENT]
    var pool0: BurstPool
    var pool1: BurstPool
    var pool2: BurstPool

    fn base(self, rank: Int) -> Int:
        if rank == 0: return Int(self.arena0.base)
        if rank == 1: return Int(self.arena1.base)
        return Int(self.arena2.base)

    @staticmethod
    fn load(path: Path, node_ids: List[Int]) -> Optional[Self]:
        """Load SmolLM2 TP=3 from a safetensors file.

        Args:
            path: Path to the safetensors file.
            node_ids: NUMA node ID for each rank (len must be TP=3).
                      For single-node systems, pass [0, 0, 0].
        """
        if len(node_ids) != TP:
            print("TP3: expected", TP, "node_ids, got", len(node_ids))
            return None

        comptime host_rank = 0

        # Allocate per-rank arenas.
        var a0 = NumaArena[alignment=DEFAULT_ALIGNMENT](node_ids[0], Self.M.host_arena_bytes())
        if not a0:
            print("TP3: arena allocation failed for rank 0")
            return None
        var a1 = NumaArena[alignment=DEFAULT_ALIGNMENT](node_ids[1], Self.M.arena_bytes())
        if not a1:
            print("TP3: arena allocation failed for rank 1")
            return None
        var a2 = NumaArena[alignment=DEFAULT_ALIGNMENT](node_ids[2], Self.M.arena_bytes())
        if not a2:
            print("TP3: arena allocation failed for rank 2")
            return None

        # Collect arena base addresses for the loader.
        var bases = List[Int]()
        bases.append(Int(a0.base))
        bases.append(Int(a1.base))
        bases.append(Int(a2.base))

        # Load weights — the loader handles sharding across all 3 arenas.
        var result = load_safetensors[Self.M](path, bases, host_index=host_rank)
        if not result:
            print("TP3: weight loading failed")
            return None

        # Prefault distributed weight pages + state pages on each rank.
        _ = a0.prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)
        _ = a1.prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)
        _ = a2.prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        # Create per-rank thread pools.
        var numa = NumaInfo()
        var p0 = BurstPool.for_numa_node(numa, node_ids[0])
        var p1 = BurstPool.for_numa_node(numa, node_ids[1])
        var p2 = BurstPool.for_numa_node(numa, node_ids[2])

        var model = Self(a0^, a1^, a2^, p0^, p1^, p2^)

        # Init RoPE tables on each rank.
        for rank in range(TP):
            var rs = RankState[Self.E](model.base(rank))
            init_rope_tables(
                Bound[Self.M.ROPE_COS](rs.rope_cos_ptr()),
                Bound[Self.M.ROPE_SIN](rs.rope_sin_ptr()),
                Float64(C.ROPE_THETA),
            )

        return model^

    fn forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[C.VOCAB_SIZE]
        where Self.E.DTYPE == DType.bfloat16:
        comptime M = Self.M
        comptime L = M.LAYER
        var prof = Profiler(profile)

        # Build per-rank state views.
        var rs0 = RankState[Self.E](self.base(0))
        var rs1 = RankState[Self.E](self.base(1))
        var rs2 = RankState[Self.E](self.base(2))

        # Per-rank full-hidden activation views.
        var x0 = DynView[M.X_MAIN](rs0.x_main_ptr(), seq_len)
        var x1 = DynView[M.X_MAIN](rs1.x_main_ptr(), seq_len)
        var x2 = DynView[M.X_MAIN](rs2.x_main_ptr(), seq_len)

        var x_res0 = DynView[M.X_RESIDUAL](rs0.x_residual_ptr(), seq_len)
        var x_res1 = DynView[M.X_RESIDUAL](rs1.x_residual_ptr(), seq_len)
        var x_res2 = DynView[M.X_RESIDUAL](rs2.x_residual_ptr(), seq_len)

        var rope_cos0 = Bound[M.ROPE_COS](rs0.rope_cos_ptr())
        var rope_sin0 = Bound[M.ROPE_SIN](rs0.rope_sin_ptr())
        var rope_cos1 = Bound[M.ROPE_COS](rs1.rope_cos_ptr())
        var rope_sin1 = Bound[M.ROPE_SIN](rs1.rope_sin_ptr())
        var rope_cos2 = Bound[M.ROPE_COS](rs2.rope_cos_ptr())
        var rope_sin2 = Bound[M.ROPE_SIN](rs2.rope_sin_ptr())

        # --- Embed (host rank only, then broadcast to all ranks) ---
        prof.section("embed")
        embed_lookup(rs0.weight[M.EMBED](), tokens_ptr, x0, self.pool0).join()
        broadcast_x_missing_stub(x0, x1, x2, seq_len)

        for layer_idx in range(C.NUM_LAYERS):
            var k_cache0 = CacheView[L.K_CACHE](rs0.k_cache_ptr(layer_idx))
            var v_cache0 = CacheView[L.V_CACHE](rs0.v_cache_ptr(layer_idx))
            var k_cache1 = CacheView[L.K_CACHE](rs1.k_cache_ptr(layer_idx))
            var v_cache1 = CacheView[L.V_CACHE](rs1.v_cache_ptr(layer_idx))
            var k_cache2 = CacheView[L.K_CACHE](rs2.k_cache_ptr(layer_idx))
            var v_cache2 = CacheView[L.V_CACHE](rs2.v_cache_ptr(layer_idx))

            # --- RMSNorm (all ranks in parallel) ---
            prof.section("rmsnorm")
            parallel(
                rmsnorm(x0, rs0.layer_weight[L.INPUT_NORM](layer_idx), x_res0, self.pool0),
                rmsnorm(x1, rs1.layer_weight[L.INPUT_NORM](layer_idx), x_res1, self.pool1),
                rmsnorm(x2, rs2.layer_weight[L.INPUT_NORM](layer_idx), x_res2, self.pool2),
            )

            # --- Q/K/V projections (all ranks in parallel per projection) ---
            var q0 = DynView[M.Q_SCRATCH](rs0.q_scratch_ptr(), seq_len)
            var q1 = DynView[M.Q_SCRATCH](rs1.q_scratch_ptr(), seq_len)
            var q2 = DynView[M.Q_SCRATCH](rs2.q_scratch_ptr(), seq_len)

            var k0 = DynView[M.KV_SCRATCH](rs0.kv_scratch_ptr(0), seq_len)
            var k1 = DynView[M.KV_SCRATCH](rs1.kv_scratch_ptr(0), seq_len)
            var k2 = DynView[M.KV_SCRATCH](rs2.kv_scratch_ptr(0), seq_len)

            var v0 = DynView[M.KV_SCRATCH](rs0.kv_scratch_ptr(1), seq_len)
            var v1 = DynView[M.KV_SCRATCH](rs1.kv_scratch_ptr(1), seq_len)
            var v2 = DynView[M.KV_SCRATCH](rs2.kv_scratch_ptr(1), seq_len)

            prof.section("gemm_q")
            parallel(
                gemm(x_res0, rs0.layer_weight[L.Q_PROJ](layer_idx), q0, self.pool0),
                gemm(x_res1, rs1.layer_weight[L.Q_PROJ](layer_idx), q1, self.pool1),
                gemm(x_res2, rs2.layer_weight[L.Q_PROJ](layer_idx), q2, self.pool2),
            )

            prof.section("gemm_k")
            parallel(
                gemm(x_res0, rs0.layer_weight[L.K_PROJ](layer_idx), k0, self.pool0),
                gemm(x_res1, rs1.layer_weight[L.K_PROJ](layer_idx), k1, self.pool1),
                gemm(x_res2, rs2.layer_weight[L.K_PROJ](layer_idx), k2, self.pool2),
            )

            prof.section("gemm_v")
            parallel(
                gemm(x_res0, rs0.layer_weight[L.V_PROJ](layer_idx), v0, self.pool0),
                gemm(x_res1, rs1.layer_weight[L.V_PROJ](layer_idx), v1, self.pool1),
                gemm(x_res2, rs2.layer_weight[L.V_PROJ](layer_idx), v2, self.pool2),
            )

            # --- RoPE (caller-thread, sequential across ranks) ---
            prof.section("rope")
            rope[C.HEAD_DIM, LOCAL_HEADS](q0, rope_cos0, rope_sin0, pos)
            rope[C.HEAD_DIM, LOCAL_HEADS](q1, rope_cos1, rope_sin1, pos)
            rope[C.HEAD_DIM, LOCAL_HEADS](q2, rope_cos2, rope_sin2, pos)
            rope[C.HEAD_DIM, LOCAL_KV_HEADS](k0, rope_cos0, rope_sin0, pos)
            rope[C.HEAD_DIM, LOCAL_KV_HEADS](k1, rope_cos1, rope_sin1, pos)
            rope[C.HEAD_DIM, LOCAL_KV_HEADS](k2, rope_cos2, rope_sin2, pos)

            # --- KV cache write (caller-thread, sequential) ---
            prof.section("kv_write")
            kv_cache_write(k0, k_cache0, pos)
            kv_cache_write(k1, k_cache1, pos)
            kv_cache_write(k2, k_cache2, pos)
            kv_cache_write(v0, v_cache0, pos)
            kv_cache_write(v1, v_cache1, pos)
            kv_cache_write(v2, v_cache2, pos)

            # --- Attention (all ranks in parallel) ---
            var attn_out0 = DynView[M.Q_SCRATCH](rs0.kv_scratch_ptr(0), seq_len)
            var attn_out1 = DynView[M.Q_SCRATCH](rs1.kv_scratch_ptr(0), seq_len)
            var attn_out2 = DynView[M.Q_SCRATCH](rs2.kv_scratch_ptr(0), seq_len)

            prof.section("attention")
            parallel(
                attention[LOCAL_HEADS, LOCAL_KV_HEADS, C.HEAD_DIM](
                    q0, k_cache0, v_cache0, attn_out0, pos, self.pool0),
                attention[LOCAL_HEADS, LOCAL_KV_HEADS, C.HEAD_DIM](
                    q1, k_cache1, v_cache1, attn_out1, pos, self.pool1),
                attention[LOCAL_HEADS, LOCAL_KV_HEADS, C.HEAD_DIM](
                    q2, k_cache2, v_cache2, attn_out2, pos, self.pool2),
            )

            # --- O projection (all ranks in parallel, then allreduce) ---
            prof.section("gemm_o")
            parallel(
                gemm(attn_out0, rs0.layer_weight[L.O_PROJ](layer_idx), x_res0, self.pool0),
                gemm(attn_out1, rs1.layer_weight[L.O_PROJ](layer_idx), x_res1, self.pool1),
                gemm(attn_out2, rs2.layer_weight[L.O_PROJ](layer_idx), x_res2, self.pool2),
            )

            prof.section("allreduce_o")
            allreduce_sum_missing_stub[M.X_RESIDUAL](x_res0, x_res1, x_res2)

            # --- Residual add (each rank local) ---
            prof.section("elem_add")
            elem_add(x0, x_res0, x0)
            elem_add(x1, x_res1, x1)
            elem_add(x2, x_res2, x2)

            # --- MLP: RMSNorm (all ranks in parallel) ---
            prof.section("rmsnorm")
            parallel(
                rmsnorm(x0, rs0.layer_weight[L.POST_ATTN_NORM](layer_idx), x_res0, self.pool0),
                rmsnorm(x1, rs1.layer_weight[L.POST_ATTN_NORM](layer_idx), x_res1, self.pool1),
                rmsnorm(x2, rs2.layer_weight[L.POST_ATTN_NORM](layer_idx), x_res2, self.pool2),
            )

            # --- Gate/Up projections (all ranks in parallel) ---
            var gate0 = DynView[M.MLP_SCRATCH](rs0.mlp_scratch_ptr(0), seq_len)
            var gate1 = DynView[M.MLP_SCRATCH](rs1.mlp_scratch_ptr(0), seq_len)
            var gate2 = DynView[M.MLP_SCRATCH](rs2.mlp_scratch_ptr(0), seq_len)

            var up0 = DynView[M.MLP_SCRATCH](rs0.mlp_scratch_ptr(1), seq_len)
            var up1 = DynView[M.MLP_SCRATCH](rs1.mlp_scratch_ptr(1), seq_len)
            var up2 = DynView[M.MLP_SCRATCH](rs2.mlp_scratch_ptr(1), seq_len)

            prof.section("gemm_gate")
            parallel(
                gemm(x_res0, rs0.layer_weight[L.GATE_PROJ](layer_idx), gate0, self.pool0),
                gemm(x_res1, rs1.layer_weight[L.GATE_PROJ](layer_idx), gate1, self.pool1),
                gemm(x_res2, rs2.layer_weight[L.GATE_PROJ](layer_idx), gate2, self.pool2),
            )

            prof.section("gemm_up")
            parallel(
                gemm(x_res0, rs0.layer_weight[L.UP_PROJ](layer_idx), up0, self.pool0),
                gemm(x_res1, rs1.layer_weight[L.UP_PROJ](layer_idx), up1, self.pool1),
                gemm(x_res2, rs2.layer_weight[L.UP_PROJ](layer_idx), up2, self.pool2),
            )

            # --- SiLU * Up (caller-thread, sequential) ---
            prof.section("silu_mul")
            silu_mul(gate0, up0, gate0)
            silu_mul(gate1, up1, gate1)
            silu_mul(gate2, up2, gate2)

            # --- Down projection (all ranks in parallel, then allreduce) ---
            prof.section("gemm_down")
            parallel(
                gemm(gate0, rs0.layer_weight[L.DOWN_PROJ](layer_idx), x_res0, self.pool0),
                gemm(gate1, rs1.layer_weight[L.DOWN_PROJ](layer_idx), x_res1, self.pool1),
                gemm(gate2, rs2.layer_weight[L.DOWN_PROJ](layer_idx), x_res2, self.pool2),
            )

            prof.section("allreduce_down")
            allreduce_sum_missing_stub[M.X_RESIDUAL](x_res0, x_res1, x_res2)

            # --- Residual add (each rank local) ---
            prof.section("elem_add")
            elem_add(x0, x_res0, x0)
            elem_add(x1, x_res1, x1)
            elem_add(x2, x_res2, x2)

        # --- Final norm + LM head (host rank only) ---
        prof.section("final_norm")
        rmsnorm(x0, rs0.weight[M.FINAL_NORM](), x0, self.pool0).join()

        var logits = DynView[M.LOGITS](rs0.q_scratch_ptr(), seq_len)

        prof.section("lm_head")
        gemm(x0, rs0.weight[M.EMBED](), logits, self.pool0).join()

        prof.finish()
        prof.report()

        return LogitsView[C.VOCAB_SIZE](logits.ptr, seq_len)


# =============================================================================
# Additional stubs
# =============================================================================


fn broadcast_x_missing_stub[T: Encoding & Shaped](
    src: DynView[T], dst1: DynView[T], dst2: DynView[T], seq_len: Int,
):
    """Copy src buffer to dst1 and dst2 (full hidden state broadcast after embed).

    On single-node shared memory: two memcpy operations, each seq_len * HIDDEN * 2 bytes.
    Should be parallelized across pools for the destination ranks.

    STUB: currently a no-op. Replace with real implementation.
    """
    pass


# =============================================================================
# Entry point
# =============================================================================

comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"


fn main():
    var node_ids = List[Int]()
    node_ids.append(0)
    node_ids.append(0)
    node_ids.append(0)

    var model_opt = SmolLM2TP3Loaded[BF16].load(Path(MODEL_PATH), node_ids)
    if not model_opt:
        return
    var model = model_opt.take()

    # Write a test token into rank 0's scratch area.
    var rank0 = RankState[BF16](model.base(0))
    var tp_ptr = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=rank0.q_scratch_ptr()
    )
    tp_ptr[0] = Scalar[DType.int32](42)

    var logits = model.forward(rank0.q_scratch_ptr(), 1, 0, profile=True)
    _ = logits
