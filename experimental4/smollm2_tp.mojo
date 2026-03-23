"""SmolLM2-135M with parametric tensor parallelism.

TP degree is a comptime parameter. Valid values must cleanly divide
NUM_HEADS, NUM_KV_HEADS, and INTERMEDIATE. For SmolLM2-135M:
  TP=1 (trivial), TP=3 (3 query heads, 1 KV head per rank).

Each rank has its own NUMA arena, BurstPool, and activation buffers.
Megatron-style: RowShard for output projections (no comm), ColShard
for input projections (allreduce after). Dispatch via parallel_for.
"""

from pathlib import Path

from memory import UnsafePointer, memcpy
from collections import InlineArray
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from experimental4.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32,
    ShardStrategy, RowShard, ColShard, Replicated,
    PrincipleNodeLocal,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
)
from experimental4.kernel_ops import (
    gemm, rmsnorm, embed_lookup, silu_mul, elem_add, rope, kv_cache_write,
    attention, init_rope_tables,
    PoolFence, parallel, parallel_for,
)
from experimental4.loader import load_safetensors
from experimental4.profiler import Profiler
from experimental4.smollm2 import SmolLM2Config, LogitsView, LogitAccess


# =============================================================================
# Parametric model spec
# =============================================================================

comptime C = SmolLM2Config


struct TPLayer[E: Encoding, tp: Int]:
    comptime Q_PROJ      = PlacedSlot[Self.E, RowShard, C.HIDDEN, C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight"]
    comptime K_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight"]
    comptime V_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight"]
    comptime O_PROJ      = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight"]
    comptime GATE_PROJ   = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight"]
    comptime UP_PROJ     = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight"]
    comptime DOWN_PROJ   = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight"]
    comptime INPUT_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    comptime K_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]

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
    fn cache_bytes() -> Int:
        return byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()


struct TPModel[E: Encoding, tp: Int](WeightIterable):
    comptime LAYER = TPLayer[Self.E, Self.tp]

    comptime LAYERS_OFF = 0
    comptime LAYER_STRIDE = Self.LAYER.STRIDE
    comptime DISTRIBUTED_BYTES = C.NUM_LAYERS * Self.LAYER.STRIDE

    # Per-rank head counts.
    comptime LOCAL_HEADS = C.NUM_HEADS // Self.tp
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // Self.tp

    # Per-rank activation slots.
    comptime ROPE_HALF = C.HEAD_DIM // 2
    comptime ROPE_COS = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime ROPE_SIN = Slot[F32, Replicated, C.MAX_SEQ_LEN, Self.ROPE_HALF, Self.tp]
    comptime X_MAIN = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime X_RESIDUAL = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime Q_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime KV_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime MLP_SCRATCH = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    # Per-rank state layout.
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

    # NodeLocal weights (host arena only).
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, C.HIDDEN, 1, Self.tp, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[Self.E, PrincipleNodeLocal, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

    @staticmethod
    fn for_each_weight[
        func: fn[S: ShardStrategy, T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ]():
        @parameter
        for i in range(C.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = Self.LAYERS_OFF + i * Self.LAYER_STRIDE
            Self.LAYER.for_each_weight[func](prefix, base)
        func[PrincipleNodeLocal, Self.FINAL_NORM]("", 0)
        func[PrincipleNodeLocal, Self.EMBED]("", 0)

    @staticmethod
    fn arena_bytes() -> Int:
        return Self.DISTRIBUTED_BYTES + Self.STATE_BYTES

    @staticmethod
    fn host_arena_bytes() -> Int:
        return next_offset[Self.EMBED]()


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankView[E: Encoding, tp: Int]:
    comptime M = TPModel[Self.E, Self.tp]
    comptime L = Self.M.LAYER
    var base: Int

    fn __init__(out self, arena_base: Int):
        self.base = arena_base

    fn weight_base(self) -> Int:
        return self.base

    fn state_base(self) -> Int:
        return self.base + Self.M.DISTRIBUTED_BYTES

    fn layer_weight[T: Encoding & Shaped & Placed & Named](self, layer: Int) -> Bound[T]:
        return bind[T](self.weight_base() + Self.M.LAYERS_OFF + layer * Self.M.LAYER_STRIDE)

    fn weight[T: Encoding & Shaped & Placed & Named](self) -> Bound[T]:
        return bind[T](self.weight_base())

    fn k_cache(self, layer: Int) -> CacheView[Self.L.K_CACHE]:
        return CacheView[Self.L.K_CACHE](self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE)

    fn v_cache(self, layer: Int) -> CacheView[Self.L.V_CACHE]:
        return CacheView[Self.L.V_CACHE](
            self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE + byte_count[Self.L.K_CACHE]()
        )

    fn x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    fn x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    fn q_scratch(self, seq_len: Int) -> DynView[Self.M.Q_SCRATCH]:
        return DynView[Self.M.Q_SCRATCH](self.state_base() + Self.M.Q_SCRATCH_OFF, seq_len)

    fn kv_scratch(self, index: Int, seq_len: Int) -> DynView[Self.M.KV_SCRATCH]:
        if index == 0:
            return DynView[Self.M.KV_SCRATCH](self.state_base() + Self.M.KV_SCRATCH_0_OFF, seq_len)
        return DynView[Self.M.KV_SCRATCH](self.state_base() + Self.M.KV_SCRATCH_1_OFF, seq_len)

    fn mlp_scratch(self, index: Int, seq_len: Int) -> DynView[Self.M.MLP_SCRATCH]:
        if index == 0:
            return DynView[Self.M.MLP_SCRATCH](self.state_base() + Self.M.MLP_SCRATCH_0_OFF, seq_len)
        return DynView[Self.M.MLP_SCRATCH](self.state_base() + Self.M.MLP_SCRATCH_1_OFF, seq_len)

    fn rope_cos(self) -> Bound[Self.M.ROPE_COS]:
        return Bound[Self.M.ROPE_COS](self.state_base() + Self.M.ROPE_COS_OFF)

    fn rope_sin(self) -> Bound[Self.M.ROPE_SIN]:
        return Bound[Self.M.ROPE_SIN](self.state_base() + Self.M.ROPE_SIN_OFF)


# =============================================================================
# Ranks — dispatch helper
# =============================================================================


@fieldwise_init
struct Ranks[E: Encoding, tp: Int]:
    """Rank-indexed dispatch helper. Captures base addresses and pool pointers
    for use in parallel_for closures and sequential loops."""
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], Self.tp]

    fn view(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    fn parallel[body: fn[rank: Int] (RankView[Self.E, Self.tp], mut BurstPool) capturing -> PoolFence](self):
        """Dispatch body(rv, pool) for each rank in parallel, then join all."""
        var bases = self.bases.copy()
        var ptrs = self.pool_ptrs.copy()
        @parameter
        fn dispatch[rank: Int]() -> PoolFence:
            var rv = RankView[Self.E, Self.tp](bases[rank])
            return body[rank](rv, ptrs[rank][])
        parallel_for[Self.tp, dispatch]()

    fn each[body: fn (RankView[Self.E, Self.tp]) capturing -> None](self):
        """Run body(rv) for each rank sequentially on the caller thread."""
        for r in range(Self.tp):
            body(self.view(r))

    fn x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_main(seq_len).ptr
        return ptrs^

    fn x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for r in range(Self.tp):
            ptrs[r] = self.view(r).x_residual(seq_len).ptr
        return ptrs^


# =============================================================================
# Stubs
# =============================================================================


fn allreduce_sum_missing_stub[T: Encoding & Shaped, tp: Int](
    ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Megatron-style allreduce: sum across tp rank buffers, result in all buffers.
    Each ptr points to [seq_len, T.COLS] bf16 data.
    STUB: no-op. Replace with real implementation."""
    pass


fn ring_broadcast[T: Encoding & Shaped, tp: Int](
    src_ptr: Int, dst_ptrs: InlineArray[Int, tp], seq_len: Int,
):
    """Ring broadcast: forward data along the ring, one hop per step.

    Step 0: ensure dst_ptrs[0] has the data (copy from src if needed).
    Step s (s=0..tp-2): rank s forwards to rank s+1.

    Each copy traverses one ring edge (minimum NUMA distance). Spreads
    memory bandwidth across all controllers instead of bottlenecking
    at the source rank.
    """
    var total_bytes = seq_len * T.COLS * T.ELEMENT_BYTES
    if total_bytes <= 0 or tp <= 1:
        return

    # Ensure rank 0 has the data.
    if src_ptr != dst_ptrs[0]:
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[0]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=src_ptr),
            count=total_bytes,
        )

    # Forward along the ring: rank 0 → 1 → 2 → ... → tp-1.
    for step in range(tp - 1):
        memcpy(
            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[step + 1]),
            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=dst_ptrs[step]),
            count=total_bytes,
        )


# =============================================================================
# Loaded model
# =============================================================================


struct SmolLM2TP[E: Encoding, tp: Int](Movable):
    comptime M = TPModel[Self.E, Self.tp]

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[BurstPool]
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], Self.tp]

    fn __init__(out self, var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
                var pools: HeapMoveArray[BurstPool]):
        self.bases = InlineArray[Int, Self.tp](fill=0)
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool, MutAnyOrigin]()
        )
        self.arenas = arenas^
        self.pools = pools^
        for rank in range(Self.tp):
            self.bases[rank] = Int(self.arenas[rank][].base)
            self.pool_ptrs[rank] = UnsafePointer[BurstPool, MutAnyOrigin](
                unsafe_from_address=Int(self.pools[rank])
            )

    fn rank(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    @staticmethod
    fn load(path: Path) -> Optional[Self]:
        """Load SmolLM2 with automatic NUMA-aware rank placement.
        Discovers topology, selects the tightest node cluster,
        and orders ranks for optimal ring allreduce adjacency."""
        constrained[
            C.NUM_HEADS % Self.tp == 0,
            "TP must evenly divide NUM_HEADS",
        ]()
        constrained[
            C.NUM_KV_HEADS % Self.tp == 0,
            "TP must evenly divide NUM_KV_HEADS",
        ]()
        constrained[
            C.INTERMEDIATE % Self.tp == 0,
            "TP must evenly divide INTERMEDIATE",
        ]()

        var numa = NumaInfo()
        var topo = numa.plan_topology(Self.tp)
        comptime host_rank = 0

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = Self.M.host_arena_bytes() if rank == host_rank else Self.M.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo[rank], size)
            if not arena:
                print("TP: arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank][].base))

        var result = load_safetensors[Self.M](path, arena_bases, host_index=host_rank)
        if not result:
            print("TP: weight loading failed")
            return None

        for rank in range(Self.tp):
            _ = arenas[rank][].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        var pools = HeapMoveArray[BurstPool](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool.for_numa_node(numa, topo[rank]))

        var model = Self(arenas^, pools^)

        for rank in range(Self.tp):
            var rv = model.rank(rank)
            init_rope_tables(rv.rope_cos(), rv.rope_sin(), Float64(C.ROPE_THETA))

        return model^

    fn forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[C.VOCAB_SIZE]
        where Self.E.DTYPE == DType.bfloat16:
        comptime M = Self.M
        comptime L = M.LAYER
        var prof = Profiler(profile)

        var ranks = Ranks[Self.E, Self.tp](self.bases.copy(), self.pool_ptrs.copy())
        var host = ranks.view(0)

        # --- Embed (host rank, then broadcast) ---
        prof.section("embed")
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), ranks.pool_ptrs[0][]).join()
        ring_broadcast[M.X_MAIN, Self.tp](host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len)

        for layer_idx in range(C.NUM_LAYERS):

            # --- RMSNorm ---
            prof.section("rmsnorm")
            @parameter
            fn do_input_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.INPUT_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_input_norm]()

            # --- Q/K/V projections ---
            prof.section("gemm_q")
            @parameter
            fn do_q[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.Q_PROJ](layer_idx), rv.q_scratch(seq_len), pool)
            ranks.parallel[do_q]()

            prof.section("gemm_k")
            @parameter
            fn do_k[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.K_PROJ](layer_idx), rv.kv_scratch(0, seq_len), pool)
            ranks.parallel[do_k]()

            prof.section("gemm_v")
            @parameter
            fn do_v[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.V_PROJ](layer_idx), rv.kv_scratch(1, seq_len), pool)
            ranks.parallel[do_v]()

            # --- RoPE (caller-thread) ---
            prof.section("rope")
            @parameter
            fn do_rope(rv: RankView[Self.E, Self.tp]):
                rope[C.HEAD_DIM, M.LOCAL_HEADS](rv.q_scratch(seq_len), rv.rope_cos(), rv.rope_sin(), pos)
                rope[C.HEAD_DIM, M.LOCAL_KV_HEADS](rv.kv_scratch(0, seq_len), rv.rope_cos(), rv.rope_sin(), pos)
            ranks.each[do_rope]()

            # --- KV cache write (caller-thread) ---
            prof.section("kv_write")
            @parameter
            fn do_kv_write(rv: RankView[Self.E, Self.tp]):
                kv_cache_write(rv.kv_scratch(0, seq_len), rv.k_cache(layer_idx), pos)
                kv_cache_write(rv.kv_scratch(1, seq_len), rv.v_cache(layer_idx), pos)
            ranks.each[do_kv_write]()

            # --- Attention ---
            prof.section("attention")
            @parameter
            fn do_attn[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return attention[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM](
                    rv.q_scratch(seq_len), rv.k_cache(layer_idx), rv.v_cache(layer_idx),
                    rv.q_scratch(seq_len), pos, pool)
            ranks.parallel[do_attn]()

            # --- O projection + allreduce + residual ---
            prof.section("gemm_o")
            @parameter
            fn do_o[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.q_scratch(seq_len), rv.layer_weight[L.O_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_o]()

            prof.section("allreduce_o")
            allreduce_sum_missing_stub[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len)

            prof.section("elem_add")
            @parameter
            fn do_res_add(rv: RankView[Self.E, Self.tp]):
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))
            ranks.each[do_res_add]()

            # --- MLP: RMSNorm ---
            prof.section("rmsnorm")
            @parameter
            fn do_post_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.POST_ATTN_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_post_norm]()

            # --- Gate/Up projections ---
            prof.section("gemm_gate")
            @parameter
            fn do_gate[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.GATE_PROJ](layer_idx), rv.mlp_scratch(0, seq_len), pool)
            ranks.parallel[do_gate]()

            prof.section("gemm_up")
            @parameter
            fn do_up[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.UP_PROJ](layer_idx), rv.mlp_scratch(1, seq_len), pool)
            ranks.parallel[do_up]()

            # --- SiLU * Up (caller-thread) ---
            prof.section("silu_mul")
            @parameter
            fn do_silu(rv: RankView[Self.E, Self.tp]):
                silu_mul(rv.mlp_scratch(0, seq_len), rv.mlp_scratch(1, seq_len), rv.mlp_scratch(0, seq_len))
            ranks.each[do_silu]()

            # --- Down projection + allreduce + residual ---
            prof.section("gemm_down")
            @parameter
            fn do_down[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool) -> PoolFence:
                return gemm(rv.mlp_scratch(0, seq_len), rv.layer_weight[L.DOWN_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_down]()

            prof.section("allreduce_down")
            allreduce_sum_missing_stub[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len)

            prof.section("elem_add")
            ranks.each[do_res_add]()

        # --- Final norm + LM head (host rank only) ---
        prof.section("final_norm")
        rmsnorm(host.x_main(seq_len), host.weight[M.FINAL_NORM](), host.x_main(seq_len), ranks.pool_ptrs[0][]).join()

        var logits = DynView[M.LOGITS](host.state_base() + M.Q_SCRATCH_OFF, seq_len)

        prof.section("lm_head")
        gemm(host.x_main(seq_len), host.weight[M.EMBED](), logits, ranks.pool_ptrs[0][]).join()

        prof.finish()
        prof.report()

        return LogitsView[C.VOCAB_SIZE](logits.ptr, seq_len)


# =============================================================================
# Entry point — TP=3
# =============================================================================

comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"


fn main():
    var model_opt = SmolLM2TP[BF16, 3].load(Path(MODEL_PATH))
    if not model_opt:
        return
    var model = model_opt.take()

    var rank0 = model.rank(0)
    var tp_ptr = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=rank0.state_base() + TPModel[BF16, 3].Q_SCRATCH_OFF,
    )
    tp_ptr[0] = Scalar[DType.int32](42)
    var logits = model.forward(
        rank0.state_base() + TPModel[BF16, 3].Q_SCRATCH_OFF, 1, 0, profile=True,
    )
    _ = logits
