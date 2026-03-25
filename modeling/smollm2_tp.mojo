"""SmolLM2-135M with parametric tensor parallelism.

TP degree is a comptime parameter. Valid values must cleanly divide
NUM_HEADS, NUM_KV_HEADS, and INTERMEDIATE. For SmolLM2-135M:
  TP=1 (trivial), TP=3 (3 query heads, 1 KV head per rank).

Each rank has its own NUMA arena, BurstPool, and activation buffers.
Megatron-style: RowShard for output projections (no comm), ColShard
for input projections (allreduce after). Dispatch via parallel_for.
"""

from std.pathlib import Path

from std.memory import UnsafePointer
from std.collections import InlineArray
from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32,
    RowShard, ColShard, Replicated,
    PrincipleNodeLocal,
    IsQuantizable, IsPassthrough,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
    WeightIterable,
    next_offset,
    DEFAULT_ALIGNMENT,
    Dims, Attention, GQA, FFN, Vocab, Sequence, RoPEConfig, RMSNormConfig,
)
from kernels.kernel_ops import (
    gemm, rmsnorm, embed_lookup, silu_mul, elem_add, rope, kv_cache_write,
    attention, init_rope_tables,
    PoolFence, parallel, parallel_for,
)
from kernels.reductions import ring_allreduce, ring_broadcast
from modeling.loader import load_safetensors
from kernels.profiler import Profiler


# =============================================================================
# Shared types: model config, logit access
# =============================================================================


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


# =============================================================================
# Parametric model spec
# =============================================================================

comptime C = SmolLM2Config


struct TPLayer[E: Encoding, tp: Int]:
    comptime Q_PROJ      = PlacedSlot[Self.E, RowShard, C.HIDDEN, C.HIDDEN, Self.tp, 0, "self_attn.q_proj.weight", IsQuantizable]
    comptime K_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.Q_PROJ](), "self_attn.k_proj.weight", IsQuantizable]
    comptime V_PROJ      = PlacedSlot[Self.E, RowShard, C.KV_HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.K_PROJ](), "self_attn.v_proj.weight", IsQuantizable]
    comptime O_PROJ      = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.HIDDEN, Self.tp, next_offset[Self.V_PROJ](), "self_attn.o_proj.weight", IsQuantizable]
    comptime GATE_PROJ   = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.O_PROJ](), "mlp.gate_proj.weight", IsQuantizable]
    comptime UP_PROJ     = PlacedSlot[Self.E, RowShard, C.INTERMEDIATE, C.HIDDEN, Self.tp, next_offset[Self.GATE_PROJ](), "mlp.up_proj.weight", IsQuantizable]
    comptime DOWN_PROJ   = PlacedSlot[Self.E, ColShard, C.HIDDEN, C.INTERMEDIATE, Self.tp, next_offset[Self.UP_PROJ](), "mlp.down_proj.weight", IsQuantizable]
    comptime INPUT_NORM  = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.DOWN_PROJ](), "input_layernorm.weight"]
    comptime POST_ATTN_NORM = PlacedSlot[BF16, Replicated, C.HIDDEN, 1, Self.tp, next_offset[Self.INPUT_NORM](), "post_attention_layernorm.weight"]
    comptime STRIDE      = next_offset[Self.POST_ATTN_NORM]()

    comptime K_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]

    @staticmethod
    def for_each_weight[
        func: def[T: Encoding & Shaped & Placed & Named] (String, Int) capturing -> None,
    ](prefix: String, base: Int):
        func[Self.Q_PROJ](prefix, base)
        func[Self.K_PROJ](prefix, base)
        func[Self.V_PROJ](prefix, base)
        func[Self.O_PROJ](prefix, base)
        func[Self.GATE_PROJ](prefix, base)
        func[Self.UP_PROJ](prefix, base)
        func[Self.DOWN_PROJ](prefix, base)
        func[Self.INPUT_NORM](prefix, base)
        func[Self.POST_ATTN_NORM](prefix, base)

    @staticmethod
    def cache_bytes() -> Int:
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
    comptime LOGITS = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.VOCAB_SIZE, Self.tp]

    # Shared scratch: 3 slots, each full INTERMEDIATE width (no sharding).
    # Scratch is working memory — always full-width regardless of TP.
    # Reuse pattern per layer:
    #   slot0: Q output      → gate output → logits (host only)
    #   slot1: K output      → attn output → up output
    #   slot2: V output
    comptime SCRATCH = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime SCRATCH_COUNT = 3

    # Typed views into scratch slots (same memory, narrower logical width).
    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime KV_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]

    # Per-rank state layout.
    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_STRIDE = byte_count[Self.SCRATCH]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE
    comptime ROPE_SIN_OFF = Self.ROPE_COS_OFF + byte_count[Self.ROPE_COS]()
    comptime STATE_BYTES = Self.ROPE_SIN_OFF + byte_count[Self.ROPE_SIN]()

    # NodeLocal weights (host arena only).
    comptime NODE_LOCAL_OFF = ((Self.DISTRIBUTED_BYTES + Self.STATE_BYTES + DEFAULT_ALIGNMENT - 1) // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
    comptime FINAL_NORM = PlacedSlot[BF16, PrincipleNodeLocal, C.HIDDEN, 1, Self.tp, Self.NODE_LOCAL_OFF, "model.norm.weight"]
    comptime EMBED = PlacedSlot[Self.E, PrincipleNodeLocal, C.VOCAB_SIZE, C.HIDDEN, Self.tp, next_offset[Self.FINAL_NORM](), "model.embed_tokens.weight"]

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


# =============================================================================
# Per-rank state accessor
# =============================================================================


struct RankView[E: Encoding, tp: Int]:
    comptime M = TPModel[Self.E, Self.tp]
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

    def k_cache(self, layer: Int) -> CacheView[Self.L.K_CACHE]:
        return CacheView[Self.L.K_CACHE](self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE)

    def v_cache(self, layer: Int) -> CacheView[Self.L.V_CACHE]:
        return CacheView[Self.L.V_CACHE](
            self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE + byte_count[Self.L.K_CACHE]()
        )

    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    def scratch_slot(self, index: Int) -> Int:
        """Raw address of scratch slot `index` (0, 1, or 2)."""
        return self.state_base() + Self.M.SCRATCH_OFF + index * Self.M.SCRATCH_STRIDE

    def q_view(self, seq_len: Int) -> DynView[Self.M.Q_VIEW]:
        """Q output / attn input — slot 0, Q-width view."""
        return DynView[Self.M.Q_VIEW](self.scratch_slot(0), seq_len)

    def attn_out_view(self, seq_len: Int) -> DynView[Self.M.Q_VIEW]:
        """Attention output — slot 1, Q-width view (reuses K slot after KV cache write)."""
        return DynView[Self.M.Q_VIEW](self.scratch_slot(1), seq_len)

    def kv_view(self, index: Int, seq_len: Int) -> DynView[Self.M.KV_VIEW]:
        """K (index=0, slot 1) or V (index=1, slot 2) — KV-width view."""
        return DynView[Self.M.KV_VIEW](self.scratch_slot(index + 1), seq_len)

    def mlp_view(self, index: Int, seq_len: Int) -> DynView[Self.M.MLP_VIEW]:
        """Gate (index=0, slot 0) or Up (index=1, slot 1) — per-rank intermediate width."""
        return DynView[Self.M.MLP_VIEW](self.scratch_slot(index), seq_len)

    def rope_cos(self) -> Bound[Self.M.ROPE_COS]:
        return Bound[Self.M.ROPE_COS](self.state_base() + Self.M.ROPE_COS_OFF)

    def rope_sin(self) -> Bound[Self.M.ROPE_SIN]:
        return Bound[Self.M.ROPE_SIN](self.state_base() + Self.M.ROPE_SIN_OFF)


# =============================================================================
# Ranks — dispatch helper
# =============================================================================


@fieldwise_init
struct Ranks[E: Encoding, tp: Int]:
    """Rank-indexed dispatch helper. Captures base addresses and pool pointers
    for use in parallel_for closures and sequential loops."""
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def view(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    def parallel[body: def[rank: Int] (RankView[Self.E, Self.tp], mut BurstPool[]) capturing -> PoolFence](self):
        """Dispatch body(rv, pool) for each rank in parallel, then join all."""
        @parameter
        def dispatch[rank: Int]() -> PoolFence:
            var rv = RankView[Self.E, Self.tp](self.bases[rank])
            return body[rank](rv, self.pool_ptrs[rank][])
        parallel_for[Self.tp, dispatch]()

    def each[body: def (RankView[Self.E, Self.tp]) capturing -> None](self):
        """Run body(rv) for each rank sequentially on the caller thread."""
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
# Loaded model
# =============================================================================


struct SmolLM2TP[E: Encoding, tp: Int](Movable):
    comptime M = TPModel[Self.E, Self.tp]

    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[BurstPool[]]
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def __init__(out self, var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
                var pools: HeapMoveArray[BurstPool[]]):
        self.bases = InlineArray[Int, Self.tp](fill=0)
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
        )
        self.arenas = arenas^
        self.pools = pools^
        for rank in range(Self.tp):
            self.bases[rank] = Int(self.arenas[rank].base)
            self.pool_ptrs[rank] = UnsafePointer[BurstPool[], MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.pools[rank]))
            )

    def rank(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    @staticmethod
    def load(path: Path) -> Optional[Self]:
        """Load SmolLM2 with automatic NUMA-aware rank placement.
        Discovers topology, selects the tightest node cluster,
        and orders ranks for optimal ring allreduce adjacency."""
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
                print("TP: arena allocation failed for rank", rank, "on node", topo[rank])
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var result = load_safetensors[Self.M](path, arena_bases, host_index=host_rank)
        if not result:
            print("TP: weight loading failed")
            return None

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(Self.M.DISTRIBUTED_BYTES, Self.M.STATE_BYTES)

        var pools = HeapMoveArray[BurstPool[]](Self.tp)
        for rank in range(Self.tp):
            pools.push(BurstPool[].for_numa_node(numa, topo[rank]))

        var model = Self(arenas^, pools^)

        for rank in range(Self.tp):
            var rv = model.rank(rank)
            init_rope_tables(rv.rope_cos(), rv.rope_sin(), Float64(C.ROPE_THETA))

        return model^

    def forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[C.VOCAB_SIZE]
        where Self.E.DTYPE == DType.bfloat16:
        comptime M = Self.M
        comptime L = M.LAYER
        var prof = Profiler(profile)

        var ranks = Ranks[Self.E, Self.tp](self.bases, self.pool_ptrs)
        var host = ranks.view(0)

        # --- Embed (host rank, then broadcast) ---
        embed_lookup(host.weight[M.EMBED](), tokens_ptr, host.x_main(seq_len), ranks.pool_ptrs[0][]).join()
        ring_broadcast[M.X_MAIN, Self.tp](host.x_main(seq_len).ptr, ranks.x_main_ptrs(seq_len), seq_len, ranks.pool_ptrs)

        for layer_idx in range(C.NUM_LAYERS):
            @parameter
            def do_input_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.INPUT_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_input_norm]()

            @parameter
            def do_q[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.Q_PROJ](layer_idx), rv.q_view(seq_len), pool)
            ranks.parallel[do_q]()

            @parameter
            def do_k[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.K_PROJ](layer_idx), rv.kv_view(0, seq_len), pool)
            ranks.parallel[do_k]()

            @parameter
            def do_v[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.V_PROJ](layer_idx), rv.kv_view(1, seq_len), pool)
            ranks.parallel[do_v]()

            @parameter
            def do_rope(rv: RankView[Self.E, Self.tp]):
                rope[C.HEAD_DIM, M.LOCAL_HEADS](rv.q_view(seq_len), rv.rope_cos(), rv.rope_sin(), pos)
                rope[C.HEAD_DIM, M.LOCAL_KV_HEADS](rv.kv_view(0, seq_len), rv.rope_cos(), rv.rope_sin(), pos)
            ranks.each[do_rope]()

            @parameter
            def do_kv_write(rv: RankView[Self.E, Self.tp]):
                kv_cache_write(rv.kv_view(0, seq_len), rv.k_cache(layer_idx), pos)
                kv_cache_write(rv.kv_view(1, seq_len), rv.v_cache(layer_idx), pos)
            ranks.each[do_kv_write]()

            @parameter
            def do_attn[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return attention[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM](
                    rv.q_view(seq_len), rv.k_cache(layer_idx), rv.v_cache(layer_idx),
                    rv.attn_out_view(seq_len), pos, pool)
            ranks.parallel[do_attn]()

            @parameter
            def do_o[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.attn_out_view(seq_len), rv.layer_weight[L.O_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_o]()

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_res_add(rv: RankView[Self.E, Self.tp]):
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))
            ranks.each[do_res_add]()

            @parameter
            def do_post_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.POST_ATTN_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_post_norm]()

            @parameter
            def do_gate[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.GATE_PROJ](layer_idx), rv.mlp_view(0, seq_len), pool)
            ranks.parallel[do_gate]()

            @parameter
            def do_up[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.UP_PROJ](layer_idx), rv.mlp_view(1, seq_len), pool)
            ranks.parallel[do_up]()

            @parameter
            def do_silu(rv: RankView[Self.E, Self.tp]):
                silu_mul(rv.mlp_view(0, seq_len), rv.mlp_view(1, seq_len), rv.mlp_view(0, seq_len))
            ranks.each[do_silu]()

            @parameter
            def do_down[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.mlp_view(0, seq_len), rv.layer_weight[L.DOWN_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_down]()

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            ranks.each[do_res_add]()

            _ = layer_idx  # Anchor: closures capture layer_idx but compiler doesn't track it

        # --- Final norm + LM head (host rank only) ---
        rmsnorm(host.x_main(seq_len), host.weight[M.FINAL_NORM](), host.x_main(seq_len), ranks.pool_ptrs[0][]).join()

        var logits = DynView[M.LOGITS](host.scratch_slot(0), seq_len)
        gemm(host.x_main(seq_len), host.weight[M.EMBED](), logits, ranks.pool_ptrs[0][]).join()
        prof.finish()
        prof.report()

        return LogitsView[C.VOCAB_SIZE](logits.ptr, seq_len)


# =============================================================================
# Entry point — TP=3
# =============================================================================

comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"


def main():
    var model_opt = SmolLM2TP[BF16, 3].load(Path(MODEL_PATH))
    if not model_opt:
        return
    var model = model_opt.take()

    var rank0 = model.rank(0)
    var tokens_addr = rank0.scratch_slot(0)
    var tp_ptr = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=tokens_addr,
    )
    tp_ptr[0] = Scalar[DType.int32](42)
    var logits = model.forward(tokens_addr, 1, 0, profile=True)
    _ = logits
