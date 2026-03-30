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
    LogitsView,
)
from kernels.kernel_ops import (
    gemm, rmsnorm, embed_lookup, silu_mul, elem_add, rope, kv_cache_write,
    attention, init_rope_tables,
    PoolFence, parallel_for,
)
from kernels.reductions import ring_allreduce, ring_broadcast
from modeling.loader import load_safetensors
from kernels.profiler import Profiler
from experimental.linear_borrow_pool import ScratchPool, ScratchLease


# =============================================================================
# Shared types: model config, logit access
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

    # Typed DynView slots — used to construct views over borrowed scratch.
    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime KV_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]

    # Scratch capacity: derived from the peak phase.
    comptime SCRATCH_CAPACITY = Self.calculate_peak_scratch()

    @staticmethod
    def calculate_peak_scratch() -> Int:
        """Peak scratch bytes across both phases of one layer.

        Each phase creates a fresh ScratchPool. The pool is a bump
        allocator (no reclaim), so the peak is the cumulative sum of
        all borrows within the largest phase.

        Attention phase borrows (in order, all bf16):
            q:        MAX_SEQ_LEN * HIDDEN/tp * 2
            k:        MAX_SEQ_LEN * KV_HIDDEN/tp * 2
            v:        MAX_SEQ_LEN * KV_HIDDEN/tp * 2
            attn_out: MAX_SEQ_LEN * HIDDEN/tp * 2

        MLP phase borrows (in order, all bf16):
            gate:     MAX_SEQ_LEN * INTERMEDIATE/tp * 2
            up:       MAX_SEQ_LEN * INTERMEDIATE/tp * 2

        Post-loop: logits reuse scratch from offset 0 (fresh pool).
        An assert verifies they fit within the layer-phase capacity.
        """
        comptime S = C.MAX_SEQ_LEN
        comptime H = C.HIDDEN
        comptime KV = C.KV_HIDDEN
        comptime I = C.INTERMEDIATE
        comptime TP = Self.tp

        comptime attn_peak = (
            S * (H // TP) * 2      # q
            + S * (KV // TP) * 2   # k
            + S * (KV // TP) * 2   # v
            + S * (H // TP) * 2    # attn_out
        )

        comptime mlp_peak = (
            S * (I // TP) * 2      # gate
            + S * (I // TP) * 2    # up
        )

        comptime if attn_peak > mlp_peak:
            return attn_peak
        else:
            return mlp_peak

    # Per-rank state layout.
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

    def scratch_base(self) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF

    def scratch_view[V: Encoding & Shaped](self, read lease: ScratchLease, seq_len: Int) -> DynView[V]:
        """Materialize a DynView from a scratch lease offset."""
        return DynView[V](self.scratch_base() + lease.offset, seq_len)

    def scratch_ptr[T: AnyType](self, read lease: ScratchLease) -> UnsafePointer[T, MutAnyOrigin]:
        """Materialize a typed pointer from a scratch lease offset."""
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.scratch_base() + lease.offset)

    def rope_cos(self) -> Bound[Self.M.ROPE_COS]:
        return Bound[Self.M.ROPE_COS](self.state_base() + Self.M.ROPE_COS_OFF)

    def rope_sin(self) -> Bound[Self.M.ROPE_SIN]:
        return Bound[Self.M.ROPE_SIN](self.state_base() + Self.M.ROPE_SIN_OFF)


# =============================================================================
# Ranks — dispatch helper
# =============================================================================


@fieldwise_init
struct Ranks[E: Encoding, tp: Int]:
    """Rank-indexed dispatch helper."""
    var bases: InlineArray[Int, Self.tp]
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], Self.tp]

    def view(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    def parallel[body: def[rank: Int] (RankView[Self.E, Self.tp], mut BurstPool[]) capturing -> PoolFence](self):
        @parameter
        def dispatch[rank: Int]() -> PoolFence:
            var rv = RankView[Self.E, Self.tp](self.bases[rank])
            return body[rank](rv, self.pool_ptrs[rank][])
        parallel_for[Self.tp, dispatch]()

    def each[body: def (RankView[Self.E, Self.tp]) capturing -> None](self):
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

    def rank(self, r: Int) -> RankView[Self.E, Self.tp]:
        return RankView[Self.E, Self.tp](self.bases[r])

    @staticmethod
    def print_memory():
        """Print total memory required to run this model."""
        comptime arena_per_rank = Self.M.arena_bytes()
        comptime host_arena = Self.M.host_arena_bytes()
        comptime total = host_arena + (Self.tp - 1) * arena_per_rank

        print("SmolLM2 TP=" + String(Self.tp) + ": " + String(total // (1024 * 1024)) + " MB total")
        comptime if Self.tp == 1:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
        else:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
            comptime for r in range(1, Self.tp):
                print("  rank " + String(r) + ":        " + String(arena_per_rank // (1024 * 1024)) + " MB")

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        """Pointer for writing input token IDs. Points to rank 0's scratch region.
        Valid until the next forward() call (which borrows scratch for computation).
        """
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.rank(0).state_base() + Self.M.SCRATCH_OFF
        )

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

            # === Attention block ===

            # Borrow scratch for Q, K, V (offsets, same for every rank)
            var q = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.Q_VIEW.COLS]()
            var k = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.KV_VIEW.COLS]()
            var v = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.KV_VIEW.COLS]()

            # This serves as a way to get the right pointers in parallel easily.
            @parameter
            def do_input_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.INPUT_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_input_norm]()

            @parameter
            def do_q[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.Q_PROJ](layer_idx), rv.scratch_view[M.Q_VIEW](q, seq_len), pool)
            ranks.parallel[do_q]()

            @parameter
            def do_k[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.K_PROJ](layer_idx), rv.scratch_view[M.KV_VIEW](k, seq_len), pool)
            ranks.parallel[do_k]()

            @parameter
            def do_v[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.V_PROJ](layer_idx), rv.scratch_view[M.KV_VIEW](v, seq_len), pool)
            ranks.parallel[do_v]()

            @parameter
            def do_rope(rv: RankView[Self.E, Self.tp]):
                rope[C.HEAD_DIM, M.LOCAL_HEADS](rv.scratch_view[M.Q_VIEW](q, seq_len), rv.rope_cos(), rv.rope_sin(), pos)
                rope[C.HEAD_DIM, M.LOCAL_KV_HEADS](rv.scratch_view[M.KV_VIEW](k, seq_len), rv.rope_cos(), rv.rope_sin(), pos)
            ranks.each[do_rope]()

            @parameter
            def do_kv_write(rv: RankView[Self.E, Self.tp]):
                kv_cache_write(rv.scratch_view[M.KV_VIEW](k, seq_len), rv.k_cache(layer_idx), pos)
                kv_cache_write(rv.scratch_view[M.KV_VIEW](v, seq_len), rv.v_cache(layer_idx), pos)
            ranks.each[do_kv_write]()

            # K, V written to cache — release and borrow attn_out in their place
            v^.release()
            k^.release()

            var attn_out = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.Q_VIEW.COLS]()

            @parameter
            def do_attn[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return attention[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM](
                    rv.scratch_view[M.Q_VIEW](q, seq_len), rv.k_cache(layer_idx), rv.v_cache(layer_idx),
                    rv.scratch_view[M.Q_VIEW](attn_out, seq_len), pos, pool)
            ranks.parallel[do_attn]()

            q^.release()

            @parameter
            def do_o[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.scratch_view[M.Q_VIEW](attn_out, seq_len), rv.layer_weight[L.O_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_o]()

            attn_out^.release()

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_res_add(rv: RankView[Self.E, Self.tp]):
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))
            ranks.each[do_res_add]()

            # === MLP block ===

            @parameter
            def do_post_norm[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rmsnorm(rv.x_main(seq_len), rv.layer_weight[L.POST_ATTN_NORM](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_post_norm]()

            var gate = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.MLP_VIEW.COLS]()
            var up = self.scratch.borrow[Scalar[Self.E.DTYPE], C.MAX_SEQ_LEN * M.MLP_VIEW.COLS]()

            @parameter
            def do_gate[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.GATE_PROJ](layer_idx), rv.scratch_view[M.MLP_VIEW](gate, seq_len), pool)
            ranks.parallel[do_gate]()

            @parameter
            def do_up[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.x_residual(seq_len), rv.layer_weight[L.UP_PROJ](layer_idx), rv.scratch_view[M.MLP_VIEW](up, seq_len), pool)
            ranks.parallel[do_up]()

            @parameter
            def do_silu(rv: RankView[Self.E, Self.tp]):
                silu_mul(rv.scratch_view[M.MLP_VIEW](gate, seq_len), rv.scratch_view[M.MLP_VIEW](up, seq_len), rv.scratch_view[M.MLP_VIEW](gate, seq_len))
            ranks.each[do_silu]()

            up^.release()

            @parameter
            def do_down[rank: Int](rv: RankView[Self.E, Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return gemm(rv.scratch_view[M.MLP_VIEW](gate, seq_len), rv.layer_weight[L.DOWN_PROJ](layer_idx), rv.x_residual(seq_len), pool)
            ranks.parallel[do_down]()

            gate^.release()

            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)
            ranks.each[do_res_add]()

            _ = layer_idx

        # --- Final norm + LM head (host rank only) ---
        rmsnorm(host.x_main(seq_len), host.weight[M.FINAL_NORM](), host.x_main(seq_len), ranks.pool_ptrs[0][]).join()

        var last_row_off = (seq_len - 1) * C.HIDDEN * M.X_MAIN.ELEMENT_BYTES
        var last_hidden = DynView[M.X_MAIN](host.x_main(seq_len).ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = host.scratch_view[M.LOGITS](logit_lease, 1)
        gemm(last_hidden, host.weight[M.EMBED](), logit_view, ranks.pool_ptrs[0][]).join()
        prof.finish()
        prof.report()

        return LogitsView[C.VOCAB_SIZE](
            host.scratch_ptr[Scalar[DType.bfloat16]](logit_lease), logit_lease^,
        )


# =============================================================================
# Entry point — TP=3
# =============================================================================

comptime MODEL_PATH = "checkpoints/SmolLM2/model.safetensors"


def main():
    var model_opt = SmolLM2TP[BF16, 3].load(Path(MODEL_PATH))
    if not model_opt:
        return
    var model = model_opt.take()

    var tp = model.token_buffer()
    tp[0] = Scalar[DType.int32](42)
    var logits = model.forward(Int(tp), 1, 0, profile=True)
    logits^.release()
