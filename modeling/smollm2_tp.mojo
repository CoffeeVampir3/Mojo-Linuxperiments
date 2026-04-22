from std.pathlib import Path
from std.memory import UnsafePointer
from std.collections import InlineArray

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading import BurstPool
from threading.threading_traits import BurstThreadPool

from modeling.linear_borrow_pool import ScratchLease, ScratchPool, scratch_block_bytes

from modeling.model_spec import (
    BF16, F32,
    Shape, ShapeLike, Mat, CacheView, WeightDesc,
    DEFAULT_ALIGNMENT, HOST_RANK, LogitsView,
)
from modeling.modeling_common import (
    SlotOffset, Repeated, SectionBuilder, align_up,
    DynamicTensorView, dynamic_tensor_view,
    static_tensor_view, scratch_tensor_view, scratch_ptr,
    LayerShard, LayerBuilder,
)
from modeling.loader import discover_shards, load_weights_from_descs
from kernels.kernel_ops import (
    gemm, rmsnorm, embed_lookup, silu_mul, elem_add, kv_cache_write,
    attention,
    PoolFence, parallel_for,
)
from kernels.kv_rotors import init_rope_tables, rope
from kernels.reductions import ring_allreduce, ring_broadcast
from kernels.profiler import Profiler


struct SmolLM2Config:
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


struct SmolLM2Shapes[tp: Int]:
    comptime QProj = Shape[C.HIDDEN, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime KVProj = Shape[C.KV_HIDDEN, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime OProj = Shape[C.HIDDEN, C.HIDDEN, shard_m=True, tp=Self.tp]
    comptime GateUp = Shape[C.INTERMEDIATE, C.HIDDEN, shard_n=True, tp=Self.tp]
    comptime Down = Shape[C.HIDDEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp]

    comptime QAct = Shape[C.MAX_SEQ_LEN, C.HIDDEN, shard_m=True, tp=Self.tp]
    comptime KVAct = Shape[C.MAX_SEQ_LEN, C.KV_HIDDEN, shard_m=True, tp=Self.tp]
    comptime MLPAct = Shape[C.MAX_SEQ_LEN, C.INTERMEDIATE, shard_m=True, tp=Self.tp]

    comptime LocalHeads = C.NUM_HEADS // Self.tp
    comptime LocalKVHeads = C.NUM_KV_HEADS // Self.tp
    comptime RopeHalf = C.HEAD_DIM // 2


@fieldwise_init
struct AttentionRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = SmolLM2Shapes[Self.tp]
    var q_proj: SlotOffset[BF16, Self.S.QProj]
    var k_proj: SlotOffset[BF16, Self.S.KVProj]
    var v_proj: SlotOffset[BF16, Self.S.KVProj]
    var o_proj: SlotOffset[BF16, Self.S.OProj]


@fieldwise_init
struct BodyRefs[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = SmolLM2Shapes[Self.tp]
    var input_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var gate_proj: SlotOffset[BF16, Self.S.GateUp]
    var up_proj: SlotOffset[BF16, Self.S.GateUp]
    var down_proj: SlotOffset[BF16, Self.S.Down]


@fieldwise_init
struct LayerRefs[tp: Int](Copyable, ImplicitlyCopyable):
    var attn: AttentionRefs[Self.tp]
    var body: BodyRefs[Self.tp]


@fieldwise_init
struct KVSlots[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = SmolLM2Shapes[Self.tp]
    var k: SlotOffset[BF16, Self.S.KVAct]
    var v: SlotOffset[BF16, Self.S.KVAct]


@fieldwise_init
struct RopeSlots[half: Int](Copyable, ImplicitlyCopyable):
    var cos: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]
    var sin: SlotOffset[F32, Shape[C.MAX_SEQ_LEN, Self.half]]


@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: SlotOffset[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


@fieldwise_init
struct HostSlots(Copyable, ImplicitlyCopyable):
    var final_norm: SlotOffset[BF16, Shape[C.HIDDEN, 1]]
    var embed: SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]]


@fieldwise_init
struct SmolLM2Topology[tp: Int](Copyable, ImplicitlyCopyable):
    var arena_base: Int
    var layers: Repeated[LayerRefs[Self.tp]]
    var distributed_bytes: Int

    var kv: Repeated[KVSlots[Self.tp]]
    var activations: ActivationSlots
    var scratch_off: Int
    var scratch_capacity: Int
    var rope: RopeSlots[C.HEAD_DIM // 2]
    var state_bytes: Int

    var host: HostSlots
    var host_bytes: Int

    def bind(self, base: Int) -> Self:
        var t = self
        t.arena_base = base
        return t

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    @always_inline
    def layer_base(self, idx: Int) -> Int:
        return self.arena_base + self.layers.off + idx * self.layers.stride

    @always_inline
    def state_base(self) -> Int:
        return self.arena_base + self.distributed_bytes

    @always_inline
    def scratch_base(self) -> Int:
        return self.state_base() + self.scratch_off

    @always_inline
    def scratch_addr(self, read lease: ScratchLease) -> Int:
        return self.scratch_base() + lease.offset

    @always_inline
    def x_main(self, seq_len: Int) -> DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]:
        return dynamic_tensor_view(self.state_base(), self.activations.x_main, seq_len)

    @always_inline
    def x_residual(self, seq_len: Int) -> DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]:
        return dynamic_tensor_view(self.state_base(), self.activations.x_residual, seq_len)

    @always_inline
    def k_cache(self, layer_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_HIDDEN // Self.tp]]:
        var kb = self.kv.base(self.state_base(), layer_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_HIDDEN // Self.tp]](
            kb + self.kv.proto.k.offset)

    @always_inline
    def v_cache(self, layer_idx: Int) -> CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_HIDDEN // Self.tp]]:
        var kb = self.kv.base(self.state_base(), layer_idx)
        return CacheView[Mat[BF16, C.MAX_SEQ_LEN, C.KV_HIDDEN // Self.tp]](
            kb + self.kv.proto.v.offset)


def emit_body[tp: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[tp]:
    comptime S = SmolLM2Shapes[tp]
    comptime REPL = LayerShard.REPL
    comptime H = C.HIDDEN
    return BodyRefs[tp](
        input_norm=SlotOffset[BF16, Shape[H, 1]](
            b.bf(e, "input_layernorm.weight", H, 1, REPL)),
        post_attn_norm=SlotOffset[BF16, Shape[H, 1]](
            b.bf(e, "post_attention_layernorm.weight", H, 1, REPL)),
        gate_proj=SlotOffset[BF16, S.GateUp](
            b.bfs[S.GateUp](e, "mlp.gate_proj.weight")),
        up_proj=SlotOffset[BF16, S.GateUp](
            b.bfs[S.GateUp](e, "mlp.up_proj.weight")),
        down_proj=SlotOffset[BF16, S.Down](
            b.bfs[S.Down](e, "mlp.down_proj.weight")),
    )


def emit_layer[tp: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[LayerRefs[tp], Int]:
    var b = LayerBuilder(tp, prefix, layer_base)
    comptime S = SmolLM2Shapes[tp]
    var attn = AttentionRefs[tp](
        q_proj=SlotOffset[BF16, S.QProj](b.bfs[S.QProj](e, "self_attn.q_proj.weight")),
        k_proj=SlotOffset[BF16, S.KVProj](b.bfs[S.KVProj](e, "self_attn.k_proj.weight")),
        v_proj=SlotOffset[BF16, S.KVProj](b.bfs[S.KVProj](e, "self_attn.v_proj.weight")),
        o_proj=SlotOffset[BF16, S.OProj](b.bfs[S.OProj](e, "self_attn.o_proj.weight")),
    )
    return (LayerRefs[tp](attn=attn, body=emit_body[tp](b, e)), b.cursor)


def calculate_peak_scratch[tp: Int]() -> Int:
    comptime S = SmolLM2Shapes[tp]
    comptime bf16 = BF16.ELEMENT_BYTES

    comptime q_raw_bytes = S.QAct.bytes_for[bf16]()
    comptime kv_raw_bytes = S.KVAct.bytes_for[bf16]()
    comptime mlp_raw_bytes = S.MLPAct.bytes_for[bf16]()
    comptime q_bytes = scratch_block_bytes[q_raw_bytes]()
    comptime kv_bytes = scratch_block_bytes[kv_raw_bytes]()
    comptime mlp_bytes = scratch_block_bytes[mlp_raw_bytes]()

    comptime attn_qkv = q_bytes + kv_bytes + kv_bytes
    comptime attn_out = q_bytes + q_bytes
    comptime attn_peak = attn_qkv if attn_qkv > attn_out else attn_out
    comptime mlp_peak = mlp_bytes + mlp_bytes
    return attn_peak if attn_peak > mlp_peak else mlp_peak


@fieldwise_init
struct SmolLM2LoadPlan[tp: Int](Movable):
    var topology: SmolLM2Topology[Self.tp]
    var descs: List[WeightDesc]


def build_smollm2_plan[tp: Int]() -> SmolLM2LoadPlan[tp]:
    debug_assert(C.NUM_HEADS % tp == 0, "NUM_HEADS % tp")
    debug_assert(C.NUM_KV_HEADS % tp == 0, "NUM_KV_HEADS % tp")
    debug_assert(C.INTERMEDIATE % tp == 0, "INTERMEDIATE % tp")

    var descs = List[WeightDesc]()

    var probe = List[WeightDesc]()
    var layer_r = emit_layer[tp]("", 0, probe)
    var layer_proto = layer_r[0]
    var layer_stride = layer_r[1]
    var distributed = C.NUM_LAYERS * layer_stride

    for i in range(C.NUM_LAYERS):
        var prefix = "model.layers." + String(i) + "."
        _ = emit_layer[tp](prefix, i * layer_stride, descs)

    var state = SectionBuilder()

    var kv_sb = SectionBuilder()
    var kv_proto = KVSlots[tp](
        k=kv_sb.reserve[BF16, SmolLM2Shapes[tp].KVAct](),
        v=kv_sb.reserve[BF16, SmolLM2Shapes[tp].KVAct]())
    var kv = Repeated[KVSlots[tp]](
        kv_proto, state.cursor, kv_sb.bytes(), C.NUM_LAYERS)
    _ = state.reserve_bytes(C.NUM_LAYERS * kv_sb.bytes())

    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]())

    var scratch_cap = calculate_peak_scratch[tp]()
    var scratch_off = state.reserve_bytes(scratch_cap)

    var rope = RopeSlots[C.HEAD_DIM // 2](
        cos=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM // 2]](),
        sin=state.reserve[F32, Shape[C.MAX_SEQ_LEN, C.HEAD_DIM // 2]]())

    var host_off = align_up(distributed + state.bytes())
    comptime HOST = LayerShard.HOST
    var hb = LayerBuilder(tp, "", 0)
    hb.cursor = host_off
    var host = HostSlots(
        final_norm=SlotOffset[BF16, Shape[C.HIDDEN, 1]](
            hb.bf(descs, "model.norm.weight", C.HIDDEN, 1, HOST)),
        embed=SlotOffset[BF16, Shape[C.VOCAB_SIZE, C.HIDDEN]](
            hb.bf(descs, "model.embed_tokens.weight", C.VOCAB_SIZE, C.HIDDEN, HOST)))

    var topo = SmolLM2Topology[tp](
        arena_base=0,
        layers=Repeated[LayerRefs[tp]](layer_proto, 0, layer_stride, C.NUM_LAYERS),
        distributed_bytes=distributed,
        kv=kv,
        activations=activations,
        scratch_off=scratch_off,
        scratch_capacity=scratch_cap,
        rope=rope,
        state_bytes=state.bytes(),
        host=host,
        host_bytes=hb.cursor,
    )
    return SmolLM2LoadPlan[tp](topo, descs^)


def tp_parallel[
    Pool: BurstThreadPool, //,
    tp: Int,
    body: def[rank: Int](SmolLM2Topology[tp], mut Pool) capturing -> PoolFence[Pool],
](
    topos: InlineArray[SmolLM2Topology[tp], tp],
    pool_ptrs: InlineArray[UnsafePointer[Pool, MutAnyOrigin], tp],
):
    @parameter
    def dispatch[rank: Int]() -> PoolFence[Pool]:
        return body[rank](topos[rank], pool_ptrs[rank][])
    parallel_for[tp, dispatch]()


struct SmolLM2TP[tp: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[Self.Pool]
    var scratch: ScratchPool
    var topos: InlineArray[SmolLM2Topology[Self.tp], Self.tp]

    def __init__(
        out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[Self.Pool],
        var scratch: ScratchPool,
        topos: InlineArray[SmolLM2Topology[Self.tp], Self.tp],
    ):
        self.arenas = arenas^
        self.pools = pools^
        self.scratch = scratch^
        self.topos = topos

    @staticmethod
    def print_memory():
        comptime arena_per_rank = build_smollm2_plan[Self.tp]().topology.arena_bytes()
        comptime host_arena = build_smollm2_plan[Self.tp]().topology.host_arena_bytes()
        comptime total = host_arena + (Self.tp - 1) * arena_per_rank

        print("SmolLM2 TP=" + String(Self.tp) + ": "
            + String(total // (1024 * 1024)) + " MB total")
        if Self.tp == 1:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
        else:
            print("  rank 0 (host): " + String(host_arena // (1024 * 1024)) + " MB")
            comptime for r in range(1, Self.tp):
                print("  rank " + String(r) + ":        "
                    + String(arena_per_rank // (1024 * 1024)) + " MB")

    def init_state(mut self):
        for rank in range(Self.tp):
            var topo = self.topos[rank]
            var sb = topo.state_base()
            init_rope_tables(
                topo.rope.cos.bound(sb),
                topo.rope.sin.bound(sb),
                Float64(C.ROPE_THETA))

    def token_buffer(mut self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.topos[0].scratch_base())

    def pool_ptrs(self) -> InlineArray[UnsafePointer[Self.Pool, MutAnyOrigin], Self.tp]:
        var ptrs = InlineArray[UnsafePointer[Self.Pool, MutAnyOrigin], Self.tp](
            fill=UnsafePointer[Self.Pool, MutAnyOrigin]())
        for rank in range(Self.tp):
            ptrs[rank] = UnsafePointer[Self.Pool, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.pools[rank])))
        return ptrs^

    def x_main_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for rank in range(Self.tp):
            ptrs[rank] = self.topos[rank].x_main(seq_len).ptr
        return ptrs^

    def x_residual_ptrs(self, seq_len: Int) -> InlineArray[Int, Self.tp]:
        var ptrs = InlineArray[Int, Self.tp](fill=0)
        for rank in range(Self.tp):
            ptrs[rank] = self.topos[rank].x_residual(seq_len).ptr
        return ptrs^

    @staticmethod
    def load(
        dir_path: Path,
        numa: NumaInfo,
        numa_topo: NumaTopology,
        var pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_smollm2_plan[Self.tp]()
        var topo = plan.topology

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.tp)
        for rank in range(Self.tp):
            var size = topo.host_arena_bytes() if rank == HOST_RANK else topo.arena_bytes()
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](numa_topo[rank], size)
            if not arena:
                print("arena allocation failed for rank", rank)
                return None
            arenas.push(arena^)

        var arena_bases = List[Int]()
        for rank in range(Self.tp):
            arena_bases.append(Int(arenas[rank].base))

        var load_result = load_weights_from_descs(plan.descs, shards, arena_bases, numa_topo)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.tp):
            _ = arenas[rank].prefault(topo.distributed_bytes, topo.state_bytes)

        var topos = InlineArray[SmolLM2Topology[Self.tp], Self.tp](fill=topo)
        for rank in range(Self.tp):
            topos[rank] = topo.bind(Int(arenas[rank].base))

        var scratch = ScratchPool(topo.scratch_capacity)
        var model = Self(arenas^, pools^, scratch^, topos)
        model.init_state()
        return model^

    def forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[C.VOCAB_SIZE]:
        comptime S = SmolLM2Shapes[Self.tp]
        comptime XSlot = Mat[BF16, C.MAX_SEQ_LEN, C.HIDDEN]

        var prof = Profiler(profile)
        debug_assert(seq_len > 0 and pos >= 0 and pos + seq_len <= C.MAX_SEQ_LEN,
            "forward: sequence range exceeds MAX_SEQ_LEN")

        var topos = self.topos
        var host = topos[0]
        var mp = self.pool_ptrs()
        var x_main = host.x_main(seq_len)
        var embed = static_tensor_view(host.arena_base, host.host.embed)
        var final_norm = static_tensor_view(host.arena_base, host.host.final_norm)

        embed_lookup(embed, tokens_ptr, x_main, self.pools[0]).join()
        ring_broadcast[XSlot, Self.tp](
            x_main.ptr, self.x_main_ptrs(seq_len), seq_len, mp)

        for layer_idx in range(C.NUM_LAYERS):
            var layer = host.layers.proto

            var q_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.QAct.M]()
            var k_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.KVAct.M]()
            var v_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.KVAct.M]()

            @parameter
            def do_input_norm[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return rmsnorm(
                    topo.x_main(seq_len),
                    static_tensor_view(lb, layer.body.input_norm),
                    topo.x_residual(seq_len),
                    pool,
                    Float32(C.RMS_NORM_EPS))
            tp_parallel[Self.tp, do_input_norm](topos, mp)

            @parameter
            def do_q[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    topo.x_residual(seq_len),
                    static_tensor_view(lb, layer.attn.q_proj),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.QAct.M](
                        topo.scratch_base(), q_lease, seq_len),
                    pool)
            tp_parallel[Self.tp, do_q](topos, mp)

            @parameter
            def do_k[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    topo.x_residual(seq_len),
                    static_tensor_view(lb, layer.attn.k_proj),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.KVAct.M](
                        topo.scratch_base(), k_lease, seq_len),
                    pool)
            tp_parallel[Self.tp, do_k](topos, mp)

            @parameter
            def do_v[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    topo.x_residual(seq_len),
                    static_tensor_view(lb, layer.attn.v_proj),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.KVAct.M](
                        topo.scratch_base(), v_lease, seq_len),
                    pool)
            tp_parallel[Self.tp, do_v](topos, mp)

            for rank in range(Self.tp):
                var topo = topos[rank]
                var sb = topo.state_base()
                rope[C.HEAD_DIM, S.LocalHeads](
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.QAct.M](
                        topo.scratch_base(), q_lease, seq_len),
                    static_tensor_view(sb, topo.rope.cos),
                    static_tensor_view(sb, topo.rope.sin),
                    pos)
                rope[C.HEAD_DIM, S.LocalKVHeads](
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.KVAct.M](
                        topo.scratch_base(), k_lease, seq_len),
                    static_tensor_view(sb, topo.rope.cos),
                    static_tensor_view(sb, topo.rope.sin),
                    pos)
                kv_cache_write(
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.KVAct.M](
                        topo.scratch_base(), k_lease, seq_len),
                    topo.k_cache(layer_idx),
                    pos)
                kv_cache_write(
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.KVAct.M](
                        topo.scratch_base(), v_lease, seq_len),
                    topo.v_cache(layer_idx),
                    pos)

            v_lease^.release()
            k_lease^.release()

            var attn_out_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.QAct.M]()

            @parameter
            def do_attn[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                return attention[S.LocalHeads, S.LocalKVHeads, C.HEAD_DIM](
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.QAct.M](
                        topo.scratch_base(), q_lease, seq_len),
                    topo.k_cache(layer_idx),
                    topo.v_cache(layer_idx),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.QAct.M](
                        topo.scratch_base(), attn_out_lease, seq_len),
                    pos,
                    pool)
            tp_parallel[Self.tp, do_attn](topos, mp)

            @parameter
            def do_o[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.QAct.M](
                        topo.scratch_base(), attn_out_lease, seq_len),
                    static_tensor_view(lb, layer.attn.o_proj),
                    topo.x_residual(seq_len),
                    pool)
            tp_parallel[Self.tp, do_o](topos, mp)

            attn_out_lease^.release()
            q_lease^.release()

            ring_allreduce[XSlot, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len, mp)

            for rank in range(Self.tp):
                var topo = topos[rank]
                elem_add(
                    topo.x_main(seq_len),
                    topo.x_residual(seq_len),
                    topo.x_main(seq_len))

            @parameter
            def do_post_norm[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return rmsnorm(
                    topo.x_main(seq_len),
                    static_tensor_view(lb, layer.body.post_attn_norm),
                    topo.x_residual(seq_len),
                    pool,
                    Float32(C.RMS_NORM_EPS))
            tp_parallel[Self.tp, do_post_norm](topos, mp)

            var gate_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.MLPAct.M]()
            var up_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.MAX_SEQ_LEN * S.MLPAct.M]()

            @parameter
            def do_gate[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    topo.x_residual(seq_len),
                    static_tensor_view(lb, layer.body.gate_proj),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), gate_lease, seq_len),
                    pool)
            tp_parallel[Self.tp, do_gate](topos, mp)

            @parameter
            def do_up[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    topo.x_residual(seq_len),
                    static_tensor_view(lb, layer.body.up_proj),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), up_lease, seq_len),
                    pool)
            tp_parallel[Self.tp, do_up](topos, mp)

            for rank in range(Self.tp):
                var topo = topos[rank]
                silu_mul(
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), gate_lease, seq_len),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), up_lease, seq_len),
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), gate_lease, seq_len))

            up_lease^.release()

            @parameter
            def do_down[rank: Int](
                topo: SmolLM2Topology[Self.tp], mut pool: Self.Pool,
            ) -> PoolFence[Self.Pool]:
                var lb = topo.layer_base(layer_idx)
                return gemm(
                    scratch_tensor_view[BF16, C.MAX_SEQ_LEN, S.MLPAct.M](
                        topo.scratch_base(), gate_lease, seq_len),
                    static_tensor_view(lb, layer.body.down_proj),
                    topo.x_residual(seq_len),
                    pool)
            tp_parallel[Self.tp, do_down](topos, mp)

            gate_lease^.release()

            ring_allreduce[XSlot, Self.tp](
                self.x_residual_ptrs(seq_len), seq_len, mp)

            for rank in range(Self.tp):
                var topo = topos[rank]
                elem_add(
                    topo.x_main(seq_len),
                    topo.x_residual(seq_len),
                    topo.x_main(seq_len))

        rmsnorm(
            x_main,
            final_norm,
            x_main,
            self.pools[0],
            Float32(C.RMS_NORM_EPS)).join()

        var last_row_off = (seq_len - 1) * C.HIDDEN * BF16.ELEMENT_BYTES
        var last_hidden = DynamicTensorView[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](
            x_main.ptr + last_row_off, 1)
        var logit_lease = self.scratch.borrow[Scalar[DType.bfloat16], C.VOCAB_SIZE]()
        var logit_view = scratch_tensor_view[BF16, 1, C.VOCAB_SIZE](
            host.scratch_base(), logit_lease, 1)
        gemm(last_hidden, embed, logit_view, self.pools[0]).join()
        prof.finish()
        prof.report()

        return LogitsView[C.VOCAB_SIZE](
            scratch_ptr[Scalar[DType.bfloat16]](host.scratch_base(), logit_lease),
            logit_lease^)
