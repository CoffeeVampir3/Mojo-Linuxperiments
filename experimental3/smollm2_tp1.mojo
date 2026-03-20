"""SmolLM2 135M TP1 — descriptor-driven, operational forward pass.

Descriptors are the single source of truth. They drive:
  - Spec generation (for the loader)
  - Byte accounting (for workspace/arena sizing)
  - Tile construction (for kernel dispatch via tile_from[desc])
"""

from collections import Dict
from memory import UnsafePointer

from experimental3.core import (
    StaticModelDescriptor,
    StaticTensorSpec,
    align_up_int,
    LAYER_NONE,
    AXIS_NONE,
    AXIS_HOST,
)
from experimental3.loader import (
    DEFAULT_IO_QUEUE_DEPTH,
    LoadedStaticPackage,
    LoadedTensorRecord,
    TensorShardView,
    load_static_package,
    make_slot_key,
)
from experimental3.tensor_contracts import (
    Descriptor,
    DenseDesc,
    BufferDesc,
    ActTile,
    CacheTile,
    tile_from,
    buffer_tile_from,
    storage_bytes,
)
from experimental3.kernel_ops import (
    gemm,
    rmsnorm,
    silu_mul,
    elem_add,
    embed_lookup,
    init_rope_tables,
    rope,
    kv_cache_write,
    attention,
    broadcast,
    all_reduce,
)
from experimental3.logits import LogitsView
from experimental3.model_traits import Inference
from threading import BurstPool
from numa import NumaInfo
from time import perf_counter_ns


# ================================================================
# MODEL CONSTANTS
# ================================================================

comptime TP = 1
comptime HIDDEN = 576
comptime INTERMEDIATE = 1536
comptime KV_HIDDEN = 192
comptime VOCAB = 49152
comptime NUM_LAYERS = 30
comptime NUM_HEADS = 9
comptime NUM_KV_HEADS = 3
comptime HEAD_DIM = 64
comptime MAX_SEQ = 8192
comptime BATCH = 1
comptime BF16 = DType.bfloat16
comptime DTYPE_BYTES = 2
comptime WEIGHT_ALIGNMENT = 64
comptime ROPE_THETA = 100000.0
comptime MANIFEST_PATH = "experimental3/smollm2_tp1_manifest.json"


# ================================================================
# WEIGHT DESCRIPTORS
# ================================================================

# Host tensors — single copy on host node, not replicated
comptime EmbedDesc = DenseDesc[1, BF16, VOCAB, HIDDEN, AXIS_HOST, "model.embed_tokens.weight"]
comptime FinalNormDesc = DenseDesc[2, BF16, 1, HIDDEN, AXIS_HOST, "model.norm.weight"]

# Per-layer tensors
comptime InputLNDesc = DenseDesc[3, BF16, 1, HIDDEN, AXIS_NONE, "input_layernorm.weight"]
comptime QProjDesc = DenseDesc[4, BF16, HIDDEN, HIDDEN, 0, "self_attn.q_proj.weight"]
comptime KProjDesc = DenseDesc[5, BF16, KV_HIDDEN, HIDDEN, 0, "self_attn.k_proj.weight"]
comptime VProjDesc = DenseDesc[6, BF16, KV_HIDDEN, HIDDEN, 0, "self_attn.v_proj.weight"]
comptime OProjDesc = DenseDesc[7, BF16, HIDDEN, HIDDEN, 1, "self_attn.o_proj.weight"]
comptime PostLNDesc = DenseDesc[8, BF16, 1, HIDDEN, AXIS_NONE, "post_attention_layernorm.weight"]
comptime GateProjDesc = DenseDesc[9, BF16, INTERMEDIATE, HIDDEN, 0, "mlp.gate_proj.weight"]
comptime UpProjDesc = DenseDesc[10, BF16, INTERMEDIATE, HIDDEN, 0, "mlp.up_proj.weight"]
comptime DownProjDesc = DenseDesc[11, BF16, HIDDEN, INTERMEDIATE, 1, "mlp.down_proj.weight"]


# ================================================================
# BUFFER DESCRIPTORS — precomputed at init, not loaded from checkpoint
# ================================================================

comptime ROPE_HALF = HEAD_DIM // 2
comptime RopeCosBuffer = BufferDesc[DType.float32, MAX_SEQ, ROPE_HALF]
comptime RopeSinBuffer = BufferDesc[DType.float32, MAX_SEQ, ROPE_HALF]


# ================================================================
# SPEC GENERATION
# ================================================================

fn build_tensor_specs() -> List[StaticTensorSpec]:
    var specs = List[StaticTensorSpec]()
    EmbedDesc.emit_specs(LAYER_NONE, specs)
    FinalNormDesc.emit_specs(LAYER_NONE, specs)
    for layer in range(NUM_LAYERS):
        InputLNDesc.emit_specs(layer, specs)
        QProjDesc.emit_specs(layer, specs)
        KProjDesc.emit_specs(layer, specs)
        VProjDesc.emit_specs(layer, specs)
        OProjDesc.emit_specs(layer, specs)
        PostLNDesc.emit_specs(layer, specs)
        GateProjDesc.emit_specs(layer, specs)
        UpProjDesc.emit_specs(layer, specs)
        DownProjDesc.emit_specs(layer, specs)
    return specs^


# ================================================================
# WORKSPACE
# ================================================================

comptime TOKENS = BATCH * MAX_SEQ
comptime ACT_BYTES = TOKENS * HIDDEN * DTYPE_BYTES
comptime SCRATCH_SLOT_BYTES = TOKENS * INTERMEDIATE * DTYPE_BYTES
comptime KV_BYTES_PER_LAYER = 2 * MAX_SEQ * KV_HIDDEN * DTYPE_BYTES


fn compute_state_bytes(alignment: Int) -> Int:
    var kv_cache = NUM_LAYERS * KV_BYTES_PER_LAYER
    var x_main = align_up_int(ACT_BYTES, alignment)
    var x_res = align_up_int(ACT_BYTES, alignment)
    var scratch = 3 * align_up_int(SCRATCH_SLOT_BYTES, alignment)
    var rope_cos = align_up_int(storage_bytes[RopeCosBuffer](), alignment)
    var rope_sin = align_up_int(storage_bytes[RopeSinBuffer](), alignment)
    return kv_cache + x_main + x_res + scratch + rope_cos + rope_sin


@fieldwise_init
struct Workspace(Copyable):
    var kv_cache_base: Int
    var x_main_ptr: Int
    var x_residual_ptr: Int
    var scratch_base: Int
    var scratch_slot_bytes: Int
    var rope_cos_ptr: Int
    var rope_sin_ptr: Int


fn make_workspace(state_base: Int, alignment: Int) -> Workspace:
    var kv_cache_base = state_base
    var off = state_base + NUM_LAYERS * KV_BYTES_PER_LAYER
    var x_main_ptr = off
    off += align_up_int(ACT_BYTES, alignment)
    var x_residual_ptr = off
    off += align_up_int(ACT_BYTES, alignment)
    var scratch_base = off
    var scratch_slot_bytes = align_up_int(SCRATCH_SLOT_BYTES, alignment)
    off += 3 * scratch_slot_bytes
    var rope_cos_ptr = off
    off += align_up_int(storage_bytes[RopeCosBuffer](), alignment)
    var rope_sin_ptr = off
    return Workspace(kv_cache_base, x_main_ptr, x_residual_ptr, scratch_base, scratch_slot_bytes, rope_cos_ptr, rope_sin_ptr)


# ================================================================
# MODEL
# ================================================================

@fieldwise_init
struct SmolLM2Model(Inference):
    """Loaded SmolLM2-135M model. Owns the arena (weights + state),
    weight pointers, workspace layout, and thread pool.
    Dropping this struct frees all model memory."""
    comptime VOCAB = VOCAB
    var package: LoadedStaticPackage
    var ptrs: ModelPtrs
    var ws: Workspace
    var pool: BurstPool

    fn forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
    ) -> LogitsView[VOCAB]:
        return forward(self.ptrs, self.ws, tokens_ptr, seq_len, pos, self.pool, False)

    fn forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
        profile: Bool = False,
    ) -> LogitsView[VOCAB]:
        return forward(self.ptrs, self.ws, tokens_ptr, seq_len, pos, self.pool, profile)

    fn scratch_ptr(self) -> Int:
        """Address of scratch memory for writing token IDs before forward().
        Safe to use as tokens_ptr — consumed by embed_lookup before
        scratch is reused for intermediates."""
        return self.ws.scratch_base


fn load_smollm2(manifest_path: String = MANIFEST_PATH) -> Optional[SmolLM2Model]:
    """Load SmolLM2-135M from a manifest file. Returns the model ready
    for inference, or None on failure. Handles all setup: arena allocation,
    weight loading, pointer binding, RoPE init, and thread pool creation."""
    var package_opt = load_static_package[
        SmolLM2TP1Descriptor, DEFAULT_IO_QUEUE_DEPTH, WEIGHT_ALIGNMENT
    ](manifest_path)
    if not package_opt:
        print("Failed to load static package")
        return None
    var package = package_opt.take()

    var node_id = package.nodes[0].node_id
    var ptrs_opt = bind_model_ptrs(package.records, node_id)
    if not ptrs_opt:
        print("Failed to bind model weights")
        return None
    var ptrs = ptrs_opt.take()

    var ws = make_workspace(package.nodes[0].state_base, WEIGHT_ALIGNMENT)
    init_rope_tables(
        buffer_tile_from[RopeCosBuffer](ws.rope_cos_ptr),
        buffer_tile_from[RopeSinBuffer](ws.rope_sin_ptr),
        ROPE_THETA,
    )

    var numa = NumaInfo()
    var pool = BurstPool.for_numa_node(numa, 0)

    return SmolLM2Model(package^, ptrs^, ws.copy(), pool^)


# ================================================================
# MODEL DESCRIPTOR
# ================================================================

struct SmolLM2TP1Descriptor(StaticModelDescriptor):
    comptime TP = 1
    comptime NODE_COUNT = 1
    comptime NUM_LAYERS = NUM_LAYERS

    @staticmethod
    fn model_id() -> String:
        return "smollm2-135m-tp1"

    @staticmethod
    fn node_ids() -> List[Int]:
        var ids = List[Int]()
        ids.append(0)
        return ids^

    @staticmethod
    fn host_node_index() -> Int:
        return 0

    @staticmethod
    fn tensor_specs() -> List[StaticTensorSpec]:
        return build_tensor_specs()

    @staticmethod
    fn state_bytes_per_node(alignment: Int) -> Int:
        return compute_state_bytes(alignment)


# ================================================================
# WEIGHT BINDING
# ================================================================

fn require_ptr[desc: Descriptor](
    records: Dict[Int, LoadedTensorRecord],
    layer_idx: Int,
    node_id: Int,
) -> Optional[Int]:
    var key = make_slot_key(layer_idx, UInt16(desc.SLOT_ID))
    var existing_opt = records.get(key)
    if not existing_opt:
        print(
            "Missing tensor: slot", desc.SLOT_ID,
            "layer", layer_idx,
            "name", desc.tensor_name(layer_idx),
        )
        return None
    var record = existing_opt.value().copy()
    for i in range(len(record.shards)):
        if record.shards[i].node_id == node_id:
            return record.shards[i].ptr
    print(
        "Missing shard for node", node_id,
        "slot", desc.SLOT_ID,
        "name", desc.tensor_name(layer_idx),
    )
    return None


@fieldwise_init
struct LayerPtrs(Copyable):
    var input_ln: Int
    var q: Int
    var k: Int
    var v: Int
    var o: Int
    var post_ln: Int
    var gate: Int
    var up: Int
    var down: Int


@fieldwise_init
struct ModelPtrs(Movable):
    var embed: Int
    var final_norm: Int
    var layers: List[LayerPtrs]


fn bind_model_ptrs(
    records: Dict[Int, LoadedTensorRecord],
    node_id: Int,
) -> Optional[ModelPtrs]:
    var embed_opt = require_ptr[EmbedDesc](records, LAYER_NONE, node_id)
    if not embed_opt:
        return None
    var norm_opt = require_ptr[FinalNormDesc](records, LAYER_NONE, node_id)
    if not norm_opt:
        return None

    var layers = List[LayerPtrs]()
    for layer_idx in range(NUM_LAYERS):
        var iln = require_ptr[InputLNDesc](records, layer_idx, node_id)
        var q = require_ptr[QProjDesc](records, layer_idx, node_id)
        var k = require_ptr[KProjDesc](records, layer_idx, node_id)
        var v = require_ptr[VProjDesc](records, layer_idx, node_id)
        var o = require_ptr[OProjDesc](records, layer_idx, node_id)
        var pln = require_ptr[PostLNDesc](records, layer_idx, node_id)
        var gate = require_ptr[GateProjDesc](records, layer_idx, node_id)
        var up = require_ptr[UpProjDesc](records, layer_idx, node_id)
        var down = require_ptr[DownProjDesc](records, layer_idx, node_id)

        if not iln or not q or not k or not v or not o or not pln or not gate or not up or not down:
            return None

        layers.append(LayerPtrs(
            iln.value(), q.value(), k.value(), v.value(), o.value(),
            pln.value(), gate.value(), up.value(), down.value(),
        ))

    return ModelPtrs(embed_opt.value(), norm_opt.value(), layers^)


# ================================================================
# FORWARD PASS — operational, explicit
# ================================================================

fn probe_bf16(label: String, ptr: Int, cols: Int, row: Int = 0):
    """Print first 4 values and scan for NaN in one row."""
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=ptr + row * cols * 2
    )
    var nan_count = 0
    var min_val = Float32(1e30)
    var max_val = Float32(-1e30)
    for i in range(cols):
        var v = p[i].cast[DType.float32]()
        if v != v:
            nan_count += 1
        else:
            if v < min_val:
                min_val = v
            if v > max_val:
                max_val = v
    print(
        label,
        "| [0:4]:",
        p[0].cast[DType.float32](),
        p[1].cast[DType.float32](),
        p[2].cast[DType.float32](),
        p[3].cast[DType.float32](),
        "| nan:", nan_count,
        "| range: [", min_val, ",", max_val, "]",
    )


fn forward(
    ptrs: ModelPtrs, ws: Workspace,
    tokens_ptr: Int, seq_len: Int, pos: Int,
    mut pool: BurstPool,
    profile: Bool = False,
) -> LogitsView[VOCAB]:
    """Run the full forward pass, return a read-only view of the logits.
    Logits are written to scratch memory — valid until the next forward call."""

    # Timing accumulators (ns)
    var t_embed: UInt = 0
    var t_rmsnorm: UInt = 0
    var t_gemm_qkv: UInt = 0
    var t_rope: UInt = 0
    var t_kv_write: UInt = 0
    var t_attention: UInt = 0
    var t_gemm_o: UInt = 0
    var t_elem_add: UInt = 0
    var t_gemm_gate_up: UInt = 0
    var t_silu_mul: UInt = 0
    var t_gemm_down: UInt = 0
    var t_final_norm: UInt = 0
    var t_lm_head: UInt = 0
    var t: UInt

    var x_main = ActTile[BF16, HIDDEN](ws.x_main_ptr, seq_len)
    var x_res = ActTile[BF16, HIDDEN](ws.x_residual_ptr, seq_len)
    var rope_cos = buffer_tile_from[RopeCosBuffer](ws.rope_cos_ptr)
    var rope_sin = buffer_tile_from[RopeSinBuffer](ws.rope_sin_ptr)

    t = perf_counter_ns()
    embed_lookup(tile_from[EmbedDesc](ptrs.embed), tokens_ptr, x_main, pool)
    t_embed += perf_counter_ns() - t
    broadcast(x_main, TP)

    for layer_idx in range(NUM_LAYERS):
        var lp = ptrs.layers[layer_idx].copy()
        var layer_kv = ws.kv_cache_base + layer_idx * KV_BYTES_PER_LAYER
        var k_cache = CacheTile[BF16, KV_HIDDEN, MAX_SEQ](layer_kv)
        var v_cache = CacheTile[BF16, KV_HIDDEN, MAX_SEQ](layer_kv + MAX_SEQ * KV_HIDDEN * DTYPE_BYTES)

        # --- Q/K/V projections ---
        var sq = ActTile[BF16, HIDDEN](ws.scratch_base, seq_len)
        var sk = ActTile[BF16, KV_HIDDEN](ws.scratch_base + ws.scratch_slot_bytes, seq_len)
        var sv = ActTile[BF16, KV_HIDDEN](ws.scratch_base + 2 * ws.scratch_slot_bytes, seq_len)

        t = perf_counter_ns()
        rmsnorm(x_main, tile_from[InputLNDesc](lp.input_ln), x_res, pool)
        t_rmsnorm += perf_counter_ns() - t

        t = perf_counter_ns()
        gemm(x_res, tile_from[QProjDesc](lp.q), sq, pool)
        gemm(x_res, tile_from[KProjDesc](lp.k), sk, pool)
        gemm(x_res, tile_from[VProjDesc](lp.v), sv, pool)
        t_gemm_qkv += perf_counter_ns() - t

        # --- Attention ---
        t = perf_counter_ns()
        rope[HEAD_DIM, NUM_HEADS](sq, rope_cos, rope_sin, pos)
        rope[HEAD_DIM, NUM_KV_HEADS](sk, rope_cos, rope_sin, pos)
        t_rope += perf_counter_ns() - t

        t = perf_counter_ns()
        kv_cache_write(sk, k_cache, pos)
        kv_cache_write(sv, v_cache, pos)
        t_kv_write += perf_counter_ns() - t

        var attn_out = ActTile[BF16, HIDDEN](ws.scratch_base + ws.scratch_slot_bytes, seq_len)
        t = perf_counter_ns()
        attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](sq, k_cache, v_cache, pos, attn_out, pool)
        t_attention += perf_counter_ns() - t

        t = perf_counter_ns()
        gemm(attn_out, tile_from[OProjDesc](lp.o), x_res, pool)
        t_gemm_o += perf_counter_ns() - t

        all_reduce(x_res, TP)
        t = perf_counter_ns()
        elem_add(x_main, x_res, x_main)
        t_elem_add += perf_counter_ns() - t

        # --- FFN ---
        var sg = ActTile[BF16, INTERMEDIATE](ws.scratch_base, seq_len)
        var su = ActTile[BF16, INTERMEDIATE](ws.scratch_base + ws.scratch_slot_bytes, seq_len)

        t = perf_counter_ns()
        rmsnorm(x_main, tile_from[PostLNDesc](lp.post_ln), x_res, pool)
        t_rmsnorm += perf_counter_ns() - t

        t = perf_counter_ns()
        gemm(x_res, tile_from[GateProjDesc](lp.gate), sg, pool)
        gemm(x_res, tile_from[UpProjDesc](lp.up), su, pool)
        t_gemm_gate_up += perf_counter_ns() - t

        t = perf_counter_ns()
        silu_mul(sg, su, sg)
        t_silu_mul += perf_counter_ns() - t

        t = perf_counter_ns()
        gemm(sg, tile_from[DownProjDesc](lp.down), x_res, pool)
        t_gemm_down += perf_counter_ns() - t

        all_reduce(x_res, TP)
        t = perf_counter_ns()
        elem_add(x_main, x_res, x_main)
        t_elem_add += perf_counter_ns() - t

    # --- Final norm + lm_head ---
    t = perf_counter_ns()
    rmsnorm(x_main, tile_from[FinalNormDesc](ptrs.final_norm), x_main, pool)
    t_final_norm += perf_counter_ns() - t

    t = perf_counter_ns()
    gemm(x_main, tile_from[EmbedDesc](ptrs.embed), ActTile[BF16, VOCAB](ws.scratch_base, seq_len), pool)
    t_lm_head += perf_counter_ns() - t

    if profile:
        var total = (t_embed + t_rmsnorm + t_gemm_qkv + t_rope + t_kv_write
            + t_attention + t_gemm_o + t_elem_add + t_gemm_gate_up
            + t_silu_mul + t_gemm_down + t_final_norm + t_lm_head)
        print("--- forward profile (us) ---")
        print("  embed:       ", Int(t_embed / 1_000))
        print("  rmsnorm:     ", Int(t_rmsnorm / 1_000))
        print("  gemm_qkv:    ", Int(t_gemm_qkv / 1_000))
        print("  rope:        ", Int(t_rope / 1_000))
        print("  kv_write:    ", Int(t_kv_write / 1_000))
        print("  attention:   ", Int(t_attention / 1_000))
        print("  gemm_o:      ", Int(t_gemm_o / 1_000))
        print("  gemm_gate_up:", Int(t_gemm_gate_up / 1_000))
        print("  silu_mul:    ", Int(t_silu_mul / 1_000))
        print("  gemm_down:   ", Int(t_gemm_down / 1_000))
        print("  elem_add:    ", Int(t_elem_add / 1_000))
        print("  final_norm:  ", Int(t_final_norm / 1_000))
        print("  lm_head:     ", Int(t_lm_head / 1_000))
        print("  total:       ", Int(total / 1_000))

    return LogitsView[VOCAB](ws.scratch_base, seq_len)


# ================================================================
# MAIN
# ================================================================

fn main():
    var model_opt = load_smollm2()
    if not model_opt:
        return
    var model = model_opt.take()

    var tp = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=model.scratch_ptr()
    )
    tp[0] = Scalar[DType.int32](42)
    var logits = model.forward(model.scratch_ptr(), 1, 0)
    _ = logits
