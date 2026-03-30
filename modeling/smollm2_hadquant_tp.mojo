"""SmolLM2-135M Hadamard-rotated int8 quantization.

Projection weights are FWHT-rotated and quantized to I8 with per-row F32
scales. RMSNorm gamma is absorbed into projection weights offline — norm
weights are consumed during quantization and absent from the quantized
file. At runtime, RMSNorm is just x / rms(x).

KV cache stores pre-rotated int8 with per-head F32 scales. K and V are
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
# Stub kernel signatures — correct contracts, printing bodies
#
# These define the interface the real kernels must satisfy. Each takes
# typed views + BurstPool, returns PoolFence. Zero allocation.
# =============================================================================


def rms_fwht_quantize[block: Int,
    InT: Encoding & Shaped, QiT: Encoding & Shaped, ScT: Encoding & Shaped](
    input: DynView[InT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence:
    """Fused RMSNorm + FWHT rotation + int8 quantization in one pass.

    Gamma is already absorbed into downstream weights. The implementation
    rotates the raw (unnormalized) input first, then computes rms and
    absmax in a single dual reduction over the rotated values. FWHT
    preserves norms (Parseval), so rms(FWHT(x)) = rms(x). The rms
    factor is folded into scale_out so the int8_gemm epilogue produces
    W * gamma * (x / rms(x)) = W * RMSNorm(x) without materializing
    a normalized bf16 intermediate.

    Implementation:
        x_rot = FWHT(x)                           # rotate raw input
        rms = sqrt(sum(x_rot^2) / K)              # = rms(x) by Parseval
        absmax = max(|x_rot|)
        scale_out[m] = absmax / (rms * 127)        # normalization folded in
        qi_out[m,k] = round(x_rot[k] * rms * 127 / absmax)
    """
    comptime assert InT.DTYPE == DType.bfloat16, "rms_fwht_quantize: input must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "rms_fwht_quantize: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "rms_fwht_quantize: scale output must be f32"
    print("    [stub] rms_fwht_quantize [" + String(input.seq_len)
          + "x" + String(InT.COLS) + "] block=" + String(block))
    return PoolFence.completed()


def int8_gqa_attention[num_heads: Int, num_kv_heads: Int, head_dim: Int,
    QT: Encoding & Shaped,
    KcT: Encoding & Shaped, VcT: Encoding & Shaped,
    KsT: Encoding & Shaped, VsT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    q: DynView[QT],
    k_cache: CacheView[KcT], k_cache_scale: CacheView[KsT],
    v_cache: CacheView[VcT], v_cache_scale: CacheView[VsT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Int8 GQA attention with fused RoPE on Q and pre-rotated int8 KV cache.

    Q is bf16 [seq_len, num_heads * head_dim] in original domain (no RoPE).
    K and V cache entries are int8, pre-rotated per head with per-head scales.

    pos is the cache position of the FIRST query in the batch:
      - Decode (seq_len=1): single query at position pos.
        Context = pos+1 cache entries. GEMV per head.
      - Prefill (seq_len>1): queries at positions pos..pos+seq_len-1.
        Causal mask: query m attends to cache positions 0..pos+m.
        GEMM per head with triangular mask.

    Per query row m, per head h:
      1. RoPE(Q[m, h], pos + m)
      2. FWHT + quantize Q per head
      3. Int8 scoring against cache[0..pos+m]: causal boundary = pos + m + 1
      4. Masked softmax in f32
      5. Absorb V per-entry scales into attention weights
      6. Quantize absorbed weights to int8
      7. Int8 aggregation over cache[0..pos+m]
      8. Quantize output → int8 directly

    Output is int8 + f32 scale in per-head rotated domain.
    """
    comptime assert QT.DTYPE == DType.bfloat16, "int8_gqa_attention: Q must be bf16"
    comptime assert KcT.DTYPE == DType.int8, "int8_gqa_attention: K cache must be int8"
    comptime assert VcT.DTYPE == DType.int8, "int8_gqa_attention: V cache must be int8"
    comptime assert QiT.DTYPE == DType.int8, "int8_gqa_attention: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gqa_attention: scale output must be f32"
    var ctx = pos + q.seq_len  # total cache entries after this batch
    print("    [stub] int8_gqa_attention seq_len=" + String(q.seq_len)
          + " heads=" + String(num_heads)
          + " ctx=" + String(ctx)
          + (" (decode)" if q.seq_len == 1 else " (prefill)"))
    return PoolFence.completed()


def int8_gemm_k_to_cache[block: Int, head_dim: Int, num_kv_heads: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    KcT: Encoding & Shaped, KsT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    k_cache: CacheView[KcT],
    k_cache_scale: CacheView[KsT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused K projection → cache: int8_gemm epilogue → RoPE → FWHT → quantize → write.

    Computes the K projection one head at a time (HEAD_DIM output elements).
    The tile epilogue applies RoPE + FWHT per head and writes int8 directly
    to the KV cache. K never materializes as bf16.

    Writes seq_len rows to cache positions pos..pos+seq_len-1.
    Row m gets RoPE at position pos+m.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_k_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_k_to_cache: weight must be int8"
    comptime assert KcT.DTYPE == DType.int8, "int8_gemm_k_to_cache: K cache must be int8"
    print("    [stub] int8_gemm_k_to_cache pos=" + String(pos)
          + " seq_len=" + String(act_qi.seq_len)
          + " [" + String(act_qi.seq_len) + "x" + String(WT.COLS)
          + "] -> K cache[" + String(pos) + ".." + String(pos + act_qi.seq_len - 1) + "]")
    return PoolFence.completed()


def int8_gemm_v_to_cache[block: Int, num_kv_heads: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    VcT: Encoding & Shaped, VsT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    v_cache: CacheView[VcT],
    v_cache_scale: CacheView[VsT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused V projection → cache: int8_gemm epilogue → FWHT → quantize → write.

    Same as K but without RoPE. V never materializes as bf16.
    Writes seq_len rows to cache positions pos..pos+seq_len-1.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_v_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_v_to_cache: weight must be int8"
    comptime assert VcT.DTYPE == DType.int8, "int8_gemm_v_to_cache: V cache must be int8"
    print("    [stub] int8_gemm_v_to_cache pos=" + String(pos)
          + " seq_len=" + String(act_qi.seq_len)
          + " [" + String(act_qi.seq_len) + "x" + String(WT.COLS)
          + "] -> V cache[" + String(pos) + ".." + String(pos + act_qi.seq_len - 1) + "]")
    return PoolFence.completed()


def int8_gemm_gate_up[
    GWT: Encoding & Shaped & Placed, GWsT: Encoding & Shaped & Placed, GCsT: Encoding & Shaped & Placed,
    UWT: Encoding & Shaped & Placed, UWsT: Encoding & Shaped & Placed, UCsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    GOutT: Encoding & Shaped, UOutT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    gate_weight: Bound[GWT], gate_scale: Bound[GWsT], gate_colsum: Bound[GCsT],
    up_weight: Bound[UWT], up_scale: Bound[UWsT], up_colsum: Bound[UCsT],
    gate_out: DynView[GOutT],
    up_out: DynView[UOutT],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused GATE+UP: reads int8 activation once, produces both bf16 outputs.

    Halves activation memory bandwidth vs two separate gemms.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_gate_up: act must be int8"
    comptime assert GOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: gate output must be bf16"
    comptime assert UOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: up output must be bf16"
    print("    [stub] int8_gemm_gate_up [" + String(act_qi.seq_len)
          + "x" + String(GWT.COLS) + "] -> gate[" + String(GWT.ROWS)
          + "] + up[" + String(UWT.ROWS) + "]")
    return PoolFence.completed()


def silu_fwht_quantize[block: Int,
    GT: Encoding & Shaped, UT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped](
    gate: DynView[GT],
    up: DynView[UT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused: SiLU(gate) * up → FWHT(blocks) → absmax → int8.

    Replaces silu_mul + separate quantize. The nonlinearity, rotation,
    and quantization share a single pass.

    silu_out[k] = gate[k] * sigmoid(gate[k]) * up[k]
    scale_out[m] = max(|FWHT(silu_out[m])|) / 127
    qi_out[m,k] = round(FWHT(silu_out[m])[k] / scale_out[m])
    """
    comptime assert GT.DTYPE == DType.bfloat16, "silu_fwht_quantize: gate must be bf16"
    comptime assert UT.DTYPE == DType.bfloat16, "silu_fwht_quantize: up must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "silu_fwht_quantize: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "silu_fwht_quantize: scale output must be f32"
    print("    [stub] silu_fwht_quantize [" + String(gate.seq_len)
          + "x" + String(GT.COLS) + "] block=" + String(block))
    return PoolFence.completed()


def rms_norm_no_gamma[InT: Encoding & Shaped, OutT: Encoding & Shaped](
    input: DynView[InT], output: DynView[OutT],
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence:
    """RMSNorm without gamma: output = input / rms(input).

    Used for final_norm where gamma is not absorbed (the final norm
    has no downstream projection to absorb into).
    """
    comptime assert InT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: output must be bf16"
    print("    [stub] rms_norm_no_gamma [" + String(input.seq_len)
          + "x" + String(InT.COLS) + "]")
    return PoolFence.completed()


def int8_gemm[
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    OutT: Encoding & Shaped,
](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    output: DynView[OutT],
    mut pool: BurstPool[],
) -> PoolFence:
    """Int8 GEMM with VNNI u8/i8 bias correction and per-row scale epilogue.

    Activations are stored as i8 but converted to u8 (XOR sign bit) at
    register load for vpdpbusd (u8 x i8 -> i32). This introduces a bias
    of 128 * column_sum[n] per output element, corrected in the epilogue.

    Implementation:
        raw_acc[m,n] = sum_k u8(act[m,k] ^ 0x80) * i8(weight[n,k])
        corrected    = float(raw_acc[m,n]) - 128.0 * weight_colsum[n]
        output[m,n]  = bf16(corrected * act_scale[m] * weight_scale[n])

    weight_colsum[n] = float(sum_k weight[n,k]) is precomputed at load time.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm: activation must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gemm: act scale must be f32"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm: weight must be int8"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemm: weight scale must be f32"
    comptime assert CsT.DTYPE == DType.float32, "int8_gemm: column sum must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "int8_gemm: output must be bf16"
    print("    [stub] int8_gemm [" + String(act_qi.seq_len)
          + "x" + String(WT.COLS) + "] -> [" + String(act_qi.seq_len)
          + "x" + String(WT.ROWS) + "]")
    return PoolFence.completed()


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

    # --- KV cache: int8 pre-rotated (FWHT per head at write time) ---

    comptime K_CACHE = Slot[I8, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime V_CACHE = Slot[I8, ColShard, C.MAX_SEQ_LEN, C.KV_HIDDEN, Self.tp]
    comptime K_CACHE_SCALE = Slot[F32, ColShard, C.MAX_SEQ_LEN, C.NUM_KV_HEADS, Self.tp]
    comptime V_CACHE_SCALE = Slot[F32, ColShard, C.MAX_SEQ_LEN, C.NUM_KV_HEADS, Self.tp]

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
        return (byte_count[Self.K_CACHE]() + byte_count[Self.V_CACHE]()
              + byte_count[Self.K_CACHE_SCALE]() + byte_count[Self.V_CACHE_SCALE]())


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

    # bf16 scratch: 3 slots at INTERMEDIATE width.
    # K and V go directly to int8 cache (no bf16 intermediate).
    # Reuse: slot0=Q/gate, slot1=up, slot2=int8 activation overlay
    comptime SCRATCH = Slot[BF16, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime SCRATCH_COUNT = 3

    # Typed views into scratch
    comptime Q_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime MLP_VIEW = Slot[BF16, ColShard, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]

    # --- Int8 activation scratch: overlaid on slot 2 ---
    #
    # Slot 2 is fully available — K and V never materialize as bf16
    # (they go directly from gemm epilogue to int8 cache).
    # The int8 activation and scale overlay slot 2 at zero cost.

    comptime ACT_I8_HIDDEN = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.HIDDEN, Self.tp]
    comptime ACT_I8_INTERMEDIATE = Slot[I8, Replicated, C.MAX_SEQ_LEN, C.INTERMEDIATE, Self.tp]
    comptime ACT_SCALE = Slot[F32, Replicated, C.MAX_SEQ_LEN, 1, Self.tp]

    # --- State layout ---

    comptime KV_STRIDE = Self.LAYER.cache_bytes()
    comptime KV_OFF = 0
    comptime X_MAIN_OFF = Self.KV_OFF + C.NUM_LAYERS * Self.KV_STRIDE
    comptime X_RESIDUAL_OFF = Self.X_MAIN_OFF + byte_count[Self.X_MAIN]()
    comptime SCRATCH_OFF = Self.X_RESIDUAL_OFF + byte_count[Self.X_RESIDUAL]()
    comptime SCRATCH_STRIDE = byte_count[Self.SCRATCH]()
    comptime ROPE_COS_OFF = Self.SCRATCH_OFF + Self.SCRATCH_COUNT * Self.SCRATCH_STRIDE
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

    # --- KV cache (int8 pre-rotated + per-head f32 scales) ---

    def kv_base(self, layer: Int) -> Int:
        return self.state_base() + Self.M.KV_OFF + layer * Self.M.KV_STRIDE

    def k_cache(self, layer: Int) -> CacheView[Self.L.K_CACHE]:
        return CacheView[Self.L.K_CACHE](self.kv_base(layer))

    def v_cache(self, layer: Int) -> CacheView[Self.L.V_CACHE]:
        return CacheView[Self.L.V_CACHE](
            self.kv_base(layer) + byte_count[Self.L.K_CACHE]()
        )

    def k_cache_scale(self, layer: Int) -> CacheView[Self.L.K_CACHE_SCALE]:
        return CacheView[Self.L.K_CACHE_SCALE](
            self.kv_base(layer) + byte_count[Self.L.K_CACHE]() + byte_count[Self.L.V_CACHE]()
        )

    def v_cache_scale(self, layer: Int) -> CacheView[Self.L.V_CACHE_SCALE]:
        return CacheView[Self.L.V_CACHE_SCALE](
            self.kv_base(layer) + byte_count[Self.L.K_CACHE]() + byte_count[Self.L.V_CACHE]()
            + byte_count[Self.L.K_CACHE_SCALE]()
        )

    # --- bf16 activation views ---

    def x_main(self, seq_len: Int) -> DynView[Self.M.X_MAIN]:
        return DynView[Self.M.X_MAIN](self.state_base() + Self.M.X_MAIN_OFF, seq_len)

    def x_residual(self, seq_len: Int) -> DynView[Self.M.X_RESIDUAL]:
        return DynView[Self.M.X_RESIDUAL](self.state_base() + Self.M.X_RESIDUAL_OFF, seq_len)

    def scratch_slot(self, index: Int) -> Int:
        return self.state_base() + Self.M.SCRATCH_OFF + index * Self.M.SCRATCH_STRIDE

    def q_view(self, seq_len: Int) -> DynView[Self.M.Q_VIEW]:
        """Q projection output (slot 0)."""
        return DynView[Self.M.Q_VIEW](self.scratch_slot(0), seq_len)

    def mlp_view(self, index: Int, seq_len: Int) -> DynView[Self.M.MLP_VIEW]:
        """Gate (index=0, slot 0) or Up (index=1, slot 1) output."""
        return DynView[Self.M.MLP_VIEW](self.scratch_slot(index), seq_len)

    # --- Quantized activation + scale (overlaid on slot 2) ---
    #     Slot 2 is fully available — K and V go directly to int8 cache.

    def activation_hidden(self, seq_len: Int) -> DynView[Self.M.ACT_I8_HIDDEN]:
        """Quantized activation [M, HIDDEN] for QKV / O projections."""
        return DynView[Self.M.ACT_I8_HIDDEN](self.scratch_slot(2), seq_len)

    def activation_intermediate(self, seq_len: Int) -> DynView[Self.M.ACT_I8_INTERMEDIATE]:
        """Quantized activation [M, INTERMEDIATE] for DOWN projection."""
        return DynView[Self.M.ACT_I8_INTERMEDIATE](self.scratch_slot(2), seq_len)

    def activation_scale(self, seq_len: Int) -> DynView[Self.M.ACT_SCALE]:
        """Per-row f32 scale for the quantized activation."""
        return DynView[Self.M.ACT_SCALE](self.scratch_slot(2) + byte_count[Self.M.ACT_I8_INTERMEDIATE](), seq_len)

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


def compute_column_sum[W: Encoding & Shaped & Placed, Cs: Encoding & Shaped & Placed](
    arena_base: Int, layer_base: Int,
):
    """Sum each row of a packed i8 weight → f32 column sum buffer.

    colsum[n] = float(sum_k weight[n, k])

    Reads from the VNNI-packed weight at W.OFFSET, writes to the
    column sum buffer at Cs.OFFSET. Both relative to layer_base.
    """
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + W.OFFSET,
    )
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + Cs.OFFSET,
    )
    for n in range(W.ROWS):
        var acc = Int(0)
        for k in range(W.COLS):
            acc += Int(wp[n * W.COLS + k])
        cp[n] = Float32(acc)


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


@fieldwise_init
struct LogitsView[vocab: Int, dtype: DType = DType.bfloat16]:
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


# =============================================================================
# Loaded model + forward pass
# =============================================================================


struct SmolLM2HadQuant[tp: Int](Movable):
    comptime M = HadQuantTPModel[Self.tp]

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

            # 1. rms_fwht_quantize(x) → act_i8, act_sc
            @parameter
            def do_rms_fwht_qkv[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rms_fwht_quantize[FWHT_BLOCK](
                    rv.x_main(seq_len),
                    rv.activation_hidden(seq_len),
                    rv.activation_scale(seq_len),
                    pool,
                    Float32(C.RMS_NORM_EPS),
                )
            ranks.parallel[do_rms_fwht_qkv]()

            # 2. Q gemm → bf16 (only Q materializes as bf16)
            @parameter
            def do_q[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.Q_PROJ, L.Q_SCALE, L.Q_COLSUM, M.ACT_I8_HIDDEN, M.ACT_SCALE, M.Q_VIEW](
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.Q_PROJ](layer_idx), rv.layer_weight[L.Q_SCALE](layer_idx), rv.layer_weight[L.Q_COLSUM](layer_idx),
                    rv.q_view(seq_len), pool,
                )
            ranks.parallel[do_q]()

            # 3. K gemm → RoPE → FWHT → int8 cache (K never materializes as bf16)
            @parameter
            def do_k_to_cache[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_k_to_cache[FWHT_BLOCK, C.HEAD_DIM, M.LOCAL_KV_HEADS](
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.K_PROJ](layer_idx), rv.layer_weight[L.K_SCALE](layer_idx), rv.layer_weight[L.K_COLSUM](layer_idx),
                    rv.k_cache(layer_idx), rv.k_cache_scale(layer_idx),
                    rv.rope_cos(), rv.rope_sin(),
                    pos, pool,
                )
            ranks.parallel[do_k_to_cache]()

            # 4. V gemm → FWHT → int8 cache (V never materializes as bf16)
            @parameter
            def do_v_to_cache[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_v_to_cache[FWHT_BLOCK, M.LOCAL_KV_HEADS](
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.V_PROJ](layer_idx), rv.layer_weight[L.V_SCALE](layer_idx), rv.layer_weight[L.V_COLSUM](layer_idx),
                    rv.v_cache(layer_idx), rv.v_cache_scale(layer_idx),
                    pos, pool,
                )
            ranks.parallel[do_v_to_cache]()

            # 5. Int8 GQA attention (RoPE on Q fused inside) → int8 output
            @parameter
            def do_attn[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gqa_attention[M.LOCAL_HEADS, M.LOCAL_KV_HEADS, C.HEAD_DIM](
                    rv.q_view(seq_len),
                    rv.k_cache(layer_idx), rv.k_cache_scale(layer_idx),
                    rv.v_cache(layer_idx), rv.v_cache_scale(layer_idx),
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.rope_cos(), rv.rope_sin(),
                    pos, pool,
                )
            ranks.parallel[do_attn]()

            # 6. O projection
            @parameter
            def do_o[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.O_PROJ, L.O_SCALE, L.O_COLSUM, M.ACT_I8_HIDDEN, M.ACT_SCALE, M.X_RESIDUAL](
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.O_PROJ](layer_idx), rv.layer_weight[L.O_SCALE](layer_idx), rv.layer_weight[L.O_COLSUM](layer_idx),
                    rv.x_residual(seq_len), pool,
                )
            ranks.parallel[do_o]()

            # 7. Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)

            @parameter
            def do_res_add(rv: RankView[Self.tp]):
                elem_add(rv.x_main(seq_len), rv.x_residual(seq_len), rv.x_main(seq_len))
            ranks.each[do_res_add]()

            # === MLP block ===

            # 8. rms_fwht_quantize(x) → act_i8, act_sc
            @parameter
            def do_rms_fwht_mlp[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return rms_fwht_quantize[FWHT_BLOCK](
                    rv.x_main(seq_len),
                    rv.activation_hidden(seq_len),
                    rv.activation_scale(seq_len),
                    pool,
                    Float32(C.RMS_NORM_EPS),
                )
            ranks.parallel[do_rms_fwht_mlp]()

            # 9. Fused GATE+UP gemm (one activation read, two outputs)
            @parameter
            def do_gate_up[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm_gate_up(
                    rv.activation_hidden(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.GATE_PROJ](layer_idx), rv.layer_weight[L.GATE_SCALE](layer_idx), rv.layer_weight[L.GATE_COLSUM](layer_idx),
                    rv.layer_weight[L.UP_PROJ](layer_idx), rv.layer_weight[L.UP_SCALE](layer_idx), rv.layer_weight[L.UP_COLSUM](layer_idx),
                    rv.mlp_view(0, seq_len), rv.mlp_view(1, seq_len),
                    pool,
                )
            ranks.parallel[do_gate_up]()

            # 10. silu_fwht_quantize(gate, up) → act_i8, act_sc
            @parameter
            def do_silu_fwht[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return silu_fwht_quantize[FWHT_BLOCK](
                    rv.mlp_view(0, seq_len),
                    rv.mlp_view(1, seq_len),
                    rv.activation_intermediate(seq_len),
                    rv.activation_scale(seq_len),
                    pool,
                )
            ranks.parallel[do_silu_fwht]()

            # 11. DOWN gemm
            @parameter
            def do_down[rank: Int](rv: RankView[Self.tp], mut pool: BurstPool[]) -> PoolFence:
                return int8_gemm[L.DOWN_PROJ, L.DOWN_SCALE, L.DOWN_COLSUM, M.ACT_I8_INTERMEDIATE, M.ACT_SCALE, M.X_RESIDUAL](
                    rv.activation_intermediate(seq_len), rv.activation_scale(seq_len),
                    rv.layer_weight[L.DOWN_PROJ](layer_idx), rv.layer_weight[L.DOWN_SCALE](layer_idx), rv.layer_weight[L.DOWN_COLSUM](layer_idx),
                    rv.x_residual(seq_len), pool,
                )
            ranks.parallel[do_down]()

            # 12. Allreduce + residual add
            ring_allreduce[M.X_RESIDUAL, Self.tp](ranks.x_residual_ptrs(seq_len), seq_len, ranks.pool_ptrs)
            ranks.each[do_res_add]()

            _ = layer_idx

        # --- Final norm (no gamma — model.norm.weight is passthrough but not absorbed) ---
        rms_norm_no_gamma(host.x_main(seq_len), host.x_main(seq_len), ranks.pool_ptrs[0][]).join()

        # --- LM head (bf16 gemm against embedding table — tied weights) ---
        # Embedding is bf16 passthrough, not int8. Use bf16 gemm.
        from kernels.kernel_ops import gemm
        var logits = DynView[M.LOGITS](host.scratch_slot(0), seq_len)
        gemm(host.x_main(seq_len), host.weight[M.EMBED](), logits, ranks.pool_ptrs[0][]).join()

        prof.finish()
        prof.report()
        return LogitsView[C.VOCAB_SIZE](logits.ptr, seq_len)
