"""ButterQuant vs bf16 kernel validation for SmolLM2.

Runs both TP=1 and TP=3 against the bf16 control path.

Coverage:
  - coarse layer-boundary drift across all 30 layers
  - detailed layer-0 prefill breakdown for:
      embed
      attn_norm
      q/k/v gemv outputs
      kv_write (decoded K/V cache contents)
      attention output
      o_proj partial
      attention block residual output
      mlp_norm
      fused gate/up/silu output
      down_proj partial
      mlp block residual output
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.pathlib import Path

from tokenizer import load_tokenizer
from modeling.smollm2_tp import SmolLM2TP, SmolLM2Config
from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant, FWHT_BLOCK
from modeling.model_spec import BF16, CacheView, DynView
from kernels.kernel_ops import rmsnorm, gemm, kv_cache_write, attention, silu_mul
from kernels.kv_rotors import rope
from experimental2.kv_cache import KVCache
from experimental2.kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize, fwht_block
from experimental2.kernels.int8_gemv import int8_gemv, fused_gu_silu
from experimental2.kernels.rope_and_kv_cache_write import rope_and_kv_cache_write
from experimental2.attn_amx_prefill import prefill as amx_prefill
from experimental.amx import init_intel_amx, TILE_N, TILE_BYTES, K_STEP


comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime BF16_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"
comptime PROMPT = "The quick brown fox jumps over the lazy dog. " * 5


@fieldwise_init
struct Metrics(Copyable):
    var cosine: Float64
    var rel_l2: Float64
    var rmse: Float64
    var max_abs: Float64


def print_metrics(label: String, m: Metrics):
    print(label,
        " cos=", m.cosine,
        " rel_l2=", m.rel_l2,
        " rmse=", m.rmse,
        " max_abs=", m.max_abs)


def metrics_bf16_vs_bf16(
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
) -> Metrics:
    comptime width = simd_width_of[DType.float32]()
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    var dsq = Float64(0)
    var max_abs = Float64(0)
    var k = 0
    while k + width <= count:
        var av = (a + k).load[width=width]().cast[DType.float32]()
        var bv = (b + k).load[width=width]().cast[DType.float32]()
        var dv = av - bv
        dot += (av * bv).cast[DType.float64]().reduce_add()
        na += (av * av).cast[DType.float64]().reduce_add()
        nb += (bv * bv).cast[DType.float64]().reduce_add()
        dsq += (dv * dv).cast[DType.float64]().reduce_add()
        var ad = dv.__abs__().cast[DType.float64]().reduce_max()
        if ad > max_abs:
            max_abs = ad
        k += width
    while k < count:
        var av = Float64((a + k)[].cast[DType.float32]())
        var bv = Float64((b + k)[].cast[DType.float32]())
        var d = av - bv
        var ad = d if d >= 0 else -d
        dot += av * bv
        na += av * av
        nb += bv * bv
        dsq += d * d
        if ad > max_abs:
            max_abs = ad
        k += 1
    var denom = na.__pow__(0.5)
    return Metrics(
        cosine=dot / (na.__pow__(0.5) * nb.__pow__(0.5) + Float64(1e-30)),
        rel_l2=dsq.__pow__(0.5) / (denom + Float64(1e-30)),
        rmse=(dsq / Float64(count)).__pow__(0.5),
        max_abs=max_abs,
    )


def metrics_f32_vs_bf16(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
) -> Metrics:
    comptime width = simd_width_of[DType.float32]()
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    var dsq = Float64(0)
    var max_abs = Float64(0)
    var k = 0
    while k + width <= count:
        var av = (a + k).load[width=width]()
        var bv = (b + k).load[width=width]().cast[DType.float32]()
        var dv = av - bv
        dot += (av * bv).cast[DType.float64]().reduce_add()
        na += (av * av).cast[DType.float64]().reduce_add()
        nb += (bv * bv).cast[DType.float64]().reduce_add()
        dsq += (dv * dv).cast[DType.float64]().reduce_add()
        var ad = dv.__abs__().cast[DType.float64]().reduce_max()
        if ad > max_abs:
            max_abs = ad
        k += width
    while k < count:
        var av = Float64((a + k)[])
        var bv = Float64((b + k)[].cast[DType.float32]())
        var d = av - bv
        var ad = d if d >= 0 else -d
        dot += av * bv
        na += av * av
        nb += bv * bv
        dsq += d * d
        if ad > max_abs:
            max_abs = ad
        k += 1
    var denom = na.__pow__(0.5)
    return Metrics(
        cosine=dot / (na.__pow__(0.5) * nb.__pow__(0.5) + Float64(1e-30)),
        rel_l2=dsq.__pow__(0.5) / (denom + Float64(1e-30)),
        rmse=(dsq / Float64(count)).__pow__(0.5),
        max_abs=max_abs,
    )


def copy_bf16(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.bfloat16]()
    var k = 0
    while k + width <= count:
        (dst + k).store((src + k).load[width=width]())
        k += width
    while k < count:
        dst[k] = src[k]
        k += 1


def extract_bf16_segment(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
    src_cols: Int,
    col_off: Int,
    cols: Int,
):
    comptime width = simd_width_of[DType.bfloat16]()
    for m in range(rows):
        var src_row = src + m * src_cols + col_off
        var dst_row = dst + m * cols
        var k = 0
        while k + width <= cols:
            (dst_row + k).store((src_row + k).load[width=width]())
            k += width
        while k < cols:
            dst_row[k] = src_row[k]
            k += 1


def dequant_fwht_rows[cols: Int, block: Int](
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    rows: Int,
    row_stride: Int,
    scales: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime width = simd_width_of[DType.float32]()
    for m in range(rows):
        var dst_row = dst + m * cols
        var src_row = src + m * row_stride
        var sc = scales[m] / Float32(127)
        var vsc = SIMD[DType.float32, width](sc)
        var k = 0
        while k + width <= cols:
            var qi = (src_row + k).load[width=width]().cast[DType.float32]()
            (dst_row + k).store(qi * vsc)
            k += width
        while k < cols:
            dst_row[k] = Float32(src_row[k]) * sc
            k += 1
        for b in range(cols // block):
            fwht_block[block](dst_row + b * block)


def decode_k_row_from_cache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int](
    cache: KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    pos: Int,
    head: Int,
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
):
    comptime KVC = KVCache[max_seq, head_dim, num_kv_heads, num_q_heads]
    var tile_idx = pos // TILE_N
    var col = pos % TILE_N
    var head_base = cache.k_base + head * KVC.K_HEAD_STRIDE
    for ki in range(KVC.K_SLICES):
        var tile_base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=head_base + tile_idx * KVC.TILE_K_BYTES + ki * TILE_BYTES)
        for d in range(TILE_N):
            var cell = tile_base + (d * TILE_N + col) * 4
            for b in range(4):
                dst[ki * K_STEP + d * 4 + b] = Scalar[DType.int8](Int8(Int(cell[b]) - 128))


def scale_ptr_from_cache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int](
    cache: KVCache[max_seq, head_dim, num_kv_heads, num_q_heads],
    head: Int,
    pos: Int,
) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=cache.k_scale_base
            + (head * max_seq + pos) * size_of[Float32]())


def compare_hidden_state(
    label: String,
    ref_ptr: Int,
    quant_ptr: Int,
    count: Int,
):
    var m = metrics_bf16_vs_bf16(
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ref_ptr),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=quant_ptr),
        count)
    print_metrics(label, m)


def print_buffer_metrics(
    label: String,
    ref_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    got_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    print_metrics(label, metrics_bf16_vs_bf16(ref_ptr, got_ptr, count))


def validate_prefill_layer0[tp: Int](
    mut reference: SmolLM2TP[BF16, tp],
    mut quant: SmolLM2ButterQuant[tp],
    seq_len: Int,
):
    comptime RefM = SmolLM2TP[BF16, tp].M
    comptime QuantM = SmolLM2ButterQuant[tp].M
    comptime RefL = RefM.LAYER
    comptime QuantL = QuantM.LAYER
    comptime Q_ROWS = C.HIDDEN // tp
    comptime KV_ROWS = C.KV_HIDDEN // tp
    comptime QKV_N = QuantL.QKV_N
    comptime GATE_ROWS = C.INTERMEDIATE // tp
    comptime LOCAL_HEADS = C.NUM_HEADS // tp
    comptime LOCAL_KV_HEADS = C.NUM_KV_HEADS // tp
    comptime Q_COLS = Q_ROWS
    comptime O_K = Q_COLS
    comptime DOWN_K = GATE_ROWS
    comptime MAX_WORKERS = 64
    comptime WORK_F32 = C.HIDDEN * MAX_WORKERS
    comptime ACT_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
    comptime ATTN_SCRATCH_BYTES = C.MAX_SEQ_LEN * Q_COLS * (size_of[Float32]() + 1) + 1024 * 1024
    comptime MLP_WORK_BYTES = C.MAX_SEQ_LEN * C.HIDDEN + WORK_F32 * size_of[Float32]()
    comptime COUNT_HIDDEN = C.HIDDEN * C.MAX_SEQ_LEN

    var layer_idx = 0
    var ref_host = reference.rank(0)
    var quant_host = quant.rank(0)
    var ls = quant.layer_scales[layer_idx]

    print()
    print("== TP=" + String(tp) + " layer 0 prefill breakdown ==")

    compare_hidden_state(
        "embed",
        reference.debug_x_main_ptr(seq_len),
        quant.debug_x_main_ptr(seq_len),
        seq_len * C.HIDDEN)

    # -----------------------------------------------------------------
    # attn_norm
    # -----------------------------------------------------------------
    var ref_attn_norm = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
    rmsnorm(
        ref_host.x_main(seq_len),
        ref_host.layer_weight[RefL.INPUT_NORM](layer_idx),
        DynView[RefM.X_RESIDUAL](Int(ref_attn_norm), seq_len),
        reference.pools[0]).join()

    var quant_act_i8 = alloc[Scalar[DType.int8]](seq_len * C.HIDDEN)
    var quant_attn_norm_f32 = alloc[Float32](seq_len * C.HIDDEN)
    var quant_attn_work = alloc[UInt8](ACT_WORK_BYTES)
    var quant_attn_scales = alloc[Float32](seq_len)
    rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
        quant_host.x_main(seq_len).ptr, Int(quant_act_i8), Int(quant_attn_work),
        Int(quant_attn_scales), seq_len, quant.pools[0]).join()
    dequant_fwht_rows[C.HIDDEN, FWHT_BLOCK](
        quant_attn_norm_f32, quant_act_i8, seq_len, C.HIDDEN, quant_attn_scales)
    print_metrics("attn_norm", metrics_f32_vs_bf16(
        quant_attn_norm_f32, ref_attn_norm, seq_len * C.HIDDEN))

    # -----------------------------------------------------------------
    # qkv_gemv + kv_write + attention + o_proj partial
    # -----------------------------------------------------------------
    for rank in range(tp):
        var ref_rv = reference.rank(rank)
        var quant_rv = quant.rank(rank)
        var q_ref = alloc[Scalar[DType.bfloat16]](seq_len * Q_ROWS)
        var k_ref = alloc[Scalar[DType.bfloat16]](seq_len * KV_ROWS)
        var v_ref = alloc[Scalar[DType.bfloat16]](seq_len * KV_ROWS)
        gemm(
            DynView[RefM.X_RESIDUAL](Int(ref_attn_norm), seq_len),
            ref_rv.layer_weight[RefL.Q_PROJ](layer_idx),
            DynView[RefM.Q_VIEW](Int(q_ref), seq_len),
            reference.pools[rank]).join()
        gemm(
            DynView[RefM.X_RESIDUAL](Int(ref_attn_norm), seq_len),
            ref_rv.layer_weight[RefL.K_PROJ](layer_idx),
            DynView[RefM.KV_VIEW](Int(k_ref), seq_len),
            reference.pools[rank]).join()
        gemm(
            DynView[RefM.X_RESIDUAL](Int(ref_attn_norm), seq_len),
            ref_rv.layer_weight[RefL.V_PROJ](layer_idx),
            DynView[RefM.KV_VIEW](Int(v_ref), seq_len),
            reference.pools[rank]).join()

        var qkv_quant = alloc[Scalar[DType.bfloat16]](seq_len * QKV_N)
        int8_gemv[QKV_N, C.HIDDEN](
            Int(quant_act_i8),
            quant_rv.layer_weight[QuantL.Q_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.Q_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.Q_ROW_SCALE](layer_idx).ptr,
            Int(qkv_quant), seq_len,
            Int(quant_attn_scales),
            quant.pools[rank]).join()

        var q_quant = alloc[Scalar[DType.bfloat16]](seq_len * Q_ROWS)
        var k_quant = alloc[Scalar[DType.bfloat16]](seq_len * KV_ROWS)
        var v_quant = alloc[Scalar[DType.bfloat16]](seq_len * KV_ROWS)
        extract_bf16_segment(q_quant, qkv_quant, seq_len, QKV_N, 0, Q_ROWS)
        extract_bf16_segment(k_quant, qkv_quant, seq_len, QKV_N, Q_ROWS, KV_ROWS)
        extract_bf16_segment(v_quant, qkv_quant, seq_len, QKV_N, Q_ROWS + KV_ROWS, KV_ROWS)

        print_metrics("qkv.q tp=" + String(tp) + " rank=" + String(rank),
            metrics_bf16_vs_bf16(q_quant, q_ref, seq_len * Q_ROWS))
        print_metrics("qkv.k tp=" + String(tp) + " rank=" + String(rank),
            metrics_bf16_vs_bf16(k_quant, k_ref, seq_len * KV_ROWS))
        print_metrics("qkv.v tp=" + String(tp) + " rank=" + String(rank),
            metrics_bf16_vs_bf16(v_quant, v_ref, seq_len * KV_ROWS))

        var k_ref_rope = alloc[Scalar[DType.bfloat16]](seq_len * KV_ROWS)
        copy_bf16(k_ref_rope, k_ref, seq_len * KV_ROWS)
        rope[C.HEAD_DIM, LOCAL_KV_HEADS](
            DynView[RefM.KV_VIEW](Int(k_ref_rope), seq_len),
            ref_rv.rope_cos(), ref_rv.rope_sin(), 0)

        var ref_k_cache = alloc[Scalar[DType.bfloat16]](C.MAX_SEQ_LEN * KV_ROWS)
        var ref_v_cache = alloc[Scalar[DType.bfloat16]](C.MAX_SEQ_LEN * KV_ROWS)
        kv_cache_write(
            DynView[RefM.KV_VIEW](Int(k_ref_rope), seq_len),
            CacheView[RefL.K_CACHE](Int(ref_k_cache)), 0)
        kv_cache_write(
            DynView[RefM.KV_VIEW](Int(v_ref), seq_len),
            CacheView[RefL.V_CACHE](Int(ref_v_cache)), 0)

        var quant_cache_mem = alloc[UInt8](QuantL.KVC.TOTAL_BYTES)
        var quant_cache = QuantL.KVC(Int(quant_cache_mem))
        rope_and_kv_cache_write[C.HEAD_DIM, LOCAL_KV_HEADS, C.MAX_SEQ_LEN, LOCAL_HEADS](
            k_quant, v_quant,
            QKV_N, QKV_N,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=quant_rv.rope_cos().ptr),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=quant_rv.rope_sin().ptr),
            quant_cache, 0, seq_len, ls.v_layer_scale)

        var k_dec_i8 = alloc[Scalar[DType.int8]](C.HEAD_DIM)
        var k_dec_f32 = alloc[Float32](seq_len * KV_ROWS)
        var v_dec_f32 = alloc[Float32](seq_len * KV_ROWS)
        var v_scales = alloc[Float32](seq_len)
        for m in range(seq_len):
            v_scales[m] = ls.v_layer_scale
        for head in range(LOCAL_KV_HEADS):
            for m in range(seq_len):
                decode_k_row_from_cache[C.MAX_SEQ_LEN, C.HEAD_DIM, LOCAL_KV_HEADS, LOCAL_HEADS](
                    quant_cache, m, head, k_dec_i8)
                dequant_fwht_rows[C.HEAD_DIM, C.HEAD_DIM](
                    k_dec_f32 + (m * LOCAL_KV_HEADS + head) * C.HEAD_DIM,
                    k_dec_i8, 1, C.HEAD_DIM,
                    scale_ptr_from_cache[C.MAX_SEQ_LEN, C.HEAD_DIM, LOCAL_KV_HEADS, LOCAL_HEADS](
                        quant_cache, head, m))
                dequant_fwht_rows[C.HEAD_DIM, C.HEAD_DIM](
                    v_dec_f32 + (m * LOCAL_KV_HEADS + head) * C.HEAD_DIM,
                    quant_cache.v_head(m, head).bitcast[Scalar[DType.int8]](), 1, C.HEAD_DIM,
                    v_scales + m)
            _ = head
        print_metrics("kv_write.k tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(k_dec_f32, k_ref_rope, seq_len * KV_ROWS))
        print_metrics("kv_write.v tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(v_dec_f32, v_ref, seq_len * KV_ROWS))

        var q_ref_rope = alloc[Scalar[DType.bfloat16]](seq_len * Q_ROWS)
        copy_bf16(q_ref_rope, q_ref, seq_len * Q_ROWS)
        rope[C.HEAD_DIM, LOCAL_HEADS](
            DynView[RefM.Q_VIEW](Int(q_ref_rope), seq_len),
            ref_rv.rope_cos(), ref_rv.rope_sin(), 0)
        var attn_ref = alloc[Scalar[DType.bfloat16]](seq_len * Q_ROWS)
        attention[LOCAL_HEADS, LOCAL_KV_HEADS, C.HEAD_DIM](
            DynView[RefM.Q_VIEW](Int(q_ref_rope), seq_len),
            CacheView[RefL.K_CACHE](Int(ref_k_cache)),
            CacheView[RefL.V_CACHE](Int(ref_v_cache)),
            DynView[RefM.Q_VIEW](Int(attn_ref), seq_len),
            0, reference.pools[rank]).join()

        var attn_scratch = alloc[UInt8](ATTN_SCRATCH_BYTES)
        amx_prefill[LOCAL_HEADS, LOCAL_KV_HEADS, C.HEAD_DIM, C.MAX_SEQ_LEN, LOCAL_HEADS](
            DynView[QuantM.Q_VIEW](Int(qkv_quant), seq_len),
            QKV_N, quant_cache,
            quant_rv.rope_cos(), quant_rv.rope_sin(),
            Int(attn_scratch), 0, ls.v_layer_scale,
            quant.pools[rank]).join()
        var attn_quant_f32 = alloc[Float32](seq_len * Q_ROWS)
        var attn_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(attn_scratch) + seq_len * Q_ROWS * size_of[Float32]())
        dequant_fwht_rows[Q_ROWS, C.HEAD_DIM](
            attn_quant_f32, attn_i8, seq_len, Q_ROWS, v_scales)
        print_metrics("attention tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(attn_quant_f32, attn_ref, seq_len * Q_ROWS))

        var o_ref = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
        gemm(
            DynView[RefM.Q_VIEW](Int(attn_ref), seq_len),
            ref_rv.layer_weight[RefL.O_PROJ](layer_idx),
            DynView[RefM.X_RESIDUAL](Int(o_ref), seq_len),
            reference.pools[rank]).join()

        var o_quant = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
        int8_gemv[C.HIDDEN, O_K](
            Int(attn_i8),
            quant_rv.layer_weight[QuantL.O_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.O_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.O_ROW_SCALE](layer_idx).ptr,
            Int(o_quant), seq_len,
            Int(v_scales),
            quant.pools[rank]).join()
        print_metrics("o_proj tp=" + String(tp) + " rank=" + String(rank),
            metrics_bf16_vs_bf16(o_quant, o_ref, seq_len * C.HIDDEN))
    reference.debug_layer_attn(layer_idx, seq_len, 0)
    quant.debug_layer_attn(layer_idx, seq_len, 0)
    compare_hidden_state(
        "attn_block",
        reference.debug_x_main_ptr(seq_len),
        quant.debug_x_main_ptr(seq_len),
        seq_len * C.HIDDEN)
    var ref_attn_out = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
    var quant_attn_out = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
    copy_bf16(
        ref_attn_out,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=reference.debug_x_main_ptr(seq_len)),
        seq_len * C.HIDDEN)
    copy_bf16(
        quant_attn_out,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=quant.debug_x_main_ptr(seq_len)),
        seq_len * C.HIDDEN)

    # -----------------------------------------------------------------
    # mlp_norm + fused_gu_silu + down_proj
    # -----------------------------------------------------------------
    var ref_mlp_norm = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
    rmsnorm(
        ref_host.x_main(seq_len),
        ref_host.layer_weight[RefL.POST_ATTN_NORM](layer_idx),
        DynView[RefM.X_RESIDUAL](Int(ref_mlp_norm), seq_len),
        reference.pools[0]).join()

    var quant_mlp_i8 = alloc[Scalar[DType.int8]](seq_len * C.HIDDEN)
    var quant_mlp_norm_f32 = alloc[Float32](seq_len * C.HIDDEN)
    var quant_mlp_work = alloc[UInt8](MLP_WORK_BYTES)
    var quant_post_scales = alloc[Float32](seq_len)
    rmsnorm_fwht_quantize[C.HIDDEN, FWHT_BLOCK](
        quant_host.x_main(seq_len).ptr, Int(quant_mlp_i8), Int(quant_mlp_work),
        Int(quant_attn_scales), seq_len, quant.pools[0]).join()
    dequant_fwht_rows[C.HIDDEN, FWHT_BLOCK](
        quant_mlp_norm_f32, quant_mlp_i8, seq_len, C.HIDDEN, quant_attn_scales)
    print_metrics("mlp_norm", metrics_f32_vs_bf16(
        quant_mlp_norm_f32, ref_mlp_norm, seq_len * C.HIDDEN))

    for rank in range(tp):
        var ref_rv = reference.rank(rank)
        var quant_rv = quant.rank(rank)
        var gate_ref = alloc[Scalar[DType.bfloat16]](seq_len * GATE_ROWS)
        var up_ref = alloc[Scalar[DType.bfloat16]](seq_len * GATE_ROWS)
        gemm(
            DynView[RefM.X_RESIDUAL](Int(ref_mlp_norm), seq_len),
            ref_rv.layer_weight[RefL.GATE_PROJ](layer_idx),
            DynView[RefM.MLP_VIEW](Int(gate_ref), seq_len),
            reference.pools[rank]).join()
        gemm(
            DynView[RefM.X_RESIDUAL](Int(ref_mlp_norm), seq_len),
            ref_rv.layer_weight[RefL.UP_PROJ](layer_idx),
            DynView[RefM.MLP_VIEW](Int(up_ref), seq_len),
            reference.pools[rank]).join()
        var silu_ref = alloc[Scalar[DType.bfloat16]](seq_len * GATE_ROWS)
        silu_mul(
            DynView[RefM.MLP_VIEW](Int(gate_ref), seq_len),
            DynView[RefM.MLP_VIEW](Int(up_ref), seq_len),
            DynView[RefM.MLP_VIEW](Int(silu_ref), seq_len))

        var post_i8 = alloc[Scalar[DType.int8]](seq_len * GATE_ROWS)
        fused_gu_silu[GATE_ROWS, C.HIDDEN, FWHT_BLOCK](
            Int(quant_mlp_i8),
            quant_rv.layer_weight[QuantL.GATE_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.GATE_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.GATE_ROW_SCALE](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.UP_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.UP_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.UP_ROW_SCALE](layer_idx).ptr,
            Int(post_i8), Int(quant_post_scales),
            seq_len, Int(quant_attn_scales),
            quant.pools[rank]).join()
        var silu_quant_f32 = alloc[Float32](seq_len * GATE_ROWS)
        dequant_fwht_rows[GATE_ROWS, FWHT_BLOCK](
            silu_quant_f32, post_i8, seq_len, GATE_ROWS, quant_post_scales)
        print_metrics("fused_gu_silu tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(silu_quant_f32, silu_ref, seq_len * GATE_ROWS))

        var down_ref = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
        gemm(
            DynView[RefM.MLP_VIEW](Int(silu_ref), seq_len),
            ref_rv.layer_weight[RefL.DOWN_PROJ](layer_idx),
            DynView[RefM.X_RESIDUAL](Int(down_ref), seq_len),
            reference.pools[rank]).join()

        var down_quant = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
        int8_gemv[C.HIDDEN, DOWN_K](
            Int(post_i8),
            quant_rv.layer_weight[QuantL.DOWN_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.DOWN_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.DOWN_ROW_SCALE](layer_idx).ptr,
            Int(down_quant), seq_len,
            Int(quant_post_scales),
            quant.pools[rank]).join()
        print_metrics("down_proj tp=" + String(tp) + " rank=" + String(rank),
            metrics_bf16_vs_bf16(down_quant, down_ref, seq_len * C.HIDDEN))
    reference.debug_layer_mlp(layer_idx, seq_len, 0)
    quant.debug_layer_mlp(layer_idx, seq_len, 0)
    compare_hidden_state(
        "mlp_block",
        reference.debug_x_main_ptr(seq_len),
        quant.debug_x_main_ptr(seq_len),
        seq_len * C.HIDDEN)
    var ref_layer0_out = alloc[Scalar[DType.bfloat16]](seq_len * C.HIDDEN)
    copy_bf16(
        ref_layer0_out,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=reference.debug_x_main_ptr(seq_len)),
        seq_len * C.HIDDEN)

    # Attribution at the layer boundary:
    # 1. Quantized attention output passed through bf16 MLP.
    reference.debug_set_x_main(Int(quant_attn_out), seq_len)
    reference.debug_layer_mlp(layer_idx, seq_len, 0)
    print_buffer_metrics(
        "attn_only_to_layer_out",
        ref_layer0_out,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=reference.debug_x_main_ptr(seq_len)),
        seq_len * C.HIDDEN)

    # 2. Reference attention output passed through quantized MLP.
    quant.debug_set_x_main(Int(ref_attn_out), seq_len)
    quant.debug_layer_mlp(layer_idx, seq_len, 0)
    print_buffer_metrics(
        "mlp_only_to_layer_out",
        ref_layer0_out,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=quant.debug_x_main_ptr(seq_len)),
        seq_len * C.HIDDEN)

    for layer in range(1, C.NUM_LAYERS):
        reference.debug_layer_attn(layer, seq_len, 0)
        quant.debug_layer_attn(layer, seq_len, 0)
        compare_hidden_state(
            "layer " + String(layer) + " attn",
            reference.debug_x_main_ptr(seq_len),
            quant.debug_x_main_ptr(seq_len),
            seq_len * C.HIDDEN)

        reference.debug_layer_mlp(layer, seq_len, 0)
        quant.debug_layer_mlp(layer, seq_len, 0)
        compare_hidden_state(
            "layer " + String(layer) + " mlp",
            reference.debug_x_main_ptr(seq_len),
            quant.debug_x_main_ptr(seq_len),
            seq_len * C.HIDDEN)


def run_validation[tp: Int](token_ids: List[Int]):
    print()
    print("============================================================")
    print("TP=" + String(tp))
    print("============================================================")

    var reference_opt = SmolLM2TP[BF16, tp].load(Path(BF16_PATH))
    if not reference_opt:
        print("bf16 model load failed for TP=" + String(tp))
        return
    var reference = reference_opt.take()

    var quant_opt = SmolLM2ButterQuant[tp].load(Path(QUANT_PATH))
    if not quant_opt:
        print("quant model load failed for TP=" + String(tp))
        return
    var quant = quant_opt.take()

    var seq_len = len(token_ids)
    var ref_tp = reference.token_buffer()
    var quant_tp = quant.token_buffer()
    for i in range(seq_len):
        ref_tp[i] = Scalar[DType.int32](token_ids[i])
        quant_tp[i] = Scalar[DType.int32](token_ids[i])

    reference.debug_embed(Int(ref_tp), seq_len)
    quant.debug_embed(Int(quant_tp), seq_len)
    validate_prefill_layer0[tp](reference, quant, seq_len)


def main():
    if not init_intel_amx():
        print("SKIP: no AMX")
        return

    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("tokenizer load failed")
        return
    var tok = tok_opt.take()
    var token_ids = tok.encode(PROMPT)
    print("prompt tokens:", len(token_ids))

    run_validation[1](token_ids)
    run_validation[3](token_ids)
