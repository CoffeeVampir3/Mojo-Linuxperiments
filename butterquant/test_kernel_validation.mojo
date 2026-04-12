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

from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from linux.io_uring import IoRing, ReadOp, ReadMode
from tokenizer import load_tokenizer
from modeling.smollm2_tp import SmolLM2TP, SmolLM2Config
from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant, FWHT_BLOCK
from modeling.model_spec import BF16, CacheView, DynView
from kernels.kernel_ops import rmsnorm, gemm, kv_cache_write, attention, silu_mul
from kernels.kv_rotors import rope
from experimental2.kv_cache import KVCache
from experimental2.kernels.rmsnorm_fwht_quantize import rmsnorm_fwht_quantize, fwht_block
from experimental2.kernels.int8_gemv import int8_gemv, fused_gu_silu, gemv_row
from experimental2.kernels.rope_and_kv_cache_write import rope_and_kv_cache_write
from experimental2.attn_amx_prefill import prefill as amx_prefill
from experimental.amx import init_intel_amx, TILE_N, TILE_BYTES, K_STEP
from kernels.vnni import compute_n_block, VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK
from simd_math import roundeven, exp_f32


comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime BF16_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"
comptime PROMPT = "The quick brown fox jumps over the lazy dog. " * 5


struct FileReader(Movable):
    var ring: IoRing[]
    var data_offset: Int

    def __init__(out self, data_offset: Int):
        self.ring = IoRing[]()
        self.data_offset = data_offset

    def read(mut self, offset: Int, dest: UnsafePointer[UInt8, MutAnyOrigin], length: Int) -> Bool:
        try:
            _ = self.ring.submit_one(ReadOp(
                file_idx=0, offset=self.data_offset + offset,
                length=length, dest=dest, id=0,
            ))
            var completions = self.ring.wait()
            return len(completions) > 0 and Int(completions[0].result) == length
        except:
            return False


@fieldwise_init
struct Metrics(Copyable):
    var cosine: Float64
    var rel_l2: Float64
    var rmse: Float64
    var max_abs: Float64


@fieldwise_init
struct QuantDiag(Copyable):
    var rel_l2: Float64
    var sat_frac: Float64
    var zero_frac: Float64
    var mean_absmax_over_rms: Float64
    var max_absmax_over_rms: Float64


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


def metrics_f32_vs_f32(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
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
        var bv = (b + k).load[width=width]()
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
        var bv = Float64((b + k)[])
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


def copy_f32(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= count:
        (dst + k).store((src + k).load[width=width]())
        k += width
    while k < count:
        dst[k] = src[k]
        k += 1


def copy_bf16_to_f32(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= count:
        (dst + k).store((src + k).load[width=width]().cast[DType.float32]())
        k += width
    while k < count:
        dst[k] = src[k].cast[DType.float32]()
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


def print_quant_diag(label: String, diag: QuantDiag):
    print(
        label,
        " rel_l2=", diag.rel_l2,
        " sat_frac=", diag.sat_frac,
        " zero_frac=", diag.zero_frac,
        " mean_absmax_over_rms=", diag.mean_absmax_over_rms,
        " max_absmax_over_rms=", diag.max_absmax_over_rms)


def analyze_absmax_qdq_no_fwht[cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
) -> QuantDiag:
    comptime width = simd_width_of[DType.float32]()
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)

    var refsq = Float64(0)
    var dsq = Float64(0)
    var sat = Int(0)
    var zero = Int(0)
    var sum_absmax_over_rms = Float64(0)
    var max_absmax_over_rms = Float64(0)

    for r in range(rows):
        var row = src + r * cols
        var vmax = SIMD[DType.float32, width](0)
        var vsq = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            var v = (row + k).load[width=width]()
            vmax = max(vmax, v.__abs__())
            vsq = v.fma(v, vsq)
            k += width
        var absmax = vmax.reduce_max()
        var sumsq = Float64(vsq.reduce_add())
        while k < cols:
            var v = row[k]
            var a = v if v >= Float32(0) else -v
            if a > absmax:
                absmax = a
            sumsq += Float64(v * v)
            k += 1
        if absmax < Float32(1e-10):
            absmax = Float32(1e-10)
        var rms = Float64((sumsq / Float64(cols)).__pow__(0.5))
        var ratio = Float64(absmax) / (rms + Float64(1e-30))
        sum_absmax_over_rms += ratio
        if ratio > max_absmax_over_rms:
            max_absmax_over_rms = ratio

        var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
        var vdq = SIMD[DType.float32, width](absmax / Float32(127))
        k = 0
        while k + width <= cols:
            var v = (row + k).load[width=width]()
            var q = min(max(roundeven(v * vinv), lo), hi)
            var dq = q * vdq
            dsq += ((v - dq) * (v - dq)).cast[DType.float64]().reduce_add()
            refsq += (v * v).cast[DType.float64]().reduce_add()
            var qi = q.cast[DType.int32]()
            for lane in range(width):
                if qi[lane] == Int32(127) or qi[lane] == Int32(-127) or qi[lane] == Int32(-128):
                    sat += 1
                if qi[lane] == Int32(0):
                    zero += 1
            k += width
        while k < cols:
            var v = Float64(row[k])
            var q = Float64(roundeven(Float32(v * Float64(127) / Float64(absmax))))
            if q < Float64(-128):
                q = Float64(-128)
            if q > Float64(127):
                q = Float64(127)
            var dq = q * Float64(absmax) / Float64(127)
            var d = v - dq
            dsq += d * d
            refsq += v * v
            var qi = Int(q)
            if qi == 127 or qi == -127 or qi == -128:
                sat += 1
            if qi == 0:
                zero += 1
            k += 1

    var total = Float64(rows * cols)
    return QuantDiag(
        rel_l2=dsq.__pow__(0.5) / (refsq.__pow__(0.5) + Float64(1e-30)),
        sat_frac=Float64(sat) / total,
        zero_frac=Float64(zero) / total,
        mean_absmax_over_rms=sum_absmax_over_rms / Float64(rows),
        max_absmax_over_rms=max_absmax_over_rms,
    )


def analyze_absmax_qdq_fwht[cols: Int, block: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
) -> QuantDiag:
    comptime width = simd_width_of[DType.float32]()
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)

    var work = alloc[Float32](cols)
    var refsq = Float64(0)
    var dsq = Float64(0)
    var sat = Int(0)
    var zero = Int(0)
    var sum_absmax_over_rms = Float64(0)
    var max_absmax_over_rms = Float64(0)

    for r in range(rows):
        copy_f32(work, src + r * cols, cols)
        for b in range(cols // block):
            fwht_block[block](work + b * block)

        var vmax = SIMD[DType.float32, width](0)
        var vsq = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            var v = (work + k).load[width=width]()
            vmax = max(vmax, v.__abs__())
            vsq = v.fma(v, vsq)
            k += width
        var absmax = vmax.reduce_max()
        var sumsq = Float64(vsq.reduce_add())
        while k < cols:
            var v = work[k]
            var a = v if v >= Float32(0) else -v
            if a > absmax:
                absmax = a
            sumsq += Float64(v * v)
            k += 1
        if absmax < Float32(1e-10):
            absmax = Float32(1e-10)
        var rms = Float64((sumsq / Float64(cols)).__pow__(0.5))
        var ratio = Float64(absmax) / (rms + Float64(1e-30))
        sum_absmax_over_rms += ratio
        if ratio > max_absmax_over_rms:
            max_absmax_over_rms = ratio

        var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
        var vdq = SIMD[DType.float32, width](absmax / Float32(127))
        k = 0
        while k + width <= cols:
            var v = (work + k).load[width=width]()
            var q = min(max(roundeven(v * vinv), lo), hi)
            var dq = q * vdq
            dsq += ((v - dq) * (v - dq)).cast[DType.float64]().reduce_add()
            refsq += (v * v).cast[DType.float64]().reduce_add()
            var qi = q.cast[DType.int32]()
            for lane in range(width):
                if qi[lane] == Int32(127) or qi[lane] == Int32(-127) or qi[lane] == Int32(-128):
                    sat += 1
                if qi[lane] == Int32(0):
                    zero += 1
            k += width
        while k < cols:
            var v = Float64(work[k])
            var q = Float64(roundeven(Float32(v * Float64(127) / Float64(absmax))))
            if q < Float64(-128):
                q = Float64(-128)
            if q > Float64(127):
                q = Float64(127)
            var dq = q * Float64(absmax) / Float64(127)
            var d = v - dq
            dsq += d * d
            refsq += v * v
            var qi = Int(q)
            if qi == 127 or qi == -127 or qi == -128:
                sat += 1
            if qi == 0:
                zero += 1
            k += 1

    work.free()

    var total = Float64(rows * cols)
    return QuantDiag(
        rel_l2=dsq.__pow__(0.5) / (refsq.__pow__(0.5) + Float64(1e-30)),
        sat_frac=Float64(sat) / total,
        zero_frac=Float64(zero) / total,
        mean_absmax_over_rms=sum_absmax_over_rms / Float64(rows),
        max_absmax_over_rms=max_absmax_over_rms,
    )


def analyze_absmax_qdq_fwht_grouped[cols: Int, block: Int, group_cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
) -> QuantDiag:
    comptime width = simd_width_of[DType.float32]()
    comptime assert cols % group_cols == 0, "analyze_absmax_qdq_fwht_grouped: cols must divide group_cols"
    comptime assert group_cols % width == 0, "analyze_absmax_qdq_fwht_grouped: group_cols must be SIMD aligned"
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)

    var work = alloc[Float32](cols)
    var refsq = Float64(0)
    var dsq = Float64(0)
    var sat = Int(0)
    var zero = Int(0)
    var sum_absmax_over_rms = Float64(0)
    var max_absmax_over_rms = Float64(0)

    for r in range(rows):
        copy_f32(work, src + r * cols, cols)
        for b in range(cols // block):
            fwht_block[block](work + b * block)

        for g in range(cols // group_cols):
            var group = work + g * group_cols
            var vmax = SIMD[DType.float32, width](0)
            var vsq = SIMD[DType.float32, width](0)
            var k = 0
            while k + width <= group_cols:
                var v = (group + k).load[width=width]()
                vmax = max(vmax, v.__abs__())
                vsq = v.fma(v, vsq)
                k += width
            var absmax = vmax.reduce_max()
            var sumsq = Float64(vsq.reduce_add())
            while k < group_cols:
                var v = group[k]
                var a = v if v >= Float32(0) else -v
                if a > absmax:
                    absmax = a
                sumsq += Float64(v * v)
                k += 1
            if absmax < Float32(1e-10):
                absmax = Float32(1e-10)
            var rms = Float64((sumsq / Float64(group_cols)).__pow__(0.5))
            var ratio = Float64(absmax) / (rms + Float64(1e-30))
            sum_absmax_over_rms += ratio
            if ratio > max_absmax_over_rms:
                max_absmax_over_rms = ratio

            var vinv = SIMD[DType.float32, width](Float32(127) / absmax)
            var vdq = SIMD[DType.float32, width](absmax / Float32(127))
            k = 0
            while k + width <= group_cols:
                var v = (group + k).load[width=width]()
                var q = min(max(roundeven(v * vinv), lo), hi)
                var dq = q * vdq
                dsq += ((v - dq) * (v - dq)).cast[DType.float64]().reduce_add()
                refsq += (v * v).cast[DType.float64]().reduce_add()
                var qi = q.cast[DType.int32]()
                for lane in range(width):
                    if qi[lane] == Int32(127) or qi[lane] == Int32(-127) or qi[lane] == Int32(-128):
                        sat += 1
                    if qi[lane] == Int32(0):
                        zero += 1
                k += width
            while k < group_cols:
                var v = Float64(group[k])
                var q = Float64(roundeven(Float32(v * Float64(127) / Float64(absmax))))
                if q < Float64(-128):
                    q = Float64(-128)
                if q > Float64(127):
                    q = Float64(127)
                var dq = q * Float64(absmax) / Float64(127)
                var d = v - dq
                dsq += d * d
                refsq += v * v
                var qi = Int(q)
                if qi == 127 or qi == -127 or qi == -128:
                    sat += 1
                if qi == 0:
                    zero += 1
                k += 1

        for b in range(cols // block):
            fwht_block[block](work + b * block)

    work.free()

    var total = Float64(rows * cols)
    var num_groups = Float64(rows * (cols // group_cols))
    return QuantDiag(
        rel_l2=dsq.__pow__(0.5) / (refsq.__pow__(0.5) + Float64(1e-30)),
        sat_frac=Float64(sat) / total,
        zero_frac=Float64(zero) / total,
        mean_absmax_over_rms=sum_absmax_over_rms / num_groups,
        max_absmax_over_rms=max_absmax_over_rms,
    )


def fwht_quantize_dequant_fwht_group_absmax[cols: Int, block: Int, group_cols: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
):
    comptime width = simd_width_of[DType.float32]()
    comptime assert cols % group_cols == 0, "fwht_quantize_dequant_fwht_group_absmax: cols must divide group_cols"
    comptime assert group_cols % width == 0, "fwht_quantize_dequant_fwht_group_absmax: group_cols must be SIMD aligned"
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    for m in range(rows):
        var row = buf + m * cols
        for b in range(cols // block):
            fwht_block[block](row + b * block)

        for g in range(cols // group_cols):
            var group = row + g * group_cols
            var vmax = SIMD[DType.float32, width](0)
            var k = 0
            while k + width <= group_cols:
                vmax = max(vmax, (group + k).load[width=width]().__abs__())
                k += width
            var scale = vmax.reduce_max()
            while k < group_cols:
                var a = group[k]
                if a < Float32(0):
                    a = -a
                if a > scale:
                    scale = a
                k += 1
            if scale < Float32(1e-10):
                scale = Float32(1e-10)

            var vinv = SIMD[DType.float32, width](Float32(127) / scale)
            var vdq = SIMD[DType.float32, width](scale / Float32(127))
            k = 0
            while k + width <= group_cols:
                var v = (group + k).load[width=width]()
                (group + k).store(min(max(roundeven(v * vinv), lo), hi) * vdq)
                k += width

        for b in range(cols // block):
            fwht_block[block](row + b * block)


def fwht_quantize_dequant_fwht_absmax[cols: Int, block: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
):
    comptime width = simd_width_of[DType.float32]()
    comptime assert cols % width == 0, "fwht_quantize_dequant_fwht_absmax: cols must be SIMD aligned"
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    for m in range(rows):
        var row = buf + m * cols
        for b in range(cols // block):
            fwht_block[block](row + b * block)

        var vmax = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            vmax = max(vmax, (row + k).load[width=width]().__abs__())
            k += width
        var scale = vmax.reduce_max()
        while k < cols:
            var a = row[k]
            if a < Float32(0):
                a = -a
            if a > scale:
                scale = a
            k += 1
        if scale < Float32(1e-10):
            scale = Float32(1e-10)

        var vinv = SIMD[DType.float32, width](Float32(127) / scale)
        var vdq = SIMD[DType.float32, width](scale / Float32(127))
        k = 0
        while k + width <= cols:
            var v = (row + k).load[width=width]()
            (row + k).store(min(max(roundeven(v * vinv), lo), hi) * vdq)
            k += width

        for b in range(cols // block):
            fwht_block[block](row + b * block)


def fwht_quantize_dequant_fwht_clipped_rms[cols: Int, block: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    alpha: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    comptime assert cols % width == 0, "fwht_quantize_dequant_fwht_clipped_rms: cols must be SIMD aligned"
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    for m in range(rows):
        var row = buf + m * cols
        for b in range(cols // block):
            fwht_block[block](row + b * block)

        var vmax = SIMD[DType.float32, width](0)
        var vsq = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            var v = (row + k).load[width=width]()
            vmax = max(vmax, v.__abs__())
            vsq = v.fma(v, vsq)
            k += width
        var absmax = vmax.reduce_max()
        var sumsq = vsq.reduce_add()
        while k < cols:
            var v = row[k]
            var a = v if v >= Float32(0) else -v
            if a > absmax:
                absmax = a
            sumsq += v * v
            k += 1
        if absmax < Float32(1e-10):
            absmax = Float32(1e-10)
        var rms = Float32((Float64(sumsq) / Float64(cols)).__pow__(0.5))
        var scale = min(absmax, alpha * rms)
        if scale < Float32(1e-10):
            scale = Float32(1e-10)

        var vinv = SIMD[DType.float32, width](Float32(127) / scale)
        var vdq = SIMD[DType.float32, width](scale / Float32(127))
        k = 0
        while k + width <= cols:
            var v = (row + k).load[width=width]()
            (row + k).store(min(max(roundeven(v * vinv), lo), hi) * vdq)
            k += width

        for b in range(cols // block):
            fwht_block[block](row + b * block)


def sweep_post_quantizer[cols: Int, block: Int](
    label_prefix: String,
    ref_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
):
    var ref_f32 = alloc[Float32](rows * cols)
    var work = alloc[Float32](rows * cols)
    copy_bf16_to_f32(ref_f32, ref_bf16, rows * cols)

    copy_f32(work, ref_f32, rows * cols)
    fwht_quantize_dequant_fwht_absmax[cols, block](work, rows)
    print_metrics(label_prefix + " absmax",
        metrics_f32_vs_f32(work, ref_f32, rows * cols))

    var alphas = InlineArray[Float32, 6](
        fill=Float32(0))
    alphas[0] = Float32(2.0)
    alphas[1] = Float32(2.5)
    alphas[2] = Float32(3.0)
    alphas[3] = Float32(3.5)
    alphas[4] = Float32(4.0)
    alphas[5] = Float32(5.0)

    var best_alpha = Float32(0)
    var best = Metrics(cosine=Float64(-2), rel_l2=Float64(1e30), rmse=Float64(1e30), max_abs=Float64(1e30))
    for i in range(6):
        var alpha = alphas[i]
        copy_f32(work, ref_f32, rows * cols)
        fwht_quantize_dequant_fwht_clipped_rms[cols, block](work, rows, alpha)
        var m = metrics_f32_vs_f32(work, ref_f32, rows * cols)
        print_metrics(label_prefix + " clip_rms alpha=" + String(alpha), m)
        if m.rel_l2 < best.rel_l2:
            best = m.copy()
            best_alpha = alpha
    print_metrics(label_prefix + " clip_rms best alpha=" + String(best_alpha), best)

    ref_f32.free()
    work.free()


def gemv_rows_f32[N: Int, K: Int](
    dst_ptr: Int,
    act_ptr: Int,
    wpacked_ptr: Int,
    colsum_ptr: Int,
    wscale_ptr: Int,
    act_scale_ptr: Int,
    rows: Int,
):
    var dst = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=dst_ptr)
    var act_i8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=wpacked_ptr)
    var colsum = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=colsum_ptr)
    var wscale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=wscale_ptr)
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale_ptr)
    for m in range(rows):
        var act_dequant = act_scales[m] / Float32(127)
        gemv_row[N, K, DType.float32](
            act_i8 + m * K, wpacked, act_dequant, wscale, colsum, dst + m * N)


def silu_mul_f32(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.float32]()
    debug_assert(count % width == 0, "silu_mul_f32: count must be SIMD aligned")
    var k = 0
    while k + width <= count:
        var g = (gate + k).load[width=width]()
        var u = (up + k).load[width=width]()
        var sig = SIMD[DType.float32, width](1.0) / (
            SIMD[DType.float32, width](1.0) + exp_f32[width](-g))
        (dst + k).store(g * sig * u)
        k += width


def unpack_vnni_to_rowmajor(
    src_packed: UnsafePointer[UInt8, MutAnyOrigin],
    dst_rowmajor: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    debug_assert(rows % VNNI_N_STEP == 0, "unpack_vnni_to_rowmajor: rows must be multiple of VNNI_N_STEP")
    debug_assert(cols % VNNI_K_STEP == 0, "unpack_vnni_to_rowmajor: cols must be multiple of VNNI_K_STEP")
    var n_block = compute_n_block(rows, cols)
    var k_block = cols

    for n_block_begin in range(0, rows, n_block):
        var n_block_size = min(n_block, rows - n_block_begin)

        for k_block_begin in range(0, cols, k_block):
            var k_block_size = min(k_block, cols - k_block_begin)

            for n_begin in range(0, n_block_size, VNNI_N_STEP):
                for k_begin in range(0, k_block_size, VNNI_K_STEP):
                    var tile_base = (
                        n_block_begin * cols
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * VNNI_N_STEP
                    )
                    var tile0 = UnsafePointer[Int32, MutAnyOrigin](
                        unsafe_from_address=Int(src_packed) + tile_base)
                    var tile1 = UnsafePointer[Int32, MutAnyOrigin](
                        unsafe_from_address=Int(src_packed) + tile_base + VNNI_TILE_N * VNNI_K_STEP)

                    for i in range(VNNI_TILE_N):
                        var row0 = UnsafePointer[Int32, MutAnyOrigin](
                            unsafe_from_address=Int(dst_rowmajor + (n_block_begin + n_begin + i) * cols + k_block_begin + k_begin))
                        var row1 = UnsafePointer[Int32, MutAnyOrigin](
                            unsafe_from_address=Int(dst_rowmajor + (n_block_begin + n_begin + VNNI_TILE_N + i) * cols + k_block_begin + k_begin))
                        for j in range(VNNI_K_STEP // VNNI_BLK):
                            row0[j] = tile0[j * VNNI_TILE_N + i]
                            row1[j] = tile1[j * VNNI_TILE_N + i]


def dequant_i8_weights_f32(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    comptime width = simd_width_of[DType.float32]()
    for n in range(rows):
        var sc = scales[n]
        var vsc = SIMD[DType.float32, width](sc)
        var src_row = src + n * cols
        var dst_row = dst + n * cols
        var k = 0
        while k + width <= cols:
            (dst_row + k).store((src_row + k).load[width=width]().cast[DType.float32]() * vsc)
            k += width
        while k < cols:
            dst_row[k] = Float32(src_row[k]) * sc
            k += 1


def fwht_rows_inplace[cols: Int, block: Int](
    buf: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
):
    for r in range(rows):
        var row = buf + r * cols
        for b in range(cols // block):
            fwht_block[block](row + b * block)


def linear_f32_f32[N: Int, K: Int](
    dst: UnsafePointer[Float32, MutAnyOrigin],
    act: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
):
    comptime width = simd_width_of[DType.float32]()
    for m in range(rows):
        var act_row = act + m * K
        var dst_row = dst + m * N
        for n in range(N):
            var w_row = weight + n * K
            var acc = SIMD[DType.float32, width](0)
            var k = 0
            while k + width <= K:
                acc = (act_row + k).load[width=width]().fma(
                    (w_row + k).load[width=width](), acc)
                k += width
            var sum = acc.reduce_add()
            while k < K:
                sum += act_row[k] * w_row[k]
                k += 1
            dst_row[n] = sum


def rmsnorm_no_gamma_f32[cols: Int](
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
):
    comptime width = simd_width_of[DType.float32]()
    for m in range(rows):
        var src_row = src + m * cols
        var dst_row = dst + m * cols
        var vsq = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            var x = (src_row + k).load[width=width]().cast[DType.float32]()
            (dst_row + k).store(x)
            vsq = x.fma(x, vsq)
            k += width
        var sumsq = vsq.reduce_add()
        while k < cols:
            var x = src_row[k].cast[DType.float32]()
            dst_row[k] = x
            sumsq += x * x
            k += 1
        var inv_rms = Float32(1.0 / (Float64(sumsq / Float32(cols) + Float32(1e-5)).__pow__(0.5)))
        var vinv = SIMD[DType.float32, width](inv_rms)
        k = 0
        while k + width <= cols:
            (dst_row + k).store((dst_row + k).load[width=width]() * vinv)
            k += width
        while k < cols:
            dst_row[k] = dst_row[k] * inv_rms
            k += 1


def absorb_gamma_weight_f32(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src_weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    comptime width = simd_width_of[DType.float32]()
    for n in range(rows):
        var src_row = src_weight + n * cols
        var dst_row = dst + n * cols
        var k = 0
        while k + width <= cols:
            var w = (src_row + k).load[width=width]().cast[DType.float32]()
            var g = (gamma + k).load[width=width]().cast[DType.float32]()
            (dst_row + k).store(w * g)
            k += width
        while k < cols:
            dst_row[k] = src_row[k].cast[DType.float32]() * gamma[k].cast[DType.float32]()
            k += 1


def read_row_shard(
    mut reader: FileReader,
    meta: TensorMeta,
    row_start: Int,
    rows: Int,
    cols: Int,
    elem_bytes: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
) -> Bool:
    return reader.read(
        meta.start + row_start * cols * elem_bytes,
        dst,
        rows * cols * elem_bytes)


def projection_source_probe_from_file[N: Int, K: Int](
    label: String,
    mut reader: FileReader,
    ref_x_main_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    qdq_act_f32: UnsafePointer[Float32, MutAnyOrigin],
    gamma_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ref_weight_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    header: SafetensorsHeader,
    quant_weight_name: String,
    row_start: Int,
    kernel_out_f32: UnsafePointer[Float32, MutAnyOrigin],
    ref_out_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
):
    var weight_meta = header.tensors.get(quant_weight_name).value().copy()
    var scale_meta = header.tensors.get(quant_weight_name + "_scale").value().copy()

    var weight_qi = alloc[Scalar[DType.int8]](N * K)
    var weight_scales = alloc[Float32](N)
    if not read_row_shard(
        reader, weight_meta, row_start, N, K, 1,
        weight_qi.bitcast[UInt8]()):
        print(label, " read weight shard failed")
        weight_qi.free()
        weight_scales.free()
        return
    if not read_row_shard(
        reader, scale_meta, row_start, N, 1, size_of[Float32](),
        weight_scales.bitcast[UInt8]()):
        print(label, " read scale shard failed")
        weight_qi.free()
        weight_scales.free()
        return

    var act_ref_nogamma_f32 = alloc[Float32](rows * K)
    rmsnorm_no_gamma_f32[K](act_ref_nogamma_f32, ref_x_main_bf16, rows)

    var ref_weight_gammaabs_f32 = alloc[Float32](N * K)
    absorb_gamma_weight_f32(
        ref_weight_gammaabs_f32, ref_weight_bf16, gamma_bf16, N, K)

    var weight_qdq_f32 = alloc[Float32](N * K)
    dequant_i8_weights_f32(
        weight_qdq_f32, weight_qi, weight_scales, N, K)
    fwht_rows_inplace[K, FWHT_BLOCK](weight_qdq_f32, N)

    var out_act_only = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_act_only, qdq_act_f32, ref_weight_gammaabs_f32, rows)
    print_metrics(label + ".act_only",
        metrics_f32_vs_bf16(out_act_only, ref_out_bf16, rows * N))

    var out_weight_only = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_weight_only, act_ref_nogamma_f32, weight_qdq_f32, rows)
    print_metrics(label + ".weight_only",
        metrics_f32_vs_bf16(out_weight_only, ref_out_bf16, rows * N))

    var out_both_qdq = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_both_qdq, qdq_act_f32, weight_qdq_f32, rows)
    print_metrics(label + ".both_qdq",
        metrics_f32_vs_bf16(out_both_qdq, ref_out_bf16, rows * N))
    print_metrics(label + ".kernel_vs_both_qdq",
        metrics_f32_vs_f32(kernel_out_f32, out_both_qdq, rows * N))

    weight_qi.free()
    weight_scales.free()
    act_ref_nogamma_f32.free()
    ref_weight_gammaabs_f32.free()
    weight_qdq_f32.free()
    out_act_only.free()
    out_weight_only.free()
    out_both_qdq.free()


def projection_act_only_probe[N: Int, K: Int](
    label: String,
    qdq_act_f32: UnsafePointer[Float32, MutAnyOrigin],
    gamma_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ref_weight_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ref_out_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
):
    var ref_weight_gammaabs_f32 = alloc[Float32](N * K)
    absorb_gamma_weight_f32(
        ref_weight_gammaabs_f32, ref_weight_bf16, gamma_bf16, N, K)

    var out = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out, qdq_act_f32, ref_weight_gammaabs_f32, rows)
    print_metrics(label, metrics_f32_vs_bf16(out, ref_out_bf16, rows * N))

    ref_weight_gammaabs_f32.free()
    out.free()


def projection_source_probe[N: Int, K: Int](
    label: String,
    ref_act_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    qdq_act_f32: UnsafePointer[Float32, MutAnyOrigin],
    ref_weight_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    packed_weight_ptr: Int,
    row_scale_ptr: Int,
    kernel_out_f32: UnsafePointer[Float32, MutAnyOrigin],
    ref_out_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows: Int,
):
    var ref_act_f32 = alloc[Float32](rows * K)
    copy_bf16_to_f32(ref_act_f32, ref_act_bf16, rows * K)

    var ref_weight_f32 = alloc[Float32](N * K)
    copy_bf16_to_f32(ref_weight_f32, ref_weight_bf16, N * K)

    var weight_qi = alloc[Scalar[DType.int8]](N * K)
    unpack_vnni_to_rowmajor(
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=packed_weight_ptr),
        weight_qi, N, K)

    var weight_qdq_f32 = alloc[Float32](N * K)
    dequant_i8_weights_f32(
        weight_qdq_f32,
        weight_qi,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=row_scale_ptr),
        N, K)

    var out_act_only = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_act_only, qdq_act_f32, ref_weight_f32, rows)
    print_metrics(label + ".act_only",
        metrics_f32_vs_bf16(out_act_only, ref_out_bf16, rows * N))

    var out_weight_only = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_weight_only, ref_act_f32, weight_qdq_f32, rows)
    print_metrics(label + ".weight_only",
        metrics_f32_vs_bf16(out_weight_only, ref_out_bf16, rows * N))

    var out_both_qdq = alloc[Float32](rows * N)
    linear_f32_f32[N, K](out_both_qdq, qdq_act_f32, weight_qdq_f32, rows)
    print_metrics(label + ".both_qdq",
        metrics_f32_vs_bf16(out_both_qdq, ref_out_bf16, rows * N))
    print_metrics(label + ".kernel_vs_both_qdq",
        metrics_f32_vs_f32(kernel_out_f32, out_both_qdq, rows * N))

    ref_act_f32.free()
    ref_weight_f32.free()
    weight_qi.free()
    weight_qdq_f32.free()
    out_act_only.free()
    out_weight_only.free()
    out_both_qdq.free()


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
    var quant_path = Path(QUANT_PATH)
    var header_opt = parse_safetensors_header(quant_path)
    if not header_opt:
        print("quant header parse failed")
        return
    var quant_header = header_opt.take()
    var reader = FileReader(quant_header.data_offset)
    var paths = List[Path]()
    paths.append(quant_path)
    try:
        _ = reader.ring.register_files[ReadMode](paths)
    except:
        print("quant file register failed")
        return

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
        Int(quant_attn_scales), Float32(1e-5), seq_len, quant.pools[0]).join()
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
        Int(quant_attn_scales), Float32(1e-5), seq_len, quant.pools[0]).join()
    dequant_fwht_rows[C.HIDDEN, FWHT_BLOCK](
        quant_mlp_norm_f32, quant_mlp_i8, seq_len, C.HIDDEN, quant_attn_scales)
    print_metrics("mlp_norm", metrics_f32_vs_bf16(
        quant_mlp_norm_f32, ref_mlp_norm, seq_len * C.HIDDEN))
    var ref_mlp_norm_nogamma_f32 = alloc[Float32](seq_len * C.HIDDEN)
    rmsnorm_no_gamma_f32[C.HIDDEN](
        ref_mlp_norm_nogamma_f32,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=ref_host.x_main(seq_len).ptr),
        seq_len)
    print_metrics("mlp_norm_nogamma", metrics_f32_vs_f32(
        quant_mlp_norm_f32, ref_mlp_norm_nogamma_f32, seq_len * C.HIDDEN))
    print_quant_diag(
        "mlp_norm_qdq_no_fwht",
        analyze_absmax_qdq_no_fwht[C.HIDDEN](
            ref_mlp_norm_nogamma_f32, seq_len))
    print_quant_diag(
        "mlp_norm_qdq_fwht",
        analyze_absmax_qdq_fwht[C.HIDDEN, FWHT_BLOCK](
            ref_mlp_norm_nogamma_f32, seq_len))
    comptime MLP_GROUP3 = FWHT_BLOCK * 3
    var quant_mlp_norm_group3_f32 = alloc[Float32](seq_len * C.HIDDEN)
    copy_f32(quant_mlp_norm_group3_f32, ref_mlp_norm_nogamma_f32, seq_len * C.HIDDEN)
    fwht_quantize_dequant_fwht_group_absmax[C.HIDDEN, FWHT_BLOCK, MLP_GROUP3](
        quant_mlp_norm_group3_f32, seq_len)
    print_metrics("mlp_norm_qdq_fwht_group3",
        metrics_f32_vs_f32(
            quant_mlp_norm_group3_f32, ref_mlp_norm_nogamma_f32, seq_len * C.HIDDEN))
    print_quant_diag(
        "mlp_norm_qdq_fwht_group3",
        analyze_absmax_qdq_fwht_grouped[C.HIDDEN, FWHT_BLOCK, MLP_GROUP3](
            ref_mlp_norm_nogamma_f32, seq_len))

    var quant_mlp_norm_group1_f32 = alloc[Float32](seq_len * C.HIDDEN)
    copy_f32(quant_mlp_norm_group1_f32, ref_mlp_norm_nogamma_f32, seq_len * C.HIDDEN)
    fwht_quantize_dequant_fwht_group_absmax[C.HIDDEN, FWHT_BLOCK, FWHT_BLOCK](
        quant_mlp_norm_group1_f32, seq_len)
    print_metrics("mlp_norm_qdq_fwht_group1",
        metrics_f32_vs_f32(
            quant_mlp_norm_group1_f32, ref_mlp_norm_nogamma_f32, seq_len * C.HIDDEN))
    print_quant_diag(
        "mlp_norm_qdq_fwht_group1",
        analyze_absmax_qdq_fwht_grouped[C.HIDDEN, FWHT_BLOCK, FWHT_BLOCK](
            ref_mlp_norm_nogamma_f32, seq_len))

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
        sweep_post_quantizer[GATE_ROWS, FWHT_BLOCK](
            "fused_gu_silu.roundtrip tp=" + String(tp) + " rank=" + String(rank),
            silu_ref, seq_len)

        var gate_quant_f32 = alloc[Float32](seq_len * GATE_ROWS)
        var up_quant_f32 = alloc[Float32](seq_len * GATE_ROWS)
        gemv_rows_f32[GATE_ROWS, C.HIDDEN](
            Int(gate_quant_f32),
            Int(quant_mlp_i8),
            quant_rv.layer_weight[QuantL.GATE_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.GATE_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.GATE_ROW_SCALE](layer_idx).ptr,
            Int(quant_attn_scales),
            seq_len)
        gemv_rows_f32[GATE_ROWS, C.HIDDEN](
            Int(up_quant_f32),
            Int(quant_mlp_i8),
            quant_rv.layer_weight[QuantL.UP_PROJ](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.UP_COLSUM](layer_idx).ptr,
            quant_rv.layer_weight[QuantL.UP_ROW_SCALE](layer_idx).ptr,
            Int(quant_attn_scales),
            seq_len)
        print_metrics("gate_proj_f32 tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(gate_quant_f32, gate_ref, seq_len * GATE_ROWS))
        print_metrics("up_proj_f32 tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(up_quant_f32, up_ref, seq_len * GATE_ROWS))
        projection_act_only_probe[GATE_ROWS, C.HIDDEN](
            "gate_proj fwht_group3.act_only tp=" + String(tp) + " rank=" + String(rank),
            quant_mlp_norm_group3_f32,
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.GATE_PROJ](layer_idx).ptr),
            gate_ref,
            seq_len)
        projection_act_only_probe[GATE_ROWS, C.HIDDEN](
            "up_proj fwht_group3.act_only tp=" + String(tp) + " rank=" + String(rank),
            quant_mlp_norm_group3_f32,
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.UP_PROJ](layer_idx).ptr),
            up_ref,
            seq_len)
        projection_act_only_probe[GATE_ROWS, C.HIDDEN](
            "gate_proj fwht_group1.act_only tp=" + String(tp) + " rank=" + String(rank),
            quant_mlp_norm_group1_f32,
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.GATE_PROJ](layer_idx).ptr),
            gate_ref,
            seq_len)
        projection_act_only_probe[GATE_ROWS, C.HIDDEN](
            "up_proj fwht_group1.act_only tp=" + String(tp) + " rank=" + String(rank),
            quant_mlp_norm_group1_f32,
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                unsafe_from_address=ref_rv.layer_weight[RefL.UP_PROJ](layer_idx).ptr),
            up_ref,
            seq_len)
        if rank == 0:
            projection_source_probe_from_file[GATE_ROWS, C.HIDDEN](
                "gate_proj tp=" + String(tp) + " rank=" + String(rank),
                reader,
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_host.x_main(seq_len).ptr),
                quant_mlp_norm_f32,
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_rv.layer_weight[RefL.GATE_PROJ](layer_idx).ptr),
                quant_header,
                "model.layers." + String(layer_idx) + ".mlp.gate_proj.weight",
                rank * GATE_ROWS,
                gate_quant_f32,
                gate_ref,
                seq_len)
            projection_source_probe_from_file[GATE_ROWS, C.HIDDEN](
                "up_proj tp=" + String(tp) + " rank=" + String(rank),
                reader,
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_host.x_main(seq_len).ptr),
                quant_mlp_norm_f32,
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_rv.layer_weight[RefL.POST_ATTN_NORM](layer_idx).ptr),
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=ref_rv.layer_weight[RefL.UP_PROJ](layer_idx).ptr),
                quant_header,
                "model.layers." + String(layer_idx) + ".mlp.up_proj.weight",
                rank * GATE_ROWS,
                up_quant_f32,
                up_ref,
                seq_len)

        var gate_ref_f32 = alloc[Float32](seq_len * GATE_ROWS)
        var up_ref_f32 = alloc[Float32](seq_len * GATE_ROWS)
        copy_bf16_to_f32(gate_ref_f32, gate_ref, seq_len * GATE_ROWS)
        copy_bf16_to_f32(up_ref_f32, up_ref, seq_len * GATE_ROWS)

        var silu_sep_f32 = alloc[Float32](seq_len * GATE_ROWS)
        silu_mul_f32(
            silu_sep_f32,
            gate_quant_f32,
            up_quant_f32,
            seq_len * GATE_ROWS)
        print_metrics("silu_prequant_sep tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(silu_sep_f32, silu_ref, seq_len * GATE_ROWS))

        var silu_gate_only_f32 = alloc[Float32](seq_len * GATE_ROWS)
        silu_mul_f32(
            silu_gate_only_f32,
            gate_quant_f32,
            up_ref_f32,
            seq_len * GATE_ROWS)
        print_metrics("silu_gate_only tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(silu_gate_only_f32, silu_ref, seq_len * GATE_ROWS))

        var silu_up_only_f32 = alloc[Float32](seq_len * GATE_ROWS)
        silu_mul_f32(
            silu_up_only_f32,
            gate_ref_f32,
            up_quant_f32,
            seq_len * GATE_ROWS)
        print_metrics("silu_up_only tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(silu_up_only_f32, silu_ref, seq_len * GATE_ROWS))

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

        var silu_sep_qdq_f32 = alloc[Float32](seq_len * GATE_ROWS)
        copy_f32(silu_sep_qdq_f32, silu_sep_f32, seq_len * GATE_ROWS)
        fwht_quantize_dequant_fwht_absmax[GATE_ROWS, FWHT_BLOCK](
            silu_sep_qdq_f32, seq_len)
        print_metrics("silu_sep_absmax_qdq tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_bf16(silu_sep_qdq_f32, silu_ref, seq_len * GATE_ROWS))
        print_metrics("fused_vs_sep_absmax tp=" + String(tp) + " rank=" + String(rank),
            metrics_f32_vs_f32(silu_quant_f32, silu_sep_qdq_f32, seq_len * GATE_ROWS))

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
