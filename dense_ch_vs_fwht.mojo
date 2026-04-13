"""Dense FFN quantization quality: FWHT vs channelwise on real weights+activations.

Loads dumped x_main activations (from running test_gemma4_butterquant with
DUMP_FFN_INPUTS=True) and bf16 HF checkpoint weights for each layer's dense
FFN. Computes:
  1. f32 reference: norm(x) -> gate/up GEMV -> gelu_tanh*up -> down GEMV
  2. FWHT path: FWHT-rotate + per-block i8 quantize intermediate + weight
  3. Channelwise path: per-block i8 quantize intermediate + weight (no FWHT)
Reports RMSE, NRMSE, cosine similarity, max absolute error per layer.

Run: pixi run mojo build -I . dense_ch_vs_fwht.mojo && ./dense_ch_vs_fwht
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.math import log, abs
from std.pathlib import Path
from std.collections import InlineArray

from simd_math import sqrt as simd_sqrt, roundeven
from safetensors.parser import parse_safetensors_header, SafetensorsHeader
from notstdcollections import HeapMoveArray
from modeling.loader import discover_shards, find_tensor

from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.common_math import F32Ptr, I8Ptr, BF16Ptr


comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112
comptime FWHT_BLK = 64
comptime NUM_BLK_INT = INTERMEDIATE // FWHT_BLK  # 33
comptime NUM_BLK_HID = HIDDEN // FWHT_BLK        # 44
comptime NUM_LAYERS = 30
comptime BF16_MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime DUMP_DIR = "dump_activations"
comptime RMS_NORM_EPS = Float32(1e-6)
comptime DUMP_POS = 4


# ── Tensor loading ───────────────────────────────────────────────────

def load_bf16_tensor(
    name: String,
    ref headers: HeapMoveArray[SafetensorsHeader],
    ref shards: List[Path],
) -> Optional[BF16Ptr]:
    var found = find_tensor(name, headers)
    if not found:
        print("  missing: " + name)
        return None
    var shard_idx = found.value()[0]
    var meta = found.value()[1].copy()
    var byte_off = headers[shard_idx].data_offset + meta.start
    var nbytes = meta.end - meta.start
    try:
        with open(shards[shard_idx], "r") as f:
            _ = f.seek(UInt64(byte_off), 0)
            var data = f.read_bytes(size=nbytes)
            if len(data) != nbytes:
                print("  short read: " + name)
                return None
            var buf = alloc[Scalar[DType.bfloat16]](nbytes // 2)
            memcpy(dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf)),
                   src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(data.unsafe_ptr())),
                   count=nbytes)
            return BF16Ptr(unsafe_from_address=Int(buf))
    except:
        print("  read failed: " + name)
        return None


def load_dump(path: String, count: Int) -> Optional[BF16Ptr]:
    var nbytes = count * 2
    try:
        with open(path, "r") as f:
            var data = f.read_bytes(size=nbytes)
            if len(data) != nbytes:
                print("  short read: " + path + " got " + String(len(data)))
                return None
            var buf = alloc[Scalar[DType.bfloat16]](count)
            memcpy(dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf)),
                   src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(data.unsafe_ptr())),
                   count=nbytes)
            return BF16Ptr(unsafe_from_address=Int(buf))
    except:
        print("  read failed: " + path)
        return None


# ── Math helpers ─────────────────────────────────────────────────────

def rmsnorm_f32(x_bf16: BF16Ptr, gamma_bf16: BF16Ptr, dst: F32Ptr, n: Int):
    """RMSNorm: dst = x / rms(x) * gamma, computed in f32."""
    var ss = Float32(0)
    for i in range(n):
        var v = Float32(x_bf16[i])
        ss += v * v
    var inv = Float32(1.0) / simd_sqrt(ss / Float32(n) + RMS_NORM_EPS)
    for i in range(n):
        dst[i] = Float32(x_bf16[i]) * inv * Float32(gamma_bf16[i])


def gemv_f32(
    weight_bf16: BF16Ptr, x_f32: F32Ptr,
    dst: F32Ptr, rows: Int, cols: Int,
):
    """Dense GEMV: dst[n] = sum_k weight[n,k] * x[k], in f32."""
    for n in range(rows):
        var acc = Float32(0)
        var row_off = n * cols
        for k in range(cols):
            acc += Float32(weight_bf16[row_off + k]) * x_f32[k]
        dst[n] = acc


def fwht_rotate_f32[block: Int](buf: F32Ptr, n: Int):
    """Block-diagonal FWHT rotation in-place."""
    for b in range(n // block):
        fwht_block[block](buf + b * block)


def absmax_quantize_row(src: F32Ptr, dst: I8Ptr, cols: Int) -> Float32:
    """Per-row absmax i8 quantization. Returns scale = absmax/127."""
    var amax = Float32(0)
    for k in range(cols):
        var v = abs(src[k])
        if v > amax:
            amax = v
    if amax < Float32(1e-10):
        amax = Float32(1e-10)
    var inv = Float32(127.0) / amax
    for k in range(cols):
        dst[k] = roundeven(src[k] * inv).cast[DType.int8]()
    return amax / Float32(127.0)


def absmax_quantize_block(src: F32Ptr, dst: I8Ptr, scales: F32Ptr,
    n: Int, block: Int):
    """Per-block absmax i8 quantization."""
    for b in range(n // block):
        var off = b * block
        var amax = Float32(0)
        for k in range(block):
            var v = abs(src[off + k])
            if v > amax:
                amax = v
        if amax < Float32(1e-10):
            amax = Float32(1e-10)
        scales[b] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax
        for k in range(block):
            dst[off + k] = roundeven(src[off + k] * inv).cast[DType.int8]()


def gemv_i8_blocked[block: Int](
    act_i8: I8Ptr, act_blk_scale: F32Ptr,
    w_i8: I8Ptr, w_row_scale: F32Ptr,
    dst: F32Ptr, rows: Int, cols: Int,
):
    """i8 GEMV with per-block activation scales, per-row weight scales.

    dst[n] = w_row_scale[n] * sum_blk( act_blk_scale[blk] *
             sum_{k in blk} w_i8[n,k] * act_i8[k] )
    """
    var num_blk = cols // block
    for n in range(rows):
        var row_off = n * cols
        var acc = Float32(0)
        for blk in range(num_blk):
            var blk_off = blk * block
            var blk_acc = Float32(0)
            for k in range(block):
                blk_acc += Float32(Int(w_i8[row_off + blk_off + k])) * Float32(Int(act_i8[blk_off + k]))
            acc += blk_acc * act_blk_scale[blk]
        dst[n] = acc * w_row_scale[n]


# ── Error statistics ─────────────────────────────────────────────────

@fieldwise_init
struct ErrorReport(Copyable, ImplicitlyCopyable):
    var rmse: Float64
    var nrmse: Float64
    var cosine: Float64
    var max_abs: Float64

def compute_error(ref_buf: F32Ptr, test: F32Ptr, n: Int) -> ErrorReport:
    var dot_rt = Float64(0)
    var norm_r = Float64(0)
    var norm_t = Float64(0)
    var sum_sq_err = Float64(0)
    var sum_sq_ref = Float64(0)
    var max_abs_err = Float64(0)
    for i in range(n):
        var r = Float64(ref_buf[i])
        var t = Float64(test[i])
        var e = (t - r).__abs__()
        dot_rt += r * t
        norm_r += r * r
        norm_t += t * t
        sum_sq_err += (t - r) * (t - r)
        sum_sq_ref += r * r
        if e > max_abs_err:
            max_abs_err = e
    var cos = dot_rt / (Float64(simd_sqrt(Float32(norm_r))) * Float64(simd_sqrt(Float32(norm_t)))) if norm_r > 0 and norm_t > 0 else Float64(0)
    var rmse = Float64(simd_sqrt(Float32(sum_sq_err / Float64(n))))
    var nrmse = Float64(simd_sqrt(Float32(sum_sq_err / sum_sq_ref))) if sum_sq_ref > 0 else Float64(0)
    return ErrorReport(rmse, nrmse, cos, max_abs_err)


def print_error(label: String, err: ErrorReport):
    print("  " + label + ":"
        + " cosine=" + String(err.cosine)
        + " NRMSE=" + String(err.nrmse)
        + " RMSE=" + String(err.rmse)
        + " max=" + String(err.max_abs))


# ── Per-layer comparison ─────────────────────────────────────────────

def compare_layer(
    layer_idx: Int,
    x_main: BF16Ptr,
    norm_gamma: BF16Ptr,
    gate_w: BF16Ptr,
    up_w: BF16Ptr,
    down_w: BF16Ptr,
    # scratch buffers
    normed: F32Ptr,
    gate_out: F32Ptr,
    up_out: F32Ptr,
    inter: F32Ptr,
    y_ref: F32Ptr,
    y_test: F32Ptr,
    work: F32Ptr,
    w_row_f32: F32Ptr,
    act_i8: I8Ptr,
    w_i8: I8Ptr,
    act_blk_sc: F32Ptr,
    w_row_sc: F32Ptr,
):
    print("layer " + String(layer_idx) + ":")

    # 1. RMSNorm
    rmsnorm_f32(x_main, norm_gamma, normed, HIDDEN)

    # 2. gate/up GEMV in f32
    gemv_f32(gate_w, normed, gate_out, INTERMEDIATE, HIDDEN)
    gemv_f32(up_w, normed, up_out, INTERMEDIATE, HIDDEN)

    # 3. gelu_tanh(gate) * up → intermediate f32
    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= INTERMEDIATE:
        var g = (gate_out + k).load[width=width]()
        var u = (up_out + k).load[width=width]()
        (inter + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    # 4. f32 reference: y_ref = down @ inter
    gemv_f32(down_w, inter, y_ref, HIDDEN, INTERMEDIATE)

    # Measure intermediate activation stats
    var inter_absmax = Float32(0)
    var inter_energy = Float64(0)
    for i in range(INTERMEDIATE):
        var v = abs(inter[i])
        if v > inter_absmax:
            inter_absmax = v
        inter_energy += Float64(inter[i]) * Float64(inter[i])
    var inter_rms = Float64(simd_sqrt(Float32(inter_energy / Float64(INTERMEDIATE))))
    print("  intermediate: absmax=" + String(inter_absmax) + " rms=" + String(inter_rms)
        + " peak/rms=" + String(Float64(inter_absmax) / inter_rms))

    # === Path A: FWHT ===
    # Quantize down_proj weight: FWHT-rotate each row, then per-row absmax
    for n in range(HIDDEN):
        var row_off = n * INTERMEDIATE
        for j in range(INTERMEDIATE):
            w_row_f32[j] = Float32(down_w[row_off + j])
        fwht_rotate_f32[FWHT_BLK](w_row_f32, INTERMEDIATE)
        w_row_sc[n] = absmax_quantize_row(w_row_f32, w_i8 + row_off, INTERMEDIATE)

    # Quantize activation: FWHT-rotate, then per-block absmax
    memcpy(dest=work, src=inter, count=INTERMEDIATE)
    fwht_rotate_f32[FWHT_BLK](work, INTERMEDIATE)
    absmax_quantize_block(work, act_i8, act_blk_sc, INTERMEDIATE, FWHT_BLK)

    # GEMV
    gemv_i8_blocked[FWHT_BLK](act_i8, act_blk_sc, w_i8, w_row_sc,
        y_test, HIDDEN, INTERMEDIATE)

    var err_fwht = compute_error(y_ref, y_test, HIDDEN)
    print_error("FWHT (block=" + String(FWHT_BLK) + ")", err_fwht)

    # === Path B: Channelwise (no FWHT) ===
    # Quantize down_proj weight: per-row absmax, no rotation
    for n in range(HIDDEN):
        var row_off = n * INTERMEDIATE
        for j in range(INTERMEDIATE):
            w_row_f32[j] = Float32(down_w[row_off + j])
        w_row_sc[n] = absmax_quantize_row(w_row_f32, w_i8 + row_off, INTERMEDIATE)

    # Quantize activation: per-block absmax, no FWHT
    absmax_quantize_block(inter, act_i8, act_blk_sc, INTERMEDIATE, FWHT_BLK)

    # GEMV
    gemv_i8_blocked[FWHT_BLK](act_i8, act_blk_sc, w_i8, w_row_sc,
        y_test, HIDDEN, INTERMEDIATE)

    var err_ch = compute_error(y_ref, y_test, HIDDEN)
    print_error("channelwise", err_ch)

    # === Summary ===
    var ratio = err_ch.nrmse / err_fwht.nrmse if err_fwht.nrmse > 0 else Float64(0)
    print("  NRMSE ratio (ch/fwht): " + String(ratio)
        + " (" + (String("FWHT wins") if ratio > 1.0 else String("channelwise wins")) + ")")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=== Dense FFN Quantization: FWHT vs Channelwise ===")
    print("HIDDEN=" + String(HIDDEN) + " INTERMEDIATE=" + String(INTERMEDIATE)
        + " FWHT_BLK=" + String(FWHT_BLK))
    print()

    # Parse bf16 checkpoint
    var shards = discover_shards(Path(BF16_MODEL_DIR))
    if len(shards) == 0:
        print("no shards found in " + BF16_MODEL_DIR)
        return
    print("found " + String(len(shards)) + " shard(s)")

    var headers = HeapMoveArray[SafetensorsHeader](len(shards))
    for i in range(len(shards)):
        var h = parse_safetensors_header(shards[i])
        if not h:
            print("failed to parse header for shard " + String(i))
            return
        headers.push(h.take())

    # Allocate scratch
    var normed = alloc[Float32](HIDDEN)
    var gate_out = alloc[Float32](INTERMEDIATE)
    var up_out = alloc[Float32](INTERMEDIATE)
    var inter = alloc[Float32](INTERMEDIATE)
    var y_ref = alloc[Float32](HIDDEN)
    var y_test = alloc[Float32](HIDDEN)
    var work = alloc[Float32](INTERMEDIATE)
    var w_row_f32 = alloc[Float32](INTERMEDIATE)
    var act_i8 = alloc[Scalar[DType.int8]](INTERMEDIATE)
    var w_i8 = alloc[Scalar[DType.int8]](HIDDEN * INTERMEDIATE)
    var act_blk_sc = alloc[Float32](NUM_BLK_INT)
    var w_row_sc = alloc[Float32](HIDDEN)

    var fwht_nrmse_sum = Float64(0)
    var ch_nrmse_sum = Float64(0)
    var layers_tested = 0

    for layer_idx in range(NUM_LAYERS):
        # Load dumped activation
        var dump_path = String(DUMP_DIR) + "/pos_" + String(DUMP_POS) + "_layer_" + String(layer_idx) + ".bin"
        var x_opt = load_dump(dump_path, HIDDEN)
        if not x_opt:
            print("skipping layer " + String(layer_idx) + " (no dump at pos " + String(DUMP_POS) + ")")
            continue
        var x_main = x_opt.take()

        var prefix = "model.language_model.layers." + String(layer_idx) + "."

        # Load norm weight
        var norm_opt = load_bf16_tensor(prefix + "pre_feedforward_layernorm.weight", headers, shards)
        if not norm_opt:
            x_main.free()
            continue
        var norm_gamma = norm_opt.take()

        # Load gate/up/down weights
        var gate_opt = load_bf16_tensor(prefix + "mlp.gate_proj.weight", headers, shards)
        var up_opt = load_bf16_tensor(prefix + "mlp.up_proj.weight", headers, shards)
        var down_opt = load_bf16_tensor(prefix + "mlp.down_proj.weight", headers, shards)
        if not gate_opt or not up_opt or not down_opt:
            x_main.free()
            norm_gamma.free()
            if gate_opt:
                gate_opt.take().free()
            if up_opt:
                up_opt.take().free()
            if down_opt:
                down_opt.take().free()
            continue
        var gate_w = gate_opt.take()
        var up_w = up_opt.take()
        var down_w = down_opt.take()

        compare_layer(layer_idx, x_main, norm_gamma, gate_w, up_w, down_w,
            F32Ptr(unsafe_from_address=Int(normed)),
            F32Ptr(unsafe_from_address=Int(gate_out)),
            F32Ptr(unsafe_from_address=Int(up_out)),
            F32Ptr(unsafe_from_address=Int(inter)),
            F32Ptr(unsafe_from_address=Int(y_ref)),
            F32Ptr(unsafe_from_address=Int(y_test)),
            F32Ptr(unsafe_from_address=Int(work)),
            F32Ptr(unsafe_from_address=Int(w_row_f32)),
            I8Ptr(unsafe_from_address=Int(act_i8)),
            I8Ptr(unsafe_from_address=Int(w_i8)),
            F32Ptr(unsafe_from_address=Int(act_blk_sc)),
            F32Ptr(unsafe_from_address=Int(w_row_sc)))

        x_main.free()
        norm_gamma.free()
        gate_w.free()
        up_w.free()
        down_w.free()
        layers_tested += 1

    if layers_tested == 0:
        print("no layers tested! run test_gemma4_butterquant first to dump activations.")
    else:
        print("=== tested " + String(layers_tested) + " / " + String(NUM_LAYERS) + " layers ===")
