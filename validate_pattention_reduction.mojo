"""Validate fused context-parallel attention reduction.

Reads dumped partial attention states from taps/full_attn_*/,
runs baseline merge vs fused CP merge, compares O-proj outputs.

Requires: model loaded at TP=1 (for O-proj weight access),
and prior run with DUMP_ATTENTION=True to produce tap files.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from modeling.gemma_4_moe_butterquant_tp import (
    Gemma4Config, Gemma4ButterQuant, Gemma4Topology,
)
from modeling.gemma4_common import is_full_layer
from experimental3.kernels.full_chunked_attention import (
    merge_and_quantize, partial_head_stride, partial_chunk_stride,
)
from experimental3.kernels.gemm import int8_gemv_blocked_worker
from experimental3.kernels.dispatch_args import Int8GemvBlockedArgs
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from simd_math import exp_f32, sqrt


comptime C = Gemma4Config
comptime TP = 1
comptime MODEL_DIR = "quantized_models"
comptime HEAD_DIM = C.HEAD_DIM_FULL
comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
comptime HIDDEN = C.HIDDEN
comptime Q_DIM_LOCAL = C.Q_DIM_FULL // TP
comptime NUM_HEADS_LOCAL = C.NUM_HEADS // TP
comptime NUM_KV_LOCAL = C.NUM_KV_HEADS_FULL // TP
comptime HEAD_STRIDE = partial_head_stride[HEAD_DIM]()
comptime CHUNK_STRIDE = partial_chunk_stride[HEAD_DIM, HPG]()
comptime NUM_FULL_LAYERS = C.NUM_FULL_LAYERS
comptime QI_GROUP = HPG * HEAD_DIM


# =============================================================================
# File I/O
# =============================================================================


def read_f32(path: String) -> Optional[List[Float32]]:
    try:
        var fh = FileHandle(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        var n = len(raw) // 4
        var src = raw.unsafe_ptr().bitcast[Float32]()
        var buf = List[Float32](capacity=n)
        for i in range(n):
            buf.append(src[i])
        return buf^
    except:
        return None


def read_i32(path: String) -> Optional[Int]:
    try:
        var fh = FileHandle(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        if len(raw) < 4:
            return None
        return Int(raw.unsafe_ptr().bitcast[Int32]()[])
    except:
        return None


def read_bf16_as_f32(path: String) -> Optional[List[Float32]]:
    try:
        var fh = FileHandle(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        var n = len(raw) // 2
        var raw_ptr = raw.unsafe_ptr()
        var buf = List[Float32](capacity=n)
        for i in range(n):
            var lo = UInt16(raw_ptr[i * 2])
            var hi = UInt16(raw_ptr[i * 2 + 1])
            var bits = UInt32(lo) | (UInt32(hi) << 8)
            var f32_bits = bits << 16
            var val = UnsafePointer(to=f32_bits).bitcast[Float32]()[]
            buf.append(val)
        return buf^
    except:
        return None


# =============================================================================
# Comparison
# =============================================================================


def compare(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Tuple[Float32, Float32, Float32]:
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    var dot_ab = Float32(0)
    var na2 = Float32(0)
    var nb2 = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= 0 else -d
        if ad > max_diff:
            max_diff = ad
        sum_diff += ad
        dot_ab += a[i] * b[i]
        na2 += a[i] * a[i]
        nb2 += b[i] * b[i]
    var denom = sqrt(Float64(na2)) * sqrt(Float64(nb2))
    var cosine = Float32(Float64(dot_ab) / denom) if denom > 0 else Float32(0)
    return (max_diff, sum_diff / Float32(n), cosine)


def report(label: String, a: UnsafePointer[Float32, MutAnyOrigin],
           b: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var result = compare(a, b, n)
    var tag: String
    if result[2] > 0.999:
        tag = "OK"
    elif result[2] > 0.99:
        tag = "WARN"
    else:
        tag = "BAD"
    print("  ", label, "max:", result[0], "mean:", result[1],
          "cos:", result[2], tag)


# =============================================================================
# Merge partial states within a group of chunks → (m, l, v) per head
# =============================================================================


def merge_chunk_group(
    partial_base: UnsafePointer[Float32, _],
    chunk_indices: List[Int],
    out_m: UnsafePointer[Float32, MutAnyOrigin],
    out_l: UnsafePointer[Float32, MutAnyOrigin],
    out_v: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime width = simd_width_of[DType.float32]()

    for qh in range(HPG):
        var global_max = Float32(-1e30)
        for ci in range(len(chunk_indices)):
            var c = chunk_indices[ci]
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            global_max = max(global_max, p[])

        var total_sum = Float32(0)
        var v_ptr = out_v + qh * HEAD_DIM
        for d in range(HEAD_DIM):
            v_ptr[d] = Float32(0)

        for ci in range(len(chunk_indices)):
            var c = chunk_indices[ci]
            var p = partial_base + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            var chunk_max = p[]
            var chunk_sum = (p + 1)[]
            var v_acc = p + 2
            if chunk_sum <= 0:
                continue
            var rescale = Float32(exp_f32[1](chunk_max - global_max))
            total_sum += chunk_sum * rescale
            var d = 0
            while d + width <= HEAD_DIM:
                (v_ptr + d).store(
                    (v_acc + d).load[width=width]().fma(rescale,
                        (v_ptr + d).load[width=width]()))
                d += width

        out_m[qh] = global_max
        out_l[qh] = total_sum


# =============================================================================
# Fused CP: per-rank rescale → FWHT → i8 quantize
# =============================================================================


def fused_quantize_rank(
    v_raw: UnsafePointer[Float32, MutAnyOrigin],
    m_r: UnsafePointer[Float32, MutAnyOrigin],
    l_r: UnsafePointer[Float32, MutAnyOrigin],
    global_M: UnsafePointer[Float32, MutAnyOrigin],
    global_L: UnsafePointer[Float32, MutAnyOrigin],
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    head_scales: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime width = simd_width_of[DType.float32]()

    var work = InlineArray[Float32, HEAD_DIM](fill=Float32(0))
    var wp = UnsafePointer(to=work).bitcast[Float32]()

    for qh in range(HPG):
        var alpha = Float32(exp_f32[1](m_r[qh] - global_M[qh]))
        var scale = alpha / (Float32(127) * global_L[qh])
        var src = v_raw + qh * HEAD_DIM
        var d = 0
        while d + width <= HEAD_DIM:
            (wp + d).store((src + d).load[width=width]() * scale)
            d += width

        head_scales[qh] = absmax_quantize_i8[HEAD_DIM](
            wp, qi_out + qh * HEAD_DIM)


# =============================================================================
# Run baseline: merge_and_quantize → O-proj GEMV
# =============================================================================


def run_baseline(
    kv_partials: List[List[Float32]],
    kv_num_chunks: List[Int],
    topo: Gemma4Topology[TP],
    full_idx: Int,
    dst_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    var qi_buf = InlineArray[Scalar[DType.int8], Q_DIM_LOCAL](fill=Scalar[DType.int8](0))
    var qi = UnsafePointer(to=qi_buf).bitcast[Scalar[DType.int8]]()
    var scales_buf = InlineArray[Float32, NUM_HEADS_LOCAL](fill=Float32(0))
    var scales = UnsafePointer(to=scales_buf).bitcast[Float32]()

    for kv in range(NUM_KV_LOCAL):
        merge_and_quantize[HEAD_DIM, HPG](
            F32Ptr(unsafe_from_address=Int(kv_partials[kv].unsafe_ptr())),
            kv_num_chunks[kv],
            I8Ptr(unsafe_from_address=Int(qi) + kv * QI_GROUP),
            F32Ptr(unsafe_from_address=Int(scales) + kv * HPG * 4))

    var lb = topo.full_base(full_idx)
    var fl = topo.full.proto
    var args = Int8GemvBlockedArgs(
        I8Ptr(unsafe_from_address=Int(qi)),
        U8Ptr(unsafe_from_address=fl.attn.o_proj.addr(lb)),
        F32Ptr(unsafe_from_address=Int(scales)),
        fl.attn.o_proj_sc.bound(lb).as_ptr(),
        F32Ptr(unsafe_from_address=lb + fl.attn.o_colsum),
        BF16Ptr(unsafe_from_address=Int(dst_bf16)),
        Float32(1.0))
    int8_gemv_blocked_worker[HIDDEN, Q_DIM_LOCAL, HEAD_DIM](args)


# =============================================================================
# Run fused CP: per-rank merge → rescale → quantize → O-proj → sum
# =============================================================================


def run_fused(
    kv_partials: List[List[Float32]],
    kv_num_chunks: List[Int],
    num_cp_ranks: Int,
    topo: Gemma4Topology[TP],
    full_idx: Int,
    dst_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    comptime width = simd_width_of[DType.float32]()

    # Per KV head, per virtual rank: merge chunks → (m, l, v)
    # Layout: rank_states[kv][r] = (m[HPG], l[HPG], v[HPG * HEAD_DIM])
    var all_m = List[Float32]()
    var all_l = List[Float32]()
    var all_v = List[Float32]()
    for i in range(NUM_KV_LOCAL * num_cp_ranks * HPG):
        all_m.append(Float32(-1e30))
        all_l.append(Float32(0))
    for i in range(NUM_KV_LOCAL * num_cp_ranks * HPG * HEAD_DIM):
        all_v.append(Float32(0))

    var global_M = List[Float32]()
    var global_L = List[Float32]()
    for i in range(NUM_KV_LOCAL * HPG):
        global_M.append(Float32(-1e30))
        global_L.append(Float32(0))

    for kv in range(NUM_KV_LOCAL):
        var num_chunks = kv_num_chunks[kv]
        var partial_ptr = kv_partials[kv].unsafe_ptr()
        var chunks_per_rank = (num_chunks + num_cp_ranks - 1) // num_cp_ranks

        for r in range(num_cp_ranks):
            var idx = (kv * num_cp_ranks + r) * HPG
            var vidx = (kv * num_cp_ranks + r) * HPG * HEAD_DIM
            var indices = List[Int]()
            for c in range(r * chunks_per_rank,
                           min((r + 1) * chunks_per_rank, num_chunks)):
                indices.append(c)
            if len(indices) == 0:
                continue
            merge_chunk_group(
                partial_ptr, indices,
                all_m.unsafe_ptr() + idx,
                all_l.unsafe_ptr() + idx,
                all_v.unsafe_ptr() + vidx)

        for qh in range(HPG):
            var gidx = kv * HPG + qh
            for r in range(num_cp_ranks):
                var idx = (kv * num_cp_ranks + r) * HPG + qh
                global_M.unsafe_ptr()[gidx] = max(global_M[gidx], all_m[idx])
            for r in range(num_cp_ranks):
                var idx = (kv * num_cp_ranks + r) * HPG + qh
                global_L.unsafe_ptr()[gidx] += all_l[idx] * Float32(
                    exp_f32[1](all_m[idx] - global_M[gidx]))

    var acc_f32 = InlineArray[Float32, HIDDEN](fill=Float32(0))
    var acc = UnsafePointer(to=acc_f32).bitcast[Float32]()
    var qi_buf = InlineArray[Scalar[DType.int8], Q_DIM_LOCAL](fill=Scalar[DType.int8](0))
    var qi = UnsafePointer(to=qi_buf).bitcast[Scalar[DType.int8]]()
    var scales_buf = InlineArray[Float32, NUM_HEADS_LOCAL](fill=Float32(0))
    var scales = UnsafePointer(to=scales_buf).bitcast[Float32]()
    var rank_out_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
        fill=Scalar[DType.bfloat16](0))
    var rank_out = UnsafePointer(to=rank_out_buf).bitcast[Scalar[DType.bfloat16]]()

    var lb = topo.full_base(full_idx)
    var fl = topo.full.proto

    for r in range(num_cp_ranks):
        for kv in range(NUM_KV_LOCAL):
            var idx = (kv * num_cp_ranks + r) * HPG
            var vidx = (kv * num_cp_ranks + r) * HPG * HEAD_DIM
            var gidx = kv * HPG
            fused_quantize_rank(
                all_v.unsafe_ptr() + vidx,
                all_m.unsafe_ptr() + idx,
                all_l.unsafe_ptr() + idx,
                global_M.unsafe_ptr() + gidx,
                global_L.unsafe_ptr() + gidx,
                qi + kv * QI_GROUP,
                scales + kv * HPG)

        var args = Int8GemvBlockedArgs(
            I8Ptr(unsafe_from_address=Int(qi)),
            U8Ptr(unsafe_from_address=fl.attn.o_proj.addr(lb)),
            F32Ptr(unsafe_from_address=Int(scales)),
            fl.attn.o_proj_sc.bound(lb).as_ptr(),
            F32Ptr(unsafe_from_address=lb + fl.attn.o_colsum),
            BF16Ptr(unsafe_from_address=Int(rank_out)),
            Float32(1.0))
        int8_gemv_blocked_worker[HIDDEN, Q_DIM_LOCAL, HEAD_DIM](args)

        var k = 0
        while k + width <= HIDDEN:
            (acc + k).store(
                (acc + k).load[width=width]()
                + (rank_out + k).load[width=width]().cast[DType.float32]())
            k += width

    var k = 0
    while k + width <= HIDDEN:
        (dst_bf16 + k).store(
            (acc + k).load[width=width]().cast[DType.bfloat16]())
        k += width


# =============================================================================
# Main
# =============================================================================


def main():
    print("loading model at TP=1 for weight access...")
    var model_opt = Gemma4ButterQuant[TP].load(Path(MODEL_DIR))
    if not model_opt:
        print("failed to load model")
        return
    var model = model_opt.take()
    var topo = model.topos[0]
    print("model loaded\n")

    for full_idx in range(NUM_FULL_LAYERS):
        var kv_partials = List[List[Float32]]()
        var kv_num_chunks = List[Int]()
        var all_loaded = True
        var min_chunks = 9999

        for kv in range(NUM_KV_LOCAL):
            var dir = "taps/full_attn_" + String(full_idx) + "_kv" + String(kv)
            var maybe_nc = read_i32(dir + "/num_chunks.bin")
            if not maybe_nc:
                print("skipping full_idx", full_idx, "- no dump for kv", kv)
                all_loaded = False
                break
            var nc = maybe_nc.take()
            kv_num_chunks.append(nc)
            if nc < min_chunks:
                min_chunks = nc

            var maybe_p = read_f32(dir + "/partials.bin")
            if not maybe_p:
                print("skipping full_idx", full_idx, "- no partials for kv", kv)
                all_loaded = False
                break
            kv_partials.append(maybe_p.take())

        if not all_loaded:
            continue

        var base_dir = "taps/full_attn_" + String(full_idx)
        var maybe_baseline = read_bf16_as_f32(base_dir + "/baseline_oproj.bin")
        if not maybe_baseline:
            print("skipping full_idx", full_idx, "- no baseline oproj")
            continue
        var baseline_model = maybe_baseline.take()

        print("=== full_idx", full_idx, "kv_heads:", NUM_KV_LOCAL,
              "chunks:", kv_num_chunks[0], "===")

        var baseline_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
            fill=Scalar[DType.bfloat16](0))
        var baseline_bf16 = UnsafePointer(to=baseline_buf).bitcast[Scalar[DType.bfloat16]]()
        run_baseline(kv_partials, kv_num_chunks, topo, full_idx, baseline_bf16)

        var baseline_f32 = InlineArray[Float32, HIDDEN](fill=Float32(0))
        var bf = UnsafePointer(to=baseline_f32).bitcast[Float32]()
        for i in range(HIDDEN):
            bf[i] = Float32(baseline_bf16[i])

        var model_f32 = baseline_model.unsafe_ptr()
        print("  sanity (offline vs model):")
        report("", bf, model_f32, HIDDEN)

        for num_cp in range(2, 5, 2):
            if num_cp > min_chunks:
                print("  cp=" + String(num_cp), "skipped (need", num_cp,
                      "chunks, have", min_chunks, ")")
                continue

            var fused_buf = InlineArray[Scalar[DType.bfloat16], HIDDEN](
                fill=Scalar[DType.bfloat16](0))
            var fused_bf16 = UnsafePointer(to=fused_buf).bitcast[Scalar[DType.bfloat16]]()
            run_fused(kv_partials, kv_num_chunks, num_cp, topo, full_idx, fused_bf16)

            var fused_f32 = InlineArray[Float32, HIDDEN](fill=Float32(0))
            var ff = UnsafePointer(to=fused_f32).bitcast[Float32]()
            for i in range(HIDDEN):
                ff[i] = Float32(fused_bf16[i])

            print("  baseline vs fused (cp=" + String(num_cp) + "):")
            report("", bf, ff, HIDDEN)
        print()
