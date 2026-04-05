"""Scale factor proof — per-head Frobenius analysis of K/V/gate/up projections.

For a weight matrix W' in R^{M x K} with isotropic input x (||x|| = sqrt(d)):
  E[y_i^2] = sum_k W'[i,k]^2     (when d = K, which holds for all projections)

Per-head RMS = ||W'_head||_F / sqrt(d_k), and after FWHT the absmax concentrates
at RMS * C(d_k).

This test computes per-head Frobenius norms from the actual checkpoint, derives
the correct scale from those norms, and compares against both:
  S_current   = ||W'||_F / sqrt(HIDDEN) * C(d_k)    (what the code uses)
  S_corrected = ||W'||_F / sqrt(M)      * C(d_k)    (correct for M != HIDDEN)

If the corrected formula tracks the empirical per-head data and the current
formula doesn't, the scale bug is confirmed.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import size_of

from safetensors.parser import parse_safetensors_header, HEADER_LEN_BYTES
from linux.io_uring import IoRing, ReadOp, ReadMode
from modeling.smollm2_butterquant_tp import SmolLM2Config, concentration_constant, FWHT_BLOCK

comptime C = SmolLM2Config
comptime MODEL_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"


# =============================================================================
# Simple file reader
# =============================================================================


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


# =============================================================================
# Per-head Frobenius norm from quantized weights
# =============================================================================


def per_head_frob_sq(
    w_i8: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int, head_dim: Int,
    results: UnsafePointer[Float64, MutAnyOrigin],
):
    """Compute ||W'_head||_F^2 for each head.

    results must have at least rows/head_dim elements.
    """
    var wp = w_i8.bitcast[Scalar[DType.int8]]()
    var sp = scale.bitcast[Float32]()
    var num_heads = rows // head_dim

    for g in range(num_heads):
        var head_frob = Float64(0)
        for i in range(head_dim):
            var n = g * head_dim + i
            var row_sq = Float64(0)
            for k in range(cols):
                var v = Float64(wp[n * cols + k])
                row_sq += v * v
            var s = Float64(sp[n])
            head_frob += s * s * row_sq
        results[g] = head_frob


def full_frob_sq(
    w_i8: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
) -> Float64:
    """Compute ||W'||_F^2 for the full matrix."""
    var wp = w_i8.bitcast[Scalar[DType.int8]]()
    var sp = scale.bitcast[Float32]()
    var total = Float64(0)
    for n in range(rows):
        var row_sq = Float64(0)
        for k in range(cols):
            var v = Float64(wp[n * cols + k])
            row_sq += v * v
        var s = Float64(sp[n])
        total += s * s * row_sq
    return total


# =============================================================================
# Main
# =============================================================================


def main():
    var path = Path(MODEL_PATH)

    # Parse header
    var header_opt = parse_safetensors_header(path)
    if not header_opt:
        print("failed to parse safetensors header")
        return
    var header = header_opt.take()

    # Setup I/O
    var reader = FileReader(header.data_offset)
    var paths = List[Path]()
    paths.append(path)
    try:
        _ = reader.ring.register_files[ReadMode](paths)
    except:
        print("failed to register file")
        return

    # Allocate read buffers (largest weight is Q: 576*576 = 331776 bytes)
    comptime MAX_WEIGHT_BYTES = C.HIDDEN * C.HIDDEN
    comptime MAX_SCALE_BYTES = C.HIDDEN * size_of[Float32]()
    var w_buf = alloc[UInt8](MAX_WEIGHT_BYTES)
    var s_buf = alloc[UInt8](MAX_SCALE_BYTES)

    var cn = concentration_constant[FWHT_BLOCK]()
    var sqrt_dk = Float64(C.HEAD_DIM).__pow__(0.5)

    print("SmolLM2 ButterQuant Scale Factor Analysis")
    print("==========================================")
    print("HIDDEN =", C.HIDDEN, " KV_HIDDEN =", C.KV_HIDDEN,
          " INTERMEDIATE =", C.INTERMEDIATE)
    print("HEAD_DIM =", C.HEAD_DIM, " NUM_HEADS =", C.NUM_HEADS,
          " NUM_KV_HEADS =", C.NUM_KV_HEADS)
    print("GQA_FACTOR =", C.GQA_FACTOR)
    print("C(", FWHT_BLOCK, ") =", cn)
    print()

    # --- Per-projection analysis ---

    # We'll track max/mean head sigma across all layers for summary stats
    var q_clip_layers = 0
    var k_clip_layers = 0
    var v_clip_layers = 0

    print("Legend:")
    print("  S_cur     = ||W'||_F / sqrt(HIDDEN) * C(n)       [current code]")
    print("  S_fix     = ||W'||_F / sqrt(M) * C(n)            [corrected]")
    print("  max_head  = max over heads of (||W'_head||_F / sqrt(d_k) * C(n))")
    print("  ratio     = max_head / S_cur   (>1 means current scale clips)")
    print()

    var head_results = alloc[Float64](C.NUM_HEADS)

    for layer in range(C.NUM_LAYERS):
        var prefix = "model.layers." + String(layer) + "."

        # --- Q projection (M = HIDDEN, should be correct) ---
        var q_name = prefix + "self_attn.q_proj.weight"
        var qs_name = q_name + "_scale"
        var q_meta = header.tensors.get(q_name).value().copy()
        var qs_meta = header.tensors.get(qs_name).value().copy()
        if not reader.read(q_meta.start, w_buf, q_meta.byte_size()):
            print("failed to read", q_name); return
        if not reader.read(qs_meta.start, s_buf, qs_meta.byte_size()):
            print("failed to read", qs_name); return

        var q_frob = full_frob_sq(w_buf, s_buf, C.HIDDEN, C.HIDDEN).__pow__(0.5)
        per_head_frob_sq(w_buf, s_buf, C.HIDDEN, C.HIDDEN, C.HEAD_DIM, head_results)
        var q_max_sigma = Float64(0)
        for g in range(C.NUM_HEADS):
            var sigma = head_results[g].__pow__(0.5) / sqrt_dk
            if sigma > q_max_sigma:
                q_max_sigma = sigma
        var q_s_cur = q_frob / Float64(C.HIDDEN).__pow__(0.5) * cn
        var q_s_fix = q_s_cur  # M = HIDDEN, no change
        var q_max_pred = q_max_sigma * cn
        var q_ratio = q_max_pred / q_s_cur

        # --- K projection (M = KV_HIDDEN, this is the bug) ---
        var k_name = prefix + "self_attn.k_proj.weight"
        var ks_name = k_name + "_scale"
        var k_meta = header.tensors.get(k_name).value().copy()
        var ks_meta = header.tensors.get(ks_name).value().copy()
        if not reader.read(k_meta.start, w_buf, k_meta.byte_size()):
            print("failed to read", k_name); return
        if not reader.read(ks_meta.start, s_buf, ks_meta.byte_size()):
            print("failed to read", ks_name); return

        var k_frob = full_frob_sq(w_buf, s_buf, C.KV_HIDDEN, C.HIDDEN).__pow__(0.5)
        per_head_frob_sq(w_buf, s_buf, C.KV_HIDDEN, C.HIDDEN, C.HEAD_DIM, head_results)
        var k_max_sigma = Float64(0)
        for g in range(C.NUM_KV_HEADS):
            var sigma = head_results[g].__pow__(0.5) / sqrt_dk
            if sigma > k_max_sigma:
                k_max_sigma = sigma
        var k_s_cur = k_frob / Float64(C.HIDDEN).__pow__(0.5) * cn
        var k_s_fix = k_frob / Float64(C.KV_HIDDEN).__pow__(0.5) * cn
        var k_max_pred = k_max_sigma * cn
        var k_ratio = k_max_pred / k_s_cur

        # --- V projection (M = KV_HIDDEN, same bug) ---
        var v_name = prefix + "self_attn.v_proj.weight"
        var vs_name = v_name + "_scale"
        var v_meta = header.tensors.get(v_name).value().copy()
        var vs_meta = header.tensors.get(vs_name).value().copy()
        if not reader.read(v_meta.start, w_buf, v_meta.byte_size()):
            print("failed to read", v_name); return
        if not reader.read(vs_meta.start, s_buf, vs_meta.byte_size()):
            print("failed to read", vs_name); return

        var v_frob = full_frob_sq(w_buf, s_buf, C.KV_HIDDEN, C.HIDDEN).__pow__(0.5)
        per_head_frob_sq(w_buf, s_buf, C.KV_HIDDEN, C.HIDDEN, C.HEAD_DIM, head_results)
        var v_max_sigma = Float64(0)
        for g in range(C.NUM_KV_HEADS):
            var sigma = head_results[g].__pow__(0.5) / sqrt_dk
            if sigma > v_max_sigma:
                v_max_sigma = sigma
        var v_s_cur = v_frob / Float64(C.HIDDEN).__pow__(0.5) * cn
        var v_s_fix = v_frob / Float64(C.KV_HIDDEN).__pow__(0.5) * cn
        var v_max_pred = v_max_sigma * cn
        var v_ratio = v_max_pred / v_s_cur

        if q_ratio > 1.0:
            q_clip_layers += 1
        if k_ratio > 1.0:
            k_clip_layers += 1
        if v_ratio > 1.0:
            v_clip_layers += 1

        print("layer", layer)
        print("  Q: S_cur=", q_s_cur, " S_fix=", q_s_fix,
              " max_head=", q_max_pred, " ratio=", q_ratio)
        print("  K: S_cur=", k_s_cur, " S_fix=", k_s_fix,
              " max_head=", k_max_pred, " ratio=", k_ratio)
        print("  V: S_cur=", v_s_cur, " S_fix=", v_s_fix,
              " max_head=", v_max_pred, " ratio=", v_ratio)

    print()
    print("=== SUMMARY ===")
    print("Layers where max per-head absmax > S_current (ratio > 1 = clipping):")
    print("  Q:", q_clip_layers, "/", C.NUM_LAYERS)
    print("  K:", k_clip_layers, "/", C.NUM_LAYERS)
    print("  V:", v_clip_layers, "/", C.NUM_LAYERS)
    print()
    print("Expected: Q should rarely clip (M=HIDDEN, formula is correct).")
    print("Expected: K and V should frequently clip (M=KV_HIDDEN, formula uses wrong denom).")
    print("If K/V clip in most layers, the scale is confirmed too small by sqrt(GQA_FACTOR) =",
          Float64(C.GQA_FACTOR).__pow__(0.5))

    head_results.free()
    w_buf.free()
    s_buf.free()
