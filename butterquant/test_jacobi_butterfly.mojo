"""Butterfly-Jacobi sweep experiment.

Tests whether butterfly-constrained Jacobi eigenvalue iteration can find a
good quantization rotation from the weight Gram matrix alone — no calibration
data, no gradient descent.

Algorithm:
  1. Form G = W'^T W' for a projection weight (e.g. Q, layer 0)
  2. Apply butterfly-Jacobi sweeps: at each level k, for each pair (i, i^(2^k)),
     compute the closed-form Jacobi angle that zeros G[p,q], apply the Givens
     rotation to G, record the angle in a butterfly factor
  3. Track off-diagonal Frobenius norm per sweep (convergence metric)
  4. Apply the accumulated butterfly rotation to real Q/K vectors and measure
     quantization round-trip quality vs fixed Hadamard

The key question: does the off-diagonal norm drop fast enough in a few sweeps
to be useful for quantization?
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.pathlib import Path

from safetensors.parser import parse_safetensors_header, HEADER_LEN_BYTES
from linux.io_uring import IoRing, ReadOp, ReadMode
from modeling.smollm2_butterquant_tp import (
    SmolLM2ButterQuant, SmolLM2Config, concentration_constant, FWHT_BLOCK,
)
from tokenizer import load_tokenizer
from experimental.amx import init_intel_amx
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from simd_math import roundeven, sqrt

comptime C = SmolLM2Config
comptime TOKENIZER_PATH = "checkpoints/SmolLM2/tokenizer.json"
comptime QUANT_PATH = "quantized_models/SmolLM2-ButterQuant/model.safetensors"
comptime HD = C.HEAD_DIM  # 64
comptime LOG_HD = 6  # log2(64)


# =============================================================================
# Dense symmetric matrix operations (G is HD x HD, small enough for dense)
# =============================================================================


struct DenseMat:
    """HD x HD dense f64 matrix, heap-allocated."""
    var data: UnsafePointer[Float64, MutAnyOrigin]

    def __init__(out self):
        self.data = alloc[Float64](HD * HD)
        for i in range(HD * HD):
            self.data[i] = Float64(0)

    def __getitem__(self, r: Int, c: Int) -> Float64:
        return self.data[r * HD + c]

    def __setitem__(mut self, r: Int, c: Int, v: Float64):
        self.data[r * HD + c] = v

    def off_diag_norm_sq(self) -> Float64:
        """Sum of squares of off-diagonal entries."""
        var total = Float64(0)
        for i in range(HD):
            for j in range(HD):
                if i != j:
                    total += self[i, j] * self[i, j]
        return total

    def free(mut self):
        self.data.free()


# =============================================================================
# Butterfly-Jacobi sweep
# =============================================================================


struct ButterflyCS:
    """Stores (cos, sin) pairs for one butterfly sweep (LOG_HD levels, HD/2 pairs each)."""
    var cs: UnsafePointer[Float64, MutAnyOrigin]  # interleaved [c0, s0, c1, s1, ...]

    def __init__(out self):
        self.cs = alloc[Float64](LOG_HD * (HD // 2) * 2)
        for i in range(LOG_HD * (HD // 2)):
            self.cs[i * 2] = Float64(1)      # cos = 1
            self.cs[i * 2 + 1] = Float64(0)  # sin = 0

    def get_c(self, level: Int, pair: Int) -> Float64:
        return self.cs[(level * (HD // 2) + pair) * 2]

    def get_s(self, level: Int, pair: Int) -> Float64:
        return self.cs[(level * (HD // 2) + pair) * 2 + 1]

    def set(mut self, level: Int, pair: Int, c: Float64, s: Float64):
        self.cs[(level * (HD // 2) + pair) * 2] = c
        self.cs[(level * (HD // 2) + pair) * 2 + 1] = s

    def free(mut self):
        self.cs.free()


def jacobi_cs(gpp: Float64, gqq: Float64, gpq: Float64) -> Tuple[Float64, Float64]:
    """Compute (cos θ, sin θ) for the Jacobi rotation zeroing G[p,q].

    Direct formula from the 2x2 symmetric eigenproblem, no trig needed.
    """
    if gpq.__abs__() < Float64(1e-30):
        return (Float64(1), Float64(0))
    var tau = (gqq - gpp) / (Float64(2) * gpq)
    # t = sign(tau) / (|tau| + sqrt(1 + tau^2))  — the smaller root for stability
    var t: Float64
    if tau >= Float64(0):
        t = Float64(1) / (tau + (Float64(1) + tau * tau).__pow__(0.5))
    else:
        t = Float64(-1) / (-tau + (Float64(1) + tau * tau).__pow__(0.5))
    var c = Float64(1) / (Float64(1) + t * t).__pow__(0.5)
    var s = t * c
    return (c, s)


def apply_givens_to_gram(mut g: DenseMat, p: Int, q: Int, c: Float64, s: Float64):
    """Apply J(p,q,θ)^T G J(p,q,θ) in-place. Only modifies rows/cols p,q."""

    # First: G <- J^T G (rotate rows p, q)
    for j in range(HD):
        var gp = g[p, j]
        var gq = g[q, j]
        g[p, j] = c * gp + s * gq
        g[q, j] = -s * gp + c * gq

    # Then: G <- G J (rotate cols p, q)
    for i in range(HD):
        var gip = g[i, p]
        var giq = g[i, q]
        g[i, p] = c * gip + s * giq
        g[i, q] = -s * gip + c * giq


def butterfly_jacobi_sweep(mut g: DenseMat, mut sweep_cs: ButterflyCS):
    """One butterfly-Jacobi sweep: iterate over LOG_HD levels, HD/2 pairs each.

    Level k pairs: (i, i ^ (1 << k)) for i where bit k is 0.
    """
    for level in range(LOG_HD):
        var stride = 1 << level
        var pair_idx = 0
        for i in range(HD):
            if (i >> level) & 1 == 0:
                var j = i ^ stride
                if j < HD and j > i:
                    var result = jacobi_cs(g[i, i], g[j, j], g[i, j])
                    var c = result[0]
                    var s = result[1]
                    apply_givens_to_gram(g, i, j, c, s)
                    sweep_cs.set(level, pair_idx, c, s)
                    pair_idx += 1


# =============================================================================
# Apply accumulated butterfly rotation to a vector
# =============================================================================


def apply_butterfly_rotation(
    buf: UnsafePointer[Float64, MutAnyOrigin],
    num_sweeps: Int,
    all_cs: UnsafePointer[ButterflyCS, MutAnyOrigin],
):
    """Apply the accumulated butterfly rotation V = B^(1) B^(2) ... B^(s) to buf."""
    for sweep in range(num_sweeps):
        var cs_ptr = all_cs[sweep].cs
        for level in range(LOG_HD):
            var stride = 1 << level
            var pair_idx = 0
            for i in range(HD):
                if (i >> level) & 1 == 0:
                    var j = i ^ stride
                    if j < HD and j > i:
                        var idx = (level * (HD // 2) + pair_idx) * 2
                        var c = cs_ptr[idx]
                        var s = cs_ptr[idx + 1]
                        var vi = buf[i]
                        var vj = buf[j]
                        buf[i] = c * vi + s * vj
                        buf[j] = -s * vi + c * vj
                        pair_idx += 1


def apply_inverse_butterfly_rotation(
    buf: UnsafePointer[Float64, MutAnyOrigin],
    num_sweeps: Int,
    all_cs: UnsafePointer[ButterflyCS, MutAnyOrigin],
):
    """Apply V^T (inverse rotation): reverse order of sweeps and levels, negate sin."""
    for sweep in range(num_sweeps - 1, -1, -1):
        var cs_ptr = all_cs[sweep].cs
        for level in range(LOG_HD - 1, -1, -1):
            var stride = 1 << level
            var pair_idx = 0
            for i in range(HD):
                if (i >> level) & 1 == 0:
                    var j = i ^ stride
                    if j < HD and j > i:
                        var idx = (level * (HD // 2) + pair_idx) * 2
                        var c = cs_ptr[idx]
                        var s = -cs_ptr[idx + 1]  # negate sin for inverse
                        var vi = buf[i]
                        var vj = buf[j]
                        buf[i] = c * vi + s * vj
                        buf[j] = -s * vi + c * vj
                        pair_idx += 1


# =============================================================================
# Quantize round-trip using butterfly rotation instead of FWHT
# =============================================================================


def jacobi_quantize_roundtrip(
    buf_f64: UnsafePointer[Float64, MutAnyOrigin],
    num_sweeps: Int,
    all_cs: UnsafePointer[ButterflyCS, MutAnyOrigin],
):
    """In-place: butterfly_rotate → quantize(absmax) → dequant → inverse_rotate."""
    # Forward rotation
    apply_butterfly_rotation(buf_f64, num_sweeps, all_cs)

    # Dynamic quantize/dequant (per-head absmax, same as our proven-good approach)
    var absmax = Float64(0)
    for i in range(HD):
        var a = buf_f64[i]
        if a < Float64(0):
            a = -a
        if a > absmax:
            absmax = a
    if absmax < Float64(1e-30):
        absmax = Float64(1e-30)

    var quant_inv = Float64(127) / absmax
    var dequant_sc = absmax / Float64(127)
    for i in range(HD):
        var v = buf_f64[i]
        var qi = v * quant_inv
        # Round to nearest
        if qi > Float64(0):
            qi = Float64(Int(qi + Float64(0.5)))
        else:
            qi = Float64(Int(qi - Float64(0.5)))
        # Clamp
        if qi < Float64(-128):
            qi = Float64(-128)
        if qi > Float64(127):
            qi = Float64(127)
        buf_f64[i] = qi * dequant_sc

    # Inverse rotation
    apply_inverse_butterfly_rotation(buf_f64, num_sweeps, all_cs)


# =============================================================================
# Comparison helpers
# =============================================================================


def compare_f64(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    count: Int,
) -> Tuple[Float64, Float64]:
    """Returns (cos_sim, rel_l2)."""
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    var dsq = Float64(0)
    for i in range(count):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
        dsq += (a[i] - b[i]) * (a[i] - b[i])
    var cos = dot / (na.__pow__(0.5) * nb.__pow__(0.5) + Float64(1e-30))
    var rel = dsq.__pow__(0.5) / (na.__pow__(0.5) + Float64(1e-30))
    return (cos, rel)


# =============================================================================
# Build Gram matrix from quantized weight + row scales
# =============================================================================


struct FileReader:
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


def build_head_gram(
    w_i8: UnsafePointer[UInt8, MutAnyOrigin],
    scale: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
    head: Int, head_dim: Int,
    mut gram: DenseMat,
):
    """Build G = W_head'^T W_head' for one head of the weight matrix.

    W_head' is the submatrix rows [head*head_dim : (head+1)*head_dim] of the
    dequantized weight W'[n,k] = W_i8[n,k] * scale[n].
    """
    var wp = w_i8.bitcast[Scalar[DType.int8]]()
    var sp = scale.bitcast[Float32]()
    var h_start = head * head_dim

    # G[j1, j2] = sum_{i in head} W'[i, j1] * W'[i, j2]
    #           = sum_{i} (w_i8[i,j1]*s[i]) * (w_i8[i,j2]*s[i])
    #           = sum_{i} s[i]^2 * w_i8[i,j1] * w_i8[i,j2]
    for j1 in range(head_dim):
        for j2 in range(j1, head_dim):
            var val = Float64(0)
            for i in range(head_dim):
                var n = h_start + i
                var s = Float64(sp[n])
                var w1 = Float64(wp[n * cols + j1])
                var w2 = Float64(wp[n * cols + j2])
                val += s * s * w1 * w2
            gram[j1, j2] = val
            gram[j2, j1] = val


# =============================================================================
# Main
# =============================================================================


def main():
    _ = init_intel_amx()

    # --- Load model for Q/K/V vectors ---
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("tokenizer load failed"); return
    var tok = tok_opt.take()

    var prompt = "The quick brown fox jumps over the lazy dog. " * 5
    var token_ids = tok.encode(prompt)
    var seq_len = len(token_ids)

    print("loading quant model...")
    var model_opt = SmolLM2ButterQuant[1].load(Path(QUANT_PATH))
    if not model_opt:
        print("model load failed"); return
    var model = model_opt.take()

    var tp = model.token_buffer()
    for i in range(seq_len):
        tp[i] = Scalar[DType.int32](token_ids[i])
    model.debug_embed(Int(tp), seq_len)

    # Get bf16 Q from layer 0
    var q_bf16 = alloc[Scalar[DType.bfloat16]](C.HIDDEN)
    var k_bf16 = alloc[Scalar[DType.bfloat16]](C.KV_HIDDEN)
    var v_bf16 = alloc[Scalar[DType.bfloat16]](C.KV_HIDDEN)
    model.debug_qkv(0, seq_len, q_bf16, k_bf16, v_bf16)

    # RoPE tables for last token
    var rv = model.rank(0)
    comptime HALF = HD // 2
    var last_pos = seq_len - 1
    var cos_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_cos().ptr) + last_pos * HALF
    var sin_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rv.rope_sin().ptr) + last_pos * HALF

    # --- Load raw weight for Gram matrix ---
    var path = Path(QUANT_PATH)
    var header_opt = parse_safetensors_header(path)
    if not header_opt:
        print("header parse failed"); return
    var header = header_opt.take()

    var reader = FileReader(header.data_offset)
    var paths = List[Path]()
    paths.append(path)
    try:
        _ = reader.ring.register_files[ReadMode](paths)
    except:
        print("register failed"); return

    var q_name = "model.layers.0.self_attn.q_proj.weight"
    var qs_name = q_name + "_scale"
    var q_meta = header.tensors.get(q_name).value().copy()
    var qs_meta = header.tensors.get(qs_name).value().copy()
    var w_buf = alloc[UInt8](q_meta.byte_size())
    var s_buf = alloc[UInt8](qs_meta.byte_size())
    if not reader.read(q_meta.start, w_buf, q_meta.byte_size()):
        print("read weight failed"); return
    if not reader.read(qs_meta.start, s_buf, qs_meta.byte_size()):
        print("read scale failed"); return

    print("tokens:", seq_len)
    print()

    # === Experiment 1: Convergence of butterfly-Jacobi sweeps ===
    print("=== Butterfly-Jacobi convergence on Q head 0 Gram matrix ===")

    var gram = DenseMat()
    build_head_gram(w_buf, s_buf, C.HIDDEN, C.HIDDEN, 0, HD, gram)
    var initial_offdiag = gram.off_diag_norm_sq().__pow__(0.5)
    var diag_norm = Float64(0)
    for i in range(HD):
        diag_norm += gram[i, i] * gram[i, i]
    diag_norm = diag_norm.__pow__(0.5)
    print("initial off-diag norm:", initial_offdiag, " diag norm:", diag_norm,
          " ratio:", initial_offdiag / (diag_norm + Float64(1e-30)))

    comptime MAX_SWEEPS = 10
    var all_angles = alloc[ButterflyCS](MAX_SWEEPS)
    for s in range(MAX_SWEEPS):
        all_angles[s] = ButterflyCS()

    for sweep in range(MAX_SWEEPS):
        butterfly_jacobi_sweep(gram, all_angles[sweep])
        var offdiag = gram.off_diag_norm_sq().__pow__(0.5)
        var reduction = offdiag / initial_offdiag
        print("  sweep", sweep + 1, " off-diag norm:", offdiag,
              " reduction:", reduction)

    print()

    # === Experiment 2: Round-trip quality comparison ===
    print("=== Round-trip quantization quality on Q heads (layer 0) ===")
    print("Comparing: fixed Hadamard | dynamic Hadamard | Jacobi-butterfly (5 sweeps) | Jacobi-butterfly (10 sweeps)")

    var original = alloc[Float64](HD)
    var rt = alloc[Float64](HD)
    var rt_f32 = alloc[Float32](HD)
    comptime width = simd_width_of[DType.float32]()

    for head in range(C.NUM_HEADS):
        # Build per-head Gram and run sweeps
        var head_gram = DenseMat()
        build_head_gram(w_buf, s_buf, C.HIDDEN, C.HIDDEN, head, HD, head_gram)
        var head_angles = alloc[ButterflyCS](MAX_SWEEPS)
        for s in range(MAX_SWEEPS):
            head_angles[s] = ButterflyCS()
            butterfly_jacobi_sweep(head_gram, head_angles[s])

        # Load Q head, apply RoPE, convert to f64
        var src = q_bf16 + head * HD
        var k = 0
        while k + width <= HD:
            var v = (src + k).load[width=width]().cast[DType.float32]()
            for lane in range(width):
                original[k + lane] = Float64(v[lane])
            k += width
        # RoPE in f64
        for j in range(HALF):
            var lo = original[j]
            var hi = original[HALF + j]
            var cv = Float64(cos_ptr[j])
            var sv = Float64(sin_ptr[j])
            original[j] = lo * cv - hi * sv
            original[HALF + j] = hi * cv + lo * sv

        # --- Fixed Hadamard (current scheme, layer-wide scale) ---
        var s_q = Float64(model.layer_scales[0].q_layer_scale)
        for i in range(HD):
            rt_f32[i] = Float32(original[i])
        fwht_block[HD](rt_f32)
        var qi_f = Float32(127) / Float32(s_q)
        var dq_f = Float32(s_q) / Float32(127)
        comptime lo_v = SIMD[DType.float32, width](-128.0)
        comptime hi_v = SIMD[DType.float32, width](127.0)
        k = 0
        while k + width <= HD:
            var v = (rt_f32 + k).load[width=width]()
            (rt_f32 + k).store(min(max(roundeven(v * qi_f), lo_v), hi_v) * dq_f)
            k += width
        fwht_block[HD](rt_f32)
        for i in range(HD):
            rt[i] = Float64(rt_f32[i])
        var fixed_result = compare_f64(original, rt, HD)

        # --- Dynamic Hadamard (per-head absmax) ---
        for i in range(HD):
            rt_f32[i] = Float32(original[i])
        fwht_block[HD](rt_f32)
        var amax_d = Float32(0)
        for i in range(HD):
            var a = rt_f32[i]
            if a < Float32(0):
                a = -a
            if a > amax_d:
                amax_d = a
        if amax_d < Float32(1e-10):
            amax_d = Float32(1e-10)
        var qi_d = Float32(127) / amax_d
        var dq_d = amax_d / Float32(127)
        k = 0
        while k + width <= HD:
            var v = (rt_f32 + k).load[width=width]()
            (rt_f32 + k).store(min(max(roundeven(v * qi_d), lo_v), hi_v) * dq_d)
            k += width
        fwht_block[HD](rt_f32)
        for i in range(HD):
            rt[i] = Float64(rt_f32[i])
        var dynamic_result = compare_f64(original, rt, HD)

        # --- Jacobi-butterfly 5 sweeps ---
        memcpy(dest=rt, src=original, count=HD)
        jacobi_quantize_roundtrip(rt, 5, head_angles)
        var jacobi5_result = compare_f64(original, rt, HD)

        # --- Jacobi-butterfly 10 sweeps ---
        memcpy(dest=rt, src=original, count=HD)
        jacobi_quantize_roundtrip(rt, 10, head_angles)
        var jacobi10_result = compare_f64(original, rt, HD)

        print("  head", head,
              " fixed:", fixed_result[0],
              " dynamic:", dynamic_result[0],
              " jacobi5:", jacobi5_result[0],
              " jacobi10:", jacobi10_result[0])

        head_gram.free()
        for s in range(MAX_SWEEPS):
            head_angles[s].free()
        head_angles.free()

    # Cleanup
    gram.free()
    for s in range(MAX_SWEEPS):
        all_angles[s].free()
    all_angles.free()
    original.free()
    rt.free()
    rt_f32.free()
    w_buf.free()
    s_buf.free()
    q_bf16.free()
    k_bf16.free()
    v_bf16.free()
    _ = model
