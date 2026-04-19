"""Validate two-sided V weight quantization.

Two-sided V applies H_128 to the OUTPUT dimension (rows) of the V weight
matrix in addition to the standard contraction-dimension FWHT. This lets
the GEMV output arrive pre-rotated, eliminating the explicit per-head FWHT
at runtime.

The test: for a [HEAD_DIM, HIDDEN] weight block (one V head), compare
reconstruction error between:
  Path A (single-sided): per-row i8 quantize -> dequant
  Path B (two-sided):    H_128 on rows -> per-row i8 quantize -> dequant -> H_128 on rows

By MSE invariance (Parseval), Frobenius reconstruction error measured in
either domain is the same. But per-row absmax changes because H mixes rows,
so the actual quantization (which rows get which scales) differs. The test
measures whether this difference helps, hurts, or is neutral.

Run: pixi run mojo -I . test_two_sided_v.mojo
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from simd_math import sqrt


comptime HEAD_DIM = 128
comptime HIDDEN = 3072
comptime FWHT_BLK = 128
comptime NUM_ELEMENTS = HEAD_DIM * HIDDEN


struct XorShift64:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def next_f32(mut self) -> Float32:
        return Float32(Int64(self.next() >> 40)) / Float32(1 << 24) - 0.5


def fill_random(buf: UnsafePointer[Float32, MutAnyOrigin], count: Int, mut rng: XorShift64):
    for i in range(count):
        (buf + i).store(rng.next_f32())


def copy_f32(dst: UnsafePointer[Float32, MutAnyOrigin],
             src: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= n:
        (dst + k).store((src + k).load[width=width]())
        k += width


def cosine_similarity(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var dot_ab = SIMD[DType.float32, width](0)
    var dot_aa = SIMD[DType.float32, width](0)
    var dot_bb = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= n:
        var va = (a + k).load[width=width]()
        var vb = (b + k).load[width=width]()
        dot_ab = va.fma(vb, dot_ab)
        dot_aa = va.fma(va, dot_aa)
        dot_bb = vb.fma(vb, dot_bb)
        k += width
    var dab = dot_ab.reduce_add()
    var daa = dot_aa.reduce_add()
    var dbb = dot_bb.reduce_add()
    return dab / sqrt[DType.float32, 1](daa * dbb)


def frobenius_mse(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= n:
        var d = (a + k).load[width=width]() - (b + k).load[width=width]()
        acc = d.fma(d, acc)
        k += width
    return acc.reduce_add() / Float32(n)


def row_absmax_stats(
    w: UnsafePointer[Float32, MutAnyOrigin], rows: Int, cols: Int,
) -> Tuple[Float32, Float32, Float32]:
    var mn = Float32(1e30)
    var mx = Float32(0)
    var sm = Float32(0)
    for r in range(rows):
        var row_mx = Float32(0)
        for c in range(cols):
            var v = (w + r * cols + c).load().__abs__()
            if v > row_mx:
                row_mx = v
        if row_mx < mn:
            mn = row_mx
        if row_mx > mx:
            mx = row_mx
        sm += row_mx
    return (mn, mx, sm / Float32(rows))


def quantize_dequant_rows(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
    qi_buf: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
):
    for r in range(rows):
        var row_src = src + r * cols
        var row_qi = qi_buf + r * cols
        var absmax = absmax_quantize_i8[HIDDEN](row_src, row_qi)
        var inv_scale = absmax / Float32(127.0)
        for c in range(cols):
            (dst + r * cols + c).store(Float32((row_qi + c).load()) * inv_scale)


def fwht_columns(buf: UnsafePointer[Float32, MutAnyOrigin], rows: Int, cols: Int):
    """Apply FWHT along the row (output) dimension, column by column.

    For each column c, gather buf[r][c] for r in 0..rows into a contiguous
    scratch buffer, apply FWHT(HEAD_DIM), scatter back.
    """
    var scratch = alloc[Float32](rows)
    for c in range(cols):
        for r in range(rows):
            (scratch + r).store((buf + r * cols + c).load())
        fwht_block[HEAD_DIM](scratch)
        for r in range(rows):
            (buf + r * cols + c).store((scratch + r).load())


def main():
    print("=== Two-sided V quantization validation ===")
    print("HEAD_DIM =", HEAD_DIM, " HIDDEN =", HIDDEN)
    print()

    var rng = XorShift64(42)

    var w_ref = alloc[Float32](NUM_ELEMENTS)
    var w_work = alloc[Float32](NUM_ELEMENTS)
    var recon = alloc[Float32](NUM_ELEMENTS)
    var qi_buf = alloc[Scalar[DType.int8]](NUM_ELEMENTS)

    # ── Build reference: W after contraction-dim FWHT (shared baseline) ──
    fill_random(w_ref, NUM_ELEMENTS, rng)
    for r in range(HEAD_DIM):
        var row = w_ref + r * HIDDEN
        var col = 0
        while col + FWHT_BLK <= HIDDEN:
            fwht_block[FWHT_BLK](row + col)
            col += FWHT_BLK

    # ── Path A: single-sided (current approach) ─────────────────────────
    copy_f32(w_work, w_ref, NUM_ELEMENTS)

    var ss_stats = row_absmax_stats(w_work, HEAD_DIM, HIDDEN)
    print("Single-sided row absmax: min =", ss_stats[0],
          " max =", ss_stats[1], " mean =", ss_stats[2],
          " ratio =", ss_stats[1] / ss_stats[0])

    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    var cos_ss = cosine_similarity(w_ref, recon, NUM_ELEMENTS)
    var mse_ss = frobenius_mse(w_ref, recon, NUM_ELEMENTS)

    # ── Path B: two-sided (proposed) ────────────────────────────────────
    copy_f32(w_work, w_ref, NUM_ELEMENTS)

    fwht_columns(w_work, HEAD_DIM, HIDDEN)

    var ts_stats = row_absmax_stats(w_work, HEAD_DIM, HIDDEN)
    print("Two-sided row absmax:   min =", ts_stats[0],
          " max =", ts_stats[1], " mean =", ts_stats[2],
          " ratio =", ts_stats[1] / ts_stats[0])

    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    fwht_columns(recon, HEAD_DIM, HIDDEN)

    var cos_ts = cosine_similarity(w_ref, recon, NUM_ELEMENTS)
    var mse_ts = frobenius_mse(w_ref, recon, NUM_ELEMENTS)

    print()
    print("Reconstruction vs reference (contraction-dim rotated W):")
    print("  Single-sided: cos =", cos_ss, " mse =", mse_ss)
    print("  Two-sided:    cos =", cos_ts, " mse =", mse_ts)
    print("  MSE ratio (two-sided / single-sided):", mse_ts / mse_ss)
    print()

    # ── Per-row error breakdown ─────────────────────────────────────────
    copy_f32(w_work, w_ref, NUM_ELEMENTS)
    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    var ss_better = 0
    var ts_better = 0
    var ss_total_sq = Float32(0)

    for r in range(HEAD_DIM):
        var row_sq = Float32(0)
        for c in range(HIDDEN):
            var d = (w_ref + r * HIDDEN + c).load() - (recon + r * HIDDEN + c).load()
            row_sq += d * d
        ss_total_sq += row_sq

    copy_f32(w_work, w_ref, NUM_ELEMENTS)
    fwht_columns(w_work, HEAD_DIM, HIDDEN)
    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    fwht_columns(recon, HEAD_DIM, HIDDEN)
    var ts_total_sq = Float32(0)

    for r in range(HEAD_DIM):
        var ss_row_sq = Float32(0)
        var ts_row_sq = Float32(0)
        for c in range(HIDDEN):
            var idx = r * HIDDEN + c
            var orig = (w_ref + idx).load()
            # recon is already de-rotated back to single-sided domain
            var d_ts = orig - (recon + idx).load()
            ts_row_sq += d_ts * d_ts
        ts_total_sq += ts_row_sq

    print("Total squared error:")
    print("  Single-sided:", ss_total_sq)
    print("  Two-sided:   ", ts_total_sq)
    if ss_total_sq > 0:
        print("  Ratio (ts/ss):", ts_total_sq / ss_total_sq)

    # ── End-to-end: GEMV output → cache write simulation ───────────────
    print()
    print("--- End-to-end GEMV → cache simulation ---")
    var act = alloc[Float32](HIDDEN)
    fill_random(act, HIDDEN, rng)

    var v_exact = alloc[Float32](HEAD_DIM)
    var v_path = alloc[Float32](HEAD_DIM)
    var v_qi = alloc[Scalar[DType.int8]](HEAD_DIM)

    # Exact: W_ref · act → FWHT → ideal rotated V for cache
    for r in range(HEAD_DIM):
        var dot = Float32(0)
        for c in range(HIDDEN):
            dot += (w_ref + r * HIDDEN + c).load() * (act + c).load()
        (v_exact + r).store(dot)
    fwht_block[HEAD_DIM](v_exact)

    # Path A: single-sided quantized W · act → FWHT → quantize → dequant
    copy_f32(w_work, w_ref, NUM_ELEMENTS)
    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    for r in range(HEAD_DIM):
        var dot = Float32(0)
        for c in range(HIDDEN):
            dot += (recon + r * HIDDEN + c).load() * (act + c).load()
        (v_path + r).store(dot)
    fwht_block[HEAD_DIM](v_path)
    var absmax_a = absmax_quantize_i8[HEAD_DIM](v_path, v_qi)
    var inv_a = absmax_a / Float32(127.0)
    for i in range(HEAD_DIM):
        (v_path + i).store(Float32((v_qi + i).load()) * inv_a)
    var cache_cos_ss = cosine_similarity(v_exact, v_path, HEAD_DIM)

    # Path B: two-sided quantized W · act → (already rotated) → quantize → dequant
    copy_f32(w_work, w_ref, NUM_ELEMENTS)
    fwht_columns(w_work, HEAD_DIM, HIDDEN)
    quantize_dequant_rows(w_work, recon, HEAD_DIM, HIDDEN, qi_buf)
    for r in range(HEAD_DIM):
        var dot = Float32(0)
        for c in range(HIDDEN):
            dot += (recon + r * HIDDEN + c).load() * (act + c).load()
        (v_path + r).store(dot)
    # Output already in Hadamard domain — quantize directly for cache
    var absmax_b = absmax_quantize_i8[HEAD_DIM](v_path, v_qi)
    var inv_b = absmax_b / Float32(127.0)
    for i in range(HEAD_DIM):
        (v_path + i).store(Float32((v_qi + i).load()) * inv_b)
    var cache_cos_ts = cosine_similarity(v_exact, v_path, HEAD_DIM)

    print("V cache cosine similarity vs exact rotated V:")
    print("  Single-sided (GEMV → FWHT → quant):  cos =", cache_cos_ss)
    print("  Two-sided    (GEMV → quant directly): cos =", cache_cos_ts)
