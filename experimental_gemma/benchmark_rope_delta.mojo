"""Benchmark original dense full-RoPE vs compact full-RoPE.

This compares the old Gemma full-attention RoPE behavior:
  - dense [max_pos, 256] half-dim cos/sin tables
  - generic full-width rope apply over all 256 half-dim pairs

against the compact implementation:
  - compact [max_pos, 64] active half-dim tables
  - partial apply over only the active 64 half-dim pairs
"""

from std.math import abs
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from modeling.model_spec import BF16, F32, Encoding, Shaped, Slot, Replicated, Bound, DynView
from kernels.kv_rotors import rope
from experimental_gemma.rope import init_full_rope_tables, apply_full_rope
from simd_math import sincos


def init_full_rope_tables_dense[CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    cos_buf: Bound[CosT], sin_buf: Bound[SinT],
) where CosT.DTYPE == DType.float32:
    """Original full-attention table shape: store all 256 half-dim entries."""
    comptime assert CosT.ROWS == SinT.ROWS, "dense rope init: cos/sin rows mismatch"
    comptime assert CosT.COLS == SinT.COLS, "dense rope init: cos/sin cols mismatch"
    comptime assert CosT.COLS == 256, "dense rope init: expected full 256 half-dim entries"

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=cos_buf.ptr)
    var sp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=sin_buf.ptr)
    comptime half = CosT.COLS
    comptime head_dim = 512
    comptime f64w = simd_width_of[DType.float64]()
    comptime theta = Float64(1000000.0)
    comptime rope_angles = 64

    for j in range(0, rope_angles, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            inv[k] = 1.0 / (theta ** (Float64(2 * (j + k)) / Float64(head_dim)))

        for pos in range(CosT.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]())

    for pos in range(CosT.ROWS):
        for j in range(rope_angles, half):
            (cp + pos * half + j).store(Scalar[DType.float32](1.0))
            (sp + pos * half + j).store(Scalar[DType.float32](0.0))


def fill_bf16(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32(-2.0) + Float32(i % 257) * Float32(0.0078125))


def copy_bf16(
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    comptime width = simd_width_of[DType.bfloat16]()
    for i in range(0, count, width):
        (dst + i).store((src + i).load[width=width]())


def checksum_f32(
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
) -> Float32:
    var sum = Float32(0)
    for i in range(count):
        sum += Float32(ptr[i])
    return sum


def max_abs_diff(
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
) -> Float64:
    var m = Float64(0)
    for i in range(count):
        var d = abs(Float64(Float32(a[i])) - Float64(Float32(b[i])))
        if d > m:
            m = d
    return m


def bench_full_rope_init[max_pos: Int]():
    print("=== full-rope init delta ===")

    comptime DenseHalf = 256
    comptime CompactHalf = 64
    comptime DenseCos = Slot[F32, Replicated, max_pos, DenseHalf, 1]
    comptime DenseSin = Slot[F32, Replicated, max_pos, DenseHalf, 1]
    comptime CompactCos = Slot[F32, Replicated, max_pos, CompactHalf, 1]
    comptime CompactSin = Slot[F32, Replicated, max_pos, CompactHalf, 1]

    var dense_cos = alloc[Scalar[DType.float32]](max_pos * DenseHalf)
    var dense_sin = alloc[Scalar[DType.float32]](max_pos * DenseHalf)
    var compact_cos = alloc[Scalar[DType.float32]](max_pos * CompactHalf)
    var compact_sin = alloc[Scalar[DType.float32]](max_pos * CompactHalf)

    var dense_cos_buf = Bound[DenseCos](Int(dense_cos))
    var dense_sin_buf = Bound[DenseSin](Int(dense_sin))
    var compact_cos_buf = Bound[CompactCos](Int(compact_cos))
    var compact_sin_buf = Bound[CompactSin](Int(compact_sin))

    var t0 = Int(perf_counter_ns())
    init_full_rope_tables_dense(dense_cos_buf, dense_sin_buf)
    var t1 = Int(perf_counter_ns())

    var t2 = Int(perf_counter_ns())
    init_full_rope_tables(compact_cos_buf, compact_sin_buf)
    var t3 = Int(perf_counter_ns())

    var dense_ms = Float64(t1 - t0) / 1_000_000.0
    var compact_ms = Float64(t3 - t2) / 1_000_000.0
    var speedup = dense_ms / compact_ms

    print("dense  table bytes=" + String(2 * max_pos * DenseHalf * 4) + " time_ms=" + String(dense_ms))
    print("compact table bytes=" + String(2 * max_pos * CompactHalf * 4) + " time_ms=" + String(compact_ms))
    print("speedup=" + String(speedup) + "x")
    print("dense sample cos[10,63]=" + String(dense_cos[10 * DenseHalf + 63]) + " sin[10,63]=" + String(dense_sin[10 * DenseHalf + 63]))
    print("compact sample cos[10,63]=" + String(compact_cos[10 * CompactHalf + 63]) + " sin[10,63]=" + String(compact_sin[10 * CompactHalf + 63]))
    print()

    dense_cos.free()
    dense_sin.free()
    compact_cos.free()
    compact_sin.free()


def bench_full_rope_apply[num_heads: Int, seq_len: Int](
    label: String,
    pos: Int,
    iters: Int,
):
    comptime HEAD_DIM = 512
    comptime DenseHalf = 256
    comptime CompactHalf = 64
    comptime MAX_POS = 4096
    comptime COLS = num_heads * HEAD_DIM
    comptime DenseCos = Slot[F32, Replicated, MAX_POS, DenseHalf, 1]
    comptime DenseSin = Slot[F32, Replicated, MAX_POS, DenseHalf, 1]
    comptime CompactCos = Slot[F32, Replicated, MAX_POS, CompactHalf, 1]
    comptime CompactSin = Slot[F32, Replicated, MAX_POS, CompactHalf, 1]
    comptime XView = Slot[BF16, Replicated, seq_len, COLS, 1]

    var dense_cos = alloc[Scalar[DType.float32]](MAX_POS * DenseHalf)
    var dense_sin = alloc[Scalar[DType.float32]](MAX_POS * DenseHalf)
    var compact_cos = alloc[Scalar[DType.float32]](MAX_POS * CompactHalf)
    var compact_sin = alloc[Scalar[DType.float32]](MAX_POS * CompactHalf)
    init_full_rope_tables_dense(Bound[DenseCos](Int(dense_cos)), Bound[DenseSin](Int(dense_sin)))
    init_full_rope_tables(Bound[CompactCos](Int(compact_cos)), Bound[CompactSin](Int(compact_sin)))

    var orig = alloc[Scalar[DType.bfloat16]](seq_len * COLS)
    var dense_x = alloc[Scalar[DType.bfloat16]](seq_len * COLS)
    var compact_x = alloc[Scalar[DType.bfloat16]](seq_len * COLS)
    fill_bf16(orig, seq_len * COLS)

    # Correctness: single application should match exactly in bf16.
    copy_bf16(dense_x, orig, seq_len * COLS)
    copy_bf16(compact_x, orig, seq_len * COLS)
    var dense_view = DynView[XView](Int(dense_x), seq_len)
    var compact_view = DynView[XView](Int(compact_x), seq_len)
    rope[HEAD_DIM, num_heads](dense_view, Bound[DenseCos](Int(dense_cos)), Bound[DenseSin](Int(dense_sin)), pos)
    apply_full_rope[num_heads](compact_view, Bound[CompactCos](Int(compact_cos)), Bound[CompactSin](Int(compact_sin)), pos)
    var diff = max_abs_diff(dense_x, compact_x, seq_len * COLS)

    # Time dense apply.
    copy_bf16(dense_x, orig, seq_len * COLS)
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        rope[HEAD_DIM, num_heads](dense_view, Bound[DenseCos](Int(dense_cos)), Bound[DenseSin](Int(dense_sin)), pos)
    var t1 = Int(perf_counter_ns())

    # Time compact apply.
    copy_bf16(compact_x, orig, seq_len * COLS)
    var t2 = Int(perf_counter_ns())
    for _ in range(iters):
        apply_full_rope[num_heads](compact_view, Bound[CompactCos](Int(compact_cos)), Bound[CompactSin](Int(compact_sin)), pos)
    var t3 = Int(perf_counter_ns())

    var dense_us = Float64(t1 - t0) / Float64(iters) / 1000.0
    var compact_us = Float64(t3 - t2) / Float64(iters) / 1000.0
    var speedup = dense_us / compact_us

    print(label)
    print("  max_abs_diff=" + String(diff))
    print("  dense_us=" + String(dense_us) + " checksum=" + String(checksum_f32(dense_x, seq_len * COLS)))
    print("  compact_us=" + String(compact_us) + " checksum=" + String(checksum_f32(compact_x, seq_len * COLS)))
    print("  speedup=" + String(speedup) + "x")

    dense_cos.free()
    dense_sin.free()
    compact_cos.free()
    compact_sin.free()
    orig.free()
    dense_x.free()
    compact_x.free()


def main():
    bench_full_rope_init[262144]()

    print("=== full-rope apply delta ===")
    bench_full_rope_apply[16, 1]("Q decode   heads=16 seq=1", 1024, 4000)
    bench_full_rope_apply[2, 1]("K decode   heads=2  seq=1", 1024, 12000)
    bench_full_rope_apply[16, 256]("Q prefill  heads=16 seq=256", 1024, 40)
    bench_full_rope_apply[2, 256]("K prefill  heads=2  seq=256", 1024, 100)
