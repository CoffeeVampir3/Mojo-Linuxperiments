"""Validate Gemma4 RoPE table initialization.

Checks:
  1. Sliding tables: theta=10000, all 128 half-dim entries have real frequencies
  2. Full tables: theta=1M, compact storage for the 64 active half-dim entries
  3. Both: cos^2 + sin^2 = 1 for all stored entries
  4. Both: position 0 has cos=1, sin=0 (no rotation at pos 0)
  5. Full compact apply rotates only the active 128 of 512 dims
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs

from modeling.model_spec import BF16, F32, Slot, Replicated, Bound, DynView
from experimental_gemma.rope import init_sliding_rope_tables, init_full_rope_tables, apply_full_rope


def test_sliding_rope():
    print("=== Sliding RoPE (theta=10000, head_dim=256) ===")

    comptime MAX_POS = 64
    comptime HALF = 128
    comptime CosSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]

    var cos_ptr = alloc[Scalar[DType.float32]](MAX_POS * HALF)
    var sin_ptr = alloc[Scalar[DType.float32]](MAX_POS * HALF)
    var cos_buf = Bound[CosSlot](Int(cos_ptr))
    var sin_buf = Bound[SinSlot](Int(sin_ptr))

    init_sliding_rope_tables(cos_buf, sin_buf)

    # Check position 0: cos=1, sin=0 for all dims
    print("  pos=0 check (all should be cos=1, sin=0):")
    var max_cos_err_p0 = Float64(0)
    var max_sin_err_p0 = Float64(0)
    for j in range(HALF):
        var ce = abs(Float64(cos_ptr[j]) - 1.0)
        var se = abs(Float64(sin_ptr[j]))
        if ce > max_cos_err_p0:
            max_cos_err_p0 = ce
        if se > max_sin_err_p0:
            max_sin_err_p0 = se
    print("    max |cos-1|=" + String(max_cos_err_p0) + "  max |sin|=" + String(max_sin_err_p0))

    # Check cos^2 + sin^2 = 1 for all entries
    print("  cos^2+sin^2 check:")
    var max_unit_err = Float64(0)
    for pos in range(MAX_POS):
        for j in range(HALF):
            var c = Float64(cos_ptr[pos * HALF + j])
            var s = Float64(sin_ptr[pos * HALF + j])
            var err = abs(c * c + s * s - 1.0)
            if err > max_unit_err:
                max_unit_err = err
    print("    max |cos^2+sin^2-1|=" + String(max_unit_err))

    # Show a few values for eyeball check
    print("  sample values (pos, dim, cos, sin):")
    for pos in range(0, MAX_POS, 16):
        for j in range(0, HALF, 32):
            var c = cos_ptr[pos * HALF + j]
            var s = sin_ptr[pos * HALF + j]
            print("    pos=" + String(pos) + " dim=" + String(j) + " cos=" + String(c) + " sin=" + String(s))

    # Check frequency at dim=0: freq = 1/10000^(0/256) = 1.0
    # At pos=1: cos(1.0), sin(1.0)
    print("  freq check dim=0 pos=1: cos=" + String(cos_ptr[1 * HALF + 0]) + " sin=" + String(sin_ptr[1 * HALF + 0]) + " (expect cos(1)=0.5403, sin(1)=0.8415)")

    # Check frequency at dim=64: freq = 1/10000^(128/256) = 1/100 = 0.01
    # At pos=1: cos(0.01), sin(0.01)
    print("  freq check dim=64 pos=1: cos=" + String(cos_ptr[1 * HALF + 64]) + " sin=" + String(sin_ptr[1 * HALF + 64]) + " (expect cos(0.01)=0.99995, sin(0.01)=0.01)")
    print()

    cos_ptr.free()
    sin_ptr.free()


def test_full_rope():
    print("=== Full RoPE (theta=1M, head_dim=512, compact partial 128/512) ===")

    comptime MAX_POS = 64
    comptime HALF = 64
    comptime CosSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]

    var cos_ptr = alloc[Scalar[DType.float32]](MAX_POS * HALF)
    var sin_ptr = alloc[Scalar[DType.float32]](MAX_POS * HALF)
    var cos_buf = Bound[CosSlot](Int(cos_ptr))
    var sin_buf = Bound[SinSlot](Int(sin_ptr))

    init_full_rope_tables(cos_buf, sin_buf)

    # Check position 0: cos=1, sin=0 for all dims
    print("  pos=0 check:")
    var max_cos_err_p0 = Float64(0)
    var max_sin_err_p0 = Float64(0)
    for j in range(HALF):
        var ce = abs(Float64(cos_ptr[j]) - 1.0)
        var se = abs(Float64(sin_ptr[j]))
        if ce > max_cos_err_p0:
            max_cos_err_p0 = ce
        if se > max_sin_err_p0:
            max_sin_err_p0 = se
    print("    max |cos-1|=" + String(max_cos_err_p0) + "  max |sin|=" + String(max_sin_err_p0))

    # Check cos^2+sin^2=1 in the stored region
    print("  cos^2+sin^2 check:")
    var max_unit_err = Float64(0)
    for pos in range(MAX_POS):
        for j in range(HALF):
            var c = Float64(cos_ptr[pos * HALF + j])
            var s = Float64(sin_ptr[pos * HALF + j])
            var err = abs(c * c + s * s - 1.0)
            if err > max_unit_err:
                max_unit_err = err
    print("    max |cos^2+sin^2-1|=" + String(max_unit_err))

    print("  compact table width=" + String(HALF) + " (stores only active rotary pairs)")
    print("  sample values (pos, dim, cos, sin):")
    for pos in range(0, MAX_POS, 16):
        for j in range(0, HALF, 16):
            var c = cos_ptr[pos * HALF + j]
            var s = sin_ptr[pos * HALF + j]
            print("    pos=" + String(pos) + " dim=" + String(j) + " cos=" + String(c) + " sin=" + String(s))

    # Frequency at dim=0: freq = 1/1000000^(0/512) = 1.0
    # At pos=1: cos(1), sin(1)
    print("  freq check dim=0 pos=1: cos=" + String(cos_ptr[1 * HALF + 0]) + " sin=" + String(sin_ptr[1 * HALF + 0]) + " (expect 0.5403, 0.8415)")

    print("  boundary pos=10:")
    print("    dim=63 (last active pair): cos=" + String(cos_ptr[10 * HALF + 63]) + " sin=" + String(sin_ptr[10 * HALF + 63]))

    cos_ptr.free()
    sin_ptr.free()


def test_full_rope_apply():
    print("=== Full RoPE apply (compact cache, partial 128/512) ===")

    comptime MAX_POS = 64
    comptime ACTIVE_HALF = 64
    comptime NUM_HEADS = 2
    comptime HEAD_DIM = 512
    comptime HALF = HEAD_DIM // 2
    comptime COLS = NUM_HEADS * HEAD_DIM
    comptime CosSlot = Slot[F32, Replicated, MAX_POS, ACTIVE_HALF, 1]
    comptime SinSlot = Slot[F32, Replicated, MAX_POS, ACTIVE_HALF, 1]
    comptime XView = Slot[BF16, Replicated, 1, COLS, 1]

    var cos_ptr = alloc[Scalar[DType.float32]](MAX_POS * ACTIVE_HALF)
    var sin_ptr = alloc[Scalar[DType.float32]](MAX_POS * ACTIVE_HALF)
    var cos_buf = Bound[CosSlot](Int(cos_ptr))
    var sin_buf = Bound[SinSlot](Int(sin_ptr))
    init_full_rope_tables(cos_buf, sin_buf)

    var orig = alloc[Scalar[DType.bfloat16]](COLS)
    var act = alloc[Scalar[DType.bfloat16]](COLS)
    for i in range(COLS):
        var v = Scalar[DType.bfloat16](Float32(-2.0) + Float32(i) * Float32(0.0078125))
        orig[i] = v
        act[i] = v

    var act_view = DynView[XView](Int(act), 1)
    var pos = 11
    apply_full_rope[NUM_HEADS](act_view, cos_buf, sin_buf, pos)

    var max_rot_err = Float64(0)
    var max_passthrough_err = Float64(0)

    for h in range(NUM_HEADS):
        var head_base = h * HEAD_DIM
        for j in range(ACTIVE_HALF):
            var lo_idx = head_base + j
            var hi_idx = head_base + HALF + j
            var lo = Float32(orig[lo_idx])
            var hi = Float32(orig[hi_idx])
            var cv = Float32(cos_ptr[pos * ACTIVE_HALF + j])
            var sv = Float32(sin_ptr[pos * ACTIVE_HALF + j])
            var lo_ref = Float32(Scalar[DType.bfloat16](lo * cv - hi * sv))
            var hi_ref = Float32(Scalar[DType.bfloat16](hi * cv + lo * sv))
            var lo_err = abs(Float64(Float32(act[lo_idx])) - Float64(lo_ref))
            var hi_err = abs(Float64(Float32(act[hi_idx])) - Float64(hi_ref))
            if lo_err > max_rot_err:
                max_rot_err = lo_err
            if hi_err > max_rot_err:
                max_rot_err = hi_err

        for j in range(ACTIVE_HALF, HALF):
            var lo_idx = head_base + j
            var hi_idx = head_base + HALF + j
            var lo_err = abs(Float64(Float32(act[lo_idx])) - Float64(Float32(orig[lo_idx])))
            var hi_err = abs(Float64(Float32(act[hi_idx])) - Float64(Float32(orig[hi_idx])))
            if lo_err > max_passthrough_err:
                max_passthrough_err = lo_err
            if hi_err > max_passthrough_err:
                max_passthrough_err = hi_err

    print("  max rotated bf16 error=" + String(max_rot_err))
    print("  max passthrough error=" + String(max_passthrough_err) + " (should be 0)")
    print("  sample active pair j=63: lo=" + String(Float32(act[63])) + " hi=" + String(Float32(act[HALF + 63])))
    print("  sample untouched pair j=64: lo=" + String(Float32(act[64])) + " hi=" + String(Float32(act[HALF + 64])))
    print()

    cos_ptr.free()
    sin_ptr.free()
    orig.free()
    act.free()


def main():
    test_sliding_rope()
    test_full_rope()
    test_full_rope_apply()
