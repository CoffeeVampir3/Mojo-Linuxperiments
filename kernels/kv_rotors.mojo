"""Rotary position embedding kernels.

Standard RoPE and YaRN-extended RoPE for MLA attention.
"""

from std.math import log, sqrt
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from simd_math import sincos


# =============================================================================
# Table initialization
# =============================================================================


def yarn_find_correction_dim(
    num_rotations: Float64, dim: Int, theta: Float64, max_pos: Int,
) -> Float64:
    return (Float64(dim) * log(Float64(max_pos) / (num_rotations * 2.0 * 3.14159265358979323846))) / (2.0 * log(theta))


def yarn_find_correction_range(
    beta_fast: Float64, beta_slow: Float64,
    dim: Int, theta: Float64, max_pos: Int,
) -> Tuple[Int, Int]:
    var low_f = yarn_find_correction_dim(beta_fast, dim, theta, max_pos)
    var high_f = yarn_find_correction_dim(beta_slow, dim, theta, max_pos)
    var low = max(Int(low_f), 0)
    var high = Int(high_f)
    if Float64(high) < high_f:
        high += 1
    return (low, min(high, dim - 1))


def yarn_linear_ramp(low: Int, high: Int, i: Int) -> Float64:
    if low == high:
        return Float64(1 if i > low else 0)
    if i <= low:
        return 0.0
    if i >= high:
        return 1.0
    return Float64(i - low) / Float64(high - low)


def init_rope_tables[CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    cos_buf: Bound[CosT], sin_buf: Bound[SinT],
    theta: Float64 = 10000.0,
    factor: Float64 = 1.0,
    original_max_pos: Int = 4096,
    beta_fast: Float64 = 32.0,
    beta_slow: Float64 = 1.0,
    mscale: Float64 = 0.0,
) where CosT.DTYPE == DType.float32:
    """Precompute cos/sin tables for RoPE. Supports standard, YaRN, and mscale.

    With factor=1.0 (default), produces standard RoPE frequencies.
    With factor>1.0, applies YaRN linear ramp interpolation between
    original and scaled frequencies.
    With mscale>0.0, multiplies cos/sin values by yarn_mscale(factor, mscale).
    This bakes the attention scaling correction into the positional encoding
    so the scoring kernel needs no extra scale factor.
    """
    comptime assert SinT.DTYPE == DType.float32, "rope init: sin must be f32"
    comptime assert CosT.ROWS == SinT.ROWS, "rope init: cos/sin rows mismatch"
    comptime assert CosT.COLS == SinT.COLS, "rope init: cos/sin cols mismatch"
    comptime assert CosT.COLS % simd_width_of[DType.float64]() == 0, "rope init: cols must be f64-simd-aligned"

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=cos_buf.ptr
    )
    var sp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=sin_buf.ptr
    )
    comptime half = CosT.COLS
    comptime head_dim = half * 2
    comptime f64w = simd_width_of[DType.float64]()

    var ramp_low = 0
    var ramp_high = half - 1
    if factor > 1.0:
        var ramp = yarn_find_correction_range(
            beta_fast, beta_slow, head_dim, theta, original_max_pos,
        )
        ramp_low = ramp[0]
        ramp_high = ramp[1]

    var ms = Float64(1.0)
    if mscale > 0.0 and factor > 1.0:
        ms = yarn_mscale(factor, mscale)
    var ms_f32 = SIMD[DType.float32, f64w](Float32(ms))

    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            var dim_idx = j + k
            var orig_freq = 1.0 / (theta ** (Float64(2 * dim_idx) / Float64(head_dim)))
            if factor > 1.0:
                var scaled_freq = orig_freq / factor
                var ramp_mix = yarn_linear_ramp(ramp_low, ramp_high, dim_idx)
                inv[k] = orig_freq * (1.0 - ramp_mix) + scaled_freq * ramp_mix
            else:
                inv[k] = orig_freq

        for pos in range(CosT.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]() * ms_f32)
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]() * ms_f32)


def yarn_mscale(factor: Float64, mscale_all_dim: Float64) -> Float64:
    """Compute YaRN mscale correction for softmax_scale.

    softmax_scale = (1/sqrt(head_dim)) * mscale^2
    """
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale_all_dim * log(factor) + 1.0


def yarn_softmax_scale(head_dim: Int, factor: Float64, mscale_all_dim: Float64) -> Float64:
    var ms = yarn_mscale(factor, mscale_all_dim)
    return (1.0 / sqrt(Float64(head_dim))) * ms * ms


# =============================================================================
# RoPE application — single parameterized kernel
# =============================================================================


def rope_apply[
    rope_dim: Int,
    num_blocks: Int,
    block_stride: Int,
    block_offset: Int,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped,
](
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_stride: Int,
    seq_len: Int,
    cos_table: Bound[CosT], sin_table: Bound[SinT],
    pos: Int,
) where CosT.DTYPE == DType.float32:
    """Apply half-split RoPE rotation in-place.

    Iterates over seq_len rows. Within each row, rotates num_blocks
    segments of rope_dim elements, each located at
    (block_index * block_stride + block_offset) from the row start.
    """
    comptime assert SinT.DTYPE == DType.float32, "rope: sin must be f32"
    comptime assert CosT.COLS == rope_dim // 2, "rope: cos cols mismatch"
    comptime assert SinT.COLS == rope_dim // 2, "rope: sin cols mismatch"
    comptime assert CosT.ROWS == SinT.ROWS, "rope: cos/sin capacity mismatch"
    comptime assert rope_dim % 2 == 0, "rope: rope_dim must be even"
    comptime assert (rope_dim // 2) % simd_width_of[DType.float32]() == 0, "rope: half must be f32-simd-aligned"

    if seq_len == 0:
        return
    debug_assert(pos >= 0 and pos + seq_len <= CosT.ROWS,
        "rope: position range exceeds table capacity")

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=cos_table.ptr
    )
    var sn = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=sin_table.ptr
    )
    comptime half = rope_dim // 2
    comptime width = simd_width_of[DType.float32]()

    for m in range(seq_len):
        var actual_pos = pos + m
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half
        var row_base = ptr + m * row_stride

        for b in range(num_blocks):
            var base = row_base + b * block_stride + block_offset
            for j in range(0, half, width):
                var x_lo = (base + j).load[width=width]().cast[DType.float32]()
                var x_hi = (base + half + j).load[width=width]().cast[DType.float32]()
                var cv = (cos_row + j).load[width=width]()
                var sv = (sin_row + j).load[width=width]()
                (base + j).store(
                    (x_lo * cv - x_hi * sv).cast[DType.bfloat16]()
                )
                (base + half + j).store(
                    (x_hi * cv + x_lo * sv).cast[DType.bfloat16]()
                )


def rope_apply_partial[
    head_dim: Int,
    rotary_dim: Int,
    num_blocks: Int,
    block_stride: Int,
    block_offset: Int,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped,
](
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    row_stride: Int,
    seq_len: Int,
    cos_table: Bound[CosT], sin_table: Bound[SinT],
    pos: Int,
) where CosT.DTYPE == DType.float32:
    """Apply RoPE in-place to only the leading rotary_dim channels of each head.

    The input still uses the standard rotate_half layout for the full head_dim:
    the first rotary_dim/2 entries of the low half are rotated against the
    first rotary_dim/2 entries of the high half, while the remaining channels
    are left untouched.
    """
    comptime assert SinT.DTYPE == DType.float32, "rope partial: sin must be f32"
    comptime assert head_dim % 2 == 0, "rope partial: head_dim must be even"
    comptime assert rotary_dim % 2 == 0, "rope partial: rotary_dim must be even"
    comptime assert rotary_dim <= head_dim, "rope partial: rotary_dim must fit within head_dim"
    comptime assert CosT.COLS == rotary_dim // 2, "rope partial: cos cols mismatch"
    comptime assert SinT.COLS == rotary_dim // 2, "rope partial: sin cols mismatch"
    comptime assert CosT.ROWS == SinT.ROWS, "rope partial: cos/sin capacity mismatch"
    comptime assert (rotary_dim // 2) % simd_width_of[DType.float32]() == 0, "rope partial: half must be f32-simd-aligned"

    if seq_len == 0:
        return
    debug_assert(pos >= 0 and pos + seq_len <= CosT.ROWS,
        "rope partial: position range exceeds table capacity")

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=cos_table.ptr
    )
    var sn = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=sin_table.ptr
    )
    comptime full_half = head_dim // 2
    comptime rotary_half = rotary_dim // 2
    comptime width = simd_width_of[DType.float32]()

    for m in range(seq_len):
        var actual_pos = pos + m
        var cos_row = cp + actual_pos * rotary_half
        var sin_row = sn + actual_pos * rotary_half
        var row_base = ptr + m * row_stride

        for b in range(num_blocks):
            var base = row_base + b * block_stride + block_offset
            for j in range(0, rotary_half, width):
                var x_lo = (base + j).load[width=width]().cast[DType.float32]()
                var x_hi = (base + full_half + j).load[width=width]().cast[DType.float32]()
                var cv = (cos_row + j).load[width=width]()
                var sv = (sin_row + j).load[width=width]()
                (base + j).store(
                    (x_lo * cv - x_hi * sv).cast[DType.bfloat16]()
                )
                (base + full_half + j).store(
                    (x_hi * cv + x_lo * sv).cast[DType.bfloat16]()
                )


# =============================================================================
# Typed wrappers — constrain shapes at compile time via DynView
# =============================================================================


def rope[head_dim: Int, num_heads: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """Standard RoPE: rotate full head_dim per head."""
    comptime assert XT.DTYPE == DType.bfloat16, "rope: must be bf16"
    comptime assert XT.COLS == head_dim * num_heads, "rope: cols != heads * dim"

    var xp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=x.ptr
    )
    rope_apply[head_dim, num_heads, head_dim, 0](
        xp, XT.COLS, x.seq_len, cos_table, sin_table, pos,
    )


def rope_partial[head_dim: Int, rotary_dim: Int, num_heads: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """RoPE wrapper for heads where only a prefix of channels is rotary."""
    comptime assert XT.DTYPE == DType.bfloat16, "rope partial: must be bf16"
    comptime assert XT.COLS == head_dim * num_heads, "rope partial: cols != heads * dim"

    var xp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=x.ptr
    )
    rope_apply_partial[head_dim, rotary_dim, num_heads, head_dim, 0](
        xp, XT.COLS, x.seq_len, cos_table, sin_table, pos,
    )


def mla_rope_q[
    rope_dim: Int, nope_dim: Int, num_heads: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped,
](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """MLA query RoPE: rotate only the q_pe portion of each head."""
    comptime head_dim = nope_dim + rope_dim
    comptime assert XT.DTYPE == DType.bfloat16, "mla_rope_q: must be bf16"
    comptime assert XT.COLS == num_heads * head_dim, "mla_rope_q: cols mismatch"

    var xp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=x.ptr
    )
    rope_apply[rope_dim, num_heads, head_dim, nope_dim](
        xp, XT.COLS, x.seq_len, cos_table, sin_table, pos,
    )


def mla_rope_kr[
    rope_dim: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped,
](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """MLA key RoPE: rotate the k_R tail of the kv_a output."""
    comptime kv_lora_rank = XT.COLS - rope_dim
    comptime assert XT.DTYPE == DType.bfloat16, "mla_rope_kr: must be bf16"

    var xp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=x.ptr
    )
    rope_apply[rope_dim, 1, XT.COLS, kv_lora_rank](
        xp, XT.COLS, x.seq_len, cos_table, sin_table, pos,
    )
