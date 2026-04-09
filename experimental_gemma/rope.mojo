"""Gemma4 dual-config RoPE table initialization.

Two RoPE configurations:
  Sliding: theta=10000, head_dim=256, full rotation (all 256 dims)
  Full:    theta=1000000, head_dim=512, partial rotation (128 of 512 dims)

Sliding tables store the full 128 half-dim entries. Full-attention
tables store only the 64 active half-dim entries, and the untouched
channels are skipped by a partial RoPE apply wrapper.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding, Shaped, Bound, DynView
from kernels.kv_rotors import rope_partial
from simd_math import sincos


def init_sliding_rope_tables[CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    cos_buf: Bound[CosT], sin_buf: Bound[SinT],
) where CosT.DTYPE == DType.float32:
    """Sliding RoPE: theta=10000, full head_dim=256 rotation.

    cos/sin tables have COLS = head_dim/2 = 128.
    inv_freq[i] = 1 / (10000 ^ (2i / 256)) for i in 0..127.
    """
    comptime assert SinT.DTYPE == DType.float32, "rope init: sin must be f32"
    comptime assert CosT.ROWS == SinT.ROWS, "rope init: cos/sin rows mismatch"
    comptime assert CosT.COLS == SinT.COLS, "rope init: cos/sin cols mismatch"

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=cos_buf.ptr)
    var sp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=sin_buf.ptr)
    comptime half = CosT.COLS
    comptime head_dim = half * 2
    comptime f64w = simd_width_of[DType.float64]()
    comptime theta = Float64(10000.0)

    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            inv[k] = 1.0 / (theta ** (Float64(2 * (j + k)) / Float64(head_dim)))

        for pos in range(CosT.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]())


def init_full_rope_tables[CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    cos_buf: Bound[CosT], sin_buf: Bound[SinT],
) where CosT.DTYPE == DType.float32:
    """Full-attention RoPE: theta=1000000, partial rotation (128 of 512 dims).

    The compact tables store only the active rotary half-dim entries:
    cos/sin tables have COLS = rotary_dim/2 = 64.

    The real frequencies are:
      inv_freq[i] = 1 / (1000000 ^ (2i / 512)) for i in 0..63

    The denominator uses full head_dim=512, not rotary_dim=128.
    """
    comptime assert SinT.DTYPE == DType.float32, "rope init: sin must be f32"
    comptime assert CosT.ROWS == SinT.ROWS, "rope init: cos/sin rows mismatch"
    comptime assert CosT.COLS == SinT.COLS, "rope init: cos/sin cols mismatch"
    comptime assert CosT.COLS == 64, "full rope init: tables must store only the 64 active half-dim entries"

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=cos_buf.ptr)
    var sp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](unsafe_from_address=sin_buf.ptr)
    comptime half = CosT.COLS
    comptime head_dim = 512
    comptime f64w = simd_width_of[DType.float64]()
    comptime theta = Float64(1000000.0)

    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            inv[k] = 1.0 / (theta ** (Float64(2 * (j + k)) / Float64(head_dim)))

        for pos in range(CosT.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]())


def apply_full_rope[num_heads: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """Apply Gemma4 full-attention RoPE using the compact 128-dim rotary cache."""
    comptime assert XT.DTYPE == DType.bfloat16, "full rope: input must be bf16"
    comptime assert XT.COLS == num_heads * 512, "full rope: cols != heads * 512"

    rope_partial[512, 128, num_heads](x, cos_table, sin_table, pos)
