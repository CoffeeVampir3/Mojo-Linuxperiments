"""Attention cache — K/V data with per-head dynamic scales for Q and K.

K stored in VNNI tile format for direct AMX B-tile loads (no runtime
transpose). Layout: [head][tile_idx][k_slice × TILE_BYTES] where each
tile covers TILE_N=16 consecutive positions. Writes scatter into VNNI
via the 16×16 uint32 butterfly transpose.

V stored as i8 in row-major [head][pos][dim] for tdpbusd signed operand.
V uses a corrected per-layer fixed scale (not stored here).

Per-head dynamic scales (f32):
  K scales: [num_kv_heads][max_seq] — written at K cache write time
  Q scales: [num_q_heads][max_seq]  — written at Q prep time

These enable per-head absmax quantization for Q and K, eliminating the
clipping caused by a single per-layer scale across heads with varying norms.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from experimental.amx import TILE_N, TILE_BYTES, K_STEP


struct KVCache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int = 0]:
    var k_base: Int   # VNNI K: [head][tile][k_slice × TILE_BYTES]
    var v_base: Int   # row-major V: [head][pos][dim]
    var k_scale_base: Int  # f32 K scales: [num_kv_heads][max_seq]
    var q_scale_base: Int  # f32 Q scales: [num_q_heads][max_seq]

    comptime K_SLICES = Self.head_dim // K_STEP
    comptime TILE_K_BYTES = Self.K_SLICES * TILE_BYTES
    comptime MAX_TILES = (Self.max_seq + TILE_N - 1) // TILE_N
    comptime K_HEAD_STRIDE = Self.MAX_TILES * Self.TILE_K_BYTES
    comptime K_TOTAL_BYTES = Self.num_kv_heads * Self.K_HEAD_STRIDE

    comptime V_HEAD_STRIDE = Self.max_seq * Self.head_dim
    comptime V_TOTAL_BYTES = Self.num_kv_heads * Self.V_HEAD_STRIDE

    comptime ACTUAL_Q_HEADS = Self.num_q_heads if Self.num_q_heads > 0 else Self.num_kv_heads
    comptime K_SCALE_BYTES = Self.num_kv_heads * Self.max_seq * size_of[Float32]()
    comptime Q_SCALE_BYTES = Self.ACTUAL_Q_HEADS * Self.max_seq * size_of[Float32]()

    comptime TOTAL_BYTES = Self.K_TOTAL_BYTES + Self.V_TOTAL_BYTES + Self.K_SCALE_BYTES + Self.Q_SCALE_BYTES

    def __init__(out self, base: Int):
        """Layout: [K data | V data | K scales | Q scales]."""
        self.k_base = base
        self.v_base = base + Self.K_TOTAL_BYTES
        self.k_scale_base = self.v_base + Self.V_TOTAL_BYTES
        self.q_scale_base = self.k_scale_base + Self.K_SCALE_BYTES

    def write_k(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Int8, MutAnyOrigin],
    ):
        """Write K: i8 → u8 (XOR 0x80) → scatter into VNNI tile format.

        Transposes the full tile containing this position. For sequential
        writes filling a tile (pos % TILE_N == 0..15), the tile is rebuilt
        each time with zeros for not-yet-written positions.
        """
        comptime xor_w = simd_width_of[DType.uint8]()
        comptime xor_mask = SIMD[DType.uint8, xor_w](0x80)
        var tile_idx = pos // TILE_N
        var col = pos % TILE_N
        var src = data_i8.bitcast[UInt8]()
        var vnni_head = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.k_base + head * Self.K_HEAD_STRIDE)

        # XOR to u8 and scatter: write column `col` of the VNNI tile
        for ki in range(Self.K_SLICES):
            var k_off = ki * K_STEP
            # Load this position's K-slice as 16 uint32 (64 bytes)
            var row_u8 = InlineArray[UInt8, K_STEP](fill=UInt8(0))
            var rp = UnsafePointer(to=row_u8).bitcast[UInt8]()
            var b = 0
            while b + xor_w <= K_STEP:
                (rp + b).store((src + k_off + b).load[width=xor_w]() ^ xor_mask)
                b += xor_w
            var src_u32 = UnsafePointer(to=row_u8).bitcast[UInt32]()
            var tile_base = (vnni_head + tile_idx * Self.TILE_K_BYTES + ki * TILE_BYTES).bitcast[UInt32]()
            # Scatter: element d goes to row d, column col of the transposed tile
            for d in range(TILE_N):
                (tile_base + d * TILE_N + col)[] = src_u32.load[width=1](offset=d)

    def write_v(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Int8, MutAnyOrigin],
    ):
        """Write V head: i8 stored directly for tdpbusd signed operand."""
        comptime w = simd_width_of[DType.int8]()
        var dst = UnsafePointer[Int8, MutAnyOrigin](
            unsafe_from_address=self.v_base + head * Self.V_HEAD_STRIDE + pos * Self.head_dim)
        var i = 0
        while i + w <= Self.head_dim:
            (dst + i).store(data_i8.load[width=w](offset=i))
            i += w

    def v_head(self, pos: Int, head: Int) -> UnsafePointer[Int8, MutAnyOrigin]:
        """V data pointer: i8 at [head][pos]."""
        return UnsafePointer[Int8, MutAnyOrigin](
            unsafe_from_address=self.v_base + head * Self.V_HEAD_STRIDE + pos * Self.head_dim)

    def write_k_scale(self, pos: Int, head: Int, scale: Float32):
        """Store the per-head absmax scale used to quantize K at this position."""
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.k_scale_base + head * Self.max_seq * size_of[Float32]() + pos * size_of[Float32]()
        )[] = scale

