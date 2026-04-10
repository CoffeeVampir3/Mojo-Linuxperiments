"""Gemma 4 KV cache — VNNI-packed at hardware SIMD width.

K packed for scoring GEMV (Q × K^T via vpdpbusd):
  Layout: [head][pos_group][k_dim_group × WIDTH × VNNI_BLK]
  WIDTH positions packed together, 4 K-dim values per VNNI dot.
  Stored as u8 (XOR 0x80) for unsigned first operand.

V packed for V-agg GEMV (W × V via vpdpbusd):
  Layout: [head][pos_group][channel_group × sub_quad × WIDTH × VNNI_BLK]
  WIDTH channels as lanes, VNNI_BLK=4 positions per dot product.
  Stored as i8 for signed second operand.
  pos_group matches K granularity (WIDTH positions) for single-pass scoring.

Both layouts parameterized on WIDTH = simd_width_of[DType.int32]() so the
hardware VNNI instruction reads contiguous data with no transposition.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray

from experimental.amx import VNNI_BLK
from simd_math import sqrt


comptime CACHE_WIDTH = simd_width_of[DType.int32]()


struct Gemma4KVCache[max_seq: Int, head_dim: Int, num_kv_heads: Int, num_q_heads: Int = 0]:
    comptime WIDTH = CACHE_WIDTH

    # K: [head][pos_group][k_dim_group × WIDTH × VNNI_BLK]
    comptime K_DIM_GROUPS = Self.head_dim // VNNI_BLK
    comptime K_POS_GROUPS = (Self.max_seq + Self.WIDTH - 1) // Self.WIDTH
    comptime K_PG_BYTES = Self.K_DIM_GROUPS * Self.WIDTH * VNNI_BLK
    comptime K_HEAD_BYTES = Self.K_POS_GROUPS * Self.K_PG_BYTES
    comptime K_TOTAL = Self.num_kv_heads * Self.K_HEAD_BYTES

    # V: [head][pos_group][channel_group × sub_quad × WIDTH × VNNI_BLK]
    comptime V_CHANNEL_GROUPS = Self.head_dim // Self.WIDTH
    comptime V_SUB_QUADS = Self.WIDTH // VNNI_BLK
    comptime V_CG_BYTES = Self.V_SUB_QUADS * Self.WIDTH * VNNI_BLK
    comptime V_PG_BYTES = Self.V_CHANNEL_GROUPS * Self.V_CG_BYTES
    comptime V_HEAD_BYTES = Self.K_POS_GROUPS * Self.V_PG_BYTES
    comptime V_TOTAL = Self.num_kv_heads * Self.V_HEAD_BYTES

    # Scales
    comptime ACTUAL_Q_HEADS = Self.num_q_heads if Self.num_q_heads > 0 else Self.num_kv_heads
    comptime K_SCALE_BYTES = Self.num_kv_heads * Self.max_seq * size_of[Float32]()
    comptime V_SCALE_BYTES = Self.num_kv_heads * Self.max_seq * size_of[Float32]()
    comptime Q_SCALE_BYTES = Self.ACTUAL_Q_HEADS * Self.max_seq * size_of[Float32]()

    comptime TOTAL_BYTES = Self.K_TOTAL + Self.V_TOTAL + Self.K_SCALE_BYTES + Self.V_SCALE_BYTES + Self.Q_SCALE_BYTES

    var k_base: Int
    var v_base: Int
    var k_scale_base: Int
    var v_scale_base: Int
    var q_scale_base: Int

    def __init__(out self, base: Int):
        self.k_base = base
        self.v_base = base + Self.K_TOTAL
        self.k_scale_base = self.v_base + Self.V_TOTAL
        self.v_scale_base = self.k_scale_base + Self.K_SCALE_BYTES
        self.q_scale_base = self.v_scale_base + Self.V_SCALE_BYTES

    # ================================================================
    # K write — scatter into width-packed VNNI layout
    # ================================================================

    def write_k(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ):
        """K write: i8 → u8 (XOR 0x80) → scatter into VNNI-packed layout.

        Each K dim group holds WIDTH positions × VNNI_BLK K values.
        This write places one position's data at its slot within each group.
        """
        var pos_group = pos // Self.WIDTH
        var slot = pos % Self.WIDTH
        var k_pg = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.k_base + head * Self.K_HEAD_BYTES + pos_group * Self.K_PG_BYTES)

        for kdg in range(Self.K_DIM_GROUPS):
            var src = data_i8 + kdg * VNNI_BLK
            var dst = k_pg + kdg * Self.WIDTH * VNNI_BLK + slot * VNNI_BLK
            var v = src.load[width=VNNI_BLK]().cast[DType.uint8]() ^ SIMD[DType.uint8, VNNI_BLK](0x80)
            dst.store(v)

    # ================================================================
    # V write — scatter into transposed VNNI layout
    # ================================================================

    def write_v(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ):
        """V write: i8 → scatter into VNNI layout (transposed from K).

        V lanes are channels (WIDTH), VNNI_BLK inner axis is positions.
        Each write scatters head_dim values across channel_groups.
        """
        var pos_group = pos // Self.WIDTH
        var sub_quad = (pos % Self.WIDTH) // VNNI_BLK
        var vnni_slot = pos % VNNI_BLK
        var v_pg = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=self.v_base + head * Self.V_HEAD_BYTES + pos_group * Self.V_PG_BYTES)

        for cg in range(Self.V_CHANNEL_GROUPS):
            var cg_base = v_pg + cg * Self.V_CG_BYTES + sub_quad * Self.WIDTH * VNNI_BLK
            for ci in range(Self.WIDTH):
                cg_base[ci * VNNI_BLK + vnni_slot] = data_i8[cg * Self.WIDTH + ci]

    # ================================================================
    # Scale write
    # ================================================================

    def write_k_scale(self, pos: Int, head: Int, scale: Float32):
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.k_scale_base + head * Self.max_seq * size_of[Float32]() + pos * size_of[Float32]()
        )[] = scale

    def write_v_scale(self, pos: Int, head: Int, scale: Float32):
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.v_scale_base + head * Self.max_seq * size_of[Float32]() + pos * size_of[Float32]()
        )[] = scale

    # ================================================================
    # Scoring/V-agg access helpers
    # ================================================================

    def k_pg_ptr(self, head: Int, pos_group: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """K data pointer for one position group (all K dim groups)."""
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.k_base + head * Self.K_HEAD_BYTES + pos_group * Self.K_PG_BYTES)

    def v_pg_ptr(self, head: Int, pos_group: Int) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        """V data pointer for one position group (all channel groups)."""
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=self.v_base + head * Self.V_HEAD_BYTES + pos_group * Self.V_PG_BYTES)

    def k_scale_ptr(self, head: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.k_scale_base + head * Self.max_seq * size_of[Float32]())

    def v_scale_ptr(self, head: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.v_scale_base + head * Self.max_seq * size_of[Float32]())


# ============================================================================
# Validation
# ============================================================================


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def validate_k_roundtrip[head_dim: Int, num_positions: Int]():
    """Write K positions, read back via scoring layout, verify XOR and packing."""
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]
    comptime WIDTH = Cache.WIDTH

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    var rng = UInt64(0xFEEDFACE12345678)
    var golden_i8 = alloc[Scalar[DType.int8]](num_heads * num_positions * head_dim)

    for h in range(num_heads):
        for pos in range(num_positions):
            var row = alloc[Scalar[DType.int8]](head_dim)
            for d in range(head_dim):
                var val = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))
                row[d] = val
                golden_i8[h * num_positions * head_dim + pos * head_dim + d] = val
            cache.write_k(pos, h, row)
            row.free()

    # Read back: for each position, reconstruct i8 from VNNI layout
    var mismatches = 0
    for h in range(num_heads):
        for pos in range(num_positions):
            var pos_group = pos // WIDTH
            var slot = pos % WIDTH
            var k_pg = cache.k_pg_ptr(h, pos_group)

            for kdg in range(Cache.K_DIM_GROUPS):
                var base = k_pg + kdg * WIDTH * VNNI_BLK + slot * VNNI_BLK
                for b in range(VNNI_BLK):
                    var stored_u8 = base.bitcast[Scalar[DType.uint8]]()[b]
                    var recovered_i8 = (stored_u8 ^ UInt8(0x80)).cast[DType.int8]()
                    var expected = golden_i8[h * num_positions * head_dim + pos * head_dim + kdg * VNNI_BLK + b]
                    if recovered_i8 != expected:
                        mismatches += 1

    print("  K bytes compared:  " + String(num_heads * num_positions * head_dim))
    print("  K mismatches:      " + String(mismatches))

    cache_buf.free()
    golden_i8.free()


def validate_v_roundtrip[head_dim: Int, num_positions: Int]():
    """Write V positions, read back via V-agg layout, verify packing."""
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]
    comptime WIDTH = Cache.WIDTH

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    var rng = UInt64(0xDEADCAFE87654321)
    var golden = alloc[Scalar[DType.int8]](num_heads * num_positions * head_dim)

    for h in range(num_heads):
        for pos in range(num_positions):
            var row = alloc[Scalar[DType.int8]](head_dim)
            for d in range(head_dim):
                var val = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))
                row[d] = val
                golden[h * num_positions * head_dim + pos * head_dim + d] = val
            cache.write_v(pos, h, row)
            row.free()

    # Read back: reconstruct from VNNI V layout
    var mismatches = 0
    for h in range(num_heads):
        for pos in range(num_positions):
            var pos_group = pos // WIDTH
            var sub_quad = (pos % WIDTH) // VNNI_BLK
            var vnni_slot = pos % VNNI_BLK
            var v_pg = cache.v_pg_ptr(h, pos_group)

            for cg in range(Cache.V_CHANNEL_GROUPS):
                var cg_base = v_pg + cg * Cache.V_CG_BYTES + sub_quad * WIDTH * VNNI_BLK
                for ci in range(WIDTH):
                    var got = cg_base[ci * VNNI_BLK + vnni_slot]
                    var expected = golden[h * num_positions * head_dim + pos * head_dim + cg * WIDTH + ci]
                    if got != expected:
                        mismatches += 1

    print("  V bytes compared:  " + String(num_heads * num_positions * head_dim))
    print("  V mismatches:      " + String(mismatches))

    cache_buf.free()
    golden.free()


def validate_k_scale_roundtrip[head_dim: Int]():
    comptime num_heads = 2
    comptime max_seq = 128
    comptime Cache = Gemma4KVCache[max_seq, head_dim, num_heads]

    var cache_buf = alloc[UInt8](Cache.TOTAL_BYTES)
    for i in range(Cache.TOTAL_BYTES):
        cache_buf[i] = UInt8(0)
    var cache = Cache(Int(cache_buf))

    for h in range(num_heads):
        for pos in range(16):
            cache.write_k_scale(pos, h, Float32(pos) * 0.1 + Float32(h) * 10.0)

    var max_err = Float64(0)
    var scale_mismatches = 0
    for h in range(num_heads):
        var sp = cache.k_scale_ptr(h)
        for pos in range(16):
            var expected = Float32(pos) * 0.1 + Float32(h) * 10.0
            var got = sp[pos]
            var err = Float64((got - expected).__abs__())
            if err > max_err:
                max_err = err
            if err > 1e-7:
                scale_mismatches += 1

    print("  scales compared: " + String(32))
    print("  max abs error:   " + String(max_err))
    print("  mismatches:      " + String(scale_mismatches))

    cache_buf.free()


def main():
    print("=== Gemma4KVCache validation (width=" + String(CACHE_WIDTH) + ") ===")

    print("\nK VNNI roundtrip (head_dim=256, 64 positions, 2 heads):")
    validate_k_roundtrip[256, 64]()

    print("\nK VNNI roundtrip (head_dim=512, 64 positions, 2 heads):")
    validate_k_roundtrip[512, 64]()

    print("\nV VNNI roundtrip (head_dim=256, 64 positions, 2 heads):")
    validate_v_roundtrip[256, 64]()

    print("\nV VNNI roundtrip (head_dim=512, 64 positions, 2 heads):")
    validate_v_roundtrip[512, 64]()

    print("\nK scale roundtrip (head_dim=256):")
    validate_k_scale_roundtrip[256]()

    print("\nK scale roundtrip (head_dim=512):")
    validate_k_scale_roundtrip[512]()
