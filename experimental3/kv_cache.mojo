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
from std.sys.info import simd_width_of, size_of

from experimental3.amx import VNNI_BLK


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

    # K and V scales are both dynamic per cached position. The V scale is
    # folded into the W (attention-weight) quantization in v_agg_group, so the
    # inner loop sees no extra cost — only the W max absorbs both factors.
    comptime K_SCALE_BYTES = Self.num_kv_heads * Self.max_seq * size_of[Float32]()
    comptime V_SCALE_BYTES = Self.num_kv_heads * Self.max_seq * size_of[Float32]()

    comptime TOTAL_BYTES = Self.K_TOTAL + Self.V_TOTAL + Self.K_SCALE_BYTES + Self.V_SCALE_BYTES

    var k_base: Int
    var v_base: Int
    var k_scale_base: Int
    var v_scale_base: Int

    def __init__(out self, base: Int):
        self.k_base = base
        self.v_base = base + Self.K_TOTAL
        self.k_scale_base = self.v_base + Self.V_TOTAL
        self.v_scale_base = self.k_scale_base + Self.K_SCALE_BYTES

    @always_inline
    def assert_layout(self):
        debug_assert(Self.head_dim % VNNI_BLK == 0, "head_dim must be a multiple of VNNI_BLK")
        debug_assert(Self.head_dim % Self.WIDTH == 0, "head_dim must be a multiple of cache WIDTH")
        debug_assert(Self.WIDTH % VNNI_BLK == 0, "cache WIDTH must be a multiple of VNNI_BLK")

    @always_inline
    def assert_head(self, head: Int):
        debug_assert(head >= 0 and head < Self.num_kv_heads, "head out of range")

    @always_inline
    def assert_pos(self, pos: Int):
        debug_assert(pos >= 0 and pos < Self.max_seq, "position out of range")

    @always_inline
    def assert_pos_group(self, pos_group: Int):
        debug_assert(pos_group >= 0 and pos_group < Self.K_POS_GROUPS, "position group out of range")

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
        self.assert_layout()
        self.assert_head(head)
        self.assert_pos(pos)
        var pos_group = pos // Self.WIDTH
        var slot = pos % Self.WIDTH
        var k_pg = self.k_pg_ptr(head, pos_group)
        var u8_bias = SIMD[DType.uint8, VNNI_BLK](0x80)

        for kdg in range(Self.K_DIM_GROUPS):
            var src = data_i8 + kdg * VNNI_BLK
            var dst = k_pg + kdg * Self.WIDTH * VNNI_BLK + slot * VNNI_BLK
            var v = src.load[width=VNNI_BLK]().cast[DType.uint8]() ^ u8_bias
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
        self.assert_layout()
        self.assert_head(head)
        self.assert_pos(pos)
        var pos_group = pos // Self.WIDTH
        var sub_quad = (pos % Self.WIDTH) // VNNI_BLK
        var vnni_slot = pos % VNNI_BLK
        var v_pg = self.v_pg_ptr(head, pos_group)

        for cg in range(Self.V_CHANNEL_GROUPS):
            var cg_base = v_pg + cg * Self.V_CG_BYTES + sub_quad * Self.WIDTH * VNNI_BLK
            var channels = (data_i8 + cg * Self.WIDTH).load[width=Self.WIDTH]()
            (cg_base + vnni_slot).strided_store[width=Self.WIDTH](channels, VNNI_BLK)

    # ================================================================
    # Scale write
    # ================================================================

    def write_k_scale(self, pos: Int, head: Int, scale: Float32):
        self.assert_head(head)
        self.assert_pos(pos)
        self.k_scale_ptr(head)[pos] = scale

    def write_v_scale(self, pos: Int, head: Int, scale: Float32):
        self.assert_head(head)
        self.assert_pos(pos)
        self.v_scale_ptr(head)[pos] = scale

    # ================================================================
    # Scoring/V-agg access helpers
    # ================================================================

    def k_pg_ptr(self, head: Int, pos_group: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """K data pointer for one position group (all K dim groups)."""
        self.assert_head(head)
        self.assert_pos_group(pos_group)
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.k_base + head * Self.K_HEAD_BYTES + pos_group * Self.K_PG_BYTES)

    def v_pg_ptr(self, head: Int, pos_group: Int) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        """V data pointer for one position group (all channel groups)."""
        self.assert_head(head)
        self.assert_pos_group(pos_group)
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=self.v_base + head * Self.V_HEAD_BYTES + pos_group * Self.V_PG_BYTES)

    def k_scale_ptr(self, head: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
        self.assert_head(head)
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.k_scale_base + head * Self.max_seq * size_of[Float32]())

    def v_scale_ptr(self, head: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
        self.assert_head(head)
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.v_scale_base + head * Self.max_seq * size_of[Float32]())
