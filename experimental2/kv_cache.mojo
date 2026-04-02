"""Flat KV cache — zero per-token metadata.

K stored as u8 (i8 XOR 0x80) for tdpbsud (signed Q × unsigned K).
V stored as i8 directly for tdpbusd (unsigned W × signed V).
Quantization uses a per-layer fixed scale (not stored here).
Layout: [head][pos][dim] for both K and V.
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of


struct KVCache[max_seq: Int, head_dim: Int, num_heads: Int]:
    var data_base: Int

    comptime HEAD_STRIDE = Self.max_seq * Self.head_dim
    comptime TOTAL_BYTES = Self.num_heads * Self.HEAD_STRIDE

    def __init__(out self, base: Int):
        self.data_base = base

    def write_k(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Int8, MutAnyOrigin],
    ):
        """Write K head: i8 → u8 (XOR 0x80) for tdpbsud unsigned operand."""
        comptime w = simd_width_of[DType.uint8]()
        comptime xor_mask = SIMD[DType.uint8, w](0x80)
        var dst = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)
        var src = data_i8.bitcast[UInt8]()
        var i = 0
        while i + w <= Self.head_dim:
            (dst + i).store(src.load[width=w](offset=i) ^ xor_mask)
            i += w

    def write_v(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Int8, MutAnyOrigin],
    ):
        """Write V head: i8 stored directly for tdpbusd signed operand."""
        comptime w = simd_width_of[DType.int8]()
        var dst = UnsafePointer[Int8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)
        var i = 0
        while i + w <= Self.head_dim:
            (dst + i).store(data_i8.load[width=w](offset=i))
            i += w

    def k_head(self, pos: Int, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """K data pointer: u8 (XOR'd i8) at [head][pos]."""
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)

    def k_head_base(self, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """K data pointer: u8 at [head][0]."""
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE)

    def v_head(self, pos: Int, head: Int) -> UnsafePointer[Int8, MutAnyOrigin]:
        """V data pointer: i8 at [head][pos]."""
        return UnsafePointer[Int8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)

    def v_head_base(self, head: Int) -> UnsafePointer[Int8, MutAnyOrigin]:
        """V data pointer: i8 at [head][0]."""
        return UnsafePointer[Int8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE)
