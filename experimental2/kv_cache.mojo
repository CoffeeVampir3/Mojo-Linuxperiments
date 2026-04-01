"""Flat u8 KV cache — zero per-token metadata.

Data stored as u8 (i8 XOR 0x80) for direct VNNI compatibility.
Quantization uses a per-layer fixed scale (not stored here).
Layout: [head][pos][dim] for both K and V (standard, identical).
"""

from std.memory import UnsafePointer
from std.sys.info import size_of


struct KVCache[max_seq: Int, head_dim: Int, num_heads: Int]:
    var data_base: Int

    comptime HEAD_STRIDE = Self.max_seq * Self.head_dim
    comptime TOTAL_BYTES = Self.num_heads * Self.HEAD_STRIDE

    def __init__(out self, base: Int):
        self.data_base = base

    def write_head(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ):
        """Write head_dim i8 values, XOR'd to u8 for storage."""
        var dst = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)
        var src_dw = data_i8.bitcast[Scalar[DType.uint32]]()
        var dst_dw = dst.bitcast[Scalar[DType.uint32]]()
        comptime dwords = Self.head_dim // 4
        for i in range(dwords):
            dst_dw[i] = src_dw[i] ^ UInt32(0x80808080)

    def head_data(self, pos: Int, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE + pos * Self.head_dim)

    def head_data_base(self, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.HEAD_STRIDE)
