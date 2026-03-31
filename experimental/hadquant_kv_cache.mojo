"""Per-head contiguous KV cache with u8 storage for native vpdpbusd.

Data is stored as u8 (i8 XOR'd with 0x80 at write time). This allows
vpdpbusd to use cache entries directly as the unsigned operand, with
the per-head query/weight vector as the signed operand. The bias
correction is 128 * sum(signed_operand) — computed once per head,
not per cache entry.

Layout per cache (K or V separately):
  data:  [head0: MAX_SEQ * HEAD_DIM u8 bytes][head1]...[headN]
  scale: [head0: MAX_SEQ * 4 f32 bytes][head1]...[headN]

Write: accepts i8 data, XORs to u8 internally.
Read: returns u8 pointers.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import size_of, simd_width_of


struct HadQuantKVCache[max_seq: Int, head_dim: Int, num_heads: Int]:
    """Unified KV cache view with u8 storage convention.

    Stores quantized values as u8 (i8 ^ 0x80) for direct use with
    vpdpbusd. The write path XORs on store; the read path returns
    raw u8 bytes. Scale is f32, unchanged.
    """
    var data_base: Int
    var scale_base: Int

    comptime DATA_HEAD_STRIDE = Self.max_seq * Self.head_dim
    comptime SCALE_HEAD_STRIDE = Self.max_seq * size_of[Float32]()
    comptime TOTAL_DATA_BYTES = Self.num_heads * Self.DATA_HEAD_STRIDE
    comptime TOTAL_SCALE_BYTES = Self.num_heads * Self.SCALE_HEAD_STRIDE
    comptime TOTAL_BYTES = Self.TOTAL_DATA_BYTES + Self.TOTAL_SCALE_BYTES

    def __init__(out self, base: Int):
        """Construct from a base address. Data first, then scales."""
        self.data_base = base
        self.scale_base = base + Self.TOTAL_DATA_BYTES

    # --- Write ---

    def write_head(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
        scale: Float32,
    ):
        """Write HEAD_DIM values + f32 scale for one head at one position.

        Accepts i8 data from the quantizer. XORs each byte with 0x80
        to convert to u8 representation before storing.
        """
        var dst = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.DATA_HEAD_STRIDE + pos * Self.head_dim
        )
        # XOR i8 → u8 via dword XOR (4 bytes at a time)
        var src_dw = data_i8.bitcast[Scalar[DType.uint32]]()
        var dst_dw = dst.bitcast[Scalar[DType.uint32]]()
        comptime dwords = Self.head_dim // 4
        for i in range(dwords):
            dst_dw[i] = src_dw[i] ^ UInt32(0x80808080)

        var sp = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.scale_base + head * Self.SCALE_HEAD_STRIDE + pos * size_of[Float32]()
        )
        sp[0] = scale

    # --- Read ---

    def head_data(self, pos: Int, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Pointer to HEAD_DIM u8 values for one head at one position."""
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.DATA_HEAD_STRIDE + pos * Self.head_dim
        )

    def head_scale(self, pos: Int, head: Int) -> Float32:
        """F32 scale for one head at one position."""
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.scale_base + head * Self.SCALE_HEAD_STRIDE + pos * size_of[Float32]()
        )[0]

    def head_data_base(self, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Pointer to start of one head's contiguous [MAX_SEQ, HEAD_DIM] u8 block."""
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.DATA_HEAD_STRIDE
        )

    def head_scale_base(self, head: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Pointer to start of one head's contiguous [MAX_SEQ] f32 scale vector."""
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.scale_base + head * Self.SCALE_HEAD_STRIDE
        )

    # --- Transposed layout: [head][dim][pos] ---
    # Same total memory, but data is rearranged so that for a fixed
    # dimension, positions are contiguous. This enables VNNI dot products
    # over the position (time) dimension during V aggregation.

    def write_head_transposed(self, pos: Int, head: Int,
        data_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
        scale: Float32,
    ):
        """Write HEAD_DIM values in transposed layout: data[head][dim][pos].

        Each byte is XOR'd with 0x80 (i8 -> u8) and written to its
        transposed location. Amortized over many reads per write.
        """
        var head_base = self.data_base + head * Self.DATA_HEAD_STRIDE
        for d in range(Self.head_dim):
            UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=head_base + d * Self.max_seq + pos
            )[0] = UInt8(Int(data_i8[d]) ^ 0x80)

        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.scale_base + head * Self.SCALE_HEAD_STRIDE + pos * size_of[Float32]()
        )[0] = scale

    def dim_data(self, dim: Int, head: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Pointer to MAX_SEQ contiguous u8 values for one dim of one head.

        In transposed layout [head][dim][pos], positions are contiguous
        for a fixed dimension, enabling VNNI reduction over time.
        """
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.data_base + head * Self.DATA_HEAD_STRIDE + dim * Self.max_seq
        )
