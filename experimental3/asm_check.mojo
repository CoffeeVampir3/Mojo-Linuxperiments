"""Minimal GEMV kernel reproduction for assembly inspection."""

from memory import UnsafePointer
from sys.info import simd_width_of
from benchmark import keep


@no_inline
fn gemv_tiled[K: Int, N: Int](
    input_ptr: Int, weight_ptr: Int, output_ptr: Int,
    start_col: Int, end_col: Int, seq_len: Int,
):
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=input_ptr
    )
    var wp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=weight_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=output_ptr
    )
    comptime width = simd_width_of[DType.float32]()
    comptime Mr = 4

    var m_full = (seq_len // Mr) * Mr
    for n in range(start_col, end_col):
        var row_w = wp + n * K

        for m_base in range(0, m_full, Mr):
            var acc0 = SIMD[DType.float32, width](0)
            var acc1 = SIMD[DType.float32, width](0)
            var acc2 = SIMD[DType.float32, width](0)
            var acc3 = SIMD[DType.float32, width](0)
            var r0 = ip + m_base * K
            var r1 = ip + (m_base + 1) * K
            var r2 = ip + (m_base + 2) * K
            var r3 = ip + (m_base + 3) * K
            for k in range(0, K, width):
                var w = (row_w + k).load[width=width]().cast[DType.float32]()
                acc0 = (r0 + k).load[width=width]().cast[DType.float32]().fma(w, acc0)
                acc1 = (r1 + k).load[width=width]().cast[DType.float32]().fma(w, acc1)
                acc2 = (r2 + k).load[width=width]().cast[DType.float32]().fma(w, acc2)
                acc3 = (r3 + k).load[width=width]().cast[DType.float32]().fma(w, acc3)
            (dp + m_base * N)[n] = acc0.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 1) * N)[n] = acc1.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 2) * N)[n] = acc2.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 3) * N)[n] = acc3.reduce_add().cast[DType.bfloat16]()

        for m in range(m_full, seq_len):
            var row_in = ip + m * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var w = (row_w + k).load[width=width]().cast[DType.float32]()
                acc = (row_in + k).load[width=width]().cast[DType.float32]().fma(w, acc)
            (dp + m * N)[n] = acc.reduce_add().cast[DType.bfloat16]()


fn main():
    # Just pass dummy addresses — @no_inline prevents DCE
    gemv_tiled[576, 1536](0x1000000, 0x2000000, 0x3000000, 0, 96, 5)
