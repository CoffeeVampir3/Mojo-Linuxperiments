"""Weight initialization — column sums and VNNI packing."""

from std.memory import UnsafePointer, memcpy


def colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int, rows: Int, cols: Int):
    """Compute per-row column sum: colsum[n] = sum_k i8_weight[n, k]."""
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + weight_off)
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum_off)
    for n in range(rows):
        var acc = Int(0)
        for k in range(cols):
            acc += Int(wp[n * cols + k])
        cp[n] = Float32(acc)


def block_colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int,
    rows: Int, cols: Int, block_cols: Int):
    """Per-block column sums in [num_blocks, N] layout.

    colsum[blk * rows + n] = sum_{k in block} W_i8[n, k].
    Transposed so consecutive output rows are contiguous per block.
    """
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + weight_off)
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum_off)
    var num_blocks = cols // block_cols
    for n in range(rows):
        for blk in range(num_blocks):
            var acc = Int(0)
            var k0 = blk * block_cols
            for k in range(block_cols):
                acc += Int(wp[n * cols + k0 + k])
            cp[blk * rows + n] = Float32(acc)


def block_colsum_row_major_at(arena_base: Int, weight_off: Int, colsum_off: Int,
    rows: Int, cols: Int, block_cols: Int):
    """Per-block column sums in [N, num_blocks] row-major layout.

    colsum[n * num_blocks + blk] = sum_{k in block} W_i8[n, k].
    """
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + weight_off)
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + colsum_off)
    var num_blocks = cols // block_cols
    for n in range(rows):
        for blk in range(num_blocks):
            var acc = Int(0)
            var k0 = blk * block_cols
            for k in range(block_cols):
                acc += Int(wp[n * cols + k0 + k])
            cp[n * num_blocks + blk] = Float32(acc)


def pack_at(arena_base: Int, weight_off: Int, rows: Int, cols: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """VNNI-pack weight in-place using scratch buffer."""
    from kernels.vnni import pack_vnni
    var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=arena_base + weight_off)
    memcpy(dest=scratch, src=src, count=rows * cols)
    pack_vnni(scratch, src, rows, cols)
