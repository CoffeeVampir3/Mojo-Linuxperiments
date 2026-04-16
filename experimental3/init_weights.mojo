"""Weight initialization — column sums and VNNI packing."""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from kernels.vnni import pack_vnni


@always_inline
def arena_ptr_at[dtype: DType](
    arena_base: Int,
    offset: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    return UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=arena_base + offset)


@always_inline
def sum_i8_span(
    src: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    count: Int,
) -> Float32:
    comptime width = simd_width_of[DType.int32]()
    var lane_sums = SIMD[DType.int32, width](0)
    var k = 0
    while k + width <= count:
        lane_sums += (src + k).load[width=width]().cast[DType.int32]()
        k += width

    var total = Int(lane_sums.reduce_add())
    while k < count:
        total += Int(src[k])
        k += 1
    return Float32(total)


@always_inline
def write_block_colsum[row_major: Bool](
    dst: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    num_blocks: Int,
    row: Int,
    block_idx: Int,
    value: Float32,
):
    comptime if row_major:
        dst[row * num_blocks + block_idx] = value
    else:
        dst[block_idx * rows + row] = value


@always_inline
def block_colsum_impl[row_major: Bool](
    arena_base: Int,
    weight_off: Int,
    colsum_off: Int,
    rows: Int,
    cols: Int,
    block_cols: Int,
):
    """Compute per-block row sums with either row-major or block-major output."""
    debug_assert(block_cols > 0, "block_colsum_impl: block_cols must be positive")
    debug_assert(cols % block_cols == 0,
        "block_colsum_impl: cols must be a multiple of block_cols")
    var weights = arena_ptr_at[DType.int8](arena_base, weight_off)
    var colsums = arena_ptr_at[DType.float32](arena_base, colsum_off)
    var num_blocks = cols // block_cols

    for row in range(rows):
        var weight_row = weights + row * cols
        for block_idx in range(num_blocks):
            var block_start = block_idx * block_cols
            write_block_colsum[row_major](
                colsums,
                rows,
                num_blocks,
                row,
                block_idx,
                sum_i8_span(weight_row + block_start, block_cols),
            )


def colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int, rows: Int, cols: Int):
    """Compute per-row column sum: colsum[n] = sum_k i8_weight[n, k]."""
    block_colsum_impl[True](arena_base, weight_off, colsum_off, rows, cols, cols)


def block_colsum_at(arena_base: Int, weight_off: Int, colsum_off: Int,
    rows: Int, cols: Int, block_cols: Int):
    """Per-block column sums in [num_blocks, N] layout.

    colsum[blk * rows + n] = sum_{k in block} W_i8[n, k].
    Transposed so consecutive output rows are contiguous per block.
    """
    block_colsum_impl[False](arena_base, weight_off, colsum_off, rows, cols, block_cols)


def block_colsum_row_major_at(arena_base: Int, weight_off: Int, colsum_off: Int,
    rows: Int, cols: Int, block_cols: Int):
    """Per-block column sums in [N, num_blocks] row-major layout.

    colsum[n * num_blocks + blk] = sum_{k in block} W_i8[n, k].
    """
    block_colsum_impl[True](arena_base, weight_off, colsum_off, rows, cols, block_cols)


def pack_at(arena_base: Int, weight_off: Int, rows: Int, cols: Int,
    scratch: UnsafePointer[UInt8, MutAnyOrigin]):
    """VNNI-pack weight in-place using scratch buffer."""
    var src = arena_ptr_at[DType.uint8](arena_base, weight_off)
    memcpy(dest=scratch, src=src, count=rows * cols)
    pack_vnni(scratch, src, rows, cols)
