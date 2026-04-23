"""Weight initialization — column sums, VNNI packing, padding."""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from kernels.vnni import pack_vnni

from modeling.model_spec import Encoding, ShapeLike
from modeling.modeling_common import TensorRef


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


@always_inline
def colsum_at[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
):
    """Per-row column sum: colsum[n] = sum_k i8_weight[n, k]."""
    block_colsum_impl[True](base, weight.offset, colsum.offset, S.N, S.M, S.M)


@always_inline
def block_colsum_at[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
    block_cols: Int,
):
    """Per-block column sums in [num_blocks, N] layout.

    colsum[blk * rows + n] = sum_{k in block} W_i8[n, k].
    """
    block_colsum_impl[False](base, weight.offset, colsum.offset, S.N, S.M, block_cols)


@always_inline
def block_colsum_row_major_at[
    E: Encoding, S: ShapeLike,
    CE: Encoding, CS: ShapeLike,
](
    base: Int,
    weight: TensorRef[E, S],
    colsum: TensorRef[CE, CS],
    block_cols: Int,
):
    """Per-block column sums in [N, num_blocks] row-major layout.

    colsum[n * num_blocks + blk] = sum_{k in block} W_i8[n, k].
    """
    block_colsum_impl[True](base, weight.offset, colsum.offset, S.N, S.M, block_cols)


@always_inline
def pack_at[E: Encoding, S: ShapeLike](
    base: Int,
    weight: TensorRef[E, S],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
):
    """VNNI-pack weight in-place using scratch buffer. Uses S.N rows x S.M cols."""
    var src = arena_ptr_at[DType.uint8](base, weight.offset)
    memcpy(dest=scratch, src=src, count=S.N * S.M)
    pack_vnni(scratch, src, S.N, S.M)


@always_inline
def zero_pad_tail[E: Encoding, S: ShapeLike](base: Int, slot: TensorRef[E, S]):
    """Zero the alignment padding rows at the end of a weight matrix.
    Uses S.DATA_N (logical rows) and S.N (padded rows). No-op when unpadded."""
    if S.DATA_N >= S.N:
        return
    var start = base + slot.offset + S.DATA_N * S.M * E.ELEMENT_BYTES
    var nbytes = (S.N - S.DATA_N) * S.M * E.ELEMENT_BYTES
    comptime width = simd_width_of[DType.uint8]()
    var ptr = UnsafePointer[Scalar[DType.uint8], MutAnyOrigin](unsafe_from_address=start)
    var i = 0
    while i + width <= nbytes:
        (ptr + i).store(SIMD[DType.uint8, width](0))
        i += width
    while i < nbytes:
        ptr[i] = 0
        i += 1
