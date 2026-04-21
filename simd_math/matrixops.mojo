from std.collections import InlineArray
from std.memory import UnsafePointer
from std.utils import IndexList
from std.math import iota


def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()


def bit_reverse[bits: Int, x: Int]() -> Int:
    comptime if bits == 0:
        return 0
    else:
        comptime lsb = x & 1
        comptime rest = x >> 1
        return (lsb << (bits - 1)) | bit_reverse[bits - 1, rest]()


def interleave_idx[N: Int, i: Int, stride: Int, high: Bool]() -> Int:
    comptime half = N // 2
    comptime src_offset = half if high else 0
    comptime pair = i // (2 * stride)
    comptime within = i % (2 * stride)
    comptime if within < stride:
        return src_offset + pair * stride + within
    else:
        return N + src_offset + pair * stride + (within - stride)


def interleave_mask[N: Int, stride: Int, high: Bool]() -> IndexList[N]:
    var result = IndexList[N]()
    comptime for i in range(N):
        result[i] = interleave_idx[N, i, stride, high]()
    return result


@always_inline
def simd_interleave[T: DType, N: Int, stride: Int, high: Bool](
    a: SIMD[T, N], b: SIMD[T, N],
) -> SIMD[T, N]:
    comptime idx = interleave_mask[N, stride, high]()
    return a.shuffle[mask=idx](b)


@always_inline
def transpose_rows[T: DType, N: Int](
    mut rows: InlineArray[SIMD[T, N], N],
    dst: UnsafePointer[Scalar[T], MutAnyOrigin], dst_stride: Int,
):
    """Butterfly transpose pre-loaded rows and store to dst.

    Rows are modified in-place during the butterfly stages.
    Caller loads rows (full or partial with zero padding).
    """
    comptime num_stages = log2[N]()
    comptime for stage in range(num_stages):
        comptime stride = 1 << stage
        comptime groups = N // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime idx0 = g * 2 * stride + j
                comptime idx1 = idx0 + stride
                var lo = simd_interleave[T, N, stride, False](rows[idx0], rows[idx1])
                var hi = simd_interleave[T, N, stride, True](rows[idx0], rows[idx1])
                rows[idx0] = lo
                rows[idx1] = hi

    comptime for i in range(N):
        comptime j = bit_reverse[num_stages, i]()
        comptime if i < j:
            var tmp = rows[i]
            rows[i] = rows[j]
            rows[j] = tmp

    comptime for i in range(N):
        (dst + i * dst_stride).store(rows[i])


@always_inline
def transpose_generic[T: DType, N: Int](
    src: UnsafePointer[Scalar[T], _], src_stride: Int,
    dst: UnsafePointer[Scalar[T], MutAnyOrigin], dst_stride: Int,
    mut scratch: InlineArray[SIMD[T, N], N],
):
    """In-register NxN transpose via butterfly interleave network.

    Generic over element type: int8 for byte transpose, int32 for dword.
    Loads N rows of N elements from src (strided by elements), performs
    log2(N) stages of interleave shuffles, then stores N rows to dst.
    """
    comptime for i in range(N):
        scratch[i] = (src + i * src_stride).load[width=N]()
    transpose_rows[T, N](scratch, dst, dst_stride)


def butterfly_partner[i: Int, stride: Int]() -> Int:
    return i ^ stride


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result


@always_inline
def reduce_argmax[T: DType, width: Int, regs: Int](
    mut values: InlineArray[SIMD[T, width], regs],
    mut indices: InlineArray[SIMD[DType.int32, width], regs],
) -> Tuple[Scalar[T], Int32]:
    """Butterfly argmax reduction of a tagged value bank.

    Mutates `values` and `indices`. Returns (max_value, argmax_index) —
    the lane-0 winner after log2(regs) across-register stages followed by
    log2(width) in-lane stages. Tie-break favors the smaller index.

    `regs` must be a power of 2 (>= 1). `width` must be a power of 2.
    When `regs == 1` phase A collapses to a no-op and only the in-lane
    butterfly runs.
    """
    comptime stages_across = log2[regs]()
    comptime stages_within = log2[width]()

    # Phase A: across-register pairwise reduction.
    comptime for stage in range(stages_across):
        comptime stride = 1 << stage
        comptime groups = regs // (2 * stride)
        comptime for g in range(groups):
            comptime for j in range(stride):
                comptime a = g * 2 * stride + j
                comptime b = a + stride
                var mask = values[a].ge(values[b])
                values[a] = mask.select(values[a], values[b])
                indices[a] = mask.select(indices[a], indices[b])

    # Phase B: in-lane butterfly on reg 0.
    comptime for stage in range(stages_within):
        comptime stride = 1 << stage
        comptime shuf_mask = butterfly_shuffle[width, stride]()
        var partner_v = values[0].shuffle[mask=shuf_mask](values[0])
        var partner_i = indices[0].shuffle[mask=shuf_mask](indices[0])
        var cmp = values[0].ge(partner_v)
        values[0] = cmp.select(values[0], partner_v)
        indices[0] = cmp.select(indices[0], partner_i)

    return (values[0][0], indices[0][0])


@always_inline
def reduce_top_k[T: DType, width: Int, regs: Int, k: Int](
    source_values: InlineArray[SIMD[T, width], regs],
    source_indices: InlineArray[SIMD[DType.int32, width], regs],
    sentinel: Scalar[T],
    mut out_indices: InlineArray[Int, k],
    mut out_values: InlineArray[Scalar[T], k],
):
    """Extract top-k (value, index) pairs in descending order.

    Source banks are copied into an internal workspace; caller's inputs are
    preserved. Between extractions the winner's lane is set to `sentinel`
    in the workspace so it cannot be reselected; `sentinel` must be strictly
    less than any real value in `source_values`.

    Results are written in descending-value order into `out_indices` and
    `out_values`. Each call runs k independent butterfly reductions plus a
    one-lane mask update between them — O(k · log(regs · width)) SIMD ops.
    """
    # Persistent workspace, written fully from source on the next line.
    var work_v = InlineArray[SIMD[T, width], regs](uninitialized=True)
    var work_i = InlineArray[SIMD[DType.int32, width], regs](uninitialized=True)
    comptime for r in range(regs):
        work_v[r] = source_values[r]
        work_i[r] = source_indices[r]

    var lane_iota = iota[DType.int32, width]()
    var sentinel_vec = SIMD[T, width](sentinel)

    for sel in range(k):
        # reduce_argmax consumes its inputs; copy work into tournament buffers.
        var tv = InlineArray[SIMD[T, width], regs](uninitialized=True)
        var ti = InlineArray[SIMD[DType.int32, width], regs](uninitialized=True)
        comptime for r in range(regs):
            tv[r] = work_v[r]
            ti[r] = work_i[r]

        var winner = reduce_argmax[T, width, regs](tv, ti)
        out_values[sel] = winner[0]
        var winner_idx = Int(winner[1])
        out_indices[sel] = winner_idx

        # Mask the winner lane in the persistent workspace.
        var wr = winner_idx // width
        var wl = winner_idx - wr * width
        var lane_eq = lane_iota.eq(SIMD[DType.int32, width](Int32(wl)))
        work_v[wr] = lane_eq.select(sentinel_vec, work_v[wr])


@always_inline
def fill_lane_iota[width: Int, regs: Int](
    mut indices: InlineArray[SIMD[DType.int32, width], regs],
):
    """Initialize an index bank so lane `w` of register `r` holds `r*width + w`.
    """
    comptime for r in range(regs):
        indices[r] = iota[DType.int32, width](offset=Int32(r * width))


# ============================================================================
# Port-saturating accumulation
#
# Two classes of accumulation live here:
#   A) Reduction along a SIMD axis — `cols` comptime, step = port_unroll*width.
#      Picker: pick_port_unroll[width, cols]().
#   B) Reduction along a non-SIMD axis (experts, chunks) — step = port_unroll.
#      Picker: port_unroll_for[count](), where count is the comptime bound.
#
# Both classes merge their register banks via tree_reduce_accs.
# ============================================================================


@always_inline
def port_unroll_for[count: Int]() -> Int:
    """Largest power-of-two N in {1,2,4,8} with N <= count."""
    return 8 if count >= 8 else 4 if count >= 4 else 2 if count >= 2 else 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    """Class-A picker: chains that fit within `cols` at `width` lanes each."""
    return port_unroll_for[cols // width]()


@always_inline
def tree_reduce_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> Scalar[T]:
    """Pairwise-add accumulator bank into lane-0, then horizontal reduce."""
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


@always_inline
def tree_merge_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> SIMD[T, width]:
    """Pairwise-add accumulator bank into lane-0 as a vector (no horizontal reduce).

    Used when the caller wants to keep the merged SIMD vector (e.g. to cast and
    store) rather than reduce to a scalar.
    """
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0]
