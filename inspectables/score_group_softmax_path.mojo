from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import CompilationTarget, simd_width_of
from std.sys import llvm_intrinsic


comptime WIDTH = simd_width_of[DType.int32]()
comptime HEAD_DIM = 128
comptime NUM_GROUPS = 2
comptime K_BYTES_PER_GROUP = HEAD_DIM * WIDTH
comptime VNNI_BLK = 4
comptime NEG_INF = Float32(-1e30)


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * VNNI_BLK],
    b: SIMD[DType.int8, width * VNNI_BLK],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def dot_score_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed_u8: UnsafePointer[UInt8, MutAnyOrigin],
    q_offset: Int,
) -> SIMD[DType.int32, width]:
    var k = k_packed_u8.bitcast[Scalar[DType.int8]]().load[width = width * VNNI_BLK]()
    var q4 = (q_i8 + q_offset).bitcast[UInt8]().load[width=VNNI_BLK]() ^ SIMD[DType.uint8, VNNI_BLK](0x80)
    var q8 = q4.join(q4)
    comptime if width == 8:
        var q16 = q8.join(q8)
        var q32 = q16.join(q16)
        return vpdpbusd[width](acc, q32.slice[width * VNNI_BLK](), k)
    else:
        var q16 = q8.join(q8)
        var q32 = q16.join(q16)
        var q64 = q32.join(q32)
        return vpdpbusd[width](acc, q64.slice[width * VNNI_BLK](), k)


@always_inline
def dot_score_simd[width: Int](
    acc: SIMD[DType.int32, width],
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed_u8: UnsafePointer[UInt8, MutAnyOrigin],
    q_offset: Int,
) -> SIMD[DType.int32, width]:
    var kdw = k_packed_u8.bitcast[Scalar[DType.int32]]().load[width=width]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(q_i8[q_offset]) + 128) * ((kdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(q_i8[q_offset + 1]) + 128) * ((kdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(q_i8[q_offset + 2]) + 128) * ((kdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(q_i8[q_offset + 3]) + 128) * (kdw >> 24)
    return result


@always_inline
def dot_score[width: Int](
    acc: SIMD[DType.int32, width],
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed_u8: UnsafePointer[UInt8, MutAnyOrigin],
    q_offset: Int,
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return dot_score_vnni[width](acc, q_i8, k_packed_u8, q_offset)
    else:
        return dot_score_simd[width](acc, q_i8, k_packed_u8, q_offset)


@always_inline
def score_group[head_dim: Int](
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pg: UnsafePointer[UInt8, MutAnyOrigin],
    qi_bias: Float32,
    q_factor: Float32,
    k_scales: UnsafePointer[Float32, MutAnyOrigin],
    scores_out: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime K_DIM_GROUPS = head_dim // VNNI_BLK

    var acc = SIMD[DType.int32, WIDTH](0)
    for kdg in range(K_DIM_GROUPS):
        acc = dot_score[WIDTH](
            acc, q_i8,
            k_pg + kdg * WIDTH * VNNI_BLK,
            kdg * VNNI_BLK,
        )
    var corrected = acc.cast[DType.float32]() - qi_bias
    var k_sc = k_scales.load[width=WIDTH]()
    scores_out.store(corrected * q_factor * k_sc)


@always_inline
def exp_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    comptime LN2_HI = Float32(0.693145751953125)
    comptime LN2_LO = Float32(1.4286068203094172e-06)
    comptime INV_LN2 = Float32(1.4426950408889634)
    comptime EXP_LO = Float32(-87.0)
    comptime EXP_HI = Float32(88.0)

    var lo_mask = ((x - EXP_LO).to_bits() >> 31) & 1
    var xc = x * (1 - lo_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_LO) * lo_mask.cast[DType.float32]()
    var hi_mask = ((EXP_HI - xc).to_bits() >> 31) & 1
    xc = xc * (1 - hi_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_HI) * hi_mask.cast[DType.float32]()

    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()

    var nf = n.cast[DType.float32]()
    var r = (xc - nf * LN2_HI) - nf * LN2_LO

    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999999995) + r * (
        Float32(0.5000000004) + r * (
        Float32(0.1666666456) + r * (
        Float32(0.04166685110) + r * (
        Float32(0.008333621758) + r * (
        Float32(0.001389404636)))))))

    var pow2n = SIMD[DType.float32, width](
        from_bits=(n + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n


@always_inline
def lane_mask[width: Int](
    group_start: Int, context_len: Int,
) -> SIMD[DType.bool, width]:
    var lanes = SIMD[DType.int32, width]()
    comptime for lane in range(width):
        lanes[lane] = Int32(group_start + lane)
    return lanes.lt(SIMD[DType.int32, width](context_len))


@always_inline
def scalar_fill_invalid[width: Int](
    scores: UnsafePointer[Float32, MutAnyOrigin],
    group_start: Int,
    context_len: Int,
    fill: Float32,
):
    for lane in range(width):
        if group_start + lane >= context_len:
            scores[lane] = fill


@always_inline
def weighted_sum[width: Int](
    weights: SIMD[DType.float32, width],
    values: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    return (weights * values.load[width=width]()).reduce_add()


@no_inline
def current_score_group_softmax(
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed: UnsafePointer[UInt8, MutAnyOrigin],
    qi_bias: Float32,
    q_factor: Float32,
    k_scales: UnsafePointer[Float32, MutAnyOrigin],
    values: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    var running_max = NEG_INF
    var running_sum = Float32(0)
    var value_sum = Float32(0)
    var scratch_arr = InlineArray[Float32, WIDTH](uninitialized=True)
    var scratch = UnsafePointer(to=scratch_arr).bitcast[Float32]()

    for pg in range(NUM_GROUPS):
        var group_start = pg * WIDTH
        score_group[HEAD_DIM](
            q_i8,
            k_packed + pg * K_BYTES_PER_GROUP,
            qi_bias,
            q_factor,
            k_scales + pg * WIDTH,
            scratch,
        )

        scalar_fill_invalid[WIDTH](scratch, group_start, context_len, NEG_INF)
        var scores_vec = scratch.load[width=WIDTH]()
        var group_max = scores_vec.reduce_max()
        var new_max = max(running_max, group_max)

        if running_sum > 0:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            running_sum *= rescale
            value_sum *= rescale

        running_max = new_max
        var exp_scores = exp_f32[WIDTH](scores_vec - new_max)
        for lane in range(WIDTH):
            if group_start + lane >= context_len:
                exp_scores[lane] = Float32(0)

        running_sum += exp_scores.reduce_add()
        value_sum += weighted_sum[WIDTH](exp_scores, values + pg * WIDTH)

    out_ptr[0] = running_max
    out_ptr[1] = running_sum
    out_ptr[2] = value_sum


@no_inline
def proposed_score_group_softmax(
    q_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_packed: UnsafePointer[UInt8, MutAnyOrigin],
    qi_bias: Float32,
    q_factor: Float32,
    k_scales: UnsafePointer[Float32, MutAnyOrigin],
    values: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    var running_max = NEG_INF
    var running_sum = Float32(0)
    var value_sum = Float32(0)
    var scratch_arr = InlineArray[Float32, WIDTH](uninitialized=True)
    var scratch = UnsafePointer(to=scratch_arr).bitcast[Float32]()
    var neg_inf = SIMD[DType.float32, WIDTH](NEG_INF)
    var zero = SIMD[DType.float32, WIDTH](0)

    for pg in range(NUM_GROUPS):
        var group_start = pg * WIDTH
        score_group[HEAD_DIM](
            q_i8,
            k_packed + pg * K_BYTES_PER_GROUP,
            qi_bias,
            q_factor,
            k_scales + pg * WIDTH,
            scratch,
        )

        var valid = lane_mask[WIDTH](group_start, context_len)
        var scores_vec = valid.select(scratch.load[width=WIDTH](), neg_inf)
        var group_max = scores_vec.reduce_max()
        var new_max = max(running_max, group_max)

        if running_sum > 0:
            var rescale = Float32(exp_f32[1](running_max - new_max))
            running_sum *= rescale
            value_sum *= rescale

        running_max = new_max
        var exp_scores = valid.select(exp_f32[WIDTH](scores_vec - new_max), zero)

        running_sum += exp_scores.reduce_add()
        value_sum += weighted_sum[WIDTH](exp_scores, values + pg * WIDTH)

    out_ptr[0] = running_max
    out_ptr[1] = running_sum
    out_ptr[2] = value_sum


@always_inline
def max_abs_diff(
    lhs: UnsafePointer[Float32, MutAnyOrigin],
    rhs: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) -> Float32:
    var diff = Float32(0)
    for i in range(count):
        var delta = lhs[i] - rhs[i]
        if delta < 0:
            delta = -delta
        diff = max(diff, delta)
    return diff


def run_case(context_len: Int) -> Float32:
    var q_arr = InlineArray[Scalar[DType.int8], HEAD_DIM](fill=Scalar[DType.int8](0))
    var k_arr = InlineArray[UInt8, NUM_GROUPS * K_BYTES_PER_GROUP](fill=UInt8(0))
    var k_scale_arr = InlineArray[Float32, NUM_GROUPS * WIDTH](fill=Float32(0))
    var value_arr = InlineArray[Float32, NUM_GROUPS * WIDTH](fill=Float32(0))
    var current_out_arr = InlineArray[Float32, 3](fill=Float32(0))
    var proposed_out_arr = InlineArray[Float32, 3](fill=Float32(0))

    for i in range(HEAD_DIM):
        q_arr[i] = Scalar[DType.int8]((i * 7 + 13) % 127 - 63)
    for i in range(NUM_GROUPS * K_BYTES_PER_GROUP):
        k_arr[i] = UInt8((i * 5 + 3) % 251)
    for i in range(NUM_GROUPS * WIDTH):
        k_scale_arr[i] = Float32((i * 11 + 5) % 19) * 0.0625 + 0.25
        value_arr[i] = Float32((i * 13 + 9) % 23) * 0.125 - 1.0

    var q_i8 = UnsafePointer(to=q_arr).bitcast[Scalar[DType.int8]]()
    var k_packed = UnsafePointer(to=k_arr).bitcast[UInt8]()
    var k_scales = UnsafePointer(to=k_scale_arr).bitcast[Float32]()
    var values = UnsafePointer(to=value_arr).bitcast[Float32]()
    var current_out = UnsafePointer(to=current_out_arr).bitcast[Float32]()
    var proposed_out = UnsafePointer(to=proposed_out_arr).bitcast[Float32]()

    current_score_group_softmax(
        q_i8, k_packed, Float32(-3.25), Float32(1.0 / 16129.0),
        k_scales, values, current_out, context_len)
    proposed_score_group_softmax(
        q_i8, k_packed, Float32(-3.25), Float32(1.0 / 16129.0),
        k_scales, values, proposed_out, context_len)

    var diff = max_abs_diff(current_out, proposed_out, 3)
    return current_out[0] + current_out[1] + current_out[2] + proposed_out[0] + proposed_out[1] + proposed_out[2] + diff


def main():
    keep(run_case(WIDTH + 5))
