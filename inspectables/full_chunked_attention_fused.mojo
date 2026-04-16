from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime WIDTH = simd_width_of[DType.int32]()
comptime HEAD_DIM = 128
comptime NUM_GROUPS = 4
comptime MAX_CONTEXT = NUM_GROUPS * WIDTH
comptime NEG_INF = Float32(-1e30)


@always_inline
def lane_mask[width: Int](group_start: Int, context_len: Int) -> SIMD[DType.bool, width]:
    var lanes = SIMD[DType.int32, width]()
    comptime for lane in range(width):
        lanes[lane] = Int32(group_start + lane)
    return lanes.lt(SIMD[DType.int32, width](context_len))


@always_inline
def score_vec[heads_per_group: Int, width: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    bias: Float32,
    head_index: Int,
    pg: Int,
) -> SIMD[DType.float32, width]:
    var base = score_src + (head_index * NUM_GROUPS + pg) * width
    return base.load[width=width]() + (scales + pg * width).load[width=width]() * 0.25 + bias


@always_inline
def fill_invalid_scalar[width: Int](
    scores: UnsafePointer[Float32, MutAnyOrigin],
    group_start: Int,
    context_len: Int,
    fill: Float32,
):
    for lane in range(width):
        if group_start + lane >= context_len:
            scores[lane] = fill


@always_inline
def rescale_accumulator[width: Int](
    acc: UnsafePointer[Float32, MutAnyOrigin],
    rescale: Float32,
):
    var d = 0
    while d + width <= HEAD_DIM:
        (acc + d).store((acc + d).load[width=width]() * rescale)
        d += width


@always_inline
def accumulate_group[width: Int](
    weights: SIMD[DType.float32, width],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc: UnsafePointer[Float32, MutAnyOrigin],
    pg: Int,
):
    var alpha = weights.reduce_add()
    var src = v_src + pg * HEAD_DIM
    var d = 0
    while d + width <= HEAD_DIM:
        (acc + d).store((src + d).load[width=width]().fma(
            SIMD[DType.float32, width](alpha), (acc + d).load[width=width]()))
        d += width


@always_inline
def finalize_head(
    acc: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    running_max: Float32,
    running_sum: Float32,
    head_index: Int,
):
    stats[head_index * 2] = running_max
    stats[head_index * 2 + 1] = running_sum


@always_inline
def process_head_scalar_mask[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
    head_index: Int,
):
    var running_max = NEG_INF
    var running_sum = Float32(0)
    var scratch_arr = InlineArray[Float32, WIDTH](uninitialized=True)
    var scratch = UnsafePointer(to=scratch_arr).bitcast[Float32]()
    var acc = acc_base + head_index * HEAD_DIM

    for pg in range(NUM_GROUPS):
        var group_start = pg * WIDTH
        scratch.store(score_vec[heads_per_group, WIDTH](
            score_src, scales, biases[head_index], head_index, pg))
        fill_invalid_scalar[WIDTH](scratch, group_start, context_len, NEG_INF)
        var masked_scores = scratch.load[width=WIDTH]()

        var group_max = masked_scores.reduce_max()
        var new_max = max(running_max, group_max)
        if running_sum > 0:
            var rescale = Float32(1.0) / (Float32(1.0) + new_max - running_max)
            rescale_accumulator[WIDTH](acc, rescale)
            running_sum *= rescale

        running_max = new_max
        scratch.store(masked_scores - SIMD[DType.float32, WIDTH](new_max) + 1.0)
        fill_invalid_scalar[WIDTH](scratch, group_start, context_len, Float32(0))
        var weights = scratch.load[width=WIDTH]()
        running_sum += weights.reduce_add()
        accumulate_group[WIDTH](weights, v_src, acc, pg)

    finalize_head(acc, stats, running_max, running_sum, head_index)


@always_inline
def process_head_simd_mask[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
    head_index: Int,
):
    var running_max = NEG_INF
    var running_sum = Float32(0)
    var acc = acc_base + head_index * HEAD_DIM

    for pg in range(NUM_GROUPS):
        var group_start = pg * WIDTH
        var valid = lane_mask[WIDTH](group_start, context_len)
        var masked_scores = valid.select(
            score_vec[heads_per_group, WIDTH](
                score_src, scales, biases[head_index], head_index, pg),
            SIMD[DType.float32, WIDTH](NEG_INF),
        )

        var group_max = masked_scores.reduce_max()
        var new_max = max(running_max, group_max)
        if running_sum > 0:
            var rescale = Float32(1.0) / (Float32(1.0) + new_max - running_max)
            rescale_accumulator[WIDTH](acc, rescale)
            running_sum *= rescale

        running_max = new_max
        var weights = valid.select(
            masked_scores - SIMD[DType.float32, WIDTH](new_max) + 1.0,
            SIMD[DType.float32, WIDTH](0),
        )
        running_sum += weights.reduce_add()
        accumulate_group[WIDTH](weights, v_src, acc, pg)

    finalize_head(acc, stats, running_max, running_sum, head_index)


@no_inline
def current_runtime_heads[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    for qh in range(heads_per_group):
        process_head_scalar_mask[heads_per_group](
            score_src, scales, biases, v_src, acc_base, stats, context_len, qh)


@no_inline
def simd_mask_runtime_heads[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    for qh in range(heads_per_group):
        process_head_simd_mask[heads_per_group](
            score_src, scales, biases, v_src, acc_base, stats, context_len, qh)


@no_inline
def current_comptime_heads[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    comptime for qh in range(heads_per_group):
        process_head_scalar_mask[heads_per_group](
            score_src, scales, biases, v_src, acc_base, stats, context_len, qh)


@no_inline
def simd_mask_comptime_heads[heads_per_group: Int](
    score_src: UnsafePointer[Float32, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    v_src: UnsafePointer[Float32, MutAnyOrigin],
    acc_base: UnsafePointer[Float32, MutAnyOrigin],
    stats: UnsafePointer[Float32, MutAnyOrigin],
    context_len: Int,
):
    comptime for qh in range(heads_per_group):
        process_head_simd_mask[heads_per_group](
            score_src, scales, biases, v_src, acc_base, stats, context_len, qh)


def run_variant[heads_per_group: Int, mode: Int]() -> Float32:
    comptime SCORE_COUNT = heads_per_group * NUM_GROUPS * WIDTH
    comptime ACC_COUNT = heads_per_group * HEAD_DIM
    comptime STAT_COUNT = heads_per_group * 2

    var score_arr = InlineArray[Float32, SCORE_COUNT](fill=Float32(0))
    var scale_arr = InlineArray[Float32, NUM_GROUPS * WIDTH](fill=Float32(0))
    var bias_arr = InlineArray[Float32, heads_per_group](fill=Float32(0))
    var v_arr = InlineArray[Float32, NUM_GROUPS * HEAD_DIM](fill=Float32(0))
    var acc_arr = InlineArray[Float32, ACC_COUNT](fill=Float32(0))
    var stat_arr = InlineArray[Float32, STAT_COUNT](fill=Float32(0))

    for i in range(SCORE_COUNT):
        score_arr[i] = Float32((i * 7 + 3) % 29) * 0.125 - 1.75
    for i in range(NUM_GROUPS * WIDTH):
        scale_arr[i] = Float32((i * 5 + 11) % 17) * 0.0625 + 0.25
    for i in range(heads_per_group):
        bias_arr[i] = Float32(i) * 0.03125 - 0.125
    for i in range(NUM_GROUPS * HEAD_DIM):
        v_arr[i] = Float32((i * 13 + 1) % 31) * 0.03125 - 0.5

    var score_src = UnsafePointer(to=score_arr).bitcast[Float32]()
    var scales = UnsafePointer(to=scale_arr).bitcast[Float32]()
    var biases = UnsafePointer(to=bias_arr).bitcast[Float32]()
    var v_src = UnsafePointer(to=v_arr).bitcast[Float32]()
    var acc = UnsafePointer(to=acc_arr).bitcast[Float32]()
    var stats = UnsafePointer(to=stat_arr).bitcast[Float32]()
    var context_len = MAX_CONTEXT - 3

    comptime if mode == 0:
        current_runtime_heads[heads_per_group](
            score_src, scales, biases, v_src, acc, stats, context_len)
    elif mode == 1:
        simd_mask_runtime_heads[heads_per_group](
            score_src, scales, biases, v_src, acc, stats, context_len)
    elif mode == 2:
        current_comptime_heads[heads_per_group](
            score_src, scales, biases, v_src, acc, stats, context_len)
    else:
        simd_mask_comptime_heads[heads_per_group](
            score_src, scales, biases, v_src, acc, stats, context_len)

    var total = Float32(0)
    for i in range(ACC_COUNT):
        total += acc[i]
    for i in range(STAT_COUNT):
        total += stats[i]
    return total


def run_current_runtime_8() -> Float32:
    return run_variant[8, 0]()


def run_simd_mask_runtime_8() -> Float32:
    return run_variant[8, 1]()


def run_current_comptime_8() -> Float32:
    return run_variant[8, 2]()


def run_simd_mask_comptime_8() -> Float32:
    return run_variant[8, 3]()


def run_current_runtime_16() -> Float32:
    return run_variant[16, 0]()


def run_simd_mask_runtime_16() -> Float32:
    return run_variant[16, 1]()


def run_current_comptime_16() -> Float32:
    return run_variant[16, 2]()


def run_simd_mask_comptime_16() -> Float32:
    return run_variant[16, 3]()


def main():
    keep(run_current_runtime_8())
    keep(run_simd_mask_runtime_8())
    keep(run_current_comptime_8())
    keep(run_simd_mask_comptime_8())
    keep(run_current_runtime_16())
    keep(run_simd_mask_runtime_16())
    keep(run_current_comptime_16())
    keep(run_simd_mask_comptime_16())
