"""Expert-output accumulation: serial vs port-saturated variants.

Shape mirrors `accumulate_expert_outputs` (experimental3/kernels/rmsnorm.mojo):
per width-wide output chunk, sum `local_count` bf16 expert outputs (stride =
hidden) into f32 and store as bf16.

Build:
    pixi run mojo build -D ASSERT=none expert_sum_port_saturation.mojo

Inspect (look for the `_accumulate_*` symbols):
    objdump -d expert_sum_port_saturation | less

What to compare:
  * serial: one f32 accumulator register; vaddps chain of depth local_count
  * port4:  four f32 accumulator registers; vaddps chains of depth ceil(local_count/4)
          plus a small tree-merge at the tail
"""

from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep


comptime HIDDEN = 3072        # MiniMax M2.7 hidden dim.
comptime LOCAL_COUNT = 8      # experts per rank (TOP_K=8 at tp=1).
comptime WIDTH = 16           # AVX-512 f32 SIMD width.
comptime PORT_UNROLL = 4      # independent acc chains for the port-sat variant.


@no_inline
def accumulate_serial(
    expert_buf: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    """Single-accumulator chain across experts — mirrors current kernel."""
    for i in range(0, HIDDEN, WIDTH):
        var acc = SIMD[DType.float32, WIDTH](0)
        for e in range(LOCAL_COUNT):
            acc += (expert_buf + e * HIDDEN + i).load[width=WIDTH]().cast[DType.float32]()
        (dst + i).store(acc.cast[DType.bfloat16]())


@no_inline
def accumulate_port_sat(
    expert_buf: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    """PORT_UNROLL independent accumulators; tree-merged at the tail.

    Expert indices are unrolled by PORT_UNROLL at comptime. For LOCAL_COUNT=8
    and PORT_UNROLL=4, each output chunk needs 2 adds per chain (8 total) on
    4 parallel chains plus a 2-stage tree merge.
    """
    comptime main_iters = LOCAL_COUNT // PORT_UNROLL
    comptime tail = LOCAL_COUNT - main_iters * PORT_UNROLL

    for i in range(0, HIDDEN, WIDTH):
        var accs = InlineArray[SIMD[DType.float32, WIDTH], PORT_UNROLL](
            uninitialized=True)
        comptime for u in range(PORT_UNROLL):
            accs[u] = (expert_buf + u * HIDDEN + i).load[width=WIDTH]().cast[DType.float32]()

        comptime for step in range(1, main_iters):
            comptime for u in range(PORT_UNROLL):
                accs[u] += (expert_buf + (step * PORT_UNROLL + u) * HIDDEN + i).load[width=WIDTH]().cast[DType.float32]()

        comptime for u in range(tail):
            accs[u] += (expert_buf + (main_iters * PORT_UNROLL + u) * HIDDEN + i).load[width=WIDTH]().cast[DType.float32]()

        comptime for stride in range(1, PORT_UNROLL):
            comptime if (stride & (stride - 1)) == 0:
                comptime for u in range(0, PORT_UNROLL, 2 * stride):
                    accs[u] += accs[u + stride]

        (dst + i).store(accs[0].cast[DType.bfloat16]())


@always_inline
def init_inputs(
    expert_buf: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    var x = Int32(17)
    for e in range(LOCAL_COUNT):
        for i in range(HIDDEN):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            var f = Float32(x & 0xFFFF) / Float32(0x10000) - Float32(0.5)
            (expert_buf + e * HIDDEN + i)[] = Scalar[DType.bfloat16](f)
    for i in range(HIDDEN):
        (dst + i)[] = Scalar[DType.bfloat16](Float32(0))


@no_inline
def run_case() -> Float32:
    var expert_arr = InlineArray[Scalar[DType.bfloat16], LOCAL_COUNT * HIDDEN](
        uninitialized=True)
    var dst_serial_arr = InlineArray[Scalar[DType.bfloat16], HIDDEN](uninitialized=True)
    var dst_portsat_arr = InlineArray[Scalar[DType.bfloat16], HIDDEN](uninitialized=True)

    var ep = UnsafePointer(to=expert_arr).bitcast[Scalar[DType.bfloat16]]()
    var dsp = UnsafePointer(to=dst_serial_arr).bitcast[Scalar[DType.bfloat16]]()
    var dpp = UnsafePointer(to=dst_portsat_arr).bitcast[Scalar[DType.bfloat16]]()

    init_inputs(ep, dsp)
    init_inputs(ep, dpp)

    accumulate_serial(ep, dsp)
    accumulate_port_sat(ep, dpp)

    # Mix both outputs into a single keep value so neither gets DCE'd and
    # the checksum is order-sensitive enough to flag silent divergence.
    var acc = Float32(0)
    for i in range(HIDDEN):
        acc += Float32(dsp[i]) - Float32(dpp[i])
    return acc


def main():
    keep(run_case())
