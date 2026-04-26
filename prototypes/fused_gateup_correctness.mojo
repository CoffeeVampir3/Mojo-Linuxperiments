from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of

from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from experimental3.kernels.gemm import fused_gu_gelu_tanh_worker
from experimental3.kernels.dispatch_args import FusedGuGeluTanhArgs
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from kernels.vnni import VNNI_N_STEP
from minimax.kernels.gemm import (
    fused_w1w3_gemv_row, fused_w1_w3_silu_worker,
)
from minimax.kernels.dispatch_args import FusedW1W3SiluArgs
from minimax.kernels.activations import silu_f32
from experimental3.init_weights import (
    PackColsumTask, make_pack_colsum_task, dispatch_pack_colsum_tasks,
)


@always_inline
def fused_gateup_quant_row[
    fwht_blk: Int, K: Int, fwht: Bool,
    act_fn: def[w: Int](SIMD[DType.float32, w]) thin -> SIMD[DType.float32, w],
](
    act_i8: I8Ptr,
    gate_packed: I8Ptr, gate_scale: F32Ptr, gate_colsum: F32Ptr,
    up_packed: I8Ptr, up_scale: F32Ptr, up_colsum: F32Ptr,
    act_dequant: Float32,
    qi_row: I8Ptr, blk_row: F32Ptr,
    n_off: Int, n_count: Int,
):
    comptime width = simd_width_of[DType.float32]()
    comptime chunk_size = fwht_blk if fwht_blk >= VNNI_N_STEP else VNNI_N_STEP
    comptime sub_blocks = chunk_size // fwht_blk

    var local_n = 0
    while local_n < n_count:
        var n_pos = n_off + local_n

        var gate_buf = InlineArray[Float32, chunk_size](fill=Float32(0))
        var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
        var up_buf = InlineArray[Float32, chunk_size](fill=Float32(0))
        var up = UnsafePointer(to=up_buf).bitcast[Float32]()

        fused_w1w3_gemv_row[chunk_size, K](
            act_i8,
            gate_packed + n_pos * K,
            up_packed + n_pos * K,
            act_dequant,
            gate_scale + n_pos, gate_colsum + n_pos,
            up_scale + n_pos, up_colsum + n_pos,
            gate, up)

        var k = 0
        while k + width <= chunk_size:
            var g = (gate + k).load[width=width]()
            var u = (up + k).load[width=width]()
            (gate + k).store(act_fn[width](g) * u)
            k += width

        for sb in range(sub_blocks):
            var sb_off = sb * fwht_blk
            comptime if fwht:
                fwht_block[fwht_blk](gate + sb_off)
            blk_row[(local_n + sb_off) // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate + sb_off, qi_row + local_n + sb_off)

        local_n += chunk_size


def main():
    comptime intermediate = 64
    comptime K = 64
    comptime fwht_blk = 32

    var act_arr = InlineArray[Scalar[DType.int8], K](fill=Scalar[DType.int8](0))
    var act = UnsafePointer(to=act_arr).bitcast[Scalar[DType.int8]]()
    for i in range(K):
        act[i] = Scalar[DType.int8]((i % 31) - 15)

    var act_scale_arr = InlineArray[Float32, 1](fill=Float32(1.5))
    var act_scale = UnsafePointer(to=act_scale_arr).bitcast[Float32]()

    var w1_arr = InlineArray[Scalar[DType.int8], intermediate * K](
        fill=Scalar[DType.int8](0))
    var w3_arr = InlineArray[Scalar[DType.int8], intermediate * K](
        fill=Scalar[DType.int8](0))
    var w1 = UnsafePointer(to=w1_arr).bitcast[Scalar[DType.int8]]()
    var w3 = UnsafePointer(to=w3_arr).bitcast[Scalar[DType.int8]]()
    for i in range(intermediate * K):
        w1[i] = Scalar[DType.int8]((i * 7 % 251) - 125)
        w3[i] = Scalar[DType.int8]((i * 13 % 251) - 125)

    var w1_sc_arr = InlineArray[Float32, intermediate](fill=Float32(0.01))
    var w3_sc_arr = InlineArray[Float32, intermediate](fill=Float32(0.012))
    var w1_cs_arr = InlineArray[Float32, intermediate](fill=Float32(0.0))
    var w3_cs_arr = InlineArray[Float32, intermediate](fill=Float32(0.0))
    var w1_sc = UnsafePointer(to=w1_sc_arr).bitcast[Float32]()
    var w3_sc = UnsafePointer(to=w3_sc_arr).bitcast[Float32]()
    var w1_cs = UnsafePointer(to=w1_cs_arr).bitcast[Float32]()
    var w3_cs = UnsafePointer(to=w3_cs_arr).bitcast[Float32]()

    var qi_existing_arr = InlineArray[Scalar[DType.int8], intermediate](
        fill=Scalar[DType.int8](0))
    var qi_existing = UnsafePointer(to=qi_existing_arr).bitcast[Scalar[DType.int8]]()
    var blk_existing_arr = InlineArray[Float32, intermediate // fwht_blk](
        fill=Float32(0))
    var blk_existing = UnsafePointer(to=blk_existing_arr).bitcast[Float32]()

    var qi_unified_arr = InlineArray[Scalar[DType.int8], intermediate](
        fill=Scalar[DType.int8](0))
    var qi_unified = UnsafePointer(to=qi_unified_arr).bitcast[Scalar[DType.int8]]()
    var blk_unified_arr = InlineArray[Float32, intermediate // fwht_blk](
        fill=Float32(0))
    var blk_unified = UnsafePointer(to=blk_unified_arr).bitcast[Float32]()

    var ex_args = FusedW1W3SiluArgs(
        act, act_scale,
        w1, w1_sc, w1_cs,
        w3, w3_sc, w3_cs,
        qi_existing, blk_existing,
        0, intermediate, 1)
    fused_w1_w3_silu_worker[intermediate, K, fwht_blk](ex_args)

    var act_dequant = act_scale[0] / Float32(127)
    fused_gateup_quant_row[fwht_blk, K, True, silu_f32](
        act, w1, w1_sc, w1_cs, w3, w3_sc, w3_cs,
        act_dequant, qi_unified, blk_unified, 0, intermediate)

    var max_diff = Int(0)
    var num_diff = 0
    for i in range(intermediate):
        var d = Int(qi_existing[i]) - Int(qi_unified[i])
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
        if d > 0:
            num_diff += 1

    var max_blk_diff = Float32(0)
    for b in range(intermediate // fwht_blk):
        var d = blk_existing[b] - blk_unified[b]
        if d < Float32(0):
            d = -d
        if d > max_blk_diff:
            max_blk_diff = d

    print("SiLU correctness:")
    print("  i8 max diff:", max_diff, " (mismatched lanes:", num_diff, "/", intermediate, ")")
    print("  block scale max diff:", max_blk_diff)
    if max_diff == 0 and max_blk_diff == Float32(0):
        print("  IDENTICAL")
    else:
        print("  DIFFERENT")
