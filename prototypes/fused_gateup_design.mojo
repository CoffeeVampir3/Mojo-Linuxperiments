from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray
from compile import compile_info

from experimental3.common_math import F32Ptr, BF16Ptr, I8Ptr
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.kernels.gemm import (
    fused_gu_gelu_tanh_worker, fused_gu_gelu_tanh_worker_wa,
)
from experimental3.kernels.dispatch_args import FusedGuGeluTanhArgs
from kernels.vnni import VNNI_N_STEP
from minimax.kernels.activations import silu_f32
from minimax.kernels.gemm import fused_w1w3_gemv_row, fused_w1_w3_silu_worker
from minimax.kernels.dispatch_args import FusedW1W3SiluArgs


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


def unified_silu_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_dequant = args.act_scale[m] / Float32(127)
        fused_gateup_quant_row[fwht_blk, K, True, silu_f32](
            args.act_i8 + m * K,
            args.w1_packed, args.w1_scale, args.w1_colsum,
            args.w3_packed, args.w3_scale, args.w3_colsum,
            act_dequant,
            args.qi_out + m * intermediate,
            args.blk_scale + m * num_blk_per_row,
            args.n_start, args.n_count)


def unified_gelu_tanh_worker[intermediate: Int, K: Int, fwht_blk: Int,
    fwht: Bool = True](
    args: FusedGuGeluTanhArgs,
):
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_dequant = args.act_scale[m] / Float32(127)
        fused_gateup_quant_row[fwht_blk, K, fwht, gelu_tanh_f32](
            args.act_i8 + m * K,
            args.wpacked, args.wscale, args.wcolsum,
            args.wpacked + intermediate * K,
            args.wscale + intermediate, args.wcolsum + intermediate,
            act_dequant,
            args.qi_out + m * intermediate,
            args.blk_scale + m * num_blk_per_row,
            args.n_start, args.n_count)


def unified_gelu_tanh_wa_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    comptime num_blk_per_row = intermediate // fwht_blk
    for m in range(args.row_count):
        var act_dequant = args.act_scale[m] / Float32(127)
        fused_gateup_quant_row[fwht_blk, K, True, gelu_tanh_f32](
            args.act_i8 + m * K,
            args.wpacked, args.wscale, args.wcolsum,
            args.wpacked + intermediate * K,
            args.wscale + intermediate, args.wcolsum + intermediate,
            act_dequant,
            args.qi_out + m * intermediate,
            args.blk_scale + m * num_blk_per_row,
            args.n_start, args.n_count)


def shim_existing_silu[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    fused_w1_w3_silu_worker[intermediate, K, fwht_blk](args)


def shim_existing_gelu_tanh[intermediate: Int, K: Int, fwht_blk: Int,
    fwht: Bool = True](
    args: FusedGuGeluTanhArgs,
):
    fused_gu_gelu_tanh_worker[intermediate, K, fwht_blk, fwht](args)


def shim_existing_gelu_tanh_wa[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedGuGeluTanhArgs,
):
    fused_gu_gelu_tanh_worker_wa[intermediate, K, fwht_blk](args)


@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable):
    var fmadd: Int
    var mulps: Int
    var bf16ld: Int
    var f32ldst: Int
    var stores: Int
    var calls: Int
    var bytes: Int


def stats_of(asm: String) -> Stats:
    return Stats(
        fmadd=asm.count("vfmadd"),
        mulps=asm.count("vmulps"),
        bf16ld=asm.count("vpmovzxwd"),
        f32ldst=asm.count("vmovups") + asm.count("vmovaps"),
        stores=asm.count("vmovdqu") + asm.count("vmovdqa"),
        calls=asm.count("call"),
        bytes=asm.byte_length(),
    )


def row(label: String, e: Stats, u: Stats):
    print(label)
    print("  existing: fmadd=", e.fmadd, " mul=", e.mulps,
          " bf16ld=", e.bf16ld, " f32ld/st=", e.f32ldst,
          " store=", e.stores, " call=", e.calls, " bytes=", e.bytes)
    print("  unified:  fmadd=", u.fmadd, " mul=", u.mulps,
          " bf16ld=", u.bf16ld, " f32ld/st=", u.f32ldst,
          " store=", u.stores, " call=", u.calls, " bytes=", u.bytes)
    print("  Δ bytes=", u.bytes - e.bytes,
          " fmadd=", u.fmadd - e.fmadd,
          " mul=", u.mulps - e.mulps)
    print()


def main():
    comptime intermediate = 1536
    comptime K = 3072
    comptime fwht_blk = 128

    var ex_silu = stats_of(String(compile_info[
        shim_existing_silu[intermediate, K, fwht_blk],
        emission_kind="asm"]().asm))
    var un_silu = stats_of(String(compile_info[
        unified_silu_worker[intermediate, K, fwht_blk],
        emission_kind="asm"]().asm))

    var ex_gelu = stats_of(String(compile_info[
        shim_existing_gelu_tanh[intermediate, K, fwht_blk],
        emission_kind="asm"]().asm))
    var un_gelu = stats_of(String(compile_info[
        unified_gelu_tanh_worker[intermediate, K, fwht_blk],
        emission_kind="asm"]().asm))

    comptime small_fwht_blk = 16
    var ex_gelu_wa = stats_of(String(compile_info[
        shim_existing_gelu_tanh_wa[intermediate, K, small_fwht_blk],
        emission_kind="asm"]().asm))
    var un_gelu_wa = stats_of(String(compile_info[
        unified_gelu_tanh_wa_worker[intermediate, K, small_fwht_blk],
        emission_kind="asm"]().asm))

    print("Activation-parameterized fused gate+up worker prototype")
    print("=" * 72)
    print()
    row("SiLU (minimax shape: separate w1/w3 weights)", ex_silu, un_silu)
    row("GELU-tanh (Gemma4 shape: concatenated [gate;up] weight)",
        ex_gelu, un_gelu)
    row("GELU-tanh wa (fwht_blk=16 < VNNI_N_STEP=32)",
        ex_gelu_wa, un_gelu_wa)
