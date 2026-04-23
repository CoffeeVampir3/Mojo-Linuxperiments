from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import act_broadcast_vnni, dot_vnni_broadcasted
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import F32Ptr, BF16Ptr
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs
from minimax.kernels.activations import silu_mul
from minimax.kernels.dispatch_args import FusedW1W3SiluArgs


def fused_w1w3_gemv_row[N: Int, K: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    w1_packed: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    w3_packed: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: Float32,
    w1_sc: UnsafePointer[Float32, MutAnyOrigin],
    w1_cs: UnsafePointer[Float32, MutAnyOrigin],
    w3_sc: UnsafePointer[Float32, MutAnyOrigin],
    w3_cs: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
):
    """Fused dual-matrix GEMV: i8 act × (w1, w3) u8 packed → (gate, up) f32.

    Shares the i8→u8 activation broadcast across both weight matrices and
    both VNNI tiles per K position, saving ~75% of broadcast operations
    versus two sequential gemv_row calls.
    """
    debug_assert(K % VNNI_K_STEP == 0,
        "fused_w1w3_gemv_row: K must be a multiple of VNNI_K_STEP")
    debug_assert(N % VNNI_N_STEP == 0,
        "fused_w1w3_gemv_row: N must be a multiple of VNNI_N_STEP")
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)

        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_off + dc * tile_dc_bytes
                    var t1 = t0 + tile_ks_bytes

                    comptime for p in range(passes):
                        var off = t0 + p * bytes_per_pass
                        w1_acc[p] = dot_vnni_broadcasted[width](
                            w1_acc[p], act_bytes, w1_packed + off)
                        w3_acc[p] = dot_vnni_broadcasted[width](
                            w3_acc[p], act_bytes, w3_packed + off)

                    comptime for p in range(passes):
                        var off = t1 + p * bytes_per_pass
                        w1_acc[passes + p] = dot_vnni_broadcasted[width](
                            w1_acc[passes + p], act_bytes, w1_packed + off)
                        w3_acc[passes + p] = dot_vnni_broadcasted[width](
                            w3_acc[passes + p], act_bytes, w3_packed + off)

                packed_off += 2 * tile_ks_bytes

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = w1_acc[a].cast[DType.float32]() - Float32(128) * (w1_cs + n_base).load[width=width]()
                (gate + n_base).store(w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = w3_acc[a].cast[DType.float32]() - Float32(128) * (w3_cs + n_base).load[width=width]()
                (up + n_base).store(w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


def fused_w1_w3_silu_worker[intermediate: Int, K: Int, fwht_blk: Int](
    args: FusedW1W3SiluArgs,
):
    """N-tiled gate(w1) + up(w3) GEMV -> SiLU -> FWHT -> per-block i8.

    Processes row_count activation rows, each over N-range
    [n_start, n_start + n_count). Tiles in fwht_blk-sized chunks:
      fused w1+w3 GEMV[fwht_blk, K] -> silu(gate)*up
      -> FWHT -> absmax quantize -> i8.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "fused_w1_w3_silu: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "fused_w1_w3_silu: K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_row = intermediate // fwht_blk

    for m in range(args.row_count):
        var act_i8 = args.act_i8 + m * K
        var dequant = args.act_scale[m] / 127.0
        var qi_row = args.qi_out + m * intermediate
        var blk_row = args.blk_scale + m * num_blk_per_row

        var local_n = 0
        while local_n < args.n_count:
            var n_off = args.n_start + local_n

            var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
            var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
            var up = UnsafePointer(to=up_buf).bitcast[Float32]()

            fused_w1w3_gemv_row[fwht_blk, K](
                act_i8,
                args.w1_packed + n_off * K,
                args.w3_packed + n_off * K,
                dequant,
                args.w1_scale + n_off,
                args.w1_colsum + n_off,
                args.w3_scale + n_off,
                args.w3_colsum + n_off,
                gate, up)

            var k = 0
            while k + width <= fwht_blk:
                (gate + k).store(silu_mul(
                    (gate + k).load[width=width](),
                    (up + k).load[width=width]()))
                k += width

            fwht_block[fwht_blk](gate)
            blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
                gate, qi_row + local_n)

            local_n += fwht_blk


# ============================================================================
# f32 GEMV — bf16 activation × f32 weight → f32 output (router projection)
# ============================================================================


@always_inline
def f32_gemv_row[K: Int](
    act: BF16Ptr, weight_row: F32Ptr,
) -> Float32:
    """Dot product: bf16[K] · f32[K] → f32 scalar."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, K]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        var a = (act + off).load[width=width]().cast[DType.float32]()
        var w = (weight_row + off).load[width=width]()
        accs[i] = a * w
    var k = step
    while k + step <= K:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var a = (act + off).load[width=width]().cast[DType.float32]()
            var w = (weight_row + off).load[width=width]()
            accs[i] = a.fma(w, accs[i])
        k += step
    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width]()
        accs[0] = a.fma(w, accs[0])
        k += width
    return tree_reduce_accs(accs)


