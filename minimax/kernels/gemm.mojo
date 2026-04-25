from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import act_broadcast_vnni, dot_vnni_broadcasted
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import I8Ptr, F32Ptr
from experimental3.amx import (
    TILE_M as AMX_TILE_M, TILE_N as AMX_TILE_N, K_STEP as AMX_K_STEP,
    TILE_BYTES as AMX_TILE_BYTES,
    tilezero, tileload, tilestore, tdpbssd,
)
from notstdcollections import AlignedInlineArray
from minimax.kernels.activations import silu_mul
from minimax.kernels.dispatch_args import (
    FusedW1W3SiluArgs, SparseMoePhase1Args, SparseMoePhase2Args,
)


comptime AMX_M_STEP = AMX_TILE_M * 2


@always_inline
def vnni_amx_half_offset(n_off: Int, k_off: Int, K: Int) -> Int:
    """Offset for one 16-column AMX B tile inside the existing VNNI pack."""
    comptime assert VNNI_N_STEP == AMX_TILE_N * 2, "AMX/VNNI N-step mismatch"
    debug_assert(n_off >= 0 and n_off % AMX_TILE_N == 0,
        "vnni_amx_half_offset: n_off must be non-negative and 16-aligned")
    debug_assert(k_off >= 0 and k_off % AMX_K_STEP == 0,
        "vnni_amx_half_offset: k_off must be non-negative and AMX K-step aligned")
    var group_base = (n_off // VNNI_N_STEP) * VNNI_N_STEP * K
    var half = (n_off % VNNI_N_STEP) // AMX_TILE_N
    return group_base + k_off * VNNI_N_STEP + half * AMX_TILE_BYTES


@always_inline
def copy_i8_row[K: Int](
    src: I8Ptr,
    dst: I8Ptr,
):
    comptime width = simd_width_of[DType.int8]()
    var k = 0
    while k + width <= K:
        (dst + k).store((src + k).load[width=width]())
        k += width
    while k < K:
        dst[k] = src[k]
        k += 1


@always_inline
def zero_i8_row[K: Int](dst: I8Ptr):
    comptime width = simd_width_of[DType.int8]()
    var z = SIMD[DType.int8, width](0)
    var k = 0
    while k + width <= K:
        (dst + k).store(z)
        k += width
    while k < K:
        dst[k] = Scalar[DType.int8](0)
        k += 1


@always_inline
def amx_fused_w1w3_tile16[K: Int, out_stride: Int](
    act_tile: I8Ptr,
    w1_packed: I8Ptr,
    w3_packed: I8Ptr,
    act_dequant: F32Ptr,
    w1_scale: F32Ptr,
    w3_scale: F32Ptr,
    n_off: Int,
    gate: F32Ptr,
    up: F32Ptr,
):
    """AMX Mx16 tile for MiniMax MoE phase1.

    Computes signed i8 activation rows by signed i8 W1/W3 rows. This uses the
    same VNNI-packed weight layout as the GEMV path, but because AMX consumes
    signed activations directly there is no u8 activation bias and therefore no
    colsum correction term.
    """
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    comptime width = simd_width_of[DType.float32]()
    comptime c_stride = AMX_TILE_N * 4

    var c_arr = AlignedInlineArray[
        Int32, 4 * AMX_TILE_M * AMX_TILE_N](fill=Int32(0))
    var c = c_arr.unsafe_ptr()
    var w1_c0 = c
    var w1_c1 = c + AMX_TILE_M * AMX_TILE_N
    var w3_c0 = c + 2 * AMX_TILE_M * AMX_TILE_N
    var w3_c1 = c + 3 * AMX_TILE_M * AMX_TILE_N

    tilezero[4]()
    tilezero[5]()
    tilezero[6]()
    tilezero[7]()

    for k_off in range(0, K, AMX_K_STEP):
        tileload[0, DType.int8](act_tile + k_off, K)
        tileload[1, DType.int8](act_tile + AMX_TILE_M * K + k_off, K)
        var b_off = vnni_amx_half_offset(n_off, k_off, K)
        tileload[2, DType.int8](w1_packed + b_off, AMX_K_STEP)
        tileload[3, DType.int8](w3_packed + b_off, AMX_K_STEP)
        tdpbssd[4, 0, 2]()
        tdpbssd[5, 1, 2]()
        tdpbssd[6, 0, 3]()
        tdpbssd[7, 1, 3]()

    tilestore[4, DType.int32](w1_c0, c_stride)
    tilestore[5, DType.int32](w1_c1, c_stride)
    tilestore[6, DType.int32](w3_c0, c_stride)
    tilestore[7, DType.int32](w3_c1, c_stride)

    var n = 0
    while n + width <= AMX_TILE_N:
        var w1s = (w1_scale + n_off + n).load[width=width]()
        var w3s = (w3_scale + n_off + n).load[width=width]()
        for m in range(AMX_M_STEP):
            var row = m
            var w1_src = w1_c0 + row * AMX_TILE_N
            var w3_src = w3_c0 + row * AMX_TILE_N
            if m >= AMX_TILE_M:
                row = m - AMX_TILE_M
                w1_src = w1_c1 + row * AMX_TILE_N
                w3_src = w3_c1 + row * AMX_TILE_N
            var dq = SIMD[DType.float32, width](act_dequant[m])
            (gate + m * out_stride + n).store(
                (w1_src + n).load[width=width]().cast[DType.float32]()
                * dq * w1s)
            (up + m * out_stride + n).store(
                (w3_src + n).load[width=width]().cast[DType.float32]()
                * dq * w3s)
        n += width


@always_inline
def amx_down_tile32[K: Int, fwht_blk: Int](
    act_tile: I8Ptr,
    block_scales: F32Ptr,
    down_packed: I8Ptr,
    down_scale: F32Ptr,
    n_off: Int,
    out_tile: F32Ptr,
):
    """AMX Mx32 blocked down projection tile for sparse prefill MoE."""
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    comptime assert K % fwht_blk == 0, "K must be divisible by fwht block"
    comptime assert fwht_blk % AMX_K_STEP == 0,
        "fwht block must be AMX K-step aligned"
    debug_assert(n_off >= 0 and n_off % VNNI_N_STEP == 0,
        "amx_down_tile32: n_off must be non-negative and 32-aligned")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = K // fwht_blk
    comptime k_steps_per_block = fwht_blk // AMX_K_STEP
    comptime c_stride = AMX_TILE_N * 4

    var c_arr = AlignedInlineArray[
        Int32, 4 * AMX_TILE_M * AMX_TILE_N](fill=Int32(0))
    var c = c_arr.unsafe_ptr()
    var c0_lo = c
    var c1_lo = c + AMX_TILE_M * AMX_TILE_N
    var c0_hi = c + 2 * AMX_TILE_M * AMX_TILE_N
    var c1_hi = c + 3 * AMX_TILE_M * AMX_TILE_N

    for i in range(AMX_M_STEP * VNNI_N_STEP):
        out_tile[i] = Float32(0)

    for blk in range(num_blocks):
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        var k_base = blk * fwht_blk
        for ks in range(k_steps_per_block):
            var k_off = k_base + ks * AMX_K_STEP
            tileload[0, DType.int8](act_tile + k_off, K)
            tileload[1, DType.int8](
                act_tile + AMX_TILE_M * K + k_off, K)
            var b_lo = vnni_amx_half_offset(n_off, k_off, K)
            var b_hi = vnni_amx_half_offset(
                n_off + AMX_TILE_N, k_off, K)
            tileload[2, DType.int8](down_packed + b_lo, AMX_K_STEP)
            tileload[3, DType.int8](down_packed + b_hi, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 1, 2]()
            tdpbssd[6, 0, 3]()
            tdpbssd[7, 1, 3]()

        tilestore[4, DType.int32](c0_lo, c_stride)
        tilestore[5, DType.int32](c1_lo, c_stride)
        tilestore[6, DType.int32](c0_hi, c_stride)
        tilestore[7, DType.int32](c1_hi, c_stride)

        var n = 0
        while n + width <= AMX_TILE_N:
            for m in range(AMX_M_STEP):
                var row = m
                var lo_src = c0_lo + row * AMX_TILE_N
                var hi_src = c0_hi + row * AMX_TILE_N
                if m >= AMX_TILE_M:
                    row = m - AMX_TILE_M
                    lo_src = c1_lo + row * AMX_TILE_N
                    hi_src = c1_hi + row * AMX_TILE_N
                var dq = SIMD[DType.float32, width](
                    block_scales[m * num_blocks + blk] / Float32(127))
                var out_lo = out_tile + m * VNNI_N_STEP + n
                out_lo.store(
                    out_lo.load[width=width]()
                    + (lo_src + n).load[width=width]().cast[DType.float32]()
                    * dq)
                var out_hi = (
                    out_tile + m * VNNI_N_STEP + AMX_TILE_N + n)
                out_hi.store(
                    out_hi.load[width=width]()
                    + (hi_src + n).load[width=width]().cast[DType.float32]()
                    * dq)
            n += width

    var n = 0
    while n + width <= AMX_TILE_N:
        var ws_lo = (down_scale + n_off + n).load[width=width]()
        var ws_hi = (
            down_scale + n_off + AMX_TILE_N + n).load[width=width]()
        for m in range(AMX_M_STEP):
            var out_lo = out_tile + m * VNNI_N_STEP + n
            out_lo.store(out_lo.load[width=width]() * ws_lo)
            var out_hi = out_tile + m * VNNI_N_STEP + AMX_TILE_N + n
            out_hi.store(out_hi.load[width=width]() * ws_hi)
        n += width


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


def sparse_moe_phase1_worker[
    experts_per_rank: Int, intermediate: Int, K: Int, fwht_blk: Int,
](
    args: SparseMoePhase1Args,
):
    """Expert-bucketed sparse phase1 using AMX MxN tiles.

    The worker owns a stride of local experts, gathers each expert bucket into
    AMX_M_STEP activation tiles, computes W1/W3 as true MxK by KxN tiles, then
    applies SiLU, FWHT, and per-block i8 quantization route-major.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "sparse_moe_phase1: intermediate must be a multiple of fwht_blk")
    debug_assert(K % AMX_K_STEP == 0,
        "sparse_moe_phase1: K must be a multiple of AMX_K_STEP")
    debug_assert(fwht_blk % AMX_TILE_N == 0,
        "sparse_moe_phase1: fwht_blk must be a multiple of AMX_TILE_N")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_route = intermediate // fwht_blk
    comptime phase1_n_tiles = fwht_blk // AMX_TILE_N

    var act_arr = AlignedInlineArray[
        Scalar[DType.int8], AMX_M_STEP * K](fill=Scalar[DType.int8](0))
    var act_tile = act_arr.unsafe_ptr()
    var act_scale_arr = AlignedInlineArray[
        Float32, AMX_M_STEP](fill=Float32(0))
    var act_dequant = act_scale_arr.unsafe_ptr()
    var gate_arr = AlignedInlineArray[
        Float32, AMX_M_STEP * fwht_blk](fill=Float32(0))
    var gate = gate_arr.unsafe_ptr()
    var up_arr = AlignedInlineArray[
        Float32, AMX_M_STEP * fwht_blk](fill=Float32(0))
    var up = up_arr.unsafe_ptr()

    var expert = args.worker_id
    while expert < experts_per_rank:
        var route_start = Int(args.offsets[expert])
        var route_end = Int(args.offsets[expert + 1])
        if route_start < route_end:
            var w1p = args.w1_packed + expert * args.expert_stride
            var w1s = args.w1_scale + expert * args.scale_stride
            var w3p = args.w3_packed + expert * args.expert_stride
            var w3s = args.w3_scale + expert * args.scale_stride

            var group_start = route_start
            while group_start < route_end:
                var m_count = min(AMX_M_STEP, route_end - group_start)

                for m in range(m_count):
                    var route = args.routes[group_start + m]
                    var token = Int(route.token)
                    copy_i8_row[K](
                        args.act_i8 + token * K,
                        act_tile + m * K)
                    act_dequant[m] = args.act_scale[token] / Float32(127)
                for m in range(m_count, AMX_M_STEP):
                    zero_i8_row[K](act_tile + m * K)
                    act_dequant[m] = Float32(0)

                var local_n = 0
                while local_n < intermediate:
                    comptime for tile_idx in range(phase1_n_tiles):
                        comptime tile_n = tile_idx * AMX_TILE_N
                        amx_fused_w1w3_tile16[K, fwht_blk](
                            act_tile, w1p, w3p, act_dequant,
                            w1s, w3s,
                            local_n + tile_n,
                            gate + tile_n,
                            up + tile_n)

                    for m in range(m_count):
                        var row_gate = gate + m * fwht_blk
                        var row_up = up + m * fwht_blk
                        var k = 0
                        while k + width <= fwht_blk:
                            (row_gate + k).store(silu_mul(
                                (row_gate + k).load[width=width](),
                                (row_up + k).load[width=width]()))
                            k += width

                        fwht_block[fwht_blk](row_gate)
                        var route_idx = group_start + m
                        var qi_row = args.expert_qi + route_idx * intermediate
                        var blk_row = (
                            args.expert_blk_scale
                            + route_idx * num_blk_per_route
                        )
                        blk_row[local_n // fwht_blk] = (
                            absmax_quantize_i8[fwht_blk](
                                row_gate, qi_row + local_n)
                        )

                    local_n += fwht_blk

                group_start += AMX_M_STEP

        expert += args.num_workers


def sparse_moe_phase2_worker[
    experts_per_rank: Int, hidden: Int, intermediate: Int, fwht_blk: Int,
](
    args: SparseMoePhase2Args,
):
    """Expert-bucketed sparse phase2 with hidden-stripe ownership."""
    debug_assert(intermediate % fwht_blk == 0,
        "sparse_moe_phase2: intermediate must be a multiple of fwht_blk")
    debug_assert(intermediate % AMX_K_STEP == 0,
        "sparse_moe_phase2: intermediate must be AMX K-step aligned")
    debug_assert(args.hidden_count % VNNI_N_STEP == 0,
        "sparse_moe_phase2: hidden_count must be a multiple of VNNI_N_STEP")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_route = intermediate // fwht_blk

    var act_arr = AlignedInlineArray[
        Scalar[DType.int8], AMX_M_STEP * intermediate](
        fill=Scalar[DType.int8](0))
    var act_tile = act_arr.unsafe_ptr()
    var blk_arr = AlignedInlineArray[
        Float32, AMX_M_STEP * num_blk_per_route](fill=Float32(0))
    var blk_tile = blk_arr.unsafe_ptr()
    var out_arr = AlignedInlineArray[
        Float32, AMX_M_STEP * VNNI_N_STEP](fill=Float32(0))
    var out_tile = out_arr.unsafe_ptr()

    var zero = SIMD[DType.float32, width](0)
    for token in range(args.seq_len):
        var h = 0
        while h + width <= args.hidden_count:
            (args.accum + token * hidden + args.hidden_start + h).store(zero)
            h += width
        while h < args.hidden_count:
            args.accum[token * hidden + args.hidden_start + h] = Float32(0)
            h += 1

    for expert in range(experts_per_rank):
        var route_start = Int(args.offsets[expert])
        var route_end = Int(args.offsets[expert + 1])
        if route_start < route_end:
            var w = args.down_packed + expert * args.expert_stride
            var ws = args.down_scale + expert * args.scale_stride

            var group_start = route_start
            while group_start < route_end:
                var m_count = min(AMX_M_STEP, route_end - group_start)

                for m in range(m_count):
                    var route_idx = group_start + m
                    copy_i8_row[intermediate](
                        args.expert_qi + route_idx * intermediate,
                        act_tile + m * intermediate)
                    var src_sc = (
                        args.expert_blk_scale
                        + route_idx * num_blk_per_route)
                    for blk in range(num_blk_per_route):
                        blk_tile[m * num_blk_per_route + blk] = src_sc[blk]
                for m in range(m_count, AMX_M_STEP):
                    zero_i8_row[intermediate](act_tile + m * intermediate)
                    for blk in range(num_blk_per_route):
                        blk_tile[m * num_blk_per_route + blk] = Float32(0)

                var h_off = args.hidden_start
                while h_off < args.hidden_start + args.hidden_count:
                    amx_down_tile32[intermediate, fwht_blk](
                        act_tile, blk_tile, w, ws, h_off, out_tile)

                    var n = 0
                    while n + width <= VNNI_N_STEP:
                        for m in range(m_count):
                            var route = args.routes[group_start + m]
                            var token = Int(route.token)
                            var weight = SIMD[DType.float32, width](
                                route.weight)
                            var dst = args.accum + token * hidden + h_off + n
                            dst.store(
                                dst.load[width=width]()
                                + (out_tile + m * VNNI_N_STEP + n).load[
                                    width=width]() * weight)
                        n += width

                    h_off += VNNI_N_STEP

                group_start += AMX_M_STEP

    for token in range(args.seq_len):
        var dst = args.dst + token * hidden + args.hidden_start
        var h = 0
        while h + width <= args.hidden_count:
            (dst + h).store(
                (args.accum + token * hidden + args.hidden_start + h).load[
                    width=width]().cast[DType.bfloat16]())
            h += width
        while h < args.hidden_count:
            dst[h] = Scalar[DType.bfloat16](
                args.accum[token * hidden + args.hidden_start + h])
            h += 1
