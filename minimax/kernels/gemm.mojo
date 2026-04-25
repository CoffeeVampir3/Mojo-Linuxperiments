from std.sys.info import simd_width_of
from std.collections import InlineArray

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.dot_prod import act_broadcast_vnni, dot_vnni_broadcasted
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.common_math import I8Ptr, F32Ptr, BF16Ptr
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs
from minimax.kernels.activations import silu_mul
from minimax.kernels.dispatch_args import (
    FusedW1W3SiluArgs, SparseMoePhase1Args, SparseMoePhase2Args,
)


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
    """Expert-bucketed sparse phase1.

    One persistent pool worker strides through local experts. For every route
    in an expert bucket, it computes gate/up -> SiLU -> FWHT -> per-block i8
    into route-major buffers indexed by the compact route id.

    This keeps the sparse schedule contract independent of the inner math
    body. The current body reuses the proven VNNI GEMV row kernel; the AMX
    prefill body can replace the route loop with gathered M_STEP tiles.
    """
    debug_assert(intermediate % fwht_blk == 0,
        "sparse_moe_phase1: intermediate must be a multiple of fwht_blk")
    debug_assert(K % 64 == 0,
        "sparse_moe_phase1: K must be a multiple of 64")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_route = intermediate // fwht_blk

    var expert = args.worker_id
    while expert < experts_per_rank:
        var route_start = Int(args.offsets[expert])
        var route_end = Int(args.offsets[expert + 1])
        if route_start < route_end:
            var w1p = args.w1_packed + expert * args.expert_stride
            var w1s = args.w1_scale + expert * args.scale_stride
            var w1c = args.w1_colsum + expert * args.scale_stride
            var w3p = args.w3_packed + expert * args.expert_stride
            var w3s = args.w3_scale + expert * args.scale_stride
            var w3c = args.w3_colsum + expert * args.scale_stride

            for route_idx in range(route_start, route_end):
                var route = args.routes[route_idx]
                var token = Int(route.token)
                var act_i8 = args.act_i8 + token * K
                var dequant = args.act_scale[token] / 127.0
                var qi_row = args.expert_qi + route_idx * intermediate
                var blk_row = (
                    args.expert_blk_scale
                    + route_idx * num_blk_per_route
                )

                var local_n = 0
                while local_n < intermediate:
                    var n_off = local_n

                    var gate_buf = InlineArray[Float32, fwht_blk](
                        fill=Float32(0))
                    var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
                    var up_buf = InlineArray[Float32, fwht_blk](
                        fill=Float32(0))
                    var up = UnsafePointer(to=up_buf).bitcast[Float32]()

                    fused_w1w3_gemv_row[fwht_blk, K](
                        act_i8,
                        w1p + n_off * K,
                        w3p + n_off * K,
                        dequant,
                        w1s + n_off,
                        w1c + n_off,
                        w3s + n_off,
                        w3c + n_off,
                        gate, up)

                    var k = 0
                    while k + width <= fwht_blk:
                        (gate + k).store(silu_mul(
                            (gate + k).load[width=width](),
                            (up + k).load[width=width]()))
                        k += width

                    fwht_block[fwht_blk](gate)
                    blk_row[local_n // fwht_blk] = (
                        absmax_quantize_i8[fwht_blk](
                            gate, qi_row + local_n)
                    )

                    local_n += fwht_blk

        expert += args.num_workers


@always_inline
def gemv_row_blocked_accumulate_f32[N: Int, K: Int, fwht_block_size: Int](
    act_row: I8Ptr,
    wpacked: I8Ptr,
    block_scales: F32Ptr,
    wsc: F32Ptr,
    block_colsums: F32Ptr,
    acc: F32Ptr,
    output_scale: Float32,
    subrange: Int = N,
    colsum_stride: Int = N,
):
    """Blocked int8 GEMV that accumulates weighted f32 into an existing slice."""
    debug_assert(K % fwht_block_size == 0,
        "gemv_row_blocked_accumulate_f32: K must be a multiple of fwht_block_size")
    debug_assert(fwht_block_size >= VNNI_K_STEP,
        "gemv_row_blocked_accumulate_f32: fwht_block_size must be >= VNNI_K_STEP")
    debug_assert(subrange % VNNI_N_STEP == 0,
        "gemv_row_blocked_accumulate_f32: subrange must be a multiple of VNNI_N_STEP")
    comptime num_blocks = K // fwht_block_size
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = compute_n_block(subrange, K)
    var packed_off = 0
    var route_scale = SIMD[DType.float32, width](output_scale)

    for nb in range(0, subrange, n_block):
        var nb_size = min(n_block, subrange - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_acc = InlineArray[SIMD[DType.float32, width], acc_count](
                fill=SIMD[DType.float32, width](0))
            for blk in range(num_blocks):
                var i32_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                    fill=SIMD[DType.int32, width](0))
                for ks in range(0, fwht_block_size, VNNI_K_STEP):
                    for dc in range(dc_count):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                        var t0 = packed_off + dc * tile_dc_bytes
                        var t1 = t0 + tile_ks_bytes
                        comptime for p in range(passes_per_subtile):
                            var off = t0 + p * bytes_per_pass
                            i32_acc[p] = dot_vnni_broadcasted[width](
                                i32_acc[p], act_bytes, wpacked + off)
                        comptime for p in range(passes_per_subtile):
                            var off = t1 + p * bytes_per_pass
                            i32_acc[passes_per_subtile + p] = dot_vnni_broadcasted[width](
                                i32_acc[passes_per_subtile + p],
                                act_bytes, wpacked + off)
                    packed_off += 2 * tile_ks_bytes
                var blk_dequant = block_scales[blk] / 127.0
                for a in range(acc_count):
                    var n_base = nb + ns + a * width
                    var corrected = (
                        i32_acc[a].cast[DType.float32]()
                        - 128.0 * (
                            block_colsums + blk * colsum_stride + n_base
                        ).load[width=width]()
                    )
                    f32_acc[a] += corrected * blk_dequant
            for a in range(acc_count):
                var n_base = nb + ns + a * width
                var weighted = (
                    f32_acc[a]
                    * (wsc + n_base).load[width=width]()
                    * route_scale
                )
                var old = (acc + n_base).load[width=width]()
                (acc + n_base).store(old + weighted)


def sparse_moe_phase2_worker[
    top_k: Int, experts_per_rank: Int,
    hidden: Int, intermediate: Int, fwht_blk: Int,
](
    args: SparseMoePhase2Args,
):
    """Token/slot sparse phase2 with hidden-stripe ownership."""
    debug_assert(intermediate % fwht_blk == 0,
        "sparse_moe_phase2: intermediate must be a multiple of fwht_blk")
    debug_assert(args.hidden_count % VNNI_N_STEP == 0,
        "sparse_moe_phase2: hidden_count must be a multiple of VNNI_N_STEP")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blk_per_route = intermediate // fwht_blk

    var acc_buf = InlineArray[Float32, hidden](fill=Float32(0))
    var acc = UnsafePointer(to=acc_buf).bitcast[Float32]()
    var zero = SIMD[DType.float32, width](0)

    for token in range(args.seq_len):
        var h = 0
        while h + width <= args.hidden_count:
            (acc + h).store(zero)
            h += width
        while h < args.hidden_count:
            acc[h] = Float32(0)
            h += 1

        for slot in range(top_k):
            var route_idx = Int(args.route_indices[token * top_k + slot])
            if route_idx >= 0:
                var route = args.routes[route_idx]
                var local_expert = Int(route.local_expert)
                debug_assert(
                    local_expert >= 0 and local_expert < experts_per_rank,
                    "sparse_moe_phase2: local expert out of range")

                var act = args.expert_qi + route_idx * intermediate
                var blk_sc = args.expert_blk_scale + (
                    route_idx * num_blk_per_route)
                var w = (
                    args.down_packed
                    + local_expert * args.expert_stride
                    + args.hidden_start * intermediate
                )
                var ws = (
                    args.down_scale
                    + local_expert * args.scale_stride
                    + args.hidden_start
                )
                var bcs = (
                    args.down_colsum
                    + local_expert * args.colsum_stride
                    + args.hidden_start
                )
                gemv_row_blocked_accumulate_f32[
                    hidden, intermediate, fwht_blk](
                    act, w, blk_sc, ws, bcs, acc, route.weight,
                    args.hidden_count, hidden)

        var dst = args.dst + token * hidden + args.hidden_start
        h = 0
        while h + width <= args.hidden_count:
            (dst + h).store((acc + h).load[width=width]().cast[DType.bfloat16]())
            h += width
        while h < args.hidden_count:
            dst[h] = Scalar[DType.bfloat16](acc[h])
            h += 1


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
        var w = (weight_row + off).load[width=width, non_temporal=True]()
        accs[i] = a * w
    var k = step
    while k + step <= K:
        comptime for i in range(port_unroll):
            var off = k + i * width
            var a = (act + off).load[width=width]().cast[DType.float32]()
            var w = (weight_row + off).load[width=width, non_temporal=True]()
            accs[i] = a.fma(w, accs[i])
        k += step
    while k + width <= K:
        var a = (act + k).load[width=width]().cast[DType.float32]()
        var w = (weight_row + k).load[width=width, non_temporal=True]()
        accs[0] = a.fma(w, accs[0])
        k += width
    return tree_reduce_accs(accs)
