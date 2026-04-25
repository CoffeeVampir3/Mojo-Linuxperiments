from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, TILE_BYTES,
    tilezero, tileload, tilestore, tdpbssd,
)
from experimental3.common_math import I8Ptr, F32Ptr
from experimental3.kernels.dispatch_args import WorkerConfig, Int8GemvBlockedArgs
from notstdcollections import AlignedInlineArray


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime SIMD_W = simd_width_of[DType.float32]()


def amx_gemm[N: Int, K: Int, OutDType: DType = DType.bfloat16](
    act: I8Ptr,
    wpacked: I8Ptr,
    act_scale: F32Ptr,
    w_scale: F32Ptr,
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
    M: Int,
):
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            for k in range(0, K, K_STEP):
                tileload[0, DType.int8](act_mb + k, K)
                tileload[1, DType.int8](act_mb + 16 * K + k, K)
                var b_base = wpacked + nb * K + k * N_STEP
                tileload[2, DType.int8](b_base, K_STEP)
                tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            comptime c_stride = N_STEP * 4
            tilestore[4, DType.int32](c_buf, c_stride)
            tilestore[5, DType.int32](c_buf + 16, c_stride)
            tilestore[6, DType.int32](c_buf + 16 * N_STEP, c_stride)
            tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, c_stride)

            for m in range(m_count):
                var dq = act_scale[mb + m] / Float32(127)
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                    var f32_v = i32_v.cast[DType.float32]() * dq
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(
                        (f32_v * ws).cast[OutDType]())

    _ = c_arr


def amx_gemm_blocked[N: Int, K: Int, fwht_blk: Int,
    OutDType: DType = DType.bfloat16](
    act: I8Ptr,
    wpacked: I8Ptr,
    blk_scales: F32Ptr,
    w_scale: F32Ptr,
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
    M: Int,
    output_scale: Float32 = Float32(1.0),
):
    comptime assert N % N_STEP == 0
    comptime assert K % K_STEP == 0
    comptime assert K % fwht_blk == 0
    comptime assert fwht_blk % K_STEP == 0
    comptime num_blocks = K // fwht_blk
    comptime k_steps_per_block = fwht_blk // K_STEP

    var c_arr = AlignedInlineArray[Int32, M_STEP * N_STEP](fill=Int32(0))
    var c_buf = c_arr.unsafe_ptr()
    var f32_arr = AlignedInlineArray[Float32, M_STEP * N_STEP](fill=Float32(0))
    var f32_buf = f32_arr.unsafe_ptr()

    for mb in range(0, M, M_STEP):
        var m_count = min(M_STEP, M - mb)
        var act_mb = act + mb * K

        for nb in range(0, N, N_STEP):
            for i in range(M_STEP * N_STEP):
                f32_buf[i] = Float32(0)

            for blk in range(num_blocks):
                tilezero[4]()
                tilezero[5]()
                tilezero[6]()
                tilezero[7]()

                var k_base = blk * fwht_blk
                for ks in range(k_steps_per_block):
                    var k = k_base + ks * K_STEP
                    tileload[0, DType.int8](act_mb + k, K)
                    tileload[1, DType.int8](act_mb + 16 * K + k, K)
                    var b_base = wpacked + nb * K + k * N_STEP
                    tileload[2, DType.int8](b_base, K_STEP)
                    tileload[3, DType.int8](b_base + TILE_BYTES, K_STEP)
                    tdpbssd[4, 0, 2]()
                    tdpbssd[5, 0, 3]()
                    tdpbssd[6, 1, 2]()
                    tdpbssd[7, 1, 3]()

                comptime c_stride = N_STEP * 4
                tilestore[4, DType.int32](c_buf, c_stride)
                tilestore[5, DType.int32](c_buf + 16, c_stride)
                tilestore[6, DType.int32](c_buf + 16 * N_STEP, c_stride)
                tilestore[7, DType.int32](c_buf + 16 * N_STEP + 16, c_stride)

                for m in range(m_count):
                    var dq = blk_scales[(mb + m) * num_blocks + blk] / Float32(127)
                    var row_off = m * N_STEP
                    for n in range(0, N_STEP, SIMD_W):
                        var i32_v = (c_buf + row_off + n).load[width=SIMD_W]()
                        (f32_buf + row_off + n).store(
                            (f32_buf + row_off + n).load[width=SIMD_W]()
                            + i32_v.cast[DType.float32]() * dq)

            for m in range(m_count):
                var row_off = m * N_STEP
                for n in range(0, N_STEP, SIMD_W):
                    var acc = (f32_buf + row_off + n).load[width=SIMD_W]()
                    var ws = (w_scale + nb + n).load[width=SIMD_W]()
                    (dst + (mb + m) * N + nb + n).store(
                        (acc * ws * output_scale).cast[OutDType]())

    _ = c_arr
    _ = f32_arr


def int8_gemm_amx_worker[N: Int, K: Int](cfg: WorkerConfig):
    debug_assert(cfg.count > 0, "int8_gemm_amx_worker: count must be positive")
    amx_gemm[N, K, DType.bfloat16](
        cfg.act_ptr,
        cfg.wpacked_ptr,
        cfg.act_scale_ptr + cfg.start,
        cfg.weight_scale_ptr,
        cfg.dst_ptr,
        cfg.count)


def int8_gemm_blocked_amx_worker[N: Int, K: Int, fwht_blk: Int](
    args: Int8GemvBlockedArgs,
):
    debug_assert(args.row_count > 0,
        "int8_gemm_blocked_amx_worker: row_count must be positive")
    amx_gemm_blocked[N, K, fwht_blk, DType.bfloat16](
        args.act,
        args.wpacked,
        args.blk_scale,
        args.wscale,
        args.dst,
        args.row_count,
        args.output_scale)
