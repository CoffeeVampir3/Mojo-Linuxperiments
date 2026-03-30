"""Self-contained int8_gemm inner loop for assembly inspection.

Parameterized on hardware SIMD width. Comptime-dispatches VNNI vs fallback.
"""

from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of, CompilationTarget
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from std.benchmark import keep

comptime W = simd_width_of[DType.int32]()


# --- vpdpbusd intrinsic ---

@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


# --- SIMD fallback ---

@always_inline
def dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[width=width]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128) * ((wdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128) * ((wdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128) * ((wdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128) * (wdw >> 24)
    return result


# --- VNNI path ---

@always_inline
def dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width = width * 4]()
    var dword = (act_row + k_pos).bitcast[Scalar[DType.uint32]]()[0] ^ UInt32(0x80808080)
    var dwords = SIMD[DType.uint32, width](dword)
    var tmp = InlineArray[SIMD[DType.uint32, width], 1](fill=dwords)
    var a = UnsafePointer(to=tmp).bitcast[Scalar[DType.int8]]().load[width = width * 4]()
    return vpdpbusd[width](acc, a, w)


# --- Dispatched ---

@always_inline
def dot[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return dot_vnni[width](acc, act_row, wpacked, k_pos)
    else:
        return dot_simd[width](acc, act_row, wpacked, k_pos)


# --- Epilogue ---

@always_inline
def epilogue[width: Int](
    acc: SIMD[DType.int32, width],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_base: Int,
):
    var corrected = acc.cast[DType.float32]() - Float32(128.0) * (wcs + n_base).load[width=width]()
    var result = corrected * act_sc * (wsc + n_base).load[width=width]()
    (dst + n_base).store(result.cast[DType.bfloat16]())


# --- Kernel tile ---

@no_inline
def int8_gemm_tile[N: Int, K: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
):
    comptime tile_n = 16
    comptime blk = 4
    comptime n_step = 32
    comptime k_step = 64
    comptime passes = tile_n // W
    comptime bytes_per_pass = W * blk
    comptime acc_count = n_step // W

    var packed_off = 0
    for ns in range(0, N, n_step):
        var acc_buf = InlineArray[SIMD[DType.int32, W], acc_count](
            fill=SIMD[DType.int32, W](0)
        )
        var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, W]]()

        for ks in range(0, K, k_step):
            for dc in range(k_step // blk):
                var k_pos = ks + dc * blk
                for p in range(passes):
                    acc[p] = dot[W](
                        acc[p], act_row,
                        wpacked + packed_off + p * bytes_per_pass,
                        k_pos,
                    )
                packed_off += tile_n * blk

            for dc in range(k_step // blk):
                var k_pos = ks + dc * blk
                for p in range(passes):
                    acc[passes + p] = dot[W](
                        acc[passes + p], act_row,
                        wpacked + packed_off + p * bytes_per_pass,
                        k_pos,
                    )
                packed_off += tile_n * blk

        for a in range(acc_count):
            epilogue[W](acc[a], act_sc, wsc, wcs, dst, ns + a * W)


def main():
    comptime N = 576
    comptime K = 576

    var act = alloc[Scalar[DType.int8]](K)
    var wpacked = alloc[UInt8](N * K)
    var wsc = alloc[Float32](N)
    var wcs = alloc[Float32](N)
    var dst = alloc[Scalar[DType.bfloat16]](N)

    for k in range(K):
        act[k] = Scalar[DType.int8]((k * 7) % 251 - 125)
    for i in range(N * K):
        wpacked[i] = UInt8(i % 256)
    for n in range(N):
        wsc[n] = Float32(0.01)
        wcs[n] = Float32(n)

    int8_gemm_tile[N, K](act, wpacked, Float32(0.02), wsc, wcs, dst)

    keep(dst[0])

    act.free()
    wpacked.free()
    wsc.free()
    wcs.free()
    dst.free()
