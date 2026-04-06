from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of, CompilationTarget
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block


# ============================================================================
# VNNI dot product primitives
# ============================================================================


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """AVX-512 VNNI: u8 x i8 -> i32 dot product accumulate.
    Per dword lane i: acc[i] += sum_{j=0..3} u8(a[i].byte[j]) * i8(b[i].byte[j])
    """
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """VNNI dot: width channels x 4 K values via vpdpbusd."""
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width = width * 4]()
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[DType.uint8, 4](0x80)
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    var b32 = b16.join(b16)
    var b64 = b32.join(b32)
    return vpdpbusd[width](acc, b64.slice[width * 4](), w)


@always_inline
def dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """Fallback dot: width channels x 4 K values via widen-to-i32 multiply."""
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[width=width]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128) * ((wdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128) * ((wdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128) * ((wdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128) * (wdw >> 24)
    return result


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


# ============================================================================
# GEMV row — one activation row x VNNI-packed weight -> N output elements
# ============================================================================


def gemv_row[N: Int, K: Int, OutDType: DType](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
):
    """One activation row x VNNI-packed weight -> N output elements.

    Epilogue: (raw_i32 - 128*colsum[n]) * act_sc * weight_scale[n] -> OutDType.
    """
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)

        for ns in range(0, nb_size, VNNI_N_STEP):
            var acc_buf = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0)
            )
            var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, width]]()

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[p] = dot[width](
                            acc[p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[passes_per_subtile + p] = dot[width](
                            acc[passes_per_subtile + p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

            for a in range(acc_count):
                var n_base = nb + ns + a * width
                var corrected = acc[a].cast[DType.float32]() - Float32(128) * (wcs + n_base).load[width=width]()
                var result = corrected * act_sc * (wsc + n_base).load[width=width]()
                (dst + n_base).store(result.cast[OutDType]())


# ============================================================================
# Worker config — caller-owned, passed by pointer
# ============================================================================


@fieldwise_init
struct WorkerConfig(Copyable, ImplicitlyCopyable):
    var act_scale_ptr: Int  # pointer to f32[seq_len] per-row activation scales
    var start_row: Int      # global row offset for indexing into act_scale_ptr
    var row_count: Int


# ============================================================================
# Worker kernel (BurstPool ABI: 6 Int args)
# ============================================================================


def int8_gemv_worker[N: Int, K: Int](
    act_ptr: Int, wpacked_ptr: Int, colsum_ptr: Int,
    weight_scale_ptr: Int, dst_ptr: Int, config_ptr: Int,
):
    var cfg = UnsafePointer[WorkerConfig, MutAnyOrigin](unsafe_from_address=config_ptr)
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=wpacked_ptr)
    var colsum = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=colsum_ptr)
    var wscale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_scale_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst_ptr)
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg[].act_scale_ptr)
    var start = cfg[].start_row

    for m in range(cfg[].row_count):
        var act_dequant = act_scales[start + m] / Float32(127)
        gemv_row[N, K, DType.bfloat16](
            act + m * K, wpacked, act_dequant, wscale, colsum, dst + m * N,
        )


# ============================================================================
# Dispatch
# ============================================================================


def int8_gemv[N: Int, K: Int, P: BurstThreadPool](
    act_ptr: Int, wpacked_ptr: Int,
    colsum_ptr: Int, weight_scale_ptr: Int, dst_ptr: Int,
    seq_len: Int,
    act_scale_ptr: Int,
    mut configs: InlineArray[WorkerConfig, 128],
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch int8 GEMV: [seq_len, K] x [N, K]^T -> [seq_len, N] bf16.

    act_scale_ptr: f32[seq_len] per-row activation scales (absmax from quantize).
    Dequant per row: (raw - 128*colsum) * (act_scale[m]/127) * weight_scale[n].
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    var cfg_base = Int(UnsafePointer(to=configs).bitcast[WorkerConfig]())
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        configs[i] = WorkerConfig(act_scale_ptr, start, end - start)
        var pack = pool.get_args_base() + i
        pack[].arg0 = act_ptr + start * K
        pack[].arg1 = wpacked_ptr
        pack[].arg2 = colsum_ptr
        pack[].arg3 = weight_scale_ptr
        pack[].arg4 = dst_ptr + start * N * 2
        pack[].arg5 = cfg_base + i * size_of[WorkerConfig]()

    pool.dispatch(int8_gemv_worker[N, K], pool.get_args_base(), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
