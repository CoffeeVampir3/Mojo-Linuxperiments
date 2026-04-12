from std.memory import UnsafePointer
from std.sys.info import simd_width_of, CompilationTarget
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from simd_math import exp_f32
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8


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
    var act_ptr: Int
    var wpacked_ptr: Int
    var colsum_ptr: Int
    var weight_scale_ptr: Int
    var dst_ptr: Int
    var act_scale_ptr: Int
    var start_row: Int
    var row_count: Int


# ============================================================================
# Worker kernel
# ============================================================================


def int8_gemv_worker[N: Int, K: Int](cfg: WorkerConfig):
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=cfg.act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cfg.wpacked_ptr)
    var colsum = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.colsum_ptr)
    var wscale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.weight_scale_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=cfg.dst_ptr)
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.act_scale_ptr)
    var start = cfg.start_row

    for m in range(cfg.row_count):
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
    mut pool: P,
) -> PoolFence[P]:
    """Dispatch int8 GEMV: [seq_len, K] x [N, K]^T -> [seq_len, N] bf16.

    act_scale_ptr: f32[seq_len] per-row activation scales (absmax from quantize).
    Dequant per row: (raw - 128*colsum) * (act_scale[m]/127) * weight_scale[n].
    """
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](
        fill=WorkerConfig(0, 0, 0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = WorkerConfig(
            act_ptr + start * K, wpacked_ptr, colsum_ptr,
            weight_scale_ptr, dst_ptr + start * N * 2,
            act_scale_ptr, start, end - start)

    pool.dispatch[WorkerConfig, int8_gemv_worker[N, K]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# Fused gate+up GEMV → SiLU → FWHT → i8 quantize
# ============================================================================


@fieldwise_init
struct FusedGUSiluConfig(Copyable, ImplicitlyCopyable):
    var act_ptr: Int
    var gate_wpacked_ptr: Int
    var up_wpacked_ptr: Int
    var qi_ptr: Int
    var scale_ptr: Int
    var act_scale_ptr: Int
    var start_row: Int
    var row_count: Int
    var gate_colsum_ptr: Int
    var up_colsum_ptr: Int
    var gate_wscale_ptr: Int
    var up_wscale_ptr: Int


def fused_gu_silu_worker[GATE_ROWS: Int, K: Int, FWHT_BLK: Int](cfg: FusedGUSiluConfig):
    """Fused gate+up GEMV -> SiLU(gate)*up -> FWHT -> per-row i8 quantize."""
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=cfg.act_ptr)
    var gate_wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cfg.gate_wpacked_ptr)
    var up_wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=cfg.up_wpacked_ptr)
    var qi_out = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=cfg.qi_ptr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.scale_ptr)
    var act_scales = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.act_scale_ptr)
    var gate_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.gate_colsum_ptr)
    var up_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.up_colsum_ptr)
    var gate_ws = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.gate_wscale_ptr)
    var up_ws = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cfg.up_wscale_ptr)
    var start = cfg.start_row

    # Stack buffers for one row of gate and up (f32)
    var gate_buf = InlineArray[Float32, GATE_ROWS](fill=Float32(0))
    var up_buf = InlineArray[Float32, GATE_ROWS](fill=Float32(0))
    var gate_f32 = UnsafePointer(to=gate_buf).bitcast[Scalar[DType.float32]]()
    var up_f32 = UnsafePointer(to=up_buf).bitcast[Scalar[DType.float32]]()

    # Work buffer for FWHT (reuse gate_buf after silu consumes it)
    var work = UnsafePointer(to=gate_buf).bitcast[Float32]()

    comptime width = simd_width_of[DType.float32]()

    for m in range(cfg.row_count):
        var act_dequant = act_scales[start + m] / Float32(127)

        # Gate GEMV → f32 stack buffer
        gemv_row[GATE_ROWS, K, DType.float32](
            act + m * K, gate_wp, act_dequant, gate_ws, gate_cs, gate_f32)

        # Up GEMV → f32 stack buffer
        gemv_row[GATE_ROWS, K, DType.float32](
            act + m * K, up_wp, act_dequant, up_ws, up_cs, up_f32)

        # SiLU(gate) * up → work buffer (reuses gate_buf)
        var k = 0
        while k + width <= GATE_ROWS:
            var g = (gate_f32 + k).load[width=width]()
            var u = (up_f32 + k).load[width=width]()
            var sig = SIMD[DType.float32, width](1.0) / (SIMD[DType.float32, width](1.0) + exp_f32[width](-g))
            (work + k).store(g * sig * u)
            k += width

        # Block-diagonal FWHT
        for b in range(GATE_ROWS // FWHT_BLK):
            fwht_block[FWHT_BLK](work + b * FWHT_BLK)

        # Dynamic absmax + quantize → i8 output
        scales[start + m] = absmax_quantize_i8[GATE_ROWS](work, qi_out + (start + m) * GATE_ROWS)


def fused_gu_silu[GATE_ROWS: Int, K: Int, FWHT_BLK: Int, P: BurstThreadPool](
    act_ptr: Int,
    gate_wpacked_ptr: Int, gate_colsum_ptr: Int, gate_wscale_ptr: Int,
    up_wpacked_ptr: Int, up_colsum_ptr: Int, up_wscale_ptr: Int,
    qi_ptr: Int, scale_ptr: Int,
    seq_len: Int, act_scale_ptr: Int,
    mut pool: P,
) -> PoolFence[P]:
    """Fused gate+up GEMV -> SiLU(gate)*up -> FWHT -> per-row i8 quantize."""
    if seq_len == 0:
        return PoolFence[P].completed()

    comptime MAX_POOL_CAPACITY = 128
    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[FusedGUSiluConfig, MAX_POOL_CAPACITY](
        fill=FusedGUSiluConfig(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = FusedGUSiluConfig(
            act_ptr + start * K, gate_wpacked_ptr, up_wpacked_ptr,
            qi_ptr, scale_ptr, act_scale_ptr, start, end - start,
            gate_colsum_ptr, up_colsum_ptr,
            gate_wscale_ptr, up_wscale_ptr)

    pool.dispatch[FusedGUSiluConfig, fused_gu_silu_worker[GATE_ROWS, K, FWHT_BLK]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
