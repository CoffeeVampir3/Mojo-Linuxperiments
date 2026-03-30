"""Hadamard-rotated int8 kernel implementations.

All runtime computational kernels for the quantized Hadamard inference path.
This is the implementation module — the modeling file provides the spec, slot
layout, and forward pass wiring; this file provides the operations.

Contents:
  - FWHT (Fast Walsh-Hadamard Transform): scalar reference + SIMD
  - rms_fwht_quantize: fused RMSNorm + FWHT + int8 quantization
  - silu_fwht_quantize: fused SiLU + FWHT + int8 quantization
  - int8_gemm: VNNI u8/i8 matmul with bias correction
  - int8_gemm_k_to_cache: fused K projection → RoPE → FWHT → u8 cache
  - int8_gemm_v_to_cache: fused V projection → FWHT → u8 cache
  - int8_gemm_gate_up: fused GATE+UP projection (one activation read)
  - int8_gqa_attention: full int8 GQA with fused RoPE, V scale absorption
  - rms_norm_no_gamma: plain RMSNorm for final layer
  - compute_column_sum: VNNI bias correction precomputation

The algebraic invariant:
    output[n] = <Q(H·a), Q(H·W[n])> · scales ≈ <H·a, H·W[n]> = <a, W[n]>

H is block-diagonal Walsh-Hadamard (block = HEAD_DIM = 64). Rotation cancels
via Parseval's theorem: <Hx, Hy> = <x, y> for any orthonormal H.
"""

from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of, CompilationTarget
from std.sys import llvm_intrinsic
from std.collections import InlineArray
from std.utils import IndexList
from threading import BurstPool

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named, BF16, F32, I8,
    Slot, PlacedSlot, Bound, DynView, CacheView, bind, byte_count,
)
from kernels.kernel_ops import PoolFence
from experimental.hadquant_kv_cache import HadQuantKVCache
from kernels.vnni import (
    VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block,
)
from simd_math import sqrt, roundeven, exp_f32, quantize_i8, quantize_i8_scalar
from simd_math.matrixops import log2


# =============================================================================
# FWHT — Fast Walsh-Hadamard Transform (SIMD butterfly network)
#
# Same butterfly network structure as the transpose in matrixops.mojo,
# but add/sub instead of interleave. Fully comptime-unrolled.
#
# Butterfly ordering: stride doubling (1, 2, 4, 8, 16, 32)
# Normalization: 1/sqrt(n) applied after all stages
# Self-inverse: H(H(x)) = x
#
# For block=64 with AVX-512 (16 f32 lanes), 4 ZMM registers hold the block:
#   Stages 0-3 (stride 1,2,4,8): in-register shuffle + FMA
#   Stage 4 (stride 16):         cross-register add/sub (stride = width)
#   Stage 5 (stride 32):         cross-register add/sub (register pairs)
#
# Each in-register stage: shuffle partner via XOR mask, then FMA with
# sign vector (+1 at a-positions, -1 at b-positions):
#   partner[p] = vec[p ^ stride]
#   result[p]  = sign[p] * vec[p] + partner[p]
#     a-pos: +vec + partner = a + b
#     b-pos: -vec + partner = a - b
# =============================================================================


def butterfly_partner[i: Int, stride: Int]() -> Int:
    """Butterfly XOR: element i pairs with element i ^ stride."""
    return i ^ stride


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    """Compile-time shuffle mask: mask[p] = p ^ stride."""
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result


def fwht_width[T: DType, block: Int]() -> Int:
    """SIMD width for FWHT: min(block, hardware width)."""
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


@always_inline
def fwht_block[T: DType, block: Int](
    buf: UnsafePointer[Scalar[T], MutAnyOrigin],
):
    """In-register FWHT on one block of `block` elements.

    Parameterized on DType (f32, f64) and block size (power of 2).
    Fully comptime-unrolled butterfly network with SIMD shuffles for
    in-register stages and plain add/sub for cross-register stages.

    For block=64 on AVX-512 f32: 4 ZMM registers, 6 stages
    (4 shuffle+FMA + 2 cross-register add/sub), final 1/sqrt(64) scale.
    """
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    # Load block into registers
    var r = InlineArray[SIMD[T, width], regs](fill=SIMD[T, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()

    # Butterfly stages
    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            # In-register: shuffle to get butterfly partners, FMA with sign
            comptime mask = butterfly_shuffle[width, stride]()

            # Sign vector: +1 at a-positions (bit `stage` == 0),
            #              -1 at b-positions (bit `stage` == 1)
            var sign_buf = InlineArray[Scalar[T], width](fill=Scalar[T](1.0))
            comptime for k in range(width):
                comptime if (k >> stage) & 1 != 0:
                    sign_buf[k] = Scalar[T](-1.0)
            var sign = UnsafePointer(to=sign_buf).bitcast[Scalar[T]]().load[width=width]()

            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            # Cross-register: pairs at reg_stride apart, plain add/sub
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val

    # Normalize: 1/sqrt(block) for orthonormality
    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc

    # Store
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


def fwht_row[T: DType, block: Int](
    buf: UnsafePointer[Scalar[T], MutAnyOrigin], cols: Int,
):
    """Block-diagonal FWHT on one row of length cols.

    Partitions the row into cols/block independent blocks and applies
    fwht_block to each. block must be a power of 2 and divide cols.
    """
    for b in range(cols // block):
        fwht_block[T, block](buf + b * block)


# =============================================================================
# Kernel: rms_fwht_quantize
#
# Fused RMSNorm + FWHT rotation + int8 quantization.
#
# Gamma is already absorbed into downstream weights. Rotate raw
# (unnormalized) input first, then dual reduction (rms + absmax)
# over rotated values. Parseval guarantees rms(Hx) = rms(x).
# The rms factor is folded into scale_out so the int8_gemm epilogue
# produces W * gamma * (x / rms(x)) = W * RMSNorm(x).
#
# Per row m:
#   x_rot = FWHT(x[m])
#   rms = sqrt(sum(x_rot^2) / K + eps)      = rms(x) by Parseval
#   absmax = max(|x_rot|)
#   scale[m] = absmax / (rms * 127)          rms folded into scale
#   qi[m,k] = round(x_rot[k] * 127 / absmax) standard absmax quantization
# =============================================================================


def rms_fwht_quantize[block: Int,
    InT: Encoding & Shaped, QiT: Encoding & Shaped, ScT: Encoding & Shaped](
    input: DynView[InT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    work: UnsafePointer[Float32, MutAnyOrigin],
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence:
    """Fused RMSNorm + FWHT rotation + int8 quantization.

    Per row: load bf16 → f32, FWHT, dual reduce (rms + absmax),
    quantize to int8. The rms normalization is folded into scale_out
    so the int8_gemm epilogue reconstructs W * RMSNorm(x) via Parseval.

    work: f32 scratch buffer, at least InT.COLS elements. Reused across rows.
    """
    comptime assert InT.DTYPE == DType.bfloat16, "rms_fwht_quantize: input must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "rms_fwht_quantize: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "rms_fwht_quantize: scale output must be f32"

    comptime cols = InT.COLS
    comptime width = simd_width_of[DType.float32]()

    var in_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=input.ptr
    )
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=qi_out.ptr
    )
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scale_out.ptr
    )

    for m in range(input.seq_len):
        var row_in = in_ptr + m * cols
        var row_qi = qi_ptr + m * cols

        # --- Load bf16 → f32 ---
        var k = 0
        while k + width <= cols:
            (work + k).store(
                (row_in + k).load[width=width]().cast[DType.float32]()
            )
            k += width
        while k < cols:
            work[k] = Float32(row_in[k])
            k += 1

        # --- FWHT (in-place on f32 buffer) ---
        fwht_row[DType.float32, block](work, cols)

        # --- Dual reduction: sum-of-squares + absmax ---
        var vsum = SIMD[DType.float32, width](0)
        var vmax = SIMD[DType.float32, width](0)
        k = 0
        while k + width <= cols:
            var v = (work + k).load[width=width]()
            vsum = v.fma(v, vsum)
            vmax = max(vmax, v.__abs__())
            k += width
        var sum_sq = vsum.reduce_add()
        var absmax = vmax.reduce_max()
        while k < cols:
            var v = work[k]
            sum_sq += v * v
            var a = v if v >= 0 else -v
            if a > absmax:
                absmax = a
            k += 1

        # --- Scale: rms folded in ---
        var rms = sqrt[DType.float32, 1](sum_sq / Float32(cols) + eps)
        sc_ptr[m] = absmax / (rms * Float32(127.0))

        # --- Quantize: standard absmax int8 ---
        var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
        var vinv = SIMD[DType.float32, width](inv)
        k = 0
        while k + width <= cols:
            (row_qi + k).store(quantize_i8((work + k).load[width=width](), vinv))
            k += width
        while k < cols:
            row_qi[k] = quantize_i8_scalar(work[k], inv)
            k += 1

    return PoolFence.completed()


# =============================================================================
# Kernel: silu_fwht_quantize
#
# Fused SiLU(gate) * up → FWHT → absmax → int8.
#
# Per row m:
#   silu[k] = gate[k] * sigmoid(gate[k]) * up[k]
#   silu_rot = FWHT(silu)
#   scale[m] = max(|silu_rot|) / 127
#   qi[m,k] = round(silu_rot[k] / scale[m])
# =============================================================================


def silu_fwht_quantize[block: Int,
    GT: Encoding & Shaped, UT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped](
    gate: DynView[GT],
    up: DynView[UT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    work: UnsafePointer[Float32, MutAnyOrigin],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused: SiLU(gate) * up → FWHT → absmax → int8.

    Per row: load gate + up as f32, compute SiLU(gate) * up, FWHT,
    absmax quantize to int8. Two bf16 reads, one int8 write.

    work: f32 scratch buffer, at least GT.COLS elements. Reused across rows.
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
    """
    comptime assert GT.DTYPE == DType.bfloat16, "silu_fwht_quantize: gate must be bf16"
    comptime assert UT.DTYPE == DType.bfloat16, "silu_fwht_quantize: up must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "silu_fwht_quantize: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "silu_fwht_quantize: scale output must be f32"

    comptime cols = GT.COLS
    comptime width = simd_width_of[DType.float32]()

    var gp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=gate.ptr)
    var up_ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=up.ptr)
    var qi_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=qi_out.ptr)
    var sc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_out.ptr)

    for m in range(gate.seq_len):
        var row_g = gp + m * cols
        var row_u = up_ptr + m * cols
        var row_qi = qi_ptr + m * cols

        # --- SiLU(gate) * up → f32 work buffer ---
        var k = 0
        while k + width <= cols:
            var g = (row_g + k).load[width=width]().cast[DType.float32]()
            var u = (row_u + k).load[width=width]().cast[DType.float32]()
            # sigmoid(g) = 1 / (1 + exp(-g))
            var sig = SIMD[DType.float32, width](1.0) / (SIMD[DType.float32, width](1.0) + exp_f32(-g))
            (work + k).store(g * sig * u)
            k += width
        while k < cols:
            var g = Float32(row_g[k])
            var u = Float32(row_u[k])
            var sig = Float32(1.0) / (Float32(1.0) + exp_f32[1](-g))
            work[k] = g * sig * u
            k += 1

        # --- FWHT ---
        fwht_row[DType.float32, block](work, cols)

        # --- Absmax + quantize ---
        var vmax = SIMD[DType.float32, width](0)
        k = 0
        while k + width <= cols:
            vmax = max(vmax, (work + k).load[width=width]().__abs__())
            k += width
        var absmax = vmax.reduce_max()

        sc_ptr[m] = absmax / Float32(127.0)
        var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
        var vinv = SIMD[DType.float32, width](inv)
        k = 0
        while k + width <= cols:
            (row_qi + k).store(quantize_i8((work + k).load[width=width](), vinv))
            k += width
        while k < cols:
            row_qi[k] = quantize_i8_scalar(work[k], inv)
            k += 1

    work.free()
    return PoolFence.completed()


# =============================================================================
# Kernel: int8_gemm
#
# Int8 matmul with VNNI-packed weights, u8/i8 bias correction, scale epilogue.
#
# The weight is stored in 6D VNNI layout (see kernels/vnni.mojo):
#   Innermost [TILE_N=16, VNNI_BLK=4]: 16 output channels × 4 K values = 64 bytes.
#   Each 64-byte group covers VNNI_TILE_N(16) channels.
#
# The kernel processes these 16 packed channels in chunks of `width`
# (= simd_width_of[int32]), so on 256-bit hardware (width=8) each
# 64-byte group is two loads; on 512-bit (width=16) it's one load.
#
# This implementation uses widen-to-i32 multiply (non-VNNI SIMD).
# To swap to VNNI: replace int8_gemm_dot with a vpdpbusd intrinsic.
# =============================================================================


# --- vpdpbusd intrinsic ---
#
# LLVM intrinsic signature: (<N x i32>, <4N x i8>, <4N x i8>) → <N x i32>
# The hardware interprets operand a bytes as u8 and operand b bytes as i8.
# The intrinsic name encodes the bit width: .256 for 8 lanes, .512 for 16.

@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """AVX-VNNI / AVX-512 VNNI: u8 × i8 → i32 dot product accumulate.

    Per dword lane i: acc[i] += Σ_{j=0..3} u8(a[i].byte[j]) * i8(b[i].byte[j])
    """
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


# --- Non-VNNI fallback dot product ---

@always_inline
def int8_gemm_dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """Non-VNNI dot product: `width` channels × 4 K values.

    Loads `width` dwords from packed weight, extracts each byte lane
    (sign-extended i8 → i32), multiplies by broadcast u8 activation
    (i8 + 128), accumulates into `width` i32 lanes.
    """
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[width=width]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128) * ((wdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128) * ((wdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128) * ((wdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128) * (wdw >> 24)
    return result


# --- VNNI dot product ---

@always_inline
def int8_gemm_dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """VNNI dot product: `width` channels × 4 K values in one instruction.

    Loads weight bytes directly (already VNNI-packed). Loads 4 activation
    bytes, XORs 0x80 per byte for u8 conversion, broadcasts the 4-byte
    pattern across all dword lanes. Single vpdpbusd accumulates the result.
    """
    # Weight (signed operand): width*4 packed i8 bytes
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width = width * 4]()

    # Activation (unsigned operand): load 4 i8 bytes as dword, XOR 0x80,
    # broadcast, load as uint8 for vpdpbusd.
    var dword = (act_row + k_pos).bitcast[Scalar[DType.uint32]]()[0] ^ UInt32(0x80808080)
    var dwords = SIMD[DType.uint32, width](dword)
    var tmp = InlineArray[SIMD[DType.uint32, width], 1](fill=dwords)
    var a = UnsafePointer(to=tmp).bitcast[UInt8]().load[width = width * 4]()

    return vpdpbusd[width](acc, a, w)


# --- Dispatched dot product ---

@always_inline
def int8_gemm_dot[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """Dot product with comptime VNNI dispatch.

    Same signature, same result. Only the instruction sequence differs.
    """
    comptime if CompilationTarget.has_vnni():
        return int8_gemm_dot_vnni[width](acc, act_row, wpacked, k_pos)
    else:
        return int8_gemm_dot_simd[width](acc, act_row, wpacked, k_pos)


# --- Shared gemm row: accumulate + bias correct + scale → store as OutDType ---

@always_inline
# =============================================================================
# Row-major i8 dot product (for attention scoring / V aggregation)
#
# Same vpdpbusd instruction as the VNNI-packed gemm, but on contiguous
# row-major data. Both operands are HEAD_DIM contiguous bytes.
# Uses u8×i8 convention: operand a is XORed with 0x80, producing a bias
# of 128 * sum(b) that must be corrected by the caller.
# =============================================================================


@always_inline
def i8_dot_row_major[head_dim: Int](
    a_u8: UnsafePointer[UInt8, MutAnyOrigin],
    b_i8: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> Int32:
    """Row-major dot product: u8 × i8 → i32, length head_dim.

    a_u8 is the unsigned operand (e.g. KV cache entry, stored as u8).
    b_i8 is the signed operand (e.g. quantized Q or attention weight).
    Returns raw u8×i8 accumulation. Caller corrects bias:
        true_dot = raw - 128 * sum(b_i8)
    where sum(b_i8) is computed once per head (constant across entries).

    vpdpbusd when available (4x throughput), widen-to-i32 fallback.
    """
    comptime width = simd_width_of[DType.int32]()
    var acc = SIMD[DType.int32, width](0)

    comptime if CompilationTarget.has_vnni():
        var d = 0
        while d + width * 4 <= head_dim:
            var av = (a_u8 + d).load[width = width * 4]()
            var bv = (b_i8 + d).load[width = width * 4]()
            acc = vpdpbusd[width](acc, av, bv)
            d += width * 4
        while d + width <= head_dim:
            var av = (a_u8 + d).load[width=width]().cast[DType.int32]()
            var bv = (b_i8 + d).load[width=width]().cast[DType.int32]()
            acc += av * bv
            d += width
    else:
        var d = 0
        while d + width <= head_dim:
            # Widen u8 → u32 and i8 → i32
            var av = (a_u8 + d).load[width=width]().cast[DType.int32]()
            var bv = (b_i8 + d).load[width=width]().cast[DType.int32]()
            acc += av * bv
            d += width

    return acc.reduce_add()


@always_inline
def i8_sum[head_dim: Int](
    data: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> Int:
    """Sum all i8 values in a vector of length head_dim.

    Used for vpdpbusd bias correction: true_dot = raw - 128 * sum.
    """
    comptime width = simd_width_of[DType.int32]()
    var acc = SIMD[DType.int32, width](0)
    var d = 0
    while d + width <= head_dim:
        acc += (data + d).load[width=width]().cast[DType.int32]()
        d += width
    return Int(acc.reduce_add())


# =============================================================================
# VNNI-packed GEMM row (for projections with packed weights)
# =============================================================================


def int8_gemm_row[N: Int, K: Int, OutDType: DType](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[OutDType], MutAnyOrigin],
):
    """One activation row × VNNI-packed weight → N output elements.

    Parameterized on OutDType: bfloat16 for terminal gemm, float32 when
    the output feeds further per-head processing (RoPE, FWHT, etc.).
    cast[float32] on f32 is identity; cast[bfloat16] truncates.
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
                        acc[p] = int8_gemm_dot[width](
                            acc[p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[passes_per_subtile + p] = int8_gemm_dot[width](
                            acc[passes_per_subtile + p], act_row,
                            wpacked + packed_off + p * bytes_per_pass,
                            k_pos,
                        )
                    packed_off += VNNI_TILE_N * VNNI_BLK

            for a in range(acc_count):
                var n_base = nb + ns + a * width
                var corrected = acc[a].cast[DType.float32]() - Float32(128.0) * (wcs + n_base).load[width=width]()
                var result = corrected * act_sc * (wsc + n_base).load[width=width]()
                (dst + n_base).store(result.cast[OutDType]())


def int8_gemm[
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    OutT: Encoding & Shaped,
](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    output: DynView[OutT],
    mut pool: BurstPool[],
) -> PoolFence:
    """Int8 GEMM: int8_gemm_row[bfloat16] per activation row."""
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm: activation must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gemm: act scale must be f32"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm: weight must be int8"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemm: weight scale must be f32"
    comptime assert CsT.DTYPE == DType.float32, "int8_gemm: column sum must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "int8_gemm: output must be bf16"

    comptime N = WT.ROWS
    comptime K = WT.COLS

    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_qi.ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=weight.ptr)
    var wsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_scale.ptr)
    var wcs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_colsum.ptr)
    var asc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale.ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=output.ptr)

    for m in range(act_qi.seq_len):
        int8_gemm_row[N, K, DType.bfloat16](
            act + m * K, wpacked, asc[m], wsc, wcs, dst + m * N,
        )

    return PoolFence.completed()



# =============================================================================
# Kernel: int8_gemm_k_to_cache
#
# Fused K projection → RoPE → FWHT → quantize → cache write.
# K never materializes as bf16.
#
# Per row m, per head g:
#   raw = sum_k u8(act) * i8(K_weight[g*HD+d, k])     VNNI
#   k_bf16[d] = (float(raw) - 128*colsum) * act_sc * w_sc
#   RoPE(k_bf16, HEAD_DIM, pos + m)
#   k_rot = FWHT(k_bf16)
#   k_scale = max(|k_rot|) / 127
#   k_qi = round(k_rot / k_scale)
#   write k_qi, k_scale → cache[pos+m, g]
# =============================================================================


def int8_gemm_k_to_cache[block: Int, head_dim: Int, num_kv_heads: Int, max_seq: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    pos: Int,
    f32_buf: UnsafePointer[Float32, MutAnyOrigin],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused K projection → u8 cache: gemm → RoPE → FWHT → quantize → write.

    Two passes per row:
      1. Int8 gemm → f32_buf [N = num_kv_heads * head_dim]
      2. Per head: RoPE(pos+m) → FWHT → quantize i8 → cache.write_head() (XORs to u8)

    f32_buf: scratch buffer, at least N elements. Reused across rows.
    K never materializes as bf16. Cache stores u8 for native vpdpbusd.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_k_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_k_to_cache: weight must be int8"

    comptime N = WT.ROWS
    comptime K = WT.COLS
    comptime half = head_dim // 2
    comptime f32_width = simd_width_of[DType.float32]()

    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_qi.ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=weight.ptr)
    var wsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_scale.ptr)
    var wcs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_colsum.ptr)
    var asc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale.ptr)
    var cp = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=cos_table.ptr)
    var sn = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sin_table.ptr)

    # Stack buffer for per-head quantize output
    var qi_head = InlineArray[Scalar[DType.int8], head_dim](fill=Scalar[DType.int8](0))
    var qi_ptr = UnsafePointer(to=qi_head).bitcast[Scalar[DType.int8]]()

    for m in range(act_qi.seq_len):
        int8_gemm_row[N, K, DType.float32](
            act + m * K, wpacked, asc[m], wsc, wcs, f32_buf,
        )

        var actual_pos = pos + m
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half

        for g in range(num_kv_heads):
            var head = f32_buf + g * head_dim

            # RoPE
            var j = 0
            while j + f32_width <= half:
                var x_lo = (head + j).load[width=f32_width]()
                var x_hi = (head + half + j).load[width=f32_width]()
                var cv = (cos_row + j).load[width=f32_width]()
                var sv = (sin_row + j).load[width=f32_width]()
                (head + j).store(x_lo * cv - x_hi * sv)
                (head + half + j).store(x_hi * cv + x_lo * sv)
                j += f32_width

            # FWHT + quantize
            fwht_block[DType.float32, block](head)

            var vmax = SIMD[DType.float32, f32_width](0)
            j = 0
            while j + f32_width <= head_dim:
                vmax = max(vmax, (head + j).load[width=f32_width]().__abs__())
                j += f32_width
            var absmax = vmax.reduce_max()
            var scale = absmax / Float32(127.0)
            var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
            var vinv = SIMD[DType.float32, f32_width](inv)

            j = 0
            while j + f32_width <= head_dim:
                (qi_ptr + j).store(quantize_i8((head + j).load[width=f32_width](), vinv))
                j += f32_width

            cache.write_head(actual_pos, g, qi_ptr, scale)

    return PoolFence.completed()


# =============================================================================
# Kernel: int8_gemm_v_to_cache
#
# Same as K but without RoPE. V is not position-encoded.
# =============================================================================


def int8_gemm_v_to_cache[block: Int, head_dim: Int, num_kv_heads: Int, max_seq: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    cache: HadQuantKVCache[max_seq, head_dim, num_kv_heads],
    pos: Int,
    f32_buf: UnsafePointer[Float32, MutAnyOrigin],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused V projection → u8 cache: gemm → FWHT → quantize → write.

    Same as K but without RoPE. V never materializes as bf16.
    Cache stores u8 for native vpdpbusd.
    f32_buf: scratch buffer, at least N elements. Reused across rows.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_v_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_v_to_cache: weight must be int8"

    comptime N = WT.ROWS
    comptime K = WT.COLS
    comptime head_dim = cache.head_dim
    comptime f32_width = simd_width_of[DType.float32]()

    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_qi.ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=weight.ptr)
    var wsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_scale.ptr)
    var wcs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_colsum.ptr)
    var asc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale.ptr)

    var qi_head = InlineArray[Scalar[DType.int8], head_dim](fill=Scalar[DType.int8](0))
    var qi_ptr = UnsafePointer(to=qi_head).bitcast[Scalar[DType.int8]]()

    for m in range(act_qi.seq_len):
        int8_gemm_row[N, K, DType.float32](
            act + m * K, wpacked, asc[m], wsc, wcs, f32_buf,
        )

        var actual_pos = pos + m

        for g in range(num_kv_heads):
            var head = f32_buf + g * head_dim

            fwht_block[DType.float32, block](head)

            var vmax = SIMD[DType.float32, f32_width](0)
            var j = 0
            while j + f32_width <= head_dim:
                vmax = max(vmax, (head + j).load[width=f32_width]().__abs__())
                j += f32_width
            var absmax = vmax.reduce_max()
            var scale = absmax / Float32(127.0)
            var inv = Float32(127.0) / absmax if absmax > 0 else Float32(0)
            var vinv = SIMD[DType.float32, f32_width](inv)

            j = 0
            while j + f32_width <= head_dim:
                (qi_ptr + j).store(quantize_i8((head + j).load[width=f32_width](), vinv))
                j += f32_width

            cache.write_head(actual_pos, g, qi_ptr, scale)

    return PoolFence.completed()


# =============================================================================
# Kernel: int8_gemm_gate_up
#
# Fused GATE+UP: reads int8 activation once, two independent dot products
# per output element against same activation row. Halves activation BW.
# =============================================================================


def int8_gemm_gate_up[
    GWT: Encoding & Shaped & Placed, GWsT: Encoding & Shaped & Placed, GCsT: Encoding & Shaped & Placed,
    UWT: Encoding & Shaped & Placed, UWsT: Encoding & Shaped & Placed, UCsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    GOutT: Encoding & Shaped, UOutT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    gate_weight: Bound[GWT], gate_scale: Bound[GWsT], gate_colsum: Bound[GCsT],
    up_weight: Bound[UWT], up_scale: Bound[UWsT], up_colsum: Bound[UCsT],
    gate_out: DynView[GOutT],
    up_out: DynView[UOutT],
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused GATE+UP: two int8_gemm_row calls sharing one activation row.

    Both projections read the same i8 activation and produce bf16 output.
    The activation stays in cache across both calls.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_gate_up: act must be int8"
    comptime assert GOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: gate output must be bf16"
    comptime assert UOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: up output must be bf16"

    comptime G_N = GWT.ROWS
    comptime U_N = UWT.ROWS
    comptime K = GWT.COLS

    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_qi.ptr)
    var asc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale.ptr)
    var g_wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=gate_weight.ptr)
    var g_sc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=gate_scale.ptr)
    var g_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=gate_colsum.ptr)
    var u_wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=up_weight.ptr)
    var u_sc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=up_scale.ptr)
    var u_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=up_colsum.ptr)
    var g_dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=gate_out.ptr)
    var u_dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=up_out.ptr)

    for m in range(act_qi.seq_len):
        var act_row = act + m * K
        var act_sc_m = asc[m]
        int8_gemm_row[G_N, K, DType.bfloat16](act_row, g_wp, act_sc_m, g_sc, g_cs, g_dst + m * G_N)
        int8_gemm_row[U_N, K, DType.bfloat16](act_row, u_wp, act_sc_m, u_sc, u_cs, u_dst + m * U_N)

    return PoolFence.completed()


# int8_gqa_attention is in experimental/hadquant_attn.mojo


# =============================================================================
# Kernel: rms_norm_no_gamma
#
# Plain RMSNorm: output = input / rms(input). No FWHT, no gamma.
# Used for final_norm only (gamma not absorbed — no downstream projection).
# =============================================================================


def rms_norm_no_gamma[InT: Encoding & Shaped, OutT: Encoding & Shaped](
    input: DynView[InT], output: DynView[OutT],
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence:
    """RMSNorm without gamma: output = input / rms(input).

    Used for final_norm where gamma is not absorbed (the final norm
    has no downstream projection to absorb into). In-place safe
    (input and output may alias).
    """
    comptime assert InT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: output must be bf16"

    comptime cols = InT.COLS
    comptime width = simd_width_of[DType.float32]()

    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=input.ptr)
    var out = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=output.ptr)

    for m in range(input.seq_len):
        var row_in = inp + m * cols
        var row_out = out + m * cols

        # Sum of squares
        var vsum = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= cols:
            var v = (row_in + k).load[width=width]().cast[DType.float32]()
            vsum = v.fma(v, vsum)
            k += width
        var sum_sq = vsum.reduce_add()

        # rms = sqrt(sum_sq / cols + eps), inv_rms = 1/rms
        var rms = sqrt[DType.float32, 1](sum_sq / Float32(cols) + eps)
        var inv_rms = Float32(1.0) / rms
        var vinv = SIMD[DType.float32, width](inv_rms)

        # Normalize
        k = 0
        while k + width <= cols:
            var v = (row_in + k).load[width=width]().cast[DType.float32]()
            (row_out + k).store((v * vinv).cast[DType.bfloat16]())
            k += width

    return PoolFence.completed()


# =============================================================================
# Utility: compute_column_sum
#
# VNNI bias correction precomputation. Each i8 weight row is summed to f32.
# colsum[n] = float(sum_k weight[n, k])
#
# Must be computed from row-major weights BEFORE VNNI packing — after
# packing the byte layout is permuted and a naive scan gives wrong results.
# =============================================================================


def compute_column_sum[W: Encoding & Shaped & Placed, Cs: Encoding & Shaped & Placed](
    arena_base: Int, layer_base: Int,
):
    """Sum each row of an i8 weight → f32 column sum buffer.

    colsum[n] = float(sum_k weight[n, k])

    Reads from the weight at W.OFFSET, writes to the column sum buffer
    at Cs.OFFSET. Both relative to layer_base within the arena.
    """
    var wp = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + W.OFFSET,
    )
    var cp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=arena_base + layer_base + Cs.OFFSET,
    )
    for n in range(W.ROWS):
        var acc = Int(0)
        for k in range(W.COLS):
            acc += Int(wp[n * W.COLS + k])
        cp[n] = Float32(acc)
