"""Hadamard-rotated int8 kernel implementations.

All runtime computational kernels for the quantized Hadamard inference path.
This is the implementation module — the modeling file provides the spec, slot
layout, and forward pass wiring; this file provides the operations.

Contents:
  - FWHT (Fast Walsh-Hadamard Transform): scalar reference + SIMD
  - rms_fwht_quantize: fused RMSNorm + FWHT + int8 quantization
  - silu_fwht_quantize: fused SiLU + FWHT + int8 quantization
  - int8_gemm: VNNI u8/i8 matmul with bias correction
  - int8_gemm_k_to_cache: fused K projection → RoPE → FWHT → int8 cache
  - int8_gemm_v_to_cache: fused V projection → FWHT → int8 cache
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
from std.memory.unsafe_pointer import alloc
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
from kernels.vnni import (
    VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block,
)
from simd_math import sqrt, roundeven
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
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence:
    """Fused RMSNorm + FWHT rotation + int8 quantization.

    Per row: load bf16 → f32, FWHT, dual reduce (rms + absmax),
    quantize to int8. The rms normalization is folded into scale_out
    so the int8_gemm epilogue reconstructs W * RMSNorm(x) via Parseval.

    qi[k] uses standard absmax quantization of the rotated values.
    scale encodes absmax / (rms * 127) — the 1/rms factor makes the
    gemm output equivalent to matmul against the normalized input.
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

    # Per-row f32 work buffer — fits in L1 (e.g. 576 * 4 = 2.3 KB)
    var work = alloc[Float32](cols)

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
        comptime vlo = SIMD[DType.float32, width](-128.0)
        comptime vhi = SIMD[DType.float32, width](127.0)
        k = 0
        while k + width <= cols:
            var v = (work + k).load[width=width]()
            var q = min(max(roundeven(v * vinv), vlo), vhi)
            (row_qi + k).store(q.cast[DType.int8]())
            k += width
        while k < cols:
            var v = roundeven[DType.float32, 1](work[k] * inv)
            var q = min(max(v, Float32(-128.0)), Float32(127.0))
            row_qi[k] = q.cast[DType.int8]()
            k += 1

    work.free()
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
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused: SiLU(gate) * up → FWHT(blocks) → absmax → int8.

    Two bf16 reads (gate + up), one int8 write. SiLU involves exp() —
    fast f32 approximation available in simd_math/ops.mojo (exp_f32).
    The sigmoid is 1 / (1 + exp(-x)).
    """
    comptime assert GT.DTYPE == DType.bfloat16, "silu_fwht_quantize: gate must be bf16"
    comptime assert UT.DTYPE == DType.bfloat16, "silu_fwht_quantize: up must be bf16"
    comptime assert QiT.DTYPE == DType.int8, "silu_fwht_quantize: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "silu_fwht_quantize: scale output must be f32"
    print("    [stub] silu_fwht_quantize [" + String(gate.seq_len)
          + "x" + String(GT.COLS) + "] block=" + String(block))
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
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """AVX-VNNI / AVX-512 VNNI: u8 × i8 → i32 dot product accumulate.

    Per dword lane i: acc[i] += Σ_{j=0..3} u8(a[i].byte[j]) * i8(b[i].byte[j])
    a bytes are interpreted as unsigned by hardware regardless of Mojo signedness.
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
    # Weight: width*4 packed bytes
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width = width * 4]()

    # Activation: load 4 bytes as one dword, XOR all bytes with 0x80
    # simultaneously, broadcast dword to all lanes, bitcast to int8.
    # No InlineArray indexing — avoids bounds check pollution.
    var dword = (act_row + k_pos).bitcast[Scalar[DType.uint32]]()[0] ^ UInt32(0x80808080)
    var dwords = SIMD[DType.uint32, width](dword)
    var tmp = InlineArray[SIMD[DType.uint32, width], 1](fill=dwords)
    var a = UnsafePointer(to=tmp).bitcast[Scalar[DType.int8]]().load[width = width * 4]()

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


@always_inline
def int8_gemm_epilogue[width: Int](
    acc: SIMD[DType.int32, width],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_base: Int,
):
    """Epilogue for `width` output channels: bias correction + scale + bf16 cast."""
    var corrected = acc.cast[DType.float32]() - Float32(128.0) * (wcs + n_base).load[width=width]()
    var result = corrected * act_sc * (wsc + n_base).load[width=width]()
    (dst + n_base).store(result.cast[DType.bfloat16]())


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
    """Int8 GEMM over VNNI-packed weights with u8/i8 bias correction.

    Iterates the packed weight in memory order (sequential reads).
    The packed sub-tile is VNNI_TILE_N(16) channels wide. The kernel
    processes it in SIMD chunks of `width` = simd_width_of[int32],
    so 2 passes on 256-bit hardware, 1 pass on 512-bit.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm: activation must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gemm: act scale must be f32"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm: weight must be int8"
    comptime assert WsT.DTYPE == DType.float32, "int8_gemm: weight scale must be f32"
    comptime assert CsT.DTYPE == DType.float32, "int8_gemm: column sum must be f32"
    comptime assert OutT.DTYPE == DType.bfloat16, "int8_gemm: output must be bf16"

    comptime N = WT.ROWS
    comptime K = WT.COLS
    comptime width = simd_width_of[DType.int32]()
    # How many SIMD passes to cover one packed sub-tile of VNNI_TILE_N channels
    comptime passes_per_subtile = VNNI_TILE_N // width
    # Bytes per SIMD load: width dwords × 4 bytes
    comptime bytes_per_pass = width * VNNI_BLK

    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=act_qi.ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=weight.ptr)
    var wsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_scale.ptr)
    var wcs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=weight_colsum.ptr)
    var asc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=act_scale.ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=output.ptr)

    var n_block = compute_n_block(N, K)

    for m in range(act_qi.seq_len):
        var act_row = act + m * K
        var out_row = dst + m * N
        var act_sc_m = asc[m]
        var packed_off = 0

        for nb in range(0, N, n_block):
            var nb_size = min(n_block, N - nb)

            for ns in range(0, nb_size, VNNI_N_STEP):
                # Accumulators accessed via raw pointer to avoid bounds check pollution.
                comptime acc_count = VNNI_N_STEP // width
                var acc_buf = InlineArray[SIMD[DType.int32, width], acc_count](
                    fill=SIMD[DType.int32, width](0)
                )
                var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, width]]()

                for ks in range(0, K, VNNI_K_STEP):
                    # Sub-tile 0: first VNNI_TILE_N channels of this N_STEP
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            acc[p] = int8_gemm_dot[width](
                                acc[p], act_row,
                                wpacked + packed_off + p * bytes_per_pass,
                                k_pos,
                            )
                        packed_off += VNNI_TILE_N * VNNI_BLK

                    # Sub-tile 1: next VNNI_TILE_N channels
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            acc[passes_per_subtile + p] = int8_gemm_dot[width](
                                acc[passes_per_subtile + p], act_row,
                                wpacked + packed_off + p * bytes_per_pass,
                                k_pos,
                            )
                        packed_off += VNNI_TILE_N * VNNI_BLK

                # Epilogue for all accumulators
                for a in range(acc_count):
                    int8_gemm_epilogue[width](
                        acc[a], act_sc_m, wsc, wcs, out_row, nb + ns + a * width,
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


def int8_gemm_k_to_cache[block: Int, head_dim: Int, num_kv_heads: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    KcT: Encoding & Shaped, KsT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    k_cache: CacheView[KcT],
    k_cache_scale: CacheView[KsT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused K projection → cache: int8_gemm epilogue → RoPE → FWHT → quantize → write.

    Computes the K projection one head at a time (HEAD_DIM output elements).
    The tile epilogue applies RoPE + FWHT per head and writes int8 directly
    to the KV cache. K never materializes as bf16.

    Writes seq_len rows to cache positions pos..pos+seq_len-1.
    Row m gets RoPE at position pos+m.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_k_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_k_to_cache: weight must be int8"
    comptime assert KcT.DTYPE == DType.int8, "int8_gemm_k_to_cache: K cache must be int8"
    print("    [stub] int8_gemm_k_to_cache pos=" + String(pos)
          + " seq_len=" + String(act_qi.seq_len)
          + " [" + String(act_qi.seq_len) + "x" + String(WT.COLS)
          + "] -> K cache[" + String(pos) + ".." + String(pos + act_qi.seq_len - 1) + "]")
    return PoolFence.completed()


# =============================================================================
# Kernel: int8_gemm_v_to_cache
#
# Same as K but without RoPE. V is not position-encoded.
# =============================================================================


def int8_gemm_v_to_cache[block: Int, num_kv_heads: Int,
    WT: Encoding & Shaped & Placed, WsT: Encoding & Shaped & Placed,
    CsT: Encoding & Shaped & Placed,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    VcT: Encoding & Shaped, VsT: Encoding & Shaped](
    act_qi: DynView[QiT],
    act_scale: DynView[ScT],
    weight: Bound[WT],
    weight_scale: Bound[WsT],
    weight_colsum: Bound[CsT],
    v_cache: CacheView[VcT],
    v_cache_scale: CacheView[VsT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Fused V projection → cache: int8_gemm epilogue → FWHT → quantize → write.

    Same as K but without RoPE. V never materializes as bf16.
    Writes seq_len rows to cache positions pos..pos+seq_len-1.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_v_to_cache: act must be int8"
    comptime assert WT.DTYPE == DType.int8, "int8_gemm_v_to_cache: weight must be int8"
    comptime assert VcT.DTYPE == DType.int8, "int8_gemm_v_to_cache: V cache must be int8"
    print("    [stub] int8_gemm_v_to_cache pos=" + String(pos)
          + " seq_len=" + String(act_qi.seq_len)
          + " [" + String(act_qi.seq_len) + "x" + String(WT.COLS)
          + "] -> V cache[" + String(pos) + ".." + String(pos + act_qi.seq_len - 1) + "]")
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
    """Fused GATE+UP: reads int8 activation once, produces both bf16 outputs.

    Each output element computes two independent dot products against the
    same activation row — one for GATE, one for UP. Maximize reuse of
    loaded activation tiles across both weight matrices.
    """
    comptime assert QiT.DTYPE == DType.int8, "int8_gemm_gate_up: act must be int8"
    comptime assert GOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: gate output must be bf16"
    comptime assert UOutT.DTYPE == DType.bfloat16, "int8_gemm_gate_up: up output must be bf16"
    print("    [stub] int8_gemm_gate_up [" + String(act_qi.seq_len)
          + "x" + String(GWT.COLS) + "] -> gate[" + String(GWT.ROWS)
          + "] + up[" + String(UWT.ROWS) + "]")
    return PoolFence.completed()


# =============================================================================
# Kernel: int8_gqa_attention
#
# Full int8 GQA attention with fused RoPE on Q, pre-rotated int8 KV
# cache, V scale absorption, and direct int8 output.
#
# Per query row m, per head h (kv group g = h / GQA_FACTOR):
#   1. RoPE(Q[m, h], pos + m)
#   2. FWHT + quantize Q per head
#   3. Int8 scoring: score[t] = dot(qi_q, qi_K[t,g]) * scales / sqrt(HD)
#   4. Masked softmax in f32 (causal: m attends to 0..pos+m)
#   5. Absorb V per-entry scales into attention weights
#   6. Quantize absorbed weights to int8
#   7. Int8 aggregation: out[d] = sum_t qi_w[t] * qi_V[t,g,d]
#   8. Quantize output → int8 directly
#
# V scale absorption: w_abs[t] = softmax[t] * v_scale[t,g]
# This folds varying V scales into attention weights, enabling a single
# scalar w_sc for the aggregation epilogue (clean i32 accumulation).
# =============================================================================


def int8_gqa_attention[num_heads: Int, num_kv_heads: Int, head_dim: Int,
    QT: Encoding & Shaped,
    KcT: Encoding & Shaped, VcT: Encoding & Shaped,
    KsT: Encoding & Shaped, VsT: Encoding & Shaped,
    QiT: Encoding & Shaped, ScT: Encoding & Shaped,
    CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    q: DynView[QT],
    k_cache: CacheView[KcT], k_cache_scale: CacheView[KsT],
    v_cache: CacheView[VcT], v_cache_scale: CacheView[VsT],
    qi_out: DynView[QiT],
    scale_out: DynView[ScT],
    cos_table: Bound[CosT],
    sin_table: Bound[SinT],
    pos: Int,
    mut pool: BurstPool[],
) -> PoolFence:
    """Int8 GQA attention with fused RoPE on Q and pre-rotated int8 KV cache.

    Q is bf16 [seq_len, num_heads * head_dim] in original domain (no RoPE).
    K and V cache entries are int8, pre-rotated per head with per-head scales.

    pos is the cache position of the FIRST query in the batch:
      - Decode (seq_len=1): single query at position pos.
        Context = pos+1 cache entries. GEMV per head.
      - Prefill (seq_len>1): queries at positions pos..pos+seq_len-1.
        Causal mask: query m attends to cache positions 0..pos+m.
        GEMM per head with triangular mask.

    Output is int8 + f32 scale in per-head rotated domain. Feeds the O
    projection whose weight is rotated with block=HEAD_DIM — rotation
    cancels: <H*attn_out, H*O_weight[n]> = <attn_out, O_weight[n]>.
    """
    comptime assert QT.DTYPE == DType.bfloat16, "int8_gqa_attention: Q must be bf16"
    comptime assert KcT.DTYPE == DType.int8, "int8_gqa_attention: K cache must be int8"
    comptime assert VcT.DTYPE == DType.int8, "int8_gqa_attention: V cache must be int8"
    comptime assert QiT.DTYPE == DType.int8, "int8_gqa_attention: qi output must be int8"
    comptime assert ScT.DTYPE == DType.float32, "int8_gqa_attention: scale output must be f32"
    var ctx = pos + q.seq_len
    print("    [stub] int8_gqa_attention seq_len=" + String(q.seq_len)
          + " heads=" + String(num_heads)
          + " ctx=" + String(ctx)
          + (" (decode)" if q.seq_len == 1 else " (prefill)"))
    return PoolFence.completed()


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
    has no downstream projection to absorb into).
    """
    comptime assert InT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rms_norm_no_gamma: output must be bf16"
    print("    [stub] rms_norm_no_gamma [" + String(input.seq_len)
          + "x" + String(InT.COLS) + "]")
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
