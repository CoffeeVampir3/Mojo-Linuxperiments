"""Gemma 4 int8 MoE — expert kernel, rank-local dispatch, post-reduce combine.

Entry points used by the forward pass:

  gemma4_moe_dispatch_local — filter local experts, dispatch to expert_pool.
  Returns local_count. Caller owns the join.

  moe_combine — post-allreduce: rmsnorm(moe_out) + dense_normed →
  rmsnorm(combined) → residual add → layer_scalar. Called from a dispatched
  kernel wrapper (body threads never compute).

Expert weights are distributed round-robin (expert e on rank e % tp, local
index e // tp). All memory is rank-local.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from simd_math import exp_f32, sqrt, roundeven
from experimental2.kernels.rmsnorm_fwht_quantize import fwht_block
from experimental2.kernels.quantize import absmax_quantize_i8
from experimental2.kernels.int8_gemv import gemv_row, dot
from experimental_gemma.router import Gemma4TopKResult

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# GELU-tanh
# ============================================================================


@always_inline
def gelu_tanh_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    var inner = Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)
    var sig = 1.0 / (1.0 + exp_f32[width](-2.0 * inner))
    return 0.5 * x * (1.0 + 2.0 * sig - 1.0)


# ============================================================================
# Blocked GEMV — per-K-block activation scales for down projection
# ============================================================================


def gemv_row_blocked[N: Int, K: Int, fwht_block_size: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    """GEMV with per-K-block activation scales. Accumulates per block in i32,
    dequants to f32 per block, applies weight scale at the end."""
    comptime num_blocks = K // fwht_block_size
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_acc = InlineArray[SIMD[DType.float32, width], acc_count](
                fill=SIMD[DType.float32, width](0))
            for blk in range(num_blocks):
                var i32_acc = InlineArray[SIMD[DType.int32, width], acc_count](
                    fill=SIMD[DType.int32, width](0))
                for ks in range(0, fwht_block_size, VNNI_K_STEP):
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[p] = dot[width](i32_acc[p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos)
                        packed_off += VNNI_TILE_N * VNNI_BLK
                    for dc in range(VNNI_K_STEP // VNNI_BLK):
                        var k_pos = blk * fwht_block_size + ks + dc * VNNI_BLK
                        for p in range(passes_per_subtile):
                            i32_acc[passes_per_subtile + p] = dot[width](
                                i32_acc[passes_per_subtile + p], act_row,
                                wpacked + packed_off + p * bytes_per_pass, k_pos)
                        packed_off += VNNI_TILE_N * VNNI_BLK
                var blk_dequant = block_scales[blk] / 127.0
                for a in range(acc_count):
                    var n_base = nb + ns + a * width
                    var corrected = i32_acc[a].cast[DType.float32]() - 128.0 * (block_colsums + blk * N + n_base).load[width=width]()
                    f32_acc[a] += corrected * blk_dequant
            for a in range(acc_count):
                var n_base = nb + ns + a * width
                (dst + n_base).store(f32_acc[a] * (wsc + n_base).load[width=width]())


# ============================================================================
# Per-expert kernel — one BurstPool job
# ============================================================================


@fieldwise_init
struct Gemma4ExpertI8Args(Copyable, ImplicitlyCopyable):
    var act_i8: I8Ptr
    var act_scale: F32Ptr
    var gate_up_packed: U8Ptr
    var gate_up_wscale: F32Ptr
    var gate_up_colsum: F32Ptr
    var down_packed: U8Ptr
    var down_wscale: F32Ptr
    var down_block_colsum: F32Ptr
    var output: BF16Ptr
    var routing_weight: Float32


def gemma4_expert_i8_kernel[intermediate: Int, hidden: Int, fwht_blk: Int](
    args: Gemma4ExpertI8Args,
):
    """Fused int8 expert FFN: gate_up GEMV → gelu_tanh → FWHT+quantize → down GEMV.

    The hidden-side RMSNorm+γ+FWHT+quantize is hoisted out into a shared
    rank-local phase (rmsnorm_dual_gamma_fwht_quantize) so that all local
    experts on a rank consume one pre-quantized activation buffer.
    All intermediates on the stack.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime gate_up_dim = 2 * intermediate
    comptime num_blocks = intermediate // fwht_blk

    var dequant = args.act_scale[0] / 127.0

    # Phase 1: gate_up GEMV → f32 stack
    var gu_buf = InlineArray[Float32, gate_up_dim](fill=Float32(0))
    var work = UnsafePointer(to=gu_buf).bitcast[Float32]()
    gemv_row[gate_up_dim, hidden, DType.float32](
        args.act_i8,
        args.gate_up_packed,
        dequant,
        args.gate_up_wscale,
        args.gate_up_colsum,
        work.bitcast[Scalar[DType.float32]]())

    # Phase 2: gelu_tanh(gate) * up → reuse first half
    var up_f32 = work + intermediate
    var k = 0
    while k + width <= intermediate:
        var g = (work + k).load[width=width]()
        var u = (up_f32 + k).load[width=width]()
        (work + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    # Phase 3: FWHT + per-block quantize
    var qi_buf = InlineArray[Scalar[DType.int8], intermediate](uninitialized=True)
    var qi = UnsafePointer(to=qi_buf).bitcast[Scalar[DType.int8]]()
    var blk_scales = InlineArray[Float32, num_blocks](fill=Float32(0))
    var blk_sc = UnsafePointer(to=blk_scales).bitcast[Float32]()
    for b in range(num_blocks):
        fwht_block[fwht_blk](work + b * fwht_blk)
        blk_sc[b] = absmax_quantize_i8[fwht_blk](work + b * fwht_blk, qi + b * fwht_blk)

    # Phase 4: down GEMV with per-block scales
    var down_buf = InlineArray[Float32, hidden](fill=Float32(0))
    var down_ptr = UnsafePointer(to=down_buf).bitcast[Float32]()
    gemv_row_blocked[hidden, intermediate, fwht_blk](
        qi,
        args.down_packed,
        blk_sc,
        args.down_wscale,
        args.down_block_colsum,
        down_ptr)

    # Phase 5: routing weight → bf16 output
    var rw = args.routing_weight
    k = 0
    while k + width <= hidden:
        (args.output + k).store(((down_ptr + k).load[width=width]() * rw).cast[DType.bfloat16]())
        k += width


# ============================================================================
# Rank-local dispatch — filter, dispatch, join, accumulate
# ============================================================================


def gemma4_moe_dispatch_local[
    num_experts: Int, top_k: Int, intermediate: Int, hidden: Int,
    fwht_blk: Int, tp: Int, P: BurstThreadPool,
](
    act_i8: I8Ptr,
    act_scale: F32Ptr,
    routing: Gemma4TopKResult[top_k],
    gate_up_base: Int,
    gate_up_stride: Int,
    gate_up_sc_base: Int,
    gate_up_sc_stride: Int,
    gate_up_cs_base: Int,
    gate_up_cs_stride: Int,
    down_base: Int,
    down_stride: Int,
    down_sc_base: Int,
    down_sc_stride: Int,
    down_bcs_base: Int,
    down_bcs_stride: Int,
    expert_out_buf: BF16Ptr,
    rank: Int,
    mut pool: P,
) -> Int:
    """Dispatch this rank's local experts. Returns local_count.

    Caller must have already produced act_i8 / act_scale via the rank-local
    rmsnorm_dual_gamma_fwht_quantize phase using PRE_FFN_NORM_2 γ.
    Each weight field is a contiguous array across all experts with its own
    base and per-expert stride.
    """
    var local_count = 0
    var jobs = InlineArray[Gemma4ExpertI8Args, top_k](uninitialized=True)

    for s in range(top_k):
        var eid = routing.indices[s]
        if eid % tp != rank:
            continue
        var local_idx = eid // tp

        jobs[local_count] = Gemma4ExpertI8Args(
            act_i8,
            act_scale,
            U8Ptr(unsafe_from_address=gate_up_base + local_idx * gate_up_stride),
            F32Ptr(unsafe_from_address=gate_up_sc_base + local_idx * gate_up_sc_stride),
            F32Ptr(unsafe_from_address=gate_up_cs_base + local_idx * gate_up_cs_stride),
            U8Ptr(unsafe_from_address=down_base + local_idx * down_stride),
            F32Ptr(unsafe_from_address=down_sc_base + local_idx * down_sc_stride),
            F32Ptr(unsafe_from_address=down_bcs_base + local_idx * down_bcs_stride),
            expert_out_buf + local_count * hidden,
            routing.weights[s],
        )
        local_count += 1

    if local_count > 0:
        pool.dispatch[Gemma4ExpertI8Args, gemma4_expert_i8_kernel[intermediate, hidden, fwht_blk]](
            UnsafePointer(to=jobs[0]), local_count)

    return local_count


# ============================================================================
# Post-allreduce combine
# ============================================================================


def moe_combine[hidden: Int](
    moe_out: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    moe_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dense_normed: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    combine_norm_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    x_main: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    layer_scalar: Float32,
    eps: Float32,
):
    """Post-allreduce combine. All inline on 2816 elements.

    1. moe_normed = rmsnorm(moe_out, POST_FFN_NORM_2.γ)
    2. combined = moe_normed + dense_normed
    3. combined_normed = rmsnorm(combined, POST_FFN_NORM.γ)
    4. x_main = (x_main + combined_normed) * layer_scalar

    moe_out:       bf16[hidden] — allreduced MoE output (read, overwritten as scratch)
    dense_normed:  bf16[hidden] — pre-computed rmsnorm(dense_out, POST_FFN_NORM_1.γ)
    x_main:        bf16[hidden] — residual stream (read-modify-write)
    """
    comptime width = simd_width_of[DType.float32]()

    # Step 1: rmsnorm(moe_out, γ₂) → overwrite moe_out with moe_normed
    var sum_sq = SIMD[DType.float32, width](0)
    var i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    var inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (moe_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 2: combined = moe_normed + dense_normed (in-place into moe_out)
    i = 0
    while i + width <= hidden:
        var m = (moe_out + i).load[width=width]().cast[DType.float32]()
        var d = (dense_normed + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((m + d).cast[DType.bfloat16]())
        i += width

    # Step 3: rmsnorm(combined, γ₃)
    sum_sq = SIMD[DType.float32, width](0)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        sum_sq = v.fma(v, sum_sq)
        i += width
    inv_rms = 1.0 / sqrt[DType.float32, 1](sum_sq.reduce_add() / Float32(hidden) + eps)
    i = 0
    while i + width <= hidden:
        var v = (moe_out + i).load[width=width]().cast[DType.float32]()
        var g = (combine_norm_w + i).load[width=width]().cast[DType.float32]()
        (moe_out + i).store((v * inv_rms * g).cast[DType.bfloat16]())
        i += width

    # Step 4: x_main = (x_main + combined_normed) * layer_scalar
    i = 0
    while i + width <= hidden:
        var x = (x_main + i).load[width=width]().cast[DType.float32]()
        var c = (moe_out + i).load[width=width]().cast[DType.float32]()
        (x_main + i).store(((x + c) * layer_scalar).cast[DType.bfloat16]())
        i += width


# ============================================================================
# Validation
# ============================================================================


def scalar_fwht_f64(buf: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var s = 1
    while s < n:
        var i = 0
        while i < n:
            for j in range(s):
                var a = buf[i + j]
                var b = buf[i + j + s]
                buf[i + j] = a + b
                buf[i + j + s] = a - b
            i += s * 2
        s *= 2
    var sc = 1.0 / Float64(sqrt[DType.float64, 1](Float64(n)))
    for i in range(n):
        buf[i] *= sc


def xorshift64(mut state: UInt64) -> Float64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float64(Int64(state & 0xFFFFFF).cast[DType.float64]()) / Float64(0x800000) * 4.0 - 2.0


def validate_expert_pipeline():
    """Validate one expert's full i8 pipeline against scalar f64 reference."""
    comptime intermediate = 704
    comptime hidden = 2816
    comptime gate_up_dim = 1408
    comptime fwht_blk = 64
    comptime num_blocks = intermediate // fwht_blk

    var rng = UInt64(0xCAFEDEAD12345678)

    var act_i8 = alloc[Scalar[DType.int8]](hidden)
    var act_scale = Float32(15.0)
    for i in range(hidden):
        act_i8[i] = Scalar[DType.int8](Int8(Int(xorshift64(rng) * 60.0)))

    var gate_up_w = alloc[Float32](gate_up_dim * hidden)
    var down_w = alloc[Float32](hidden * intermediate)
    for i in range(gate_up_dim * hidden):
        gate_up_w[i] = Float32(xorshift64(rng) * 0.01)
    for i in range(hidden * intermediate):
        down_w[i] = Float32(xorshift64(rng) * 0.01)

    var routing_weight = Float32(0.15)
    var act_dequant = Float64(act_scale) / 127.0

    # f64 reference: gate_up → gelu_tanh → FWHT+quantize → dequant → down
    var gu_f64 = alloc[Float64](gate_up_dim)
    for n in range(gate_up_dim):
        var acc = Float64(0)
        for k in range(hidden):
            acc += Float64(Int(act_i8[k])) * act_dequant * Float64(gate_up_w[n * hidden + k])
        gu_f64[n] = acc

    var activated = alloc[Float64](intermediate)
    for i in range(intermediate):
        var g = gu_f64[i]
        var inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        var e = Float64(exp_f32[1](Float32(-2.0 * inner)))
        activated[i] = 0.5 * g * (1.0 + (1.0 - e) / (1.0 + e)) * gu_f64[intermediate + i]

    # FWHT + per-block quantize round-trip
    var fwht_buf = alloc[Float64](intermediate)
    for i in range(intermediate):
        fwht_buf[i] = activated[i]

    for b in range(num_blocks):
        scalar_fwht_f64(fwht_buf + b * fwht_blk, fwht_blk)

    var qi = alloc[Scalar[DType.int8]](intermediate)
    var bsc = alloc[Float64](num_blocks)
    for b in range(num_blocks):
        var bmax = Float64(0)
        for j in range(fwht_blk):
            var a = fwht_buf[b * fwht_blk + j].__abs__()
            if a > bmax:
                bmax = a
        if bmax < 1e-10:
            bmax = 1e-10
        bsc[b] = bmax
        var inv = 127.0 / bmax
        for j in range(fwht_blk):
            qi[b * fwht_blk + j] = Scalar[DType.int8](Int(Float64(roundeven[DType.float64, 1](fwht_buf[b * fwht_blk + j] * inv)).clamp(-128.0, 127.0)))

    var dequant = alloc[Float64](intermediate)
    for b in range(num_blocks):
        var dq = bsc[b] / 127.0
        for j in range(fwht_blk):
            dequant[b * fwht_blk + j] = Float64(Int(qi[b * fwht_blk + j])) * dq
    for b in range(num_blocks):
        scalar_fwht_f64(dequant + b * fwht_blk, fwht_blk)

    # Down matmul
    var output_quant = alloc[Float64](hidden)
    var output_exact = alloc[Float64](hidden)
    for n in range(hidden):
        var acc_q = Float64(0)
        var acc_e = Float64(0)
        for k in range(intermediate):
            acc_q += dequant[k] * Float64(down_w[n * intermediate + k])
            acc_e += activated[k] * Float64(down_w[n * intermediate + k])
        output_quant[n] = acc_q * Float64(routing_weight)
        output_exact[n] = acc_e * Float64(routing_weight)

    # Report
    var dot_val = Float64(0)
    var n_e = Float64(0)
    var n_q = Float64(0)
    var sq_err = Float64(0)
    for i in range(hidden):
        dot_val += output_exact[i] * output_quant[i]
        n_e += output_exact[i] * output_exact[i]
        n_q += output_quant[i] * output_quant[i]
        sq_err += (output_exact[i] - output_quant[i]) * (output_exact[i] - output_quant[i])

    print("  output RMS:    " + String(Float64(sqrt[DType.float64, 1](n_e / Float64(hidden)))))
    print("  cosine:        " + String(dot_val / (Float64(sqrt[DType.float64, 1](n_e)) * Float64(sqrt[DType.float64, 1](n_q)))))
    print("  NRMSE:         " + String(Float64(sqrt[DType.float64, 1](sq_err / n_e))))

    act_i8.free()
    gate_up_w.free()
    down_w.free()
    gu_f64.free()
    activated.free()
    fwht_buf.free()
    qi.free()
    bsc.free()
    dequant.free()
    output_quant.free()
    output_exact.free()


def main():
    print("=== Gemma4 MoE expert pipeline validation ===")
    print("\nSingle expert i8 pipeline (gate_up → gelu_tanh → FWHT → down):")
    validate_expert_pipeline()
