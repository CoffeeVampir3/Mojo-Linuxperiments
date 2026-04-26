from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of

from experimental3.common_math import F32Ptr
from gated_delta_net.gdn import (
    GdnChunkArgs, l2_norm_inplace_f32, dot_row_f32,
)
from simd_math import exp_f32


# ============================================================================
# Final-form chunk kernel.
# ----------------------------------------------------------------------------
# Implements the algebraically reduced gated-delta chunk update:
#
#     A_0 = D_β · strictLower(K K^T)                  decay-free WY system
#     for each BV-wide column tile of S_0:
#         (QS, KS) = stacked GEMM [Q; K] · S_0[:, tile]
#         Δ        = D_{β/γ} V[:, tile]  -  D_β · KS                  (fused)
#         (I + A_0) y = Δ                                              (fwd solve)
#         O[:, tile]    = D_γ · (QS  +  (Q K^T ⊙ M) · y)
#         S_0[:, tile]  = γ_C · (S_0[:, tile]  +  K^T · y)
#
# The reduction folds out W, T̃, T, U, Λ, v_new, K̂ — none of those are stored.
# All decay enters as four diagonal/scalar maps (D_γ, D_{β/γ}, D_β, γ_C);
# the WY inversion is on the decay-free A_0 and never materialized as a matrix.
#
# Three true dense GEMMs:
#   (1) stacked [Q; K] · S_0  (one (2C × d_k × d_v) call, S_0 amortized 2x)
#   (2) Q K^T  (precomputed, lower-tri only)
#   (3) K^T y  (state carry)
# Plus one triangular forward solve for y, and one causal-masked attention.
#
# Memory traffic per chunk: lower-bound ~2·d_k·d_v + C·(2 d_k + d_v) for inputs,
# achieved by tile-fusing the read/update of S_0 across all phases that touch it.
# ============================================================================


@always_inline
def gdn_final_chunk_kernel[
    CHUNK: Int, DK: Int, DV: Int,
    BV: Int = 2 * simd_width_of[DType.float32](),
    BR: Int = 4,
    l2_norm_qk: Bool = True,
](args: GdnChunkArgs, l2_eps: Float32):
    """One chunk through the gated delta rule, algebraically minimal form.

    Caller contract:
      - Q and K are L2-normalized in place. V is NOT mutated.
      - State row-major (DK, DV); updated in place.
      - Output row-major (CHUNK, DV); written.

    Tile parameters:
      - BV: state column-tile width. Default 2x SIMD width, sized so the
        per-tile S_0 slab (DK*BV*4 bytes) plus working tiles fit in L1.
      - BR: register-tile of output rows. 4 fits AVX-512 with two accumulator
        banks (qs+ks for Phase A, single bank for Phase D).
    """
    comptime width = simd_width_of[DType.float32]()
    comptime assert DK % width == 0, "DK must be SIMD-aligned"
    comptime assert DV % width == 0, "DV must be SIMD-aligned"
    comptime assert BV % width == 0, "BV must be SIMD-aligned"
    comptime assert DV % BV == 0, "DV must be a multiple of BV"
    comptime assert CHUNK % BR == 0, "CHUNK must be a multiple of BR"
    comptime VPR = BV // width

    # ------------------------------------------------------------------------
    # Phase 0: setup (mutates Q, K via L2-norm; computes scalar arrays).
    # ------------------------------------------------------------------------
    comptime if l2_norm_qk:
        for r in range(CHUNK):
            l2_norm_inplace_f32[DK](args.q + r * DK, l2_eps)
            l2_norm_inplace_f32[DK](args.k + r * DK, l2_eps)

    var gamma_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var bg_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var acc = Float32(0)
    for r in range(CHUNK):
        acc += args.g[r]
        var gamma_r = Float32(exp_f32[1](acc))
        gamma_arr[r] = gamma_r
        bg_arr[r] = args.beta[r] / gamma_r
    var gamma_C = gamma_arr[CHUNK - 1]

    # ------------------------------------------------------------------------
    # Phase 1: build A_0 = D_β · strictLower(K K^T).
    # A_0[i, j] = β[i] · (k[i] · k[j])  for i > j, else 0.
    # No exponentials, no decay-aware mask — pure dot products and a row-scale.
    # ------------------------------------------------------------------------
    var a0_arr = InlineArray[Float32, CHUNK * CHUNK](fill=Float32(0))
    var a0 = UnsafePointer(to=a0_arr).bitcast[Float32]()
    for i in range(1, CHUNK):
        var ki = args.k + i * DK
        var bi = args.beta[i]
        for j in range(i):
            a0[i * CHUNK + j] = bi * dot_row_f32[DK](ki, args.k + j * DK)

    # ------------------------------------------------------------------------
    # Phase 2: build qk_mat = lowerTri(Q K^T).
    # qk[r, j] = q[r] · k[j]  for j ≤ r, else 0.
    # Used by Phase D's masked attention; precomputed once, reused per tile.
    # ------------------------------------------------------------------------
    var qk_arr = InlineArray[Float32, CHUNK * CHUNK](fill=Float32(0))
    var qk = UnsafePointer(to=qk_arr).bitcast[Float32]()
    for r in range(CHUNK):
        var qr = args.q + r * DK
        for j in range(r + 1):
            qk[r * CHUNK + j] = dot_row_f32[DK](qr, args.k + j * DK)

    # ------------------------------------------------------------------------
    # Phase 3: state-tile loop. For each BV-wide column slab of S_0, run the
    # complete (read → solve → output → carry → write) sequence with the slab
    # held L1-resident the entire time.
    # ------------------------------------------------------------------------
    var y_arr = InlineArray[Float32, CHUNK * BV](uninitialized=True)
    var y_buf = UnsafePointer(to=y_arr).bitcast[Float32]()

    var dv_tile = 0
    while dv_tile < DV:
        # ====================================================================
        # Phase A: stacked GEMM [Q; K] · S_0[:, tile] → (qs0, ks0).
        # Each S_0 row is loaded once and feeds both accumulator banks.
        # qs0 lands directly in the output buffer (will be summed with the
        # masked-attn term in Phase D); ks0 lands in y_buf (will become Δ then y).
        # ====================================================================
        var ir = 0
        while ir < CHUNK:
            var qs_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            var ks_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            comptime for r in range(BR):
                comptime for v in range(VPR):
                    qs_acc[r * VPR + v] = SIMD[DType.float32, width](0)
                    ks_acc[r * VPR + v] = SIMD[DType.float32, width](0)

            for d in range(DK):
                var s_row = InlineArray[
                    SIMD[DType.float32, width], VPR
                ](uninitialized=True)
                comptime for v in range(VPR):
                    s_row[v] = (args.state + d * DV + dv_tile + v * width
                        ).load[width=width]()

                comptime for r in range(BR):
                    var i = ir + r
                    var q_b = SIMD[DType.float32, width](
                        (args.q + i * DK)[d])
                    var k_b = SIMD[DType.float32, width](
                        (args.k + i * DK)[d])
                    comptime for v in range(VPR):
                        qs_acc[r * VPR + v] = q_b.fma(
                            s_row[v], qs_acc[r * VPR + v])
                        ks_acc[r * VPR + v] = k_b.fma(
                            s_row[v], ks_acc[r * VPR + v])

            comptime for r in range(BR):
                comptime for v in range(VPR):
                    (args.out + (ir + r) * DV + dv_tile + v * width).store(
                        qs_acc[r * VPR + v])
                    (y_buf + (ir + r) * BV + v * width).store(
                        ks_acc[r * VPR + v])
            ir += BR

        # ====================================================================
        # Phase B: in-place residual.  y_buf := (β/γ) · V[:, tile]  -  β · y_buf.
        # ====================================================================
        for r in range(CHUNK):
            var bg_v = SIMD[DType.float32, width](bg_arr[r])
            var nb_v = SIMD[DType.float32, width](-args.beta[r])
            var v_row = args.v + r * DV + dv_tile
            var y_row = y_buf + r * BV
            comptime for v in range(VPR):
                var v_in = (v_row + v * width).load[width=width]() * bg_v
                var ks = (y_row + v * width).load[width=width]()
                (y_row + v * width).store(nb_v.fma(ks, v_in))

        # ====================================================================
        # Phase C: in-place forward solve.  y[r] -= sum_{j<r} A_0[r,j] · y[j].
        # Single sequential dependency over r; SIMD over BV inside.
        # y_r is held in registers across the j-loop, reducing memory traffic.
        # ====================================================================
        for r in range(1, CHUNK):
            var y_r_regs = InlineArray[
                SIMD[DType.float32, width], VPR
            ](uninitialized=True)
            comptime for v in range(VPR):
                y_r_regs[v] = (y_buf + r * BV + v * width
                    ).load[width=width]()

            for j in range(r):
                var coeff = SIMD[DType.float32, width](
                    -a0[r * CHUNK + j])
                comptime for v in range(VPR):
                    y_r_regs[v] = coeff.fma(
                        (y_buf + j * BV + v * width).load[width=width](),
                        y_r_regs[v])

            comptime for v in range(VPR):
                (y_buf + r * BV + v * width).store(y_r_regs[v])

        # ====================================================================
        # Phase D: O[:, tile] = D_γ · (Q·S_0[:, tile]  +  (Q K^T ⊙ M) · y).
        # The Q·S_0 contribution is already in args.out from Phase A.
        # We accumulate the masked-attention contribution and apply D_γ in one pass.
        # ====================================================================
        var ir2 = 0
        while ir2 < CHUNK:
            var o_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            comptime for r in range(BR):
                comptime for v in range(VPR):
                    o_acc[r * VPR + v] = (args.out + (ir2 + r) * DV
                        + dv_tile + v * width).load[width=width]()

            # Iterate j only up to the maximum row in this BR-block.
            # Entries with j > ir2 + r are zero by construction (qk's strict
            # upper triangle is left at zero from the fill=0 initialization).
            for j in range(ir2 + BR):
                var y_row = InlineArray[
                    SIMD[DType.float32, width], VPR
                ](uninitialized=True)
                comptime for v in range(VPR):
                    y_row[v] = (y_buf + j * BV + v * width
                        ).load[width=width]()

                comptime for r in range(BR):
                    var coeff = SIMD[DType.float32, width](
                        qk[(ir2 + r) * CHUNK + j])
                    comptime for v in range(VPR):
                        o_acc[r * VPR + v] = coeff.fma(
                            y_row[v], o_acc[r * VPR + v])

            comptime for r in range(BR):
                var gamma_v = SIMD[DType.float32, width](gamma_arr[ir2 + r])
                comptime for v in range(VPR):
                    (args.out + (ir2 + r) * DV + dv_tile + v * width).store(
                        o_acc[r * VPR + v] * gamma_v)
            ir2 += BR

        # ====================================================================
        # Phase E: state carry.  S_0[:, tile] = γ_C · (S_0[:, tile] + K^T · y).
        # S_0 slab is read once into registers, accumulated, scalar-scaled,
        # and written back. Achieves the lower bound of one read + one write
        # per S_0 element across the entire chunk.
        # ====================================================================
        var gamma_C_v = SIMD[DType.float32, width](gamma_C)
        for d in range(DK):
            var s_off = args.state + d * DV + dv_tile
            var s_row = InlineArray[
                SIMD[DType.float32, width], VPR
            ](uninitialized=True)
            comptime for v in range(VPR):
                s_row[v] = (s_off + v * width).load[width=width]()

            for r in range(CHUNK):
                var k_b = SIMD[DType.float32, width](
                    (args.k + r * DK)[d])
                comptime for v in range(VPR):
                    s_row[v] = k_b.fma(
                        (y_buf + r * BV + v * width).load[width=width](),
                        s_row[v])

            comptime for v in range(VPR):
                (s_off + v * width).store(s_row[v] * gamma_C_v)

        dv_tile += BV

    _ = gamma_arr
    _ = bg_arr
    _ = a0_arr
    _ = qk_arr
    _ = y_arr
