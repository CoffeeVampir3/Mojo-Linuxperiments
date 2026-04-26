from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of

from experimental3.common_math import F32Ptr
from gated_delta_net.gdn import (
    GdnChunkArgs, l2_norm_inplace_f32, dot_row_f32,
)
from simd_math import exp_f32


# ============================================================================
# Optimized chunk kernel
# ----------------------------------------------------------------------------
# Same math, same args, same state semantics as gdn_chunk_kernel; rewritten so
# the state matrix S is touched once (one read, one write) per element instead
# of six passes, K is touched ~3x instead of ~32x, and intermediate U/W/vp
# buffers are folded into the work loop.
#
# Memory traffic comparison (CHUNK=64, DK=DV=128):
#   Reference:   S ~384 KB, K ~32 MB, V ~8 MB, plus U/W/vp materialization
#   Optimized:   S ~128 KB, K ~512 KB, V ~32 KB, no separate U/W/vp loads
# A 30-50x reduction in cache traffic for the same FMA count.
# ============================================================================


@always_inline
def gdn_opt_chunk_kernel[
    CHUNK: Int, DK: Int, DV: Int,
    BV: Int = 2 * simd_width_of[DType.float32](),
    BR: Int = 4,
    l2_norm_qk: Bool = True,
](args: GdnChunkArgs, l2_eps: Float32):
    """One chunk of CHUNK tokens through the gated delta rule, optimized.

    Tile-fused chunk-wise WY-form forward pass. Same math as
    `gdn_chunk_kernel` (see gdn.mojo for the equation derivation); restructured
    for cache and register efficiency.

    Layout assumptions:
      - State row-major (DK, DV); S[d, m] at state + d*DV + m.
      - Q, K row-major (CHUNK, DK); modified in place (L2 norm, then prescale).
      - V row-major (CHUNK, DV).

    Tile parameters (defaults work for AVX-512 / AVX-2 / NEON):
      - BV: state-column tile width. Default 2x SIMD width (32 on AVX-512,
        16 on AVX-2). Each tile keeps a (DK, BV) slab of S resident in L1
        through both the read and write halves of the chunk update.
      - BR: output row tile (small, register-resident). 4 fits AVX-512's
        register file with both v_prime and out_state accumulator banks.

    Caller contract:
      - Q and K are L2-normalized then prescaled in place; do not reuse them
        after the call. V, beta, g are read-only. State and out are written.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime assert DK % width == 0, "DK must be SIMD-aligned"
    comptime assert DV % width == 0, "DV must be SIMD-aligned"
    comptime assert BV % width == 0, "BV must be SIMD-aligned"
    comptime assert DV % BV == 0, "DV must be a multiple of BV"
    comptime assert CHUNK % BR == 0, "CHUNK must be a multiple of BR"
    comptime VPR = BV // width  # SIMD vectors per tile row

    # ------------------------------------------------------------------------
    # Phase 1: preamble (single pass over data; runs once per chunk)
    # ------------------------------------------------------------------------

    # Step 0: L2-normalize Q and K rows.
    comptime if l2_norm_qk:
        for r in range(CHUNK):
            l2_norm_inplace_f32[DK](args.q + r * DK, l2_eps)
            l2_norm_inplace_f32[DK](args.k + r * DK, l2_eps)

    # Step 1: cumulative log-decay and its exp.
    var decay_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var dexp_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var carry_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var acc = Float32(0)
    for r in range(CHUNK):
        acc += args.g[r]
        decay_arr[r] = acc
        dexp_arr[r] = Float32(exp_f32[1](acc))
    var decay_C = decay_arr[CHUNK - 1]
    var dexp_C = dexp_arr[CHUNK - 1]
    for r in range(CHUNK):
        # carry_arr[r] = exp(decay_C - decay[r]) = γ_C / γ_r, in (0, 1].
        carry_arr[r] = Float32(exp_f32[1](decay_C - decay_arr[r]))

    # Step 2: build A in `mat` as -strictLower(diag(β) (Γ ⊙ K K^T)).
    # The Gram-Schmidt step inverts (I + A); initializing with -A makes the
    # standard inversion loop produce (I + A)^{-1} directly.
    var mat_arr = InlineArray[Float32, CHUNK * CHUNK](fill=Float32(0))
    var mat = UnsafePointer(to=mat_arr).bitcast[Float32]()
    for i in range(1, CHUNK):
        var nb = -args.beta[i]
        var de_i = decay_arr[i]
        var ki = args.k + i * DK
        for j in range(i):
            var kk = dot_row_f32[DK](ki, args.k + j * DK)
            var lij = Float32(exp_f32[1](de_i - decay_arr[j]))
            mat[i * CHUNK + j] = nb * kk * lij

    # Step 3: forward-substitution inversion, then diagonal patch.
    for i in range(1, CHUNK):
        var row_i = mat + i * CHUNK
        for j in range(i):
            var s = Float32(0)
            for m in range(i):
                s += row_i[m] * mat[m * CHUNK + j]
            row_i[j] += s
    for i in range(CHUNK):
        mat[i * CHUNK + i] = Float32(1)

    # Step 4: fused U + W. Both share the (mat[i,j] β[j]) factor; W also
    # picks up γ[j] = dexp_arr[j]. Each (V row, K row) read once.
    var u_arr = InlineArray[Float32, CHUNK * DV](fill=Float32(0))
    var w_arr = InlineArray[Float32, CHUNK * DK](fill=Float32(0))
    var u = UnsafePointer(to=u_arr).bitcast[Float32]()
    var w = UnsafePointer(to=w_arr).bitcast[Float32]()
    for j in range(CHUNK):
        var beta_j = args.beta[j]
        var dexp_j = dexp_arr[j]
        var v_j = args.v + j * DV
        var k_j = args.k + j * DK
        for i in range(j, CHUNK):
            var t_ij = mat[i * CHUNK + j]
            var coeff_u = SIMD[DType.float32, width](t_ij * beta_j)
            var coeff_w = SIMD[DType.float32, width](t_ij * beta_j * dexp_j)
            var u_i = u + i * DV
            var w_i = w + i * DK
            var m = 0
            while m < DV:
                (u_i + m).store(
                    (v_j + m).load[width=width]().fma(
                        coeff_u, (u_i + m).load[width=width]()))
                m += width
            var d = 0
            while d < DK:
                (w_i + d).store(
                    (k_j + d).load[width=width]().fma(
                        coeff_w, (w_i + d).load[width=width]()))
                d += width

    # Step 5: rebuild attn_local in `mat` (overwrites T_inv; we no longer need
    # it). attn_local[r, s] = (q[r] · k[s]) * exp(decay[r] - decay[s]) for r >= s.
    for i in range(CHUNK):
        var qi = args.q + i * DK
        var de_i = decay_arr[i]
        var row_i = mat + i * CHUNK
        for j in range(i + 1):
            var qk = dot_row_f32[DK](qi, args.k + j * DK)
            var lij = Float32(exp_f32[1](de_i - decay_arr[j]))
            row_i[j] = qk * lij
        for j in range(i + 1, CHUNK):
            row_i[j] = Float32(0)

    # Step 6: prescale Q and K in place. After this:
    #   Q[i, d] holds γ_i * q[i, d]               (used in out_state read of S)
    #   K[r, d] holds (γ_C / γ_r) * k[r, d]       (used in state carry write)
    for i in range(CHUNK):
        var qi = args.q + i * DK
        var dexp_i_v = SIMD[DType.float32, width](dexp_arr[i])
        var d = 0
        while d < DK:
            (qi + d).store((qi + d).load[width=width]() * dexp_i_v)
            d += width
    for r in range(CHUNK):
        var kr = args.k + r * DK
        var c_v = SIMD[DType.float32, width](carry_arr[r])
        var d = 0
        while d < DK:
            (kr + d).store((kr + d).load[width=width]() * c_v)
            d += width

    # ------------------------------------------------------------------------
    # Phase 2: state-tile loop.
    #
    # For each BV-wide column tile of S:
    #   A. Read S_tile (DK × BV) into L1; fused vp + out_state accumulation.
    #   B. v_new[i, m] = U[i, m] - vp[i, m].
    #   C. out[i, m] = out_state[i, m] + sum_s attn_local[i, s] * v_new[s, m].
    #   D. S_tile <- γ_C * S_tile + sum_r K_scaled[r, :] · v_new[r, :].
    #
    # vp_tile, out_state_tile, v_new_tile are all (CHUNK, BV) = small scratch.
    # ------------------------------------------------------------------------

    var vp_tile_arr = InlineArray[Float32, CHUNK * BV](uninitialized=True)
    var os_tile_arr = InlineArray[Float32, CHUNK * BV](uninitialized=True)
    var vn_tile_arr = InlineArray[Float32, CHUNK * BV](uninitialized=True)
    var vp_tile = UnsafePointer(to=vp_tile_arr).bitcast[Float32]()
    var os_tile = UnsafePointer(to=os_tile_arr).bitcast[Float32]()
    var vn_tile = UnsafePointer(to=vn_tile_arr).bitcast[Float32]()

    var v_dC = SIMD[DType.float32, width](dexp_C)

    var dv_tile = 0
    while dv_tile < DV:
        # ---- Subphase A: fused vp + out_state via single S_tile read ----
        # Outer over BR-row blocks of i. For each block, hold accumulators in
        # registers across the d loop so each S_tile row loads once.
        var i_blk = 0
        while i_blk < CHUNK:
            var vp_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            var os_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            comptime for r in range(BR):
                comptime for v in range(VPR):
                    vp_acc[r * VPR + v] = SIMD[DType.float32, width](0)
                    os_acc[r * VPR + v] = SIMD[DType.float32, width](0)

            for d in range(DK):
                # Load one BV-wide S row (VPR SIMD vectors).
                var s_row = InlineArray[
                    SIMD[DType.float32, width], VPR
                ](uninitialized=True)
                comptime for v in range(VPR):
                    s_row[v] = (args.state + d * DV + dv_tile + v * width
                        ).load[width=width]()

                comptime for r in range(BR):
                    var i = i_blk + r
                    var w_b = SIMD[DType.float32, width]((w + i * DK)[d])
                    var sq_b = SIMD[DType.float32, width]((args.q + i * DK)[d])
                    comptime for v in range(VPR):
                        vp_acc[r * VPR + v] = w_b.fma(
                            s_row[v], vp_acc[r * VPR + v])
                        os_acc[r * VPR + v] = sq_b.fma(
                            s_row[v], os_acc[r * VPR + v])

            comptime for r in range(BR):
                comptime for v in range(VPR):
                    (vp_tile + (i_blk + r) * BV + v * width).store(
                        vp_acc[r * VPR + v])
                    (os_tile + (i_blk + r) * BV + v * width).store(
                        os_acc[r * VPR + v])
            i_blk += BR

        # ---- Subphase B: v_new = U_tile - vp_tile ----
        # U_tile is U[:, dv_tile:dv_tile+BV]. Read from u (CHUNK, DV).
        for i in range(CHUNK):
            var u_i = u + i * DV + dv_tile
            var vp_i = vp_tile + i * BV
            var vn_i = vn_tile + i * BV
            var bv = 0
            while bv < BV:
                (vn_i + bv).store(
                    (u_i + bv).load[width=width]() -
                    (vp_i + bv).load[width=width]())
                bv += width

        # ---- Subphase C: out += out_state + attn_local @ v_new ----
        # Outer BR row tiles. attn_local[r, s] is L1-resident from phase 1.
        var i_blk2 = 0
        while i_blk2 < CHUNK:
            var oi_acc = InlineArray[
                SIMD[DType.float32, width], BR * VPR
            ](uninitialized=True)
            comptime for r in range(BR):
                comptime for v in range(VPR):
                    # Initialize from out_state (already read during subphase A).
                    oi_acc[r * VPR + v] = (os_tile + (i_blk2 + r) * BV
                        + v * width).load[width=width]()

            # Sum over s in [0, max_i_in_block]. The inner block contains rows
            # i_blk2..i_blk2+BR-1; only s <= i contribute. Iterate s up to the
            # block max and mask via per-row bounds on the scalar broadcast.
            var s_max = i_blk2 + BR - 1
            for s in range(s_max + 1):
                var v_row = InlineArray[
                    SIMD[DType.float32, width], VPR
                ](uninitialized=True)
                comptime for v in range(VPR):
                    v_row[v] = (vn_tile + s * BV + v * width
                        ).load[width=width]()

                comptime for r in range(BR):
                    # When s > i_blk2 + r the entry attn_local[i, s] is zero
                    # by construction (rebuilt in step 5 with the upper triangle
                    # explicitly zeroed), so the FMA is a guaranteed no-op. We
                    # still issue it to keep the unrolled schedule branch-free.
                    var al = SIMD[DType.float32, width](
                        mat[(i_blk2 + r) * CHUNK + s])
                    comptime for v in range(VPR):
                        oi_acc[r * VPR + v] = al.fma(
                            v_row[v], oi_acc[r * VPR + v])

            comptime for r in range(BR):
                comptime for v in range(VPR):
                    (args.out + (i_blk2 + r) * DV + dv_tile + v * width
                        ).store(oi_acc[r * VPR + v])
            i_blk2 += BR

        # ---- Subphase D: S_tile <- γ_C * S_tile + K_scaled^T @ v_new ----
        # K is already pre-scaled to hold (γ_C / γ_r) * k[r, d] from step 6.
        # For each d, S_tile[d, :BV] is read once, modified, written once.
        for d in range(DK):
            var s_off = args.state + d * DV + dv_tile
            var s_row = InlineArray[
                SIMD[DType.float32, width], VPR
            ](uninitialized=True)
            comptime for v in range(VPR):
                s_row[v] = (s_off + v * width).load[width=width]() * v_dC

            for r in range(CHUNK):
                var k_rd = SIMD[DType.float32, width]((args.k + r * DK)[d])
                comptime for v in range(VPR):
                    s_row[v] = k_rd.fma(
                        (vn_tile + r * BV + v * width).load[width=width](),
                        s_row[v])

            comptime for v in range(VPR):
                (s_off + v * width).store(s_row[v])

        dv_tile += BV

    _ = decay_arr
    _ = dexp_arr
    _ = carry_arr
    _ = mat_arr
    _ = u_arr
    _ = w_arr
    _ = vp_tile_arr
    _ = os_tile_arr
    _ = vn_tile_arr
