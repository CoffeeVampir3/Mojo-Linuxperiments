from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import simd_width_of

from experimental3.common_math import F32Ptr, rms_reduce_f32
from simd_math import exp_f32, sqrt
from simd_math.matrixops import pick_port_unroll, tree_reduce_accs


# ============================================================================
# Helpers
# ============================================================================


@always_inline
def l2_norm_inplace_f32[N: Int](x: F32Ptr, eps: Float32):
    """In-place L2 normalize a length-N f32 row: x /= sqrt(sum_sq(x) + eps).

    L2 (not RMS): no division by N inside the sqrt. Matches FLA's
    `l2norm(x, dim=-1, eps=1e-6)` used by the gated delta-rule kernel for
    per-head Q and K stabilization.
    """
    comptime width = simd_width_of[DType.float32]()
    var sum_sq = rms_reduce_f32[N](x)
    var inv = Float32(1.0) / sqrt[DType.float32, 1](sum_sq + eps)
    var vinv = SIMD[DType.float32, width](inv)
    var k = 0
    while k < N:
        (x + k).store((x + k).load[width=width]() * vinv)
        k += width


@always_inline
def dot_row_f32[N: Int](a: F32Ptr, b: F32Ptr) -> Float32:
    """Inner product sum_i a[i] * b[i] over a length-N f32 row."""
    comptime width = simd_width_of[DType.float32]()
    comptime port_unroll = pick_port_unroll[width, N]()
    comptime step = port_unroll * width
    var accs = InlineArray[SIMD[DType.float32, width], port_unroll](
        uninitialized=True)
    comptime for i in range(port_unroll):
        var off = i * width
        accs[i] = (a + off).load[width=width]() * (b + off).load[width=width]()
    var k = step
    while k + step <= N:
        comptime for i in range(port_unroll):
            var off = k + i * width
            accs[i] = (a + off).load[width=width]().fma(
                (b + off).load[width=width](), accs[i])
        k += step
    while k + width <= N:
        accs[0] = (a + k).load[width=width]().fma(
            (b + k).load[width=width](), accs[0])
        k += width
    return tree_reduce_accs(accs)


# ============================================================================
# Recurrent step (decode path)
# ============================================================================


@fieldwise_init
struct GdnRecurrentArgs(Copyable, ImplicitlyCopyable):
    var q: F32Ptr            # (DK,)
    var k: F32Ptr            # (DK,)         L2-normed by caller if desired
    var v: F32Ptr            # (DV,)
    var beta: Float32        # sigmoid(b)
    var g: Float32           # log decay, < 0
    var state: F32Ptr        # (DK, DV)      updated in-place, row-major
    var out: F32Ptr          # (DV,)


def gdn_recurrent_step[DK: Int, DV: Int](args: GdnRecurrentArgs):
    """Single-token gated delta rule update for one head.

        S      *= exp(g)                    decay
        v_pred  = S^T k                     state's prediction at this key
        v_new   = beta * (v - v_pred)       SGD residual
        S      += k v_new^T                 rank-1 write
        out     = S^T q                     read

    State is row-major (DK, DV); S[d, m] at state + d*DV + m. DV must be a
    multiple of the SIMD width.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime assert DV % width == 0, "DV must be SIMD-aligned"

    var alpha = SIMD[DType.float32, width](Float32(exp_f32[1](args.g)))

    for d in range(DK):
        var row = args.state + d * DV
        var m = 0
        while m < DV:
            (row + m).store((row + m).load[width=width]() * alpha)
            m += width

    var vp_arr = InlineArray[Float32, DV](fill=Float32(0))
    var vp = UnsafePointer(to=vp_arr).bitcast[Float32]()

    for d in range(DK):
        var k_d = SIMD[DType.float32, width](args.k[d])
        var row = args.state + d * DV
        var m = 0
        while m < DV:
            (vp + m).store(
                (row + m).load[width=width]().fma(
                    k_d, (vp + m).load[width=width]()))
            m += width

    var v_beta = SIMD[DType.float32, width](args.beta)
    var m0 = 0
    while m0 < DV:
        var vn = ((args.v + m0).load[width=width]() -
                  (vp + m0).load[width=width]()) * v_beta
        (vp + m0).store(vn)
        m0 += width

    for d in range(DK):
        var k_d = SIMD[DType.float32, width](args.k[d])
        var row = args.state + d * DV
        var m = 0
        while m < DV:
            (row + m).store(
                (vp + m).load[width=width]().fma(
                    k_d, (row + m).load[width=width]()))
            m += width

    var oi_arr = InlineArray[Float32, DV](fill=Float32(0))
    var oi = UnsafePointer(to=oi_arr).bitcast[Float32]()

    for d in range(DK):
        var q_d = SIMD[DType.float32, width](args.q[d])
        var row = args.state + d * DV
        var m = 0
        while m < DV:
            (oi + m).store(
                (row + m).load[width=width]().fma(
                    q_d, (oi + m).load[width=width]()))
            m += width

    var m1 = 0
    while m1 < DV:
        (args.out + m1).store((oi + m1).load[width=width]())
        m1 += width

    _ = vp_arr
    _ = oi_arr


# ============================================================================
# Chunk kernel (training / prefill path) — chunked WY-form
# ============================================================================


@fieldwise_init
struct GdnChunkArgs(Copyable, ImplicitlyCopyable):
    var q: F32Ptr            # (CHUNK, DK)   L2-normed in place if l2_norm_qk
    var k: F32Ptr            # (CHUNK, DK)
    var v: F32Ptr            # (CHUNK, DV)
    var beta: F32Ptr         # (CHUNK,)      sigmoid(b)
    var g: F32Ptr            # (CHUNK,)      log decay, < 0
    var state: F32Ptr        # (DK, DV)      updated in-place, row-major
    var out: F32Ptr          # (CHUNK, DV)


def gdn_chunk_kernel[
    CHUNK: Int, DK: Int, DV: Int, l2_norm_qk: Bool = True,
](args: GdnChunkArgs, l2_eps: Float32):
    """One chunk of CHUNK tokens through the gated delta rule (WY-form).

    Implements the chunkwise forward pass of Yang-Kautz-Hatamizadeh 2024.
    Notation: γ_r := exp(decay[r]) is the within-chunk cumulative decay,
    where decay[r] = sum_{i<=r} g[i] (g < 0 so decay is monotone decreasing).

        decay[r]   = sum_{i<=r} g[i]                            cumsum, < 0
        L[r, s]    = exp(decay[r] - decay[s])                   for r >= s, else 0
        A[i, j]    = beta[i] (k[i] · k[j]) L[i, j]              strict lower
        T_inv      = (I + A)^{-1}                               built by Gram-Schmidt
        U[r, m]    = sum_j T_inv[r, j] * beta[j] * v[j, m]      = (T_inv diag(beta)) V
        W[r, d]    = sum_j T_inv[r, j] * beta[j] * γ_j * k[j, d]
        v_prime    = W S                                        chunk-start state read
        v_new      = U - v_prime                                SGD residual within chunk
        attn[r, s] = (q[r] · k[s]) L[r, s]                      intra-chunk attention
        out[r]     = γ_r * S^T q[r] + attn[r] @ v_new
        S = γ_C S + sum_r (γ_C / γ_r) k[r] v_new[r]^T           state carry to next chunk

    Q and K are L2-normalized in place along the head dim before the recurrence
    when l2_norm_qk is true (the FLA `use_qk_l2norm_in_kernel` contract).

    State is row-major (DK, DV). Both DK and DV must be SIMD-aligned.
    """
    comptime width = simd_width_of[DType.float32]()
    comptime assert DK % width == 0, "DK must be SIMD-aligned"
    comptime assert DV % width == 0, "DV must be SIMD-aligned"

    # Step 0: L2-normalize Q and K rows.
    comptime if l2_norm_qk:
        for r in range(CHUNK):
            l2_norm_inplace_f32[DK](args.q + r * DK, l2_eps)
            l2_norm_inplace_f32[DK](args.k + r * DK, l2_eps)

    # Step 1: within-chunk cumulative log-decay and its exp.
    var decay_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var dexp_arr = InlineArray[Float32, CHUNK](uninitialized=True)
    var acc = Float32(0)
    for r in range(CHUNK):
        acc += args.g[r]
        decay_arr[r] = acc
        dexp_arr[r] = Float32(exp_f32[1](acc))
    var decay_C = decay_arr[CHUNK - 1]
    var dexp_C = dexp_arr[CHUNK - 1]

    # Step 2: build A in `mat` as the strict lower triangle of the system.
    # A[i, j] = beta[i] * (k[i] · k[j]) * exp(decay[i] - decay[j]) for i > j.
    # The Gram-Schmidt step inverts (I + A); we initialize with -A so that the
    # standard inversion loop produces (I + A)^{-1} directly.
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

    # Step 3: forward substitution. After this, mat[i, j] for i > j equals
    # the (i, j) entry of (I + A)^{-1}. Pure scalar; CHUNK is small (= 64
    # in the canonical setup) so the O(CHUNK^3) cost is tiny.
    for i in range(1, CHUNK):
        var row_i = mat + i * CHUNK
        for j in range(i):
            var s = Float32(0)
            for m in range(i):
                s += row_i[m] * mat[m * CHUNK + j]
            row_i[j] += s

    # Patch the diagonal so mat fully represents (I + A)^{-1}.
    for i in range(CHUNK):
        mat[i * CHUNK + i] = Float32(1)

    # Step 4: U = (T_inv diag(beta)) V. Fold beta[j] into the per-j coefficient.
    var u_arr = InlineArray[Float32, CHUNK * DV](fill=Float32(0))
    var u = UnsafePointer(to=u_arr).bitcast[Float32]()
    for i in range(CHUNK):
        var row_i = mat + i * CHUNK
        var u_i = u + i * DV
        for j in range(i + 1):
            var coeff = SIMD[DType.float32, width](row_i[j] * args.beta[j])
            var v_j = args.v + j * DV
            var m = 0
            while m < DV:
                (u_i + m).store(
                    (v_j + m).load[width=width]().fma(
                        coeff, (u_i + m).load[width=width]()))
                m += width

    # Step 5: W = T_inv diag(beta) diag(gamma) K. Fold beta[j]*gamma[j] into the
    # per-j coefficient.
    var w_arr = InlineArray[Float32, CHUNK * DK](fill=Float32(0))
    var w = UnsafePointer(to=w_arr).bitcast[Float32]()
    for i in range(CHUNK):
        var row_i = mat + i * CHUNK
        var w_i = w + i * DK
        for j in range(i + 1):
            var coeff = SIMD[DType.float32, width](
                row_i[j] * args.beta[j] * dexp_arr[j])
            var k_j = args.k + j * DK
            var d = 0
            while d < DK:
                (w_i + d).store(
                    (k_j + d).load[width=width]().fma(
                        coeff, (w_i + d).load[width=width]()))
                d += width

    # Step 6: v_prime = W S. Stored in vp; we then in-place subtract from u.
    var vp_arr = InlineArray[Float32, CHUNK * DV](fill=Float32(0))
    var vp = UnsafePointer(to=vp_arr).bitcast[Float32]()
    for i in range(CHUNK):
        var w_i = w + i * DK
        var vp_i = vp + i * DV
        for d in range(DK):
            var w_id = SIMD[DType.float32, width](w_i[d])
            var s_row = args.state + d * DV
            var m = 0
            while m < DV:
                (vp_i + m).store(
                    (s_row + m).load[width=width]().fma(
                        w_id, (vp_i + m).load[width=width]()))
                m += width

    # Step 7: v_new = U - v_prime, written back into u.
    var idx = 0
    while idx < CHUNK * DV:
        var diff = (u + idx).load[width=width]() - (vp + idx).load[width=width]()
        (u + idx).store(diff)
        idx += width

    # Step 8: rebuild attn_local in `mat` (it no longer holds T_inv).
    # attn[r, s] = (q[r] · k[s]) * exp(decay[r] - decay[s]) for r >= s, else 0.
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

    # Step 9: out[r] = gamma_r * S^T q[r] + attn[r] @ v_new.
    for i in range(CHUNK):
        var out_i = args.out + i * DV
        var mi = 0
        while mi < DV:
            (out_i + mi).store(SIMD[DType.float32, width](0))
            mi += width

        # State read scaled by chunk-relative decay.
        var qi = args.q + i * DK
        var dexp_i_scalar = dexp_arr[i]
        for d in range(DK):
            var qd = SIMD[DType.float32, width](qi[d] * dexp_i_scalar)
            var s_row = args.state + d * DV
            var m = 0
            while m < DV:
                (out_i + m).store(
                    (s_row + m).load[width=width]().fma(
                        qd, (out_i + m).load[width=width]()))
                m += width

        # Intra-chunk attention against the chunk-local v_new.
        var al_i = mat + i * CHUNK
        for s in range(i + 1):
            var coeff = SIMD[DType.float32, width](al_i[s])
            var vn_s = u + s * DV
            var m2 = 0
            while m2 < DV:
                (out_i + m2).store(
                    (vn_s + m2).load[width=width]().fma(
                        coeff, (out_i + m2).load[width=width]()))
                m2 += width

    # Step 10: state *= gamma_C.
    var v_dC = SIMD[DType.float32, width](dexp_C)
    for d in range(DK):
        var s_row = args.state + d * DV
        var m = 0
        while m < DV:
            (s_row + m).store((s_row + m).load[width=width]() * v_dC)
            m += width

    # Step 11: state += sum_r (gamma_C / gamma_r) k[r] v_new[r]^T. Computed as
    # exp(decay_C - decay[r]) instead of the ratio to stay bounded for any
    # decay magnitude (the difference is in (-inf, 0] for r in [0, CHUNK)).
    for r in range(CHUNK):
        var coeff_r = Float32(exp_f32[1](decay_C - decay_arr[r]))
        var k_r = args.k + r * DK
        var vn_r = u + r * DV
        for d in range(DK):
            var k_rd = SIMD[DType.float32, width](k_r[d] * coeff_r)
            var s_row = args.state + d * DV
            var m = 0
            while m < DV:
                (s_row + m).store(
                    (vn_r + m).load[width=width]().fma(
                        k_rd, (s_row + m).load[width=width]()))
                m += width

    _ = decay_arr
    _ = dexp_arr
    _ = mat_arr
    _ = u_arr
    _ = w_arr
    _ = vp_arr
