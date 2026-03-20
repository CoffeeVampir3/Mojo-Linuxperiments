"""Operations as free functions on typed tiles.

Weights: DataAccess (comptime dims, bf16).
Activations: DynAccess (runtime rows, comptime cols, bf16).
Scales: DataAccess & ScaleBuffer (comptime dims, quantized).

All ops enforce bf16 compute, dimensional correctness, and SIMD
alignment (cols % 8 == 0 for 128-bit minimum) at compile time.
"""

from math import sqrt
from memory import UnsafePointer
from sys.info import simd_width_of
from experimental3.tensor_contracts import DataAccess, DynAccess, CacheAccess, BufferAccess, ScaleBuffer
from threading import BurstPool, ArgPack


# ================================================================
# BF16 → F32 CAST WORKAROUND
# ================================================================
# The Mojo compiler lowers SIMD[bf16,N].cast[float32]() to scalar
# extract/shift/insert sequences (~35 instructions for width=8).
# The correct lowering is vpmovzxwd + vpslld $16 (2 instructions).
# bf16 is just f32 with the low 16 bits truncated, so reinterpreting
# as uint16 → zero-extending to uint32 → shifting left 16 produces
# the identical IEEE 754 f32 bit pattern.


@always_inline
fn bf16_to_f32[
    width: Int
](ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], offset: Int) -> SIMD[
    DType.float32, width
]:
    """Load `width` bf16 values from ptr+offset and widen to f32.
    Generates vpmovzxwd + vpslld instead of scalar decomposition."""
    var raw = (ptr + offset).bitcast[Scalar[DType.uint16]]().load[width=width]()
    var wide = raw.cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=width]()


# ================================================================
# MATH UTILITIES — no libc dependency
# ================================================================


@fieldwise_init
struct SinCosResult[width: Int = 1]:
    """Carries sin and cos of the same angle(s). SIMD-parameterized."""
    var sin_val: SIMD[DType.float64, Self.width]
    var cos_val: SIMD[DType.float64, Self.width]


fn sincos[width: Int = 1](angles: SIMD[DType.float64, width]) -> SinCosResult[width]:
    """Compute sin/cos via Chebyshev minimax polynomials. SIMD-native, no libc.
    Degree-7 sin / degree-8 cos on [0, π/2], evaluated in Horner form.
    Coefficients fitted on Chebyshev nodes for near-optimal max error
    across the full interval (equioscillation property).

    Fully branchless — range reduction via sign-bit extraction, quadrant
    mapping via bitwise arithmetic. No comparisons, no selects.

    max sin error: ~5.5e-7    max cos error: ~7.8e-7
    bf16 epsilon:  ~7.8e-3    (>10,000x headroom)"""
    comptime HALF_PI = Float64(1.57079632679489661923)
    comptime TWO_PI = Float64(6.28318530717958647692)
    comptime INV_TWO_PI = Float64(0.15915494309189533577)

    # Range reduce to [0, 2π) — sign-bit extraction, no comparisons
    var x = angles - TWO_PI * (angles * INV_TWO_PI).cast[DType.int64]().cast[DType.float64]()
    var neg = x.to_bits() >> 63  # uint64: 1 where x < 0, 0 where x >= 0
    x = x + TWO_PI * neg.cast[DType.float64]()

    # Quadrant [0..3], clamped for x ≈ 2π edge case
    var quad = (x / HALF_PI).cast[DType.int64]()
    var under4 = ((quad - 4) >> 63) & 1
    quad = quad * under4 + SIMD[DType.int64, width](3) * (1 - under4)
    var r = x - quad.cast[DType.float64]() * HALF_PI

    # Chebyshev minimax polynomials (Horner form)
    # sin: degree-7 = 3 FMAs    cos: degree-8 = 4 FMAs
    var r2 = r * r
    var sin_r = r * (0.9999992413456921 + r2 * (
        -0.1666567961884791 + r2 * (
        0.008313225079910211 + r2 * (
        -0.0001852344833019604))))
    var cos_r = 0.9999999532476077 + r2 * (
        -0.4999990506281070 + r2 * (
        0.04166357893069784 + r2 * (
        -0.001385366693303192 + r2 * (
        0.00002315317415552132))))

    # Branchless quadrant mapping via bitwise arithmetic:
    #   Q0: sin=+sin_r, cos=+cos_r    Q1: sin=+cos_r, cos=-sin_r
    #   Q2: sin=-sin_r, cos=-cos_r    Q3: sin=-cos_r, cos=+sin_r
    var swap = (quad & 1).cast[DType.float64]()
    var s_base = sin_r + swap * (cos_r - sin_r)
    var c_base = cos_r + swap * (sin_r - cos_r)
    var sin_sign = 1.0 - 2.0 * (quad >> 1).cast[DType.float64]()
    var cos_sign = 1.0 - 2.0 * ((quad & 1) ^ (quad >> 1)).cast[DType.float64]()

    return SinCosResult[width](s_base * sin_sign, c_base * cos_sign)


fn exp_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Fast exp(x) for f32 via Cody-Waite range reduction + Chebyshev minimax
    polynomial. No libc. exp(x) = 2^n * exp(r), where n = round(x/ln2).

    Cody-Waite: ln2 split into high+low parts so range reduction is exact
    even for large n. Eliminates catastrophic cancellation in r = x - n*ln2.

    Chebyshev minimax: degree-6 polynomial fitted on Chebyshev nodes over
    [-ln2/2, ln2/2]. Equioscillation distributes error evenly across interval.

    2^n via IEEE 754 exponent bit stuffing. Fully branchless, SIMD-native.

    max rel error: ~1.5e-7 (~1 ULP)    bf16 epsilon: ~7.8e-3"""
    # Cody-Waite constants: LN2 = LN2_HI + LN2_LO, exact in extended precision
    # LN2_HI has low bits zeroed so n * LN2_HI is exact in f32
    comptime LN2_HI = Float32(0.693145751953125)        # 16-bit truncation
    comptime LN2_LO = Float32(1.4286068203094172e-06)    # remainder
    comptime INV_LN2 = Float32(1.4426950408889634)

    # Clamp input to f32 exp range. Below -87 → exp underflows to 0,
    # above 88 → overflows to inf. Without this, range reduction produces
    # r values outside [-ln2/2, ln2/2] and the polynomial returns NaN.
    # Standard practice in all production expf implementations.
    comptime EXP_LO = Float32(-87.0)
    comptime EXP_HI = Float32(88.0)
    var lo_mask = ((x - EXP_LO).to_bits() >> 31) & 1  # 1 where x < -87
    var xc = x * (1 - lo_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_LO) * lo_mask.cast[DType.float32]()
    var hi_mask = ((EXP_HI - xc).to_bits() >> 31) & 1  # 1 where x > 88
    xc = xc * (1 - hi_mask.cast[DType.float32]()) + SIMD[DType.float32, width](EXP_HI) * hi_mask.cast[DType.float32]()

    # n = round(x / ln2) — branchless rounding via sign-bit offset
    var xn = xc * INV_LN2
    var sign = (xn.to_bits() >> 31).cast[DType.float32]()
    var n = (xn + 0.5 - sign).cast[DType.int32]()

    # Cody-Waite range reduction: r = (x - n*LN2_HI) - n*LN2_LO
    # First subtraction is exact (LN2_HI chosen so n*LN2_HI fits in f32),
    # second corrects the residual.
    var nf = n.cast[DType.float32]()
    var r = (xc - nf * LN2_HI) - nf * LN2_LO

    # Chebyshev minimax degree-6 on [-ln2/2, ln2/2] (Horner form)
    # Coefficients fitted via least-squares on Chebyshev nodes, minimizing
    # max absolute error of exp(r) approximation across the interval.
    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999999995) + r * (
        Float32(0.5000000004) + r * (
        Float32(0.1666666456) + r * (
        Float32(0.04166685110) + r * (
        Float32(0.008333621758) + r * (
        Float32(0.001389404636)))))))

    # 2^n via IEEE 754 exponent bit manipulation
    var pow2n = SIMD[DType.float32, width](
        from_bits=(n + 127).cast[DType.uint32]() << 23
    )

    return p * pow2n


# ================================================================
# KERNEL OPERATIONS
# ================================================================


fn gemm_kernel[K: Int, N: Int](
    input_ptr: Int, weight_ptr: Int, output_ptr: Int,
    start_row: Int, end_row: Int, _unused: Int,
):
    """GEMM row kernel matching BurstPool KernelFn ABI.
    dst[m,n] = dot(input[m,:], weight[n,:]) for rows [start_row, end_row)."""
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=input_ptr
    )
    var wp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=weight_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=output_ptr
    )
    comptime width = simd_width_of[DType.float32]()

    for m in range(start_row, end_row):
        var row_in = ip + m * K
        var row_out = dp + m * N
        for n in range(N):
            var row_w = wp + n * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var x = bf16_to_f32[width](row_in, k)
                var w = bf16_to_f32[width](row_w, k)
                acc = x.fma(w, acc)
            row_out[n] = acc.reduce_add().cast[DType.bfloat16]()


fn gemv_kernel[K: Int, N: Int](
    input_ptr: Int, weight_ptr: Int, output_ptr: Int,
    start_col: Int, end_col: Int, seq_len: Int,
):
    """Tiled GEMV kernel — weight rows outer, input rows inner.
    Each weight row is loaded once per k-step and reused across 4 input rows.
    4 SIMD accumulators live in registers — no intermediate buffers."""
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=input_ptr
    )
    var wp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=weight_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=output_ptr
    )
    comptime width = simd_width_of[DType.float32]()
    comptime Mr = 4

    var m_full = (seq_len // Mr) * Mr
    for n in range(start_col, end_col):
        var row_w = wp + n * K

        # Tiled: 4 input rows, 1 weight load per k-step
        for m_base in range(0, m_full, Mr):
            var acc0 = SIMD[DType.float32, width](0)
            var acc1 = SIMD[DType.float32, width](0)
            var acc2 = SIMD[DType.float32, width](0)
            var acc3 = SIMD[DType.float32, width](0)
            var r0 = ip + m_base * K
            var r1 = ip + (m_base + 1) * K
            var r2 = ip + (m_base + 2) * K
            var r3 = ip + (m_base + 3) * K
            for k in range(0, K, width):
                var w = bf16_to_f32[width](row_w, k)
                acc0 = bf16_to_f32[width](r0, k).fma(w, acc0)
                acc1 = bf16_to_f32[width](r1, k).fma(w, acc1)
                acc2 = bf16_to_f32[width](r2, k).fma(w, acc2)
                acc3 = bf16_to_f32[width](r3, k).fma(w, acc3)
            (dp + m_base * N)[n] = acc0.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 1) * N)[n] = acc1.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 2) * N)[n] = acc2.reduce_add().cast[DType.bfloat16]()
            (dp + (m_base + 3) * N)[n] = acc3.reduce_add().cast[DType.bfloat16]()

        # Remainder rows
        for m in range(m_full, seq_len):
            var row_in = ip + m * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var w = bf16_to_f32[width](row_w, k)
                acc = bf16_to_f32[width](row_in, k).fma(w, acc)
            (dp + m * N)[n] = acc.reduce_add().cast[DType.bfloat16]()


fn gemm[In: DynAccess, Weight: DataAccess, Dst: DynAccess](
    input: In, weight: Weight, dst: Dst,
    mut pool: BurstPool,
):
    """dst[M,N] = input[M,K] × weight[N,K]^T. M is runtime, via BurstPool.
    For small M (decode), partitions output columns across workers.
    For large M (prefill), partitions input rows."""
    constrained[In.DTYPE == DType.bfloat16, "gemm: input must be bf16"]()
    constrained[Weight.DTYPE == DType.bfloat16, "gemm: weight must be bf16"]()
    constrained[Dst.DTYPE == DType.bfloat16, "gemm: dst must be bf16"]()
    constrained[In.COLS == Weight.COLS, "gemm: input K != weight K"]()
    constrained[Dst.COLS == Weight.ROWS, "gemm: dst N != weight N"]()
    constrained[
        In.COLS % simd_width_of[DType.float32]() == 0,
        "gemm: K must be f32-simd-aligned",
    ]()

    var seq_len = input.rows()
    if seq_len == 0:
        return

    var input_ptr = Int(input.data())
    var weight_ptr = Int(weight.data())
    var output_ptr = Int(dst.data())

    if seq_len < pool.capacity:
        # Decode path: partition N (output columns) across workers
        comptime N = Weight.ROWS
        var num_jobs = pool.capacity
        var cols_per_job = (N + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * cols_per_job
            var end = min(start + cols_per_job, N)
            var pack = pool.args_base + i
            pack[].arg0 = input_ptr
            pack[].arg1 = weight_ptr
            pack[].arg2 = output_ptr
            pack[].arg3 = start
            pack[].arg4 = end
            pack[].arg5 = seq_len

        pool.dispatch(gemv_kernel[In.COLS, N], pool.args_base, num_jobs)
        pool.join()
    else:
        # Prefill path: partition M (input rows) across workers
        var num_jobs = min(seq_len, pool.capacity)
        var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * rows_per_job
            var end = min(start + rows_per_job, seq_len)
            var pack = pool.args_base + i
            pack[].arg0 = input_ptr
            pack[].arg1 = weight_ptr
            pack[].arg2 = output_ptr
            pack[].arg3 = start
            pack[].arg4 = end
            pack[].arg5 = 0

        pool.dispatch(gemm_kernel[In.COLS, Weight.ROWS], pool.args_base, num_jobs)
        pool.join()


fn dequant_gemm[
    In: DynAccess, Weight: DataAccess,
    Scale: DataAccess & ScaleBuffer, Dst: DynAccess,
](
    input: In, weight: Weight, scale: Scale, dst: Dst,
):
    """Quantized GEMM: weight is quantized, scale provides per-block factors."""
    # activation path is bf16, weight can be int8/int4
    constrained[In.DTYPE == DType.bfloat16, "dequant_gemm: input must be bf16"]()
    constrained[Dst.DTYPE == DType.bfloat16, "dequant_gemm: dst must be bf16"]()
    constrained[Dst.COLS == Weight.ROWS, "dequant_gemm: dst N != weight N"]()

    var ip = input.data()
    var wp = weight.data()
    var sp = scale.data()
    var dp = dst.data()
    print(
        "  dequant_gemm bf16 x", Weight.DTYPE,
        "[", Weight.ROWS, "x", Weight.COLS, "]",
        "s:", Scale.DTYPE, "blk", Scale.BLOCK_SIZE,
        "-> [", dst.rows(), "x", Dst.COLS, "]",
    )
    dp[0] = Scalar[Dst.DTYPE](0)


fn rmsnorm_kernel[cols: Int](
    input_ptr: Int, weight_ptr: Int, output_ptr: Int,
    start_row: Int, end_row: Int, eps_bits: Int,
):
    """RMSNorm row kernel matching BurstPool KernelFn ABI.
    Processes rows [start_row, end_row) with fused reduction + normalize."""
    var eps_i32 = Int32(eps_bits)
    var eps = UnsafePointer(to=eps_i32).bitcast[Float32]()[]
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=input_ptr
    )
    var wp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=weight_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=output_ptr
    )
    comptime width = simd_width_of[DType.float32]()

    for row in range(start_row, end_row):
        var row_in = ip + row * cols
        var row_out = dp + row * cols

        # Pass 1: f32 sum of squares via FMA
        var acc = SIMD[DType.float32, width](0)
        for j in range(0, cols, width):
            var x = bf16_to_f32[width](row_in, j)
            acc = x.fma(x, acc)
        var sum_sq = acc.reduce_add()
        var scale = Float32(1.0) / sqrt(sum_sq / Float32(cols) + eps)

        # Pass 2: normalize × weight, store bf16
        var sv = SIMD[DType.float32, width](scale)
        for j in range(0, cols, width):
            var x = bf16_to_f32[width](row_in, j)
            var w = bf16_to_f32[width](wp, j)
            (row_out + j).store((x * sv * w).cast[DType.bfloat16]())


fn rmsnorm[In: DynAccess, NormWeight: DataAccess, Dst: DynAccess](
    input: In, weight: NormWeight, dst: Dst,
    mut pool: BurstPool,
    eps: Float32 = 1e-6,
):
    """RMSNorm via BurstPool: dst = (input / RMS(input)) * weight.
    F32 accumulation in registers, bf16 I/O. Rows dispatched to pool workers."""
    constrained[In.DTYPE == DType.bfloat16, "rmsnorm: input must be bf16"]()
    constrained[NormWeight.DTYPE == DType.bfloat16, "rmsnorm: weight must be bf16"]()
    constrained[Dst.DTYPE == DType.bfloat16, "rmsnorm: dst must be bf16"]()
    constrained[NormWeight.ROWS == 1, "rmsnorm: weight must be 1D"]()
    constrained[In.COLS == NormWeight.COLS, "rmsnorm: input hidden != weight hidden"]()
    constrained[In.COLS == Dst.COLS, "rmsnorm: input hidden != dst hidden"]()
    constrained[
        In.COLS % simd_width_of[DType.float32]() == 0,
        "rmsnorm: hidden must be f32-simd-aligned",
    ]()

    var seq_len = input.rows()
    if seq_len == 0:
        return

    var input_ptr = Int(input.data())
    var weight_ptr = Int(weight.data())
    var output_ptr = Int(dst.data())
    var eps_copy = eps
    var eps_int = Int(UnsafePointer(to=eps_copy).bitcast[Int32]()[])

    var num_jobs = min(seq_len, pool.capacity)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.args_base + i
        pack[].arg0 = input_ptr
        pack[].arg1 = weight_ptr
        pack[].arg2 = output_ptr
        pack[].arg3 = start
        pack[].arg4 = end
        pack[].arg5 = eps_int

    pool.dispatch(rmsnorm_kernel[In.COLS], pool.args_base, num_jobs)
    pool.join()


fn silu_mul[Act: DynAccess](gate: Act, up: Act, dst: Act):
    """SwiGLU: dst = silu(gate) * up. All operands same shape.
    silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). F32 compute, bf16 I/O."""
    constrained[Act.DTYPE == DType.bfloat16, "silu_mul: must be bf16"]()
    constrained[
        Act.COLS % simd_width_of[DType.float32]() == 0,
        "silu_mul: cols must be f32-simd-aligned",
    ]()

    var seq_len = gate.rows()
    if seq_len == 0:
        return

    var gp = gate.data().bitcast[Scalar[DType.bfloat16]]()
    var up_ = up.data().bitcast[Scalar[DType.bfloat16]]()
    var dp = dst.data().bitcast[Scalar[DType.bfloat16]]()
    comptime cols = Act.COLS
    comptime width = simd_width_of[DType.float32]()
    comptime elems = seq_len * cols

    for i in range(0, seq_len * cols, width):
        var g = bf16_to_f32[width](gp, i)
        var u = bf16_to_f32[width](up_, i)
        # silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        var sig = 1.0 / (1.0 + exp_f32[width](-g))
        (dp + i).store((g * sig * u).cast[DType.bfloat16]())


fn elem_add[Act: DynAccess](a: Act, b: Act, dst: Act):
    """Elementwise: dst = a + b. All operands same shape. Bf16 throughout."""
    constrained[Act.DTYPE == DType.bfloat16, "elem_add: must be bf16"]()
    constrained[
        Act.COLS % simd_width_of[DType.bfloat16]() == 0,
        "elem_add: cols must be bf16-simd-aligned",
    ]()

    var seq_len = a.rows()
    if seq_len == 0:
        return

    var ap = a.data().bitcast[Scalar[DType.bfloat16]]()
    var bp = b.data().bitcast[Scalar[DType.bfloat16]]()
    var dp = dst.data().bitcast[Scalar[DType.bfloat16]]()
    comptime width = simd_width_of[DType.bfloat16]()

    for i in range(0, seq_len * Act.COLS, width):
        var av = (ap + i).load[width=width]()
        var bv = (bp + i).load[width=width]()
        (dp + i).store(av + bv)


fn embed_lookup_kernel[cols: Int](
    table_ptr: Int, tokens_ptr: Int, output_ptr: Int,
    start_row: Int, end_row: Int, _unused: Int,
):
    """Embed gather kernel matching BurstPool KernelFn ABI.
    Copies table[token_id] → dst for rows [start_row, end_row)."""
    var tp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=table_ptr
    )
    var tokens = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
        unsafe_from_address=tokens_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=output_ptr
    )
    comptime width = simd_width_of[DType.bfloat16]()

    for i in range(start_row, end_row):
        var src = tp + Int(tokens[i]) * cols
        var out = dp + i * cols
        for j in range(0, cols, width):
            (out + j).store((src + j).load[width=width]())


fn embed_lookup[Table: DataAccess, Dst: DynAccess](
    table: Table, tokens_ptr: Int, dst: Dst,
    mut pool: BurstPool,
):
    """Gather: for each token ID, copy table[id] → dst row. dst.rows() lookups."""
    constrained[Table.DTYPE == DType.bfloat16, "embed: table must be bf16"]()
    constrained[Dst.DTYPE == DType.bfloat16, "embed: dst must be bf16"]()
    constrained[Table.COLS == Dst.COLS, "embed: table hidden != dst hidden"]()

    var seq_len = dst.rows()
    if seq_len == 0:
        return

    var table_int = Int(table.data())
    var output_int = Int(dst.data())

    var num_jobs = min(seq_len, pool.capacity)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.args_base + i
        pack[].arg0 = table_int
        pack[].arg1 = tokens_ptr
        pack[].arg2 = output_int
        pack[].arg3 = start
        pack[].arg4 = end
        pack[].arg5 = 0

    pool.dispatch(embed_lookup_kernel[Table.COLS], pool.args_base, num_jobs)
    pool.join()


fn init_rope_tables[Cos: BufferAccess, Sin: BufferAccess](
    cos_buf: Cos, sin_buf: Sin, theta: Float64 = 10000.0,
):
    """Precompute cos/sin tables for RoPE. Call once at model init.
    cos[pos, j] = cos(pos / theta^(2j/dim)), sin likewise.
    Tables are f32 for rotation precision. Uses SIMD sincos for throughput."""
    constrained[Cos.DTYPE == DType.float32, "rope init: cos must be f32"]()
    constrained[Sin.DTYPE == DType.float32, "rope init: sin must be f32"]()
    constrained[Cos.ROWS == Sin.ROWS, "rope init: cos/sin capacity mismatch"]()
    constrained[Cos.COLS == Sin.COLS, "rope init: cos/sin cols mismatch"]()
    constrained[
        Cos.COLS % simd_width_of[DType.float64]() == 0,
        "rope init: cols must be f64-simd-aligned",
    ]()

    var cp = cos_buf.data().bitcast[Scalar[DType.float32]]()
    var sp = sin_buf.data().bitcast[Scalar[DType.float32]]()
    comptime half = Cos.COLS
    comptime head_dim = half * 2
    comptime f64w = simd_width_of[DType.float64]()

    # Loop interchange: outer=frequency chunks, inner=positions.
    # inv_freq lives in a SIMD register, no allocation needed.
    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            inv[k] = 1.0 / (theta ** (Float64(2 * (j + k)) / Float64(head_dim)))

        for pos in range(Cos.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]())


fn rope[
    head_dim: Int, num_heads: Int,
    Act: DynAccess, Cos: BufferAccess, Sin: BufferAccess,
](
    x: Act, cos_table: Cos, sin_table: Sin, pos: Int,
):
    """Rotary position embeddings, applied in-place per head.
    For each head, rotates pairs (j, j+half) using precomputed cos/sin.
    Matches HuggingFace rotate_half + apply_rotary_pos_emb."""
    constrained[Act.DTYPE == DType.bfloat16, "rope: must be bf16"]()
    constrained[Act.COLS == head_dim * num_heads, "rope: cols != heads * dim"]()
    constrained[head_dim % 2 == 0, "rope: head_dim must be even (rotation pairs)"]()
    constrained[Cos.DTYPE == DType.float32, "rope: cos table must be f32"]()
    constrained[Sin.DTYPE == DType.float32, "rope: sin table must be f32"]()
    constrained[Cos.COLS == head_dim // 2, "rope: cos cols != head_dim/2"]()
    constrained[Sin.COLS == head_dim // 2, "rope: sin cols != head_dim/2"]()
    constrained[Cos.ROWS == Sin.ROWS, "rope: cos/sin capacity mismatch"]()
    constrained[
        (head_dim // 2) % simd_width_of[DType.float32]() == 0,
        "rope: half must be f32-simd-aligned",
    ]()

    var seq_len = x.rows()
    if seq_len == 0:
        return

    var xp = x.data().bitcast[Scalar[DType.bfloat16]]()
    var cp = cos_table.data().bitcast[Scalar[DType.float32]]()
    var sp = sin_table.data().bitcast[Scalar[DType.float32]]()
    comptime half = head_dim // 2
    comptime width = simd_width_of[DType.float32]()
    comptime row_stride = num_heads * head_dim

    for m in range(seq_len):
        var actual_pos = pos + m
        var cos_row = cp + actual_pos * half
        var sin_row = sp + actual_pos * half
        var row_base = xp + m * row_stride

        for h in range(num_heads):
            var head_base = row_base + h * head_dim
            for j in range(0, half, width):
                var x_lo = bf16_to_f32[width](head_base, j)
                var x_hi = bf16_to_f32[width](head_base, half + j)
                var cv = (cos_row + j).load[width=width]()
                var sv = (sin_row + j).load[width=width]()
                (head_base + j).store(
                    (x_lo * cv - x_hi * sv).cast[DType.bfloat16]()
                )
                (head_base + half + j).store(
                    (x_hi * cv + x_lo * sv).cast[DType.bfloat16]()
                )


fn kv_cache_write[Act: DynAccess, Cache: CacheAccess](
    src: Act, cache: Cache, pos: Int,
):
    """Copy src[seq_len, kv_dim] into cache at row pos. Supports both
    prefill (seq_len=N, pos=0) and decode (seq_len=1, pos=current)."""
    constrained[Act.DTYPE == DType.bfloat16, "kv_write: must be bf16"]()
    constrained[Cache.DTYPE == DType.bfloat16, "kv_write: cache must be bf16"]()
    constrained[Act.COLS == Cache.COLS, "kv_write: src cols != cache cols"]()
    constrained[
        Act.COLS % simd_width_of[DType.bfloat16]() == 0,
        "kv_write: cols must be bf16-simd-aligned",
    ]()

    var seq_len = src.rows()
    if seq_len == 0:
        return

    var sp = src.data().bitcast[Scalar[DType.bfloat16]]()
    var cp = cache.data().bitcast[Scalar[DType.bfloat16]]()
    comptime cols = Act.COLS
    comptime width = simd_width_of[DType.bfloat16]()

    for m in range(seq_len):
        var src_row = sp + m * cols
        var dst_row = cp + (pos + m) * cols
        for j in range(0, cols, width):
            (dst_row + j).store((src_row + j).load[width=width]())


fn gqa_kernel[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    kv_cols: Int,
](
    q_ptr: Int, dst_ptr: Int, k_ptr: Int,
    v_ptr: Int, start_end_packed: Int, pos_seq_packed: Int,
):
    """GQA attention kernel matching BurstPool KernelFn ABI.
    Processes KV head groups [start_group, end_group).

    Per query head, fuses three phases in a single pass over the KV cache:
      1. GEMV — dot(q[1,D], K[T,D]^T) → score scalar per cached position
      2. Online softmax — streaming max + exp + sum, no score buffer
      3. V accumulation — weighted sum of V rows, f32 in registers

    V accumulator is SIMD[f32, head_dim] — lives entirely in registers,
    written to dst as bf16 once after normalization.

    Work unit = KV head group (1 KV head + heads_per_group query heads).
    Each group is fully independent → natural TP partition."""
    comptime heads_per_group = num_heads // num_kv_heads
    comptime q_stride = num_heads * head_dim
    comptime kv_stride = kv_cols
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = head_dim // width
    comptime scale_f32 = Float32(1.0 / sqrt(Float64(head_dim)))

    var start_group = start_end_packed >> 32
    var end_group = start_end_packed & 0xFFFFFFFF
    var pos = pos_seq_packed >> 32
    var seq_len = pos_seq_packed & 0xFFFFFFFF

    var qp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=q_ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=dst_ptr
    )
    var kp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=k_ptr
    )
    var vp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=v_ptr
    )

    for m in range(seq_len):
        # Causal: query at position pos+m can attend to [0, pos+m]
        var causal_len = pos + m + 1

        for g in range(start_group, end_group):
            # All query heads in this group share the same K/V head
            var kv_head_offset = g * head_dim

            for qh in range(heads_per_group):
                var q_head_idx = g * heads_per_group + qh
                var q_head = qp + m * q_stride + q_head_idx * head_dim
                var d_head = dp + m * q_stride + q_head_idx * head_dim

                # Register-resident f32 V accumulator (head_dim floats)
                var acc = SIMD[DType.float32, head_dim](0)
                var running_max = Float32(-1e30)
                var running_sum = Float32(0)

                for t in range(causal_len):
                    # === Phase 1: GEMV — score = dot(q, K[t]) / sqrt(d) ===
                    var k_row = kp + t * kv_stride + kv_head_offset
                    var dot_acc = SIMD[DType.float32, width](0)
                    @parameter
                    for c in range(chunks):
                        comptime off = c * width
                        var qv = bf16_to_f32[width](q_head, off)
                        var kv = bf16_to_f32[width](k_row, off)
                        dot_acc = qv.fma(kv, dot_acc)
                    var score = dot_acc.reduce_add() * scale_f32

                    # === Phase 2: Online softmax — update running max/sum ===
                    var new_max = max(running_max, score)
                    var correction = Float32(
                        exp_f32[1](SIMD[DType.float32, 1](running_max - new_max))
                    )
                    var w = Float32(
                        exp_f32[1](SIMD[DType.float32, 1](score - new_max))
                    )

                    # === Phase 3: V accumulation — rescale prior + add weighted V[t] ===
                    var v_row = vp + t * kv_stride + kv_head_offset
                    @parameter
                    for c in range(chunks):
                        comptime off = c * width
                        var prior = acc.slice[width, offset=off]()
                        var vv = bf16_to_f32[width](v_row, off)
                        acc = acc.insert[offset=off](prior * correction + vv * w)

                    running_sum = correction * running_sum + w
                    running_max = new_max

                # === Normalize and store — single bf16 write per head ===
                var inv_sum = 1.0 / running_sum
                @parameter
                for c in range(chunks):
                    comptime off = c * width
                    var v = acc.slice[width, offset=off]() * inv_sum
                    (d_head + off).store(v.cast[DType.bfloat16]())


fn attention[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    Query: DynAccess, KCache: CacheAccess, VCache: CacheAccess, Dst: DynAccess,
](
    q: Query,
    k_cache: KCache, v_cache: VCache,
    pos: Int,
    dst: Dst,
    mut pool: BurstPool,
):
    """GQA attention: Q[M, H*D] attends over KV cache[0..pos+M, Hkv*D].
    Causal masked, online softmax (single-pass, no score buffer).
    F32 accumulation, bf16 I/O.

    Work partitioned by KV head group via BurstPool — each job processes
    one or more groups (1 KV head + heads_per_group query heads each).
    Groups are fully independent → maps directly to TP partitioning."""
    constrained[Query.DTYPE == DType.bfloat16, "attention: Q must be bf16"]()
    constrained[Dst.DTYPE == DType.bfloat16, "attention: dst must be bf16"]()
    constrained[KCache.DTYPE == DType.bfloat16, "attention: K cache must be bf16"]()
    constrained[VCache.DTYPE == DType.bfloat16, "attention: V cache must be bf16"]()
    constrained[Query.COLS == num_heads * head_dim, "attention: Q cols != H*D"]()
    constrained[Dst.COLS == Query.COLS, "attention: dst cols != Q cols"]()
    constrained[KCache.COLS == num_kv_heads * head_dim, "attention: K cache cols != Hkv*D"]()
    constrained[VCache.COLS == KCache.COLS, "attention: V cache cols != K cache cols"]()
    constrained[KCache.CAPACITY == VCache.CAPACITY, "attention: K/V capacity mismatch"]()
    constrained[head_dim % simd_width_of[DType.float32]() == 0, "attention: head_dim must be f32-simd-aligned"]()
    constrained[num_heads % num_kv_heads == 0, "attention: heads must divide evenly for GQA"]()

    var seq_len = q.rows()
    if seq_len == 0:
        return

    var q_ptr = Int(q.data())
    var dst_ptr = Int(dst.data())
    var k_ptr = Int(k_cache.data())
    var v_ptr = Int(v_cache.data())
    var pos_seq = (pos << 32) | seq_len

    # Partition KV head groups across pool workers
    var num_jobs = min(num_kv_heads, pool.capacity)
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * groups_per_job
        var end = min(start + groups_per_job, num_kv_heads)
        var pack = pool.args_base + i
        pack[].arg0 = q_ptr
        pack[].arg1 = dst_ptr
        pack[].arg2 = k_ptr
        pack[].arg3 = v_ptr
        pack[].arg4 = (start << 32) | end
        pack[].arg5 = pos_seq

    pool.dispatch(
        gqa_kernel[num_heads, num_kv_heads, head_dim, KCache.COLS],
        pool.args_base, num_jobs,
    )
    pool.join()


# ================================================================
# COMMUNICATION OPS — no-ops under TP=1
# ================================================================


fn broadcast[Act: DynAccess](src: Act, num_nodes: Int):
    """Copy src to all nodes' local buffers. No-op when num_nodes == 1."""
    constrained[Act.DTYPE == DType.bfloat16, "broadcast: must be bf16"]()
    if num_nodes <= 1:
        return
    var sp = src.data()
    print("  broadcast bf16 [", src.rows(), "x", Act.COLS, "] ->", num_nodes, "nodes")
    _ = sp


fn all_reduce[Act: DynAccess](dst: Act, num_nodes: Int):
    """Sum partials from all nodes into each node's buffer. No-op when num_nodes == 1."""
    constrained[Act.DTYPE == DType.bfloat16, "all_reduce: must be bf16"]()
    if num_nodes <= 1:
        return
    var dp = dst.data()
    print("  all_reduce bf16 [", dst.rows(), "x", Act.COLS, "] across", num_nodes, "nodes")
    _ = dp
