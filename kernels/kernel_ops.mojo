"""Operations as free functions on typed views.

Pool-dispatched kernels return PoolFence — a linear type (@explicit_destroy)
representing in-flight work. Must be consumed via .join() or parallel().

TP=1: gemm(...).join()
TP=N: parallel(gemm(..., pool0), gemm(..., pool1), gemm(..., pool2))
"""

from std.math import sqrt
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from threading import BurstPool
import linux.sys as linux

from modeling.model_spec import (
    Encoding, Shaped, Bound, DynView, CacheView,
)
from simd_math import bf16_to_f32, SinCosResult, sincos, exp_f32


# ================================================================
# POOL FENCE — linear synchronization token
# ================================================================


@explicit_destroy
@fieldwise_init
struct PoolFence(Movable):
    """Linear token for in-flight pool work. Unconsumed fences are a compile error.

    Three consumption paths:
        .join()  — wait immediately (TP=1)
        .take()  — extract raw pool ptr for deferred batch join (parametric TP)
        parallel(f0, f1, ...) — variadic barrier (fixed TP)
    """
    var pool: UnsafePointer[BurstPool[], MutAnyOrigin]

    @staticmethod
    def completed() -> Self:
        return Self(UnsafePointer[BurstPool[], MutAnyOrigin]())

    def join(deinit self):
        """Consume fence, wait for work to complete."""
        if self.pool:
            self.pool[].join()

    def take(deinit self) -> UnsafePointer[BurstPool[], MutAnyOrigin]:
        """Consume fence, return raw pool pointer for deferred join."""
        return self.pool


def parallel(var *fences: PoolFence):
    """Variadic barrier: joins all fences, consuming each."""
    @parameter
    def do_join(idx: Int, var fence: PoolFence) capturing:
        fence^.join()
    fences^.consume_elements[do_join]()


def parallel_for[tp: Int, body: def[rank: Int] () capturing -> PoolFence]():
    """Parametric barrier: dispatch body[rank]() for each rank, then join all.
    Works for any TP degree. body returns PoolFence — consumed internally via .take()."""
    var ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], tp](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]()
    )
    comptime for rank in range(tp):
        ptrs[rank] = body[rank]().take()
    for i in range(tp):
        if ptrs[i]:
            ptrs[i][].join()


# ================================================================
# KERNEL FUNCTIONS (BurstPool dispatch targets)
# ================================================================


def gemm_kernel[K: Int, N: Int](
    ip: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    wp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_row: Int, end_row: Int, _unused: Int,
):
    """GEMM row kernel. dst[m,n] = dot(input[m,:], weight[n,:])
    for rows [start_row, end_row)."""
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


def gemv_kernel[K: Int, N: Int](
    ip: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    wp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_col: Int, end_col: Int, seq_len: Int,
):
    """N-tiled GEMV kernel — processes 4 output columns simultaneously,
    reusing each input load across 4 independent FMA chains to hide
    FMA latency and reduce input bandwidth."""
    comptime width = simd_width_of[DType.float32]()
    comptime Nr = 4

    var n_full = end_col - ((end_col - start_col) % Nr)

    for m in range(seq_len):
        var row_in = ip + m * K
        var row_out = dp + m * N

        for n in range(start_col, n_full, Nr):
            var w0 = wp + n * K
            var w1 = wp + (n + 1) * K
            var w2 = wp + (n + 2) * K
            var w3 = wp + (n + 3) * K
            var acc0 = SIMD[DType.float32, width](0)
            var acc1 = SIMD[DType.float32, width](0)
            var acc2 = SIMD[DType.float32, width](0)
            var acc3 = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var x = bf16_to_f32[width](row_in, k)
                acc0 = x.fma(bf16_to_f32[width](w0, k), acc0)
                acc1 = x.fma(bf16_to_f32[width](w1, k), acc1)
                acc2 = x.fma(bf16_to_f32[width](w2, k), acc2)
                acc3 = x.fma(bf16_to_f32[width](w3, k), acc3)
            row_out[n] = acc0.reduce_add().cast[DType.bfloat16]()
            row_out[n + 1] = acc1.reduce_add().cast[DType.bfloat16]()
            row_out[n + 2] = acc2.reduce_add().cast[DType.bfloat16]()
            row_out[n + 3] = acc3.reduce_add().cast[DType.bfloat16]()

        for n in range(n_full, end_col):
            var row_w = wp + n * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                acc = bf16_to_f32[width](row_in, k).fma(bf16_to_f32[width](row_w, k), acc)
            row_out[n] = acc.reduce_add().cast[DType.bfloat16]()


def rmsnorm_kernel[cols: Int](
    ip: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    wp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_row: Int, end_row: Int, eps_bits: Int,
):
    """RMSNorm row kernel. Fused reduction + normalize for
    rows [start_row, end_row)."""
    var eps_i32 = Int32(eps_bits)
    var eps = UnsafePointer(to=eps_i32).bitcast[Float32]()[]
    comptime width = simd_width_of[DType.float32]()

    for row in range(start_row, end_row):
        var row_in = ip + row * cols
        var row_out = dp + row * cols

        var acc = SIMD[DType.float32, width](0)
        for j in range(0, cols, width):
            var x = bf16_to_f32[width](row_in, j)
            acc = x.fma(x, acc)
        var sum_sq = acc.reduce_add()
        var scale = Float32(1.0) / sqrt(sum_sq / Float32(cols) + eps)

        var sv = SIMD[DType.float32, width](scale)
        for j in range(0, cols, width):
            var x = bf16_to_f32[width](row_in, j)
            var w = bf16_to_f32[width](wp, j)
            (row_out + j).store((x * sv * w).cast[DType.bfloat16]())


def embed_lookup_kernel[cols: Int](
    tp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    tokens: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_row: Int, end_row: Int, _unused: Int,
):
    """Embed gather kernel. Copies table[token_id] → dst
    for rows [start_row, end_row)."""
    comptime width = simd_width_of[DType.bfloat16]()

    for i in range(start_row, end_row):
        var src = tp + Int(tokens[i]) * cols
        var out = dp + i * cols
        for j in range(0, cols, width):
            (out + j).store((src + j).load[width=width]())


def gqa_kernel[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    kv_cols: Int,
](
    qp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    kp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    vp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    start_end_packed: Int, pos_seq_packed: Int,
):
    """GQA attention kernel. Per query head, fuses:
      1. GEMV — dot(q, K[t]) → score
      2. Online softmax — streaming max + exp + sum
      3. V accumulation — weighted sum in f32 registers"""
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

    for m in range(seq_len):
        var causal_len = pos + m + 1

        for g in range(start_group, end_group):
            var kv_head_offset = g * head_dim

            for qh in range(heads_per_group):
                var q_head_idx = g * heads_per_group + qh
                var q_head = qp + m * q_stride + q_head_idx * head_dim
                var d_head = dp + m * q_stride + q_head_idx * head_dim

                var acc = SIMD[DType.float32, head_dim](0)
                var running_max = Float32(-1e30)
                var running_sum = Float32(0)

                for t in range(causal_len):
                    # Phase 1: GEMV — score = dot(q, K[t]) / sqrt(d)
                    var k_row = kp + t * kv_stride + kv_head_offset
                    var dot_acc = SIMD[DType.float32, width](0)
                    comptime for c in range(chunks):
                        comptime off = c * width
                        var qv = bf16_to_f32[width](q_head, off)
                        var kv = bf16_to_f32[width](k_row, off)
                        dot_acc = qv.fma(kv, dot_acc)
                    var score = dot_acc.reduce_add() * scale_f32

                    # Phase 2: Online softmax — update running max/sum
                    var new_max = max(running_max, score)
                    var correction = Float32(
                        exp_f32[1](SIMD[DType.float32, 1](running_max - new_max))
                    )
                    var w = Float32(
                        exp_f32[1](SIMD[DType.float32, 1](score - new_max))
                    )

                    # Phase 3: V accumulation — rescale prior + add weighted V[t]
                    var v_row = vp + t * kv_stride + kv_head_offset
                    comptime for c in range(chunks):
                        comptime off = c * width
                        var prior = acc.slice[width, offset=off]()
                        var vv = bf16_to_f32[width](v_row, off)
                        acc = acc.insert[offset=off](prior * correction + vv * w)

                    running_sum = correction * running_sum + w
                    running_max = new_max

                # Normalize and store
                var inv_sum = 1.0 / running_sum
                comptime for c in range(chunks):
                    comptime off = c * width
                    var v = acc.slice[width, offset=off]() * inv_sum
                    (d_head + off).store(v.cast[DType.bfloat16]())


# ================================================================
# HIGH-LEVEL OPERATIONS
# ================================================================


def gemm[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: BurstPool[],
) -> PoolFence where W.DTYPE == DType.bfloat16:
    """dst[M,N] = input[M,K] × weight[N,K]^T. M is runtime, via BurstPool.
    For small M (decode), partitions output columns across workers.
    For large M (prefill), partitions input rows."""
    comptime assert InT.DTYPE == DType.bfloat16, "gemm: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "gemm: output must be bf16"
    comptime assert InT.COLS == W.COLS, "gemm: input K != weight K"
    comptime assert OutT.COLS == W.ROWS, "gemm: output N != weight N"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "gemm: K must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence.completed()

    var fence = PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))

    if seq_len < pool.capacity:
        # Decode path: partition N (output columns) across workers
        comptime N = W.ROWS
        var num_jobs = pool.capacity
        var cols_per_job = (N + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * cols_per_job
            var end = min(start + cols_per_job, N)
            var pack = pool.args_base + i
            pack[].arg0 = input.ptr
            pack[].arg1 = weight.ptr
            pack[].arg2 = output.ptr
            pack[].arg3 = start
            pack[].arg4 = end
            pack[].arg5 = seq_len

        pool.dispatch(gemv_kernel[InT.COLS, N], pool.args_base, num_jobs)
    else:
        # Prefill path: partition M (input rows) across workers
        var num_jobs = min(seq_len, pool.capacity)
        var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * rows_per_job
            var end = min(start + rows_per_job, seq_len)
            var pack = pool.args_base + i
            pack[].arg0 = input.ptr
            pack[].arg1 = weight.ptr
            pack[].arg2 = output.ptr
            pack[].arg3 = start
            pack[].arg4 = end
            pack[].arg5 = 0

        pool.dispatch(gemm_kernel[InT.COLS, W.ROWS], pool.args_base, num_jobs)

    return fence^


def rmsnorm[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    mut pool: BurstPool[],
    eps: Float32 = 1e-5,
) -> PoolFence where W.DTYPE == DType.bfloat16:
    """RMSNorm via BurstPool: output = (input / RMS(input)) * weight.
    F32 accumulation in registers, bf16 I/O."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm: hidden must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence.completed()

    var eps_copy = eps
    var eps_int = Int(UnsafePointer(to=eps_copy).bitcast[Int32]()[])

    var num_jobs = min(seq_len, pool.capacity)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.args_base + i
        pack[].arg0 = input.ptr
        pack[].arg1 = weight.ptr
        pack[].arg2 = output.ptr
        pack[].arg3 = start
        pack[].arg4 = end
        pack[].arg5 = eps_int

    pool.dispatch(rmsnorm_kernel[InT.COLS], pool.args_base, num_jobs)
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def embed_lookup[W: Encoding & Shaped, OutT: Encoding & Shaped](
    table: Bound[W], tokens: Int, output: DynView[OutT],
    mut pool: BurstPool[],
) -> PoolFence where W.DTYPE == DType.bfloat16:
    """Gather: for each token ID, copy table[id] → output row."""
    comptime assert OutT.DTYPE == DType.bfloat16, "embed: output must be bf16"
    comptime assert W.COLS == OutT.COLS, "embed: table hidden != output hidden"

    var seq_len = output.seq_len
    if seq_len == 0:
        return PoolFence.completed()

    var num_jobs = min(seq_len, pool.capacity)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        var pack = pool.args_base + i
        pack[].arg0 = table.ptr
        pack[].arg1 = tokens
        pack[].arg2 = output.ptr
        pack[].arg3 = start
        pack[].arg4 = end
        pack[].arg5 = 0

    pool.dispatch(embed_lookup_kernel[W.COLS], pool.args_base, num_jobs)
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


def silu_mul[GT: Encoding & Shaped, UT: Encoding & Shaped, DstT: Encoding & Shaped](
    gate: DynView[GT], up: DynView[UT], dst: DynView[DstT],
):
    """SwiGLU: dst = silu(gate) * up. F32 compute, bf16 I/O."""
    comptime assert GT.DTYPE == DType.bfloat16, "silu_mul: gate must be bf16"
    comptime assert UT.DTYPE == DType.bfloat16, "silu_mul: up must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "silu_mul: dst must be bf16"
    comptime assert GT.COLS == UT.COLS, "silu_mul: gate/up cols mismatch"
    comptime assert GT.COLS == DstT.COLS, "silu_mul: gate/dst cols mismatch"
    comptime assert GT.COLS % simd_width_of[DType.float32]() == 0, "silu_mul: cols must be f32-simd-aligned"

    var seq_len = gate.seq_len
    if seq_len == 0:
        return

    var gp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=gate.ptr
    )
    var up_ = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=up.ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=dst.ptr
    )
    comptime cols = GT.COLS
    comptime width = simd_width_of[DType.float32]()

    for i in range(0, seq_len * cols, width):
        var g = bf16_to_f32[width](gp, i)
        var u = bf16_to_f32[width](up_, i)
        var sig = 1.0 / (1.0 + exp_f32[width](-g))
        (dp + i).store((g * sig * u).cast[DType.bfloat16]())


def elem_add[AT: Encoding & Shaped, BT: Encoding & Shaped, DstT: Encoding & Shaped](
    a: DynView[AT], b: DynView[BT], dst: DynView[DstT],
):
    """Elementwise: dst = a + b. F32 compute, bf16 I/O."""
    comptime assert AT.DTYPE == DType.bfloat16, "elem_add: a must be bf16"
    comptime assert BT.DTYPE == DType.bfloat16, "elem_add: b must be bf16"
    comptime assert DstT.DTYPE == DType.bfloat16, "elem_add: dst must be bf16"
    comptime assert AT.COLS == BT.COLS, "elem_add: a/b cols mismatch"
    comptime assert AT.COLS == DstT.COLS, "elem_add: a/dst cols mismatch"
    comptime assert AT.COLS % simd_width_of[DType.float32]() == 0, "elem_add: cols must be f32-simd-aligned"

    var seq_len = a.seq_len
    if seq_len == 0:
        return

    var ap = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=a.ptr
    )
    var bp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=b.ptr
    )
    var dp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=dst.ptr
    )
    comptime width = simd_width_of[DType.float32]()

    for i in range(0, seq_len * AT.COLS, width):
        var av = bf16_to_f32[width](ap, i)
        var bv = bf16_to_f32[width](bp, i)
        (dp + i).store((av + bv).cast[DType.bfloat16]())


def init_rope_tables[CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    cos_buf: Bound[CosT], sin_buf: Bound[SinT], theta: Float64 = 10000.0,
) where CosT.DTYPE == DType.float32:
    """Precompute cos/sin tables for RoPE. Call once at model init."""
    comptime assert SinT.DTYPE == DType.float32, "rope init: sin must be f32"
    comptime assert CosT.ROWS == SinT.ROWS, "rope init: cos/sin rows mismatch"
    comptime assert CosT.COLS == SinT.COLS, "rope init: cos/sin cols mismatch"
    comptime assert CosT.COLS % simd_width_of[DType.float64]() == 0, "rope init: cols must be f64-simd-aligned"

    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=cos_buf.ptr
    )
    var sp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=sin_buf.ptr
    )
    comptime half = CosT.COLS
    comptime head_dim = half * 2
    comptime f64w = simd_width_of[DType.float64]()

    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            inv[k] = 1.0 / (theta ** (Float64(2 * (j + k)) / Float64(head_dim)))

        for pos in range(CosT.ROWS):
            var sc = sincos[f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cp + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sp + pos * half + j).store(sc.sin_val.cast[DType.float32]())


def rope[head_dim: Int, num_heads: Int,
    XT: Encoding & Shaped, CosT: Encoding & Shaped, SinT: Encoding & Shaped](
    x: DynView[XT], cos_table: Bound[CosT], sin_table: Bound[SinT], pos: Int,
) where CosT.DTYPE == DType.float32:
    """Rotary position embeddings, applied in-place per head."""
    comptime assert XT.DTYPE == DType.bfloat16, "rope: must be bf16"
    comptime assert XT.COLS == head_dim * num_heads, "rope: cols != heads * dim"
    comptime assert head_dim % 2 == 0, "rope: head_dim must be even"
    comptime assert SinT.DTYPE == DType.float32, "rope: sin table must be f32"
    comptime assert CosT.COLS == head_dim // 2, "rope: cos cols != head_dim/2"
    comptime assert SinT.COLS == head_dim // 2, "rope: sin cols != head_dim/2"
    comptime assert CosT.ROWS == SinT.ROWS, "rope: cos/sin capacity mismatch"
    comptime assert (head_dim // 2) % simd_width_of[DType.float32]() == 0, "rope: half must be f32-simd-aligned"

    var seq_len = x.seq_len
    if seq_len == 0:
        return

    var xp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=x.ptr
    )
    var cp = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=cos_table.ptr
    )
    var sn = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=sin_table.ptr
    )
    comptime half = head_dim // 2
    comptime width = simd_width_of[DType.float32]()
    comptime row_stride = num_heads * head_dim

    for m in range(seq_len):
        var actual_pos = pos + m
        var cos_row = cp + actual_pos * half
        var sin_row = sn + actual_pos * half
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


def kv_cache_write[SrcT: Encoding & Shaped, CT: Encoding & Shaped](
    src: DynView[SrcT], cache: CacheView[CT], pos: Int,
):
    """Copy src[seq_len, kv_dim] into cache at row pos."""
    comptime assert SrcT.DTYPE == DType.bfloat16, "kv_write: src must be bf16"
    comptime assert CT.DTYPE == DType.bfloat16, "kv_write: cache must be bf16"
    comptime assert SrcT.COLS == CT.COLS, "kv_write: src cols != cache cols"
    comptime assert SrcT.COLS % simd_width_of[DType.bfloat16]() == 0, "kv_write: cols must be bf16-simd-aligned"

    var seq_len = src.seq_len
    if seq_len == 0:
        return

    var sp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=src.ptr
    )
    var cp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=cache.ptr
    )
    comptime cols = SrcT.COLS
    comptime width = simd_width_of[DType.bfloat16]()

    for m in range(seq_len):
        var src_row = sp + m * cols
        var dst_row = cp + (pos + m) * cols
        for j in range(0, cols, width):
            (dst_row + j).store((src_row + j).load[width=width]())


def attention[num_heads: Int, num_kv_heads: Int, head_dim: Int,
    QT: Encoding & Shaped, KCT: Encoding & Shaped, VCT: Encoding & Shaped,
    OutT: Encoding & Shaped](
    q: DynView[QT], k_cache: CacheView[KCT], v_cache: CacheView[VCT],
    output: DynView[OutT], pos: Int,
    mut pool: BurstPool[],
) -> PoolFence where KCT.DTYPE == DType.bfloat16:
    """GQA attention: Q[M, H*D] attends over KV cache[0..pos+M, Hkv*D].
    Causal masked, online softmax (single-pass, no score buffer).
    Work partitioned by KV head group via BurstPool."""
    comptime assert QT.DTYPE == DType.bfloat16, "attention: Q must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "attention: output must be bf16"
    comptime assert VCT.DTYPE == DType.bfloat16, "attention: V cache must be bf16"
    comptime assert QT.COLS == num_heads * head_dim, "attention: Q cols != H*D"
    comptime assert OutT.COLS == QT.COLS, "attention: output cols != Q cols"
    comptime assert KCT.COLS == num_kv_heads * head_dim, "attention: K cache cols != Hkv*D"
    comptime assert VCT.COLS == KCT.COLS, "attention: V cache cols != K cache cols"
    comptime assert KCT.ROWS == VCT.ROWS, "attention: K/V capacity mismatch"
    comptime assert head_dim % simd_width_of[DType.float32]() == 0, "attention: head_dim must be f32-simd-aligned"
    comptime assert num_heads % num_kv_heads == 0, "attention: heads must divide evenly for GQA"

    var seq_len = q.seq_len
    if seq_len == 0:
        return PoolFence.completed()

    var pos_seq = (pos << 32) | seq_len

    var num_jobs = min(num_kv_heads, pool.capacity)
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * groups_per_job
        var end = min(start + groups_per_job, num_kv_heads)
        var pack = pool.args_base + i
        pack[].arg0 = q.ptr
        pack[].arg1 = output.ptr
        pack[].arg2 = k_cache.ptr
        pack[].arg3 = v_cache.ptr
        pack[].arg4 = (start << 32) | end
        pack[].arg5 = pos_seq

    pool.dispatch(
        gqa_kernel[num_heads, num_kv_heads, head_dim, KCT.COLS],
        pool.args_base, num_jobs,
    )
    return PoolFence(UnsafePointer[BurstPool[], MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
