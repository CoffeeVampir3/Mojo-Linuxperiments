from std.math import sqrt
from std.memory import UnsafePointer, Pointer
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from threading.threading_traits import BurstThreadPool
from threading.threading_shared import ptr as tptr
from notstdcollections import HeapMoveArray
import linux.sys as linux

from modeling.model_spec import (
    Encoding, Shaped, Bound, DynView, CacheView,
)
from simd_math import exp_f32


comptime MAX_POOL_CAPACITY = 128
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@fieldwise_init
struct GemmArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int
    var seq_len: Int


@fieldwise_init
struct RMSNormArgs(Copyable, ImplicitlyCopyable):
    var input: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int
    var eps: Float64


@fieldwise_init
struct EmbedArgs(Copyable, ImplicitlyCopyable):
    var table: BF16Ptr
    var tokens: UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
    var output: BF16Ptr
    var start_row: Int
    var end_row: Int


@fieldwise_init
struct GQAArgs(Copyable, ImplicitlyCopyable):
    var qp: BF16Ptr
    var dp: BF16Ptr
    var kp: BF16Ptr
    var vp: BF16Ptr
    var start_group: Int
    var end_group: Int
    var pos: Int
    var seq_len: Int

@explicit_destroy
@fieldwise_init
struct PoolFence[P: BurstThreadPool, origin: MutOrigin](Movable):
    var pool: Pointer[Self.P, Self.origin]

    @staticmethod
    def over(ref [Self.origin] pool: Self.P) -> Self:
        return Self(Pointer(to=pool))

    def join(deinit self):
        self.pool[].join()

    def finish(deinit self) -> Int:
        self.pool[].join()
        return self.pool[].last_worker_timestamp()


def tp_dispatch_recursive[
    Pool: BurstThreadPool,
    Topo: Copyable & ImplicitlyCopyable, //,
    rank: Int, tp: Int,
    body: def[r: Int, origin: MutOrigin](Topo, ref [origin] Pool) capturing -> PoolFence[Pool, origin],
](
    topos: InlineArray[Topo, tp],
    mut pools: HeapMoveArray[Pool],
):
    comptime if rank < tp:
        var fence = body[rank, origin_of(pools)](topos[rank], pools[rank])
        tp_dispatch_recursive[rank + 1, tp, body](topos, pools)
        fence^.join()


def tp_parallel[
    Pool: BurstThreadPool,
    Topo: Copyable & ImplicitlyCopyable, //,
    tp: Int,
    body: def[r: Int, origin: MutOrigin](Topo, ref [origin] Pool) capturing -> PoolFence[Pool, origin],
](
    topos: InlineArray[Topo, tp],
    mut pools: HeapMoveArray[Pool],
):
    tp_dispatch_recursive[0, tp, body](topos, pools)


def gemm_kernel[K: Int, N: Int](args: GemmArgs):
    """GEMM row kernel. dst[m,n] = dot(input[m,:], weight[n,:])
    for rows [start_row, end_row)."""
    var ip = args.input
    var wp = args.weight
    var dp = args.output
    comptime width = simd_width_of[DType.float32]()

    for m in range(args.start, args.end):
        var row_in = ip + m * K
        var row_out = dp + m * N
        for n in range(N):
            var row_w = wp + n * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var x = (row_in + k).load[width=width]().cast[DType.float32]()
                var w = (row_w + k).load[width=width]().cast[DType.float32]()
                acc = x.fma(w, acc)
            row_out[n] = acc.reduce_add().cast[DType.bfloat16]()


def gemv_kernel[K: Int, N: Int](args: GemmArgs):
    """N-tiled GEMV kernel — processes 4 output columns simultaneously,
    reusing each input load across 4 independent FMA chains to hide
    FMA latency and reduce input bandwidth."""
    var ip = args.input
    var wp = args.weight
    var dp = args.output
    comptime width = simd_width_of[DType.float32]()
    comptime Nr = 4

    var n_full = args.end - ((args.end - args.start) % Nr)

    for m in range(args.seq_len):
        var row_in = ip + m * K
        var row_out = dp + m * N

        for n in range(args.start, n_full, Nr):
            var w0 = wp + n * K
            var w1 = wp + (n + 1) * K
            var w2 = wp + (n + 2) * K
            var w3 = wp + (n + 3) * K
            var acc0 = SIMD[DType.float32, width](0)
            var acc1 = SIMD[DType.float32, width](0)
            var acc2 = SIMD[DType.float32, width](0)
            var acc3 = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                var x = (row_in + k).load[width=width]().cast[DType.float32]()
                acc0 = x.fma((w0 + k).load[width=width]().cast[DType.float32](), acc0)
                acc1 = x.fma((w1 + k).load[width=width]().cast[DType.float32](), acc1)
                acc2 = x.fma((w2 + k).load[width=width]().cast[DType.float32](), acc2)
                acc3 = x.fma((w3 + k).load[width=width]().cast[DType.float32](), acc3)
            row_out[n] = acc0.reduce_add().cast[DType.bfloat16]()
            row_out[n + 1] = acc1.reduce_add().cast[DType.bfloat16]()
            row_out[n + 2] = acc2.reduce_add().cast[DType.bfloat16]()
            row_out[n + 3] = acc3.reduce_add().cast[DType.bfloat16]()

        for n in range(n_full, args.end):
            var row_w = wp + n * K
            var acc = SIMD[DType.float32, width](0)
            for k in range(0, K, width):
                acc = (row_in + k).load[width=width]().cast[DType.float32]().fma(
                    (row_w + k).load[width=width]().cast[DType.float32](), acc)
            row_out[n] = acc.reduce_add().cast[DType.bfloat16]()


def rmsnorm_kernel[cols: Int](args: RMSNormArgs):
    """RMSNorm row kernel. Fused reduction + normalize for
    rows [start_row, end_row)."""
    var ip = args.input
    var wp = args.weight
    var dp = args.output
    comptime width = simd_width_of[DType.float32]()

    for row in range(args.start_row, args.end_row):
        var row_in = ip + row * cols
        var row_out = dp + row * cols

        var acc = SIMD[DType.float32, width](0)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            acc = x.fma(x, acc)
        var sum_sq = acc.reduce_add()
        var scale = Float32(1.0) / sqrt(sum_sq / Float32(cols) + Float32(args.eps))

        var sv = SIMD[DType.float32, width](scale)
        for j in range(0, cols, width):
            var x = (row_in + j).load[width=width]().cast[DType.float32]()
            var w = (wp + j).load[width=width]().cast[DType.float32]()
            (row_out + j).store((x * sv * w).cast[DType.bfloat16]())


def embed_lookup_kernel[cols: Int](args: EmbedArgs):
    """Embed gather kernel. Copies table[token_id] → dst
    for rows [start_row, end_row)."""
    var tp = args.table
    var tokens = args.tokens
    var dp = args.output
    comptime width = simd_width_of[DType.bfloat16]()

    for i in range(args.start_row, args.end_row):
        var src = tp + Int(tokens[i]) * cols
        var out = dp + i * cols
        for j in range(0, cols, width):
            (out + j).store((src + j).load[width=width]())


def gqa_kernel[
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    kv_cols: Int,
](args: GQAArgs):
    """GQA attention kernel. Per query head, fuses:
      1. GEMV — dot(q, K[t]) → score
      2. Online softmax — streaming max + exp + sum
      3. V accumulation — weighted sum in f32 registers"""
    var qp = args.qp
    var dp = args.dp
    var kp = args.kp
    var vp = args.vp
    comptime heads_per_group = num_heads // num_kv_heads
    comptime q_stride = num_heads * head_dim
    comptime kv_stride = kv_cols
    comptime width = simd_width_of[DType.float32]()
    comptime chunks = head_dim // width
    comptime scale_f32 = Float32(1.0 / sqrt(Float64(head_dim)))

    for m in range(args.seq_len):
        var causal_len = args.pos + m + 1

        for g in range(args.start_group, args.end_group):
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
                        var qv = (q_head + off).load[width=width]().cast[DType.float32]()
                        var kv = (k_row + off).load[width=width]().cast[DType.float32]()
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
                        var vv = (v_row + off).load[width=width]().cast[DType.float32]()
                        acc = acc.insert[offset=off](prior * correction + vv * w)

                    running_sum = correction * running_sum + w
                    running_max = new_max

                # Normalize and store
                var inv_sum = 1.0 / running_sum
                comptime for c in range(chunks):
                    comptime off = c * width
                    var v = acc.slice[width, offset=off]() * inv_sum
                    (d_head + off).store(v.cast[DType.bfloat16]())


def gemm[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool, origin: MutOrigin, //](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    ref [origin] pool: P,
) -> PoolFence[P, origin] where W.DTYPE == DType.bfloat16:
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
        return PoolFence[P, origin].over(pool)

    var ip = input.as_ptr[DType.bfloat16]()
    var wp = weight.as_ptr[DType.bfloat16]()
    var op = output.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[GemmArgs, MAX_POOL_CAPACITY](uninitialized=True)

    if seq_len < pool.get_capacity():
        comptime N = W.ROWS
        var num_jobs = pool.get_capacity()
        var cols_per_job = (N + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * cols_per_job
            var end = min(start + cols_per_job, N)
            jobs[i] = GemmArgs(ip, wp, op, start, end, seq_len)

        pool.dispatch[GemmArgs, gemv_kernel[InT.COLS, N]](
            UnsafePointer(to=jobs[0]), num_jobs)
    else:
        var num_jobs = min(seq_len, pool.get_capacity())
        var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

        for i in range(num_jobs):
            var start = i * rows_per_job
            var end = min(start + rows_per_job, seq_len)
            jobs[i] = GemmArgs(ip, wp, op, start, end, 0)

        pool.dispatch[GemmArgs, gemm_kernel[InT.COLS, W.ROWS]](
            UnsafePointer(to=jobs[0]), num_jobs)

    return PoolFence[P, origin].over(pool)


def rmsnorm[W: Encoding & Shaped, InT: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool, origin: MutOrigin, //](
    input: DynView[InT], weight: Bound[W], output: DynView[OutT],
    ref [origin] pool: P,
    eps: Float32 = 1e-5,
) -> PoolFence[P, origin] where W.DTYPE == DType.bfloat16:
    """RMSNorm via BurstPool: output = (input / RMS(input)) * weight.
    F32 accumulation in registers, bf16 I/O."""
    comptime assert InT.DTYPE == DType.bfloat16, "rmsnorm: input must be bf16"
    comptime assert OutT.DTYPE == DType.bfloat16, "rmsnorm: output must be bf16"
    comptime assert InT.COLS == OutT.COLS, "rmsnorm: input/output cols mismatch"
    comptime assert InT.COLS % simd_width_of[DType.float32]() == 0, "rmsnorm: hidden must be f32-simd-aligned"

    var seq_len = input.seq_len
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var ip = input.as_ptr[DType.bfloat16]()
    var wp = weight.as_ptr[DType.bfloat16]()
    var op = output.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[RMSNormArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = RMSNormArgs(ip, wp, op, start, end, Float64(eps))

    pool.dispatch[RMSNormArgs, rmsnorm_kernel[InT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


def embed_lookup[W: Encoding & Shaped, OutT: Encoding & Shaped,
    P: BurstThreadPool, origin: MutOrigin, //](
    table: Bound[W], tokens: Int, output: DynView[OutT],
    ref [origin] pool: P,
) -> PoolFence[P, origin] where W.DTYPE == DType.bfloat16:
    """Gather: for each token ID, copy table[id] → output row."""
    comptime assert OutT.DTYPE == DType.bfloat16, "embed: output must be bf16"
    comptime assert W.COLS == OutT.COLS, "embed: table hidden != output hidden"

    var seq_len = output.seq_len
    if seq_len == 0:
        return PoolFence[P, origin].over(pool)

    var num_jobs = min(seq_len, pool.get_capacity())
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var tp = table.as_ptr[DType.bfloat16]()
    var tkp = tptr[Scalar[DType.int32]](tokens)
    var op = output.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[EmbedArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, seq_len)
        jobs[i] = EmbedArgs(tp, tkp, op, start, end)

    pool.dispatch[EmbedArgs, embed_lookup_kernel[W.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)


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

    var gp = gate.as_ptr[DType.bfloat16]()
    var up_ = up.as_ptr[DType.bfloat16]()
    var dp = dst.as_ptr[DType.bfloat16]()
    comptime cols = GT.COLS
    comptime width = simd_width_of[DType.float32]()

    for i in range(0, seq_len * cols, width):
        var g = (gp + i).load[width=width]().cast[DType.float32]()
        var u = (up_ + i).load[width=width]().cast[DType.float32]()
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

    var ap = a.as_ptr[DType.bfloat16]()
    var bp = b.as_ptr[DType.bfloat16]()
    var dp = dst.as_ptr[DType.bfloat16]()
    comptime width = simd_width_of[DType.float32]()

    for i in range(0, seq_len * AT.COLS, width):
        var av = (ap + i).load[width=width]().cast[DType.float32]()
        var bv = (bp + i).load[width=width]().cast[DType.float32]()
        (dp + i).store((av + bv).cast[DType.bfloat16]())


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
    debug_assert(pos >= 0 and pos + seq_len <= CT.ROWS,
        "kv_write: position range exceeds cache capacity")

    var sp = src.as_ptr[DType.bfloat16]()
    var cp = cache.as_ptr[DType.bfloat16]()
    comptime cols = SrcT.COLS
    comptime width = simd_width_of[DType.bfloat16]()

    for m in range(seq_len):
        var src_row = sp + m * cols
        var dst_row = cp + (pos + m) * cols
        for j in range(0, cols, width):
            (dst_row + j).store((src_row + j).load[width=width]())


def attention[
    P: BurstThreadPool, origin: MutOrigin, //,
    num_heads: Int, num_kv_heads: Int, head_dim: Int,
    QT: Encoding & Shaped, KCT: Encoding & Shaped, VCT: Encoding & Shaped,
    OutT: Encoding & Shaped,
](
    q: DynView[QT], k_cache: CacheView[KCT], v_cache: CacheView[VCT],
    output: DynView[OutT], pos: Int,
    ref [origin] pool: P,
) -> PoolFence[P, origin] where KCT.DTYPE == DType.bfloat16:
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
        return PoolFence[P, origin].over(pool)

    var num_jobs = min(num_kv_heads, pool.get_capacity())
    var groups_per_job = (num_kv_heads + num_jobs - 1) // num_jobs

    var qpp = q.as_ptr[DType.bfloat16]()
    var opp = output.as_ptr[DType.bfloat16]()
    var kpp = k_cache.as_ptr[DType.bfloat16]()
    var vpp = v_cache.as_ptr[DType.bfloat16]()
    var jobs = InlineArray[GQAArgs, MAX_POOL_CAPACITY](uninitialized=True)
    for i in range(num_jobs):
        var start = i * groups_per_job
        var end = min(start + groups_per_job, num_kv_heads)
        jobs[i] = GQAArgs(qpp, opp, kpp, vpp, start, end, pos, seq_len)

    pool.dispatch[GQAArgs, gqa_kernel[num_heads, num_kv_heads, head_dim, KCT.COLS]](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P, origin].over(pool)
