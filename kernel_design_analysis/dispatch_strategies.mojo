"""Kernel dispatch strategy benchmarks.

Measures the cost of different dispatch patterns on NUMA topology to
determine optimal kernel composition strategy for butterquant forward.

Hypotheses tested:
  H1: ranks.each (main-thread sequential) vs ranks.parallel (1 worker/rank)
      for a lightweight NUMA-local op. Tests remote read/write penalty.

  H2: Redundant replicated work (all ranks compute independently) vs
      single-producer + broadcast for a small op (rmsnorm-scale vector).

  H3: Batched GEMM dispatch (Q+K+V as one dispatch with 3x jobs) vs
      three separate dispatch/join cycles.

  H4: Redundant small GEMM (every worker computes full GEMM, decode-sized)
      vs standard tiled parallel dispatch.
"""

from std.time import perf_counter_ns
from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of, size_of
from std.collections import InlineArray
from std.benchmark import keep

from numa import NumaInfo, NumaTopology
from numa.arena import NumaArena
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from kernels.kernel_ops import (
    PoolFence, parallel_for, timed_parallel_for, ParallelTiming,
)
from modeling.model_spec import BF16, F32, Slot, Bound, DynView, Replicated

from simd_math import sqrt


def memcpy_kernel(
    dst: Int, src: Int, count: Int,
    unused0: Int, unused1: Int, unused2: Int,
):
    memcpy(
        dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        count=count,
    )


comptime NUM_NODES = 4
comptime TRIALS = 20
comptime WARMUP = 3


struct ModelDims:
    var hidden: Int
    var kv_hidden: Int
    var q_out: Int     # local Q output per rank (HIDDEN / TP for RowShard)
    var kv_out: Int    # local KV output per rank
    var intermediate: Int

    def __init__(out self, hidden: Int, kv_hidden: Int, q_out: Int, kv_out: Int, intermediate: Int):
        self.hidden = hidden
        self.kv_hidden = kv_hidden
        self.q_out = q_out
        self.kv_out = kv_out
        self.intermediate = intermediate


# ============================================================================
# Lightweight op — stand-in for rmsnorm_fwht_quantize on a single row
# ============================================================================


def lightweight_op_kernel(
    src: Int, dst: Int, cols: Int,
    unused0: Int, unused1: Int, unused2: Int,
):
    """RMSNorm-like: read bf16, compute norm, scale, write bf16.
    Simulates a per-row lightweight op on NUMA-local memory."""
    comptime width = simd_width_of[DType.float32]()
    var ip = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=src)
    var op = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dst)

    var acc = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (ip + k).load[width=width]().cast[DType.float32]()
        acc = x.fma(x, acc)
        k += width
    var rms = sqrt[DType.float32, 1](acc.reduce_add() / Float32(cols) + 1e-5)
    var inv_rms = SIMD[DType.float32, width](Float32(1.0) / rms)

    k = 0
    while k + width <= cols:
        var x = (ip + k).load[width=width]().cast[DType.float32]()
        (op + k).store((x * inv_rms).cast[DType.bfloat16]())
        k += width


def lightweight_op_inline(
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    cols: Int,
):
    """Same op, callable directly (not through BurstPool ABI)."""
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0)
    var k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        acc = x.fma(x, acc)
        k += width
    var rms = sqrt[DType.float32, 1](acc.reduce_add() / Float32(cols) + 1e-5)
    var inv_rms = SIMD[DType.float32, width](Float32(1.0) / rms)

    k = 0
    while k + width <= cols:
        var x = (src + k).load[width=width]().cast[DType.float32]()
        (dst + k).store((x * inv_rms).cast[DType.bfloat16]())
        k += width


# ============================================================================
# Small GEMV — decode-sized [1, K] x [N, K]^T -> [1, N]
# ============================================================================


def gemv_full_kernel(
    ip: Int, wp: Int, dp: Int,
    k_dim: Int, n_dim: Int, unused: Int,
):
    """Full GEMV: one worker computes all N output columns."""
    comptime width = simd_width_of[DType.float32]()
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ip)
    var wgt = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=wp)
    var out = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dp)

    for n in range(n_dim):
        var row_w = wgt + n * k_dim
        var acc = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= k_dim:
            var x = (inp + k).load[width=width]().cast[DType.float32]()
            var w = (row_w + k).load[width=width]().cast[DType.float32]()
            acc = x.fma(w, acc)
            k += width
        out[n] = acc.reduce_add().cast[DType.bfloat16]()


def gemv_slice_kernel(
    ip: Int, wp: Int, dp: Int,
    k_dim: Int, start_n: Int, end_n: Int,
):
    """Tiled GEMV: one worker computes columns [start_n, end_n)."""
    comptime width = simd_width_of[DType.float32]()
    var inp = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ip)
    var wgt = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=wp)
    var out = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=dp)

    for n in range(start_n, end_n):
        var row_w = wgt + n * k_dim
        var acc = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= k_dim:
            var x = (inp + k).load[width=width]().cast[DType.float32]()
            var w = (row_w + k).load[width=width]().cast[DType.float32]()
            acc = x.fma(w, acc)
            k += width
        out[n] = acc.reduce_add().cast[DType.bfloat16]()


# ============================================================================
# Batched multi-GEMV kernel — job index encodes which weight matrix
# ============================================================================


def batched_gemv_kernel(
    ctx_addr: Int, start_n: Int, end_n: Int,
    which_gemm: Int, unused0: Int, unused1: Int,
):
    """Batched GEMV: ctx_addr points to BatchedGemvCtx, which_gemm selects
    which of the 3 weight matrices (Q/K/V) this job handles."""
    comptime width = simd_width_of[DType.float32]()
    var ctx = UnsafePointer[BatchedGemvCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var ip = ctx[].input
    var k_dim = ctx[].k_dim
    var wp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var dp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    if which_gemm == 0:
        wp = ctx[].w0; dp = ctx[].d0
    elif which_gemm == 1:
        wp = ctx[].w1; dp = ctx[].d1
    else:
        wp = ctx[].w2; dp = ctx[].d2

    for n in range(start_n, end_n):
        var row_w = wp + n * k_dim
        var acc = SIMD[DType.float32, width](0)
        var k = 0
        while k + width <= k_dim:
            var x = (ip + k).load[width=width]().cast[DType.float32]()
            var w = (row_w + k).load[width=width]().cast[DType.float32]()
            acc = x.fma(w, acc)
            k += width
        dp[n] = acc.reduce_add().cast[DType.bfloat16]()


@fieldwise_init
struct BatchedGemvCtx(Copyable, ImplicitlyCopyable):
    var input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var w0: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var w1: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var w2: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var d0: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var d1: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var d2: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var k_dim: Int


# ============================================================================
# Main
# ============================================================================


def main():
    var numa = NumaInfo()
    if numa.num_nodes < NUM_NODES:
        print("SKIP: need " + String(NUM_NODES) + " NUMA nodes, have " + String(numa.num_nodes))
        return
    var topo = numa.plan_topology(NUM_NODES)

    print("=== Dispatch Strategy Benchmarks ===")
    print("NUMA: " + String(NUM_NODES) + " nodes")

    # SmolLM2-135M TP=3: tiny model, dispatch overhead dominates
    var smol = ModelDims(
        hidden=576, kv_hidden=192,
        q_out=192, kv_out=64,       # 576/3, 192/3
        intermediate=512,            # 1536/3
    )
    # DeepSeek-V3 TP=4: large model, compute dominates
    # hidden=7168, 128 heads, hd=128, 128 kv_heads (dense attn), intermediate=18432
    var dsv3 = ModelDims(
        hidden=7168, kv_hidden=7168,
        q_out=1792, kv_out=1792,     # 7168/4
        intermediate=4608,           # 18432/4
    )

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](NUM_NODES)
        for i in range(NUM_NODES):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_all(numa, topo, pools, smol, "SmolLM2-135M (TP=3)")
        run_all(numa, topo, pools, dsv3, "DeepSeek-V3 (TP=4)")
    else:
        print("mode: cold")
        var pools = HeapMoveArray[BurstPool[]](NUM_NODES)
        for i in range(NUM_NODES):
            pools.push(BurstPool[].for_topology(numa, topo[i], stack_size=2 * 1024 * 1024))
            print("  node " + String(topo[i]) + ": " + String(pools[i].get_capacity()) + " workers")
        run_all(numa, topo, pools, smol, "SmolLM2-135M (TP=3)")
        run_all(numa, topo, pools, dsv3, "DeepSeek-V3 (TP=4)")


def run_all[P: BurstThreadPool](
    numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P],
    dims: ModelDims,
    name: String,
):
    print("\n\n========== " + name + " ==========")
    print("  hidden=" + String(dims.hidden) + " kv_hidden=" + String(dims.kv_hidden)
        + " q_out=" + String(dims.q_out) + " kv_out=" + String(dims.kv_out))

    # Arena per node: enough for weights + activations
    # Q weight: q_out × hidden, K/V weight: kv_out × hidden
    var weight_bytes = (dims.q_out * dims.hidden + 2 * dims.kv_out * dims.hidden) * 2
    var act_bytes = dims.hidden * 2 * 4  # x, y, plus some output buffers
    var arena_size = weight_bytes + act_bytes + 16 * 1024 * 1024  # headroom

    var arenas = HeapMoveArray[NumaArena[]](NUM_NODES)
    for i in range(NUM_NODES):
        arenas.push(NumaArena[](topo[i], arena_size))

    var x_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var y_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var w_q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var w_k_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var w_v_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var d_q_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var d_k_ptrs = InlineArray[Int, NUM_NODES](fill=0)
    var d_v_ptrs = InlineArray[Int, NUM_NODES](fill=0)

    for node in range(NUM_NODES):
        var x = arenas[node].alloc[Scalar[DType.bfloat16]](dims.hidden)
        var y = arenas[node].alloc[Scalar[DType.bfloat16]](dims.hidden)
        for i in range(dims.hidden):
            x[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 128.0)
        x_ptrs[node] = Int(x)
        y_ptrs[node] = Int(y)

        var wq = arenas[node].alloc[Scalar[DType.bfloat16]](dims.q_out * dims.hidden)
        var wk = arenas[node].alloc[Scalar[DType.bfloat16]](dims.kv_out * dims.hidden)
        var wv = arenas[node].alloc[Scalar[DType.bfloat16]](dims.kv_out * dims.hidden)
        for i in range(dims.q_out * dims.hidden):
            wq[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 256.0)
        for i in range(dims.kv_out * dims.hidden):
            wk[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 256.0)
            wv[i] = Scalar[DType.bfloat16](Float32(i % 256 - 128) / 256.0)
        w_q_ptrs[node] = Int(wq)
        w_k_ptrs[node] = Int(wk)
        w_v_ptrs[node] = Int(wv)

        var dq = arenas[node].alloc[Scalar[DType.bfloat16]](dims.q_out)
        var dk = arenas[node].alloc[Scalar[DType.bfloat16]](dims.kv_out)
        var dv = arenas[node].alloc[Scalar[DType.bfloat16]](dims.kv_out)
        d_q_ptrs[node] = Int(dq)
        d_k_ptrs[node] = Int(dk)
        d_v_ptrs[node] = Int(dv)

    h1_ranks_each_vs_parallel(numa, topo, pools, arenas, x_ptrs, y_ptrs, dims)
    h2_replicate_vs_broadcast(numa, topo, pools, arenas, x_ptrs, y_ptrs, dims)
    h3_batched_vs_separate_gemm(numa, topo, pools, arenas,
        x_ptrs, w_q_ptrs, w_k_ptrs, w_v_ptrs, d_q_ptrs, d_k_ptrs, d_v_ptrs, dims)
    h4_redundant_vs_tiled_gemv(numa, topo, pools, arenas,
        x_ptrs, w_q_ptrs, d_q_ptrs, dims)


# ============================================================================
# H1: ranks.each vs ranks.parallel for lightweight op
# ============================================================================


def h1_ranks_each_vs_parallel[P: BurstThreadPool](
    numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[]],
    x_ptrs: InlineArray[Int, NUM_NODES],
    y_ptrs: InlineArray[Int, NUM_NODES],
    dims: ModelDims,
):
    var h = dims.hidden
    print("\n--- H1: ranks.each vs ranks.parallel (lightweight op, " + String(h) + " elements) ---")

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            lightweight_op_inline(
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=x_ptrs[node]),
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=y_ptrs[node]),
                h,
            )

    var best_each = Int(1 << 60)
    for trial in range(TRIALS):
        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            lightweight_op_inline(
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=x_ptrs[node]),
                UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=y_ptrs[node]),
                h,
            )
        var elapsed = Int(perf_counter_ns()) - t0
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=y_ptrs[0])[0])
        if elapsed < best_each:
            best_each = elapsed
    print("  ranks.each (main thread):  " + String(best_each) + " ns")

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            var pack = pools[node].get_args_base()
            pack[].arg0 = x_ptrs[node]
            pack[].arg1 = y_ptrs[node]
            pack[].arg2 = h
            pools[node].dispatch(lightweight_op_kernel, pools[node].get_args_base(), 1)
        for node in range(NUM_NODES):
            pools[node].join()

    var best_parallel = Int(1 << 60)
    var best_par_dispatch = Int(1 << 60)
    var best_par_last_worker = Int(1 << 60)
    for trial in range(TRIALS):
        for node in range(NUM_NODES):
            var pack = pools[node].get_args_base()
            pack[].arg0 = x_ptrs[node]
            pack[].arg1 = y_ptrs[node]
            pack[].arg2 = h

        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            pools[node].dispatch(lightweight_op_kernel, pools[node].get_args_base(), 1)
        var t_dispatched = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            pools[node].join()
        var t_joined = Int(perf_counter_ns())

        var max_worker_ts = 0
        for node in range(NUM_NODES):
            var ts = pools[node].last_worker_timestamp()
            if ts > max_worker_ts:
                max_worker_ts = ts

        var elapsed = t_joined - t0
        var dispatch_ns = t_dispatched - t0
        var join_overhead = t_joined - max_worker_ts

        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=y_ptrs[0])[0])
        if elapsed < best_parallel:
            best_parallel = elapsed
            best_par_dispatch = dispatch_ns
            best_par_last_worker = join_overhead

    print("  ranks.parallel (1w/rank):  " + String(best_parallel) + " ns"
        + "  (dispatch=" + String(best_par_dispatch) + " ns"
        + ", join_oh=" + String(best_par_last_worker) + " ns)")


# ============================================================================
# H2: Redundant replicated work vs single-producer + broadcast
# ============================================================================


def h2_replicate_vs_broadcast[P: BurstThreadPool](
    numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[]],
    x_ptrs: InlineArray[Int, NUM_NODES],
    y_ptrs: InlineArray[Int, NUM_NODES],
    dims: ModelDims,
):
    var h = dims.hidden
    var bc_bytes = h * 2
    print("\n--- H2: replicate vs broadcast (" + String(h) + " elements) ---")

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            var pack = pools[node].get_args_base()
            pack[].arg0 = x_ptrs[node]
            pack[].arg1 = y_ptrs[node]
            pack[].arg2 = h
            pools[node].dispatch(lightweight_op_kernel, pools[node].get_args_base(), 1)
        for node in range(NUM_NODES):
            pools[node].join()

    var best_replicate = Int(1 << 60)
    for trial in range(TRIALS):
        for node in range(NUM_NODES):
            var pack = pools[node].get_args_base()
            pack[].arg0 = x_ptrs[node]
            pack[].arg1 = y_ptrs[node]
            pack[].arg2 = h

        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            pools[node].dispatch(lightweight_op_kernel, pools[node].get_args_base(), 1)
        for node in range(NUM_NODES):
            pools[node].join()
        var elapsed = Int(perf_counter_ns()) - t0
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=y_ptrs[0])[0])
        if elapsed < best_replicate:
            best_replicate = elapsed

    for _ in range(WARMUP):
        var pack = pools[0].get_args_base()
        pack[].arg0 = x_ptrs[0]
        pack[].arg1 = y_ptrs[0]
        pack[].arg2 = h
        pools[0].dispatch(lightweight_op_kernel, pools[0].get_args_base(), 1)
        pools[0].join()
        for dst in range(1, NUM_NODES):
            var p = pools[dst].get_args_base()
            p[].arg0 = y_ptrs[dst]
            p[].arg1 = y_ptrs[0]
            p[].arg2 = bc_bytes
            pools[dst].dispatch(memcpy_kernel, pools[dst].get_args_base(), 1)
        for dst in range(1, NUM_NODES):
            pools[dst].join()

    var best_broadcast = Int(1 << 60)
    for trial in range(TRIALS):
        var pack = pools[0].get_args_base()
        pack[].arg0 = x_ptrs[0]
        pack[].arg1 = y_ptrs[0]
        pack[].arg2 = h

        var t0 = Int(perf_counter_ns())
        pools[0].dispatch(lightweight_op_kernel, pools[0].get_args_base(), 1)
        pools[0].join()
        for dst in range(1, NUM_NODES):
            var p = pools[dst].get_args_base()
            p[].arg0 = y_ptrs[dst]
            p[].arg1 = y_ptrs[0]
            p[].arg2 = bc_bytes
            pools[dst].dispatch(memcpy_kernel, pools[dst].get_args_base(), 1)
        for dst in range(1, NUM_NODES):
            pools[dst].join()
        var elapsed = Int(perf_counter_ns()) - t0
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=y_ptrs[NUM_NODES - 1])[0])
        if elapsed < best_broadcast:
            best_broadcast = elapsed

    print("  replicate (all compute):   " + String(best_replicate) + " ns")
    print("  single + broadcast:        " + String(best_broadcast) + " ns")


# ============================================================================
# H3: Batched Q+K+V GEMV vs three separate dispatches
# ============================================================================


def h3_batched_vs_separate_gemm[P: BurstThreadPool](
    numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[]],
    x_ptrs: InlineArray[Int, NUM_NODES],
    w_q_ptrs: InlineArray[Int, NUM_NODES],
    w_k_ptrs: InlineArray[Int, NUM_NODES],
    w_v_ptrs: InlineArray[Int, NUM_NODES],
    d_q_ptrs: InlineArray[Int, NUM_NODES],
    d_k_ptrs: InlineArray[Int, NUM_NODES],
    d_v_ptrs: InlineArray[Int, NUM_NODES],
    dims: ModelDims,
):
    var h = dims.hidden
    var qn = dims.q_out
    var kvn = dims.kv_out
    print("\n--- H3: batched Q+K+V vs 3 separate (decode, K=" + String(h)
        + ", Q_N=" + String(qn) + ", KV_N=" + String(kvn) + ") ---")

    var cap = pools[0].get_capacity()

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_q_ptrs[node], d_q_ptrs[node], h, qn)
        for node in range(NUM_NODES): pools[node].join()
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_k_ptrs[node], d_k_ptrs[node], h, kvn)
        for node in range(NUM_NODES): pools[node].join()
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_v_ptrs[node], d_v_ptrs[node], h, kvn)
        for node in range(NUM_NODES): pools[node].join()

    var best_separate = Int(1 << 60)
    for trial in range(TRIALS):
        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_q_ptrs[node], d_q_ptrs[node], h, qn)
        for node in range(NUM_NODES): pools[node].join()
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_k_ptrs[node], d_k_ptrs[node], h, kvn)
        for node in range(NUM_NODES): pools[node].join()
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_v_ptrs[node], d_v_ptrs[node], h, kvn)
        for node in range(NUM_NODES): pools[node].join()
        var elapsed = Int(perf_counter_ns()) - t0
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=d_q_ptrs[0])[0])
        if elapsed < best_separate:
            best_separate = elapsed

    var ctxs = InlineArray[BatchedGemvCtx, NUM_NODES](fill=BatchedGemvCtx(
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](),
        0,
    ))
    for node in range(NUM_NODES):
        ctxs[node] = BatchedGemvCtx(
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=x_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=w_q_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=w_k_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=w_v_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=d_q_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=d_k_ptrs[node]),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=d_v_ptrs[node]),
            h,
        )

    var total_n = qn + kvn + kvn
    var q_jobs = max(1, cap * qn // total_n)
    var k_jobs = max(1, cap * kvn // total_n)
    var v_jobs = max(1, cap - q_jobs - k_jobs)
    var total_jobs = q_jobs + k_jobs + v_jobs

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            _dispatch_batched_gemv(pools[node], ctxs[node], q_jobs, k_jobs, v_jobs, qn, kvn)
        for node in range(NUM_NODES): pools[node].join()

    var best_batched = Int(1 << 60)
    for trial in range(TRIALS):
        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            _dispatch_batched_gemv(pools[node], ctxs[node], q_jobs, k_jobs, v_jobs, qn, kvn)
        for node in range(NUM_NODES): pools[node].join()
        var elapsed = Int(perf_counter_ns()) - t0
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=d_q_ptrs[0])[0])
        if elapsed < best_batched:
            best_batched = elapsed

    print("  3 separate dispatch/join:  " + String(best_separate // 1000) + " us")
    print("  1 batched dispatch:        " + String(best_batched // 1000) + " us")
    print("  jobs: Q=" + String(q_jobs) + " K=" + String(k_jobs) + " V=" + String(v_jobs)
        + " total=" + String(total_jobs) + "/" + String(cap))


# ============================================================================
# H4: Redundant full GEMV (every worker) vs tiled parallel GEMV
# ============================================================================


def h4_redundant_vs_tiled_gemv[P: BurstThreadPool](
    numa: NumaInfo,
    topo: NumaTopology,
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[]],
    x_ptrs: InlineArray[Int, NUM_NODES],
    w_ptrs: InlineArray[Int, NUM_NODES],
    d_ptrs: InlineArray[Int, NUM_NODES],
    dims: ModelDims,
):
    var k = dims.hidden
    var n = dims.q_out
    print("\n--- H4: redundant full GEMV vs tiled parallel (decode, [1," + String(k)
        + "] x [" + String(n) + "," + String(k) + "]^T) ---")

    var cap = pools[0].get_capacity()

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            for w in range(cap):
                var pack = pools[node].get_args_base() + w
                pack[].arg0 = x_ptrs[node]
                pack[].arg1 = w_ptrs[node]
                pack[].arg2 = d_ptrs[node]
                pack[].arg3 = k
                pack[].arg4 = n
            pools[node].dispatch(gemv_full_kernel, pools[node].get_args_base(), cap)
        for node in range(NUM_NODES): pools[node].join()

    var best_redundant = Int(1 << 60)
    var best_red_last_worker = Int(1 << 60)
    for trial in range(TRIALS):
        for node in range(NUM_NODES):
            for w in range(cap):
                var pack = pools[node].get_args_base() + w
                pack[].arg0 = x_ptrs[node]
                pack[].arg1 = w_ptrs[node]
                pack[].arg2 = d_ptrs[node]
                pack[].arg3 = k
                pack[].arg4 = n

        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            pools[node].dispatch(gemv_full_kernel, pools[node].get_args_base(), cap)
        for node in range(NUM_NODES): pools[node].join()
        var t_end = Int(perf_counter_ns())

        var max_worker_ts = 0
        for node in range(NUM_NODES):
            var ts = pools[node].last_worker_timestamp()
            if ts > max_worker_ts:
                max_worker_ts = ts

        var elapsed = t_end - t0
        var join_oh = t_end - max_worker_ts
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=d_ptrs[0])[0])
        if elapsed < best_redundant:
            best_redundant = elapsed
            best_red_last_worker = join_oh

    for _ in range(WARMUP):
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_ptrs[node], d_ptrs[node], k, n)
        for node in range(NUM_NODES): pools[node].join()

    var best_tiled = Int(1 << 60)
    var best_tiled_last_worker = Int(1 << 60)
    for trial in range(TRIALS):
        var t0 = Int(perf_counter_ns())
        for node in range(NUM_NODES):
            _dispatch_gemv_tiled(pools[node], x_ptrs[node], w_ptrs[node], d_ptrs[node], k, n)
        for node in range(NUM_NODES): pools[node].join()
        var t_end = Int(perf_counter_ns())

        var max_worker_ts = 0
        for node in range(NUM_NODES):
            var ts = pools[node].last_worker_timestamp()
            if ts > max_worker_ts:
                max_worker_ts = ts

        var elapsed = t_end - t0
        var join_oh = t_end - max_worker_ts
        keep(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=d_ptrs[0])[0])
        if elapsed < best_tiled:
            best_tiled = elapsed
            best_tiled_last_worker = join_oh

    print("  redundant (all " + String(cap) + "w full):  " + String(best_redundant // 1000) + " us"
        + "  (join_oh=" + String(best_red_last_worker) + " ns)")
    print("  tiled (" + String(cap) + "w split N):    " + String(best_tiled // 1000) + " us"
        + "  (join_oh=" + String(best_tiled_last_worker) + " ns)")


# ============================================================================
# Dispatch helpers
# ============================================================================


def _dispatch_gemv_tiled[P: BurstThreadPool](
    mut pool: P,
    x_ptr: Int, w_ptr: Int, d_ptr: Int,
    k_dim: Int, n_dim: Int,
):
    var cap = pool.get_capacity()
    var cols_per_job = (n_dim + cap - 1) // cap
    for w in range(cap):
        var start = w * cols_per_job
        var end = min(start + cols_per_job, n_dim)
        var pack = pool.get_args_base() + w
        pack[].arg0 = x_ptr
        pack[].arg1 = w_ptr
        pack[].arg2 = d_ptr
        pack[].arg3 = k_dim
        pack[].arg4 = start
        pack[].arg5 = end
    pool.dispatch(gemv_slice_kernel, pool.get_args_base(), cap)


def _dispatch_batched_gemv[P: BurstThreadPool](
    mut pool: P,
    mut ctx: BatchedGemvCtx,
    q_jobs: Int, k_jobs: Int, v_jobs: Int,
    q_n: Int, kv_n: Int,
):
    var ctx_addr = Int(UnsafePointer(to=ctx))
    var job = 0

    var q_cols_per = (q_n + q_jobs - 1) // q_jobs
    for j in range(q_jobs):
        var pack = pool.get_args_base() + job
        pack[].arg0 = ctx_addr
        pack[].arg1 = j * q_cols_per
        pack[].arg2 = min((j + 1) * q_cols_per, q_n)
        pack[].arg3 = 0
        job += 1

    var k_cols_per = (kv_n + k_jobs - 1) // k_jobs
    for j in range(k_jobs):
        var pack = pool.get_args_base() + job
        pack[].arg0 = ctx_addr
        pack[].arg1 = j * k_cols_per
        pack[].arg2 = min((j + 1) * k_cols_per, kv_n)
        pack[].arg3 = 1
        job += 1

    var v_cols_per = (kv_n + v_jobs - 1) // v_jobs
    for j in range(v_jobs):
        var pack = pool.get_args_base() + job
        pack[].arg0 = ctx_addr
        pack[].arg1 = j * v_cols_per
        pack[].arg2 = min((j + 1) * v_cols_per, kv_n)
        pack[].arg3 = 2
        job += 1

    pool.dispatch(batched_gemv_kernel, pool.get_args_base(), job)
