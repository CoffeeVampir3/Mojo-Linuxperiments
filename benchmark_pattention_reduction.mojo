"""Benchmark fused context-parallel attention reduction at TP=4.

Measures the full fused CP pipeline with real NUMA topology and BurstPools.
Each NUMA rank IS a CP rank — it owns a chunk group of the KV cache positions.

Per iteration:
  1. Each rank merges its local chunks → (m_r, l_r, v_r) per Q head
  2. Scalar all-gather: read (m_r, l_r) from all ranks → global M, L
  3. Each rank rescales v_r by alpha_r/(127*L), absmax quantize to i8
  4. Each rank dispatches O-proj GEMV to its NUMA-local BurstPool (parallel)
  5. ring_allreduce sums the bf16 O-proj outputs across all 4 ranks

This is mathematically equivalent to merge-first by O-proj linearity:
  O(sum_r(v_r * alpha_r / L)) = sum_r(O(v_r * alpha_r / L))

The allreduce is the same one already used for TP — no extra communication.
The only new cross-rank traffic is the scalar all-gather (~128 bytes).

Synthetic data, no model load. Designed for remote_build.fish → 4-node server.
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.collections import InlineArray

from numa import NumaArena, NumaInfo
from notstdcollections import HeapMoveArray
from threading import BurstPool
from kernels.kernel_ops import PoolFence
from kernels.reductions import ring_allreduce

from experimental3.kernels.full_chunked_attention import (
    partial_head_stride, partial_chunk_stride,
)
from experimental3.kernels.full_chunked_attention_fused import cp_merge_and_quantize
from experimental3.kernels.dense_ffn import int8_gemv_blocked
from experimental3.common_math import I8Ptr, U8Ptr, F32Ptr, BF16Ptr
from modeling.model_spec import Mat, BF16

from modeling.gemma_4_moe_butterquant_tp import Gemma4Config

comptime C = Gemma4Config
comptime HEAD_DIM = C.HEAD_DIM_FULL
comptime HPG = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
comptime HIDDEN = C.HIDDEN
comptime Q_DIM = C.Q_DIM_FULL
comptime NUM_HEADS = C.NUM_HEADS
comptime NUM_KV = C.NUM_KV_HEADS_FULL
comptime HEAD_STRIDE = partial_head_stride[HEAD_DIM]()
comptime CHUNK_STRIDE = partial_chunk_stride[HEAD_DIM, HPG]()
comptime QI_GROUP = HPG * HEAD_DIM
comptime TP = 4
comptime NUM_CHUNKS = 8
comptime WARMUP = 200
comptime ITERS = 2000
comptime X_SLOT = Mat[BF16, 1, HIDDEN]

# O-proj column-sharded: each rank does HIDDEN × Q_DIM_LOCAL
comptime Q_DIM_LOCAL = Q_DIM // TP
comptime WEIGHT_BYTES = HIDDEN * Q_DIM_LOCAL
comptime COLSUM_F32S = (Q_DIM_LOCAL // HEAD_DIM) * HIDDEN
comptime HEADS_PER_RANK = NUM_HEADS // TP

# Each rank's chunk allocation from the total NUM_CHUNKS
comptime CHUNKS_PER_RANK = (NUM_CHUNKS + TP - 1) // TP


# =============================================================================
# Timing
# =============================================================================


def sort_samples(mut samples: List[Int]):
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if samples[j] < samples[i]:
                var tmp = samples[i]
                samples[i] = samples[j]
                samples[j] = tmp


def report_timing(label: String, mut samples: List[Int]):
    sort_samples(samples)
    var total = 0
    for i in range(len(samples)):
        total += samples[i]
    var mean = total // len(samples)
    var p50 = samples[len(samples) // 2]
    var p99 = samples[Int(Float64(len(samples)) * 0.99)]
    var min_ns = samples[0]
    print("  ", label,
          "mean:", mean, "ns",
          "p50:", p50, "ns",
          "p99:", p99, "ns",
          "min:", min_ns, "ns",
          "(", mean // 1000, "us )")


# =============================================================================
# Synthetic data
# =============================================================================


def fill_partials(buf: UnsafePointer[Float32, MutAnyOrigin], num_chunks: Int):
    var seed = UInt32(0xDEADBEEF)
    for c in range(num_chunks):
        for qh in range(HPG):
            var base = buf + c * CHUNK_STRIDE + qh * HEAD_STRIDE
            base[] = Float32(-0.5) + Float32(c) * Float32(0.1)
            (base + 1)[] = Float32(5.0) + Float32(qh) * Float32(0.3)
            for d in range(HEAD_DIM):
                seed = seed * 1664525 + 1013904223
                var f = Float32(Int32(seed >> 16)) / Float32(32768.0)
                (base + 2 + d)[] = f * Float32(0.01)


def fill_weights(
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    wscale: UnsafePointer[Float32, MutAnyOrigin],
    colsum: UnsafePointer[Float32, MutAnyOrigin],
):
    var seed = UInt32(0xCAFEBABE)
    for i in range(WEIGHT_BYTES):
        seed = seed * 1664525 + 1013904223
        (wpacked + i)[] = UInt8(seed >> 24)
    for i in range(HIDDEN):
        wscale[i] = Float32(0.01)
    for i in range(COLSUM_F32S):
        colsum[i] = Float32(64.0)


# =============================================================================
# Per-rank NUMA-local buffer layout
# =============================================================================


@fieldwise_init
struct RankBuffers(Copyable, ImplicitlyCopyable):
    var partials: UnsafePointer[Float32, MutAnyOrigin]
    var qi: UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
    var head_scales: UnsafePointer[Float32, MutAnyOrigin]
    var oproj_out: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
    var wpacked: UnsafePointer[UInt8, MutAnyOrigin]
    var wscale: UnsafePointer[Float32, MutAnyOrigin]
    var colsum: UnsafePointer[Float32, MutAnyOrigin]
    var local_m: UnsafePointer[Float32, MutAnyOrigin]
    var local_l: UnsafePointer[Float32, MutAnyOrigin]
    var local_v: UnsafePointer[Float32, MutAnyOrigin]

    def __init__(out self):
        self.partials = UnsafePointer[Float32, MutAnyOrigin]()
        self.qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]()
        self.head_scales = UnsafePointer[Float32, MutAnyOrigin]()
        self.oproj_out = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]()
        self.wpacked = UnsafePointer[UInt8, MutAnyOrigin]()
        self.wscale = UnsafePointer[Float32, MutAnyOrigin]()
        self.colsum = UnsafePointer[Float32, MutAnyOrigin]()
        self.local_m = UnsafePointer[Float32, MutAnyOrigin]()
        self.local_l = UnsafePointer[Float32, MutAnyOrigin]()
        self.local_v = UnsafePointer[Float32, MutAnyOrigin]()


def alloc_rank_buffers(mut arena: NumaArena[alignment=64]) -> RankBuffers:
    var rb = RankBuffers()
    comptime PARTIAL_TOTAL = NUM_KV * CHUNKS_PER_RANK * CHUNK_STRIDE
    rb.partials = arena.alloc[Float32](PARTIAL_TOTAL)
    rb.qi = arena.alloc[Scalar[DType.int8]](Q_DIM_LOCAL)
    rb.head_scales = arena.alloc[Float32](HEADS_PER_RANK)
    rb.oproj_out = arena.alloc[Scalar[DType.bfloat16]](HIDDEN)
    rb.wpacked = arena.alloc[UInt8](WEIGHT_BYTES)
    rb.wscale = arena.alloc[Float32](HIDDEN)
    rb.colsum = arena.alloc[Float32](COLSUM_F32S)
    # Per-rank scalars: m and l per Q head (all heads, for all KV groups)
    rb.local_m = arena.alloc[Float32](NUM_HEADS)
    rb.local_l = arena.alloc[Float32](NUM_HEADS)
    # Per-rank merged v: all heads × HEAD_DIM
    rb.local_v = arena.alloc[Float32](NUM_HEADS * HEAD_DIM)
    return rb^


# =============================================================================
# Benchmark
# =============================================================================


def bench_fused(
    rank_bufs: InlineArray[RankBuffers, TP],
    pool_ptrs: InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP],
):
    var merge_samples = List[Int]()
    var gemv_samples = List[Int]()
    var reduce_samples = List[Int]()
    var e2e_samples = List[Int]()

    # Build pointer arrays for cross-rank access
    var all_m = InlineArray[UnsafePointer[Float32, MutAnyOrigin], TP](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    var all_l = InlineArray[UnsafePointer[Float32, MutAnyOrigin], TP](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    var all_v = InlineArray[UnsafePointer[Float32, MutAnyOrigin], TP](
        fill=UnsafePointer[Float32, MutAnyOrigin]())
    for r in range(TP):
        all_m[r] = rank_bufs[r].local_m
        all_l[r] = rank_bufs[r].local_l
        all_v[r] = rank_bufs[r].local_v

    for iteration in range(WARMUP + ITERS):
        var t0 = Int(perf_counter_ns())

        # =================================================================
        # Phases 1-3: cp_merge_and_quantize per rank
        # In production each rank runs this on its own core. Here we run
        # sequentially — the kernel is ~20us total, noise vs the GEMV.
        # =================================================================
        for rank in range(TP):
            var rb = rank_bufs[rank]
            cp_merge_and_quantize[HEAD_DIM, HPG, NUM_KV, NUM_HEADS, TP](
                rank,
                rb.partials,
                CHUNKS_PER_RANK,
                rb.local_m, rb.local_l, rb.local_v,
                all_m, all_l, all_v,
                rb.qi, rb.head_scales,
                rank * HEADS_PER_RANK, HEADS_PER_RANK)

        var t1 = Int(perf_counter_ns())

        # =================================================================
        # Phase 3: O-proj GEMV — dispatch all 4 ranks simultaneously
        # Each rank's pool runs GEMV on its NUMA-local data.
        # =================================================================
        var fence_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP](
            fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
        for rank in range(TP):
            var rb = rank_bufs[rank]
            fence_ptrs[rank] = int8_gemv_blocked[HIDDEN, Q_DIM_LOCAL, HEAD_DIM](
                I8Ptr(unsafe_from_address=Int(rb.qi)),
                U8Ptr(unsafe_from_address=Int(rb.wpacked)),
                F32Ptr(unsafe_from_address=Int(rb.head_scales)),
                F32Ptr(unsafe_from_address=Int(rb.wscale)),
                F32Ptr(unsafe_from_address=Int(rb.colsum)),
                BF16Ptr(unsafe_from_address=Int(rb.oproj_out)),
                1, pool_ptrs[rank][]).take()
        for rank in range(TP):
            if fence_ptrs[rank]:
                fence_ptrs[rank][].join()
        var t2 = Int(perf_counter_ns())

        # =================================================================
        # Phase 4: ring_allreduce — sum per-rank bf16 O-proj outputs
        # Same allreduce already used for TP, no extra communication.
        # =================================================================
        var out_ptrs = InlineArray[Int, TP](fill=0)
        for rank in range(TP):
            out_ptrs[rank] = Int(rank_bufs[rank].oproj_out)
        ring_allreduce[X_SLOT, TP](out_ptrs, 1, pool_ptrs)
        var t3 = Int(perf_counter_ns())

        if iteration >= WARMUP:
            merge_samples.append(t1 - t0)
            gemv_samples.append(t2 - t1)
            reduce_samples.append(t3 - t2)
            e2e_samples.append(t3 - t0)

    print("fused cp (tp=" + String(TP) + ", chunks=" + String(NUM_CHUNKS) + "):")
    report_timing("merge+quant ", merge_samples)
    report_timing("o-proj gemv ", gemv_samples)
    report_timing("allreduce   ", reduce_samples)
    report_timing("end-to-end  ", e2e_samples)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=== Context-Parallel Attention Reduction Benchmark ===")
    print("TP:", TP, "HEAD_DIM:", HEAD_DIM, "HPG:", HPG,
          "HIDDEN:", HIDDEN, "Q_DIM:", Q_DIM)
    print("Q_DIM_LOCAL:", Q_DIM_LOCAL, "NUM_KV:", NUM_KV,
          "HEADS_PER_RANK:", HEADS_PER_RANK)
    print("chunks:", NUM_CHUNKS, "chunks/rank:", CHUNKS_PER_RANK)
    print("warmup:", WARMUP, "iters:", ITERS)
    print()

    # --- NUMA topology ---
    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)
    print("NUMA topology (ring order):", end="")
    for r in range(TP):
        print(" node", numa_topo[r], end="")
    print()

    # --- NUMA-local arenas ---
    comptime PARTIAL_BYTES = NUM_KV * CHUNKS_PER_RANK * CHUNK_STRIDE * 4
    comptime SCALAR_BYTES = NUM_HEADS * 4 * 2 + NUM_HEADS * HEAD_DIM * 4
    comptime ARENA_SIZE = (PARTIAL_BYTES + Q_DIM_LOCAL + HEADS_PER_RANK * 4
        + HIDDEN * 2 + WEIGHT_BYTES + HIDDEN * 4 + COLSUM_F32S * 4
        + SCALAR_BYTES + 65536)

    var arenas = HeapMoveArray[NumaArena[alignment=64]](TP)
    for rank in range(TP):
        var arena = NumaArena[alignment=64](numa_topo[rank], ARENA_SIZE)
        if not arena:
            print("arena alloc failed rank", rank, "node", numa_topo[rank])
            return
        arenas.push(arena^)

    # --- Per-rank buffers ---
    var rank_bufs = InlineArray[RankBuffers, TP](fill=RankBuffers())
    for rank in range(TP):
        rank_bufs[rank] = alloc_rank_buffers(arenas[rank])

    # --- Synthetic data ---
    for rank in range(TP):
        var rb = rank_bufs[rank]
        for kv in range(NUM_KV):
            fill_partials(rb.partials + kv * CHUNKS_PER_RANK * CHUNK_STRIDE, CHUNKS_PER_RANK)
        fill_weights(rb.wpacked, rb.wscale, rb.colsum)
    print("synthetic data filled")

    for rank in range(TP):
        _ = arenas[rank].prefault()
    print("arenas prefaulted")

    # --- BurstPools ---
    var pools = HeapMoveArray[BurstPool[]](TP)
    for rank in range(TP):
        pools.push(BurstPool[].for_numa_node(numa, numa_topo[rank]))
    var pool_ptrs = InlineArray[UnsafePointer[BurstPool[], MutAnyOrigin], TP](
        fill=UnsafePointer[BurstPool[], MutAnyOrigin]())
    for rank in range(TP):
        pool_ptrs[rank] = UnsafePointer[BurstPool[], MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pools[rank])))

    print("pools:", end="")
    for rank in range(TP):
        print(" rank", rank, "=", pools[rank].capacity, "workers", end="")
    print()
    print()

    bench_fused(rank_bufs, pool_ptrs)

    _ = arenas
    _ = pools
