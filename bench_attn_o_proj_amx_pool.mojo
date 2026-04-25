from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.collections import InlineArray
from std.math import max

from numa import NumaArena, NumaInfo, NumaTopology
from notstdcollections import HeapMoveArray
from threading.isolated_burst_pool import IsolatedBurstPool

from experimental3.amx import (
    TILE_M, TILE_N, K_STEP, VNNI_BLK, TILE_BYTES,
    make_224_i8_config, init_intel_amx, ldtilecfg,
)
from experimental3.kernels.dispatch_args import WorkerConfig, Int8GemvBlockedArgs
from experimental3.kernels.gemm import (
    int8_gemv_worker, int8_gemv_blocked_worker,
)
from experimental3.kernels.gemm_amx import (
    int8_gemm_amx_worker, int8_gemm_blocked_amx_worker,
)
from minimax.kernels.amx_attention import AmxConfigArgs, amx_prefill_config_kernel
from kernels.kernel_ops import MAX_POOL_CAPACITY
from modeling.model_spec import DEFAULT_ALIGNMENT
from simd_math import set_subnormal_zeroing


comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime TP = 4
comptime ITERS = 50
comptime WARMUP = 10

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# Test data (heap-allocated, simple)
# ============================================================================


def alloc_zeroed[T: DType](count: Int) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
    var p = alloc[Scalar[T]](count, alignment=64)
    for i in range(count):
        (p + i)[] = Scalar[T](0)
    return UnsafePointer[Scalar[T], MutAnyOrigin](unsafe_from_address=Int(p))


def fill_random_i8(ptr: I8Ptr, count: Int, seed: UInt64 = 0xDEADBEEF12345678):
    var state = seed
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        ptr[i] = Scalar[DType.int8]((state >> 33).cast[DType.int8]())


def fill_random_f32(ptr: F32Ptr, count: Int, scale: Float32):
    var state = UInt64(0xFEEDFACE11111111)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var frac = Float32(Int(state >> 33)) / Float32(2147483648)
        ptr[i] = frac * scale + Float32(0.001)


def pack_vnni(src: I8Ptr, dst: I8Ptr, N: Int, K: Int):
    for nb in range(0, N, N_STEP):
        for kb in range(0, K, K_STEP):
            var tile_base = nb * K + kb * N_STEP
            for sub in range(2):
                var sub_base = tile_base + sub * TILE_BYTES
                for kg in range(TILE_M):
                    for n in range(TILE_N):
                        comptime for s in range(VNNI_BLK):
                            var src_row = nb + sub * TILE_N + n
                            var src_col = kb + kg * VNNI_BLK + s
                            dst[sub_base + kg * K_STEP + n * VNNI_BLK + s] = (
                                src[src_row * K + src_col])


def compute_colsum(src: I8Ptr, colsum: F32Ptr, N: Int, K: Int):
    for n in range(N):
        var acc = Int32(0)
        for k in range(K):
            acc += Int32(src[n * K + k])
        colsum[n] = Float32(acc)


def compute_block_colsums(
    src: I8Ptr, colsum: F32Ptr, N: Int, K: Int, fwht_blk: Int,
):
    var num_blocks = K // fwht_blk
    for n in range(N):
        for blk in range(num_blocks):
            var acc = Int32(0)
            for kk in range(fwht_blk):
                acc += Int32(src[n * K + blk * fwht_blk + kk])
            colsum[blk * N + n] = Float32(acc)


# ============================================================================
# Per-row dispatch round-trip
# ============================================================================


def time_per_row_dispatch[N: Int, K: Int, vnni: Bool](
    mut pool: IsolatedBurstPool[],
    act: I8Ptr, wpacked: I8Ptr, colsum: F32Ptr, w_scale: F32Ptr,
    act_scale: F32Ptr, dst: BF16Ptr,
    seq_len: Int,
) -> Int:
    var cap = Int(pool.get_capacity())
    var num_jobs = min(seq_len, cap)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[WorkerConfig, MAX_POOL_CAPACITY](fill=WorkerConfig())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= seq_len:
            break
        var end = min(start + rows_per_job, seq_len)
        jobs[actual] = WorkerConfig(
            act + start * K, wpacked,
            colsum, w_scale, dst + start * N,
            act_scale, start, end - start)
        actual += 1

    for _ in range(WARMUP):
        comptime if vnni:
            pool.dispatch[WorkerConfig, int8_gemv_worker[N, K]](
                UnsafePointer(to=jobs[0]), actual)
        else:
            pool.dispatch[WorkerConfig, int8_gemm_amx_worker[N, K]](
                UnsafePointer(to=jobs[0]), actual)
        pool.join()

    var t = Int(perf_counter_ns())
    for _ in range(ITERS):
        comptime if vnni:
            pool.dispatch[WorkerConfig, int8_gemv_worker[N, K]](
                UnsafePointer(to=jobs[0]), actual)
        else:
            pool.dispatch[WorkerConfig, int8_gemm_amx_worker[N, K]](
                UnsafePointer(to=jobs[0]), actual)
        pool.join()
    return (Int(perf_counter_ns()) - t) // ITERS


def time_blocked_dispatch[N: Int, K: Int, fwht_blk: Int, vnni: Bool](
    mut pool: IsolatedBurstPool[],
    act: I8Ptr, wpacked: I8Ptr, blk_colsum: F32Ptr, w_scale: F32Ptr,
    blk_scale: F32Ptr, dst: BF16Ptr,
    seq_len: Int, output_scale: Float32,
) -> Int:
    comptime num_blocks = K // fwht_blk
    var cap = Int(pool.get_capacity())
    var num_jobs = min(seq_len, cap)
    var rows_per_job = (seq_len + num_jobs - 1) // num_jobs

    var jobs = InlineArray[Int8GemvBlockedArgs, MAX_POOL_CAPACITY](
        fill=Int8GemvBlockedArgs())
    var actual = 0
    for i in range(num_jobs):
        var start = i * rows_per_job
        if start >= seq_len:
            break
        var end = min(start + rows_per_job, seq_len)
        jobs[actual] = Int8GemvBlockedArgs(
            act + start * K, wpacked,
            blk_scale + start * num_blocks, w_scale, blk_colsum,
            dst + start * N, output_scale, N, N, end - start)
        actual += 1

    for _ in range(WARMUP):
        comptime if vnni:
            pool.dispatch[Int8GemvBlockedArgs,
                int8_gemv_blocked_worker[N, K, fwht_blk]](
                UnsafePointer(to=jobs[0]), actual)
        else:
            pool.dispatch[Int8GemvBlockedArgs,
                int8_gemm_blocked_amx_worker[N, K, fwht_blk]](
                UnsafePointer(to=jobs[0]), actual)
        pool.join()

    var t = Int(perf_counter_ns())
    for _ in range(ITERS):
        comptime if vnni:
            pool.dispatch[Int8GemvBlockedArgs,
                int8_gemv_blocked_worker[N, K, fwht_blk]](
                UnsafePointer(to=jobs[0]), actual)
        else:
            pool.dispatch[Int8GemvBlockedArgs,
                int8_gemm_blocked_amx_worker[N, K, fwht_blk]](
                UnsafePointer(to=jobs[0]), actual)
        pool.join()
    return (Int(perf_counter_ns()) - t) // ITERS


# ============================================================================
# Sweep drivers
# ============================================================================


comptime SEQ_SWEEP_LEN = 7


def make_seq_sweep() -> InlineArray[Int, SEQ_SWEEP_LEN]:
    var s = InlineArray[Int, SEQ_SWEEP_LEN](fill=0)
    s[0] = 16
    s[1] = 32
    s[2] = 64
    s[3] = 128
    s[4] = 256
    s[5] = 512
    s[6] = 1024
    return s^


def run_per_row_sweep[N: Int, K: Int](
    label: String, mut pool: IsolatedBurstPool[],
):
    print("--- pool per_row: " + label
        + " (N=" + String(N) + ", K=" + String(K) + ") ---")

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xCAFEBABE87654321)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var colsum = alloc_zeroed[DType.float32](N)
    compute_colsum(weight_raw, colsum, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))

    var max_seq = 1024
    var act = alloc_zeroed[DType.int8]((max_seq + M_STEP) * K)
    fill_random_i8(act, (max_seq + M_STEP) * K)
    var act_scale = alloc_zeroed[DType.float32](max_seq + M_STEP)
    fill_random_f32(act_scale, max_seq + M_STEP, Float32(1.0))
    var dst = alloc_zeroed[DType.bfloat16]((max_seq + M_STEP) * N)

    var cap = Int(pool.get_capacity())
    print("  pool capacity=" + String(cap))
    print("       seq | per-worker M | VNNI ns | AMX ns |  speedup")
    var sweep = make_seq_sweep()
    var speedup_at_64 = Float32(0)
    var speedup_at_1024 = Float32(0)
    for idx in range(SEQ_SWEEP_LEN):
        var seq_len = sweep[idx]
        var num_jobs = min(seq_len, cap)
        var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
        var vnni_ns = time_per_row_dispatch[N, K, True](
            pool, act, wpacked, colsum, w_scale, act_scale, dst, seq_len)
        var amx_ns = time_per_row_dispatch[N, K, False](
            pool, act, wpacked, colsum, w_scale, act_scale, dst, seq_len)
        var speedup = Float32(vnni_ns) / Float32(amx_ns)
        print("   " + String(seq_len) + "    | " + String(rows_per_job)
            + " | " + String(vnni_ns)
            + " | " + String(amx_ns)
            + " | " + String(speedup))
        if seq_len == 64:
            speedup_at_64 = speedup
        if seq_len == 1024:
            speedup_at_1024 = speedup

    print("  summary: VNNI/AMX pool speedup at seq=64 = "
        + String(speedup_at_64)
        + "  at seq=1024 = " + String(speedup_at_1024))
    if speedup_at_1024 >= Float32(1.5):
        print("    GATE_PASS_1024 (>=1.5x at production seq)")
    else:
        print("    GATE_FAIL_1024 (<1.5x at production seq)")

    weight_raw.free()
    wpacked.free()
    colsum.free()
    w_scale.free()
    act.free()
    act_scale.free()
    dst.free()


def run_blocked_sweep[N: Int, K: Int, fwht_blk: Int](
    label: String, mut pool: IsolatedBurstPool[],
    output_scale: Float32 = Float32(0.85),
):
    print("--- pool blocked: " + label
        + " (N=" + String(N) + ", K=" + String(K)
        + ", blk=" + String(fwht_blk) + ") ---")

    comptime num_blocks = K // fwht_blk

    var weight_raw = alloc_zeroed[DType.int8](N * K)
    fill_random_i8(weight_raw, N * K, seed=0xB10C4ED123456789)
    var wpacked = alloc_zeroed[DType.int8](N * K)
    pack_vnni(weight_raw, wpacked, N, K)
    var w_scale = alloc_zeroed[DType.float32](N)
    fill_random_f32(w_scale, N, Float32(0.01))
    var blk_colsum = alloc_zeroed[DType.float32](num_blocks * N)
    compute_block_colsums(weight_raw, blk_colsum, N, K, fwht_blk)

    var max_seq = 1024
    var act = alloc_zeroed[DType.int8]((max_seq + M_STEP) * K)
    fill_random_i8(act, (max_seq + M_STEP) * K, seed=0xB10CA0C712340000)
    var blk_scale = alloc_zeroed[DType.float32]((max_seq + M_STEP) * num_blocks)
    fill_random_f32(blk_scale, (max_seq + M_STEP) * num_blocks, Float32(1.0))
    var dst = alloc_zeroed[DType.bfloat16]((max_seq + M_STEP) * N)

    var cap = Int(pool.get_capacity())
    print("  pool capacity=" + String(cap))
    print("       seq | per-worker M | VNNI ns | AMX ns |  speedup")
    var sweep = make_seq_sweep()
    var speedup_at_64 = Float32(0)
    var speedup_at_1024 = Float32(0)
    for idx in range(SEQ_SWEEP_LEN):
        var seq_len = sweep[idx]
        var num_jobs = min(seq_len, cap)
        var rows_per_job = (seq_len + num_jobs - 1) // num_jobs
        var vnni_ns = time_blocked_dispatch[N, K, fwht_blk, True](
            pool, act, wpacked, blk_colsum, w_scale, blk_scale, dst,
            seq_len, output_scale)
        var amx_ns = time_blocked_dispatch[N, K, fwht_blk, False](
            pool, act, wpacked, blk_colsum, w_scale, blk_scale, dst,
            seq_len, output_scale)
        var speedup = Float32(vnni_ns) / Float32(amx_ns)
        print("   " + String(seq_len) + "    | " + String(rows_per_job)
            + " | " + String(vnni_ns)
            + " | " + String(amx_ns)
            + " | " + String(speedup))
        if seq_len == 64:
            speedup_at_64 = speedup
        if seq_len == 1024:
            speedup_at_1024 = speedup

    print("  summary: VNNI/AMX pool speedup at seq=64 = "
        + String(speedup_at_64)
        + "  at seq=1024 = " + String(speedup_at_1024))
    if speedup_at_1024 >= Float32(1.5):
        print("    GATE_PASS_1024 (>=1.5x at production seq)")
    else:
        print("    GATE_FAIL_1024 (<1.5x at production seq)")

    weight_raw.free()
    wpacked.free()
    w_scale.free()
    blk_colsum.free()
    act.free()
    blk_scale.free()
    dst.free()


# ============================================================================
# Main: configure pool with 224_i8 once, run all sweeps on rank 0
# ============================================================================


def configure_amx_prefill[origin: MutOrigin](
    ref [origin] pool: IsolatedBurstPool[],
):
    var cap = Int(pool.get_capacity())
    var args = InlineArray[AmxConfigArgs, MAX_POOL_CAPACITY](
        fill=AmxConfigArgs())
    pool.dispatch[AmxConfigArgs, amx_prefill_config_kernel](
        UnsafePointer(to=args[0]), cap)
    pool.join()


def main():
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))

    var numa = NumaInfo()
    if not numa.has_isolation():
        print("ERROR: this bench requires CPU isolation (IsolatedBurstPool).")
        return
    var numa_topo = numa.plan_topology(TP)

    var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
    for rank in range(TP):
        pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))

    print("=== AMX vs VNNI: pool-dispatched attn_proj / o_proj bench ===")
    print("TP=" + String(TP) + " (running on rank 0 pool only)")
    print("")

    configure_amx_prefill(pools[0])

    print("[1/4] QKV TP=4 (N=2048, K=3072)")
    run_per_row_sweep[2048, 3072]("QKV TP=4", pools[0])
    print("")

    print("[2/4] QKV TP=8 (N=1024, K=3072)")
    run_per_row_sweep[1024, 3072]("QKV TP=8", pools[0])
    print("")

    print("[3/4] O TP=4 (N=3072, K=1536, blk=128)")
    run_blocked_sweep[3072, 1536, 128]("O TP=4", pools[0])
    print("")

    print("[4/4] O TP=8 (N=3072, K=768, blk=128)")
    run_blocked_sweep[3072, 768, 128]("O TP=8", pools[0])
    print("")

    print("Done. Gate to dispatcher edit: VNNI/AMX pool speedup >= 1.5x at seq=1024.")
    _ = pools^
