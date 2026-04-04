"""Int8 GEMV — VNNI-packed weights, precomputed bias/scale epilogue.

dst[m, n] = (dot_i32(act_u8[m], weight_i8[n]) - bias[n]) * scale[n]

Where bias and scale are precomputed at load time from colsum + row_scale.
Activation is i8, XOR'd to u8 in the VNNI dot product.
Weight is i8, VNNI-packed at load time.
Output is bf16.

Dispatch modes:
  int8_gemv:         single matrix, workers split N columns
  int8_gemv_batched: multiple matrices in one dispatch
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of, CompilationTarget
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from kernels.vnni import VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental.hadquant_impl import int8_gemm_dot


# ============================================================================
# Row kernel — one activation row × VNNI weight → bf16 for [start_n, end_n)
# ============================================================================


@always_inline
def int8_gemv_row(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    bias: UnsafePointer[Float32, MutAnyOrigin],
    scale: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_total: Int,
    k_total: Int,
    start_n: Int,
    end_n: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width

    var n_block = compute_n_block(n_total, k_total)
    var packed_off = 0

    for nb in range(0, n_total, n_block):
        var nb_size = min(n_block, n_total - nb)

        for ns in range(0, nb_size, VNNI_N_STEP):
            var n_base = nb + ns

            # Skip tile groups outside our assigned range
            if n_base + VNNI_N_STEP <= start_n or n_base >= end_n:
                # Advance packed_off past this N_STEP's K iterations
                packed_off += (k_total // VNNI_BLK) * VNNI_TILE_N * VNNI_BLK * 2
                continue

            var acc_buf = InlineArray[SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0)
            )
            var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, width]]()
            var local_off = packed_off

            for ks in range(0, k_total, VNNI_K_STEP):
                # First subtile (lower VNNI_TILE_N channels)
                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[p] = int8_gemm_dot[width](
                            acc[p], act_row,
                            wpacked + local_off + p * bytes_per_pass,
                            k_pos,
                        )
                    local_off += VNNI_TILE_N * VNNI_BLK

                # Second subtile (upper VNNI_TILE_N channels)
                for dc in range(VNNI_K_STEP // VNNI_BLK):
                    var k_pos = ks + dc * VNNI_BLK
                    for p in range(passes_per_subtile):
                        acc[passes_per_subtile + p] = int8_gemm_dot[width](
                            acc[passes_per_subtile + p], act_row,
                            wpacked + local_off + p * bytes_per_pass,
                            k_pos,
                        )
                    local_off += VNNI_TILE_N * VNNI_BLK

            packed_off = local_off

            # Epilogue: (raw_i32 - bias[n]) * scale[n] → bf16
            for a in range(acc_count):
                var nc = n_base + a * width
                if nc >= start_n and nc + width <= end_n:
                    var raw = acc[a].cast[DType.float32]()
                    var result = (raw - (bias + nc).load[width=width]()) * (scale + nc).load[width=width]()
                    (dst + nc).store(result.cast[DType.bfloat16]())


# ============================================================================
# Shared context — lives on caller stack, workers read via pointer
# ============================================================================


@fieldwise_init
struct GemvCtx:
    var act_ptr: Int
    var dst_ptr: Int
    var n_total: Int
    var k_total: Int
    var seq_len: Int


@fieldwise_init
struct BatchedGemvEntry(Copyable, ImplicitlyCopyable):
    var weight_ptr: Int
    var bias_ptr: Int
    var scale_ptr: Int
    var dst_ptr: Int
    var n_total: Int

    def __init__(out self):
        self.weight_ptr = 0
        self.bias_ptr = 0
        self.scale_ptr = 0
        self.dst_ptr = 0
        self.n_total = 0


@fieldwise_init
struct BatchedGemvCtx:
    var act_ptr: Int
    var k_total: Int
    var seq_len: Int
    var entries: InlineArray[BatchedGemvEntry, 3]
    var count: Int

    def __init__(out self):
        self.act_ptr = 0
        self.k_total = 0
        self.seq_len = 0
        self.entries = InlineArray[BatchedGemvEntry, 3](fill=BatchedGemvEntry())
        self.count = 0


# ============================================================================
# Worker kernels (BurstPool ABI: 6 Int args)
# ============================================================================


def single_gemv_worker(
    ctx_addr: Int, weight_ptr: Int, bias_ptr: Int,
    scale_ptr: Int, start_n: Int, end_n: Int,
):
    var ctx = UnsafePointer[GemvCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=ctx[].act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=weight_ptr)
    var bias = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=bias_ptr)
    var scale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=scale_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ctx[].dst_ptr)
    var n_total = ctx[].n_total
    var k_total = ctx[].k_total

    for m in range(ctx[].seq_len):
        int8_gemv_row(
            act + m * k_total, wpacked, bias, scale,
            dst + m * n_total,
            n_total, k_total, start_n, end_n,
        )


def batched_gemv_worker(
    ctx_addr: Int, start_n: Int, end_n: Int,
    which: Int, unused0: Int, unused1: Int,
):
    var ctx = UnsafePointer[BatchedGemvCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var entry = ctx[].entries[which]
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=ctx[].act_ptr)
    var wpacked = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=entry.weight_ptr)
    var bias = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=entry.bias_ptr)
    var scale = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=entry.scale_ptr)
    var dst = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=entry.dst_ptr)
    var n_total = entry.n_total
    var k_total = ctx[].k_total

    for m in range(ctx[].seq_len):
        int8_gemv_row(
            act + m * k_total, wpacked, bias, scale,
            dst + m * n_total,
            n_total, k_total, start_n, end_n,
        )


# ============================================================================
# Single dispatch — one matrix, workers split N
# ============================================================================


def int8_gemv[P: BurstThreadPool](
    mut ctx: GemvCtx,
    weight_ptr: Int, bias_ptr: Int, scale_ptr: Int,
    mut pool: P,
) -> PoolFence[P]:
    if ctx.seq_len == 0:
        return PoolFence[P].completed()

    var ctx_addr = Int(UnsafePointer(to=ctx))
    var num_jobs = pool.get_capacity()
    var cols_per_job = (ctx.n_total + num_jobs - 1) // num_jobs

    for i in range(num_jobs):
        var start = i * cols_per_job
        var end = min(start + cols_per_job, ctx.n_total)
        var pack = pool.get_args_base() + i
        pack[].arg0 = ctx_addr
        pack[].arg1 = weight_ptr
        pack[].arg2 = bias_ptr
        pack[].arg3 = scale_ptr
        pack[].arg4 = start
        pack[].arg5 = end

    pool.dispatch(single_gemv_worker, pool.get_args_base(), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ============================================================================
# Batched dispatch — multiple matrices, one dispatch
# ============================================================================


def int8_gemv_batched[P: BurstThreadPool](
    mut ctx: BatchedGemvCtx,
    mut pool: P,
) -> PoolFence[P]:
    if ctx.seq_len == 0 or ctx.count == 0:
        return PoolFence[P].completed()

    var cap = pool.get_capacity()
    var ctx_addr = Int(UnsafePointer(to=ctx))

    var total_n = 0
    for i in range(ctx.count):
        total_n += ctx.entries[i].n_total

    var job = 0
    for i in range(ctx.count):
        var n_i = ctx.entries[i].n_total
        var jobs_i = max(1, cap * n_i // total_n)
        if i == ctx.count - 1:
            jobs_i = cap - job
        var cols_per = (n_i + jobs_i - 1) // jobs_i
        for j in range(jobs_i):
            if job >= cap:
                break
            var pack = pool.get_args_base() + job
            pack[].arg0 = ctx_addr
            pack[].arg1 = j * cols_per
            pack[].arg2 = min((j + 1) * cols_per, n_i)
            pack[].arg3 = i
            job += 1

    pool.dispatch(batched_gemv_worker, pool.get_args_base(), job)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))
