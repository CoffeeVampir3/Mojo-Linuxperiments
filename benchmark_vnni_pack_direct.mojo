from std.benchmark import keep
from std.collections import InlineArray
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from kernels.vnni import (
    VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK,
    compute_n_block, required_pack_scratch_bytes, pack_and_colsum_vnni,
)
from simd_math.matrixops import transpose_generic


comptime ROWS = 2048
comptime COLS = 3072
comptime BYTES = ROWS * COLS
comptime BLOCK_COLS = 128
comptime NUM_BLOCKS = COLS // BLOCK_COLS
comptime WARMUP = 8
comptime ITERS = 1024

# Kernel under test for perf attribution. Flip to switch between
# "legacy" (write-then-reread packer) and "fused" (streaming + colsum).
# Validation still exercises both paths before the timed section.
comptime BENCH_LEGACY = False


@no_inline
def pack_vnni_legacy(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """The previous production packer — kept here as a perf/correctness
    baseline. Copies each 32x64 tile into dst row-major, then re-reads it
    as int32 rows and in-place transposes. The write-then-read of dst is
    the source of the store-forward stalls we want to measure."""
    var N = rows
    var K = cols
    debug_assert(K % VNNI_K_STEP == 0,
        "pack_vnni_legacy: K must be a multiple of VNNI_K_STEP (64)")
    debug_assert(N % VNNI_N_STEP == 0,
        "pack_vnni_legacy: N must be a multiple of VNNI_N_STEP (32)")
    var n_block = compute_n_block(N, K)
    var k_block = K
    var scratch = InlineArray[SIMD[DType.int32, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, n_block):
        var n_block_size = min(n_block, N - n_block_begin)
        for k_block_begin in range(0, K, k_block):
            var k_block_size = min(k_block, K - k_block_begin)
            for n_begin in range(0, n_block_size, VNNI_N_STEP):
                for k_begin in range(0, k_block_size, VNNI_K_STEP):
                    var tile_base = (
                        n_block_begin * K
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * VNNI_N_STEP
                    )
                    for i in range(VNNI_N_STEP):
                        var src_off = (n_block_begin + n_begin + i) * K + k_block_begin + k_begin
                        var dst_off = tile_base + i * VNNI_K_STEP
                        memcpy(
                            dest=UnsafePointer[Byte, MutAnyOrigin](
                                unsafe_from_address=Int(dst) + dst_off),
                            src=UnsafePointer[Byte, MutAnyOrigin](
                                unsafe_from_address=Int(src) + src_off),
                            count=VNNI_K_STEP,
                        )
                    comptime dword_stride = VNNI_K_STEP // VNNI_BLK
                    var t0 = (dst + tile_base).bitcast[Int32]()
                    var t1 = (dst + tile_base + VNNI_TILE_N * VNNI_K_STEP).bitcast[Int32]()
                    transpose_generic[DType.int32, 16](
                        t0, dword_stride, t0, VNNI_TILE_N, scratch)
                    transpose_generic[DType.int32, 16](
                        t1, dword_stride, t1, VNNI_TILE_N, scratch)


@no_inline
def fill_u8(p: UnsafePointer[UInt8, MutAnyOrigin], count: Int, seed0: UInt32):
    var seed = seed0
    for i in range(count):
        seed = seed * 1664525 + 1013904223
        p[i] = UInt8(seed >> 24)


@no_inline
def checksum_u8(p: UnsafePointer[UInt8, MutAnyOrigin], count: Int) -> UInt64:
    var total = UInt64(0)
    var step = count // 2048
    if step < 1:
        step = 1
    var i = 0
    while i < count:
        total += UInt64(p[i])
        i += step
    return total


@no_inline
def compare_u8(
    a: UnsafePointer[UInt8, MutAnyOrigin],
    b: UnsafePointer[UInt8, MutAnyOrigin],
    count: Int,
) -> Int:
    for i in range(count):
        if a[i] != b[i]:
            return i
    return -1


@no_inline
def reference_colsum(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int, cols: Int, block_cols: Int,
    dst: UnsafePointer[Float32, MutAnyOrigin],
    row_major: Bool,
):
    """Naive scalar block colsum over signed int8 interpretation of src."""
    var num_blocks = cols // block_cols
    var src_i8 = src.bitcast[Int8]()
    for r in range(rows):
        for b in range(num_blocks):
            var total = 0
            var base = b * block_cols
            for c in range(block_cols):
                total += Int(src_i8[r * cols + base + c])
            var v = Float32(total)
            if row_major:
                dst[r * num_blocks + b] = v
            else:
                dst[b * rows + r] = v


@no_inline
def compare_f32(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) -> Int:
    for i in range(count):
        if a[i] != b[i]:
            return i
    return -1


@no_inline
def run_legacy(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    scratch_full: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    iters: Int,
) -> UInt64:
    """Legacy: full-matrix memcpy into scratch, then write-then-reread pack."""
    for i in range(iters):
        src[(i * 4099) % BYTES] = UInt8(i & 0xFF)
        memcpy(dest=scratch_full, src=src, count=BYTES)
        pack_vnni_legacy(scratch_full, dst, ROWS, COLS)
    keep(dst)
    return checksum_u8(dst, BYTES)


@no_inline
def run_fused(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    scratch_small: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    iters: Int,
) -> UInt64:
    """New: streaming per-n_block scratch with fused colsum during the copy."""
    for i in range(iters):
        src[(i * 4099) % BYTES] = UInt8(i & 0xFF)
        pack_and_colsum_vnni(
            src, dst, scratch_small,
            ROWS, COLS, BLOCK_COLS,
            colsum, True,
        )
    keep(dst)
    keep(colsum)
    return checksum_u8(dst, BYTES)


@no_inline
def report_case(label: String, elapsed_ns: Int, iters: Int, chk: UInt64):
    print("  ", label)
    print("    elapsed_ns:", elapsed_ns)
    print("    ns/iter:  ", elapsed_ns // iters)
    print("    checksum: ", chk)


def main():
    print("benchmark_vnni_pack_direct")
    print("  rows:", ROWS, "cols:", COLS, "bytes/iter:", BYTES)
    print("  block_cols:", BLOCK_COLS, "num_blocks:", NUM_BLOCKS)
    print("  warmup:", WARMUP, "iters:", ITERS)
    print("  simd_width[int8]:", simd_width_of[DType.int8]())
    print("  legacy_scratch:", BYTES, "bytes")
    print("  fused_scratch:", required_pack_scratch_bytes(ROWS, COLS), "bytes")

    var src = alloc[UInt8](BYTES)
    var scratch_full = alloc[UInt8](BYTES)
    var scratch_small = alloc[UInt8](required_pack_scratch_bytes(ROWS, COLS))
    var legacy_dst = alloc[UInt8](BYTES)
    var fused_dst = alloc[UInt8](BYTES)
    var fused_colsum = alloc[Float32](ROWS * NUM_BLOCKS)
    var ref_colsum = alloc[Float32](ROWS * NUM_BLOCKS)

    fill_u8(src, BYTES, UInt32(0x12345678))
    fill_u8(legacy_dst, BYTES, UInt32(0xCAFEBABE))
    fill_u8(fused_dst, BYTES, UInt32(0xDEADBEEF))

    memcpy(dest=scratch_full, src=src, count=BYTES)
    pack_vnni_legacy(scratch_full, legacy_dst, ROWS, COLS)
    pack_and_colsum_vnni(
        src, fused_dst, scratch_small,
        ROWS, COLS, BLOCK_COLS,
        fused_colsum, True,
    )
    reference_colsum(src, ROWS, COLS, BLOCK_COLS, ref_colsum, True)

    var mismatch = compare_u8(legacy_dst, fused_dst, BYTES)
    if mismatch >= 0:
        print("packed-byte mismatch at byte", mismatch,
              "legacy=", legacy_dst[mismatch],
              "fused=", fused_dst[mismatch])
        return

    var colsum_mismatch = compare_f32(fused_colsum, ref_colsum, ROWS * NUM_BLOCKS)
    if colsum_mismatch >= 0:
        print("colsum mismatch at idx", colsum_mismatch,
              "fused=", fused_colsum[colsum_mismatch],
              "ref=", ref_colsum[colsum_mismatch])
        return
    print("  validation: packed bytes and colsums match reference")

    comptime if BENCH_LEGACY:
        print("  timed kernel: LEGACY")
        _ = run_legacy(src, scratch_full, legacy_dst, WARMUP)
        fill_u8(src, BYTES, UInt32(0x12345678))
        var t0 = perf_counter_ns()
        var chk_legacy = run_legacy(src, scratch_full, legacy_dst, ITERS)
        var t1 = perf_counter_ns()
        report_case("legacy memcpy + pack_vnni (write-then-reread)",
            Int(t1 - t0), ITERS, chk_legacy)
    else:
        print("  timed kernel: FUSED")
        _ = run_fused(src, scratch_small, fused_dst, fused_colsum, WARMUP)
        fill_u8(src, BYTES, UInt32(0x12345678))
        var t2 = perf_counter_ns()
        var chk_fused = run_fused(src, scratch_small, fused_dst, fused_colsum, ITERS)
        var t3 = perf_counter_ns()
        report_case("fused streaming pack + colsum",
            Int(t3 - t2), ITERS, chk_fused)

    src.free()
    scratch_full.free()
    scratch_small.free()
    legacy_dst.free()
    fused_dst.free()
    fused_colsum.free()
    ref_colsum.free()
