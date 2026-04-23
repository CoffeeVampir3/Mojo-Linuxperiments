"""Benchmark current VNNI packing against direct source-to-packed packing.

The current runtime path is:
  row-major arena -> scratch copy -> pack_vnni(scratch, arena)

`pack_vnni` writes each 32x64 panel to the destination in row-major order, then
immediately reloads that destination panel as int32 rows, transposes it, and
stores the final VNNI layout back to the same destination address.

This benchmark compares that against a direct packer that reads row-major source
rows and stores the transposed VNNI panel directly to the final destination.

Run:
  pixi run mojo build -I . benchmark_vnni_pack_direct.mojo
  ./benchmark_vnni_pack_direct

Remote perf:
  ./remote_perf.fish benchmark_vnni_pack_direct.mojo
"""

from std.benchmark import keep
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns

from kernels.vnni import (
    VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK,
    compute_n_block, pack_vnni,
)
from simd_math.matrixops import transpose_generic


comptime ROWS = 2048
comptime COLS = 3072
comptime BYTES = ROWS * COLS
comptime WARMUP = 8
comptime ITERS = 1024


@always_inline
def pack_vnni_direct(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    """Direct row-major source -> final VNNI destination.

    Source and destination must not overlap. The destination is never used as a
    pre-transpose staging tile.
    """
    debug_assert(cols % VNNI_K_STEP == 0,
        "pack_vnni_direct: K must be a multiple of VNNI_K_STEP")
    debug_assert(rows % VNNI_N_STEP == 0,
        "pack_vnni_direct: N must be a multiple of VNNI_N_STEP")

    var n_block = compute_n_block(rows, cols)
    var k_block = cols
    var scratch = InlineArray[SIMD[DType.int32, 16], 16](uninitialized=True)
    var src_i32_stride = cols // VNNI_BLK

    for n_block_begin in range(0, rows, n_block):
        var n_block_size = min(n_block, rows - n_block_begin)

        for k_block_begin in range(0, cols, k_block):
            var k_block_size = min(k_block, cols - k_block_begin)

            for n_begin in range(0, n_block_size, VNNI_N_STEP):
                for k_begin in range(0, k_block_size, VNNI_K_STEP):
                    var tile_base = (
                        n_block_begin * cols
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * VNNI_N_STEP
                    )

                    var row0 = n_block_begin + n_begin
                    var src0_off = row0 * cols + k_block_begin + k_begin
                    var dst0_off = tile_base
                    transpose_generic[DType.int32, 16](
                        (src + src0_off).bitcast[Int32](),
                        src_i32_stride,
                        (dst + dst0_off).bitcast[Int32](),
                        VNNI_TILE_N,
                        scratch,
                    )

                    var row1 = row0 + VNNI_TILE_N
                    var src1_off = row1 * cols + k_block_begin + k_begin
                    var dst1_off = tile_base + VNNI_TILE_N * VNNI_K_STEP
                    transpose_generic[DType.int32, 16](
                        (src + src1_off).bitcast[Int32](),
                        src_i32_stride,
                        (dst + dst1_off).bitcast[Int32](),
                        VNNI_TILE_N,
                        scratch,
                    )


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
def run_current(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    iters: Int,
) -> UInt64:
    for i in range(iters):
        src[(i * 4099) % BYTES] = UInt8(i & 0xFF)
        memcpy(dest=scratch, src=src, count=BYTES)
        pack_vnni(scratch, dst, ROWS, COLS)
    keep(dst)
    return checksum_u8(dst, BYTES)


@no_inline
def run_direct_with_scratch(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    iters: Int,
) -> UInt64:
    for i in range(iters):
        src[(i * 4099) % BYTES] = UInt8(i & 0xFF)
        memcpy(dest=scratch, src=src, count=BYTES)
        pack_vnni_direct(scratch, dst, ROWS, COLS)
    keep(dst)
    return checksum_u8(dst, BYTES)


@no_inline
def run_direct_no_scratch(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    iters: Int,
) -> UInt64:
    for i in range(iters):
        src[(i * 4099) % BYTES] = UInt8(i & 0xFF)
        pack_vnni_direct(src, dst, ROWS, COLS)
    keep(dst)
    return checksum_u8(dst, BYTES)


@no_inline
def report_case(label: String, elapsed_ns: Int, iters: Int, chk: UInt64):
    print("  ", label)
    print("    elapsed_ns:", elapsed_ns)
    print("    ns/iter:", elapsed_ns // iters)
    print("    checksum:", chk)


def main():
    print("benchmark_vnni_pack_direct")
    print("  rows:", ROWS, "cols:", COLS, "bytes/iter:", BYTES)
    print("  warmup:", WARMUP, "iters:", ITERS)

    var src = alloc[UInt8](BYTES)
    var scratch = alloc[UInt8](BYTES)
    var current_dst = alloc[UInt8](BYTES)
    var direct_dst = alloc[UInt8](BYTES)
    var noscratch_dst = alloc[UInt8](BYTES)

    fill_u8(src, BYTES, UInt32(0x12345678))
    fill_u8(scratch, BYTES, UInt32(0x87654321))
    fill_u8(current_dst, BYTES, UInt32(0xCAFEBABE))
    fill_u8(direct_dst, BYTES, UInt32(0xCAFEBABE))
    fill_u8(noscratch_dst, BYTES, UInt32(0xCAFEBABE))

    memcpy(dest=scratch, src=src, count=BYTES)
    pack_vnni(scratch, current_dst, ROWS, COLS)
    pack_vnni_direct(scratch, direct_dst, ROWS, COLS)
    var mismatch = compare_u8(current_dst, direct_dst, BYTES)
    if mismatch >= 0:
        print("direct_with_scratch mismatch at byte", mismatch,
              "current=", current_dst[mismatch],
              "direct=", direct_dst[mismatch])
        return
    pack_vnni_direct(src, noscratch_dst, ROWS, COLS)
    mismatch = compare_u8(current_dst, noscratch_dst, BYTES)
    if mismatch >= 0:
        print("direct_no_scratch mismatch at byte", mismatch,
              "current=", current_dst[mismatch],
              "direct=", noscratch_dst[mismatch])
        return
    print("  validation: outputs match")

    _ = run_current(src, scratch, current_dst, WARMUP)
    _ = run_direct_with_scratch(src, scratch, direct_dst, WARMUP)
    _ = run_direct_no_scratch(src, noscratch_dst, WARMUP)

    fill_u8(src, BYTES, UInt32(0x12345678))
    var t0 = perf_counter_ns()
    var chk_current = run_current(src, scratch, current_dst, ITERS)
    var t1 = perf_counter_ns()

    fill_u8(src, BYTES, UInt32(0x12345678))
    var t2 = perf_counter_ns()
    var chk_direct = run_direct_with_scratch(src, scratch, direct_dst, ITERS)
    var t3 = perf_counter_ns()

    fill_u8(src, BYTES, UInt32(0x12345678))
    var t4 = perf_counter_ns()
    var chk_noscratch = run_direct_no_scratch(src, noscratch_dst, ITERS)
    var t5 = perf_counter_ns()

    report_case("current memcpy + pack_vnni", Int(t1 - t0), ITERS, chk_current)
    report_case("direct memcpy + source-to-packed", Int(t3 - t2), ITERS, chk_direct)
    report_case("direct source-to-packed only", Int(t5 - t4), ITERS, chk_noscratch)

    var final_mismatch = compare_u8(current_dst, direct_dst, BYTES)
    if final_mismatch >= 0:
        print("final direct_with_scratch mismatch at byte", final_mismatch)
    final_mismatch = compare_u8(current_dst, noscratch_dst, BYTES)
    if final_mismatch >= 0:
        print("final direct_no_scratch mismatch at byte", final_mismatch)

    src.free()
    scratch.free()
    current_dst.free()
    direct_dst.free()
    noscratch_dst.free()
