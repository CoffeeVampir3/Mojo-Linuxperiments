from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of
from std.sys import llvm_intrinsic


comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4

comptime WIDTH = simd_width_of[DType.int32]()
comptime PASSES_PER_SUBTILE = VNNI_TILE_N // WIDTH
comptime BYTES_PER_PASS = WIDTH * VNNI_BLK
comptime ACC_COUNT = VNNI_N_STEP // WIDTH
comptime PACKED_BYTES = VNNI_N_STEP * VNNI_K_STEP


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * VNNI_BLK],
    b: SIMD[DType.int8, width * VNNI_BLK],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def act_broadcast_bytes(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.uint8, WIDTH * VNNI_BLK]:
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=VNNI_BLK]() ^ SIMD[DType.uint8, VNNI_BLK](0x80)
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    var b32 = b16.join(b16)
    var b64 = b32.join(b32)
    return b64.slice[WIDTH * VNNI_BLK]()


@always_inline
def dot_current(
    acc: SIMD[DType.int32, WIDTH],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, WIDTH]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width=WIDTH * VNNI_BLK]()
    return vpdpbusd[WIDTH](acc, act_broadcast_bytes(act_row, k_pos), w)


@always_inline
def dot_hoisted(
    acc: SIMD[DType.int32, WIDTH],
    act_bytes: SIMD[DType.uint8, WIDTH * VNNI_BLK],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
) -> SIMD[DType.int32, WIDTH]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width=WIDTH * VNNI_BLK]()
    return vpdpbusd[WIDTH](acc, act_bytes, w)


@no_inline
def current_gemv_tile(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
):
    var acc_buf = InlineArray[SIMD[DType.int32, WIDTH], ACC_COUNT](
        fill=SIMD[DType.int32, WIDTH](0))
    var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, WIDTH]]()
    var packed_off = 0

    for dc in range(VNNI_K_STEP // VNNI_BLK):
        var k_pos = dc * VNNI_BLK
        for p in range(PASSES_PER_SUBTILE):
            acc[p] = dot_current(
                acc[p], act_row,
                wpacked + packed_off + p * BYTES_PER_PASS,
                k_pos,
            )
        packed_off += VNNI_TILE_N * VNNI_BLK

        for p in range(PASSES_PER_SUBTILE):
            acc[PASSES_PER_SUBTILE + p] = dot_current(
                acc[PASSES_PER_SUBTILE + p], act_row,
                wpacked + packed_off + p * BYTES_PER_PASS,
                k_pos,
            )
        packed_off += VNNI_TILE_N * VNNI_BLK

    for a in range(ACC_COUNT):
        (out_ptr + a * WIDTH).store(acc[a])


@no_inline
def hoisted_activation_broadcast_gemv_tile(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
):
    var acc_buf = InlineArray[SIMD[DType.int32, WIDTH], ACC_COUNT](
        fill=SIMD[DType.int32, WIDTH](0))
    var acc = UnsafePointer(to=acc_buf).bitcast[SIMD[DType.int32, WIDTH]]()
    var packed_off = 0

    for dc in range(VNNI_K_STEP // VNNI_BLK):
        var k_pos = dc * VNNI_BLK
        var act_bytes = act_broadcast_bytes(act_row, k_pos)

        for p in range(PASSES_PER_SUBTILE):
            acc[p] = dot_hoisted(
                acc[p], act_bytes,
                wpacked + packed_off + p * BYTES_PER_PASS,
            )
        packed_off += VNNI_TILE_N * VNNI_BLK

        for p in range(PASSES_PER_SUBTILE):
            acc[PASSES_PER_SUBTILE + p] = dot_hoisted(
                acc[PASSES_PER_SUBTILE + p], act_bytes,
                wpacked + packed_off + p * BYTES_PER_PASS,
            )
        packed_off += VNNI_TILE_N * VNNI_BLK

    for a in range(ACC_COUNT):
        (out_ptr + a * WIDTH).store(acc[a])


@always_inline
def init_inputs(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
):
    var act = Int32(7)
    for k in range(VNNI_K_STEP):
        act = (act * 17 + 29) % 101
        act_row[k] = Scalar[DType.int8](Int8(act - 50))

    var wt = Int32(11)
    for i in range(PACKED_BYTES):
        wt = (wt * 31 + 13) % 251
        wpacked[i] = UInt8(wt)


@no_inline
def run_case() -> Int:
    var act_arr = InlineArray[Scalar[DType.int8], VNNI_K_STEP](uninitialized=True)
    var wpacked_arr = InlineArray[UInt8, PACKED_BYTES](uninitialized=True)
    var current_out_arr = InlineArray[Scalar[DType.int32], VNNI_N_STEP](uninitialized=True)
    var hoisted_out_arr = InlineArray[Scalar[DType.int32], VNNI_N_STEP](uninitialized=True)

    var act_row = UnsafePointer(to=act_arr).bitcast[Scalar[DType.int8]]()
    var wpacked = UnsafePointer(to=wpacked_arr).bitcast[UInt8]()
    var current_out = UnsafePointer(to=current_out_arr).bitcast[Scalar[DType.int32]]()
    var hoisted_out = UnsafePointer(to=hoisted_out_arr).bitcast[Scalar[DType.int32]]()

    init_inputs(act_row, wpacked)
    current_gemv_tile(act_row, wpacked, current_out)
    hoisted_activation_broadcast_gemv_tile(act_row, wpacked, hoisted_out)

    var checksum = Int(0)
    for i in range(VNNI_N_STEP):
        var cur = Int(current_out[i])
        var alt = Int(hoisted_out[i])
        checksum += cur
        checksum -= alt
        if cur >= alt:
            checksum += cur - alt
        else:
            checksum += alt - cur
    return checksum


def main():
    keep(run_case())
