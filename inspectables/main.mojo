from std.memory import UnsafePointer
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import CompilationTarget
from std.sys import llvm_intrinsic


comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4
comptime WIDTH = 8
comptime REGS = VNNI_TILE_N // WIDTH
comptime FWHT_BLK = 16
comptime DC_PER_KSTEP = VNNI_K_STEP // VNNI_BLK
comptime DC_PER_FWHT_BLK = FWHT_BLK // VNNI_BLK
comptime SUB_BLKS = VNNI_K_STEP // FWHT_BLK
comptime TILE_BYTES = VNNI_TILE_N * VNNI_BLK
comptime INSERT_TILE = 2 * WIDTH
comptime NUM_BLOCKS = 4
comptime PACKED_BYTES = VNNI_N_STEP * VNNI_K_STEP


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    var w = wpacked.bitcast[Scalar[DType.int8]]().load[width=width * 4]()
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[DType.uint8, 4](0x80)
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    var b32 = b16.join(b16)
    return vpdpbusd[width](acc, b32.slice[width * 4](), w)


@always_inline
def dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[width=width]()
    var result = acc
    result += SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128) * ((wdw << 24) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128) * ((wdw << 16) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128) * ((wdw << 8) >> 24)
    result += SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128) * (wdw >> 24)
    return result


@always_inline
def dot[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return dot_vnni[width](acc, act_row, wpacked, k_pos)
    else:
        return dot_simd[width](acc, act_row, wpacked, k_pos)


@always_inline
def dot_tile_simd_rewrite(
    acc: SIMD[DType.int32, VNNI_TILE_N],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, VNNI_TILE_N]:
    comptime assert REGS == 2, "slice SIMD rewrite assumes one VNNI tile is two WIDTH chunks"
    var result = acc
    comptime for reg in range(REGS):
        comptime lane_offset = reg * WIDTH
        result = result.insert[offset=lane_offset](
            dot[WIDTH](
                result.slice[WIDTH, offset=lane_offset](),
                act_row,
                wpacked + lane_offset * VNNI_BLK,
                k_pos,
            ))
    return result


@no_inline
def phase2_inlinearray_storage(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var f32_t0 = InlineArray[SIMD[DType.float32, WIDTH], REGS](
        fill=SIMD[DType.float32, WIDTH](0))
    var f32_t1 = InlineArray[SIMD[DType.float32, WIDTH], REGS](
        fill=SIMD[DType.float32, WIDTH](0))
    var t0 = InlineArray[SIMD[DType.int32, WIDTH], SUB_BLKS * REGS](
        fill=SIMD[DType.int32, WIDTH](0))
    var t1 = InlineArray[SIMD[DType.int32, WIDTH], SUB_BLKS * REGS](
        fill=SIMD[DType.int32, WIDTH](0))

    comptime for dc in range(DC_PER_KSTEP):
        comptime sb = dc // DC_PER_FWHT_BLK
        comptime k_pos = dc * VNNI_BLK
        comptime base = sb * REGS
        comptime off = dc * TILE_BYTES
        t0[base] = dot[WIDTH](t0[base], act_row, wpacked + off, k_pos)
        t0[base + 1] = dot[WIDTH](t0[base + 1], act_row, wpacked + off + WIDTH * VNNI_BLK, k_pos)

    comptime for dc in range(DC_PER_KSTEP):
        comptime sb = dc // DC_PER_FWHT_BLK
        comptime k_pos = dc * VNNI_BLK
        comptime base = sb * REGS
        comptime off = DC_PER_KSTEP * TILE_BYTES + dc * TILE_BYTES
        t1[base] = dot[WIDTH](t1[base], act_row, wpacked + off, k_pos)
        t1[base + 1] = dot[WIDTH](t1[base + 1], act_row, wpacked + off + WIDTH * VNNI_BLK, k_pos)

    comptime for sb in range(SUB_BLKS):
        var dq = block_scales[sb] / 127.0
        comptime base = sb * REGS
        comptime lane0 = 0
        comptime lane1 = WIDTH
        var cs00 = (block_colsums + sb * VNNI_N_STEP + lane0).load[width=WIDTH]()
        var cs01 = (block_colsums + sb * VNNI_N_STEP + lane1).load[width=WIDTH]()
        var cs10 = (block_colsums + sb * VNNI_N_STEP + VNNI_TILE_N + lane0).load[width=WIDTH]()
        var cs11 = (block_colsums + sb * VNNI_N_STEP + VNNI_TILE_N + lane1).load[width=WIDTH]()
        f32_t0[0] += (t0[base].cast[DType.float32]() - 128.0 * cs00) * dq
        f32_t0[1] += (t0[base + 1].cast[DType.float32]() - 128.0 * cs01) * dq
        f32_t1[0] += (t1[base].cast[DType.float32]() - 128.0 * cs10) * dq
        f32_t1[1] += (t1[base + 1].cast[DType.float32]() - 128.0 * cs11) * dq

    dst.store(f32_t0[0] * wsc.load[width=WIDTH]())
    (dst + WIDTH).store(f32_t0[1] * (wsc + WIDTH).load[width=WIDTH]())
    (dst + VNNI_TILE_N).store(f32_t1[0] * (wsc + VNNI_TILE_N).load[width=WIDTH]())
    (dst + VNNI_TILE_N + WIDTH).store(
        f32_t1[1] * (wsc + VNNI_TILE_N + WIDTH).load[width=WIDTH]())


@no_inline
def phase2_slice_simd_rewrite(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var f32_t0 = SIMD[DType.float32, VNNI_TILE_N](0)
    var f32_t1 = SIMD[DType.float32, VNNI_TILE_N](0)
    var t0 = InlineArray[SIMD[DType.int32, VNNI_TILE_N], SUB_BLKS](
        fill=SIMD[DType.int32, VNNI_TILE_N](0))
    var t1 = InlineArray[SIMD[DType.int32, VNNI_TILE_N], SUB_BLKS](
        fill=SIMD[DType.int32, VNNI_TILE_N](0))

    comptime for dc in range(DC_PER_KSTEP):
        comptime sb = dc // DC_PER_FWHT_BLK
        comptime k_pos = dc * VNNI_BLK
        comptime off = dc * TILE_BYTES
        t0[sb] = dot_tile_simd_rewrite(t0[sb], act_row, wpacked + off, k_pos)

    comptime for dc in range(DC_PER_KSTEP):
        comptime sb = dc // DC_PER_FWHT_BLK
        comptime k_pos = dc * VNNI_BLK
        comptime off = DC_PER_KSTEP * TILE_BYTES + dc * TILE_BYTES
        t1[sb] = dot_tile_simd_rewrite(t1[sb], act_row, wpacked + off, k_pos)

    comptime for sb in range(SUB_BLKS):
        var dq = block_scales[sb] / 127.0
        var cs0 = (block_colsums + sb * VNNI_N_STEP).load[width=VNNI_TILE_N]()
        var cs1 = (block_colsums + sb * VNNI_N_STEP + VNNI_TILE_N).load[width=VNNI_TILE_N]()
        f32_t0 += (t0[sb].cast[DType.float32]() - 128.0 * cs0) * dq
        f32_t1 += (t1[sb].cast[DType.float32]() - 128.0 * cs1) * dq

    dst.store(f32_t0 * wsc.load[width=VNNI_TILE_N]())
    (dst + VNNI_TILE_N).store(
        f32_t1 * (wsc + VNNI_TILE_N).load[width=VNNI_TILE_N]())


@no_inline
def phase2_slice_insert_storage(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    phase2_slice_simd_rewrite(
        act_row,
        wpacked,
        block_scales,
        wsc,
        block_colsums,
        dst,
    )


def run_inline() -> Float32:
    var act = InlineArray[Scalar[DType.int8], VNNI_K_STEP](fill=Scalar[DType.int8](0))
    var packed = InlineArray[UInt8, PACKED_BYTES](fill=UInt8(0))
    var block_scales = InlineArray[Float32, NUM_BLOCKS](fill=Float32(0))
    var wsc = InlineArray[Float32, VNNI_N_STEP](fill=Float32(0))
    var block_colsums = InlineArray[Float32, NUM_BLOCKS * VNNI_N_STEP](fill=Float32(0))
    var dst = InlineArray[Float32, VNNI_N_STEP](fill=Float32(0))

    for i in range(VNNI_K_STEP):
        act[i] = Scalar[DType.int8]((i * 7 + 13) % 127 - 63)
    for i in range(PACKED_BYTES):
        packed[i] = UInt8((i * 5 + 3) % 251)
    for i in range(NUM_BLOCKS):
        block_scales[i] = 0.5 + Float32(i) * 0.03125
    for i in range(VNNI_N_STEP):
        wsc[i] = 0.01 + Float32(i % 7) * 0.005
    for i in range(NUM_BLOCKS * VNNI_N_STEP):
        block_colsums[i] = Float32((i * 3) % 23) - 11.0

    phase2_inlinearray_storage(
        UnsafePointer(to=act).bitcast[Scalar[DType.int8]](),
        UnsafePointer(to=packed).bitcast[UInt8](),
        UnsafePointer(to=block_scales).bitcast[Float32](),
        UnsafePointer(to=wsc).bitcast[Float32](),
        UnsafePointer(to=block_colsums).bitcast[Float32](),
        UnsafePointer(to=dst).bitcast[Float32](),
    )

    var sum = Float32(0)
    var dp = UnsafePointer(to=dst).bitcast[Float32]()
    for i in range(VNNI_N_STEP):
        sum += dp[i]
    return sum


def run_slice_insert() -> Float32:
    var act = InlineArray[Scalar[DType.int8], VNNI_K_STEP](fill=Scalar[DType.int8](0))
    var packed = InlineArray[UInt8, PACKED_BYTES](fill=UInt8(0))
    var block_scales = InlineArray[Float32, NUM_BLOCKS](fill=Float32(0))
    var wsc = InlineArray[Float32, VNNI_N_STEP](fill=Float32(0))
    var block_colsums = InlineArray[Float32, NUM_BLOCKS * VNNI_N_STEP](fill=Float32(0))
    var dst = InlineArray[Float32, VNNI_N_STEP](fill=Float32(0))

    for i in range(VNNI_K_STEP):
        act[i] = Scalar[DType.int8]((i * 7 + 13) % 127 - 63)
    for i in range(PACKED_BYTES):
        packed[i] = UInt8((i * 5 + 3) % 251)
    for i in range(NUM_BLOCKS):
        block_scales[i] = 0.5 + Float32(i) * 0.03125
    for i in range(VNNI_N_STEP):
        wsc[i] = 0.01 + Float32(i % 7) * 0.005
    for i in range(NUM_BLOCKS * VNNI_N_STEP):
        block_colsums[i] = Float32((i * 3) % 23) - 11.0

    phase2_slice_insert_storage(
        UnsafePointer(to=act).bitcast[Scalar[DType.int8]](),
        UnsafePointer(to=packed).bitcast[UInt8](),
        UnsafePointer(to=block_scales).bitcast[Float32](),
        UnsafePointer(to=wsc).bitcast[Float32](),
        UnsafePointer(to=block_colsums).bitcast[Float32](),
        UnsafePointer(to=dst).bitcast[Float32](),
    )

    var sum = Float32(0)
    var dp = UnsafePointer(to=dst).bitcast[Float32]()
    for i in range(VNNI_N_STEP):
        sum += dp[i]
    return sum


def main():
    comptime assert REGS == 2, "This inspectable is intentionally fixed to WIDTH=8."
    keep(run_inline())
    keep(run_slice_insert())
