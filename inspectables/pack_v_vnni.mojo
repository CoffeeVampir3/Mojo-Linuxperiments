"""Assembly inspection for V VNNI interleave — shift/OR vs SIMD interleave."""

from std.memory.unsafe_pointer import alloc
from std.benchmark import keep


comptime TILE_N = 16


@no_inline
def pack_v_shift_or(
    v_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    pos_off: Int, dim_off: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Current kernel: cast to u32, shift, OR to build VNNI dwords."""
    var r0 = (v_base + (pos_off + 0) * head_dim + dim_off).load[width=TILE_N]()
    var r1 = (v_base + (pos_off + 1) * head_dim + dim_off).load[width=TILE_N]()
    var r2 = (v_base + (pos_off + 2) * head_dim + dim_off).load[width=TILE_N]()
    var r3 = (v_base + (pos_off + 3) * head_dim + dim_off).load[width=TILE_N]()
    (dst).bitcast[Scalar[DType.uint32]]().store[width=TILE_N](
        r0.cast[DType.uint32]()
        | (r1.cast[DType.uint32]() << 8)
        | (r2.cast[DType.uint32]() << 16)
        | (r3.cast[DType.uint32]() << 24))


@no_inline
def pack_v_interleave(
    v_base: UnsafePointer[UInt8, MutAnyOrigin], head_dim: Int,
    pos_off: Int, dim_off: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """SIMD interleave: 3 interleave ops → VNNI byte order."""
    var r0 = (v_base + (pos_off + 0) * head_dim + dim_off).load[width=TILE_N]()
    var r1 = (v_base + (pos_off + 1) * head_dim + dim_off).load[width=TILE_N]()
    var r2 = (v_base + (pos_off + 2) * head_dim + dim_off).load[width=TILE_N]()
    var r3 = (v_base + (pos_off + 3) * head_dim + dim_off).load[width=TILE_N]()
    (dst).store(r0.interleave(r2).interleave(r1.interleave(r3)))


def main():
    comptime HD = 128
    var v_data = alloc[UInt8](8 * HD)
    var dst1 = alloc[UInt8](64)
    var dst2 = alloc[UInt8](64)

    for i in range(8 * HD):
        v_data[i] = UInt8(i % 256)

    pack_v_shift_or(v_data, HD, 0, 0, dst1)
    pack_v_interleave(v_data, HD, 0, 0, dst2)

    # Verify they produce the same result
    var ok = True
    for i in range(64):
        if dst1[i] != dst2[i]:
            ok = False
            print("MISMATCH at", i, ":", dst1[i], "vs", dst2[i])

    if ok:
        print("MATCH — both produce identical VNNI layout")

    keep(dst1[0])
    keep(dst2[0])

    v_data.free()
    dst1.free()
    dst2.free()
