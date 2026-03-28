"""Proof: VNNI 6D packing produces correct matmul results via vpdpbusd.

1. Scalar reference matmul on row-major weights (ground truth)
2. vpdpbusd matmul on 6D-packed weights
3. Exact integer comparison — if they match, the packing is correct.
"""

from std.memory import UnsafePointer, memcpy, bitcast
from std.memory.unsafe_pointer import alloc
from std.collections import InlineArray
from std.sys.intrinsics import llvm_intrinsic

from simd_math.matrixops import transpose_generic


# =============================================================================
# vpdpbusd intrinsic
#
# LLVM expects (i32x16 acc, u8x64 a, s8x64 b). The public interface takes
# uint8 activations and int8 weights directly. The i32 broadcast for the
# activation dword is the caller's job — bitcast there, not here.
# =============================================================================


@always_inline
def vpdpbusd_512(
    acc: SIMD[DType.int32, 16],
    a: SIMD[DType.uint8, 64],
    b: SIMD[DType.int8, 64],
) -> SIMD[DType.int32, 16]:
    """Acc[i] += dot(a.u8[4i:4i+4], b.s8[4i:4i+4]) per lane."""
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512",
        SIMD[DType.int32, 16],
    ](acc, a, b)


# =============================================================================
# Packing constants
# =============================================================================

comptime TILE_N = 16
comptime VNNI_BLK = 4
comptime N_STEP = 32
comptime K_STEP = 64
comptime L2_TARGET = 256 * 1024


@always_inline
def compute_n_block(n: Int, k: Int) -> Int:
    var max_n = L2_TARGET // k
    var n_block = (max_n // N_STEP) * N_STEP
    if n_block >= n:
        return n
    if n_block >= N_STEP:
        return n_block
    return N_STEP


# =============================================================================
# 6D VNNI packer
#
# Int8 in, int8 out. The dword transpose is an internal detail —
# bitcast to i32 pointers just for the transpose call.
# =============================================================================


def pack_vnni_6d(
    src: UnsafePointer[Int8, MutAnyOrigin],
    dst: UnsafePointer[Int8, MutAnyOrigin],
    N: Int, K: Int,
    n_block: Int, k_block: Int,
):
    var scratch = InlineArray[SIMD[DType.int32, 16], 16](uninitialized=True)

    for n_block_begin in range(0, N, n_block):
        var n_block_size = min(n_block, N - n_block_begin)
        for k_block_begin in range(0, K, k_block):
            var k_block_size = min(k_block, K - k_block_begin)
            for n_begin in range(0, n_block_size, N_STEP):
                for k_begin in range(0, k_block_size, K_STEP):
                    var tile_base = (
                        n_block_begin * K
                        + k_block_begin * n_block_size
                        + n_begin * k_block_size
                        + k_begin * N_STEP
                    )
                    for i in range(N_STEP):
                        var src_off = (n_block_begin + n_begin + i) * K + k_block_begin + k_begin
                        var dst_off = tile_base + i * K_STEP
                        memcpy(
                            dest=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=Int(dst) + dst_off),
                            src=UnsafePointer[Byte, MutAnyOrigin](unsafe_from_address=Int(src) + src_off),
                            count=K_STEP,
                        )
                    # Dword transpose: bitcast to i32 ptrs just for the transpose
                    comptime dword_stride = K_STEP // 4
                    var t0 = (dst + tile_base).bitcast[Int32]()
                    var t1 = (dst + tile_base + TILE_N * K_STEP).bitcast[Int32]()
                    transpose_generic[DType.int32, 16](t0, dword_stride, t0, TILE_N, scratch)
                    transpose_generic[DType.int32, 16](t1, dword_stride, t1, TILE_N, scratch)


# =============================================================================
# Scalar reference: C[m,n] = sum_k u8(A[m,k]) * s8(B[n,k])
# =============================================================================


def matmul_reference(
    a: UnsafePointer[UInt8, MutAnyOrigin],
    b: UnsafePointer[Int8, MutAnyOrigin],
    c: UnsafePointer[Int32, MutAnyOrigin],
    M: Int, N: Int, K: Int,
):
    for m in range(M):
        for n in range(N):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(a[m * K + k]) * Int32(b[n * K + k])
            c[m * N + n] = acc


# =============================================================================
# vpdpbusd matmul on 6D-packed weights
#
# A is uint8 [M, K] row-major. B is int8 [N, K] VNNI-packed.
# For each TILE_N=16 output channels: broadcast 4 activation bytes,
# load 16 packed weight dwords, vpdpbusd.
# The i32 reinterpretation is only for the broadcast and tile load.
# =============================================================================


def matmul_vnni(
    a: UnsafePointer[UInt8, MutAnyOrigin],
    b_packed: UnsafePointer[Int8, MutAnyOrigin],
    c: UnsafePointer[Int32, MutAnyOrigin],
    M: Int, N: Int, K: Int,
    n_block: Int, k_block: Int,
):
    for m in range(M):
        for n_block_begin in range(0, N, n_block):
            var n_block_size = min(n_block, N - n_block_begin)
            for k_block_begin in range(0, K, k_block):
                var k_block_size = min(k_block, K - k_block_begin)
                for n_begin in range(0, n_block_size, N_STEP):
                    for n_sub in range(2):
                        var n_global = n_block_begin + n_begin + n_sub * TILE_N
                        var acc = SIMD[DType.int32, 16](0)

                        if k_block_begin > 0:
                            acc = (c + m * N + n_global).load[width=16]()

                        for k_begin in range(0, k_block_size, K_STEP):
                            var tile_base = (
                                n_block_begin * K
                                + k_block_begin * n_block_size
                                + n_begin * k_block_size
                                + k_begin * N_STEP
                                + n_sub * TILE_N * K_STEP
                            )

                            for kv in range(K_STEP // VNNI_BLK):
                                # Broadcast: load 4 activation bytes as i32, splat to 16 lanes,
                                # bitcast back to u8x64 for vpdpbusd
                                var k_abs = k_block_begin + k_begin + kv * VNNI_BLK
                                var a_dword = (a + m * K + k_abs).bitcast[Int32]()[]
                                var a_broadcast = bitcast[DType.uint8, 64](SIMD[DType.int32, 16](a_dword))

                                # Load 16 packed weight dwords, bitcast to s8x64
                                var b_ptr = b_packed + tile_base + kv * TILE_N * VNNI_BLK
                                var b_tile = bitcast[DType.int8, 64](b_ptr.bitcast[Int32]().load[width=16]())

                                acc = vpdpbusd_512(acc, a_broadcast, b_tile)

                        (c + m * N + n_global).store(acc)


# =============================================================================
# Test
# =============================================================================


def test_matmul(M: Int, N: Int, K: Int):
    var n_block = compute_n_block(N, K)
    var k_block = K

    print(
        "\n--- M=" + String(M) + " N=" + String(N) + " K=" + String(K)
        + " N_BLOCK=" + String(n_block) + " ---"
    )

    var a = alloc[UInt8](M * K)
    var b = alloc[Int8](N * K)
    var b_packed = alloc[Int8](N * K)
    var c_expected = alloc[Int32](M * N)
    var c_vnni = alloc[Int32](M * N)

    for i in range(M * K):
        a[i] = UInt8(i % 13 + 1)
    for i in range(N * K):
        b[i] = Int8((i % 17) - 8)
    for i in range(M * N):
        c_expected[i] = Int32(0)
        c_vnni[i] = Int32(0)

    pack_vnni_6d(b, b_packed, N, K, n_block, k_block)
    matmul_reference(a, b, c_expected, M, N, K)
    matmul_vnni(a, b_packed, c_vnni, M, N, K, n_block, k_block)

    var mismatches = 0
    for i in range(M * N):
        if c_expected[i] != c_vnni[i]:
            if mismatches < 5:
                print(
                    "  MISMATCH (" + String(i // N) + "," + String(i % N) + ")"
                    + " expected=" + String(Int(c_expected[i]))
                    + " got=" + String(Int(c_vnni[i]))
                )
            mismatches += 1

    if mismatches == 0:
        var nonzero = 0
        for i in range(M * N):
            if c_expected[i] != 0:
                nonzero += 1
        print("PASS (" + String(nonzero) + "/" + String(M * N) + " nonzero)")
    else:
        print("FAIL: " + String(mismatches) + " mismatches")

    a.free()
    b.free()
    b_packed.free()
    c_expected.free()
    c_vnni.free()


def main():
    print("=== VNNI 6D packing proof via vpdpbusd matmul ===")
    test_matmul(1, 576, 576)
    test_matmul(4, 576, 576)
    test_matmul(1, 192, 576)
    test_matmul(1, 1536, 576)
    test_matmul(1, 576, 1536)
    test_matmul(4, 576, 1536)
    test_matmul(1, 32, 64)
    test_matmul(2, 32, 64)
