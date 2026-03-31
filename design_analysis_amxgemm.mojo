"""Design Analysis: AMX 2-2-4 Int8 GEMM

Validates AMX tile operations by implementing a standalone i8 GEMM
using the 2-2-4 tile layout. Foundation for integrating AMX into
the hadquant attention kernel.

Tile assignment (int8, tdpbssd):
  TMM0, TMM1: A tiles  (16 rows x 64 bytes, row-major i8)
  TMM2, TMM3: B tiles  (16 rows x 64 bytes, VNNI-packed i8)
  TMM4-TMM7:  C tiles  (16 rows x 64 bytes = 16x16 i32)

One atomic 2-2-4 step: C[32,32] += A[32,64] x B[64,32]
  tdpbssd(4, 0, 2)  C[ 0:16,  0:16] += A[ 0:16, :] x B[:,  0:16]
  tdpbssd(5, 0, 3)  C[ 0:16, 16:32] += A[ 0:16, :] x B[:, 16:32]
  tdpbssd(6, 1, 2)  C[16:32,  0:16] += A[16:32, :] x B[:,  0:16]
  tdpbssd(7, 1, 3)  C[16:32, 16:32] += A[16:32, :] x B[:, 16:32]

VNNI packing for B tiles:
  B_vnni[kg, 4*col + b] = B_orig[4*kg + b, col]
  Groups 4 consecutive K values per i32 lane for tdpbssd.

Maps to attention operations:
  Scoring Q[16,128] x K^T[128,64]: M=16, K=128 (2 K-steps), N=64 (2 N-steps)
  V agg   W[16,64]  x V[64,128]:  M=16, K=64  (1 K-step),  N=128 (4 N-steps)

Build and run on AMX-capable system (Sapphire Rapids+).
"""

from std.sys import llvm_intrinsic
from std.memory import UnsafePointer
from std.collections import InlineArray
from std.sys.info import size_of


# ============================================================================
# Constants
# ============================================================================

comptime TILE_M = 16
comptime TILE_K = 64
comptime TILE_N = 16
comptime VNNI_BLK = 4
comptime M_STEP = TILE_M * 2     # 32
comptime N_STEP = TILE_N * 2     # 32
comptime K_STEP = TILE_K         # 64
comptime TILE_BYTES = TILE_M * TILE_K  # 1024


# ============================================================================
# Tile Configuration (64-byte config block for ldtilecfg)
# ============================================================================

@fieldwise_init
struct TileConfig(Movable):
    var palette_id: UInt8
    var start_row: UInt8
    var reserved: InlineArray[UInt8, 14]
    var colsb: InlineArray[UInt16, 16]
    var rows: InlineArray[UInt8, 16]

    def __init__(out self):
        self.palette_id = 0
        self.start_row = 0
        self.reserved = InlineArray[UInt8, 14](fill=0)
        self.colsb = InlineArray[UInt16, 16](fill=0)
        self.rows = InlineArray[UInt8, 16](fill=0)


def make_224_i8_config() -> TileConfig:
    """2-2-4 int8 config: all 8 tiles at 16 rows x 64 col-bytes."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    for i in range(8):
        cfg.rows[i] = 16
        cfg.colsb[i] = 64
    return cfg^


# ============================================================================
# AMX System Init (Linux)
# ============================================================================

def init_intel_amx() -> Bool:
    """Request kernel permission for AMX tile data via arch_prctl."""
    comptime SYS_arch_prctl = 158
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    var result = __mlir_op.`pop.external_call`[
        func = "syscall".value,
        _type = Int64,
    ](Int64(SYS_arch_prctl), Int64(ARCH_REQ_XCOMP_PERM), Int64(XFEATURE_XTILEDATA))
    return result == 0


# ============================================================================
# AMX Intrinsic Wrappers
# ============================================================================

@always_inline
def ldtilecfg(cfg: UnsafePointer[TileConfig, MutAnyOrigin]):
    """Load tile configuration. Expensive (~100+ cycles), call rarely."""
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](cfg)

@always_inline
def tilerelease():
    """Release AMX tile state."""
    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()

@always_inline
def tilezero[tile: Int]():
    """Zero tile register TMM<tile>."""
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tile))

@always_inline
def tileload[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], stride: Int,
):
    """Load tile from memory. stride = byte distance between rows."""
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(tile), ptr, Int64(stride))

@always_inline
def tilestore[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], stride: Int,
):
    """Store tile to memory. stride = byte distance between rows."""
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(tile), ptr, Int64(stride))

@always_inline
def tdpbssd[dst: Int, src_a: Int, src_b: Int]():
    """Tile dot product: i8 x i8 -> i32. TMM<dst> += TMM<src_a> x TMM<src_b>."""
    comptime assert dst >= 0 and dst < 8 and src_a >= 0 and src_a < 8 and src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbssd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))

@always_inline
def tdpbusd[dst: Int, src_a: Int, src_b: Int]():
    """Tile dot product: u8 x i8 -> i32. TMM<dst> += TMM<src_a>(u8) x TMM<src_b>(i8)."""
    comptime assert dst >= 0 and dst < 8 and src_a >= 0 and src_a < 8 and src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbusd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))

@always_inline
def tdpbsud[dst: Int, src_a: Int, src_b: Int]():
    """Tile dot product: i8 x u8 -> i32. TMM<dst> += TMM<src_a>(i8) x TMM<src_b>(u8)."""
    comptime assert dst >= 0 and dst < 8 and src_a >= 0 and src_a < 8 and src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbsud", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


# ============================================================================
# VNNI B-Tile Packing
# ============================================================================

def pack_b_vnni(
    b: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    ldb: Int,
    k_off: Int,
    n_off: Int,
    n_cols: Int,
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
):
    """Pack B[k_off:k_off+64, n_off:n_off+n_cols] into VNNI for one tile.

    B is [K, N] row-major, stride ldb (= N).
    Output: 16 VNNI rows x 64 bytes, stride 64.
    VNNI: dst[kg*64 + col*4 + byte] = B[k_off + 4*kg + byte, n_off + col].
    """
    for kg in range(TILE_K // VNNI_BLK):
        for col in range(n_cols):
            for bx in range(VNNI_BLK):
                dst[kg * 64 + col * 4 + bx] = b[(k_off + 4 * kg + bx) * ldb + n_off + col]
        # Zero-pad unused columns (if n_cols < 16)
        for col in range(n_cols, TILE_N):
            for bx in range(VNNI_BLK):
                dst[kg * 64 + col * 4 + bx] = Scalar[DType.int8](0)


# ============================================================================
# AMX 2-2-4 GEMM
# ============================================================================

def amx_gemm_224(
    a: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    c: UnsafePointer[Int32, MutAnyOrigin],
    m: Int, k: Int, n: Int,
):
    """C[M,N] = A[M,K] x B[K,N] using AMX 2-2-4 (tdpbssd, i8xi8->i32).

    Requires: K % 64 == 0, N % 32 == 0.
    A must be padded to ceil(M/32)*32 rows (unused rows zeroed).
    Tile config must be loaded before calling.
    """
    # Pack buffer for 2 B tiles (lo + hi halves of N_STEP)
    var bpack = InlineArray[Scalar[DType.int8], 2 * TILE_BYTES](
        fill=Scalar[DType.int8](0))
    var bp = UnsafePointer(to=bpack).bitcast[Scalar[DType.int8]]()

    for n_blk in range(0, n, N_STEP):
        for m_blk in range(0, m, M_STEP):
            # Zero C accumulators (tiles 4-7)
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()

            # Accumulate across K dimension
            for k_blk in range(0, k, K_STEP):
                # Pack B into VNNI: lo tile (cols 0..15), hi tile (cols 16..31)
                pack_b_vnni(b, n, k_blk, n_blk,
                            min(TILE_N, n - n_blk), bp)
                pack_b_vnni(b, n, k_blk, n_blk + TILE_N,
                            min(TILE_N, n - n_blk - TILE_N), bp + TILE_BYTES)

                # Load A tiles: rows [m_blk..m_blk+31], cols [k_blk..k_blk+63]
                tileload[0](a + m_blk * k + k_blk, k)
                tileload[1](a + (m_blk + TILE_M) * k + k_blk, k)

                # Load packed B tiles (VNNI stride = 64 bytes)
                tileload[2](bp, TILE_N * VNNI_BLK)
                tileload[3](bp + TILE_BYTES, TILE_N * VNNI_BLK)

                # 4 tile dot products (C stays in registers across K iterations)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()

            # Store C tiles to output (stride = N * sizeof(i32))
            var c0 = c + m_blk * n + n_blk
            var c_stride = n * size_of[Int32]()
            tilestore[4](c0, c_stride)
            tilestore[5](c0 + TILE_N, c_stride)
            tilestore[6](c0 + TILE_M * n, c_stride)
            tilestore[7](c0 + TILE_M * n + TILE_N, c_stride)


# ============================================================================
# Scalar Reference GEMM
# ============================================================================

def ref_i8_gemm(
    a: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    c: UnsafePointer[Int32, MutAnyOrigin],
    m: Int, k: Int, n: Int,
):
    """C[M,N] = A[M,K] x B[K,N], scalar i8 reference (exact integer)."""
    for i in range(m):
        for j in range(n):
            var acc = Int32(0)
            for kk in range(k):
                acc += Int32(a[i * k + kk]) * Int32(b[kk * n + j])
            c[i * n + j] = acc


# ============================================================================
# Test Harness
# ============================================================================

def run_test[M: Int, K: Int, N: Int]():
    """Run AMX GEMM at [M,K]x[K,N] and compare with scalar reference."""
    # Pad M to M_STEP so AMX reads valid memory for both A tiles
    comptime M_PAD = ((M + M_STEP - 1) // M_STEP) * M_STEP

    var a_arr = InlineArray[Scalar[DType.int8], M_PAD * K](
        fill=Scalar[DType.int8](0))
    var b_arr = InlineArray[Scalar[DType.int8], K * N](
        fill=Scalar[DType.int8](0))
    var c_amx_arr = InlineArray[Int32, M_PAD * N](fill=Int32(0))
    var c_ref_arr = InlineArray[Int32, M * N](fill=Int32(0))

    var a = UnsafePointer(to=a_arr).bitcast[Scalar[DType.int8]]()
    var b = UnsafePointer(to=b_arr).bitcast[Scalar[DType.int8]]()
    var ca = UnsafePointer(to=c_amx_arr).bitcast[Int32]()
    var cr = UnsafePointer(to=c_ref_arr).bitcast[Int32]()

    # Deterministic fill: values in [-125, 125]
    for i in range(M):
        for j in range(K):
            a[i * K + j] = Scalar[DType.int8]((i * K + j) % 251 - 125)
    for i in range(K * N):
        b[i] = Scalar[DType.int8](i % 239 - 119)

    ref_i8_gemm(a, b, cr, M, K, N)
    amx_gemm_224(a, b, ca, M_PAD, K, N)

    # Compare first M rows (bottom rows of M_PAD are padding)
    var mismatches = 0
    var max_err = 0
    for i in range(M):
        for j in range(N):
            var amx_val = ca[i * N + j]
            var ref_val = cr[i * N + j]
            if amx_val != ref_val:
                mismatches += 1
                var err = Int(amx_val - ref_val)
                if err < 0:
                    err = -err
                if err > max_err:
                    max_err = err
                if mismatches <= 3:
                    print("  MISMATCH [" + String(i) + "," + String(j)
                          + "] amx=" + String(amx_val) + " ref=" + String(ref_val))

    if mismatches == 0:
        print("  PASS (" + String(M * N) + " elements, exact match)")
    else:
        print("  FAIL: " + String(mismatches) + "/" + String(M * N)
              + " mismatches, max_err=" + String(max_err))


def main():
    print("=== AMX 2-2-4 Int8 GEMM Validation ===")

    if not init_intel_amx():
        print("SKIP: AMX not available (arch_prctl failed)")
        return

    var cfg = make_224_i8_config()
    ldtilecfg(UnsafePointer(to=cfg))
    print("Tiles configured: 2-2-4 i8, all 16x64")

    # Test 1: One atomic 2-2-4 call
    print("\nTest 1: M=32, K=64, N=32 (single 2-2-4 operation)")
    run_test[32, 64, 32]()

    # Test 2: Multiple K-steps and N-steps
    print("\nTest 2: M=32, K=128, N=64 (2 K-steps, 2 N-steps)")
    run_test[32, 128, 64]()

    # Test 3: M < M_STEP (decode attention: gqa_factor=16)
    print("\nTest 3: M=16, K=128, N=64 (decode: M < M_STEP)")
    run_test[16, 128, 64]()

    # Test 4: V aggregation dimensions (M=16, K=64, N=128)
    print("\nTest 4: M=16, K=64, N=128 (V aggregation shape)")
    run_test[16, 64, 128]()

    tilerelease()
    print("\nAll tests complete.")
