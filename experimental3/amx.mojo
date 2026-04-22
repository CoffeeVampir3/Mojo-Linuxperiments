"""AMX tile intrinsic wrappers and configuration for Sapphire Rapids+.

Provides the 2-2-4 tile layout used across all int8 GEMM operations:
  TMM0, TMM1: A tiles  (16 rows x 64 bytes, row-major)
  TMM2, TMM3: B tiles  (16 rows x 64 bytes, VNNI-packed)
  TMM4-TMM7:  C tiles  (16 rows x 64 bytes = 16x16 i32)

  M_STEP=32, N_STEP=32, K_STEP=64.

Tile config is set once per thread via ldtilecfg. All tile operations
use the configured dimensions implicitly.
"""

from std.sys import llvm_intrinsic
from std.memory import UnsafePointer
from std.collections import InlineArray


# ============================================================================
# Tile Dimensions
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
# Tile Configuration
# ============================================================================

@fieldwise_init
struct TileConfig(Movable):
    """64-byte config block for ldtilecfg.

    Layout: palette(1) + start_row(1) + reserved(14) + colsb(16 x u16) + rows(16 x u8).
    """
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


def make_224_decode_config[hpg: Int]() -> TileConfig:
    """2-2-4 decode config: A/C tiles use hpg rows, B tiles use 16 (K dimension)."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    cfg.rows[0] = UInt8(hpg)
    cfg.rows[1] = UInt8(hpg)
    cfg.rows[2] = 16
    cfg.rows[3] = 16
    cfg.rows[4] = UInt8(hpg)
    cfg.rows[5] = UInt8(hpg)
    cfg.rows[6] = UInt8(hpg)
    cfg.rows[7] = UInt8(hpg)
    for i in range(8):
        cfg.colsb[i] = 64
    return cfg^


def make_133_i8_config() -> TileConfig:
    """1-3-3 int8 config for decode: TMM0 A, TMM1-3 B, TMM4-6 C.
    All active tiles 16 rows x 64 col-bytes. TMM7 unused."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    for i in range(7):
        cfg.rows[i] = 16
        cfg.colsb[i] = 64
    return cfg^


# ============================================================================
# AMX System Init (Linux)
# ============================================================================

def init_intel_amx() -> Bool:
    """Request kernel permission for AMX tile data via arch_prctl.

    Must be called once per process before any AMX instruction.
    """
    comptime SYS_arch_prctl = 158
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    var result = __mlir_op.`pop.external_call`[
        func = "syscall".value,
        _type = Int64,
    ](Int64(SYS_arch_prctl), Int64(ARCH_REQ_XCOMP_PERM), Int64(XFEATURE_XTILEDATA))
    return result == 0


# ============================================================================
# Intrinsic Wrappers
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

@always_inline
def tdpbf16ps[dst: Int, src_a: Int, src_b: Int]():
    """Tile dot product: bf16 x bf16 -> f32. TMM<dst> += TMM<src_a>(bf16) x TMM<src_b>(bf16).

    VNNI groups of 2: each dword lane accumulates 2 bf16 multiplies.
    TILE_K for bf16 = 32 elements (64 bytes). Same tile dimensions.
    """
    comptime assert dst >= 0 and dst < 8 and src_a >= 0 and src_a < 8 and src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbf16ps", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def tile_dp[dst: Int, src_a: Int, src_b: Int, a_dtype: DType, b_dtype: DType]():
    """Tile dot product — instruction auto-selected from A/B dtypes."""
    comptime assert dst >= 0 and dst < 8 and src_a >= 0 and src_a < 8 and src_b >= 0 and src_b < 8
    comptime if a_dtype == DType.int8 and b_dtype == DType.int8:
        llvm_intrinsic["llvm.x86.tdpbssd", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    elif a_dtype == DType.int8 and b_dtype == DType.uint8:
        llvm_intrinsic["llvm.x86.tdpbsud", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    elif a_dtype == DType.uint8 and b_dtype == DType.int8:
        llvm_intrinsic["llvm.x86.tdpbusd", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    elif a_dtype == DType.uint8 and b_dtype == DType.uint8:
        llvm_intrinsic["llvm.x86.tdpbuud", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    elif a_dtype == DType.bfloat16 and b_dtype == DType.bfloat16:
        llvm_intrinsic["llvm.x86.tdpbf16ps", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    elif a_dtype == DType.float16 and b_dtype == DType.float16:
        llvm_intrinsic["llvm.x86.tdpfp16ps", NoneType](Int8(dst), Int8(src_a), Int8(src_b))
    else:
        comptime assert False, "unsupported dtype combination for tile_dp"
