"""Trait-first tensor type system.

Trait lattice:
  Encoded    — DType
  Shaped     — comptime ROWS, COLS (weights)
  DynShaped  — comptime COLS, runtime rows() (activations)
  Sharded    — SHARD_AXIS
  Identified — SLOT_ID, tensor_name()
  ScaleBuffer — BLOCK_SIZE (marker for scale tensors)
  DataAccess(Encoded, Shaped)     — .data() pointer (weights)
  DynAccess(Encoded, DynShaped)   — .data() pointer (activations)
  CacheAccess(Encoded)            — .data() pointer (KV cache)
  BufferAccess(Encoded, Shaped)   — .data() pointer (precomputed buffers)

Composition aliases:
  Descriptor = Encoded & Shaped & Sharded & Identified

Descriptor structs:
  DenseDesc  — weight/norm/embedding descriptor (checkpoint-loaded)
  ScaleDesc  — DenseDesc + ScaleBuffer (for quantization scales)
  BufferDesc — precomputed buffer descriptor (computed at init, not loaded)

DenseDesc/ScaleDesc parameters rows/cols are GLOBAL (checkpoint) dims.
Trait members ROWS/COLS are LOCAL (per-node) dims, computed as
global // tp on the shard axis. For TP=1, local == global.

Tile structs:
  DenseTile  — DataAccess carrier, comptime dims (weights)
  ScaleTile  — DenseTile + ScaleBuffer
  ActTile    — DynAccess carrier, runtime rows (activations)
  CacheTile  — CacheAccess carrier, comptime capacity (KV cache)
  BufferTile — BufferAccess carrier, comptime dims (precomputed buffers)
"""

from memory import UnsafePointer

from experimental3.core import (
    StaticTensorSpec,
    AXIS_NONE,
    AXIS_HOST,
    dtype_bytes,
    shape_1d,
    shape_2d,
)


# ================================================================
# TRAIT LATTICE
# ================================================================

trait Encoded:
    comptime DTYPE: DType

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait Sharded:
    comptime SHARD_AXIS: Int

trait Identified:
    comptime SLOT_ID: Int
    @staticmethod
    fn tensor_name(layer_idx: Int) -> String: ...

trait DynShaped:
    comptime COLS: Int
    fn rows(self) -> Int: ...

trait ScaleBuffer:
    """Marker: this buffer carries scale data with a known block size."""
    comptime BLOCK_SIZE: Int

trait DataAccess(Encoded, Shaped):
    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]: ...

trait DynAccess(Encoded, DynShaped):
    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]: ...

trait CacheAccess(Encoded):
    """Fixed-capacity, position-addressed cache buffer (KV cache)."""
    comptime COLS: Int
    comptime CAPACITY: Int
    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]: ...

trait BufferAccess(Encoded, Shaped):
    """Precomputed read-only buffer (RoPE tables, etc). Computed at init, immutable after."""
    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]: ...

comptime Descriptor = Encoded & Shaped & Sharded & Identified


# ================================================================
# FUNCTIONS ON TRAIT COMPOSITIONS
# ================================================================

fn storage_bytes[T: Encoded & Shaped]() -> Int:
    return T.ROWS * T.COLS * dtype_bytes(T.DTYPE)


# ================================================================
# DESCRIPTORS
# ================================================================

struct DenseDesc[
    slot_id: Int, dtype: DType, rows: Int, cols: Int,
    shard_axis: Int, name_suffix: StringLiteral,
    tp: Int = 1,
](Encoded, Shaped, Sharded, Identified):
    comptime DTYPE = Self.dtype
    # ROWS/COLS are local (per-node) dims. Parameters rows/cols are global.
    comptime ROWS = (Self.rows // Self.tp) if Self.shard_axis == 0 else Self.rows
    comptime COLS = (Self.cols // Self.tp) if Self.shard_axis == 1 else Self.cols
    comptime SHARD_AXIS = Self.shard_axis
    comptime SLOT_ID = Self.slot_id

    @staticmethod
    fn tensor_name(layer_idx: Int) -> String:
        if layer_idx < 0:
            return String(Self.name_suffix)
        return "model.layers." + String(layer_idx) + "." + String(Self.name_suffix)

    @staticmethod
    fn emit_specs(layer_idx: Int, mut specs: List[StaticTensorSpec]):
        @parameter
        if Self.rows == 1:
            specs.append(StaticTensorSpec(
                UInt16(Self.slot_id),
                String(Self.name_suffix),
                layer_idx,
                Self.tensor_name(layer_idx),
                shape_1d(Self.cols),
                Self.dtype,
                Self.shard_axis,
            ))
        else:
            specs.append(StaticTensorSpec(
                UInt16(Self.slot_id),
                String(Self.name_suffix),
                layer_idx,
                Self.tensor_name(layer_idx),
                shape_2d(Self.rows, Self.cols),
                Self.dtype,
                Self.shard_axis,
            ))


struct ScaleDesc[
    slot_id: Int, dtype: DType, rows: Int, cols: Int,
    shard_axis: Int, block_size: Int, name_suffix: StringLiteral,
    tp: Int = 1,
](Encoded, Shaped, Sharded, Identified, ScaleBuffer):
    """Descriptor for a scale tensor. Same traits as DenseDesc, plus BLOCK_SIZE."""
    comptime DTYPE = Self.dtype
    comptime ROWS = (Self.rows // Self.tp) if Self.shard_axis == 0 else Self.rows
    comptime COLS = (Self.cols // Self.tp) if Self.shard_axis == 1 else Self.cols
    comptime SHARD_AXIS = Self.shard_axis
    comptime SLOT_ID = Self.slot_id
    comptime BLOCK_SIZE = Self.block_size

    @staticmethod
    fn tensor_name(layer_idx: Int) -> String:
        return "model.layers." + String(layer_idx) + "." + String(Self.name_suffix)

    @staticmethod
    fn emit_specs(layer_idx: Int, mut specs: List[StaticTensorSpec]):
        @parameter
        if Self.rows == 1:
            specs.append(StaticTensorSpec(
                UInt16(Self.slot_id),
                String(Self.name_suffix),
                layer_idx,
                Self.tensor_name(layer_idx),
                shape_1d(Self.cols),
                Self.dtype,
                Self.shard_axis,
            ))
        else:
            specs.append(StaticTensorSpec(
                UInt16(Self.slot_id),
                String(Self.name_suffix),
                layer_idx,
                Self.tensor_name(layer_idx),
                shape_2d(Self.rows, Self.cols),
                Self.dtype,
                Self.shard_axis,
            ))


struct BufferDesc[dtype: DType, rows: Int, cols: Int](Encoded, Shaped):
    """Descriptor for a precomputed buffer. Not loaded from checkpoint.
    Participates in byte accounting via storage_bytes[BufferDesc]()."""
    comptime DTYPE = Self.dtype
    comptime ROWS = Self.rows
    comptime COLS = Self.cols


# ================================================================
# TILES — runtime data carriers
# ================================================================

struct DenseTile[dtype: DType, rows: Int, cols: Int](DataAccess):
    comptime DTYPE = Self.dtype
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    fn __init__(out self, addr: Int):
        self.ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )

    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]:
        return self.ptr


struct ScaleTile[dtype: DType, rows: Int, cols: Int, block_size: Int](DataAccess, ScaleBuffer):
    """DenseTile that also declares BLOCK_SIZE. Same .data() interface."""
    comptime DTYPE = Self.dtype
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime BLOCK_SIZE = Self.block_size
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    fn __init__(out self, addr: Int):
        self.ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )

    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]:
        return self.ptr


struct ActTile[dtype: DType, cols: Int](DynAccess):
    """Activation tile — runtime rows, comptime cols and dtype."""
    comptime DTYPE = Self.dtype
    comptime COLS = Self.cols
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _rows: Int

    fn __init__(out self, addr: Int, rows: Int):
        self.ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )
        self._rows = rows

    fn rows(self) -> Int:
        return self._rows

    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]:
        return self.ptr


struct CacheTile[dtype: DType, cols: Int, capacity: Int](CacheAccess):
    """Fixed-capacity cache tile — comptime dtype, cols, and capacity."""
    comptime DTYPE = Self.dtype
    comptime COLS = Self.cols
    comptime CAPACITY = Self.capacity
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    fn __init__(out self, addr: Int):
        self.ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )

    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]:
        return self.ptr


struct BufferTile[dtype: DType, rows: Int, cols: Int](BufferAccess):
    """Precomputed buffer tile — comptime dtype, rows, and cols."""
    comptime DTYPE = Self.dtype
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    fn __init__(out self, addr: Int):
        self.ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )

    fn data(self) -> UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]:
        return self.ptr


# ================================================================
# DESCRIPTOR → TILE BRIDGE
# ================================================================

@always_inline
fn tile_from[desc: Descriptor](addr: Int) -> DenseTile[desc.DTYPE, desc.ROWS, desc.COLS]:
    return DenseTile[desc.DTYPE, desc.ROWS, desc.COLS](addr)

@always_inline
fn scale_tile_from[desc: Descriptor & ScaleBuffer](addr: Int) -> ScaleTile[desc.DTYPE, desc.ROWS, desc.COLS, desc.BLOCK_SIZE]:
    return ScaleTile[desc.DTYPE, desc.ROWS, desc.COLS, desc.BLOCK_SIZE](addr)

@always_inline
fn buffer_tile_from[desc: Encoded & Shaped](addr: Int) -> BufferTile[desc.DTYPE, desc.ROWS, desc.COLS]:
    return BufferTile[desc.DTYPE, desc.ROWS, desc.COLS](addr)
