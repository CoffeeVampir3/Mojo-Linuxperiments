from std.memory import UnsafePointer
from modeling.linear_borrow_pool import ScratchLease


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait Aligned:
    comptime ALIGNMENT: Int

trait HasPtr(Encoding):
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]: ...
    def addr(self) -> Int: ...

trait Dynamic:
    def seq_len(self) -> Int: ...


# Shorthand for the common trait compositions used by kernel signatures.
comptime StaticTensor = Encoding & Shaped & HasPtr & Aligned
comptime DynamicTensor = Encoding & Shaped & HasPtr & Aligned & Dynamic


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1


comptime DEFAULT_ALIGNMENT = 64


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


trait ShapeLike:
    comptime GLOBAL_N: Int
    comptime GLOBAL_M: Int
    comptime DATA_N: Int
    comptime DATA_M: Int
    comptime N: Int
    comptime M: Int
    comptime ELEMS: Int

    @staticmethod
    def bytes[E: Encoding]() -> Int: ...
    @staticmethod
    def row_bytes[E: Encoding]() -> Int: ...
    @staticmethod
    def col_bytes[E: Encoding]() -> Int: ...
    @staticmethod
    def rows_bytes[E: Encoding](rows: Int) -> Int: ...

struct Shape[
    global_n: Int, global_m: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    tp: Int = 1,
    align_n: Int = 1, align_m: Int = 1,
](ShapeLike):
    comptime GLOBAL_N = Self.global_n
    comptime GLOBAL_M = Self.global_m
    comptime DATA_N = Self.global_n // Self.tp if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.tp if Self.shard_m else Self.global_m
    comptime N = align_up(Self.DATA_N, Self.align_n)
    comptime M = align_up(Self.DATA_M, Self.align_m)
    comptime PAD_N = Self.N - Self.DATA_N
    comptime PAD_M = Self.M - Self.DATA_M
    comptime ELEMS = Self.N * Self.M

    @staticmethod
    def bytes[E: Encoding]() -> Int:
        """Bytes for the full static tensor: N * M * element-bytes(E)."""
        return Self.ELEMS * E.ELEMENT_BYTES

    @staticmethod
    def row_bytes[E: Encoding]() -> Int:
        """Bytes for one row (M elements) of encoding E."""
        return Self.M * E.ELEMENT_BYTES

    @staticmethod
    def col_bytes[E: Encoding]() -> Int:
        """Bytes for one column (N elements) of encoding E."""
        return Self.N * E.ELEMENT_BYTES

    @staticmethod
    def rows_bytes[E: Encoding](rows: Int) -> Int:
        """Bytes for `rows` rows of M elements each (for dynamic seq_len)."""
        return rows * Self.M * E.ELEMENT_BYTES


comptime DISTRIBUTED = -1
comptime HOST_RANK = 0


@fieldwise_init
struct StaticView[E: Encoding, S: ShapeLike, alignment: Int = DEFAULT_ALIGNMENT](
    Encoding, Shaped, HasPtr, Aligned,
):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.N
    comptime COLS = Self.S.M
    comptime ALIGNMENT = Self.alignment
    var ptr: UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](self.ptr)

    @always_inline
    def addr(self) -> Int:
        return Int(self.ptr)

    @always_inline
    def load[width: Int = 1](self, offset: Int = 0) -> SIMD[Self.DTYPE, width]:
        return self.ptr.load[width=width, alignment=Self.ALIGNMENT](offset)

    @always_inline
    def store[width: Int = 1](self, offset: Int, val: SIMD[Self.DTYPE, width]):
        self.ptr.store[alignment=Self.ALIGNMENT](offset, val)


@fieldwise_init
struct DynamicView[E: Encoding, S: ShapeLike, alignment: Int = DEFAULT_ALIGNMENT](
    Encoding, Shaped, HasPtr, Aligned, Dynamic,
):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.N
    comptime COLS = Self.S.M
    comptime ALIGNMENT = Self.alignment
    var ptr: UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]
    var runtime_rows: Int

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](self.ptr)

    @always_inline
    def addr(self) -> Int:
        return Int(self.ptr)

    @always_inline
    def seq_len(self) -> Int:
        return self.runtime_rows

    @always_inline
    def load[width: Int = 1](self, offset: Int = 0) -> SIMD[Self.DTYPE, width]:
        return self.ptr.load[width=width, alignment=Self.ALIGNMENT](offset)

    @always_inline
    def store[width: Int = 1](self, offset: Int, val: SIMD[Self.DTYPE, width]):
        self.ptr.store[alignment=Self.ALIGNMENT](offset, val)


@fieldwise_init
struct ScratchView[
    E: Encoding, S: ShapeLike, origin: MutOrigin,
    alignment: Int = DEFAULT_ALIGNMENT,
](Encoding, Shaped, HasPtr, Aligned, Dynamic):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.S.N
    comptime COLS = Self.S.M
    comptime ALIGNMENT = Self.alignment
    var ptr: UnsafePointer[Scalar[Self.DTYPE], Self.origin]
    var runtime_rows: Int

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            self.ptr.as_any_origin())

    @always_inline
    def addr(self) -> Int:
        return Int(self.ptr)

    @always_inline
    def seq_len(self) -> Int:
        return self.runtime_rows

    @always_inline
    def load[width: Int = 1](self, offset: Int = 0) -> SIMD[Self.DTYPE, width]:
        return self.ptr.load[width=width, alignment=Self.ALIGNMENT](offset)

    @always_inline
    def store[width: Int = 1](self, offset: Int, val: SIMD[Self.DTYPE, width]):
        self.ptr.store[alignment=Self.ALIGNMENT](offset, val)

    @always_inline
    def any(self) -> DynamicView[Self.E, Self.S, Self.alignment]:
        return DynamicView[Self.E, Self.S, Self.alignment](
            self.ptr.as_any_origin(), self.runtime_rows)


@fieldwise_init
struct WeightDesc(Copyable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_rows: Int
    var local_cols: Int
    var data_rows: Int
    var data_cols: Int
    var quantizable: Bool
    var target_rank: Int


from quant.source_format import Converter


trait QuantizeSpec:
    comptime SOURCE_DTYPE: DType
    comptime AUX_DTYPE: DType
    comptime SOURCE_ELEMENT_BYTES: Int
    comptime AUX_SUFFIX: StaticString
    comptime AUX_ROW_BLOCK: Int

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int: ...

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ): ...

    def weight_name(self) -> String: ...
    def fwht_block(self) -> Int: ...
    def is_per_block(self) -> Bool: ...
    def gamma_source(self) -> String: ...
    def two_sided_head_dim(self) -> Int: ...


trait TaskVisitor:
    def quantize[T: QuantizeSpec](mut self, task: T) -> Bool: ...
    def passthrough(mut self, name: String, expected_dtype: DType) -> Bool: ...
    def router_gauge_bf16(mut self, name: String) -> Bool: ...


struct PerRow[Src: Converter](QuantizeSpec):
    comptime SOURCE_DTYPE = Self.Src.SOURCE_DTYPE
    comptime AUX_DTYPE = Self.Src.AUX_DTYPE
    comptime SOURCE_ELEMENT_BYTES = Self.Src.SOURCE_ELEMENT_BYTES
    comptime AUX_SUFFIX = Self.Src.AUX_SUFFIX
    comptime AUX_ROW_BLOCK = Self.Src.AUX_ROW_BLOCK

    var name: String
    var block: Int

    def __init__(out self, name: String, block: Int):
        self.name = name
        self.block = block

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return Self.Src.aux_bytes_for(rows, cols)

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.Src.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.Src.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        Self.Src.convert[dst_dtype](src, aux, dst, rows, cols)

    def weight_name(self) -> String: return self.name
    def fwht_block(self) -> Int: return self.block
    def is_per_block(self) -> Bool: return False
    def gamma_source(self) -> String: return String("")
    def two_sided_head_dim(self) -> Int: return 0


struct PerRowAbsorbed[Src: Converter](QuantizeSpec):
    comptime SOURCE_DTYPE = Self.Src.SOURCE_DTYPE
    comptime AUX_DTYPE = Self.Src.AUX_DTYPE
    comptime SOURCE_ELEMENT_BYTES = Self.Src.SOURCE_ELEMENT_BYTES
    comptime AUX_SUFFIX = Self.Src.AUX_SUFFIX
    comptime AUX_ROW_BLOCK = Self.Src.AUX_ROW_BLOCK

    var name: String
    var block: Int
    var gamma: String

    def __init__(out self, name: String, block: Int, gamma: String):
        self.name = name
        self.block = block
        self.gamma = gamma

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return Self.Src.aux_bytes_for(rows, cols)

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.Src.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.Src.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        Self.Src.convert[dst_dtype](src, aux, dst, rows, cols)

    def weight_name(self) -> String: return self.name
    def fwht_block(self) -> Int: return self.block
    def is_per_block(self) -> Bool: return False
    def gamma_source(self) -> String: return self.gamma
    def two_sided_head_dim(self) -> Int: return 0


struct PerBlock[Src: Converter](QuantizeSpec):
    comptime SOURCE_DTYPE = Self.Src.SOURCE_DTYPE
    comptime AUX_DTYPE = Self.Src.AUX_DTYPE
    comptime SOURCE_ELEMENT_BYTES = Self.Src.SOURCE_ELEMENT_BYTES
    comptime AUX_SUFFIX = Self.Src.AUX_SUFFIX
    comptime AUX_ROW_BLOCK = Self.Src.AUX_ROW_BLOCK

    var name: String
    var block: Int

    def __init__(out self, name: String, block: Int):
        self.name = name
        self.block = block

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return Self.Src.aux_bytes_for(rows, cols)

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.Src.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.Src.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        Self.Src.convert[dst_dtype](src, aux, dst, rows, cols)

    def weight_name(self) -> String: return self.name
    def fwht_block(self) -> Int: return self.block
    def is_per_block(self) -> Bool: return True
    def gamma_source(self) -> String: return String("")
    def two_sided_head_dim(self) -> Int: return 0


struct PerBlockAbsorbed[Src: Converter](QuantizeSpec):
    comptime SOURCE_DTYPE = Self.Src.SOURCE_DTYPE
    comptime AUX_DTYPE = Self.Src.AUX_DTYPE
    comptime SOURCE_ELEMENT_BYTES = Self.Src.SOURCE_ELEMENT_BYTES
    comptime AUX_SUFFIX = Self.Src.AUX_SUFFIX
    comptime AUX_ROW_BLOCK = Self.Src.AUX_ROW_BLOCK

    var name: String
    var block: Int
    var gamma: String

    def __init__(out self, name: String, block: Int, gamma: String):
        self.name = name
        self.block = block
        self.gamma = gamma

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return Self.Src.aux_bytes_for(rows, cols)

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.Src.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.Src.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        Self.Src.convert[dst_dtype](src, aux, dst, rows, cols)

    def weight_name(self) -> String: return self.name
    def fwht_block(self) -> Int: return self.block
    def is_per_block(self) -> Bool: return True
    def gamma_source(self) -> String: return self.gamma
    def two_sided_head_dim(self) -> Int: return 0


struct TwoSidedAbsorbed[Src: Converter](QuantizeSpec):
    comptime SOURCE_DTYPE = Self.Src.SOURCE_DTYPE
    comptime AUX_DTYPE = Self.Src.AUX_DTYPE
    comptime SOURCE_ELEMENT_BYTES = Self.Src.SOURCE_ELEMENT_BYTES
    comptime AUX_SUFFIX = Self.Src.AUX_SUFFIX
    comptime AUX_ROW_BLOCK = Self.Src.AUX_ROW_BLOCK

    var name: String
    var block: Int
    var gamma: String
    var hdim: Int

    def __init__(out self, name: String, block: Int, gamma: String, hdim: Int):
        self.name = name
        self.block = block
        self.gamma = gamma
        self.hdim = hdim

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return Self.Src.aux_bytes_for(rows, cols)

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.Src.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.Src.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        Self.Src.convert[dst_dtype](src, aux, dst, rows, cols)

    def weight_name(self) -> String: return self.name
    def fwht_block(self) -> Int: return self.block
    def is_per_block(self) -> Bool: return False
    def gamma_source(self) -> String: return self.gamma
    def two_sided_head_dim(self) -> Int: return self.hdim


@explicit_destroy
struct LogitsView[vocab: Int, dtype: DType = DType.bfloat16](Movable):
    """Owning view of one logit vector (VOCAB elements).

    Holds a ScratchLease — the scratch offset is reserved until this
    view is dropped. The caller must drop the previous LogitsView
    before calling forward() again.
    """
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var lease: ScratchLease

    def __init__(out self, ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin], var lease: ScratchLease):
        self.ptr = ptr
        self.lease = lease^

    def load_f32[width: Int](self, offset: Int) -> SIMD[DType.float32, width]:
        return (self.ptr + offset).load[width=width]().cast[DType.float32]()

    def release(deinit self):
        """Drop the view, returning the scratch offset to the pool."""
        self.lease^.release()
