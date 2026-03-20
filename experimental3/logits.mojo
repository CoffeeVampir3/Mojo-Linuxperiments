"""Read-only logit view backed by arena memory.

The forward pass writes raw logits into a scratch buffer as bf16.
LogitsView provides typed, SIMD-friendly read access without exposing
arena internals. Consumer code (samplers, greedy decode, beam search)
depends on the LogitAccess trait, not the concrete backing.

Trait lattice extension:
  LogitAccess(Encoded) — read-only, SIMD load in f32, comptime VOCAB
"""

from memory import UnsafePointer
from sys.info import simd_width_of
from experimental3.tensor_contracts import Encoded


# ================================================================
# TRAIT
# ================================================================

trait LogitAccess(Encoded):
    """Read-only access to model output logits.
    VOCAB is comptime for SIMD loop bounds. rows() is runtime
    (1 for decode, N for prefill). load_f32 upcasts from storage
    dtype so consumers always work in f32."""
    comptime VOCAB: Int
    fn rows(self) -> Int: ...
    fn load_f32[width: Int](self, row: Int, offset: Int) -> SIMD[DType.float32, width]: ...


# ================================================================
# CONCRETE VIEW
# ================================================================

struct LogitsView[vocab: Int, dtype: DType = DType.bfloat16](LogitAccess):
    """Non-owning, read-only view of logits in arena scratch memory.
    Constructed from the address where the forward pass writes its
    final GEMM output. Valid only while the backing arena is alive.

    Parameterized on vocab (comptime) for SIMD-optimized access.
    Storage dtype is typically bf16 but carried as a parameter for
    generality."""
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab
    var _ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _rows: Int

    fn __init__(out self, addr: Int, seq_len: Int):
        self._ptr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=addr
        )
        self._rows = seq_len

    fn rows(self) -> Int:
        return self._rows

    fn load_f32[width: Int](self, row: Int, offset: Int) -> SIMD[DType.float32, width]:
        """Load a SIMD chunk of logits as f32 from the given row and offset.
        Upcasts from storage dtype. Offset is in elements, not bytes."""
        return (self._ptr + row * Self.vocab + offset).load[width=width]().cast[DType.float32]()
