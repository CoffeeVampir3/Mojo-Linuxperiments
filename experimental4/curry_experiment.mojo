# Design experiment: currying a pointer onto trait-carrying types
#
# Question: can we bind a runtime pointer to a comptime-described slot
# and keep ALL trait information alive?
#
# Path A: comptime traits + Bound[T] (lift to curry pointer)
# Path B: runtime from the start (var fields, just add ptr)


# ===================================================================
# PATH A: Comptime traits, curry pointer via Bound[T]
# ===================================================================

trait Enc:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Dim:
    comptime ROWS: Int
    comptime COLS: Int

struct BF16(Enc):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Enc):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

fn byte_count[T: Enc & Dim]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES

struct Slot[E: Enc, rows: Int, cols: Int](Enc, Dim):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.rows
    comptime COLS = Self.cols

# Bound[T] — curries a runtime pointer onto any Enc & Dim type.
# Written ONCE. Works for any slot. Preserves all trait conformance.
struct Bound[T: Enc & Dim](Enc, Dim):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    var ptr: Int

    fn __init__(out self, ptr: Int):
        self.ptr = ptr

# Generic ops — work on ANYTHING conforming to Enc & Dim
fn describe[T: Enc & Dim](label: String):
    print("  " + label + ":", T.ROWS, "x", T.COLS, T.DTYPE,
        "=", byte_count[T](), "bytes")

# This op ALSO gets the pointer — Bound conforms to Enc & Dim
fn gemv[T: Enc & Dim](view: Bound[T], x_ptr: Int):
    # comptime: knows dtype, rows, cols for codegen
    # runtime: has the actual pointer
    print("  gemv:", T.ROWS, "x", T.COLS, T.DTYPE,
        "@ ptr", view.ptr, "* x @", x_ptr)


# ===================================================================
# PATH B: Runtime from the start
# ===================================================================

struct RSlot:
    var rows: Int
    var cols: Int
    var dtype: DType
    var element_bytes: Int
    var ptr: Int

    fn __init__(out self, rows: Int, cols: Int, dtype: DType, element_bytes: Int):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.element_bytes = element_bytes
        self.ptr = 0

    fn bind(mut self, ptr: Int):
        self.ptr = ptr

    fn byte_count(self) -> Int:
        return self.rows * self.cols * self.element_bytes

fn describe_r(label: String, s: RSlot):
    print("  " + label + ":", s.rows, "x", s.cols, s.dtype,
        "=", s.byte_count(), "bytes")

fn gemv_r(s: RSlot, x_ptr: Int):
    # runtime only — can't specialize codegen on dtype or dims
    print("  gemv:", s.rows, "x", s.cols, s.dtype,
        "@ ptr", s.ptr, "* x @", x_ptr)


# ===================================================================
# COMPARISON
# ===================================================================

fn main():
    print("=== Path A: Comptime traits + Bound ===")

    comptime Q = Slot[BF16, 576, 576]
    comptime NORM = Slot[F32, 576, 1]

    # Spec phase — pure types, no pointer
    describe[Q]("q_proj spec")
    describe[NORM]("norm spec")

    # Load phase — Bound curries pointer onto spec
    var q_bound = Bound[Q](ptr=0xDEAD0000)
    var norm_bound = Bound[NORM](ptr=0xBEEF0000)

    # Still works with generic Enc & Dim functions
    describe[Bound[Q]]("q_proj bound")
    describe[Bound[NORM]]("norm bound")

    # AND has typed pointer access
    gemv[Q](q_bound, x_ptr=0xAAAA)

    print()
    print("=== Path B: Runtime slots ===")

    var rq = RSlot(576, 576, DType.bfloat16, 2)
    var rnorm = RSlot(576, 1, DType.float32, 4)

    describe_r("q_proj spec", rq)
    describe_r("norm spec", rnorm)

    rq.bind(0xDEAD0000)
    describe_r("q_proj bound", rq)
    gemv_r(rq, x_ptr=0xAAAA)

    print()
    print("=== Tradeoff ===")
    print("  A: dtype, dims known at comptime -> compiler specializes codegen")
    print("  B: dtype, dims are runtime vars -> generic but no specialization")
