# Can Bound[T] conform to MORE traits than its parameter T requires?
#
# If T: Enc & Dim & Placed & Named, then Bound[T] can forward all four
# PLUS add DataAccess. Different consumers bind on different subsets:
#   - Loader: [T: Enc & Dim & Placed & Named] — ignores pointer
#   - Ops:    uses Bound[T] which has pointer + all trait info
#   - Spec:   [T: Enc & Dim] — just needs sizing

trait Enc:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int

trait Dim:
    comptime ROWS: Int
    comptime COLS: Int

trait Placed:
    comptime OFFSET: Int
    comptime GLOBAL_ROWS: Int
    comptime GLOBAL_COLS: Int

trait Named:
    comptime NAME: StaticString

struct BF16(Enc):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Enc):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4

fn byte_count[T: Enc & Dim]() -> Int:
    return T.ROWS * T.COLS * T.ELEMENT_BYTES

# Full spec slot — all four traits
struct Slot[
    E: Enc, rows: Int, cols: Int, offset: Int, name: StringLiteral,
](Enc, Dim, Placed, Named):
    comptime DTYPE = Self.E.DTYPE
    comptime ELEMENT_BYTES = Self.E.ELEMENT_BYTES
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime OFFSET = Self.offset
    comptime GLOBAL_ROWS = Self.rows
    comptime GLOBAL_COLS = Self.cols
    comptime NAME: StaticString = Self.name

# Bound — forwards ALL of T's traits, adds pointer
struct Bound[T: Enc & Dim & Placed & Named](Enc, Dim, Placed, Named):
    comptime DTYPE = Self.T.DTYPE
    comptime ELEMENT_BYTES = Self.T.ELEMENT_BYTES
    comptime ROWS = Self.T.ROWS
    comptime COLS = Self.T.COLS
    comptime OFFSET = Self.T.OFFSET
    comptime GLOBAL_ROWS = Self.T.GLOBAL_ROWS
    comptime GLOBAL_COLS = Self.T.GLOBAL_COLS
    comptime NAME: StaticString = Self.T.NAME
    var ptr: Int

    fn __init__(out self, ptr: Int):
        self.ptr = ptr


# ===--- Different consumers, different trait bounds ---===

# Loader only needs Enc & Dim & Placed & Named — no pointer
fn loader_validate[T: Enc & Dim & Placed & Named]():
    print("  validate:", String(T.NAME), "@" + String(T.OFFSET),
        String(T.GLOBAL_ROWS) + "x" + String(T.GLOBAL_COLS), T.DTYPE)

# Sizing only needs Enc & Dim
fn size_of[T: Enc & Dim]() -> Int:
    return byte_count[T]()

# Operation needs the pointer — takes Bound directly
fn gemv[T: Enc & Dim & Placed & Named](view: Bound[T], x_ptr: Int):
    print("  gemv:", String(T.NAME), T.ROWS, "x", T.COLS, T.DTYPE,
        "@ ptr", view.ptr)


# ===--- Model with layer iteration ---===

struct TinyLayer[E: Enc, tp: Int]:
    comptime Q    = Slot[Self.E, 576, 576, 0, "self_attn.q_proj.weight"]
    comptime K    = Slot[Self.E, 192, 576, 663552, "self_attn.k_proj.weight"]
    comptime NORM = Slot[F32, 576, 1, 884736, "input_layernorm.weight"]

struct TinyModel[E: Enc, tp: Int]:
    comptime NUM_LAYERS = 2
    comptime EMBED = Slot[Self.E, 1024, 576, 0, "model.embed_tokens.weight"]
    comptime LAYER = TinyLayer[Self.E, Self.tp]
    comptime LAYERS_OFF = 1179648
    comptime LAYER_STRIDE = 886400


fn main():
    comptime M = TinyModel[BF16, 1]
    comptime L = M.LAYER

    print("=== Loader: uses Enc & Dim & Placed & Named ===")
    loader_validate[M.EMBED]()
    loader_validate[L.Q]()
    loader_validate[L.K]()
    loader_validate[L.NORM]()

    print()
    print("=== Sizing: uses Enc & Dim only ===")
    print("  embed:", size_of[M.EMBED](), "bytes")
    print("  q:", size_of[L.Q](), "bytes")
    print("  norm:", size_of[L.NORM](), "bytes")

    print()
    print("=== After loading: Bound carries everything + pointer ===")
    var q = Bound[L.Q](ptr=0xDEAD0000)
    var k = Bound[L.K](ptr=0xBEEF0000)

    # Bound still works with ALL trait-bounded functions
    loader_validate[Bound[L.Q]]()  # same function, Bound conforms
    print("  size:", size_of[Bound[L.Q]](), "bytes")  # same function

    # AND can do operations with the pointer
    gemv(q, x_ptr=0xAAAA)
    gemv(k, x_ptr=0xAAAA)

    print()
    print("=== Same Bound[T] type, three different consumer views ===")
    print("  Loader sees:  name, offset, global shape, dtype")
    print("  Sizer sees:   rows, cols, element_bytes")
    print("  Op sees:      all of the above + pointer")
