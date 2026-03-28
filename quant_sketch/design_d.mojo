"""Design D: Modular requirements via trait conformance.

Recipes declare what they need by conforming to traits. The engine
dispatches via comptime if conforms_to(), calling the right overload.
Trait bounds enforce correctness — you can't forget to provide gamma
for a recipe that needs it, and you can't accidentally pass gamma
to one that doesn't.

The key idea: instead of one monolithic apply(ctx, aux) where aux might
be null, different recipes have different apply signatures expressed via
separate traits. The dispatch is compile-time resolved.

Model-side:

    comptime Q_PROJ = Weight[..., HadamardGamma[64]]   # needs gamma
    comptime O_PROJ = Weight[..., Hadamard[64]]         # no gamma
    comptime Q_PROJ_CH = Weight[..., Channelwise]       # no rotation

    # In for_each_weight callback, comptime if selects the right call:
    comptime if conforms_to(R, NeedsGamma):
        R.apply(ctx, gamma)
    elif conforms_to(R, BasicRecipe):
        R.apply(ctx)
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]


# ============================================================================
# Atomic ops
# ============================================================================


def scale_columns_impl(f32: Ptr, gamma: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        for k in range(cols):
            f32[r * cols + k] = f32[r * cols + k] * gamma[k]

def fwht_inplace(buf: Ptr, n: Int):
    var half = 1
    while half < n:
        var i = 0
        while i < n:
            for j in range(half):
                var a = buf[i + j]
                var b = buf[i + j + half]
                buf[i + j] = a + b
                buf[i + j + half] = a - b
            i += half * 2
        half *= 2
    var sc = 1.0 / Float32(n)
    for i in range(n):
        buf[i] = buf[i] * sc

def fwht_rows_impl(f32: Ptr, rows: Int, cols: Int, block: Int):
    for r in range(rows):
        var base = f32 + r * cols
        for b in range(cols // block):
            fwht_inplace(base + b * block, block)

def compute_scales_impl(f32: Ptr, scales: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        var amax = Float32(0)
        for k in range(cols):
            var v = f32[r * cols + k]
            var a = v if v >= 0 else -v
            if a > amax: amax = a
        scales[r] = amax / 127.0

def quantize_int8_impl(f32: Ptr, scales: Ptr, qi: PtrI8, rows: Int, cols: Int):
    for r in range(rows):
        var sc = scales[r]
        var inv = 1.0 / sc if sc > 0 else Float32(0)
        for k in range(cols):
            var v = f32[r * cols + k] * inv
            var rounded = Int(v + 0.5) if v >= 0 else -Int(-v + 0.5)
            if rounded > 127: rounded = 127
            elif rounded < -128: rounded = -128
            qi[r * cols + k] = Scalar[DType.int8](rounded)


# ============================================================================
# Quantization context — the buffers the engine owns
# ============================================================================


struct QuantContext:
    var f32: Ptr
    var qi: PtrI8
    var scales: Ptr
    var rows: Int
    var cols: Int

    def __init__(out self, f32: Ptr, qi: PtrI8, scales: Ptr,
                 rows: Int, cols: Int):
        self.f32 = f32
        self.qi = qi
        self.scales = scales
        self.rows = rows
        self.cols = cols


# ============================================================================
# Requirement traits — recipes conform to declare what they need
# ============================================================================


trait BasicRecipe:
    """Recipe that operates on source data alone."""
    @staticmethod
    def apply(mut ctx: QuantContext): ...

trait GammaRecipe:
    """Recipe that also requires an auxiliary gamma vector."""
    @staticmethod
    def apply(mut ctx: QuantContext, gamma: Ptr): ...


# ============================================================================
# Concrete recipes
# ============================================================================


struct Channelwise(BasicRecipe):
    @staticmethod
    def apply(mut ctx: QuantContext):
        compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


struct Hadamard[block: Int](BasicRecipe):
    @staticmethod
    def apply(mut ctx: QuantContext):
        fwht_rows_impl(ctx.f32, ctx.rows, ctx.cols, Self.block)
        compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


struct HadamardGamma[block: Int](GammaRecipe):
    @staticmethod
    def apply(mut ctx: QuantContext, gamma: Ptr):
        scale_columns_impl(ctx.f32, gamma, ctx.rows, ctx.cols)
        fwht_rows_impl(ctx.f32, ctx.rows, ctx.cols, Self.block)
        compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


# ============================================================================
# Weight types with recipe associations
# ============================================================================


trait Named:
    comptime NAME: StaticString

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait Quantizable: ...

struct Weight[
    rows: Int, cols: Int, name: StringLiteral,
    R: BasicRecipe,
](Named, Shaped, Quantizable):
    comptime NAME: StaticString = Self.name
    comptime ROWS = Self.rows
    comptime COLS = Self.cols

struct GammaWeight[
    rows: Int, cols: Int, name: StringLiteral,
    R: GammaRecipe,
](Named, Shaped, Quantizable):
    comptime NAME: StaticString = Self.name
    comptime ROWS = Self.rows
    comptime COLS = Self.cols


# ============================================================================
# Engine with trait-dispatched processing
# ============================================================================


struct QuantEngine:
    var f32_buf: Ptr
    var qi_buf: PtrI8
    var scale_buf: Ptr

    def __init__(out self, max_rows: Int, max_cols: Int):
        self.f32_buf = alloc[Float32](max_rows * max_cols)
        self.qi_buf = alloc[Scalar[DType.int8]](max_rows * max_cols)
        self.scale_buf = alloc[Float32](max_rows)

    def load(mut self, data: Ptr, rows: Int, cols: Int):
        for i in range(rows * cols):
            self.f32_buf[i] = data[i]

    def ctx(mut self, rows: Int, cols: Int) -> QuantContext:
        return QuantContext(self.f32_buf, self.qi_buf, self.scale_buf,
                            rows, cols)

    # Overload 1: BasicRecipe — no gamma needed
    def process[W: Named & Shaped, R: BasicRecipe](
        mut self, data: Ptr,
    ):
        self.load(data, W.ROWS, W.COLS)
        var c = self.ctx(W.ROWS, W.COLS)
        R.apply(c)
        print("  [basic]  " + String(W.NAME)
              + " [" + String(W.ROWS) + "x" + String(W.COLS) + "]")

    # Overload 2: GammaRecipe — gamma required
    def process[W: Named & Shaped, R: GammaRecipe](
        mut self, data: Ptr, gamma: Ptr,
    ):
        self.load(data, W.ROWS, W.COLS)
        var c = self.ctx(W.ROWS, W.COLS)
        R.apply(c, gamma)
        print("  [gamma]  " + String(W.NAME)
              + " [" + String(W.ROWS) + "x" + String(W.COLS) + "]")


# ============================================================================
# Test: explicit dispatch (what the model would do)
# ============================================================================


def test_explicit_dispatch():
    """Model explicitly calls the right overload per weight."""
    print("Design D: Explicit trait-dispatched overloads")

    comptime HIDDEN = 64
    comptime KV_HIDDEN = 32
    comptime INTERMEDIATE = 128
    comptime BLOCK = 16

    # Hadamard weight types
    comptime Q = GammaWeight[HIDDEN, HIDDEN, "q_proj", HadamardGamma[BLOCK]]
    comptime K = GammaWeight[KV_HIDDEN, HIDDEN, "k_proj", HadamardGamma[BLOCK]]
    comptime V = GammaWeight[KV_HIDDEN, HIDDEN, "v_proj", HadamardGamma[BLOCK]]
    comptime O = Weight[HIDDEN, HIDDEN, "o_proj", Hadamard[BLOCK]]
    comptime GATE = GammaWeight[INTERMEDIATE, HIDDEN, "gate_proj", HadamardGamma[BLOCK]]
    comptime UP = GammaWeight[INTERMEDIATE, HIDDEN, "up_proj", HadamardGamma[BLOCK]]
    comptime DOWN = Weight[HIDDEN, INTERMEDIATE, "down_proj", Hadamard[BLOCK]]

    var engine = QuantEngine(max_rows=INTERMEDIATE, max_cols=INTERMEDIATE)
    var src = alloc[Float32](INTERMEDIATE * INTERMEDIATE)
    for i in range(INTERMEDIATE * INTERMEDIATE):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    var gamma = alloc[Float32](HIDDEN)
    for k in range(HIDDEN):
        gamma[k] = 1.0

    # Gamma weights — must provide gamma (trait enforced)
    engine.process[Q, HadamardGamma[BLOCK]](src, gamma)
    engine.process[K, HadamardGamma[BLOCK]](src, gamma)
    engine.process[V, HadamardGamma[BLOCK]](src, gamma)

    # Non-gamma weights — no gamma argument (trait enforced)
    engine.process[O, Hadamard[BLOCK]](src)

    engine.process[GATE, HadamardGamma[BLOCK]](src, gamma)
    engine.process[UP, HadamardGamma[BLOCK]](src, gamma)
    engine.process[DOWN, Hadamard[BLOCK]](src)

    src.free()
    gamma.free()
    print("  Done\n")


# ============================================================================
# Test: channelwise model uses same engine, different recipes
# ============================================================================


def test_channelwise():
    print("Design D: Channelwise model")

    comptime HIDDEN = 64
    comptime Q = Weight[HIDDEN, HIDDEN, "q_proj", Channelwise]
    comptime O = Weight[HIDDEN, HIDDEN, "o_proj", Channelwise]

    var engine = QuantEngine(max_rows=HIDDEN, max_cols=HIDDEN)
    var src = alloc[Float32](HIDDEN * HIDDEN)
    for i in range(HIDDEN * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    engine.process[Q, Channelwise](src)
    engine.process[O, Channelwise](src)

    src.free()
    print("  Done\n")


def main():
    test_explicit_dispatch()
    test_channelwise()
