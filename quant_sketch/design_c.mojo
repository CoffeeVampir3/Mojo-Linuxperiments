"""Design C: Recipe trait — per-weight quantization strategy at compile time.

Each weight carries a QuantRecipe trait that describes its quantization
steps. The quantizer dispatches via trait bounds. The model spec declares
the recipe per-weight, making the quantization scheme fully self-describing.

Model-side usage (in the model spec):

    comptime Q_PROJ = PlacedSlot[
        I8, RowShard, HIDDEN, HIDDEN, tp, ...,
        "self_attn.q_proj.weight",
        Recipe=HadamardGamma[64, "input_layernorm.weight"],
    ]
    comptime O_PROJ = PlacedSlot[
        I8, ColShard, HIDDEN, HIDDEN, tp, ...,
        "self_attn.o_proj.weight",
        Recipe=Hadamard[64],
    ]

    # Channelwise model would just use:
    comptime Q_PROJ = PlacedSlot[..., Recipe=Channelwise]

    # Quantization is then:
    def quantize(source, output):
        var engine = QuantEngine(source, output)
        Self.for_each_weight[engine.process_weight]()
        engine.finalize()

Pro: Fully declarative. Strategy is part of the model spec — self-describing.
     Adding a new strategy is just a new Recipe struct. The quantizer is
     generic and strategy-agnostic.
Con: Per-weight compile-time dispatch requires trait-level composition.
     Gamma association is a string baked into a type parameter — slightly
     rigid. Harder to debug than explicit procedural code.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]


# ============================================================================
# Atomic ops (same as A/B)
# ============================================================================


def scale_columns_op(f32: Ptr, gamma: Ptr, rows: Int, cols: Int):
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

def fwht_rows_op(f32: Ptr, rows: Int, cols: Int, block: Int):
    for r in range(rows):
        var base = f32 + r * cols
        for b in range(cols // block):
            fwht_inplace(base + b * block, block)

def compute_scales_op(f32: Ptr, scales: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        var amax = Float32(0)
        for k in range(cols):
            var v = f32[r * cols + k]
            var a = v if v >= 0 else -v
            if a > amax: amax = a
        scales[r] = amax / 127.0

def quantize_int8_op(f32: Ptr, scales: Ptr, qi: PtrI8, rows: Int, cols: Int):
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
# QuantContext — what the recipe receives (owned by the engine)
# ============================================================================


struct QuantContext:
    """Buffers + dimensions for a single tensor being quantized."""
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
# Recipe trait + concrete recipes
# ============================================================================


trait QuantRecipe:
    """Describes how to quantize a single weight tensor."""

    @staticmethod
    def apply(mut ctx: QuantContext, aux: Ptr): ...


struct Channelwise(QuantRecipe):
    """Plain channelwise: absmax + round."""

    @staticmethod
    def apply(mut ctx: QuantContext, aux: Ptr):
        compute_scales_op(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_op(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


struct Hadamard[block: Int](QuantRecipe):
    """FWHT + channelwise. No gamma absorption."""

    @staticmethod
    def apply(mut ctx: QuantContext, aux: Ptr):
        fwht_rows_op(ctx.f32, ctx.rows, ctx.cols, Self.block)
        compute_scales_op(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_op(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


struct HadamardGamma[block: Int](QuantRecipe):
    """Gamma absorption + FWHT + channelwise. aux = gamma vector."""

    @staticmethod
    def apply(mut ctx: QuantContext, aux: Ptr):
        scale_columns_op(ctx.f32, aux, ctx.rows, ctx.cols)
        fwht_rows_op(ctx.f32, ctx.rows, ctx.cols, Self.block)
        compute_scales_op(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
        quantize_int8_op(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


# ============================================================================
# Simulated PlacedSlot with Recipe parameter
# ============================================================================


trait Named:
    comptime NAME: StaticString

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait HasRecipe:
    comptime RECIPE_FN: def(mut QuantContext, Ptr) -> None

struct Weight[
    rows: Int, cols: Int, name: StringLiteral,
    R: QuantRecipe,
](Named, Shaped, HasRecipe):
    comptime NAME: StaticString = Self.name
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime RECIPE_FN = Self.R.apply


# ============================================================================
# Engine — generic quantizer that dispatches via recipe
# ============================================================================


struct QuantEngine:
    var f32_buf: Ptr
    var qi_buf: PtrI8
    var scale_buf: Ptr

    def __init__(out self, max_rows: Int, max_cols: Int):
        self.f32_buf = alloc[Float32](max_rows * max_cols)
        self.qi_buf = alloc[Scalar[DType.int8]](max_rows * max_cols)
        self.scale_buf = alloc[Float32](max_rows)

    def process[W: Named & Shaped & HasRecipe](
        mut self, data: Ptr, aux: Ptr,
    ):
        """Read source, apply recipe, produce qi + scales."""
        comptime rows = W.ROWS
        comptime cols = W.COLS
        for i in range(rows * cols):
            self.f32_buf[i] = data[i]
        var ctx = QuantContext(
            self.f32_buf, self.qi_buf, self.scale_buf, rows, cols,
        )
        W.RECIPE_FN(ctx, aux)
        print("  Processed: " + String(W.NAME)
              + " [" + String(rows) + "x" + String(cols) + "]")


# ============================================================================
# Example: Model spec with recipes
# ============================================================================


def hadamard_model_example():
    """Hadamard model — recipes baked into weight types."""
    print("Design C: Hadamard model")

    comptime HIDDEN = 64   # small for test
    comptime KV_HIDDEN = 32
    comptime INTERMEDIATE = 128
    comptime BLOCK = 16

    # Weight types with recipes
    comptime Q = Weight[HIDDEN, HIDDEN, "q_proj", HadamardGamma[BLOCK]]
    comptime K = Weight[KV_HIDDEN, HIDDEN, "k_proj", HadamardGamma[BLOCK]]
    comptime V = Weight[KV_HIDDEN, HIDDEN, "v_proj", HadamardGamma[BLOCK]]
    comptime O = Weight[HIDDEN, HIDDEN, "o_proj", Hadamard[BLOCK]]
    comptime GATE = Weight[INTERMEDIATE, HIDDEN, "gate_proj", HadamardGamma[BLOCK]]
    comptime UP = Weight[INTERMEDIATE, HIDDEN, "up_proj", HadamardGamma[BLOCK]]
    comptime DOWN = Weight[HIDDEN, INTERMEDIATE, "down_proj", Hadamard[BLOCK]]

    var engine = QuantEngine(max_rows=INTERMEDIATE, max_cols=max(HIDDEN, INTERMEDIATE))

    # Simulate source data
    var src = alloc[Float32](INTERMEDIATE * max(HIDDEN, INTERMEDIATE))
    for i in range(INTERMEDIATE * INTERMEDIATE):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    # Simulate gamma
    var gamma = alloc[Float32](HIDDEN)
    for k in range(HIDDEN):
        gamma[k] = 1.0 + Float32(k) * 0.01
    var null_gamma = Ptr()

    # Quantize — engine dispatches via recipe trait
    engine.process[Q](src, gamma)
    engine.process[K](src, gamma)
    engine.process[V](src, gamma)
    engine.process[O](src, null_gamma)
    engine.process[GATE](src, gamma)
    engine.process[UP](src, gamma)
    engine.process[DOWN](src, null_gamma)

    src.free()
    gamma.free()
    print("  Done")


def channelwise_model_example():
    """Channelwise model — same engine, Channelwise recipe."""
    print("Design C: Channelwise model")

    comptime HIDDEN = 64
    comptime Q = Weight[HIDDEN, HIDDEN, "q_proj", Channelwise]
    comptime O = Weight[HIDDEN, HIDDEN, "o_proj", Channelwise]

    var engine = QuantEngine(max_rows=HIDDEN, max_cols=HIDDEN)
    var src = alloc[Float32](HIDDEN * HIDDEN)
    for i in range(HIDDEN * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    engine.process[Q](src, Ptr())
    engine.process[O](src, Ptr())

    src.free()
    print("  Done")


def main():
    hadamard_model_example()
    channelwise_model_example()
