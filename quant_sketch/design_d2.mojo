"""Design D2: Recipe carried by the weight, engine extracts it.

Refines D: the weight type carries its recipe via a comptime function
pointer field. The engine dispatches via conforms_to on the weight type
itself — no redundant type parameter at the call site.

Model-side:

    engine.process[Q](src, gamma)     # Q conforms to GammaQuantizable
    engine.process[O](src)            # O conforms to Quantizable only

The weight type determines which overload is valid. Providing gamma for
a weight that doesn't need it, or omitting it for one that does, is a
compile error.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]


# ============================================================================
# Atomic ops (same)
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
    for i in range(n): buf[i] = buf[i] * sc

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
# Context
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
# Recipe function types + requirement traits
# ============================================================================

comptime BasicApplyFn = def(mut QuantContext) -> None
comptime GammaApplyFn = def(mut QuantContext, Ptr) -> None


trait Named:
    comptime NAME: StaticString

trait Shaped:
    comptime ROWS: Int
    comptime COLS: Int

trait Quantizable(Named, Shaped):
    """Weight that can be quantized without auxiliary data."""
    comptime APPLY: BasicApplyFn

trait GammaQuantizable(Named, Shaped):
    """Weight that requires a gamma vector for quantization."""
    comptime APPLY: GammaApplyFn


# ============================================================================
# Recipes — just functions matching the apply signatures
# ============================================================================


def channelwise_apply(mut ctx: QuantContext):
    compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
    quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


def hadamard_apply[block: Int](mut ctx: QuantContext):
    fwht_rows_impl(ctx.f32, ctx.rows, ctx.cols, block)
    compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
    quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


def hadamard_gamma_apply[block: Int](mut ctx: QuantContext, gamma: Ptr):
    scale_columns_impl(ctx.f32, gamma, ctx.rows, ctx.cols)
    fwht_rows_impl(ctx.f32, ctx.rows, ctx.cols, block)
    compute_scales_impl(ctx.f32, ctx.scales, ctx.rows, ctx.cols)
    quantize_int8_impl(ctx.f32, ctx.scales, ctx.qi, ctx.rows, ctx.cols)


# ============================================================================
# Weight types — recipe is a comptime field, no separate Recipe type needed
# ============================================================================


struct QWeight[
    rows: Int, cols: Int, name: StringLiteral,
    apply_fn: BasicApplyFn,
](Quantizable):
    comptime NAME: StaticString = Self.name
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime APPLY = Self.apply_fn


struct QGammaWeight[
    rows: Int, cols: Int, name: StringLiteral,
    apply_fn: GammaApplyFn,
](GammaQuantizable):
    comptime NAME: StaticString = Self.name
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime APPLY = Self.apply_fn


# ============================================================================
# Engine — overloaded on trait conformance of W
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

    def process[W: Quantizable](mut self, data: Ptr):
        """Quantize a weight that needs no auxiliary data."""
        self.load(data, W.ROWS, W.COLS)
        var ctx = QuantContext(self.f32_buf, self.qi_buf, self.scale_buf,
                               W.ROWS, W.COLS)
        W.APPLY(ctx)
        print("  [basic]  " + String(W.NAME)
              + " [" + String(W.ROWS) + "x" + String(W.COLS) + "]")

    def process[W: GammaQuantizable](mut self, data: Ptr, gamma: Ptr):
        """Quantize a weight that requires gamma absorption."""
        self.load(data, W.ROWS, W.COLS)
        var ctx = QuantContext(self.f32_buf, self.qi_buf, self.scale_buf,
                               W.ROWS, W.COLS)
        W.APPLY(ctx, gamma)
        print("  [gamma]  " + String(W.NAME)
              + " [" + String(W.ROWS) + "x" + String(W.COLS) + "]")


# ============================================================================
# Example: Hadamard model
# ============================================================================


def test_hadamard():
    print("Design D2: Hadamard model")
    comptime H = 64
    comptime KV = 32
    comptime I = 128
    comptime B = 16

    # Weight declarations — recipe baked in
    comptime Q      = QGammaWeight[H,  H, "q_proj",    hadamard_gamma_apply[B]]
    comptime K_P    = QGammaWeight[KV, H, "k_proj",    hadamard_gamma_apply[B]]
    comptime V      = QGammaWeight[KV, H, "v_proj",    hadamard_gamma_apply[B]]
    comptime O      = QWeight[H,  H, "o_proj",         hadamard_apply[B]]
    comptime GATE   = QGammaWeight[I,  H, "gate_proj", hadamard_gamma_apply[B]]
    comptime UP     = QGammaWeight[I,  H, "up_proj",   hadamard_gamma_apply[B]]
    comptime DOWN   = QWeight[H,  I, "down_proj",      hadamard_apply[B]]

    var engine = QuantEngine(max_rows=I, max_cols=I)
    var src = alloc[Float32](I * I)
    var gamma = alloc[Float32](H)
    for i in range(I * I): src[i] = Float32(i % 17) * 0.01
    for k in range(H): gamma[k] = 1.0

    # Call site: W determines which overload, gamma presence enforced
    engine.process[Q](src, gamma)
    engine.process[K_P](src, gamma)
    engine.process[V](src, gamma)
    engine.process[O](src)              # no gamma — correct
    engine.process[GATE](src, gamma)
    engine.process[UP](src, gamma)
    engine.process[DOWN](src)           # no gamma — correct

    # These would be compile errors:
    # engine.process[Q](src)            # missing gamma for GammaQuantizable
    # engine.process[O](src, gamma)     # extra gamma for Quantizable

    src.free()
    gamma.free()
    print("  Done\n")


def test_channelwise():
    print("Design D2: Channelwise model")
    comptime H = 64

    comptime Q = QWeight[H, H, "q_proj", channelwise_apply]
    comptime O = QWeight[H, H, "o_proj", channelwise_apply]

    var engine = QuantEngine(max_rows=H, max_cols=H)
    var src = alloc[Float32](H * H)
    for i in range(H * H): src[i] = Float32(i % 17) * 0.01

    engine.process[Q](src)
    engine.process[O](src)

    src.free()
    print("  Done\n")


def main():
    test_hadamard()
    test_channelwise()
