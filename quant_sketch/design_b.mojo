"""Design B: Tensor handle — ops are methods on a QuantTensor.

Session.read() returns a QuantTensor that carries buffer pointers and
dimensions. Ops are methods on QuantTensor, so the model never passes
rows/cols manually. Commit writes the result.

Model-side usage:

    var s = QuantSession.open(src, dst, weights)
    var gamma = s.read_1d("input_layernorm.weight")

    s.read("q_proj.weight")
        .absorb_gamma(gamma)
        .fwht(block=64)
        .quantize_int8()
        .commit("q_proj.weight", "q_proj.weight_scale")

    s.read("o_proj.weight")
        .fwht(block=64)
        .quantize_int8()
        .commit("o_proj.weight", "o_proj.weight_scale")

    s.passthrough("input_layernorm.weight")
    s.finalize()

Pro: Concise. Dimensions travel with the tensor. Reads like a pipeline.
Con: Method chaining in Mojo requires care with ownership/references.
     The handle is a reference to session-owned buffers, not a value type.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]


# ============================================================================
# Atomic ops (same as Design A — standalone, reusable)
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
            if a > amax:
                amax = a
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
# QuantTensor — a view into session-owned buffers with methods
# ============================================================================


struct QuantTensor:
    """A tensor being quantized. Carries buffer refs + dimensions.

    Not a value type — borrows session buffers. Methods mutate the
    underlying buffers and return self for chaining.
    """
    var f32: Ptr
    var qi: PtrI8
    var scales: Ptr
    var rows: Int
    var cols: Int
    var quantized: Bool

    def __init__(out self, f32: Ptr, qi: PtrI8, scales: Ptr,
                 rows: Int, cols: Int):
        self.f32 = f32
        self.qi = qi
        self.scales = scales
        self.rows = rows
        self.cols = cols
        self.quantized = False

    def absorb_gamma(mut self, gamma: Ptr):
        """Column-wise multiply by gamma vector."""
        scale_columns_op(self.f32, gamma, self.rows, self.cols)

    def fwht(mut self, block: Int):
        """Block-diagonal Walsh-Hadamard transform."""
        fwht_rows_op(self.f32, self.rows, self.cols, block)

    def quantize_int8(mut self):
        """Absmax scales + round-to-nearest int8."""
        compute_scales_op(self.f32, self.scales, self.rows, self.cols)
        quantize_int8_op(self.f32, self.scales, self.qi, self.rows, self.cols)
        self.quantized = True


# ============================================================================
# Session — creates QuantTensor handles from source data
# ============================================================================


struct QuantSession:
    var f32_buf: Ptr
    var qi_buf: PtrI8
    var scale_buf: Ptr
    var max_elements: Int
    var max_rows: Int

    def __init__(out self, max_rows: Int, max_cols: Int):
        self.max_elements = max_rows * max_cols
        self.max_rows = max_rows
        self.f32_buf = alloc[Float32](max_rows * max_cols)
        self.qi_buf = alloc[Scalar[DType.int8]](max_rows * max_cols)
        self.scale_buf = alloc[Float32](max_rows)

    def read(mut self, data: Ptr, rows: Int, cols: Int) -> QuantTensor:
        """Load source data into work buffer and return a handle."""
        for i in range(rows * cols):
            self.f32_buf[i] = data[i]
        return QuantTensor(self.f32_buf, self.qi_buf, self.scale_buf,
                           rows, cols)


# ============================================================================
# Example: Hadamard model — method-based composition
# ============================================================================


def hadamard_quantize_example():
    comptime HIDDEN = 576
    comptime KV_HIDDEN = 192
    comptime INTERMEDIATE = 1536
    comptime BLOCK = 64

    var s = QuantSession(max_rows=INTERMEDIATE, max_cols=max(HIDDEN, INTERMEDIATE))

    var input_gamma = alloc[Float32](HIDDEN)
    var post_attn_gamma = alloc[Float32](HIDDEN)
    for k in range(HIDDEN):
        input_gamma[k] = 1.0 + Float32(k) * 0.001
        post_attn_gamma[k] = 1.0 - Float32(k) * 0.001

    var src = alloc[Float32](INTERMEDIATE * HIDDEN)
    for i in range(INTERMEDIATE * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    # Q_PROJ: gamma + fwht + quantize
    var t = s.read(src, HIDDEN, HIDDEN)
    t.absorb_gamma(input_gamma)
    t.fwht(BLOCK)
    t.quantize_int8()
    # session.commit(t, "q_proj.weight", "q_proj.weight_scale")

    # O_PROJ: fwht + quantize (no gamma)
    t = s.read(src, HIDDEN, HIDDEN)
    t.fwht(BLOCK)
    t.quantize_int8()

    # GATE_PROJ: different gamma source
    t = s.read(src, INTERMEDIATE, HIDDEN)
    t.absorb_gamma(post_attn_gamma)
    t.fwht(BLOCK)
    t.quantize_int8()

    # DOWN_PROJ: no gamma, different K dim
    t = s.read(src, HIDDEN, INTERMEDIATE)
    t.fwht(BLOCK)
    t.quantize_int8()

    src.free()
    input_gamma.free()
    post_attn_gamma.free()
    print("Design B: hadamard quantize complete")


def channelwise_quantize_example():
    comptime HIDDEN = 576

    var s = QuantSession(max_rows=HIDDEN, max_cols=HIDDEN)
    var src = alloc[Float32](HIDDEN * HIDDEN)
    for i in range(HIDDEN * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    var t = s.read(src, HIDDEN, HIDDEN)
    t.quantize_int8()

    src.free()
    print("Design B: channelwise quantize complete")


def main():
    hadamard_quantize_example()
    channelwise_quantize_example()
