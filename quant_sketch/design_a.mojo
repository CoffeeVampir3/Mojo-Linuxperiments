"""Design A: Flat procedural — session + free-function ops.

The model drives everything. Session handles I/O and buffer management.
Ops are standalone functions that take buffer pointers. The model composes
ops per-weight in whatever order it wants.

Model-side usage:

    var s = QuantSession.open(src, dst, weights, num_workers=4)
    var gamma = s.read_1d("model.layers.0.input_layernorm.weight")

    s.read("self_attn.q_proj.weight")       # bf16 → s.f32 work buffer
    scale_columns(s.pool, s.f32, gamma, rows, cols)
    fwht_rows(s.pool, s.f32, rows, cols, block=64)
    compute_scales(s.pool, s.f32, s.scales, rows, cols)
    quantize_int8(s.pool, s.f32, s.scales, s.qi, rows, cols)
    s.write_weight("...", rows, cols)
    s.write_scale("..._scale", rows)

    s.passthrough("input_layernorm.weight")
    s.finalize()

Pro: Maximum clarity. No hidden control flow. Trivially composable.
Con: Verbose. Buffer management (s.f32, s.qi, s.scales) is exposed.
     The model needs to remember to call things in order.
"""

from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


# ============================================================================
# Ops — standalone functions, no coupling to session or model
# ============================================================================


def scale_columns(f32: Ptr, gamma: Ptr, rows: Int, cols: Int):
    """Column-wise multiply: f32[r, k] *= gamma[k]. Gamma absorption."""
    for r in range(rows):
        for k in range(cols):
            f32[r * cols + k] = f32[r * cols + k] * gamma[k]


def fwht_rows(f32: Ptr, rows: Int, cols: Int, block: Int):
    """Block-diagonal FWHT on each row of f32[rows, cols]."""
    for r in range(rows):
        var base = f32 + r * cols
        for b in range(cols // block):
            fwht_inplace(base + b * block, block)


def fwht_inplace(buf: Ptr, n: Int):
    """In-place Walsh-Hadamard transform on n elements."""
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
    var sc = 1.0 / Float32(n)  # TODO: sqrt via SIMD
    for i in range(n):
        buf[i] = buf[i] * sc


def compute_scales(f32: Ptr, scales: Ptr, rows: Int, cols: Int):
    """Per-row absmax: scales[r] = max(|f32[r, :]|) / 127."""
    for r in range(rows):
        var amax = Float32(0)
        for k in range(cols):
            var v = f32[r * cols + k]
            var a = v if v >= 0 else -v
            if a > amax:
                amax = a
        scales[r] = amax / 127.0


def quantize_int8(f32: Ptr, scales: Ptr, qi: PtrI8, rows: Int, cols: Int):
    """Round-to-nearest int8: qi[r,k] = clamp(round(f32[r,k] / scale[r]))."""
    for r in range(rows):
        var sc = scales[r]
        var inv = 1.0 / sc if sc > 0 else Float32(0)
        for k in range(cols):
            var v = f32[r * cols + k] * inv
            # roundeven
            var rounded = Int(v + 0.5) if v >= 0 else -Int(-v + 0.5)
            if rounded > 127:
                rounded = 127
            elif rounded < -128:
                rounded = -128
            qi[r * cols + k] = Scalar[DType.int8](rounded)


# ============================================================================
# Session — I/O + buffer management, knows nothing about quantization math
# ============================================================================


struct QuantSession:
    """Manages source reading, output writing, and work buffers.

    The model creates a session, then calls ops + session.write in whatever
    order it wants. Session allocates work buffers sized to the largest
    tensor.
    """
    var f32: Ptr              # work buffer (rows * cols f32)
    var qi: PtrI8             # int8 output buffer
    var scales: Ptr           # per-row scales buffer
    var max_elements: Int     # max rows * cols across all weights
    var max_rows: Int

    def __init__(out self, max_rows: Int, max_cols: Int):
        self.max_elements = max_rows * max_cols
        self.max_rows = max_rows
        self.f32 = alloc[Float32](max_rows * max_cols)
        self.qi = alloc[Scalar[DType.int8]](max_rows * max_cols)
        self.scales = alloc[Float32](max_rows)

    def load_f32(mut self, data: Ptr, rows: Int, cols: Int):
        """Load data into the f32 work buffer (simulates bf16→f32 read)."""
        for i in range(rows * cols):
            self.f32[i] = data[i]

    # In real code: read_tensor, write_weight, write_scale, passthrough
    # would interact with io_uring + safetensors. Stubbed here.


# ============================================================================
# Example: Hadamard model quantize — the model drives composition
# ============================================================================


def hadamard_quantize_example():
    """What the Hadamard model's quantize() would look like."""
    comptime HIDDEN = 576
    comptime KV_HIDDEN = 192
    comptime INTERMEDIATE = 1536
    comptime NUM_LAYERS = 2  # abbreviated
    comptime BLOCK = 64

    var s = QuantSession(max_rows=INTERMEDIATE, max_cols=HIDDEN)

    # Simulate pre-loaded gamma vectors (one per layer)
    var input_gamma = alloc[Float32](HIDDEN)
    var post_attn_gamma = alloc[Float32](HIDDEN)
    for k in range(HIDDEN):
        input_gamma[k] = 1.0 + Float32(k) * 0.001
        post_attn_gamma[k] = 1.0 - Float32(k) * 0.001

    # Simulate source weight data
    var src = alloc[Float32](INTERMEDIATE * HIDDEN)
    for i in range(INTERMEDIATE * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    # --- Per-layer quantization, model-driven ---

    for layer in range(NUM_LAYERS):
        # Q_PROJ: gamma + fwht + quantize
        s.load_f32(src, HIDDEN, HIDDEN)
        scale_columns(s.f32, input_gamma, HIDDEN, HIDDEN)
        fwht_rows(s.f32, HIDDEN, HIDDEN, BLOCK)
        compute_scales(s.f32, s.scales, HIDDEN, HIDDEN)
        quantize_int8(s.f32, s.scales, s.qi, HIDDEN, HIDDEN)
        # s.write_weight("model.layers.N.self_attn.q_proj.weight", ...)
        # s.write_scale("model.layers.N.self_attn.q_proj.weight_scale", ...)

        # K_PROJ: same ops, different shape
        s.load_f32(src, KV_HIDDEN, HIDDEN)
        scale_columns(s.f32, input_gamma, KV_HIDDEN, HIDDEN)
        fwht_rows(s.f32, KV_HIDDEN, HIDDEN, BLOCK)
        compute_scales(s.f32, s.scales, KV_HIDDEN, HIDDEN)
        quantize_int8(s.f32, s.scales, s.qi, KV_HIDDEN, HIDDEN)

        # O_PROJ: fwht + quantize (no gamma)
        s.load_f32(src, HIDDEN, HIDDEN)
        fwht_rows(s.f32, HIDDEN, HIDDEN, BLOCK)
        compute_scales(s.f32, s.scales, HIDDEN, HIDDEN)
        quantize_int8(s.f32, s.scales, s.qi, HIDDEN, HIDDEN)

        # GATE_PROJ: different gamma source
        s.load_f32(src, INTERMEDIATE, HIDDEN)
        scale_columns(s.f32, post_attn_gamma, INTERMEDIATE, HIDDEN)
        fwht_rows(s.f32, INTERMEDIATE, HIDDEN, BLOCK)
        compute_scales(s.f32, s.scales, INTERMEDIATE, HIDDEN)
        quantize_int8(s.f32, s.scales, s.qi, INTERMEDIATE, HIDDEN)

        # DOWN_PROJ: no gamma, different K dimension
        s.load_f32(src, HIDDEN, INTERMEDIATE)
        fwht_rows(s.f32, HIDDEN, INTERMEDIATE, BLOCK)
        compute_scales(s.f32, s.scales, HIDDEN, INTERMEDIATE)
        quantize_int8(s.f32, s.scales, s.qi, HIDDEN, INTERMEDIATE)

    src.free()
    input_gamma.free()
    post_attn_gamma.free()

    print("Design A: hadamard quantize complete")


# ============================================================================
# Example: Channelwise model quantize — same ops, simpler subset
# ============================================================================


def channelwise_quantize_example():
    """What the channelwise model's quantize() would look like.
    Same ops, just no scale_columns or fwht_rows."""
    comptime HIDDEN = 576

    var s = QuantSession(max_rows=HIDDEN, max_cols=HIDDEN)

    var src = alloc[Float32](HIDDEN * HIDDEN)
    for i in range(HIDDEN * HIDDEN):
        src[i] = Float32(i % 17) * 0.01 - 0.08

    s.load_f32(src, HIDDEN, HIDDEN)
    compute_scales(s.f32, s.scales, HIDDEN, HIDDEN)
    quantize_int8(s.f32, s.scales, s.qi, HIDDEN, HIDDEN)

    src.free()
    print("Design A: channelwise quantize complete")


def main():
    hadamard_quantize_example()
    channelwise_quantize_example()
