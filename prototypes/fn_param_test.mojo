from std.sys.info import simd_width_of


@always_inline
def silu_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return x * (Float32(1.0) / (Float32(1.0) + x))


@always_inline
def gelu_f32[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return Float32(0.5) * x * x


def caller[
    width: Int,
    act_fn: def[w: Int](SIMD[DType.float32, w]) thin -> SIMD[DType.float32, w],
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return act_fn[width](x)


def main():
    var x = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var y1 = caller[4, silu_f32](x)
    var y2 = caller[4, gelu_f32](x)
    print("silu:", y1)
    print("gelu:", y2)
