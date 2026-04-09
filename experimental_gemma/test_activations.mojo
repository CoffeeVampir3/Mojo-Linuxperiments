"""Validate gelu_tanh_mul against torch reference values.

Reference computed with:
  import torch; torch.nn.functional.gelu(torch.tensor([x]), approximate='tanh')
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs

from modeling.model_spec import BF16, Slot, Replicated, DynView
from experimental_gemma.activations import gelu_tanh_mul, gelu_tanh_f32


def test_gelu_tanh_scalar():
    print("=== gelu_tanh_f32 vs torch reference (f32) ===")
    print("  x          | got          | torch        | abs_err")
    print("  -----------+--------------+--------------+---------")

    var inp = InlineArray[Float32, 8](fill=Float32(0))
    var torch_val = InlineArray[Float32, 8](fill=Float32(0))
    inp[0] = -3.0;  torch_val[0] = -0.00405
    inp[1] = -1.0;  torch_val[1] = -0.15880
    inp[2] = -0.5;  torch_val[2] = -0.15426
    inp[3] =  0.0;  torch_val[3] =  0.0
    inp[4] =  0.5;  torch_val[4] =  0.34574
    inp[5] =  1.0;  torch_val[5] =  0.84120
    inp[6] =  2.0;  torch_val[6] =  1.95460
    inp[7] =  3.0;  torch_val[7] =  2.99595

    var max_err = Float64(0)
    var sum_sq_err = Float64(0)
    for i in range(8):
        var x = SIMD[DType.float32, 1](inp[i])
        var got = gelu_tanh_f32(x)
        var err = abs(Float64(got[0]) - Float64(torch_val[i]))
        if err > max_err:
            max_err = err
        sum_sq_err += err * err
        print("  " + String(inp[i]) + " | " + String(got[0]) + " | " + String(torch_val[i]) + " | " + String(err))

    print("  max_err=" + String(max_err) + "  rmse=" + String(sum_sq_err / 8.0))
    print()


def test_gelu_tanh_mul_bf16():
    print("=== gelu_tanh_mul bf16 error decomposition (up=1.0) ===")
    print("  gate(bf16) | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err     | total_err")
    print("  -----------+--------------+--------------+--------------+--------------+--------------+----------")

    comptime COLS = 16
    comptime View = Slot[BF16, Replicated, 1, COLS, 1]

    var gate = alloc[Scalar[DType.bfloat16]](COLS)
    var up = alloc[Scalar[DType.bfloat16]](COLS)
    var dst = alloc[Scalar[DType.bfloat16]](COLS)

    for i in range(COLS):
        gate[i] = Scalar[DType.bfloat16](Float32(-2.0) + Float32(i) * Float32(0.5))
        up[i] = Scalar[DType.bfloat16](1.0)
        dst[i] = Scalar[DType.bfloat16](0.0)

    var gate_v = DynView[View](Int(gate), 1)
    var up_v = DynView[View](Int(up), 1)
    var dst_v = DynView[View](Int(dst), 1)

    gelu_tanh_mul(gate_v, up_v, dst_v)

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    var max_total_err = Float64(0)

    for i in range(COLS):
        var g = Float32(gate[i])
        var d = Float32(dst[i])

        # f32 reference: gelu_tanh computed in f32 from the bf16-quantized input
        var f32_result = gelu_tanh_f32(SIMD[DType.float32, 1](g))
        var f32val = f32_result[0]

        # bf16-quantized reference: what the f32 result becomes after bf16 round-trip
        var bf16_quantized = Float32(Scalar[DType.bfloat16](f32val))

        # kernel error: kernel output vs bf16-quantized f32 reference
        # this isolates computational error from representation error
        var kern_err = abs(Float64(d) - Float64(bf16_quantized))

        # bf16 error: loss from f32 → bf16 quantization
        var bf16_err = abs(Float64(f32val) - Float64(bf16_quantized))

        # total error: kernel output vs f32 reference
        var total_err = abs(Float64(d) - Float64(f32val))

        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        if total_err > max_total_err:
            max_total_err = total_err

        print("  " + String(g) + " | " + String(d) + " | " + String(bf16_quantized) + " | " + String(f32val) + " | " + String(kern_err) + " | " + String(bf16_err) + " | " + String(total_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err) + "  max total_err=" + String(max_total_err))
    print()

    # Scaled test
    print("=== gelu_tanh_mul bf16 error decomposition (varying up) ===")
    print("  gate | up   | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err     | total_err")
    print("  -----+------+--------------+--------------+--------------+--------------+--------------+----------")

    max_kern_err = Float64(0)
    max_bf16_err = Float64(0)
    max_total_err = Float64(0)

    for i in range(COLS):
        up[i] = Scalar[DType.bfloat16](Float32(0.5) + Float32(i) * Float32(0.1))
        dst[i] = Scalar[DType.bfloat16](0.0)

    gelu_tanh_mul(gate_v, up_v, dst_v)

    for i in range(COLS):
        var g = Float32(gate[i])
        var u = Float32(up[i])
        var d = Float32(dst[i])

        var gelu_result = gelu_tanh_f32(SIMD[DType.float32, 1](g))
        var f32val = gelu_result[0] * u
        var bf16_quantized = Float32(Scalar[DType.bfloat16](f32val))

        var kern_err = abs(Float64(d) - Float64(bf16_quantized))
        var bf16_err = abs(Float64(f32val) - Float64(bf16_quantized))
        var total_err = abs(Float64(d) - Float64(f32val))

        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        if total_err > max_total_err:
            max_total_err = total_err

        print("  " + String(g) + " | " + String(u) + " | " + String(d) + " | " + String(bf16_quantized) + " | " + String(f32val) + " | " + String(kern_err) + " | " + String(bf16_err) + " | " + String(total_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err) + "  max total_err=" + String(max_total_err))

    gate.free()
    up.free()
    dst.free()


def main():
    test_gelu_tanh_scalar()
    test_gelu_tanh_mul_bf16()
