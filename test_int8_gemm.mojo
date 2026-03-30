"""Test int8_gemm against scalar reference."""

from std.memory.unsafe_pointer import alloc
from std.memory import memcpy

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    BF16, F32, I8, Replicated,
    IsQuantizable, IsPassthrough,
    Slot, PlacedSlot, Bound, DynView,
    Kernel3DTiling,
)
from kernels.vnni import VnniPacked, pack_vnni
from experimental.hadquant_impl import int8_gemm
from threading import BurstPool
from numa import NumaInfo


def test_gemm[N: Int, K: Int, M: Int]():
    """Test int8_gemm [M, K] × [N, K]^T → [M, N] against scalar reference."""
    print("--- test_gemm M=" + String(M) + " N=" + String(N) + " K=" + String(K) + " ---")

    # --- Allocate ---
    var weight_raw = alloc[Scalar[DType.int8]](N * K)
    var weight_packed = alloc[UInt8](N * K)
    var activation = alloc[Scalar[DType.int8]](M * K)
    var act_scale = alloc[Float32](M)
    var w_scale = alloc[Float32](N)
    var colsum = alloc[Float32](N)
    var output = alloc[Scalar[DType.bfloat16]](M * N)
    var expected = alloc[Float32](M * N)

    # --- Fill test data ---
    for n in range(N):
        for k in range(K):
            weight_raw[n * K + k] = Scalar[DType.int8]((n * 7 + k * 3) % 251 - 125)

    for m in range(M):
        for k in range(K):
            activation[m * K + k] = Scalar[DType.int8]((m * 13 + k * 5) % 251 - 125)
        act_scale[m] = Float32(0.01) + Float32(m) * Float32(0.005)

    for n in range(N):
        w_scale[n] = Float32(0.02) + Float32(n) * Float32(0.001)

    # --- Column sums (before packing) ---
    for n in range(N):
        var s = Int(0)
        for k in range(K):
            s += Int(weight_raw[n * K + k])
        colsum[n] = Float32(s)

    # --- VNNI pack ---
    pack_vnni(
        weight_raw.bitcast[UInt8](),
        weight_packed,
        N, K,
    )

    # --- Reference: u8 × i8 dot product + epilogue ---
    for m in range(M):
        for n in range(N):
            var raw = Int(0)
            for k in range(K):
                var a_u8 = Int(activation[m * K + k]) + 128
                var w_i8 = Int(weight_raw[n * K + k])
                raw += a_u8 * w_i8
            var corrected = Float32(raw) - Float32(128.0) * colsum[n]
            expected[m * N + n] = corrected * act_scale[m] * w_scale[n]

    # --- Run kernel ---
    comptime ActSlot = Slot[I8, Replicated, 1, K, 1]
    comptime ScSlot = Slot[F32, Replicated, 1, 1, 1]
    comptime OutSlot = Slot[BF16, Replicated, 1, N, 1]
    comptime WSlot = PlacedSlot[I8, Replicated, N, K, 1, 0, "w", IsQuantizable, VnniPacked, Kernel3DTiling[32, 64, 32]]
    comptime WsSlot = PlacedSlot[F32, Replicated, N, 1, 1, 0, "ws"]
    comptime CsSlot = PlacedSlot[F32, Replicated, N, 1, 1, 0, "cs"]

    var act_view = DynView[ActSlot](Int(activation), M)
    var sc_view = DynView[ScSlot](Int(act_scale), M)
    var out_view = DynView[OutSlot](Int(output), M)
    var w_bound = Bound[WSlot](Int(weight_packed))
    var ws_bound = Bound[WsSlot](Int(w_scale))
    var cs_bound = Bound[CsSlot](Int(colsum))

    var numa = NumaInfo()
    var pool = BurstPool[].for_numa_node(numa, 0)
    int8_gemm(act_view, sc_view, w_bound, ws_bound, cs_bound, out_view, pool).join()

    # --- Compare ---
    var max_abs_err = Float64(0)
    var max_rel_err = Float64(0)
    var fail_count = 0

    for m in range(M):
        for n in range(N):
            var got = Float64(Float32(output[m * N + n]))
            var exp = Float64(expected[m * N + n])
            var abs_err = got - exp
            if abs_err < 0:
                abs_err = -abs_err
            if abs_err > max_abs_err:
                max_abs_err = abs_err

            var denom = exp if exp >= 0 else -exp
            if denom > 1e-6:
                var rel = abs_err / denom
                if rel > max_rel_err:
                    max_rel_err = rel

            # bf16 has ~0.4% precision, allow 1% for accumulated error
            var tol = Float64(0.01) * denom + Float64(0.001)
            if abs_err > tol:
                if fail_count < 5:
                    print("  FAIL [" + String(m) + "," + String(n)
                          + "] got=" + String(got) + " exp=" + String(exp)
                          + " err=" + String(abs_err))
                fail_count += 1

    if fail_count == 0:
        print("  PASS all " + String(M * N) + " elements")
    else:
        print("  FAIL " + String(fail_count) + "/" + String(M * N) + " elements")
    print("  max_abs_err=" + String(max_abs_err) + " max_rel_err=" + String(max_rel_err))

    # Cleanup
    weight_raw.free()
    weight_packed.free()
    activation.free()
    act_scale.free()
    w_scale.free()
    colsum.free()
    output.free()
    expected.free()


def main():
    # Minimal: one N tile, one K tile
    test_gemm[32, 64, 1]()

    # Decode-sized: one row, full SmolLM2 Q projection dimensions
    test_gemm[576, 576, 1]()

    # Multi-row prefill
    test_gemm[576, 576, 4]()

    # KV projection (smaller N)
    test_gemm[192, 576, 2]()

    print("\nDone.")
