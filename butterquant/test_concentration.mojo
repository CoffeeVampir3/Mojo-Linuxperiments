"""Validate C(n) — the Hadamard concentration constant.

Three independent computations:
  1. Comptime FWHT Monte Carlo (from the model file)
  2. Runtime FWHT Monte Carlo (production fwht_block kernel)
  3. Direct sphere sampling (no FWHT — pure S^{n-1} max)

Method 3 exploits the fact that any orthonormal transform of a uniform
point on S^{n-1} is still uniform on S^{n-1}. So C(n) is really just
sqrt(n) * E[max_i |u_i|] for u ~ Uniform(S^{n-1}), regardless of the
specific orthogonal matrix. If methods 1-2 agree with method 3, the
FWHT is verified orthonormal in practice.

Analytical ground truth: C(2) = 4/pi (exactly computable).
Asymptotic bound: C(n) < sqrt(2 * ln(n)) (independent normal upper bound).
"""

from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
from experimental.hadquant_impl import fwht_block
from modeling.smollm2_butterquant_tp import concentration_constant, comptime_sqrt


# ============================================================================
# Runtime C(n) via production FWHT
# ============================================================================

def runtime_fwht_concentration[n: Int](num_samples: Int = 10000) -> Float64:
    """C(n) via the production FWHT kernel on real f32 buffers."""
    var state = UInt64(0xDEADBEEF12345678)
    var total = Float64(0)
    var sqrt_n = comptime_sqrt(Float64(n))

    var buf = alloc[Scalar[DType.float32]](n)

    for _ in range(num_samples):
        var norm_sq = Float64(0)
        for i in range(n):
            var val = Float64(0)
            for _ in range(12):
                state = state * 6364136223846793005 + 1442695040888963407
                val += Float64(state >> 11) / Float64(UInt64(1) << 53)
            val -= Float64(6)
            buf[i] = Float32(val)
            norm_sq += val * val

        var inv_norm = Float32(Float64(1) / comptime_sqrt(norm_sq))
        for i in range(n):
            buf[i] *= inv_norm

        fwht_block[DType.float32, n](buf)

        var absmax = Float64(0)
        for i in range(n):
            var a = Float64(buf[i])
            if a < Float64(0):
                a = -a
            if a > absmax:
                absmax = a

        total += sqrt_n * absmax

    buf.free()
    return total / Float64(num_samples)


# ============================================================================
# Direct sphere sampling (no FWHT)
# ============================================================================

def sphere_max_concentration[n: Int](num_samples: Int = 50000) -> Float64:
    """C(n) = sqrt(n) * E[max_i |u_i|] for u uniform on S^{n-1}.

    No Hadamard matrix involved. Uses a different PRNG seed to ensure
    statistical independence from the FWHT methods.
    """
    var state = UInt64(0xCAFEBABE87654321)  # Different seed
    var total = Float64(0)
    var sqrt_n = comptime_sqrt(Float64(n))

    for _ in range(num_samples):
        var norm_sq = Float64(0)
        var vec = InlineArray[Float64, n](fill=Float64(0))
        for i in range(n):
            var val = Float64(0)
            for _ in range(12):
                state = state * 6364136223846793005 + 1442695040888963407
                val += Float64(state >> 11) / Float64(UInt64(1) << 53)
            val -= Float64(6)
            vec[i] = val
            norm_sq += val * val

        var inv_norm = Float64(1) / comptime_sqrt(norm_sq)

        var absmax = Float64(0)
        for i in range(n):
            var a = vec[i] * inv_norm
            if a < Float64(0):
                a = -a
            if a > absmax:
                absmax = a

        total += sqrt_n * absmax

    return total / Float64(num_samples)


def fabs(x: Float64) -> Float64:
    if x < Float64(0):
        return -x
    return x


# ============================================================================
# Entry point
# ============================================================================

def main():
    # Analytical: C(2) = 4/pi
    var c2_exact = Float64(4) / 3.14159265358979323846
    var c2_sphere = sphere_max_concentration[2](num_samples=100000)

    print("=== Analytical ground truth ===")
    print("C(2) exact  = " + String(c2_exact))
    print("C(2) sphere = " + String(c2_sphere))
    print("delta       = " + String(fabs(c2_exact - c2_sphere)))
    if fabs(c2_exact - c2_sphere) < 0.02:
        print("OK")
    else:
        print("FAIL")

    # Asymptotic upper bound: C(n) < sqrt(2 * ln(n))
    # (holds because sphere constraint reduces max vs independent normals)
    print("\n=== Asymptotic bounds ===")
    var ln32 = 3.4657359  # ln(32)
    var ln64 = 4.1588830  # ln(64)
    var ln128 = 4.8520302 # ln(128)
    print("sqrt(2*ln(32))  = " + String(comptime_sqrt(Float64(2) * ln32)))
    print("sqrt(2*ln(64))  = " + String(comptime_sqrt(Float64(2) * ln64)))
    print("sqrt(2*ln(128)) = " + String(comptime_sqrt(Float64(2) * ln128)))

    # Three-way comparison for n = 32, 64, 128
    print("\n=== Three-way C(n) comparison ===")
    print("  n  |  comptime FWHT |  runtime FWHT |  direct sphere | asymptotic bound")
    print("  ---|---------------|---------------|----------------|------------------")

    var ct_32 = concentration_constant[32]()
    var ct_64 = concentration_constant[64]()
    var ct_128 = concentration_constant[128]()

    var rt_32 = runtime_fwht_concentration[32]()
    var rt_64 = runtime_fwht_concentration[64]()
    var rt_128 = runtime_fwht_concentration[128]()

    var sp_32 = sphere_max_concentration[32]()
    var sp_64 = sphere_max_concentration[64]()
    var sp_128 = sphere_max_concentration[128]()

    print("  32 |  " + String(ct_32) + "  |  " + String(rt_32)
        + "  |  " + String(sp_32)
        + "  |  " + String(comptime_sqrt(Float64(2) * ln32)))
    print("  64 |  " + String(ct_64) + "  |  " + String(rt_64)
        + "  |  " + String(sp_64)
        + "  |  " + String(comptime_sqrt(Float64(2) * ln64)))
    print(" 128 |  " + String(ct_128) + "  |  " + String(rt_128)
        + "  |  " + String(sp_128)
        + "  |  " + String(comptime_sqrt(Float64(2) * ln128)))

    # Validation
    var pass_all = True
    var impl_tol = 0.05   # comptime vs runtime (same seed)
    var method_tol = 0.05 # FWHT vs sphere (different seed, statistical variance)

    # Comptime vs runtime FWHT (should be near-identical, same seed)
    for pair in List((ct_32, rt_32, 32), (ct_64, rt_64, 64), (ct_128, rt_128, 128)):
        var delta = fabs(pair[][0] - pair[][1])
        if delta > impl_tol:
            print("FAIL: C(" + String(pair[][2]) + ") comptime/runtime delta=" + String(delta))
            pass_all = False

    # FWHT vs direct sphere (proves FWHT is orthonormal in practice)
    for pair in List((ct_32, sp_32, 32), (ct_64, sp_64, 64), (ct_128, sp_128, 128)):
        var delta = fabs(pair[][0] - pair[][1])
        if delta > method_tol:
            print("FAIL: C(" + String(pair[][2]) + ") FWHT/sphere delta=" + String(delta))
            pass_all = False

    # All values must be below asymptotic bound
    for pair in List((ct_32, ln32, 32), (ct_64, ln64, 64), (ct_128, ln128, 128)):
        var bound = comptime_sqrt(Float64(2) * pair[][1])
        if pair[][0] > bound + 0.01:
            print("FAIL: C(" + String(pair[][2]) + ")=" + String(pair[][0])
                + " exceeds asymptotic bound " + String(bound))
            pass_all = False

    print("\n=== Spec comparison (informational) ===")
    print("Spec claims: C(32)=3.18, C(64)=3.53, C(128)=3.85")
    print("These exceed the asymptotic bound sqrt(2*ln(n)) and are therefore incorrect.")
    print("Measured:    C(32)=" + String(ct_32) + ", C(64)=" + String(ct_64)
        + ", C(128)=" + String(ct_128))

    if pass_all:
        print("\nPASS")
    else:
        print("\nFAIL")
