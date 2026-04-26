from std.collections import InlineArray
from std.memory import UnsafePointer

from gated_delta_net.gdn import GdnChunkArgs, gdn_chunk_kernel
from gated_delta_net.gdn_opt import gdn_opt_chunk_kernel


def fill_pattern(p: UnsafePointer[Float32, MutAnyOrigin], n: Int, base: Float32):
    for i in range(n):
        var x = base + Float32(i) * Float32(0.013)
        # Cheap deterministic spread to get non-uniform values.
        if (i & 1) == 1:
            x = -x * Float32(0.7)
        p[i] = x


def max_abs_diff(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    var m = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def main():
    comptime CHUNK = 64
    comptime DK = 128
    comptime DV = 128

    # Two independent input buffers — one for the reference, one for opt.
    var q_ref = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var k_ref = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var v_ref = InlineArray[Float32, CHUNK * DV](uninitialized=True)
    var b_ref = InlineArray[Float32, CHUNK](uninitialized=True)
    var g_ref = InlineArray[Float32, CHUNK](uninitialized=True)
    var s_ref = InlineArray[Float32, DK * DV](fill=Float32(0.001))
    var o_ref = InlineArray[Float32, CHUNK * DV](fill=Float32(0))

    var qrp = UnsafePointer(to=q_ref).bitcast[Float32]()
    var krp = UnsafePointer(to=k_ref).bitcast[Float32]()
    var vrp = UnsafePointer(to=v_ref).bitcast[Float32]()
    var brp = UnsafePointer(to=b_ref).bitcast[Float32]()
    var grp = UnsafePointer(to=g_ref).bitcast[Float32]()
    var srp = UnsafePointer(to=s_ref).bitcast[Float32]()
    var orp = UnsafePointer(to=o_ref).bitcast[Float32]()

    fill_pattern(qrp, CHUNK * DK, Float32(0.1))
    fill_pattern(krp, CHUNK * DK, Float32(0.07))
    fill_pattern(vrp, CHUNK * DV, Float32(0.05))
    for i in range(CHUNK):
        brp[i] = Float32(0.3) + Float32(i) * Float32(0.01)   # in (0, 1)
        grp[i] = Float32(-0.05) - Float32(i) * Float32(0.001)   # < 0

    # Mirror buffers for opt.
    var q_opt = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var k_opt = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var v_opt = InlineArray[Float32, CHUNK * DV](uninitialized=True)
    var b_opt = InlineArray[Float32, CHUNK](uninitialized=True)
    var g_opt = InlineArray[Float32, CHUNK](uninitialized=True)
    var s_opt = InlineArray[Float32, DK * DV](uninitialized=True)
    var o_opt = InlineArray[Float32, CHUNK * DV](fill=Float32(0))

    var qop = UnsafePointer(to=q_opt).bitcast[Float32]()
    var kop = UnsafePointer(to=k_opt).bitcast[Float32]()
    var vop = UnsafePointer(to=v_opt).bitcast[Float32]()
    var bop = UnsafePointer(to=b_opt).bitcast[Float32]()
    var gop = UnsafePointer(to=g_opt).bitcast[Float32]()
    var sop = UnsafePointer(to=s_opt).bitcast[Float32]()
    var oop = UnsafePointer(to=o_opt).bitcast[Float32]()

    for i in range(CHUNK * DK):
        qop[i] = qrp[i]
        kop[i] = krp[i]
    for i in range(CHUNK * DV):
        vop[i] = vrp[i]
    for i in range(CHUNK):
        bop[i] = brp[i]
        gop[i] = grp[i]
    for i in range(DK * DV):
        sop[i] = srp[i]

    var ref_args = GdnChunkArgs(
        q=qrp, k=krp, v=vrp, beta=brp, g=grp, state=srp, out=orp)
    gdn_chunk_kernel[CHUNK, DK, DV](ref_args, Float32(1e-6))

    var opt_args = GdnChunkArgs(
        q=qop, k=kop, v=vop, beta=bop, g=gop, state=sop, out=oop)
    gdn_opt_chunk_kernel[CHUNK, DK, DV](opt_args, Float32(1e-6))

    var out_diff = max_abs_diff(orp, oop, CHUNK * DV)
    var state_diff = max_abs_diff(srp, sop, DK * DV)

    print("max abs diff out  :", out_diff)
    print("max abs diff state:", state_diff)
    print("ref out[0]        :", orp[0])
    print("opt out[0]        :", oop[0])
    print("ref state[0]      :", srp[0])
    print("opt state[0]      :", sop[0])

    _ = q_ref; _ = k_ref; _ = v_ref; _ = b_ref; _ = g_ref; _ = s_ref; _ = o_ref
    _ = q_opt; _ = k_opt; _ = v_opt; _ = b_opt; _ = g_opt; _ = s_opt; _ = o_opt
