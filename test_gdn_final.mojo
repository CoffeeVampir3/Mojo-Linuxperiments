from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns

from gated_delta_net.gdn import GdnChunkArgs, gdn_chunk_kernel
from gated_delta_net.gdn_opt import gdn_opt_chunk_kernel
from gated_delta_net.gdn_final import gdn_final_chunk_kernel


def fill_pattern(p: UnsafePointer[Float32, MutAnyOrigin], n: Int, base: Float32):
    for i in range(n):
        var x = base + Float32(i) * Float32(0.013)
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


def init_inputs(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    g: UnsafePointer[Float32, MutAnyOrigin],
    s: UnsafePointer[Float32, MutAnyOrigin],
    chunk: Int, dk: Int, dv: Int,
):
    fill_pattern(q, chunk * dk, Float32(0.1))
    fill_pattern(k, chunk * dk, Float32(0.07))
    fill_pattern(v, chunk * dv, Float32(0.05))
    for i in range(chunk):
        b[i] = Float32(0.3) + Float32(i) * Float32(0.01)
        g[i] = Float32(-0.05) - Float32(i) * Float32(0.001)
    for i in range(dk * dv):
        s[i] = Float32(0.001)


def main():
    comptime CHUNK = 64
    comptime DK = 128
    comptime DV = 128
    comptime ITERS = 200

    # Three sets of buffers — one per kernel.
    var q_a = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var k_a = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var v_a = InlineArray[Float32, CHUNK * DV](uninitialized=True)
    var b_a = InlineArray[Float32, CHUNK](uninitialized=True)
    var g_a = InlineArray[Float32, CHUNK](uninitialized=True)
    var s_a = InlineArray[Float32, DK * DV](uninitialized=True)
    var o_a = InlineArray[Float32, CHUNK * DV](fill=Float32(0))

    var q_b = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var k_b = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var v_b = InlineArray[Float32, CHUNK * DV](uninitialized=True)
    var b_b = InlineArray[Float32, CHUNK](uninitialized=True)
    var g_b = InlineArray[Float32, CHUNK](uninitialized=True)
    var s_b = InlineArray[Float32, DK * DV](uninitialized=True)
    var o_b = InlineArray[Float32, CHUNK * DV](fill=Float32(0))

    var q_c = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var k_c = InlineArray[Float32, CHUNK * DK](uninitialized=True)
    var v_c = InlineArray[Float32, CHUNK * DV](uninitialized=True)
    var b_c = InlineArray[Float32, CHUNK](uninitialized=True)
    var g_c = InlineArray[Float32, CHUNK](uninitialized=True)
    var s_c = InlineArray[Float32, DK * DV](uninitialized=True)
    var o_c = InlineArray[Float32, CHUNK * DV](fill=Float32(0))

    var qa = UnsafePointer(to=q_a).bitcast[Float32]()
    var ka = UnsafePointer(to=k_a).bitcast[Float32]()
    var va = UnsafePointer(to=v_a).bitcast[Float32]()
    var ba = UnsafePointer(to=b_a).bitcast[Float32]()
    var ga = UnsafePointer(to=g_a).bitcast[Float32]()
    var sa = UnsafePointer(to=s_a).bitcast[Float32]()
    var oa = UnsafePointer(to=o_a).bitcast[Float32]()

    var qb = UnsafePointer(to=q_b).bitcast[Float32]()
    var kb = UnsafePointer(to=k_b).bitcast[Float32]()
    var vb = UnsafePointer(to=v_b).bitcast[Float32]()
    var bb = UnsafePointer(to=b_b).bitcast[Float32]()
    var gb = UnsafePointer(to=g_b).bitcast[Float32]()
    var sb = UnsafePointer(to=s_b).bitcast[Float32]()
    var ob = UnsafePointer(to=o_b).bitcast[Float32]()

    var qc = UnsafePointer(to=q_c).bitcast[Float32]()
    var kc = UnsafePointer(to=k_c).bitcast[Float32]()
    var vc = UnsafePointer(to=v_c).bitcast[Float32]()
    var bc = UnsafePointer(to=b_c).bitcast[Float32]()
    var gc = UnsafePointer(to=g_c).bitcast[Float32]()
    var sc = UnsafePointer(to=s_c).bitcast[Float32]()
    var oc = UnsafePointer(to=o_c).bitcast[Float32]()

    init_inputs(qa, ka, va, ba, ga, sa, CHUNK, DK, DV)
    init_inputs(qb, kb, vb, bb, gb, sb, CHUNK, DK, DV)
    init_inputs(qc, kc, vc, bc, gc, sc, CHUNK, DK, DV)

    gdn_chunk_kernel[CHUNK, DK, DV](
        GdnChunkArgs(q=qa, k=ka, v=va, beta=ba, g=ga, state=sa, out=oa),
        Float32(1e-6))
    gdn_opt_chunk_kernel[CHUNK, DK, DV](
        GdnChunkArgs(q=qb, k=kb, v=vb, beta=bb, g=gb, state=sb, out=ob),
        Float32(1e-6))
    gdn_final_chunk_kernel[CHUNK, DK, DV](
        GdnChunkArgs(q=qc, k=kc, v=vc, beta=bc, g=gc, state=sc, out=oc),
        Float32(1e-6))

    print("=== Parity vs reference (gdn.mojo) ===")
    print("opt   max diff out  :", max_abs_diff(oa, ob, CHUNK * DV))
    print("opt   max diff state:", max_abs_diff(sa, sb, DK * DV))
    print("final max diff out  :", max_abs_diff(oa, oc, CHUNK * DV))
    print("final max diff state:", max_abs_diff(sa, sc, DK * DV))
    print("ref   out[0]        :", oa[0])
    print("opt   out[0]        :", ob[0])
    print("final out[0]        :", oc[0])
    print("ref   state[0]      :", sa[0])
    print("final state[0]      :", sc[0])

    # ---- Wall-clock comparison ----
    # Warm
    for _ in range(2):
        init_inputs(qa, ka, va, ba, ga, sa, CHUNK, DK, DV)
        gdn_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qa, k=ka, v=va, beta=ba, g=ga, state=sa, out=oa),
            Float32(1e-6))
    for _ in range(2):
        init_inputs(qb, kb, vb, bb, gb, sb, CHUNK, DK, DV)
        gdn_opt_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qb, k=kb, v=vb, beta=bb, g=gb, state=sb, out=ob),
            Float32(1e-6))
    for _ in range(2):
        init_inputs(qc, kc, vc, bc, gc, sc, CHUNK, DK, DV)
        gdn_final_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qc, k=kc, v=vc, beta=bc, g=gc, state=sc, out=oc),
            Float32(1e-6))

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        init_inputs(qa, ka, va, ba, ga, sa, CHUNK, DK, DV)
        gdn_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qa, k=ka, v=va, beta=ba, g=ga, state=sa, out=oa),
            Float32(1e-6))
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        init_inputs(qb, kb, vb, bb, gb, sb, CHUNK, DK, DV)
        gdn_opt_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qb, k=kb, v=vb, beta=bb, g=gb, state=sb, out=ob),
            Float32(1e-6))
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        init_inputs(qc, kc, vc, bc, gc, sc, CHUNK, DK, DV)
        gdn_final_chunk_kernel[CHUNK, DK, DV](
            GdnChunkArgs(q=qc, k=kc, v=vc, beta=bc, g=gc, state=sc, out=oc),
            Float32(1e-6))
    var t3 = perf_counter_ns()

    var t_setup_start = perf_counter_ns()
    for _ in range(ITERS):
        init_inputs(qa, ka, va, ba, ga, sa, CHUNK, DK, DV)
    var t_setup = perf_counter_ns() - t_setup_start

    var ref_ns = Float64(t1 - t0 - t_setup) / Float64(ITERS)
    var opt_ns = Float64(t2 - t1 - t_setup) / Float64(ITERS)
    var fin_ns = Float64(t3 - t2 - t_setup) / Float64(ITERS)
    print("=== Per-chunk wall clock (ns) ===")
    print("ref  :", ref_ns)
    print("opt  :", opt_ns, "  speedup vs ref:", ref_ns / opt_ns)
    print("final:", fin_ns, "  speedup vs ref:", ref_ns / fin_ns)
    print("                speedup vs opt:", opt_ns / fin_ns)

    _ = q_a; _ = k_a; _ = v_a; _ = b_a; _ = g_a; _ = s_a; _ = o_a
    _ = q_b; _ = k_b; _ = v_b; _ = b_b; _ = g_b; _ = s_b; _ = o_b
    _ = q_c; _ = k_c; _ = v_c; _ = b_c; _ = g_c; _ = s_c; _ = o_c
