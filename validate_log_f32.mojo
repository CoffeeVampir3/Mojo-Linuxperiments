"""Measure simd_math.log_f32 against std.math.log (Float64 reference).

Run: pixi run mojo -I . validate_log_f32.mojo
"""

from std.math import log as ref_log
from std.collections import List

from simd_math import log_f32


@always_inline
def scalar_log(x: Float32) -> Float32:
    return log_f32[1](SIMD[DType.float32, 1](x))[0]


@always_inline
def abs_f64(x: Float64) -> Float64:
    return x if x >= Float64(0) else -x


@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable, Movable):
    var max_abs: Float64
    var max_abs_x: Float32
    var max_abs_ref: Float64
    var max_abs_got: Float32
    var max_rel: Float64
    var max_rel_x: Float32
    var sum_abs: Float64
    var count: Int


def measure(xs: List[Float32]) -> Stats:
    var s = Stats(
        max_abs=Float64(0), max_abs_x=Float32(0),
        max_abs_ref=Float64(0), max_abs_got=Float32(0),
        max_rel=Float64(0), max_rel_x=Float32(0),
        sum_abs=Float64(0), count=0)
    for i in range(len(xs)):
        var x = xs[i]
        var truth = ref_log(Float64(x))
        var got = scalar_log(x)
        var err = abs_f64(Float64(got) - truth)
        var truth_abs = abs_f64(truth)
        var rel = err / truth_abs if truth_abs > Float64(0.5) else Float64(0)
        if err > s.max_abs:
            s.max_abs = err
            s.max_abs_x = x
            s.max_abs_ref = truth
            s.max_abs_got = got
        if rel > s.max_rel:
            s.max_rel = rel
            s.max_rel_x = x
        s.sum_abs += err
        s.count += 1
    return s


def report(label: String, s: Stats):
    var mean = s.sum_abs / Float64(s.count) if s.count > 0 else Float64(0)
    print(
        label,
        "n=", s.count,
        " max_abs=", s.max_abs, "@x=", s.max_abs_x,
        " (ref=", s.max_abs_ref, " got=", s.max_abs_got, ")",
        " max_rel=", s.max_rel, "@x=", s.max_rel_x,
        " mean_abs=", mean)


def main():
    var anchors = List[Float32]()
    anchors.append(Float32(1.0))
    anchors.append(Float32(2.0))
    anchors.append(Float32(0.5))
    anchors.append(Float32(2.7182817))
    anchors.append(Float32(1.4142135))
    anchors.append(Float32(1.4142137))
    anchors.append(Float32(1.4142133))
    anchors.append(Float32(4.0))
    anchors.append(Float32(16.0))
    anchors.append(Float32(256.0))
    anchors.append(Float32(0.0625))
    anchors.append(Float32(1.0e-6))
    anchors.append(Float32(1.0e6))
    anchors.append(Float32(1.0e-20))
    anchors.append(Float32(1.0e20))

    print("---- Anchor-by-anchor ----")
    for i in range(len(anchors)):
        var x = anchors[i]
        var truth = ref_log(Float64(x))
        var got = scalar_log(x)
        print(
            "  x=", x,
            "  log_f32=", got,
            "  ref=", truth,
            "  |err|=", abs_f64(Float64(got) - truth))
    print()
    report("anchors     ", measure(anchors))
    print()

    var wide = List[Float32]()
    var x = Float32(1.0e-10)
    while x < Float32(1.0e10):
        wide.append(x)
        x = x * Float32(1.001)
    report("wide sweep  ", measure(wide))

    var centered = List[Float32]()
    for i in range(1, 10001):
        centered.append(Float32(i) * Float32(0.001))
    report("linear 0..10", measure(centered))

    var boundary = List[Float32]()
    for i in range(-2000, 2001):
        boundary.append(Float32(1.4142135) + Float32(i) * Float32(1.0e-7))
    report("near sqrt(2)", measure(boundary))

    var gumbel = List[Float32]()
    var y = Float64(1.0e-8)
    while y < Float64(20.0):
        gumbel.append(Float32(y))
        y = y * 1.005
    report("gumbel arg  ", measure(gumbel))

    print()
    print("---- log_f32 at exact boundary inputs ----")
    var b_xs = List[Float32]()
    b_xs.append(Float32(0.0))
    b_xs.append(Float32(1.0))
    b_xs.append(Float32(1.0 / 256.0))
    b_xs.append(Float32(255.0 / 256.0))
    b_xs.append(Float32(1.0 / 257.0))
    b_xs.append(Float32(256.0 / 257.0))
    for i in range(len(b_xs)):
        var x = b_xs[i]
        var got = scalar_log(x)
        var truth = ref_log(Float64(x)) if x > Float32(0) else Float64(0)
        print(
            "  x=", x,
            "  log_f32=", got,
            "  ref=", truth)

    print()
    print("---- uint8 Gumbel: u = u_int / 256, u_int in [0, 256] ----")
    var u8_max_err = Float64(0)
    var u8_max_err_ui = 0
    var u8_sum_err = Float64(0)
    var u8_count = 0
    var u8_denom_f64 = Float64(256.0)
    var u8_denom_f32 = Float32(256.0)
    for ui in range(0, 257):
        var u_f64 = Float64(ui) / u8_denom_f64
        var u_f32 = Float32(ui) / u8_denom_f32
        var truth = -ref_log(-ref_log(u_f64)) if u_f64 > Float64(0) and u_f64 < Float64(1) else Float64(0)
        var nlu = -scalar_log(u_f32)
        var got = -scalar_log(nlu)
        if ui == 0 or ui == 256:
            print(
                "  [edge] u_int=", ui,
                "  u=", u_f64,
                "  got=", got,
                "  (ref=±inf, skipped)")
            continue
        var err = abs_f64(Float64(got) - truth)
        u8_sum_err += err
        u8_count += 1
        if err > u8_max_err:
            u8_max_err = err
            u8_max_err_ui = ui

    print(
        "  n=", u8_count,
        "  max_abs=", u8_max_err, "@u_int=", u8_max_err_ui,
        "  mean_abs=", u8_sum_err / Float64(u8_count))

    print()
    print("---- uint8 Gumbel per-input listing ----")
    for ui in range(0, 257):
        var u_f64 = Float64(ui) / u8_denom_f64
        var u_f32 = Float32(ui) / u8_denom_f32
        var nlu = -scalar_log(u_f32)
        var got = -scalar_log(nlu)
        if ui == 0 or ui == 256:
            print("  u_int=", ui, "  u=", u_f64, "  got=", got, "  (undefined)")
            continue
        var truth = -ref_log(-ref_log(u_f64))
        var err = abs_f64(Float64(got) - truth)
        print("  u_int=", ui, "  u=", u_f64, "  ref=", truth, "  got=", got, "  |err|=", err)
