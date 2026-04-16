"""Compare tensor dumps between TP=1 and TP=2 runs.

Expects: taps/tp1/*.bin and taps/tp2/*.bin
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import size_of
from simd_math import sqrt


comptime HIDDEN = 2816
comptime NUM_LAYERS = 30


def read_bf16_as_f32(path: String) -> Optional[List[Float32]]:
    try:
        var fh = FileHandle(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        var num_elements = len(raw) // 2
        var raw_ptr = raw.unsafe_ptr()
        var buf = List[Float32](capacity=num_elements)
        for i in range(num_elements):
            var lo = UInt16(raw_ptr[i * 2])
            var hi = UInt16(raw_ptr[i * 2 + 1])
            var bits = UInt32(lo) | (UInt32(hi) << 8)
            var f32_bits = bits << 16
            var val = UnsafePointer(to=f32_bits).bitcast[Float32]()[]
            buf.append(val)
        return buf^
    except:
        return None


def max_abs(buf: List[Float32]) -> Float32:
    var m = Float32(0)
    for i in range(len(buf)):
        var a = buf[i] if buf[i] >= 0 else -buf[i]
        if a > m:
            m = a
    return m


def compare(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32, Float32]:
    var n = min(len(a), len(b))
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    var dot = Float32(0)
    var na2 = Float32(0)
    var nb2 = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
        sum_diff += d
        dot += a[i] * b[i]
        na2 += a[i] * a[i]
        nb2 += b[i] * b[i]
    var denom = sqrt(Float64(na2)) * sqrt(Float64(nb2))
    var cosine = Float32(Float64(dot) / denom) if denom > 0 else Float32(0)
    return (max_diff, sum_diff / Float32(n), cosine)


def check(label: String, path_a: String, path_b: String):
    var maybe_a = read_bf16_as_f32(path_a)
    var maybe_b = read_bf16_as_f32(path_b)
    if not maybe_a or not maybe_b:
        return
    var a = maybe_a.take()
    var b = maybe_b.take()
    var result = compare(a, b)
    var max_d = result[0]
    var mean_d = result[1]
    var cos = result[2]
    var tag: String
    if cos > 0.999:
        tag = "OK"
    elif cos > 0.99:
        tag = "WARN"
    else:
        tag = "BAD"
    print(label, " max:", max_d, " mean:", mean_d, " cos:", cos, " ", tag)


def main():
    var tp1 = String("taps/tp1")
    var tp2 = String("taps/tp2")

    print("=== TP=2 cross-rank (post-broadcast, should match) ===")
    var er0 = tp2 + "/embed_r0.bin"
    var er1 = tp2 + "/embed_r1.bin"
    check("embed", er0, er1)
    print()

    print("=== TP=2 cross-rank (pre-allreduce, will differ) ===")
    for i in range(NUM_LAYERS):
        var l = String(i)
        var r0 = tp2 + "/L" + l + "_pre_attn_reduce_r0.bin"
        var r1 = tp2 + "/L" + l + "_pre_attn_reduce_r1.bin"
        check("L" + l + "_pre_attn_reduce", r0, r1)
    print()

    print("=== TP=1 vs TP=2 post-attn allreduce ===")
    for i in range(NUM_LAYERS):
        var l = String(i)
        var a = tp1 + "/L" + l + "_post_attn_reduce.bin"
        var b = tp2 + "/L" + l + "_post_attn_reduce.bin"
        check("L" + l + "_post_attn_reduce", a, b)
    print()

    print("=== TP=1 vs TP=2 dense allreduce ===")
    for i in range(NUM_LAYERS):
        var l = String(i)
        var a = tp1 + "/L" + l + "_dense_post.bin"
        var b = tp2 + "/L" + l + "_dense_post.bin"
        check("L" + l + "_dense_post", a, b)
    print()

    print("=== TP=1 vs TP=2 expert allreduce ===")
    for i in range(NUM_LAYERS):
        var l = String(i)
        var a = tp1 + "/L" + l + "_expert_post.bin"
        var b = tp2 + "/L" + l + "_expert_post.bin"
        check("L" + l + "_expert_post", a, b)
    print()

    print("=== TP=1 vs TP=2 layer outputs ===")
    for i in range(NUM_LAYERS):
        var l = String(i)
        var a = tp1 + "/L" + l + "_out.bin"
        var b = tp2 + "/L" + l + "_out.bin"
        check("L" + l + "_out", a, b)
