"""Validate scaled_add, embed_lookup_scaled, and logit_softcap.

Tests with small dimensions for full A/B visibility.
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs, exp
from std.sys.info import simd_width_of
from std.collections import InlineArray

from modeling.model_spec import BF16, Slot, Replicated, Bound, DynView
from experimental_gemma.ops import scaled_add, embed_lookup_scaled, logit_softcap

from numa import NumaInfo
from threading import BurstPool


def test_scaled_add():
    print("=== scaled_add: dst += src * scale ===")

    comptime COLS = 32
    comptime View = Slot[BF16, Replicated, 1, COLS, 1]
    var scale = Float32(0.73)

    var src = alloc[Scalar[DType.bfloat16]](COLS)
    var dst = alloc[Scalar[DType.bfloat16]](COLS)
    var ref_out = alloc[Float32](COLS)

    for i in range(COLS):
        src[i] = Scalar[DType.bfloat16](Float32(-1.0) + Float32(i) * Float32(0.07))
        dst[i] = Scalar[DType.bfloat16](Float32(0.5) - Float32(i) * Float32(0.03))

    # Reference: dst + src * scale
    for i in range(COLS):
        ref_out[i] = Float32(dst[i]) + Float32(src[i]) * scale

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    var src_v = DynView[View](Int(src), 1)
    var dst_v = DynView[View](Int(dst), 1)
    scaled_add(src_v, dst_v, scale, pool).join()

    print("  scale=" + String(scale))
    print("  idx | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+---------")

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for i in range(COLS):
        var kv = Float32(dst[i])
        var fv = ref_out[i]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        print("  " + String(i) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    src.free()
    dst.free()
    ref_out.free()
    _ = pool^


def test_embed_lookup_scaled():
    print()
    print("=== embed_lookup_scaled: out = table[token] * scale ===")

    comptime VOCAB = 16
    comptime HIDDEN = 32
    comptime SEQ_LEN = 4
    comptime TableSlot = Slot[BF16, Replicated, VOCAB, HIDDEN, 1]
    comptime OutSlot = Slot[BF16, Replicated, SEQ_LEN, HIDDEN, 1]

    var scale = Float32(7.25)

    var table = alloc[Scalar[DType.bfloat16]](VOCAB * HIDDEN)
    var tokens = alloc[Scalar[DType.int32]](SEQ_LEN)
    var output = alloc[Scalar[DType.bfloat16]](SEQ_LEN * HIDDEN)
    var ref_out = alloc[Float32](SEQ_LEN * HIDDEN)

    # Fill embedding table
    for i in range(VOCAB * HIDDEN):
        table[i] = Scalar[DType.bfloat16](Float32(0.02) * Float32(i % 37 - 18))

    # Token IDs
    tokens[0] = Scalar[DType.int32](3)
    tokens[1] = Scalar[DType.int32](7)
    tokens[2] = Scalar[DType.int32](0)
    tokens[3] = Scalar[DType.int32](15)

    # Reference
    for s in range(SEQ_LEN):
        var tid = Int(tokens[s])
        for j in range(HIDDEN):
            ref_out[s * HIDDEN + j] = Float32(table[tid * HIDDEN + j]) * scale

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    var table_b = Bound[TableSlot](Int(table))
    var out_v = DynView[OutSlot](Int(output), SEQ_LEN)
    embed_lookup_scaled(table_b, Int(tokens), out_v, scale, pool).join()

    print("  scale=" + String(scale) + " tokens=[3,7,0,15]")
    print("  seq | idx | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err")
    print("  ----+-----+--------------+--------------+--------------+---------")

    var max_kern_err = Float64(0)
    for s in range(SEQ_LEN):
        for j in range(HIDDEN):
            var idx = s * HIDDEN + j
            var kv = Float32(output[idx])
            var fv = ref_out[idx]
            var bv = Float32(Scalar[DType.bfloat16](fv))
            var kern_err = abs(Float64(kv) - Float64(bv))
            if kern_err > max_kern_err:
                max_kern_err = kern_err
            if j < 4 or j == HIDDEN - 1:
                print("  " + String(s) + " | " + String(j) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err))

    print("  max kernel_err=" + String(max_kern_err))

    table.free()
    tokens.free()
    output.free()
    ref_out.free()
    _ = pool^


def test_logit_softcap():
    print()
    print("=== logit_softcap: tanh(x/30) * 30 ===")

    comptime COLS = 32
    comptime View = Slot[BF16, Replicated, 1, COLS, 1]

    var logits = alloc[Scalar[DType.bfloat16]](COLS)
    var ref_out = alloc[Float32](COLS)

    # Range of values including extremes
    for i in range(COLS):
        var v = Float32(-60.0) + Float32(i) * Float32(4.0)
        logits[i] = Scalar[DType.bfloat16](v)

    # Reference: tanh(x/30) * 30 via (e^(2u) - 1) / (e^(2u) + 1) * 30
    for i in range(COLS):
        var v = Float64(Float32(logits[i]))
        var u = v / 30.0
        var e2u = exp(2.0 * u)
        ref_out[i] = Float32((e2u - 1.0) / (e2u + 1.0) * 30.0)

    var dst_v = DynView[View](Int(logits), 1)
    logit_softcap(dst_v)

    print("  idx | input        | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+--------------+---------")

    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for i in range(COLS):
        var input_v = Float32(-60.0) + Float32(i) * Float32(4.0)
        var kv = Float32(logits[i])
        var fv = ref_out[i]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        print("  " + String(i) + " | " + String(input_v) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))

    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    logits.free()
    ref_out.free()


def main():
    test_scaled_add()
    test_embed_lookup_scaled()
    test_logit_softcap()
