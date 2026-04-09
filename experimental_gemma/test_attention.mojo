"""Validate Gemma4 local and global attention.

Local: 4Q/2KV heads (2:1 GQA), head_dim=16, window=4
Global: 8Q/1KV heads (8:1 GQA), head_dim=32, full causal

Verifies:
  1. Window bound (local) and full causal (global)
  2. Scale=1.0 for both
  3. Online softmax matches explicit softmax reference
  4. GQA grouping at different ratios (2:1 and 8:1)
  5. Stack-array accumulator (global) matches reference
"""

from std.memory.unsafe_pointer import alloc
from std.math import abs, sqrt, exp
from std.sys.info import simd_width_of
from std.collections import InlineArray

from modeling.model_spec import BF16, F32, Slot, Replicated, DynView, CacheView
from experimental_gemma.attention import local_attention, global_attention

from numa import NumaInfo
from threading import BurstPool


def f32_local_attention_ref(
    q_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    k_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    v_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    pos: Int,
    window_size: Int,
):
    """Reference local attention in f32 with explicit softmax."""
    var heads_per_group = num_heads // num_kv_heads
    var kv_cols = num_kv_heads * head_dim
    var q_cols = num_heads * head_dim
    var causal_len = pos + 1
    var window_start = max(0, causal_len - window_size)
    var attend_len = causal_len - window_start

    var scores = alloc[Float32](attend_len)

    for g in range(num_kv_heads):
        for qh in range(heads_per_group):
            var q_head_idx = g * heads_per_group + qh
            var q_off = q_head_idx * head_dim
            var kv_off = g * head_dim

            # Compute scores: dot(q, K[t]) with no scaling
            var max_score = Float32(-1e30)
            for ti in range(attend_len):
                var t = window_start + ti
                var dot = Float32(0)
                for d in range(head_dim):
                    dot += Float32(q_ptr[q_off + d]) * Float32(k_ptr[t * kv_cols + kv_off + d])
                scores[ti] = dot
                if dot > max_score:
                    max_score = dot

            # Softmax
            var exp_sum = Float32(0)
            for ti in range(attend_len):
                scores[ti] = exp(Float64(scores[ti] - max_score)).cast[DType.float32]()
                exp_sum += scores[ti]
            for ti in range(attend_len):
                scores[ti] /= exp_sum

            # Weighted V sum
            for d in range(head_dim):
                var acc = Float32(0)
                for ti in range(attend_len):
                    var t = window_start + ti
                    acc += scores[ti] * Float32(v_ptr[t * kv_cols + kv_off + d])
                out_ptr[q_off + d] = acc

    scores.free()


def test_local_attention():
    print("=== Local (sliding window) attention ===")

    comptime NUM_HEADS = 4
    comptime NUM_KV_HEADS = 2
    comptime HEAD_DIM = 16
    comptime WINDOW = 4
    comptime Q_COLS = NUM_HEADS * HEAD_DIM
    comptime KV_COLS = NUM_KV_HEADS * HEAD_DIM
    comptime MAX_POS = 16

    comptime QSlot = Slot[BF16, Replicated, 1, Q_COLS, 1]
    comptime KCSlot = Slot[BF16, Replicated, MAX_POS, KV_COLS, 1]
    comptime VCSlot = Slot[BF16, Replicated, MAX_POS, KV_COLS, 1]
    comptime OutSlot = Slot[BF16, Replicated, 1, Q_COLS, 1]

    # Allocate
    var q_buf = alloc[Scalar[DType.bfloat16]](Q_COLS)
    var k_cache = alloc[Scalar[DType.bfloat16]](MAX_POS * KV_COLS)
    var v_cache = alloc[Scalar[DType.bfloat16]](MAX_POS * KV_COLS)
    var out_buf = alloc[Scalar[DType.bfloat16]](Q_COLS)
    var ref_out = alloc[Float32](Q_COLS)

    # Fill Q with deterministic values
    for i in range(Q_COLS):
        q_buf[i] = Scalar[DType.bfloat16](Float32(0.1) * Float32(i % 7 - 3))

    # Fill KV cache: 10 positions of data
    comptime FILLED_POS = 10
    for t in range(FILLED_POS):
        for j in range(KV_COLS):
            var kv = Float32(0.05) * Float32((t * KV_COLS + j) % 11 - 5)
            k_cache[t * KV_COLS + j] = Scalar[DType.bfloat16](kv)
            var vv = Float32(0.08) * Float32((t * KV_COLS + j + 3) % 13 - 6)
            v_cache[t * KV_COLS + j] = Scalar[DType.bfloat16](vv)

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    # Test at multiple positions to exercise window behavior
    var test_positions = InlineArray[Int, 4](fill=0)
    test_positions[0] = 0   # pos=0: attend to just [0]
    test_positions[1] = 2   # pos=2: attend to [0,1,2] (within window)
    test_positions[2] = 5   # pos=5: attend to [2,3,4,5] (window clips)
    test_positions[3] = 9   # pos=9: attend to [6,7,8,9] (window clips)

    for pi in range(4):
        var pos = test_positions[pi]
        var causal_len = pos + 1
        var window_start = max(0, causal_len - WINDOW)

        print()
        print("  --- pos=" + String(pos) + " window=[" + String(window_start) + "," + String(pos) + "] ---")

        # Clear output
        for i in range(Q_COLS):
            out_buf[i] = Scalar[DType.bfloat16](0)

        # Reference
        f32_local_attention_ref(
            q_buf, k_cache, v_cache,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_out)),
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, pos, WINDOW,
        )

        # Kernel
        var q_view = DynView[QSlot](Int(q_buf), 1)
        var out_view = DynView[OutSlot](Int(out_buf), 1)
        var kc_view = CacheView[KCSlot](Int(k_cache))
        var vc_view = CacheView[VCSlot](Int(v_cache))

        local_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, WINDOW](
            q_view, kc_view, vc_view, out_view, pos, pool).join()

        # Compare per head
        for h in range(NUM_HEADS):
            var max_kern_err = Float64(0)
            var max_bf16_err = Float64(0)
            for d in range(HEAD_DIM):
                var idx = h * HEAD_DIM + d
                var kv = Float32(out_buf[idx])
                var fv = ref_out[idx]
                var bv = Float32(Scalar[DType.bfloat16](fv))
                var kern_err = abs(Float64(kv) - Float64(bv))
                var bf16_err = abs(Float64(fv) - Float64(bv))
                if kern_err > max_kern_err:
                    max_kern_err = kern_err
                if bf16_err > max_bf16_err:
                    max_bf16_err = bf16_err
            print("    head " + String(h) + " (kv_group=" + String(h // 2) + "): kern_err=" + String(max_kern_err) + "  bf16_err=" + String(max_bf16_err))

    # Detailed A/B for one position
    var detail_pos = 5
    var detail_ws = max(0, detail_pos + 1 - WINDOW)
    print()
    print("  --- detailed A/B at pos=" + String(detail_pos) + " window=[" + String(detail_ws) + "," + String(detail_pos) + "] head=0 ---")

    for i in range(Q_COLS):
        out_buf[i] = Scalar[DType.bfloat16](0)

    f32_local_attention_ref(
        q_buf, k_cache, v_cache,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_out)),
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, detail_pos, WINDOW,
    )

    var q_view = DynView[QSlot](Int(q_buf), 1)
    var out_view = DynView[OutSlot](Int(out_buf), 1)
    var kc_view = CacheView[KCSlot](Int(k_cache))
    var vc_view = CacheView[VCSlot](Int(v_cache))
    local_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, WINDOW](
        q_view, kc_view, vc_view, out_view, detail_pos, pool).join()

    print("  dim | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+---------")
    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for d in range(HEAD_DIM):
        var kv = Float32(out_buf[d])
        var fv = ref_out[d]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        print("  " + String(d) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))
    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    q_buf.free()
    k_cache.free()
    v_cache.free()
    out_buf.free()
    ref_out.free()
    _ = pool^


def f32_global_attention_ref(
    q_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    k_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    v_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    pos: Int,
):
    """Reference full-causal attention in f32 with explicit softmax."""
    var heads_per_group = num_heads // num_kv_heads
    var kv_cols = num_kv_heads * head_dim
    var causal_len = pos + 1

    var scores = alloc[Float32](causal_len)

    for g in range(num_kv_heads):
        for qh in range(heads_per_group):
            var q_head_idx = g * heads_per_group + qh
            var q_off = q_head_idx * head_dim
            var kv_off = g * head_dim

            # Scores: dot(q, K[t]) with no scaling
            var max_score = Float32(-1e30)
            for t in range(causal_len):
                var dot = Float32(0)
                for d in range(head_dim):
                    dot += Float32(q_ptr[q_off + d]) * Float32(k_ptr[t * kv_cols + kv_off + d])
                scores[t] = dot
                if dot > max_score:
                    max_score = dot

            # Softmax
            var exp_sum = Float32(0)
            for t in range(causal_len):
                scores[t] = exp(Float64(scores[t] - max_score)).cast[DType.float32]()
                exp_sum += scores[t]
            for t in range(causal_len):
                scores[t] /= exp_sum

            # Weighted V sum
            for d in range(head_dim):
                var acc = Float32(0)
                for t in range(causal_len):
                    acc += scores[t] * Float32(v_ptr[t * kv_cols + kv_off + d])
                out_ptr[q_off + d] = acc

    scores.free()


def test_global_attention():
    print("=== Global (full causal) attention ===")

    comptime NUM_HEADS = 8
    comptime NUM_KV_HEADS = 1
    comptime HEAD_DIM = 32
    comptime Q_COLS = NUM_HEADS * HEAD_DIM
    comptime KV_COLS = NUM_KV_HEADS * HEAD_DIM
    comptime MAX_POS = 16

    comptime QSlot = Slot[BF16, Replicated, 1, Q_COLS, 1]
    comptime KCSlot = Slot[BF16, Replicated, MAX_POS, KV_COLS, 1]
    comptime VCSlot = Slot[BF16, Replicated, MAX_POS, KV_COLS, 1]
    comptime OutSlot = Slot[BF16, Replicated, 1, Q_COLS, 1]

    var q_buf = alloc[Scalar[DType.bfloat16]](Q_COLS)
    var k_cache = alloc[Scalar[DType.bfloat16]](MAX_POS * KV_COLS)
    var v_cache = alloc[Scalar[DType.bfloat16]](MAX_POS * KV_COLS)
    var out_buf = alloc[Scalar[DType.bfloat16]](Q_COLS)
    var ref_out = alloc[Float32](Q_COLS)

    # Fill Q
    for i in range(Q_COLS):
        q_buf[i] = Scalar[DType.bfloat16](Float32(0.08) * Float32(i % 9 - 4))

    # Fill KV cache: 12 positions
    comptime FILLED_POS = 12
    for t in range(FILLED_POS):
        for j in range(KV_COLS):
            k_cache[t * KV_COLS + j] = Scalar[DType.bfloat16](
                Float32(0.04) * Float32((t * KV_COLS + j) % 13 - 6))
            v_cache[t * KV_COLS + j] = Scalar[DType.bfloat16](
                Float32(0.06) * Float32((t * KV_COLS + j + 5) % 11 - 5))

    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    var pool = BurstPool[].for_numa_node(numa, topo[0])

    # Test at multiple positions
    var test_positions = InlineArray[Int, 4](fill=0)
    test_positions[0] = 0   # single token
    test_positions[1] = 3   # short history
    test_positions[2] = 7   # medium
    test_positions[3] = 11  # longest

    for pi in range(4):
        var pos = test_positions[pi]
        print()
        print("  --- pos=" + String(pos) + " causal_len=" + String(pos + 1) + " (8:1 GQA) ---")

        for i in range(Q_COLS):
            out_buf[i] = Scalar[DType.bfloat16](0)

        f32_global_attention_ref(
            q_buf, k_cache, v_cache,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_out)),
            NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, pos,
        )

        var q_view = DynView[QSlot](Int(q_buf), 1)
        var out_view = DynView[OutSlot](Int(out_buf), 1)
        var kc_view = CacheView[KCSlot](Int(k_cache))
        var vc_view = CacheView[VCSlot](Int(v_cache))

        global_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](
            q_view, kc_view, vc_view, out_view, pos, pool).join()

        for h in range(NUM_HEADS):
            var max_kern_err = Float64(0)
            var max_bf16_err = Float64(0)
            for d in range(HEAD_DIM):
                var idx = h * HEAD_DIM + d
                var kv = Float32(out_buf[idx])
                var fv = ref_out[idx]
                var bv = Float32(Scalar[DType.bfloat16](fv))
                var kern_err = abs(Float64(kv) - Float64(bv))
                var bf16_err = abs(Float64(fv) - Float64(bv))
                if kern_err > max_kern_err:
                    max_kern_err = kern_err
                if bf16_err > max_bf16_err:
                    max_bf16_err = bf16_err
            print("    head " + String(h) + " (kv_group=0): kern_err=" + String(max_kern_err) + "  bf16_err=" + String(max_bf16_err))

    # Detailed A/B at pos=7, head=0
    var detail_pos = 7
    print()
    print("  --- detailed A/B at pos=" + String(detail_pos) + " head=0 (8:1 GQA, full causal) ---")

    for i in range(Q_COLS):
        out_buf[i] = Scalar[DType.bfloat16](0)

    f32_global_attention_ref(
        q_buf, k_cache, v_cache,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ref_out)),
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, detail_pos,
    )

    var q_view = DynView[QSlot](Int(q_buf), 1)
    var out_view = DynView[OutSlot](Int(out_buf), 1)
    var kc_view = CacheView[KCSlot](Int(k_cache))
    var vc_view = CacheView[VCSlot](Int(v_cache))
    global_attention[NUM_HEADS, NUM_KV_HEADS, HEAD_DIM](
        q_view, kc_view, vc_view, out_view, detail_pos, pool).join()

    print("  dim | kernel(bf16) | bf16(f32ref) | f32ref       | kern_err     | bf16_err")
    print("  ----+--------------+--------------+--------------+--------------+---------")
    var max_kern_err = Float64(0)
    var max_bf16_err = Float64(0)
    for d in range(HEAD_DIM):
        var kv = Float32(out_buf[d])
        var fv = ref_out[d]
        var bv = Float32(Scalar[DType.bfloat16](fv))
        var kern_err = abs(Float64(kv) - Float64(bv))
        var bf16_err = abs(Float64(fv) - Float64(bv))
        if kern_err > max_kern_err:
            max_kern_err = kern_err
        if bf16_err > max_bf16_err:
            max_bf16_err = bf16_err
        print("  " + String(d) + " | " + String(kv) + " | " + String(bv) + " | " + String(fv) + " | " + String(kern_err) + " | " + String(bf16_err))
    print("  max kernel_err=" + String(max_kern_err) + "  max bf16_err=" + String(max_bf16_err))

    q_buf.free()
    k_cache.free()
    v_cache.free()
    out_buf.free()
    ref_out.free()
    _ = pool^


def main():
    test_local_attention()
    print()
    test_global_attention()
