"""Probe embedding table to determine tokenizer byte encoding.

Compares embedding variance for Ġ-prefixed tokens (GPT2 byte encoding)
vs non-prefixed tokens to determine which the model was trained with.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from modeling.deepseekv2_lite import DeepSeekV2Lite, DeepSeekV2LiteConfig
from modeling.model_spec import bind

comptime C = DeepSeekV2LiteConfig
comptime MODEL_DIR = "checkpoints/deepseekv2-lite"


def embed_variance(embed_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
                   token_id: Int, hidden: Int) -> Float64:
    var row = embed_ptr + token_id * hidden
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float64, 1](0)
    for i in range(0, hidden, width):
        var v = (row + i).load[width=width]().cast[DType.float64]()
        acc += (v * v).reduce_add()
    return acc / Float64(hidden)


def main():
    var model_opt = DeepSeekV2Lite[1].load(Path(MODEL_DIR))
    if not model_opt:
        print("load failed")
        return
    var model = model_opt.take()

    comptime M = DeepSeekV2Lite[1].M
    var host = model.rank(0)
    var embed = host.host_weight[M.EMBED]()
    var ep = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=embed.ptr)

    # Token pairs: Ġ-prefixed (space-attached) vs bare
    # These are the actual vocab entries we verified
    print("=== Embedding variance comparison ===")
    print("Higher variance = actively trained token")
    print()

    var pairs = List[Tuple[Int, String, Int, String]]()
    # (Ġ-prefixed id, name, bare id, name)
    # Token 1843 = Ġworld, Token 11123 = world
    # Token 1002 = ĠThis, Token 1567 = This
    # Token 317 = Ġis, Token 262 = is
    # Token 245 = Ġa, Token ??? = a

    var test_ids = List[Tuple[Int, String]]()
    test_ids.append((1843, "Ġworld"))
    test_ids.append((11123, "world"))
    test_ids.append((1002, "ĠThis"))
    test_ids.append((1567, "This"))
    test_ids.append((317, "Ġis"))
    test_ids.append((262, "is"))
    test_ids.append((245, "Ġa"))
    test_ids.append((17464, "Hello"))
    test_ids.append((11, ","))
    test_ids.append((0, "!"))

    for i in range(len(test_ids)):
        var tid = test_ids[i][0]
        var name = test_ids[i][1]
        var var_ = embed_variance(ep, tid, C.HIDDEN)
        print("  id=", tid, "var=", var_, " ", name)

    # Also check some high-id tokens that might be special/unused
    print()
    print("=== Special/boundary tokens ===")
    var special_ids = List[Tuple[Int, String]]()
    special_ids.append((100000, "BOS"))
    special_ids.append((100001, "EOS"))
    special_ids.append((99999, "last regular"))
    special_ids.append((50000, "mid vocab"))

    for i in range(len(special_ids)):
        var tid = special_ids[i][0]
        var name = special_ids[i][1]
        var var_ = embed_variance(ep, tid, C.HIDDEN)
        print("  id=", tid, "var=", var_, " ", name)

    # Aggregate: average variance of Ġ-prefixed vs bare for common words
    print()
    print("=== Aggregate comparison ===")
    var g_sum = Float64(0)
    var bare_sum = Float64(0)
    # Ġ-prefixed: 1843, 1002, 317, 245
    g_sum += embed_variance(ep, 1843, C.HIDDEN)
    g_sum += embed_variance(ep, 1002, C.HIDDEN)
    g_sum += embed_variance(ep, 317, C.HIDDEN)
    g_sum += embed_variance(ep, 245, C.HIDDEN)
    # Bare: 11123, 1567, 262
    bare_sum += embed_variance(ep, 11123, C.HIDDEN)
    bare_sum += embed_variance(ep, 1567, C.HIDDEN)
    bare_sum += embed_variance(ep, 262, C.HIDDEN)

    print("avg Ġ-prefixed variance:", g_sum / 4.0)
    print("avg bare variance:      ", bare_sum / 3.0)

    _ = model
