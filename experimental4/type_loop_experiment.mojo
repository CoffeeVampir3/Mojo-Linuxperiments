# Design experiment: unified model descriptor
#
# Goal: ONE type system used by both the loader and runtime model.
# No bridge. The spec IS the descriptor. var fields, not comptime.
# Architecture constants stay comptime (they're true constants).
# Weight slots are var (they hold computed sharding/offset results).

# ===--- Weight slot: the one descriptor type ---===

@fieldwise_init
struct WeightSlot(Copyable):
    var name: String         # tensor name in checkpoint
    var offset: Int          # byte offset in arena
    var rows: Int            # local (post-shard) rows
    var cols: Int            # local (post-shard) cols
    var global_rows: Int     # full tensor rows (for validation)
    var global_cols: Int     # full tensor cols
    var dtype: DType
    var element_bytes: Int

    fn byte_count(self) -> Int:
        return self.rows * self.cols * self.element_bytes


# ===--- Comptime helpers ---===

fn align_up(value: Int, alignment: Int = 64) -> Int:
    return ((value + alignment - 1) // alignment) * alignment

fn next_off(s: WeightSlot) -> Int:
    return align_up(s.offset + s.byte_count())


# ===--- Layer descriptor ---===

struct TinyLayer(Copyable):
    var q: WeightSlot
    var k: WeightSlot
    var norm: WeightSlot
    var stride: Int

    fn __init__(out self, tp: Int):
        comptime H = 576
        comptime KV = 192

        var off = 0
        self.q = WeightSlot(
            "self_attn.q_proj.weight", off,
            H // tp, H, H, H, DType.bfloat16, 2,
        )
        off = next_off(self.q)
        self.k = WeightSlot(
            "self_attn.k_proj.weight", off,
            KV // tp, H, KV, H, DType.bfloat16, 2,
        )
        off = next_off(self.k)
        self.norm = WeightSlot(
            "input_layernorm.weight", off,
            H, 1, H, 1, DType.float32, 4,
        )
        self.stride = next_off(self.norm)

    fn all_weights(self, prefix: String, base: Int) -> List[WeightSlot]:
        var out = List[WeightSlot]()
        out.append(WeightSlot(prefix + self.q.name, base + self.q.offset,
            self.q.rows, self.q.cols, self.q.global_rows, self.q.global_cols,
            self.q.dtype, self.q.element_bytes))
        out.append(WeightSlot(prefix + self.k.name, base + self.k.offset,
            self.k.rows, self.k.cols, self.k.global_rows, self.k.global_cols,
            self.k.dtype, self.k.element_bytes))
        out.append(WeightSlot(prefix + self.norm.name, base + self.norm.offset,
            self.norm.rows, self.norm.cols, self.norm.global_rows, self.norm.global_cols,
            self.norm.dtype, self.norm.element_bytes))
        return out^


# ===--- Model descriptor ---===

struct TinyModel:
    comptime NUM_LAYERS = 3

    var embed: WeightSlot
    var layer: TinyLayer
    var layers_off: Int

    fn __init__(out self, tp: Int):
        comptime VOCAB = 1024
        comptime H = 576

        var off = 0
        self.embed = WeightSlot(
            "model.embed_tokens.weight", off,
            VOCAB, H, VOCAB, H, DType.bfloat16, 2,
        )
        self.layers_off = next_off(self.embed)
        self.layer = TinyLayer(tp)

    fn total_weight_bytes(self) -> Int:
        return self.layers_off + Self.NUM_LAYERS * self.layer.stride

    fn all_weights(self) -> List[WeightSlot]:
        var out = List[WeightSlot]()
        out.append(self.embed.copy())
        for i in range(Self.NUM_LAYERS):
            var prefix = "model.layers." + String(i) + "."
            var base = self.layers_off + i * self.layer.stride
            var lw = self.layer.all_weights(prefix, base)
            for j in range(len(lw)):
                out.append(lw[j].copy())
        return out^


# ===--- "Loader": completely generic, knows nothing about the model ---===

fn fake_load(weights: List[WeightSlot], total_bytes: Int):
    print("Arena:", total_bytes, "bytes")
    print("Weights:", len(weights))
    for i in range(len(weights)):
        var w = weights[i].copy()
        print("  " + String(w.name), "@" + String(w.offset),
            String(w.rows) + "x" + String(w.cols), "=", w.byte_count(), "bytes",
            w.dtype)


# ===--- Runtime model: uses same descriptor for pointer access ---===

fn fake_inference(model: TinyModel, arena_base: Int):
    # The inference code uses the SAME weight slots for pointer computation
    var q_ptr = arena_base + model.layers_off + 0 * model.layer.stride + model.layer.q.offset
    var k_ptr = arena_base + model.layers_off + 0 * model.layer.stride + model.layer.k.offset
    print("Layer 0 q_proj @", q_ptr, "(" + String(model.layer.q.rows) + "x" + String(model.layer.q.cols) + ")")
    print("Layer 0 k_proj @", k_ptr, "(" + String(model.layer.k.rows) + "x" + String(model.layer.k.cols) + ")")


fn main():
    print("=== tp=1 ===")
    var m1 = TinyModel(tp=1)
    fake_load(m1.all_weights(), m1.total_weight_bytes())
    print()
    fake_inference(m1, arena_base=0x10000)

    print()
    print("=== tp=2 ===")
    var m2 = TinyModel(tp=2)
    fake_load(m2.all_weights(), m2.total_weight_bytes())
    print()
    fake_inference(m2, arena_base=0x10000)
