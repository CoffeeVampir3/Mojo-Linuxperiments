"""Design reproduction: explicit quantize-task variants.

The previous `QuantizeTask` was one monolithic struct wrapping a flat
`QuantOp` bag-of-flags (`quantize`, `rotate`, `block`, `per_block`,
`smooth_src`). Call sites wrote `Rotated(128).to_op()` and the real
operation was hidden behind the factory — readers had to chase into the
quantizer to know whether a task rotates, quantizes, emits a scale
tensor, absorbs a gamma, or none of the above. Adding variants meant
adding flags and checking "is this combination legitimate" at runtime.

This design replaces that with an explicit variant set: one concrete
struct per real operation. Each struct's field list is exactly what the
operation needs — no sentinels, no hidden rules, no "read the
quantizer to understand." The task list becomes a list of variants;
the driver matches on the variant and dispatches to its execute path.

Today's complete set — derived from every `QuantizeTask(...)` call site
in the codebase:

  1. Passthrough(name, expected_dtype)
  2. ButterquantI8PerRow(name, source, block)
  3. ButterquantI8PerRowAbsorbed(name, source, block, gamma_src)
  4. ButterquantI8PerBlock(name, source, block)
  5. ButterquantI8PerBlockAbsorbed(name, source, block, gamma_src)

Five variants. That's the full space. Adding a new operation means a
new struct + one arm in the driver match, nowhere else.

The `source` field on the butterquant variants is a `SourceFormat` tag
handed to the `Converter` machinery from design_parametric_quant; that
parametric stays as-is and decides how many input tensors the source
format needs and how their bytes turn into f32. The task variant is
agnostic to that detail — it just carries the tag.
"""

from std.memory import UnsafePointer
from std.utils.variant import Variant


# =============================================================================
# SourceFormat tags — mirror the ones in modeling/model_spec.mojo so this
# design file compiles standalone.
# =============================================================================


struct SourceFormat:
    comptime BF16 = 0
    comptime F32 = 1
    comptime FP8_E4M3_BLOCK128 = 2


def source_format_name(tag: Int) -> String:
    if tag == SourceFormat.BF16: return "bf16"
    if tag == SourceFormat.F32: return "f32"
    if tag == SourceFormat.FP8_E4M3_BLOCK128: return "fp8_e4m3_block128"
    return "?"


def dtype_name(dt: DType) -> String:
    if dt == DType.bfloat16: return "bf16"
    if dt == DType.float32: return "f32"
    if dt == DType.float16: return "f16"
    if dt == DType.int8: return "i8"
    if dt == DType.uint8: return "u8"
    if dt == DType.float8_e4m3fn: return "fp8_e4m3"
    return "?"


# =============================================================================
# Trait — shared contract for reading a task, not for dispatch
#
# Mojo's stdlib `Variant` gives us the sum type; the trait exists so each
# concrete variant can tell callers what it does without callers having to
# know which variant they hold. Today that's just describe(); we can grow
# it to input_tensor_names() / output_entries() / execute() once the real
# driver moves. Keeping it thin now makes each concrete struct cheap.
# =============================================================================


trait QuantizeTask:
    def describe(self) -> String: ...


# =============================================================================
# Passthrough — one tensor in, one tensor out, bytes unchanged
#
# `expected_dtype` is a sanity gate: the planner checks the on-disk dtype
# against it and errors early if they disagree. Catches the "I thought
# this was bf16 but it's f32" class of misconfiguration without tying the
# passthrough path to any decode logic. The bytes themselves move through
# unchanged either way.
# =============================================================================


@fieldwise_init
struct Passthrough(QuantizeTask, Copyable, Movable):
    var name: String
    var expected_dtype: DType

    def describe(self) -> String:
        return ("passthrough "
            + self.name + " (expect " + dtype_name(self.expected_dtype) + ")")


# =============================================================================
# ButterquantI8PerRow — FWHT rotate + per-row absmax i8
#
# Reads: the input tensors declared by `source` (one for bf16/f32, two
#        for FP8_E4M3_BLOCK128, etc.).
# Writes: <name> as int8 + <name>_scale as f32[rows] (one scale per row).
# =============================================================================


@fieldwise_init
struct ButterquantI8PerRow(QuantizeTask, Copyable, Movable):
    var name: String
    var source: Int      # SourceFormat tag
    var block: Int       # FWHT rotation block

    def describe(self) -> String:
        return ("butterquant_i8_per_row "
            + self.name + " source=" + source_format_name(self.source)
            + " block=" + String(self.block))


# =============================================================================
# ButterquantI8PerRowAbsorbed — same as PerRow, gamma absorbed first
#
# Reads: source-declared input tensors + the gamma tensor at gamma_src.
# Writes: same as PerRow.
# The gamma is applied as W' = W * sqrt(|gamma|) before rotation, so the
# runtime RMSNorm can skip the gamma multiply at inference time.
# =============================================================================


@fieldwise_init
struct ButterquantI8PerRowAbsorbed(QuantizeTask, Copyable, Movable):
    var name: String
    var source: Int
    var block: Int
    var gamma_src: String

    def describe(self) -> String:
        return ("butterquant_i8_per_row_absorbed "
            + self.name + " source=" + source_format_name(self.source)
            + " block=" + String(self.block) + " gamma=" + self.gamma_src)


# =============================================================================
# ButterquantI8PerBlock — FWHT rotate + per-block absmax i8
#
# Reads: source-declared input tensors.
# Writes: <name> as int8 + <name>_scale as f32[rows, cols / block].
# The same block size governs both FWHT and scale granularity — one block
# per scale entry per row.
# =============================================================================


@fieldwise_init
struct ButterquantI8PerBlock(QuantizeTask, Copyable, Movable):
    var name: String
    var source: Int
    var block: Int

    def describe(self) -> String:
        return ("butterquant_i8_per_block "
            + self.name + " source=" + source_format_name(self.source)
            + " block=" + String(self.block))


# =============================================================================
# ButterquantI8PerBlockAbsorbed — same as PerBlock, gamma absorbed first
# =============================================================================


@fieldwise_init
struct ButterquantI8PerBlockAbsorbed(QuantizeTask, Copyable, Movable):
    var name: String
    var source: Int
    var block: Int
    var gamma_src: String

    def describe(self) -> String:
        return ("butterquant_i8_per_block_absorbed "
            + self.name + " source=" + source_format_name(self.source)
            + " block=" + String(self.block) + " gamma=" + self.gamma_src)


# =============================================================================
# Variant umbrella — heterogeneous task list element type
#
# The driver pattern-matches on this at dispatch time; each arm hands off
# to a per-variant execute path. No flat "kind: Int + all-fields" struct,
# no unused fields, no "is this combination legitimate" check.
# =============================================================================


comptime Task = Variant[
    Passthrough,
    ButterquantI8PerRow,
    ButterquantI8PerRowAbsorbed,
    ButterquantI8PerBlock,
    ButterquantI8PerBlockAbsorbed,
]


def describe_task(read task: Task) -> String:
    if task.isa[Passthrough]():
        return task[Passthrough].describe()
    if task.isa[ButterquantI8PerRow]():
        return task[ButterquantI8PerRow].describe()
    if task.isa[ButterquantI8PerRowAbsorbed]():
        return task[ButterquantI8PerRowAbsorbed].describe()
    if task.isa[ButterquantI8PerBlock]():
        return task[ButterquantI8PerBlock].describe()
    if task.isa[ButterquantI8PerBlockAbsorbed]():
        return task[ButterquantI8PerBlockAbsorbed].describe()
    return "<unknown task>"


# =============================================================================
# Demo — build the tasks a Gemma-4 layer needs under the new API.
#
# Compare against gemma_4_moe_butterquant_tp.build_quantizer_tasks today:
# every `Rotated(HB).to_op()` becomes a `ButterquantI8PerRow(...)`, every
# `SmoothPerBlock(LB, "...")` becomes a `ButterquantI8PerBlockAbsorbed`,
# every `NoQuant().to_op()` becomes a `Passthrough`. The source tag makes
# FP8 weights (on MiniMax-M2.7) a three-character change — swap BF16 for
# FP8_E4M3_BLOCK128 at the construction site, nothing else touches it.
# =============================================================================


def demo_gemma_layer_zero():
    var p = "model.language_model.layers.0."
    var tasks = List[Task]()

    # Attention projections — rotated per-row i8, bf16 source, block 256.
    tasks.append(ButterquantI8PerRow(p + "self_attn.q_proj.weight",   SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "self_attn.k_proj.weight",   SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "self_attn.v_proj.weight",   SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "self_attn.o_proj.weight",   SourceFormat.BF16, 256))

    # Dense MLP — gate/up at block 256, down at block 16.
    tasks.append(ButterquantI8PerRow(p + "mlp.gate_proj.weight",      SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "mlp.up_proj.weight",        SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "mlp.down_proj.weight",      SourceFormat.BF16, 16))

    # Norms and scalars — passthrough bf16.
    tasks.append(Passthrough(p + "input_layernorm.weight",            DType.bfloat16))
    tasks.append(Passthrough(p + "post_attention_layernorm.weight",   DType.bfloat16))
    tasks.append(Passthrough(p + "self_attn.q_norm.weight",           DType.bfloat16))
    tasks.append(Passthrough(p + "self_attn.k_norm.weight",           DType.bfloat16))
    tasks.append(Passthrough(p + "layer_scalar",                      DType.bfloat16))

    # MoE router proj — rotated per-row.
    tasks.append(ButterquantI8PerRow(p + "router.proj.weight",        SourceFormat.BF16, 256))
    tasks.append(Passthrough(p + "router.scale",                      DType.bfloat16))
    tasks.append(Passthrough(p + "router.per_expert_scale",           DType.bfloat16))

    # Experts — stacked gate/up rotated per-row, down rotated per-row at block 64.
    tasks.append(ButterquantI8PerRow(p + "experts.gate_up_proj",      SourceFormat.BF16, 256))
    tasks.append(ButterquantI8PerRow(p + "experts.down_proj",         SourceFormat.BF16, 64))

    print("--- gemma layer 0 tasks (", len(tasks), ") ---")
    for i in range(len(tasks)):
        print("  " + describe_task(tasks[i]))


def demo_top_level():
    var tasks = List[Task]()

    # Final norm — passthrough.
    tasks.append(Passthrough("model.language_model.norm.weight",      DType.bfloat16))

    # Tied embed / LM head — per-block i8 with absorbed final-norm gamma.
    tasks.append(ButterquantI8PerBlockAbsorbed(
        "model.language_model.embed_tokens.weight",
        SourceFormat.BF16,
        64,
        "model.language_model.norm.weight"))

    print("--- top-level tasks (", len(tasks), ") ---")
    for i in range(len(tasks)):
        print("  " + describe_task(tasks[i]))


def demo_minimax_fp8():
    # Same shape as gemma, but the matmul weights declare FP8 source. The
    # only change from bf16 sources is the SourceFormat tag — the variant
    # type and its other fields are identical.
    var p = "model.layers.0."
    var tasks = List[Task]()

    tasks.append(ButterquantI8PerRow(p + "self_attn.q_proj.weight",   SourceFormat.FP8_E4M3_BLOCK128, 128))
    tasks.append(ButterquantI8PerRow(p + "self_attn.k_proj.weight",   SourceFormat.FP8_E4M3_BLOCK128, 128))
    tasks.append(ButterquantI8PerRow(p + "self_attn.v_proj.weight",   SourceFormat.FP8_E4M3_BLOCK128, 128))
    tasks.append(ButterquantI8PerRow(p + "self_attn.o_proj.weight",   SourceFormat.FP8_E4M3_BLOCK128, 128))

    tasks.append(Passthrough(p + "input_layernorm.weight",            DType.bfloat16))
    tasks.append(Passthrough(p + "post_attention_layernorm.weight",   DType.bfloat16))
    tasks.append(Passthrough(p + "self_attn.q_norm.weight",           DType.bfloat16))
    tasks.append(Passthrough(p + "self_attn.k_norm.weight",           DType.bfloat16))

    # Router is f32 passthrough on MiniMax — expected_dtype catches it.
    tasks.append(Passthrough(p + "block_sparse_moe.gate.weight",      DType.float32))
    tasks.append(Passthrough(p + "block_sparse_moe.e_score_correction_bias", DType.float32))

    print("--- minimax-m2.7 layer 0 tasks (", len(tasks), ") ---")
    for i in range(len(tasks)):
        print("  " + describe_task(tasks[i]))


def main():
    demo_gemma_layer_zero()
    print()
    demo_top_level()
    print()
    demo_minimax_fp8()
