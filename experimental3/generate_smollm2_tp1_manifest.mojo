from pathlib import Path

from experimental3.core import validate_dtype, validate_shape_exact
from experimental3.smollm2_tp1 import SmolLM2TP1Descriptor
from safetensors.parser import parse_safetensors_header


comptime CHECKPOINT_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime OUTPUT_PATH = "experimental3/smollm2_tp1_manifest.json"


fn main():
    var header_opt = parse_safetensors_header(Path(CHECKPOINT_PATH))
    if not header_opt:
        print("Failed to parse safetensors header from", CHECKPOINT_PATH)
        return
    var header = header_opt.take()
    var specs = SmolLM2TP1Descriptor.tensor_specs()

    var out = String("")
    out += "{\n"
    out += "  \"model_id\": \"" + SmolLM2TP1Descriptor.model_id() + "\",\n"
    out += "  \"files\": [\n"
    out += "    \"" + CHECKPOINT_PATH + "\"\n"
    out += "  ],\n"
    out += "  \"slot_layout\": [\n"

    for i in range(len(specs)):
        var spec = specs[i].copy()
        var meta_opt = header.tensors.get(spec.tensor_name)
        if not meta_opt:
            print("Static descriptor tensor missing from checkpoint:", spec.tensor_name)
            return
        var meta = meta_opt.value().copy()
        if not validate_shape_exact(spec.expected_shape, meta.shape):
            print("Shape mismatch while generating manifest for", spec.tensor_name)
            return
        if not validate_dtype(meta.dtype, spec.dtype):
            print("DType mismatch while generating manifest for", spec.tensor_name)
            return

        var absolute_offset = header.data_offset + meta.start
        out += "    [0, " + String(absolute_offset) + "]"
        if i + 1 != len(specs):
            out += ","
        out += "\n"

    out += "  ]\n"
    out += "}\n"

    try:
        Path(OUTPUT_PATH).write_text(out)
    except e:
        print("Failed to write package manifest:", e)
        return

    print("Wrote", OUTPUT_PATH, "with", len(specs), "static slot offsets")
