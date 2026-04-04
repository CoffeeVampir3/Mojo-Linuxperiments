"""Quantize SmolLM2-135M to ButterQuant i8 format."""

from std.pathlib import Path
from modeling.smollm2_butterquant_tp import ButterQuantTPModel
from std.os import mkdir

comptime SOURCE = "checkpoints/SmolLM2/model.safetensors"
comptime OUTPUT = "quantized_models/SmolLM2-ButterQuant/model.safetensors"


def main() raises:
    var out_dir = Path("quantized_models/SmolLM2-ButterQuant")
    if not out_dir.exists():
        mkdir(out_dir)

    if ButterQuantTPModel[].quantize(Path(SOURCE), Path(OUTPUT)):
        print("done: " + OUTPUT)
    else:
        print("quantization failed")
