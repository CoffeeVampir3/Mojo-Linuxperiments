"""End-to-end quantizer test: quantize SmolLM2-135M to int8 channelwise
with per-shape VNNI packing.

The model descriptor owns the quantization specification. We just call
quantize on it.
"""

from std.pathlib import Path
from modeling.smollm2_int8ch_tp import Int8TPModel


comptime SOURCE_PATH = "checkpoints/SmolLM2/model.safetensors"
comptime OUTPUT_PATH = "quantized_scratch/model_int8_packed.safetensors"


def main():
    print("=== SmolLM2-135M int8 channelwise + VNNI pack ===\n")

    var ok = Int8TPModel[1].quantize(Path(SOURCE_PATH), Path(OUTPUT_PATH))

    if ok:
        print("\nsuccess: quantized model at " + OUTPUT_PATH)
    else:
        print("\nfailed")
