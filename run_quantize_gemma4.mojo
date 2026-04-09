"""Run ButterQuant quantization on Gemma4 26B-A4B."""

from std.pathlib import Path
from modeling.gemma_4_moe_butterquant_tp import Gemma4Model
from quant.butterquant import quantize


def main():
    var source_dir = Path("checkpoints/gemma-4-26B-A4B")
    var output = Path("quantized_models/gemma4_butterquant.safetensors")

    print("source: " + String(source_dir))
    print("output: " + String(output))
    print("")

    var ok = quantize[Gemma4Model[1]](source_dir, output)
    if ok:
        print("\nquantization succeeded")
    else:
        print("\nquantization FAILED")
