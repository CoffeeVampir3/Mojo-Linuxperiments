"""Run ButterQuant quantization on Gemma4 26B-A4B."""

from std.pathlib import Path
from modeling.gemma_4_moe_butterquant_tp import Gemma4ButterQuant
from quant.butterquant import run_quantizer


def main():
    var source_dir = Path("checkpoints/gemma-4-26B-A4B")
    var output = Path("quantized_models/gemma4_butterquant.safetensors")

    print("source: " + String(source_dir))
    print("output: " + String(output))
    print("")

    var tasks = Gemma4ButterQuant[1].build_quantizer_tasks()
    print("quantizer tasks: " + String(len(tasks)))

    var ok = run_quantizer(tasks, source_dir, output)
    if ok:
        print("\nquantization succeeded")
    else:
        print("\nquantization FAILED")
