"""Run ButterQuant quantization on MiniMax M2.7."""

from std.pathlib import Path
from modeling.minimax_m27_moe_butterquant_tp import MiniMaxM27ButterQuant
from quant.butterquant import run_quantizer


def main():
    var source_dir = Path("checkpoints/Minimax-M2.7")
    var output = Path("quantized_models/minimax_m27_butterquant.safetensors")

    print("source: " + String(source_dir))
    print("output: " + String(output))
    print("")

    var tasks = MiniMaxM27ButterQuant[1].build_quantizer_tasks()
    print("quantizer tasks: " + String(len(tasks)))

    var ok = run_quantizer(tasks, source_dir, output)
    if ok:
        print("\nquantization succeeded")
    else:
        print("\nquantization FAILED")
