"""Run Hadamard quantization on SmolLM2-135M."""

from std.pathlib import Path
from modeling.smollm2_hadquant_tp import HadQuantTPModel

def main():
    var source = Path("checkpoints/SmolLM2/model.safetensors")
    var output = Path("quantized_models/smollm2_hadquant.safetensors")

    print("source: " + String(source))
    print("output: " + String(output))
    print("")

    var ok = HadQuantTPModel[1].quantize(source, output)
    if ok:
        print("\nquantization succeeded")
    else:
        print("\nquantization FAILED")
