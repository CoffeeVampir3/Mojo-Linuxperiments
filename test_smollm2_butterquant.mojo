"""SmolLM2 ButterQuant — load quantized model + stub forward pass."""

from std.pathlib import Path
from modeling.smollm2_butterquant_tp import SmolLM2ButterQuant


def main():
    SmolLM2ButterQuant[3].print_memory()

    var model_opt = SmolLM2ButterQuant[3].load(
        Path("quantized_models/SmolLM2-ButterQuant/model.safetensors")
    )
    if not model_opt:
        return
    var model = model_opt.take()
    print("Model loaded successfully.\n")

    print("=== Forward pass (stub) ===")
    var tp = model.token_buffer()
    tp[0] = Scalar[DType.int32](42)
    print("calling forward...")
    var logits = model.forward(Int(tp), 1, 0)
    print("forward returned, releasing logits...")
    logits^.release()
    print("logits released")
    _ = model
