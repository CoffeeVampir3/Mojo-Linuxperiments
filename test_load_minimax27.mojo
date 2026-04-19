"""Load test for MiniMax-M2.7 ButterQuant."""

from std.pathlib import Path
from std.time import perf_counter_ns

from modeling.minimax_m27_moe_butterquant_tp import MiniMaxM27ButterQuant


comptime MODEL_DIR = "quantized_models"
comptime TP = 1


def main():
    var t0 = perf_counter_ns()
    var model_opt = MiniMaxM27ButterQuant[TP].load(Path(MODEL_DIR))
    if not model_opt:
        return
    var model = model_opt.take()
    var t1 = perf_counter_ns()
    print("loaded in", (t1 - t0) // 1_000_000, "ms")
