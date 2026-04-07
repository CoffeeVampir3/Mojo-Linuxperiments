"""Load DeepSeek-V2-Lite and validate weights by printing per-layer variance."""

from std.pathlib import Path
from std.time import perf_counter_ns

from modeling.deepseekv2_lite import DeepSeekV2Lite, DeepSeekV2LiteConfig

comptime C = DeepSeekV2LiteConfig
comptime MODEL_DIR = "checkpoints/deepseekv2-lite"


def main():
    DeepSeekV2Lite[1].print_memory()
    print()

    var t0 = perf_counter_ns()
    var model_opt = DeepSeekV2Lite[1].load(Path(MODEL_DIR))
    if not model_opt:
        print("load failed")
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("loaded in", load_ms, "ms")
    print()

    model.report_weight_variance()

    _ = model
