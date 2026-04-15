"""Long-rollout drift analysis for the dense FFN quantization path.

This reuses the dense FFN stack harness with a deeper 300-layer rollout and a
reduced number of trials so long-horizon drift is easier to inspect.
"""

from quant_analysis.dense_ffn_stack import run_dense_ffn_stack_analysis


comptime NUM_LAYERS = 300
comptime NUM_TRIALS = 2
comptime REPORT_FIRST = 8
comptime REPORT_EVERY = 25


def main():
    run_dense_ffn_stack_analysis[NUM_LAYERS, NUM_TRIALS](
        "dense_ffn_stack_300",
        report_first=REPORT_FIRST,
        report_every=REPORT_EVERY,
    )
