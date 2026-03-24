"""Pipeline infrastructure — shared types and utilities for quantization pipelines.

PipelineFn is the uniform stage signature. Timer is a simple stopwatch for
profiling any operation without knowing its signature.
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns
from threading import BurstPool


# =============================================================================
# Pipeline function signature
# =============================================================================


comptime PipelineFn = def(
    UnsafePointer[BurstPool[], MutAnyOrigin],
    UnsafePointer[UInt8, MutAnyOrigin],
    UnsafePointer[UInt8, MutAnyOrigin],
    UnsafePointer[UInt8, MutAnyOrigin],
    Int, Int,
)


# =============================================================================
# Timer — stopwatch for profiling any operation
# =============================================================================


struct Timer:
    var label: String
    var t0: UInt

    def __init__(out self, label: String):
        self.label = label
        self.t0 = perf_counter_ns()

    def stop(deinit self):
        var us = Int(perf_counter_ns() - self.t0) // 1_000
        print("      " + self.label + ": " + String(us) + "us")
