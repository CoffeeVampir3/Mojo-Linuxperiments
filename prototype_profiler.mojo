"""Prototype: string-keyed profiler that auto-registers phases."""

from std.collections import InlineArray


comptime MAX_PHASES = 64


struct PhaseTiming(Copyable, ImplicitlyCopyable):
    var dispatch_ns: Int
    var kernel_ns: Int
    var join_ns: Int

    def __init__(out self):
        self.dispatch_ns = 0
        self.kernel_ns = 0
        self.join_ns = 0

    def __init__(out self, dispatch_ns: Int, kernel_ns: Int, join_ns: Int):
        self.dispatch_ns = dispatch_ns
        self.kernel_ns = kernel_ns
        self.join_ns = join_ns

    @always_inline
    def total(self) -> Int:
        return self.dispatch_ns + self.kernel_ns + self.join_ns

    def add(mut self, other: Self):
        self.dispatch_ns += other.dispatch_ns
        self.kernel_ns += other.kernel_ns
        self.join_ns += other.join_ns

    @staticmethod
    def opaque(total_ns: Int) -> Self:
        return Self(0, total_ns, 0)


struct ForwardSample(Copyable, ImplicitlyCopyable):
    var pos: Int
    var wall_ns: Int
    var phases: InlineArray[PhaseTiming, MAX_PHASES]

    def __init__(out self, pos: Int):
        self.pos = pos
        self.wall_ns = 0
        self.phases = InlineArray[PhaseTiming, MAX_PHASES](fill=PhaseTiming())

    def add(mut self, phase: Int, timing: PhaseTiming):
        self.phases[phase].add(timing)

    def phase_sum_ns(self, count: Int) -> Int:
        var total = 0
        for i in range(count):
            total += self.phases[i].total()
        return total


struct ForwardLogger(Movable):
    var samples: List[ForwardSample]
    var names: List[String]

    def __init__(out self):
        self.samples = List[ForwardSample]()
        self.names = List[String]()

    def clear(mut self):
        self.samples = List[ForwardSample]()

    def phase(mut self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        var idx = len(self.names)
        self.names.append(name)
        return idx

    def record(mut self, sample: ForwardSample):
        self.samples.append(sample)

    def report(self, label: String):
        var n = len(self.samples)
        var np = len(self.names)
        if n == 0:
            print(label + ": no samples")
            return

        print(label + " (" + String(n) + " samples, " + String(np) + " phases)")
        for p in range(np):
            var sum_ns = 0
            for i in range(n):
                sum_ns += self.samples[i].phases[p].total()
            var avg_ns = sum_ns // n
            var avg_ms = Float64(avg_ns) / 1_000_000.0
            print("  " + self.names[p] + ": avg " + String(avg_ms) + " ms")


def main():
    var logger = ForwardLogger()

    for tok in range(5):
        var sample = ForwardSample(tok)

        # Simulate recording phases — names auto-register on first use
        sample.add(logger.phase("embed"), PhaseTiming(100, 5000, 50))
        sample.add(logger.phase("broadcast"), PhaseTiming.opaque(2000))
        sample.add(logger.phase("local_attn_proj"), PhaseTiming(200, 30000, 100))
        sample.add(logger.phase("local_attn_proj"), PhaseTiming(150, 25000, 80))
        sample.add(logger.phase("global_attn_proj"), PhaseTiming(200, 40000, 100))
        sample.add(logger.phase("router_quantize"), PhaseTiming(100, 3000, 50))
        sample.add(logger.phase("expert_phase1"), PhaseTiming(300, 10000, 100))

        sample.wall_ns = 200000
        logger.record(sample)

    logger.report("prototype test")
    print("")
    print("phase count: " + String(len(logger.names)))
    print("phase_sum_ns(sample[0]): " + String(logger.samples[0].phase_sum_ns(len(logger.names))))
