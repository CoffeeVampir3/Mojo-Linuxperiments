from time import perf_counter_ns


@fieldwise_init
struct ProfileSection(Copyable):
    var name: String
    var ns: UInt


struct Profiler:
    """Lightweight section-based profiler. Each section() call ends the
    previous section and starts a new one. Sections with the same name
    accumulate. No-ops when disabled."""
    var sections: List[ProfileSection]
    var enabled: Bool
    var t: UInt
    var current: String

    fn __init__(out self, enabled: Bool):
        self.sections = List[ProfileSection]()
        self.enabled = enabled
        self.t = 0
        self.current = String("")

    @always_inline
    fn section(mut self, name: String):
        if not self.enabled:
            return
        var now = perf_counter_ns()
        if self.t > 0:
            self._accumulate(now - self.t)
        self.t = now
        self.current = name

    @always_inline
    fn finish(mut self):
        if not self.enabled or self.t == 0:
            return
        self._accumulate(perf_counter_ns() - self.t)
        self.t = 0

    fn _accumulate(mut self, elapsed: UInt):
        for i in range(len(self.sections)):
            if self.sections[i].name == self.current:
                self.sections[i].ns += elapsed
                return
        self.sections.append(ProfileSection(self.current, elapsed))

    fn report(self):
        if not self.enabled or len(self.sections) == 0:
            return
        var total: UInt = 0
        for i in range(len(self.sections)):
            total += self.sections[i].ns
        print("--- forward profile (us) ---")
        for i in range(len(self.sections)):
            print("  " + self.sections[i].name + ":", Int(self.sections[i].ns / 1_000))
        print("  total:", Int(total / 1_000))
