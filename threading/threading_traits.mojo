"""Trait-based interface for burst dispatch pools.
"""

from std.memory import UnsafePointer


trait BurstThreadPool(Movable, ImplicitlyDestructible):
    def get_capacity(self) -> Int: ...

    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int): ...

    def join(mut self): ...

    def last_worker_timestamp(self) -> Int: ...


trait SleepableThreadPool:
    def wake(mut self): ...
    def sleep(mut self): ...
