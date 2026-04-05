"""Trait-based interface for burst dispatch pools.
"""

from std.memory import UnsafePointer
from .threading_shared import ArgPack


trait BurstThreadPool(Movable, ImplicitlyDestructible):
    def get_capacity(self) -> Int: ...
    def get_args_base(self) -> UnsafePointer[ArgPack, MutAnyOrigin]: ...

    def dispatch[T0: TrivialRegisterPassable, T1: TrivialRegisterPassable,
        T2: TrivialRegisterPassable, T3: TrivialRegisterPassable,
        T4: TrivialRegisterPassable, T5: TrivialRegisterPassable](
        mut self, kernel: def(T0, T1, T2, T3, T4, T5) -> None,
        packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int): ...

    def join(mut self): ...

    def last_worker_timestamp(self) -> Int: ...


trait SleepableThreadPool:
    def wake(mut self): ...
    def sleep(mut self): ...
