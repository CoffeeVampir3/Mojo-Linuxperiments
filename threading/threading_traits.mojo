"""Thread pool traits — trait-based interface for burst dispatch pools.

BurstThreadPool: core dispatch/join interface. Both BurstPool and
IsolatedBurstPool conform.

SleepableThreadPool: explicit sleep/wake for power management.
Only IsolatedBurstPool conforms (BurstPool self-manages via Dekker backoff).

Consumers require what they need:
    def my_kernel[P: BurstThreadPool](mut pool: P): ...
    def my_system[P: BurstThreadPool & SleepableThreadPool](mut pool: P): ...
"""

from std.memory import UnsafePointer
from .isolated_burst_pool import ArgPack


trait BurstThreadPool(Movable, ImplicitlyDestructible):
    """Core thread pool interface: dispatch work, join, measure overhead."""

    def get_capacity(self) -> Int: ...
    def get_args_base(self) -> UnsafePointer[ArgPack, MutAnyOrigin]: ...

    def dispatch[F: TrivialRegisterPassable](mut self, kernel: F,
        packs: UnsafePointer[ArgPack, MutAnyOrigin], num_jobs: Int): ...

    def join(mut self): ...

    def last_worker_timestamp(self) -> Int: ...


trait SleepableThreadPool:
    """Explicit sleep/wake for pools on isolated cores.

    sleep(): park workers (futex_wait, zero CPU burn).
    wake():  resume workers (futex_wake, back to spinning).
    """

    def wake(mut self): ...
    def sleep(mut self): ...
