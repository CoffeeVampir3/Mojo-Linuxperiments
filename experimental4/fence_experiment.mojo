"""Experiment: dispatch_group using consume_elements on owned VariadicPack."""

from threading import BurstPool
from memory import UnsafePointer


@explicit_destroy
@fieldwise_init
struct PoolFence(Movable):
    var pool: UnsafePointer[BurstPool, MutAnyOrigin]

    @staticmethod
    fn completed() -> Self:
        return Self(UnsafePointer[BurstPool, MutAnyOrigin]())

    fn join(deinit self):
        if self.pool:
            self.pool[].join()


fn dispatch_group(var *fences: PoolFence):
    @parameter
    fn do_join(idx: Int, var fence: PoolFence) capturing:
        fence^.join()
    fences^.consume_elements[do_join]()


fn fake_dispatch(mut pool: BurstPool) -> PoolFence:
    return PoolFence(UnsafePointer[BurstPool, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


fn test_tp1(mut pool: BurstPool):
    fake_dispatch(pool).join()

fn test_tp3(mut p0: BurstPool, mut p1: BurstPool, mut p2: BurstPool):
    dispatch_group(
        fake_dispatch(p0),
        fake_dispatch(p1),
        fake_dispatch(p2),
    )

fn test_tp4(mut p0: BurstPool, mut p1: BurstPool, mut p2: BurstPool, mut p3: BurstPool):
    dispatch_group(
        fake_dispatch(p0),
        fake_dispatch(p1),
        fake_dispatch(p2),
        fake_dispatch(p3),
    )

fn main():
    pass
