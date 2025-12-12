"""Debug test for burst threading hang."""

from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray

fn empty_kernel():
    pass

fn main():
    print("Testing pool size 32...")
    var pool = BurstPool(32)
    if not pool:
        print("Pool creation failed")
        return
    print("Pool created")

    var packs = HeapMoveArray[ArgPack](pool.capacity)
    for _ in range(pool.capacity):
        packs.push(ArgPack())

    for i in range(10000):
        if i % 1000 == 0:
            print("Iteration", i)
        pool.dispatch(empty_kernel, packs.ptr)

    print("Done!")
