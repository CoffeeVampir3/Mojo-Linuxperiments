"""Debug test for burst threading hang."""

from threading.burst_threading import BurstPool

fn empty_work():
    pass

fn main():
    print("Testing pool size 32...")
    var pool = BurstPool(32)
    if not pool:
        print("Pool creation failed")
        return
    print("Pool created")

    for i in range(10000):
        if i % 1000 == 0:
            print("Iteration", i)
        pool.sync(empty_work)

    print("Done!")
