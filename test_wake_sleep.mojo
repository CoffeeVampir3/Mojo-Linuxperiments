"""Stress test for wake/sleep cycling on IsolatedModel.

Verifies:
  1. Basic wake → dispatch → join → sleep cycle
  2. Rapid wake/sleep toggling without dispatch
  3. Multiple dispatch/join rounds within a single wake window
  4. Wake after extended sleep (workers must actually park and unpark)
  5. Sleep is idempotent (double sleep doesn't break)
  6. Wake is idempotent (double wake doesn't break)
"""

from threading.burst_threading import BurstPool, IsolatedModel, ColdModel, ArgPack
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from numa import NumaInfo
from notstdcollections import HeapMoveArray


def noop_kernel(out_addr: Int, job_id: Int, unused0: Int,
                unused1: Int, unused2: Int, unused3: Int):
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = job_id + 1


def accumulate_kernel(out_addr: Int, job_id: Int, rounds: Int,
                      unused1: Int, unused2: Int, unused3: Int):
    var x = job_id + 1
    for _ in range(rounds):
        x = x ^ (x >> 17)
        x = x * 0xBF58476D1CE4E5B9
        x = x ^ (x >> 31)
    UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_addr)[] = x


def main():
    var numa = NumaInfo()
    if not numa.has_isolation():
        print("SKIP: no CPU isolation configured")
        return

    var node = numa.plan_topology(1)[0]
    var pool = BurstPool[IsolatedModel].for_topology(numa, node)
    if not pool:
        print("FAIL: pool creation failed")
        return

    var cap = pool.capacity
    print("IsolatedModel: " + String(cap) + " workers on node " + String(node))

    var output = HeapMoveArray[Int](cap)
    for _ in range(cap):
        output.push(0)

    var packs = HeapMoveArray[ArgPack](cap)
    for _ in range(cap):
        packs.push(ArgPack())

    for j in range(cap):
        (packs.ptr + j)[].arg0 = Int(output.ptr + j)
        (packs.ptr + j)[].arg1 = j

    var passed = 0
    var failed = 0

    # --- Test 1: basic wake → dispatch → join → sleep ---
    pool.wake()
    pool.dispatch(noop_kernel, packs.ptr)
    pool.join()
    var ok = True
    for j in range(cap):
        if output[j] != j + 1:
            ok = False
            break
    pool.sleep()
    if ok:
        print("PASS: basic wake/dispatch/join/sleep")
        passed += 1
    else:
        print("FAIL: basic wake/dispatch/join/sleep — wrong output")
        failed += 1

    # --- Test 2: rapid wake/sleep toggling (no dispatch) ---
    ok = True
    for _ in range(1000):
        pool.wake()
        pool.sleep()
    # Verify pool still works after rapid toggling
    pool.wake()
    for j in range(cap):
        (output.ptr + j)[] = 0
    pool.dispatch(noop_kernel, packs.ptr)
    pool.join()
    for j in range(cap):
        if output[j] != j + 1:
            ok = False
            break
    pool.sleep()
    if ok:
        print("PASS: 1000x rapid wake/sleep toggle")
        passed += 1
    else:
        print("FAIL: 1000x rapid wake/sleep toggle — wrong output")
        failed += 1

    # --- Test 3: multiple dispatch/join within one wake window ---
    pool.wake()
    ok = True
    for round in range(100):
        for j in range(cap):
            (output.ptr + j)[] = 0
            (packs.ptr + j)[].arg1 = j + round
        pool.dispatch(noop_kernel, packs.ptr)
        pool.join()
        for j in range(cap):
            if output[j] != j + round + 1:
                ok = False
                break
        if not ok:
            break
    pool.sleep()
    if ok:
        print("PASS: 100 dispatch/join rounds in one wake window")
        passed += 1
    else:
        print("FAIL: 100 dispatch/join rounds in one wake window")
        failed += 1

    # --- Test 4: wake after extended sleep (workers must actually park) ---
    pool.sleep()
    # Wait long enough for workers to enter futex_wait
    var t0 = Int(perf_counter_ns())
    while Int(perf_counter_ns()) - t0 < 50_000_000:  # 50ms
        pass
    pool.wake()
    for j in range(cap):
        (output.ptr + j)[] = 0
        (packs.ptr + j)[].arg1 = j
    pool.dispatch(noop_kernel, packs.ptr)
    pool.join()
    ok = True
    for j in range(cap):
        if output[j] != j + 1:
            ok = False
            break
    pool.sleep()
    if ok:
        print("PASS: wake after 50ms sleep")
        passed += 1
    else:
        print("FAIL: wake after 50ms sleep — wrong output")
        failed += 1

    # --- Test 5: double sleep is idempotent ---
    pool.sleep()
    pool.sleep()
    pool.wake()
    for j in range(cap):
        (output.ptr + j)[] = 0
        (packs.ptr + j)[].arg1 = j
    pool.dispatch(noop_kernel, packs.ptr)
    pool.join()
    ok = True
    for j in range(cap):
        if output[j] != j + 1:
            ok = False
            break
    pool.sleep()
    if ok:
        print("PASS: double sleep idempotent")
        passed += 1
    else:
        print("FAIL: double sleep idempotent")
        failed += 1

    # --- Test 6: double wake is idempotent ---
    pool.wake()
    pool.wake()
    for j in range(cap):
        (output.ptr + j)[] = 0
        (packs.ptr + j)[].arg1 = j
    pool.dispatch(noop_kernel, packs.ptr)
    pool.join()
    ok = True
    for j in range(cap):
        if output[j] != j + 1:
            ok = False
            break
    pool.sleep()
    if ok:
        print("PASS: double wake idempotent")
        passed += 1
    else:
        print("FAIL: double wake idempotent")
        failed += 1

    # --- Test 7: heavy work across wake/sleep boundaries ---
    ok = True
    for round in range(10):
        pool.wake()
        for j in range(cap):
            (output.ptr + j)[] = 0
            (packs.ptr + j)[].arg1 = j + 1
            (packs.ptr + j)[].arg2 = 10000
        pool.dispatch(accumulate_kernel, packs.ptr)
        pool.join()
        for j in range(cap):
            if output[j] == 0:
                ok = False
                break
        pool.sleep()
        if not ok:
            break
    if ok:
        print("PASS: 10x heavy wake/dispatch/join/sleep cycles")
        passed += 1
    else:
        print("FAIL: 10x heavy wake/dispatch/join/sleep cycles")
        failed += 1

    print("\n" + String(passed) + "/" + String(passed + failed) + " passed")
    if failed > 0:
        print("FAILURES DETECTED")
