from threading.burst_threading import BurstPool
from time import perf_counter_ns, sleep

fn empty_work():
    pass

fn busy_work():
    var x = 0
    for i in range(1000):
        x += i
    _ = x

fn main():
    # Get CPU count to test oversubscription
    var cpu_count = 0
    try:
        with open("/proc/cpuinfo", "r") as f:
            var content = f.read()
            for line in content.split("\n"):
                if line[].startswith("processor"):
                    cpu_count += 1
    except:
        cpu_count = 32  # fallback

    print("Detected", cpu_count, "CPUs")

    # Test with exactly CPU count (should be fine)
    print("\n=== Test 1: Exact subscription (", cpu_count, "workers) ===")
    var pool1 = BurstPool(cpu_count)
    if not pool1:
        print("Pool creation failed")
        return

    for i in range(10):
        pool1.sync(empty_work)
    print("Exact subscription: 10 syncs completed")

    # Test with 2x oversubscription
    var oversubscribed = cpu_count * 2
    print("\n=== Test 2: 2x oversubscription (", oversubscribed, "workers) ===")
    var pool2 = BurstPool(oversubscribed)
    if not pool2:
        print("Pool creation failed")
        return

    for i in range(10):
        pool2.sync(empty_work)
    print("2x oversubscription: 10 syncs completed")

    # Test with busy work under oversubscription
    print("\n=== Test 3: 2x oversubscription with busy work ===")
    for i in range(10):
        pool2.sync(busy_work)
    print("2x oversubscription with busy work: 10 syncs completed")

    print("\nAll tests passed!")
