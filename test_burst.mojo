from threading.burst_threading import BurstPool, ArgPack
from notstdcollections import HeapMoveArray
from memory import MutUnsafePointer

fn empty_kernel():
    pass

fn busy_kernel():
    var x = 0
    for i in range(1000):
        x += i
    _ = x

fn write_pack_kernel(_worker: Int64, dst_addr: Int64, value: Int64):
    var ptr = MutUnsafePointer[Int64, MutOrigin.external](unsafe_from_address=Int(dst_addr))
    ptr[] = value

fn make_zero_packs(n: Int) -> HeapMoveArray[ArgPack]:
    var packs = HeapMoveArray[ArgPack](n)
    for _ in range(n):
        packs.push(ArgPack())
    return packs^

fn main():
    # Get CPU count to test oversubscription
    var cpu_count = 0
    try:
        with open("/proc/cpuinfo", "r") as f:
            var content = f.read()
            for line in content.split("\n"):
                if line.startswith("processor"):
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
    var packs1 = make_zero_packs(pool1.capacity)
    for _ in range(10):
        pool1.dispatch(empty_kernel, packs1.ptr)
    print("Exact subscription: 10 syncs completed")

    # Test with 2x oversubscription
    var oversubscribed = cpu_count * 2
    print("\n=== Test 2: 2x oversubscription (", oversubscribed, "workers) ===")
    var pool2 = BurstPool(oversubscribed)
    if not pool2:
        print("Pool creation failed")
        return
    var packs2 = make_zero_packs(pool2.capacity)
    for _ in range(10):
        pool2.dispatch(empty_kernel, packs2.ptr)
    print("2x oversubscription: 10 syncs completed")

    # Test with busy work under oversubscription
    print("\n=== Test 3: 2x oversubscription with busy work ===")
    for _ in range(10):
        pool2.dispatch(busy_kernel, packs2.ptr)
    print("2x oversubscription with busy work: 10 syncs completed")

    # Test indexed kernel data path
    print("\n=== Test 4: Per-pack kernel writes into buffer ===")
    var pool3 = BurstPool(8)
    if not pool3:
        print("Pool creation failed")
        return
    var values = HeapMoveArray[Int64](pool3.capacity)
    for _ in range(pool3.capacity):
        values.push(0)

    var packs3 = HeapMoveArray[ArgPack](pool3.capacity)
    for i in range(pool3.capacity):
        var pack = ArgPack()
        pack.arg0 = Int64(Int(values.ptr + i))
        pack.arg1 = Int64(i + 1)
        packs3.push(pack)

    pool3.dispatch(write_pack_kernel, packs3.ptr)

    var all_good = True
    for i in range(pool3.capacity):
        var value = (values.ptr + i)[]
        if value != Int64(i + 1):
            print("Kernel write mismatch at", i, "got", value)
            all_good = False
            break
    if all_good:
        print("Per-pack kernel populated data")
        print("\nAll tests passed!")
    else:
        print("Per-pack kernel failed")
