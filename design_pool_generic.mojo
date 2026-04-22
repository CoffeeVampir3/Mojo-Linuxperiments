from std.memory import UnsafePointer
from std.collections import InlineArray
from notstdcollections import HeapMoveArray


trait BurstThreadPool(Movable, ImplicitlyDestructible):
    def get_capacity(self) -> Int: ...
    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int): ...
    def join(mut self): ...
    def last_worker_timestamp(self) -> Int: ...


trait SleepableThreadPool:
    def wake(mut self): ...
    def sleep(mut self): ...


struct MockSerialPool(BurstThreadPool):
    var cap: Int
    var ts: Int
    var marker: Int

    def __init__(out self, capacity: Int, marker: Int):
        self.cap = capacity
        self.ts = 0
        self.marker = marker

    def get_capacity(self) -> Int:
        return self.cap

    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        var n = num_jobs if num_jobs >= 0 else self.cap
        for i in range(n):
            kernel((args + i)[])
        self.ts += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts * 100 + self.marker


struct MockBatchedPool(BurstThreadPool):
    var cap: Int
    var ts: Int
    var marker: Int

    def __init__(out self, capacity: Int, marker: Int):
        self.cap = capacity
        self.ts = 0
        self.marker = marker

    def get_capacity(self) -> Int:
        return self.cap

    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        var n = num_jobs if num_jobs >= 0 else self.cap
        for i in range(n):
            kernel((args + i)[])
            self.ts += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts * 1000 + self.marker


struct MockIsolatedPool(BurstThreadPool, SleepableThreadPool):
    var cap: Int
    var ts: Int
    var marker: Int
    var asleep: Bool

    def __init__(out self, capacity: Int, marker: Int):
        self.cap = capacity
        self.ts = 0
        self.marker = marker
        self.asleep = False

    def get_capacity(self) -> Int:
        return self.cap

    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        var n = num_jobs if num_jobs >= 0 else self.cap
        for i in range(n):
            kernel((args + i)[])
        self.ts += n

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts * 10000 + self.marker

    def wake(mut self):
        self.asleep = False

    def sleep(mut self):
        self.asleep = True


struct MockParamPool[mask: Int = 128](BurstThreadPool):
    var cap: Int
    var ts: Int
    var marker: Int

    def __init__(out self, capacity: Int, marker: Int):
        self.cap = capacity
        self.ts = 0
        self.marker = marker

    def get_capacity(self) -> Int:
        return self.cap

    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def(Args) thin -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        var n = num_jobs if num_jobs >= 0 else self.cap
        for i in range(n):
            kernel((args + i)[])
        self.ts += n

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts * 100000 + self.marker


@fieldwise_init
struct NumaConfig(Copyable, ImplicitlyCopyable):
    var num_nodes: Int
    var threads_per_node: Int


def pool_from_numa(numa: NumaConfig, node: Int) -> MockBatchedPool:
    return MockBatchedPool(numa.threads_per_node, node)


@explicit_destroy
@fieldwise_init
struct PoolFence[P: BurstThreadPool](Movable):
    var pool: UnsafePointer[Self.P, MutAnyOrigin]

    @staticmethod
    def completed() -> Self:
        return Self(UnsafePointer[Self.P, MutAnyOrigin]())

    def join(deinit self):
        if self.pool:
            self.pool[].join()

    def take(deinit self) -> UnsafePointer[Self.P, MutAnyOrigin]:
        return self.pool


def parallel_for[P: BurstThreadPool, tp: Int,
    body: def[rank: Int] () capturing -> PoolFence[P]]():
    var ptrs = InlineArray[UnsafePointer[P, MutAnyOrigin], tp](
        fill=UnsafePointer[P, MutAnyOrigin]())
    comptime for rank in range(tp):
        ptrs[rank] = body[rank]().take()
    for i in range(tp):
        if ptrs[i]:
            ptrs[i][].join()


@fieldwise_init
struct ScaleArgs(Copyable, ImplicitlyCopyable):
    var src: UnsafePointer[Int, MutAnyOrigin]
    var dst: UnsafePointer[Int, MutAnyOrigin]
    var scale: Int
    var start: Int
    var end: Int


def scale_kernel(args: ScaleArgs):
    for i in range(args.start, args.end):
        args.dst[i] = args.src[i] * args.scale


@fieldwise_init
struct Topology(Copyable, ImplicitlyCopyable):
    var arena_base: Int
    var src_off: Int
    var dst_off: Int
    var n: Int


def scale_op[P: BurstThreadPool, max_jobs: Int = 16](
    topo: Topology, scale_val: Int, mut pool: P,
) -> PoolFence[P]:
    if topo.n == 0:
        return PoolFence[P].completed()
    var num_jobs = min(topo.n, pool.get_capacity())
    var rows_per_job = (topo.n + num_jobs - 1) // num_jobs
    var src = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=topo.arena_base + topo.src_off)
    var dst = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=topo.arena_base + topo.dst_off)
    var jobs = InlineArray[ScaleArgs, max_jobs](
        fill=ScaleArgs(src, src, 0, 0, 0))
    for i in range(num_jobs):
        var start = i * rows_per_job
        var end = min(start + rows_per_job, topo.n)
        jobs[i] = ScaleArgs(src, dst, scale_val, start, end)
    pool.dispatch[ScaleArgs, scale_kernel](
        UnsafePointer(to=jobs[0]), num_jobs)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))


def tp_parallel[
    Pool: BurstThreadPool, //,
    tp: Int,
    body: def[rank: Int](Topology, mut Pool) capturing -> PoolFence[Pool],
](
    topos: InlineArray[Topology, tp],
    pool_ptrs: InlineArray[UnsafePointer[Pool, MutAnyOrigin], tp],
):
    @parameter
    def dispatch[rank: Int]() -> PoolFence[Pool]:
        return body[rank](topos[rank], pool_ptrs[rank][])
    parallel_for[Pool, tp, dispatch]()


struct ToyModel[tp: Int, Pool: BurstThreadPool = MockSerialPool](Movable):
    var pools: HeapMoveArray[Self.Pool]
    var storage: InlineArray[Int, 1024]

    def __init__(out self, var pools: HeapMoveArray[Self.Pool]):
        self.pools = pools^
        self.storage = InlineArray[Int, 1024](fill=0)
        for i in range(1024):
            self.storage[i] = i

    def pool_ptrs(self) -> InlineArray[UnsafePointer[Self.Pool, MutAnyOrigin], Self.tp]:
        var ptrs = InlineArray[UnsafePointer[Self.Pool, MutAnyOrigin], Self.tp](
            fill=UnsafePointer[Self.Pool, MutAnyOrigin]())
        for rank in range(Self.tp):
            ptrs[rank] = UnsafePointer[Self.Pool, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.pools[rank])))
        return ptrs^

    def rank_topos(self) -> InlineArray[Topology, Self.tp]:
        var out = InlineArray[Topology, Self.tp](fill=Topology(0, 0, 0, 0))
        var base = Int(UnsafePointer(to=self.storage[0]))
        for rank in range(Self.tp):
            out[rank] = Topology(base, 0, 512 * 8, 128)
        return out^

    def forward(mut self, scale_val: Int) -> Int:
        var mp = self.pool_ptrs()
        var topos = self.rank_topos()

        @parameter
        def do_scale[rank: Int](
            topo: Topology, mut pool: Self.Pool,
        ) -> PoolFence[Self.Pool]:
            return scale_op(topo, scale_val, pool)

        tp_parallel[Self.tp, do_scale](topos, mp)

        var check = 0
        for rank in range(Self.tp):
            check += self.pools[rank].last_worker_timestamp()
        return check


def load_array[
    Pool: BurstThreadPool, //, tp: Int,
](var pools: HeapMoveArray[Pool]) -> ToyModel[tp, Pool]:
    return ToyModel[tp, Pool](pools^)


def load_factory[
    Pool: BurstThreadPool, //,
    tp: Int,
    make_pool: def(Int) capturing -> Pool,
]() -> ToyModel[tp, Pool]:
    var pools = HeapMoveArray[Pool](tp)
    for rank in range(tp):
        pools.push(make_pool(rank))
    return ToyModel[tp, Pool](pools^)


def park_all[tp: Int, Pool: BurstThreadPool & SleepableThreadPool](
    mut model: ToyModel[tp, Pool],
):
    for rank in range(tp):
        model.pools[rank].sleep()


def unpark_all[tp: Int, Pool: BurstThreadPool & SleepableThreadPool](
    mut model: ToyModel[tp, Pool],
):
    for rank in range(tp):
        model.pools[rank].wake()


def case1_array():
    var pools = HeapMoveArray[MockSerialPool](4)
    for n in range(4):
        pools.push(MockSerialPool(1, n))
    var m = load_array[4](pools^)
    print("case1 array  :", m.forward(3))


def case1_factory():
    @parameter
    def mk(n: Int) -> MockSerialPool:
        return MockSerialPool(1, n)
    var m = load_factory[4, mk]()
    print("case1 factory:", m.forward(3))


def case2_array():
    var numa = NumaConfig(2, 8)
    var pools = HeapMoveArray[MockBatchedPool](4)
    for n in range(4):
        pools.push(pool_from_numa(numa, n))
    var m = load_array[4](pools^)
    print("case2 array  :", m.forward(3))


def case2_factory():
    var numa = NumaConfig(2, 8)
    @parameter
    def mk(n: Int) -> MockBatchedPool:
        return pool_from_numa(numa, n)
    var m = load_factory[4, mk]()
    print("case2 factory:", m.forward(3))


def case3_array():
    var pools = HeapMoveArray[MockParamPool[256]](4)
    for n in range(4):
        pools.push(MockParamPool[256](4, n))
    var m = load_array[4](pools^)
    print("case3 array  :", m.forward(3))


def case3_factory():
    @parameter
    def mk(n: Int) -> MockParamPool[256]:
        return MockParamPool[256](4, n)
    var m = load_factory[4, mk]()
    print("case3 factory:", m.forward(3))


def has_isolcpus() -> Bool:
    return False


def case4_array():
    if has_isolcpus():
        var pools = HeapMoveArray[MockIsolatedPool](4)
        for n in range(4):
            pools.push(MockIsolatedPool(2, n))
        var m = load_array[4](pools^)
        print("case4 array  : isolated", m.forward(3))
    else:
        var pools = HeapMoveArray[MockSerialPool](4)
        for n in range(4):
            pools.push(MockSerialPool(1, n))
        var m = load_array[4](pools^)
        print("case4 array  : serial  ", m.forward(3))


def case4_factory():
    if has_isolcpus():
        @parameter
        def mk_iso(n: Int) -> MockIsolatedPool:
            return MockIsolatedPool(2, n)
        var m = load_factory[4, mk_iso]()
        print("case4 factory: isolated", m.forward(3))
    else:
        @parameter
        def mk_ser(n: Int) -> MockSerialPool:
            return MockSerialPool(1, n)
        var m = load_factory[4, mk_ser]()
        print("case4 factory: serial  ", m.forward(3))


def case5_array():
    var pools = HeapMoveArray[MockIsolatedPool](2)
    pools.push(MockIsolatedPool(2, 0))
    pools.push(MockIsolatedPool(2, 1))
    var m = load_array[2](pools^)
    park_all(m)
    var a = m.pools[0].asleep
    unpark_all(m)
    var b = m.pools[0].asleep
    print("case5 array  : park=", a, " unpark=", b)


def case5_factory():
    @parameter
    def mk(n: Int) -> MockIsolatedPool:
        return MockIsolatedPool(2, n)
    var m = load_factory[2, mk]()
    park_all(m)
    var a = m.pools[0].asleep
    unpark_all(m)
    var b = m.pools[0].asleep
    print("case5 factory: park=", a, " unpark=", b)


def main():
    case1_array(); case1_factory()
    case2_array(); case2_factory()
    case3_array(); case3_factory()
    case4_array(); case4_factory()
    case5_array(); case5_factory()
