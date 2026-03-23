"""Unified design: PoolFence with .join() + .take(), satisfying both
linear safety and parametric parallel_for."""

from threading import BurstPool
from memory import UnsafePointer
from collections import InlineArray


@explicit_destroy
@fieldwise_init
struct PoolFence(Movable):
    var pool: UnsafePointer[BurstPool, MutAnyOrigin]

    @staticmethod
    fn completed() -> Self:
        return Self(UnsafePointer[BurstPool, MutAnyOrigin]())

    fn join(deinit self):
        """Consume fence, wait for work to complete."""
        if self.pool:
            self.pool[].join()

    fn take(deinit self) -> UnsafePointer[BurstPool, MutAnyOrigin]:
        """Consume fence, return raw pool pointer for deferred join."""
        return self.pool


# ================================================================
# parallel() — variadic barrier (fixed TP, direct use)
# ================================================================

fn parallel(var *fences: PoolFence):
    @parameter
    fn do_join(idx: Int, var fence: PoolFence) capturing:
        fence^.join()
    fences^.consume_elements[do_join]()


# ================================================================
# parallel_for — parametric barrier (any TP, closure-based)
# ================================================================

fn parallel_for[tp: Int, body: fn[rank: Int] () capturing -> PoolFence]():
    """Dispatch body[rank]() for each rank, then join all.
    body returns PoolFence — consumed internally via .take()."""
    var ptrs = InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], tp](
        fill=UnsafePointer[BurstPool, MutAnyOrigin]()
    )
    @parameter
    for rank in range(tp):
        ptrs[rank] = body[rank]().take()
    for i in range(tp):
        if ptrs[i]:
            ptrs[i][].join()


# ================================================================
# Fake kernel (returns PoolFence, same as real gemm/rmsnorm/etc.)
# ================================================================

fn fake_kernel(mut pool: BurstPool) -> PoolFence:
    return PoolFence(UnsafePointer[BurstPool, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))
    ))


# ================================================================
# Test: TP=1 direct
# ================================================================

fn test_tp1(mut pool: BurstPool):
    fake_kernel(pool).join()


# ================================================================
# Test: TP=3 variadic (the fixed-arity API)
# ================================================================

fn test_tp3_variadic(mut p0: BurstPool, mut p1: BurstPool, mut p2: BurstPool):
    parallel(
        fake_kernel(p0),
        fake_kernel(p1),
        fake_kernel(p2),
    )


# ================================================================
# Test: TP=3 parametric (the generic API)
# ================================================================

fn test_tp3_parametric(
    mut p0: BurstPool, mut p1: BurstPool, mut p2: BurstPool,
):
    @parameter
    fn work[rank: Int]() -> PoolFence:
        @parameter
        if rank == 0: return fake_kernel(p0)
        elif rank == 1: return fake_kernel(p1)
        else: return fake_kernel(p2)
    parallel_for[3, work]()


# ================================================================
# Test: TP=N fully parametric struct
# ================================================================

struct TPModel[tp: Int]:
    var pool_ptrs: InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], Self.tp]

    fn __init__(out self):
        self.pool_ptrs = InlineArray[UnsafePointer[BurstPool, MutAnyOrigin], Self.tp](
            fill=UnsafePointer[BurstPool, MutAnyOrigin]()
        )

    fn set_pool(mut self, rank: Int, mut pool: BurstPool):
        self.pool_ptrs[rank] = UnsafePointer[BurstPool, MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pool))
        )

    fn forward(mut self):
        @parameter
        fn do_gemm[rank: Int]() -> PoolFence:
            return fake_kernel(self.pool_ptrs[rank][])
        parallel_for[Self.tp, do_gemm]()

        @parameter
        fn do_attn[rank: Int]() -> PoolFence:
            return fake_kernel(self.pool_ptrs[rank][])
        parallel_for[Self.tp, do_attn]()


fn test_tp_model_2(mut p0: BurstPool, mut p1: BurstPool):
    var model = TPModel[2]()
    model.set_pool(0, p0)
    model.set_pool(1, p1)
    model.forward()


fn test_tp_model_4(mut p0: BurstPool, mut p1: BurstPool, mut p2: BurstPool, mut p3: BurstPool):
    var model = TPModel[4]()
    model.set_pool(0, p0)
    model.set_pool(1, p1)
    model.set_pool(2, p2)
    model.set_pool(3, p3)
    model.forward()


fn main():
    pass
