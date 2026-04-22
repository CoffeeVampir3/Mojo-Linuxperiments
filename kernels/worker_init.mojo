from std.memory import UnsafePointer
from std.collections import InlineArray
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence, MAX_POOL_CAPACITY
from experimental3.amx import (
    make_224_decode_config, init_intel_amx, ldtilecfg,
)
from simd_math import set_subnormal_zeroing


@fieldwise_init
struct WorkerInitArgs(Copyable, ImplicitlyCopyable):
    var dummy: Int

    def __init__(out self):
        self.dummy = 0


def worker_init_kernel[heads_per_group: Int](args: WorkerInitArgs):
    _ = init_intel_amx()
    set_subnormal_zeroing()
    var cfg = make_224_decode_config[heads_per_group]()
    ldtilecfg(UnsafePointer(to=cfg))


def worker_init_dispatch[heads_per_group: Int, P: BurstThreadPool](
    mut pool: P,
) -> PoolFence[P]:
    var jobs = InlineArray[WorkerInitArgs, MAX_POOL_CAPACITY](
        fill=WorkerInitArgs())
    var cap = pool.get_capacity()
    pool.dispatch[WorkerInitArgs, worker_init_kernel[heads_per_group]](
        UnsafePointer(to=jobs[0]), cap)
    return PoolFence[P](UnsafePointer[P, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=pool))))
