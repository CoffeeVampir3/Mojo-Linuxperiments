from std.memory import Pointer, Span
from threading.threading_traits import BurstThreadPool

from experimental3.common_math import BF16Ptr, F32Ptr
from kernels.kernel_ops import PoolFence
from modeling.linear_borrow_pool import ScratchLease
from modeling.minimax_m27_moe_butterquant_tp import (
    C,
    LayerRefs,
    MiniMaxM27Topology,
    MiniMaxShapes,
)
from minimax.kernels.dispatch_kernels import kv_write_dispatch


@fieldwise_init
struct M27AttentionLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var topo: MiniMaxM27Topology[Self.tp]
    var layer_idx: Int

    @always_inline
    def layer_base(self) -> Int:
        return self.topo.layers.base(self.topo.arena.base, self.layer_idx)

    @always_inline
    def layer(self) -> LayerRefs[Self.tp]:
        return self.topo.layers.proto

    @always_inline
    def rope_cos0(self) -> F32Ptr:
        return self.topo.rope.cos.bound_row(
            self.topo.arena.base, 0).as_ptr[DType.float32]()

    @always_inline
    def rope_sin0(self) -> F32Ptr:
        return self.topo.rope.sin.bound_row(
            self.topo.arena.base, 0).as_ptr[DType.float32]()

    @always_inline
    def kv_cache_base(self) -> Int:
        return (
            self.topo.arena.base
            + self.topo.kv_cache_off
            + self.layer_idx * self.topo.kv_cache_stride
        )

    @always_inline
    def k_norm(self) -> BF16Ptr:
        return self.layer().attn.k_norm.bound(
            self.layer_base()).as_ptr[DType.bfloat16]()


@fieldwise_init
struct M27InvRmsRows[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var rows: Span[Float32, Self.origin]

    @always_inline
    def ptr(self) -> F32Ptr:
        return self.rows.unsafe_ptr()


@fieldwise_init
struct M27QkvScratch[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var scratch_base: Int
    var qkv_lease: Pointer[ScratchLease, Self.origin]

    @always_inline
    def base_ptr(self) -> BF16Ptr:
        return self.qkv_lease[].as_ptr[Scalar[DType.bfloat16]](
            self.scratch_base)


@always_inline
def bind_qkv_scratch[origin: MutOrigin](
    ref [origin] qkv_lease: ScratchLease,
    scratch_base: Int,
) -> M27QkvScratch[origin]:
    return M27QkvScratch[origin](scratch_base, Pointer(to=qkv_lease))


@always_inline
def kv_write_probe[
    tp: Int,
    P: BurstThreadPool,
    pool_origin: MutOrigin,
    qkv_origin: MutOrigin,
    inv_origin: MutOrigin,
](
    rank_layer: M27AttentionLayer[tp],
    qkv: M27QkvScratch[qkv_origin],
    inv_rms_k: M27InvRmsRows[inv_origin],
    start_pos: Int,
    seq_len: Int,
    ref [pool_origin] pool: P,
) -> PoolFence[P, pool_origin]:
    comptime S = MiniMaxShapes[tp]
    comptime Q_LOCAL = S.Q_LOCAL
    comptime KV_PER_RANK = S.KV_LOCAL // C.HEAD_DIM
    comptime QKV_LOCAL = S.QKV_LOCAL
    comptime ROPE_HALF = C.ROPE_DIM // 2
    return kv_write_dispatch[
        C.HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
        C.MAX_SEQ_LEN, KV_PER_RANK, QKV_LOCAL,
        Q_LOCAL, ROPE_HALF,
    ](
        qkv.base_ptr(),
        rank_layer.k_norm(),
        rank_layer.rope_cos0(),
        rank_layer.rope_sin0(),
        inv_rms_k.ptr(),
        rank_layer.kv_cache_base(),
        start_pos,
        seq_len,
        pool,
    )


def main():
    pass
