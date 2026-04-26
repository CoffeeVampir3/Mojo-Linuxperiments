from std.memory import Pointer
from threading.threading_traits import BurstThreadPool

from kernels.kernel_ops import PoolFence
from modeling.linear_borrow_pool import ScratchLease
from modeling.model_spec import F32, I8, Shape, ScratchView, StaticView
from modeling.minimax_m27_moe_butterquant_tp import (
    BodyRefs,
    C,
    FWHT_BLK,
    MOE_DOWN_NUM_BLK,
    LayerRefs,
    MiniMaxM27Topology,
    MiniMaxShapes,
)
from minimax.kernels.dispatch_kernels import minimax_moe_phase1
from minimax.kernels.router import TopKResult


@fieldwise_init
struct M27RankLayer[tp: Int](Copyable, ImplicitlyCopyable):
    var topo: MiniMaxM27Topology[Self.tp]
    var layer_idx: Int

    @always_inline
    def arena_base(self) -> Int:
        return self.topo.arena.base

    @always_inline
    def scratch_base(self) -> Int:
        return self.topo.arena.scratch_base()

    @always_inline
    def layer_base(self) -> Int:
        return self.topo.layers.base(self.topo.arena.base, self.layer_idx)

    @always_inline
    def layer(self) -> LayerRefs[Self.tp]:
        return self.topo.layers.proto

    @always_inline
    def gate_up_experts(self) -> M27GateUpExpertSlab[Self.tp]:
        var layer = self.layer()
        return M27GateUpExpertSlab[Self.tp](layer.body, self.layer_base())


@fieldwise_init
struct M27DenseMoeScratch[
    input_origin: MutOrigin,
    scale_origin: MutOrigin,
    qi_origin: MutOrigin,
    block_scale_origin: MutOrigin,
](Copyable, ImplicitlyCopyable):
    var scratch_base: Int
    var input_i8_lease: Pointer[ScratchLease, Self.input_origin]
    var input_scale_lease: Pointer[ScratchLease, Self.scale_origin]
    var expert_qi_lease: Pointer[ScratchLease, Self.qi_origin]
    var expert_block_scale_lease: Pointer[ScratchLease, Self.block_scale_origin]

    @always_inline
    def input_i8(self) -> ScratchView[I8, Shape[1, C.HIDDEN], Self.input_origin]:
        return self.input_i8_lease[].view[I8, Shape[1, C.HIDDEN]](
            self.scratch_base, 1)

    @always_inline
    def input_scale(self) -> ScratchView[F32, Shape[1, 1], Self.scale_origin]:
        return self.input_scale_lease[].view[F32, Shape[1, 1]](
            self.scratch_base, 1)

    @always_inline
    def expert_qi(self) -> ScratchView[
        I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE], Self.qi_origin,
    ]:
        return self.expert_qi_lease[].view[
            I8, Shape[C.TOP_K, C.MOE_INTERMEDIATE]](
            self.scratch_base, C.TOP_K)

    @always_inline
    def expert_block_scale(self) -> ScratchView[
        F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK], Self.block_scale_origin,
    ]:
        return self.expert_block_scale_lease[].view[
            F32, Shape[C.TOP_K, MOE_DOWN_NUM_BLK]](
            self.scratch_base, C.TOP_K)


@always_inline
def bind_dense_moe_scratch[
    input_origin: MutOrigin,
    scale_origin: MutOrigin,
    qi_origin: MutOrigin,
    block_scale_origin: MutOrigin,
](
    ref [input_origin] input_i8_lease: ScratchLease,
    ref [scale_origin] input_scale_lease: ScratchLease,
    ref [qi_origin] expert_qi_lease: ScratchLease,
    ref [block_scale_origin] expert_block_scale_lease: ScratchLease,
    scratch_base: Int,
) -> M27DenseMoeScratch[
    input_origin, scale_origin, qi_origin, block_scale_origin,
]:
    return M27DenseMoeScratch[
        input_origin, scale_origin, qi_origin, block_scale_origin,
    ](
        scratch_base,
        Pointer(to=input_i8_lease),
        Pointer(to=input_scale_lease),
        Pointer(to=expert_qi_lease),
        Pointer(to=expert_block_scale_lease),
    )


@fieldwise_init
struct M27GateUpExpertSlab[tp: Int](Copyable, ImplicitlyCopyable):
    comptime S = MiniMaxShapes[Self.tp]
    comptime weight_stride = C.MOE_INTERMEDIATE * C.HIDDEN
    comptime aux_stride = C.MOE_INTERMEDIATE * F32.ELEMENT_BYTES

    var body: BodyRefs[Self.tp]
    var layer_base: Int

    @always_inline
    def w1(self) -> StaticView[
        I8, Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN],
    ]:
        return self.body.experts_w1.bound(self.layer_base)

    @always_inline
    def w1_scale(self) -> StaticView[
        F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1],
    ]:
        return self.body.experts_w1_sc.bound(self.layer_base)

    @always_inline
    def w1_colsum(self) -> StaticView[
        F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1],
    ]:
        return self.body.experts_w1_colsum.bound(self.layer_base)

    @always_inline
    def w3(self) -> StaticView[
        I8, Shape[C.NUM_EXPERTS * C.MOE_INTERMEDIATE, C.HIDDEN],
    ]:
        return self.body.experts_w3.bound(self.layer_base)

    @always_inline
    def w3_scale(self) -> StaticView[
        F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1],
    ]:
        return self.body.experts_w3_sc.bound(self.layer_base)

    @always_inline
    def w3_colsum(self) -> StaticView[
        F32, Shape[Self.S.EXPERTS_LOCAL * C.MOE_INTERMEDIATE, 1],
    ]:
        return self.body.experts_w3_colsum.bound(self.layer_base)

    @always_inline
    def phase1[
        P: BurstThreadPool,
        pool_origin: MutOrigin,
        input_origin: MutOrigin,
        scale_origin: MutOrigin,
        qi_origin: MutOrigin,
        block_scale_origin: MutOrigin,
    ](
        self,
        scratch: M27DenseMoeScratch[
            input_origin, scale_origin, qi_origin, block_scale_origin,
        ],
        routing: TopKResult[C.TOP_K],
        rank: Int,
        ref [pool_origin] pool: P,
    ) -> PoolFence[P, pool_origin]:
        return minimax_moe_phase1[
            C.MOE_INTERMEDIATE, C.HIDDEN, FWHT_BLK,
            C.TOP_K, C.NUM_EXPERTS, Self.tp,
        ](
            scratch.input_i8(),
            scratch.input_scale(),
            routing,
            self.w1(),
            Self.weight_stride,
            self.w1_scale(),
            Self.aux_stride,
            self.w1_colsum(),
            Self.aux_stride,
            self.w3(),
            Self.weight_stride,
            self.w3_scale(),
            Self.aux_stride,
            self.w3_colsum(),
            Self.aux_stride,
            scratch.expert_qi(),
            scratch.expert_block_scale(),
            rank,
            pool,
        )


@always_inline
def dense_moe_phase1_probe[
    tp: Int,
    P: BurstThreadPool,
    pool_origin: MutOrigin,
    input_origin: MutOrigin,
    scale_origin: MutOrigin,
    qi_origin: MutOrigin,
    block_scale_origin: MutOrigin,
](
    topo: MiniMaxM27Topology[tp],
    layer_idx: Int,
    ref [input_origin] input_i8_lease: ScratchLease,
    ref [scale_origin] input_scale_lease: ScratchLease,
    ref [qi_origin] expert_qi_lease: ScratchLease,
    ref [block_scale_origin] expert_block_scale_lease: ScratchLease,
    routing: TopKResult[C.TOP_K],
    rank_id: Int,
    ref [pool_origin] pool: P,
) -> PoolFence[P, pool_origin]:
    var rank_layer = M27RankLayer[tp](topo, layer_idx)
    var scratch = bind_dense_moe_scratch(
        input_i8_lease,
        input_scale_lease,
        expert_qi_lease,
        expert_block_scale_lease,
        rank_layer.scratch_base(),
    )
    return rank_layer.gate_up_experts().phase1(scratch, routing, rank_id, pool)


def main():
    pass
